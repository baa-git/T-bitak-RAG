"""Tests for the Qdrant vector store against a fake ``qdrant_client`` module."""

import sys
import types
import uuid
from collections.abc import Callable
from typing import Any

import pytest

from tests.builders import make_chunk
from tubitak_rag.adapters.qdrant_store import QdrantVectorStore, filter_from, point_id
from tubitak_rag.domain.retrieval import MetadataFilter
from tubitak_rag.indexing.fingerprint import chunk_digest, doc_fingerprint
from tubitak_rag.ports import VectorStore


def _factory(kind: str) -> Callable[..., dict[str, Any]]:
    def make(**kwargs: Any) -> dict[str, Any]:
        return {"_type": kind, **kwargs}

    return make


class _FakeClient:
    """Records every call; return values are configured per test through attributes."""

    last: "_FakeClient"

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.exists = False
        self.points: list[Any] = []
        self.scroll_pages: list[tuple[list[Any], Any]] = []
        _FakeClient.last = self

    def _record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((name, kwargs))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for call, kwargs in self.calls if call == name]

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs: Any) -> None:
        self._record("create_collection", kwargs)

    def create_payload_index(self, **kwargs: Any) -> None:
        self._record("create_payload_index", kwargs)

    def upsert(self, **kwargs: Any) -> None:
        self._record("upsert", kwargs)

    def delete(self, **kwargs: Any) -> None:
        self._record("delete", kwargs)

    def delete_collection(self, **kwargs: Any) -> None:
        self._record("delete_collection", kwargs)

    def query_points(self, **kwargs: Any) -> Any:
        self._record("query_points", kwargs)
        return types.SimpleNamespace(points=self.points)

    def scroll(self, **kwargs: Any) -> tuple[list[Any], Any]:
        self._record("scroll", kwargs)
        return self.scroll_pages.pop(0)

    def retrieve(self, **kwargs: Any) -> list[Any]:
        self._record("retrieve", kwargs)
        return self.points

    def count(self, **kwargs: Any) -> Any:
        self._record("count", kwargs)
        return types.SimpleNamespace(count=len(self.points))


@pytest.fixture(autouse=True)
def fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    models = types.ModuleType("qdrant_client.models")
    for kind in [
        "VectorParams",
        "PointStruct",
        "Filter",
        "FieldCondition",
        "MatchValue",
        "FilterSelector",
    ]:
        setattr(models, kind, _factory(kind))
    models.Distance = types.SimpleNamespace(COSINE="Cosine")  # type: ignore[attr-defined]
    models.PayloadSchemaType = types.SimpleNamespace(KEYWORD="keyword")  # type: ignore[attr-defined]
    package = types.ModuleType("qdrant_client")
    package.QdrantClient = _FakeClient  # type: ignore[attr-defined]
    package.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", package)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)


def _point(chunk_id: str, score: float = 0.0) -> Any:
    chunk = make_chunk(chunk_id)
    payload = {**chunk.model_dump(mode="json"), "chunk_digest": chunk_digest(chunk)}
    return types.SimpleNamespace(id=point_id(chunk_id), score=score, payload=payload)


def test_is_a_vector_store() -> None:
    assert isinstance(QdrantVectorStore("http://q:6333"), VectorStore)


def test_client_gets_url_and_timeout() -> None:
    QdrantVectorStore("http://q:6333", timeout_s=5)
    assert _FakeClient.last.init_kwargs == {"url": "http://q:6333", "timeout": 5}


def test_point_id_is_uuid5_of_chunk_id() -> None:
    assert point_id("d:0001:00") == str(uuid.uuid5(uuid.NAMESPACE_URL, "d:0001:00"))


def test_ensure_collection_creates_when_missing() -> None:
    store = QdrantVectorStore("http://q")
    store.ensure_collection("c", 768)
    client = _FakeClient.last
    assert client.named("create_collection") == [
        {
            "collection_name": "c",
            "vectors_config": {"_type": "VectorParams", "size": 768, "distance": "Cosine"},
        }
    ]
    indexed = [(k["field_name"], k["field_schema"]) for k in client.named("create_payload_index")]
    assert indexed == [("doc_id", "keyword"), ("program_kodu", "keyword"), ("doc_turu", "keyword")]


def test_ensure_collection_keeps_existing() -> None:
    store = QdrantVectorStore("http://q")
    _FakeClient.last.exists = True
    store.ensure_collection("c", 768)
    assert _FakeClient.last.calls == []


def test_upsert_uses_uuid5_ids_and_payload() -> None:
    store = QdrantVectorStore("http://q")
    chunk = make_chunk("mevzuat-031:0001:00")
    store.upsert("c", [chunk], [[0.5, 0.5]])
    (call,) = _FakeClient.last.named("upsert")
    assert call["collection_name"] == "c" and call["wait"] is True
    (point,) = call["points"]
    assert point == {
        "_type": "PointStruct",
        "id": point_id("mevzuat-031:0001:00"),
        "vector": [0.5, 0.5],
        "payload": {**chunk.model_dump(mode="json"), "chunk_digest": chunk_digest(chunk)},
    }


def test_upsert_batches_by_128() -> None:
    store = QdrantVectorStore("http://q")
    chunks = [make_chunk(f"d:{i:04d}:00") for i in range(130)]
    store.upsert("c", chunks, [[1.0]] * 130)
    assert [len(call["points"]) for call in _FakeClient.last.named("upsert")] == [128, 2]


def test_upsert_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="aynı uzunlukta"):
        QdrantVectorStore("http://q").upsert("c", [make_chunk()], [])


def test_delete_doc_filters_by_doc_id() -> None:
    store = QdrantVectorStore("http://q")
    store.delete_doc("c", "mevzuat-031")
    (call,) = _FakeClient.last.named("delete")
    condition = call["points_selector"]["filter"]["must"][0]
    assert call["collection_name"] == "c" and call["wait"] is True
    assert condition["key"] == "doc_id" and condition["match"]["value"] == "mevzuat-031"


def test_search_maps_points_to_scored_chunks() -> None:
    store = QdrantVectorStore("http://q")
    _FakeClient.last.points = [_point("a:0001:00", 0.9), _point("b:0001:00", 0.4)]
    results = store.search("c", [0.1, 0.2], k=2)
    (call,) = _FakeClient.last.named("query_points")
    assert call == {
        "collection_name": "c",
        "query": [0.1, 0.2],
        "query_filter": None,
        "limit": 2,
        "with_payload": True,
    }
    assert [r.chunk.chunk_id for r in results] == ["a:0001:00", "b:0001:00"]
    assert results[0].score == 0.9 and results[0].scores == {"dense": 0.9}
    assert results[0].chunk == make_chunk("a:0001:00")  # digest key is not part of the chunk


def test_search_passes_metadata_filter() -> None:
    store = QdrantVectorStore("http://q")
    store.search("c", [0.1], k=1, filters=MetadataFilter(program_kodu="1507"))
    (call,) = _FakeClient.last.named("query_points")
    assert call["query_filter"]["must"][0]["key"] == "program_kodu"


def test_filter_from_program_and_type() -> None:
    result = filter_from(MetadataFilter(program_kodu="1507", doc_turu="kilavuz"))
    assert result == {
        "_type": "Filter",
        "must": [
            {
                "_type": "FieldCondition",
                "key": "program_kodu",
                "match": {"_type": "MatchValue", "value": "1507"},
            },
            {
                "_type": "FieldCondition",
                "key": "doc_turu",
                "match": {"_type": "MatchValue", "value": "kilavuz"},
            },
        ],
    }


def test_filter_from_none_or_empty_is_none() -> None:
    assert filter_from(None) is None
    assert filter_from(MetadataFilter()) is None


def _digest_point(doc_id: str, digest: str) -> Any:
    return types.SimpleNamespace(payload={"doc_id": doc_id, "chunk_digest": digest})


def test_indexed_docs_paginates_scroll_and_fingerprints_docs() -> None:
    store = QdrantVectorStore("http://q")
    client = _FakeClient.last
    client.exists = True
    page1 = [_digest_point("a", "d1"), _digest_point("b", "d3")]
    page2 = [_digest_point("a", "d2")]
    client.scroll_pages = [(page1, "next"), (page2, None)]
    assert store.indexed_docs("c") == {
        "a": doc_fingerprint(["d2", "d1"]),
        "b": doc_fingerprint(["d3"]),
    }
    first, second = client.named("scroll")
    assert first == {
        "collection_name": "c",
        "limit": 1000,
        "offset": None,
        "with_payload": ["doc_id", "chunk_digest"],
        "with_vectors": False,
    }
    assert second["offset"] == "next"


def test_indexed_docs_point_without_digest_never_matches() -> None:
    store = QdrantVectorStore("http://q")
    client = _FakeClient.last
    client.exists = True
    client.scroll_pages = [([types.SimpleNamespace(payload={"doc_id": "a"})], None)]
    chunk = make_chunk("a:0001:00", doc_id="a")
    assert store.indexed_docs("c")["a"] != doc_fingerprint([chunk_digest(chunk)])


def test_indexed_docs_of_missing_collection_is_empty() -> None:
    assert QdrantVectorStore("http://q").indexed_docs("c") == {}
    assert _FakeClient.last.named("scroll") == []


def test_get_chunks_keeps_requested_order_and_skips_unknown() -> None:
    store = QdrantVectorStore("http://q")
    _FakeClient.last.points = [_point("a:0001:00"), _point("b:0001:00")]
    chunks = store.get_chunks("c", ["b:0001:00", "x:0001:00", "a:0001:00"])
    (call,) = _FakeClient.last.named("retrieve")
    assert call["ids"] == [point_id(c) for c in ["b:0001:00", "x:0001:00", "a:0001:00"]]
    assert call["with_payload"] is True
    assert [c.chunk_id for c in chunks] == ["b:0001:00", "a:0001:00"]
    assert chunks[1] == make_chunk("a:0001:00")


def test_count_is_exact() -> None:
    store = QdrantVectorStore("http://q")
    _FakeClient.last.exists = True
    _FakeClient.last.points = [_point("a:0001:00")]
    assert store.count("c") == 1
    assert _FakeClient.last.named("count") == [{"collection_name": "c", "exact": True}]


def test_drop_collection_deletes_existing_collection() -> None:
    store = QdrantVectorStore("http://q")
    _FakeClient.last.exists = True
    store.drop_collection("c")
    assert _FakeClient.last.named("delete_collection") == [{"collection_name": "c"}]


def test_drop_collection_ignores_missing_collection() -> None:
    store = QdrantVectorStore("http://q")
    store.drop_collection("c")
    assert _FakeClient.last.named("delete_collection") == []


def test_count_of_missing_collection_is_zero() -> None:
    store = QdrantVectorStore("http://q")
    assert store.count("c") == 0
    assert _FakeClient.last.named("count") == []


def test_doc_chunk_counts_scrolls_doc_ids_only() -> None:
    store = QdrantVectorStore("http://q")
    client = _FakeClient.last
    client.exists = True
    page1 = [_digest_point("a", "d1"), _digest_point("b", "d3")]
    client.scroll_pages = [(page1, "next"), ([_digest_point("a", "d2")], None)]
    assert store.doc_chunk_counts("c") == {"a": 2, "b": 1}
    first, second = client.named("scroll")
    assert first["with_payload"] == ["doc_id"] and first["with_vectors"] is False
    assert second["offset"] == "next"


def test_doc_chunk_counts_for_given_docs_uses_filtered_exact_counts() -> None:
    store = QdrantVectorStore("http://q")
    client = _FakeClient.last
    client.exists = True
    client.points = [_point("a:0001:00"), _point("a:0002:00")]
    assert store.doc_chunk_counts("c", ["a"]) == {"a": 2}
    (call,) = client.named("count")
    assert call["exact"] is True
    assert call["count_filter"]["must"][0]["key"] == "doc_id"
    assert client.named("scroll") == []


def test_doc_chunk_counts_of_missing_collection_is_empty() -> None:
    assert QdrantVectorStore("http://q").doc_chunk_counts("c", ["a"]) == {}
    assert _FakeClient.last.named("count") == []


def test_export_points_scrolls_payloads_and_vectors() -> None:
    store = QdrantVectorStore("http://q")
    client = _FakeClient.last
    client.exists = True
    first, second = _point("a:0001:00"), _point("b:0001:00")
    first.vector, second.vector = [1, 0], [0.5, 0.5]
    client.scroll_pages = [([first], "next"), ([second], None)]
    chunks, vectors = store.export_points("c")
    assert [c.chunk_id for c in chunks] == ["a:0001:00", "b:0001:00"]
    assert chunks[0] == make_chunk("a:0001:00")  # chunk_digest is not part of the chunk
    assert vectors == [[1.0, 0.0], [0.5, 0.5]]
    calls = client.named("scroll")
    assert calls[0]["with_vectors"] is True and calls[0]["with_payload"] is True
    assert calls[1]["offset"] == "next"


def test_export_points_of_missing_collection_is_empty() -> None:
    assert QdrantVectorStore("http://q").export_points("c") == ([], [])
