"""Qdrant adapter against a real Qdrant server (``QDRANT_URL``, e.g. http://localhost:6333)."""

import uuid
from collections.abc import Iterator

import pytest

from tests.builders import make_chunk
from tubitak_rag.adapters.qdrant_store import QdrantVectorStore
from tubitak_rag.domain.retrieval import MetadataFilter
from tubitak_rag.indexing.fingerprint import fingerprints

pytestmark = pytest.mark.integration


@pytest.fixture
def collection(qdrant_url: str) -> Iterator[str]:
    from qdrant_client import QdrantClient

    name = f"test__{uuid.uuid4().hex[:8]}"
    yield name
    QdrantClient(url=qdrant_url).delete_collection(collection_name=name)


def test_upsert_search_filter_and_delete_by_doc(qdrant_url: str, collection: str) -> None:
    store = QdrantVectorStore(qdrant_url)
    chunks = [
        make_chunk("a:0001:00", doc_id="a", program_kodu="1507", doc_sha256="1" * 64),
        make_chunk("a:0002:00", doc_id="a", program_kodu="1507", doc_sha256="1" * 64, sira=2),
        make_chunk("b:0001:00", doc_id="b", program_kodu="1501", doc_sha256="2" * 64),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]

    store.ensure_collection(collection, 3)
    store.ensure_collection(collection, 3)  # idempotent
    store.upsert(collection, chunks, vectors)
    store.upsert(collection, chunks[:1], vectors[:1])  # same id -> replaced, not duplicated
    assert store.count(collection) == 3

    results = store.search(collection, [1.0, 0.0, 0.0], k=2)
    assert [r.chunk.chunk_id for r in results] == ["a:0001:00", "a:0002:00"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].chunk == chunks[0]

    only_1501 = store.search(
        collection, [1.0, 0.0, 0.0], k=3, filters=MetadataFilter(program_kodu="1501")
    )
    assert [r.chunk.chunk_id for r in only_1501] == ["b:0001:00"]

    assert store.indexed_docs(collection) == fingerprints({"a": chunks[:2], "b": chunks[2:]})
    assert [c.chunk_id for c in store.get_chunks(collection, ["b:0001:00", "a:0001:00"])] == [
        "b:0001:00",
        "a:0001:00",
    ]

    store.delete_doc(collection, "a")
    assert store.count(collection) == 1
    assert store.indexed_docs(collection) == fingerprints({"b": chunks[2:]})
    assert [r.chunk.doc_id for r in store.search(collection, [1.0, 0.0, 0.0], k=3)] == ["b"]


def test_indexed_docs_of_missing_collection_is_empty(qdrant_url: str) -> None:
    assert QdrantVectorStore(qdrant_url).indexed_docs(f"yok__{uuid.uuid4().hex[:8]}") == {}
