"""Tests for the incremental sync plan and its application (FR-I2, FR-I3, FR-I4)."""

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from tests.builders import make_chunk
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.vector_store import InMemoryVectorStore
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.indexing.fingerprint import fingerprints
from tubitak_rag.indexing.sync import SyncPlan, apply_sync, plan_sync

COLLECTION = "structural_ctx__fake"


class _CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that counts every embedded document text."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts += texts
        return super().embed_documents(texts)


def _chunks(doc_id: str, sha: str, n: int, version: str = "") -> list[Chunk]:
    return [
        make_chunk(
            f"{doc_id}:{i:04d}:00",
            doc_id=doc_id,
            doc_sha256=sha,
            sira=i,
            text=f"{doc_id} {version} parça {i}",
            embed_text=f"{doc_id} {version} parça {i}",
        )
        for i in range(n)
    ]


def _indexed(store: InMemoryVectorStore) -> dict[str, list[str]]:
    by_doc: dict[str, list[str]] = {}
    for chunk, _ in store.collections[COLLECTION].values():
        by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)
    return {doc: sorted(ids) for doc, ids in by_doc.items()}


# --- plan_sync ------------------------------------------------------------------------------


def test_plan_new_doc_is_add() -> None:
    assert plan_sync({"a": "1"}, {}) == SyncPlan(add=["a"], update=[], remove=[], unchanged=[])


def test_plan_changed_sha_is_update() -> None:
    assert plan_sync({"a": "2"}, {"a": "1"}).update == ["a"]


def test_plan_missing_in_manifest_is_remove() -> None:
    assert plan_sync({}, {"a": "1"}).remove == ["a"]


def test_plan_same_sha_unchanged() -> None:
    assert plan_sync({"a": "1"}, {"a": "1"}) == SyncPlan(
        add=[], update=[], remove=[], unchanged=["a"]
    )


def test_plan_lists_are_sorted() -> None:
    manifest = {"c": "1", "a": "1", "b": "2", "d": "1"}
    indexed = {"b": "1", "d": "1", "z": "1", "y": "1"}
    plan = plan_sync(manifest, indexed)
    assert (plan.add, plan.update, plan.remove, plan.unchanged) == (
        ["a", "c"],
        ["b"],
        ["y", "z"],
        ["d"],
    )


def test_sync_plan_is_immutable() -> None:
    plan = SyncPlan(add=[], update=[], remove=[], unchanged=[])
    with pytest.raises(ValidationError):
        plan.add = ["x"]  # type: ignore[misc]


# --- apply_sync -----------------------------------------------------------------------------


def test_apply_ensures_collection_exists() -> None:
    store = InMemoryVectorStore()
    apply_sync(plan_sync({}, {}), {}, FakeEmbedder(), store, COLLECTION)
    assert store.count(COLLECTION) == 0


def _sync(
    chunks_by_doc: dict[str, list[Chunk]],
    store: InMemoryVectorStore,
    embedder: FakeEmbedder | None = None,
) -> dict[str, int]:
    """Plan against what the store holds now and apply, as an ingest would."""
    plan = plan_sync(fingerprints(chunks_by_doc), store.indexed_docs(COLLECTION))
    return apply_sync(plan, chunks_by_doc, embedder or FakeEmbedder(), store, COLLECTION)


def test_apply_only_embeds_changed_docs() -> None:
    store, embedder = InMemoryVectorStore(), _CountingEmbedder()
    old = {"a": _chunks("a", "1", 2), "b": _chunks("b", "1", 3)}
    _sync(old, store, embedder)
    embedder.texts.clear()

    chunks_by_doc = {**old, "c": _chunks("c", "1", 4)}
    stats = _sync(chunks_by_doc, store, embedder)

    assert embedder.texts == [c.embed_text for c in chunks_by_doc["c"]]
    assert stats == {"added_chunks": 4, "updated_chunks": 0, "removed_docs": 0}
    assert store.count(COLLECTION) == 9


def test_second_sync_is_a_noop() -> None:
    store, embedder = InMemoryVectorStore(), _CountingEmbedder()
    docs = {"a": _chunks("a", "1", 2)}
    _sync(docs, store, embedder)
    assert _sync(docs, store, embedder) == {
        "added_chunks": 0,
        "updated_chunks": 0,
        "removed_docs": 0,
    }
    assert embedder.calls == 1


def test_apply_update_replaces_chunks() -> None:
    store = InMemoryVectorStore()
    _sync({"a": _chunks("a", "1", 3)}, store)

    new_version = _chunks("a", "2", 2, version="v2")
    stats = _sync({"a": new_version}, store)

    assert stats == {"added_chunks": 0, "updated_chunks": 2, "removed_docs": 0}
    assert _indexed(store) == {"a": sorted(c.chunk_id for c in new_version)}
    assert store.indexed_docs(COLLECTION) == fingerprints({"a": new_version})
    stored = store.get_chunks(COLLECTION, [new_version[0].chunk_id])
    assert stored[0].text == new_version[0].text


def test_same_raw_sha_with_changed_chunks_is_reindexed() -> None:
    # A parser or chunker fix changes the chunks while the file (doc_sha256) stays the same.
    store = InMemoryVectorStore()
    _sync({"a": _chunks("a", "1", 2)}, store)
    rechunked = _chunks("a", "1", 2, version="düzeltilmiş")
    assert rechunked[0].doc_sha256 == "1"

    stats = _sync({"a": rechunked}, store)

    assert stats["updated_chunks"] == 2
    assert [c.text for c in store.get_chunks(COLLECTION, [rechunked[0].chunk_id])] == [
        rechunked[0].text
    ]


def test_half_written_document_is_reindexed() -> None:
    store = InMemoryVectorStore()
    docs = {"a": _chunks("a", "1", 3)}
    _sync(docs, store)
    store.collections[COLLECTION].pop(docs["a"][2].chunk_id)  # e.g. a crash mid-upsert

    assert _sync(docs, store)["updated_chunks"] == 3
    assert store.count(COLLECTION) == 3


def test_apply_remove_deletes() -> None:
    store = InMemoryVectorStore()
    docs = {"a": _chunks("a", "1", 2), "b": _chunks("b", "1", 1)}
    _sync(docs, store)

    stats = _sync({"b": docs["b"]}, store)

    assert stats == {"added_chunks": 0, "updated_chunks": 0, "removed_docs": 1}
    assert set(store.indexed_docs(COLLECTION)) == {"b"}
    hits = store.search(COLLECTION, FakeEmbedder().embed_query(docs["a"][0].embed_text), k=10)
    assert all(hit.chunk.doc_id != "a" for hit in hits)


def test_apply_embeds_in_batches_and_reports_progress() -> None:
    store, embedder = InMemoryVectorStore(), _CountingEmbedder()
    events: list[tuple[str, int, int]] = []
    docs = {"a": _chunks("a", "1", 5), "b": _chunks("b", "1", 2)}
    apply_sync(
        plan_sync({"a": "1", "b": "1"}, {}),
        docs,
        embedder,
        store,
        COLLECTION,
        batch_size=2,
        progress=lambda stage, done, total: events.append((stage, done, total)),
    )
    assert embedder.calls == 4  # a: 2+2+1, b: 2
    assert events == [("embed", 5, 7), ("embed", 7, 7)]


def test_apply_embeds_embed_text_not_text() -> None:
    store, embedder = InMemoryVectorStore(), _CountingEmbedder()
    chunk = make_chunk("a:0001:00", doc_id="a", text="gövde", embed_text="Başlık\n\ngövde")
    apply_sync(plan_sync({"a": chunk.doc_sha256}, {}), {"a": [chunk]}, embedder, store, COLLECTION)
    assert embedder.texts == ["Başlık\n\ngövde"]
