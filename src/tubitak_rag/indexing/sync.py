"""Incremental index synchronisation: compare document fingerprints, touch only what changed.

The plan is a pure set difference between the corpus (``doc_id -> fingerprint`` of the
current chunks, see :mod:`tubitak_rag.indexing.fingerprint`) and the index
(``VectorStore.indexed_docs``, the fingerprint of the stored chunks):

* in corpus only                     -> ``add``       (FR-I2)
* in both, different fingerprint     -> ``update``    (FR-I3: delete old chunks, insert new)
* in index only                      -> ``remove``    (FR-I4)
* in both, same fingerprint          -> ``unchanged`` (nothing is embedded again)

Fingerprints are built from chunk contents, not from the raw file hash, so a parser or
chunker change re-indexes the affected documents even when the files did not change.
"""

from collections.abc import Callable, Mapping, Sequence

from tubitak_rag.domain.base import DomainModel
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.ports import Embedder, VectorStore

Progress = Callable[[str, int, int], None]


class SyncPlan(DomainModel):
    """Document ids per action, each list sorted."""

    add: list[str]
    update: list[str]
    remove: list[str]
    unchanged: list[str]


def plan_sync(manifest: Mapping[str, str], indexed: Mapping[str, str]) -> SyncPlan:
    """Compare ``doc_id -> fingerprint`` maps of the corpus (``manifest``) and the index."""
    common = manifest.keys() & indexed.keys()
    return SyncPlan(
        add=sorted(manifest.keys() - indexed.keys()),
        update=sorted(d for d in common if manifest[d] != indexed[d]),
        remove=sorted(indexed.keys() - manifest.keys()),
        unchanged=sorted(d for d in common if manifest[d] == indexed[d]),
    )


def apply_sync(
    plan: SyncPlan,
    chunks_by_doc: Mapping[str, Sequence[Chunk]],
    embedder: Embedder,
    store: VectorStore,
    collection: str,
    *,
    batch_size: int = 64,
    progress: Progress | None = None,
) -> dict[str, int]:
    """Apply ``plan`` to ``collection``; only added/updated documents are embedded.

    Each document is embedded first (``embed_text``, at most ``batch_size`` texts per
    call) and written afterwards, so a failure while embedding never leaves a document
    half-indexed under its new hash. ``progress("embed", done_chunks, total_chunks)`` is
    called after each document.

    Returns:
        ``{"added_chunks", "updated_chunks", "removed_docs"}``.
    """
    store.ensure_collection(collection, embedder.dimension)
    for doc_id in plan.remove:
        store.delete_doc(collection, doc_id)

    to_index = [*plan.add, *plan.update]
    total = sum(len(chunks_by_doc[doc_id]) for doc_id in to_index)
    done = 0
    for doc_id in to_index:
        chunks = chunks_by_doc[doc_id]
        texts = [chunk.embed_text for chunk in chunks]
        vectors = [
            vector
            for start in range(0, len(texts), batch_size)
            for vector in embedder.embed_documents(texts[start : start + batch_size])
        ]
        store.delete_doc(collection, doc_id)  # no-op for new documents
        store.upsert(collection, chunks, vectors)
        done += len(chunks)
        if progress is not None:
            progress("embed", done, total)

    return {
        "added_chunks": sum(len(chunks_by_doc[d]) for d in plan.add),
        "updated_chunks": sum(len(chunks_by_doc[d]) for d in plan.update),
        "removed_docs": len(plan.remove),
    }
