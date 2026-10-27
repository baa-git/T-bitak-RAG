"""Vector store port: persists chunk vectors and answers nearest-neighbour queries."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    """Collection-based storage of chunks and their vectors."""

    def ensure_collection(self, name: str, dimension: int) -> None:
        """Create the collection if it does not exist yet."""
        ...

    def drop_collection(self, name: str) -> None:
        """Delete the collection with all its chunks; a missing collection is ignored."""
        ...

    def upsert(
        self, name: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Insert or replace chunks with their vectors (same order and length)."""
        ...

    def delete_doc(self, name: str, doc_id: str) -> None:
        """Remove every chunk that belongs to ``doc_id``."""
        ...

    def search(
        self,
        name: str,
        vector: Sequence[float],
        k: int,
        filters: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        """Return the ``k`` most similar chunks, best first, honouring ``filters``."""
        ...

    def indexed_docs(self, name: str) -> dict[str, str]:
        """Map each indexed ``doc_id`` to the fingerprint of its stored chunks.

        The fingerprint is :func:`tubitak_rag.indexing.fingerprint.doc_fingerprint` over the
        :func:`~tubitak_rag.indexing.fingerprint.chunk_digest` of every stored chunk of the
        document, so it changes whenever the chunks change, even if the raw file did not.
        Returns ``{}`` when the collection does not exist yet.
        """
        ...

    def get_chunks(self, name: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id in the requested order, skipping unknown ids."""
        ...

    def count(self, name: str) -> int:
        """Return the number of chunks in the collection (0 if it does not exist)."""
        ...
