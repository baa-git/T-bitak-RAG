"""Retriever port: finds candidate chunks for a question."""

from typing import Protocol, runtime_checkable

from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk


@runtime_checkable
class Retriever(Protocol):
    """A retrieval strategy (dense, BM25 or hybrid) identified by ``name``."""

    name: str

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        """Return up to ``k`` scored chunks, best first."""
        ...
