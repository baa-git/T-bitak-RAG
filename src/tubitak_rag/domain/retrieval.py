"""Retrieval models: metadata filters and scored search results."""

from tubitak_rag.domain.base import DomainModel
from tubitak_rag.domain.chunk import Chunk


class MetadataFilter(DomainModel):
    """Optional equality filters applied during search (FR-R5)."""

    program_kodu: str | None = None
    doc_turu: str | None = None


class ScoredChunk(DomainModel):
    """A chunk with the primary ranking score and every component score that produced it."""

    chunk: Chunk
    score: float
    scores: dict[str, float]  # {"dense": cos, "bm25": s, "rrf": s, "rerank": logit}
