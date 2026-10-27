"""Query normalisation and the in-memory metadata filter (FR-R5).

The dense side filters inside the vector store (Qdrant payload filter); keyword search has
no payload, so BM25 results are filtered here after their chunks are fetched.
"""

from collections.abc import Sequence

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.ingestion.normalize import collapse_ws, normalize_text


def normalize_query(q: str) -> str:
    """Normalise a question like the indexed text (NFC, no invisible chars, single spaces)."""
    return collapse_ws(normalize_text(q))


def _matches(chunk: Chunk, filters: MetadataFilter | None) -> bool:
    """Return whether ``chunk`` satisfies every field set in ``filters``."""
    if filters is None:
        return True
    if filters.program_kodu is not None and chunk.program_kodu != filters.program_kodu:
        return False
    return filters.doc_turu is None or chunk.doc_turu == filters.doc_turu


def apply_filters(
    results: Sequence[ScoredChunk], filters: MetadataFilter | None
) -> list[ScoredChunk]:
    """Keep the results whose chunk matches ``filters``, preserving their order."""
    return [result for result in results if _matches(result.chunk, filters)]
