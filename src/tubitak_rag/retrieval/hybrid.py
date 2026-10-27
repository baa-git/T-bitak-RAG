"""Hybrid retriever: dense and BM25 candidates fused with Reciprocal Rank Fusion (FR-R4)."""

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.ports import Retriever
from tubitak_rag.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """:class:`~tubitak_rag.ports.Retriever` that fuses a dense and a lexical retriever.

    Each side returns ``k * candidate_multiplier`` candidates; a chunk found by only one
    side still takes part in the fusion. The primary score is the RRF score; the component
    scores (``dense``, ``bm25``) are kept so guardrails can use a calibrated score.
    """

    name = "hybrid"

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        rrf_k: int = 60,
        candidate_multiplier: int = 2,
    ) -> None:
        if rrf_k <= 0 or candidate_multiplier <= 0:
            raise ValueError("rrf_k ve candidate_multiplier pozitif olmalıdır")
        self._dense = dense
        self._lexical = lexical
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        """Return up to ``k`` fused chunks, best first, with ``scores["rrf"]`` set."""
        n_candidates = k * self._candidate_multiplier
        sides = [
            retriever.retrieve(query, n_candidates, filters)
            for retriever in (self._dense, self._lexical)
        ]
        chunks: dict[str, Chunk] = {}
        components: dict[str, dict[str, float]] = {}
        for results in sides:
            for result in results:
                chunk_id = result.chunk.chunk_id
                chunks.setdefault(chunk_id, result.chunk)
                components.setdefault(chunk_id, {}).update(result.scores)
        fused = reciprocal_rank_fusion(
            [[result.chunk.chunk_id for result in results] for results in sides], self._rrf_k
        )
        return [
            ScoredChunk(
                chunk=chunks[chunk_id], score=rrf, scores={**components[chunk_id], "rrf": rrf}
            )
            for chunk_id, rrf in fused[:k]
        ]
