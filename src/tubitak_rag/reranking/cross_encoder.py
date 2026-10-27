"""Core reranking step: validated call of a :class:`~tubitak_rag.ports.Reranker` (FR-RR1/2)."""

from collections.abc import Sequence

from tubitak_rag.domain.retrieval import ScoredChunk
from tubitak_rag.ports import Reranker


def rerank_with(
    reranker: Reranker, query: str, candidates: Sequence[ScoredChunk], top_n: int
) -> list[ScoredChunk]:
    """Rerank ``candidates`` and return at most ``top_n`` results, best first.

    Every result must carry ``scores["rerank"]`` so guardrails can use it (FR-RR2). An empty
    candidate list returns ``[]`` without calling the model.
    """
    if top_n <= 0:
        raise ValueError(f"top_n pozitif olmalıdır, verilen: {top_n}")
    if not candidates:
        return []
    results = reranker.rerank(query, candidates, top_n)
    if len(results) > top_n:
        raise ValueError(f"{reranker.model_name} top_n={top_n} yerine {len(results)} sonuç döndü")
    if any("rerank" not in result.scores for result in results):
        raise ValueError(f"{reranker.model_name} sonuçlarında scores['rerank'] eksik")
    return results
