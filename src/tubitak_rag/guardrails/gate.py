"""Retrieval score gate: no LLM call when the best retrieval score is too low (FR-GR1, FR-GR9).

Thresholds come from config (``guardrails.thresholds``), one per score key, because scores of
different stages are not comparable: a reranker logit, a dense cosine and a BM25 score live on
different scales, and RRF scores are not comparable across runs at all.
"""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from tubitak_rag.domain.retrieval import ScoredChunk

# The most informative score that is present decides.
SCORE_KEYS = ("rerank", "dense", "bm25")


class GateResult(BaseModel):
    """Gate decision with the score key and the best score it was based on."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    key: str | None
    score: float | None


class RetrievalGate:
    """Passes when the best score of the chosen key reaches that key's threshold."""

    def __init__(self, thresholds: Mapping[str, float]) -> None:
        missing = sorted(set(SCORE_KEYS) - set(thresholds))
        if missing:
            raise ValueError(
                f"guardrails.thresholds içinde eksik eşik anahtarı: {', '.join(missing)} "
                f"(gerekli: {', '.join(SCORE_KEYS)})"
            )
        self.thresholds = dict(thresholds)

    def evaluate(self, results: Sequence[ScoredChunk]) -> GateResult:
        """Gate ``results``; an empty list (or no known score) never passes."""
        for key in SCORE_KEYS:
            scores = [r.scores[key] for r in results if key in r.scores]
            if scores:
                best = max(scores)
                return GateResult(passed=best >= self.thresholds[key], key=key, score=best)
        return GateResult(passed=False, key=None, score=None)
