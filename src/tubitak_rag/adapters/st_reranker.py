"""Reranker backed by a sentence-transformers ``CrossEncoder`` (BAAI/bge-reranker-v2-m3)."""

from collections.abc import Sequence
from typing import Any

from tubitak_rag.adapters.st_embedder import resolve_device
from tubitak_rag.domain.retrieval import ScoredChunk

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


def _raw_logits(scores: Any) -> Any:
    """Identity activation: keep the model's logits (guardrail thresholds are logits)."""
    return scores


class CrossEncoderReranker:
    """:class:`~tubitak_rag.ports.Reranker` scoring ``(query, chunk.embed_text)`` pairs.

    The passage is ``embed_text`` (context header + text): the header carries the document
    title and programme code, which the bare ``text`` of many articles lacks. Pairs longer
    than ``max_length`` tokens are truncated from the end by the tokenizer.
    """

    def __init__(
        self,
        model: str = DEFAULT_RERANKER,
        *,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model
        self._batch_size = batch_size
        # Without an explicit activation, a 1-label model's logits go through a sigmoid.
        self._model = CrossEncoder(
            model,
            device=resolve_device(device),
            max_length=max_length,
            activation_fn=_raw_logits,
        )

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Return the ``top_n`` best candidates; ``score`` and ``scores["rerank"]`` = logit."""
        if not candidates:
            return []
        logits = self._model.predict(
            [(query, candidate.chunk.embed_text) for candidate in candidates],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        rescored = [
            candidate.model_copy(
                update={
                    "score": float(logit),
                    "scores": {**candidate.scores, "rerank": float(logit)},
                }
            )
            for candidate, logit in zip(candidates, logits, strict=True)
        ]
        rescored.sort(key=lambda result: result.score, reverse=True)
        return rescored[:top_n]
