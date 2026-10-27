"""Reranker port: re-orders retrieval candidates with a stronger model."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from tubitak_rag.domain.retrieval import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, passage) pairs and keeps the best ``top_n``."""

    model_name: str

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Return the ``top_n`` best candidates with ``scores["rerank"]`` filled in."""
        ...
