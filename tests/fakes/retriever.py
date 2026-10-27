"""Scripted retriever for pipeline tests: returns fixed results and records every call."""

from collections.abc import Sequence
from typing import Any

from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk


class FakeRetriever:
    """Returns the first ``k`` of ``results``; :attr:`calls` keeps query, k and filters."""

    def __init__(self, results: Sequence[ScoredChunk] = (), name: str = "fake") -> None:
        self.name = name
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        self.calls.append({"query": query, "k": k, "filters": filters})
        return self.results[:k]
