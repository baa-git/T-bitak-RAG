"""Word-overlap reranker for unit tests."""

import re
from collections.abc import Sequence

from tubitak_rag.domain.retrieval import ScoredChunk

_WORD = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class FakeReranker:
    """Score = how many times the query's words occur in the chunk text."""

    def __init__(self, model_name: str = "fake") -> None:
        self.model_name = model_name

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        terms = set(_words(query))
        rescored = []
        for cand in candidates:
            score = float(sum(1 for word in _words(cand.chunk.text) if word in terms))
            scores = {**cand.scores, "rerank": score}
            rescored.append(cand.model_copy(update={"score": score, "scores": scores}))
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_n]
