"""Tests for the core reranking wrapper (FR-RR1, FR-RR2)."""

from collections.abc import Sequence

import pytest

from tests.builders import make_chunk, make_scored
from tests.fakes.reranker import FakeReranker
from tubitak_rag.domain.retrieval import ScoredChunk
from tubitak_rag.reranking.cross_encoder import rerank_with


def _candidates(n: int) -> list[ScoredChunk]:
    # Candidate i mentions "destek" i % 7 times: the reranker order differs from the input.
    return [
        make_scored(
            make_chunk(f"c{i:02d}", text="destek " * (i % 7) + "proje"),
            1.0 / (i + 1),
            rrf=1.0 / (i + 1),
        )
        for i in range(n)
    ]


def test_rerank_with_returns_top_n_sorted() -> None:
    results = rerank_with(FakeReranker(), "destek", _candidates(30), top_n=5)
    assert len(results) == 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 6.0


def test_rerank_scores_attached() -> None:
    for result in rerank_with(FakeReranker(), "destek", _candidates(30), top_n=5):
        assert result.scores["rerank"] == result.score
        assert "rrf" in result.scores  # first-stage scores are kept


def test_empty_candidates() -> None:
    class _MustNotBeCalled(FakeReranker):
        def rerank(
            self, query: str, candidates: Sequence[ScoredChunk], top_n: int
        ) -> list[ScoredChunk]:
            raise AssertionError("reranker called without candidates")

    assert rerank_with(_MustNotBeCalled(), "destek", [], top_n=5) == []


@pytest.mark.parametrize("top_n", [0, -1])
def test_non_positive_top_n_rejected(top_n: int) -> None:
    with pytest.raises(ValueError, match="top_n"):
        rerank_with(FakeReranker(), "destek", _candidates(3), top_n=top_n)


def test_fewer_candidates_than_top_n() -> None:
    assert len(rerank_with(FakeReranker(), "destek", _candidates(3), top_n=5)) == 3


def test_result_without_rerank_score_rejected() -> None:
    class _Forgetful(FakeReranker):
        def rerank(
            self, query: str, candidates: Sequence[ScoredChunk], top_n: int
        ) -> list[ScoredChunk]:
            return list(candidates[:top_n])

    with pytest.raises(ValueError, match="rerank"):
        rerank_with(_Forgetful(), "destek", _candidates(3), top_n=2)


def test_more_than_top_n_results_rejected() -> None:
    class _Greedy(FakeReranker):
        def rerank(
            self, query: str, candidates: Sequence[ScoredChunk], top_n: int
        ) -> list[ScoredChunk]:
            return super().rerank(query, candidates, len(candidates))

    with pytest.raises(ValueError, match="top_n"):
        rerank_with(_Greedy(), "destek", _candidates(3), top_n=2)
