"""Tests for the hybrid retriever: dense + BM25 fused with RRF (FR-R4, FR-R5)."""

import math

import pytest

from tests.builders import make_chunk, make_scored
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.ports import Retriever
from tubitak_rag.retrieval.hybrid import HybridRetriever


class _StubRetriever:
    """Returns fixed results (truncated to ``k``) and records every call."""

    def __init__(self, name: str, ids: list[str], key: str) -> None:
        self.name = name
        self._results = [
            make_scored(make_chunk(cid), score, **{key: score})
            for cid, score in zip(ids, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], strict=False)
        ]
        self.calls: list[tuple[str, int, MetadataFilter | None]] = []

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        self.calls.append((query, k, filters))
        return self._results[:k]


def _hybrid(
    dense_ids: list[str], bm25_ids: list[str], **kwargs: int
) -> tuple[_StubRetriever, _StubRetriever, HybridRetriever]:
    dense = _StubRetriever("dense", dense_ids, "dense")
    lexical = _StubRetriever("bm25", bm25_ids, "bm25")
    return dense, lexical, HybridRetriever(dense, lexical, **kwargs)


def test_is_a_retriever_named_hybrid() -> None:
    _, _, hybrid = _hybrid(["a"], ["a"])
    assert isinstance(hybrid, Retriever)
    assert hybrid.name == "hybrid"


def test_fusion_order_matches_rrf() -> None:
    _, _, hybrid = _hybrid(["a", "b", "c"], ["b", "a", "d"])
    results = hybrid.retrieve("q", k=4)
    assert [r.chunk.chunk_id for r in results] == ["a", "b", "c", "d"]
    assert math.isclose(results[0].score, 1 / 61 + 1 / 62)
    assert math.isclose(results[2].score, 1 / 63)


def test_returns_only_top_k() -> None:
    _, _, hybrid = _hybrid(["a", "b", "c"], ["b", "a", "d"])
    assert [r.chunk.chunk_id for r in hybrid.retrieve("q", k=2)] == ["a", "b"]


def test_scores_merged_from_both_sides() -> None:
    _, _, hybrid = _hybrid(["a", "b"], ["b", "c"])
    by_id = {r.chunk.chunk_id: r for r in hybrid.retrieve("q", k=3)}
    assert by_id["b"].scores == {"rrf": by_id["b"].score, "dense": 0.8, "bm25": 0.9}
    assert by_id["a"].scores == {"rrf": by_id["a"].score, "dense": 0.9}
    assert by_id["c"].scores == {"rrf": by_id["c"].score, "bm25": 0.8}


def test_result_present_in_only_one_side_still_returned() -> None:
    _, _, hybrid = _hybrid(["a"], [])
    assert [r.chunk.chunk_id for r in hybrid.retrieve("q", k=5)] == ["a"]


def test_filters_forwarded_to_both() -> None:
    dense, lexical, hybrid = _hybrid(["a"], ["b"], candidate_multiplier=3)
    only_1507 = MetadataFilter(program_kodu="1507")
    hybrid.retrieve("destek oranı", k=4, filters=only_1507)
    assert dense.calls == [("destek oranı", 12, only_1507)]
    assert lexical.calls == [("destek oranı", 12, only_1507)]


def test_rrf_k_is_used() -> None:
    _, _, hybrid = _hybrid(["a"], [], rrf_k=1)
    assert hybrid.retrieve("q", k=1)[0].score == 0.5


@pytest.mark.parametrize("kwargs", [{"rrf_k": 0}, {"candidate_multiplier": 0}])
def test_invalid_parameters_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        _hybrid(["a"], ["a"], **kwargs)
