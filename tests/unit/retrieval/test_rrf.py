"""Tests for Reciprocal Rank Fusion (FR-R4: own code, known example)."""

import math

import pytest

from tubitak_rag.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_known_example() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [doc_id for doc_id, _ in fused] == ["a", "b", "c", "d"]  # a/b tie -> first seen
    scores = dict(fused)
    assert math.isclose(scores["a"], 1 / 61 + 1 / 62)
    assert math.isclose(scores["a"], 0.03252, abs_tol=1e-5)
    assert scores["a"] == scores["b"]
    assert math.isclose(scores["c"], 1 / 63)
    assert math.isclose(scores["d"], 1 / 63)


def test_rrf_item_in_both_lists_beats_single_top_hit() -> None:
    fused = reciprocal_rank_fusion([["x", "y"], ["y"]])
    assert [doc_id for doc_id, _ in fused] == ["y", "x"]
    assert math.isclose(dict(fused)["y"], 1 / 61 + 1 / 62)
    assert math.isclose(dict(fused)["x"], 1 / 61)


def test_rrf_k_changes_scores() -> None:
    (_, score), *_ = reciprocal_rank_fusion([["a"]], k=1)
    assert score == 0.5


@pytest.mark.parametrize("k", [0, -5])
def test_rrf_k_zero_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion([["a"]], k=k)


def test_rrf_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
