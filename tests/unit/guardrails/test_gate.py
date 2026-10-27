"""Tests for the retrieval score gate (FR-GR1, FR-GR9)."""

import inspect

import pytest

from tests.builders import make_chunk, make_scored
from tubitak_rag.config import GuardrailConfig
from tubitak_rag.guardrails import gate as gate_module
from tubitak_rag.guardrails.gate import GateResult, RetrievalGate

THRESHOLDS = {"dense": 0.5, "rerank": -1.0, "bm25": 2.0}


def test_uses_rerank_key_when_present() -> None:
    results = [
        make_scored(make_chunk("a"), 0.9, dense=0.9, rerank=-3.0),
        make_scored(make_chunk("b"), 0.1, dense=0.1, rerank=-0.5),
    ]
    assert RetrievalGate(THRESHOLDS).evaluate(results) == GateResult(
        passed=True, key="rerank", score=-0.5
    )


def test_falls_back_to_dense() -> None:
    results = [make_scored(make_chunk("a"), 0.7, dense=0.7, bm25=1.0, rrf=0.03)]
    assert RetrievalGate(THRESHOLDS).evaluate(results) == GateResult(
        passed=True, key="dense", score=0.7
    )


def test_falls_back_to_bm25() -> None:
    results = [make_scored(make_chunk("a"), 0.03, bm25=1.5, rrf=0.03)]
    assert RetrievalGate(THRESHOLDS).evaluate(results) == GateResult(
        passed=False, key="bm25", score=1.5
    )


def test_below_threshold_fails() -> None:
    results = [make_scored(make_chunk("a"), 0.4, dense=0.4)]
    assert not RetrievalGate(THRESHOLDS).evaluate(results).passed


def test_score_equal_to_threshold_passes() -> None:
    results = [make_scored(make_chunk("a"), 0.5, dense=0.5)]
    assert RetrievalGate(THRESHOLDS).evaluate(results).passed


def test_gate_empty_results_fails() -> None:
    assert RetrievalGate(THRESHOLDS).evaluate([]) == GateResult(passed=False, key=None, score=None)


def test_results_without_known_score_fail() -> None:
    results = [make_scored(make_chunk("a"), 0.03, rrf=0.03)]
    assert RetrievalGate(THRESHOLDS).evaluate(results) == GateResult(
        passed=False, key=None, score=None
    )


def test_thresholds_from_config_not_constants() -> None:
    cfg = GuardrailConfig(thresholds={"dense": 0.95, "rerank": 5.0, "bm25": 0.0})
    results = [make_scored(make_chunk("a"), 0.9, dense=0.9)]
    assert not RetrievalGate(cfg.thresholds).evaluate(results).passed
    source = inspect.getsource(gate_module)
    for default in ("0.45", "-2.0"):
        assert default not in source


def test_missing_threshold_key_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match=r"guardrails\.thresholds.*bm25.*rerank"):
        RetrievalGate({"dense": 0.5})
