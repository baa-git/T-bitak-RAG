"""Tests for gate-threshold calibration (FR-GR10)."""

import pytest

from tests.eval_builders import make_result, refused
from tubitak_rag.domain.enums import QuestionType
from tubitak_rag.evaluation.calibration import (
    CalibrationRow,
    calibrate,
    calibration_key,
    default_grid,
    suggest_threshold,
)

T = QuestionType


def _synthetic() -> list:
    answerable = [
        make_result(f"a{i}", gate_key="dense", gate_score=s)
        for i, s in enumerate([0.5, 0.6, 0.7, 0.8])
    ]
    unanswerable = [
        refused(f"u{i}", T.CEVAPSIZ, "esik_alti", gate_key="dense", gate_score=s)
        for i, s in enumerate([0.3, 0.4, 0.55])
    ]
    other_key = [make_result("r1", gate_key="rerank", gate_score=-3.0)]
    scope = [
        refused(
            "s1", T.KAPSAM_DISI, "kapsam_disi", durum="kapsam_disi", gate_key=None, gate_score=None
        )
    ]
    return answerable + unanswerable + other_key + scope


def test_calibrate_rates_on_synthetic_scores() -> None:
    rows = calibrate(_synthetic(), "dense", [0.45, 0.56, 0.9])
    low, mid, high = rows
    assert (low.correct_rejection_rate, low.false_rejection_rate) == (pytest.approx(2 / 3), 0.0)
    assert (mid.correct_rejection_rate, mid.false_rejection_rate) == (1.0, 0.25)
    assert (high.correct_rejection_rate, high.false_rejection_rate) == (1.0, 1.0)
    assert low.f_score == pytest.approx(2 * (2 / 3) * 1 / (2 / 3 + 1))
    assert high.f_score == 0.0


def test_calibrate_ignores_other_keys_and_ungated_questions() -> None:
    rows = calibrate(_synthetic(), "rerank", [0.0])
    assert rows[0].false_rejection_rate == 1.0  # the single rerank-gated answerable question
    assert rows[0].correct_rejection_rate == 0.0  # no unanswerable question gated by rerank


def test_suggest_respects_false_rejection_cap() -> None:
    rows = [
        CalibrationRow(
            threshold=0.4, correct_rejection_rate=0.5, false_rejection_rate=0.0, f_score=0.67
        ),
        CalibrationRow(
            threshold=0.5, correct_rejection_rate=0.8, false_rejection_rate=0.1, f_score=0.85
        ),
        CalibrationRow(
            threshold=0.6, correct_rejection_rate=1.0, false_rejection_rate=0.3, f_score=0.9
        ),
    ]
    assert suggest_threshold(rows).threshold == 0.5
    assert suggest_threshold(rows, max_false_rejection=0.5).threshold == 0.6
    strict = [r.model_copy(update={"false_rejection_rate": 0.2}) for r in rows]
    assert suggest_threshold(strict).threshold == 0.6  # nobody within the cap: lowest FR, best F
    with pytest.raises(ValueError):
        suggest_threshold([])


def test_grid_ranges() -> None:
    dense = default_grid("dense")
    assert (dense[0], dense[-1], len(dense)) == (0.2, 0.8, 31)
    rerank = default_grid("rerank")
    assert (rerank[0], rerank[-1], rerank[1] - rerank[0]) == (-11.0, 6.0, 0.5)
    bm25 = default_grid("bm25")
    assert (bm25[0], bm25[-1], len(bm25)) == (0.0, 30.0, 31)
    with pytest.raises(ValueError, match="rrf"):
        default_grid("rrf")


def test_calibration_key_is_most_frequent_threshold_key() -> None:
    assert calibration_key(_synthetic()) == "dense"
    assert calibration_key([make_result("x", gate_key="rrf")]) is None
    assert calibration_key([]) is None


def test_no_results_rows_count_as_rejected_at_every_threshold() -> None:
    rows = [
        refused("u1", T.CEVAPSIZ, "sonuc_yok", gate_key=None, gate_score=None),
        refused("u2", T.CEVAPSIZ, "esik_alti", gate_key="dense", gate_score=0.5),
        refused("a1", T.TEK_BELGE, "sonuc_yok", gate_key=None, gate_score=None),
        make_result("a2", gate_key="dense", gate_score=0.7),
        refused(
            "s1", T.KAPSAM_DISI, "kapsam_disi", durum="kapsam_disi", gate_key=None, gate_score=None
        ),
    ]
    (low,) = calibrate(rows, "dense", [0.1])
    assert (low.negatives, low.positives) == (2, 2)  # scope-stopped s1 is not counted
    assert low.correct_rejection_rate == 0.5  # u1 has no score: refused at any threshold
    assert low.false_rejection_rate == 0.5  # a1 likewise
