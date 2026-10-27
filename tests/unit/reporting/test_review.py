"""Tests for the manual answer review (Claude) used by the reports."""

import json
from pathlib import Path

import pytest

from tests.eval_builders import REVIEW_OTHERS, make_metrics, write_review
from tubitak_rag.domain.evaluation import QuestionResult, RunMetrics
from tubitak_rag.reporting.report import load_run
from tubitak_rag.reporting.review import (
    apply_review,
    covers,
    find_review,
    load_review,
    review_counts,
    review_dir,
    strip_keys,
)


@pytest.fixture
def run(fixtures: Path) -> tuple[RunMetrics, list[QuestionResult]]:
    return load_run(fixtures / "run_example")


def test_review_dir_is_next_to_runs(tmp_path: Path) -> None:
    assert review_dir(tmp_path / "artifacts" / "runs") == tmp_path / "artifacts" / "judge"


def test_load_review_needs_both_files(tmp_path: Path) -> None:
    assert load_review(tmp_path, "v1") is None
    write_review(tmp_path)
    review = load_review(tmp_path, "v1")
    assert review is not None and set(review.answers) == {"q001", "q002", "q005"}
    (tmp_path / "v1_rest_claude.json").write_text("{bozuk", encoding="utf-8")
    assert load_review(tmp_path, "v1") is None


def test_review_must_cover_every_row(
    tmp_path: Path, run: tuple[RunMetrics, list[QuestionResult]]
) -> None:
    _, results = run
    write_review(tmp_path, others=REVIEW_OTHERS[:-1])  # q040 missing
    review = load_review(tmp_path, "v1")
    assert review is not None and not covers(review, results)
    assert find_review(tmp_path, "v1", results) is None


def test_stale_review_of_other_answers_is_ignored(
    tmp_path: Path, run: tuple[RunMetrics, list[QuestionResult]]
) -> None:
    _, results = run
    write_review(tmp_path)
    assert find_review(tmp_path, "v1", results) is not None
    snapshot = [{"id": r.id, "cevap": r.answer["cevap"]} for r in results[:3]]
    (tmp_path / "v1_items.json").write_text(json.dumps(snapshot), encoding="utf-8")
    assert find_review(tmp_path, "v1", results) is not None  # same answers
    snapshot[0]["cevap"] = "Başka bir koşunun cevabı."
    (tmp_path / "v1_items.json").write_text(json.dumps(snapshot), encoding="utf-8")
    assert find_review(tmp_path, "v1", results) is None
    (tmp_path / "v1_items.json").unlink()
    rest = [{"id": "q010", "durum": "cevaplandi"}]
    (tmp_path / "v1_rest.json").write_text(json.dumps(rest), encoding="utf-8")
    assert find_review(tmp_path, "v1", results) is None


def test_apply_review_replaces_judge_metrics_and_notes(
    tmp_path: Path, run: tuple[RunMetrics, list[QuestionResult]]
) -> None:
    metrics, results = run
    write_review(tmp_path)
    review = find_review(tmp_path, "v1", results)
    assert review is not None
    shown, rows = apply_review(metrics, results, review)
    assert shown.generation["answer_accuracy"] == pytest.approx(1 / 5)
    assert shown.generation["faithfulness"] == pytest.approx(1.0)  # undecided q005 left out
    assert shown.generation["correct_rejection_rate"] == pytest.approx(2 / 3)
    assert shown.generation["citation_precision"] == metrics.generation["citation_precision"]
    by_id = {r.id: r for r in rows}
    assert by_id["q002"].judge["dogru"] is False
    assert by_id["q002"].judge["gerekce"] == "Süre eksik verilmiş."
    assert by_id["q031"].judge["karar"] == "uydurma"
    assert review_counts(rows) == {
        "gereksiz_ret": 2,
        "gereksiz_ret:retrieval": 1,
        "gereksiz_ret:uretim": 1,
        "dogru_ret": 2,
        "uydurma": 1,
    }
    assert results[1].judge["dogru"] is True  # the input rows are not changed


def test_strip_keys_drops_judge_metrics() -> None:
    stripped = strip_keys(make_metrics())
    assert "answer_accuracy" not in stripped.generation
    assert "faithfulness" not in stripped.generation
    assert stripped.generation["correct_rejection_rate"] == 0.7
