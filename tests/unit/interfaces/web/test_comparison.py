"""Run summary (``/api/runs``) and two-run comparison (demo reports)."""

from tubitak_rag.domain.evaluation import RunMetrics
from tubitak_rag.interfaces.web.comparison import (
    HEADLINE,
    compare,
    summarize,
)


def _metrics(run_id: str, recall: float, total_p50: float, **extra: object) -> RunMetrics:
    fields: dict[str, object] = {
        "run_id": run_id,
        "experiment": run_id.split("_")[0],
        "started_at": "2026-10-15T09:00:00",
        "finished_at": "2026-10-15T09:30:00",
        "wall_clock_s": 1800.0,
        "judge_model": "qwen3:4b",
        "llm_model": "qwen3:4b",
        "embedder": "google/embeddinggemma-2",
        "chunker": "structural_ctx",
        "retriever": "hybrid",
        "reranker": None,
        "counts": {"questions": 10},
        "retrieval": {"recall_at_5": recall, "mrr": 0.5},
        "generation": {"answer_accuracy": 0.6, "false_rejection_rate": 0.1},
        "latency_ms": {"total_p50": total_p50},
        "by_type": {},
        "stage_times_s": {},
    }
    fields.update(extra)
    return RunMetrics.model_validate(fields)


def test_summarize_keeps_headline_metrics_present() -> None:
    summary = summarize(_metrics("v1_a", 0.5, 7000.0))
    assert summary.run_id == "v1_a" and summary.deney == "v1" and summary.durum == "tamam"
    assert set(summary.metrikler) <= set(HEADLINE)
    assert summary.metrikler["recall_at_5"] == 0.5
    assert "correct_rejection_rate" not in summary.metrikler  # missing stays missing


def test_summarize_marks_unfinished_run() -> None:
    assert summarize(_metrics("v1_a", 0.5, 1.0, finished_at="")).durum == "suruyor"


def test_compare_deltas_and_improvement_direction() -> None:
    result = compare(
        _metrics("v1_a", 0.5, 7000.0),
        _metrics("v2_b", 0.7, 7200.0, reranker="BAAI/bge-reranker-v2-m3"),
    )
    rows = {row.spec.key: row for _, group in result.groups for row in group}
    assert round(rows["recall_at_5"].delta or 0, 3) == 0.2
    assert rows["recall_at_5"].improved is True
    assert rows["total_p50"].improved is False  # slower is worse
    assert rows["mrr"].improved is None  # unchanged
    assert "recall_at_10" not in rows  # absent in both runs -> not shown
    settings = {row.label: row for row in result.settings}
    assert settings["Yeniden sıralayıcı"].a == "yok" and settings["Yeniden sıralayıcı"].changed
    assert not settings["Parçalama"].changed


def test_compare_metric_missing_in_one_run_has_no_delta() -> None:
    a = _metrics("v1_a", 0.5, 7000.0)
    b = _metrics("v2_b", 0.6, 7000.0, generation={"answer_accuracy": 0.7})
    rows = {row.spec.key: row for _, group in compare(a, b).groups for row in group}
    assert rows["false_rejection_rate"].b is None
    assert rows["false_rejection_rate"].delta is None
    assert rows["false_rejection_rate"].improved is None
