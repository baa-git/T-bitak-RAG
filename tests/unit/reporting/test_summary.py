"""Tests for the experiment-ladder summary ``ozet.md`` (FR-X8)."""

import shutil
from pathlib import Path

from tests.eval_builders import make_metrics, write_review
from tubitak_rag.domain.evaluation import RunMetrics
from tubitak_rag.reporting.summary import render_summary, write_summary


def _metrics(name: str, recall: float, p50: float, **overrides: object) -> RunMetrics:
    return make_metrics(
        name,
        run_id=f"{name}_20261016-120000",
        retrieval={"recall_at_5": recall, "recall_at_10": recall, "mrr": recall},
        latency_ms={"total_p50": p50},
        **overrides,
    )


def _write_run(runs: Path, run_id: str, metrics: RunMetrics) -> None:
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(metrics.model_dump_json(), encoding="utf-8")


def test_summary_table_lists_runs_in_ladder_order(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_run(runs, "v3_20261016-120000", _metrics("v3", 0.8, 2500.0, retriever="hybrid"))
    _write_run(runs, "v0_20261015-120000", _metrics("v0", 0.4, 900.0, chunker="fixed"))
    _write_run(runs, "v0_20261016-120000", _metrics("v0", 0.5, 950.0, chunker="fixed"))
    _write_run(runs, "v1_20261016-120000", _metrics("v1", 0.6, 1000.0))

    path = write_summary(runs)
    assert path == runs / "ozet.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Deney özeti\n")
    assert "| Koşu | Yöntem | R@5 | R@10 | MRR | Doğru ret | Yanlış ret | Gecikme p50 |" in text
    rows = [line for line in text.splitlines() if line.startswith("| v")]
    assert [row.split(" | ")[0] for row in rows] == ["| v0", "| v1", "| v3"]
    assert "| v0 | sabit pencere, dense arama, embedding EmbeddingGemma 2 | 0,500 |" in rows[0]
    assert "hibrit arama (dense + BM25, RRF)" in rows[2]
    assert "8 soru" in text and "doğruluk sütunu yok" in text
    assert "hakem" not in text and "qwen3:4b" not in text and "cevap doğruluğu" not in text


def test_highlight_sentence() -> None:
    text = render_summary([_metrics("v0", 0.4, 900.0), _metrics("v3", 0.8, 2500.0)])
    assert "En yüksek Recall@5 v3 (0,800)" in text
    assert "en düşük gecikme v0 (900 ms)" in text  # lower latency is better
    assert text.count("\n|") == 4  # a single table: header, separator and one row per run


def test_empty_summary(tmp_path: Path) -> None:
    assert "Henüz tamamlanmış koşu yok." in render_summary([])
    assert "Henüz" in write_summary(tmp_path / "yeni").read_text(encoding="utf-8")


def test_summary_shows_claude_accuracy_when_every_run_is_reviewed(
    tmp_path: Path, fixtures: Path
) -> None:
    runs = tmp_path / "runs"
    shutil.copytree(fixtures / "run_example", runs / "v1_20261016-120000")
    write_review(tmp_path / "judge")
    text = write_summary(runs).read_text(encoding="utf-8")
    assert "| Doğruluk (Claude) | Doğru ret (Claude) |" in text
    assert "| 0,200 | 0,667 |" in text
    assert "Claude (Opus) ile tek tek değerlendirilmesinden" in text
    # A second, unreviewed run: no accuracy column for anyone.
    v0 = _metrics("v0", 0.4, 900.0, chunker="fixed")
    _write_run(runs, "v0_20261015-120000", v0)
    mixed = write_summary(runs).read_text(encoding="utf-8")
    assert "Doğruluk" not in mixed and "Claude" not in mixed
