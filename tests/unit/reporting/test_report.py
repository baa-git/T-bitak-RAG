"""Tests for the short run report (FR-X3 … FR-X7, FR-E8, FR-GR10)."""

import shutil
from pathlib import Path

import pytest

from tests.eval_builders import make_metrics, make_result, refused, write_review
from tubitak_rag.config import AppConfig
from tubitak_rag.domain.enums import QuestionType
from tubitak_rag.domain.evaluation import QuestionResult, RunMetrics
from tubitak_rag.evaluation.calibration import CalibrationRow, calibrate
from tubitak_rag.evaluation.metrics import recompute
from tubitak_rag.reporting.concepts import CONCEPTS
from tubitak_rag.reporting.report import (
    SECTIONS,
    comparison_rows,
    describe_method,
    find_previous_run,
    fmt,
    fmt_delta,
    load_reviewed_run,
    load_run,
    render_report,
    write_report,
)
from tubitak_rag.reporting.review import REVIEWER

CALIBRATION = [
    CalibrationRow(
        threshold=0.3, correct_rejection_rate=0.0, false_rejection_rate=0.0, f_score=0.0
    ),
    CalibrationRow(
        threshold=0.4, correct_rejection_rate=0.67, false_rejection_rate=0.2, f_score=0.72
    ),
    CalibrationRow(
        threshold=0.5, correct_rejection_rate=1.0, false_rejection_rate=0.4, f_score=0.75
    ),
]


@pytest.fixture
def run(fixtures: Path) -> tuple[RunMetrics, list[QuestionResult]]:
    return load_run(fixtures / "run_example")


def _previous() -> RunMetrics:
    return make_metrics(
        "v0",
        run_id="v0_20261015-100000",
        chunker="fixed",
        retrieval={"recall_at_5": 0.5, "recall_at_10": 0.7, "mrr": 0.6, "doc_hit_at_5": 0.8},
        generation={"answer_accuracy": 0.3, "false_rejection_rate": 0.2},
        latency_ms={"total_p50": 900.0},
        by_type={"tek_belge": {"answer_accuracy": 0.4}, "surum": {"answer_accuracy": 0.5}},
    )


def _render(
    run: tuple[RunMetrics, list[QuestionResult]], previous: RunMetrics | None = None
) -> str:
    metrics, results = run
    return render_report(
        metrics, results, previous, CONCEPTS["structural_context"], CALIBRATION, "dense"
    )


def test_report_is_short_and_has_sections_in_order(
    run: tuple[RunMetrics, list[QuestionResult]],
) -> None:
    text = _render(run, _previous())
    assert text.startswith(
        "# Deney raporu: v1 – yapı bazlı parçalama + bağlam başlığı, dense arama, "
        "embedding EmbeddingGemma 2\n"
    )
    assert CONCEPTS["structural_context"].summary in text
    positions = [text.index(f"\n{heading}\n") for heading in SECTIONS]
    assert positions == sorted(positions)
    assert text.count("\n## ") == len(SECTIONS)
    assert len(text.splitlines()) < 60


def test_header_names_run_and_corpus_but_not_the_automatic_judge(
    run: tuple[RunMetrics, list[QuestionResult]],
) -> None:
    header = _render(run).split("\n## ")[0]
    assert "koşu `v1_20261016-120000`" in header
    assert "8 soru" in header and "120 belge kaydı, 4.000 parça" in header
    assert "dil modeli qwen3:4b" in header and "süre 30 dk 5 sn" in header
    assert "hakem" not in header and "değerlendirildi" not in header


def test_describe_method_names_every_component() -> None:
    full = make_metrics(
        "v3b", retriever="hybrid", reranker="BAAI/bge-reranker-v2-m3", embedder="BAAI/bge-m3"
    )
    assert describe_method(full) == (
        "yapı bazlı parçalama + bağlam başlığı, hibrit arama (dense + BM25, RRF), reranker, "
        "embedding bge-m3"
    )
    other = make_metrics(chunker="fixed", embedder="org/yeni-model")
    assert describe_method(other) == "sabit pencere, dense arama, embedding yeni-model"


def test_key_metrics_table_with_previous_rung(
    run: tuple[RunMetrics, list[QuestionResult]],
) -> None:
    text = _render(run, _previous())
    assert "| Metrik | Önceki (v0) | Bu koşu | Fark |" in text
    assert "| Recall@5 | 0,500 | 0,700 | +0,200 |" in text
    assert "| Doğru ret oranı | — | 0,667 | — |" in text  # no previous value
    assert "| Toplam p50 | 900 ms | 1 005 ms | +105 ms |" in text
    assert "Belge isabeti" not in text  # key metrics only
    # Without a manual review the automatic judge's verdicts are not shown at all.
    assert "Cevap doğruluğu" not in text and "Kaynağa sadakat" not in text
    assert "qwen3:4b" not in text.split("\n## ", 1)[1]
    no_previous = _render(run)
    assert "| Metrik | Bu koşu |" in no_previous and "| Recall@5 | 0,700 |" in no_previous
    assert "taban çizgisidir" in no_previous


def test_refusals_and_calibration_in_one_line_each(
    run: tuple[RunMetrics, list[QuestionResult]],
) -> None:
    text = _render(run)
    refusals = next(line for line in text.splitlines() if line.startswith("- Retler"))
    assert "kaynakta olmayan sayı 1" in refusals
    assert (
        "Eşik kalibrasyonu: `dense` skoru için yanlış ret ≤ 0,100 koşuluyla önerilen eşik "
        "**0,3** (doğru ret 0,000, yanlış ret 0,000)." in text
    )
    metrics, results = run
    empty = render_report(metrics, results, None, CONCEPTS["full_pipeline"], None)
    assert "Eşik kalibrasyonu" not in empty


def test_examples_show_question_evidence_answer_and_judge(
    run: tuple[RunMetrics, list[QuestionResult]],
) -> None:
    examples = _render(run).split(f"\n{SECTIONS[1]}\n")[1]
    good, first_bad, second_bad = examples.split("\n### ")[1:]
    assert good.startswith("İyi örnek: q001 (tek belge)")
    assert "**Soru:** 1507 programında destek oranı" in good
    assert "- Kanıt: 1. sırada · durum: cevaplandı" in good
    assert "- Cevap: 1507 programında destek oranı en fazla %75'tir [1]." in good
    assert "notu" not in examples  # the automatic judge's reasoning is never shown
    assert first_bad.startswith("Kötü örnek: q031 (cevapsız)")
    assert "beklenmiyor (corpus'ta cevabı yok)" in first_bad and "- Cevap:" in first_bad
    assert second_bad.startswith("Kötü örnek: q010 (sürüm)")
    assert "durum: bilgi bulunamadı (skor eşiğin altında)" in second_bad
    assert "- Cevap:" not in second_bad  # refusal text is not repeated


def test_long_answers_are_cut() -> None:
    long = make_result("q001", answer=make_result().answer | {"cevap": "uzun " * 200})
    text = render_report(make_metrics(), [long], None, CONCEPTS["full_pipeline"], None)
    answer = next(line for line in text.splitlines() if line.startswith("- Cevap:"))
    assert answer.endswith("…") and len(answer) < 420


def test_comparison_rows_skip_metrics_missing_in_current() -> None:
    current = make_metrics(latency_ms={})
    rows = comparison_rows(current, _previous())
    assert "total_p50" not in {r.metric for r in rows}
    assert next(r for r in rows if r.metric == "recall_at_5").delta == pytest.approx(0.1)
    every = comparison_rows(current, _previous(), keys=("doc_hit_at_5",))
    assert [r.metric for r in every] == ["doc_hit_at_5"]


def test_number_formatting() -> None:
    assert fmt(0.72) == "0,720" and fmt(None) == "—" and fmt(1250.4, "ms") == "1 250 ms"
    assert fmt_delta(-0.05) == "−0,050" and fmt_delta(35.0, "ms") == "+35 ms"


def test_find_previous_run_picks_newest_finished(tmp_path: Path, fixtures: Path) -> None:
    runs = tmp_path / "runs"
    for name in ("v0_20261015-100000", "v0_20261016-090000", "v0r_20261017-000000"):
        shutil.copytree(fixtures / "run_example", runs / name)
    (runs / "v0_20261018-000000").mkdir()  # unfinished: no metrics.json
    assert find_previous_run(runs, "v0") == runs / "v0_20261016-090000"
    assert find_previous_run(runs, "v2") is None
    assert find_previous_run(runs, None) is None
    assert find_previous_run(tmp_path / "yok", "v0") is None


def test_write_report_compares_with_previous_rung(tmp_path: Path, fixtures: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "v1_20261016-120000"
    shutil.copytree(fixtures / "run_example", run_dir)
    previous_dir = runs / "v0_20261015-100000"
    previous_dir.mkdir()
    (previous_dir / "metrics.json").write_text(_previous().model_dump_json(), encoding="utf-8")
    cfg = AppConfig.model_validate(
        {"experiment": {"name": "v1", "concept": "structural_context", "previous": "v0"}}
    )
    path = write_report(run_dir, runs, cfg)
    assert path == run_dir / "report.md"
    text = path.read_text(encoding="utf-8")
    assert "| Metrik | Önceki (v0) | Bu koşu | Fark |" in text
    assert "`dense` skoru için" in text  # calibrated on the run's own gate key
    assert CONCEPTS["structural_context"].summary in text


def test_citation_repairs_scope_misses_and_llm_errors_are_noted() -> None:
    answered = make_result("a")
    error = refused("c", QuestionType.TEK_BELGE, "llm_hatasi", llm_calls=1)
    scope = refused("d", QuestionType.TEK_BELGE, "kapsam_disi", durum="kapsam_disi")
    results = [answered, error, scope]
    metrics = recompute(make_metrics(refusals={"llm_hatasi": 1}), results)
    text = render_report(metrics, results, None, CONCEPTS["full_pipeline"], None)
    assert "**Dikkat:** 1 soruda dil modeli hata verdi" in text
    assert "Kapsam kontrolü 1 cevaplanabilir soruyu kapsam dışı saydı." in text
    assert "dil modeli hatası 1" in text
    plain = render_report(make_metrics(), [answered], None, CONCEPTS["full_pipeline"], None)
    assert "Dikkat" not in plain and "Retler" not in plain


# --- R3 review fixes ------------------------------------------------------------------------


def _write_run(runs: Path, run_id: str, metrics: RunMetrics, rows: list[QuestionResult]) -> Path:
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(metrics.model_dump_json(), encoding="utf-8")
    text = "".join(r.model_dump_json() + "\n" for r in rows)
    (run_dir / "per_question.jsonl").write_text(text, encoding="utf-8")
    return run_dir


def test_find_previous_run_prefers_full_runs_over_newer_partial_ones(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    full = _write_run(runs, "v0_20261015-100000", make_metrics("v0", counts={"questions": 8}), [])
    _write_run(runs, "v0_20261016-100000", make_metrics("v0", counts={"questions": 5}), [])
    assert find_previous_run(runs, "v0") == full
    newer_full = _write_run(
        runs, "v0_20261017-100000", make_metrics("v0", counts={"questions": 8}), []
    )
    assert find_previous_run(runs, "v0") == newer_full


def test_warning_when_previous_run_differs_in_size(
    run: tuple[RunMetrics, list[QuestionResult]],
) -> None:
    metrics, results = run
    same = _previous().model_copy(update={"counts": dict(metrics.counts)})
    assert "Dikkat" not in render_report(metrics, results, same, CONCEPTS["reranker"], None)
    differs = _previous().model_copy(update={"counts": {"questions": 5}, "judge_model": "yok"})
    text = render_report(metrics, results, differs, CONCEPTS["reranker"], None)
    assert "**Dikkat:** önceki koşu 5 soru, bu koşu 8 soru. Karşılaştırma eşit değil." in text
    assert "hakem" not in text


def test_no_judge_run_reports_number_rule_accuracy(
    run: tuple[RunMetrics, list[QuestionResult]],
) -> None:
    metrics, results = run
    no_judge = recompute(metrics.model_copy(update={"judge_model": "yok"}), results)
    text = render_report(no_judge, results, None, CONCEPTS["full_pipeline"], None)
    assert "| Sayı kuralı doğruluğu |" in text and "| Cevap doğruluğu" not in text


def test_write_report_recomputes_metrics_from_rows(tmp_path: Path, fixtures: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "v1_20261016-120000"
    shutil.copytree(fixtures / "run_example", run_dir)
    stale = RunMetrics.model_validate_json((run_dir / "metrics.json").read_text("utf-8"))
    stale = stale.model_copy(update={"retrieval": {"recall_at_5": 0.0}})
    (run_dir / "metrics.json").write_text(stale.model_dump_json(), encoding="utf-8")
    cfg = AppConfig.model_validate({"experiment": {"name": "v1", "previous": "v0"}})

    write_report(run_dir, runs, cfg)

    saved = RunMetrics.model_validate_json((run_dir / "metrics.json").read_text("utf-8"))
    assert saved.retrieval["recall_at_5"] == pytest.approx(0.7)
    assert saved.counts["judge_errors"] == 0
    assert "| Recall@5 | 0,700 |" in (run_dir / "report.md").read_text("utf-8")


def test_calibration_uses_the_run_rows(run: tuple[RunMetrics, list[QuestionResult]]) -> None:
    metrics, results = run
    rows = calibrate(results, "dense", [0.3, 0.4])
    text = render_report(metrics, results, None, CONCEPTS["full_pipeline"], rows, "dense")
    assert "`dense` skoru için" in text and "önerilen eşik" in text


# --- manual review (Claude) ----------------------------------------------------------------


def test_reviewed_report_shows_claude_metrics_and_notes(tmp_path: Path, fixtures: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "v1_20261016-120000"
    shutil.copytree(fixtures / "run_example", run_dir)
    write_review(tmp_path / "judge")
    metrics, results, reviewer = load_reviewed_run(run_dir, runs)
    assert reviewer == REVIEWER
    text = render_report(metrics, results, None, CONCEPTS["full_pipeline"], None, reviewer=reviewer)
    header = text.split("\n## ")[0]
    assert "cevaplar Claude (Opus) ile tek tek değerlendirildi" in header
    assert "| Cevap doğruluğu (Claude) | 0,200 |" in text
    assert "| Kaynağa sadakat (Claude) | 1,000 |" in text
    assert "| Doğru ret oranı (Claude) | 0,667 |" in text
    assert "- Claude değerlendirmesi: uydurma 1 · gereksiz ret 2 (hatanın yeri: arama 1, " in text
    assert "- Değerlendirme notu (Claude): Noter kuralı kaynakta yok." in text
    assert "Hakem notu" not in text and "hakem" not in text


def test_reviewed_report_does_not_compare_with_unreviewed_previous_run(
    tmp_path: Path, fixtures: Path
) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "v1_20261016-120000"
    shutil.copytree(fixtures / "run_example", run_dir)
    write_review(tmp_path / "judge")
    previous_dir = runs / "v0_20261015-100000"
    previous_dir.mkdir()
    (previous_dir / "metrics.json").write_text(_previous().model_dump_json(), encoding="utf-8")
    cfg = AppConfig.model_validate(
        {"experiment": {"name": "v1", "concept": "structural_context", "previous": "v0"}}
    )
    text = write_report(run_dir, runs, cfg).read_text(encoding="utf-8")
    assert "| Cevap doğruluğu (Claude) | — | 0,200 | — |" in text  # 0,300 is the judge's
    assert "önceki koşunun cevapları Claude (Opus) ile değerlendirilmedi" in text
    saved = RunMetrics.model_validate_json((run_dir / "metrics.json").read_text("utf-8"))
    assert saved.judge_model == "qwen3:4b"  # metrics.json keeps the run's own metrics


def test_write_report_without_review_has_no_accuracy(tmp_path: Path, fixtures: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "v1_20261016-120000"
    shutil.copytree(fixtures / "run_example", run_dir)
    cfg = AppConfig.model_validate({"experiment": {"name": "v1", "previous": "v0"}})
    text = write_report(run_dir, runs, cfg).read_text(encoding="utf-8")
    assert "Cevap doğruluğu" not in text and "notu" not in text and "hakem" not in text
