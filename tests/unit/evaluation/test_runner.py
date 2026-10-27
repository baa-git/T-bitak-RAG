"""Tests for the experiment runner (FR-X1, FR-X2, NFR-REPRO1)."""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.builders import make_chunk, make_scored
from tests.eval_builders import QUOTE, make_answer, make_question, make_unanswerable
from tests.fakes.llm import FakeLLM
from tests.fakes.pipeline import FakePipeline, FakeTrace, Scripted
from tubitak_rag.config import AppConfig, load_config
from tubitak_rag.domain.enums import AnswerStatus, QuestionType
from tubitak_rag.domain.evaluation import QuestionResult, RunMetrics, TestQuestion
from tubitak_rag.evaluation.judge import LlmJudge
from tubitak_rag.evaluation.runner import (
    QuestionDone,
    RunError,
    RunPaths,
    config_hash,
    is_run_of,
    make_run_id,
    ranked_results,
    run_experiment,
)
from tubitak_rag.guardrails.gate import GateResult
from tubitak_rag.guardrails.numbers import NumberVerdict
from tubitak_rag.ingestion.ingest import IngestReport

_VERDICT = json.dumps({"dogru": True, "sadik": True, "gerekce": "Doğru oran."})


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "experiment": {"name": "v1", "concept": "structural_context", "previous": "v0"},
            "paths": {"artifacts_dir": str(tmp_path / "artifacts")},
        }
    )


def _hit() -> list:
    return [
        make_scored(make_chunk(text=f"Madde 7. {QUOTE}."), 0.8, dense=0.8),
        make_scored(make_chunk("x1", doc_id="diger", text="Başka."), 0.5, dense=0.5),
    ]


def _setup() -> tuple[list[TestQuestion], FakePipeline]:
    """Five questions: answered, gate refusal, number refusal, scope refusal, number strip."""
    q_ok = make_question("q001")
    q_gate = make_question("q002", tur=QuestionType.SURUM)
    q_num = make_question("q003")
    q_scope = make_unanswerable("q004", QuestionType.KAPSAM_DISI)
    q_strip = make_question("q005", tur=QuestionType.COK_BELGE)
    passed = GateResult(passed=True, key="dense", score=0.8)
    stripped = NumberVerdict(
        ok=True, reason=None, cevap="x", eksik_kisim="y", atiflar=[1], ungrounded=["24"]
    )
    refused_answer = make_answer(durum=AnswerStatus.BILGI_BULUNAMADI)
    script = {
        q_ok.soru: Scripted(make_answer(q_ok.soru), candidates=_hit(), gate=passed),
        q_gate.soru: Scripted(
            refused_answer,
            candidates=_hit()[1:],
            gate=GateResult(passed=False, key="dense", score=0.5),
        ),
        q_num.soru: Scripted(
            refused_answer, candidates=_hit(), gate=passed, rejection_reason="dayanaksiz_sayi"
        ),
        q_scope.soru: Scripted(
            make_answer(durum=AnswerStatus.KAPSAM_DISI, gate_key=None, gate_score=None)
        ),
        q_strip.soru: Scripted(
            make_answer(q_strip.soru), candidates=_hit(), gate=passed, number_check=stripped
        ),
    }
    return [q_ok, q_gate, q_num, q_scope, q_strip], FakePipeline(script)


def _run(
    tmp_path: Path,
    judge: LlmJudge | None = None,
    ingest: Callable[[], IngestReport] | None = None,
    limit: int | None = None,
    resume: Path | None = None,
    pipeline: FakePipeline | None = None,
    progress: Callable[[QuestionDone], None] | None = None,
    cfg: AppConfig | None = None,
) -> tuple[RunMetrics, RunPaths]:
    questions, default_pipeline = _setup()
    return run_experiment(
        cfg or _cfg(tmp_path),
        pipeline=pipeline or default_pipeline,
        new_trace=FakeTrace,
        judge=judge,
        questions=questions,
        runs_dir=tmp_path / "runs",
        ingest=ingest,
        limit=limit,
        progress=progress,
        resume=resume,
    )


def _rows(paths: RunPaths) -> list[QuestionResult]:
    lines = paths.per_question_jsonl.read_text(encoding="utf-8").splitlines()
    return [QuestionResult.model_validate_json(line) for line in lines]


def test_make_run_id_and_run_membership() -> None:
    run_id = make_run_id("v3", datetime(2026, 10, 16, 9, 5, 7))
    assert run_id == "v3_20261016-090507"
    assert is_run_of(run_id, "v3")
    assert is_run_of("v3_20261016-090507-2", "v3")
    assert not is_run_of("v3b_20261016-090507", "v3")
    assert not is_run_of("v3_notes", "v3")


def test_run_writes_three_files_and_report_path(tmp_path: Path) -> None:
    metrics, paths = _run(tmp_path)
    assert paths.directory.parent == tmp_path / "runs"
    assert paths.run_id == metrics.run_id and is_run_of(metrics.run_id, "v1")
    for path in (paths.config_yaml, paths.metrics_json, paths.per_question_jsonl):
        assert path.is_file()
    assert paths.report_md == paths.directory / "report.md"
    assert not paths.report_md.exists()  # written by `rag run` via write_report
    reloaded = load_config(paths.config_yaml)
    assert reloaded == _cfg(tmp_path)
    saved = RunMetrics.model_validate_json(paths.metrics_json.read_text(encoding="utf-8"))
    assert saved == metrics


def test_per_question_rows_valid_schema(tmp_path: Path) -> None:
    _, paths = _run(tmp_path)
    rows = {r.id: r for r in _rows(paths)}
    assert list(rows) == ["q001", "q002", "q003", "q004", "q005"]
    ok = rows["q001"]
    assert ok.evidence_hits == [1] and ok.first_hit_rank == 1 and ok.doc_hit_at_5
    assert ok.retrieved[0]["konum"] == "Madde 7"
    assert set(ok.retrieved[0]) == {
        "chunk_id",
        "doc_id",
        "rank",
        "score",
        "scores",
        "doc_title",
        "konum",
        "metin",
    }
    assert ok.answer["atiflar"] == ["mevzuat-031:0001:00"]
    assert ok.rule_check["sayilar_bulundu"] is True
    assert ok.latency_ms["total"] == 111.0 and ok.llm_calls == 1
    assert rows["q002"].evidence_hits == [None] and rows["q002"].gate_score == 0.7


def test_refusal_reasons_recorded(tmp_path: Path) -> None:
    metrics, paths = _run(tmp_path)
    reasons = {r.id: r.answer["red_nedeni"] for r in _rows(paths)}
    assert reasons == {
        "q001": None,
        "q002": "esik_alti",
        "q003": "dayanaksiz_sayi",
        "q004": "kapsam_disi",
        "q005": None,
    }
    assert metrics.refusals == {"esik_alti": 1, "dayanaksiz_sayi": 1, "kapsam_disi": 1}
    assert [r.answer["sayi_ayiklandi"] for r in _rows(paths)] == [False, False, False, False, True]
    # no number check ran -> None; q005's verdict reports no repaired citation -> 0
    assert [r.answer["atif_onarildi"] for r in _rows(paths)] == [None, None, None, None, 0]
    # reached the LLM: q001, q003, q005
    assert metrics.generation["number_rejection_rate"] == pytest.approx(1 / 3)
    assert metrics.generation["number_strip_rate"] == pytest.approx(1 / 3)


def test_metrics_json_fields(tmp_path: Path) -> None:
    metrics, _ = _run(tmp_path)
    assert metrics.experiment == "v1" and metrics.judge_model == "yok"
    assert metrics.chunker == "structural_ctx" and metrics.retriever == "hybrid"
    assert metrics.reranker == "BAAI/bge-reranker-v2-m3"
    assert metrics.counts == {"questions": 5, "judge_errors": 0, "scope_false_rejections": 0}
    assert set(metrics.stage_times_s) == {"questions", "judge"}
    assert metrics.retrieval["recall_at_5"] == pytest.approx(3 / 4)
    assert metrics.generation["scope_rejection_rate"] == 1.0
    assert "tek_belge" in metrics.by_type


def test_limit(tmp_path: Path) -> None:
    metrics, paths = _run(tmp_path, limit=2)
    assert metrics.counts["questions"] == 2
    assert len(_rows(paths)) == 2


def test_same_inputs_same_retrieval_metrics(tmp_path: Path) -> None:
    first, first_paths = _run(tmp_path)
    second, second_paths = _run(tmp_path)
    assert first_paths.directory != second_paths.directory  # same second -> suffixed run id
    assert first.retrieval == second.retrieval
    assert first.generation == second.generation


def test_ingest_callback_called_once(tmp_path: Path) -> None:
    calls: list[int] = []

    def ingest() -> IngestReport:
        calls.append(1)
        return IngestReport.model_validate_json(_ingest_report_json())

    metrics, _ = _run(tmp_path, ingest=ingest)
    assert calls == [1]
    assert metrics.counts == {
        "questions": 5,
        "chunks": 4000,
        "docs": 120,
        "judge_errors": 0,
        "scope_false_rejections": 0,
    }
    assert "ingest" in metrics.stage_times_s


def test_counts_from_last_ingest_report_of_same_collection(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "ingest_report.json").write_text(_ingest_report_json(), encoding="utf-8")
    metrics, _ = _run(tmp_path)
    assert metrics.counts["chunks"] == 4000
    other = json.loads(_ingest_report_json()) | {"collection": "fixed__baska"}
    (artifacts / "ingest_report.json").write_text(json.dumps(other), encoding="utf-8")
    metrics, _ = _run(tmp_path)
    assert "chunks" not in metrics.counts


def test_judge_called_only_for_answered_answerable(tmp_path: Path) -> None:
    llm = FakeLLM([_VERDICT, _VERDICT], model_name="hakem")
    metrics, paths = _run(tmp_path, judge=LlmJudge(llm))
    assert len(llm.calls) == 2  # q001 and q005; refusals and out-of-scope are not judged
    assert QUOTE in llm.calls[0]["user"]  # the cited chunk text is shown to the judge
    rows = {r.id: r for r in _rows(paths)}
    assert rows["q001"].judge["dogru"] is True and rows["q002"].judge["dogru"] is None
    assert metrics.judge_model == "hakem"
    assert metrics.generation["faithfulness"] == 1.0


def test_ranked_results_puts_reranked_first_then_rest_of_pool() -> None:
    a, b, c = (make_scored(make_chunk(i), 0.1) for i in ("a", "b", "c"))
    trace = FakeTrace(candidates=[a, b, c], reranked=[c, a])
    assert [r.chunk.chunk_id for r in ranked_results(trace)] == ["c", "a", "b"]
    assert [r.chunk.chunk_id for r in ranked_results(FakeTrace(candidates=[a, b]))] == ["a", "b"]


def _ingest_report_json() -> str:
    return json.dumps(
        {
            "collection": "structural_ctx__google-embeddinggemma-2",
            "chunker": "structural_ctx",
            "embedder": "google/embeddinggemma-2",
            "full": False,
            "docs_total": 120,
            "added": [],
            "updated": [],
            "removed": [],
            "unchanged": 120,
            "scanned": [],
            "empty": [],
            "missing": [],
            "errors": [],
            "chunks_total": 4000,
            "chunks_indexed": 0,
            "parsed": 0,
            "parse_cached": 120,
            "chunked": 0,
            "chunk_cached": 120,
            "embedded": 0,
            "embed_cached": 0,
            "bm25_size": 4000,
            "stage_times_s": {"total": 1.0},
        }
    )


def test_progress_events_count_every_question(tmp_path: Path) -> None:
    events: list[QuestionDone] = []
    _run(tmp_path, progress=events.append)
    assert [(e.done, e.total, e.resumed) for e in events] == [(i, 5, 0) for i in range(1, 6)]
    assert events[0].result.id == "q001"


def test_resume_skips_done_questions_and_keeps_rows(tmp_path: Path) -> None:
    _, first = _run(tmp_path, limit=2)
    first.metrics_json.unlink()  # as if interrupted before the end
    questions, pipeline = _setup()
    events: list[QuestionDone] = []

    metrics, paths = _run(
        tmp_path, resume=first.directory, pipeline=pipeline, progress=events.append
    )

    assert paths == first
    assert pipeline.calls == [q.soru for q in questions[2:]]
    assert [r.id for r in _rows(paths)] == ["q001", "q002", "q003", "q004", "q005"]
    assert [(e.done, e.resumed) for e in events] == [(3, 2), (4, 2), (5, 2)]
    assert metrics.counts["questions"] == 5 and metrics.counts["resumed"] == 2
    assert metrics.started_at == datetime.strptime(paths.run_id[3:18], "%Y%m%d-%H%M%S").isoformat()
    fresh, _ = _run(tmp_path)
    assert metrics.retrieval == fresh.retrieval and metrics.generation == fresh.generation


def test_resume_drops_a_truncated_last_line(tmp_path: Path) -> None:
    _, first = _run(tmp_path, limit=2)
    with first.per_question_jsonl.open("a", encoding="utf-8") as out:
        out.write('{"id": "q003", "tur": "tek_b')  # killed mid-write
    metrics, paths = _run(tmp_path, resume=first.directory)
    assert [r.id for r in _rows(paths)] == ["q001", "q002", "q003", "q004", "q005"]
    assert metrics.counts["resumed"] == 2


def test_resume_refuses_a_changed_config_or_missing_run(tmp_path: Path) -> None:
    _, first = _run(tmp_path, limit=1)
    changed = _cfg(tmp_path).model_copy(
        update={"retriever": _cfg(tmp_path).retriever.model_copy(update={"top_n": 3})}
    )
    with pytest.raises(RunError, match="yapılandırması değişmiş"):
        _run(tmp_path, resume=first.directory, cfg=changed)
    with pytest.raises(RunError, match="bulunamadı"):
        _run(tmp_path, resume=tmp_path / "runs" / "v1_20000101-000000")
    assert config_hash("a") != config_hash("b") and len(config_hash("a")) == 12


def test_citation_repairs_recorded_when_the_verdict_reports_them(tmp_path: Path) -> None:
    questions, pipeline = _setup()
    repaired = SimpleNamespace(ok=True, ungrounded=[], repaired_citations=[1, 3])
    pipeline.script[questions[0].soru].number_check = repaired
    metrics, paths = _run(tmp_path, pipeline=pipeline)
    rows = {r.id: r for r in _rows(paths)}
    assert rows["q001"].answer["atif_onarildi"] == 2
    assert metrics.generation["citation_repair_rate"] == 0.5  # q001 of q001, q005


class _ExplodingOnce(FakePipeline):
    """Raises for the first question, then answers like the wrapped fake."""

    def ask(self, question, filters=None, *, trace=None):  # type: ignore[no-untyped-def]
        if not self.calls:
            self.calls.append(question)
            raise ValueError("bozuk sayı")
        return super().ask(question, filters, trace=trace)


def test_question_error_is_recorded_and_run_continues(tmp_path: Path) -> None:
    questions, fake = _setup()
    pipeline = _ExplodingOnce(fake.script)
    metrics, paths = _run(tmp_path, pipeline=pipeline)
    assert metrics.counts["pipeline_errors"] == 1
    assert len(_rows(paths)) == len(questions) - 1
    errors = paths.per_question_jsonl.with_name("errors.jsonl").read_text(encoding="utf-8")
    assert questions[0].id in errors and "ValueError" in errors
