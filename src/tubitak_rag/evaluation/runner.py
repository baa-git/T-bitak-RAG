"""Experiment runner: one YAML config in, one ``artifacts/runs/<run_id>/`` folder out (FR-X1).

``run_experiment`` asks every test question through the pipeline, scores retrieval from the
pipeline's own trace (no second retrieval call), judges answered answerable questions and
writes

* ``config.yaml`` — the fully resolved configuration (FR-X2, NFR-REPRO1);
* ``per_question.jsonl`` — one :class:`QuestionResult` per line, appended and flushed as soon
  as the question is done, so an interrupted run keeps what it measured;
* ``metrics.json`` — the aggregated :class:`RunMetrics`.

``report.md`` is written afterwards by ``reporting.report.write_report`` (``rag run``).

The pipeline is used through the small :class:`TracingPipeline` protocol (``RagPipeline``
satisfies it with ``new_trace=AskTrace``). Two *optional* trace attributes refine the
recorded refusal reason when the pipeline provides them: ``rejection_reason`` (e.g.
``"gecersiz_atif"``, ``"dayanaksiz_sayi"``) and ``number_check``
(:class:`~tubitak_rag.guardrails.numbers.NumberVerdict`, to count stripped numbers and, when
the verdict has ``repaired_citations``, citations the guardrail added).
"""

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from tubitak_rag.chunking.cache import chunker_key
from tubitak_rag.config import AppConfig, ConfigError
from tubitak_rag.domain.answer import Answer
from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.domain.evaluation import QuestionResult, RunMetrics, TestQuestion
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.evaluation.judge import (
    JudgeVerdict,
    LlmJudge,
    citation_precision,
    numbers_present,
    programs_split,
)
from tubitak_rag.evaluation.matching import doc_hit, evidence_hits, first_hit_rank
from tubitak_rag.evaluation.metrics import (
    REASON_BELOW_THRESHOLD,
    REASON_CITATION,
    REASON_NO_RESULTS,
    REASON_NOT_FOUND,
    REASON_OUT_OF_SCOPE,
    aggregate,
    run_counts,
)
from tubitak_rag.evaluation.testset import ANSWERABLE
from tubitak_rag.generation.prompts import source_location
from tubitak_rag.guardrails.gate import GateResult
from tubitak_rag.indexing.naming import collection_name
from tubitak_rag.ingestion.ingest import REPORT_FILE, IngestReport

PREVIEW_CHARS = 300
LF = "\n"
_RUN_ID_RE = r"\d{8}-\d{6}(?:-\d+)?"


class PipelineTrace(Protocol):
    """What the runner reads from a filled pipeline trace (``AskTrace``)."""

    @property
    def candidates(self) -> Sequence[ScoredChunk]: ...

    @property
    def reranked(self) -> Sequence[ScoredChunk]: ...

    @property
    def gate(self) -> GateResult | None: ...


class TracingPipeline(Protocol):
    """A pipeline that fills the trace it is given while answering (``RagPipeline.ask``).

    ``trace`` is typed loosely on purpose: the runner creates it with ``new_trace`` (whose
    return type is checked against :class:`PipelineTrace`) and only hands it through.
    """

    def ask(
        self, question: str, filters: MetadataFilter | None = None, *, trace: Any = None
    ) -> Answer: ...


@dataclass(frozen=True)
class RunPaths:
    """Files of one run folder."""

    directory: Path

    @property
    def run_id(self) -> str:
        return self.directory.name

    @property
    def config_yaml(self) -> Path:
        return self.directory / "config.yaml"

    @property
    def metrics_json(self) -> Path:
        return self.directory / "metrics.json"

    @property
    def per_question_jsonl(self) -> Path:
        return self.directory / "per_question.jsonl"

    @property
    def report_md(self) -> Path:
        return self.directory / "report.md"


def make_run_id(experiment_name: str, now: datetime | None = None) -> str:
    """``<experiment>_<YYYYmmdd-HHMMSS>``."""
    return f"{experiment_name}_{(now or datetime.now()):%Y%m%d-%H%M%S}"


def is_run_of(run_id: str, experiment_name: str) -> bool:
    """Whether ``run_id`` belongs to ``experiment_name`` ("v3_…" is not a run of "v3b")."""
    return re.fullmatch(rf"{re.escape(experiment_name)}_{_RUN_ID_RE}", run_id) is not None


def _new_run_dir(runs_dir: Path, experiment_name: str) -> RunPaths:
    base = make_run_id(experiment_name)
    candidate, n = runs_dir / base, 1
    while candidate.exists():  # two runs in the same second
        n += 1
        candidate = runs_dir / f"{base}-{n}"
    candidate.mkdir(parents=True)
    return RunPaths(candidate)


# --- one question --------------------------------------------------------------------------


def ranked_results(trace: PipelineTrace) -> list[ScoredChunk]:
    """Final ranking: reranked results first, then the remaining candidates in their order.

    ``RagPipeline`` reranks the whole candidate pool, so the tail is usually empty; it keeps
    recall@10 meaningful for a pipeline whose reranker returns fewer results.
    """
    reranked = list(trace.reranked)
    seen = {r.chunk.chunk_id for r in reranked}
    return reranked + [c for c in trace.candidates if c.chunk.chunk_id not in seen]


def refusal_reason(answer: Answer, trace: PipelineTrace) -> str | None:
    """Why the question was not answered (None when answered); see ``metrics`` for codes."""
    if answer.durum == AnswerStatus.CEVAPLANDI:
        return None
    explicit = getattr(trace, "rejection_reason", None)
    if explicit:
        return str(explicit)
    if answer.durum == AnswerStatus.KAPSAM_DISI:
        return REASON_OUT_OF_SCOPE
    gate = trace.gate
    if gate is None or gate.key is None:
        return REASON_NO_RESULTS
    if not gate.passed:
        return REASON_BELOW_THRESHOLD
    if answer.durum == AnswerStatus.ATIF_REDDI:
        return REASON_CITATION
    return REASON_NOT_FOUND


def numbers_stripped(answer: Answer, trace: PipelineTrace) -> bool:
    """Whether the number guardrail removed sentences from an answer that was still given."""
    verdict = getattr(trace, "number_check", None)
    return bool(
        answer.durum == AnswerStatus.CEVAPLANDI
        and verdict is not None
        and verdict.ok
        and verdict.ungrounded
    )


def citations_repaired(answer: Answer, trace: PipelineTrace) -> int | None:
    """How many citations the number guardrail added (None when the pipeline does not say)."""
    verdict = getattr(trace, "number_check", None)
    repaired = getattr(verdict, "repaired_citations", None)
    if repaired is None:
        return None
    return len(repaired) if answer.durum == AnswerStatus.CEVAPLANDI else 0


def _retrieved_row(rank: int, result: ScoredChunk) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "rank": rank,
        "score": result.score,
        "scores": dict(result.scores),
        "doc_title": chunk.doc_title,
        "konum": source_location(chunk),
        "metin": chunk.text[:PREVIEW_CHARS],
    }


def evaluate_answer(
    question: TestQuestion,
    answer: Answer,
    trace: PipelineTrace,
    judge: LlmJudge | None,
) -> QuestionResult:
    """Score one answered question into a ``per_question.jsonl`` row."""
    retrieved = ranked_results(trace)
    hits = evidence_hits(question, retrieved)
    precision = citation_precision(answer, question, retrieved)
    verdict = JudgeVerdict(dogru=None, sadik=None, gerekce="")
    if judge is not None and answer.durum == AnswerStatus.CEVAPLANDI and question.tur in ANSWERABLE:
        texts = {r.chunk.chunk_id: r.chunk.text for r in retrieved}
        cited = [texts[c.chunk_id] for c in answer.atiflar if c.chunk_id in texts]
        verdict = judge.judge(question, answer, cited)
    return QuestionResult(
        id=question.id,
        tur=question.tur,
        soru=question.soru,
        retrieved=[_retrieved_row(rank, r) for rank, r in enumerate(retrieved, start=1)],
        evidence_hits=hits,
        first_hit_rank=first_hit_rank(hits),
        doc_hit_at_5=doc_hit(question, retrieved, 5),
        doc_hit_at_10=doc_hit(question, retrieved, 10),
        gate_key=answer.gate_key,
        gate_score=answer.gate_score,
        answer={
            "durum": answer.durum.value,
            "cevap": answer.cevap,
            "atiflar": [c.chunk_id for c in answer.atiflar],
            "red_nedeni": refusal_reason(answer, trace),
            "sayi_ayiklandi": numbers_stripped(answer, trace),
            "atif_onarildi": citations_repaired(answer, trace),
        },
        judge={
            "dogru": verdict.dogru,
            "sadik": verdict.sadik,
            "atif_dogru": None if precision is None else precision == 1.0,
            "gerekce": verdict.gerekce,
        },
        rule_check={
            "sayilar_bulundu": numbers_present(question.beklenen_sayilar, answer.cevap),
            "atif_kesinligi": precision,
            "programlara_ayrildi": programs_split(question, answer.cevap),
        },
        latency_ms=dict(answer.gecikme_ms),
        llm_calls=answer.llm_cagri_sayisi,
    )


# --- the run -------------------------------------------------------------------------------


def _corpus_counts(cfg: AppConfig, ingest_report: IngestReport | None) -> dict[str, int]:
    """Chunks and documents of the run's collection (from this ingest or the last one)."""
    report = ingest_report
    if report is None:
        path = cfg.paths.artifacts_dir / REPORT_FILE
        try:
            report = IngestReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return {}
        if report.collection != collection_name(chunker_key(cfg.chunker), cfg.embedder.model):
            return {}
    return {"chunks": report.chunks_total, "docs": report.docs_total}


class RunError(ValueError):
    """A run cannot start or resume (message in Turkish)."""


@dataclass(frozen=True)
class QuestionDone:
    """Progress event after one question: ``done`` counts resumed questions too."""

    run_id: str
    done: int
    total: int
    resumed: int  # questions taken over from the interrupted run (not timed in this session)
    result: QuestionResult


def config_hash(config_yaml: str) -> str:
    """Short fingerprint of a resolved configuration (``AppConfig.to_yaml()`` text)."""
    return hashlib.sha256(config_yaml.encode("utf-8")).hexdigest()[:12]


def load_results(path: Path) -> list[QuestionResult]:
    """Rows of a ``per_question.jsonl``; a truncated *last* line (interrupted write) is dropped
    and the file rewritten without it, any other bad line raises :class:`RunError`."""
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[QuestionResult] = []
    for number, line in enumerate(lines, start=1):
        try:
            rows.append(QuestionResult.model_validate_json(line))
        except ValidationError as exc:
            if number < len(lines):
                message = exc.errors()[0]["msg"]
                raise RunError(f"{path} satır {number} okunamadı: {message}") from exc
            path.write_text(
                "".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8", newline="\n"
            )
    return rows


def _resume_dir(resume: Path, cfg: AppConfig) -> RunPaths:
    paths = RunPaths(resume)
    if not paths.config_yaml.is_file():
        raise RunError(f"Devam ettirilecek koşu bulunamadı: {resume}")
    stored = paths.config_yaml.read_text(encoding="utf-8")
    if config_hash(stored) != config_hash(cfg.to_yaml()):
        raise RunError(
            f"{paths.run_id} koşusunun yapılandırması değişmiş (özet {config_hash(stored)} ≠ "
            f"{config_hash(cfg.to_yaml())}); aynı YAML ile devam edin ya da yeni koşu başlatın."
        )
    return paths


def _started_at(run_id: str) -> datetime:
    match = re.search(r"_(\d{8}-\d{6})(?:-\d+)?$", run_id)
    return datetime.strptime(match[1], "%Y%m%d-%H%M%S") if match else datetime.now()


def run_experiment(
    cfg: AppConfig,
    *,
    pipeline: TracingPipeline,
    new_trace: Callable[[], PipelineTrace],
    judge: LlmJudge | None,
    questions: Sequence[TestQuestion],
    runs_dir: Path,
    ingest: Callable[[], IngestReport] | None,
    limit: int | None = None,
    progress: Callable[[QuestionDone], None] | None = None,
    resume: Path | None = None,
) -> tuple[RunMetrics, RunPaths]:
    """Run the questions (first ``limit``) sequentially and write the run folder.

    With ``resume`` (a run folder of the same configuration) the questions already in its
    ``per_question.jsonl`` are kept and skipped; the metrics cover old and new rows. The wall
    clock of a resumed run adds the interrupted session's question latencies to this one.
    """
    clock = time.perf_counter()
    if resume is not None:
        paths = _resume_dir(resume, cfg)
        started = _started_at(paths.run_id)
    else:
        started = datetime.now()
        paths = _new_run_dir(runs_dir, cfg.experiment.name)
        paths.config_yaml.write_text(cfg.to_yaml(), encoding="utf-8")
    previous = load_results(paths.per_question_jsonl)
    stage_times: dict[str, float] = {}

    ingest_report = None
    if ingest is not None:
        t0 = time.perf_counter()
        ingest_report = ingest()
        stage_times["ingest"] = time.perf_counter() - t0

    limit = limit if limit is not None else cfg.eval.limit
    selected = list(questions[:limit] if limit is not None else questions)
    done_ids = {r.id for r in previous}
    todo = [q for q in selected if q.id not in done_ids]
    resumed = len(selected) - len(todo)
    results = list(previous)
    judge_s, t0 = 0.0, time.perf_counter()
    failures: list[dict[str, str]] = []
    errors_path = paths.per_question_jsonl.with_name("errors.jsonl")
    with paths.per_question_jsonl.open("a", encoding="utf-8", newline="\n") as out:
        for index, question in enumerate(todo, start=1):
            trace = new_trace()
            try:
                answer = pipeline.ask(question.soru, trace=trace)
            except ConfigError:
                raise  # setup problems (e.g. missing index) must stop the run
            except Exception as exc:  # one broken question must not abort a multi-hour run
                failures.append({"id": question.id, "hata": f"{type(exc).__name__}: {exc}"})
                with errors_path.open("a", encoding="utf-8", newline=LF) as err:
                    err.write(json.dumps(failures[-1], ensure_ascii=False) + LF)
                continue
            t_judge = time.perf_counter()
            result = evaluate_answer(question, answer, trace, judge)
            judge_s += time.perf_counter() - t_judge
            out.write(result.model_dump_json() + "\n")
            out.flush()
            results.append(result)
            if progress is not None:
                progress(
                    QuestionDone(paths.run_id, resumed + index, len(selected), resumed, result)
                )
    stage_times["questions"] = time.perf_counter() - t0 - judge_s
    stage_times["judge"] = judge_s

    earlier_s = sum(r.latency_ms.get("total", 0.0) for r in previous) / 1000
    counts = {
        "questions": len(results),
        **_corpus_counts(cfg, ingest_report),
        **run_counts(results),
    }
    if previous:
        counts["resumed"] = len(previous)
    if failures:
        counts["pipeline_errors"] = len(failures)
    metrics = RunMetrics(
        run_id=paths.run_id,
        experiment=cfg.experiment.name,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now().isoformat(timespec="seconds"),
        wall_clock_s=time.perf_counter() - clock + earlier_s,
        judge_model=judge.model_name if judge is not None else "yok",
        llm_model=cfg.llm.model_name,
        embedder=cfg.embedder.model,
        chunker=chunker_key(cfg.chunker),
        retriever=cfg.retriever.mode,
        reranker=cfg.reranker.model if cfg.reranker.enabled else None,
        counts=counts,
        stage_times_s=stage_times,
        **aggregate(results, k_values=cfg.eval.k_values, judged=judge is not None),
    )
    paths.metrics_json.write_text(
        json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics, paths
