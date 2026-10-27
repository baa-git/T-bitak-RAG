"""``rag`` command-line interface (Typer). All user-facing text is Turkish (NFR-LANG1).

Every command loads and validates the configuration first, so an invalid config stops the
program with exit code 2 before any work starts (NFR-CONF1). Commands are filled in by later
phases; for now each reports which phase will implement it.
"""

import json
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from tubitak_rag.config import AppConfig, ConfigError, LlmConfig, load_config
from tubitak_rag.domain.answer import Answer
from tubitak_rag.domain.enums import AnswerStatus, DocType
from tubitak_rag.domain.evaluation import RunMetrics, TestQuestion
from tubitak_rag.domain.retrieval import MetadataFilter
from tubitak_rag.evaluation.judge import LlmJudge
from tubitak_rag.evaluation.runner import QuestionDone, RunError, RunPaths, run_experiment
from tubitak_rag.evaluation.testset import TestsetError, load_testset
from tubitak_rag.factory import (
    build_chunker,
    build_embedder,
    build_lexical_index,
    build_llm,
    build_pipeline,
    build_vector_store,
)
from tubitak_rag.ingestion.fetch import (
    FetchReport,
    FetchStatus,
    fetch_all,
    make_client,
    write_report,
)
from tubitak_rag.ingestion.ingest import REPORT_FILE, IngestReport, LazyEmbedder, run_ingest
from tubitak_rag.ingestion.manifest import MEVZUAT, PROGRAMLAR, SANAYI, load_sources
from tubitak_rag.interfaces.gpu_lock import GPU_LOCK_FILE, GpuLockError, gpu_lock, uses_gpu
from tubitak_rag.interfaces.index_bundle_cli import export_index
from tubitak_rag.interfaces.space_bundle_cli import export_space
from tubitak_rag.pipeline import REFUSAL_REASONS, AskTrace
from tubitak_rag.reporting.report import SPECS, fmt
from tubitak_rag.reporting.report import write_report as write_run_report
from tubitak_rag.reporting.summary import write_summary

app = typer.Typer(help="TÜBİTAK Mevzuat RAG komut satırı", no_args_is_help=True)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Yapılandırma dosyası (YAML)."),
]
DEFAULT_CONFIG = Path("configs/default.yaml")


def _load(config: Path, overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """Load the config or exit with code 2 and a Turkish error message."""
    try:
        return load_config(config, overrides)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


class SourceGroup(StrEnum):
    """Source groups selectable with ``rag fetch --only``."""

    MEVZUAT = "mevzuat"
    PROGRAMLAR = "programlar"
    SANAYI = "sanayi"


_GROUP_KEYS = {
    SourceGroup.MEVZUAT: MEVZUAT,
    SourceGroup.PROGRAMLAR: PROGRAMLAR,
    SourceGroup.SANAYI: SANAYI,
}
_PROBLEMS = (FetchStatus.UYUSMAZLIK, FetchStatus.HATA)


def _print_fetch_summary(report: FetchReport, report_path: Path) -> None:
    typer.echo("Durum                   Adet")
    for status, count in report.counts().items():
        typer.echo(f"{status:<22}  {count:>4}")
    problems = [r for r in report.results if r.status in _PROBLEMS]
    if problems:
        typer.echo("\nUyuşmazlık / hata:")
        for r in problems:
            typer.echo(f"- {r.doc_id} ({r.status.value}): {r.local_path} {r.message}".rstrip())
    typer.echo(f"\nRapor: {report_path}")


@app.command()
def fetch(
    config: ConfigOption = DEFAULT_CONFIG,
    force: Annotated[
        bool, typer.Option("--force", help="Yerelde olan dosyaları da yeniden indir.")
    ] = False,
    only: Annotated[
        SourceGroup | None, typer.Option("--only", help="Yalnızca bu kaynak grubunu indir.")
    ] = None,
) -> None:
    """Kaynak belgeleri indirir ve sha256 özetlerini doğrular."""
    cfg = _load(config)
    entries = load_sources(cfg.paths.manifests_dir)
    if only is not None:
        entries = [e for e in entries if e.kaynak_grubu == _GROUP_KEYS[only]]
    with make_client() as client:
        report = fetch_all(entries, cfg.paths.raw_dir, client, force=force)
    report_path = cfg.paths.artifacts_dir / "fetch_report.json"
    write_report(report, report_path)
    _print_fetch_summary(report, report_path)
    counts = report.counts()
    if counts[FetchStatus.UYUSMAZLIK] or counts[FetchStatus.HATA]:
        raise typer.Exit(code=1)


# Texts per embedding batch. GPU batches are capped at 32; ingest defaults lower so a long
# run keeps the GPU (and the power supply) below peak load.
INGEST_BATCH_SIZE = 16
MAX_GPU_BATCH_SIZE = 32
_STAGE_LABELS = {
    "parse": "ayrıştırma",
    "chunk": "parçalama",
    "embed": "embedding + indeks",
    "bm25": "BM25",
}


class ProgressPrinter:
    """Prints one progress line per stage at most every ``interval_s`` seconds.

    The last event of a stage (``done == total``) is always printed. Embedding lines also
    show the throughput and the estimated remaining time.
    """

    def __init__(
        self, interval_s: float = 5.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._interval_s = interval_s
        self._clock = clock
        self._stage = ""
        self._started = 0.0
        self._last_print = 0.0

    def __call__(self, stage: str, done: int, total: int) -> None:
        now = self._clock()
        if stage != self._stage:
            self._stage, self._started, self._last_print = stage, now, now
        if done < total and now - self._last_print < self._interval_s:
            return
        self._last_print = now
        line = f"  {_STAGE_LABELS.get(stage, stage)}: {done}/{total}"
        elapsed = now - self._started
        if stage == "embed" and done and elapsed > 0:
            rate = done / elapsed
            line += f" parça · {rate:.0f} parça/s · kalan ~{(total - done) / rate:.0f} s"
        typer.echo(line)


@contextmanager
def _gpu_guard(cfg: AppConfig, owner: str, *, answering: bool = True) -> Iterator[None]:
    """Hold ``artifacts/.gpu.lock`` while a real model runs; all-fake runs skip it.

    ``answering=False`` (ingest) only uses the embedder.
    """
    if not uses_gpu(cfg, answering=answering):
        yield
        return
    with gpu_lock(cfg.paths.artifacts_dir / GPU_LOCK_FILE, owner):
        yield


def _print_ids(title: str, ids: list[str]) -> None:
    if ids:
        typer.echo(f"{title}: {', '.join(ids)}")


def _print_ingest_summary(report: IngestReport, report_path: Path) -> None:
    times = report.stage_times_s
    typer.echo(
        f"\nKoleksiyon: {report.collection}{'  (--full: baştan kuruldu)' if report.full else ''}"
    )
    typer.echo(f"Belgeler (manifestte {report.docs_total}):")
    rows = [
        ("eklendi", len(report.added)),
        ("güncellendi", len(report.updated)),
        ("silindi", len(report.removed)),
        ("değişmedi", report.unchanged),
        ("taranmış", len(report.scanned)),
        ("boş", len(report.empty)),
        ("eksik dosya", len(report.missing)),
        ("hata", len(report.errors)),
    ]
    for label, count in rows:
        typer.echo(f"  {label:<12} {count:>5}")
    typer.echo(
        f"Parçalar: indekste {report.chunks_total} · bu koşuda yazılan {report.chunks_indexed} "
        f"({report.embedded} embed edildi, {report.embed_cached} önbellekten)"
    )
    typer.echo(
        f"Önbellek isabeti: ayrıştırma {report.parse_cached}/{report.parse_cached + report.parsed}"
        f" · parçalama {report.chunk_cached}/{report.chunk_cached + report.chunked}"
        f" · embedding {report.embed_cached}/{report.chunks_indexed}"
    )
    typer.echo(f"BM25: {report.bm25_size} parça")
    typer.echo(
        f"Süreler (s): ayrıştırma {times.get('parse', 0):.1f} · parçalama "
        f"{times.get('chunk', 0):.1f} · embedding + indeks {times.get('embed_index', 0):.1f} · "
        f"BM25 {times.get('bm25', 0):.1f} · toplam {times.get('total', 0):.1f}"
    )
    _print_ids("Taranmış (indekslenmedi)", report.scanned)
    _print_ids("Parçası çıkmayan", report.empty)
    _print_ids("Eksik dosya", report.missing)
    if report.errors:
        typer.echo("Hatalar:")
        for error in report.errors:
            typer.echo(f"- {error['doc_id']}: {error['message']}")
    typer.echo(f"Rapor: {report_path}")


@app.command()
def ingest(
    config: ConfigOption = DEFAULT_CONFIG,
    full: Annotated[
        bool,
        typer.Option("--full", help="Koleksiyonu silip baştan kurar (önbellekler korunur)."),
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            max=MAX_GPU_BATCH_SIZE,
            help="Embedding partisindeki en fazla metin sayısı (GPU kuralı: en fazla 32).",
        ),
    ] = INGEST_BATCH_SIZE,
) -> None:
    """Belgeleri ayrıştırır, parçalar ve indeksler (artımlı; yarıda kalırsa kaldığı yerden)."""
    cfg = _load(config, {"embedder": {"batch_size": batch_size}})
    owner = f"rag ingest ({cfg.experiment.name}, {cfg.embedder.model}) pid={os.getpid()}"
    embedder = LazyEmbedder(
        lambda: build_embedder(cfg), model_name=cfg.embedder.model, dimension=cfg.embedder.dimension
    )
    typer.echo(
        f"Ingest: deney {cfg.experiment.name} · chunker {cfg.chunker.name} · "
        f"embedder {cfg.embedder.model} · parti {batch_size}"
    )
    try:
        with _gpu_guard(cfg, owner, answering=False):
            report = run_ingest(
                cfg,
                embedder=embedder,
                store=build_vector_store(cfg),
                chunker=build_chunker(cfg),
                lexical=build_lexical_index(cfg),
                full=full,
                progress=ProgressPrinter(),
            )
    except GpuLockError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _print_ingest_summary(report, cfg.paths.artifacts_dir / REPORT_FILE)


RUNS_DIR = "runs"
# Headline metrics printed at the end of `rag run`.
_RUN_SUMMARY = (
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "answer_accuracy",
    "number_rule_accuracy",
    "faithfulness",
    "correct_rejection_rate",
    "false_rejection_rate",
    "error_rate",
    "total_p50",
)


def _duration_tr(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} sa {minutes} dk"
    return f"{minutes} dk {secs} sn" if minutes else f"{secs} sn"


def _decimal_tr(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


class RunProgressPrinter:
    """One line per question with the average time per question and the remaining time.

    Questions taken over from an interrupted run (``--resume``) are not timed in this
    session, so the estimate uses only the questions asked since the printer was created.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        echo: Callable[[str], None] = typer.echo,
    ) -> None:
        self._clock = clock
        self._echo = echo
        self._started = clock()
        self.run_id: str | None = None

    def __call__(self, event: QuestionDone) -> None:
        if self.run_id is None and event.resumed:
            self._echo(f"devam: {event.resumed} soru önceki oturumdan alındı")
        self.run_id = event.run_id
        asked = max(event.done - event.resumed, 1)
        per_question = (self._clock() - self._started) / asked
        remaining = per_question * (event.total - event.done)
        answer = event.result.answer
        reason = answer.get("red_nedeni")
        status = f"{answer['durum']} ({reason})" if reason else str(answer["durum"])
        self._echo(
            f"[{event.done}/{event.total}] {event.result.id} {status} · "
            f"{_decimal_tr(per_question)} s/soru · kalan ~{_duration_tr(remaining)}"
        )


def _judge_llm_config(cfg: AppConfig) -> LlmConfig:
    """The judge's LLM: the answer LLM's connection settings with the judge model, temp 0."""
    judge = cfg.eval.judge
    return cfg.llm.model_copy(
        update={"provider": judge.provider, "model": judge.model, "temperature": 0.0}
    )


def _select(questions: list[TestQuestion], ids: str | None) -> list[TestQuestion]:
    """Questions whose id is in the comma-separated ``ids`` (test-set order kept)."""
    if not ids:
        return questions
    wanted = {part.strip() for part in ids.split(",") if part.strip()}
    unknown = sorted(wanted - {q.id for q in questions})
    if unknown:
        typer.echo(f"Test setinde olmayan soru kimlikleri: {', '.join(unknown)}", err=True)
        raise typer.Exit(code=2)
    return [q for q in questions if q.id in wanted]


def _ingest_for(cfg: AppConfig) -> Callable[[], IngestReport]:
    """Incremental ingest of the experiment's collection (unchanged documents are skipped).

    The embedder lives only inside the call, so its model is freed before the pipeline
    loads its own copy.
    """

    def ingest() -> IngestReport:
        typer.echo(f"Ingest (artımlı): {cfg.chunker.name} · {cfg.embedder.model}")
        embedder = LazyEmbedder(
            lambda: build_embedder(cfg),
            model_name=cfg.embedder.model,
            dimension=cfg.embedder.dimension,
        )
        return run_ingest(
            cfg,
            embedder=embedder,
            store=build_vector_store(cfg),
            chunker=build_chunker(cfg),
            lexical=build_lexical_index(cfg),
            full=False,
            progress=ProgressPrinter(),
        )

    return ingest


class _LazyPipeline:
    """Builds the pipeline at the first question, i.e. after the ingest (BM25 index exists)."""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._pipeline: Any = None

    def ask(
        self, question: str, filters: MetadataFilter | None = None, *, trace: Any = None
    ) -> Answer:
        if self._pipeline is None:
            self._pipeline = build_pipeline(self._cfg)
        answer: Answer = self._pipeline.ask(question, filters, trace=trace)
        return answer


def _print_run_summary(metrics: RunMetrics, paths: RunPaths, summary: Path) -> None:
    typer.echo(
        f"\nKoşu {metrics.run_id} tamamlandı: {metrics.counts.get('questions', 0)} soru · "
        f"{_duration_tr(metrics.wall_clock_s)} · hakem {metrics.judge_model}"
    )
    for key in _RUN_SUMMARY:
        spec = SPECS[key]
        value = getattr(metrics, spec.group).get(key)
        if value is not None:
            typer.echo(f"  {spec.label:<26} {fmt(value, spec.kind)}")
    if metrics.refusals:
        reasons = ", ".join(f"{code} {n}" for code, n in metrics.refusals.items())
        typer.echo(f"Ret nedenleri: {reasons}")
    typer.echo(f"Rapor: {paths.report_md}")
    typer.echo(f"Özet: {summary}")


@app.command()
def run(
    config: Annotated[
        Path, typer.Argument(help="Deney yapılandırması (ör. configs/experiments/v3.yaml).")
    ],
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Yalnız ilk N soruyu koş.")
    ] = None,
    ids: Annotated[
        str | None, typer.Option("--ids", help="Yalnız bu soruları koş (virgülle: q001,q002).")
    ] = None,
    no_judge: Annotated[
        bool, typer.Option("--no-judge", help="Hakem modelini çağırma (yalnız kural kontrolleri).")
    ] = False,
    no_ingest: Annotated[
        bool, typer.Option("--no-ingest", help="Koşudan önce artımlı ingest yapma.")
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Yarıda kalan koşuya kaldığı yerden devam et (run_id)."),
    ] = None,
) -> None:
    """Bir deney koşusu çalıştırır; sonuçları, raporu ve özeti yazar (GPU kilidiyle)."""
    cfg = _load(config, {"llm": {"temperature": 0.0}})  # NFR-REPRO2
    _tolerant_stdout()
    providers = {cfg.llm.provider} | (set() if no_judge else {cfg.eval.judge.provider})
    external = sorted(p for p in providers if cfg.llm.is_external(p))
    if external:
        typer.echo(
            f"Uyarı: yapılandırma dış bir API kullanıyor ({', '.join(external)}; llm ya da "
            "eval.judge); sorular ve kaynak metinler o servise gider ve ücretlendirilebilir. "
            "Varsayılan ve test edilen yol yerel Ollama'dır.",
            err=True,
        )
    try:
        questions = _select(load_testset(cfg.paths.testset), ids)
    except (OSError, TestsetError) as exc:
        typer.echo(f"Test seti okunamadı ({cfg.paths.testset}): {exc}", err=True)
        raise typer.Exit(code=2) from exc
    runs_dir = cfg.paths.artifacts_dir / RUNS_DIR
    resume_dir = runs_dir / resume if resume else None
    owner = f"rag run ({cfg.experiment.name}) pid={os.getpid()}"
    printer = RunProgressPrinter()
    count = len(questions[:limit] if limit else questions)
    judge_name = "yok" if no_judge else cfg.eval.judge.model
    typer.echo(
        f"Deney {cfg.experiment.name}: {count} soru · LLM {cfg.llm.model} · hakem {judge_name}"
    )
    try:
        with _gpu_guard(cfg, owner):
            judge = None if no_judge else LlmJudge(build_llm(_judge_llm_config(cfg)))
            metrics, paths = run_experiment(
                cfg,
                pipeline=_LazyPipeline(cfg),
                new_trace=AskTrace,
                judge=judge,
                questions=questions,
                runs_dir=runs_dir,
                ingest=None if no_ingest or resume_dir else _ingest_for(cfg),
                limit=limit,
                progress=printer,
                resume=resume_dir,
            )
    except GpuLockError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except (ConfigError, RunError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except KeyboardInterrupt as exc:
        typer.echo("\nKoşu kesildi; GPU kilidi bırakıldı.", err=True)
        if printer.run_id:
            typer.echo(
                f"Kaldığı yerden devam: rag run {config} --resume {printer.run_id}", err=True
            )
        raise typer.Exit(code=130) from exc
    write_run_report(paths.directory, runs_dir, cfg)
    _print_run_summary(metrics, paths, write_summary(runs_dir))


def _finished_runs(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    return sorted(d for d in runs_dir.iterdir() if RunPaths(d).metrics_json.is_file())


def _rewrite_report(run_dir: Path, runs_dir: Path) -> Path:
    """Rebuild ``report.md`` with the run's own saved configuration."""
    return write_run_report(run_dir, runs_dir, _load(RunPaths(run_dir).config_yaml))


@app.command()
def report(
    run_id: Annotated[
        str | None, typer.Argument(help="Raporu (yeniden) üretilecek koşu (run_id).")
    ] = None,
    all_runs: Annotated[
        bool, typer.Option("--all", help="Tüm koşuların raporlarını ve özeti yeniden üret.")
    ] = False,
    summary_only: Annotated[
        bool, typer.Option("--summary", help="Yalnız özet tabloyu (runs/ozet.md) üret.")
    ] = False,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Koşu sonuçlarından Markdown rapor ve deney özeti üretir (model çalıştırmaz)."""
    cfg = _load(config)
    runs_dir = cfg.paths.artifacts_dir / RUNS_DIR
    if run_id is None and not all_runs and not summary_only:
        typer.echo("Bir run_id verin ya da --all / --summary kullanın.", err=True)
        raise typer.Exit(code=2)
    if run_id is not None:
        run_dir = runs_dir / run_id
        if not RunPaths(run_dir).metrics_json.is_file():
            typer.echo(f"Tamamlanmış koşu bulunamadı: {run_dir}", err=True)
            raise typer.Exit(code=2)
        typer.echo(f"Rapor: {_rewrite_report(run_dir, runs_dir)}")
    if all_runs:
        for run_dir in _finished_runs(runs_dir):
            typer.echo(f"Rapor: {_rewrite_report(run_dir, runs_dir)}")
    typer.echo(f"Özet: {write_summary(runs_dir)}")


_STATUS_LABELS = {
    AnswerStatus.CEVAPLANDI: "cevaplandı",
    AnswerStatus.BILGI_BULUNAMADI: "bilgi bulunamadı",
    AnswerStatus.KAPSAM_DISI: "kapsam dışı",
    AnswerStatus.ATIF_REDDI: "atıf denetiminden geçmedi (bilgi bulunamadı)",
}
_LATENCY_LABELS = {
    "scope": "kapsam",
    "retrieval": "arama",
    "rerank": "rerank",
    "llm": "LLM",
    "total": "toplam",
}


def _tolerant_stdout() -> None:
    """Replace characters the console encoding lacks (cp1254 pipes) instead of crashing.

    Answers quote source text, which may hold symbols such as "≤" that the Turkish Windows
    codepage cannot encode when the output is piped (``| Tee-Object``).
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(errors="replace")


def _print_answer(answer: Answer, trace: AskTrace, thresholds: Mapping[str, float]) -> None:
    typer.echo(f"Soru: {answer.soru}")
    typer.echo(f"Durum: {_STATUS_LABELS[answer.durum]}\n")
    typer.echo(answer.cevap)
    if answer.atiflar:
        typer.echo("\nAtıflar:")
        for citation in answer.atiflar:
            location = f" — {citation.konum}" if citation.konum else ""
            typer.echo(f"  [{citation.no}] {citation.doc_title}{location}")
            typer.echo(f"      {citation.kaynak_url}")
    typer.echo(f"\n{answer.uyari}\n")
    if answer.gate_key is not None and answer.gate_score is not None:
        threshold = thresholds.get(answer.gate_key)
        verdict = "geçti" if trace.gate is None or trace.gate.passed else "geçmedi"
        limit = f"eşik {threshold:.2f}, " if threshold is not None else ""
        typer.echo(
            f"Arama eşiği: {answer.gate_key} skoru {answer.gate_score:.2f} ({limit}{verdict}) · "
            f"kaynak panelinde {len(answer.kaynaklar)} parça"
        )
    if trace.number_check is not None and trace.number_check.ok and trace.number_check.ungrounded:
        typer.echo("Sayı denetimi: kaynakta geçmeyen sayı içeren cümle cevaptan çıkarıldı.")
    if answer.atif_onarimi:
        repaired = " ".join(f"[{number}]" for number in answer.atif_onarimi)
        typer.echo(f"Atıf onarımı: {repaired} guardrail tarafından eklendi")
    if trace.rejection_reason:
        reason = REFUSAL_REASONS.get(trace.rejection_reason, trace.rejection_reason)
        detail = f" ({trace.error})" if trace.error else ""
        typer.echo(f"Ret gerekçesi: {reason}{detail}")
    timings = " · ".join(
        f"{label} {answer.gecikme_ms[stage]:.0f}"
        for stage, label in _LATENCY_LABELS.items()
        if stage in answer.gecikme_ms
    )
    typer.echo(f"Süreler (ms): {timings} · LLM çağrısı: {answer.llm_cagri_sayisi}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Sorulacak soru.")],
    config: ConfigOption = DEFAULT_CONFIG,
    program: Annotated[
        str | None,
        typer.Option("--program", help="Yalnız bu programın kaynaklarında ara (ör. 1507)."),
    ] = None,
    tur: Annotated[
        DocType | None, typer.Option("--tur", help="Yalnız bu belge türünde ara.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Cevabı JSON olarak yazdır (Answer modeli).")
    ] = False,
    with_trace: Annotated[
        bool,
        typer.Option("--trace", help='Cevabı ve aşama izini JSON yazdır: {"cevap", "iz"}.'),
    ] = False,
) -> None:
    """Tek bir soruyu cevaplar ve kaynakları gösterir."""
    cfg = _load(config)
    _tolerant_stdout()
    filters = None
    if program is not None or tur is not None:
        filters = MetadataFilter(program_kodu=program, doc_turu=tur.value if tur else None)
    trace = AskTrace()
    owner = f"rag ask ({cfg.experiment.name}, {cfg.llm.model}) pid={os.getpid()}"
    try:
        with _gpu_guard(cfg, owner):
            pipeline = build_pipeline(cfg)
            answer = pipeline.ask(question, filters, trace=trace)
    except GpuLockError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    if with_trace:
        payload = {"cevap": answer.model_dump(mode="json"), "iz": trace.model_dump(mode="json")}
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
        return
    _print_answer(answer, trace, cfg.guardrails.thresholds)


class WebBackendKind(StrEnum):
    """Backends selectable with ``rag serve --backend``."""

    PIPELINE = "pipeline"
    FAKE = "fake"


@app.command()
def serve(
    config: ConfigOption = DEFAULT_CONFIG,
    host: Annotated[str, typer.Option("--host", help="Dinlenecek adres.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Port.")] = 8000,
    backend: Annotated[
        WebBackendKind,
        typer.Option("--backend", help="pipeline: gerçek sistem; fake: gösterim verisi."),
    ] = WebBackendKind.PIPELINE,
) -> None:
    """Web arayüzünü ve REST API'yi başlatır (OpenAPI belgeleri: /docs)."""
    _load(config)  # an invalid config stops here with exit code 2
    try:
        import uvicorn

        from tubitak_rag.interfaces.web.app import create_app
        from tubitak_rag.interfaces.web.state import build_backend
    except ImportError as exc:
        typer.echo(f'Web arayüzü için web eki gerekli: pip install -e ".[web]" ({exc})', err=True)
        raise typer.Exit(code=2) from exc
    web_app = create_app(backend=build_backend(backend.value, config))
    base = f"http://{host}:{port}"
    typer.echo(f"Web arayüzü: {base} · API belgeleri: {base}/docs · arka uç: {backend.value}")
    uvicorn.run(web_app, host=host, port=port)


app.command("export-index")(export_index)
app.command("export-space")(export_space)


if __name__ == "__main__":  # ``python -m tubitak_rag.interfaces.cli`` (web background jobs)
    app()
