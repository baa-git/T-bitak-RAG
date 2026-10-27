"""Tests for the Typer command-line skeleton (FR-CLI1, NFR-CONF1)."""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from tests.fakes.parser import FakeParser
from tests.fixtures.corpus import CorpusOnDisk
from tubitak_rag.config import AppConfig, ConfigError
from tubitak_rag.domain.answer import Answer, Citation, SourceView
from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.domain.retrieval import MetadataFilter
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.generation.prompts import KAPSAM_DISI_MESAJI
from tubitak_rag.guardrails.gate import GateResult
from tubitak_rag.ingestion import parsed_cache
from tubitak_rag.ingestion.fetch import FetchReport, FetchResult, FetchStatus
from tubitak_rag.ingestion.ingest import IngestReport, LazyEmbedder
from tubitak_rag.interfaces import cli
from tubitak_rag.interfaces.cli import app
from tubitak_rag.pipeline import REFUSAL_REASONS

runner = CliRunner()
COMMANDS = ["fetch", "ingest", "run", "report", "ask", "serve"]


def test_help_lists_six_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


def test_invalid_config_path_exits_with_code_2() -> None:
    result = runner.invoke(app, ["ask", "x", "--config", "yok.yaml"])
    assert result.exit_code == 2
    assert "Yapılandırma dosyası bulunamadı" in result.output


def test_invalid_config_field_exits_with_code_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm:\n  modelx: x\n", encoding="utf-8")
    result = runner.invoke(app, ["ingest", "--config", str(bad)])
    assert result.exit_code == 2
    assert "llm.modelx" in result.output


def _fetch_config(tmp_path: Path, fixtures: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        "paths:\n"
        f"  raw_dir: {(tmp_path / 'raw').as_posix()}\n"
        f"  manifests_dir: {(fixtures / 'manifests').as_posix()}\n"
        f"  artifacts_dir: {(tmp_path / 'artifacts').as_posix()}\n",
        encoding="utf-8",
    )
    return config


def _fake_fetch(seen: list[SourceEntry], mismatch: bool) -> Any:
    def fake(entries: Sequence[SourceEntry], raw_dir: Path, client: Any, **kw: Any) -> FetchReport:
        seen.extend(entries)
        results = [
            FetchResult(
                doc_id=e.doc_id,
                local_path=e.local_path,
                status=FetchStatus.UYUSMAZLIK if mismatch and i == 0 else FetchStatus.ATLANDI,
                expected_sha256=e.sha256,
                actual_sha256="b" * 64,
            )
            for i, e in enumerate(entries)
        ]
        return FetchReport(results=results)

    return fake


def test_fetch_cli_writes_report_and_exit_code(
    tmp_path: Path, fixtures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[SourceEntry] = []
    monkeypatch.setattr(cli, "fetch_all", _fake_fetch(seen, mismatch=True))
    result = runner.invoke(app, ["fetch", "--config", str(_fetch_config(tmp_path, fixtures))])
    assert result.exit_code == 1
    assert "kaynakta_guncellenmis" in result.output and "mevzuat-030" in result.output
    report = json.loads((tmp_path / "artifacts" / "fetch_report.json").read_text("utf-8"))
    assert report["counts"]["kaynakta_guncellenmis"] == 1
    assert len(seen) == 8


def test_fetch_cli_only_filters_group_and_exits_0_when_clean(
    tmp_path: Path, fixtures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[SourceEntry] = []
    monkeypatch.setattr(cli, "fetch_all", _fake_fetch(seen, mismatch=False))
    config = str(_fetch_config(tmp_path, fixtures))
    result = runner.invoke(app, ["fetch", "--config", config, "--only", "sanayi"])
    assert result.exit_code == 0, result.output
    assert {e.kaynak_grubu for e in seen} == {"sanayi_mevzuati"}
    assert "atlandi" in result.output


def test_fetch_cli_rejects_unknown_group(tmp_path: Path, fixtures: Path) -> None:
    config = str(_fetch_config(tmp_path, fixtures))
    result = runner.invoke(app, ["fetch", "--config", config, "--only", "yok"])
    assert result.exit_code == 2


# --- rag ingest ---------------------------------------------------------------------------


def _ingest_report(**overrides: Any) -> IngestReport:
    fields: dict[str, Any] = {
        "collection": "structural_ctx__google-embeddinggemma-2",
        "chunker": "structural_ctx",
        "embedder": "google/embeddinggemma-2",
        "full": False,
        "docs_total": 10,
        "added": ["mevzuat-001", "mevzuat-002"],
        "updated": ["mevzuat-003"],
        "removed": ["mevzuat-004"],
        "unchanged": 4,
        "scanned": ["mevzuat-005"],
        "empty": [],
        "missing": ["mevzuat-006"],
        "errors": [{"doc_id": "mevzuat-007", "message": "bozuk dosya"}],
        "chunks_total": 120,
        "chunks_indexed": 30,
        "parsed": 1,
        "parse_cached": 8,
        "chunked": 1,
        "chunk_cached": 7,
        "embedded": 25,
        "embed_cached": 5,
        "bm25_size": 120,
        "stage_times_s": {
            "parse": 1.0,
            "chunk": 0.5,
            "embed_index": 3.0,
            "bm25": 0.2,
            "total": 4.7,
        },
    }
    fields.update(overrides)
    return IngestReport(**fields)


class _IngestSpy:
    """Stands in for ``run_ingest``; records its arguments and checks the GPU lock."""

    def __init__(self, lock: Path, report: IngestReport | None = None) -> None:
        self.lock = lock
        self.report = report or _ingest_report()
        self.calls: list[dict[str, Any]] = []
        self.lock_held: list[bool] = []

    def __call__(self, cfg: AppConfig, **kwargs: Any) -> IngestReport:
        self.calls.append({"cfg": cfg, **kwargs})
        self.lock_held.append(self.lock.exists())
        return self.report


@pytest.fixture
def ingest_corpus(tmp_path: Path, fake_components: None) -> CorpusOnDisk:
    return CorpusOnDisk(tmp_path)


def _spy(monkeypatch: pytest.MonkeyPatch, corpus: CorpusOnDisk) -> _IngestSpy:
    spy = _IngestSpy(corpus.artifacts / ".gpu.lock")
    monkeypatch.setattr(cli, "run_ingest", spy)
    return spy


def test_ingest_cli_prints_summary(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    _spy(monkeypatch, ingest_corpus)

    result = runner.invoke(app, ["ingest", "--config", str(ingest_corpus.config_file())])

    assert result.exit_code == 0, result.output
    out = result.output
    assert "structural_ctx__google-embeddinggemma-2" in out
    for label, count in [("eklendi", 2), ("güncellendi", 1), ("silindi", 1), ("değişmedi", 4)]:
        assert re.search(rf"{label}\s+{count}\b", out), label
    assert re.search(r"taranmış\s+1\b", out) and "mevzuat-005" in out
    assert "mevzuat-006" in out and "mevzuat-007: bozuk dosya" in out
    assert "25 embed edildi" in out and "5 önbellekten" in out
    assert "ingest_report.json" in out


def test_ingest_cli_end_to_end_with_fake_components(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parsed_cache, "parser_for", lambda entry: FakeParser())
    ingest_corpus.add("001_a.pdf", "Destek oranı yüzde yetmiş beştir.")
    ingest_corpus.add("002_b.pdf", "TARANMIS")

    result = runner.invoke(app, ["ingest", "--config", str(ingest_corpus.config_file())])

    assert result.exit_code == 0, result.output
    assert re.search(r"eklendi\s+1\b", result.output) and "structural_ctx__fake" in result.output
    saved = json.loads((ingest_corpus.artifacts / "ingest_report.json").read_text("utf-8"))
    assert saved["added"] == ["mevzuat-001"] and saved["scanned"] == ["mevzuat-002"]


def test_ingest_cli_holds_gpu_lock_for_a_real_embedder(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _spy(monkeypatch, ingest_corpus)
    config = ingest_corpus.config_file(embedder={"provider": "sentence_transformers"})

    result = runner.invoke(app, ["ingest", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert spy.lock_held == [True] and not spy.lock.exists()
    assert isinstance(spy.calls[0]["embedder"], LazyEmbedder)  # model not loaded up front


def test_ingest_cli_refuses_to_start_while_gpu_lock_is_held(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _spy(monkeypatch, ingest_corpus)
    config = ingest_corpus.config_file(embedder={"provider": "ollama"})
    spy.lock.parent.mkdir(parents=True)
    spy.lock.write_text("deney koşusu v3\n", encoding="utf-8")

    result = runner.invoke(app, ["ingest", "--config", str(config)])

    assert result.exit_code == 1
    assert "GPU kilidi dolu" in result.output and "deney koşusu v3" in result.output
    assert spy.calls == [] and spy.lock.exists()


def test_ingest_cli_fake_embedder_needs_no_gpu_lock(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _spy(monkeypatch, ingest_corpus)
    config = ingest_corpus.config_file()
    spy.lock.parent.mkdir(parents=True)
    spy.lock.write_text("başka iş\n", encoding="utf-8")

    result = runner.invoke(app, ["ingest", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert len(spy.calls) == 1


def test_ingest_cli_batch_size_defaults_low_and_passes_full(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _spy(monkeypatch, ingest_corpus)
    config = str(ingest_corpus.config_file(embedder={"batch_size": 32}))

    runner.invoke(app, ["ingest", "--config", config])
    runner.invoke(app, ["ingest", "--config", config, "--batch-size", "8", "--full"])

    assert [c["cfg"].embedder.batch_size for c in spy.calls] == [cli.INGEST_BATCH_SIZE, 8]
    assert cli.INGEST_BATCH_SIZE <= 16
    assert [c["full"] for c in spy.calls] == [False, True]


def test_ingest_cli_rejects_batch_size_above_gpu_rule(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _spy(monkeypatch, ingest_corpus)
    config = str(ingest_corpus.config_file())

    result = runner.invoke(app, ["ingest", "--config", config, "--batch-size", "64"])

    assert result.exit_code == 2 and spy.calls == []


def test_progress_printer_throttles_and_shows_rate(capsys: pytest.CaptureFixture[str]) -> None:
    now = [0.0]
    printer = cli.ProgressPrinter(interval_s=5.0, clock=lambda: now[0])

    printer("parse", 1, 3)  # inside the interval: silent
    printer("parse", 3, 3)  # stage finished: always printed
    printer("embed", 0, 100)
    now[0] = 1.0
    printer("embed", 10, 100)  # inside the interval: silent
    now[0] = 10.0
    printer("embed", 50, 100)
    now[0] = 20.0
    printer("embed", 100, 100)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert "ayrıştırma" in lines[0] and "3/3" in lines[0]
    assert "50/100" in lines[1] and "5 parça/s" in lines[1] and "kalan" in lines[1]
    assert "100/100" in lines[2]


def test_ingest_cli_output_fits_the_turkish_windows_codepage(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Piped output (``| Tee-Object``) is encoded with cp1254 on Turkish Windows."""
    _spy(monkeypatch, ingest_corpus)
    command = typer.main.get_command(app).commands["ingest"]  # type: ignore[attr-defined]
    texts = [command.help or "", *(getattr(p, "help", None) or "" for p in command.params)]

    result = runner.invoke(app, ["ingest", "--config", str(ingest_corpus.config_file())])

    for text in [*texts, result.output]:
        text.encode("cp1254")  # raises UnicodeEncodeError on e.g. "≤"


# --- rag ask (Task 6.9) ----------------------------------------------------------------------

_ASK_DISCLAIMER = "Bu cevap bilgilendirme amaçlıdır; resmî görüş değildir."


def _answered() -> Answer:
    return Answer(
        soru="1507 programında destek oranı en fazla yüzde kaçtır?",
        durum=AnswerStatus.CEVAPLANDI,
        cevaplanabilir=True,
        cevap="Destek oranı her dönem için en fazla %75'tir [1].",
        atiflar=[
            Citation(
                no=1,
                chunk_id="mevzuat-031:0007:00",
                doc_id="mevzuat-031",
                doc_title="1507 Uygulama Esasları",
                konum="Madde 7 (1)",
                kaynak_url="https://tubitak.gov.tr/1507.pdf",
            )
        ],
        kaynaklar=[
            SourceView(
                no=1,
                chunk_id="mevzuat-031:0007:00",
                doc_title="1507 Uygulama Esasları",
                konum="Madde 7 (1)",
                kaynak_url="https://tubitak.gov.tr/1507.pdf",
                text="... en fazla %75 ...",
                score=3.2,
            )
        ],
        uyari=_ASK_DISCLAIMER,
        gecikme_ms={"scope": 1.2, "retrieval": 63.4, "rerank": 241.7, "llm": 2812.5, "total": 3120},
        llm_cagri_sayisi=1,
        gate_key="rerank",
        gate_score=3.2,
    )


def _refused() -> Answer:
    return _answered().model_copy(
        update={
            "durum": AnswerStatus.BILGI_BULUNAMADI,
            "cevaplanabilir": False,
            "cevap": "Elimdeki mevzuat kaynaklarında bu soruya dayanak bulamadım.",
            "atiflar": [],
            "kaynaklar": [],
            "gecikme_ms": {"scope": 1.0, "retrieval": 50.0, "rerank": 200.0, "total": 260.0},
            "llm_cagri_sayisi": 0,
            "gate_score": -4.5,
        }
    )


class _FakePipeline:
    """Stands in for ``RagPipeline``; fills the trace like the real one."""

    def __init__(self, answer: Answer, reason: str | None = None) -> None:
        self.answer = answer
        self.reason = reason
        self.calls: list[tuple[str, MetadataFilter | None]] = []

    def ask(
        self, question: str, filters: MetadataFilter | None = None, *, trace: Any = None
    ) -> Answer:
        self.calls.append((question, filters))
        if trace is not None:
            trace.rejection_reason = self.reason
            if self.answer.gate_key:
                passed = self.answer.durum is AnswerStatus.CEVAPLANDI
                trace.gate = GateResult(
                    passed=passed, key=self.answer.gate_key, score=self.answer.gate_score
                )
        return self.answer


def _ask(
    corpus: CorpusOnDisk,
    monkeypatch: pytest.MonkeyPatch,
    answer: Answer,
    *args: str,
    reason: str | None = None,
    config: Path | None = None,
) -> tuple[Any, _FakePipeline]:
    pipeline = _FakePipeline(answer, reason)
    monkeypatch.setattr(cli, "build_pipeline", lambda cfg: pipeline)
    config = config or corpus.config_file()
    result = runner.invoke(app, ["ask", *args, "--config", str(config)])
    return result, pipeline


def test_ask_cli_prints_answer_and_sources(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, pipeline = _ask(ingest_corpus, monkeypatch, _answered(), "1507 destek oranı?")

    assert result.exit_code == 0, result.output
    out = result.output
    assert pipeline.calls == [("1507 destek oranı?", None)]
    assert "Destek oranı her dönem için en fazla %75'tir [1]." in out
    assert "[1] 1507 Uygulama Esasları — Madde 7 (1)" in out
    assert "https://tubitak.gov.tr/1507.pdf" in out
    assert out.count(_ASK_DISCLAIMER) == 1
    assert "Durum: cevaplandı" in out
    assert re.search(r"rerank skoru 3[.,]20 \(eşik -2[.,]00", out)
    assert "arama 63" in out and "LLM 2812" in out and "toplam 3120" in out
    assert "LLM çağrısı: 1" in out


def test_ask_cli_json_prints_the_answer_model(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = _answered()

    result, _ = _ask(ingest_corpus, monkeypatch, answer, "1507 destek oranı?", "--json")

    assert result.exit_code == 0, result.output
    assert Answer.model_validate_json(result.output) == answer


def test_ask_cli_program_and_type_become_filters(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = ["destek oranı?", "--program", "1507", "--tur", "uygulama_esaslari"]

    result, pipeline = _ask(ingest_corpus, monkeypatch, _answered(), *args)

    assert result.exit_code == 0, result.output
    expected = MetadataFilter(program_kodu="1507", doc_turu="uygulama_esaslari")
    assert pipeline.calls == [("destek oranı?", expected)]


def test_ask_cli_explains_a_refusal(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _ask(
        ingest_corpus, monkeypatch, _refused(), "TÜBİTAK'ın 2030 bütçesi?", reason="esik_alti"
    )

    assert result.exit_code == 0, result.output
    out = result.output
    assert "Durum: bilgi bulunamadı" in out
    assert "Atıflar" not in out
    assert f"Ret gerekçesi: {REFUSAL_REASONS['esik_alti']}" in out
    assert "LLM çağrısı: 0" in out


def test_ask_cli_missing_index_exits_with_code_2(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build(cfg: AppConfig) -> Any:
        raise ConfigError("BM25 indeksi bulunamadı. Önce `rag ingest` çalıştırın.")

    monkeypatch.setattr(cli, "build_pipeline", build)
    config = ingest_corpus.config_file()

    result = runner.invoke(app, ["ask", "1507 destek oranı?", "--config", str(config)])

    assert result.exit_code == 2
    assert "rag ingest" in result.output


def test_ask_cli_holds_gpu_lock_for_real_models(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = ingest_corpus.artifacts / ".gpu.lock"
    held: list[bool] = []

    class _Spy(_FakePipeline):
        def ask(self, question: str, filters: Any = None, *, trace: Any = None) -> Answer:
            held.append(lock.exists())
            return super().ask(question, filters, trace=trace)

    monkeypatch.setattr(cli, "build_pipeline", lambda cfg: _Spy(_answered()))
    config = ingest_corpus.config_file(embedder={"provider": "sentence_transformers"})

    result = runner.invoke(app, ["ask", "1507 destek oranı?", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert held == [True] and not lock.exists()


def test_ask_cli_refuses_to_start_while_gpu_lock_is_held(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = ingest_corpus.artifacts / ".gpu.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("embedding işi\n", encoding="utf-8")
    config = ingest_corpus.config_file(embedder={"provider": "sentence_transformers"})

    result, pipeline = _ask(
        ingest_corpus, monkeypatch, _answered(), "1507 destek oranı?", config=config
    )

    assert result.exit_code == 1
    assert "GPU kilidi dolu" in result.output
    assert pipeline.calls == [] and lock.exists()


def test_ask_cli_end_to_end_with_fake_components(ingest_corpus: CorpusOnDisk) -> None:
    config = ingest_corpus.config_file(
        retriever={"mode": "dense"},
        reranker={"enabled": False},
        llm={"provider": "fake"},
        guardrails={"scope_mode": "lexicon"},
    )

    result = runner.invoke(app, ["ask", "İstanbul'da yarın hava nasıl?", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert KAPSAM_DISI_MESAJI in result.output
    assert REFUSAL_REASONS["kapsam_disi"] in result.output


def test_ask_cli_output_fits_the_turkish_windows_codepage(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = typer.main.get_command(app).commands["ask"]  # type: ignore[attr-defined]
    texts = [command.help or "", *(getattr(p, "help", None) or "" for p in command.params)]
    outputs = [
        _ask(ingest_corpus, monkeypatch, _answered(), "1507 destek oranı?")[0].output,
        _ask(ingest_corpus, monkeypatch, _refused(), "2030?", reason="esik_alti")[0].output,
    ]

    for text in [*texts, *outputs]:
        text.encode("cp1254")


# --- rag ask: R2 review --------------------------------------------------------------------


def _lock_seen_by_ask(corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch, config: Path) -> bool:
    lock = corpus.artifacts / ".gpu.lock"
    held: list[bool] = []

    class _Spy(_FakePipeline):
        def ask(self, question: str, filters: Any = None, *, trace: Any = None) -> Answer:
            held.append(lock.exists())
            return super().ask(question, filters, trace=trace)

    monkeypatch.setattr(cli, "build_pipeline", lambda cfg: _Spy(_answered()))
    result = runner.invoke(app, ["ask", "1507 destek oranı?", "--config", str(config)])
    assert result.exit_code == 0, result.output
    return held == [True]


def test_ask_cli_holds_gpu_lock_when_only_the_llm_is_real(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fake embedder, no reranker, but a real (Ollama) LLM still needs the GPU.
    config = ingest_corpus.config_file(reranker={"enabled": False})

    assert _lock_seen_by_ask(ingest_corpus, monkeypatch, config)


def test_ask_cli_holds_gpu_lock_when_only_the_reranker_is_real(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ingest_corpus.config_file(llm={"provider": "fake"})

    assert _lock_seen_by_ask(ingest_corpus, monkeypatch, config)


def test_ask_cli_skips_gpu_lock_when_every_model_is_fake(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ingest_corpus.config_file(llm={"provider": "fake"}, reranker={"provider": "fake"})

    assert not _lock_seen_by_ask(ingest_corpus, monkeypatch, config)


def test_ask_cli_shows_citation_repair(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = _answered().model_copy(update={"atif_onarimi": [1]})

    result, _ = _ask(ingest_corpus, monkeypatch, answer, "1507 destek oranı?")

    assert result.exit_code == 0, result.output
    assert "Atıf onarımı: [1] guardrail tarafından eklendi" in result.output
    assert "Atıf onarımı" not in _ask(ingest_corpus, monkeypatch, _answered(), "x?")[0].output


def test_ask_cli_trace_prints_answer_and_trace_as_json(
    ingest_corpus: CorpusOnDisk, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _ask(
        ingest_corpus, monkeypatch, _refused(), "2030 bütçesi?", "--trace", reason="esik_alti"
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert Answer.model_validate(data["cevap"]) == _refused()
    assert data["iz"]["rejection_reason"] == "esik_alti"
    assert data["iz"]["gate"]["key"] == "rerank"
    assert {"candidates", "reranked", "number_check", "atif_onarimi"} <= set(data["iz"])


# --- rag serve ------------------------------------------------------------------------------


def _capture_uvicorn(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_run(web_app: Any, host: str, port: int, **_: Any) -> None:
        calls.update(app=web_app, host=host, port=port)

    monkeypatch.setattr("uvicorn.run", fake_run)
    return calls


def test_serve_starts_uvicorn_with_the_lazy_pipeline_backend(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    from tubitak_rag.interfaces.web.pipeline_backend import PipelineBackend

    calls = _capture_uvicorn(monkeypatch)
    config = repo_root / "configs" / "default.yaml"
    result = runner.invoke(app, ["serve", "--config", str(config), "--port", "8123"])
    assert result.exit_code == 0, result.output
    backend = calls["app"].state.backend
    assert isinstance(backend, PipelineBackend) and backend.loaded_pipelines() == []
    assert (calls["host"], calls["port"]) == ("127.0.0.1", 8123)
    assert "http://127.0.0.1:8123/docs" in result.output


def test_serve_can_use_the_demo_backend(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    from tubitak_rag.interfaces.web.fake_backend import FakeBackend

    calls = _capture_uvicorn(monkeypatch)
    config = repo_root / "configs" / "default.yaml"
    result = runner.invoke(
        app, ["serve", "-c", str(config), "--backend", "fake", "--host", "0.0.0.0"]
    )
    assert result.exit_code == 0, result.output
    assert isinstance(calls["app"].state.backend, FakeBackend) and calls["host"] == "0.0.0.0"


def test_serve_with_invalid_config_exits_with_code_2(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_uvicorn(monkeypatch)
    result = runner.invoke(app, ["serve", "--config", "yok.yaml"])
    assert result.exit_code == 2 and calls == {}


def test_cli_module_runs_with_python_dash_m() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "tubitak_rag.interfaces.cli", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    assert result.returncode == 0 and "serve" in result.stdout
