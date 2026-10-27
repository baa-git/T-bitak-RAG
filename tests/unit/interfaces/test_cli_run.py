"""Tests for ``rag run`` and ``rag report`` (Task 8.8; FR-CLI1, FR-X1, FR-X2, FR-X8)."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tests.builders import make_chunk, make_scored
from tests.eval_builders import QUOTE, make_answer, make_question, make_unanswerable
from tests.fakes.llm import FakeLLM
from tests.fakes.pipeline import FakePipeline, Scripted
from tubitak_rag.config import AppConfig, ConfigError, LlmConfig
from tubitak_rag.domain.answer import Answer
from tubitak_rag.domain.enums import AnswerStatus, QuestionType
from tubitak_rag.domain.evaluation import QuestionResult, RunMetrics
from tubitak_rag.evaluation.runner import QuestionDone
from tubitak_rag.evaluation.testset import save_testset
from tubitak_rag.guardrails.gate import GateResult
from tubitak_rag.interfaces import cli
from tubitak_rag.interfaces.cli import RunProgressPrinter, app

runner = CliRunner()
VERDICT = json.dumps({"dogru": True, "sadik": True, "gerekce": "Doğru."})
PASSED = GateResult(passed=True, key="dense", score=0.8)
QUESTIONS = [
    make_question("q001"),
    make_question("q002", tur=QuestionType.COK_BELGE),
    make_unanswerable("q003"),
]


def _script() -> dict[str, Scripted]:
    hit = [make_scored(make_chunk(text=f"Madde 7. {QUOTE}."), 0.8, dense=0.8)]
    refused = make_answer(durum=AnswerStatus.BILGI_BULUNAMADI, llm_cagri_sayisi=0)
    return {
        QUESTIONS[0].soru: Scripted(make_answer(QUESTIONS[0].soru), candidates=hit, gate=PASSED),
        QUESTIONS[1].soru: Scripted(make_answer(QUESTIONS[1].soru), candidates=hit, gate=PASSED),
        QUESTIONS[2].soru: Scripted(
            refused, candidates=hit, gate=GateResult(passed=False, key="dense", score=0.3)
        ),
    }


class Env:
    """A temporary project: experiment YAML, test set and patched pipeline/judge/ingest."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp_path
        self.artifacts = tmp_path / "artifacts"
        self.runs = self.artifacts / "runs"
        self.testset = tmp_path / "testset.jsonl"
        save_testset(self.testset, QUESTIONS)
        self.pipeline = FakePipeline(_script())
        self.built: list[AppConfig] = []
        self.judge_cfgs: list[LlmConfig] = []
        self.events: list[str] = []
        self.judge_replies = [VERDICT] * 10

        def build_pipeline(cfg: AppConfig) -> FakePipeline:
            self.events.append("pipeline")
            self.built.append(cfg)
            return self.pipeline

        def build_llm(llm_cfg: LlmConfig) -> FakeLLM:
            self.judge_cfgs.append(llm_cfg)
            return FakeLLM(self.judge_replies, model_name=f"{llm_cfg.provider}:{llm_cfg.model}")

        def run_ingest(cfg: AppConfig, **kwargs: Any) -> Any:
            self.events.append("ingest")
            raise AssertionError("ingest disabled in this test")

        monkeypatch.setattr(cli, "build_pipeline", build_pipeline)
        monkeypatch.setattr(cli, "build_llm", build_llm)
        monkeypatch.setattr(cli, "run_ingest", run_ingest)

    def config(self, name: str = "v1", **sections: Any) -> Path:
        data: dict[str, Any] = {
            "experiment": {"name": name, "concept": "structural_context", "previous": "v0"},
            "paths": {
                "artifacts_dir": self.artifacts.as_posix(),
                "testset": self.testset.as_posix(),
            },
            "embedder": {"provider": "fake"},
        }
        for key, value in sections.items():
            data[key] = {**data.get(key, {}), **value}
        path = self.tmp / f"{name}.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return path

    def run(self, *args: str, config: Path | None = None) -> Any:
        return runner.invoke(app, ["run", str(config or self.config()), "--no-ingest", *args])

    def run_dirs(self) -> list[Path]:
        return sorted(d for d in self.runs.iterdir() if d.is_dir()) if self.runs.is_dir() else []


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def _rows(run_dir: Path) -> list[QuestionResult]:
    lines = (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
    return [QuestionResult.model_validate_json(line) for line in lines]


def test_run_cli_executes_and_writes_report(env: Env) -> None:
    result = env.run()

    assert result.exit_code == 0, result.output
    (run_dir,) = env.run_dirs()
    for name in ("config.yaml", "metrics.json", "per_question.jsonl", "report.md"):
        assert (run_dir / name).is_file(), name
    assert (env.runs / "ozet.md").is_file()
    assert env.pipeline.calls == [q.soru for q in QUESTIONS]  # sequential, in order
    metrics = RunMetrics.model_validate_json((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics.judge_model == "ollama:qwen3:4b"
    out = result.output
    assert "[1/3] q001 cevaplandi" in out and "[3/3] q003 bilgi_bulunamadi (esik_alti)" in out
    assert "kalan ~" in out
    assert "Recall@5" in out and "Cevap doğruluğu" in out
    assert str(run_dir / "report.md") in out and "ozet.md" in out
    assert "# Deney raporu: v1" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_run_cli_judge_is_local_and_deterministic(env: Env) -> None:
    config = env.config(llm={"temperature": 0.7}, eval={"judge": {"model": "qwen3:8b"}})

    result = env.run(config=config)

    assert result.exit_code == 0, result.output
    assert env.built[0].llm.temperature == 0.0  # NFR-REPRO2: forced for evaluation runs
    (judge_cfg,) = env.judge_cfgs
    assert (judge_cfg.provider, judge_cfg.model, judge_cfg.temperature) == ("ollama", "qwen3:8b", 0)


def test_run_cli_no_judge(env: Env) -> None:
    result = env.run("--no-judge")

    assert result.exit_code == 0, result.output
    assert env.judge_cfgs == []
    (run_dir,) = env.run_dirs()
    assert all(r.judge["dogru"] is None for r in _rows(run_dir))


@pytest.mark.parametrize(
    "sections",
    [{"llm": {"provider": "anthropic"}}, {"eval": {"judge": {"provider": "anthropic"}}}],
)
def test_run_cli_warns_when_an_api_provider_is_configured(
    env: Env, sections: dict[str, Any]
) -> None:
    result = env.run(config=env.config(**sections))

    assert result.exit_code == 0, result.output
    assert "anthropic" in result.output and "ücretlendirilebilir" in result.output


def test_run_cli_warns_for_a_hosted_openai_compatible_endpoint(env: Env) -> None:
    llm = {"provider": "openai_compat", "openai_compat": {"base_url": "https://api.example.com/v1"}}

    result = env.run(config=env.config(llm=llm))

    assert result.exit_code == 0, result.output
    assert "openai_compat" in result.output and "ücretlendirilebilir" in result.output


def test_run_cli_does_not_warn_for_a_local_openai_compatible_endpoint(env: Env) -> None:
    llm = {"provider": "openai_compat", "openai_compat": {"base_url": "http://localhost:11434/v1"}}

    result = env.run(config=env.config(llm=llm))

    assert result.exit_code == 0, result.output
    assert "Uyarı" not in result.output


def test_run_cli_limit_and_ids(env: Env) -> None:
    assert env.run("--limit", "1").exit_code == 0
    assert env.pipeline.calls == [QUESTIONS[0].soru]

    result = env.run("--ids", "q003,q002")
    assert result.exit_code == 0, result.output
    assert env.pipeline.calls[1:] == [QUESTIONS[1].soru, QUESTIONS[2].soru]  # test-set order

    bad = env.run("--ids", "q002,q999")
    assert bad.exit_code == 2 and "q999" in bad.output


def test_run_cli_holds_gpu_lock_for_the_whole_run(env: Env) -> None:
    lock = env.artifacts / ".gpu.lock"
    held: list[bool] = []
    original = env.pipeline.ask

    def ask(question: str, filters: Any = None, *, trace: Any = None) -> Answer:
        held.append(lock.exists())
        return original(question, filters, trace=trace)

    env.pipeline.ask = ask  # type: ignore[method-assign]
    config = env.config(embedder={"provider": "sentence_transformers"})

    result = env.run(config=config)

    assert result.exit_code == 0, result.output
    assert held == [True, True, True] and not lock.exists()


def test_run_cli_releases_gpu_lock_when_interrupted_and_explains_resume(env: Env) -> None:
    lock = env.artifacts / ".gpu.lock"
    original = env.pipeline.ask

    def ask(question: str, filters: Any = None, *, trace: Any = None) -> Answer:
        if question == QUESTIONS[2].soru:
            raise KeyboardInterrupt
        return original(question, filters, trace=trace)

    env.pipeline.ask = ask  # type: ignore[method-assign]
    config = env.config(embedder={"provider": "sentence_transformers"})

    result = env.run(config=config)

    assert result.exit_code == 130
    assert not lock.exists()
    (run_dir,) = env.run_dirs()
    assert [r.id for r in _rows(run_dir)] == ["q001", "q002"]
    assert f"--resume {run_dir.name}" in result.output


def test_run_cli_refuses_to_start_while_gpu_lock_is_held(env: Env) -> None:
    lock = env.artifacts / ".gpu.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("ingest işi\n", encoding="utf-8")
    config = env.config(embedder={"provider": "sentence_transformers"})

    result = env.run(config=config)

    assert result.exit_code == 1 and "GPU kilidi dolu" in result.output
    assert env.pipeline.calls == [] and lock.exists() and env.run_dirs() == []


def test_run_cli_resume(env: Env) -> None:
    assert env.run("--limit", "2").exit_code == 0
    (run_dir,) = env.run_dirs()
    (run_dir / "metrics.json").unlink()

    result = env.run("--resume", run_dir.name)

    assert result.exit_code == 0, result.output
    assert env.run_dirs() == [run_dir]
    assert env.pipeline.calls[2:] == [QUESTIONS[2].soru]  # q001, q002 were not asked again
    assert [r.id for r in _rows(run_dir)] == ["q001", "q002", "q003"]
    assert "devam: 2 soru önceki oturumdan" in result.output


def test_run_cli_resume_errors(env: Env) -> None:
    missing = env.run("--resume", "v1_20000101-000000")
    assert missing.exit_code == 2 and "bulunamadı" in missing.output

    assert env.run("--limit", "1").exit_code == 0
    (run_dir,) = env.run_dirs()
    changed = env.config(retriever={"top_n": 3})
    result = env.run("--resume", run_dir.name, config=changed)
    assert result.exit_code == 2 and "yapılandırması değişmiş" in result.output


def test_run_cli_ingests_before_building_the_pipeline(
    env: Env, monkeypatch: pytest.MonkeyPatch, fake_components: None
) -> None:
    def run_ingest(cfg: AppConfig, **kwargs: Any) -> Any:
        env.events.append("ingest")
        assert kwargs["full"] is False
        raise _StopError

    class _StopError(Exception):
        pass

    monkeypatch.setattr(cli, "run_ingest", run_ingest)
    config = env.config(
        vector_store={"provider": "fake"}, bm25={"provider": "fake"}, chunker={"tokenizer": "regex"}
    )

    with pytest.raises(_StopError):
        runner.invoke(app, ["run", str(config)], catch_exceptions=False)

    assert env.events == ["ingest"]  # the pipeline is built only after the ingest


def test_run_cli_missing_index_exits_with_code_2(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def build(cfg: AppConfig) -> Any:
        raise ConfigError("BM25 indeksi bulunamadı. Önce `rag ingest` çalıştırın.")

    monkeypatch.setattr(cli, "build_pipeline", build)

    result = env.run()

    assert result.exit_code == 2 and "rag ingest" in result.output


def test_report_cli_single_all_and_summary(env: Env) -> None:
    assert env.run().exit_code == 0
    assert env.run(config=env.config("v0")).exit_code == 0
    v0, v1 = sorted(env.run_dirs(), key=lambda d: d.name)
    for run_dir in (v0, v1):
        (run_dir / "report.md").unlink()
    (env.runs / "ozet.md").unlink()
    base = ["--config", str(env.config())]

    single = runner.invoke(app, ["report", v1.name, *base])
    assert single.exit_code == 0, single.output
    assert (v1 / "report.md").is_file() and not (v0 / "report.md").exists()
    assert "| Metrik | Önceki (v0) | Bu koşu | Fark |" in (v1 / "report.md").read_text("utf-8")
    assert (env.runs / "ozet.md").is_file()

    every = runner.invoke(app, ["report", "--all", *base])
    assert every.exit_code == 0, every.output
    assert (v0 / "report.md").is_file()

    (env.runs / "ozet.md").unlink()
    summary = runner.invoke(app, ["report", "--summary", *base])
    assert summary.exit_code == 0 and (env.runs / "ozet.md").is_file()
    assert "| v0 |" in (env.runs / "ozet.md").read_text(encoding="utf-8")


def test_report_cli_errors(env: Env) -> None:
    base = ["--config", str(env.config())]
    assert runner.invoke(app, ["report", *base]).exit_code == 2
    unknown = runner.invoke(app, ["report", "v9_20000101-000000", *base])
    assert unknown.exit_code == 2 and "bulunamadı" in unknown.output


def test_progress_printer_eta_ignores_resumed_questions() -> None:
    times = iter([100.0, 110.0, 130.0])
    lines: list[str] = []
    printer = RunProgressPrinter(clock=lambda: next(times), echo=lines.append)
    row = QuestionResult.model_validate_json(
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("fixtures", "run_example", "per_question.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    printer(QuestionDone("v1_x", 51, 100, 50, row))
    printer(QuestionDone("v1_x", 52, 100, 50, row))
    assert lines[0] == "devam: 50 soru önceki oturumdan alındı"
    assert lines[1] == "[51/100] q001 cevaplandi · 10,0 s/soru · kalan ~8 dk 10 sn"
    assert lines[2].endswith("15,0 s/soru · kalan ~12 dk 0 sn")
