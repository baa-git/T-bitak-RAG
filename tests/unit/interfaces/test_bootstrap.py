"""First-start preparation of the Docker stack (NFR-DOCKER2) with fake HTTP and commands."""

import json
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from tubitak_rag.config import (
    AppConfig,
    EmbedderConfig,
    EvalConfig,
    JudgeConfig,
    LlmConfig,
    PathsConfig,
    VectorStoreConfig,
)
from tubitak_rag.interfaces import bootstrap as bs

QDRANT = "http://qdrant:6333"
OLLAMA = "http://ollama:11434"
COLLECTION = "structural_ctx__google-embeddinggemma-2"


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(artifacts_dir=tmp_path / "artifacts"),
        vector_store=VectorStoreConfig(url=QDRANT),
        llm=LlmConfig(url=OLLAMA),
    )


class FakeServices:
    """Qdrant + Ollama REST endpoints used by the bootstrap."""

    def __init__(self, *, models: Sequence[str] = (), points: int | None = 0) -> None:
        self.models = list(models)
        self.points = points
        self.pulled: list[str] = []
        self.health_failures = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/healthz":
            if self.health_failures:
                self.health_failures -= 1
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, text="ok")
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if path == "/api/pull":
            model = json.loads(request.content)["model"]
            self.pulled.append(model)
            self.models.append(model)
            lines = [
                {"status": "pulling manifest"},
                {
                    "status": "pulling abc",
                    "digest": "abc",
                    "total": 100_000_000,
                    "completed": 50_000_000,
                },
                {
                    "status": "pulling abc",
                    "digest": "abc",
                    "total": 100_000_000,
                    "completed": 100_000_000,
                },
                {"status": "success"},
            ]
            return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))
        if path == f"/collections/{COLLECTION}":
            if self.points is None:
                return httpx.Response(404, json={"status": {"error": "not found"}})
            return httpx.Response(200, json={"result": {"points_count": self.points}})
        return httpx.Response(500)


class Commands:
    """Records CLI invocations; returns the configured exit code per subcommand."""

    def __init__(self, codes: dict[str, int] | None = None) -> None:
        self.codes = codes or {}
        self.calls: list[str] = []

    def __call__(self, argv: Sequence[str]) -> int:
        command = argv[argv.index("tubitak_rag.interfaces.cli") + 1]
        self.calls.append(command)
        return self.codes.get(command, 0)


def _run(
    tmp_path: Path,
    services: FakeServices,
    commands: Commands,
    environ: dict[str, str] | None = None,
    warmed: list[str] | None = None,
) -> None:
    client = httpx.Client(transport=httpx.MockTransport(services))
    bs.bootstrap(
        _cfg(tmp_path),
        Path("configs/default.yaml"),
        client,
        environ=environ or {},
        run=commands,
        warm=lambda cfg: (warmed if warmed is not None else []).append(cfg.embedder.model),
        sleep=lambda _s: None,
        restore=lambda *args, **kwargs: False,  # a ./release in the repo must not leak in
    )


def test_first_start_pulls_model_warms_and_ingests(tmp_path: Path) -> None:
    services, commands, warmed = FakeServices(points=None), Commands(), []
    _run(tmp_path, services, commands, warmed=warmed)
    assert services.pulled == ["qwen3:4b"]
    assert warmed == ["google/embeddinggemma-2"]
    assert commands.calls == ["fetch", "ingest"]


def test_second_start_skips_present_models_and_full_collection(tmp_path: Path) -> None:
    (tmp_path / "artifacts" / "cache" / "bm25" / "structural_ctx").mkdir(parents=True)
    services, commands = FakeServices(models=["qwen3:4b"], points=7017), Commands()
    _run(tmp_path, services, commands)
    assert services.pulled == [] and commands.calls == []


def test_missing_bm25_index_triggers_ingest(tmp_path: Path) -> None:
    services, commands = FakeServices(models=["qwen3:4b"], points=7017), Commands()
    _run(tmp_path, services, commands)
    assert commands.calls == ["fetch", "ingest"]


def test_failed_fetch_is_only_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    commands = Commands({"fetch": 1})
    _run(tmp_path, FakeServices(models=["qwen3:4b"]), commands)
    assert commands.calls == ["fetch", "ingest"]
    assert "rag fetch bazı belgeleri" in caplog.text


def test_failed_ingest_fails_bootstrap(tmp_path: Path) -> None:
    with pytest.raises(bs.BootstrapError, match="rag ingest"):
        _run(tmp_path, FakeServices(models=["qwen3:4b"]), Commands({"ingest": 2}))


def test_switches_skip_ingest_and_warmup(tmp_path: Path) -> None:
    commands, warmed = Commands(), []
    env = {bs.AUTO_INGEST_ENV: "0", bs.WARMUP_ENV: "hayır"}
    _run(tmp_path, FakeServices(models=["qwen3:4b"]), commands, environ=env, warmed=warmed)
    assert commands.calls == [] and warmed == []


def test_waits_until_qdrant_answers(tmp_path: Path) -> None:
    services = FakeServices(models=["qwen3:4b"], points=None)
    services.health_failures = 2
    _run(tmp_path, services, Commands(), environ={bs.AUTO_INGEST_ENV: "0"})
    assert services.health_failures == 0


def test_wait_for_times_out() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    ticks = iter(range(100))
    with pytest.raises(bs.BootstrapError, match="Qdrant 3 s"):
        bs.wait_for(
            "Qdrant",
            f"{QDRANT}/healthz",
            httpx.Client(transport=httpx.MockTransport(refuse)),
            timeout_s=3,
            sleep=lambda _s: None,
            clock=lambda: float(next(ticks)),
        )


def test_pull_error_is_reported() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps({"error": "disk dolu"}))

    with pytest.raises(bs.BootstrapError, match="disk dolu"):
        bs.pull_model(httpx.Client(transport=httpx.MockTransport(fail)), OLLAMA, "qwen3:4b")


def test_required_models_include_ollama_judge_and_embedder() -> None:
    cfg = AppConfig(
        embedder=EmbedderConfig(provider="ollama", model="embeddinggemma-2:270m"),
        eval=EvalConfig(judge=JudgeConfig(provider="ollama", model="qwen3")),
    )
    assert bs.required_ollama_models(cfg) == [
        "qwen3:4b",
        "qwen3:latest",
        "embeddinggemma-2:270m",
    ]
    assert bs.required_ollama_models(AppConfig()) == ["qwen3:4b"]  # judge = same model


def test_flag_parsing() -> None:
    assert bs.flag({}, "X") is True
    assert bs.flag({"X": "0"}, "X") is False
    assert bs.flag({"X": "1"}, "X") is True
    assert bs.flag({}, "X", default=False) is False


# --- prebuilt index bundle -----------------------------------------------------------------


class FakeRestore:
    """Stands in for the bundle step: fills the collection and the BM25 dir, or fails."""

    def __init__(self, services: FakeServices, artifacts: Path, *, error: str | None = None):
        self.services, self.artifacts, self.error = services, artifacts, error
        self.calls: list[bool] = []

    def __call__(
        self, cfg: AppConfig, client: httpx.Client, environ: dict[str, str], *, ready: bool
    ) -> bool:
        self.calls.append(ready)
        if self.error:
            raise bs.BootstrapError(self.error)
        self.services.points = 7017
        (self.artifacts / "cache" / "bm25" / "structural_ctx").mkdir(parents=True, exist_ok=True)
        return True


def _run_with_restore(
    tmp_path: Path,
    services: FakeServices,
    commands: Commands,
    restore: object,
    environ: dict[str, str] | None = None,
) -> None:
    bs.bootstrap(
        _cfg(tmp_path),
        Path("configs/default.yaml"),
        httpx.Client(transport=httpx.MockTransport(services)),
        environ=environ or {},
        run=commands,
        warm=lambda cfg: None,
        sleep=lambda _s: None,
        restore=restore,  # type: ignore[arg-type]
    )


def test_bundle_restore_skips_fetch_and_ingest(tmp_path: Path) -> None:
    services, commands = FakeServices(models=["qwen3:4b"], points=None), Commands()
    restore = FakeRestore(services, tmp_path / "artifacts")
    _run_with_restore(tmp_path, services, commands, restore)
    assert restore.calls == [False] and commands.calls == []


def test_bundle_restore_can_also_fetch_the_raw_documents(tmp_path: Path) -> None:
    services, commands = FakeServices(models=["qwen3:4b"], points=None), Commands({"fetch": 1})
    restore = FakeRestore(services, tmp_path / "artifacts")
    _run_with_restore(tmp_path, services, commands, restore, {bs.BUNDLE_FETCH_ENV: "1"})
    assert commands.calls == ["fetch"]  # a failed fetch stays a warning, no ingest


def test_without_a_bundle_the_ingest_path_runs(tmp_path: Path) -> None:
    services, commands = FakeServices(models=["qwen3:4b"], points=None), Commands()
    _run_with_restore(tmp_path, services, commands, lambda *a, **k: False)
    assert commands.calls == ["fetch", "ingest"]


def test_failed_bundle_restore_fails_the_service(tmp_path: Path) -> None:
    services = FakeServices(models=["qwen3:4b"], points=None)
    restore = FakeRestore(services, tmp_path / "artifacts", error="paket bozuk")
    with pytest.raises(bs.BootstrapError, match="paket bozuk"):
        _run_with_restore(tmp_path, services, Commands(), restore)


def _bundle_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {bs.BUNDLE_DIR_ENV: str(tmp_path / "release"), **extra}


def test_restore_step_without_bundle_returns_false(tmp_path: Path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(FakeServices()))
    assert not bs.restore_index_bundle(_cfg(tmp_path), client, _bundle_env(tmp_path), ready=False)
    env = _bundle_env(tmp_path, **{bs.USE_BUNDLE_ENV: "0", bs.BUNDLE_URL_ENV: "https://x.test/"})
    assert not bs.restore_index_bundle(_cfg(tmp_path), client, env, ready=False)


def test_restore_step_bad_url_fails_only_when_the_index_is_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(not_found))
    env = _bundle_env(tmp_path, **{bs.BUNDLE_URL_ENV: "https://example.test/rel/"})
    with pytest.raises(bs.BootstrapError, match="INDEX_BUNDLE_URL"):
        bs.restore_index_bundle(_cfg(tmp_path), client, env, ready=False)
    assert not bs.restore_index_bundle(_cfg(tmp_path), client, env, ready=True)
    assert "Hazır indeks paketi" in caplog.text


# --- HuggingFace access ----------------------------------------------------------------------


class GatedRepoError(Exception):
    """Same class name as huggingface_hub's error for a gated model."""


def test_hf_access_error_explains_hf_token() -> None:
    try:
        try:
            raise GatedRepoError("401 Client Error. Access to model google/x is restricted.")
        except GatedRepoError as inner:
            raise OSError("You are trying to access a gated repo.") from inner
    except OSError as exc:
        message = bs.hf_access_error(exc, "google/x", token_set=False)
    assert message is not None and "HF_TOKEN" in message and "google/x" in message
    with_token = bs.hf_access_error(GatedRepoError("403 Forbidden"), "google/x", token_set=True)
    assert with_token is not None and "HF_TOKEN tanımlı" in with_token
    assert (
        bs.hf_access_error(RuntimeError("CUDA out of memory"), "google/x", token_set=False) is None
    )


def test_warm_turns_a_denied_download_into_a_turkish_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from tubitak_rag import factory
    from tubitak_rag.adapters import hf_cache

    def denied(cfg: AppConfig) -> None:
        raise GatedRepoError("401 Unauthorized")

    monkeypatch.setattr(hf_cache, "is_cached", lambda repo: False)
    monkeypatch.setattr(factory, "runtime_config", lambda cfg: cfg)  # no GPU probe
    monkeypatch.setattr(factory, "build_embedder", denied)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    caplog.set_level("INFO", logger="tubitak_rag.hazirlik")
    with pytest.raises(bs.BootstrapError, match="HF_TOKEN"):
        bs.warm_hf_models(_cfg(tmp_path))
    assert "Cihaz: embedder auto, reranker auto" in caplog.text  # the device is logged
