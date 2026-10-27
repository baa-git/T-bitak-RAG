"""Gradio app settings: environment, ZeroGPU Space defaults, GPU lock, overlay, backend."""

from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.corpus import CorpusOnDisk
from tubitak_rag.config import ConfigError, load_config
from tubitak_rag.interfaces.gradio_app import settings as gr_settings
from tubitak_rag.interfaces.gradio_app.settings import (
    GPU_LOCK_ENV,
    MAX_PIPELINES,
    PRELOAD_ENV,
    SPACE_DEFAULT_PIPELINE,
    SPACE_OVERLAY,
    AppSettings,
    build_backend,
    gpu_lock_enabled,
    load_overlay,
    settings_from_env,
    space_defaults,
)
from tubitak_rag.interfaces.web.fake_backend import FakeBackend
from tubitak_rag.interfaces.web.pipeline_backend import PipelineBackend


def test_defaults_without_environment() -> None:
    settings = settings_from_env({})
    assert settings == AppSettings()
    assert settings.backend == "pipeline" and not settings.is_demo and not settings.cpu_only


def test_environment_is_read() -> None:
    settings = settings_from_env(
        {
            "TUBITAK_RAG_BACKEND": " Fake ",
            "TUBITAK_RAG_CONFIG": "c/x.yaml",
            "TUBITAK_RAG_OVERLAY": "c/katman.yaml",
            "TUBITAK_RAG_DEFAULT_PIPELINE": "v2",
            "TUBITAK_RAG_DEVICE": "CPU",
            "SPACE_INDEX_URL": "https://x/space-index.tar.gz",
            "SPACE_INDEX_SHA256": "ab",
        }
    )
    assert settings.is_demo and settings.cpu_only
    assert settings.config_path == Path("c/x.yaml")
    assert settings.overlay_path == Path("c/katman.yaml")
    assert settings.default_pipeline == "v2"
    assert (settings.index_url, settings.index_sha256) == ("https://x/space-index.tar.gz", "ab")


def test_unknown_backend_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="arka uç"):
        settings_from_env({"TUBITAK_RAG_BACKEND": "uzay"})


def test_load_overlay(tmp_path: Path) -> None:
    assert load_overlay(None) is None
    path = tmp_path / "k.yaml"
    path.write_text("llm:\n  model: ${LLM_MODEL:}\n", encoding="utf-8")
    assert load_overlay(path) == {"llm": {"model": "${LLM_MODEL:}"}}  # interpolated later
    path.write_text("", encoding="utf-8")
    assert load_overlay(path) is None
    path.write_text("- liste\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="eşlemesi"):
        load_overlay(path)
    path.write_text("a: [kapanmamış\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_overlay(path)
    with pytest.raises(ConfigError, match="okunamadı"):
        load_overlay(tmp_path / "yok.yaml")


def test_space_defaults_only_on_a_space_and_never_override() -> None:
    local: dict[str, str] = {}
    space_defaults(local)
    assert local == {}
    space = {"SPACE_ID": "lolalol57/tubitak-mevzuat-rag", "TUBITAK_RAG_DEFAULT_PIPELINE": "v2"}
    space_defaults(space)
    assert space["TUBITAK_RAG_OVERLAY"] == SPACE_OVERLAY.as_posix()
    assert space["TUBITAK_RAG_DEVICE"] == "cuda" and space[GPU_LOCK_ENV] == "0"
    assert space["TUBITAK_RAG_DEFAULT_PIPELINE"] == "v2"  # a Space variable wins
    assert SPACE_DEFAULT_PIPELINE == "v3"


def test_gpu_lock_is_on_locally_off_on_cpu_and_when_switched_off() -> None:
    assert gpu_lock_enabled(AppSettings(), {})
    assert not gpu_lock_enabled(AppSettings(cpu_only=True), {})
    assert not gpu_lock_enabled(AppSettings(), {GPU_LOCK_ENV: "0"})
    assert gpu_lock_enabled(AppSettings(), {GPU_LOCK_ENV: "1"})


def test_repository_zerogpu_overlay_is_valid(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("LLM_MODEL", "LLM_MAX_NEW_TOKENS", "TUBITAK_RAG_EXPORT_DIR"):
        monkeypatch.delenv(name, raising=False)
    overlay = load_overlay(repo_root / "deploy" / "hf-space-gradio" / "space-overlay.yaml")
    assert overlay is not None
    cfg = load_config(repo_root / "configs" / "experiments" / "v3b.yaml", overlay)
    assert cfg.vector_store.provider == "numpy"
    assert cfg.vector_store.export_dir == Path("artifacts/export")
    assert (cfg.embedder.device, cfg.reranker.device) == ("cuda", "cuda")
    assert cfg.llm.provider == "transformers" and cfg.llm.model == "Qwen/Qwen3-4B"
    assert cfg.llm.transformers.device == "cuda" and cfg.llm.transformers.dtype == "bfloat16"
    assert cfg.llm.transformers.max_new_tokens == 768 and cfg.llm.temperature == 0.0
    assert not cfg.llm.think and not cfg.llm.is_external()
    assert cfg.embedder.model == "BAAI/bge-m3" and cfg.reranker.enabled  # methods unchanged
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen3-0.6B")
    small = load_config(repo_root / "configs" / "experiments" / "v2.yaml", overlay)
    assert small.llm.model == "Qwen/Qwen3-0.6B"


def test_build_backend_fake() -> None:
    assert isinstance(build_backend(AppSettings(backend="fake"), {}), FakeBackend)


def test_build_backend_preloads_unless_switched_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preloaded: list[PipelineBackend] = []

    def preload(self: PipelineBackend) -> list[str]:
        preloaded.append(self)
        return []

    monkeypatch.setattr(PipelineBackend, "preload", preload)
    config = CorpusOnDisk(tmp_path).config_file()

    backend = build_backend(AppSettings(config_path=config), {})

    assert isinstance(backend, PipelineBackend) and preloaded == [backend]
    assert backend._max_pipelines == MAX_PIPELINES and backend._use_gpu_lock
    off = build_backend(AppSettings(config_path=config), {PRELOAD_ENV: "0", GPU_LOCK_ENV: "0"})
    assert preloaded == [backend] and isinstance(off, PipelineBackend)
    assert not off._use_gpu_lock


def test_pipelines_built_by_the_backend_share_their_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []

    def build_pipeline(cfg: Any, shared: Any) -> object:
        calls.append(shared)
        return object()

    monkeypatch.setattr("tubitak_rag.factory.build_pipeline", build_pipeline)
    config = CorpusOnDisk(tmp_path).config_file()
    backend = build_backend(AppSettings(config_path=config), {PRELOAD_ENV: "0"})
    assert isinstance(backend, PipelineBackend)
    backend._pipeline_factory(backend.cfg)
    backend._pipeline_factory(backend.cfg)
    assert len(calls) == 2 and calls[0] is calls[1] and calls[0] is not None


def test_build_backend_fetches_the_space_index_for_the_numpy_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, str | None, str | None]] = []

    def ensure(artifacts: Path, *, url: str | None, sha256: str | None = None) -> None:
        calls.append((artifacts, url, sha256))

    monkeypatch.setattr(gr_settings, "ensure_space_index", ensure)
    corpus = CorpusOnDisk(tmp_path)
    config = corpus.config_file(vector_store={"provider": "numpy"})
    build_backend(
        AppSettings(config_path=config, index_url="https://x", index_sha256="ab"),
        {PRELOAD_ENV: "0"},
    )
    assert calls == [(corpus.artifacts, "https://x", "ab")]
