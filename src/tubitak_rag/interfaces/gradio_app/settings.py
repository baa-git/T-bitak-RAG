"""Where the Gradio app gets its backend, and the Hugging Face ZeroGPU Space defaults.

Plain Python (no Gradio), unit-tested directly. Environment variables (all optional):

* ``TUBITAK_RAG_BACKEND`` — ``pipeline`` (default, the real system) or ``fake`` (demo data);
* ``TUBITAK_RAG_CONFIG`` — server config, default ``configs/default.yaml``;
* ``TUBITAK_RAG_OVERLAY`` — YAML merged on top of the server config *and* every experiment
  config (the Space uses ``configs/space-overlay.yaml``);
* ``TUBITAK_RAG_DEFAULT_PIPELINE`` — the config selected when the page opens (e.g. ``v3``);
* ``SPACE_INDEX_URL`` / ``SPACE_INDEX_SHA256`` — download the Space index archive when it is
  not shipped with the Space files (``rag export-space --archive``);
* ``TUBITAK_RAG_DEVICE=cpu`` — besides pinning the models to the CPU (device_policy.py), it
  turns off the GPU lock and shows the speed note of reranker configs;
* ``TUBITAK_RAG_GPU_LOCK=0`` — do not take ``artifacts/.gpu.lock`` per question (the Space
  has a GPU of its own; locally the lock keeps one GPU job at a time);
* ``TUBITAK_RAG_PRELOAD=0`` — build the pipelines at the first question instead of at start
  (local debugging only: on ZeroGPU every model must be loaded at start);
* ``TUBITAK_RAG_GPU_DURATION`` — seconds of GPU reserved per question (``chat.py``).

On a Space (``SPACE_ID`` set by Hugging Face) :func:`space_defaults` fills in the overlay
``configs/space-overlay.yaml`` (NumPy index, CUDA, Qwen3 4B via transformers), ``cuda``, no
GPU lock and the default configuration ``v3``.
"""

import logging
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tubitak_rag.config import AppConfig, ConfigError
from tubitak_rag.interfaces.space_bundle import ensure_space_index
from tubitak_rag.interfaces.web.backend import WebBackend
from tubitak_rag.interfaces.web.fake_backend import FakeBackend

logger = logging.getLogger(__name__)

BACKEND_ENV = "TUBITAK_RAG_BACKEND"
CONFIG_ENV = "TUBITAK_RAG_CONFIG"
OVERLAY_ENV = "TUBITAK_RAG_OVERLAY"
DEFAULT_PIPELINE_ENV = "TUBITAK_RAG_DEFAULT_PIPELINE"
DEVICE_ENV = "TUBITAK_RAG_DEVICE"
INDEX_URL_ENV = "SPACE_INDEX_URL"
INDEX_SHA_ENV = "SPACE_INDEX_SHA256"
SPACE_ID_ENV = "SPACE_ID"
GPU_LOCK_ENV = "TUBITAK_RAG_GPU_LOCK"
PRELOAD_ENV = "TUBITAK_RAG_PRELOAD"
DEFAULT_CONFIG = Path("configs/default.yaml")
SPACE_OVERLAY = Path("configs/space-overlay.yaml")
# On the GPU the reranker costs well under a second, so the full pipeline is the default.
SPACE_DEFAULT_PIPELINE = "v3"
# All six configs (v0 … v3b) stay loaded; they share their models (factory.SharedComponents).
MAX_PIPELINES = 16
_OFF = {"0", "false", "no", "hayir", "hayır", "off"}


@dataclass(frozen=True)
class AppSettings:
    """The resolved settings of one app process."""

    backend: str = "pipeline"
    config_path: Path = DEFAULT_CONFIG
    overlay_path: Path | None = None
    default_pipeline: str | None = None
    cpu_only: bool = False
    index_url: str | None = None
    index_sha256: str | None = None

    @property
    def is_demo(self) -> bool:
        """True for the fake backend (no services, demo answers)."""
        return self.backend == "fake"


def _value(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name, "").strip()
    return value or None


def settings_from_env(environ: Mapping[str, str] | None = None) -> AppSettings:
    """Read :class:`AppSettings` from the environment.

    Raises:
        ConfigError: for an unknown backend kind.
    """
    env = os.environ if environ is None else environ
    backend = (_value(env, BACKEND_ENV) or "pipeline").lower()
    if backend not in ("pipeline", "fake"):
        raise ConfigError(f"Bilinmeyen arka uç: {backend!r} (seçenekler: pipeline, fake)")
    overlay = _value(env, OVERLAY_ENV)
    return AppSettings(
        backend=backend,
        config_path=Path(_value(env, CONFIG_ENV) or DEFAULT_CONFIG),
        overlay_path=Path(overlay) if overlay else None,
        default_pipeline=_value(env, DEFAULT_PIPELINE_ENV),
        cpu_only=(_value(env, DEVICE_ENV) or "").lower() == "cpu",
        index_url=_value(env, INDEX_URL_ENV),
        index_sha256=_value(env, INDEX_SHA_ENV),
    )


def load_overlay(path: Path | None) -> dict[str, Any] | None:
    """The overlay mapping of ``path`` (interpolated later by ``load_config``), or None.

    Raises:
        ConfigError: when the file is missing or is not a YAML mapping.
    """
    if path is None:
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Katman dosyası okunamadı ({path}): {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Katman dosyası YAML olarak okunamadı ({path}): {exc}") from exc
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ConfigError(f"Katman dosyası 'anahtar: değer' eşlemesi olmalıdır: {path}")
    return data


def _enabled(environ: Mapping[str, str], name: str) -> bool:
    return environ.get(name, "").strip().lower() not in _OFF


def space_defaults(environ: MutableMapping[str, str]) -> None:
    """On a Hugging Face Space (``SPACE_ID`` set), fill the ZeroGPU defaults that are unset."""
    if not environ.get(SPACE_ID_ENV, "").strip():
        return
    environ.setdefault(OVERLAY_ENV, SPACE_OVERLAY.as_posix())
    environ.setdefault(DEVICE_ENV, "cuda")
    environ.setdefault(DEFAULT_PIPELINE_ENV, SPACE_DEFAULT_PIPELINE)
    environ.setdefault(GPU_LOCK_ENV, "0")


def gpu_lock_enabled(settings: AppSettings, environ: Mapping[str, str]) -> bool:
    """Whether each question takes the GPU lock file (not on CPU, not when switched off)."""
    return not settings.cpu_only and _enabled(environ, GPU_LOCK_ENV)


def build_backend(settings: AppSettings, environ: Mapping[str, str]) -> WebBackend:
    """The backend of the app, with every pipeline built and its models loaded.

    With the NumPy vector store the Space index is fetched first when it is missing and
    ``SPACE_INDEX_URL`` is set. All configs share one ``SharedComponents``: v0 … v3 use the
    same EmbeddingGemma 2, v3b adds bge-m3, and the reranker and the LLM are loaded once.

    Raises:
        ConfigError: for an invalid config or overlay; ``SpaceIndexError`` when a download fails.
    """
    if settings.is_demo:
        return FakeBackend()
    # Imported here: the demo backend must not pull in the factory and its adapters.
    from tubitak_rag.factory import SharedComponents, build_pipeline
    from tubitak_rag.interfaces.web.pipeline_backend import PipelineBackend
    from tubitak_rag.pipeline import RagPipeline

    shared = SharedComponents()

    def pipeline_factory(cfg: AppConfig) -> RagPipeline:
        return build_pipeline(cfg, shared)

    backend = PipelineBackend.from_config(
        settings.config_path,
        config_overrides=load_overlay(settings.overlay_path),
        default_pipeline=settings.default_pipeline,
        use_gpu_lock=gpu_lock_enabled(settings, environ),
        pipeline_factory=pipeline_factory,
        max_pipelines=MAX_PIPELINES,
    )
    if backend.cfg.vector_store.provider == "numpy":
        ensure_space_index(
            backend.cfg.paths.artifacts_dir, url=settings.index_url, sha256=settings.index_sha256
        )
    if _enabled(environ, PRELOAD_ENV):
        loaded = backend.preload()
        logger.info("gradio: yüklenen yapılandırmalar: %s", ", ".join(loaded) or "yok")
    return backend
