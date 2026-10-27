"""Which backend serves the app, and how routes receive it (dependency injection).

The app object holds exactly one ``WebBackend`` in ``app.state.backend``. Routes never
create it; they declare ``backend: WebBackend = Depends(get_backend)`` and FastAPI passes the
app's instance. Tests therefore swap the whole system for ``FakeBackend`` with one argument.

Selection: ``TUBITAK_RAG_BACKEND=fake|pipeline`` (default ``fake``, which keeps tests and a
bare ``uvicorn …:app`` free of services) and ``TUBITAK_RAG_CONFIG`` (default
``configs/default.yaml``). ``rag serve`` chooses ``pipeline`` unless told otherwise.
"""

import os
from collections.abc import Mapping
from pathlib import Path

from fastapi import Request

from tubitak_rag.interfaces.web.backend import WebBackend
from tubitak_rag.interfaces.web.fake_backend import FakeBackend

BACKEND_ENV = "TUBITAK_RAG_BACKEND"
CONFIG_ENV = "TUBITAK_RAG_CONFIG"
DEFAULT_CONFIG = Path("configs/default.yaml")
BACKEND_KINDS = ("fake", "pipeline")


def build_backend(kind: str, config_path: Path = DEFAULT_CONFIG) -> WebBackend:
    """``FakeBackend`` or a lazy ``PipelineBackend`` over ``config_path``.

    Raises:
        ValueError: for an unknown kind; ``ConfigError`` for an invalid config.
    """
    if kind == "fake":
        return FakeBackend()
    if kind == "pipeline":
        # Imported here: the fake backend must not pull in the factory and its adapters.
        from tubitak_rag.interfaces.web.pipeline_backend import PipelineBackend

        return PipelineBackend.from_config(config_path)
    raise ValueError(f"Bilinmeyen arka uç: {kind!r} (seçenekler: {', '.join(BACKEND_KINDS)})")


def backend_from_env(environ: Mapping[str, str] | None = None) -> WebBackend:
    """The backend named by ``TUBITAK_RAG_BACKEND`` / ``TUBITAK_RAG_CONFIG``."""
    env = os.environ if environ is None else environ
    return build_backend(env.get(BACKEND_ENV, "fake"), Path(env.get(CONFIG_ENV, DEFAULT_CONFIG)))


def get_backend(request: Request) -> WebBackend:
    """FastAPI dependency: the backend of the running app."""
    backend: WebBackend = request.app.state.backend
    return backend
