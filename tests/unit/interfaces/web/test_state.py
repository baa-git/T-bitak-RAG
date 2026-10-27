"""Backend selection (fake | pipeline) from the environment and FastAPI injection."""

from pathlib import Path
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tubitak_rag.interfaces.web.backend import WebBackend
from tubitak_rag.interfaces.web.fake_backend import FakeBackend
from tubitak_rag.interfaces.web.pipeline_backend import PipelineBackend
from tubitak_rag.interfaces.web.state import (
    BACKEND_ENV,
    CONFIG_ENV,
    backend_from_env,
    build_backend,
    get_backend,
)


def test_fake_is_the_default_without_environment() -> None:
    assert isinstance(backend_from_env({}), FakeBackend)


def test_pipeline_backend_is_selected_and_stays_lazy(repo_root: Path) -> None:
    config = repo_root / "configs" / "default.yaml"
    backend = backend_from_env({BACKEND_ENV: "pipeline", CONFIG_ENV: str(config)})
    assert isinstance(backend, PipelineBackend)
    assert backend.config_path == config and backend.loaded_pipelines() == []


def test_unknown_backend_kind_is_a_turkish_error() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen arka uç"):
        build_backend("gercek")


def test_get_backend_injects_the_app_backend() -> None:
    app = FastAPI()
    app.state.backend = FakeBackend()

    @app.get("/demo")
    def demo(backend: Annotated[WebBackend, Depends(get_backend)]) -> dict[str, bool]:
        return {"demo": backend.is_demo}

    assert TestClient(app).get("/demo").json() == {"demo": True}
