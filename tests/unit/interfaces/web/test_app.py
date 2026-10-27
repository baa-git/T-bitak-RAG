"""Application factory: default backend, static files and the HTML 404 page."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tubitak_rag.interfaces.web import app as app_module
from tubitak_rag.interfaces.web.app import create_app
from tubitak_rag.interfaces.web.fake_backend import FakeBackend
from tubitak_rag.interfaces.web.pipeline_backend import PipelineBackend
from tubitak_rag.interfaces.web.state import BACKEND_ENV, CONFIG_ENV


def test_default_backend_is_demo_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    assert isinstance(create_app().state.backend, FakeBackend)


def test_backend_is_switched_by_environment(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    monkeypatch.setenv(BACKEND_ENV, "pipeline")
    monkeypatch.setenv(CONFIG_ENV, str(repo_root / "configs" / "default.yaml"))
    backend = create_app().state.backend
    assert isinstance(backend, PipelineBackend) and not backend.is_demo
    assert backend.loaded_pipelines() == []  # lazy: nothing built at app start


def test_injected_backend_is_used() -> None:
    backend = FakeBackend()
    assert create_app(backend=backend).state.backend is backend


def test_module_level_app_exists_for_uvicorn() -> None:
    assert app_module.app.title == "TÜBİTAK Mevzuat Rehberi"


def test_static_assets_are_served_locally() -> None:
    client = TestClient(create_app())
    for path in (
        "/static/style.css",
        "/static/app.js",
        "/static/htmx.min.js",
        "/static/fonts/source-serif-4-latin-ext-opsz-normal.woff2",
    ):
        assert client.get(path).status_code == 200


def test_unknown_page_returns_html_404_for_browsers() -> None:
    response = TestClient(create_app()).get("/yok", headers={"Accept": "text/html"})
    assert response.status_code == 404
    assert "Sayfa bulunamadı" in response.text


def test_unknown_api_path_keeps_json_404() -> None:
    response = TestClient(create_app()).get("/api/yok", headers={"Accept": "text/html"})
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
