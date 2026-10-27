"""Uniform Turkish error body {hata, detay} for every API failure (FR-A4)."""

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tubitak_rag.interfaces.gpu_lock import GpuLockError
from tubitak_rag.interfaces.web.backend import ServiceUnavailableError, UnknownPipelineError
from tubitak_rag.interfaces.web.errors import ErrorBody, install_error_handlers


class _Body(BaseModel):
    soru: str


def _app() -> FastAPI:
    app = FastAPI()

    def html_error(request: Request, status: int, message: str) -> HTMLResponse:
        return HTMLResponse(f"<p>{message}</p>", status_code=status)

    install_error_handlers(app, html_error=html_error)

    @app.post("/api/body")
    def body(payload: _Body) -> dict[str, str]:
        return {"soru": payload.soru}

    @app.get("/api/missing")
    def missing() -> None:
        raise HTTPException(404, detail="Koşu bulunamadı: x")

    @app.get("/api/gpu")
    def gpu() -> None:
        raise GpuLockError("GPU kilidi dolu: artifacts/.gpu.lock (rag run)")

    @app.get("/api/service")
    def service() -> None:
        raise ServiceUnavailableError("Qdrant'a ulaşılamadı")

    @app.get("/api/pipeline")
    def pipeline() -> None:
        raise UnknownPipelineError("Bilinmeyen yapılandırma: v9")

    @app.get("/api/boom")
    def boom() -> None:
        raise RuntimeError("beklenmeyen")

    @app.get("/belgeler")
    def page() -> None:
        raise ServiceUnavailableError("Qdrant'a ulaşılamadı")

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_app(), raise_server_exceptions=False)


def _body(response: object) -> ErrorBody:
    return ErrorBody.model_validate(response.json())  # type: ignore[attr-defined]


def test_validation_error_is_422_with_turkish_title_and_details(client: TestClient) -> None:
    response = client.post("/api/body", json={})
    assert response.status_code == 422
    body = _body(response)
    assert body.hata == "Geçersiz istek"
    assert body.detay[0]["loc"] == ["body", "soru"]


def test_malformed_json_is_422_too(client: TestClient) -> None:
    response = client.post(
        "/api/body", content=b"{bozuk", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422 and _body(response).hata == "Geçersiz istek"


def test_http_404_has_uniform_body(client: TestClient) -> None:
    response = client.get("/api/missing")
    assert response.status_code == 404
    assert _body(response) == ErrorBody(hata="Bulunamadı", detay="Koşu bulunamadı: x")


def test_unknown_route_has_uniform_body(client: TestClient) -> None:
    response = client.get("/api/yok")
    assert response.status_code == 404 and _body(response).hata == "Bulunamadı"


def test_wrong_method_has_uniform_body(client: TestClient) -> None:
    response = client.get("/api/body")
    assert response.status_code == 405 and _body(response).hata == "Yöntem desteklenmiyor"


def test_gpu_lock_is_409(client: TestClient) -> None:
    response = client.get("/api/gpu")
    assert response.status_code == 409
    body = _body(response)
    assert body.hata == "GPU meşgul" and "GPU kilidi dolu" in body.detay


def test_unavailable_service_is_503(client: TestClient) -> None:
    response = client.get("/api/service")
    assert response.status_code == 503
    assert _body(response) == ErrorBody(hata="Servis kullanılamıyor", detay="Qdrant'a ulaşılamadı")


def test_unknown_pipeline_is_422(client: TestClient) -> None:
    response = client.get("/api/pipeline")
    assert response.status_code == 422
    assert _body(response).detay == "Bilinmeyen yapılandırma: v9"


def test_unexpected_error_is_500_with_message(client: TestClient) -> None:
    response = client.get("/api/boom")
    assert response.status_code == 500
    assert _body(response) == ErrorBody(hata="Sunucu hatası", detay="beklenmeyen")


def test_page_errors_render_html_for_browsers(client: TestClient) -> None:
    response = client.get("/belgeler", headers={"Accept": "text/html"})
    assert response.status_code == 503
    assert "Qdrant'a ulaşılamadı" in response.text and "hata" not in response.text


def test_error_body_detail_is_optional() -> None:
    assert ErrorBody(hata="Bulunamadı").detay is None
