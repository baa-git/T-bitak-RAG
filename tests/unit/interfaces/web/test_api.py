"""REST API (FR-A1..A4) over the demo backend: happy paths, validation and error bodies."""

import pytest
from fastapi.testclient import TestClient

from tubitak_rag.interfaces.web.api import ADMIN_TOKEN_ENV
from tubitak_rag.interfaces.web.app import create_app
from tubitak_rag.interfaces.web.fake_backend import (
    EXAMPLE_ANSWERED,
    EXAMPLE_NOT_FOUND,
    EXAMPLE_OUT_OF_SCOPE,
    FakeBackend,
)


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


ADMIN_TOKEN = "test-admin-token"


@pytest.fixture
def client(backend: FakeBackend, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv(ADMIN_TOKEN_ENV, ADMIN_TOKEN)
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    return TestClient(create_app(backend=backend), headers=headers)


def _error(response: object, status: int, hata: str) -> object:
    assert response.status_code == status  # type: ignore[attr-defined]
    body = response.json()  # type: ignore[attr-defined]
    assert set(body) == {"hata", "detay"} and body["hata"] == hata
    return body["detay"]


# --- /api/ask ------------------------------------------------------------------------------


def test_ask_returns_answer_with_citations_and_disclaimer(client: TestClient) -> None:
    response = client.post("/api/ask", json={"soru": EXAMPLE_ANSWERED})
    assert response.status_code == 200
    answer = response.json()
    assert answer["durum"] == "cevaplandi" and answer["uyari"]
    assert [c["no"] for c in answer["atiflar"]] == [1, 2, 3]
    assert len(answer["kaynaklar"]) == 3
    assert response.headers["X-Yapilandirma"] == "v3"


def test_ask_forwards_filters_and_config(client: TestClient, backend: FakeBackend) -> None:
    payload = {"soru": EXAMPLE_ANSWERED, "program_kodu": "1507", "doc_turu": "kanun"}
    response = client.post("/api/ask", json={**payload, "config": "v1"})
    assert response.status_code == 200 and response.headers["X-Yapilandirma"] == "v1"
    assert backend.asked[-1] == (EXAMPLE_ANSWERED, "1507", "kanun", "v1")


@pytest.mark.parametrize("soru", [EXAMPLE_NOT_FOUND, EXAMPLE_OUT_OF_SCOPE])
def test_refusals_return_no_sources(client: TestClient, soru: str) -> None:
    answer = client.post("/api/ask", json={"soru": soru}).json()
    assert answer["kaynaklar"] == [] and answer["atiflar"] == []
    assert not answer["cevaplanabilir"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"soru": ""}, {"soru": "   "}, {"soru": "x" * 1001}, {"soru": "s", "doc_turu": "roman"}],
)
def test_request_schema_validation_422(client: TestClient, payload: dict[str, object]) -> None:
    detail = _error(client.post("/api/ask", json=payload), 422, "Geçersiz istek")
    assert isinstance(detail, list) and detail


def test_unknown_config_is_422(client: TestClient) -> None:
    detail = _error(
        client.post("/api/ask", json={"soru": EXAMPLE_ANSWERED, "config": "v9"}),
        422,
        "Geçersiz istek",
    )
    assert "v9" in str(detail)


def test_ask_while_gpu_is_busy_is_409(client: TestClient, backend: FakeBackend) -> None:
    backend.gpu_busy = True
    detail = _error(client.post("/api/ask", json={"soru": EXAMPLE_ANSWERED}), 409, "GPU meşgul")
    assert "GPU kilidi dolu" in str(detail)


# --- /api/documents, /api/ingest -------------------------------------------------------------


def test_documents_lists_status(client: TestClient) -> None:
    body = client.get("/api/documents").json()
    assert body["koleksiyon"] and body["toplam_chunk"] > 0
    assert {d["durum"] for d in body["belgeler"]} == {
        "indeksli",
        "taranmis",
        "eksik",
        "guncel_degil",
    }


def test_ingest_starts_job_202(client: TestClient, backend: FakeBackend) -> None:
    response = client.post("/api/ingest", json={"full": True})
    assert response.status_code == 202
    assert response.json() == {"job_id": "ingest-0001", "durum": "basladi"}
    assert client.post("/api/ingest").status_code == 202  # body is optional
    assert backend.jobs == ["ingest-0001", "ingest-0002"]


def test_ingest_while_gpu_is_busy_is_409(client: TestClient, backend: FakeBackend) -> None:
    backend.gpu_busy = True
    _error(client.post("/api/ingest", json={}), 409, "GPU meşgul")
    assert backend.jobs == []


def test_ingest_rejects_unknown_fields(client: TestClient) -> None:
    _error(client.post("/api/ingest", json={"tam": True}), 422, "Geçersiz istek")


# --- /api/runs -----------------------------------------------------------------------------


def test_runs_list_and_detail(client: TestClient) -> None:
    runs = client.get("/api/runs").json()["kosular"]
    assert [r["deney"] for r in runs][:2] == ["v3b", "v3"]
    detail = client.get(f"/api/runs/{runs[0]['run_id']}").json()
    assert detail["metrics"]["run_id"] == runs[0]["run_id"] and detail["sorular"] == []


def test_unknown_run_is_404_everywhere(client: TestClient) -> None:
    for path in ["/api/runs/yok", "/api/runs/yok/report"]:
        assert "yok" in str(_error(client.get(path), 404, "Bulunamadı"))


def test_run_report_markdown_and_html(client: TestClient) -> None:
    run_id = client.get("/api/runs").json()["kosular"][0]["run_id"]
    markdown = client.get(f"/api/runs/{run_id}/report").json()["markdown"]
    assert markdown.startswith("#")
    html = client.get(f"/api/runs/{run_id}/report", params={"format": "html"})
    assert html.headers["content-type"].startswith("text/html")
    assert "<table" in html.text and "<h" in html.text


def test_report_format_is_validated(client: TestClient) -> None:
    run_id = client.get("/api/runs").json()["kosular"][0]["run_id"]
    _error(
        client.get(f"/api/runs/{run_id}/report", params={"format": "pdf"}), 422, "Geçersiz istek"
    )


def test_runs_post_starts_job(client: TestClient, backend: FakeBackend) -> None:
    response = client.post("/api/runs", json={"config": "configs/experiments/v2.yaml", "limit": 3})
    assert response.status_code == 202 and response.json()["durum"] == "basladi"
    assert backend.runs_started == [("v2", 3)]


@pytest.mark.parametrize(
    "payload", [{"config": "v9"}, {"config": "v3", "limit": 0}, {"limit": 3}, {"config": ""}]
)
def test_runs_post_validation(client: TestClient, payload: dict[str, object]) -> None:
    _error(client.post("/api/runs", json=payload), 422, "Geçersiz istek")


def test_runs_post_while_gpu_is_busy_is_409(client: TestClient, backend: FakeBackend) -> None:
    backend.gpu_busy = True
    _error(client.post("/api/runs", json={"config": "v3"}), 409, "GPU meşgul")


# --- /api/health, OpenAPI ------------------------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["durum"] == "ok" and set(body) == {
        "durum",
        "qdrant",
        "ollama",
        "koleksiyon",
        "surum",
    }


def test_openapi_docs_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    operations = {
        (method.upper(), path) for path, item in schema["paths"].items() for method in item
    }
    assert operations == {
        ("POST", "/api/ask"),
        ("GET", "/api/documents"),
        ("POST", "/api/ingest"),
        ("GET", "/api/runs"),
        ("POST", "/api/runs"),
        ("GET", "/api/runs/{run_id}"),
        ("GET", "/api/runs/{run_id}/report"),
        ("GET", "/api/health"),
    }
    assert "ErrorBody" in schema["components"]["schemas"]
