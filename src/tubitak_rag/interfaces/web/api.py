"""REST API: eight endpoints (FR-A1), documented at ``/docs`` (FR-A2).

Every request and response is a Pydantic model (FR-A3), every failure the uniform
``{"hata", "detay"}`` body of ``errors.py`` (FR-A4). Long work (ingest, experiment run) is
only *started* here: the endpoint answers ``202 Accepted`` with a job id at once. The routes
receive the backend through ``Depends(get_backend)`` and never build anything themselves.

Starting work is administrative: it needs ``TUBITAK_RAG_ADMIN_TOKEN`` on the server and the
same value as a bearer token. Without the variable those two endpoints are closed, so a
server reachable by others can only answer questions.
"""

import os
import secrets
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, Header, Query, Response
from fastapi.responses import HTMLResponse
from starlette.exceptions import HTTPException

from tubitak_rag.domain.answer import Answer
from tubitak_rag.interfaces.web.backend import WebBackend
from tubitak_rag.interfaces.web.errors import ErrorBody
from tubitak_rag.interfaces.web.rendering import render_markdown
from tubitak_rag.interfaces.web.schemas import (
    AskRequest,
    DocumentsResponse,
    HealthStatus,
    IngestRequest,
    JobStarted,
    ReportResponse,
    RunDetail,
    RunRequest,
    RunsResponse,
)
from tubitak_rag.interfaces.web.state import get_backend

Backend = Annotated[WebBackend, Depends(get_backend)]


def _errors(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {code: {"model": ErrorBody} for code in codes}


router = APIRouter(prefix="/api", responses=_errors(422, 500))


ADMIN_TOKEN_ENV = "TUBITAK_RAG_ADMIN_TOKEN"


def require_admin(authorization: Annotated[str | None, Header()] = None) -> None:
    """Allow ingest and experiment runs only with the server's admin token."""
    expected = os.environ.get(ADMIN_TOKEN_ENV, "")
    if not expected:
        raise HTTPException(
            403, f"Yönetim uç noktaları kapalı; açmak için sunucuda {ADMIN_TOKEN_ENV} tanımlayın."
        )
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip(), expected):
        raise HTTPException(401, "Geçerli bir yönetici anahtarı gerekli (Authorization: Bearer).")


Admin = Depends(require_admin)


def _config_name(value: str) -> str:
    """``configs/experiments/v2.yaml`` and ``v2`` both name the experiment ``v2``."""
    return Path(value).stem if value.endswith((".yaml", ".yml")) else value


@router.post(
    "/ask",
    response_model=Answer,
    summary="Soru sor",
    responses=_errors(409, 503),
    tags=["sorgu"],
)
def ask(request: AskRequest, backend: Backend, response: Response) -> Answer:
    """Soruyu seçilen pipeline ile cevaplar; reddedilen sorular kaynaksız döner.

    ``X-Yapilandirma`` başlığı cevabı üreten deney yapılandırmasını taşır.
    """
    config = _config_name(request.config) if request.config else None
    doc_type = request.doc_turu.value if request.doc_turu else None
    result = backend.ask(request.soru, request.program_kodu or None, doc_type, config)
    if result.yapilandirma:
        response.headers["X-Yapilandirma"] = result.yapilandirma
    return result.answer


@router.get(
    "/documents",
    response_model=DocumentsResponse,
    summary="Belgeleri listele",
    responses=_errors(503),
    tags=["belgeler"],
)
def documents(backend: Backend) -> DocumentsResponse:
    """Manifestteki belgeler ve varsayılan koleksiyondaki indeks durumları."""
    return backend.documents()


@router.post(
    "/ingest",
    status_code=202,
    response_model=JobStarted,
    summary="Ingest başlat",
    responses=_errors(401, 403, 409),
    tags=["belgeler"],
    dependencies=[Admin],
)
def ingest(backend: Backend, request: Annotated[IngestRequest | None, Body()] = None) -> JobStarted:
    """Artımlı (``full=true`` ise baştan) ingest'i arka planda başlatır; GPU meşgulse 409."""
    return backend.start_ingest(full=request.full if request else False)


@router.get("/runs", response_model=RunsResponse, summary="Koşuları listele", tags=["deneyler"])
def runs(backend: Backend) -> RunsResponse:
    """Tamamlanan ve süren deney koşuları, en yenisi önce."""
    return RunsResponse(kosular=backend.runs())


@router.post(
    "/runs",
    status_code=202,
    response_model=JobStarted,
    summary="Deney koşusu başlat",
    responses=_errors(401, 403, 409),
    tags=["deneyler"],
    dependencies=[Admin],
)
def start_run(request: RunRequest, backend: Backend) -> JobStarted:
    """``configs/experiments`` altındaki bir yapılandırmayı arka planda koşar; GPU meşgulse 409."""
    return backend.start_run(_config_name(request.config), request.limit)


def _not_found(run_id: str) -> HTTPException:
    return HTTPException(404, detail=f"Koşu bulunamadı: {run_id}")


@router.get(
    "/runs/{run_id}",
    response_model=RunDetail,
    summary="Koşu ayrıntısı",
    responses=_errors(404),
    tags=["deneyler"],
)
def run_detail(run_id: str, backend: Backend) -> RunDetail:
    """Koşunun metrikleri ve soru bazında sonuçları."""
    detail = backend.run_detail(run_id)
    if detail is None:
        raise _not_found(run_id)
    return detail


@router.get(
    "/runs/{run_id}/report",
    response_model=ReportResponse,
    summary="Koşu raporu",
    responses={**_errors(404), 200: {"content": {"text/html": {}}}},
    tags=["deneyler"],
)
def run_report(
    run_id: str,
    backend: Backend,
    format: Annotated[Literal["markdown", "html"], Query()] = "markdown",
) -> ReportResponse | HTMLResponse:
    """``report.md``; ``?format=html`` ile HTML'e çevrilmiş hâli."""
    markdown = backend.run_report(run_id)
    if markdown is None:
        raise _not_found(run_id)
    if format == "html":
        return HTMLResponse(str(render_markdown(markdown)))
    return ReportResponse(markdown=markdown)


@router.get("/health", response_model=HealthStatus, summary="Sağlık kontrolü", tags=["sistem"])
def health(backend: Backend) -> HealthStatus:
    """Qdrant ve Ollama erişilebilir mi? Model yüklemez, pipeline kurmaz."""
    return backend.health()
