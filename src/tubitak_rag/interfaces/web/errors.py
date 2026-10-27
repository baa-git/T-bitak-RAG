"""One error contract for the whole API: ``{"hata": str, "detay": Any}`` (FR-A4).

``hata`` is a short Turkish title chosen by the HTTP status, ``detay`` carries the specifics
(validation errors, the missing id, the GPU lock holder). Clients therefore parse a single
shape for every failure. Browser requests to pages get an HTML page instead of JSON.
"""

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from tubitak_rag.interfaces.gpu_lock import GpuLockError
from tubitak_rag.interfaces.web.backend import ServiceUnavailableError, UnknownPipelineError

STATUS_TITLES = {
    400: "Hatalı istek",
    401: "Yetkisiz",
    403: "Erişim kapalı",
    404: "Bulunamadı",
    405: "Yöntem desteklenmiyor",
    409: "GPU meşgul",
    422: "Geçersiz istek",
    500: "Sunucu hatası",
    503: "Servis kullanılamıyor",
}

HtmlError = Callable[[Request, int, str], Response]


class ErrorBody(BaseModel):
    """Body of every API error response."""

    hata: str
    detay: Any = None


def error_response(status: int, detail: Any = None) -> JSONResponse:
    """JSON error with the uniform body; the title follows the status code."""
    body = ErrorBody(hata=STATUS_TITLES.get(status, "Hata"), detay=jsonable_encoder(detail))
    return JSONResponse(body.model_dump(), status_code=status)


def _wants_html(request: Request) -> bool:
    accepts_html = "text/html" in request.headers.get("accept", "")
    return accepts_html and not request.url.path.startswith("/api")


def install_error_handlers(app: FastAPI, *, html_error: HtmlError | None = None) -> None:
    """Map validation, HTTP, GPU, service and unexpected errors to the uniform body.

    ``html_error(request, status, message)`` renders failures of page requests from a
    browser; without it every response is JSON.
    """

    def respond(request: Request, status: int, detail: Any) -> Response:
        if html_error is not None and _wants_html(request):
            return html_error(request, status, str(detail))
        return error_response(status, detail)

    async def validation(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, RequestValidationError)
        return error_response(422, exc.errors())

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, StarletteHTTPException)
        response = respond(request, exc.status_code, exc.detail)
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    async def gpu_busy(request: Request, exc: Exception) -> Response:
        return respond(request, 409, str(exc))

    async def unavailable(request: Request, exc: Exception) -> Response:
        return respond(request, 503, str(exc))

    async def unknown_pipeline(request: Request, exc: Exception) -> Response:
        return respond(request, 422, str(exc))

    async def unexpected(request: Request, exc: Exception) -> Response:
        return respond(request, 500, str(exc))

    app.add_exception_handler(RequestValidationError, validation)
    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(GpuLockError, gpu_busy)
    app.add_exception_handler(ServiceUnavailableError, unavailable)
    app.add_exception_handler(UnknownPipelineError, unknown_pipeline)
    app.add_exception_handler(Exception, unexpected)
