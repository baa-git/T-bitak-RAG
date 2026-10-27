"""FastAPI application factory: pages, REST API, static files and error handlers.

Run locally: ``rag serve`` (real pipeline) or ``uvicorn tubitak_rag.interfaces.web.app:app``
(backend from ``TUBITAK_RAG_BACKEND``, demo data by default). Tests pass a backend:
``create_app(backend=FakeBackend())``.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from tubitak_rag import __version__
from tubitak_rag.interfaces.web import api, pages
from tubitak_rag.interfaces.web.backend import WebBackend
from tubitak_rag.interfaces.web.errors import install_error_handlers
from tubitak_rag.interfaces.web.pages import STATIC_DIR
from tubitak_rag.interfaces.web.state import backend_from_env

DESCRIPTION = (
    "TÜBİTAK ve Ar-Ge mevzuatı üzerinde kaynaklı soru-cevap, belge durumu ve deney "
    'koşuları. Tüm hatalar aynı gövdeyi döndürür: `{"hata": str, "detay": Any}`.'
)


def _html_error(request: Request, status: int, message: str) -> Response:
    if status == 404:
        return pages.not_found_page(request)
    return pages.error_page(request, status, message)


def create_app(*, backend: WebBackend | None = None) -> FastAPI:
    """Build the app; without ``backend`` it is chosen from the environment (default fake)."""
    app = FastAPI(title="TÜBİTAK Mevzuat Rehberi", version=__version__, description=DESCRIPTION)
    app.state.backend = backend if backend is not None else backend_from_env()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(api.router)
    app.include_router(pages.router)
    install_error_handlers(app, html_error=_html_error)
    return app


app = create_app()
