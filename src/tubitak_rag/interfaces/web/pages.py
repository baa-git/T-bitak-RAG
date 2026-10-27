"""Server-rendered pages (Sohbet, Hakkında, document preview) and the HTMX turn fragment.

The query page is a conversation panel. A question is sent with HTMX to ``/ui/ask``, which
returns one *turn* (the question and the reply with its cited sources) that the browser
appends to the thread. Without JavaScript the same form loads ``/?soru=…`` and the page
renders that single turn.
"""

import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from tubitak_rag.interfaces.gpu_lock import GpuLockError
from tubitak_rag.interfaces.web.about import DOCS_URL, REPO_URL, methods_text
from tubitak_rag.interfaces.web.backend import (
    ServiceUnavailableError,
    UnknownPipelineError,
    WebBackend,
)
from tubitak_rag.interfaces.web.fake_backend import EXAMPLES
from tubitak_rag.interfaces.web.rendering import (
    DOC_TYPE_LABELS,
    DOCUMENT_STATE_LABELS,
    LANGUAGE_LABELS,
    cited_sources,
    date_tr,
    link_citations,
    location_parts,
    number_tr,
    plain_paragraphs,
    seconds_tr,
)
from tubitak_rag.interfaces.web.schemas import MAX_QUESTION_CHARS, AskResult, PipelineOption

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=TEMPLATES_DIR)
templates.env.filters.update(
    sayi=number_tr, tarih=date_tr, paragraflar=plain_paragraphs, saniye=seconds_tr
)
templates.env.globals.update(
    atifli_metin=link_citations,
    atif_kaynaklari=cited_sources,
    konum_parcalari=location_parts,
    yontem_metni=methods_text,
    DOC_TYPE_LABELS=DOC_TYPE_LABELS,
    DOCUMENT_STATE_LABELS=DOCUMENT_STATE_LABELS,
    LANGUAGE_LABELS=LANGUAGE_LABELS,
    REPO_URL=REPO_URL,
    DOCS_URL=DOCS_URL,
)

router = APIRouter(include_in_schema=False)


def _backend(request: Request) -> WebBackend:
    backend: WebBackend = request.app.state.backend
    return backend


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def _render(
    request: Request, name: str, context: dict[str, Any], status_code: int = 200
) -> HTMLResponse:
    context = {"is_demo": _backend(request).is_demo, **context}
    response: HTMLResponse = templates.TemplateResponse(
        request, name, context, status_code=status_code
    )
    return response


# --- Sohbet --------------------------------------------------------------------------------


def _default(options: list[PipelineOption]) -> str:
    return next((o.ad for o in options if o.varsayilan), options[0].ad if options else "")


def _turn(request: Request, soru: str, yapilandirma: str | None) -> dict[str, Any]:
    """One conversation turn: the question plus the answer or an inline error message.

    ``onek`` is a fresh anchor prefix, so the sources of several answers in one thread never
    share an id ("kaynak-t1a2b3c-1").
    """
    options = _backend(request).pipelines()
    chosen = (yapilandirma or "").strip() or _default(options)
    question = soru.strip()
    turn: dict[str, Any] = {
        "soru": question,
        "sonuc": None,
        "hata": None,
        "onek": f"t{secrets.token_hex(3)}-",
        "yontemler": {o.ad: methods_text(o.yontemler) for o in options},
    }
    if not question:
        turn["hata"] = "Lütfen bir soru yazın."
    elif len(question) > MAX_QUESTION_CHARS:
        turn["hata"] = f"Soru en fazla {MAX_QUESTION_CHARS} karakter olabilir."
    elif chosen not in turn["yontemler"]:
        turn["hata"] = f"Bilinmeyen yapılandırma: {chosen}"
    else:
        try:
            result: AskResult = _backend(request).ask(question, None, None, chosen)
            turn["sonuc"] = result
        except (GpuLockError, ServiceUnavailableError, UnknownPipelineError) as exc:
            turn["hata"] = str(exc)
    return turn


@router.get("/", response_class=HTMLResponse)
def sorgu_page(
    request: Request, soru: str | None = None, yapilandirma: str | None = None
) -> HTMLResponse:
    """Conversation panel; with ``?soru=`` it renders that turn (works without JavaScript)."""
    backend = _backend(request)
    options = backend.pipelines()
    chosen = (yapilandirma or "").strip() or _default(options)
    context: dict[str, Any] = {
        "aktif": "sorgu",
        "yapilandirmalar": options,
        "secili": chosen,
        "ornekler": EXAMPLES,
        "uyari_metni": backend.disclaimer(),
        "tur": _turn(request, soru, yapilandirma) if soru is not None else None,
    }
    return _render(request, "sorgu.html", context)


@router.get("/ui/ask", response_class=HTMLResponse)
def ask_fragment(request: Request, soru: str = "", yapilandirma: str | None = None) -> Response:
    """HTMX fragment: one turn to append to the thread (plain requests go to the page)."""
    if not _is_htmx(request):
        query = urlencode({k: v for k, v in (("soru", soru), ("yapilandirma", yapilandirma)) if v})
        return RedirectResponse(f"/?{query}", status_code=303)
    return _render(request, "partials/tur.html", {"tur": _turn(request, soru, yapilandirma)})


# --- Hakkında and document preview ---------------------------------------------------------


@router.get("/hakkinda", response_class=HTMLResponse)
def hakkinda_page(request: Request) -> HTMLResponse:
    """What the system is built from: corpus, counts, models, methods, guardrails."""
    context = {"aktif": "hakkinda", "bilgi": _backend(request).about()}
    return _render(request, "hakkinda.html", context)


@router.get("/belgeler")
def belgeler_page() -> RedirectResponse:
    """The document list was replaced by the Hakkında page; old links land there."""
    return RedirectResponse("/hakkinda", status_code=301)


@router.get("/belgeler/{doc_id}", response_class=HTMLResponse)
def belge_page(request: Request, doc_id: str) -> HTMLResponse:
    """Document preview reached from a source link: identity and the parsed sections."""
    preview = _backend(request).document(doc_id)
    if preview is None:
        return _not_found(request, f"“{doc_id}” kimlikli bir belge bulunamadı.")
    return _render(request, "belge.html", {"aktif": None, "onizleme": preview})


def _not_found(request: Request, mesaj: str) -> HTMLResponse:
    return _render(request, "bulunamadi.html", {"aktif": None, "mesaj": mesaj}, status_code=404)


def not_found_page(request: Request) -> HTMLResponse:
    """HTML 404 for unknown page URLs (used by the app's exception handler)."""
    return _not_found(request, "Aradığınız sayfa bulunamadı.")


ERROR_TITLES = {409: "GPU şu anda meşgul", 503: "Servis kullanılamıyor"}


def error_page(request: Request, status: int, mesaj: str) -> HTMLResponse:
    """HTML error page for browsers (service down, GPU busy, unexpected error)."""
    context = {"aktif": None, "mesaj": mesaj, "baslik": ERROR_TITLES.get(status, "Bir hata oluştu")}
    return _render(request, "bulunamadi.html", context, status_code=status)
