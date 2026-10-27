"""Text helpers for templates: Turkish number/date formats, citation links, sources, Markdown."""

import re
from collections.abc import Iterable
from datetime import date, datetime

import markdown
from markupsafe import Markup, escape

from tubitak_rag.domain.answer import Answer, SourceView
from tubitak_rag.generation.schema import CITE_RE
from tubitak_rag.interfaces.web.schemas import SourceContext

_URL_RE = re.compile(r"https?://[^\s<>\"]+[^\s<>\".,;:!?)]")
_HEADING_RE = re.compile(r"<(/?)h([1-5])\b")

DOC_TYPE_LABELS = {
    "kanun": "Kanun",
    "yonetmelik": "Yönetmelik",
    "uygulama_esaslari": "Uygulama esasları",
    "yonerge": "Yönerge",
    "esaslar": "Esaslar",
    "program_sayfasi": "Program sayfası",
    "cagri": "Çağrı",
    "kilavuz": "Kılavuz",
    "tablo": "Tablo",
    "diger": "Diğer",
}
DOCUMENT_STATE_LABELS = {
    "indeksli": "İndeksli",
    "taranmis": "Taranmış, metin yok",
    "eksik": "Dosya eksik",
    "guncel_degil": "Güncel değil",
}
LANGUAGE_LABELS = {"tr": "Türkçe", "en": "İngilizce"}


def number_tr(value: float | None, digits: int = 3) -> str:
    """Format with Turkish separators: ``1234.5 -> "1.234,500"``; None -> "—"."""
    if value is None:
        return "—"
    text = f"{value:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def signed_tr(value: float | None, digits: int = 3) -> str:
    """Like :func:`number_tr` with an explicit sign (typographic minus) for deltas."""
    if value is None:
        return "—"
    if value == 0:
        return number_tr(0.0, digits)
    sign = "+" if value > 0 else "−"
    return sign + number_tr(abs(value), digits)


def seconds_tr(ms: float) -> str:
    """Milliseconds as seconds with one decimal: ``7131.9 -> "7,1 sn"``."""
    return f"{number_tr(ms / 1000, 1)} sn"


def cited_sources(answer: Answer) -> list[SourceView]:
    """Sources the answer actually cites, in panel order (all of them if none is cited).

    The pipeline hands the LLM the top-n sources; the panel lists only those referenced by a
    citation, under their original numbers so the ``[n]`` in the text still match.
    """
    cited = {citation.no for citation in answer.atiflar}
    if not cited:
        return list(answer.kaynaklar)
    return [source for source in answer.kaynaklar if source.no in cited]


def location_parts(konum: str, context: SourceContext | None) -> list[str]:
    """Where a passage sits: article heading (or ``konum``), page, enclosing section."""
    if context is None:
        return [konum]
    parts = [context.madde_baslik or konum]
    if context.sayfa is not None and not konum.casefold().startswith("sayfa"):
        parts.append(f"sayfa {context.sayfa}")
    last_section = (context.hiyerarsi or "").split("›")[-1].strip()
    if context.hiyerarsi and last_section.casefold() not in konum.casefold():
        parts.append(context.hiyerarsi)
    return parts


def date_tr(value: str | date | None) -> str:
    """ISO date or timestamp -> ``"25.09.2026"`` / ``"15.10.2026 14:32"``; None -> "—"."""
    if value is None or value == "":
        return "—"
    if isinstance(value, date):
        return value.strftime("%d.%m.%Y")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return str(value)
    has_time = "T" in value or " " in value.strip()
    return parsed.strftime("%d.%m.%Y %H:%M" if has_time else "%d.%m.%Y")


def _paragraphs(html_text: str) -> Markup:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", html_text) if block.strip()]
    return Markup("".join(f"<p>{block.replace(chr(10), '<br>')}</p>" for block in blocks))


def _linkify(escaped: str) -> str:
    return _URL_RE.sub(lambda m: f'<a href="{m.group(0)}" rel="noopener">{m.group(0)}</a>', escaped)


def link_citations(text: str, valid: Iterable[int], prefix: str = "") -> Markup:
    """Escape ``text``, turn ``[n]`` into links to ``#kaynak-<prefix>n`` (known sources only).

    A grouped citation (``[1, 2]``, as the model sometimes writes it) becomes one link per
    number. Unknown numbers stay plain text so a link never points at a missing source.
    ``prefix`` keeps the anchors of several answers in one thread apart ("t1a2b3c-").
    """

    known = set(valid)

    def link(no: int) -> str:
        if no not in known:
            return f"[{no}]"
        return (
            f'<a class="atif" href="#kaynak-{prefix}{no}" data-kaynak="{no}" '
            f'aria-label="Kaynak {no}">[{no}]</a>'
        )

    def replace(match: re.Match[str]) -> str:
        numbers = [int(part) for part in match.group(1).split(",")]
        if not any(no in known for no in numbers):
            return match.group(0)
        return "".join(link(no) for no in numbers)

    escaped = _linkify(str(escape(text)))
    return _paragraphs(CITE_RE.sub(replace, escaped))


def plain_paragraphs(text: str) -> Markup:
    """Escape ``text`` and split it into paragraphs (for source excerpts)."""
    return _paragraphs(str(escape(text)))


def render_markdown(text: str, *, shift_headings: int = 0) -> Markup:
    """Render a run report to HTML: tables, fenced code; raw HTML is shown as text.

    Reports quote questions and model answers, so embedded HTML is never trusted.
    Tables are wrapped in a scroll container so wide tables never widen the page.
    """
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
    md.preprocessors.deregister("html_block")
    md.inlinePatterns.deregister("html")
    html_text = md.convert(text)
    if shift_headings:
        html_text = _HEADING_RE.sub(
            lambda m: f"<{m.group(1)}h{min(int(m.group(2)) + shift_headings, 6)}", html_text
        )
    html_text = html_text.replace("<table>", '<div class="tablo-kaydir"><table>').replace(
        "</table>", "</table></div>"
    )
    return Markup(html_text)
