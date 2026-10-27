"""What the Gradio pages show, as escaped HTML (no Gradio import; unit-tested directly).

The conversation thread is one HTML block laid out like the web page's turns
(``templates/partials/tur.html``): the question, the answer with ``[n]`` links, and the
numbered sources with the original document link and an expandable quote. Texts and the
answer/refusal logic come from ``content.py``, which reuses the web helpers, so both
interfaces say the same thing. Every value that comes from a model, a config or a manifest
is escaped here.
"""

from collections.abc import Iterable, Sequence

from markupsafe import Markup

from tubitak_rag.interfaces.gradio_app import content as shared
from tubitak_rag.interfaces.gradio_app.content import (
    ABOUT_INTRO,
    CHAT_INTRO,
    CHAT_TITLE,
    HOW_IT_WORKS,
    PLACEHOLDER,
    TITLE,
    error_html,
    footer_html,
    question_html,
)
from tubitak_rag.interfaces.web.schemas import AboutInfo, AskResult, PipelineOption

__all__ = [  # texts re-exported from content.py, so the app imports one module
    "ABOUT_INTRO",
    "CHAT_INTRO",
    "CHAT_TITLE",
    "EXAMPLES_HEADING",
    "HOW_IT_WORKS",
    "PENDING",
    "PLACEHOLDER",
    "TITLE",
    "about_html",
    "chat_header_html",
    "disclaimer_html",
    "footer_html",
    "methods_html",
    "option_choices",
    "page_footer_html",
    "pending_turn_html",
    "thread_html",
    "turn_html",
]

PENDING = "Cevap hazırlanıyor…"
EXAMPLES_HEADING = "Örnek sorular"


def _sources(reply: shared.AnswerView, prefix: str) -> Markup:
    items = []
    for source in reply.sources:
        repair = (
            Markup('<p class="kaynak-onarim">guardrail tarafından eklendi</p>')
            if source.repaired
            else ""
        )
        items.append(
            Markup(
                '<li id="{anchor}" class="kaynak"><span class="kaynak-no">[{no}]</span>'
                '<div class="kaynak-icerik"><p class="kaynak-belge">{title}</p>'
                '<p class="kaynak-konum">{location}</p>{repair}'
                '<p class="kaynak-baglantilar"><a href="{url}" rel="noopener" target="_blank">'
                "Belgenin aslı</a></p>"
                "<details><summary>Alıntıyı göster</summary>"
                '<blockquote class="kaynak-metni">{quote}</blockquote></details></div></li>'
            ).format(
                anchor=source.anchor,
                no=source.no,
                title=source.title,
                location=source.location,
                repair=repair,
                url=source.url,
                quote=source.quote_html,
            )
        )
    return Markup(
        '<section class="kaynaklar" aria-labelledby="{p}kaynaklar">'
        '<h3 id="{p}kaynaklar">Kaynaklar</h3><ol class="kaynak-listesi">{items}</ol></section>'
    ).format(p=prefix, items=Markup("").join(items))


def turn_html(
    question: str,
    result: AskResult | None,
    *,
    prefix: str,
    methods: str = "",
    error: str | None = None,
) -> str:
    """One turn: the question, then the answer with sources, a refusal or an inline error."""
    if error is not None or result is None:
        reply = Markup('<div class="mesaj cevap-mesaji hata">{}</div>').format(
            error_html(error or "Beklenmeyen bir hata oluştu.")
        )
    else:
        view = shared.answer_view(result, prefix, methods)
        sources = _sources(view, prefix) if view.sources else ""
        css = "mesaj cevap-mesaji" + ("" if view.answered else " ret")
        reply = Markup('<article class="{}" aria-label="Cevap">{}{}{}</article>').format(
            css, view.header_html, view.body_html, sources
        )
    return str(
        Markup('<section class="tur" aria-label="Soru ve cevap">{}{}</section>').format(
            question_html(question), reply
        )
    )


def pending_turn_html(question: str) -> str:
    """The question with a "being prepared" note, shown until the answer arrives."""
    return str(
        Markup(
            '<section class="tur bekliyor" aria-busy="true">{}'
            '<p class="bekleme" role="status">{}</p></section>'
        ).format(question_html(question), PENDING)
    )


def thread_html(turns: Sequence[str], pending: str | None = None) -> str:
    """The whole conversation; with no turn yet, the short "how it works" note."""
    blocks = [Markup(turn) for turn in turns]  # each turn was escaped when it was rendered
    if pending is not None:
        blocks.append(Markup(pending_turn_html(pending)))
    if not blocks:
        blocks = [Markup('<p class="nasil">{}</p>').format(HOW_IT_WORKS)]
    return str(Markup('<div class="sohbet">{}</div>').format(Markup("").join(blocks)))


def chat_header_html() -> str:
    """Page title and the one-line introduction above the thread."""
    return str(
        Markup('<header class="sayfa-ust"><h1>{}</h1><p class="giris">{}</p></header>').format(
            CHAT_TITLE, CHAT_INTRO
        )
    )


def option_choices(options: Iterable[PipelineOption]) -> list[tuple[str, str]]:
    """``(label, name)`` pairs for the configuration dropdown ("v1 (indeks yok)")."""
    return [(shared.option_label(option), option.ad) for option in options]


def methods_html(option: PipelineOption | None, *, cpu_only: bool = False) -> str:
    """The plain method line under the selector, plus the CPU speed note when it applies."""
    if option is None:
        return ""
    html = Markup('<p class="yontem">{}</p>').format(shared.methods_line(option))
    note = shared.speed_note(option, cpu_only)
    if note:
        html += Markup('<p class="hiz-notu">{}</p>').format(note)
    return str(html)


def disclaimer_html(text: str) -> str:
    """The disclaimer every answer carries, shown once under the question box."""
    return str(Markup('<p class="alt-not">{}</p>').format(text))


def about_html(info: AboutInfo) -> str:
    """The Hakkında tab: introduction and the web page's sections."""
    sections = Markup("").join(
        Markup('<section class="hakkinda-bolum"><h2>{}</h2>{}</section>').format(heading, body)
        for heading, body in shared.about_sections(info)
    )
    return str(
        Markup('<div class="hakkinda"><h1>Hakkında</h1><p class="giris">{}</p>{}</div>').format(
            ABOUT_INTRO, sections
        )
    )


def page_footer_html(is_demo: bool) -> str:
    """The footer (and the demo notice of the fake backend)."""
    return str(footer_html(is_demo))
