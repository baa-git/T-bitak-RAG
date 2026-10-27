"""The Gradio app: Sohbet (conversation with numbered sources) and Hakkında.

Start it with ``python app_gradio.py`` from the repository root (on the Space the same file
is ``app.py``). :func:`create_app` only lays out the page; the backend and every model are
created before it (:func:`main`), because on a ZeroGPU Space models must be loaded at start
and only the answer function (wrapped by ``spaces.GPU``) runs on the GPU.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr

from tubitak_rag.config import ConfigError
from tubitak_rag.interfaces.gradio_app import view
from tubitak_rag.interfaces.gradio_app.chat import Chat, Turns
from tubitak_rag.interfaces.gradio_app.gpu import gpu_task
from tubitak_rag.interfaces.gradio_app.settings import (
    build_backend,
    settings_from_env,
)
from tubitak_rag.interfaces.space_bundle import SpaceIndexError
from tubitak_rag.interfaces.web.fake_backend import EXAMPLES
from tubitak_rag.interfaces.web.schemas import MAX_QUESTION_CHARS

logger = logging.getLogger(__name__)

STYLE_FILE = Path(__file__).with_name("style.css")
AnswerFn = Callable[[str, str, Turns], tuple[str, Turns, str]]

# Source Serif 4 for headings and answers, as on the web page (OFL, Google Fonts).
HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:'
    'opsz,wght@8..60,400..700&display=swap">'
)
# The page is designed for its light paper palette; Gradio would follow the OS dark mode.
FORCE_LIGHT = "() => { document.body.classList.remove('dark'); }"
SCROLL_TO_LAST_TURN = (
    "() => { const t = document.querySelectorAll('#sohbet .tur');"
    " if (t.length) t[t.length - 1].scrollIntoView({behavior: 'smooth', block: 'start'}); }"
)


def theme() -> gr.themes.Base:
    """Warm paper ground, charcoal ink and one navy accent; no shadows, no gradients."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_sm,
        font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        font_mono=["Cascadia Mono", "SF Mono", "Consolas", "monospace"],
    ).set(
        body_background_fill="#f6f4ef",
        body_text_color="#1e1e1c",
        body_text_color_subdued="#5c5850",
        background_fill_primary="#fdfcf9",
        background_fill_secondary="#ece8df",
        block_background_fill="#fdfcf9",
        block_border_color="#ddd8ce",
        block_border_width="1px",
        block_shadow="none",
        block_label_text_color="#5c5850",
        border_color_primary="#ddd8ce",
        border_color_accent="#1f3a5f",
        input_background_fill="#fdfcf9",
        input_border_color="#8a8478",
        input_border_color_focus="#1f3a5f",
        input_shadow="none",
        input_shadow_focus="none",
        button_primary_background_fill="#1f3a5f",
        button_primary_background_fill_hover="#142840",
        button_primary_text_color="#ffffff",
        button_primary_border_color="#1f3a5f",
        button_secondary_background_fill="#fdfcf9",
        button_secondary_background_fill_hover="#efece5",
        button_secondary_text_color="#1e1e1c",
        button_secondary_border_color="#ddd8ce",
        button_primary_shadow="none",
        button_primary_shadow_hover="none",
        button_secondary_shadow="none",
        button_secondary_shadow_hover="none",
        shadow_drop="none",
        shadow_drop_lg="none",
        link_text_color="#1f3a5f",
        link_text_color_hover="#142840",
        color_accent="#1f3a5f",
        color_accent_soft="#e3e8ef",
    )


def launch_options() -> dict[str, Any]:
    """Theme, stylesheet, fonts and the light-mode switch (``Blocks.launch`` in Gradio 6)."""
    return {
        "theme": theme(),
        "css": STYLE_FILE.read_text(encoding="utf-8"),
        "head": HEAD,
        "js": FORCE_LIGHT,
    }


def _html(value: str, **kwargs: Any) -> gr.HTML:
    return gr.HTML(value, apply_default_css=False, js_on_load=None, **kwargs)


def create_app(chat: Chat, answer: AnswerFn | None = None, *, about_html: str = "") -> gr.Blocks:
    """Lay out the two tabs around ``chat``; ``answer`` defaults to ``chat.answer``.

    On the Space ``answer`` is ``chat.answer`` wrapped by ``spaces.GPU`` (:func:`main`).
    Nothing is answered or loaded here, so the page can be built in tests without models.
    """
    answer_fn = answer or chat.answer
    demo = gr.Blocks(title=view.TITLE)
    with demo:
        turns = gr.State([])
        with gr.Tabs():
            with gr.Tab("Sohbet", id="sohbet-sekmesi"):
                _html(view.chat_header_html())
                thread = _html(view.thread_html([]), elem_id="sohbet")
                with gr.Column(elem_id="ornekler") as examples:
                    _html(f'<h2 class="ornek-baslik">{view.EXAMPLES_HEADING}</h2>')
                    example_buttons = [
                        gr.Button(text, variant="secondary", elem_classes=["ornek"])
                        for text in EXAMPLES
                    ]
                with gr.Group(elem_id="yazma"):
                    question = gr.Textbox(
                        label="Sorunuz",
                        show_label=False,
                        placeholder=view.PLACEHOLDER,
                        lines=1,  # Enter sends (multi-line boxes send with Shift+Enter)
                        max_lines=6,
                        max_length=MAX_QUESTION_CHARS,
                        elem_id="soru",
                    )
                    with gr.Row(elem_id="dugmeler"):
                        ask = gr.Button("Sor", variant="primary", scale=0, min_width=120)
                        clear = gr.Button("Yeni sohbet", variant="secondary", scale=0)
                with gr.Row(elem_id="secici"):
                    config = gr.Dropdown(
                        choices=chat.choices(),
                        value=chat.default,
                        label="Yapılandırma",
                        interactive=True,
                        scale=0,
                        min_width=180,
                    )
                    methods = _html(chat.methods(chat.default), elem_id="yontem")
                _html(view.disclaimer_html(chat.backend.disclaimer()))
            with gr.Tab("Hakkında", id="hakkinda-sekmesi"):
                _html(about_html or view.about_html(chat.backend.about()), elem_id="hakkinda")
        _html(view.page_footer_html(chat.backend.is_demo))

        def hide_examples(current: Turns) -> Any:
            return gr.update(visible=not current)

        def then_answer(shown: Any) -> None:
            """After the question is shown as pending: answer, hide the examples, scroll."""
            (
                shown.then(answer_fn, [question, config, turns], [thread, turns, question])
                .then(hide_examples, turns, examples, queue=False)
                .then(None, js=SCROLL_TO_LAST_TURN)
            )

        shown = [question, turns]
        then_answer(question.submit(chat.pending, shown, thread, queue=False))
        then_answer(ask.click(chat.pending, shown, thread, queue=False))
        for button, text in zip(example_buttons, EXAMPLES, strict=True):
            filled = button.click(lambda t=text: t, None, question, queue=False)
            then_answer(filled.then(chat.pending, shown, thread, queue=False))
        config.change(chat.methods, config, methods, queue=False)
        clear.click(
            lambda: ([], view.thread_html([]), gr.update(visible=True)),
            None,
            [turns, thread, examples],
            queue=False,
        )
    return demo


def _error_app(message: str) -> gr.Blocks:
    demo = gr.Blocks(title=view.TITLE)
    with demo:
        _html(f"<h1>{view.TITLE}</h1>")
        _html(str(view.error_html(f"Uygulama başlatılamadı: {message}")))
    return demo


def build(environ: dict[str, str] | None = None) -> gr.Blocks:
    """Create the backend (every model loaded), wrap the answer for ZeroGPU, lay out the page."""
    env = dict(os.environ) if environ is None else environ
    try:
        settings = settings_from_env(env)
        backend = build_backend(settings, env)
    except (ConfigError, SpaceIndexError) as exc:
        logger.exception("gradio: uygulama başlatılamadı")
        return _error_app(str(exc))
    chat = Chat(backend, cpu_only=settings.cpu_only, environ=env)
    answer = gpu_task(chat.answer, duration=chat.gpu_duration)
    return create_app(chat, answer, about_html=view.about_html(backend.about()))


def main() -> None:
    """Build the app and serve it (Gradio's queue keeps one answer at a time)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build().launch(**launch_options())
