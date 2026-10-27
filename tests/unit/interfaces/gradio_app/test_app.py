"""Smoke tests of the Gradio page with the demo backend: built, never launched."""

import runpy
from pathlib import Path
from typing import Any

import gradio as gr
import pytest

from tubitak_rag.interfaces.gradio_app import app as gradio_app
from tubitak_rag.interfaces.gradio_app.chat import Chat
from tubitak_rag.interfaces.web.fake_backend import EXAMPLES, FakeBackend


def _by_type(demo: gr.Blocks, kind: str) -> list[dict[str, Any]]:
    return [c for c in demo.get_config_file()["components"] if c["type"] == kind]


def _html(demo: gr.Blocks) -> str:
    return "\n".join(str(c["props"].get("value", "")) for c in _by_type(demo, "html"))


def test_page_has_two_tabs_selector_examples_and_disclaimer() -> None:
    demo = gradio_app.create_app(Chat(FakeBackend(), environ={}))
    assert isinstance(demo, gr.Blocks)
    tabs = [c["props"]["label"] for c in _by_type(demo, "tabitem")]
    assert tabs == ["Sohbet", "Hakkında"]
    (dropdown,) = _by_type(demo, "dropdown")
    assert dropdown["props"]["value"] == "v3" and dropdown["props"]["label"] == "Yapılandırma"
    assert [c[1] for c in dropdown["props"]["choices"]] == ["v0", "v0r", "v1", "v2", "v3", "v3b"]
    buttons = [c["props"]["value"] for c in _by_type(demo, "button")]
    assert buttons == [*EXAMPLES, "Sor", "Yeni sohbet"]
    html = _html(demo)
    assert "Mevzuata soru sorun" in html and "resmî görüş değildir" in html
    assert "Reranker: BAAI/bge-reranker-v2-m3" in html and "Kullanılan belgeler" in html
    assert "demo-notu" in html


def test_answer_function_is_bound_to_every_way_of_asking() -> None:
    chat = Chat(FakeBackend(), environ={})

    def answer(question: str, config: str, turns: list[str]) -> tuple[str, list[str], str]:
        return chat.answer(question, config, turns)

    demo = gradio_app.create_app(chat, answer)
    bound = [fn for fn in demo.fns.values() if fn.fn is answer]
    assert len(bound) == 2 + len(EXAMPLES)  # Enter, "Sor", one per example question


def test_launch_options_theme_and_stylesheet() -> None:
    options = gradio_app.launch_options()
    assert isinstance(options["theme"], gr.themes.Base)
    assert "--vurgu: #1f3a5f" in options["css"] and ".kaynak-listesi" in options["css"]
    assert "Source+Serif+4" in options["head"] and "dark" in options["js"]


def test_build_with_the_demo_backend_and_with_a_broken_config(tmp_path: Path) -> None:
    demo = gradio_app.build({"TUBITAK_RAG_BACKEND": "fake"})
    assert _by_type(demo, "dropdown")
    broken = gradio_app.build({"TUBITAK_RAG_CONFIG": str(tmp_path / "yok.yaml")})
    assert "Uygulama başlatılamadı" in _html(broken) and not _by_type(broken, "dropdown")


def test_entry_file_builds_the_demo(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TUBITAK_RAG_BACKEND", "fake")
    monkeypatch.delenv("SPACE_ID", raising=False)
    namespace = runpy.run_path(str(repo_root / "app_gradio.py"), run_name="app_gradio")
    assert isinstance(namespace["demo"], gr.Blocks)
