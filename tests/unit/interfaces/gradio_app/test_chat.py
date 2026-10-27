"""Gradio conversation: questions in, rendered thread out; GPU time per configuration."""

import pytest

from tubitak_rag.interfaces.gradio_app import view
from tubitak_rag.interfaces.gradio_app.chat import (
    DEFAULT_DURATION_S,
    DURATION_ENV,
    RERANKER_EXTRA_S,
    Chat,
)
from tubitak_rag.interfaces.web.backend import ServiceUnavailableError
from tubitak_rag.interfaces.web.fake_backend import (
    EXAMPLE_ANSWERED,
    EXAMPLE_OUT_OF_SCOPE,
    FakeBackend,
)
from tubitak_rag.interfaces.web.schemas import MAX_QUESTION_CHARS, AskResult


class _Failing(FakeBackend):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def ask(
        self,
        soru: str,
        program_kodu: str | None,
        doc_turu: str | None,
        yapilandirma: str | None = None,
    ) -> AskResult:
        raise self.error


def test_selector_choices_default_and_methods() -> None:
    chat = Chat(FakeBackend(), environ={})
    assert [name for _, name in chat.choices()] == ["v0", "v0r", "v1", "v2", "v3", "v3b"]
    assert chat.default == "v3"
    assert "Reranker: BAAI/bge-reranker-v2-m3" in chat.methods("v3")
    assert "Reranker: yok" in chat.methods("v2")
    assert chat.methods("v9") == ""
    assert "hiz-notu" in Chat(FakeBackend(), cpu_only=True, environ={}).methods("v3")


def test_answer_appends_a_turn_and_clears_the_box() -> None:
    backend = FakeBackend()
    chat = Chat(backend, environ={})
    html, turns, box = chat.answer(f"  {EXAMPLE_ANSWERED} ", "v2", [])
    assert box == "" and len(turns) == 1 and html == view.thread_html(turns)
    assert backend.asked == [(EXAMPLE_ANSWERED, None, None, "v2")]
    assert 'class="kaynak-listesi"' in turns[0] and ">v2<" in turns[0]
    html, more, _ = chat.answer(EXAMPLE_OUT_OF_SCOPE, "v3", turns)
    assert len(more) == 2 and more[0] == turns[0] and "Bu araç yalnızca TÜBİTAK" in html


def test_each_turn_gets_its_own_anchor_prefix() -> None:
    chat = Chat(FakeBackend(), environ={})
    _, turns, _ = chat.answer(EXAMPLE_ANSWERED, "v3", [])
    _, turns, _ = chat.answer(EXAMPLE_ANSWERED, "v3", turns)
    first, second = (turn.split('id="kaynak-')[1].split('"')[0] for turn in turns)
    assert first != second


def test_empty_question_changes_nothing() -> None:
    backend = FakeBackend()
    html, turns, box = Chat(backend, environ={}).answer("   ", "v3", ["<p>eski</p>"])
    assert turns == ["<p>eski</p>"] and "eski" in html and box == ""
    assert backend.asked == []


def test_too_long_question_is_refused_inline() -> None:
    backend = FakeBackend()
    _, turns, _ = Chat(backend, environ={}).answer("a" * (MAX_QUESTION_CHARS + 1), "v3", [])
    assert "en fazla 1000 karakter" in turns[0] and backend.asked == []


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ServiceUnavailableError("v1 yapılandırmasının indeks yok"), "indeks yok"),
        (RuntimeError("CUDA kapalı"), "Beklenmeyen bir hata oluştu: CUDA kapalı"),
    ],
)
def test_backend_errors_become_inline_errors(error: Exception, message: str) -> None:
    _, turns, _ = Chat(_Failing(error), environ={}).answer("soru", "v3", [])
    assert "form-hatasi" in turns[0] and message in turns[0]


def test_pending_shows_the_question_at_once() -> None:
    chat = Chat(FakeBackend(), environ={})
    assert view.PENDING in chat.pending(" soru ", [])
    assert view.PENDING not in chat.pending("  ", [])


def test_gpu_duration_depends_on_the_reranker_and_the_environment() -> None:
    chat = Chat(FakeBackend(), environ={})
    assert chat.gpu_duration("s", "v2", []) == DEFAULT_DURATION_S
    assert chat.gpu_duration("s", "v3", []) == DEFAULT_DURATION_S + RERANKER_EXTRA_S
    assert chat.gpu_duration("s", "v9", []) == DEFAULT_DURATION_S
    custom = Chat(FakeBackend(), environ={DURATION_ENV: "30"})
    assert custom.gpu_duration("s", "v3b", []) == 30 + RERANKER_EXTRA_S
    invalid = Chat(FakeBackend(), environ={DURATION_ENV: "abc"})
    assert invalid.gpu_duration("s", "v2", []) == DEFAULT_DURATION_S


def test_turns_are_plain_strings() -> None:
    """ZeroGPU pickles arguments and results between processes: keep them plain."""
    _, turns, _ = Chat(FakeBackend(), environ={}).answer(EXAMPLE_ANSWERED, "v3", [])
    assert all(type(turn) is str for turn in turns)
