"""The conversation behind the Gradio page: a question in, the updated thread out (no Gradio).

The conversation of a visitor is a list of rendered turns (HTML strings) kept in a Gradio
``State``. Plain strings travel well: on a ZeroGPU Space the answer function runs in a forked
GPU worker and its arguments and results are pickled between the processes.
"""

import logging
import os
import secrets
from collections.abc import Mapping, Sequence

from tubitak_rag.config import ConfigError
from tubitak_rag.interfaces.gpu_lock import GpuLockError
from tubitak_rag.interfaces.gradio_app import content as shared
from tubitak_rag.interfaces.gradio_app import view
from tubitak_rag.interfaces.web.backend import (
    ServiceUnavailableError,
    UnknownPipelineError,
    WebBackend,
)
from tubitak_rag.interfaces.web.schemas import MAX_QUESTION_CHARS, PipelineOption

logger = logging.getLogger(__name__)

DURATION_ENV = "TUBITAK_RAG_GPU_DURATION"
# ZeroGPU reserves the GPU for this long per question; a visitor's daily quota must cover it,
# so it is kept close to the worst case: retrieval + reranking take well under a second on
# the GPU, the LLM call is bounded by llm.transformers.max_new_tokens (768 on the Space).
DEFAULT_DURATION_S = 40
RERANKER_EXTRA_S = 5
EXPECTED_ERRORS = (GpuLockError, ServiceUnavailableError, UnknownPipelineError, ConfigError)

Turns = list[str]


def _base_duration(environ: Mapping[str, str]) -> int:
    raw = environ.get(DURATION_ENV, "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_DURATION_S


class Chat:
    """Answers questions with a ``WebBackend`` and renders the thread of one visitor."""

    def __init__(
        self,
        backend: WebBackend,
        *,
        cpu_only: bool = False,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.backend = backend
        self.cpu_only = cpu_only
        self.options: list[PipelineOption] = backend.pipelines()
        self._by_name = {option.ad: option for option in self.options}
        self.default = shared.default_option(self.options)
        self._base_duration = _base_duration(os.environ if environ is None else environ)

    # --- selector ------------------------------------------------------------------------

    def choices(self) -> list[tuple[str, str]]:
        """``(label, name)`` pairs of the configuration dropdown."""
        return view.option_choices(self.options)

    def methods(self, config: str | None) -> str:
        """Method line (and speed note) of the selected configuration."""
        return view.methods_html(self._by_name.get(config or ""), cpu_only=self.cpu_only)

    # --- conversation --------------------------------------------------------------------

    def pending(self, question: str, turns: Sequence[str]) -> str:
        """The thread with the new question and a "being prepared" note (shown at once)."""
        question = (question or "").strip()
        return view.thread_html(turns, pending=question or None)

    def answer(self, question: str, config: str, turns: Turns) -> tuple[str, Turns, str]:
        """Answer ``question`` with the configuration ``config`` and append it to the thread.

        Args:
            question: the visitor's question (at most 1000 characters).
            config: name of the experiment configuration (v0 … v3b).
            turns: the turns rendered so far.

        Returns:
            The thread HTML, the new list of turns and an empty string for the question box.
        """
        question = (question or "").strip()
        if not question:
            return view.thread_html(turns), list(turns), ""
        prefix = f"t{secrets.token_hex(3)}-"
        option = self._by_name.get(config or "")
        methods = shared.methods_line(option) if option is not None else ""
        result, error = None, None
        if len(question) > MAX_QUESTION_CHARS:
            error = f"Soru en fazla {MAX_QUESTION_CHARS} karakter olabilir."
        else:
            try:
                result = self.backend.ask(question, None, None, config or None)
            except EXPECTED_ERRORS as exc:
                error = str(exc)
            except Exception as exc:  # a crashed question must not take the page down
                logger.exception("gradio: soru cevaplanamadı")
                error = f"Beklenmeyen bir hata oluştu: {exc}"
        turn = view.turn_html(question, result, prefix=prefix, methods=methods, error=error)
        new_turns = [*turns, turn]
        return view.thread_html(new_turns), new_turns, ""

    def gpu_duration(self, question: str, config: str, turns: Turns) -> int:
        """Seconds of GPU time to reserve for one answer (same arguments as :meth:`answer`)."""
        option = self._by_name.get(config or "")
        reranker = option is not None and shared.uses_reranker(option.yontemler)
        return self._base_duration + (RERANKER_EXTRA_S if reranker else 0)
