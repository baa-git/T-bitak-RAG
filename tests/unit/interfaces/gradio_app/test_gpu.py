"""The ZeroGPU shim: ``spaces.GPU`` when installed, the function itself otherwise."""

import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

from tubitak_rag.interfaces.gradio_app.gpu import gpu_task


def _answer(question: str) -> str:
    return question.upper()


def test_without_spaces_the_function_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "spaces", None)  # the import raises ImportError
    assert gpu_task(_answer, duration=40) is _answer


def test_with_spaces_the_function_is_wrapped_with_the_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def gpu(*, duration: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        seen["duration"] = duration

        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            seen["fn"] = fn
            return lambda *args: ("gpu", fn(*args))

        return wrap

    module = types.ModuleType("spaces")
    module.GPU = gpu  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spaces", module)

    def duration(question: str) -> int:
        return 45

    wrapped = gpu_task(_answer, duration=duration)
    assert wrapped("soru") == ("gpu", "SORU")
    assert seen == {"duration": duration, "fn": _answer}
