"""Tests for the llm port."""

from tests.fakes.llm import FakeLLM
from tubitak_rag.ports.llm import LLM, LlmError


def test_fake_llm_satisfies_protocol() -> None:
    assert isinstance(FakeLLM(["{}"]), LLM)


def test_plain_object_is_not_an_llm() -> None:
    assert not isinstance(object(), LLM)


def test_llm_error_is_a_runtime_error() -> None:
    assert issubclass(LlmError, RuntimeError)
