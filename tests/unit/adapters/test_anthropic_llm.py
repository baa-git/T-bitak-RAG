"""Tests for the Anthropic LLM adapter with a fake ``anthropic`` module.

The project's tests never call the real Claude API; there is no integration test and no
API key. Every test here replaces ``sys.modules["anthropic"]``.
"""

import inspect
import sys
import types
from typing import Any

import pytest

from tubitak_rag.adapters.anthropic_llm import AnthropicLLM
from tubitak_rag.ports import LLM
from tubitak_rag.ports.llm import LlmError

SCHEMA = {"type": "object", "properties": {"cevap": {"type": "string"}}}


class FakeAPIError(Exception):
    """Stands in for ``anthropic.APIError``."""


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response = response
        self._error = error

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def _response(text: str = '{"cevap": "x"}', stop_reason: str = "end_turn") -> Any:
    blocks = [
        types.SimpleNamespace(type="thinking", thinking=""),
        types.SimpleNamespace(type="text", text=text),
    ]
    return types.SimpleNamespace(content=blocks, stop_reason=stop_reason)


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, messages: FakeMessages) -> None:
    module = types.ModuleType("anthropic")

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {}  # the key comes only from the environment (NFR-CONF2)
            self.beta = types.SimpleNamespace(messages=messages)

    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    module.APIError = FakeAPIError  # type: ignore[attr-defined]
    module.transform_schema = lambda schema: {**schema, "additionalProperties": False}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)


def test_is_an_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch, FakeMessages(_response()))
    llm = AnthropicLLM("claude-opus-5")
    assert isinstance(llm, LLM)
    assert llm.model_name == "claude-opus-5"


def test_uses_effort_not_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = FakeMessages(_response())
    _install_fake_sdk(monkeypatch, messages)
    AnthropicLLM("claude-opus-5", effort="low").generate("sistem", "soru", temperature=0.7)
    call = messages.calls[0]
    assert "temperature" not in call
    assert call["output_config"] == {"effort": "low"}
    assert call["model"] == "claude-opus-5"
    assert call["system"] == "sistem"
    assert call["messages"] == [{"role": "user", "content": "soru"}]
    assert call["max_tokens"] >= 4096


def test_json_schema_goes_to_output_config_format(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = FakeMessages(_response())
    _install_fake_sdk(monkeypatch, messages)
    AnthropicLLM().generate("s", "u", json_schema=SCHEMA)
    assert messages.calls[0]["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {**SCHEMA, "additionalProperties": False}},
    }


def test_server_side_refusal_fallback_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = FakeMessages(_response())
    _install_fake_sdk(monkeypatch, messages)
    AnthropicLLM().generate("s", "u")
    assert messages.calls[0]["fallbacks"] == "default"
    assert messages.calls[0]["betas"] == ["server-side-fallback-2026-07-01"]


def test_returns_text_blocks_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch, FakeMessages(_response('{"cevap": "evet"}')))
    assert AnthropicLLM().generate("s", "u") == '{"cevap": "evet"}'


def test_refusal_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch, FakeMessages(_response("", stop_reason="refusal")))
    with pytest.raises(LlmError, match="reddetti"):
        AnthropicLLM().generate("s", "u")


def test_max_tokens_stop_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch, FakeMessages(_response('{"cevap": "ya', "max_tokens")))
    with pytest.raises(LlmError, match="çıktı kesildi"):
        AnthropicLLM().generate("s", "u")


def test_installed_sdk_supports_the_parameters_we_send() -> None:
    """Contract check against the real SDK signature: no client, no key, no network."""
    anthropic = pytest.importorskip("anthropic")
    from anthropic.resources.beta.messages import Messages

    params = set(inspect.signature(Messages.create).parameters)
    assert {"output_config", "betas", "fallbacks", "system", "max_tokens"} <= params
    assert hasattr(anthropic, "transform_schema")
    assert hasattr(anthropic, "APIError")


def test_api_error_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch, FakeMessages(error=FakeAPIError("401 anahtar yok")))
    with pytest.raises(LlmError, match="Anthropic API isteği başarısız"):
        AnthropicLLM().generate("s", "u")


def test_missing_sdk_gives_turkish_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes "import anthropic" fail
    with pytest.raises(LlmError, match=r"pip install 'tubitak-rag\[api\]'"):
        AnthropicLLM()
