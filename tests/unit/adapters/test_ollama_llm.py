"""Tests for the Ollama chat LLM using ``httpx.MockTransport`` (no running Ollama)."""

import json
from typing import Any

import httpx
import pytest

from tubitak_rag.adapters.ollama_llm import OllamaLLM
from tubitak_rag.ports import LLM
from tubitak_rag.ports.llm import LlmError

SCHEMA = {"type": "object", "properties": {"cevap": {"type": "string"}}}


def _client(
    requests: list[dict[str, Any]],
    *,
    status: int = 200,
    content: str = '{"cevap": "x"}',
    done_reason: str = "stop",
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append({"url": str(request.url), **json.loads(request.content)})
        body = {"message": {"role": "assistant", "content": content}, "done_reason": done_reason}
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_truncated_output_raises_llm_error() -> None:
    llm = OllamaLLM(
        "http://o", "qwen3:4b", client=_client([], content='{"cevap": "ya', done_reason="length")
    )
    with pytest.raises(LlmError, match="çıktı kesildi"):
        llm.generate("s", "u")


def test_is_an_llm() -> None:
    llm = OllamaLLM("http://localhost:11434", "qwen3:4b", client=_client([]))
    assert isinstance(llm, LLM)
    assert llm.model_name == "qwen3:4b"


def test_request_body_has_think_false_format_and_temperature_zero() -> None:
    requests: list[dict[str, Any]] = []
    llm = OllamaLLM("http://ollama:11434/", "qwen3:4b", num_ctx=4096, client=_client(requests))
    llm.generate("sistem", "kullanıcı", json_schema=SCHEMA, temperature=0.0)
    assert requests == [
        {
            "url": "http://ollama:11434/api/chat",
            "model": "qwen3:4b",
            "messages": [
                {"role": "system", "content": "sistem"},
                {"role": "user", "content": "kullanıcı"},
            ],
            "stream": False,
            "think": False,
            "format": SCHEMA,
            "options": {"temperature": 0.0, "num_ctx": 4096},
        }
    ]


def test_format_omitted_without_schema() -> None:
    requests: list[dict[str, Any]] = []
    OllamaLLM("http://o", "qwen3:4b", client=_client(requests)).generate("s", "u")
    assert "format" not in requests[0]


def test_returns_message_content() -> None:
    llm = OllamaLLM("http://o", "qwen3:4b", client=_client([], content='{"cevap": "evet"}'))
    assert llm.generate("s", "u") == '{"cevap": "evet"}'


def test_connection_error_raises_llm_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("bağlantı reddedildi", request=request)

    client = httpx.Client(transport=httpx.MockTransport(refuse))
    with pytest.raises(LlmError, match="Ollama'ya ulaşılamadı"):
        OllamaLLM("http://o", "qwen3:4b", client=client).generate("s", "u")


def test_http_error_status_raises_llm_error_with_model_hint() -> None:
    llm = OllamaLLM("http://o", "qwen3:9b", client=_client([], status=404))
    with pytest.raises(LlmError, match=r"qwen3:9b.*ollama pull"):
        llm.generate("s", "u")
