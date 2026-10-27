"""Tests for the OpenAI-compatible LLM adapter with ``httpx.MockTransport`` only.

Project rule: no real provider (paid or free) is ever called; every response here is canned.
"""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tubitak_rag.adapters.openai_compat_llm import SCHEMA_INSTRUCTION, OpenAICompatLLM
from tubitak_rag.config import LlmConfig, OpenAICompatConfig
from tubitak_rag.factory import build_llm
from tubitak_rag.ports import LLM
from tubitak_rag.ports.llm import LlmError

SCHEMA = {"type": "object", "properties": {"cevap": {"type": "string"}}}
Handler = Callable[[httpx.Request], httpx.Response]


def _reply(content: str | None = '{"cevap": "x"}', finish_reason: str = "stop") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    return {"choices": [{"index": 0, "finish_reason": finish_reason, "message": message}]}


def _recording(
    requests: list[dict[str, Any]], responses: list[httpx.Response] | None = None
) -> Handler:
    """Record each request (url, headers, body); answer with ``responses`` in turn."""
    queue = list(responses or [])

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("Authorization"),
                "body": json.loads(request.content),
            }
        )
        return queue.pop(0) if queue else httpx.Response(200, json=_reply())

    return handler


def _llm(handler: Handler, **kwargs: Any) -> OpenAICompatLLM:
    kwargs.setdefault("environ", {})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatLLM("https://api.example.test/v1/", "model-x", client=client, **kwargs)


def test_is_an_llm() -> None:
    llm = _llm(_recording([]))
    assert isinstance(llm, LLM)
    assert llm.model_name == "model-x"


def test_request_has_json_schema_and_temperature_zero() -> None:
    requests: list[dict[str, Any]] = []
    _llm(_recording(requests), max_tokens=512).generate("sistem", "soru", json_schema=SCHEMA)
    (request,) = requests
    assert request["url"] == "https://api.example.test/v1/chat/completions"
    assert request["body"] == {
        "model": "model-x",
        "messages": [
            {"role": "system", "content": "sistem"},
            {"role": "user", "content": "soru"},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "cevap", "schema": SCHEMA, "strict": False},
        },
    }


def test_no_response_format_without_schema() -> None:
    requests: list[dict[str, Any]] = []
    _llm(_recording(requests)).generate("s", "u")
    assert "response_format" not in requests[0]["body"]
    assert requests[0]["body"]["messages"][0]["content"] == "s"


def test_api_key_read_from_the_named_environment_variable() -> None:
    requests: list[dict[str, Any]] = []
    llm = _llm(_recording(requests), api_key_env="BENIM_ANAHTAR", environ={"BENIM_ANAHTAR": "k1"})
    llm.generate("s", "u")
    assert requests[0]["auth"] == "Bearer k1"


def test_no_authorization_header_without_key() -> None:
    requests: list[dict[str, Any]] = []
    _llm(_recording(requests)).generate("s", "u")
    assert requests[0]["auth"] is None


def test_rejected_json_schema_falls_back_to_json_object_and_remembers() -> None:
    requests: list[dict[str, Any]] = []
    rejected = httpx.Response(400, json={"error": {"message": "response_format not supported"}})
    llm = _llm(_recording(requests, [rejected]))
    assert llm.generate("sistem", "u", json_schema=SCHEMA) == '{"cevap": "x"}'
    llm.generate("sistem", "u", json_schema=SCHEMA)
    formats = [r["body"].get("response_format", {}).get("type") for r in requests]
    assert formats == ["json_schema", "json_object", "json_object"]
    assert llm.json_mode == "json_object"
    system = requests[1]["body"]["messages"][0]["content"]
    assert system.startswith("sistem") and '"cevap"' in system  # schema in the prompt


def test_falls_back_to_plain_prompt_when_json_object_is_rejected_too() -> None:
    requests: list[dict[str, Any]] = []
    rejected = [
        httpx.Response(400, text="json_schema is not supported for this model"),
        httpx.Response(422, text="Unsupported value: response_format.type json_object"),
    ]
    llm = _llm(_recording(requests, rejected))
    llm.generate("s", "u", json_schema=SCHEMA)
    assert "response_format" not in requests[-1]["body"]
    assert requests[-1]["body"]["messages"][0]["content"] == "s" + SCHEMA_INSTRUCTION.format(
        schema=json.dumps(SCHEMA, ensure_ascii=False)
    )
    assert llm.json_mode == "prompt"


def test_unrelated_bad_request_is_not_a_fallback() -> None:
    requests: list[dict[str, Any]] = []
    bad = httpx.Response(400, text="context length exceeded")
    llm = _llm(_recording(requests, [bad]))
    with pytest.raises(LlmError, match="HTTP 400"):
        llm.generate("s", "u", json_schema=SCHEMA)
    assert len(requests) == 1 and llm.json_mode == "json_schema"


def test_configured_prompt_mode_never_sends_response_format() -> None:
    requests: list[dict[str, Any]] = []
    _llm(_recording(requests), json_mode="prompt").generate("s", "u", json_schema=SCHEMA)
    assert "response_format" not in requests[0]["body"]


def test_unauthorized_names_the_variable_without_leaking_the_key() -> None:
    denied = httpx.Response(401, json={"error": "invalid api key"})
    llm = _llm(_recording([], [denied]), environ={"LLM_API_KEY": "gizli-123"})
    with pytest.raises(LlmError) as error:
        llm.generate("s", "u")
    assert "LLM_API_KEY" in str(error.value) and "gizli-123" not in str(error.value)


def test_missing_key_is_reported_on_401() -> None:
    llm = _llm(_recording([], [httpx.Response(401, text="no key")]))
    with pytest.raises(LlmError, match="API anahtarı yok"):
        llm.generate("s", "u")


def test_rate_limit_has_a_retry_hint() -> None:
    llm = _llm(_recording([], [httpx.Response(429, text="slow down")]))
    with pytest.raises(LlmError, match="istek sınırı"):
        llm.generate("s", "u")


def test_not_found_points_at_model_and_url() -> None:
    llm = _llm(_recording([], [httpx.Response(404, text="model not found")]))
    with pytest.raises(LlmError, match=r"llm\.model"):
        llm.generate("s", "u")


def test_truncated_output_raises() -> None:
    llm = _llm(_recording([], [httpx.Response(200, json=_reply('{"cevap": "ya', "length"))]))
    with pytest.raises(LlmError, match="çıktı kesildi"):
        llm.generate("s", "u", json_schema=SCHEMA)


def test_empty_content_and_refusal_raise() -> None:
    empty = httpx.Response(200, json=_reply(None))
    refused = _reply(None)
    refused["choices"][0]["message"]["refusal"] = "yapamam"
    llm = _llm(_recording([], [empty, httpx.Response(200, json=refused)]))
    with pytest.raises(LlmError, match="boş yanıt"):
        llm.generate("s", "u")
    with pytest.raises(LlmError, match="reddetti"):
        llm.generate("s", "u")


def test_malformed_body_raises() -> None:
    llm = _llm(_recording([], [httpx.Response(200, json={"nope": []})]))
    with pytest.raises(LlmError, match="beklenen biçimde değil"):
        llm.generate("s", "u")


def test_connection_error_raises_llm_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("bağlantı reddedildi", request=request)

    with pytest.raises(LlmError, match="ulaşılamadı"):
        _llm(refuse).generate("s", "u")


def test_invalid_json_mode_is_refused() -> None:
    with pytest.raises(LlmError, match="json_mode"):
        _llm(_recording([]), json_mode="xml")  # type: ignore[arg-type]


def test_factory_builds_it_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UCRETSIZ_ANAHTAR", "k")
    cfg = LlmConfig(
        provider="openai_compat",
        model="model-y",
        timeout_s=7,
        openai_compat=OpenAICompatConfig(
            base_url="http://localhost:11434/v1",
            api_key_env="UCRETSIZ_ANAHTAR",
            json_mode="json_object",
            max_tokens=99,
        ),
    )
    llm = build_llm(cfg)
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.model_name == "model-y" and llm.json_mode == "json_object"
