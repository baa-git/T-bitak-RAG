"""LLM over any OpenAI-compatible ``POST /chat/completions`` endpoint (selectable by config).

Hosted free tiers (Groq, OpenRouter, Hugging Face Inference Providers, Gemini's OpenAI
endpoint) and local servers (Ollama's ``/v1``) speak the same request format but differ in
how much of it they support. Structured output is the main difference, so it is requested in
the strongest form first and degraded when the provider rejects it:

1. ``json_schema`` — ``response_format: {"type": "json_schema", ...}`` (constrained decoding);
2. ``json_object`` — ``response_format: {"type": "json_object"}`` plus the schema in the prompt;
3. ``prompt`` — no ``response_format``; the schema is described in the system prompt and the
   reply is parsed tolerantly (``generation.schema.parse_json_reply`` strips fences/prose).

A mode that was rejected once (HTTP 400/422 naming the response format) is not tried again
for the lifetime of the adapter. The API key is read from the environment variable named in
the config (``llm.openai_compat.api_key_env``) and is never logged or written anywhere.
Project rule: tests use ``httpx.MockTransport`` only; no real provider is called.
"""

import logging
import os
from collections.abc import Mapping
from typing import Any, Literal

import httpx

from tubitak_rag.adapters.json_prompt import SCHEMA_INSTRUCTION, with_schema_instruction
from tubitak_rag.ports.llm import LlmError

logger = logging.getLogger(__name__)

JsonMode = Literal["json_schema", "json_object", "prompt"]
JSON_MODES: tuple[JsonMode, ...] = ("json_schema", "json_object", "prompt")
_NEXT_MODE: dict[JsonMode, JsonMode] = {"json_schema": "json_object", "json_object": "prompt"}
_REJECTED_STATUS = {400, 422}
# Words of a provider error that blame the requested output format (Groq, OpenRouter, HF,
# Gemini and vLLM phrase it differently; all mention one of these).
_FORMAT_WORDS = ("response_format", "json", "schema", "structured")
_AUTH_STATUS = {401, 403}
_NOT_FOUND = 404
_RATE_LIMIT = 429
_ERROR_TEXT_LIMIT = 300
SCHEMA_NAME = "cevap"
# Re-exported: the prompt-mode instruction lives in ``json_prompt`` (shared with transformers).
__all__ = ["JSON_MODES", "SCHEMA_INSTRUCTION", "OpenAICompatLLM"]


def _format_rejected(response: httpx.Response) -> bool:
    """Whether an error response says the requested ``response_format`` is unsupported."""
    if response.status_code not in _REJECTED_STATUS:
        return False
    text = response.text.casefold()
    return any(word in text for word in _FORMAT_WORDS)


class OpenAICompatLLM:
    """Non-streaming chat completions with temperature control and JSON output fallbacks."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "LLM_API_KEY",
        json_mode: JsonMode = "json_schema",
        max_tokens: int = 2048,
        timeout_s: float = 120,
        client: httpx.Client | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if json_mode not in JSON_MODES:
            raise LlmError(f"Geçersiz json_mode: {json_mode!r} (seçenekler: {JSON_MODES})")
        env = os.environ if environ is None else environ
        self.model_name = model
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key if api_key is not None else env.get(api_key_env, "").strip()
        self._api_key_env = api_key_env
        self._mode: JsonMode = json_mode
        self._max_tokens = max_tokens
        self._client = client or httpx.Client(timeout=timeout_s)

    @property
    def json_mode(self) -> JsonMode:
        """The structured-output mode currently in use (after any fallback)."""
        return self._mode

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant message; with ``json_schema`` a JSON document is requested."""
        while True:
            body = self._body(system, user, json_schema, temperature)
            response = self._post(body)
            if json_schema is not None and self._mode in _NEXT_MODE and _format_rejected(response):
                fallback = _NEXT_MODE[self._mode]
                logger.info(
                    "LLM sağlayıcısı %s çıktı biçimini reddetti (%s, HTTP %d); %s deneniyor.",
                    self._mode,
                    self.model_name,
                    response.status_code,
                    fallback,
                )
                self._mode = fallback
                continue
            self._raise_for_status(response)
            return self._content(response)

    # --- request -------------------------------------------------------------------------

    def _body(
        self, system: str, user: str, json_schema: dict[str, Any] | None, temperature: float
    ) -> dict[str, Any]:
        if json_schema is not None and self._mode != "json_schema":
            system = with_schema_instruction(system, json_schema)
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if json_schema is not None and self._mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                # strict mode would require every property, but optional fields have defaults
                "json_schema": {"name": SCHEMA_NAME, "schema": json_schema, "strict": False},
            }
        elif json_schema is not None and self._mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post(self._endpoint, json=body, headers=self._headers())
        except httpx.HTTPError as exc:
            raise LlmError(f"LLM sağlayıcısına ulaşılamadı ({self._endpoint}): {exc}") from exc

    # --- response ------------------------------------------------------------------------

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        status = response.status_code
        detail = response.text[:_ERROR_TEXT_LIMIT]
        if status in _AUTH_STATUS:
            hint = (
                f"{self._api_key_env} ortam değişkenindeki API anahtarını kontrol edin."
                if self._api_key
                else f"API anahtarı yok: {self._api_key_env} ortam değişkenini tanımlayın."
            )
            raise LlmError(f"LLM sağlayıcısı isteği yetkisiz buldu (HTTP {status}). {hint}")
        if status == _RATE_LIMIT:
            raise LlmError(
                f"LLM sağlayıcısının istek sınırı aşıldı ({self.model_name}, HTTP 429). "
                "Biraz bekleyip tekrar deneyin."
            )
        hint = (
            " Model adını (llm.model) ve uç noktayı (llm.openai_compat.base_url) kontrol edin."
            if status == _NOT_FOUND
            else ""
        )
        raise LlmError(f"LLM isteği başarısız ({self.model_name}, HTTP {status}): {detail}.{hint}")

    def _content(self, response: httpx.Response) -> str:
        try:
            choice = response.json()["choices"][0]
            message = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LlmError(
                f"LLM yanıtı beklenen biçimde değil ({self.model_name}): "
                f"{response.text[:_ERROR_TEXT_LIMIT]}"
            ) from exc
        if choice.get("finish_reason") == "length":
            raise LlmError(
                f"LLM yanıtında çıktı kesildi ({self.model_name}, çıktı sınırı); "
                "llm.openai_compat.max_tokens değerini artırın."
            )
        content = message.get("content")
        if not content:
            if message.get("refusal"):
                raise LlmError(f"LLM ({self.model_name}) isteği reddetti: {message['refusal']}")
            raise LlmError(f"LLM ({self.model_name}) boş yanıt döndürdü.")
        return str(content)
