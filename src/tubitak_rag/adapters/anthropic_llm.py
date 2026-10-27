"""Optional LLM backed by the Anthropic Messages API (selectable by config only, FR-G4).

Never the default (NFR-PRIV1). Project rule: the real API is not called by tests or runs; the
adapter is verified with a fake ``anthropic`` module only. The API key is read by the SDK from
the environment (``ANTHROPIC_API_KEY``); this class takes no key parameter (NFR-CONF2).
"""

from typing import Any

from tubitak_rag.ports.llm import LlmError

DEFAULT_MODEL = "claude-opus-5"
# Server-side refusal fallback: a declined request is re-run on Anthropic's recommended model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    """Claude via ``client.beta.messages.create`` with effort control and JSON-schema output.

    Current Claude models reject sampling parameters, so ``temperature`` is ignored and
    ``output_config.effort`` is sent instead (reported as such in evaluation reports).
    """

    def __init__(
        self, model: str = DEFAULT_MODEL, *, effort: str = "low", max_tokens: int = 16000
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LlmError(
                "Anthropic sağlayıcısı için SDK kurulu değil: pip install 'tubitak-rag[api]'"
            ) from exc
        self.model_name = model
        self._sdk = anthropic
        self._client = anthropic.Anthropic()
        self._effort = effort
        self._max_tokens = max_tokens

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,  # ignored: rejected by current Claude models
    ) -> str:
        """Return the reply text; with ``json_schema`` the API guarantees schema-valid JSON."""
        output_config: dict[str, Any] = {"effort": self._effort}
        if json_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": self._sdk.transform_schema(json_schema),
            }
        request: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        try:
            response = self._client.beta.messages.create(**request)
        except self._sdk.APIError as exc:
            raise LlmError(f"Anthropic API isteği başarısız ({self.model_name}): {exc}") from exc
        if response.stop_reason == "refusal":
            raise LlmError(f"Anthropic modeli ({self.model_name}) isteği reddetti.")
        if response.stop_reason == "max_tokens":
            raise LlmError(
                f"Anthropic yanıtında çıktı kesildi ({self.model_name}, max_tokens sınırı)."
            )
        return "".join(block.text for block in response.content if block.type == "text")
