"""LLM backed by a local Ollama server (``POST /api/chat``), the default provider (NFR-PRIV1)."""

from typing import Any

import httpx

from tubitak_rag.ports.llm import LlmError

_NOT_FOUND = 404


class OllamaLLM:
    """Non-streaming chat with thinking off and optional JSON-schema constrained output."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        num_ctx: int = 8192,
        think: bool = False,
        timeout_s: float = 120,
        client: httpx.Client | None = None,
    ) -> None:
        self.model_name = model
        self._endpoint = f"{url.rstrip('/')}/api/chat"
        self._num_ctx = num_ctx
        self._think = think
        self._client = client or httpx.Client(timeout=timeout_s)

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant message; with ``json_schema`` Ollama constrains it to the schema."""
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self._think,
            "options": {"temperature": temperature, "num_ctx": self._num_ctx},
        }
        if json_schema is not None:
            body["format"] = json_schema
        try:
            response = self._client.post(self._endpoint, json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            hint = (
                f" Model yüklü değilse: ollama pull {self.model_name}"
                if exc.response.status_code == _NOT_FOUND
                else ""
            )
            raise LlmError(
                f"Ollama isteği başarısız ({self.model_name}, HTTP {exc.response.status_code}): "
                f"{exc.response.text}.{hint}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LlmError(f"Ollama'ya ulaşılamadı ({self._endpoint}): {exc}") from exc
        data = response.json()
        if data.get("done_reason") == "length":
            raise LlmError(
                f"Ollama yanıtında çıktı kesildi ({self.model_name}, bağlam/çıktı sınırı); "
                "llm.num_ctx değerini artırın."
            )
        content: str = data["message"]["content"]
        return content
