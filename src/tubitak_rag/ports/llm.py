"""LLM port: text generation, optionally constrained by a JSON schema."""

from typing import Any, Protocol, runtime_checkable


class LlmError(RuntimeError):
    """Raised by LLM adapters when the model cannot be reached or refuses (message in Turkish)."""


@runtime_checkable
class LLM(Protocol):
    """A chat-style language model."""

    model_name: str

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the reply text; a JSON document when ``json_schema`` is given."""
        ...
