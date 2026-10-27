"""Structured LLM output (cevaplanabilir, cevap, atiflar, eksik_kisim) and its parser (FR-G2)."""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")
# Inline source markers: "[3]" and the list form small models also write, "[1, 2]". At most
# two digits, so a bracketed year such as "[2026]" is not taken for a citation.
CITE_RE = re.compile(r"\[\s*(\d{1,2}(?:\s*,\s*\d{1,2})*)\s*\]")


class LlmOutputError(ValueError):
    """Raised when the model reply is not a JSON object matching :class:`LlmOutput`."""


class LlmOutput(BaseModel):
    """What the LLM must return: answerability, the answer text and cited source numbers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cevaplanabilir: bool
    cevap: str
    atiflar: list[int]
    eksik_kisim: str | None = None


def inline_citation_numbers(text: str) -> set[int]:
    """Source numbers of the inline markers in ``text`` ("[1] … [2, 3]" -> {1, 2, 3})."""
    return {int(n) for group in CITE_RE.findall(text) for n in group.split(",")}


def cited_numbers(output: LlmOutput) -> set[int]:
    """Every source number the output refers to, in ``atiflar`` or inline in ``cevap``."""
    return inline_citation_numbers(output.cevap) | set(output.atiflar)


def output_json_schema() -> dict[str, Any]:
    """JSON schema passed to the LLM for constrained decoding (Ollama ``format``, API)."""
    return LlmOutput.model_json_schema()


def parse_json_reply[M: BaseModel](text: str, model: type[M]) -> M:
    """Parse a JSON reply into ``model``, tolerating ``<think>``, code fences and prose."""
    cleaned = _FENCE_RE.sub("", _THINK_RE.sub("", text).strip())
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        candidates.append(cleaned[start : end + 1])
    error: ValidationError | None = None
    for candidate in candidates:
        try:
            return model.model_validate_json(candidate)
        except ValidationError as exc:
            error = exc
    raise LlmOutputError(f"LLM çıktısı beklenen JSON şemasına uymuyor: {error}") from error


def parse_llm_output(text: str) -> LlmOutput:
    """Parse the answer reply (see :func:`parse_json_reply`)."""
    return parse_json_reply(text, LlmOutput)
