"""Structured output without constrained decoding: the JSON schema is described in the prompt.

Ollama (``format``) and most hosted APIs (``response_format``) can force a reply to match a
schema. When that is unavailable (an OpenAI-compatible provider that rejects both response
formats, or plain ``transformers`` generation on a Hugging Face Space), the schema is appended
to the system prompt and the reply is parsed tolerantly afterwards
(``generation.schema.parse_json_reply`` strips ``<think>`` blocks, code fences and prose).
"""

import json
import re
from typing import Any

SCHEMA_INSTRUCTION = (
    "\n\nYanıtını yalnızca aşağıdaki JSON şemasına uyan tek bir JSON nesnesi olarak ver; "
    "JSON dışında açıklama ya da kod bloğu yazma.\nJSON şeması: {schema}"
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def with_schema_instruction(system: str, json_schema: dict[str, Any]) -> str:
    """``system`` followed by the instruction to answer with one object matching the schema."""
    schema_text = json.dumps(json_schema, ensure_ascii=False)
    return system + SCHEMA_INSTRUCTION.format(schema=schema_text)


def strip_thinking(text: str) -> str:
    """``text`` without Qwen3 ``<think>…</think>`` blocks, stripped of outer whitespace."""
    return _THINK_RE.sub("", text).strip()


def extract_json_object(text: str) -> str:
    """The outermost ``{...}`` of a reply, without ``<think>`` blocks; the text when none.

    Small local models sometimes wrap the object in a code fence or add a sentence before
    it. Only the first ``{`` and the last ``}`` are used, so nested objects stay intact; the
    final validation is the parser's job.
    """
    cleaned = strip_thinking(text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    return cleaned[start : end + 1] if 0 <= start < end else cleaned
