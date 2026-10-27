"""Tests for the prompt-mode JSON helpers shared by the OpenAI-compatible and transformers LLMs."""

import json

from tubitak_rag.adapters.json_prompt import (
    SCHEMA_INSTRUCTION,
    extract_json_object,
    strip_thinking,
    with_schema_instruction,
)


def test_schema_is_appended_to_the_system_prompt_unescaped() -> None:
    schema = {"properties": {"cevap": {"description": "Türkçe cevap"}}}
    text = with_schema_instruction("sistem", schema)
    assert text == "sistem" + SCHEMA_INSTRUCTION.format(
        schema=json.dumps(schema, ensure_ascii=False)
    )
    assert "Türkçe" in text


def test_extract_json_object_drops_think_fences_and_prose() -> None:
    reply = '<think>\nx\n</think>\nİşte: ```json\n{"a": {"b": 1}}\n```'
    assert extract_json_object(reply) == '{"a": {"b": 1}}'


def test_extract_json_object_without_object_returns_cleaned_text() -> None:
    assert extract_json_object("<think></think>  yok ") == "yok"


def test_strip_thinking() -> None:
    assert strip_thinking("<think>\n\n</think>\n\nMerhaba ") == "Merhaba"
