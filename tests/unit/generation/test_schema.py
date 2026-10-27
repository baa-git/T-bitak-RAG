"""Tests for the structured LLM output schema and its tolerant parser (FR-G2)."""

import pytest

from tubitak_rag.generation.schema import (
    CITE_RE,
    LlmOutput,
    LlmOutputError,
    cited_numbers,
    output_json_schema,
    parse_llm_output,
)

VALID = '{"cevaplanabilir": true, "cevap": "En fazla %75 [1].", "atiflar": [1]}'


def test_parses_plain_json() -> None:
    output = parse_llm_output(VALID)
    assert output == LlmOutput(cevaplanabilir=True, cevap="En fazla %75 [1].", atiflar=[1])
    assert output.eksik_kisim is None


def test_strips_code_fence_and_think_block() -> None:
    text = f"<think>\nkaynak 1'e bakmalıyım\n</think>\n```json\n{VALID}\n```\n"
    assert parse_llm_output(text).atiflar == [1]


def test_extracts_object_from_surrounding_prose() -> None:
    assert parse_llm_output(f"İşte cevap:\n{VALID}\nUmarım yardımcı olur.").cevaplanabilir


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Bilmiyorum.",
        '{"cevaplanabilir": true, "cevap": "x"}',  # atiflar missing
        '{"cevaplanabilir": true, "cevap": "x", "atiflar": [1], "fazla": 1}',  # extra field
        '{"cevaplanabilir": true, "cevap": "x", "atiflar": [1]',  # truncated
    ],
)
def test_invalid_raises(text: str) -> None:
    with pytest.raises(LlmOutputError):
        parse_llm_output(text)


def test_schema_has_required_fields() -> None:
    schema = output_json_schema()
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"cevaplanabilir", "cevap", "atiflar"}
    assert schema["properties"]["atiflar"]["items"]["type"] == "integer"
    assert "eksik_kisim" in schema["properties"]
    assert schema["additionalProperties"] is False


def test_cited_numbers_reads_lists_and_atiflar() -> None:
    output = LlmOutput(cevaplanabilir=True, cevap="a [2] b [1, 3] c [ 4 ]", atiflar=[5])
    assert cited_numbers(output) == {1, 2, 3, 4, 5}


def test_cite_re_ignores_long_bracketed_numbers() -> None:
    assert CITE_RE.findall("Yıl [2026], madde [123], kaynak [12]") == ["12"]
