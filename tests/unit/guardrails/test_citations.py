"""Tests for citation validation (FR-GR3)."""

from tubitak_rag.generation.schema import LlmOutput
from tubitak_rag.guardrails.citations import CitationVerdict, validate


def _output(cevap: str, atiflar: list[int], cevaplanabilir: bool = True) -> LlmOutput:
    return LlmOutput(cevaplanabilir=cevaplanabilir, cevap=cevap, atiflar=atiflar)


def test_answerable_without_citations_rejected() -> None:
    assert validate(_output("Destek oranı %75'tir.", []), 5) == CitationVerdict(
        ok=False, reason="atifsiz_cevap"
    )


def test_out_of_range_citation_rejected() -> None:
    assert validate(_output("… [1]", [7]), 5) == CitationVerdict(ok=False, reason="gecersiz_atif")


def test_zero_citation_rejected() -> None:
    assert not validate(_output("… [0]", [0]), 5).ok


def test_inline_bracket_out_of_range_rejected() -> None:
    verdict = validate(_output("Oran %75'tir [1]; süre 24 aydır [6].", [1]), 5)
    assert verdict == CitationVerdict(ok=False, reason="gecersiz_atif")


def test_inline_bracket_list_out_of_range_rejected() -> None:
    # qwen3:4b writes several sources in one bracket: "[1, 2]".
    assert not validate(_output("Oran %75'tir [1, 9].", [1]), 2).ok
    assert validate(_output("Oran %75'tir [1, 2].", [1, 2]), 2).ok


def test_not_answerable_without_citations_ok() -> None:
    output = _output("Kaynaklarda bilgi yok.", [], cevaplanabilir=False)
    assert validate(output, 5) == CitationVerdict(ok=True, reason=None)


def test_valid_passes() -> None:
    assert validate(_output("Destek oranı en fazla %75'tir [1].", [1]), 3).ok


def test_no_sources_rejects_any_answer() -> None:
    assert not validate(_output("x [1]", [1]), 0).ok


def test_bracketed_year_is_not_a_citation() -> None:
    assert validate(_output("2026 bütçesi [2026] açıklandı [1].", [1]), 1).ok
