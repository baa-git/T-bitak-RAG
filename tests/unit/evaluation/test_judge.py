"""Tests for the LLM judge and the rule-based checks (FR-E7, FR-E8, FR-GR8)."""

import json

import pytest

from tests.builders import make_chunk, make_scored
from tests.eval_builders import QUOTE, make_answer, make_question
from tests.fakes.llm import FakeLLM
from tubitak_rag.domain.answer import Citation
from tubitak_rag.domain.enums import QuestionType
from tubitak_rag.evaluation.judge import (
    JUDGE_ERROR,
    JUDGE_SYSTEM_PROMPT,
    LlmJudge,
    citation_precision,
    numbers_present,
    programs_split,
)


def _verdict(dogru: bool, sadik: bool, gerekce: str = "Oran doğru.") -> str:
    return json.dumps({"dogru": dogru, "sadik": sadik, "gerekce": gerekce}, ensure_ascii=False)


def test_judge_parses_verdict() -> None:
    llm = FakeLLM([f"<think>düşün</think>```json\n{_verdict(True, False)}\n```"], "hakem-model")
    judge = LlmJudge(llm)
    verdict = judge.judge(make_question(), make_answer(), [QUOTE])
    assert (verdict.dogru, verdict.sadik, verdict.gerekce) == (True, False, "Oran doğru.")
    assert judge.model_name == "hakem-model"
    assert llm.calls[0]["json_schema"]["required"] == ["dogru", "sadik", "gerekce"]


@pytest.mark.parametrize("reply", ["bu bir JSON değil", '{"dogru": "belki"}'])
def test_judge_error_yields_none(reply: str) -> None:
    verdict = LlmJudge(FakeLLM([reply])).judge(make_question(), make_answer(), [])
    assert (verdict.dogru, verdict.sadik, verdict.gerekce) == (None, None, JUDGE_ERROR)


def test_judge_llm_failure_yields_none() -> None:
    verdict = LlmJudge(FakeLLM([])).judge(make_question(), make_answer(), [])  # raises inside
    assert verdict.gerekce == JUDGE_ERROR


def test_judge_prompt_contains_expected_and_generated() -> None:
    llm = FakeLLM([_verdict(True, True)])
    question = make_question(beklenen_cevap="Beklenen: en fazla %75.")
    answer = make_answer(cevap="Üretilen: %75 [1].")
    LlmJudge(llm).judge(question, answer, ["Kaynak metni A", "Kaynak metni B"])
    call = llm.calls[0]
    assert call["system"] == JUDGE_SYSTEM_PROMPT
    for part in (question.soru, "Beklenen: en fazla %75.", "Üretilen: %75 [1].", "Kaynak metni B"):
        assert part in call["user"]


def test_numbers_present_ignores_percent_and_separators() -> None:
    text = "Destek oranı yüzde 75, üst limit 2300000 TL'dir."
    assert numbers_present(["%75", "2.300.000"], text) is True
    assert numbers_present(["%75", "2.300.000"], "Oran %75, limit 2.300.000 TL.") is True
    assert numbers_present(["%60"], text) is False
    assert numbers_present(["24"], "süre 2 aydır") is False
    assert numbers_present([], text) is None


def test_citation_precision() -> None:
    question = make_question()
    relevant = make_scored(make_chunk("c1"), 0.9)
    quote_elsewhere = make_scored(make_chunk("c2", doc_id="mevzuat-031", text=QUOTE), 0.8)
    unrelated = make_scored(make_chunk("x1", doc_id="diger", text="Başka bir konu."), 0.7)

    def cite(*chunks: tuple[str, str]) -> list[Citation]:
        return [
            Citation(no=i, chunk_id=c, doc_id=d, doc_title="t", konum="", kaynak_url="u")
            for i, (c, d) in enumerate(chunks, start=1)
        ]

    retrieved = [relevant, quote_elsewhere, unrelated]
    both = make_answer(atiflar=cite(("c1", "mevzuat-031"), ("x1", "diger")))
    assert citation_precision(both, question, retrieved) == 0.5
    quote_doc = make_question(beklenen_belgeler=["baska"])
    assert (
        citation_precision(make_answer(atiflar=cite(("c2", "mevzuat-031"))), quote_doc, retrieved)
        == 1.0
    )
    assert citation_precision(make_answer(atiflar=[]), question, retrieved) is None


def test_programs_split() -> None:
    question = make_question(tur=QuestionType.BELIRSIZ, program_kodu=["1501", "1507", "4006"])
    assert programs_split(question, "1501'de %60, 1507'de %75 [1][2].") is True
    assert programs_split(question, "1507'de %75 [1].") is False
    assert programs_split(make_question(program_kodu=["1507"]), "1507") is None
