"""Open issue: a number that appears only in the question must not count as evidence.

Known defects are strict xfails until the guardrail is fixed. Run with ``--runxfail``
to see the failing security expectations; an unexpected pass fails the normal suite.
No real model or external service is used: the LLM output is deliberately adversarial.
"""

from typing import Literal

import pytest

from tests.builders import make_chunk, make_scored
from tests.fakes.llm import FakeLLM
from tests.fakes.retriever import FakeRetriever
from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.generation.generator import AnswerGenerator
from tubitak_rag.generation.schema import LlmOutput
from tubitak_rag.guardrails.gate import RetrievalGate
from tubitak_rag.guardrails.numbers import UNGROUNDED_NUMBER, check_numbers
from tubitak_rag.guardrails.scope import ScopeChecker
from tubitak_rag.pipeline import AskTrace, RagPipeline

Mode = Literal["strip", "reject"]
CORRECT_SENTENCE = "1507 programında destek oranı en fazla %75'tir [1]."
SOURCE = make_chunk(
    text="1507 programında destek oranı en fazla %75'tir. Başvurular çağrıyla alınır."
)
QUESTION_CASES = [
    pytest.param(
        "1507 programında destek oranı %99 mudur?",
        "Destek oranı %99 olarak uygulanır [1].",
        "99",
        id="percentage",
    ),
    pytest.param(
        "1507 programında destek oranı yüzde doksan dokuz mudur?",
        "Destek oranı yüzde doksan dokuz olarak uygulanır [1].",
        "99",
        id="percentage-in-words",
    ),
    pytest.param(
        "1507 programında bütçe üst sınırı 9 milyon TL midir?",
        "Bütçe üst sınırı 9.000.000 TL olarak uygulanır [1].",
        "9000000",
        id="amount",
    ),
    pytest.param(
        "1507 programında proje süresi 36 ay mıdır?",
        "Proje süresi en fazla 36 aydır [1].",
        "36",
        id="duration",
    ),
    pytest.param(
        "1507 programında başvuru son tarihi 31.12.2027 midir?",
        "Başvurular 31.12.2027 tarihinde sona erer [1].",
        "2027-12-31",
        id="deadline",
    ),
]
KNOWN_QUESTION_GROUNDING = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Open issue: grounding_tokens accepts unsupported quantities from the question",
)


@KNOWN_QUESTION_GROUNDING
@pytest.mark.parametrize("mode", ["strip", "reject"])
@pytest.mark.parametrize(("question", "fabrication", "unsupported"), QUESTION_CASES)
def test_question_quantity_cannot_ground_an_unsupported_claim(
    mode: Mode, question: str, fabrication: str, unsupported: str
) -> None:
    output = LlmOutput(cevaplanabilir=True, cevap=f"{CORRECT_SENTENCE} {fabrication}", atiflar=[1])

    verdict = check_numbers(output, question, [SOURCE], mode=mode, sources=[SOURCE])

    if mode == "reject":
        assert not verdict.ok, "An unsupported claim from the question must be rejected"
        assert verdict.reason == UNGROUNDED_NUMBER
    else:
        assert verdict.ok, "The source-supported sentence should survive stripping"
        assert verdict.cevap == CORRECT_SENTENCE, "The unsupported sentence must be removed"
        assert verdict.atiflar == [1]
    assert unsupported in verdict.ungrounded
    assert verdict.repaired_citations == []


@pytest.mark.parametrize("mode", ["strip", "reject"])
@pytest.mark.parametrize(("question", "fabrication", "unsupported"), QUESTION_CASES)
def test_question_with_a_false_premise_can_receive_a_source_supported_answer(
    mode: Mode, question: str, fabrication: str, unsupported: str
) -> None:
    output = LlmOutput(cevaplanabilir=True, cevap=CORRECT_SENTENCE, atiflar=[1])

    verdict = check_numbers(output, question, [SOURCE], mode=mode, sources=[SOURCE])

    assert verdict.ok
    assert verdict.cevap == CORRECT_SENTENCE
    assert verdict.ungrounded == []


@pytest.mark.parametrize("mode", ["strip", "reject"])
def test_suggested_quantity_is_allowed_when_a_source_actually_supports_it(mode: Mode) -> None:
    source = make_chunk(text="1507 programında proje süresi en fazla 36 aydır.")
    output = LlmOutput(cevaplanabilir=True, cevap="Proje süresi 36 aydır [1].", atiflar=[1])

    verdict = check_numbers(
        output, "1507 programında proje süresi 36 ay mıdır?", [source], mode=mode
    )

    assert verdict.ok and verdict.ungrounded == []
    assert verdict.cevap == output.cevap


@KNOWN_QUESTION_GROUNDING
@pytest.mark.parametrize("mode", ["strip", "reject"])
def test_pipeline_does_not_publish_a_quantity_supported_only_by_the_question(mode: Mode) -> None:
    output = LlmOutput(
        cevaplanabilir=True,
        cevap=f"{CORRECT_SENTENCE} Destek oranı %99 olarak uygulanır [1].",
        atiflar=[1],
    )
    llm = FakeLLM([output.model_dump_json()])
    pipeline = RagPipeline(
        retriever=FakeRetriever([make_scored(SOURCE, 0.9, dense=0.9)]),
        reranker=None,
        generator=AnswerGenerator(llm),
        scope=ScopeChecker("lexicon", llm),
        gate=RetrievalGate({"dense": 0.45, "rerank": -2.0, "bm25": 0.0}),
        k_candidates=5,
        top_n=1,
        disclaimer="Test uyarısı.",
        number_check=mode,
    )
    trace = AskTrace()

    answer = pipeline.ask("1507 programında destek oranı %99 mudur?", trace=trace)

    assert len(llm.calls) == 1 and trace.generation is not None
    if mode == "reject":
        assert not answer.cevaplanabilir, "The pipeline must refuse the unsupported quantity"
        assert answer.durum == AnswerStatus.BILGI_BULUNAMADI
        assert trace.rejection_reason == UNGROUNDED_NUMBER
        assert answer.atiflar == [] and answer.kaynaklar == []
    else:
        assert answer.cevap == CORRECT_SENTENCE, "Only the grounded sentence may be published"
        assert answer.durum == AnswerStatus.CEVAPLANDI
        assert [citation.no for citation in answer.atiflar] == [1]
