"""Tests for the answer generator and citation mapping (FR-G1..G3)."""

from typing import Any

from tests.builders import make_chunk, make_scored
from tests.fakes.llm import FakeLLM
from tubitak_rag.domain.answer import Citation, SourceView
from tubitak_rag.domain.retrieval import ScoredChunk
from tubitak_rag.generation.generator import AnswerGenerator, citations_from, source_views
from tubitak_rag.generation.prompts import SYSTEM_PROMPT
from tubitak_rag.generation.schema import LlmOutput, output_json_schema

ANSWER = '{"cevaplanabilir": true, "cevap": "En fazla %75 [1].", "atiflar": [1]}'


class TemperatureRecordingLLM(FakeLLM):
    """FakeLLM that also records the sampling temperature of each call."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.temperatures: list[float] = []

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.temperatures.append(temperature)
        return super().generate(system, user, json_schema=json_schema, temperature=temperature)


def _sources() -> list[ScoredChunk]:
    return [
        make_scored(make_chunk("c1", text="Destek oranı en fazla %75."), 0.9),
        make_scored(
            make_chunk(
                "c2",
                doc_id="sayfa-1501",
                doc_title="1501 Program Sayfası",
                program_kodu="1501",
                madde_no=None,
                bolum="Kimler başvurabilir?",
                sayfa=None,
                kaynak_url="https://tubitak.gov.tr/1501",
                text="Soru: Kimler başvurabilir?\nCevap: KOBİ'ler.",
            ),
            0.5,
        ),
    ]


def test_generate_passes_schema_and_temperature_to_llm() -> None:
    llm = TemperatureRecordingLLM([ANSWER])
    AnswerGenerator(llm, temperature=0.0).generate("Destek oranı nedir?", _sources())
    call = llm.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["json_schema"] == output_json_schema()
    assert "[1] 1507 Uygulama Esasları" in call["user"]
    assert "[2] 1501 Program Sayfası" in call["user"]
    assert "Destek oranı nedir?" in call["user"]
    assert llm.temperatures == [0.0]


def test_generate_parses_output() -> None:
    result = AnswerGenerator(FakeLLM([ANSWER])).generate("Soru?", _sources())
    assert result.output == LlmOutput(cevaplanabilir=True, cevap="En fazla %75 [1].", atiflar=[1])
    assert result.error is None
    assert [s.chunk.chunk_id for s in result.sources] == ["c1", "c2"]
    assert result.prompt_chars > 0
    assert result.latency_ms >= 0


def test_sources_returned_in_prompt_order() -> None:
    old = make_scored(make_chunk("old", surum_tarihi="2020-01-01"), 0.9)
    new = make_scored(make_chunk("new", surum_tarihi="2026-01-01"), 0.1)
    llm = FakeLLM([ANSWER])
    result = AnswerGenerator(llm).generate("Soru?", [old, new])
    assert [s.chunk.chunk_id for s in result.sources] == ["new", "old"]
    assert llm.calls[0]["user"].index("2026-01-01") < llm.calls[0]["user"].index("2020-01-01")


def test_invalid_output_recorded_as_error() -> None:
    result = AnswerGenerator(FakeLLM(["Bilmiyorum."])).generate("Soru?", _sources())
    assert result.output is None
    assert result.error == "gecersiz_cikti"


def test_citations_have_title_location_url() -> None:
    output = LlmOutput(cevaplanabilir=True, cevap="… [1] … [2]", atiflar=[2, 1, 2])
    citations = citations_from(output, _sources())
    assert citations == [
        Citation(
            no=1,
            chunk_id="c1",
            doc_id="mevzuat-031",
            doc_title="1507 Uygulama Esasları",
            konum="Madde 7",
            kaynak_url="https://tubitak.gov.tr/x.pdf",
        ),
        Citation(
            no=2,
            chunk_id="c2",
            doc_id="sayfa-1501",
            doc_title="1501 Program Sayfası",
            konum="SSS: Kimler başvurabilir?",
            kaynak_url="https://tubitak.gov.tr/1501",
        ),
    ]
    assert all(c.doc_title and c.konum and c.kaynak_url for c in citations)


def test_citations_include_inline_markers_missing_from_atiflar() -> None:
    output = LlmOutput(cevaplanabilir=True, cevap="a [1] b [2]", atiflar=[1])
    assert [c.no for c in citations_from(output, _sources())] == [1, 2]


def test_citations_ignore_numbers_outside_sources() -> None:
    output = LlmOutput(cevaplanabilir=True, cevap="x", atiflar=[0, 1, 9])
    assert [c.no for c in citations_from(output, _sources())] == [1]


def test_source_views_numbered_like_prompt() -> None:
    views = source_views(_sources())
    assert views[0] == SourceView(
        no=1,
        chunk_id="c1",
        doc_title="1507 Uygulama Esasları",
        konum="Madde 7",
        kaynak_url="https://tubitak.gov.tr/x.pdf",
        text="Destek oranı en fazla %75.",
        score=0.9,
    )
    assert [v.no for v in views] == [1, 2]
