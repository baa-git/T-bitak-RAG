"""Tests for the end-to-end ask flow with fake retriever, reranker and LLM."""

import json
from typing import Any

import pytest

from tests.builders import make_chunk, make_scored
from tests.fakes.llm import FakeLLM
from tests.fakes.reranker import FakeReranker
from tests.fakes.retriever import FakeRetriever
from tests.fakes.vector_store import InMemoryVectorStore
from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.generation.generator import AnswerGenerator
from tubitak_rag.generation.prompts import BILGI_BULUNAMADI_MESAJI, KAPSAM_DISI_MESAJI
from tubitak_rag.guardrails.gate import RetrievalGate
from tubitak_rag.guardrails.numbers import STRIPPED_NOTE
from tubitak_rag.guardrails.scope import ScopeChecker
from tubitak_rag.pipeline import (
    LLM_HATASI_MESAJI,
    REFUSAL_REASONS,
    AskTrace,
    RagPipeline,
)
from tubitak_rag.ports.llm import LlmError

DISCLAIMER = "Bu cevap bilgilendirme amaçlıdır; resmî görüş değildir."
THRESHOLDS = {"dense": 0.45, "rerank": 0.5, "bm25": 0.0}
QUESTION = "1507 programında destek oranı en fazla yüzde kaçtır?"

RATE = make_chunk(
    "mevzuat-031:0007:00",
    text="Bu programda uygulanacak destek oranı her dönem için en fazla %75 olarak belirlenmiştir.",
)
DATES = make_chunk(
    "mevzuat-031:0010:00",
    madde_no="10",
    hiyerarsi="Üçüncü Bölüm › Madde 10 – Başvuru",
    text="Başvuru tarihleri ve çağrı koşulları çağrı duyurusunda belirtilir.",
)
DURATION = make_chunk(
    "mevzuat-031:0006:00",
    madde_no="6",
    hiyerarsi="Üçüncü Bölüm › Madde 6 – Destek süresi",
    text="Projelerin destek süresi en fazla 24 aydır.",
)
PAGE = make_chunk(
    "sayfa-1507:0001:00",
    doc_id="sayfa-1507",
    doc_title="1507 - KOBİ Ar-Ge Başlangıç Destek Programı",
    doc_turu="program_sayfasi",
    madde_no=None,
    hiyerarsi="",
    bolum="Çağrılar",
    sayfa=None,
    surum_tarihi=None,
    kaynak_url="https://tubitak.gov.tr/tr/destekler/1507",
    text="Açık çağrılar bu sayfada duyurulur.",
)


def _reply(
    cevap: str,
    atiflar: list[int],
    *,
    cevaplanabilir: bool = True,
    eksik_kisim: str | None = None,
) -> str:
    return json.dumps(
        {
            "cevaplanabilir": cevaplanabilir,
            "cevap": cevap,
            "atiflar": atiflar,
            "eksik_kisim": eksik_kisim,
        },
        ensure_ascii=False,
    )


ANSWER_75 = _reply("Destek oranı her dönem için en fazla %75'tir [1].", [1])


def _dense(*chunks_and_scores: tuple[Any, float]) -> list[ScoredChunk]:
    return [make_scored(chunk, score, dense=score) for chunk, score in chunks_and_scores]


class _Setup:
    def __init__(
        self,
        responses: list[str],
        results: list[ScoredChunk] | None = None,
        *,
        reranker: FakeReranker | None = None,
        scope_mode: str = "lexicon",
        number_check: str = "strip",
        top_n: int = 5,
        store: InMemoryVectorStore | None = None,
    ) -> None:
        self.llm = FakeLLM(responses)
        default = _dense((RATE, 0.82), (DATES, 0.61))
        self.retriever = FakeRetriever(default if results is None else results)
        self.pipeline = RagPipeline(
            retriever=self.retriever,
            reranker=reranker,
            generator=AnswerGenerator(self.llm),
            scope=ScopeChecker(scope_mode, self.llm),  # type: ignore[arg-type]
            gate=RetrievalGate(THRESHOLDS),
            k_candidates=30,
            top_n=top_n,
            disclaimer=DISCLAIMER,
            number_check=number_check,  # type: ignore[arg-type]
            store=store,
            collection="koleksiyon",
        )


class _FailingLLM(FakeLLM):
    def generate(self, system: str, user: str, **kwargs: Any) -> str:
        self.calls.append({"system": system, "user": user})
        raise LlmError("Ollama'ya ulaşılamadı: HTTP 500")


def _assert_refused(answer: Any, durum: AnswerStatus, message: str) -> None:
    assert answer.durum is durum
    assert answer.cevaplanabilir is False
    assert answer.cevap == message
    assert answer.atiflar == []
    assert answer.kaynaklar == []  # FR-GR2
    assert answer.uyari == DISCLAIMER


# --- scope (FR-GR4) ------------------------------------------------------------------------


def test_out_of_scope_returns_kapsam_disi_without_retrieval_or_sources() -> None:
    setup = _Setup([])
    trace = AskTrace()

    answer = setup.pipeline.ask("İstanbul'da yarın hava nasıl olacak?", trace=trace)

    _assert_refused(answer, AnswerStatus.KAPSAM_DISI, KAPSAM_DISI_MESAJI)
    assert setup.retriever.calls == [] and setup.llm.calls == []
    assert answer.llm_cagri_sayisi == 0
    assert answer.gate_key is None and answer.gate_score is None
    assert trace.rejection_reason == "kapsam_disi"
    assert set(answer.gecikme_ms) == {"scope", "total"}


def test_out_of_scope_by_llm_classifier_counts_its_call() -> None:
    setup = _Setup(['{"kapsam_ici": false}'], scope_mode="llm")

    answer = setup.pipeline.ask("Bugün hangi film izlenir?")

    assert answer.durum is AnswerStatus.KAPSAM_DISI
    assert answer.llm_cagri_sayisi == 1
    assert setup.retriever.calls == []


# --- retrieval gate (FR-GR1, FR-GR2) -------------------------------------------------------


def test_below_threshold_returns_bilgi_bulunamadi_with_no_llm_call_and_no_sources() -> None:
    setup = _Setup([], _dense((RATE, 0.31), (DATES, 0.12)))
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    _assert_refused(answer, AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI)
    assert setup.llm.calls == []
    assert answer.llm_cagri_sayisi == 0
    assert (answer.gate_key, answer.gate_score) == ("dense", 0.31)
    assert trace.rejection_reason == "esik_alti"
    assert trace.gate is not None and trace.gate.passed is False
    assert "llm" not in answer.gecikme_ms


def test_pipeline_no_results_no_llm_call() -> None:
    # An empty source list must never reach the LLM.
    setup = _Setup([], [])
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    _assert_refused(answer, AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI)
    assert setup.llm.calls == []
    assert answer.gate_key is None
    assert trace.rejection_reason == "sonuc_yok"


# --- happy path (FR-G3, FR-G5) -------------------------------------------------------------


def test_valid_answer_has_citations_sources_and_disclaimer() -> None:
    setup = _Setup([ANSWER_75])
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    assert answer.durum is AnswerStatus.CEVAPLANDI
    assert answer.cevaplanabilir is True
    assert answer.cevap == "Destek oranı her dönem için en fazla %75'tir [1]."
    (citation,) = answer.atiflar
    assert citation.no == 1 and citation.chunk_id == RATE.chunk_id
    assert citation.doc_title == RATE.doc_title
    assert citation.konum == "Madde 7"
    assert citation.kaynak_url == RATE.kaynak_url
    assert [(s.no, s.chunk_id) for s in answer.kaynaklar] == [
        (1, RATE.chunk_id),
        (2, DATES.chunk_id),
    ]
    assert answer.llm_cagri_sayisi == 1
    assert (answer.gate_key, answer.gate_score) == ("dense", 0.82)
    assert trace.rejection_reason is None
    assert answer.atif_onarimi == []


def test_disclaimer_is_given_once_in_its_own_field() -> None:
    answer = _Setup([ANSWER_75]).pipeline.ask(QUESTION)

    assert answer.uyari == DISCLAIMER
    assert DISCLAIMER not in answer.cevap


def test_citation_numbers_follow_prompt_order_not_retrieval_order() -> None:
    # order_sources puts the newer version first (FR-GR7); [1] must map to that chunk.
    old = make_chunk("mevzuat-030:0007:00", doc_id="mevzuat-030", surum_tarihi="2024-01-10")
    setup = _Setup([ANSWER_75], _dense((old, 0.9), (RATE, 0.8)))

    answer = setup.pipeline.ask(QUESTION)

    assert answer.atiflar[0].chunk_id == RATE.chunk_id
    assert [s.chunk_id for s in answer.kaynaklar] == [RATE.chunk_id, old.chunk_id]


# --- LLM output problems -------------------------------------------------------------------


def test_llm_answer_with_fabricated_citation_is_rejected_to_bilgi_bulunamadi() -> None:
    # FR-GR3: the message is "bilgi bulunamadı"; the status records why (atıf reddi).
    setup = _Setup([_reply("Destek oranı %75'tir [7].", [7])])
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    _assert_refused(answer, AnswerStatus.ATIF_REDDI, BILGI_BULUNAMADI_MESAJI)
    assert trace.rejection_reason == "gecersiz_atif"
    assert answer.llm_cagri_sayisi == 1


def test_uncited_answer_is_refused() -> None:
    setup = _Setup([_reply("Destek oranı %75'tir.", [])])
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    _assert_refused(answer, AnswerStatus.ATIF_REDDI, BILGI_BULUNAMADI_MESAJI)
    assert trace.rejection_reason == "atifsiz_cevap"


def test_llm_not_answerable_returns_no_sources() -> None:
    reply = _reply("Kaynaklarda bu bilgi yok.", [], cevaplanabilir=False)
    setup = _Setup([reply])
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    _assert_refused(answer, AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI)
    assert trace.rejection_reason == "cevaplanamaz"


def test_invalid_llm_json_treated_as_not_found() -> None:
    setup = _Setup(["Destek oranı yüzde yetmiş beştir."])
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    _assert_refused(answer, AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI)
    assert trace.rejection_reason == "gecersiz_cikti"
    assert trace.generation is not None and trace.generation.error == "gecersiz_cikti"


def test_llm_error_gives_turkish_message_without_sources() -> None:
    # Ollama HTTP 500 under GPU contention or a truncated reply: no crash, no sources.
    setup = _Setup([])
    setup.pipeline.generator = AnswerGenerator(_FailingLLM([]))
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    _assert_refused(answer, AnswerStatus.BILGI_BULUNAMADI, LLM_HATASI_MESAJI)
    assert trace.rejection_reason == "llm_hatasi"
    assert trace.error is not None and "HTTP 500" in trace.error
    assert answer.llm_cagri_sayisi == 1
    assert "llm" in answer.gecikme_ms
    assert answer.gate_key == "dense"


# --- number guardrail (call site: after citations, before post-processing) -----------------

INVENTED_24 = _reply("Destek oranı en fazla %75'tir [1]. Proje süresi en fazla 24 aydır [1].", [1])


def test_number_guardrail_strip_drops_the_ungrounded_sentence() -> None:
    setup = _Setup([INVENTED_24], number_check="strip")
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    assert answer.durum is AnswerStatus.CEVAPLANDI
    assert "24" not in answer.cevap
    assert answer.cevap.startswith("Destek oranı en fazla %75'tir [1].")
    assert f"Cevaplanamayan kısım: {STRIPPED_NOTE}" in answer.cevap
    assert trace.number_check is not None and trace.number_check.ungrounded == ["24"]
    assert [c.no for c in answer.atiflar] == [1]


def test_number_guardrail_reject_mode_refuses() -> None:
    setup = _Setup([INVENTED_24], number_check="reject")
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    _assert_refused(answer, AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI)
    assert trace.rejection_reason == "dayanaksiz_sayi"


def test_number_guardrail_strip_refuses_when_no_cited_sentence_remains() -> None:
    reply = _reply("Destek oranı %75, proje süresi ise en fazla 24 aydır [1].", [1])
    trace = AskTrace()

    answer = _Setup([reply], number_check="strip").pipeline.ask(QUESTION, trace=trace)

    _assert_refused(answer, AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI)
    assert trace.rejection_reason == "dayanaksiz_sayi"


def test_number_in_uncited_prompt_source_is_kept_and_cited_not_stripped() -> None:
    # "24" is in source [2] but the sentence cites [1]: citation repair adds [2].
    reply = _reply("Destek oranı en fazla %75'tir [1]. Süre en fazla 24 aydır [1].", [1])
    setup = _Setup([reply], _dense((RATE, 0.8), (DURATION, 0.7)), number_check="strip")
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    assert "Süre en fazla 24 aydır [1, 2]." in answer.cevap
    assert [c.no for c in answer.atiflar] == [1, 2]
    assert trace.atif_onarimi == [2] and answer.atif_onarimi == [2]


def test_number_guardrail_off_keeps_the_answer() -> None:
    answer = _Setup([INVENTED_24], number_check="off").pipeline.ask(QUESTION)

    assert answer.durum is AnswerStatus.CEVAPLANDI
    assert "24 aydır" in answer.cevap


# --- post-processing (FR-GR5, FR-GR6) -------------------------------------------------------


def test_announcement_link_added_when_cited_source_defers_to_call() -> None:
    reply = _reply("Başvuru tarihleri çağrı duyurusunda belirtilir [1].", [1])
    setup = _Setup([reply], _dense((DATES, 0.8), (RATE, 0.7)))

    answer = setup.pipeline.ask("1507 başvuru tarihleri ne zaman?")

    assert answer.cevap.endswith(f"İlgili duyuru/çağrı sayfası: {DATES.kaynak_url}")


def test_announcement_link_prefers_cited_program_page() -> None:
    reply = _reply("Tarihler çağrı duyurusunda belirtilir [1]; çağrılar sayfada [2].", [1, 2])
    setup = _Setup([reply], _dense((DATES, 0.8), (PAGE, 0.7)))

    answer = setup.pipeline.ask("1507 başvuru tarihleri ne zaman?")

    assert f"İlgili duyuru/çağrı sayfası: {PAGE.kaynak_url}" in answer.cevap


def test_no_announcement_link_from_uncited_source() -> None:
    setup = _Setup([ANSWER_75], _dense((RATE, 0.8), (DATES, 0.7)))

    answer = setup.pipeline.ask(QUESTION)

    assert "İlgili duyuru" not in answer.cevap


def test_missing_part_appended() -> None:
    reply = _reply("Destek oranı en fazla %75'tir [1].", [1], eksik_kisim="Proje süresi.")

    answer = _Setup([reply]).pipeline.ask(QUESTION)

    assert answer.cevap.endswith("\n\nCevaplanamayan kısım: Proje süresi.")


# --- reranking, top_n, timings, filters, trace ---------------------------------------------


def test_reranker_used_when_present_and_scores_kept() -> None:
    weak = make_chunk("mevzuat-031:0001:00", madde_no="1", text="Amaç ve kapsam.")
    candidates = _dense((weak, 0.9), (DATES, 0.8), (RATE, 0.7))
    setup = _Setup([ANSWER_75], candidates, reranker=FakeReranker(), top_n=2)
    trace = AskTrace()

    answer = setup.pipeline.ask("destek oranı en fazla yüzde kaç", trace=trace)

    assert answer.kaynaklar[0].chunk_id == RATE.chunk_id
    assert len(answer.kaynaklar) == 2  # only top_n reach the prompt
    assert answer.gate_key == "rerank"
    # The trace keeps the whole reranked candidate list (the runner's recall@10 needs > top_n).
    assert [r.chunk.chunk_id for r in trace.reranked][0] == RATE.chunk_id
    assert len(trace.reranked) == len(candidates)
    assert all({"dense", "rerank"} <= set(r.scores) for r in trace.reranked)


def test_without_reranker_candidates_are_cut_to_top_n() -> None:
    chunks = [make_chunk(f"mevzuat-031:{i:04d}:00", text=RATE.text) for i in range(7)]
    results = _dense(*((chunk, 0.9 - i / 100) for i, chunk in enumerate(chunks)))
    setup = _Setup([ANSWER_75], results, top_n=3)
    trace = AskTrace()

    answer = setup.pipeline.ask(QUESTION, trace=trace)

    assert len(answer.kaynaklar) == 3
    assert trace.reranked == []
    assert len(trace.candidates) == 7
    assert "rerank" not in answer.gecikme_ms


def test_latency_fields_filled_and_llm_call_count() -> None:
    setup = _Setup([ANSWER_75], reranker=FakeReranker())

    answer = setup.pipeline.ask(QUESTION)

    assert set(answer.gecikme_ms) == {"scope", "retrieval", "rerank", "llm", "total"}
    assert all(value >= 0 for value in answer.gecikme_ms.values())
    stages = sum(v for k, v in answer.gecikme_ms.items() if k != "total")
    assert answer.gecikme_ms["total"] >= stages
    assert answer.llm_cagri_sayisi == 1


def test_scope_llm_call_and_answer_call_are_both_counted() -> None:
    setup = _Setup(['{"kapsam_ici": true}', ANSWER_75], scope_mode="both")

    answer = setup.pipeline.ask("Bir şirketin yazılım geliştirme giderleri karşılanır mı?")

    assert answer.durum is AnswerStatus.CEVAPLANDI
    assert answer.llm_cagri_sayisi == 2


def test_filters_forwarded() -> None:
    # FR-R5
    setup = _Setup([ANSWER_75])
    filters = MetadataFilter(program_kodu="1507", doc_turu="uygulama_esaslari")

    setup.pipeline.ask(QUESTION, filters)

    assert setup.retriever.calls == [{"query": QUESTION, "k": 30, "filters": filters}]


def test_trace_is_filled_when_provided() -> None:
    setup = _Setup([ANSWER_75], reranker=FakeReranker())
    trace = AskTrace()

    setup.pipeline.ask(QUESTION, trace=trace)

    assert trace.scope is not None and trace.scope.in_scope
    assert [r.chunk.chunk_id for r in trace.candidates] == [RATE.chunk_id, DATES.chunk_id]
    assert trace.reranked and trace.gate is not None and trace.gate.passed
    assert trace.generation is not None and trace.generation.output is not None
    assert trace.citation is not None and trace.citation.ok
    assert trace.number_check is not None and trace.number_check.ok
    assert trace.rejection_reason is None and trace.error is None


def test_every_refusal_reason_has_a_turkish_label() -> None:
    expected = {
        "kapsam_disi",
        "sonuc_yok",
        "esik_alti",
        "llm_hatasi",
        "gecersiz_cikti",
        "cevaplanamaz",
        "atifsiz_cevap",
        "gecersiz_atif",
        "dayanaksiz_sayi",
    }
    assert set(REFUSAL_REASONS) == expected
    assert all(label.strip() for label in REFUSAL_REASONS.values())


# --- chunk lookup for the web source panel -------------------------------------------------


def test_get_chunks_reads_the_pipeline_collection() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection("koleksiyon", 8)
    store.upsert("koleksiyon", [RATE], [[1.0] * 8])
    setup = _Setup([], store=store)

    assert setup.pipeline.get_chunks([RATE.chunk_id, "yok"]) == [RATE]


def test_get_chunks_without_store_is_empty() -> None:
    assert _Setup([]).pipeline.get_chunks([RATE.chunk_id]) == []


@pytest.mark.parametrize("top_n", [0, -1])
def test_top_n_must_be_positive(top_n: int) -> None:
    with pytest.raises(ValueError, match="top_n"):
        _Setup([], top_n=top_n)


def test_number_in_uncited_prompt_source_is_answered_with_repaired_citation() -> None:
    # Seen in a smoke test: qwen3:4b cited only the FAQ [5] for "%75 … 18 ay"; "18 ay" is in [1].
    duration = make_chunk(
        "mevzuat-031:0006:00",
        madde_no="6",
        hiyerarsi="İkinci Bölüm › Madde 6 – Destek süresi",
        text="Projelerin destek süresi en fazla 18 aydır.",
    )
    filler = [
        make_chunk(f"mevzuat-031:00{i}0:00", madde_no=str(i), text="Amaç ve kapsam.")
        for i in (2, 3, 4)
    ]
    faq = PAGE.model_copy(
        update={"text": "Soru: Destek oranı nedir? Cevap: KOBİ'ler için destek oranı %75'tir."}
    )
    results = _dense((duration, 0.9), *((c, 0.8) for c in filler), (faq, 0.7))
    reply = _reply(
        "1507 programında KOBİ'lere destek oranı %75'tir ve proje süresi en fazla 18 aydır [5].",
        [5],
    )
    setup = _Setup([reply], results, number_check="reject")
    trace = AskTrace()

    answer = setup.pipeline.ask(
        "1507 programında KOBİ'lere destek oranı yüzde kaç ve proje süresi en fazla kaç ay?",
        trace=trace,
    )

    assert answer.durum is AnswerStatus.CEVAPLANDI
    assert [(c.no, c.chunk_id) for c in answer.atiflar] == [
        (1, duration.chunk_id),
        (5, faq.chunk_id),
    ]
    assert len(answer.kaynaklar) == 5
    assert "18 aydır [1, 5]." in answer.cevap  # the repair is visible to readers and judges
    assert trace.atif_onarimi == [1] and answer.atif_onarimi == [1]
    assert trace.number_check is not None and trace.number_check.repaired_citations == [1]
