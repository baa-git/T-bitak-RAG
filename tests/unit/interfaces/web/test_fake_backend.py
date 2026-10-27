"""The demo backend gives every UI state deterministically, without services."""

import pytest

from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.interfaces.gpu_lock import GpuLockError
from tubitak_rag.interfaces.web.backend import UnknownPipelineError
from tubitak_rag.interfaces.web.fake_backend import (
    EXAMPLE_ANSWERED,
    EXAMPLE_NOT_FOUND,
    EXAMPLE_OUT_OF_SCOPE,
    FakeBackend,
)


def test_answered_question_has_citations_sources_and_context() -> None:
    result = FakeBackend().ask(EXAMPLE_ANSWERED, None, None)
    answer = result.answer
    assert answer.durum is AnswerStatus.CEVAPLANDI and answer.cevaplanabilir
    assert [s.no for s in answer.kaynaklar] == [1, 2, 3]
    assert [c.no for c in answer.atiflar] == [1, 2, 3]
    assert all(s.chunk_id in result.baglam for s in answer.kaynaklar)
    assert answer.uyari  # FR-G5


def test_unanswerable_and_out_of_scope_have_no_sources() -> None:
    backend = FakeBackend()
    not_found = backend.ask(EXAMPLE_NOT_FOUND, None, None).answer
    out_of_scope = backend.ask(EXAMPLE_OUT_OF_SCOPE, None, None).answer
    assert not_found.durum is AnswerStatus.BILGI_BULUNAMADI
    assert out_of_scope.durum is AnswerStatus.KAPSAM_DISI
    for answer in (not_found, out_of_scope):
        assert answer.kaynaklar == [] and answer.atiflar == [] and not answer.cevaplanabilir


def test_documents_cover_every_state() -> None:
    documents = FakeBackend().documents()
    assert {d.durum for d in documents.belgeler} == {
        "indeksli",
        "taranmis",
        "eksik",
        "guncel_degil",
    }
    assert documents.toplam_chunk == sum(d.chunk_sayisi for d in documents.belgeler)


def test_ingest_jobs_get_increasing_ids() -> None:
    backend = FakeBackend()
    assert backend.start_ingest().job_id == "ingest-0001"
    assert backend.start_ingest(full=True).job_id == "ingest-0002"


def test_runs_newest_first_with_reports() -> None:
    backend = FakeBackend()
    runs = backend.runs()
    assert [r.deney for r in runs] == ["v3b", "v3", "v2", "v1", "v0r", "v0"]
    first = backend.run_report(runs[-1].run_id) or ""
    assert "Önceki basamakla" not in first  # v0 has no previous rung
    later = backend.run_report(runs[0].run_id) or ""
    assert "Önceki basamakla karşılaştırma (v3)" in later and "```mermaid" in later
    assert backend.run_metrics("yok") is None and backend.run_report("yok") is None
    assert "Deney merdiveni özeti" in (backend.summary_report() or "")


def test_pipelines_list_the_ladder_with_v3_as_default() -> None:
    options = FakeBackend().pipelines()
    assert [o.ad for o in options] == ["v0", "v0r", "v1", "v2", "v3", "v3b"]
    assert [o.ad for o in options if o.varsayilan] == ["v3"]
    assert all(o.etiket == o.ad for o in options)  # plain labels, no slogans


def test_pipeline_methods_are_derived_from_each_config() -> None:
    options = {o.ad: {m.etiket: m.deger for m in o.yontemler} for o in FakeBackend().pipelines()}
    assert options["v0"]["Parçalama"] == "sabit pencere (512 token, 64 örtüşme)"
    assert options["v0"]["Reranker"] == "yok"
    assert options["v3"]["Arama"] == "hibrit (BM25 + dense, RRF k=60)"
    assert options["v3"]["Reranker"] == "BAAI/bge-reranker-v2-m3"
    assert options["v3b"]["Embedding"] == "BAAI/bge-m3 (1024 boyut)"


def test_about_describes_corpus_and_components() -> None:
    about = FakeBackend().about()
    assert [g.kaynak_grubu for g in about.gruplar] == [
        "tubitak_mevzuat",
        "tubitak_program_sayfalari",
        "sanayi_mevzuati",
    ]
    assert about.belge_sayisi == sum(g.belge_sayisi for g in about.gruplar)
    assert any("qwen3:4b" in line.deger for line in about.bilesenler)
    assert [o.ad for o in about.yapilandirmalar][-1] == "v3b"


def test_disclaimer_is_the_configured_text() -> None:
    assert "resmî görüş değildir" in FakeBackend().disclaimer()


def test_ask_reports_the_config_used_and_rejects_unknown_ones() -> None:
    backend = FakeBackend()
    assert backend.ask(EXAMPLE_ANSWERED, None, None).yapilandirma == "v3"
    v0 = backend.ask(EXAMPLE_ANSWERED, None, None, "v0")
    assert v0.yapilandirma == "v0"
    assert "rerank" not in v0.answer.gecikme_ms  # v0 has no reranker
    with pytest.raises(UnknownPipelineError):
        backend.ask(EXAMPLE_ANSWERED, None, None, "v9")
    assert backend.asked[-1] == (EXAMPLE_ANSWERED, None, None, "v0")


def test_source_context_links_to_document_sections() -> None:
    result = FakeBackend().ask(EXAMPLE_ANSWERED, None, None)
    first = result.baglam[result.answer.kaynaklar[0].chunk_id]
    assert first.doc_id == "mevzuat-031" and first.bolum_id == "mevzuat-031:0012"


def test_document_preview_has_sections_with_repealed_and_amendment_notes() -> None:
    backend = FakeBackend()
    preview = backend.document("mevzuat-031")
    assert preview is not None and preview.ayristirildi
    ids = [s.bolum_id for s in preview.bolumler]
    assert "mevzuat-031:0012" in ids
    assert any(s.mulga for s in preview.bolumler)
    assert any(s.degisiklik_notlari for s in preview.bolumler)
    assert preview.chunk_sayilari and preview.belge.doc_id == "mevzuat-031"
    unparsed = backend.document("mevzuat-064")
    assert unparsed is not None and not unparsed.ayristirildi and unparsed.bolumler == []
    assert backend.document("yok") is None


def test_runs_can_be_started_and_detailed() -> None:
    backend = FakeBackend()
    assert backend.start_run("v2", limit=5).job_id == "run-0001"
    assert backend.runs_started == [("v2", 5)]
    with pytest.raises(UnknownPipelineError):
        backend.start_run("v9")
    run_id = backend.runs()[0].run_id
    detail = backend.run_detail(run_id)
    assert detail is not None and detail.metrics.run_id == run_id
    assert backend.run_detail("yok") is None


def test_busy_gpu_refuses_jobs_and_questions() -> None:
    backend = FakeBackend(gpu_busy=True)
    for call in (
        backend.start_ingest,
        lambda: backend.start_run("v3"),
        lambda: backend.ask(EXAMPLE_ANSWERED, None, None),
    ):
        with pytest.raises(GpuLockError, match="GPU"):
            call()


def test_health_of_demo_backend() -> None:
    health = FakeBackend().health()
    assert health.durum == "ok" and health.koleksiyon
