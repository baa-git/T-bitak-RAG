"""Response models of the web layer."""

import pytest
from pydantic import ValidationError

from tubitak_rag.interfaces.web.fake_backend import EXAMPLE_NOT_FOUND, FakeBackend
from tubitak_rag.interfaces.web.schemas import (
    AskRequest,
    AskResult,
    DocumentPreview,
    DocumentStatus,
    HealthStatus,
    IngestRequest,
    JobStarted,
    RunRequest,
    SectionView,
    SourceContext,
)


def _doc(durum: str) -> dict[str, object]:
    return {
        "doc_id": "mevzuat-031",
        "baslik": "1507",
        "program_kodu": "1507",
        "doc_turu": "uygulama_esaslari",
        "kaynak_url": "https://example.org/1507.pdf",
        "surum_tarihi": "2026-09-25",
        "dil": "tr",
        "durum": durum,
        "chunk_sayisi": 3,
    }


def test_document_state_is_restricted() -> None:
    assert DocumentStatus.model_validate(_doc("indeksli")).durum == "indeksli"
    with pytest.raises(ValidationError):
        DocumentStatus.model_validate(_doc("bilinmiyor"))


def test_job_started_defaults_to_basladi() -> None:
    assert JobStarted(job_id="x").durum == "basladi"


def test_ask_result_context_is_optional() -> None:
    answer = FakeBackend().ask(EXAMPLE_NOT_FOUND, None, None).answer
    assert AskResult(answer=answer).baglam == {}
    assert SourceContext().sayfa is None


def test_ask_request_strips_and_bounds_the_question() -> None:
    request = AskRequest.model_validate({"soru": "  1507 destek oranı?  ", "config": "v2"})
    assert request.soru == "1507 destek oranı?" and request.config == "v2"
    for soru in ["", "   ", "x" * 1001]:
        with pytest.raises(ValidationError):
            AskRequest.model_validate({"soru": soru})


def test_ask_request_rejects_unknown_doc_type_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AskRequest.model_validate({"soru": "soru", "doc_turu": "roman"})
    with pytest.raises(ValidationError):
        AskRequest.model_validate({"soru": "soru", "fazla": 1})
    assert AskRequest.model_validate({"soru": "s", "doc_turu": "kanun"}).doc_turu == "kanun"


def test_run_request_limit_must_be_positive() -> None:
    assert RunRequest.model_validate({"config": "v3"}).limit is None
    with pytest.raises(ValidationError):
        RunRequest.model_validate({"config": "v3", "limit": 0})
    with pytest.raises(ValidationError):
        RunRequest.model_validate({"config": ""})


def test_ingest_request_defaults_to_incremental() -> None:
    assert IngestRequest().full is False


def test_source_context_carries_document_anchor() -> None:
    context = SourceContext(doc_id="mevzuat-031", bolum_id="mevzuat-031:0012")
    assert context.bolum_id == "mevzuat-031:0012"


def test_health_and_preview_models() -> None:
    health = HealthStatus(durum="ok", qdrant=True, ollama=False, koleksiyon="k", surum="0.1.0")
    assert health.durum == "ok"
    section = SectionView(
        bolum_id="d:0001",
        baslik="Amaç",
        hiyerarsi=["Birinci Bölüm"],
        madde_no="1",
        sayfa=1,
        metin="(1) Amaç ...",
        mulga=False,
        degisiklik_notlari=[],
    )
    preview = DocumentPreview(
        belge=DocumentStatus.model_validate(_doc("indeksli")),
        kaynak_sayfa_url=None,
        ayristirildi=True,
        bolumler=[section],
        chunk_sayilari={"k": 3},
    )
    assert preview.bolum_sayisi == 1
