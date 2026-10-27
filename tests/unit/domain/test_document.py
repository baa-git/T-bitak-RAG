"""Tests for parsed document models."""

from tubitak_rag.domain.document import Document, Paragraph, Section
from tubitak_rag.domain.enums import DocType, SectionKind, SourceKind


def _section(idx: int, text: str) -> Section:
    return Section(
        section_id=f"mevzuat-031:{idx:04d}",
        kind=SectionKind.MADDE,
        title="Destek oranı",
        hierarchy=["Üçüncü Bölüm – Destek"],
        madde_no=str(idx),
        page_start=1,
        page_end=1,
        text=text,
        paragraphs=[Paragraph(label="(1)", text=text, page=1)],
        amendment_notes=[],
        repealed=False,
    )


def _document(sections: list[Section]) -> Document:
    return Document(
        doc_id="mevzuat-031",
        title="1507 Uygulama Esasları",
        program_kodu="1507",
        doc_turu=DocType.UYGULAMA_ESASLARI,
        kind=SourceKind.PDF,
        kaynak_url="https://tubitak.gov.tr/x.pdf",
        surum_tarihi=None,
        dil="tr",
        sha256="b" * 64,
        kaynak_grubu="tubitak_mevzuat",
        scanned=False,
        structured=True,
        page_count=10,
        sections=sections,
    )


def test_document_full_text_joins_sections() -> None:
    doc = _document([_section(1, "a"), _section(2, "b")])
    assert doc.full_text() == "a\n\nb"


def test_document_full_text_empty_when_no_sections() -> None:
    assert _document([]).full_text() == ""


def test_document_round_trips_json() -> None:
    doc = _document([_section(1, "a")])
    assert Document.model_validate_json(doc.model_dump_json()) == doc
