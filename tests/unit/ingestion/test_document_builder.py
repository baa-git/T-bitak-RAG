"""Tests for helpers shared by all parsers (FR-P1, FR-P10)."""

from datetime import date

from tubitak_rag.domain.document import Section
from tubitak_rag.domain.enums import DocType, SectionKind, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.document_builder import (
    attach_page_notes,
    build_document_skeleton,
    join_lines,
    make_section,
    make_section_id,
)

ENTRY = SourceEntry(
    doc_id="mevzuat-031",
    title="1507 Uygulama Esasları",
    url="https://tubitak.gov.tr/x.pdf",
    local_path="tubitak_mevzuat/031.pdf",
    kind=SourceKind.PDF,
    program_kodu="1507",
    doc_turu=DocType.UYGULAMA_ESASLARI,
    surum_tarihi=date(2026, 9, 25),
    dil="tr",
    sha256="0" * 64,
    kaynak_grubu="tubitak_mevzuat",
    kaynak_sayfa_url="https://tubitak.gov.tr/tr/mevzuat",
)


def test_make_section_id() -> None:
    assert make_section_id("mevzuat-031", 7) == "mevzuat-031:0007"


def test_skeleton_copies_entry_metadata_and_uses_real_sha() -> None:
    doc = build_document_skeleton(ENTRY, "f" * 64, page_count=45)
    assert doc.doc_id == "mevzuat-031" and doc.title == "1507 Uygulama Esasları"
    assert doc.program_kodu == "1507" and doc.doc_turu == DocType.UYGULAMA_ESASLARI
    assert doc.kaynak_url == "https://tubitak.gov.tr/x.pdf"
    assert doc.surum_tarihi == date(2026, 9, 25) and doc.dil == "tr"
    assert doc.sha256 == "f" * 64  # digest of the parsed file, not the manifest value
    assert doc.kaynak_grubu == "tubitak_mevzuat" and doc.kind == SourceKind.PDF
    assert (doc.scanned, doc.structured, doc.page_count, doc.sections) == (False, False, 45, [])


def test_skeleton_keeps_missing_fields_as_none() -> None:
    entry = ENTRY.model_copy(update={"program_kodu": None, "surum_tarihi": None})
    doc = build_document_skeleton(entry, "f" * 64, scanned=True)
    assert doc.program_kodu is None and doc.surum_tarihi is None and doc.page_count is None
    assert doc.scanned is True


def test_join_lines_uses_spaces_and_keeps_line_end_hyphens() -> None:
    lines = ["(Değişik:11/05/2023-51YK-", "15/06/2023 tarihli) Araştırma-", "Geliştirme", "projesi"]
    assert (
        join_lines(lines)
        == "(Değişik:11/05/2023-51YK-15/06/2023 tarihli) Araştırma-Geliştirme projesi"
    )


def test_join_lines_lone_dash_is_not_a_hyphen() -> None:
    assert join_lines(["Destek oranı -", "yüzde 75"]) == "Destek oranı - yüzde 75"


def _section(index: int, page: int | None) -> Section:
    return Section(
        section_id=make_section_id("d", index),
        kind=SectionKind.BOLUM,
        title="t",
        hierarchy=[],
        madde_no=None,
        page_start=page,
        page_end=page,
        text="x",
        paragraphs=[],
        amendment_notes=["eski"],
        repealed=False,
    )


def test_attach_page_notes_uses_last_section_started_on_or_before_the_page() -> None:
    sections = [_section(1, 1), _section(2, 3), _section(3, 5)]
    updated = attach_page_notes(sections, [(4, ["not A"]), (1, ["not B"]), (9, [])])
    assert [s.amendment_notes for s in updated] == [["eski", "not B"], ["eski", "not A"], ["eski"]]


def test_attach_page_notes_before_first_section_goes_to_first() -> None:
    updated = attach_page_notes([_section(1, 3)], [(1, ["not"])])
    assert updated[0].amendment_notes == ["eski", "not"]
    assert attach_page_notes([], [(1, ["not"])]) == []


def test_make_section_defaults() -> None:
    section = make_section("d", 3, SectionKind.PARAGRAF, text="Metin.")
    assert section.section_id == "d:0003" and section.kind == SectionKind.PARAGRAF
    assert section.text == "Metin." and section.paragraphs == []
    assert (section.title, section.hierarchy, section.madde_no) == (None, [], None)
    assert (section.page_start, section.page_end, section.amendment_notes) == (None, None, [])
    assert section.repealed is False


def test_make_section_passes_fields_through() -> None:
    section = make_section(
        "d",
        1,
        SectionKind.MADDE,
        text="x",
        title="Amaç",
        hierarchy=["Bölüm"],
        madde_no="1",
        page_start=2,
        page_end=3,
        amendment_notes=["(Ek: 1/1/2020-1YK)"],
        repealed=True,
    )
    assert (section.title, section.hierarchy, section.madde_no) == ("Amaç", ["Bölüm"], "1")
    assert (section.page_start, section.page_end, section.repealed) == (2, 3, True)
