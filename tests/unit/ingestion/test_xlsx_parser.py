"""Tests for the programme limits XLSX parser (FR-P1)."""

from pathlib import Path

import pytest

from tubitak_rag.domain.document import Document
from tubitak_rag.domain.enums import DocType, SectionKind, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.xlsx_parser import XlsxParser, cell_text


@pytest.fixture
def doc(fixtures: Path) -> Document:
    entry = SourceEntry(
        doc_id="dosya-abc",
        title="Destek üst limitleri",
        url="https://tubitak.gov.tr/x.xlsx",
        local_path="x.xlsx",
        kind=SourceKind.XLSX,
        program_kodu="1501",
        doc_turu=DocType.TABLO,
        surum_tarihi=None,
        dil="tr",
        sha256=None,
        kaynak_grubu="tubitak_program_sayfalari",
        kaynak_sayfa_url=None,
    )
    return XlsxParser().parse(entry, fixtures / "ust_limit.xlsx")


def test_one_table_section_per_sheet(doc: Document) -> None:
    assert [s.title for s in doc.sections] == ["Limitler", "1501"]
    assert all(s.kind == SectionKind.TABLO for s in doc.sections)
    assert doc.doc_turu == DocType.TABLO and doc.structured is False


def test_xlsx_limits_sheet_rows_mention_program_codes(doc: Document) -> None:
    lines = doc.sections[0].text.split("\n")
    assert lines[0] == "TEYDEB DESTEK PROGRAMLARI"
    assert "Proje Destek Süresi — 1501: 36 AY; 1507: 18 AY" in lines
    assert "Destek Oranı — 1501: KOBİ:%75 BÜYÜK:%60; 1507: KOBİ:%75" in lines


def test_xlsx_program_sheet_section(doc: Document) -> None:
    sheet = doc.sections[1]
    assert sheet.text == "Proje Süresi Üst Sınırı | Destek Oranı\n36 AY | KOBİ:%75"
    assert [p.text for p in sheet.paragraphs] == sheet.text.split("\n")
    assert sheet.hierarchy == []


@pytest.mark.parametrize(
    "value,expected",
    [(None, ""), (1501, "1501"), (3500000.0, "3500000"), (0.15, "0.15"), (" a\n b ", "a b")],
)
def test_cell_text(value: object, expected: str) -> None:
    assert cell_text(value) == expected
