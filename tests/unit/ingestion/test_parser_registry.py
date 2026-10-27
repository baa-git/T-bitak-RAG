"""Tests for parser dispatch and the PDF parser (FR-P1, FR-P9, FR-C5)."""

import json
from datetime import date
from pathlib import Path

import pytest

from tubitak_rag.domain.enums import DocType, SectionKind, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.gib_parser import GibParser
from tubitak_rag.ingestion.hashing import sha256_of
from tubitak_rag.ingestion.html_parser import HtmlParser
from tubitak_rag.ingestion.parser_registry import PdfParser, parser_for
from tubitak_rag.ingestion.pdf_text import PageText
from tubitak_rag.ingestion.xlsx_parser import XlsxParser
from tubitak_rag.ports.parser import DocumentParser


def _entry(kind: SourceKind = SourceKind.PDF) -> SourceEntry:
    return SourceEntry(
        doc_id="mevzuat-999",
        title="Örnek Destek Programı Uygulama Esasları",
        url="https://tubitak.gov.tr/x.pdf",
        local_path="x",
        kind=kind,
        program_kodu="1507",
        doc_turu=DocType.UYGULAMA_ESASLARI,
        surum_tarihi=date(2026, 1, 1),
        dil="tr",
        sha256=None,
        kaynak_grubu="tubitak_mevzuat",
        kaynak_sayfa_url=None,
    )


@pytest.mark.parametrize(
    "kind,parser_type",
    [
        (SourceKind.PDF, PdfParser),
        (SourceKind.HTML, HtmlParser),
        (SourceKind.GIB_JSON, GibParser),
        (SourceKind.XLSX, XlsxParser),
    ],
)
def test_parser_for_each_kind(kind: SourceKind, parser_type: type) -> None:
    parser = parser_for(_entry(kind))
    assert isinstance(parser, parser_type)
    assert isinstance(parser, DocumentParser)


def test_pdf_parser_marks_scanned(fixtures: Path) -> None:
    doc = PdfParser().parse(_entry(), fixtures / "pdf" / "bos.pdf")
    assert doc.scanned is True and doc.sections == [] and doc.page_count == 1
    assert doc.sha256 == sha256_of(fixtures / "pdf" / "bos.pdf")


def test_pdf_parser_falls_back_to_guide_when_few_articles(fixtures: Path) -> None:
    doc = PdfParser().parse(_entry(), fixtures / "pdf" / "kilavuz.pdf")
    assert doc.structured is False and doc.scanned is False and doc.page_count == 2
    assert [s.title for s in doc.sections] == [
        "1. Giris",
        "2. Basvuru Kosullari",
        "3. Destek Orani",
    ]
    assert all(s.kind == SectionKind.BOLUM for s in doc.sections)


def test_pdf_parser_parses_legislation(fixtures: Path) -> None:
    data = json.loads((fixtures / "mevzuat_pages.json").read_text(encoding="utf-8"))
    pages = [PageText(**page) for page in data]
    parser = PdfParser(extract=lambda _path: pages)
    doc = parser.parse(_entry(), fixtures / "pdf" / "metin.pdf")
    assert doc.structured is True and doc.page_count == 5
    assert [s.madde_no for s in doc.sections] == ["1", "2", "3", "4", "Geçici 1", "5"]
    full = doc.full_text()
    assert "KANUN VE YÖNETMELİKLER III-8" not in full and "tıklayınız" not in full  # FR-P3


def test_pdf_parser_moves_page_footnotes_to_notes(fixtures: Path) -> None:
    note = "1 “Eski başlık” şeklindeki başlık 5 sayılı YK'da değiştirilmiştir."
    pages = [
        PageText(number=1, lines=["Amaç1", "MADDE 1- (1) Bir.", "MADDE 2- (1) İki ve", note]),
        PageText(number=2, lines=["devam eder.", "MADDE 3- (1) " + "Uzun metin. " * 20]),
    ]
    doc = PdfParser(extract=lambda _path: pages).parse(_entry(), fixtures / "pdf" / "metin.pdf")
    assert doc.sections[1].text == "(1) İki ve devam eder."
    assert doc.sections[0].amendment_notes == [note]
