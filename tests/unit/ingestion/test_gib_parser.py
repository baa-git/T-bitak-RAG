"""Tests for the GİB law JSON parser (FR-P1, FR-P5)."""

from pathlib import Path

import pytest

from tubitak_rag.domain.document import Document
from tubitak_rag.domain.enums import DocType, SectionKind, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.gib_parser import (
    GibParser,
    html_to_text,
    parse_gib_title,
    split_superseded,
)
from tubitak_rag.ingestion.hashing import sha256_of

NOTE_6676 = "(6676 sayılı Kanunun 26 ncı maddesiyle değişen ibare Yürürlük; 01.03.2016)"


@pytest.fixture
def doc(fixtures: Path) -> Document:
    entry = SourceEntry(
        doc_id="kanun-5746",
        title="Araştırma, Geliştirme ve Tasarım Faaliyetlerinin Desteklenmesi Hakkında Kanun",
        url="https://gib.gov.tr/api/gibportal/mevzuat/madde/findByKanunId?kanunId=485",
        local_path="sanayi_mevzuati/5746_kanun_gib.json",
        kind=SourceKind.GIB_JSON,
        program_kodu=None,
        doc_turu=DocType.KANUN,
        surum_tarihi=None,
        dil="tr",
        sha256=None,
        kaynak_grubu="sanayi_mevzuati",
        kaynak_sayfa_url="https://www.gib.gov.tr/mevzuat/kanun/485",
    )
    return GibParser().parse(entry, fixtures / "gib_ornek.json")


def test_gib_articles_parsed(doc: Document) -> None:
    assert [s.madde_no for s in doc.sections] == ["1", "Geçici 2"]
    assert [s.title for s in doc.sections] == ["Amaç ve kapsam", "Muafiyetler"]
    assert all(s.kind == SectionKind.MADDE for s in doc.sections)
    assert [s.section_id for s in doc.sections] == ["kanun-5746:0001", "kanun-5746:0002"]


def test_gib_document_metadata(doc: Document, fixtures: Path) -> None:
    assert doc.kind == SourceKind.GIB_JSON and doc.doc_turu == DocType.KANUN
    assert doc.sha256 == sha256_of(fixtures / "gib_ornek.json")
    # Citations should point to the readable law page, not to the JSON API.
    assert doc.kaynak_url == "https://www.gib.gov.tr/mevzuat/kanun/485"
    assert doc.page_count is None and doc.scanned is False


def test_gib_amendment_note_extracted(doc: Document) -> None:
    first = doc.sections[0]
    assert NOTE_6676 in first.amendment_notes
    assert "6676" not in first.text
    assert "(Değişik fıkra: 16/2/2016-6676/14 md.)" in doc.sections[1].amendment_notes


def test_gib_footnote_markers_removed(doc: Document) -> None:
    text = doc.sections[0].text
    assert "(*)" not in text and "(…)" not in text and "kapsar.(4)" not in text
    assert "işletmelerini, kapsar." in text  # bold "(2)" footnote marker removed
    assert "Ar-Ge merkezlerini kapsar." in text


def test_gib_paragraph_labels(doc: Document) -> None:
    assert [p.label for p in doc.sections[0].paragraphs] == ["(1)", "(2)", "b)"]
    assert doc.sections[0].paragraphs[0].text.startswith("Bu Kanunun amacı; Ar-Ge, yenilik")


def test_html_to_text_paragraphs_become_lines() -> None:
    html = "<p>(1) Bir<br/>iki</p><p>(2) Üç <strong>(**)</strong> dört.(5)</p>"
    assert html_to_text(html) == "(1) Bir\niki\n(2) Üç dört."


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Madde 1 Amaç ve kapsam", ("1", "Amaç ve kapsam", [])),
        ("Madde 3/A", ("3/A", None, [])),
        ("Madde 4(*) Kuruluş", ("4", "Kuruluş", [])),
        (
            "Madde 3 İndirim, istisna, destek ve teşvik unsurları (1)",
            ("3", "İndirim, istisna, destek ve teşvik unsurları", []),
        ),
        ("Geçici Madde 2", ("Geçici 2", None, [])),
        (
            "Ek Madde 1 (6676 sayılı Kanunun 13'üncü maddesiyle eklenen Madde)",
            ("Ek 1", None, ["(6676 sayılı Kanunun 13'üncü maddesiyle eklenen Madde)"]),
        ),
        (
            "EK MADDE 3 (7263 sayılı kanunun 6 ncı maddesiyle eklenen Ek Madde; "
            "Yürürlük; 03.02.2021)",
            (
                "Ek 3",
                None,
                ["(7263 sayılı kanunun 6 ncı maddesiyle eklenen Ek Madde; Yürürlük; 03.02.2021)"],
            ),
        ),
    ],
)
def test_parse_gib_title(title: str, expected: tuple[str, str | None, list[str]]) -> None:
    assert parse_gib_title(title) == expected


def test_unrecognised_title_is_kept_whole() -> None:
    assert parse_gib_title("Yürürlük") == (None, "Yürürlük", [])


def test_superseded_wording_is_not_current_text(doc: Document) -> None:
    # I-1: GİB appends the pre-amendment wording after the article; it is history, not law.
    text = doc.sections[0].text
    assert "değişmeden önceki" not in text and "Değişmeden önceki şekli" not in text
    assert "Kurucu Heyet" not in text and "Bakanlar Kurulu kararı" not in text  # continuation
    notes = doc.sections[0].amendment_notes
    assert "(6676 sayılı Kanunla değişmeden önceki ibare) Ar-Ge ve yenilik" in notes
    assert "Bölgeler Bakanlar Kurulu kararı ile tespit edilir." in notes
    assert "Değişmeden önceki şekli: 31/12/2023" in notes


def test_footnote_block_after_separator_goes_to_notes(doc: Document) -> None:
    section = doc.sections[1]
    assert "değiştirilmiştir" not in section.text and "–––" not in section.text
    assert section.text.endswith("Bu fıkra kapsamındaki istisna şartlara bağlanabilir.")
    assert any(n.startswith("(1) 21/3/2018 tarihli ve 7103") for n in section.amendment_notes)
    assert any(n.startswith("[2] 16/2/2016") for n in section.amendment_notes)


@pytest.mark.parametrize(
    "lines,current",
    [
        (["(1) Metin.", "(2)(6170 sayılı Kanunla değişmeden önceki bent)c) Eski"], 1),
        (["(1) Metin.", "(6676 sayılı Kanunla düzenlenmeden önceki bent) 1) Eski"], 1),
        (["(1) Metin.", "Değişmeden önceki şekli:(7033 sayılı …) ee) Eski", "ifade eder."], 1),
        (["(1) Önceki kadroları için tespit edilen tutar.", "(2) Metin."], 2),
    ],
)
def test_split_superseded(lines: list[str], current: int) -> None:
    kept, stale = split_superseded(lines)
    assert kept == lines[:current] and stale == lines[current:]
