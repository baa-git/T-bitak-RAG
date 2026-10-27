"""Tests for PDF page extraction, scanned detection and header/footer removal (FR-P3, FR-P9)."""

import json
from pathlib import Path

import pytest

from tubitak_rag.ingestion.pdf_text import (
    PAGE_NUMBER_RE,
    PageText,
    extract_pages,
    is_scanned,
    pages_from_texts,
    split_footnotes,
    strip_headers_footers,
)


def load_mevzuat_pages(fixtures: Path) -> list[PageText]:
    data = json.loads((fixtures / "mevzuat_pages.json").read_text(encoding="utf-8"))
    return [PageText(**page) for page in data]


def _all_lines(pages: list[PageText]) -> list[str]:
    return [line for page in pages for line in page.lines]


def test_extract_pages_reads_fixture_pdf(fixtures: Path) -> None:
    pages = extract_pages(fixtures / "pdf" / "metin.pdf")
    assert [p.number for p in pages] == [1, 2]
    assert pages[0].lines[0] == "TUBITAK"
    assert any(line.startswith("MADDE 1 - (1)") for line in pages[0].lines)


def test_extract_pages_removes_overprinted_duplicate_chars(fixtures: Path) -> None:
    (page,) = extract_pages(fixtures / "pdf" / "kalin.pdf")
    assert page.lines == ["Madde 1- Bu esaslar programin kurallarini belirler."]


def test_empty_page_has_no_lines(fixtures: Path) -> None:
    (page,) = extract_pages(fixtures / "pdf" / "bos.pdf")
    assert page.lines == []


def test_pages_from_texts_normalizes_and_drops_blank_lines() -> None:
    pages = pages_from_texts(["MADDE 1 -  (1)\n\n  gördüğü\u0308  \n", ""])
    assert pages == [
        PageText(number=1, lines=["MADDE 1 - (1)", "gördüğü"]),
        PageText(number=2, lines=[]),
    ]


def test_pages_from_texts_repairs_legacy_glyphs_document_wide() -> None:
    # The signal ("‹") is on page 1 only; the repair must still reach page 2 ("fl" -> "ş").
    pages = pages_from_texts(["TÜB‹TAK", "kurulufllar"])
    assert pages[1].lines == ["kuruluşlar"]


def test_is_scanned_for_empty_pdf(fixtures: Path) -> None:
    assert is_scanned(extract_pages(fixtures / "pdf" / "bos.pdf")) is True


def test_is_scanned_false_for_text_pdf(fixtures: Path) -> None:
    assert is_scanned(extract_pages(fixtures / "pdf" / "metin.pdf")) is False


def test_is_scanned_when_too_few_chars_per_page() -> None:
    # 40 characters over 2 pages, like the corpus' "proje_ozel_hesaplari.pdf".
    pages = [PageText(number=1, lines=["x" * 30]), PageText(number=2, lines=["y" * 10])]
    assert is_scanned(pages) is True
    long_pages = [PageText(number=n, lines=["z" * 150]) for n in (1, 2)]
    assert is_scanned(long_pages) is False


@pytest.mark.parametrize(
    "line,is_number",
    [
        ("7/47", True),
        ("45 / 45", True),
        ("2", True),
        ("- 3 -", True),
        ("Sayfa 4", True),
        ("MADDE 2", False),
        ("(1) Bu", False),
        ("2025 yılında", False),
    ],
)
def test_page_number_regex(line: str, is_number: bool) -> None:
    assert bool(PAGE_NUMBER_RE.search(line)) is is_number


def test_strips_repeated_header_and_page_numbers(fixtures: Path) -> None:
    lines = _all_lines(strip_headers_footers(load_mevzuat_pages(fixtures)))
    assert "KANUN VE YÖNETMELİKLER III-8" not in lines
    assert "2/5" not in lines and "4 / 5" not in lines
    assert "Başvuru" in lines
    assert any(line.startswith("MADDE 3 –") for line in lines)


def test_keeps_non_repeated_first_line(fixtures: Path) -> None:
    pages = strip_headers_footers(load_mevzuat_pages(fixtures))
    assert pages[0].lines[0] == "TÜBİTAK"


def test_removes_tiklayiniz_lines(fixtures: Path) -> None:
    lines = _all_lines(strip_headers_footers(load_mevzuat_pages(fixtures)))
    assert not any("tıklayınız" in line for line in lines)


def test_short_documents_keep_repeated_lines() -> None:
    # With fewer than 4 pages a repeated line may be content, so only page numbers go.
    pages = [PageText(number=n, lines=["ORTAK BAŞLIK", f"metin {n}", f"{n}/3"]) for n in (1, 2, 3)]
    stripped = strip_headers_footers(pages)
    assert [p.lines for p in stripped] == [["ORTAK BAŞLIK", f"metin {n}"] for n in (1, 2, 3)]


# --- page footnotes ------------------------------------------------------------------

FOOTNOTE_PAGE = PageText(
    number=29,
    lines=[
        "Tamamlanan projeler için destek sürecinin sonlandırılması ve ticarileşme",
        "izleme süreci3",
        "MADDE 40- (4) Proje sonuç raporu, kuruluş tarafından",
        "3 “Tamamlanan projeler için destek sürecinin sonlandırılması” şeklindeki başlık "
        "03/06/2021 tarihli ve 32",
        "sayılı Yönetim Kurulunda “… ticarileşme izleme süreci” şeklinde değiştirilmiştir.",
        "4 “Hakem ücretleri” şeklindeki başlık 51 sayılı Yönetim Kurulunda değiştirilmiştir.",
    ],
)


def test_split_footnotes_moves_bottom_block_out_of_the_text() -> None:
    (page,) = split_footnotes([FOOTNOTE_PAGE])
    assert page.lines == FOOTNOTE_PAGE.lines[:3]
    assert page.footnotes == [
        "3 “Tamamlanan projeler için destek sürecinin sonlandırılması” şeklindeki başlık "
        "03/06/2021 tarihli ve 32 sayılı Yönetim Kurulunda “… ticarileşme izleme süreci” "
        "şeklinde değiştirilmiştir.",
        "4 “Hakem ücretleri” şeklindeki başlık 51 sayılı Yönetim Kurulunda değiştirilmiştir.",
    ]


def test_quoted_number_line_without_amendment_wording_stays() -> None:
    page = PageText(number=1, lines=["Metin başlar:", "2 “Proje” kelimesi iki kez geçer.", "son"])
    assert split_footnotes([page]) == [page]


def test_footnote_must_be_at_the_bottom_of_the_page() -> None:
    lines = ["1 “Başlık” şeklindeki başlık değiştirilmiştir.", *[f"satır {n}" for n in range(20)]]
    page = PageText(number=1, lines=lines)
    assert split_footnotes([page])[0].lines == lines
