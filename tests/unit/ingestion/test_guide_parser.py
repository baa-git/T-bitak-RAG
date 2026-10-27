"""Tests for the heading/paragraph parser used for guides and call texts (FR-C5, FR-P8)."""

import pytest

from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.ingestion.guide_parser import is_heading, parse_sections, split_paragraphs
from tubitak_rag.ingestion.pdf_text import PageText

CALL_PAGES = [
    PageText(
        number=1,
        lines=[
            "1501-SANAYİ AR-GE DESTEK PROGRAMI 2026/2 ÇAĞRI DUYURUSU",
            "1. Çağrı Kodu",
            "1501-2026-2",
            "2. Çağrı Başlığı",
            "1501-Sanayi Ar-Ge Destek Programı 2026 yılı 2. Çağrısı",
            "3. Çağrı Gerekçeleri ve Amaçlar",
            "Bu çağrı kapsamında firmaların;",
            "\uf0b7 uluslararası rekabet güçlerinin artırılması,",
            "\uf0b7 güçlü ticarileşme potansiyeline sahip ürünlerin",
            "geliştirilmesi amaçlanmaktadır.",
            "KOBİ ölçeğindeki firmaların Ar-Ge kapasitesinin geliştirilmesi hedeflenmektedir.",
        ],
    ),
    PageText(
        number=2,
        lines=[
            "4. Proje Değerlendirme Süreci",
            "1. Ön değerlendirmesi uygun bulunan projelerin hakem değerlendirme süreci başlatılır.",
            "2. Kritik tespitler",
            "4.1 Eşik Değer",
            "Kurul Değerlendirme Eşik Değeri 60 puandır.",
        ],
    ),
]


@pytest.mark.parametrize(
    "line,expected",
    [
        ("1. Giriş", True),
        ("4.2 Bütçe", True),
        ("5.2.10. Proje Personeli", True),
        ("BAŞVURU KOŞULLARI", True),
        ("Destek oranı en fazla %75'tir.", False),
        ("1. Projelerin puanı ve çağrı bütçesi dikkate alınarak karar verilir.", False),
        ("2026 yılında başvurular alınır", False),
        ("1. Giriş ..........................................", False),
        ("KOBİ:%75 BÜYÜK:%60.", False),
        ("", False),
    ],
)
def test_is_heading(line: str, expected: bool) -> None:
    assert is_heading(line) is expected


def test_numbered_headings_make_sections() -> None:
    sections = parse_sections(CALL_PAGES, "dosya-abc", "1501 Çağrı")
    titles = [s.title for s in sections]
    # The caps title line has no body of its own, so it yields no section.
    assert titles[:3] == [
        "1. Çağrı Kodu",
        "2. Çağrı Başlığı",
        "3. Çağrı Gerekçeleri ve Amaçlar",
    ]
    assert all(s.kind == SectionKind.BOLUM for s in sections)
    assert sections[0].text == "1501-2026-2"
    assert sections[0].section_id == "dosya-abc:0001"


def test_list_items_out_of_sequence_are_not_headings() -> None:
    sections = parse_sections(CALL_PAGES, "d", "t")
    titles = [s.title for s in sections]
    assert "2. Kritik tespitler" not in titles  # numbering restarted inside section 4
    assert titles[-2:] == ["4. Proje Değerlendirme Süreci", "4.1 Eşik Değer"]
    assert "2. Kritik tespitler" in sections[-2].text


def test_sub_heading_hierarchy() -> None:
    sections = parse_sections(CALL_PAGES, "d", "t")
    assert sections[-1].hierarchy == ["4. Proje Değerlendirme Süreci"]
    assert sections[1].hierarchy == []


def test_all_caps_heading() -> None:
    pages = [PageText(number=1, lines=["GİRİŞ", "Metin birinci satır.", "BAŞVURU", "Metin."])]
    sections = parse_sections(pages, "d", "Belge")
    assert [s.title for s in sections] == ["GİRİŞ", "BAŞVURU"]


def test_preamble_before_first_heading_uses_document_title() -> None:
    pages = [PageText(number=1, lines=["Bu rehber başvuru içindir.", "1. Giriş", "Metin."])]
    sections = parse_sections(pages, "d", "1001 Başvuru Rehberi")
    assert [s.title for s in sections] == ["1001 Başvuru Rehberi", "1. Giriş"]


def test_table_of_contents_lines_are_dropped() -> None:
    pages = [
        PageText(number=1, lines=["1. Giriş ........ 2", "2. Başvuru ...... 4"]),
        PageText(number=2, lines=["1. Giriş", "Rehber metni.", "2. Başvuru", "Başvuru metni."]),
    ]
    sections = parse_sections(pages, "d", "t")
    assert [(s.title, s.page_start) for s in sections] == [("1. Giriş", 2), ("2. Başvuru", 2)]


def test_no_headings_gives_paragraph_sections() -> None:
    pages = [
        PageText(number=1, lines=["Birinci paragrafın ilk satırı", "devam eder."]),
        PageText(number=2, lines=["İkinci paragraf burada başlar."]),
    ]
    sections = parse_sections(pages, "d", "t")
    assert [s.kind for s in sections] == [SectionKind.PARAGRAF] * 2
    assert [s.title for s in sections] == [None, None]
    assert sections[0].text == "Birinci paragrafın ilk satırı devam eder."
    assert (sections[1].page_start, sections[1].page_end) == (2, 2)


def test_bullets_stay_in_paragraph() -> None:
    sections = parse_sections(CALL_PAGES, "d", "t")
    goals = sections[2]
    assert len(goals.paragraphs) == 2
    assert goals.paragraphs[0].text == (
        "Bu çağrı kapsamında firmaların;\n"
        "• uluslararası rekabet güçlerinin artırılması,\n"
        "• güçlü ticarileşme potansiyeline sahip ürünlerin geliştirilmesi amaçlanmaktadır."
    )


def test_split_paragraphs_on_blank_line_and_sentence_end() -> None:
    paragraphs = split_paragraphs(
        [(1, "Birinci cümle bitti."), (1, "İkinci paragraf"), (1, "sürer"), (2, ""), (2, "Üçüncü")]
    )
    assert [(p.text, p.page) for p in paragraphs] == [
        ("Birinci cümle bitti.", 1),
        ("İkinci paragraf sürer", 1),
        ("Üçüncü", 2),
    ]


def test_pages_tracked() -> None:
    sections = parse_sections(CALL_PAGES, "d", "t")
    assert (sections[2].page_start, sections[2].page_end) == (1, 1)
    assert sections[-2].page_start == 2


def test_amendment_notes_are_moved_out_of_guide_text() -> None:
    # Legislation PDFs without articles (e.g. BİTO idari ve mali esaslar) fall back to this
    # parser; their "(Değişik: …)" notes must still leave the text (FR-P5).
    pages = [
        PageText(
            number=1,
            lines=[
                "1. KAPSAM",
                "(Değişik:09/01/2024-60YK) Bu Esaslar proje destek programlarını kapsar.",
                "2. İDARİ ESASLAR (Değişik:09/06/2018-279BK)",
                "Proje süresi çağrı metninde belirtilir.",
            ],
        )
    ]
    sections = parse_sections(pages, "d", "t")
    assert sections[0].text == "Bu Esaslar proje destek programlarını kapsar."
    assert sections[0].amendment_notes == ["(Değişik:09/01/2024-60YK)"]
    assert sections[1].title == "2. İDARİ ESASLAR"
    assert sections[1].amendment_notes == ["(Değişik:09/06/2018-279BK)"]


def test_page_footnotes_attach_to_the_section_on_that_page() -> None:
    note = "1 “Eski” şeklindeki başlık değiştirilmiştir."
    pages = [
        PageText(number=1, lines=["1. Giriş", "Metin."]),
        PageText(number=2, lines=["devam.", "2. Başvuru", "Metin."], footnotes=[note]),
    ]
    sections = parse_sections(pages, "d", "t")
    assert sections[1].amendment_notes == [note]
    assert sections[0].amendment_notes == []
