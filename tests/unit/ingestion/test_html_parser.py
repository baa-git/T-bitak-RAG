"""Tests for the Drupal programme page parser (FR-P1, FR-P7, FR-P8)."""

from pathlib import Path

import pytest

from tubitak_rag.domain.document import Section
from tubitak_rag.domain.enums import DocType, SectionKind, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.hashing import sha256_of
from tubitak_rag.ingestion.html_parser import HtmlParser, extract_sections

PAGE_TITLE = "1501 - TÜBİTAK Sanayi Ar-Ge Projeleri Destekleme Programı"


@pytest.fixture
def html(fixtures: Path) -> str:
    return (fixtures / "program_1501.html").read_text(encoding="utf-8")


@pytest.fixture
def sections(html: str) -> list[Section]:
    return extract_sections(html, "sayfa-1501", PAGE_TITLE)


def _by_title(sections: list[Section], title: str) -> Section:
    return next(s for s in sections if s.title == title)


def test_faq_pairs_are_separate_sections(sections: list[Section]) -> None:
    faq = [s for s in sections if s.kind == SectionKind.SSS]
    assert len(faq) == 3  # FR-P7
    assert faq[0].title == "Kimler başvurabilir?"
    assert faq[0].hierarchy == ["Sıkça Sorulan Sorular"]
    assert faq[0].text == (
        "Soru: Kimler başvurabilir?\nCevap: Türkiye’de yerleşik Küçük ve Orta Büyüklükteki "
        "İşletmeler (KOBİ) ölçeğindeki sermaye şirketleri başvurabilir."
    )
    assert [p.text for p in faq[0].paragraphs] == [faq[0].text.split("Cevap: ")[1]]


def test_faq_answer_with_link_and_several_paragraphs(sections: list[Section]) -> None:
    faq = [s for s in sections if s.kind == SectionKind.SSS]
    assert "www.tubitak.gov.tr adresinden takip edilebilir." in faq[1].text
    assert faq[2].text.endswith(
        "Cevap: Çağrı duyurularında belirtilmektedir.\n\n"
        "Sağlanan destek hibe şeklindedir (geri ödemesizdir)."
    )


def test_plain_text_block_is_section(sections: list[Section]) -> None:
    first = sections[0]
    assert first.kind == SectionKind.BOLUM and first.title == PAGE_TITLE
    assert first.text.split("\n\n") == [
        "Ar-Ge, bilimsel ve teknik bilgi birikimini artırmak amacıyla, sistematik bir temelde "
        "yürütülen yenilikçi faaliyetlerdir.",
        "Program, KOBİ’lerin Ar-Ge projelerini destekler.",  # NBSP normalised
    ]


def test_titled_block_keeps_tables_and_lists(sections: list[Section]) -> None:
    support = _by_title(sections, "Destek Kapsamı")
    blocks = support.text.split("\n\n")
    assert blocks[0] == "Destek oranı, firmanın desteklenen proje sayısına göre belirlenecektir."
    assert "Firmanın desteklenen ilk 5 projesi için:" in blocks
    assert "• Destek oranı %75 ve hibe destek tutarı en fazla 20 M TL*" in blocks
    assert "Proje süresi | 36 ay" in blocks
    assert blocks[-2:] == ["a) Personel giderleri,", "b) Seyahat giderleri."]
    assert "" not in blocks


def test_inner_headings_split_block(sections: list[Section]) -> None:
    apply, review = _by_title(sections, "Başvuru"), _by_title(sections, "Değerlendirme")
    assert apply.hierarchy == ["Süreç"] and review.hierarchy == ["Süreç"]
    assert apply.text == "Proje önerileri PRODİS üzerinden TÜBİTAK’a gönderilir."
    assert all(s.title != "Süreç" for s in sections)  # no text before the first heading


def test_links_block_and_empty_blocks_ignored(sections: list[Section]) -> None:
    full = "\n".join(s.text for s in sections)
    assert "807.15 KB" not in full and "Çağrı Dokümanı" not in full
    assert "Üniversitelerin Araştırma" not in full  # navigation block (gosterim-referansi)
    assert all(s.title != "Boş Blok" for s in sections)


def test_section_hierarchy_name_present(sections: list[Section]) -> None:
    # FR-P8: every HTML section carries a section name (title or hierarchy).
    assert all(s.title or s.hierarchy for s in sections)
    assert [s.section_id for s in sections[:2]] == ["sayfa-1501:0001", "sayfa-1501:0002"]
    assert all(s.page_start is None and s.madde_no is None for s in sections)


def test_page_without_article_tag_uses_body() -> None:
    html = (
        '<div class="paragraph paragraph--type--duz-metin"><div class="field--name-field-icerik">'
        "<p>Metin</p></div></div>"
    )
    assert [s.text for s in extract_sections(html, "d", "Başlık")] == ["Metin"]


def test_parse_returns_document_with_metadata(fixtures: Path) -> None:
    path = fixtures / "program_1501.html"
    entry = SourceEntry(
        doc_id="sayfa-1501",
        title=PAGE_TITLE,
        url="https://tubitak.gov.tr/tr/destekler/sanayi/1501",
        local_path="x.html",
        kind=SourceKind.HTML,
        program_kodu="1501",
        doc_turu=DocType.PROGRAM_SAYFASI,
        surum_tarihi=None,
        dil="tr",
        sha256=None,
        kaynak_grubu="tubitak_program_sayfalari",
        kaynak_sayfa_url=None,
    )
    doc = HtmlParser().parse(entry, path)
    assert doc.doc_turu == DocType.PROGRAM_SAYFASI and doc.program_kodu == "1501"
    assert doc.kind == SourceKind.HTML and doc.sha256 == sha256_of(path)
    assert doc.structured is False and doc.scanned is False and doc.page_count is None
    assert len([s for s in doc.sections if s.kind == SectionKind.SSS]) == 3


def test_inline_text_without_paragraph_tags_is_one_block() -> None:
    html = (
        '<article><div class="paragraph paragraph--type--duz-metin">'
        '<div class="field--name-field-icerik">Başvuru <a href="#">PRODİS</a> ile yapılır.'
        "<br/>Ayrıntı <strong>çağrıda</strong>.<p>Ayrı paragraf.</p></div></div></article>"
    )
    (section,) = extract_sections(html, "d", "Başlık")
    assert section.text == "Başvuru PRODİS ile yapılır. Ayrıntı çağrıda.\n\nAyrı paragraf."
