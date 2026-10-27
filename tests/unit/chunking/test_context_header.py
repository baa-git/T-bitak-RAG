"""Tests for the optional context header of ``embed_text`` (FR-C6)."""

from tests.fixtures.documents import make_guide_doc, make_legislation_doc
from tubitak_rag.chunking.context_header import (
    build_context_header,
    build_embed_text,
    embed_text_for,
)


def test_header_has_title_program_and_section_label() -> None:
    header = build_context_header(make_legislation_doc(), "Birinci Bölüm › Madde 7 – Destek")
    assert header == (
        "TÜBİTAK 1507- KOBİ Ar-Ge Başlangıç Destek Programı Uygulama Esasları (1507)"
        " › Birinci Bölüm › Madde 7 – Destek"
    )


def test_header_without_program_or_label() -> None:
    doc = make_guide_doc().model_copy(update={"program_kodu": None})
    assert build_context_header(doc, "") == "1501 Başvuru Kılavuzu"


def test_embed_text_is_header_line_then_text() -> None:
    assert build_embed_text("Belge › Madde 7", "(1) Metin.") == "Belge › Madde 7\n(1) Metin."


def test_embed_text_for_switches_on_flag() -> None:
    doc = make_legislation_doc()
    assert embed_text_for(doc, "Madde 7", "metin", context_header=False) == "metin"
    headed = embed_text_for(doc, "Madde 7", "metin", context_header=True)
    assert headed == f"{doc.title} (1507) › Madde 7\nmetin"
