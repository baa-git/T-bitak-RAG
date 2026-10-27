"""Tests for chunk ids and shared chunk metadata helpers (FR-C7)."""

import re

from tests.builders import make_chunk
from tests.fixtures.documents import (
    make_article,
    make_legislation_doc,
    make_program_page_doc,
    make_section,
)
from tubitak_rag.chunking.ids import (
    article_name,
    base_chunk_fields,
    chunk_id_for,
    hierarchy_label,
    section_bolum,
)
from tubitak_rag.domain.enums import SectionKind


def test_chunk_id_is_16_hex() -> None:
    chunk_id = chunk_id_for("mevzuat-031", "structural", 0, "metin")
    assert re.fullmatch(r"[0-9a-f]{16}", chunk_id)
    assert chunk_id == chunk_id_for("mevzuat-031", "structural", 0, "metin")


def test_chunk_ids_unique_across_docs_with_same_text() -> None:
    text = "Destek oranı ve sınırı"
    ids = {
        chunk_id_for("mevzuat-031", "structural", 7, text),
        chunk_id_for("mevzuat-018", "structural", 7, text),
        chunk_id_for("mevzuat-031", "fixed", 7, text),
        chunk_id_for("mevzuat-031", "structural", 8, text),
    }
    assert len(ids) == 4


def test_base_fields_match_spec_schema() -> None:
    fields = base_chunk_fields(make_legislation_doc(), "structural")
    assert fields == {
        "doc_id": "mevzuat-031",
        "doc_title": "TÜBİTAK 1507- KOBİ Ar-Ge Başlangıç Destek Programı Uygulama Esasları",
        "program_kodu": "1507",
        "doc_turu": "uygulama_esaslari",
        "kaynak_url": "https://tubitak.gov.tr/1507_ue.pdf",
        "surum_tarihi": "2026-09-25",
        "dil": "tr",
        "doc_sha256": "a" * 64,
        "chunker": "structural",
    }
    # together with the per-chunk fields they build a valid Chunk (spec §7.3)
    per_chunk = {"hiyerarsi", "madde_no", "sayfa", "bolum", "text", "embed_text", "sira"}
    assert set(make_chunk().model_dump()) == set(fields) | per_chunk | {
        "chunk_id",
        "degisiklik_notlari",
    }


def test_base_fields_without_version_date() -> None:
    assert base_chunk_fields(make_program_page_doc(), "fixed")["surum_tarihi"] is None


def test_article_name_variants() -> None:
    assert article_name("7") == "Madde 7"
    assert article_name("5/A") == "Madde 5/A"
    assert article_name("Geçici 16") == "Geçici Madde 16"
    assert article_name("Ek 1") == "Ek Madde 1"


def test_hierarchy_label_formats() -> None:
    article = make_article(1, "7", "Destek oranı", [("(1)", "x")])
    assert hierarchy_label(article) == (
        "İkinci Kısım – Başvuru › Birinci Bölüm – Çağrı › Madde 7 – Destek oranı"
    )
    untitled = make_article(1, "Geçici 1", None, [(None, "x")], hierarchy=[])
    assert hierarchy_label(untitled) == "Geçici Madde 1"
    faq = make_section(1, SectionKind.SSS, "Kimler?", ["y"], hierarchy=["SSS"])
    assert hierarchy_label(faq) == "SSS › Kimler?"
    assert hierarchy_label(make_section(1, SectionKind.PARAGRAF, None, ["z"])) == ""


def test_section_bolum_is_title_except_for_articles() -> None:
    assert section_bolum(make_article(1, "7", "Destek oranı", [("(1)", "x")])) is None
    heading = make_section(1, SectionKind.BOLUM, "4. Kapsam", ["z"])
    assert section_bolum(heading) == "4. Kapsam"
    assert section_bolum(make_section(1, SectionKind.PARAGRAF, None, ["p"])) is None
