"""Tests for the fixed-size sliding-window chunker (FR-C1, FR-C6, FR-P6, FR-P8)."""

import pytest

from tests.fixtures.documents import (
    make_article,
    make_document,
    make_legislation_doc,
    make_section,
)
from tubitak_rag.chunking.fixed import FixedSizeChunker
from tubitak_rag.chunking.tokens import RegexTokenCounter
from tubitak_rag.domain.document import Document
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.ports import Chunker


def _doc_of_tokens(n: int) -> Document:
    words = " ".join(f"k{i}" for i in range(n))
    return make_document([make_section(1, SectionKind.PARAGRAF, None, [words], page=1)])


def test_is_a_chunker() -> None:
    chunker = FixedSizeChunker(RegexTokenCounter())
    assert isinstance(chunker, Chunker) and chunker.name == "fixed"


def test_windows_respect_size_and_overlap() -> None:
    counter = RegexTokenCounter()
    chunks = FixedSizeChunker(counter, size=512, overlap=64).chunk(_doc_of_tokens(1300))
    assert len(chunks) == 3
    assert all(counter.count(c.text) <= 512 for c in chunks)
    first, second, third = (c.text.split() for c in chunks)
    assert second[:64] == first[-64:] and third[:64] == second[-64:]
    assert first[0] == "k0" and third[-1] == "k1299"


def test_no_window_that_only_repeats_the_overlap() -> None:
    chunks = FixedSizeChunker(RegexTokenCounter(), size=10, overlap=2).chunk(_doc_of_tokens(18))
    assert [c.text.split()[-1] for c in chunks] == ["k9", "k17"]


def test_short_text_single_chunk() -> None:
    chunks = FixedSizeChunker(RegexTokenCounter()).chunk(_doc_of_tokens(20))
    assert len(chunks) == 1 and chunks[0].text.split() == [f"k{i}" for i in range(20)]


def test_empty_document_has_no_chunks() -> None:
    assert FixedSizeChunker(RegexTokenCounter()).chunk(make_document([])) == []


def test_repealed_sections_excluded() -> None:
    doc = make_document(
        [
            make_article(1, "1", "Amaç", [("(1)", "yürürlükte")]),
            make_article(2, "2", "Eski", [("(1)", "mülgametin")], repealed=True),
        ]
    )
    chunks = FixedSizeChunker(RegexTokenCounter()).chunk(doc)
    assert len(chunks) == 1 and "mülgametin" not in chunks[0].text


def test_page_and_hierarchy_tracked() -> None:
    # three 8-token articles ("Madde N – Başlık" + "(1) x"); windows of 8, overlap 2 start at
    # tokens 0, 6, 12, 18
    doc = make_document(
        [
            make_article(1, "5", "Kapsam", [("(1)", "alfa")], page=2),
            make_article(2, "6", "Başvuru", [("(1)", "beta")], page=3),
            make_article(3, "7", "Oran", [("(1)", "gama")], page=4),
        ]
    )
    chunks = FixedSizeChunker(RegexTokenCounter(), size=8, overlap=2).chunk(doc)
    assert [c.sayfa for c in chunks] == [2, 2, 3, 4]
    assert [c.madde_no for c in chunks] == ["5", "5", "6", "7"]
    assert chunks[3].hiyerarsi == "İkinci Kısım – Başvuru › Birinci Bölüm – Çağrı › Madde 7 – Oran"
    assert all(c.bolum is None for c in chunks)  # articles are located by madde_no


def test_section_headings_are_part_of_the_text() -> None:
    # like raw PDF text, the window sees "Madde 7 – Başlık" and block titles (fair baseline)
    doc = make_document(
        [
            make_article(1, "7", "Destek oranı ve sınırı", [("(1)", "Oran %75'tir.")]),
            make_article(2, "Geçici 1", None, [(None, "Geçiş hükmü.")]),
            make_section(3, SectionKind.BOLUM, "Kapsam", ["Blok metni."]),
            make_section(4, SectionKind.PARAGRAF, None, ["Başlıksız paragraf."]),
        ]
    )
    (chunk,) = FixedSizeChunker(RegexTokenCounter()).chunk(doc)
    assert chunk.text == (
        "Madde 7 – Destek oranı ve sınırı ( 1 ) Oran % 75 ' tir . "
        "Geçici Madde 1 Geçiş hükmü . Kapsam Blok metni . Başlıksız paragraf ."
    )


def test_heading_sections_fill_bolum() -> None:
    doc = make_document([make_section(1, SectionKind.BOLUM, "4. Kapsam", ["metin"], page=3)])
    (chunk,) = FixedSizeChunker(RegexTokenCounter()).chunk(doc)
    assert chunk.bolum == "4. Kapsam" and chunk.madde_no is None and chunk.sayfa == 3


def test_chunk_fields_filled() -> None:
    doc = make_legislation_doc()
    chunks = FixedSizeChunker(RegexTokenCounter(), size=10, overlap=2).chunk(doc)
    assert len(chunks) > 1
    assert [c.sira for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        assert c.chunker == "fixed" and c.embed_text == c.text
        assert c.doc_id == "mevzuat-031" and c.program_kodu == "1507"
        assert c.surum_tarihi == "2026-09-25" and c.doc_sha256 == "a" * 64


def test_context_header_prefixes_embed_text_only() -> None:
    doc = make_legislation_doc()
    plain = FixedSizeChunker(RegexTokenCounter(), size=10, overlap=2).chunk(doc)
    headed = FixedSizeChunker(RegexTokenCounter(), size=10, overlap=2, context_header=True)
    for without, with_header in zip(plain, headed.chunk(doc), strict=True):
        assert with_header.text == without.text
        expected = f"{doc.title} (1507) › {with_header.hiyerarsi}\n{with_header.text}"
        assert with_header.embed_text == expected


def test_rejects_overlap_not_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FixedSizeChunker(RegexTokenCounter(), size=64, overlap=64)


def test_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="size") as error:
        FixedSizeChunker(RegexTokenCounter(), size=0, overlap=0)
    assert "overlap" not in str(error.value)
