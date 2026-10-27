"""Tests for the structure-aware chunker (FR-C2 … FR-C7, FR-P5, FR-P6, FR-P8)."""

import logging
import re
from collections.abc import Sequence

import pytest

from tests.fixtures.documents import (
    make_article,
    make_document,
    make_guide_doc,
    make_legislation_doc,
    make_long_article_doc,
    make_program_page_doc,
    make_section,
)
from tubitak_rag.chunking.structural import StructuralChunker, pack_paragraphs, render_paragraph
from tubitak_rag.chunking.tokens import RegexTokenCounter
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document, Paragraph
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.ports import Chunker


def _chunk(doc: Document, max_tokens: int = 450, context_header: bool = True) -> list[Chunk]:
    return StructuralChunker(RegexTokenCounter(), max_tokens, context_header).chunk(doc)


def test_is_a_chunker() -> None:
    chunker = StructuralChunker(RegexTokenCounter())
    assert isinstance(chunker, Chunker) and chunker.name == "structural"


# --- packing ------------------------------------------------------------------------------


def test_render_paragraph_keeps_label() -> None:
    assert render_paragraph(Paragraph(label="(2)", text="Metin.", page=1)) == "(2) Metin."
    assert render_paragraph(Paragraph(label=None, text="Metin.", page=1)) == "Metin."


def test_pack_paragraphs_is_greedy_and_ordered() -> None:
    paragraphs = [Paragraph(label=None, text=" ".join(["k"] * n), page=1) for n in (4, 4, 3, 6)]
    groups = pack_paragraphs(paragraphs, RegexTokenCounter(), max_tokens=8)
    assert [[len(p.text.split()) for p in group] for group in groups] == [[4, 4], [3], [6]]


class _NewlineCounter:
    """Counts whitespace-separated words and every newline (like a subword tokenizer)."""

    def encode(self, text: str) -> list[int]:
        return list(range(self.count(text)))

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join("x" for _ in ids)

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+|\n", text))


def test_pack_paragraphs_counts_the_separator_tokens() -> None:
    paragraphs = [Paragraph(label=None, text="a b c d", page=1) for _ in range(3)]
    joined_by_newline = pack_paragraphs(paragraphs, _NewlineCounter(), 9, separator="\n")
    assert [len(group) for group in joined_by_newline] == [2, 1]  # 4 + 1 + 4 = 9
    joined_by_blank_line = pack_paragraphs(paragraphs, _NewlineCounter(), 9, separator="\n\n")
    assert [len(group) for group in joined_by_blank_line] == [1, 1, 1]  # 4 + 2 + 4 = 10


def test_split_chunks_stay_within_the_limit_with_separators() -> None:
    doc = make_long_article_doc(12)
    chunker = StructuralChunker(_NewlineCounter(), max_tokens=60)
    assert all(_NewlineCounter().count(c.text) <= 60 for c in chunker.chunk(doc))


def test_pack_paragraphs_empty() -> None:
    assert pack_paragraphs([], RegexTokenCounter(), max_tokens=8) == []


# --- articles -----------------------------------------------------------------------------


def test_each_article_is_one_chunk() -> None:
    chunks = _chunk(make_legislation_doc())
    assert [c.madde_no for c in chunks] == ["5", "6", "7", "8", "Geçici 1"]
    assert chunks[2].text == (
        "(1) Destek oranı her dönem için en fazla %75 olarak belirlenmiştir.\n"
        "(2) Bütçe sınırı Yönetim Kurulu kararı ile uygulanır."
    )


def test_long_article_split_at_paragraph_boundaries() -> None:
    doc = make_long_article_doc(12)  # 12 fıkras × 15 tokens
    article = doc.sections[0]
    fikras = [render_paragraph(p) for p in article.paragraphs]
    chunks = _chunk(doc, max_tokens=60)
    assert len(chunks) == 3
    covered: list[str] = []
    for c in chunks:
        lines = c.text.split("\n")
        assert all(line in fikras for line in lines)  # no fıkra is cut in the middle
        assert RegexTokenCounter().count(c.text) <= 60
        covered += lines
    assert covered == fikras
    assert [c.hiyerarsi.rsplit(" › ", 1)[-1] for c in chunks] == ["(1)", "(5)", "(9)"]
    assert all(c.madde_no == "7" and c.sayfa == 6 for c in chunks)


def test_split_part_starting_with_a_bent_is_labelled_with_its_fikra() -> None:
    paragraphs = [
        ("(1)", "Destek kapsamındaki aşamalar şunlardır:"),
        ("a)", "kavram geliştirme ve yapılabilirlik etüdü,"),
        ("b)", "prototip üretimi ve saha testleri."),
        ("(2)", "Yatırım ağırlıklı projeler desteklenmez."),
    ]
    doc = make_document([make_article(1, "9", "Aşamalar", paragraphs)])
    chunks = _chunk(doc, max_tokens=12)
    assert [c.text.split(" ", 1)[0] for c in chunks] == ["(1)", "a)", "b)", "(2)"]
    assert [c.hiyerarsi.rsplit(" › ", 1)[-1] for c in chunks] == ["(1)", "(1)", "(1)", "(2)"]


def test_oversized_single_paragraph_kept_whole(caplog: pytest.LogCaptureFixture) -> None:
    long_text = " ".join(f"k{i}" for i in range(100))
    doc = make_document(
        [make_article(1, "3", "Tanımlar", [("(1)", long_text), ("(2)", "kısa fıkra")])]
    )
    with caplog.at_level(logging.WARNING, logger="tubitak_rag.chunking.structural"):
        chunks = _chunk(doc, max_tokens=20)
    assert [c.text for c in chunks] == [f"(1) {long_text}", "(2) kısa fıkra"]
    assert "bölünmeden" in caplog.text and "mevzuat-031:0001" in caplog.text


def test_long_article_with_one_paragraph_is_one_chunk() -> None:
    long_text = " ".join(f"k{i}" for i in range(100))
    doc = make_document([make_article(1, "3", "Tanımlar", [(None, long_text)])])
    (chunk,) = _chunk(doc, max_tokens=20)
    assert chunk.text == long_text and chunk.hiyerarsi.endswith("Madde 3 – Tanımlar")


# --- program pages, guides, tables --------------------------------------------------------


def test_faq_pair_single_chunk() -> None:
    chunks = [
        c
        for c in _chunk(make_program_page_doc(), max_tokens=5)
        if c.bolum == "Kimler başvurabilir?"
    ]
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Soru: Kimler başvurabilir?\nCevap: Türkiye'de")
    assert chunks[0].hiyerarsi == "Sıkça Sorulan Sorular › Kimler başvurabilir?"


def test_html_sections_separate_chunks() -> None:
    chunks = _chunk(make_program_page_doc())
    assert [c.bolum for c in chunks] == [
        "Program Hakkında",
        "Destek Oranı",
        "Kimler başvurabilir?",
        "Başvuru ne zaman yapılır?",
    ]
    assert (
        chunks[0].text
        == "1501 programı sanayi Ar-Ge projelerini destekler.\n\nBaşvurular süreklidir."
    )
    assert all(c.sayfa is None and c.madde_no is None for c in chunks)


@pytest.mark.parametrize("kind", [SectionKind.BOLUM, SectionKind.TABLO])
def test_long_heading_or_table_section_packed_by_paragraphs(kind: SectionKind) -> None:
    rows = [f"Satır {i} — 1501: {i} AY; 1507: {i} AY" for i in range(1, 7)]  # 12 tokens each
    doc = make_document([make_section(1, kind, "Limitler", rows, page=5)])
    chunks = _chunk(doc, max_tokens=25)
    assert [c.text for c in chunks] == [
        "\n\n".join(rows[0:2]),
        "\n\n".join(rows[2:4]),
        "\n\n".join(rows[4:6]),
    ]
    assert all(c.bolum == "Limitler" and c.hiyerarsi == "Limitler" for c in chunks)


def test_guide_paragraphs_packed() -> None:
    chunks = _chunk(make_guide_doc(10), max_tokens=50)  # 12 tokens each -> 4 + 4 + 2
    assert [c.text.count("Paragraf") for c in chunks] == [4, 4, 2]
    assert chunks[0].text.startswith("Paragraf 1 ") and chunks[2].text.endswith("açıklar .")
    assert [c.sayfa for c in chunks] == [1, 2, 3]
    assert [c.sira for c in chunks] == [0, 1, 2]


def test_paragraph_runs_do_not_cross_headings() -> None:
    doc = make_document(
        [
            make_section(1, SectionKind.PARAGRAF, None, ["birinci paragraf"], page=1),
            make_section(2, SectionKind.PARAGRAF, None, ["ikinci paragraf"], page=1),
            make_section(3, SectionKind.BOLUM, "2. Kapsam", ["başlık gövdesi"], page=2),
            make_section(4, SectionKind.PARAGRAF, None, ["üçüncü paragraf"], page=3),
        ]
    )
    assert [c.text for c in _chunk(doc)] == [
        "birinci paragraf\n\nikinci paragraf",
        "başlık gövdesi",
        "üçüncü paragraf",
    ]


# --- exclusion and metadata ---------------------------------------------------------------


def test_repealed_skipped() -> None:
    doc = make_document(
        [
            make_article(1, "1", "Amaç", [("(1)", "Yürürlükte.")]),
            make_article(2, "2", None, [], notes=["(Mülga: 21/11/2019-14YK)"], repealed=True),
            make_article(3, "3", "Eski", [("(1)", "Mülga metin.")], repealed=True),
        ]
    )
    assert [c.madde_no for c in _chunk(doc)] == ["1"]


def test_metadata_traceability() -> None:
    chunks = _chunk(make_legislation_doc())
    article = chunks[2]
    assert article.hiyerarsi == (
        "İkinci Kısım – Başvuru › Birinci Bölüm – Çağrı › Madde 7 – Destek oranı ve sınırı"
    )
    assert (article.madde_no, article.sayfa, article.bolum) == ("7", 3, None)
    assert article.kaynak_url == "https://tubitak.gov.tr/1507_ue.pdf"
    assert all(c.chunker == "structural" for c in chunks)
    assert [c.sira for c in chunks] == list(range(5))
    assert len({c.chunk_id for c in chunks}) == 5


def test_amendment_notes_in_metadata_not_text() -> None:
    article = _chunk(make_legislation_doc())[2]
    assert article.degisiklik_notlari == ["(Değişik:29/12/2025-77YK)"]
    assert "Değişik" not in article.text and "Değişik" not in article.embed_text


# --- context header -----------------------------------------------------------------------


def test_embed_text_starts_with_doc_and_hierarchy() -> None:
    article = _chunk(make_legislation_doc())[2]
    assert article.embed_text.startswith(
        "TÜBİTAK 1507- KOBİ Ar-Ge Başlangıç Destek Programı Uygulama Esasları (1507)"
        " › İkinci Kısım – Başvuru › Birinci Bölüm – Çağrı › Madde 7 – Destek oranı ve sınırı\n"
    )
    assert article.embed_text.endswith("\n" + article.text)


def test_text_unchanged_when_header_on() -> None:
    doc = make_legislation_doc()
    on, off = _chunk(doc, context_header=True), _chunk(doc, context_header=False)
    assert [c.text for c in on] == [c.text for c in off]


def test_header_off_embed_equals_text() -> None:
    assert all(c.embed_text == c.text for c in _chunk(make_legislation_doc(), context_header=False))


def test_same_title_different_program_yields_different_embed_text() -> None:
    doc_1507 = make_legislation_doc()
    doc_1501 = doc_1507.model_copy(
        update={
            "doc_id": "mevzuat-018",
            "title": "TÜBİTAK 1501 Uygulama Esasları",
            "program_kodu": "1501",
        }
    )
    a, b = _chunk(doc_1507)[2], _chunk(doc_1501)[2]
    assert a.text == b.text
    assert a.embed_text != b.embed_text and a.chunk_id != b.chunk_id


def test_split_article_without_fikras_has_no_bent_suffix() -> None:
    paragraphs = [
        (f"{letter})", f"bent {letter} için uzunca bir açıklama metni") for letter in "abcd"
    ]
    doc = make_document([make_article(1, "3", "Tanımlar", paragraphs)])
    chunks = _chunk(doc, max_tokens=10)
    assert len(chunks) == 4
    assert all(c.hiyerarsi.endswith("Madde 3 – Tanımlar") for c in chunks)


def test_split_parts_keep_the_section_separator_verbatim() -> None:
    # xlsx rows and annex blocks are joined by a single newline in the parsed section text
    rows = [f"Satır {i} — 1501: {i} AY; 1507: {i} AY" for i in range(1, 7)]
    section = make_section(1, SectionKind.TABLO, "Limitler", rows, text="\n".join(rows))
    chunks = _chunk(make_document([section]), max_tokens=25)
    assert [c.text for c in chunks] == ["\n".join(rows[i : i + 2]) for i in (0, 2, 4)]
    assert all(c.text in section.text for c in chunks)


def test_unknown_section_separator_falls_back_to_kind() -> None:
    rows = [f"Satır {i} — 1501: {i} AY; 1507: {i} AY" for i in range(1, 7)]
    section = make_section(1, SectionKind.BOLUM, "Limitler", rows, text="Önsöz\n" + " ".join(rows))
    chunks = _chunk(make_document([section]), max_tokens=25)
    assert chunks[0].text == "\n\n".join(rows[0:2])
