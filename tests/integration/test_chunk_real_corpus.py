"""Chunk statistics on the real parsed corpus with the EmbeddingGemma 2 tokenizer.

Run with ``pytest tests/integration/test_chunk_real_corpus.py -m integration -s``; the printed
statistics go into the development log. Skipped when ``data/raw`` is absent or the tokenizer
cannot be loaded. Documents come from the parse cache (``artifacts/cache/parsed``).
"""

import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest

from tubitak_rag.chunking.cache import chunk_documents
from tubitak_rag.chunking.fixed import FixedSizeChunker
from tubitak_rag.chunking.structural import StructuralChunker, render_paragraph
from tubitak_rag.config import ChunkerConfig
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.ingestion.manifest import load_sources
from tubitak_rag.ingestion.parsed_cache import ParsedCache, parse_all
from tubitak_rag.ports import TokenCounter

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CFG = ChunkerConfig()  # structural, max_tokens 450, context header, hf:google/embeddinggemma-2
PAGE_1501 = (
    "sayfa-sanayi__ulusal-destek-programlari__"
    "1501-tubitak-sanayi-ar-ge-projeleri-destekleme-programi"
)
EVIDENCE_1507 = "destek oranı her dönem için en fazla %75 olarak belirlenmiştir"

ChunkMap = dict[str, list[Chunk]]


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    if not RAW.is_dir():
        pytest.skip("data/raw yok; gerçek corpus testi atlandı")
    entries = load_sources(ROOT / "data" / "manifests")
    parsed, _ = parse_all(entries, RAW, ParsedCache(ROOT / "artifacts" / "cache" / "parsed"))
    return [d for d in parsed if not d.scanned]


@pytest.fixture(scope="module")
def counter() -> TokenCounter:
    model = CFG.tokenizer.removeprefix("hf:")
    try:
        from tubitak_rag.adapters.hf_tokenizer import HfTokenCounter

        return HfTokenCounter(model)
    except Exception as exc:  # no `tokenizers` or no Hub access
        pytest.skip(f"{model} tokenizer yüklenemedi: {exc}")


@pytest.fixture(scope="module")
def structural(docs: list[Document], counter: TokenCounter) -> ChunkMap:
    chunker = StructuralChunker(counter, CFG.max_tokens, context_header=True)
    return chunk_documents(docs, chunker, None)


@pytest.fixture(scope="module")
def fixed(docs: list[Document], counter: TokenCounter) -> ChunkMap:
    chunker = FixedSizeChunker(counter, CFG.fixed.size, CFG.fixed.overlap, context_header=False)
    return chunk_documents(docs, chunker, None)


def _flat(chunks: ChunkMap) -> list[Chunk]:
    return [c for doc_chunks in chunks.values() for c in doc_chunks]


def _distribution(lengths: Sequence[int]) -> str:
    p95 = statistics.quantiles(lengths, n=20)[18]
    return (
        f"min={min(lengths)} medyan={statistics.median(lengths):.0f} "
        f"p95={p95:.0f} maks={max(lengths)}"
    )


def _split_articles(chunks: ChunkMap) -> Counter[tuple[str, str]]:
    parts: Counter[tuple[str, str]] = Counter()
    for c in _flat(chunks):
        if c.madde_no:
            parts[(c.doc_id, c.madde_no)] += 1
    return Counter({key: n for key, n in parts.items() if n > 1})


def _oversized_paragraphs(docs: list[Document], counter: TokenCounter) -> int:
    return sum(
        counter.count(render_paragraph(p)) > CFG.max_tokens
        for d in docs
        for s in d.sections
        if not s.repealed and s.kind != SectionKind.SSS and len(s.paragraphs) > 1
        for p in s.paragraphs
    )


def test_print_statistics(
    docs: list[Document], counter: TokenCounter, structural: ChunkMap, fixed: ChunkMap
) -> None:
    s_chunks, f_chunks = _flat(structural), _flat(fixed)
    print("\n== Chunk istatistikleri (tokenizer: EmbeddingGemma 2) ==")
    print(f"Belge (taranmamış): {len(docs)}")
    print(f"Sabit 512/64      : {len(f_chunks)} chunk")
    print("  text token      :", _distribution([counter.count(c.text) for c in f_chunks]))
    print(f"Yapısal 450 + başlık: {len(s_chunks)} chunk")
    print("  text token      :", _distribution([counter.count(c.text) for c in s_chunks]))
    print("  embed_text token:", _distribution([counter.count(c.embed_text) for c in s_chunks]))
    kinds = Counter(s.kind.value for d in docs for s in d.sections if not s.repealed)
    print("  bölüm türleri   :", dict(kinds))
    split = _split_articles(structural)
    print(
        f"  fıkra sınırından bölünen madde: {len(split)} "
        f"(toplam {sum(split.values())} parça, en çok {max(split.values(), default=0)})"
    )
    print(f"  sınırı tek başına aşan paragraf: {_oversized_paragraphs(docs, counter)}")
    over = sum(counter.count(c.text) > CFG.max_tokens for c in s_chunks)
    print(f"  {CFG.max_tokens} token'ı aşan chunk: {over}")
    over_embed = sum(counter.count(c.embed_text) > CFG.fixed.size for c in s_chunks)
    print(f"  embed_text'i {CFG.fixed.size} token'ı aşan chunk: {over_embed}")
    repealed = sum(s.repealed for d in docs for s in d.sections)
    print(f"  dışlanan mülga madde: {repealed}")
    empty = [d.doc_id for d in docs if not structural[d.doc_id]]
    print(f"  chunk üretmeyen belge: {len(empty)}")


def test_repealed_articles_not_chunked(docs: list[Document], structural: ChunkMap) -> None:
    for doc in docs:
        live = {s.madde_no for s in doc.sections if s.madde_no and not s.repealed}
        repealed = {s.madde_no for s in doc.sections if s.repealed} - live
        assert not {c.madde_no for c in structural[doc.doc_id]} & repealed, doc.doc_id


def test_every_structural_chunk_is_locatable(structural: ChunkMap) -> None:
    for c in _flat(structural):  # FR-P8: page, section name or article number
        assert c.sayfa is not None or c.bolum or c.madde_no, c.chunk_id


def test_faq_pairs_are_whole_chunks(docs: list[Document], structural: ChunkMap) -> None:
    for doc in docs:  # FR-C4: every question/answer pair is one whole chunk
        texts = Counter(c.text for c in structural[doc.doc_id])
        for section in doc.sections:
            if section.kind == SectionKind.SSS:
                assert texts[section.text] >= 1, section.section_id


def test_only_unsplittable_units_exceed_max_tokens(
    docs: list[Document], counter: TokenCounter, structural: ChunkMap
) -> None:
    # FR-C3: a chunk longer than max_tokens is a single paragraph, FAQ pair or 1-paragraph unit
    for doc in docs:
        whole = {s.text for s in doc.sections}
        whole |= {render_paragraph(p) for s in doc.sections for p in s.paragraphs}
        for c in structural[doc.doc_id]:
            if counter.count(c.text) > CFG.max_tokens:
                assert c.text in whole, c.chunk_id


def test_fixed_windows_respect_size(counter: TokenCounter, fixed: ChunkMap) -> None:
    # windows are cut in token space; re-tokenizing the decoded text may differ by a few tokens
    assert max(counter.count(c.text) for c in _flat(fixed)) <= CFG.fixed.size + 8


def test_examples_1507_article_7_and_1501_faq(structural: ChunkMap, fixed: ChunkMap) -> None:
    (article,) = [c for c in structural["mevzuat-031"] if c.madde_no == "7"]
    assert EVIDENCE_1507 in article.text
    assert article.embed_text.startswith("TÜBİTAK 1507")
    # a sentence shorter than the overlap is always whole in at least one window
    windows = [c for c in fixed["mevzuat-031"] if EVIDENCE_1507 in c.text]
    assert windows
    faq = next(c for c in structural[PAGE_1501] if c.text.startswith("Soru: "))
    print("\n== Örnek: 1507 madde 7, yapısal chunk ==")
    print(article.model_dump_json(indent=1, exclude={"text"}))
    print("== Örnek: 1507 madde 7'yi içeren sabit pencere(ler) ==")
    for w in windows:
        print(f"[sira={w.sira} sayfa={w.sayfa} hiyerarsi={w.hiyerarsi!r}]")
        print(w.text[:600], "…", w.text[-300:])
    print("== Örnek: 1501 SSS chunk ==")
    print(faq.model_dump_json(indent=1))
