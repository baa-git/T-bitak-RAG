"""Tests for query normalisation and the in-memory metadata filter (FR-R5)."""

import unicodedata

from tests.builders import make_chunk, make_scored
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.retrieval.filters import apply_filters, normalize_query


def _results() -> list[ScoredChunk]:
    return [
        make_scored(make_chunk("a", program_kodu="1507", doc_turu="uygulama_esaslari"), 3.0),
        make_scored(make_chunk("b", program_kodu="1501", doc_turu="uygulama_esaslari"), 2.0),
        make_scored(make_chunk("c", program_kodu="1507", doc_turu="program_sayfasi"), 1.0),
        make_scored(make_chunk("d", program_kodu=None, doc_turu="kanun"), 0.5),
    ]


def _ids(results: list[ScoredChunk]) -> list[str]:
    return [r.chunk.chunk_id for r in results]


def test_apply_filters_program_and_type() -> None:
    assert _ids(apply_filters(_results(), MetadataFilter(program_kodu="1507"))) == ["a", "c"]
    assert _ids(apply_filters(_results(), MetadataFilter(doc_turu="uygulama_esaslari"))) == [
        "a",
        "b",
    ]
    both = MetadataFilter(program_kodu="1507", doc_turu="program_sayfasi")
    assert _ids(apply_filters(_results(), both)) == ["c"]


def test_none_filter_passthrough() -> None:
    assert _ids(apply_filters(_results(), None)) == ["a", "b", "c", "d"]
    assert _ids(apply_filters(_results(), MetadataFilter())) == ["a", "b", "c", "d"]


def test_normalize_query_strips_and_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "  1507 destek oranı\n nedir?​ ")
    assert normalize_query(decomposed) == "1507 destek oranı nedir?"
    assert unicodedata.is_normalized("NFC", normalize_query(decomposed))
