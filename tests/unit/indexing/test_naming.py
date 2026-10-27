"""Tests for collection and BM25 directory naming (FR-I1)."""

from pathlib import Path

from tubitak_rag.indexing.naming import bm25_dir, collection_name, slug


def test_collection_name_combines_chunker_and_gemma_model() -> None:
    assert (
        collection_name("structural_ctx", "google/embeddinggemma-2")
        == "structural_ctx__google-embeddinggemma-2"
    )


def test_collection_name_lowercases_model() -> None:
    assert collection_name("fixed", "BAAI/bge-m3") == "fixed__baai-bge-m3"


def test_slug_replaces_other_characters_with_underscore() -> None:
    assert slug("embeddinggemma-2:270m") == "embeddinggemma-2_270m"


def test_bm25_dir_is_per_chunker_key() -> None:
    assert bm25_dir(Path("artifacts"), "fixed_ctx") == Path("artifacts/cache/bm25/fixed_ctx")
