"""Tests for rebuilding and loading the persisted BM25 index (FR-I6)."""

from pathlib import Path

from tests.builders import make_chunk
from tests.fakes.lexical_index import FakeLexicalIndex
from tubitak_rag.adapters.bm25s_index import Bm25sIndex
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.indexing.bm25_index import load_bm25, rebuild_bm25
from tubitak_rag.indexing.naming import bm25_dir
from tubitak_rag.retrieval.turkish import TurkishTokenizer


def _chunks() -> list[Chunk]:
    return [
        make_chunk("a:0001:00", text="destek oranı", embed_text="destek oranı"),
        make_chunk("b:0001:00", text="personel ücreti", embed_text="personel ücreti"),
    ]


def test_rebuild_writes_directory(tmp_path: Path) -> None:
    directory = bm25_dir(tmp_path / "artifacts", "structural_ctx")
    n = rebuild_bm25(_chunks(), Bm25sIndex(TurkishTokenizer()), directory)
    assert n == 2
    assert (tmp_path / "artifacts/cache/bm25/structural_ctx/chunk_ids.json").is_file()


def test_rebuild_replaces_old_index(tmp_path: Path) -> None:
    index = FakeLexicalIndex()
    rebuild_bm25(_chunks(), index, tmp_path)
    assert rebuild_bm25(_chunks()[:1], index, tmp_path) == 1
    reloaded = load_bm25(FakeLexicalIndex(), tmp_path)
    assert reloaded.search("personel", k=5) == []


def test_load_returns_the_loaded_index(tmp_path: Path) -> None:
    rebuild_bm25(_chunks(), Bm25sIndex(TurkishTokenizer()), tmp_path)
    fresh = Bm25sIndex(TurkishTokenizer())
    loaded = load_bm25(fresh, tmp_path)
    assert loaded is fresh and len(loaded) == 2
    assert loaded.search("personel ücretleri", k=1)[0][0] == "b:0001:00"
