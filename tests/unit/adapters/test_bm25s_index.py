"""Tests for the bm25s lexical index (the real, light ``bm25s`` library is used)."""

import json
from pathlib import Path

import pytest

from tests.builders import make_chunk
from tubitak_rag.adapters.bm25s_index import Bm25sIndex
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.ports import LexicalIndex
from tubitak_rag.retrieval.turkish import TurkishTokenizer


def _corpus() -> list[Chunk]:
    texts = {
        "a:0001:00": "1507 programında destek oranı en fazla %75 olarak belirlenmiştir.",
        "a:0002:00": "Projelerin başvuru dönemleri çağrı duyurusunda belirtilir.",
        "b:0001:00": "Teknoloji geliştirme bölgelerinde personel ücretleri gelir vergisinden "
        "istisnadır.",
    }
    return [make_chunk(cid, text=text, embed_text=text) for cid, text in texts.items()]


@pytest.fixture
def index() -> Bm25sIndex:
    built = Bm25sIndex(TurkishTokenizer())
    built.build(_corpus())
    return built


def test_is_a_lexical_index() -> None:
    assert isinstance(Bm25sIndex(TurkishTokenizer()), LexicalIndex)


def test_build_search_returns_best_chunk(index: Bm25sIndex) -> None:
    results = index.search("projeler için başvuru ne zaman?", k=3)
    assert results[0][0] == "a:0002:00"
    assert all(isinstance(score, float) and score > 0 for _, score in results)


def test_search_matches_program_code(index: Bm25sIndex) -> None:
    assert [cid for cid, _ in index.search("1507 destek oranı", k=1)] == ["a:0001:00"]


def test_search_returns_only_positive_scores(index: Bm25sIndex) -> None:
    assert index.search("hava durumu", k=3) == []
    assert index.search("ve ile", k=3) == []  # only stopwords -> no tokens


def test_k_larger_than_corpus(index: Bm25sIndex) -> None:
    assert len(index.search("destek başvuru personel", k=50)) == 3


def test_len_is_number_of_chunks(index: Bm25sIndex) -> None:
    assert len(index) == 3


def test_build_uses_embed_text() -> None:
    chunk = make_chunk("a:0001:00", text="gövde", embed_text="Başlık: hibe\n\ngövde")
    built = Bm25sIndex(TurkishTokenizer())
    built.build([chunk])
    assert built.search("hibe", k=1) != []


def test_save_load_roundtrip(index: Bm25sIndex, tmp_path: Path) -> None:
    index.save(tmp_path / "bm25")
    meta = json.loads((tmp_path / "bm25" / "meta.json").read_text(encoding="utf-8"))
    assert meta == {"k1": 1.5, "b": 0.75, "stemming": True, "n": 3}
    assert json.loads((tmp_path / "bm25" / "chunk_ids.json").read_text(encoding="utf-8")) == [
        "a:0001:00",
        "a:0002:00",
        "b:0001:00",
    ]
    loaded = Bm25sIndex(TurkishTokenizer())
    loaded.load(tmp_path / "bm25")
    assert len(loaded) == 3
    assert loaded.search("personel ücreti", k=2) == index.search("personel ücreti", k=2)


def test_load_rejects_other_stemming_setting(index: Bm25sIndex, tmp_path: Path) -> None:
    index.save(tmp_path)
    with pytest.raises(ValueError, match="kök bulma"):
        Bm25sIndex(TurkishTokenizer(stemming=False)).load(tmp_path)


def test_custom_k1_b_are_saved(tmp_path: Path) -> None:
    built = Bm25sIndex(TurkishTokenizer(stemming=False), k1=1.2, b=0.5)
    built.build(_corpus())
    built.save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert (meta["k1"], meta["b"], meta["stemming"]) == (1.2, 0.5, False)


def test_empty_index_searches_nothing_and_roundtrips(tmp_path: Path) -> None:
    empty = Bm25sIndex(TurkishTokenizer())
    assert empty.search("destek", k=5) == []
    empty.build([])
    empty.save(tmp_path)
    loaded = Bm25sIndex(TurkishTokenizer())
    loaded.load(tmp_path)
    assert len(loaded) == 0 and loaded.search("destek", k=5) == []


def test_rebuild_replaces_previous_corpus(index: Bm25sIndex) -> None:
    index.build(_corpus()[:1])
    assert len(index) == 1
    assert index.search("personel", k=3) == []
