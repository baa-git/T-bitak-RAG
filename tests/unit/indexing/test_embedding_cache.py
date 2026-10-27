"""Tests for the SQLite embedding cache keyed by (model, text hash) (FR-I5)."""

import hashlib
from pathlib import Path

import pytest

from tubitak_rag.indexing.embedding_cache import EmbeddingCache


def test_text_hash_is_sha256_of_utf8() -> None:
    assert (
        EmbeddingCache.text_hash("destek oranı")
        == hashlib.sha256("destek oranı".encode()).hexdigest()
    )


def test_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "embeddings" / "cache.sqlite"
    EmbeddingCache(path)
    assert path.is_file()


def test_cache_roundtrip_float32(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    key = EmbeddingCache.text_hash("a")
    cache.put_many("m", {key: [0.1, -0.2, 0.3]})
    got = cache.get_many("m", ["a", "b"])
    assert list(got) == [key]
    assert got[key] == pytest.approx([0.1, -0.2, 0.3], abs=1e-7)  # stored as float32
    assert all(isinstance(x, float) for x in got[key])


def test_values_survive_reopening(tmp_path: Path) -> None:
    EmbeddingCache(tmp_path / "c.sqlite").put_many("m", {EmbeddingCache.text_hash("a"): [1.0]})
    assert EmbeddingCache(tmp_path / "c.sqlite").get_many("m", ["a"]) != {}


def test_put_replaces_existing_vector(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    key = EmbeddingCache.text_hash("a")
    cache.put_many("m", {key: [1.0]})
    cache.put_many("m", {key: [2.0]})
    assert cache.get_many("m", ["a"]) == {key: [2.0]}


def test_lookup_is_per_model(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    cache.put_many("m1", {EmbeddingCache.text_hash("a"): [1.0]})
    assert cache.get_many("m2", ["a"]) == {}


def test_many_texts_beyond_sql_variable_limit(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    texts = [f"t{i}" for i in range(2500)]
    cache.put_many("m", {EmbeddingCache.text_hash(t): [float(i)] for i, t in enumerate(texts)})
    assert len(cache.get_many("m", texts)) == 2500


def test_empty_inputs(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    cache.put_many("m", {})
    assert cache.get_many("m", []) == {}
