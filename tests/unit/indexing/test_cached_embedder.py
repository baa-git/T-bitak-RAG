"""Tests for the caching embedder wrapper (FR-I5, FR-X9)."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from tests.fakes.embedder import FakeEmbedder
from tubitak_rag.indexing.cached_embedder import CachedEmbedder
from tubitak_rag.indexing.embedding_cache import EmbeddingCache
from tubitak_rag.ports import Embedder


class _RecordingEmbedder(FakeEmbedder):
    """FakeEmbedder that also remembers which texts each call received."""

    def __init__(self, dimension: int = 8, model_name: str = "fake") -> None:
        super().__init__(dimension=dimension, model_name=model_name)
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return super().embed_documents(texts)


def _assert_close(actual: list[list[float]], expected: list[list[float]]) -> None:
    """Cached vectors are float32, so compare with a tolerance."""
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-6)


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "cache.sqlite")


def test_is_an_embedder_with_inner_identity(cache: EmbeddingCache) -> None:
    embedder = CachedEmbedder(FakeEmbedder(dimension=4, model_name="m"), cache)
    assert isinstance(embedder, Embedder)
    assert (embedder.model_name, embedder.dimension) == ("m", 4)


def test_second_call_does_not_hit_inner(cache: EmbeddingCache) -> None:
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner, cache)
    first = embedder.embed_documents(["a", "b"])
    assert inner.calls == 1
    second = embedder.embed_documents(["a", "b"])
    assert inner.calls == 1
    _assert_close(second, first)


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    CachedEmbedder(FakeEmbedder(), EmbeddingCache(tmp_path / "c.sqlite")).embed_documents(["a"])
    inner = FakeEmbedder()
    CachedEmbedder(inner, EmbeddingCache(tmp_path / "c.sqlite")).embed_documents(["a"])
    assert inner.calls == 0


def test_partial_miss_only_embeds_missing_in_order(cache: EmbeddingCache) -> None:
    inner = _RecordingEmbedder()
    embedder = CachedEmbedder(inner, cache)
    embedder.embed_documents(["b"])
    vectors = embedder.embed_documents(["c", "b", "a"])
    assert inner.batches == [["b"], ["c", "a"]]
    _assert_close(vectors, FakeEmbedder().embed_documents(["c", "b", "a"]))


def test_duplicate_texts_are_embedded_once(cache: EmbeddingCache) -> None:
    inner = _RecordingEmbedder()
    vectors = CachedEmbedder(inner, cache).embed_documents(["a", "b", "a"])
    assert inner.batches == [["a", "b"]]
    assert vectors[0] == vectors[2]


def test_cache_is_per_model(cache: EmbeddingCache) -> None:
    CachedEmbedder(FakeEmbedder(model_name="m1"), cache).embed_documents(["a"])
    other = FakeEmbedder(model_name="m2")
    CachedEmbedder(other, cache).embed_documents(["a"])
    assert other.calls == 1


def test_cache_is_per_dimension(cache: EmbeddingCache) -> None:
    # Same model truncated to another Matryoshka size must not reuse the stored vectors.
    CachedEmbedder(FakeEmbedder(dimension=8, model_name="m"), cache).embed_documents(["a"])
    smaller = FakeEmbedder(dimension=4, model_name="m")
    vectors = CachedEmbedder(smaller, cache).embed_documents(["a"])
    assert smaller.calls == 1 and len(vectors[0]) == 4


def test_empty_input_does_not_call_inner(cache: EmbeddingCache) -> None:
    inner = FakeEmbedder()
    assert CachedEmbedder(inner, cache).embed_documents([]) == []
    assert inner.calls == 0


def test_query_not_cached(cache: EmbeddingCache) -> None:
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner, cache)
    assert embedder.embed_query("soru") == inner.embed_query("soru")
    embedder.embed_documents(["soru"])  # the query vector was not stored as a document
    assert inner.calls == 1
