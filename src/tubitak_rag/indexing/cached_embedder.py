"""Embedder wrapper that serves document vectors from :class:`EmbeddingCache` (FR-I5)."""

from collections.abc import Sequence

from tubitak_rag.indexing.embedding_cache import EmbeddingCache
from tubitak_rag.ports import Embedder


class CachedEmbedder:
    """Embeds only texts missing from the cache, in one inner call, keeping input order.

    The cache key includes the vector dimension: the same model truncated to another
    Matryoshka size produces different vectors. Queries are never cached; they use a
    different prompt and are not repeated across ingests.
    """

    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache
        self.model_name = inner.model_name
        self.dimension = inner.dimension
        self._cache_model = f"{inner.model_name}|{inner.dimension}d"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text; cache misses are embedded once and stored."""
        vectors = self._cache.get_many(self._cache_model, texts)
        hashes = [EmbeddingCache.text_hash(text) for text in texts]
        missing = {h: text for h, text in zip(hashes, texts, strict=True) if h not in vectors}
        if missing:
            fresh = dict(
                zip(missing, self._inner.embed_documents(list(missing.values())), strict=True)
            )
            self._cache.put_many(self._cache_model, fresh)
            vectors.update(fresh)
        return [vectors[h] for h in hashes]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query directly with the inner embedder (not cached)."""
        return self._inner.embed_query(text)
