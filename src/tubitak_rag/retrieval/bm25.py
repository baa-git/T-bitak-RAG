"""BM25 retriever: keyword search on the lexical index, chunks loaded from the vector store."""

import logging
from collections.abc import Sequence

from tubitak_rag.config import ConfigError
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.ports import LexicalIndex, VectorStore
from tubitak_rag.retrieval.filters import apply_filters, normalize_query

# With a filter, fetch this many times more hits than needed; grow by the same factor until
# ``k`` hits survive the filter or the index has no more matches (FR-R5).
_FILTER_OVERFETCH = 3

logger = logging.getLogger(__name__)


def check_index_matches_collection(
    index: LexicalIndex, store: VectorStore, collection: str
) -> None:
    """Refuse a BM25 index whose size differs from the collection's point count.

    ``rag ingest`` rebuilds BM25 from every chunk of the chunker, but an interrupted ingest
    (or one run for another embedding model) can leave the collection partly filled; BM25
    hits missing from the collection would then silently disappear from the results.

    Raises:
        ConfigError: when the sizes differ (message in Turkish).
    """
    points = store.count(collection)
    if len(index) != points:
        raise ConfigError(
            f"BM25 indeksinde {len(index)} parça var ama {collection!r} koleksiyonunda "
            f"{points} nokta var; indeksler tutarsız. Bu yapılandırma için `rag ingest` "
            "komutunu tamamlanana kadar çalıştırın."
        )


class Bm25Retriever:
    """:class:`~tubitak_rag.ports.Retriever` on a :class:`~tubitak_rag.ports.LexicalIndex`.

    The index only knows chunk ids, so chunks (and their metadata for filtering) come from
    the vector-store collection that was built from the same chunks.
    """

    name = "bm25"

    def __init__(self, index: LexicalIndex, store: VectorStore, collection: str) -> None:
        self._index = index
        self._store = store
        self._collection = collection

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        """Return up to ``k`` chunks, best first, with ``scores = {"bm25": score}``."""
        if k <= 0:
            return []
        query = normalize_query(query)
        fetch = k * _FILTER_OVERFETCH if filters else k
        while True:
            hits = self._index.search(query, fetch)
            results = apply_filters(self._load(hits), filters)
            if len(results) >= k or len(hits) < fetch or fetch >= len(self._index):
                return results[:k]
            fetch *= _FILTER_OVERFETCH

    def _load(self, hits: Sequence[tuple[str, float]]) -> list[ScoredChunk]:
        """Turn ``(chunk_id, score)`` hits into scored chunks; unknown ids are skipped."""
        scores = dict(hits)
        chunks = self._store.get_chunks(self._collection, [chunk_id for chunk_id, _ in hits])
        if len(chunks) < len(hits):
            logger.warning(
                "BM25: %d parça %r koleksiyonunda bulunamadı ve atlandı; indeksler tutarsız, "
                "`rag ingest` komutunu tamamlayın.",
                len(hits) - len(chunks),
                self._collection,
            )
        return [
            ScoredChunk(
                chunk=chunk,
                score=scores[chunk.chunk_id],
                scores={"bm25": scores[chunk.chunk_id]},
            )
            for chunk in chunks
        ]
