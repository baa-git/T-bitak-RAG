"""Dense retriever: nearest chunks to the query embedding by cosine similarity (FR-R1)."""

from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.ports import Embedder, VectorStore
from tubitak_rag.retrieval.filters import normalize_query


class DenseRetriever:
    """:class:`~tubitak_rag.ports.Retriever` over one vector-store collection.

    Metadata filters are applied inside the store (FR-R5), so ``k`` results are returned
    even when most of the collection is filtered out.
    """

    name = "dense"

    def __init__(self, embedder: Embedder, store: VectorStore, collection: str) -> None:
        self._embedder = embedder
        self._store = store
        self._collection = collection

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        """Return up to ``k`` chunks, best first, with ``scores = {"dense": cosine}``."""
        vector = self._embedder.embed_query(normalize_query(query))
        return self._store.search(self._collection, vector, k, filters)
