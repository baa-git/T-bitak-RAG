"""In-memory vector store with cosine similarity, for unit tests."""

import math
from collections.abc import Sequence

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.indexing.fingerprint import fingerprints


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _matches(chunk: Chunk, filters: MetadataFilter | None) -> bool:
    if filters is None:
        return True
    if filters.program_kodu is not None and chunk.program_kodu != filters.program_kodu:
        return False
    return filters.doc_turu is None or chunk.doc_turu == filters.doc_turu


class InMemoryVectorStore:
    """Stores ``collection -> chunk_id -> (chunk, vector)`` in plain dictionaries."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, tuple[Chunk, list[float]]]] = {}

    def ensure_collection(self, name: str, dimension: int) -> None:
        self.collections.setdefault(name, {})

    def drop_collection(self, name: str) -> None:
        self.collections.pop(name, None)

    def upsert(
        self, name: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        collection = self.collections[name]
        for chunk, vector in zip(chunks, vectors, strict=True):
            collection[chunk.chunk_id] = (chunk, list(vector))

    def delete_doc(self, name: str, doc_id: str) -> None:
        collection = self.collections[name]
        for chunk_id in [cid for cid, (c, _) in collection.items() if c.doc_id == doc_id]:
            del collection[chunk_id]

    def search(
        self,
        name: str,
        vector: Sequence[float],
        k: int,
        filters: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        scored = [
            (chunk, _cosine(vector, stored))
            for chunk, stored in self.collections[name].values()
            if _matches(chunk, filters)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [ScoredChunk(chunk=c, score=s, scores={"dense": s}) for c, s in scored[:k]]

    def indexed_docs(self, name: str) -> dict[str, str]:
        by_doc: dict[str, list[Chunk]] = {}
        for chunk, _ in self.collections.get(name, {}).values():
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        return fingerprints(by_doc)

    def doc_chunk_counts(self, name: str, doc_ids: Sequence[str] | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk, _ in self.collections.get(name, {}).values():
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        if doc_ids is None:
            return counts
        return {doc_id: counts.get(doc_id, 0) for doc_id in doc_ids if name in self.collections}

    def get_chunks(self, name: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        collection = self.collections[name]
        return [collection[cid][0] for cid in chunk_ids if cid in collection]

    def count(self, name: str) -> int:
        return len(self.collections.get(name, {}))
