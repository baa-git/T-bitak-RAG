"""Vector store backed by a Qdrant server (cosine distance, payload = full chunk).

Besides the chunk fields every payload carries ``chunk_digest``; :meth:`indexed_docs`
rebuilds document fingerprints from these digests without downloading whole payloads.
"""

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.indexing.fingerprint import chunk_digest, doc_fingerprint

if TYPE_CHECKING:
    from qdrant_client.models import Filter

_UPSERT_BATCH = 128
_SCROLL_PAGE = 1000
# Keyword payload indexes keep filtered search and delete-by-document fast.
_KEYWORD_FIELDS = ("doc_id", "program_kodu", "doc_turu")
_DIGEST_KEY = "chunk_digest"


def point_id(chunk_id: str) -> str:
    """Return the deterministic Qdrant point id of a chunk (UUIDv5 of its chunk id)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _payload(chunk: Chunk) -> dict[str, Any]:
    return {**chunk.model_dump(mode="json"), _DIGEST_KEY: chunk_digest(chunk)}


def _chunk(payload: dict[str, Any] | None) -> Chunk:
    return Chunk.model_validate({k: v for k, v in (payload or {}).items() if k != _DIGEST_KEY})


def _match(key: str, value: str) -> Any:
    from qdrant_client import models

    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def filter_from(filters: MetadataFilter | None) -> "Filter | None":
    """Translate a :class:`MetadataFilter` into a Qdrant ``Filter`` (``None`` if empty)."""
    from qdrant_client import models

    if filters is None:
        return None
    must = [
        _match(key, value)
        for key, value in (("program_kodu", filters.program_kodu), ("doc_turu", filters.doc_turu))
        if value is not None
    ]
    return models.Filter(must=must) if must else None


class QdrantVectorStore:
    """:class:`~tubitak_rag.ports.VectorStore` on Qdrant; one collection per chunker+model."""

    def __init__(self, url: str, *, timeout_s: float = 60) -> None:
        from qdrant_client import QdrantClient, models

        self._models = models
        self._client = QdrantClient(url=url, timeout=int(timeout_s))

    def ensure_collection(self, name: str, dimension: int) -> None:
        """Create the collection with cosine distance and keyword indexes if it is missing."""
        if self._client.collection_exists(collection_name=name):
            return
        models = self._models
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        for field in _KEYWORD_FIELDS:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def drop_collection(self, name: str) -> None:
        """Delete the whole collection (used by a full rebuild); no-op if it is missing."""
        if self._client.collection_exists(collection_name=name):
            self._client.delete_collection(collection_name=name)

    def upsert(
        self, name: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Insert or replace chunks in batches; the payload is the whole chunk as JSON."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks ve vectors aynı uzunlukta olmalıdır")
        points = [
            self._models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector=[float(x) for x in vector],
                payload=_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            self._client.upsert(
                collection_name=name, points=points[start : start + _UPSERT_BATCH], wait=True
            )

    def delete_doc(self, name: str, doc_id: str) -> None:
        """Delete every point whose payload ``doc_id`` equals ``doc_id``."""
        models = self._models
        self._client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[_match("doc_id", doc_id)])
            ),
            wait=True,
        )

    def search(
        self,
        name: str,
        vector: Sequence[float],
        k: int,
        filters: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        """Return the ``k`` nearest chunks by cosine similarity, best first."""
        response = self._client.query_points(
            collection_name=name,
            query=list(vector),
            query_filter=filter_from(filters),
            limit=k,
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk=_chunk(point.payload),
                score=point.score,
                scores={"dense": point.score},
            )
            for point in response.points
        ]

    def indexed_docs(self, name: str) -> dict[str, str]:
        """Map ``doc_id`` to the fingerprint of its stored chunks (empty if no collection)."""
        if not self._client.collection_exists(collection_name=name):
            return {}
        digests: dict[str, list[str]] = {}
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=name,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=["doc_id", _DIGEST_KEY],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                # A point without a digest (older layout) makes its document look changed.
                digests.setdefault(payload["doc_id"], []).append(payload.get(_DIGEST_KEY, ""))
            if offset is None:
                return {doc_id: doc_fingerprint(d) for doc_id, d in digests.items()}

    def doc_chunk_counts(self, name: str, doc_ids: Sequence[str] | None = None) -> dict[str, int]:
        """Number of stored chunks per document (web documents page and preview).

        With ``doc_ids`` each document gets one exact filtered count; without, the whole
        collection is scrolled once reading only the ``doc_id`` payload field.
        """
        if not self._client.collection_exists(collection_name=name):
            return {}
        if doc_ids is not None:
            return {
                doc_id: int(
                    self._client.count(
                        collection_name=name,
                        count_filter=self._models.Filter(must=[_match("doc_id", doc_id)]),
                        exact=True,
                    ).count
                )
                for doc_id in doc_ids
            }
        counts: dict[str, int] = {}
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=name,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            for point in points:
                doc_id = (point.payload or {})["doc_id"]
                counts[doc_id] = counts.get(doc_id, 0) + 1
            if offset is None:
                return counts

    def export_points(self, name: str) -> tuple[list[Chunk], list[list[float]]]:
        """Every chunk of the collection with its stored vector, in scroll order.

        Used by ``rag export-space`` to write the read-only NumPy copy of a collection;
        the collection itself is only read. Empty lists when the collection is missing.
        """
        chunks: list[Chunk] = []
        vectors: list[list[float]] = []
        if not self._client.collection_exists(collection_name=name):
            return chunks, vectors
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=name,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                chunks.append(_chunk(point.payload))
                # one unnamed dense vector per point (ensure_collection creates no others)
                vectors.append([float(x) for x in cast(list[float], point.vector)])
            if offset is None:
                return chunks, vectors

    def get_chunks(self, name: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id in the requested order, skipping unknown ids."""
        points = self._client.retrieve(
            collection_name=name, ids=[point_id(cid) for cid in chunk_ids], with_payload=True
        )
        found = {str(point.id): _chunk(point.payload) for point in points}
        return [found[pid] for pid in map(point_id, chunk_ids) if pid in found]

    def count(self, name: str) -> int:
        """Return the exact number of points in the collection (0 if it does not exist)."""
        if not self._client.collection_exists(collection_name=name):
            return 0
        return int(self._client.count(collection_name=name, exact=True).count)
