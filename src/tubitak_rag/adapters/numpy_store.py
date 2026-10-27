"""Read-only vector store over exported NumPy files (hosted demo without a Qdrant server).

A free Hugging Face Space has no Docker-in-Docker, so it cannot run Qdrant; Qdrant snapshots
also cannot be opened without a server. The collections are therefore exported once to plain
files (``rag export-space``) and searched in memory, which is exact and fast enough for a
corpus of a few thousand chunks (one matrix product per query). Layout per collection::

    <export_dir>/<collection>/vectors.npy   float32, N x d, rows L2-normalised
    <export_dir>/<collection>/chunks.jsonl  the N chunks (same order), one JSON per line
    <export_dir>/<collection>/meta.json     {"format", "collection", "embedder_model",
                                             "dimension", "points"}

Scores are cosine similarities, the same as Qdrant's cosine distance, so the retrieval-gate
thresholds calibrated on Qdrant apply unchanged. Writing methods raise: the store is filled
only by :func:`write_collection`.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.indexing.fingerprint import fingerprints

EXPORT_FORMAT = 1
VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"
READ_ONLY_MESSAGE = (
    "NumPy vektör deposu salt okunurdur; indeksi `rag ingest` ile Qdrant'ta kurup "
    "`rag export-space` ile dışa aktarın."
)


class ReadOnlyStoreError(NotImplementedError):
    """A writing method was called on the read-only NumPy store (message in Turkish)."""


def _normalised(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def write_collection(
    export_dir: Path,
    name: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]] | np.ndarray,
    *,
    embedder_model: str,
) -> Path:
    """Write one collection (vectors, chunks, meta) under ``export_dir/name``; return the dir.

    Raises:
        ValueError: when the lengths differ or the collection is empty.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if len(chunks) != len(matrix):
        raise ValueError("chunks ve vectors aynı uzunlukta olmalıdır")
    if not len(chunks) or matrix.ndim != 2:
        raise ValueError(f"Dışa aktarılacak koleksiyon boş: {name}")
    directory = export_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / VECTORS_FILE, _normalised(matrix).astype(np.float32))
    with (directory / CHUNKS_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
    meta = {
        "format": EXPORT_FORMAT,
        "collection": name,
        "embedder_model": embedder_model,
        "dimension": int(matrix.shape[1]),
        "points": len(chunks),
    }
    # meta.json last: a directory with meta.json is a complete collection
    temporary = directory / f"{META_FILE}.tmp"
    temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, directory / META_FILE)
    return directory


def read_meta(export_dir: Path, name: str) -> dict[str, Any] | None:
    """``meta.json`` of an exported collection, or None when it is missing or unreadable."""
    try:
        meta: dict[str, Any] = json.loads((export_dir / name / META_FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return meta


@dataclass(frozen=True)
class _Collection:
    chunks: list[Chunk]
    vectors: np.ndarray  # float32, N x d, normalised
    positions: dict[str, int]  # chunk_id -> row
    program_kodu: np.ndarray  # object arrays for vectorised filtering
    doc_turu: np.ndarray


class NumpyVectorStore:
    """:class:`~tubitak_rag.ports.VectorStore` on exported files; collections load on first use."""

    def __init__(self, export_dir: Path) -> None:
        self.export_dir = Path(export_dir)
        self._loaded: dict[str, _Collection] = {}

    # --- reading -------------------------------------------------------------------------

    def _load(self, name: str) -> _Collection | None:
        if name in self._loaded:
            return self._loaded[name]
        meta = read_meta(self.export_dir, name)
        if meta is None:
            return None
        directory = self.export_dir / name
        vectors = np.load(directory / VECTORS_FILE).astype(np.float32, copy=False)
        lines = (directory / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
        if len(chunks) != len(vectors) or len(chunks) != int(meta.get("points", -1)):
            raise ValueError(
                f"Dışa aktarılan koleksiyon tutarsız ({directory}): {len(chunks)} parça, "
                f"{len(vectors)} vektör, meta.json {meta.get('points')} diyor."
            )
        collection = _Collection(
            chunks=chunks,
            vectors=vectors,
            positions={chunk.chunk_id: row for row, chunk in enumerate(chunks)},
            program_kodu=np.array([c.program_kodu for c in chunks], dtype=object),
            doc_turu=np.array([c.doc_turu for c in chunks], dtype=object),
        )
        self._loaded[name] = collection
        return collection

    def _require(self, name: str) -> _Collection:
        collection = self._load(name)
        if collection is None:
            raise ValueError(
                f"Dışa aktarılmış koleksiyon bulunamadı: {self.export_dir / name} "
                f"({META_FILE} yok)."
            )
        return collection

    def search(
        self,
        name: str,
        vector: Sequence[float],
        k: int,
        filters: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        """Return the ``k`` chunks with the highest cosine similarity, best first."""
        collection = self._require(name)
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        scores = collection.vectors @ (query / norm if norm else query)
        mask = np.ones(len(scores), dtype=bool)
        if filters is not None and filters.program_kodu is not None:
            mask &= collection.program_kodu == filters.program_kodu
        if filters is not None and filters.doc_turu is not None:
            mask &= collection.doc_turu == filters.doc_turu
        candidates = np.flatnonzero(mask)
        if k <= 0 or not len(candidates):
            return []
        if len(candidates) > k:
            top = np.argpartition(-scores[candidates], k - 1)[:k]
            candidates = candidates[top]
        # best first; ties keep the stored order so results are deterministic
        ordered = candidates[np.lexsort((candidates, -scores[candidates]))]
        return [
            ScoredChunk(
                chunk=collection.chunks[row],
                score=float(scores[row]),
                scores={"dense": float(scores[row])},
            )
            for row in ordered
        ]

    def indexed_docs(self, name: str) -> dict[str, str]:
        """Map ``doc_id`` to the fingerprint of its stored chunks (empty if no collection)."""
        collection = self._load(name)
        if collection is None:
            return {}
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in collection.chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        return fingerprints(by_doc)

    def doc_chunk_counts(self, name: str, doc_ids: Sequence[str] | None = None) -> dict[str, int]:
        """Number of stored chunks per document (all documents, or only ``doc_ids``)."""
        collection = self._load(name)
        if collection is None:
            return {}
        counts: dict[str, int] = {}
        for chunk in collection.chunks:
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        if doc_ids is None:
            return counts
        return {doc_id: counts.get(doc_id, 0) for doc_id in doc_ids}

    def get_chunks(self, name: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id in the requested order, skipping unknown ids."""
        collection = self._load(name)
        if collection is None:
            return []
        return [
            collection.chunks[collection.positions[cid]]
            for cid in chunk_ids
            if cid in collection.positions
        ]

    def count(self, name: str) -> int:
        """Points of the collection from ``meta.json`` (0 if missing); loads no vectors."""
        meta = read_meta(self.export_dir, name)
        return int(meta.get("points", 0)) if meta else 0

    # --- writing (refused) ---------------------------------------------------------------

    def ensure_collection(self, name: str, dimension: int) -> None:
        """Accept an existing collection of the same dimension; anything else is refused."""
        meta = read_meta(self.export_dir, name)
        if meta is None or int(meta.get("dimension", -1)) != dimension:
            raise ReadOnlyStoreError(READ_ONLY_MESSAGE)

    def drop_collection(self, name: str) -> None:
        """Refused: the store is read-only."""
        raise ReadOnlyStoreError(READ_ONLY_MESSAGE)

    def upsert(
        self, name: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Refused: the store is read-only."""
        raise ReadOnlyStoreError(READ_ONLY_MESSAGE)

    def delete_doc(self, name: str, doc_id: str) -> None:
        """Refused: the store is read-only."""
        raise ReadOnlyStoreError(READ_ONLY_MESSAGE)
