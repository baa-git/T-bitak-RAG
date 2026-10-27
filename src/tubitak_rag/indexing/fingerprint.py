"""Index fingerprints: what exactly of a document is in the index.

The raw file hash (``doc_sha256``) is not enough to decide whether a document must be
re-indexed: a parser fix or a chunker change produces different chunks from the same file.
So each chunk gets a digest of its full content (text, ``embed_text`` and metadata), and a
document's fingerprint is the hash of its chunk digests. Same fingerprint == same chunks.

The vector store keeps each chunk's digest and rebuilds the fingerprints from them, so a
document that was only half written (e.g. a crash during upsert) also shows up as changed.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence

from tubitak_rag.domain.chunk import Chunk


def chunk_digest(chunk: Chunk) -> str:
    """Return the SHA-256 of the chunk's canonical JSON (every field counts)."""
    canonical = json.dumps(chunk.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def doc_fingerprint(digests: Iterable[str]) -> str:
    """Return the order-independent fingerprint of a document's chunk digests."""
    return hashlib.sha256("\n".join(sorted(digests)).encode("utf-8")).hexdigest()


def fingerprints(chunks_by_doc: Mapping[str, Sequence[Chunk]]) -> dict[str, str]:
    """Map each ``doc_id`` to the fingerprint of its current chunks."""
    return {
        doc_id: doc_fingerprint(chunk_digest(chunk) for chunk in chunks)
        for doc_id, chunks in chunks_by_doc.items()
    }
