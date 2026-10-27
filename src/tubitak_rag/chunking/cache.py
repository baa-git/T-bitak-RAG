"""Chunk cache (FR-X9): unchanged documents are not chunked again.

Chunks of one document are stored as ``<chunker_key>/<doc_id>.<sha256[:12]>.json``. Two more
inputs decide whether an entry is still valid and are recorded inside it:

* the digest of the parsed ``Document`` — a parser fix changes the sections while the raw
  file (and so ``doc.sha256``) stays the same;
* the chunker settings (``chunk_settings``): tokenizer, sizes and ``CHUNKER_VERSION``.

Bump ``CHUNKER_VERSION`` whenever a change to the chunkers alters their output.
"""

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from tubitak_rag.config import ChunkerConfig
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document
from tubitak_rag.ports import Chunker

CHUNKER_VERSION = "2"


def chunker_key(cfg: ChunkerConfig) -> str:
    """Name of the chunk variant: "structural_ctx" | "structural" | "fixed_ctx" | "fixed"."""
    return f"{cfg.name}_ctx" if cfg.context_header else cfg.name


def chunk_settings(cfg: ChunkerConfig) -> str:
    """Every setting besides the document itself that changes the produced chunks."""
    if cfg.name == "fixed":
        sizes = f"size={cfg.fixed.size},overlap={cfg.fixed.overlap}"
    else:
        sizes = f"max_tokens={cfg.max_tokens}"
    return f"v{CHUNKER_VERSION}|{chunker_key(cfg)}|{cfg.tokenizer}|{sizes}"


def _document_digest(doc: Document) -> str:
    return hashlib.sha256(doc.model_dump_json().encode("utf-8")).hexdigest()


class _CacheEntry(BaseModel):
    settings: str
    document_digest: str
    chunks: list[Chunk]


class ChunkCache:
    """Directory of chunk lists (``artifacts/cache/chunks/<chunker_key>``)."""

    def __init__(self, directory: Path, chunker_key: str, settings: str) -> None:
        self.directory = directory / chunker_key
        self.settings = settings

    def path_for(self, doc: Document) -> Path:
        """Cache file of one document version."""
        return self.directory / f"{doc.doc_id}.{doc.sha256[:12]}.json"

    def get(self, doc: Document) -> list[Chunk] | None:
        """Cached chunks, or None when absent, unreadable or produced from other inputs."""
        try:
            entry = _CacheEntry.model_validate_json(self.path_for(doc).read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return None
        valid = entry.settings == self.settings and entry.document_digest == _document_digest(doc)
        return entry.chunks if valid else None

    def put(self, doc: Document, chunks: Sequence[Chunk]) -> None:
        """Store the chunks of ``doc``."""
        entry = _CacheEntry(
            settings=self.settings, document_digest=_document_digest(doc), chunks=list(chunks)
        )
        path = self.path_for(doc)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        try:  # write, then rename: a crash never leaves a half-written entry behind
            temporary.write_text(entry.model_dump_json(), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def chunk_documents(
    docs: Sequence[Document], chunker: Chunker, cache: ChunkCache | None
) -> dict[str, list[Chunk]]:
    """Chunk every document (doc_id -> chunks); cache hits skip the chunker."""
    result: dict[str, list[Chunk]] = {}
    for doc in docs:
        chunks = cache.get(doc) if cache else None
        if chunks is None:
            chunks = chunker.chunk(doc)
            if cache:
                cache.put(doc, chunks)
        result[doc.doc_id] = chunks
    return result
