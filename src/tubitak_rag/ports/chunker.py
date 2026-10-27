"""Chunker port: splits a parsed document into retrievable chunks."""

from typing import Protocol, runtime_checkable

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document


@runtime_checkable
class Chunker(Protocol):
    """A chunking strategy identified by ``name`` (e.g. "fixed", "structural")."""

    name: str

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into ordered chunks carrying citation metadata."""
        ...
