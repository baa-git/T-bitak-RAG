"""Tests for the chunker port."""

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document
from tubitak_rag.ports.chunker import Chunker


class _NoopChunker:
    name = "noop"

    def chunk(self, doc: Document) -> list[Chunk]:
        return []


def test_structural_stub_satisfies_protocol() -> None:
    assert isinstance(_NoopChunker(), Chunker)


def test_plain_object_is_not_a_chunker() -> None:
    assert not isinstance(object(), Chunker)
