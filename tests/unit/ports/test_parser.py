"""Tests for the parser port."""

from pathlib import Path

from tubitak_rag.domain.document import Document
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ports.parser import DocumentParser


class _StubParser:
    def parse(self, entry: SourceEntry, path: Path) -> Document:
        raise NotImplementedError


def test_structural_stub_satisfies_protocol() -> None:
    assert isinstance(_StubParser(), DocumentParser)


def test_plain_object_is_not_a_parser() -> None:
    assert not isinstance(object(), DocumentParser)
