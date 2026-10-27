"""Tests for the lexical_index port."""

from tests.fakes.lexical_index import FakeLexicalIndex
from tubitak_rag.ports.lexical_index import LexicalIndex


def test_fake_lexical_index_satisfies_protocol() -> None:
    assert isinstance(FakeLexicalIndex(), LexicalIndex)


def test_plain_object_is_not_a_lexical_index() -> None:
    assert not isinstance(object(), LexicalIndex)
