"""Tests for the reranker port."""

from tests.fakes.reranker import FakeReranker
from tubitak_rag.ports.reranker import Reranker


def test_fake_reranker_satisfies_protocol() -> None:
    assert isinstance(FakeReranker(), Reranker)


def test_plain_object_is_not_a_reranker() -> None:
    assert not isinstance(object(), Reranker)
