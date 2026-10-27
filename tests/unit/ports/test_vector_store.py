"""Tests for the vector_store port."""

from tests.fakes.vector_store import InMemoryVectorStore
from tubitak_rag.ports.vector_store import VectorStore


def test_fake_vector_store_satisfies_protocol() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStore)


def test_plain_object_is_not_a_vector_store() -> None:
    assert not isinstance(object(), VectorStore)
