"""Tests for the retriever port."""

from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.ports.retriever import Retriever


class _EmptyRetriever:
    name = "empty"

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        return []


def test_structural_stub_satisfies_protocol() -> None:
    assert isinstance(_EmptyRetriever(), Retriever)


def test_plain_object_is_not_a_retriever() -> None:
    assert not isinstance(object(), Retriever)
