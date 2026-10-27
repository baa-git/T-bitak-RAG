"""Tests for the embedder port."""

from tests.fakes.embedder import FakeEmbedder
from tubitak_rag.ports.embedder import Embedder


def test_fake_embedder_satisfies_protocol() -> None:
    assert isinstance(FakeEmbedder(), Embedder)


def test_object_without_embed_query_is_not_an_embedder() -> None:
    class Partial:
        model_name = "x"
        dimension = 1

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return []

    assert not isinstance(Partial(), Embedder)
