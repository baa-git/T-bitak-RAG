"""Test doubles for external services (no network, no GPU, no model downloads)."""

from collections.abc import Callable, MutableMapping
from typing import Any

from tests.fakes.embedder import FakeEmbedder
from tests.fakes.lexical_index import FakeLexicalIndex
from tests.fakes.llm import FakeLLM
from tests.fakes.reranker import FakeReranker
from tests.fakes.tokens import RegexTokenCounter
from tests.fakes.vector_store import InMemoryVectorStore
from tubitak_rag import factory

SetItem = Callable[[MutableMapping[Any, Any], Any, Any], None]


def register_fakes(setitem: SetItem) -> None:
    """Register the "fake" providers (and the regex token counter) in the factory registries.

    ``setitem`` is normally ``monkeypatch.setitem`` so that registrations are undone after
    each test.
    """
    setitem(factory.EMBEDDERS, "fake", lambda cfg: FakeEmbedder())
    setitem(factory.VECTOR_STORES, "fake", lambda cfg: InMemoryVectorStore())
    setitem(factory.RERANKERS, "fake", lambda cfg: FakeReranker())
    setitem(factory.LEXICAL_INDEXES, "fake", lambda cfg: FakeLexicalIndex())
    setitem(factory.LLMS, "fake", lambda llm_cfg: FakeLLM([]))
    setitem(factory.TOKEN_COUNTERS, "regex", lambda _model: RegexTokenCounter())
