"""Tests for the package port."""

from typing import Protocol

import pytest

from tubitak_rag import ports

EXPECTED = [
    "Embedder",
    "VectorStore",
    "Chunker",
    "LexicalIndex",
    "Retriever",
    "Reranker",
    "LLM",
    "DocumentParser",
    "TokenCounter",
]


def test_ports_module_lists_all_components() -> None:
    assert ports.__all__ == EXPECTED  # NFR-MOD1


@pytest.mark.parametrize("name", EXPECTED)
def test_every_port_is_a_runtime_checkable_protocol(name: str) -> None:
    port = getattr(ports, name)
    assert issubclass(port, Protocol)
    assert getattr(port, "_is_runtime_protocol", False)
