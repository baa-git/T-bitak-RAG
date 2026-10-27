"""Embedder port: turns text into dense vectors."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Embeds documents and queries into vectors of a fixed dimension."""

    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text (document side of retrieval)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return the vector of a search query (query side of retrieval)."""
        ...
