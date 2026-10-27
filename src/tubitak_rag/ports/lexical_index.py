"""Lexical index port: keyword (BM25-style) search over chunk texts."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from tubitak_rag.domain.chunk import Chunk


@runtime_checkable
class LexicalIndex(Protocol):
    """A persistable keyword index that returns chunk ids with scores."""

    def build(self, chunks: Sequence[Chunk]) -> None:
        """(Re)build the index from scratch for ``chunks``."""
        ...

    def save(self, directory: Path) -> None:
        """Persist the index into ``directory``."""
        ...

    def load(self, directory: Path) -> None:
        """Load an index previously written by :meth:`save`."""
        ...

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs, best first."""
        ...

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        ...
