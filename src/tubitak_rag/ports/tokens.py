"""Token counter port: the notion of a "token" used for chunk sizes."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Encodes text to token ids and back, and counts tokens."""

    def encode(self, text: str) -> list[int]:
        """Return the token ids of ``text``."""
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Return the text for ``ids``."""
        ...

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...
