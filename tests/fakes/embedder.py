"""Deterministic, dependency-free embedder for unit tests."""

import hashlib
import math
from collections.abc import Sequence


class FakeEmbedder:
    """Maps each text to a unit vector derived from its SHA-256 digest."""

    def __init__(self, dimension: int = 8, model_name: str = "fake") -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.calls = 0  # number of embed_documents calls (cache tests count these)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text; increments :attr:`calls` once per call."""
        self.calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query exactly like a document (no prompt differences)."""
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(self.dimension)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]
