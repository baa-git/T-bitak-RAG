"""Embedder backed by a running Ollama server (``POST /api/embed``)."""

import math
from collections.abc import Sequence

import httpx

from tubitak_rag.adapters.st_embedder import is_embeddinggemma

# Ollama embeds raw text, so EmbeddingGemma's retrieval prompts are added here explicitly
# (the sentence-transformers path selects the same prompts by name).
GEMMA_QUERY_PREFIX = "task: search result | query: "
GEMMA_DOCUMENT_PREFIX = "title: none | text: "


class EmbeddingError(RuntimeError):
    """Raised when the embedding service fails or returns unusable vectors."""


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [float(x) / norm for x in vector]


class OllamaEmbedder:
    """Embeds through Ollama in batches; vectors are L2-normalised (cosine = dot)."""

    def __init__(
        self,
        model: str,
        *,
        dimension: int,
        url: str = "http://localhost:11434",
        batch_size: int = 32,
        timeout_s: float = 120,
        client: httpx.Client | None = None,
    ) -> None:
        self.model_name = model
        self.dimension = dimension
        self._endpoint = f"{url.rstrip('/')}/api/embed"
        self._batch_size = batch_size
        self._client = client or httpx.Client(timeout=timeout_s)
        gemma = is_embeddinggemma(model)
        self._query_prefix = GEMMA_QUERY_PREFIX if gemma else ""
        self._document_prefix = GEMMA_DOCUMENT_PREFIX if gemma else ""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk texts (with the document prefix for EmbeddingGemma)."""
        inputs = [self._document_prefix + text for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), self._batch_size):
            vectors += self._embed(inputs[start : start + self._batch_size])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (with the query prefix for EmbeddingGemma)."""
        return self._embed([self._query_prefix + text])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            response = self._client.post(
                self._endpoint, json={"model": self.model_name, "input": inputs}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"Ollama embedding isteği başarısız ({self.model_name}, {self._endpoint}): {exc}"
            ) from exc
        vectors = [_l2_normalize(v) for v in response.json()["embeddings"]]
        if any(len(v) != self.dimension for v in vectors):
            raise EmbeddingError(
                f"Ollama {self.model_name} beklenmeyen boyutta vektör döndürdü "
                f"(beklenen {self.dimension}, gelen {len(vectors[0])}); "
                "embedder.dimension ayarını kontrol edin."
            )
        return vectors
