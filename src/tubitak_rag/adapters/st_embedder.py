"""Embedder backed by a local sentence-transformers model (EmbeddingGemma 2, bge-m3)."""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Context size of EmbeddingGemma 2 and bge-m3. EmbeddingGemma 2 reports no limit of its own
# to sentence-transformers, so the limit is always set explicitly.
DEFAULT_MAX_SEQ_LENGTH = 8192
# Activation memory grows with (texts in batch) x (longest text): a batch of 32 chunks of
# ~5K tokens needs >9 GiB. Batches are therefore also capped by this padded-token budget.
DEFAULT_MAX_BATCH_TOKENS = 16384
# EmbeddingGemma ships asymmetric prompts: the query and the document side differ.
_GEMMA_QUERY_PROMPT = "Retrieval-query"
_GEMMA_DOCUMENT_PROMPT = "Retrieval-document"
# Loading only the text tower of EmbeddingGemma 2 keeps it at ~270M parameters.
_TEXT_ONLY_CONFIG = {"vision_config": None, "audio_config": None}


def cuda_available() -> bool:
    """Whether PyTorch can use a CUDA GPU (False without torch installed)."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(device: str) -> str:
    """Map ``"auto"`` to ``"cuda"`` when a GPU is usable, else ``"cpu"`` (NFR-HW2)."""
    if device != "auto":
        return device
    return "cuda" if cuda_available() else "cpu"


def is_embeddinggemma(model: str) -> bool:
    """Return whether ``model`` is an EmbeddingGemma variant (needs retrieval prompts)."""
    return "embeddinggemma" in model.lower()


def _checked_dimension(model: str, native: int, requested: int | None) -> int:
    """Return the output dimension; only Matryoshka models (EmbeddingGemma) may be truncated.

    Truncating a model that was not trained for it (bge-m3) would silently produce poor
    vectors, so any other mismatch with the native size is an error.
    """
    if requested is None or requested == native:
        return native
    if is_embeddinggemma(model) and 0 < requested < native:
        return requested
    raise ValueError(
        f"{model} modelinin doğal boyutu {native}; yapılandırmadaki boyut {requested} "
        "kullanılamaz (yalnızca EmbeddingGemma daha küçük bir Matryoshka boyutuna "
        "kırpılabilir). embedder.dimension değerini düzeltin."
    )


class SentenceTransformerEmbedder:
    """Embeds with ``SentenceTransformer.encode``; vectors are L2-normalised (cosine = dot).

    Texts are sorted by token length and grouped so that a batch holds at most
    ``batch_size`` texts and at most ``max_batch_tokens`` padded tokens; results are
    returned in input order.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        batch_size: int = 32,
        text_only: bool = True,
        dimension: int | None = None,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self._batch_size = batch_size
        self._max_batch_tokens = max_batch_tokens
        self._gemma = is_embeddinggemma(model)
        kwargs: dict[str, Any] = {"device": resolve_device(device)}
        if self._gemma and text_only:
            kwargs["config_kwargs"] = dict(_TEXT_ONLY_CONFIG)
        self._model = SentenceTransformer(model, **kwargs)
        native = self._model.get_embedding_dimension()  # before any truncation
        if native is None:
            raise ValueError(f"{model} modelinin embedding boyutu belirlenemedi.")
        self.dimension: int = _checked_dimension(model, native, dimension)
        if self.dimension < native:
            self._model.truncate_dim = self.dimension  # Matryoshka: keep the first dims
        model_limit = self._model.max_seq_length
        self.max_seq_length = min(max_seq_length, model_limit) if model_limit else max_seq_length
        self._model.max_seq_length = self.max_seq_length

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk texts with the document-side prompt (if the model has one)."""
        return self._encode(texts, _GEMMA_DOCUMENT_PROMPT if self._gemma else None)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with the query-side prompt (if the model has one)."""
        return self._encode([text], _GEMMA_QUERY_PROMPT if self._gemma else None)[0]

    def _encode(self, texts: Sequence[str], prompt_name: str | None) -> list[list[float]]:
        if not texts:
            return []
        lengths = self._token_lengths(texts, prompt_name)
        rows: list[list[float]] = [[] for _ in texts]
        for group in self._batches(lengths):
            vectors = self._model.encode(
                [texts[i] for i in group],
                prompt_name=prompt_name,
                batch_size=len(group),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for i, row in zip(group, _renormalize(vectors), strict=True):
                rows[i] = row
        return rows

    def _token_lengths(self, texts: Sequence[str], prompt_name: str | None) -> list[int]:
        """Token count of each text with its prompt; logs texts that will be truncated."""
        prompt = (self._model.prompts.get(prompt_name) if prompt_name else None) or ""
        encoded = self._model.tokenizer([prompt + text for text in texts], add_special_tokens=True)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        too_long = [n for n in lengths if n > self.max_seq_length]
        if too_long:
            logger.warning(
                "%d metin %d token sınırını aşıyor ve kesilerek embed edilecek (en uzun %d token).",
                len(too_long),
                self.max_seq_length,
                max(too_long),
            )
        return [min(n, self.max_seq_length) for n in lengths]

    def _batches(self, lengths: Sequence[int]) -> list[list[int]]:
        """Group text indices, shortest first, within the text and padded-token limits."""
        batches: list[list[int]] = []
        current: list[int] = []
        for i in sorted(range(len(lengths)), key=lengths.__getitem__):
            padded = (len(current) + 1) * lengths[i]  # sorted: the new text is the longest
            if current and (len(current) == self._batch_size or padded > self._max_batch_tokens):
                batches.append(current)
                current = []
            current.append(i)
        batches.append(current)
        return batches


def _renormalize(vectors: Any) -> list[list[float]]:
    """L2-normalise in float64: models running in bfloat16 return norms like 1.0007."""
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1.0, norms)).tolist()  # type: ignore[no-any-return]
