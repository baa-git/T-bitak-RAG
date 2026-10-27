"""Real embedding models: EmbeddingGemma 2 / bge-m3 (sentence-transformers) and Ollama.

Run with ``pytest tests/integration/test_embedders.py -m integration -s``. The first run
downloads the models into the HuggingFace cache (~0.6 GB gemma, ~2.3 GB bge-m3).
"""

import math
from collections.abc import Sequence

import pytest

from tubitak_rag.adapters.ollama_embedder import OllamaEmbedder
from tubitak_rag.adapters.st_embedder import SentenceTransformerEmbedder
from tubitak_rag.ports import Embedder

pytestmark = pytest.mark.integration


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # vectors are unit length


def _assert_semantics(embedder: Embedder) -> None:
    support, grant, weather = embedder.embed_documents(
        ["destek oranı", "hibe oranı", "hava durumu"]
    )
    assert math.isclose(sum(x * x for x in support), 1.0, rel_tol=1e-4)
    assert _cos(support, grant) > _cos(support, weather)
    query = embedder.embed_query("Projeye verilen destek oranı yüzde kaç?")
    assert len(query) == embedder.dimension
    assert _cos(query, support) > _cos(query, weather)


def test_embeddinggemma2_real_768d() -> None:
    pytest.importorskip("sentence_transformers")
    embedder = SentenceTransformerEmbedder("google/embeddinggemma-2")
    assert embedder.dimension == 768
    _assert_semantics(embedder)


def test_bge_m3_real_1024d() -> None:
    pytest.importorskip("sentence_transformers")
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    assert embedder.dimension == 1024
    _assert_semantics(embedder)


def test_ollama_embeddinggemma_270m(ollama_url: str) -> None:
    embedder = OllamaEmbedder("embeddinggemma-2:270m", dimension=768, url=ollama_url)
    _assert_semantics(embedder)


def test_embeddinggemma2_caps_long_input_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("sentence_transformers")
    embedder = SentenceTransformerEmbedder("google/embeddinggemma-2")
    assert embedder.max_seq_length == 8192
    with caplog.at_level("WARNING", logger="tubitak_rag.adapters.st_embedder"):
        (vector,) = embedder.embed_documents(["destek oranı " * 6000])
    assert len(vector) == 768
    assert "1 metin 8192 token sınırını aşıyor" in caplog.text
