"""Tests for the Ollama embedder using ``httpx.MockTransport`` (no running Ollama)."""

import json
import math

import httpx
import pytest

from tubitak_rag.adapters.ollama_embedder import EmbeddingError, OllamaEmbedder
from tubitak_rag.ports import Embedder


def _client(requests: list[dict[str, object]], *, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append({"url": str(request.url), **body})
        vectors = [[3.0, 4.0] for _ in body["input"]]
        return httpx.Response(status, json={"embeddings": vectors})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_is_an_embedder() -> None:
    assert isinstance(OllamaEmbedder("bge-m3", dimension=2, client=_client([])), Embedder)


def test_posts_batch_and_returns_normalized_vectors() -> None:
    requests: list[dict[str, object]] = []
    embedder = OllamaEmbedder(
        "bge-m3", url="http://ollama:11434/", dimension=2, client=_client(requests)
    )
    vectors = embedder.embed_documents(["a", "b"])
    assert requests == [
        {"url": "http://ollama:11434/api/embed", "model": "bge-m3", "input": ["a", "b"]}
    ]
    assert vectors == [pytest.approx([0.6, 0.8])] * 2
    assert math.isclose(sum(x * x for x in vectors[0]), 1.0)


def test_splits_large_inputs_into_batches() -> None:
    requests: list[dict[str, object]] = []
    embedder = OllamaEmbedder("bge-m3", dimension=2, batch_size=2, client=_client(requests))
    assert len(embedder.embed_documents(["a", "b", "c"])) == 3
    assert [r["input"] for r in requests] == [["a", "b"], ["c"]]


def test_empty_input_sends_nothing() -> None:
    requests: list[dict[str, object]] = []
    assert OllamaEmbedder("bge-m3", dimension=2, client=_client(requests)).embed_documents([]) == []
    assert requests == []


def test_embeddinggemma_gets_retrieval_prompts_as_prefixes() -> None:
    requests: list[dict[str, object]] = []
    embedder = OllamaEmbedder("embeddinggemma-2:270m", dimension=2, client=_client(requests))
    embedder.embed_documents(["destek oranı"])
    embedder.embed_query("hibe oranı nedir?")
    assert requests[0]["input"] == ["title: none | text: destek oranı"]
    assert requests[1]["input"] == ["task: search result | query: hibe oranı nedir?"]


def test_other_models_get_no_prefix() -> None:
    requests: list[dict[str, object]] = []
    OllamaEmbedder("bge-m3", dimension=2, client=_client(requests)).embed_query("soru")
    assert requests[0]["input"] == ["soru"]


def test_http_error_raises_embedding_error() -> None:
    embedder = OllamaEmbedder("bge-m3", dimension=2, client=_client([], status=500))
    with pytest.raises(EmbeddingError, match="Ollama embedding isteği başarısız"):
        embedder.embed_documents(["a"])


def test_connection_error_raises_embedding_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("bağlantı reddedildi", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="Ollama"):
        OllamaEmbedder("bge-m3", dimension=2, client=client).embed_query("a")


def test_dimension_mismatch_raises_embedding_error() -> None:
    embedder = OllamaEmbedder("bge-m3", dimension=1024, client=_client([]))
    with pytest.raises(EmbeddingError, match="boyut"):
        embedder.embed_documents(["a"])


def test_embedding_error_is_runtime_error() -> None:
    assert issubclass(EmbeddingError, RuntimeError)
