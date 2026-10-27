"""Tests for the sentence-transformers embedder with a fake ``sentence_transformers`` module."""

import sys
import types
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from tubitak_rag.adapters.st_embedder import SentenceTransformerEmbedder
from tubitak_rag.ports import Embedder


def _word_tokenizer(texts: list[str], add_special_tokens: bool = True) -> dict[str, Any]:
    """One token per whitespace-separated word, plus one special token."""
    return {"input_ids": [[0] * (len(text.split()) + add_special_tokens) for text in texts]}


class _FakeModel:
    """Records constructor and ``encode`` arguments; returns one fixed row per input."""

    instances: list["_FakeModel"] = []
    native_max_seq_length: int | None = 10**30  # EmbeddingGemma 2 reports no real limit
    native_dimension: int | None = 3

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.encode_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.max_seq_length = _FakeModel.native_max_seq_length
        self.prompts = {"Retrieval-query": "soru: ", "Retrieval-document": "belge: "}
        self.tokenizer = _word_tokenizer
        self.truncate_dim: int | None = kwargs.get("truncate_dim")
        _FakeModel.instances.append(self)

    def get_embedding_dimension(self) -> int | None:  # same rule as sentence-transformers
        native = _FakeModel.native_dimension
        if self.truncate_dim is None or native is None:
            return self.truncate_dim or native
        return min(native, self.truncate_dim)

    def encode(self, inputs: Sequence[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls.append((list(inputs), kwargs))
        return np.array([self.row(text) for text in inputs], dtype=np.float32)

    @staticmethod
    def row(text: str) -> list[float]:
        # Not unit length on purpose: half-precision models return slightly off norms.
        return [3.0, 4.0, 0.0]


@pytest.fixture(autouse=True)
def fake_st(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(_FakeModel, "native_max_seq_length", 10**30)
    monkeypatch.setattr(_FakeModel, "native_dimension", 3)
    _FakeModel.instances = []


def _model() -> _FakeModel:
    return _FakeModel.instances[-1]


def test_is_an_embedder() -> None:
    assert isinstance(SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu"), Embedder)


def test_gemma_uses_retrieval_prompts_and_text_only_kwargs() -> None:
    embedder = SentenceTransformerEmbedder("google/embeddinggemma-2", device="cpu")
    embedder.embed_documents(["destek oranı"])
    embedder.embed_query("hibe oranı nedir?")
    model = _model()
    assert model.kwargs["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert model.kwargs["device"] == "cpu"
    (_, doc_kwargs), (query_inputs, query_kwargs) = model.encode_calls
    assert doc_kwargs["prompt_name"] == "Retrieval-document"
    assert query_kwargs["prompt_name"] == "Retrieval-query"
    assert query_inputs == ["hibe oranı nedir?"]


def test_gemma_without_text_only_loads_full_config() -> None:
    SentenceTransformerEmbedder("google/embeddinggemma-2", device="cpu", text_only=False)
    assert "config_kwargs" not in _model().kwargs


def test_bge_m3_no_prompt() -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu")
    embedder.embed_documents(["a"])
    embedder.embed_query("b")
    model = _model()
    assert "config_kwargs" not in model.kwargs
    assert all(kwargs.get("prompt_name") is None for _, kwargs in model.encode_calls)


def test_normalizes() -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu")
    vectors = embedder.embed_documents(["a", "b"])
    _, kwargs = _model().encode_calls[0]
    assert kwargs["normalize_embeddings"] is True
    assert vectors == [pytest.approx([0.6, 0.8, 0.0])] * 2
    assert all(isinstance(x, float) for x in vectors[0])


def test_empty_input_does_not_call_model() -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu")
    assert embedder.embed_documents([]) == []
    assert _model().encode_calls == []


def test_dimension_comes_from_model_and_truncates_matryoshka(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu").dimension == 3
    monkeypatch.setattr(_FakeModel, "native_dimension", 768)
    embedder = SentenceTransformerEmbedder("google/embeddinggemma-2", device="cpu", dimension=256)
    assert _model().truncate_dim == 256
    assert "truncate_dim" not in _model().kwargs  # native size read before truncating
    assert embedder.dimension == 256
    assert embedder.model_name == "google/embeddinggemma-2"


def test_dimension_equal_to_native_is_not_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FakeModel, "native_dimension", 1024)
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu", dimension=1024)
    assert embedder.dimension == 1024 and _model().truncate_dim is None


def test_non_matryoshka_model_with_other_dimension_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FakeModel, "native_dimension", 1024)
    with pytest.raises(ValueError, match="boyut"):
        SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu", dimension=768)


def test_matryoshka_dimension_above_native_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FakeModel, "native_dimension", 768)
    with pytest.raises(ValueError, match="boyut"):
        SentenceTransformerEmbedder("google/embeddinggemma-2", device="cpu", dimension=1024)


def test_device_auto_without_torch_is_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    SentenceTransformerEmbedder("BAAI/bge-m3")
    assert _model().kwargs["device"] == "cpu"


def test_device_auto_with_cuda_is_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    SentenceTransformerEmbedder("BAAI/bge-m3", device="auto")
    assert _model().kwargs["device"] == "cuda"


def test_unknown_model_dimension_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FakeModel, "get_embedding_dimension", lambda self: None)
    with pytest.raises(ValueError, match="boyutu"):
        SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu")


def test_max_seq_length_is_capped_explicitly() -> None:
    embedder = SentenceTransformerEmbedder("google/embeddinggemma-2", device="cpu")
    assert _model().max_seq_length == embedder.max_seq_length == 8192


def test_smaller_model_limit_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FakeModel, "native_max_seq_length", 512)
    assert SentenceTransformerEmbedder("x/small", device="cpu").max_seq_length == 512


def test_max_seq_length_can_be_lowered() -> None:
    SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu", max_seq_length=1024)
    assert _model().max_seq_length == 1024


def test_logs_texts_that_will_be_truncated(caplog: pytest.LogCaptureFixture) -> None:
    embedder = SentenceTransformerEmbedder(
        "google/embeddinggemma-2", device="cpu", max_seq_length=4
    )
    with caplog.at_level("WARNING", logger="tubitak_rag.adapters.st_embedder"):
        embedder.embed_documents(["bir iki", "bir iki üç dört", "bir"])
    # "belge: " prompt + words + 1 special token: 4, 6, 3 tokens -> one text over the limit
    (record,) = caplog.records
    assert "1 metin" in record.getMessage() and "en uzun 6" in record.getMessage()


def test_no_log_when_everything_fits(caplog: pytest.LogCaptureFixture) -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu")
    with caplog.at_level("WARNING"):
        embedder.embed_documents(["kısa metin"])
        embedder.embed_query("kısa soru")
    assert caplog.records == []


def test_batches_hold_at_most_batch_size_texts() -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu", batch_size=4)
    embedder.embed_documents([f"metin {i}" for i in range(10)])
    calls = _model().encode_calls
    assert [len(inputs) for inputs, _ in calls] == [4, 4, 2]
    assert [kwargs["batch_size"] for _, kwargs in calls] == [4, 4, 2]


def test_long_texts_get_smaller_batches_and_order_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Padded batch cost = texts x longest text; 2-word texts cost 3 tokens, 8-word ones 9.
    monkeypatch.setattr(_FakeModel, "row", staticmethod(lambda t: [float(len(t)), 1.0, 0.0]))
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu", max_batch_tokens=10)
    long_text = "a b c d e f g h"
    texts = ["x y", long_text, "z w", long_text, "q r"]
    vectors = embedder.embed_documents(texts)
    assert [inputs for inputs, _ in _model().encode_calls] == [
        ["x y", "z w", "q r"],
        [long_text],
        [long_text],
    ]
    expected = [_FakeModel.row(t) for t in texts]
    assert vectors == [pytest.approx(np.array(e) / np.linalg.norm(e)) for e in expected]
