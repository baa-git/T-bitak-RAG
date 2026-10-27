"""Tests for the registry-based component factory (NFR-CLEAN3, NFR-MOD2, FR-G4)."""

import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tests.builders import make_chunk
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.lexical_index import FakeLexicalIndex
from tests.fakes.llm import FakeLLM
from tests.fakes.reranker import FakeReranker
from tests.fakes.tokens import RegexTokenCounter
from tests.fakes.vector_store import InMemoryVectorStore
from tubitak_rag.adapters.anthropic_llm import AnthropicLLM
from tubitak_rag.adapters.bm25s_index import Bm25sIndex
from tubitak_rag.adapters.hf_tokenizer import HfTokenCounter
from tubitak_rag.adapters.numpy_store import NumpyVectorStore
from tubitak_rag.adapters.ollama_embedder import OllamaEmbedder
from tubitak_rag.adapters.ollama_llm import OllamaLLM
from tubitak_rag.adapters.qdrant_store import QdrantVectorStore
from tubitak_rag.adapters.st_embedder import SentenceTransformerEmbedder
from tubitak_rag.adapters.st_reranker import CrossEncoderReranker
from tubitak_rag.chunking.cache import chunker_key
from tubitak_rag.chunking.fixed import FixedSizeChunker
from tubitak_rag.chunking.structural import StructuralChunker
from tubitak_rag.config import (
    AppConfig,
    Bm25Config,
    ChunkerConfig,
    ConfigError,
    EmbedderConfig,
    FixedChunkConfig,
    GuardrailConfig,
    LlmConfig,
    PathsConfig,
    RerankerConfig,
    RetrieverConfig,
    VectorStoreConfig,
)
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document
from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.factory import (
    CHUNKERS,
    EMBEDDERS,
    LLMS,
    TOKEN_COUNTERS,
    VECTOR_STORES,
    SharedComponents,
    UnknownComponentError,
    build_chunker,
    build_embedder,
    build_lexical_index,
    build_llm,
    build_pipeline,
    build_reranker,
    build_retriever,
    build_token_counter,
    build_vector_store,
    regex_token_counter,
)
from tubitak_rag.indexing.naming import bm25_dir, collection_name
from tubitak_rag.retrieval.bm25 import Bm25Retriever
from tubitak_rag.retrieval.dense import DenseRetriever
from tubitak_rag.retrieval.hybrid import HybridRetriever

pytestmark = pytest.mark.usefixtures("fake_components")


def _fake_cfg() -> AppConfig:
    return AppConfig(
        embedder=EmbedderConfig(provider="fake"),
        vector_store=VectorStoreConfig(provider="fake"),
        reranker=RerankerConfig(provider="fake"),
        llm=LlmConfig(provider="fake"),
        bm25=Bm25Config(provider="fake"),
    )


def test_build_embedder_unknown_provider_lists_registered() -> None:
    # provider is a Literal; model_copy skips validation to simulate an unregistered key.
    cfg = AppConfig().model_copy(
        update={"embedder": EmbedderConfig().model_copy(update={"provider": "yok"})}
    )
    with pytest.raises(UnknownComponentError) as e:
        build_embedder(cfg)
    message = str(e.value)
    assert message.startswith("Bilinmeyen embedder sağlayıcısı: 'yok'")
    assert "'fake'" in message and "'sentence_transformers'" in message


def test_unknown_component_error_is_a_key_error() -> None:
    assert issubclass(UnknownComponentError, KeyError)


def test_fake_providers_are_built_from_config() -> None:
    cfg = _fake_cfg()
    assert isinstance(build_embedder(cfg), FakeEmbedder)
    assert isinstance(build_vector_store(cfg), InMemoryVectorStore)
    assert isinstance(build_reranker(cfg), FakeReranker)
    assert isinstance(build_llm(cfg.llm), FakeLLM)
    assert isinstance(build_lexical_index(cfg), FakeLexicalIndex)


def test_registering_new_component_is_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(EMBEDDERS, "dummy", lambda cfg: FakeEmbedder(model_name="dummy"))
    # provider is a Literal; model_copy skips validation so the test needs no schema change.
    cfg = AppConfig().model_copy(
        update={"embedder": EmbedderConfig().model_copy(update={"provider": "dummy"})}
    )
    assert build_embedder(cfg).model_name == "dummy"


def test_build_reranker_returns_none_when_disabled() -> None:
    cfg = _fake_cfg().model_copy(update={"reranker": RerankerConfig(enabled=False)})
    assert build_reranker(cfg) is None


def test_provider_switch_by_config_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # FR-G4. The Anthropic SDK is faked: the real API is never called.
    module = types.ModuleType("anthropic")
    module.Anthropic = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)

    ollama = build_llm(LlmConfig(model="qwen3:0.6b", url="http://ollama:11434"))
    assert isinstance(ollama, OllamaLLM)
    assert ollama.model_name == "qwen3:0.6b"

    cfg = LlmConfig.model_validate(
        {"provider": "anthropic", "anthropic": {"model": "claude-sonnet-5", "effort": "medium"}}
    )
    claude = build_llm(cfg)
    assert isinstance(claude, AnthropicLLM)
    assert claude.model_name == "claude-sonnet-5"


def test_default_llm_is_local_and_small() -> None:
    # NFR-PRIV1 / NFR-HW3: no API provider and a < 8B model by default.
    cfg = AppConfig()
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "qwen3:4b"
    assert cfg.eval.judge.provider == "ollama"


def test_build_chunker_uses_chunker_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Noop:
        name = "structural"

        def chunk(self, doc: Document) -> list[Chunk]:
            return []

    monkeypatch.setitem(CHUNKERS, "structural", lambda cfg: _Noop())
    assert build_chunker(AppConfig()).name == "structural"
    # name is a Literal; model_copy skips validation to simulate an unregistered key.
    cfg = AppConfig().model_copy(
        update={"chunker": ChunkerConfig().model_copy(update={"name": "yok"})}
    )
    with pytest.raises(UnknownComponentError):
        build_chunker(cfg)


def test_build_token_counter_parses_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def _hf(model: str) -> RegexTokenCounter:
        seen.append(model)
        return RegexTokenCounter()

    monkeypatch.setitem(TOKEN_COUNTERS, "hf", _hf)
    assert isinstance(build_token_counter("regex"), RegexTokenCounter)
    build_token_counter("hf:google/embeddinggemma-2")
    assert seen == ["google/embeddinggemma-2"]


def test_build_token_counter_unknown_kind() -> None:
    with pytest.raises(UnknownComponentError) as e:
        build_token_counter("sentencepiece:x")
    assert "token sayacı" in str(e.value)


def test_ollama_embedder_is_built_from_config() -> None:
    cfg = AppConfig(
        embedder=EmbedderConfig(
            provider="ollama", model="embeddinggemma-2:270m", url="http://o:1", batch_size=4
        )
    )
    embedder = build_embedder(cfg)
    assert isinstance(embedder, OllamaEmbedder)
    assert (embedder.model_name, embedder.dimension) == ("embeddinggemma-2:270m", 768)


def test_sentence_transformers_embedder_is_built_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class _Model:
        max_seq_length = 8192

        def __init__(self, name: str, **kwargs: Any) -> None:
            seen.update(name=name, **kwargs)

        def get_embedding_dimension(self) -> int:
            return 1024

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    cfg = AppConfig(embedder=EmbedderConfig(model="BAAI/bge-m3", dimension=1024, device="cpu"))
    embedder = build_embedder(cfg)
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert seen == {"name": "BAAI/bge-m3", "device": "cpu"}  # native 1024: no truncation
    assert embedder.dimension == 1024


def test_chunkers_are_built_from_config() -> None:
    fixed_cfg = ChunkerConfig(
        name="fixed",
        tokenizer="regex",
        context_header=False,
        fixed=FixedChunkConfig(size=100, overlap=10),
    )
    fixed = build_chunker(AppConfig(chunker=fixed_cfg))
    assert isinstance(fixed, FixedSizeChunker) and isinstance(fixed.counter, RegexTokenCounter)
    assert (fixed.size, fixed.overlap, fixed.context_header) == (100, 10, False)
    structural = build_chunker(AppConfig(chunker=ChunkerConfig(tokenizer="regex", max_tokens=300)))
    assert isinstance(structural, StructuralChunker)
    assert (structural.max_tokens, structural.context_header) == (300, True)


def test_hf_token_counter_is_registered() -> None:
    assert TOKEN_COUNTERS["hf"] is HfTokenCounter


def test_qdrant_vector_store_is_built_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    module = types.ModuleType("qdrant_client")
    module.QdrantClient = _Client  # type: ignore[attr-defined]
    module.models = types.ModuleType("qdrant_client.models")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    store = build_vector_store(AppConfig(vector_store=VectorStoreConfig(url="http://q:6333")))
    assert isinstance(store, QdrantVectorStore)
    assert seen["url"] == "http://q:6333"


def test_numpy_vector_store_is_built_from_config(tmp_path: Path) -> None:
    cfg = AppConfig(vector_store=VectorStoreConfig(provider="numpy", export_dir=tmp_path))
    store = build_vector_store(cfg)
    assert isinstance(store, NumpyVectorStore) and store.export_dir == tmp_path


def test_bm25s_lexical_index_is_built_from_config(tmp_path: Path) -> None:
    index = build_lexical_index(AppConfig(bm25=Bm25Config(stemming=False, k1=1.2, b=0.5)))
    assert isinstance(index, Bm25sIndex)
    index.build([])
    index.save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert (meta["k1"], meta["b"], meta["stemming"]) == (1.2, 0.5, False)


def test_build_lexical_index_unknown_provider() -> None:
    cfg = AppConfig().model_copy(
        update={"bm25": Bm25Config().model_copy(update={"provider": "elastic"})}
    )
    with pytest.raises(UnknownComponentError, match="sözcüksel indeks sağlayıcısı"):
        build_lexical_index(cfg)


def _retrieval_parts(cfg: AppConfig) -> tuple[FakeEmbedder, InMemoryVectorStore, FakeLexicalIndex]:
    chunk = make_chunk("a", text="destek oranı")
    embedder, store, lexical = FakeEmbedder(), InMemoryVectorStore(), FakeLexicalIndex()
    collection = collection_name(chunker_key(cfg.chunker), cfg.embedder.model)
    store.ensure_collection(collection, embedder.dimension)
    store.upsert(collection, [chunk], embedder.embed_documents([chunk.embed_text]))
    lexical.build([chunk])
    return embedder, store, lexical


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("dense", DenseRetriever), ("bm25", Bm25Retriever), ("hybrid", HybridRetriever)],
)
def test_build_retriever_by_mode_uses_config_collection(mode: str, expected: type) -> None:
    cfg = AppConfig(retriever=RetrieverConfig(mode=mode))  # type: ignore[arg-type]
    retriever = build_retriever(cfg, *_retrieval_parts(cfg))
    assert isinstance(retriever, expected)
    assert [r.chunk.chunk_id for r in retriever.retrieve("destek oranı", k=1)] == ["a"]


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_build_retriever_rejects_bm25_out_of_sync_with_collection(mode: str) -> None:
    cfg = AppConfig(retriever=RetrieverConfig(mode=mode))  # type: ignore[arg-type]
    embedder, store, lexical = _retrieval_parts(cfg)
    lexical.build([make_chunk("a", text="destek oranı"), make_chunk("b", text="hibe")])
    with pytest.raises(ConfigError, match="rag ingest"):
        build_retriever(cfg, embedder, store, lexical)


def test_build_retriever_passes_rrf_k() -> None:
    cfg = AppConfig(retriever=RetrieverConfig(mode="hybrid", rrf_k=1))
    (result,) = build_retriever(cfg, *_retrieval_parts(cfg)).retrieve("destek oranı", k=1)
    assert result.scores["rrf"] == 1 / 2 + 1 / 2


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_build_retriever_lexical_modes_need_bm25(mode: str) -> None:
    cfg = AppConfig(retriever=RetrieverConfig(mode=mode), bm25=Bm25Config(enabled=False))  # type: ignore[arg-type]
    embedder, store, lexical = _retrieval_parts(cfg)
    with pytest.raises(ConfigError, match="bm25"):
        build_retriever(cfg, embedder, store, lexical)
    enabled = AppConfig(retriever=RetrieverConfig(mode=mode))  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="bm25"):
        build_retriever(enabled, embedder, store, None)


def test_regex_token_counter_warns_that_it_is_for_tests(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="tubitak_rag.factory"):
        counter = regex_token_counter("")
    assert isinstance(counter, RegexTokenCounter)
    assert "yalnız testler" in caplog.text


def test_cross_encoder_reranker_is_built_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _CrossEncoder:
        def __init__(self, name: str, **kwargs: Any) -> None:
            seen.update(kwargs, name=name)

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = _CrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    cfg = AppConfig(reranker=RerankerConfig(model="x/rr", device="cpu", max_length=1024))
    reranker = build_reranker(cfg)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "x/rr"
    assert (seen["name"], seen["device"], seen["max_length"]) == ("x/rr", "cpu", 1024)


# --- build_pipeline (Task 6.8) --------------------------------------------------------------

_PIPELINE_ANSWER = (
    '{"cevaplanabilir": true, "cevap": "Destek oranı en fazla %75 olarak belirlenmiştir [1].", '
    '"atiflar": [1], "eksik_kisim": null}'
)


def _pipeline_cfg(tmp_path: Path, mode: str, rerank: bool) -> AppConfig:
    return _fake_cfg().model_copy(
        update={
            "paths": PathsConfig(artifacts_dir=tmp_path / "artifacts"),
            "retriever": RetrieverConfig(mode=mode, top_n=3, k_candidates=7),  # type: ignore[arg-type]
            "reranker": RerankerConfig(provider="fake", enabled=rerank),
            # Fake embeddings are hash vectors: any cosine must pass the dense gate here.
            "guardrails": GuardrailConfig(
                scope_mode="lexicon",
                number_check="reject",
                thresholds={"dense": -1.0, "rerank": -2.0, "bm25": 0.0},
            ),
        }
    )


def _indexed_corpus(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, *, save_bm25: bool = True
) -> InMemoryVectorStore:
    """An indexed fake corpus at the collection/BM25 paths ingest would write for ``cfg``."""
    chunk = make_chunk("a", text="Destek oranı en fazla %75 olarak belirlenmiştir.")
    embedder, store, lexical = FakeEmbedder(), InMemoryVectorStore(), FakeLexicalIndex()
    key = chunker_key(cfg.chunker)
    collection = collection_name(key, cfg.embedder.model)
    store.ensure_collection(collection, embedder.dimension)
    store.upsert(collection, [chunk], embedder.embed_documents([chunk.embed_text]))
    if save_bm25:
        lexical.build([chunk])
        lexical.save(bm25_dir(cfg.paths.artifacts_dir, key))
    monkeypatch.setitem(VECTOR_STORES, "fake", lambda _cfg: store)
    monkeypatch.setitem(LLMS, "fake", lambda _llm_cfg: FakeLLM([_PIPELINE_ANSWER]))
    return store


@pytest.mark.parametrize(
    ("mode", "rerank", "retriever_type", "gate_key"),
    [
        ("dense", False, DenseRetriever, "dense"),
        ("dense", True, DenseRetriever, "rerank"),
        ("hybrid", False, HybridRetriever, "dense"),
        ("hybrid", True, HybridRetriever, "rerank"),
        ("bm25", False, Bm25Retriever, "bm25"),
    ],
)
def test_build_pipeline_follows_retriever_and_reranker_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    rerank: bool,
    retriever_type: type,
    gate_key: str,
) -> None:
    cfg = _pipeline_cfg(tmp_path, mode, rerank)
    _indexed_corpus(cfg, monkeypatch)

    pipeline = build_pipeline(cfg)

    assert isinstance(pipeline.retriever, retriever_type)
    assert isinstance(pipeline.reranker, FakeReranker) is rerank
    answer = pipeline.ask("1507 destek oranı en fazla yüzde kaç?")
    assert answer.durum is AnswerStatus.CEVAPLANDI
    assert answer.gate_key == gate_key
    assert [c.chunk_id for c in answer.atiflar] == ["a"]


def test_build_pipeline_wires_guardrail_and_retrieval_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _pipeline_cfg(tmp_path, "dense", False)
    store = _indexed_corpus(cfg, monkeypatch)

    pipeline = build_pipeline(cfg)

    assert (pipeline.k_candidates, pipeline.top_n) == (7, 3)
    assert pipeline.number_check == "reject"
    assert pipeline.scope.mode == "lexicon"
    assert pipeline.disclaimer == cfg.guardrails.disclaimer
    assert pipeline.gate.thresholds == cfg.guardrails.thresholds
    # Same collection name as ``rag ingest`` writes; chunk lookup for the web source panel.
    assert pipeline.store is store
    assert pipeline.collection == collection_name(chunker_key(cfg.chunker), cfg.embedder.model)
    assert [c.chunk_id for c in pipeline.get_chunks(["a"])] == ["a"]


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_build_pipeline_without_bm25_index_asks_for_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    cfg = _pipeline_cfg(tmp_path, mode, False)
    _indexed_corpus(cfg, monkeypatch, save_bm25=False)

    with pytest.raises(ConfigError, match="rag ingest"):
        build_pipeline(cfg)


def test_build_pipeline_dense_mode_needs_no_bm25_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _pipeline_cfg(tmp_path, "dense", True)
    _indexed_corpus(cfg, monkeypatch, save_bm25=False)

    assert isinstance(build_pipeline(cfg).retriever, DenseRetriever)


# --- shared components (Gradio Space: every config loaded at once) --------------------------


def test_shared_components_build_each_key_once() -> None:
    shared = SharedComponents()
    built: list[str] = []

    def make(name: str) -> object:
        built.append(name)
        return object()

    first = shared.get("embedder", EmbedderConfig(model="a"), lambda: make("a"))
    again = shared.get("embedder", EmbedderConfig(model="a"), lambda: make("a"))
    other = shared.get("embedder", EmbedderConfig(model="b"), lambda: make("b"))
    assert first is again and other is not first
    assert built == ["a", "b"] and shared.kinds() == ["embedder", "embedder"]


def test_build_pipeline_with_shared_components_reuses_models_across_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dense = _pipeline_cfg(tmp_path, "dense", True)
    hybrid = _pipeline_cfg(tmp_path, "hybrid", True)
    no_rerank = _pipeline_cfg(tmp_path, "hybrid", False)
    _indexed_corpus(dense, monkeypatch)
    calls: list[str] = []
    monkeypatch.setitem(EMBEDDERS, "fake", lambda _cfg: calls.append("embedder") or FakeEmbedder())
    monkeypatch.setitem(LLMS, "fake", lambda _cfg: calls.append("llm") or FakeLLM([]))
    shared = SharedComponents()

    first = build_pipeline(dense, shared)
    second = build_pipeline(hybrid, shared)
    third = build_pipeline(no_rerank, shared)

    assert calls == ["embedder", "llm"]  # one embedder and one LLM for all three
    assert first.generator.llm is second.generator.llm is third.generator.llm
    assert first.reranker is second.reranker and third.reranker is None
    assert first.store is second.store
    assert sorted(set(shared.kinds())) == ["bm25", "embedder", "llm", "reranker", "vector_store"]


def test_transformers_llm_is_registered() -> None:
    assert "transformers" in LLMS
