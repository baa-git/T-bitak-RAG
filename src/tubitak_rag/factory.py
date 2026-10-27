"""Component factory: the only place where concrete implementations are chosen (NFR-CLEAN3).

Each registry maps a config key to a constructor. Adding a new implementation is one file
plus one registration line here (NFR-MOD2), for example::

    RERANKERS["my"] = lambda cfg: MyReranker(cfg.reranker.model)

Components never build their own dependencies; they receive them from these functions.
"""

import logging
import os
from collections.abc import Callable, Mapping
from typing import Any, cast

from pydantic import BaseModel

from tubitak_rag.adapters.anthropic_llm import AnthropicLLM
from tubitak_rag.adapters.bm25s_index import Bm25sIndex
from tubitak_rag.adapters.hf_tokenizer import HfTokenCounter
from tubitak_rag.adapters.numpy_store import NumpyVectorStore
from tubitak_rag.adapters.ollama_embedder import OllamaEmbedder
from tubitak_rag.adapters.ollama_llm import OllamaLLM
from tubitak_rag.adapters.openai_compat_llm import OpenAICompatLLM
from tubitak_rag.adapters.qdrant_store import QdrantVectorStore
from tubitak_rag.adapters.st_embedder import SentenceTransformerEmbedder, cuda_available
from tubitak_rag.adapters.st_reranker import CrossEncoderReranker
from tubitak_rag.adapters.transformers_llm import TransformersLLM
from tubitak_rag.chunking.cache import chunker_key
from tubitak_rag.chunking.fixed import FixedSizeChunker
from tubitak_rag.chunking.structural import StructuralChunker
from tubitak_rag.chunking.tokens import RegexTokenCounter
from tubitak_rag.config import AppConfig, ConfigError, LlmConfig
from tubitak_rag.device_policy import apply_device_policy, choose_device, device_setting
from tubitak_rag.generation.generator import AnswerGenerator
from tubitak_rag.guardrails.gate import RetrievalGate
from tubitak_rag.guardrails.scope import ScopeChecker
from tubitak_rag.indexing.bm25_index import load_bm25
from tubitak_rag.indexing.naming import bm25_dir, collection_name
from tubitak_rag.pipeline import RagPipeline
from tubitak_rag.ports import (
    LLM,
    Chunker,
    Embedder,
    LexicalIndex,
    Reranker,
    Retriever,
    TokenCounter,
    VectorStore,
)
from tubitak_rag.retrieval.bm25 import Bm25Retriever, check_index_matches_collection
from tubitak_rag.retrieval.dense import DenseRetriever
from tubitak_rag.retrieval.hybrid import HybridRetriever
from tubitak_rag.retrieval.turkish import TurkishTokenizer

logger = logging.getLogger(__name__)

EMBEDDERS: dict[str, Callable[[AppConfig], Embedder]] = {}
VECTOR_STORES: dict[str, Callable[[AppConfig], VectorStore]] = {}
CHUNKERS: dict[str, Callable[[AppConfig], Chunker]] = {}
RERANKERS: dict[str, Callable[[AppConfig], Reranker]] = {}
LEXICAL_INDEXES: dict[str, Callable[[AppConfig], LexicalIndex]] = {}
LLMS: dict[str, Callable[[LlmConfig], LLM]] = {}
# Key is the part before ":" in specs such as "hf:google/embeddinggemma-2" or "regex".
TOKEN_COUNTERS: dict[str, Callable[[str], TokenCounter]] = {}


class UnknownComponentError(KeyError):
    """Raised when a config names a component that has no registration (message in Turkish)."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _lookup[T](registry: Mapping[str, T], key: str, label: str) -> T:
    try:
        return registry[key]
    except KeyError:
        raise UnknownComponentError(
            f"Bilinmeyen {label}: {key!r}. Kayıtlı: {sorted(registry)}"
        ) from None


class SharedComponents:
    """Builds each model, store and index once and lends it to every pipeline that needs it.

    The experiment configs differ from each other in one method, so most of them use the
    same embedding model, reranker and LLM. A component is keyed by the config section that
    determines it (after the device policy): with one ``SharedComponents`` the six configs
    v0 … v3b load EmbeddingGemma 2, bge-m3, the reranker and the LLM once each. The Gradio
    Space keeps every config loaded at the same time this way (``build_pipeline(cfg, shared)``).
    """

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], Any] = {}

    def get[T](self, kind: str, key: BaseModel | str, build: Callable[[], T]) -> T:
        """The component of ``kind`` built for ``key``; ``build()`` runs only the first time."""
        ident = (kind, key if isinstance(key, str) else key.model_dump_json())
        if ident not in self._items:
            self._items[ident] = build()
        return cast(T, self._items[ident])

    def kinds(self) -> list[str]:
        """The kind of every component built so far, in build order (for logs and tests)."""
        return [kind for kind, _ in self._items]


def runtime_device(configured: str) -> str:
    """Concrete ``cpu``/``cuda`` for a device setting, after ``TUBITAK_RAG_DEVICE``."""
    return choose_device(device_setting(configured, os.environ), cuda_available)


def runtime_config(cfg: AppConfig) -> AppConfig:
    """``cfg`` with concrete devices and the CPU reranker policy applied (device_policy.py)."""
    return apply_device_policy(cfg, cuda_available=cuda_available, environ=os.environ)


def build_embedder(cfg: AppConfig) -> Embedder:
    """Build the embedder selected by ``cfg.embedder.provider``."""
    return _lookup(EMBEDDERS, cfg.embedder.provider, "embedder sağlayıcısı")(cfg)


def build_vector_store(cfg: AppConfig) -> VectorStore:
    """Build the vector store selected by ``cfg.vector_store.provider``."""
    return _lookup(VECTOR_STORES, cfg.vector_store.provider, "vektör deposu sağlayıcısı")(cfg)


def build_chunker(cfg: AppConfig) -> Chunker:
    """Build the chunker selected by ``cfg.chunker.name``."""
    return _lookup(CHUNKERS, cfg.chunker.name, "chunker")(cfg)


def build_reranker(cfg: AppConfig) -> Reranker | None:
    """Build the reranker, or return ``None`` when reranking is disabled."""
    if not cfg.reranker.enabled:
        return None
    return _lookup(RERANKERS, cfg.reranker.provider, "reranker sağlayıcısı")(cfg)


def build_lexical_index(cfg: AppConfig) -> LexicalIndex:
    """Build the (empty) lexical index selected by ``cfg.bm25.provider``."""
    return _lookup(LEXICAL_INDEXES, cfg.bm25.provider, "sözcüksel indeks sağlayıcısı")(cfg)


def build_retriever(
    cfg: AppConfig,
    embedder: Embedder,
    store: VectorStore,
    lexical: LexicalIndex | None,
) -> Retriever:
    """Build the retriever selected by ``cfg.retriever.mode`` over the configured collection.

    ``bm25`` and ``hybrid`` need an enabled, loaded lexical index holding exactly the chunks
    of the collection (``ConfigError`` otherwise).
    """
    collection = collection_name(chunker_key(cfg.chunker), cfg.embedder.model)
    dense = DenseRetriever(embedder, store, collection)
    mode = cfg.retriever.mode
    if mode == "dense":
        return dense
    if not cfg.bm25.enabled or lexical is None:
        raise ConfigError(f"retriever.mode={mode!r} için bm25 etkin olmalı ve indeksi yüklenmeli.")
    check_index_matches_collection(lexical, store, collection)
    bm25 = Bm25Retriever(lexical, store, collection)
    if mode == "bm25":
        return bm25
    return HybridRetriever(dense, bm25, rrf_k=cfg.retriever.rrf_k)


def build_llm(llm_cfg: LlmConfig) -> LLM:
    """Build the LLM selected by ``llm_cfg.provider`` (switching needs only config, FR-G4)."""
    return _lookup(LLMS, llm_cfg.provider, "LLM sağlayıcısı")(llm_cfg)


def build_pipeline(cfg: AppConfig, shared: SharedComponents | None = None) -> RagPipeline:
    """Wire the full question-answering pipeline of ``cfg``.

    The collection and the BM25 directory are the ones ``rag ingest`` writes for the same
    chunker and embedding model. ``bm25``/``hybrid`` modes need the saved BM25 index
    (``ConfigError`` asking for ``rag ingest`` when it is missing). With ``shared`` the
    models, the vector store and the BM25 index already built for another config are reused.
    """
    cfg = runtime_config(cfg)  # on CPU the reranker may see fewer candidates (NFR-PERF1)
    parts = shared if shared is not None else SharedComponents()
    embedder = parts.get("embedder", cfg.embedder, lambda: build_embedder(cfg))
    store = parts.get("vector_store", cfg.vector_store, lambda: build_vector_store(cfg))
    lexical: LexicalIndex | None = None
    if cfg.retriever.mode != "dense" and cfg.bm25.enabled:
        directory = bm25_dir(cfg.paths.artifacts_dir, chunker_key(cfg.chunker))
        if not directory.is_dir():
            raise ConfigError(
                f"BM25 indeksi bulunamadı: {directory}. Önce `rag ingest` çalıştırın."
            )
        lexical = parts.get(
            "bm25",
            f"{directory.as_posix()} {cfg.bm25.model_dump_json()}",
            lambda: load_bm25(build_lexical_index(cfg), directory),
        )
    llm = parts.get("llm", cfg.llm, lambda: build_llm(cfg.llm))
    reranker = parts.get("reranker", cfg.reranker, lambda: build_reranker(cfg))
    return RagPipeline(
        retriever=build_retriever(cfg, embedder, store, lexical),
        reranker=reranker,
        generator=AnswerGenerator(llm, temperature=cfg.llm.temperature),
        scope=ScopeChecker(cfg.guardrails.scope_mode, llm),
        gate=RetrievalGate(cfg.guardrails.thresholds),
        k_candidates=cfg.retriever.k_candidates,
        top_n=cfg.retriever.top_n,
        disclaimer=cfg.guardrails.disclaimer,
        number_check=cfg.guardrails.number_check,
        store=store,
        collection=collection_name(chunker_key(cfg.chunker), cfg.embedder.model),
    )


def build_token_counter(spec: str) -> TokenCounter:
    """Build a token counter from a spec like ``"regex"`` or ``"hf:<model id>"``."""
    kind, _, model = spec.partition(":")
    return _lookup(TOKEN_COUNTERS, kind, "token sayacı")(model)


def regex_token_counter(_model: str) -> TokenCounter:
    """The approximate regex counter; it is meant for tests, so using it is flagged."""
    logger.warning(
        "tokenizer: regex yalnız testler içindir; gerçek chunk boyutları için "
        "hf:<model> kullanın (ör. hf:google/embeddinggemma-2)"
    )
    return RegexTokenCounter()


# --- registrations ------------------------------------------------------------------------

EMBEDDERS["sentence_transformers"] = lambda cfg: SentenceTransformerEmbedder(
    cfg.embedder.model,
    device=runtime_device(cfg.embedder.device),
    batch_size=cfg.embedder.batch_size,
    text_only=cfg.embedder.text_only,
    dimension=cfg.embedder.dimension,
)
EMBEDDERS["ollama"] = lambda cfg: OllamaEmbedder(
    cfg.embedder.model,
    dimension=cfg.embedder.dimension,
    url=cfg.embedder.url,
    batch_size=cfg.embedder.batch_size,
)
RERANKERS["cross_encoder"] = lambda cfg: CrossEncoderReranker(
    cfg.reranker.model,
    device=runtime_device(cfg.reranker.device),
    max_length=cfg.reranker.max_length,
)
VECTOR_STORES["qdrant"] = lambda cfg: QdrantVectorStore(cfg.vector_store.url)
VECTOR_STORES["numpy"] = lambda cfg: NumpyVectorStore(cfg.vector_store.export_dir)
LEXICAL_INDEXES["bm25s"] = lambda cfg: Bm25sIndex(
    TurkishTokenizer(cfg.bm25.stemming), k1=cfg.bm25.k1, b=cfg.bm25.b
)

TOKEN_COUNTERS["regex"] = regex_token_counter
TOKEN_COUNTERS["hf"] = HfTokenCounter
CHUNKERS["fixed"] = lambda cfg: FixedSizeChunker(
    build_token_counter(cfg.chunker.tokenizer),
    size=cfg.chunker.fixed.size,
    overlap=cfg.chunker.fixed.overlap,
    context_header=cfg.chunker.context_header,
)
CHUNKERS["structural"] = lambda cfg: StructuralChunker(
    build_token_counter(cfg.chunker.tokenizer),
    max_tokens=cfg.chunker.max_tokens,
    context_header=cfg.chunker.context_header,
)
LLMS["ollama"] = lambda llm_cfg: OllamaLLM(
    llm_cfg.url,
    llm_cfg.model,
    num_ctx=llm_cfg.num_ctx,
    think=llm_cfg.think,
    timeout_s=llm_cfg.timeout_s,
)
LLMS["transformers"] = lambda llm_cfg: TransformersLLM(
    llm_cfg.model,
    device=runtime_device(llm_cfg.transformers.device),
    dtype=llm_cfg.transformers.dtype,
    max_new_tokens=llm_cfg.transformers.max_new_tokens,
    think=llm_cfg.think,
)
LLMS["anthropic"] = lambda llm_cfg: AnthropicLLM(
    llm_cfg.anthropic.model, effort=llm_cfg.anthropic.effort
)
LLMS["openai_compat"] = lambda llm_cfg: OpenAICompatLLM(
    llm_cfg.openai_compat.base_url,
    llm_cfg.model,
    api_key_env=llm_cfg.openai_compat.api_key_env,
    json_mode=llm_cfg.openai_compat.json_mode,
    max_tokens=llm_cfg.openai_compat.max_tokens,
    timeout_s=llm_cfg.timeout_s,
)
