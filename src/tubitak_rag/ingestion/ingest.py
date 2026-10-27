"""Ingest orchestration: manifest -> parse -> chunk -> embed + index -> BM25 -> report.

Every stage sits behind a content-addressed cache, so a second run with an unchanged corpus
does no parsing, no chunking and no embedding (FR-X9):

* parse:  ``artifacts/cache/parsed``      key = doc id + raw file SHA-256 + parser version
* chunk:  ``artifacts/cache/chunks``      key = parsed document digest + chunker settings
* embed:  ``artifacts/cache/embeddings``  key = model + dimension + SHA-256 of the text

The vector index is synchronised incrementally: document fingerprints of the current chunks
are compared with the fingerprints stored in the collection, and only added or changed
documents are (re-)embedded and written; documents gone from the corpus are deleted
(FR-I2..FR-I4). Documents are written one by one and embeddings are cached per batch, so an
interrupted run resumes where it stopped. The BM25 index is rebuilt from all chunks on every
run because its statistics depend on the whole corpus (FR-I6).
"""

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from tubitak_rag.chunking.cache import ChunkCache, chunk_documents, chunk_settings, chunker_key
from tubitak_rag.config import AppConfig
from tubitak_rag.domain.base import DomainModel
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document
from tubitak_rag.indexing.bm25_index import rebuild_bm25
from tubitak_rag.indexing.cached_embedder import CachedEmbedder
from tubitak_rag.indexing.embedding_cache import EmbeddingCache
from tubitak_rag.indexing.fingerprint import fingerprints
from tubitak_rag.indexing.naming import bm25_dir, collection_name
from tubitak_rag.indexing.sync import apply_sync, plan_sync
from tubitak_rag.ingestion.manifest import load_sources
from tubitak_rag.ingestion.parsed_cache import ParsedCache, ParseOutcome, parse_entry
from tubitak_rag.ports import Chunker, Embedder, LexicalIndex, VectorStore

# progress(stage, done, total); stages: "parse", "chunk", "embed", "bm25"
Progress = Callable[[str, int, int], None]
REPORT_FILE = "ingest_report.json"


class IngestReport(DomainModel):
    """Outcome of one ingest run (``artifacts/ingest_report.json``)."""

    collection: str
    chunker: str
    embedder: str
    full: bool
    docs_total: int  # records in the manifests
    added: list[str]
    updated: list[str]
    removed: list[str]
    unchanged: int
    scanned: list[str]  # no text layer: reported, never indexed (FR-P9)
    empty: list[str]  # parsed but produced no chunks
    missing: list[str]  # raw file not found
    errors: list[dict[str, str]]  # {"doc_id", "message"}
    chunks_total: int  # chunks of the corpus = points in the collection after the run
    chunks_indexed: int  # chunks written in this run (added + updated documents)
    parsed: int
    parse_cached: int
    chunked: int
    chunk_cached: int
    embedded: int  # texts computed by the model in this run
    embed_cached: int  # texts served from the embedding cache
    bm25_size: int
    stage_times_s: dict[str, float]  # parse, chunk, embed_index, bm25, total
    message: str = ""  # set when the run deliberately changed nothing (no raw files)


class LazyEmbedder:
    """Builds the real embedder on first use.

    Loading a model takes seconds and occupies the GPU; a run whose vectors all come from
    the cache (or that has nothing to embed) never loads it.
    """

    def __init__(self, build: Callable[[], Embedder], *, model_name: str, dimension: int) -> None:
        self._build = build
        self._inner: Embedder | None = None
        self.model_name = model_name
        self.dimension = dimension

    def _get(self) -> Embedder:
        if self._inner is None:
            inner = self._build()
            if inner.dimension != self.dimension:
                raise ValueError(
                    f"{self.model_name} modelinin boyutu {inner.dimension}, yapılandırmadaki "
                    f"boyut {self.dimension}; embedder.dimension değerini düzeltin."
                )
            self._inner = inner
        return self._inner

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents with the (lazily built) inner embedder."""
        return self._get().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the (lazily built) inner embedder."""
        return self._get().embed_query(text)


class _CountingEmbedder:
    """Counts the texts that actually reach the model (cache misses)."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.model_name = inner.model_name
        self.dimension = inner.dimension
        self.texts = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts += len(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


class _CountingChunker:
    """Counts the documents that are actually chunked (chunk cache misses)."""

    def __init__(self, inner: Chunker) -> None:
        self._inner = inner
        self.name = inner.name
        self.calls = 0

    def chunk(self, doc: Document) -> list[Chunk]:
        self.calls += 1
        return self._inner.chunk(doc)


class _StageTimer:
    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[stage] = round(time.perf_counter() - start, 3)


def _no_progress(stage: str, done: int, total: int) -> None:
    """Default progress callback: report nothing."""


def run_ingest(
    cfg: AppConfig,
    *,
    embedder: Embedder,
    store: VectorStore,
    chunker: Chunker,
    lexical: LexicalIndex,
    full: bool = False,
    progress: Progress | None = None,
) -> IngestReport:
    """Bring the collection of ``cfg`` (chunker + embedding model) in line with the corpus.

    Args:
        embedder: the raw embedder; it is wrapped in the embedding cache when
            ``cfg.embedder.cache`` is set. Batches hold at most ``cfg.embedder.batch_size``
            texts.
        full: drop the collection first and write every document again. The caches stay
            valid (they are content-addressed), so a full rebuild needs no recomputation.
        progress: called as ``progress(stage, done, total)`` during the run.
    """
    notify = progress or _no_progress
    timer = _StageTimer()
    started = time.perf_counter()
    key = chunker_key(cfg.chunker)
    collection = collection_name(key, cfg.embedder.model)
    artifacts = cfg.paths.artifacts_dir

    entries = load_sources(cfg.paths.manifests_dir)
    with timer("parse"):
        parse_cache = ParsedCache(artifacts / "cache" / "parsed")
        documents: list[Document] = []
        outcomes: list[ParseOutcome] = []
        for done, entry in enumerate(entries, start=1):
            doc, outcome = parse_entry(entry, cfg.paths.raw_dir, parse_cache)
            outcomes.append(outcome)
            if doc is not None:
                documents.append(doc)
            notify("parse", done, len(entries))

    if outcomes and all(o.status == "missing" for o in outcomes):
        existing = store.count(collection)
        if existing:  # no raw file at all (e.g. a bundle-restored stack): keep the index
            return _untouched_report(cfg, collection, key, full, outcomes, existing, timer)

    indexable = [doc for doc in documents if not doc.scanned]
    with timer("chunk"):
        counting_chunker = _CountingChunker(chunker)
        chunk_cache = ChunkCache(artifacts / "cache" / "chunks", key, chunk_settings(cfg.chunker))
        chunks_by_doc: dict[str, list[Chunk]] = {}
        for done, doc in enumerate(indexable, start=1):
            chunks_by_doc.update(chunk_documents([doc], counting_chunker, chunk_cache))
            notify("chunk", done, len(indexable))
    empty = sorted(doc_id for doc_id, chunks in chunks_by_doc.items() if not chunks)
    chunks_by_doc = {doc_id: chunks for doc_id, chunks in chunks_by_doc.items() if chunks}

    with timer("embed_index"):
        if full:
            store.drop_collection(collection)
        plan = plan_sync(fingerprints(chunks_by_doc), store.indexed_docs(collection))
        counting_embedder = _CountingEmbedder(embedder)
        active: Embedder = counting_embedder
        if cfg.embedder.cache:
            cache = EmbeddingCache(artifacts / "cache" / "embeddings" / "cache.sqlite")
            active = CachedEmbedder(counting_embedder, cache)
        to_index = sum(len(chunks_by_doc[d]) for d in [*plan.add, *plan.update])
        notify("embed", 0, to_index)
        apply_sync(
            plan,
            chunks_by_doc,
            active,
            store,
            collection,
            batch_size=cfg.embedder.batch_size,
            progress=notify,
        )

    all_chunks = [chunk for doc_id in sorted(chunks_by_doc) for chunk in chunks_by_doc[doc_id]]
    with timer("bm25"):
        bm25_size = 0
        if cfg.bm25.enabled:
            bm25_size = rebuild_bm25(all_chunks, lexical, bm25_dir(artifacts, key))
            notify("bm25", bm25_size, bm25_size)
    timer.times["total"] = round(time.perf_counter() - started, 3)

    by_status = _ids_by_status(outcomes)
    report = IngestReport(
        collection=collection,
        chunker=key,
        embedder=cfg.embedder.model,
        full=full,
        docs_total=len(entries),
        added=plan.add,
        updated=plan.update,
        removed=plan.remove,
        unchanged=len(plan.unchanged),
        scanned=sorted(doc.doc_id for doc in documents if doc.scanned),
        empty=empty,
        missing=by_status.get("missing", []),
        errors=[
            {"doc_id": o.doc_id, "message": o.message} for o in outcomes if o.status == "error"
        ],
        chunks_total=len(all_chunks),
        chunks_indexed=to_index,
        parsed=len(by_status.get("parsed", [])) + len(by_status.get("scanned", [])),
        parse_cached=len(by_status.get("cached", [])),
        chunked=counting_chunker.calls,
        chunk_cached=len(indexable) - counting_chunker.calls,
        embedded=counting_embedder.texts,
        embed_cached=to_index - counting_embedder.texts,
        bm25_size=bm25_size,
        stage_times_s=timer.times,
    )
    write_report(report, artifacts / REPORT_FILE)
    return report


def _untouched_report(
    cfg: AppConfig,
    collection: str,
    key: str,
    full: bool,
    outcomes: Sequence[ParseOutcome],
    existing: int,
    timer: _StageTimer,
) -> IngestReport:
    """Report of a run that found none of the raw files and therefore changed nothing.

    Without this guard every indexed document would look removed from the corpus and the
    sync would empty the collection and the BM25 index. Not written to ``REPORT_FILE``: the
    last real report stays valid.
    """
    timer.times["total"] = sum(timer.times.values())
    return IngestReport(
        collection=collection,
        chunker=key,
        embedder=cfg.embedder.model,
        full=full,
        docs_total=len(outcomes),
        added=[],
        updated=[],
        removed=[],
        unchanged=0,
        scanned=[],
        empty=[],
        missing=sorted(o.doc_id for o in outcomes),
        errors=[],
        chunks_total=existing,
        chunks_indexed=0,
        parsed=0,
        parse_cached=0,
        chunked=0,
        chunk_cached=0,
        embedded=0,
        embed_cached=0,
        bm25_size=0,
        stage_times_s=timer.times,
        message=(
            f"Hiçbir kaynak dosya bulunamadı ({cfg.paths.raw_dir}); indeks değiştirilmedi. "
            "Belgeleri indirmek için önce `rag fetch` çalıştırın."
        ),
    )


def _ids_by_status(outcomes: Sequence[ParseOutcome]) -> Mapping[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.status, []).append(outcome.doc_id)
    return {status: sorted(ids) for status, ids in grouped.items()}


def write_report(report: IngestReport, path: Path) -> None:
    """Write the report as indented JSON (UTF-8, LF line endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(report.model_dump_json(indent=2).encode("utf-8"))
