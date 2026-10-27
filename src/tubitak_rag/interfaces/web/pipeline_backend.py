"""``WebBackend`` over the real system: pipelines, vector store, manifests, run artifacts.

Nothing heavy happens when the backend is created. A pipeline (embedding model, reranker,
BM25 index, LLM client) is built at the first question that needs it and kept for the next
ones. At most ``max_pipelines`` stay loaded (least recently used first out), so switching
between experiment configs on the query page keeps GPU memory bounded. Each question holds
the GPU lock while it is answered; ingest and experiment runs are CLI subprocesses started
through the ``JobRegistry`` and are refused (``GpuLockError``) while the GPU is busy.
"""

import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
import yaml

from tubitak_rag import __version__
from tubitak_rag.adapters.gpu_memory import release_gpu_memory
from tubitak_rag.adapters.numpy_store import read_meta
from tubitak_rag.chunking.cache import chunker_key
from tubitak_rag.chunking.ids import HIERARCHY_SEPARATOR
from tubitak_rag.config import AppConfig, ConfigError, load_config
from tubitak_rag.domain.answer import Answer
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document
from tubitak_rag.domain.evaluation import RunMetrics
from tubitak_rag.domain.retrieval import MetadataFilter
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.evaluation.runner import RunPaths
from tubitak_rag.factory import build_pipeline, build_vector_store
from tubitak_rag.indexing.naming import bm25_dir, collection_name
from tubitak_rag.ingestion.hashing import sha256_of
from tubitak_rag.ingestion.ingest import REPORT_FILE
from tubitak_rag.ingestion.manifest import load_sources
from tubitak_rag.ingestion.parsed_cache import ParsedCache
from tubitak_rag.interfaces.gpu_lock import (
    GPU_LOCK_FILE,
    GpuLockError,
    busy_message,
    gpu_lock,
    lock_holder,
    uses_gpu,
)
from tubitak_rag.interfaces.index_bundle import qdrant_points
from tubitak_rag.interfaces.web.about import (
    component_lines,
    corpus_groups,
    describe_methods,
    guardrail_lines,
)
from tubitak_rag.interfaces.web.backend import ServiceUnavailableError, UnknownPipelineError
from tubitak_rag.interfaces.web.comparison import summarize
from tubitak_rag.interfaces.web.jobs import JobRegistry, cli_command, run_command
from tubitak_rag.interfaces.web.schemas import (
    AboutInfo,
    AskResult,
    DocumentPreview,
    DocumentsResponse,
    DocumentState,
    DocumentStatus,
    HealthStatus,
    JobStarted,
    PipelineOption,
    RunDetail,
    RunSummary,
    SectionView,
    SourceContext,
)
from tubitak_rag.ports import VectorStore
from tubitak_rag.reporting.report import load_run
from tubitak_rag.reporting.summary import SUMMARY_FILE

logger = logging.getLogger(__name__)

RUNS_DIR = "runs"
JOB_LOGS_DIR = Path("logs") / "web"
DEFAULT_PIPELINE = "v3"
MAX_PIPELINES = 2
INDEX_TTL_S = 15.0  # how long a "collection / BM25 present?" answer is reused
NO_INDEX_NOTE = "indeks yok"
_ARTICLE_RE = re.compile(r"^(?:Geçici |Ek )?Madde\b")
_RUN_ID_RE = re.compile(r"^[\w.-]+$")
_RUN_STAMP_RE = re.compile(r"_(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})(?:-\d+)?$")


class AnswerPipeline(Protocol):
    """What the backend needs from a pipeline (``RagPipeline`` satisfies it)."""

    def ask(self, question: str, filters: MetadataFilter | None = None) -> Answer: ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]: ...


@runtime_checkable
class DocChunkCounter(Protocol):
    """Optional vector store capability: chunks per document (Qdrant adapter has it)."""

    def doc_chunk_counts(
        self, name: str, doc_ids: Sequence[str] | None = None
    ) -> dict[str, int]: ...


CommandRunner = Callable[[Sequence[str], Path], None]
IndexStatus = Callable[[AppConfig], str | None]  # None = usable, else what is missing


# --- pure helpers --------------------------------------------------------------------------


def split_hierarchy(hiyerarsi: str) -> tuple[str | None, str | None]:
    """Split a chunk hierarchy into (enclosing sections, article heading).

    ``"Üçüncü Bölüm › Madde 7 – Destek oranı › (2)"`` -> ``("Üçüncü Bölüm", "Madde 7 – …")``;
    a hierarchy without an article stays whole.
    """
    parts = [part.strip() for part in hiyerarsi.split(HIERARCHY_SEPARATOR.strip())]
    parts = [part for part in parts if part]
    for index, part in enumerate(parts):
        if _ARTICLE_RE.match(part):
            return HIERARCHY_SEPARATOR.join(parts[:index]) or None, part
    return HIERARCHY_SEPARATOR.join(parts) or None, None


def section_for(doc: Document, chunk: Chunk) -> str | None:
    """Parsed section a chunk came from: same article number, else the page that holds it."""
    if chunk.madde_no:
        for section in doc.sections:
            if section.madde_no == chunk.madde_no:
                return section.section_id
    if chunk.sayfa is not None:
        for section in doc.sections:
            start = section.page_start
            end = section.page_end if section.page_end is not None else start
            if start is not None and end is not None and start <= chunk.sayfa <= end:
                return section.section_id
    return None


def _collection(cfg: AppConfig) -> str:
    return collection_name(chunker_key(cfg.chunker), cfg.embedder.model)


def index_problem(cfg: AppConfig, points: int | None) -> str | None:
    """What ``cfg`` lacks to answer: an empty collection or a missing BM25 directory.

    ``points`` is the number of points in its collection, ``None`` when unknown (Qdrant
    unreachable, or another vector store): then only the BM25 directory is checked.
    """
    if points == 0:
        return f"koleksiyon boş ya da yok: {_collection(cfg)}"
    if cfg.retriever.mode != "dense" and cfg.bm25.enabled:
        directory = bm25_dir(cfg.paths.artifacts_dir, chunker_key(cfg.chunker))
        if not directory.is_dir():
            return f"BM25 indeksi yok: {directory}"
    return None


def qdrant_index_status(cfg: AppConfig) -> str | None:
    """``index_problem`` with the point count read from Qdrant's REST API (2 s timeout)."""
    points: int | None = None
    if cfg.vector_store.provider == "qdrant":
        try:
            with httpx.Client(timeout=2.0) as client:
                points = qdrant_points(client, cfg.vector_store.url.rstrip("/"), _collection(cfg))
        except httpx.HTTPError:
            points = None  # unknown: do not hide configs because Qdrant is briefly away
    return index_problem(cfg, points)


def store_index_status(cfg: AppConfig) -> str | None:
    """``index_problem`` for the configured vector store: Qdrant's REST API or the NumPy export.

    The NumPy store (Hugging Face Space) is read from ``meta.json`` without loading vectors.
    """
    if cfg.vector_store.provider == "numpy":
        meta = read_meta(cfg.vector_store.export_dir, _collection(cfg))
        return index_problem(cfg, int(meta.get("points", 0)) if meta else 0)
    return qdrant_index_status(cfg)


def _run_started(run_id: str) -> str:
    match = _RUN_STAMP_RE.search(run_id)
    if match is None:
        return ""
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}"


# --- backend -------------------------------------------------------------------------------


class PipelineBackend:
    """The real system behind the pages and the REST API."""

    is_demo = False

    def __init__(
        self,
        cfg: AppConfig,
        *,
        config_path: Path,
        experiments_dir: Path | None = None,
        pipeline_factory: Callable[[AppConfig], AnswerPipeline] = build_pipeline,
        store_factory: Callable[[AppConfig], VectorStore] = build_vector_store,
        command_runner: CommandRunner = run_command,
        release_memory: Callable[[], None] = release_gpu_memory,
        jobs: JobRegistry | None = None,
        probes: Mapping[str, Callable[[], bool]] | None = None,
        max_pipelines: int = MAX_PIPELINES,
        index_status: IndexStatus | None = None,
        index_ttl_s: float = INDEX_TTL_S,
        config_overrides: Mapping[str, Any] | None = None,
        default_pipeline: str | None = None,
        use_gpu_lock: bool = True,
    ) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.experiments_dir = experiments_dir or config_path.parent / "experiments"
        self._pipeline_factory = pipeline_factory
        self._store_factory = store_factory
        self._run_command = command_runner
        self._release_memory = release_memory
        self._jobs = jobs or JobRegistry()
        self._probes = dict(probes) if probes is not None else self._default_probes()
        self._max_pipelines = max_pipelines
        self._lock = threading.RLock()  # one question / job start at a time
        self._pipelines: OrderedDict[str, AnswerPipeline] = OrderedDict()
        self._options: dict[str, tuple[Path, AppConfig]] | None = None
        self._store: VectorStore | None = None
        self._parsed: dict[tuple[str, int, int], Document | None] = {}
        self._index_status = index_status or store_index_status
        self._index_ttl_s = index_ttl_s
        self._index_cache: dict[str, tuple[float, str | None]] = {}
        # Merged on top of every experiment config (e.g. the Hugging Face Space overlay).
        self._overrides = dict(config_overrides) if config_overrides else None
        self._default_pipeline = default_pipeline
        self._use_gpu_lock = use_gpu_lock

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        config_overrides: Mapping[str, Any] | None = None,
        **kwargs: object,
    ) -> "PipelineBackend":
        """Load ``config_path`` (``ConfigError`` if invalid) and build a lazy backend.

        ``config_overrides`` are merged on top of the server config and of every experiment
        config (``load_config`` overrides, interpolated like the files).
        """
        cfg = load_config(config_path, config_overrides)
        return cls(
            cfg,
            config_path=config_path,
            config_overrides=config_overrides,
            **kwargs,  # type: ignore[arg-type]
        )

    # --- selectable pipelines ----------------------------------------------------------

    def _configs(self) -> dict[str, tuple[Path, AppConfig]]:
        if self._options is None:
            options: dict[str, tuple[Path, AppConfig]] = {}
            for path in sorted(self.experiments_dir.glob("*.yaml")):
                cfg = load_config(path, self._overrides)
                options[cfg.experiment.name] = (path, cfg)
            if not options:
                options[self.cfg.experiment.name] = (self.config_path, self.cfg)
            self._options = options
        return self._options

    def _missing_index(self, name: str) -> str | None:
        """What the named config lacks (cached ``index_ttl_s``); a failing check means usable."""
        now = time.monotonic()
        cached = self._index_cache.get(name)
        if cached is not None and now - cached[0] < self._index_ttl_s:
            return cached[1]
        try:
            problem = self._index_status(self._configs()[name][1])
        except Exception:
            logger.info("web: %s indeks durumu okunamadı", name, exc_info=True)
            problem = None
        self._index_cache[name] = (now, problem)
        return problem

    def _default_name(self) -> str:
        names = list(self._configs())
        wanted = (self._default_pipeline, self.cfg.experiment.name, DEFAULT_PIPELINE)
        preferred = [n for n in wanted if n in names]
        for candidate in [*preferred, *names]:
            if self._missing_index(candidate) is None:
                return candidate
        return preferred[0] if preferred else names[0]

    def _resolve(self, name: str | None) -> str:
        if name is None:
            return self._default_name()
        stem = Path(name).stem if name.endswith((".yaml", ".yml")) else name
        if stem not in self._configs():
            known = ", ".join(self._configs())
            raise UnknownPipelineError(f"Bilinmeyen yapılandırma: {name} (seçenekler: {known})")
        return stem

    def pipelines(self) -> list[PipelineOption]:
        """Experiment configs of ``configs/experiments``: names, methods and index state.

        A config whose collection is empty or whose BM25 index is missing stays in the list
        but is marked ``hazir=False`` ("indeks yok"), e.g. in a Docker stack restored from a
        bundle that carries only some collections.
        """
        default = self._default_name()
        return [
            PipelineOption(
                ad=name,
                etiket=name,
                aciklama=cfg.experiment.description,
                varsayilan=name == default,
                yontemler=describe_methods(cfg),
                hazir=(ready := self._missing_index(name) is None),
                durum_notu=None if ready else NO_INDEX_NOTE,
            )
            for name, (_, cfg) in self._configs().items()
        ]

    def loaded_pipelines(self) -> list[str]:
        """Names of the pipelines currently in memory, least recently used first."""
        return list(self._pipelines)

    def _pipeline(self, name: str) -> AnswerPipeline:
        if name in self._pipelines:
            self._pipelines.move_to_end(name)
            return self._pipelines[name]
        while len(self._pipelines) >= self._max_pipelines:
            self._evict()
        pipeline = self._pipeline_factory(self._configs()[name][1])
        self._pipelines[name] = pipeline
        return pipeline

    def preload(self) -> list[str]:
        """Build now, not at the first question, every pipeline whose index is present.

        The Gradio Space (ZeroGPU) loads every model when the app starts: its answer function
        runs in a forked GPU worker, so whatever that worker builds is lost after the call.
        Fetching no chunks opens each collection (the NumPy store reads it into memory). At
        most ``max_pipelines`` are built; returns the names now in memory.
        """
        names = [name for name in self._configs() if self._missing_index(name) is None]
        if len(names) > self._max_pipelines:
            logger.warning(
                "web: %d yapılandırmadan yalnız %d tanesi önceden yükleniyor (max_pipelines)",
                len(names),
                self._max_pipelines,
            )
        with self._lock:
            for name in names[: self._max_pipelines]:
                self._pipeline(name).get_chunks([])
                logger.info("web: %s pipeline'ı yüklendi", name)
        return self.loaded_pipelines()

    def _evict(self) -> None:
        name, pipeline = self._pipelines.popitem(last=False)
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()
        del pipeline
        logger.info("web: %s pipeline'ı bellekten çıkarıldı", name)
        self._release_memory()

    def _release_all(self) -> None:
        while self._pipelines:
            self._evict()

    @contextmanager
    def _gpu(self, cfg: AppConfig, owner: str, *, answering: bool) -> Iterator[None]:
        if not self._use_gpu_lock or not uses_gpu(cfg, answering=answering):
            yield
            return
        with gpu_lock(self._lock_path, owner):
            yield

    @property
    def _lock_path(self) -> Path:
        return self.cfg.paths.artifacts_dir / GPU_LOCK_FILE

    # --- questions ---------------------------------------------------------------------

    def disclaimer(self) -> str:
        """``guardrails.disclaimer`` of the server config."""
        return self.cfg.guardrails.disclaimer

    def ask(
        self,
        soru: str,
        program_kodu: str | None,
        doc_turu: str | None,
        yapilandirma: str | None = None,
    ) -> AskResult:
        """Answer with the named pipeline, built on first use; holds the GPU lock meanwhile."""
        name = self._resolve(yapilandirma)
        cfg = self._configs()[name][1]
        problem = self._missing_index(name)
        if problem is not None:
            raise ServiceUnavailableError(
                f"{name} yapılandırmasının {NO_INDEX_NOTE} ({problem}). Başka bir yapılandırma "
                f"seçin ya da bu yapılandırmayla `rag ingest -c` çalıştırın."
            )
        filters = None
        if program_kodu or doc_turu:
            filters = MetadataFilter(program_kodu=program_kodu, doc_turu=doc_turu)
        owner = f"web ask ({name}) pid={os.getpid()}"
        with self._lock:
            try:
                with self._gpu(cfg, owner, answering=True):
                    pipeline = self._pipeline(name)
                    answer = pipeline.ask(soru, filters)
                    context = self._context(pipeline, answer)
            except GpuLockError:
                self._release_all()  # another job wants the GPU: give our memory back
                raise
            except ConfigError as exc:
                raise ServiceUnavailableError(str(exc)) from exc
        return AskResult(answer=answer, baglam=context, yapilandirma=name)

    def _context(self, pipeline: AnswerPipeline, answer: Answer) -> dict[str, SourceContext]:
        if not answer.kaynaklar:
            return {}
        try:
            chunks = pipeline.get_chunks([source.chunk_id for source in answer.kaynaklar])
        except Exception:  # the panel still shows konum/text without the extra context
            logger.warning("web: kaynak bağlamı okunamadı", exc_info=True)
            return {}
        context = {}
        for chunk in chunks:
            hiyerarsi, madde_baslik = split_hierarchy(chunk.hiyerarsi)
            doc = self._parsed_document(chunk.doc_id)
            context[chunk.chunk_id] = SourceContext(
                hiyerarsi=hiyerarsi,
                madde_baslik=madde_baslik,
                sayfa=chunk.sayfa,
                surum_tarihi=chunk.surum_tarihi,
                program_kodu=chunk.program_kodu,
                doc_id=chunk.doc_id,
                bolum_id=section_for(doc, chunk) if doc is not None else None,
            )
        return context

    # --- documents ---------------------------------------------------------------------

    def _entries(self) -> dict[str, SourceEntry]:
        return {e.doc_id: e for e in load_sources(self.cfg.paths.manifests_dir)}

    def _vector_store(self) -> VectorStore:
        if self._store is None:
            self._store = self._store_factory(self.cfg)
        return self._store

    def _chunk_counts(
        self, collection: str, doc_ids: Sequence[str] | None = None
    ) -> dict[str, int]:
        """Chunks per *indexed* document: a key is present exactly when the doc is indexed."""
        store = self._vector_store()
        try:
            if isinstance(store, DocChunkCounter):
                counts = store.doc_chunk_counts(collection, doc_ids)
                return {doc_id: n for doc_id, n in counts.items() if n > 0}
            # Without the capability only the presence of a document is known (count 0).
            return {doc_id: 0 for doc_id in store.indexed_docs(collection)}
        except Exception as exc:
            raise ServiceUnavailableError(
                f"Vektör deposuna ulaşılamadı ({self.cfg.vector_store.url}): {exc}"
            ) from exc

    def _scanned(self) -> set[str]:
        """Documents the last ingest found without a text layer (``ingest_report.json``)."""
        try:
            data = json.loads((self.cfg.paths.artifacts_dir / REPORT_FILE).read_text("utf-8"))
            return set(data.get("scanned", []))
        except (OSError, ValueError, AttributeError):
            return set()

    def _status(self, entry: SourceEntry, chunks: int | None, scanned: set[str]) -> DocumentStatus:
        """State of one manifest record; ``chunks`` is None when it is not in the index."""
        state: DocumentState
        if chunks is not None:  # indexed, even without the raw file (bundle restore)
            state = "indeksli"
        elif not (self.cfg.paths.raw_dir / entry.local_path).is_file():
            state = "eksik"
        elif entry.doc_id in scanned:
            state = "taranmis"
        else:
            state = "guncel_degil"  # downloaded but not (yet) in the index
        return DocumentStatus(
            doc_id=entry.doc_id,
            baslik=entry.title,
            program_kodu=entry.program_kodu,
            doc_turu=entry.doc_turu.value,
            kaynak_url=entry.url,
            surum_tarihi=entry.surum_tarihi.isoformat() if entry.surum_tarihi else None,
            dil=entry.dil,
            durum=state,
            chunk_sayisi=chunks or 0,
        )

    def documents(self) -> DocumentsResponse:
        """Manifest records with their state in the default collection."""
        collection = _collection(self.cfg)
        counts = self._chunk_counts(collection)
        scanned = self._scanned()
        rows = [
            self._status(entry, counts.get(entry.doc_id), scanned)
            for entry in self._entries().values()
        ]
        return DocumentsResponse(
            belgeler=rows, koleksiyon=collection, toplam_chunk=sum(counts.values())
        )

    def _parsed_document(self, doc_id: str) -> Document | None:
        """The parsed document from the ingest cache (None if never parsed or missing)."""
        entry = self._entries().get(doc_id)
        if entry is None:
            return None
        path = self.cfg.paths.raw_dir / entry.local_path
        cache = ParsedCache(self.cfg.paths.artifacts_dir / "cache" / "parsed")
        try:
            stat = path.stat()
        except OSError:
            # No raw file (a stack restored from a prebuilt bundle): newest parsed version.
            key = (doc_id, -1, -1)
            if key not in self._parsed:
                self._parsed[key] = cache.latest(doc_id)
            return self._parsed[key]
        key = (doc_id, stat.st_mtime_ns, stat.st_size)
        if key not in self._parsed:
            self._parsed[key] = cache.get(doc_id, sha256_of(path))
        return self._parsed[key]

    def document(self, doc_id: str) -> DocumentPreview | None:
        """Identity, chunk counts per experiment collection and parsed sections."""
        entry = self._entries().get(doc_id)
        if entry is None:
            return None
        collections = sorted({_collection(cfg) for _, cfg in self._configs().values()})
        default_collection = _collection(self.cfg)
        if default_collection not in collections:
            collections.insert(0, default_collection)
        counts = {name: self._chunk_counts(name, [doc_id]).get(doc_id) for name in collections}
        doc = self._parsed_document(doc_id)
        sections = [
            SectionView(
                bolum_id=section.section_id,
                baslik=section.title,
                hiyerarsi=section.hierarchy,
                madde_no=section.madde_no,
                sayfa=section.page_start,
                metin=section.text,
                mulga=section.repealed,
                degisiklik_notlari=section.amendment_notes,
            )
            for section in (doc.sections if doc is not None else [])
        ]
        return DocumentPreview(
            belge=self._status(entry, counts[default_collection], self._scanned()),
            kaynak_sayfa_url=entry.kaynak_sayfa_url,
            ayristirildi=doc is not None,
            bolumler=sections,
            chunk_sayilari={name: count or 0 for name, count in counts.items()},
        )

    # --- Hakkında ----------------------------------------------------------------------

    def _section_count(self, doc_ids: set[str]) -> int | None:
        """Parsed sections of the manifest documents, read from the parse cache files.

        A document parsed more than once (changed file) counts with its newest cache file.
        """
        directory = self.cfg.paths.artifacts_dir / "cache" / "parsed"
        newest: dict[str, tuple[int, int]] = {}  # doc_id -> (mtime, sections)
        for path in directory.glob("*.json") if directory.is_dir() else []:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))["document"]
                doc_id, sections = document["doc_id"], len(document["sections"])
                mtime = path.stat().st_mtime_ns
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if doc_id in doc_ids and mtime >= newest.get(doc_id, (-1, 0))[0]:
                newest[doc_id] = (mtime, sections)
        return sum(n for _, n in newest.values()) if newest else None

    def _chunk_total(self, collection: str) -> int | None:
        """Chunks in the default collection; the last ingest report if the store is down."""
        try:
            count = self._vector_store().count(collection)
            if count > 0:
                return count
        except Exception:
            logger.info("web: vektör deposu sayımı okunamadı", exc_info=True)
        try:
            data = json.loads((self.cfg.paths.artifacts_dir / REPORT_FILE).read_text("utf-8"))
            total = data.get("chunks_total")
            return int(total) if total else None
        except (OSError, ValueError, AttributeError, TypeError):
            return None

    def about(self) -> AboutInfo:
        """Corpus groups from the manifests, sections from the parse cache, chunks from the index.

        Builds no pipeline and loads no model; a missing count is ``None`` ("—" on the page).
        """
        entries = self._entries()
        collection = _collection(self.cfg)
        configs = [cfg for _, cfg in self._configs().values()]
        return AboutInfo(
            gruplar=corpus_groups(entries.values()),
            belge_sayisi=len(entries),
            bolum_sayisi=self._section_count(set(entries)),
            parca_sayisi=self._chunk_total(collection),
            koleksiyon=collection,
            bilesenler=component_lines(configs),
            guardrailler=guardrail_lines(self.cfg),
            yapilandirmalar=self.pipelines(),
        )

    # --- background jobs ---------------------------------------------------------------

    def _ensure_gpu_free(self, cfg: AppConfig, *, answering: bool) -> None:
        if self._jobs.busy():
            raise GpuLockError(
                "Bu sunucunun başlattığı bir iş sürüyor; aynı anda tek bir ingest ya da "
                "deney koşusu çalışabilir. İş bitince tekrar deneyin."
            )
        if uses_gpu(cfg, answering=answering):
            holder = lock_holder(self._lock_path)
            if holder is not None:
                raise GpuLockError(busy_message(self._lock_path, holder))

    def _start(self, kind: str, argv: list[str]) -> JobStarted:
        logs = self.cfg.paths.artifacts_dir / JOB_LOGS_DIR

        def work(job_id: str) -> None:
            self._run_command(argv, logs / f"{job_id}.log")

        self._release_all()  # the job's process needs the GPU memory
        return JobStarted(job_id=self._jobs.start(kind, work))

    def start_ingest(self, full: bool = False) -> JobStarted:
        """``rag ingest`` of the server config in a subprocess (409 while the GPU is busy)."""
        with self._lock:
            self._ensure_gpu_free(self.cfg, answering=False)
            argv = cli_command("ingest", "-c", str(self.config_path))
            return self._start("ingest", argv + (["--full"] if full else []))

    def start_run(self, config: str, limit: int | None = None) -> JobStarted:
        """``rag run <experiment>`` in a subprocess (409 while the GPU is busy)."""
        name = self._resolve(config)
        path, cfg = self._configs()[name]
        with self._lock:
            self._ensure_gpu_free(cfg, answering=True)
            argv = cli_command("run", str(path))
            return self._start("run", argv + (["--limit", str(limit)] if limit else []))

    # --- runs --------------------------------------------------------------------------

    @property
    def _runs_dir(self) -> Path:
        return self.cfg.paths.artifacts_dir / RUNS_DIR

    def _run_dir(self, run_id: str) -> Path | None:
        if not _RUN_ID_RE.match(run_id) or set(run_id) == {"."}:
            return None
        directory = self._runs_dir / run_id
        return directory if RunPaths(directory).metrics_json.is_file() else None

    def _running(self, directory: Path) -> RunSummary | None:
        paths = RunPaths(directory)
        if not paths.config_yaml.is_file():
            return None
        try:
            data = yaml.safe_load(paths.config_yaml.read_text(encoding="utf-8")) or {}
            experiment = str(data["experiment"]["name"])
        except (OSError, yaml.YAMLError, KeyError, TypeError):
            experiment = directory.name.split("_", 1)[0]
        return RunSummary(
            run_id=directory.name,
            deney=experiment,
            tarih=_run_started(directory.name),
            durum="suruyor",
            metrikler={},
        )

    def runs(self) -> list[RunSummary]:
        """Finished runs (``metrics.json``) and runs still writing results, newest first."""
        if not self._runs_dir.is_dir():
            return []
        rows = []
        for directory in self._runs_dir.iterdir():
            if not directory.is_dir():
                continue
            if RunPaths(directory).metrics_json.is_file():
                rows.append(summarize(load_run(directory)[0]))
            elif (running := self._running(directory)) is not None:
                rows.append(running)
        return sorted(rows, key=lambda row: (row.tarih, row.run_id), reverse=True)

    def run_metrics(self, run_id: str) -> RunMetrics | None:
        """Metrics recomputed from ``per_question.jsonl`` (as in the report)."""
        directory = self._run_dir(run_id)
        return None if directory is None else load_run(directory)[0]

    def run_detail(self, run_id: str) -> RunDetail | None:
        """Metrics plus every per-question row of a finished run."""
        directory = self._run_dir(run_id)
        if directory is None:
            return None
        metrics, questions = load_run(directory)
        return RunDetail(metrics=metrics, sorular=questions)

    def run_report(self, run_id: str) -> str | None:
        """``report.md`` of a finished run."""
        directory = self._run_dir(run_id)
        report = RunPaths(directory).report_md if directory is not None else None
        return report.read_text(encoding="utf-8") if report and report.is_file() else None

    def summary_report(self) -> str | None:
        """``runs/ozet.md``."""
        path = self._runs_dir / SUMMARY_FILE
        return path.read_text(encoding="utf-8") if path.is_file() else None

    # --- health ------------------------------------------------------------------------

    def _default_probes(self) -> dict[str, Callable[[], bool]]:
        def qdrant() -> bool:
            self._vector_store().count(_collection(self.cfg))
            return True

        def ollama() -> bool:
            url = self.cfg.llm.url.rstrip("/")
            return httpx.get(f"{url}/api/tags", timeout=2.0).is_success

        return {"qdrant": qdrant, "ollama": ollama}

    def _probe(self, name: str) -> bool:
        try:
            return bool(self._probes[name]())
        except Exception:
            return False

    def health(self) -> HealthStatus:
        """Reachability of Qdrant and Ollama; builds no pipeline, loads no model."""
        qdrant, ollama = self._probe("qdrant"), self._probe("ollama")
        return HealthStatus(
            durum="ok" if qdrant and ollama else "sorunlu",
            qdrant=qdrant,
            ollama=ollama,
            koleksiyon=_collection(self.cfg),
            surum=__version__,
        )
