"""The real web backend wired to fakes: lazy pipelines, LRU, documents, runs, jobs, health."""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.builders import make_chunk
from tests.eval_builders import make_answer
from tests.fakes.vector_store import InMemoryVectorStore
from tests.fixtures.corpus import CorpusOnDisk
from tubitak_rag.adapters.numpy_store import write_collection
from tubitak_rag.chunking.cache import chunker_key
from tubitak_rag.config import AppConfig, ConfigError, load_config
from tubitak_rag.domain.answer import Answer, SourceView
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document, Section
from tubitak_rag.domain.enums import AnswerStatus, DocType, SectionKind, SourceKind
from tubitak_rag.domain.retrieval import MetadataFilter
from tubitak_rag.indexing.naming import collection_name
from tubitak_rag.ingestion.hashing import sha256_of
from tubitak_rag.ingestion.parsed_cache import ParsedCache
from tubitak_rag.interfaces.gpu_lock import GPU_LOCK_FILE, GpuLockError
from tubitak_rag.interfaces.web.backend import (
    ServiceUnavailableError,
    UnknownPipelineError,
    WebBackend,
)
from tubitak_rag.interfaces.web.jobs import JobRegistry
from tubitak_rag.interfaces.web.pipeline_backend import (
    PipelineBackend,
    index_problem,
    section_for,
    split_hierarchy,
    store_index_status,
)

# --- fakes ---------------------------------------------------------------------------------


class _Pipeline:
    """Answers every question with ``answer``; records calls, chunk lookups and close()."""

    def __init__(self, cfg: AppConfig, answer: Answer, chunks: Sequence[Chunk]) -> None:
        self.cfg = cfg
        self.answer = answer
        self.chunks = {c.chunk_id: c for c in chunks}
        self.questions: list[tuple[str, MetadataFilter | None]] = []
        self.chunk_lookups: list[list[str]] = []
        self.closed = False
        self.lock_seen: bool | None = None

    def ask(
        self, question: str, filters: MetadataFilter | None = None, *, trace: Any = None
    ) -> Answer:
        self.questions.append((question, filters))
        lock = self.cfg.paths.artifacts_dir / GPU_LOCK_FILE
        self.lock_seen = lock.exists()
        return self.answer.model_copy(update={"soru": question})

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        self.chunk_lookups.append(list(chunk_ids))
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self, answer: Answer, chunks: Sequence[Chunk] = ()) -> None:
        self.answer = answer
        self.chunks = list(chunks)
        self.built: list[_Pipeline] = []
        self.error: Exception | None = None

    def __call__(self, cfg: AppConfig) -> _Pipeline:
        if self.error is not None:
            raise self.error
        pipeline = _Pipeline(cfg, self.answer, self.chunks)
        self.built.append(pipeline)
        return pipeline

    def names(self) -> list[str]:
        return [p.cfg.experiment.name for p in self.built]


class _Commands:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, argv: Sequence[str], log_path: Path) -> None:
        self.calls.append((list(argv), log_path))


# --- fixtures ------------------------------------------------------------------------------

SOURCE_CHUNK = make_chunk(
    "mevzuat-001:c7",
    doc_id="mevzuat-001",
    hiyerarsi="Üçüncü Bölüm – Destek › Madde 7 – Destek oranı › (2)",
    madde_no="7",
    sayfa=6,
    program_kodu="1507",
)


def _answered() -> Answer:
    source = SourceView(
        no=1,
        chunk_id=SOURCE_CHUNK.chunk_id,
        doc_title="Belge 001",
        konum="Madde 7 (2)",
        kaynak_url="https://tubitak.gov.tr/001_a.pdf",
        text=SOURCE_CHUNK.text,
        score=0.9,
    )
    return make_answer(kaynaklar=[source], atif_onarimi=[1])


class _Setup:
    def __init__(self, root: Path) -> None:
        self.corpus = CorpusOnDisk(root)
        self.corpus.add("001_a.pdf", "Madde 7 metni")
        self.corpus.add("002_b.pdf", "taranmış")
        self.corpus.add("003_c.pdf", "henüz indekslenmedi")
        self.corpus.register_only("004_d.pdf")  # missing download
        configs = root / "configs"
        (configs / "experiments").mkdir(parents=True)
        # Real providers in the config (so the GPU rules apply); the factories stay fakes.
        data = self.corpus.config_data(
            llm={"provider": "ollama"}, embedder={"provider": "sentence_transformers"}
        )
        data["experiment"] = {"name": "default", "description": "Varsayılan"}
        self.config_path = configs / "default.yaml"
        self.config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        for name, chunker, concept in [
            ("v1", "structural", "structural_context"),
            ("v2", "structural", "hybrid_rrf"),
            ("v3", "structural", "full_pipeline"),
            ("v0", "fixed", "fixed_chunking"),
        ]:
            experiment = {
                "extends": "../default.yaml",
                "experiment": {"name": name, "description": f"{name} açıklama", "concept": concept},
                "chunker": {"name": chunker},
            }
            (configs / "experiments" / f"{name}.yaml").write_text(
                yaml.safe_dump(experiment, allow_unicode=True), encoding="utf-8"
            )
        self.cfg = load_config(self.config_path)
        self.store = InMemoryVectorStore()
        self.factory = _Factory(_answered(), [SOURCE_CHUNK])
        self.commands = _Commands()
        self.released: list[str] = []
        self.jobs = JobRegistry()

    @property
    def artifacts(self) -> Path:
        return self.corpus.artifacts

    def collection(self, cfg: AppConfig | None = None) -> str:
        cfg = cfg or self.cfg
        return collection_name(chunker_key(cfg.chunker), cfg.embedder.model)

    def backend(self, **overrides: Any) -> PipelineBackend:
        kwargs: dict[str, Any] = {
            "config_path": self.config_path,
            "pipeline_factory": self.factory,
            "store_factory": lambda cfg: self.store,
            "command_runner": self.commands,
            "release_memory": lambda: self.released.append("release"),
            "jobs": self.jobs,
            "probes": {"qdrant": lambda: True, "ollama": lambda: False},
            "index_status": lambda cfg: None,  # every index present (see the index tests)
        }
        kwargs.update(overrides)
        return PipelineBackend(self.cfg, **kwargs)


@pytest.fixture
def setup(tmp_path: Path) -> _Setup:
    return _Setup(tmp_path)


def _parsed_document(setup: _Setup) -> Document:
    raw = setup.corpus.root / "raw" / "tubitak_mevzuat" / "001_a.pdf"

    def section(order: int, madde: str | None, **fields: Any) -> Section:
        defaults: dict[str, Any] = {
            "section_id": f"mevzuat-001:{order:04d}",
            "kind": SectionKind.MADDE,
            "title": None,
            "hierarchy": ["Üçüncü Bölüm – Destek"],
            "madde_no": madde,
            "page_start": order,
            "page_end": order,
            "text": f"Madde {madde} metni",
            "paragraphs": [],
            "amendment_notes": [],
            "repealed": False,
        }
        return Section(**{**defaults, **fields})

    doc = Document(
        doc_id="mevzuat-001",
        title="Belge 001",
        program_kodu=None,
        doc_turu=DocType.YONETMELIK,
        kind=SourceKind.PDF,
        kaynak_url="https://tubitak.gov.tr/001_a.pdf",
        surum_tarihi=None,
        dil="tr",
        sha256=sha256_of(raw),
        kaynak_grubu="tubitak_mevzuat",
        scanned=False,
        structured=True,
        page_count=7,
        sections=[
            section(1, "1", title="Amaç"),
            section(5, "5", repealed=True, text="(Mülga)"),
            section(6, "7", title="Destek oranı", amendment_notes=["(Değişik: 1/1/2026-1YK)"]),
        ],
    )
    ParsedCache(setup.artifacts / "cache" / "parsed").put(doc)
    return doc


# --- helpers -------------------------------------------------------------------------------


def test_split_hierarchy_separates_article_heading() -> None:
    assert split_hierarchy("Üçüncü Bölüm › Madde 7 – Destek oranı › (2)") == (
        "Üçüncü Bölüm",
        "Madde 7 – Destek oranı",
    )
    assert split_hierarchy("Geçici Madde 2 – Geçiş") == (None, "Geçici Madde 2 – Geçiş")
    assert split_hierarchy("SSS › Başvuru") == ("SSS › Başvuru", None)
    assert split_hierarchy("") == (None, None)


def test_section_for_matches_article_then_page(setup: _Setup) -> None:
    doc = _parsed_document(setup)
    assert section_for(doc, SOURCE_CHUNK) == "mevzuat-001:0006"
    by_page = make_chunk("x", doc_id="mevzuat-001", madde_no=None, sayfa=5)
    assert section_for(doc, by_page) == "mevzuat-001:0005"
    assert section_for(doc, make_chunk("y", madde_no=None, sayfa=None)) is None


# --- pipelines: lazy, selectable, LRU ------------------------------------------------------


def test_satisfies_the_web_backend_protocol(setup: _Setup) -> None:
    assert isinstance(setup.backend(), WebBackend)


def test_nothing_is_built_at_construction_or_listing(setup: _Setup) -> None:
    backend = setup.backend(store_factory=lambda cfg: pytest.fail("store built"))
    options = backend.pipelines()
    assert [o.ad for o in options] == ["v0", "v1", "v2", "v3"]
    assert [o.ad for o in options if o.varsayilan] == ["v3"]
    assert options[3].etiket == "v3"
    methods = {m.etiket: m.deger for m in options[0].yontemler}  # v0: fixed chunker
    assert methods["Parçalama"].startswith("sabit pencere")
    assert setup.factory.built == [] and backend.loaded_pipelines() == []


def test_ask_builds_the_default_pipeline_once_and_reuses_it(setup: _Setup) -> None:
    backend = setup.backend()
    first = backend.ask("Destek oranı?", "1507", "uygulama_esaslari")
    backend.ask("Süre?", None, None)
    assert setup.factory.names() == ["v3"] and first.yapilandirma == "v3"
    question, filters = setup.factory.built[0].questions[0]
    assert question == "Destek oranı?"
    assert filters == MetadataFilter(program_kodu="1507", doc_turu="uygulama_esaslari")
    assert setup.factory.built[0].questions[1][1] is None


def test_ask_with_a_named_config_and_unknown_config(setup: _Setup) -> None:
    backend = setup.backend()
    assert backend.ask("soru", None, None, "v1").yapilandirma == "v1"
    assert setup.factory.built[0].cfg.chunker.name == "structural"
    with pytest.raises(UnknownPipelineError, match="v9"):
        backend.ask("soru", None, None, "v9")


def test_lru_keeps_two_pipelines_and_releases_the_oldest(setup: _Setup) -> None:
    backend = setup.backend()
    for name in ["v1", "v3", "v1", "v2"]:
        backend.ask("soru", None, None, name)
    assert setup.factory.names() == ["v1", "v3", "v2"]
    evicted = setup.factory.built[1]
    assert evicted.closed and not setup.factory.built[0].closed
    assert setup.released == ["release"]
    assert backend.loaded_pipelines() == ["v1", "v2"]


def test_preload_builds_every_ready_pipeline_and_opens_its_collection(setup: _Setup) -> None:
    missing = {"v0"}
    backend = setup.backend(
        max_pipelines=4,
        index_status=lambda cfg: "yok" if cfg.experiment.name in missing else None,
    )
    assert backend.preload() == ["v1", "v2", "v3"]
    assert setup.factory.names() == ["v1", "v2", "v3"]
    assert all(p.chunk_lookups == [[]] for p in setup.factory.built)
    backend.ask("soru", None, None, "v2")
    assert len(setup.factory.built) == 3  # the question reuses the preloaded pipeline


def test_preload_stops_at_max_pipelines(setup: _Setup) -> None:
    backend = setup.backend()  # max_pipelines=2
    assert backend.preload() == ["v0", "v1"]
    assert setup.released == []


def test_answer_context_comes_from_stored_chunks_and_parsed_sections(setup: _Setup) -> None:
    _parsed_document(setup)
    result = setup.backend().ask("Destek oranı?", None, None)
    context = result.baglam[SOURCE_CHUNK.chunk_id]
    assert context.hiyerarsi == "Üçüncü Bölüm – Destek"
    assert context.madde_baslik == "Madde 7 – Destek oranı"
    assert context.sayfa == 6 and context.program_kodu == "1507"
    assert context.doc_id == "mevzuat-001" and context.bolum_id == "mevzuat-001:0006"
    assert result.answer.atif_onarimi == [1]


def test_refusal_has_no_sources_and_skips_the_chunk_lookup(setup: _Setup) -> None:
    setup.factory.answer = make_answer(durum=AnswerStatus.BILGI_BULUNAMADI)
    result = setup.backend().ask("2244 bursu?", None, None)
    assert result.answer.kaynaklar == [] and result.baglam == {}
    assert setup.factory.built[0].chunk_lookups == []


def test_ask_holds_the_gpu_lock_only_while_answering(setup: _Setup) -> None:
    setup.backend().ask("soru", None, None)
    assert setup.factory.built[0].lock_seen is True
    assert not (setup.artifacts / GPU_LOCK_FILE).exists()


def test_held_gpu_lock_refuses_the_question_and_frees_pipelines(setup: _Setup) -> None:
    backend = setup.backend()
    backend.ask("soru", None, None)
    lock = setup.artifacts / GPU_LOCK_FILE
    lock.write_text("rag run (v3b) pid=1\n", encoding="utf-8")
    with pytest.raises(GpuLockError, match="rag run"):
        backend.ask("soru", None, None)
    assert backend.loaded_pipelines() == [] and setup.factory.built[0].closed
    assert lock.exists()


def test_missing_index_is_service_unavailable(setup: _Setup) -> None:
    setup.factory.error = ConfigError("BM25 indeksi bulunamadı. Önce `rag ingest` çalıştırın.")
    with pytest.raises(ServiceUnavailableError, match="rag ingest"):
        setup.backend().ask("soru", None, None)


# --- documents -----------------------------------------------------------------------------


def test_documents_combine_manifest_index_and_ingest_report(setup: _Setup) -> None:
    collection = setup.collection()
    setup.store.ensure_collection(collection, 8)
    chunks = [make_chunk(f"mevzuat-001:{i}", doc_id="mevzuat-001") for i in range(3)]
    setup.store.upsert(collection, chunks, [[1.0] * 8] * 3)
    setup.artifacts.mkdir(parents=True, exist_ok=True)
    report = {"scanned": ["mevzuat-002"]}
    (setup.artifacts / "ingest_report.json").write_text(json.dumps(report), encoding="utf-8")
    documents = setup.backend().documents()
    states = {d.doc_id: (d.durum, d.chunk_sayisi) for d in documents.belgeler}
    assert states == {
        "mevzuat-001": ("indeksli", 3),
        "mevzuat-002": ("taranmis", 0),
        "mevzuat-003": ("guncel_degil", 0),
        "mevzuat-004": ("eksik", 0),
    }
    assert documents.koleksiyon == collection and documents.toplam_chunk == 3


def test_documents_without_vector_store_are_service_unavailable(setup: _Setup) -> None:
    class Down(InMemoryVectorStore):
        def doc_chunk_counts(
            self, name: str, doc_ids: Sequence[str] | None = None
        ) -> dict[str, int]:
            raise ConnectionError("bağlantı reddedildi")

    with pytest.raises(ServiceUnavailableError, match="Vektör deposuna"):
        setup.backend(store_factory=lambda cfg: Down()).documents()


def test_about_counts_manifest_parse_cache_and_index(setup: _Setup) -> None:
    _parsed_document(setup)  # three parsed sections
    collection = setup.collection()
    setup.store.ensure_collection(collection, 8)
    chunks = [make_chunk(f"mevzuat-001:{i}", doc_id="mevzuat-001") for i in range(3)]
    setup.store.upsert(collection, chunks, [[1.0] * 8] * 3)
    about = setup.backend().about()
    assert [(g.kaynak_grubu, g.belge_sayisi, g.ozet) for g in about.gruplar] == [
        ("tubitak_mevzuat", 4, "4 PDF")
    ]
    assert about.belge_sayisi == 4 and about.bolum_sayisi == 3 and about.parca_sayisi == 3
    assert about.koleksiyon == collection
    assert [o.ad for o in about.yapilandirmalar] == ["v0", "v1", "v2", "v3"]
    assert setup.factory.built == []  # no pipeline is built for the page


def test_about_without_index_falls_back_to_the_ingest_report(setup: _Setup) -> None:
    class Down(InMemoryVectorStore):
        def count(self, name: str) -> int:
            raise ConnectionError("bağlantı reddedildi")

    backend = setup.backend(store_factory=lambda cfg: Down())
    about = backend.about()
    assert about.bolum_sayisi is None and about.parca_sayisi is None
    setup.artifacts.mkdir(parents=True, exist_ok=True)
    report = {"chunks_total": 42}
    (setup.artifacts / "ingest_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert backend.about().parca_sayisi == 42


def test_disclaimer_comes_from_the_server_config(setup: _Setup) -> None:
    assert setup.backend().disclaimer() == setup.cfg.guardrails.disclaimer


def test_document_preview_lists_parsed_sections_and_counts(setup: _Setup) -> None:
    _parsed_document(setup)
    collection = setup.collection()
    setup.store.ensure_collection(collection, 8)
    setup.store.upsert(collection, [SOURCE_CHUNK], [[1.0] * 8])
    preview = setup.backend().document("mevzuat-001")
    assert preview is not None and preview.ayristirildi
    assert [s.bolum_id for s in preview.bolumler] == [
        "mevzuat-001:0001",
        "mevzuat-001:0005",
        "mevzuat-001:0006",
    ]
    repealed, amended = preview.bolumler[1], preview.bolumler[2]
    assert repealed.mulga and amended.degisiklik_notlari == ["(Değişik: 1/1/2026-1YK)"]
    assert amended.baslik == "Destek oranı" and amended.sayfa == 6
    assert preview.chunk_sayilari[collection] == 1
    fixed = setup.collection(load_config(setup.config_path.parent / "experiments" / "v0.yaml"))
    assert preview.chunk_sayilari[fixed] == 0
    assert preview.belge.durum == "indeksli"


def test_document_preview_of_unparsed_and_unknown_documents(setup: _Setup) -> None:
    backend = setup.backend()
    missing = backend.document("mevzuat-004")
    assert missing is not None and not missing.ayristirildi and missing.belge.durum == "eksik"
    assert backend.document("yok") is None


# --- runs ----------------------------------------------------------------------------------


def _runs_dir(setup: _Setup, fixtures: Path) -> Path:
    runs = setup.artifacts / "runs"
    shutil.copytree(fixtures / "run_example", runs / "v1_20261016-120000")
    (runs / "v1_20261016-120000" / "report.md").write_text("# Rapor v1\n", encoding="utf-8")
    running = runs / "v2_20261016-130000"
    running.mkdir()
    (running / "config.yaml").write_text(
        yaml.safe_dump({"experiment": {"name": "v2"}}), encoding="utf-8"
    )
    (runs / "ozet.md").write_text("# Özet\n", encoding="utf-8")
    return runs


def test_runs_list_finished_and_running_newest_first(setup: _Setup, fixtures: Path) -> None:
    _runs_dir(setup, fixtures)
    runs = setup.backend().runs()
    assert [(r.run_id, r.durum) for r in runs] == [
        ("v2_20261016-130000", "suruyor"),
        ("v1_20261016-120000", "tamam"),
    ]
    assert runs[0].deney == "v2" and runs[0].tarih == "2026-10-16T13:00:00"
    assert runs[1].metrikler


def test_run_artifacts_by_id(setup: _Setup, fixtures: Path) -> None:
    _runs_dir(setup, fixtures)
    backend = setup.backend()
    detail = backend.run_detail("v1_20261016-120000")
    assert detail is not None and detail.sorular
    assert detail.metrics.run_id == "v1_20261016-120000"
    metrics = backend.run_metrics("v1_20261016-120000")
    assert metrics is not None and metrics.experiment == "v1"
    assert backend.run_report("v1_20261016-120000") == "# Rapor v1\n"
    assert backend.summary_report() == "# Özet\n"
    for run_id in ["v2_20261016-130000", "yok", "../runs", ".."]:
        assert backend.run_detail(run_id) is None and backend.run_metrics(run_id) is None
        assert backend.run_report(run_id) is None


def test_no_runs_directory_means_no_runs(setup: _Setup) -> None:
    backend = setup.backend()
    assert backend.runs() == [] and backend.summary_report() is None


# --- background jobs -----------------------------------------------------------------------


def test_ingest_job_runs_the_cli_and_frees_cached_pipelines(setup: _Setup) -> None:
    backend = setup.backend()
    backend.ask("soru", None, None)
    job = backend.start_ingest(full=True)
    assert setup.jobs.wait(job.job_id, timeout=5) is not None
    ((argv, log),) = setup.commands.calls
    assert argv[-4:] == ["ingest", "-c", str(setup.config_path), "--full"]
    assert log.name == f"{job.job_id}.log"
    assert backend.loaded_pipelines() == [] and setup.factory.built[0].closed


def test_run_job_runs_the_named_experiment_with_limit(setup: _Setup) -> None:
    backend = setup.backend()
    job = backend.start_run("v2", limit=5)
    setup.jobs.wait(job.job_id, timeout=5)
    ((argv, _),) = setup.commands.calls
    experiment = setup.config_path.parent / "experiments" / "v2.yaml"
    assert argv[-4:] == ["run", str(experiment), "--limit", "5"]
    with pytest.raises(UnknownPipelineError):
        backend.start_run("v9")


def test_run_accepts_a_config_path(setup: _Setup) -> None:
    backend = setup.backend()
    job = backend.start_run("configs/experiments/v1.yaml")
    setup.jobs.wait(job.job_id, timeout=5)
    assert setup.commands.calls[0][0][-1].endswith("v1.yaml")


def test_jobs_respect_the_gpu_lock(setup: _Setup) -> None:
    setup.artifacts.mkdir(parents=True, exist_ok=True)
    (setup.artifacts / GPU_LOCK_FILE).write_text("rag run (v3)\n", encoding="utf-8")
    backend = setup.backend()
    with pytest.raises(GpuLockError, match="GPU kilidi dolu"):
        backend.start_ingest()
    with pytest.raises(GpuLockError):
        backend.start_run("v3")
    assert setup.commands.calls == []


def test_only_one_job_at_a_time(setup: _Setup) -> None:
    import threading

    release = threading.Event()

    def slow(argv: Sequence[str], log_path: Path) -> None:
        release.wait(5)

    backend = setup.backend(command_runner=slow)
    job = backend.start_ingest()
    with pytest.raises(GpuLockError, match="iş sürüyor"):
        backend.start_run("v3")
    release.set()
    setup.jobs.wait(job.job_id, timeout=5)


# --- health --------------------------------------------------------------------------------


def test_health_reports_probes_without_building_anything(setup: _Setup) -> None:
    backend = setup.backend()
    health = backend.health()
    assert (health.qdrant, health.ollama, health.durum) == (True, False, "sorunlu")
    assert health.koleksiyon == setup.collection()
    ok = setup.backend(probes={"qdrant": lambda: True, "ollama": lambda: True}).health()
    assert ok.durum == "ok"
    assert setup.factory.built == []


def test_failing_probe_counts_as_down(setup: _Setup) -> None:
    def broken() -> bool:
        raise ConnectionError("yok")

    health = setup.backend(probes={"qdrant": broken, "ollama": broken}).health()
    assert not health.qdrant and not health.ollama


def test_from_config_loads_the_file(setup: _Setup) -> None:
    backend = PipelineBackend.from_config(setup.config_path, pipeline_factory=setup.factory)
    assert backend.pipelines()[-1].ad == "v3" and setup.factory.built == []


def test_default_probes_ask_qdrant_and_the_ollama_tags_endpoint(
    setup: _Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    class _Response:
        is_success = True

    def fake_get(url: str, timeout: float) -> _Response:
        urls.append(url)
        return _Response()

    monkeypatch.setattr("httpx.get", fake_get)
    health = setup.backend(probes=None).health()
    assert health.qdrant and health.ollama and health.durum == "ok"
    assert urls == [f"{setup.cfg.llm.url.rstrip('/')}/api/tags"]
    assert setup.factory.built == []


def test_without_experiment_configs_the_server_config_is_the_only_pipeline(
    setup: _Setup, tmp_path: Path
) -> None:
    empty = tmp_path / "bos"
    empty.mkdir()
    options = setup.backend(experiments_dir=empty).pipelines()
    assert [(o.ad, o.varsayilan) for o in options] == [("default", True)]


def test_failing_chunk_lookup_keeps_the_answer_without_context(setup: _Setup) -> None:
    backend = setup.backend()
    backend.ask("soru", None, None)
    pipeline = setup.factory.built[0]

    def broken(chunk_ids: Sequence[str]) -> list[Chunk]:
        raise ConnectionError("Qdrant yok")

    pipeline.get_chunks = broken  # type: ignore[method-assign]
    result = backend.ask("soru", None, None)
    assert result.answer.kaynaklar and result.baglam == {}


def test_store_without_chunk_counts_still_marks_indexed_documents(setup: _Setup) -> None:
    class Minimal:
        def indexed_docs(self, name: str) -> dict[str, str]:
            return {"mevzuat-001": "parmak-izi"}

    documents = setup.backend(store_factory=lambda cfg: Minimal()).documents()
    states = {d.doc_id: d.durum for d in documents.belgeler}
    assert states["mevzuat-001"] == "indeksli" and states["mevzuat-003"] == "guncel_degil"


# --- configs whose index is missing (Docker: only the bundle's collections exist) ----------


def _missing(*names: str) -> Any:
    return lambda cfg: "koleksiyon boş" if cfg.experiment.name in names else None


def test_configs_without_index_are_listed_disabled_with_a_hint(setup: _Setup) -> None:
    options = setup.backend(index_status=_missing("v0", "v2")).pipelines()
    state = {o.ad: (o.hazir, o.durum_notu) for o in options}
    assert state == {
        "v0": (False, "indeks yok"),
        "v1": (True, None),
        "v2": (False, "indeks yok"),
        "v3": (True, None),
    }
    assert [o.ad for o in options if o.varsayilan] == ["v3"]


def test_default_moves_to_a_config_that_has_an_index(setup: _Setup) -> None:
    backend = setup.backend(index_status=_missing("v3"))
    assert [o.ad for o in backend.pipelines() if o.varsayilan] == ["v0"]
    assert backend.ask("soru", None, None).yapilandirma == "v0"


def test_asking_a_config_without_index_is_service_unavailable(setup: _Setup) -> None:
    backend = setup.backend(index_status=_missing("v1"))
    with pytest.raises(ServiceUnavailableError, match="v1.*indeks yok.*koleksiyon boş"):
        backend.ask("soru", None, None, "v1")
    assert setup.factory.built == []


def test_index_status_is_cached_for_a_short_time(setup: _Setup) -> None:
    calls: list[str] = []

    def status(cfg: AppConfig) -> str | None:
        calls.append(cfg.experiment.name)
        return None

    backend = setup.backend(index_status=status)
    backend.pipelines()
    backend.pipelines()
    assert sorted(calls) == ["v0", "v1", "v2", "v3"]
    fresh = setup.backend(index_status=status, index_ttl_s=0)
    fresh.pipelines()
    fresh.pipelines()
    assert len(calls) >= 4 + 8  # every listing asks again


def test_failing_index_status_counts_as_available(setup: _Setup) -> None:
    def broken(cfg: AppConfig) -> str | None:
        raise RuntimeError("qdrant kapalı")

    assert all(o.hazir for o in setup.backend(index_status=broken).pipelines())


def test_index_problem_needs_points_and_bm25_for_lexical_modes(setup: _Setup) -> None:
    hybrid = setup.cfg
    dense = hybrid.model_copy(
        update={"retriever": hybrid.retriever.model_copy(update={"mode": "dense"})}
    )
    assert index_problem(hybrid, 0) is not None and "boş" in index_problem(hybrid, 0)  # type: ignore[operator]
    assert index_problem(dense, 12) is None and index_problem(dense, None) is None
    problem = index_problem(hybrid, 12)
    assert problem is not None and "BM25" in problem
    (setup.artifacts / "cache" / "bm25" / "structural_ctx").mkdir(parents=True)
    assert index_problem(hybrid, 12) is None


def test_default_index_status_without_qdrant_checks_only_bm25(setup: _Setup) -> None:
    kwargs: dict[str, Any] = {"index_status": None}
    options = {o.ad: o.hazir for o in setup.backend(**kwargs).pipelines()}
    assert options == {"v0": False, "v1": False, "v2": False, "v3": False}  # all hybrid
    (setup.artifacts / "cache" / "bm25" / "structural_ctx").mkdir(parents=True)
    options = {o.ad: o.hazir for o in setup.backend(**kwargs).pipelines()}
    assert options == {"v0": False, "v1": True, "v2": True, "v3": True}


def test_preview_and_state_without_raw_files_after_a_bundle_restore(setup: _Setup) -> None:
    _parsed_document(setup)
    collection = setup.collection()
    setup.store.ensure_collection(collection, 8)
    setup.store.upsert(collection, [SOURCE_CHUNK], [[1.0] * 8])
    (setup.corpus.root / "raw" / "tubitak_mevzuat" / "001_a.pdf").unlink()  # not downloaded

    preview = setup.backend().document("mevzuat-001")

    assert preview is not None and preview.ayristirildi and len(preview.bolumler) == 3
    assert preview.belge.durum == "indeksli"  # indexed counts more than the missing file
    rows = {row.doc_id: row.durum for row in setup.backend().documents().belgeler}
    assert rows["mevzuat-001"] == "indeksli" and rows["mevzuat-004"] == "eksik"


# --- Hugging Face Space options ------------------------------------------------------------


def test_overrides_apply_to_the_server_and_every_experiment_config(setup: _Setup) -> None:
    overlay = {"llm": {"model": "${KATMAN_MODELI:katman-modeli}"}, "reranker": {"device": "cpu"}}
    backend = PipelineBackend.from_config(
        setup.config_path,
        config_overrides=overlay,
        pipeline_factory=setup.factory,
        index_status=lambda cfg: None,
    )
    assert backend.cfg.llm.model == "katman-modeli"  # interpolated like the files
    backend.ask("soru", None, None, "v1")
    built = setup.factory.built[0].cfg
    assert built.experiment.name == "v1" and built.llm.model == "katman-modeli"
    assert built.reranker.device == "cpu" and built.chunker.name == "structural"


def test_default_pipeline_option_wins_when_its_index_exists(setup: _Setup) -> None:
    backend = setup.backend(default_pipeline="v1")
    assert [o.ad for o in backend.pipelines() if o.varsayilan] == ["v1"]
    assert backend.ask("soru", None, None).yapilandirma == "v1"
    fallback = setup.backend(default_pipeline="v1", index_status=_missing("v1"))
    assert [o.ad for o in fallback.pipelines() if o.varsayilan] == ["v3"]


def test_gpu_lock_can_be_turned_off_on_cpu_hosts(setup: _Setup) -> None:
    lock = setup.artifacts / GPU_LOCK_FILE
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("başka bir iş", encoding="utf-8")
    # a held (or stale) lock does not refuse questions when the lock is off
    assert setup.backend(use_gpu_lock=False).ask("soru", None, None).yapilandirma == "v3"
    assert lock.read_text(encoding="utf-8") == "başka bir iş"  # neither taken nor removed


def test_store_index_status_reads_the_numpy_export(setup: _Setup, tmp_path: Path) -> None:
    export = tmp_path / "export"
    cfg = setup.cfg.model_copy(
        update={
            "vector_store": setup.cfg.vector_store.model_copy(
                update={"provider": "numpy", "export_dir": export}
            ),
            "retriever": setup.cfg.retriever.model_copy(update={"mode": "dense"}),
        }
    )
    assert store_index_status(cfg) is not None  # nothing exported yet
    write_collection(
        export, setup.collection(cfg), [SOURCE_CHUNK], [[1.0, 0.0]], embedder_model="m"
    )
    assert store_index_status(cfg) is None
