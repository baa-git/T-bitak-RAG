"""Tests for the ingest orchestration (FR-I2..FR-I6, FR-D5, FR-P9, FR-X9).

A tiny corpus is written to ``tmp_path`` (manifests + raw files). The real parse cache,
chunk cache, structural chunker (regex tokens), embedding cache, sync and BM25 rebuild run;
only the parser, the embedder, the vector store and the lexical index are fakes.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.fakes.embedder import FakeEmbedder
from tests.fakes.lexical_index import FakeLexicalIndex
from tests.fakes.parser import FakeParser
from tests.fakes.vector_store import InMemoryVectorStore
from tests.fixtures.corpus import CorpusOnDisk
from tubitak_rag.factory import build_chunker
from tubitak_rag.ingestion import parsed_cache
from tubitak_rag.ingestion.ingest import IngestReport, LazyEmbedder, run_ingest

COLLECTION = "structural_ctx__fake"


class TextCountingEmbedder(FakeEmbedder):
    """FakeEmbedder that remembers every text it embedded; can fail on a given call."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.texts: list[str] = []
        self.fail_on_call = fail_on_call

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail_on_call is not None and self.calls + 1 == self.fail_on_call:
            raise RuntimeError("GPU gitti")
        self.texts += texts
        return super().embed_documents(texts)


class Corpus(CorpusOnDisk):
    """The corpus on disk plus the fake components; ``ingest()`` runs one ingest."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.parser = FakeParser()
        self.embedder = TextCountingEmbedder()
        self.store = InMemoryVectorStore()
        self.lexical = FakeLexicalIndex()

    def ingest(self, *, full: bool = False, **sections: dict[str, Any]) -> IngestReport:
        cfg = self.config(**sections)
        return run_ingest(
            cfg,
            embedder=self.embedder,
            store=self.store,
            chunker=build_chunker(cfg),
            lexical=self.lexical,
            full=full,
        )

    def chunk_ids(self, doc_id: str) -> set[str]:
        return {
            cid for cid, (c, _) in self.store.collections[COLLECTION].items() if c.doc_id == doc_id
        }


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_components: None) -> Corpus:
    corpus = Corpus(tmp_path)
    monkeypatch.setattr(parsed_cache, "parser_for", lambda entry: corpus.parser)
    corpus.add("001_a.pdf", "Destek oranı yüzde yetmiş beştir.\nBütçe sınırı vardır.")
    corpus.add("002_b.pdf", "Başvurular çağrı ile alınır.")
    return corpus


def test_first_ingest_adds_all_and_builds_bm25(corpus: Corpus) -> None:
    report = corpus.ingest()

    assert report.collection == COLLECTION
    assert report.added == ["mevzuat-001", "mevzuat-002"]
    assert report.updated == [] and report.removed == [] and report.unchanged == 0
    assert set(corpus.store.indexed_docs(COLLECTION)) == {"mevzuat-001", "mevzuat-002"}
    assert report.chunks_total == corpus.store.count(COLLECTION) == report.chunks_indexed
    assert report.bm25_size == report.chunks_total
    assert (corpus.artifacts / "cache" / "bm25" / "structural_ctx").is_dir()
    best = corpus.lexical.search("çağrı", 5)[0][0]
    assert corpus.store.get_chunks(COLLECTION, [best])[0].doc_id == "mevzuat-002"


def test_report_is_written_and_counts_cache_layers(corpus: Corpus) -> None:
    report = corpus.ingest()

    saved = json.loads((corpus.artifacts / "ingest_report.json").read_text(encoding="utf-8"))
    assert IngestReport.model_validate(saved) == report
    assert (report.parsed, report.parse_cached) == (2, 0)
    assert (report.chunked, report.chunk_cached) == (2, 0)
    assert (report.embedded, report.embed_cached) == (report.chunks_indexed, 0)
    assert set(report.stage_times_s) == {"parse", "chunk", "embed_index", "bm25", "total"}


def test_second_ingest_is_noop_and_skips_parse_and_embed(corpus: Corpus) -> None:
    corpus.ingest()
    parses, embeds = corpus.parser.calls, corpus.embedder.calls

    report = corpus.ingest()

    assert corpus.parser.calls == parses and corpus.embedder.calls == embeds  # FR-X9, FR-I2
    assert report.added == [] and report.updated == [] and report.removed == []
    assert report.unchanged == 2 and report.chunks_indexed == 0
    assert (report.parse_cached, report.chunk_cached, report.embedded) == (2, 2, 0)
    assert report.bm25_size == report.chunks_total  # BM25 is rebuilt every time (FR-I6)


def test_changed_document_is_reindexed(corpus: Corpus) -> None:
    corpus.ingest()
    corpus.add("001_a.pdf", "Destek oranı yüzde yetmiş beştir.\nBütçe sınırı vardır.\nYeni fıkra.")
    corpus.embedder.texts.clear()

    report = corpus.ingest()

    assert report.updated == ["mevzuat-001"] and report.unchanged == 1  # FR-I3
    assert len(corpus.chunk_ids("mevzuat-001")) == report.chunks_indexed > 0
    assert corpus.embedder.texts and all(t.startswith("Belge 001") for t in corpus.embedder.texts)
    assert corpus.lexical.search("fıkra", 5)


def test_ingest_removes_deleted_doc_from_both_indexes(corpus: Corpus) -> None:
    corpus.ingest()
    removed_ids = corpus.chunk_ids("mevzuat-002")
    corpus.drop_from_manifest("002_b.pdf")

    report = corpus.ingest()

    assert report.removed == ["mevzuat-002"]  # FR-I4
    assert "mevzuat-002" not in corpus.store.indexed_docs(COLLECTION)
    reloaded = FakeLexicalIndex()
    reloaded.load(corpus.artifacts / "cache" / "bm25" / "structural_ctx")
    for index in (corpus.lexical, reloaded):
        assert not {cid for cid, _ in index.search("başvurular çağrı alınır", 10)} & removed_ids


def test_scanned_docs_reported_not_indexed(corpus: Corpus) -> None:
    corpus.add("003_c.pdf", "TARANMIS")

    report = corpus.ingest()

    assert report.scanned == ["mevzuat-003"]  # FR-P9
    assert "mevzuat-003" not in corpus.store.indexed_docs(COLLECTION)
    assert "mevzuat-003" not in report.added


def test_missing_and_broken_sources_are_reported(corpus: Corpus) -> None:
    corpus.register_only("003_c.pdf")
    corpus.add("004_d.pdf", "BOZUK")

    report = corpus.ingest()

    assert report.missing == ["mevzuat-003"]
    assert report.errors == [{"doc_id": "mevzuat-004", "message": "bozuk dosya"}]
    assert report.docs_total == 4 and report.added == ["mevzuat-001", "mevzuat-002"]


def test_no_raw_files_at_all_leaves_the_index_untouched(corpus: Corpus) -> None:
    # A Docker stack restored from a prebuilt bundle has the index but no raw documents:
    # an ingest there must not read "every document was removed" and empty the index.
    import shutil

    corpus.ingest()
    before = corpus.store.count(COLLECTION)
    bm25 = corpus.artifacts / "cache" / "bm25" / "structural_ctx"
    shutil.rmtree(corpus.root / "raw")

    report = corpus.ingest()

    assert report.removed == [] and corpus.store.count(COLLECTION) == before > 0
    assert report.missing == ["mevzuat-001", "mevzuat-002"] and bm25.is_dir()
    assert report.chunks_total == before and "rag fetch" in report.message


def test_document_without_chunks_is_reported_once_not_added_every_time(corpus: Corpus) -> None:
    corpus.add("003_c.pdf", "   \n")

    first = corpus.ingest()
    second = corpus.ingest()

    assert first.empty == ["mevzuat-003"] and "mevzuat-003" not in first.added
    assert second.added == []


def test_manually_added_doc_is_indexed(corpus: Corpus) -> None:
    corpus.ingest()
    corpus.add("005_e.pdf", "Elle eklenen yönerge metni.")  # FR-D5: one manifest record

    report = corpus.ingest()

    assert report.added == ["mevzuat-005"] and report.unchanged == 2
    assert "mevzuat-005" in corpus.store.indexed_docs(COLLECTION)


def test_full_flag_rebuilds_everything_from_the_caches(corpus: Corpus) -> None:
    corpus.ingest()
    stray = next(iter(corpus.store.collections[COLLECTION].values()))
    corpus.store.collections[COLLECTION]["eski:9999:00"] = stray  # leftover point
    embeds = corpus.embedder.calls

    report = corpus.ingest(full=True)

    assert report.full and report.added == ["mevzuat-001", "mevzuat-002"]
    assert "eski:9999:00" not in corpus.store.collections[COLLECTION]
    assert corpus.store.count(COLLECTION) == report.chunks_total
    assert corpus.embedder.calls == embeds  # vectors come from the embedding cache
    assert (report.embedded, report.embed_cached) == (0, report.chunks_indexed)


def test_embedding_cache_can_be_disabled(corpus: Corpus) -> None:
    corpus.ingest(embedder={"cache": False})
    embeds = corpus.embedder.calls

    report = corpus.ingest(full=True, embedder={"cache": False})

    assert corpus.embedder.calls > embeds and report.embedded == report.chunks_indexed
    assert not (corpus.artifacts / "cache" / "embeddings").exists()


def test_interrupted_ingest_resumes_without_recomputing(corpus: Corpus) -> None:
    corpus.embedder = TextCountingEmbedder(fail_on_call=2)  # dies on the second document
    with pytest.raises(RuntimeError):
        corpus.ingest()
    assert set(corpus.store.indexed_docs(COLLECTION)) == {"mevzuat-001"}
    corpus.embedder = TextCountingEmbedder()

    report = corpus.ingest()

    assert report.added == ["mevzuat-002"] and report.unchanged == 1
    assert all(text.startswith("Belge 002") for text in corpus.embedder.texts)


def test_embedding_batches_follow_config_batch_size(corpus: Corpus) -> None:
    corpus.add("003_c.pdf", "\n".join(f"Madde metni {i} burada." for i in range(5)))
    sizes: list[int] = []
    inner = corpus.embedder.embed_documents
    corpus.embedder.embed_documents = lambda texts: sizes.append(len(texts)) or inner(texts)  # type: ignore[method-assign]

    corpus.ingest(embedder={"batch_size": 2})

    assert sizes and max(sizes) <= 2


def test_bm25_disabled_skips_lexical_index(corpus: Corpus) -> None:
    report = corpus.ingest(bm25={"enabled": False})

    assert report.bm25_size == 0 and len(corpus.lexical) == 0
    assert not (corpus.artifacts / "cache" / "bm25").exists()


def test_progress_reports_every_stage(corpus: Corpus) -> None:
    events: list[tuple[str, int, int]] = []
    cfg = corpus.config()

    run_ingest(
        cfg,
        embedder=corpus.embedder,
        store=corpus.store,
        chunker=build_chunker(cfg),
        lexical=corpus.lexical,
        progress=lambda stage, done, total: events.append((stage, done, total)),
    )

    assert [e for e in events if e[0] == "parse"][-1] == ("parse", 2, 2)
    assert [e for e in events if e[0] == "chunk"][-1] == ("chunk", 2, 2)
    embed = [e for e in events if e[0] == "embed"]
    assert embed[0][1] == 0 and embed[-1][1] == embed[-1][2] > 0
    assert events[-1][0] == "bm25"


# --- LazyEmbedder -----------------------------------------------------------------------


def test_lazy_embedder_builds_inner_only_when_documents_are_embedded() -> None:
    built: list[FakeEmbedder] = []

    def build() -> FakeEmbedder:
        built.append(FakeEmbedder(dimension=8))
        return built[-1]

    lazy = LazyEmbedder(build, model_name="fake", dimension=8)
    assert (lazy.model_name, lazy.dimension, built) == ("fake", 8, [])

    lazy.embed_documents(["a"])
    lazy.embed_query("b")

    assert len(built) == 1 and built[0].calls == 1


def test_lazy_embedder_rejects_dimension_mismatch() -> None:
    lazy = LazyEmbedder(lambda: FakeEmbedder(dimension=4), model_name="fake", dimension=8)

    with pytest.raises(ValueError, match="boyut"):
        lazy.embed_documents(["a"])
