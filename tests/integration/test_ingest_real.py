"""Checks the three real collections after ``rag ingest``.

Run *after* the real ingests (``rag ingest`` for every experiment config). The test re-runs the
ingest with an embedder that refuses to load a model: every document must already be in the
collection with the same fingerprint, so nothing is embedded and the GPU is never touched.
Needs ``QDRANT_URL``; a collection that was not ingested yet is skipped. Note: the run
rewrites ``artifacts/ingest_report.json`` and the BM25 directory (same content).
"""

from pathlib import Path

import pytest

from tubitak_rag.chunking.cache import chunker_key
from tubitak_rag.config import load_config
from tubitak_rag.factory import build_chunker, build_lexical_index, build_vector_store
from tubitak_rag.indexing.naming import collection_name
from tubitak_rag.ingestion.ingest import LazyEmbedder, run_ingest
from tubitak_rag.ports import Embedder

pytestmark = pytest.mark.integration


def _refuse_model() -> Embedder:
    raise AssertionError("Bu test model yüklememeli: koleksiyon güncel değil, önce rag ingest.")


@pytest.mark.parametrize("experiment", ["v0", "v1", "v3b"])
def test_real_collection_is_complete_and_second_ingest_is_noop(
    qdrant_url: str, repo_root: Path, experiment: str
) -> None:
    cfg = load_config(
        repo_root / "configs" / "experiments" / f"{experiment}.yaml",
        {"vector_store": {"url": qdrant_url}},
    )
    store = build_vector_store(cfg)
    lazy = LazyEmbedder(
        _refuse_model, model_name=cfg.embedder.model, dimension=cfg.embedder.dimension
    )
    lexical = build_lexical_index(cfg)
    chunker = build_chunker(cfg)
    collection = collection_name(chunker_key(cfg.chunker), cfg.embedder.model)
    if not store.indexed_docs(collection):
        pytest.skip(
            f"{collection} boş; önce: rag ingest --config configs/experiments/{experiment}.yaml"
        )

    report = run_ingest(cfg, embedder=lazy, store=store, chunker=chunker, lexical=lexical)

    assert report.added == report.updated == report.removed == []
    assert report.embedded == 0 and report.chunks_indexed == 0
    skipped = len(report.scanned) + len(report.empty) + len(report.missing) + len(report.errors)
    assert len(store.indexed_docs(collection)) == report.docs_total - skipped
    assert store.count(collection) == report.chunks_total == report.bm25_size
    hits = lexical.search("1507 destek oranı", 5)
    assert hits, "BM25 indeksi '1507 destek oranı' için sonuç döndürmedi"
    top = store.get_chunks(collection, [hits[0][0]])
    assert top and top[0].text
