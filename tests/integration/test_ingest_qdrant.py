"""Ingest against a real Qdrant server with fake embeddings (no model, no GPU).

Checks the parts the in-memory fake cannot prove: fingerprints read back from Qdrant
payloads make the second run a no-op, deletions reach the server and ``--full`` drops and
recreates the collection. Needs ``QDRANT_URL``; the test collection is always dropped.
"""

import uuid
from pathlib import Path

import pytest

from tests.fakes.embedder import FakeEmbedder
from tests.fakes.lexical_index import FakeLexicalIndex
from tests.fakes.parser import FakeParser
from tests.fixtures.corpus import CorpusOnDisk
from tubitak_rag.adapters.qdrant_store import QdrantVectorStore
from tubitak_rag.config import AppConfig
from tubitak_rag.factory import build_chunker
from tubitak_rag.ingestion import parsed_cache
from tubitak_rag.ingestion.ingest import IngestReport, run_ingest

pytestmark = pytest.mark.integration


def test_ingest_cycle_on_real_qdrant(
    qdrant_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_components: None
) -> None:
    monkeypatch.setattr(parsed_cache, "parser_for", lambda entry: FakeParser())
    corpus = CorpusOnDisk(tmp_path)
    corpus.add("001_a.pdf", "Destek oranı yüzde yetmiş beştir.\nBütçe sınırı vardır.")
    corpus.add("002_b.pdf", "Başvurular çağrı ile alınır.")
    model = f"fake-ingest-{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(qdrant_url)
    embedder = FakeEmbedder(model_name=model)
    lexical = FakeLexicalIndex()

    def ingest(cfg: AppConfig, *, full: bool = False) -> IngestReport:
        return run_ingest(
            cfg,
            embedder=embedder,
            store=store,
            chunker=build_chunker(cfg),
            lexical=lexical,
            full=full,
        )

    cfg = corpus.config(embedder={"model": model})
    collection = f"structural_ctx__{model}"
    try:
        first = ingest(cfg)
        assert first.collection == collection and first.added == ["mevzuat-001", "mevzuat-002"]
        assert store.count(collection) == first.chunks_total

        calls = embedder.calls
        second = ingest(cfg)
        assert second.added == second.updated == [] and second.unchanged == 2
        assert embedder.calls == calls

        corpus.drop_from_manifest("002_b.pdf")
        third = ingest(corpus.config(embedder={"model": model}))
        assert third.removed == ["mevzuat-002"]
        assert set(store.indexed_docs(collection)) == {"mevzuat-001"}

        rebuilt = ingest(corpus.config(embedder={"model": model}), full=True)
        assert rebuilt.added == ["mevzuat-001"] and rebuilt.embedded == 0
        assert store.count(collection) == rebuilt.chunks_total
    finally:
        store.drop_collection(collection)
