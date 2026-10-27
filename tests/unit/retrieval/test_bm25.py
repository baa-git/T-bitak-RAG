"""Tests for the BM25 (keyword) retriever (FR-R5)."""

import pytest

from tests.builders import make_chunk
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.lexical_index import FakeLexicalIndex
from tests.fakes.vector_store import InMemoryVectorStore
from tubitak_rag.config import ConfigError
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.retrieval import MetadataFilter
from tubitak_rag.ports import Retriever
from tubitak_rag.retrieval.bm25 import Bm25Retriever, check_index_matches_collection

COLLECTION = "structural_ctx__fake"


class _RecordingIndex(FakeLexicalIndex):
    def __init__(self) -> None:
        super().__init__()
        self.ks: list[int] = []

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        self.ks.append(k)
        return super().search(query, k)


def _setup(
    chunks: list[Chunk], stored: list[Chunk] | None = None
) -> tuple[_RecordingIndex, Bm25Retriever]:
    index, store = _RecordingIndex(), InMemoryVectorStore()
    index.build(chunks)
    stored = chunks if stored is None else stored
    store.ensure_collection(COLLECTION, 8)
    store.upsert(COLLECTION, stored, FakeEmbedder().embed_documents([c.text for c in stored]))
    return index, Bm25Retriever(index, store, COLLECTION)


def _ids(retriever: Bm25Retriever, query: str, k: int, **kw: MetadataFilter) -> list[str]:
    return [r.chunk.chunk_id for r in retriever.retrieve(query, k, **kw)]


def test_is_a_retriever_named_bm25() -> None:
    _, retriever = _setup([make_chunk("a")])
    assert isinstance(retriever, Retriever)
    assert retriever.name == "bm25"


def test_bm25_exact_code_match_ranks_first() -> None:
    chunks = [
        make_chunk("a", text="Ar-Ge projeleri destek programı başvuru koşulları"),
        make_chunk("b", text="1507 programı destek oranı"),
        make_chunk("c", text="destek programı destek oranı destek süresi"),
    ]
    _, retriever = _setup(chunks)
    results = retriever.retrieve("1507", k=3)
    assert [r.chunk.chunk_id for r in results] == ["b"]
    assert results[0].scores == {"bm25": results[0].score}
    assert results[0].score > 0


def test_results_sorted_by_bm25_score() -> None:
    chunks = [
        make_chunk("a", text="destek"),
        make_chunk("b", text="destek destek destek"),
        make_chunk("c", text="destek destek"),
    ]
    _, retriever = _setup(chunks)
    assert _ids(retriever, "destek", k=2) == ["b", "c"]


def test_query_is_normalized() -> None:
    _, retriever = _setup([make_chunk("a", text="destek oranı")])
    assert _ids(retriever, " destek​ ", k=1) == ["a"]


def test_filter_applied_after_lexical() -> None:
    chunks = [
        make_chunk("a", text="destek destek destek", program_kodu="1501"),
        make_chunk("b", text="destek destek", program_kodu="1507"),
        make_chunk("c", text="destek", program_kodu="1507"),
    ]
    index, retriever = _setup(chunks)
    assert _ids(retriever, "destek", k=1, filters=MetadataFilter(program_kodu="1507")) == ["b"]
    assert index.ks == [3]  # over-fetch k * 3 so the filter still leaves k results


def test_filter_fetches_more_until_k_matches() -> None:
    chunks = [make_chunk(f"x{i}", text="destek " * (20 - i), program_kodu="1501") for i in range(8)]
    chunks.append(make_chunk("y", text="destek", program_kodu="1507"))
    index, retriever = _setup(chunks)
    assert _ids(retriever, "destek", k=1, filters=MetadataFilter(program_kodu="1507")) == ["y"]
    assert index.ks == [3, 9]


def test_missing_chunk_in_store_skipped() -> None:
    chunks = [make_chunk("a", text="destek destek"), make_chunk("b", text="destek")]
    _, retriever = _setup(chunks, stored=chunks[1:])
    assert _ids(retriever, "destek", k=2) == ["b"]


def test_no_lexical_match_returns_empty() -> None:
    _, retriever = _setup([make_chunk("a", text="destek")])
    assert retriever.retrieve("hava durumu", k=5) == []


def test_k_zero_returns_empty() -> None:
    _, retriever = _setup([make_chunk("a", text="destek")])
    assert retriever.retrieve("destek", k=0, filters=MetadataFilter(program_kodu="1507")) == []


def test_missing_chunks_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    chunks = [make_chunk("a", text="destek destek"), make_chunk("b", text="destek")]
    _, retriever = _setup(chunks, stored=chunks[1:])
    with caplog.at_level("WARNING"):
        retriever.retrieve("destek", k=2)
    assert "1 parça" in caplog.text and "rag ingest" in caplog.text


def test_no_log_when_every_hit_is_in_the_collection(caplog: pytest.LogCaptureFixture) -> None:
    _, retriever = _setup([make_chunk("a", text="destek")])
    with caplog.at_level("WARNING"):
        retriever.retrieve("destek", k=1)
    assert caplog.text == ""


def test_index_matching_the_collection_passes_the_check() -> None:
    chunks = [make_chunk("a", text="destek"), make_chunk("b", text="hibe")]
    index, retriever = _setup(chunks)
    check_index_matches_collection(index, retriever._store, COLLECTION)


def test_index_out_of_sync_with_collection_is_a_config_error() -> None:
    chunks = [make_chunk("a", text="destek"), make_chunk("b", text="hibe")]
    index, retriever = _setup(chunks, stored=chunks[:1])
    with pytest.raises(ConfigError, match="rag ingest") as caught:
        check_index_matches_collection(index, retriever._store, COLLECTION)
    assert "2" in str(caught.value) and "1" in str(caught.value)


def test_missing_collection_is_a_config_error() -> None:
    index = FakeLexicalIndex()
    index.build([make_chunk("a")])
    with pytest.raises(ConfigError, match="rag ingest"):
        check_index_matches_collection(index, InMemoryVectorStore(), COLLECTION)
