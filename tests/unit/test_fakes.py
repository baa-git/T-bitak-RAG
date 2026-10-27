"""Tests for the test doubles in tests/fakes (they must behave predictably for every test)."""

import math
from pathlib import Path

import pytest

from tests.builders import make_chunk, make_scored
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.lexical_index import FakeLexicalIndex
from tests.fakes.llm import FakeLLM
from tests.fakes.reranker import FakeReranker
from tests.fakes.tokens import RegexTokenCounter
from tests.fakes.vector_store import InMemoryVectorStore
from tubitak_rag.domain.retrieval import MetadataFilter
from tubitak_rag.indexing.fingerprint import fingerprints

# --- FakeEmbedder -------------------------------------------------------------------------


def test_fake_embedder_is_deterministic() -> None:
    assert FakeEmbedder().embed_query("a") == FakeEmbedder().embed_query("a")


def test_fake_embedder_vectors_are_unit_length_and_distinct() -> None:
    emb = FakeEmbedder()
    first, second = emb.embed_documents(["a", "b"])
    assert len(first) == emb.dimension == 8
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0)
    assert first != second


def test_fake_embedder_counts_document_calls() -> None:
    emb = FakeEmbedder()
    emb.embed_documents(["a"])
    emb.embed_documents(["b", "c"])
    emb.embed_query("q")
    assert emb.calls == 2


# --- InMemoryVectorStore ------------------------------------------------------------------


def _store_with_two_programs() -> tuple[InMemoryVectorStore, FakeEmbedder]:
    emb = FakeEmbedder()
    store = InMemoryVectorStore()
    store.ensure_collection("c", emb.dimension)
    chunks = [
        make_chunk("a:1", doc_id="a", program_kodu="1507", doc_turu="uygulama_esaslari"),
        make_chunk("b:1", doc_id="b", program_kodu="1501", doc_turu="cagri", doc_sha256="f" * 64),
    ]
    store.upsert("c", chunks, emb.embed_documents([c.embed_text for c in chunks]))
    return store, emb


def test_inmemory_store_filters_by_program() -> None:
    store, emb = _store_with_two_programs()
    hits = store.search("c", emb.embed_query("x"), k=5, filters=MetadataFilter(program_kodu="1501"))
    assert [h.chunk.chunk_id for h in hits] == ["b:1"]


def test_inmemory_store_filters_by_doc_type() -> None:
    store, emb = _store_with_two_programs()
    flt = MetadataFilter(doc_turu="uygulama_esaslari")
    hits = store.search("c", emb.embed_query("x"), k=5, filters=flt)
    assert [h.chunk.chunk_id for h in hits] == ["a:1"]


def test_inmemory_store_ranks_by_cosine_and_records_dense_score() -> None:
    emb = FakeEmbedder()
    store = InMemoryVectorStore()
    store.ensure_collection("c", emb.dimension)
    chunks = [make_chunk("x", embed_text="hedef"), make_chunk("y", embed_text="başka")]
    store.upsert("c", chunks, emb.embed_documents(["hedef", "başka"]))
    hits = store.search("c", emb.embed_query("hedef"), k=1)
    assert [h.chunk.chunk_id for h in hits] == ["x"]
    assert math.isclose(hits[0].score, 1.0) and hits[0].scores == {"dense": hits[0].score}


def test_inmemory_store_bookkeeping() -> None:
    store, _ = _store_with_two_programs()
    assert store.count("c") == 2
    # doc_id -> fingerprint of the stored chunks (not the raw file hash; see indexing.fingerprint)
    stored = {doc: store.get_chunks("c", [f"{doc}:1"]) for doc in ("a", "b")}
    assert store.indexed_docs("c") == fingerprints(stored)
    assert [c.chunk_id for c in store.get_chunks("c", ["b:1", "missing", "a:1"])] == ["b:1", "a:1"]
    store.delete_doc("c", "a")
    assert store.count("c") == 1 and set(store.indexed_docs("c")) == {"b"}


def test_inmemory_store_indexed_docs_of_missing_collection_is_empty() -> None:
    assert InMemoryVectorStore().indexed_docs("yok") == {}


def test_inmemory_store_drop_collection_removes_everything() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection("c", 8)
    store.upsert("c", [make_chunk()], [[1.0] * 8])
    store.drop_collection("c")
    store.drop_collection("yok")  # missing collections are ignored
    assert store.indexed_docs("c") == {} and "c" not in store.collections


def test_inmemory_store_count_of_missing_collection_is_zero() -> None:
    assert InMemoryVectorStore().count("yok") == 0


def test_inmemory_store_rejects_mismatched_lengths() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection("c", 8)
    with pytest.raises(ValueError):
        store.upsert("c", [make_chunk()], [])


# --- FakeLLM ------------------------------------------------------------------------------


def test_fake_llm_returns_responses_in_order_and_records_calls() -> None:
    llm = FakeLLM(["bir", "iki"])
    schema = {"type": "object"}
    assert llm.generate("s", "u1", json_schema=schema) == "bir"
    assert llm.generate("s", "u2") == "iki"
    assert llm.calls == [
        {"system": "s", "user": "u1", "json_schema": schema},
        {"system": "s", "user": "u2", "json_schema": None},
    ]


def test_fake_llm_fails_loudly_when_out_of_responses() -> None:
    llm = FakeLLM([])
    with pytest.raises(RuntimeError):
        llm.generate("s", "u")


# --- FakeReranker -------------------------------------------------------------------------


def test_fake_reranker_scores_by_query_word_overlap() -> None:
    low = make_scored(make_chunk("low", text="ilgisiz metin"), 0.9, dense=0.9)
    high = make_scored(make_chunk("high", text="destek oranı yüzde destek"), 0.1, dense=0.1)
    ranked = FakeReranker().rerank("destek oranı", [low, high], top_n=1)
    assert [r.chunk.chunk_id for r in ranked] == ["high"]
    assert ranked[0].score == 3.0
    assert ranked[0].scores == {"dense": 0.1, "rerank": 3.0}


# --- FakeLexicalIndex ---------------------------------------------------------------------


def test_fake_lexical_index_counts_terms_and_round_trips(tmp_path: Path) -> None:
    index = FakeLexicalIndex()
    index.build(
        [
            make_chunk("a", embed_text="1507 destek 1507"),
            make_chunk("b", embed_text="1501 destek"),
            make_chunk("c", embed_text="ilgisiz"),
        ]
    )
    assert len(index) == 3
    assert index.search("1507 destek", k=5) == [("a", 3.0), ("b", 1.0)]
    index.save(tmp_path)
    reloaded = FakeLexicalIndex()
    reloaded.load(tmp_path)
    assert reloaded.search("1501", k=1) == [("b", 1.0)] and len(reloaded) == 3


# --- RegexTokenCounter --------------------------------------------------------------------


def test_regex_token_counter_counts_words_and_punctuation() -> None:
    counter = RegexTokenCounter()
    assert counter.count("Madde 7 – destek.") == 5
    ids = counter.encode("destek oranı")
    assert len(ids) == 2 and counter.decode(ids) == "destek oranı"
