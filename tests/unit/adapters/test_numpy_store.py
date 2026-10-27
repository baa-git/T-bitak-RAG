"""Tests for the read-only NumPy vector store (Hugging Face Space index)."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.builders import make_chunk
from tubitak_rag.adapters.numpy_store import (
    META_FILE,
    NumpyVectorStore,
    ReadOnlyStoreError,
    read_meta,
    write_collection,
)
from tubitak_rag.domain.retrieval import MetadataFilter
from tubitak_rag.indexing.fingerprint import fingerprints
from tubitak_rag.ports import VectorStore

CHUNKS = [
    make_chunk("a:0001:00", doc_id="a", program_kodu="1507", doc_turu="uygulama_esaslari"),
    make_chunk("a:0002:00", doc_id="a", program_kodu="1507", doc_turu="yonetmelik"),
    make_chunk("b:0001:00", doc_id="b", program_kodu="1501", doc_turu="uygulama_esaslari"),
]
VECTORS = [[1.0, 0.0], [0.6, 0.8], [0.0, 2.0]]  # the last one is not normalised


@pytest.fixture
def store(tmp_path: Path) -> NumpyVectorStore:
    write_collection(tmp_path, "c", CHUNKS, VECTORS, embedder_model="m")
    return NumpyVectorStore(tmp_path)


def test_is_a_vector_store(store: NumpyVectorStore) -> None:
    assert isinstance(store, VectorStore)


def test_write_collection_files_and_meta(tmp_path: Path) -> None:
    directory = write_collection(tmp_path, "c", CHUNKS, VECTORS, embedder_model="m")
    vectors = np.load(directory / "vectors.npy")
    assert vectors.dtype == np.float32 and vectors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    lines = (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert read_meta(tmp_path, "c") == {
        "format": 1,
        "collection": "c",
        "embedder_model": "m",
        "dimension": 2,
        "points": 3,
    }


def test_write_collection_refuses_mismatched_or_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aynı uzunlukta"):
        write_collection(tmp_path, "c", CHUNKS, VECTORS[:2], embedder_model="m")
    with pytest.raises(ValueError, match="boş"):
        write_collection(tmp_path, "c", [], [], embedder_model="m")


def test_search_orders_by_cosine_similarity(store: NumpyVectorStore) -> None:
    results = store.search("c", [0.0, 3.0], k=2)
    assert [r.chunk.chunk_id for r in results] == ["b:0001:00", "a:0002:00"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.8) and results[1].scores == {
        "dense": results[1].score
    }


def test_search_honours_filters_and_k(store: NumpyVectorStore) -> None:
    by_program = store.search("c", [0.0, 1.0], k=5, filters=MetadataFilter(program_kodu="1507"))
    assert [r.chunk.chunk_id for r in by_program] == ["a:0002:00", "a:0001:00"]
    both = MetadataFilter(program_kodu="1507", doc_turu="uygulama_esaslari")
    assert [r.chunk.chunk_id for r in store.search("c", [0.0, 1.0], k=5, filters=both)] == [
        "a:0001:00"
    ]
    nothing = MetadataFilter(program_kodu="9999")
    assert store.search("c", [1.0, 0.0], k=5, filters=nothing) == []
    assert store.search("c", [1.0, 0.0], k=0) == []


def test_search_of_missing_collection_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bulunamadı"):
        NumpyVectorStore(tmp_path).search("yok", [1.0, 0.0], k=1)


def test_reading_methods(store: NumpyVectorStore) -> None:
    assert store.count("c") == 3 and store.count("yok") == 0
    assert [c.chunk_id for c in store.get_chunks("c", ["b:0001:00", "x", "a:0001:00"])] == [
        "b:0001:00",
        "a:0001:00",
    ]
    assert store.get_chunks("yok", ["a"]) == []
    assert store.doc_chunk_counts("c") == {"a": 2, "b": 1}
    assert store.doc_chunk_counts("c", ["b", "z"]) == {"b": 1, "z": 0}
    expected = fingerprints({"a": CHUNKS[:2], "b": CHUNKS[2:]})
    assert store.indexed_docs("c") == expected and store.indexed_docs("yok") == {}


def test_inconsistent_export_is_reported(tmp_path: Path) -> None:
    write_collection(tmp_path, "c", CHUNKS, VECTORS, embedder_model="m")
    meta = json.loads((tmp_path / "c" / META_FILE).read_text(encoding="utf-8"))
    (tmp_path / "c" / META_FILE).write_text(json.dumps({**meta, "points": 9}), encoding="utf-8")
    with pytest.raises(ValueError, match="tutarsız"):
        NumpyVectorStore(tmp_path).search("c", [1.0, 0.0], k=1)


def test_writing_is_refused(store: NumpyVectorStore) -> None:
    store.ensure_collection("c", 2)  # existing collection of that dimension: accepted
    with pytest.raises(ReadOnlyStoreError):
        store.ensure_collection("c", 768)
    with pytest.raises(ReadOnlyStoreError):
        store.ensure_collection("yeni", 2)
    with pytest.raises(ReadOnlyStoreError, match="salt okunur"):
        store.upsert("c", CHUNKS[:1], VECTORS[:1])
    with pytest.raises(ReadOnlyStoreError):
        store.delete_doc("c", "a")
    with pytest.raises(ReadOnlyStoreError):
        store.drop_collection("c")
