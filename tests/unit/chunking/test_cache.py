"""Tests for the chunk cache keyed by document and chunker settings (FR-X9)."""

from pathlib import Path
from typing import Literal

import pytest

from tests.fixtures.documents import make_guide_doc, make_legislation_doc
from tubitak_rag.chunking.cache import ChunkCache, chunk_documents, chunk_settings, chunker_key
from tubitak_rag.chunking.structural import StructuralChunker
from tubitak_rag.chunking.tokens import RegexTokenCounter
from tubitak_rag.config import ChunkerConfig, FixedChunkConfig
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document


class _CountingChunker:
    name = "structural"

    def __init__(self) -> None:
        self.calls = 0
        self._inner = StructuralChunker(RegexTokenCounter())

    def chunk(self, doc: Document) -> list[Chunk]:
        self.calls += 1
        return self._inner.chunk(doc)


def _chunks(doc: Document) -> list[Chunk]:
    return StructuralChunker(RegexTokenCounter()).chunk(doc)


def test_roundtrip(tmp_path: Path) -> None:
    doc = make_legislation_doc()
    cache = ChunkCache(tmp_path, "structural_ctx", settings="s1")
    assert cache.get(doc) is None
    cache.put(doc, _chunks(doc))
    assert cache.get(doc) == _chunks(doc)
    assert cache.path_for(doc) == tmp_path / "structural_ctx" / f"mevzuat-031.{'a' * 12}.json"
    assert cache.path_for(doc).is_file()


def test_miss_when_settings_differ(tmp_path: Path) -> None:
    doc = make_legislation_doc()
    ChunkCache(tmp_path, "structural_ctx", settings="max_tokens=450").put(doc, _chunks(doc))
    assert ChunkCache(tmp_path, "structural_ctx", settings="max_tokens=300").get(doc) is None


def test_miss_when_parsed_document_changed_with_same_file(tmp_path: Path) -> None:
    doc = make_legislation_doc()
    cache = ChunkCache(tmp_path, "structural_ctx", settings="s1")
    cache.put(doc, _chunks(doc))
    reparsed = doc.model_copy(update={"sections": doc.sections[:2]})  # e.g. a parser fix
    assert reparsed.sha256 == doc.sha256
    assert cache.get(reparsed) is None


def test_miss_when_file_is_corrupt(tmp_path: Path) -> None:
    doc = make_legislation_doc()
    cache = ChunkCache(tmp_path, "structural_ctx", settings="s1")
    cache.path_for(doc).parent.mkdir(parents=True)
    cache.path_for(doc).write_text("{bozuk", encoding="utf-8")
    assert cache.get(doc) is None


def test_chunk_documents_uses_cache(tmp_path: Path) -> None:
    docs = [make_legislation_doc(), make_guide_doc()]
    cache = ChunkCache(tmp_path, "structural_ctx", settings="s1")
    chunker = _CountingChunker()
    first = chunk_documents(docs, chunker, cache)
    assert chunker.calls == 2 and list(first) == ["mevzuat-031", "dosya-abc"]
    second = chunk_documents(docs, chunker, cache)
    assert chunker.calls == 2  # second run: every document comes from the cache
    assert second == first


def test_chunk_documents_without_cache() -> None:
    chunker = _CountingChunker()
    chunk_documents([make_guide_doc()], chunker, None)
    chunk_documents([make_guide_doc()], chunker, None)
    assert chunker.calls == 2


@pytest.mark.parametrize(
    ("name", "header", "key"),
    [
        ("structural", True, "structural_ctx"),
        ("structural", False, "structural"),
        ("fixed", True, "fixed_ctx"),
        ("fixed", False, "fixed"),
    ],
)
def test_chunker_key_variants(name: Literal["fixed", "structural"], header: bool, key: str) -> None:
    assert chunker_key(ChunkerConfig(name=name, context_header=header)) == key


def test_chunk_settings_change_with_parameters() -> None:
    base = ChunkerConfig()
    variants = {
        chunk_settings(base),
        chunk_settings(ChunkerConfig(max_tokens=300)),
        chunk_settings(ChunkerConfig(tokenizer="regex")),
        chunk_settings(ChunkerConfig(name="fixed")),
        chunk_settings(ChunkerConfig(name="fixed", fixed=FixedChunkConfig(size=256, overlap=32))),
    }
    assert len(variants) == 5
    # parameters of the other strategy do not matter
    assert chunk_settings(ChunkerConfig(fixed=FixedChunkConfig(size=256))) == chunk_settings(base)


def test_failed_write_keeps_previous_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = make_legislation_doc()
    cache = ChunkCache(tmp_path, "structural_ctx", settings="s1")
    cache.put(doc, _chunks(doc))

    def _fail(src: object, dst: object) -> None:
        raise OSError("disk dolu")

    monkeypatch.setattr("tubitak_rag.chunking.cache.os.replace", _fail)
    with pytest.raises(OSError):
        cache.put(doc, _chunks(doc)[:1])
    assert cache.get(doc) == _chunks(doc)  # the old file was never half-overwritten
    assert [p.name for p in cache.path_for(doc).parent.iterdir()] == [cache.path_for(doc).name]
