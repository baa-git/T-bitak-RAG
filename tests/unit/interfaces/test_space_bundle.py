"""Space index: rebuilding collections from the ingest caches, writing, packing, downloading."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from tests.builders import make_chunk
from tests.fixtures.corpus import CorpusOnDisk
from tubitak_rag.adapters.numpy_store import NumpyVectorStore
from tubitak_rag.chunking.cache import chunk_settings, chunker_key
from tubitak_rag.config import AppConfig
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.indexing.embedding_cache import EmbeddingCache
from tubitak_rag.indexing.naming import bm25_dir, collection_name
from tubitak_rag.ingestion.ingest import REPORT_FILE
from tubitak_rag.interfaces import space_bundle as sb


class _Artifacts:
    """A config over a tiny corpus plus the caches ``rag ingest`` would have written."""

    def __init__(self, root: Path) -> None:
        self.corpus = CorpusOnDisk(root)
        self.corpus.add("001_a.pdf", "Madde 7")
        self.cfg = self.corpus.config(embedder={"provider": "fake", "model": "m", "dimension": 2})
        self.key = chunker_key(self.cfg.chunker)
        self.chunks = [
            make_chunk("mevzuat-001:0001:00", doc_id="mevzuat-001", text="birinci"),
            make_chunk("mevzuat-001:0002:00", doc_id="mevzuat-001", text="ikinci"),
        ]

    @property
    def artifacts(self) -> Path:
        return self.cfg.paths.artifacts_dir

    def write_chunk_cache(
        self, chunks: list[Chunk], name: str, settings: str | None = None
    ) -> None:
        directory = self.artifacts / "cache" / "chunks" / self.key
        directory.mkdir(parents=True, exist_ok=True)
        entry = {
            "settings": settings or chunk_settings(self.cfg.chunker),
            "document_digest": "x",
            "chunks": [c.model_dump(mode="json") for c in chunks],
        }
        (directory / name).write_text(json.dumps(entry), encoding="utf-8")

    def write_bm25(self, ids: list[str]) -> None:
        directory = bm25_dir(self.artifacts, self.key)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / sb.BM25_IDS_FILE).write_text(json.dumps(ids), encoding="utf-8")
        (directory / "meta.json").write_text("{}", encoding="utf-8")

    def write_vectors(self, chunks: list[Chunk]) -> None:
        cache = EmbeddingCache(self.artifacts / "cache" / "embeddings" / "cache.sqlite")
        vectors = {
            EmbeddingCache.text_hash(c.embed_text): [float(i), 1.0] for i, c in enumerate(chunks)
        }
        cache.put_many("m|2d", vectors)

    def complete(self) -> "_Artifacts":
        self.write_chunk_cache(self.chunks, "mevzuat-001.aaaaaaaaaaaa.json")
        self.write_bm25([c.chunk_id for c in reversed(self.chunks)])
        self.write_vectors(self.chunks)
        parsed = self.artifacts / "cache" / "parsed"
        parsed.mkdir(parents=True, exist_ok=True)
        (parsed / "mevzuat-001.aaaaaaaaaaaa.json").write_text("{}", encoding="utf-8")
        (self.artifacts / REPORT_FILE).write_text('{"chunks_total": 2}', encoding="utf-8")
        return self


@pytest.fixture
def art(tmp_path: Path) -> _Artifacts:
    return _Artifacts(tmp_path)


def test_cache_points_follow_the_bm25_order_with_cached_vectors(art: _Artifacts) -> None:
    art.complete()
    chunks, vectors = sb.cache_points(art.cfg)
    assert [c.chunk_id for c in chunks] == ["mevzuat-001:0002:00", "mevzuat-001:0001:00"]
    assert vectors == [[1.0, 1.0], [0.0, 1.0]]


def test_cache_points_ignore_entries_written_with_other_settings(
    art: _Artifacts,
) -> None:
    art.complete()
    stale = [c.model_copy(update={"text": "eski"}) for c in art.chunks]
    art.write_chunk_cache(stale, "mevzuat-001.bbbbbbbbbbbb.json", settings="v0|eski")
    chunks, _ = sb.cache_points(art.cfg)
    assert {c.text for c in chunks} == {"birinci", "ikinci"}


def test_cache_points_report_missing_chunks_and_vectors(art: _Artifacts) -> None:
    with pytest.raises(sb.SpaceIndexError, match="BM25 parça listesi"):
        sb.cache_points(art.cfg)
    art.write_bm25(["mevzuat-001:0001:00", "yok:0001:00"])
    art.write_chunk_cache(art.chunks, "mevzuat-001.aaaaaaaaaaaa.json")
    with pytest.raises(sb.SpaceIndexError, match="1 parça önbellekte yok"):
        sb.cache_points(art.cfg)
    art.write_bm25(["mevzuat-001:0001:00"])
    with pytest.raises(sb.SpaceIndexError, match="embedding önbelleğinde yok"):
        sb.cache_points(art.cfg)


def _second_config(cfg: AppConfig) -> AppConfig:
    """Another experiment over the same collection (exported once)."""
    return cfg.model_copy(update={"experiment": cfg.experiment.model_copy(update={"name": "v9"})})


def test_export_writes_collections_bm25_meta_and_manifest(art: _Artifacts, tmp_path: Path) -> None:
    art.complete()
    out = tmp_path / "space-index"
    (out / "eski").mkdir(parents=True)  # a previous export is replaced
    manifest = sb.export_space_index(
        [art.cfg, _second_config(art.cfg)],
        source=sb.cache_points,
        source_name="cache",
        out_dir=out,
    )
    name = collection_name(art.key, "m")
    assert [(c.name, c.points, c.dimension, c.chunker) for c in manifest.collections] == [
        (name, 2, 2, art.key)
    ]
    assert manifest.bm25 == [art.key] and manifest.parsed_documents == 1
    assert not (out / "eski").exists()
    assert (bm25_dir(out, art.key) / sb.BM25_IDS_FILE).is_file()
    assert (out / REPORT_FILE).is_file()
    assert (out / "cache" / "parsed" / "mevzuat-001.aaaaaaaaaaaa.json").is_file()
    assert sb.read_space_manifest(out) == manifest
    store = NumpyVectorStore(out / sb.EXPORT_SUBDIR)
    assert store.count(name) == 2
    assert store.search(name, [0.0, 1.0], k=1)[0].chunk.chunk_id == "mevzuat-001:0001:00"


def test_export_can_be_limited_and_refuses_empty_collections(
    art: _Artifacts, tmp_path: Path
) -> None:
    art.complete()
    manifest = sb.export_space_index(
        [art.cfg],
        source=sb.cache_points,
        source_name="cache",
        out_dir=tmp_path / "o",
        collections=[],
    )
    assert manifest.collections == []
    with pytest.raises(sb.SpaceIndexError, match="boş ya da yok"):
        sb.export_space_index(
            [art.cfg], source=lambda cfg: ([], []), source_name="qdrant", out_dir=tmp_path / "p"
        )
    with pytest.raises(sb.SpaceIndexError, match="yapılandırma yok"):
        sb.export_space_index([], source=sb.cache_points, source_name="cache", out_dir=tmp_path)


def test_qdrant_source_reads_points_and_wraps_errors(
    monkeypatch: pytest.MonkeyPatch, art: _Artifacts
) -> None:
    calls: list[str] = []

    class _Store:
        def __init__(self, url: str) -> None:
            calls.append(url)

        def export_points(self, name: str) -> tuple[list[Chunk], list[list[float]]]:
            if name == "bozuk":
                raise ConnectionError("bağlantı yok")
            return [make_chunk()], [[1.0]]

    monkeypatch.setattr("tubitak_rag.adapters.qdrant_store.QdrantVectorStore", _Store)
    source = sb.qdrant_source("http://q:6333")
    assert source(art.cfg)[1] == [[1.0]] and calls == ["http://q:6333"]
    monkeypatch.setattr(sb, "collection_name", lambda key, model: "bozuk")
    with pytest.raises(sb.SpaceIndexError, match="Qdrant'tan okunamadı"):
        source(art.cfg)


# --- pack / download -----------------------------------------------------------------------


def _packed(art: _Artifacts, tmp_path: Path) -> tuple[bytes, str]:
    art.complete()
    out = tmp_path / "space-index"
    sb.export_space_index([art.cfg], source=sb.cache_points, source_name="cache", out_dir=out)
    archive = tmp_path / sb.ARCHIVE_NAME
    digest = sb.pack_space_index(out, archive)
    data = archive.read_bytes()
    assert digest == hashlib.sha256(data).hexdigest()
    return data, digest


def _client(data: bytes, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=data)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_pack_keeps_paths_relative_to_the_index(art: _Artifacts, tmp_path: Path) -> None:
    data, _ = _packed(art, tmp_path)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
    assert sb.SPACE_MANIFEST in names
    assert any(n.startswith("export/") and n.endswith("vectors.npy") for n in names)


def test_ensure_downloads_verifies_and_extracts(art: _Artifacts, tmp_path: Path) -> None:
    data, digest = _packed(art, tmp_path)
    target = tmp_path / "space-artifacts"
    manifest = sb.ensure_space_index(
        target, url="https://example.test/space-index.tar.gz", sha256=digest, client=_client(data)
    )
    assert manifest is not None and manifest.collections[0].points == 2
    assert not list(target.glob("*.part"))
    # present index: no second download (a failing client would raise)
    again = sb.ensure_space_index(target, url="https://x", client=_client(b"", 500))
    assert again == manifest


def test_ensure_without_index_and_url_returns_none(tmp_path: Path) -> None:
    assert sb.ensure_space_index(tmp_path, url=None) is None
    assert sb.ensure_space_index(tmp_path, url="  ") is None


def test_ensure_rejects_bad_digest_failed_download_and_foreign_archive(
    art: _Artifacts, tmp_path: Path
) -> None:
    data, _ = _packed(art, tmp_path)
    with pytest.raises(sb.SpaceIndexError, match="SHA-256"):
        sb.ensure_space_index(
            tmp_path / "a", url="https://x", sha256="0" * 64, client=_client(data)
        )
    with pytest.raises(sb.SpaceIndexError, match="indirilemedi"):
        sb.ensure_space_index(tmp_path / "b", url="https://x", client=_client(b"", 404))
    with pytest.raises(sb.SpaceIndexError, match="açılamadı"):
        sb.ensure_space_index(tmp_path / "c", url="https://x", client=_client(b"not a tar"))
    empty = io.BytesIO()
    with tarfile.open(fileobj=empty, mode="w:gz"):
        pass
    with pytest.raises(sb.SpaceIndexError, match=sb.SPACE_MANIFEST):
        sb.ensure_space_index(tmp_path / "d", url="https://x", client=_client(empty.getvalue()))
    assert not list((tmp_path / "a").glob("*.part"))
