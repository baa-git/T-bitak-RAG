"""Space index: the prebuilt index as plain files, for hosts that cannot run Qdrant.

The Docker bundle (``index_bundle.py``) carries Qdrant snapshots, which only a Qdrant server
can open. A free Hugging Face Space has no Docker-in-Docker and no Qdrant, so ``rag
export-space`` writes the same index as files that the read-only NumPy vector store
(``adapters/numpy_store.py``) searches in memory. The directory mirrors ``artifacts/``, so it
is used by copying it there (or by extracting its archive there)::

    export/<collection>/{vectors.npy, chunks.jsonl, meta.json}   one per collection
    cache/bm25/<chunker>/...                                     BM25 index per chunker
    cache/parsed/<doc>.<sha>.json                                parsed documents (Hakkında)
    ingest_report.json                                           chunk total of the last ingest
    space_manifest.json                                          what the directory holds

Vectors come either from a running Qdrant (authoritative, ``--source qdrant``) or from the
local caches that ``rag ingest`` filled (``--source cache``: chunk cache + embedding cache,
restricted to the chunk ids of the saved BM25 index, i.e. exactly the indexed chunks).
"""

import hashlib
import json
import logging
import shutil
import tarfile
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel

from tubitak_rag import __version__
from tubitak_rag.adapters.numpy_store import write_collection
from tubitak_rag.chunking.cache import chunk_settings, chunker_key
from tubitak_rag.config import AppConfig
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.indexing.embedding_cache import EmbeddingCache
from tubitak_rag.indexing.naming import bm25_dir, collection_name
from tubitak_rag.ingestion.ingest import REPORT_FILE
from tubitak_rag.ingestion.manifest import load_sources
from tubitak_rag.ingestion.parsed_cache import ParsedCache

logger = logging.getLogger("tubitak_rag.hazirlik")

SPACE_FORMAT = 1
SPACE_MANIFEST = "space_manifest.json"
EXPORT_SUBDIR = Path("export")
PARSED_SUBDIR = Path("cache") / "parsed"
BM25_IDS_FILE = "chunk_ids.json"
ARCHIVE_NAME = "space-index.tar.gz"
_CHUNK = 1 << 20
_TIMEOUT = httpx.Timeout(30.0, read=600.0)

Points = tuple[list[Chunk], list[list[float]]]
PointSource = Callable[[AppConfig], Points]


class SpaceIndexError(RuntimeError):
    """The Space index cannot be written, downloaded or verified (Turkish message)."""


class SpaceCollection(BaseModel):
    """One exported collection."""

    name: str
    points: int
    embedder_model: str
    dimension: int
    chunker: str


class SpaceManifest(BaseModel):
    """``space_manifest.json``."""

    format: int = SPACE_FORMAT
    created_at: str
    app_version: str
    source: str  # "qdrant" | "cache"
    collections: list[SpaceCollection]
    bm25: list[str]
    parsed_documents: int


# --- point sources -------------------------------------------------------------------------


def _cache_entry_chunks(path: Path, settings: str) -> list[Chunk]:
    """Chunks of one chunk-cache file written with ``settings`` (others are stale)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if data.get("settings") != settings:
        return []
    return [Chunk.model_validate(item) for item in data.get("chunks", [])]


def _indexed_ids(cfg: AppConfig) -> list[str]:
    path = bm25_dir(cfg.paths.artifacts_dir, chunker_key(cfg.chunker)) / BM25_IDS_FILE
    try:
        ids: list[str] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpaceIndexError(
            f"BM25 parça listesi okunamadı ({path}); önce `rag ingest -c` çalıştırın."
        ) from exc
    return ids


def cache_points(cfg: AppConfig) -> Points:
    """Chunks and vectors of ``cfg``'s collection rebuilt from the local ingest caches.

    The chunk ids of the saved BM25 index are the indexed chunks (ingest rebuilds BM25 from
    exactly the chunks it syncs into the collection); their chunks come from the chunk cache
    and their vectors from the embedding cache. When a stale cache file holds the same id,
    the chunk of the document version listed in the manifests wins.

    Raises:
        SpaceIndexError: when a chunk or a vector is missing from the caches.
    """
    key = chunker_key(cfg.chunker)
    artifacts = cfg.paths.artifacts_dir
    ids = _indexed_ids(cfg)
    wanted = set(ids)
    current_sha = {e.doc_id: e.sha256 for e in load_sources(cfg.paths.manifests_dir) if e.sha256}
    settings = chunk_settings(cfg.chunker)
    found: dict[str, Chunk] = {}
    for path in sorted((artifacts / "cache" / "chunks" / key).glob("*.json")):
        for chunk in _cache_entry_chunks(path, settings):
            if chunk.chunk_id not in wanted:
                continue
            known = found.get(chunk.chunk_id)
            if known is None or chunk.doc_sha256 == current_sha.get(chunk.doc_id):
                found[chunk.chunk_id] = chunk
    missing = [cid for cid in ids if cid not in found]
    if missing:
        raise SpaceIndexError(
            f"{len(missing)} parça önbellekte yok ({key}; ilk: {missing[0]}); "
            "`rag ingest -c` ile önbelleği tamamlayın ya da --source qdrant kullanın."
        )
    chunks = [found[cid] for cid in ids]
    model = f"{cfg.embedder.model}|{cfg.embedder.dimension}d"
    texts = [chunk.embed_text for chunk in chunks]
    cached = EmbeddingCache(artifacts / "cache" / "embeddings" / "cache.sqlite").get_many(
        model, texts
    )
    hashes = [EmbeddingCache.text_hash(text) for text in texts]
    absent = sum(1 for h in hashes if h not in cached)
    if absent:
        raise SpaceIndexError(
            f"{absent} parçanın vektörü embedding önbelleğinde yok ({model}); "
            "--source qdrant kullanın ya da `rag ingest -c` çalıştırın."
        )
    return chunks, [cached[h] for h in hashes]


def qdrant_source(url: str) -> PointSource:
    """A point source reading every point (payload + vector) of the collection from Qdrant."""
    from tubitak_rag.adapters.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore(url)

    def points(cfg: AppConfig) -> Points:
        name = collection_name(chunker_key(cfg.chunker), cfg.embedder.model)
        try:
            return store.export_points(name)
        except Exception as exc:  # qdrant-client wraps connection errors in its own types
            raise SpaceIndexError(f"Qdrant'tan okunamadı ({url}, {name}): {exc}") from exc

    return points


# --- export --------------------------------------------------------------------------------


def _distinct_collections(configs: Iterable[AppConfig]) -> dict[str, AppConfig]:
    by_name: dict[str, AppConfig] = {}
    for cfg in configs:
        by_name.setdefault(collection_name(chunker_key(cfg.chunker), cfg.embedder.model), cfg)
    return by_name


def _copy_parsed(cfg: AppConfig, out_dir: Path) -> int:
    """Copy the parsed document of every manifest entry (its current version only)."""
    source = ParsedCache(cfg.paths.artifacts_dir / PARSED_SUBDIR)
    target = out_dir / PARSED_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for entry in load_sources(cfg.paths.manifests_dir):
        path = source.path_for(entry.doc_id, entry.sha256) if entry.sha256 else None
        if path is None or not path.is_file():
            stem = source.path_for(entry.doc_id, "").name.removesuffix("..json")
            versions = sorted(
                source.directory.glob(f"{stem}.*.json"), key=lambda p: p.stat().st_mtime_ns
            )
            path = versions[-1] if versions else None
        if path is not None:
            shutil.copy2(path, target / path.name)
            copied += 1
    return copied


def export_space_index(
    configs: Sequence[AppConfig],
    *,
    source: PointSource,
    source_name: str,
    out_dir: Path,
    collections: Sequence[str] | None = None,
) -> SpaceManifest:
    """Write the Space index of the distinct collections of ``configs`` to ``out_dir``.

    ``configs[0]`` supplies the paths (artifacts, manifests). ``collections`` limits the
    export to these collection names. An existing ``out_dir`` is replaced.

    Raises:
        SpaceIndexError: when a collection is empty or its BM25 index is missing.
    """
    if not configs:
        raise SpaceIndexError("Dışa aktarılacak yapılandırma yok.")
    base = configs[0]
    selected = _distinct_collections(configs)
    if collections is not None:
        selected = {name: cfg for name, cfg in selected.items() if name in collections}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    written: list[SpaceCollection] = []
    bm25_keys: list[str] = []
    for name, cfg in selected.items():
        chunks, vectors = source(cfg)
        if not chunks:
            raise SpaceIndexError(f"Koleksiyon boş ya da yok: {name}")
        logger.info("Dışa aktarılıyor: %s (%d parça)", name, len(chunks))
        write_collection(
            out_dir / EXPORT_SUBDIR, name, chunks, vectors, embedder_model=cfg.embedder.model
        )
        key = chunker_key(cfg.chunker)
        written.append(
            SpaceCollection(
                name=name,
                points=len(chunks),
                embedder_model=cfg.embedder.model,
                dimension=len(vectors[0]),
                chunker=key,
            )
        )
        directory = bm25_dir(cfg.paths.artifacts_dir, key)
        if key not in bm25_keys and directory.is_dir():
            shutil.copytree(directory, bm25_dir(out_dir, key))
            bm25_keys.append(key)
    report = base.paths.artifacts_dir / REPORT_FILE
    if report.is_file():
        shutil.copy2(report, out_dir / REPORT_FILE)
    manifest = SpaceManifest(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        app_version=__version__,
        source=source_name,
        collections=written,
        bm25=bm25_keys,
        parsed_documents=_copy_parsed(base, out_dir),
    )
    (out_dir / SPACE_MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def pack_space_index(directory: Path, archive: Path) -> str:
    """Write ``directory`` as a ``.tar.gz`` (members relative to it); return its SHA-256."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.resolve() != archive.resolve():
                tar.add(path, arcname=path.relative_to(directory).as_posix())
    return _sha256(archive)


# --- use on the host -----------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read_space_manifest(artifacts_dir: Path) -> SpaceManifest | None:
    """``space_manifest.json`` under ``artifacts_dir``, or None when absent or unreadable."""
    try:
        text = (artifacts_dir / SPACE_MANIFEST).read_text(encoding="utf-8")
        return SpaceManifest.model_validate_json(text)
    except (OSError, ValueError):
        return None


def ensure_space_index(
    artifacts_dir: Path,
    *,
    url: str | None,
    sha256: str | None = None,
    client: httpx.Client | None = None,
) -> SpaceManifest | None:
    """Make sure the Space index is under ``artifacts_dir``; download it from ``url`` if not.

    A present ``space_manifest.json`` wins (index shipped with the Space files). Otherwise
    ``url`` names a ``.tar.gz`` written by ``rag export-space --archive``; it is verified
    against ``sha256`` when given and extracted into ``artifacts_dir``. Returns the manifest,
    or None when there is neither an index nor a URL.

    Raises:
        SpaceIndexError: on a failed download, a digest mismatch or an archive without
            ``space_manifest.json``.
    """
    manifest = read_space_manifest(artifacts_dir)
    if manifest is not None or not (url and url.strip()):
        return manifest
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    archive = artifacts_dir / f"{ARCHIVE_NAME}.part"
    owns_client = client is None
    http = client or httpx.Client(timeout=_TIMEOUT)
    logger.info("Space indeksi indiriliyor: %s", url)
    try:
        with http.stream("GET", url.strip(), follow_redirects=True) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                for block in response.iter_bytes(_CHUNK):
                    handle.write(block)
    except httpx.HTTPError as exc:
        archive.unlink(missing_ok=True)
        raise SpaceIndexError(f"Space indeksi indirilemedi ({url}): {exc}") from exc
    finally:
        if owns_client:
            http.close()
    try:
        if sha256 and _sha256(archive) != sha256.strip().lower():
            raise SpaceIndexError("İndirilen Space indeksinin SHA-256 değeri tutmuyor.")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(artifacts_dir, filter="data")
    except tarfile.TarError as exc:
        raise SpaceIndexError(f"Space indeksi açılamadı: {exc}") from exc
    finally:
        archive.unlink(missing_ok=True)
    manifest = read_space_manifest(artifacts_dir)
    if manifest is None:
        raise SpaceIndexError(f"İndirilen arşivde {SPACE_MANIFEST} yok: {url}")
    return manifest
