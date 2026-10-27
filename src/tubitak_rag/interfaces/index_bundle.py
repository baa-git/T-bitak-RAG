"""Prebuilt index bundle: skip fetch + parse + embed on a fresh Docker stack.

Building the index from scratch takes ~18 minutes of PDF parsing plus the embedding of ~8k
chunks (hours on a CPU). The *result* of that work is small: one Qdrant collection per
chunker/embedding model pair, a BM25 directory per chunker and a little metadata. A bundle
carries exactly that:

* ``qdrant-<collection>.snapshot`` — a Qdrant snapshot (``POST /collections/{c}/snapshots``),
  restored with ``POST /collections/{c}/snapshots/upload?priority=snapshot``;
* ``bm25-<chunker>.tar.gz`` — ``artifacts/cache/bm25/<chunker>``;
* ``meta.tar.gz`` — the parse cache (document preview, Hakkında section count) and
  ``ingest_report.json`` (scanned documents, chunk total);
* ``manifest.json`` — names, point counts, embedding model + dimension and the SHA-256 and
  size of every file.

``export_bundle`` writes a bundle from a running Qdrant (``rag export-index``; creating a
snapshot only reads the collection). ``restore_bundle`` puts it into an empty stack (the
Docker ``init`` service): every file is verified before use, and a collection or directory
that already exists is never overwritten, so a restore is idempotent.
"""

import hashlib
import logging
import shutil
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel

from tubitak_rag import __version__
from tubitak_rag.chunking.cache import chunker_key
from tubitak_rag.config import AppConfig
from tubitak_rag.indexing.naming import bm25_dir, collection_name
from tubitak_rag.ingestion.hashing import sha256_of
from tubitak_rag.ingestion.ingest import REPORT_FILE
from tubitak_rag.ingestion.parsed_cache import PARSER_VERSION

logger = logging.getLogger("tubitak_rag.hazirlik")

BUNDLE_FORMAT = 1
MANIFEST_FILE = "manifest.json"
META_FILE = "meta.tar.gz"
PARSED_DIR = Path("cache") / "parsed"
_CHUNK = 1 << 20
_TIMEOUT = httpx.Timeout(30.0, read=600.0)  # snapshot creation/upload of ~300 MB


class BundleError(RuntimeError):
    """The bundle cannot be written, downloaded, verified or restored (Turkish message)."""


class BundleFile(BaseModel):
    """One file of the bundle with its digest."""

    name: str
    sha256: str
    size: int


class BundleCollection(BaseModel):
    """A Qdrant collection in the bundle."""

    name: str
    points: int
    embedder_model: str
    dimension: int
    chunker: str
    file: str


class BundleBm25(BaseModel):
    """A BM25 index directory in the bundle."""

    chunker: str
    file: str


class BundleManifest(BaseModel):
    """``manifest.json`` of a bundle."""

    format: int = BUNDLE_FORMAT
    created_at: str
    app_version: str
    qdrant_version: str | None = None
    parser_version: str = PARSER_VERSION
    collections: list[BundleCollection]
    bm25: list[BundleBm25] = []
    meta_file: str | None = None
    files: list[BundleFile]

    def file(self, name: str) -> BundleFile:
        """The listed file ``name`` (``BundleError`` if the manifest does not list it)."""
        for item in self.files:
            if item.name == name:
                return item
        raise BundleError(f"manifest.json {name} dosyasını listelemiyor.")


@dataclass(frozen=True)
class CollectionSpec:
    """A collection the experiment configs use: name, chunker key and embedding model."""

    name: str
    chunker: str
    embedder_model: str


def collection_specs(configs: Iterable[AppConfig]) -> list[CollectionSpec]:
    """Distinct collections of ``configs``, in first-seen order."""
    specs: dict[str, CollectionSpec] = {}
    for cfg in configs:
        key = chunker_key(cfg.chunker)
        name = collection_name(key, cfg.embedder.model)
        specs.setdefault(name, CollectionSpec(name, key, cfg.embedder.model))
    return list(specs.values())


# --- Qdrant REST helpers -------------------------------------------------------------------


def _collection_info(client: httpx.Client, qdrant_url: str, name: str) -> dict[str, object] | None:
    response = client.get(f"{qdrant_url}/collections/{name}", timeout=_TIMEOUT)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    result: dict[str, object] = response.json()["result"]
    return result


def qdrant_points(client: httpx.Client, qdrant_url: str, name: str) -> int:
    """Points in collection ``name``; 0 when it does not exist."""
    info = _collection_info(client, qdrant_url, name)
    return int(info.get("points_count") or 0) if info else 0  # type: ignore[call-overload]


def _dimension(info: dict[str, object]) -> int:
    vectors = info["config"]["params"]["vectors"]  # type: ignore[index]
    if isinstance(vectors, dict) and "size" in vectors:
        return int(vectors["size"])
    raise BundleError("Adlandırılmış vektörlü koleksiyonlar desteklenmiyor.")


def _qdrant_version(client: httpx.Client, qdrant_url: str) -> str | None:
    try:
        version = client.get(f"{qdrant_url}/", timeout=_TIMEOUT).json().get("version")
    except (httpx.HTTPError, ValueError):
        return None
    return str(version) if version else None


def _download(client: httpx.Client, url: str, target: Path, label: str) -> str:
    """Stream ``url`` into ``target`` (via ``.part``); return the SHA-256 of the bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with client.stream("GET", url, timeout=_TIMEOUT, follow_redirects=True) as response:
        if response.status_code == 404:
            raise BundleError(f"{label} bulunamadı (404): {url}")
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        done, last_decile = 0, -1
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes(_CHUNK):
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if total >= 50 * _CHUNK and (decile := done * 10 // total) != last_decile:
                    last_decile = decile
                    logger.info(
                        "  %s: %%%d (%.0f / %.0f MB)", label, decile * 10, done / 1e6, total / 1e6
                    )
    partial.replace(target)
    return digest.hexdigest()


def _snapshot(client: httpx.Client, qdrant_url: str, spec: CollectionSpec, target: Path) -> None:
    """Create a snapshot of ``spec.name``, download it to ``target`` and delete it on the server."""
    base = f"{qdrant_url}/collections/{spec.name}/snapshots"
    response = client.post(base, params={"wait": "true"}, timeout=_TIMEOUT)
    response.raise_for_status()
    snapshot = response.json()["result"]
    name, checksum = snapshot["name"], snapshot.get("checksum")
    try:
        digest = _download(client, f"{base}/{name}", target, f"anlık görüntü {spec.name}")
    finally:
        client.delete(f"{base}/{name}", timeout=_TIMEOUT)  # only the snapshot file goes
    if checksum and checksum != digest:
        target.unlink(missing_ok=True)
        raise BundleError(f"{spec.name} anlık görüntüsünün sağlama toplamı tutmuyor.")


def _pack(target: Path, members: Iterable[tuple[Path, str]]) -> None:
    with tarfile.open(target, "w:gz") as tar:
        for path, arcname in members:
            tar.add(path, arcname=arcname)


def _bundle_file(path: Path) -> BundleFile:
    return BundleFile(name=path.name, sha256=sha256_of(path), size=path.stat().st_size)


# --- export --------------------------------------------------------------------------------


def export_bundle(
    specs: Iterable[CollectionSpec],
    *,
    client: httpx.Client,
    qdrant_url: str,
    artifacts_dir: Path,
    out_dir: Path,
) -> BundleManifest:
    """Write a bundle of the existing ``specs`` collections to ``out_dir``.

    Collections missing from Qdrant are skipped with a warning; the collections themselves
    are only read (a snapshot is created, downloaded and its file deleted again).

    Raises:
        BundleError: when none of the collections exists or a download is corrupt.
    """
    qdrant_url = qdrant_url.rstrip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    collections: list[BundleCollection] = []
    for spec in specs:
        info = _collection_info(client, qdrant_url, spec.name)
        if info is None or not info.get("points_count"):
            logger.warning("Koleksiyon yok ya da boş, pakete girmedi: %s", spec.name)
            continue
        target = out_dir / f"qdrant-{spec.name}.snapshot"
        logger.info("Anlık görüntü alınıyor: %s (%s nokta)", spec.name, info["points_count"])
        _snapshot(client, qdrant_url, spec, target)
        collections.append(
            BundleCollection(
                name=spec.name,
                points=int(info["points_count"]),  # type: ignore[call-overload]
                embedder_model=spec.embedder_model,
                dimension=_dimension(info),
                chunker=spec.chunker,
                file=target.name,
            )
        )
    if not collections:
        raise BundleError("Pakete girecek koleksiyon yok: önce `rag ingest` çalıştırın.")

    bm25: list[BundleBm25] = []
    for key in dict.fromkeys(c.chunker for c in collections):
        directory = bm25_dir(artifacts_dir, key)
        if not directory.is_dir():
            logger.warning("BM25 indeksi yok, pakete girmedi: %s", directory)
            continue
        target = out_dir / f"bm25-{key}.tar.gz"
        _pack(target, [(directory, key)])
        bm25.append(BundleBm25(chunker=key, file=target.name))

    meta_members = [
        (artifacts_dir / rel, rel.as_posix())
        for rel in (PARSED_DIR, Path(REPORT_FILE))
        if (artifacts_dir / rel).exists()
    ]
    meta_file = None
    if meta_members:
        _pack(out_dir / META_FILE, meta_members)
        meta_file = META_FILE

    names = (
        [c.file for c in collections] + [b.file for b in bm25] + ([meta_file] if meta_file else [])
    )
    manifest = BundleManifest(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        app_version=__version__,
        qdrant_version=_qdrant_version(client, qdrant_url),
        collections=collections,
        bm25=bm25,
        meta_file=meta_file,
        files=[_bundle_file(out_dir / name) for name in names],
    )
    (out_dir / MANIFEST_FILE).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


# --- sources -------------------------------------------------------------------------------


class BundleSource(Protocol):
    """Where the bundle files come from; ``path`` returns a verified local file."""

    manifest: BundleManifest

    def path(self, item: BundleFile) -> Path: ...

    def cleanup(self) -> None: ...


def _verify(path: Path, item: BundleFile) -> Path:
    if not path.is_file():
        raise BundleError(f"Paket dosyası yok: {path}")
    if path.stat().st_size != item.size or sha256_of(path) != item.sha256:
        raise BundleError(f"Paket dosyası bozuk (SHA-256 tutmuyor): {item.name}")
    return path


def read_manifest(path: Path) -> BundleManifest:
    """Parse ``manifest.json`` (``BundleError`` if missing, invalid or of another format)."""
    try:
        manifest = BundleManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(f"manifest.json okunamadı ({path}): {exc}") from exc
    if manifest.format != BUNDLE_FORMAT:
        raise BundleError(
            f"Paket biçimi {manifest.format} desteklenmiyor (beklenen {BUNDLE_FORMAT})."
        )
    return manifest


@dataclass
class LocalBundle:
    """A bundle directory on disk (``./release`` mounted at ``/app/release``)."""

    directory: Path
    manifest: BundleManifest = field(init=False)

    def __post_init__(self) -> None:
        self.manifest = read_manifest(self.directory / MANIFEST_FILE)

    def path(self, item: BundleFile) -> Path:
        """The verified file ``item``."""
        return _verify(self.directory / item.name, item)

    def cleanup(self) -> None:
        """Nothing to remove: the directory belongs to the user."""


@dataclass
class RemoteBundle:
    """A bundle published under a URL (the Hugging Face dataset); files are fetched on demand.

    Redirects are followed: HF ``resolve/`` URLs answer 302 with a CDN address.
    """

    base_url: str
    client: httpx.Client
    cache_dir: Path
    manifest: BundleManifest

    @classmethod
    def open(cls, url: str, client: httpx.Client, cache_dir: Path) -> "RemoteBundle":
        """Download ``manifest.json`` from ``url`` (a directory URL or the manifest itself)."""
        url = url.strip()
        manifest_url = url if url.endswith(".json") else f"{url.rstrip('/')}/{MANIFEST_FILE}"
        target = cache_dir / MANIFEST_FILE
        logger.info("Hazır indeks paketi okunuyor: %s", manifest_url)
        try:
            _download(client, manifest_url, target, "manifest.json")
        except httpx.HTTPError as exc:
            raise BundleError(f"manifest.json indirilemedi ({manifest_url}): {exc}") from exc
        return cls(manifest_url.rsplit("/", 1)[0], client, cache_dir, read_manifest(target))

    def path(self, item: BundleFile) -> Path:
        """Download ``item`` unless a verified copy is already in the cache directory."""
        target = self.cache_dir / item.name
        if target.is_file():
            try:
                return _verify(target, item)
            except BundleError:
                target.unlink()
        logger.info("İndiriliyor: %s (%.0f MB)", item.name, item.size / 1e6)
        try:
            _download(self.client, f"{self.base_url}/{item.name}", target, item.name)
        except httpx.HTTPError as exc:
            raise BundleError(f"{item.name} indirilemedi: {exc}") from exc
        return _verify(target, item)

    def cleanup(self) -> None:
        """Remove the downloads after a complete restore (the data now lives in the stack)."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)


def locate_bundle(
    local_dir: Path, url: str | None, client: httpx.Client, cache_dir: Path
) -> BundleSource | None:
    """A local bundle (``local_dir/manifest.json``) first, else ``url``, else None."""
    if (local_dir / MANIFEST_FILE).is_file():
        return LocalBundle(local_dir)
    if url and url.strip():
        return RemoteBundle.open(url, client, cache_dir)
    return None


# --- restore -------------------------------------------------------------------------------


@dataclass
class RestoreReport:
    """What a restore did; skipped items already existed."""

    restored_collections: list[str] = field(default_factory=list)
    skipped_collections: list[str] = field(default_factory=list)
    restored_bm25: list[str] = field(default_factory=list)
    meta_files: int = 0


def pending_work(
    manifest: BundleManifest, client: httpx.Client, qdrant_url: str, artifacts_dir: Path
) -> bool:
    """Whether a restore would do anything (an empty collection or a missing BM25 dir)."""
    qdrant_url = qdrant_url.rstrip("/")
    if any(qdrant_points(client, qdrant_url, c.name) == 0 for c in manifest.collections):
        return True
    return any(not bm25_dir(artifacts_dir, b.chunker).is_dir() for b in manifest.bm25)


def _upload(
    client: httpx.Client, qdrant_url: str, collection: BundleCollection, path: Path
) -> None:
    logger.info(
        "Koleksiyon geri yükleniyor: %s (%d parça, %.0f MB)",
        collection.name,
        collection.points,
        path.stat().st_size / 1e6,
    )
    with path.open("rb") as handle:
        response = client.post(
            f"{qdrant_url}/collections/{collection.name}/snapshots/upload",
            params={"priority": "snapshot", "wait": "true"},
            files={"snapshot": (path.name, handle, "application/octet-stream")},
            timeout=_TIMEOUT,
        )
    if not response.is_success:
        raise BundleError(
            f"Qdrant {collection.name} anlık görüntüsünü yükleyemedi "
            f"({response.status_code}): {response.text[:300]}"
        )
    points = qdrant_points(client, qdrant_url, collection.name)
    if points != collection.points:
        raise BundleError(
            f"{collection.name}: geri yüklemeden sonra {points} parça var, "
            f"manifest {collection.points} diyor."
        )


def _extract_missing(archive: Path, destination: Path) -> int:
    """Extract the members of ``archive`` that do not exist under ``destination`` yet."""
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if (destination / member.name).exists():
                continue
            tar.extract(member, destination, filter="data")
            written += member.isfile()
    return written


def restore_bundle(
    source: BundleSource, *, client: httpx.Client, qdrant_url: str, artifacts_dir: Path
) -> RestoreReport:
    """Restore the collections, BM25 directories and metadata of ``source``.

    Existing non-empty collections and existing BM25 directories / metadata files are left
    alone (a local ingest may have rebuilt them). Downloaded files are removed afterwards.

    Raises:
        BundleError: on a corrupt file, a failed upload or a point count mismatch.
    """
    qdrant_url = qdrant_url.rstrip("/")
    manifest, report = source.manifest, RestoreReport()
    for collection in manifest.collections:
        points = qdrant_points(client, qdrant_url, collection.name)
        if points:
            if points != collection.points:
                logger.warning(
                    "Koleksiyon zaten var: %s (%d parça; pakette %d). Dokunulmadı.",
                    collection.name,
                    points,
                    collection.points,
                )
            else:
                logger.info("Koleksiyon zaten yüklü: %s (%d parça)", collection.name, points)
            report.skipped_collections.append(collection.name)
            continue
        path = source.path(manifest.file(collection.file))
        _upload(client, qdrant_url, collection, path)
        report.restored_collections.append(collection.name)

    bm25_root = artifacts_dir / "cache" / "bm25"
    for item in manifest.bm25:
        if bm25_dir(artifacts_dir, item.chunker).is_dir():
            logger.info("BM25 indeksi zaten var: %s", item.chunker)
            continue
        _extract_missing(source.path(manifest.file(item.file)), bm25_root)
        report.restored_bm25.append(item.chunker)

    if manifest.meta_file:
        report.meta_files = _extract_missing(
            source.path(manifest.file(manifest.meta_file)), artifacts_dir
        )
    source.cleanup()
    return report
