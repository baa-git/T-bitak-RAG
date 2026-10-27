"""Prebuilt index bundle: export from Qdrant + artifacts, restore into an empty stack.

Qdrant's snapshot REST API and a release download server are ``httpx.MockTransport`` fakes;
nothing touches the network.
"""

import hashlib
import json
import re
import tarfile
from pathlib import Path

import httpx
import pytest

from tubitak_rag.config import AppConfig, ChunkerConfig, EmbedderConfig
from tubitak_rag.interfaces import index_bundle as ib

QDRANT = "http://qdrant:6333"
GEMMA_STRUCT = "structural_ctx__google-embeddinggemma-2"
GEMMA_FIXED = "fixed__google-embeddinggemma-2"
BGE_STRUCT = "structural_ctx__baai-bge-m3"


class FakeQdrant:
    """Collections, snapshot create/download/delete and snapshot upload (restore)."""

    def __init__(self, collections: dict[str, tuple[int, int]] | None = None) -> None:
        self.collections = dict(collections or {})  # name -> (points, dimension)
        self.snapshots: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.uploads: list[str] = []
        self.bad_checksum = False

    @staticmethod
    def content(name: str, points: int) -> bytes:
        return f"SNAP|{name}|{points}|".encode() + b"\x00" * 64

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path == "/":
            return httpx.Response(200, json={"title": "qdrant", "version": "1.19.0"})
        match = re.fullmatch(r"/collections/([^/]+)(/snapshots(?:/upload|/[^/]+)?)?", path)
        if match is None:
            return httpx.Response(500)
        name, rest = match.group(1), match.group(2) or ""
        if rest == "" and method == "GET":
            if name not in self.collections:
                return httpx.Response(404, json={"status": {"error": "Not found"}})
            points, dim = self.collections[name]
            result = {
                "points_count": points,
                "config": {"params": {"vectors": {"size": dim, "distance": "Cosine"}}},
            }
            return httpx.Response(200, json={"result": result})
        if rest == "/snapshots" and method == "POST":
            points, _ = self.collections[name]
            data = self.content(name, points)
            snap = f"{name}-1.snapshot"
            self.snapshots[snap] = data
            checksum = "0" * 64 if self.bad_checksum else hashlib.sha256(data).hexdigest()
            result = {"name": snap, "size": len(data), "checksum": checksum}
            return httpx.Response(200, json={"result": result})
        if rest == "/snapshots/upload" and method == "POST":
            assert request.url.params["priority"] == "snapshot"
            body = request.content
            start = body.index(b"SNAP|")
            _, snap_name, points, _ = body[start : start + 200].split(b"|", 3)
            assert snap_name.decode() == name
            self.uploads.append(name)
            self.collections[name] = (int(points), 768)
            return httpx.Response(200, json={"result": True})
        snap = rest.removeprefix("/snapshots/")
        if method == "GET":
            return httpx.Response(200, content=self.snapshots[snap])
        if method == "DELETE":
            self.deleted.append(snap)
            return httpx.Response(200, json={"result": True})
        return httpx.Response(500)


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def _configs() -> list[AppConfig]:
    return [
        AppConfig(chunker=ChunkerConfig(name="fixed", context_header=False)),
        AppConfig(),
        AppConfig(),  # same collection twice (v2 and v3)
        AppConfig(embedder=EmbedderConfig(model="BAAI/bge-m3", dimension=1024)),
    ]


def _artifacts(root: Path) -> Path:
    artifacts = root / "artifacts"
    for key in ("fixed", "structural_ctx"):
        bm25 = artifacts / "cache" / "bm25" / key
        (bm25 / "bm25s").mkdir(parents=True)
        (bm25 / "meta.json").write_text(json.dumps({"n": 3}), encoding="utf-8")
        (bm25 / "bm25s" / "data.npz").write_bytes(b"bm25-" + key.encode())
    parsed = artifacts / "cache" / "parsed"
    parsed.mkdir(parents=True)
    (parsed / "mevzuat-001.abc.json").write_text('{"document": {}}', encoding="utf-8")
    (artifacts / "ingest_report.json").write_text('{"chunks_total": 7017}', encoding="utf-8")
    (artifacts / "cache" / "embeddings").mkdir(parents=True)
    (artifacts / "cache" / "embeddings" / "cache.sqlite").write_bytes(b"not exported")
    return artifacts


def _export(tmp_path: Path, qdrant: FakeQdrant) -> tuple[ib.BundleManifest, Path]:
    out = tmp_path / "release"
    manifest = ib.export_bundle(
        ib.collection_specs(_configs()),
        client=_client(qdrant),
        qdrant_url=QDRANT,
        artifacts_dir=_artifacts(tmp_path / "host"),
        out_dir=out,
    )
    return manifest, out


@pytest.fixture
def host() -> FakeQdrant:
    return FakeQdrant(
        {GEMMA_FIXED: (4193, 768), GEMMA_STRUCT: (7017, 768), BGE_STRUCT: (7017, 1024)}
    )


# --- export --------------------------------------------------------------------------------


def test_collection_specs_are_unique_and_carry_model_and_chunker() -> None:
    specs = ib.collection_specs(_configs())
    assert [s.name for s in specs] == [GEMMA_FIXED, GEMMA_STRUCT, BGE_STRUCT]
    assert specs[2].embedder_model == "BAAI/bge-m3" and specs[2].chunker == "structural_ctx"


def test_export_writes_snapshots_bm25_meta_and_manifest(tmp_path: Path, host: FakeQdrant) -> None:
    manifest, out = _export(tmp_path, host)

    assert [(c.name, c.points, c.dimension) for c in manifest.collections] == [
        (GEMMA_FIXED, 4193, 768),
        (GEMMA_STRUCT, 7017, 768),
        (BGE_STRUCT, 7017, 1024),
    ]
    assert manifest.collections[1].embedder_model == "google/embeddinggemma-2"
    assert [b.chunker for b in manifest.bm25] == ["fixed", "structural_ctx"]
    assert manifest.qdrant_version == "1.19.0" and manifest.created_at
    on_disk = json.loads((out / ib.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert ib.BundleManifest.model_validate(on_disk) == manifest
    for item in manifest.files:  # every file listed with its real digest and size
        data = (out / item.name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item.sha256 and len(data) == item.size
    assert sorted(host.deleted) == sorted(host.snapshots)  # server-side snapshots cleaned up
    with tarfile.open(out / manifest.meta_file) as tar:  # type: ignore[operator]
        names = set(tar.getnames())
    assert "ingest_report.json" in names and "cache/parsed/mevzuat-001.abc.json" in names
    assert not any("embeddings" in n for n in names)


def test_export_skips_missing_collections_and_bm25(tmp_path: Path) -> None:
    qdrant = FakeQdrant({GEMMA_STRUCT: (7017, 768)})
    manifest, _ = _export(tmp_path, qdrant)
    assert [c.name for c in manifest.collections] == [GEMMA_STRUCT]
    assert [b.chunker for b in manifest.bm25] == ["structural_ctx"]


def test_export_without_any_collection_fails(tmp_path: Path) -> None:
    with pytest.raises(ib.BundleError, match="koleksiyon"):
        _export(tmp_path, FakeQdrant({}))


def test_export_rejects_a_corrupt_snapshot_download(tmp_path: Path, host: FakeQdrant) -> None:
    host.bad_checksum = True
    with pytest.raises(ib.BundleError, match="sağlama"):
        _export(tmp_path, host)


# --- restore -------------------------------------------------------------------------------


def _restore(source: ib.BundleSource, qdrant: FakeQdrant, artifacts: Path) -> ib.RestoreReport:
    return ib.restore_bundle(
        source, client=_client(qdrant), qdrant_url=QDRANT, artifacts_dir=artifacts
    )


def test_restore_into_empty_stack(tmp_path: Path, host: FakeQdrant) -> None:
    _, out = _export(tmp_path, host)
    target, artifacts = FakeQdrant(), tmp_path / "container"

    report = _restore(ib.LocalBundle(out), target, artifacts)

    assert target.uploads == [GEMMA_FIXED, GEMMA_STRUCT, BGE_STRUCT]
    assert target.collections[GEMMA_STRUCT][0] == 7017
    assert report.restored_collections == [GEMMA_FIXED, GEMMA_STRUCT, BGE_STRUCT]
    data = artifacts / "cache" / "bm25" / "structural_ctx" / "bm25s" / "data.npz"
    assert data.read_bytes() == b"bm25-structural_ctx"
    assert (artifacts / "ingest_report.json").is_file()
    assert (artifacts / "cache" / "parsed" / "mevzuat-001.abc.json").is_file()
    assert report.restored_bm25 == ["fixed", "structural_ctx"]


def test_restore_is_idempotent_and_never_overwrites(tmp_path: Path, host: FakeQdrant) -> None:
    _, out = _export(tmp_path, host)
    target, artifacts = FakeQdrant(), tmp_path / "container"
    _restore(ib.LocalBundle(out), target, artifacts)
    target.uploads.clear()
    target.collections[BGE_STRUCT] = (12, 1024)  # rebuilt locally since: left alone
    report_file = artifacts / "ingest_report.json"
    report_file.write_text("{}", encoding="utf-8")

    report = _restore(ib.LocalBundle(out), target, artifacts)

    assert target.uploads == [] and report.restored_collections == []
    assert set(report.skipped_collections) == {GEMMA_FIXED, GEMMA_STRUCT, BGE_STRUCT}
    assert report_file.read_text(encoding="utf-8") == "{}"
    assert report.restored_bm25 == []


def test_restore_rejects_a_tampered_file(tmp_path: Path, host: FakeQdrant) -> None:
    manifest, out = _export(tmp_path, host)
    (out / manifest.collections[0].file).write_bytes(b"bozuk")
    with pytest.raises(ib.BundleError, match="SHA-256"):
        _restore(ib.LocalBundle(out), FakeQdrant(), tmp_path / "container")


def test_restore_reports_a_point_count_mismatch(tmp_path: Path, host: FakeQdrant) -> None:
    manifest, out = _export(tmp_path, host)
    data = json.loads((out / ib.MANIFEST_FILE).read_text(encoding="utf-8"))
    data["collections"][0]["points"] = 99
    (out / ib.MANIFEST_FILE).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ib.BundleError, match="99"):
        _restore(ib.LocalBundle(out), FakeQdrant(), tmp_path / "container")


class ReleaseServer:
    """Serves the exported files under https://example.test/rel/ (with a redirect)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "qdrant":
            raise AssertionError("routed elsewhere")
        self.requests.append(request.url.path)
        if request.url.path.startswith("/rel/"):  # GitHub answers 302 to a storage host
            target = "https://storage.example.test/" + request.url.path.removeprefix("/rel/")
            return httpx.Response(302, headers={"location": target})
        path = self.directory / request.url.path.removeprefix("/")
        return (
            httpx.Response(200, content=path.read_bytes())
            if path.is_file()
            else httpx.Response(404)
        )


def test_remote_bundle_downloads_verifies_and_restores(tmp_path: Path, host: FakeQdrant) -> None:
    _, out = _export(tmp_path, host)
    server, target = ReleaseServer(out), FakeQdrant()

    def route(request: httpx.Request) -> httpx.Response:
        return target(request) if request.url.host == "qdrant" else server(request)

    client = _client(route)
    cache = tmp_path / "container" / "release-download"
    source = ib.RemoteBundle.open("https://example.test/rel/", client, cache)
    report = ib.restore_bundle(
        source, client=client, qdrant_url=QDRANT, artifacts_dir=tmp_path / "container"
    )

    assert report.restored_collections == [GEMMA_FIXED, GEMMA_STRUCT, BGE_STRUCT]
    assert "/rel/manifest.json" in server.requests
    assert not any(cache.glob("*.snapshot"))  # downloads are removed after a full restore


def test_remote_bundle_url_may_name_the_manifest(tmp_path: Path, host: FakeQdrant) -> None:
    _, out = _export(tmp_path, host)
    source = ib.RemoteBundle.open(
        "https://example.test/rel/manifest.json", _client(ReleaseServer(out)), tmp_path / "c"
    )
    assert [c.name for c in source.manifest.collections][0] == GEMMA_FIXED


def test_remote_bundle_missing_manifest_is_a_bundle_error(tmp_path: Path) -> None:
    with pytest.raises(ib.BundleError, match="manifest"):
        ib.RemoteBundle.open(
            "https://example.test/rel/", _client(ReleaseServer(tmp_path)), tmp_path
        )


def test_locate_prefers_local_bundle_then_url(tmp_path: Path, host: FakeQdrant) -> None:
    _, out = _export(tmp_path, host)
    client = _client(ReleaseServer(out))
    local = ib.locate_bundle(out, None, client, tmp_path / "dl")
    assert isinstance(local, ib.LocalBundle)
    remote = ib.locate_bundle(
        tmp_path / "yok", "https://example.test/rel/", client, tmp_path / "dl"
    )
    assert isinstance(remote, ib.RemoteBundle)
    assert ib.locate_bundle(tmp_path / "yok", None, client, tmp_path / "dl") is None
    assert ib.locate_bundle(tmp_path / "yok", "  ", client, tmp_path / "dl") is None


def test_pending_work_tells_whether_anything_must_be_restored(
    tmp_path: Path, host: FakeQdrant
) -> None:
    _, out = _export(tmp_path, host)
    source, artifacts, target = ib.LocalBundle(out), tmp_path / "container", FakeQdrant()
    client = _client(target)
    assert ib.pending_work(source.manifest, client, QDRANT, artifacts)
    _restore(source, target, artifacts)
    assert not ib.pending_work(source.manifest, client, QDRANT, artifacts)
