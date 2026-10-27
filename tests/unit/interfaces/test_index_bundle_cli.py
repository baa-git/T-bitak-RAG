"""``rag export-index``: wiring of configs, Qdrant URL and output directory (export is faked)."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tubitak_rag.interfaces import index_bundle as ib
from tubitak_rag.interfaces import index_bundle_cli
from tubitak_rag.interfaces.cli import app

runner = CliRunner()


def _configs(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    (configs / "experiments").mkdir(parents=True)
    default = {"vector_store": {"url": "http://qdrant.test:6333"}}
    (configs / "default.yaml").write_text(yaml.safe_dump(default), encoding="utf-8")
    for name, extra in [
        ("v0", {"chunker": {"name": "fixed", "context_header": False}}),
        ("v3", {}),
        ("v3b", {"embedder": {"model": "BAAI/bge-m3", "dimension": 1024}}),
    ]:
        data = {"extends": "../default.yaml", "experiment": {"name": name}, **extra}
        (configs / "experiments" / f"{name}.yaml").write_text(yaml.safe_dump(data), "utf-8")
    return configs / "default.yaml"


class _Export:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def __call__(self, specs: Any, **kwargs: Any) -> ib.BundleManifest:
        if self.error:
            raise self.error
        specs = list(specs)
        self.calls.append({"specs": specs, **kwargs})
        out: Path = kwargs["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "qdrant-x.snapshot").write_bytes(b"x" * 2_000_000)
        return ib.BundleManifest(
            created_at="2026-10-16T12:00:00+00:00",
            app_version="0.1.0",
            collections=[
                ib.BundleCollection(
                    name=s.name,
                    points=10,
                    embedder_model=s.embedder_model,
                    dimension=768,
                    chunker=s.chunker,
                    file="qdrant-x.snapshot",
                )
                for s in specs
            ],
            files=[ib.BundleFile(name="qdrant-x.snapshot", sha256="a" * 64, size=2_000_000)],
        )


def test_export_index_uses_every_experiment_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _Export()
    monkeypatch.setattr(index_bundle_cli, "export_bundle", export)
    out = tmp_path / "release"

    result = runner.invoke(app, ["export-index", "-c", str(_configs(tmp_path)), "--out", str(out)])

    assert result.exit_code == 0, result.output
    (call,) = export.calls
    assert [s.name for s in call["specs"]] == [
        "structural_ctx__google-embeddinggemma-2",  # default config first
        "fixed__google-embeddinggemma-2",
        "structural_ctx__baai-bge-m3",
    ]
    assert call["qdrant_url"] == "http://qdrant.test:6333" and call["out_dir"] == out
    assert "manifest.json" in result.output and "2,0 MB" in result.output


def test_export_index_can_be_limited_to_named_collections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _Export()
    monkeypatch.setattr(index_bundle_cli, "export_bundle", export)
    args = ["export-index", "-c", str(_configs(tmp_path)), "--out", str(tmp_path / "r")]
    result = runner.invoke(app, [*args, "--collection", "structural_ctx__baai-bge-m3"])
    assert result.exit_code == 0, result.output
    assert [s.name for s in export.calls[0]["specs"]] == ["structural_ctx__baai-bge-m3"]

    unknown = runner.invoke(app, [*args, "--collection", "yok"])
    assert unknown.exit_code == 2 and "yok" in unknown.output


def test_export_index_reports_bundle_errors_in_turkish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        index_bundle_cli, "export_bundle", _Export(ib.BundleError("koleksiyon yok"))
    )
    result = runner.invoke(app, ["export-index", "-c", str(_configs(tmp_path))])
    assert result.exit_code == 1 and "koleksiyon yok" in result.output
