"""``rag export-space``: wiring of configs, point source and output (export is faked)."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tubitak_rag.interfaces import space_bundle as sb
from tubitak_rag.interfaces import space_bundle_cli
from tubitak_rag.interfaces.cli import app

runner = CliRunner()


def _configs(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    (configs / "experiments").mkdir(parents=True)
    default = {"vector_store": {"url": "http://qdrant.test:6333"}}
    (configs / "default.yaml").write_text(yaml.safe_dump(default), encoding="utf-8")
    for name in ["v1", "v3"]:
        data = {"extends": "../default.yaml", "experiment": {"name": name}}
        (configs / "experiments" / f"{name}.yaml").write_text(yaml.safe_dump(data), "utf-8")
    return configs / "default.yaml"


class _Export:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def __call__(self, configs: Any, **kwargs: Any) -> sb.SpaceManifest:
        if self.error:
            raise self.error
        self.calls.append({"configs": list(configs), **kwargs})
        out: Path = kwargs["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "dosya.bin").write_bytes(b"x" * 1000)
        return sb.SpaceManifest(
            created_at="2026-10-19T12:00:00+00:00",
            app_version="0.1.0",
            source=kwargs["source_name"],
            collections=[
                sb.SpaceCollection(
                    name="structural_ctx__m",
                    points=7,
                    embedder_model="m",
                    dimension=4,
                    chunker="structural_ctx",
                )
            ],
            bm25=["structural_ctx"],
            parsed_documents=3,
        )


@pytest.fixture
def export(monkeypatch: pytest.MonkeyPatch) -> _Export:
    fake = _Export()
    monkeypatch.setattr(space_bundle_cli, "export_space_index", fake)
    return fake


def test_export_space_from_caches_with_archive(export: _Export, tmp_path: Path) -> None:
    config = _configs(tmp_path)
    out = tmp_path / "release" / "space-index"
    result = runner.invoke(
        app, ["export-space", "-c", str(config), "--source", "cache", "-o", str(out), "--archive"]
    )
    assert result.exit_code == 0, result.output
    (call,) = export.calls
    assert [c.experiment.name for c in call["configs"]] == ["default", "v1", "v3"]
    assert call["source"] is sb.cache_points and call["source_name"] == "cache"
    assert call["out_dir"] == out and call["collections"] is None
    assert "structural_ctx__m: 7 parça" in result.output and "sha256" in result.output
    assert (out.parent / sb.ARCHIVE_NAME).is_file()


def test_export_space_defaults_to_qdrant_of_the_config(
    export: _Export, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def source(url: str) -> Any:
        urls.append(url)
        return lambda cfg: ([], [])

    monkeypatch.setattr(space_bundle_cli, "qdrant_source", source)
    config = _configs(tmp_path)
    result = runner.invoke(
        app, ["export-space", "-c", str(config), "-o", str(tmp_path / "o"), "--collection", "x"]
    )
    assert result.exit_code == 0, result.output
    assert urls == ["http://qdrant.test:6333"]
    assert export.calls[0]["source_name"] == "qdrant" and export.calls[0]["collections"] == ["x"]


def test_export_space_errors_exit_non_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        space_bundle_cli, "export_space_index", _Export(sb.SpaceIndexError("koleksiyon boş"))
    )
    config = _configs(tmp_path)
    failed = runner.invoke(app, ["export-space", "-c", str(config), "--source", "cache"])
    assert failed.exit_code == 1 and "koleksiyon boş" in failed.output
    invalid = tmp_path / "bozuk.yaml"
    invalid.write_text("llm:\n  modelx: a\n", encoding="utf-8")
    bad = runner.invoke(app, ["export-space", "-c", str(invalid)])
    assert bad.exit_code == 2 and "Yapılandırma hatası" in bad.output
