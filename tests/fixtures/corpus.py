"""A tiny corpus on disk (manifests + raw files) and a config pointing at it (ingest tests)."""

import json
from pathlib import Path
from typing import Any

import yaml

from tubitak_rag.config import AppConfig


class CorpusOnDisk:
    """Legislation records in ``manifests/`` and their files in ``raw/tubitak_mevzuat/``.

    The first three characters of a file name become the document id
    (``"001_a.pdf"`` -> ``mevzuat-001``), as in the real legislation manifest.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, str] = {}  # file name -> title, in manifest order

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    def add(self, name: str, text: str) -> None:
        """Write (or overwrite) a raw file and list it in the manifest."""
        path = self.root / "raw" / "tubitak_mevzuat" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        self.register_only(name)

    def register_only(self, name: str) -> None:
        """List a file in the manifest without writing it (a missing download)."""
        self.files[name] = f"Belge {name[:3]}"

    def drop_from_manifest(self, name: str) -> None:
        del self.files[name]

    def config_data(self, **sections: dict[str, Any]) -> dict[str, Any]:
        """Write the manifests; return a config with fake components, merged with ``sections``."""
        manifests = self.root / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        docs = [
            {"baslik": title, "url": f"https://tubitak.gov.tr/{name}", "dosya": name}
            for name, title in self.files.items()
        ]
        payloads = {
            "tubitak_mevzuat.json": {"dokumanlar": docs},
            "tubitak_program_sayfalari.json": {"sayfalar": [], "dosyalar": []},
            "sanayi_mevzuati.json": {"dokumanlar": []},
        }
        for file_name, payload in payloads.items():
            (manifests / file_name).write_text(json.dumps(payload), encoding="utf-8")
        data: dict[str, Any] = {
            "paths": {
                "raw_dir": (self.root / "raw").as_posix(),
                "manifests_dir": manifests.as_posix(),
                "artifacts_dir": self.artifacts.as_posix(),
            },
            "embedder": {"provider": "fake", "model": "fake", "dimension": 8},
            "chunker": {"name": "structural", "context_header": True, "tokenizer": "regex"},
            "vector_store": {"provider": "fake"},
            "bm25": {"provider": "fake"},
        }
        for key, values in sections.items():
            data[key] = {**data.get(key, {}), **values}
        return data

    def config(self, **sections: dict[str, Any]) -> AppConfig:
        return AppConfig.model_validate(self.config_data(**sections))

    def config_file(self, **sections: dict[str, Any]) -> Path:
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(self.config_data(**sections)), encoding="utf-8")
        return path
