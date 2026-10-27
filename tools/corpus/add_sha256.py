"""One-off: add missing ``sha256`` fields to the manifests from the local corpus files.

The legislation manifest, the programme page list and the law manifest were written without
digests; ``rag fetch`` needs them to verify downloads (FR-D2). Existing values are kept.

Usage (from the repository root)::

    .venv\\Scripts\\python.exe tools/corpus/add_sha256.py
"""

import json
from pathlib import Path
from typing import Any

from tubitak_rag.ingestion.hashing import sha256_of

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "data" / "manifests"
RAW = ROOT / "data" / "raw"

# manifest file -> (list key, field holding the relative file name, raw sub-directory)
TARGETS = {
    "tubitak_mevzuat.json": ("dokumanlar", "dosya", "tubitak_mevzuat"),
    "tubitak_program_sayfalari.json": ("sayfalar", "html_dosya", "tubitak_program_sayfalari"),
    "sanayi_mevzuati.json": ("dokumanlar", "dosya", "sanayi_mevzuati"),
}


def add_digests(manifest: Path, list_key: str, file_key: str, raw_dir: Path) -> int:
    """Fill ``sha256`` for entries that lack it; return how many were written."""
    original = manifest.read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(original)
    written = 0
    for entry in data[list_key]:
        if entry.get("sha256"):
            continue
        entry["sha256"] = sha256_of(raw_dir / entry[file_key])
        written += 1
    if written:
        # Keep the file's own indentation and trailing newline so the diff shows only digests.
        second_line = original.splitlines()[1]
        indent = len(second_line) - len(second_line.lstrip(" "))
        text = json.dumps(data, ensure_ascii=False, indent=indent)
        text += "\n" if original.endswith("\n") else ""
        manifest.write_text(text, encoding="utf-8", newline="\n")
    return written


def main() -> None:
    """Update all three manifests in place and print the counts."""
    counts = [
        add_digests(MANIFESTS / name, list_key, file_key, RAW / sub)
        for name, (list_key, file_key, sub) in TARGETS.items()
    ]
    print(" + ".join(str(c) for c in counts), "sha256 yazıldı")


if __name__ == "__main__":
    main()
