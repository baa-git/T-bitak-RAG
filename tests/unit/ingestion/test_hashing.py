"""Tests for file hashing shared by fetch, parsers and the parse cache."""

import hashlib
from pathlib import Path

from tubitak_rag.ingestion.hashing import sha256_of

PARSER_MODULES = [
    "pdf_text",
    "legislation_parser",
    "guide_parser",
    "html_parser",
    "gib_parser",
    "xlsx_parser",
    "parser_registry",
    "parsed_cache",
    "document_builder",
]


def test_sha256_of_reads_the_whole_file(tmp_path: Path) -> None:
    data = b"abc" * 1_000_000  # larger than one read block
    path = tmp_path / "a.bin"
    path.write_bytes(data)
    assert sha256_of(path) == hashlib.sha256(data).hexdigest()


def test_parsers_do_not_depend_on_the_http_module(repo_root: Path) -> None:
    package = repo_root / "src" / "tubitak_rag" / "ingestion"
    for name in PARSER_MODULES:
        source = (package / f"{name}.py").read_text(encoding="utf-8")
        assert "ingestion.fetch" not in source and "httpx" not in source, name
