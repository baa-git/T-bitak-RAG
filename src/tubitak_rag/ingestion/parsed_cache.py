"""Content-addressed cache of parsed documents and batch parsing (FR-X9).

A parsed ``Document`` is stored as ``<doc_id>.<sha256[:12]>.json``. The key contains the
digest of the raw file, so an unchanged file is never parsed twice and a changed file is
parsed again automatically. The parser code is an input as well: each file records
``PARSER_VERSION`` and an entry written by another version is treated as a miss. Bump the
version whenever a change to the parsers alters their output.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from tubitak_rag.domain.document import Document
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.hashing import sha256_of
from tubitak_rag.ingestion.parser_registry import parser_for

ParseStatus = Literal["parsed", "cached", "scanned", "missing", "error"]
# Bump when a parser change alters the parsed output; a unit test compares the parser sources
# with the digest recorded for this version so the bump cannot be forgotten.
PARSER_VERSION = "2"
PARSER_MODULES = (
    "normalize",
    "pdf_text",
    "document_builder",
    "legislation_parser",
    "guide_parser",
    "html_parser",
    "gib_parser",
    "xlsx_parser",
    "parser_registry",
)
_MAX_ID_IN_NAME = 48


def parser_sources_digest(paths: Sequence[Path] | None = None) -> str:
    """Short SHA-256 over the parser source files (line endings normalised)."""
    package = Path(__file__).parent
    files = paths if paths is not None else [package / f"{m}.py" for m in PARSER_MODULES]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


class _CacheEntry(BaseModel):
    parser_version: str
    document: Document


class ParsedCache:
    """Directory of parsed ``Document`` JSON files (``artifacts/cache/parsed``)."""

    def __init__(self, directory: Path, parser_version: str = PARSER_VERSION) -> None:
        self.directory = directory
        self.parser_version = parser_version

    def path_for(self, doc_id: str, sha256: str) -> Path:
        """Cache file of one document version.

        Long ids (programme pages reach ~160 characters) are shortened to a prefix plus a
        hash of the full id, keeping paths below the Windows ``MAX_PATH`` limit.
        """
        name = doc_id
        if len(doc_id) > _MAX_ID_IN_NAME:
            digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8]
            name = f"{doc_id[:_MAX_ID_IN_NAME]}~{digest}"
        return self.directory / f"{name}.{sha256[:12]}.json"

    def get(self, doc_id: str, sha256: str) -> Document | None:
        """Cached document, or None when absent, unreadable or from another parser version."""
        path = self.path_for(doc_id, sha256)
        try:
            entry = _CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return None
        return entry.document if entry.parser_version == self.parser_version else None

    def latest(self, doc_id: str) -> Document | None:
        """Newest cached version of ``doc_id`` without knowing the raw file's digest.

        For readers that have the cache but not the raw files (a Docker stack restored from
        a prebuilt index bundle): the document preview still shows the parsed sections.
        """
        stem = self.path_for(doc_id, "").name.removesuffix("..json")
        candidates = sorted(
            self.directory.glob(f"{stem}.*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True
        )
        for path in candidates:
            try:
                entry = _CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValidationError):
                continue
            if entry.parser_version == self.parser_version and entry.document.doc_id == doc_id:
                return entry.document
        return None

    def put(self, doc: Document) -> None:
        """Store a parsed document."""
        path = self.path_for(doc.doc_id, doc.sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = _CacheEntry(parser_version=self.parser_version, document=doc)
        path.write_text(entry.model_dump_json(), encoding="utf-8")


class ParseOutcome(BaseModel):
    """What happened to one source during parsing (for the ingest report)."""

    doc_id: str
    status: ParseStatus
    sections: int
    message: str = ""


def parse_entry(
    entry: SourceEntry, raw_dir: Path, cache: ParsedCache
) -> tuple[Document | None, ParseOutcome]:
    """Parse one source (or load it from the cache); never raises for a bad file."""
    path = raw_dir / entry.local_path
    if not path.is_file():
        message = f"dosya bulunamadı: {entry.local_path}"
        return None, ParseOutcome(
            doc_id=entry.doc_id, status="missing", sections=0, message=message
        )
    sha256 = sha256_of(path)  # the real file, not the manifest value
    cached = cache.get(entry.doc_id, sha256)
    if cached is not None:
        return cached, ParseOutcome(
            doc_id=entry.doc_id, status="cached", sections=len(cached.sections)
        )
    try:
        doc = parser_for(entry).parse(entry, path)
    except Exception as exc:  # one broken file must not stop the batch
        return None, ParseOutcome(doc_id=entry.doc_id, status="error", sections=0, message=str(exc))
    message = ""
    try:
        cache.put(doc)
    except OSError as exc:  # a full disk or a too-long path only costs the cache
        message = f"önbelleğe yazılamadı: {exc}"
    status: ParseStatus = "scanned" if doc.scanned else "parsed"
    return doc, ParseOutcome(
        doc_id=entry.doc_id, status=status, sections=len(doc.sections), message=message
    )


def parse_all(
    entries: Sequence[SourceEntry], raw_dir: Path, cache: ParsedCache
) -> tuple[list[Document], list[ParseOutcome]]:
    """Parse every entry; documents that could not be produced are only in the outcomes."""
    documents: list[Document] = []
    outcomes: list[ParseOutcome] = []
    for entry in entries:
        doc, outcome = parse_entry(entry, raw_dir, cache)
        outcomes.append(outcome)
        if doc is not None:
            documents.append(doc)
    return documents, outcomes
