"""Line-based document parser for ingest tests (no PDF/HTML libraries involved)."""

from pathlib import Path

from tests.fixtures.documents import make_document, make_section
from tubitak_rag.domain.document import Document
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.hashing import sha256_of


class FakeParser:
    """Each non-empty line of the raw file becomes one paragraph section.

    A file starting with ``TARANMIS`` is a scan (no text layer), one starting with
    ``BOZUK`` raises. :attr:`calls` counts real parses (parse cache tests use it).
    """

    def __init__(self) -> None:
        self.calls = 0

    def parse(self, entry: SourceEntry, path: Path) -> Document:
        self.calls += 1
        text = path.read_text(encoding="utf-8")
        if text.startswith("BOZUK"):
            raise ValueError("bozuk dosya")
        scanned = text.startswith("TARANMIS")
        lines = [] if scanned else [line for line in text.splitlines() if line.strip()]
        sections = [
            make_section(i, SectionKind.PARAGRAF, None, [line], doc_id=entry.doc_id)
            for i, line in enumerate(lines, start=1)
        ]
        return make_document(
            sections,
            doc_id=entry.doc_id,
            title=entry.title,
            program_kodu=None,
            sha256=sha256_of(path),
            scanned=scanned,
        )
