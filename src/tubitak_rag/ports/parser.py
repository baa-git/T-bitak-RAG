"""Document parser port: turns a raw source file into a structured Document."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from tubitak_rag.domain.document import Document
from tubitak_rag.domain.source import SourceEntry


@runtime_checkable
class DocumentParser(Protocol):
    """Parses one source format (PDF, HTML, GIB JSON, XLSX)."""

    def parse(self, entry: SourceEntry, path: Path) -> Document:
        """Parse the file at ``path`` described by ``entry``."""
        ...
