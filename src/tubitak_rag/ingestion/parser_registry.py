"""Choose the parser for a source and parse PDFs (legislation first, guide as fallback).

All parsers return the same ``Document`` type (FR-P1). A new source format is one parser
class plus one line in ``_PARSERS``.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

from tubitak_rag.domain.document import Document
from tubitak_rag.domain.enums import SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion import guide_parser, legislation_parser
from tubitak_rag.ingestion.document_builder import build_document_skeleton
from tubitak_rag.ingestion.gib_parser import GibParser
from tubitak_rag.ingestion.hashing import sha256_of
from tubitak_rag.ingestion.html_parser import HtmlParser
from tubitak_rag.ingestion.pdf_text import (
    PageText,
    extract_pages,
    is_scanned,
    split_footnotes,
    strip_headers_footers,
)
from tubitak_rag.ingestion.xlsx_parser import XlsxParser
from tubitak_rag.ports.parser import DocumentParser

MIN_ARTICLES = 3  # fewer articles than this: the PDF is treated as a guide (FR-C5)

PageExtractor = Callable[[Path], Sequence[PageText]]


class PdfParser:
    """``DocumentParser`` for PDFs: legislation structure, else headings/paragraphs."""

    def __init__(self, extract: PageExtractor = extract_pages) -> None:
        self._extract = extract

    def parse(self, entry: SourceEntry, path: Path) -> Document:
        """Parse a PDF; a scan without a text layer is returned empty with ``scanned=True``."""
        pages = list(self._extract(path))
        doc = build_document_skeleton(entry, sha256_of(path), page_count=len(pages))
        if is_scanned(pages):
            return doc.model_copy(update={"scanned": True})  # FR-P9
        pages = split_footnotes(strip_headers_footers(pages))
        sections = legislation_parser.parse_sections(pages, entry.doc_id)
        structured = len(sections) >= MIN_ARTICLES
        if not structured:
            sections = guide_parser.parse_sections(pages, entry.doc_id, entry.title)
        return doc.model_copy(update={"sections": sections, "structured": structured})


_PARSERS: dict[SourceKind, Callable[[], DocumentParser]] = {
    SourceKind.PDF: PdfParser,
    SourceKind.HTML: HtmlParser,
    SourceKind.GIB_JSON: GibParser,
    SourceKind.XLSX: XlsxParser,
}


def parser_for(entry: SourceEntry) -> DocumentParser:
    """Return a parser for the entry's file format."""
    return _PARSERS[entry.kind]()
