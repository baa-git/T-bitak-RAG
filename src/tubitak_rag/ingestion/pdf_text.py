"""PDF page text: extraction, scanned-PDF detection and header/footer removal.

pdfplumber reads the characters on each page. Two corpus quirks are handled here:
"fake bold" text printed twice at the same position ("MMaaddddee 11") is removed with
``dedupe_chars``, and legacy Turkish fonts are repaired for the whole document at once.
"""

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pdfplumber
from pydantic import BaseModel, ConfigDict, Field

from tubitak_rag.ingestion.document_builder import join_lines
from tubitak_rag.ingestion.normalize import collapse_ws, normalize_text, repair_legacy_glyphs

PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:-\s*)?\d{1,3}\s*(?:/\s*\d{1,3})?\s*(?:-)?\s*$|^\s*Sayfa\s+\d+", re.IGNORECASE
)
_LINK_LEFTOVER = "tıklayınız"  # "… için tıklayınız" rows of the amendment-history tables
_PAGE_BREAK = "\f"
# Page-bottom footnote of an amended title: 3 “Eski başlık” şeklindeki başlık … değiştirilmiştir.
FOOTNOTE_START_RE = re.compile(r"^\d{1,2}\s+[“\"]")
_AMENDED_RE = re.compile(r"değiştirilmiştir|düzenlenmiştir|eklenmiştir|kaldırılmıştır")
_MAX_FOOTNOTE_LINES = 15


class PageText(BaseModel):
    """Non-empty, normalised text lines of one PDF page (``number`` is 1-based).

    ``footnotes`` holds the page-bottom footnotes removed by ``split_footnotes``.
    """

    model_config = ConfigDict(frozen=True)

    number: int
    lines: list[str]
    footnotes: list[str] = Field(default_factory=list)


def pages_from_texts(texts: Sequence[str]) -> list[PageText]:
    """Turn raw per-page strings into normalised ``PageText`` objects."""
    document = repair_legacy_glyphs(_PAGE_BREAK.join(texts))
    return [
        PageText(number=n, lines=[ln for ln in normalize_text(text).split("\n") if ln])
        for n, text in enumerate(document.split(_PAGE_BREAK), start=1)
    ]


def extract_pages(path: Path) -> list[PageText]:
    """Read every page of a PDF; an image-only page yields ``lines=[]``."""
    with pdfplumber.open(path) as pdf:
        texts = [page.dedupe_chars().extract_text() or "" for page in pdf.pages]
    return pages_from_texts(texts)


def is_scanned(
    pages: Sequence[PageText], *, min_total_chars: int = 200, min_chars_per_page: int = 20
) -> bool:
    """True when the PDF carries (almost) no text layer, i.e. it is a scan (FR-P9)."""
    total = sum(len(line) for page in pages for line in page.lines)
    return total < min_total_chars or total / max(len(pages), 1) < min_chars_per_page


def _edge_indices(page: PageText, edge_lines: int) -> set[int]:
    count = len(page.lines)
    return set(range(min(edge_lines, count))) | set(range(max(count - edge_lines, 0), count))


def _key(line: str) -> str:
    return re.sub(r"\d+", "#", collapse_ws(line)).casefold()


def strip_headers_footers(
    pages: Sequence[PageText], *, edge_lines: int = 2, min_ratio: float = 0.3
) -> list[PageText]:
    """Remove page numbers, repeated running headers/footers and link leftovers (FR-P3).

    Only the first/last ``edge_lines`` lines of a page are candidates. A candidate is dropped
    when it looks like a page number, or when (with at least four pages) the same line — with
    digits ignored — appears at the edge of at least ``min_ratio`` of the pages.
    """
    edges = [_edge_indices(page, edge_lines) for page in pages]
    seen = Counter(
        key
        for page, idx in zip(pages, edges, strict=True)
        for key in {_key(page.lines[i]) for i in idx}
    )
    threshold = min_ratio * len(pages)

    def drop(line: str, at_edge: bool) -> bool:
        if _LINK_LEFTOVER in line.casefold():
            return True
        if not at_edge:
            return False
        if PAGE_NUMBER_RE.search(line):
            return True
        return len(pages) >= 4 and seen[_key(line)] >= threshold

    return [
        PageText(
            number=page.number,
            lines=[ln for i, ln in enumerate(page.lines) if not drop(ln, i in idx)],
        )
        for page, idx in zip(pages, edges, strict=True)
    ]


def _footnote_block_start(lines: Sequence[str]) -> int | None:
    first = max(len(lines) - _MAX_FOOTNOTE_LINES, 0)
    for index in range(first, len(lines)):
        if FOOTNOTE_START_RE.match(lines[index]):
            block = " ".join(lines[index:])
            return index if _AMENDED_RE.search(block) else None
    return None


def split_footnotes(pages: Sequence[PageText]) -> list[PageText]:
    """Move page-bottom amendment footnotes out of ``lines`` into ``footnotes``.

    A footnote block starts with a line such as ``3 “Eski başlık” …`` among the last lines of
    the page, runs to the page end and mentions an amendment ("değiştirilmiştir"). Without
    this, the footnote text lands in the middle of the article that continues on the next page.
    """
    result: list[PageText] = []
    for page in pages:
        start = _footnote_block_start(page.lines)
        if start is None:
            result.append(page)
            continue
        notes: list[list[str]] = []
        for line in page.lines[start:]:
            if FOOTNOTE_START_RE.match(line) or not notes:
                notes.append([line])
            else:
                notes[-1].append(line)
        footnotes = [*page.footnotes, *(join_lines(parts) for parts in notes)]
        result.append(page.model_copy(update={"lines": page.lines[:start], "footnotes": footnotes}))
    return result
