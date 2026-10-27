"""Parser for laws downloaded from the GİB legislation API (5746 and 4691 sayılı Kanun).

The JSON holds one item per article: ``title`` ("Madde 3 İndirim, istisna …", "Geçici
Madde 2", "Ek Madde 1 (…)") and ``description`` (HTML). The HTML carries amendment remarks
in bold ("(6676 sayılı Kanunun 26 ncı maddesiyle değişen ibare …)"), which go to
``amendment_notes``, and footnote markers ("(*)", bold "(3)", "(…)8") which are dropped.
After the article, GİB appends footnotes: the superseded wording ("(6676 sayılı Kanunla
değişmeden önceki ibare) …", "Değişmeden önceki şekli: …") and amendment explanations after
a "–––" separator or "[1]". Everything from the first such line on is history, not current
law, and goes to ``amendment_notes``.
"""

import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from tubitak_rag.domain.document import Document, Section
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.document_builder import build_document_skeleton, make_section
from tubitak_rag.ingestion.hashing import sha256_of
from tubitak_rag.ingestion.legislation_parser import (
    build_paragraphs,
    extract_amendment_notes,
    is_repealed,
    paragraphs_text,
)
from tubitak_rag.ingestion.normalize import normalize_text, tidy_punctuation, tr_lower

GIB_TITLE_RE = re.compile(
    r"^(?:(?P<prefix>Geçici|Ek)\s+)?Madde\s+(?P<no>\d+(?:/[A-ZÇĞİÖŞÜ])?)\s*(?:\(\*+\))?\s*"
    r"(?P<title>.*)$",
    re.IGNORECASE,
)
_MARKER_RE = re.compile(r"^\s*\((?:\*+|\d{1,2})\)\s*$")  # whole content of a bold footnote tag
_FOOTNOTES_RE = re.compile(r"\(\*+\)|\((?:…|\.\.\.)\)\d*|(?<=[.;:,])\(\d{1,2}\)")
_TITLE_FOOTNOTE_RE = re.compile(r"\s*\(\d{1,2}\)$")
_SPACES_RE = re.compile(r" {2,}")
_FOOTNOTE_START_RE = re.compile(
    r"^(?:\(\d{1,2}\)\s*|\[\d{1,2}\]\s*)?"
    r"(?:\(\d{4}\s+sayılı\s+Kanunla\s+\w+\s+önceki\s+\w+\)|Değişmeden\s+önceki\s+şekli\s*:)"
    r"|^\[\d{1,2}\]|^[–—_-]{5,}$",
    re.IGNORECASE,
)
_SEPARATOR_RE = re.compile(r"^[–—_-]{5,}$")
_STRUCTURED_MIN_ARTICLES = 3
_PREFIXES = {"geçici": "Geçici", "ek": "Ek"}


def html_to_text(desc_html: str) -> str:
    """Article HTML -> plain lines (one per ``<p>``), footnote markers removed."""
    soup = BeautifulSoup(desc_html, "html.parser")
    for tag in soup.find_all(["strong", "b", "span", "sup"]):
        if isinstance(tag, Tag) and _MARKER_RE.match(tag.get_text()):
            tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "tr", "li"]):
        if isinstance(block, Tag):
            block.append("\n")
    text = _FOOTNOTES_RE.sub("", normalize_text(soup.get_text()))
    lines = (tidy_punctuation(_SPACES_RE.sub(" ", ln)).strip() for ln in text.split("\n"))
    return "\n".join(line for line in lines if line)


def split_superseded(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split article lines into (current text, trailing footnotes/superseded wording)."""
    for index, line in enumerate(lines):
        if _FOOTNOTE_START_RE.match(line):
            return lines[:index], lines[index:]
    return lines, []


def parse_gib_title(title: str) -> tuple[str | None, str | None, list[str]]:
    """Split "Madde 4(*) Kuruluş" into (madde_no, article title, amendment notes)."""
    match = GIB_TITLE_RE.match(title.strip())
    if not match:
        return None, title.strip() or None, []
    prefix = match["prefix"]
    number = f"{_PREFIXES[tr_lower(prefix)]} {match['no']}" if prefix else match["no"]
    rest, notes = extract_amendment_notes(_TITLE_FOOTNOTE_RE.sub("", match["title"]).strip())
    return number, rest.strip() or None, notes


def _section(doc_id: str, index: int, item: dict[str, Any]) -> Section:
    number, title, title_notes = parse_gib_title(item.get("title") or "")
    lines, footnotes = split_superseded(html_to_text(item.get("description") or "").split("\n"))
    paragraphs, notes = build_paragraphs([(None, line) for line in lines])
    notes = title_notes + notes + [ln for ln in footnotes if not _SEPARATOR_RE.match(ln)]
    text = paragraphs_text(paragraphs)
    return make_section(
        doc_id,
        index,
        SectionKind.MADDE,
        text=text,
        paragraphs=paragraphs,
        title=title,
        madde_no=number,
        amendment_notes=notes,
        repealed=is_repealed(text, notes),
    )


class GibParser:
    """``DocumentParser`` for GİB law JSON files."""

    def parse(self, entry: SourceEntry, path: Path) -> Document:
        """Parse ``resultContainer.content[]`` into one MADDE section per item."""
        data = json.loads(path.read_text(encoding="utf-8"))
        items: list[dict[str, Any]] = data["resultContainer"]["content"]
        sections = [_section(entry.doc_id, n, item) for n, item in enumerate(items, start=1)]
        doc = build_document_skeleton(entry, sha256_of(path))
        return doc.model_copy(
            update={
                "sections": sections,
                "structured": len(sections) >= _STRUCTURED_MIN_ARTICLES,
                "kaynak_url": entry.kaynak_sayfa_url or entry.url,
            }
        )
