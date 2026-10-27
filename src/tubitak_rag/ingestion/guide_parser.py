"""Heading/paragraph parser for guides (kılavuz) and call texts (çağrı), FR-C5.

These documents have no articles. They are split at headings ("1. Giriş", "4.2 Bütçe",
"BAŞVURU KOŞULLARI"); a document without headings is split into paragraphs, which the
chunker later packs. Numbered list items look like headings, so a numbered heading is only
accepted when it continues the document's numbering (after "5." comes "5.1" or "6.", not "1.").
"""

import re
from collections.abc import Sequence

from tubitak_rag.domain.document import Paragraph, Section
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.ingestion.document_builder import (
    attach_page_notes,
    join_lines,
    make_section,
)
from tubitak_rag.ingestion.legislation_parser import extract_amendment_notes
from tubitak_rag.ingestion.pdf_text import PageText

HEADING_RE = re.compile(r"^\s*(?P<num>\d{1,2}(?:\.\d{1,2})*)[.)]?\s+(?P<text>\S.{0,72})$")
_TOC_RE = re.compile(r"\.{5,}|…{2,}")  # table-of-contents dot leaders
_BULLETS = "•▪◦●\uf0b7\uf0a7\uf0d8\uf076\uf0fc"  # incl. Symbol-font bullets
_BULLET_RE = re.compile(rf"^(?:[{_BULLETS}]|[-–o])(?:\s+|$)")
_MAX_CAPS_WORDS = 12

Numbering = tuple[int, ...]


def _numbered(line: str) -> tuple[Numbering, str] | None:
    match = HEADING_RE.match(line)
    if not match or _TOC_RE.search(line):
        return None
    text = match["text"].strip()
    if not text[0].isupper() or text.endswith((".", ",", ";")):
        return None
    return tuple(int(part) for part in match["num"].split(".")), line.strip()


def _is_caps_heading(line: str) -> bool:
    letters = [ch for ch in line if ch.isalpha()]
    return (
        len(letters) >= 2
        and all(ch.isupper() for ch in letters)
        and len(line.split()) <= _MAX_CAPS_WORDS
        and not line.rstrip().endswith(".")
        and not _TOC_RE.search(line)
    )


def is_heading(line: str) -> bool:
    """Does ``line`` look like a heading (numbered or all capitals)?"""
    return bool(line.strip()) and (_numbered(line) is not None or _is_caps_heading(line))


def _follows(number: Numbering, last: Numbering | None) -> bool:
    """Is ``number`` a valid next heading number after ``last``?"""
    if last is None:
        return number[0] == 1
    if number == (*last, 1):
        return True
    return any(number == (*last[:depth], last[depth] + 1) for depth in range(len(last)))


def _bullet(line: str) -> str | None:
    match = _BULLET_RE.match(line)
    return "• " + line[match.end() :] if match else None


def split_paragraphs(lines: Sequence[tuple[int, str]]) -> list[Paragraph]:
    """Group lines into paragraphs; bullet items stay in their paragraph, one per line.

    A paragraph ends at a blank line, or when a line ending in ``.``/``:``/``;`` is followed
    by a line starting with a capital letter or digit.
    """
    groups: list[tuple[int, list[str]]] = []
    previous = ""
    for page, line in lines:
        if not line.strip():
            previous = ""
            continue
        bullet = _bullet(line)
        starts_new = not groups or not previous
        if bullet is None and previous.endswith((".", ":", ";")):
            starts_new = starts_new or line[0].isupper() or line[0].isdigit()
        if starts_new:
            groups.append((page, [bullet or line]))
        elif bullet is not None:
            groups[-1][1].append("\n" + bullet)
        else:
            groups[-1][1].append(line)
        previous = line
    return [Paragraph(label=None, text=_join(parts), page=page) for page, parts in groups]


def _join(parts: list[str]) -> str:
    """Join lines; parts starting with a newline (bullets) begin a new output line."""
    segments: list[list[str]] = [[]]
    for part in parts:
        if part.startswith("\n"):
            segments.append([part[1:]])
        else:
            segments[-1].append(part)
    return "\n".join(join_lines(segment) for segment in segments if segment)


def _without_notes(paragraphs: list[Paragraph]) -> tuple[list[Paragraph], list[str]]:
    """Move amendment notes out of paragraph text, line by line (bullets keep their lines)."""
    clean: list[Paragraph] = []
    notes: list[str] = []
    for paragraph in paragraphs:
        lines: list[str] = []
        for line in paragraph.text.split("\n"):
            text, found = extract_amendment_notes(line)
            notes += found
            if text:
                lines.append(text)
        if lines:
            clean.append(paragraph.model_copy(update={"text": "\n".join(lines)}))
    return clean, notes


def _section(
    doc_id: str,
    index: int,
    kind: SectionKind,
    title: str | None,
    hierarchy: list[str],
    paragraphs: list[Paragraph],
    page_end: int | None,
) -> Section:
    paragraphs, notes = _without_notes(paragraphs)
    if title:
        title, title_notes = extract_amendment_notes(title)
        notes = title_notes + notes
    return make_section(
        doc_id,
        index,
        kind,
        text="\n\n".join(p.text for p in paragraphs),
        paragraphs=paragraphs,
        title=title,
        hierarchy=hierarchy,
        page_start=paragraphs[0].page,
        page_end=page_end,
        amendment_notes=notes,
    )


def parse_sections(pages: Sequence[PageText], doc_id: str, doc_title: str) -> list[Section]:
    """Split a guide/call text into heading sections, or paragraph sections if none."""
    lines = [(p.number, ln) for p in pages for ln in p.lines if not _TOC_RE.search(ln)]
    # (title, hierarchy, body lines); the first block collects text before any heading.
    blocks: list[tuple[str | None, list[str], list[tuple[int, str]]]] = [(None, [], [])]
    last: Numbering | None = None
    parents: list[tuple[Numbering, str]] = []
    for page, line in lines:
        numbered = _numbered(line)
        if numbered and _follows(numbered[0], last):
            number, title = numbered
            parents = [(n, t) for n, t in parents if number[: len(n)] == n and n != number]
            blocks.append((title, [t for _, t in parents], []))
            parents.append((number, title))
            last = number
        elif not numbered and _is_caps_heading(line):
            blocks.append((line.strip(), [], []))
        else:
            blocks[-1][2].append((page, line))
    page_notes = [(page.number, page.footnotes) for page in pages]
    if len(blocks) == 1:  # no headings: one section per paragraph
        paragraph_sections = [
            _section(doc_id, n, SectionKind.PARAGRAF, None, [], [p], p.page)
            for n, p in enumerate(split_paragraphs(lines), start=1)
        ]
        return attach_page_notes(paragraph_sections, page_notes)
    sections: list[Section] = []
    for block_title, hierarchy, body in blocks:
        paragraphs = split_paragraphs(body)
        if paragraphs:
            heading = block_title if block_title is not None else doc_title
            sections.append(
                _section(
                    doc_id,
                    len(sections) + 1,
                    SectionKind.BOLUM,
                    heading,
                    hierarchy,
                    paragraphs,
                    body[-1][0],
                )
            )
    return attach_page_notes(sections, page_notes)
