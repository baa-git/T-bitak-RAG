"""Structure-aware chunker (FR-C2 … FR-C5): experiments v1 … v3b.

One chunk per legal article (madde), FAQ pair, HTML block or guide heading; consecutive
heading-less guide paragraphs are packed together. A unit longer than ``max_tokens`` is split
only between its paragraphs (fıkra/bent), never inside one (FR-C3). Repealed articles are
skipped (FR-P6) and amendment notes travel as metadata, not text (FR-P5).
"""

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import groupby

from tubitak_rag.chunking.context_header import embed_text_for
from tubitak_rag.chunking.ids import (
    HIERARCHY_SEPARATOR,
    base_chunk_fields,
    chunk_id_for,
    hierarchy_label,
    section_bolum,
)
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document, Paragraph, Section
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.ports import TokenCounter

logger = logging.getLogger(__name__)

# Parsers join paragraphs with one of these; split parts reuse the separator of their section.
ARTICLE_SEPARATOR = "\n"
BLOCK_SEPARATOR = "\n\n"
_FIKRA_LABEL = re.compile(r"\(\d+\)")


def render_paragraph(paragraph: Paragraph) -> str:
    """Paragraph text with its label, e.g. "(2) Metin." (as it appears in the section text)."""
    return f"{paragraph.label} {paragraph.text}" if paragraph.label else paragraph.text


def pack_paragraphs(
    paragraphs: Sequence[Paragraph],
    counter: TokenCounter,
    max_tokens: int,
    separator: str = ARTICLE_SEPARATOR,
    source: str = "",
) -> list[list[Paragraph]]:
    """Greedily group consecutive paragraphs; a group joined by ``separator`` fits ``max_tokens``.

    A single paragraph longer than the limit is never cut; it becomes a group of its own and
    a warning naming ``source`` (e.g. the section id) is logged.
    """
    separator_tokens = counter.count(separator)
    groups: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    used = 0
    for paragraph in paragraphs:
        tokens = counter.count(render_paragraph(paragraph))
        if tokens > max_tokens:
            logger.warning(
                "%s: paragraf %d token, sınır %d; bölünmeden tek parça bırakıldı",
                source,
                tokens,
                max_tokens,
            )
        if current and used + separator_tokens + tokens > max_tokens:
            groups.append(current)
            current, used = [], 0
        used += tokens + (separator_tokens if current else 0)
        current.append(paragraph)
    if current:
        groups.append(current)
    return groups


@dataclass
class _Piece:
    """The text of one future chunk and the section it is located in."""

    section: Section
    text: str
    page: int | None
    label_suffix: str | None = None
    notes: list[str] = field(default_factory=list)


class StructuralChunker:
    """``Chunker`` that follows the document structure produced by the parsers."""

    name = "structural"

    def __init__(
        self, counter: TokenCounter, max_tokens: int = 450, context_header: bool = True
    ) -> None:
        self.counter = counter
        self.max_tokens = max_tokens
        self.context_header = context_header

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into structural chunks in document order."""
        base = base_chunk_fields(doc, self.name)
        chunks: list[Chunk] = []
        for piece in self._pieces(doc):
            label = hierarchy_label(piece.section)
            if piece.label_suffix:
                label += f"{HIERARCHY_SEPARATOR}{piece.label_suffix}"
            chunks.append(
                Chunk(
                    **base,
                    chunk_id=chunk_id_for(doc.doc_id, self.name, len(chunks), piece.text),
                    hiyerarsi=label,
                    madde_no=piece.section.madde_no,
                    sayfa=piece.page,
                    bolum=section_bolum(piece.section),
                    text=piece.text,
                    embed_text=embed_text_for(
                        doc, label, piece.text, context_header=self.context_header
                    ),
                    sira=len(chunks),
                    degisiklik_notlari=piece.notes,
                )
            )
        return chunks

    def _pieces(self, doc: Document) -> Iterator[_Piece]:
        sections = [s for s in doc.sections if not s.repealed and s.text.strip()]
        for is_paragraph, run in groupby(sections, key=lambda s: s.kind == SectionKind.PARAGRAF):
            if is_paragraph:
                yield from self._paragraph_run(list(run))
            else:
                for section in run:
                    yield from self._section_pieces(section)

    def _section_pieces(self, section: Section) -> Iterator[_Piece]:
        """One piece per section; long articles/blocks/tables are split between paragraphs."""
        whole = (
            section.kind == SectionKind.SSS  # a question and its answer stay together (FR-C4)
            or len(section.paragraphs) < 2
            or self.counter.count(section.text) <= self.max_tokens
        )
        notes = list(section.amendment_notes)
        if whole:
            yield _Piece(section, section.text, _first_page(section), notes=notes)
            return
        is_article = section.kind == SectionKind.MADDE
        separator = _separator_of(section)
        fikras = _enclosing_fikra_labels(section.paragraphs)
        start = 0
        for group in pack_paragraphs(
            section.paragraphs, self.counter, self.max_tokens, separator, section.section_id
        ):
            yield _Piece(
                section,
                separator.join(render_paragraph(p) for p in group),
                group[0].page or section.page_start,
                label_suffix=fikras[start] if is_article else None,
                notes=notes,
            )
            start += len(group)

    def _paragraph_run(self, sections: list[Section]) -> Iterator[_Piece]:
        """Pack consecutive heading-less paragraph sections up to ``max_tokens`` (FR-C5)."""
        as_paragraphs = [Paragraph(label=None, text=s.text, page=s.page_start) for s in sections]
        start = 0
        for group in pack_paragraphs(
            as_paragraphs, self.counter, self.max_tokens, BLOCK_SEPARATOR, sections[0].section_id
        ):
            members = sections[start : start + len(group)]
            yield _Piece(
                members[0],
                BLOCK_SEPARATOR.join(s.text for s in members),
                members[0].page_start,
                notes=[note for s in members for note in s.amendment_notes],
            )
            start += len(group)


def _separator_of(section: Section) -> str:
    """The separator the parser used, so that split parts stay verbatim slices of the text."""
    for separator in (ARTICLE_SEPARATOR, BLOCK_SEPARATOR):
        if separator.join(render_paragraph(p) for p in section.paragraphs) == section.text:
            return separator
    return ARTICLE_SEPARATOR if section.kind == SectionKind.MADDE else BLOCK_SEPARATOR


def _first_page(section: Section) -> int | None:
    first = section.paragraphs[0].page if section.paragraphs else None
    return first or section.page_start


def _enclosing_fikra_labels(paragraphs: Sequence[Paragraph]) -> list[str | None]:
    """For each paragraph the label of the fıkra it belongs to ("(1)" for bent "a)" under it).

    Paragraphs before the first fıkra (or in an article without fıkras) get None: a bare bent
    label such as "f)" is not a meaningful location on its own.
    """
    labels: list[str | None] = []
    current: str | None = None
    for paragraph in paragraphs:
        if paragraph.label and _FIKRA_LABEL.fullmatch(paragraph.label):
            current = paragraph.label
        labels.append(current)
    return labels
