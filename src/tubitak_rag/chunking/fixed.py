"""Fixed-size sliding-window chunker (FR-C1): the naive baseline of experiments v0/v0r.

The document text (repealed sections left out, FR-P6) becomes one token stream that is cut
into windows of ``size`` tokens; consecutive windows share ``overlap`` tokens so that a
sentence cut at a window edge still appears whole in one of them. Each section contributes
its own heading line ("Madde 7 – Destek oranı ve sınırı", a block title) and then its text,
as in the raw document, so the baseline is not deprived of titles. Windows still ignore
structure: the metadata (page, hierarchy, article) is that of the section where the window
starts.
"""

from bisect import bisect_right

from tubitak_rag.chunking.context_header import embed_text_for
from tubitak_rag.chunking.ids import (
    article_name,
    base_chunk_fields,
    chunk_id_for,
    hierarchy_label,
    section_bolum,
)
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.document import Document, Section
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.ports import TokenCounter

SECTION_SEPARATOR = "\n\n"


class FixedSizeChunker:
    """``Chunker`` that cuts windows of ``size`` tokens overlapping by ``overlap`` tokens."""

    name = "fixed"

    def __init__(
        self,
        counter: TokenCounter,
        size: int = 512,
        overlap: int = 64,
        context_header: bool = False,
    ) -> None:
        if size <= 0:
            raise ValueError(f"Geçersiz pencere boyutu: size ({size}) pozitif olmalı")
        if not 0 <= overlap < size:
            raise ValueError(
                f"Geçersiz pencere: overlap ({overlap}) 0 ile size ({size}) arasında olmalı"
            )
        self.counter = counter
        self.size = size
        self.overlap = overlap
        self.context_header = context_header

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split the non-repealed text of ``doc`` into overlapping token windows."""
        sections = [s for s in doc.sections if not s.repealed and s.text.strip()]
        ids, starts = self._token_stream(sections)
        base = base_chunk_fields(doc, self.name)
        chunks: list[Chunk] = []
        for start in range(0, len(ids), self.size - self.overlap):
            text = self.counter.decode(ids[start : start + self.size]).strip()
            section = sections[bisect_right(starts, start) - 1]
            label = hierarchy_label(section)
            chunks.append(
                Chunk(
                    **base,
                    chunk_id=chunk_id_for(doc.doc_id, self.name, len(chunks), text),
                    hiyerarsi=label,
                    madde_no=section.madde_no,
                    sayfa=section.page_start,
                    bolum=section_bolum(section),
                    text=text,
                    embed_text=embed_text_for(doc, label, text, context_header=self.context_header),
                    sira=len(chunks),
                )
            )
            if start + self.size >= len(ids):
                break
        return chunks

    def _token_stream(self, sections: list[Section]) -> tuple[list[int], list[int]]:
        """Token ids of all sections joined by blank lines, and each section's first index."""
        separator = self.counter.encode(SECTION_SEPARATOR)
        ids: list[int] = []
        starts: list[int] = []
        for section in sections:
            if ids:
                ids.extend(separator)
            starts.append(len(ids))
            ids.extend(self.counter.encode(_with_heading(section)))
        return ids, starts


def _with_heading(section: Section) -> str:
    """Section text preceded by its own heading line, as it reads in the source document."""
    if section.madde_no:
        name = article_name(section.madde_no)
        heading = f"{name} – {section.title}" if section.title else name
    elif section.kind == SectionKind.BOLUM and section.title:
        heading = section.title
    else:  # FAQ text already starts with "Soru:"; paragraphs and tables have no heading line
        return section.text
    return f"{heading}\n{section.text}"
