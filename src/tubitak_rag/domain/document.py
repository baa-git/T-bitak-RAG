"""Parsed document model: a source split into structural sections."""

from datetime import date

from tubitak_rag.domain.base import DomainModel
from tubitak_rag.domain.enums import DocType, SectionKind, SourceKind


class Paragraph(DomainModel):
    """A numbered paragraph/clause (fıkra/bent) or a plain paragraph."""

    label: str | None  # "(1)", "a)", None
    text: str
    page: int | None


class Section(DomainModel):
    """A structural unit: an article (madde), chapter, FAQ item, paragraph or table."""

    section_id: str  # f"{doc_id}:{order:04d}"
    kind: SectionKind
    title: str | None
    hierarchy: list[str]  # enclosing headings, excluding the article itself
    madde_no: str | None  # "7", "Geçici 16"
    page_start: int | None
    page_end: int | None
    text: str  # clean text, amendment notes removed
    paragraphs: list[Paragraph]
    amendment_notes: list[str]  # "(Değişik: 29/12/2025-77YK)" …
    repealed: bool


class Document(DomainModel):
    """A fully parsed document ready for chunking."""

    doc_id: str
    title: str
    program_kodu: str | None
    doc_turu: DocType
    kind: SourceKind
    kaynak_url: str
    surum_tarihi: date | None
    dil: str
    sha256: str
    kaynak_grubu: str
    scanned: bool  # no text could be extracted from the PDF
    structured: bool  # at least three articles were found
    page_count: int | None
    sections: list[Section]

    def full_text(self) -> str:
        """Return all section texts joined by blank lines."""
        return "\n\n".join(section.text for section in self.sections)
