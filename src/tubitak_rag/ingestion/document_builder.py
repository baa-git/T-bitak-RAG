"""Helpers shared by every parser: the document skeleton, section ids and line joining.

Kept apart from ``parser_registry`` so that parsers can use them without an import cycle
(the registry imports the parsers).
"""

import re
from collections.abc import Iterable, Sequence

from tubitak_rag.domain.document import Document, Paragraph, Section
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.domain.source import SourceEntry

_WORD_HYPHEN_END = re.compile(r"\w-$")


def make_section_id(doc_id: str, index: int) -> str:
    """Section id ``<doc_id>:<index:04d>`` (sections are numbered from 1)."""
    return f"{doc_id}:{index:04d}"


def build_document_skeleton(
    entry: SourceEntry, sha256: str, *, scanned: bool = False, page_count: int | None = None
) -> Document:
    """A ``Document`` with all metadata of ``entry`` and no sections yet (FR-P10).

    ``sha256`` is the digest of the file that was actually parsed; unknown metadata stays
    ``None`` explicitly.
    """
    return Document(
        doc_id=entry.doc_id,
        title=entry.title,
        program_kodu=entry.program_kodu,
        doc_turu=entry.doc_turu,
        kind=entry.kind,
        kaynak_url=entry.url,
        surum_tarihi=entry.surum_tarihi,
        dil=entry.dil,
        sha256=sha256,
        kaynak_grubu=entry.kaynak_grubu,
        scanned=scanned,
        structured=False,
        page_count=page_count,
        sections=[],
    )


def join_lines(lines: Iterable[str]) -> str:
    """Join wrapped PDF lines with spaces; a word ending in "-" continues without a space.

    Turkish legal texts do not hyphenate words at line ends, so such a hyphen is real
    ("Araştırma-" + "Geliştirme", "51YK-" + "15/06/2023") and is kept.
    """
    text = ""
    for line in lines:
        if not line:
            continue
        if not text:
            text = line
        elif _WORD_HYPHEN_END.search(text):
            text += line
        else:
            text += " " + line
    return text


def attach_page_notes(
    sections: Sequence[Section], page_notes: Sequence[tuple[int, Sequence[str]]]
) -> list[Section]:
    """Append each page's notes to the last section that started on or before that page."""
    extra: dict[int, list[str]] = {}
    for page, notes in page_notes:
        if not notes or not sections:
            continue
        started = [i for i, s in enumerate(sections) if (s.page_start or 0) <= page]
        extra.setdefault(started[-1] if started else 0, []).extend(notes)
    return [
        s.model_copy(update={"amendment_notes": [*s.amendment_notes, *extra[i]]})
        if i in extra
        else s
        for i, s in enumerate(sections)
    ]


def make_section(
    doc_id: str,
    index: int,
    kind: SectionKind,
    *,
    text: str,
    paragraphs: Sequence[Paragraph] = (),
    title: str | None = None,
    hierarchy: Sequence[str] = (),
    madde_no: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    amendment_notes: Sequence[str] = (),
    repealed: bool = False,
) -> Section:
    """Build a ``Section``; parsers pass only the fields their format has."""
    return Section(
        section_id=make_section_id(doc_id, index),
        kind=kind,
        title=title,
        hierarchy=list(hierarchy),
        madde_no=madde_no,
        page_start=page_start,
        page_end=page_end,
        text=text,
        paragraphs=list(paragraphs),
        amendment_notes=list(amendment_notes),
        repealed=repealed,
    )
