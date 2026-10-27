"""Chunk ids and the metadata shared by every chunker (FR-C7)."""

import hashlib
from typing import Any

from tubitak_rag.domain.document import Document, Section

HIERARCHY_SEPARATOR = " › "
_ARTICLE_PREFIXES = ("Geçici ", "Ek ")


def chunk_id_for(doc_id: str, chunker_name: str, index: int, text: str) -> str:
    """Stable 16-hex id; the doc id is part of the key so equal texts in two docs differ."""
    key = f"{doc_id}|{chunker_name}|{index}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def base_chunk_fields(doc: Document, chunker_name: str) -> dict[str, Any]:
    """Document-level chunk metadata that lets a chunk be traced back to its source."""
    return {
        "doc_id": doc.doc_id,
        "doc_title": doc.title,
        "program_kodu": doc.program_kodu,
        "doc_turu": doc.doc_turu.value,
        "kaynak_url": doc.kaynak_url,
        "surum_tarihi": doc.surum_tarihi.isoformat() if doc.surum_tarihi else None,
        "dil": doc.dil,
        "doc_sha256": doc.sha256,
        "chunker": chunker_name,
    }


def article_name(madde_no: str) -> str:
    """Turkish article name: "Madde 7", "Madde 5/A", "Geçici Madde 16", "Ek Madde 1"."""
    for prefix in _ARTICLE_PREFIXES:
        if madde_no.startswith(prefix):
            return f"{prefix}Madde {madde_no.removeprefix(prefix)}"
    return f"Madde {madde_no}"


def _own_label(section: Section) -> str:
    if section.madde_no:
        name = article_name(section.madde_no)
        return f"{name} – {section.title}" if section.title else name
    return section.title or ""


def hierarchy_label(section: Section) -> str:
    """Enclosing headings plus the section itself, e.g. "… Bölüm › Madde 7 – Destek oranı"."""
    parts = [*section.hierarchy, _own_label(section)]
    return HIERARCHY_SEPARATOR.join(part for part in parts if part)


def section_bolum(section: Section) -> str | None:
    """The ``bolum`` metadata: the section title, except for articles (located by madde_no)."""
    return None if section.madde_no else section.title
