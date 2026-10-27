"""Optional context header prepended to ``embed_text`` (FR-C6, "contextual chunk headers").

A chunk such as "(1) Destek oranı en fazla %75'tir." does not say which program or article it
belongs to. The header line "Belge (program) › Bölüm › Madde N – Başlık" puts that context
into the embedded/indexed text while ``Chunk.text`` (shown to the user and the LLM) stays
unchanged.
"""

from tubitak_rag.chunking.ids import HIERARCHY_SEPARATOR
from tubitak_rag.domain.document import Document


def build_context_header(doc: Document, section_label: str) -> str:
    """Return "Title (program) › section label"; the parts that are missing are left out."""
    header = doc.title
    if doc.program_kodu:
        header += f" ({doc.program_kodu})"
    if section_label:
        header += f"{HIERARCHY_SEPARATOR}{section_label}"
    return header


def build_embed_text(header: str, text: str) -> str:
    """Return the header line followed by the chunk text."""
    return f"{header}\n{text}"


def embed_text_for(doc: Document, section_label: str, text: str, *, context_header: bool) -> str:
    """Return the text to embed: ``text`` itself, or with the context header line when enabled."""
    if not context_header:
        return text
    return build_embed_text(build_context_header(doc, section_label), text)
