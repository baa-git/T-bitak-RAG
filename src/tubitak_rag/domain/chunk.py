"""Chunk model: the unit that is embedded, indexed and cited (metadata schema of spec §7.3)."""

from tubitak_rag.domain.base import DomainModel


class Chunk(DomainModel):
    """A retrievable piece of a document with its citation metadata (FR-C7)."""

    chunk_id: str
    doc_id: str
    doc_title: str
    program_kodu: str | None
    doc_turu: str
    hiyerarsi: str  # "Üçüncü Bölüm › Madde 7 – Destek oranı › (2)"
    madde_no: str | None
    sayfa: int | None
    bolum: str | None
    kaynak_url: str
    surum_tarihi: str | None  # ISO date
    dil: str
    doc_sha256: str
    chunker: str
    text: str  # shown to the user and the LLM
    embed_text: str  # text that is embedded / indexed (may carry a context header)
    sira: int  # order of the chunk inside its document
    degisiklik_notlari: list[str] = []
