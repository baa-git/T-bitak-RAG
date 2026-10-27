"""Small builders that create valid domain objects with sensible defaults for tests."""

from typing import Any

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.retrieval import ScoredChunk


def make_chunk(chunk_id: str = "mevzuat-031:0001:00", **overrides: Any) -> Chunk:
    """Return a valid Chunk; any field can be overridden by keyword."""
    text = overrides.pop("text", "Destek oranı her dönem için en fazla %75 olarak belirlenmiştir.")
    fields: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": "mevzuat-031",
        "doc_title": "1507 Uygulama Esasları",
        "program_kodu": "1507",
        "doc_turu": "uygulama_esaslari",
        "hiyerarsi": "Üçüncü Bölüm › Madde 7 – Destek oranı",
        "madde_no": "7",
        "sayfa": 6,
        "bolum": None,
        "kaynak_url": "https://tubitak.gov.tr/x.pdf",
        "surum_tarihi": "2026-09-25",
        "dil": "tr",
        "doc_sha256": "a" * 64,
        "chunker": "structural",
        "text": text,
        "embed_text": text,
        "sira": 1,
    }
    fields.update(overrides)
    return Chunk(**fields)


def make_scored(chunk: Chunk, score: float, **scores: float) -> ScoredChunk:
    """Wrap a chunk with a primary score and optional named component scores."""
    return ScoredChunk(chunk=chunk, score=score, scores=scores)
