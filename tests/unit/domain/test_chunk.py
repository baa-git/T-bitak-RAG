"""Tests for the Chunk model (FR-C7 metadata schema)."""

from tests.builders import make_chunk
from tubitak_rag.domain.chunk import Chunk

SPEC_FIELDS = {
    "chunk_id",
    "doc_id",
    "doc_title",
    "program_kodu",
    "doc_turu",
    "hiyerarsi",
    "madde_no",
    "sayfa",
    "bolum",
    "kaynak_url",
    "surum_tarihi",
    "dil",
    "doc_sha256",
    "chunker",
    "text",
    "embed_text",
}


def test_chunk_schema_has_spec_fields() -> None:
    assert set(Chunk.model_fields) >= SPEC_FIELDS  # FR-C7


def test_chunk_round_trips_json() -> None:
    chunk = make_chunk(degisiklik_notlari=["(Değişik: 29/12/2025-77YK)"])
    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_chunk_amendment_notes_default_is_not_shared() -> None:
    first, second = make_chunk("a"), make_chunk("b")
    assert (
        first.degisiklik_notlari == [] and first.degisiklik_notlari is not second.degisiklik_notlari
    )
