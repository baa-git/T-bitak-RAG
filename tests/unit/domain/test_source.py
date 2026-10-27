"""Tests for the SourceEntry manifest model."""

from datetime import date

import pytest
from pydantic import ValidationError

from tubitak_rag.domain.enums import DocType, SourceKind
from tubitak_rag.domain.source import SourceEntry


def _entry(**overrides: object) -> SourceEntry:
    fields: dict[str, object] = {
        "doc_id": "mevzuat-031",
        "title": "1507 Uygulama Esasları",
        "url": "https://tubitak.gov.tr/x.pdf",
        "local_path": "tubitak_mevzuat/031_1507.pdf",
        "kind": "pdf",
        "program_kodu": "1507",
        "doc_turu": "uygulama_esaslari",
        "surum_tarihi": "2026-09-25",
        "dil": "tr",
        "sha256": None,
        "kaynak_grubu": "tubitak_mevzuat",
        "kaynak_sayfa_url": None,
    }
    fields.update(overrides)
    return SourceEntry.model_validate(fields)


def test_source_entry_parses_enums_and_date() -> None:
    entry = _entry()
    assert entry.kind is SourceKind.PDF
    assert entry.doc_turu is DocType.UYGULAMA_ESASLARI
    assert entry.surum_tarihi == date(2026, 9, 25)


def test_source_entry_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        _entry(kind="docx")


def test_source_entry_is_frozen() -> None:
    entry = _entry()
    with pytest.raises(ValidationError):
        entry.title = "x"  # type: ignore[misc]
