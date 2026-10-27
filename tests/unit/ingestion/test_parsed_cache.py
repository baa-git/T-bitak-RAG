"""Tests for the parsed-document cache and batch parsing (FR-X9, FR-P9, FR-P10)."""

import shutil
from datetime import date
from pathlib import Path

import pytest

from tubitak_rag.domain.document import Document
from tubitak_rag.domain.enums import DocType, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion import parsed_cache
from tubitak_rag.ingestion.document_builder import build_document_skeleton
from tubitak_rag.ingestion.hashing import sha256_of
from tubitak_rag.ingestion.parsed_cache import ParsedCache, parse_all, parse_entry
from tubitak_rag.ports.parser import DocumentParser


def _entry(local_path: str, kind: SourceKind = SourceKind.PDF, **kw: object) -> SourceEntry:
    fields: dict[str, object] = {
        "doc_id": "dosya-abc",
        "title": "Örnek Kılavuz",
        "url": "https://tubitak.gov.tr/x.pdf",
        "local_path": local_path,
        "kind": kind,
        "program_kodu": None,
        "doc_turu": DocType.KILAVUZ,
        "surum_tarihi": None,
        "dil": "tr",
        "sha256": None,
        "kaynak_grubu": "tubitak_program_sayfalari",
        "kaynak_sayfa_url": None,
    }
    fields.update(kw)
    return SourceEntry.model_validate(fields)


@pytest.fixture
def raw(tmp_path: Path, fixtures: Path) -> Path:
    raw_dir = tmp_path / "raw"
    (raw_dir / "pdf").mkdir(parents=True)
    for name in ("kilavuz.pdf", "bos.pdf"):
        shutil.copy(fixtures / "pdf" / name, raw_dir / "pdf" / name)
    shutil.copy(fixtures / "program_1501.html", raw_dir / "sayfa.html")
    return raw_dir


@pytest.fixture
def cache(tmp_path: Path) -> ParsedCache:
    return ParsedCache(tmp_path / "artifacts" / "cache" / "parsed")


class CountingParser:
    """Wraps a real parser and counts calls."""

    def __init__(self, inner: DocumentParser) -> None:
        self.inner = inner
        self.calls = 0

    def parse(self, entry: SourceEntry, path: Path) -> Document:
        self.calls += 1
        return self.inner.parse(entry, path)


def test_path_for(cache: ParsedCache) -> None:
    path = cache.path_for("mevzuat-031", "abcdef0123456789" + "0" * 48)
    assert path.name == "mevzuat-031.abcdef012345.json"


def test_cache_roundtrip(cache: ParsedCache) -> None:
    doc = build_document_skeleton(_entry("x.pdf"), "f" * 64, page_count=3)
    assert cache.get(doc.doc_id, doc.sha256) is None
    cache.put(doc)
    assert cache.get(doc.doc_id, doc.sha256) == doc
    assert cache.get(doc.doc_id, "e" * 64) is None  # changed file -> miss


def test_latest_returns_the_newest_version_without_the_raw_file(cache: ParsedCache) -> None:
    import os

    assert cache.latest("dosya-abc") is None
    old = build_document_skeleton(_entry("x.pdf"), "a" * 64, page_count=1)
    new = build_document_skeleton(_entry("x.pdf"), "b" * 64, page_count=2)
    other = build_document_skeleton(_entry("y.pdf", doc_id="dosya-abcd"), "c" * 64)
    for doc in (old, new, other):
        cache.put(doc)
    os.utime(cache.path_for(old.doc_id, old.sha256), (1, 1))  # older file
    assert cache.latest("dosya-abc") == new
    assert cache.latest("dosya-abcd") == other
    assert ParsedCache(cache.directory, parser_version="0").latest("dosya-abc") is None


def test_corrupt_cache_file_is_a_miss(cache: ParsedCache) -> None:
    path = cache.path_for("d", "f" * 64)
    path.parent.mkdir(parents=True)
    path.write_text("{bozuk", encoding="utf-8")
    assert cache.get("d", "f" * 64) is None


def test_parse_entry_parses_and_caches(raw: Path, cache: ParsedCache) -> None:
    doc, outcome = parse_entry(_entry("pdf/kilavuz.pdf"), raw, cache)
    assert doc is not None and outcome.status == "parsed"
    assert outcome.doc_id == "dosya-abc" and outcome.sections == len(doc.sections) == 3
    assert cache.path_for(doc.doc_id, doc.sha256).is_file()


def test_parse_entry_uses_cache_second_time(
    raw: Path, cache: ParsedCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FR-X9: an unchanged file is not parsed again.
    counting = CountingParser(parsed_cache.parser_for(_entry("pdf/kilavuz.pdf")))
    monkeypatch.setattr(parsed_cache, "parser_for", lambda entry: counting)
    first, _ = parse_entry(_entry("pdf/kilavuz.pdf"), raw, cache)
    second, outcome = parse_entry(_entry("pdf/kilavuz.pdf"), raw, cache)
    assert counting.calls == 1
    assert outcome.status == "cached" and second == first


def test_document_sha_comes_from_file_not_manifest(raw: Path, cache: ParsedCache) -> None:
    doc, _ = parse_entry(_entry("pdf/kilavuz.pdf", sha256="0" * 64), raw, cache)
    assert doc is not None and doc.sha256 == sha256_of(raw / "pdf" / "kilavuz.pdf")


def test_missing_file_outcome(raw: Path, cache: ParsedCache) -> None:
    doc, outcome = parse_entry(_entry("pdf/yok.pdf"), raw, cache)
    assert doc is None and outcome.status == "missing" and outcome.sections == 0
    assert "pdf/yok.pdf" in outcome.message


def test_scanned_outcome(raw: Path, cache: ParsedCache) -> None:
    doc, outcome = parse_entry(_entry("pdf/bos.pdf"), raw, cache)  # FR-P9
    assert doc is not None and doc.scanned is True
    assert outcome.status == "scanned" and outcome.sections == 0


def test_error_outcome(raw: Path, cache: ParsedCache, monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def parse(self, entry: SourceEntry, path: Path) -> Document:
            raise ValueError("bozuk dosya")

    monkeypatch.setattr(parsed_cache, "parser_for", lambda entry: Broken())
    doc, outcome = parse_entry(_entry("pdf/kilavuz.pdf"), raw, cache)
    assert doc is None and outcome.status == "error" and "bozuk dosya" in outcome.message
    assert not cache.directory.exists()  # nothing cached for a failed parse


def test_document_has_all_metadata_fields(raw: Path, cache: ParsedCache) -> None:
    entry = _entry(
        "sayfa.html",
        kind=SourceKind.HTML,
        doc_id="sayfa-1501",
        program_kodu="1501",
        doc_turu=DocType.PROGRAM_SAYFASI,
        surum_tarihi=date(2026, 10, 15),
    )
    doc, _ = parse_entry(entry, raw, cache)
    assert doc is not None
    data = doc.model_dump()
    for field in (
        "doc_id",
        "title",
        "program_kodu",
        "doc_turu",
        "kaynak_url",
        "surum_tarihi",
        "dil",
        "sha256",
    ):
        assert field in data  # FR-P10
    assert (doc.program_kodu, doc.surum_tarihi, doc.dil) == ("1501", date(2026, 10, 15), "tr")
    no_meta, _ = parse_entry(_entry("pdf/kilavuz.pdf"), raw, cache)
    assert no_meta is not None and no_meta.program_kodu is None and no_meta.surum_tarihi is None


def test_parse_all_collects_documents_and_outcomes(raw: Path, cache: ParsedCache) -> None:
    entries = [
        _entry("pdf/kilavuz.pdf", doc_id="a"),
        _entry("pdf/yok.pdf", doc_id="b"),
        _entry("pdf/bos.pdf", doc_id="c"),
    ]
    docs, outcomes = parse_all(entries, raw, cache)
    assert [d.doc_id for d in docs] == ["a", "c"]
    assert [o.status for o in outcomes] == ["parsed", "missing", "scanned"]


def test_cache_entry_from_another_parser_version_is_a_miss(tmp_path: Path) -> None:
    # The parser code is an input too: after a parser change the cache must not be reused.
    doc = build_document_skeleton(_entry("x.pdf"), "f" * 64)
    ParsedCache(tmp_path, parser_version="1").put(doc)
    assert ParsedCache(tmp_path, parser_version="1").get(doc.doc_id, doc.sha256) == doc
    assert ParsedCache(tmp_path, parser_version="2").get(doc.doc_id, doc.sha256) is None


def test_default_parser_version_is_set(tmp_path: Path) -> None:
    assert parsed_cache.PARSER_VERSION
    assert ParsedCache(tmp_path).parser_version == parsed_cache.PARSER_VERSION


def test_long_doc_ids_get_short_unique_file_names(cache: ParsedCache) -> None:
    # I-5: programme page ids are up to ~160 characters; Windows MAX_PATH is 260.
    sha = "abcdef0123456789" + "0" * 48
    long_a = "sayfa-akademik__ulusal-destek-programlari__" + "a" * 120
    long_b = "sayfa-akademik__ulusal-destek-programlari__" + "a" * 119 + "b"
    name_a, name_b = cache.path_for(long_a, sha).name, cache.path_for(long_b, sha).name
    assert len(name_a) <= 80 and name_a != name_b
    assert name_a.startswith(long_a[:48] + "~") and name_a.endswith(".abcdef012345.json")
    assert cache.path_for("mevzuat-031", sha).name == "mevzuat-031.abcdef012345.json"


def test_cache_write_failure_does_not_stop_parsing(
    raw: Path, cache: ParsedCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_put(doc: Document) -> None:
        raise OSError("disk dolu")

    monkeypatch.setattr(cache, "put", broken_put)
    doc, outcome = parse_entry(_entry("pdf/kilavuz.pdf"), raw, cache)
    assert doc is not None and outcome.status == "parsed"
    assert "önbelleğe yazılamadı" in outcome.message and "disk dolu" in outcome.message


# Digest of the parser sources each PARSER_VERSION was released with. When this
# test fails you changed a parser: bump PARSER_VERSION if the parsed output changes (stale
# cache entries are then ignored), otherwise only record the new digest here.
RECORDED_PARSER_DIGESTS = {"2": "c0c48bbc62c042b2"}


def test_parser_version_is_bumped_when_parser_sources_change() -> None:
    digest = parsed_cache.parser_sources_digest()
    recorded = RECORDED_PARSER_DIGESTS.get(parsed_cache.PARSER_VERSION)
    assert digest == recorded, (
        f"Ayrıştırıcı kaynakları değişti (özet {digest}). Çıktı değiştiyse PARSER_VERSION'ı "
        "artırın; değişmediyse RECORDED_PARSER_DIGESTS'i güncelleyin."
    )


def test_parser_sources_digest_ignores_line_endings(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_bytes(b"x = 1\r\n")
    (tmp_path / "b.py").write_bytes(b"x = 1\n")
    digest_a = parsed_cache.parser_sources_digest([tmp_path / "a.py"])
    assert digest_a == parsed_cache.parser_sources_digest([tmp_path / "b.py"])
