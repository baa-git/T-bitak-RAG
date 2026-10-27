"""Tests for downloading the corpus with SHA-256 verification (FR-D1, FR-D2, FR-D3)."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from tubitak_rag.domain.enums import DocType, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.fetch import (
    FetchReport,
    FetchResult,
    FetchStatus,
    fetch_all,
    write_report,
)
from tubitak_rag.ingestion.hashing import sha256_of

PDF_BYTES = b"%PDF-1.4 example body"
HTML_PAGE = (
    "<html><body><nav>menü</nav><article><p>Destek oranı</p></article><footer/></body></html>"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(**overrides: Any) -> SourceEntry:
    fields: dict[str, Any] = {
        "doc_id": "mevzuat-031",
        "title": "1507 Uygulama Esasları",
        "url": "https://tubitak.gov.tr/x.pdf",
        "local_path": "tubitak_mevzuat/031.pdf",
        "kind": SourceKind.PDF,
        "program_kodu": "1507",
        "doc_turu": DocType.UYGULAMA_ESASLARI,
        "surum_tarihi": None,
        "dil": "tr",
        "sha256": _sha(PDF_BYTES),
        "kaynak_grubu": "tubitak_mevzuat",
        "kaynak_sayfa_url": None,
    }
    fields.update(overrides)
    return SourceEntry(**fields)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _serve(body: bytes, calls: list[str] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(200, content=body)

    return _client(handler)


def _fetch(entries: list[SourceEntry], raw: Path, client: httpx.Client, **kw: Any) -> FetchReport:
    return fetch_all(entries, raw, client, sleep=lambda _s: None, **kw)


def test_sha256_of(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    assert sha256_of(path) == _sha(b"abc")


def test_downloads_missing_file_and_verifies_sha(tmp_path: Path) -> None:
    report = _fetch([_entry()], tmp_path, _serve(PDF_BYTES))
    (result,) = report.results
    assert result.status == FetchStatus.INDIRILDI
    assert result.actual_sha256 == result.expected_sha256 == _sha(PDF_BYTES)
    assert (tmp_path / "tubitak_mevzuat/031.pdf").read_bytes() == PDF_BYTES


def test_download_without_manifest_sha_is_accepted(tmp_path: Path) -> None:
    report = _fetch([_entry(sha256=None)], tmp_path, _serve(PDF_BYTES))
    assert report.results[0].status == FetchStatus.INDIRILDI
    assert report.results[0].actual_sha256 == _sha(PDF_BYTES)


def test_skips_existing_matching_file(tmp_path: Path) -> None:
    target = tmp_path / "tubitak_mevzuat/031.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(PDF_BYTES)
    calls: list[str] = []
    report = _fetch([_entry()], tmp_path, _serve(b"%PDF other", calls))
    assert report.results[0].status == FetchStatus.ATLANDI
    assert calls == []


def test_reports_mismatch_without_overwriting_existing(tmp_path: Path) -> None:
    target = tmp_path / "tubitak_mevzuat/031.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF local edited copy")
    calls: list[str] = []
    report = _fetch([_entry()], tmp_path, _serve(PDF_BYTES, calls))
    result = report.results[0]
    assert result.status == FetchStatus.UYUSMAZLIK
    assert result.actual_sha256 == _sha(b"%PDF local edited copy")
    assert result.message == "yerel dosya manifestle uyuşmuyor"
    assert target.read_bytes() == b"%PDF local edited copy"
    assert calls == []


def test_downloaded_file_with_different_sha_is_reported_as_updated_at_source(
    tmp_path: Path,
) -> None:
    report = _fetch([_entry()], tmp_path, _serve(b"%PDF-1.4 new version"))
    result = report.results[0]
    assert result.status == FetchStatus.UYUSMAZLIK
    assert "kaynakta güncellenmiş" in result.message
    assert (tmp_path / "tubitak_mevzuat/031.pdf").read_bytes() == b"%PDF-1.4 new version"


def test_force_downloads_even_when_file_exists(tmp_path: Path) -> None:
    target = tmp_path / "tubitak_mevzuat/031.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF old")
    report = _fetch([_entry()], tmp_path, _serve(PDF_BYTES), force=True)
    assert report.results[0].status == FetchStatus.INDIRILDI
    assert target.read_bytes() == PDF_BYTES


def test_html_is_trimmed_to_article(tmp_path: Path) -> None:
    entry = _entry(doc_id="sayfa-x", kind=SourceKind.HTML, local_path="p/html/x.html", sha256=None)
    _fetch([entry], tmp_path, _serve(HTML_PAGE.encode("utf-8")))
    saved = (tmp_path / "p/html/x.html").read_text(encoding="utf-8")
    assert saved == "<article><p>Destek oranı</p></article>"


def test_non_pdf_response_for_pdf_entry_is_an_error(tmp_path: Path) -> None:
    report = _fetch([_entry()], tmp_path, _serve(b"<html>WAF</html>"))
    assert report.results[0].status == FetchStatus.HATA
    assert not (tmp_path / "tubitak_mevzuat/031.pdf").exists()


def test_error_is_recorded_not_raised(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "bad" in str(request.url):
            raise httpx.ConnectError("bağlantı yok")
        return httpx.Response(200, content=PDF_BYTES)

    entries = [
        _entry(doc_id="a", url="https://x/bad.pdf", local_path="m/a.pdf"),
        _entry(doc_id="b", url="https://x/ok.pdf", local_path="m/b.pdf"),
    ]
    report = _fetch(entries, tmp_path, _client(handler))
    assert [r.status for r in report.results] == [FetchStatus.HATA, FetchStatus.INDIRILDI]
    assert "bağlantı yok" in report.results[0].message


def test_http_error_status_is_an_error(tmp_path: Path) -> None:
    client = _client(lambda request: httpx.Response(404))
    report = _fetch([_entry()], tmp_path, client)
    assert report.results[0].status == FetchStatus.HATA


def test_waits_between_downloads(tmp_path: Path) -> None:
    waits: list[float] = []
    entries = [_entry(doc_id=f"d{i}", local_path=f"m/{i}.pdf") for i in range(2)]
    fetch_all(entries, tmp_path, _serve(PDF_BYTES), sleep=waits.append)
    assert waits == [0.3, 0.3]


def test_report_counts() -> None:
    def result(status: FetchStatus) -> FetchResult:
        return FetchResult(
            doc_id="x", local_path="x", status=status, expected_sha256=None, actual_sha256=None
        )

    report = FetchReport(
        results=[result(FetchStatus.ATLANDI), result(FetchStatus.ATLANDI), result(FetchStatus.HATA)]
    )
    assert report.counts() == {
        "indirildi": 0,
        "atlandi": 2,
        "kaynakta_guncellenmis": 0,
        "hata": 1,
    }


def test_write_report(tmp_path: Path) -> None:
    report = FetchReport(results=[])
    path = tmp_path / "artifacts" / "fetch_report.json"
    write_report(report, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["results"] == [] and data["counts"]["hata"] == 0


@pytest.mark.parametrize("status", list(FetchStatus))
def test_status_values_are_turkish_keys(status: FetchStatus) -> None:
    assert status.value in {"indirildi", "atlandi", "kaynakta_guncellenmis", "hata"}


def test_make_client_sets_browser_user_agent() -> None:
    from tubitak_rag.ingestion.fetch import make_client

    with make_client() as client:
        assert client.headers["User-Agent"] == "Mozilla/5.0"
        assert client.follow_redirects is True


def test_download_leaves_no_partial_file(tmp_path: Path) -> None:
    _fetch([_entry()], tmp_path, _serve(PDF_BYTES))
    assert sorted(p.name for p in (tmp_path / "tubitak_mevzuat").iterdir()) == ["031.pdf"]


def test_failed_write_keeps_previous_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # M-4: the file is written to "<name>.part" and moved into place only when complete.
    target = tmp_path / "tubitak_mevzuat/031.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF old")

    def broken_replace(src: object, dst: object) -> None:
        raise OSError("disk dolu")

    monkeypatch.setattr("tubitak_rag.ingestion.fetch.os.replace", broken_replace)
    report = _fetch([_entry()], tmp_path, _serve(PDF_BYTES), force=True)
    assert report.results[0].status == FetchStatus.HATA
    assert "disk dolu" in report.results[0].message
    assert target.read_bytes() == b"%PDF old"
    assert not target.with_name("031.pdf.part").exists()
