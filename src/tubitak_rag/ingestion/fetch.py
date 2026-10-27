"""Download the corpus listed in the manifests and verify every file with SHA-256.

``rag fetch`` is idempotent: a file that already exists locally is not downloaded again; its
digest is only compared with the manifest (FR-D1, FR-D2). A digest that differs from the
manifest means the publisher changed the document; it is reported as "kaynakta
güncellenmiş" and never silently accepted (FR-D3).
"""

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel

from tubitak_rag.domain.enums import SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.hashing import sha256_of

USER_AGENT = "Mozilla/5.0"
TIMEOUT_S = 120.0
DELAY_S = 0.3
UPDATED_MESSAGE = "kaynakta güncellenmiş; manifest sha256 güncellenmeli"
LOCAL_MISMATCH_MESSAGE = "yerel dosya manifestle uyuşmuyor"


class FetchStatus(StrEnum):
    """Outcome of fetching one manifest entry (values are report keys)."""

    INDIRILDI = "indirildi"
    ATLANDI = "atlandi"
    UYUSMAZLIK = "kaynakta_guncellenmis"
    HATA = "hata"


class FetchResult(BaseModel):
    """Verification record of one file."""

    doc_id: str
    local_path: str
    status: FetchStatus
    expected_sha256: str | None
    actual_sha256: str | None
    message: str = ""


class FetchReport(BaseModel):
    """All results of one ``rag fetch`` run."""

    results: list[FetchResult]

    def counts(self) -> dict[str, int]:
        """Number of results per status, including zero counts."""
        return {s.value: sum(r.status == s for r in self.results) for s in FetchStatus}


def make_client() -> httpx.Client:
    """HTTP client with the headers and timeout the source sites expect."""
    return httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S, follow_redirects=True
    )


def _article_only(body: bytes) -> bytes:
    """Keep only ``<article>…</article>`` of a page (drops menus, footers, scripts)."""
    text = body.decode("utf-8", errors="replace")
    start, end = text.find("<article"), text.rfind("</article>")
    if start < 0 or end < 0:
        return body
    return text[start : end + len("</article>")].encode("utf-8")


def _download(entry: SourceEntry, client: httpx.Client, html_article_only: bool) -> bytes:
    response = client.get(entry.url)
    response.raise_for_status()
    body = response.content
    if entry.kind == SourceKind.PDF and not body.startswith(b"%PDF"):
        raise ValueError("yanıt bir PDF değil (%PDF imzası yok)")
    if entry.kind == SourceKind.HTML and html_article_only:
        body = _article_only(body)
    return body


def _result(entry: SourceEntry, status: FetchStatus, actual: str | None, msg: str) -> FetchResult:
    return FetchResult(
        doc_id=entry.doc_id,
        local_path=entry.local_path,
        status=status,
        expected_sha256=entry.sha256,
        actual_sha256=actual,
        message=msg,
    )


def _verify(entry: SourceEntry, actual: str, ok: FetchStatus) -> FetchResult:
    if entry.sha256 is None:
        return _result(entry, ok, actual, "manifestte sha256 yok")
    if actual == entry.sha256:
        return _result(entry, ok, actual, "")
    # A downloaded file that differs means the publisher changed it; a local file that
    # differs was not re-downloaded, so only the local copy is known to disagree.
    message = UPDATED_MESSAGE if ok == FetchStatus.INDIRILDI else LOCAL_MISMATCH_MESSAGE
    return _result(entry, FetchStatus.UYUSMAZLIK, actual, message)


def _write_atomically(target: Path, body: bytes) -> None:
    """Write to ``<name>.part`` and move it into place, so a crash never leaves half a file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        partial.write_bytes(body)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def fetch_all(
    entries: Sequence[SourceEntry],
    raw_dir: Path,
    client: httpx.Client,
    *,
    force: bool = False,
    html_article_only: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchReport:
    """Fetch every entry into ``raw_dir`` and verify it; errors are recorded, not raised.

    An existing file is kept unless ``force`` is set: a digest mismatch on an existing file is
    reported without downloading, so a local copy is never overwritten by surprise.
    """
    results: list[FetchResult] = []
    for entry in entries:
        target = raw_dir / entry.local_path
        if target.exists() and not force:
            results.append(_verify(entry, sha256_of(target), FetchStatus.ATLANDI))
            continue
        try:
            body = _download(entry, client, html_article_only)
            sleep(DELAY_S)
        except (httpx.HTTPError, ValueError) as exc:
            results.append(_result(entry, FetchStatus.HATA, None, str(exc)))
            continue
        try:
            _write_atomically(target, body)
        except OSError as exc:
            results.append(_result(entry, FetchStatus.HATA, None, str(exc)))
            continue
        results.append(_verify(entry, hashlib.sha256(body).hexdigest(), FetchStatus.INDIRILDI))
    return FetchReport(results=results)


def write_report(report: FetchReport, path: Path) -> None:
    """Save the report (with counts) as JSON, e.g. ``artifacts/fetch_report.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"counts": report.counts(), **report.model_dump(mode="json")}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
