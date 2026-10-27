"""Merge the three corpus manifests in ``data/manifests`` into one list of ``SourceEntry``.

Manifests are hand-curated JSON files kept in the repository (FR-D4); adding a record to one
of them is enough to bring a new document into the corpus (FR-D5). Each source group has its
own JSON shape, so each gets a small converter below.
"""

import json
import re
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from tubitak_rag.domain.enums import DocType, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.normalize import tr_lower

MEVZUAT = "tubitak_mevzuat"
PROGRAMLAR = "tubitak_program_sayfalari"
SANAYI = "sanayi_mevzuati"
PROGRAM_PAGE_BASE = "https://tubitak.gov.tr/tr/destekler/"

# Program codes are four digits; 20xx is excluded because those are years, not programmes
# (BİDEB codes start at 2201). Digit look-arounds also match inside "ardeb_1001_rehber".
PROGRAM_CODE_RE = re.compile(r"(?<!\d)(1\d{3}|2[1-9]\d{2}|3\d{3}|4\d{3}|5\d{3})(?!\d)")
FILE_MONTH_RE = re.compile(r"(?:^|/)(\d{4})-(\d{2})__")

# Keyword (Turkish lower case) -> type. The keyword that appears *last* in a title wins,
# because the document type noun ends the title ("... Uygulama Esasları Yönergesi").
_TYPE_KEYWORDS: tuple[tuple[str, DocType], ...] = (
    ("uygulama esasları", DocType.UYGULAMA_ESASLARI),
    ("usul ve esaslar", DocType.UYGULAMA_ESASLARI),
    ("esas ve usul", DocType.UYGULAMA_ESASLARI),
    ("esaslar", DocType.ESASLAR),
    ("yönetmeli", DocType.YONETMELIK),  # stem: Yönetmelik / Yönetmeliği
    ("yönerge", DocType.YONERGE),
    ("kanun", DocType.KANUN),
)
_KAT_TYPES = {"cagri": DocType.CAGRI, "kilavuz": DocType.KILAVUZ}


def infer_doc_type(title: str, kind: SourceKind, kat: str | None) -> DocType:
    """Guess the document type from its kind, its manifest category and its title."""
    if kind == SourceKind.XLSX:
        return DocType.TABLO
    if kat in _KAT_TYPES:
        return _KAT_TYPES[kat]
    if kind == SourceKind.HTML:
        return DocType.PROGRAM_SAYFASI
    if kind == SourceKind.GIB_JSON:
        return DocType.KANUN
    lowered = tr_lower(title)
    best: tuple[int, int, DocType] | None = None
    for keyword, doc_type in _TYPE_KEYWORDS:
        position = lowered.rfind(keyword)
        if position < 0:
            continue
        candidate = (position + len(keyword), len(keyword), doc_type)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best else DocType.DIGER


def infer_program_code(text: str) -> str | None:
    """Return the first programme code ("1501", "2209") found in ``text``."""
    match = PROGRAM_CODE_RE.search(text)
    return match.group(1) if match else None


def parse_tr_date(value: str | None) -> date | None:
    """Parse "25.09.2026" or "2026-06-10"; return None for empty or invalid values."""
    if not value:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def doc_id_for(group: str, key: str) -> str:
    """Build a stable document id such as ``mevzuat-031`` or ``kanun-5746``."""
    return f"{group}-{key}"


def _last_slug_part(slug: str) -> str:
    return slug.rsplit("__", 1)[-1]


def _mevzuat(doc: Mapping[str, Any]) -> SourceEntry:
    dosya = doc["dosya"]
    return SourceEntry(
        doc_id=doc_id_for("mevzuat", dosya[:3]),
        title=doc["baslik"],
        url=doc["url"],
        local_path=f"{MEVZUAT}/{dosya}",
        kind=SourceKind.PDF,
        # The file name carries the code when the manifest does not ("034_…_1503_proje…").
        program_kodu=doc.get("program_kodu") or infer_program_code(dosya),
        doc_turu=infer_doc_type(doc["baslik"], SourceKind.PDF, None),
        surum_tarihi=parse_tr_date(doc.get("tarih")),
        dil="tr",
        sha256=doc.get("sha256"),
        kaynak_grubu=MEVZUAT,
        kaynak_sayfa_url=doc.get("kaynak_sayfa"),
    )


def _program_page(page: Mapping[str, Any]) -> SourceEntry:
    slug = page["slug"]
    return SourceEntry(
        doc_id=doc_id_for("sayfa", slug),
        title=page["baslik"],
        url=page["url"],
        local_path=f"{PROGRAMLAR}/{page['html_dosya']}",
        kind=SourceKind.HTML,
        program_kodu=infer_program_code(_last_slug_part(slug)),
        doc_turu=DocType.PROGRAM_SAYFASI,
        surum_tarihi=None,
        dil="tr",
        sha256=page.get("sha256"),
        kaynak_grubu=PROGRAMLAR,
        kaynak_sayfa_url=page["url"],
    )


def _file_month(dosya: str) -> date | None:
    match = FILE_MONTH_RE.search(dosya)
    return date(int(match.group(1)), int(match.group(2)), 1) if match else None


def _program_file(item: Mapping[str, Any]) -> SourceEntry:
    kind = SourceKind.XLSX if item["ext"] == ".xlsx" else SourceKind.PDF
    pages: list[str] = item["sayfalar"]
    page_codes = {infer_program_code(_last_slug_part(page)) for page in pages} - {None}
    # A file shared by several programme pages (e.g. the limits table) gets no single code.
    code = infer_program_code(item["baslik"]) or (
        page_codes.pop() if len(page_codes) == 1 else None
    )
    return SourceEntry(
        doc_id=doc_id_for("dosya", item["sha256"][:12]),
        title=item["baslik"],
        url=item["url"],
        local_path=f"{PROGRAMLAR}/{item['dosya']}",
        kind=kind,
        program_kodu=code,
        doc_turu=infer_doc_type(item["baslik"], kind, item.get("kat")),
        surum_tarihi=_file_month(item["dosya"]),
        dil=item.get("dil") or "tr",
        sha256=item["sha256"],
        kaynak_grubu=PROGRAMLAR,
        kaynak_sayfa_url=PROGRAM_PAGE_BASE + pages[0].replace("__", "/"),
    )


def _law(doc: Mapping[str, Any]) -> SourceEntry:
    return SourceEntry(
        doc_id=doc_id_for("kanun", doc["kanun_no"]),
        title=doc["baslik"],
        url=doc["api"],
        local_path=f"{SANAYI}/{doc['dosya']}",
        kind=SourceKind.GIB_JSON,
        program_kodu=None,
        doc_turu=DocType.KANUN,
        surum_tarihi=parse_tr_date(doc.get("kaynakta_son_guncelleme")),
        dil="tr",
        sha256=doc.get("sha256"),
        kaynak_grubu=SANAYI,
        kaynak_sayfa_url=doc.get("kaynak"),
    )


class ManifestError(ValueError):
    """A manifest record cannot be turned into a source (message in Turkish)."""


def _convert[T](
    manifest: str, items: list[Mapping[str, Any]], convert: Callable[[Mapping[str, Any]], T]
) -> list[T]:
    """Convert every record; a broken (e.g. hand-edited) record names itself in the error."""
    result: list[T] = []
    for number, item in enumerate(items, start=1):
        try:
            result.append(convert(item))
        except KeyError as exc:
            raise ManifestError(
                f"{manifest}: {number}. kayıt geçersiz ({_describe(item)}); "
                f"eksik alan: {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError, IndexError) as exc:
            raise ManifestError(
                f"{manifest}: {number}. kayıt geçersiz ({_describe(item)}): {exc}"
            ) from exc
    return result


def _describe(item: Mapping[str, Any]) -> str:
    return str(item.get("baslik") or item.get("dosya") or item.get("slug") or "adsız")


def _read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_sources(manifests_dir: Path) -> list[SourceEntry]:
    """Read the three manifests in order: legislation, programme pages + files, laws.

    ``elenenler`` (excluded files) and ``erisilemeyen_sayfalar`` (unreachable pages) are
    bookkeeping lists and are ignored.
    """
    mevzuat = _read(manifests_dir / f"{MEVZUAT}.json")
    programs = _read(manifests_dir / f"{PROGRAMLAR}.json")
    laws = _read(manifests_dir / f"{SANAYI}.json")
    return [
        *_convert(f"{MEVZUAT}.json", mevzuat["dokumanlar"], _mevzuat),
        *_convert(f"{PROGRAMLAR}.json", programs["sayfalar"], _program_page),
        *_convert(f"{PROGRAMLAR}.json", programs["dosyalar"], _program_file),
        *_convert(f"{SANAYI}.json", laws["dokumanlar"], _law),
    ]
