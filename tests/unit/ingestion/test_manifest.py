"""Tests for merging the three corpus manifests into SourceEntry objects (FR-D5, FR-P10)."""

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from tubitak_rag.domain.enums import DocType, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.manifest import (
    ManifestError,
    doc_id_for,
    infer_doc_type,
    infer_program_code,
    load_sources,
    parse_tr_date,
)

PAGE_1501 = (
    "sayfa-sanayi__ulusal-destek-programlari__"
    "1501-tubitak-sanayi-ar-ge-projeleri-destekleme-programi"
)


@pytest.fixture
def sources(fixtures: Path) -> dict[str, SourceEntry]:
    return {e.doc_id: e for e in load_sources(fixtures / "manifests")}


def test_load_sources_merges_three_manifests(fixtures: Path) -> None:
    entries = load_sources(fixtures / "manifests")
    assert len(entries) == 8  # 2 mevzuat + 2 sayfa + 2 dosya + 2 kanun
    assert [e.kaynak_grubu for e in entries[:2]] == ["tubitak_mevzuat"] * 2
    assert entries[-1].kaynak_grubu == "sanayi_mevzuati"


def test_mevzuat_entry_fields(sources: dict[str, SourceEntry]) -> None:
    e = sources["mevzuat-031"]
    assert e.program_kodu == "1507"
    assert e.doc_turu == DocType.UYGULAMA_ESASLARI
    assert e.surum_tarihi == date(2026, 9, 25)
    assert e.kind == SourceKind.PDF
    assert e.local_path == "tubitak_mevzuat/031_1507_UE_25.09.2026_84YK_islenmis_hali.pdf"
    assert e.dil == "tr"
    assert e.kaynak_sayfa_url == "https://tubitak.gov.tr/tr/mevzuat"


def test_program_page_program_code_from_slug(sources: dict[str, SourceEntry]) -> None:
    e = sources[PAGE_1501]
    assert e.program_kodu == "1501"
    assert e.kind == SourceKind.HTML
    assert e.doc_turu == DocType.PROGRAM_SAYFASI
    assert e.local_path.startswith("tubitak_program_sayfalari/html/sanayi__")
    assert sources["sayfa-akademik"].program_kodu is None


def test_program_file_doc_type_from_kat(sources: dict[str, SourceEntry]) -> None:
    files = [e for e in sources.values() if e.doc_id.startswith("dosya-")]
    by_kind = {e.kind: e for e in files}
    cagri = by_kind[SourceKind.PDF]
    assert cagri.doc_turu == DocType.CAGRI
    assert cagri.program_kodu == "1501"
    assert cagri.surum_tarihi == date(2026, 7, 1)  # "dosyalar/2026-07__..." prefix
    assert cagri.kaynak_sayfa_url is not None
    assert cagri.kaynak_sayfa_url.startswith("https://tubitak.gov.tr/tr/destekler/sanayi/")
    table = by_kind[SourceKind.XLSX]
    assert table.doc_turu == DocType.TABLO
    assert table.dil == "tr"  # manifest has dil=null
    # Linked to six programme pages: no single programme code applies.
    assert table.program_kodu is None
    assert table.sha256 is not None
    assert table.doc_id == f"dosya-{table.sha256[:12]}"


def test_gib_entry(sources: dict[str, SourceEntry]) -> None:
    e = sources["kanun-5746"]
    assert e.kind == SourceKind.GIB_JSON
    assert e.doc_turu == DocType.KANUN
    assert e.url.startswith("https://gib.gov.tr/api/")
    assert e.local_path == "sanayi_mevzuati/5746_kanun_gib.json"
    assert e.kaynak_sayfa_url == "https://www.gib.gov.tr/mevzuat/kanun/485"
    assert e.surum_tarihi == date(2026, 6, 10)


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Ar-Ge Faaliyetlerinin Desteklenmesi Hakkında Kanun", DocType.KANUN),
        ("Teknoloji ve Yenilik Destek Programlarına İlişkin Yönetmelik", DocType.YONETMELIK),
        ("Yenilik Destek Programı Uygulama Esasları Yönergesi", DocType.YONERGE),
        ("Kamu İhtiyaçlarına Yönelik Destek Programları Yönetmeliği", DocType.YONETMELIK),
        ("1507- KOBİ Ar-Ge Başlangıç Destek Programı Uygulama Esasları", DocType.UYGULAMA_ESASLARI),
        ("1004 - Mükemmeliyet Merkezi Destek Programı Usul ve Esasları", DocType.UYGULAMA_ESASLARI),
        ("Araştırma Projelerinde Uygulanacak İdari ve Mali Esaslar", DocType.ESASLAR),
        (
            "Kaynaklardan 5018 Sayılı Kamu Malî Yönetimi ve Kontrol Kanununun Kapsamı Dışındaki "
            "İdarelere Aktarılan Tutarlara İlişkin Usul ve Esaslar",
            DocType.UYGULAMA_ESASLARI,
        ),
        ("Proje Pazarı Duyurusu", DocType.DIGER),
    ],
)
def test_infer_doc_type_table(title: str, expected: DocType) -> None:
    assert infer_doc_type(title, SourceKind.PDF, None) == expected


def test_infer_doc_type_kind_and_kat_take_precedence() -> None:
    assert infer_doc_type("Limitler", SourceKind.XLSX, "cagri") == DocType.TABLO
    assert infer_doc_type("Çağrı Metni Esasları", SourceKind.PDF, "cagri") == DocType.CAGRI
    assert infer_doc_type("Başvuru Kılavuzu", SourceKind.PDF, "kilavuz") == DocType.KILAVUZ
    assert infer_doc_type("1501", SourceKind.HTML, None) == DocType.PROGRAM_SAYFASI
    assert infer_doc_type("x", SourceKind.GIB_JSON, None) == DocType.KANUN


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1501 - TÜBİTAK Sanayi Ar-Ge", "1501"),
        ("ardeb_1001_basvuru_rehberi", "1001"),
        ("2209-A Üniversite Öğrencileri", "2209"),
        ("2026 Yılı Çağrı Duyurusu", None),  # years are not program codes
        ("Akademik", None),
    ],
)
def test_infer_program_code(text: str, expected: str | None) -> None:
    assert infer_program_code(text) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("25.09.2026", date(2026, 9, 25)),
        ("2026-06-10", date(2026, 6, 10)),
        (None, None),
        ("", None),
        ("31.02.2026", None),
    ],
)
def test_parse_tr_date(value: str | None, expected: date | None) -> None:
    assert parse_tr_date(value) == expected


def test_doc_id_for() -> None:
    assert doc_id_for("mevzuat", "031") == "mevzuat-031"
    assert doc_id_for("kanun", "5746") == "kanun-5746"


def test_doc_ids_unique(fixtures: Path) -> None:
    ids = [e.doc_id for e in load_sources(fixtures / "manifests")]
    assert len(set(ids)) == len(ids)


def test_manually_added_entry_is_loaded(fixtures: Path, tmp_path: Path) -> None:
    # FR-D5: a record added by hand to a manifest becomes a source on the next load.
    manifests = tmp_path / "manifests"
    shutil.copytree(fixtures / "manifests", manifests)
    path = manifests / "tubitak_mevzuat.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["dokumanlar"].append(
        {
            "baslik": "Elle Eklenen Destek Programı Uygulama Esasları",
            "url": "https://example.org/elle.pdf",
            "tarih": "01.10.2026",
            "dosya": "900_elle.pdf",
            "program_kodu": None,
            "kaynak_sayfa": "https://example.org",
        }
    )
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    entries = {e.doc_id: e for e in load_sources(manifests)}
    assert len(entries) == 9
    assert entries["mevzuat-900"].sha256 is None


def _with_mevzuat_entry(fixtures: Path, tmp_path: Path, entry: dict[str, object]) -> Path:
    manifests = tmp_path / "manifests"
    shutil.copytree(fixtures / "manifests", manifests)
    path = manifests / "tubitak_mevzuat.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["dokumanlar"].append(entry)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return manifests


def test_mevzuat_program_code_falls_back_to_file_name(fixtures: Path, tmp_path: Path) -> None:
    # I-6: the manifest leaves program_kodu empty for 1503, 1602, 1505, 1514, 1515, 1513.
    entry = {
        "baslik": "TÜBİTAK Proje Pazarları Destekleme Programı Uygulama Esasları",
        "url": "https://example.org/x.pdf",
        "tarih": "09.01.2024",
        "dosya": "034_09.01.2023_tarihli_ve_60yk_1503_proje_pazarlari_islenmis.pdf",
        "program_kodu": None,
    }
    sources = {e.doc_id: e for e in load_sources(_with_mevzuat_entry(fixtures, tmp_path, entry))}
    assert sources["mevzuat-034"].program_kodu == "1503"


def test_real_manifest_program_codes(repo_root: Path) -> None:
    sources = {e.doc_id: e for e in load_sources(repo_root / "data" / "manifests")}
    expected = {"034": "1503", "068": "1602", "070": "1505", "088": "1514", "097": "1515"}
    for key, code in {**expected, "103": "1513", "031": "1507"}.items():
        assert sources[f"mevzuat-{key}"].program_kodu == code
    assert sources["mevzuat-065"].program_kodu is None  # "5018" is a law number, not a programme


@pytest.mark.parametrize(
    "entry,missing",
    [
        ({"baslik": "Elle eklenen", "url": "https://x/y.pdf"}, "dosya"),
        ({"dosya": "900_x.pdf", "url": "https://x/y.pdf"}, "baslik"),
    ],
)
def test_invalid_hand_added_entry_gives_turkish_error(
    fixtures: Path, tmp_path: Path, entry: dict[str, object], missing: str
) -> None:
    manifests = _with_mevzuat_entry(fixtures, tmp_path, entry)
    with pytest.raises(ManifestError) as error:
        load_sources(manifests)
    message = str(error.value)
    assert "tubitak_mevzuat.json" in message and "3. kayıt" in message
    assert f"eksik alan: '{missing}'" in message
    assert isinstance(error.value, ValueError)
