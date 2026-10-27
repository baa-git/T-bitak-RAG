"""In-memory demo backend: every UI state without services (tests, design work, demos).

The answer excerpts are real article texts from the corpus (1507 and 1501 Uygulama Esasları);
the run metrics are illustrative numbers, not results. Pages say so via ``is_demo``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from tubitak_rag import __version__
from tubitak_rag.config import AppConfig, GuardrailConfig
from tubitak_rag.domain.answer import Answer, Citation, SourceView
from tubitak_rag.domain.enums import AnswerStatus, DocType, SourceKind
from tubitak_rag.domain.evaluation import RunMetrics
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.generation.prompts import BILGI_BULUNAMADI_MESAJI, KAPSAM_DISI_MESAJI
from tubitak_rag.guardrails.scope import ScopeChecker
from tubitak_rag.interfaces.gpu_lock import GpuLockError, busy_message
from tubitak_rag.interfaces.web.about import (
    component_lines,
    corpus_groups,
    describe_methods,
    guardrail_lines,
)
from tubitak_rag.interfaces.web.backend import UnknownPipelineError
from tubitak_rag.interfaces.web.comparison import GROUP_LABELS, METRICS, compare, summarize
from tubitak_rag.interfaces.web.rendering import number_tr, signed_tr
from tubitak_rag.interfaces.web.schemas import (
    AboutInfo,
    AskResult,
    DocumentPreview,
    DocumentsResponse,
    DocumentState,
    DocumentStatus,
    HealthStatus,
    JobStarted,
    PipelineOption,
    RunDetail,
    RunSummary,
    SectionView,
    SourceContext,
)

URL_1507 = (
    "https://tubitak.gov.tr/sites/default/files/2026-10/1507_UE_25.09.2026_84YK_islenmis_hali.pdf"
)
URL_1501 = "https://tubitak.gov.tr/sites/default/files/2026-01/77_YK_1501_islenmis_hali.pdf"
TITLE_1507 = "TÜBİTAK 1507- KOBİ Ar-Ge Başlangıç Destek Programı Uygulama Esasları"
TITLE_1501 = (
    "TÜBİTAK 1501- Sanayi Araştırma Teknoloji Geliştirme ve Yenilik Projeleri "
    "Destekleme Programı Uygulama Esasları"
)
DESTEK_KAPSAMI = "İkinci Bölüm – Destek Kapsamı"

EXAMPLE_ANSWERED = "1507 programında destek oranı en fazla yüzde kaçtır?"
EXAMPLE_NOT_FOUND = "2244 programında bursiyere ödenen aylık burs tutarı ne kadardır?"
EXAMPLE_OUT_OF_SCOPE = "İstanbul'da yarın hava nasıl olacak?"
EXAMPLES = (EXAMPLE_ANSWERED, EXAMPLE_NOT_FOUND, EXAMPLE_OUT_OF_SCOPE)

_ANSWER_TEXT = (
    "1507 KOBİ Ar-Ge Başlangıç Destek Programında uygulanacak destek oranı her dönem için en "
    "fazla %75'tir [1]. Projeye uygulanacak oran; değerlendirme puanı, bütçe, firma ölçeği ve "
    "projenin başarıyla tamamlanma durumu gibi ölçütlere göre belirlenir ve projenin destek "
    "kapsamında belirtilir [1].\n\n"
    "Destek, dönemsel olarak geri ödemesiz (hibe) biçimde verilir; proje bazında destek süresi "
    "en fazla 18 aydır [2]. Karşılaştırma için: 1501 programında KOBİ ölçeğindeki kuruluşlar "
    "için oran yine en fazla %75, büyük kuruluşlar için en fazla %60'tır [3]."
)
_SOURCES = (
    (
        "mevzuat-031:0012:00",
        "mevzuat-031",
        TITLE_1507,
        "Madde 7 (1)",
        URL_1507,
        SourceContext(
            hiyerarsi=DESTEK_KAPSAMI,
            madde_baslik="Madde 7 – Destek oranı ve sınırı",
            sayfa=6,
            surum_tarihi="2026-09-25",
            program_kodu="1507",
            doc_id="mevzuat-031",
            bolum_id="mevzuat-031:0012",
        ),
        7.84,
        "(1) Program kapsamında desteklenen projeler desteklemeye esas harcama tutarına "
        'uygulanacak "destek oranı" ile desteklenir. Bu programa başvuran projelerin azami '
        "bütçesi Yönetim Kurulu tarafından belirlenir. Belirlenen bütçede değişiklik yapıldığı "
        "durumlarda TÜBİTAK tarafından TEYDEB internet adresinde yayımlanır. Bu programda "
        "uygulanacak destek oranı her dönem için en fazla %75 olarak belirlenmiştir. Projelere "
        "uygulanacak destek oranları projenin değerlendirme puanı, bütçesi, firma ölçeği, "
        "projenin başarı ile tamamlanma durumu vb. kriterler dikkate alınarak belirlenir. Destek "
        "oranının belirlenmesinde kullanılacak kriterler ve yöntem TEYDEB Başkanının teklifi "
        "üzerine TÜBİTAK Başkanı onayı ile belirlenir. Projeye uygulanacak destek oranı projenin "
        "destek kapsamında belirtilir.\n\n"
        "(2) Desteklenen projelere, destek kararının alındığı yılda geçerli Yönetim Kurulu "
        "kararı ile alınmış olan bütçe sınırı uygulanır.",
    ),
    (
        "mevzuat-031:0011:00",
        "mevzuat-031",
        TITLE_1507,
        "Madde 6 (2)",
        URL_1507,
        SourceContext(
            hiyerarsi=DESTEK_KAPSAMI,
            madde_baslik="Madde 6 – Destekleme dönemi ve destek süresi",
            sayfa=5,
            surum_tarihi="2026-09-25",
            program_kodu="1507",
            doc_id="mevzuat-031",
            bolum_id="mevzuat-031:0011",
        ),
        4.12,
        "(1) Program kapsamında desteklenen projelerin, dönem tanımı içerisinde gerçekleşen "
        "faaliyetleri teknik ve mali yönden izlenir, değerlendirilir ve dönemsel olarak destek "
        "oranı uygulanarak geri ödemesiz (hibe) olarak desteklenir.\n\n"
        "(2) Projenin destek süresi, projede destek kapsamına alınan faaliyetlerin "
        "gerçekleştirilmesi için gereken ve TÜBİTAK tarafından kabul edilen süredir. Destek "
        "süresi proje bazında en fazla on sekiz (18) aydır.",
    ),
    (
        "mevzuat-030:0012:00",
        "mevzuat-030",
        TITLE_1501,
        "Madde 7 (1)",
        URL_1501,
        SourceContext(
            hiyerarsi=DESTEK_KAPSAMI,
            madde_baslik="Madde 7 – Destek oranı ve sınırı",
            sayfa=6,
            surum_tarihi="2025-12-29",
            program_kodu="1501",
            doc_id="mevzuat-030",
            bolum_id="mevzuat-030:0012",
        ),
        2.37,
        "(1) Desteklenen projeler desteklemeye esas harcama tutarına uygulanacak dönemsel "
        "destek oranı ile desteklenir. Destek oranı büyük kuruluşlar için en fazla % 60 ve KOBİ "
        "ölçeğindeki kuruluşlar içinse en fazla %75 olarak uygulanır. Projelere uygulanacak "
        "destek oranları projenin değerlendirme puanı, bütçesi, firma ölçeği, projenin başarı "
        "ile tamamlanma durumu vb. kriterler dikkate alınarak belirlenir.",
    ),
)


def _doc(
    doc_id: str,
    baslik: str,
    program_kodu: str | None,
    doc_turu: str,
    kaynak_url: str,
    surum_tarihi: str | None,
    durum: DocumentState,
    chunk_sayisi: int,
) -> DocumentStatus:
    return DocumentStatus(
        doc_id=doc_id,
        baslik=baslik,
        program_kodu=program_kodu,
        doc_turu=doc_turu,
        kaynak_url=kaynak_url,
        surum_tarihi=surum_tarihi,
        dil="tr",
        durum=durum,
        chunk_sayisi=chunk_sayisi,
    )


class _Run(NamedTuple):
    experiment: str
    started_at: str
    chunker: str
    embedder: str
    retriever: str
    reranker: str | None
    retrieval: tuple[float, float, float, float, float]
    generation: tuple[float, float, float, float, float, float]
    latency: tuple[float, float, float, float, float, float]


_DOCS = (
    _doc(
        "mevzuat-013",
        "TÜBİTAK Teknoloji ve Yenilik Destek Programlarına İlişkin Yönetmelik",
        None,
        "yonetmelik",
        "https://tubitak.gov.tr/sites/default/files/2026-08/TEYDEB_Yonetmeligi_2020_islenmis_hali.pdf",
        "2020-07-18",
        "indeksli",
        118,
    ),
    _doc(
        "mevzuat-030",
        TITLE_1501,
        "1501",
        "uygulama_esaslari",
        URL_1501,
        "2025-12-29",
        "indeksli",
        164,
    ),
    _doc(
        "mevzuat-031",
        TITLE_1507,
        "1507",
        "uygulama_esaslari",
        URL_1507,
        "2026-09-25",
        "indeksli",
        152,
    ),
    _doc(
        "mevzuat-032",
        "TÜBİTAK 1509 - Uluslararası Sanayi Ar-Ge Projeleri Destekleme Programı Uygulama Esasları",
        "1509",
        "uygulama_esaslari",
        "https://tubitak.gov.tr/sites/default/files/2026-03/05.03.2026_tarihli_Baskanlik_Olur_u_77YK_Yetki_1509_islenmis_hali.pdf",
        "2026-03-05",
        "guncel_degil",
        97,
    ),
    _doc(
        "mevzuat-048",
        "Araştırma Projelerinde Uygulanacak İdari ve Mali Esaslar",
        None,
        "esaslar",
        "https://tubitak.gov.tr/sites/default/files/2026-08/ARDEB_Idari_Mali_Esaslar_77YK_islenmis_1.pdf",
        "2025-12-29",
        "indeksli",
        211,
    ),
    _doc(
        "mevzuat-066",
        "1007 - Kamu Kurumları Araştırma ve Geliştirme Projelerini Destekleme "
        "Programı Usul ve Esasları",
        "1007",
        "esaslar",
        "https://tubitak.gov.tr/sites/default/files/2026-01/1007_Usul_ve_Esaslari_77_YK_islenmis_hali_SON.pdf",
        "2025-12-29",
        "indeksli",
        88,
    ),
    _doc(
        "mevzuat-073",
        "1003-Öncelikli Alanlar Ar-Ge Projeleri Destekleme Programı Usul ve Esasları",
        "1003",
        "esaslar",
        "https://tubitak.gov.tr/sites/default/files/2026-04/80_sayili_YK_karari_islenmis_hali.pdf",
        "2026-04-03",
        "indeksli",
        76,
    ),
    _doc(
        "mevzuat-079",
        "TÜBİTAK 1511 - Öncelikli Alanlar Araştırma Teknoloji Geliştirme ve "
        "Yenilik Projeleri Destekleme Programı Uygulama Esasları",
        "1511",
        "uygulama_esaslari",
        "https://tubitak.gov.tr/sites/default/files/2026-05/1511_10_YK_BaskanlikOlur_u_07.05.2026_938769_islenmis_0.pdf",
        "2026-05-07",
        "indeksli",
        131,
    ),
    _doc(
        "mevzuat-101",
        "TÜBİTAK 1512 - Girişimcilik Destek Programı Uygulama Esasları",
        "1512",
        "uygulama_esaslari",
        "https://tubitak.gov.tr/sites/default/files/2026-09/1512_67YK_BaskanlikYetkisi_25.10.2024_665159_BaskanlikOLURu_islenmis.pdf",
        "2024-10-25",
        "indeksli",
        104,
    ),
    _doc(
        "mevzuat-129",
        "TÜBİTAK 1812 - Yatırım Tabanlı Girişimcilik Destek Programı Uygulama Esasları",
        "1812",
        "uygulama_esaslari",
        "https://tubitak.gov.tr/sites/default/files/2026-01/1812_72YK_islenmis_hali.pdf",
        None,
        "taranmis",
        0,
    ),
    _doc(
        "mevzuat-064",
        "TÜBİTAK Tarafından Özel Kuruluşlar ve Vakıf Üniversitelerine Proje "
        "Karşılığı Aktarılacak Tutarlar",
        None,
        "esaslar",
        "https://tubitak.gov.tr/sites/default/files/mevzuat/tubitak_tarafindan_ozel_kuruluslar_ve_vakif_universitelerine_proje_karsiligi._yeni_2.pdf",
        "2014-04-08",
        "eksik",
        0,
    ),
    _doc(
        "kanun-5746",
        "Araştırma, Geliştirme ve Tasarım Faaliyetlerinin Desteklenmesi Hakkında Kanun",
        None,
        "kanun",
        "https://www.gib.gov.tr/mevzuat/kanun/485",
        "2008-03-12",
        "indeksli",
        39,
    ),
    _doc(
        "sayfa-1507",
        "1507 - KOBİ Ar-Ge Başlangıç Destek Programı (program sayfası)",
        "1507",
        "program_sayfasi",
        "https://tubitak.gov.tr/tr/destekler/sanayi/ulusal-destek-programlari/1507-tubitak-kobi-ar-ge-baslangic-destek-programi",
        None,
        "indeksli",
        14,
    ),
    _doc(
        "sayfa-1501",
        "1501 - TÜBİTAK Sanayi Ar-Ge Projeleri Destekleme Programı (program sayfası)",
        "1501",
        "program_sayfasi",
        "https://tubitak.gov.tr/tr/destekler/sanayi/ulusal-destek-programlari/1501-tubitak-sanayi-ar-ge-projeleri-destekleme-programi",
        None,
        "indeksli",
        17,
    ),
)

_RUNS = (
    _Run(
        "v0",
        "2026-10-14T09:12:41",
        "fixed",
        "google/embeddinggemma-2",
        "dense",
        None,
        (0.412, 0.537, 0.318, 0.581, 0.694),
        (0.447, 0.712, 0.538, 0.615, 0.143, 0.917),
        (41.0, 66.0, 0.0, 7310.0, 7420.0, 11280.0),
    ),
    _Run(
        "v0r",
        "2026-10-14T11:03:15",
        "fixed",
        "google/embeddinggemma-2",
        "dense",
        "BAAI/bge-reranker-v2-m3",
        (0.489, 0.537, 0.402, 0.643, 0.694),
        (0.506, 0.741, 0.592, 0.692, 0.131, 0.917),
        (43.0, 70.0, 238.0, 7180.0, 7536.0, 11410.0),
    ),
    _Run(
        "v1",
        "2026-10-14T13:47:02",
        "structural_ctx",
        "google/embeddinggemma-2",
        "dense",
        None,
        (0.553, 0.668, 0.437, 0.704, 0.806),
        (0.582, 0.788, 0.651, 0.769, 0.109, 0.917),
        (39.0, 63.0, 0.0, 6940.0, 7052.0, 10630.0),
    ),
    _Run(
        "v2",
        "2026-10-14T16:20:37",
        "structural_ctx",
        "google/embeddinggemma-2",
        "hybrid",
        None,
        (0.628, 0.749, 0.502, 0.765, 0.857),
        (0.633, 0.806, 0.689, 0.769, 0.098, 0.958),
        (58.0, 91.0, 0.0, 6890.0, 7021.0, 10410.0),
    ),
    _Run(
        "v3",
        "2026-10-15T09:05:18",
        "structural_ctx",
        "google/embeddinggemma-2",
        "hybrid",
        "BAAI/bge-reranker-v2-m3",
        (0.714, 0.812, 0.596, 0.837, 0.898),
        (0.694, 0.843, 0.752, 0.846, 0.087, 0.958),
        (61.0, 97.0, 244.0, 6720.0, 7108.0, 10270.0),
    ),
    _Run(
        "v3b",
        "2026-10-15T11:41:56",
        "structural_ctx",
        "BAAI/bge-m3",
        "hybrid",
        "BAAI/bge-reranker-v2-m3",
        (0.699, 0.806, 0.581, 0.827, 0.888),
        (0.684, 0.836, 0.744, 0.846, 0.098, 0.958),
        (73.0, 118.0, 246.0, 6750.0, 7166.0, 10390.0),
    ),
)
_CONCEPTS = {
    "v0": ("Sabit boyutlu parçalama", "Metni 512 belirteçlik eşit parçalara bölmek"),
    "v0r": ("Yeniden sıralama", "Çapraz kodlayıcı ile adayları yeniden sıralamak"),
    "v1": ("Yapı bazlı parçalama", "Madde sınırlarına göre bölmek ve bağlam başlığı eklemek"),
    "v2": (
        "Hibrit arama (RRF)",
        "Anlamsal arama ile BM25'i karşılıklı sıra füzyonuyla birleştirmek",
    ),
    "v3": ("Tam hat", "Hibrit arama ve yeniden sıralamayı birlikte kullanmak"),
    "v3b": ("Embedding karşılaştırması", "EmbeddingGemma 2 yerine bge-m3 kullanmak"),
}


def _latency(name: str) -> dict[str, float]:
    """Stage timings of the demo answer; pipelines without a reranker skip that stage."""
    timings = {"scope": 2.8, "retrieval": 63.4, "rerank": 241.7, "llm": 6812.5}
    if name not in _WITH_RERANKER:
        del timings["rerank"]
    return {**timings, "total": round(sum(timings.values()) + 11.5, 1)}


# The experiment ladder as config differences from the defaults (as in configs/experiments);
# the selector's method lines are derived from these objects, never written by hand.
_EXPERIMENTS: dict[str, tuple[str, dict[str, Any]]] = {
    "v0": (
        "Sabit 512 token / 64 örtüşmeli parçalar ve yalnızca dense (vektör) arama.",
        {
            "chunker": {"name": "fixed", "context_header": False},
            "retriever": {"mode": "dense"},
            "reranker": {"enabled": False},
        },
    ),
    "v0r": (
        "v0 ile aynı sabit parçalar, üstüne bge-reranker-v2-m3.",
        {"chunker": {"name": "fixed", "context_header": False}, "retriever": {"mode": "dense"}},
    ),
    "v1": (
        "Madde/bölüm sınırlarına göre parçalama ve her parçaya bağlam başlığı.",
        {"retriever": {"mode": "dense"}, "reranker": {"enabled": False}},
    ),
    "v2": (
        "BM25 ile dense aramayı RRF ile birleştiren hibrit arama.",
        {"reranker": {"enabled": False}},
    ),
    "v3": ("Yapı bazlı parçalama, hibrit arama ve cross-encoder reranker.", {}),
    "v3b": (
        "v3 ile aynı pipeline, EmbeddingGemma 2 yerine bge-m3.",
        {"embedder": {"model": "BAAI/bge-m3", "dimension": 1024}},
    ),
}


def _experiment_config(name: str) -> AppConfig:
    description, diff = _EXPERIMENTS[name]
    return AppConfig.model_validate(
        {"experiment": {"name": name, "description": description}, **diff}
    )


def _demo_entry(doc_id: str, group: str, kind: SourceKind, title: str) -> SourceEntry:
    return SourceEntry(
        doc_id=doc_id,
        title=title,
        url=f"https://tubitak.gov.tr/{doc_id}",
        local_path=f"{group}/{doc_id}",
        kind=kind,
        program_kodu=None,
        doc_turu=DocType.DIGER,
        surum_tarihi=None,
        dil="tr",
        sha256=None,
        kaynak_grubu=group,
        kaynak_sayfa_url=None,
    )


_LAWS = (
    ("kanun-5746", "Araştırma, Geliştirme ve Tasarım Faaliyetlerinin Desteklenmesi Hakkında Kanun"),
    ("kanun-4691", "Teknoloji Geliştirme Bölgeleri Kanunu"),
)


def _demo_corpus() -> list[SourceEntry]:
    """Manifest-shaped records with the sizes of the real corpus (60 + 105 + 115 + 2)."""
    mevzuat, pages, gib = "tubitak_mevzuat", "tubitak_program_sayfalari", "sanayi_mevzuati"
    entries = [_demo_entry(f"mevzuat-{i:03d}", mevzuat, SourceKind.PDF, "") for i in range(60)]
    entries += [_demo_entry(f"sayfa-{i}", pages, SourceKind.HTML, "") for i in range(105)]
    entries += [_demo_entry(f"dosya-{i}", pages, SourceKind.PDF, "") for i in range(114)]
    entries.append(_demo_entry("dosya-xlsx", pages, SourceKind.XLSX, ""))
    entries += [_demo_entry(doc_id, gib, SourceKind.GIB_JSON, title) for doc_id, title in _LAWS]
    return entries


_DEFAULT_PIPELINE = "v3"
_WITH_RERANKER = {"v0r", "v3", "v3b"}
_PREVIOUS = {"v0": None, "v0r": "v0", "v1": "v0", "v2": "v1", "v3": "v2", "v3b": "v3"}


def _metrics(row: _Run) -> RunMetrics:
    name, started, chunker, embedder, retriever, reranker, ret, gen, lat = row
    stamp = started.replace("-", "").replace(":", "").replace("T", "-")
    return RunMetrics(
        run_id=f"{name}_{stamp}",
        experiment=name,
        started_at=started,
        finished_at=started,  # demo data: the duration lives in wall_clock_s
        wall_clock_s=1980.0 + 37.0 * len(name),
        embedder=embedder,
        chunker=chunker,
        retriever=retriever,
        reranker=reranker,
        retrieval=dict(
            zip(
                ["recall_at_5", "recall_at_10", "mrr", "doc_hit_at_5", "doc_hit_at_10"],
                ret,
                strict=True,
            )
        ),
        generation=dict(
            zip(
                [
                    "answer_accuracy",
                    "faithfulness",
                    "citation_precision",
                    "correct_rejection_rate",
                    "false_rejection_rate",
                    "scope_rejection_rate",
                ],
                gen,
                strict=True,
            )
        ),
        latency_ms=dict(
            zip(
                [
                    "retrieval_p50",
                    "retrieval_p95",
                    "rerank_p50",
                    "llm_p50",
                    "total_p50",
                    "total_p95",
                ],
                lat,
                strict=True,
            )
        ),
        stage_times_s={"questions": 1210.0, "judge": 700.0, "report": 2.4},
        judge_model="qwen3:4b",
        llm_model="qwen3:4b",
        counts={"questions": 98, "chunks": 4812, "docs": 61},
        by_type={},
    )


def _metric_table(metrics: RunMetrics) -> list[str]:
    lines = ["| Grup | Metrik | Değer |", "|---|---|---:|"]
    for spec in METRICS:
        value = getattr(metrics, spec.group).get(spec.key)
        digits = 0 if spec.kind == "ms" else 3
        unit = " ms" if spec.kind == "ms" else ""
        lines.append(
            f"| {GROUP_LABELS[spec.group]} | {spec.label} | {number_tr(value, digits)}{unit} |"
        )
    return lines


def _report(metrics: RunMetrics, previous: RunMetrics | None) -> str:
    title, concept = _CONCEPTS[metrics.experiment]
    lines = [
        f"# {metrics.experiment} — {title}",
        "",
        "> Bu rapor arayüz tasarımı için üretilmiş **gösterim verisidir**; gerçek deney "
        "sonuçları Aşama 8B koşularından sonra burada görünür.",
        "",
        "## Bu deney ne öğretiyor?",
        "",
        f"{concept}. Basamak, deney merdiveninde yalnızca bir değişkeni değiştirir; böylece "
        "metriklerdeki fark o değişikliğe bağlanabilir.",
        "",
        "## Ayar",
        "",
        "```yaml",
        f"experiment:\n  name: {metrics.experiment}",
        f"chunker:\n  name: {metrics.chunker}",
        f"embedder:\n  model: {metrics.embedder}",
        f"retriever:\n  mode: {metrics.retriever}",
        f"reranker:\n  enabled: {'true' if metrics.reranker else 'false'}",
        "llm:\n  model: qwen3:4b\n  temperature: 0.0",
        "```",
        "",
        "## Sonuçlar",
        "",
        *_metric_table(metrics),
        "",
    ]
    if previous is not None:
        lines += [
            f"## Önceki basamakla karşılaştırma ({previous.experiment})",
            "",
            f"| Metrik | {previous.experiment} | {metrics.experiment} | Fark |",
            "|---|---:|---:|---:|",
        ]
        for _, rows in compare(previous, metrics).groups:
            for row in rows:
                digits = 0 if row.spec.kind == "ms" else 3
                lines.append(
                    f"| {row.spec.label} | {number_tr(row.a, digits)} | "
                    f"{number_tr(row.b, digits)} | {signed_tr(row.delta, digits)} |"
                )
        lines.append("")
    lines += [
        "## Sorgu akışı",
        "",
        "```mermaid",
        "flowchart LR",
        "  S[Soru] --> K{Kapsam}",
        "  K -->|içinde| A[Arama]",
        "  A --> R[Yeniden sıralama]",
        "  R --> G{Eşik}",
        "  G -->|geçti| L[Dil modeli]",
        "  G -->|geçmedi| B[Bilgi bulunamadı]",
        "```",
        "",
        "## Örnek bir soru",
        "",
        f"**Soru:** {EXAMPLE_ANSWERED}  ",
        "**Beklenen:** Her dönem için en fazla %75.  ",
        "**Hakem:** doğru, kaynağa sadık, atıf doğru.",
        "",
    ]
    return "\n".join(lines)


def _section(
    doc_id: str,
    order: int,
    madde: tuple[str | None, str | None],
    sayfa: int,
    metin: str,
    *,
    mulga: bool = False,
    notlar: tuple[str, ...] = (),
) -> SectionView:
    madde_no, baslik = madde
    return SectionView(
        bolum_id=f"{doc_id}:{order:04d}",
        baslik=baslik,
        hiyerarsi=[DESTEK_KAPSAMI] if order >= 10 else ["Birinci Bölüm – Genel Hükümler"],
        madde_no=madde_no,
        sayfa=sayfa,
        metin=metin,
        mulga=mulga,
        degisiklik_notlari=list(notlar),
    )


def _preview_sections(doc_id: str) -> list[SectionView]:
    """Demo sections; the two support articles reuse the real texts of the answer sources."""
    sections = [
        _section(
            doc_id,
            1,
            ("1", "Amaç"),
            1,
            "(1) Bu Uygulama Esaslarının amacı, programın uygulanmasına ilişkin usul ve "
            "esasları belirlemektir.",
        ),
        _section(
            doc_id,
            4,
            ("4", "Tanımlar"),
            2,
            "(1) Bu Uygulama Esaslarında geçen kısaltma ve tanımlar …",
            notlar=("(Değişik: 29/12/2025-77YK)",),
        ),
        _section(doc_id, 5, ("5", None), 3, "(Mülga: 29/12/2025-77YK)", mulga=True),
    ]
    texts = {source[5].bolum_id: source[7] for source in _SOURCES}
    articles = ((11, "6", "Destekleme dönemi ve destek süresi", 5), (12, "7", "Destek oranı", 6))
    for order, madde_no, baslik, sayfa in articles:
        text = texts.get(f"{doc_id}:{order:04d}")
        if text is not None:
            sections.append(_section(doc_id, order, (madde_no, baslik), sayfa, text))
    return sections


@dataclass
class FakeBackend:
    """Deterministic demo data for every page; no services, no GPU."""

    is_demo: bool = True
    gpu_busy: bool = False  # simulate a held GPU lock (409 / inline message)
    jobs: list[str] = field(default_factory=list)
    runs_started: list[tuple[str, int | None]] = field(default_factory=list)
    asked: list[tuple[str, str | None, str | None, str | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._runs = {m.run_id: m for m in map(_metrics, _RUNS)}
        self._scope = ScopeChecker("lexicon")
        self._disclaimer = GuardrailConfig().disclaimer

    def pipelines(self) -> list[PipelineOption]:
        """The experiment ladder v0 … v3b; v3 is the default."""
        options = []
        for name in _EXPERIMENTS:
            cfg = _experiment_config(name)
            options.append(
                PipelineOption(
                    ad=name,
                    etiket=name,
                    aciklama=cfg.experiment.description,
                    varsayilan=name == _DEFAULT_PIPELINE,
                    yontemler=describe_methods(cfg),
                )
            )
        return options

    def disclaimer(self) -> str:
        """The default disclaimer text."""
        return self._disclaimer

    def about(self) -> AboutInfo:
        """Demo corpus with the real corpus sizes; section/chunk counts are illustrative."""
        groups = corpus_groups(_demo_corpus())
        return AboutInfo(
            gruplar=groups,
            belge_sayisi=sum(g.belge_sayisi for g in groups),
            bolum_sayisi=5210,
            parca_sayisi=7017,
            koleksiyon="structural_ctx__google-embeddinggemma-2",
            bilesenler=component_lines([_experiment_config(name) for name in _EXPERIMENTS]),
            guardrailler=guardrail_lines(_experiment_config(_DEFAULT_PIPELINE)),
            yapilandirmalar=self.pipelines(),
        )

    def _resolve(self, name: str | None) -> str:
        if name is None:
            return _DEFAULT_PIPELINE
        if name not in _EXPERIMENTS:
            raise UnknownPipelineError(f"Bilinmeyen yapılandırma: {name}")
        return name

    def _check_gpu(self) -> None:
        if self.gpu_busy:
            raise GpuLockError(busy_message(Path("artifacts/.gpu.lock"), "rag run (v3b) demo"))

    def _answer(self, soru: str, durum: AnswerStatus, cevap: str, total: float) -> Answer:
        return Answer(
            soru=soru,
            durum=durum,
            cevaplanabilir=durum is AnswerStatus.CEVAPLANDI,
            cevap=cevap,
            atiflar=[],
            kaynaklar=[],
            uyari=self._disclaimer,
            gecikme_ms={"total": total},
            llm_cagri_sayisi=0,
            gate_key=None,
            gate_score=None,
        )

    def ask(
        self,
        soru: str,
        program_kodu: str | None,
        doc_turu: str | None,
        yapilandirma: str | None = None,
    ) -> AskResult:
        """Answered for support-rate questions, out of scope / not found otherwise."""
        name = self._resolve(yapilandirma)
        self._check_gpu()
        self.asked.append((soru, program_kodu, doc_turu, yapilandirma))
        lowered = soru.casefold()
        if not self._scope.check(soru).in_scope:
            answer = self._answer(soru, AnswerStatus.KAPSAM_DISI, KAPSAM_DISI_MESAJI, 3.1)
            return AskResult(answer=answer, yapilandirma=name)
        if "destek oran" not in lowered:
            answer = self._answer(
                soru, AnswerStatus.BILGI_BULUNAMADI, BILGI_BULUNAMADI_MESAJI, 412.6
            )
            gated = answer.model_copy(update={"gate_key": "rerank"})
            return AskResult(answer=gated, yapilandirma=name)
        sources = [
            SourceView(
                no=no,
                chunk_id=cid,
                doc_title=title,
                konum=konum,
                kaynak_url=url,
                text=text,
                score=score,
            )
            for no, (cid, _, title, konum, url, _, score, text) in enumerate(_SOURCES, start=1)
        ]
        citations = [
            Citation(
                no=no, chunk_id=cid, doc_id=doc_id, doc_title=title, konum=konum, kaynak_url=url
            )
            for no, (cid, doc_id, title, konum, url, *_) in enumerate(_SOURCES, start=1)
        ]
        answer = Answer(
            soru=soru,
            durum=AnswerStatus.CEVAPLANDI,
            cevaplanabilir=True,
            cevap=_ANSWER_TEXT,
            atiflar=citations,
            kaynaklar=sources,
            uyari=self._disclaimer,
            gecikme_ms=_latency(name),
            llm_cagri_sayisi=1,
            gate_key="rerank" if name in _WITH_RERANKER else "dense",
            gate_score=7.84,
        )
        baglam = {source[0]: source[5] for source in _SOURCES}
        return AskResult(answer=answer, baglam=baglam, yapilandirma=name)

    def documents(self) -> DocumentsResponse:
        """The demo corpus table."""
        docs = list(_DOCS)
        return DocumentsResponse(
            belgeler=docs,
            koleksiyon="structural_ctx__google-embeddinggemma-2",
            toplam_chunk=sum(d.chunk_sayisi for d in docs),
        )

    def document(self, doc_id: str) -> DocumentPreview | None:
        """Demo preview: 1507 and 1501 have parsed sections, the rest only metadata."""
        doc = next((d for d in _DOCS if d.doc_id == doc_id), None)
        if doc is None:
            return None
        sections = _preview_sections(doc_id) if doc_id in {"mevzuat-030", "mevzuat-031"} else []
        counts = {"structural_ctx__google-embeddinggemma-2": doc.chunk_sayisi}
        if sections:
            counts["fixed__google-embeddinggemma-2"] = round(doc.chunk_sayisi * 0.8)
        return DocumentPreview(
            belge=doc,
            kaynak_sayfa_url=None,
            ayristirildi=bool(sections),
            bolumler=sections,
            chunk_sayilari=counts if doc.chunk_sayisi else {},
        )

    def start_ingest(self, full: bool = False) -> JobStarted:
        """Record the request; nothing actually runs."""
        self._check_gpu()
        job_id = f"ingest-{len(self.jobs) + 1:04d}"
        self.jobs.append(job_id)
        return JobStarted(job_id=job_id)

    def start_run(self, config: str, limit: int | None = None) -> JobStarted:
        """Record the request; nothing actually runs."""
        name = self._resolve(config)
        self._check_gpu()
        self.runs_started.append((name, limit))
        return JobStarted(job_id=f"run-{len(self.runs_started):04d}")

    def runs(self) -> list[RunSummary]:
        """Demo runs, newest first."""
        ordered = sorted(self._runs.values(), key=lambda m: m.started_at, reverse=True)
        return [summarize(m) for m in ordered]

    def run_metrics(self, run_id: str) -> RunMetrics | None:
        """Metrics of a demo run."""
        return self._runs.get(run_id)

    def run_detail(self, run_id: str) -> RunDetail | None:
        """Demo runs have metrics but no per-question rows."""
        metrics = self._runs.get(run_id)
        return None if metrics is None else RunDetail(metrics=metrics, sorular=[])

    def health(self) -> HealthStatus:
        """The demo needs no service; it reports none as reachable."""
        return HealthStatus(
            durum="ok",
            qdrant=False,
            ollama=False,
            koleksiyon="structural_ctx__google-embeddinggemma-2",
            surum=__version__,
        )

    def run_report(self, run_id: str) -> str | None:
        """Generated Markdown report of a demo run."""
        metrics = self._runs.get(run_id)
        if metrics is None:
            return None
        previous_name = _PREVIOUS[metrics.experiment]
        previous = next((m for m in self._runs.values() if m.experiment == previous_name), None)
        return _report(metrics, previous)

    def summary_report(self) -> str | None:
        """All demo runs side by side."""
        lines = [
            "# Deney merdiveni özeti",
            "",
            "> Gösterim verisi; gerçek sonuçlar Aşama 8B koşularından sonra görünür.",
            "",
            "| Koşu | Kavram | Recall@5 | MRR | Cevap doğruluğu | Doğru ret | Toplam p50 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for m in sorted(self._runs.values(), key=lambda m: m.started_at):
            lines.append(
                f"| {m.experiment} | {_CONCEPTS[m.experiment][0]} | "
                f"{number_tr(m.retrieval['recall_at_5'])} | {number_tr(m.retrieval['mrr'])} | "
                f"{number_tr(m.generation['answer_accuracy'])} | "
                f"{number_tr(m.generation['correct_rejection_rate'])} | "
                f"{number_tr(m.latency_ms['total_p50'], 0)} ms |"
            )
        return "\n".join(lines) + "\n"
