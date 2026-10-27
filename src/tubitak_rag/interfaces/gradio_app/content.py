"""What the Gradio page says: its texts, the answer/refusal model and the Hakkında sections.

Escaped HTML snippets and plain strings (no Gradio import; unit-tested directly). The texts
and the structure follow the web templates (``templates/partials/tur.html``,
``templates/hakkinda.html``) and reuse their helpers (``web/rendering.py``, ``web/about.py``),
so both interfaces say the same thing. Every value that comes from a model, a config or a
manifest is escaped here; ``view.py`` lays the snippets out.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from markupsafe import Markup

from tubitak_rag.domain.answer import Answer, SourceView
from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.interfaces.web.about import DOCS_URL, REPO_URL, methods_text
from tubitak_rag.interfaces.web.rendering import (
    cited_sources,
    link_citations,
    location_parts,
    number_tr,
    plain_paragraphs,
    seconds_tr,
)
from tubitak_rag.interfaces.web.schemas import AboutInfo, AskResult, MethodLine, PipelineOption
from tubitak_rag.pipeline import LLM_HATASI_MESAJI

TITLE = "TÜBİTAK Mevzuat Rehberi"
CHAT_TITLE = "Mevzuata soru sorun"
CHAT_INTRO = (
    "Cevaplar yalnızca indekslenen TÜBİTAK ve Ar-Ge mevzuatına dayanır. Her cevabın altında "
    "dayandığı kaynaklar numaralarıyla listelenir."
)
HOW_IT_WORKS = (
    "Soru, belgelerde anlam ve anahtar kelime aramasıyla aranır; bulunan parçalar yeniden "
    "sıralanır. Yeterince ilgili parça yoksa sistem tahmin yürütmez, bilgi bulunamadığını "
    "söyler. Dil modeli yalnızca bu parçalara dayanarak yazar ve her bilgiye kaynak numarası "
    "ekler."
)
PLACEHOLDER = "Örneğin: 1507 programında destek oranı en fazla yüzde kaçtır?"
FOOTER = (
    "Açık kaynak eğitim projesi; resmî bir TÜBİTAK hizmeti değildir. Cevaplar yalnızca "
    "indekslenen belgelere dayanır."
)
DEMO_NOTE = (
    "Bu sayfalar gösterim verisiyle çalışıyor (sahte arka uç). Gerçek sistem için "
    "TUBITAK_RAG_BACKEND=pipeline kullanın."
)
NOT_FOUND = "Bu konuda elimdeki dokümanlarda bilgi bulunamadı."
OUT_OF_SCOPE_NOTE = (
    "Bu araç yalnızca TÜBİTAK destek programları, Ar-Ge ve tasarım merkezleri ile teknoloji "
    "geliştirme bölgelerine ilişkin mevzuatı yanıtlar."
)
CITATION_REJECTED_NOTE = "Üretilen cevap atıf denetiminden geçmediği için gösterilmiyor."
BELOW_THRESHOLD_NOTE = (
    "Soruyu destekleyecek kadar ilgili bir mevzuat parçası bulunamadı; sistem tahmin yürütmek "
    "yerine cevap vermemeyi seçti."
)
RETRY_NOTE = (
    "Soruyu farklı sözcüklerle ya da ilgili program ve madde adıyla yeniden sormayı "
    "deneyebilirsiniz."
)
RERANKER_SPEED_NOTE = (
    "Bu yapılandırma reranker kullanır; ücretsiz CPU'da cevap belirgin biçimde daha uzun "
    "sürer. Hızlı bir cevap için reranker'sız bir yapılandırma (ör. v2) seçin."
)
NO_INDEX_NOTE = "indeks yok"

# --- configuration selector ----------------------------------------------------------------


def option_label(option: PipelineOption) -> str:
    """Selector text: ``"v3"``, or ``"v1 (indeks yok)"`` when its index is missing."""
    if option.hazir:
        return option.etiket
    return f"{option.etiket} ({option.durum_notu or NO_INDEX_NOTE})"


def default_option(options: Sequence[PipelineOption]) -> str:
    """Name of the default option (the first one when none is marked)."""
    return next((o.ad for o in options if o.varsayilan), options[0].ad if options else "")


def uses_reranker(lines: Iterable[MethodLine]) -> bool:
    """Whether the method lines name a reranker."""
    return any(line.etiket == "Reranker" and line.deger != "yok" for line in lines)


def speed_note(option: PipelineOption, cpu_only: bool) -> str | None:
    """The CPU speed note of a reranker config (None on a GPU host or without reranker)."""
    if cpu_only and uses_reranker(option.yontemler):
        return RERANKER_SPEED_NOTE
    return None


# --- one conversation turn ----------------------------------------------------------------


@dataclass(frozen=True)
class SourceItem:
    """One numbered source under an answer (``view.py`` lays it out)."""

    no: int
    anchor: str  # HTML id the [n] links of the answer point to
    repaired: bool  # added by the number guardrail's citation repair
    title: str  # document title
    location: str  # "Madde 7 – Destek oranı · sayfa 6 · Üçüncü Bölüm"
    url: str  # the original document
    quote_html: Markup  # the passage as escaped paragraphs


@dataclass(frozen=True)
class AnswerView:
    """An answer (or refusal) ready for the page."""

    header_html: Markup  # "Cevap · v3 · 2,1 sn"
    body_html: Markup  # answer text with linked citations, or the refusal explanation
    sources: list[SourceItem]
    answered: bool


def _header(answer: Answer, config: str | None, methods: str) -> Markup:
    parts = [Markup('<span class="mesaj-kim">Cevap</span>')]
    if config:
        parts.append(
            Markup('<span class="yapilandirma-adi" title="{}">{}</span>').format(methods, config)
        )
    parts.append(
        Markup('<span class="sure">{}</span>').format(seconds_tr(answer.gecikme_ms.get("total", 0)))
    )
    separator = Markup('<span class="ayrac" aria-hidden="true">·</span>')
    return Markup('<p class="cevap-ust">{}</p>').format(separator.join(parts))


def _source_item(source: SourceView, result: AskResult, prefix: str) -> SourceItem:
    context = result.baglam.get(source.chunk_id)
    location = " · ".join(location_parts(source.konum, context))
    return SourceItem(
        no=source.no,
        anchor=f"kaynak-{prefix}{source.no}",
        repaired=source.no in result.answer.atif_onarimi,
        title=source.doc_title,
        location=location,
        url=source.kaynak_url,
        quote_html=plain_paragraphs(source.text),
    )


def _refusal(answer: Answer) -> Markup:
    if answer.durum == AnswerStatus.KAPSAM_DISI:
        return Markup('<p class="durum-mesaji">{}</p><p class="aciklama">{}</p>').format(
            answer.cevap, OUT_OF_SCOPE_NOTE
        )
    if answer.cevap == LLM_HATASI_MESAJI:  # the LLM call failed: say so, not "not found"
        return Markup('<p class="durum-mesaji">{}</p>').format(answer.cevap)
    reason = (
        CITATION_REJECTED_NOTE
        if answer.durum == AnswerStatus.ATIF_REDDI
        else (BELOW_THRESHOLD_NOTE)
    )
    return Markup('<p class="durum-mesaji">{}</p><p class="aciklama">{} {}</p>').format(
        NOT_FOUND, reason, RETRY_NOTE
    )


def answer_view(result: AskResult, prefix: str, methods: str = "") -> AnswerView:
    """The reply of one turn: header, text with ``[n]`` links (or refusal) and sources.

    As on the web page only the cited sources are listed, under their original numbers.
    """
    answer = result.answer
    header = _header(answer, result.yapilandirma, methods)
    answered = answer.durum == AnswerStatus.CEVAPLANDI and bool(answer.kaynaklar)
    if not answered:
        return AnswerView(header, _refusal(answer), [], answered=False)
    sources = cited_sources(answer)
    text = link_citations(answer.cevap, [s.no for s in sources], prefix)
    body = Markup('<div class="cevap-metni">{}</div>').format(text)
    if answer.atif_onarimi:
        numbers = ", ".join(f"[{no}]" for no in answer.atif_onarimi)
        body += Markup(
            '<p class="onarim-notu">{} numaralı atıf guardrail tarafından eklendi: '
            "cümledeki sayı yalnızca bu kaynakta geçiyordu.</p>"
        ).format(numbers)
    items = [_source_item(source, result, prefix) for source in sources]
    return AnswerView(header, body, items, answered=True)


def question_html(question: str) -> Markup:
    """The user's question block."""
    return Markup(
        '<div class="soru-mesaji"><p class="mesaj-kim">Soru</p><p class="soru-metni">{}</p></div>'
    ).format(question)


def error_html(message: str) -> Markup:
    """An inline error in place of the answer (service down, unknown config, …)."""
    return Markup('<p class="form-hatasi" role="alert">{}</p>').format(message)


def methods_line(option: PipelineOption) -> str:
    """The plain method line under the selector."""
    return methods_text(option.yontemler)


# --- Hakkında -------------------------------------------------------------------------------


def _count(value: int | None) -> str:
    return number_tr(value, 0) if value is not None else "—"


def _rows(lines: Iterable[MethodLine]) -> Markup:
    rows = Markup("").join(
        Markup("<div><dt>{}</dt><dd>{}</dd></div>").format(line.etiket, line.deger)
        for line in lines
    )
    return Markup('<dl class="satirlar">{}</dl>').format(rows)


def _corpus(info: AboutInfo) -> Markup:
    items = []
    for group in info.gruplar:
        origin: Markup | str = group.kaynak
        if group.kaynak_url:
            origin = Markup('<a href="{}" rel="noopener" target="_blank">{}</a>').format(
                group.kaynak_url, group.kaynak
            )
        details = Markup("")
        if group.ayrinti:
            details = Markup('<ul class="derlem-ayrinti">{}</ul>').format(
                Markup("").join(Markup("<li>{}</li>").format(line) for line in group.ayrinti)
            )
        items.append(
            Markup(
                '<li><h3>{name}</h3><p class="derlem-ozet">{summary}</p>'
                '<p class="derlem-kaynak">Kaynak: {origin}</p>{details}</li>'
            ).format(name=group.ad, summary=group.ozet, origin=origin, details=details)
        )
    return Markup('<ul class="derlem">{}</ul>').format(Markup("").join(items))


def _configs(options: Iterable[PipelineOption]) -> Markup:
    rows = []
    for option in options:
        notes = Markup("")
        if option.varsayilan:
            notes += Markup(' <span class="soluk">(varsayılan)</span>')
        if not option.hazir:
            notes += Markup(' <span class="soluk">({})</span>').format(
                option.durum_notu or NO_INDEX_NOTE
            )
        methods = Markup("").join(
            Markup('<li><span class="etiket-adi">{}:</span> {}</li>').format(m.etiket, m.deger)
            for m in option.yontemler
        )
        rows.append(
            Markup("<div><dt>{}{}</dt><dd><ul>{}</ul></dd></div>").format(option.ad, notes, methods)
        )
    return Markup('<dl class="yapilandirmalar">{}</dl>').format(Markup("").join(rows))


def about_sections(info: AboutInfo) -> list[tuple[str, Markup]]:
    """The Hakkında page as (heading, HTML) sections, in the web page's order."""
    counts = Markup(
        '<dl class="sayilar"><div><dt>Belge</dt><dd>{}</dd></div>'
        "<div><dt>Bölüm</dt><dd>{}</dd></div>"
        "<div><dt>Parça (chunk)</dt><dd>{}</dd></div></dl>"
    ).format(_count(info.belge_sayisi), _count(info.bolum_sayisi), _count(info.parca_sayisi))
    corpus_note = Markup(
        '<p class="not">Bölüm, ayrıştırıcının çıkardığı madde veya başlıktır; parça, aramada '
        "kullanılan metin birimidir (varsayılan koleksiyon <code>{}</code>). Cevaplardaki "
        "kaynak bağlantıları belgenin aslını açar.</p>"
    ).format(info.koleksiyon)
    configs_note = Markup(
        '<p class="not">Sohbet sayfasındaki seçici bu yapılandırmalardan birini kullanır. Her '
        "biri bir öncekinden tek bir yöntemle ayrılır; karşılaştırmalı sonuçlar GitHub'daki "
        "öğrenme belgesindedir.</p>"
    )
    guard_note = Markup(
        '<p class="not">Dil modelinin önünde ve arkasında çalışan denetimler. Biri devreye '
        "girerse sistem kaynaksız cevap vermek yerine bilgi bulunamadığını ya da sorunun "
        "kapsam dışı olduğunu söyler.</p>"
    )
    code = Markup(
        "<p>Kod, geliştirme günlüğü ve deney sonuçları GitHub'dadır: "
        '<a href="{repo}" rel="noopener" target="_blank">{repo}</a>. Her aşamanın gerekçesi '
        '<a href="{docs}" rel="noopener" target="_blank">docs</a> klasöründe anlatılır.</p>'
    ).format(repo=REPO_URL, docs=DOCS_URL)
    return [
        ("Kullanılan belgeler", counts + _corpus(info) + corpus_note),
        ("Modeller ve yöntemler", _rows(info.bilesenler)),
        ("Yapılandırmalar", configs_note + _configs(info.yapilandirmalar)),
        ("Guardrail'ler", guard_note + _rows(info.guardrailler)),
        ("Kaynak kod ve belgeler", code),
    ]


ABOUT_INTRO = (
    "Bu araç, TÜBİTAK destek programları ve Ar-Ge mevzuatı üzerine kurulmuş, kaynak gösteren "
    "bir soru-cevap sistemidir (RAG). Açık kaynak bir eğitim projesidir; resmî bir TÜBİTAK "
    "hizmeti değildir. Aşağıdaki bilgiler sistemin yapılandırma dosyalarından ve belge "
    "listelerinden okunur."
)


def footer_html(is_demo: bool) -> Markup:
    """The page footer (and the demo notice of the fake backend)."""
    demo = Markup('<p class="demo-notu">{}</p>').format(DEMO_NOTE) if is_demo else ""
    return Markup('<footer class="alt"><p>{}</p>{}</footer>').format(FOOTER, demo)
