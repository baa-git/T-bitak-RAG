"""Gradio view: the web page's turn layout as escaped HTML, plus the Hakkında tab."""

from tests.eval_builders import make_answer
from tubitak_rag.domain.answer import SourceView
from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.interfaces.gradio_app import view
from tubitak_rag.interfaces.web.fake_backend import EXAMPLE_ANSWERED, FakeBackend
from tubitak_rag.interfaces.web.schemas import AskResult, MethodLine, PipelineOption, SourceContext


def _source(no: int, **fields: object) -> SourceView:
    values: dict[str, object] = {
        "no": no,
        "chunk_id": f"c{no}",
        "doc_title": f"Belge {no}",
        "konum": "Madde 7 (2)",
        "kaynak_url": f"https://tubitak.gov.tr/{no}.pdf",
        "text": f"Metin {no}",
        "score": 0.9,
    }
    values.update(fields)
    return SourceView(**values)  # type: ignore[arg-type]


def _option(name: str, reranker: str = "yok", ready: bool = True) -> PipelineOption:
    return PipelineOption(
        ad=name,
        etiket=name,
        aciklama="",
        yontemler=[MethodLine(etiket="Reranker", deger=reranker)],
        hazir=ready,
        durum_notu=None if ready else "indeks yok",
    )


def test_answered_turn_has_linked_citations_and_numbered_sources() -> None:
    sources = [_source(1), _source(2, doc_title="Belge <2>", text="<b>alıntı</b>")]
    answer = make_answer(cevap="Oran %75 [2]. Ayrıca [1].", kaynaklar=sources, atif_onarimi=[1])
    answer = answer.model_copy(update={"atiflar": [], "gecikme_ms": {"total": 2140.0}})
    context = {"c2": SourceContext(madde_baslik="Madde 7 – Destek oranı", sayfa=6)}
    result = AskResult(answer=answer, baglam=context, yapilandirma="v3")

    html = view.turn_html("Oran <nedir>?", result, prefix="t1-", methods="Arama: hibrit")

    assert html.startswith('<section class="tur"') and "Oran &lt;nedir&gt;?" in html
    assert "2,1 sn" in html and 'title="Arama: hibrit"' in html
    assert 'href="#kaynak-t1-2"' in html and 'id="kaynak-t1-2"' in html
    assert '<span class="kaynak-no">[2]</span>' in html
    assert "Belge &lt;2&gt;" in html and "Madde 7 – Destek oranı · sayfa 6" in html
    assert 'href="https://tubitak.gov.tr/2.pdf"' in html and "Belgenin aslı" in html
    assert "<details><summary>Alıntıyı göster</summary>" in html
    assert "&lt;b&gt;alıntı&lt;/b&gt;" in html and "<b>alıntı" not in html
    assert html.count('class="kaynak-onarim"') == 1  # source [1] was repaired
    assert 'class="mesaj cevap-mesaji"' in html


def test_refusal_and_error_turns() -> None:
    refusal = AskResult(answer=make_answer(durum=AnswerStatus.BILGI_BULUNAMADI))
    html = view.turn_html("2244?", refusal, prefix="p-")
    assert "cevap-mesaji ret" in html and "Kaynaklar" not in html
    assert "bilgi bulunamadı" in html
    error = view.turn_html("soru", None, prefix="p-", error="Qdrant <kapalı>")
    assert 'class="form-hatasi"' in error and "Qdrant &lt;kapalı&gt;" in error


def test_thread_pending_and_empty_state() -> None:
    assert view.HOW_IT_WORKS in view.thread_html([])
    turn = view.turn_html("a", None, prefix="p-", error="x")
    thread = view.thread_html([turn], pending="Yeni <soru>")
    assert thread.count('class="tur') == 2 and view.PENDING in thread
    assert "Yeni &lt;soru&gt;" in thread and view.HOW_IT_WORKS not in thread


def test_selector_helpers() -> None:
    options = [_option("v2"), _option("v1", ready=False), _option("v3", "BAAI/bge-reranker-v2-m3")]
    assert view.option_choices(options) == [
        ("v2", "v2"),
        ("v1 (indeks yok)", "v1"),
        ("v3", "v3"),
    ]
    assert view.methods_html(options[2]) == (
        '<p class="yontem">Reranker: BAAI/bge-reranker-v2-m3</p>'
    )
    assert "hiz-notu" in view.methods_html(options[2], cpu_only=True)
    assert "hiz-notu" not in view.methods_html(options[0], cpu_only=True)
    assert view.methods_html(None) == ""
    assert view.disclaimer_html("<uyarı>") == '<p class="alt-not">&lt;uyarı&gt;</p>'


def test_about_tab_and_footer() -> None:
    html = view.about_html(FakeBackend().about())
    assert "<h1>Hakkında</h1>" in html and view.ABOUT_INTRO in html
    for heading in ["Kullanılan belgeler", "Modeller ve yöntemler", "Yapılandırmalar"]:
        assert f"<h2>{heading}</h2>" in html
    assert "7.017" in html and "github.com" in html
    assert "demo-notu" in view.page_footer_html(True)
    assert "demo-notu" not in view.page_footer_html(False)
    assert view.CHAT_TITLE in view.chat_header_html()


def test_fake_backend_answer_renders() -> None:
    result = FakeBackend().ask(EXAMPLE_ANSWERED, None, None)
    html = view.turn_html(EXAMPLE_ANSWERED, result, prefix="t-")
    assert 'class="kaynak-listesi"' in html and 'class="atif"' in html
