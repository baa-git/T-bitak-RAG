"""Gradio page content: the same texts as the web templates, escaped HTML everywhere."""

from tests.eval_builders import make_answer
from tubitak_rag.domain.answer import SourceView
from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.interfaces.gradio_app import content as view
from tubitak_rag.interfaces.web.fake_backend import EXAMPLE_ANSWERED, FakeBackend
from tubitak_rag.interfaces.web.schemas import AskResult, MethodLine, PipelineOption, SourceContext
from tubitak_rag.pipeline import LLM_HATASI_MESAJI


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
        yontemler=[
            MethodLine(etiket="Arama", deger="dense"),
            MethodLine(etiket="Reranker", deger=reranker),
        ],
        hazir=ready,
        durum_notu=None if ready else "indeks yok",
    )


def test_option_label_default_and_speed_note() -> None:
    ready, missing = _option("v2"), _option("v1", ready=False)
    assert view.option_label(ready) == "v2" and view.option_label(missing) == "v1 (indeks yok)"
    assert view.default_option([ready, missing]) == "v2"
    marked = ready.model_copy(update={"varsayilan": True})
    assert view.default_option([missing, marked]) == "v2"
    assert view.default_option([]) == ""
    reranker = _option("v3", reranker="BAAI/bge-reranker-v2-m3")
    assert view.speed_note(reranker, cpu_only=True) == view.RERANKER_SPEED_NOTE
    assert view.speed_note(reranker, cpu_only=False) is None
    assert view.speed_note(ready, cpu_only=True) is None
    assert view.methods_line(ready) == "Arama: dense · Reranker: yok"


def test_answer_lists_only_cited_sources_with_links_and_context() -> None:
    sources = [_source(1), _source(2, doc_title="Belge <2>"), _source(3)]
    answer = make_answer(
        cevap="Oran %75'tir [2]. Ayrıca [1, 3].", kaynaklar=sources, atif_onarimi=[3]
    )
    answer = answer.model_copy(update={"atiflar": [], "gecikme_ms": {"total": 2140.0}})
    context = {"c2": SourceContext(madde_baslik="Madde 7 – Destek oranı", sayfa=6)}
    result = AskResult(answer=answer, baglam=context, yapilandirma="v3")
    reply = view.answer_view(result, "t1-", methods="Arama: hibrit")
    assert reply.answered
    assert "Cevap" in reply.header_html and "2,1 sn" in reply.header_html
    assert 'title="Arama: hibrit"' in reply.header_html and ">v3<" in reply.header_html
    assert 'href="#kaynak-t1-2"' in reply.body_html and "onarim-notu" in reply.body_html
    assert [s.no for s in reply.sources] == [1, 2, 3]  # all of them: none in atiflar
    second = reply.sources[1]
    assert second.anchor == "kaynak-t1-2"
    assert reply.sources[2].repaired and not second.repaired
    assert (second.title, second.url) == ("Belge <2>", "https://tubitak.gov.tr/2.pdf")
    assert second.location == "Madde 7 – Destek oranı · sayfa 6"
    assert str(second.quote_html) == "<p>Metin 2</p>"


def test_model_text_is_escaped() -> None:
    answer = make_answer(cevap="<script>x</script> [1]", kaynaklar=[_source(1, text="<b>q</b>")])
    reply = view.answer_view(AskResult(answer=answer), "p-")
    assert "<script>" not in reply.body_html and "&lt;script&gt;" in reply.body_html
    assert "<b>q</b>" not in reply.sources[0].quote_html
    assert "<i>" not in view.question_html("<i>soru</i>")
    assert "&lt;" in view.error_html("<hata>")


def test_refusals_follow_the_web_texts() -> None:
    out = make_answer(durum=AnswerStatus.KAPSAM_DISI, cevap="Kapsam dışı.")
    not_found = make_answer(durum=AnswerStatus.BILGI_BULUNAMADI)
    rejected = make_answer(durum=AnswerStatus.ATIF_REDDI)
    llm_down = make_answer(durum=AnswerStatus.BILGI_BULUNAMADI).model_copy(
        update={"cevap": LLM_HATASI_MESAJI}
    )
    views = {
        name: view.answer_view(AskResult(answer=a), "p-")
        for name, a in [("out", out), ("nf", not_found), ("rej", rejected), ("llm", llm_down)]
    }
    assert not any(v.answered or v.sources for v in views.values())
    assert view.OUT_OF_SCOPE_NOTE in views["out"].body_html
    assert view.NOT_FOUND in views["nf"].body_html
    assert view.BELOW_THRESHOLD_NOTE in views["nf"].body_html
    assert view.CITATION_REJECTED_NOTE in views["rej"].body_html
    assert (
        "Dil modelinden" in views["llm"].body_html and view.NOT_FOUND not in views["llm"].body_html
    )


def test_fake_backend_answer_renders() -> None:
    result = FakeBackend().ask(EXAMPLE_ANSWERED, None, None)
    reply = view.answer_view(result, "t-")
    assert reply.answered and reply.sources and reply.sources[0].title


def test_about_sections_mirror_the_web_page() -> None:
    info = FakeBackend().about()
    sections = view.about_sections(info)
    assert [heading for heading, _ in sections] == [
        "Kullanılan belgeler",
        "Modeller ve yöntemler",
        "Yapılandırmalar",
        "Guardrail'ler",
        "Kaynak kod ve belgeler",
    ]
    corpus = sections[0][1]
    assert "7.017" in corpus and "TÜBİTAK mevzuatı" in corpus and info.koleksiyon in corpus
    assert "(varsayılan)" in sections[2][1] and "v3b" in sections[2][1]
    assert "Kapsam kontrolü" in sections[3][1]
    assert "github.com" in sections[4][1]
    missing = info.model_copy(update={"bolum_sayisi": None})
    assert "—" in view.about_sections(missing)[0][1]


def test_footer() -> None:
    assert view.FOOTER in view.footer_html(False) and "demo-notu" not in view.footer_html(False)
    assert "demo-notu" in view.footer_html(True)
