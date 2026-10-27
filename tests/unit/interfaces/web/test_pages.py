"""Page routes and HTMX fragments rendered with the demo backend.

The query page is a conversation panel: each question is appended to the thread as a turn
(question + reply with its cited sources). The Hakkında page replaces the old Belgeler and
Deneyler pages; the document preview stays reachable from source links.
"""

import re

import pytest
from fastapi.testclient import TestClient

from tubitak_rag.interfaces.web.about import methods_text
from tubitak_rag.interfaces.web.app import create_app
from tubitak_rag.interfaces.web.backend import ServiceUnavailableError
from tubitak_rag.interfaces.web.fake_backend import (
    EXAMPLE_ANSWERED,
    EXAMPLE_NOT_FOUND,
    EXAMPLE_OUT_OF_SCOPE,
    URL_1507,
    FakeBackend,
)
from tubitak_rag.interfaces.web.schemas import AboutInfo, AskResult, MethodLine, PipelineOption

HX = {"HX-Request": "true"}
REFUSAL = "Bu konuda elimdeki dokümanlarda bilgi bulunamadı."
DISCLAIMER = "resmî görüş değildir"
V3_METHODS = (
    "Parçalama: yapı bazlı + bağlam başlığı · Embedding: google/embeddinggemma-2 (768 boyut)"
    " · Arama: hibrit (BM25 + dense, RRF k=60) · Reranker: BAAI/bge-reranker-v2-m3"
    " · LLM: qwen3:4b"
)


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def client(backend: FakeBackend) -> TestClient:
    return TestClient(create_app(backend=backend))


def _turn(client: TestClient, soru: str, **params: str) -> str:
    response = client.get("/ui/ask", params={"soru": soru, **params}, headers=HX)
    assert response.status_code == 200
    return response.text


def _prefix(html: str) -> str:
    """The per-turn anchor prefix ("t1a2b3c-") of the first source in ``html``."""
    match = re.search(r'id="kaynak-(t[0-9a-f]{6}-)1"', html)
    assert match, "no source anchor in the turn"
    return match.group(1)


# --- Sohbet: page --------------------------------------------------------------------------


def test_sorgu_page_is_a_conversation_panel(client: TestClient) -> None:
    html = client.get("/").text
    assert '<html lang="tr">' in html
    assert 'id="sohbet"' in html and 'role="log"' in html
    assert 'hx-get="/ui/ask"' in html
    assert 'hx-target="#sohbet"' in html and 'hx-swap="beforeend"' in html
    assert 'name="soru"' in html and ">Sor</button>" in html
    assert "Yeni sohbet" in html
    assert EXAMPLE_ANSWERED in html  # example questions in the empty thread


def test_sorgu_page_has_no_filters(client: TestClient) -> None:
    html = client.get("/").text
    for removed in ('name="program_kodu"', 'name="doc_turu"', "Belge türü", "Program kodu"):
        assert removed not in html


def test_config_selector_uses_plain_labels(client: TestClient) -> None:
    html = client.get("/").text
    assert 'name="yapilandirma"' in html
    assert '<option value="v3" selected>v3</option>' in html
    for name in ("v0", "v0r", "v1", "v2", "v3b"):
        assert f'<option value="{name}">{name}</option>' in html
    for slogan in ("Endüstri standardı", "Naive RAG", "endüstri"):
        assert slogan not in html


def test_selected_config_methods_are_shown_under_the_selector(client: TestClient) -> None:
    html = client.get("/").text
    selected = re.search(r'<p class="yontem" data-yapilandirma="v3">(.*?)</p>', html, re.S)
    assert selected and " ".join(selected.group(1).split()) == V3_METHODS
    assert re.search(r'<p class="yontem" data-yapilandirma="v0" hidden>', html)  # JS toggles


class _CustomMethods(FakeBackend):
    def pipelines(self) -> list[PipelineOption]:
        line = [MethodLine(etiket="Arama", deger="özel arama"), MethodLine(etiket="LLM", deger="x")]
        return [PipelineOption(ad="p1", etiket="p1", aciklama="", varsayilan=True, yontemler=line)]


def test_method_line_comes_from_the_backend_not_the_template() -> None:
    html = TestClient(create_app(backend=_CustomMethods())).get("/").text
    assert "Arama: özel arama · LLM: x" in html
    assert "bge-reranker" not in html


def test_disclaimer_is_shown_once_on_the_page(client: TestClient) -> None:
    assert client.get("/").text.count(DISCLAIMER) == 1
    assert client.get("/", params={"soru": EXAMPLE_ANSWERED}).text.count(DISCLAIMER) == 1


def test_navigation_has_no_experiments_or_documents_page(client: TestClient) -> None:
    html = client.get("/").text
    nav = html[html.index("<nav") : html.index("</nav>")]
    assert 'href="/"' in nav and 'href="/hakkinda"' in nav
    assert "Deneyler" not in nav and "/deneyler" not in nav and "/belgeler" not in nav


@pytest.mark.parametrize("path", ["/deneyler", "/deneyler/ozet", "/ui/karsilastir"])
def test_removed_pages_are_gone(client: TestClient, path: str) -> None:
    assert client.get(path, headers={"Accept": "text/html"}).status_code == 404


def test_ingest_action_is_gone(client: TestClient, backend: FakeBackend) -> None:
    assert client.post("/ui/ingest", headers=HX).status_code in {404, 405}
    assert backend.jobs == []


def test_no_external_scripts_or_styles(client: TestClient) -> None:
    html = client.get("/").text
    assert not re.search(r'<script[^>]+src="https?://', html)
    assert not re.search(r'<link[^>]+href="https?://', html)
    for asset in ("/static/htmx.min.js", "/static/style.css", "/static/app.js"):
        assert asset in html
        assert client.get(asset).status_code == 200


def test_demo_notice_follows_backend_flag(backend: FakeBackend) -> None:
    assert "gösterim verisiyle" in TestClient(create_app(backend=backend)).get("/").text
    backend.is_demo = False
    assert "gösterim verisiyle" not in TestClient(create_app(backend=backend)).get("/").text


# --- Sohbet: one turn ----------------------------------------------------------------------


def test_turn_has_the_question_and_the_reply(client: TestClient) -> None:
    html = _turn(client, EXAMPLE_ANSWERED)
    assert "<html" not in html  # a fragment appended to the thread
    question = html.index('class="mesaj soru-mesaji"')
    reply = html.index('class="mesaj cevap-mesaji"')
    assert question < reply
    assert EXAMPLE_ANSWERED in html[question:reply]
    assert "destek oranı her dönem için en fazla %75" in html[reply:]


def test_turn_does_not_rewrite_the_browser_url(client: TestClient) -> None:
    response = client.get("/ui/ask", params={"soru": EXAMPLE_ANSWERED}, headers=HX)
    assert "HX-Push-Url" not in response.headers


def test_sources_block_lists_cited_sources_with_location_and_links(client: TestClient) -> None:
    html = _turn(client, EXAMPLE_ANSWERED)
    prefix = _prefix(html)
    block = html[html.index('class="kaynaklar"') :]
    assert ">Kaynaklar<" in block
    for no in (1, 2, 3):
        assert f'href="#kaynak-{prefix}{no}"' in html  # inline [n] in the answer
        assert f'id="kaynak-{prefix}{no}"' in block  # its entry in the block
    assert "TÜBİTAK 1507- KOBİ Ar-Ge Başlangıç Destek Programı Uygulama Esasları" in block
    assert "Madde 7 – Destek oranı ve sınırı" in block and "sayfa 6" in block
    assert "İkinci Bölüm – Destek Kapsamı" in block
    assert f'href="{URL_1507}"' in block and "Belgenin aslı" in block
    assert 'href="/belgeler/mevzuat-031#bolum-mevzuat-031:0012"' in block
    assert block.count("<details") == 3 and "Alıntıyı göster" in block
    assert "en fazla %75 olarak belirlenmiştir" in block  # the cited passage


def test_two_turns_never_share_anchors(client: TestClient) -> None:
    first, second = _turn(client, EXAMPLE_ANSWERED), _turn(client, EXAMPLE_ANSWERED)
    assert _prefix(first) != _prefix(second)


class _PartlyCited(FakeBackend):
    def ask(
        self,
        soru: str,
        program_kodu: str | None,
        doc_turu: str | None,
        yapilandirma: str | None = None,
    ) -> AskResult:
        result = super().ask(soru, program_kodu, doc_turu, yapilandirma)
        answer = result.answer
        cited = [c for c in answer.atiflar if c.no != 2]
        text = answer.cevap.replace(" [2]", "")
        return result.model_copy(
            update={"answer": answer.model_copy(update={"atiflar": cited, "cevap": text})}
        )


def test_sources_block_lists_only_cited_sources() -> None:
    html = _turn(TestClient(create_app(backend=_PartlyCited())), EXAMPLE_ANSWERED)
    prefix = _prefix(html)
    assert f'id="kaynak-{prefix}1"' in html and f'id="kaynak-{prefix}3"' in html
    assert f'id="kaynak-{prefix}2"' not in html
    assert "Madde 6 – Destekleme dönemi" not in html  # source 2 was given but not cited


def test_reply_names_the_config_and_its_time(client: TestClient, backend: FakeBackend) -> None:
    html = _turn(client, EXAMPLE_ANSWERED, yapilandirma="v1")
    assert backend.asked[-1][1:] == (None, None, "v1")  # no filters are sent
    assert re.search(r'class="yapilandirma-adi"[^>]*>v1</span>', html)
    v1 = next(o for o in backend.pipelines() if o.ad == "v1")
    assert f'title="{methods_text(v1.yontemler)}"' in html
    assert "6,9 sn" in html  # demo latency without the reranker stage: 6890,2 ms


def test_reply_does_not_repeat_the_disclaimer(client: TestClient) -> None:
    assert DISCLAIMER not in _turn(client, EXAMPLE_ANSWERED)


def test_not_found_is_a_reply_without_sources(client: TestClient) -> None:
    html = _turn(client, EXAMPLE_NOT_FOUND)
    assert 'class="mesaj cevap-mesaji ret"' in html
    assert REFUSAL in html
    assert "kaynak-" not in html and "<details" not in html and "Kaynaklar" not in html


def test_out_of_scope_is_a_reply_without_sources(client: TestClient) -> None:
    html = _turn(client, EXAMPLE_OUT_OF_SCOPE)
    assert "kapsamı dışında" in html and REFUSAL not in html
    assert "kaynak-" not in html and "<details" not in html


def test_full_page_with_question_renders_the_turn_without_javascript(client: TestClient) -> None:
    html = client.get("/", params={"soru": EXAMPLE_ANSWERED, "yapilandirma": "v1"}).text
    assert '<html lang="tr">' in html and 'class="mesaj soru-mesaji"' in html
    assert _prefix(html)
    assert '<option value="v1" selected>v1</option>' in html


@pytest.mark.parametrize(
    ("soru", "message"),
    [("   ", "Lütfen bir soru yazın."), ("x" * 1001, "en fazla 1000 karakter")],
)
def test_invalid_question_is_an_inline_error(client: TestClient, soru: str, message: str) -> None:
    html = _turn(client, soru)
    assert message in html and 'role="alert"' in html


def test_unknown_config_is_an_inline_error(client: TestClient, backend: FakeBackend) -> None:
    html = _turn(client, EXAMPLE_ANSWERED, yapilandirma="v9")
    assert "Bilinmeyen yapılandırma" in html and 'role="alert"' in html
    assert backend.asked == []


def test_busy_gpu_is_an_inline_error(client: TestClient, backend: FakeBackend) -> None:
    backend.gpu_busy = True
    html = _turn(client, EXAMPLE_ANSWERED)
    assert "GPU kilidi dolu" in html and 'role="alert"' in html


def test_ask_fragment_without_htmx_redirects_to_page(client: TestClient) -> None:
    response = client.get(
        "/ui/ask", params={"soru": "1507", "yapilandirma": "v2"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/?soru=1507&yapilandirma=v2"


def test_user_text_is_escaped(client: TestClient) -> None:
    html = _turn(client, "<script>alert(1)</script> 1507 destek oranı")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# --- Repaired citations --------------------------------------------------------------------


class _RepairedBackend(FakeBackend):
    def ask(
        self,
        soru: str,
        program_kodu: str | None,
        doc_turu: str | None,
        yapilandirma: str | None = None,
    ) -> AskResult:
        result = super().ask(soru, program_kodu, doc_turu, yapilandirma)
        answer = result.answer.model_copy(update={"atif_onarimi": [3]})
        return result.model_copy(update={"answer": answer})


def test_repaired_citation_is_marked_as_added_by_the_guardrail() -> None:
    html = _turn(TestClient(create_app(backend=_RepairedBackend())), EXAMPLE_ANSWERED)
    assert html.count("guardrail tarafından eklendi") == 2  # under the answer + on source 3
    source_3 = html[html.index(f'id="kaynak-{_prefix(html)}3"') :]
    assert "guardrail tarafından eklendi" in source_3


def test_unrepaired_answer_has_no_repair_note(client: TestClient) -> None:
    assert "guardrail tarafından eklendi" not in _turn(client, EXAMPLE_ANSWERED)


# --- Hakkında ------------------------------------------------------------------------------


def test_about_page_describes_corpus_models_and_methods(client: TestClient) -> None:
    response = client.get("/hakkinda")
    assert response.status_code == 200
    html = response.text
    assert 'aria-current="page"' in html
    for text in (
        "tubitak.gov.tr/tr/mevzuat",
        "60 PDF",
        "105 sayfa + 115 dosya",
        "5746 sayılı",
        "4691 sayılı",
        "google/embeddinggemma-2",
        "BAAI/bge-m3",
        "BAAI/bge-reranker-v2-m3",
        "qwen3:4b",
        "Qdrant",
        "Türkçe kök bulma",
        "RRF",
        "Kapsam kontrolü",
        "Arama eşiği",
        "Atıf doğrulama",
        "Sayı doğrulama",
        "atıf onarımı",
        "https://github.com/baa-git/T-bitak-RAG",
    ):
        assert text in html, text
    assert V3_METHODS.split(" · ")[2].split(": ")[1] in html  # per-config method table


def test_about_page_has_no_ingest_controls(client: TestClient) -> None:
    html = client.get("/hakkinda").text
    assert "indeksi güncelle" not in html.casefold()
    assert "hx-post" not in html and "/ui/ingest" not in html


def test_old_documents_page_redirects_to_about(client: TestClient) -> None:
    response = client.get("/belgeler", follow_redirects=False)
    assert response.status_code == 301 and response.headers["location"] == "/hakkinda"


class _DownBackend(FakeBackend):
    def about(self) -> AboutInfo:
        raise ServiceUnavailableError("Vektör deposuna ulaşılamadı (http://localhost:6333)")


def test_unavailable_service_renders_an_html_error_page() -> None:
    client = TestClient(create_app(backend=_DownBackend()))
    response = client.get("/hakkinda", headers={"Accept": "text/html"})
    assert response.status_code == 503
    assert "Servis kullanılamıyor" in response.text and "localhost:6333" in response.text


# --- Document preview (reachable from source links, not from the menu) ---------------------


def test_document_preview_lists_sections_with_anchors(client: TestClient) -> None:
    response = client.get("/belgeler/mevzuat-031")
    assert response.status_code == 200
    html = response.text
    assert "TÜBİTAK 1507- KOBİ Ar-Ge Başlangıç Destek Programı Uygulama Esasları" in html
    assert 'id="bolum-mevzuat-031:0012"' in html and "Madde 7" in html
    assert "Mülga" in html and "(Değişik: 29/12/2025-77YK)" in html
    assert 'href="https://tubitak.gov.tr/sites/default/files/2026-10/1507_UE' in html
    assert html.count("<details") == 5
    assert 'href="/belgeler"' not in html  # no link back to the removed list


def test_document_preview_without_parsed_text(client: TestClient) -> None:
    html = client.get("/belgeler/mevzuat-064").text
    assert "ayrıştırılmış metni henüz yok" in html and "<details" not in html
    assert "güncelleme" not in html.casefold()


def test_unknown_document_is_404_page(client: TestClient) -> None:
    response = client.get("/belgeler/yok")
    assert response.status_code == 404 and "Sayfa bulunamadı" in response.text


def test_configs_without_index_are_disabled_in_the_selector(
    backend: FakeBackend, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = backend.pipelines()
    patched = [
        o.model_copy(update={"hazir": False, "durum_notu": "indeks yok"}) if o.ad == "v0" else o
        for o in options
    ]
    monkeypatch.setattr(backend, "pipelines", lambda: patched)
    html = client.get("/").text
    assert re.search(r'<option value="v0" disabled>v0 \(indeks yok\)</option>', html)
    assert re.search(r'<option value="v3" selected>v3</option>', html)
    about = client.get("/hakkinda").text
    assert "indeks yok" in about
