# Mimari

Bu doküman `tubitak_rag` paketinin nasıl katmanlara ayrıldığını, bileşenlerin birbirine nasıl
bağlandığını ve yeni bir bileşenin nasıl ekleneceğini anlatır. Tasarımın gerekçeleri
[`ihtiyaclar.md`](ihtiyaclar.md) §7'de; imza düzeyindeki sözleşmeler `ports/` ve `domain/` altındaki kodda.

## 1. Katmanlar

```
interfaces/   CLI (Typer) · FastAPI (REST + HTML sayfalar) · Gradio (Hugging Face Space)
      │  yalnızca çekirdeği çağırır
core          domain · ports · ingestion · chunking · indexing · retrieval ·
              reranking · generation · guardrails · evaluation · reporting ·
              pipeline · factory · config
      │  ports (Protocol) üzerinden
adapters      Qdrant · Ollama · HuggingFace (sentence-transformers) · Anthropic · bm25s
```

Bu yapıya **hexagonal mimari** (ports & adapters) denir:

- **domain/** — Pydantic ile yazılmış, değişmez (frozen) veri modelleri: `Document`, `Chunk`,
  `ScoredChunk`, `Answer`, `TestQuestion` … Hiçbir dış kütüphaneye bağımlı değildir.
- **ports/** — `typing.Protocol` arayüzleri. Çekirdek bir embedding modelini, vektör
  veritabanını ya da LLM'i **yalnızca bu arayüzler üzerinden** tanır (NFR-MOD1).
- **çekirdek modüller** — iş mantığı (ayrıştırma, parçalama, arama, guardrail, değerlendirme).
  Somut sınıfları kendileri oluşturmaz; dışarıdan alırlar (dependency injection).
- **adapters/** — dış dünyaya bağlanan sınıflar. Ağır kütüphaneler (`torch`,
  `sentence_transformers`, `qdrant_client`, `anthropic`, `bm25s` …) **yalnızca burada** ve
  fonksiyon / `__init__` gövdesinde import edilir; böylece birim testleri GPU, model indirme ya
  da çalışan servis olmadan koşar (NFR-TEST2).
- **interfaces/** — kullanıcıya dönük katman. Çekirdek bu katmanı import etmez; FastAPI veya
  Gradio çekirdeğe sızamaz (NFR-MOD3). Bu kurallar `tests/unit/test_architecture.py`
  içindeki AST tabanlı testlerle her koşuda denetlenir.

## 2. Paket ağacı

```
src/tubitak_rag/
├── __init__.py            __version__
├── config.py              AppConfig ve alt modeller, load_config()
├── factory.py             kayıt sözlükleri + build_* fonksiyonları
├── pipeline.py            RagPipeline                                   (aşama 6)
├── domain/                base · enums · source · document · chunk · retrieval · answer · evaluation · reports
├── ports/                 embedder · vector_store · chunker · lexical_index · retriever · reranker · llm · parser · tokens
├── ingestion/             manifest · fetch · normalize · pdf_text · *_parser · parser_registry · parsed_cache · ingest   (aşama 2, 4)
├── chunking/              ids · context_header · fixed · structural · tokens · cache   (aşama 3)
├── indexing/              naming · embedding_cache · cached_embedder · sync · bm25_index · export   (aşama 4, 11)
├── retrieval/             turkish · rrf · filters · dense · bm25 · hybrid   (aşama 5)
├── reranking/             cross_encoder   (aşama 5)
├── generation/            prompts · schema · generator   (aşama 6)
├── guardrails/            scope · gate · citations · postprocess   (aşama 6)
├── evaluation/            testset · matching · metrics · judge · runner · calibration · generator   (aşama 7, 8)
├── reporting/             concepts · report · summary · examples   (aşama 8)
├── adapters/              st_embedder · ollama_embedder · qdrant_store · bm25s_index · st_reranker · ollama_llm · anthropic_llm · hf_tokenizer · numpy_store
└── interfaces/            cli · warmup · bootstrap · index_bundle · space_bundle · web/{app, api, pages, schemas, errors, templates/, static/} · gradio_app/{app, chat, content, view, settings, gpu}
```

`tests/unit/` bu ağacı aynalar: her `src/tubitak_rag/<paket>/<modül>.py` için
`tests/unit/<paket>/test_<modül>.py` bulunur (NFR-TEST1; `adapters/` hariç, onların gerçek
servisli testleri `tests/integration/` altındadır). `tests/fakes/` dış servislerin test
taklitlerini (`FakeEmbedder`, `InMemoryVectorStore`, `FakeLLM`, `FakeReranker`,
`FakeLexicalIndex`, `RegexTokenCounter`) içerir.

## 3. Port tablosu

| Port | Görev | Ana metotlar | Uygulamalar (planlanan) |
|---|---|---|---|
| `Embedder` | Metni vektöre çevirir | `embed_documents`, `embed_query` | sentence-transformers (EmbeddingGemma 2, bge-m3), Ollama |
| `VectorStore` | Vektörleri saklar, en yakın komşu arar | `ensure_collection`, `upsert`, `delete_doc`, `search`, `indexed_docs`, `get_chunks`, `count` | Qdrant, NumPy (salt okunur) |
| `Chunker` | `Document` → `list[Chunk]` | `chunk` | sabit pencere, yapı bazlı |
| `LexicalIndex` | Anahtar kelime (BM25) araması | `build`, `save`, `load`, `search` | bm25s + Türkçe kök bulma |
| `Retriever` | Soruya aday parçalar bulur | `retrieve` | dense, BM25, hibrit (RRF) |
| `Reranker` | Adayları yeniden sıralar | `rerank` | bge-reranker-v2-m3 (CrossEncoder) |
| `LLM` | Metin / JSON şemalı cevap üretir | `generate` | Ollama (Qwen3), Anthropic (opsiyonel) |
| `DocumentParser` | Ham dosyayı `Document`'a çevirir | `parse` | mevzuat PDF, kılavuz, HTML, GİB JSON, XLSX |
| `TokenCounter` | "Token" tanımı (chunk boyutu için) | `encode`, `decode`, `count` | HF tokenizer, regex |

Hepsi `@runtime_checkable`'dır: `isinstance(nesne, Embedder)` ile bir sınıfın sözleşmeye
uyduğu testlerde denetlenebilir. Protocol'ler **yapısal** tiplemedir; bir sınıfın port'tan
miras alması gerekmez, aynı metotlara sahip olması yeter.

## 4. Veri akışı

**Ingest (indeksleme):**

```
data/manifests/*.json → SourceEntry → DocumentParser → Document (önbellek: artifacts/cache/parsed)
   → Chunker → list[Chunk] (önbellek: artifacts/cache/chunks/<chunker>)
   → Embedder (önbellek: artifacts/cache/embeddings/cache.sqlite) → VectorStore (Qdrant)
   → LexicalIndex (artifacts/cache/bm25/<chunker_key>)
```

Artımlıdır: belgenin `sha256`'sı değişmediyse yeniden işlenmez; manifestten silinen belge her
iki indeksten de silinir.

**Sorgu:**

```
soru → [kapsam kontrolü] → Retriever (dense | bm25 | hibrit) → Reranker (ops.)
     → [retrieval eşiği] → prompt (numaralı kaynaklar) → LLM (yapılandırılmış çıktı)
     → [atıf doğrulama] → Answer {cevaplanabilir, cevap, atiflar, kaynaklar, gecikme_ms}
```

## 5. Config ve factory ilişkisi

1. `configs/default.yaml` tüm alanları açıkça yazar; deney dosyaları
   (`configs/experiments/v0 … v3b.yaml`) `extends: ../default.yaml` ile yalnızca farkları yazar.
2. `load_config(path)` YAML'ı okur, `extends` zincirini derin birleştirir, `${DEGISKEN:varsayilan}`
   ifadelerini ortam değişkeninden (ve varsa `.env`'den) doldurur, sonucu `AppConfig` ile
   doğrular. Bilinmeyen bir alan (`llm.modelx`) Türkçe ve alan yolunu içeren bir `ConfigError`
   üretir; CLI 2 koduyla çıkar (NFR-CONF1).
3. `factory.py` config'teki anahtara göre somut sınıfı seçer:

   ```python
   EMBEDDERS: dict[str, Callable[[AppConfig], Embedder]]
   build_embedder(cfg)  # EMBEDDERS[cfg.embedder.provider](cfg)
   ```

   Bileşenler bağımlılıklarını kendileri kurmaz; hepsi factory'den gelir (NFR-CLEAN3). Kayıtsız
   bir anahtar, kayıtlı seçenekleri listeleyen `UnknownComponentError` verir.

Gizli anahtarlar (ör. `ANTHROPIC_API_KEY`) hiçbir zaman YAML'a yazılmaz; yalnızca ortam
değişkeni / `.env` (NFR-CONF2). Varsayılan config'te hiçbir harici API seçili değildir (NFR-PRIV1).

## 6. Yeni bir bileşen ekleme: örnek yeni reranker

Bir bileşen **tek dosya + tek kayıt satırı** ile eklenir (NFR-MOD2). Örnek: `MyReranker`.

1. **Dosya:** `src/tubitak_rag/adapters/my_reranker.py` içinde `Reranker` port'unun metodunu
   uygulayın. Ağır kütüphaneyi fonksiyon gövdesinde import edin:

   ```python
   from collections.abc import Sequence

   from tubitak_rag.domain.retrieval import ScoredChunk


   class MyReranker:
       def __init__(self, model_name: str) -> None:
           from my_heavy_lib import Model  # lazy import (NFR-TEST2)

           self.model_name = model_name
           self._model = Model(model_name)

       def rerank(
           self, query: str, candidates: Sequence[ScoredChunk], top_n: int
       ) -> list[ScoredChunk]:
           scores = self._model.score(query, [c.chunk.text for c in candidates])
           rescored = [
               c.model_copy(update={"score": s, "scores": {**c.scores, "rerank": s}})
               for c, s in zip(candidates, scores, strict=True)
           ]
           return sorted(rescored, key=lambda c: c.score, reverse=True)[:top_n]
   ```

2. **Kayıt satırı:** `factory.py` içine

   ```python
   RERANKERS["my"] = lambda cfg: MyReranker(cfg.reranker.model)
   ```

3. **Config şeması:** `config.py` içinde `RerankerConfig.provider` Literal'ine `"my"` ekleyin
   (`Literal["cross_encoder", "fake", "my"]`), böylece yazım hataları yükleme anında yakalanır.
4. **Kullanım:** config dosyasında

   ```yaml
   reranker:
     enabled: true
     provider: my
     model: my-org/my-reranker
   ```

Pipeline kodunda hiçbir değişiklik gerekmez. Test için `tests/unit/` altında sahte modelle bir
birim testi, gerçek modelle `tests/integration/` altında `@pytest.mark.integration` testi yazın.

## 7. Artifact düzeni

```
artifacts/                                (gitignore)
├── fetch_report.json                      rag fetch sonucu (indirilen, atlanan, uyuşmayan)
├── ingest_report.json                     son ingest: eklenen/güncellenen/silinen, chunk sayıları
├── cache/parsed/<doc_id>.<sha256[:12]>.json             Document JSON
├── cache/chunks/<chunker>/<doc_id>.<sha256[:12]>.json   list[Chunk] JSON
├── cache/embeddings/cache.sqlite          (model, metin sha256) -> float32 vektör
├── cache/bm25/<chunker_key>/              bm25s dizini + chunk_ids.json + meta.json
└── runs/<run_id>/{config.yaml, metrics.json, per_question.jsonl, report.md}
    runs/ozet.md                           tüm koşuların karşılaştırması
```

`run_id = "<deney adı>_<YYYYmmdd-HHMMSS>"`. Qdrant koleksiyon adı chunker anahtarı ile embedding
modelinden türetilir: `"<chunker_key>__<model slug>"` (ör. `structural_ctx__google-embeddinggemma-2`).
