---
title: TÜBİTAK Mevzuat Rehberi
emoji: 📚
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: TÜBİTAK ve Ar-Ge mevzuatına kaynaklı soru-cevap
startup_duration_timeout: 1h
models:
  - Qwen/Qwen3-4B
  - google/embeddinggemma-2
  - BAAI/bge-m3
  - BAAI/bge-reranker-v2-m3
preload_from_hub:
  - Qwen/Qwen3-4B
  - google/embeddinggemma-2
  - BAAI/bge-m3
  - BAAI/bge-reranker-v2-m3
---

# TÜBİTAK Mevzuat Rehberi

TÜBİTAK destek programları ve Ar-Ge mevzuatı üzerinde çalışan, her cevabın altında dayandığı
kaynakları numaralı olarak gösteren bir soru-cevap aracı (RAG). Açık kaynak bir eğitim
projesidir; **resmî bir TÜBİTAK hizmeti değildir**. Cevaplar bilgilendirme amaçlıdır; kesin
bilgi için ilgili mevzuata ve TÜBİTAK'a başvurunuz.

- **Sohbet:** soru sorun; cevaptaki `[n]` numaraları alttaki kaynaklara karşılık gelir. Her
  kaynakta belgenin aslına bağlantı ve açılır kapanır bir alıntı vardır. *Yapılandırma*
  seçicisi deneylerdeki pipeline'lardan birini (v0 … v3b) seçer; yanındaki satır kullanılan
  yöntemleri yazar.
- **Hakkında:** kullanılan belgeler, modeller, yapılandırmalar ve guardrail'ler.

Her şey bu Space'in GPU'sunda (ZeroGPU) çalışır; sorular hiçbir dış API'ye gönderilmez.
İndeks önceden hesaplanmıştır (bellek içi NumPy vektörleri + BM25); sorunun embedding'i
(EmbeddingGemma 2, v3b'de bge-m3), reranker (bge-reranker-v2-m3) ve cevabı yazan dil modeli
(Qwen3 4B, düşünme modu kapalı, sıcaklık 0) GPU'da çalışır. Varsayılan yapılandırma tam
pipeline'dır (`v3`: yapı bazlı parçalama, hibrit arama, reranker).

ZeroGPU ziyaretçi başına günlük bir GPU süresi tanır (giriş yapmadan daha az, ücretsiz bir
Hugging Face hesabıyla daha çok). Süre dolarsa sayfa bunu söyler; ertesi gün yenilenir.

Kaynak kod, kurulum ve deney sonuçları: <https://github.com/baa-git/T-bitak-RAG>
