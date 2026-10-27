# Deney özeti

Altı yapılandırma aynı 104 soruluk test setiyle ve aynı dil modeliyle (`qwen3:4b`) koşuldu; her
basamak karşılaştırıldığı basamağa göre tek bir şeyi değiştirir (v1 doğrudan v0 ile
karşılaştırılır, v0r ile değil). Arama metrikleri (R@5, MRR) kanıt alıntısıyla ölçülür ve
hakemden bağımsızdır. Doğruluk ve doğru ret, 624 cevabın Claude (Opus) ile tek tek
değerlendirilmesinden gelir.

| Koşu | Yöntem | R@5 | MRR | Doğruluk | Doğru ret | Yanlış ret | Gecikme p50 |
|---|---|---:|---:|---:|---:|---:|---:|
| v0 | Sabit pencere (512/64), dense arama | 0,284 | 0,241 | 0,380 | 0,760 | 0,076 | 1.370 ms |
| v0r | Sabit pencere, dense arama + reranker | 0,487 | 0,401 | 0,557 | 0,680 | 0,051 | 2.076 ms |
| v1 | Yapı bazlı parçalama + bağlam başlığı, dense arama | 0,439 | 0,419 | 0,418 | 0,840 | 0,114 | 1.105 ms |
| v2 | Yapı bazlı parçalama + bağlam başlığı, hibrit arama (dense + BM25, RRF) | 0,543 | 0,477 | 0,519 | 0,760 | 0,076 | 1.126 ms |
| v3 | Yapı bazlı parçalama + bağlam başlığı, hibrit arama + reranker | 0,699 | 0,612 | **0,633** | 0,800 | 0,063 | 1.771 ms |
| v3b | v3 ile aynı, embedding bge-m3 | **0,727** | **0,623** | 0,620 | **0,880** | 0,076 | 2.923 ms |

Doğruluk: doğru cevaplanan / cevabı olan 79 soru. Doğru ret: cevabı corpus'ta olmayan 15 ve
kapsam dışı 10 soruda cevap vermeme oranı (cevap metninde "kaynakta yok" diyen cevaplar da doğru
ret sayıldı). Yanlış ret: cevabı olan 79 soruda gereksiz ret oranı. Gecikmeler RTX 3080 GPU'da
ölçüldü.

- **Arama:** Recall@5 en basit sistemden (v0) tam sisteme (v3) 0,284'ten 0,699'a çıktı.
  Reranker sıralamayı düzeltti (kanıtı 6–30. sıradan ilk 5'e taşıdı), yapı bazlı parçalama ve
  bağlam başlığı kanıtın aday havuzuna hiç girmediği soru sayısını 29'dan 12'ye indirdi.
- **Cevap:** En iyi doğruluk yaklaşık %63 (v3). Hataların çoğu arama değil okuma hatası: belirsiz
  sorularda programlara ayırmamak (10 belirsiz sorunun hiçbiri hiçbir koşuda doğru değil), eski
  sürüm değerini seçmek, koşulu ters okumak, programları karıştırmak.
- **Bedel:** reranker ve daha büyük embedding modeli gecikme ekliyor; v3b'nin toplam p95'i
  5.251 ms. v3 ile v3b arasındaki kalite farkları 104 soruluk sette ölçüm belirsizliğinin içinde.

Koşu raporları: [v0](v0.md) · [v0r](v0r.md) · [v1](v1.md) · [v2](v2.md) · [v3](v3.md) ·
[v3b](v3b.md) · [Claude değerlendirmesi](claude-degerlendirme.md) · Ayrıntılı anlatım:
[RAG rehberi](../RAG-GELISTIRME-REHBERI.md).
