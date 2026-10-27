# Deney raporu: v0r – sabit pencere, dense arama, reranker, embedding EmbeddingGemma 2

Taban çizgisiyle aynı parçalar ve aynı dense arama; tek fark, aramanın getirdiği 30 adayın bir cross-encoder reranker (bge-reranker-v2-m3) ile yeniden sıralanmasıdır. Reranker soruyu ve parçayı birlikte okuduğu için daha isabetlidir ama daha yavaştır ve yalnız ilk aşamanın getirdiği havuzu sıralayabilir. Amaç, yeniden sıralamanın kötü parçalamayı ne ölçüde telafi edebildiğini görmektir.

Künye: koşu `v0r_20261016-005933` · 104 soru · 282 belge kaydı, 4.193 parça · dil modeli qwen3:4b · cevaplar Claude (Opus) ile tek tek değerlendirildi · süre 6 dk 20 sn.

## Ana metrikler

| Metrik | Önceki (v0) | Bu koşu | Fark |
|---|---|---|---|
| Recall@5 | 0,284 | 0,487 | +0,203 |
| Recall@10 | 0,355 | 0,513 | +0,158 |
| MRR | 0,241 | 0,401 | +0,160 |
| Cevap doğruluğu (Claude) | 0,380 | 0,557 | +0,177 |
| Kaynağa sadakat (Claude) | 0,532 | 0,554 | +0,021 |
| Doğru ret oranı (Claude) | 0,760 | 0,680 | −0,080 |
| Yanlış ret oranı | 0,076 | 0,051 | −0,025 |
| Toplam p50 | 1 370 ms | 2 076 ms | +706 ms |

Recall@k: kanıt alıntılarının ilk k parçada bulunma oranı. MRR: ilk doğru parçanın sırasının tersinin ortalaması (1. sıra = 1, 2. sıra = 0,5). Doğru ret: cevapsız ve kapsam dışı sorularda cevap vermeme oranı. Yanlış ret: cevabı olan sorularda gereksiz ret oranı. Doğruluk, sadakat ve doğru ret Claude (Opus) ile tek tek yapılan değerlendirmeden gelir; cevap metninde 'kaynakta yok' diyen cevap da doğru ret sayılır.

- Retler (soru sayısı): kapsam dışı 12 · model 'kaynaklarda yok' dedi 5 · kaynakta olmayan sayı 1 · geçersiz atıf 1 · skor eşiğin altında 1 · atıfsız cevap 1.
- Claude değerlendirmesi: uydurma 8 · gereksiz ret 4 (hatanın yeri: arama 1, üretim 3).
- Kapsam kontrolü 2 cevaplanabilir soruyu kapsam dışı saydı.
- Eşik kalibrasyonu: `rerank` skoru için yanlış ret ≤ 0,100 koşuluyla önerilen eşik **1,5** (doğru ret 0,533, yanlış ret 0,078).

## Örnekler

### İyi örnek: q019 (tek belge)

**Soru:** 1002-B Acil Destek Modülünde proje süresi ve bütçe sınırı nedir?

- Kanıt: 1. sırada · durum: cevaplandı
- Cevap: 1002-B Acil Destek Modülünde proje süresi en fazla 6 ay olabilir ve proje destek üst limiti (burs dahil) 100.000 TL’dir [1].

### Kötü örnek: q079 (cevapsız)

**Soru:** 1507 programında 2027 yılı için belirlenen proje bütçesi üst limiti kaç TL?

- Kanıt: beklenmiyor (corpus'ta cevabı yok) · durum: cevaplandı
- Cevap: 1507 programında 2027 yılı için belirlenen proje bütçesi üst limiti 3.500.000 TL'dir [4].
- Değerlendirme notu (Claude): 3.500.000 TL kaynakta yok ve 2027'ye atfedildi

### Kötü örnek: q002 (çok belge)

**Soru:** Büyük ölçekli bir firmayız; 1501 programından hangi destek oranıyla yararlanabiliriz?

- Kanıt: 30 adayda yok · durum: cevaplandı
- Cevap: Büyük ölçekli firmalar için destek oranı %60 [1, 2, 3] İlgili duyuru/çağrı sayfası: https://tubitak.gov.tr/tr/destekler/sanayi/ulusal-destek-programlari/1501-tubitak-sanayi-ar-ge-projeleri-destekleme-programi
- Değerlendirme notu (Claude): Büyük firmanın güncel 2026-2 çağrısına başvuramayacağı belirtilmemiş.
