# Deney raporu: v3b – yapı bazlı parçalama + bağlam başlığı, hibrit arama (dense + BM25, RRF), reranker, embedding bge-m3

Tam sistemle birebir aynı; yalnız embedding modeli EmbeddingGemma 2 (768 boyut) yerine BAAI/bge-m3 (1024 boyut). Parçalar, BM25, reranker, eşik ve dil modeli değişmez. Reranker'lı bir sistemde embedding modelinin etkisi en çok aday havuzunda görünür; bu yüzden Recall@10 ve kanıtın 30 adayda hiç bulunmadığı soru sayısı özellikle izlenir.

Künye: koşu `v3b_20261016-014819` · 104 soru · 282 belge kaydı, 7.017 parça · dil modeli qwen3:4b · cevaplar Claude (Opus) ile tek tek değerlendirildi · süre 10 dk 4 sn.

## Ana metrikler

| Metrik | Önceki (v3) | Bu koşu | Fark |
|---|---|---|---|
| Recall@5 | 0,699 | 0,727 | +0,028 |
| Recall@10 | 0,751 | 0,784 | +0,032 |
| MRR | 0,612 | 0,623 | +0,010 |
| Cevap doğruluğu (Claude) | 0,633 | 0,620 | −0,013 |
| Kaynağa sadakat (Claude) | 0,806 | 0,761 | −0,045 |
| Doğru ret oranı (Claude) | 0,800 | 0,880 | +0,080 |
| Yanlış ret oranı | 0,063 | 0,076 | +0,013 |
| Toplam p50 | 1 771 ms | 2 923 ms | +1 151 ms |

Recall@k: kanıt alıntılarının ilk k parçada bulunma oranı. MRR: ilk doğru parçanın sırasının tersinin ortalaması (1. sıra = 1, 2. sıra = 0,5). Doğru ret: cevapsız ve kapsam dışı sorularda cevap vermeme oranı. Yanlış ret: cevabı olan sorularda gereksiz ret oranı. Doğruluk, sadakat ve doğru ret Claude (Opus) ile tek tek yapılan değerlendirmeden gelir; cevap metninde 'kaynakta yok' diyen cevap da doğru ret sayılır.

- Retler (soru sayısı): kapsam dışı 12 · model 'kaynaklarda yok' dedi 8 · kaynakta olmayan sayı 3 · atıfsız cevap 2 · skor eşiğin altında 2.
- Claude değerlendirmesi: uydurma 3 · gereksiz ret 6 (hatanın yeri: arama 1, üretim 3, belirsiz 2).
- Kapsam kontrolü 2 cevaplanabilir soruyu kapsam dışı saydı.
- Eşik kalibrasyonu: `rerank` skoru için yanlış ret ≤ 0,100 koşuluyla önerilen eşik **2** (doğru ret 0,533, yanlış ret 0,078).

## Örnekler

### İyi örnek: q004 (tek belge)

**Soru:** 1501 projesi için aldığımız kalıp seri üretimde de kullanılacak. Bu kalıbın maliyeti desteğe nasıl dahil edilir?

- Kanıt: 1. sırada · durum: cevaplandı
- Cevap: 1501 programında kalıp maliyeti desteklemeye esas harcama tutarına dahil edilirken, proje süresiyle orantılı olarak aylık %2 oranında destek verilir. Bu durumda desteklemeye esas harcama tutarı 'maliyet x proje süresi(ay) x %2' şeklinde hesaplanır. Ayrıca, 'proje süresi(ay) x %2' çarpımından elde edilen oranın alt sınırı %40'tır [1].

### Kötü örnek: q083 (cevapsız)

**Soru:** 2211 Yurt İçi Doktora Bursu kapsamında ayda kaç TL burs ödeniyor?

- Kanıt: beklenmiyor (corpus'ta cevabı yok) · durum: cevaplandı
- Cevap: 2211 Yurt İçi Doktora Bursu kapsamında ayda 32.500 TL burs ödenir [3].
- Değerlendirme notu (Claude): 32.500 TL proje bursiyer limiti; 2211 bursuna yanlış atfedildi

### Kötü örnek: q027 (tek belge)

**Soru:** 2213 Yurt Dışı Lisansüstü Burs Programında yurt dışında toplam ne kadar süre burs alınabilir?

- Kanıt: 1. sırada · durum: atıf denetiminden geçmedi (atıfsız cevap)
- Değerlendirme notu (Claude): Madde 18 kesik; 24 ay süre kesilen kısımda olabilir
