# Cevapların Claude (Opus) ile değerlendirilmesi

Altı koşunun 624 cevabı Claude (Opus) ile tek tek okundu. Her soru için soru metni, beklenen
cevap, sistemin cevabı ve cevabın atıf yaptığı kaynak metinleri verildi. Cevabı olan 79 soruda
"cevap doğru mu, kaynağa sadık mı?", cevabı olmayan 25 soruda (15 cevapsız, 10 kapsam dışı)
"ret doğru mu, verilen cevap uydurma mı?", gereksiz retlerde "hata aramada mı, üretimde mi?" diye
bakıldı. Arama metrikleri bu değerlendirmeden bağımsızdır.

| Koşu | Yöntem | Doğruluk | Sadakat | Doğru ret | Uydurma | Gereksiz ret |
|---|---|---:|---:|---:|---:|---:|
| v0 | Sabit pencere, dense arama | 0,380 (30/79) | 0,532 | 0,760 | 6 | 6 |
| v0r | Sabit pencere, dense arama + reranker | 0,557 (44/79) | 0,554 | 0,680 | 8 | 4 |
| v1 | Yapı bazlı parçalama + bağlam başlığı, dense arama | 0,418 (33/79) | 0,574 | 0,840 | 4 | 9 |
| v2 | Yapı bazlı + bağlam başlığı, hibrit arama (BM25 + dense, RRF) | 0,519 (41/79) | 0,696 | 0,760 | 6 | 6 |
| v3 | Yapı bazlı + bağlam başlığı, hibrit arama + reranker | **0,633** (50/79) | **0,806** | 0,800 | 5 | 5 |
| v3b | v3 ile aynı, embedding bge-m3 | 0,620 (49/79) | 0,761 | **0,880** | **3** | 6 |

- **Doğruluk:** doğru cevaplanan / cevabı olan 79 soru (gereksiz ret yanlış sayılır).
- **Sadakat:** cevap verilen sorularda her iddiası atıf yapılan kaynakta geçen cevapların oranı.
  Değerlendirmede her kaynağın yaklaşık ilk 1.000 karakteri gösterildi; dayanak kesilen kısımda
  kaldıysa cevap sadık değil sayılmış olabilir.
- **Doğru ret:** cevabı olmayan 25 soruda doğru davranış oranı. Model cevap metninde "kaynakta
  yok" dediyse de doğru sayıldı.
- **Uydurma / gereksiz ret:** cevabı olmayan soruya dayanaksız cevap / cevabı olan soruya ret.

Soru türüne göre doğru cevap sayısı:

| Tür | v0 | v0r | v1 | v2 | v3 | v3b |
|---|---:|---:|---:|---:|---:|---:|
| Tek belge (43) | 23 | 32 | 26 | 29 | 34 | 34 |
| Çok belge (16) | 4 | 8 | 3 | 6 | 9 | 8 |
| Sürüm (10) | 3 | 4 | 4 | 6 | 7 | 7 |
| Belirsiz (10) | 0 | 0 | 0 | 0 | 0 | 0 |
| Cevapsız (15), doğru ret | 9 | 7 | 11 | 9 | 10 | 12 |
| Kapsam dışı (10), doğru ret | 10 | 10 | 10 | 10 | 10 | 10 |

Gereksiz retlerde hatanın yeri (arama / üretim ve denetim / belirsiz): v0 2/3/1, v0r 1/3/0,
v1 3/3/3, v2 1/2/3, v3 1/2/2, v3b 1/3/2.

Ne gösteriyor?

- En iyi koşuda bile doğruluk yaklaşık %63. Hataların çoğu: belirsiz sorularda tek programın
  değerini genel kural gibi vermek, sürüm sorularında eski değeri seçmek, koşulu ters okumak,
  başka bir programın kuralını uygulamak.
- Cevapsız sorulardaki uydurmaların çoğu, corpus'ta başka bir anlamda geçen bir tutarın soruya mal
  edilmesi (ör. proje bursiyeri üst limitinin 2211 doktora bursu diye verilmesi). Sayı denetimi
  bunları yakalamaz, çünkü sayı kaynakta geçiyor.
- Gereksiz retlerin az bir kısmı aramadan geliyor: altı koşudaki 36 gereksiz retin 9'unda kanıt
  ilk 5 kaynakta yoktu, 16'sında kanıt ilk 5'te olduğu hâlde model ya da denetim cevap vermedi,
  11'inde karar verilemedi.

Soru bazındaki örnekler: [RAG rehberi, Bölüm 6](../RAG-GELISTIRME-REHBERI.md#6-tüm-bunlara-rağmen-yaşanan-sorunlar).
