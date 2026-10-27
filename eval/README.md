# Değerlendirme test seti (`testset_v1`)

Bu klasör, RAG sisteminin her deneyinde kullanılan **altın soru setini** içerir. Retrieval
isabeti, cevap doğruluğu, doğru ret (cevapsız / kapsam dışı) ve sürüm farkındalığı bu
sorular üzerinden ölçülür ([`docs/ihtiyaclar.md`](../docs/ihtiyaclar.md) §9; ihtiyaçlar: "Test seti" kategorisi).

| Dosya | İçerik |
|---|---|
| `testset_v1.jsonl` | 104 soru, satır başına bir JSON nesnesi (UTF-8, LF) |
| `review_queue.md` | İnsan incelemesi için tür bazında, en fazla 10 soruluk partiler halinde kontrol listesi |
| `README.md` | Bu belge: şema, yöntem, dağılım, inceleme süreci, sınırlar |

> **Durum:** Tüm sorular `dogrulandi: false`. Set yazıldı ve alıntıları makineyle
> doğrulandı, ancak henüz insan tarafından gözden geçirilmedi. Doğrulanmamış sorular
> metriklerde ayrı raporlanır (`by_verified` kırılımı).

## Şema

Her satır `tubitak_rag.domain.evaluation.TestQuestion` modelidir:

| Alan | Tip | Açıklama |
|---|---|---|
| `id` | str | `q001` … `q111`, benzersiz ve kalıcı (silinen kimlik tekrar kullanılmaz; numaralarda boşluk olabilir) |
| `soru` | str | Başvuru sahibinin soracağı doğal Türkçe soru |
| `tur` | enum | `tek_belge`, `cok_belge`, `cevapsiz`, `kapsam_disi`, `surum`, `belirsiz` |
| `zorluk` | enum | `kolay`, `orta`, `zor` |
| `beklenen_cevap` | str | Kısa referans cevap; cevapsızda `"bilgi bulunamadı"`, kapsam dışında `"kapsam dışı"` |
| `kanit_alintilari` | list | 1–3 `{doc_id, alinti}`; ayrıştırılmış metinden **birebir** kısa alıntı |
| `beklenen_belgeler` | list[str] | Cevabın dayandığı belge kimlikleri (`mevzuat-030`, `dosya-…`, `sayfa-…`, `kanun-5746`) |
| `beklenen_sayilar` | list[str] | Kural tabanlı kontrol için cevapta geçmesi gereken sayılar (`"75"`, `"1.350.000"`) |
| `program_kodu` | list[str] | İlgili program kodları (`"1501"`, `"2211"` …) |
| `dogrulandi` | bool | İnsan incelemesinden geçti mi |
| `review_notes` | str | Yazarın veya inceleyicinin notu (eksik veri, belirsizlik, eski değer) |
| `uretim` | enum | `manuel` (bu sette hepsi) veya `llm` |

Örnek satır:

```json
{"id":"q001","soru":"KOBİ olarak 1507 programından proje desteği alırsam TÜBİTAK harcamalarımın en fazla yüzde kaçını karşılar?",
 "tur":"tek_belge","zorluk":"kolay","beklenen_cevap":"1507 programında destek oranı her dönem için en fazla %75'tir.",
 "kanit_alintilari":[{"doc_id":"mevzuat-031","alinti":"Bu programda uygulanacak destek oranı her dönem için en fazla %75 olarak belirlenmiştir."}],
 "beklenen_belgeler":["mevzuat-031"],"beklenen_sayilar":["75"],"program_kodu":["1507"],
 "dogrulandi":false,"review_notes":"","uretim":"manuel"}
```

## Soru türleri ve dağılım

| Tür | Tanım | Hedef (ihtiyaclar.md §9) | Gerçek | Zorluk (kolay/orta/zor) |
|---|---|---|---|---|
| `tek_belge` | Cevap tek bir belgede; olgusal (oran, süre, tutar, koşul) | %40 | **43** | 11 / 21 / 11 |
| `cok_belge` | Cevap gerçekten iki belgenin birleşimini gerektirir (ör. 1501 Esasları + 2026-2 çağrısı; 5746 Kanunu + 1501 Esasları; İtiraz Yönetmeliği + 1501 Esasları) | %15 | **16** | 1 / 8 / 7 |
| `cevapsiz` | Kapsam içi ama cevabı corpus'ta yok (2027 tarihleri, kişiye özel proje durumu, eksik iki yönetmelik, ayrı dosyadaki ülke limitleri) | %15 | **15** | 3 / 10 / 2 |
| `kapsam_disi` | TÜBİTAK/Ar-Ge ile ilgisiz (hava, futbol, yemek, kod, sağlık, KDV iadesi, borsa, çeviri) | %10 | **10** | 4 / 4 / 2 |
| `surum` | Eski ve yeni değer/çağrı bulunan ya da değerin çağrı/YK kararıyla belirlendiği sorular; beklenen cevap en güncel kaynağı esas alır | %10 | **10** | 1 / 7 / 2 |
| `belirsiz` | Program belirtilmemiş ve cevap programa göre değişiyor (ör. "Destek oranı yüzde kaç?"); beklenen cevap programları ayırır | %10 | **10** | 2 / 6 / 2 |
| **Toplam** | | ~100 | **104** | 22 / 56 / 26 |

Kapsam: 32 farklı program kodu (1001, 1002, 1003, 1004, 1005, 1007, 3001, 3501; 1501, 1505,
1507, 1509, 1511, 1512, 1513, 1515, 1602, 1707, 1709, 1711, 1719, 1812, 1832; 2209, 2211, 2213,
2224; 4004, 4005, 4006, 4007, 4008) ve programdan bağımsız mevzuat (ARDEB İdari ve Mali Esaslar,
İtiraz Yönetmeliği, Panel Yönergesi, Yenilik Destek Programı Uygulama Esasları Yönergesi, 5746
ve 4691 sayılı Kanunlar). Kanıt gösterilen 45 belgenin türleri (belge-soru eşleşmesi sayısı):
çağrı/SSS 36, uygulama esasları 30, program sayfası 22, kanun 7, esaslar 5, kılavuz 4 (Mali
Rapor, Proje Öneri Bilgileri Formu, Ön Kayıt, 4007 Proje Hazırlama), yönerge 3, yönetmelik 2,
tablo (üst limit XLSX) 2. 104 sorunun 69'unda `beklenen_sayilar` dolu; toplam 117 alıntı,
ortalama 11 kelime.

**`surum` sorularının niteliği:** 10 sorudan yalnız 4'ü corpus içinde gerçek bir eski/yeni
değer çatışmasıdır (q059 1707 bütçesi 5 → 10 milyon TL, q060 1711 bütçesi 5 → 10 milyon TL,
q061 5746 yürürlük 2023 → 2028, q066 BiGG yatırımı 900.000 → 1.350.000 TL). Diğer 6'sı
değerin tarihli olduğu ("1 Şubat 2026 itibariyle"), çağrıya/YK kararına bağlı olduğu ya da
güncelleneceği duyurulan sorulardır; sistemin tarih/sürüm bilgisini cevaba taşıyıp taşımadığını ölçer.

## Üretim yöntemi

**Sorular sentetiktir:** başlangıçta adayların yerel `qwen3:4b` ile otomatik üretilmesi
düşünülmüştü; bunun yerine **tüm sorular bir dil modeli (Claude) tarafından ayrıştırılmış
belgeler okunarak doğrudan yazıldı**. `uretim: "manuel"` değeri "otomatik aday üreticisiyle
değil, doğrudan yazıldı" anlamındadır; soruları bir insan yazmadı ve henüz bir insan gözden
geçirmedi.

1. **Kaynak:** Ham PDF değil, ayrıştırılmış belgeler kullanıldı: `load_sources(data/manifests)`
   + `parse_all(..., ParsedCache(artifacts/cache/parsed))`. Böylece alıntılar sistemin gerçekte
   indekslediği metinle birebir aynıdır.
2. **Örnekleme (elle, tabakalı):** Her program ailesinden (TEYDEB 15xx–18xx, ARDEB 100x/3xxx,
   BİDEB 22xx, BİTO 400x) ve her belge türünden (uygulama esasları, çağrı metni, program sayfası
   + SSS, kılavuz, üst limit tablosu, kanun, yönetmelik/yönerge) belgeler seçildi; belge içinde
   başvuru sahibinin gerçekten soracağı olgular (oran, süre, tutar, tarih, koşul, istisna)
   aranıp soru haline getirildi. Soru metinleri kaynak cümleyi kopyalamaz; doğal başvuru
   dilinde yazıldı (ör. "kalıp seri üretimde de kullanılacak, maliyeti nasıl desteklenir?").
3. **Alıntı kuralı:** Her cevaplanabilir soruda 1–3 kısa (çoğunlukla 6–25 kelime) birebir
   alıntı. `evaluation.testset.validate_testset(questions, docs)` her alıntıyı
   `normalize_for_match(alinti) in normalize_for_match(doc.full_text())` ile doğrular.
   `normalize_for_match`: Unicode NFC + görünmez karakter temizliği → tipografik tırnak/tire
   birleştirme → satır sonu heceleme birleştirme (`proje-\nlerin` → `projelerin`) → Türkçe
   küçük harf → boşluk daraltma → kapanış noktalamasından önceki boşluğu silme. Satır sonu
   tiresi aynı büyüklükteki iki harf arasında birleştirilir (`TÜBİ-
TAK` → `TÜBİTAK`),
   büyük/küçük değişiminde korunur (`Ar-
Ge`). Bilerek eşdeğer *sayılmayanlar*: `% 60`/`%60`,
   `dâhil`/`dahil`, `1.350.000`/`1350000`, `on sekiz`/`18`. Kelimeler
   değişmez; yani alıntı "anlamca benzer" değil, gerçekten metinde geçen bir dizidir.
   Ayrıştırıcı kalitesi nedeniyle şu bölgelerden alıntı **alınmadı**: GİB kanun metinlerindeki
   "değişmeden önceki ibare/şekli" parçaları, son maddelere sızan değişiklik geçmişi tabloları,
   cümle ortasına karışmış sayfa dipnotları.
4. **Tür kuralları (otomatik):** cevaplanabilir türlerde ≥1 alıntı ve ≥1 beklenen belge,
   alıntı belgesi `beklenen_belgeler` içinde; `cok_belge` ve `belirsiz` için alıntılar en az iki
   farklı belgeden; `cevapsiz`/`kapsam_disi` için alıntı ve belge yok, cevap sabit metin;
   `beklenen_sayilar` içindeki her sayı `beklenen_cevap` içinde **tam sayı olarak** geçer
   (`contains_number`: "2" "24 ay" içinde bulunmaz, "350" "1.350.000" içinde bulunmaz; değerlendirmedeki
   kural tabanlı sayı kontrolü de aynı yardımcıyı kullanır). 4 kelimeden kısa alıntılar
   `quote_warnings` ile uyarı olarak raporlanır (şu an 0).
5. **`surum` sözleşmesi:** `beklenen_belgeler` yalnızca **en güncel** kaynağı içerir; eski
   değer (ör. 2024 tarihli üst limit tablosunda 1707 için 5.000.000 TL) beklenen cevapta ve
   `review_notes` içinde belirtilir. Böylece retrieval metriği güncel belgenin bulunmasını ödüllendirir.
6. **Cevapsız soruların tasarımı:** Corpus'ta gerçekten bulunmadığı tek tek aranarak
   doğrulandı (ör. "2211 aylık burs tutarı" corpus'ta yalnız süre olarak geçiyor). Eksik iki
   yönetmelik (TGB Uygulama Yönetmeliği, Ar-Ge Merkezleri Uygulama ve Denetim Yönetmeliği)
   için sorular `review_notes: "eksik veri: …"` taşır; bu belgeler eklenirse sorular
   cevaplanabilir hale gelir ve türü değiştirilmelidir. KOSGEB gibi corpus'ta dolaylı geçen
   konular cevapsız olarak kullanılmadı.

## Doğrulama

```powershell
# Şema + tür kuralları + dağılım (corpus gerekmez)
.\.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_testset.py -q

# Her alıntının ayrıştırılmış corpus'ta birebir geçtiği (data/raw + parse önbelleği gerekir)
.\.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_testset.py -m integration -q
```

Son doğrulama (review 1 düzeltmeleri sonrası): 104 soru, **0 hata**, 0 uyarı, `PARSER_VERSION = "2"` ile yenilenmiş parse önbelleğine (dipnot, değişiklik tablosu ve iki satırlı başlık düzeltmelerinden sonra; v1 önbelleğinde de 0 hata)
karşı. Ayrıştırıcı değiştiğinde (sürüm artışı) integration testi yeniden koşulmalı ve
eşleşmeyen alıntılar düzeltilmelidir.

İnceleme kuyruğu yeniden üretmek için:

```python
from pathlib import Path
from tubitak_rag.evaluation.testset import load_testset, write_review_queue
from tubitak_rag.ingestion.manifest import load_sources
from tubitak_rag.ingestion.parsed_cache import ParsedCache, parse_all

docs, _ = parse_all(load_sources(Path("data/manifests")), Path("data/raw"),
                    ParsedCache(Path("artifacts/cache/parsed")))
write_review_queue(load_testset(Path("eval/testset_v1.jsonl")),
                   {d.doc_id: d for d in docs}, Path("eval/review_queue.md"))
```

## Gözden geçirme süreci

1. `review_queue.md` türlere göre sıralıdır; her parti tek bir türden en fazla 10 soru içerir ve
   başlığında türü yazar (12 parti). Her
   maddede soru, beklenen cevap, beklenen sayılar, alıntı ve alıntının konumu (belge başlığı,
   madde/bölüm, sayfa) ile varsa yazar notu bulunur.
2. İnceleyici her soruda şunlara bakar: soru doğal ve tek anlamlı mı; beklenen cevap doğru ve
   tam mı; alıntı cevabı gerçekten kanıtlıyor mu; tür doğru mu (ör. `cok_belge` gerçekten iki
   belge gerektiriyor mu); `surum` sorusunda daha yeni bir kaynak var mı.
3. Onaylanan soruda `testset_v1.jsonl` içinde `dogrulandi: true` yapılır; düzeltilen veya
   şüpheli sorular `review_notes` ile işaretlenir. Silinen sorunun kimliği tekrar kullanılmaz.
4. Değişiklikten sonra iki doğrulama komutu yeniden koşulur.
5. Önce `review_notes` dolu 24 soru incelenmelidir (çoğu cevapsız/sürüm notları ve review 1
   düzeltmeleri).

## İnceleme geçmişi

- **Review 1 (Fable, bağımsız):** 69/78 cevaplanabilir soru doğru, 14/15 cevapsız soru gerçekten
  cevapsız bulundu. Düzeltmeler: q081 (TGB personelinin bölge dışı süresi) 4691 Geçici 2 ile
  cevaplandığı için `tek_belge` yapıldı, yerine q104 (teknopark kira üst sınırı) eklendi; q066
  eski tutarın kaynağı düzeltildi (900.000 TL önceki BiGG çağrı dönemleri); q071 kaynaksız
  ifade çıkarıldı; q002 `cok_belge` yapıldı ve q070 güncel 1501 çağrısını (yalnız KOBİ, kademeli
  oran) yansıtıyor; q005, q061, q063 ifadeleri netleştirildi; q078 ikinci temiz kaynak
  bulunamadığı için silindi, yerine q111 (ortaklı başvuru) eklendi; kılavuz ve Yenilik Destek
  Yönergesi kapsamı için q105–q110 eklendi, dengeyi korumak için başka sorularla örtüşen altı
  `tek_belge` sorusu silindi (q008, q013, q016, q030, q034, q041).

## Bilinen sınırlar

- **Tek yazar yanlılığı:** Soruların tamamını tek bir model (Claude) belgeleri okuyarak yazdı.
  Sorular belgedeki terimlerle örtüşmeye eğilimlidir; bu, sözcük tabanlı aramayı (BM25)
  gerçek kullanıcı sorularına göre kayırabilir. İnsan incelemesi ve ileride gerçek kullanıcı
  sorularıyla genişletme bu yanlılığı azaltır.
- **Yazar ve değerlendirici aynı model ailesi:** Raporlardaki doğruluk ve doğru ret sayıları
  cevapların Claude (Opus) ile tek tek değerlendirilmesinden gelir; soruları da Claude yazdı.
  Değerlendirilen cevapları sistemin dil modeli (`qwen3:4b`) üretti, Claude değil; yine de
  aynı modelin hem beklenen cevabı hem kararı vermesi bir yanlılık kaynağıdır. Uygulama içi
  otomatik hakem (`qwen3:4b`) isteğe bağlıdır ve raporlanan sayılarda kullanılmaz.
- **Zaman duyarlılığı:** Çağrı tarihleri ve limitler (2026-2 çağrıları, "1 Şubat 2026
  itibariyle" limitleri) zamanla eskir. `surum` ve tarih içeren sorular corpus her
  güncellendiğinde gözden geçirilmelidir.
- **Kaynak çelişkileri:** 1501 Uygulama Esasları büyük ölçekli kuruluşlar için %60 oran
  tanımlarken program sayfası ve 2026-2 çağrısı 1501'e yalnız KOBİ'lerin başvurabileceğini
  söylüyor; güncel çağrıda oran ilk 5 projede %75, sonrakilerde %60'tır (q002 `cok_belge`, q070
  `belirsiz` bunu yansıtır). Üst limit tablosu (XLSX) 2024-03 tarihlidir.
- **Eksik veri:** İki yönetmelik corpus'ta yok; ilgili cevapsız sorular bu belgeler eklenince
  güncellenmeli.
- **Ayrıştırma kalitesine bağımlılık:** Alıntılar ayrıştırılmış metne bağlıdır; bazı belge
  başlıkları manifestte "tıklayınız" gibi bağlantı metinleridir (inceleme kuyruğunda dosya adı
  yedeği kullanılır). Kılavuz türünden yalnız 4 belge kullanıldı; büyük kılavuzlar (Mali Rapor
  Hazırlama Kılavuzu gibi) daha fazla soruyu hak ediyor.
