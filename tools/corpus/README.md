# Corpus oluşturma betikleri (tarihsel kayıt)

Bu klasördeki betikler, `data/manifests/` altındaki manifestlerin **ilk kez nasıl oluşturulduğunu** gösterir
(2026-10-15). Uygulamanın kendisi corpus'u `rag fetch` komutuyla manifestlerden indirir; bu betikler yalnızca
öğrenme ve şeffaflık amacıyla tutulur.

| Betik | Ne yaptı? |
|---|---|
| `fetch_mevzuat.py` | tubitak.gov.tr/tr/mevzuat sayfasındaki 136 PDF'ten başvuruyla ilgili 60'ını indirdi. |
| `crawl_programs.py` | sitemap'ten 108 program sayfasını tarayıp dosya linklerini sınıflandırdı (plan modu). |
| `download_programs.py` | Program sayfalarının HTML'ini ve seçilmiş çağrı/SSS/kılavuz dosyalarını indirdi, sha256 ile tekilleştirdi. |
| `add_sha256.py` | (Aşama 2) Manifestlerde eksik `sha256` alanlarını `data/raw/` altındaki yerel dosyalardan hesaplayıp yerinde yazdı (60 + 105 + 2); `rag fetch` doğrulaması bu özetleri kullanır. |
| `prog_urls.txt` | Sitemap'ten çıkarılan program sayfası URL'leri. |

5746 ve 4691 sayılı kanunlar GİB mevzuat API'sinden (`gib.gov.tr/api/gibportal/mevzuat/madde/findByKanunId`) alındı.
mevzuat.gov.tr ve sanayi.gov.tr bu makineden otomatik erişime kapalıydı.
