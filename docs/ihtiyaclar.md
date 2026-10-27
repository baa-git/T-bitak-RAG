# TÜBİTAK Mevzuat RAG — İhtiyaçlar ve Tasarım Dokümanı

| | |
|---|---|
| **Tarih** | 2026-10-21 |
| **Durum** | Uygulandı — Docker paketi ve Hugging Face (Gradio + ZeroGPU) sürümü yayında |
| **Sahip** | Batur Alp |
| **Sürüm** | 2.3 |

Bu doküman projenin **ne** yapacağını ve **hangi kararlarla** yapılacağını tanımlar. Her ihtiyacın bir kimliği vardır (`FR-…`, `NFR-…`). Durum takibi [`ihtiyaclar.xlsx`](ihtiyaclar.xlsx) dosyasında yapılır.

---

## 1. Amaç

Türkçe bir mevzuat corpus'u (TÜBİTAK destek programları, Ar-Ge ve teknopark mevzuatı) üzerinde çalışan, **indirilip yerelde Docker ile kullanılabilen**, modüler bir RAG uygulaması.

Projenin iki amacı vardır:

1. **Kullanılabilir araç:** Kullanıcı TÜBİTAK başvuru süreçleri ve mevzuatı hakkında soru sorar; sistem kaynak (belge + madde/sayfa) göstererek cevap verir, bilmediğinde "bilmiyorum" der.
2. **Öğretici deney platformu:** RAG teknikleri (chunking, dense/BM25/hibrit arama, reranking) adım adım eklenir, her adım aynı test setiyle ölçülür ve sistem her deney için **ders niteliğinde Markdown rapor** üretir.

Proje GitHub'da açık kaynak paylaşılacaktır; kod clean code ilkelerine göre yazılır.

### 1.1 Başarı ölçütleri

- Tek komutla (`docker compose up`) ayağa kalkar; GPU'suz bir bilgisayarda varsayılan ayarlarla çalışır.
- 6 deney koşusu (bkz. §8) tek komutla tekrar üretilebilir ve her biri bir MD rapor üretir.
- Test setindeki "cevabı olmayan" sorularda sistem uydurmaz, kaynak döndürmez.
- Her modülün birim testi vardır; lint, biçim, tip denetimi ve birim testleri hatasız geçer.
- Her deney basamağının yöntemi ve ölçülen sonucu ders anlatımı biçiminde belgelenir ([`RAG-GELISTIRME-REHBERI.md`](RAG-GELISTIRME-REHBERI.md)).

---

## 2. Kapsam

### 2.1 Kapsam içi

- Corpus'un manifest'ten indirilmesi ve doğrulanması
- PDF, HTML ve GİB JSON kaynaklarının ayrıştırılması ve normalizasyonu
- 2 chunking stratejisi, dense + BM25 + hibrit arama, reranker
- Kaynak gösteren cevap üretimi ve "bilmiyorum" guardrail'leri
- 104 soruluk sentetik test seti (Claude yazdı; insan incelemesi bekliyor); arama ve cevap için ayrı değerlendirme
- Deney koşturucu + MD rapor üretici
- Artımlı (incremental) indexleme
- FastAPI REST API + sohbet biçiminde sade web arayüzü (cevap başına numaralı kaynaklar, Hakkında sayfası)
- Docker Compose paketi (CPU ve GPU); hazır indeks paketi Hugging Face'ten indirilir, ingest gerekmez
- Hugging Face Spaces'te barındırılan sürüm (Gradio + ZeroGPU, Qwen3 4B Space GPU'sunda)

### 2.2 Kapsam dışı (ilk sürüm)

- Recursive / semantik / parent-child chunking, overlap varyasyonları
- Sorgu dönüşümü (HyDE, multi-query), agentic RAG, GraphRAG
- Görsel (multimodal) embedding deneyi
- Kullanıcı hesabı, yetkilendirme, çok kullanıcılı kullanım
- Eski çağrı arşivinin (1.309 sayfa) corpus'a alınması
- Model eğitimi / fine-tuning

Bunlar ileride yeni deney basamağı olarak eklenebilecek şekilde tasarlanır (bkz. NFR-MOD).

---

## 3. Kullanıcılar ve senaryolar

| Kullanıcı | Senaryo |
|---|---|
| **Başvuru hazırlayan kişi** | "1507'de destek oranı ve azami süre nedir?" diye sorar; madde numarasıyla cevap alır. |
| **RAG öğrenen geliştirici** | Repoyu indirir, deneyleri koşturur, raporları okuyarak hangi tekniğin neyi değiştirdiğini öğrenir. |
| **Aracı çalıştıran kişi** | Docker ile (CPU ya da GPU) kendi bilgisayarında çalıştırır ya da Hugging Face Space'te kurulumsuz dener. |
| **Katkıda bulunan geliştirici** | Yeni bir reranker / embedder ekler; tek dosya + tek kayıt satırıyla pipeline'a bağlar. |

---

## 4. Corpus (veri)

Mevcut durum (2026-10-15 itibarıyla indirilmiş, `data/raw/` altında):

| Kaynak | İçerik | Biçim | Boyut |
|---|---|---|---|
| tubitak.gov.tr/tr/mevzuat | Başvuruyla ilgili 60 doküman (program esasları, yönetmelikler, idari-mali esaslar, süreç yönergeleri) | PDF | 25,9 MB |
| tubitak.gov.tr program sayfaları | 105 sayfa HTML'i + 114 PDF + 1 XLSX (çağrı metinleri, SSS, başvuru kılavuzları, destek üst limitleri) | HTML/PDF/XLSX | 131 MB (~3,1 MB metin) |
| gib.gov.tr | 5746 ve 4691 sayılı kanunların güncel konsolide metni | JSON (madde bazlı) | 145 KB |

Bilinen özellikler ve sorunlar (ingestion bunları ele almalıdır):

- Mevzuat hiyerarşisi: Kısım → Bölüm → Madde → fıkra `(1)` → bent `a)` → alt bent `1)`; bazı belgelerde bölüm numaraları her kısımda baştan başlar.
- Yüzlerce değişiklik notu (`(Değişik: …-77YK)`, `(Ek: …)`, `(Mülga: …)`), boş mülga maddeler, sonda değişiklik geçmişi tablosu.
- Sayfa altlığı/üstlüğü, sayfa numarası, madde ortasına giren dipnotlar.
- Yenilik Destek Yönergesi'nde ~7.900 ayrık Unicode birleştirici işaret → NFC normalizasyonu şart.
- 1501 / 1507 / Yenilik Yönergesi'nde madde başlıkları birebir aynı → program bilgisi kaybolursa yanlış belgeden cevap riski.
- 3 PDF taranmış (metin çıkmıyor; ingest raporunda listelenir), 7 doküman İngilizce.
- Eksik: TGB Uygulama Yönetmeliği ve Ar-Ge Merkezleri Uygulama ve Denetim Yönetmeliği (sanayi.gov.tr otomatik indirmeyi engelliyor; engel aşılmaz, elle eklenmeli).

---

## 5–6. İhtiyaçlar (fonksiyonel ve fonksiyonel olmayan)

İhtiyaçlar EARS kalıbına göre, her biri tek bir yükümlülük içeren cümlelerle yazılmıştır.
Tek kaynak [`requirements/ihtiyaclar.yaml`](requirements/ihtiyaclar.yaml); takip matrisi [`ihtiyaclar.xlsx`](ihtiyaclar.xlsx).

<!-- IHTIYACLAR:BASLA -->

> Bu bölüm `docs/requirements/build.py` tarafından `ihtiyaclar.yaml` dosyasından üretilir; elle düzenlemeyin.
> Takip için Excel: [`docs/ihtiyaclar.xlsx`](ihtiyaclar.xlsx).


### Ayrıştırma ve temizleme

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-P1 | Her zaman | Sistem, PDF, HTML, XLSX ve GİB JSON kaynaklarını ortak bir belge modeline dönüştürmelidir. | Zorunlu | Test | Karşılandı |
| FR-P3 | Her zaman | Sistem, çıkarılan metinden sayfa numaralarını, tekrarlayan üstlük/altlıkları ve bozuk karakter kodlamalarını temizlemelidir. | Zorunlu | Test | Karşılandı |
| FR-P4 | Her zaman | Sistem, mevzuat belgelerinden kısım, bölüm, madde, fıkra ve bent hiyerarşisini çıkarmalıdır. | Zorunlu | Test | Karşılandı |
| FR-P5 | Her zaman | Sistem, değişiklik notlarını madde metninden ayırıp metadata olarak saklamalıdır. | Önemli | Test | Karşılandı |
| FR-P6 | Her zaman | Sistem, yürürlükten kaldırılmış (mülga) maddeleri indeks dışında bırakmalıdır. | Önemli | Test | Karşılandı |
| FR-P9 | İstenmeyen | Bir PDF'ten metin çıkarılamadığı durumunda, sistem belgeyi ingest raporunda 'taranmış' olarak listelemelidir. | Önemli | Test | Karşılandı |
| FR-P10 | Her zaman | Sistem, her belgeye kimlik, başlık, program kodu, belge türü, kaynak URL, sürüm tarihi, dil ve SHA-256 metadata alanlarını atamalıdır. | Zorunlu | Test | Kısmi |

### Parçalama (chunking)

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-C1 | Her zaman | Sistem, sabit pencereli parçalamada parça boyutunun ve örtüşmenin token cinsinden yapılandırılmasına izin vermelidir. | Zorunlu | Test | Karşılandı |
| FR-C2 | Opsiyonel | Yapı bazlı parçalama etkinleştirildiğinde, sistem her maddeyi, her HTML bölümünü ve her SSS soru-cevap çiftini ayrı bir parça olarak üretmelidir. | Zorunlu | Test | Karşılandı |
| FR-C6 | Opsiyonel | Bağlam başlığı seçeneği etkinleştirildiğinde, sistem her parçanın embed edilen metninin başına belge ve hiyerarşi bilgisini eklemelidir. | Zorunlu | Test | Karşılandı |

### İndeksleme

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-I2 | Olay | Corpus'ta bir belge eklendiğinde, değiştiğinde veya silindiğinde, sistem yalnızca o belgenin dense ve BM25 indeks kayıtlarını güncellemelidir. | Zorunlu | Test | Karşılandı |
| FR-X9 | Durum | Bir belgenin içeriği ve işlem ayarları değişmemişken, sistem ayrıştırma, parçalama ve embedding sonuçlarını önbellekten yeniden kullanmalıdır. | Zorunlu | Test | Karşılandı |

### Arama ve yeniden sıralama

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-R1 | Her zaman | Sistem, arama yönteminin dense, BM25 veya hibrit olarak yapılandırmayla seçilmesine izin vermelidir. | Zorunlu | Test | Karşılandı |
| FR-R3 | Opsiyonel | Kök bulma seçeneği etkinleştirildiğinde, sistem BM25 belirteçlerine Türkçe kök bulma uygulamalıdır. | Önemli | Test | Karşılandı |
| FR-R4 | Opsiyonel | Hibrit arama etkinleştirildiğinde, sistem dense ve BM25 sonuç sıralarını Reciprocal Rank Fusion ile birleştirmelidir. | Zorunlu | Test | Karşılandı |
| FR-R5 | Opsiyonel | Bir soruyla birlikte metadata filtresi verildiğinde, sistem sonuçları program kodu ve belge türüne göre süzmelidir. | Önemli | Test | Karşılandı |
| FR-RR1 | Opsiyonel | Yeniden sıralayıcı etkinleştirildiğinde, sistem ilk aşama adaylarını cross-encoder ile yeniden sıralayıp en iyi n parçayı döndürmelidir. | Zorunlu | Test | Karşılandı |

### Cevap üretimi ve atıf

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-G3 | Her zaman | Sistem, cevaptaki her atıfı belge adı, madde veya sayfa bilgisi ve kaynak bağlantısıyla birlikte sunmalıdır. | Zorunlu | Test | Karşılandı |
| FR-G4 | Her zaman | Sistem, cevap üreten dil modelinin yapılandırmayla Ollama üzerinden yerel bir model veya OpenAI uyumlu bir API (ör. OpenRouter, Groq, Hugging Face Inference Providers) olarak seçilmesine izin vermelidir. | Zorunlu | Test | Karşılandı |
| FR-G7 | Her zaman | Sistem, değerlendirmede kullanılan hakem modelinin sağlayıcısının ve modelinin cevap modelinden bağımsız olarak yapılandırmayla seçilmesine izin vermelidir. | Önemli | Test | Karşılandı |
| FR-G5 | Her zaman | Sistem, her cevabı bilginin resmî görüş olmadığını belirten bir uyarıyla birlikte sunmalıdır. | Önemli | Test | Karşılandı |
| FR-G6 | Her zaman | Sistem, kaynak metinleri dil modeline talimat olarak değil veri olarak işaretleyerek vermelidir. | Önemli | Test | Karşılandı |

### Güvenlik önlemleri (guardrail)

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-GR1 | İstenmeyen | En yüksek arama skoru yapılandırılmış eşiğin altında kaldığı durumunda, sistem dil modelini çağırmadan 'bilgi bulunamadı' cevabı döndürmelidir. | Zorunlu | Test | Karşılandı |
| FR-GR2 | İstenmeyen | Sistem bir soruyu reddettiği durumunda, sistem cevapla birlikte hiçbir kaynak göstermemelidir. | Zorunlu | Test | Karşılandı |
| FR-GR3 | İstenmeyen | Dil modeli atıfsız bir cevap veya verilmemiş bir kaynağa atıf ürettiği durumunda, sistem cevabı 'bilgi bulunamadı' cevabına dönüştürmelidir. | Zorunlu | Test | Karşılandı |
| FR-GR4 | İstenmeyen | Soru TÜBİTAK destek programları ve Ar-Ge mevzuatı kapsamı dışında olduğu durumunda, sistem soruyu kapsam dışı olduğunu belirterek reddetmelidir. | Zorunlu | Test | Karşılandı |
| FR-GR5 | Durum | Soru kaynaklarla yalnızca kısmen cevaplanabilirken, sistem cevaplanamayan kısmı cevapta açıkça belirtmelidir. | Önemli | Test | Karşılandı |
| FR-GR6 | Durum | Atıf yapılan kaynak bilgiyi çağrı duyurusuna yönlendirirken, sistem cevapta ilgili duyuru sayfasının bağlantısını vermelidir. | Önemli | Test | Karşılandı |
| FR-GR7 | Durum | Aynı konuda farklı sürüm tarihli kaynaklar bulunurken, sistem cevabı en güncel sürüm tarihli kaynağa dayandırmalıdır. | Önemli | Test | Kısmi |
| FR-GR8 | Durum | Program belirtilmeyen bir sorunun cevabı programa göre değişirken, sistem cevabı programlara göre ayrıştırarak vermelidir. | Önemli | Test | Karşılanmadı |
| FR-GR11 | İstenmeyen | Cevap kaynaklarda geçmeyen bir sayı içerdiği durumunda, sistem o cümleyi cevaptan çıkarmalıdır. | Zorunlu | Test | Kısmi |

### Test seti

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-E1 | Her zaman | Test seti, TÜBİTAK başvuru süreçleri ve mevzuatı hakkında en az 100 soru içermelidir. | Zorunlu | Test | Karşılandı |
| FR-E3 | Her zaman | Test seti, corpus'tan cevaplanması gereken tek belgeli ve çok belgeli sorular içermelidir. | Zorunlu | Analiz | Karşılandı |
| FR-E9 | Her zaman | Test seti, sistemin reddetmesi gereken kapsam dışı ve cevabı corpus'ta bulunmayan sorular içermelidir. | Zorunlu | Analiz | Karşılandı |
| FR-E11 | Her zaman | Test seti, program belirtilmeden sorulan ve cevabı programa göre değişen belirsiz sorular içermelidir. | Önemli | Analiz | Karşılandı |
| FR-E12 | Her zaman | Test seti, eski ve yeni değeri bulunan bilgilere ilişkin sürüm soruları içermelidir. | Önemli | Analiz | Karşılandı |
| FR-E16 | Her zaman | Test seti, her sorunun cevaplanması mı yoksa reddedilmesi mi gerektiğini belirtmelidir. | Zorunlu | Test | Karşılandı |
| FR-E22 | Her zaman | Test setindeki her soru, kısa bir beklenen cevap içermelidir. | Zorunlu | Test | Karşılandı |
| FR-E13 | Her zaman | Test setindeki her cevaplanabilir soru, corpus metninde birebir geçen 1–3 kanıt alıntısı içermelidir. | Zorunlu | Test | Karşılandı |
| FR-E23 | Her zaman | Test setindeki her cevaplanabilir soru, cevabın dayandığı beklenen belgeleri listelemelidir. | Zorunlu | Test | Karşılandı |
| FR-E24 | Durum | Beklenen cevap bir oran, tutar, süre veya tarih içerirken, test seti bu sayıları beklenen sayılar alanında listelemelidir. | Önemli | Test | Karşılandı |
| FR-E4 | Her zaman | Test seti dokümantasyonu, soruların nasıl üretildiğini, şemasını, tür dağılımını ve inceleme durumunu açıklamalıdır. | Önemli | İnceleme | Karşılandı |
| FR-E14 | Her zaman | Test setindeki her soru, sonuçları raporlanmadan önce bir insan tarafından gözden geçirilmelidir. | Önemli | İnceleme | Karşılanmadı |

### Arama değerlendirmesi

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-E5 | Olay | Getirilen parçalar beklenen belgedeki kanıt alıntısını içerdiğinde, Değerlendirme modülü aramayı isabetli saymalıdır. | Zorunlu | Test | Karşılandı |
| FR-E6 | Her zaman | Değerlendirme modülü, cevaplanabilir sorular için Recall@5, Recall@10 ve MRR değerlerini hesaplamalıdır. | Zorunlu | Test | Karşılandı |
| FR-E26 | Her zaman | Değerlendirme modülü, beklenen belgelerden en az birinin ilk 5 ve ilk 10 sonuç içinde gelme oranını (belge isabeti) hesaplamalıdır. | Önemli | Test | Karşılandı |
| FR-E27 | Her zaman | Değerlendirme modülü, arama metriklerini soru türüne göre ayrı ayrı raporlamalıdır. | Önemli | Test | Karşılandı |
| FR-E29 | Her zaman | Değerlendirme modülü, her cevaplanabilir soru için kanıtın aday havuzundaki ilk sırasını kaydetmelidir. | Önemli | Test | Karşılandı |
| FR-E30 | Her zaman | Rapor üretici, her yapılandırma için kanıtın ilk sırasının dağılımını (1. sıra, 2–5, 6–30, havuzda yok) göstermelidir. | İsteğe bağlı | İnceleme | Kısmi |

### Cevap değerlendirmesi

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-E7 | Her zaman | Cevap değerlendirmesi, her yapılandırmanın cevaplanabilir sorulara verdiği her cevabı beklenen cevap ve atıf yapılan kaynaklarla karşılaştırarak doğru veya yanlış olarak işaretlemelidir. | Zorunlu | Test | Karşılandı |
| FR-E17 | Her zaman | Cevap değerlendirmesi, cevap verilen her soruda cevabın her iddiasının atıf yapılan kaynaklarda geçip geçmediğini (sadakat) işaretlemelidir. | Zorunlu | Test | Karşılandı |
| FR-E18 | Her zaman | Cevap değerlendirmesi, reddedilmesi gereken sorularda doğru ret oranını ölçmelidir. | Zorunlu | Test | Karşılandı |
| FR-E19 | Her zaman | Cevap değerlendirmesi, cevaplanabilir sorularda gereksiz ret oranını ölçmelidir. | Zorunlu | Test | Karşılandı |
| FR-E31 | Olay | Cevaplanabilir bir soru reddedildiğinde, cevap değerlendirmesi reddin hangi aşamadan kaynaklandığını (arama, üretim ve denetim, belirsiz) kaydetmelidir. | Önemli | Test | Karşılandı |
| FR-E32 | İstenmeyen | Reddedilmesi gereken bir soruya dayanaksız bir cevap verildiği durumunda, cevap değerlendirmesi bu cevabı uydurma olarak saymalıdır. | Önemli | Test | Karşılandı |
| FR-E20 | Her zaman | Değerlendirme modülü, beklenen sayıların (oran, tutar, süre) cevapta geçip geçmediğini kural tabanlı olarak denetlemelidir. | Önemli | Test | Karşılandı |
| FR-E33 | Her zaman | Değerlendirme modülü, cevaptaki atıfların beklenen kanıtı içeren kaynaklara işaret etme oranını (atıf kesinliği) kural tabanlı olarak ölçmelidir. | Önemli | Test | Karşılandı |
| FR-E21 | Her zaman | Değerlendirme modülü, kaynaklarda bulunmayan (uydurma) sayı içeren cevapların oranını ölçmelidir. | Önemli | Test | Kısmi |
| FR-E34 | Opsiyonel | Otomatik hakem etkinleştirildiğinde, Değerlendirme modülü cevap doğruluğunu ve sadakati yapılandırılmış hakem modeliyle puanlamalıdır. | İsteğe bağlı | Test | Karşılandı |
| FR-GR10 | Her zaman | Değerlendirme modülü, güvenlik önlemi eşiği için farklı eşik değerlerindeki doğru ret ve yanlış ret oranlarına dayanarak bir eşik önermelidir. | Önemli | Gösterim | Karşılandı |

### Deney karşılaştırma ve dokümantasyon

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-X1 | Her zaman | Sistem, bir deneyi tek bir YAML yapılandırma dosyasıyla tanımlayıp tek bir komutla çalıştırmalıdır. | Zorunlu | Gösterim | Karşılandı |
| FR-X3 | Her zaman | Rapor üretici, her koşu için yöntemi, önceki basamakla karşılaştırmalı ana metrikleri ve iyi ve kötü örnek soruları içeren kısa bir Markdown rapor üretmelidir. | Zorunlu | İnceleme | Karşılandı |
| FR-X8 | Her zaman | Rapor üretici, tüm deney yapılandırmalarını aynı test setinde arama ve cevap metrikleriyle karşılaştıran bir özet rapor üretmelidir. | Önemli | İnceleme | Karşılandı |
| FR-X12 | Her zaman | Proje dokümantasyonu, her deney yapılandırmasının yöntemini ve ölçülen sonuçlarını ders anlatımı biçiminde açıklamalıdır. | Önemli | İnceleme | Karşılandı |

### Web arayüzü

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-U2 | Her zaman | Web arayüzü, soruları ve cevapları aynı sayfada sohbet akışı olarak göstermelidir. | Zorunlu | Gösterim | Karşılandı |
| FR-U7 | Her zaman | Web arayüzü, her cevabın altında yalnızca o cevapta atıf yapılan kaynakları numaralı olarak listelemelidir. | Zorunlu | Test | Karşılandı |
| FR-U8 | Her zaman | Web arayüzü, kullanıcının cevabı üretecek yapılandırmayı (v0–v3b) yöntemleri açıkça yazılmış bir seçiciden seçmesine izin vermelidir. | Önemli | Test | Karşılandı |
| FR-U3 | Her zaman | Web arayüzü, corpus kaynaklarını, belge sayılarını, kullanılan modelleri, yöntemleri ve güvenlik önlemlerini anlatan bir Hakkında sayfası sunmalıdır. | Önemli | Gösterim | Karşılandı |
| FR-U11 | Olay | Kullanıcı bir kaynağın önizleme bağlantısına tıkladığında, Web arayüzü belgenin kimlik bilgilerini ve ayrıştırılmış bölümlerini göstermelidir. | İsteğe bağlı | Test | Karşılandı |
| FR-U5 | Her zaman | Web arayüzü, betik, stil ve yazı tiplerini harici bir sunucuya bağlanmadan ve ayrı bir frontend derleme adımı gerektirmeden sunmalıdır. | Önemli | Test | Karşılandı |

### Barındırılan sürüm (Hugging Face Spaces)

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-HF1 | Her zaman | Barındırılan sürüm, Hugging Face Spaces'te ZeroGPU üzerinde çalışan bir Gradio uygulaması olarak cevap modelini (Qwen3 4B), embedding modellerini ve yeniden sıralayıcıyı Space'in GPU'sunda çalıştırmalıdır. | Önemli | Gösterim | Karşılandı |
| FR-HF2 | Her zaman | Barındırılan sürüm, Qdrant ve Ollama olmadan önceden hesaplanmış NumPy indeks paketiyle arama yapmalıdır. | Önemli | Test | Karşılandı |
| FR-HF3 | Her zaman | Barındırılan sürüm, web arayüzüyle aynı sohbet akışını, yapılandırma seçicisini ve cevap başına numaralı kaynak listesini sunmalıdır. | Önemli | Test | Karşılandı |

### REST API

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-A1 | Her zaman | REST API, soru sorma, belge listeleme, ingest başlatma, koşu listeleme, koşu başlatma, koşu ayrıntısı, rapor ve sağlık kontrolü uç noktalarını OpenAPI belgeleriyle sunmalıdır. | Zorunlu | Test | Karşılandı |
| FR-A4 | İstenmeyen | Bir istek geçersiz olduğu veya işlenirken hata oluştuğu durumunda, REST API Türkçe ve tutarlı bir {hata, detay} gövdesi döndürmelidir. | Önemli | Test | Karşılandı |
| FR-A5 | İstenmeyen | Yetkisiz bir istek ingest veya deney koşusu başlatmaya çalıştığı durumunda, REST API isteği işi başlatmadan reddetmelidir. | Önemli | Test | Karşılandı |

### Komut satırı arayüzü

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| FR-CLI1 | Her zaman | Komut satırı arayüzü, fetch, ingest, run, report, ask, serve, export-index ve export-space komutlarını sunmalıdır. | Zorunlu | Test | Karşılandı |
| FR-CLI2 | Her zaman | Komut satırı arayüzü, kurulu indeksi başka bir makinede yüklenebilecek, SHA-256 özetli bir pakete aktarmalıdır. | Önemli | Test | Karşılandı |
| FR-CLI3 | Olay | Yapılandırma ücretli bir harici dil modeli API'si seçtiğinde, Komut satırı arayüzü deney koşusu başlamadan önce isteklerin dış servise gideceğini ve ücretlendirileceğini bildirmelidir. | Önemli | Test | Karşılandı |

### Docker paketi ve donanım

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| NFR-DOCKER1 | Her zaman | Docker paketi, uygulamayı, Qdrant'ı ve Ollama'yı gerekli modelleri indirerek tek bir docker compose komutuyla başlatmalıdır. | Zorunlu | Gösterim | Karşılandı |
| NFR-DOCKER4 | Opsiyonel | GPU eki etkinleştirildiğinde, Docker paketi CUDA'lı PyTorch içeren imajla modelleri NVIDIA GPU üzerinde çalıştırmalıdır. | Önemli | Gösterim | Karşılandı |
| NFR-DOCKER5 | Olay | Hazır indeks paketi bulunduğunda, Docker paketi belge indirme, ayrıştırma ve embedding adımlarını atlayıp indeksi paketten yüklemelidir. | Önemli | Test | Karşılandı |
| NFR-DOCKER6 | Opsiyonel | HF_TOKEN ortam değişkeni tanımlandığında, Docker paketi bu token'ı imaja yazmadan model indirmelerinde kullanmalıdır. | İsteğe bağlı | Test | Karşılandı |
| NFR-HW1 | Her zaman | Sistem, varsayılan yapılandırmayla GPU bulunmayan sıradan bir bilgisayarda çalışmalıdır. | Zorunlu | Gösterim | Karşılandı |
| NFR-HW2 | Durum | Cihaz ayarı 'auto' iken, sistem modelleri uyumlu bir GPU varsa GPU'da, yoksa CPU'da çalıştırmalıdır. | Önemli | Test | Karşılandı |
| NFR-PERF1 | Her zaman | Sistem, varsayılan CPU modunda tek bir soru için arama ve yeniden sıralamayı 5 saniyenin altında tamamlamalıdır. | Önemli | Analiz | Karşılandı |

### Yazılım kalitesi ve testler

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| NFR-MOD2 | Her zaman | Sistem, yeni bir embedding, reranker, dil modeli veya vektör veritabanı uygulamasının tek bir dosya ve tek bir kayıt satırıyla eklenmesine izin vermelidir. | Önemli | Test | Karşılandı |
| NFR-MOD3 | Her zaman | Çekirdek paket, hiçbir web çatısını (FastAPI, Gradio) içe aktarmamalıdır. | Zorunlu | Test | Karşılandı |
| NFR-TEST1 | Her zaman | Proje deposu, her kaynak modülü için bir birim testi modülü içermelidir. | Zorunlu | Test | Karşılandı |
| NFR-TEST2 | Her zaman | Birim testleri, harici servis, model indirmesi veya API anahtarı gerektirmeden çalışmalıdır. | Zorunlu | Test | Karşılandı |
| NFR-TEST3 | Her zaman | Çekirdek paketin birim testi satır kapsamı en az %80 olmalıdır. | Önemli | Test | Karşılandı |

### Güvenlik, gizlilik ve dil

| ID | Tip | İhtiyaç | Öncelik | Doğrulama | Durum |
|---|---|---|---|---|---|
| NFR-PRIV1 | Her zaman | Sistem, varsayılan yapılandırmada soruları ve belgeleri hiçbir harici servise göndermemelidir. | Zorunlu | İnceleme | Karşılandı |
| NFR-SEC1 | Her zaman | Web arayüzü, kullanıcı girdisini ve kaynak metinlerini HTML olarak yorumlamadan göstermelidir. | Zorunlu | Test | Karşılandı |
| NFR-CONF1 | İstenmeyen | Yapılandırma dosyası geçersiz olduğu durumunda, sistem çalışmaya başlamadan anlaşılır bir hata mesajıyla durmalıdır. | Zorunlu | Test | Karşılandı |
| NFR-CONF2 | Her zaman | Sistem, gizli anahtarları yalnızca ortam değişkenlerinden veya .env dosyasından okumalıdır. | Zorunlu | İnceleme | Karşılandı |
| NFR-LANG1 | Her zaman | Sistem, kullanıcı arayüzünü, hata mesajlarını, komut çıktılarını ve raporları Türkçe sunmalıdır. | Zorunlu | İnceleme | Karşılandı |
| NFR-LIC1 | Her zaman | Proje deposu, MIT lisans dosyasını ve kullanılan modellerin lisans listesini içermelidir. | Önemli | İnceleme | Kısmi |

<!-- IHTIYACLAR:BITIS -->

## 7. Mimari

### 7.1 Katmanlar

```
interfaces/   CLI · FastAPI (API + HTML sayfalar) · Gradio (Hugging Face Space)
      │  yalnızca çekirdeği çağırır
core          domain · ports · ingestion · chunking · indexing · retrieval ·
              reranking · generation · guardrails · evaluation · reporting ·
              pipeline · factory · config
      │  ports (Protocol) üzerinden
adapters      Qdrant · NumPy vektör deposu (Space) · bm25s · HuggingFace (sentence-transformers)
              LLM: Ollama · OpenAI uyumlu API · Anthropic · transformers (Space)
```

### 7.2 Dosya yapısı

```
tubitak-rag/
├── README.md · LICENSE · pyproject.toml · Dockerfile · docker-compose.yml · .env.example
├── configs/  default.yaml · experiments/{v0,v0r,v1,v2,v3,v3b}.yaml
├── data/     manifests/ (repoda) · raw/ (gitignore)
├── eval/     testset_v1.jsonl · README.md (üretim yöntemi)
├── artifacts/ (gitignore)  cache/{parsed,chunks,embeddings,bm25}/ · runs/<run_id>/
├── docs/     ihtiyaclar.md · mimari.md · RAG-GELISTIRME-REHBERI.md · sample-reports/
├── src/tubitak_rag/
│   ├── domain/  ports/  ingestion/  chunking/  indexing/  retrieval/  reranking/
│   ├── generation/  guardrails/  evaluation/  reporting/
│   ├── pipeline.py · factory.py · config.py
│   └── interfaces/  cli.py · web/{api.py, pages.py, templates/, static/}
└── tests/    unit/ (src aynası) · integration/ · fixtures/
```

### 7.3 Chunk metadata şeması

```json
{
  "chunk_id": "…", "doc_id": "…", "doc_title": "…",
  "program_kodu": "1507", "doc_turu": "uygulama_esaslari",
  "hiyerarsi": "Üçüncü Bölüm › Madde 7 – Destek oranı › (2)",
  "madde_no": "7", "sayfa": 6, "bolum": null,
  "kaynak_url": "…", "surum_tarihi": "2026-09-25", "dil": "tr",
  "doc_sha256": "…", "chunker": "structural", "text": "…", "embed_text": "…"
}
```

### 7.4 Sorgu akışı

```
soru → [kapsam kontrolü] → retriever (dense | bm25 | hibrit) → reranker (ops.)
     → [retrieval eşiği] → prompt (numaralı kaynaklar) → LLM (yapılandırılmış çıktı)
     → [atıf doğrulama] → Answer {cevaplanabilir, cevap, atiflar, kaynaklar, gecikme}
```

---

## 8. Deney merdiveni

| Koşu | Chunking | Arama | Reranker | Embedding | Öğretilen ders |
|---|---|---|---|---|---|
| **v0** | Sabit 512 / 64 overlap | Dense | – | EmbeddingGemma 2 | Naive RAG nereye kadar gider? |
| **v0r** | Sabit 512 / 64 overlap | Dense | bge-reranker-v2-m3 | EmbeddingGemma 2 | Reranker kötü chunking'i kurtarabilir mi? |
| **v1** | Yapı bazlı + bağlam başlığı | Dense | – | EmbeddingGemma 2 | Yapı ve bağlam neden önemli? |
| **v2** | Yapı bazlı + bağlam başlığı | Hibrit (BM25 + dense, RRF) | – | EmbeddingGemma 2 | Kelime eşleşmesi ne kazandırır? (kodlar, sayılar) |
| **v3** | Yapı bazlı + bağlam başlığı | Hibrit | bge-reranker-v2-m3 | EmbeddingGemma 2 | Endüstri standardı tam pipeline |
| **v3b** | v3 ile aynı | Hibrit | bge-reranker-v2-m3 | **bge-m3** | Embedding modeli ne kadar fark eder? |

Toplam 6 koşu, 3 embedding hesaplaması (2 chunking × EmbeddingGemma 2 + 1 × bge-m3).

---

## 9. Değerlendirme yöntemi

- **Test seti** (`eval/testset_v1.jsonl`): 104 sentetik soru (Claude yazdı, insan incelemesi bekliyor). Cevaplanması gereken 79 soru: 43 tek belgeli, 16 çok belgeli, 10 sürüm, 10 belirsiz. Reddedilmesi gereken 25 soru: 15 cevabı corpus'ta olmayan, 10 kapsam dışı. Her soruda beklenen cevap; cevaplanabilir sorularda beklenen belgeler ve 1–3 birebir kanıt alıntısı; 69 soruda beklenen sayılar.
- **Arama testi**: Kanıt alıntısı getirilen parçalarda geçiyorsa arama isabetlidir; bu ölçüm parçalama stratejisinden bağımsızdır. Recall@5, Recall@10, MRR ve belge isabeti hem tüm set hem soru türü için, her yapılandırma ayrı ayrı hesaplanır; kanıtın 30 aday içindeki ilk sırası (1 / 2–5 / 6–30 / yok) hangi katmanın neyi düzelttiğini gösterir.
- **Cevap testi**: Her yapılandırmanın her cevabı Claude (Opus) ile beklenen cevap ve atıf yapılan kaynaklarla karşılaştırılır: doğruluk (79 soru), sadakat, doğru ret (25 soru), gereksiz ret ve hatanın yeri (arama / üretim ve denetim / belirsiz), uydurma cevaplar. Ek olarak kural tabanlı denetimler: beklenen sayılar cevapta var mı, atıflar kanıtı içeren kaynaklara mı işaret ediyor, sayı denetimi kaç cevabı düzeltti. Uygulama içi otomatik hakem (`eval.judge`) isteğe bağlıdır; raporlanan sayılarda kullanılmaz.
- **Eşik kalibrasyonu**: Guardrail eşiği, doğru ret ve yanlış ret oranları arasındaki dengeye göre test seti üzerinde seçilir; seçim raporda gösterilir.

---

## 10. Varsayılan teknoloji ve modeller

| Bileşen | Varsayılan | Alternatif (config) |
|---|---|---|
| Embedding | EmbeddingGemma 2 (yalnız metin, 768 boyut) | bge-m3, Gemini Embedding 2 (API) |
| Vektör DB | Qdrant (Docker) | — |
| BM25 | `bm25s` + Türkçe kök bulma (`snowballstemmer` turkish) | kök bulmasız |
| Reranker | bge-reranker-v2-m3 (sentence-transformers CrossEncoder) | Qwen3-Reranker-0.6B |
| LLM ve hakem | Qwen3 4B (Ollama, Q4, düşünme modu kapalı) | OpenAI uyumlu API (OpenRouter, Groq, Hugging Face Inference Providers; ücretli olabilir — uyarı verilir), Anthropic Claude (API, ücretli), transformers (Space) |
| API / UI | FastAPI + Jinja2 + HTMX (sohbet sayfası + Hakkında) | Gradio (Hugging Face ZeroGPU Space) |
| PDF | `pdfplumber` / `pypdf` | — |
| Config | Pydantic + YAML | — |
| Kalite | pytest, ruff, mypy, pre-commit (yerelde) | — |

Cihaz `auto | cpu | cuda` ile seçilir (`TUBITAK_RAG_DEVICE`); CPU'da reranker aday sayısı sınırlanır.

---

## 11. Açık konular ve riskler

| Konu | Etki | Plan |
|---|---|---|
| EmbeddingGemma 2 çok yeni (2026-10-06); araç desteği olgunlaşmamış olabilir | Embedding katmanı | Uygulamada doğrulanır; çalışmazsa bge-m3 varsayılan olur, karar günlüğe yazılır |
| Türkçe mevzuat dilinde model performansı bilinmiyor | Tüm metrikler | Zaten deneyin konusu; raporlarda açıkça yorumlanır |
| Qwen3 4B'nin CPU'da yavaşlığı | Kullanıcı deneyimi, eval süresi | Hafif mod (0.6B) ve API seçeneği; gecikme raporlanır |
| Yerel LLM-hakemin güvenilirliği | Cevap metrikleri | Raporlanan doğruluk, sadakat ve doğru ret Claude (Opus) ile tek tek değerlendirmeden gelir; otomatik hakem isteğe bağlıdır; kural tabanlı denetimler eklenir |
| 2 yönetmelik eksik | Teknopark/Ar-Ge merkezi soruları | Kullanıcı ekleyince artımlı ingest; test setinde bu sorular "eksik veri" olarak işaretlenir |
| Taranmış 3 PDF | Corpus kapsamı | Index dışı, ingest raporunda listelenir |
| TÜBİTAK dokümanlarının yeniden dağıtımı | Lisans | Repo yalnızca manifest + indirme betiği içerir |

---

## 12. İhtiyaç karşılama matrisi

Matris [`ihtiyaclar.xlsx`](ihtiyaclar.xlsx) dosyasındadır (Özet + İhtiyaçlar sayfaları). Her ihtiyacın durumu ve kanıtı (dosya, test adı veya ölçülen sonuç) `ihtiyaclar.yaml` içinde tutulur; son doğrulama 2026-10-21. Kaldırılan ve birleştirilen ihtiyaçlar Excel'in 'Değişiklik Geçmişi' sayfasındadır.

---

## 13. Karar günlüğü

| Tarih | Karar | Gerekçe |
|---|---|---|
| 2026-10-15 | Amaç: öğrenme + indirilebilir kullanılabilir araç | Kullanıcı tercihi |
| 2026-10-15 | Ayrı scraper yerine manifest + indirme betiği | Tekrarlanabilirlik, yeniden dağıtım güvenliği |
| 2026-10-15 | 5746/4691 metinleri GİB API'sinden | mevzuat.gov.tr erişilemiyor; GİB metinleri güncel |
| 2026-10-15 | Program sayfaları + seçilmiş çağrı/SSS/kılavuz corpus'a dahil | Mevzuatta "çağrıda belirtilir" diye geçen bilgileri tamamlar |
| 2026-10-15 | Embedding: EmbeddingGemma 2, karşılaştırma: bge-m3 | Yerel, hafif, 8K bağlam; bge-m3 güçlü çok dilli referans |
| 2026-10-15 | Vektör DB: Qdrant; BM25 ve RRF projenin kendi kodunda | Docker'a uygun, artımlı index kolay; öğretici şeffaflık |
| 2026-10-15 | Deney kapsamı 6 koşuya daraltıldı | Süre ve sadelik |
| 2026-10-15 | LLM: Qwen3 4B (hafif mod 0.6B), API opsiyonel | Basit bilgisayar hedefi |
| 2026-10-15 | Sıfırdan, ince kütüphanelerle; LlamaIndex/LangChain yok | Öğreticilik ve clean code kontrolü |
| 2026-10-15 | Arayüz: FastAPI + Jinja2 + HTMX; Streamlit opsiyonel host | "AI demosu" görünümünden kaçınma, tek servis |
| 2026-10-15 | Rapor ve günlük dili Türkçe; kod İngilizce | Hedef kitle ve kod standardı |
| 2026-10-16 | Arayüz sohbet sayfası + Hakkında'ya sadeleştirildi; Deneyler ve Belgeler sayfaları, filtreler ve ingest düğmesi kaldırıldı | Asıl kullanım soru sormak; deney sonuçları raporlarda ve REST API'de |
| 2026-10-16 | Docker için hazır indeks paketi, GPU imajı ve `HF_TOKEN` desteği | İlk açılışın yavaş kısmı ayrıştırma ve embedding; paketle ~4 dk |
| 2026-10-19 | İhtiyaçlar ürün ve kullanıcı bakışıyla yeniden yazıldı (v2.0, 94 ihtiyaç) | Süreç maddeleri ihtiyaç değildir; corpus ve test ihtiyaçları açıkça yazılmalı |
| 2026-10-21 | Streamlit sürümü kaldırıldı; Docker'ın hazır indeks paketi Hugging Face veri setinden iner | Barındırılan demo Gradio + ZeroGPU ile çalışıyor; Docker'da ilk açılış ingest'siz |
| 2026-10-21 | İhtiyaçlar 2.3: test seti, arama testi ve cevap testi ayrıntılı ihtiyaçlar olarak yazıldı; tekrar üretilebilirlik, CI ve Streamlit maddeleri çıkarıldı; plan dosyaları depodan kaldırıldı | Ürün sahibinin geri bildirimi: test ihtiyaçları somut olmalı; kullanılmayan süreç ve sürüm maddeleri ihtiyaç değildir |
