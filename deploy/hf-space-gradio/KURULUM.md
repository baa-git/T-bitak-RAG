# Hugging Face Space'e kurulum (Gradio + ZeroGPU, ücretsiz GPU)

Bu klasör, web arayüzüyle aynı sohbet ve Hakkında sayfalarını sunan Gradio uygulamasını
ücretsiz bir Hugging Face **ZeroGPU** Space'inde yayınlamak için gerekenleri içerir. Her şey
Space'in GPU'sunda çalışır: sorunun embedding'i, reranker ve cevabı yazan dil modeli (yerelde
Ollama'nın sunduğu **Qwen3 4B**'nin aynısı, `transformers` ile). Dış bir LLM API'si, API
anahtarı ya da ücretli bir hizmet yoktur.

| Dosya | Görevi |
|---|---|
| `README.md` | Space'in kendi README'si: YAML başlığı Space ayarlarıdır (Gradio SDK 6.30.0, Python 3.12, `app.py`, kurulumda indirilecek modeller). |
| `requirements.txt` | Space'in kurduğu paketler (torch 2.11.0, transformers, sentence-transformers, çekirdek bağımlılıklar). `gradio` ve `spaces`'i Hugging Face kurar. |
| `space-overlay.yaml` | Her deney yapılandırmasının üstüne yazılan katman: NumPy indeksi, CUDA, `llm.provider: transformers` (`Qwen/Qwen3-4B`). Space'te `configs/space-overlay.yaml` olur. |
| `prepare_space.py` | Depodan Space klasörünü (`build/hf-space-gradio`) kurar; depodaki `app_gradio.py` Space'te `app.py` olur. |

## Space nasıl çalışır?

- **ZeroGPU** paylaşımlı bir GPU'dur (yarım RTX Pro 6000 Blackwell, 48 GB). Uygulama açılırken
  bütün modeller belleğe yüklenir (EmbeddingGemma 2, bge-m3, bge-reranker-v2-m3, Qwen3 4B
  bfloat16: toplam ~14 GB); GPU yalnız bir soru cevaplanırken bağlanır (`@spaces.GPU`).
  Altı yapılandırma (v0 … v3b) aynı anda bellekte durur ve modelleri paylaşır: v0 … v3 aynı
  EmbeddingGemma 2'yi, v3b ayrıca bge-m3'ü kullanır; reranker ve dil modeli bir kez yüklenir.
- **İndeks önceden hesaplanır.** `rag export-space` koleksiyonları düz dosyalara yazar
  (`export/<koleksiyon>/vectors.npy` + `chunks.jsonl`), BM25 indekslerini ve Hakkında için
  ayrıştırma önbelleğini ekler (ölçülen: 3 koleksiyon, 7.017 + 4.193 + 7.017 parça,
  **~149 MB**). Uygulama bunları NumPy ile bellekte arar (kosinüs; Qdrant ile aynı skorlar).
- **Dil modeli** yerel kurulumdaki gibi çalışır: Qwen3 4B, düşünme modu kapalı
  (`enable_thinking=False`), sıcaklık 0 (açgözlü çözümleme). JSON şeması istemde tarif edilir;
  cevap en fazla 768 token olabilir (`LLM_MAX_NEW_TOKENS`).
- **GPU süresi:** her soru için ZeroGPU'dan 40 sn (reranker'lı yapılandırmalarda 45 sn) istenir.
  Ziyaretçilerin günlük GPU kotası vardır (giriş yapmadan ~2 dk, ücretsiz hesapla ~5 dk);
  kalan kota istenen süreden azsa soru reddedilebilir ve sayfa kotanın dolduğunu söyler. Bu
  yüzden istenen süre kısa tutuldu; sıradan bir soru gerçekte birkaç saniye sürer.
- **Varsayılan yapılandırma `v3`** (yapı bazlı parçalama, hibrit arama, reranker); GPU'da
  reranker bir saniyeden kısa sürer.

## 1. Space'i oluşturun (web formu)

<https://huggingface.co/new-space> formunu şöyle doldurun:

| Alan | Değer |
|---|---|
| Owner | `lolalol57` |
| Space name | `tubitak-mevzuat-rag` |
| Short description | `TÜBİTAK ve Ar-Ge mevzuatına kaynaklı soru-cevap` (en fazla 60 karakter) |
| License | MIT |
| Select the Space SDK | **Gradio** |
| Template | **Blank** |
| Space hardware | **ZeroGPU** (ücretsiz seçenek) |
| Visibility | **Public** |

**Create Space**'e basın. Boş bir depo açılır; dosyaları 3. adımda yükleyeceğiz. Formdaki
SDK ve donanım seçimi sonradan yüklenen `README.md` ile uyumludur (`sdk: gradio`).

> **ZeroGPU şartı:** ZeroGPU Space'i barındırabilmek için hesabın e-postası doğrulanmış ve
> hesap **en az 30 günlük** olmalıdır (ya da PRO üyelik). Seçenek formda görünmüyorsa ya da
> pasifse nedeni budur; o zaman Space'i "CPU basic" ile açıp sonra *Settings → Hardware*'den
> ZeroGPU'ya geçebilirsiniz (şartlar sağlandığında).

Komut satırından aynı Space (formun yerine):

```powershell
hf repos create lolalol57/tubitak-mevzuat-rag --type space --space-sdk gradio --flavor zero-a10g --public --exist-ok
```

Space'e değişken ya da sır eklemeniz **gerekmez**: `SPACE_ID`'yi Hugging Face tanımlar;
uygulama onu görünce ZeroGPU ayarlarını (katman dosyası, CUDA, GPU kilidi kapalı, varsayılan
`v3`) kendiliğinden açar.

## 2. İndeksi ve Space klasörünü hazırlayın

Depo kökünden, PowerShell'de:

```powershell
# a) Space indeksi (yerel ingest önbelleklerinden; Qdrant gerekmez)
.\.venv\Scripts\rag.exe export-space --source cache              # → release\space-index (~149 MB)

# b) Space klasörü (app.py, src/, configs/, data/manifests/, artifacts/ = indeks)
.\.venv\Scripts\python.exe deploy\hf-space-gradio\prepare_space.py   # → build\hf-space-gradio (~150 MB)
```

`export-space` her koleksiyonun parça sayısını yazar; hazır indeks paketindeki sayılarla aynı
olmalıdır (structural_ctx: 7.017, fixed: 4.193, bge-m3: 7.017). Indeks zaten
`release\space-index`'te varsa (a) adımını atlayabilirsiniz. `build\` git'e girmez.

## 3. Giriş yapın ve yükleyin

```powershell
.\.venv\Scripts\python.exe -m pip install -U huggingface_hub     # hf komutu
hf auth login          # Settings → Access Tokens'tan "write" izinli bir token yapıştırın
hf upload lolalol57/tubitak-mevzuat-rag build\hf-space-gradio . --repo-type space --exclude "**/__pycache__/**"
```

Token'ı yalnız `hf auth login` sorduğunda yapıştırın; hiçbir dosyaya yazmayın. `--repo-type
space` zorunludur (yoksa model deposu sanılır). Büyük dosyaları (vektörler, BM25) `hf upload`
kendisi uygun biçimde (Xet) yükler. Kod değişince 2b ve 3. adımları tekrarlayın; değişmeyen
dosyalar yeniden yüklenmez.

Durumu ve günlüğü komut satırından izlemek için:

```powershell
hf spaces info lolalol57/tubitak-mevzuat-rag --expand runtime
hf spaces logs lolalol57/tubitak-mevzuat-rag --tail 200
```

## İlk açılışta ne olur?

1. **Build** (birkaç dakika + model indirme): `requirements.txt` kurulur (torch 2.11 CUDA
   tekerleği büyüktür), `preload_from_hub` dört modeli imaja indirir (Qwen3 4B ~8 GB,
   bge-m3 ~2,3 GB, bge-reranker-v2-m3 ~2,3 GB, EmbeddingGemma 2). Toplam 10–20 dakika
   sürebilir; `startup_duration_timeout: 1h` bu yüzden uzun tutuldu.
2. **Başlangıç:** `app.py` önce `spaces`'i, sonra paketi yükler; bütün yapılandırmaların
   pipeline'ları kurulur ve modeller CUDA'ya taşınır (ZeroGPU bunu kaydeder, GPU'ya ilk soruda
   aktarır). Günlükte `gradio: yüklenen yapılandırmalar: v0, v0r, v1, v2, v3, v3b` satırı görünür.
3. **İlk soru** modellerin GPU'ya aktarılması nedeniyle birkaç saniye daha uzun sürer;
   sonrakiler hızlıdır.

## Yerelde deneme

GPU'suz ya da küçük GPU'lu bir makinede uygulamayı gösterim verisiyle açmak için:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gradio]"
$env:TUBITAK_RAG_BACKEND = "fake"
.\.venv\Scripts\python.exe app_gradio.py          # http://localhost:7860
```

Gerçek sistemle, Space kipinde (Qdrant ve Docker gerekmez; ~10 GB GPU'ya sığması için küçük bir
Qwen3 ile):

```powershell
$env:TUBITAK_RAG_BACKEND = "pipeline"
$env:TUBITAK_RAG_OVERLAY = "deploy/hf-space-gradio/space-overlay.yaml"
$env:TUBITAK_RAG_EXPORT_DIR = "release/space-index/export"   # BM25 ve önbellekler artifacts/'tan
$env:LLM_MODEL = "Qwen/Qwen3-0.6B"                            # Space'te Qwen/Qwen3-4B
.\.venv\Scripts\python.exe app_gradio.py
```

Yerelde her soru `artifacts\.gpu.lock` GPU kilidini alır (aynı anda tek GPU işi kuralı);
kapatmak için `$env:TUBITAK_RAG_GPU_LOCK = "0"`. 10 GB'lık bir kartta altı yapılandırmanın
hepsini açılışta yüklemek belleği doldurur; `$env:TUBITAK_RAG_PRELOAD = "0"` ile modeller ilk
kullanıldıkları soruda yüklenir. (Windows'taki PyTorch'ta flash attention yoktur; Qwen3'ün
gruplu dikkati bu yüzden "math" çekirdeğine düşer ve 5 bin token'lık bir istem birkaç GB ek
bellek ister. Space'in 48 GB'lık GPU'sunda bu sorun olmaz.)

## Sorun giderme

- **Build "torch" ya da "CUDA" hatasıyla durdu:** ZeroGPU yalnız belirli torch sürümlerini
  kabul eder (2.8.0, 2.9.1, 2.10.0, 2.11.0). `requirements.txt`'teki `torch==2.11.0` ve
  `torchvision==0.26.0` çiftini birlikte değiştirin (ör. 2.10.0 + 0.25.0); `gradio` ve
  `spaces` satırı eklemeyin.
- **"CUDA has been initialized before importing the `spaces` package":** `app.py`'de
  `import spaces` en üstte olmalıdır; dosyayı elle değiştirdiyseniz `prepare_space.py`'yi
  yeniden çalıştırın.
- **"You have exceeded your GPU quota" / "ZeroGPU quota":** ziyaretçinin günlük kotası doldu.
  Giriş yapmak kotayı artırır; süre ertesi gün yenilenir. Soru başına istenen süreyi
  düşürmek için Space'e `TUBITAK_RAG_GPU_DURATION` değişkeni (saniye, varsayılan 40)
  ekleyebilirsiniz; çok kısa olursa uzun cevaplar yarıda kesilir.
- **"GPU task aborted" / süre aşımı:** cevap ayrılan süreye sığmadı. `TUBITAK_RAG_GPU_DURATION`
  değerini artırın ya da `LLM_MAX_NEW_TOKENS`'ı düşürün.
- **"Dil modelinden şu anda cevap alınamadı":** model hata verdi ya da cevap 768 token
  sınırını aştı; ayrıntı Space günlüğündedir (*Logs*).
- **Yapılandırma "(indeks yok)":** yüklenen indeks o koleksiyonu içermiyor; `rag export-space`'i
  `--collection` olmadan yeniden çalıştırıp 2b ve 3. adımları tekrarlayın.
- **Build `sdk_version` hatası:** `README.md`'deki `sdk_version` değerini Hugging Face'in
  sunduğu güncel bir Gradio 6 sürümüyle değiştirin (uygulama 6.30.0 ile denendi).
- **"Uygulama başlatılamadı" sayfası:** yapılandırma ya da indeks okunamadı; mesaj nedenini
  yazar, ayrıntı *Logs*'tadır.
