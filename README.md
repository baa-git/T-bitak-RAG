# TÜBİTAK Mevzuat RAG

TÜBİTAK destek programları ve Ar-Ge mevzuatı hakkında Türkçe soru soran, cevabını belge ve
madde numarasıyla veren, bilmediğinde "bilgi bulunamadı" diyen bir RAG (Retrieval-Augmented
Generation) uygulaması. Her şey kendi bilgisayarınızda çalışır: arama indeksi hazır gelir,
dil modeli Ollama üzerinde `qwen3:4b`'dir, API anahtarı gerekmez.

Proje öğrenme amacıyla geliştirildi. RAG'in üç adımı olan **Retrieval** (ilgili parçaları
bulma), **Augmentation** (bulunan parçaları dil modelinin girdisine ekleme) ve **Generation**
(cevabı bu parçalara dayanarak yazma) için farklı teknikler denendi: altı yapılandırma
(v0 … v3b) aynı 104 soruluk test setiyle ölçüldü ve her tekniğin neyi değiştirdiği raporlandı.

**Canlı demo:** [huggingface.co/spaces/lolalol57/tubitak-mevzuat-rag](https://huggingface.co/spaces/lolalol57/tubitak-mevzuat-rag)
(kurulum gerekmez; ilk soru modeller GPU'ya alınırken biraz bekletebilir).

> **RAG'i bu projeyle öğren:** [`docs/RAG-GELISTIRME-REHBERI.md`](docs/RAG-GELISTIRME-REHBERI.md)
> parçalama, dense/BM25/hibrit arama, yeniden sıralama, güvenlik önlemleri ve arama ile cevabın
> ayrı ayrı nasıl test edildiğini bu projenin gerçek sonuçlarıyla anlatır.

Diğer belgeler: [ihtiyaçlar ve tasarım](docs/ihtiyaclar.md) ([Excel](docs/ihtiyaclar.xlsx)) ·
[mimari](docs/mimari.md) · [deney özeti](docs/sample-reports/ozet.md) · [test seti](eval/README.md) ·
[testler ve sonuçlar (Excel)](docs/testler.xlsx).

## Hızlı başlangıç (Docker)

> **Gerekenler**
>
> - Docker Desktop (Windows/macOS) ya da Docker Engine + Compose v2 (Linux)
> - ~20 GB boş disk (GPU imajıyla ~30 GB)
> - Docker'a ayrılmış en az 8 GB RAM
> - İsteğe bağlı NVIDIA GPU: sürücü + Windows'ta Docker Desktop WSL2 GPU desteği, Linux'ta
>   NVIDIA Container Toolkit. GPU yoksa CPU ile çalışır, yalnız cevaplar daha yavaştır.

**1. Depoyu indirin**

```bash
git clone https://github.com/baa-git/T-bitak-RAG.git
cd T-bitak-RAG
```

**2. (İsteğe bağlı) Ayar dosyasını oluşturun.** Varsayılanlarla çalışır; port, HF token'ı ya
da dil modeli sağlayıcısı değiştirmek isterseniz:

```bash
cp .env.example .env              # PowerShell: Copy-Item .env.example .env
```

**3. Başlatın**

```bash
docker compose up -d --build
```

NVIDIA GPU ile:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

**4. İlk açılışı izleyin** (Türkçe günlük; `init` bitince `app` başlar)

```bash
docker compose logs -f init
```

**5. Açın:** <http://localhost:8000> · API belgeleri: <http://localhost:8000/docs>

**İlk açılışta ne olur?** İmajlar derlenir/çekilir. Tek seferlik `init` servisi Ollama'ya
`qwen3:4b`'yi (2,5 GB), embedding modeli ile yeniden sıralayıcıyı Hugging Face önbelleğine
indirir; ardından **hazır indeks paketini** Hugging Face veri setinden
([`lolalol57/tubitak-mevzuat-rag-index`](https://huggingface.co/datasets/lolalol57/tubitak-mevzuat-rag-index),
~244 MB) indirip SHA-256 ile doğrular ve Qdrant'a yükler. Belge indirme, PDF ayrıştırma ve
embedding (ingest) yapılmaz. RTX 3080'li bir makinede imaj derleme hariç ilk açılış ~4 dk
sürdü (CPU imajını derlemek ~2,5 dk, `ollama/ollama` imajını çekmek ~2 dk); sonraki
`docker compose up` ~15 s. Model bellekteyken bir soru GPU'da ~2,3 s, CPU'da 12–57 s.

## Günlük kullanım

```bash
docker compose stop                       # durdur (her şey volume'larda kalır)
docker compose start                      # yeniden başlat
docker compose ps                         # durum ve sağlık
docker compose logs -f app                # uygulama günlüğü
docker compose exec app rag ask "1507 programında destek oranı nedir?"

git pull                                  # güncelle: önce kodu çekin,
docker compose up -d --build              # sonra yeniden derleyip başlatın
docker compose down                       # kaldır; modeller ve indeks volume'larda korunur
docker compose down -v                    # her şeyi sil (modeller yeniden iner)
```

GPU ekiyle kurduysanız `up` komutlarında iki `-f` dosyasını yine verin.

### Yapılandırma seçmek (v0 … v3b)

Sohbet sayfasındaki seçiciden yapılandırmayı değiştirin; seçilen yapılandırmanın parçalama,
embedding, arama, yeniden sıralayıcı ve dil modeli bilgisi altında yazar. Varsayılan `v3`'tür
(yapı bazlı parçalama, hibrit arama, yeniden sıralayıcı). Komut satırından:

```bash
docker compose exec app rag ask -c configs/experiments/v1.yaml "1507 programında destek oranı nedir?"
```

| | Parçalama | Arama | Yeniden sıralayıcı | Embedding |
|---|---|---|---|---|
| v0 | sabit pencere (512/64) | dense | – | EmbeddingGemma 2 |
| v0r | sabit pencere | dense | bge-reranker-v2-m3 | EmbeddingGemma 2 |
| v1 | yapı bazlı + bağlam başlığı | dense | – | EmbeddingGemma 2 |
| v2 | yapı bazlı + bağlam başlığı | hibrit (dense + BM25, RRF) | – | EmbeddingGemma 2 |
| v3 | yapı bazlı + bağlam başlığı | hibrit | bge-reranker-v2-m3 | EmbeddingGemma 2 |
| v3b | v3 ile aynı | hibrit | bge-reranker-v2-m3 | bge-m3 |

v3b ilk kullanıldığında bge-m3 modelini indirir. Ölçülen farklar:
[deney özeti](docs/sample-reports/ozet.md).

### Ollama yerine OpenAI uyumlu bir API kullanmak

Cevap modeli yapılandırmayla değişir: Ollama (varsayılan, yerel) ya da OpenAI uyumlu herhangi
bir `/chat/completions` servisi (OpenRouter, Groq, Hugging Face Inference Providers…).

> **Dikkat:** API seçildiğinde sorular ve getirilen belge parçaları o servise gider ve
> ücretlendirilebilir. Anahtarı yalnız `.env` dosyasına yazın; `.env` git'e ve imaja girmez.

1. `configs/default.yaml` içinde `llm` bölümünü değiştirin (bütün yapılandırmalar bu dosyayı
   temel alır):

   ```yaml
   llm:
     provider: openai_compat
     model: <sağlayıcının model kimliği>
   ```

2. `.env` dosyasına adresi ve anahtarı yazın:

   ```bash
   LLM_BASE_URL=https://openrouter.ai/api/v1     # Groq: https://api.groq.com/openai/v1
                                                 # Hugging Face: https://router.huggingface.co/v1
   LLM_API_KEY=...
   ```

3. Yeniden derleyip başlatın (yapılandırma dosyaları imaja kopyalanır):
   `docker compose up -d --build`. Yerel kurulumda `rag serve` yeterlidir.

Yerel kurulumda aynı alana `anthropic` (Claude API, `ANTHROPIC_API_KEY`, `.[api]` eki) da
yazılabilir; Docker imajında bu paket kurulu değildir. OpenAI uyumlu
bağdaştırıcı birim testlerinde sahte HTTP ile ve Ollama'nın `/v1` uç noktasıyla aynı biçimde
doğrulandı; ücretli bir sağlayıcıyla denenmedi.

<details>
<summary><b>Docker ayrıntıları</b>: hazır indeks paketi, HF token'ı, CPU/GPU farkı, yönetim uç noktaları</summary>

**Hazır indeks paketi.** `init` önce `./release/manifest.json`'a bakar, yoksa
`INDEX_BUNDLE_URL`'den indirir (varsayılan yukarıdaki veri seti). Yalnız eksik koleksiyonlar
yüklenir, dolu bir koleksiyonun üzerine yazılmaz. `TUBITAK_RAG_USE_BUNDLE=0` paketi yok sayar
ve belgeleri indirip baştan indeksler (`rag fetch` + `rag ingest`; GPU'da ~20–30 dk, CPU'da
saatler). Paketle kurulan yığında ham belgeler inmez; sonradan `rag ingest` yapacaksanız önce
`docker compose exec app rag fetch` çalıştırın (ya da `TUBITAK_RAG_BUNDLE_FETCH=1`).

Paketi yenilemek (depo sahibi; Qdrant localhost:6333'te çalışırken):

```bash
rag export-index --out release    # release/ git'e ve imaja girmez
python -c "from huggingface_hub import upload_folder; upload_folder(folder_path='release', repo_id='lolalol57/tubitak-mevzuat-rag-index', repo_type='dataset', allow_patterns=['manifest.json', '*.snapshot', '*.tar.gz'])"
```

**Hugging Face erişimi (`HF_TOKEN`).** Varsayılan modeller anonim iner. Erişimi kısıtlı bir
model seçilirse `init` "erişim reddedildi" diye durur: model sayfasında lisansı kabul edin,
*Settings → Access Tokens*'tan "read" token'ı oluşturup `.env`'e `HF_TOKEN=hf_...` yazın.
Token yalnız tanımlıysa container'a geçer; imaja ve günlüğe girmez.

**CPU ve GPU farkı.** Yeniden sıralayıcı CPU'da 30 aday için ~19 s sürer; bu yüzden CPU'da yalnız
ilk `RERANKER_CPU_CANDIDATES` aday (varsayılan 6, ~4,7 s) yeniden sıralanır (`0` = kapalı,
`none` = sınırsız). `TUBITAK_RAG_DEVICE=auto|cpu|cuda` cihazı seçer; GPU ekinde varsayılan
`cuda`'dır ve GPU görünmezse hata verir. CPU'da Qwen3 4B yavaş olduğu için Docker'da
`LLM_TIMEOUT_S` 300 s'dir.

**Yönetim uç noktaları.** `POST /api/ingest` ve `POST /api/runs` varsayılan olarak kapalıdır
(403). Açmak için `TUBITAK_RAG_ADMIN_TOKEN` tanımlayın ve isteğe
`Authorization: Bearer <anahtar>` ekleyin. Soru sorma her zaman açıktır.

Bütün ayarlar ve açıklamaları: [`.env.example`](.env.example).

</details>

## Sorun giderme

- **Port 8000 dolu:** `.env`'e `APP_PORT=8010` yazıp `docker compose up -d` çalıştırın.
- **`init` hata verdi:** `docker compose logs init`. İnternet yoksa model ve paket indirme
  başarısız olur; bağlantı gelince `docker compose up -d` yeniden dener. "Hazır indeks paketi
  yüklenemedi" görürseniz `INDEX_BUNDLE_URL`'i kontrol edin.
- **GPU ekinde "could not select device driver nvidia":** NVIDIA sürücüsünü ve Docker
  Desktop → Settings → Resources → WSL integration ayarını kontrol edin;
  `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi` çalışmalıdır.
- **Host'ta zaten Qdrant/Ollama var:** çakışma olmaz; yığının Qdrant ve Ollama'sı host'a port
  açmaz, proje ve volume adları (`tubitak-rag_…`) ayrıdır.
- **Soru 503 "BM25 indeksi bulunamadı":** indeks kurulmamış; `docker compose exec app rag ingest`.
- **Disk doldu:** `docker system df`, ardından `docker image prune`.

## Geliştirici kurulumu

Python 3.13 gerekir. Yerel çalıştırmak için Qdrant (6333) ve Ollama (11434, `ollama pull qwen3:4b`)
ayrıca çalışıyor olmalıdır; birim testleri için gerekmezler.

```bash
python -m venv .venv
source .venv/bin/activate              # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# NVIDIA GPU varsa önce CUDA'lı torch (yoksa atlayın; CPU sürümü kurulur):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[ml,web,api,dev]"     # Space için ayrıca: ".[gradio]"
cp .env.example .env                   # PowerShell: Copy-Item .env.example .env
rag --help                             # fetch, ingest, run, report, ask, serve, export-index, export-space
```

Yapılandırma `configs/default.yaml`'dadır; deneyler `configs/experiments/*.yaml` ile yalnız
farkları yazar. Değerlerde `${DEGISKEN:varsayilan}` ortam değişkeninden okunur.

```bash
rag fetch                                        # kaynak belgeleri manifestlerden indir (SHA-256)
rag ingest -c configs/experiments/v3.yaml        # ayrıştır, parçala, indeksle (artımlı)
rag ask "1507 programında destek oranı nedir?"
rag serve                                        # web arayüzü: http://localhost:8000
rag run configs/experiments/v3.yaml              # test setiyle deney koşusu + rapor
rag report --all                                 # raporları yeniden üret

python -m pytest tests/unit -q                   # servis, model veya anahtar gerekmez
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

Altı yapılandırma üç indeks kullanır (sabit pencere + EmbeddingGemma 2, yapı bazlı +
EmbeddingGemma 2, yapı bazlı + bge-m3); her biri için bir kez `rag ingest` yeterlidir (örn.
v0, v3 ve v3b yapılandırmalarıyla).

<details>
<summary><b>Geliştirme ortamında sık sorunlar</b></summary>

- **`OMP: Error #15`** (Anaconda'nın libiomp kütüphanesiyle çakışma): geçici olarak
  `KMP_DUPLICATE_LIB_OK=TRUE` ortam değişkenini tanımlayın.
- **Windows konsolunda bozuk Türkçe karakterler:** `$env:PYTHONUTF8="1"` (kalıcı:
  `setx PYTHONUTF8 1`); gerekirse `chcp 65001`.
- **pre-commit:** `pre-commit install` ile kurulur. Yerel mypy kancası
  `.venv\Scripts\python.exe` yolunu kullanır; Linux/macOS'ta `.pre-commit-config.yaml`'daki
  `entry` satırını `.venv/bin/python -m mypy src` yapın. Proje yolu Türkçe karakter içeriyorsa
  `.git/hooks/pre-commit` içindeki `INSTALL_PYTHON=` satırı bozulabilir; commit'i venv
  etkinleştirilmiş bir kabukta yapın ya da satırı `INSTALL_PYTHON='.venv/Scripts/python.exe'`
  olarak düzeltin.
- **Boş ortam değişkenleri:** `.env`'deki boş `HF_HOME=` satırları yok sayılır; kabukta boş bir
  `HF_HOME` tanımlıysa modeller çalışma klasörüne inebilir, tanımı kaldırın.
- **Ayrıştırıcıyı değiştirdiyseniz:** `tests/unit/ingestion/test_parsed_cache.py` kırmızıya
  döner. Çıktı değiştiyse `ingestion/parsed_cache.py`'deki `PARSER_VERSION`'ı artırın, yalnız
  yeniden düzenleme yaptıysanız testteki kayıtlı özeti güncelleyin.

</details>

## Hugging Face Space (Gradio + ZeroGPU)

[Canlı demo](https://huggingface.co/spaces/lolalol57/tubitak-mevzuat-rag), ücretsiz bir
ZeroGPU Space'inde çalışan Gradio uygulamasıdır: web arayüzüyle aynı sohbet, yapılandırma
seçici ve numaralı kaynaklar. Qdrant ve Ollama yoktur: indeks önceden hesaplanıp NumPy ile
aranır (`rag export-space`), Qwen3 4B, embedding modelleri ve yeniden sıralayıcı Space'in
GPU'sunda çalışır; API anahtarı gerekmez. Kod `src/tubitak_rag/interfaces/gradio_app/`, giriş
dosyası `app_gradio.py`. Kurulum: [`deploy/hf-space-gradio/KURULUM.md`](deploy/hf-space-gradio/KURULUM.md).

## Lisans

MIT — bkz. [`LICENSE`](LICENSE).
