# syntax=docker/dockerfile:1
#
# TÜBİTAK Mevzuat RAG — uygulama imajı (web arayüzü + REST API + `rag` komut satırı).
#
# İki aşama: "builder" bağımlılıkları bir sanal ortama (/opt/venv) kurar; "runtime" yalnızca
# o ortamı ve çalışma için gereken dosyaları alır (derleme araçları ve pip önbelleği imaja girmez).
#
# Varsayılan PyTorch CPU tekerleğidir (imaj küçük kalır). GPU imajı için:
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
# (TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ile CUDA tekerleği kurulur.)

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH

RUN python -m venv /opt/venv

# 1) PyTorch ayrı ve ilk katmanda: en büyük ve en seyrek değişen bağımlılık.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.6" "torchvision>=0.21"

# 2) Paketin bağımlılıkları: yalnız pyproject.toml değişince yeniden kurulur. Kaynak kod yerine
#    boş bir paket iskeleti kullanılır ki kod değişikliği bu katmanı geçersiz kılmasın.
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/tubitak_rag && touch src/tubitak_rag/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[ml,web]"

# 3) Asıl kaynak kod (bağımlılıklar zaten kurulu: --no-deps). setuptools'un build/ klasörü
#    silinir: yoksa iskeletin boş __init__.py'si, ondan eski tarihli gerçek dosyanın yerine kalır.
COPY src ./src
RUN rm -rf build src/*.egg-info     && pip install --no-deps --no-cache-dir --force-reinstall . \
    && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="tubitak-rag" \
      org.opencontainers.image.description="TÜBİTAK mevzuatı üzerinde yerel, kaynaklı soru-cevap (RAG)" \
      org.opencontainers.image.licenses="MIT"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/hf_cache \
    TUBITAK_RAG_BACKEND=pipeline \
    TUBITAK_RAG_CONFIG=configs/default.yaml \
    QDRANT_URL=http://qdrant:6333 \
    OLLAMA_URL=http://ollama:11434

# Root olmayan kullanıcı (uid 1000). Volume bağlama noktaları imajda bu kullanıcıya ait
# oluşturulur: yeni bir named volume ilk bağlandığında bu sahipliği devralır.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY configs ./configs
COPY data/manifests ./data/manifests
COPY eval ./eval
RUN mkdir -p /app/data/raw /app/artifacts /hf_cache \
    && chown -R app:app /app/data/raw /app/artifacts /hf_cache

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["rag"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
