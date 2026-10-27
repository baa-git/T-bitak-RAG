"""First-start preparation of the Docker stack (the one-shot ``init`` service).

``python -m tubitak_rag.interfaces.bootstrap`` runs these steps and exits; ``app`` starts
only after it succeeded (compose ``service_completed_successfully``). Every step is
idempotent, so the second ``docker compose up`` finishes in seconds:

1. wait until Qdrant (``/healthz``) and Ollama (``/api/tags``) answer;
2. pull the Ollama models the config needs (``llm.model``, an Ollama judge or embedder) when
   ``/api/tags`` does not list them — NFR-DOCKER2;
3. download the HuggingFace models (embedder, reranker) when they are not in the HF cache
   (a denied download - gated model, missing ``HF_TOKEN`` - fails with a Turkish hint);
4. prebuilt index bundle (``index_bundle.py``): when ``./release`` (``/app/release``) holds a
   ``manifest.json`` or ``INDEX_BUNDLE_URL`` is set, restore the Qdrant snapshots, BM25
   directories and metadata it carries that the stack does not have yet - no fetch, no
   parsing, no embedding. A broken bundle fails the service (unless the index is already
   there); ``TUBITAK_RAG_BUNDLE_FETCH=1`` also downloads the raw documents afterwards;
5. when the configured collection is still empty or missing (or its BM25 index is missing):
   ``rag fetch`` then ``rag ingest``. A partly failed fetch is only a warning (ingest skips
   missing files); a failed ingest fails the service.

Environment: ``TUBITAK_RAG_CONFIG`` (default ``configs/default.yaml``),
``TUBITAK_RAG_AUTO_INGEST=0`` skips step 5, ``TUBITAK_RAG_WARMUP=0`` skips step 3,
``TUBITAK_RAG_USE_BUNDLE=0`` skips step 4, ``TUBITAK_RAG_BUNDLE_DIR`` (default ``release``),
``INDEX_BUNDLE_URL`` (compose default: the project's Hugging Face dataset),
``TUBITAK_RAG_WAIT_S`` bounds step 1 (default 300 s). All log lines are Turkish.
"""

import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import httpx

from tubitak_rag.chunking.cache import chunker_key
from tubitak_rag.config import AppConfig, ConfigError, load_config
from tubitak_rag.indexing.naming import bm25_dir, collection_name
from tubitak_rag.interfaces.index_bundle import (
    BundleError,
    locate_bundle,
    pending_work,
    restore_bundle,
)
from tubitak_rag.interfaces.web.jobs import cli_command

logger = logging.getLogger("tubitak_rag.hazirlik")

CONFIG_ENV = "TUBITAK_RAG_CONFIG"
AUTO_INGEST_ENV = "TUBITAK_RAG_AUTO_INGEST"
WARMUP_ENV = "TUBITAK_RAG_WARMUP"
WAIT_ENV = "TUBITAK_RAG_WAIT_S"
USE_BUNDLE_ENV = "TUBITAK_RAG_USE_BUNDLE"
BUNDLE_DIR_ENV = "TUBITAK_RAG_BUNDLE_DIR"
BUNDLE_URL_ENV = "INDEX_BUNDLE_URL"
BUNDLE_FETCH_ENV = "TUBITAK_RAG_BUNDLE_FETCH"
HF_TOKEN_ENV = "HF_TOKEN"
DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_BUNDLE_DIR = Path("release")
BUNDLE_DOWNLOAD_DIR = "release-download"  # under artifacts/, removed after a restore
_HF_DENIED_TYPES = {"GatedRepoError", "RepositoryNotFoundError"}
_HF_DENIED_TEXT = ("401", "403", "gated", "unauthorized", "forbidden", "restricted")
_FALSE = {"0", "false", "no", "hayir", "hayır", "off"}
_PROGRESS_MIN_BYTES = 10_000_000  # smaller layers (licence, params) get one line, no percent

Runner = Callable[[Sequence[str]], int]
Sleeper = Callable[[float], None]
Clock = Callable[[], float]


class Restorer(Protocol):
    """Step 4: restore a prebuilt index bundle; True when something was restored."""

    def __call__(
        self, cfg: AppConfig, client: httpx.Client, environ: Mapping[str, str], *, ready: bool
    ) -> bool: ...


class BootstrapError(RuntimeError):
    """A preparation step failed (message in Turkish)."""


def flag(environ: Mapping[str, str], name: str, default: bool = True) -> bool:
    """Read a yes/no environment switch (``0``/``false``/``hayır`` = no)."""
    raw = environ.get(name, "").strip().lower()
    return default if not raw else raw not in _FALSE


def wait_for(
    name: str,
    url: str,
    client: httpx.Client,
    *,
    timeout_s: float = 300.0,
    interval_s: float = 2.0,
    sleep: Sleeper = time.sleep,
    clock: Clock = time.monotonic,
) -> None:
    """Poll ``url`` until it answers 2xx.

    Raises:
        BootstrapError: when ``timeout_s`` passes without a successful answer.
    """
    deadline = clock() + timeout_s
    logger.info("%s bekleniyor: %s", name, url)
    while True:
        try:
            if client.get(url, timeout=5.0).is_success:
                logger.info("%s hazır.", name)
                return
        except httpx.HTTPError:
            pass
        if clock() >= deadline:
            raise BootstrapError(f"{name} {timeout_s:.0f} s içinde yanıt vermedi: {url}")
        sleep(interval_s)


def _with_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def required_ollama_models(cfg: AppConfig) -> list[str]:
    """Ollama models the config uses: the LLM, plus an Ollama judge or embedder if chosen."""
    models = []
    if cfg.llm.provider == "ollama":
        models.append(cfg.llm.model)
    if cfg.eval.judge.provider == "ollama":
        models.append(cfg.eval.judge.model)
    if cfg.embedder.provider == "ollama":
        models.append(cfg.embedder.model)
    return list(dict.fromkeys(_with_tag(m) for m in models))


def installed_models(client: httpx.Client, ollama_url: str) -> set[str]:
    """Names listed by Ollama's ``/api/tags``."""
    response = client.get(f"{ollama_url}/api/tags", timeout=10.0)
    response.raise_for_status()
    return {_with_tag(m["name"]) for m in response.json().get("models", [])}


def pull_model(client: httpx.Client, ollama_url: str, model: str) -> None:
    """``POST /api/pull`` (streamed); logs each status and every 10 % of each large layer.

    Raises:
        BootstrapError: when Ollama reports an error or the stream ends without success.
    """
    logger.info("Ollama modeli indiriliyor: %s (ilk açılışta birkaç dakika sürebilir)", model)
    last_status, last_decile, success = "", -1, False
    with client.stream(
        "POST", f"{ollama_url}/api/pull", json={"model": model, "stream": True}, timeout=None
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            if "error" in event:
                raise BootstrapError(f"Ollama {model} indiremedi: {event['error']}")
            status = str(event.get("status", ""))
            total, completed = event.get("total"), event.get("completed")
            large = bool(total) and total >= _PROGRESS_MIN_BYTES
            if status != last_status:
                last_status, last_decile = status, -1
                if not large:
                    logger.info("  %s: %s", model, status)
            if large and completed is not None:
                decile = int(completed * 10 / total)
                if decile != last_decile:
                    last_decile = decile
                    logger.info(
                        "  %s: %s %%%d (%.0f / %.0f MB)",
                        model,
                        status.split(" ")[0],
                        decile * 10,
                        completed / 1e6,
                        total / 1e6,
                    )
            success = success or status == "success"
    if not success:
        raise BootstrapError(f"Ollama {model} indirmesi tamamlanmadan bitti.")
    logger.info("Ollama modeli hazır: %s", model)


def ensure_ollama_models(client: httpx.Client, ollama_url: str, models: Iterable[str]) -> list[str]:
    """Pull every model in ``models`` that Ollama does not have yet; return the pulled ones."""
    present = installed_models(client, ollama_url)
    pulled = []
    for model in models:
        if model in present:
            logger.info("Ollama modeli zaten var: %s", model)
            continue
        pull_model(client, ollama_url, model)
        pulled.append(model)
    return pulled


def collection_points(client: httpx.Client, qdrant_url: str, collection: str) -> int:
    """Points in ``collection`` (0 when it does not exist), via Qdrant's REST API."""
    response = client.get(f"{qdrant_url}/collections/{collection}", timeout=10.0)
    if response.status_code == 404:
        return 0
    response.raise_for_status()
    return int(response.json()["result"].get("points_count") or 0)


def needs_ingest(cfg: AppConfig, points: int) -> bool:
    """Whether the corpus must be indexed: no points, or the BM25 index is missing."""
    if points == 0:
        return True
    if cfg.bm25.enabled:
        return not bm25_dir(cfg.paths.artifacts_dir, chunker_key(cfg.chunker)).is_dir()
    return False


def run_subprocess(argv: Sequence[str]) -> int:
    """Run ``argv`` with inherited stdout/stderr (the CLI's Turkish output reaches the logs)."""
    return subprocess.run(list(argv), check=False).returncode


def fetch_and_ingest(config_path: Path, run: Runner = run_subprocess) -> None:
    """``rag fetch`` (problems are warnings) then ``rag ingest`` (must succeed).

    Raises:
        BootstrapError: when ingest exits with a non-zero code.
    """
    logger.info("Belgeler indiriliyor (rag fetch; zaten olan dosyalar atlanır)…")
    if run(cli_command("fetch", "-c", str(config_path))) != 0:
        logger.warning(
            "rag fetch bazı belgeleri indiremedi ya da özet uyuşmazlığı buldu; "
            "eldeki belgelerle devam ediliyor (ayrıntı: artifacts/fetch_report.json)."
        )
    logger.info(
        "İndeksleme başlıyor (rag ingest). CPU'da önbelleksiz ilk ingest uzun sürer "
        "(~8 bin parçanın embedding'i; saatlerle ölçülebilir), GPU'da birkaç dakika."
    )
    started = time.monotonic()
    if run(cli_command("ingest", "-c", str(config_path))) != 0:
        raise BootstrapError("rag ingest başarısız oldu; yukarıdaki çıktıya bakın.")
    logger.info("İndeksleme bitti (%.0f s).", time.monotonic() - started)


def hf_access_error(exc: BaseException, model: str, *, token_set: bool) -> str | None:
    """Turkish explanation when ``exc`` is a denied HuggingFace download, else None.

    huggingface_hub raises ``GatedRepoError`` (401/403) for a gated model without access;
    transformers wraps it in an ``OSError``. Both are recognised along the exception chain
    by class name and message, so huggingface_hub is not imported here.
    """
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    names = {type(e).__name__ for e in chain}
    text = " ".join(str(e) for e in chain).lower()
    if not (names & _HF_DENIED_TYPES or any(word in text for word in _HF_DENIED_TEXT)):
        return None
    advice = (
        "HF_TOKEN tanımlı ama bu modele erişimi yok: lisansı token'ın sahibi olan hesapla "
        "kabul ettiğinizden ve token'ın 'read' yetkisi olduğundan emin olun."
        if token_set
        else "Model erişimi kısıtlı (gated) olabilir: huggingface.co'da oturum açıp model "
        "sayfasında lisansı kabul edin, Settings > Access Tokens'tan 'read' token'ı "
        "oluşturun ve .env dosyasına HF_TOKEN=... yazıp `docker compose up -d` ile yeniden "
        "başlatın."
    )
    return f"HuggingFace modeli indirilemedi: {model} - erişim reddedildi. {advice}"


def warm_hf_models(cfg: AppConfig) -> None:
    """Download the embedder and (if it will run) the reranker into the HF cache."""
    # Imported here: the factory pulls in the adapters, which tests of the other steps skip.
    from tubitak_rag.adapters.hf_cache import is_cached
    from tubitak_rag.factory import build_embedder, build_reranker, runtime_config

    runtime = runtime_config(cfg)
    logger.info(
        "Cihaz: embedder %s, reranker %s (TUBITAK_RAG_DEVICE=%s).",
        runtime.embedder.device,
        runtime.reranker.device if runtime.reranker.enabled else "kapalı",
        os.environ.get("TUBITAK_RAG_DEVICE", "auto"),
    )
    wanted: list[tuple[str, str, Callable[[], object]]] = []
    if runtime.embedder.provider == "sentence_transformers":
        wanted.append(("embedder", runtime.embedder.model, lambda: build_embedder(runtime)))
    if runtime.reranker.enabled and runtime.reranker.provider == "cross_encoder":
        wanted.append(("reranker", runtime.reranker.model, lambda: build_reranker(runtime)))
    for role, model, build in wanted:
        if is_cached(model):
            logger.info("HF modeli önbellekte: %s (%s)", model, role)
            continue
        logger.info("HF modeli indiriliyor: %s (%s)…", model, role)
        started = time.monotonic()
        try:
            build()
        except Exception as exc:
            hint = hf_access_error(exc, model, token_set=bool(os.environ.get(HF_TOKEN_ENV)))
            if hint is None:
                raise
            raise BootstrapError(hint) from exc
        logger.info("HF modeli hazır: %s (%.0f s)", model, time.monotonic() - started)


def restore_index_bundle(
    cfg: AppConfig, client: httpx.Client, environ: Mapping[str, str], *, ready: bool
) -> bool:
    """Step 4: restore the local or downloadable bundle when the stack lacks part of it.

    ``ready`` tells whether the configured collection is already usable; then a bundle
    problem is only a warning, otherwise it fails the service (falling back to a multi-hour
    ingest because of a typo in ``INDEX_BUNDLE_URL`` would be worse).
    """
    if not flag(environ, USE_BUNDLE_ENV):
        logger.info("Hazır indeks paketi kullanılmıyor (%s=0).", USE_BUNDLE_ENV)
        return False
    local = Path(environ.get(BUNDLE_DIR_ENV) or DEFAULT_BUNDLE_DIR)
    url = environ.get(BUNDLE_URL_ENV)
    qdrant_url = cfg.vector_store.url.rstrip("/")
    artifacts = cfg.paths.artifacts_dir
    try:
        source = locate_bundle(local, url, client, artifacts / BUNDLE_DOWNLOAD_DIR)
        if source is None:
            return False
        manifest = source.manifest
        if not pending_work(manifest, client, qdrant_url, artifacts):
            logger.info("Hazır indeks paketi zaten yüklü (%s).", manifest.created_at)
            source.cleanup()
            return False
        logger.info(
            "Hazır indeks paketi bulundu (%s, %d koleksiyon): belge indirme, ayrıştırma ve "
            "embedding atlanıyor.",
            manifest.created_at,
            len(manifest.collections),
        )
        started = time.monotonic()
        report = restore_bundle(
            source, client=client, qdrant_url=qdrant_url, artifacts_dir=artifacts
        )
    except (BundleError, httpx.HTTPError, OSError) as exc:
        if ready:
            logger.warning("Hazır indeks paketi okunamadı, mevcut indeksle devam: %s", exc)
            return False
        raise BootstrapError(
            f"Hazır indeks paketi yüklenemedi: {exc}. ./release klasörünü ya da "
            f"{BUNDLE_URL_ENV} değerini kontrol edin; paketi kullanmamak için onu kaldırın "
            f"(ya da {USE_BUNDLE_ENV}=0): o zaman belgeler indirilip baştan indekslenir."
        ) from exc
    logger.info(
        "Paket geri yüklendi (%.0f s): koleksiyon %s; BM25 %s; %d meta dosyası.",
        time.monotonic() - started,
        ", ".join(report.restored_collections) or "-",
        ", ".join(report.restored_bm25) or "-",
        report.meta_files,
    )
    return True


def bootstrap(
    cfg: AppConfig,
    config_path: Path,
    client: httpx.Client,
    *,
    environ: Mapping[str, str],
    run: Runner = run_subprocess,
    warm: Callable[[AppConfig], None] = warm_hf_models,
    sleep: Sleeper = time.sleep,
    restore: Restorer = restore_index_bundle,
) -> None:
    """Run the preparation steps described in the module docstring."""
    qdrant_url = cfg.vector_store.url.rstrip("/")
    ollama_url = cfg.llm.url.rstrip("/")
    wait_s = float(environ.get(WAIT_ENV, "300") or 300)
    wait_for("Qdrant", f"{qdrant_url}/healthz", client, timeout_s=wait_s, sleep=sleep)
    wait_for("Ollama", f"{ollama_url}/api/tags", client, timeout_s=wait_s, sleep=sleep)

    ensure_ollama_models(client, ollama_url, required_ollama_models(cfg))

    if flag(environ, WARMUP_ENV):
        warm(cfg)
    else:
        logger.info("HF model ısınması atlandı (%s=0).", WARMUP_ENV)

    collection = collection_name(chunker_key(cfg.chunker), cfg.embedder.model)
    points = collection_points(client, qdrant_url, collection)
    if restore(cfg, client, environ, ready=not needs_ingest(cfg, points)):
        points = collection_points(client, qdrant_url, collection)
        if flag(environ, BUNDLE_FETCH_ENV, default=False):
            logger.info("Kaynak belgeler indiriliyor (rag fetch; önizleme ve elle ingest için)…")
            if run(cli_command("fetch", "-c", str(config_path))) != 0:
                logger.warning("rag fetch bazı belgeleri indiremedi (ayrıntı: fetch_report.json).")
    if not needs_ingest(cfg, points):
        logger.info("Koleksiyon hazır: %s (%d parça); ingest gerekmiyor.", collection, points)
        return
    if not flag(environ, AUTO_INGEST_ENV):
        logger.warning(
            "Koleksiyon boş ya da BM25 indeksi yok (%s) ve %s=0: ingest atlandı. "
            "Elle başlatın: docker compose exec app rag ingest",
            collection,
            AUTO_INGEST_ENV,
        )
        return
    logger.info("Koleksiyon boş ya da eksik: %s (%d parça).", collection, points)
    fetch_and_ingest(config_path, run)


def main() -> None:
    """Entry point of the ``init`` service; exits 1 with a Turkish message on failure."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [hazırlık] %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
    config_path = Path(os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG)
    started = time.monotonic()
    try:
        cfg = load_config(config_path)
        logger.info("Yapılandırma: %s (deney %s)", config_path, cfg.experiment.name)
        with httpx.Client() as client:
            bootstrap(cfg, config_path, client, environ=os.environ)
    except (BootstrapError, ConfigError, httpx.HTTPError) as exc:
        logger.error("Hazırlık başarısız: %s", exc)
        sys.exit(1)
    logger.info(
        "Hazırlık tamamlandı (%.0f s). Uygulama başlatılabilir.", time.monotonic() - started
    )


if __name__ == "__main__":
    main()
