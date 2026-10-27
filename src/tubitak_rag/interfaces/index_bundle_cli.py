"""``rag export-index``: write the prebuilt index bundle (see ``index_bundle.py``).

Kept out of ``cli.py`` so the command is one registration line there.
"""

from pathlib import Path
from typing import Annotated

import httpx
import typer

from tubitak_rag.config import ConfigError, load_config
from tubitak_rag.interfaces.index_bundle import (
    MANIFEST_FILE,
    BundleError,
    collection_specs,
    export_bundle,
)

DEFAULT_OUT = Path("release")


def _mb(size: int) -> str:
    return f"{size / 1e6:.1f}".replace(".", ",") + " MB"


def export_index(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Yapılandırma dosyası (YAML).")
    ] = Path("configs/default.yaml"),
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Paket klasörü (docker compose varsayılan olarak ./release'i okur).",
        ),
    ] = DEFAULT_OUT,
    collections: Annotated[
        list[str] | None,
        typer.Option("--collection", help="Yalnız bu koleksiyon(lar); tekrarlanabilir."),
    ] = None,
) -> None:
    """Hazır indeks paketi üretir: Qdrant anlık görüntüleri + BM25 + meta + manifest.json.

    Yapılandırmanın ve configs/experiments altındaki deneylerin koleksiyonları Qdrant'ın
    snapshot API'siyle okunur (koleksiyonlar değişmez). Docker yığını bu paketle ilk
    açılışta belge indirme, ayrıştırma ve embedding yapmadan hazır olur.
    """
    try:
        cfg = load_config(config)
        experiments = sorted((config.parent / "experiments").glob("*.yaml"))
        specs = collection_specs([cfg, *(load_config(path) for path in experiments)])
    except ConfigError as exc:
        typer.echo(f"Yapılandırma hatası: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if collections:
        unknown = sorted(set(collections) - {s.name for s in specs})
        if unknown:
            known = ", ".join(s.name for s in specs)
            typer.echo(
                f"Bilinmeyen koleksiyon: {', '.join(unknown)} (seçenekler: {known})", err=True
            )
            raise typer.Exit(code=2)
        specs = [s for s in specs if s.name in collections]
    typer.echo(
        f"Qdrant: {cfg.vector_store.url} · koleksiyonlar: {', '.join(s.name for s in specs)}"
    )
    try:
        with httpx.Client() as client:
            manifest = export_bundle(
                specs,
                client=client,
                qdrant_url=cfg.vector_store.url,
                artifacts_dir=cfg.paths.artifacts_dir,
                out_dir=out,
            )
    except BundleError as exc:
        typer.echo(f"Paket üretilemedi: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        typer.echo(f"Qdrant'a ulaşılamadı ({cfg.vector_store.url}): {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for collection in manifest.collections:
        typer.echo(
            f"  {collection.name}: {collection.points} parça · {collection.embedder_model} "
            f"({collection.dimension} boyut)"
        )
    total = 0
    for item in manifest.files:
        total += item.size
        typer.echo(f"  {item.name:<52} {_mb(item.size):>10}  sha256 {item.sha256[:12]}…")
    typer.echo(f"Toplam {_mb(total)} · {out / MANIFEST_FILE}")
