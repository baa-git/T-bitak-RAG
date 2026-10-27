"""``rag export-space``: write the Space index (see ``space_bundle.py``).

Kept out of ``cli.py`` so the command is one registration line there.
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from tubitak_rag.config import ConfigError, load_config
from tubitak_rag.interfaces.space_bundle import (
    ARCHIVE_NAME,
    SpaceIndexError,
    cache_points,
    export_space_index,
    pack_space_index,
    qdrant_source,
)

DEFAULT_OUT = Path("release") / "space-index"


class PointSourceName(StrEnum):
    """Where the vectors are read from."""

    qdrant = "qdrant"
    cache = "cache"


def _mb(size: int) -> str:
    return f"{size / 1e6:.1f}".replace(".", ",") + " MB"


def _size(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def export_space(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Yapılandırma dosyası (YAML).")
    ] = Path("configs/default.yaml"),
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Space indeksinin yazılacağı klasör.")
    ] = DEFAULT_OUT,
    source: Annotated[
        PointSourceName,
        typer.Option(
            "--source",
            help="qdrant: çalışan Qdrant'tan oku · cache: yerel ingest önbelleklerinden kur.",
        ),
    ] = PointSourceName.qdrant,
    collections: Annotated[
        list[str] | None,
        typer.Option("--collection", help="Yalnız bu koleksiyon(lar); tekrarlanabilir."),
    ] = None,
    archive: Annotated[
        bool, typer.Option("--archive", help=f"Ayrıca {ARCHIVE_NAME} arşivi yaz (URL ile dağıtım).")
    ] = False,
) -> None:
    """Hugging Face Space için indeks üretir: NumPy vektörleri + BM25 + ayrıştırma önbelleği.

    Space'te Qdrant çalışmaz; koleksiyonlar düz dosyalara yazılır ve bellekte aranır.
    Çıktı klasörü artifacts/ düzenindedir: Space'te artifacts/ altına kopyalanır.
    """
    try:
        cfg = load_config(config)
        experiments = sorted((config.parent / "experiments").glob("*.yaml"))
        configs = [cfg, *(load_config(path) for path in experiments)]
    except ConfigError as exc:
        typer.echo(f"Yapılandırma hatası: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    point_source = (
        cache_points if source is PointSourceName.cache else qdrant_source(cfg.vector_store.url)
    )
    typer.echo(f"Kaynak: {source.value} · çıktı: {out}")
    try:
        manifest = export_space_index(
            configs,
            source=point_source,
            source_name=source.value,
            out_dir=out,
            collections=collections,
        )
    except SpaceIndexError as exc:
        typer.echo(f"Space indeksi üretilemedi: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for collection in manifest.collections:
        typer.echo(
            f"  {collection.name}: {collection.points} parça · {collection.embedder_model} "
            f"({collection.dimension} boyut)"
        )
    typer.echo(
        f"  BM25: {', '.join(manifest.bm25) or 'yok'} · ayrıştırılmış belge: "
        f"{manifest.parsed_documents}"
    )
    typer.echo(f"Toplam {_mb(_size(out))} · {out}")
    if archive:
        target = out.parent / ARCHIVE_NAME
        digest = pack_space_index(out, target)
        typer.echo(f"Arşiv: {target} ({_mb(target.stat().st_size)}) · sha256 {digest}")
