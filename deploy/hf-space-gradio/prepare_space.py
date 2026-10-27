"""Assemble the Hugging Face Space folder (``build/hf-space-gradio``) for Gradio + ZeroGPU.

The Space repository is not this repository: it needs its own ``README.md`` (Space settings in
the YAML front matter: Gradio SDK, Python 3.12, models to preload) and ``requirements.txt``,
the entry file ``app.py`` (``app_gradio.py`` of this repository), the code under ``src/``,
the configs plus the ZeroGPU overlay, the manifests (Hakkında) and, by default, the Space index
under ``artifacts/`` (written by ``rag export-space``). Usage, from the repository root::

    python deploy/hf-space-gradio/prepare_space.py              # index from release/space-index
    python deploy/hf-space-gradio/prepare_space.py --no-index   # index via SPACE_INDEX_URL
    hf upload <kullanici>/<space> build/hf-space-gradio . --repo-type space

The Space runs Python 3.12 while the package targets 3.13; nothing is installed from
``pyproject.toml`` there (``app.py`` puts ``src/`` on the import path), so its
``requires-python`` pin does not apply. Step-by-step instructions (Turkish):
``deploy/hf-space-gradio/KURULUM.md``.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPACE_MANIFEST = "space_manifest.json"
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
# Large binary files go through Git LFS / Xet when the folder is pushed with git
# (``hf upload`` handles them by itself).
GITATTRIBUTES = """\
*.npy filter=lfs diff=lfs merge=lfs -text
*.jsonl filter=lfs diff=lfs merge=lfs -text
artifacts/cache/** filter=lfs diff=lfs merge=lfs -text
"""


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, ignore=IGNORED)
    else:
        shutil.copy2(source, target)


def _size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def prepare(out: Path, index: Path | None) -> None:
    """Write the Space folder to ``out`` (replaced if it exists)."""
    if index is not None and not (index / SPACE_MANIFEST).is_file():
        sys.exit(
            f"Space indeksi bulunamadı: {index / SPACE_MANIFEST}\n"
            "Önce `rag export-space` çalıştırın ya da --no-index ile SPACE_INDEX_URL kullanın."
        )
    if out.exists():
        shutil.rmtree(out)
    files = {
        ROOT / "app_gradio.py": out / "app.py",
        ROOT / "LICENSE": out / "LICENSE",
        ROOT / "src" / "tubitak_rag": out / "src" / "tubitak_rag",
        ROOT / "configs" / "default.yaml": out / "configs" / "default.yaml",
        ROOT / "configs" / "experiments": out / "configs" / "experiments",
        HERE / "space-overlay.yaml": out / "configs" / "space-overlay.yaml",
        ROOT / "data" / "manifests": out / "data" / "manifests",
        HERE / "README.md": out / "README.md",
        HERE / "requirements.txt": out / "requirements.txt",
    }
    for source, target in files.items():
        _copy(source, target)
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    if index is not None:
        _copy(index, out / "artifacts")
        print(f"İndeks: {index} ({_size_mb(out / 'artifacts'):.1f} MB)")
    print(f"Space klasörü hazır: {out} ({_size_mb(out):.1f} MB)")
    print(
        f"Yüklemek için: hf upload <kullanici>/<space> {out} . --repo-type space "
        '--exclude "**/__pycache__/**"'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio + ZeroGPU Space klasörünü hazırlar.")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "hf-space-gradio")
    parser.add_argument("--index", type=Path, default=ROOT / "release" / "space-index")
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="İndeksi klasöre koyma (Space onu SPACE_INDEX_URL'den indirir).",
    )
    args = parser.parse_args()
    prepare(args.out.resolve(), None if args.no_index else args.index.resolve())


if __name__ == "__main__":
    main()
