"""Gradio entry point: ``python app_gradio.py`` (repository root); ``app.py`` on the Space.

``spaces`` is imported first: on a ZeroGPU Space it patches ``torch.cuda`` and must come
before anything that touches CUDA (torch, transformers, sentence-transformers). Locally the
package is optional; without it the GPU decorator is a no-op (``gradio_app/gpu.py``).

The Space installs only ``requirements.txt`` (no package), so ``src/`` is put on the import
path when ``tubitak_rag`` is not importable. On a Space (``SPACE_ID`` is set by Hugging
Face) the ZeroGPU defaults are filled in: the overlay ``configs/space-overlay.yaml`` (NumPy
index, CUDA, Qwen3 4B with transformers), no GPU lock and the default configuration ``v3``.
Details: ``deploy/hf-space-gradio/KURULUM.md``.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

try:
    import spaces  # noqa: F401  (must precede torch: ZeroGPU's CUDA emulation)
except ImportError:
    spaces = None

if importlib.util.find_spec("tubitak_rag") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tubitak_rag.interfaces.gradio_app.settings import space_defaults  # noqa: E402

space_defaults(os.environ)

from tubitak_rag.interfaces.gradio_app.app import build, launch_options  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
demo = build()  # every model is loaded here, at import time (ZeroGPU rule)

if __name__ == "__main__":
    demo.launch(**launch_options())
