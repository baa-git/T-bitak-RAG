"""ZeroGPU without a hard dependency: ``spaces.GPU`` when the package is there, else a no-op.

On a Hugging Face ZeroGPU Space the ``spaces`` package is preinstalled. Its decorator gives the
wrapped function a real GPU for at most ``duration`` seconds, in a forked worker process;
outside ZeroGPU (``SPACES_ZERO_GPU`` unset) it returns the function unchanged. Locally the
package is usually absent, and then this module returns the function unchanged as well.

``spaces`` must be imported before torch touches CUDA, so the entry files (``app.py`` on the
Space, ``app_gradio.py`` in the repository) import it first; importing it again here is free.
"""

from collections.abc import Callable
from typing import Any, cast

Duration = int | Callable[..., int]


def gpu_task[F: Callable[..., Any]](fn: F, duration: Duration) -> F:
    """``fn`` wrapped by ``spaces.GPU(duration=duration)``, or ``fn`` itself without ``spaces``.

    ``duration`` may be a callable that receives the same arguments as ``fn`` (ZeroGPU's
    dynamic duration) and returns seconds.
    """
    try:
        import spaces
    except ImportError:
        return fn
    return cast(F, spaces.GPU(duration=duration)(fn))
