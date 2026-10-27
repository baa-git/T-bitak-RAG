"""Give GPU memory back after models are dropped (web interface pipeline cache).

Python frees a model only when the last reference disappears; PyTorch then keeps the freed
blocks in its caching allocator. ``gc.collect()`` breaks reference cycles and
``torch.cuda.empty_cache()`` returns the cached blocks to the driver, so another process
(an experiment run) can use them. Models served by Ollama live in the Ollama server and are
unloaded by its own ``keep_alive`` timer, not here.
"""

import gc


def release_gpu_memory() -> None:
    """Collect garbage and empty the CUDA cache when PyTorch and a GPU are available."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
