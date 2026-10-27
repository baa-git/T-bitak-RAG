"""Whether a HuggingFace model is already in the local HF cache (``HF_HOME``)."""


def is_cached(repo_id: str) -> bool:
    """True when ``config.json`` of ``repo_id`` is cached; no network request is made.

    A model with its config cached but weights missing (an interrupted download) still
    reports True; the model loader then fetches the rest on first use.
    """
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
