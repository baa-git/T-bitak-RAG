"""Gradio tests: the helpers need the ``web`` extra, the app smoke test ``gradio`` too."""

import importlib.util

_HAS_WEB = importlib.util.find_spec("markdown") is not None
_HAS_GRADIO = importlib.util.find_spec("gradio") is not None

collect_ignore_glob = [] if _HAS_WEB else ["test_*.py"]
if _HAS_WEB and not _HAS_GRADIO:
    collect_ignore_glob.append("test_app.py")
