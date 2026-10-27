"""Web tests need the optional ``web`` extra (``pip install -e ".[dev,web]"``)."""

import importlib.util

# Without FastAPI/Jinja2 (a plain ``.[dev]`` install) these modules cannot be imported.
collect_ignore_glob = [] if importlib.util.find_spec("fastapi") else ["test_*.py"]
