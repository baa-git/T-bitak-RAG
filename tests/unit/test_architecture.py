"""Architecture guards (NFR-MOD3, NFR-TEST1, NFR-TEST2).

Each guard is a small function over a source tree. The guards run against the real
repository and, to prove they can fail, against tiny synthetic trees in ``tmp_path``.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

WEB_FRAMEWORKS = {"fastapi", "starlette", "streamlit", "gradio", "spaces", "jinja2", "uvicorn"}
HEAVY = {
    "torch",
    "sentence_transformers",
    "transformers",
    "tokenizers",
    "qdrant_client",
    "anthropic",
    "bm25s",
}
PACKAGE = "tubitak_rag"


# --- helpers ------------------------------------------------------------------------------


def _top_names(nodes: Iterator[ast.AST]) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _all_imports(tree: ast.Module) -> set[str]:
    return _top_names(ast.walk(tree))


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _import_time_nodes(body: list[ast.stmt]) -> Iterator[ast.AST]:
    """Yield statements executed at import time (skips function bodies, TYPE_CHECKING)."""
    for stmt in body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        yield stmt
        if isinstance(stmt, ast.If):
            if not _is_type_checking(stmt.test):
                yield from _import_time_nodes(stmt.body)
            yield from _import_time_nodes(stmt.orelse)
        elif isinstance(stmt, ast.Try | ast.TryStar):
            for block in (stmt.body, stmt.orelse, stmt.finalbody):
                yield from _import_time_nodes(block)
            for handler in stmt.handlers:
                yield from _import_time_nodes(handler.body)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                yield from _import_time_nodes(case.body)
        elif isinstance(stmt, ast.For | ast.AsyncFor | ast.While):
            yield from _import_time_nodes(stmt.body)
            yield from _import_time_nodes(stmt.orelse)
        elif isinstance(stmt, ast.ClassDef | ast.With | ast.AsyncWith):
            yield from _import_time_nodes(stmt.body)


def _parse(py: Path) -> ast.Module:
    return ast.parse(py.read_text(encoding="utf-8"), filename=str(py))


def _relative_parts(py: Path, src_pkg: Path) -> tuple[str, ...]:
    return py.relative_to(src_pkg).parts


def web_import_violations(src_pkg: Path) -> list[str]:
    """Core modules (outside ``interfaces/``) that import a web framework."""
    return [
        str(py)
        for py in sorted(src_pkg.rglob("*.py"))
        if "interfaces" not in _relative_parts(py, src_pkg)
        and _all_imports(_parse(py)) & WEB_FRAMEWORKS
    ]


def heavy_import_violations(src_pkg: Path) -> list[str]:
    """Heavy imports outside ``adapters/`` or at module level inside ``adapters/``."""
    problems = []
    for py in sorted(src_pkg.rglob("*.py")):
        tree = _parse(py)
        if _relative_parts(py, src_pkg)[0] == "adapters":
            bad = _top_names(_import_time_nodes(tree.body)) & HEAVY
            problems += [f"{py}: modül seviyesinde {name}" for name in sorted(bad)]
        else:
            problems += [f"{py}: {name}" for name in sorted(_all_imports(tree) & HEAVY)]
    return problems


def missing_unit_tests(src_pkg: Path, unit_dir: Path) -> list[str]:
    """Source modules without a mirrored ``tests/unit/<pkg>/test_<module>.py``."""
    missing = []
    for py in sorted(src_pkg.rglob("*.py")):
        parts = _relative_parts(py, src_pkg)
        if py.name == "__init__.py" or parts[0] == "adapters" or "templates" in parts:
            continue
        expected = unit_dir.joinpath(*parts[:-1], f"test_{py.name}")
        if not expected.is_file():
            missing.append(str(expected.relative_to(unit_dir)))
    return missing


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- guards on the real repository --------------------------------------------------------


def test_core_does_not_import_web_frameworks(repo_root: Path) -> None:
    assert web_import_violations(repo_root / "src" / PACKAGE) == []  # NFR-MOD3


def test_heavy_libs_only_in_adapters_and_lazily(repo_root: Path) -> None:
    assert heavy_import_violations(repo_root / "src" / PACKAGE) == []  # NFR-TEST2


def test_unit_tests_mirror_src(repo_root: Path) -> None:
    unit_dir = repo_root / "tests" / "unit"
    assert missing_unit_tests(repo_root / "src" / PACKAGE, unit_dir) == []  # NFR-TEST1


# --- the guards themselves detect violations ----------------------------------------------


def test_web_guard_flags_core_module_and_allows_interfaces(tmp_path: Path) -> None:
    _write(tmp_path / "retrieval" / "dense.py", "from fastapi import FastAPI\n")
    _write(tmp_path / "interfaces" / "web" / "app.py", "import fastapi\n")
    violations = web_import_violations(tmp_path)
    assert len(violations) == 1 and violations[0].endswith("dense.py")


def test_heavy_guard_flags_core_and_module_level_adapter_imports(tmp_path: Path) -> None:
    _write(tmp_path / "chunking" / "fixed.py", "def f() -> None:\n    import tokenizers\n")
    _write(tmp_path / "adapters" / "eager.py", "import torch\n")
    _write(
        tmp_path / "adapters" / "lazy.py",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import torch\n"
        "class A:\n    def __init__(self) -> None:\n"
        "        from sentence_transformers import SentenceTransformer\n",
    )
    _write(
        tmp_path / "adapters" / "branches.py",
        "match 1:\n    case 1:\n        import qdrant_client\n"
        "try:\n    pass\nexcept* ValueError:\n    import bm25s\n"
        "for _ in []:\n    pass\nelse:\n    import anthropic\n",
    )
    violations = heavy_import_violations(tmp_path)
    assert {v.rsplit(" ", 1)[-1] for v in violations if "branches.py" in v} == {
        "qdrant_client",
        "bm25s",
        "anthropic",
    }
    violations = [v for v in violations if "branches.py" not in v]
    assert len(violations) == 2
    assert any("fixed.py: tokenizers" in v for v in violations)
    assert any("eager.py: modül seviyesinde torch" in v for v in violations)


def test_mirror_guard_reports_missing_test_module(tmp_path: Path) -> None:
    src, unit = tmp_path / "src", tmp_path / "unit"
    for rel in ["config.py", "domain/__init__.py", "domain/chunk.py", "adapters/x.py"]:
        _write(src / rel, "")
    _write(unit / "test_config.py", "")
    assert missing_unit_tests(src, unit) == [str(Path("domain") / "test_chunk.py")]
