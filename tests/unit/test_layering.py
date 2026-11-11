# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Layering rules from AGENTS.md / the plan §3, enforced on the source tree (T15.5).

`sim` (and the layers below it) must never import `render`, `audio`, `ui` or `app`,
not even under `TYPE_CHECKING` or inside a function.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import pyhexen

SRC = Path(pyhexen.__file__).parent
FORBIDDEN = ("pyhexen.render", "pyhexen.audio", "pyhexen.ui", "pyhexen.app")
LOWER_LAYERS = ("sim", "world", "wad", "core")


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    """Every module a file imports, from any `import` / `from ... import` statement."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
            # `from pyhexen import ui` imports the package too
            if node.module == "pyhexen":
                found.extend((node.lineno, f"pyhexen.{alias.name}") for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append((node.lineno, node.args[0].value))
    return found


def _is_forbidden(module: str) -> bool:
    return any(module == f or module.startswith(f + ".") for f in FORBIDDEN)


@pytest.mark.parametrize("layer", LOWER_LAYERS)
def test_lower_layers_do_not_import_upper_layers(layer: str) -> None:
    offenders = [
        f"{path.relative_to(SRC)}:{line} imports {module}"
        for path in sorted((SRC / layer).rglob("*.py"))
        for line, module in _imported_modules(path)
        if _is_forbidden(module)
    ]
    assert offenders == []


def test_checker_catches_type_checking_imports(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from pyhexen.ui.menu import MenuManager\n"
        "def f():\n"
        "    import pyhexen.render.camera\n",
        encoding="utf-8",
    )
    modules = [m for _, m in _imported_modules(sample) if _is_forbidden(m)]
    assert modules == ["pyhexen.ui.menu", "pyhexen.render.camera"]
