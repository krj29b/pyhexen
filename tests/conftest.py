# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.app.config import resolve_iwad


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--iwad",
        action="store",
        default=None,
        help="Path to HEXEN.WAD for @pytest.mark.wad tests",
    )


@pytest.fixture(scope="session")
def iwad_path(request: pytest.FixtureRequest) -> Path:
    """Resolve the HEXEN.WAD path from --iwad, PYHEXEN_IWAD, or config.toml.
    Skips the test cleanly if no valid IWAD is configured.
    """
    cli_opt = request.config.getoption("--iwad")
    resolved = resolve_iwad(cli_opt, must_exist=True)
    if resolved is None:
        pytest.skip("No HEXEN.WAD configured (set --iwad, PYHEXEN_IWAD, or iwad in config.toml)")
    return resolved
