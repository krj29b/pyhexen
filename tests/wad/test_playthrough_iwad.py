# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""T15.7 on the real IWAD: every exit, hub persistence and reborn, through the game flow.

The per-class soak takes minutes, so it is left to `python -m pyhexen.tools.playthrough`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.tools.playthrough import run


@pytest.mark.wad
def test_every_exit_hub_and_reborn_check_passes(iwad_path: Path) -> None:
    report = run(iwad_path, only=["exits", "hubs", "reborn"], log=lambda _msg: None)
    failures = [c.line() for c in report.checks if c.status == "FAIL"]
    assert not failures, "\n".join(failures)
    exits = [c for c in report.checks if c.kind == "exit"]
    assert len(exits) >= 61  # every exit line in the IWAD, plus the ACS exits
    assert sum(c.kind == "hub" for c in report.checks) >= 6
