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


@pytest.mark.wad
def test_deathkings_every_exit_hub_and_reborn_check_passes(
    iwad_path: Path, hexdd_path: Path
) -> None:
    """S3b: every exit, hub return, cluster change and reborn in Deathkings (HEXDD.WAD)."""
    report = run(
        iwad_path,
        pwads=[hexdd_path],
        only=["exits", "hubs", "reborn"],
        log=lambda _msg: None,
    )
    failures = [c.line() for c in report.checks if c.status == "FAIL"]
    assert not failures, "\n".join(failures)
    exits = [c for c in report.checks if c.kind == "exit"]
    assert len(exits) >= 85  # every exit line and ACS exit across MAP33-38 and MAP41-60
    hubs = [c for c in report.checks if c.kind == "hub"]
    assert len(hubs) >= 4  # 3 hub returns + 1 cluster change
    assert any(c.name.startswith("cluster change") and c.status == "PASS" for c in hubs)
    reborns = [c for c in report.checks if c.kind == "reborn"]
    assert len(reborns) == 3 and all(c.status == "PASS" for c in reborns)
