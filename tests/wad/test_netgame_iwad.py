# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""S2d on the real IWAD: `pyhexen` processes play coop and deathmatch over localhost UDP.

Longer sessions: run the same command lines with a larger `--headless-frames` (see
README, "Multiplayer").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures.netsession import run_netgame


@pytest.mark.wad
def test_three_players_coop_on_winnowing_hall(iwad_path: Path) -> None:
    results = run_netgame(iwad_path, 3, 35 * 30, host_args=("--warp", "1"))
    for r in results:
        assert r.returncode == 0 and r.tic == 35 * 30 and r.desync == -1, r.output[-2000:]
    assert len({r.digest for r in results}) == 1


@pytest.mark.wad
def test_two_players_deathmatch_with_timer(iwad_path: Path) -> None:
    results = run_netgame(
        iwad_path, 2, 35 * 60 + 35 * 10, host_args=("--warp", "1", "--deathmatch", "--timer", "1")
    )
    for r in results:
        assert r.returncode == 0 and r.desync == -1, r.output[-2000:]
    assert all(r.map_num == 2 for r in results)
    assert len({r.digest for r in results}) == 1
