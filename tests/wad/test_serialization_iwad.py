# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Real-IWAD serialization round-trip tests (T13.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import spawn_player
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


@pytest.mark.wad
def test_real_iwad_map01_serialization_roundtrip(iwad_path: Path) -> None:
    with WadStack(iwad_path) as wad:
        level1 = load_level(wad, "MAP01")
        sim1 = GameSim(level=level1)
        spawn_map_things(sim1)
        spawn_player(sim1, player_num=0)

        # Advance 35 tics (1 second of simulation)
        for _ in range(35):
            sim1.tick()

        hash_before = sim1.state_hash()
        data = sim1.save_to_bytes()

        # Load fresh level2
        level2 = load_level(wad, "MAP01")
        sim2 = GameSim(level=level2)
        sim2.load_from_bytes(data)

        # Hash must match immediately
        assert sim2.state_hash() == hash_before

        # Continue ticking both for another 35 tics
        for _ in range(35):
            sim1.tick()
            sim2.tick()
            assert sim1.state_hash() == sim2.state_hash()
