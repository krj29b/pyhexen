# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Real-IWAD tests for T5.4: spawn map things on MAP01 across skills and player classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


@pytest.mark.wad
def test_map01_spawning_counts_by_skill_and_class(iwad_path: Path) -> None:
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP01")

    counts_by_skill: dict[int, int] = {}
    for skill in (0, 2, 4):
        sim = GameSim(level=level, skill=skill, player_class=0)
        spawned = spawn_map_things(sim)
        counts_by_skill[skill] = len(spawned)
        assert len(spawned) > 0
        assert (0, 0) in sim.player_starts

    # Hard/Nightmare spawns at least as many things as Baby/Easy on MAP01
    assert counts_by_skill[4] >= counts_by_skill[0]

    # Verify all 3 classes (Fighter=0, Cleric=1, Mage=2) spawn valid mobjs and deterministic hashes
    for pclass in (0, 1, 2):
        sim1 = GameSim(level=level, skill=2, player_class=pclass)
        sim2 = GameSim(level=level, skill=2, player_class=pclass)
        s1 = spawn_map_things(sim1)
        s2 = spawn_map_things(sim2)
        assert len(s1) == len(s2) > 0
        assert sim1.state_hash() == sim2.state_hash()
