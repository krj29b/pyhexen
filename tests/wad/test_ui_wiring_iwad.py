# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""T15.3 with the real IWAD: automap line marking and save/load on MAP01."""

from __future__ import annotations

import time
from pathlib import Path

import pygame
import pytest

from pyhexen.app.game_flow import GameFlow, GameState
from pyhexen.app.game_ui import GameUI
from pyhexen.sim.savegame import SaveGameManager
from pyhexen.ui.menu import MenuManager
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import ML_MAPPED


@pytest.mark.wad
def test_map01_lines_seen_and_save_load_with_real_iwad(iwad_path: Path, tmp_path: Path) -> None:
    stack = WadStack(iwad_path)
    mapinfo = load_mapinfo(stack)
    saves = SaveGameManager(tmp_path)
    flow = GameFlow(stack, mapinfo, saves=saves)
    menu = MenuManager(wad=stack)
    ui = GameUI(stack, flow, menu, saves=saves)
    flow.on_level_loaded = ui.on_level_loaded
    flow.init_new(1, 0, 2)
    assert ui.map_name == "WINNOWING HALL"
    sim = flow.sim
    assert sim is not None and sim.level is not None

    # Looking around from the start marks part of the map, not all of it.
    player = flow.player
    assert player is not None and player.mo is not None
    start_angle = player.mo.angle
    t0 = time.perf_counter()
    for k in range(16):
        player.mo.angle = (start_angle + k * (2**32 // 16)) & 0xFFFFFFFF
        ui.mark_seen()
    per_pass_ms = (time.perf_counter() - t0) / 16 * 1000
    mapped = sum(1 for line in sim.level.lines if line.flags & ML_MAPPED)
    assert 50 < mapped < len(sim.level.lines) // 2
    assert per_pass_ms < 5.0
    player.mo.angle = start_angle

    # The mapped lines are part of the save (vanilla archives line flags).
    flow.save_game(0, "START")
    flow.tick()
    for line in sim.level.lines:
        line.flags &= ~ML_MAPPED
    ui.handle_key_down(pygame.K_F3)
    menu.handle_key(pygame.K_RETURN)
    flow.tick()
    assert flow.state == GameState.LEVEL and flow.sim is not sim
    assert flow.sim is not None and flow.sim.level is not None
    assert sum(1 for line in flow.sim.level.lines if line.flags & ML_MAPPED) == mapped
