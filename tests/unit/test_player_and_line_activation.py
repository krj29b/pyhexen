# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for player thinker, line activation, switch toggling, and special sectors."""

from __future__ import annotations

from typing import Any

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.ticcmd import AFLAG_JUMP, TicCmd
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.line_activation import (
    SPAC_CROSS,
    SPAC_USE,
    p_activate_line,
)
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import (
    VIEWHEIGHT,
    PlayerState,
    p_death_think,
    p_move_player,
    spawn_player,
)
from pyhexen.sim.specials.dispatcher import register_special
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import ML_REPEAT_SPECIAL, ML_SPAC_SHIFT, load_level


def _create_test_sim() -> GameSim:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 512, 512)
    # Player 1 start
    mb.add_thing(tid=0, x=256, y=256, z=0, angle=0, type=1)
    wb.add_map(mb)
    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)
    spawn_map_things(sim)
    return sim


def test_player_spawn_and_movement() -> None:
    sim = _create_test_sim()
    player = spawn_player(sim, player_num=0)
    assert player.mo is not None
    assert player.health == 100
    assert player.player_state == PlayerState.PST_LIVE
    assert player.viewheight == VIEWHEIGHT

    # Apply forward movement cmd
    player.cmd = TicCmd(forwardmove=50)
    p_move_player(sim, player)
    assert player.mo.momx > 0

    # Test jump
    player.cmd = TicCmd(arti=AFLAG_JUMP)
    p_move_player(sim, player)
    assert player.mo.momz == 9 * FRACUNIT
    assert player.jump_tics > 0


def test_line_spac_activations() -> None:
    sim = _create_test_sim()
    assert sim.level is not None
    line = sim.level.lines[0]
    line.special = 999
    line.flags = (SPAC_USE << ML_SPAC_SHIFT) | ML_REPEAT_SPECIAL

    activated = []

    def mock_handler(sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any) -> bool:
        activated.append(special)
        return True

    register_special(999, mock_handler, "TestSpecial")

    player = spawn_player(sim, player_num=0)
    # Attempt SPAC_CROSS on a SPAC_USE line -> should fail
    res_cross = p_activate_line(sim, line, player.mo, side=0, activation_type=SPAC_CROSS)
    assert not res_cross
    assert len(activated) == 0

    # Attempt SPAC_USE -> succeeds
    res_use = p_activate_line(sim, line, player.mo, side=0, activation_type=SPAC_USE)
    assert res_use
    assert activated == [999]


def test_player_death_think() -> None:
    sim = _create_test_sim()
    player = spawn_player(sim, player_num=0)
    player.player_state = PlayerState.PST_DEAD
    player.viewheight = VIEWHEIGHT
    p_death_think(sim, player)
    # View height lowers on death
    assert player.viewheight < VIEWHEIGHT
