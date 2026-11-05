# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for sector movers: doors, floors, ceilings, plats, pillars, lights, teleports."""

from __future__ import annotations

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.events import MapExit, PrintMessage
from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import spawn_player
from pyhexen.sim.specials.dispatcher import execute_line_special
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def _create_two_sector_sim() -> GameSim:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms()
    # Player 1 start in room 1
    mb.add_thing(tid=10, x=128, y=128, z=0, angle=0, type=1)
    # Destination thing with TID 20 in room 2 (type 14 = MT_TELEPORTMAN)
    mb.add_thing(tid=20, x=384, y=128, z=0, angle=0, type=14)
    wb.add_map(mb)
    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    level.sectors[1].tag = 1
    sim = GameSim(level=level)
    spawn_map_things(sim)
    return sim


def test_vertical_door_lifecycle() -> None:
    sim = _create_two_sector_sim()
    assert sim.level is not None
    sec = sim.level.sectors[1]
    # Door starts closed (ceiling at floor)
    sec.ceiling_height = sec.floor_height
    initial_ceil = sec.ceiling_height

    # Door_Raise (special 12)
    success = execute_line_special(sim, 12, (1, 16, 10, 0, 0))
    assert success
    assert len(sim.thinkers) > 0

    # Tick simulation until door raises to top
    for _ in range(50):
        sim.tick()
    assert sec.ceiling_height > initial_ceil


def test_locked_door_requires_key() -> None:
    sim = _create_two_sector_sim()
    player = spawn_player(sim, player_num=0)
    # Player has no keys:
    player.keys = 0

    messages: list[str] = []
    sim.events.subscribe(PrintMessage, lambda evt: messages.append(evt.text))

    # Door_LockedRaise (special 13) requiring key 1
    success = execute_line_special(sim, 13, (1, 16, 10, 1, 0), mo=player.mo)
    assert not success
    assert len(messages) > 0
    assert "KEY" in messages[0]

    # Give player key 1 (bit 0)
    player.keys = 1
    success_with_key = execute_line_special(sim, 13, (1, 16, 10, 1, 0), mo=player.mo)
    assert success_with_key


def test_floor_and_ceiling_movers() -> None:
    sim = _create_two_sector_sim()
    assert sim.level is not None
    sec = sim.level.sectors[1]
    init_floor = sec.floor_height

    # Floor_RaiseByValue (special 23)
    execute_line_special(sim, 23, (1, 16, 64, 0, 0))
    for _ in range(50):
        sim.tick()
    assert sec.floor_height == init_floor + (64 * FRACUNIT)

    # Ceiling_LowerByValue (special 40)
    init_ceil = sec.ceiling_height
    execute_line_special(sim, 40, (1, 16, 32, 0, 0))
    for _ in range(50):
        sim.tick()
    assert sec.ceiling_height == init_ceil - (32 * FRACUNIT)


def test_plat_and_pillar() -> None:
    sim = _create_two_sector_sim()
    assert sim.level is not None
    sec = sim.level.sectors[1]

    # Pillar_Build (special 29)
    execute_line_special(sim, 29, (1, 16, 0, 0, 0))
    for _ in range(80):
        sim.tick()
    # Floor and ceiling meet at the midpoint
    assert sec.floor_height == sec.ceiling_height


def test_light_specials() -> None:
    sim = _create_two_sector_sim()
    assert sim.level is not None
    sec = sim.level.sectors[1]

    # Light_ChangeToValue (special 112)
    execute_line_special(sim, 112, (1, 240, 0, 0, 0))
    assert sec.light_level == 240

    # Light_Fade (special 113)
    execute_line_special(sim, 113, (1, 50, 20, 0, 0))
    for _ in range(25):
        sim.tick()
    assert sec.light_level == 50


def test_teleport_and_thing_specials() -> None:
    sim = _create_two_sector_sim()
    player = spawn_player(sim, player_num=0)
    assert player.mo is not None
    player.mo.x = 0
    player.mo.y = 0

    # Teleport to destination with TID 20
    success = execute_line_special(sim, 70, (20, 0, 0, 0, 0), mo=player.mo)
    assert success
    dest = sim.find_mobjs_by_tid(20)[0]
    assert (player.mo.x, player.mo.y) == (dest.x, dest.y)

    # Teleport_NewMap (special 74)
    map_exits: list[MapExit] = []
    sim.events.subscribe(MapExit, lambda evt: map_exits.append(evt))
    execute_line_special(sim, 74, (2, 1, 0, 0, 0))
    assert len(map_exits) == 1
    assert map_exits[0].next_map == 2
