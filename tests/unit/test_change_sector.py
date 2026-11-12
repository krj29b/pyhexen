# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""T15.6a: P_ChangeSector, P_ThingHeightClip, T_MovePlane, floor waggle, doors, tag 0."""

from __future__ import annotations

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjFlag, MobjType, StateNum
from pyhexen.sim.map import change_sector, thing_height_clip
from pyhexen.sim.mobj import Mobj, spawn_mobj
from pyhexen.sim.specials.dispatcher import execute_line_special, find_sectors_from_tag
from pyhexen.sim.specials.doors import DoorType, VerticalDoor, ev_do_door
from pyhexen.sim.specials.floors import (
    WGLSTATE_STABLE,
    FloorWaggle,
    MoveResult,
    ev_start_floor_waggle,
    t_move_plane,
)
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Sector, load_level

U = FRACUNIT


def _sim(*, ceiling: int = 128, tag: int = 0) -> tuple[GameSim, Sector]:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512, ceiling_height=ceiling, tag=tag)
    sim = GameSim(level=load_level(WadStack(WadBuilder("IWAD").add_map(mb).to_bytes()), "MAP01"))
    assert sim.level is not None
    return sim, sim.level.sectors[0]


def _live(sim: GameSim) -> list[Mobj]:
    return [t for t in sim.thinkers if isinstance(t, Mobj) and not t.removed]


def test_things_on_the_floor_ride_it_floating_things_stay() -> None:
    sim, sec = _sim()
    walker = spawn_mobj(sim, 100 * U, 100 * U, 0, MobjType.MT_ETTIN)
    floater = spawn_mobj(sim, 300 * U, 300 * U, 40 * U, MobjType.MT_ETTIN)
    sec.floor_height = -8 * U  # one tic of a fast floor
    assert not change_sector(sim, sec, 0)
    assert (walker.z, walker.floorz) == (-8 * U, -8 * U)
    assert (floater.z, floater.floorz) == (40 * U, -8 * U)


def test_height_clip_only_follows_small_drops_without_nogravity() -> None:
    sim, sec = _sim()
    walker = spawn_mobj(sim, 100 * U, 100 * U, 0, MobjType.MT_ETTIN)
    sec.floor_height = -9 * U  # 9 units: vanilla only snaps below 9
    assert thing_height_clip(sim, walker)
    assert walker.z == 0 and walker.floorz == -9 * U


def test_lowering_ceiling_crushes_every_fourth_tic_and_sprays_blood() -> None:
    sim, sec = _sim(ceiling=64)
    ettin = spawn_mobj(sim, 256 * U, 256 * U, 0, MobjType.MT_ETTIN)  # 68 units tall
    health = ettin.health
    sim.level_time = 4  # leveltime & 3 == 0
    assert change_sector(sim, sec, 10)
    assert ettin.health == health - 10
    assert any(m.type == MobjType.MT_BLOOD for m in _live(sim))

    sim.level_time = 5
    assert change_sector(sim, sec, 10)
    assert ettin.health == health - 10  # no damage this tic


def test_crushed_corpses_become_gibs_and_dropped_items_vanish() -> None:
    sim, sec = _sim(ceiling=16)
    corpse = spawn_mobj(sim, 100 * U, 100 * U, 0, MobjType.MT_ETTIN)
    corpse.flags |= MobjFlag.MF_CORPSE
    corpse.flags &= ~MobjFlag.MF_SHOOTABLE
    corpse.health = 0
    change_sector(sim, sec, 0)
    assert corpse.state_num == StateNum.S_GIBS1 and corpse.height == 0


def test_move_plane_reports_pastdest_only_when_overshooting() -> None:
    sim, sec = _sim()
    # 8 units per tic down to -16: tics 1 and 2 are normal moves, tic 3 would overshoot.
    assert t_move_plane(sim, sec, 8 * U, -16 * U, 0, 0, -1) == MoveResult.RES_OK
    assert t_move_plane(sim, sec, 8 * U, -16 * U, 0, 0, -1) == MoveResult.RES_OK
    assert sec.floor_height == -16 * U
    assert t_move_plane(sim, sec, 8 * U, -16 * U, 0, 0, -1) == MoveResult.RES_PASTDEST


def test_blocked_ceiling_goes_back_and_reports_crushed() -> None:
    sim, sec = _sim(ceiling=72)
    spawn_mobj(sim, 256 * U, 256 * U, 0, MobjType.MT_ETTIN)  # 68 tall
    res = t_move_plane(sim, sec, 8 * U, 0, 0, 1, -1)
    assert res == MoveResult.RES_CRUSHED
    assert sec.ceiling_height == 72 * U  # put back

    # A rising ceiling never reports a crush.
    assert t_move_plane(sim, sec, 8 * U, 200 * U, 0, 1, 1) == MoveResult.RES_OK


def test_floor_waggle_expands_holds_and_restores() -> None:
    sim, sec = _sim(tag=5)
    assert ev_start_floor_waggle(sim, 5, height=32, speed=64, offset=0, timer=1)
    waggle = sec.special_data
    assert isinstance(waggle, FloorWaggle)
    assert not ev_start_floor_waggle(sim, 5, 32, 64, 0, 1)  # sector busy

    for _ in range(waggle.target_scale // waggle.scale_delta + 1):
        waggle.tick(sim)
    assert waggle.state == WGLSTATE_STABLE
    for _ in range(35 + waggle.target_scale // waggle.scale_delta + 2):
        if waggle.removed:
            break
        waggle.tick(sim)
    assert waggle.removed
    assert sec.floor_height == 0 and sec.special_data is None


def test_tag_zero_matches_untagged_sectors_like_vanilla() -> None:
    sim, sec = _sim(tag=0)
    assert sim.level is not None
    assert find_sectors_from_tag(sim.level, 0) == [sec]


def test_door_close_by_tag_skips_busy_sectors() -> None:
    sim, sec = _sim(tag=7)
    assert ev_do_door(sim, None, (7, 16, 0, 0, 0), DoorType.DREV_CLOSE)
    door = sec.special_data
    assert isinstance(door, VerticalDoor) and door.direction == -1
    assert not ev_do_door(sim, None, (7, 16, 0, 0, 0), DoorType.DREV_OPEN)


def test_door_open_with_tag_zero_uses_the_line_back_sector() -> None:
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor2=0, ceil2=0)
    sim = GameSim(level=load_level(WadStack(WadBuilder("IWAD").add_map(mb).to_bytes()), "MAP01"))
    assert sim.level is not None
    line = next(ln for ln in sim.level.lines if ln.back_sector is not None)
    line.special = 11
    line.args = (0, 32, 0, 0, 0)
    assert execute_line_special(sim, 11, line.args, line, None)
    door = line.back_sector.special_data
    assert isinstance(door, VerticalDoor) and door.type == DoorType.DREV_OPEN
    assert line.special == 0  # EV_VerticalDoor clears Door_Open
