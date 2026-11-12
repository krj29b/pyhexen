# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Line specials ported in T15.8: every P_ExecuteLineSpecial case has a handler.

The expected values are worked out from p_floor.c, p_ceilng.c, p_plats.c and p_things.c.
The test map is two rooms sharing a two-sided line: sector 0 (floor 0, ceiling 128)
and sector 1 (floor 16, ceiling 128, tag 1), both with the same floor flat.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjFlag, MobjFlag2, MobjType
from pyhexen.sim.mobj import Mobj, insert_mobj_into_tid_list, spawn_map_things, spawn_mobj
from pyhexen.sim.player import spawn_player
from pyhexen.sim.specials.ceilings import CeilingMove
from pyhexen.sim.specials.dispatcher import _HANDLERS, execute_line_special, get_specials_report
from pyhexen.sim.specials.floors import FloorMove, FloorType, MoveResult, t_move_plane
from pyhexen.sim.specials.pillars import Pillar
from pyhexen.sim.specials.plats import Plat, PlatStatus
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Sector, load_level

F = FRACUNIT

# Every `case` of P_ExecuteLineSpecial (p_spec.c), 81 in all.
VANILLA_SPECIALS: frozenset[int] = frozenset(
    [
        *range(1, 9),
        *range(10, 14),
        *range(20, 33),
        35,
        36,
        *range(40, 47),
        *range(60, 76),
        *range(80, 84),
        *range(90, 97),
        *range(109, 117),
        120,
        *range(129, 139),
        140,
    ]
)

# Thing types of TranslateThingType (p_things.c) used below.
T_ETTIN = 4
T_FIREBALL1 = 10

SPOT_TID = 20


def _sim(*, nomonsters: bool = False) -> GameSim:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms()
    mb.add_thing(tid=10, x=128, y=128, z=0, angle=0, type=1)
    # A map spot (type 14 = MT_TELEPORTMAN) with TID 20 in the second room.
    mb.add_thing(tid=SPOT_TID, x=384, y=128, z=0, angle=0, type=14)
    wb.add_map(mb)
    level = load_level(WadStack(wb.to_bytes()), "MAP01")
    level.sectors[1].tag = 1
    sim = GameSim(level=level, nomonsters=nomonsters)
    spawn_map_things(sim)
    return sim


def _sectors(sim: GameSim) -> tuple[Sector, Sector]:
    assert sim.level is not None
    return sim.level.sectors[0], sim.level.sectors[1]


def _run(sim: GameSim, tics: int) -> None:
    for _ in range(tics):
        sim.tick()


def _mobjs(sim: GameSim, mobj_type: MobjType) -> list[Mobj]:
    return [t for t in sim.thinkers if isinstance(t, Mobj) and t.type == mobj_type]


def _movers(sim: GameSim, cls: type) -> list:
    return [t for t in sim.thinkers if isinstance(t, cls)]


# --- Coverage of P_ExecuteLineSpecial ------------------------------------------------


def test_every_vanilla_special_has_a_handler() -> None:
    report = get_specials_report()
    missing = sorted(s for s in VANILLA_SPECIALS if s not in _HANDLERS)
    assert missing == []
    assert len(VANILLA_SPECIALS) == 81
    assert all(report[s][1] for s in VANILLA_SPECIALS)


def test_vanilla_special_list_matches_reference() -> None:
    source = Path(__file__).resolve().parents[2] / "reference" / "src" / "hexen" / "p_spec.c"
    if not source.exists():
        pytest.skip("reference source not fetched")
    text = source.read_text(encoding="latin-1")
    body = text[text.index("boolean P_ExecuteLineSpecial") : text.index("// P_ActivateLine")]
    cases = {int(n) for n in re.findall(r"^\s*case (\d+):", body, flags=re.MULTILINE)}
    assert cases == VANILLA_SPECIALS


def test_unknown_special_is_inert() -> None:
    assert execute_line_special(_sim(), 200, (1, 2, 3, 4, 5)) is False


# --- T_MovePlane ---------------------------------------------------------------------


def test_move_plane_with_direction_zero_does_nothing() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert t_move_plane(sim, sec, 4 * F, 64 * F, 0, 0, 0) == MoveResult.RES_OK
    assert sec.floor_height == 16 * F


# --- EV_DoFloor ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("special", "args", "expected_floor"),
    [
        (20, (1, 16, 8, 0, 0), 8),  # Floor_LowerByValue
        (21, (1, 16, 0, 0, 0), 0),  # Floor_LowerToLowest: the neighbour's floor
        (22, (1, 16, 0, 0, 0), 0),  # Floor_LowerToNearest: highest surrounding floor
        (23, (1, 16, 8, 0, 0), 24),  # Floor_RaiseByValue
        (35, (1, 16, 2, 0, 0), 32),  # Floor_RaiseByValueTimes8
        (36, (1, 16, 2, 0, 0), 0),  # Floor_LowerByValueTimes8
        (68, (1, 16, 5, 0, 0), 40),  # Floor_MoveToValueTimes8, up
        (68, (1, 16, 1, 1, 0), -8),  # Floor_MoveToValueTimes8, negative value
    ],
)
def test_floor_specials_reach_their_destination(
    special: int, args: tuple[int, ...], expected_floor: int
) -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, special, args)
    assert isinstance(sec.special_data, FloorMove)
    assert sec.special_data.speed == 2 * F  # 16 * FRACUNIT / 8
    _run(sim, 20)
    assert sec.floor_height == expected_floor * F
    assert sec.special_data is None
    assert _movers(sim, FloorMove) == []


def test_floor_raise_to_nearest_uses_next_highest_floor() -> None:
    sim = _sim()
    sec0, _ = _sectors(sim)
    sec0.tag = 2
    assert execute_line_special(sim, 25, (2, 16, 0, 0, 0))
    _run(sim, 12)
    assert sec0.floor_height == 16 * F


@pytest.mark.parametrize(("special", "expected_floor"), [(66, 0), (67, 32)])
def test_floor_instant_moves_in_one_tic(special: int, expected_floor: int) -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, special, (1, 0, 2, 0, 0))
    assert sec.special_data.speed == 2000 * F
    sim.tick()
    assert sec.floor_height == expected_floor * F
    assert sec.special_data is None


def test_floor_move_to_current_value_leaves_a_stuck_thinker() -> None:
    """Vanilla reports failure but keeps a thinker with direction 0 in the sector."""
    sim = _sim()
    _, sec = _sectors(sim)
    assert not execute_line_special(sim, 68, (1, 16, 2, 0, 0))  # 2 * 8 = 16, where it is
    stuck = sec.special_data
    assert isinstance(stuck, FloorMove) and stuck.direction == 0
    _run(sim, 10)
    assert sec.floor_height == 16 * F
    assert sec.special_data is stuck and not stuck.removed
    # The sector stays busy for other floor specials.
    assert not execute_line_special(sim, 23, (1, 16, 8, 0, 0))


def test_floor_special_skips_busy_sectors() -> None:
    sim = _sim()
    assert execute_line_special(sim, 23, (1, 16, 8, 0, 0))
    assert not execute_line_special(sim, 20, (1, 16, 8, 0, 0))
    assert len(_movers(sim, FloorMove)) == 1


def test_floor_raise_and_crush_and_crush_stop() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 28, (1, 8, 10, 0, 0))
    floor = sec.special_data
    assert floor.type == FloorType.FLEV_RAISEFLOORCRUSH
    assert (floor.crush, floor.floordestheight) == (10, 120 * F)  # ceiling - 8
    _run(sim, 3)
    # Floor_CrushStop ignores its arguments and removes every crushing floor.
    assert execute_line_special(sim, 46, (99, 0, 0, 0, 0))
    assert sec.special_data is None and floor.removed
    height = sec.floor_height
    _run(sim, 3)
    assert sec.floor_height == height
    assert not execute_line_special(sim, 46, (1, 0, 0, 0, 0))


@pytest.mark.parametrize(("special", "floor", "ceiling"), [(95, 8, 120), (96, 24, 136)])
def test_floor_and_ceiling_move_together(special: int, floor: int, ceiling: int) -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, special, (1, 16, 8, 0, 0))
    assert len(_movers(sim, FloorMove)) == 1 and len(_movers(sim, CeilingMove)) == 1
    assert isinstance(sec.special_data, CeilingMove)  # the floor's claim was dropped
    _run(sim, 10)
    assert (sec.floor_height, sec.ceiling_height) == (floor * F, ceiling * F)
    assert sec.special_data is None


# --- EV_BuildStairs ------------------------------------------------------------------


def _stairs_sim() -> GameSim:
    """Sector 1 (tag 1) starts the stairs; sector 0 has stair special 26, so it is next."""
    sim = _sim()
    sec0, sec1 = _sectors(sim)
    sec0.special = 26
    sec1.special = 5
    return sim


def test_stairs_build_up_normal() -> None:
    sim = _stairs_sim()
    sec0, sec1 = _sectors(sim)
    assert execute_line_special(sim, 27, (1, 16, 16, 0, 0))
    steps = _movers(sim, FloorMove)
    assert [s.sector.index for s in steps] == [1, 0]
    assert all(s.type == FloorType.FLEV_RAISEBUILDSTEP and s.speed == 2 * F for s in steps)
    # Each step ends one step height above the previous step's destination.
    assert [s.floordestheight for s in steps] == [32 * F, 48 * F]
    assert sec1.special == 0  # the tagged sector's special is cleared
    assert sec0.special == 26  # a queued sector keeps its special
    _run(sim, 30)
    assert (sec1.floor_height, sec0.floor_height) == (32 * F, 48 * F)
    assert sec0.special_data is None and sec1.special_data is None
    assert _movers(sim, FloorMove) == []


def test_stairs_build_down_normal() -> None:
    sim = _stairs_sim()
    sec0, sec1 = _sectors(sim)
    assert execute_line_special(sim, 26, (1, 16, 16, 0, 0))
    _run(sim, 20)
    assert (sec1.floor_height, sec0.floor_height) == (0, -16 * F)


def test_stairs_need_matching_special_and_flat() -> None:
    for change in ("special", "flat"):
        sim = _stairs_sim()
        sec0, _ = _sectors(sim)
        if change == "special":
            sec0.special = 27  # the first neighbour must be 26, the next 27
        else:
            sec0.floor_pic = "OTHER"
        assert execute_line_special(sim, 27, (1, 16, 16, 0, 0))
        assert [s.sector.index for s in _movers(sim, FloorMove)] == [1]


def test_stairs_sync_scales_speed_by_step_number() -> None:
    sim = _stairs_sim()
    assert execute_line_special(sim, 31, (1, 8, 16, 0, 0))
    first, second = _movers(sim, FloorMove)
    # speed * (dest - start height) / step: the second step is two steps from the start.
    assert (first.speed, second.speed) == (F, 2 * F)
    assert (first.floordestheight, second.floordestheight) == (0, -16 * F)


def test_stairs_reset_after_delay() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 27, (1, 16, 16, 0, 12))
    step = sec.special_data
    assert (step.reset_delay, step.reset_delay_count, step.reset_height) == (12, 12, 16 * F)
    _run(sim, 9)  # 8 tics of travel, then it reaches the top and waits
    assert sec.floor_height == 32 * F and sec.special_data is step
    _run(sim, 2)
    assert sec.floor_height == 32 * F
    _run(sim, 1)  # tic 12: the reset turns it around and it moves the same tic
    assert (step.direction, step.floordestheight) == (-1, 16 * F)
    assert sec.floor_height == 30 * F
    _run(sim, 10)
    assert sec.floor_height == 16 * F
    assert sec.special_data is None and step.removed


def test_stairs_pause_at_each_step_height() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 27, (1, 16, 8, 5, 0))
    step = sec.special_data
    assert (step.delay_total, step.stairs_delay_height) == (5, 24 * F)
    _run(sim, 4)
    assert sec.floor_height == 24 * F and step.delay_count == 5
    _run(sim, 5)
    assert step.delay_count == 0 and not step.removed
    _run(sim, 1)
    assert step.removed


# --- EV_DoCeiling --------------------------------------------------------------------


def test_ceiling_crush_and_raise_cycles_and_changes_speed() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 42, (1, 16, 5, 0, 0))
    ceiling = sec.special_data
    assert sim.active_ceilings[0] is ceiling
    assert (ceiling.topheight, ceiling.bottomheight) == (128 * F, 24 * F)  # floor + 8
    assert (ceiling.direction, ceiling.crush, ceiling.speed) == (-1, 5, 2 * F)
    _run(sim, 53)  # 104 units at 2 per tic, then one tic to notice
    assert sec.ceiling_height == 24 * F
    assert (ceiling.direction, ceiling.speed) == (1, F)  # up at half speed
    _run(sim, 105)
    assert sec.ceiling_height == 128 * F
    assert (ceiling.direction, ceiling.speed) == (-1, 2 * F)  # and down again at full
    assert not ceiling.removed


def test_ceiling_crush_stop_matches_tag() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 42, (1, 16, 5, 0, 0))
    ceiling = sec.special_data
    assert not execute_line_special(sim, 44, (7, 0, 0, 0, 0))
    assert execute_line_special(sim, 44, (1, 0, 0, 0, 0))
    assert ceiling.removed and sec.special_data is None
    assert sim.active_ceilings == [None] * 30
    assert not execute_line_special(sim, 44, (1, 0, 0, 0, 0))


def test_ceiling_crush_raise_and_stay_ends_at_the_top() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 45, (1, 16, 5, 0, 0))
    ceiling = sec.special_data
    _run(sim, 53)
    assert (ceiling.direction, ceiling.speed) == (1, F)
    _run(sim, 105)
    assert sec.ceiling_height == 128 * F
    assert ceiling.removed and sec.special_data is None
    assert sim.active_ceilings == [None] * 30


def test_ceiling_lower_and_crush_stops_above_the_floor() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 43, (1, 16, 7, 0, 0))
    assert sec.special_data.crush == 7
    _run(sim, 60)
    assert sec.ceiling_height == 24 * F
    assert sec.special_data is None


def test_ceiling_move_to_value() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 69, (1, 16, 8, 0, 0))
    _run(sim, 40)
    assert sec.ceiling_height == 64 * F and sec.special_data is None
    # Already there: vanilla reports failure, but the thinker exists for one tic.
    assert not execute_line_special(sim, 69, (1, 16, 8, 0, 0))
    assert isinstance(sec.special_data, CeilingMove)
    _run(sim, 1)
    assert sec.special_data is None


# --- EV_DoPlat -----------------------------------------------------------------------


def test_plat_perpetual_raise_takes_its_start_direction_from_p_random() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    state = sim.rng.get_state()
    expected = sim.rng.p_random() & 1
    after_one_call = sim.rng.get_state()
    sim.rng.set_state(state)

    assert execute_line_special(sim, 60, (1, 16, 5, 0, 0))
    plat = sec.special_data
    assert sim.rng.get_state() == after_one_call
    assert plat.status == expected
    # lowest neighbour floor + 8, and no neighbour is higher than the plat itself
    assert (plat.low, plat.high, plat.wait, plat.tag) == (8 * F, 16 * F, 5, 1)
    _run(sim, 200)
    assert not plat.removed  # it never stops by itself
    assert sim.active_plats[0] is plat


def test_plat_up_wait_down_stay() -> None:
    sim = _sim()
    sec0, _ = _sectors(sim)
    sec0.tag = 2
    assert execute_line_special(sim, 64, (2, 16, 5, 0, 0))
    plat = sec0.special_data
    assert (plat.low, plat.high, plat.status) == (0, 16 * F, PlatStatus.PLAT_UP)
    _run(sim, 9)
    assert sec0.floor_height == 16 * F
    assert (plat.status, plat.count) == (PlatStatus.PLAT_WAITING, 5)
    _run(sim, 5)
    assert plat.status == PlatStatus.PLAT_DOWN
    _run(sim, 9)
    assert sec0.floor_height == 0
    assert plat.removed and sec0.special_data is None
    assert sim.active_plats == [None] * 30


@pytest.mark.parametrize(
    ("special", "args", "low", "high", "status"),
    [
        (62, (1, 16, 5, 0, 0), 8, 16, PlatStatus.PLAT_DOWN),  # lowest neighbour + 8
        (63, (1, 16, 5, 1, 0), 8, 16, PlatStatus.PLAT_DOWN),  # down 1 * 8
        (65, (1, 16, 5, 2, 0), 16, 32, PlatStatus.PLAT_UP),  # up 2 * 8
    ],
)
def test_plat_ranges(
    special: int, args: tuple[int, ...], low: int, high: int, status: PlatStatus
) -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, special, args)
    plat = sec.special_data
    assert isinstance(plat, Plat)
    assert (plat.low, plat.high, plat.status) == (low * F, high * F, status)
    _run(sim, 40)
    assert sec.floor_height == 16 * F  # back where it started
    assert plat.removed and sec.special_data is None


def test_plat_stop_removes_the_first_active_plat_whatever_its_tag() -> None:
    """EV_StopPlat does not search for the tag; it overwrites the first plat's tag."""
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 62, (1, 16, 5, 0, 0))
    plat = sec.special_data
    assert execute_line_special(sim, 61, (9, 0, 0, 0, 0)) is False  # never a success
    assert plat.tag == 9
    assert plat.removed and sec.special_data is None
    assert sim.active_plats == [None] * 30
    # Nothing left: vanilla would dereference a null pointer, we do nothing.
    assert execute_line_special(sim, 61, (9, 0, 0, 0, 0)) is False


def test_plat_stop_with_tag_zero_only_clears_tags() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 62, (1, 16, 5, 0, 0))
    plat = sec.special_data
    execute_line_special(sim, 61, (0, 0, 0, 0, 0))
    assert plat.tag == 0 and not plat.removed
    assert sec.special_data is plat


# --- Pillars -------------------------------------------------------------------------


def test_pillar_build_meets_in_the_middle_and_opens_again() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert not execute_line_special(sim, 30, (1, 16, 0, 0, 0))  # not closed: cannot open
    assert execute_line_special(sim, 29, (1, 16, 0, 9, 0))
    pillar = sec.special_data
    assert isinstance(pillar, Pillar)
    assert (pillar.floor_speed, pillar.ceiling_speed, pillar.crush) == (2 * F, 2 * F, 0)
    _run(sim, 30)
    assert sec.floor_height == sec.ceiling_height == 72 * F
    assert sec.special_data is None

    assert not execute_line_special(sim, 29, (1, 16, 0, 0, 0))  # already closed
    assert execute_line_special(sim, 30, (1, 16, 0, 0, 0))
    pillar = sec.special_data
    # To the lowest neighbour floor and the highest neighbour ceiling; the plane with
    # less to travel (the ceiling, 56 against 72) is slowed so both arrive together.
    assert (pillar.floordest, pillar.ceilingdest, pillar.direction) == (0, 128 * F, -1)
    assert pillar.floor_speed == 2 * F
    # FixedMul(-56.0, FixedDiv(2.0, -72.0)): about 2 * 56 / 72, with vanilla's rounding.
    assert pillar.ceiling_speed == 101920
    _run(sim, 40)
    assert (sec.floor_height, sec.ceiling_height) == (0, 128 * F)
    assert sec.special_data is None


def test_pillar_build_with_height_scales_the_slower_plane() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    assert execute_line_special(sim, 94, (1, 16, 16, 9, 0))  # Pillar_BuildAndCrush
    pillar = sec.special_data
    assert pillar.crush == 9
    assert (pillar.floordest, pillar.ceilingdest) == (32 * F, 32 * F)
    assert pillar.ceiling_speed == 2 * F  # the ceiling has further to go (96 against 16)
    assert 0 < pillar.floor_speed < pillar.ceiling_speed
    _run(sim, 60)
    assert sec.floor_height == sec.ceiling_height == 32 * F


def test_pillar_open_by_value() -> None:
    sim = _sim()
    _, sec = _sectors(sim)
    sec.ceiling_height = sec.floor_height
    assert execute_line_special(sim, 30, (1, 16, 8, 24, 0))
    pillar = sec.special_data
    assert (pillar.floordest, pillar.ceilingdest) == (8 * F, 40 * F)
    _run(sim, 20)
    assert (sec.floor_height, sec.ceiling_height) == (8 * F, 40 * F)


# --- p_things.c ----------------------------------------------------------------------


def _spot(sim: GameSim) -> Mobj:
    return sim.find_mobjs_by_tid(SPOT_TID)[0]


def test_thing_spawn_with_and_without_fog() -> None:
    sim = _sim()
    spot = _spot(sim)
    assert execute_line_special(sim, 135, (SPOT_TID, T_ETTIN, 64, 0, 0))
    (ettin,) = _mobjs(sim, MobjType.MT_ETTIN)
    assert (ettin.x, ettin.y, ettin.z) == (spot.x, spot.y, spot.z)
    assert ettin.angle == 64 << 24
    assert ettin.flags2 & MobjFlag2.MF2_DROPPED
    assert len(_mobjs(sim, MobjType.MT_TFOG)) == 1

    # The spot is now occupied: the second ettin does not fit and is removed.
    assert not execute_line_special(sim, 137, (SPOT_TID, T_ETTIN, 0, 0, 0))
    assert len(_mobjs(sim, MobjType.MT_ETTIN)) == 1
    assert len(_mobjs(sim, MobjType.MT_TFOG)) == 1


def test_thing_spawn_no_fog() -> None:
    sim = _sim()
    assert execute_line_special(sim, 137, (SPOT_TID, T_ETTIN, 0, 0, 0))
    assert len(_mobjs(sim, MobjType.MT_ETTIN)) == 1
    assert _mobjs(sim, MobjType.MT_TFOG) == []


def test_thing_spawn_respects_nomonsters_and_missing_tid() -> None:
    sim = _sim(nomonsters=True)
    assert not execute_line_special(sim, 135, (SPOT_TID, T_ETTIN, 0, 0, 0))
    assert _mobjs(sim, MobjType.MT_ETTIN) == []
    assert not execute_line_special(_sim(), 135, (99, T_ETTIN, 0, 0, 0))


@pytest.mark.parametrize("special", [134, 136])
def test_thing_projectile(special: int) -> None:
    sim = _sim()
    spot = _spot(sim)
    # Angle 128 (west, into the open room), speed 16 -> 2 units per tic.
    assert execute_line_special(sim, special, (SPOT_TID, T_FIREBALL1, 128, 16, 8))
    (missile,) = _mobjs(sim, MobjType.MT_FIREBALL1)
    assert missile.target is spot
    assert missile.angle == 128 << 24
    # FixedMul(2.0, finecosine[4096]) and FixedMul(2.0, finesine[4096]): the tables give
    # -65535 and -25 for due west, not -1.0 and 0.
    assert (missile.momx, missile.momy, missile.momz) == (-131070, -50, F)
    assert missile.flags2 & MobjFlag2.MF2_DROPPED
    # P_CheckMissileSpawn moved it half a step.
    assert (missile.x, missile.y) == (spot.x - 65535, spot.y - 25)
    gravity = special == 136
    assert bool(missile.flags & MobjFlag.MF_NOGRAVITY) is not gravity
    assert bool(missile.flags2 & MobjFlag2.MF2_LOGRAV) is gravity


def _ettin_with_tid(sim: GameSim, tid: int = 30) -> Mobj:
    spot = _spot(sim)
    ettin = spawn_mobj(sim, spot.x, spot.y, spot.z, MobjType.MT_ETTIN)
    insert_mobj_into_tid_list(sim, ettin, tid)
    return ettin


def test_thing_deactivate_and_activate_monster() -> None:
    sim = _sim()
    ettin = _ettin_with_tid(sim)
    assert execute_line_special(sim, 131, (30, 0, 0, 0, 0))
    assert ettin.flags2 & MobjFlag2.MF2_DORMANT and ettin.tics == -1
    assert not execute_line_special(sim, 131, (30, 0, 0, 0, 0))  # already dormant
    assert execute_line_special(sim, 130, (30, 0, 0, 0, 0))
    assert not (ettin.flags2 & MobjFlag2.MF2_DORMANT) and ettin.tics == 1
    assert not execute_line_special(sim, 130, (30, 0, 0, 0, 0))  # already awake
    # A map spot is neither a monster nor one of the switchable decorations.
    assert not execute_line_special(sim, 130, (SPOT_TID, 0, 0, 0, 0))
    assert not execute_line_special(sim, 131, (SPOT_TID, 0, 0, 0, 0))


def test_thing_destroy_only_hits_shootable_things() -> None:
    sim = _sim()
    ettin = _ettin_with_tid(sim)
    assert execute_line_special(sim, 133, (30, 0, 0, 0, 0))
    assert ettin.health <= 0
    assert not execute_line_special(sim, 133, (SPOT_TID, 0, 0, 0, 0))


def test_thing_remove() -> None:
    sim = _sim()
    ettin = _ettin_with_tid(sim)
    assert execute_line_special(sim, 132, (30, 0, 0, 0, 0))
    assert ettin.removed
    assert sim.find_mobjs_by_tid(30) == []
    assert not execute_line_special(sim, 132, (30, 0, 0, 0, 0))


def test_radius_quake_spawns_a_focus_on_each_spot() -> None:
    sim = _sim()
    spot = _spot(sim)
    assert execute_line_special(sim, 120, (5, 20, 1, 2, SPOT_TID))
    (focus,) = _mobjs(sim, MobjType.MT_QUAKE_FOCUS)
    assert (focus.x, focus.y) == (spot.x, spot.y)
    assert tuple(focus.args) == (5, 10, 1, 2, SPOT_TID)  # duration is halved
    assert not execute_line_special(sim, 120, (5, 20, 1, 2, 99))


def test_damage_thing_goes_through_damage_mobj() -> None:
    sim = _sim()
    player = spawn_player(sim, player_num=0)
    assert player.mo is not None
    before = player.mo.health
    assert execute_line_special(sim, 73, (25, 0, 0, 0, 0), mo=player.mo)
    assert 0 < player.mo.health < before
    assert player.health == player.mo.health
    # Zero damage guarantees a kill.
    assert execute_line_special(sim, 73, (0, 0, 0, 0, 0), mo=player.mo)
    assert player.mo.health <= 0


def test_use_puzzle_item_needs_a_player_with_a_matching_item() -> None:
    sim = _sim()
    assert sim.level is not None
    line = sim.level.lines[0]
    line.args = (3, 0, 0, 0, 0)
    assert not execute_line_special(sim, 129, (3, 0, 0, 0, 0), line=line, mo=None)
    assert not execute_line_special(sim, 129, (3, 0, 0, 0, 0), line=line, mo=_spot(sim))
    player = spawn_player(sim, player_num=0)
    assert not execute_line_special(sim, 129, (3, 0, 0, 0, 0), line=line, mo=player.mo)
    assert not execute_line_special(sim, 129, (3, 0, 0, 0, 0), line=None, mo=player.mo)
