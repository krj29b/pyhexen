# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Z movement and standing on things (T15.6e).

Covers P_MobjThinker's BlockingMobj / P_CheckOnmobj branch, P_FakeZMovement, and the
parts of P_ZMovement the earlier port left out (landing, floor bounce, crash state).
"""

from __future__ import annotations

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.fixed import FRACUNIT, fixed_mul
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjFlag, MobjFlag2, MobjType
from pyhexen.sim.map import check_on_mobj
from pyhexen.sim.mobj import Mobj, spawn_mobj
from pyhexen.sim.movement import (
    BOUNCE_NORMAL,
    GRAVITY,
    p_fake_z_movement,
    p_mobj_thinker,
    p_z_movement,
)
from pyhexen.sim.player import VIEWHEIGHT, Player, spawn_player
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level

F = FRACUNIT


def _sim() -> GameSim:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512, floor_height=0, ceiling_height=256)
    mb.add_thing(tid=0, x=128, y=256, z=0, angle=0, type=1)
    wb.add_map(mb)
    return GameSim(level=load_level(WadStack(wb.to_bytes()), "MAP01"))


def _player_next_to(sim: GameSim, thing_height: int) -> tuple[Player, Mobj]:
    """A player on the floor whose next step east overlaps a solid thing of `thing_height`."""
    player = spawn_player(sim, player_num=0)
    mo = player.mo
    assert mo is not None and mo.flags2 & MobjFlag2.MF2_PASSMOBJ
    assert mo.z == mo.floorz == 0
    # Radii are 16 and 25: at 41.5 apart they do not touch, one unit closer they do.
    thing = spawn_mobj(sim, mo.x + 41 * F + F // 2, mo.y, 0, MobjType.MT_ETTIN)
    thing.height = thing_height
    mo.momx = F
    return player, thing


def test_player_steps_up_onto_a_low_thing() -> None:
    sim = _sim()
    player, thing = _player_next_to(sim, 17 * F)
    mo = player.mo
    assert mo is not None
    start_x = mo.x

    p_mobj_thinker(sim, mo)

    assert mo.x == start_x + F  # the move went through
    assert mo.z == 17 * F  # on top of the thing, though z == floorz and momz == 0 before
    assert mo.flags2 & MobjFlag2.MF2_ONMOBJ
    assert mo.momz == 0
    # The view sinks by the step height and then recovers an eighth per tic.
    assert player.viewheight == VIEWHEIGHT - 17 * F
    assert player.deltaviewheight == (17 * F) >> 3


def test_player_is_blocked_by_a_tall_thing() -> None:
    sim = _sim()
    player, _ = _player_next_to(sim, 25 * F)  # one unit more than a step
    mo = player.mo
    assert mo is not None
    start_x = mo.x

    p_mobj_thinker(sim, mo)

    assert mo.x == start_x
    assert mo.z == 0
    assert not (mo.flags2 & MobjFlag2.MF2_ONMOBJ)
    assert player.viewheight == VIEWHEIGHT


def test_player_walking_off_a_thing_loses_onmobj_and_falls() -> None:
    sim = _sim()
    player, thing = _player_next_to(sim, 17 * F)
    mo = player.mo
    assert mo is not None
    p_mobj_thinker(sim, mo)
    assert mo.flags2 & MobjFlag2.MF2_ONMOBJ

    thing.flags &= ~MobjFlag.MF_SOLID  # nothing to stand on any more
    mo.momx = 0
    p_mobj_thinker(sim, mo)
    assert not (mo.flags2 & MobjFlag2.MF2_ONMOBJ)
    assert mo.momz == -2 * GRAVITY  # gravity took over


def test_non_player_on_a_thing_is_left_alone() -> None:
    """Vanilla only acts on players in the on-mobj branch: others just skip P_ZMovement."""
    sim = _sim()
    base = spawn_mobj(sim, 300 * F, 300 * F, 0, MobjType.MT_ETTIN)
    pot = spawn_mobj(sim, 300 * F, 300 * F, base.height, MobjType.MT_POTTERY1)
    assert pot.flags2 & MobjFlag2.MF2_PASSMOBJ
    pot.momz = -3 * F

    p_mobj_thinker(sim, pot)

    assert (pot.z, pot.momz) == (base.height, -3 * F)
    assert not (pot.flags2 & MobjFlag2.MF2_ONMOBJ)


def test_check_on_mobj_undoes_the_fake_z_move() -> None:
    sim = _sim()
    mo = spawn_mobj(sim, 128 * F, 128 * F, 64 * F, MobjType.MT_PLAYER_FIGHTER)
    mo.momz = -5 * F

    assert check_on_mobj(sim, mo) is None
    assert (mo.z, mo.momz) == (64 * F, -5 * F)

    p_fake_z_movement(sim, mo)
    assert (mo.z, mo.momz) == (59 * F, -6 * F)  # moved, then gravity
    mo.z = F
    p_fake_z_movement(sim, mo)
    assert (mo.z, mo.momz) == (0, 0)  # clipped to the floor


def test_hard_landing_squats_and_delays_jumping() -> None:
    sim = _sim()
    player = spawn_player(sim, player_num=0)
    mo = player.mo
    assert mo is not None
    mo.z = 4 * F
    mo.momz = -10 * F

    p_z_movement(sim, mo)

    assert (mo.z, mo.momz) == (0, 0)
    assert player.jump_tics == 7
    assert player.deltaviewheight == (-10 * F) >> 3
    assert player.centering is True


def test_soft_landing_only_delays_jumping() -> None:
    sim = _sim()
    player = spawn_player(sim, player_num=0)
    mo = player.mo
    assert mo is not None
    mo.z = F
    mo.momz = -2 * F  # not below -8 * GRAVITY
    player.deltaviewheight = 0

    p_z_movement(sim, mo)

    assert (mo.z, mo.momz, player.jump_tics) == (0, 0, 7)
    assert player.deltaviewheight == 0
    assert player.centering is False


def test_floor_bounce_missile_keeps_most_of_its_speed() -> None:
    sim = _sim()
    bomb = spawn_mobj(sim, 128 * F, 128 * F, F, MobjType.MT_THROWINGBOMB)
    assert bomb.flags & MobjFlag.MF_MISSILE and bomb.flags2 & MobjFlag2.MF2_FLOORBOUNCE
    bomb.momx, bomb.momy, bomb.momz = 3 * F, -6 * F, -4 * F

    p_z_movement(sim, bomb)

    assert not bomb.removed and bomb.flags & MobjFlag.MF_MISSILE  # it did not explode
    assert bomb.z == 0
    assert bomb.momz == fixed_mul(-4 * F, BOUNCE_NORMAL) > 0  # about 0.7 of the speed, upward
    assert (bomb.momx, bomb.momy) == (2 * F, -4 * F)  # two thirds


def test_blasted_flag_resets_when_momentum_is_gone() -> None:
    sim = _sim()
    mo = spawn_mobj(sim, 128 * F, 128 * F, 0, MobjType.MT_ETTIN)
    mo.tics = -1
    mo.flags2 |= MobjFlag2.MF2_BLASTED | MobjFlag2.MF2_SLIDE

    p_mobj_thinker(sim, mo)

    assert not (mo.flags2 & (MobjFlag2.MF2_BLASTED | MobjFlag2.MF2_SLIDE))
