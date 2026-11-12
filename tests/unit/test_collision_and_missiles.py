# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Collision and missile details found while syncing DEMO3 (T15.6c).

Each test pins one vanilla behaviour from p_map.c, p_mobj.c or a_action.c that the
earlier ports got wrong.
"""

from __future__ import annotations

import pytest

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim import movement
from pyhexen.sim.actions.environment import a_fogspawn
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjFlag, MobjFlag2, MobjType
from pyhexen.sim.map import check_position
from pyhexen.sim.mobj import Mobj, spawn_mobj
from pyhexen.sim.movement import p_blaster_mobj_thinker, p_xy_movement
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level

F = FRACUNIT


def _sim(floor_pic: str = "FLOOR01") -> GameSim:
    """One closed 512x512 room, floor 0, ceiling 256."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(
        x=0, y=0, width=512, height=512, floor_height=0, ceiling_height=256, floor_pic=floor_pic
    )
    mb.add_thing(tid=0, x=64, y=64, z=0, angle=0, type=1)
    wb.add_map(mb)
    return GameSim(level=load_level(WadStack(wb.to_bytes()), "MAP01"))


def _things(sim: GameSim, mobj_type: MobjType) -> list[Mobj]:
    return [t for t in sim.thinkers if isinstance(t, Mobj) and t.type == mobj_type]


# --- A_FogSpawn (a_action.c) ----------------------------------------------------------


def test_fog_spawner_tests_its_counter_before_decrementing() -> None:
    """`if (actor->special1.i-- > 0) return;`: a counter of 1 waits one more call."""
    sim = _sim()
    spawner = spawn_mobj(sim, 256 * F, 256 * F, 64 * F, MobjType.MT_FOGSPAWNER)
    spawner.args = (0, 0, 3, 10, 0)  # speed 0 is raised to 1, frequency 3, lifetime 10
    spawner.special1 = 1

    a_fogspawn(spawner, sim)
    assert spawner.special1 == 0
    assert _fog(sim) == []

    a_fogspawn(spawner, sim)
    (fog,) = _fog(sim)
    assert spawner.special1 == 3  # reset to the frequency
    assert spawner.args[0] == 1  # vanilla writes the raised speed back
    assert fog.args[0] == 1 and fog.args[3] == 10 and fog.args[4] == 1


def _fog(sim: GameSim) -> list[Mobj]:
    fog_types = (MobjType.MT_FOGPATCHS, MobjType.MT_FOGPATCHM, MobjType.MT_FOGPATCHL)
    return [t for t in sim.thinkers if isinstance(t, Mobj) and t.type in fog_types]


# --- P_BlasterMobjThinker (p_mobj.c) --------------------------------------------------


def test_blaster_missile_splashes_when_it_hits_a_liquid_floor() -> None:
    sim = _sim(floor_pic="X_009")  # sludge
    missile = spawn_mobj(sim, 256 * F, 256 * F, F, MobjType.MT_MWAND_MISSILE)
    missile.momz = -16 * F  # 2 units per sub-step: under the floor at the first one

    p_blaster_mobj_thinker(sim, missile)

    assert missile.z == 0
    assert not (missile.flags & MobjFlag.MF_MISSILE)  # exploded
    assert len(_things(sim, MobjType.MT_SLUDGECHUNK)) == 1  # P_HitFloor's big splash
    assert len(_things(sim, MobjType.MT_SLUDGESPLASH)) == 1


def test_blaster_flame_trail_counts_down_to_exactly_zero() -> None:
    sim = _sim()
    missile = spawn_mobj(sim, 64 * F, 256 * F, 32 * F, MobjType.MT_CFLAME_MISSILE)
    missile.momx = 16 * F  # 2 units per sub-step, 16 in all
    missile.special1 = 2

    p_blaster_mobj_thinker(sim, missile)

    # Counts 1, 0 (flame, reset to 4), 3, 2, 1, 0 (flame, reset to 4), 3, 2.
    assert len(_things(sim, MobjType.MT_CFLAMEFLOOR)) == 2
    assert missile.special1 == 2


def test_wand_missile_whose_roll_fails_runs_the_flame_counter() -> None:
    """`else if (!--mobj->special1.i)` also runs for MT_MWAND_MISSILE (a vanilla quirk)."""
    sim = _sim()
    missile = spawn_mobj(sim, 64 * F, 256 * F, 32 * F, MobjType.MT_MWAND_MISSILE)
    missile.momx = 16 * F
    missile.special1 = 0
    smoke_before = len(_things(sim, MobjType.MT_MWANDSMOKE))

    p_blaster_mobj_thinker(sim, missile)

    smoke = len(_things(sim, MobjType.MT_MWANDSMOKE)) - smoke_before
    assert missile.special1 == -(8 - smoke)  # one decrement per failed roll, never 0
    assert _things(sim, MobjType.MT_CFLAMEFLOOR) == []


# --- P_XYMovement (p_mobj.c) ----------------------------------------------------------


@pytest.mark.parametrize(("momx", "steps"), [(20 * F, 2), (-20 * F, 1), (-30 * F, 1)])
def test_only_positive_fast_moves_are_split(
    monkeypatch: pytest.MonkeyPatch, momx: int, steps: int
) -> None:
    sim = _sim()
    mo = spawn_mobj(sim, 256 * F, 256 * F, 0, MobjType.MT_ETTIN)
    mo.momx = momx
    tries: list[int] = []
    real_try_move = movement.try_move

    def counting_try_move(sim_: GameSim, thing: Mobj, x: int, y: int) -> bool:
        tries.append(x)
        return real_try_move(sim_, thing, x, y)

    monkeypatch.setattr(movement, "try_move", counting_try_move)
    p_xy_movement(sim, mo)
    assert len(tries) == steps
    assert mo.x == 256 * F + momx


def test_blasted_thing_is_hurt_twice_by_a_wall() -> None:
    """PIT_CheckLine and P_TryMove's pushline both deal mass >> 5 to a blasted thing."""
    sim = _sim()
    ettin = spawn_mobj(sim, 26 * F, 256 * F, 0, MobjType.MT_ETTIN)
    ettin.flags2 |= MobjFlag2.MF2_BLASTED
    ettin.momx = -8 * F

    p_xy_movement(sim, ettin)

    assert ettin.health == 175 - 2 * (175 >> 5)
    assert ettin.x == 26 * F and ettin.momx == 0


def test_floor_bounce_missile_bounces_off_a_wall() -> None:
    sim = _sim()
    bomb = spawn_mobj(sim, 20 * F, 256 * F, 32 * F, MobjType.MT_THROWINGBOMB)
    bomb.momx = -16 * F

    p_xy_movement(sim, bomb)

    assert not bomb.removed and bomb.flags & MobjFlag.MF_MISSILE  # did not explode
    assert bomb.momx > 0  # heading back into the room
    # Three quarters of the speed (P_BounceWall friction): FixedMul(12.0, finecosine[0]),
    # and finecosine[0] is 65535, not 1.0.
    assert bomb.momx == 12 * 65535


# --- P_CheckPosition / PIT_CheckThing (p_map.c) ---------------------------------------


def test_thing_that_did_not_block_is_not_the_blocker_when_a_wall_does() -> None:
    sim = _sim()
    ettin = spawn_mobj(sim, 40 * F, 256 * F, 0, MobjType.MT_ETTIN)
    # A pickup at the destination: touched (BlockingMobj set) but it does not block.
    spawn_mobj(sim, 20 * F, 256 * F, 0, MobjType.MT_HEALINGBOTTLE)

    assert not check_position(sim, ettin, 20 * F, 256 * F)  # the wall blocks
    assert sim.map_state.blocking_mobj is None


def test_missile_passes_through_a_nonshootable_solid_thing() -> None:
    """A submerged serpent is solid but MF2_NONSHOOTABLE: vanilla checks that first."""
    sim = _sim()
    serpent = spawn_mobj(sim, 256 * F, 256 * F, 0, MobjType.MT_SERPENT)
    assert serpent.flags & MobjFlag.MF_SOLID and not (serpent.flags & MobjFlag.MF_SHOOTABLE)
    assert serpent.flags2 & MobjFlag2.MF2_NONSHOOTABLE
    missile = spawn_mobj(sim, 200 * F, 256 * F, 16 * F, MobjType.MT_MWAND_MISSILE)

    assert check_position(sim, missile, 256 * F, 256 * F)

    serpent.flags2 &= ~MobjFlag2.MF2_NONSHOOTABLE  # surfaced, but still not shootable
    assert not check_position(sim, missile, 256 * F, 256 * F)
