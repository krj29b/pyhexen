# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for Task T10.8: Cleric weapons."""

from __future__ import annotations

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.sim.actions import is_action_implemented
from pyhexen.sim.actions.weapons_cleric import (
    STARTHOLYPAL,
    a_cflameattack,
    a_cflamemissile,
    a_cflamepuff,
    a_cflamerotate,
    a_cholyattack,
    a_cholypalette,
    a_cmaceattack,
    a_cstaffattack,
    a_cstaffcheck,
    a_cstaffcheckblink,
    a_cstaffinitblink,
    a_cstaffmissileslither,
)
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjFlag2, MobjType, StateNum
from pyhexen.sim.mobj import ONFLOORZ, Mobj, spawn_mobj
from pyhexen.sim.movement import p_blaster_mobj_thinker
from pyhexen.sim.player import PlayerClass, spawn_player
from pyhexen.sim.psprites import (
    MANA_1,
    MANA_2,
    PS_WEAPON,
    WP_FIRST,
    WP_FOURTH,
    WP_SECOND,
    WP_THIRD,
    p_set_psprite,
)
from pyhexen.wad.wadfile import WadFile, WadStack
from pyhexen.world.level import load_level


@pytest.fixture
def sim() -> GameSim:
    """Create a synthetic GameSim box room with a Cleric player."""
    wb = WadBuilder()
    mb = MapBuilder("MAP01")
    mb.add_room(-512, -512, 1024, 1024, floor_height=0, ceiling_height=256)
    wb.add_map(mb)
    wad = WadFile.from_bytes(wb.to_bytes())
    stack = WadStack(wad)
    level = load_level(stack, "MAP01")
    return GameSim(level=level, player_class=int(PlayerClass.PCLASS_CLERIC), rng=Random(42))


def test_cleric_actions_registered() -> None:
    """Verify all Cleric weapon actions are registered."""
    for act_name in (
        "A_CMaceAttack",
        "A_CStaffCheck",
        "A_CStaffAttack",
        "A_CStaffMissileSlither",
        "A_CStaffInitBlink",
        "A_CStaffCheckBlink",
        "A_CFlameAttack",
        "A_CFlamePuff",
        "A_CFlameMissile",
        "A_CFlameRotate",
        "A_CHolyAttack",
        "A_CHolyPalette",
    ):
        assert is_action_implemented(act_name), f"{act_name} should be implemented"


def test_mace_attack_miss_and_hit(sim: GameSim) -> None:
    """Test Mace of Contrition swinging at empty air vs hitting a monster."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_FIRST
    psp = player.psprites[PS_WEAPON]

    # Miss in air
    a_cmaceattack(sim, player, psp)

    # Monster in melee range
    target = spawn_mobj(sim, 60 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_ETTIN)
    initial_health = target.health
    a_cmaceattack(sim, player, psp)
    assert target.health < initial_health
    assert 25 <= (initial_health - target.health) <= 40


def test_serpent_staff_melee_drain(sim: GameSim) -> None:
    """Test Serpent Staff life drain when an enemy is within melee range."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_SECOND
    player.mana[MANA_1] = 20
    player.health = 50
    player.mo.health = 50
    psp = player.psprites[PS_WEAPON]

    # Monster in melee drain range (50 units)
    target = spawn_mobj(sim, 50 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_ETTIN)
    initial_target_hp = target.health

    a_cstaffcheck(sim, player, psp)

    assert target.health < initial_target_hp
    damage = initial_target_hp - target.health
    expected_gain = damage >> 3
    assert player.health == 50 + expected_gain
    assert player.mo.health == player.health
    assert player.mana[MANA_1] == 19
    assert psp.state_num == int(StateNum.S_CSTAFFATK2_1)

    # Health capped at 100
    player.health = 98
    player.mo.health = 98
    a_cstaffcheck(sim, player, psp)
    assert player.health <= 100


def test_serpent_staff_ranged_missiles_and_slither(sim: GameSim) -> None:
    """Test Serpent Staff missile fire and sinusoidal slither movement."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_SECOND
    player.mana[MANA_1] = 10
    psp = player.psprites[PS_WEAPON]

    # No monster in melee range -> fire dual missiles
    a_cstaffattack(sim, player, psp)
    assert player.mana[MANA_1] == 9

    missiles = [
        th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_CSTAFF_MISSILE
    ]
    assert len(missiles) == 2
    specials = {m.special2 for m in missiles}
    assert specials == {0, 32}

    # Test slither
    m0 = missiles[0]
    initial_special2 = m0.special2
    a_cstaffmissileslither(sim, m0)
    assert m0.special2 == (initial_special2 + 3) & 63


def test_serpent_staff_blink(sim: GameSim) -> None:
    """Test Serpent Staff eye blink initialization and countdown."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    psp = player.psprites[PS_WEAPON]

    a_cstaffinitblink(sim, player, psp)
    assert isinstance(player.mo.special1, int)
    assert player.mo.special1 >= 20

    # Countdown to 1 tic left
    player.mo.special1 = 1
    a_cstaffcheckblink(sim, player, psp)
    assert psp.state_num == int(StateNum.S_CSTAFFBLINK1)
    assert player.mo.special1 > 0


def test_flame_attack_and_blaster_thinker(sim: GameSim) -> None:
    """Test Firestorm flame missile spawn and sub-step blaster thinker."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_THIRD
    player.mana[MANA_2] = 20
    psp = player.psprites[PS_WEAPON]

    a_cflameattack(sim, player, psp)
    assert player.mana[MANA_2] == 16

    flame_missiles = [
        th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_CFLAME_MISSILE
    ]
    assert len(flame_missiles) == 1
    fmo = flame_missiles[0]
    assert fmo.thinker_func == p_blaster_mobj_thinker
    assert fmo.special1 == 2

    # Tick the blaster thinker: special1 decrements; on reaching 0 spawns floor flames
    p_blaster_mobj_thinker(sim, fmo)
    p_blaster_mobj_thinker(sim, fmo)
    floor_flames = [
        th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_CFLAMEFLOOR
    ]
    assert len(floor_flames) >= 1


def test_flame_puff_and_missile_circle(sim: GameSim) -> None:
    """Test Firestorm impact creating orbiting flame circle around shootable victim."""
    target = spawn_mobj(sim, 100 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_ETTIN)
    missile = spawn_mobj(sim, 100 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_CFLAME_MISSILE)
    missile.flags2 |= MobjFlag2.MF2_DONTDRAW

    # Test flame puff
    a_cflamepuff(sim, missile)
    assert not (missile.flags2 & MobjFlag2.MF2_DONTDRAW)
    assert missile.momx == 0 and missile.momy == 0 and missile.momz == 0

    # Test flame circle spawn on impact with blocking shootable actor
    sim.map_state.blocking_mobj = target
    a_cflamemissile(sim, missile)
    assert missile.state_num == int(StateNum.S_FLAMEPUFF2_1)

    circle_flames = [
        th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_CIRCLEFLAME
    ]
    assert len(circle_flames) == 8

    # Test orbiting flame rotation
    c0 = circle_flames[0]
    orig_angle = c0.angle
    a_cflamerotate(sim, c0)
    assert c0.angle != orig_angle
    assert c0.momx != 0 or c0.momy != 0


def test_holy_attack_and_palette(sim: GameSim) -> None:
    """Test Wraithverge holy attack mana deduction, projectile, and palette fade."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_FOURTH
    player.mana[MANA_1] = 50
    player.mana[MANA_2] = 50
    psp = player.psprites[PS_WEAPON]

    a_cholyattack(sim, player, psp)
    assert player.mana[MANA_1] == 32
    assert player.mana[MANA_2] == 32
    assert player.palette == STARTHOLYPAL
    assert player.damagecount == 0
    assert player.bonuscount == 0

    holy_missiles = [
        th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_HOLY_MISSILE
    ]
    assert len(holy_missiles) == 1

    # Test palette fade across states S_CHOLYATK_6..9
    p_set_psprite(sim, player, PS_WEAPON, StateNum.S_CHOLYATK_6)
    a_cholypalette(sim, player, psp)
    assert player.palette == STARTHOLYPAL  # 22

    p_set_psprite(sim, player, PS_WEAPON, StateNum.S_CHOLYATK_7)
    a_cholypalette(sim, player, psp)
    assert player.palette == STARTHOLYPAL + 1  # 23

    p_set_psprite(sim, player, PS_WEAPON, StateNum.S_CHOLYATK_8)
    a_cholypalette(sim, player, psp)
    assert player.palette == STARTHOLYPAL + 2  # 24

    p_set_psprite(sim, player, PS_WEAPON, StateNum.S_CHOLYATK_9)
    a_cholypalette(sim, player, psp)
    assert player.palette == 0  # reset to normal
