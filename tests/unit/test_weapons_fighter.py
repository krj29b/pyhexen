# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for Task T10.7: Fighter weapons."""

from __future__ import annotations

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.sim.actions import is_action_implemented
from pyhexen.sim.actions.weapons_fighter import (
    a_faxeattack,
    a_fhammerattack,
    a_fhammerthrow,
    a_fpunchattack,
    a_fswordattack,
)
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjType, StateNum
from pyhexen.sim.mobj import ONFLOORZ, Mobj, spawn_mobj
from pyhexen.sim.player import PlayerClass, spawn_player
from pyhexen.sim.psprites import (
    MANA_1,
    MANA_2,
    PS_WEAPON,
    WP_FOURTH,
    WP_SECOND,
    WP_THIRD,
    p_set_psprite,
)
from pyhexen.wad.wadfile import WadFile, WadStack
from pyhexen.world.level import load_level


@pytest.fixture
def sim() -> GameSim:
    """Create a synthetic GameSim box room with a Fighter player."""
    wb = WadBuilder()
    mb = MapBuilder("MAP01")
    mb.add_room(-512, -512, 1024, 1024, floor_height=0, ceiling_height=256)
    wb.add_map(mb)
    wad = WadFile.from_bytes(wb.to_bytes())
    stack = WadStack(wad)
    level = load_level(stack, "MAP01")
    return GameSim(level=level, player_class=int(PlayerClass.PCLASS_FIGHTER), rng=Random(42))


def test_fighter_actions_registered() -> None:
    """Verify all Fighter weapon actions are registered."""
    for act_name in (
        "A_FPunchAttack",
        "A_FAxeAttack",
        "A_FHammerAttack",
        "A_FHammerThrow",
        "A_FSwordAttack",
    ):
        assert is_action_implemented(act_name), f"{act_name} should be implemented"


def test_punch_attack_miss_and_wall_hit(sim: GameSim) -> None:
    """Test Spiked Gauntlets punching air or hitting a wall."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_PUNCHREADY))

    # Punch in empty air
    a_fpunchattack(sim, player, psp)
    assert player.mo is not None
    assert player.mo.special1 == 0


def test_punch_attack_combo(sim: GameSim) -> None:
    """Test Spiked Gauntlets combo: 3 consecutive hits deal extra damage and grunt."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_PUNCHREADY))

    # Spawn an Ettin 50 units in front of player
    target = spawn_mobj(sim, 50 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_ETTIN)
    initial_health = target.health

    # Hit 1
    a_fpunchattack(sim, player, psp)
    assert player.mo.special1 == 1
    h1 = target.health
    assert h1 < initial_health
    dmg1 = initial_health - h1
    assert 40 <= dmg1 <= 55

    # Hit 2
    a_fpunchattack(sim, player, psp)
    assert player.mo.special1 == 2
    h2 = target.health
    assert h2 < h1

    # Hit 3: Combo finisher! Double damage, resets special1, changes psprite to S_PUNCHATK2_1
    a_fpunchattack(sim, player, psp)
    assert player.mo.special1 == 0
    h3 = target.health
    dmg3 = h2 - h3
    assert 80 <= dmg3 <= 110
    assert psp.state_num == int(StateNum.S_PUNCHATK2_1)


def test_axe_attack_without_mana(sim: GameSim) -> None:
    """Test Timon's Axe attack without Blue Mana deals unpowered damage and costs 0 mana."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    player.readyweapon = WP_SECOND
    player.mana[MANA_1] = 0
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_FAXEREADY))

    target = spawn_mobj(sim, 60 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_ETTIN)
    initial_health = target.health

    a_faxeattack(sim, player, psp)
    h = target.health
    assert h < initial_health
    dmg = initial_health - h
    # Normal damage: 40 + (r & 15) + (P_Random() & 7) -> [40 .. 62]
    assert 40 <= dmg <= 62
    assert player.mana[MANA_1] == 0
    # No thrust
    assert target.momx == 0
    assert target.momy == 0


def test_axe_attack_with_mana(sim: GameSim) -> None:
    """Test Timon's Axe attack with Blue Mana deals 2x damage, thrusts, and deducts mana."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    player.readyweapon = WP_SECOND
    player.mana[MANA_1] = 10
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_FAXEREADY_G))

    target = spawn_mobj(sim, 60 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_ETTIN)
    initial_health = target.health

    a_faxeattack(sim, player, psp)
    h = target.health
    assert h < initial_health
    dmg = initial_health - h
    # Powered damage: double damage [80 .. 124]
    assert 80 <= dmg <= 124
    # Consumes 2 mana
    assert player.mana[MANA_1] == 8
    # Target was thrust along positive X (player angle 0)
    assert target.momx > 0


def test_axe_attack_depletes_mana(sim: GameSim) -> None:
    """Test Timon's Axe attack depleting last mana transitions to S_FAXEATK_5."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    player.readyweapon = WP_SECOND
    player.mana[MANA_1] = 2  # exactly one powered swing
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_FAXEREADY_G))

    spawn_mobj(sim, 60 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_ETTIN)
    a_faxeattack(sim, player, psp)
    assert player.mana[MANA_1] == 0
    assert psp.state_num == int(StateNum.S_FAXEATK_5)


def test_hammer_attack_melee_hit(sim: GameSim) -> None:
    """Test Hammer of Retribution melee hit does not launch missile and does not use mana."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_THIRD
    player.mana[MANA_2] = 20
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_FHAMMERREADY))

    target = spawn_mobj(sim, 60 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_ETTIN)
    initial_health = target.health

    a_fhammerattack(sim, player, psp)
    assert target.health < initial_health
    dmg = initial_health - target.health
    # Damage: 60 + (r & 63) -> [60 .. 123]
    assert 60 <= dmg <= 123
    assert player.mo.special1 == 0  # flag to throw hammer is FALSE
    assert player.mana[MANA_2] == 20  # melee attack is free

    # A_FHammerThrow does nothing because special1 is 0
    a_fhammerthrow(sim, player, psp)
    assert player.mana[MANA_2] == 20
    # No missile spawned
    missiles = [
        t for t in sim.thinkers if isinstance(t, Mobj) and t.type == MobjType.MT_HAMMER_MISSILE
    ]
    assert len(missiles) == 0


def test_hammer_attack_throw(sim: GameSim) -> None:
    """Test Hammer swinging at nothing throws a hammer missile costing 3 Green Mana."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_THIRD
    player.mana[MANA_2] = 10
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_FHAMMERREADY))

    # Swing into open space
    a_fhammerattack(sim, player, psp)
    assert player.mo.special1 == 1  # will throw hammer

    # Throw hammer
    a_fhammerthrow(sim, player, psp)
    assert player.mana[MANA_2] == 7  # cost 3 mana
    missiles = [
        t for t in sim.thinkers if isinstance(t, Mobj) and t.type == MobjType.MT_HAMMER_MISSILE
    ]
    assert len(missiles) == 1
    h_missile = missiles[0]
    assert h_missile.target is player.mo
    assert h_missile.momx > 0  # flying forward


def test_hammer_attack_no_mana_for_throw(sim: GameSim) -> None:
    """Test Hammer of Retribution swinging at nothing with < 3 mana sets special1 to 0."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_THIRD
    player.mana[MANA_2] = 2  # not enough to throw
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_FHAMMERREADY))

    a_fhammerattack(sim, player, psp)
    assert player.mo.special1 == 0
    a_fhammerthrow(sim, player, psp)
    assert player.mana[MANA_2] == 2
    missiles = [
        t for t in sim.thinkers if isinstance(t, Mobj) and t.type == MobjType.MT_HAMMER_MISSILE
    ]
    assert len(missiles) == 0


def test_sword_attack(sim: GameSim) -> None:
    """Test Quietus fires 5 spread missiles and deducts 14 of both mana."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_FOURTH
    player.mana[MANA_1] = 50
    player.mana[MANA_2] = 50
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_FSWORDREADY))

    a_fswordattack(sim, player, psp)
    # Deducts 14 Blue and 14 Green mana
    assert player.mana[MANA_1] == 36
    assert player.mana[MANA_2] == 36

    missiles = [
        t for t in sim.thinkers if isinstance(t, Mobj) and t.type == MobjType.MT_FSWORD_MISSILE
    ]
    assert len(missiles) == 5

    # Check angles: pmo.angle + ANG45/4, ANG45/8, 0, -ANG45/8, -ANG45/4
    angles = sorted([m.angle for m in missiles])
    assert len(set(angles)) == 5
