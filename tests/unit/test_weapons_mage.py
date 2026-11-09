# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for Task T10.9: Mage weapons."""

from __future__ import annotations

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.sim.actions import is_action_implemented
from pyhexen.sim.actions.weapons_mage import (
    SHARDSPAWN_DOWN,
    SHARDSPAWN_LEFT,
    SHARDSPAWN_RIGHT,
    SHARDSPAWN_UP,
    STARTSCOURGEPAL,
    a_fireconepl1,
    a_lastzap,
    a_lightningclip,
    a_lightningready,
    a_lightningremove,
    a_lightningzap,
    a_mlightningattack,
    a_mstaffattack,
    a_mstaffpalette,
    a_mwandattack,
    a_shedshard,
    a_zapmimic,
)
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjType, StateNum
from pyhexen.sim.mobj import ONFLOORZ, Mobj, remove_mobj, spawn_mobj
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
    """Create a synthetic GameSim box room with a Mage player."""
    wb = WadBuilder()
    mb = MapBuilder("MAP01")
    mb.add_room(-512, -512, 1024, 1024, floor_height=0, ceiling_height=256)
    wb.add_map(mb)
    wad = WadFile.from_bytes(wb.to_bytes())
    stack = WadStack(wad)
    level = load_level(stack, "MAP01")
    return GameSim(level=level, player_class=int(PlayerClass.PCLASS_MAGE), rng=Random(42))


def test_mage_actions_registered() -> None:
    """Verify all Mage weapon actions are registered."""
    for act_name in (
        "A_MWandAttack",
        "A_FireConePL1",
        "A_ShedShard",
        "A_LightningReady",
        "A_LightningClip",
        "A_LightningZap",
        "A_MLightningAttack",
        "A_MLightningAttack2",
        "A_ZapMimic",
        "A_LastZap",
        "A_LightningRemove",
        "A_MStaffAttack",
        "A_MStaffPalette",
    ):
        assert is_action_implemented(act_name), f"{act_name} should be implemented"


def test_mwand_attack(sim: GameSim) -> None:
    """Test Sapphire Wand attack spawns MT_MWAND_MISSILE with blaster thinker."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_FIRST
    psp = player.psprites[PS_WEAPON]

    a_mwandattack(sim, player, psp)

    missiles = [
        th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_MWAND_MISSILE
    ]
    assert len(missiles) == 1
    assert missiles[0].thinker_func == p_blaster_mobj_thinker


def test_fire_cone_melee_and_ranged(sim: GameSim) -> None:
    """Test Frost Shards melee ice damage and ranged shard projectile fire."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_SECOND
    player.mana[MANA_1] = 20
    psp = player.psprites[PS_WEAPON]

    # Case 1: Enemy in melee range
    target = spawn_mobj(sim, 40 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_ETTIN)
    initial_hp = target.health
    a_fireconepl1(sim, player, psp)
    assert player.mana[MANA_1] == 17
    assert target.health < initial_hp
    assert 90 <= (initial_hp - target.health) <= 105

    # Case 2: Fire into empty room -> spawns MT_SHARDFX1
    remove_mobj(sim, target)
    player.mana[MANA_1] = 20
    a_fireconepl1(sim, player, psp)
    assert player.mana[MANA_1] == 17

    shards = [th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_SHARDFX1]
    assert len(shards) == 1
    s = shards[0]
    assert s.special1 == (SHARDSPAWN_LEFT | SHARDSPAWN_RIGHT | SHARDSPAWN_UP | SHARDSPAWN_DOWN)
    assert s.special2 == 3
    assert s.args[0] == 3


def test_shed_shard(sim: GameSim) -> None:
    """Test Frost Shards reproduction and daughter shard shedding."""
    parent_shard = spawn_mobj(sim, 0, 0, ONFLOORZ, MobjType.MT_SHARDFX1)
    parent_shard.special1 = SHARDSPAWN_LEFT | SHARDSPAWN_RIGHT | SHARDSPAWN_UP | SHARDSPAWN_DOWN
    parent_shard.special2 = 3

    a_shedshard(sim, parent_shard)
    assert parent_shard.special2 == 0

    all_shards = [
        th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_SHARDFX1
    ]
    # Parent + 4 daughter shards
    assert len(all_shards) == 5


def test_lightning_attack_and_discharge(sim: GameSim) -> None:
    """Test Arc of Death lightning spawn, floor/ceiling linking, and discharge."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_THIRD
    player.mana[MANA_2] = 20
    psp = player.psprites[PS_WEAPON]

    # Test idle ready sound
    a_lightningready(sim, player, psp)

    # Fire lightning
    a_mlightningattack(sim, player, psp)
    assert player.mana[MANA_2] == 15

    floor_mo = next(
        th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_LIGHTNING_FLOOR
    )
    ceil_mo = next(
        th
        for th in sim.thinkers
        if isinstance(th, Mobj) and th.type == MobjType.MT_LIGHTNING_CEILING
    )
    assert floor_mo.special2 == ceil_mo
    assert ceil_mo.special2 == floor_mo

    # Test lightning clipping and zig-zag
    a_lightningclip(sim, floor_mo)

    # Test lightning zap discharge
    hp_before = floor_mo.health
    a_lightningzap(sim, floor_mo)
    assert floor_mo.health == hp_before - 8

    zaps = [
        th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_LIGHTNING_ZAP
    ]
    assert len(zaps) >= 1

    # Test zap mimic
    z0 = zaps[0]
    z0.special2 = floor_mo
    floor_mo.momx = 5 * FRACUNIT
    floor_mo.momy = 5 * FRACUNIT
    a_zapmimic(sim, z0)
    assert z0.momx == floor_mo.momx
    assert z0.momy == floor_mo.momy

    # Test last zap
    a_lastzap(sim, floor_mo)
    terminal_zaps = [
        th
        for th in sim.thinkers
        if isinstance(th, Mobj)
        and th.type == MobjType.MT_LIGHTNING_ZAP
        and th.state_num == int(StateNum.S_LIGHTNING_ZAP_X1)
    ]
    assert len(terminal_zaps) == 1
    assert terminal_zaps[0].momz == 40 * FRACUNIT

    # Test lightning counterpart removal
    a_lightningremove(sim, floor_mo)
    assert ceil_mo.state_num == int(ceil_mo.info.deathstate)


def test_mstaff_attack_and_palette(sim: GameSim) -> None:
    """Test Bloodscourge triple projectile fire, mana deduction, and palette fade."""
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None
    player.readyweapon = WP_FOURTH
    player.mana[MANA_1] = 50
    player.mana[MANA_2] = 50
    psp = player.psprites[PS_WEAPON]

    a_mstaffattack(sim, player, psp)
    assert player.mana[MANA_1] == 35
    assert player.mana[MANA_2] == 35
    assert player.palette == STARTSCOURGEPAL  # 25
    assert player.damagecount == 0
    assert player.bonuscount == 0

    blood_missiles = [
        th for th in sim.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_MSTAFF_FX2
    ]
    assert len(blood_missiles) == 3

    # Test palette fade across states S_MSTAFFATK_2..5
    p_set_psprite(sim, player, PS_WEAPON, StateNum.S_MSTAFFATK_2)
    a_mstaffpalette(sim, player, psp)
    assert player.palette == STARTSCOURGEPAL  # 25

    p_set_psprite(sim, player, PS_WEAPON, StateNum.S_MSTAFFATK_3)
    a_mstaffpalette(sim, player, psp)
    assert player.palette == STARTSCOURGEPAL + 1  # 26

    p_set_psprite(sim, player, PS_WEAPON, StateNum.S_MSTAFFATK_4)
    a_mstaffpalette(sim, player, psp)
    assert player.palette == STARTSCOURGEPAL + 2  # 27

    p_set_psprite(sim, player, PS_WEAPON, StateNum.S_MSTAFFATK_5)
    a_mstaffpalette(sim, player, psp)
    assert player.palette == 0  # reset to normal
