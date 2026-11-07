# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for combat damage and death subsystem (Task T10.1).

Covers P_DamageMobj, P_KillMobj, class-based armor absorption, momentum thrust physics,
pain chance response, and death variant state branching.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from pyhexen.core.angles import (
    ANG180,
    ANGLETOFINESHIFT,
    FINEMASK,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.events import SoundEvent
from pyhexen.core.fixed import FRACUNIT, fixed_mul
from pyhexen.core.random import Random
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import (
    STATES,
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
    StateNum,
)
from pyhexen.sim.inter import (
    ARMOR_ARMOR,
    damage_mobj,
    falling_damage,
    kill_mobj,
    poison_damage,
    poison_player,
)
from pyhexen.sim.mobj import Mobj, spawn_mobj
from pyhexen.sim.player import (
    CF_GODMODE,
    PW_INVULNERABILITY,
    Player,
    PlayerClass,
    PlayerState,
)


@pytest.fixture
def sim() -> GameSim:
    """Create a standalone GameSim for isolated combat tests."""
    sim_obj = GameSim(rng=Random())
    sim_obj.rng.clear()
    return sim_obj


def make_mobj(
    sim: GameSim,
    mtype: MobjType = MobjType.MT_ETTIN,
    x: int = 0,
    y: int = 0,
    z: int = 0,
) -> Mobj:
    """Helper to instantiate an actor mobj."""
    return spawn_mobj(sim, x, y, z, mtype)


def make_player(
    sim: GameSim,
    pclass: PlayerClass = PlayerClass.PCLASS_FIGHTER,
) -> Player:
    """Helper to instantiate a Player and its associated actor mobj."""
    mtypes = {
        PlayerClass.PCLASS_FIGHTER: MobjType.MT_PLAYER_FIGHTER,
        PlayerClass.PCLASS_CLERIC: MobjType.MT_PLAYER_CLERIC,
        PlayerClass.PCLASS_MAGE: MobjType.MT_PLAYER_MAGE,
        PlayerClass.PCLASS_PIG: MobjType.MT_PIGPLAYER,
    }
    mo = spawn_mobj(sim, 0, 0, 0, mtypes[pclass])
    player = Player(
        mo=mo,
        player_class=int(pclass),
        health=100,
        player_state=PlayerState.PST_LIVE,
    )
    mo.player = player
    sim.players.append(player)
    return player


def set_player_health(player: Player, hp: int) -> None:
    """Helper to set health synchronously on player and underlying actor mobj."""
    player.health = hp
    if player.mo is not None:
        player.mo.health = hp


# ===========================================================================
# Suite 1: Damage & Immunity Checks (test_damage_mobj_*)
# ===========================================================================


def test_damage_reduces_health(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    initial_health = target.health
    damage_mobj(sim, target, inflictor=None, source=None, damage=50)
    assert target.health == initial_health - 50


def test_damage_non_shootable_ignored(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.flags &= ~MobjFlag.MF_SHOOTABLE
    initial_health = target.health
    damage_mobj(sim, target, inflictor=None, source=None, damage=50)
    assert target.health == initial_health


def test_damage_already_dead_ignored(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.health = 0
    target.flags &= ~MobjFlag.MF_ICECORPSE
    initial_tics = target.tics
    damage_mobj(sim, target, inflictor=None, source=None, damage=50)
    assert target.health == 0
    assert target.tics == initial_tics


def test_damage_dormant_immune(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.flags2 |= MobjFlag2.MF2_DORMANT
    initial_health = target.health
    damage_mobj(sim, target, inflictor=None, source=None, damage=100)
    assert target.health == initial_health


def test_damage_invulnerable_flag_blocks_normal(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.flags2 |= MobjFlag2.MF2_INVULNERABLE
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1)
    initial_health = target.health
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=500)
    assert target.health == initial_health


def test_damage_invulnerable_flag_bypassed_by_telefrag(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.flags2 |= MobjFlag2.MF2_INVULNERABLE
    damage_mobj(sim, target, inflictor=None, source=None, damage=10000)
    assert target.health <= 0
    assert target.flags & MobjFlag.MF_CORPSE


def test_damage_invulnerable_flag_bypassed_by_holy_cloud_firebomb(sim: GameSim) -> None:
    penetrating_types = [
        MobjType.MT_HOLY_FX,
        MobjType.MT_POISONCLOUD,
        MobjType.MT_FIREBOMB,
    ]
    for mtype in penetrating_types:
        target = make_mobj(sim, MobjType.MT_ETTIN)
        target.flags2 |= MobjFlag2.MF2_INVULNERABLE
        inflictor = make_mobj(sim, mtype)
        initial_health = target.health
        damage_mobj(sim, target, inflictor=inflictor, source=None, damage=30)
        assert target.health == initial_health - 30


def test_damage_player_godmode_immune_normal(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    player.cheats |= CF_GODMODE
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=500)
    assert player.health == 100
    assert player.mo.health == 100


def test_damage_player_godmode_killed_by_telefrag(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    player.cheats |= CF_GODMODE
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=10000)
    assert player.health == 0
    assert player.player_state == PlayerState.PST_DEAD


def test_damage_player_invulnerability_power_immune(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    player.powers[PW_INVULNERABILITY] = 100
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=500)
    assert player.health == 100
    assert player.mo.health == 100


def test_damage_trainer_mode_halves_damage(sim: GameSim) -> None:
    sim.skill = 0  # sk_baby
    player = make_player(sim, PlayerClass.PCLASS_PIG)
    player.armor_points = [0, 0, 0, 0]
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=40)
    assert player.health == 80  # 40 halved to 20; 100 - 20 = 80


def test_damage_skullfly_halted(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.flags |= MobjFlag.MF_SKULLFLY
    target.momx = 10 * FRACUNIT
    target.momy = 10 * FRACUNIT
    target.momz = 10 * FRACUNIT
    damage_mobj(sim, target, inflictor=None, source=None, damage=10)
    assert target.momx == 0
    assert target.momy == 0
    assert target.momz == 0


# ===========================================================================
# Suite 2: Class Armor Absorption & Slot Depletion (test_armor_*)
# ===========================================================================


def test_fighter_passive_armor_absorbs_15_percent(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    player.armor_points = [0, 0, 0, 0]
    assert player.mo is not None
    # 100 dmg -> 15% saved = 15 -> 85 applied
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=100)
    assert player.health == 15
    assert player.mo.health == 15


def test_cleric_passive_armor_absorbs_10_percent(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_CLERIC)
    player.armor_points = [0, 0, 0, 0]
    assert player.mo is not None
    # 100 dmg -> 10% saved = 10 -> 90 applied
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=100)
    assert player.health == 10
    assert player.mo.health == 10


def test_mage_passive_armor_absorbs_5_percent(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_MAGE)
    player.armor_points = [0, 0, 0, 0]
    assert player.mo is not None
    # 100 dmg -> 5% saved = 5 -> 95 applied
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=100)
    assert player.health == 5
    assert player.mo.health == 5


def test_armor_slot_deduction_fighter_mesh(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    player.armor_points = [25 * FRACUNIT, 0, 0, 0]
    assert player.mo is not None
    # 30 damage: deduction = (30 * 25) / 300 = 2.5 FRACUNIT
    # Armor becomes 22.5 FRACUNIT
    # Saved percent = 15 + 25 = 40%. Saved damage = 30 * 40% = 12. Damage applied = 18.
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=30)
    assert player.armor_points[ARMOR_ARMOR] == int(22.5 * FRACUNIT)
    assert player.health == 82


def test_armor_slot_depletion_under_2_fracunit(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    # Give just enough armor that deduction drops it below 2 * FRACUNIT
    player.armor_points = [int(2.2 * FRACUNIT), 0, 0, 0]
    assert player.mo is not None
    # 10 damage: deduction = (10 * 25) / 300 = 0.8333 FRACUNIT
    # Remaining = 2.2 - 0.8333 = 1.3667 < 2 * FRACUNIT -> snaps to 0
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=10)
    assert player.armor_points[ARMOR_ARMOR] == 0


def test_armor_total_save_capped_at_100_percent(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    # Force total savedPercent > 100%
    player.armor_points = [
        50 * FRACUNIT,
        50 * FRACUNIT,
        50 * FRACUNIT,
        50 * FRACUNIT,
    ]
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=20)
    # 100% absorbed: saved = 20, damage = 0
    assert player.health == 100


def test_armor_saved_damage_cap_double_saved_percent(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    player.armor_points = [0, 0, 0, 0]
    assert player.mo is not None
    # Passive 15%. savedPercent = 15 * FRACUNIT.
    # Incoming 1000 damage: theoretical 15% save is 150.
    # Cap is savedPercent * 2 = 30 * FRACUNIT (30 damage).
    # Final damage applied = 1000 - 30 = 970.
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=1000)
    assert player.health == 0
    assert player.mo.health == 100 - 970


def test_player_damagecount_and_attacker_tracking(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_PIG)
    player.armor_points = [0, 0, 0, 0]
    attacker = make_mobj(sim, MobjType.MT_ETTIN)
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=None, source=attacker, damage=40)
    assert player.attacker is attacker
    assert player.damagecount == 40


def test_player_damagecount_capped_at_100(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_PIG)
    player.damagecount = 90
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=50)
    assert player.damagecount == 100


# ===========================================================================
# Suite 3: Pain Chance & State Transitions (test_pain_*)
# ===========================================================================


def test_pain_chance_success(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    painchance = target.info.painchance
    # Mock RNG to return less than painchance
    monkeypatch.setattr(Random, "p_random", lambda self: painchance - 1)
    damage_mobj(sim, target, inflictor=None, source=None, damage=20)
    assert target.flags & MobjFlag.MF_JUSTHIT
    assert target.state_num == target.info.painstate


def test_pain_chance_failure(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    painchance = target.info.painchance
    # Mock RNG to return >= painchance
    monkeypatch.setattr(Random, "p_random", lambda self: painchance + 1)
    damage_mobj(sim, target, inflictor=None, source=None, damage=20)
    assert not (target.flags & MobjFlag.MF_JUSTHIT)
    assert target.state_num != target.info.painstate


def test_pain_ignored_during_skullfly(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.flags |= MobjFlag.MF_SKULLFLY
    monkeypatch.setattr(Random, "p_random", lambda self: 0)
    damage_mobj(sim, target, inflictor=None, source=None, damage=20)
    assert not (target.flags & MobjFlag.MF_JUSTHIT)
    assert target.state_num != target.info.painstate


def test_pain_null_painstate_safe(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.info = replace(target.info, painstate=StateNum.S_NULL, painchance=255)
    monkeypatch.setattr(Random, "p_random", lambda self: 0)
    damage_mobj(sim, target, inflictor=None, source=None, damage=20)
    assert not target.removed
    assert target.state_num != StateNum.S_NULL


def test_retaliation_target_assignment(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    attacker = make_mobj(sim, MobjType.MT_CENTAUR)
    # Fail pain chance so target transitions to seestate
    monkeypatch.setattr(Random, "p_random", lambda self: 255)
    damage_mobj(sim, target, inflictor=None, source=attacker, damage=20)
    assert target.target is attacker
    assert target.threshold == 100
    assert target.state_num == target.info.seestate


def test_retaliation_boss_exempt(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    boss = make_mobj(sim, MobjType.MT_KORAX)
    boss.flags2 |= MobjFlag2.MF2_BOSS
    monkeypatch.setattr(Random, "p_random", lambda self: 255)
    damage_mobj(sim, target, inflictor=None, source=boss, damage=20)
    assert target.target is None
    assert target.threshold == 0


# ===========================================================================
# Suite 4: Momentum Thrust Physics (test_thrust_*)
# ===========================================================================


def test_thrust_direction_due_east(sim: GameSim) -> None:
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1, x=0, y=0)
    target = make_mobj(sim, MobjType.MT_ETTIN, x=100 * FRACUNIT, y=0)
    mass = target.info.mass
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=10)
    expected_thrust = (10 * (FRACUNIT >> 3) * 150) // mass
    assert target.momx == fixed_mul(expected_thrust, fine_cosine(0))
    assert target.momy == fixed_mul(expected_thrust, fine_sine(0))


def test_thrust_direction_due_north(sim: GameSim) -> None:
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1, x=0, y=0)
    target = make_mobj(sim, MobjType.MT_ETTIN, x=0, y=100 * FRACUNIT)
    mass = target.info.mass
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=10)
    expected_thrust = (10 * (FRACUNIT >> 3) * 150) // mass
    ang = (target.y - inflictor.y) and (
        (point_to_angle2(inflictor.x, inflictor.y, target.x, target.y) >> ANGLETOFINESHIFT)
        & FINEMASK
    )
    assert target.momx == fixed_mul(expected_thrust, fine_cosine(ang))
    assert target.momy == fixed_mul(expected_thrust, fine_sine(ang))


def test_thrust_mass_scaling(sim: GameSim) -> None:
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1, x=0, y=0)
    ettin = make_mobj(sim, MobjType.MT_ETTIN, x=100 * FRACUNIT, y=0)
    korax = make_mobj(sim, MobjType.MT_KORAX, x=100 * FRACUNIT, y=0)

    damage_mobj(sim, ettin, inflictor=inflictor, source=None, damage=20)
    damage_mobj(sim, korax, inflictor=inflictor, source=None, damage=20)

    expected_ettin_thrust = (20 * 1228800) // ettin.info.mass
    expected_korax_thrust = (20 * 1228800) // korax.info.mass
    assert ettin.momx == fixed_mul(expected_ettin_thrust, fine_cosine(0))
    assert korax.momx == fixed_mul(expected_korax_thrust, fine_cosine(0))
    assert ettin.momx > korax.momx


def test_thrust_nodmgthrust_ignored(sim: GameSim) -> None:
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1, x=0, y=0)
    inflictor.flags2 |= MobjFlag2.MF2_NODMGTHRUST
    target = make_mobj(sim, MobjType.MT_ETTIN, x=100 * FRACUNIT, y=0)
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=50)
    assert target.momx == 0
    assert target.momy == 0


def test_thrust_player_melee_source_ignored(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    target = make_mobj(sim, MobjType.MT_ETTIN, x=100 * FRACUNIT, y=0)
    assert player.mo is not None
    # Source is player mobj, inflictor is player mobj
    damage_mobj(sim, target, inflictor=player.mo, source=player.mo, damage=20)
    assert target.momx == 0
    assert target.momy == 0


def test_thrust_no_inflictor_safe(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    damage_mobj(sim, target, inflictor=None, source=None, damage=50)
    assert target.momx == 0
    assert target.momy == 0


def test_thrust_identical_coordinates(sim: GameSim) -> None:
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1, x=50 * FRACUNIT, y=50 * FRACUNIT)
    target = make_mobj(sim, MobjType.MT_ETTIN, x=50 * FRACUNIT, y=50 * FRACUNIT)
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=10)
    expected_thrust = (10 * 1228800) // target.info.mass
    # Angle defaults to 0 (East)
    assert target.momx == fixed_mul(expected_thrust, fine_cosine(0))
    assert target.momy == fixed_mul(expected_thrust, fine_sine(0))


def test_thrust_fall_forward_pitch(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1, x=0, y=0, z=0)
    target = make_mobj(
        sim,
        MobjType.MT_ETTIN,
        x=100 * FRACUNIT,
        y=0,
        z=100 * FRACUNIT,
    )
    target.health = 15
    # Force RNG bit 0 to be 1
    monkeypatch.setattr(Random, "p_random", lambda self: 1)
    # damage < 40, damage > health, dz > 64 * FRACUNIT, RNG & 1 == 1
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=20)
    base_thrust = (20 * 1228800) // target.info.mass
    fine = (ANG180 >> ANGLETOFINESHIFT) & FINEMASK
    expected_thrust = base_thrust * 4
    assert target.momx == fixed_mul(expected_thrust, fine_cosine(fine))
    assert target.momy == fixed_mul(expected_thrust, fine_sine(fine))


# ===========================================================================
# Suite 5: Death State Transitions & Variants (test_death_*)
# ===========================================================================


def test_normal_death_transition(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    # Ettin spawnhealth = 175. Overkill boundary: health < -87
    # 180 damage: health = 175 - 180 = -5 (normal death)
    damage_mobj(sim, target, inflictor=None, source=None, damage=180)
    assert target.health == -5
    assert target.state_num == target.info.deathstate


def test_extreme_gib_death_threshold_exact(sim: GameSim) -> None:
    # Ettin spawnhealth = 175. Boundary: -(175 >> 1) == -87.
    # Case A: health == -87 -> Normal death
    target_norm = make_mobj(sim, MobjType.MT_ETTIN)
    target_norm.health = 10
    damage_mobj(sim, target_norm, inflictor=None, source=None, damage=97)
    assert target_norm.health == -87
    assert target_norm.state_num == target_norm.info.deathstate

    # Case B: health == -88 -> Extreme death
    target_gib = make_mobj(sim, MobjType.MT_ETTIN)
    target_gib.health = 10
    damage_mobj(sim, target_gib, inflictor=None, source=None, damage=98)
    assert target_gib.health == -88
    assert target_gib.state_num == target_gib.info.xdeathstate


def test_extreme_death_fallback_if_no_xdeath(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.info = replace(target.info, xdeathstate=StateNum.S_NULL)
    damage_mobj(sim, target, inflictor=None, source=None, damage=10000)
    assert target.state_num == target.info.deathstate
    assert not target.removed


def test_fourth_weapon_forces_extreme_death(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    player.readyweapon = 3  # WP_FOURTH
    target = make_mobj(sim, MobjType.MT_ETTIN)
    assert player.mo is not None
    # 10 damage is enough to kill low-health target
    target.health = 5
    damage_mobj(sim, target, inflictor=None, source=player.mo, damage=10)
    assert target.health == -5000
    assert target.state_num == target.info.xdeathstate


def test_player_fire_death_fighter(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1)
    inflictor.flags2 |= MobjFlag2.MF2_FIREDAMAGE
    set_player_health(player, 20)
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=inflictor, source=None, damage=30)
    assert player.mo.state_num == int(StateNum.S_PLAY_F_FDTH1)
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert int(SfxEnum.SFX_PLAYER_FIGHTER_BURN_DEATH) in sound_ids


def test_player_fire_death_cleric(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_CLERIC)
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1)
    inflictor.flags2 |= MobjFlag2.MF2_FIREDAMAGE
    set_player_health(player, 20)
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=inflictor, source=None, damage=30)
    assert player.mo.state_num == int(StateNum.S_PLAY_C_FDTH1)
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert int(SfxEnum.SFX_PLAYER_CLERIC_BURN_DEATH) in sound_ids


def test_player_fire_death_mage(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_MAGE)
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1)
    inflictor.flags2 |= MobjFlag2.MF2_FIREDAMAGE
    set_player_health(player, 20)
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=inflictor, source=None, damage=30)
    assert player.mo.state_num == int(StateNum.S_PLAY_M_FDTH1)
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert int(SfxEnum.SFX_PLAYER_MAGE_BURN_DEATH) in sound_ids


def test_player_fire_overkill_gibs_instead_of_burn(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1)
    inflictor.flags2 |= MobjFlag2.MF2_FIREDAMAGE
    set_player_health(player, 10)
    assert player.mo is not None
    # 85 damage: armor absorbs 15% (12), applied is 73.
    # Health becomes 10 - 73 = -63 < -50 overkill threshold
    damage_mobj(sim, player.mo, inflictor=inflictor, source=None, damage=85)
    assert player.mo.state_num == player.mo.info.xdeathstate


def test_player_ice_death(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    inflictor = make_mobj(sim, MobjType.MT_ICESHARD)
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=inflictor, source=None, damage=150)
    assert player.mo.flags & MobjFlag.MF_ICECORPSE
    assert player.mo.state_num == int(StateNum.S_FPLAY_ICE)


def test_monster_ice_death(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    inflictor = make_mobj(sim, MobjType.MT_ICESHARD)
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=200)
    assert target.flags & MobjFlag.MF_ICECORPSE
    assert target.state_num == int(StateNum.S_ETTIN_ICE1)


def test_monster_ice_death_unknown_type(sim: GameSim) -> None:
    # Korax has no ice state in Hexen
    target = make_mobj(sim, MobjType.MT_KORAX)
    inflictor = make_mobj(sim, MobjType.MT_ICESHARD)
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=10000)
    assert not (target.flags & MobjFlag.MF_ICECORPSE)
    assert target.state_num == target.info.deathstate


def test_frozen_corpse_shatter_on_damage(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.health = 0
    target.flags |= MobjFlag.MF_ICECORPSE
    target.momx = 10 * FRACUNIT
    target.momy = 10 * FRACUNIT
    target.tics = 20
    # Hit with non-ice attack
    damage_mobj(sim, target, inflictor=None, source=None, damage=10)
    assert target.tics == 1
    assert target.momx == 0
    assert target.momy == 0


def test_frozen_corpse_immune_to_more_ice(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.health = 0
    target.flags |= MobjFlag.MF_ICECORPSE
    target.tics = 20
    inflictor = make_mobj(sim, MobjType.MT_ICESHARD)
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=10)
    assert target.tics == 20


def test_destructible_tree_fire_death(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_TREEDESTRUCTIBLE)
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1)
    inflictor.flags2 |= MobjFlag2.MF2_FIREDAMAGE
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=100)
    assert target.state_num == int(StateNum.S_ZTREEDES_X1)
    assert target.height == 24 * FRACUNIT
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert int(SfxEnum.SFX_TREE_EXPLODE) in sound_ids


def test_corpse_flags_and_dimensions(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.flags |= (
        MobjFlag.MF_SHOOTABLE | MobjFlag.MF_FLOAT | MobjFlag.MF_SKULLFLY | MobjFlag.MF_NOGRAVITY
    )
    target.flags2 |= MobjFlag2.MF2_PASSMOBJ
    initial_height = target.height
    damage_mobj(sim, target, inflictor=None, source=None, damage=target.health + 10)
    assert not (
        target.flags
        & (MobjFlag.MF_SHOOTABLE | MobjFlag.MF_FLOAT | MobjFlag.MF_SKULLFLY | MobjFlag.MF_NOGRAVITY)
    )
    assert target.flags & MobjFlag.MF_CORPSE
    assert target.flags & MobjFlag.MF_DROPOFF
    assert not (target.flags2 & MobjFlag2.MF2_PASSMOBJ)
    assert target.height == initial_height >> 2


def test_corpse_tic_desynchronization(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    initial_health = target.health
    monkeypatch.setattr(Random, "p_random", lambda self: 3)
    damage_mobj(sim, target, None, None, initial_health + 10)
    base_tics = STATES[target.info.deathstate].tics
    assert target.tics == base_tics - 3


# ===========================================================================
# Suite 6: Monster Death Specials & ACS Actions (test_specials_*)
# ===========================================================================


def test_monster_death_executes_special(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.flags |= MobjFlag.MF_COUNTKILL
    target.special = 10  # Door_Open
    target.args = (1, 16, 0, 0, 0)
    executed = []

    def mock_special(
        s: GameSim,
        spec: int,
        args: Any,
        line: Any = None,
        mo: Any = None,
        *,
        side: int = 0,
    ) -> bool:
        executed.append((spec, args, mo))
        return True

    monkeypatch.setattr("pyhexen.sim.inter.execute_line_special", mock_special)
    damage_mobj(sim, target, inflictor=None, source=None, damage=target.health + 50)
    assert len(executed) == 1
    assert executed[0][0] == 10
    assert executed[0][1] == (1, 16, 0, 0, 0)
    assert executed[0][2] is target


def test_sorcboss_death_triggers_acs(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_SORCBOSS)
    target.flags |= MobjFlag.MF_COUNTKILL
    target.special = 99
    started_scripts = []

    class MockVM:
        def start_script(
            self,
            number: int,
            args: Any = (0, 0, 0),
            activator: Any = None,
            line: Any = None,
            side: int = 0,
            sim: Any = None,
        ) -> Any:
            started_scripts.append((number, args, activator))

    sim.acs_vm = MockVM()  # type: ignore[assignment]
    damage_mobj(sim, target, inflictor=None, source=None, damage=target.health + 100)
    assert len(started_scripts) == 1
    assert started_scripts[0][0] == 99
    assert started_scripts[0][2] is target


# ===========================================================================
# Suite 7: Boundary, Edge Cases & Error Robustness (test_edge_cases_*)
# ===========================================================================


def test_zero_damage(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    initial_health = target.health
    damage_mobj(sim, target, inflictor=None, source=None, damage=0)
    assert target.health == initial_health


def test_negative_damage_clamped(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    initial_health = target.health
    damage_mobj(sim, target, inflictor=None, source=None, damage=-50)
    assert target.health == initial_health


def test_zero_mass_actor_safe(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    target.info = replace(target.info, mass=0)
    inflictor = make_mobj(sim, MobjType.MT_FIREBALL1, x=0, y=0)
    target.x = 100 * FRACUNIT
    target.y = 0
    damage_mobj(sim, target, inflictor=inflictor, source=None, damage=20)
    assert target.health < target.info.spawnhealth


def test_target_tracer_cleaned_on_death(sim: GameSim) -> None:
    target = make_mobj(sim, MobjType.MT_ETTIN)
    other = make_mobj(sim, MobjType.MT_CENTAUR)
    target.target = other
    kill_mobj(sim, None, target)
    assert target.target is None


def test_environmental_death_no_source(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=None, source=None, damage=1000)
    assert player.health == 0
    assert player.player_state == PlayerState.PST_DEAD


def test_inflictor_damage_modifiers(sim: GameSim) -> None:
    # Bishop missile halved
    target = make_mobj(sim, MobjType.MT_ETTIN)
    bish_fx = make_mobj(sim, MobjType.MT_BISH_FX)
    damage_mobj(sim, target, inflictor=bish_fx, source=None, damage=40)
    assert target.health == target.info.spawnhealth - 20

    # IceGuy missile halved
    target2 = make_mobj(sim, MobjType.MT_ETTIN)
    iceguy_fx = make_mobj(sim, MobjType.MT_ICEGUY_FX2)
    damage_mobj(sim, target2, inflictor=iceguy_fx, source=None, damage=40)
    assert target2.health == target2.info.spawnhealth - 20

    # Fighter sword missile reduced by 25% against player
    player = make_player(sim, PlayerClass.PCLASS_PIG)
    player.armor_points = [0, 0, 0, 0]
    fsword = make_mobj(sim, MobjType.MT_FSWORD_MISSILE)
    assert player.mo is not None
    damage_mobj(sim, player.mo, inflictor=fsword, source=None, damage=40)
    # 40 - (40 >> 2) = 40 - 10 = 30
    assert player.health == 70


def test_falling_damage(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    mo = player.mo
    assert mo is not None

    # Fall velocity >= 63 * FRACUNIT triggers instant fatal damage (10000)
    mo.momz = -65 * FRACUNIT
    falling_damage(sim, player)
    assert mo.health <= 0
    assert player.player_state == PlayerState.PST_DEAD

    # Moderate fall with momz > -39 * FRACUNIT
    player2 = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    mo2 = player2.mo
    assert mo2 is not None
    mo2.momz = -30 * FRACUNIT
    falling_damage(sim, player2)
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert int(SfxEnum.SFX_PLAYER_LAND) in sound_ids
    assert mo2.health < 100
    assert mo2.health > 0


def test_poison_damage(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    player.armor_points = [
        25 * FRACUNIT,
        25 * FRACUNIT,
        25 * FRACUNIT,
        25 * FRACUNIT,
    ]
    # Poison damage bypasses armor completely!
    poison_damage(sim, player, source=None, damage=20, play_pain_sound=False)
    assert player.armor_points == [
        25 * FRACUNIT,
        25 * FRACUNIT,
        25 * FRACUNIT,
        25 * FRACUNIT,
    ]
    assert player.health == 80
    assert player.mo is not None
    assert player.mo.health == 80


def test_poison_player(sim: GameSim) -> None:
    player = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    attacker = make_mobj(sim, MobjType.MT_ETTIN)
    poison_player(player, attacker, 40)
    assert player.poisoncount == 40
    poison_player(player, attacker, 80)
    # Capped at 100
    assert player.poisoncount == 100
