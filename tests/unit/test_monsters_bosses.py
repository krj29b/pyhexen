# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.
"""Unit tests for Task T10.5c: Boss monster actions."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.angles import ANG1, ANG45, ANG90, fine_cosine
from pyhexen.core.fixed import FRACUNIT, fixed_mul
from pyhexen.core.random import Random
from pyhexen.sim.actions import get_action, is_action_implemented
from pyhexen.sim.actions.monsters_bosses import (
    BALL2_ANGLEOFFSET,
    KORAX_FIRST_TELEPORT_TID,
    MAULATORTICS,
    SORC_ACCELERATE,
    SORC_DECELERATE,
    SORC_FIRESPELL,
    SORC_FIRING_SPELL,
    SORC_NORMAL,
    SORC_STOPPED,
    SORC_STOPPING,
    SORCBALL_INITIAL_SPEED,
    SORCBALL_SPEED_ROTATIONS,
    SORCBALL_TERMINAL_SPEED,
    SORCFX4_RAPIDFIRE_TIME,
    a_accelballs,
    a_castsorcererspell,
    a_cholyattack2,
    a_cholycheckscream,
    a_cholyseek,
    a_cholyspawnpuff,
    a_cholytail,
    a_classbosshealth,
    a_clericattack,
    a_decelballs,
    a_dragonattack,
    a_dragoncheckcrash,
    a_dragonflap,
    a_dragonfx2,
    a_dragoninitflight,
    a_dragonpain,
    a_fighterattack,
    a_fswordflames,
    a_kbolt,
    a_kboltraise,
    a_koraxbonepop,
    a_koraxchase,
    a_koraxcommand,
    a_koraxdecide,
    a_koraxmissile,
    a_koraxstep,
    a_koraxstep2,
    a_kspiritroam,
    a_mageattack,
    a_minotauratk1,
    a_minotauratk2,
    a_minotauratk3,
    a_minotaurcharge,
    a_minotaurchase,
    a_minotaurdecide,
    a_minotaurfade0,
    a_minotaurfade1,
    a_minotaurfade2,
    a_minotaurlook,
    a_minotaurroam,
    a_mntrfloorfire,
    a_mstafftrack,
    a_mstaffweave,
    a_smokepuffexit,
    a_sorcballorbit,
    a_sorcbossattack,
    a_sorcererbishopentry,
    a_sorcfx1seek,
    a_sorcfx2orbit,
    a_sorcfx2split,
    a_sorcfx4check,
    a_sorcoffense1,
    a_sorcoffense2,
    a_sorcspinballs,
    a_spawnbishop,
    a_speedballs,
    a_stopballs,
    check_minotaur_age,
    dragon_seek,
    p_teleport,
    p_test_mobj_location,
    rough_monster_search,
    spawn_korax_missile,
)
from pyhexen.sim.attacks import radius_attack
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import (
    STATES,
    MobjFlag,
    MobjFlag2,
    MobjType,
    StateNum,
)
from pyhexen.sim.inter import kill_mobj, minotaur_slam
from pyhexen.sim.mobj import (
    Mobj,
    insert_mobj_into_tid_list,
    spawn_mobj,
)
from pyhexen.wad.wadfile import WadFile, WadStack
from pyhexen.world.level import load_level


@pytest.fixture
def sim(tmp_path: Path) -> GameSim:
    """Create a synthetic large-room GameSim for boss monster action tests."""
    wb = WadBuilder()
    mb = MapBuilder("MAP01")
    mb.add_room(-1024, -1024, 2048, 2048, floor_height=0, ceiling_height=512)
    wb.add_map(mb)
    wad_path = tmp_path / "test_monsters_bosses.wad"
    wb.write(wad_path)
    stack = WadStack(WadFile(wad_path))
    level = load_level(stack, "MAP01")
    return GameSim(level=level, rng=Random())


def _mobjs_of_type(sim: GameSim, mtype: MobjType) -> list[Mobj]:
    return [t for t in sim.thinkers if isinstance(t, Mobj) and t.type == mtype]


BOSS_ACTION_NAMES: tuple[str, ...] = (
    # Death Wyvern
    "A_DragonInitFlight",
    "A_DragonFlight",
    "A_DragonFlap",
    "A_DragonAttack",
    "A_DragonFX2",
    "A_DragonPain",
    "A_DragonCheckCrash",
    # Maulotaur
    "A_MinotaurFade0",
    "A_MinotaurFade1",
    "A_MinotaurFade2",
    "A_MinotaurRoam",
    "A_MinotaurLook",
    "A_MinotaurChase",
    "A_MinotaurAtk1",
    "A_MinotaurDecide",
    "A_MinotaurCharge",
    "A_MinotaurAtk2",
    "A_MinotaurAtk3",
    "A_MntrFloorFire",
    "A_SmokePuffExit",
    # Heresiarch
    "A_SorcSpinBalls",
    "A_SorcBallOrbit",
    "A_SpeedBalls",
    "A_SlowBalls",
    "A_StopBalls",
    "A_AccelBalls",
    "A_DecelBalls",
    "A_SorcUpdateBallAngle",
    "A_CastSorcererSpell",
    "A_SorcOffense1",
    "A_SorcOffense2",
    "A_SorcBossAttack",
    "A_SpawnFizzle",
    "A_SorcFX1Seek",
    "A_SorcFX2Split",
    "A_SorcFX2Orbit",
    "A_SpawnBishop",
    "A_SorcererBishopEntry",
    "A_SorcFX4Check",
    "A_SorcBallPop",
    "A_BounceCheck",
    # Class Bosses & Projectiles
    "A_FighterAttack",
    "A_FSwordAttack2",
    "A_FSwordFlames",
    "A_ClericAttack",
    "A_CHolyAttack3",
    "A_CHolyAttack2",
    "A_CHolySeek",
    "A_CHolyTail",
    "A_CHolyCheckScream",
    "A_CHolySpawnPuff",
    "A_MageAttack",
    "A_MStaffAttack2",
    "A_MStaffWeave",
    "A_MStaffTrack",
    "A_ClassBossHealth",
    # Korax
    "A_KoraxChase",
    "A_KoraxStep",
    "A_KoraxStep2",
    "A_KoraxBonePop",
    "A_KoraxDecide",
    "A_KoraxMissile",
    "A_KoraxCommand",
    "A_KSpiritRoam",
    "A_KBolt",
    "A_KBoltRaise",
)


def test_boss_actions_registered_and_states_covered() -> None:
    """Verify all boss actions are registered and all boss/projectile states are covered."""
    for name in BOSS_ACTION_NAMES:
        assert is_action_implemented(name), f"{name} is not implemented"
        fn = get_action(name)
        assert fn is not None
        assert not getattr(fn, "_is_stub", False), f"{name} resolved to stub"

    prefixes = (
        "S_DRAGON",
        "S_MNTR",
        "S_MINOSMOKE",
        "S_SORC",
        "S_BISHMORPH",
        "S_FIGHTER",
        "S_CLERIC",
        "S_MAGE",
        "S_FSWORD_MISSILE",
        "S_FSWORD_FLAME",
        "S_HOLY_FX",
        "S_HOLY_TAIL",
        "S_HOLY_MISSILE",
        "S_MSTAFF_FX2",
        "S_KORAX",
        "S_KSPIRIT",
        "S_KBOLT",
    )
    for st_enum in StateNum:
        if st_enum >= len(STATES):
            continue
        st = STATES[st_enum]
        if st_enum.name.startswith(prefixes) and st.action_name is not None:
            assert is_action_implemented(st.action_name), (
                f"State {st_enum.name} uses unimplemented action {st.action_name}"
            )


def test_dragon_init_flight_seek_and_combat(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Death Wyvern waypoint navigation, flight, flap, attack, FX2, pain, and crash."""
    dragon = spawn_mobj(sim, 0, 0, 128 * FRACUNIT, MobjType.MT_DRAGON)
    insert_mobj_into_tid_list(sim, dragon, 10)

    spot1 = spawn_mobj(sim, 64 * FRACUNIT, 0, 128 * FRACUNIT, MobjType.MT_MAPSPOT)
    insert_mobj_into_tid_list(sim, spot1, 10)
    spot1.args = (11, 12, 0, 0, 0)

    spot2 = spawn_mobj(sim, 256 * FRACUNIT, 64 * FRACUNIT, 160 * FRACUNIT, MobjType.MT_MAPSPOT)
    insert_mobj_into_tid_list(sim, spot2, 11)

    spot3 = spawn_mobj(sim, 256 * FRACUNIT, -64 * FRACUNIT, 96 * FRACUNIT, MobjType.MT_MAPSPOT)
    insert_mobj_into_tid_list(sim, spot3, 12)

    a_dragoninitflight(sim, dragon)
    assert dragon.tracer is spot1

    target = spawn_mobj(sim, 300 * FRACUNIT, 64 * FRACUNIT, 128 * FRACUNIT, MobjType.MT_ETTIN)
    dragon.target = target
    dragon.angle = 0

    # Place dragon close to spot1 so dist < 4 triggers waypoint advancement toward target
    dragon.x = 50 * FRACUNIT
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 10)
    dragon_seek(sim, dragon, 4 * ANG1, 8 * ANG1)
    assert dragon.tracer is spot2

    # Flap and attack
    a_dragonflap(sim, dragon)
    a_dragonattack(sim, dragon)
    fireballs = _mobjs_of_type(sim, MobjType.MT_DRAGON_FX)
    assert len(fireballs) >= 1

    # Explosion sub-explosions (A_DragonFX2)
    fb = fireballs[0]
    fb.x = 200 * FRACUNIT
    a_dragonfx2(sim, fb)
    assert len(_mobjs_of_type(sim, MobjType.MT_DRAGON_FX2)) >= 1

    # Pain transitions to S_DRAGON_INIT if tracer is None
    dragon.tracer = None
    a_dragonpain(sim, dragon)
    assert dragon.state_num == StateNum.S_DRAGON_INIT

    # Crash when hitting floor
    dragon.z = dragon.floorz
    a_dragoncheckcrash(sim, dragon)
    assert dragon.state_num == StateNum.S_DRAGON_CRASH1


def test_minotaur_fades_summon_age_and_combat(
    sim: GameSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Maulotaur fade flags, summon lifespan, target search, melee, charge, and fire."""
    mntr = spawn_mobj(sim, 0, 0, 0, MobjType.MT_MINOTAUR)
    a_minotaurfade0(mntr)
    assert mntr.flags & MobjFlag.MF_SHADOW
    assert not (mntr.flags & MobjFlag.MF_ALTSHADOW)

    a_minotaurfade1(mntr)
    assert mntr.flags & MobjFlag.MF_ALTSHADOW
    assert not (mntr.flags & MobjFlag.MF_SHADOW)

    a_minotaurfade2(mntr)
    assert not (mntr.flags & (MobjFlag.MF_SHADOW | MobjFlag.MF_ALTSHADOW))

    # Summoned minotaur looking for enemy monsters
    master = spawn_mobj(sim, -64 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    enemy = spawn_mobj(sim, 128 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    mntr.special1 = master
    mntr.args = (0, 0, 0, 0, 0)
    sim.level_time = 10

    found = rough_monster_search(sim, mntr, 10)
    assert found is enemy

    a_minotaurlook(sim, mntr)
    assert mntr.target is enemy
    assert mntr.state_num == StateNum.S_MNTR_WALK1

    a_minotaurroam(sim, mntr)
    a_minotaurchase(sim, mntr)

    # Melee hammer attack
    enemy.x = 40 * FRACUNIT
    enemy.y = 0
    initial_hp = enemy.health
    a_minotauratk1(sim, mntr)
    assert enemy.health < initial_hp

    # Spread fire attack (Atk2) at range
    enemy.x = 256 * FRACUNIT
    a_minotauratk2(sim, mntr)
    fx1 = _mobjs_of_type(sim, MobjType.MT_MNTRFX1)
    assert len(fx1) == 5

    # Floor fire attack (Atk3 + A_MntrFloorFire)
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 250)
    a_minotauratk3(sim, mntr)
    fx2 = _mobjs_of_type(sim, MobjType.MT_MNTRFX2)
    assert len(fx2) == 1
    fx2[0].x = 128 * FRACUNIT
    a_mntrfloorfire(sim, fx2[0])
    assert len(_mobjs_of_type(sim, MobjType.MT_MNTRFX3)) == 1

    # Decide -> Charge attack & A_MinotaurCharge
    enemy.x = 256 * FRACUNIT
    enemy.z = 0
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 50)
    a_minotaurdecide(sim, mntr)
    assert mntr.flags & MobjFlag.MF_SKULLFLY
    assert mntr.args[4] == 35 // 2
    a_minotaurcharge(sim, mntr)
    assert len(_mobjs_of_type(sim, MobjType.MT_PUNCHPUFF)) == 1

    # Minotaur slam
    old_hp = enemy.health
    minotaur_slam(sim, mntr, enemy)
    assert enemy.health < old_hp
    assert enemy.momx != 0 or enemy.momy != 0

    # Smoke puff exit & age expiry
    a_smokepuffexit(sim, mntr)
    assert len(_mobjs_of_type(sim, MobjType.MT_MNTRSMOKEEXIT)) == 1

    sim.level_time = MAULATORTICS + 20
    assert not check_minotaur_age(sim, mntr)
    assert mntr.health <= 0


def _spawn_sorcerer(sim: GameSim) -> tuple[Mobj, Mobj, Mobj, Mobj, Mobj]:
    sorc = spawn_mobj(sim, 0, 0, 0, MobjType.MT_SORCBOSS)
    target = spawn_mobj(sim, 256 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    sorc.target = target
    a_sorcspinballs(sim, sorc)
    b1 = _mobjs_of_type(sim, MobjType.MT_SORCBALL1)
    b2 = _mobjs_of_type(sim, MobjType.MT_SORCBALL2)
    b3 = _mobjs_of_type(sim, MobjType.MT_SORCBALL3)
    assert len(b1) == len(b2) == len(b3) == 1
    return sorc, target, b1[0], b2[0], b3[0]


def test_sorc_spin_balls_initial_state(sim: GameSim) -> None:
    """A_SorcSpinBalls: slow-ball target speed, normal mode, ball 1 holds the FX4 countdown."""
    sorc, _target, b1, b2, b3 = _spawn_sorcerer(sim)
    assert sorc.args[0] == 0
    assert sorc.args[2] == SORCBALL_INITIAL_SPEED  # from A_SlowBalls
    assert sorc.args[3] == SORC_NORMAL
    assert sorc.args[4] == SORCBALL_INITIAL_SPEED
    assert sorc.special1 == ANG1
    assert b1.special2 == SORCFX4_RAPIDFIRE_TIME
    assert b2.target is sorc and b3.target is sorc


def test_sorc_ball_orbit_positions_and_advances_base_angle(sim: GameSim) -> None:
    """Only ball 1 advances the base angle; each ball sits at its fixed offset."""
    sorc, _target, b1, b2, _b3 = _spawn_sorcerer(sim)
    a_sorcballorbit(sim, b2)
    assert sorc.special1 == ANG1  # ball 2 does not advance the angle
    assert b2.angle == (ANG1 + BALL2_ANGLEOFFSET) & 0xFFFFFFFF
    a_sorcballorbit(sim, b1)
    assert sorc.special1 == ANG1 + ANG1 * SORCBALL_INITIAL_SPEED
    assert b1.special1 == ANG1 >> 19  # previous fine angle
    dist = sorc.radius - (b1.radius << 1)
    assert b1.x == sorc.x + fixed_mul(dist, fine_cosine(ANG1 >> 19))
    assert b1.z == sorc.z - sorc.floorclip + sorc.info.height


def test_sorc_accel_to_terminal_speed_starts_stopping(sim: GameSim) -> None:
    """A_SpeedBalls -> A_AccelBalls to terminal speed -> A_StopBalls picks a ball."""
    sorc, _target, b1, _b2, _b3 = _spawn_sorcerer(sim)
    a_speedballs(sim, sorc)
    assert sorc.args[3] == SORC_ACCELERATE
    assert sorc.args[2] == SORCBALL_TERMINAL_SPEED
    for _ in range(SORCBALL_TERMINAL_SPEED - SORCBALL_INITIAL_SPEED):
        a_accelballs(sim, b1)
    assert sorc.args[4] == SORCBALL_TERMINAL_SPEED
    assert sorc.args[3] == SORC_ACCELERATE
    before = sim.rng.prndindex
    a_accelballs(sim, b1)  # reached the target: stop the balls
    assert sim.rng.prndindex == (before + 1) & 0xFF
    assert sorc.args[3] == SORC_STOPPING
    assert sorc.args[1] == 0
    assert sorc.special2 in (
        int(MobjType.MT_SORCBALL1),
        int(MobjType.MT_SORCBALL2),
        int(MobjType.MT_SORCBALL3),
    )


def test_sorc_stop_balls_choice(sim: GameSim) -> None:
    """A_StopBalls: blue with no defence, green below half health, else yellow."""
    sorc = spawn_mobj(sim, 0, 0, 0, MobjType.MT_SORCBOSS)
    sim.rng.prndindex = 0  # next P_Random is RNDTABLE[1] = 8 (< 200)
    a_stopballs(sim, sorc)
    assert sorc.special2 == int(MobjType.MT_SORCBALL2)
    sorc.args = (5, 0, 0, 0, 0)
    sorc.health = sorc.info.spawnhealth // 2 - 1
    sim.rng.prndindex = 0
    a_stopballs(sim, sorc)
    assert sorc.special2 == int(MobjType.MT_SORCBALL3)
    sorc.health = sorc.info.spawnhealth
    sim.rng.prndindex = 0
    a_stopballs(sim, sorc)
    assert sorc.special2 == int(MobjType.MT_SORCBALL1)


def test_sorc_decel_returns_to_normal(sim: GameSim) -> None:
    sorc, _target, b1, _b2, _b3 = _spawn_sorcerer(sim)
    sorc.args = (0, 0, SORCBALL_INITIAL_SPEED, SORC_DECELERATE, SORCBALL_INITIAL_SPEED + 2)
    a_decelballs(sim, b1)
    a_decelballs(sim, b1)
    assert sorc.args[4] == SORCBALL_INITIAL_SPEED
    assert sorc.args[3] == SORC_DECELERATE
    a_decelballs(sim, b1)
    assert sorc.args[3] == SORC_NORMAL


def test_sorc_stopping_ball_stops_facing_sorcerer(sim: GameSim) -> None:
    """In stopping mode the chosen ball stops when it lines up with the sorcerer."""
    sorc, _target, _b1, b2, _b3 = _spawn_sorcerer(sim)
    sorc.angle = ANG90
    sorc.special2 = int(MobjType.MT_SORCBALL2)
    sorc.special1 = (ANG90 - BALL2_ANGLEOFFSET) & 0xFFFFFFFF
    sorc.args = (0, SORCBALL_SPEED_ROTATIONS + 1, 0, SORC_STOPPING, SORCBALL_TERMINAL_SPEED)
    a_sorcballorbit(sim, b2)
    assert sorc.args[3] == SORC_FIRESPELL
    assert sorc.args[4] == 0
    assert sorc.special1 == (ANG90 - BALL2_ANGLEOFFSET) & 0xFFFFFFFF


def test_sorc_fire_defensive_spell(sim: GameSim) -> None:
    """Fire-spell mode with the blue ball casts the reflective shield."""
    sorc, _target, _b1, b2, _b3 = _spawn_sorcerer(sim)
    sorc.special2 = int(MobjType.MT_SORCBALL2)
    sorc.args = (0, 0, 0, SORC_FIRESPELL, 0)
    a_sorcballorbit(sim, b2)
    assert sorc.args[3] == SORC_STOPPED
    assert sorc.args[0] == 255
    assert sorc.flags2 & MobjFlag2.MF2_REFLECTIVE
    assert sorc.flags2 & MobjFlag2.MF2_INVULNERABLE
    assert sorc.state_num == StateNum.S_SORC_ATTACK4
    fx2 = _mobjs_of_type(sim, MobjType.MT_SORCFX2)
    assert len(fx2) == 1 and fx2[0].target is sorc

    a_sorcfx2split(sim, fx2[0])
    halves = [m for m in _mobjs_of_type(sim, MobjType.MT_SORCFX2) if not m.removed]
    assert sorted(m.args[0] for m in halves if m is not fx2[0]) == [0, 1]
    for half in halves:
        if half is not fx2[0]:
            a_sorcfx2orbit(sim, half)
    assert sorc.args[0] == 254  # only the counter-clockwise half counts down
    assert _mobjs_of_type(sim, MobjType.MT_SORCFX2_T1)


def test_sorc_rapid_fire_spell(sim: GameSim) -> None:
    """Yellow ball with P_Random < 200 rapid-fires FX4 until its countdown runs out."""
    sorc, _target, b1, _b2, _b3 = _spawn_sorcerer(sim)
    sorc.special2 = int(MobjType.MT_SORCBALL1)
    sorc.args = (0, 0, 0, SORC_FIRESPELL, 0)
    sim.rng.prndindex = 0  # P_Random -> 8
    a_sorcballorbit(sim, b1)
    assert sorc.args[3] == SORC_FIRING_SPELL
    assert b1.special2 == SORCFX4_RAPIDFIRE_TIME
    assert b1.args[4] == 128
    a_sorcballorbit(sim, b1)
    assert len(_mobjs_of_type(sim, MobjType.MT_SORCFX4)) == 1
    assert b1.args[4] == 143
    b1.special2 = 0
    a_sorcballorbit(sim, b1)
    assert sorc.args[3] == SORC_STOPPED
    assert sorc.state_num == StateNum.S_SORC_ATTACK4


def test_sorc_offense2_wraps_byte_args(sim: GameSim) -> None:
    sorc, _target, b1, _b2, _b3 = _spawn_sorcerer(sim)
    b1.args = (0, 0, 0, 0, 250)
    a_sorcoffense2(sim, b1)
    assert b1.args[4] == (250 + 15) & 0xFF


def test_sorc_woosh_bumps_rotation_counter(sim: GameSim) -> None:
    """A full turn at terminal speed bumps args[1]."""
    sorc, _target, b1, _b2, _b3 = _spawn_sorcerer(sim)
    sorc.args = (0, 0, 0, SORC_STOPPED, SORCBALL_TERMINAL_SPEED)
    b1.special1 = 8000  # previous fine angle near the end of a turn
    sorc.special1 = ANG1
    a_sorcballorbit(sim, b1)
    assert sorc.args[1] == 1


def test_sorc_boss_attack_resumes_spinning(sim: GameSim) -> None:
    sorc = spawn_mobj(sim, 0, 0, 0, MobjType.MT_SORCBOSS)
    a_sorcbossattack(sim, sorc)
    assert sorc.args[3] == SORC_ACCELERATE
    assert sorc.args[2] == SORCBALL_INITIAL_SPEED


def test_sorc_reinforcement_and_offense_spells(sim: GameSim) -> None:
    """Green ball spawns FX3 missiles from the sorcerer; yellow fires two FX1."""
    sorc, _target, b1, _b2, b3 = _spawn_sorcerer(sim)
    a_castsorcererspell(sim, b3)
    fx3 = _mobjs_of_type(sim, MobjType.MT_SORCFX3)
    assert fx3 and all(m.target is sorc for m in fx3)
    a_castsorcererspell(sim, b1)
    fx1 = _mobjs_of_type(sim, MobjType.MT_SORCFX1)
    assert len(fx1) == 2
    assert all(m.target is sorc and m.args[3] == 15 for m in fx1)
    a_sorcfx1seek(sim, fx1[0])
    a_sorcoffense1(sim, b1)
    assert len(_mobjs_of_type(sim, MobjType.MT_SORCFX1)) == 4

    fx3[0].x = 128 * FRACUNIT
    a_sorcererbishopentry(sim, fx3[0])
    assert len(_mobjs_of_type(sim, MobjType.MT_SORCFX3_EXPLOSION)) == 1
    a_spawnbishop(sim, fx3[0])
    bishops = _mobjs_of_type(sim, MobjType.MT_BISHOP)
    assert len(bishops) == 1 and bishops[0].target is None


def test_sorc_fx4_check_countdown(sim: GameSim) -> None:
    fx4 = spawn_mobj(sim, 0, 0, 0, MobjType.MT_SORCFX4)
    fx4.special2 = 0
    a_sorcfx4check(sim, fx4)
    assert fx4.state_num == fx4.info.deathstate


def test_radius_attack_box_is_distance_wide(sim: GameSim) -> None:
    """Vanilla's (distance + MAXRADIUS) << FRACBITS wraps in int32, so the block box is
    only `distance` units wide: a thing in range but in an unvisited block is missed."""
    bm = sim.level.blockmap
    block = 128 * FRACUNIT
    # Spot 42 units into block 5, so a 40-unit box stays in block 5 (a 72-unit one would not).
    spot = spawn_mobj(sim, bm.org_x + 5 * block + 42 * FRACUNIT, 0, 0, MobjType.MT_POISONCLOUD)
    victim = spawn_mobj(sim, spot.x - 60 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    near = spawn_mobj(sim, spot.x - 10 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    assert (victim.x - bm.org_x) >> 23 == 4  # in range: 60 - radius 25 = 35 < 40
    radius_attack(sim, spot, None, 4, 40, True)
    assert victim.health == victim.info.spawnhealth
    assert near.health == near.info.spawnhealth - 5


def test_kill_mobj_keeps_target_and_always_rolls_tics(sim: GameSim) -> None:
    """P_KillMobj leaves target->target and calls P_Random even if the thing is removed."""
    killer = spawn_mobj(sim, 256 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    armor = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ZSUITOFARMOR)
    armor.target = killer
    kill_mobj(sim, killer, armor)
    assert armor.target is killer
    armor2 = spawn_mobj(sim, 64 * FRACUNIT, 0, 0, MobjType.MT_ZSUITOFARMOR)
    before = sim.rng.prndindex
    kill_mobj(sim, None, armor2)
    # A_SoAExplode: 10 chunks x (3 + spawn + 1 + 2 + 2) calls, then the tics roll.
    assert armor2.removed
    assert (sim.rng.prndindex - before) & 0xFF == (10 * 9 + 1) & 0xFF


def test_class_bosses_and_weapon_projectiles(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Fighter, Cleric, and Mage class bosses and their signature projectiles."""
    target = spawn_mobj(sim, 256 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)

    # Coop 5x health multiplier
    fboss = spawn_mobj(sim, 0, 0, 0, MobjType.MT_FIGHTER_BOSS)
    fboss.target = target
    sim.netgame = True
    sim.deathmatch = False
    base_hp = fboss.health
    a_classbosshealth(sim, fboss)
    assert fboss.health == base_hp * 5
    sim.netgame = False

    # 1. Fighter Boss: Quietus sword spread + flames
    a_fighterattack(sim, fboss)
    swords = _mobjs_of_type(sim, MobjType.MT_FSWORD_MISSILE)
    assert len(swords) == 5
    a_fswordflames(sim, swords[0])
    assert len(_mobjs_of_type(sim, MobjType.MT_FSWORD_FLAME)) >= 1

    # 2. Cleric Boss: Wraithverge missile -> 4 spirits + tails + seek + puff
    cboss = spawn_mobj(sim, 0, 64 * FRACUNIT, 0, MobjType.MT_CLERIC_BOSS)
    cboss.target = target
    a_clericattack(sim, cboss)
    holy_missiles = _mobjs_of_type(sim, MobjType.MT_HOLY_MISSILE)
    assert len(holy_missiles) == 1
    hm = holy_missiles[0]
    hm.x = 128 * FRACUNIT
    a_cholyspawnpuff(sim, hm)
    assert len(_mobjs_of_type(sim, MobjType.MT_HOLY_MISSILE_PUFF)) == 1

    a_cholyattack2(sim, hm)
    spirits = _mobjs_of_type(sim, MobjType.MT_HOLY_FX)
    tails = _mobjs_of_type(sim, MobjType.MT_HOLY_TAIL)
    assert len(spirits) == 4
    assert len(tails) == 12

    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 10)
    a_cholycheckscream(sim, spirits[0])
    assert spirits[0].special1 is target
    a_cholyseek(sim, spirits[0])

    # Tail follows spirit, then removes when spirit dies
    lead_tail = next(t for t in tails if t.special2 is spirits[0])
    a_cholytail(sim, lead_tail)
    spirits[0].health = 1
    a_cholyseek(sim, spirits[0])
    assert spirits[0].state_num == spirits[0].info.deathstate
    a_cholytail(sim, lead_tail)
    assert lead_tail.removed

    # 3. Mage Boss: Bloodscourge 3 missiles + weave + track
    mboss = spawn_mobj(sim, 0, -64 * FRACUNIT, 0, MobjType.MT_MAGE_BOSS)
    mboss.target = target
    a_mageattack(sim, mboss)
    mstaff = _mobjs_of_type(sim, MobjType.MT_MSTAFF_FX2)
    assert len(mstaff) == 3
    a_mstaffweave(sim, mstaff[0])
    a_mstafftrack(sim, mstaff[0])


def test_korax_chase_missiles_commands_spirits_and_bolts(
    sim: GameSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Korax chase/teleport, 6-arm missile volleys, commands/bolts, and spirits."""
    korax = spawn_mobj(sim, 0, 0, 0, MobjType.MT_KORAX)
    target = spawn_mobj(sim, 384 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    korax.target = target

    spot = spawn_mobj(sim, -256 * FRACUNIT, -256 * FRACUNIT, 0, MobjType.MT_MAPSPOT)
    spot.angle = ANG90
    insert_mobj_into_tid_list(sim, spot, KORAX_FIRST_TELEPORT_TID)

    a_koraxstep(sim, korax)
    a_koraxstep2(sim, korax)

    # Low health triggers one-time teleport to TID 248
    korax.health = (korax.info.spawnhealth >> 1) - 10
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 200)
    a_koraxchase(sim, korax)
    assert korax.x == spot.x and korax.y == spot.y
    assert korax.special1 == 1

    # Decide between attack and command
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 100)
    a_koraxdecide(sim, korax)
    assert korax.state_num == StateNum.S_KORAX_ATTACK1

    # Test all 6 Korax missile types (6 arms each)
    korax.x = 0
    korax.y = 0
    for fire_idx, proj_type in enumerate(
        (
            MobjType.MT_WRAITHFX1,
            MobjType.MT_DEMONFX1,
            MobjType.MT_DEMON2FX1,
            MobjType.MT_FIREDEMON_FX3,
            MobjType.MT_CENTAUR_FX,
            MobjType.MT_SERPENTFX,
        )
    ):
        monkeypatch.setattr(type(sim.rng), "p_random", lambda _self, idx=fire_idx: idx)
        a_koraxmissile(sim, korax)
        assert len(_mobjs_of_type(sim, proj_type)) == 6

    # Command spawns lightning bolt and triggers ACS script
    a_koraxcommand(sim, korax)
    bolts = _mobjs_of_type(sim, MobjType.MT_KORAX_BOLT)
    assert len(bolts) >= 1
    bolt = bolts[0]
    bolt.z = 0
    a_kboltraise(sim, bolt)
    assert len(_mobjs_of_type(sim, MobjType.MT_KORAX_BOLT)) >= 2
    bolt.special1 = 1
    a_kbolt(sim, bolt)
    assert bolt.special1 == 0
    a_kbolt(sim, bolt)
    assert bolt.removed

    # BonePop spawns 6 Korax spirits; test roam and death
    a_koraxbonepop(sim, korax)
    sp1 = _mobjs_of_type(sim, MobjType.MT_KORAX_SPIRIT1)
    assert len(sp1) == 1
    a_kspiritroam(sim, sp1[0])
    sp1[0].health = 0
    a_kspiritroam(sim, sp1[0])
    assert sp1[0].state_num == StateNum.S_KSPIRIT_DEATH1


def test_boss_helpers_location_teleport_and_korax_missile(sim: GameSim) -> None:
    """Verify p_test_mobj_location, p_teleport with fog, and spawn_korax_missile."""
    mo = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ETTIN)
    assert p_test_mobj_location(sim, mo)
    mo.z = mo.ceilingz
    assert not p_test_mobj_location(sim, mo)
    mo.z = 0

    assert p_teleport(sim, mo, 128 * FRACUNIT, 128 * FRACUNIT, ANG45, True)
    assert mo.x == 128 * FRACUNIT and mo.y == 128 * FRACUNIT
    assert len(_mobjs_of_type(sim, MobjType.MT_TFOG)) == 2

    dest = spawn_mobj(sim, 256 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    dest.flags |= MobjFlag.MF_SHADOW
    km = spawn_korax_missile(sim, mo.x, mo.y, mo.z + 32 * FRACUNIT, mo, dest, MobjType.MT_DEMONFX1)
    assert km is not None
    assert km.target is mo
