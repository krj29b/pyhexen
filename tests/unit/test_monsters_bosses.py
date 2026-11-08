# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.
"""Unit tests for Task T10.5c: Boss monster actions."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.angles import ANG1, ANG45, ANG90
from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.sim.actions import get_action, is_action_implemented
from pyhexen.sim.actions.monsters_bosses import (
    KORAX_FIRST_TELEPORT_TID,
    MAULATORTICS,
    SORC_ACCELERATE,
    SORC_DECELERATE,
    SORC_NORMAL,
    SORC_STOPPED,
    SORCBALL_INITIAL_SPEED,
    SORCBALL_TERMINAL_SPEED,
    a_accelballs,
    a_bouncecheck,
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
    a_slowballs,
    a_smokepuffexit,
    a_sorcballorbit,
    a_sorcballpop,
    a_sorcbossattack,
    a_sorcererbishopentry,
    a_sorcfx1seek,
    a_sorcfx2orbit,
    a_sorcfx2split,
    a_sorcfx4check,
    a_sorcoffense1,
    a_sorcoffense2,
    a_sorcspinballs,
    a_sorcupdateballangle,
    a_spawnbishop,
    a_spawnfizzle,
    a_stopballs,
    check_minotaur_age,
    dragon_seek,
    p_teleport,
    p_test_mobj_location,
    rough_monster_search,
    spawn_korax_missile,
)
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import (
    STATES,
    MobjFlag,
    MobjFlag2,
    MobjType,
    StateNum,
)
from pyhexen.sim.inter import minotaur_slam
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


def test_heresiarch_balls_orbit_and_spells(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Heresiarch spinning balls, acceleration/deceleration, and all 3 spell types."""
    sorc = spawn_mobj(sim, 0, 0, 0, MobjType.MT_SORCBOSS)
    target = spawn_mobj(sim, 256 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    sorc.target = target

    a_sorcspinballs(sim, sorc)
    b1 = _mobjs_of_type(sim, MobjType.MT_SORCBALL1)
    b2 = _mobjs_of_type(sim, MobjType.MT_SORCBALL2)
    b3 = _mobjs_of_type(sim, MobjType.MT_SORCBALL3)
    assert len(b1) == 1 and len(b2) == 1 and len(b3) == 1
    assert sorc.args[3] == SORC_NORMAL
    assert sorc.args[4] == SORCBALL_INITIAL_SPEED

    # Orbit in normal mode updates ball coordinates and parent angle
    a_sorcballorbit(sim, b1[0])
    assert b1[0].z > sorc.z

    # Speed balls -> accelerate -> stop balls -> slow balls -> decelerate
    a_sorcbossattack(sim, sorc)
    assert sorc.args[3] == SORC_ACCELERATE
    for _ in range(25):
        a_accelballs(sim, b1[0])
    assert sorc.args[4] == SORCBALL_TERMINAL_SPEED

    a_slowballs(sorc)
    assert sorc.args[3] == SORC_DECELERATE
    for _ in range(25):
        a_decelballs(b1[0])
    assert sorc.args[4] == SORCBALL_INITIAL_SPEED
    assert sorc.args[3] == SORC_NORMAL

    a_stopballs(b1[0])
    assert sorc.args[3] == SORC_STOPPED
    a_sorcupdateballangle(b1[0])

    # Spell 2: Defensive shield (MT_SORCBALL2) + split + orbit
    sorc.args = (0, 0, int(MobjType.MT_SORCBALL2), SORC_STOPPED, 7)
    a_castsorcererspell(sim, sorc)
    assert sorc.flags2 & MobjFlag2.MF2_REFLECTIVE
    assert sorc.flags2 & MobjFlag2.MF2_INVULNERABLE
    fx2 = _mobjs_of_type(sim, MobjType.MT_SORCFX2)
    assert len(fx2) == 1
    a_sorcfx2split(sim, fx2[0])
    orbiters = _mobjs_of_type(sim, MobjType.MT_SORCFX2_T1)
    assert len(orbiters) == 16
    a_sorcfx2orbit(sim, orbiters[0])
    a_sorcfx2orbit(sim, orbiters[1])

    # Spell 1: Summon Bishop (when health <= spawnhealth // 3)
    sorc.health = 100
    sorc.args = (0, 0, int(MobjType.MT_SORCBALL1), SORC_STOPPED, 7)
    a_castsorcererspell(sim, sorc)
    fx3 = _mobjs_of_type(sim, MobjType.MT_SORCFX3)
    assert len(fx3) == 1
    fx3[0].x = 128 * FRACUNIT
    a_sorcererbishopentry(sim, fx3[0])
    assert len(_mobjs_of_type(sim, MobjType.MT_SORCFX3_EXPLOSION)) == 1
    a_spawnbishop(sim, fx3[0])
    assert len(_mobjs_of_type(sim, MobjType.MT_BISHOP)) == 1

    # Spell 3: Offensive attacks (A_SorcOffense1 & A_SorcOffense2)
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 64)
    sorc.args = (0, 0, int(MobjType.MT_SORCBALL3), SORC_STOPPED, 7)
    a_castsorcererspell(sim, sorc)
    a_sorcoffense1(sim, sorc)
    fx1 = _mobjs_of_type(sim, MobjType.MT_SORCFX1)
    assert len(fx1) == 2
    a_sorcfx1seek(sim, fx1[0])

    a_sorcoffense2(sim, sorc)
    fx4 = _mobjs_of_type(sim, MobjType.MT_SORCFX4)
    assert len(fx4) == 1
    fx4[0].special2 = 0
    a_sorcfx4check(sim, fx4[0])
    assert fx4[0].state_num == fx4[0].info.deathstate

    # Fizzle, ball pop, and bounce check
    a_spawnfizzle(sim, sorc)
    assert len(_mobjs_of_type(sim, MobjType.MT_SORCSPARK1)) == 5

    a_sorcballpop(sim, b1[0])
    assert b1[0].flags2 & MobjFlag2.MF2_LOGRAV
    b1[0].args = (0, 0, 0, 0, 0)
    a_bouncecheck(sim, b1[0])
    assert b1[0].state_num == b1[0].info.deathstate


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
