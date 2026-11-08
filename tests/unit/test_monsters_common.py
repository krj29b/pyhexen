# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for Task T10.5a: Monster actions — common actors."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.events import SoundEvent
from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.actions import get_action, is_action_implemented
from pyhexen.sim.actions.monsters_common import (
    FLOOR_ICE,
    FLOOR_LAVA,
    FLOOR_SLUDGE,
    FLOOR_SOLID,
    FLOOR_WATER,
    ICEGUY_FX2_MOMZ,
    SMALLSPLASHCLIP,
    a_centaurattack,
    a_centaurattack2,
    a_centaurdefend,
    a_centaurdropstuff,
    a_contmobjsound,
    a_delaygib,
    a_demon2death,
    a_demonattack1,
    a_demonattack2,
    a_demondeath,
    a_dropmace,
    a_ettinattack,
    a_firedattack,
    a_firedchase,
    a_firedrocks,
    a_firedsplotch,
    a_floatgib,
    a_hidething,
    a_iceguyattack,
    a_iceguychase,
    a_iceguydie,
    a_iceguylook,
    a_iceguymissileexplode,
    a_iceguymissilepuff,
    a_serpentbirthscream,
    a_serpentchase,
    a_serpentchooseattack,
    a_serpentdivesound,
    a_serpentheadcheck,
    a_serpentheadpop,
    a_serpenthide,
    a_serpenthumpdecide,
    a_serpentlowerhump,
    a_serpentmeleeattack,
    a_serpentmissileattack,
    a_serpentraisehump,
    a_serpentspawngibs,
    a_serpentunhide,
    a_serpentwalk,
    a_setaltshadow,
    a_setshootable,
    a_sinkgib,
    a_smbounce,
    a_unhidething,
    a_unsetshootable,
    face_movement_direction,
    get_thing_floor_type,
    hit_floor,
    hitdice,
)
from pyhexen.sim.enemy import DI_EAST, DI_NORTHWEST, DI_SOUTHEAST, DI_WEST
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import STATES, MobjFlag, MobjFlag2, MobjType, SfxEnum, StateNum
from pyhexen.sim.mobj import FLOAT_BOB_OFFSETS, Mobj, spawn_mobj
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


@pytest.fixture
def sim() -> GameSim:
    """Create a synthetic open-arena GameSim instance for common monster action tests."""
    mb = MapBuilder("MAP01")
    mb.add_room(-512, -512, 1024, 1024)
    stack = WadStack(WadBuilder("IWAD").add_map(mb).to_bytes())
    level = load_level(stack, "MAP01")
    return GameSim(level=level)


def _mobjs_of_type(sim: GameSim, mtype: MobjType) -> list[Mobj]:
    return [t for t in sim.thinkers if isinstance(t, Mobj) and t.type == mtype]


# =============================================================================
# 1. Registry & Static AST Invariant Checks
# =============================================================================


def test_all_common_monster_state_actions_implemented() -> None:
    """Every action referenced by common monster states is implemented."""
    prefixes = (
        "S_ETTIN",
        "S_FIRED",
        "S_CENTAUR",
        "S_DEMN",
        "S_DEMN2",
        "S_ICEGUY",
        "S_ICEFX",
        "S_SERPENT",
    )
    missing: list[str] = []
    for idx, st in enumerate(STATES):
        sname = StateNum(idx).name
        if sname.startswith(prefixes) and st.action_name and st.action_name != "NULL":
            if not is_action_implemented(st.action_name):
                missing.append(f"{sname}:{st.action_name}")
    assert missing == []


def test_monsters_common_ast_invariants() -> None:
    """Verify zero floats, zero true divisions, zero stdlib random, and zero UI/render imports."""
    path = Path("src/pyhexen/sim/actions/monsters_common.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            pytest.fail(f"Float literal at line {node.lineno}")
        if isinstance(node, ast.Name) and node.id == "float":
            pytest.fail(f"float() call at line {node.lineno}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            pytest.fail(f"True division (/) at line {node.lineno}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "random"
                assert not alias.name.startswith(("pyhexen.render", "pyhexen.audio", "pyhexen.ui"))
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "random"
            assert not node.module.startswith(
                ("pyhexen.render", "pyhexen.audio", "pyhexen.ui", "render", "audio", "ui")
            )


# =============================================================================
# 2. Helpers & Floor Terrain / Splash Mechanics
# =============================================================================


def test_hitdice_and_face_movement_direction(sim: GameSim) -> None:
    """Verify HITDICE formula and 8-way FaceMovementDirection snapping."""
    sim.rng.prndindex = 0
    expected_rnd = sim.rng.p_random()
    sim.rng.prndindex = 0
    assert hitdice(sim, 5) == (1 + (expected_rnd & 7)) * 5

    ettin = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ETTIN)
    for d, expected_ang in (
        (DI_EAST, 0 << 29),
        (DI_NORTHWEST, 3 << 29),
        (DI_WEST, 4 << 29),
        (DI_SOUTHEAST, 7 << 29),
    ):
        ettin.movedir = d
        face_movement_direction(ettin)
        assert ettin.angle == (expected_ang & 0xFFFFFFFF)


def test_floor_types_and_hit_floor_splashes(sim: GameSim) -> None:
    """Verify get_thing_floor_type and hit_floor for water, lava, sludge, and solid floors."""
    actor = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ETTIN)
    assert get_thing_floor_type(actor) == FLOOR_SOLID
    assert hit_floor(sim, actor) == FLOOR_SOLID

    assert actor.subsector is not None and actor.subsector.sector is not None
    sec = actor.subsector.sector

    # Water large splash
    sec.floor_pic = "X_005"
    assert get_thing_floor_type(actor) == FLOOR_WATER
    res = hit_floor(sim, actor)
    assert res == FLOOR_WATER
    assert len(_mobjs_of_type(sim, MobjType.MT_SPLASH)) == 1
    assert len(_mobjs_of_type(sim, MobjType.MT_SPLASHBASE)) == 1

    # Water small splash (mass < 10)
    small = spawn_mobj(sim, 64 * FRACUNIT, 0, 0, MobjType.MT_WATER_DRIP)
    assert small.info.mass < 10
    res_small = hit_floor(sim, small)
    assert res_small == FLOOR_WATER
    bases = _mobjs_of_type(sim, MobjType.MT_SPLASHBASE)
    assert len(bases) == 2
    assert bases[-1].floorclip >= SMALLSPLASHCLIP

    # Lava splash
    sec.floor_pic = "X_001"
    assert get_thing_floor_type(actor) == FLOOR_LAVA
    assert hit_floor(sim, actor) == FLOOR_LAVA
    assert len(_mobjs_of_type(sim, MobjType.MT_LAVASMOKE)) == 1
    assert len(_mobjs_of_type(sim, MobjType.MT_LAVASPLASH)) == 1

    # Sludge splash
    sec.floor_pic = "X_009"
    assert get_thing_floor_type(actor) == FLOOR_SLUDGE
    assert hit_floor(sim, actor) == FLOOR_SLUDGE
    assert len(_mobjs_of_type(sim, MobjType.MT_SLUDGECHUNK)) == 1
    assert len(_mobjs_of_type(sim, MobjType.MT_SLUDGESPLASH)) == 1

    # Ice floor
    sec.floor_pic = "F_033"
    assert get_thing_floor_type(actor) == FLOOR_ICE


def test_shared_visibility_and_sound_actions(sim: GameSim) -> None:
    """Verify A_HideThing, A_UnHideThing, A_SetShootable, A_UnSetShootable, and A_SetAltShadow."""
    mo = spawn_mobj(sim, 0, 0, 0, MobjType.MT_SERPENT)

    a_hidething(mo, sim)
    assert bool(mo.flags2 & MobjFlag2.MF2_DONTDRAW)
    a_unhidething(mo, sim)
    assert not bool(mo.flags2 & MobjFlag2.MF2_DONTDRAW)

    a_unsetshootable(mo, sim)
    assert bool(mo.flags2 & MobjFlag2.MF2_NONSHOOTABLE)
    assert not bool(mo.flags & MobjFlag.MF_SHOOTABLE)

    a_setshootable(mo, sim)
    assert not bool(mo.flags2 & MobjFlag2.MF2_NONSHOOTABLE)
    assert bool(mo.flags & MobjFlag.MF_SHOOTABLE)

    mo.flags |= MobjFlag.MF_SHADOW
    a_setaltshadow(mo, sim)
    assert not bool(mo.flags & MobjFlag.MF_SHADOW)
    assert bool(mo.flags & MobjFlag.MF_ALTSHADOW)

    sounds: list[SoundEvent] = []
    sim.events.subscribe(SoundEvent, sounds.append)
    sfx_mo = spawn_mobj(sim, 0, 0, 0, MobjType.MT_SERPENTFX)
    a_contmobjsound(sfx_mo, sim)
    assert any(s.sound_id == int(SfxEnum.SFX_SERPENTFX_CONTINUOUS) for s in sounds)


# =============================================================================
# 3. Ettin Actions
# =============================================================================


def test_ettin_attack_and_drop_mace(sim: GameSim) -> None:
    """Verify A_EttinAttack melee damage and A_DropMace mace spawn."""
    ettin = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ETTIN)
    target = spawn_mobj(sim, 32 * FRACUNIT, 0, 0, MobjType.MT_CENTAUR)
    ettin.target = target
    initial_hp = target.health

    a_ettinattack(ettin, sim)
    assert target.health < initial_hp
    assert 2 <= initial_hp - target.health <= 16

    # Out of melee range -> no damage
    hp_after = target.health
    target.x = 300 * FRACUNIT
    a_ettinattack(ettin, sim)
    assert target.health == hp_after

    # Drop mace
    a_dropmace(ettin, sim)
    maces = _mobjs_of_type(sim, MobjType.MT_ETTIN_MACE)
    assert len(maces) == 1
    mace = maces[0]
    assert mace.z == ettin.z + (ettin.height >> 1)
    assert mace.target is ettin
    assert mace.momz >= 10 * FRACUNIT


# =============================================================================
# 4. Afrit (FireDemon) Actions
# =============================================================================


def test_firedemon_rocks_attack_bounce_and_splotch(sim: GameSim) -> None:
    """Verify A_FiredSpawnRock, A_FiredRocks, A_FiredAttack, A_SmBounce, and A_FiredSplotch."""
    afrit = spawn_mobj(sim, 0, 0, 64 * FRACUNIT, MobjType.MT_FIREDEMON)
    target = spawn_mobj(sim, 256 * FRACUNIT, 0, 64 * FRACUNIT, MobjType.MT_ETTIN)
    afrit.target = target
    afrit.special2 = 3
    afrit.flags |= MobjFlag.MF_JUSTATTACKED

    a_firedrocks(afrit, sim)
    rock_types = {
        MobjType.MT_FIREDEMON_FX1,
        MobjType.MT_FIREDEMON_FX2,
        MobjType.MT_FIREDEMON_FX3,
        MobjType.MT_FIREDEMON_FX4,
        MobjType.MT_FIREDEMON_FX5,
    }
    rocks = [t for t in sim.thinkers if isinstance(t, Mobj) and t.type in rock_types]
    assert len(rocks) == 5
    assert all(r.special1 == 2 and r.target is afrit for r in rocks)
    assert afrit.special2 == 0
    assert not bool(afrit.flags & MobjFlag.MF_JUSTATTACKED)

    # Small bounce on rock
    rock = rocks[0]
    a_smbounce(rock, sim)
    assert rock.z == rock.floorz + FRACUNIT
    assert rock.momz >= 2 * FRACUNIT
    assert rock.momx in (0, FRACUNIT, 2 * FRACUNIT)
    assert rock.momy in (0, FRACUNIT, 2 * FRACUNIT)

    # FiredAttack spawns MT_FIREDEMON_FX6 and plays SFX_FIRED_ATTACK
    sounds: list[SoundEvent] = []
    sim.events.subscribe(SoundEvent, sounds.append)
    a_firedattack(afrit, sim)
    assert len(_mobjs_of_type(sim, MobjType.MT_FIREDEMON_FX6)) == 1
    assert any(s.sound_id == int(SfxEnum.SFX_FIRED_ATTACK) for s in sounds)

    # FiredSplotch spawns SPLOTCH1 and SPLOTCH2
    a_firedsplotch(afrit, sim)
    assert len(_mobjs_of_type(sim, MobjType.MT_FIREDEMON_SPLOTCH1)) == 1
    assert len(_mobjs_of_type(sim, MobjType.MT_FIREDEMON_SPLOTCH2)) == 1


def test_firedemon_chase_bobbing_hover_and_strafe(
    sim: GameSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify A_FiredChase vertical bobbing, minimum floor+64 hover, and strafing."""
    afrit = spawn_mobj(sim, 0, 0, 10 * FRACUNIT, MobjType.MT_FIREDEMON)
    target = spawn_mobj(sim, 200 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    afrit.target = target
    afrit.special1 = 0
    afrit.special2 = 0
    afrit.movedir = DI_WEST

    # Force strafe trigger (rnd < 30) and prevent missile state transition (rnd = 200)
    seq = iter([10, 64, 250, 250, 250])
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: next(seq, 250))

    start_z = afrit.z
    a_firedchase(afrit, sim)
    # Bob offset at index 0 plus +2*FRACUNIT hover adjustment since z < floorz + 64
    assert afrit.z == start_z + FLOAT_BOB_OFFSETS[0] + 2 * FRACUNIT
    assert afrit.special1 == 2
    assert afrit.special2 == 3
    assert afrit.angle == (DI_WEST << 29) & 0xFFFFFFFF


# =============================================================================
# 5. Centaur & Slaughtaur Actions
# =============================================================================


def test_centaur_attacks_dropstuff_and_defend(
    sim: GameSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify A_CentaurAttack, A_CentaurAttack2, A_CentaurDropStuff, and A_CentaurDefend."""
    centaur = spawn_mobj(sim, 0, 0, 0, MobjType.MT_CENTAURLEADER)
    target = spawn_mobj(sim, 40 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    centaur.target = target

    hp0 = target.health
    a_centaurattack(centaur, sim)
    assert 3 <= hp0 - target.health <= 9

    # Slaughtaur missile attack
    target.x = 200 * FRACUNIT
    a_centaurattack2(centaur, sim)
    assert len(_mobjs_of_type(sim, MobjType.MT_CENTAUR_FX)) == 1

    # Drop shield & sword
    a_centaurdropstuff(centaur, sim)
    shields = _mobjs_of_type(sim, MobjType.MT_CENTAUR_SHIELD)
    swords = _mobjs_of_type(sim, MobjType.MT_CENTAUR_SWORD)
    assert len(shields) == 1 and shields[0].target is centaur
    assert len(swords) == 1 and swords[0].target is centaur

    # Defend counter-attack when in melee range and p_random() < 32
    target.x = 40 * FRACUNIT
    centaur.flags2 |= MobjFlag2.MF2_INVULNERABLE
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 10)
    a_centaurdefend(centaur, sim)
    assert not bool(centaur.flags2 & MobjFlag2.MF2_INVULNERABLE)


# =============================================================================
# 6. Chaos Serpent (Demon & Demon2) Actions
# =============================================================================


def test_demon_attacks_and_deaths(sim: GameSim) -> None:
    """Verify A_DemonAttack1, A_DemonAttack2, A_DemonDeath, and A_Demon2Death."""
    demon1 = spawn_mobj(sim, 0, 0, 0, MobjType.MT_DEMON)
    demon2 = spawn_mobj(sim, 0, 128 * FRACUNIT, 0, MobjType.MT_DEMON2)
    target = spawn_mobj(sim, 40 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    demon1.target = target
    demon2.target = target

    hp0 = target.health
    a_demonattack1(demon1, sim)
    assert 2 <= hp0 - target.health <= 16

    target.x = 256 * FRACUNIT
    a_demonattack2(demon1, sim)
    fx1 = _mobjs_of_type(sim, MobjType.MT_DEMONFX1)
    assert len(fx1) == 1

    a_demonattack2(demon2, sim)
    fx2 = _mobjs_of_type(sim, MobjType.MT_DEMON2FX1)
    assert len(fx2) == 1

    a_demondeath(demon1, sim)
    for ctype in (
        MobjType.MT_DEMONCHUNK1,
        MobjType.MT_DEMONCHUNK2,
        MobjType.MT_DEMONCHUNK3,
        MobjType.MT_DEMONCHUNK4,
        MobjType.MT_DEMONCHUNK5,
    ):
        chunks = _mobjs_of_type(sim, ctype)
        assert len(chunks) == 1
        assert chunks[0].momz == 8 * FRACUNIT

    a_demon2death(demon2, sim)
    for ctype in (
        MobjType.MT_DEMON2CHUNK1,
        MobjType.MT_DEMON2CHUNK2,
        MobjType.MT_DEMON2CHUNK3,
        MobjType.MT_DEMON2CHUNK4,
        MobjType.MT_DEMON2CHUNK5,
    ):
        assert len(_mobjs_of_type(sim, ctype)) == 1


# =============================================================================
# 7. Wendigo (IceGuy) Actions
# =============================================================================


def test_iceguy_look_chase_attack_puff_die_and_explode(
    sim: GameSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Wendigo look, chase, attack, puff, die, and missile explode actions."""
    iceguy = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ICEGUY)
    target = spawn_mobj(sim, 256 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    iceguy.target = target

    # Force wisp spawn in A_IceGuyLook (p_random() < 64)
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 10)
    a_iceguylook(iceguy, sim)
    wisps = _mobjs_of_type(sim, MobjType.MT_ICEGUY_WISP1) + _mobjs_of_type(
        sim, MobjType.MT_ICEGUY_WISP2
    )
    assert len(wisps) >= 1

    # A_IceGuyChase spawns wisp inheriting actor momentum
    iceguy.momx = 3 * FRACUNIT
    iceguy.momy = -2 * FRACUNIT
    a_iceguychase(iceguy, sim)
    wisps_after = _mobjs_of_type(sim, MobjType.MT_ICEGUY_WISP1) + _mobjs_of_type(
        sim, MobjType.MT_ICEGUY_WISP2
    )
    assert len(wisps_after) > len(wisps)
    assert wisps_after[-1].target is iceguy

    # A_IceGuyAttack spawns 2 MT_ICEGUY_FX missiles
    a_iceguyattack(iceguy, sim)
    fx = _mobjs_of_type(sim, MobjType.MT_ICEGUY_FX)
    assert len(fx) == 2

    # A_IceGuyMissilePuff spawns MT_ICEFX_PUFF
    a_iceguymissilepuff(fx[0], sim)
    assert len(_mobjs_of_type(sim, MobjType.MT_ICEFX_PUFF)) == 1

    # A_IceGuyMissileExplode spawns 8 radial MT_ICEGUY_FX2 shards with momz == -19660
    fx[0].x = 128 * FRACUNIT
    a_iceguymissileexplode(fx[0], sim)
    shards = _mobjs_of_type(sim, MobjType.MT_ICEGUY_FX2)
    assert len(shards) == 8
    assert all(s.momz == ICEGUY_FX2_MOMZ and s.target is fx[0].target for s in shards)

    # A_IceGuyDie shatters into MT_ICECHUNK pieces
    a_iceguydie(iceguy, sim)
    assert len(_mobjs_of_type(sim, MobjType.MT_ICECHUNK)) > 0


# =============================================================================
# 8. Stalker (Serpent & SerpentLeader) Actions
# =============================================================================


def test_serpent_hide_unhide_hump_and_gibs(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Stalker hide/unhide, hump raise/lower/decide, head pop/check, and gib actions."""
    serpent = spawn_mobj(sim, 0, 0, 0, MobjType.MT_SERPENTLEADER)
    target = spawn_mobj(sim, 256 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    serpent.target = target

    a_serpentunhide(serpent, sim)
    assert not bool(serpent.flags2 & MobjFlag2.MF2_DONTDRAW)
    assert serpent.floorclip == 24 * FRACUNIT

    a_serpentraisehump(serpent, sim)
    assert serpent.floorclip == 20 * FRACUNIT
    a_serpentlowerhump(serpent, sim)
    assert serpent.floorclip == 24 * FRACUNIT

    a_serpenthide(serpent, sim)
    assert bool(serpent.flags2 & MobjFlag2.MF2_DONTDRAW)
    assert serpent.floorclip == 0

    # HumpDecide for leader -> S_SERPENT_SURFACE1 when p_random() <= 30 and next < 40
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 10)
    a_serpenthumpdecide(serpent, sim)
    assert serpent.state_num == StateNum.S_SERPENT_SURFACE1

    # Gibs and head pop
    a_serpentheadpop(serpent, sim)
    heads = _mobjs_of_type(sim, MobjType.MT_SERPENT_HEAD)
    assert len(heads) == 1
    head = heads[0]
    head.z = head.floorz
    a_serpentheadcheck(head, sim)
    assert head.state_num == StateNum.S_SERPENT_HEAD_X1

    a_serpentspawngibs(serpent, sim)
    g1 = _mobjs_of_type(sim, MobjType.MT_SERPENT_GIB1)
    g2 = _mobjs_of_type(sim, MobjType.MT_SERPENT_GIB2)
    g3 = _mobjs_of_type(sim, MobjType.MT_SERPENT_GIB3)
    assert len(g1) == 1 and len(g2) == 1 and len(g3) == 1
    gib = g1[0]
    assert gib.floorclip == 6 * FRACUNIT
    a_floatgib(gib, sim)
    assert gib.floorclip == 5 * FRACUNIT
    a_sinkgib(gib, sim)
    assert gib.floorclip == 6 * FRACUNIT
    tics_before = gib.tics
    a_delaygib(gib, sim)
    assert gib.tics <= tics_before


def test_serpent_combat_and_chase(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Stalker chase, walk, attack choice, melee attack, and missile attack."""
    serpent = spawn_mobj(sim, 0, 0, 0, MobjType.MT_SERPENTLEADER)
    target = spawn_mobj(sim, 256 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    serpent.target = target

    # ChooseAttack at distance for MT_SERPENTLEADER -> S_SERPENT_MISSILE1
    a_serpentchooseattack(serpent, sim)
    assert serpent.state_num == StateNum.S_SERPENT_MISSILE1

    # Missile attack spawns MT_SERPENTFX
    count_before = len(_mobjs_of_type(sim, MobjType.MT_SERPENTFX))
    a_serpentmissileattack(serpent, sim)
    assert len(_mobjs_of_type(sim, MobjType.MT_SERPENTFX)) == count_before + 1

    # Melee attack within range deals HITDICE(5) damage
    target.x = 40 * FRACUNIT
    hp0 = target.health
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 200)
    a_serpentmeleeattack(serpent, sim)
    assert 5 <= hp0 - target.health <= 40

    # SerpentWalk in melee range enters S_SERPENT_ATK1
    a_serpentwalk(serpent, sim)
    assert serpent.state_num == StateNum.S_SERPENT_ATK1

    # Sound actions
    sounds: list[SoundEvent] = []
    sim.events.subscribe(SoundEvent, sounds.append)
    a_serpentbirthscream(serpent, sim)
    a_serpentdivesound(serpent, sim)
    assert any(s.sound_id == int(SfxEnum.SFX_SERPENT_BIRTH) for s in sounds)
    assert any(s.sound_id == int(SfxEnum.SFX_SERPENT_ACTIVE) for s in sounds)
    assert get_action("A_SerpentChase") is a_serpentchase
