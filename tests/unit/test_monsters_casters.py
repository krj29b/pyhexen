# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.
"""Unit tests for Task T10.5b: Caster monster actions (Dark Bishop, Reiver/Wraith)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.angles import ANG90
from pyhexen.core.events import SoundEvent
from pyhexen.core.fixed import FRACBITS, FRACUNIT
from pyhexen.core.random import Random
from pyhexen.sim.actions import get_action, is_action_implemented
from pyhexen.sim.actions.monsters_casters import (
    a_bishopattack,
    a_bishopattack2,
    a_bishopchase,
    a_bishopdecide,
    a_bishopmissileseek,
    a_bishopmissileweave,
    a_bishoppainblur,
    a_bishoppuff,
    a_bishopspawnblur,
    a_wraithchase,
    a_wraithfx2,
    a_wraithfx3,
    a_wraithfx4,
    a_wraithinit,
    a_wraithlook,
    a_wraithmelee,
    a_wraithmissile,
    a_wraithraise,
    a_wraithraiseinit,
    raise_mobj,
    sink_mobj,
    spawn_dirt,
)
from pyhexen.sim.actions.monsters_common import FLOAT_BOB_OFFSETS
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import (
    STATES,
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
    StateNum,
)
from pyhexen.sim.mobj import Mobj, spawn_mobj
from pyhexen.wad.wadfile import WadFile, WadStack
from pyhexen.world.level import load_level


@pytest.fixture
def sim(tmp_path: Path) -> GameSim:
    """Create a synthetic box-room GameSim for caster monster action tests."""
    wb = WadBuilder()
    mb = MapBuilder("MAP01")
    mb.add_room(-512, -512, 1024, 1024)
    wb.add_map(mb)
    wad_path = tmp_path / "test_monsters_casters.wad"
    wb.write(wad_path)
    stack = WadStack(WadFile(wad_path))
    level = load_level(stack, "MAP01")
    return GameSim(level=level, rng=Random())


def _mobjs_of_type(sim: GameSim, mtype: MobjType) -> list[Mobj]:
    return [t for t in sim.thinkers if isinstance(t, Mobj) and t.type == mtype]


CASTER_ACTION_NAMES: tuple[str, ...] = (
    # Dark Bishop
    "A_BishopAttack",
    "A_BishopAttack2",
    "A_BishopMissileWeave",
    "A_BishopMissileSeek",
    "A_BishopDecide",
    "A_BishopDoBlur",
    "A_BishopSpawnBlur",
    "A_BishopChase",
    "A_BishopPuff",
    "A_BishopPainBlur",
    # Reiver (Wraith)
    "A_WraithInit",
    "A_WraithRaiseInit",
    "A_WraithRaise",
    "A_WraithMelee",
    "A_WraithMissile",
    "A_WraithFX2",
    "A_WraithFX3",
    "A_WraithFX4",
    "A_WraithLook",
    "A_WraithChase",
    # Slaughtaur (CentaurLeader)
    "A_CentaurAttack",
    "A_CentaurAttack2",
    "A_CentaurDropStuff",
    "A_CentaurDefend",
)


def test_caster_actions_registered_and_states_covered() -> None:
    """Verify all caster actions are registered and all Bishop/Wraith/Centaur states are covered."""
    for name in CASTER_ACTION_NAMES:
        assert is_action_implemented(name), f"{name} is not implemented"
        fn = get_action(name)
        assert fn is not None
        assert not getattr(fn, "_is_stub", False), f"{name} resolved to stub"

    prefixes = (
        "S_BISHOP",
        "S_BISHFX",
        "S_WRAITH",
        "S_WRTHFX",
        "S_CENTAUR",
    )
    for st_idx, st in enumerate(STATES):
        st_name = StateNum(st_idx).name
        if st_name.startswith(prefixes) and st.action_name is not None:
            assert is_action_implemented(st.action_name), (
                f"State {st_name} uses unimplemented action {st.action_name}"
            )


def test_sink_raise_and_spawn_dirt_helpers(sim: GameSim) -> None:
    """Verify sink_mobj, raise_mobj, and spawn_dirt."""
    wraith = spawn_mobj(sim, 0, 0, 0, MobjType.MT_WRAITHB)
    wraith.floorclip = 4 * FRACUNIT

    # raise_mobj decrements by 2 * FRACUNIT per call for MT_WRAITHB
    assert raise_mobj(wraith) is False
    assert wraith.floorclip == 2 * FRACUNIT
    assert raise_mobj(wraith) is True
    assert wraith.floorclip == 0
    assert raise_mobj(wraith) is True

    # sink_mobj increments by FRACUNIT for normal actors and 6*FRACUNIT for thrust spikes
    wraith.floorclip = wraith.info.height - FRACUNIT
    assert sink_mobj(wraith) is False
    assert wraith.floorclip == wraith.info.height
    assert sink_mobj(wraith) is True

    spike = spawn_mobj(sim, 64 * FRACUNIT, 0, 0, MobjType.MT_THRUSTFLOOR_UP)
    spike.floorclip = 0
    assert sink_mobj(spike) is False
    assert spike.floorclip == 6 * FRACUNIT
    spike.special2 = 3
    assert raise_mobj(spike) is False
    assert spike.floorclip == 3 * FRACUNIT

    # spawn_dirt creates one of MT_DIRT1..MT_DIRT6 with upward momentum
    dirt = spawn_dirt(sim, wraith, wraith.radius)
    assert dirt.type in (
        MobjType.MT_DIRT1,
        MobjType.MT_DIRT2,
        MobjType.MT_DIRT3,
        MobjType.MT_DIRT4,
        MobjType.MT_DIRT5,
        MobjType.MT_DIRT6,
    )
    assert dirt.z >= wraith.z + FRACUNIT


def test_bishop_melee_and_missile_volley(sim: GameSim) -> None:
    """Verify A_BishopAttack (melee and ranged setup) and A_BishopAttack2 volley loop."""
    bishop = spawn_mobj(sim, 0, 0, 0, MobjType.MT_BISHOP)
    target = spawn_mobj(sim, 40 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    bishop.target = target

    # Melee attack deals HITDICE(4) damage (4..32)
    hp0 = target.health
    a_bishopattack(bishop, sim)
    assert 4 <= hp0 - target.health <= 32

    # Ranged attack sets special1 to 5..8 missile count
    target.x = 256 * FRACUNIT
    a_bishopattack(bishop, sim)
    assert isinstance(bishop.special1, int)
    assert 5 <= bishop.special1 <= 8

    # A_BishopAttack2 spawns MT_BISH_FX with homing target in special1 and weave in special2
    bishop.special1 = 2
    a_bishopattack2(bishop, sim)
    assert bishop.special1 == 1
    missiles = _mobjs_of_type(sim, MobjType.MT_BISH_FX)
    assert len(missiles) == 1
    fx = missiles[0]
    assert fx.special1 is target
    assert fx.special2 == 16

    a_bishopattack2(bishop, sim)
    assert bishop.special1 == 0
    assert len(_mobjs_of_type(sim, MobjType.MT_BISH_FX)) == 2

    # Next call with special1 == 0 transitions Bishop back to S_BISHOP_WALK1
    a_bishopattack2(bishop, sim)
    assert bishop.state_num == StateNum.S_BISHOP_WALK1


def test_bishop_missile_weave_and_seek(sim: GameSim) -> None:
    """Verify A_BishopMissileWeave XY/Z bobbing and A_BishopMissileSeek homing."""
    bishop = spawn_mobj(sim, 0, 0, 0, MobjType.MT_BISHOP)
    target = spawn_mobj(sim, 256 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    bishop.target = target
    bishop.special1 = 1
    a_bishopattack2(bishop, sim)
    fx = _mobjs_of_type(sim, MobjType.MT_BISH_FX)[0]

    # Move missile away from shooter before weaving
    fx.x = 64 * FRACUNIT
    z_before = fx.z
    a_bishopmissileweave(fx, sim)
    # weaveXY was 0 -> 2, weaveZ was 16 -> 18
    assert fx.special2 == 18 + (2 << 16)
    expected_dz = FLOAT_BOB_OFFSETS[18] - FLOAT_BOB_OFFSETS[16]
    assert fx.z == z_before + expected_dz

    # Seek adjusts angle toward target
    fx.angle = ANG90
    a_bishopmissileseek(fx, sim)
    assert fx.angle < ANG90


def test_bishop_blur_chase_puff_and_painblur(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify A_BishopDecide, A_BishopDoBlur, A_BishopSpawnBlur, A_BishopChase, Puff, PainBlur."""
    bishop = spawn_mobj(sim, 0, 0, 64 * FRACUNIT, MobjType.MT_BISHOP)
    target = spawn_mobj(sim, 256 * FRACUNIT, 0, 64 * FRACUNIT, MobjType.MT_ETTIN)
    bishop.target = target

    # Decide when p_random() >= 220 transitions to S_BISHOP_BLUR1 (which runs A_BishopDoBlur)
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 230)
    a_bishopdecide(bishop, sim)
    assert bishop.state_num == StateNum.S_BISHOP_BLUR1
    assert isinstance(bishop.special1, int) and 3 <= bishop.special1 <= 6

    # SpawnBlur decrements special1 and spawns MT_BISHOPBLUR; at 0 stops momentum
    bishop.special1 = 1
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 200)
    a_bishopspawnblur(bishop, sim)
    assert bishop.special1 == 0
    assert bishop.momx == 0 and bishop.momy == 0
    assert bishop.state_num == StateNum.S_BISHOP_WALK1
    assert len(_mobjs_of_type(sim, MobjType.MT_BISHOPBLUR)) == 1

    # BishopChase vertical floatbob
    bishop.special2 = 0
    z0 = bishop.z
    a_bishopchase(bishop, sim)
    assert bishop.special2 == 4
    assert bishop.z == z0 - (FLOAT_BOB_OFFSETS[0] >> 1) + (FLOAT_BOB_OFFSETS[4] >> 1)

    # BishopPuff spawns MT_BISHOP_PUFF at z + 40 with momz = FRACUNIT // 2
    a_bishoppuff(bishop, sim)
    puffs = _mobjs_of_type(sim, MobjType.MT_BISHOP_PUFF)
    assert len(puffs) == 1
    assert puffs[0].momz == FRACUNIT // 2

    # PainBlur spawns MT_BISHOPPAINBLUR when p_random() >= 64
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 100)
    a_bishoppainblur(bishop, sim)
    assert len(_mobjs_of_type(sim, MobjType.MT_BISHOPPAINBLUR)) == 1


def test_wraith_init_raise_and_combat(sim: GameSim, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Reiver init, buried raise, melee lifesteal, missile, and FX2/3/4."""
    wraith = spawn_mobj(sim, 0, 0, 0, MobjType.MT_WRAITH)
    a_wraithinit(wraith, sim)
    assert wraith.z == 48 << FRACBITS
    assert wraith.special1 == 0

    # Buried wraith raise
    buried = spawn_mobj(sim, 128 * FRACUNIT, 0, 0, MobjType.MT_WRAITHB)
    target = spawn_mobj(sim, 400 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    buried.target = target
    buried.reactiontime = 8
    a_wraithraiseinit(buried, sim)
    assert not bool(buried.flags2 & MobjFlag2.MF2_DONTDRAW)
    assert not bool(buried.flags2 & MobjFlag2.MF2_NONSHOOTABLE)
    assert bool(buried.flags & (MobjFlag.MF_SHOOTABLE | MobjFlag.MF_SOLID))
    assert buried.floorclip == buried.info.height

    buried.floorclip = 2 * FRACUNIT
    a_wraithraise(buried, sim)
    assert buried.floorclip == 0
    assert buried.state_num == StateNum.S_WRAITH_CHASE1

    # Melee attack steals health from target and adds to wraith.health
    target.x = buried.x + 32 * FRACUNIT
    target.y = buried.y
    buried.health = 50
    hp_target_before = target.health
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 10)
    a_wraithmelee(buried, sim)
    stolen = hp_target_before - target.health
    assert 2 <= stolen <= 16
    assert buried.health == 50 + stolen

    # Missile attack spawns MT_WRAITHFX1 and emits SFX_WRAITH_MISSILE_FIRE
    sounds: list[SoundEvent] = []
    sim.events.subscribe(SoundEvent, sounds.append)
    target.x = 320 * FRACUNIT
    a_wraithmissile(buried, sim)
    fx1 = _mobjs_of_type(sim, MobjType.MT_WRAITHFX1)
    assert len(fx1) == 1
    assert any(s.sound_id == int(SfxEnum.SFX_WRAITH_MISSILE_FIRE) for s in sounds)

    # FX2 spawns 2 sparkle tail mobjs with floorclip = 10 * FRACUNIT
    a_wraithfx2(fx1[0], sim)
    fx2 = _mobjs_of_type(sim, MobjType.MT_WRAITHFX2)
    assert len(fx2) == 2
    assert all(m.floorclip == 10 * FRACUNIT and m.target is fx1[0] for m in fx2)

    # FX3 spawns P_Random() % 15 droplets (with p_random=10 -> 10 droplets)
    a_wraithfx3(buried, sim)
    assert len(_mobjs_of_type(sim, MobjType.MT_WRAITHFX3)) == 10

    # FX4 spawns both FX4 and FX5 when 20 <= chance < 25
    monkeypatch.setattr(type(sim.rng), "p_random", lambda _self: 22)
    a_wraithfx4(buried, sim)
    assert len(_mobjs_of_type(sim, MobjType.MT_WRAITHFX4)) >= 1
    assert len(_mobjs_of_type(sim, MobjType.MT_WRAITHFX5)) >= 1

    # WraithLook and WraithChase
    a_wraithlook(wraith, sim)
    wraith.target = target
    wraith.special1 = 0
    a_wraithchase(wraith, sim)
    assert wraith.special1 == 2
