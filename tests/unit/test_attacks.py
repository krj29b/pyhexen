# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for combat attacks.

Covers hitscan raycasting, projectiles, rippers, seeker homing, and radius damage.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.angles import (
    ANG1,
    ANG45,
    ANG90,
    ANG180,
    ANGLETOFINESHIFT,
    FINEMASK,
    fine_cosine,
    fine_sine,
)
from pyhexen.core.events import SoundEvent
from pyhexen.core.fixed import FRACUNIT, fixed_mul
from pyhexen.core.random import Random
from pyhexen.sim.attacks import (
    aim_line_attack,
    face_mobj,
    line_attack,
    radius_attack,
    seeker_missile,
    spawn_missile,
    spawn_missile_angle,
    spawn_ripper_blood,
)
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import (
    MOBJINFO,
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
)
from pyhexen.sim.mobj import Mobj, spawn_mobj
from pyhexen.sim.movement import p_xy_movement, p_z_movement
from pyhexen.sim.player import Player, PlayerClass
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level

if TYPE_CHECKING:
    pass


@pytest.fixture
def sim() -> GameSim:
    sim_obj = GameSim(rng=Random())
    sim_obj.rng.clear()
    return sim_obj


def _build_level(mb: MapBuilder, tmp_path: Path) -> Level:
    unique_name = f"{mb.name}_{time.time_ns()}"
    wad_path = WadBuilder("IWAD").add_map(mb).write(tmp_path / f"{unique_name}.wad")
    with WadStack(wad_path) as wad:
        return load_level(wad, mb.name)


class MockRandom(Random):
    """Deterministic controllable RNG for test scenarios."""

    def __init__(self, p_val: int = 0, sub_val: int = 0) -> None:
        super().__init__()
        self._p_val = p_val
        self._sub_val = sub_val

    def p_random(self) -> int:
        return self._p_val

    def p_subrandom(self) -> int:
        return self._sub_val


def make_player(
    sim: GameSim, pclass: PlayerClass = PlayerClass.PCLASS_FIGHTER
) -> tuple[Player, Mobj]:
    mtypes = {
        PlayerClass.PCLASS_FIGHTER: MobjType.MT_PLAYER_FIGHTER,
        PlayerClass.PCLASS_CLERIC: MobjType.MT_PLAYER_CLERIC,
        PlayerClass.PCLASS_MAGE: MobjType.MT_PLAYER_MAGE,
    }
    mo = spawn_mobj(sim, 0, 0, 0, mtypes[pclass])
    player = Player(player_class=pclass, mo=mo)
    mo.player = player
    sim.players.append(player)
    return player, mo


# ==============================================================================
# Suite 1: Line Attacks & Pitch Aiming (aim_line_attack, line_attack)
# ==============================================================================


def test_aim_line_attack_horizontal(sim: GameSim) -> None:
    # Fighter at (0, 0, 0) height 64 -> shootz = 40 * FRACUNIT
    # Target Ettin at (200 * FRACUNIT, 0, 0) height 68
    # topslope = 9175, bottomslope = -13109 -> aimslope = -1967
    shooter = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, 200 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)

    slope = aim_line_attack(sim, shooter, angle=0, distance=300 * FRACUNIT)
    assert slope == -1967
    assert sim.linetarget is target


def test_aim_line_attack_elevated_target(sim: GameSim) -> None:
    shooter = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, 200 * FRACUNIT, 0, 60 * FRACUNIT, MobjType.MT_ETTIN)

    slope = aim_line_attack(sim, shooter, angle=0, distance=300 * FRACUNIT)
    assert slope == 17701
    assert sim.linetarget is target


def test_aim_line_attack_depressed_target(sim: GameSim) -> None:
    shooter = spawn_mobj(sim, 0, 0, 100 * FRACUNIT, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, 200 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)

    slope = aim_line_attack(sim, shooter, angle=0, distance=300 * FRACUNIT)
    assert slope < 0
    assert sim.linetarget is target


def test_aim_line_attack_too_high_out_of_pitch_bounds(sim: GameSim) -> None:
    shooter = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    spawn_mobj(sim, 200 * FRACUNIT, 0, 300 * FRACUNIT, MobjType.MT_ETTIN)

    slope = aim_line_attack(sim, shooter, angle=0, distance=300 * FRACUNIT)
    assert slope == 0
    assert sim.linetarget is None


def test_aim_line_attack_too_low_out_of_pitch_bounds(sim: GameSim) -> None:
    shooter = spawn_mobj(sim, 0, 0, 300 * FRACUNIT, MobjType.MT_PLAYER_FIGHTER)
    spawn_mobj(sim, 200 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)

    slope = aim_line_attack(sim, shooter, angle=0, distance=300 * FRACUNIT)
    assert slope == 0
    assert sim.linetarget is None


def test_aim_line_attack_blocked_by_onesided_wall(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    # Inside room
    shooter = spawn_mobj(sim_obj, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    # Outside room beyond the west 1-sided wall
    spawn_mobj(sim_obj, -100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    # Aiming west (ANG180) through the 1-sided wall
    slope = aim_line_attack(sim_obj, shooter, angle=ANG180, distance=400 * FRACUNIT)
    assert slope == 0
    assert sim_obj.linetarget is None


def test_aim_line_attack_twosided_window_clear(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=256,
        width2=256,
        height=256,
        floor1=0,
        ceil1=128,
        floor2=0,
        ceil2=128,
    )
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    shooter = spawn_mobj(sim_obj, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim_obj, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    # Aim East (0 deg) across the open 2-sided portal into room 2
    slope = aim_line_attack(sim_obj, shooter, angle=0, distance=400 * FRACUNIT)
    assert sim_obj.linetarget is target
    assert slope != 0


def test_aim_line_attack_twosided_window_closed(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    # Room 2 has ceiling = floor = 0 (completely closed door)
    mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=256,
        width2=256,
        height=256,
        floor1=0,
        ceil1=128,
        floor2=0,
        ceil2=0,
    )
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    shooter = spawn_mobj(sim_obj, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    spawn_mobj(sim_obj, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    slope = aim_line_attack(sim_obj, shooter, angle=0, distance=400 * FRACUNIT)
    assert slope == 0
    assert sim_obj.linetarget is None


def test_aim_line_attack_ignores_non_shootable_corpses(sim: GameSim) -> None:
    shooter = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, 200 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    target.flags &= ~MobjFlag.MF_SHOOTABLE

    slope = aim_line_attack(sim, shooter, angle=0, distance=300 * FRACUNIT)
    assert slope == 0
    assert sim.linetarget is None


def test_aim_line_attack_coop_ignores_friendly_player(sim: GameSim) -> None:
    sim.netgame = True
    sim.deathmatch = False
    shooter = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, 200 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_CLERIC)
    target.player = Player(player_class=PlayerClass.PCLASS_CLERIC, mo=target)

    slope = aim_line_attack(sim, shooter, angle=0, distance=300 * FRACUNIT)
    assert slope == 0
    assert sim.linetarget is None


def test_line_attack_hits_actor_applies_damage(sim: GameSim) -> None:
    shooter = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    initial_health = target.health

    line_attack(
        sim,
        t1=shooter,
        angle=0,
        distance=200 * FRACUNIT,
        slope=0,
        damage=25,
        puff_type=MobjType.MT_PUNCHPUFF,
    )
    assert target.health < initial_health


def test_line_attack_bleeding_target_spawns_blood(sim: GameSim) -> None:
    shooter = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    assert not (target.flags & MobjFlag.MF_NOBLOOD)

    # Force blood splatter chance < 192
    sim.rng = MockRandom(p_val=50)
    initial_thinkers = len(list(sim.thinkers))

    line_attack(
        sim,
        t1=shooter,
        angle=0,
        distance=200 * FRACUNIT,
        slope=0,
        damage=20,
        puff_type=MobjType.MT_PUNCHPUFF,
    )
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    blood = [mo for mo in mobjs if mo.type == MobjType.MT_BLOODSPLATTER]
    assert len(blood) >= 1
    assert len(list(sim.thinkers)) > initial_thinkers


def test_line_attack_non_bleeding_target_spawns_puff_no_blood(sim: GameSim) -> None:
    shooter = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_CENTAUR)
    target.flags |= MobjFlag.MF_NOBLOOD

    line_attack(
        sim,
        t1=shooter,
        angle=0,
        distance=200 * FRACUNIT,
        slope=0,
        damage=20,
        puff_type=MobjType.MT_PUNCHPUFF,
    )
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    blood = [mo for mo in mobjs if mo.type in (MobjType.MT_BLOOD, MobjType.MT_BLOODSPLATTER)]
    assert len(blood) == 0
    puffs = [mo for mo in mobjs if mo.type == MobjType.MT_PUNCHPUFF]
    assert len(puffs) >= 1


def test_line_attack_hits_wall_spawns_puff(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    shooter = spawn_mobj(sim_obj, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    # Fire East into the right room wall at x=256
    line_attack(
        sim_obj,
        t1=shooter,
        angle=0,
        distance=300 * FRACUNIT,
        slope=0,
        damage=20,
        puff_type=MobjType.MT_PUNCHPUFF,
    )
    puffs = [
        th for th in sim_obj.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_PUNCHPUFF
    ]
    assert len(puffs) >= 1


def test_line_attack_sky_wall_suppresses_puff(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    # Room with sky ceiling
    mb.add_room(x=0, y=0, width=256, height=256, ceiling_pic="F_SKY")
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    shooter = spawn_mobj(sim_obj, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    # Shoot upward into sky wall
    line_attack(
        sim_obj,
        t1=shooter,
        angle=0,
        distance=300 * FRACUNIT,
        slope=40000,
        damage=20,
        puff_type=MobjType.MT_PUNCHPUFF,
    )
    puffs = [
        th for th in sim_obj.thinkers if isinstance(th, Mobj) and th.type == MobjType.MT_PUNCHPUFF
    ]
    assert len(puffs) == 0


def test_line_attack_miss_sound_event(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=1024, height=1024)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    shooter = spawn_mobj(sim_obj, 500 * FRACUNIT, 500 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    sim_obj.events.drain()

    # Short distance melee attack into empty room space (miss)
    line_attack(
        sim_obj,
        t1=shooter,
        angle=0,
        distance=64 * FRACUNIT,
        slope=0,
        damage=20,
        puff_type=MobjType.MT_PUNCHPUFF,
    )
    events = sim_obj.events.drain()
    sound_events = [e for e in events if isinstance(e, SoundEvent)]
    assert any(e.sound_id == int(SfxEnum.SFX_FIGHTER_PUNCH_MISS) for e in sound_events)


def test_line_attack_flamepuff2_fire_damage(sim: GameSim) -> None:
    shooter = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_CLERIC)
    target = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    # Lethal damage
    line_attack(
        sim,
        t1=shooter,
        angle=0,
        distance=200 * FRACUNIT,
        slope=0,
        damage=500,
        puff_type=MobjType.MT_FLAMEPUFF2,
    )
    assert target.health <= 0
    # Lethal fire damage flags target with MF2_FIREDAMAGE
    assert bool(target.flags2 & MobjFlag2.MF2_FIREDAMAGE)


# ==============================================================================
# Suite 2: Missile Spawning, Trajectory & Collision (spawn_missile, check_missile_spawn)
# ==============================================================================


def test_spawn_missile_velocity_and_aiming(sim: GameSim) -> None:
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)

    # Spawn arrow projectile
    missile = spawn_missile(sim, source, dest, MobjType.MT_ARROW)
    assert missile is not None
    assert missile.flags & MobjFlag.MF_MISSILE
    assert missile.momx == fixed_mul(missile.info.speed, fine_cosine(0))
    assert missile.momy == fixed_mul(missile.info.speed, fine_sine(0))
    assert missile.momz == 0
    assert missile.target is source


def test_spawn_missile_elevated_dest_momz(sim: GameSim) -> None:
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim, 100 * FRACUNIT, 0, 50 * FRACUNIT, MobjType.MT_ETTIN)

    missile = spawn_missile(sim, source, dest, MobjType.MT_ARROW)
    assert missile is not None
    assert missile.momz > 0


def test_spawn_missile_custom_z_mobj_types(sim: GameSim) -> None:
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)

    m1 = spawn_missile(sim, source, dest, MobjType.MT_MNTRFX1)
    assert m1 is not None and m1.z == 40 * FRACUNIT

    m2 = spawn_missile(sim, source, dest, MobjType.MT_CENTAUR_FX)
    assert m2 is not None and m2.z == 45 * FRACUNIT

    m3 = spawn_missile(sim, source, dest, MobjType.MT_ICEGUY_FX)
    assert m3 is not None and m3.z == 40 * FRACUNIT

    m4 = spawn_missile(sim, source, dest, MobjType.MT_HOLY_MISSILE)
    assert m4 is not None and m4.z == 40 * FRACUNIT


def test_spawn_missile_shadow_target_jitter(sim: GameSim) -> None:
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    dest.flags |= MobjFlag.MF_SHADOW

    sim.rng = MockRandom(sub_val=100)
    missile = spawn_missile(sim, source, dest, MobjType.MT_ARROW)
    assert missile is not None
    assert missile.angle != 0  # Deflected


def test_spawn_missile_angle_velocity(sim: GameSim) -> None:
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    missile = spawn_missile_angle(
        sim,
        source=source,
        missile_type=MobjType.MT_ARROW,
        angle=ANG90,
        vz=5 * FRACUNIT,
    )
    assert missile is not None
    assert missile.angle == ANG90
    assert missile.momx == fixed_mul(missile.info.speed, fine_cosine(ANG90))
    assert missile.momy == fixed_mul(missile.info.speed, fine_sine(ANG90))
    assert missile.momz == 5 * FRACUNIT


def test_spawn_missile_blocked_at_spawn_explodes_immediately(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    # Source right at west room boundary wall
    source = spawn_mobj(sim_obj, 5 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim_obj, -100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    # Fire west directly into solid wall
    missile = spawn_missile(sim_obj, source, dest, MobjType.MT_ARROW)
    assert missile is None  # Exploded immediately upon spawn check


def test_missile_trajectory_progression_across_tics(sim: GameSim) -> None:
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim, 500 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    missile = spawn_missile(sim, source, dest, MobjType.MT_ARROW)
    assert missile is not None
    initial_x = missile.x

    # Advance 5 tics
    for _ in range(5):
        p_xy_movement(sim, missile)

    assert missile.x == initial_x + (5 * missile.momx)


def test_missile_wall_collision_explodes(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    source = spawn_mobj(sim_obj, 200 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim_obj, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    missile = spawn_missile(sim_obj, source, dest, MobjType.MT_ARROW)
    assert missile is not None

    # Step missile into East room wall at x=256
    for _ in range(10):
        p_xy_movement(sim_obj, missile)

    assert not (missile.flags & MobjFlag.MF_MISSILE)
    assert missile.momx == 0 and missile.momy == 0
    assert missile.state_num == MOBJINFO[missile.type].deathstate


def test_missile_sky_ceiling_removed_without_explosion(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256, ceiling_pic="F_SKY")
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    source = spawn_mobj(sim_obj, 200 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim_obj, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    missile = spawn_missile(sim_obj, source, dest, MobjType.MT_ARROW)
    assert missile is not None

    # Step upward into sky ceiling
    missile.momz = 10 * FRACUNIT
    missile.z = missile.ceilingz - missile.height + 1
    p_z_movement(sim_obj, missile)
    # If removed, thinker is marked removed
    assert missile.removed


def test_missile_hits_solid_shootable_actor_damages_and_explodes(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    source = spawn_mobj(sim_obj, 100 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim_obj, 150 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    initial_health = target.health

    missile = spawn_missile(sim_obj, source, target, MobjType.MT_ARROW)
    assert missile is not None

    # Step missile into target
    for _ in range(5):
        if not (missile.flags & MobjFlag.MF_MISSILE):
            break
        p_xy_movement(sim_obj, missile)

    assert target.health < initial_health
    assert not (missile.flags & MobjFlag.MF_MISSILE)
    assert missile.state_num == MOBJINFO[missile.type].deathstate


def test_missile_originator_immunity_at_spawn(sim: GameSim) -> None:
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim, 200 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)

    missile = spawn_missile(sim, source, dest, MobjType.MT_ARROW)
    assert missile is not None
    assert missile.target is source


def test_missile_same_species_monster_explodes_no_damage(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    # Monster 1 of type Centaur
    source = spawn_mobj(sim_obj, 100 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_CENTAUR)
    # Monster 2 of exact same species
    target = spawn_mobj(sim_obj, 150 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_CENTAUR)
    initial_health = target.health

    missile = spawn_missile(sim_obj, source, target, MobjType.MT_CENTAUR_FX)
    assert missile is not None

    for _ in range(5):
        if not (missile.flags & MobjFlag.MF_MISSILE):
            break
        p_xy_movement(sim_obj, missile)

    # Exploded without damaging same species
    assert not (missile.flags & MobjFlag.MF_MISSILE)
    assert target.health == initial_health


# ==============================================================================
# Suite 3: Ripper Projectiles (MF2_RIP)
# ==============================================================================


def test_ripper_missile_penetrates_single_actor_without_exploding(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    source = spawn_mobj(sim_obj, 50 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_MAGE)
    target = spawn_mobj(sim_obj, 100 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    initial_health = target.health

    # MT_RIPPERBALL has MF2_RIP
    missile = spawn_missile(sim_obj, source, target, MobjType.MT_RIPPERBALL)
    assert missile is not None
    assert missile.flags2 & MobjFlag2.MF2_RIP

    # Step through target
    for _ in range(15):
        p_xy_movement(sim_obj, missile)

    # Target took damage
    assert target.health < initial_health
    # Projectile penetrated without detonating!
    assert missile.flags & MobjFlag.MF_MISSILE
    assert missile.x > target.x


def test_ripper_missile_penetrates_multiple_actors_in_line(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    source = spawn_mobj(sim_obj, 50 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_MAGE)
    target1 = spawn_mobj(sim_obj, 100 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    target2 = spawn_mobj(sim_obj, 200 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)

    missile = spawn_missile(sim_obj, source, target1, MobjType.MT_RIPPERBALL)
    assert missile is not None

    for _ in range(30):
        p_xy_movement(sim_obj, missile)

    assert target1.health < target1.info.spawnhealth
    assert target2.health < target2.info.spawnhealth
    assert missile.flags & MobjFlag.MF_MISSILE


def test_ripper_missile_damage_formula(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=MockRandom(p_val=3))

    source = spawn_mobj(sim_obj, 50 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_MAGE)
    target = spawn_mobj(sim_obj, 80 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)

    # Controlled RNG: ((3 & 3) + 2) * damage = 5 * damage
    missile = spawn_missile(sim_obj, source, target, MobjType.MT_RIPPERBALL)
    assert missile is not None
    expected_damage = 5 * missile.damage

    # Single collision occurs during spawn check
    assert target.health == target.info.spawnhealth - expected_damage


def test_ripper_missile_spawns_ripper_blood(sim: GameSim) -> None:
    mo = spawn_mobj(sim, 0, 0, 0, MobjType.MT_RIPPERBALL)
    mo.momx = 10 * FRACUNIT
    mo.momy = 10 * FRACUNIT

    blood = spawn_ripper_blood(sim, mo)
    assert blood.type == MobjType.MT_BLOOD
    assert blood.momx == 5 * FRACUNIT
    assert blood.momy == 5 * FRACUNIT


def test_ripper_missile_non_bleeding_target_no_blood(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    source = spawn_mobj(sim_obj, 50 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_MAGE)
    target = spawn_mobj(sim_obj, 100 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_CENTAUR)
    target.flags |= MobjFlag.MF_NOBLOOD

    missile = spawn_missile(sim_obj, source, target, MobjType.MT_RIPPERBALL)
    assert missile is not None

    p_xy_movement(sim_obj, missile)
    mobjs = [th for th in sim_obj.thinkers if isinstance(th, Mobj)]
    blood = [m for m in mobjs if m.type in (MobjType.MT_BLOOD, MobjType.MT_BLOODSPLATTER)]
    assert len(blood) == 0


def test_ripper_missile_pushes_pushable_target(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=MockRandom(p_val=3))

    source = spawn_mobj(sim_obj, 50 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_MAGE)
    player = Player(player_class=PlayerClass.PCLASS_MAGE, mo=source)
    source.player = player
    sim_obj.players.append(player)

    target = spawn_mobj(sim_obj, 80 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    target.flags2 |= MobjFlag2.MF2_PUSHABLE

    missile = spawn_missile(sim_obj, source, target, MobjType.MT_RIPPERBALL)
    assert missile is not None
    assert target.momx == (missile.momx >> 2)


def test_ripper_missile_explodes_when_hitting_solid_wall(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    source = spawn_mobj(sim_obj, 200 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_MAGE)
    dest = spawn_mobj(sim_obj, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    missile = spawn_missile(sim_obj, source, dest, MobjType.MT_RIPPERBALL)
    assert missile is not None

    # Step into East wall
    for _ in range(10):
        p_xy_movement(sim_obj, missile)

    assert not (missile.flags & MobjFlag.MF_MISSILE)
    assert missile.state_num == MOBJINFO[missile.type].deathstate


# ==============================================================================
# Suite 4: Seeker Projectiles (seeker_missile, face_mobj)
# ==============================================================================


def test_face_mobj_clockwise_and_counter_clockwise(sim: GameSim) -> None:
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARROW)
    source.angle = ANG90  # North

    target_east = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    dir_turn, delta = face_mobj(source, target_east)
    # North to East (0) is counter-clockwise in math/clockwise BAM delta
    assert dir_turn in (0, 1)
    assert delta == ANG90

    target_west = spawn_mobj(sim, -100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    dir_turn_w, delta_w = face_mobj(source, target_west)
    assert delta_w in (ANG90 - 1, ANG90)
    assert dir_turn_w != dir_turn


def test_seeker_missile_turns_clockwise(sim: GameSim) -> None:
    missile = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARROW)
    missile.angle = ANG90
    target = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    missile.special1 = target

    # Target is at 45 deg, missile is at 90 deg. Shortest turn brings angle closer to 45 deg.
    ok = seeker_missile(missile, thresh=ANG1, turn_max=ANG45)
    assert ok is True
    assert missile.angle < ANG90


def test_seeker_missile_turns_counter_clockwise(sim: GameSim) -> None:
    missile = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARROW)
    missile.angle = ANG90
    target = spawn_mobj(sim, -100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    missile.special1 = target

    # Target is at 135 deg, missile at 90 deg. Heading should increase towards 135 deg.
    ok = seeker_missile(missile, thresh=ANG1, turn_max=ANG45)
    assert ok is True
    assert missile.angle > ANG90


def test_seeker_missile_max_turn_clamping(sim: GameSim) -> None:
    missile = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARROW)
    missile.angle = ANG90
    target = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    missile.special1 = target

    # Turn max clamped to 5 deg
    turn_limit = ANG1 * 5
    seeker_missile(missile, thresh=ANG1, turn_max=turn_limit)
    assert missile.angle == ANG90 - turn_limit


def test_seeker_missile_within_threshold_turns_full_delta(sim: GameSim) -> None:
    missile = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARROW)
    missile.angle = ANG90
    # Place target at ~89 deg (delta ~ 1 deg)
    target = spawn_mobj(sim, 100 * FRACUNIT, 5000 * FRACUNIT, 0, MobjType.MT_ETTIN)
    missile.special1 = target

    # With threshold 45 deg, delta is within threshold so turns full delta
    seeker_missile(missile, thresh=ANG45, turn_max=ANG45)
    assert missile.angle < ANG90


def test_seeker_missile_updates_momentum_components(sim: GameSim) -> None:
    missile = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARROW)
    missile.angle = ANG90
    target = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    missile.special1 = target

    seeker_missile(missile, thresh=ANG1, turn_max=ANG45)
    fine = (missile.angle >> ANGLETOFINESHIFT) & FINEMASK
    assert missile.momx == fixed_mul(missile.info.speed, fine_cosine(fine))
    assert missile.momy == fixed_mul(missile.info.speed, fine_sine(fine))


def test_seeker_missile_vertical_seeking(sim: GameSim) -> None:
    missile = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARROW)
    missile.height = 16 * FRACUNIT
    target = spawn_mobj(sim, 100 * FRACUNIT, 0, 100 * FRACUNIT, MobjType.MT_ETTIN)
    missile.special1 = target

    seeker_missile(missile, thresh=ANG1, turn_max=ANG45)
    assert missile.momz > 0


def test_seeker_missile_dead_target_aborts_and_clears(sim: GameSim) -> None:
    missile = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARROW)
    target = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    target.flags &= ~MobjFlag.MF_SHOOTABLE
    missile.special1 = target

    ok = seeker_missile(missile, thresh=ANG1, turn_max=ANG45)
    assert ok is False
    assert missile.special1 is None


def test_seeker_missile_none_target_returns_false(sim: GameSim) -> None:
    missile = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARROW)
    missile.special1 = None
    ok = seeker_missile(missile, thresh=ANG1, turn_max=ANG45)
    assert ok is False


# ==============================================================================
# Suite 5: Radius Damage (radius_attack)
# ==============================================================================


def test_radius_attack_damage_falloff_exact_values(sim: GameSim) -> None:
    spot = spawn_mobj(sim, 0, 0, 0, MobjType.MT_CIRCLEFLAME)
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    # Target monster radius = 24
    target = spawn_mobj(sim, 50 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    target.radius = 24 * FRACUNIT
    initial_health = target.health

    # Damage = 100, distance = 100
    # dist = (50 - 24) = 26
    # damage = (100 * (100 - 26) // 100) + 1 = 75
    radius_attack(sim, spot=spot, source=source, damage=100, distance=100)
    assert target.health == initial_health - 75


def test_radius_attack_chebyshev_distance(sim: GameSim) -> None:
    spot = spawn_mobj(sim, 0, 0, 0, MobjType.MT_CIRCLEFLAME)
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    # Chebyshev: max(dx, dy)
    target = spawn_mobj(sim, 30 * FRACUNIT, 60 * FRACUNIT, 0, MobjType.MT_ETTIN)
    target.radius = 20 * FRACUNIT
    initial_health = target.health

    # max(30, 60) = 60; dist = 60 - 20 = 40
    # damage = (100 * (100 - 40) // 100) + 1 = 61
    radius_attack(sim, spot=spot, source=source, damage=100, distance=100)
    assert target.health == initial_health - 61


def test_radius_attack_out_of_range_zero_damage(sim: GameSim) -> None:
    spot = spawn_mobj(sim, 0, 0, 0, MobjType.MT_CIRCLEFLAME)
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, 150 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    target.radius = 20 * FRACUNIT
    initial_health = target.health

    radius_attack(sim, spot=spot, source=source, damage=100, distance=100)
    assert target.health == initial_health


def test_radius_attack_source_immunity_by_default(sim: GameSim) -> None:
    spot = spawn_mobj(sim, 0, 0, 0, MobjType.MT_CIRCLEFLAME)
    source = spawn_mobj(sim, 10 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    initial_health = source.health

    radius_attack(sim, spot=spot, source=source, damage=100, distance=100, damage_source=False)
    assert source.health == initial_health


def test_radius_attack_source_damaged_when_flag_true(sim: GameSim) -> None:
    spot = spawn_mobj(sim, 0, 0, 0, MobjType.MT_CIRCLEFLAME)
    source = spawn_mobj(sim, 10 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    initial_health = source.health

    radius_attack(sim, spot=spot, source=source, damage=100, distance=100, damage_source=True)
    assert source.health < initial_health


def test_radius_attack_player_quarter_damage(sim: GameSim) -> None:
    spot = spawn_mobj(sim, 0, 0, 0, MobjType.MT_CIRCLEFLAME)
    player, p_mo = make_player(sim, PlayerClass.PCLASS_FIGHTER)
    p_mo.x = 0
    p_mo.y = 0
    initial_health = player.health

    # Point blank: raw damage = (100 * 100 // 100) + 1 = 101
    # Player reduction: 101 >> 2 = 25
    # Fighter passive armor absorbs 15% (3 damage) -> 22 damage applied
    radius_attack(sim, spot=spot, source=None, damage=100, distance=100)
    assert player.health == initial_health - 22


def test_radius_attack_z_axis_culling(sim: GameSim) -> None:
    spot = spawn_mobj(sim, 0, 0, 0, MobjType.MT_CIRCLEFLAME)
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    # Z difference is 250 > 2 * distance (200)
    target = spawn_mobj(sim, 10 * FRACUNIT, 0, 250 * FRACUNIT, MobjType.MT_ETTIN)
    initial_health = target.health

    radius_attack(sim, spot=spot, source=source, damage=100, distance=100)
    assert target.health == initial_health


def test_radius_attack_blocked_by_wall_los(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    spot = spawn_mobj(sim_obj, 50 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CIRCLEFLAME)
    target = spawn_mobj(sim_obj, -50 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    initial_health = target.health

    # Target is separated by solid 1-sided room wall
    radius_attack(sim_obj, spot=spot, source=None, damage=100, distance=200)
    assert target.health == initial_health


def test_radius_attack_unblocked_across_portal(tmp_path: Path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=256,
        width2=256,
        height=256,
        floor1=0,
        ceil1=128,
        floor2=0,
        ceil2=128,
    )
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    spot = spawn_mobj(sim_obj, 200 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CIRCLEFLAME)
    target = spawn_mobj(sim_obj, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    initial_health = target.health

    # Open portal allows splash damage through
    radius_attack(sim_obj, spot=spot, source=None, damage=100, distance=200)
    assert target.health < initial_health


def test_radius_attack_ignores_non_shootable_corpses(sim: GameSim) -> None:
    spot = spawn_mobj(sim, 0, 0, 0, MobjType.MT_CIRCLEFLAME)
    target = spawn_mobj(sim, 20 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    target.flags &= ~MobjFlag.MF_SHOOTABLE
    initial_health = target.health

    radius_attack(sim, spot=spot, source=None, damage=100, distance=100)
    assert target.health == initial_health


# ==============================================================================
# Suite 6: Determinism, Quality & Layering Constraints
# ==============================================================================


def test_attacks_determinism_identical_rng_seeds() -> None:
    def run_simulation() -> list[tuple[int, int, int]]:
        sim_obj = GameSim(rng=Random())
        sim_obj.rng.clear()
        source = spawn_mobj(sim_obj, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
        target = spawn_mobj(sim_obj, 200 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
        target.flags |= MobjFlag.MF_SHADOW

        missile = spawn_missile(sim_obj, source, target, MobjType.MT_ARROW)
        assert missile is not None

        positions = []
        for _ in range(5):
            p_xy_movement(sim_obj, missile)
            positions.append((missile.x, missile.y, missile.z))
        return positions

    run1 = run_simulation()
    run2 = run_simulation()
    assert run1 == run2


def test_attacks_no_floats_in_simulation() -> None:
    source_path = Path(__file__).parents[2] / "src" / "pyhexen" / "sim" / "attacks.py"
    with open(source_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(source_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            pytest.fail(f"Disallowed float constant {node.value} at line {node.lineno}")


def test_attacks_no_render_audio_ui_imports() -> None:
    source_path = Path(__file__).parents[2] / "src" / "pyhexen" / "sim" / "attacks.py"
    with open(source_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(source_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in ("render", "audio", "ui"):
                    if forbidden in alias.name.split("."):
                        pytest.fail(f"Forbidden import '{alias.name}' at line {node.lineno}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            for forbidden in ("render", "audio", "ui"):
                if forbidden in node.module.split("."):
                    pytest.fail(f"Forbidden import from '{node.module}' at line {node.lineno}")
