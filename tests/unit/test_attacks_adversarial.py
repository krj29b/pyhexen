# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Adversarial stress-testing suite for combat attacks.

Empirical verification of hitscan pitch corridor, extreme coordinates, narrow portals,
line occlusion, bleed/invulnerability flags, zero-distance projectile spawning,
and ripper multi-target penetration.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.angles import (
    ANG45,
    ANG90,
    ANG180,
    ANG270,
)
from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.sim.attacks import (
    TOPSLOPE,
    aim_line_attack,
    line_attack,
    radius_attack,
    spawn_missile,
)
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import (
    MobjFlag,
    MobjFlag2,
    MobjType,
)
from pyhexen.sim.mobj import Mobj, spawn_mobj
from pyhexen.sim.movement import p_xy_movement
from pyhexen.sim.player import Player, PlayerClass
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


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
    def __init__(self, p_val: int = 0, sub_val: int = 0) -> None:
        super().__init__()
        self._p_val = p_val
        self._sub_val = sub_val

    def p_random(self) -> int:
        return self._p_val

    def p_subrandom(self) -> int:
        return self._sub_val


# ==============================================================================
# 1. Hitscan & Aim Line Attack Adversarial Tests
# ==============================================================================


def test_aim_line_attack_steep_cardinal_angles(sim: GameSim) -> None:
    """Test aim_line_attack at all steep cardinal angles (90, 180, 270)."""
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)

    # North (ANG90)
    target_north = spawn_mobj(sim, 0, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    aim_line_attack(sim, source, ANG90, 200 * FRACUNIT)
    assert sim.linetarget is target_north

    # South (ANG270)
    sim.linetarget = None
    target_south = spawn_mobj(sim, 0, -100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    aim_line_attack(sim, source, ANG270, 200 * FRACUNIT)
    assert sim.linetarget is target_south

    # West (ANG180)
    sim.linetarget = None
    target_west = spawn_mobj(sim, -100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    aim_line_attack(sim, source, ANG180, 200 * FRACUNIT)
    assert sim.linetarget is target_west

    # Diagonal (ANG45)
    sim.linetarget = None
    d = int(70.7106 * FRACUNIT)
    target_diag = spawn_mobj(sim, d, d, 0, MobjType.MT_ETTIN)
    aim_line_attack(sim, source, ANG45, 200 * FRACUNIT)
    assert sim.linetarget is target_diag


def test_aim_line_attack_extreme_distances(sim: GameSim) -> None:
    """Test distance 0, 1 FRACUNIT, and extreme map scale (16384 FRACUNIT)."""
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, 50 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)

    # Distance 0: should not reach target, returns 0 and linetarget is None
    slope = aim_line_attack(sim, source, 0, 0)
    assert slope == 0
    assert sim.linetarget is None

    # Distance 1: does not reach target
    slope = aim_line_attack(sim, source, 0, 1)
    assert slope == 0
    assert sim.linetarget is None

    # Distance 16384 * FRACUNIT: large valid 32-bit fixed-point range
    slope = aim_line_attack(sim, source, 0, 16384 * FRACUNIT)
    assert sim.linetarget is target
    assert slope != 0


def test_aim_line_attack_negative_coordinates(sim: GameSim) -> None:
    """Test raycasting across large negative coordinate spaces."""
    source = spawn_mobj(sim, -50000 * FRACUNIT, -50000 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    target = spawn_mobj(sim, -49900 * FRACUNIT, -50000 * FRACUNIT, 0, MobjType.MT_ETTIN)

    slope = aim_line_attack(sim, source, 0, 200 * FRACUNIT)
    assert sim.linetarget is target
    assert slope != 0 or sim.linetarget is not None


def test_aim_line_attack_pitch_corridor_boundaries(sim: GameSim) -> None:
    """Test targets placed right at TOPSLOPE (40960) and BOTTOMSLOPE (-40960) edges."""
    dist = 100 * FRACUNIT
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    shootz = source.z + (source.height >> 1) + 8 * FRACUNIT

    # 1. Target right inside upper pitch corridor
    target_high = spawn_mobj(sim, dist, 0, 0, MobjType.MT_ETTIN)
    target_high.z = shootz + 4096000 - target_high.height
    slope = aim_line_attack(sim, source, 0, dist * 2)
    assert sim.linetarget is target_high
    assert slope <= TOPSLOPE

    # 2. Target strictly ABOVE upper pitch corridor
    target_too_high = spawn_mobj(sim, dist, 0, shootz + 4096001 + 10 * FRACUNIT, MobjType.MT_ETTIN)
    target_high.z = -1000 * FRACUNIT  # move previous target out of way
    slope = aim_line_attack(sim, source, 0, dist * 2)
    assert sim.linetarget != target_too_high

    # 3. Target strictly BELOW lower pitch corridor (BOTTOMSLOPE = -40960)
    target_too_low = spawn_mobj(
        sim, dist, 0, shootz - 4096000 - target_high.height - 100 * FRACUNIT, MobjType.MT_ETTIN
    )
    slope = aim_line_attack(sim, source, 0, dist * 2)
    assert sim.linetarget != target_too_low


def test_aim_line_attack_narrow_vertical_portal_constriction(tmp_path: Path) -> None:
    """Test narrow 2-sided portal opening constricts pitch corridor and blocks occluded targets."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=256,
        width2=256,
        height=512,
        floor1=0,
        ceil1=128,
        floor2=40,
        ceil2=60,
    )
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    # Player at x=100, y=256, z=0 (shootz = 32 * FRACUNIT)
    source = spawn_mobj(sim_obj, 100 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    # 1. Target inside the window slit (z=40)
    target_in_slit = spawn_mobj(
        sim_obj, 350 * FRACUNIT, 256 * FRACUNIT, 40 * FRACUNIT, MobjType.MT_ETTIN
    )
    aim_line_attack(sim_obj, source, 0, 500 * FRACUNIT)
    assert sim_obj.linetarget is target_in_slit

    # 2. Target above the window slit (z=80 in a closed-ceiling portion)
    target_in_slit.z = -1000 * FRACUNIT
    spawn_mobj(sim_obj, 350 * FRACUNIT, 256 * FRACUNIT, 80 * FRACUNIT, MobjType.MT_ETTIN)
    slope_blocked = aim_line_attack(sim_obj, source, 0, 500 * FRACUNIT)
    assert slope_blocked == 0
    assert sim_obj.linetarget is None


def test_aim_line_attack_closed_door_blocks(tmp_path: Path) -> None:
    """Test that a 2-sided portal with openbottom >= opentop completely blocks aiming."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=256,
        width2=256,
        height=512,
        floor1=0,
        ceil1=128,
        floor2=40,
        ceil2=20,  # closed door: ceiling lower than floor
    )
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=Random())

    source = spawn_mobj(sim_obj, 100 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    spawn_mobj(sim_obj, 350 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)

    slope = aim_line_attack(sim_obj, source, 0, 500 * FRACUNIT)
    assert slope == 0
    assert sim_obj.linetarget is None


def test_line_attack_target_behind_1sided_line_vs_2sided_line(tmp_path: Path) -> None:
    """Adversarially verify 1-sided line blocks line_attack and clears linetarget,

    while open 2-sided line allows hitscan penetration and inflicts damage.
    """
    # 1. 1-sided line separation
    mb1 = MapBuilder("MAP01")
    mb1.add_room(x=0, y=0, width=256, height=512)
    mb1.add_room(x=300, y=0, width=256, height=512)
    level1 = _build_level(mb1, tmp_path)
    sim1 = GameSim(level=level1, rng=Random())

    source1 = spawn_mobj(sim1, 100 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    target1 = spawn_mobj(sim1, 350 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    initial_health1 = target1.health

    line_attack(sim1, source1, angle=0, distance=500 * FRACUNIT, slope=0, damage=50)
    assert target1.health == initial_health1
    assert sim1.linetarget is None

    # 2. 2-sided portal connecting rooms
    mb2 = MapBuilder("MAP02")
    mb2.add_two_sector_rooms(
        x=0,
        y=0,
        width1=256,
        width2=256,
        height=512,
        floor1=0,
        ceil1=128,
        floor2=0,
        ceil2=128,
    )
    level2 = _build_level(mb2, tmp_path)
    sim2 = GameSim(level=level2, rng=Random())

    source2 = spawn_mobj(sim2, 100 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    target2 = spawn_mobj(sim2, 350 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    initial_health2 = target2.health

    line_attack(sim2, source2, angle=0, distance=500 * FRACUNIT, slope=0, damage=50)
    assert target2.health < initial_health2
    assert sim2.linetarget is target2


def test_line_attack_puff_vs_blood_bleed_and_invulnerable_flags() -> None:
    """Verify puff is always spawned, but blood splatters only spawn if bleedable."""
    # 1. MF_NOBLOOD target
    sim1 = GameSim(rng=Random())
    source1 = spawn_mobj(sim1, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target_noblood = spawn_mobj(sim1, 100 * FRACUNIT, 0, 0, MobjType.MT_CENTAUR)
    target_noblood.flags |= MobjFlag.MF_NOBLOOD

    line_attack(sim1, source1, angle=0, distance=200 * FRACUNIT, slope=0, damage=20)
    mobjs1 = [th for th in sim1.thinkers if isinstance(th, Mobj)]
    puffs1 = [m for m in mobjs1 if m.type == MobjType.MT_PUNCHPUFF]
    blood1 = [m for m in mobjs1 if m.type in (MobjType.MT_BLOODSPLATTER, MobjType.MT_AXEBLOOD)]
    assert len(puffs1) >= 1
    assert len(blood1) == 0

    # 2. MF2_INVULNERABLE target
    sim2 = GameSim(rng=Random())
    source2 = spawn_mobj(sim2, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    target_invuln = spawn_mobj(sim2, 100 * FRACUNIT, 0, 0, MobjType.MT_CENTAUR)
    target_invuln.flags &= ~MobjFlag.MF_NOBLOOD
    target_invuln.flags2 |= MobjFlag2.MF2_INVULNERABLE

    line_attack(sim2, source2, angle=0, distance=200 * FRACUNIT, slope=0, damage=20)
    mobjs2 = [th for th in sim2.thinkers if isinstance(th, Mobj)]
    puffs2 = [m for m in mobjs2 if m.type == MobjType.MT_PUNCHPUFF]
    blood2 = [m for m in mobjs2 if m.type in (MobjType.MT_BLOODSPLATTER, MobjType.MT_AXEBLOOD)]
    assert len(puffs2) >= 1
    assert len(blood2) == 0

    # 3. Normal bleeding target with MT_AXEPUFF
    sim3 = GameSim(rng=MockRandom(p_val=100))  # p_random < 192
    source3 = spawn_mobj(sim3, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    spawn_mobj(sim3, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)

    line_attack(
        sim3,
        source3,
        angle=0,
        distance=200 * FRACUNIT,
        slope=0,
        damage=20,
        puff_type=MobjType.MT_AXEPUFF,
    )
    mobjs3 = [th for th in sim3.thinkers if isinstance(th, Mobj)]
    axe_blood = [m for m in mobjs3 if m.type == MobjType.MT_AXEBLOOD]
    splatter = [m for m in mobjs3 if m.type == MobjType.MT_BLOODSPLATTER]
    assert len(axe_blood) == 1
    assert len(splatter) == 1


# ==============================================================================
# 2. Projectile Spawning & Ripper Penetration Adversarial Tests
# ==============================================================================


def test_spawn_missile_identical_coordinates_zero_distance(sim: GameSim) -> None:
    """Stress test spawn_missile when source and dest are at the exact same coordinates."""
    source = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)

    missile = spawn_missile(sim, source, dest, MobjType.MT_CENTAUR_FX)
    assert missile is not None
    assert missile.momx != 0 or missile.momy != 0
    assert missile.momz == 0  # dest.z == source.z


def test_spawn_missile_sub_unit_distance(sim: GameSim) -> None:
    """Stress test spawn_missile when source and dest distance is less than 1 FRACUNIT."""
    source = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(
        sim, 100 * FRACUNIT + 10, 100 * FRACUNIT + 10, 50 * FRACUNIT, MobjType.MT_CENTAUR_FX
    )

    missile = spawn_missile(sim, source, dest, MobjType.MT_CENTAUR_FX)
    assert missile is not None
    assert missile.momz != 0


def test_spawn_missile_speed_zero_fallback(sim: GameSim) -> None:
    """Verify that a missile type with speed=0 does not cause ZeroDivisionError in spawn_missile."""
    source = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    dest = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)

    # MT_BLOOD has speed 0
    missile = spawn_missile(sim, source, dest, MobjType.MT_BLOOD)
    assert missile is not None
    assert missile.momx == 0
    assert missile.momy == 0


def test_ripper_penetrates_multiple_actors_in_single_step(tmp_path: Path) -> None:
    """Verify multiple lined-up actors take damage from a ripper in a single move step."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=MockRandom(p_val=2))

    source = spawn_mobj(sim_obj, 50 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_MAGE)
    player = Player(player_class=PlayerClass.PCLASS_MAGE, mo=source)
    source.player = player
    sim_obj.players.append(player)

    # Place 3 Ettins lined up within one move step (within MAXMOVE = 30 * FRACUNIT)
    e1 = spawn_mobj(sim_obj, 62 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    e2 = spawn_mobj(sim_obj, 72 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    e3 = spawn_mobj(sim_obj, 80 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)

    h1_init, h2_init, h3_init = e1.health, e2.health, e3.health

    # Spawn Mage ripper ball MT_RIPPERBALL
    missile = spawn_missile(sim_obj, source, e1, MobjType.MT_RIPPERBALL)
    assert missile is not None

    # Set velocity so that one move step covers all 3 actors (29 FRACUNIT <= MAXMOVE)
    missile.momx = 29 * FRACUNIT
    missile.momy = 0

    # Advance 1 step of XY movement
    p_xy_movement(sim_obj, missile)

    # 1. Verify all 3 actors received damage
    assert e1.health < h1_init, f"Ettin 1 did not take damage: {e1.health} vs {h1_init}"
    assert e2.health < h2_init, f"Ettin 2 did not take damage: {e2.health} vs {h2_init}"
    assert e3.health < h3_init, f"Ettin 3 did not take damage: {e3.health} vs {h3_init}"

    # 2. Verify ripper missile did not explode or lose MF_MISSILE
    assert missile.flags & MobjFlag.MF_MISSILE
    assert missile.x >= 80 * FRACUNIT


def test_ripper_penetration_with_mixed_bleeding_and_nonbleeding(tmp_path: Path) -> None:
    """Verify ripper penetrating non-bleeding and bleeding actors damages both and stays alive."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    level = _build_level(mb, tmp_path)
    sim_obj = GameSim(level=level, rng=MockRandom(p_val=2))

    source = spawn_mobj(sim_obj, 50 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_MAGE)

    # e1: non-bleeding
    e1 = spawn_mobj(sim_obj, 65 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    e1.flags |= MobjFlag.MF_NOBLOOD

    # e2: normal bleeding
    e2 = spawn_mobj(sim_obj, 78 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)

    missile = spawn_missile(sim_obj, source, e1, MobjType.MT_RIPPERBALL)
    assert missile is not None

    missile.momx = 28 * FRACUNIT
    missile.momy = 0

    p_xy_movement(sim_obj, missile)

    assert e1.health < e1.info.spawnhealth
    assert e2.health < e2.info.spawnhealth
    assert missile.flags & MobjFlag.MF_MISSILE

    mobjs = [th for th in sim_obj.thinkers if isinstance(th, Mobj)]
    blood = [m for m in mobjs if m.type in (MobjType.MT_BLOOD, MobjType.MT_BLOODSPLATTER)]
    # Blood spawned for e2
    assert len(blood) > 0


# ==============================================================================
# 3. Radius Damage Adversarial Tests
# ==============================================================================


def test_radius_attack_zero_and_negative_distance(sim: GameSim) -> None:
    """Verify radius_attack with distance <= 0 does not crash or loop infinitely."""
    spot = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ETTIN)
    target = spawn_mobj(sim, 10 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    initial_health = target.health

    radius_attack(sim, spot, source=None, damage=50, distance=0)
    assert target.health == initial_health

    radius_attack(sim, spot, source=None, damage=50, distance=-10)
    assert target.health == initial_health


def test_radius_attack_exact_chebyshev_boundary(sim: GameSim) -> None:
    """Verify targets right on the Chebyshev distance boundary."""
    spot = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ETTIN)
    # Target radius is 25 * FRACUNIT (Ettin)
    # dist = (chebyshev - thing.radius) >> FRACBITS
    # if dist >= distance: excluded
    # Chebyshev distance for target placed at x = 125 * FRACUNIT:
    # dist = (125*FRACUNIT - 25*FRACUNIT) >> 16 = 100 == distance -> should NOT take damage
    target_outside = spawn_mobj(sim, 125 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    h_out = target_outside.health

    # Target placed at x = 124 * FRACUNIT:
    # dist = (124*FRACUNIT - 25*FRACUNIT) >> 16 = 99 < 100 -> takes minimum damage
    target_inside = spawn_mobj(sim, 124 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    h_in = target_inside.health

    radius_attack(sim, spot, source=None, damage=100, distance=100)
    assert target_outside.health == h_out
    assert target_inside.health < h_in
