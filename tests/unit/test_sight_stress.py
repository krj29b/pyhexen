# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Adversarial stress test harness for Task T10.3 (Sight & Map Traversal).

Covers:
1. Blockmap DDA Grid Traversal Stress:
   - 360-degree sweep in open circular arena (all return True).
   - 1024 BAM fine angles (no crashes, no divisions by zero, all unobstructed).
   - 360-degree occluded sweep with wall ring (all return False).
   - Rays crossing blockmap boundaries, corners, and multiples of 128 (+1 FRACUNIT snapping).
   - Long rays (> 64 blocks) verifying reference loop termination (p_sight.c count < 64).
   - Zero-length rays (same x/y different z, same x/y same z, same mobj).
2. Multi-Sector Synthetic Map Invariants:
   - S-shaped / zigzag corridor with alternating 1-sided and 2-sided linedefs.
   - Dynamic sector doors (lowering floor, raising ceiling) flipping check_sight dynamically.
   - Precedence invariants (RejectMatrix overrides open geometry; 1-sided wall blocks even
     with 0 reject).
3. Combat Integration Invariants:
   - radius_attack splash damage with multiple mobjs simultaneously (exposed vs wall vs
     closed door vs high step).
   - radius_attack splash damage penetration through open doors and window slits vs
     occlusion below sill.
   - radius_attack dynamic door transition (blast 1 blocked, door opens, blast 2 damages).
"""

from __future__ import annotations

import math

from tests.fixtures.wadbuilder import ML_BLOCKING, ML_TWOSIDED, MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.sim import check_sight
from pyhexen.sim.attacks import radius_attack
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjType
from pyhexen.sim.mobj import spawn_mobj
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def _make_sim(mb: MapBuilder) -> GameSim:
    """Build synthetic level and initialize GameSim."""
    stack = WadStack(WadBuilder("IWAD").add_map(mb).to_bytes())
    level = load_level(stack, mb.name)
    return GameSim(level=level, rng=Random())


# ==============================================================================
# Suite 1: Blockmap DDA Grid Traversal Stress
# ==============================================================================


def test_dda_all_360_degrees_unobstructed() -> None:
    """Ray angles across all 360 degrees return True in an open circular arena."""
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 1024, 1024)
    sim = _make_sim(mb)

    viewer = spawn_mobj(sim, 512 * FRACUNIT, 512 * FRACUNIT, 0, MobjType.MT_ETTIN)
    target = spawn_mobj(sim, 512 * FRACUNIT, 512 * FRACUNIT, 0, MobjType.MT_ETTIN)

    radius = 250
    for deg in range(360):
        rad = math.radians(deg)
        target.x = int((512 + radius * math.cos(rad)) * FRACUNIT)
        target.y = int((512 + radius * math.sin(rad)) * FRACUNIT)
        assert check_sight(sim, viewer, target), f"Sight failed at angle {deg} deg"


def test_dda_bam_fine_angles_sweep() -> None:
    """1024 BAM fine angles around center execute without crash or division by zero."""
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 1024, 1024)
    sim = _make_sim(mb)

    viewer = spawn_mobj(sim, 512 * FRACUNIT, 512 * FRACUNIT, 0, MobjType.MT_ETTIN)
    target = spawn_mobj(sim, 512 * FRACUNIT, 512 * FRACUNIT, 0, MobjType.MT_ETTIN)

    radius = 200
    for i in range(1024):
        bam = (i * 0x100000000) // 1024
        rad = (bam / 0x100000000) * 2 * math.pi
        target.x = int((512 + radius * math.cos(rad)) * FRACUNIT)
        target.y = int((512 + radius * math.sin(rad)) * FRACUNIT)
        assert check_sight(sim, viewer, target), f"Sight failed at BAM angle index {i}"


def test_dda_all_360_degrees_occluded_by_wall_ring() -> None:
    """Rays across all 360 degrees hitting surrounding walls return False."""
    # Room is [0, 1024] x [0, 1024] with center at (512, 512).
    # Room walls are at dist 512 along cardinal directions.
    # At non-diagonal angles where target is placed outside the room (radius 600..750),
    # all rays crossing 1-sided walls return False.
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 1024, 1024)
    sim = _make_sim(mb)

    viewer = spawn_mobj(sim, 500 * FRACUNIT, 512 * FRACUNIT, 0, MobjType.MT_ETTIN)
    target = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ETTIN)

    # Test non-diagonal angles (avoiding the historical Carmack 45-deg corner stall)
    # Target placed at radius 750 (outside room)
    occluded_count = 0
    for deg in range(360):
        if deg in (44, 45, 46, 134, 135, 136, 224, 225, 226, 314, 315, 316):
            continue  # Near corner grazing angles
        rad = math.radians(deg)
        target.x = int((500 + 750 * math.cos(rad)) * FRACUNIT)
        target.y = int((512 + 750 * math.sin(rad)) * FRACUNIT)
        assert not check_sight(sim, viewer, target), f"Failed to occlude at {deg} deg"
        occluded_count += 1

    assert occluded_count == 348


def test_dda_exact_grid_multiples_128_snapping() -> None:
    """Rays starting exactly on block boundaries (+1 FRACUNIT snapping) traverse cleanly."""
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 1024, 1024)
    sim = _make_sim(mb)

    # (256, 256) is an exact corner of 4 blockmap blocks (256 = 2 * 128)
    viewer = spawn_mobj(sim, 256 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    target = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ETTIN)

    # Test 4 cardinal directions and offsets
    offsets = [
        (100, 0),  # East
        (-100, 0),  # West
        (0, 100),  # North
        (0, -100),  # South
        (80, 50),  # Diagonal
        (-80, -50),
    ]
    for dx, dy in offsets:
        target.x = (256 + dx) * FRACUNIT
        target.y = (256 + dy) * FRACUNIT
        assert check_sight(sim, viewer, target), f"Grid snapping failed for offset ({dx}, {dy})"


def test_dda_long_rays_64_block_termination() -> None:
    """Empirically verifies 64-block loop termination matching Chocolate Doom p_sight.c.

    In Chocolate Doom reference p_sight.c (line 324):
      `for (count = 0; count < 64; count++)`
    Rays traversing > 64 blocks terminate blockmap stepping at 64 iterations:
    1. Obstacle within first 64 blocks is checked and blocks sight (returns False).
    2. Obstacle beyond 64 blocks is never visited during the 64 steps (returns True).
    """
    # 1. Obstacle at block 20 (within 64 blocks)
    mb1 = MapBuilder("MAP01")
    mb1.add_room(0, 0, 2560, 512)
    mb1.add_room(2560, 0, 9440, 512)
    sim1 = _make_sim(mb1)

    t1_1 = spawn_mobj(sim1, 200 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t2_1 = spawn_mobj(sim1, 9600 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert not check_sight(sim1, t1_1, t2_1), "Wall at block 20 must block sight"

    # 2. Obstacle at block 70 (> 64 blocks away)
    mb2 = MapBuilder("MAP01")
    mb2.add_room(0, 0, 8960, 512)
    mb2.add_room(8960, 0, 3040, 512)
    sim2 = _make_sim(mb2)

    t1_2 = spawn_mobj(sim2, 200 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t2_2 = spawn_mobj(sim2, 9600 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    # Loop terminates after 64 blocks without inspecting block 70, returning True
    assert check_sight(sim2, t1_2, t2_2), "64-block loop limit terminates and returns True"


def test_dda_zero_length_rays() -> None:
    """Zero-length rays (same x, y, same/different z) return True cleanly."""
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 256, 256)
    sim = _make_sim(mb)

    t0 = spawn_mobj(sim, 128 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 128 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t2 = spawn_mobj(sim, 128 * FRACUNIT, 128 * FRACUNIT, 64 * FRACUNIT, MobjType.MT_ETTIN)

    # Identical mobj instance
    assert check_sight(sim, t0, t0)
    # Same (x, y), same z
    assert check_sight(sim, t0, t1)
    # Same (x, y), different z
    assert check_sight(sim, t0, t2)
    assert check_sight(sim, t2, t0)


# ==============================================================================
# Suite 2: Multi-Sector Synthetic Map Invariants
# ==============================================================================


def _build_zigzag_map() -> MapBuilder:
    """Build an S-shaped / zigzag corridor with alternating 1-sided and 2-sided lines."""
    mb = MapBuilder("MAP01")
    s0 = mb.add_sector(0, 128)  # Sec 0: East corridor [0, 256] x [0, 128]
    s1 = mb.add_sector(0, 128)  # Sec 1: North corridor [128, 256] x [128, 384]
    s2 = mb.add_sector(0, 128)  # Sec 2: East corridor [256, 512] x [256, 384]

    v0 = mb.add_vertex(0, 128)
    v1 = mb.add_vertex(128, 128)
    v2 = mb.add_vertex(128, 384)
    v3 = mb.add_vertex(512, 384)
    v4 = mb.add_vertex(512, 256)
    v5 = mb.add_vertex(256, 256)
    v6 = mb.add_vertex(256, 0)
    v7 = mb.add_vertex(0, 0)

    v_p1_r = mb.add_vertex(256, 128)
    v_p2_t = mb.add_vertex(256, 384)

    sd_s0 = mb.add_sidedef(s0)
    sd_s1 = mb.add_sidedef(s1)
    sd_s2 = mb.add_sidedef(s2)

    sd_p1_0 = mb.add_sidedef(s0)
    sd_p1_1 = mb.add_sidedef(s1)
    sd_p2_1 = mb.add_sidedef(s1)
    sd_p2_2 = mb.add_sidedef(s2)

    # 1-sided walls for Sec 0
    l_s0_top = mb.add_linedef(v0, v1, sd_s0, flags=ML_BLOCKING)
    l_s0_left = mb.add_linedef(v7, v0, sd_s0, flags=ML_BLOCKING)
    l_s0_bot = mb.add_linedef(v6, v7, sd_s0, flags=ML_BLOCKING)
    l_s0_right = mb.add_linedef(v_p1_r, v6, sd_s0, flags=ML_BLOCKING)

    # 2-sided Portal 1 (v1 to v_p1_r)
    l_p1 = mb.add_linedef(v1, v_p1_r, sd_p1_0, sd_p1_1, flags=ML_TWOSIDED)

    # 1-sided walls for Sec 1
    l_s1_left = mb.add_linedef(v1, v2, sd_s1, flags=ML_BLOCKING)
    l_s1_top = mb.add_linedef(v2, v_p2_t, sd_s1, flags=ML_BLOCKING)
    l_s1_bot = mb.add_linedef(v_p1_r, v5, sd_s1, flags=ML_BLOCKING)

    # 2-sided Portal 2 (v_p2_t to v5)
    l_p2 = mb.add_linedef(v_p2_t, v5, sd_p2_1, sd_p2_2, flags=ML_TWOSIDED)

    # 1-sided walls for Sec 2
    l_s2_top = mb.add_linedef(v_p2_t, v3, sd_s2, flags=ML_BLOCKING)
    l_s2_right = mb.add_linedef(v3, v4, sd_s2, flags=ML_BLOCKING)
    l_s2_bot = mb.add_linedef(v4, v5, sd_s2, flags=ML_BLOCKING)

    # Subsectors
    mb.add_seg(v0, v1, 0, l_s0_top)
    mb.add_seg(v1, v_p1_r, 0, l_p1, side=0)
    mb.add_seg(v_p1_r, v6, -16384, l_s0_right)
    mb.add_seg(v6, v7, 32768, l_s0_bot)
    mb.add_seg(v7, v0, 16384, l_s0_left)
    mb.add_subsector(5, 0, bbox=(128, 0, 0, 256))

    mb.add_seg(v1, v2, 16384, l_s1_left)
    mb.add_seg(v2, v_p2_t, 0, l_s1_top)
    mb.add_seg(v_p2_t, v5, -16384, l_p2, side=0)
    mb.add_seg(v5, v_p1_r, -16384, l_s1_bot)
    mb.add_seg(v_p1_r, v1, 32768, l_p1, side=1)
    mb.add_subsector(5, 5, bbox=(384, 128, 128, 256))

    mb.add_seg(v_p2_t, v3, 0, l_s2_top)
    mb.add_seg(v3, v4, -16384, l_s2_right)
    mb.add_seg(v4, v5, 32768, l_s2_bot)
    mb.add_seg(v5, v_p2_t, 16384, l_p2, side=1)
    mb.add_subsector(4, 10, bbox=(384, 256, 256, 512))

    return mb


def test_multisector_zigzag_corridor_invariants() -> None:
    """Stress-test sightlines in an S-shaped zigzag corridor."""
    mb = _build_zigzag_map()
    sim = _make_sim(mb)

    # 1. Direct line within Room 0
    t0_start = spawn_mobj(sim, 50 * FRACUNIT, 64 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t0_end = spawn_mobj(sim, 200 * FRACUNIT, 64 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert check_sight(sim, t0_start, t0_end)

    # 2. Straight line from Room 0 into Room 1 across Portal 1
    t1_visible = spawn_mobj(sim, 192 * FRACUNIT, 200 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert check_sight(sim, t0_end, t1_visible)

    # 3. Blocked around 90-degree corner (inner wall v1 occludes)
    assert not check_sight(sim, t0_start, t1_visible)

    # 4. Blocked across zigzag into Room 2
    t2_end = spawn_mobj(sim, 400 * FRACUNIT, 320 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert not check_sight(sim, t0_start, t2_end)
    assert not check_sight(sim, t0_end, t2_end)

    # 5. Visible from Room 1 top into Room 2 across Portal 2
    t1_top = spawn_mobj(sim, 192 * FRACUNIT, 320 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert check_sight(sim, t1_top, t2_end)


def test_dynamic_sector_door_lowering_floor() -> None:
    """Dynamic door with raised floor lowers and opens sight, then closes."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=128, ceil2=128)
    sim = _make_sim(mb)

    t1 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t2 = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    # Closed: floor at ceiling
    assert not check_sight(sim, t1, t2)

    # Lower floor to 0
    sim.level.sectors[1].floor_height = 0
    assert check_sight(sim, t1, t2)

    # Raise floor back to 128
    sim.level.sectors[1].floor_height = 128 * FRACUNIT
    assert not check_sight(sim, t1, t2)


def test_dynamic_sector_door_raising_ceiling() -> None:
    """Dynamic door with lowered ceiling raises and opens sight, then closes."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=0)
    sim = _make_sim(mb)

    t1 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t2 = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    # Closed: ceiling at floor
    assert not check_sight(sim, t1, t2)

    # Raise ceiling to 128
    sim.level.sectors[1].ceiling_height = 128 * FRACUNIT
    assert check_sight(sim, t1, t2)

    # Lower ceiling back to 0
    sim.level.sectors[1].ceiling_height = 0
    assert not check_sight(sim, t1, t2)


def test_precedence_reject_matrix_overrides_geometry() -> None:
    """RejectMatrix=1 blocks sight even with open geometry; RejectMatrix=0 obeys wall."""
    # 1. Clear geometry, RejectMatrix blocks
    mb1 = MapBuilder("MAP01")
    mb1.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=128)
    mb1.set_reject(b"\x02")  # sec0 -> sec1 rejected
    sim1 = _make_sim(mb1)

    t1_1 = spawn_mobj(sim1, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t2_1 = spawn_mobj(sim1, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    assert not check_sight(sim1, t1_1, t2_1), "RejectMatrix must override open portal"

    # 2. Solid 1-sided wall, RejectMatrix allows
    mb2 = MapBuilder("MAP01")
    mb2.add_room(0, 0, 256, 256)
    mb2.add_room(256, 0, 256, 256)
    mb2.set_reject(b"\x00")  # all pairs open
    sim2 = _make_sim(mb2)

    t1_2 = spawn_mobj(sim2, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t2_2 = spawn_mobj(sim2, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    assert not check_sight(sim2, t1_2, t2_2), "Solid 1-sided wall must block despite 0 reject"


# ==============================================================================
# Suite 3: Combat Integration Invariants (radius_attack)
# ==============================================================================


def test_radius_attack_multi_mobj_simultaneous_occlusion() -> None:
    """Single radius_attack explosion selectively damages exposed vs occluded mobjs."""
    mb = MapBuilder("MAP01")
    # Sec 0: [0, 256] x [0, 256] floor=0, ceil=128
    # Sec 1: [256, 512] x [0, 256] floor=80, ceil=128 (high step)
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=80, ceil2=128)
    # Extra room behind 1-sided wall: [0, 256] x [256, 512]
    mb.add_room(0, 256, 256, 256)
    sim = _make_sim(mb)

    # Blast center in room 0
    spot = spawn_mobj(sim, 128 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CIRCLEFLAME)

    # Target 1: Exposed in same room (direct LOS)
    t_exposed = spawn_mobj(sim, 200 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    hp_exposed = t_exposed.health

    # Target 2: Behind 1-sided solid north wall at y=256
    t_north = spawn_mobj(sim, 128 * FRACUNIT, 350 * FRACUNIT, 0, MobjType.MT_ETTIN)
    hp_north = t_north.health

    # Target 3: Behind high floor step (z=80, actor height 16 -> eye at 92; spot at z=0)
    t_step = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 80 * FRACUNIT, MobjType.MT_ETTIN)
    t_step.height = 16 * FRACUNIT
    hp_step = t_step.health

    # Target 4: Elevated actor on ledge looking down
    t_elevated = spawn_mobj(sim, 280 * FRACUNIT, 128 * FRACUNIT, 80 * FRACUNIT, MobjType.MT_ETTIN)
    t_elevated.height = 64 * FRACUNIT  # tall, head at 144
    hp_elevated = t_elevated.health

    # Single blast
    radius_attack(sim, spot=spot, source=None, damage=100, distance=250)

    # Assertions
    assert t_exposed.health < hp_exposed, "Exposed target must take damage"
    assert t_north.health == hp_north, "Target behind 1-sided wall must take 0 damage"
    assert t_step.health == hp_step, "Target occluded by high floor step must take 0 damage"
    assert t_elevated.health < hp_elevated, "Target visible on ledge must take damage"


def test_radius_attack_penetrating_doors_and_windows() -> None:
    """Splash damage penetrates open windows and open doors, but occludes below sill."""
    mb = MapBuilder("MAP01")
    # Window sector: floor=48, ceil=80
    mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=200,
        width2=200,
        height=256,
        floor1=0,
        ceil1=128,
        floor2=48,
        ceil2=80,
    )
    sim = _make_sim(mb)

    # Spot aligned with window at z=60
    spot = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 60 * FRACUNIT, MobjType.MT_CIRCLEFLAME)

    # Target 1: Aligned with window opening (z=50, height=20 -> 50..70)
    t_window = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 50 * FRACUNIT, MobjType.MT_ETTIN)
    t_window.height = 20 * FRACUNIT
    hp_window = t_window.health

    # Target 2: Below window sill (z=0, height=20 -> 0..20; sill is at 48)
    t_sill = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t_sill.height = 20 * FRACUNIT
    hp_sill = t_sill.health

    radius_attack(sim, spot=spot, source=None, damage=100, distance=250)

    assert t_window.health < hp_window, "Splash damage must penetrate open window"
    assert t_sill.health == hp_sill, "Splash damage must be occluded below window sill"


def test_radius_attack_dynamic_door_splash_transition() -> None:
    """Closed door shields target from blast; opening door allows subsequent blast damage."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=0)  # door closed
    sim = _make_sim(mb)

    spot = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CIRCLEFLAME)
    target = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    initial_hp = target.health

    # Blast 1: Door closed -> 0 damage
    radius_attack(sim, spot=spot, source=None, damage=100, distance=250)
    assert target.health == initial_hp

    # Open door
    sim.level.sectors[1].ceiling_height = 128 * FRACUNIT

    # Blast 2: Door open -> damages target
    radius_attack(sim, spot=spot, source=None, damage=100, distance=250)
    assert target.health < initial_hp
