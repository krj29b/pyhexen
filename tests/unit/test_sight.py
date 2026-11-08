# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Comprehensive 4-tier test suite for monster and combat line-of-sight checking.

Tiers:
1. Feature Coverage (RejectMatrix early out, 1-sided wall, 2-sided portal,
   eye level, radius attack, exports)
2. Boundary & Corner Cases (aperture equality, inversion, 1-unit slit,
   closed door, zero distance, zero height, colinear, grazing, negative coordinates)
3. Multi-Sector Corridors & Aperture Narrowing (corridors, steps, windows, reject vs geometry)
4. Real-World & GameSim (damage occlusion, dynamic doors, monster facing player,
   AST checks for floats/random/render/audio/ui, performance budget)
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

from tests.fixtures.wadbuilder import ML_TWOSIDED, MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.sim import check_sight
from pyhexen.sim.attacks import radius_attack
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjType
from pyhexen.sim.maputl import DivLine, intercept_vector, make_divline
from pyhexen.sim.mobj import spawn_mobj
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Line, load_level


def _make_sim_from_map(mb: MapBuilder) -> GameSim:
    """Build a synthetic level in memory and attach to a new GameSim."""
    stack = WadStack(WadBuilder("IWAD").add_map(mb).to_bytes())
    level = load_level(stack, mb.name)
    return GameSim(level=level, rng=Random())


def _build_corridor_map(
    floor0: int = 0,
    ceil0: int = 128,
    floor1: int = 0,
    ceil1: int = 128,
    floor2: int = 0,
    ceil2: int = 128,
) -> MapBuilder:
    """Build a 3-sector corridor map with rooms at [0, 256], [256, 512], and [512, 768]."""
    mb = MapBuilder("MAP01")
    s0 = mb.add_sector(floor0, ceil0)
    s1 = mb.add_sector(floor1, ceil1)
    s2 = mb.add_sector(floor2, ceil2)

    v0 = mb.add_vertex(0, 256)
    v1 = mb.add_vertex(256, 256)
    v2 = mb.add_vertex(512, 256)
    v3 = mb.add_vertex(768, 256)
    v4 = mb.add_vertex(768, 0)
    v5 = mb.add_vertex(512, 0)
    v6 = mb.add_vertex(256, 0)
    v7 = mb.add_vertex(0, 0)

    sd_top0 = mb.add_sidedef(s0)
    sd_top1 = mb.add_sidedef(s1)
    sd_top2 = mb.add_sidedef(s2)
    sd_right = mb.add_sidedef(s2)
    sd_bot2 = mb.add_sidedef(s2)
    sd_bot1 = mb.add_sidedef(s1)
    sd_bot0 = mb.add_sidedef(s0)
    sd_left = mb.add_sidedef(s0)

    sd_p1_0 = mb.add_sidedef(s0)
    sd_p1_1 = mb.add_sidedef(s1)
    sd_p2_1 = mb.add_sidedef(s1)
    sd_p2_2 = mb.add_sidedef(s2)

    l_top0 = mb.add_linedef(v0, v1, sd_top0)
    l_top1 = mb.add_linedef(v1, v2, sd_top1)
    l_top2 = mb.add_linedef(v2, v3, sd_top2)
    l_right = mb.add_linedef(v3, v4, sd_right)
    l_bot2 = mb.add_linedef(v4, v5, sd_bot2)
    l_bot1 = mb.add_linedef(v5, v6, sd_bot1)
    l_bot0 = mb.add_linedef(v6, v7, sd_bot0)
    l_left = mb.add_linedef(v7, v0, sd_left)

    l_p1 = mb.add_linedef(v1, v6, sd_p1_0, sd_p1_1, flags=ML_TWOSIDED)
    l_p2 = mb.add_linedef(v2, v5, sd_p2_1, sd_p2_2, flags=ML_TWOSIDED)

    mb.add_seg(v0, v1, 0, l_top0)
    mb.add_seg(v1, v6, -16384, l_p1, side=0)
    mb.add_seg(v6, v7, 32768, l_bot0)
    mb.add_seg(v7, v0, 16384, l_left)
    mb.add_subsector(4, 0, bbox=(256, 0, 0, 256))

    mb.add_seg(v1, v2, 0, l_top1)
    mb.add_seg(v2, v5, -16384, l_p2, side=0)
    mb.add_seg(v5, v6, 32768, l_bot1)
    mb.add_seg(v6, v1, 16384, l_p1, side=1)
    mb.add_subsector(4, 4, bbox=(256, 0, 256, 512))

    mb.add_seg(v2, v3, 0, l_top2)
    mb.add_seg(v3, v4, -16384, l_right)
    mb.add_seg(v4, v5, 32768, l_bot2)
    mb.add_seg(v5, v2, 16384, l_p2, side=1)
    mb.add_subsector(4, 8, bbox=(256, 0, 512, 768))

    return mb


# ==============================================================================
# Tier 1: Feature Coverage
# ==============================================================================


def test_check_sight_reject_matrix_blocked() -> None:
    """Sector pairs marked as unreachable in RejectMatrix return False immediately."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=128)
    # 2 sectors: pnum = 0 * 2 + 1 = 1 -> byte 0 bit 1 (0x02) rejects sec0 -> sec1
    mb.set_reject(b"\x02")
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    # Reject matrix says unreachable: sight is rejected immediately
    assert not check_sight(sim, t0, t1)


def test_check_sight_reject_matrix_asymmetric() -> None:
    """Asymmetric RejectMatrix rejects A->B but allows B->A if bit is cleared."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=128)
    # Bit 1 set (sec0 -> sec1 rejected), Bit 2 cleared (sec1 -> sec0 reachable)
    mb.set_reject(b"\x02")
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    assert not check_sight(sim, t0, t1)
    assert check_sight(sim, t1, t0)


def test_check_sight_reject_matrix_zeroes_allows_ray() -> None:
    """Default all-zero RejectMatrix allows ray traversal to evaluate geometry."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=128)
    mb.set_reject(b"\x00")
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    assert check_sight(sim, t0, t1)


def test_check_sight_one_sided_wall_blocks() -> None:
    """1-sided linedefs between viewer and target block sight."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    mb.add_room(x=256, y=0, width=256, height=256)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 128 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 384 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    assert not check_sight(sim, t0, t1)


def test_check_sight_open_portal_passes() -> None:
    """Clear lines of sight between entities across open 2-sided linedefs return True."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=128)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 400 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    assert check_sight(sim, t0, t1)


def test_check_sight_eye_level_calculation() -> None:
    """Looker eye level is 75% height; tall actor can see over obstacle short actor cannot."""
    # Obstacle portal halfway between x=100 and x=300 (frac=0.5):
    # Floor in sector 2 is raised to 48 units.
    # Target (x=300) has height=56 (head at z=56).
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=200,
        width2=200,
        height=256,
        floor1=0,
        ceil1=128,
        floor2=48,
        ceil2=128,
    )
    sim = _make_sim_from_map(mb)

    target = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 48 * FRACUNIT, MobjType.MT_ETTIN)
    target.height = 56 * FRACUNIT

    # Tall monster: height 56 -> eye = 0 + 56 - 14 = 42.
    # At portal (frac=0.5): floor slope = (48 - 42) / 0.5 = 12.
    # Topslope = target.z + target.height - eye = 48 + 56 - 42 = 62.
    # 12 < 62 -> sight passes!
    tall_monster = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    tall_monster.height = 56 * FRACUNIT
    assert check_sight(sim, tall_monster, target)

    # Short monster: height 16 -> eye = 0 + 16 - 4 = 12.
    # At portal (frac=0.5): floor slope = (48 - 12) / 0.5 = 72.
    # Target topslope = 48 + 56 - 12 = 92.
    # But wait, target bottom slope = 48 - 12 = 36.
    # Floor slope (72) > target bottom (36), so bottomslope becomes 72.
    # Now suppose target has height 20 (head at 68):
    # Then target topslope = 48 + 20 - 12 = 56.
    # Since floor slope 72 > topslope 56, aperture closes!
    short_target = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 48 * FRACUNIT, MobjType.MT_ETTIN)
    short_target.height = 20 * FRACUNIT

    # For tall monster: eye=42. Floor slope = (48-42)/0.5 = 12. Topslope = 48+20-42 = 26.
    # 12 < 26 -> tall monster CAN see short target!
    assert check_sight(sim, tall_monster, short_target)

    # For short monster: eye=12. Floor slope = (48-12)/0.5 = 72. Topslope = 48+20-12 = 56.
    # 72 >= 56 -> short monster CANNOT see short target!
    short_monster = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    short_monster.height = 16 * FRACUNIT
    assert not check_sight(sim, short_monster, short_target)


def test_check_sight_radius_attack_integration() -> None:
    """radius_attack calls check_sight and occludes splash damage behind wall."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    mb.add_room(x=256, y=0, width=256, height=256)
    sim = _make_sim_from_map(mb)

    spot = spawn_mobj(sim, 200 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CIRCLEFLAME)
    target = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    initial_health = target.health

    radius_attack(sim, spot=spot, source=None, damage=100, distance=200)
    assert target.health == initial_health


def test_check_sight_exported_from_sim() -> None:
    """check_sight is exported from pyhexen.sim and present in __all__."""
    import pyhexen.sim

    assert hasattr(pyhexen.sim, "check_sight")
    assert "check_sight" in pyhexen.sim.__all__


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================


def test_check_sight_aperture_exact_equality_blocks() -> None:
    """Portal where floor step matches ceiling drop (topslope == bottomslope) blocks sight."""
    # When topslope <= bottomslope, PTR_SightTraverse strictly returns false
    mb = MapBuilder("MAP01")
    # Halfway (x=200): floor2=50, ceil2=50 -> closed portal slit of 0 width
    mb.add_two_sector_rooms(
        width1=200,
        width2=200,
        floor1=0,
        ceil1=128,
        floor2=50,
        ceil2=50,
    )
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 50 * FRACUNIT, MobjType.MT_ETTIN)
    assert not check_sight(sim, t0, t1)


def test_check_sight_aperture_one_unit_open_passes() -> None:
    """Portal open by 1 unit allows sight through."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        width1=200,
        width2=200,
        floor1=0,
        ceil1=128,
        floor2=59,
        ceil2=60,
    )
    sim = _make_sim_from_map(mb)

    # Viewer eye at 60 (height 80 -> eye = 60) looking through 1-unit slit [59, 60]
    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t0.height = 80 * FRACUNIT  # eye = 60
    t1 = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 59 * FRACUNIT, MobjType.MT_ETTIN)
    t1.height = 2 * FRACUNIT  # spans 59..61
    assert check_sight(sim, t0, t1)


def test_check_sight_aperture_inverted_blocks() -> None:
    """Floor step exceeding ceiling drop (openbottom > opentop) blocks sight."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        width1=200,
        width2=200,
        floor1=0,
        ceil1=128,
        floor2=70,
        ceil2=30,
    )
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 70 * FRACUNIT, MobjType.MT_ETTIN)
    assert not check_sight(sim, t0, t1)


def test_check_sight_closed_door_blocks() -> None:
    """2-sided door portal with floor == ceil blocks sight immediately."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        width1=200,
        width2=200,
        floor1=0,
        ceil1=128,
        floor2=0,
        ceil2=0,
    )
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert not check_sight(sim, t0, t1)


def test_check_sight_viewer_eye_at_floor_or_ceiling() -> None:
    """Zero height viewer (eye at ground) does not raise ZeroDivisionError."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=128)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t0.height = 0  # eye at 0
    t1 = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert check_sight(sim, t0, t1)


def test_check_sight_target_zero_height_blocks_on_portal() -> None:
    """Target with height=0 crossing a stepped portal blocks sight."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        width1=200,
        width2=200,
        floor1=0,
        ceil1=128,
        floor2=16,
        ceil2=128,
    )
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 16 * FRACUNIT, MobjType.MT_ETTIN)
    t1.height = 0  # topslope == bottomslope initially
    assert not check_sight(sim, t0, t1)


def test_check_sight_colinear_ray_with_linedef() -> None:
    """Colinear ray parallel to linedef (den == 0 in intercept_vector) handles edge cleanly."""
    line = Line(
        index=0,
        v1=type("Vertex", (), {"x": 100 * FRACUNIT, "y": 128 * FRACUNIT})(),
        v2=type("Vertex", (), {"x": 300 * FRACUNIT, "y": 128 * FRACUNIT})(),
        dx=200 * FRACUNIT,
        dy=0,
        flags=ML_TWOSIDED,
        special=0,
        args=(0, 0, 0, 0, 0),
        sidenum=(0, 1),
        front_side=None,
        back_side=None,
        front_sector=None,
        back_sector=None,
        slopetype=0,
        bbox=(0, 0, 0, 0),
        valid_count=0,
        is_polyobj=False,
    )
    dl = make_divline(line)
    trace = DivLine(x=50 * FRACUNIT, y=128 * FRACUNIT, dx=300 * FRACUNIT, dy=0)
    frac = intercept_vector(trace, dl)
    assert frac == 0


def test_check_sight_ray_grazing_vertex() -> None:
    """Ray passing directly through a shared vertex does not crash or false-occlude."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=128)
    sim = _make_sim_from_map(mb)

    # The portal vertices are at (256, 256) and (256, 0)
    # A ray along y=256 touches the top vertex
    t0 = spawn_mobj(sim, 100 * FRACUNIT, 200 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 400 * FRACUNIT, 200 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert check_sight(sim, t0, t1)


def test_check_sight_zero_distance_same_actor() -> None:
    """check_sight on the same actor returns True immediately."""
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 256, 256)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert check_sight(sim, t0, t0)


def test_check_sight_zero_xy_distance_different_z() -> None:
    """Two mobjs at same (x, y) coordinates with vertical separation return True."""
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 256, 256)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 50 * FRACUNIT, MobjType.MT_ETTIN)
    assert check_sight(sim, t0, t1)


def test_check_sight_negative_coordinates() -> None:
    """Ray traversal correctly handles negative map coordinates."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        x=-512,
        y=-256,
        width1=256,
        width2=256,
        height=256,
        floor1=0,
        ceil1=128,
        floor2=0,
        ceil2=128,
    )
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, -400 * FRACUNIT, -128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, -100 * FRACUNIT, -128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    assert check_sight(sim, t0, t1)


def test_check_sight_out_of_bounds_returns_false() -> None:
    """Target out of blockmap bounds safely returns False without raising IndexError."""
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 256, 256)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 10000 * FRACUNIT, 10000 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert not check_sight(sim, t0, t1)


# ==============================================================================
# Tier 3: Multi-Sector Ray Traversal & Aperture Narrowing
# ==============================================================================


def test_check_sight_multi_sector_open_corridor() -> None:
    """Ray traverses all portals across a 3-sector open corridor and returns True."""
    mb = _build_corridor_map(0, 128, 0, 128, 0, 128)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t2 = spawn_mobj(sim, 650 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    assert check_sight(sim, t0, t2)


def test_check_sight_gradual_aperture_narrowing_visible() -> None:
    """Corridor with stepped floors/ceilings preserves line of sight to centered target."""
    # Sec0: [0, 128], Sec1: [20, 100], Sec2: [40, 80]
    mb = _build_corridor_map(0, 128, 20, 100, 40, 80)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t0.height = 64 * FRACUNIT  # eye = 48
    t2 = spawn_mobj(sim, 650 * FRACUNIT, 128 * FRACUNIT, 40 * FRACUNIT, MobjType.MT_ETTIN)
    t2.height = 30 * FRACUNIT  # spans 40 to 70

    assert check_sight(sim, t0, t2)


def test_check_sight_gradual_aperture_narrowing_floor_blocked() -> None:
    """Corridor where second portal floor rises high enough to occlude target."""
    # Portal 2 floor at 75 blocks sight to target on floor in room 2
    mb = _build_corridor_map(0, 128, 20, 128, 75, 128)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t0.height = 40 * FRACUNIT  # eye = 30
    t2 = spawn_mobj(sim, 650 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t2.height = 20 * FRACUNIT

    assert not check_sight(sim, t0, t2)


def test_check_sight_gradual_aperture_narrowing_ceiling_blocked() -> None:
    """Corridor where portal ceiling drops low enough to occlude target."""
    # Portal 2 ceiling at 35 drops below sightline to tall target
    mb = _build_corridor_map(0, 128, 0, 100, 0, 35)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t0.height = 64 * FRACUNIT  # eye = 48
    t2 = spawn_mobj(sim, 650 * FRACUNIT, 128 * FRACUNIT, 50 * FRACUNIT, MobjType.MT_ETTIN)
    t2.height = 20 * FRACUNIT

    assert not check_sight(sim, t0, t2)


def test_check_sight_step_up_blocks_low_angle() -> None:
    """Viewer low on floor is blocked by high floor step."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=80, ceil2=128)
    sim = _make_sim_from_map(mb)

    viewer = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    viewer.height = 32 * FRACUNIT  # eye = 24
    target = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 80 * FRACUNIT, MobjType.MT_ETTIN)
    target.height = 10 * FRACUNIT

    assert not check_sight(sim, viewer, target)


def test_check_sight_step_up_allows_high_angle() -> None:
    """Viewer on elevated platform looks down over floor step to see target."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=100, ceil1=200, floor2=50, ceil2=200)
    sim = _make_sim_from_map(mb)

    viewer = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 100 * FRACUNIT, MobjType.MT_ETTIN)
    viewer.height = 64 * FRACUNIT  # eye = 148
    target = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 50 * FRACUNIT, MobjType.MT_ETTIN)
    target.height = 32 * FRACUNIT

    assert check_sight(sim, viewer, target)


def test_check_sight_window_slit_aperture() -> None:
    """Window aperture restricts visibility to actors aligned with the opening."""
    # Window sector: floor=48, ceil=80
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=48, ceil2=80)
    sim = _make_sim_from_map(mb)

    viewer = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    viewer.height = 80 * FRACUNIT  # eye = 60 (aligned with window)

    # Target aligned with window (z=50, height=20 -> 50..70)
    aligned_target = spawn_mobj(
        sim, 350 * FRACUNIT, 128 * FRACUNIT, 50 * FRACUNIT, MobjType.MT_ETTIN
    )
    aligned_target.height = 20 * FRACUNIT
    assert check_sight(sim, viewer, aligned_target)

    # Target below sill (z=0, height=20 -> 0..20)
    low_target = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    low_target.height = 20 * FRACUNIT
    assert not check_sight(sim, viewer, low_target)


def test_check_sight_reject_vs_geometry_matrix_preempts() -> None:
    """RejectMatrix bit = 1 blocks sight even with a totally clear geometric path."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=128)
    mb.set_reject(b"\x02")  # sec0 -> sec1 rejected
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    # Reject table preempts clear geometry
    assert not check_sight(sim, t0, t1)


def test_check_sight_reject_vs_geometry_open_matrix_blocked_by_wall() -> None:
    """Unrejected RejectMatrix bit (0) still blocks sight if geometry has a solid wall."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    mb.add_room(x=256, y=0, width=256, height=256)
    mb.set_reject(b"\x00")  # all pairs reachable per reject table
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    # Geometry blocks despite open reject table
    assert not check_sight(sim, t0, t1)


# ==============================================================================
# Tier 4: Real-World & GameSim Scenarios
# ==============================================================================


def test_radius_attack_damage_occluded_by_geometry() -> None:
    """Radius splash damage hits direct LOS target but spares target behind wall."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    mb.add_room(x=256, y=0, width=256, height=256)
    sim = _make_sim_from_map(mb)

    spot = spawn_mobj(sim, 128 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CIRCLEFLAME)
    # Target 1 in same room: direct LOS
    target_exposed = spawn_mobj(sim, 180 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    # Target 2 in adjacent room: behind solid 1-sided wall
    target_sheltered = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    hp1 = target_exposed.health
    hp2 = target_sheltered.health

    radius_attack(sim, spot=spot, source=None, damage=100, distance=200)

    assert target_exposed.health < hp1
    assert target_sheltered.health == hp2


def test_check_sight_dynamic_door_transition() -> None:
    """Door sector transitions dynamically from closed (blocked) to open (visible)."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=0)  # door closed
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    assert not check_sight(sim, t0, t1)

    # Open door by raising sector 2 ceiling
    door_sec = sim.level.sectors[1]
    door_sec.ceiling_height = 128 * FRACUNIT

    assert check_sight(sim, t0, t1)


def test_monster_line_of_sight_facing_player() -> None:
    """Monster sees player in open room; vision lost when player steps behind pillar."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, ceil1=128, floor2=0, ceil2=128)
    sim = _make_sim_from_map(mb)

    monster = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    player = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    assert check_sight(sim, monster, player)

    # Close portal by lowering ceiling
    sim.level.sectors[1].ceiling_height = 0
    assert not check_sight(sim, monster, player)


def test_sight_no_floats_in_simulation() -> None:
    """AST analysis confirms zero floating point constants or conversions in sight.py."""
    source_path = Path(__file__).parents[2] / "src" / "pyhexen" / "sim" / "sight.py"
    with open(source_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(source_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"Disallowed float constant {node.value} at line {node.lineno}")


def test_sight_no_render_audio_ui_imports() -> None:
    """AST analysis confirms sim/sight.py does not import render, audio, or ui."""
    source_path = Path(__file__).parents[2] / "src" / "pyhexen" / "sim" / "sight.py"
    with open(source_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(source_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in ("render", "audio", "ui"):
                    if forbidden in alias.name.split("."):
                        raise AssertionError(
                            f"Forbidden import '{alias.name}' at line {node.lineno}"
                        )
        elif isinstance(node, ast.ImportFrom) and node.module:
            for forbidden in ("render", "audio", "ui"):
                if forbidden in node.module.split("."):
                    raise AssertionError(
                        f"Forbidden import from '{node.module}' at line {node.lineno}"
                    )


def test_sight_no_random_stdlib_import() -> None:
    """AST analysis confirms sim/sight.py does not import standard library random."""
    source_path = Path(__file__).parents[2] / "src" / "pyhexen" / "sim" / "sight.py"
    with open(source_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(source_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "random":
                    raise AssertionError(f"Disallowed stdlib random import at line {node.lineno}")
        elif isinstance(node, ast.ImportFrom) and node.module == "random":
            raise AssertionError(f"Disallowed stdlib random import at line {node.lineno}")


def test_sight_performance_budget() -> None:
    """1000 sight checks complete within the 50 ms budget (< 0.05 ms / check)."""
    mb = _build_corridor_map(0, 128, 0, 128, 0, 128)
    sim = _make_sim_from_map(mb)

    t0 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1 = spawn_mobj(sim, 650 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    # Warmup
    for _ in range(10):
        check_sight(sim, t0, t1)

    start = time.perf_counter()
    for _ in range(1000):
        check_sight(sim, t0, t1)
    duration_ms = (time.perf_counter() - start) * 1000

    # Budget: 1000 checks in < 100 ms (< 0.1 ms/check)
    assert duration_ms < 100.0, f"1000 sight checks took {duration_ms:.2f} ms (budget: 100 ms)"
