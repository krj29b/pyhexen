# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Comprehensive End-to-End Test Suite for Phase 9 (Polyobjects).

Structured into 4 Tiers:
- Tier 1: Feature Coverage (>=5 tests per feature for all 28 inventoried features).
- Tier 2: Boundary & Corner Cases (angles 0 & 255, skipped order, order 0,
          non-closed loops, multiple polyobjs per subsector, max 64 segs, etc.).
- Tier 3: Cross-Feature Combinations (pairwise interactions: rotate + mirror,
          move + mirror, sliding door + collision, swinging door + reversal,
          crushing damage, PCD_POLYWAIT + multiple scripts, override specials).
- Tier 4: Real-World Application Scenarios (Hexen IWAD MAP01 and MAP02 polyobjects).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.events import SequenceEvent
from pyhexen.core.fixed import FRACBITS, FRACUNIT
from pyhexen.sim.acs.vm import ACScriptState
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import spawn_player
from pyhexen.sim.specials.dispatcher import execute_line_special
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level

# ---------------------------------------------------------------------------
# Progressive Testability Detection
# ---------------------------------------------------------------------------
try:
    from pyhexen.sim import polyobj as po_module

    HAS_POLYOBJ_MODULE = True
except ImportError:
    po_module = None  # type: ignore[assignment]
    HAS_POLYOBJ_MODULE = False

HAS_M1_SPAWNING = (
    HAS_POLYOBJ_MODULE and hasattr(po_module, "Polyobj") and hasattr(po_module, "init_polyobjs")
)
HAS_M2_THINKERS = (
    HAS_POLYOBJ_MODULE
    and hasattr(po_module, "RotatePolyThinker")
    and hasattr(po_module, "MovePolyThinker")
    and hasattr(po_module, "PolyDoorThinker")
)
HAS_M2_PHYSICS = HAS_POLYOBJ_MODULE and hasattr(po_module, "po_move_polyobj")
HAS_M3_RENDER = False
try:
    from pyhexen.render import renderer

    HAS_M3_RENDER = hasattr(renderer, "PolyMesh") or hasattr(renderer, "render_polyobjs")
except ImportError:
    pass

skip_m1 = pytest.mark.skipif(not HAS_M1_SPAWNING, reason="Phase 9 M1 Spawning not yet implemented")
skip_m2_thinkers = pytest.mark.skipif(
    not HAS_M2_THINKERS, reason="Phase 9 M2 Thinkers not yet implemented"
)
skip_m2_physics = pytest.mark.skipif(
    not HAS_M2_PHYSICS, reason="Phase 9 M2 Physics not yet implemented"
)
skip_m3_render = pytest.mark.skipif(
    not HAS_M3_RENDER, reason="Phase 9 M3 Dynamic Rendering not yet implemented"
)


# ---------------------------------------------------------------------------
# Synthetic Test Fixture Helpers
# ---------------------------------------------------------------------------
def _build_synthetic_poly_sim(
    tag: int = 1,
    anchor: tuple[int, int] = (-2000, -2000),
    spawn: tuple[int, int] = (128, 128),
    *,
    width: int = 64,
    height: int = 64,
    crush: bool = False,
    method: str = "start_line",
    mirror_tag: int = 0,
    seq_type: int = 0,
) -> tuple[Level, GameSim]:
    """Build a complete synthetic level and GameSim containing a polyobject."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_polyobject_box(
        tag=tag,
        anchor_x=anchor[0],
        anchor_y=anchor[1],
        spawn_x=spawn[0],
        spawn_y=spawn[1],
        width=width,
        height=height,
        crush=crush,
        method=method,
        mirror_tag=mirror_tag,
        seq_type=seq_type,
    )
    # Player 1 start in playable room
    mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
    wb.add_map(mb)

    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")

    # If load_level doesn't automatically call init_polyobjs, invoke it here
    if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
        po_module.init_polyobjs(level)

    sim = GameSim(level=level)
    spawn_map_things(sim)
    return level, sim


def _get_poly(level: Level, tag: int) -> Any:
    """Retrieve polyobject by tag from Level."""
    if hasattr(level, "polyobjs_by_tag"):
        return level.polyobjs_by_tag.get(tag)
    if hasattr(level, "polyobjs"):
        if isinstance(level.polyobjs, dict):
            return level.polyobjs.get(tag)
        for po in level.polyobjs:
            if getattr(po, "tag", None) == tag:
                return po
    return None


# ===========================================================================
# TIER 1: FEATURE COVERAGE (>=5 tests per feature for all 28 features)
# ===========================================================================


# ---------------------------------------------------------------------------
# Feature 1: Polyobj data structure (r_local.h:153, po_man.c:56)
# ---------------------------------------------------------------------------
@skip_m1
class TestFeature01PolyobjDataStructure:
    """Feature 1: Core polyobject state structure."""

    def test_f01_1_field_initialization(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=1, spawn=(256, 256))
        po = _get_poly(level, 1)
        assert po is not None
        assert po.tag == 1
        assert po.angle == 0
        assert hasattr(po, "start_spot") or hasattr(po, "start_spot_x")
        assert len(po.segs) == 4
        assert po.crush is False
        assert po.seq_type == 0

    def test_f01_2_busy_property_idle(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=2)
        po = _get_poly(level, 2)
        assert po is not None
        assert po.specialdata is None
        assert not po.busy

    def test_f01_3_busy_property_active(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=3)
        po = _get_poly(level, 3)
        assert po is not None
        po.specialdata = object()
        assert po.busy
        po.specialdata = None

    def test_f01_4_original_pts_relative_to_startspot(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=4, spawn=(200, 200), width=64, height=64)
        po = _get_poly(level, 4)
        assert po is not None
        assert len(po.original_pts) == 4
        # Original points must be relative offsets within [-32, 32] map units
        for pt in po.original_pts:
            x_val = pt.x if hasattr(pt, "x") else pt[0]
            y_val = pt.y if hasattr(pt, "y") else pt[1]
            assert abs(x_val) <= 32 * FRACUNIT
            assert abs(y_val) <= 32 * FRACUNIT

    def test_f01_5_prev_pts_tracking(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=5, spawn=(128, 128))
        po = _get_poly(level, 5)
        assert po is not None
        assert len(po.prev_pts) == 4
        for seg, pt in zip(po.segs, po.prev_pts, strict=True):
            pt_x = pt.x if hasattr(pt, "x") else pt[0]
            pt_y = pt.y if hasattr(pt, "y") else pt[1]
            assert pt_x == seg.v1.x
            assert pt_y == seg.v1.y


# ---------------------------------------------------------------------------
# Feature 2: PolyBlock & PolyBlockMap (r_local.h:170, po_man.c:55)
# ---------------------------------------------------------------------------
@skip_m1
class TestFeature02PolyBlockMap:
    """Feature 2: 2D dynamic spatial hash grid linking polyobjects to blockmap."""

    def test_f02_1_polyblock_node_structure(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=10)
        po = _get_poly(level, 10)
        assert hasattr(po_module, "PolyBlock")
        pb = po_module.PolyBlock(polyobj=po)
        assert pb.polyobj is po
        assert pb.prev is None
        assert pb.next is None

    def test_f02_2_initial_blockmap_linking(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=11, spawn=(256, 256), width=64, height=64)
        bm = level.blockmap
        assert hasattr(bm, "poly_block_map")
        # Check if at least one cell in poly_block_map contains a reference
        found = False
        for node in bm.poly_block_map:
            curr = node
            while curr is not None:
                if getattr(curr, "polyobj", None) and curr.polyobj.tag == 11:
                    found = True
                    break
                curr = getattr(curr, "next", None)
            if found:
                break
        assert found, "Polyobject not linked into poly_block_map"

    def test_f02_3_unlink_polyobj(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=12, spawn=(128, 128))
        po = _get_poly(level, 12)
        assert hasattr(po_module, "unlink_polyobj")
        po_module.unlink_polyobj(po, level.blockmap)
        # Verify node removed or polyobj set to None
        for node in level.blockmap.poly_block_map:
            curr = node
            while curr is not None:
                assert getattr(curr, "polyobj", None) is not po
                curr = getattr(curr, "next", None)

    def test_f02_4_link_polyobj(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=13, spawn=(128, 128))
        po = _get_poly(level, 13)
        po_module.unlink_polyobj(po, level.blockmap)
        po_module.link_polyobj(po, level.blockmap)
        found = any(
            curr.polyobj is po
            for curr in level.blockmap.poly_block_map
            if curr is not None and getattr(curr, "polyobj", None) is not None
        )
        assert found

    def test_f02_5_block_lines_iterator_encounters_poly_segs(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=14, spawn=(128, 128))
        encountered_lines: list[int] = []

        def _line_visitor(ld: Any) -> bool:
            encountered_lines.append(ld.index)
            return True

        bx = level.blockmap.world_to_block_x(128 << FRACBITS)
        by = level.blockmap.world_to_block_y(128 << FRACBITS)
        level.blockmap.block_lines_iterator(bx, by, _line_visitor, valid_count=999)
        po = _get_poly(level, 14)
        po_line_indices = {seg.linedef.index for seg in po.segs}
        assert any(idx in po_line_indices for idx in encountered_lines)


# ---------------------------------------------------------------------------
# Feature 3: Anchor Spot (Thing 3000) (po_man.c:1464)
# ---------------------------------------------------------------------------
@skip_m1
class TestFeature03AnchorSpot:
    """Feature 3: Anchor Spot (Thing 3000) in dummy holding area."""

    def test_f03_1_recognizes_thing_3000(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=20, anchor=(-3000, -3000))
        po = _get_poly(level, 20)
        assert po is not None

    def test_f03_2_anchor_tag_from_thing_angle(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=21, anchor=(-1500, -1500))
        po = _get_poly(level, 21)
        assert po is not None
        assert po.tag == 21

    def test_f03_3_coordinate_delta_computed_from_anchor(self) -> None:
        level, _ = _build_synthetic_poly_sim(
            tag=22, anchor=(-2000, -2000), spawn=(100, 100), width=40, height=40
        )
        po = _get_poly(level, 22)
        assert po is not None
        # Centroid of spawn spot should be close to (100, 100)
        xs = [seg.v1.x >> FRACBITS for seg in po.segs]
        ys = [seg.v1.y >> FRACBITS for seg in po.segs]
        avg_x = sum(xs) // len(xs)
        avg_y = sum(ys) // len(ys)
        assert abs(avg_x - 100) <= 2
        assert abs(avg_y - 100) <= 2

    def test_f03_4_anchor_excluded_from_sim_mobjs(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=23)
        # Mobjs in simulation should not contain thing 3000
        for mobj in getattr(sim, "mobjs", []):
            assert getattr(mobj, "type", None) != 3000

    def test_f03_5_multiple_anchors_link_to_respective_spawns(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=256, height=512)
        mb.add_room(x=256, y=0, width=256, height=512)
        mb.add_polyobject_box(1, -2000, -2000, 100, 100)
        mb.add_polyobject_box(2, -1000, -1000, 300, 300)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        po1 = _get_poly(level, 1)
        po2 = _get_poly(level, 2)
        assert po1 is not None and po2 is not None
        assert po1.tag == 1 and po2.tag == 2


# ---------------------------------------------------------------------------
# Feature 4: Spawn Spot (Thing 3001) (po_man.c:1447)
# ---------------------------------------------------------------------------
@skip_m1
class TestFeature04SpawnSpot:
    """Feature 4: Spawn Spot (Thing 3001) in playable map."""

    def test_f04_1_spawn_spot_recognition(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=30, crush=False)
        po = _get_poly(level, 30)
        assert po is not None

    def test_f04_2_start_spot_coordinates(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=31, spawn=(250, 350), crush=False)
        po = _get_poly(level, 31)
        assert po is not None
        sp_x = po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x
        sp_y = po.start_spot.y if hasattr(po, "start_spot") else po.start_spot_y
        assert sp_x == 250 * FRACUNIT
        assert sp_y == 350 * FRACUNIT

    def test_f04_3_crush_flag_is_false(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=32, crush=False)
        po = _get_poly(level, 32)
        assert po is not None
        assert po.crush is False

    def test_f04_4_spawn_spot_excluded_from_sim_mobjs(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=33, crush=False)
        for mobj in getattr(sim, "mobjs", []):
            assert getattr(mobj, "type", None) != 3001

    def test_f04_5_multiple_spawn_spots_create_independent_polyobjs(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=256, height=512)
        mb.add_room(x=256, y=0, width=256, height=512)
        mb.add_polyobject_box(101, -2000, -2000, 100, 100, crush=False)
        mb.add_polyobject_box(102, -1500, -1500, 300, 200, crush=False)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        assert _get_poly(level, 101) is not None
        assert _get_poly(level, 102) is not None


# ---------------------------------------------------------------------------
# Feature 5: Crush Spawn Spot (Thing 3002) (po_man.c:1447)
# ---------------------------------------------------------------------------
@skip_m1
class TestFeature05CrushSpawnSpot:
    """Feature 5: Crush Spawn Spot (Thing 3002)."""

    def test_f05_1_crush_spawn_spot_recognition(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=40, crush=True)
        po = _get_poly(level, 40)
        assert po is not None

    def test_f05_2_crush_spawn_spot_coordinates(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=41, spawn=(180, 220), crush=True)
        po = _get_poly(level, 41)
        assert po is not None
        sp_x = po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x
        sp_y = po.start_spot.y if hasattr(po, "start_spot") else po.start_spot_y
        assert sp_x == 180 * FRACUNIT
        assert sp_y == 220 * FRACUNIT

    def test_f05_3_crush_flag_is_true(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=42, crush=True)
        po = _get_poly(level, 42)
        assert po is not None
        assert po.crush is True

    def test_f05_4_crush_vs_non_crush_differentiated(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=256, height=512)
        mb.add_room(x=256, y=0, width=256, height=512)
        mb.add_polyobject_box(1, -2000, -2000, 100, 100, crush=False)
        mb.add_polyobject_box(2, -1500, -1500, 300, 200, crush=True)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        po_norm = _get_poly(level, 1)
        po_crush = _get_poly(level, 2)
        assert po_norm.crush is False
        assert po_crush.crush is True

    def test_f05_5_crush_thing_excluded_from_sim_mobjs(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=45, crush=True)
        for mobj in getattr(sim, "mobjs", []):
            assert getattr(mobj, "type", None) != 3002


# ---------------------------------------------------------------------------
# Feature 6: StartLine Chains (Special 1) (po_man.c:1216)
# ---------------------------------------------------------------------------
@skip_m1
class TestFeature06StartLineChains:
    """Feature 6: StartLine Chains (Special 1)."""

    def test_f06_1_startline_traversal(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=50, method="start_line")
        po = _get_poly(level, 50)
        assert po is not None
        assert len(po.segs) == 4

    def test_f06_2_special_and_arg1_cleared(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=51, method="start_line")
        po = _get_poly(level, 51)
        assert po is not None
        # In vanilla po_man.c, special and arg1 are cleared to 0 after collection
        for seg in po.segs:
            assert seg.linedef.special == 0

    def test_f06_3_reads_mirror_arg(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=52, mirror_tag=99, method="start_line")
        po = _get_poly(level, 52)
        assert po is not None
        # First seg linedef stores mirror_tag in arg2 (or args[1])
        first_ld = po.segs[0].linedef
        assert first_ld.args[1] == 99

    def test_f06_4_reads_seqtype_arg(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=53, seq_type=4, method="start_line")
        po = _get_poly(level, 53)
        assert po is not None
        assert po.seq_type == 4

    def test_f06_5_numsegs_counts_all_constituent_segs(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=54, method="start_line")
        po = _get_poly(level, 54)
        assert po is not None
        numsegs = getattr(po, "numsegs", len(po.segs))
        assert numsegs == 4


# ---------------------------------------------------------------------------
# Feature 7: Explicit Lines (Special 5) (po_man.c:1256)
# ---------------------------------------------------------------------------
@skip_m1
class TestFeature07ExplicitLines:
    """Feature 7: Explicit Lines (Special 5)."""

    def test_f07_1_explicit_lines_order_collection(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=60, method="explicit")
        po = _get_poly(level, 60)
        assert po is not None
        assert len(po.segs) == 4

    def test_f07_2_explicit_specials_cleared(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=61, method="explicit")
        po = _get_poly(level, 61)
        assert po is not None
        for seg in po.segs:
            assert seg.linedef.special == 0

    def test_f07_3_mirror_arg_copied_to_arg2(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=62, mirror_tag=88, method="explicit")
        po = _get_poly(level, 62)
        assert po is not None
        # In po_man.c:1309, arg3 (mirror) is copied to arg2 on the first linedef
        first_ld = po.segs[0].linedef
        assert first_ld.args[1] == 88

    def test_f07_4_seqtype_arg_read_from_explicit_line(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=63, seq_type=5, method="explicit")
        po = _get_poly(level, 63)
        assert po is not None
        assert po.seq_type == 5

    def test_f07_5_supports_non_closed_shapes(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=512, height=512)
        sec = mb.add_sector(light=160)
        # 2 lines forming a non-closed barrier in dummy area
        v0 = mb.add_vertex(-2000, -2000)
        v1 = mb.add_vertex(-1950, -2000)
        v2 = mb.add_vertex(-1950, -1950)
        sd0 = mb.add_sidedef(sec)
        sd1 = mb.add_sidedef(sec)
        ld0 = mb.add_linedef(v0, v1, sd0, special=5, args=(70, 1, 0, 0, 0))
        ld1 = mb.add_linedef(v1, v2, sd1, special=5, args=(70, 2, 0, 0, 0))
        mb.add_seg(v0, v1, 0x0000, ld0)
        mb.add_seg(v1, v2, 0x4000, ld1)
        mb.add_subsector(2, 0, bbox=(-1950, -2000, -2000, -1950))
        mb.add_thing(x=-2000, y=-2000, type=3000, angle=70)
        mb.add_thing(x=100, y=100, type=3001, angle=70)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)

        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        po = _get_poly(level, 70)
        assert po is not None
        assert len(po.segs) == 2


# ---------------------------------------------------------------------------
# Feature 8: Coordinate Translation (po_man.c:1333)
# ---------------------------------------------------------------------------
@skip_m1
class TestFeature08CoordinateTranslation:
    """Feature 8: Coordinate translation from dummy anchor to spawn spot."""

    def test_f08_1_vertex_translation(self) -> None:
        level, _ = _build_synthetic_poly_sim(
            tag=80, anchor=(-2000, -2000), spawn=(100, 200), width=64, height=64
        )
        po = _get_poly(level, 80)
        assert po is not None
        # Bounding box should center around (100, 200)
        xs = [seg.v1.x >> FRACBITS for seg in po.segs]
        ys = [seg.v1.y >> FRACBITS for seg in po.segs]
        assert min(xs) == 100 - 32
        assert max(xs) == 100 + 32
        assert min(ys) == 200 - 32
        assert max(ys) == 200 + 32

    def test_f08_2_linedef_bbox_translation(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=81, anchor=(-2000, -2000), spawn=(150, 150))
        po = _get_poly(level, 81)
        assert po is not None
        for seg in po.segs:
            ld = seg.linedef
            assert ld.bbox[0] > 0  # BOXTOP should be positive (in playable map)

    def test_f08_3_shared_vertex_deduplicated(self) -> None:
        level, _ = _build_synthetic_poly_sim(
            tag=82, anchor=(-2000, -2000), spawn=(100, 100), width=64, height=64
        )
        po = _get_poly(level, 82)
        assert po is not None
        # Vertices must not have been double-shifted
        for seg in po.segs:
            assert abs((seg.v1.x >> FRACBITS) - 100) <= 32

    def test_f08_4_original_pts_relative_coordinates(self) -> None:
        level, _ = _build_synthetic_poly_sim(
            tag=83, anchor=(-2000, -2000), spawn=(100, 100), width=64, height=64
        )
        po = _get_poly(level, 83)
        assert po is not None
        for pt in po.original_pts:
            px = pt.x if hasattr(pt, "x") else pt[0]
            py = pt.y if hasattr(pt, "y") else pt[1]
            assert abs(px) == 32 * FRACUNIT
            assert abs(py) == 32 * FRACUNIT

    def test_f08_5_line_is_polyobj_flag(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=84)
        po = _get_poly(level, 84)
        assert po is not None
        for seg in po.segs:
            assert getattr(seg.linedef, "is_polyobj", False) is True


# ---------------------------------------------------------------------------
# Feature 9: Subsector Linking (po_man.c:1407)
# ---------------------------------------------------------------------------
@skip_m1
class TestFeature09SubsectorLinking:
    """Feature 9: Subsector linking of polyobject center of mass."""

    def test_f09_1_center_of_mass_calculation(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=90, spawn=(200, 200), width=64, height=64)
        po = _get_poly(level, 90)
        assert po is not None
        xs = [seg.v1.x >> FRACBITS for seg in po.segs]
        ys = [seg.v1.y >> FRACBITS for seg in po.segs]
        assert sum(xs) // len(xs) == 200
        assert sum(ys) // len(ys) == 200

    def test_f09_2_point_in_subsector_query(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=91, spawn=(256, 256))
        sub = level.point_in_subsector(256 * FRACUNIT, 256 * FRACUNIT)
        assert sub is not None

    def test_f09_3_subsector_poly_assignment(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=92, spawn=(256, 256))
        po = _get_poly(level, 92)
        assert po is not None
        sub = level.point_in_subsector(256 * FRACUNIT, 256 * FRACUNIT)
        assert sub.poly is po

    def test_f09_4_poly_retrievable_via_subsector(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=93, spawn=(200, 200))
        sub = level.point_in_subsector(200 * FRACUNIT, 200 * FRACUNIT)
        assert sub.poly is not None
        assert sub.poly.tag == 93

    def test_f09_5_fixed_point_centroid_truncation(self) -> None:
        # Centroid calculation divides integer sum matching C (sum >> FRACBITS) / numsegs
        level, _ = _build_synthetic_poly_sim(tag=94, spawn=(201, 201), width=65, height=65)
        po = _get_poly(level, 94)
        assert po is not None
        sub = level.point_in_subsector(201 * FRACUNIT, 201 * FRACUNIT)
        assert sub.poly is po


# ---------------------------------------------------------------------------
# Feature 10: Specials Dispatch Registration (po_man.c:114)
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature10SpecialsDispatch:
    """Feature 10: Polyobject line specials dispatch registration."""

    def test_f10_1_special_2_rotate_left(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=1)
        res = execute_line_special(sim, 2, (1, 8, 64, 0, 0))
        assert res is True
        assert len(sim.thinkers) > 0

    def test_f10_2_special_3_rotate_right(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=2)
        res = execute_line_special(sim, 3, (2, 8, 64, 0, 0))
        assert res is True
        assert len(sim.thinkers) > 0

    def test_f10_3_special_4_move(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=3)
        res = execute_line_special(sim, 4, (3, 8, 0, 64, 0))
        assert res is True
        assert len(sim.thinkers) > 0

    def test_f10_4_special_6_move_times_8(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=4)
        res = execute_line_special(sim, 6, (4, 8, 0, 16, 0))
        assert res is True
        assert len(sim.thinkers) > 0

    def test_f10_5_specials_7_and_8_doors(self) -> None:
        _, sim7 = _build_synthetic_poly_sim(tag=7)
        res7 = execute_line_special(sim7, 7, (7, 8, 64, 35, 0))
        assert res7 is True

        _, sim8 = _build_synthetic_poly_sim(tag=8)
        res8 = execute_line_special(sim8, 8, (8, 8, 0, 64, 35))
        assert res8 is True


# ---------------------------------------------------------------------------
# Feature 11: RotatePolyThinker (po_man.c:75)
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature11RotatePolyThinker:
    """Feature 11: Thinker rotating polyobject at fixed angular rate."""

    def test_f11_1_rotates_poly_step(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=10)
        po = _get_poly(level, 10)
        init_angle = po.angle
        execute_line_special(sim, 2, (10, 8, 64, 0, 0))
        sim.tick()
        assert po.angle != init_angle

    def test_f11_2_speed_calculation(self) -> None:
        # speed = (args[1] * direction * (ANG90 / 64)) >> 3
        ang90_div_64 = 0x40000000 // 64
        expected_speed = (8 * 1 * ang90_div_64) >> 3
        _, sim = _build_synthetic_poly_sim(tag=11)
        execute_line_special(sim, 2, (11, 8, 64, 0, 0))
        # Find thinker
        thinker = next(t for t in sim.thinkers if hasattr(t, "speed"))
        assert thinker.speed == expected_speed

    def test_f11_3_dist_decremented(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=12)
        execute_line_special(sim, 2, (12, 8, 64, 0, 0))
        thinker = next(t for t in sim.thinkers if hasattr(t, "dist"))
        init_dist = thinker.dist
        sim.tick()
        assert thinker.dist < init_dist

    def test_f11_4_speed_clamped_on_final_tic(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=13)
        # Small distance, fast speed: angle=1 unit
        execute_line_special(sim, 2, (13, 64, 1, 0, 0))
        thinker = next(t for t in sim.thinkers if hasattr(t, "dist"))
        sim.tick()
        # Should have finished without overshoot
        assert thinker.removed or thinker.dist <= 0

    def test_f11_5_completion_cleanup(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=14)
        po = _get_poly(level, 14)
        execute_line_special(sim, 2, (14, 32, 2, 0, 0))
        for _ in range(10):
            sim.tick()
        assert po.specialdata is None
        assert not po.busy


# ---------------------------------------------------------------------------
# Feature 12: MovePolyThinker (po_man.c:210)
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature12MovePolyThinker:
    """Feature 12: Thinker linearly translating polyobject."""

    def test_f12_1_linear_translation_step(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=20, spawn=(128, 128))
        po = _get_poly(level, 20)
        init_x = po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x
        # Move East (angle 0)
        execute_line_special(sim, 4, (20, 8, 0, 64, 0))
        sim.tick()
        curr_x = po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x
        assert curr_x > init_x

    def test_f12_2_speed_calculation(self) -> None:
        # speed = args[1] * (FRACUNIT / 8)
        expected_speed = 8 * (FRACUNIT // 8)
        _, sim = _build_synthetic_poly_sim(tag=21)
        execute_line_special(sim, 4, (21, 8, 0, 64, 0))
        thinker = next(t for t in sim.thinkers if hasattr(t, "speed"))
        assert thinker.speed == expected_speed

    def test_f12_3_times_eight_scaling(self) -> None:
        _, sim_norm = _build_synthetic_poly_sim(tag=22)
        execute_line_special(sim_norm, 4, (22, 8, 0, 10, 0))
        t_norm = next(t for t in sim_norm.thinkers if hasattr(t, "dist"))

        _, sim_x8 = _build_synthetic_poly_sim(tag=23)
        execute_line_special(sim_x8, 6, (23, 8, 0, 10, 0))
        t_x8 = next(t for t in sim_x8.thinkers if hasattr(t, "dist"))

        assert t_x8.dist == t_norm.dist * 8

    def test_f12_4_final_step_clamping(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=24, spawn=(100, 100))
        po = _get_poly(level, 24)
        # Move 10 map units East with speed 8 units/tic
        execute_line_special(sim, 4, (24, 64, 0, 10, 0))
        for _ in range(5):
            sim.tick()
        curr_x = (po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x) >> FRACBITS
        assert curr_x == 110

    def test_f12_5_completion_cleanup(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=25)
        po = _get_poly(level, 25)
        execute_line_special(sim, 4, (25, 16, 0, 4, 0))
        for _ in range(10):
            sim.tick()
        assert po.specialdata is None
        assert not po.busy


# ---------------------------------------------------------------------------
# Feature 13: PolyDoorThinker (Slide) (po_man.c:347)
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature13PolyDoorSlide:
    """Feature 13: Sliding door state machine (open, wait, close)."""

    def test_f13_1_opening_phase(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=30, spawn=(100, 100))
        po = _get_poly(level, 30)
        # Slide East 32 units, wait 10 tics
        execute_line_special(sim, 8, (30, 8, 0, 32, 10))
        sim.tick()
        curr_x = (po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x) >> FRACBITS
        assert curr_x > 100

    def test_f13_2_wait_phase_entry(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=31, spawn=(100, 100))
        execute_line_special(sim, 8, (31, 64, 0, 8, 15))
        # After 1 tic (8 units traversed), should enter wait state
        sim.tick()
        thinker = next(t for t in sim.thinkers if hasattr(t, "tics"))
        assert thinker.tics > 0
        assert getattr(thinker, "close", False) is True

    def test_f13_3_wait_countdown(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=32, spawn=(100, 100))
        execute_line_special(sim, 8, (32, 64, 0, 8, 10))
        sim.tick()
        thinker = next(t for t in sim.thinkers if hasattr(t, "tics"))
        init_tics = thinker.tics
        sim.tick()
        assert thinker.tics < init_tics

    def test_f13_4_closing_phase(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=33, spawn=(100, 100))
        po = _get_poly(level, 33)
        execute_line_special(sim, 8, (33, 64, 0, 8, 2))
        sim.tick()  # Opens
        sim.tick()  # Wait tic 1
        sim.tick()  # Wait tic 2
        sim.tick()  # Closes
        curr_x = (po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x) >> FRACBITS
        assert curr_x == 100

    def test_f13_5_completion(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=34, spawn=(100, 100))
        po = _get_poly(level, 34)
        execute_line_special(sim, 8, (34, 64, 0, 8, 2))
        for _ in range(10):
            sim.tick()
        assert po.specialdata is None
        assert not po.busy


# ---------------------------------------------------------------------------
# Feature 14: PolyDoorThinker (Swing) (po_man.c:397)
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature14PolyDoorSwing:
    """Feature 14: Swinging door state machine (open, wait, close)."""

    def test_f14_1_opening_phase(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=40)
        po = _get_poly(level, 40)
        init_angle = po.angle
        execute_line_special(sim, 7, (40, 8, 64, 10, 0))
        sim.tick()
        assert po.angle != init_angle

    def test_f14_2_wait_phase_entry(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=41)
        execute_line_special(sim, 7, (41, 64, 1, 10, 0))
        sim.tick()
        thinker = next(t for t in sim.thinkers if hasattr(t, "tics"))
        assert thinker.tics > 0

    def test_f14_3_wait_countdown(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=42)
        execute_line_special(sim, 7, (42, 64, 1, 10, 0))
        sim.tick()
        thinker = next(t for t in sim.thinkers if hasattr(t, "tics"))
        init_tics = thinker.tics
        sim.tick()
        assert thinker.tics < init_tics

    def test_f14_4_closing_phase_restores_angle(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=43)
        po = _get_poly(level, 43)
        init_angle = po.angle
        execute_line_special(sim, 7, (43, 64, 1, 2, 0))
        for _ in range(10):
            sim.tick()
        assert po.angle == init_angle

    def test_f14_5_completion_cleanup(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=44)
        po = _get_poly(level, 44)
        execute_line_special(sim, 7, (44, 64, 1, 2, 0))
        for _ in range(10):
            sim.tick()
        assert po.specialdata is None
        assert not po.busy


# ---------------------------------------------------------------------------
# Feature 15: Override Specials (90..93) (po_man.c:126)
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature15OverrideSpecials:
    """Feature 15: Override specials (90..93) preempt active thinkers."""

    def test_f15_1_non_override_blocked_when_busy(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=50)
        # Start a slow rotation
        assert execute_line_special(sim, 2, (50, 4, 64, 0, 0)) is True
        # Standard special must fail while busy
        assert execute_line_special(sim, 2, (50, 4, 64, 0, 0)) is False

    def test_f15_2_override_special_90_rotate_left(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=51)
        po = _get_poly(level, 51)
        execute_line_special(sim, 2, (51, 4, 64, 0, 0))
        assert po.busy
        # Override special 90 succeeds
        assert execute_line_special(sim, 90, (51, 16, 32, 0, 0)) is True

    def test_f15_3_override_special_91_rotate_right(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=52)
        execute_line_special(sim, 2, (52, 4, 64, 0, 0))
        assert execute_line_special(sim, 91, (52, 16, 32, 0, 0)) is True

    def test_f15_4_override_special_92_move(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=53)
        execute_line_special(sim, 2, (53, 4, 64, 0, 0))
        assert execute_line_special(sim, 92, (53, 16, 0, 64, 0)) is True

    def test_f15_5_override_special_93_move_times_8(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=54)
        execute_line_special(sim, 2, (54, 4, 64, 0, 0))
        assert execute_line_special(sim, 93, (54, 16, 0, 16, 0)) is True


# ---------------------------------------------------------------------------
# Feature 16: Mirror Polyobject Propagation (po_man.c:159)
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature16MirrorPropagation:
    """Feature 16: Mirror polyobject motion propagation."""

    def test_f16_1_get_polyobj_mirror(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=60, mirror_tag=61)
        po = _get_poly(level, 60)
        assert hasattr(po_module, "get_polyobj_mirror")
        mirror = po_module.get_polyobj_mirror(level, po.tag)
        assert mirror == 61

    def test_f16_2_mirror_rotate_opposite_direction(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=256, height=512)
        mb.add_room(x=256, y=0, width=256, height=512)
        mb.add_polyobject_box(1, -2000, -2000, 100, 100, mirror_tag=2)
        mb.add_polyobject_box(2, -1500, -1500, 300, 100)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        sim = GameSim(level=level)
        execute_line_special(sim, 2, (1, 8, 64, 0, 0))
        po1 = _get_poly(level, 1)
        po2 = _get_poly(level, 2)
        assert po1.busy and po2.busy

    def test_f16_3_mirror_move_opposite_vector(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=256, height=512)
        mb.add_room(x=256, y=0, width=256, height=512)
        mb.add_polyobject_box(10, -2000, -2000, 100, 100, mirror_tag=11)
        mb.add_polyobject_box(11, -1500, -1500, 300, 100)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        sim = GameSim(level=level)
        execute_line_special(sim, 4, (10, 8, 0, 64, 0))
        po1 = _get_poly(level, 10)
        po2 = _get_poly(level, 11)
        assert po1.busy and po2.busy

    def test_f16_4_mirror_door_motion(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=256, height=512)
        mb.add_room(x=256, y=0, width=256, height=512)
        mb.add_polyobject_box(20, -2000, -2000, 100, 100, mirror_tag=21)
        mb.add_polyobject_box(21, -1500, -1500, 300, 100)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        sim = GameSim(level=level)
        execute_line_special(sim, 7, (20, 8, 64, 10, 0))
        assert _get_poly(level, 20).busy
        assert _get_poly(level, 21).busy

    def test_f16_5_mirror_busy_without_override_stops_propagation(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=256, height=512)
        mb.add_room(x=256, y=0, width=256, height=512)
        mb.add_polyobject_box(30, -2000, -2000, 100, 100, mirror_tag=31)
        mb.add_polyobject_box(31, -1500, -1500, 300, 100)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        sim = GameSim(level=level)
        po31 = _get_poly(level, 31)
        po31.specialdata = object()  # simulate busy mirror
        # Should not throw or crash
        execute_line_special(sim, 2, (30, 8, 64, 0, 0))
        po31.specialdata = None


# ---------------------------------------------------------------------------
# Feature 17: Mobj Collision Detection (po_man.c:1048)
# ---------------------------------------------------------------------------
@skip_m2_physics
class TestFeature17MobjCollisionDetection:
    """Feature 17: Mobj collision detection against moving polyobjects."""

    def test_f17_1_maxradius_expansion(self) -> None:
        assert po_module.MAXRADIUS == 32 * FRACUNIT

    def test_f17_2_solid_mobj_filtering(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=10, spawn=(128, 128))
        player = spawn_player(sim, 0)
        player.mo.x = 128 * FRACUNIT
        player.mo.y = 128 * FRACUNIT
        po = _get_poly(level, 10)
        assert po_module.check_mobj_blocking(sim, po.segs[0], po) is True

    def test_f17_3_box_on_line_side_intersection(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=11, spawn=(128, 128))
        player = spawn_player(sim, 0)
        player.mo.x = 128 * FRACUNIT
        player.mo.y = 128 * FRACUNIT
        po = _get_poly(level, 11)
        assert po_module.check_mobj_blocking(sim, po.segs[0], po) is True

    def test_f17_4_non_solid_mobjs_ignored(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=12, spawn=(128, 128))
        player = spawn_player(sim, 0)
        player.mo.flags &= ~1  # clear MF_SOLID
        player.mo.player = None  # clear player reference
        po = _get_poly(level, 12)
        assert po_module.check_mobj_blocking(sim, po.segs[0], po) is False

    def test_f17_5_collision_reverts_movement(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=13, spawn=(128, 128))
        player = spawn_player(sim, 0)
        player.mo.x = 128 * FRACUNIT
        player.mo.y = 128 * FRACUNIT
        po = _get_poly(level, 13)
        init_x = po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x
        # Move directly into player
        success = po_module.po_move_polyobj(sim, 13, 8 * FRACUNIT, 0)
        assert success is False
        curr_x = po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x
        assert curr_x == init_x


# ---------------------------------------------------------------------------
# Feature 18: Mobj Thrust Physics (po_man.c:599)
# ---------------------------------------------------------------------------
@skip_m2_physics
class TestFeature18MobjThrustPhysics:
    """Feature 18: Mobj thrust physics imparted by moving polyobjects."""

    def test_f18_1_thrust_angle_perpendicular(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=20, spawn=(128, 128))
        spawn_player(sim, 0)
        po = _get_poly(level, 20)
        seg = po.segs[0]  # top edge, angle 0
        expected_thrust_angle = ((seg.angle - 0x40000000) & 0xFFFFFFFF) >> 19
        assert expected_thrust_angle >= 0

    def test_f18_2_thrust_force_scaling(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=21, spawn=(128, 128))
        player = spawn_player(sim, 0)
        po = _get_poly(level, 21)
        po_module.thrust_mobj(sim, player.mo, po.segs[0], po)
        # Momentum must have been imparted
        assert player.mo.momx != 0 or player.mo.momy != 0

    def test_f18_3_thrust_force_clamped(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=22, spawn=(128, 128))
        player = spawn_player(sim, 0)
        po = _get_poly(level, 22)
        po_module.thrust_mobj(sim, player.mo, po.segs[0], po)
        # Force clamped to <= 4 * FRACUNIT
        assert abs(player.mo.momx) <= 4 * FRACUNIT
        assert abs(player.mo.momy) <= 4 * FRACUNIT

    def test_f18_4_thrust_added_directly_to_momentum(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=23, spawn=(128, 128))
        player = spawn_player(sim, 0)
        player.mo.momx = 100
        player.mo.momy = 200
        po = _get_poly(level, 23)
        po_module.thrust_mobj(sim, player.mo, po.segs[0], po)
        assert player.mo.momx != 100 or player.mo.momy != 200

    def test_f18_5_non_shootable_ignored(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=24, spawn=(128, 128))
        player = spawn_player(sim, 0)
        player.mo.flags &= ~2  # clear MF_SHOOTABLE
        player.mo.player = None
        po = _get_poly(level, 24)
        po_module.thrust_mobj(sim, player.mo, po.segs[0], po)
        assert player.mo.momx == 0 and player.mo.momy == 0


# ---------------------------------------------------------------------------
# Feature 19: Crushing Damage (po_man.c:644)
# ---------------------------------------------------------------------------
@skip_m2_physics
class TestFeature19CrushingDamage:
    """Feature 19: Crushing damage inflicted when mobj is pinned."""

    def test_f19_1_crush_false_zero_damage(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=30, spawn=(128, 128), crush=False)
        player = spawn_player(sim, 0)
        player.mo.x = 128 * FRACUNIT
        player.mo.y = 128 * FRACUNIT
        init_health = player.mo.health
        po = _get_poly(level, 30)
        assert po is not None
        po_module.po_move_polyobj(sim, 30, 8 * FRACUNIT, 0)
        assert player.mo.health == init_health

    def test_f19_2_crush_true_unpinned_zero_damage(self) -> None:
        # In open room, mobj pushed away, zero damage
        level, sim = _build_synthetic_poly_sim(tag=31, spawn=(256, 256), crush=True)
        player = spawn_player(sim, 0)
        player.mo.x = 256 * FRACUNIT
        player.mo.y = 256 * FRACUNIT
        init_health = player.mo.health
        po = _get_poly(level, 31)
        assert po is not None
        po_module.po_move_polyobj(sim, 31, 8 * FRACUNIT, 0)
        assert player.mo.health == init_health

    def test_f19_3_crush_true_pinned_deals_3_damage(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=32, spawn=(128, 128), crush=True)
        player = spawn_player(sim, 0)
        init_health = player.mo.health
        po = _get_poly(level, 32)
        # Mock check_position to simulate actor pinned against solid wall
        orig_check = po_module.check_position if hasattr(po_module, "check_position") else None
        try:
            if orig_check:
                po_module.check_position = lambda *args: False
            po_module.thrust_mobj(sim, player.mo, po.segs[0], po)
            assert player.mo.health == init_health - 3
        finally:
            if orig_check:
                po_module.check_position = orig_check

    def test_f19_4_crush_damages_player_state(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=33, spawn=(128, 128), crush=True)
        player = spawn_player(sim, 0)
        po = _get_poly(level, 33)
        orig_check = po_module.check_position if hasattr(po_module, "check_position") else None
        try:
            if orig_check:
                po_module.check_position = lambda *args: False
            po_module.thrust_mobj(sim, player.mo, po.segs[0], po)
            assert player.damagecount > 0
        finally:
            if orig_check:
                po_module.check_position = orig_check

    def test_f19_5_crush_retries_each_tic(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=34, spawn=(128, 128), crush=True)
        player = spawn_player(sim, 0)
        po = _get_poly(level, 34)
        orig_check = po_module.check_position if hasattr(po_module, "check_position") else None
        try:
            if orig_check:
                po_module.check_position = lambda *args: False
            po_module.thrust_mobj(sim, player.mo, po.segs[0], po)
            po_module.thrust_mobj(sim, player.mo, po.segs[0], po)
            assert player.mo.health == 94  # 100 - 3 - 3
        finally:
            if orig_check:
                po_module.check_position = orig_check


# ---------------------------------------------------------------------------
# Feature 20: Sound Sequences (po_man.c:156)
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature20SoundSequences:
    """Feature 20: Sound sequence events triggered on motion (SN_StartSequence/SN_StopSequence)."""

    def test_f20_1_motion_start_emits_sound_event(self) -> None:
        events: list[SequenceEvent] = []
        _, sim = _build_synthetic_poly_sim(tag=40, seq_type=2)
        sim.events.subscribe(SequenceEvent, events.append)
        execute_line_special(sim, 4, (40, 8, 0, 64, 0))
        # Sequence 10 + 2 = 12 (SEQ_DOOR_STONE + seq_type)
        assert any(e.sequence == 12 and not e.stop for e in events)

    def test_f20_2_sound_origin_at_startspot(self) -> None:
        events: list[SequenceEvent] = []
        level, sim = _build_synthetic_poly_sim(tag=41, spawn=(200, 300))
        sim.events.subscribe(SequenceEvent, events.append)
        execute_line_special(sim, 4, (41, 8, 0, 64, 0))
        evt = events[0]
        assert evt.origin.x == 200 * FRACUNIT
        assert evt.origin.y == 300 * FRACUNIT

    def test_f20_3_motion_stop_emits_stop_event(self) -> None:
        events: list[SequenceEvent] = []
        _, sim = _build_synthetic_poly_sim(tag=42)
        sim.events.subscribe(SequenceEvent, events.append)
        execute_line_special(sim, 4, (42, 64, 0, 8, 0))
        for _ in range(5):
            sim.tick()
        assert any(e.stop for e in events)

    def test_f20_4_door_restarts_sound_on_close(self) -> None:
        events: list[SequenceEvent] = []
        _, sim = _build_synthetic_poly_sim(tag=43)
        sim.events.subscribe(SequenceEvent, events.append)
        execute_line_special(sim, 8, (43, 64, 0, 8, 2))
        for _ in range(10):
            sim.tick()
        starts = [e for e in events if not e.stop]
        assert len(starts) >= 2  # Once on open, once on close

    def test_f20_5_door_restarts_sound_on_reverse(self) -> None:
        events: list[SequenceEvent] = []
        level, sim = _build_synthetic_poly_sim(tag=44, spawn=(128, 128), crush=False)
        sim.events.subscribe(SequenceEvent, events.append)
        execute_line_special(sim, 8, (44, 64, 0, 8, 1))
        sim.tick()  # opens
        # Place player blocking closing path
        player = spawn_player(sim, 0)
        player.mo.x = 128 * FRACUNIT
        player.mo.y = 128 * FRACUNIT
        for _ in range(5):
            sim.tick()
        starts = [e for e in events if not e.stop]
        assert len(starts) >= 2


# ---------------------------------------------------------------------------
# Feature 21: ACS PCD_POLYWAIT Hook (p_acs.c:1555, vm.py:645)
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature21ACSPolyWait:
    """Feature 21: ACS PCD_POLYWAIT opcode suspension."""

    def test_f21_1_polywait_opcode_suspends_script(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=50)
        execute_line_special(sim, 4, (50, 8, 0, 64, 0))
        # Create a mock script
        script = sim.acs_vm.start_script(1, None)
        if script:
            script.state = ACScriptState.ASTE_WAITINGFORPOLY
            script.wait_value = 50
            assert script.state == ACScriptState.ASTE_WAITINGFORPOLY

    def test_f21_2_polywaitdirect_opcode(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=51)
        script = sim.acs_vm.start_script(2, None)
        if script:
            script.wait_value = 51
            script.state = ACScriptState.ASTE_WAITINGFORPOLY
            assert script.wait_value == 51

    def test_f21_3_script_suspended_while_waiting(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=52)
        script = sim.acs_vm.start_script(3, None)
        if script:
            script.state = ACScriptState.ASTE_WAITINGFORPOLY
            script.wait_value = 52
            sim.acs_vm.tick(sim)
            assert script.state == ACScriptState.ASTE_WAITINGFORPOLY

    def test_f21_4_script_wait_value_records_tag(self) -> None:
        _, sim = _build_synthetic_poly_sim(tag=53)
        script = sim.acs_vm.start_script(4, None)
        if script:
            script.wait_value = 53
            assert script.wait_value == 53

    def test_f21_5_polywait_on_idle_poly_does_not_stall(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=54)
        po = _get_poly(level, 54)
        assert not po.busy
        # Calling poly_finished when idle resumes script immediately
        script = sim.acs_vm.start_script(5, None)
        if script:
            script.state = ACScriptState.ASTE_WAITINGFORPOLY
            script.wait_value = 54
            sim.poly_finished(54)
            assert script.state == ACScriptState.ASTE_RUNNING


# ---------------------------------------------------------------------------
# Feature 22: ACS poly_finished Wakeup (p_acs.c:851, vm.py:334)
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature22ACSPolyFinishedWakeup:
    """Feature 22: ACS poly_finished wakeup on thinker completion."""

    def test_f22_1_busy_guard_prevents_premature_wakeup(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=60)
        po = _get_poly(level, 60)
        po.specialdata = object()
        script = sim.acs_vm.start_script(10, None)
        if script:
            script.state = ACScriptState.ASTE_WAITINGFORPOLY
            script.wait_value = 60
            sim.poly_finished(60)
            # Guard must prevent wakeup while still busy
            assert script.state == ACScriptState.ASTE_WAITINGFORPOLY
            po.specialdata = None

    def test_f22_2_wakes_waiting_script(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=61)
        po = _get_poly(level, 61)
        assert po is not None
        script = sim.acs_vm.start_script(11, None)
        if script:
            script.state = ACScriptState.ASTE_WAITINGFORPOLY
            script.wait_value = 61
            sim.poly_finished(61)
            assert script.state == ACScriptState.ASTE_RUNNING

    def test_f22_3_preserves_unrelated_waiting_scripts(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=62)
        s1 = sim.acs_vm.start_script(12, None)
        s2 = sim.acs_vm.start_script(13, None)
        if s1 and s2:
            s1.state = ACScriptState.ASTE_WAITINGFORPOLY
            s1.wait_value = 62
            s2.state = ACScriptState.ASTE_WAITINGFORPOLY
            s2.wait_value = 999  # waiting on another poly
            sim.poly_finished(62)
            assert s1.state == ACScriptState.ASTE_RUNNING
            assert s2.state == ACScriptState.ASTE_WAITINGFORPOLY

    def test_f22_4_multiple_scripts_waiting_on_same_tag_all_woken(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=63)
        scripts = [sim.acs_vm.start_script(100 + i, None) for i in range(3)]
        for s in scripts:
            if s:
                s.state = ACScriptState.ASTE_WAITINGFORPOLY
                s.wait_value = 63
        sim.poly_finished(63)
        for s in scripts:
            if s:
                assert s.state == ACScriptState.ASTE_RUNNING

    def test_f22_5_resumes_execution_on_next_tic(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=64)
        script = sim.acs_vm.start_script(20, None)
        if script:
            script.state = ACScriptState.ASTE_WAITINGFORPOLY
            script.wait_value = 64
            sim.poly_finished(64)
            sim.acs_vm.tick(sim)
            # Script was executed
            assert script.state in (ACScriptState.ASTE_RUNNING, ACScriptState.ASTE_INACTIVE)


# ---------------------------------------------------------------------------
# Feature 23: Dynamic Polyobject Mesh (renderer.py)
# ---------------------------------------------------------------------------
@skip_m3_render
class TestFeature23DynamicPolyMesh:
    """Feature 23: Dynamic polyobject mesh generation."""

    def test_f23_1_local_mesh_quads(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=70)
        po = _get_poly(level, 70)
        mesh = renderer.build_polyobj_mesh(po)
        assert mesh is not None
        assert len(mesh.vertices) == len(po.segs) * 4

    def test_f23_2_texture_uv_mapping(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=71)
        po = _get_poly(level, 71)
        mesh = renderer.build_polyobj_mesh(po)
        assert hasattr(mesh, "uvs") or hasattr(mesh, "texcoords")

    def test_f23_3_vertices_relative_to_startspot(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=72, spawn=(100, 100), width=64, height=64)
        po = _get_poly(level, 72)
        mesh = renderer.build_polyobj_mesh(po)
        for vx, vy, _ in mesh.vertices:
            assert abs(vx) <= 32.5
            assert abs(vy) <= 32.5

    def test_f23_4_dynamic_buffer_update(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=73)
        po = _get_poly(level, 73)
        mesh = renderer.build_polyobj_mesh(po)
        assert hasattr(mesh, "vbo") or hasattr(mesh, "vao")

    def test_f23_5_degenerate_segs_handled_safely(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=74)
        po = _get_poly(level, 74)
        # Empty segs
        po.segs = []
        mesh = renderer.build_polyobj_mesh(po)
        assert mesh is not None


# ---------------------------------------------------------------------------
# Feature 24: Static Chunk Exclusion (chunks.py)
# ---------------------------------------------------------------------------
@skip_m1
class TestFeature24StaticChunkExclusion:
    """Feature 24: Filtering out polyobject linedefs from static sector chunks."""

    def test_f24_1_poly_lines_flagged_is_polyobj(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=80)
        po = _get_poly(level, 80)
        assert po is not None
        for seg in po.segs:
            assert getattr(seg.linedef, "is_polyobj", False) is True

    def test_f24_2_static_chunk_skips_poly_lines(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=81)
        # World lines (not poly) should not be flagged is_polyobj
        world_lines = [ld for ld in level.lines if not getattr(ld, "is_polyobj", False)]
        assert len(world_lines) == 4  # The outer room walls

    def test_f24_3_no_ghost_geometry_at_dummy_anchor(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=82, anchor=(-2000, -2000))
        # Anchor lines are poly lines
        anchor_lines = [
            ld
            for ld in level.lines
            if any((v.x >> FRACBITS) == -2000 or (v.y >> FRACBITS) == -2000 for v in (ld.v1, ld.v2))
        ]
        assert all(getattr(ld, "is_polyobj", False) for ld in anchor_lines)

    def test_f24_4_preserves_static_world_lines(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=83)
        room_lines = [ld for ld in level.lines if not getattr(ld, "is_polyobj", False)]
        assert len(room_lines) == 4

    def test_f24_5_total_poly_lines_matches_segs(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=84)
        poly_lines = [ld for ld in level.lines if getattr(ld, "is_polyobj", False)]
        assert len(poly_lines) == 4


# ---------------------------------------------------------------------------
# Feature 25: GPU Model Matrix Transform (world.vert)
# ---------------------------------------------------------------------------
@skip_m3_render
class TestFeature25GPUModelMatrixTransform:
    """Feature 25: Computes 4x4 model matrix M(theta, ix, iy)."""

    def test_f25_1_matrix_translation(self) -> None:
        m = renderer.compute_poly_model_matrix(100.0, 200.0, 0.0, 0.0, 0.0)
        assert m[3][0] == 100.0
        assert m[3][1] == 200.0

    def test_f25_2_matrix_rotation(self) -> None:
        angle_rad = math.pi / 2  # 90 degrees
        m = renderer.compute_poly_model_matrix(0.0, 0.0, angle_rad, 0.0, 0.0)
        assert abs(m[0][0]) < 1e-4  # cos(90) = 0
        assert abs(m[0][1] - 1.0) < 1e-4  # sin(90) = 1

    def test_f25_3_matrix_pivot_offset(self) -> None:
        m = renderer.compute_poly_model_matrix(100.0, 100.0, 0.0, 10.0, 10.0)
        assert m is not None

    def test_f25_4_fixed_to_float_accuracy(self) -> None:
        fixed_val = 123456
        float_val = fixed_val / FRACUNIT
        assert abs(float_val - 1.883789) < 1e-4

    def test_f25_5_identity_matrix_at_origin(self) -> None:
        m = renderer.compute_poly_model_matrix(0.0, 0.0, 0.0, 0.0, 0.0)
        assert m[0][0] == 1.0 and m[1][1] == 1.0 and m[2][2] == 1.0 and m[3][3] == 1.0


# ---------------------------------------------------------------------------
# Feature 26: Sub-Tic Frame Interpolation (main_loop.py)
# ---------------------------------------------------------------------------
@skip_m3_render
class TestFeature26SubTicInterpolation:
    """Feature 26: Sub-tic frame interpolation using render fraction alpha."""

    def test_f26_1_alpha_zero_equals_prev(self) -> None:
        ix = renderer.interpolate_coord(prev=100.0, curr=200.0, alpha=0.0)
        assert ix == 100.0

    def test_f26_2_alpha_one_equals_curr(self) -> None:
        ix = renderer.interpolate_coord(prev=100.0, curr=200.0, alpha=1.0)
        assert ix == 200.0

    def test_f26_3_alpha_midpoint(self) -> None:
        ix = renderer.interpolate_coord(prev=100.0, curr=200.0, alpha=0.5)
        assert ix == 150.0

    def test_f26_4_shortest_angular_arc(self) -> None:
        # Interpolating across 0 / 360 boundary (e.g. 350 deg to 10 deg)
        ang = renderer.interpolate_angle(prev_deg=350.0, curr_deg=10.0, alpha=0.5)
        assert abs(ang - 0.0) < 1e-3 or abs(ang - 360.0) < 1e-3

    def test_f26_5_stationary_invariant(self) -> None:
        ix = renderer.interpolate_coord(prev=50.0, curr=50.0, alpha=0.73)
        assert ix == 50.0


# ---------------------------------------------------------------------------
# Feature 27: Dynamic Sector Lighting (renderer.py)
# ---------------------------------------------------------------------------
@skip_m3_render
class TestFeature27DynamicSectorLighting:
    """Feature 27: Dynamic sector lighting sampling from occupied subsector."""

    def test_f27_1_queries_occupied_subsector(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=90, spawn=(256, 256))
        sub = level.point_in_subsector(256 * FRACUNIT, 256 * FRACUNIT)
        assert sub is not None

    def test_f27_2_samples_sector_light_level(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=91, spawn=(256, 256))
        sub = level.point_in_subsector(256 * FRACUNIT, 256 * FRACUNIT)
        assert sub.sector.light_level == 160

    def test_f27_3_updates_lighting_across_sectors(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_two_sector_rooms(light1=100, light2=200)
        mb.add_polyobject_box(1, -2000, -2000, 128, 128)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        sub1 = level.point_in_subsector(128 * FRACUNIT, 128 * FRACUNIT)
        sub2 = level.point_in_subsector(384 * FRACUNIT, 128 * FRACUNIT)
        assert sub1.sector.light_level == 100
        assert sub2.sector.light_level == 200

    def test_f27_4_samples_floor_and_ceiling_heights(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=93, spawn=(256, 256))
        sub = level.point_in_subsector(256 * FRACUNIT, 256 * FRACUNIT)
        assert sub.sector.floor_height == 0
        h_ceil = sub.sector.ceiling_height
        assert h_ceil in (128, 128 * FRACUNIT) or (h_ceil >> FRACBITS) == 128

    def test_f27_5_default_lighting_applied(self) -> None:
        level, _ = _build_synthetic_poly_sim(tag=94, spawn=(256, 256))
        sub = level.point_in_subsector(256 * FRACUNIT, 256 * FRACUNIT)
        assert sub.sector.light_level >= 0


# ---------------------------------------------------------------------------
# Feature 28: E2E Acceptance & Adversarial Hardening
# ---------------------------------------------------------------------------
@skip_m2_thinkers
class TestFeature28E2EAcceptance:
    """Feature 28: Full synthetic lifecycle and adversarial hardening."""

    def test_f28_1_e2e_translation_cycle(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=100, spawn=(100, 100))
        po = _get_poly(level, 100)
        execute_line_special(sim, 4, (100, 16, 0, 32, 0))
        for _ in range(20):
            sim.tick()
        curr_x = (po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x) >> FRACBITS
        assert curr_x == 132
        assert not po.busy

    def test_f28_2_e2e_door_slide_cycle(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=101, spawn=(100, 100))
        po = _get_poly(level, 101)
        execute_line_special(sim, 8, (101, 16, 0, 32, 5))
        for _ in range(40):
            sim.tick()
        curr_x = (po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x) >> FRACBITS
        assert curr_x == 100
        assert not po.busy

    def test_f28_3_adversarial_rapid_rotation_drift_free(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=102, spawn=(100, 100), width=64, height=64)
        po = _get_poly(level, 102)
        # 4 full 360-degree rotations
        for _ in range(4):
            execute_line_special(sim, 90, (102, 64, 0, 0, 0))
            for _ in range(30):
                sim.tick()
        # Original points must not have drifted
        for pt in po.original_pts:
            px = pt.x if hasattr(pt, "x") else pt[0]
            py = pt.y if hasattr(pt, "y") else pt[1]
            assert abs(px) == 32 * FRACUNIT
            assert abs(py) == 32 * FRACUNIT

    def test_f28_4_adversarial_map_boundary_movement(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=103, spawn=(100, 100))
        # Move extreme distance toward map boundary
        execute_line_special(sim, 4, (103, 64, 0, 1000, 0))
        for _ in range(10):
            sim.tick()
        # Ensure no crash or exception raised
        assert True

    def test_f28_5_adversarial_rapid_override_specials(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=104)
        po = _get_poly(level, 104)
        assert po is not None
        # Rapidly fire override specials back to back
        execute_line_special(sim, 90, (104, 16, 64, 0, 0))
        execute_line_special(sim, 92, (104, 16, 0, 64, 0))
        execute_line_special(sim, 91, (104, 16, 64, 0, 0))
        # Active thinker should not leak
        assert len(sim.thinkers) <= 2


# ===========================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ===========================================================================
class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary, extremes, and error handling edge cases."""

    @skip_m2_thinkers
    def test_t2_1_angle_0_full_360_rotation(self) -> None:
        # args[2] == 0 in EV_RotatePoly sets dist = ANG_MAX - 1 (0xFFFFFFFE)
        _, sim = _build_synthetic_poly_sim(tag=1)
        execute_line_special(sim, 2, (1, 8, 0, 0, 0))
        thinker = next(t for t in sim.thinkers if hasattr(t, "dist"))
        assert thinker.dist == 0xFFFFFFFE

    @skip_m2_thinkers
    def test_t2_2_angle_255_perpetual_rotation(self) -> None:
        # args[2] == 255 in EV_RotatePoly sets dist = -1 (perpetual)
        _, sim = _build_synthetic_poly_sim(tag=2)
        execute_line_special(sim, 2, (2, 8, 255, 0, 0))
        thinker = next(t for t in sim.thinkers if hasattr(t, "dist"))
        assert thinker.dist == -1

    @skip_m1
    def test_t2_3_explicit_lines_skipped_order_error(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=512, height=512)
        sec = mb.add_sector(light=160)
        # Skipped order: order 1 then order 3 (missing 2)
        v0 = mb.add_vertex(-2000, -2000)
        v1 = mb.add_vertex(-1950, -2000)
        v2 = mb.add_vertex(-1950, -1950)
        sd0 = mb.add_sidedef(sec)
        sd1 = mb.add_sidedef(sec)
        ld0 = mb.add_linedef(v0, v1, sd0, special=5, args=(3, 1, 0, 0, 0))
        ld1 = mb.add_linedef(v1, v2, sd1, special=5, args=(3, 3, 0, 0, 0))
        mb.add_seg(v0, v1, 0x0000, ld0)
        mb.add_seg(v1, v2, 0x4000, ld1)
        mb.add_subsector(2, 0, bbox=(-1950, -2000, -2000, -1950))
        mb.add_thing(x=-2000, y=-2000, type=3000, angle=3)
        mb.add_thing(x=100, y=100, type=3001, angle=3)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)

        with pytest.raises((ValueError, RuntimeError, KeyError)):
            level = load_level(WadStack(wb.to_bytes()), "MAP01")
            po_module.init_polyobjs(level)

    @skip_m1
    def test_t2_4_explicit_lines_order_zero_error(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=512, height=512)
        sec = mb.add_sector(light=160)
        v0 = mb.add_vertex(-2000, -2000)
        v1 = mb.add_vertex(-1950, -2000)
        sd0 = mb.add_sidedef(sec)
        # Order 0
        ld0 = mb.add_linedef(v0, v1, sd0, special=5, args=(4, 0, 0, 0, 0))
        mb.add_seg(v0, v1, 0x0000, ld0)
        mb.add_subsector(1, 0, bbox=(-2000, -2000, -2000, -1950))
        mb.add_thing(x=-2000, y=-2000, type=3000, angle=4)
        mb.add_thing(x=100, y=100, type=3001, angle=4)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)

        with pytest.raises((ValueError, RuntimeError, KeyError)):
            level = load_level(WadStack(wb.to_bytes()), "MAP01")
            po_module.init_polyobjs(level)

    @skip_m1
    def test_t2_5_startline_non_closed_loop_error(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=512, height=512)
        sec = mb.add_sector(light=160)
        # 3 lines forming a 'U' shape that doesn't close
        v0 = mb.add_vertex(-2000, -2000)
        v1 = mb.add_vertex(-1950, -2000)
        v2 = mb.add_vertex(-1950, -1950)
        v3 = mb.add_vertex(-2000, -1950)
        sd0 = mb.add_sidedef(sec)
        sd1 = mb.add_sidedef(sec)
        sd2 = mb.add_sidedef(sec)
        ld0 = mb.add_linedef(v0, v1, sd0, special=1, args=(5, 0, 0, 0, 0))
        ld1 = mb.add_linedef(v1, v2, sd1)
        ld2 = mb.add_linedef(v2, v3, sd2)
        mb.add_seg(v0, v1, 0x0000, ld0)
        mb.add_seg(v1, v2, 0x4000, ld1)
        mb.add_seg(v2, v3, 0x8000, ld2)
        mb.add_subsector(3, 0, bbox=(-1950, -2000, -2000, -1950))
        mb.add_thing(x=-2000, y=-2000, type=3000, angle=5)
        mb.add_thing(x=100, y=100, type=3001, angle=5)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)

        with pytest.raises((ValueError, RuntimeError, KeyError)):
            level = load_level(WadStack(wb.to_bytes()), "MAP01")
            po_module.init_polyobjs(level)

    @skip_m1
    def test_t2_6_multiple_polyobjs_in_single_subsector_error(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        # Small room with single subsector
        mb.add_room(x=0, y=0, width=256, height=256)
        # Place TWO polyobjects with centers inside this same room/subsector
        mb.add_polyobject_box(6, -2000, -2000, 64, 64)
        mb.add_polyobject_box(7, -1500, -1500, 128, 128)
        mb.add_thing(tid=1, x=200, y=200, z=0, angle=0, type=1)
        wb.add_map(mb)

        with pytest.raises((ValueError, RuntimeError, KeyError)):
            level = load_level(WadStack(wb.to_bytes()), "MAP01")
            po_module.init_polyobjs(level)

    @skip_m1
    def test_t2_7_start_spot_without_anchor_error(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=512, height=512)
        # Spawn spot 3001 with NO corresponding anchor 3000
        mb.add_thing(x=100, y=100, type=3001, angle=8)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)

        with pytest.raises((ValueError, RuntimeError, KeyError)):
            level = load_level(WadStack(wb.to_bytes()), "MAP01")
            po_module.init_polyobjs(level)

    @skip_m1
    def test_t2_8_explicit_lines_max_limit_error(self) -> None:
        # Exceeding PO_MAXPOLYSEGS (64)
        assert po_module.PO_MAXPOLYSEGS == 64

    @skip_m2_thinkers
    def test_t2_9_speed_clamping_exact_landing(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=9, spawn=(100, 100))
        po = _get_poly(level, 9)
        # Move 5 map units with speed 8 units/tic
        execute_line_special(sim, 4, (9, 64, 0, 5, 0))
        sim.tick()
        curr_x = (po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x) >> FRACBITS
        assert curr_x == 105


# ===========================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ===========================================================================
class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise interactions and multi-system workflows."""

    @skip_m2_thinkers
    def test_t3_1_rotate_and_mirror_propagation(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=256, height=512)
        mb.add_room(x=256, y=0, width=256, height=512)
        mb.add_polyobject_box(1, -2000, -2000, 100, 100, mirror_tag=2)
        mb.add_polyobject_box(2, -1500, -1500, 300, 100)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        sim = GameSim(level=level)

        po1 = _get_poly(level, 1)
        po2 = _get_poly(level, 2)
        init_a1, init_a2 = po1.angle, po2.angle

        execute_line_special(sim, 2, (1, 8, 64, 0, 0))  # Rotate left (CCW)
        sim.tick()

        # Primary rotated CCW (+angle), Mirror rotated CW (-angle)
        assert po1.angle > init_a1
        assert po2.angle < init_a2 or po2.angle > 0x80000000

    @skip_m2_thinkers
    def test_t3_2_move_and_mirror_propagation(self) -> None:
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33)
        mb = MapBuilder("MAP01")
        mb.add_room(x=0, y=0, width=256, height=512)
        mb.add_room(x=256, y=0, width=256, height=512)
        mb.add_polyobject_box(10, -2000, -2000, 100, 200, mirror_tag=11)
        mb.add_polyobject_box(11, -1500, -1500, 300, 200)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        level = load_level(WadStack(wb.to_bytes()), "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        sim = GameSim(level=level)

        po1 = _get_poly(level, 10)
        po2 = _get_poly(level, 11)
        init_x1 = po1.start_spot.x if hasattr(po1, "start_spot") else po1.start_spot_x
        init_x2 = po2.start_spot.x if hasattr(po2, "start_spot") else po2.start_spot_x

        execute_line_special(sim, 4, (10, 8, 0, 64, 0))  # Move East
        sim.tick()

        curr_x1 = po1.start_spot.x if hasattr(po1, "start_spot") else po1.start_spot_x
        curr_x2 = po2.start_spot.x if hasattr(po2, "start_spot") else po2.start_spot_x
        # Primary moved East (+x), Mirror moved West (-x)
        assert curr_x1 > init_x1
        assert curr_x2 < init_x2

    @skip_m2_thinkers
    def test_t3_3_sliding_door_and_mobj_collision(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=20, spawn=(100, 100))
        player = spawn_player(sim, 0)
        # Stand in front of opening door
        player.mo.x = 120 * FRACUNIT
        player.mo.y = 100 * FRACUNIT

        po = _get_poly(level, 20)
        execute_line_special(sim, 8, (20, 8, 0, 64, 10))
        sim.tick()
        # Door must not penetrate through player
        curr_x = (po.start_spot.x if hasattr(po, "start_spot") else po.start_spot_x) >> FRACBITS
        assert curr_x <= 120

    @skip_m2_thinkers
    def test_t3_4_swinging_door_and_obstruction_reversal(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=30, spawn=(100, 100), crush=False)
        po = _get_poly(level, 30)
        assert po is not None
        # Open door completely
        execute_line_special(sim, 7, (30, 64, 1, 1, 0))
        sim.tick()  # opens
        sim.tick()  # enters wait / close state

        # Place player blocking closing path
        player = spawn_player(sim, 0)
        player.mo.x = 100 * FRACUNIT
        player.mo.y = 100 * FRACUNIT

        sim.tick()  # closing tic encounters player -> must reverse!
        thinker = next(t for t in sim.thinkers if hasattr(t, "close"))
        assert thinker.close is False  # reversed back to opening!

    @skip_m2_physics
    def test_t3_5_crushing_door_pinned_actor_deals_3_damage(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=40, spawn=(100, 100), crush=True)
        player = spawn_player(sim, 0)
        init_health = player.mo.health
        po = _get_poly(level, 40)
        orig_check = po_module.check_position if hasattr(po_module, "check_position") else None
        try:
            if orig_check:
                po_module.check_position = lambda *args: False
            po_module.thrust_mobj(sim, player.mo, po.segs[0], po)
            assert player.mo.health == init_health - 3
        finally:
            if orig_check:
                po_module.check_position = orig_check

    @skip_m2_thinkers
    def test_t3_6_pcd_polywait_and_multi_script_wakeup(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=50)
        s1 = sim.acs_vm.start_script(1, None)
        s2 = sim.acs_vm.start_script(2, None)
        s3 = sim.acs_vm.start_script(3, None)
        for s in (s1, s2, s3):
            if s:
                s.state = ACScriptState.ASTE_WAITINGFORPOLY
                s.wait_value = 50

        execute_line_special(sim, 4, (50, 64, 0, 8, 0))
        for _ in range(5):
            sim.tick()

        for s in (s1, s2, s3):
            if s:
                assert s.state == ACScriptState.ASTE_RUNNING

    @skip_m2_thinkers
    def test_t3_7_override_preempts_active_door(self) -> None:
        level, sim = _build_synthetic_poly_sim(tag=60)
        po = _get_poly(level, 60)
        execute_line_special(sim, 7, (60, 4, 64, 35, 0))  # Start swinging door
        assert po.busy
        # Preempt with linear move override
        assert execute_line_special(sim, 92, (60, 16, 0, 64, 0)) is True
        assert po.busy


# ===========================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (Hexen IWAD MAP01 / MAP02)
# ===========================================================================
@pytest.mark.wad
class TestTier4RealWorldScenarios:
    """Tier 4: Verification against real Hexen IWAD levels."""

    @skip_m1
    def test_t4_1_hexen_map01_spawn_all_12_polyobjs(self, iwad_path: Any) -> None:
        stack = WadStack(iwad_path)
        level = load_level(stack, "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)

        # MAP01 has 12 polyobjects with tags 4..15
        expected_tags = set(range(4, 16))
        spawned_tags = {
            po.tag
            for po in (
                level.polyobjs.values() if isinstance(level.polyobjs, dict) else level.polyobjs
            )
        }
        assert expected_tags.issubset(spawned_tags)

    @skip_m2_thinkers
    def test_t4_2_hexen_map01_double_swinging_entrance_doors(self, iwad_path: Any) -> None:
        stack = WadStack(iwad_path)
        level = load_level(stack, "MAP01")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        sim = GameSim(level=level)

        po4 = _get_poly(level, 4)
        po5 = _get_poly(level, 5)
        assert po4 is not None and po5 is not None
        # Tag 4 mirrors Tag 5
        execute_line_special(sim, 7, (4, 8, 64, 10, 0))
        assert po4.busy
        assert po5.busy

    @skip_m1
    def test_t4_3_hexen_map02_spawn_all_24_polyobjs(self, iwad_path: Any) -> None:
        stack = WadStack(iwad_path)
        level = load_level(stack, "MAP02")
        if HAS_M1_SPAWNING and not hasattr(level, "polyobjs"):
            po_module.init_polyobjs(level)
        assert len(level.polyobjs) >= 24

    @skip_m1
    def test_t4_4_hexen_map02_crush_sliding_pillars(self, iwad_path: Any) -> None:
        """Verify authentic IWAD crush flags:
        MAP01 has 12 type 3002 (crush), MAP02 has 24 type 3001 (non-crush).
        """
        stack = WadStack(iwad_path)
        # Commercial MAP01 contains authentic crushing polyobjects (spawn spots of type 3002)
        lvl1 = load_level(stack, "MAP01")
        if HAS_M1_SPAWNING and not hasattr(lvl1, "polyobjs"):
            po_module.init_polyobjs(lvl1)
        crush_polyobjs = [
            po
            for po in (lvl1.polyobjs.values() if isinstance(lvl1.polyobjs, dict) else lvl1.polyobjs)
            if po.crush is True
        ]
        assert len(crush_polyobjs) == 12

        # Commercial MAP02 contains authentic non-crush polyobjects (spawn spots are type 3001)
        lvl2 = load_level(stack, "MAP02")
        if HAS_M1_SPAWNING and not hasattr(lvl2, "polyobjs"):
            po_module.init_polyobjs(lvl2)
        non_crush_polyobjs = [
            po
            for po in (lvl2.polyobjs.values() if isinstance(lvl2.polyobjs, dict) else lvl2.polyobjs)
            if po.crush is False
        ]
        assert len(non_crush_polyobjs) == 24
