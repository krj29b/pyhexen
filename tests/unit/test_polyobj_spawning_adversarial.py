# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Adversarial stress-testing suite for Milestone 1 (T9.1 Polyobject Spawning).

Challenges boundary conditions, non-convex loops, order edge cases, coordinate extremes,
vertex deduplication, and blockmap spatial indexing.
"""

from __future__ import annotations

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACBITS
from pyhexen.sim.polyobj import (
    PO_ANCHOR_TYPE,
    PO_LINE_EXPLICIT,
    PO_LINE_START,
    PO_MAXPOLYSEGS,
    PO_SPAWN_TYPE,
    link_polyobj,
    unlink_polyobj,
)
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import (
    BOXBOTTOM,
    BOXLEFT,
    BOXRIGHT,
    BOXTOP,
    load_level,
)


def _build_map_wad(mb: MapBuilder) -> WadStack:
    """Helper to pack a MapBuilder into a WadStack."""
    wb = WadBuilder("IWAD").add_map(mb)
    return WadStack(wb.to_bytes())


# ==============================================================================
# Challenge Category 1: Polyobj_StartLine Loops & Traversal
# ==============================================================================


def test_startline_non_closed_missing_terminal_seg() -> None:
    """Challenge: Non-closed seg loop where chain abruptly terminates without returning."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-1000, -900)
    v2 = mb.add_vertex(-900, -900)
    sd = mb.add_sidedef(sec_dummy)

    # v0 -> v1 (StartLine)
    ld1 = mb.add_linedef(v0, v1, sd, special=PO_LINE_START, args=(1, 0, 0, 0, 0))
    mb.add_seg(v0, v1, 0, ld1)
    # v1 -> v2 (Regular line)
    ld2 = mb.add_linedef(v1, v2, sd, special=0)
    mb.add_seg(v1, v2, 0, ld2)
    # No seg connecting v2 back to v0!
    mb.add_subsector(2, 0, bbox=(-900, -1000, -1000, -900))

    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=1)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=1)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="IterFindPolySegs:  Non-closed Polyobj located"):
            load_level(wad, "MAP01")


def test_startline_lasso_loop_infinite_cycle() -> None:
    """Challenge: Lasso loop that cycles between sub-vertices without returning to start vertex."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-1000, -900)
    v2 = mb.add_vertex(-900, -900)
    sd = mb.add_sidedef(sec_dummy)

    # v0 -> v1 (StartLine)
    ld1 = mb.add_linedef(v0, v1, sd, special=PO_LINE_START, args=(2, 0, 0, 0, 0))
    mb.add_seg(v0, v1, 0, ld1)
    # v1 -> v2
    ld2 = mb.add_linedef(v1, v2, sd, special=0)
    mb.add_seg(v1, v2, 0, ld2)
    # v2 -> v1 (Cycles back to v1, never reaching v0!)
    ld3 = mb.add_linedef(v2, v1, sd, special=0)
    mb.add_seg(v2, v1, 0, ld3)
    mb.add_subsector(3, 0, bbox=(-900, -1000, -1000, -900))

    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=2)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=2)

    with _build_map_wad(mb) as wad:
        # Must terminate and raise Non-closed Polyobj located error via max_segs guard
        with pytest.raises(RuntimeError, match="IterFindPolySegs:  Non-closed Polyobj located"):
            load_level(wad, "MAP01")


def test_startline_concave_l_shape_polygon() -> None:
    """Challenge: Concave 6-sided L-shaped polygon traversal."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    # 6 vertices forming an L-shape
    coords = [
        (-1000, -1000),  # v0
        (-940, -1000),  # v1
        (-940, -980),  # v2
        (-980, -980),  # v3
        (-980, -940),  # v4
        (-1000, -940),  # v5
    ]
    verts = [mb.add_vertex(x, y) for x, y in coords]
    sd = mb.add_sidedef(sec_dummy)

    first_seg = len(mb.segs)
    for i in range(len(verts)):
        v_curr = verts[i]
        v_next = verts[(i + 1) % len(verts)]
        sp = PO_LINE_START if i == 0 else 0
        args = (3, 0, 1, 0, 0) if i == 0 else (0, 0, 0, 0, 0)
        ld = mb.add_linedef(v_curr, v_next, sd, special=sp, args=args)
        mb.add_seg(v_curr, v_next, 0, ld)

    mb.add_subsector(6, first_seg, bbox=(-940, -1000, -1000, -940))
    mb.add_thing(x=-970, y=-970, type=PO_ANCHOR_TYPE, angle=3)
    mb.add_thing(x=150, y=150, type=PO_SPAWN_TYPE, angle=3)

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[3]
    assert po.numsegs == 6
    assert len(po.segs) == 6
    assert len(po.original_pts) == 6
    assert len(po.prev_pts) == 6
    assert po.subsector is not None
    # All 6 segs must be in continuous head-to-tail order
    for i in range(len(po.segs)):
        assert po.segs[i].v2 is po.segs[(i + 1) % 6].v1


def test_startline_concave_cross_12_seg_polygon() -> None:
    """Challenge: Concave 12-sided cross-shaped polygon traversal."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    # 12 vertices forming a plus sign
    cross_coords = [
        (-1010, -1030),
        (-990, -1030),
        (-990, -1010),
        (-970, -1010),
        (-970, -990),
        (-990, -990),
        (-990, -970),
        (-1010, -970),
        (-1010, -990),
        (-1030, -990),
        (-1030, -1010),
        (-1010, -1010),
    ]
    verts = [mb.add_vertex(x, y) for x, y in cross_coords]
    sd = mb.add_sidedef(sec_dummy)

    first_seg = len(mb.segs)
    for i in range(12):
        v_curr = verts[i]
        v_next = verts[(i + 1) % 12]
        sp = PO_LINE_START if i == 0 else 0
        args = (4, 0, 0, 0, 0) if i == 0 else (0, 0, 0, 0, 0)
        ld = mb.add_linedef(v_curr, v_next, sd, special=sp, args=args)
        mb.add_seg(v_curr, v_next, 0, ld)

    mb.add_subsector(12, first_seg, bbox=(-970, -1030, -1030, -970))
    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=4)
    mb.add_thing(x=200, y=200, type=PO_SPAWN_TYPE, angle=4)

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[4]
    assert po.numsegs == 12
    assert len(po.segs) == 12
    # Verify cyclic continuity
    for i in range(12):
        assert po.segs[i].v2 is po.segs[(i + 1) % 12].v1


def test_startline_self_intersecting_hourglass_polygon() -> None:
    """Challenge: Self-intersecting / hourglass polygon topology."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    # 4 vertices where diagonal edges cross in an hourglass: v0->v1->v2->v3->v0
    coords = [
        (-1000, -1000),  # v0 bottom-left
        (-960, -960),  # v1 top-right (diagonal)
        (-960, -1000),  # v2 bottom-right (vertical down)
        (-1000, -960),  # v3 top-left (diagonal crossing)
    ]
    verts = [mb.add_vertex(x, y) for x, y in coords]
    sd = mb.add_sidedef(sec_dummy)

    first_seg = len(mb.segs)
    for i in range(4):
        v_curr = verts[i]
        v_next = verts[(i + 1) % 4]
        sp = PO_LINE_START if i == 0 else 0
        args = (5, 0, 0, 0, 0) if i == 0 else (0, 0, 0, 0, 0)
        ld = mb.add_linedef(v_curr, v_next, sd, special=sp, args=args)
        mb.add_seg(v_curr, v_next, 0, ld)

    mb.add_subsector(4, first_seg, bbox=(-960, -1000, -1000, -960))
    mb.add_thing(x=-980, y=-980, type=PO_ANCHOR_TYPE, angle=5)
    mb.add_thing(x=250, y=250, type=PO_SPAWN_TYPE, angle=5)

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[5]
    assert po.numsegs == 4
    for i in range(4):
        assert po.segs[i].v2 is po.segs[(i + 1) % 4].v1


# ==============================================================================
# Challenge Category 2: Polyobj_ExplicitLine Orders & Limits
# ==============================================================================


def test_explicit_lines_out_of_order_in_map() -> None:
    """Challenge: Explicit lines defined out of order in map (orders 4, 2, 1, 3)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-950, -1000)
    v2 = mb.add_vertex(-950, -950)
    v3 = mb.add_vertex(-1000, -950)
    sd = mb.add_sidedef(sec_dummy)

    # Shuffled order of definition in linedef lump: 4, 2, 1, 3
    shuffled_lines = [
        (v3, v0, 4, 0, 0),  # Order 4
        (v1, v2, 2, 0, 0),  # Order 2
        (v0, v1, 1, 99, 5),  # Order 1 (mirror_tag=99, seq_type=5)
        (v2, v3, 3, 0, 0),  # Order 3
    ]

    first_seg = len(mb.segs)
    for v_start, v_end, order, mirror, seq in shuffled_lines:
        args = (10, order, mirror, seq, 0)
        ld = mb.add_linedef(v_start, v_end, sd, special=PO_LINE_EXPLICIT, args=args)
        mb.add_seg(v_start, v_end, 0, ld)

    mb.add_subsector(4, first_seg, bbox=(-950, -1000, -1000, -950))
    mb.add_thing(x=-975, y=-975, type=PO_ANCHOR_TYPE, angle=10)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=10)

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[10]
    assert po.numsegs == 4
    v0_obj = level.vertexes[v0]
    v1_obj = level.vertexes[v1]
    v2_obj = level.vertexes[v2]
    v3_obj = level.vertexes[v3]

    # Segs MUST be sorted into exact order: 1, 2, 3, 4
    assert po.segs[0].v1.x == v0_obj.x
    assert po.segs[1].v1.x == v1_obj.x
    assert po.segs[2].v1.x == v2_obj.x
    assert po.segs[3].v1.x == v3_obj.x

    # First line args[1] must receive the mirror tag (99)
    assert po.segs[0].linedef.args[1] == 99
    assert po.mirror_tag == 99
    assert po.seq_type == 5


def test_explicit_line_order_0_error() -> None:
    """Challenge: Explicit line with order 0 must raise exact missing order error."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-950, -1000)
    sd = mb.add_sidedef(sec_dummy)

    # Order is 0!
    ld = mb.add_linedef(v0, v1, sd, special=PO_LINE_EXPLICIT, args=(11, 0, 0, 0, 0))
    mb.add_seg(v0, v1, 0, ld)
    mb.add_subsector(1, 0, bbox=(-950, -1000, -1000, -950))

    mb.add_thing(x=-975, y=-1000, type=PO_ANCHOR_TYPE, angle=11)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=11)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="Explicit line missing order number"):
            load_level(wad, "MAP01")


def test_explicit_lines_missing_order_1_error() -> None:
    """Challenge: Explicit lines starting at order 2 (missing order 1)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-950, -1000)
    v2 = mb.add_vertex(-950, -950)
    sd = mb.add_sidedef(sec_dummy)

    # Order 2 and 3 exist, but order 1 is missing!
    ld1 = mb.add_linedef(v0, v1, sd, special=PO_LINE_EXPLICIT, args=(12, 2, 0, 0, 0))
    mb.add_seg(v0, v1, 0, ld1)
    ld2 = mb.add_linedef(v1, v2, sd, special=PO_LINE_EXPLICIT, args=(12, 3, 0, 0, 0))
    mb.add_seg(v1, v2, 0, ld2)
    mb.add_subsector(2, 0, bbox=(-950, -1000, -1000, -950))

    mb.add_thing(x=-975, y=-1000, type=PO_ANCHOR_TYPE, angle=12)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=12)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="Missing explicit line 1 for poly 12"):
            load_level(wad, "MAP01")


def test_explicit_lines_skipped_intermediate_order() -> None:
    """Challenge: Explicit lines with orders 1, 2, 4 (order 3 skipped)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-950, -1000)
    v2 = mb.add_vertex(-950, -950)
    v3 = mb.add_vertex(-1000, -950)
    sd = mb.add_sidedef(sec_dummy)

    ld1 = mb.add_linedef(v0, v1, sd, special=PO_LINE_EXPLICIT, args=(13, 1, 0, 0, 0))
    mb.add_seg(v0, v1, 0, ld1)
    ld2 = mb.add_linedef(v1, v2, sd, special=PO_LINE_EXPLICIT, args=(13, 2, 0, 0, 0))
    mb.add_seg(v1, v2, 0, ld2)
    # Order 3 skipped!
    ld4 = mb.add_linedef(v2, v3, sd, special=PO_LINE_EXPLICIT, args=(13, 4, 0, 0, 0))
    mb.add_seg(v2, v3, 0, ld4)
    mb.add_subsector(3, 0, bbox=(-950, -1000, -1000, -950))

    mb.add_thing(x=-975, y=-1000, type=PO_ANCHOR_TYPE, angle=13)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=13)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="Missing explicit line 3 for poly 13"):
            load_level(wad, "MAP01")


def test_explicit_lines_exceed_maxpolysegs_error() -> None:
    """Challenge: Explicit lines exceeding PO_MAXPOLYSEGS (64 segs) must raise error."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)
    sd = mb.add_sidedef(sec_dummy)

    # Create 65 explicit segs for tag 14 (orders 1..63, with order 1 having 3 segs totaling 65)
    first_seg = len(mb.segs)
    v_prev = mb.add_vertex(-1000, -1000)

    # 3 segs under order 1
    for k in range(3):
        v_next = mb.add_vertex(-1000 + (k + 1) * 5, -1000)
        ld = mb.add_linedef(v_prev, v_next, sd, special=PO_LINE_EXPLICIT, args=(14, 1, 0, 0, 0))
        mb.add_seg(v_prev, v_next, 0, ld)
        v_prev = v_next

    # Remaining 62 segs under orders 2..63 -> total segs = 3 + 62 = 65 > 64
    for j in range(2, PO_MAXPOLYSEGS):
        v_next = mb.add_vertex(-1000 + (j + 3) * 5, -1000)
        ld = mb.add_linedef(v_prev, v_next, sd, special=PO_LINE_EXPLICIT, args=(14, j, 0, 0, 0))
        mb.add_seg(v_prev, v_next, 0, ld)
        v_prev = v_next

    mb.add_subsector(65, first_seg, bbox=(-1000, -1000, -1000, -600))
    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=14)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=14)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="SpawnPolyobj:  psIndex > PO_MAXPOLYSEGS"):
            load_level(wad, "MAP01")


# ==============================================================================
# Challenge Category 3: Coordinate Translation & Shared Vertex Deduplication
# ==============================================================================


def test_translation_extreme_coordinates() -> None:
    """Challenge: Coordinate translation across massive span without numerical drift."""
    mb = MapBuilder("MAP01")
    # Room at positive span
    mb.add_room(x=7000, y=7000, width=2048, height=2048)

    # Anchor at (-8000, -8000), Spawn at (+8000, +8000)
    # Total translation delta = -16000 map units (over 1.04 billion in 16.16 fixed point)
    mb.add_polyobject_box(
        tag=20,
        anchor_x=-8000,
        anchor_y=-8000,
        spawn_x=8000,
        spawn_y=8000,
        width=128,
        height=128,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[20]

    # Verify center of mass
    center_x = sum(v.x for v in po.prev_pts) // (4 << FRACBITS)
    center_y = sum(v.y for v in po.prev_pts) // (4 << FRACBITS)
    assert center_x == 8000
    assert center_y == 8000

    # Verify original points relative to start_spot
    orig_cx = sum(v.x for v in po.original_pts) // (4 << FRACBITS)
    orig_cy = sum(v.y for v in po.original_pts) // (4 << FRACBITS)
    assert orig_cx == 0
    assert orig_cy == 0

    # Verify that delta was applied exactly without float drift
    v0 = po.segs[0].v1
    # Before shift: -8000 - 64 = -8064. After shift: -8064 - (-16000) = 7936
    assert v0.x == 7936 << FRACBITS
    assert v0.y == 8064 << FRACBITS


def test_translation_triangle_sharing_3_vertices() -> None:
    """Challenge: Triangle sharing 3 vertices across 3 segs; verify each translated once."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    # 3 vertices
    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-950, -900)
    v2 = mb.add_vertex(-900, -1000)
    sd = mb.add_sidedef(sec_dummy)

    # 3 segs forming a closed triangle
    ld1 = mb.add_linedef(v0, v1, sd, special=PO_LINE_START, args=(21, 0, 0, 0, 0))
    ld2 = mb.add_linedef(v1, v2, sd, special=0)
    ld3 = mb.add_linedef(v2, v0, sd, special=0)
    mb.add_seg(v0, v1, 0, ld1)
    mb.add_seg(v1, v2, 0, ld2)
    mb.add_seg(v2, v0, 0, ld3)
    mb.add_subsector(3, 0, bbox=(-900, -1000, -1000, -900))

    anchor_x, anchor_y = -950, -966
    spawn_x, spawn_y = 150, 200
    mb.add_thing(x=anchor_x, y=anchor_y, type=PO_ANCHOR_TYPE, angle=21)
    mb.add_thing(x=spawn_x, y=spawn_y, type=PO_SPAWN_TYPE, angle=21)

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[21]
    assert po.numsegs == 3
    assert len(po.original_pts) == 3
    assert len(po.prev_pts) == 3

    delta_x = (anchor_x - spawn_x) << FRACBITS  # -1100 << 16
    delta_y = (anchor_y - spawn_y) << FRACBITS  # -1166 << 16

    v0_obj = level.vertexes[v0]
    v1_obj = level.vertexes[v1]
    v2_obj = level.vertexes[v2]

    # Verify each vertex was shifted by exactly -delta (x - delta_x)
    assert v0_obj.x == (-1000 << FRACBITS) - delta_x
    assert v0_obj.y == (-1000 << FRACBITS) - delta_y
    assert v1_obj.x == (-950 << FRACBITS) - delta_x
    assert v1_obj.y == (-900 << FRACBITS) - delta_y
    assert v2_obj.x == (-900 << FRACBITS) - delta_x
    assert v2_obj.y == (-1000 << FRACBITS) - delta_y

    # Verify vertex sharing
    assert po.segs[0].v2 is po.segs[1].v1
    assert po.segs[1].v2 is po.segs[2].v1
    assert po.segs[2].v2 is po.segs[0].v1


def test_translation_two_sided_linedef_bbox_single_shift() -> None:
    """Challenge: Two segs sharing one linedef must shift linedef bbox exactly once."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-950, -1000)
    sd = mb.add_sidedef(sec_dummy)

    # Explicit 2-sided line where seg 0 is forward and seg 1 is backward on same linedef
    ld = mb.add_linedef(v0, v1, sd, special=PO_LINE_EXPLICIT, args=(22, 1, 0, 0, 0))
    mb.add_seg(v0, v1, 0, ld)
    # Order 2 uses the same linedef in reverse
    mb.add_seg(v1, v0, 0, ld)
    mb.add_subsector(2, 0, bbox=(-1000, -1000, -1000, -950))

    anchor_x, anchor_y = -1000, -1000
    spawn_x, spawn_y = 100, 200
    mb.add_thing(x=anchor_x, y=anchor_y, type=PO_ANCHOR_TYPE, angle=22)
    mb.add_thing(x=spawn_x, y=spawn_y, type=PO_SPAWN_TYPE, angle=22)

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[22]
    assert po.numsegs == 2

    # Initial bbox before translation was (-1000, -1000, -1000, -950)
    delta_x = (anchor_x - spawn_x) << FRACBITS  # -1100 << 16
    delta_y = (anchor_y - spawn_y) << FRACBITS  # -1200 << 16

    ld_obj = level.lines[ld]
    top, bottom, left, right = ld_obj.bbox
    # If shifted once:
    assert left == (-1000 << FRACBITS) - delta_x
    assert right == (-950 << FRACBITS) - delta_x
    assert top == (-1000 << FRACBITS) - delta_y
    assert bottom == (-1000 << FRACBITS) - delta_y


# ==============================================================================
# Challenge Category 4: Blockmap Boundaries & Edge Conditions
# ==============================================================================


def test_polyobj_crossing_multiple_blockmap_cells() -> None:
    """Challenge: Polyobject spanning multiple blockmap cells in X and Y."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)

    # Box size 300x300 centered at (256, 256) -> spans width=300, height=300
    # In 128-unit blocks, a 300-wide box must span at least 3 blocks horizontally and vertically
    mb.add_polyobject_box(
        tag=30,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=256,
        spawn_y=256,
        width=300,
        height=300,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[30]
    bm = level.blockmap

    bx_left, bx_right = po.bbox[BOXLEFT], po.bbox[BOXRIGHT]
    by_bottom, by_top = po.bbox[BOXBOTTOM], po.bbox[BOXTOP]

    # Must span at least 3 blocks in both dimensions
    assert (bx_right - bx_left) >= 2
    assert (by_top - by_bottom) >= 2

    # Verify that every overlapping block contains po in its poly_block_map list
    overlapping_blocks_count = 0
    for by in range(by_bottom, by_top + 1):
        for bx in range(bx_left, bx_right + 1):
            if 0 <= bx < bm.width and 0 <= by < bm.height:
                overlapping_blocks_count += 1
                node = bm.poly_block_map[by * bm.width + bx]
                found = False
                while node is not None:
                    if node.polyobj is po:
                        found = True
                        break
                    node = node.next
                assert found is True, f"Polyobj 30 not found in block ({bx}, {by})"

    assert overlapping_blocks_count >= 9  # At least 3x3 = 9 blocks linked

    # Unlink and verify complete removal
    unlink_polyobj(po, bm)
    for by in range(by_bottom, by_top + 1):
        for bx in range(bx_left, bx_right + 1):
            if 0 <= bx < bm.width and 0 <= by < bm.height:
                node = bm.poly_block_map[by * bm.width + bx]
                while node is not None:
                    assert node.polyobj is not po
                    node = node.next


def test_polyobj_at_blockmap_negative_boundary() -> None:
    """Challenge: Polyobject placed partially outside blockmap bounds (negative coordinates)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)

    # In MapBuilder, dummy vertices are at -1000, so bm.org_x is -1000.
    # To place a polyobject spanning past the left boundary (x < bm.org_x),
    # place spawn at x = -1050 with width = 64 (spans -1082 to -1018).
    mb.add_polyobject_box(
        tag=31,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=-1020,
        spawn_y=256,
        width=64,
        height=64,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[31]
    bm = level.blockmap

    # BOXLEFT should be negative (outside blockmap bounds)
    assert po.bbox[BOXLEFT] < 0

    # Linking must succeed without crashing or out-of-bounds error
    link_polyobj(po, bm)

    # Block (0, y) should have the polyobj linked since it overlaps x in [bm.org_x, bm.org_x + 128]
    by = po.bbox[BOXBOTTOM]
    if 0 <= by < bm.height:
        node = bm.poly_block_map[by * bm.width + 0]
        found = False
        while node is not None:
            if node.polyobj is po:
                found = True
                break
            node = node.next
        assert found is True


def test_polyobj_multiple_in_same_subsector_small_room_error() -> None:
    """Challenge: Error when two polyobjects are placed in same subsector in small room."""
    mb = MapBuilder("MAP01")
    # Small room 128x128
    mb.add_room(x=0, y=0, width=128, height=128)

    # Poly 1
    mb.add_polyobject_box(
        tag=40,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=64,
        spawn_y=64,
        width=16,
        height=16,
    )
    # Poly 2 at identical coordinates
    mb.add_polyobject_box(
        tag=41,
        anchor_x=-500,
        anchor_y=-500,
        spawn_x=64,
        spawn_y=64,
        width=16,
        height=16,
    )

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="Multiple polyobjs in a single subsector"):
            load_level(wad, "MAP01")


def test_duplicate_spawn_spot_same_tag_error() -> None:
    """Challenge: Two spawn spots (3001) with identical tag must raise duplicate error."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)

    mb.add_polyobject_box(
        tag=50,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=100,
        spawn_y=100,
    )
    # Duplicate spawn spot thing with tag 50
    mb.add_thing(x=200, y=200, type=PO_SPAWN_TYPE, angle=50)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="Polyobj 50 already spawned"):
            load_level(wad, "MAP01")
