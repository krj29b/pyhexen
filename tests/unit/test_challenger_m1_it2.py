# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Empirical challenger stress-test suite for Milestone 1 (T9.1 Polyobject Spawning).

Stress-tests boundary conditions, missing anchors/spawns, corrupt parameters,
closed vs explicit loops, subsector conflict authentic RuntimeError,
coordinate extremes, and edge-of-blockmap positioning.
"""

from __future__ import annotations

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACBITS
from pyhexen.sim.polyobj import (
    PO_ANCHOR_TYPE,
    PO_LINE_EXPLICIT,
    PO_LINE_START,
    PO_SPAWN_TYPE,
    get_polyobj,
    get_polyobj_mirror,
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
    """Helper to package MapBuilder into WadStack."""
    wb = WadBuilder("IWAD").add_map(mb)
    return WadStack(wb.to_bytes())


# ==============================================================================
# Challenge 1: Missing Anchors, Missing Spawn Spots, and Tag Mismatches
# ==============================================================================


def test_stress_spawn_spot_without_anchor_raises_runtime_error() -> None:
    """Challenge: Spawn spot (3001) exists, but anchor (3000) is missing entirely.

    Reference: po_man.c:1477-1480 -> 'PO_Init: StartSpot located without an Anchor point: <tag>'.
    """
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)
    sd = mb.add_sidedef(sec_dummy)

    # Linedefs for polyobject 100
    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-950, -1000)
    v2 = mb.add_vertex(-950, -950)
    v3 = mb.add_vertex(-1000, -950)

    ld0 = mb.add_linedef(v0, v1, sd, special=PO_LINE_START, args=(100, 0, 0, 0, 0))
    ld1 = mb.add_linedef(v1, v2, sd, special=0)
    ld2 = mb.add_linedef(v2, v3, sd, special=0)
    ld3 = mb.add_linedef(v3, v0, sd, special=0)

    mb.add_seg(v0, v1, 0, ld0)
    mb.add_seg(v1, v2, 0, ld1)
    mb.add_seg(v2, v3, 0, ld2)
    mb.add_seg(v3, v0, 0, ld3)
    mb.add_subsector(4, 0, bbox=(-950, -1000, -1000, -950))

    # Add ONLY spawn spot (3001), NO anchor (3000)
    mb.add_thing(x=128, y=128, type=PO_SPAWN_TYPE, angle=100)

    with _build_map_wad(mb) as wad:
        with pytest.raises(
            RuntimeError,
            match=r"PO_Init:\s+StartSpot located without an Anchor point:\s+100",
        ):
            load_level(wad, "MAP01")


def test_stress_anchor_without_spawn_spot_raises_runtime_error() -> None:
    """Challenge: Anchor (3000) exists, but spawn spot (3001/3002) is missing entirely.

    Reference: po_man.c:1356-1357 -> 'TranslateToStartSpot: Unable to match polyobj tag: <tag>'.
    """
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)

    # Add ONLY anchor (3000) with tag 200, NO spawn spot
    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=200)

    with _build_map_wad(mb) as wad:
        with pytest.raises(
            RuntimeError,
            match=r"TranslateToStartSpot:\s+Unable to match polyobj tag:\s+200",
        ):
            load_level(wad, "MAP01")


def test_stress_anchor_and_spawn_spot_with_no_lines_raises_runtime_error() -> None:
    """Challenge: Anchor and spawn spot exist, but no linedefs have the matching tag.

    Reference: po_man.c:1361-1363 ->
    'TranslateToStartSpot: Anchor point located without a StartSpot point: <tag>'.
    """
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)

    # Add anchor and spawn spot for tag 300, but no polyobject linedefs
    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=300)
    mb.add_thing(x=256, y=256, type=PO_SPAWN_TYPE, angle=300)

    with _build_map_wad(mb) as wad:
        with pytest.raises(
            RuntimeError,
            match=r"TranslateToStartSpot:\s+Anchor point located without a StartSpot point:\s+300",
        ):
            load_level(wad, "MAP01")


def test_stress_mismatched_tags_anchor_vs_spawn() -> None:
    """Challenge: Spawn spot has tag 401, but anchor has tag 402."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)
    sd = mb.add_sidedef(sec_dummy)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-950, -1000)
    v2 = mb.add_vertex(-950, -950)
    v3 = mb.add_vertex(-1000, -950)

    ld0 = mb.add_linedef(v0, v1, sd, special=PO_LINE_START, args=(401, 0, 0, 0, 0))
    ld1 = mb.add_linedef(v1, v2, sd, special=0)
    ld2 = mb.add_linedef(v2, v3, sd, special=0)
    ld3 = mb.add_linedef(v3, v0, sd, special=0)

    mb.add_seg(v0, v1, 0, ld0)
    mb.add_seg(v1, v2, 0, ld1)
    mb.add_seg(v2, v3, 0, ld2)
    mb.add_seg(v3, v0, 0, ld3)
    mb.add_subsector(4, 0, bbox=(-950, -1000, -1000, -950))

    # Spawn spot is 401, Anchor is 402
    mb.add_thing(x=128, y=128, type=PO_SPAWN_TYPE, angle=401)
    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=402)

    with _build_map_wad(mb) as wad:
        # Pass 2 encounters anchor 402 with no spawn spot
        with pytest.raises(
            RuntimeError,
            match=r"TranslateToStartSpot:\s+Unable to match polyobj tag:\s+402",
        ):
            load_level(wad, "MAP01")


# ==============================================================================
# Challenge 2: Corrupt Angle / Sequence Parameters Clamping
# ==============================================================================


def test_stress_corrupt_seq_type_clamping_startline() -> None:
    """Challenge: Corrupt seq_type in StartLine args[2] (< 0 or >= 10) must clamp to 0."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_room(x=512, y=0, width=512, height=512)
    mb.add_room(x=1024, y=0, width=512, height=512)

    # Box 1: seq_type = -7 (negative)
    mb.add_polyobject_box(
        tag=1,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=256,
        spawn_y=256,
        seq_type=-7,
    )
    # Box 2: seq_type = 25 (out of bounds >= SEQTYPE_NUMSEQ)
    mb.add_polyobject_box(
        tag=2,
        anchor_x=-2000,
        anchor_y=-2000,
        spawn_x=768,
        spawn_y=256,
        seq_type=25,
    )
    # Box 3: seq_type = 5 (valid)
    mb.add_polyobject_box(
        tag=3,
        anchor_x=-3000,
        anchor_y=-3000,
        spawn_x=1280,
        spawn_y=256,
        seq_type=5,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po1 = level.polyobjs[1]
    po2 = level.polyobjs[2]
    po3 = level.polyobjs[3]

    assert po1.seq_type == 0, "Negative seq_type must be clamped to 0"
    assert po2.seq_type == 0, "seq_type >= 10 must be clamped to 0"
    assert po3.seq_type == 5, "Valid seq_type must be preserved"


def test_stress_corrupt_seq_type_clamping_explicit() -> None:
    """Challenge: Corrupt seq_type in Explicit line args[3] (< 0 or >= 10) must clamp to 0."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_room(x=512, y=0, width=512, height=512)

    # Explicit box 1: seq_type = -99
    mb.add_polyobject_box(
        tag=10,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=256,
        spawn_y=256,
        method="explicit",
        seq_type=-99,
    )
    # Explicit box 2: seq_type = 99
    mb.add_polyobject_box(
        tag=11,
        anchor_x=-2000,
        anchor_y=-2000,
        spawn_x=768,
        spawn_y=256,
        method="explicit",
        seq_type=99,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    assert level.polyobjs[10].seq_type == 0
    assert level.polyobjs[11].seq_type == 0


def test_stress_mirror_tag_cross_mirroring_and_query() -> None:
    """Challenge: Mutual cross-mirroring between two polyobjects in separate subsectors."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_room(x=512, y=0, width=512, height=512)

    # Poly 20 mirrors 21
    mb.add_polyobject_box(
        tag=20,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=256,
        spawn_y=256,
        mirror_tag=21,
    )
    # Poly 21 mirrors 20
    mb.add_polyobject_box(
        tag=21,
        anchor_x=-2000,
        anchor_y=-2000,
        spawn_x=768,
        spawn_y=256,
        mirror_tag=20,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po20 = get_polyobj(level, 20)
    po21 = get_polyobj(level, 21)

    assert po20 is not None and po21 is not None
    assert po20.mirror == 21
    assert po21.mirror == 20
    assert get_polyobj_mirror(level, 20) == 21
    assert get_polyobj_mirror(level, 21) == 20
    # Query non-existent polyobj returns None / 0
    assert get_polyobj(level, 999) is None
    assert get_polyobj_mirror(level, 999) == 0


# ==============================================================================
# Challenge 3: Subsector Conflict Authentic RuntimeError
# ==============================================================================


def test_stress_polyobj_shared_subsector_authentic_runtime_error() -> None:
    """Challenge: Two polyobjects placed in the exact same convex subsector room.

    Reference: po_man.c:1410-1411 ->
    'PO_TranslateToStartSpot: Multiple polyobjs in a single subsector.\n'
    Must authentically raise RuntimeError containing this exact diagnostic.
    """
    mb = MapBuilder("MAP01")
    # Single convex room of 512x512
    mb.add_room(x=0, y=0, width=512, height=512)

    # Polyobject 50 placed in room
    mb.add_polyobject_box(
        tag=50,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=200,
        spawn_y=200,
        width=32,
        height=32,
    )
    # Polyobject 51 placed in the same room (same subsector)
    mb.add_polyobject_box(
        tag=51,
        anchor_x=-2000,
        anchor_y=-2000,
        spawn_x=300,
        spawn_y=300,
        width=32,
        height=32,
    )

    with _build_map_wad(mb) as wad:
        with pytest.raises(
            RuntimeError,
            match=(
                r"PO_TranslateToStartSpot:\s+Multiple polyobjs in a single subsector:\s+"
                r"subsector=\d+,\s+existing_poly=50,\s+new_poly=51"
            ),
        ):
            load_level(wad, "MAP01")


def test_stress_partitioned_rooms_distinct_subsectors_success() -> None:
    """Challenge: Polyobjects placed in cleanly partitioned subsectors succeed."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=512)
    mb.add_room(x=256, y=0, width=256, height=512)

    mb.add_polyobject_box(
        tag=60,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=128,
        spawn_y=256,
        width=32,
        height=32,
    )
    mb.add_polyobject_box(
        tag=61,
        anchor_x=-2000,
        anchor_y=-2000,
        spawn_x=384,
        spawn_y=256,
        width=32,
        height=32,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po60 = level.polyobjs[60]
    po61 = level.polyobjs[61]
    assert po60.subsector is not None
    assert po61.subsector is not None
    assert po60.subsector is not po61.subsector
    assert po60.subsector.poly is po60
    assert po61.subsector.poly is po61


# ==============================================================================
# Challenge 4: Geometric Invariance: StartLine Chain vs Explicit Lines
# ==============================================================================


def test_stress_startline_vs_explicit_geometric_invariance() -> None:
    """Challenge: Identical geometry spawned via StartLine vs ExplicitLine produces
    identical relative state.
    """
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_room(x=512, y=0, width=512, height=512)

    # Box A: StartLine in Room 1
    mb.add_polyobject_box(
        tag=70,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=256,
        spawn_y=256,
        width=64,
        height=64,
        method="start_line",
        mirror_tag=99,
        seq_type=2,
    )
    # Box B: Explicit in Room 2 (shifted by 512 in X)
    mb.add_polyobject_box(
        tag=71,
        anchor_x=-2000,
        anchor_y=-1000,
        spawn_x=768,
        spawn_y=256,
        width=64,
        height=64,
        method="explicit",
        mirror_tag=99,
        seq_type=2,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po_a = level.polyobjs[70]
    po_b = level.polyobjs[71]

    assert po_a.numsegs == po_b.numsegs == 4
    assert po_a.mirror_tag == po_b.mirror_tag == 99
    assert po_a.seq_type == po_b.seq_type == 2

    # Relative coordinates in original_pts must be identical across both methods
    for p_a, p_b in zip(po_a.original_pts, po_b.original_pts, strict=True):
        assert p_a.x == p_b.x
        assert p_a.y == p_b.y

    # Bounding box span (width and height in blocks) must be identical
    span_x_a = po_a.bbox[BOXRIGHT] - po_a.bbox[BOXLEFT]
    span_x_b = po_b.bbox[BOXRIGHT] - po_b.bbox[BOXLEFT]
    span_y_a = po_a.bbox[BOXTOP] - po_a.bbox[BOXBOTTOM]
    span_y_b = po_b.bbox[BOXTOP] - po_b.bbox[BOXBOTTOM]
    assert span_x_a == span_x_b
    assert span_y_a == span_y_b


# ==============================================================================
# Challenge 5: Blockmap Edge-of-Grid, Completely Off-Grid, and Negative Coordinates
# ==============================================================================


def test_stress_edge_of_blockmap_min_origin() -> None:
    """Challenge: Polyobject placed exactly at blockmap minimum corner (block 0, 0)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)

    # Place anchor at (16, 16) and spawn at (16, 16) with width=16
    # Vertices span [8, 24]. Min vertex of map is 0 (room origin).
    # Blockmap org_x = 0, org_y = 0.
    # Block 0 is [0, 128], so [8, 24] is entirely within block 0!
    mb.add_polyobject_box(
        tag=80,
        anchor_x=16,
        anchor_y=16,
        spawn_x=16,
        spawn_y=16,
        width=16,
        height=16,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[80]
    assert po.bbox[BOXLEFT] == 0
    assert po.bbox[BOXBOTTOM] == 0

    # Block (0, 0) must contain po
    node = level.blockmap.poly_block_map[0]
    found = False
    while node is not None:
        if node.polyobj is po:
            found = True
            break
        node = node.next
    assert found is True

    # Unlink cleanly
    unlink_polyobj(po, level.blockmap)
    node = level.blockmap.poly_block_map[0]
    while node is not None:
        assert node.polyobj is not po
        node = node.next


def test_stress_completely_off_blockmap_robustness() -> None:
    """Challenge: Polyobjects completely outside blockmap (negative indices and > grid max).

    Reference: po_man.c:1002-1004 & 1037: off the map polyobjs must NOT cause IndexError
    or memory corruption during link_polyobj or unlink_polyobj.
    """
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=512)
    mb.add_room(x=256, y=0, width=256, height=512)

    # Keep anchor vertices inside [0, 512] so blockmap org_x = 0, org_y = 0
    # Poly 90: spawned at (-500, -500), width=32 -> bbox is negative
    # Poly 91: spawned at (2000, 2000), width=32 -> bbox is beyond bm.width/height
    mb.add_polyobject_box(
        tag=90,
        anchor_x=100,
        anchor_y=100,
        spawn_x=-500,
        spawn_y=-500,
        width=32,
        height=32,
    )
    mb.add_polyobject_box(
        tag=91,
        anchor_x=300,
        anchor_y=100,
        spawn_x=2000,
        spawn_y=2000,
        width=32,
        height=32,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po_neg = level.polyobjs[90]
    po_pos = level.polyobjs[91]

    # Negative bbox indices
    assert po_neg.bbox[BOXRIGHT] < 0
    assert po_neg.bbox[BOXTOP] < 0

    # Positive bbox indices beyond width/height
    assert po_pos.bbox[BOXLEFT] >= level.blockmap.width
    assert po_pos.bbox[BOXBOTTOM] >= level.blockmap.height

    # Re-link and unlink must be safe no-ops without exception
    link_polyobj(po_neg, level.blockmap)
    link_polyobj(po_pos, level.blockmap)
    unlink_polyobj(po_neg, level.blockmap)
    unlink_polyobj(po_pos, level.blockmap)


def test_stress_massive_coordinate_shift_precision() -> None:
    """Challenge: Coordinate delta of 16,000 map units (approx 1.05 billion in fixed 16.16)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=7000, y=7000, width=2048, height=2048)

    # Anchor at (-8000, -8000), Spawn at (+8000, +8000)
    # Total translation delta = 16,000 map units,
    # staying within Doom 64KB blockmap offset table limit
    mb.add_polyobject_box(
        tag=95,
        anchor_x=-8000,
        anchor_y=-8000,
        spawn_x=8000,
        spawn_y=8000,
        width=128,
        height=128,
        crush=True,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[95]
    assert po.crush is True
    assert po.start_spot.x == 8000 << FRACBITS
    assert po.start_spot.y == 8000 << FRACBITS

    # Check center of translated vertices
    cx = sum(v.x for v in po.prev_pts) // (4 << FRACBITS)
    cy = sum(v.y for v in po.prev_pts) // (4 << FRACBITS)
    assert cx == 8000
    assert cy == 8000

    # Original pts must be relative to start_spot
    orig_cx = sum(v.x for v in po.original_pts) // (4 << FRACBITS)
    orig_cy = sum(v.y for v in po.original_pts) // (4 << FRACBITS)
    assert orig_cx == 0
    assert orig_cy == 0


def test_stress_duplicate_startlines_same_tag_raises_error() -> None:
    """Challenge: Two StartLine (special 1) linedefs with identical tag.

    Reference: po_man.c:1221 -> 'SpawnPolyobj: Polyobj <tag> already spawned.\n'.
    """
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)
    sd = mb.add_sidedef(sec_dummy)

    v0 = mb.add_vertex(100, 100)
    v1 = mb.add_vertex(150, 100)
    v2 = mb.add_vertex(150, 150)
    v3 = mb.add_vertex(100, 150)

    # Line 1: StartLine for tag 99
    ld0 = mb.add_linedef(v0, v1, sd, special=PO_LINE_START, args=(99, 0, 0, 0, 0))
    # Line 2: ALSO StartLine for tag 99!
    ld1 = mb.add_linedef(v1, v2, sd, special=PO_LINE_START, args=(99, 0, 0, 0, 0))
    ld2 = mb.add_linedef(v2, v3, sd, special=0)
    ld3 = mb.add_linedef(v3, v0, sd, special=0)

    mb.add_seg(v0, v1, 0, ld0)
    mb.add_seg(v1, v2, 0, ld1)
    mb.add_seg(v2, v3, 0, ld2)
    mb.add_seg(v3, v0, 0, ld3)
    mb.add_subsector(4, 0, bbox=(150, 100, 100, 150))

    mb.add_thing(x=100, y=100, type=PO_ANCHOR_TYPE, angle=99)
    mb.add_thing(x=256, y=256, type=PO_SPAWN_TYPE, angle=99)

    with _build_map_wad(mb) as wad:
        with pytest.raises(
            RuntimeError,
            match=r"SpawnPolyobj:\s+Polyobj 99 already spawned",
        ):
            load_level(wad, "MAP01")


def test_stress_open_chain_explicit_line_translates_terminal_v2() -> None:
    """Challenge: Open explicit polyobject (non-closed wall) translates unshared terminal v2."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)
    sd = mb.add_sidedef(sec_dummy)

    # 3 vertices forming an open line: v0 -> v1 -> v2 (v2 does NOT close back to v0)
    v0 = mb.add_vertex(100, 100)
    v1 = mb.add_vertex(150, 100)
    v2 = mb.add_vertex(200, 100)

    ld0 = mb.add_linedef(v0, v1, sd, special=PO_LINE_EXPLICIT, args=(88, 1, 0, 0, 0))
    ld1 = mb.add_linedef(v1, v2, sd, special=PO_LINE_EXPLICIT, args=(88, 2, 0, 0, 0))

    mb.add_seg(v0, v1, 0, ld0)
    mb.add_seg(v1, v2, 0, ld1)
    mb.add_subsector(2, 0, bbox=(100, 100, 100, 200))

    anchor_x, anchor_y = 100, 100
    spawn_x, spawn_y = 250, 250
    mb.add_thing(x=anchor_x, y=anchor_y, type=PO_ANCHOR_TYPE, angle=88)
    mb.add_thing(x=spawn_x, y=spawn_y, type=PO_SPAWN_TYPE, angle=88)

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[88]
    assert po.numsegs == 2

    # Delta = anchor - spawn = (100 - 250) = -150
    delta_x = (anchor_x - spawn_x) << FRACBITS  # -150 << 16
    delta_y = (anchor_y - spawn_y) << FRACBITS  # -150 << 16

    v0_obj = level.vertexes[v0]
    v1_obj = level.vertexes[v1]
    v2_obj = level.vertexes[v2]

    # Every vertex, including unshared terminal v2, MUST be translated by -delta
    assert v0_obj.x == (100 << FRACBITS) - delta_x
    assert v0_obj.y == (100 << FRACBITS) - delta_y
    assert v1_obj.x == (150 << FRACBITS) - delta_x
    assert v1_obj.y == (100 << FRACBITS) - delta_y
    assert v2_obj.x == (200 << FRACBITS) - delta_x
    assert v2_obj.y == (100 << FRACBITS) - delta_y


def test_stress_crush_spawn_spot_3002_sets_crush_true() -> None:
    """Challenge: Thing 3002 unconditionally sets crush=True,
    3001 unconditionally sets crush=False.
    """
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=512)
    mb.add_room(x=256, y=0, width=256, height=512)

    # Poly 77: Non-crush (3001)
    mb.add_polyobject_box(
        tag=77,
        anchor_x=50,
        anchor_y=50,
        spawn_x=128,
        spawn_y=256,
        crush=False,
    )
    # Poly 78: Crush (3002)
    mb.add_polyobject_box(
        tag=78,
        anchor_x=300,
        anchor_y=50,
        spawn_x=384,
        spawn_y=256,
        crush=True,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po_non_crush = level.polyobjs[77]
    po_crush = level.polyobjs[78]

    assert po_non_crush.crush is False
    assert po_crush.crush is True
