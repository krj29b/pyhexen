# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit and synthetic tests for Milestone 1 (T9.1 Polyobject Spawning)."""

from __future__ import annotations

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACBITS
from pyhexen.render.mesh_walls import build_sector_walls
from pyhexen.sim.polyobj import (
    PO_ANCHOR_TYPE,
    PO_LINE_EXPLICIT,
    PO_LINE_START,
    PO_SPAWN_TYPE,
    Polyobj,
    get_polyobj,
    get_polyobj_mirror,
    link_polyobj,
    unlink_polyobj,
)
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Line, load_level


def _build_map_wad(mb: MapBuilder) -> WadStack:
    """Helper to pack a MapBuilder into a WadStack."""
    wb = WadBuilder("IWAD").add_map(mb)
    return WadStack(wb.to_bytes())


def test_spawn_startline_chain() -> None:
    """Test standard closed-loop polyobject spawning via StartLine (special 1)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_idx, lines, ss_idx = mb.add_polyobject_box(
        tag=4,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=256,
        spawn_y=256,
        width=64,
        height=64,
        crush=False,
        method="start_line",
        mirror_tag=5,
        seq_type=3,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    assert 4 in level.polyobjs
    po: Polyobj = level.polyobjs[4]
    assert po.tag == 4
    assert po.crush is False
    assert po.seq_type == 3
    assert po.mirror_tag == 5
    assert po.mirror == 5
    assert po.numsegs == 4
    assert len(po.segs) == 4
    assert len(po.original_pts) == 4
    assert len(po.prev_pts) == 4

    # Coordinates: anchor (-1000, -1000) -> spawn (256, 256)
    # Box was centered around (-1000, -1000) with width=64, height=64
    # Translated box should be centered around (256, 256)
    center_x = sum(v.x for v in po.prev_pts) // (4 << FRACBITS)
    center_y = sum(v.y for v in po.prev_pts) // (4 << FRACBITS)
    assert center_x == 256
    assert center_y == 256

    # Original points should be centered around (0, 0) relative to start_spot
    orig_center_x = sum(v.x for v in po.original_pts) // (4 << FRACBITS)
    orig_center_y = sum(v.y for v in po.original_pts) // (4 << FRACBITS)
    assert orig_center_x == 0
    assert orig_center_y == 0

    # Subsector linking
    assert po.subsector is not None
    assert po.subsector.poly is po

    # Linedefs marked as polyobj and special cleared
    for seg in po.segs:
        assert seg.linedef.is_polyobj is True
        assert seg.linedef.special == 0

    # Blockmap linking
    box_top, box_bottom, box_left, box_right = po.bbox
    bm = level.blockmap
    found_block = False
    for by in range(box_bottom, box_top + 1):
        for bx in range(box_left, box_right + 1):
            if 0 <= bx < bm.width and 0 <= by < bm.height:
                node = bm.poly_block_map[by * bm.width + bx]
                while node is not None:
                    if node.polyobj is po:
                        found_block = True
                        break
                    node = node.next
    assert found_block is True


def test_spawn_crush_start_spot() -> None:
    """Test crush spawn spot (thing type 3002) setting po.crush = True."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_polyobject_box(
        tag=10,
        anchor_x=-800,
        anchor_y=-800,
        spawn_x=128,
        spawn_y=128,
        crush=True,
        method="start_line",
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[10]
    assert po.crush is True


def test_spawn_explicit_lines() -> None:
    """Test explicit line collection (special 5, orders 1..4)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_polyobject_box(
        tag=7,
        anchor_x=-1500,
        anchor_y=-1500,
        spawn_x=200,
        spawn_y=200,
        method="explicit",
        mirror_tag=8,
        seq_type=2,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    assert 7 in level.polyobjs
    po = level.polyobjs[7]
    assert po.tag == 7
    assert po.numsegs == 4
    assert po.mirror_tag == 8
    assert po.seq_type == 2

    # Verify that order 1 line was updated so args[1] == mirror_tag
    assert po.segs[0].linedef.args[1] == 8
    for seg in po.segs:
        assert seg.linedef.is_polyobj is True
        assert seg.linedef.special == 0


def test_error_explicit_line_missing_order_0() -> None:
    """Test error when an explicit line has order 0 (args[1] == 0)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-1000, -950)
    sd = mb.add_sidedef(sec_dummy)
    # Order is 0!
    ld = mb.add_linedef(v0, v1, sd, special=PO_LINE_EXPLICIT, args=(12, 0, 0, 0, 0))
    mb.add_seg(v0, v1, 0, ld)
    mb.add_subsector(1, 4, bbox=(-950, -1000, -1000, -1000))

    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=12)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=12)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="Explicit line missing order number"):
            load_level(wad, "MAP01")


def test_error_explicit_line_order_skipped() -> None:
    """Test error when explicit line orders are non-contiguous (e.g. 1, 3)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-1000, -950)
    v2 = mb.add_vertex(-950, -950)
    sd = mb.add_sidedef(sec_dummy)

    # Order 1
    ld1 = mb.add_linedef(v0, v1, sd, special=PO_LINE_EXPLICIT, args=(14, 1, 0, 0, 0))
    mb.add_seg(v0, v1, 0, ld1)
    # Order 3 (order 2 skipped!)
    ld3 = mb.add_linedef(v1, v2, sd, special=PO_LINE_EXPLICIT, args=(14, 3, 0, 0, 0))
    mb.add_seg(v1, v2, 0, ld3)
    mb.add_subsector(2, 4, bbox=(-950, -1000, -1000, -950))

    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=14)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=14)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="Missing explicit line 2"):
            load_level(wad, "MAP01")


def test_error_non_closed_startline() -> None:
    """Test error when StartLine seg chain fails to loop back to origin."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-1000, -950)
    v2 = mb.add_vertex(-950, -950)
    sd = mb.add_sidedef(sec_dummy)

    # Line 1 is StartLine (special 1)
    ld1 = mb.add_linedef(v0, v1, sd, special=PO_LINE_START, args=(20, 0, 0, 0, 0))
    mb.add_seg(v0, v1, 0, ld1)
    # Line 2 continues to v2, but no closing line back to v0!
    ld2 = mb.add_linedef(v1, v2, sd, special=0)
    mb.add_seg(v1, v2, 0, ld2)
    mb.add_subsector(2, 4, bbox=(-950, -1000, -1000, -950))

    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=20)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=20)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="Non-closed Polyobj located"):
            load_level(wad, "MAP01")


def test_error_spawn_spot_without_anchor() -> None:
    """Test error when a spawn spot (3001) exists without a matching anchor (3000)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-1000, -950)
    sd = mb.add_sidedef(sec_dummy)
    ld1 = mb.add_linedef(v0, v1, sd, special=PO_LINE_START, args=(30, 0, 0, 0, 0))
    ld2 = mb.add_linedef(v1, v0, sd, special=0)
    mb.add_seg(v0, v1, 0, ld1)
    mb.add_seg(v1, v0, 0, ld2)
    mb.add_subsector(2, 4, bbox=(-950, -1000, -1000, -1000))

    # Spawn spot present, but ANCHOR THING IS MISSING
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=30)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="StartSpot located without an Anchor point"):
            load_level(wad, "MAP01")


def test_error_anchor_without_spawn_spot() -> None:
    """Test error when an anchor (3000) exists without a matching spawn spot (3001/3002)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    # Anchor thing present with tag 99, but no spawn spot
    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=99)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="Unable to match polyobj tag"):
            load_level(wad, "MAP01")


def test_error_multiple_polyobjs_in_subsector() -> None:
    """Test error when two polyobjects are placed into the same subsector."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)  # Subsector 0

    # Poly 1
    mb.add_polyobject_box(
        tag=1,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=256,
        spawn_y=256,
        width=32,
        height=32,
    )
    # Poly 2 placed at identical center in same room
    mb.add_polyobject_box(
        tag=2,
        anchor_x=-500,
        anchor_y=-500,
        spawn_x=256,
        spawn_y=256,
        width=32,
        height=32,
    )

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="Multiple polyobjs in a single subsector"):
            load_level(wad, "MAP01")


def test_shared_vertex_deduplication() -> None:
    """Verify that shared vertices between segs are shifted exactly once."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_polyobject_box(
        tag=4,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=100,
        spawn_y=200,
        width=64,
        height=64,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[4]
    # Anchor = (-1000, -1000), Spawn = (100, 200)
    # delta_x = -1000 - 100 = -1100
    # delta_y = -1000 - 200 = -1200
    # v_tl was (-1032, -968) -> shifted by -delta should be (68, 232)
    v_tl = po.segs[0].v1
    assert v_tl.x == 68 << FRACBITS
    assert v_tl.y == 232 << FRACBITS

    # Ensure v2 of seg 0 is v1 of seg 1
    assert po.segs[0].v2 is po.segs[1].v1
    # Ensure seg 1.v1 is not double-shifted
    assert po.segs[1].v1.x == 132 << FRACBITS
    assert po.segs[1].v1.y == 232 << FRACBITS


def test_static_wall_exclusion() -> None:
    """Verify that build_sector_walls excludes lines with is_polyobj = True."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    dummy_sec, line_ids, _ = mb.add_polyobject_box(
        tag=4,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=256,
        spawn_y=256,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    # In the dummy holding sector, all 4 lines belong to the polyobject
    dummy_quads = build_sector_walls(level, dummy_sec)
    assert len(dummy_quads) == 0


def test_blockmap_linking_and_unlinking() -> None:
    """Verify link_polyobj and unlink_polyobj integration with Blockmap."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_polyobject_box(
        tag=4,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=256,
        spawn_y=256,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[4]
    bm = level.blockmap

    # Poly is linked on load
    visited_lines: list[Line] = []
    bx = bm.world_to_block_x(256 << FRACBITS)
    by = bm.world_to_block_y(256 << FRACBITS)
    bm.block_lines_iterator(bx, by, lambda ld: visited_lines.append(ld) or True)
    poly_lines = [ld for ld in visited_lines if ld.is_polyobj]
    assert len(poly_lines) > 0

    # Unlink polyobject
    unlink_polyobj(po, bm)
    visited_lines.clear()
    bm.next_valid_count()
    bm.block_lines_iterator(bx, by, lambda ld: visited_lines.append(ld) or True)
    poly_lines_after = [ld for ld in visited_lines if ld.is_polyobj]
    assert len(poly_lines_after) == 0

    # Re-link polyobject
    link_polyobj(po, bm)
    visited_lines.clear()
    bm.next_valid_count()
    bm.block_lines_iterator(bx, by, lambda ld: visited_lines.append(ld) or True)
    poly_lines_relinked = [ld for ld in visited_lines if ld.is_polyobj]
    assert len(poly_lines_relinked) > 0


def test_mirror_polyobject_linking() -> None:
    """Verify linking between polyobject and its mirror tag."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(x=0, y=0, width1=256, width2=256, height=256)
    # Poly 1 mirrors 2 (in left room, subsector 0)
    mb.add_polyobject_box(
        tag=1,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=128,
        spawn_y=128,
        mirror_tag=2,
    )
    # Poly 2 mirrors 1 (in right room, subsector 1)
    mb.add_polyobject_box(
        tag=2,
        anchor_x=-500,
        anchor_y=-500,
        spawn_x=384,
        spawn_y=128,
        mirror_tag=1,
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po1 = get_polyobj(level, 1)
    po2 = get_polyobj(level, 2)
    assert po1 is not None and po2 is not None
    assert po1.mirror_tag == 2
    assert po2.mirror_tag == 1
    assert get_polyobj_mirror(level, 1) == 2
    assert get_polyobj_mirror(level, 2) == 1


def test_seq_type_clamping() -> None:
    """Verify that an invalid seq_type (>=10 or <0) clamps to 0 (stone)."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_polyobject_box(
        tag=4,
        anchor_x=-1000,
        anchor_y=-1000,
        spawn_x=256,
        spawn_y=256,
        seq_type=15,  # Invalid seq_type
    )

    with _build_map_wad(mb) as wad:
        level = load_level(wad, "MAP01")

    po = level.polyobjs[4]
    assert po.seq_type == 0


def test_error_duplicate_startline() -> None:
    """Verify that two distinct StartLines with the same tag trigger an error."""
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    sec_dummy = mb.add_sector(floor_height=0, ceiling_height=128)

    v0 = mb.add_vertex(-1000, -1000)
    v1 = mb.add_vertex(-1000, -950)
    v2 = mb.add_vertex(-950, -950)
    sd = mb.add_sidedef(sec_dummy)

    # First StartLine with tag 5
    ld1 = mb.add_linedef(v0, v1, sd, special=PO_LINE_START, args=(5, 0, 0, 0, 0))
    # Second StartLine with tag 5!
    ld2 = mb.add_linedef(v1, v2, sd, special=PO_LINE_START, args=(5, 0, 0, 0, 0))
    ld3 = mb.add_linedef(v2, v0, sd, special=0)

    mb.add_seg(v0, v1, 0, ld1)
    mb.add_seg(v1, v2, 0, ld2)
    mb.add_seg(v2, v0, 0, ld3)
    mb.add_subsector(3, 4, bbox=(-950, -1000, -1000, -950))

    mb.add_thing(x=-1000, y=-1000, type=PO_ANCHOR_TYPE, angle=5)
    mb.add_thing(x=100, y=100, type=PO_SPAWN_TYPE, angle=5)

    with _build_map_wad(mb) as wad:
        with pytest.raises(RuntimeError, match="already spawned"):
            load_level(wad, "MAP01")
