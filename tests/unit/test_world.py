# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T3.1 (maplumps), T3.2 (level), and T3.3 (bsp, blockmap, reject)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.wad.maplumps import MapFormatError, parse_map_lumps
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.bsp import point_on_seg_side, point_on_side
from pyhexen.world.level import (
    BOXBOTTOM,
    BOXLEFT,
    BOXRIGHT,
    BOXTOP,
    GET_SPAC,
    ML_BLOCKING,
    ML_REPEAT_SPECIAL,
    ML_SPAC_SHIFT,
    ML_TWOSIDED,
    SPAC_USE,
    Line,
    SlopeType,
    load_level,
)
from pyhexen.world.reject import RejectMatrix


def _build_single_room_wad() -> WadStack:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=128, floor_height=-16, ceiling_height=128, light=144)
    mb.add_thing(x=64, y=64, type=1, tid=7, z=0, angle=90, special=80, args=(1, 2, 3, 4, 5))
    wb = WadBuilder("IWAD").add_map(mb)
    return WadStack(wb.to_bytes())


def _build_two_room_wad() -> WadStack:
    mb = MapBuilder("MAP02")
    mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=256,
        width2=256,
        height=256,
        floor1=0,
        ceil1=128,
        floor2=24,
        ceil2=160,
        portal_flags=ML_TWOSIDED | ML_REPEAT_SPECIAL | (SPAC_USE << ML_SPAC_SHIFT),
        portal_special=12,
        portal_args=(1, 16, 0, 0, 0),
    )
    wb = WadBuilder("IWAD").add_map(mb)
    return WadStack(wb.to_bytes())


def test_parse_map_lumps_roundtrip() -> None:
    wad = _build_single_room_wad()
    raw = parse_map_lumps(wad, "MAP01")
    assert raw.name == "MAP01"
    assert len(raw.things) == 1
    assert int(raw.things[0]["tid"]) == 7
    assert tuple(int(x) for x in raw.things[0]["args"]) == (1, 2, 3, 4, 5)
    assert len(raw.vertexes) == 4
    assert len(raw.linedefs) == 4
    assert len(raw.sidedefs) == 4
    assert len(raw.segs) == 4
    assert len(raw.ssectors) == 1
    assert len(raw.nodes) == 1
    assert len(raw.sectors) == 1


def test_parse_map_lumps_rejects_doom_format() -> None:
    mb = MapBuilder("MAP01")
    mb.add_room()
    lumps = mb.build_lumps()
    # Strip the trailing BEHAVIOR lump to simulate a Doom-format map
    doom_lumps = [pair for pair in lumps if pair[0] != "BEHAVIOR"]
    wb = WadBuilder("PWAD")
    for name, data in doom_lumps:
        wb.add_lump(name, data)
    wad = WadStack(wb.to_bytes())

    with pytest.raises(MapFormatError, match="BEHAVIOR"):
        parse_map_lumps(wad, "MAP01")


def test_level_construction_invariants_single_room() -> None:
    wad = _build_single_room_wad()
    level = load_level(wad, "MAP01")

    assert level.name == "MAP01"
    assert len(level.sectors) == 1
    sec = level.sectors[0]
    assert sec.floor_height == -16 * FRACUNIT
    assert sec.ceiling_height == 128 * FRACUNIT
    assert sec.light_level == 144
    assert len(sec.lines) == 4
    # Center of [0..256] x [0..128] is (128, 64) in fixed-point
    assert sec.sound_org_x == 128 * FRACUNIT
    assert sec.sound_org_y == 64 * FRACUNIT

    # Check linedef slopes and bboxes
    slopes = {line.slopetype for line in level.lines}
    assert slopes == {SlopeType.ST_HORIZONTAL, SlopeType.ST_VERTICAL}

    for line in level.lines:
        assert line.front_sector is sec
        assert line.back_sector is None
        assert line.bbox[BOXLEFT] == min(line.v1.x, line.v2.x)
        assert line.bbox[BOXRIGHT] == max(line.v1.x, line.v2.x)
        assert line.bbox[BOXBOTTOM] == min(line.v1.y, line.v2.y)
        assert line.bbox[BOXTOP] == max(line.v1.y, line.v2.y)

    assert len(level.things) == 1
    assert level.things[0].tid == 7
    assert level.things[0].args == (1, 2, 3, 4, 5)


def test_level_construction_two_sectors_and_diagonal_slopes() -> None:
    mb = MapBuilder("MAP03")
    mb.add_two_sector_rooms()
    # Add two diagonal lines inside sector 0 to exercise ST_POSITIVE and ST_NEGATIVE
    v_a = mb.add_vertex(16, 16)
    v_b = mb.add_vertex(48, 48)
    v_c = mb.add_vertex(16, 48)
    v_d = mb.add_vertex(48, 16)
    sd = mb.add_sidedef(0)
    ld_pos = mb.add_linedef(v_a, v_b, sd, flags=ML_BLOCKING)
    ld_neg = mb.add_linedef(v_c, v_d, sd, flags=ML_BLOCKING)

    wad = WadStack(WadBuilder("IWAD").add_map(mb).to_bytes())
    level = load_level(wad, "MAP03")

    assert len(level.sectors) == 2
    assert level.lines[ld_pos].slopetype == SlopeType.ST_POSITIVE
    assert level.lines[ld_neg].slopetype == SlopeType.ST_NEGATIVE

    # Find the two-sided portal line
    two_sided = [li for li in level.lines if li.flags & ML_TWOSIDED]
    assert len(two_sided) == 1
    portal = two_sided[0]
    assert portal.front_sector is level.sectors[0]
    assert portal.back_sector is level.sectors[1]
    assert portal in level.sectors[0].lines
    assert portal in level.sectors[1].lines


def test_spac_extraction() -> None:
    wad = _build_two_room_wad()
    level = load_level(wad, "MAP02")
    portal = [li for li in level.lines if li.flags & ML_TWOSIDED][0]
    assert GET_SPAC(portal.flags) == SPAC_USE
    assert bool(portal.flags & ML_REPEAT_SPECIAL)


def test_bsp_point_in_subsector_and_seg_side() -> None:
    wad = _build_two_room_wad()
    level = load_level(wad, "MAP02")

    # Left room is x in [0..256], right room is x in [256..512]
    ss_left = level.point_in_subsector(64 * FRACUNIT, 128 * FRACUNIT)
    ss_right = level.point_in_subsector(384 * FRACUNIT, 128 * FRACUNIT)

    assert ss_left.sector is level.sectors[0]
    assert ss_right.sector is level.sectors[1]
    assert ss_left.index != ss_right.index

    # Check point_on_seg_side on top horizontal seg (v1=(0,256) -> v2=(256,256), facing East)
    top_seg = level.segs[0]
    # Interior point (128, 128) is on the right (front = 0) of the clockwise top edge
    assert point_on_seg_side(128 * FRACUNIT, 128 * FRACUNIT, top_seg) == 0
    # Exterior point (128, 300) is on the left (back = 1)
    assert point_on_seg_side(128 * FRACUNIT, 300 * FRACUNIT, top_seg) == 1

    # Also test diagonal BSP node partition in point_on_side
    diag_node = level.nodes[0]
    assert point_on_side(64 * FRACUNIT, 128 * FRACUNIT, diag_node) == 0
    assert point_on_side(384 * FRACUNIT, 128 * FRACUNIT, diag_node) == 1


def test_blockmap_lines_and_things_iterators() -> None:
    wad = _build_single_room_wad()
    level = load_level(wad, "MAP01")
    bmap = level.blockmap

    assert bmap.width >= 2
    assert bmap.height >= 1

    visited: list[Line] = []
    vc = bmap.next_valid_count()

    # Iterate over all blocks with the same valid_count: each line should be visited at most once
    for by in range(bmap.height):
        for bx in range(bmap.width):
            bmap.block_lines_iterator(
                bx, by, lambda ld: (visited.append(ld), True)[1], valid_count=vc
            )

    assert len(visited) == len(level.lines)
    assert {ld.index for ld in visited} == {0, 1, 2, 3}

    # Test early-out when callback returns False
    vc2 = bmap.next_valid_count()
    early_visited: list[Line] = []

    def stop_first(ld: Line) -> bool:
        early_visited.append(ld)
        return False

    res = bmap.block_lines_iterator(0, 0, stop_first, valid_count=vc2)
    assert res is False
    assert len(early_visited) == 1

    # Out-of-bounds block returns True without calling callback
    assert bmap.block_lines_iterator(-1, 0, stop_first) is True
    assert bmap.block_lines_iterator(0, 999, stop_first) is True

    # Test block_things_iterator
    @dataclass
    class DummyMobj:
        name: str
        bnext: DummyMobj | None = None

    m2 = DummyMobj("M2", bnext=None)
    m1 = DummyMobj("M1", bnext=m2)
    bmap.block_links[0] = m1

    found_things: list[str] = []
    ok = bmap.block_things_iterator(0, 0, lambda m: (found_things.append(m.name), True)[1])
    assert ok is True
    assert found_things == ["M1", "M2"]


def test_reject_matrix() -> None:
    # 3 sectors -> 9 bits -> 2 bytes
    # Bit for (sec1=1, sec2=2) is pnum = 1*3 + 2 = 5 -> byte 0, bit (1 << 5) = 0x20
    rm = RejectMatrix(bytes([0x20, 0x00]), num_sectors=3)
    assert rm.check_sight_reject(1, 2) is True
    assert rm.check_sight_reject(2, 1) is False
    assert rm.check_sight_reject(0, 0) is False
    assert rm.check_sight_reject(-1, 0) is False
    assert rm.check_sight_reject(0, 99) is False
