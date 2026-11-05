# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T4.3 (mesh_walls), T4.4 (mesh_flats), and T4.7 (chunks)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from fixtures.wadbuilder import (
    ML_BLOCKING,
    ML_DONTPEGBOTTOM,
    ML_DONTPEGTOP,
    ML_TWOSIDED,
    NF_SUBSECTOR,
    MapBuilder,
    WadBuilder,
)
from pyhexen.core.events import EventBus, SectorChanged
from pyhexen.core.fixed import FRACBITS
from pyhexen.render.chunks import ChunkManager
from pyhexen.render.mesh_flats import (
    SURFACE_FLAT,
    SURFACE_SKY,
    build_sector_flats,
)
from pyhexen.render.mesh_walls import (
    SURFACE_MASKED_WALL,
    SURFACE_OPAQUE_WALL,
    build_sector_walls,
)
from pyhexen.wad.maplumps import parse_map_lumps
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, build_level_from_raw


def _build_level(mb: MapBuilder) -> Level:
    wad_bytes = WadBuilder("PWAD").add_map(mb).to_bytes()
    with WadStack(wad_bytes) as stack:
        raw = parse_map_lumps(stack, mb.name)
        return build_level_from_raw(raw)


def test_one_sided_wall_pegging_uvs() -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=64, height=64, floor_height=0, ceiling_height=96)
    level = _build_level(mb)

    # Set texture_offset=10, row_offset=5 on line 0's front side
    line0 = level.lines[0]
    assert line0.front_side is not None
    line0.front_side.texture_offset = 10 << FRACBITS
    line0.front_side.row_offset = 5 << FRACBITS

    # 1. Default one-sided pegging: top of texture at ceiling (v0 = 5, v1 = 5 + 96 = 101)
    quads = build_sector_walls(level, 0)
    q0 = next(q for q in quads if q.line_index == 0)
    assert q0.part == "mid"
    assert q0.surface_type == SURFACE_OPAQUE_WALL
    # TL is vertex 0, BL is vertex 1, BR is vertex 2
    u0, v_top = float(q0.vertices[0, 3]), float(q0.vertices[0, 4])
    u1, v_bot = float(q0.vertices[2, 3]), float(q0.vertices[2, 4])
    assert (u0, u1) == pytest.approx((10.0, 74.0))
    assert (v_top, v_bot) == pytest.approx((5.0, 101.0))

    # 2. ML_DONTPEGBOTTOM on one-sided: bottom at floor (v1 = 5 + 128 = 133, v0 = 133 - 96 = 37)
    line0.flags |= ML_DONTPEGBOTTOM
    quads_unpegged = build_sector_walls(level, 0)
    q0_unpegged = next(q for q in quads_unpegged if q.line_index == 0)
    v_top_u = float(q0_unpegged.vertices[0, 4])
    v_bot_u = float(q0_unpegged.vertices[2, 4])
    assert (v_top_u, v_bot_u) == pytest.approx((37.0, 133.0))


def test_two_sided_upper_lower_masked_and_sky_hack() -> None:
    mb = MapBuilder("MAP01")
    # Sector 0: floor=0, ceil=128; Sector 1: floor=32, ceil=96 (creates both upper and lower walls)
    sec0, sec1, portal_ld = mb.add_two_sector_rooms(
        width1=128,
        width2=128,
        height=128,
        floor1=0,
        ceil1=128,
        floor2=32,
        ceil2=96,
    )
    level = _build_level(mb)
    portal = level.lines[portal_ld]
    assert portal.front_side is not None
    portal.front_side.row_offset = 4 << FRACBITS
    portal.front_side.mid_texture = "MIDGRATE"

    # Default two-sided pegging (neither DONTPEGTOP nor DONTPEGBOTTOM):
    # Upper wall (span = 128 - 96 = 32): v1 = 4 + 128 = 132, v0 = 132 - 32 = 100
    # Lower wall (span = 32 - 0 = 32): v0 = 4, v1 = 4 + 32 = 36
    # Masked mid wall (opening = [32, 96], span=64, tex_h=128):
    #   unpegged top at open_top + row_offset = 96 + 4 = 100, bot = 100 - 128 = -28
    #   clipped to [32, 96]: z_top=96 (v0 = 100 - 96 = 4), z_bot=32 (v1 = 100 - 32 = 68)
    quads_default = build_sector_walls(level, sec0)
    upper_q = next(q for q in quads_default if q.line_index == portal_ld and q.part == "upper")
    lower_q = next(q for q in quads_default if q.line_index == portal_ld and q.part == "lower")
    masked_q = next(q for q in quads_default if q.line_index == portal_ld and q.part == "masked")

    assert (float(upper_q.vertices[0, 4]), float(upper_q.vertices[1, 4])) == pytest.approx(
        (100.0, 132.0)
    )
    assert (float(lower_q.vertices[0, 4]), float(lower_q.vertices[1, 4])) == pytest.approx(
        (4.0, 36.0)
    )
    assert masked_q.surface_type == SURFACE_MASKED_WALL
    assert (float(masked_q.vertices[0, 1]), float(masked_q.vertices[1, 1])) == pytest.approx(
        (96.0, 32.0)
    )
    assert (float(masked_q.vertices[0, 4]), float(masked_q.vertices[1, 4])) == pytest.approx(
        (4.0, 68.0)
    )

    # Set ML_DONTPEGTOP and ML_DONTPEGBOTTOM:
    # Upper wall: v0 = 4, v1 = 4 + 32 = 36
    # Lower wall: v0 = 4 + (128 - 32) = 100, v1 = 100 + 32 = 132
    # Masked mid wall: tex_bot_z = 32 + 4 = 36, tex_top_z = 36 + 128 = 164
    #   clipped to [32, 96]: z_top=96 (v0 = 164 - 96 = 68), z_bot=36 (v1 = 164 - 36 = 128)
    portal.flags |= ML_DONTPEGTOP | ML_DONTPEGBOTTOM
    quads_pegged = build_sector_walls(level, sec0)
    upper_p = next(q for q in quads_pegged if q.line_index == portal_ld and q.part == "upper")
    lower_p = next(q for q in quads_pegged if q.line_index == portal_ld and q.part == "lower")
    masked_p = next(q for q in quads_pegged if q.line_index == portal_ld and q.part == "masked")

    assert (float(upper_p.vertices[0, 4]), float(upper_p.vertices[1, 4])) == pytest.approx(
        (4.0, 36.0)
    )
    assert (float(lower_p.vertices[0, 4]), float(lower_p.vertices[1, 4])) == pytest.approx(
        (100.0, 132.0)
    )
    assert (float(masked_p.vertices[0, 1]), float(masked_p.vertices[1, 1])) == pytest.approx(
        (96.0, 36.0)
    )
    assert (float(masked_p.vertices[0, 4]), float(masked_p.vertices[1, 4])) == pytest.approx(
        (68.0, 128.0)
    )

    # Sky hack: when both front and back sectors have ceiling_pic == "F_SKY", upper wall is omitted
    level.sectors[sec0].ceiling_pic = "F_SKY"
    level.sectors[sec1].ceiling_pic = "F_SKY"
    quads_sky = build_sector_walls(level, sec0)
    assert not any(q.line_index == portal_ld and q.part == "upper" for q in quads_sky)


def test_flat_mesh_builder_hole_and_bsp_fallback() -> None:
    # Build an outer sector 0 ([0..256]^2) with an inner island sector 1 ([64..128]^2)
    mb = MapBuilder("MAP01")
    sec_outer = mb.add_sector(floor_height=0, ceiling_height=128, ceiling_pic="F_SKY")
    sec_inner = mb.add_sector(floor_height=16, ceiling_height=112)

    # Outer clockwise loop: (0,256)->(256,256)->(256,0)->(0,0)
    vo0 = mb.add_vertex(0, 256)
    vo1 = mb.add_vertex(256, 256)
    vo2 = mb.add_vertex(256, 0)
    vo3 = mb.add_vertex(0, 0)
    for v1, v2, ang in ((vo0, vo1, 0), (vo1, vo2, 0xC000), (vo2, vo3, 0x8000), (vo3, vo0, 0x4000)):
        sd = mb.add_sidedef(sec_outer)
        ld = mb.add_linedef(v1, v2, sd, flags=ML_BLOCKING)
        mb.add_seg(v1, v2, ang, ld, side=0)
    mb.add_subsector(4, 0, bbox=(256, 0, 0, 256))

    # Inner pillar CW wrt sec_inner, with back_side in sec_outer (CCW hole in sec_outer)
    vi0 = mb.add_vertex(64, 128)
    vi1 = mb.add_vertex(128, 128)
    vi2 = mb.add_vertex(128, 64)
    vi3 = mb.add_vertex(64, 64)
    first_inner_seg = len(mb.segs)
    for v1, v2, ang in ((vi0, vi1, 0), (vi1, vi2, 0xC000), (vi2, vi3, 0x8000), (vi3, vi0, 0x4000)):
        sd_front = mb.add_sidedef(sec_inner)
        sd_back = mb.add_sidedef(sec_outer)
        ld = mb.add_linedef(v1, v2, sd_front, sd_back, flags=ML_TWOSIDED)
        mb.add_seg(v1, v2, ang, ld, side=0)
    mb.add_subsector(4, first_inner_seg, bbox=(128, 64, 64, 128))
    mb.add_node(
        128, 256, 0, -256, (256, 0, 0, 256), (128, 64, 64, 128), NF_SUBSECTOR | 0, NF_SUBSECTOR | 1
    )

    level = _build_level(mb)
    outer_mesh = build_sector_flats(level, sec_outer)
    assert outer_mesh.used_fallback is False
    assert outer_mesh.ceil_is_sky is True
    # 4 outer vertices + 4 hole vertices -> 8 triangles = 24 vertices
    assert outer_mesh.floor_vertices.shape == (24, 8)
    assert outer_mesh.ceiling_vertices.shape == (24, 8)
    assert np.all(outer_mesh.floor_vertices[:, 6] == SURFACE_FLAT)
    assert np.all(outer_mesh.ceiling_vertices[:, 6] == SURFACE_SKY)

    # Check total triangulated area equals 256^2 - 64^2 = 65536 - 4096 = 61440
    total_area = 0.0
    fv = outer_mesh.floor_vertices
    for i in range(0, len(fv), 3):
        x0, y0 = float(fv[i, 0]), -float(fv[i, 2])
        x1, y1 = float(fv[i + 1, 0]), -float(fv[i + 1, 2])
        x2, y2 = float(fv[i + 2, 0]), -float(fv[i + 2, 2])
        total_area += 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    assert total_area == pytest.approx(61440.0)

    # Break closure of sec_inner by removing one line -> triggers BSP fallback
    level.sectors[sec_inner].lines.pop()
    broken_mesh = build_sector_flats(level, sec_inner)
    assert broken_mesh.used_fallback is True
    assert len(broken_mesh.floor_vertices) >= 6


def test_chunk_manager_dirty_rebuild_and_performance() -> None:
    mb = MapBuilder("MAP01")
    sec0, sec1, _portal = mb.add_two_sector_rooms(
        width1=128,
        width2=128,
        height=128,
        floor1=0,
        ceil1=128,
        floor2=0,
        ceil2=128,
    )
    level = _build_level(mb)
    bus = EventBus()
    cm = ChunkManager(level, event_bus=bus)

    assert cm.chunks[sec0].neighbor_sectors == (sec1,)
    assert cm.chunks[sec1].neighbor_sectors == (sec0,)
    assert not cm.chunks[sec0].dirty
    assert not cm.chunks[sec1].dirty

    # Initially floor1 == floor2 == 0, so neither chunk has a lower portal wall; each has:
    # 6 floor verts + 6 ceil verts + 3 outer walls * 6 verts = 30 verts
    assert len(cm.chunks[sec0].vertices) == 30
    assert len(cm.chunks[sec1].vertices) == 30

    # Raise sector 1's floor to 32 and emit SectorChanged(sec1)
    level.sectors[sec1].floor_height = 32 << FRACBITS
    bus.emit(SectorChanged(sector_index=sec1))

    # Both sec1 and its neighbour sec0 must be marked dirty
    assert cm.chunks[sec1].dirty is True
    assert cm.chunks[sec0].dirty is True

    t0 = time.perf_counter()
    rebuilt = cm.rebuild_dirty()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert rebuilt == [sec0, sec1]
    assert cm.chunks[sec0].dirty is False
    assert cm.chunks[sec1].dirty is False
    # Sector 0 now has a lower wall on the shared portal line (+6 vertices -> 36 vertices)
    assert len(cm.chunks[sec0].vertices) == 36
    # Sector 1's floor vertices (first 6 vertices) now have GL y == 32.0
    assert np.all(cm.chunks[sec1].vertices[:6, 1] == pytest.approx(32.0))
    # Budget: < 2 ms per sector (< 4 ms for 2 sectors)
    assert (elapsed_ms / 2.0) < 2.0


@pytest.mark.wad
def test_real_map01_all_sectors_build_geometry(iwad_path: Path) -> None:
    with WadStack(iwad_path) as stack:
        raw = parse_map_lumps(stack, "MAP01")
        level = build_level_from_raw(raw)
        cm = ChunkManager(level)
        assert len(cm.chunks) == len(level.sectors)
        for chunk in cm.chunks:
            assert len(chunk.vertices) > 0
