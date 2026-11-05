# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Wall mesh builder for one-sided and two-sided linedefs.

Implements Doom/Hexen upper, middle, lower, and two-sided masked middle wall
pegging rules (ML_DONTPEGTOP, ML_DONTPEGBOTTOM), texture offsets, and the
sky hack (suppressing upper walls between two F_SKY ceilings) per r_segs.c.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pyhexen.core.fixed import FRACUNIT
from pyhexen.wad.textures import TextureRegistry
from pyhexen.wad.wadfile import normalize_lump_name
from pyhexen.world.level import (
    ML_DONTPEGBOTTOM,
    ML_DONTPEGTOP,
    Level,
    Line,
    Sector,
    Side,
    Vertex,
)

SURFACE_OPAQUE_WALL: float = 0.0
SURFACE_FLAT: float = 1.0
SURFACE_SKY: float = 2.0
SURFACE_MASKED_WALL: float = 3.0

SKY_FLAT_NAME = "F_SKY"
DEFAULT_TEX_HEIGHT = 128.0


@dataclass(slots=True)
class WallQuad:
    """A single textured wall quad (2 triangles = 6 vertices of 8 float32 attributes)."""

    sector_index: int
    line_index: int
    part: str  # "upper", "mid", "lower", "masked", "sky_upper"
    tex_id: int
    tex_name: str
    surface_type: float
    vertices: np.ndarray  # float32[6, 8]


def _resolve_texture(
    raw_name: str,
    textures: TextureRegistry | None,
    default_height: float = DEFAULT_TEX_HEIGHT,
) -> tuple[int, str, float]:
    """Return `(tex_id, normalized_name, tex_height)` for `raw_name`."""
    name = normalize_lump_name(raw_name)
    if not name or name.startswith("-"):
        return 0, "-", 0.0
    if textures is None:
        return 1, name, default_height
    tex_id = textures.check_num_for_name(name)
    if tex_id <= 0:
        return 0, name, 0.0
    tex = textures.get(tex_id)
    return tex_id, tex.name, float(tex.height)


def _make_quad_vertices(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    z_bottom: float,
    z_top: float,
    u0: float,
    u1: float,
    v_top: float,
    v_bottom: float,
    light: float,
    surface_type: float,
    tex_id: int,
) -> np.ndarray:
    """Build a `(6, 8)` float32 array for two CCW triangles (TL, BL, BR) and (TL, BR, TR).

    Coordinate mapping: map `(x, y, z)` -> GL `(x, z, -y)`.
    """
    tid = float(tex_id)
    tl = (x1, z_top, -y1, u0, v_top, light, surface_type, tid)
    bl = (x1, z_bottom, -y1, u0, v_bottom, light, surface_type, tid)
    br = (x2, z_bottom, -y2, u1, v_bottom, light, surface_type, tid)
    tr = (x2, z_top, -y2, u1, v_top, light, surface_type, tid)
    return np.array([tl, bl, br, tl, br, tr], dtype=np.float32)


def _build_side_quads(
    line: Line,
    v_start: Vertex,
    v_end: Vertex,
    side: Side,
    front_sec: Sector,
    back_sec: Sector | None,
    textures: TextureRegistry | None,
) -> list[WallQuad]:
    """Port of R_StoreWallRange and R_RenderMaskedSegRange (r_segs.c) for one directed side."""
    x1 = v_start.x / FRACUNIT
    y1 = v_start.y / FRACUNIT
    x2 = v_end.x / FRACUNIT
    y2 = v_end.y / FRACUNIT

    wall_len = math.hypot(x2 - x1, y2 - y1)
    if wall_len <= 0.0:
        return []

    tex_offset = side.texture_offset / FRACUNIT
    row_offset = side.row_offset / FRACUNIT
    u0 = tex_offset
    u1 = tex_offset + wall_len

    front_floor = front_sec.floor_height / FRACUNIT
    front_ceil = front_sec.ceiling_height / FRACUNIT
    light = float(front_sec.light_level)
    sec_idx = front_sec.index
    quads: list[WallQuad] = []

    if back_sec is None:
        # One-sided wall
        span = front_ceil - front_floor
        if span <= 0.0:
            return quads
        tex_id, tex_name, tex_h = _resolve_texture(side.mid_texture, textures)
        if tex_id <= 0:
            return quads

        if line.flags & ML_DONTPEGBOTTOM:
            v1 = row_offset + tex_h
            v0 = v1 - span
        else:
            v0 = row_offset
            v1 = row_offset + span

        verts = _make_quad_vertices(
            x1,
            y1,
            x2,
            y2,
            front_floor,
            front_ceil,
            u0,
            u1,
            v0,
            v1,
            light,
            SURFACE_OPAQUE_WALL,
            tex_id,
        )
        quads.append(
            WallQuad(
                sector_index=sec_idx,
                line_index=line.index,
                part="mid",
                tex_id=tex_id,
                tex_name=tex_name,
                surface_type=SURFACE_OPAQUE_WALL,
                vertices=verts,
            )
        )
        return quads

    # Two-sided line
    back_floor = back_sec.floor_height / FRACUNIT
    back_ceil = back_sec.ceiling_height / FRACUNIT

    both_sky = (
        normalize_lump_name(front_sec.ceiling_pic) == SKY_FLAT_NAME
        and normalize_lump_name(back_sec.ceiling_pic) == SKY_FLAT_NAME
    )

    # 1. Upper wall (skipped when sky hack applies)
    if front_ceil > back_ceil and not both_sky:
        tex_id, tex_name, tex_h = _resolve_texture(side.top_texture, textures)
        if tex_id > 0:
            span = front_ceil - back_ceil
            if line.flags & ML_DONTPEGTOP:
                v0 = row_offset
                v1 = row_offset + span
            else:
                v1 = row_offset + tex_h
                v0 = v1 - span

            verts = _make_quad_vertices(
                x1,
                y1,
                x2,
                y2,
                back_ceil,
                front_ceil,
                u0,
                u1,
                v0,
                v1,
                light,
                SURFACE_OPAQUE_WALL,
                tex_id,
            )
            quads.append(
                WallQuad(
                    sector_index=sec_idx,
                    line_index=line.index,
                    part="upper",
                    tex_id=tex_id,
                    tex_name=tex_name,
                    surface_type=SURFACE_OPAQUE_WALL,
                    vertices=verts,
                )
            )

    # 2. Lower wall
    if back_floor > front_floor:
        tex_id, tex_name, _tex_h = _resolve_texture(side.bottom_texture, textures)
        if tex_id > 0:
            span = back_floor - front_floor
            if line.flags & ML_DONTPEGBOTTOM:
                v0 = row_offset + (front_ceil - back_floor)
                v1 = v0 + span
            else:
                v0 = row_offset
                v1 = row_offset + span

            verts = _make_quad_vertices(
                x1,
                y1,
                x2,
                y2,
                front_floor,
                back_floor,
                u0,
                u1,
                v0,
                v1,
                light,
                SURFACE_OPAQUE_WALL,
                tex_id,
            )
            quads.append(
                WallQuad(
                    sector_index=sec_idx,
                    line_index=line.index,
                    part="lower",
                    tex_id=tex_id,
                    tex_name=tex_name,
                    surface_type=SURFACE_OPAQUE_WALL,
                    vertices=verts,
                )
            )

    # 3. Two-sided masked middle wall (no vertical tiling, clipped to opening)
    mid_name = normalize_lump_name(side.mid_texture)
    if mid_name and not mid_name.startswith("-"):
        tex_id, tex_name, tex_h = _resolve_texture(mid_name, textures)
        if tex_id > 0 and tex_h > 0.0:
            open_bottom = max(front_floor, back_floor)
            open_top = min(front_ceil, back_ceil)
            if open_top > open_bottom:
                if line.flags & ML_DONTPEGBOTTOM:
                    tex_bot_z = open_bottom + row_offset
                    tex_top_z = tex_bot_z + tex_h
                else:
                    tex_top_z = open_top + row_offset
                    tex_bot_z = tex_top_z - tex_h

                z_top = min(open_top, tex_top_z)
                z_bottom = max(open_bottom, tex_bot_z)
                if z_top > z_bottom:
                    v0 = tex_top_z - z_top
                    v1 = tex_top_z - z_bottom
                    verts = _make_quad_vertices(
                        x1,
                        y1,
                        x2,
                        y2,
                        z_bottom,
                        z_top,
                        u0,
                        u1,
                        v0,
                        v1,
                        light,
                        SURFACE_MASKED_WALL,
                        tex_id,
                    )
                    quads.append(
                        WallQuad(
                            sector_index=sec_idx,
                            line_index=line.index,
                            part="masked",
                            tex_id=tex_id,
                            tex_name=tex_name,
                            surface_type=SURFACE_MASKED_WALL,
                            vertices=verts,
                        )
                    )

    return quads


def build_sector_walls(
    level: Level,
    sector_index: int,
    textures: TextureRegistry | None = None,
) -> list[WallQuad]:
    """Build all wall quads owned by `sector_index`."""
    sector = level.sectors[sector_index]
    quads: list[WallQuad] = []

    for line in sector.lines:
        if line.is_polyobj:
            continue
        if line.front_side is not None and line.front_side.sector.index == sector_index:
            back_sec = line.back_side.sector if line.back_side is not None else None
            quads.extend(
                _build_side_quads(
                    line,
                    line.v1,
                    line.v2,
                    line.front_side,
                    line.front_side.sector,
                    back_sec,
                    textures,
                )
            )
        if line.back_side is not None and line.back_side.sector.index == sector_index:
            front_sec = line.front_side.sector if line.front_side is not None else None
            quads.extend(
                _build_side_quads(
                    line,
                    line.v2,
                    line.v1,
                    line.back_side,
                    line.back_side.sector,
                    front_sec,
                    textures,
                )
            )

    return quads
