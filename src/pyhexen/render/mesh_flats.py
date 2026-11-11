# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sector floor and ceiling flat mesh builder.

Traces 2D boundary polygon loops (including inner hole rings) and triangulates
them via `mapbox_earcut`, with a Sutherland-Hodgman BSP/seg half-plane clipping
fallback for unclosed or degenerate sectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mapbox_earcut as earcut
import numpy as np

from pyhexen.core.fixed import FRACUNIT
from pyhexen.render.mesh_walls import (
    SKY_FLAT_NAME,
    SURFACE_FLAT,
    SURFACE_SKY,
)
from pyhexen.wad.flats import FlatRegistry
from pyhexen.wad.wadfile import normalize_lump_name
from pyhexen.world.bsp import NF_SUBSECTOR
from pyhexen.world.level import Level, Sector, Subsector

Point2D = tuple[float, float]


@dataclass(slots=True)
class SectorFlatMesh:
    """Triangulated floor and ceiling vertex buffers for a single sector."""

    sector_index: int
    floor_tex_id: int
    ceil_tex_id: int
    floor_is_sky: bool
    ceil_is_sky: bool
    used_fallback: bool
    floor_vertices: np.ndarray  # float32[N, 8]
    ceiling_vertices: np.ndarray  # float32[N, 8]


def _resolve_flat(raw_name: str, flats: FlatRegistry | None) -> tuple[int, bool]:
    """Return `(flat_index, is_sky)` for `raw_name`."""
    norm = normalize_lump_name(raw_name)
    is_sky = norm == SKY_FLAT_NAME
    if flats is None:
        return (0 if is_sky else 1), is_sky
    idx = flats.check_num_for_name(norm)
    return max(0, idx), is_sky


def _signed_ring_area(ring: list[Point2D]) -> float:
    """Return 2D signed area (negative = clockwise, positive = counter-clockwise)."""
    area2 = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return 0.5 * area2


def _point_in_ring(pt: Point2D, ring: list[Point2D]) -> bool:
    """Ray-casting point-in-polygon test."""
    px, py = pt
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _trace_sector_loops(sector: Sector) -> tuple[list[list[Point2D]], bool]:
    """Extract closed directed boundary rings for `sector`.

    Returns `(rings, cleanly_closed)`.
    """
    edges: list[tuple[Point2D, Point2D]] = []
    for line in sector.lines:
        if line.is_polyobj:
            # Polyobject lines are moved to their spawn spot by PO_Init; they do
            # not bound the sector they were drawn in.
            continue
        f_idx = line.front_sector.index if line.front_sector is not None else -1
        b_idx = line.back_sector.index if line.back_sector is not None else -1
        if f_idx == sector.index and b_idx == sector.index:
            # Self-referencing internal line does not bound the 2D polygon
            continue
        p1: Point2D = (line.v1.x / FRACUNIT, line.v1.y / FRACUNIT)
        p2: Point2D = (line.v2.x / FRACUNIT, line.v2.y / FRACUNIT)
        if p1 == p2:
            continue
        if f_idx == sector.index:
            edges.append((p1, p2))
        elif b_idx == sector.index:
            edges.append((p2, p1))

    if not edges:
        return [], False

    outgoing: dict[Point2D, list[Point2D]] = {}
    in_degree: dict[Point2D, int] = {}
    for u, v in edges:
        outgoing.setdefault(u, []).append(v)
        in_degree[v] = in_degree.get(v, 0) + 1
        in_degree.setdefault(u, in_degree.get(u, 0))

    cleanly_closed = all(len(outgoing.get(pt, ())) == deg for pt, deg in in_degree.items())
    if not cleanly_closed:
        return [], False

    rings: list[list[Point2D]] = []
    while outgoing:
        start_u = next(iter(outgoing))
        u_list = outgoing[start_u]
        curr = u_list.pop()
        if not u_list:
            del outgoing[start_u]

        ring: list[Point2D] = [start_u]
        prev = start_u

        while curr != start_u:
            ring.append(curr)
            candidates = outgoing.get(curr)
            if not candidates:
                return [], False

            if len(candidates) == 1:
                nxt = candidates.pop()
            else:
                # Choose the turn that keeps the interior on the right
                in_dx = curr[0] - prev[0]
                in_dy = curr[1] - prev[1]
                in_angle = math.atan2(in_dy, in_dx)

                def _turn_score(
                    target: Point2D,
                    origin: Point2D = curr,
                    base_ang: float = in_angle,
                ) -> float:
                    out_ang = math.atan2(target[1] - origin[1], target[0] - origin[0])
                    diff = (out_ang - base_ang + math.pi) % (2.0 * math.pi) - math.pi
                    return diff

                best_idx = min(range(len(candidates)), key=lambda i: _turn_score(candidates[i]))
                nxt = candidates.pop(best_idx)

            if not candidates:
                del outgoing[curr]
            prev = curr
            curr = nxt

        if len(ring) >= 3 and abs(_signed_ring_area(ring)) > 1e-5:
            rings.append(ring)

    return rings, bool(rings)


def _rings_well_formed(rings: list[list[Point2D]]) -> bool:
    """Return True if every ring that would be triangulated as an outer boundary is clockwise.

    Rings are traced with the sector's interior on the right, so a real outer
    boundary is clockwise and a hole is counter-clockwise. A counter-clockwise ring
    at even depth means the sector's outer boundary is not made of its own
    linedefs (e.g. MAP01 sector 198, whose own lines only ring three pillars), and
    triangulating it would fill the pillars instead of the floor.
    """
    for ring in rings:
        sample = _ring_sample_point(ring)
        depth = sum(1 for other in rings if other is not ring and _point_in_ring(sample, other))
        if depth % 2 == 0 and _signed_ring_area(ring) > 0:
            return False
    return True


def _ring_sample_point(ring: list[Point2D]) -> Point2D:
    """Return a point just inside `ring`, for containment tests against other rings.

    A ring vertex is a poor sample: rings of one sector often share vertices, and
    a shared vertex gives an arbitrary inside/outside answer (MAP01 sector 63 had
    a hole classified as an outer ring this way). Use the midpoint of the longest
    edge, nudged a little toward the ring's interior.
    """
    n = len(ring)
    best = max(
        range(n),
        key=lambda i: (
            (ring[(i + 1) % n][0] - ring[i][0]) ** 2 + (ring[(i + 1) % n][1] - ring[i][1]) ** 2
        ),
    )
    (x1, y1), (x2, y2) = ring[best], ring[(best + 1) % n]
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    # Interior is to the left of each edge for a counter-clockwise ring, right for clockwise.
    sign = 1.0 if _signed_ring_area(ring) > 0 else -1.0
    nudge = min(0.5, length * 0.01)
    return (
        (x1 + x2) * 0.5 - sign * (y2 - y1) / length * nudge,
        (y1 + y2) * 0.5 + sign * (x2 - x1) / length * nudge,
    )


def _triangulate_rings(rings: list[list[Point2D]]) -> list[tuple[Point2D, Point2D, Point2D]]:
    """Classify outer vs hole rings by containment depth and triangulate via earcut."""
    if not rings:
        return []

    # Sort rings from largest area to smallest area
    sorted_rings = sorted(rings, key=lambda r: abs(_signed_ring_area(r)), reverse=True)
    n = len(sorted_rings)
    parent: list[int | None] = [None] * n
    depth: list[int] = [0] * n

    for i in range(n):
        sample_pt = _ring_sample_point(sorted_rings[i])
        for j in range(i - 1, -1, -1):
            if _point_in_ring(sample_pt, sorted_rings[j]):
                if parent[i] is None:
                    parent[i] = j
                depth[i] += 1

    # Group each outer ring (even depth) with its direct hole rings (odd depth)
    groups: dict[int, list[list[Point2D]]] = {}
    for i in range(n):
        if depth[i] % 2 == 0:
            groups[i] = [sorted_rings[i]]
    for i in range(n):
        if depth[i] % 2 == 1 and parent[i] is not None and parent[i] in groups:
            groups[parent[i]].append(sorted_rings[i])

    triangles: list[tuple[Point2D, Point2D, Point2D]] = []
    for ring_group in groups.values():
        flat_coords: list[tuple[float, float]] = []
        ring_ends: list[int] = []
        for r in ring_group:
            flat_coords.extend(r)
            ring_ends.append(len(flat_coords))

        verts_arr = np.asarray(flat_coords, dtype=np.float32)
        ends_arr = np.asarray(ring_ends, dtype=np.uint32)
        idx_arr = earcut.triangulate_float32(verts_arr, ends_arr)

        for k in range(0, len(idx_arr), 3):
            p0 = flat_coords[int(idx_arr[k])]
            p1 = flat_coords[int(idx_arr[k + 1])]
            p2 = flat_coords[int(idx_arr[k + 2])]
            triangles.append((p0, p1, p2))

    return triangles


def _clip_polygon_halfplane(
    poly: list[Point2D],
    lx: float,
    ly: float,
    ldx: float,
    ldy: float,
) -> list[Point2D]:
    """Sutherland-Hodgman clip keeping points on the right side of `(lx, ly) -> (lx+ldx, ly+ldy)`.

    Right side condition (matching R_PointOnSide == 0):
    `(x - lx) * ldy - (y - ly) * ldx >= -1e-5`
    """
    if not poly or (ldx == 0.0 and ldy == 0.0):
        return poly

    def _eval(pt: Point2D) -> float:
        return (pt[0] - lx) * ldy - (pt[1] - ly) * ldx

    out: list[Point2D] = []
    n = len(poly)
    for i in range(n):
        curr = poly[i]
        prev = poly[i - 1]
        c_val = _eval(curr)
        p_val = _eval(prev)
        c_in = c_val >= -1e-5
        p_in = p_val >= -1e-5

        if c_in:
            if not p_in:
                denom = p_val - c_val
                t = p_val / denom if abs(denom) > 1e-12 else 0.0
                ix = prev[0] + t * (curr[0] - prev[0])
                iy = prev[1] + t * (curr[1] - prev[1])
                out.append((ix, iy))
            out.append(curr)
        elif p_in:
            denom = p_val - c_val
            t = p_val / denom if abs(denom) > 1e-12 else 0.0
            ix = prev[0] + t * (curr[0] - prev[0])
            iy = prev[1] + t * (curr[1] - prev[1])
            out.append((ix, iy))

    return out


def _find_bsp_path_to_subsector(
    level: Level,
    target_ss: int,
) -> list[tuple[int, int]]:
    """Return list of `(node_index, child_side)` from root node to `NF_SUBSECTOR | target_ss`."""
    if not level.nodes:
        return []
    target_child = NF_SUBSECTOR | target_ss

    def _dfs(node_idx: int) -> list[tuple[int, int]] | None:
        node = level.nodes[node_idx]
        for side in (0, 1):
            c = node.children[side]
            if c == target_child:
                return [(node_idx, side)]
            if not (c & NF_SUBSECTOR) and 0 <= c < len(level.nodes):
                sub_path = _dfs(c)
                if sub_path is not None:
                    return [(node_idx, side), *sub_path]
        return None

    return _dfs(len(level.nodes) - 1) or []


def _bsp_fallback_triangles(
    level: Level,
    sector: Sector,
) -> list[tuple[Point2D, Point2D, Point2D]]:
    """Build convex polygons per subsector of `sector` by clipping with BSP and seg half-planes.

    This is how vanilla sees a sector: as the union of its subsectors. It handles
    sectors whose boundary cannot be traced from linedefs (open or self-overlapping
    sectors, and mapping tricks where a sector owns only one or two linedefs).
    """
    sector_subsectors: list[Subsector] = [
        ss for ss in level.subsectors if ss.sector.index == sector.index
    ]

    # Start every subsector from the whole map's bounding box: a subsector's
    # region is bounded by BSP partition lines, which can reach far beyond the
    # sector's own linedefs.
    xs: list[float] = [v.x / FRACUNIT for v in level.vertexes]
    ys: list[float] = [v.y / FRACUNIT for v in level.vertexes]
    for line in sector.lines:
        xs.extend((line.v1.x / FRACUNIT, line.v2.x / FRACUNIT))
        ys.extend((line.v1.y / FRACUNIT, line.v2.y / FRACUNIT))

    if not xs:
        return []

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if max_x - min_x < 1.0:
        min_x -= 32.0
        max_x += 32.0
    if max_y - min_y < 1.0:
        min_y -= 32.0
        max_y += 32.0

    # Clockwise initial bounding box
    base_box: list[Point2D] = [
        (min_x, max_y),
        (max_x, max_y),
        (max_x, min_y),
        (min_x, min_y),
    ]

    triangles: list[tuple[Point2D, Point2D, Point2D]] = []
    if not sector_subsectors:
        # Even if no subsectors reference this sector, triangulate the sector's bounding box
        return [(base_box[0], base_box[1], base_box[2]), (base_box[0], base_box[2], base_box[3])]

    for ss in sector_subsectors:
        poly = list(base_box)
        bsp_path = _find_bsp_path_to_subsector(level, ss.index)
        for node_idx, side in bsp_path:
            node = level.nodes[node_idx]
            nx = node.x / FRACUNIT
            ny = node.y / FRACUNIT
            ndx = node.dx / FRACUNIT
            ndy = node.dy / FRACUNIT
            if side == 0:
                poly = _clip_polygon_halfplane(poly, nx, ny, ndx, ndy)
            else:
                poly = _clip_polygon_halfplane(poly, nx, ny, -ndx, -ndy)

        for seg in level.segs[ss.first_line : ss.first_line + ss.num_lines]:
            if seg.linedef is not None and seg.linedef.is_polyobj:
                # Polyobject segs have been moved to the spawn spot (PO_Init) and
                # no longer lie on this subsector's boundary.
                continue
            sx1 = seg.v1.x / FRACUNIT
            sy1 = seg.v1.y / FRACUNIT
            sdx = (seg.v2.x - seg.v1.x) / FRACUNIT
            sdy = (seg.v2.y - seg.v1.y) / FRACUNIT
            poly = _clip_polygon_halfplane(poly, sx1, sy1, sdx, sdy)

        # A subsector that clips away to nothing has no area; never fall back to
        # the whole box, which would cover unrelated parts of the map.
        if len(poly) < 3 or abs(_signed_ring_area(poly)) < 1e-3:
            continue

        p0 = poly[0]
        for i in range(1, len(poly) - 1):
            triangles.append((p0, poly[i], poly[i + 1]))

    return triangles


def _emit_flat_vertices(
    triangles: list[tuple[Point2D, Point2D, Point2D]],
    z_height: float,
    light: float,
    surface_type: float,
    tex_id: int,
    *,
    is_ceiling: bool,
) -> np.ndarray:
    """Build `(3 * num_tris, 8)` float32 vertex array with world-aligned UVs `(x, -y)`."""
    if not triangles:
        return np.zeros((0, 8), dtype=np.float32)

    tid = float(tex_id)
    rows: list[tuple[float, float, float, float, float, float, float, float]] = []
    for p0, p1, p2 in triangles:
        # Ensure consistent winding for floor (upward +Y normal) vs ceiling (downward -Y normal)
        pts = (p0, p2, p1) if is_ceiling else (p0, p1, p2)
        for x, y in pts:
            rows.append((x, z_height, -y, x, -y, light, surface_type, tid))

    return np.asarray(rows, dtype=np.float32)


def build_sector_flats_from_triangles(
    sector: Sector,
    triangles: list[tuple[Point2D, Point2D, Point2D]],
    used_fallback: bool,
    flats: FlatRegistry | None = None,
) -> SectorFlatMesh:
    """Build floor and ceiling vertex arrays from pre-triangulated 2D `(x, y)` triangles."""
    floor_tex_id, floor_is_sky = _resolve_flat(sector.floor_pic, flats)
    ceil_tex_id, ceil_is_sky = _resolve_flat(sector.ceiling_pic, flats)

    floor_z = sector.floor_height / FRACUNIT
    ceil_z = sector.ceiling_height / FRACUNIT
    light = float(sector.light_level)

    floor_surf = SURFACE_SKY if floor_is_sky else SURFACE_FLAT
    ceil_surf = SURFACE_SKY if ceil_is_sky else SURFACE_FLAT

    floor_verts = _emit_flat_vertices(
        triangles, floor_z, light, floor_surf, floor_tex_id, is_ceiling=False
    )
    ceil_verts = _emit_flat_vertices(
        triangles, ceil_z, light, ceil_surf, ceil_tex_id, is_ceiling=True
    )

    return SectorFlatMesh(
        sector_index=sector.index,
        floor_tex_id=floor_tex_id,
        ceil_tex_id=ceil_tex_id,
        floor_is_sky=floor_is_sky,
        ceil_is_sky=ceil_is_sky,
        used_fallback=used_fallback,
        floor_vertices=floor_verts,
        ceiling_vertices=ceil_verts,
    )


def triangulate_sector(
    level: Level,
    sector_index: int,
) -> tuple[list[tuple[Point2D, Point2D, Point2D]], bool]:
    """Return `(triangles, used_fallback)` for `sector_index`.

    Every triangle is counter-clockwise in map coordinates, which is the winding
    `_emit_flat_vertices` and back-face culling expect.
    """
    sector = level.sectors[sector_index]
    rings, cleanly_closed = _trace_sector_loops(sector)
    if cleanly_closed and rings and _rings_well_formed(rings):
        triangles = _triangulate_rings(rings)
        if triangles:
            return _counter_clockwise(triangles), False

    triangles = _bsp_fallback_triangles(level, sector)
    return _counter_clockwise(triangles), True


def _counter_clockwise(
    triangles: list[tuple[Point2D, Point2D, Point2D]],
) -> list[tuple[Point2D, Point2D, Point2D]]:
    """Return `triangles` with any clockwise triangle flipped to counter-clockwise.

    The BSP fallback clips a clockwise box, so its triangles came out clockwise
    and were dropped by back-face culling: those floors and ceilings never drew.
    """
    out: list[tuple[Point2D, Point2D, Point2D]] = []
    for p0, p1, p2 in triangles:
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        out.append((p0, p2, p1) if cross < 0 else (p0, p1, p2))
    return out


def build_sector_flats(
    level: Level,
    sector_index: int,
    flats: FlatRegistry | None = None,
) -> SectorFlatMesh:
    """Triangulate `sector_index` and build its floor and ceiling vertex buffers."""
    sector = level.sectors[sector_index]
    triangles, used_fallback = triangulate_sector(level, sector_index)
    return build_sector_flats_from_triangles(sector, triangles, used_fallback, flats=flats)
