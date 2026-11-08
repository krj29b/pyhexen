# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Line-of-sight checks for monsters and combat radius damage.

Port of P_CheckSight, P_SightPathTraverse, P_SightBlockLinesIterator,
P_SightTraverseIntercepts, and PTR_SightTraverse (p_sight.c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACBITS, FRACUNIT, abs32, fixed_div, fixed_mul
from pyhexen.sim.maputl import line_opening
from pyhexen.sim.mobj import Mobj
from pyhexen.world.blockmap import MAPBLOCKSHIFT, MAPBLOCKSIZE, MAPBTOFRAC

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.world.level import Level, Line, Sector

__all__ = ["check_sight"]


def _point_on_divline_coords_side(x: int, y: int, lx: int, ly: int, ldx: int, ldy: int) -> int:
    """Return 0 (front) or 1 (back) for point (x, y) relative to divline (lx, ly, ldx, ldy).

    Inlined coordinate port of P_PointOnDivlineSide (p_maputl.c).
    """
    if ldx == 0:
        if x <= lx:
            return 1 if ldy > 0 else 0
        return 1 if ldy < 0 else 0

    if ldy == 0:
        if y <= ly:
            return 1 if ldx < 0 else 0
        return 1 if ldx > 0 else 0

    dx = x - lx
    dy = y - ly

    if (ldy ^ ldx ^ dx ^ dy) & 0x80000000:
        if (ldy ^ dx) & 0x80000000:
            return 1
        return 0

    left = fixed_mul(ldy >> 8, dx >> 8)
    right = fixed_mul(dy >> 8, ldx >> 8)
    return 0 if right < left else 1


def _intercept_vector_coords(
    v2x: int,
    v2y: int,
    v2dx: int,
    v2dy: int,
    v1x: int,
    v1y: int,
    v1dx: int,
    v1dy: int,
) -> int:
    """Return fractional intercept along trace (v2) crossed by line (v1).

    Inlined coordinate port of P_InterceptVector (p_maputl.c).
    """
    den = fixed_mul(v1dy >> 8, v2dx) - fixed_mul(v1dx >> 8, v2dy)
    if den == 0:
        return 0
    num = fixed_mul((v1x - v2x) >> 8, v1dy) + fixed_mul((v2y - v1y) >> 8, v1dx)
    return fixed_div(num, den)


def check_sight(
    viewer_or_sim: Any,
    target_or_viewer: Any,
    target: Any = None,
    *,
    sim: GameSim | None = None,
    level: Level | None = None,
) -> bool:
    """Port of P_CheckSight (p_sight.c).

    Return True if an unobstructed line of sight exists between viewer and target.
    Supports flexible invocation patterns:
      check_sight(sim_or_level, t1, t2)
      check_sight(t1, t2, sim=sim, level=level)
      check_sight(t1, t2)
    """
    t1: Mobj
    t2: Mobj
    resolved_level: Level | None = level

    if target is not None:
        if isinstance(viewer_or_sim, Mobj) and isinstance(target_or_viewer, Mobj):
            t1 = viewer_or_sim
            t2 = target_or_viewer
            if resolved_level is None:
                if hasattr(target, "blockmap"):
                    resolved_level = target
                elif hasattr(target, "level"):
                    resolved_level = target.level
        else:
            if resolved_level is None:
                if hasattr(viewer_or_sim, "blockmap"):
                    resolved_level = viewer_or_sim
                elif hasattr(viewer_or_sim, "level"):
                    resolved_level = viewer_or_sim.level
            t1 = target_or_viewer
            t2 = target
    else:
        t1 = viewer_or_sim
        t2 = target_or_viewer

    if resolved_level is None and sim is not None:
        resolved_level = getattr(sim, "level", None)

    if resolved_level is None:
        return True

    # 1. Fast sector rejection lookup via RejectMatrix
    sec1: Sector | None = getattr(t1, "sector", None)
    if sec1 is None and getattr(t1, "subsector", None) is not None:
        sec1 = t1.subsector.sector
    if sec1 is None and resolved_level is not None:
        ss1 = resolved_level.point_in_subsector(t1.x, t1.y)
        if ss1 is not None:
            sec1 = ss1.sector

    sec2: Sector | None = getattr(t2, "sector", None)
    if sec2 is None and getattr(t2, "subsector", None) is not None:
        sec2 = t2.subsector.sector
    if sec2 is None and resolved_level is not None:
        ss2 = resolved_level.point_in_subsector(t2.x, t2.y)
        if ss2 is not None:
            sec2 = ss2.sector

    if resolved_level.reject is not None and sec1 is not None and sec2 is not None:
        if resolved_level.reject.check_sight_reject(sec1.index, sec2.index):
            return False

    # 2. Trivial same-actor / same-point check
    if t1 is t2 or (t1.x == t2.x and t1.y == t2.y):
        return True

    # 3. Eye level calculation & slope aperture initialization
    # Looker eye is at 75% of actor height: sightzstart = z + height - (height >> 2)
    sightzstart = t1.z + t1.height - (t1.height >> 2)
    topslope = (t2.z + t2.height) - sightzstart
    bottomslope = t2.z - sightzstart

    return _sight_path_traverse(
        resolved_level,
        t1.x,
        t1.y,
        t2.x,
        t2.y,
        sightzstart,
        topslope,
        bottomslope,
    )


def _sight_path_traverse(
    level: Level,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    sightzstart: int,
    topslope: int,
    bottomslope: int,
) -> bool:
    """Port of P_SightPathTraverse, P_SightBlockLinesIterator, and
    P_SightTraverseIntercepts (p_sight.c).
    """
    bm = level.blockmap
    if bm is None:
        return True

    valid_count = bm.next_valid_count()

    # Nudge off exact block boundary if sitting on line
    if ((x1 - bm.org_x) & (MAPBLOCKSIZE - 1)) == 0:
        x1 += FRACUNIT
    if ((y1 - bm.org_y) & (MAPBLOCKSIZE - 1)) == 0:
        y1 += FRACUNIT

    trace_x = x1
    trace_y = y1
    trace_dx = x2 - x1
    trace_dy = y2 - y1
    trace_end_x = x2
    trace_end_y = y2

    x1_rel = x1 - bm.org_x
    y1_rel = y1 - bm.org_y
    xt1 = x1_rel >> MAPBLOCKSHIFT
    yt1 = y1_rel >> MAPBLOCKSHIFT

    x2_rel = x2 - bm.org_x
    y2_rel = y2 - bm.org_y
    xt2 = x2_rel >> MAPBLOCKSHIFT
    yt2 = y2_rel >> MAPBLOCKSHIFT

    if (
        xt1 < 0
        or yt1 < 0
        or xt1 >= bm.width
        or yt1 >= bm.height
        or xt2 < 0
        or yt2 < 0
        or xt2 >= bm.width
        or yt2 >= bm.height
    ):
        return False

    if xt2 > xt1:
        mapxstep = 1
        partial = FRACUNIT - ((x1_rel >> MAPBTOFRAC) & (FRACUNIT - 1))
        ystep = fixed_div(y2_rel - y1_rel, abs32(x2_rel - x1_rel))
    elif xt2 < xt1:
        mapxstep = -1
        partial = (x1_rel >> MAPBTOFRAC) & (FRACUNIT - 1)
        ystep = fixed_div(y2_rel - y1_rel, abs32(x2_rel - x1_rel))
    else:
        mapxstep = 0
        partial = FRACUNIT
        ystep = 256 * FRACUNIT
    yintercept = (y1_rel >> MAPBTOFRAC) + fixed_mul(partial, ystep)

    if yt2 > yt1:
        mapystep = 1
        partial = FRACUNIT - ((y1_rel >> MAPBTOFRAC) & (FRACUNIT - 1))
        xstep = fixed_div(x2_rel - x1_rel, abs32(y2_rel - y1_rel))
    elif yt2 < yt1:
        mapystep = -1
        partial = (y1_rel >> MAPBTOFRAC) & (FRACUNIT - 1)
        xstep = fixed_div(x2_rel - x1_rel, abs32(y2_rel - y1_rel))
    else:
        mapystep = 0
        partial = FRACUNIT
        xstep = 256 * FRACUNIT
    xintercept = (x1_rel >> MAPBTOFRAC) + fixed_mul(partial, xstep)

    crossed_lines: list[Line] = []

    def check_line(ld: Line) -> bool:
        v1 = ld.v1
        v2 = ld.v2
        v1x = v1.x
        v1y = v1.y
        v2x = v2.x
        v2y = v2.y

        s1 = _point_on_divline_coords_side(v1x, v1y, trace_x, trace_y, trace_dx, trace_dy)
        s2 = _point_on_divline_coords_side(v2x, v2y, trace_x, trace_y, trace_dx, trace_dy)
        if s1 == s2:
            return True

        ldx = ld.dx
        ldy = ld.dy
        s1 = _point_on_divline_coords_side(trace_x, trace_y, v1x, v1y, ldx, ldy)
        s2 = _point_on_divline_coords_side(trace_end_x, trace_end_y, v1x, v1y, ldx, ldy)
        if s1 == s2:
            return True

        # Solid 1-sided line blocks vision immediately (early out)
        if ld.back_sector is None or ld.front_sector is None:
            return False

        crossed_lines.append(ld)
        return True

    mapx = xt1
    mapy = yt1
    for _ in range(64):
        if not bm.block_lines_iterator(mapx, mapy, check_line, valid_count=valid_count):
            return False

        if mapx == xt2 and mapy == yt2:
            break

        if (yintercept >> FRACBITS) == mapy:
            yintercept += ystep
            mapx += mapxstep
        elif (xintercept >> FRACBITS) == mapx:
            xintercept += xstep
            mapy += mapystep

    intercepts: list[tuple[int, Line]] = []
    for ld in crossed_lines:
        frac = _intercept_vector_coords(
            trace_x,
            trace_y,
            trace_dx,
            trace_dy,
            ld.v1.x,
            ld.v1.y,
            ld.dx,
            ld.dy,
        )
        if frac <= 0:
            frac = 1
        elif frac > FRACUNIT:
            frac = FRACUNIT
        intercepts.append((frac, ld))

    intercepts.sort(key=lambda item: item[0])

    for frac, ld in intercepts:
        opening = line_opening(ld)

        if opening.openbottom >= opening.opentop:
            return False

        front = ld.front_sector
        back = ld.back_sector
        if front is not None and back is not None:
            if front.floor_height != back.floor_height:
                slope = fixed_div(opening.openbottom - sightzstart, frac)
                if slope > bottomslope:
                    bottomslope = slope

            if front.ceiling_height != back.ceiling_height:
                slope = fixed_div(opening.opentop - sightzstart, frac)
                if slope < topslope:
                    topslope = slope

        if topslope <= bottomslope:
            return False

    return True
