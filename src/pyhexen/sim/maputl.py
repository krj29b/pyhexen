# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Map geometry, line opening, intercept vector, and path traversal utilities.

Port of P_PointOnLineSide, P_BoxOnLineSide, P_PointOnDivlineSide, P_MakeDivline,
P_InterceptVector, P_LineOpening, P_SetThingPosition, P_UnsetThingPosition,
and P_PathTraverse (p_maputl.c).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyhexen.core.fixed import FRACBITS, FRACUNIT, MAXINT, abs32, fixed_div, fixed_mul
from pyhexen.sim.mobj import Mobj, set_thing_position, unset_thing_position
from pyhexen.world.blockmap import MAPBLOCKSHIFT, MAPBLOCKSIZE
from pyhexen.world.level import (
    BOXBOTTOM,
    BOXLEFT,
    BOXRIGHT,
    BOXTOP,
    Level,
    Line,
    SlopeType,
)

if TYPE_CHECKING:
    pass

PT_ADDLINES: int = 1
PT_ADDTHINGS: int = 2
PT_EARLYOUT: int = 4

MAPBTOFRAC: int = MAPBLOCKSHIFT - FRACBITS

__all__ = [
    "DivLine",
    "Intercept",
    "LineOpening",
    "PT_ADDLINES",
    "PT_ADDTHINGS",
    "PT_EARLYOUT",
    "box_on_line_side",
    "intercept_vector",
    "line_opening",
    "make_divline",
    "path_traverse",
    "point_on_divline_side",
    "point_on_line_side",
    "set_thing_position",
    "unset_thing_position",
]


@dataclass(slots=True)
class DivLine:
    """Partition/trace line `(x, y, dx, dy)` in 16.16 fixed point.

    Port of divline_t (p_local.h).
    """

    x: int
    y: int
    dx: int
    dy: int


@dataclass(slots=True)
class LineOpening:
    """Vertical opening gap across a two-sided linedef.

    Port of opentop, openbottom, lowfloor, openrange globals set by P_LineOpening (p_maputl.c).
    """

    opentop: int = 0
    openbottom: int = 0
    lowfloor: int = 0
    openrange: int = 0


@dataclass(slots=True)
class Intercept:
    """Ray/trace intersection along a DivLine trace.

    Port of intercept_t (p_local.h).
    """

    frac: int
    is_aline: bool
    line: Line | None = None
    thing: Mobj | None = None


def point_on_line_side(x: int, y: int, line: Line) -> int:
    """Return 0 (front/right) or 1 (back/left) for point `(x, y)` relative to `line`.

    Port of P_PointOnLineSide (p_maputl.c).
    """
    if line.dx == 0:
        if x <= line.v1.x:
            return 1 if line.dy > 0 else 0
        return 1 if line.dy < 0 else 0

    if line.dy == 0:
        if y <= line.v1.y:
            return 1 if line.dx < 0 else 0
        return 1 if line.dx > 0 else 0

    dx = x - line.v1.x
    dy = y - line.v1.y

    left = fixed_mul(line.dy >> FRACBITS, dx)
    right = fixed_mul(dy, line.dx >> FRACBITS)

    if right < left:
        return 0  # front side
    return 1  # back side


def box_on_line_side(tmbox: Sequence[int], line: Line) -> int:
    """Return 0 (front), 1 (back), or -1 if `tmbox` crosses `line`.

    Port of P_BoxOnLineSide (p_maputl.c).
    """
    if line.slopetype == SlopeType.ST_HORIZONTAL:
        p1 = 1 if tmbox[BOXTOP] > line.v1.y else 0
        p2 = 1 if tmbox[BOXBOTTOM] > line.v1.y else 0
        if line.dx < 0:
            p1 ^= 1
            p2 ^= 1
    elif line.slopetype == SlopeType.ST_VERTICAL:
        p1 = 1 if tmbox[BOXRIGHT] < line.v1.x else 0
        p2 = 1 if tmbox[BOXLEFT] < line.v1.x else 0
        if line.dy < 0:
            p1 ^= 1
            p2 ^= 1
    elif line.slopetype == SlopeType.ST_POSITIVE:
        p1 = point_on_line_side(tmbox[BOXLEFT], tmbox[BOXTOP], line)
        p2 = point_on_line_side(tmbox[BOXRIGHT], tmbox[BOXBOTTOM], line)
    else:  # SlopeType.ST_NEGATIVE
        p1 = point_on_line_side(tmbox[BOXRIGHT], tmbox[BOXTOP], line)
        p2 = point_on_line_side(tmbox[BOXLEFT], tmbox[BOXBOTTOM], line)

    if p1 == p2:
        return p1
    return -1


def point_on_divline_side(x: int, y: int, line: DivLine) -> int:
    """Return 0 (front) or 1 (back) for point `(x, y)` relative to `DivLine`.

    Port of P_PointOnDivlineSide (p_maputl.c).
    """
    if line.dx == 0:
        if x <= line.x:
            return 1 if line.dy > 0 else 0
        return 1 if line.dy < 0 else 0

    if line.dy == 0:
        if y <= line.y:
            return 1 if line.dx < 0 else 0
        return 1 if line.dx > 0 else 0

    dx = x - line.x
    dy = y - line.y

    if (line.dy ^ line.dx ^ dx ^ dy) & 0x80000000:
        if (line.dy ^ dx) & 0x80000000:
            return 1  # left is negative
        return 0

    left = fixed_mul(line.dy >> 8, dx >> 8)
    right = fixed_mul(dy >> 8, line.dx >> 8)

    if right < left:
        return 0  # front side
    return 1  # back side


def make_divline(line: Line) -> DivLine:
    """Convert a `Line` into a `DivLine`.

    Port of P_MakeDivline (p_maputl.c).
    """
    return DivLine(x=line.v1.x, y=line.v1.y, dx=line.dx, dy=line.dy)


def intercept_vector(v2: DivLine, v1: DivLine) -> int:
    """Return fractional intercept point along the first divline `v2` where `v1` crosses it.

    Port of P_InterceptVector (p_maputl.c).
    """
    den = fixed_mul(v1.dy >> 8, v2.dx) - fixed_mul(v1.dx >> 8, v2.dy)
    if den == 0:
        return 0
    num = fixed_mul((v1.x - v2.x) >> 8, v1.dy) + fixed_mul((v2.y - v1.y) >> 8, v1.dx)
    return fixed_div(num, den)


def line_opening(line: Line) -> LineOpening:
    """Compute the vertical opening across `line`.

    Port of P_LineOpening (p_maputl.c).
    """
    if line.sidenum[1] == -1 or line.back_sector is None or line.front_sector is None:
        return LineOpening(opentop=0, openbottom=0, lowfloor=0, openrange=-1)

    front = line.front_sector
    back = line.back_sector

    opentop = min(front.ceiling_height, back.ceiling_height)
    if front.floor_height > back.floor_height:
        openbottom = front.floor_height
        lowfloor = back.floor_height
    else:
        openbottom = back.floor_height
        lowfloor = front.floor_height

    openrange = opentop - openbottom
    return LineOpening(
        opentop=opentop,
        openbottom=openbottom,
        lowfloor=lowfloor,
        openrange=openrange,
    )


def _traverse_intercepts(
    intercepts: list[Intercept],
    max_frac: int,
    trav: Callable[[Intercept], bool],
) -> bool:
    """Process `intercepts` with `frac <= max_frac` in ascending order of `frac`.

    Port of P_TraverseIntercepts (p_maputl.c).
    """
    remaining = list(intercepts)
    while remaining:
        dist = MAXINT
        best_idx = -1
        for i, item in enumerate(remaining):
            if item.frac < dist:
                dist = item.frac
                best_idx = i
        if dist > max_frac or best_idx == -1:
            return True
        scan_choice = remaining.pop(best_idx)
        if not trav(scan_choice):
            return False
    return True


def path_traverse(
    level: Level,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    flags: int,
    trav: Callable[[Intercept], bool],
) -> bool:
    """Trace a path from `(x1, y1)` to `(x2, y2)`, invoking `trav` on intercepts in order.

    Port of P_PathTraverse, PIT_AddLineIntercepts, and PIT_AddThingIntercepts (p_maputl.c).
    """
    earlyout = bool(flags & PT_EARLYOUT)
    bm = level.blockmap
    valid_count = bm.next_valid_count()

    if ((x1 - bm.org_x) & (MAPBLOCKSIZE - 1)) == 0:
        x1 += FRACUNIT
    if ((y1 - bm.org_y) & (MAPBLOCKSIZE - 1)) == 0:
        y1 += FRACUNIT

    trace = DivLine(x=x1, y=y1, dx=x2 - x1, dy=y2 - y1)
    intercepts: list[Intercept] = []

    x1_rel = x1 - bm.org_x
    y1_rel = y1 - bm.org_y
    xt1 = x1_rel >> MAPBLOCKSHIFT
    yt1 = y1_rel >> MAPBLOCKSHIFT

    x2_rel = x2 - bm.org_x
    y2_rel = y2 - bm.org_y
    xt2 = x2_rel >> MAPBLOCKSHIFT
    yt2 = y2_rel >> MAPBLOCKSHIFT

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

    def add_line_intercepts(ld: Line) -> bool:
        if (
            trace.dx > FRACUNIT * 16
            or trace.dy > FRACUNIT * 16
            or trace.dx < -FRACUNIT * 16
            or trace.dy < -FRACUNIT * 16
        ):
            s1 = point_on_divline_side(ld.v1.x, ld.v1.y, trace)
            s2 = point_on_divline_side(ld.v2.x, ld.v2.y, trace)
        else:
            s1 = point_on_line_side(trace.x, trace.y, ld)
            s2 = point_on_line_side(trace.x + trace.dx, trace.y + trace.dy, ld)

        if s1 == s2:
            return True

        dl = make_divline(ld)
        frac = intercept_vector(trace, dl)
        if frac < 0:
            return True

        if earlyout and frac < FRACUNIT and ld.back_sector is None:
            return False

        intercepts.append(Intercept(frac=frac, is_aline=True, line=ld))
        return True

    def add_thing_intercepts(thing: Mobj) -> bool:
        tracepositive = (trace.dx ^ trace.dy) > 0
        if tracepositive:
            tx1 = thing.x - thing.radius
            ty1 = thing.y + thing.radius
            tx2 = thing.x + thing.radius
            ty2 = thing.y - thing.radius
        else:
            tx1 = thing.x - thing.radius
            ty1 = thing.y - thing.radius
            tx2 = thing.x + thing.radius
            ty2 = thing.y + thing.radius

        s1 = point_on_divline_side(tx1, ty1, trace)
        s2 = point_on_divline_side(tx2, ty2, trace)
        if s1 == s2:
            return True

        dl = DivLine(x=tx1, y=ty1, dx=tx2 - tx1, dy=ty2 - ty1)
        frac = intercept_vector(trace, dl)
        if frac < 0:
            return True

        intercepts.append(Intercept(frac=frac, is_aline=False, thing=thing))
        return True

    mapx = xt1
    mapy = yt1
    for _ in range(64):
        if flags & PT_ADDLINES:
            if not bm.block_lines_iterator(
                mapx, mapy, add_line_intercepts, valid_count=valid_count
            ):
                return False
        if flags & PT_ADDTHINGS:
            if not bm.block_things_iterator(mapx, mapy, add_thing_intercepts):
                return False

        if mapx == xt2 and mapy == yt2:
            break

        if (yintercept >> FRACBITS) == mapy:
            yintercept += ystep
            mapx += mapxstep
        elif (xintercept >> FRACBITS) == mapx:
            xintercept += xstep
            mapy += mapystep

    return _traverse_intercepts(intercepts, FRACUNIT, trav)
