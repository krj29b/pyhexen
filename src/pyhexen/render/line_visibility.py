# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Mark the lines the player has seen, for the automap (ML_MAPPED).

Vanilla sets `ML_MAPPED` in R_StoreWallRange (r_segs.c) whenever a wall reaches the
screen. The GPU renderer has no software wall pass, so this module runs the CPU half
of it: a port of R_RenderBSPNode, R_Subsector, R_AddLine, R_CheckBBox and the
solid-seg clip list (r_bsp.c), at vanilla's 320-column resolution. Nothing is drawn;
a line is marked when any of its columns is not hidden behind solid walls.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import TYPE_CHECKING

from pyhexen.core.angles import ANG180, point_to_angle2
from pyhexen.world.bsp import NF_SUBSECTOR, point_on_side
from pyhexen.world.level import ML_MAPPED

if TYPE_CHECKING:
    from pyhexen.world.level import Level, Sector, Seg

ANGLE_MASK: int = 0xFFFFFFFF
BAM_TO_RAD: float = math.tau / 2**32

# R_CheckBBox corner table: which bbox edges bound the box from each view region.
# Indices into a node bbox (BOXTOP, BOXBOTTOM, BOXLEFT, BOXRIGHT) = (0, 1, 2, 3).
CHECKCOORD: tuple[tuple[int, int, int, int], ...] = (
    (3, 0, 2, 1),
    (3, 0, 2, 0),
    (3, 1, 2, 0),
    (0, 0, 0, 0),
    (2, 0, 2, 1),
    (0, 0, 0, 0),
    (3, 1, 3, 0),
    (0, 0, 0, 0),
    (2, 0, 3, 1),
    (2, 1, 3, 1),
    (2, 1, 3, 0),
)


def _no_mid_texture(name: str) -> bool:
    return not name or name.startswith("-")


class LineVisibility:
    """Marks lines visible from a viewpoint with ML_MAPPED (R_RenderBSPNode, r_bsp.c)."""

    def __init__(self, width: int = 320, fov_deg: float = 90.0) -> None:
        self.width: int = width
        self.fov_deg: float = fov_deg
        # Solid column ranges [first, last], sorted, merged when touching (solidsegs).
        self._firsts: list[int] = []
        self._lasts: list[int] = []
        self._view_x: int = 0
        self._view_y: int = 0
        self._view_angle: int = 0
        self._clip_angle: int = 0
        self._center_x: float = 0.0
        self._focal: float = 1.0
        self._level: Level | None = None
        self.newly_mapped: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark(self, level: Level, view_x: int, view_y: int, view_angle: int) -> int:
        """Run one BSP pass from (`view_x`, `view_y`) facing `view_angle` (BAM).

        Sets ML_MAPPED on every line with a visible seg and returns how many lines
        were newly marked.
        """
        half_fov = math.radians(min(max(self.fov_deg, 1.0), 170.0)) * 0.5
        self._clip_angle = int(half_fov / BAM_TO_RAD) & ANGLE_MASK
        self._center_x = self.width / 2.0
        self._focal = self._center_x / math.tan(half_fov)
        self._view_x = view_x
        self._view_y = view_y
        self._view_angle = view_angle & ANGLE_MASK
        self._level = level
        self._firsts = []
        self._lasts = []
        self.newly_mapped = 0

        if level.nodes:
            self._render_bsp_node(len(level.nodes) - 1)
        elif level.subsectors:
            self._subsector(0)
        self._level = None
        return self.newly_mapped

    # ------------------------------------------------------------------
    # Clip list
    # ------------------------------------------------------------------

    def _covered(self, first: int, last: int) -> bool:
        """True if columns first..last are all inside one solid range."""
        i = bisect_right(self._firsts, first) - 1
        return i >= 0 and self._lasts[i] >= last

    def _screen_full(self) -> bool:
        return self._covered(0, self.width - 1)

    def _add_solid(self, first: int, last: int) -> None:
        """Merge first..last into the solid list (the bookkeeping of R_ClipSolidWallSegment)."""
        firsts, lasts = self._firsts, self._lasts
        # Ranges that overlap or touch (adjacent pixels count as touching).
        lo = bisect_right(lasts, first - 2)
        hi = bisect_right(firsts, last + 1)
        if lo < hi:
            first = min(first, firsts[lo])
            last = max(last, lasts[hi - 1])
        firsts[lo:hi] = [first]
        lasts[lo:hi] = [last]

    # ------------------------------------------------------------------
    # Angles
    # ------------------------------------------------------------------

    def _angle_to_x(self, angle: int) -> int:
        """viewangletox: view-relative BAM angle (within +-clipangle) to a column."""
        signed = angle - (1 << 32) if angle & 0x80000000 else angle
        t = math.tan(signed * BAM_TO_RAD) * self._focal
        x = math.ceil(self._center_x - t)
        return min(max(x, 0), self.width)

    def _clip_span(self, angle1: int, angle2: int) -> tuple[int, int] | None:
        """Clip a view-relative angle span to the view edges; return its columns.

        Shared by R_AddLine and R_CheckBBox. Returns None when the span is off
        screen or does not cross a column.
        """
        clip = self._clip_angle
        span = (angle1 - angle2) & ANGLE_MASK
        tspan = (angle1 + clip) & ANGLE_MASK
        if tspan > 2 * clip:
            tspan -= 2 * clip
            if tspan >= span:
                return None  # totally off the left edge
            angle1 = clip
        tspan = (clip - angle2) & ANGLE_MASK
        if tspan > 2 * clip:
            tspan -= 2 * clip
            if tspan >= span:
                return None  # totally off the right edge
            angle2 = (-clip) & ANGLE_MASK
        x1 = self._angle_to_x(angle1)
        x2 = self._angle_to_x(angle2)
        if x1 == x2:
            return None  # does not cross a pixel
        return x1, x2

    # ------------------------------------------------------------------
    # BSP walk
    # ------------------------------------------------------------------

    def _render_bsp_node(self, bspnum: int) -> None:
        """Port of R_RenderBSPNode (r_bsp.c): front side first, back side if its box shows."""
        level = self._level
        assert level is not None
        if self._screen_full():
            return
        if bspnum & NF_SUBSECTOR:
            self._subsector(0 if bspnum == 0xFFFF else bspnum & ~NF_SUBSECTOR)
            return
        node = level.nodes[bspnum]
        side = point_on_side(self._view_x, self._view_y, node)
        self._render_bsp_node(node.children[side])
        if self._check_bbox(node.bbox[side ^ 1]):
            self._render_bsp_node(node.children[side ^ 1])

    def _check_bbox(self, box: tuple[int, int, int, int]) -> bool:
        """Port of R_CheckBBox (r_bsp.c): could any part of the box be visible?"""
        vx, vy = self._view_x, self._view_y
        if vx <= box[2]:
            boxx = 0
        elif vx < box[3]:
            boxx = 1
        else:
            boxx = 2
        if vy >= box[0]:
            boxy = 0
        elif vy > box[1]:
            boxy = 1
        else:
            boxy = 2
        boxpos = (boxy << 2) + boxx
        if boxpos == 5:
            return True
        c = CHECKCOORD[boxpos]
        angle1 = (point_to_angle2(vx, vy, box[c[0]], box[c[1]]) - self._view_angle) & ANGLE_MASK
        angle2 = (point_to_angle2(vx, vy, box[c[2]], box[c[3]]) - self._view_angle) & ANGLE_MASK
        if ((angle1 - angle2) & ANGLE_MASK) >= ANG180:
            return True  # sitting on a line
        cols = self._clip_span(angle1, angle2)
        if cols is None:
            return False
        return not self._covered(cols[0], cols[1] - 1)

    def _subsector(self, num: int) -> None:
        """Port of R_Subsector (r_bsp.c): the polyobj's segs first, then the subsector's."""
        level = self._level
        assert level is not None
        sub = level.subsectors[num]
        front = sub.sector
        poly = sub.poly
        if poly is not None:
            for seg in poly.segs:
                self._add_line(seg, front)
        segs = level.segs
        for i in range(sub.first_line, sub.first_line + sub.num_lines):
            self._add_line(segs[i], front)

    def _add_line(self, seg: Seg, front: Sector) -> None:
        """Port of R_AddLine (r_bsp.c); R_StoreWallRange's only effect here is ML_MAPPED."""
        vx, vy = self._view_x, self._view_y
        angle1 = point_to_angle2(vx, vy, seg.v1.x, seg.v1.y)
        angle2 = point_to_angle2(vx, vy, seg.v2.x, seg.v2.y)
        if ((angle1 - angle2) & ANGLE_MASK) >= ANG180:
            return  # back side
        cols = self._clip_span(
            (angle1 - self._view_angle) & ANGLE_MASK, (angle2 - self._view_angle) & ANGLE_MASK
        )
        if cols is None:
            return
        first, last = cols[0], cols[1] - 1

        back = seg.back_sector
        solid = False
        if back is None:
            solid = True  # single sided line
        elif back.ceiling_height <= front.floor_height or back.floor_height >= front.ceiling_height:
            solid = True  # closed door
        elif (
            back.ceiling_height == front.ceiling_height
            and back.floor_height == front.floor_height
            and back.ceiling_pic == front.ceiling_pic
            and back.floor_pic == front.floor_pic
            and back.light_level == front.light_level
            and back.special == front.special
            and _no_mid_texture(seg.sidedef.mid_texture)
        ):
            return  # empty line used for triggers and special events

        if not self._covered(first, last):
            line = seg.linedef
            if not line.flags & ML_MAPPED:
                line.flags |= ML_MAPPED
                self.newly_mapped += 1
        if solid:
            self._add_solid(first, last)
