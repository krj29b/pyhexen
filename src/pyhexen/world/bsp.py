# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""BSP traversal and point-on-side queries.

Port of R_PointOnSide, R_PointOnSegSide, and R_PointInSubsector (r_main.c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.fixed import FRACBITS, fixed_mul

if TYPE_CHECKING:
    from pyhexen.world.level import Node, Seg, Subsector

NF_SUBSECTOR: int = 0x8000


def point_on_side(x: int, y: int, node: Node) -> int:
    """Port of R_PointOnSide (r_main.c).

    Returns side 0 (front / right) or 1 (back / left) of the BSP partition line.
    """
    if node.dx == 0:
        if x <= node.x:
            return 1 if node.dy > 0 else 0
        return 1 if node.dy < 0 else 0
    if node.dy == 0:
        if y <= node.y:
            return 1 if node.dx < 0 else 0
        return 1 if node.dx > 0 else 0

    dx = x - node.x
    dy = y - node.y

    # Try to quickly decide by looking at sign bits
    if (node.dy ^ node.dx ^ dx ^ dy) & 0x80000000:
        if (node.dy ^ dx) & 0x80000000:
            return 1  # (left is negative)
        return 0

    left = fixed_mul(node.dy >> FRACBITS, dx)
    right = fixed_mul(dy, node.dx >> FRACBITS)

    if right < left:
        return 0  # front side
    return 1  # back side


def point_on_seg_side(x: int, y: int, line: Seg) -> int:
    """Port of R_PointOnSegSide (r_main.c).

    Returns side 0 (front / right) or 1 (back / left) of the seg.
    """
    lx = line.v1.x
    ly = line.v1.y

    ldx = line.v2.x - lx
    ldy = line.v2.y - ly

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

    # Try to quickly decide by looking at sign bits
    if (ldy ^ ldx ^ dx ^ dy) & 0x80000000:
        if (ldy ^ dx) & 0x80000000:
            return 1  # (left is negative)
        return 0

    left = fixed_mul(ldy >> FRACBITS, dx)
    right = fixed_mul(dy, ldx >> FRACBITS)

    if right < left:
        return 0  # front side
    return 1  # back side


def point_in_subsector(
    x: int,
    y: int,
    nodes: list[Node],
    subsectors: list[Subsector],
) -> Subsector:
    """Port of R_PointInSubsector (r_main.c).

    Traverses the BSP tree from the root (`len(nodes) - 1`) to locate the subsector
    containing `(x, y)` (in 16.16 fixed point).
    """
    if not nodes:
        return subsectors[0]

    nodenum = len(nodes) - 1
    while not (nodenum & NF_SUBSECTOR):
        node = nodes[nodenum]
        side = point_on_side(x, y, node)
        nodenum = node.children[side]

    return subsectors[nodenum & ~NF_SUBSECTOR]


class BSP:
    """Wrapper over a level's BSP nodes and subsectors."""

    __slots__ = ("nodes", "subsectors")

    def __init__(self, nodes: list[Node], subsectors: list[Subsector]) -> None:
        self.nodes = nodes
        self.subsectors = subsectors

    def point_on_side(self, x: int, y: int, node: Node) -> int:
        """Port of R_PointOnSide (r_main.c)."""
        return point_on_side(x, y, node)

    def point_on_seg_side(self, x: int, y: int, seg: Seg) -> int:
        """Port of R_PointOnSegSide (r_main.c)."""
        return point_on_seg_side(x, y, seg)

    def point_in_subsector(self, x: int, y: int) -> Subsector:
        """Port of R_PointInSubsector (r_main.c)."""
        return point_in_subsector(x, y, self.nodes, self.subsectors)
