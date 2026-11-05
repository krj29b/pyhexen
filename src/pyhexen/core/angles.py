# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Binary Angle Measurement (BAM), trig table lookups, and angle/distance helpers.

Port of tables.h, tables.c, R_PointToAngle2 (r_main.c), and P_AproxDistance (p_maputl.c).
"""

from __future__ import annotations

from pyhexen.core.fixed import abs32, to_int32
from pyhexen.core.tables_data import (
    ANGLETOFINESHIFT,
    DBITS,
    FINEANGLES,
    FINECOSINE,
    FINEMASK,
    FINESINE,
    FINETANGENT,
    SLOPEBITS,
    SLOPERANGE,
    TANTOANGLE,
)

ANG45: int = 0x20000000
ANG90: int = 0x40000000
ANG180: int = 0x80000000
ANG270: int = 0xC0000000
ANG_MAX: int = 0xFFFFFFFF

ANG1: int = ANG45 // 45
ANG60: int = ANG180 // 3
ANG1_X: int = 0x01000000

__all__ = [
    "ANG1",
    "ANG1_X",
    "ANG45",
    "ANG60",
    "ANG90",
    "ANG180",
    "ANG270",
    "ANG_MAX",
    "ANGLETOFINESHIFT",
    "DBITS",
    "FINEANGLES",
    "FINECOSINE",
    "FINEMASK",
    "FINESINE",
    "FINETANGENT",
    "SLOPEBITS",
    "SLOPERANGE",
    "TANTOANGLE",
    "approx_distance",
    "fine_cosine",
    "fine_sine",
    "point_to_angle2",
    "slope_div",
]


def fine_sine(angle_or_fine: int) -> int:
    """Look up sine for a 32-bit BAM angle or a fine angle index in 0..FINEANGLES-1."""
    if 0 <= angle_or_fine < FINEANGLES:
        return FINESINE[angle_or_fine]
    return FINESINE[((angle_or_fine & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK]


def fine_cosine(angle_or_fine: int) -> int:
    """Look up cosine for a 32-bit BAM angle or a fine angle index in 0..FINEANGLES-1."""
    if 0 <= angle_or_fine < FINEANGLES:
        return FINECOSINE[angle_or_fine]
    return FINECOSINE[((angle_or_fine & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK]


def slope_div(num: int, den: int) -> int:
    """Compute TANTOANGLE index from unsigned numerator and denominator.

    Port of SlopeDiv (tables.c).
    """
    if den < 512:
        return SLOPERANGE
    ans = (num << 3) // (den >> 8)
    return ans if ans <= SLOPERANGE else SLOPERANGE


def point_to_angle2(x1: int, y1: int, x2: int, y2: int) -> int:
    """Compute 32-bit unsigned BAM angle from (x1, y1) to (x2, y2).

    Port of R_PointToAngle2 and R_PointToAngle (r_main.c).
    """
    x = x2 - x1
    y = y2 - y1
    if x == 0 and y == 0:
        return 0

    if x >= 0:
        if y >= 0:
            if x > y:
                return TANTOANGLE[slope_div(y, x)] & 0xFFFFFFFF  # octant 0
            return (ANG90 - 1 - TANTOANGLE[slope_div(x, y)]) & 0xFFFFFFFF  # octant 1
        y = -y
        if x > y:
            return (-TANTOANGLE[slope_div(y, x)]) & 0xFFFFFFFF  # octant 8
        return (ANG270 + TANTOANGLE[slope_div(x, y)]) & 0xFFFFFFFF  # octant 7

    x = -x
    if y >= 0:
        if x > y:
            return (ANG180 - 1 - TANTOANGLE[slope_div(y, x)]) & 0xFFFFFFFF  # octant 3
        return (ANG90 + TANTOANGLE[slope_div(x, y)]) & 0xFFFFFFFF  # octant 2

    y = -y
    if x > y:
        return (ANG180 + TANTOANGLE[slope_div(y, x)]) & 0xFFFFFFFF  # octant 4
    return (ANG270 - 1 - TANTOANGLE[slope_div(x, y)]) & 0xFFFFFFFF  # octant 5


def approx_distance(dx: int, dy: int) -> int:
    """Give a fast octagonal estimation of distance.

    Port of P_AproxDistance (p_maputl.c).
    """
    dx = abs32(dx)
    dy = abs32(dy)
    if dx < dy:
        return to_int32(dx + dy - (dx >> 1))
    return to_int32(dx + dy - (dy >> 1))
