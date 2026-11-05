# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""REJECT lump sight rejection matrix.

Port of rejectmatrix sight check in P_CheckSight (p_sight.c).
"""

from __future__ import annotations


class RejectMatrix:
    """Fast sector-to-sector line-of-sight rejection table.

    Port of rejectmatrix lookup in P_CheckSight (p_sight.c).
    """

    __slots__ = ("data", "num_sectors")

    def __init__(self, data: bytes, num_sectors: int) -> None:
        self.data: bytes = bytes(data)
        self.num_sectors: int = num_sectors

    def check_sight_reject(self, sec1_idx: int, sec2_idx: int) -> bool:
        """Return True if line of sight between `sec1_idx` and `sec2_idx` is rejected.

        Matches `p_sight.c`:
            pnum = s1 * numsectors + s2;
            bytenum = pnum >> 3;
            bitnum = 1 << (pnum & 7);
            if (rejectmatrix[bytenum] & bitnum) return false; // rejected
        """
        if (
            sec1_idx < 0
            or sec2_idx < 0
            or sec1_idx >= self.num_sectors
            or sec2_idx >= self.num_sectors
        ):
            return False
        pnum = sec1_idx * self.num_sectors + sec2_idx
        bytenum = pnum >> 3
        if bytenum >= len(self.data):
            return False
        bitnum = 1 << (pnum & 7)
        return bool(self.data[bytenum] & bitnum)
