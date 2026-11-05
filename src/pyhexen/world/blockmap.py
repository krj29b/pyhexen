# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Blockmap spatial grid loader and line/thing iterators.

Port of P_LoadBlockMap (p_setup.c), P_BlockLinesIterator, and P_BlockThingsIterator (p_maputl.c).
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACBITS, FRACUNIT, to_int16

if TYPE_CHECKING:
    from pyhexen.world.level import Line

MAPBLOCKUNITS: int = 128
MAPBLOCKSIZE: int = MAPBLOCKUNITS * FRACUNIT
MAPBLOCKSHIFT: int = FRACBITS + 7
MAPBMASK: int = MAPBLOCKSIZE - 1
MAPBTOFRAC: int = MAPBLOCKSHIFT - FRACBITS
MAXRADIUS: int = 32 * FRACUNIT


class Blockmap:
    """Spatial grid of 128x128 map-unit blocks for collision and line/thing queries.

    Port of blockmaplump / blockmap / blocklinks (p_setup.c, p_maputl.c).
    """

    __slots__ = (
        "org_x",
        "org_y",
        "width",
        "height",
        "lump_words",
        "block_links",
        "poly_block_map",
        "_lines",
        "valid_count",
    )

    def __init__(
        self,
        org_x: int,
        org_y: int,
        width: int,
        height: int,
        lump_words: tuple[int, ...],
        lines: list[Line],
    ) -> None:
        self.org_x: int = org_x
        self.org_y: int = org_y
        self.width: int = width
        self.height: int = height
        self.lump_words: tuple[int, ...] = lump_words
        num_blocks = max(0, width * height)
        self.block_links: list[Any] = [None] * num_blocks
        self.poly_block_map: list[Any] = [None] * num_blocks
        self._lines: list[Line] = lines
        self.valid_count: int = 1

    @classmethod
    def from_bytes(cls, data: bytes, lines: list[Line]) -> Blockmap:
        """Port of P_LoadBlockMap (p_setup.c)."""
        if len(data) < 8:
            return cls(0, 0, 0, 0, (), lines)
        count = len(data) // 2
        words = struct.unpack(f"<{count}h", data[: count * 2])
        org_x = words[0] << FRACBITS
        org_y = words[1] << FRACBITS
        width = words[2]
        height = words[3]
        return cls(org_x, org_y, width, height, words, lines)

    def next_valid_count(self) -> int:
        """Increment and return the blockmap's valid_count counter."""
        self.valid_count += 1
        return self.valid_count

    def world_to_block_x(self, x: int) -> int:
        """Convert a 16.16 fixed-point X coordinate to blockmap column index."""
        return (x - self.org_x) >> MAPBLOCKSHIFT

    def world_to_block_y(self, y: int) -> int:
        """Convert a 16.16 fixed-point Y coordinate to blockmap row index."""
        return (y - self.org_y) >> MAPBLOCKSHIFT

    def block_lines_iterator(
        self,
        x: int,
        y: int,
        func: Callable[[Line], bool],
        *,
        valid_count: int | None = None,
    ) -> bool:
        """Port of P_BlockLinesIterator (p_maputl.c).

        Calls `func(line)` for each line in block `(x, y)` not yet visited under `valid_count`.
        If `valid_count` is omitted, uses `self.valid_count`.
        Note: in `P_LoadBlockMap`, `blockmaplump` is `short *`, so unsigned offsets >= 0x8000
        are read as unsigned 16-bit indices into `lump_words` (or wrapped signed shorts in vanilla).
        We mask the offset table entry with `& 0xFFFF` so maps with large blockmaps work cleanly,
        while stopping at `-1` (`0xFFFF` in the blocklist).
        """
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True

        vc = self.valid_count if valid_count is None else valid_count
        block_idx = y * self.width + x

        # Check polyobj block links first if any exist (p_maputl.c lines 440-464)
        poly_link = self.poly_block_map[block_idx]
        while poly_link is not None:
            polyobj = getattr(poly_link, "polyobj", None)
            if polyobj is not None and getattr(polyobj, "valid_count", 0) != vc:
                polyobj.valid_count = vc
                for seg in getattr(polyobj, "segs", ()):
                    ld = seg.linedef
                    if ld.valid_count == vc:
                        continue
                    ld.valid_count = vc
                    if not func(ld):
                        return False
            poly_link = getattr(poly_link, "next", None)

        table_index = 4 + block_idx
        if table_index >= len(self.lump_words):
            return True

        # Offset from start of blockmaplump in 16-bit words
        list_offset = self.lump_words[table_index] & 0xFFFF
        n_words = len(self.lump_words)
        n_lines = len(self._lines)

        # Skip the leading 0 blocklist header word when present.
        idx = list_offset
        if idx < n_words and self.lump_words[idx] == 0:
            idx += 1

        while idx < n_words:
            line_num = to_int16(self.lump_words[idx])
            if line_num == -1:
                break
            idx += 1
            if 0 <= line_num < n_lines:
                ld = self._lines[line_num]
                if ld.valid_count == vc:
                    continue
                ld.valid_count = vc
                if not func(ld):
                    return False

        return True

    def block_things_iterator(
        self,
        x: int,
        y: int,
        func: Callable[[Any], bool],
    ) -> bool:
        """Port of P_BlockThingsIterator (p_maputl.c)."""
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True

        mobj = self.block_links[y * self.width + x]
        while mobj is not None:
            nxt = getattr(mobj, "bnext", None)
            if not func(mobj):
                return False
            mobj = nxt

        return True
