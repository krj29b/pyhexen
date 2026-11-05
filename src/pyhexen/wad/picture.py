# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Doom/Hexen picture format decoder for patches, sprites, and UI graphics.

Port of patch_t / column_t / post_t handling in v_patch.h, r_data.c, and r_things.c.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

_PATCH_HEADER = struct.Struct("<4h")
_MAX_DIMENSION = 4096


@dataclass(slots=True)
class Picture:
    """Decoded Doom-format picture (patch, sprite, or UI graphic)."""

    width: int
    height: int
    left: int
    top: int
    indices: np.ndarray  # uint8[height, width]
    mask: np.ndarray  # bool[height, width]


def decode_picture(data: bytes | bytearray | memoryview) -> Picture:
    """Decode a Doom-format picture byte buffer into a `Picture`.

    Validates header dimensions, column offset table, and column posts,
    raising `ValueError` on truncated or malformed input.
    """
    buf = memoryview(data)
    total_len = len(buf)
    if total_len < _PATCH_HEADER.size:
        raise ValueError(
            f"Picture data too short for header ({total_len} bytes < {_PATCH_HEADER.size})"
        )

    width, height, left, top = _PATCH_HEADER.unpack_from(buf, 0)
    if width <= 0 or height <= 0 or width > _MAX_DIMENSION or height > _MAX_DIMENSION:
        raise ValueError(f"Invalid picture dimensions: width={width}, height={height}")

    ofs_table_end = _PATCH_HEADER.size + 4 * width
    if total_len < ofs_table_end:
        raise ValueError(
            f"Picture data too short for {width} column offsets "
            f"({total_len} bytes < {ofs_table_end})"
        )

    column_offsets = struct.unpack_from(f"<{width}I", buf, _PATCH_HEADER.size)
    indices = np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)

    for x, col_ofs in enumerate(column_offsets):
        if col_ofs < _PATCH_HEADER.size or col_ofs >= total_len:
            raise ValueError(
                f"Column {x} offset {col_ofs} out of bounds [{_PATCH_HEADER.size}..{total_len})"
            )

        ofs = col_ofs
        while True:
            if ofs >= total_len:
                raise ValueError(
                    f"Column {x} missing 0xFF terminator before end of buffer ({total_len})"
                )
            topdelta = buf[ofs]
            if topdelta == 0xFF:
                break

            if ofs + 4 > total_len:
                raise ValueError(
                    f"Column {x} post at offset {ofs} truncated before length/padding bytes"
                )
            length = buf[ofs + 1]
            post_end = ofs + 4 + length
            if post_end > total_len:
                raise ValueError(
                    f"Column {x} post at {ofs} (len {length}) exceeds buffer ({total_len})"
                )

            y0 = topdelta
            y1 = y0 + length
            if y1 > height:
                raise ValueError(f"Column {x} post [{y0}..{y1}] exceeds picture height {height}")

            if length > 0:
                pixels = np.frombuffer(buf, dtype=np.uint8, count=length, offset=ofs + 3)
                indices[y0:y1, x] = pixels
                mask[y0:y1, x] = True

            ofs = post_end

    return Picture(
        width=width,
        height=height,
        left=left,
        top=top,
        indices=indices,
        mask=mask,
    )
