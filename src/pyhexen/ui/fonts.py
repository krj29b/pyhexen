# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Bitmap font loader and text measurement/drawing for FontA, FontAYellow, and FontB (T12.1).

Port of InitFonts, MN_DrTextA, MN_DrTextAYellow, MN_TextAWidth, MN_DrTextB,
and MN_TextBWidth (mn_menu.c).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyhexen.wad.picture import Picture, decode_picture
from pyhexen.wad.wadfile import WadFile, WadStack

FONT_FIRST_CHAR: int = 33  # '!'
FONT_A_NUM_GLYPHS: int = 59
FONT_B_NUM_GLYPHS: int = 58
FONT_A_SPACE_WIDTH: int = 5
FONT_B_SPACE_WIDTH: int = 8


def blit_picture(
    canvas_indices: np.ndarray,
    canvas_mask: np.ndarray,
    pic: Picture,
    x: int,
    y: int,
    *,
    flip: bool = False,
    translucent: bool = False,
) -> None:
    """Blit a `Picture` onto 2D `(canvas_indices, canvas_mask)` arrays using vectorised slicing.

    Port of V_DrawPatch / V_DrawTLPatch (v_video.c). Applies `pic.left` and `pic.top` offsets
    and clips cleanly against canvas boundaries.
    """
    canvas_h, canvas_w = canvas_indices.shape
    draw_x = int(x) - int(pic.left)
    draw_y = int(y) - int(pic.top)

    src_x0 = max(0, -draw_x)
    src_y0 = max(0, -draw_y)
    dst_x0 = max(0, draw_x)
    dst_y0 = max(0, draw_y)

    copy_w = min(pic.width - src_x0, canvas_w - dst_x0)
    copy_h = min(pic.height - src_y0, canvas_h - dst_y0)
    if copy_w <= 0 or copy_h <= 0:
        return

    src_indices = pic.indices[:, ::-1] if flip else pic.indices
    src_mask = pic.mask[:, ::-1] if flip else pic.mask

    sub_idx = src_indices[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    sub_msk = src_mask[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]

    dst_idx_slice = canvas_indices[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w]
    dst_msk_slice = canvas_mask[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w]

    if translucent:
        # Simple 50% stipple or direct overwrite where mask is opaque
        np.copyto(dst_idx_slice, sub_idx, where=sub_msk)
        np.logical_or(dst_msk_slice, sub_msk, out=dst_msk_slice)
    else:
        np.copyto(dst_idx_slice, sub_idx, where=sub_msk)
        np.logical_or(dst_msk_slice, sub_msk, out=dst_msk_slice)


def _make_synthetic_glyph(
    char_code: int,
    width: int,
    height: int,
    color_index: int,
) -> Picture:
    """Create a deterministic synthetic bitmap glyph `Picture` for CI / headless tests."""
    indices = np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)

    # Draw a simple recognizable pattern inside [0..height-1, 0..width-2]
    usable_w = max(1, width - 1)
    # Outer vertical/horizontal bars derived from character bits
    for row in range(height):
        for col in range(usable_w):
            bit = ((char_code * 17 + row * 7 + col * 3) & 0x07) != 0
            if row in (0, height - 1) or col == 0 or bit:
                indices[row, col] = color_index & 0xFF
                mask[row, col] = True

    return Picture(
        width=width,
        height=height,
        left=0,
        top=0,
        indices=indices,
        mask=mask,
    )


class BitmapFont:
    """Bitmap font composed of indexed `Picture` glyphs for ASCII characters >= 33 ('!').

    Port of FontA / FontAYellow / FontB handling in mn_menu.c.
    """

    def __init__(
        self,
        glyphs: list[Picture] | tuple[Picture, ...],
        *,
        space_width: int = FONT_A_SPACE_WIDTH,
        line_height: int | None = None,
    ) -> None:
        self.glyphs: tuple[Picture, ...] = tuple(glyphs)
        self.space_width: int = int(space_width)
        if line_height is not None:
            self.line_height: int = int(line_height)
        elif self.glyphs:
            self.line_height = max(g.height for g in self.glyphs) + 1
        else:
            self.line_height = 9

    @classmethod
    def synthetic(
        cls,
        *,
        num_glyphs: int = FONT_A_NUM_GLYPHS,
        glyph_width: int = 6,
        glyph_height: int = 8,
        color_index: int = 200,
        space_width: int = FONT_A_SPACE_WIDTH,
    ) -> BitmapFont:
        """Build a synthetic `BitmapFont` when real WAD font lumps are absent."""
        glyphs = [
            _make_synthetic_glyph(FONT_FIRST_CHAR + i, glyph_width, glyph_height, color_index)
            for i in range(num_glyphs)
        ]
        return cls(glyphs, space_width=space_width, line_height=glyph_height + 1)

    @classmethod
    def from_wad(
        cls,
        wad: WadStack | WadFile | None,
        start_marker: str = "FONTA_S",
        prefix: str = "FONTA",
        num_glyphs: int = FONT_A_NUM_GLYPHS,
        *,
        space_width: int = FONT_A_SPACE_WIDTH,
        fallback_width: int = 6,
        fallback_height: int = 8,
        fallback_color: int = 200,
    ) -> BitmapFont:
        """Load a `BitmapFont` from WAD lumps (`FONTA01..59`, `FONTAY01..59`, `FONTB01..58`).

        Falls back to synthetic glyphs if the font lumps are not present in `wad`.
        """
        if wad is None:
            return cls.synthetic(
                num_glyphs=num_glyphs,
                glyph_width=fallback_width,
                glyph_height=fallback_height,
                color_index=fallback_color,
                space_width=space_width,
            )

        glyphs: list[Picture] = []
        # Try explicit lump names first (e.g. FONTA01..FONTA59)
        for i in range(1, num_glyphs + 1):
            lump_name = f"{prefix}{i:02d}"
            entry = wad.find(lump_name)
            if entry is None:
                glyphs.clear()
                break
            glyphs.append(decode_picture(wad.get_bytes(lump_name)))

        if glyphs:
            return cls(glyphs, space_width=space_width)

        # Fallback: check start_marker index in directory
        directory = getattr(wad, "lumps", ())
        marker_upper = start_marker.upper()
        for idx, entry in enumerate(directory):
            if entry.name == marker_upper and idx + num_glyphs < len(directory):
                try:
                    for offset in range(1, num_glyphs + 1):
                        lump_entry = directory[idx + offset]
                        glyphs.append(decode_picture(wad.get_bytes(lump_entry.name)))
                    return cls(glyphs, space_width=space_width)
                except Exception:
                    glyphs.clear()
                    break

        return cls.synthetic(
            num_glyphs=num_glyphs,
            glyph_width=fallback_width,
            glyph_height=fallback_height,
            color_index=fallback_color,
            space_width=space_width,
        )

    def glyph_for_char(self, ch: str) -> Picture | None:
        """Return the glyph `Picture` for a single character, or `None` for whitespace."""
        if not ch:
            return None
        code = ord(ch[0])
        if code < FONT_FIRST_CHAR:
            return None
        idx = code - FONT_FIRST_CHAR
        if idx >= len(self.glyphs):
            upper_code = ord(ch[0].upper())
            idx = upper_code - FONT_FIRST_CHAR
        if 0 <= idx < len(self.glyphs):
            return self.glyphs[idx]
        return None

    def text_width(self, text: str) -> int:
        """Return the pixel width of `text` (or the widest line if `text` contains newlines).

        Port of MN_TextAWidth / MN_TextBWidth (mn_menu.c):
        whitespace advances by `space_width`, and each glyph advances by `p.width - 1`.
        """
        if not text:
            return 0
        if "\n" in text:
            return max((self._line_width(line) for line in text.splitlines()), default=0)
        return self._line_width(text)

    def _line_width(self, line: str) -> int:
        width = 0
        for ch in line:
            if ord(ch) < FONT_FIRST_CHAR:
                width += self.space_width
            else:
                glyph = self.glyph_for_char(ch)
                if glyph is not None:
                    width += glyph.width - 1
                else:
                    width += self.space_width
        return width

    def draw_text(
        self,
        canvas_indices: np.ndarray,
        canvas_mask: np.ndarray,
        text: str,
        x: int,
        y: int,
    ) -> int:
        """Draw `text` onto `(canvas_indices, canvas_mask)` at `(x, y)` and return final `x`.

        Port of MN_DrTextA / MN_DrTextAYellow / MN_DrTextB (mn_menu.c).
        Supports multi-line strings separated by `\\n`.
        """
        cur_x = int(x)
        cur_y = int(y)
        for ch in text:
            if ch == "\n":
                cur_x = int(x)
                cur_y += self.line_height
                continue
            if ord(ch) < FONT_FIRST_CHAR:
                cur_x += self.space_width
                continue
            glyph = self.glyph_for_char(ch)
            if glyph is None:
                cur_x += self.space_width
                continue
            blit_picture(canvas_indices, canvas_mask, glyph, cur_x, cur_y)
            cur_x += glyph.width - 1
        return cur_x


@dataclass(slots=True)
class FontSet:
    """Bundle of Hexen's three standard UI bitmap fonts (`FontA`, `FontAYellow`, `FontB`)."""

    font_a: BitmapFont
    font_a_yellow: BitmapFont
    font_b: BitmapFont

    @classmethod
    def synthetic(cls) -> FontSet:
        """Create a synthetic `FontSet` for headless tests without a real IWAD."""
        return cls(
            font_a=BitmapFont.synthetic(
                num_glyphs=FONT_A_NUM_GLYPHS,
                glyph_width=6,
                glyph_height=8,
                color_index=180,
                space_width=FONT_A_SPACE_WIDTH,
            ),
            font_a_yellow=BitmapFont.synthetic(
                num_glyphs=FONT_A_NUM_GLYPHS,
                glyph_width=6,
                glyph_height=8,
                color_index=231,
                space_width=FONT_A_SPACE_WIDTH,
            ),
            font_b=BitmapFont.synthetic(
                num_glyphs=FONT_B_NUM_GLYPHS,
                glyph_width=10,
                glyph_height=14,
                color_index=140,
                space_width=FONT_B_SPACE_WIDTH,
            ),
        )

    @classmethod
    def from_wad(cls, wad: WadStack | WadFile | None = None) -> FontSet:
        """Load `FontA`, `FontAYellow`, and `FontB` from `wad` (with synthetic fallback)."""
        if wad is None:
            return cls.synthetic()
        return cls(
            font_a=BitmapFont.from_wad(
                wad,
                start_marker="FONTA_S",
                prefix="FONTA",
                num_glyphs=FONT_A_NUM_GLYPHS,
                space_width=FONT_A_SPACE_WIDTH,
                fallback_width=6,
                fallback_height=8,
                fallback_color=180,
            ),
            font_a_yellow=BitmapFont.from_wad(
                wad,
                start_marker="FONTAY_S",
                prefix="FONTAY",
                num_glyphs=FONT_A_NUM_GLYPHS,
                space_width=FONT_A_SPACE_WIDTH,
                fallback_width=6,
                fallback_height=8,
                fallback_color=231,
            ),
            font_b=BitmapFont.from_wad(
                wad,
                start_marker="FONTB_S",
                prefix="FONTB",
                num_glyphs=FONT_B_NUM_GLYPHS,
                space_width=FONT_B_SPACE_WIDTH,
                fallback_width=10,
                fallback_height=14,
                fallback_color=140,
            ),
        )
