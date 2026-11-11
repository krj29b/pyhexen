# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T12.1 (Fonts & Overlay2D) and T12.3 (Messages).

The T11.2 SoundSystem tests moved to test_sound_system.py when T15.2 made it a Hexen port.
"""

from __future__ import annotations

from dataclasses import dataclass

import moderngl
import numpy as np
from tests.fixtures.wadbuilder import WadBuilder

from pyhexen.core.events import EventBus, PrintMessage
from pyhexen.render.overlay2d import (
    VIRTUAL_HEIGHT,
    VIRTUAL_WIDTH,
    Overlay2D,
    compute_4_3_viewport,
)
from pyhexen.render.textures import GpuTextureManager
from pyhexen.ui.fonts import FONT_A_NUM_GLYPHS, BitmapFont, FontSet
from pyhexen.ui.messages import MESSAGE_TICS, YELLOW_MESSAGE_Y, MessageSystem
from pyhexen.wad.picture import Picture
from pyhexen.wad.wadfile import WadFile


@dataclass
class _DummyMobj:
    x: int
    y: int
    z: int = 0
    angle: int = 0


def test_bitmap_font_and_overlay2d_cpu_and_gpu() -> None:
    wb = WadBuilder("IWAD")
    wb.add_playpal(num_palettes=2)
    wb.add_colormap(num_rows=33, include_fogmap=False)

    # Add FONTA_S and FONTA01..FONTA59 with width=6, height=7
    wb.add_marker("FONTA_S")
    glyph_pixels = np.full((7, 6), 192, dtype=np.uint8)
    glyph_mask = np.ones((7, 6), dtype=bool)
    glyph_mask[:, -1] = False  # 5 opaque columns + 1 spacing column
    for i in range(1, FONT_A_NUM_GLYPHS + 1):
        wb.add_picture(f"FONTA{i:02d}", glyph_pixels, mask=glyph_mask)

    wad = WadFile.from_bytes(wb.to_bytes())
    font_a = BitmapFont.from_wad(wad, start_marker="FONTA_S", prefix="FONTA")

    # Each non-space character advances by (6 - 1) = 5 px; space advances by 5 px
    assert font_a.text_width("HEXEN") == 5 * 5
    assert font_a.text_width("A B") == 5 + 5 + 5

    fonts = FontSet.from_wad(wad)
    overlay = Overlay2D(fonts=fonts)
    overlay.clear()
    assert not np.any(overlay.mask)

    # Fill a background bar and draw text + picture
    overlay.fill_rect(10, 10, 40, 20, color_index=45)
    assert np.all(overlay.indices[10:30, 10:50] == 45)
    assert np.all(overlay.mask[10:30, 10:50])

    pic = Picture(
        width=4,
        height=4,
        left=0,
        top=0,
        indices=np.full((4, 4), 100, dtype=np.uint8),
        mask=np.ones((4, 4), dtype=bool),
    )
    overlay.draw_picture(pic, 60, 15)
    assert np.all(overlay.indices[15:19, 60:64] == 100)

    overlay.draw_text_a("HI", 80, 20, yellow=False)
    overlay.draw_text_b("BIG", 160, 100, center=True)

    # Verify 4:3 aspect-correct viewport helper
    assert compute_4_3_viewport(1920, 1080) == (240, 0, 1440, 1080)
    assert compute_4_3_viewport(640, 480) == (0, 0, 640, 480)
    assert compute_4_3_viewport(800, 800) == (0, 100, 800, 600)

    # Headless ModernGL rendering verification
    ctx = moderngl.create_standalone_context()
    fbo = ctx.simple_framebuffer((320, 240))
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)

    tex_mgr = GpuTextureManager(ctx, wad)
    vp = overlay.render(ctx, tex_mgr, palette_index=0, gamma=1.0)
    assert vp == (0, 0, 320, 240)

    raw_rgb = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape((240, 320, 3))
    # Flip vertically from OpenGL bottom-left origin to top-left origin
    frame_top_down = raw_rgb[::-1]
    # The filled rectangle at y=10..30, x=10..50 (scaled 200 -> 240 vertically) must be non-black
    assert np.any(frame_top_down[12:34, 10:50] > 0)
    # Unmasked area at (y=190, x=300) remains black (0, 0, 0)
    assert np.all(frame_top_down[228, 300] == 0)

    overlay.release_gpu()
    fbo.release()
    ctx.release()


def test_message_system_events_expiration_and_drawing() -> None:
    bus = EventBus()
    msg_sys = MessageSystem(bus)
    overlay = Overlay2D(fonts=FontSet.synthetic())

    # Emit normal pickup message
    bus.emit(PrintMessage(text="GREETINGS MORTAL", bold=False))
    assert msg_sys.text == "GREETINGS MORTAL"
    assert msg_sys.tics == MESSAGE_TICS

    # Emit bold/yellow ACS message with multi-line text
    bus.emit(PrintMessage(text="ONE HALF OF THE PUZZLE\nIS COMPLETE", bold=True))
    assert msg_sys.yellow_text == "ONE HALF OF THE PUZZLE\nIS COMPLETE"
    assert msg_sys.yellow_tics == MESSAGE_TICS

    # Draw onto Overlay2D and verify pixels are written both at top-left and around y=68
    overlay.clear()
    msg_sys.draw(overlay)
    assert np.any(overlay.mask[1:10, 1:120])
    assert np.any(overlay.mask[YELLOW_MESSAGE_Y : YELLOW_MESSAGE_Y + 18, 60:260])

    # Advance 139 tics -> still active with 1 tic remaining
    for _ in range(MESSAGE_TICS - 1):
        msg_sys.tick()
    assert msg_sys.tics == 1
    assert msg_sys.yellow_tics == 1
    assert msg_sys.text != ""
    assert msg_sys.yellow_text != ""

    # 140th tic -> both expire
    msg_sys.tick()
    assert msg_sys.tics == 0
    assert msg_sys.yellow_tics == 0
    assert msg_sys.text == ""
    assert msg_sys.yellow_text == ""

    # After expiration, drawing leaves overlay empty
    overlay.clear()
    msg_sys.draw(overlay)
    assert not np.any(overlay.mask)
    assert overlay.indices.shape == (VIRTUAL_HEIGHT, VIRTUAL_WIDTH)
