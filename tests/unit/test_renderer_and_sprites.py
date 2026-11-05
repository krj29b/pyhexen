# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for sprite rotation selection, billboard quad generation, and WorldRenderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fixtures.wadbuilder import WadBuilder
from pyhexen.core.fixed import FRACUNIT
from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
from pyhexen.render.renderer import WorldRenderer
from pyhexen.render.sprites import select_sprite_rotation
from pyhexen.wad.sprites import SpriteFrame
from pyhexen.wad.wadfile import LumpInfo, WadStack
from pyhexen.world.level import load_level


def test_select_sprite_rotation() -> None:
    lumps = [LumpInfo(f"TESTA{i}", 0, 10, "test", i) for i in range(1, 9)]
    flip = [False] * 8

    # Rotate = 0 (same lump for all viewing angles)
    frame_rot0 = SpriteFrame(rotate=0, lumps=tuple([lumps[0]] * 8), flip=tuple([False] * 8))
    lump, is_flipped = select_sprite_rotation(frame_rot0, 0, 0, 0, 100 * FRACUNIT, 0)
    assert lump.name == "TESTA1"
    assert not is_flipped

    # Rotate = 1 (8 rotations)
    frame_rot1 = SpriteFrame(rotate=1, lumps=tuple(lumps), flip=tuple(flip))
    # Mobj facing East (angle 0):
    # Viewer is East of mobj (+X) looking West at mobj -> viewer is in front of mobj (rotation 1)
    lump_front, _ = select_sprite_rotation(frame_rot1, 0, 100 * FRACUNIT, 0, 0, 0)
    assert lump_front.name == "TESTA1"

    # Viewer is West of mobj (-X) looking East at mobj -> viewer is behind mobj (rotation 5)
    lump_back, _ = select_sprite_rotation(frame_rot1, 0, -100 * FRACUNIT, 0, 0, 0)
    assert lump_back.name == "TESTA5"


def test_world_renderer_and_screenshot(tmp_path: Path) -> None:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2)
    wb.add_colormap(33, include_fogmap=True)

    dummy_pic = np.full((16, 16), 128, dtype=np.uint8)
    wb.add_marker("S_START")
    wb.add_picture("BAR1A0", dummy_pic)
    wb.add_marker("S_END")

    wb.add_flat("FLOOR01", fill=10)
    wb.add_flat("CEIL01", fill=20)
    wb.add_flat("F_SKY", fill=0)

    from fixtures.wadbuilder import MapBuilder

    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 512, 512, floor_pic="FLOOR01", ceiling_pic="CEIL01")
    mb.add_thing(tid=0, x=256, y=256, z=0, angle=0, type=1)
    wb.add_map(mb)

    wad_bytes = wb.to_bytes()
    with WadStack(wad_bytes) as stack:
        level = load_level(stack, "MAP01")
        ctx = RenderContext(standalone=True, width=320, height=200)
        renderer = WorldRenderer(ctx, wad=stack)
        renderer.load_map(stack, level)

        camera = Camera()
        camera.x = 256.0
        camera.y = 32.0
        camera.z = -256.0

        # Render frame
        renderer.render_frame(camera)

        # Save screenshot
        shot_path = tmp_path / "screenshot.png"
        saved = renderer.save_screenshot(shot_path)
        assert saved.is_file()
        assert saved.stat().st_size > 0
