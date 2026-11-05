# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T2.1-T2.5 asset decoders (palette, picture, textures, flats, sprites)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from fixtures.wadbuilder import (
    PatchPlacement,
    TextureDef,
    WadBuilder,
    encode_picture,
)
from pyhexen.sim.info_data import SpriteNum
from pyhexen.wad.flats import FlatRegistry
from pyhexen.wad.palette import apply_palette, load_colormap, load_playpal, save_png
from pyhexen.wad.picture import decode_picture
from pyhexen.wad.sprites import SpriteRegistry
from pyhexen.wad.textures import TextureRegistry
from pyhexen.wad.wadfile import WadStack


def test_palette_and_colormap_loading_and_png(tmp_path: Path) -> None:
    wb = WadBuilder("IWAD").add_playpal(14).add_colormap(33, include_fogmap=True)
    with WadStack(wb.to_bytes()) as stack:
        pal = load_playpal(stack)
        assert pal.shape == (14, 256, 3)
        assert pal.dtype == np.uint8

        cmap = load_colormap(stack, "COLORMAP")
        assert cmap.shape == (33, 256)
        assert cmap.dtype == np.uint8

        fmap = load_colormap(stack, "FOGMAP")
        assert fmap.shape == (33, 256)
        assert fmap.dtype == np.uint8

        indices = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        mask = np.array([[True, False], [True, True]], dtype=bool)
        rgba = apply_palette(indices, pal, mask=mask, colormap_row=cmap[2])
        assert rgba.shape == (2, 2, 4)
        assert rgba.dtype == np.uint8
        assert tuple(rgba[0, 1]) == (0, 0, 0, 0)
        assert rgba[0, 0, 3] == 255
        expected_idx = cmap[2, 10]
        assert tuple(rgba[0, 0, :3]) == tuple(pal[0, expected_idx])

        png_path = tmp_path / "debug_out" / "test.png"
        save_png(indices, png_path, palette=pal, mask=mask)
        assert png_path.is_file()
        raw_png = png_path.read_bytes()
        assert raw_png.startswith(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError):
        load_playpal(b"short")
    with pytest.raises(ValueError):
        load_colormap(b"short")


def test_picture_decode_roundtrip_and_malformed_rejection() -> None:
    rng = np.random.default_rng(12345)
    pixels = rng.integers(0, 256, size=(16, 12), dtype=np.uint8)
    mask = rng.random(size=(16, 12)) > 0.35
    # Zero out masked-off pixels in reference so equality comparison is exact
    expected_pixels = np.where(mask, pixels, 0).astype(np.uint8)

    raw = encode_picture(pixels, mask=mask, left=-6, top=14)
    pic = decode_picture(raw)
    assert (pic.width, pic.height, pic.left, pic.top) == (12, 16, -6, 14)
    np.testing.assert_array_equal(pic.mask, mask)
    np.testing.assert_array_equal(pic.indices, expected_pixels)

    # 1. Too short for header
    with pytest.raises(ValueError):
        decode_picture(b"\x01\x02\x03")

    # 2. Non-positive dimensions
    with pytest.raises(ValueError):
        decode_picture(struct.pack("<4h", 0, 10, 0, 0))
    with pytest.raises(ValueError):
        decode_picture(struct.pack("<4h", 10, -2, 0, 0))

    # 3. Truncated column offset table
    with pytest.raises(ValueError):
        decode_picture(struct.pack("<4hI", 2, 2, 0, 0, 16))

    # 4. Out-of-bounds column offset
    with pytest.raises(ValueError):
        decode_picture(struct.pack("<4hI", 1, 2, 0, 0, 999))

    # 5. Missing 0xFF column terminator
    with pytest.raises(ValueError):
        decode_picture(struct.pack("<4hI", 1, 2, 0, 0, 12) + b"\x00\x01\x00\x2a\x00")

    # 6. Truncated post payload
    with pytest.raises(ValueError):
        decode_picture(struct.pack("<4hI", 1, 4, 0, 0, 12) + b"\x00\x03\x00\x2a")

    # 7. Post vertical span exceeds picture height
    with pytest.raises(ValueError):
        decode_picture(struct.pack("<4hI", 1, 2, 0, 0, 12) + b"\x01\x02\x00\x11\x22\x00\xff")


def test_composite_wall_textures() -> None:
    # Patch 0: solid 4x4 filled with index 10
    p0_pixels = np.full((4, 4), 10, dtype=np.uint8)
    # Patch 1: 4x4 with a diagonal of index 99, transparent elsewhere
    p1_pixels = np.full((4, 4), 99, dtype=np.uint8)
    p1_mask = np.eye(4, dtype=bool)

    wb = (
        WadBuilder("IWAD")
        .add_picture("PAT0", p0_pixels)
        .add_picture("PAT1", p1_pixels, mask=p1_mask)
        .add_textures(
            textures=[
                TextureDef(
                    name="COMP01",
                    width=6,
                    height=4,
                    patches=[
                        PatchPlacement(originx=0, originy=0, patch=0),
                        PatchPlacement(originx=2, originy=0, patch=1),
                    ],
                ),
                TextureDef(
                    name="CLIPPED",
                    width=4,
                    height=4,
                    patches=[
                        PatchPlacement(originx=-2, originy=-1, patch=0),
                    ],
                ),
            ],
            pnames=["PAT0", "PAT1"],
        )
    )

    with WadStack(wb.to_bytes()) as stack:
        reg = TextureRegistry(stack)
        assert len(reg) == 3  # index 0 ("-") + COMP01 + CLIPPED
        assert reg.check_num_for_name("-") == 0
        assert reg.num_for_name("-") == 0
        assert reg.get(0).name == "-"
        assert reg.get("-").width == 0

        comp_idx = reg.num_for_name("comp01")
        assert comp_idx == 1
        comp = reg.get("COMP01")
        assert comp is reg.get(comp_idx)  # cached instance
        assert (comp.width, comp.height) == (6, 4)

        # Columns 0..1 come solely from PAT0 (10)
        assert np.all(comp.indices[:, :2] == 10)
        assert np.all(comp.mask[:, :2])
        # Diagonal of PAT1 at x=2..5 overwrites with 99, off-diagonal in x=2..3 stays 10
        assert comp.indices[0, 2] == 99
        assert comp.indices[1, 2] == 10
        assert comp.indices[1, 3] == 99
        # At x=4..5 (outside PAT0), only the diagonal of PAT1 is opaque
        assert comp.indices[2, 4] == 99
        assert bool(comp.mask[2, 4]) is True
        assert bool(comp.mask[0, 4]) is False

        # Negative origin clipping check
        clipped = reg.get("CLIPPED")
        assert np.all(clipped.mask[:3, :2])
        assert not np.any(clipped.mask[3:, :])
        assert not np.any(clipped.mask[:, 2:])


def test_flat_registry() -> None:
    f1 = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    f2 = np.full((64, 64), 77, dtype=np.uint8)
    wb = (
        WadBuilder("IWAD")
        .add_marker("F_START")
        .add_marker("F1_START")
        .add_flat("F_ONE", f1)
        .add_flat("F_TWO", f2)
        .add_marker("F1_END")
        .add_marker("F_END")
    )
    with WadStack(wb.to_bytes()) as stack:
        flats = FlatRegistry(stack)
        assert len(flats) == 2
        assert "F_ONE" in flats
        assert flats.num_for_name("f_two") == 1
        np.testing.assert_array_equal(flats.get("F_ONE").indices, f1)
        np.testing.assert_array_equal(flats.get(1).indices, f2)


def test_sprite_registry_rotations_and_mirroring() -> None:
    dummy_pic = np.ones((4, 4), dtype=np.uint8)
    wb = (
        WadBuilder("IWAD")
        .add_marker("S_START")
        # MAN1 frame A has rotation 0 (single lump for all angles)
        .add_picture("MAN1A0", dummy_pic)
        # TROO frame A has 8 rotations using mirrored pairs (A1, A2A8, A3A7, A4A6, A5)
        .add_picture("TROOA1", dummy_pic)
        .add_picture("TROOA2A8", dummy_pic)
        .add_picture("TROOA3A7", dummy_pic)
        .add_picture("TROOA4A6", dummy_pic)
        .add_picture("TROOA5", dummy_pic)
        .add_marker("S_END")
    )

    with WadStack(wb.to_bytes()) as stack:
        reg = SpriteRegistry(stack)

        man1 = reg.get(SpriteNum.SPR_MAN1)
        assert man1.name == "MAN1"
        assert len(man1.frames) == 1
        fa = man1.frames[0]
        assert fa.rotate == 0
        assert len(fa.lumps) == 8
        assert all(lump.name == "MAN1A0" for lump in fa.lumps)
        assert fa.flip == (False,) * 8

        troo = reg.get("TROO")
        assert troo.name == "TROO"
        assert len(troo.frames) == 1
        tfa = troo.frames[0]
        assert tfa.rotate == 1
        assert [lump.name for lump in tfa.lumps] == [
            "TROOA1",
            "TROOA2A8",
            "TROOA3A7",
            "TROOA4A6",
            "TROOA5",
            "TROOA4A6",
            "TROOA3A7",
            "TROOA2A8",
        ]
        assert tfa.flip == (False, False, False, False, False, True, True, True)


def test_sprite_registry_rejects_incomplete_or_conflicting_rotations() -> None:
    dummy_pic = np.ones((2, 2), dtype=np.uint8)
    # Missing rotations 2..8 for TROO frame A
    wb_incomplete = (
        WadBuilder("IWAD")
        .add_marker("S_START")
        .add_picture("TROOA1", dummy_pic)
        .add_marker("S_END")
    )
    with WadStack(wb_incomplete.to_bytes()) as stack:
        with pytest.raises(ValueError, match="missing rotation"):
            SpriteRegistry(stack, sprite_names=["TROO"])

    # Mixing rot=0 and rot=1 on the same frame
    wb_conflict = (
        WadBuilder("IWAD")
        .add_marker("S_START")
        .add_picture("TROOA0", dummy_pic)
        .add_picture("TROOA1", dummy_pic)
        .add_marker("S_END")
    )
    with WadStack(wb_conflict.to_bytes()) as stack:
        with pytest.raises(ValueError, match="rot=0"):
            SpriteRegistry(stack, sprite_names=["TROO"])
