# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Real-IWAD asset decoding tests for T2.1-T2.5."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.wad.flats import FlatRegistry
from pyhexen.wad.namespaces import patch_lumps, sprite_lumps
from pyhexen.wad.palette import load_colormap, load_playpal
from pyhexen.wad.picture import decode_picture
from pyhexen.wad.sprites import SpriteRegistry
from pyhexen.wad.textures import TextureRegistry
from pyhexen.wad.wadfile import WadStack


@pytest.mark.wad
def test_real_hexen_palettes_and_colormaps(iwad_path: Path) -> None:
    with WadStack(iwad_path) as stack:
        playpal = load_playpal(stack)
        assert playpal.ndim == 3
        assert playpal.shape[1:] == (256, 3)
        assert playpal.shape[0] >= 14

        colormap = load_colormap(stack, "COLORMAP")
        assert colormap.ndim == 2
        assert colormap.shape[1] == 256
        assert colormap.shape[0] >= 32

        fogmap = load_colormap(stack, "FOGMAP")
        assert fogmap.ndim == 2
        assert fogmap.shape[1] == 256


@pytest.mark.wad
def test_real_hexen_pictures_textures_flats_and_sprites(iwad_path: Path) -> None:
    with WadStack(iwad_path) as stack:
        # Decode every sprite lump
        sprites_raw = sprite_lumps(stack)
        assert len(sprites_raw) > 100
        for lump in sprites_raw:
            pic = decode_picture(stack.get_bytes(lump))
            assert pic.width > 0 and pic.height > 0

        # Decode every patch lump
        patches_raw = patch_lumps(stack)
        assert len(patches_raw) > 50
        for lump in patches_raw:
            pic = decode_picture(stack.get_bytes(lump))
            assert pic.width > 0 and pic.height > 0

        # Compose every wall texture
        tex_reg = TextureRegistry(stack)
        assert len(tex_reg) > 100
        for i in range(1, len(tex_reg)):
            tex = tex_reg.get(i)
            assert tex.width > 0 and tex.height > 0
            assert tex.indices.shape == (tex.height, tex.width)
            assert tex.mask.shape == (tex.height, tex.width)

        # Load every flat
        flat_reg = FlatRegistry(stack)
        assert len(flat_reg) > 50
        for flat in flat_reg:
            assert flat.indices.shape == (64, 64)

        # Resolve sprite definitions
        spr_reg = SpriteRegistry(stack)
        non_empty = [s for s in spr_reg if s.numframes > 0]
        assert len(non_empty) > 50
