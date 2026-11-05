# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Real-WAD tests for WadFile, WadStack, and namespace enumeration."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.wad.namespaces import flat_lumps, patch_lumps, sprite_lumps
from pyhexen.wad.wadfile import WadFile, WadStack


@pytest.mark.wad
def test_real_hexen_iwad_reader(iwad_path: Path) -> None:
    with WadFile(iwad_path) as iwad:
        assert iwad.wad_type == "IWAD"
        assert len(iwad) > 1000

        playpal = iwad.require("PLAYPAL")
        assert playpal.size >= 768
        assert playpal.size % 768 == 0
        assert len(iwad.get_bytes(playpal)) == playpal.size

        colormap = iwad.require("COLORMAP")
        assert colormap.size >= 256
        assert colormap.size % 256 == 0

        map01 = iwad.require("MAP01")
        assert map01.size == 0

        behaviors = iwad.find_all("BEHAVIOR")
        assert len(behaviors) >= 1
        first_behavior = iwad.get_bytes(behaviors[0])
        assert first_behavior[:4] == b"ACS\x00"

    with WadStack(iwad_path) as stack:
        assert len(stack) > 1000
        assert stack.find("PLAYPAL") is not None
        assert stack.find("MAP01") is not None
        assert stack.find("BEHAVIOR") is not None

        sprites = sprite_lumps(stack)
        assert len(sprites) > 100

        flats = flat_lumps(stack)
        assert len(flats) > 50
        assert all(not f.name.endswith(("_START", "_END")) for f in flats)
        assert all(f.size == 4096 for f in flats)

        patches = patch_lumps(stack)
        assert len(patches) > 50
