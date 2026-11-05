# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Real-IWAD tests for T3.1, T3.2, T3.3, and T8.1: load every MAPxx and BEHAVIOR in HEXEN.WAD."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pyhexen.sim.acs import parse_behavior
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level

_MAP_RE = re.compile(r"^MAP\d\d$")


@pytest.mark.wad
def test_load_all_real_iwad_maps_and_behaviors(iwad_path: Path) -> None:
    with WadStack(iwad_path) as wad:
        map_names = sorted({lump.name for lump in wad.lumps if _MAP_RE.match(lump.name)})
        assert len(map_names) >= 30, (
            f"Expected at least 30 maps in {iwad_path}, found {len(map_names)}"
        )

        for map_name in map_names:
            level = load_level(wad, map_name)
            assert level.name == map_name
            assert len(level.vertexes) > 0
            assert len(level.sectors) > 0
            assert len(level.sides) > 0
            assert len(level.lines) > 0
            assert len(level.segs) > 0
            assert len(level.subsectors) > 0
            assert len(level.nodes) > 0
            assert level.blockmap.width > 0
            assert level.blockmap.height > 0

            # Query BSP at the first thing's position
            if level.things:
                t0 = level.things[0]
                ss = level.point_in_subsector(t0.x << 16, t0.y << 16)
                assert 0 <= ss.index < len(level.subsectors)

            # Parse BEHAVIOR lump
            beh = parse_behavior(level.behavior_bytes)
            assert beh.info_offset >= 8
