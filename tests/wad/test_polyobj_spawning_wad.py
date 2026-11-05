# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Real-IWAD verification tests for Milestone 1 (T9.1 Polyobject Spawning)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.sim.polyobj import Polyobj
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level

# Verified polyobject counts across all 21 maps with polyobjects in commercial HEXEN.WAD
HEXEN_POLYOBJ_MAPS: dict[str, int] = {
    "MAP01": 12,
    "MAP02": 24,
    "MAP04": 3,
    "MAP05": 6,
    "MAP08": 2,
    "MAP09": 2,
    "MAP10": 7,
    "MAP13": 4,
    "MAP21": 5,
    "MAP23": 6,
    "MAP24": 1,
    "MAP25": 2,
    "MAP27": 9,
    "MAP28": 8,
    "MAP30": 12,
    "MAP31": 8,
    "MAP33": 3,
    "MAP35": 2,
    "MAP36": 17,
    "MAP39": 3,
    "MAP40": 2,
}


@pytest.mark.wad
def test_wad_map01_polyobjs(iwad_path: Path) -> None:
    """Validate MAP01 polyobject spawning in real HEXEN.WAD."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP01")

    # MAP01 has exactly 12 polyobjects (tags 4 through 15)
    assert len(level.polyobjs_by_index) == 12
    assert len(level.polyobjs) == 12

    for tag in range(4, 16):
        assert tag in level.polyobjs
        po: Polyobj = level.polyobjs[tag]
        assert po.tag == tag
        assert po.crush is True  # All MAP01 polyobjects use thing type 3002
        assert po.numsegs > 0
        assert len(po.segs) == po.numsegs
        assert len(po.original_pts) == po.numsegs
        assert len(po.prev_pts) == po.numsegs
        assert po.subsector is not None
        assert po.subsector.poly is po

        assert po.segs[0].linedef.special == 0
        for seg in po.segs:
            assert seg.linedef.is_polyobj is True

    # Specific pairs: Entrance swinging doors (tags 4 & 5)
    po4 = level.polyobjs[4]
    po5 = level.polyobjs[5]
    assert po4.mirror_tag == 5
    assert po4.seq_type == 3
    assert po5.mirror_tag == 0
    assert po5.seq_type == 3

    # Entrance double doors (tags 6 & 7)
    po6 = level.polyobjs[6]
    po7 = level.polyobjs[7]
    assert po6.mirror_tag == 7
    assert po6.seq_type == 0
    assert po7.mirror_tag == 0
    assert po7.seq_type == 0


@pytest.mark.wad
def test_wad_all_polyobj_maps(iwad_path: Path) -> None:
    """Validate polyobject spawning across all 21 polyobject-bearing maps in HEXEN.WAD."""
    with WadStack(iwad_path) as wad:
        for map_name, expected_count in HEXEN_POLYOBJ_MAPS.items():
            level = load_level(wad, map_name)
            count = len(level.polyobjs_by_index)
            assert count == expected_count, (
                f"{map_name}: expected {expected_count} polyobjects, got {count}"
            )

            bm = level.blockmap
            for po in level.polyobjs_by_index:
                assert po.numsegs > 0
                assert len(po.segs) == po.numsegs
                assert len(po.original_pts) == po.numsegs
                assert len(po.prev_pts) == po.numsegs
                assert po.subsector is not None
                assert po.subsector.poly is po

                # Verify bbox within blockmap dimensions
                box_top, box_bottom, box_left, box_right = po.bbox
                assert 0 <= box_left <= box_right < bm.width
                assert 0 <= box_bottom <= box_top < bm.height

                # Verify blockmap spatial link exists
                linked = False
                for by in range(box_bottom, box_top + 1):
                    for bx in range(box_left, box_right + 1):
                        curr = bm.poly_block_map[by * bm.width + bx]
                        while curr is not None:
                            if curr.polyobj is po:
                                linked = True
                                break
                            curr = curr.next
                        if linked:
                            break
                    if linked:
                        break
                assert linked is True, f"{map_name} polyobj {po.tag} not linked in blockmap"


@pytest.mark.wad
def test_wad_non_polyobj_maps(iwad_path: Path) -> None:
    """Verify that maps without polyobjects (e.g. MAP03, MAP06) load cleanly with 0 polyobjs."""
    with WadStack(iwad_path) as wad:
        for map_name in ("MAP03", "MAP06", "MAP11", "MAP12", "MAP22", "MAP26"):
            level = load_level(wad, map_name)
            assert len(level.polyobjs_by_index) == 0
            assert len(level.polyobjs) == 0
