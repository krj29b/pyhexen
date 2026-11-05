# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Empirical Challenger 2 verification suite for Milestone 1 (T9.1 Polyobject Spawning).

Stress-tests and verifies real-world Hexen IWAD map fidelity across all 21 polyobject maps,
conducts in-depth inspections of MAP01 and MAP02, and asserts zero engine invariant violations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.core.fixed import FRACBITS
from pyhexen.render.mesh_walls import build_sector_walls
from pyhexen.sim.polyobj import (
    PO_ANCHOR_TYPE,
    PO_SPAWN_TYPE,
    PO_SPAWNCRUSH_TYPE,
    Polyobj,
    get_polyobj,
    get_polyobj_mirror,
)
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level, parse_map_lumps

# Official Chocolate Doom / Hexen polyobject map list (21 maps)
HEXEN_21_POLY_MAPS: dict[str, int] = {
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
class TestEmpiricalPolyobjectSpawningAcross21Maps:
    """Empirical verification across all 21 polyobject-bearing maps in commercial HEXEN.WAD."""

    @pytest.mark.parametrize("map_name,expected_count", HEXEN_21_POLY_MAPS.items())
    def test_polyobj_fidelity_per_map(
        self, iwad_path: Path, map_name: str, expected_count: int
    ) -> None:
        """Verify full spawning fidelity, invariant adherence, and blockmap registration."""
        with WadStack(iwad_path) as wad:
            level = load_level(wad, map_name)

        assert len(level.polyobjs_by_index) == expected_count, (
            f"Map {map_name} expected {expected_count} polyobjects, "
            f"got {len(level.polyobjs_by_index)}"
        )
        assert len(level.polyobjs) == expected_count

        bm = level.blockmap

        for idx, po in enumerate(level.polyobjs_by_index):
            assert isinstance(po, Polyobj)
            assert po.index == idx
            assert po.tag in level.polyobjs
            assert level.polyobjs[po.tag] is po
            assert get_polyobj(level, po.tag) is po
            assert po.numsegs > 0
            assert len(po.segs) == po.numsegs
            assert len(po.original_pts) == po.numsegs
            assert len(po.prev_pts) == po.numsegs

            # Fixed-point simulation invariants: no floats in coords or bbox
            assert isinstance(po.start_spot.x, int)
            assert isinstance(po.start_spot.y, int)
            assert isinstance(po.start_spot.z, int)
            for pt in po.original_pts:
                assert isinstance(pt.x, int)
                assert isinstance(pt.y, int)
            for pt in po.prev_pts:
                assert isinstance(pt.x, int)
                assert isinstance(pt.y, int)
            for b in po.bbox:
                assert isinstance(b, int)

            # Mathematical translation consistency:
            # original_pts[i] = v1 - start_spot
            # segs[i].v1 = start_spot + original_pts[i]
            for i, seg in enumerate(po.segs):
                orig_pt = po.original_pts[i]
                prev_pt = po.prev_pts[i]
                assert orig_pt.x == seg.v1.x - po.start_spot.x
                orig_y = seg.v1.y - po.start_spot.y
                assert orig_pt.y == orig_y
                assert prev_pt.x == seg.v1.x
                assert prev_pt.y == seg.v1.y

                # Linedef flag
                assert seg.linedef.is_polyobj is True

            # Subsector link
            assert po.subsector is not None
            assert po.subsector.poly is po

            # Blockmap bounding box validity
            box_top, box_bottom, box_left, box_right = po.bbox
            assert 0 <= box_left <= box_right < bm.width
            assert 0 <= box_bottom <= box_top < bm.height

            # Verify presence in every block cell within bbox
            for by in range(box_bottom, box_top + 1):
                for bx in range(box_left, box_right + 1):
                    block_idx = by * bm.width + bx
                    curr = bm.poly_block_map[block_idx]
                    found = False
                    while curr is not None:
                        if curr.polyobj is po:
                            found = True
                            break
                        curr = curr.next
                    assert found is True, (
                        f"{map_name} polyobj tag {po.tag} missing from blockmap cell ({bx}, {by})"
                    )


@pytest.mark.wad
class TestEmpiricalMap01InDepth:
    """In-depth empirical verification of HEXEN.WAD MAP01."""

    def test_map01_polyobjects_and_markers(self, iwad_path: Path) -> None:
        """Check all 12 polyobjects, anchor spots 3000, and crush spawn spots 3002."""
        with WadStack(iwad_path) as wad:
            level = load_level(wad, "MAP01")

        # 12 polyobjects: tags 4 through 15
        assert len(level.polyobjs_by_index) == 12
        assert len(level.polyobjs) == 12
        expected_tags = set(range(4, 16))
        assert set(level.polyobjs.keys()) == expected_tags

        # Verify thing markers in level.things
        anchors = {t.angle: t for t in level.things if t.type == PO_ANCHOR_TYPE}
        spawns = {t.angle: t for t in level.things if t.type in (PO_SPAWN_TYPE, PO_SPAWNCRUSH_TYPE)}

        for tag in expected_tags:
            assert tag in anchors, f"MAP01 missing anchor (3000) for tag {tag}"
            assert tag in spawns, f"MAP01 missing spawn spot (3001/3002) for tag {tag}"
            spawn_thing = spawns[tag]
            # All MAP01 polyobjects use PO_SPAWNCRUSH_TYPE (3002)
            assert spawn_thing.type == PO_SPAWNCRUSH_TYPE
            po = level.polyobjs[tag]
            assert po.crush is True

    def test_map01_entrance_double_doors(self, iwad_path: Path) -> None:
        """Verify entrance double doors (Tag 4 and Tag 5) mirror link and doorway translation."""
        with WadStack(iwad_path) as wad:
            level = load_level(wad, "MAP01")
            raw = parse_map_lumps(wad, "MAP01")

        po4 = level.polyobjs[4]
        po5 = level.polyobjs[5]

        # Mirror linking: Tag 4 mirrors Tag 5; Tag 5 is slave (mirror_tag 0)
        assert po4.mirror_tag == 5
        assert get_polyobj_mirror(level, 4) == 5
        assert po5.mirror_tag == 0

        # Sound sequences (SEQ_DOOR_STONE + seq_type: 3 = stone door sequence)
        assert po4.seq_type == 3
        assert po5.seq_type == 3

        # Check start spots (hinges)
        assert (po4.start_spot.x >> FRACBITS) == -1024
        assert (po4.start_spot.y >> FRACBITS) == 1232
        assert (po5.start_spot.x >> FRACBITS) == -1152
        assert (po5.start_spot.y >> FRACBITS) == 1232

        # Check mathematical translation from raw dummy room:
        # Tag 4 anchor in raw lumps: (-2672, -1232)
        raw_anchor_4 = next(t for t in raw.things if t["type"] == 3000 and t["angle"] == 4)
        raw_spawn_4 = next(t for t in raw.things if t["type"] == 3002 and t["angle"] == 4)
        assert raw_anchor_4["x"] == -2672 and raw_anchor_4["y"] == -1232
        assert raw_spawn_4["x"] == -1024 and raw_spawn_4["y"] == 1232

        delta_x = raw_anchor_4["x"] - raw_spawn_4["x"]  # -1648
        delta_y = raw_anchor_4["y"] - raw_spawn_4["y"]  # -2464
        assert delta_x == -1648
        assert delta_y == -2464

        # Raw vertices in dummy room for tag 4 start line: 489=(-2672, -1248), 490=(-2672, -1232)
        v489_x, v489_y = raw.vertexes[489]["x"], raw.vertexes[489]["y"]
        assert (v489_x, v489_y) == (-2672, -1248)
        # Verify translation formula: v_new = v_raw - delta
        assert v489_x - delta_x == -1024
        assert v489_y - delta_y == 1216

        # Check doorway bounding box:
        # po4 spans x in [-1088, -1024], y in [1216, 1232]
        p4_xs = [s.v1.x >> FRACBITS for s in po4.segs]
        p4_ys = [s.v1.y >> FRACBITS for s in po4.segs]
        assert min(p4_xs) == -1088 and max(p4_xs) == -1024
        assert min(p4_ys) == 1216 and max(p4_ys) == 1232

        # po5 spans x in [-1152, -1088], y in [1216, 1232]
        p5_xs = [s.v1.x >> FRACBITS for s in po5.segs]
        p5_ys = [s.v1.y >> FRACBITS for s in po5.segs]
        assert min(p5_xs) == -1152 and max(p5_xs) == -1088
        assert min(p5_ys) == 1216 and max(p5_ys) == 1232

        # Verify that po4 and po5 meet precisely at the center: x = -1088
        assert max(p5_xs) == min(p4_xs) == -1088

        # Second entrance double door pair: Tag 6 & Tag 7
        po6 = level.polyobjs[6]
        po7 = level.polyobjs[7]
        assert po6.mirror_tag == 7
        assert po7.mirror_tag == 0
        assert po6.seq_type == 0
        assert po7.seq_type == 0

    def test_map01_linedef_flags_and_static_mesh_omission(self, iwad_path: Path) -> None:
        """Verify all constituent linedefs have is_polyobj=True and are omitted from wall meshes."""
        with WadStack(iwad_path) as wad:
            level = load_level(wad, "MAP01")

        poly_lines: set[int] = set()
        poly_line_indices: set[int] = set()
        for po in level.polyobjs_by_index:
            for seg in po.segs:
                assert seg.linedef.is_polyobj is True
                poly_lines.add(id(seg.linedef))
                poly_line_indices.add(seg.linedef.index)

        # 12 polyobjects in MAP01 have polyobject linedefs
        assert len(poly_line_indices) > 0

        # Check that NO polyobject linedef is included in static sector wall quads
        for sec_idx in range(len(level.sectors)):
            quads = build_sector_walls(level, sec_idx)
            for quad in quads:
                assert quad.line_index not in poly_line_indices, (
                    f"Polyobject linedef {quad.line_index} was included in static wall quads "
                    f"for sector {sec_idx}!"
                )


@pytest.mark.wad
class TestEmpiricalMap02InDepth:
    """In-depth empirical verification of HEXEN.WAD MAP02."""

    def test_map02_all_24_polyobjects(self, iwad_path: Path) -> None:
        """Verify all 24 polyobjects, sliding pillars, doors, and static mesh omission."""
        with WadStack(iwad_path) as wad:
            level = load_level(wad, "MAP02")

        assert len(level.polyobjs_by_index) == 24
        assert len(level.polyobjs) == 24

        # Sliding pillars (tags 19..25) - commercial IWAD spawn type 3001 (non-crush)
        for tag in (19, 20, 21, 22, 23, 24, 25):
            assert tag in level.polyobjs, f"MAP02 missing pillar tag {tag}"
            po = level.polyobjs[tag]
            assert po.crush is False, f"MAP02 pillar polyobj {tag} must have crush=False"

        # Check linedef is_polyobj and mesh omission
        poly_line_indices: set[int] = set()
        for po in level.polyobjs_by_index:
            assert po.numsegs > 0
            assert po.subsector is not None
            assert po.subsector.poly is po
            for seg in po.segs:
                assert seg.linedef.is_polyobj is True
                poly_line_indices.add(seg.linedef.index)

        assert len(poly_line_indices) > 0

        # Build sector walls across all sectors in MAP02
        for sec_idx in range(len(level.sectors)):
            quads = build_sector_walls(level, sec_idx)
            for quad in quads:
                assert quad.line_index not in poly_line_indices, (
                    f"Polyobject linedef {quad.line_index} was included in static wall quads "
                    f"for sector {sec_idx} in MAP02!"
                )
