# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""T15.7: the scripted playthrough checklist, on a synthetic three-map WAD.

MAP01 and MAP02 (cluster 1) lead to each other; MAP03 (cluster 2) leads to MAP01.
Every exit is the shared line of two rooms, crossed from room 1 into room 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tests.fixtures.wadbuilder import MapBuilder, PatchPlacement, TextureDef, WadBuilder

from pyhexen.sim.info_data import MOBJINFO, MobjType
from pyhexen.tools.playthrough import ExitDef, Playthrough, _acs_exits
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.wadfile import WadStack

MAPINFO_TEXT = b"""
map 1 "ONE"
warptrans 1
next 2
cluster 1
sky1 SKY1 0

map 2 "TWO"
warptrans 2
next 1
cluster 1
sky1 SKY1 0

map 3 "THREE"
warptrans 3
next 1
cluster 2
sky1 SKY1 0
"""

ETTIN_DOOMEDNUM = MOBJINFO[MobjType.MT_ETTIN].doomednum


def _build_wad() -> WadStack:
    wb = WadBuilder("IWAD")
    wb.add_playpal(14)
    wb.add_colormap(33, include_fogmap=True)
    wb.add_lump("MAPINFO", MAPINFO_TEXT)
    wb.add_marker("P_START")
    wb.add_picture("WALL01", np.full((64, 64), 160, dtype=np.uint8))
    wb.add_marker("P_END")
    wb.add_textures(
        [TextureDef(name="WALL01", width=64, height=64, patches=[PatchPlacement(patch=0)])],
        pnames=["WALL01"],
    )
    wb.add_marker("F_START")
    wb.add_flat("FLOOR01", fill=30)
    wb.add_flat("CEIL01", fill=50)
    wb.add_marker("F_END")
    for name, target in (("MAP01", 2), ("MAP02", 1), ("MAP03", 1)):
        mb = MapBuilder(name)
        mb.add_two_sector_rooms(
            width1=512,
            width2=256,
            height=512,
            floor2=0,
            portal_special=74,
            portal_args=(target, 0, 0, 0, 0),
        )
        mb.add_thing(x=128, y=256, type=1)  # player start, position 0
        mb.add_thing(x=300, y=100, type=ETTIN_DOOMEDNUM, angle=90)
        wb.add_map(mb)
    return WadStack(wb.to_bytes())


def _playthrough(tmp_path: Path) -> Playthrough:
    stack = _build_wad()
    return Playthrough(stack, load_mapinfo(stack), save_dir=tmp_path)


def test_exit_lines_are_walked_and_lead_to_the_right_start(tmp_path: Path) -> None:
    pt = _playthrough(tmp_path)
    exits = pt.find_exits(1)
    assert [(e.special, e.args) for e in exits] == [(74, (2, 0))]
    check = pt.check_exit(exits[0])
    assert check.status == "PASS", check.detail
    assert "map 2, start 0" in check.detail


def test_exit_to_a_wrong_map_fails(tmp_path: Path) -> None:
    pt = _playthrough(tmp_path)
    real = pt.find_exits(1)[0]
    wrong = ExitDef(real.map_num, real.special, (3, 0), line_index=real.line_index)
    assert pt.check_exit(wrong).status == "FAIL"  # the line still goes to MAP02


def test_hub_return_keeps_the_dead_and_cluster_change_strips_keys(tmp_path: Path) -> None:
    pt = _playthrough(tmp_path)
    pt.check_hubs()
    by_name = {c.name: c for c in pt.report.checks}
    hub = by_name["MAP01 -> MAP02 -> MAP01"]
    assert hub.status == "PASS", hub.detail
    assert "1 dead stayed dead" in hub.detail
    cluster = next(c for c in pt.report.checks if c.name.startswith("cluster change"))
    assert cluster.status == "PASS", cluster.detail


def test_reborn_and_soak(tmp_path: Path) -> None:
    pt = _playthrough(tmp_path)
    reborn = pt.check_reborn(1)
    assert reborn.status == "PASS", reborn.detail
    soak = pt.soak(1, 2, tics=105)
    assert soak.status == "PASS", soak.detail


def test_acs_exit_scan_reads_direct_and_stack_forms() -> None:
    class _Script:
        def __init__(self, number: int, offset: int) -> None:
            self.number, self.offset, self.is_open = number, offset, False

    class _Behavior:
        pass

    words = [
        # script 1: LSPEC2DIRECT Teleport_NewMap 5 2, TERMINATE
        10,
        74,
        5,
        2,
        1,
        # script 2: PUSHNUMBER 7, PUSHNUMBER 0, LSPEC2 Teleport_NewMap, TERMINATE
        3,
        7,
        3,
        0,
        5,
        74,
        1,
    ]
    behavior = _Behavior()
    behavior.code = np.array(words, dtype="<i4").tobytes()
    behavior.scripts = (_Script(1, 0), _Script(2, 20))
    behavior.info_offset = len(behavior.code)
    found = _acs_exits(4, behavior)
    assert [(e.script, e.special, e.args, e.args_known) for e in found] == [
        (1, 74, (5, 2), True),
        (2, 74, (0, 0), False),
    ]
