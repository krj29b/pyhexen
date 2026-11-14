# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""S2d: real `pyhexen` processes play netgames over localhost UDP (synthetic WAD)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fixtures.netsession import run_netgame
from fixtures.wadbuilder import MapBuilder, PatchPlacement, TextureDef, WadBuilder
from pyhexen.sim.info_data import MOBJINFO, MobjType

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
"""


def _write_wad(path: Path) -> Path:
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
    ettin = MOBJINFO[MobjType.MT_ETTIN].doomednum
    for name in ("MAP01", "MAP02"):
        mb = MapBuilder(name)
        mb.add_room(x=0, y=0, width=1024, height=1024)
        for i in range(3):
            mb.add_thing(x=100 + 150 * i, y=100, type=i + 1, angle=90)
            mb.add_thing(x=100 + 150 * i, y=900, type=11, angle=270)
        mb.add_thing(x=600, y=600, type=ettin)
        wb.add_map(mb)
    return wb.write(path)


def test_three_processes_play_coop_in_sync(tmp_path: Path) -> None:
    wad = _write_wad(tmp_path / "NET.WAD")
    results = run_netgame(wad, 3, 700)
    for r in results:
        assert r.returncode == 0 and r.tic == 700 and r.desync == -1, r.output[-2000:]
    assert sorted(r.player for r in results) == [0, 1, 2]
    assert len({r.digest for r in results}) == 1


def test_two_processes_play_deathmatch_through_a_timer_map_change(tmp_path: Path) -> None:
    wad = _write_wad(tmp_path / "NET.WAD")
    results = run_netgame(wad, 2, 35 * 60 + 600, host_args=("--deathmatch", "--timer", "1"))
    for r in results:
        assert r.returncode == 0 and r.desync == -1, r.output[-2000:]
    assert all(r.map_num == 2 for r in results)  # the timer ended MAP01
    assert len({r.digest for r in results}) == 1
