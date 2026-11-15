# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""S3a: an expansion PWAD laid out like Deathkings (HEXDD.WAD) on top of an IWAD.

HEXDD.WAD replaces MAPINFO (maps 33-38 and 41-60, MAP41 is warp 1), adds its own
cluster and win texts and DEMO1-3 recorded with Hexen v1.1 (8 player slots), and its
header says IWAD although it is loaded with -file. All synthetic here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.app.game_flow import GameFlow, GameState
from pyhexen.app.netgame import iwad_md5
from pyhexen.app.savegame import compute_wad_checksum
from pyhexen.core.ticcmd import TicCmd
from pyhexen.sim.demo import (
    DEMOMARKER,
    VanillaDemo,
    VanillaDemoHeader,
    decode_vanilla_demo,
    detect_demo_slots,
    encode_vanilla_demo,
)
from pyhexen.sim.transitions import teleport_end_game, teleport_new_map
from pyhexen.ui.intermission import DEFAULT_WIN_MESSAGES, Finale
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.wadfile import WadStack

BASE_MAPINFO = b'map 1 "BASE ONE"\nwarptrans 1\nnext 1\ncluster 1\n'
DK_MAPINFO = (
    b'map 41 "RUINED VILLAGE"\nwarptrans 1\nnext 2\ncluster 1\n\n'
    b'map 42 "BLIGHT"\nwarptrans 2\nnext 1\ncluster 2\n\n'
    b'map 83 "TEST"\nwarptrans 39\nnext 1\ncluster 4\n'
)
CLUS1 = "wiping a trembling hand across your\nbleeding face...\n"


def _room(name: str) -> MapBuilder:
    mb = MapBuilder(name)
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_thing(x=256, y=256, type=1)
    return mb


def _write_wads(tmp_path: Path) -> tuple[Path, Path]:
    base = WadBuilder("IWAD").add_playpal(2).add_colormap(33)
    base.add_lump("MAPINFO", BASE_MAPINFO)
    base.add_map(_room("MAP01"))
    dk = WadBuilder("IWAD")  # HEXDD.WAD is tagged IWAD too
    dk.add_lump("MAPINFO", DK_MAPINFO)
    dk.add_map(_room("MAP41"))
    dk.add_map(_room("MAP42"))
    dk.add_lump("CLUS1MSG", CLUS1.encode("ascii"))
    for i in range(3):
        dk.add_lump(f"WIN{i + 1}MSG", f"dk ending part {i + 1}\r\nsecond line\r\n".encode())
    return base.write(tmp_path / "base.wad"), dk.write(tmp_path / "dk.wad")


def test_expansion_mapinfo_replaces_the_iwad_one(tmp_path: Path) -> None:
    iwad, pwad = _write_wads(tmp_path)
    with WadStack([iwad, pwad]) as stack:
        table = load_mapinfo(stack)
    # Vanilla reads only the last MAPINFO lump (W_GetNumForName).
    assert sorted(table) == [41, 42, 83]
    assert table.translate_warp(1) == 41


def test_new_game_starts_on_warp_1_and_uses_the_expansion_texts(tmp_path: Path) -> None:
    iwad, pwad = _write_wads(tmp_path)
    with WadStack([iwad, pwad]) as stack:
        flow = GameFlow(stack, load_mapinfo(stack))
        flow.start_title()
        flow.new_game(0, 2)
        flow.tick()
        assert flow.state == GameState.LEVEL
        assert flow.sim is not None and flow.sim.map_num == 41

        teleport_new_map(flow.sim, 42, 0)
        for _ in range(3):
            flow.tick([TicCmd()])
        assert flow.state == GameState.INTERMISSION
        assert flow.intermission.hub_text == CLUS1  # the PWAD's CLUS1MSG

        flow.intermission.handle_key(0)
        for _ in range(40):
            flow.tick([TicCmd()])
            if flow.state == GameState.LEVEL:
                break
        assert flow.sim is not None and flow.sim.map_num == 42
        teleport_end_game(flow.sim)
        for _ in range(5):
            flow.tick([TicCmd()])
        assert flow.state == GameState.FINALE
        assert flow.finale.current_text == "dk ending part 1\nsecond line\n"


def test_finale_reads_win_lumps_and_falls_back_without_them(tmp_path: Path) -> None:
    iwad, pwad = _write_wads(tmp_path)
    with WadStack([iwad, pwad]) as stack:
        finale = Finale(stack)
        assert [finale.finale_text(i).splitlines()[0] for i in range(3)] == [
            "dk ending part 1",
            "dk ending part 2",
            "dk ending part 3",
        ]
    with WadStack(iwad) as stack:
        assert Finale(stack).finale_text(1) == DEFAULT_WIN_MESSAGES[1]


def _demo(slots: int, players: int = 1, tics: int = 5) -> bytes:
    header = VanillaDemoHeader(
        skill=2,
        episode=1,
        map_num=41,
        players=[(i < players, i % 3) for i in range(slots)],
        slots=slots,
    )
    cmds = tuple(TicCmd(forwardmove=25 + i, angleturn=0x100) for i in range(players))
    return encode_vanilla_demo(VanillaDemo(header=header, tics=[cmds] * tics))


def test_v11_demo_has_eight_player_slots() -> None:
    raw = _demo(8)
    assert len(raw) == 3 + 8 * 2 + 5 * 6 + 1
    assert raw[-1] == DEMOMARKER
    assert detect_demo_slots(raw) == 8
    decoded = decode_vanilla_demo(raw)
    assert decoded.header.slots == 8
    assert len(decoded.header.players) == 8
    assert len(decoded.tics) == 5
    assert decoded.tics[0][0].forwardmove == 25


def test_demo_slot_detection() -> None:
    for slots in (4, 8):
        for players in (1, 2, 3, 4):
            for tics in (0, 1, 7, 100):
                raw = _demo(slots, players, tics)
                assert detect_demo_slots(raw) == slots, (slots, players, tics)
                assert len(decode_vanilla_demo(raw).tics) == tics
    # Without a closing marker the layout can't be told apart: read as v1.0.
    assert detect_demo_slots(_demo(8)[:-1]) == 4
    # Forcing the slot count reads the stream from the other offset.
    assert decode_vanilla_demo(_demo(4), slots=4).header.slots == 4


def test_pwads_count_in_save_and_netgame_checksums(tmp_path: Path) -> None:
    iwad, pwad = _write_wads(tmp_path)
    alone = compute_wad_checksum(iwad)
    with WadStack(iwad) as stack:
        assert compute_wad_checksum(stack) == alone  # old saves keep loading
    with WadStack([iwad, pwad]) as stack:
        with_dk = compute_wad_checksum(stack)
    assert with_dk != alone
    assert compute_wad_checksum([iwad, pwad]) == with_dk

    assert iwad_md5(iwad) == hashlib.md5(iwad.read_bytes()).digest()
    assert iwad_md5(iwad, [pwad]) != iwad_md5(iwad)
