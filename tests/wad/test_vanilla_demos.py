# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Real-IWAD tests for vanilla 1995 Hexen DEMO1..3 lumps and playback sync (S1, T15.6).

Vanilla demos carry no state hash, so playing one to the end proves nothing about
sync. Sync is checked against per-tic traces from Chocolate Hexen, recorded with
`tools/demo_trace/record_traces.py` into `out/demo_traces/` (never committed: they
are derived from the IWAD). Those tests skip when the traces are missing or belong
to another IWAD.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pyhexen.sim.demo import (
    DemoPlayer,
    build_demo_sim,
    decode_vanilla_demo,
    play_demo_to_completion,
)
from pyhexen.tools.demo_sync import compare_demo
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.wadfile import WadStack

TRACE_DIR = Path(__file__).resolve().parents[2] / "out" / "demo_traces"
HEXEN_V10_MD5 = "b2543a03521365261d0a0f74d5dd90f0"
# Tics in sync with Chocolate Hexen on the v1.0 IWAD as of T15.6a (DEMO1 stops on the
# missing stairs special, T15.8). Raise these as divergences are fixed.
SYNCED_TICS_V10 = {"DEMO1": 419, "DEMO2": 2, "DEMO3": 170}
DEMO_LENGTHS = {"DEMO1": 968, "DEMO2": 2619, "DEMO3": 2404}


@pytest.mark.wad
def test_vanilla_iwad_demo_lumps(iwad_path: Path) -> None:
    """Verify headers and tic counts for vanilla DEMO1, DEMO2, and DEMO3 in commercial IWAD."""
    with WadStack(iwad_path) as stack:
        assert "DEMO1" in stack
        assert "DEMO2" in stack
        assert "DEMO3" in stack

        # DEMO1: Fighter, MAP03 (Seven Portals), Medium skill (2)
        d1 = decode_vanilla_demo(stack.get_bytes("DEMO1"))
        assert d1.header.skill == 2
        assert d1.header.episode == 1
        assert d1.header.map_num == 3
        assert d1.header.players[0] == (True, 0)
        assert len(d1.tics) == 968

        # DEMO2: Cleric, MAP23, Medium skill (2)
        d2 = decode_vanilla_demo(stack.get_bytes("DEMO2"))
        assert d2.header.skill == 2
        assert d2.header.episode == 1
        assert d2.header.map_num == 23
        assert d2.header.players[0] == (True, 1)
        assert len(d2.tics) == 2619

        # DEMO3: Mage, MAP24, Medium skill (2)
        d3 = decode_vanilla_demo(stack.get_bytes("DEMO3"))
        assert d3.header.skill == 2
        assert d3.header.episode == 1
        assert d3.header.map_num == 24
        assert d3.header.players[0] == (True, 2)
        assert len(d3.tics) == 2404


@pytest.mark.wad
@pytest.mark.parametrize("lump", ["DEMO1", "DEMO2", "DEMO3"])
def test_vanilla_demo_plays_to_the_end_without_a_hash_verdict(iwad_path: Path, lump: str) -> None:
    """The whole demo runs, and the missing hash no longer reads as "determinism passed"."""
    with WadStack(iwad_path) as stack:
        player = DemoPlayer.from_file(lump, wad=stack)
        sim = build_demo_sim(stack, load_mapinfo(stack), player.demo)
        assert not player.has_state_hash
        assert play_demo_to_completion(sim, player) is False
        assert player.is_finished
        assert player.current_tic == DEMO_LENGTHS[lump]


def _traces_for(iwad_path: Path) -> str:
    """MD5 of the IWAD the recorded traces belong to; skip if they don't match."""
    source = TRACE_DIR / "source.json"
    if not source.is_file():
        pytest.skip("no Chocolate Hexen traces (run tools/demo_trace/record_traces.py)")
    md5 = hashlib.md5(iwad_path.read_bytes()).hexdigest()
    if json.loads(source.read_text(encoding="ascii")).get("iwad_md5") != md5:
        pytest.skip("traces were recorded from a different IWAD")
    return md5


@pytest.mark.wad
@pytest.mark.parametrize("lump", ["DEMO1", "DEMO2", "DEMO3"])
def test_vanilla_demo_sync_against_chocolate_hexen(iwad_path: Path, lump: str) -> None:
    """Tic-by-tic comparison with the reference trace; no regression past the known tic."""
    md5 = _traces_for(iwad_path)
    if not (TRACE_DIR / f"{lump}.trace").is_file():
        pytest.skip(f"no trace for {lump}")
    with WadStack(iwad_path) as stack:
        ref_tics, div = compare_demo(stack, load_mapinfo(stack), lump, TRACE_DIR)
    assert ref_tics == DEMO_LENGTHS[lump]
    synced = ref_tics if div is None else div.index
    if md5 == HEXEN_V10_MD5:
        assert synced >= SYNCED_TICS_V10[lump], div.describe() if div else ""
