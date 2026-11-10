# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Real-IWAD tests for vanilla 1995 Hexen DEMO1..3 lumps and playback sync (S1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.sim.demo import DemoPlayer, decode_vanilla_demo, play_demo_to_completion
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import spawn_player
from pyhexen.sim.polyobj import init_polyobjs
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


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
def test_vanilla_iwad_demo1_playback_sync(iwad_path: Path) -> None:
    """Replay DEMO1 to completion; assert all 968 tics simulate deterministically."""
    with WadStack(iwad_path) as stack:
        map_table = load_mapinfo(stack)
        player = DemoPlayer.from_file("DEMO1", wad=stack)
        map_info = map_table[player.demo.header.map_num]
        level = load_level(stack, map_info.lump_name)
        init_polyobjs(level)

        sim = GameSim(
            level=level,
            skill=player.demo.header.skill - 1,
            player_class=player.demo.header.player_class,
        )
        sim.rng.set_state((0, 0))
        spawn_map_things(sim)
        spawn_player(sim, 0)

        completed = play_demo_to_completion(sim, player)
        assert completed is True
        assert player.is_finished
        assert player.current_tic == 968
        assert sim.players[0].health > 0


@pytest.mark.wad
def test_vanilla_iwad_demo2_playback_sync(iwad_path: Path) -> None:
    """Replay DEMO2 to completion; assert all 2619 tics simulate deterministically."""
    with WadStack(iwad_path) as stack:
        map_table = load_mapinfo(stack)
        player = DemoPlayer.from_file("DEMO2", wad=stack)
        map_info = map_table[player.demo.header.map_num]
        level = load_level(stack, map_info.lump_name)
        init_polyobjs(level)

        sim = GameSim(
            level=level,
            skill=player.demo.header.skill - 1,
            player_class=player.demo.header.player_class,
        )
        sim.rng.set_state((0, 0))
        spawn_map_things(sim)
        spawn_player(sim, 0)

        completed = play_demo_to_completion(sim, player)
        assert completed is True
        assert player.is_finished
        assert player.current_tic == 2619


@pytest.mark.wad
def test_vanilla_iwad_demo3_playback_sync(iwad_path: Path) -> None:
    """Replay DEMO3 to completion; assert all 2404 tics simulate deterministically."""
    with WadStack(iwad_path) as stack:
        map_table = load_mapinfo(stack)
        player = DemoPlayer.from_file("DEMO3", wad=stack)
        map_info = map_table[player.demo.header.map_num]
        level = load_level(stack, map_info.lump_name)
        init_polyobjs(level)

        sim = GameSim(
            level=level,
            skill=player.demo.header.skill - 1,
            player_class=player.demo.header.player_class,
        )
        sim.rng.set_state((0, 0))
        spawn_map_things(sim)
        spawn_player(sim, 0)

        completed = play_demo_to_completion(sim, player)
        assert completed is True
        assert player.is_finished
        assert player.current_tic == 2404
