# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for the T15.1 game-flow controller (app/game_flow.py)."""

from __future__ import annotations

import pygame
import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.app.game_flow import GameAction, GameFlow, GameState
from pyhexen.core.fixed import FRACBITS
from pyhexen.core.ticcmd import BT_USE, TicCmd
from pyhexen.sim.inter import damage_mobj
from pyhexen.sim.player import PlayerState
from pyhexen.sim.transitions import exit_normal, teleport_end_game, teleport_new_map
from pyhexen.ui.intermission import TitleSequence
from pyhexen.ui.menu import MenuManager, MenuType
from pyhexen.wad.mapinfo import MapInfo, MapInfoTable
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import SPAC_CROSS

MAX_TICS = 200


def _build_flow() -> GameFlow:
    """Three maps: MAP01 and MAP02 in cluster 1 (a hub), MAP03 in cluster 2.

    Warp numbers are deliberately not the map numbers: MAP02 is warp 3 and MAP03
    is warp 2, so translated and untranslated paths give different maps.
    """
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)

    mb1 = MapBuilder("MAP01")
    mb1.add_two_sector_rooms()
    mb1.add_thing(x=128, y=128, type=1)
    wb.add_map(mb1)

    mb2 = MapBuilder("MAP02")
    mb2.add_room(x=0, y=0, width=512, height=512)
    mb2.add_thing(x=256, y=256, type=1)
    mb2.add_thing(x=400, y=100, type=1, args=(1, 0, 0, 0, 0))  # start spot group 1
    wb.add_map(mb2)

    mb3 = MapBuilder("MAP03")
    mb3.add_room(x=0, y=0, width=512, height=512)
    mb3.add_thing(x=256, y=256, type=1)
    wb.add_map(mb3)

    mapinfo = MapInfoTable(
        {
            1: MapInfo(
                map_num=1, name="ONE", lump_name="MAP01", cluster=1, warp_trans=1, next_map=2
            ),
            2: MapInfo(
                map_num=2, name="TWO", lump_name="MAP02", cluster=1, warp_trans=3, next_map=1
            ),
            3: MapInfo(
                map_num=3, name="THREE", lump_name="MAP03", cluster=2, warp_trans=2, next_map=1
            ),
        }
    )
    return GameFlow(WadStack(wb.to_bytes()), mapinfo)


def _tick_until(flow: GameFlow, state: GameState, cmd: TicCmd | None = None) -> int:
    """Tick until `flow.state == state` and no action is pending; return tics used."""
    for n in range(1, MAX_TICS + 1):
        flow.tick([cmd] if cmd is not None else [])
        if flow.state == state and flow.pending_action == GameAction.NOTHING:
            return n
    raise AssertionError(f"never reached {state!r}; stuck in {flow.state!r}")


def _start_game(flow: GameFlow, player_class: int = 0, skill: int = 2) -> None:
    flow.start_title()
    flow.new_game(player_class, skill)
    flow.tick()
    assert flow.state == GameState.LEVEL


def _exit_through_line(flow: GameFlow, map_num: int, position: int = 0) -> None:
    """Cross a Teleport_NewMap (74) line as the player, the way the sim would."""
    sim = flow.sim
    assert sim is not None and sim.level is not None and flow.player is not None
    line = sim.level.lines[0]
    line.special = 74
    line.args = (map_num, position, 0, 0, 0)
    assert sim.on_line_activated(line, flow.player.mo, SPAC_CROSS)


def _kill_and_press_use(flow: GameFlow) -> None:
    player = flow.player
    assert player is not None and player.mo is not None and flow.sim is not None
    damage_mobj(flow.sim, player.mo, None, None, 10000)
    assert player.player_state == PlayerState.PST_DEAD
    use = TicCmd(buttons=BT_USE)
    for _ in range(MAX_TICS):
        flow.tick([use])
        if flow.player is not player:
            return
    raise AssertionError("player was never reborn")


def test_menu_new_game_starts_first_map() -> None:
    flow = _build_flow()
    flow.start_title()
    assert flow.state == GameState.TITLE
    assert flow.sim is None

    menu = MenuManager()
    menu.on_new_game = flow.new_game
    menu.activate_menu()
    menu.handle_key(pygame.K_RETURN)  # NEW GAME
    assert menu.current_menu_type == MenuType.CLASS
    menu.handle_key(pygame.K_DOWN)  # CLERIC
    menu.handle_key(pygame.K_RETURN)
    menu.handle_key(pygame.K_RETURN)  # skill at the default cursor
    chosen_skill = menu.selected_skill

    # Like vanilla ga_newgame, the game starts on the next tic.
    assert flow.state == GameState.TITLE
    flow.tick()
    assert flow.state == GameState.LEVEL
    assert flow.sim is not None and flow.sim.map_num == 1
    assert flow.sim.skill == chosen_skill
    assert flow.player is not None and flow.player.player_class == 1
    assert flow.reborn_slot is None


def test_level_loaded_callback_and_events_are_drained() -> None:
    flow = _build_flow()
    loaded: list[tuple[int, str | None]] = []
    flow.on_level_loaded = lambda sim, info: loaded.append(
        (sim.map_num, info.lump_name if info else None)
    )
    _start_game(flow)
    assert loaded == [(1, "MAP01")]

    exit_normal(flow.sim)  # type: ignore[arg-type]
    events = flow.tick()
    assert any(type(e).__name__ == "MapExit" for e in events)
    assert flow.sim is not None and flow.sim.events.drain() == []


def test_exit_line_goes_through_intermission_to_next_map_and_keeps_hub_state() -> None:
    flow = _build_flow()
    _start_game(flow)
    sim1 = flow.sim
    assert sim1 is not None and sim1.level is not None
    player = flow.player
    assert player is not None

    # Leave a visible mark in MAP01 that only hub persistence can bring back.
    sim1.level.sectors[1].light_level = 37
    player.health = 61
    assert player.mo is not None
    player.mo.health = 61

    _exit_through_line(flow, 2)
    flow.tick()  # the MapExit event is seen at the end of this tic...
    flow.tick()  # ...and G_DoCompleted runs at the start of the next one
    assert flow.state == GameState.INTERMISSION
    assert flow.intermission.hub_text == ""  # same cluster: no hub text

    _tick_until(flow, GameState.LEVEL)
    sim2 = flow.sim
    assert sim2 is not None and sim2 is not sim1
    assert sim2.map_num == 2
    assert flow.hub.has_map(1)
    assert flow.player is player  # the same player carried over
    assert player.health == 61
    assert player.mo is not None
    assert (player.mo.x >> FRACBITS, player.mo.y >> FRACBITS) == (256, 256)
    assert flow.reborn_slot is not None and flow.reborn_slot.map_num == 2

    # Go back: MAP01 is restored from the hub snapshot, and the player is placed
    # on MAP01's start spot (player starts are collected on revisits too).
    teleport_new_map(sim2, 1, 0)
    _tick_until(flow, GameState.LEVEL)
    sim1_back = flow.sim
    assert sim1_back is not None and sim1_back.level is not None
    assert sim1_back.map_num == 1
    assert sim1_back.level.sectors[1].light_level == 37
    assert (player.mo.x >> FRACBITS, player.mo.y >> FRACBITS) == (128, 128)
    assert player in sim1_back.players


def test_exit_position_selects_start_spot_group() -> None:
    flow = _build_flow()
    _start_game(flow)
    _exit_through_line(flow, 2, position=1)
    _tick_until(flow, GameState.LEVEL)
    player = flow.player
    assert player is not None and player.mo is not None
    assert (player.mo.x >> FRACBITS, player.mo.y >> FRACBITS) == (400, 100)


def test_teleport_new_map_target_is_not_warp_translated() -> None:
    # MAP03 has warp number 2; Teleport_NewMap(2) must still go to MAP02.
    flow = _build_flow()
    _start_game(flow)
    _exit_through_line(flow, 2)
    _tick_until(flow, GameState.LEVEL)
    assert flow.sim is not None and flow.sim.map_num == 2


def test_exit_normal_uses_translated_next_map() -> None:
    # MAP01's next is warp 2, which is MAP03.
    flow = _build_flow()
    _start_game(flow)
    assert flow.sim is not None and flow.sim.next_map == 3
    exit_normal(flow.sim)
    _tick_until(flow, GameState.INTERMISSION)
    flow.intermission.handle_key(pygame.K_SPACE)  # MAP03 is a new cluster: skip the text
    _tick_until(flow, GameState.LEVEL)
    assert flow.sim.map_num == 3


def test_cluster_change_shows_hub_text_and_clears_hub() -> None:
    flow = _build_flow()
    _start_game(flow)
    _exit_through_line(flow, 2)
    _tick_until(flow, GameState.LEVEL)
    assert flow.hub.has_map(1)

    teleport_new_map(flow.sim, 3, 0)  # type: ignore[arg-type]
    flow.tick()
    flow.tick()
    assert flow.state == GameState.INTERMISSION
    assert flow.intermission.hub_text != ""  # leaving cluster 1 shows its message

    # A key press skips the text, as in vanilla CheckForSkip.
    flow.intermission.handle_key(pygame.K_SPACE)
    used = _tick_until(flow, GameState.LEVEL)
    assert used < 20
    assert flow.sim is not None and flow.sim.map_num == 3
    assert not flow.hub.has_map(1)
    assert flow.hub.current_cluster == 2


def test_death_without_reborn_slot_starts_a_new_game() -> None:
    flow = _build_flow()
    _start_game(flow, player_class=2, skill=3)
    first_sim = flow.sim
    _kill_and_press_use(flow)

    assert flow.state == GameState.LEVEL
    assert flow.sim is not first_sim
    assert flow.sim is not None and flow.sim.map_num == 1
    assert flow.sim.skill == 3
    player = flow.player
    assert player is not None
    assert player.player_state == PlayerState.PST_LIVE
    assert player.player_class == 2
    assert player.health == 100


def test_death_after_map_change_reloads_the_reborn_slot() -> None:
    flow = _build_flow()
    _start_game(flow)
    assert flow.sim is not None and flow.sim.level is not None
    flow.sim.level.sectors[1].light_level = 37
    _exit_through_line(flow, 2)
    _tick_until(flow, GameState.LEVEL)
    entry_health = flow.player.health  # type: ignore[union-attr]

    # Change MAP02 after the reborn save; dying must undo it.
    assert flow.sim.level is not None
    flow.sim.level.sectors[0].light_level = 99
    _kill_and_press_use(flow)

    assert flow.state == GameState.LEVEL
    sim = flow.sim
    assert sim is not None and sim.level is not None
    assert sim.map_num == 2
    assert sim.level.sectors[0].light_level != 99
    player = flow.player
    assert player is not None and player.player_state == PlayerState.PST_LIVE
    assert player.health == entry_health
    assert player.mo is not None and player.mo.player is player

    # The hub snapshot of MAP01 came back with the reborn slot.
    assert flow.hub.has_map(1)
    teleport_new_map(sim, 1, 0)
    _tick_until(flow, GameState.LEVEL)
    assert flow.sim.level is not None
    assert flow.sim.level.sectors[1].light_level == 37


def test_victory_runs_finale_then_returns_to_title() -> None:
    flow = _build_flow()
    _start_game(flow, player_class=1)
    teleport_end_game(flow.sim)  # type: ignore[arg-type]
    _tick_until(flow, GameState.FINALE)
    assert flow.finale.active
    assert flow.finale.player_class == 1

    for _ in range(10):
        if flow.state != GameState.FINALE:
            break
        flow.finale.handle_key(pygame.K_SPACE)
    assert flow.state == GameState.TITLE
    assert flow.sim is None
    assert flow.title.active


def test_title_loop_skips_demo_pages() -> None:
    flow = _build_flow()
    flow.start_title()
    flow.title.sequence = TitleSequence.ADVISOR
    flow.title.advance()  # → DEMO_1, which asks for a demo
    assert flow.title.sequence == TitleSequence.DEMO_1
    flow.tick()
    assert flow.title.sequence == TitleSequence.CREDIT_1


def test_init_new_warps_straight_into_a_map() -> None:
    flow = _build_flow()
    flow.init_new(2, player_class=0, skill=4)
    assert flow.state == GameState.LEVEL
    assert flow.sim is not None and flow.sim.map_num == 2
    assert flow.sim.skill == 4
    assert flow.player is not None


@pytest.mark.parametrize("trigger", ["exit", "death"])
def test_demo_playback_ends_instead_of_changing_state(trigger: str) -> None:
    flow = _build_flow()
    _start_game(flow)
    flow.demo_playback = True
    sim = flow.sim
    if trigger == "exit":
        exit_normal(sim)  # type: ignore[arg-type]
        flow.tick()
        flow.tick()
    else:
        assert flow.player is not None and flow.player.mo is not None and sim is not None
        damage_mobj(sim, flow.player.mo, None, None, 10000)
        for _ in range(MAX_TICS):
            flow.tick([TicCmd(buttons=BT_USE)])
            if flow.demo_finished:
                break
    assert flow.demo_finished
    assert flow.sim is sim
    assert flow.state == GameState.LEVEL


def test_window_to_virtual_maps_letterboxed_viewport() -> None:
    from pyhexen.__main__ import window_to_virtual

    # 1280x720 window with a centred 960x720 4:3 viewport (GL origin bottom-left).
    box = (160, 0, 960, 720)
    assert window_to_virtual(160, 0, box, 720) == (0, 0)
    assert window_to_virtual(1119, 719, box, 720) == (319, 199)
    assert window_to_virtual(640, 360, box, 720) == (160, 100)
    # Pillarboxed vertically: 800x800 window, 800x600 viewport starting at y=100.
    assert window_to_virtual(0, 100, (0, 100, 800, 600), 800) == (0, 0)


def test_resolve_warp_accepts_numbers_and_lump_names() -> None:
    from pyhexen.__main__ import resolve_warp

    table = MapInfoTable({13: MapInfo(map_num=13, name="SHADOW WOOD", lump_name="MAP13")})
    assert resolve_warp("13", table) == 13
    assert resolve_warp("map13", table) == 13
    assert resolve_warp("MAP05", table) == 5
    with pytest.raises(SystemExit):
        resolve_warp("E1M1", table)
