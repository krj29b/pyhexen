# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for the T15.3 UI wiring (app/game_ui.py): automap, messages, save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.app.game_flow import TXT_GAMESAVED, GameFlow, GameState
from pyhexen.app.game_ui import GameUI
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.sim.message import set_message
from pyhexen.sim.savegame import SaveGameManager
from pyhexen.sim.transitions import exit_normal
from pyhexen.ui.fonts import FontSet
from pyhexen.ui.menu import MenuManager, MenuType
from pyhexen.ui.statusbar import StatusBar
from pyhexen.wad.mapinfo import MapInfo, MapInfoTable
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import ML_MAPPED

MAX_TICS = 200


def _build_wad() -> WadStack:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb1 = MapBuilder("MAP01")
    mb1.add_two_sector_rooms()
    mb1.add_thing(x=128, y=128, type=1, angle=0)  # facing east, toward room 2
    wb.add_map(mb1)
    mb2 = MapBuilder("MAP02")
    mb2.add_room(x=0, y=0, width=512, height=512)
    mb2.add_thing(x=256, y=256, type=1)
    wb.add_map(mb2)
    return WadStack(wb.to_bytes())


MAPINFO = MapInfoTable(
    {
        1: MapInfo(map_num=1, name="WINNOWING HALL", lump_name="MAP01", cluster=1, next_map=2),
        2: MapInfo(map_num=2, name="SEVEN PORTALS", lump_name="MAP02", cluster=1, next_map=1),
    }
)


def _setup(tmp_path: Path | None = None) -> tuple[GameFlow, GameUI, MenuManager]:
    wad = _build_wad()
    saves = SaveGameManager(tmp_path) if tmp_path is not None else None
    flow = GameFlow(wad, MAPINFO, saves=saves)
    menu = MenuManager()
    statusbar = StatusBar()
    ui = GameUI(wad, flow, menu, saves=saves, statusbar=statusbar)
    flow.on_level_loaded = ui.on_level_loaded
    return flow, ui, menu


def _start(flow: GameFlow) -> None:
    flow.start_title()
    flow.new_game(0, 2)
    flow.tick()
    assert flow.state == GameState.LEVEL


def _tick_until(flow: GameFlow, ui: GameUI, state: GameState) -> None:
    for _ in range(MAX_TICS):
        flow.tick()
        ui.tick()
        if flow.state == state and flow.pending_action == 0:
            return
    raise AssertionError(f"never reached {state!r}")


# ----------------------------------------------------------------------
# Automap
# ----------------------------------------------------------------------


def test_tab_toggles_automap_in_a_level_only() -> None:
    flow, ui, _menu = _setup()
    flow.start_title()
    assert not ui.handle_key_down(pygame.K_TAB)
    assert not ui.automap_active

    _start(flow)
    assert ui.map_name == "WINNOWING HALL"
    assert ui.handle_key_down(pygame.K_TAB)
    assert ui.automap_active and ui.view_hidden
    assert ui.statusbar is not None and ui.statusbar.automap_active

    # Follow mode is on, so arrows go to player movement (AM_Responder returns false).
    assert not ui.handle_key_down(pygame.K_LEFT)
    assert not ui.handle_key_up(pygame.K_LEFT)

    # Zoom a little; the scale survives closing and reopening on the same map.
    assert ui.handle_key_down(pygame.K_EQUALS)
    flow.tick()
    ui.tick()
    ui.handle_key_up(pygame.K_EQUALS)
    scale = ui.automap.scale_mtof

    assert ui.handle_key_down(pygame.K_TAB)
    assert not ui.automap_active
    assert not ui.statusbar.automap_active

    ui.handle_key_down(pygame.K_TAB)
    assert ui.automap.scale_mtof == scale


def test_automap_follow_toggle_shows_player_message() -> None:
    flow, ui, _menu = _setup()
    _start(flow)
    ui.handle_key_down(pygame.K_TAB)
    assert ui.handle_key_down(pygame.K_f)
    player = flow.player
    assert player is not None
    assert player.message == "FOLLOW MODE OFF"
    assert player.ultimate_message

    # With follow off, a held arrow pans every tic.
    x0 = ui.automap.m_x
    assert ui.handle_key_down(pygame.K_RIGHT)
    flow.tick()
    ui.tick()
    assert ui.automap.m_x > x0


def test_automap_closes_when_the_level_changes() -> None:
    flow, ui, _menu = _setup()
    _start(flow)
    ui.handle_key_down(pygame.K_TAB)
    assert flow.sim is not None
    exit_normal(flow.sim)
    _tick_until(flow, ui, GameState.INTERMISSION)
    assert not ui.automap_active


def test_automap_draws_map_name_and_seen_lines() -> None:
    flow, ui, _menu = _setup()
    _start(flow)
    ui.handle_key_down(pygame.K_TAB)
    overlay = Overlay2D(fonts=FontSet.synthetic())
    overlay.clear()
    ui.draw_automap(overlay)
    assert np.any(overlay.mask[144:152, 38:120])  # map name at (38, 144)


def test_mark_seen_maps_visible_lines_only() -> None:
    flow, ui, _menu = _setup()
    _start(flow)
    sim = flow.sim
    assert sim is not None and sim.level is not None
    assert not any(line.flags & ML_MAPPED for line in sim.level.lines)

    marked = ui.mark_seen()
    assert marked > 0

    def line_at_x(x: int) -> list[bool]:
        fx = x << 16
        return [
            bool(line.flags & ML_MAPPED)
            for line in sim.level.lines
            if line.v1.x == fx and line.v2.x == fx
        ]

    # Facing east from (128, 128): the far east wall of room 2 is seen, the wall
    # behind the player (x = 0) is not.
    assert any(line_at_x(512))
    assert not any(line_at_x(0))

    # Nothing is marked while the parchment automap hides the view.
    for line in sim.level.lines:
        line.flags &= ~ML_MAPPED
    ui.handle_key_down(pygame.K_TAB)
    assert ui.mark_seen() == 0


# ----------------------------------------------------------------------
# Messages
# ----------------------------------------------------------------------


def test_player_message_is_drawn_and_cleared_by_intermission() -> None:
    flow, ui, _menu = _setup()
    _start(flow)
    sim, player = flow.sim, flow.player
    assert sim is not None and player is not None
    set_message(sim, player, "QUIETUS BLADE")

    overlay = Overlay2D(fonts=FontSet.synthetic())
    overlay.clear()
    ui.draw_message(overlay, player)
    assert np.any(overlay.mask[1:10, :])

    exit_normal(sim)
    _tick_until(flow, ui, GameState.INTERMISSION)
    assert player.message_tics == 0 and player.message == ""


def test_f8_toggles_messages_in_the_sim() -> None:
    flow, ui, _menu = _setup()
    _start(flow)
    sim, player = flow.sim, flow.player
    assert sim is not None and player is not None

    assert ui.handle_key_down(pygame.K_F8)
    assert not sim.messages_on
    assert player.message == "MESSAGES OFF"  # SCMessages: ultimate message

    # The setting carries over to the next map's sim.
    exit_normal(sim)
    _tick_until(flow, ui, GameState.INTERMISSION)
    flow.intermission.handle_key(pygame.K_SPACE)
    _tick_until(flow, ui, GameState.LEVEL)
    assert flow.sim is not None and flow.sim is not sim
    assert not flow.sim.messages_on


# ----------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------


def _save_from_menu(ui: GameUI, menu: MenuManager, slot: int, name: str) -> None:
    assert ui.handle_key_down(pygame.K_F2)
    assert menu.current_menu_type == MenuType.SAVE
    for _ in range(slot):
        menu.handle_key(pygame.K_DOWN)
    menu.handle_key(pygame.K_RETURN)
    assert menu.editing_slot
    for ch in name:
        menu.handle_key(0, ch)
    menu.handle_key(pygame.K_RETURN)
    assert not menu.active


def _load_from_menu(ui: GameUI, menu: MenuManager, slot: int) -> None:
    menu.handle_key(pygame.K_F3) if not ui.in_level else ui.handle_key_down(pygame.K_F3)
    assert menu.current_menu_type == MenuType.LOAD
    for _ in range(slot):
        menu.handle_key(pygame.K_DOWN)
    menu.handle_key(pygame.K_RETURN)
    assert not menu.active


def test_save_and_load_from_the_menu(tmp_path: Path) -> None:
    flow, ui, menu = _setup(tmp_path)
    _start(flow)
    player = flow.player
    assert player is not None and player.mo is not None and flow.saves is not None

    _save_from_menu(ui, menu, 2, "HALL")
    assert not flow.saves.get_slot_path(2).exists()  # deferred to the next tic (ga_savegame)
    flow.tick()
    assert flow.saves.get_slot_path(2).exists()
    assert player.message == TXT_GAMESAVED and player.ultimate_message
    saved_x, saved_y = player.mo.x, player.mo.y

    # Change the game, then load the save back.
    player.mo.x += 64 << 16
    player.health = 5
    _load_from_menu(ui, menu, 2)
    flow.tick()
    assert flow.state == GameState.LEVEL
    restored = flow.player
    assert restored is not None and restored.mo is not None
    assert restored is not player
    assert (restored.mo.x, restored.mo.y) == (saved_x, saved_y)
    assert restored.health != 5
    # SV_UpdateRebornSlot: dying now reloads the loaded game, not a new game.
    assert flow.reborn_slot is not None and flow.reborn_slot.map_num == 1

    # A fresh UI (next launch) shows the slot description.
    flow2, ui2, menu2 = _setup(tmp_path)
    assert menu2.slot_texts[2] == "HALL" and menu2.slot_status[2]


def test_load_from_title_screen_starts_the_level(tmp_path: Path) -> None:
    flow, ui, menu = _setup(tmp_path)
    _start(flow)
    _save_from_menu(ui, menu, 0, "A")
    flow.tick()

    flow.start_title()
    ui.handle_title_key(pygame.K_F3)  # F3 works on the title screen
    assert menu.active and menu.current_menu_type == MenuType.LOAD
    menu.handle_key(pygame.K_RETURN)
    flow.tick()
    assert flow.state == GameState.LEVEL and flow.sim is not None


def test_save_keys_do_nothing_outside_a_level(tmp_path: Path) -> None:
    flow, ui, menu = _setup(tmp_path)
    flow.start_title()
    ui.handle_title_key(pygame.K_F2)  # just opens the main menu, like any other key
    assert menu.active and menu.current_menu_type == MenuType.MAIN
    menu.deactivate_menu()

    # Saving through the Files menu with no game running writes nothing.
    menu.on_save_game(1, "NOTHING")  # type: ignore[misc]
    flow.tick()
    assert flow.saves is not None and not flow.saves.get_slot_path(1).exists()
    assert not menu.slot_status[1]


def test_bad_save_file_reports_instead_of_crashing(tmp_path: Path) -> None:
    flow, ui, menu = _setup(tmp_path)
    _start(flow)
    assert flow.saves is not None
    flow.saves.get_slot_path(3).write_bytes(b"not a save")
    menu.slot_status[3] = True
    sim = flow.sim
    _load_from_menu(ui, menu, 3)
    flow.tick()
    assert flow.sim is sim  # still playing the same level
    assert flow.last_load_error
    assert flow.player is not None and flow.player.message == "CAN'T LOAD THAT GAME"
