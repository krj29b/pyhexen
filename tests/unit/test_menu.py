# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T12.4: Menus system (mn_menu.c)."""

from __future__ import annotations

import numpy as np
import pygame
import pytest

from pyhexen.app.config import Config, resolve_iwad
from pyhexen.app.input import KeyBindings
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.ui.fonts import FontSet
from pyhexen.ui.menu import (
    SFX_CHAT,
    SFX_HAMMER_HIT,
    SFX_PICKUP_KEY,
    SFX_PLATFORM_STOP,
    MenuManager,
    MenuType,
    PromptType,
)
from pyhexen.wad.wadfile import WadFile


def test_menu_initialization() -> None:
    config = Config()
    bindings = KeyBindings()
    menu_mgr = MenuManager(config=config, bindings=bindings)

    assert not menu_mgr.active
    assert menu_mgr.current_menu_type == MenuType.MAIN
    assert menu_mgr.prompt_type == PromptType.NONE
    assert menu_mgr.info_type == 0

    # Ensure all primary menus are registered
    for mtype in (
        MenuType.MAIN,
        MenuType.CLASS,
        MenuType.SKILL,
        MenuType.OPTIONS,
        MenuType.OPTIONS2,
        MenuType.FILES,
        MenuType.LOAD,
        MenuType.SAVE,
        MenuType.CONTROLS,
    ):
        assert mtype in menu_mgr.menus
        assert menu_mgr.menus[mtype].menu_type == mtype

    # Fallback patch sanity
    pic = menu_mgr.get_patch("M_HTIC")
    assert pic.width == 144
    assert pic.height == 52


def test_menu_activation_and_deactivation() -> None:
    sounds_played: list[str] = []
    menu_mgr = MenuManager()
    menu_mgr.on_sound = sounds_played.append

    menu_mgr.activate_menu()
    assert menu_mgr.active
    assert menu_mgr.current_menu_type == MenuType.MAIN
    assert SFX_PLATFORM_STOP in sounds_played

    menu_mgr.tick()
    assert menu_mgr.menu_time == 1

    menu_mgr.deactivate_menu()
    assert not menu_mgr.active
    assert sounds_played.count(SFX_PLATFORM_STOP) == 2


def test_menu_keyboard_navigation() -> None:
    sounds_played: list[str] = []
    menu_mgr = MenuManager()
    menu_mgr.on_sound = sounds_played.append
    menu_mgr.activate_menu()

    # Initial position: item 0 ("NEW GAME")
    assert menu_mgr.current_menu.cur_pos == 0

    # Down arrow moves to item 1 ("OPTIONS")
    handled = menu_mgr.handle_key(pygame.K_DOWN)
    assert handled
    assert menu_mgr.current_menu.cur_pos == 1
    assert sounds_played[-1] == SFX_HAMMER_HIT

    # Up arrow moves back to 0
    menu_mgr.handle_key(pygame.K_UP)
    assert menu_mgr.current_menu.cur_pos == 0

    # Jump by letter 'Q' -> "QUIT GAME" (item 4)
    menu_mgr.handle_key(pygame.K_q, char_typed="Q")
    assert menu_mgr.current_menu.cur_pos == 4

    # Jump by letter 'N' -> "NEW GAME" (item 0)
    menu_mgr.handle_key(pygame.K_n, char_typed="N")
    assert menu_mgr.current_menu.cur_pos == 0


def test_class_and_skill_selection() -> None:
    game_started: list[tuple[int, int]] = []
    menu_mgr = MenuManager()
    menu_mgr.on_new_game = lambda cls, skl: game_started.append((cls, skl))
    menu_mgr.activate_menu()

    # Enter on "NEW GAME" -> transitions to ClassMenu
    menu_mgr.handle_key(pygame.K_RETURN)
    assert menu_mgr.current_menu_type == MenuType.CLASS

    # Move cursor down to "CLERIC" (item 1)
    menu_mgr.handle_key(pygame.K_DOWN)
    assert menu_mgr.current_menu.cur_pos == 1

    # Enter on "CLERIC" -> transitions to SkillMenu
    menu_mgr.handle_key(pygame.K_RETURN)
    assert menu_mgr.current_menu_type == MenuType.SKILL
    assert menu_mgr.selected_class == 1

    # Verify Cleric skill names were set
    skill_items = [itm.text for itm in menu_mgr.current_menu.items]
    assert skill_items == ["ALTAR BOY", "ACOLYTE", "PRIEST", "CARDINAL", "POPE"]

    # Select "PRIEST" (item 2)
    menu_mgr.handle_key(pygame.K_RETURN)
    assert not menu_mgr.active
    assert game_started == [(1, 2)]


def test_options_messages_toggle_and_sliders() -> None:
    sounds_played: list[str] = []
    config = Config()
    config.controls.mouse_sensitivity = 0.5
    config.audio.sfx_volume = 0.5
    menu_mgr = MenuManager(config=config)
    menu_mgr.on_sound = sounds_played.append
    menu_mgr.activate_menu()

    # Navigate: MAIN -> OPTIONS
    menu_mgr.handle_key(pygame.K_DOWN)  # cur_pos = 1 (OPTIONS)
    menu_mgr.handle_key(pygame.K_RETURN)
    assert menu_mgr.current_menu_type == MenuType.OPTIONS

    # Item 1: MESSAGES toggle
    menu_mgr.handle_key(pygame.K_DOWN)  # pos 1
    assert menu_mgr.messages_on
    menu_mgr.handle_key(pygame.K_RETURN)
    assert not menu_mgr.messages_on
    assert sounds_played[-1] == SFX_CHAT

    # Item 2: MOUSE SENSITIVITY slider
    menu_mgr.handle_key(pygame.K_DOWN)  # pos 2
    cur_val = menu_mgr.current_menu.items[2].value_getter()
    menu_mgr.handle_key(pygame.K_RIGHT)
    assert menu_mgr.current_menu.items[2].value_getter() == cur_val + 1
    assert sounds_played[-1] == SFX_PICKUP_KEY

    menu_mgr.handle_key(pygame.K_LEFT)
    assert menu_mgr.current_menu.items[2].value_getter() == cur_val

    # Navigate to MORE... (OPTIONS2)
    menu_mgr.handle_key(pygame.K_DOWN)  # skips EMPTY item 3 -> pos 4 (MORE...)
    assert menu_mgr.current_menu.cur_pos == 4
    menu_mgr.handle_key(pygame.K_RETURN)
    assert menu_mgr.current_menu_type == MenuType.OPTIONS2

    # In OPTIONS2: SFX Volume is at pos 2
    menu_mgr.handle_key(pygame.K_DOWN)  # skips empty pos 1 -> pos 2 (SFX VOLUME)
    assert menu_mgr.current_menu.cur_pos == 2
    vol_before = menu_mgr.current_menu.items[2].value_getter()
    menu_mgr.handle_key(pygame.K_RIGHT)
    assert menu_mgr.current_menu.items[2].value_getter() == min(15, vol_before + 1)

    # Escape returns to OPTIONS
    menu_mgr.handle_key(pygame.K_ESCAPE)
    assert menu_mgr.current_menu_type == MenuType.OPTIONS


def test_key_controls_rebinding() -> None:
    bindings = KeyBindings(forward=pygame.K_w)
    menu_mgr = MenuManager(bindings=bindings)
    menu_mgr.activate_menu()

    # Go to OPTIONS
    menu_mgr.handle_key(pygame.K_DOWN)
    menu_mgr.handle_key(pygame.K_RETURN)

    # Go to CONTROLS
    menu_mgr.set_menu(MenuType.CONTROLS)
    assert menu_mgr.current_menu_type == MenuType.CONTROLS
    assert menu_mgr.current_menu.cur_pos == 0
    assert menu_mgr.get_action_key_name("FORWARD") == "W"

    # Select FORWARD to rebind
    menu_mgr.handle_key(pygame.K_RETURN)
    assert menu_mgr.rebinding_action == "FORWARD"

    # Press Up arrow to rebind FORWARD
    menu_mgr.handle_key(pygame.K_UP)
    assert menu_mgr.rebinding_action is None
    assert bindings.forward == pygame.K_UP
    assert menu_mgr.get_action_key_name("FORWARD") == "UP"


def test_save_and_load_slots() -> None:
    saved_games: list[tuple[int, str]] = []
    loaded_games: list[int] = []

    menu_mgr = MenuManager()
    menu_mgr.on_save_game = lambda slot, desc: saved_games.append((slot, desc))
    menu_mgr.on_load_game = lambda slot: loaded_games.append(slot)
    menu_mgr.activate_menu()

    # Navigate to SAVE menu
    menu_mgr.set_menu(MenuType.SAVE)
    assert menu_mgr.current_menu_type == MenuType.SAVE

    # Enter editing for slot 0
    menu_mgr.handle_key(pygame.K_RETURN)
    assert menu_mgr.editing_slot
    assert menu_mgr.current_slot == 0

    # Type "MY SAVE 1"
    for char in "MY SAVE 1":
        menu_mgr.handle_key(pygame.K_UNKNOWN, char_typed=char)

    assert menu_mgr.slot_texts[0] == "MY SAVE 1"

    # Backspace once and add '2'
    menu_mgr.handle_key(pygame.K_BACKSPACE)
    menu_mgr.handle_key(pygame.K_UNKNOWN, char_typed="2")
    assert menu_mgr.slot_texts[0] == "MY SAVE 2"

    # Press Enter to confirm save
    menu_mgr.handle_key(pygame.K_RETURN)
    assert not menu_mgr.editing_slot
    assert not menu_mgr.active
    assert menu_mgr.slot_status[0]
    assert saved_games == [(0, "MY SAVE 2")]
    assert menu_mgr.quicksave_slot == 0

    # Now open LOAD menu
    menu_mgr.activate_menu()
    menu_mgr.set_menu(MenuType.LOAD)
    assert menu_mgr.current_menu_type == MenuType.LOAD

    # Load slot 0
    menu_mgr.handle_key(pygame.K_RETURN)
    assert not menu_mgr.active
    assert loaded_games == [0]
    assert menu_mgr.quickload_slot == 0


def test_confirmation_prompts() -> None:
    quits: int = 0
    ends: int = 0
    suicides: int = 0

    menu_mgr = MenuManager()
    menu_mgr.on_quit_game = lambda: quits_increment()
    menu_mgr.on_end_game = lambda: ends_increment()
    menu_mgr.on_suicide = lambda: suicides_increment()

    def quits_increment() -> None:
        nonlocal quits
        quits += 1

    def ends_increment() -> None:
        nonlocal ends
        ends += 1

    def suicides_increment() -> None:
        nonlocal suicides
        suicides += 1

    # F10: Quit prompt
    menu_mgr.handle_key(pygame.K_F10)
    assert menu_mgr.prompt_type == PromptType.QUIT

    # 'N' cancels
    menu_mgr.handle_key(pygame.K_n)
    assert menu_mgr.prompt_type == PromptType.NONE
    assert quits == 0

    # F10 -> 'Y' confirms
    menu_mgr.handle_key(pygame.K_F10)
    menu_mgr.handle_key(pygame.K_y)
    assert menu_mgr.prompt_type == PromptType.NONE
    assert quits == 1

    # F5: Suicide prompt
    menu_mgr.handle_key(pygame.K_F5)
    assert menu_mgr.prompt_type == PromptType.SUICIDE
    menu_mgr.handle_key(pygame.K_RETURN)
    assert suicides == 1

    # F7: End game prompt
    menu_mgr.handle_key(pygame.K_F7)
    assert menu_mgr.prompt_type == PromptType.END_GAME
    menu_mgr.handle_key(pygame.K_y)
    assert ends == 1


def test_mouse_navigation_and_sliders() -> None:
    menu_mgr = MenuManager()
    menu_mgr.activate_menu()
    menu_mgr.set_menu(MenuType.OPTIONS)

    # Hovering over item 2 (y in [30 + 2*20, 30 + 3*20] = [70, 90])
    handled = menu_mgr.handle_mouse_move(120, 75)
    assert handled
    assert menu_mgr.current_menu.cur_pos == 2

    # Left click on slider inside slider bar: x = menu.x + 24 = 88 + 24 = 112
    # Width = 10, clicking at x = 112 + 4 * 8 = 144 -> slot 4
    item = menu_mgr.current_menu.items[2]
    menu_mgr.handle_mouse_click(144, 75, button=1)
    assert item.value_getter() == 4

    # Mouse wheel down adjusts slider left
    menu_mgr.handle_mouse_click(144, 75, button=5)
    assert item.value_getter() == 3

    # Mouse wheel up adjusts slider right
    menu_mgr.handle_mouse_click(144, 75, button=4)
    assert item.value_getter() == 4

    # Right click backs out to MAIN menu
    menu_mgr.handle_mouse_click(144, 75, button=3)
    assert menu_mgr.current_menu_type == MenuType.MAIN


def test_menu_overlay_draw() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    menu_mgr = MenuManager()

    # Draw inactive: nothing drawn
    overlay.clear()
    menu_mgr.draw(overlay)
    assert not np.any(overlay.mask)

    # Draw Main menu
    menu_mgr.activate_menu()
    overlay.clear()
    menu_mgr.draw(overlay)
    assert np.any(overlay.mask)

    # Draw Class menu
    menu_mgr.set_menu(MenuType.CLASS)
    overlay.clear()
    menu_mgr.draw(overlay)
    assert np.any(overlay.mask)

    # Draw Skill menu
    menu_mgr.set_menu(MenuType.SKILL)
    overlay.clear()
    menu_mgr.draw(overlay)
    assert np.any(overlay.mask)

    # Draw Options menu
    menu_mgr.set_menu(MenuType.OPTIONS)
    overlay.clear()
    menu_mgr.draw(overlay)
    assert np.any(overlay.mask)

    # Draw Save menu with slot editing
    menu_mgr.set_menu(MenuType.SAVE)
    menu_mgr.current_slot = 0
    menu_mgr.editing_slot = True
    menu_mgr.slot_texts[0] = "MAP01"
    overlay.clear()
    menu_mgr.draw(overlay)
    assert np.any(overlay.mask)

    # Draw Controls menu
    menu_mgr.set_menu(MenuType.CONTROLS)
    overlay.clear()
    menu_mgr.draw(overlay)
    assert np.any(overlay.mask)

    # Draw Confirmation Prompt
    menu_mgr.prompt_type = PromptType.QUIT
    overlay.clear()
    menu_mgr.draw(overlay)
    assert np.any(overlay.mask)

    # Draw Info Screen
    menu_mgr.prompt_type = PromptType.NONE
    menu_mgr.info_type = 1
    overlay.clear()
    menu_mgr.draw(overlay)
    assert np.any(overlay.mask)


def test_global_function_shortcuts() -> None:
    menu_mgr = MenuManager()

    # F1 -> Info
    menu_mgr.handle_key(pygame.K_F1)
    assert menu_mgr.active
    assert menu_mgr.info_type == 1

    # Esc closes info
    menu_mgr.handle_key(pygame.K_ESCAPE)
    assert not menu_mgr.active
    assert menu_mgr.info_type == 0

    # F2 -> Save Menu
    menu_mgr.handle_key(pygame.K_F2)
    assert menu_mgr.active
    assert menu_mgr.current_menu_type == MenuType.SAVE
    menu_mgr.deactivate_menu()

    # F3 -> Load Menu
    menu_mgr.handle_key(pygame.K_F3)
    assert menu_mgr.active
    assert menu_mgr.current_menu_type == MenuType.LOAD
    menu_mgr.deactivate_menu()

    # F4 -> Volume (Options2)
    menu_mgr.handle_key(pygame.K_F4)
    assert menu_mgr.active
    assert menu_mgr.current_menu_type == MenuType.OPTIONS2
    menu_mgr.deactivate_menu()

    # F8 -> Toggle messages
    assert menu_mgr.messages_on
    menu_mgr.handle_key(pygame.K_F8)
    assert not menu_mgr.messages_on


@pytest.mark.wad
def test_real_iwad_menu_lumps() -> None:
    iwad_path = resolve_iwad()
    if not iwad_path or not iwad_path.is_file():
        pytest.skip("No real IWAD available")

    wad = WadFile(iwad_path)
    menu_mgr = MenuManager(wad=wad)

    # Verify real graphics decode without error
    for lump_name in (
        "M_HTIC",
        "M_SLCTR1",
        "M_SLCTR2",
        "M_SLDLT",
        "M_SLDMD1",
        "M_SLDMD2",
        "M_SLDRT",
        "M_SLDKB",
        "M_FSLOT",
        "FBULA0",
        "M_FBOX",
        "M_CBOX",
        "M_MBOX",
        "M_FWALK1",
        "M_CWALK1",
        "M_MWALK1",
    ):
        pic = menu_mgr.get_patch(lump_name)
        assert pic is not None
        assert pic.width > 0
        assert pic.height > 0
