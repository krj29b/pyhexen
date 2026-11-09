# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T12.6: Intermission, finale, and title loop (in_lude.c, f_finale.c, h2_main.c)."""

from __future__ import annotations

import numpy as np
import pygame
import pytest

from pyhexen.app.config import resolve_iwad
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.ui.fonts import FontSet
from pyhexen.ui.intermission import (
    TEXTSPEED,
    Finale,
    FinaleStage,
    Intermission,
    TitleLoop,
    TitleSequence,
)
from pyhexen.wad.wadfile import WadFile


def test_intermission_start_and_text() -> None:
    music_tracks: list[str] = []
    inter = Intermission()
    inter.on_music = music_tracks.append

    # Transition from Cluster 1 to Cluster 2
    inter.start(old_cluster=1, new_cluster=2)
    assert inter.active
    assert "SEVEN PORTALS" in inter.hub_text
    assert inter.hub_count > 0
    assert music_tracks == ["hub"]

    # Same cluster -> no narrative text
    inter.start(old_cluster=1, new_cluster=1)
    assert inter.hub_text == ""
    assert inter.hub_count == 0


def test_intermission_typewriter_and_skip() -> None:
    done_called = False
    inter = Intermission()
    inter.on_done = lambda: set_done()

    def set_done() -> None:
        nonlocal done_called
        done_called = True

    inter.start(old_cluster=2, new_cluster=3)

    # Initial tick
    inter.tick()
    assert inter.intertime == 1

    # Advance 10 + 3*5 tics -> 5 characters typed
    for _ in range(10 + 5 * TEXTSPEED):
        inter.tick()

    chars_shown = (inter.intertime - 10) // TEXTSPEED
    assert chars_shown >= 5

    # Press key to skip
    inter.handle_key(pygame.K_SPACE)
    assert inter.skip_intermission

    # Ticking through interstate countdown (10 tics) completes intermission
    for _ in range(15):
        if not inter.active:
            break
        inter.tick()

    assert not inter.active
    assert done_called


def test_intermission_deathmatch_frags() -> None:
    inter = Intermission()
    inter.start(old_cluster=1, new_cluster=2, is_deathmatch=True)
    assert inter.is_deathmatch
    assert inter.hub_text == ""

    overlay = Overlay2D(FontSet.synthetic())
    overlay.clear()
    inter.draw(overlay)
    assert np.any(overlay.mask)


def test_intermission_render_draw() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    inter = Intermission()
    inter.start(old_cluster=1, new_cluster=2)

    # Inactive: draws nothing
    inter.active = False
    overlay.clear()
    inter.draw(overlay)
    assert not np.any(overlay.mask)

    # Active: draws background and text
    inter.active = True
    inter.intertime = 60  # some characters visible
    overlay.clear()
    inter.draw(overlay)
    assert np.any(overlay.mask)


def test_finale_stages_progression() -> None:
    music_tracks: list[str] = []
    done_called = False
    finale = Finale()
    finale.on_music = music_tracks.append
    finale.on_done = lambda: set_done()

    def set_done() -> None:
        nonlocal done_called
        done_called = True

    finale.start(player_class=1)  # Cleric
    assert finale.active
    assert finale.stage == FinaleStage.STAGE_0_FADE_PIC1
    assert music_tracks == ["hall"]

    # Stage 0: fade in (70 tics)
    for _ in range(70):
        finale.tick()
    assert finale.stage == FinaleStage.STAGE_1_TEXT1

    # Stage 1: Text 1 (advance by handle_key)
    finale.handle_key(pygame.K_RETURN)
    assert finale.stage == FinaleStage.STAGE_2_PIC2_TEXT2
    assert "orb" in music_tracks

    # Stage 2 -> Stage 3 (Fade out Pic 2)
    finale.handle_mouse_click()
    assert finale.stage == FinaleStage.STAGE_3_FADE_OUT_PIC2

    # Stage 3 -> Stage 4 (Fade in Chess)
    finale.handle_key(pygame.K_SPACE)
    assert finale.stage == FinaleStage.STAGE_4_FADE_IN_CHESS
    assert "chess" in music_tracks

    # Stage 4 -> Stage 5 (Text 3 Epilogue)
    finale.handle_key(pygame.K_SPACE)
    assert finale.stage == FinaleStage.STAGE_5_TEXT3_EPILOGUE
    assert "SHADOW IS VANQUISHED" in finale.current_text  # Cleric text

    # Stage 5 -> completion
    finale.handle_key(pygame.K_SPACE)
    assert not finale.active
    assert done_called


def test_finale_class_portraits_and_netgame() -> None:
    overlay = Overlay2D(FontSet.synthetic())

    # 1. Fighter single-player
    f_fighter = Finale()
    f_fighter.start(player_class=0, is_netgame=False)
    f_fighter.stage = FinaleStage.STAGE_5_TEXT3_EPILOGUE
    overlay.clear()
    f_fighter.draw(overlay)
    assert np.any(overlay.mask)

    # 2. Cleric single-player (draws chessc)
    f_cleric = Finale()
    f_cleric.start(player_class=1, is_netgame=False)
    f_cleric.stage = FinaleStage.STAGE_5_TEXT3_EPILOGUE
    overlay.clear()
    f_cleric.draw(overlay)
    assert np.any(overlay.mask)

    # 3. Netgame (draws chessall)
    f_net = Finale()
    f_net.start(player_class=0, is_netgame=True)
    f_net.stage = FinaleStage.STAGE_5_TEXT3_EPILOGUE
    overlay.clear()
    f_net.draw(overlay)
    assert np.any(overlay.mask)


def test_title_loop_sequences_and_timing() -> None:
    music_tracks: list[str] = []
    demos_played: list[str] = []
    menu_activated = False

    loop = TitleLoop()
    loop.on_music = music_tracks.append
    loop.on_play_demo = demos_played.append
    loop.on_menu_activate = lambda: set_menu()

    def set_menu() -> None:
        nonlocal menu_activated
        menu_activated = True

    loop.start()
    assert loop.active
    assert loop.sequence == TitleSequence.TITLE_1
    assert music_tracks == ["hexen"]

    # Advance after 280 tics to ADVISOR
    for _ in range(280):
        loop.tick()
    assert loop.sequence == TitleSequence.ADVISOR

    # Advance after 210 tics to DEMO_1
    for _ in range(210):
        loop.tick()
    assert loop.sequence == TitleSequence.DEMO_1
    assert demos_played == ["demo1"]

    # Key press advances through remaining sequences
    loop.handle_key(pygame.K_SPACE)
    assert loop.sequence == TitleSequence.CREDIT_1

    loop.handle_key(pygame.K_SPACE)
    assert loop.sequence == TitleSequence.DEMO_2
    assert demos_played == ["demo1", "demo2"]

    loop.handle_mouse_click()
    assert loop.sequence == TitleSequence.CREDIT_2

    loop.handle_mouse_click()
    assert loop.sequence == TitleSequence.DEMO_3
    assert demos_played == ["demo1", "demo2", "demo3"]

    loop.handle_key(pygame.K_SPACE)
    assert loop.sequence == TitleSequence.TITLE_1

    # Escape activates menu
    loop.handle_key(pygame.K_ESCAPE)
    assert menu_activated


def test_title_loop_render_draw() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    loop = TitleLoop()
    loop.start()

    # Draw TITLE_1
    overlay.clear()
    loop.draw(overlay)
    assert np.any(overlay.mask)

    # Draw ADVISOR
    loop.sequence = TitleSequence.ADVISOR
    overlay.clear()
    loop.draw(overlay)
    assert np.any(overlay.mask)

    # Draw CREDIT_1
    loop.sequence = TitleSequence.CREDIT_1
    overlay.clear()
    loop.draw(overlay)
    assert np.any(overlay.mask)


@pytest.mark.wad
def test_real_iwad_intermission_and_finale_lumps() -> None:
    iwad_path = resolve_iwad()
    if not iwad_path or not iwad_path.is_file():
        pytest.skip("No real IWAD available")

    wad = WadFile(iwad_path)

    # Verify raw screens
    for raw_lump in ("TITLE", "CREDIT", "INTERPIC", "FINALE1", "FINALE2", "FINALE3"):
        data = wad.get_bytes(raw_lump)
        assert len(data) == 64000

    # Verify patches
    for patch_lump in ("ADVISOR", "CHESSC", "CHESSM", "CHESSALL"):
        entry = wad.find(patch_lump)
        assert entry is not None
        assert entry.size > 0

    # Test components with real wad
    inter = Intermission(wad=wad)
    assert len(inter._interpic) == 64000

    finale = Finale(wad=wad)
    assert len(finale._finale1) == 64000
    assert len(finale._finale2) == 64000
    assert len(finale._finale3) == 64000
    assert finale._get_patch("chessc").width > 0

    loop = TitleLoop(wad=wad)
    assert len(loop._title_pic) == 64000
    assert len(loop._credit_pic) == 64000
    assert loop._advisor_patch.width > 0
