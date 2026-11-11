# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""In-game UI wiring: automap, HUD messages and the save/load menu (T15.3).

Connects `ui/automap.py`, `ui/messages.py` and the menu's save/load callbacks to the
running game (`GameFlow`). The main loop forwards in-level key events and calls
`tick`, `mark_seen` and `draw`; everything here can run without a window.

Event order follows H2_ProcessEvents / G_Responder: the menu responder first (Escape
and the F-keys), then the automap responder (AM_Responder), then player input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from pyhexen.app.game_flow import GameState
from pyhexen.render.line_visibility import LineVisibility
from pyhexen.sim.message import set_message
from pyhexen.sim.mobj import Mobj
from pyhexen.ui.automap import AutoMap
from pyhexen.ui.messages import draw_message

if TYPE_CHECKING:
    from pyhexen.app.game_flow import GameFlow
    from pyhexen.app.savegame import SaveGameManager
    from pyhexen.render.overlay2d import Overlay2D
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.sim.player import Player
    from pyhexen.ui.menu import MenuManager
    from pyhexen.ui.statusbar import StatusBar
    from pyhexen.wad.mapinfo import MapInfo
    from pyhexen.wad.wadfile import WadFile, WadStack

KEY_MAP_TOGGLE: int = pygame.K_TAB
# Save keys only work inside a level (MN_Responder checks gamestate == GS_LEVEL).
SAVE_KEYS: frozenset[int] = frozenset({pygame.K_F2, pygame.K_F6})


class GameUI:
    """Automap, HUD message and save/load wiring around a `GameFlow`."""

    def __init__(
        self,
        wad: WadStack | WadFile | None,
        flow: GameFlow,
        menu: MenuManager,
        *,
        saves: SaveGameManager | None = None,
        statusbar: StatusBar | None = None,
    ) -> None:
        self.flow = flow
        self.menu = menu
        self.saves = saves
        self.statusbar = statusbar
        self.automap = AutoMap(wad)
        self.automap.on_message = self.show_message
        self.line_vis = LineVisibility()
        self.map_name: str = ""

        menu.on_save_game = self._on_menu_save
        menu.on_load_game = self._on_menu_load
        menu.on_message = self.show_message
        menu.on_messages_toggled = flow.set_messages_on
        flow.on_game_saved = lambda _slot: self.refresh_save_slots()
        flow.set_messages_on(menu.messages_on)
        self.refresh_save_slots()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def in_level(self) -> bool:
        return self.flow.state == GameState.LEVEL and self.flow.player is not None

    @property
    def automap_active(self) -> bool:
        return self.automap.active

    @property
    def view_hidden(self) -> bool:
        """True when the parchment automap covers the 3D view (viewactive = false)."""
        return self.automap.active and not self.automap.overlay_mode

    def on_level_loaded(self, sim: GameSim, info: MapInfo | None) -> None:
        """A new level became current: close the automap and pick up the map name."""
        self.automap.stop()
        level_name = sim.level.name if sim.level is not None else ""
        self.map_name = info.name if info is not None and info.name else level_name
        self._sync_statusbar()

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def show_message(self, text: str, ultimate: bool = True) -> None:
        """P_SetMessage on the console player, if there is one."""
        sim = self.flow.sim
        player = self.flow.player
        if sim is not None and player is not None and self.flow.state == GameState.LEVEL:
            set_message(sim, player, text, ultimate)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def refresh_save_slots(self) -> None:
        """Reload slot descriptions from disk (vanilla `slottextloaded = false`)."""
        if self.saves is None:
            return
        for info in self.saves.list_slots():
            self.menu.slot_texts[info.slot] = info.description
            self.menu.slot_status[info.slot] = info.occupied

    def _on_menu_save(self, slot: int, description: str) -> None:
        if not self.in_level or self.saves is None:
            # Nothing to save: put back what is really on disk.
            self.refresh_save_slots()
            return
        self.flow.save_game(slot, description)

    def _on_menu_load(self, slot: int) -> None:
        if self.saves is None:
            return
        self.flow.load_game(slot)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def handle_key_down(self, key: int, char: str | None = None) -> bool:
        """Route a key press in a level; True when the menu or the automap ate it."""
        if key not in SAVE_KEYS or self.in_level:
            if self.menu.handle_key(key, char):
                return True
        if not self.in_level:
            return False
        if self.automap.active:
            eaten = self.automap.handle_key(key)
            self._sync_statusbar()
            return eaten
        if key == KEY_MAP_TOGGLE:
            self._start_automap()
            return True
        return False

    def handle_key_up(self, key: int) -> bool:
        """Route a key release; the automap never eats releases, so input sees them."""
        return self.automap.handle_key_up(key)

    def handle_title_key(self, key: int, char: str | None = None) -> None:
        """A key on the title screen: F-keys work, anything else opens the menu."""
        if key not in SAVE_KEYS and self.menu.handle_key(key, char):
            return
        self.menu.activate_menu()

    def handle_mouse_wheel(self, direction: int) -> bool:
        if not self.automap.active:
            return False
        self.automap.handle_mouse_wheel(direction)
        return True

    def _start_automap(self) -> None:
        sim = self.flow.sim
        player = self.flow.player
        if sim is None or sim.level is None or player is None:
            return
        self.automap.start(sim.level, player)
        self._sync_statusbar()

    # ------------------------------------------------------------------
    # Per tic
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Run once per game tic after `GameFlow.tick` (AM_Ticker)."""
        player = self.flow.player
        if not self.in_level or player is None:
            self.automap.stop()
        else:
            self.automap.tick(player)
        self._sync_statusbar()

    def mark_seen(self) -> int:
        """Mark the lines visible from the player's view as ML_MAPPED.

        Vanilla does this while rendering the 3D view, so nothing is marked while the
        parchment automap hides it.
        """
        sim = self.flow.sim
        player = self.flow.player
        if not self.in_level or self.view_hidden or sim is None or sim.level is None:
            return 0
        assert player is not None
        mo = player.mo
        if mo is None:
            return 0
        return self.line_vis.mark(sim.level, mo.x, mo.y, mo.angle)

    def _sync_statusbar(self) -> None:
        if self.statusbar is not None:
            self.statusbar.automap_active = self.automap.active

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_automap(self, overlay: Overlay2D) -> None:
        """AM_Drawer, in place of (or over) the 3D view."""
        sim = self.flow.sim
        player = self.flow.player
        if not self.automap.active or sim is None or sim.level is None or player is None:
            return
        actors = _mobjs(sim) if self.automap.cheating == 2 else None
        self.automap.draw(overlay, sim.level, player, actors, map_name=self.map_name)

    def draw_message(self, overlay: Overlay2D, player: Player) -> None:
        """DrawMessage: after the view and the status bar."""
        draw_message(overlay, player)


def _mobjs(sim: GameSim) -> list[Mobj]:
    return [t for t in sim.thinkers if isinstance(t, Mobj)]
