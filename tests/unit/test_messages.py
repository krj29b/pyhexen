# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Player HUD messages: P_SetMessage rules (sim/message.py) and DrawMessage (ui/messages.py)."""

from __future__ import annotations

import numpy as np
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.events import PrintMessage
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.message import (
    MESSAGETICS,
    YELLOW_MESSAGETICS,
    clear_message,
    set_message,
    set_yellow_message,
)
from pyhexen.sim.player import Player
from pyhexen.sim.specials.doors import checked_locked_door
from pyhexen.ui.fonts import FontSet
from pyhexen.ui.messages import draw_message
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def _sim_with_player() -> tuple[GameSim, Player]:
    sim = GameSim()
    player = Player()
    sim.players.append(player)
    return sim, player


def _level_sim() -> GameSim:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    mb.add_thing(x=128, y=128, type=1)
    wb.add_map(mb)
    sim = GameSim(level=load_level(WadStack(wb.to_bytes()), "MAP01"))
    sim.setup_level(player_nums=(0,))
    return sim


def test_set_message_and_ultimate_rules() -> None:
    sim, player = _sim_with_player()
    events: list[PrintMessage] = []
    sim.events.subscribe(PrintMessage, events.append)

    set_message(sim, player, "BLUE MANA")
    assert (player.message, player.message_tics) == ("BLUE MANA", MESSAGETICS)
    assert not player.yellow_message and not player.ultimate_message
    assert events[-1] == PrintMessage(text="BLUE MANA", player_index=0, bold=False)

    set_message(sim, player, "YOU NEED THE STEEL KEY\n", True)
    assert player.ultimate_message
    set_message(sim, player, "GREEN MANA")  # dropped while an ultimate message shows
    assert player.message == "YOU NEED THE STEEL KEY\n"
    set_message(sim, player, "GAME SAVED", True)  # another ultimate one replaces it
    assert player.message == "GAME SAVED"


def test_messages_off_drops_only_normal_messages() -> None:
    sim, player = _sim_with_player()
    sim.messages_on = False
    set_message(sim, player, "MESH ARMOR")
    assert player.message_tics == 0
    set_message(sim, player, "MESSAGES OFF", True)
    assert player.message == "MESSAGES OFF"


def test_yellow_message_lasts_five_times_longer() -> None:
    sim, player = _sim_with_player()
    set_yellow_message(sim, player, "YOU CANNOT USE THIS HERE")
    assert player.yellow_message
    assert player.message_tics == YELLOW_MESSAGETICS == 5 * MESSAGETICS
    set_message(sim, player, "CRYSTAL VIAL")  # P_SetMessage clears the yellow flag
    assert not player.yellow_message
    clear_message(player)
    assert player.message_tics == 0


def test_player_think_counts_down_and_clears_ultimate() -> None:
    sim = _level_sim()
    player = sim.players[0]
    set_message(sim, player, "THE PORTAL HAS BEEN SEALED", True)
    for _ in range(MESSAGETICS - 1):
        sim.tick([])
    assert player.message_tics == 1 and player.ultimate_message
    sim.tick([])
    assert player.message_tics == 0
    assert not player.ultimate_message and not player.yellow_message
    set_message(sim, player, "BLUE MANA")  # normal messages show again
    assert player.message == "BLUE MANA"


def test_locked_door_sets_ultimate_message() -> None:
    sim = _level_sim()
    player = sim.players[0]
    assert player.mo is not None
    assert not checked_locked_door(sim, player.mo, 1)
    assert player.message == "YOU NEED THE STEEL KEY\n"
    assert player.ultimate_message and not player.yellow_message
    player.keys |= 1
    assert checked_locked_door(sim, player.mo, 1)


def test_draw_message_single_centred_line() -> None:
    sim, player = _sim_with_player()
    overlay = Overlay2D(fonts=FontSet.synthetic())

    overlay.clear()
    draw_message(overlay, player)
    assert not np.any(overlay.mask)

    # MN_DrTextA has no line breaks: the newline is a space and stays on one line.
    set_message(sim, player, "ONE HALF\nIS COMPLETE")
    overlay.clear()
    draw_message(overlay, player)
    rows = np.nonzero(overlay.mask.any(axis=1))[0]
    cols = np.nonzero(overlay.mask.any(axis=0))[0]
    font_h = overlay.fonts.font_a.line_height
    assert rows.min() >= 1 and rows.max() < 1 + font_h
    width = overlay.fonts.font_a.text_width("ONE HALF IS COMPLETE")
    assert abs(int(cols.min()) - (160 - width // 2)) <= 1

    # Yellow messages use the yellow font.
    set_yellow_message(sim, player, "ONE HALF")
    overlay.clear()
    draw_message(overlay, player)
    yellow = overlay.indices[overlay.mask]
    set_message(sim, player, "ONE HALF", True)
    overlay.clear()
    draw_message(overlay, player)
    white = overlay.indices[overlay.mask]
    assert not np.array_equal(np.unique(yellow), np.unique(white))
