# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Player HUD messages: port of P_SetMessage, P_SetYellowMessage, P_ClearMessage (p_inter.c).

As in vanilla, the message lives on the player (`message`, `message_tics`,
`ultimate_message`, `yellow_message`) and counts down in `p_player_think`. The UI
only draws it (DrawMessage in h2_main.c). A `PrintMessage` event is also emitted so
other listeners (logging, tests) can see it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.events import PrintMessage

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.sim.player import Player

MESSAGETICS: int = 4 * 35  # h2def.h
YELLOW_MESSAGETICS: int = 5 * MESSAGETICS  # bold messages last longer


def _player_index(sim: GameSim, player: Player) -> int | None:
    for i, p in enumerate(sim.players):
        if p is player:
            return i
    return None


def set_message(sim: GameSim, player: Player, text: str, ultimate: bool = False) -> None:
    """Port of P_SetMessage (p_inter.c).

    A non-ultimate message is dropped while an ultimate one is showing, or when
    messages are switched off (`sim.messages_on`, vanilla `messageson`).
    """
    if (player.ultimate_message or not sim.messages_on) and not ultimate:
        return
    player.message = text
    player.message_tics = MESSAGETICS
    player.yellow_message = False
    if ultimate:
        player.ultimate_message = True
    sim.events.emit(PrintMessage(text=text, player_index=_player_index(sim, player)))


def set_yellow_message(sim: GameSim, player: Player, text: str, ultimate: bool = False) -> None:
    """Port of P_SetYellowMessage (p_inter.c)."""
    if (player.ultimate_message or not sim.messages_on) and not ultimate:
        return
    player.message = text
    player.message_tics = YELLOW_MESSAGETICS
    player.yellow_message = True
    if ultimate:
        player.ultimate_message = True
    sim.events.emit(PrintMessage(text=text, player_index=_player_index(sim, player), bold=True))


def clear_message(player: Player) -> None:
    """Port of P_ClearMessage (p_inter.c)."""
    player.message_tics = 0


def console_player(sim: GameSim) -> Player | None:
    """Return `players[consoleplayer]` (player 0), or None before players spawn."""
    return sim.players[0] if sim.players else None
