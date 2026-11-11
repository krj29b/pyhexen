# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""HUD message drawing: port of DrawMessage (h2_main.c).

The message itself is player state set by the sim (`sim/message.py`, the port of
P_SetMessage / P_SetYellowMessage). This module only draws it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.render.overlay2d import VIRTUAL_WIDTH, Overlay2D

if TYPE_CHECKING:
    from pyhexen.sim.player import Player

MESSAGE_Y: int = 1


def message_text(player: Player) -> str:
    """Return the text DrawMessage would show for `player`, or "" when there is none."""
    if player.message_tics <= 0:
        return ""
    return player.message


def draw_message(overlay: Overlay2D, player: Player) -> None:
    """Port of DrawMessage (h2_main.c): one line, centred at the top of the screen.

    MN_DrTextA has no line breaks: every character below '!' (newline included)
    is a 5-pixel space, so the text is drawn as a single line.
    """
    text = message_text(player)
    if not text:
        return
    line = "".join(" " if ord(ch) < 33 else ch for ch in text)
    font = overlay.fonts.font_a_yellow if player.yellow_message else overlay.fonts.font_a
    x = VIRTUAL_WIDTH // 2 - font.text_width(line) // 2
    font.draw_text(overlay.indices, overlay.mask, line, x, MESSAGE_Y)
