# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""HUD player and ACS message display system (T12.3).

Port of P_SetMessage, P_SetYellowMessage, P_ClearMessage (p_inter.c)
and DrawMessage (h2_main.c).
"""

from __future__ import annotations

from collections import deque

from pyhexen.core.events import EventBus, PrintMessage
from pyhexen.render.overlay2d import VIRTUAL_WIDTH, Overlay2D

MESSAGE_TICS: int = 4 * 35  # 140 tics (4 seconds at 35 Hz)
YELLOW_MESSAGE_TICS: int = 4 * 35  # 140 tics
YELLOW_MESSAGE_Y: int = 68
MAX_MESSAGE_HISTORY: int = 32


class MessageSystem:
    """Manages top-left pickup/system messages and centred yellow ACS messages.

    Port of player message state in p_inter.c and DrawMessage in h2_main.c.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        *,
        messages_on: bool = True,
        console_player: int = 0,
        message_tics: int = MESSAGE_TICS,
        yellow_message_tics: int = YELLOW_MESSAGE_TICS,
    ) -> None:
        self.messages_on: bool = messages_on
        self.console_player: int = console_player
        self.default_message_tics: int = int(message_tics)
        self.default_yellow_tics: int = int(yellow_message_tics)

        self.text: str = ""
        self.tics: int = 0
        self.ultimate_message: bool = False

        self.yellow_text: str = ""
        self.yellow_tics: int = 0

        self.history: deque[str] = deque(maxlen=MAX_MESSAGE_HISTORY)

        if event_bus is not None:
            self.attach_event_bus(event_bus)

    def attach_event_bus(self, event_bus: EventBus) -> None:
        """Subscribe this `MessageSystem` to `PrintMessage` events on `event_bus`."""
        event_bus.subscribe(PrintMessage, self.handle_print_message)

    def handle_print_message(self, event: PrintMessage) -> None:
        """Consume a `PrintMessage` event from the simulation or ACS VM."""
        if event.player_index is not None and event.player_index != self.console_player:
            return
        if event.bold:
            self.set_yellow_message(event.text)
        else:
            self.set_message(event.text)

    def set_message(
        self,
        text: str,
        ultimate: bool = False,
        *,
        tics: int | None = None,
    ) -> None:
        """Set the top-left HUD message (port of P_SetMessage in p_inter.c)."""
        if (self.ultimate_message or not self.messages_on) and not ultimate:
            return
        self.text = str(text)
        self.tics = self.default_message_tics if tics is None else int(tics)
        if ultimate:
            self.ultimate_message = True
        if self.text:
            self.history.append(self.text)

    def set_yellow_message(
        self,
        text: str,
        ultimate: bool = False,
        *,
        tics: int | None = None,
    ) -> None:
        """Set the centred bold/yellow ACS message (port of P_SetYellowMessage in p_inter.c)."""
        if (self.ultimate_message or not self.messages_on) and not ultimate:
            return
        self.yellow_text = str(text)
        self.yellow_tics = self.default_yellow_tics if tics is None else int(tics)
        if ultimate:
            self.ultimate_message = True
        if self.yellow_text:
            self.history.append(self.yellow_text)

    def clear(self) -> None:
        """Clear active messages immediately (port of P_ClearMessage in p_inter.c)."""
        self.text = ""
        self.tics = 0
        self.ultimate_message = False
        self.yellow_text = ""
        self.yellow_tics = 0

    def tick(self) -> None:
        """Advance message timers by 1 tic (35 Hz)."""
        if self.tics > 0:
            self.tics -= 1
            if self.tics <= 0:
                self.tics = 0
                self.text = ""
                self.ultimate_message = False

        if self.yellow_tics > 0:
            self.yellow_tics -= 1
            if self.yellow_tics <= 0:
                self.yellow_tics = 0
                self.yellow_text = ""

    def draw(self, overlay: Overlay2D) -> None:
        """Render active normal and yellow messages onto `overlay`."""
        if self.tics > 0 and self.text:
            overlay.draw_text_a(self.text, 1, 1, yellow=False, center=False)

        if self.yellow_tics > 0 and self.yellow_text:
            overlay.draw_text_a(
                self.yellow_text,
                VIRTUAL_WIDTH // 2,
                YELLOW_MESSAGE_Y,
                yellow=True,
                center=True,
            )
