# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Lockstep client (ADR-013): sends this player's ticcmds, receives complete tics.

Every frame: `poll()`, then `make_cmd()` once per 35 Hz tic while `can_make_cmd()`,
then run each tic from `take_tic()`. Every packet repeats all ticcmds the server has not
acknowledged, so a lost packet only costs a little latency.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum

from pyhexen.core.ticcmd import TicCmd
from pyhexen.net.protocol import (
    PROTOCOL_VERSION,
    Bye,
    Cmds,
    GameSettings,
    Hello,
    Lobby,
    Message,
    Reject,
    Start,
    Tic,
    Tics,
    Welcome,
    decode,
    encode,
)
from pyhexen.net.transport import Address, Transport

BACKUPTICS = 64  # how far ahead of the game this client may make ticcmds
HELLO_INTERVAL = 0.5
MAX_CMDS_PER_PACKET = 64
HASHES_KEPT = 4


class ClientState(Enum):
    CONNECTING = "connecting"
    LOBBY = "lobby"
    GAME = "game"
    REJECTED = "rejected"
    DISCONNECTED = "disconnected"


class NetClient:
    """One player's side of the lockstep protocol."""

    def __init__(
        self,
        transport: Transport,
        server: Address,
        *,
        name: str,
        player_class: int,
        iwad_md5: bytes,
        clock: Callable[[], float] = time.monotonic,
        timeout: float = 10.0,
    ) -> None:
        self.transport = transport
        self.server = server
        self.name = name
        self.player_class = player_class
        self.iwad_md5 = iwad_md5
        self.clock = clock
        self.timeout = timeout

        self.state = ClientState.CONNECTING
        self.reject_reason = ""
        self.player_num = -1
        self.lobby_names: list[str] = []
        self.lobby_classes: list[int] = []
        self.settings: GameSettings | None = None
        self.desync_tic = -1

        self.maketic = 0  # next tic to make a ticcmd for
        self.run_tic = 0  # next tic to hand to the game
        self.server_ack = -1  # last of our ticcmds the server has
        self._local: dict[int, TicCmd] = {}
        self._tics: dict[int, Tic] = {}
        self._received = -1  # highest contiguous tic received
        self._hashes: list[tuple[int, int]] = []
        self._last_heard = clock()
        self._last_hello = -1e9

    # ------------------------------------------------------------------

    def poll(self) -> None:
        for addr, data in self.transport.recv():
            if addr != self.server:
                continue
            msg = decode(data)
            if msg is not None:
                self._last_heard = self.clock()
                self._handle(msg)
        now = self.clock()
        if self.state in (ClientState.REJECTED, ClientState.DISCONNECTED):
            return
        if now - self._last_heard > self.timeout:
            self.state = ClientState.DISCONNECTED
            return
        if self.state in (ClientState.CONNECTING, ClientState.LOBBY):
            if now - self._last_hello >= HELLO_INTERVAL:
                self._last_hello = now
                self._send(Hello(PROTOCOL_VERSION, self.iwad_md5, self.player_class, self.name))
        elif self.state == ClientState.GAME:
            self._send_cmds()

    def can_make_cmd(self) -> bool:
        return self.state == ClientState.GAME and self.maketic - self.run_tic < BACKUPTICS

    def make_cmd(self, cmd: TicCmd) -> None:
        """Queue this player's ticcmd for the next tic (call at 35 Hz)."""
        self._local[self.maketic] = cmd
        self.maketic += 1

    def take_tic(self) -> Tic | None:
        """The next complete tic to run, if it has arrived."""
        tic = self._tics.pop(self.run_tic, None)
        if tic is not None:
            self.run_tic += 1
        return tic

    def report_hash(self, tic: int, value: int) -> None:
        """Send the sim hash after running `tic` (the server compares all players')."""
        self._hashes.append((tic, value & 0xFFFFFFFF))
        self._hashes = self._hashes[-HASHES_KEPT:]

    def leave(self) -> None:
        self._send(Bye())
        self.state = ClientState.DISCONNECTED

    # ------------------------------------------------------------------

    def _send(self, msg: Message) -> None:
        self.transport.send(self.server, encode(msg))

    def _send_cmds(self) -> None:
        first = self.server_ack + 1
        last = min(self.maketic, first + MAX_CMDS_PER_PACKET)
        cmds = [self._local[t] for t in range(first, last)]
        self._send(Cmds(self._received, first, cmds, list(self._hashes)))

    def _handle(self, msg: Message) -> None:
        if isinstance(msg, Reject):
            self.state = ClientState.REJECTED
            self.reject_reason = msg.reason
        elif isinstance(msg, Welcome):
            if self.state == ClientState.CONNECTING:
                self.state = ClientState.LOBBY
            self.player_num = msg.player_num
        elif isinstance(msg, Lobby):
            self.lobby_names = list(msg.names)
            self.lobby_classes = list(msg.classes)
        elif isinstance(msg, Start):
            if self.state != ClientState.GAME:
                self.state = ClientState.GAME
                self.player_num = msg.player_num
                self.settings = msg.settings
        elif isinstance(msg, Tics):
            if self.state != ClientState.GAME:
                return
            if msg.ack_tic > self.server_ack:
                for t in range(self.server_ack + 1, msg.ack_tic + 1):
                    self._local.pop(t, None)
                self.server_ack = msg.ack_tic
            for i, tic in enumerate(msg.tics):
                n = msg.first_tic + i
                if n >= self.run_tic:
                    self._tics.setdefault(n, tic)
            while self._received + 1 in self._tics or self._received + 1 < self.run_tic:
                self._received += 1
            if msg.desync_tic >= 0:
                self.desync_tic = msg.desync_tic
        elif isinstance(msg, Bye):
            self.state = ClientState.DISCONNECTED
