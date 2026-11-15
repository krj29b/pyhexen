# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Lockstep server (ADR-013): collects every player's ticcmds, broadcasts complete tics.

Call `poll()` often (every frame). In the lobby it accepts players; `start()` begins the
game. Tic N is complete once every player still in the game has sent its ticcmd for N.
A player that sends nothing for `timeout` seconds is dropped: from the next incomplete
tic on, its bit is clear in `Tic.in_game` and every node runs PlayerQuitGame for it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pyhexen.core.ticcmd import TicCmd
from pyhexen.net.protocol import (
    MAX_NET_PLAYERS,
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

MAX_TICS_PER_PACKET = 32
LOBBY_INTERVAL = 0.5


@dataclass
class _Peer:
    addr: Address
    player_num: int
    name: str
    player_class: int
    last_heard: float
    cmds: dict[int, TicCmd] = field(default_factory=dict)
    cmd_ack: int = -1  # highest tic with every cmd up to it received
    tic_ack: int = -1  # highest tic the client says it has
    started: bool = False  # the client has seen START
    in_game: bool = True


class NetServer:
    """The host side of the lockstep protocol."""

    def __init__(
        self,
        transport: Transport,
        iwad_md5: bytes,
        *,
        clock: Callable[[], float] = time.monotonic,
        timeout: float = 10.0,
        max_players: int = MAX_NET_PLAYERS,
    ) -> None:
        self.transport = transport
        self.iwad_md5 = iwad_md5
        self.clock = clock
        self.timeout = timeout
        self.max_players = min(max_players, MAX_NET_PLAYERS)
        self.peers: list[_Peer] = []
        self.started = False
        self.settings: GameSettings | None = None
        self.tics: list[Tic] = []
        self.desync_tic = -1
        self._hashes: dict[int, dict[int, int]] = {}
        self._last_lobby = -1e9

    # ------------------------------------------------------------------

    def start(
        self,
        *,
        map_num: int = 1,
        skill: int = 2,
        deathmatch: bool = False,
        timer_minutes: int = 0,
        random_class: bool = False,
    ) -> GameSettings:
        """Close the lobby and start the game with everyone who has joined."""
        self.settings = GameSettings(
            map_num=map_num,
            skill=skill,
            deathmatch=deathmatch,
            timer_minutes=timer_minutes,
            random_class=random_class,
            classes=[p.player_class for p in self.peers],
            names=[p.name for p in self.peers],
        )
        self.started = True
        now = self.clock()
        for peer in self.peers:
            peer.last_heard = now
        return self.settings

    def poll(self) -> None:
        for addr, data in self.transport.recv():
            msg = decode(data)
            if msg is not None:
                self._handle(addr, msg)
        now = self.clock()
        if not self.started:
            if now - self._last_lobby >= LOBBY_INTERVAL:
                self._last_lobby = now
                lobby = Lobby([p.name for p in self.peers], [p.player_class for p in self.peers])
                for peer in self.peers:
                    self._send(peer.addr, lobby)
            return
        for peer in self.peers:
            if peer.in_game and now - peer.last_heard > self.timeout:
                peer.in_game = False
        self._complete_tics()
        for peer in self.peers:
            if peer.in_game:
                self._send_tics(peer)

    def close(self) -> None:
        for peer in self.peers:
            self._send(peer.addr, Bye())
        self.transport.close()

    @property
    def in_game_mask(self) -> int:
        return sum(1 << p.player_num for p in self.peers if p.in_game)

    # ------------------------------------------------------------------

    def _send(self, addr: Address, msg: Message) -> None:
        self.transport.send(addr, encode(msg))

    def _peer(self, addr: Address) -> _Peer | None:
        return next((p for p in self.peers if p.addr == addr), None)

    def _handle(self, addr: Address, msg: Message) -> None:
        peer = self._peer(addr)
        if isinstance(msg, Hello):
            self._handle_hello(addr, peer, msg)
        elif peer is None:
            return
        elif isinstance(msg, Cmds):
            self._handle_cmds(peer, msg)
        elif isinstance(msg, Bye):
            if self.started:
                peer.in_game = False
            else:
                self.peers.remove(peer)
                for i, p in enumerate(self.peers):
                    p.player_num = i

    def _handle_hello(self, addr: Address, peer: _Peer | None, msg: Hello) -> None:
        if peer is not None:  # a resend: answer again
            peer.last_heard = self.clock()
            if self.started and self.settings is not None:
                self._send(addr, Start(peer.player_num, self.settings))
            else:
                self._send(addr, Welcome(peer.player_num))
            return
        if msg.version != PROTOCOL_VERSION:
            reason = f"protocol version {msg.version}, the host has {PROTOCOL_VERSION}"
        elif msg.iwad_md5 != self.iwad_md5:
            reason = "different IWAD or PWADs (MD5 does not match the host's)"
        elif self.started:
            reason = "the game has already started"
        elif len(self.peers) >= self.max_players:
            reason = f"the game is full ({self.max_players} players)"
        else:
            reason = ""
        if reason:
            self._send(addr, Reject(reason))
            return
        peer = _Peer(addr, len(self.peers), msg.name, msg.player_class % 3, self.clock())
        self.peers.append(peer)
        self._send(addr, Welcome(peer.player_num))

    def _handle_cmds(self, peer: _Peer, msg: Cmds) -> None:
        peer.last_heard = self.clock()
        peer.started = True
        peer.tic_ack = max(peer.tic_ack, msg.ack_tic)
        if not peer.in_game:
            return
        for i, cmd in enumerate(msg.cmds):
            tic = msg.first_tic + i
            if tic > peer.cmd_ack and tic >= len(self.tics):
                peer.cmds.setdefault(tic, cmd)
        while peer.cmd_ack + 1 in peer.cmds or peer.cmd_ack + 1 < len(self.tics):
            peer.cmd_ack += 1
        for tic, h in msg.hashes:
            self._record_hash(peer.player_num, tic, h)

    def _complete_tics(self) -> None:
        if self.settings is None:
            return
        while True:
            tic = len(self.tics)
            players = [p for p in self.peers if p.in_game]
            if not players or any(tic not in p.cmds for p in players):
                return
            cmds = [TicCmd() for _ in range(self.settings.num_players)]
            for p in players:
                cmds[p.player_num] = p.cmds.pop(tic)
            self.tics.append(Tic(self.in_game_mask, cmds))

    def _send_tics(self, peer: _Peer) -> None:
        if not peer.started and self.settings is not None:
            self._send(peer.addr, Start(peer.player_num, self.settings))
            return
        first = peer.tic_ack + 1
        chunk = self.tics[first : first + MAX_TICS_PER_PACKET]
        self._send(peer.addr, Tics(peer.cmd_ack, first, chunk, self.desync_tic))

    def _record_hash(self, player_num: int, tic: int, value: int) -> None:
        reports = self._hashes.setdefault(tic, {})
        reports[player_num] = value
        expected = {p.player_num for p in self.peers if p.in_game}
        if expected <= reports.keys():
            values = {reports[n] for n in expected}
            if len(values) > 1 and (self.desync_tic < 0 or tic < self.desync_tic):
                self.desync_tic = tic
            del self._hashes[tic]
