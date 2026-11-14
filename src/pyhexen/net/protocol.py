# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Packet formats for the lockstep protocol (ADR-013).

Every packet starts with the 4-byte magic `PHXN` and a message type byte. Integers are
little-endian; strings are a length byte and UTF-8. A ticcmd is 7 bytes. `decode`
returns None for anything malformed, so a stray datagram is just ignored.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum

from pyhexen.core.ticcmd import TicCmd

MAGIC = b"PHXN"
PROTOCOL_VERSION = 1
MAX_NET_PLAYERS = 4
MAX_NAME = 16
_TICCMD = struct.Struct("<bbhBBB")  # forwardmove, sidemove, angleturn, buttons, lookfly, arti


class Msg(IntEnum):
    HELLO = 1  # client -> server: join request
    REJECT = 2  # server -> client: refused, with the reason
    WELCOME = 3  # server -> client: accepted into the lobby as player N
    LOBBY = 4  # server -> clients: who is in the lobby
    START = 5  # server -> client: the game starts, with its settings
    CMDS = 6  # client -> server: ticcmds, tic ack, sim hashes
    TICS = 7  # server -> client: complete tics, cmd ack, desync report
    BYE = 8  # either way: leaving


@dataclass(slots=True)
class Hello:
    version: int
    iwad_md5: bytes
    player_class: int
    name: str


@dataclass(slots=True)
class Reject:
    reason: str


@dataclass(slots=True)
class Welcome:
    player_num: int


@dataclass(slots=True)
class Lobby:
    names: list[str]
    classes: list[int]


@dataclass(slots=True)
class GameSettings:
    """Everything every node needs to build the same sim."""

    map_num: int = 1
    skill: int = 2
    deathmatch: bool = False
    timer_minutes: int = 0
    random_class: bool = False
    classes: list[int] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    @property
    def num_players(self) -> int:
        return len(self.classes)


@dataclass(slots=True)
class Start:
    player_num: int
    settings: GameSettings


@dataclass(slots=True)
class Cmds:
    """`ack_tic`: last tic the client has received (-1 none). `cmds` start at `first_tic`."""

    ack_tic: int
    first_tic: int
    cmds: list[TicCmd]
    hashes: list[tuple[int, int]] = field(default_factory=list)


@dataclass(slots=True)
class Tic:
    """One complete tic: who is in the game, and a ticcmd per player slot."""

    in_game: int  # bit per player
    cmds: list[TicCmd]


@dataclass(slots=True)
class Tics:
    """`ack_tic`: last of this client's cmds the server has (-1 none)."""

    ack_tic: int
    first_tic: int
    tics: list[Tic]
    desync_tic: int = -1


@dataclass(slots=True)
class Bye:
    pass


Message = Hello | Reject | Welcome | Lobby | Start | Cmds | Tics | Bye


def _pack_str(text: str) -> bytes:
    raw = text.encode("utf-8")[:255]
    return bytes([len(raw)]) + raw


def _pack_cmd(cmd: TicCmd) -> bytes:
    return _TICCMD.pack(
        max(-128, min(127, cmd.forwardmove)),
        max(-128, min(127, cmd.sidemove)),
        ((cmd.angleturn + 0x8000) & 0xFFFF) - 0x8000,
        cmd.buttons & 0xFF,
        cmd.lookfly & 0xFF,
        cmd.arti & 0xFF,
    )


class _Reader:
    def __init__(self, data: bytes, pos: int) -> None:
        self.data = data
        self.pos = pos

    def unpack(self, fmt: str) -> tuple[int, ...]:
        s = struct.Struct(fmt)
        vals = s.unpack_from(self.data, self.pos)
        self.pos += s.size
        return vals

    def u8(self) -> int:
        return self.unpack("<B")[0]

    def i32(self) -> int:
        return self.unpack("<i")[0]

    def string(self) -> str:
        n = self.u8()
        raw = self.data[self.pos : self.pos + n]
        if len(raw) != n:
            raise ValueError("short string")
        self.pos += n
        return raw.decode("utf-8", errors="replace")

    def cmd(self) -> TicCmd:
        f, s, a, b, lf, ar = self.unpack(_TICCMD.format)
        return TicCmd(forwardmove=f, sidemove=s, angleturn=a, buttons=b, lookfly=lf, arti=ar)


def encode(msg: Message) -> bytes:
    """Serialize a message to a datagram."""
    out = bytearray(MAGIC)
    if isinstance(msg, Hello):
        out += bytes([Msg.HELLO]) + struct.pack("<H", msg.version)
        out += msg.iwad_md5[:16].ljust(16, b"\0") + bytes([msg.player_class & 0xFF])
        out += _pack_str(msg.name[:MAX_NAME])
    elif isinstance(msg, Reject):
        out += bytes([Msg.REJECT]) + _pack_str(msg.reason)
    elif isinstance(msg, Welcome):
        out += bytes([Msg.WELCOME, msg.player_num])
    elif isinstance(msg, Lobby):
        out += bytes([Msg.LOBBY, len(msg.names)])
        for name, pclass in zip(msg.names, msg.classes, strict=True):
            out += _pack_str(name) + bytes([pclass & 0xFF])
    elif isinstance(msg, Start):
        st = msg.settings
        flags = (1 if st.deathmatch else 0) | (2 if st.random_class else 0)
        out += bytes([Msg.START, msg.player_num, st.map_num & 0xFF, st.skill & 0xFF, flags])
        out += struct.pack("<H", st.timer_minutes) + bytes([len(st.classes)])
        for name, pclass in zip(st.names, st.classes, strict=True):
            out += _pack_str(name) + bytes([pclass & 0xFF])
    elif isinstance(msg, Cmds):
        out += bytes([Msg.CMDS]) + struct.pack("<iiB", msg.ack_tic, msg.first_tic, len(msg.cmds))
        for cmd in msg.cmds:
            out += _pack_cmd(cmd)
        out += bytes([len(msg.hashes)])
        for tic, h in msg.hashes:
            out += struct.pack("<iI", tic, h & 0xFFFFFFFF)
    elif isinstance(msg, Tics):
        out += bytes([Msg.TICS]) + struct.pack(
            "<iiiB", msg.ack_tic, msg.first_tic, msg.desync_tic, len(msg.tics)
        )
        for tic in msg.tics:
            out += bytes([tic.in_game & 0xFF, len(tic.cmds)])
            for cmd in tic.cmds:
                out += _pack_cmd(cmd)
    elif isinstance(msg, Bye):
        out += bytes([Msg.BYE])
    else:  # pragma: no cover - exhaustive
        raise TypeError(f"not a protocol message: {msg!r}")
    return bytes(out)


def decode(data: bytes) -> Message | None:
    """Parse a datagram, or return None if it is not a well-formed message."""
    if len(data) < 5 or data[:4] != MAGIC:
        return None
    r = _Reader(data, 5)
    try:
        kind = data[4]
        if kind == Msg.HELLO:
            version = r.unpack("<H")[0]
            md5 = data[r.pos : r.pos + 16]
            r.pos += 16
            return Hello(version, md5, r.u8(), r.string())
        if kind == Msg.REJECT:
            return Reject(r.string())
        if kind == Msg.WELCOME:
            return Welcome(r.u8())
        if kind == Msg.LOBBY:
            n = r.u8()
            names, classes = [], []
            for _ in range(n):
                names.append(r.string())
                classes.append(r.u8())
            return Lobby(names, classes)
        if kind == Msg.START:
            player_num, map_num, skill, flags = r.unpack("<BBBB")
            timer = r.unpack("<H")[0]
            n = r.u8()
            names, classes = [], []
            for _ in range(n):
                names.append(r.string())
                classes.append(r.u8())
            settings = GameSettings(
                map_num, skill, bool(flags & 1), timer, bool(flags & 2), classes, names
            )
            return Start(player_num, settings)
        if kind == Msg.CMDS:
            ack, first, n = r.unpack("<iiB")
            cmds = [r.cmd() for _ in range(n)]
            nh = r.u8()
            hashes = [(r.i32(), r.unpack("<I")[0]) for _ in range(nh)]
            return Cmds(ack, first, cmds, hashes)
        if kind == Msg.TICS:
            ack, first, desync, n = r.unpack("<iiiB")
            tics = []
            for _ in range(n):
                mask, count = r.unpack("<BB")
                tics.append(Tic(mask, [r.cmd() for _ in range(count)]))
            return Tics(ack, first, tics, desync)
        if kind == Msg.BYE:
            return Bye()
    except (struct.error, ValueError, IndexError):
        return None
    return None
