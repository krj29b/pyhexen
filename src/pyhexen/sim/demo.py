# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Demo recording, playback, and determinism verification harness (T14.1).

Port of G_RecordDemo, G_PlayDemo, G_ReadDemoTiccmd, G_WriteDemoTiccmd in g_game.c.
"""

from __future__ import annotations

import gzip
import json
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyhexen.core.ticcmd import TicCmd

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

DEMO_MAGIC: bytes = b"PXDM"  # PyHexen Demo
DEMO_VERSION: int = 1

_TICCMD_STRUCT = struct.Struct("<bbhhBBBB")


class DemoError(Exception):
    """Base exception for demo file handling."""


class DemoVersionError(DemoError):
    """Raised when demo file version is incompatible."""


class DemoDeterminismError(DemoError):
    """Raised when demo playback fails to produce the expected state hash."""


def copy_ticcmd(cmd: TicCmd) -> TicCmd:
    """Create an independent copy of a TicCmd instance."""
    return TicCmd(
        forwardmove=cmd.forwardmove,
        sidemove=cmd.sidemove,
        angleturn=cmd.angleturn,
        consistancy=cmd.consistancy,
        chatchar=cmd.chatchar,
        buttons=cmd.buttons,
        lookfly=cmd.lookfly,
        arti=cmd.arti,
    )


def pack_ticcmd(cmd: TicCmd) -> bytes:
    """Pack a TicCmd into 10 bytes."""
    # Ensure signed values stay within int8 / int16 bounds
    fwd = max(-128, min(127, cmd.forwardmove))
    side = max(-128, min(127, cmd.sidemove))
    angle = max(-32768, min(32767, cmd.angleturn))
    cons = max(-32768, min(32767, cmd.consistancy))
    return _TICCMD_STRUCT.pack(
        fwd,
        side,
        angle,
        cons,
        cmd.chatchar & 0xFF,
        cmd.buttons & 0xFF,
        cmd.lookfly & 0xFF,
        cmd.arti & 0xFF,
    )


def unpack_ticcmd(data: bytes | bytearray) -> TicCmd:
    """Unpack 10 bytes into a TicCmd."""
    fwd, side, angle, cons, chatchar, buttons, lookfly, arti = _TICCMD_STRUCT.unpack(data)
    return TicCmd(
        forwardmove=fwd,
        sidemove=side,
        angleturn=angle,
        consistancy=cons,
        chatchar=chatchar,
        buttons=buttons,
        lookfly=lookfly,
        arti=arti,
    )


@dataclass(slots=True)
class DemoHeader:
    """Header metadata for a PyHexen demo file."""

    magic: str = "PXDM"
    version: int = DEMO_VERSION
    map_num: int = 1
    skill: int = 2
    player_class: int = 0
    rng_state: tuple[int, int] = (0, 0)
    nomonsters: bool = False
    respawn: bool = False
    netgame: bool = False
    deathmatch: bool = False
    num_players: int = 1
    iwad_checksum: str = ""
    final_state_hash: str = ""
    total_tics: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "magic": self.magic,
            "version": self.version,
            "map_num": self.map_num,
            "skill": self.skill,
            "player_class": self.player_class,
            "rng_state": list(self.rng_state),
            "nomonsters": self.nomonsters,
            "respawn": self.respawn,
            "netgame": self.netgame,
            "deathmatch": self.deathmatch,
            "num_players": self.num_players,
            "iwad_checksum": self.iwad_checksum,
            "final_state_hash": self.final_state_hash,
            "total_tics": self.total_tics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DemoHeader:
        rng = tuple(d.get("rng_state", (0, 0)))
        return cls(
            magic=d.get("magic", "PXDM"),
            version=d.get("version", DEMO_VERSION),
            map_num=d.get("map_num", 1),
            skill=d.get("skill", 2),
            player_class=d.get("player_class", 0),
            rng_state=(int(rng[0]), int(rng[1])),
            nomonsters=d.get("nomonsters", False),
            respawn=d.get("respawn", False),
            netgame=d.get("netgame", False),
            deathmatch=d.get("deathmatch", False),
            num_players=d.get("num_players", 1),
            iwad_checksum=d.get("iwad_checksum", ""),
            final_state_hash=d.get("final_state_hash", ""),
            total_tics=d.get("total_tics", 0),
        )


@dataclass(slots=True)
class Demo:
    """Full demo recording including header metadata and stream of TicCmds."""

    header: DemoHeader
    tics: list[tuple[TicCmd, ...]] = field(default_factory=list)

    def save_to_bytes(self) -> bytes:
        """Serialize demo to compressed bytes."""
        hdr_dict = self.header.to_dict()
        hdr_bytes = json.dumps(hdr_dict, separators=(",", ":")).encode("utf-8")

        # Pack tic commands
        tic_buf = bytearray()
        for cmd_tuple in self.tics:
            for cmd in cmd_tuple:
                tic_buf.extend(pack_ticcmd(cmd))

        raw_payload = struct.pack("<4sI", DEMO_MAGIC, len(hdr_bytes)) + hdr_bytes + bytes(tic_buf)
        return gzip.compress(raw_payload)

    @classmethod
    def load_from_bytes(cls, data: bytes) -> Demo:
        """Deserialize demo from compressed bytes."""
        decompressed = gzip.decompress(data)
        if len(decompressed) < 8:
            raise DemoError("Demo data is truncated")

        magic, hdr_len = struct.unpack("<4sI", decompressed[:8])
        if magic != DEMO_MAGIC:
            raise DemoError(f"Invalid demo magic: expected {DEMO_MAGIC!r}, got {magic!r}")

        hdr_end = 8 + hdr_len
        if len(decompressed) < hdr_end:
            raise DemoError("Demo header is truncated")

        hdr_dict = json.loads(decompressed[8:hdr_end].decode("utf-8"))
        header = DemoHeader.from_dict(hdr_dict)
        if header.version != DEMO_VERSION:
            raise DemoVersionError(
                f"Incompatible demo version: expected {DEMO_VERSION}, got {header.version}"
            )

        # Unpack TicCmd stream
        tic_stream = decompressed[hdr_end:]
        num_players = max(1, header.num_players)
        bytes_per_tic = 10 * num_players

        tics: list[tuple[TicCmd, ...]] = []
        offset = 0
        while offset + bytes_per_tic <= len(tic_stream):
            cmds: list[TicCmd] = []
            for _ in range(num_players):
                cmds.append(unpack_ticcmd(tic_stream[offset : offset + 10]))
                offset += 10
            tics.append(tuple(cmds))

        return cls(header=header, tics=tics)

    def save_file(self, path: Path | str) -> None:
        """Save demo file to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(self.save_to_bytes())

    @classmethod
    def load_file(cls, path: Path | str) -> Demo:
        """Load demo file from disk."""
        with open(Path(path), "rb") as f:
            return cls.load_from_bytes(f.read())


class DemoRecorder:
    """Records per-tic player inputs during simulation execution."""

    def __init__(
        self,
        sim: GameSim,
        file_path: Path | str | None = None,
        iwad_checksum: str = "",
    ) -> None:
        self.sim: GameSim = sim
        self.file_path: Path | None = Path(file_path) if file_path else None
        self.header: DemoHeader = DemoHeader(
            map_num=getattr(sim, "map_num", 1),
            skill=sim.skill,
            player_class=sim.player_class,
            rng_state=sim.rng.get_state(),
            nomonsters=sim.nomonsters,
            respawn=False,
            netgame=sim.netgame,
            deathmatch=sim.deathmatch,
            num_players=len(sim.players) or 1,
            iwad_checksum=iwad_checksum,
        )
        self.tics: list[tuple[TicCmd, ...]] = []

    def record_tic(self, cmds: Sequence[TicCmd]) -> tuple[TicCmd, ...]:
        """Record player commands for current tic and return normalized copies.

        Port of G_WriteDemoTiccmd (g_game.c:2070) which reads back packed commands
        to guarantee identical simulation inputs between recording and playback.
        """
        packed_cmds = tuple(unpack_ticcmd(pack_ticcmd(c)) for c in cmds)
        self.tics.append(packed_cmds)
        return packed_cmds

    def finish(self) -> Demo:
        """Finalize recording, record expected final state hash, and write file."""
        self.header.total_tics = len(self.tics)
        self.header.final_state_hash = self.sim.state_hash()
        demo = Demo(header=self.header, tics=self.tics)
        if self.file_path:
            demo.save_file(self.file_path)
        return demo


class DemoPlayer:
    """Replays recorded demo TicCmd streams and validates determinism."""

    def __init__(self, demo: Demo) -> None:
        self.demo: Demo = demo
        self.current_tic: int = 0

    @classmethod
    def from_file(cls, path: Path | str) -> DemoPlayer:
        """Load demo from file and wrap in a DemoPlayer."""
        return cls(Demo.load_file(path))

    @property
    def is_finished(self) -> bool:
        """Return True when all recorded tics have been played back."""
        return self.current_tic >= len(self.demo.tics)

    @property
    def total_tics(self) -> int:
        """Total number of recorded tics."""
        return len(self.demo.tics)

    def get_next_cmds(self) -> tuple[TicCmd, ...]:
        """Fetch the command tuple for the next simulation tic."""
        if self.is_finished:
            return ()
        cmds = self.demo.tics[self.current_tic]
        self.current_tic += 1
        return cmds

    def check_determinism(self, sim: GameSim) -> bool:
        """Verify that simulation state hash exactly matches the recorded demo."""
        return sim.state_hash() == self.demo.header.final_state_hash


def play_demo_to_completion(
    sim: GameSim,
    player: DemoPlayer,
    on_tic: Callable[[int, GameSim], None] | None = None,
) -> bool:
    """Run simulation through all demo tics and return whether state hash matched."""
    while not player.is_finished:
        cmds = player.get_next_cmds()
        sim.tick(cmds)
        if on_tic is not None:
            on_tic(player.current_tic, sim)

    return player.check_determinism(sim)
