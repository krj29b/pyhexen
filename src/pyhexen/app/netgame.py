# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Netgames in the app (S2c): hosting or joining, the lobby, and running the GameFlow
from the server's tics (ADR-013).

`NetGameDriver.pump()` polls the network every frame. Once the host starts the game,
`tick(due, make_cmd)` sends this player's ticcmds (one per 35 Hz tic) and runs every
complete tic through `GameFlow.tick`, so all nodes run the same tics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyhexen.core.ticcmd import TicCmd
from pyhexen.net.client import ClientState, NetClient
from pyhexen.net.server import NetServer
from pyhexen.net.session import LockstepRunner
from pyhexen.net.transport import Address, UdpTransport

if TYPE_CHECKING:
    from pyhexen.app.game_flow import GameFlow
    from pyhexen.render.overlay2d import Overlay2D

DEFAULT_PORT = 2342  # Chocolate Doom's port
CLASS_NAMES = ("FIGHTER", "CLERIC", "MAGE")


def iwad_md5(path: Path | str) -> bytes:
    """MD5 of the IWAD file (every node must use the same one)."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def parse_address(text: str) -> Address:
    """`host[:port]` -> (host, port)."""
    host, _, port = text.rpartition(":") if ":" in text else (text, "", "")
    return host, int(port) if port else DEFAULT_PORT


class NetGameDriver:
    """Glue between a NetClient (and NetServer when hosting) and the GameFlow."""

    def __init__(
        self,
        flow: GameFlow,
        client: NetClient,
        server: NetServer | None = None,
        *,
        start_players: int = 0,
        map_num: int = 1,
        skill: int = 2,
        deathmatch: bool = False,
        timer_minutes: int = 0,
        random_class: bool = False,
    ) -> None:
        self.flow = flow
        self.client = client
        self.server = server
        self.start_players = start_players
        self.map_num = map_num
        self.skill = skill
        self.deathmatch = deathmatch
        self.timer_minutes = timer_minutes
        self.random_class = random_class
        self.runner: LockstepRunner | None = None
        self._desync_reported = False

    @classmethod
    def host(
        cls,
        flow: GameFlow,
        md5: bytes,
        *,
        port: int = DEFAULT_PORT,
        name: str,
        player_class: int,
        **settings: Any,
    ) -> NetGameDriver:
        server = NetServer(UdpTransport("0.0.0.0", port), md5)
        client = NetClient(
            UdpTransport("0.0.0.0", 0),
            ("127.0.0.1", server.transport.address[1]),
            name=name,
            player_class=player_class,
            iwad_md5=md5,
        )
        return cls(flow, client, server, **settings)

    @classmethod
    def join(
        cls, flow: GameFlow, md5: bytes, address: Address, *, name: str, player_class: int
    ) -> NetGameDriver:
        client = NetClient(
            UdpTransport("0.0.0.0", 0),
            address,
            name=name,
            player_class=player_class,
            iwad_md5=md5,
        )
        return cls(flow, client)

    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self.runner is not None

    @property
    def is_host(self) -> bool:
        return self.server is not None

    @property
    def failed(self) -> str:
        """Why the netgame cannot go on, or an empty string."""
        if self.client.state == ClientState.REJECTED:
            return f"REFUSED BY THE HOST: {self.client.reject_reason.upper()}"
        if self.client.state == ClientState.DISCONNECTED:
            return "LOST THE CONNECTION TO THE HOST"
        return ""

    def host_start(self) -> None:
        """Start the game with everyone in the lobby (host only)."""
        if self.server is not None and not self.server.started and self.server.peers:
            self.server.start(
                map_num=self.map_num,
                skill=self.skill,
                deathmatch=self.deathmatch,
                timer_minutes=self.timer_minutes,
                random_class=self.random_class,
            )

    def pump(self) -> None:
        if self.server is not None:
            self.server.poll()
        self.client.poll()
        if (
            self.server is not None
            and not self.server.started
            and self.start_players
            and len(self.server.peers) >= self.start_players
        ):
            self.host_start()
        if self.client.state == ClientState.GAME and self.runner is None:
            st = self.client.settings
            assert st is not None
            self.flow.start_netgame(
                map_num=st.map_num,
                skill=st.skill,
                player_classes=st.classes,
                consoleplayer=self.client.player_num,
                deathmatch=st.deathmatch,
                timer_minutes=st.timer_minutes,
                random_class=st.random_class,
            )
            self.runner = LockstepRunner.for_game(
                self.client, self.flow.tick, lambda: self.flow.sim, st.num_players
            )

    def tick(
        self, due: int, make_cmd: Callable[[], TicCmd], max_tics: int | None = None
    ) -> list[Any]:
        """Send `due` local ticcmds and run the complete tics (at most `max_tics`)."""
        if self.runner is None:
            return []
        for _ in range(due):
            if self.client.can_make_cmd():
                self.client.make_cmd(make_cmd())
        results = self.runner.run_ready(max_tics)
        self._report_desync()
        return results

    def _report_desync(self) -> None:
        if self._desync_reported or self.client.desync_tic < 0:
            return
        self._desync_reported = True
        from pyhexen.sim.message import set_message

        player = self.flow.player
        if self.flow.sim is not None and player is not None:
            set_message(self.flow.sim, player, f"DESYNC AT TIC {self.client.desync_tic}", True)

    def peers_have(self, tic: int) -> bool:
        """Host: every player still in the game has received tics up to `tic`."""
        if self.server is None:
            return True
        return all(p.tic_ack >= tic for p in self.server.peers if p.in_game)

    def close(self) -> None:
        if self.client.state == ClientState.GAME:
            self.client.leave()
        self.client.transport.close()
        if self.server is not None:
            self.server.close()

    # ------------------------------------------------------------------

    def lobby_lines(self) -> list[str]:
        if self.failed:
            return [self.failed, "", "PRESS ESCAPE"]
        if self.client.state == ClientState.CONNECTING:
            return [f"CONNECTING TO {self.client.server[0]}:{self.client.server[1]}..."]
        lines = ["DEATHMATCH" if self.deathmatch and self.is_host else "NETGAME LOBBY", ""]
        for i, (name, pclass) in enumerate(
            zip(self.client.lobby_names, self.client.lobby_classes, strict=False)
        ):
            mark = "*" if i == self.client.player_num else " "
            lines.append(f"{mark}{i + 1}. {name.upper()} - {CLASS_NAMES[pclass % 3]}")
        lines.append("")
        if self.is_host:
            if self.start_players:
                lines.append(f"STARTS WITH {self.start_players} PLAYERS")
            lines.append("PRESS ENTER TO START")
        else:
            lines.append("WAITING FOR THE HOST")
        return lines

    def draw_lobby(self, overlay: Overlay2D) -> None:
        y = 40
        for line in self.lobby_lines():
            overlay.draw_text_a(line, 40, y)
            y += 12
