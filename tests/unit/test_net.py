# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""S2b: lockstep netcode over the loopback network (virtual time) and real UDP."""

from __future__ import annotations

import time
from dataclasses import dataclass

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.ticcmd import BT_ATTACK, TicCmd
from pyhexen.net.client import ClientState, NetClient
from pyhexen.net.protocol import (
    PROTOCOL_VERSION,
    Cmds,
    GameSettings,
    Hello,
    Lobby,
    Reject,
    Start,
    Tic,
    Tics,
    Welcome,
    decode,
    encode,
)
from pyhexen.net.server import NetServer
from pyhexen.net.session import LockstepRunner, build_netgame_sim, sim_hash
from pyhexen.net.transport import LoopbackNetwork, UdpTransport
from pyhexen.sim.info_data import MOBJINFO, MobjType
from pyhexen.sim.mobj import Mobj
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level

MD5 = bytes(range(16))
FRAME = 1 / 35
ETTIN = MOBJINFO[MobjType.MT_ETTIN].doomednum


def _wad() -> WadStack:
    wb = WadBuilder("IWAD")
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=1024, height=1024)
    for i in range(4):
        mb.add_thing(x=100 + 100 * i, y=100, type=i + 1, angle=90)
        mb.add_thing(x=100 + 100 * i, y=900, type=11, angle=270)
    for i in range(3):
        mb.add_thing(x=300 + 200 * i, y=600, type=ETTIN, angle=270)
    return WadStack(wb.add_map(mb).to_bytes())


@dataclass
class Clock:
    t: float = 0.0

    def __call__(self) -> float:
        return self.t


def _bot(player: int, tic: int) -> TicCmd:
    phase = (tic // 40 + player) % 4
    return TicCmd(
        forwardmove=(25, -25, 25, 0)[phase],
        sidemove=(0, 0, 24, -24)[phase],
        angleturn=(player + 1) * 256 if tic % 70 < 20 else 0,
        buttons=BT_ATTACK if (tic + 9 * player) % 30 < 6 else 0,
    )


class Session:
    """A server and `n` clients on a loopback network, each client with its own sim."""

    def __init__(
        self,
        n: int,
        *,
        delay: float = 0.0,
        loss: float = 0.0,
        deathmatch: bool = False,
        timeout: float = 2.0,
    ) -> None:
        self.clock = Clock()
        self.net = LoopbackNetwork(self.clock, delay=delay, jitter=delay / 2, loss=loss, seed=7)
        self.server = NetServer(
            self.net.endpoint(("host", 1)), MD5, clock=self.clock, timeout=timeout
        )
        self.clients = [
            NetClient(
                self.net.endpoint((f"client{i}", 2)),
                ("host", 1),
                name=f"P{i + 1}",
                player_class=i % 3,
                iwad_md5=MD5,
                clock=self.clock,
                timeout=timeout,
            )
            for i in range(n)
        ]
        self.wad = _wad()
        self.runners: list[LockstepRunner | None] = [None] * n
        self.paused: set[int] = set()
        self.frames = 0
        for _ in range(400):
            self.frame()
            if len(self.server.peers) == n and all(
                c.state == ClientState.LOBBY for c in self.clients
            ):
                break
        assert len(self.server.peers) == n
        self.server.start(deathmatch=deathmatch, skill=2)

    def _level(self) -> Level:
        return load_level(self.wad, "MAP01")

    def frame(self, stop_at: int | None = None) -> None:
        self.clock.t += FRAME
        self.frames += 1
        self.server.poll()
        for i, client in enumerate(self.clients):
            if i in self.paused:
                continue
            client.poll()
            if client.state == ClientState.GAME and self.runners[i] is None:
                assert client.settings is not None
                sim = build_netgame_sim(self._level(), client.settings, client.player_num)
                self.runners[i] = LockstepRunner(client, sim)
            runner = self.runners[i]
            if runner is None:
                continue
            if client.can_make_cmd():
                client.make_cmd(_bot(client.player_num, client.maketic))
            limit = None if stop_at is None else max(0, stop_at - client.run_tic)
            runner.run_ready(limit)

    def run_until(
        self, tic: int, max_frames: int = 20000, players: list[int] | None = None
    ) -> None:
        active = players if players is not None else range(len(self.clients))
        while True:
            done = [self.clients[i].run_tic >= tic for i in active]
            if all(done):
                return
            assert self.frames < max_frames, "the game stalled"
            self.frame(stop_at=tic)

    def sims(self, players: list[int] | None = None):
        active = players if players is not None else range(len(self.clients))
        return [self.runners[i].sim for i in active]  # type: ignore[union-attr]


def test_messages_round_trip() -> None:
    settings = GameSettings(23, 3, True, 10, True, [0, 1, 2], ["a", "bé", "c"])
    cmd = TicCmd(
        forwardmove=-50, sidemove=24, angleturn=-1280, buttons=BT_ATTACK, lookfly=0x31, arti=5
    )
    for msg in (
        Hello(PROTOCOL_VERSION, MD5, 2, "Player"),
        Reject("different IWAD"),
        Welcome(3),
        Lobby(["a", "b"], [0, 2]),
        Start(1, settings),
        Cmds(41, 40, [cmd, TicCmd()], [(34, 0xDEADBEEF)]),
        Tics(39, 40, [Tic(0b1011, [cmd, TicCmd(), cmd, TicCmd()])], desync_tic=69),
    ):
        assert decode(encode(msg)) == msg
    assert decode(b"junk") is None
    assert decode(encode(Welcome(1))[:5]) is None


def test_four_players_on_a_lossy_network_stay_in_sync() -> None:
    session = Session(4, delay=0.1, loss=0.1)
    session.run_until(2000)
    sims = session.sims()
    assert len({sim_hash(s) for s in sims}) == 1
    assert len({s.state_hash() for s in sims}) == 1
    assert all(s.level_time == 2000 for s in sims)
    # slots follow join order; each sim is seen from its own client's player
    assert [s.consoleplayer for s in sims] == [c.player_num for c in session.clients]
    assert sorted(s.consoleplayer for s in sims) == [0, 1, 2, 3]
    assert session.server.desync_tic == -1
    assert session.net.dropped > 0  # the loss really happened
    # every player's own input reached everybody: the four players are spread out
    positions = {(p.mo.x, p.mo.y) for p in sims[0].players if p.mo is not None}
    assert len(positions) == 4


def test_a_desync_is_reported_at_the_first_hash_after_it() -> None:
    session = Session(3, delay=0.05)
    session.run_until(500)
    sim = session.sims()[2]
    ettin = next(t for t in sim.thinkers if isinstance(t, Mobj) and t.type == MobjType.MT_ETTIN)
    ettin.x += FRACUNIT  # only player 3's sim changes
    session.run_until(700)
    assert session.server.desync_tic == 524  # hashes go out after tics 34, 69, ... 524
    assert all(c.desync_tic == 524 for c in session.clients)


def test_wrong_iwad_and_protocol_version_are_refused() -> None:
    clock = Clock()
    net = LoopbackNetwork(clock)
    server = NetServer(net.endpoint(("host", 1)), MD5, clock=clock)
    other = NetClient(
        net.endpoint(("c", 1)),
        ("host", 1),
        name="x",
        player_class=0,
        iwad_md5=bytes(16),
        clock=clock,
    )
    old = net.endpoint(("old", 1))
    old.send(("host", 1), encode(Hello(PROTOCOL_VERSION + 1, MD5, 0, "old")))
    for _ in range(5):
        clock.t += FRAME
        server.poll()
        other.poll()
    assert other.state == ClientState.REJECTED
    assert "IWAD" in other.reject_reason
    replies = [decode(data) for _, data in old.recv()]
    assert any(isinstance(m, Reject) and "protocol version" in m.reason for m in replies)
    assert server.peers == []


def test_a_silent_player_is_dropped_and_the_others_go_on() -> None:
    session = Session(4, delay=0.03, timeout=2.0)
    session.run_until(300)
    gone = session.clients[3].player_num
    session.paused.add(3)  # this client stops sending
    session.run_until(1000, players=[0, 1, 2])
    sims = session.sims([0, 1, 2])
    assert all(not s.players[gone].in_game for s in sims)
    assert all(s.players[n].in_game for s in sims for n in range(4) if n != gone)
    assert len({sim_hash(s) for s in sims}) == 1
    # PlayerQuitGame's message went to each node's console player (it may have expired)
    left = f"PLAYER {gone + 1} LEFT THE GAME"
    viewer = sims[0].players[sims[0].consoleplayer]
    assert left in viewer.message or viewer.message_tics == 0
    assert session.server.desync_tic == -1


def test_deathmatch_over_the_network_stays_in_sync() -> None:
    session = Session(2, delay=0.02, deathmatch=True)
    session.run_until(700)
    sims = session.sims()
    assert sims[0].deathmatch and len({s.state_hash() for s in sims}) == 1


def test_two_players_over_real_udp() -> None:
    server = NetServer(UdpTransport("127.0.0.1", 0), MD5, timeout=5.0)
    host = ("127.0.0.1", server.transport.address[1])
    clients = [
        NetClient(
            UdpTransport("127.0.0.1", 0),
            host,
            name=f"U{i}",
            player_class=i,
            iwad_md5=MD5,
            timeout=5.0,
        )
        for i in range(2)
    ]
    wad = _wad()
    runners: list[LockstepRunner | None] = [None, None]
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            server.poll()
            for i, c in enumerate(clients):
                c.poll()
                if c.state == ClientState.LOBBY and len(server.peers) == 2 and not server.started:
                    server.start()
                if c.state == ClientState.GAME and runners[i] is None:
                    assert c.settings is not None
                    runners[i] = LockstepRunner(
                        c, build_netgame_sim(load_level(wad, "MAP01"), c.settings, c.player_num)
                    )
                runner = runners[i]
                if runner is not None:
                    if c.can_make_cmd() and c.maketic < 160:
                        c.make_cmd(_bot(c.player_num, c.maketic))
                    runner.run_ready(150 - c.run_tic)
            if all(r is not None and r.client.run_tic >= 150 for r in runners):
                break
            time.sleep(0.001)
    finally:
        for c in clients:
            c.transport.close()
        server.transport.close()
    assert all(r is not None and r.client.run_tic == 150 for r in runners)
    assert sim_hash(runners[0].sim) == sim_hash(runners[1].sim)  # type: ignore[union-attr]
