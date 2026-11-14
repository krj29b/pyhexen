# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""S2c: two GameFlows play a netgame over the loopback network through NetGameDriver.

Coop: both players walk east over MAP01's exit line, sit through the intermission (it is
skipped by a ticcmd, as in CheckForSkip) and arrive in MAP02 together. Deathmatch: the
`-timer` ends MAP01 for both. Every frame of the coop game is also rendered headlessly
from each node's console player.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fixtures.wadbuilder import MapBuilder, PatchPlacement, TextureDef, WadBuilder
from pyhexen.app.game_flow import GameFlow, GameState
from pyhexen.app.netgame import NetGameDriver
from pyhexen.core.ticcmd import BT_USE, TicCmd
from pyhexen.net.client import NetClient
from pyhexen.net.server import NetServer
from pyhexen.net.session import sim_hash
from pyhexen.net.transport import LoopbackNetwork
from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
from pyhexen.render.renderer import WorldRenderer
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.wadfile import WadStack

MD5 = b"\x01" * 16
MAPINFO_TEXT = b"""
map 1 "ONE"
warptrans 1
next 2
cluster 1
sky1 SKY1 0

map 2 "TWO"
warptrans 2
next 1
cluster 1
sky1 SKY1 0
"""


def _wad() -> WadStack:
    wb = WadBuilder("IWAD")
    wb.add_playpal(14)
    wb.add_colormap(33, include_fogmap=True)
    wb.add_lump("MAPINFO", MAPINFO_TEXT)
    wb.add_marker("P_START")
    wb.add_picture("WALL01", np.full((64, 64), 160, dtype=np.uint8))
    wb.add_marker("P_END")
    wb.add_textures(
        [TextureDef(name="WALL01", width=64, height=64, patches=[PatchPlacement(patch=0)])],
        pnames=["WALL01"],
    )
    wb.add_marker("F_START")
    wb.add_flat("FLOOR01", fill=30)
    wb.add_flat("CEIL01", fill=50)
    wb.add_marker("F_END")
    for name, target in (("MAP01", 2), ("MAP02", 1)):
        mb = MapBuilder(name)
        mb.add_two_sector_rooms(
            width1=512,
            width2=256,
            height=512,
            floor2=0,
            portal_special=74,
            portal_args=(target, 0, 0, 0, 0),
        )
        mb.add_thing(x=128, y=200, type=1)  # player 1 start, facing east
        mb.add_thing(x=128, y=320, type=2)  # player 2 start
        mb.add_thing(x=64, y=64, type=11)  # deathmatch starts
        mb.add_thing(x=64, y=448, type=11)
        mb.add_thing(x=400, y=64, type=11)
        wb.add_map(mb)
    return WadStack(wb.to_bytes())


@dataclass
class Clock:
    t: float = 0.0

    def __call__(self) -> float:
        return self.t


class Pair:
    """A host driver and a joining driver, each with its own GameFlow."""

    def __init__(self, *, deathmatch: bool = False, timer: int = 0) -> None:
        self.clock = Clock()
        net = LoopbackNetwork(self.clock, delay=0.03, loss=0.05, seed=3)
        server = NetServer(net.endpoint(("host", 1)), MD5, clock=self.clock)
        self.drivers: list[NetGameDriver] = []
        for i in range(2):
            wad = _wad()
            flow = GameFlow(wad, load_mapinfo(wad))
            client = NetClient(
                net.endpoint((f"node{i}", 9)),
                ("host", 1),
                name=f"N{i}",
                player_class=i,
                iwad_md5=MD5,
                clock=self.clock,
            )
            self.drivers.append(
                NetGameDriver(
                    flow,
                    client,
                    server if i == 0 else None,
                    start_players=2,
                    deathmatch=deathmatch,
                    timer_minutes=timer,
                    map_num=1,
                    skill=2,
                )
            )
        self.use_now = False

    def bot(self, driver: NetGameDriver):
        def make() -> TicCmd:
            flow = driver.flow
            if flow.state == GameState.INTERMISSION:
                # tap use (a new press each time, CheckForSkip needs the edge)
                return TicCmd(buttons=BT_USE if driver.client.maketic % 4 == 0 else 0)
            return TicCmd(forwardmove=25)

        return make

    def frame(self) -> None:
        self.clock.t += 1 / 35
        for d in self.drivers:
            d.pump()
            d.tick(1, self.bot(d))

    def run(self, until, max_frames: int = 6000) -> None:
        for _ in range(max_frames):
            if until():
                return
            self.frame()
        raise AssertionError("netgame did not reach the expected state")

    def in_step(self) -> bool:
        a, b = (d.client.run_tic for d in self.drivers)
        return a == b


def test_coop_netgame_moves_both_players_to_the_next_map_and_renders_each_view() -> None:
    pair = Pair()
    flows = [d.flow for d in pair.drivers]
    pair.run(lambda: all(f.state == GameState.LEVEL for f in flows))
    assert [f.consoleplayer for f in flows] == [d.client.player_num for d in pair.drivers]
    assert all(f.sim is not None and f.sim.netgame and len(f.sim.players) == 2 for f in flows)

    with RenderContext(standalone=True, width=64, height=40) as ctx:
        renderer = WorldRenderer(ctx, wad=flows[0].wad)
        rendered_views = 0

        def render(flow: GameFlow) -> None:
            nonlocal rendered_views
            sim = flow.sim
            player = flow.player
            if flow.state != GameState.LEVEL or sim is None or player is None or player.mo is None:
                return
            renderer.load_map(flow.wad, sim.level, None, event_bus=sim.events)
            camera = Camera()
            camera.set_from_fixed(player.mo.x, player.mo.y, player.viewz, player.mo.angle, 0)
            camera.snap_interpolation()
            renderer.render_frame(camera, sim)
            rendered_views += 1

        def arrived() -> bool:
            if pair.clock.t * 35 % 50 < 1:
                for f in flows:
                    render(f)
            return all(
                f.state == GameState.LEVEL and f.sim is not None and f.sim.map_num == 2
                for f in flows
            )

        pair.run(arrived)
        assert rendered_views >= 2

    pair.run(pair.in_step)
    assert sim_hash(flows[0].sim) == sim_hash(flows[1].sim)
    for f in flows:
        players = f.sim.players
        assert all(p.mo is not None and p.in_game for p in players)
        assert f.player is players[f.consoleplayer]


def test_deathmatch_timer_ends_the_map_on_both_nodes() -> None:
    pair = Pair(deathmatch=True, timer=1)
    flows = [d.flow for d in pair.drivers]
    pair.run(lambda: all(f.state == GameState.LEVEL for f in flows))
    assert all(f.sim.deathmatch and f.sim.timer_game == 35 * 60 for f in flows)
    assert all(p.keys == 2047 for f in flows for p in f.sim.players)
    pair.run(lambda: all(f.state == GameState.INTERMISSION for f in flows), max_frames=3000)
    assert all(f.intermission.is_deathmatch for f in flows)
    pair.run(lambda: all(f.state == GameState.LEVEL and f.sim.map_num == 2 for f in flows))
    pair.run(pair.in_step)
    assert sim_hash(flows[0].sim) == sim_hash(flows[1].sim)
    assert all(f.sim.timer_game > 0 for f in flows)  # the timer restarts on every map
