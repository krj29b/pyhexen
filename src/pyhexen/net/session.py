# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Running a GameSim from a NetClient's tics (the RunTic part of d_net.c)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.net.client import NetClient
from pyhexen.net.protocol import GameSettings
from pyhexen.sim.demo_trace import trace_tic
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.netplay import player_quit

if TYPE_CHECKING:
    from pyhexen.world.level import Level

HASH_INTERVAL = 35  # tics between desync checks


def sim_hash(sim: GameSim) -> int:
    """32-bit digest of the sim for desync checks (demo-trace mobj and sector hashes)."""
    t = trace_tic(sim)
    return (t.mobj_hash * 31 + t.sector_hash + t.prndindex * 0x9E3779B1) & 0xFFFFFFFF


def build_netgame_sim(level: Level, settings: GameSettings, player_num: int) -> GameSim:
    """A sim for `settings`, seen from `player_num` (its console player)."""
    sim = GameSim(
        level=level,
        skill=settings.skill,
        netgame=True,
        deathmatch=settings.deathmatch,
        player_class=settings.classes[0] if settings.classes else 0,
        player_classes=settings.classes,
        random_class=settings.random_class,
    )
    sim.map_num = settings.map_num
    sim.consoleplayer = player_num
    sim.setup_level(player_nums=range(settings.num_players), timer_minutes=settings.timer_minutes)
    return sim


class LockstepRunner:
    """Feeds complete tics from `client` to `sim`."""

    def __init__(self, client: NetClient, sim: GameSim) -> None:
        self.client = client
        self.sim = sim
        self.in_game = (1 << len(sim.players)) - 1

    def run_ready(self, max_tics: int | None = None) -> int:
        """Run every tic that has arrived (at most `max_tics`); return how many ran."""
        ran = 0
        while max_tics is None or ran < max_tics:
            tic = self.client.take_tic()
            if tic is None:
                break
            number = self.client.run_tic - 1
            gone = self.in_game & ~tic.in_game
            for num in range(len(self.sim.players)):
                if gone & (1 << num):
                    player_quit(self.sim, num)  # PlayerQuitGame, before the tic runs
            self.in_game = tic.in_game
            self.sim.tick(tic.cmds)
            if (number + 1) % HASH_INTERVAL == 0:
                self.client.report_hash(number, sim_hash(self.sim))
            ran += 1
        return ran
