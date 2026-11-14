# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Running a GameSim from a NetClient's tics (the RunTic part of d_net.c)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pyhexen.core.ticcmd import TicCmd
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
    """Feeds complete tics from `client` to a game.

    The game is a bare `GameSim`, or anything with a tick function and a current sim
    (the app's GameFlow, whose sim changes on every map change): see `for_game`.
    """

    def __init__(self, client: NetClient, sim: GameSim) -> None:
        self.client = client
        self._tick: Callable[[list[TicCmd]], Any] = sim.tick
        self._current: Callable[[], GameSim | None] = lambda: sim
        self.in_game = (1 << len(sim.players)) - 1

    @classmethod
    def for_game(
        cls,
        client: NetClient,
        tick: Callable[[list[TicCmd]], Any],
        current_sim: Callable[[], GameSim | None],
        num_players: int,
    ) -> LockstepRunner:
        runner = cls.__new__(cls)
        runner.client = client
        runner._tick = tick
        runner._current = current_sim
        runner.in_game = (1 << num_players) - 1
        return runner

    @property
    def sim(self) -> GameSim:
        sim = self._current()
        assert sim is not None
        return sim

    def run_ready(self, max_tics: int | None = None) -> list[Any]:
        """Run every tic that has arrived (at most `max_tics`); return what each tick returned."""
        results: list[Any] = []
        while max_tics is None or len(results) < max_tics:
            tic = self.client.take_tic()
            if tic is None:
                break
            number = self.client.run_tic - 1
            gone = self.in_game & ~tic.in_game
            sim = self._current()
            if gone and sim is not None:
                for num in range(len(sim.players)):
                    if gone & (1 << num):
                        player_quit(sim, num)  # PlayerQuitGame, before the tic runs
            self.in_game = tic.in_game
            results.append(self._tick(tic.cmds))
            if (number + 1) % HASH_INTERVAL == 0:
                sim = self._current()
                self.client.report_hash(number, sim_hash(sim) if sim is not None else 0)
        return results
