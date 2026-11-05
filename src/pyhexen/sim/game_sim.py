# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Top-level simulation container and ticker ported from p_tick.c."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from pyhexen.core.events import EventBus
from pyhexen.core.random import Random
from pyhexen.sim.thinker import ThinkerList


class GameSim:
    """Owns the simulation state for a level: thinkers, RNG, level time, and event bus.

    Port of P_Ticker state management (p_tick.c).
    """

    def __init__(
        self,
        *,
        rng: Random | None = None,
        events: EventBus | None = None,
        level: Any = None,
    ) -> None:
        self.rng: Random = rng if rng is not None else Random()
        self.events: EventBus = events if events is not None else EventBus()
        self.thinkers: ThinkerList = ThinkerList()
        self.level: Any = level
        self.level_time: int = 0
        self.timer_game: int = 0
        self.paused: bool = False

    def tick(self, cmds: Sequence[Any] = ()) -> None:
        """Port of P_Ticker (p_tick.c). Advances the simulation by one tic (35 Hz)."""
        _ = cmds  # Player command consumption wired in T6.4
        if self.paused:
            return

        if self.timer_game > 0:
            self.timer_game -= 1

        self.thinkers.run_all(self)
        self.level_time += 1

    def state_hash(self) -> str:
        """Compute a deterministic blake2b hex digest over the simulation state."""
        hasher = hashlib.blake2b(digest_size=32)
        state_items: list[Any] = [
            ("level_time", self.level_time),
            ("rng", self.rng.get_state()),
            ("thinkers", tuple(t.state_tuple() for t in self.thinkers)),
        ]
        if self.level is not None and hasattr(self.level, "sectors"):
            sector_tuples = tuple(
                (
                    getattr(s, "floorheight", 0),
                    getattr(s, "ceilingheight", 0),
                    getattr(s, "lightlevel", 0),
                )
                for s in self.level.sectors
            )
            state_items.append(("sectors", sector_tuples))
        hasher.update(repr(tuple(state_items)).encode("utf-8"))
        return hasher.hexdigest()
