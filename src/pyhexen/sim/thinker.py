# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Thinker doubly-linked list and deferred removal ported from p_tick.c."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


class Thinker:
    """Base class for all simulation thinkers.

    Port of thinker_t (h2def.h / p_tick.c).
    """

    __slots__ = ("_prev", "_next", "removed")

    def __init__(self) -> None:
        self._prev: Thinker | None = None
        self._next: Thinker | None = None
        self.removed: bool = False

    def tick(self, sim: GameSim) -> None:
        """Execute one 35 Hz simulation tic for this thinker."""

    def state_tuple(self) -> tuple[Any, ...]:
        """Return deterministic state tuple for `GameSim.state_hash()`."""
        return (type(self).__name__,)


class _SentinelThinker(Thinker):
    """Sentinel head/tail cap (`thinkercap` in p_tick.c)."""

    __slots__ = ()


class ThinkerList:
    """Doubly-linked circular list of thinkers with deferred removal.

    Port of P_InitThinkers, P_AddThinker, P_RemoveThinker, and RunThinkers (p_tick.c).
    """

    def __init__(self) -> None:
        self._cap = _SentinelThinker()
        self.init_thinkers()

    def init_thinkers(self) -> None:
        """Port of P_InitThinkers (p_tick.c). Resets the thinker list to empty."""
        self._cap._prev = self._cap
        self._cap._next = self._cap

    def clear(self) -> None:
        """Reset the thinker list to empty."""
        self.init_thinkers()

    def add(self, thinker: Thinker) -> None:
        """Port of P_AddThinker (p_tick.c). Appends `thinker` at the end of the list."""
        thinker.removed = False
        prev = self._cap._prev
        assert prev is not None
        prev._next = thinker
        thinker._next = self._cap
        thinker._prev = prev
        self._cap._prev = thinker

    def remove(self, thinker: Thinker) -> None:
        """Port of P_RemoveThinker (p_tick.c).

        Deallocation is lazy: marks the thinker as removed so it is unlinked on its next turn
        in `run_all()`.
        """
        thinker.removed = True

    def run_all(self, sim: GameSim) -> None:
        """Port of RunThinkers (p_tick.c).

        Iterates through thinkers in insertion order. Thinkers marked `removed` are unlinked
        when encountered; active thinkers have `tick(sim)` called.
        """
        current = self._cap._next
        while current is not None and current is not self._cap:
            if current.removed:
                nxt = current._next
                prev = current._prev
                assert nxt is not None and prev is not None
                nxt._prev = prev
                prev._next = nxt
                current._prev = None
                current._next = None
                current = nxt
            else:
                current.tick(sim)
                nxt = current._next
                current = nxt

    def __iter__(self) -> Iterator[Thinker]:
        """Yield active (non-removed) thinkers in execution order."""
        current = self._cap._next
        while current is not None and current is not self._cap:
            nxt = current._next
            if not current.removed:
                yield current
            current = nxt

    def __len__(self) -> int:
        """Return count of active (non-removed) thinkers."""
        return sum(1 for _ in self)
