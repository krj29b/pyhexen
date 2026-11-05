# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""ACS Store for queued script execution across maps within a hub cluster.

Port of ACSStore and P_CheckACSStore (p_acs.c).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

MAX_ACS_STORE: int = 20


@dataclass(slots=True)
class QueuedScript:
    """Entry in ACSStore."""

    map_num: int
    script_num: int
    args: tuple[int, int, int]


class ACSStore:
    """Manages queued script executions targeted at other maps in the current cluster."""

    def __init__(self) -> None:
        self.entries: list[QueuedScript] = []

    def queue_script(self, map_num: int, script_num: int, args: tuple[int, int, int]) -> bool:
        """Queue a script for another map. Returns False if store is full."""
        if len(self.entries) >= MAX_ACS_STORE:
            return False
        self.entries.append(QueuedScript(map_num=map_num, script_num=script_num, args=args))
        return True

    def run_pending_for_map(self, map_num: int, sim: GameSim) -> None:
        """Port of P_CheckACSStore (p_acs.c). Executes scripts queued for `map_num`."""
        remaining: list[QueuedScript] = []
        for entry in self.entries:
            if entry.map_num == map_num:
                if sim.acs_vm is not None:
                    sim.acs_vm.start_script(entry.script_num, entry.args)
            else:
                remaining.append(entry)
        self.entries = remaining

    def clear(self) -> None:
        """Clear store when switching clusters."""
        self.entries.clear()
