# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Hub persistence and level snapshot management (T13.3).

Port of SV_MapTeleport, SV_SaveMap, SV_LoadMap, ClearSaveSlot in sv_save.c,
and G_Completed in g_game.c.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pyhexen.sim.mobj import Mobj, spawn_map_things
from pyhexen.sim.transitions import (
    MapTransitionResult,
    execute_map_transition,
    spawn_player_at_transition,
)

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.wad.mapinfo import MapInfoTable
    from pyhexen.world.level import Level


class HubManager:
    """Manages level snapshots and cross-map persistence within hubs and clusters.

    Port of hub save slot management (sv_save.c:2186-2360).
    """

    def __init__(self, current_cluster: int = 1) -> None:
        self.current_cluster: int = current_cluster
        self.snapshots: dict[int, bytes] = {}
        self.world_vars: list[int] = []

    def save_map(self, sim: GameSim, map_num: int | None = None) -> bytes:
        """Serialize current map to compressed bytes and store in hub snapshots.

        Players are excluded (`save_players=False`) so they can carry over independently.
        """
        target_map = map_num if map_num is not None else getattr(sim, "map_num", 1)
        if sim.acs_vm is not None and sim.acs_vm.world_vars:
            self.world_vars = list(sim.acs_vm.world_vars)

        data = sim.save_to_bytes(save_players=False)
        self.snapshots[target_map] = data
        return data

    def has_map(self, map_num: int) -> bool:
        """Return True if a snapshot exists for `map_num`."""
        return map_num in self.snapshots

    def load_map(self, sim: GameSim, map_num: int) -> None:
        """Restore simulation state from stored snapshot for `map_num`."""
        if map_num not in self.snapshots:
            raise KeyError(f"No snapshot for map {map_num} in hub")

        data = self.snapshots[map_num]
        sim.load_from_bytes(data)
        if sim.acs_vm is not None and self.world_vars:
            sim.acs_vm.world_vars = list(self.world_vars)

    def clear(self) -> None:
        """Clear all stored map snapshots (called when changing clusters).

        Port of ClearSaveSlot(BASE_SLOT) (sv_save.c:2215).
        """
        self.snapshots.clear()

    def transition_map(
        self,
        sim: GameSim,
        target_map: int,
        position: int = 0,
        mapinfo_table: MapInfoTable | None = None,
        load_level_cb: Callable[[int], Level] | None = None,
        target_sim: GameSim | None = None,
    ) -> tuple[GameSim, MapTransitionResult]:
        """Orchestrate transition from current level to destination map in a hub.

        Port of SV_MapTeleport (sv_save.c:2190).
        """
        res = execute_map_transition(
            sim, target_map=target_map, position=position, mapinfo_table=mapinfo_table
        )
        if res.victory:
            return sim, res

        # Preserve the player structures carrying over
        players = list(sim.players)

        # Sync world vars
        if sim.acs_vm is not None and sim.acs_vm.world_vars:
            self.world_vars = list(sim.acs_vm.world_vars)

        if res.cluster_changed:
            # Entering new cluster - clear all previous snapshots and ACS store
            self.clear()
            sim.acs_store.clear()
            self.current_cluster = res.new_cluster
        else:
            # Same cluster - save the leaving map state
            self.save_map(sim, sim.map_num)

        # Prepare destination GameSim
        if target_sim is not None:
            dest_sim = target_sim
        elif load_level_cb is not None:
            from pyhexen.sim.game_sim import GameSim as SimClass

            dest_level = load_level_cb(res.target_map)
            dest_sim = SimClass(
                level=dest_level,
                rng=sim.rng,
                skill=sim.skill,
                player_class=sim.player_class,
                netgame=sim.netgame,
                deathmatch=sim.deathmatch,
                nomonsters=sim.nomonsters,
            )
        else:
            raise ValueError("Either target_sim or load_level_cb must be provided")

        dest_sim.map_num = res.target_map
        dest_sim.cluster = res.new_cluster
        dest_sim.acs_store = sim.acs_store

        # Check if destination map has been visited before in this hub
        if self.has_map(res.target_map):
            self.load_map(dest_sim, res.target_map)
        else:
            spawn_map_things(dest_sim)
            if dest_sim.level is not None and getattr(dest_sim.level, "behavior_bytes", None):
                from pyhexen.sim.acs import parse_behavior
                from pyhexen.sim.acs.vm import ACSVM

                beh = parse_behavior(dest_sim.level.behavior_bytes)
                dest_sim.acs_vm = ACSVM(beh)
                if self.world_vars:
                    dest_sim.acs_vm.world_vars = list(self.world_vars)

        # Position carrying-over players at destination start spot
        for p_idx, player in enumerate(players):
            spawn_player_at_transition(dest_sim, player, player_num=p_idx, position=res.position)

        # Redirect any mobjs whose target was player
        if dest_sim.players and dest_sim.players[0].mo is not None:
            p_mo = dest_sim.players[0].mo
            for thinker in dest_sim.thinkers:
                if isinstance(thinker, Mobj):
                    if getattr(thinker, "target", None) is None and getattr(
                        thinker, "_needs_player_target", False
                    ):
                        thinker.target = p_mo

        # Run any pending deferred ACS scripts targeted at this map
        dest_sim.acs_store.run_pending_for_map(res.target_map, dest_sim)

        return dest_sim, res
