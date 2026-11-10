# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Map transitions, carry-over rules, and spawn positioning (T13.1).

Port of G_Completed, G_PlayerExitMap, G_DoTeleportNewMap, SV_MapTeleport in g_game.c
and sv_save.c, and P_TranslateMap in p_setup.c.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyhexen.core.angles import ANG45
from pyhexen.core.events import MapExit
from pyhexen.core.fixed import FRACBITS, c_div, to_uint32
from pyhexen.sim.artifacts import ArtiType
from pyhexen.sim.info_data import MobjFlag
from pyhexen.sim.player import (
    NUMPOWERS,
    PW_FLIGHT,
    Player,
    PlayerState,
    spawn_player,
)

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.wad.mapinfo import MapInfoTable


@dataclass(slots=True, frozen=True)
class MapTransitionResult:
    """Outcome of a map transition evaluation.

    Attributes:
        target_map: The resolved destination map number (or -1 for endgame).
        position: The destination player start position index.
        cluster_changed: True if entering a different hub/cluster.
        old_cluster: Cluster index of the originating map.
        new_cluster: Cluster index of the destination map.
        victory: True if this transition triggers game victory/endgame finale.
    """

    target_map: int
    position: int
    cluster_changed: bool
    old_cluster: int
    new_cluster: int
    victory: bool = False


def resolve_target_map(
    map_arg: int,
    mapinfo_table: MapInfoTable | None = None,
) -> int:
    """Resolve destination map number, taking warp translation into account.

    Port of P_TranslateMap (p_setup.c:979).
    If `map_arg <= 0`, it is preserved (e.g. -1 for endgame).
    If `mapinfo_table` provides a `translate_warp` match for `map_arg`, that map
    is returned; otherwise `map_arg` is used directly.
    """
    if map_arg <= 0:
        return map_arg

    if mapinfo_table is not None:
        translated = mapinfo_table.translate_warp(map_arg)
        if translated > 0:
            return translated
        if map_arg in mapinfo_table:
            return map_arg

    return map_arg


def player_exit_map(
    player: Player,
    old_cluster: int,
    new_cluster: int,
    *,
    deathmatch: bool = False,
) -> None:
    """Apply carry-over rules when a player exits a map.

    Port of G_PlayerExitMap (g_game.c:1263).
    - Health, max health, class, armor, mana, weapons, and normal artifacts carry over.
    - Strips temporary powers, retaining flight power only if staying in the same cluster.
    - If entering a new cluster (different hub): clears keys, clears flight power,
      and strips flight and puzzle artifacts.
    - Clears morph (pig) state and resets view/palette/status counters.
    """
    # Retain flight power across levels in the same cluster; strip others
    flight_power = player.powers[PW_FLIGHT] if len(player.powers) > PW_FLIGHT else 0
    player.powers = [0] * NUMPOWERS
    player.powers[PW_FLIGHT] = flight_power

    if deathmatch:
        player.powers[PW_FLIGHT] = 0
    else:
        if old_cluster != new_cluster:
            # Entering new cluster (new hub):
            # 1. Strip all keys
            player.keys = 0

            # 2. Strip flight power
            player.powers[PW_FLIGHT] = 0

            # 3. Strip flight artifacts and puzzle keys/items from inventory
            new_inv = [
                slot
                for slot in player.inventory
                if slot.type != ArtiType.ARTI_FLY and slot.type < ArtiType.ARTI_FIRSTPUZZITEM
            ]
            player.inventory = new_inv
            player.artifact_count = sum(slot.count for slot in player.inventory)
            if player.inventory:
                player.inv_ptr = min(player.inv_ptr, len(player.inventory) - 1)
                player.ready_artifact = player.inventory[player.inv_ptr].type
            else:
                player.inv_ptr = 0
                player.ready_artifact = ArtiType.ARTI_NONE

    # Unmorph if currently morphed (e.g. pig)
    if player.morph_tics:
        player.player_class = player.original_class
        player.morph_tics = 0

    # Reset transient visual / message states
    player.message_tics = 0
    player.lookdir = 0
    if player.mo is not None:
        player.mo.flags &= ~MobjFlag.MF_SHADOW
    player.extralight = 0
    player.fixedcolormap = 0
    player.damagecount = 0
    player.bonuscount = 0
    player.poisoncount = 0


def spawn_player_at_transition(
    sim: GameSim,
    player: Player,
    player_num: int = 0,
    position: int = 0,
) -> None:
    """Reposition or spawn `player` at the destination map start matching `position`.

    Port of SV_MapTeleport player placement (sv_save.c:2297).
    """
    # Look for matching player start (player_num, position)
    mthing = sim.player_starts.get((player_num, position))
    if mthing is None and position != 0:
        mthing = sim.player_starts.get((player_num, 0))
    if mthing is None:
        for (pnum, _pos), st in sim.player_starts.items():
            if pnum == player_num:
                mthing = st
                break
    if mthing is None:
        mthing = sim.player_starts.get((0, position)) or sim.player_starts.get((0, 0))

    if mthing is not None:
        spawn_x = mthing.x << FRACBITS
        spawn_y = mthing.y << FRACBITS
        spawn_angle = to_uint32(ANG45 * c_div(mthing.angle, 45))
    else:
        spawn_x = 0
        spawn_y = 0
        spawn_angle = 0

    if player.mo is not None:
        mo = player.mo
        if sim.level is not None:
            from pyhexen.sim.maputl import set_thing_position, unset_thing_position

            unset_thing_position(sim.level, mo)

        mo.x = spawn_x
        mo.y = spawn_y
        mo.angle = spawn_angle
        mo.momx = 0
        mo.momy = 0
        mo.momz = 0

        if sim.level is not None:
            ss = sim.level.point_in_subsector(spawn_x, spawn_y)
            mo.subsector = ss
            mo.floorz = ss.sector.floor_height
            mo.ceilingz = ss.sector.ceiling_height
            mo.z = mo.floorz
            set_thing_position(sim.level, mo)
        else:
            mo.z = 0

        player.viewz = mo.z + player.viewheight
        player.prev_viewz = player.viewz
        player.player_state = PlayerState.PST_LIVE
        mo.health = player.health
        mo.capture_prev_state()
        if mo not in sim.thinkers:
            sim.thinkers.add(mo)
    else:
        spawned = spawn_player(
            sim,
            player_num=player_num,
            start_spot=position,
            x=spawn_x,
            y=spawn_y,
            angle=spawn_angle,
        )
        player.mo = spawned.mo
        if player.mo is not None:
            player.mo.player = player

    if player not in sim.players:
        sim.players.append(player)


def execute_map_transition(
    sim: GameSim,
    target_map: int,
    position: int = 0,
    mapinfo_table: MapInfoTable | None = None,
) -> MapTransitionResult:
    """Execute level completion and prepare player carry-over state.

    Port of G_Completed / G_DoCompleted (g_game.c:1647, 1661) and SV_MapTeleport (sv_save.c:2190).
    """
    if target_map == -1 and position == -1:
        sim.paused = True
        return MapTransitionResult(
            target_map=-1,
            position=-1,
            cluster_changed=False,
            old_cluster=0,
            new_cluster=0,
            victory=True,
        )

    # `target_map` is a real map number: G_Completed and SV_MapTeleport never
    # translate it. Only the next-map exit path translates warp numbers (exit_normal).
    real_target_map = target_map

    old_cluster = 1
    new_cluster = 1
    if mapinfo_table is not None:
        old_info = mapinfo_table.get(sim.map_num)
        if old_info is not None:
            old_cluster = old_info.cluster
        new_info = mapinfo_table.get(real_target_map)
        if new_info is not None:
            new_cluster = new_info.cluster
    else:
        old_cluster = getattr(sim, "cluster", 1)
        new_cluster = getattr(sim, "cluster", 1)

    cluster_changed = old_cluster != new_cluster

    for player in sim.players:
        player_exit_map(player, old_cluster, new_cluster, deathmatch=sim.deathmatch)

    return MapTransitionResult(
        target_map=real_target_map,
        position=position,
        cluster_changed=cluster_changed,
        old_cluster=old_cluster,
        new_cluster=new_cluster,
        victory=False,
    )


def teleport_new_map(sim: GameSim, map_num: int, position: int = 0) -> bool:
    """Initiate a map transition (line special 74 / ACS Teleport_NewMap).

    Port of G_Completed(map, position) (g_game.c:1647).
    """
    sim.events.emit(MapExit(next_map=map_num, position=position))
    return True


def teleport_end_game(sim: GameSim) -> bool:
    """Initiate game ending / victory finale (line special 75 / ACS Teleport_EndGame).

    Port of p_spec.c:710.
    """
    if sim.deathmatch:
        sim.events.emit(MapExit(next_map=1, position=0))
    else:
        sim.events.emit(MapExit(next_map=-1, position=-1))
    return True


def exit_normal(sim: GameSim, position: int = 0) -> bool:
    """Exit to the next level in sequence (special 243 / TimerGame expiry).

    Port of p_tick.c:72, `G_Completed(P_TranslateMap(P_GetMapNextMap(gamemap)), 0)`.
    The sim has no MAPINFO, so the game-flow controller stores the already
    translated next map in `sim.next_map` when it loads a level.
    """
    next_map = getattr(sim, "next_map", 0) or (sim.map_num + 1)
    sim.events.emit(MapExit(next_map=next_map, position=position, secret=False))
    return True


def exit_secret(sim: GameSim, position: int = 0) -> bool:
    """Exit to the secret level (special 244)."""
    secret_map = getattr(sim, "secret_map", 0) or getattr(sim, "next_map", 0) or (sim.map_num + 1)
    sim.events.emit(MapExit(next_map=secret_map, position=position, secret=True))
    return True
