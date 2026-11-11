# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Top-level simulation container and ticker ported from p_tick.c."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pyhexen.core.events import EventBus
from pyhexen.core.random import Random
from pyhexen.core.ticcmd import TicCmd
from pyhexen.sim.acs.store import ACSStore
from pyhexen.sim.acs.vm import ACSVM
from pyhexen.sim.map import MapCheckState
from pyhexen.sim.specials.anim import SurfaceAnimator
from pyhexen.sim.thinker import ThinkerList

if TYPE_CHECKING:
    from pyhexen.sim.mobj import Mobj
    from pyhexen.sim.player import Player
    from pyhexen.world.level import Level, Line, MapThing


class GameSim:
    """Owns the simulation state for a level: thinkers, RNG, level time, and event bus.

    Port of P_Ticker state management (p_tick.c).
    """

    def __init__(
        self,
        *,
        rng: Random | None = None,
        events: EventBus | None = None,
        level: Level | None = None,
        skill: int = 2,
        player_class: int = 0,
        netgame: bool = False,
        deathmatch: bool = False,
        nomonsters: bool = False,
    ) -> None:
        self.rng: Random = rng if rng is not None else Random()
        self.events: EventBus = events if events is not None else EventBus()
        self.thinkers: ThinkerList = ThinkerList()
        self.level: Level | None = level
        self.skill: int = skill
        self.player_class: int = player_class
        self.netgame: bool = netgame
        self.deathmatch: bool = deathmatch
        self.nomonsters: bool = nomonsters
        self.tid_map: dict[int, list[Mobj]] = {}
        self.tagged_lines: dict[int, list[Line]] = {}
        self.player_starts: dict[tuple[int, int], MapThing] = {}
        self.deathmatch_starts: list[MapThing] = []
        self.players: list[Player] = []
        self.surface_animator: SurfaceAnimator = SurfaceAnimator()
        self.map_state: MapCheckState = MapCheckState()
        self.activated_lines: list[tuple[Line, Mobj | None, int]] = []
        self.line_activation_callback: Callable[[Line, Mobj | None, int], bool] | None = None
        self.level_time: int = 0
        self.timer_game: int = 0
        self.paused: bool = False
        # Vanilla `messageson` (a menu setting): non-ultimate messages are dropped when off.
        self.messages_on: bool = True

        self.map_num: int = 1
        self.cluster: int = 1
        # Destinations for exit_normal / exit_secret; 0 means "map_num + 1".
        self.next_map: int = 0
        self.secret_map: int = 0
        if self.level is not None and getattr(self.level, "name", "").startswith("MAP"):
            try:
                self.map_num = int(self.level.name[3:])
            except ValueError:
                self.map_num = 1

        self.acs_store: ACSStore = ACSStore()
        self.acs_vm: ACSVM | None = None
        if self.level is not None and getattr(self.level, "behavior_bytes", b""):
            from pyhexen.sim.acs.loader import parse_behavior

            try:
                behavior = parse_behavior(self.level.behavior_bytes)
                self.acs_vm = ACSVM(behavior)
            except Exception:
                self.acs_vm = None

    def setup_level(self, player_nums: Sequence[int] = ()) -> None:
        """Map-entry setup shared by new games and map changes.

        Port of the level part of G_InitNew (g_game.c) and P_SetupLevel (p_setup.c):
        M_ClearRandom, P_LoadThings, P_LoadACScripts (open scripts), P_SpawnSpecials.
        Players listed in `player_nums` are spawned right after the map things.
        Polyobjects are already initialised by `load_level`.
        """
        from pyhexen.sim.mobj import spawn_map_things
        from pyhexen.sim.player import spawn_player
        from pyhexen.sim.specials.sector_specials import p_spawn_specials

        self.rng.clear()
        spawn_map_things(self)
        for num in player_nums:
            spawn_player(self, num)
        if self.acs_vm is not None:
            self.acs_vm.start_open_scripts(self)
        p_spawn_specials(self)

    def setup_level_specials(self) -> None:
        """Port of P_SpawnSpecials (p_spec.c)."""
        from pyhexen.sim.specials.sector_specials import p_spawn_specials

        p_spawn_specials(self)
        if self.acs_vm is not None:
            self.acs_store.run_pending_for_map(self.map_num, self)
            self.acs_vm.start_open_scripts(self)

    def tag_finished(self, tag: int) -> None:
        """Port of P_TagFinished (p_acs.c)."""
        if self.acs_vm is not None:
            self.acs_vm.tag_finished(tag, self)

    def poly_finished(self, po: int) -> None:
        """Port of P_PolyobjFinished (p_acs.c)."""
        if self.acs_vm is not None:
            self.acs_vm.poly_finished(po, self)

    def find_mobjs_by_tid(self, tid: int) -> list[Mobj]:
        """Return all active (non-removed) `Mobj`s registered with `tid`."""
        if tid == 0:
            return []
        return [m for m in self.tid_map.get(tid, ()) if not m.removed]

    def on_line_activated(
        self,
        line: Line,
        mobj: Mobj | None,
        spac: int,
        *,
        side: int = 0,
    ) -> bool:
        """Hook invoked when `line` is crossed, pushed, used, or impacted by `mobj`."""
        from pyhexen.sim.line_activation import p_activate_line

        self.activated_lines.append((line, mobj, spac))
        if self.line_activation_callback is not None:
            return self.line_activation_callback(line, mobj, spac)
        return p_activate_line(self, line, mobj, side, spac)

    def tick(self, cmds: Sequence[Any] = ()) -> None:
        """Port of P_Ticker (p_tick.c). Advances the simulation by one tic (35 Hz)."""
        from pyhexen.sim.player import p_player_think
        from pyhexen.sim.specials.anim import p_animate_surfaces

        if self.paused:
            return

        if self.timer_game > 0:
            self.timer_game -= 1
            if self.timer_game == 0:
                from pyhexen.sim.transitions import exit_normal

                exit_normal(self, position=0)

        for idx, player in enumerate(self.players):
            if idx < len(cmds) and isinstance(cmds[idx], TicCmd):
                player.cmd = cmds[idx]
            p_player_think(self, player)

        if self.level is not None and hasattr(self.level, "polyobjs"):
            poly_iterable = (
                self.level.polyobjs.values()
                if isinstance(self.level.polyobjs, dict)
                else self.level.polyobjs
            )
            for po in poly_iterable:
                if hasattr(po, "capture_prev_state"):
                    po.capture_prev_state()

        self.thinkers.run_all(self)
        p_animate_surfaces(self)
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
                    getattr(s, "floor_height", getattr(s, "floorheight", 0)),
                    getattr(s, "ceiling_height", getattr(s, "ceilingheight", 0)),
                    getattr(s, "light_level", getattr(s, "lightlevel", 0)),
                )
                for s in self.level.sectors
            )
            state_items.append(("sectors", sector_tuples))
        hasher.update(repr(tuple(state_items)).encode("utf-8"))
        return hasher.hexdigest()

    def to_state(self, save_players: bool = True) -> dict[str, Any]:
        """Serialize simulation state to a dict (T13.2)."""
        from pyhexen.sim.serialization import serialize_sim

        return serialize_sim(self, save_players=save_players)

    def from_state(self, state: dict[str, Any]) -> None:
        """Restore simulation state from a dict (T13.2)."""
        from pyhexen.sim.serialization import deserialize_sim

        deserialize_sim(self, state)

    def save_to_bytes(self, save_players: bool = True) -> bytes:
        """Serialize simulation state to compressed bytes (T13.2)."""
        from pyhexen.sim.serialization import save_sim_to_bytes

        return save_sim_to_bytes(self, save_players=save_players)

    def load_from_bytes(self, data: bytes) -> None:
        """Restore simulation state from compressed bytes (T13.2)."""
        from pyhexen.sim.serialization import load_sim_from_bytes

        load_sim_from_bytes(self, data)
