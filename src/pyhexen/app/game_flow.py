# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Game-flow controller: title, new game, levels, intermission, finale, reborn (T15.1).

Port of the G_Ticker game-action state machine in g_game.c (G_DoNewGame, G_InitNew,
G_Completed, G_DoCompleted, G_DoTeleportNewMap, G_DoReborn, G_DoSingleReborn,
G_DoSaveGame, G_DoLoadGame) and the SV_MapTeleport / SV_LoadGame reborn-slot handling
in sv_save.c.

The controller owns the current `GameSim` and `HubManager`. It has no rendering or
input code: the main loop feeds it tic commands and draws whatever `state` says.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.events import MapExit
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.hub import HubManager, remove_all_thinkers
from pyhexen.sim.message import set_message
from pyhexen.sim.player import PlayerState
from pyhexen.sim.savegame import SaveError
from pyhexen.ui.intermission import Finale, Intermission, TitleLoop
from pyhexen.world.level import load_level

if TYPE_CHECKING:
    from pyhexen.sim.player import Player
    from pyhexen.sim.savegame import SaveGameManager
    from pyhexen.wad.mapinfo import MapInfo, MapInfoTable
    from pyhexen.wad.wadfile import WadFile, WadStack
    from pyhexen.world.level import Level


class GameState(IntEnum):
    """What the game is showing (gamestate_t)."""

    TITLE = 0  # GS_DEMOSCREEN
    LEVEL = 1  # GS_LEVEL
    INTERMISSION = 2  # GS_INTERMISSION
    FINALE = 3  # GS_FINALE


class GameAction(IntEnum):
    """Deferred state change, run at the start of the next tic (gameaction_t)."""

    NOTHING = 0  # ga_nothing
    NEW_GAME = 1  # ga_newgame
    COMPLETED = 2  # ga_completed
    LEAVE_MAP = 3  # ga_leavemap
    SINGLE_REBORN = 4  # ga_singlereborn
    VICTORY = 5  # ga_victory
    SAVE_GAME = 6  # ga_savegame
    LOAD_GAME = 7  # ga_loadgame


TXT_GAMESAVED: str = "GAME SAVED"


@dataclass(slots=True)
class RebornSlot:
    """In-memory copy of the vanilla reborn save slot (REBORN_SLOT in sv_save.c).

    Written right after every map change in single player, and loaded when the
    player dies and presses use.
    """

    map_num: int
    sim_data: bytes
    hub_snapshots: dict[int, bytes]
    world_vars: list[int]
    cluster: int


class GameFlow:
    """Drives title → new game → level → exit → intermission → next level, and reborn.

    Port of G_Ticker (g_game.c). Call `tick()` once per game tic (35 Hz).
    """

    def __init__(
        self,
        wad: WadStack | WadFile,
        mapinfo: MapInfoTable | None = None,
        *,
        load_level_fn: Callable[[str], Level] | None = None,
        saves: SaveGameManager | None = None,
    ) -> None:
        self.wad = wad
        self.mapinfo = mapinfo
        self.saves: SaveGameManager | None = saves
        # Vanilla `messageson` menu setting, copied into every sim.
        self.messages_on: bool = True
        self._load_level_fn: Callable[[str], Level] = load_level_fn or (
            lambda name: load_level(wad, name)
        )

        self.state: GameState = GameState.TITLE
        self.sim: GameSim | None = None
        self.hub: HubManager = HubManager()
        self.skill: int = 2
        self.player_class: int = 0
        self.reborn_slot: RebornSlot | None = None

        # Demo playback: a reborn or a map exit ends the demo (G_CheckDemoStatus).
        self.demo_playback: bool = False
        self.demo_finished: bool = False

        self.intermission = Intermission(wad)
        self.finale = Finale(wad)
        self.title = TitleLoop(wad)
        self.intermission.on_done = self._on_intermission_done
        self.finale.on_done = self.start_title
        self.title.on_play_demo = self._on_title_demo

        # Called after a level becomes current: new game, map change, or reborn.
        self.on_level_loaded: Callable[[GameSim, MapInfo | None], None] | None = None
        # Called after a game was written to a save slot.
        self.on_game_saved: Callable[[int], None] | None = None

        self._action: GameAction = GameAction.NOTHING
        self._leave_map: int = 0
        self._leave_position: int = 0
        self._skip_title_demo: bool = False
        self._save_slot: int = 0
        self._save_description: str = ""
        self._load_slot: int = 0
        # Set when the last load failed (bad file, other IWAD...), for the UI to report.
        self.last_load_error: str = ""

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def player(self) -> Player | None:
        """The console player (player 0), or None outside a level."""
        if self.sim is None or not self.sim.players:
            return None
        return self.sim.players[0]

    @property
    def pending_action(self) -> GameAction:
        """The deferred action that the next `tick()` will run."""
        return self._action

    def map_lump(self, map_num: int) -> str:
        """Return the map lump name for `map_num`."""
        info = self.mapinfo.get(map_num) if self.mapinfo is not None else None
        return info.lump_name if info is not None else f"MAP{map_num:02d}"

    def map_cluster(self, map_num: int) -> int:
        """Port of P_GetMapCluster (p_setup.c)."""
        info = self.mapinfo.get(map_num) if self.mapinfo is not None else None
        return info.cluster if info is not None else 1

    def next_map(self, map_num: int) -> int:
        """Port of P_TranslateMap(P_GetMapNextMap(map)) (p_tick.c:72)."""
        info = self.mapinfo.get(map_num) if self.mapinfo is not None else None
        if info is None or self.mapinfo is None:
            return map_num + 1
        real = self.mapinfo.translate_warp(info.next_map)
        return real if real > 0 else map_num + 1

    # ------------------------------------------------------------------
    # Commands from the menu / startup code
    # ------------------------------------------------------------------

    def start_title(self) -> None:
        """Port of H2_StartTitle (h2_main.c): drop the game and show the title loop."""
        self.state = GameState.TITLE
        self.sim = None
        self._action = GameAction.NOTHING
        self.intermission.active = False
        self.finale.active = False
        self.title.start()

    def new_game(self, player_class: int, skill: int) -> None:
        """Start a new game on the next tic (G_DeferedNewGame, from the skill menu)."""
        self.player_class = player_class
        self.skill = skill
        self._action = GameAction.NEW_GAME

    def init_new(self, map_num: int, player_class: int, skill: int) -> None:
        """Start a new game on `map_num` right away (the -warp startup path).

        Port of G_StartNewInit followed by G_InitNew.
        """
        self.player_class = player_class
        self.skill = skill
        self._start_new_init()
        self._load_new_level(map_num)

    def save_game(self, slot: int, description: str) -> None:
        """Save into `slot` on the next tic (G_SaveGame; sendsave becomes ga_savegame)."""
        self._save_slot = slot
        self._save_description = description
        self._action = GameAction.SAVE_GAME

    def load_game(self, slot: int) -> None:
        """Load `slot` on the next tic (G_LoadGame)."""
        self._load_slot = slot
        self._action = GameAction.LOAD_GAME

    def set_messages_on(self, on: bool) -> None:
        """Change the `messageson` setting for this and every later sim."""
        self.messages_on = on
        if self.sim is not None:
            self.sim.messages_on = on

    def enter_level(self, sim: GameSim) -> None:
        """Make an externally built `sim` current (demo playback, free-fly)."""
        self._enter_level(sim)

    # ------------------------------------------------------------------
    # Ticker
    # ------------------------------------------------------------------

    def tick(self, cmds: Sequence[Any] = ()) -> list[Any]:
        """Run one game tic and return the sim events it produced.

        Port of G_Ticker (g_game.c): reborns first, then deferred game actions,
        then the ticker for the current state.
        """
        if self.state == GameState.LEVEL and self.sim is not None:
            for player in self.sim.players:
                if player.player_state == PlayerState.PST_REBORN:
                    self._do_reborn()
                    break

        while self._action != GameAction.NOTHING:
            action = self._action
            self._action = GameAction.NOTHING
            self._run_action(action)

        events: list[Any] = []
        if self.state == GameState.LEVEL and self.sim is not None:
            self.sim.tick(cmds)
            events = self.sim.events.drain()
            for event in events:
                if isinstance(event, MapExit):
                    self._completed(event.next_map, event.position)
        elif self.state == GameState.INTERMISSION:
            self.intermission.tick()
        elif self.state == GameState.FINALE:
            self.finale.tick()
        elif self.state == GameState.TITLE:
            if self._skip_title_demo:
                # Title demos are not played yet; move on to the next page.
                self._skip_title_demo = False
                self.title.advance()
            else:
                self.title.tick()
        return events

    def _run_action(self, action: GameAction) -> None:
        if action == GameAction.NEW_GAME:
            self._do_new_game()
        elif action == GameAction.COMPLETED:
            self._do_completed()
        elif action == GameAction.LEAVE_MAP:
            self._do_leave_map()
        elif action == GameAction.SINGLE_REBORN:
            self._do_single_reborn()
        elif action == GameAction.VICTORY:
            self._do_victory()
        elif action == GameAction.SAVE_GAME:
            self._do_save_game()
        elif action == GameAction.LOAD_GAME:
            self._do_load_game()

    # ------------------------------------------------------------------
    # New game
    # ------------------------------------------------------------------

    def _start_new_init(self) -> None:
        """Port of G_StartNewInit: clear hub slots, the reborn slot and ACS globals."""
        self.hub = HubManager()
        self.reborn_slot = None

    def _do_new_game(self) -> None:
        """Port of G_DoNewGame / G_StartNewGame: start on the map with warp number 1."""
        self._start_new_init()
        real_map = self.mapinfo.translate_warp(1) if self.mapinfo is not None else 1
        if real_map <= 0:
            real_map = 1
        self._load_new_level(real_map)

    def _load_new_level(self, map_num: int) -> None:
        """Port of G_InitNew + G_DoLoadLevel for a fresh game."""
        level = self._load_level_fn(self.map_lump(map_num))
        sim = GameSim(level=level, skill=self.skill, player_class=self.player_class)
        sim.map_num = map_num
        sim.cluster = self.map_cluster(map_num)
        self.hub.current_cluster = sim.cluster
        sim.setup_level(player_nums=(0,))
        self._enter_level(sim)

    def _enter_level(self, sim: GameSim) -> None:
        self.sim = sim
        sim.messages_on = self.messages_on
        sim.next_map = self.next_map(sim.map_num)
        self.state = GameState.LEVEL
        self.title.active = False
        self.intermission.active = False
        self.finale.active = False
        if self.on_level_loaded is not None:
            info = self.mapinfo.get(sim.map_num) if self.mapinfo is not None else None
            self.on_level_loaded(sim, info)

    # ------------------------------------------------------------------
    # Map exits
    # ------------------------------------------------------------------

    def _completed(self, map_num: int, position: int) -> None:
        """Port of G_Completed: remember the destination and defer the exit."""
        self._action = GameAction.COMPLETED
        self._leave_map = map_num
        self._leave_position = position

    def _do_completed(self) -> None:
        """Port of G_DoCompleted: start the finale or the intermission.

        G_PlayerExitMap runs later, inside `HubManager.transition_map`; nothing
        reads player state while the intermission is on screen.
        """
        if self.demo_playback:
            self.demo_finished = True
            return
        assert self.sim is not None
        if self._leave_map == -1 and self._leave_position == -1:
            self._action = GameAction.VICTORY
            return
        for player in self.sim.players:  # IN_Start clears every player's message
            player.message_tics = 0
            player.message = ""
        self.state = GameState.INTERMISSION
        self.intermission.start(
            self.map_cluster(self.sim.map_num),
            self.map_cluster(self._leave_map),
            is_deathmatch=self.sim.deathmatch,
            players=self.sim.players,
        )

    def _on_intermission_done(self) -> None:
        """Port of WaitStop (in_lude.c): leave the map on the next tic."""
        self._action = GameAction.LEAVE_MAP

    def _do_leave_map(self) -> None:
        """Port of G_DoTeleportNewMap → SV_MapTeleport, then the single-player reborn save."""
        assert self.sim is not None
        dest_sim, res = self.hub.transition_map(
            self.sim,
            target_map=self._leave_map,
            position=self._leave_position,
            mapinfo_table=self.mapinfo,
            load_level_cb=lambda num: self._load_level_fn(self.map_lump(num)),
        )
        if res.victory:
            self._action = GameAction.VICTORY
            return
        self._enter_level(dest_sim)
        if not dest_sim.netgame:
            self.reborn_slot = self._make_reborn_slot(dest_sim)

    def _do_victory(self) -> None:
        """Port of F_StartFinale (f_finale.c)."""
        self.state = GameState.FINALE
        netgame = self.sim.netgame if self.sim is not None else False
        self.finale.start(self.player_class, is_netgame=netgame)

    # ------------------------------------------------------------------
    # Death and reborn
    # ------------------------------------------------------------------

    def _do_reborn(self) -> None:
        """Port of G_DoReborn for single player: reload the reborn slot or restart."""
        if self.demo_playback:
            self.demo_finished = True
            return
        if self.reborn_slot is not None:
            self._action = GameAction.SINGLE_REBORN
        else:
            self._action = GameAction.NEW_GAME

    def _make_reborn_slot(self, sim: GameSim) -> RebornSlot:
        """Port of SV_SaveGame(REBORN_SLOT) at the end of SV_MapTeleport."""
        world_vars = list(sim.acs_vm.world_vars) if sim.acs_vm is not None else []
        return RebornSlot(
            map_num=sim.map_num,
            sim_data=sim.save_to_bytes(save_players=True),
            hub_snapshots=dict(self.hub.snapshots),
            world_vars=world_vars or list(self.hub.world_vars),
            cluster=self.hub.current_cluster,
        )

    def _do_single_reborn(self) -> None:
        """Port of G_DoSingleReborn → SV_LoadGame(REBORN_SLOT)."""
        slot = self.reborn_slot
        assert slot is not None
        sim = self._restore_level(
            slot.map_num, slot.cluster, lambda s: s.load_from_bytes(slot.sim_data)
        )
        self.hub.snapshots = dict(slot.hub_snapshots)
        self.hub.world_vars = list(slot.world_vars)
        self.hub.current_cluster = slot.cluster
        if sim.acs_vm is not None and slot.world_vars:
            sim.acs_vm.world_vars = list(slot.world_vars)
        self._enter_level(sim)

    def _restore_level(
        self, map_num: int, cluster: int, unarchive: Callable[[GameSim], None]
    ) -> GameSim:
        """Port of SV_LoadMap (sv_save.c): set the map up, drop its thinkers, unarchive.

        SV_LoadMap runs G_InitNew (full level setup, RNG cleared), removes all
        thinkers, then unarchives. The RNG is not archived, so it keeps its
        post-setup state.
        """
        level = self._load_level_fn(self.map_lump(map_num))
        sim = GameSim(level=level, skill=self.skill, player_class=self.player_class)
        sim.map_num = map_num
        sim.cluster = cluster
        sim.setup_level()
        remove_all_thinkers(sim)
        rng_state = sim.rng.get_state()
        unarchive(sim)
        sim.rng.set_state(rng_state)
        sim.map_num = map_num
        sim.cluster = cluster
        return sim

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------

    def _do_save_game(self) -> None:
        """Port of G_DoSaveGame (g_game.c): SV_SaveGame, then the "GAME SAVED" message."""
        sim = self.sim
        if self.saves is None or self.state != GameState.LEVEL or sim is None:
            return  # vanilla only offers saving inside a level
        if sim.acs_vm is not None and sim.acs_vm.world_vars:
            self.hub.world_vars = list(sim.acs_vm.world_vars)
        self.saves.save_game(self._save_slot, self._save_description, sim, self.hub)
        if self.on_game_saved is not None:
            self.on_game_saved(self._save_slot)
        player = self.player
        if player is not None:
            set_message(sim, player, TXT_GAMESAVED, True)

    def _do_load_game(self) -> None:
        """Port of G_DoLoadGame (g_game.c): SV_LoadGame, then SV_UpdateRebornSlot."""
        if self.saves is None:
            return
        try:
            payload = self.saves.read_game(self._load_slot)
        except (OSError, SaveError) as exc:
            # Not in vanilla (it reads its own files only): report instead of crashing.
            self.last_load_error = str(exc)
            if self.sim is not None and self.player is not None:
                set_message(self.sim, self.player, "CAN'T LOAD THAT GAME", True)
            return
        self.last_load_error = ""
        header = payload["header"]
        hub_data = payload.get("hub", {})
        map_num = int(header.get("map_num", 1))
        cluster = int(header.get("cluster", self.map_cluster(map_num)))
        self.skill = int(header.get("skill", self.skill))
        self.player_class = int(header.get("player_class", self.player_class))

        sim = self._restore_level(map_num, cluster, lambda s: s.from_state(payload["sim"]))
        self.hub = HubManager()
        self.hub.current_cluster = int(hub_data.get("current_cluster", cluster))
        self.hub.world_vars = list(hub_data.get("world_vars", []))
        self.hub.snapshots = {
            int(k): bytes.fromhex(v) for k, v in hub_data.get("snapshots", {}).items()
        }
        if sim.acs_vm is not None and self.hub.world_vars:
            sim.acs_vm.world_vars = list(self.hub.world_vars)
        self._enter_level(sim)
        if not sim.netgame:
            self.reborn_slot = self._make_reborn_slot(sim)

    # ------------------------------------------------------------------
    # Title loop
    # ------------------------------------------------------------------

    def _on_title_demo(self, _lump: str) -> None:
        self._skip_title_demo = True
