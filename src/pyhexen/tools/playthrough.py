# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Scripted playthrough checklist (T15.7), driven through the real game flow.

Every check plays the game through `GameFlow` (the same controller the window uses),
with player input given as ticcmds, instead of calling the hub manager directly:

* **exits**: for every exit line (Teleport_NewMap / Teleport_EndGame / Exit_*), start
  the map, walk the player over the line from its front side, sit through the
  intermission and check that the right map is loaded with the player at the start
  for that position (or that the finale starts). Exits made by ACS scripts are found
  in the BEHAVIOR bytecode and checked by running the script.
* **hubs**: in each hub, kill every monster in one map, leave, come back through an
  exit of the other map, and check the monsters are still dead and keys were kept;
  then cross a cluster boundary and check keys are stripped (G_PlayerExitMap).
* **soak**: on every map, for every class, a bot walks, turns when stuck, presses use,
  fires at monsters it can see and cycles weapons for a while. It checks the player
  stays inside the map and, with `--render`, draws frames headlessly.
* **save/load**: after the soak, save through the flow, play on, load, replay the same
  input and compare per-tic state hashes (demo-trace format).
* **reborn**: die after a map change, press use, and check the reborn slot reloads.

    uv run python -m pyhexen.tools.playthrough --iwad HEXEN.WAD
    uv run python -m pyhexen.tools.playthrough --iwad HEXEN.WAD --only exits --maps 1,2
"""

from __future__ import annotations

import argparse
import struct
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyhexen.app.game_flow import GameFlow, GameState
from pyhexen.app.savegame import SaveGameManager
from pyhexen.core.angles import (
    ANG1,
    ANG90,
    ANG180,
    ANGLETOFINESHIFT,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.events import MapExit
from pyhexen.core.fixed import FRACBITS, FRACUNIT, fixed_mul, to_int32
from pyhexen.core.ticcmd import BT_ATTACK, BT_CHANGE, BT_USE, BT_WEAPONSHIFT, TicCmd
from pyhexen.sim.acs.pcodes import PCode
from pyhexen.sim.demo_trace import trace_tic
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjFlag
from pyhexen.sim.inter import damage_mobj
from pyhexen.sim.line_activation import p_activate_line
from pyhexen.sim.map import check_on_mobj, check_position, teleport_move
from pyhexen.sim.mobj import Mobj
from pyhexen.sim.player import CF_GODMODE, PlayerState
from pyhexen.sim.sight import check_sight
from pyhexen.wad.mapinfo import MapInfoTable, load_mapinfo
from pyhexen.wad.wadfile import WadFile, WadStack
from pyhexen.world.level import SPAC_CROSS, Line

CLASS_NAMES: tuple[str, ...] = ("Fighter", "Cleric", "Mage")
EXIT_SPECIALS: dict[int, str] = {
    74: "Teleport_NewMap",
    75: "Teleport_EndGame",
    243: "Exit_Normal",
    244: "Exit_Secret",
}
PLAYER_START_TYPES = (1, 2, 3, 4, 9100, 9101, 9102, 9103)
ALL_KEYS = 0x7FF
WALK_SPEED = 25  # forwardmove for walking (forwardmove[0] in g_game.c)
MAX_TURN = 1280  # angleturn per tic (angle >> 16)


@dataclass(slots=True)
class Check:
    """One checklist line."""

    kind: str
    name: str
    status: str  # PASS, WARN or FAIL
    detail: str = ""

    def line(self) -> str:
        return f"[{self.status}] {self.kind}: {self.name}" + (
            f" - {self.detail}" if self.detail else ""
        )


@dataclass(slots=True)
class ExitDef:
    """An exit found in a map: a line special, or an ACS script calling one."""

    map_num: int
    special: int
    args: tuple[int, int]
    line_index: int = -1
    script: int = -1
    # False for an ACS LSPEC that takes its arguments from the stack.
    args_known: bool = True


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        print(check.line(), flush=True)
        return check

    def count(self, status: str) -> int:
        return sum(1 for c in self.checks if c.status == status)


class _Xorshift:
    """Small deterministic generator for bot decisions (never the game's RNG)."""

    def __init__(self, seed: int) -> None:
        self.state = (seed * 2654435761) & 0xFFFFFFFF or 1

    def next(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x


def _angle_delta(target: int, current: int) -> int:
    """Signed BAM difference target - current."""
    return to_int32((target - current) & 0xFFFFFFFF)


def _turn_cmd(delta: int) -> int:
    turn = delta >> 16
    return max(-MAX_TURN, min(MAX_TURN, turn))


class Playthrough:
    """Runs the checklist against one IWAD."""

    def __init__(
        self,
        wad: WadStack | WadFile,
        mapinfo: MapInfoTable,
        *,
        save_dir: Path,
        render: bool = False,
        skill: int = 2,
    ) -> None:
        self.wad = wad
        self.mapinfo = mapinfo
        self.save_dir = save_dir
        self.skill = skill
        self.report = Report()
        self.render = render
        self._renderer: Any = None
        self._ctx: Any = None
        lumps = {lmp.name for lmp in getattr(wad, "lumps", [])}
        self.maps: list[int] = sorted(
            num for num, info in mapinfo.items() if not lumps or info.lump_name in lumps
        )

    # ------------------------------------------------------------------
    # Flow helpers
    # ------------------------------------------------------------------

    def new_flow(self) -> GameFlow:
        flow = GameFlow(self.wad, self.mapinfo, saves=SaveGameManager(self.save_dir))
        flow.set_messages_on(False)
        if self.render:
            flow.on_level_loaded = self._on_level_loaded
        return flow

    def _on_level_loaded(self, sim: GameSim, info: Any) -> None:
        renderer = self._get_renderer()
        assert sim.level is not None
        renderer.load_map(self.wad, sim.level, info, event_bus=sim.events)

    def _get_renderer(self) -> Any:
        if self._renderer is None:
            from pyhexen.render.context import RenderContext
            from pyhexen.render.renderer import WorldRenderer

            self._ctx = RenderContext(standalone=True, width=320, height=200)
            self._renderer = WorldRenderer(self._ctx, wad=self.wad)
        return self._renderer

    def draw(self, flow: GameFlow) -> None:
        if not self.render or flow.sim is None or flow.state != GameState.LEVEL:
            return
        from pyhexen.render.camera import Camera

        player = flow.sim.players[0]
        mo = player.mo
        if mo is None:
            return
        camera = Camera()
        camera.set_from_fixed(mo.x, mo.y, player.viewz, mo.angle, player.lookdir)
        camera.snap_interpolation()
        self._get_renderer().render_frame(camera, flow.sim)

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
            self._renderer = None

    def start(self, map_num: int, player_class: int) -> GameFlow:
        flow = self.new_flow()
        flow.init_new(map_num, player_class, self.skill)
        flow.tick([TicCmd()])
        return flow

    @staticmethod
    def player_mo(flow: GameFlow) -> Mobj:
        assert flow.sim is not None
        mo = flow.sim.players[0].mo
        assert mo is not None
        return mo

    def finish_transition(self, flow: GameFlow, max_tics: int = 35 * 60) -> int:
        """Tick through the intermission (pressing a key) until a level or finale is up."""
        tics = 0
        while tics < max_tics:
            if flow.state == GameState.INTERMISSION:
                flow.intermission.handle_key(0)
            flow.tick([TicCmd()])
            tics += 1
            if flow.state in (GameState.LEVEL, GameState.FINALE) and not flow.pending_action:
                return tics
        return tics

    # ------------------------------------------------------------------
    # Exit discovery
    # ------------------------------------------------------------------

    def find_exits(self, map_num: int) -> list[ExitDef]:
        flow = self.start(map_num, 0)
        sim = flow.sim
        assert sim is not None and sim.level is not None
        exits = [
            ExitDef(map_num, ld.special, (ld.args[0], ld.args[1]), line_index=i)
            for i, ld in enumerate(sim.level.lines)
            if ld.special in EXIT_SPECIALS
        ]
        behavior = sim.acs_vm.behavior if sim.acs_vm is not None else None
        if behavior is not None:
            exits.extend(_acs_exits(map_num, behavior))
        return exits

    def expected_map(self, exit_def: ExitDef) -> int:
        if exit_def.special == 74:
            return exit_def.args[0]  # G_Completed(args[0], args[1]): no warp translation
        if exit_def.special == 75:
            return -1
        info = self.mapinfo.get(exit_def.map_num)
        if exit_def.special == 244 and info is not None:
            return self.mapinfo.translate_warp(info.secret_map)
        return self.mapinfo.translate_warp(info.next_map) if info is not None else -1

    # ------------------------------------------------------------------
    # Crossing a line
    # ------------------------------------------------------------------

    def walk_over_line(self, flow: GameFlow, line: Line) -> tuple[bool, str]:
        """Put the player in front of `line` and walk over it. Returns (exited, how)."""
        sim = flow.sim
        assert sim is not None
        mo = self.player_mo(flow)
        midx = line.v1.x + (line.dx >> 1)
        midy = line.v1.y + (line.dy >> 1)
        normal = (point_to_angle2(0, 0, line.dx, line.dy) - ANG90) & 0xFFFFFFFF
        fine = normal >> ANGLETOFINESHIFT
        placed = False
        for dist in (24, 32, 40, 56, 72, 96):
            x = midx + fixed_mul(dist * FRACUNIT, fine_cosine(fine))
            y = midy + fixed_mul(dist * FRACUNIT, fine_sine(fine))
            if teleport_move(sim, mo, x, y):
                placed = True
                break
        if placed:
            mo.angle = (normal + ANG180) & 0xFFFFFFFF
            mo.momx = mo.momy = 0
            mo.z = mo.floorz
            for _ in range(105):
                events = flow.tick([TicCmd(forwardmove=WALK_SPEED)])
                if any(isinstance(e, MapExit) for e in events) or flow.pending_action:
                    return True, "walked"
                if flow.state != GameState.LEVEL:
                    return True, "walked"
        # Could not reach or cross it on foot: activate it the way crossing would.
        mo = self.player_mo(flow)
        why = "no free spot in front of it" if not placed else _blocked_reason(sim, mo)
        p_activate_line(sim, line, mo, 0, SPAC_CROSS)
        events = flow.tick([TicCmd()])
        exited = any(isinstance(e, MapExit) for e in events) or bool(flow.pending_action)
        return exited, f"activated directly ({why})"

    def check_arrival(self, flow: GameFlow, expected: int, position: int) -> tuple[bool, str]:
        if expected == -1:
            ok = flow.state == GameState.FINALE
            return ok, "finale started" if ok else f"state {flow.state.name}, expected FINALE"
        if flow.state != GameState.LEVEL or flow.sim is None:
            return False, f"state {flow.state.name}"
        sim = flow.sim
        if sim.map_num != expected:
            return False, f"arrived on map {sim.map_num}, expected {expected}"
        mo = self.player_mo(flow)
        assert sim.level is not None
        starts = [
            t
            for t in sim.level.things
            if t.type in PLAYER_START_TYPES[:1] and t.args[0] == position
        ]
        if not starts:
            return True, f"map {expected}; no start for position {position}"
        on_start = any((mo.x >> FRACBITS) == t.x and (mo.y >> FRACBITS) == t.y for t in starts)
        if not on_start:
            return (
                False,
                f"player at ({mo.x >> FRACBITS}, {mo.y >> FRACBITS}), not on start {position}",
            )
        return True, f"map {expected}, start {position}"

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check_exits(self, maps: Iterable[int]) -> None:
        for map_num in maps:
            for exit_def in self.find_exits(map_num):
                self.check_exit(exit_def)

    def check_exit(self, exit_def: ExitDef, player_class: int = 0) -> Check:
        expected = self.expected_map(exit_def)
        what = EXIT_SPECIALS.get(exit_def.special, str(exit_def.special))
        where = (
            f"line {exit_def.line_index}"
            if exit_def.line_index >= 0
            else f"ACS script {exit_def.script}"
        )
        target = f"{exit_def.args} -> {expected}" if exit_def.args_known else "(args at run time)"
        name = f"MAP{exit_def.map_num:02d} {where} {what}{target}"
        try:
            flow = self.start(exit_def.map_num, player_class)
            sim = flow.sim
            assert sim is not None and sim.level is not None
            if exit_def.line_index >= 0:
                exited, how = self.walk_over_line(flow, sim.level.lines[exit_def.line_index])
            else:
                exited, how = self.run_exit_script(flow, exit_def.script)
            if not exited:
                # A bytecode match can be a false positive, so an ACS candidate that
                # never leaves is only a warning; an exit line that does nothing fails.
                status = "WARN" if exit_def.script >= 0 else "FAIL"
                return self.report.add(Check("exit", name, status, f"no map exit ({how})"))
            self.finish_transition(flow)
            if exit_def.script >= 0:
                return self.report.add(self._script_arrival(flow, exit_def, name, expected))
            ok, detail = self.check_arrival(flow, expected, exit_def.args[1])
            if ok and flow.state == GameState.LEVEL:
                for _ in range(35):
                    flow.tick([TicCmd()])
                self.draw(flow)
            status = "PASS" if ok and how == "walked" else ("WARN" if ok else "FAIL")
            if how != "walked":
                detail = f"{detail}; {how}"
            return self.report.add(Check("exit", name, status, detail))
        except Exception as exc:  # noqa: BLE001 - a crash is a checklist failure
            return self.report.add(Check("exit", name, "FAIL", f"{type(exc).__name__}: {exc}"))

    def _script_arrival(self, flow: GameFlow, exit_def: ExitDef, name: str, expected: int) -> Check:
        """Where an ACS exit led. The target is only compared when the bytecode names it."""
        if flow.state == GameState.FINALE:
            return Check("exit", name, "PASS", "script started the finale")
        if flow.state != GameState.LEVEL or flow.sim is None:
            return Check("exit", name, "FAIL", f"state {flow.state.name} after the script")
        arrived = flow.sim.map_num
        if arrived not in self.mapinfo:
            return Check("exit", name, "FAIL", f"script led to unknown map {arrived}")
        if exit_def.args_known and exit_def.special == 74:
            ok, detail = self.check_arrival(flow, expected, exit_def.args[1])
            return Check("exit", name, "PASS" if ok else "FAIL", f"{detail}; script ran")
        return Check("exit", name, "PASS", f"script took the player to map {arrived}")

    def run_exit_script(self, flow: GameFlow, script: int) -> tuple[bool, str]:
        sim = flow.sim
        assert sim is not None and sim.acs_vm is not None
        mo = self.player_mo(flow)
        sim.acs_vm.start_script(script, (0, 0, 0), activator=mo, sim=sim)
        for _ in range(35 * 30):
            events = flow.tick([TicCmd()])
            if any(isinstance(e, MapExit) for e in events) or flow.pending_action:
                return True, "script ran"
        return False, "script ran for 30 s without leaving"

    def check_hubs(self) -> None:
        """Hub persistence (monsters stay dead, keys kept) and the cluster key strip."""
        exits = {num: self.find_exits(num) for num in self.maps}
        done_clusters: set[int] = set()
        for a in self.maps:
            info_a = self.mapinfo.get(a)
            if info_a is None or info_a.cluster in done_clusters:
                continue
            for out in exits.get(a, []):
                b = self.expected_map(out) if out.line_index >= 0 else -1
                info_b = self.mapinfo.get(b) if b > 0 else None
                if info_b is None or info_b.cluster != info_a.cluster:
                    continue
                back = [
                    e
                    for e in exits.get(b, [])
                    if e.line_index >= 0 and e.special == 74 and self.expected_map(e) == a
                ]
                if back:
                    self.check_hub_return(out, back[0])
                    done_clusters.add(info_a.cluster)
                    break
        for a in self.maps:
            info_a = self.mapinfo.get(a)
            for out in exits.get(a, []):
                b = self.expected_map(out) if out.special == 74 and out.line_index >= 0 else -1
                info_b = self.mapinfo.get(b) if b > 0 else None
                if info_a and info_b and info_b.cluster != info_a.cluster:
                    self.check_cluster_change(out)
                    return

    def check_hub_return(self, out: ExitDef, back: ExitDef) -> Check:
        a, b = out.map_num, back.map_num
        name = f"MAP{a:02d} -> MAP{b:02d} -> MAP{a:02d}"
        try:
            flow = self.start(a, 0)
            sim = flow.sim
            assert sim is not None and sim.level is not None
            killed = 0
            mo = self.player_mo(flow)
            for th in list(sim.thinkers):
                if (
                    isinstance(th, Mobj)
                    and th.flags & MobjFlag.MF_COUNTKILL
                    and th.health > 0
                    and th.flags & MobjFlag.MF_SHOOTABLE
                ):
                    damage_mobj(sim, th, mo, mo, 10000)
                    killed += 1
            flow.sim.players[0].keys = ALL_KEYS
            for _ in range(35 * 5):  # let death animations finish (some remove the thing)
                flow.tick([TicCmd()])
            alive_before, dead_before = _monster_counts(sim)
            exited, how = self.walk_over_line(flow, sim.level.lines[out.line_index])
            if not exited:
                return self.report.add(Check("hub", name, "FAIL", f"could not leave {a}"))
            self.finish_transition(flow)
            assert flow.sim is not None and flow.sim.level is not None
            exited, how2 = self.walk_over_line(flow, flow.sim.level.lines[back.line_index])
            if not exited:
                return self.report.add(Check("hub", name, "FAIL", f"could not leave {b}"))
            self.finish_transition(flow)
            sim = flow.sim
            ok, detail = self.check_arrival(flow, a, back.args[1])
            if not ok:
                return self.report.add(Check("hub", name, "FAIL", detail))
            alive, dead = _monster_counts(sim)
            keys = sim.players[0].keys
            problems = []
            # Dormant monsters cannot be killed, and scripts may spawn more, so the test
            # is that no corpse came back: the dead count never drops.
            if dead < dead_before:
                problems.append(f"{dead_before - dead} of {dead_before} corpses are gone or alive")
            if keys != ALL_KEYS:
                problems.append(f"keys {keys:#x} after staying in the hub")
            note = (
                "" if (how, how2) == ("walked", "walked") else " (an exit was activated directly)"
            )
            status = "FAIL" if problems else ("PASS" if not note else "WARN")
            detail = "; ".join(problems) or (
                f"{dead_before} dead stayed dead ({killed} hit; alive {alive_before} -> {alive}), "
                f"keys kept{note}"
            )
            return self.report.add(Check("hub", name, status, detail))
        except Exception as exc:  # noqa: BLE001
            return self.report.add(Check("hub", name, "FAIL", f"{type(exc).__name__}: {exc}"))

    def check_cluster_change(self, out: ExitDef) -> Check:
        b = self.expected_map(out)
        name = f"cluster change MAP{out.map_num:02d} -> MAP{b:02d} strips keys"
        try:
            flow = self.start(out.map_num, 0)
            assert flow.sim is not None and flow.sim.level is not None
            flow.sim.players[0].keys = ALL_KEYS
            exited, _how = self.walk_over_line(flow, flow.sim.level.lines[out.line_index])
            if not exited:
                return self.report.add(Check("hub", name, "FAIL", "could not leave"))
            self.finish_transition(flow)
            assert flow.sim is not None
            keys = flow.sim.players[0].keys
            ok = flow.sim.map_num == b and keys == 0
            return self.report.add(
                Check(
                    "hub", name, "PASS" if ok else "FAIL", f"map {flow.sim.map_num}, keys {keys:#x}"
                )
            )
        except Exception as exc:  # noqa: BLE001
            return self.report.add(Check("hub", name, "FAIL", f"{type(exc).__name__}: {exc}"))

    def soak(self, map_num: int, player_class: int, tics: int) -> Check:
        """Bot run on one map, then the save/load replay check."""
        name = f"MAP{map_num:02d} {CLASS_NAMES[player_class]}"
        try:
            flow = self.start(map_num, player_class)
            player = flow.sim.players[0] if flow.sim is not None else None
            assert player is not None
            player.cheats |= CF_GODMODE
            player.weaponowned = [True, True, True, True]
            player.mana = [200, 200]
            player.keys = ALL_KEYS
            bot = Bot(seed=map_num * 3 + player_class)
            problems: list[str] = []
            start_xy = (self.player_mo(flow).x, self.player_mo(flow).y)
            maps_seen = {map_num}
            for tic in range(tics):
                if flow.state != GameState.LEVEL:
                    self.finish_transition(flow)
                    if flow.state != GameState.LEVEL:
                        break
                    assert flow.sim is not None
                    maps_seen.add(flow.sim.map_num)
                    flow.sim.players[0].cheats |= CF_GODMODE
                flow.tick([bot.cmd(flow)])
                if tic % 35 == 0:
                    problems.extend(_position_problems(flow))
                    if tic % 350 == 0:
                        self.draw(flow)
                if len(problems) > 5:
                    break
            moved = flow.sim is not None and (
                (self.player_mo(flow).x, self.player_mo(flow).y) != start_xy
            )
            if flow.state == GameState.LEVEL:
                problems.extend(self.save_load_replay(flow, bot))
            detail = (
                f"{bot.shots} shots, {bot.uses} uses, {bot.turns} turns, maps {sorted(maps_seen)}"
            )
            if not moved:
                problems.append("player never moved")
            status = "FAIL" if problems else "PASS"
            if problems:
                detail = "; ".join(problems[:6]) + f" ({detail})"
            return self.report.add(Check("soak", name, status, detail))
        except Exception as exc:  # noqa: BLE001
            return self.report.add(Check("soak", name, "FAIL", f"{type(exc).__name__}: {exc}"))

    def save_load_replay(self, flow: GameFlow, bot: Bot, tics: int = 70) -> list[str]:
        """Save, play on, load: the loaded game must equal the saved one, then keep running.

        Vanilla does not save the RNG (SV_LoadGame runs G_InitNew, which clears it,
        before unarchiving the map), so the comparison leaves out `prndindex` and the
        loaded game is not expected to replay the same input identically.
        """
        assert flow.sim is not None
        saved = _trace_without_rng(flow.sim)
        restored: list[str] = []
        previous_hook = flow.on_level_loaded

        def capture(sim: GameSim, info: Any) -> None:
            restored.append(_trace_without_rng(sim))
            if previous_hook is not None:
                previous_hook(sim, info)

        flow.save_game(0, "playthrough")
        for _ in range(tics):
            flow.tick([bot.cmd(flow)])
            if flow.state != GameState.LEVEL:
                return []  # left the map while playing on; nothing to compare
        flow.on_level_loaded = capture
        try:
            flow.load_game(0)
            flow.tick([TicCmd()])
        finally:
            flow.on_level_loaded = previous_hook
        if not restored:
            return [f"loading failed: {flow.last_load_error or 'no level'}"]
        if restored[0] != saved:
            return [f"loaded state differs from the saved one: {saved} != {restored[0]}"]
        for _ in range(tics):
            flow.tick([bot.cmd(flow)])
        return []

    def check_reborn(self, player_class: int = 0) -> Check:
        """Die after a map change and press use: the reborn slot reloads that map."""
        name = f"reborn after a map change ({CLASS_NAMES[player_class]})"
        try:
            first = self.maps[0]
            out = next(
                (e for e in self.find_exits(first) if e.special == 74 and e.line_index >= 0),
                None,
            )
            if out is None:
                return self.report.add(Check("reborn", name, "WARN", f"MAP{first:02d} has no exit"))
            flow = self.start(first, player_class)
            assert flow.sim is not None and flow.sim.level is not None
            exited, _how = self.walk_over_line(flow, flow.sim.level.lines[out.line_index])
            self.finish_transition(flow)
            assert flow.sim is not None
            target = flow.sim.map_num
            mo = self.player_mo(flow)
            damage_mobj(flow.sim, mo, None, None, 10000)
            for _ in range(70):
                flow.tick([TicCmd()])
            dead = flow.sim.players[0].player_state == PlayerState.PST_DEAD
            for i in range(70):
                flow.tick([TicCmd(buttons=BT_USE if i % 2 == 0 else 0)])
            assert flow.sim is not None
            player = flow.sim.players[0]
            ok = (
                exited
                and dead
                and flow.state == GameState.LEVEL
                and flow.sim.map_num == target
                and player.player_state == PlayerState.PST_LIVE
                and player.health > 0
            )
            detail = (
                f"died on map {target}: dead={dead}, "
                f"now map {flow.sim.map_num} health {player.health}"
            )
            return self.report.add(Check("reborn", name, "PASS" if ok else "FAIL", detail))
        except Exception as exc:  # noqa: BLE001
            return self.report.add(Check("reborn", name, "FAIL", f"{type(exc).__name__}: {exc}"))


class Bot:
    """Scripted player input: wander, turn when stuck, use, shoot what it sees."""

    def __init__(self, seed: int) -> None:
        self.rng = _Xorshift(seed)
        self.tic = 0
        self.turn_target: int | None = None
        self.last_pos: tuple[int, int] | None = None
        self.stuck = 0
        self.target: Mobj | None = None
        self.shots = self.uses = self.turns = 0
        self.weapon = 0

    def cmd(self, flow: GameFlow) -> TicCmd:
        self.tic += 1
        sim = flow.sim
        if flow.state == GameState.INTERMISSION:
            # A netgame intermission ends on a fire/use press (CheckForSkip needs the edge).
            return TicCmd(buttons=BT_USE if self.tic % 4 == 0 else 0)
        if sim is None or flow.state != GameState.LEVEL:
            return TicCmd()
        player = flow.player
        mo = player.mo if player is not None else None
        if player is None or mo is None:
            return TicCmd()
        if player.player_state == PlayerState.PST_DEAD:
            return TicCmd(buttons=BT_USE if self.tic % 4 == 0 else 0)  # respawn
        cmd = TicCmd(forwardmove=WALK_SPEED)
        if self.tic % 4 == 0:
            self.target = _visible_monster(sim, mo)
        if self.target is not None and self.target.health > 0:
            want = point_to_angle2(mo.x, mo.y, self.target.x, self.target.y)
            delta = _angle_delta(want, mo.angle)
            cmd.angleturn = _turn_cmd(delta)
            if abs(delta) < ANG1 * 10:
                cmd.buttons |= BT_ATTACK
                self.shots += 1
        else:
            pos = (mo.x >> 16, mo.y >> 16)
            if pos == self.last_pos:
                self.stuck += 1
            else:
                self.stuck = 0
            self.last_pos = pos
            wander = self.tic % 175 == 0  # turn now and then to explore
            if (self.stuck > 6 or wander) and self.turn_target is None:
                self.turn_target = (mo.angle + ANG90 + (self.rng.next() & 0x7FFFFFFF)) & 0xFFFFFFFF
                self.turns += 1
            if self.turn_target is not None:
                delta = _angle_delta(self.turn_target, mo.angle)
                cmd.angleturn = _turn_cmd(delta)
                cmd.forwardmove = 0
                if abs(delta) < ANG1 * 8:
                    self.turn_target = None
                    self.stuck = 0
        if self.tic % 12 == 0:
            cmd.buttons |= BT_USE
            self.uses += 1
        if self.tic % 210 == 0:
            self.weapon = (self.weapon + 1) % 4
            cmd.buttons |= BT_CHANGE | (self.weapon << BT_WEAPONSHIFT)
        return cmd


def _visible_monster(sim: GameSim, mo: Mobj) -> Mobj | None:
    best: Mobj | None = None
    best_dist = 1024 * FRACUNIT
    for th in sim.thinkers:
        if (
            isinstance(th, Mobj)
            and th.flags & MobjFlag.MF_COUNTKILL
            and th.health > 0
            and th.flags & MobjFlag.MF_SHOOTABLE
        ):
            dist = max(abs(th.x - mo.x), abs(th.y - mo.y))
            if dist < best_dist and check_sight(sim, mo, th):
                best, best_dist = th, dist
    return best


def _trace_without_rng(sim: GameSim) -> str:
    """Demo-trace line for the current state, without the RNG index."""
    parts = trace_tic(sim).format().split()
    del parts[2]  # T <leveltime> <prndindex> ...
    return " ".join(parts)


def _monster_counts(sim: GameSim) -> tuple[int, int]:
    """(alive, dead) counts of MF_COUNTKILL monsters."""
    alive = dead = 0
    for th in sim.thinkers:
        if isinstance(th, Mobj) and not th.removed and th.flags & MobjFlag.MF_COUNTKILL:
            if th.health > 0:
                alive += 1
            else:
                dead += 1
    return alive, dead


def _blocked_reason(sim: GameSim, mo: Mobj) -> str:
    """Why the player cannot step forward (for the checklist detail)."""
    fine = (mo.angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT
    x = mo.x + fixed_mul(8 * FRACUNIT, fine_cosine(fine))
    y = mo.y + fixed_mul(8 * FRACUNIT, fine_sine(fine))
    if not check_position(sim, mo, x, y):
        bm = sim.map_state.blocking_mobj
        return f"blocked by thing type {bm.type}" if bm is not None else "blocked by a wall"
    st = sim.map_state
    if st.tmceilingz - st.tmfloorz < mo.height:
        return "no headroom: a closed door or lowered ceiling"
    if st.tmfloorz - mo.z > 24 * FRACUNIT:
        return "step too high"
    return "did not cross"


def _position_problems(flow: GameFlow) -> list[str]:
    sim = flow.sim
    if sim is None or sim.level is None or flow.state != GameState.LEVEL:
        return []
    mo = sim.players[0].mo
    if mo is None:
        return ["player has no mobj"]
    problems = []
    real = sim.level.point_in_subsector(mo.x, mo.y)
    if mo.subsector is None or real.sector is not mo.subsector.sector:
        problems.append(f"player sector link is stale at ({mo.x >> 16}, {mo.y >> 16})")
    # Compare with the sector under the player, not floorz/ceilingz: those include
    # steps and things, and vanilla can leave a player under a step for a while
    # (P_CheckOnmobj finds a monster hovering over the step).
    sector = real.sector
    below = mo.z < sector.floor_height - FRACUNIT
    if below:
        saved = (mo.z, mo.momz, mo.floorz, mo.ceilingz)
        blocker = check_on_mobj(sim, mo)
        mo.z, mo.momz, mo.floorz, mo.ceilingz = saved
        if blocker is not None:
            below = False  # vanilla: a solid thing over the step keeps the player down
    if below or mo.z > sector.ceiling_height:
        problems.append(
            f"player outside its sector at ({mo.x >> 16}, {mo.y >> 16}): z {mo.z >> 16}, "
            f"floor {sector.floor_height >> 16}, ceiling {sector.ceiling_height >> 16}"
        )
    return problems


_LSPEC_RANGE = range(int(PCode.PCD_LSPEC1), int(PCode.PCD_LSPEC5DIRECT) + 1)


def _acs_exits(map_num: int, behavior: Any) -> list[ExitDef]:
    """Scripts whose bytecode has an LSPEC pcode followed by an exit special.

    Hexen ACS is a stream of int32 words. A match can be a false positive (an
    operand that looks like a pcode); running the script tells.
    """
    code: bytes = behavior.code
    scripts = sorted(behavior.scripts, key=lambda s: s.offset)
    found: list[ExitDef] = []
    for i, script in enumerate(scripts):
        if script.is_open:
            continue
        end = scripts[i + 1].offset if i + 1 < len(scripts) else behavior.info_offset
        end = min(end, len(code))
        words = [struct.unpack_from("<i", code, off)[0] for off in range(script.offset, end - 3, 4)]
        for j in range(len(words) - 1):
            if words[j] in _LSPEC_RANGE and words[j + 1] in EXIT_SPECIALS:
                direct = int(PCode.PCD_LSPEC1DIRECT) <= words[j] and j + 3 < len(words)
                args = (words[j + 2] & 0xFF, words[j + 3] & 0xFF) if direct else (0, 0)
                found.append(
                    ExitDef(map_num, words[j + 1], args, script=script.number, args_known=direct)
                )
                break
    return found


def run(
    wad_path: Path,
    *,
    only: Sequence[str] = (),
    maps: Sequence[int] = (),
    classes: Sequence[int] = (0, 1, 2),
    soak_tics: int = 35 * 60,
    render: bool = False,
    log: Callable[[str], None] = print,
) -> Report:
    stack = WadStack(wad_path)
    mapinfo = load_mapinfo(stack)
    with tempfile.TemporaryDirectory() as tmp:
        pt = Playthrough(stack, mapinfo, save_dir=Path(tmp), render=render)
        selected = [m for m in pt.maps if not maps or m in maps]
        kinds = set(only) or {"exits", "hubs", "soak", "reborn"}
        try:
            if "exits" in kinds:
                log("== exits ==")
                pt.check_exits(selected)
            if "hubs" in kinds:
                log("== hubs ==")
                pt.check_hubs()
            if "reborn" in kinds:
                log("== reborn ==")
                for pclass in classes:
                    pt.check_reborn(pclass)
            if "soak" in kinds:
                log("== soak ==")
                for map_num in selected:
                    for pclass in classes:
                        pt.soak(map_num, pclass, soak_tics)
        finally:
            pt.close()
    return pt.report


def main(argv: Sequence[str] | None = None) -> int:
    from pyhexen.app.config import resolve_iwad

    parser = argparse.ArgumentParser(description="PyHexen scripted playthrough checklist (T15.7)")
    parser.add_argument("--iwad", default=None)
    parser.add_argument("--only", default="", help="comma list: exits,hubs,soak,reborn")
    parser.add_argument("--maps", default="", help="comma list of map numbers")
    parser.add_argument("--classes", default="0,1,2", help="0=Fighter 1=Cleric 2=Mage")
    parser.add_argument("--soak-seconds", type=int, default=60)
    parser.add_argument("--render", action="store_true", help="draw frames headlessly")
    args = parser.parse_args(argv)
    iwad = resolve_iwad(args.iwad)
    if iwad is None:
        print("HEXEN.WAD not found", file=sys.stderr)
        return 2
    started = time.perf_counter()
    report = run(
        Path(iwad),
        only=[s for s in args.only.split(",") if s],
        maps=[int(s) for s in args.maps.split(",") if s],
        classes=[int(s) for s in args.classes.split(",") if s],
        soak_tics=args.soak_seconds * 35,
        render=args.render,
    )
    elapsed = time.perf_counter() - started
    print(
        f"\n{len(report.checks)} checks: {report.count('PASS')} pass, {report.count('WARN')} warn, "
        f"{report.count('FAIL')} fail in {elapsed:.0f} s"
    )
    return 1 if report.count("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
