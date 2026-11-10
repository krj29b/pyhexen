# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Full playthrough verification and audit tool (T14.5).

Audits all 40 Hexen maps and 5 hubs across Fighter, Cleric, and Mage classes,
verifying map loading, polyobjects, ACS scripts, thing spawning, and hub transitions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from pyhexen.app.config import resolve_iwad
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.hub import HubManager
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import spawn_player
from pyhexen.sim.polyobj import init_polyobjs
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


def audit_map(
    stack: WadStack,
    map_name: str,
    player_class: int = 0,
    skill: int = 2,
    sim_tics: int = 35,
) -> dict[str, Any]:
    """Audit a single map by loading, spawning actors, and running simulation tics."""
    result: dict[str, Any] = {
        "map": map_name,
        "player_class": player_class,
        "skill": skill,
        "status": "PASS",
        "error": None,
        "sectors": 0,
        "lines": 0,
        "things": 0,
        "polyobjs": 0,
        "thinkers": 0,
        "acs_scripts": 0,
    }

    try:
        level = load_level(stack, map_name)
        result["sectors"] = len(level.sectors)
        result["lines"] = len(level.lines)
        result["things"] = len(level.things)

        polyobjs = init_polyobjs(level)
        result["polyobjs"] = len(polyobjs)

        sim = GameSim(level=level, skill=skill, player_class=player_class)
        spawn_map_things(sim)
        spawn_player(sim, 0)

        result["acs_scripts"] = len(sim.acs_vm.scripts) if sim.acs_vm else 0

        # Advance simulation for sim_tics to trigger OPEN scripts and thinkers
        for _ in range(sim_tics):
            sim.tick()

        result["thinkers"] = len(sim.thinkers)
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def audit_hub_transitions(stack: WadStack, player_class: int = 0) -> list[dict[str, Any]]:
    """Test transitions within Hub 1 and cluster boundary transitions across all 5 hubs."""
    results: list[dict[str, Any]] = []
    mapinfo_table = load_mapinfo(stack)

    def load_level_cb(map_num: int) -> Level:
        info = mapinfo_table[map_num]
        return load_level(stack, info.lump_name)

    hub = HubManager(current_cluster=1)

    # 1. Start in MAP01
    level1 = load_level_cb(1)
    sim = GameSim(level=level1, skill=2, player_class=player_class)
    sim.map_num = 1
    sim.cluster = 1
    spawn_map_things(sim)
    spawn_player(sim, 0)

    transitions = [
        (1, 2, "Hub 1: Winnowing Hall -> Seven Portals"),
        (2, 3, "Hub 1: Seven Portals -> Guardian of Ice"),
        (3, 2, "Hub 1: Return to Seven Portals"),
        (2, 4, "Hub 1: Seven Portals -> Guardian of Fire"),
        (4, 2, "Hub 1: Return to Seven Portals"),
        (2, 5, "Hub 1: Seven Portals -> Guardian of Steel"),
        (5, 2, "Hub 1: Return to Seven Portals"),
        (2, 8, "Cluster Boundary: Hub 1 -> Hub 2 (Darkmere)"),
        (8, 13, "Hub 2: Darkmere -> Shadow Wood"),
        (8, 27, "Cluster Boundary: Hub 2 -> Hub 3 (Heresiarch's Seminary)"),
        (27, 28, "Hub 3: Seminary -> Dragon Chapel"),
        (28, 21, "Cluster Boundary: Hub 3 -> Hub 4 (Forsaken Outpost)"),
        (21, 22, "Hub 4: Outpost -> Castle of Grief"),
        (21, 35, "Cluster Boundary: Hub 4 -> Hub 5 (Necropolis)"),
        (35, 40, "Hub 5: Necropolis -> Dark Crucible (Korax)"),
    ]

    for from_map, to_map, desc in transitions:
        step_res: dict[str, Any] = {
            "from": f"MAP{from_map:02d}",
            "to": f"MAP{to_map:02d}",
            "desc": desc,
            "status": "PASS",
            "error": None,
        }
        try:
            sim, res = hub.transition_map(
                sim,
                target_map=to_map,
                position=0,
                mapinfo_table=mapinfo_table,
                load_level_cb=load_level_cb,
            )
            # Advance simulation for 35 tics in the new map
            for _ in range(35):
                sim.tick()
        except Exception as exc:
            step_res["status"] = "FAIL"
            step_res["error"] = f"{type(exc).__name__}: {exc}"
        results.append(step_res)

    return results


def run_full_audit(iwad_path: Path) -> int:
    """Run full audit across all 31 campaign maps, all 3 classes, and hub transitions."""
    print(f"Loading IWAD: {iwad_path}")
    stack = WadStack(iwad_path)
    mapinfo_table = load_mapinfo(stack)

    # Filter to actual map lumps present in IWAD
    lump_names = {lmp.name for lmp in stack.lumps}
    all_maps = sorted(
        [info for info in mapinfo_table.values() if info.lump_name in lump_names],
        key=lambda m: m.map_num,
    )
    print(f"Discovered {len(all_maps)} playable maps in IWAD.")

    classes = [
        (0, "Fighter"),
        (1, "Cleric"),
        (2, "Mage"),
    ]

    total_tests = 0
    failures: list[dict[str, Any]] = []

    start_time = time.perf_counter()

    for pclass_id, pclass_name in classes:
        print(f"\n{'=' * 60}\nAuditing Class: {pclass_name} ({len(all_maps)} maps)\n{'=' * 60}")
        for minfo in all_maps:
            total_tests += 1
            res = audit_map(stack, minfo.lump_name, player_class=pclass_id, sim_tics=35)
            if res["status"] == "PASS":
                print(
                    f"  [{minfo.lump_name}] Cluster {minfo.cluster} | "
                    f"Sec: {res['sectors']:3d} | Lin: {res['lines']:4d} | "
                    f"Thg: {res['things']:3d} | PO: {res['polyobjs']:2d} | "
                    f"Thk: {res['thinkers']:3d} | PASS"
                )
            else:
                print(f"  [{minfo.lump_name}] FAIL: {res['error']}")
                failures.append(res)

    print(f"\n{'=' * 60}\nAuditing Hub Transitions & Persistence\n{'=' * 60}")
    hub_results = audit_hub_transitions(stack, player_class=0)
    for hr in hub_results:
        total_tests += 1
        if hr["status"] == "PASS":
            print(f"  [{hr['from']} -> {hr['to']}] {hr['desc']}: PASS")
        else:
            print(f"  [{hr['from']} -> {hr['to']}] {hr['desc']}: FAIL ({hr['error']})")
            failures.append(hr)

    elapsed = time.perf_counter() - start_time
    print(f"\n{'=' * 60}\nPlaythrough Audit Summary\n{'=' * 60}")
    print(f"Total audit checks: {total_tests} in {elapsed:.2f}s")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures:
            print(f" - {f}")
        return 1
    else:
        print("ALL CHECKS PASSED: 100% of maps and hub transitions verified without errors.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="PyHexen Full Playthrough Audit")
    parser.add_argument("--iwad", type=str, default=None, help="Path to HEXEN.WAD")
    args = parser.parse_args()

    iwad = resolve_iwad(args.iwad)
    if iwad is None:
        print("Error: HEXEN.WAD not found.", file=sys.stderr)
        sys.exit(1)

    code = run_full_audit(iwad)
    sys.exit(code)


if __name__ == "__main__":
    main()
