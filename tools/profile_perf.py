# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Performance profiling tool for PyHexen simulation and CPU rendering.

Measures average simulation tic time and average CPU render frame time,
identifying hotspots against AGENTS budgets:
- Simulation tic budget: < 10 ms
- Render frame budget: < 8 ms
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from pathlib import Path

from pyhexen.app.config import resolve_iwad
from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
from pyhexen.render.renderer import WorldRenderer
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import spawn_player
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def profile_map(
    stack: WadStack,
    map_name: str,
    sim_tics: int = 1000,
    render_frames: int = 500,
    show_profile: bool = True,
) -> tuple[float, float]:
    """Profile simulation and CPU rendering on a specific map."""
    print(
        f"\n{'=' * 60}\n"
        f"Profiling {map_name} ({sim_tics} sim tics, {render_frames} render frames)\n"
        f"{'=' * 60}"
    )

    map_table = load_mapinfo(stack)
    map_info = map_table.get(map_name)
    level = load_level(stack, map_name)

    # 1. Profile Simulation
    sim = GameSim(level=level, skill=2, player_class=0)
    spawn_map_things(sim)
    player = spawn_player(sim, 0)

    # Warm-up 50 tics
    for _ in range(50):
        sim.tick()

    prof_sim = cProfile.Profile()
    prof_sim.enable()
    start_sim = time.perf_counter()
    for _ in range(sim_tics):
        sim.tick()
    sim_duration = time.perf_counter() - start_sim
    prof_sim.disable()

    avg_sim_ms = (sim_duration / sim_tics) * 1000.0
    print(f"Simulation: {sim_tics} tics in {sim_duration:.3f}s -> Avg: {avg_sim_ms:.3f} ms/tic")
    if avg_sim_ms < 10.0:
        print(
            f"  [PASS] Budget (<10.0 ms): {avg_sim_ms:.3f} ms ({10.0 / avg_sim_ms:.1f}x headroom)"
        )
    else:
        print(f"  [FAIL] Over budget (<10.0 ms): {avg_sim_ms:.3f} ms")

    if show_profile:
        print("\n--- Simulation Top 10 Functions ---")
        ps = pstats.Stats(prof_sim).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(10)

    # 2. Profile Rendering
    ctx = RenderContext(standalone=True, width=1280, height=800)
    renderer = WorldRenderer(ctx, wad=stack)
    renderer.load_map(stack, level, map_info)

    camera = Camera()
    camera.set_from_fixed(
        player.mo.x,
        player.mo.y,
        player.viewz,
        player.mo.angle,
        player.lookdir,
        snap=True,
    )

    # Warm-up 20 frames
    for _ in range(20):
        renderer.render_frame(camera, sim)

    prof_ren = cProfile.Profile()
    prof_ren.enable()
    start_ren = time.perf_counter()
    for _ in range(render_frames):
        renderer.render_frame(camera, sim)
    ren_duration = time.perf_counter() - start_ren
    prof_ren.disable()

    avg_ren_ms = (ren_duration / render_frames) * 1000.0
    print(
        f"\nRendering: {render_frames} frames in {ren_duration:.3f}s -> "
        f"Avg: {avg_ren_ms:.3f} ms/frame"
    )
    if avg_ren_ms < 8.0:
        print(f"  [PASS] Budget (<8.0 ms): {avg_ren_ms:.3f} ms ({8.0 / avg_ren_ms:.1f}x headroom)")
    else:
        print(f"  [FAIL] Over budget (<8.0 ms): {avg_ren_ms:.3f} ms")

    if show_profile:
        print("\n--- Rendering Top 10 Functions ---")
        ps = pstats.Stats(prof_ren).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(10)

    renderer.release()
    ctx.close()

    return avg_sim_ms, avg_ren_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PyHexen simulation and rendering.")
    parser.add_argument("--iwad", type=str, default=None, help="Path to HEXEN.WAD")
    parser.add_argument("--map", type=str, default="MAP01", help="Map to profile (default MAP01)")
    parser.add_argument(
        "--sim-tics", type=int, default=1000, help="Number of sim tics (default 1000)"
    )
    parser.add_argument(
        "--render-frames", type=int, default=500, help="Number of render frames (default 500)"
    )
    args = parser.parse_args()

    iwad_path = resolve_iwad(args.iwad)
    if iwad_path is None or not Path(iwad_path).exists():
        print("Error: No HEXEN.WAD found. Specify --iwad path.")
        return

    with WadStack(iwad_path) as stack:
        profile_map(stack, args.map, args.sim_tics, args.render_frames)


if __name__ == "__main__":
    main()
