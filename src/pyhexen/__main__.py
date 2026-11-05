# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""CLI entry point for PyHexen."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pygame

from pyhexen import __version__
from pyhexen.app.config import load_config, resolve_iwad
from pyhexen.app.main_loop import FixedTimestepAccumulator
from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
from pyhexen.render.renderer import WorldRenderer
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="pyhexen",
        description="PyHexen - Python reimplementation of the Hexen engine",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"pyhexen {__version__}",
    )
    parser.add_argument("--iwad", type=str, default=None, help="Path to HEXEN.WAD")
    parser.add_argument("--file", type=str, nargs="*", default=None, help="PWAD files to load")
    parser.add_argument("--warp", type=str, default=None, help="Map name or number to warp to")
    parser.add_argument("--skill", type=int, default=3, help="Skill level (1-5)")
    parser.add_argument(
        "--class",
        dest="player_class",
        choices=["fighter", "cleric", "mage"],
        default="fighter",
        help="Player class",
    )
    parser.add_argument("--fly", action="store_true", help="Start in free-fly camera mode")
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=0,
        help="Run for N frames headlessly (for automated testing)",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Path to save a screenshot before exiting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the PyHexen CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    iwad_path = resolve_iwad(args.iwad)
    if not args.warp and not args.fly and not args.headless_frames:
        print(f"pyhexen {__version__}")
        if iwad_path is None:
            print("Note: No HEXEN.WAD found. Use --iwad <path> or set PYHEXEN_IWAD.")
        return 0

    if iwad_path is None:
        print("Error: HEXEN.WAD could not be located. Specify --iwad <path>.")
        return 1

    config = load_config()
    stack = WadStack(iwad_path)
    if args.file:
        for pwad in args.file:
            stack.add_wad(pwad)

    # Determine map name
    map_table = load_mapinfo(stack)
    map_name = "MAP01"
    if args.warp:
        if args.warp.isdigit():
            m_num = int(args.warp)
            if m_num in map_table:
                map_name = map_table[m_num].lump_name
            else:
                map_name = f"MAP{m_num:02d}"
        else:
            warp_upper = args.warp.upper()
            if warp_upper in map_table:
                map_name = map_table[warp_upper].lump_name
            else:
                map_name = warp_upper

    map_info = map_table.get(map_name)
    level = load_level(stack, map_name)

    is_headless = args.headless_frames > 0
    ctx = RenderContext(
        standalone=is_headless,
        width=config.video.width,
        height=config.video.height,
        vsync=config.video.vsync,
        title=f"PyHexen - {map_name}",
    )
    renderer = WorldRenderer(ctx, wad=stack)
    renderer.load_map(stack, level, map_info)

    sim = GameSim(level=level, skill=args.skill - 1)
    spawn_map_things(sim)

    camera = Camera()
    # Position camera at player 1 start if found
    p1_start = level.get_player_start(1) if hasattr(level, "get_player_start") else None
    if p1_start is not None:
        camera.set_from_fixed(p1_start.x, p1_start.y, p1_start.z + (40 << 16), p1_start.angle)
    elif level.things:
        t0 = level.things[0]
        camera.set_from_fixed(t0.x, t0.y, t0.z + (40 << 16), t0.angle)

    if is_headless:
        # Run fixed number of frames headlessly
        for _ in range(args.headless_frames):
            sim.tick()
            renderer.render_frame(camera, sim)
        if args.screenshot:
            renderer.save_screenshot(args.screenshot)
        ctx.close()
        return 0

    # Interactive loop
    clock = pygame.time.Clock()
    accumulator = FixedTimestepAccumulator()
    running = True

    pygame.mouse.set_relative_mode(True)

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F12:
                    out_dir = Path("out")
                    out_dir.mkdir(exist_ok=True)
                    shot_path = out_dir / f"screenshot_{int(pygame.time.get_ticks())}.png"
                    saved = renderer.save_screenshot(shot_path)
                    print(f"Saved screenshot: {saved}")

        # Free-fly camera controls
        keys = pygame.key.get_pressed()
        forward = 0.0
        right = 0.0
        up = 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            forward += 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            forward -= 1.0
        if keys[pygame.K_d]:
            right += 1.0
        if keys[pygame.K_a]:
            right -= 1.0
        if keys[pygame.K_e] or keys[pygame.K_SPACE]:
            up += 1.0
        if keys[pygame.K_c] or keys[pygame.K_LCTRL]:
            up -= 1.0

        m_dx, m_dy = pygame.mouse.get_rel()
        yaw_delta = m_dx * 0.003
        pitch_delta = -m_dy * 0.003
        camera.move_free_fly(
            dt,
            forward=forward,
            right=right,
            up=up,
            yaw_delta=yaw_delta,
            pitch_delta=pitch_delta,
            speed=600.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 300.0,
        )

        for _ in accumulator.advance(dt):
            sim.tick()

        renderer.render_frame(camera, sim, alpha=accumulator.alpha)
        ctx.swap_buffers()

    ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
