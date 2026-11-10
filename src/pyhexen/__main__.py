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
from pyhexen.app.config import Config, load_config, resolve_iwad
from pyhexen.app.crash import crash_guard
from pyhexen.app.dialog import prompt_for_iwad, show_error_dialog
from pyhexen.app.input import InputHandler
from pyhexen.app.main_loop import FixedTimestepAccumulator
from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.render.psprites import PspriteRenderer
from pyhexen.render.renderer import WorldRenderer
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import spawn_player
from pyhexen.ui.menu import MenuManager, PromptType
from pyhexen.ui.statusbar import StatusBar, get_sb_palette
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
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Record a demo to the specified path",
    )
    parser.add_argument(
        "--playdemo",
        type=str,
        default=None,
        help="Play back a demo from the specified path",
    )
    parser.add_argument(
        "--timedemo",
        type=str,
        default=None,
        help="Play back a demo as fast as possible to benchmark and verify determinism",
    )
    return parser


def _run_game(args: argparse.Namespace, iwad_path: Path) -> int:
    """Initialize game simulation and run the main rendering/game loop."""
    config = load_config()
    stack = WadStack(iwad_path)
    if args.file:
        for pwad in args.file:
            stack.add_wad(pwad)

    from pyhexen.sim.demo import DemoPlayer, DemoRecorder, play_demo_to_completion
    from pyhexen.sim.savegame import compute_wad_checksum

    demo_player: DemoPlayer | None = None
    if args.playdemo or args.timedemo:
        demo_path = args.playdemo or args.timedemo
        demo_player = DemoPlayer.from_file(demo_path, wad=stack)

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
    elif demo_player is not None:
        map_name = f"MAP{demo_player.demo.header.map_num:02d}"

    map_info = map_table.get(map_name)
    level = load_level(stack, map_name)

    pclass_idx = (
        demo_player.demo.header.player_class
        if demo_player is not None
        else {"fighter": 0, "cleric": 1, "mage": 2}.get(args.player_class.lower(), 0)
    )
    skill = (demo_player.demo.header.skill - 1) if demo_player is not None else (args.skill - 1)
    sim = GameSim(level=level, skill=skill, player_class=pclass_idx)
    if demo_player is not None:
        sim.rng.set_state(demo_player.demo.header.rng_state)
    spawn_map_things(sim)
    player = spawn_player(sim, 0)

    demo_recorder: DemoRecorder | None = None
    if args.record:
        demo_recorder = DemoRecorder(
            sim,
            file_path=args.record,
            iwad_checksum=compute_wad_checksum(iwad_path),
        )

    if args.timedemo:
        assert demo_player is not None
        matched = play_demo_to_completion(sim, demo_player)
        status = "PASSED" if matched else "FAILED"
        print(f"Timedemo finished: {demo_player.total_tics} tics. Determinism: {status}")
        return 0 if matched else 1

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

    overlay = Overlay2D(wad=stack)
    psprite_renderer = PspriteRenderer(stack)
    statusbar = StatusBar(stack)

    camera = Camera()
    camera.set_from_fixed(
        player.mo.x,
        player.mo.y,
        player.viewz,
        player.mo.angle,
        player.lookdir,
        snap=True,
    )

    if is_headless:
        # Run fixed number of frames headlessly
        for _ in range(args.headless_frames):
            sim.tick()
            statusbar.tick(player)
            camera.save_prev_state()
            camera.set_from_fixed(
                player.mo.x,
                player.mo.y,
                player.viewz,
                player.mo.angle,
                player.lookdir,
            )
        renderer.render_frame(camera, sim)
        if not args.fly:
            overlay.clear()
            psprite_renderer.draw(overlay, player, fullscreen=statusbar.fullscreen)
            statusbar.draw(overlay, player, level_time=sim.level_time)
            overlay.render(ctx.ctx, renderer.tex_mgr, palette_index=get_sb_palette(player))
        if args.screenshot:
            renderer.save_screenshot(args.screenshot)
        ctx.close()
        return 0

    # Interactive loop
    clock = pygame.time.Clock()
    accumulator = FixedTimestepAccumulator()
    running = True

    def apply_config(cfg: Config) -> None:
        renderer.fov_deg = cfg.video.fov
        renderer.authentic_lowres = cfg.video.authentic_lowres
        renderer.set_smooth_filter(cfg.video.smooth_filter)
        camera.true_pitch = cfg.video.true_pitch
        camera.authentic_aspect = cfg.video.authentic_aspect
        input_handler.mouse_sensitivity = cfg.controls.mouse_sensitivity
        input_handler.invert_mouse = cfg.controls.invert_mouse
        ctx.set_resolution(cfg.video.width, cfg.video.height)
        ctx.set_fullscreen(cfg.video.fullscreen)

    input_handler = InputHandler(
        mouse_sensitivity=config.controls.mouse_sensitivity,
        invert_mouse=config.controls.invert_mouse,
    )

    menu_mgr = MenuManager(
        wad=stack,
        config=config,
        bindings=input_handler.bindings,
        on_config_changed=apply_config,
    )

    def on_quit() -> None:
        nonlocal running
        running = False

    menu_mgr.on_quit_game = on_quit
    menu_mgr.on_end_game = on_quit

    apply_config(config)

    pygame.mouse.set_relative_mode(True)

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if menu_mgr.active or menu_mgr.prompt_type != PromptType.NONE:
                    menu_mgr.handle_key(event.key, event.unicode)
                    if not menu_mgr.active and menu_mgr.prompt_type == PromptType.NONE:
                        pygame.mouse.set_relative_mode(True)
                else:
                    if event.key == pygame.K_ESCAPE:
                        menu_mgr.activate_menu()
                        pygame.mouse.set_relative_mode(False)
                    elif event.key == pygame.K_F11:
                        statusbar.fullscreen = not statusbar.fullscreen
                    elif event.key == pygame.K_F12:
                        out_dir = Path("out")
                        out_dir.mkdir(exist_ok=True)
                        shot_path = out_dir / f"screenshot_{int(pygame.time.get_ticks())}.png"
                        saved = renderer.save_screenshot(shot_path)
                        print(f"Saved screenshot: {saved}")
                    elif not args.fly:
                        input_handler.process_event(event)
            elif event.type == pygame.KEYUP:
                if not menu_mgr.active and menu_mgr.prompt_type == PromptType.NONE and not args.fly:
                    input_handler.process_event(event)
            elif event.type == pygame.MOUSEMOTION:
                if menu_mgr.active:
                    menu_mgr.handle_mouse_move(event.pos[0], event.pos[1])
                elif not args.fly:
                    input_handler.process_event(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if menu_mgr.active:
                    menu_mgr.handle_mouse_click(event.button)
                elif not args.fly:
                    input_handler.process_event(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                if not menu_mgr.active and not args.fly:
                    input_handler.process_event(event)

        if menu_mgr.active or menu_mgr.prompt_type != PromptType.NONE:
            menu_mgr.tick()
        else:
            if args.fly:
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

            for _ in range(accumulator.advance(dt)):
                if demo_player is not None:
                    if demo_player.is_finished:
                        running = False
                        break
                    cmds = demo_player.get_next_cmds()
                    sim.tick(cmds)
                elif not args.fly:
                    cmd = input_handler.build_ticcmd(player_class=player.player_class)
                    sim.tick([cmd])
                    if demo_recorder is not None:
                        demo_recorder.record_tic([cmd])
                else:
                    sim.tick()
                    if demo_recorder is not None:
                        demo_recorder.record_tic([p.cmd for p in sim.players if p.cmd])

                statusbar.tick(player)
                if not args.fly:
                    camera.save_prev_state()
                    camera.set_from_fixed(
                        player.mo.x,
                        player.mo.y,
                        player.viewz,
                        player.mo.angle,
                        player.lookdir,
                    )

        renderer.render_frame(camera, sim, alpha=accumulator.alpha)
        if not args.fly:
            overlay.clear()
            psprite_renderer.draw(
                overlay,
                player,
                fullscreen=statusbar.fullscreen,
                alpha=accumulator.alpha,
            )
            statusbar.draw(overlay, player, level_time=sim.level_time)
            if menu_mgr.active or menu_mgr.prompt_type != PromptType.NONE:
                menu_mgr.draw(overlay)
            overlay.render(
                ctx.ctx,
                renderer.tex_mgr,
                palette_index=get_sb_palette(player),
            )
        elif menu_mgr.active or menu_mgr.prompt_type != PromptType.NONE:
            overlay.clear()
            menu_mgr.draw(overlay)
            overlay.render(
                ctx.ctx,
                renderer.tex_mgr,
                palette_index=get_sb_palette(player),
            )
        ctx.swap_buffers()

    if demo_recorder is not None:
        saved_demo = demo_recorder.finish()
        print(f"Recorded demo: {saved_demo.header.total_tics} tics to {demo_recorder.file_path}")

    if demo_player is not None:
        matched = demo_player.check_determinism(sim)
        status = "PASSED" if matched else "FAILED"
        print(f"Demo playback finished. Determinism check: {status}")

    ctx.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the PyHexen CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    iwad_path = resolve_iwad(args.iwad)
    if iwad_path is None and not args.headless_frames:
        iwad_path = prompt_for_iwad()

    if iwad_path is None:
        msg = "Error: HEXEN.WAD could not be located. Specify --iwad <path> or set PYHEXEN_IWAD."
        print(msg)
        if not args.headless_frames:
            show_error_dialog("HEXEN.WAD Not Found", msg)
        return 1

    context_info = {
        "warp": str(args.warp),
        "skill": str(args.skill),
        "player_class": str(args.player_class),
        "fly": str(args.fly),
        "iwad": str(iwad_path),
    }
    with crash_guard(context_info=context_info, show_dialog=not args.headless_frames):
        return _run_game(args, iwad_path)


if __name__ == "__main__":
    raise SystemExit(main())
