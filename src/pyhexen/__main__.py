# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""CLI entry point for PyHexen."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pygame

from pyhexen import __version__
from pyhexen.app.config import Config, load_config, resolve_iwad
from pyhexen.app.crash import crash_guard
from pyhexen.app.dialog import prompt_for_iwad, show_error_dialog
from pyhexen.app.game_flow import GameFlow, GameState
from pyhexen.app.input import InputHandler
from pyhexen.app.main_loop import FixedTimestepAccumulator
from pyhexen.core.ticcmd import TicCmd
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
from pyhexen.wad.mapinfo import MapInfo, MapInfoTable, load_mapinfo
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level

if TYPE_CHECKING:
    from pyhexen.sim.demo import DemoPlayer


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
    parser.add_argument(
        "--warp",
        type=str,
        default=None,
        help="Map name or number to start on (without it the game opens on the title screen)",
    )
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


def resolve_warp(warp: str, map_table: MapInfoTable) -> int:
    """Return the map number for a --warp value (a map number or a lump name like MAP05)."""
    if warp.isdigit():
        return int(warp)
    name = warp.upper()
    info = map_table.get(name)
    if info is not None:
        return info.map_num
    if name.startswith("MAP") and name[3:].isdigit():
        return int(name[3:])
    raise SystemExit(f"Error: unknown map {warp!r}")


def window_to_virtual(
    px: int, py: int, box: tuple[int, int, int, int], window_height: int
) -> tuple[int, int]:
    """Convert window pixel coordinates to the 320x200 overlay space.

    `box` is the GL viewport returned by `Overlay2D.render` (origin bottom-left);
    window coordinates have their origin top-left.
    """
    x, y, w, h = box
    top = window_height - (y + h)
    vx = (px - x) * 320 // max(1, w)
    vy = (py - top) * 200 // max(1, h)
    return vx, vy


def _build_demo_sim(stack: WadStack, map_table: MapInfoTable, demo_player: DemoPlayer) -> GameSim:
    """Build the sim for a demo the same way S1 playback does."""
    header = demo_player.demo.header
    info = map_table.get(header.map_num)
    level = load_level(stack, info.lump_name if info else f"MAP{header.map_num:02d}")
    sim = GameSim(level=level, skill=header.skill - 1, player_class=header.player_class)
    sim.map_num = header.map_num
    sim.rng.set_state(header.rng_state)
    spawn_map_things(sim)
    spawn_player(sim, 0)
    return sim


def _run_game(args: argparse.Namespace, iwad_path: Path) -> int:
    """Set up the game-flow controller and run the main rendering/game loop."""
    config = load_config()
    stack = WadStack(iwad_path)
    if args.file:
        for pwad in args.file:
            stack.add_wad(pwad)

    from pyhexen.sim.demo import DemoPlayer, DemoRecorder, play_demo_to_completion
    from pyhexen.sim.savegame import compute_wad_checksum

    map_table = load_mapinfo(stack)
    pclass_idx = {"fighter": 0, "cleric": 1, "mage": 2}.get(args.player_class.lower(), 0)
    skill = args.skill - 1

    demo_player: DemoPlayer | None = None
    if args.playdemo or args.timedemo:
        demo_player = DemoPlayer.from_file(args.playdemo or args.timedemo, wad=stack)

    if args.timedemo:
        assert demo_player is not None
        sim = _build_demo_sim(stack, map_table, demo_player)
        matched = play_demo_to_completion(sim, demo_player)
        status = "PASSED" if matched else "FAILED"
        print(f"Timedemo finished: {demo_player.total_tics} tics. Determinism: {status}")
        return 0 if matched else 1

    # Headless runs and free-fly mode need a level, so they default to MAP01.
    is_headless = args.headless_frames > 0
    warp_map: int | None = resolve_warp(args.warp, map_table) if args.warp else None
    if warp_map is None and (is_headless or args.fly):
        warp_map = 1

    ctx = RenderContext(
        standalone=is_headless,
        width=config.video.width,
        height=config.video.height,
        vsync=config.video.vsync,
        title="PyHexen",
    )
    renderer = WorldRenderer(ctx, wad=stack)
    overlay = Overlay2D(wad=stack)
    psprite_renderer = PspriteRenderer(stack)
    statusbar = StatusBar(stack)
    camera = Camera()
    flow = GameFlow(stack, map_table)
    demo_recorder: DemoRecorder | None = None

    def update_camera(*, snap: bool = False) -> None:
        player = flow.player
        if player is None or player.mo is None:
            return
        if not snap:
            camera.save_prev_state()
        camera.set_from_fixed(
            player.mo.x, player.mo.y, player.viewz, player.mo.angle, player.lookdir, snap=snap
        )

    def on_level_loaded(sim: GameSim, info: MapInfo | None) -> None:
        nonlocal demo_recorder
        assert sim.level is not None
        # Passing the sim's event bus lets the renderer rebuild moving sectors.
        renderer.load_map(stack, sim.level, info, event_bus=sim.events, rng=sim.rng)
        update_camera(snap=True)
        if not is_headless:
            pygame.display.set_caption(f"PyHexen - {sim.level.name}")
        if args.record and demo_recorder is None:
            demo_recorder = DemoRecorder(
                sim, file_path=args.record, iwad_checksum=compute_wad_checksum(iwad_path)
            )

    flow.on_level_loaded = on_level_loaded

    if demo_player is not None:
        flow.demo_playback = True
        flow.enter_level(_build_demo_sim(stack, map_table, demo_player))
    elif warp_map is not None:
        flow.init_new(warp_map, pclass_idx, skill)
    else:
        flow.start_title()

    if is_headless:
        for _ in range(args.headless_frames):
            flow.tick()
            if flow.player is not None:
                statusbar.tick(flow.player)
            update_camera()
        player = flow.player
        renderer.render_frame(camera, flow.sim)
        if not args.fly and player is not None and flow.sim is not None:
            overlay.clear()
            psprite_renderer.draw(overlay, player, fullscreen=statusbar.fullscreen)
            statusbar.draw(overlay, player, level_time=flow.sim.level_time)
            overlay.render(ctx.ctx, renderer.tex_mgr, palette_index=get_sb_palette(player))
        if args.screenshot:
            renderer.save_screenshot(args.screenshot)
        ctx.close()
        return 0

    # Interactive loop
    clock = pygame.time.Clock()
    accumulator = FixedTimestepAccumulator()
    running = True
    suicide_pending = False
    overlay_box = (0, 0, ctx.width, ctx.height)
    relative_mouse: bool | None = None

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

    def on_suicide() -> None:
        nonlocal suicide_pending
        suicide_pending = True

    menu_mgr.on_quit_game = on_quit
    menu_mgr.on_end_game = flow.start_title
    menu_mgr.on_new_game = flow.new_game
    menu_mgr.on_suicide = on_suicide

    apply_config(config)

    def menu_open() -> bool:
        return menu_mgr.active or menu_mgr.prompt_type != PromptType.NONE

    def in_level() -> bool:
        return flow.state == GameState.LEVEL and flow.player is not None

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if menu_open():
                    menu_mgr.handle_key(event.key, event.unicode)
                elif event.key == pygame.K_ESCAPE or flow.state == GameState.TITLE:
                    # Any key on the title screen opens the menu (H2_Responder).
                    menu_mgr.activate_menu()
                elif flow.state == GameState.INTERMISSION:
                    flow.intermission.handle_key(event.key)
                elif flow.state == GameState.FINALE:
                    flow.finale.handle_key(event.key)
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
                if not menu_open() and in_level() and not args.fly:
                    input_handler.process_event(event)
            elif event.type == pygame.MOUSEMOTION:
                if menu_mgr.active:
                    vx, vy = window_to_virtual(*event.pos, overlay_box, ctx.height)
                    menu_mgr.handle_mouse_move(vx, vy)
                elif in_level() and not args.fly:
                    input_handler.process_event(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if menu_open():
                    vx, vy = window_to_virtual(*event.pos, overlay_box, ctx.height)
                    menu_mgr.handle_mouse_click(vx, vy, event.button)
                elif flow.state == GameState.INTERMISSION:
                    flow.intermission.handle_mouse_click(event.button)
                elif flow.state == GameState.FINALE:
                    flow.finale.handle_mouse_click(event.button)
                elif flow.state == GameState.TITLE:
                    menu_mgr.activate_menu()
                elif not args.fly:
                    input_handler.process_event(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                if not menu_open() and in_level() and not args.fly:
                    input_handler.process_event(event)

        want_relative = in_level() and not menu_open()
        if want_relative != relative_mouse:
            pygame.mouse.set_relative_mode(want_relative)
            relative_mouse = want_relative

        if menu_open():
            menu_mgr.tick()

        # The menu pauses a single-player level (MN_ActivateMenu); other screens keep running.
        if not (menu_open() and flow.state == GameState.LEVEL):
            if args.fly and in_level():
                keys = pygame.key.get_pressed()
                forward = float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(
                    keys[pygame.K_s] or keys[pygame.K_DOWN]
                )
                right = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
                up = float(keys[pygame.K_e] or keys[pygame.K_SPACE]) - float(
                    keys[pygame.K_c] or keys[pygame.K_LCTRL]
                )
                m_dx, m_dy = pygame.mouse.get_rel()
                camera.move_free_fly(
                    dt,
                    forward=forward,
                    right=right,
                    up=up,
                    yaw_delta=m_dx * 0.003,
                    pitch_delta=-m_dy * 0.003,
                    speed=600.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 300.0,
                )

            for _ in range(accumulator.advance(dt)):
                cmds: list[TicCmd] = []
                if demo_player is not None:
                    if demo_player.is_finished or flow.demo_finished:
                        running = False
                        break
                    cmds = list(demo_player.get_next_cmds())
                elif in_level() and not args.fly:
                    player = flow.player
                    assert player is not None
                    cmds = [
                        input_handler.build_ticcmd(
                            player_class=player.player_class, suicide=suicide_pending
                        )
                    ]
                    suicide_pending = False

                was_level = flow.state == GameState.LEVEL
                flow.tick(cmds)
                if was_level and demo_recorder is not None and demo_player is None:
                    demo_recorder.record_tic(cmds)

                if in_level():
                    player = flow.player
                    assert player is not None
                    statusbar.tick(player)
                    if not args.fly:
                        update_camera()

        player = flow.player
        palette = 0
        overlay.clear()
        if in_level() and flow.sim is not None and player is not None:
            renderer.render_frame(camera, flow.sim, alpha=accumulator.alpha)
            if not args.fly:
                psprite_renderer.draw(
                    overlay, player, fullscreen=statusbar.fullscreen, alpha=accumulator.alpha
                )
                statusbar.draw(overlay, player, level_time=flow.sim.level_time)
                palette = get_sb_palette(player)
        else:
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            if flow.state == GameState.TITLE:
                flow.title.draw(overlay)
            elif flow.state == GameState.INTERMISSION:
                flow.intermission.draw(overlay)
            elif flow.state == GameState.FINALE:
                flow.finale.draw(overlay)
        if menu_open():
            menu_mgr.draw(overlay)
        if not args.fly or menu_open() or not in_level():
            overlay_box = overlay.render(ctx.ctx, renderer.tex_mgr, palette_index=palette)
        ctx.swap_buffers()

    if demo_recorder is not None:
        saved_demo = demo_recorder.finish()
        print(f"Recorded demo: {saved_demo.header.total_tics} tics to {demo_recorder.file_path}")

    if demo_player is not None and flow.sim is not None:
        matched = demo_player.check_determinism(flow.sim)
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
