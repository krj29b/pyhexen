# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""CLI entry point for PyHexen."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pygame

from pyhexen import __version__
from pyhexen.app.config import Config, load_config, resolve_iwad
from pyhexen.app.crash import crash_guard
from pyhexen.app.dialog import prompt_for_iwad, show_error_dialog
from pyhexen.app.game_flow import GameFlow, GameState
from pyhexen.app.game_ui import GameUI
from pyhexen.app.input import InputHandler
from pyhexen.app.main_loop import FixedTimestepAccumulator
from pyhexen.app.netgame import NetGameDriver, iwad_md5, parse_address
from pyhexen.audio.game_audio import GameAudio
from pyhexen.core.ticcmd import TicCmd
from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.render.psprites import PspriteRenderer
from pyhexen.render.renderer import WorldRenderer
from pyhexen.sim.game_sim import GameSim
from pyhexen.ui.menu import MenuManager, PromptType
from pyhexen.ui.statusbar import StatusBar, get_sb_palette
from pyhexen.wad.mapinfo import MapInfo, MapInfoTable, load_mapinfo
from pyhexen.wad.wadfile import WadStack

if TYPE_CHECKING:
    from pyhexen.sim.demo import DemoPlayer
    from pyhexen.sim.player import Player


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
    parser.add_argument("--nosound", action="store_true", help="Disable sound effects and music")
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
    net = parser.add_argument_group("netgame (ADR-013)")
    net.add_argument(
        "--host",
        type=int,
        nargs="?",
        const=2342,
        default=None,
        metavar="PORT",
        help="Host a netgame on a UDP port (default 2342); the host also plays",
    )
    net.add_argument("--join", type=str, default=None, metavar="HOST[:PORT]", help="Join a netgame")
    net.add_argument("--name", type=str, default=None, help="Player name in a netgame")
    net.add_argument(
        "--players", type=int, default=0, help="Host: start as soon as this many have joined"
    )
    net.add_argument("--deathmatch", action="store_true", help="Host: deathmatch instead of coop")
    net.add_argument("--timer", type=int, default=0, help="Host: deathmatch map time in minutes")
    net.add_argument(
        "--randclass", action="store_true", help="Host: random class at every deathmatch spawn"
    )
    net.add_argument(
        "--net-bot",
        action="store_true",
        help="Headless netgame: play with the scripted bot instead of idle input",
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


def describe_view(sim: GameSim | None, player: Player | None) -> str:
    """One-line description of where the player is, saved next to F12 screenshots."""
    if sim is None or player is None or player.mo is None or sim.level is None:
        return "no level"
    mo = player.mo
    sector = sim.level.point_in_subsector(mo.x, mo.y).sector
    angle = (mo.angle & 0xFFFFFFFF) * 360.0 / 2**32
    return (
        f"map={sim.level.name} x={mo.x / 65536:.1f} y={mo.y / 65536:.1f} "
        f"z={mo.z / 65536:.1f} viewz={player.viewz / 65536:.1f} angle={angle:.1f} "
        f"lookdir={player.lookdir} sector={sector.index} "
        f"floor={sector.floor_height / 65536:.0f} ceiling={sector.ceiling_height / 65536:.0f} "
        f"tic={sim.level_time}"
    )


def _run_headless_netgame(net: NetGameDriver, flow: GameFlow, tics: int, *, bot: bool) -> int:
    """Play `tics` tics of a netgame without a window (S2d end-to-end runs).

    Prints one `NETGAME ...` line with this node's state hash at that tic; the exit code
    is 0 when the tic was reached with no desync.
    """
    import time

    from pyhexen.net.session import sim_hash
    from pyhexen.tools.playthrough import Bot

    player_bot: Bot | None = None

    def make_cmd() -> TicCmd:
        if player_bot is None:
            return TicCmd()
        return player_bot.cmd(flow)

    deadline = time.monotonic() + 60 + tics / 35 * 4
    while time.monotonic() < deadline and not net.failed:
        net.pump()
        if net.started:
            if bot and player_bot is None:
                player_bot = Bot(seed=net.client.player_num + 1)
            net.tick(1, make_cmd, max_tics=max(0, tics - net.client.run_tic))
            if net.client.run_tic >= tics:
                break
        time.sleep(0.0005)
    # Keep the connection up so the others get their last tics (the host serves them).
    linger = time.monotonic() + 5.0
    while time.monotonic() < linger and not net.failed:
        net.pump()
        if net.started:
            net.tick(1, make_cmd, max_tics=0)
        if net.peers_have(tics - 1) and (net.is_host or time.monotonic() > linger - 4.0):
            break
        time.sleep(0.001)
    sim = flow.sim
    digest = sim_hash(sim) if sim is not None else 0
    print(
        f"NETGAME player={net.client.player_num} tic={net.client.run_tic} "
        f"map={sim.map_num if sim is not None else 0} hash={digest:08x} "
        f"desync={net.client.desync_tic} state={net.client.state.value}",
        flush=True,
    )
    # A host leaving after this node reached the tic is the normal end, not a failure.
    ok = net.client.run_tic >= tics and net.client.desync_tic < 0
    net.close()
    return 0 if ok else 1


def _run_game(args: argparse.Namespace, iwad_path: Path) -> int:
    """Set up the game-flow controller and run the main rendering/game loop."""
    config = load_config()
    stack = WadStack(iwad_path)
    if args.file:
        for pwad in args.file:
            stack.add_wad(pwad)

    from pyhexen.app.savegame import SaveGameManager, compute_wad_checksum
    from pyhexen.sim.demo import (
        DemoPlayer,
        DemoRecorder,
        build_demo_sim,
        play_demo_to_completion,
    )

    map_table = load_mapinfo(stack)
    pclass_idx = {"fighter": 0, "cleric": 1, "mage": 2}.get(args.player_class.lower(), 0)
    skill = args.skill - 1

    demo_player: DemoPlayer | None = None
    if args.playdemo or args.timedemo:
        demo_player = DemoPlayer.from_file(args.playdemo or args.timedemo, wad=stack)

    if args.timedemo:
        assert demo_player is not None
        sim = build_demo_sim(stack, map_table, demo_player.demo)
        matched = play_demo_to_completion(sim, demo_player)
        if not demo_player.has_state_hash:
            print(
                f"Timedemo finished: {demo_player.total_tics} tics. This demo has no state "
                "hash, so sync was not checked (vanilla demos: python -m "
                "pyhexen.tools.demo_sync)."
            )
            return 0
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
    saves: SaveGameManager | None = None
    if not is_headless:
        saves = SaveGameManager("saves", iwad_checksum=compute_wad_checksum(stack))
    flow = GameFlow(stack, map_table, saves=saves)
    game_ui: GameUI | None = None
    demo_recorder: DemoRecorder | None = None
    audio: GameAudio | None = None
    if not is_headless and not args.nosound:
        audio = GameAudio(
            stack,
            map_table,
            sfx_volume=config.audio.sfx_volume,
            music_volume=config.audio.music_volume,
        )
        flow.title.on_music = audio.play_music
        flow.intermission.on_music = audio.play_music
        flow.intermission.on_sound = audio.ui_sound
        flow.finale.on_music = audio.play_music

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
        renderer.load_map(stack, sim.level, info, event_bus=sim.events)
        update_camera(snap=True)
        statusbar.deathmatch = sim.deathmatch
        if game_ui is not None:
            game_ui.on_level_loaded(sim, info)
        if audio is not None:
            audio.start_level(sim.map_num)
        if not is_headless:
            pygame.display.set_caption(f"PyHexen - {sim.level.name}")
        if args.record and demo_recorder is None:
            demo_recorder = DemoRecorder(
                sim, file_path=args.record, iwad_checksum=compute_wad_checksum(stack)
            )

    flow.on_level_loaded = on_level_loaded

    net: NetGameDriver | None = None
    if args.host is not None or args.join:
        md5 = iwad_md5(iwad_path, args.file or ())
        if args.host is not None:
            start_map = warp_map or map_table.translate_warp(1)
            net = NetGameDriver.host(
                flow,
                md5,
                port=args.host,
                name=args.name or "HOST",
                player_class=pclass_idx,
                start_players=args.players,
                map_num=start_map if start_map > 0 else 1,
                skill=skill,
                deathmatch=args.deathmatch,
                timer_minutes=args.timer,
                random_class=args.randclass,
            )
        else:
            net = NetGameDriver.join(
                flow,
                md5,
                parse_address(args.join),
                name=args.name or "PLAYER",
                player_class=pclass_idx,
            )
    elif demo_player is not None:
        flow.demo_playback = True
        demo_sim = build_demo_sim(stack, map_table, demo_player.demo)
        flow.surface_animator = demo_sim.surface_animator
        flow.enter_level(demo_sim)
    elif warp_map is not None:
        flow.init_new(warp_map, pclass_idx, skill)
    else:
        flow.start_title()
    if net is not None:
        flow.state = GameState.TITLE  # the lobby is shown until the host starts
        flow.title.active = False

    if is_headless and net is not None:
        code = _run_headless_netgame(net, flow, args.headless_frames, bot=args.net_bot)
        ctx.close()
        return code

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
        ui.line_vis.fov_deg = cfg.video.fov
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
    if audio is not None:
        menu_mgr.on_sound = audio.ui_sound

    # Automap, HUD messages and the save/load menu (sets the menu's save/load callbacks).
    ui = GameUI(stack, flow, menu_mgr, saves=saves, statusbar=statusbar)
    game_ui = ui
    if flow.sim is not None:  # a --warp level was loaded before the UI existed
        ui.on_level_loaded(flow.sim, map_table.get(flow.sim.map_num))

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
            elif net is not None and (not net.started or net.failed):
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        net.host_start()
            elif net is not None and flow.state == GameState.INTERMISSION and not menu_open():
                # Netgame intermissions skip on a player's fire/use ticcmd (CheckForSkip).
                if event.type in (
                    pygame.KEYDOWN,
                    pygame.KEYUP,
                    pygame.MOUSEBUTTONDOWN,
                    pygame.MOUSEBUTTONUP,
                ):
                    input_handler.process_event(event)
            elif event.type == pygame.KEYDOWN:
                if menu_open():
                    menu_mgr.handle_key(event.key, event.unicode)
                elif flow.state == GameState.TITLE:
                    # F-keys work; any other key opens the menu (H2_Responder).
                    ui.handle_title_key(event.key, event.unicode)
                elif event.key == pygame.K_ESCAPE and not in_level():
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
                    info = describe_view(flow.sim, flow.player)
                    Path(saved).with_suffix(".txt").write_text(info + "\n", encoding="utf-8")
                    print(f"Saved screenshot: {saved}\n{info}")
                elif ui.handle_key_down(event.key, event.unicode):
                    pass  # the menu (Escape, F-keys) or the automap (Tab, map keys) ate it
                elif not args.fly:
                    input_handler.process_event(event)
            elif event.type == pygame.KEYUP:
                if not menu_open() and in_level():
                    ui.handle_key_up(event.key)
                    if not args.fly:
                        input_handler.process_event(event)
            elif event.type == pygame.MOUSEWHEEL:
                if not menu_open() and in_level():
                    ui.handle_mouse_wheel(event.y)
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

        def after_tic(events: list[Any], cmds: list[TicCmd]) -> None:
            if audio is not None:
                audio.handle_events(events)
                listener = flow.player.mo if in_level() and flow.player else None
                audio.tick(listener)
                audio.set_volumes(config.audio.sfx_volume, config.audio.music_volume)
            ui.tick()
            if flow.sim is not None:
                statusbar.in_game_mask = sum(
                    1 << i for i, p in enumerate(flow.sim.players) if p.in_game
                )
            if in_level():
                tic_player = flow.player
                assert tic_player is not None
                ui.mark_seen()
                statusbar.tick(tic_player)
                if not args.fly:
                    update_camera()

        if net is not None:
            # A netgame never pauses: every node runs the tics the server sends.
            due = accumulator.advance(dt)
            net.pump()

            def make_cmd() -> TicCmd:
                nonlocal suicide_pending
                if menu_open():
                    return TicCmd()
                me = flow.player
                cmd = input_handler.build_ticcmd(
                    player_class=me.player_class if me is not None else flow.player_class,
                    suicide=suicide_pending,
                )
                suicide_pending = False
                return cmd

            for events in net.tick(due, make_cmd):
                after_tic(events or [], [])
        # The menu pauses a single-player level (MN_ActivateMenu); other screens keep running.
        elif not (menu_open() and flow.state == GameState.LEVEL):
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
                events = flow.tick(cmds)
                if was_level and demo_recorder is not None and demo_player is None:
                    demo_recorder.record_tic(cmds)
                after_tic(events, cmds)

        player = flow.player
        palette = 0
        overlay.clear()
        if in_level() and flow.sim is not None and player is not None:
            if ui.view_hidden:
                ctx.clear(0.0, 0.0, 0.0, 1.0)  # the parchment automap replaces the view
            else:
                renderer.render_frame(camera, flow.sim, alpha=accumulator.alpha)
            if not args.fly:
                if not ui.automap_active:
                    psprite_renderer.draw(
                        overlay, player, fullscreen=statusbar.fullscreen, alpha=accumulator.alpha
                    )
                ui.draw_automap(overlay)
                statusbar.draw(overlay, player, level_time=flow.sim.level_time)
                ui.draw_message(overlay, player)
                palette = get_sb_palette(player)
        else:
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            if net is not None and (not net.started or net.failed):
                net.draw_lobby(overlay)
            elif flow.state == GameState.TITLE:
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

    if net is not None:
        net.close()

    if demo_recorder is not None:
        saved_demo = demo_recorder.finish()
        print(f"Recorded demo: {saved_demo.header.total_tics} tics to {demo_recorder.file_path}")

    if demo_player is not None and flow.sim is not None:
        if demo_player.has_state_hash:
            matched = demo_player.check_determinism(flow.sim)
            status = "PASSED" if matched else "FAILED"
            print(f"Demo playback finished. Determinism check: {status}")
        else:
            print("Demo playback finished. The demo has no state hash; sync not checked.")

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
