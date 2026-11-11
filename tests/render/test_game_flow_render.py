# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""T15.1 integration: the game-flow controller driving a real headless renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fixtures.wadbuilder import MapBuilder, PatchPlacement, TextureDef, WadBuilder
from pyhexen.__main__ import main
from pyhexen.app.game_flow import GameAction, GameFlow, GameState
from pyhexen.core.events import SectorChanged
from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.render.renderer import WorldRenderer
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.transitions import teleport_new_map
from pyhexen.wad.mapinfo import MapInfo, load_mapinfo
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level

MAPINFO_TEXT = b"""
map 1 "ONE"
warptrans 1
next 2
cluster 1
sky1 SKY1 0

map 2 "TWO"
warptrans 2
next 1
cluster 1
sky1 SKY1 0
"""


def _build_wad() -> WadBuilder:
    wb = WadBuilder("IWAD")
    wb.add_playpal(14)
    wb.add_colormap(33, include_fogmap=True)
    wb.add_lump("MAPINFO", MAPINFO_TEXT)
    wall = np.full((64, 64), 160, dtype=np.uint8)
    wb.add_marker("P_START")
    wb.add_picture("WALL01", wall)
    wb.add_marker("P_END")
    wb.add_textures(
        [TextureDef(name="WALL01", width=64, height=64, patches=[PatchPlacement(patch=0)])],
        pnames=["WALL01"],
    )
    wb.add_flat("FLOOR01", fill=30)
    wb.add_flat("CEIL01", fill=50)
    for name, size in (("MAP01", 512), ("MAP02", 768)):
        mb = MapBuilder(name)
        mb.add_room(0, 0, size, size, floor_pic="FLOOR01", ceiling_pic="CEIL01", light=180)
        mb.add_thing(x=size // 2, y=size // 2, type=1)
        wb.add_map(mb)
    return wb


def test_headless_main_without_warp_starts_map01(tmp_path: Path) -> None:
    wad_path = _build_wad().write(tmp_path / "SYNTH.WAD")
    shot = tmp_path / "shot.png"
    code = main(["--iwad", str(wad_path), "--headless-frames", "5", "--screenshot", str(shot)])
    assert code == 0
    assert shot.is_file()


def test_flow_renders_every_state_across_a_map_change() -> None:
    stack = WadStack(_build_wad().to_bytes())
    with stack:
        ctx = RenderContext(standalone=True, width=320, height=200)
        renderer = WorldRenderer(ctx, wad=stack)
        overlay = Overlay2D(wad=stack)
        camera = Camera()
        flow = GameFlow(stack, load_mapinfo(stack))
        loaded: list[str] = []

        def on_level_loaded(sim: GameSim, info: MapInfo | None) -> None:
            assert sim.level is not None
            renderer.load_map(stack, sim.level, info, event_bus=sim.events, rng=sim.rng)
            loaded.append(sim.level.name)

        flow.on_level_loaded = on_level_loaded

        def draw() -> None:
            overlay.clear()
            if flow.state == GameState.LEVEL:
                renderer.render_frame(camera, flow.sim)
            else:
                ctx.clear(0.0, 0.0, 0.0, 1.0)
                flow.title.draw(overlay)
                flow.intermission.draw(overlay)
            assert renderer.tex_mgr is not None
            overlay.render(ctx.ctx, renderer.tex_mgr)

        flow.start_title()
        draw()  # title screen before any level exists
        flow.new_game(0, 2)
        flow.tick()
        draw()
        assert loaded == ["MAP01"]

        assert flow.sim is not None
        teleport_new_map(flow.sim, 2, 0)
        seen: set[GameState] = set()
        for _ in range(100):
            flow.tick()
            seen.add(flow.state)
            draw()
            if flow.state == GameState.LEVEL and flow.pending_action == GameAction.NOTHING:
                if loaded[-1] == "MAP02":
                    break
        assert GameState.INTERMISSION in seen
        assert loaded == ["MAP01", "MAP02"]
        assert renderer.level is flow.sim.level
        ctx.close()


def test_rebuilt_sectors_keep_rendering() -> None:
    """Regression: a sector rebuild must not leave its chunk on a released VAO.

    Rebuilds only happen when the renderer gets the sim's event bus (T15.1). The
    renderer used to release VAOs that `rebuild_dirty` had kept for the chunk, so
    moving doors and their neighbours vanished (or crashed a headless render).
    """
    wb = _build_wad()
    mb = MapBuilder("MAP03")
    # Sector 1 starts as a closed door (ceiling == floor), so opening it resizes the VBO.
    mb.add_two_sector_rooms(floor2=0, ceil2=0)
    mb.add_thing(x=64, y=128, type=1)
    wb.add_map(mb)
    stack = WadStack(wb.to_bytes())
    with stack:
        ctx = RenderContext(standalone=True, width=320, height=200)
        renderer = WorldRenderer(ctx, wad=stack)
        level = load_level(stack, "MAP03")
        sim = GameSim(level=level)
        renderer.load_map(stack, level, event_bus=sim.events)
        camera = Camera()
        camera.x, camera.y, camera.z, camera.yaw = 64.0, 128.0, 41.0, 0.0

        def shot() -> np.ndarray:
            renderer.render_frame(camera, sim)
            return np.frombuffer(ctx.ctx.fbo.read(components=3), dtype=np.uint8).copy()

        before = shot()
        # Same geometry: the VBO keeps its size and the VAOs are reused.
        for i in range(len(level.sectors)):
            sim.events.emit(SectorChanged(i))
        assert np.array_equal(shot(), before)
        assert np.array_equal(shot(), before)

        # Open the door: new geometry, new VBO size.
        level.sectors[1].ceiling_height = 128 << 16
        sim.events.emit(SectorChanged(1))
        opened = shot()
        assert not np.array_equal(opened, before)
        sim.events.emit(SectorChanged(1))
        assert np.array_equal(shot(), opened)
        ctx.close()


def test_world_keeps_depth_testing_after_the_hud_overlay() -> None:
    """Regression: the HUD overlay turned depth testing off for every later frame.

    The game draws world → HUD every frame, so from the second frame on far walls
    and sprites were painted over near walls. Headless single-frame renders never
    showed it.
    """
    wb = _build_wad()
    mb = MapBuilder("MAP04")
    # A 64-unit step up into the far room: the far room's back wall is drawn after
    # the near step wall and partly behind it, so only depth testing hides it.
    mb.add_two_sector_rooms(floor2=64, ceil1=192, ceil2=192, width1=256, width2=512)
    wb.add_map(mb)
    stack = WadStack(wb.to_bytes())
    with stack:
        ctx = RenderContext(standalone=True, width=320, height=200)
        renderer = WorldRenderer(ctx, wad=stack)
        renderer.load_map(stack, load_level(stack, "MAP04"))
        overlay = Overlay2D(wad=stack)
        camera = Camera()
        camera.x, camera.y, camera.z, camera.yaw = 32.0, 128.0, 41.0, 0.0

        def frame() -> np.ndarray:
            renderer.render_frame(camera)
            img = np.frombuffer(ctx.ctx.fbo.read(components=3), dtype=np.uint8).copy()
            overlay.clear()
            assert renderer.tex_mgr is not None
            overlay.render(ctx.ctx, renderer.tex_mgr)
            return img

        first = frame()
        assert np.array_equal(frame(), first)
        assert np.array_equal(frame(), first)
        ctx.close()
