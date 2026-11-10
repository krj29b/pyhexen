# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T14.2: Options menu and configuration persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pygame
import pytest

from pyhexen.app.config import (
    AudioConfig,
    Config,
    ControlsConfig,
    VideoConfig,
    load_config,
    save_config,
)
from pyhexen.app.input import InputHandler, KeyBindings
from pyhexen.ui.menu import COMMON_RESOLUTIONS, MenuManager, MenuType


def test_config_roundtrip_persistence() -> None:
    """Test saving and reloading config with custom video and control options."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_file = Path(tmpdir) / "test_config.toml"

        cfg = Config(
            video=VideoConfig(
                width=1920,
                height=1080,
                fullscreen=True,
                vsync=False,
                fov=105.0,
                authentic_lowres=True,
                authentic_aspect=False,
                true_pitch=True,
                smooth_filter=True,
            ),
            controls=ControlsConfig(
                mouse_sensitivity=0.8,
                invert_mouse=True,
                bindings={"forward": pygame.K_UP, "fire": pygame.K_LCTRL},
            ),
            audio=AudioConfig(sfx_volume=0.5, music_volume=0.7),
        )

        save_config(cfg, cfg_file)
        assert cfg_file.exists()

        loaded = load_config(cfg_file)
        assert loaded.video.width == 1920
        assert loaded.video.height == 1080
        assert loaded.video.fullscreen is True
        assert loaded.video.vsync is False
        assert loaded.video.fov == pytest.approx(105.0)
        assert loaded.video.authentic_lowres is True
        assert loaded.video.authentic_aspect is False
        assert loaded.video.true_pitch is True
        assert loaded.video.smooth_filter is True

        assert loaded.controls.mouse_sensitivity == pytest.approx(0.8)
        assert loaded.controls.invert_mouse is True
        assert loaded.controls.bindings == {"forward": pygame.K_UP, "fire": pygame.K_LCTRL}


def test_menu_options_getters_setters() -> None:
    """Test option getters and setters on MenuManager."""
    cfg = Config()
    changed_configs: list[Config] = []

    def on_change(c: Config) -> None:
        changed_configs.append(c)

    menu_mgr = MenuManager(config=cfg, on_config_changed=on_change)

    # Resolution
    res_idx = menu_mgr._get_resolution_idx()
    assert 0 <= res_idx < len(COMMON_RESOLUTIONS)
    menu_mgr._set_resolution_idx(len(COMMON_RESOLUTIONS) - 1)
    target_w, target_h = COMMON_RESOLUTIONS[-1]
    assert menu_mgr.config.video.width == target_w
    assert menu_mgr.config.video.height == target_h

    # Fullscreen
    assert menu_mgr._get_fullscreen() == 0
    menu_mgr._set_fullscreen(1)
    assert menu_mgr.config.video.fullscreen is True
    menu_mgr._set_fullscreen(0)
    assert menu_mgr.config.video.fullscreen is False

    # FOV slider (70 to 120 -> slider 0 to 10)
    menu_mgr._set_fov_slider(4)  # 70 + 4/10 * 50 = 90
    assert menu_mgr.config.video.fov == pytest.approx(90.0)
    assert menu_mgr._get_fov_slider() == 4
    menu_mgr._set_fov_slider(10)
    assert menu_mgr.config.video.fov == pytest.approx(120.0)
    assert menu_mgr._get_fov_slider() == 10

    # Low-res mode
    assert menu_mgr._get_lowres() == 0
    menu_mgr._set_lowres(1)
    assert menu_mgr.config.video.authentic_lowres is True

    # Pixel aspect
    assert menu_mgr._get_aspect() == 1  # default True
    menu_mgr._set_aspect(0)
    assert menu_mgr.config.video.authentic_aspect is False

    # True pitch
    assert menu_mgr._get_true_pitch() == 0
    menu_mgr._set_true_pitch(1)
    assert menu_mgr.config.video.true_pitch is True

    # Filtering
    assert menu_mgr._get_filter() == 0
    menu_mgr._set_filter(1)
    assert menu_mgr.config.video.smooth_filter is True

    # Invert mouse
    assert menu_mgr._get_invert_mouse() == 0
    menu_mgr._set_invert_mouse(1)
    assert menu_mgr.config.controls.invert_mouse is True

    # Mouse sensitivity
    menu_mgr._set_mouse_sensitivity(9)
    assert menu_mgr.config.controls.mouse_sensitivity == pytest.approx(1.0)
    assert menu_mgr._get_mouse_sensitivity() == 9


def test_video_menu_navigation_and_adjustment() -> None:
    """Test navigating to VIDEO menu and adjusting settings via keyboard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_file = Path(tmpdir) / "config.toml"
        cfg = Config()
        changes = 0

        def on_change(c: Config) -> None:
            nonlocal changes
            changes += 1

        menu_mgr = MenuManager(
            config=cfg,
            config_path=cfg_file,
            on_config_changed=on_change,
        )

        menu_mgr.activate_menu()
        menu_mgr.set_menu(MenuType.VIDEO)
        assert menu_mgr.current_menu_type == MenuType.VIDEO

        # Item 0 is RESOLUTION
        menu = menu_mgr.current_menu
        menu.cur_pos = 0
        initial_res = menu_mgr._get_resolution_idx()
        # Right arrow cycles resolution
        menu_mgr.handle_key(pygame.K_RIGHT)
        assert menu_mgr._get_resolution_idx() == (initial_res + 1) % len(COMMON_RESOLUTIONS)
        assert changes > 0

        # Item 1 is FULLSCREEN (toggle)
        menu.cur_pos = 1
        assert menu_mgr.config.video.fullscreen is False
        menu_mgr.handle_key(pygame.K_RIGHT)
        assert menu_mgr.config.video.fullscreen is True
        menu_mgr.handle_key(pygame.K_LEFT)
        assert menu_mgr.config.video.fullscreen is False

        # Item 2 is FIELD OF VIEW (clamped range)
        menu.cur_pos = 2
        menu_mgr._set_fov_slider(5)
        menu_mgr.handle_key(pygame.K_RIGHT)
        assert menu_mgr._get_fov_slider() == 6
        menu_mgr.handle_key(pygame.K_LEFT)
        assert menu_mgr._get_fov_slider() == 5

        # Item 3 is LOW-RES MODE (toggle)
        menu.cur_pos = 3
        menu_mgr.handle_key(pygame.K_RETURN)
        assert menu_mgr.config.video.authentic_lowres is True

        # Check config was saved to disk
        assert cfg_file.exists()
        reloaded = load_config(cfg_file)
        assert reloaded.video.authentic_lowres is True


def test_input_handler_invert_mouse() -> None:
    """Test that invert_mouse flag reverses mouse Y pitch delta."""
    # Standard (invert_mouse=False)
    handler = InputHandler(invert_mouse=False)
    handler._mouse_dy = 100
    cmd_normal = handler.build_ticcmd(player_class=0)

    # Inverted (invert_mouse=True)
    handler_inv = InputHandler(invert_mouse=True)
    handler_inv._mouse_dy = 100
    cmd_inv = handler_inv.build_ticcmd(player_class=0)

    # Invert mouse should invert look direction
    assert cmd_normal.look_delta == -cmd_inv.look_delta


def test_keybinding_persistence() -> None:
    """Test that rebound keys are persisted to config and loaded on init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_file = Path(tmpdir) / "config.toml"
        cfg = Config()
        bindings = KeyBindings()

        menu_mgr = MenuManager(
            config=cfg,
            bindings=bindings,
            config_path=cfg_file,
        )

        # Rebind JUMP to Key K
        menu_mgr._assign_rebound_key("JUMP", pygame.K_k)
        assert bindings.jump == pygame.K_k
        assert cfg.controls.bindings.get("jump") == pygame.K_k
        assert cfg_file.exists()

        # Create fresh MenuManager with the saved config
        loaded_cfg = load_config(cfg_file)
        fresh_bindings = KeyBindings()
        MenuManager(config=loaded_cfg, bindings=fresh_bindings)

        assert fresh_bindings.jump == pygame.K_k


def test_renderer_options(tmp_path: Path) -> None:
    """Test renderer authentic_lowres, smooth_filter, and RenderContext options."""
    from fixtures.wadbuilder import MapBuilder, WadBuilder
    from pyhexen.render.camera import Camera
    from pyhexen.render.context import RenderContext
    from pyhexen.render.renderer import WorldRenderer
    from pyhexen.wad.wadfile import WadStack
    from pyhexen.world.level import load_level

    wb = WadBuilder("IWAD")
    wb.add_playpal(2)
    wb.add_colormap(33, include_fogmap=True)
    wb.add_flat("FLOOR01", fill=10)
    wb.add_flat("CEIL01", fill=20)
    wb.add_flat("F_SKY", fill=0)

    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 512, 512, floor_pic="FLOOR01", ceiling_pic="CEIL01")
    mb.add_thing(tid=0, x=256, y=256, z=0, angle=0, type=1)
    wb.add_map(mb)

    wad_bytes = wb.to_bytes()
    with WadStack(wad_bytes) as stack:
        level = load_level(stack, "MAP01")
        ctx = RenderContext(standalone=True, width=640, height=400)
        renderer = WorldRenderer(ctx, wad=stack)
        renderer.load_map(stack, level)

        # Test context resolution change
        ctx.set_resolution(800, 600)
        assert ctx.width == 800
        assert ctx.height == 600

        # Test context fullscreen toggle
        ctx.set_fullscreen(True)
        assert ctx.fullscreen is True
        ctx.set_fullscreen(False)
        assert ctx.fullscreen is False

        # Test smooth filter toggle
        renderer.set_smooth_filter(True)
        assert renderer.smooth_filter is True
        assert renderer.tex_mgr.smooth_filter is True

        # Test authentic low-res rendering
        renderer.authentic_lowres = True
        camera = Camera()
        camera.x = 256.0
        camera.y = 32.0
        camera.z = -256.0

        renderer.render_frame(camera)
        shot_path = tmp_path / "lowres_screenshot.png"
        saved = renderer.save_screenshot(shot_path)
        assert saved.is_file()
        assert saved.stat().st_size > 0

        renderer.release()
        ctx.close()
