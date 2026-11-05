# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for config loading, saving, and IWAD resolution order."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.app.config import Config, load_config, resolve_iwad, save_config


def test_config_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg = Config(iwad="C:/games/HEXEN.WAD")
    cfg.video.width = 1920
    cfg.video.height = 1080
    cfg.audio.sfx_volume = 0.5
    save_config(cfg, cfg_path)

    loaded = load_config(cfg_path)
    assert loaded.iwad == "C:/games/HEXEN.WAD"
    assert loaded.video.width == 1920
    assert loaded.video.height == 1080
    assert loaded.audio.sfx_volume == 0.5


def test_iwad_resolution_priority(tmp_path: Path) -> None:
    cli_wad = tmp_path / "cli_hexen.wad"
    env_wad = tmp_path / "env_hexen.wad"
    cfg_wad = tmp_path / "cfg_hexen.wad"
    cli_wad.write_bytes(b"IWAD")
    env_wad.write_bytes(b"IWAD")
    cfg_wad.write_bytes(b"IWAD")

    cfg_file = tmp_path / "config.toml"
    save_config(Config(iwad=str(cfg_wad)), cfg_file)
    env = {"PYHEXEN_IWAD": str(env_wad)}

    # 1. CLI wins over env and config.toml
    assert resolve_iwad(cli_wad, env=env, config_path=cfg_file) == cli_wad

    # 2. Env wins over config.toml when CLI is None
    assert resolve_iwad(None, env=env, config_path=cfg_file) == env_wad

    # 3. config.toml used when CLI and env are empty
    assert resolve_iwad(None, env={}, config_path=cfg_file) == cfg_wad

    # 4. None returned when nothing is configured
    assert resolve_iwad(None, env={}, config_path=tmp_path / "missing.toml") is None


@pytest.mark.wad
def test_wad_marker_skips_cleanly_without_iwad(iwad_path: Path) -> None:
    assert iwad_path.is_file()
