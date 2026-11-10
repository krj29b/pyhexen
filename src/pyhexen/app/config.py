# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Configuration loading, saving, and IWAD path discovery."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

DEFAULT_CONFIG_PATH = Path("config.toml")


@dataclass(slots=True)
class VideoConfig:
    width: int = 1280
    height: int = 800
    fullscreen: bool = False
    vsync: bool = True
    fov: float = 90.0
    authentic_aspect: bool = True
    authentic_lowres: bool = False
    true_pitch: bool = False
    smooth_filter: bool = False


@dataclass(slots=True)
class AudioConfig:
    sfx_volume: float = 0.8
    music_volume: float = 0.7
    music_backend: str = "pygame"
    soundfont: str = ""


@dataclass(slots=True)
class ControlsConfig:
    mouse_sensitivity: float = 1.0
    invert_mouse: bool = False
    bindings: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class Config:
    """Application configuration and IWAD path settings."""

    iwad: str = ""
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    controls: ControlsConfig = field(default_factory=ControlsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        video_data = data.get("video", {})
        audio_data = data.get("audio", {})
        controls_data = data.get("controls", {})
        return cls(
            iwad=str(data.get("iwad", "")),
            video=VideoConfig(**{k: v for k, v in video_data.items() if hasattr(VideoConfig, k)}),
            audio=AudioConfig(**{k: v for k, v in audio_data.items() if hasattr(AudioConfig, k)}),
            controls=ControlsConfig(
                **{k: v for k, v in controls_data.items() if hasattr(ControlsConfig, k)}
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: Path | str | None = None) -> Config:
    """Load Config from a TOML file, returning defaults if the file does not exist."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        return Config()
    with path.open("rb") as f:
        data = tomllib.load(f)
    return Config.from_dict(data)


def save_config(config: Config, config_path: Path | str | None = None) -> None:
    """Save Config to a TOML file."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(config.to_dict(), f)


def resolve_iwad(
    cli_iwad: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
    config_path: Path | str | None = None,
    config: Config | None = None,
    must_exist: bool = True,
) -> Path | None:
    """Resolve the IWAD path in priority order:
    1. CLI `--iwad` argument
    2. `PYHEXEN_IWAD` environment variable
    3. `iwad =` in `config.toml`

    Returns the resolved `Path` if configured (and existing when `must_exist=True`),
    or `None` if no valid IWAD is found.
    """
    candidates: list[str] = []

    if cli_iwad:
        candidates.append(str(cli_iwad))

    environ = env if env is not None else os.environ
    env_iwad = environ.get("PYHEXEN_IWAD", "").strip()
    if env_iwad:
        candidates.append(env_iwad)

    cfg = config if config is not None else load_config(config_path)
    if cfg.iwad.strip():
        candidates.append(cfg.iwad.strip())

    for raw in candidates:
        candidate = Path(raw).expanduser()
        if not must_exist or candidate.is_file():
            return candidate

    return None
