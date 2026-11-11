# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""T15.2 with the real IWAD: game sounds resolve to real, decodable lumps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pyhexen.app.game_flow import GameFlow
from pyhexen.audio.game_audio import GameAudio
from pyhexen.audio.music import NullMusicBackend
from pyhexen.audio.sfx import SfxCache, decode_dmx_sfx
from pyhexen.sim.specials.dispatcher import execute_line_special
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.wadfile import WadStack


@dataclass
class RecordingMixer:
    playing: dict[int, str] = field(default_factory=dict)
    played: list[str] = field(default_factory=list)

    def play(self, channel: int, lump: str, left: float, right: float) -> bool:
        self.playing[channel] = lump
        self.played.append(lump)
        return True

    def set_gains(self, channel: int, left: float, right: float) -> None:
        pass

    def stop(self, channel: int) -> None:
        self.playing.pop(channel, None)

    def is_playing(self, channel: int) -> bool:
        return channel in self.playing


@pytest.mark.wad
def test_map01_doors_music_and_sounds_with_real_iwad(iwad_path: Path) -> None:
    stack = WadStack(iwad_path)
    mapinfo = load_mapinfo(stack)
    mixer = RecordingMixer()
    audio = GameAudio(stack, mapinfo, mixer=mixer, music_backend=NullMusicBackend())
    flow = GameFlow(stack, mapinfo)
    flow.on_level_loaded = lambda sim, info: audio.start_level(sim.map_num)
    flow.new_game(0, 2)

    def tick() -> None:
        audio.handle_events(flow.tick())
        audio.tick(flow.player.mo if flow.player else None)

    tick()
    sim = flow.sim
    assert sim is not None and sim.level is not None
    assert audio.music.current_lump is not None  # MAP01's song from SNDINFO/MAPINFO
    assert audio.music.get_midi_bytes(audio.music.current_lump).startswith(b"MThd")

    # Open every unlocked door on the map and let things run for 5 seconds.
    for line in sim.level.lines:
        if line.special in (10, 11, 12):
            execute_line_special(sim, line.special, line.args, line, flow.player.mo)
    for _ in range(35 * 5):
        mixer.playing.clear()  # samples end immediately
        tick()

    assert mixer.played, "no sounds were played"
    cache = SfxCache(stack)
    for lump in set(mixer.played):
        assert lump in stack, lump
        decode_dmx_sfx(stack.get_bytes(lump))
    # Title, hub and finale music exist and convert.
    for song in ("HEXEN", "HUB", "HALL", "ORB", "CHESS"):
        assert audio.music.get_midi_bytes(song).startswith(b"MThd"), song
    assert cache is not None
