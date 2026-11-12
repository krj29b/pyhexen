# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""T15.2 integration: sim events from the game flow reach the mixer and music."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.app.game_flow import GameFlow, GameState
from pyhexen.audio.game_audio import GameAudio
from pyhexen.core.events import SequenceEvent
from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.info_data import SfxEnum
from pyhexen.sim.sound import SEQ_DOOR_STONE, start_sound
from pyhexen.sim.specials.doors import ev_vertical_door
from pyhexen.sim.specials.plats import special_plat_down_wait_up_stay
from pyhexen.wad.mapinfo import MapInfo, MapInfoTable
from pyhexen.wad.wadfile import WadStack

SNDINFO = b"""
DoorOpen DOROPN
DoorCloseLight DORCLS
PlatformStart PSTART
PlatformStop PSTOP
DoorLocked DRLOCK
$MAP 1 MUSONE
"""
SNDSEQ = b"""
:DoorNormal
    playuntildone DoorOpen
    stopsound DoorCloseLight
end
:Platform
    playuntildone PlatformStart
    stopsound PlatformStop
end
"""


@dataclass
class FakeMixer:
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

    def finish_all(self) -> None:
        self.playing.clear()


@dataclass
class FakeMusic:
    played: list[bytes] = field(default_factory=list)
    volume: float = 0.0

    def play_midi(self, midi_data: bytes, *, loop: bool = True) -> bool:
        self.played.append(midi_data)
        return True

    def stop(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def set_volume(self, volume: float) -> None:
        self.volume = volume


def _midi(tag: bytes) -> bytes:
    """A tiny valid MIDI file whose track name identifies it."""
    track = b"\x00\xff\x03" + bytes([len(tag)]) + tag + b"\x00\xff\x2f\x00"
    return b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk" + len(track).to_bytes(4) + track


def _setup() -> tuple[GameFlow, GameAudio, FakeMixer, FakeMusic]:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    wb.add_lump("SNDINFO", SNDINFO).add_lump("SNDSEQ", SNDSEQ)
    wb.add_lump("MUSONE", _midi(b"one")).add_lump("HUB", _midi(b"hub"))
    mb = MapBuilder("MAP01")
    # Sector 1 is a closed door next to the start room.
    mb.add_two_sector_rooms(floor2=0, ceil2=0)
    mb.add_thing(x=64, y=128, type=1)
    wb.add_map(mb)
    stack = WadStack(wb.to_bytes())
    mapinfo = MapInfoTable(
        {1: MapInfo(map_num=1, name="ONE", lump_name="MAP01", cluster=1, warp_trans=1)}
    )
    mixer, music = FakeMixer(), FakeMusic()
    audio = GameAudio(stack, mapinfo, mixer=mixer, music_backend=music)
    flow = GameFlow(stack, mapinfo)
    flow.on_level_loaded = lambda sim, info: audio.start_level(sim.map_num)
    flow.intermission.on_music = audio.play_music
    return flow, audio, mixer, music


def _tick(flow: GameFlow, audio: GameAudio) -> None:
    audio.handle_events(flow.tick())
    audio.tick(flow.player.mo if flow.player else None)


def test_level_start_plays_map_music_from_sndinfo() -> None:
    flow, audio, _, music = _setup()
    flow.new_game(0, 2)
    _tick(flow, audio)
    assert flow.state == GameState.LEVEL
    assert music.played and b"one" in music.played[-1]


def test_door_sequence_plays_start_and_stop_sounds() -> None:
    flow, audio, mixer, _ = _setup()
    flow.new_game(0, 2)
    _tick(flow, audio)
    sim = flow.sim
    assert sim is not None and sim.level is not None
    line = next(ln for ln in sim.level.lines if ln.back_sector is sim.level.sectors[1])
    seqs: list[SequenceEvent] = []
    sim.events.subscribe(SequenceEvent, seqs.append)

    line.args = (0, 64, 0, 0, 0)  # Door_Raise speed 64 (8 units per tic)
    assert ev_vertical_door(sim, line, None)
    _tick(flow, audio)
    assert [(e.sequence, e.stop) for e in seqs] == [(SEQ_DOOR_STONE, False)]
    assert "DOROPN" in mixer.played  # SN_UpdateActiveSequences played the start sound

    for _ in range(60):
        mixer.finish_all()  # let each sample end so the sequence can move on
        _tick(flow, audio)
    assert any(e.stop for e in seqs)
    assert mixer.played[-1] == "DORCLS"  # the sequence's stop sound
    assert audio.sequences.active_sequences == []


def test_platform_sequence_and_positional_sound() -> None:
    flow, audio, mixer, _ = _setup()
    flow.new_game(0, 2)
    _tick(flow, audio)
    sim = flow.sim
    assert sim is not None and sim.level is not None
    sec = sim.level.sectors[1]
    sec.tag = 5
    sec.floor_height, sec.ceiling_height = 64 * FRACUNIT, 192 * FRACUNIT  # room to lower
    special_plat_down_wait_up_stay(sim, 62, (5, 16, 0, 0, 0), None, None)
    _tick(flow, audio)
    assert "PSTART" in mixer.played

    # A monster's sound plays from the monster and follows it.
    monster: Any = type(
        "M", (), {"x": 200 * FRACUNIT, "y": 128 * FRACUNIT, "z": 0, "player": None}
    )()
    start_sound(sim, monster, int(SfxEnum.SFX_DOOR_LOCKED))
    _tick(flow, audio)
    assert any(ch.origin is monster for ch in audio.sounds.channels)


def test_stop_all_on_new_level_and_intermission_music() -> None:
    flow, audio, mixer, music = _setup()
    flow.new_game(0, 2)
    _tick(flow, audio)
    audio.ui_sound("DoorLocked")
    assert any(ch.origin is not None for ch in audio.sounds.channels)
    audio.start_level(1)
    assert all(ch.origin is None for ch in audio.sounds.channels)
    audio.play_music("HUB")
    assert b"hub" in music.played[-1]
