# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Tests for the Hexen sound channel manager (audio/channels.py, T15.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyhexen.audio.channels import (
    MAX_SND_DIST,
    NORM_SEP,
    SoundSystem,
    sound_distance,
    sound_id_for,
    sound_separation,
    sound_volume,
    stereo_gains,
)
from pyhexen.audio.sfx_table import SFX_TABLE
from pyhexen.core.angles import ANG90
from pyhexen.core.events import SoundEvent
from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.info_data import SfxEnum
from pyhexen.wad.sndinfo import parse_sndinfo

FLAT_CURVE = bytes([127] * MAX_SND_DIST)


@dataclass(eq=False)
class Thing:
    x: int = 0
    y: int = 0
    angle: int = 0
    player: Any = None


@dataclass
class FakeMixer:
    playing: dict[int, str] = field(default_factory=dict)
    log: list[tuple[str, int, str]] = field(default_factory=list)
    gains: dict[int, tuple[float, float]] = field(default_factory=dict)

    def play(self, channel: int, lump: str, left: float, right: float) -> bool:
        self.playing[channel] = lump
        self.gains[channel] = (left, right)
        self.log.append(("play", channel, lump))
        return True

    def set_gains(self, channel: int, left: float, right: float) -> None:
        self.gains[channel] = (left, right)

    def stop(self, channel: int) -> None:
        self.playing.pop(channel, None)
        self.log.append(("stop", channel, ""))

    def is_playing(self, channel: int) -> bool:
        return channel in self.playing


def _system(**kw: Any) -> tuple[SoundSystem, FakeMixer, Thing]:
    mixer = FakeMixer()
    sndinfo = parse_sndinfo("PlayerFighterPain FPAIN\nDoorLocked DRLOCK\nPickupKey KEYUP\n")
    system = SoundSystem(sndinfo, curve=FLAT_CURVE, mixer=mixer, max_volume=15, **kw)
    listener = Thing(player=object())
    system.set_listener(listener)
    return system, mixer, listener


def test_sound_table_matches_sfx_enum() -> None:
    assert len(SFX_TABLE) == int(SfxEnum.NUMSFX)
    assert SFX_TABLE[int(SfxEnum.SFX_DOOR_LOCKED)][0] == "DoorLocked"
    assert sound_id_for("doorlocked") == int(SfxEnum.SFX_DOOR_LOCKED)
    assert sound_id_for("NoSuchSound") == 0


def test_hexen_distance_volume_and_separation() -> None:
    listener = Thing()
    assert sound_distance(listener, 300 * FRACUNIT, 400 * FRACUNIT) == 300 + 400 - 150
    curve = bytes(range(127, -1, -1)) + bytes(MAX_SND_DIST)
    # (curve[dist] * (snd_MaxVolume * 8) * volume) >> 14
    assert sound_volume(curve, 0, 10, 127) == (127 * 80 * 127) >> 14
    assert sound_volume(curve, 100, 10, 127) == (27 * 80 * 127) >> 14
    # Facing east: a sound straight ahead is centred, one to the north is hard left,
    # one to the south hard right.
    assert sound_separation(listener, 0, 100 * FRACUNIT, 0) == NORM_SEP
    assert sound_separation(listener, 0, 0, 100 * FRACUNIT) <= 2  # tantoangle rounding
    left, right = stereo_gains(127, 0)
    assert left > 0.99 and right == 0.0
    assert abs(sound_separation(listener, ANG90, 0, 100 * FRACUNIT) - NORM_SEP) <= 2


def test_sound_beyond_hearing_range_does_not_play() -> None:
    system, mixer, _ = _system()
    far = Thing(x=(MAX_SND_DIST + 10) * FRACUNIT)
    assert system.start_sound(far, int(SfxEnum.SFX_DOOR_LOCKED)) is None
    assert mixer.log == []


def test_none_origin_plays_at_listener_even_outside_a_level() -> None:
    system, mixer, _ = _system()
    system.set_listener(None)  # title screen: dummy listener
    chan = system.start_sound(None, "PickupKey")
    assert chan is not None
    assert mixer.playing[chan] == "KEYUP"
    left, right = mixer.gains[chan]
    assert abs(left - right) < 0.02  # centred


def test_mobj_gets_one_sound_but_player_gets_several() -> None:
    system, mixer, listener = _system()
    monster = Thing(x=64 * FRACUNIT)
    a = system.start_sound(monster, int(SfxEnum.SFX_DOOR_LOCKED))
    b = system.start_sound(monster, int(SfxEnum.SFX_PLAYER_FIGHTER_PAIN))
    assert a == b  # the second sound replaced the first on the same channel
    assert sum(ch.origin is monster for ch in system.channels) == 1

    c = system.start_sound(listener, int(SfxEnum.SFX_DOOR_LOCKED))
    d = system.start_sound(listener, int(SfxEnum.SFX_PLAYER_FIGHTER_PAIN))
    assert c != d
    assert sum(ch.origin is listener for ch in system.channels) == 2


def test_full_channels_replace_by_priority() -> None:
    system, mixer, _ = _system()
    low = int(SfxEnum.SFX_DOOR_LOCKED)  # priority 36
    high = int(SfxEnum.SFX_PLAYER_FIGHTER_PAIN)  # priority 256
    things = [Thing(x=10 * FRACUNIT) for _ in range(9)]
    for t in things[:8]:
        assert system.start_sound(t, low) is not None
    # All 8 channels are busy: a higher-priority sound evicts one...
    assert system.start_sound(things[8], high) is not None
    # ...but a sound lower than everything playing is dropped.
    system2, _, _ = _system()
    for t in things[:8]:
        system2.start_sound(t, high)
    assert system2.start_sound(Thing(x=10 * FRACUNIT), low) is None


def test_update_frees_finished_channels_and_follows_moving_origins() -> None:
    system, mixer, _ = _system()
    mover = Thing(x=100 * FRACUNIT)
    chan = system.start_sound(mover, int(SfxEnum.SFX_DOOR_LOCKED))
    assert chan is not None
    first = mixer.gains[chan]
    mover.y = 400 * FRACUNIT  # moves to the left (north)
    system.update()
    assert mixer.gains[chan] != first
    assert mixer.gains[chan][0] > mixer.gains[chan][1]

    mover.x = (MAX_SND_DIST + 50) * FRACUNIT  # walks out of hearing range
    system.update()
    assert system.channels[chan].origin is None

    other = Thing(x=50 * FRACUNIT)
    chan2 = system.start_sound(other, int(SfxEnum.SFX_DOOR_LOCKED))
    assert chan2 is not None
    del mixer.playing[chan2]  # the sample ends
    system.update()
    assert system.channels[chan2].origin is None
    assert not system.is_playing(other, int(SfxEnum.SFX_DOOR_LOCKED))


def test_sound_events_start_and_stop() -> None:
    system, mixer, _ = _system()
    door = Thing(x=32 * FRACUNIT)
    system.handle_event(SoundEvent(sound_id="DoorLocked", origin=door))
    assert system.is_playing(door, "DoorLocked")
    system.handle_event(SoundEvent(sound_id=0, stop=True, origin=door))
    assert not system.is_playing(door, "DoorLocked")
    assert mixer.playing == {}
