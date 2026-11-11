# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sound channels, distance attenuation and stereo separation (T11.2, reworked in T15.2).

Port of S_StartSoundAtVolume, S_StopSound, S_StopAllSound, S_GetSoundPlayingInfo and
S_UpdateSounds (Hexen s_sound.c), with the SDL panning of Chocolate Doom's
i_sdlsound.c. Hexen differs from Doom here: volume comes from the SNDCURVE lump
(MAX_SND_DIST = 2025), separation has its own formula, there are 8 channels, and a
mobj other than a player may play only one sound at a time.

Audio is not part of the simulation, so floats are allowed for mixer gains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pyhexen.audio.sfx_table import SFX_TABLE
from pyhexen.core.angles import point_to_angle2
from pyhexen.core.events import SoundEvent
from pyhexen.core.fixed import FRACBITS
from pyhexen.wad.sndinfo import SndInfo
from pyhexen.wad.wadfile import normalize_lump_name

MAX_SND_DIST: int = 2025  # sounds.h
PRIORITY_MAX_ADJUST: int = 10
DIST_ADJUST: int = MAX_SND_DIST // PRIORITY_MAX_ADJUST
NORM_SEP: int = 128
NORM_PITCH: int = 128
HEXEN_CHANNELS: int = 8  # S_Init clamps snd_Channels to 8
DEFAULT_MAX_VOLUME: int = 10  # snd_MaxVolume, 0..15

# Tag name (lower case) -> sound id, for S_GetSoundID.
_SOUND_IDS: dict[str, int] = {tag.lower(): i for i, (tag, *_rest) in enumerate(SFX_TABLE) if tag}


def sound_id_for(name_or_id: int | str) -> int:
    """Return the `SfxEnum` id for a sound id or SNDINFO tag name (0 if unknown).

    Port of S_GetSoundID (s_sound.c).
    """
    if isinstance(name_or_id, int):
        return name_or_id if 0 < name_or_id < len(SFX_TABLE) else 0
    return _SOUND_IDS.get(name_or_id.lower(), 0)


def sound_distance(listener: Any, x: int, y: int) -> int:
    """Approximate map-unit distance used by s_sound.c (`absx + absy - min/2`)."""
    absx = abs(x - int(listener.x))
    absy = abs(y - int(listener.y))
    dist = absx + absy - ((absy if absx > absy else absx) >> 1)
    return max(0, dist >> FRACBITS)


def sound_separation(listener: Any, view_angle: int, x: int, y: int) -> int:
    """Stereo separation 0..255 for a source at `(x, y)` (s_sound.c formula)."""
    angle = point_to_angle2(int(listener.x), int(listener.y), x, y)
    angle = ((angle - view_angle) & 0xFFFFFFFF) >> 24
    sep = angle * 2 - 128
    if sep < 64:
        sep = -sep
    if sep > 192:
        sep = 512 - sep
    return sep


def sound_volume(curve: bytes, dist: int, max_volume: int, volume: int) -> int:
    """`(SoundCurve[dist] * (snd_MaxVolume * 8) * volume) >> 14` from s_sound.c."""
    if dist >= len(curve):
        return 0
    return (curve[dist] * (max_volume * 8) * volume) >> 14


def stereo_gains(volume: int, separation: int) -> tuple[float, float]:
    """Return `(left, right)` gains 0..1, as I_SDL_UpdateSoundParams computes them."""
    left = max(0, min(255, ((254 - separation) * volume) // 127))
    right = max(0, min(255, (separation * volume) // 127))
    return (left / 255.0, right / 255.0)


@dataclass(slots=True)
class _NoListener:
    """Stand-in listener outside a level (GetSoundListener's dummy mobj)."""

    x: int = 0
    y: int = 0
    angle: int = 0
    player: Any = None


def _origin_xy(origin: Any) -> tuple[int, int]:
    if hasattr(origin, "sound_org_x"):
        return (int(origin.sound_org_x), int(origin.sound_org_y))
    return (int(origin.x), int(origin.y))


class Mixer(Protocol):
    """Low-level playback used by `SoundSystem` (the I_* sound interface)."""

    def play(self, channel: int, lump: str, left: float, right: float) -> bool: ...

    def set_gains(self, channel: int, left: float, right: float) -> None: ...

    def stop(self, channel: int) -> None: ...

    def is_playing(self, channel: int) -> bool: ...


class NullMixer:
    """A mixer that plays nothing; sounds finish immediately."""

    def play(self, channel: int, lump: str, left: float, right: float) -> bool:
        return False

    def set_gains(self, channel: int, left: float, right: float) -> None:
        pass

    def stop(self, channel: int) -> None:
        pass

    def is_playing(self, channel: int) -> bool:
        return False


@dataclass(slots=True)
class SoundChannel:
    """Port of channel_t (s_sound.c)."""

    origin: Any = None
    sound_id: int = 0
    priority: int = 0
    volume: int = 127
    handle: bool = False


class SoundSystem:
    """Hexen sound channel manager. Port of s_sound.c.

    `listener` is the console player's mobj (anything with `x`, `y`, `angle`).
    Sounds with origin `None` play at the listener.
    """

    def __init__(
        self,
        sndinfo: SndInfo | None = None,
        *,
        curve: bytes | None = None,
        mixer: Mixer | None = None,
        num_channels: int = HEXEN_CHANNELS,
        max_volume: int = DEFAULT_MAX_VOLUME,
    ) -> None:
        self.sndinfo = sndinfo
        # Without SNDCURVE, fall back to a linear fade over MAX_SND_DIST.
        self.curve: bytes = (
            curve
            if curve is not None
            else bytes(127 * (MAX_SND_DIST - d) // MAX_SND_DIST for d in range(MAX_SND_DIST))
        )
        self.mixer: Mixer = mixer if mixer is not None else NullMixer()
        self.channels: list[SoundChannel] = [SoundChannel() for _ in range(num_channels)]
        self.max_volume: int = max_volume
        self._no_listener = _NoListener()
        self.listener: Any = self._no_listener
        self.view_angle: int = 0
        self._sndcount: int = 0

    # -- helpers -------------------------------------------------------

    def lump_for(self, sound_id: int) -> str | None:
        """SNDINFO lump for an `SfxEnum` id (I_GetSfxLumpNum)."""
        tag = SFX_TABLE[sound_id][0]
        if self.sndinfo is not None:
            lump = self.sndinfo.get_lump(tag)
            if lump is not None:
                return lump
        try:
            return normalize_lump_name(tag)
        except ValueError:
            return None

    def set_listener(self, listener: Any, view_angle: int | None = None) -> None:
        """Set the listening mobj and view angle (defaults to the mobj's angle).

        `None` (no level) uses a dummy listener at the map origin, so menu and
        intermission sounds still play at full volume.
        """
        self.listener = listener if listener is not None else self._no_listener
        angle = view_angle if view_angle is not None else self.listener.angle
        self.view_angle = int(angle) & 0xFFFFFFFF

    def _params(self, origin: Any, dist: int, volume: int) -> tuple[int, int]:
        vol = sound_volume(self.curve, dist, self.max_volume, volume)
        if origin is self.listener:
            return vol, NORM_SEP
        x, y = _origin_xy(origin)
        return vol, sound_separation(self.listener, self.view_angle, x, y)

    # -- s_sound.c ------------------------------------------------------

    def handle_event(self, event: SoundEvent) -> None:
        """Apply a `SoundEvent` from the simulation."""
        if event.stop:
            self.stop_sound(event.origin)
        else:
            self.start_sound(event.origin, event.sound_id, event.volume)

    def start_sound(self, origin: Any, sound: int | str, volume: int = 127) -> int | None:
        """Port of S_StartSoundAtVolume. Returns the channel used, or None."""
        sound_id = sound_id_for(sound)
        if sound_id == 0 or self.max_volume == 0 or volume <= 0:
            return None
        if origin is None:
            origin = self.listener

        if origin is not self.listener:
            dist = sound_distance(self.listener, *_origin_xy(origin))
        else:
            dist = 0
        if dist >= MAX_SND_DIST:
            return None  # beyond the hearing range
        priority = SFX_TABLE[sound_id][1] * (PRIORITY_MAX_ADJUST - dist // DIST_ADJUST)

        num = len(self.channels)
        i = num
        # Only allow mobjs one sound; players may have more than one. The channel
        # of the stopped sound is reused.
        if getattr(origin, "player", None) is None:
            same = [k for k, chan in enumerate(self.channels) if chan.origin is origin]
            if same:
                i = same[0]
                self.stop_sound(origin)
        if i >= num:
            free = [k for k, chan in enumerate(self.channels) if chan.origin is None]
            if free:
                i = free[0]
            else:
                # Look for a lower priority sound to replace.
                self._sndcount = (self._sndcount + 1) % num
                for step in range(num):
                    k = (self._sndcount + step) % num
                    if priority >= self.channels[k].priority:
                        i = k
                        break
                else:
                    return None  # no free channels
                self.mixer.stop(i)

        vol, sep = self._params(origin, dist, volume)
        lump = self.lump_for(sound_id)
        playing = lump is not None and self.mixer.play(i, lump, *stereo_gains(vol, sep))
        self.channels[i] = SoundChannel(
            origin=origin, sound_id=sound_id, priority=priority, volume=volume, handle=playing
        )
        return i

    def stop_sound(self, origin: Any) -> None:
        """Port of S_StopSound: stop every channel playing from `origin`."""
        for i, chan in enumerate(self.channels):
            if chan.origin is not None and chan.origin is origin:
                self.mixer.stop(i)
                self.channels[i] = SoundChannel()

    def stop_all(self) -> None:
        """Port of S_StopAllSound."""
        for i, chan in enumerate(self.channels):
            if chan.origin is not None:
                self.mixer.stop(i)
            self.channels[i] = SoundChannel()

    def is_playing(self, origin: Any, sound: int | str) -> bool:
        """Port of S_GetSoundPlayingInfo: is `sound` playing from `origin`?"""
        sound_id = sound_id_for(sound)
        return any(
            chan.origin is origin and chan.sound_id == sound_id and chan.handle
            for chan in self.channels
        )

    def update(self) -> None:
        """Port of S_UpdateSounds: free finished channels, follow moving origins."""
        listener = self.listener
        for i, chan in enumerate(self.channels):
            if chan.origin is None:
                continue
            if not chan.handle or not self.mixer.is_playing(i):
                self.channels[i] = SoundChannel()
                continue
            if chan.origin is listener:
                continue
            dist = sound_distance(listener, *_origin_xy(chan.origin))
            if dist >= MAX_SND_DIST:
                self.stop_sound(chan.origin)
                continue
            vol, sep = self._params(chan.origin, dist, chan.volume)
            self.mixer.set_gains(i, *stereo_gains(vol, sep))
            chan.priority = SFX_TABLE[chan.sound_id][1] * (
                PRIORITY_MAX_ADJUST - dist // DIST_ADJUST
            )


class PygameMixer:
    """`Mixer` backed by `pygame.mixer` channels and an `SfxCache`."""

    def __init__(self, sfx_cache: Any) -> None:
        import pygame.mixer

        self._mixer = pygame.mixer
        self._sfx = sfx_cache

    def play(self, channel: int, lump: str, left: float, right: float) -> bool:
        try:
            sound = self._sfx.get_sound(lump)
        except Exception:
            return False  # missing or undecodable lump
        if sound is None:
            return False
        chan = self._mixer.Channel(channel)
        chan.play(sound)
        chan.set_volume(left, right)
        return True

    def set_gains(self, channel: int, left: float, right: float) -> None:
        self._mixer.Channel(channel).set_volume(left, right)

    def stop(self, channel: int) -> None:
        self._mixer.Channel(channel).stop()

    def is_playing(self, channel: int) -> bool:
        return bool(self._mixer.Channel(channel).get_busy())
