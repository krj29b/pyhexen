# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sound channel manager, distance attenuation, and stereo separation (T11.2).

Port of S_StartSound, S_StartSoundAtVolume, S_AdjustSoundParams, S_StopSound,
and S_UpdateSounds (s_sound.c).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyhexen.audio.sfx import SfxCache
from pyhexen.core.angles import approx_distance, fine_sine, point_to_angle2
from pyhexen.core.events import EventBus, SoundEvent
from pyhexen.core.fixed import FRACBITS, FRACUNIT, c_div, fixed_mul
from pyhexen.wad.sndinfo import SndInfo
from pyhexen.wad.wadfile import normalize_lump_name

MAX_CHANNELS: int = 16
SOUND_CLOSE_DIST: int = 160 * FRACUNIT
SOUND_MAX_DIST: int = 1200 * FRACUNIT
SOUND_ATTENUATOR: int = (SOUND_MAX_DIST - SOUND_CLOSE_DIST) >> FRACBITS
NORM_SEP: int = 128
S_STEREO_SWING: int = 96 * FRACUNIT
DEFAULT_PRIORITY: int = 64
PRIORITY_MAX_ADJUST: int = 10
DIST_ADJUST: int = (SOUND_MAX_DIST >> FRACBITS) // PRIORITY_MAX_ADJUST


def adjust_sound_params(
    listener_x: int,
    listener_y: int,
    listener_angle: int,
    source_x: int,
    source_y: int,
    base_volume: int = 127,
) -> tuple[int, int] | None:
    """Compute attenuated `(volume, separation)` for a sound source relative to a listener.

    Port of S_AdjustSoundParams / S_StartSoundAtVolume distance and pan math (s_sound.c).
    Coordinates are in 16.16 fixed-point and `listener_angle` is a 32-bit BAM angle.
    Returns `None` when the source is beyond `SOUND_MAX_DIST`.
    """
    clamped_base = max(0, min(127, int(base_volume)))
    if listener_x == source_x and listener_y == source_y:
        return (clamped_base, NORM_SEP)

    adx = abs(listener_x - source_x)
    ady = abs(listener_y - source_y)
    dist = approx_distance(adx, ady)

    if dist > SOUND_MAX_DIST:
        return None

    if dist < SOUND_CLOSE_DIST:
        vol = clamped_base
    else:
        vol = c_div(
            clamped_base * ((SOUND_MAX_DIST - dist) >> FRACBITS),
            SOUND_ATTENUATOR,
        )
        vol = max(0, min(127, vol))

    angle = (point_to_angle2(listener_x, listener_y, source_x, source_y) - listener_angle) & (
        0xFFFFFFFF
    )
    sep = NORM_SEP - (fixed_mul(S_STEREO_SWING, fine_sine(angle)) >> FRACBITS)
    sep = max(0, min(255, sep))
    return (vol, sep)


def separation_to_stereo_gains(volume: int, separation: int) -> tuple[float, float]:
    """Convert `(volume 0..127, separation 0..255)` to `(left_gain, right_gain)` in `0.0..1.0`."""
    vol_norm = max(0.0, min(1.0, volume / 127.0))
    sep_clamped = max(0, min(255, separation))
    left = vol_norm * ((255 - sep_clamped) / 127.0)
    right = vol_norm * (sep_clamped / 127.0)
    return (max(0.0, min(1.0, left)), max(0.0, min(1.0, right)))


def _extract_source_xy(origin: Any, fallback_x: int = 0, fallback_y: int = 0) -> tuple[int, int]:
    """Extract `(x, y)` in 16.16 fixed-point from an origin object (Mobj, Sector, etc.)."""
    if origin is None:
        return (fallback_x, fallback_y)
    if hasattr(origin, "x") and hasattr(origin, "y"):
        return (int(origin.x), int(origin.y))
    if hasattr(origin, "sound_org_x") and hasattr(origin, "sound_org_y"):
        return (int(origin.sound_org_x), int(origin.sound_org_y))
    return (fallback_x, fallback_y)


@dataclass(slots=True)
class SoundChannel:
    """State of a single active or inactive mixer channel.

    Port of channel_t (s_sound.c).
    """

    channel_id: int
    sound_name: str
    lump_name: str
    origin: Any
    volume: int
    separation: int
    priority: int
    loop: bool = False
    active: bool = True
    base_volume: int = 127
    base_priority: int = DEFAULT_PRIORITY
    origin_id: int | None = None
    source_x: int = 0
    source_y: int = 0


class SoundSystem:
    """Manages sound channels, spatial attenuation, priority eviction, and `SoundEvent`s.

    Port of s_sound.c.
    """

    def __init__(
        self,
        sndinfo: SndInfo | None = None,
        *,
        sfx_cache: SfxCache | None = None,
        max_channels: int = MAX_CHANNELS,
        event_bus: EventBus | None = None,
    ) -> None:
        if max_channels <= 0:
            raise ValueError(f"max_channels must be positive, got {max_channels}")
        self.sndinfo = sndinfo
        self.sfx_cache = sfx_cache
        self.max_channels = max_channels
        self.channels: list[SoundChannel | None] = [None] * max_channels
        self._sndcount: int = 0

        self.listener_x: int = 0
        self.listener_y: int = 0
        self.listener_angle: int = 0
        self.listener_origin: Any = None
        self.master_volume: int = 127

        self._origins_by_id: dict[int, Any] = {}
        self._priorities: dict[str, int] = {}

        if event_bus is not None:
            self.attach_event_bus(event_bus)

    def attach_event_bus(self, event_bus: EventBus) -> None:
        """Subscribe this `SoundSystem` to `SoundEvent`s on `event_bus`."""
        event_bus.subscribe(SoundEvent, self.handle_sound_event)

    def register_origin(self, origin: Any, origin_id: int | None = None) -> int:
        """Register an origin object (e.g. `Mobj` or `Sector`) so `SoundEvent`s can track it."""
        oid = origin_id if origin_id is not None else id(origin)
        self._origins_by_id[oid] = origin
        return oid

    def set_sound_priority(self, sound_name: str, priority: int) -> None:
        """Configure the base priority for a logical sound name."""
        self._priorities[sound_name.lower()] = int(priority)

    def get_sound_priority(self, sound_name: str, default: int = DEFAULT_PRIORITY) -> int:
        """Return the base priority for `sound_name`."""
        return self._priorities.get(sound_name.lower(), default)

    def resolve_lump_name(self, sound_id: int | str) -> tuple[str, str]:
        """Resolve a logical sound name or ID to `(sound_name, lump_name)`."""
        sound_name = str(sound_id)
        if self.sndinfo is not None:
            if hasattr(self.sndinfo, "get_lump"):
                lump = self.sndinfo.get_lump(sound_name)
            elif hasattr(self.sndinfo, "resolve_lump"):
                lump = self.sndinfo.resolve_lump(sound_name)
            else:
                lump = None
            if lump is not None:
                return (sound_name, lump)
        try:
            return (sound_name, normalize_lump_name(sound_name))
        except ValueError:
            return (sound_name, "DEFAULT")

    @property
    def active_channels(self) -> list[SoundChannel]:
        """Return all currently active `SoundChannel` instances."""
        return [ch for ch in self.channels if ch is not None and ch.active]

    def set_listener(
        self,
        x: int,
        y: int,
        angle: int = 0,
        origin: Any = None,
    ) -> None:
        """Set the listener's fixed-point position, BAM angle, and optional Mobj reference."""
        self.listener_x = int(x)
        self.listener_y = int(y)
        self.listener_angle = int(angle) & 0xFFFFFFFF
        self.listener_origin = origin
        if origin is not None:
            self.register_origin(origin)

    def handle_sound_event(self, event: SoundEvent) -> SoundChannel | None:
        """Handle a `SoundEvent` emitted on the simulation `EventBus`."""
        origin = self._origins_by_id.get(event.origin_id) if event.origin_id is not None else None
        if event.stop:
            self.stop_sound(origin, origin_id=event.origin_id, sound_id=event.sound_id)
            return None

        return self.start_sound(
            origin=origin,
            sound_id=event.sound_id,
            volume=event.volume,
            origin_id=event.origin_id,
            source_x=event.x,
            source_y=event.y,
        )

    def start_sound(
        self,
        origin: Any,
        sound_id: int | str,
        *,
        volume: int = 127,
        priority: int | None = None,
        loop: bool = False,
        origin_id: int | None = None,
        source_x: int | None = None,
        source_y: int | None = None,
    ) -> SoundChannel | None:
        """Start playing a sound on a channel with distance attenuation and priority eviction.

        Port of S_StartSoundAtVolume (s_sound.c).
        """
        if volume <= 0 or self.master_volume <= 0:
            return None

        if origin is not None and origin_id is None:
            origin_id = self.register_origin(origin)

        sound_name, lump_name = self.resolve_lump_name(sound_id)
        base_pri = priority if priority is not None else self.get_sound_priority(sound_name)

        # Determine source coordinates
        if origin is None and origin_id is None and source_x is None and source_y is None:
            sx, sy = self.listener_x, self.listener_y
            is_listener = True
        elif origin is not None and origin is self.listener_origin:
            sx, sy = self.listener_x, self.listener_y
            is_listener = True
        else:
            fx = source_x if source_x is not None else self.listener_x
            fy = source_y if source_y is not None else self.listener_y
            sx, sy = _extract_source_xy(origin, fx, fy)
            is_listener = False

        if is_listener:
            vol = max(0, min(127, int(volume)))
            sep = NORM_SEP
            dist_units = 0
        else:
            params = adjust_sound_params(
                self.listener_x,
                self.listener_y,
                self.listener_angle,
                sx,
                sy,
                volume,
            )
            if params is None:
                return None
            vol, sep = params
            dist_units = (
                approx_distance(abs(self.listener_x - sx), abs(self.listener_y - sy)) >> FRACBITS
            )

        eff_priority = base_pri * max(1, PRIORITY_MAX_ADJUST - (dist_units // max(1, DIST_ADJUST)))

        # Per-origin replacement: stop any existing channel from the same non-None origin
        if origin is not None or origin_id is not None:
            for idx, existing in enumerate(self.channels):
                if existing is None or not existing.active:
                    continue
                same_origin = (
                    origin is not None
                    and existing.origin is not None
                    and (existing.origin is origin or existing.origin == origin)
                ) or (origin_id is not None and existing.origin_id == origin_id)
                if same_origin:
                    self._stop_channel(idx)
                    break

        # Find a free channel slot
        slot: int | None = None
        for idx, existing in enumerate(self.channels):
            if existing is None or not existing.active:
                slot = idx
                break

        # Priority eviction if all channels are busy
        if slot is None:
            self._sndcount = (self._sndcount + 1) % self.max_channels
            lowest_idx: int | None = None
            lowest_pri = eff_priority
            for offset in range(self.max_channels):
                idx = (self._sndcount + offset) % self.max_channels
                chan = self.channels[idx]
                assert chan is not None
                if chan.priority <= lowest_pri:
                    lowest_pri = chan.priority
                    lowest_idx = idx
            if lowest_idx is None:
                return None
            self._stop_channel(lowest_idx)
            slot = lowest_idx

        scaled_vol = (vol * self.master_volume) // 127
        channel = SoundChannel(
            channel_id=slot,
            sound_name=sound_name,
            lump_name=lump_name,
            origin=origin,
            volume=scaled_vol,
            separation=sep,
            priority=eff_priority,
            loop=loop,
            active=True,
            base_volume=max(0, min(127, int(volume))),
            base_priority=base_pri,
            origin_id=origin_id,
            source_x=sx,
            source_y=sy,
        )
        self.channels[slot] = channel
        self._start_mixer_channel(channel)
        return channel

    def stop_sound(
        self,
        origin: Any = None,
        *,
        origin_id: int | None = None,
        sound_id: int | str | None = None,
    ) -> int:
        """Stop active channel(s) matching `origin`, `origin_id`, and/or `sound_id`.

        Port of S_StopSound (s_sound.c). Returns number of channels stopped.
        """
        stopped = 0
        target_name = str(sound_id) if sound_id is not None else None
        for idx, chan in enumerate(self.channels):
            if chan is None or not chan.active:
                continue
            match_origin = True
            if origin is not None or origin_id is not None:
                match_origin = (
                    origin is not None
                    and chan.origin is not None
                    and (chan.origin is origin or chan.origin == origin)
                ) or (origin_id is not None and chan.origin_id == origin_id)
            match_sound = target_name is None or chan.sound_name == target_name
            if match_origin and match_sound:
                self._stop_channel(idx)
                stopped += 1
        return stopped

    def stop_all(self) -> None:
        """Stop all active sound channels. Port of S_StopAllSound (s_sound.c)."""
        for idx in range(self.max_channels):
            if self.channels[idx] is not None:
                self._stop_channel(idx)

    def update(
        self,
        listener_x: int | None = None,
        listener_y: int | None = None,
        listener_angle: int | None = None,
        listener_origin: Any = None,
    ) -> None:
        """Update all active channels for listener movement and moving sound origins.

        Port of S_UpdateSounds (s_sound.c).
        """
        if listener_x is not None:
            self.listener_x = int(listener_x)
        if listener_y is not None:
            self.listener_y = int(listener_y)
        if listener_angle is not None:
            self.listener_angle = int(listener_angle) & 0xFFFFFFFF
        if listener_origin is not None:
            self.listener_origin = listener_origin
            if hasattr(listener_origin, "x") and hasattr(listener_origin, "y"):
                self.listener_x = int(listener_origin.x)
                self.listener_y = int(listener_origin.y)
            if hasattr(listener_origin, "angle"):
                self.listener_angle = int(listener_origin.angle) & 0xFFFFFFFF

        for idx, chan in enumerate(self.channels):
            if chan is None or not chan.active:
                continue

            if chan.origin is not None and chan.origin is self.listener_origin:
                chan.source_x = self.listener_x
                chan.source_y = self.listener_y
                chan.volume = (chan.base_volume * self.master_volume) // 127
                chan.separation = NORM_SEP
                self._update_mixer_channel(chan)
                continue

            if chan.origin is not None:
                chan.source_x, chan.source_y = _extract_source_xy(
                    chan.origin, chan.source_x, chan.source_y
                )

            params = adjust_sound_params(
                self.listener_x,
                self.listener_y,
                self.listener_angle,
                chan.source_x,
                chan.source_y,
                chan.base_volume,
            )
            if params is None:
                self._stop_channel(idx)
                continue

            vol, sep = params
            chan.volume = (vol * self.master_volume) // 127
            chan.separation = sep
            dist_units = (
                approx_distance(
                    abs(self.listener_x - chan.source_x),
                    abs(self.listener_y - chan.source_y),
                )
                >> FRACBITS
            )
            chan.priority = chan.base_priority * max(
                1, PRIORITY_MAX_ADJUST - (dist_units // max(1, DIST_ADJUST))
            )
            self._update_mixer_channel(chan)

    def _stop_channel(self, slot: int) -> None:
        chan = self.channels[slot]
        if chan is not None:
            chan.active = False
            try:
                import pygame.mixer

                if pygame.mixer.get_init() and slot < pygame.mixer.get_num_channels():
                    pygame.mixer.Channel(slot).stop()
            except Exception:
                pass
        self.channels[slot] = None

    def _start_mixer_channel(self, chan: SoundChannel) -> None:
        if self.sfx_cache is None:
            return
        try:
            import pygame.mixer

            if not pygame.mixer.get_init():
                return
            sound = self.sfx_cache.get_pygame_sound(chan.lump_name)
            if sound is None or chan.channel_id >= pygame.mixer.get_num_channels():
                return
            mixer_chan = pygame.mixer.Channel(chan.channel_id)
            left, right = separation_to_stereo_gains(chan.volume, chan.separation)
            mixer_chan.set_volume(left, right)
            mixer_chan.play(sound, loops=-1 if chan.loop else 0)
        except Exception:
            pass

    def _update_mixer_channel(self, chan: SoundChannel) -> None:
        try:
            import pygame.mixer

            if not pygame.mixer.get_init() or chan.channel_id >= pygame.mixer.get_num_channels():
                return
            mixer_chan = pygame.mixer.Channel(chan.channel_id)
            left, right = separation_to_stereo_gains(chan.volume, chan.separation)
            mixer_chan.set_volume(left, right)
        except Exception:
            pass
