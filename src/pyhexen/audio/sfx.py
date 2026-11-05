# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""DMX digital sound effect decoder, resampler, and cache.

Port of CacheSFX and ExpandSoundData_SDL (src/i_sdlsound.c).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyhexen.wad.sndinfo import SndInfo, load_sndinfo
from pyhexen.wad.wadfile import WadFile, WadStack, normalize_lump_name

if TYPE_CHECKING:
    import pygame

_DMX_HEADER = struct.Struct("<HHI")
DMX_FORMAT_PCM = 3


class SfxDecodeError(ValueError):
    """Raised when a DMX sound lump has an invalid header or truncated payload."""


@dataclass(slots=True)
class DecodedSfx:
    """Decoded DMX sound effect containing raw 8-bit unsigned and resampled 16-bit signed PCM."""

    sample_rate: int
    samples_u8: np.ndarray
    samples_i16: np.ndarray


def decode_dmx_sfx(
    data: bytes | bytearray | memoryview,
    target_rate: int = 44100,
    *,
    strip_padding: bool | None = None,
) -> DecodedSfx:
    """Decode a DMX format 3 sound lump and resample to `target_rate` Hz.

    Port of CacheSFX / ExpandSoundData_SDL (src/i_sdlsound.c).

    Parameters
    ----------
    data:
        Raw lump bytes starting with `<HHI` (`format=3`, `sample_rate`, `num_samples`).
    target_rate:
        Output sample rate in Hz for `samples_i16` (default 44100).
    strip_padding:
        When True, strips the 16-byte DMX lead-in and 16-byte lead-out padding when
        `num_samples > 32`. When None (default), strips padding only if `num_samples > 32`
        and the first 16 bytes and last 16 bytes of the sample payload are constant padding
        or `len(data) == 8 + num_samples` with `num_samples >= 49`.
    """
    if target_rate <= 0:
        raise ValueError(f"target_rate must be positive, got {target_rate}")

    buf = memoryview(data)
    if len(buf) < _DMX_HEADER.size:
        raise SfxDecodeError(f"DMX lump too short for header ({len(buf)} bytes < 8)")

    fmt, sample_rate, num_samples = _DMX_HEADER.unpack_from(buf, 0)
    if fmt != DMX_FORMAT_PCM:
        raise SfxDecodeError(f"Unsupported DMX sound format {fmt} (expected 3)")
    if sample_rate <= 0:
        raise SfxDecodeError(f"Invalid DMX sample rate {sample_rate}")
    if num_samples > len(buf) - _DMX_HEADER.size:
        raise SfxDecodeError(
            f"DMX header sample count ({num_samples}) exceeds payload size ({len(buf) - 8})"
        )

    payload = buf[_DMX_HEADER.size : _DMX_HEADER.size + num_samples]

    if strip_padding is True and num_samples > 32:
        payload = payload[16 : num_samples - 16]
    elif strip_padding is None and num_samples > 48:
        # Check if lead-in/lead-out look like standard 16-byte DMX constant pad bytes
        # (in DMX WAV->DMX tool, bytes 0..15 repeat the first sample and last 16 repeat the last)
        lead_in = payload[:16]
        lead_out = payload[num_samples - 16 :]
        if len(set(lead_in)) == 1 and len(set(lead_out)) == 1:
            payload = payload[16 : num_samples - 16]

    samples_u8 = np.frombuffer(payload, dtype=np.uint8).copy()

    if samples_u8.size == 0:
        return DecodedSfx(
            sample_rate=sample_rate,
            samples_u8=samples_u8,
            samples_i16=np.empty(0, dtype=np.int16),
        )

    # Convert unsigned 8-bit PCM (0..255, center 128) to signed 16-bit PCM (-32768..32512)
    s16_float = (samples_u8.astype(np.float64) - 128.0) * 256.0

    if sample_rate == target_rate or samples_u8.size == 1:
        samples_i16 = np.clip(np.round(s16_float), -32768, 32767).astype(np.int16)
    else:
        out_len = max(1, int(round(samples_u8.size * target_rate / sample_rate)))
        x_old = np.arange(samples_u8.size, dtype=np.float64)
        x_new = np.linspace(0.0, float(samples_u8.size - 1), out_len, dtype=np.float64)
        resampled = np.interp(x_new, x_old, s16_float)
        samples_i16 = np.clip(np.round(resampled), -32768, 32767).astype(np.int16)

    return DecodedSfx(
        sample_rate=sample_rate,
        samples_u8=samples_u8,
        samples_i16=samples_i16,
    )


class SfxCache:
    """Caches decoded DMX sound effects and optional `pygame.mixer.Sound` objects."""

    def __init__(
        self,
        wad: WadStack | WadFile,
        sndinfo: SndInfo | None = None,
        *,
        target_rate: int = 44100,
    ) -> None:
        self._wad = wad
        if sndinfo is None and "SNDINFO" in wad:
            self._sndinfo: SndInfo | None = load_sndinfo(wad)
        else:
            self._sndinfo = sndinfo
        self.target_rate = target_rate
        self._decoded_cache: dict[str, DecodedSfx] = {}
        self._sound_cache: dict[str, pygame.mixer.Sound] = {}

    def resolve_lump_name(self, name: str) -> str:
        """Resolve a logical sound name (via SNDINFO) or WAD lump name to an uppercase lump name."""
        if self._sndinfo is not None:
            mapped = self._sndinfo.get_lump(name)
            if mapped is not None:
                return mapped
        return normalize_lump_name(name)

    def get_decoded(self, name: str) -> DecodedSfx:
        """Decode and cache a sound effect by logical name or WAD lump name."""
        lump_name = self.resolve_lump_name(name)
        cached = self._decoded_cache.get(lump_name)
        if cached is not None:
            return cached

        raw = self._wad.get_bytes(lump_name)
        decoded = decode_dmx_sfx(raw, target_rate=self.target_rate)
        self._decoded_cache[lump_name] = decoded
        return decoded

    def get_sound(self, name: str) -> pygame.mixer.Sound | None:
        """Return a cached `pygame.mixer.Sound` if `pygame.mixer` is initialized, else None."""
        import pygame.mixer

        mixer_init = pygame.mixer.get_init()
        if not mixer_init:
            return None

        lump_name = self.resolve_lump_name(name)
        cached_snd = self._sound_cache.get(lump_name)
        if cached_snd is not None:
            return cached_snd

        mixer_freq, _mixer_fmt, mixer_channels = mixer_init
        if mixer_freq != self.target_rate:
            raw = self._wad.get_bytes(lump_name)
            decoded = decode_dmx_sfx(raw, target_rate=mixer_freq)
        else:
            decoded = self.get_decoded(lump_name)

        pcm = decoded.samples_i16
        if mixer_channels == 2:
            buf = np.column_stack((pcm, pcm)).ravel().tobytes()
        elif mixer_channels > 2:
            buf = np.tile(pcm[:, np.newaxis], (1, mixer_channels)).ravel().tobytes()
        else:
            buf = pcm.tobytes()

        sound = pygame.mixer.Sound(buffer=buf)
        self._sound_cache[lump_name] = sound
        return sound

    def clear(self) -> None:
        """Clear all cached decoded sounds and mixer Sound objects."""
        self._decoded_cache.clear()
        self._sound_cache.clear()
