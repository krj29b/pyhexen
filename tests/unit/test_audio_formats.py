# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for DMX sound effect decoding/resampling (T11.1) and MUS->MIDI conversion (T11.4)."""

from __future__ import annotations

import io
import struct

import mido
import numpy as np
import pytest
from tests.fixtures.wadbuilder import WadBuilder

from pyhexen.audio.mus2mid import MusError, mus_to_midi
from pyhexen.audio.sfx import SfxCache, SfxDecodeError, decode_dmx_sfx
from pyhexen.wad.wadfile import WadStack


def _build_dmx_lump(
    samples: bytes | np.ndarray,
    sample_rate: int = 11025,
    *,
    pad_16: bool = True,
) -> bytes:
    raw = (
        bytes(samples)
        if not isinstance(samples, np.ndarray)
        else samples.astype(np.uint8).tobytes()
    )
    if pad_16 and len(raw) > 0:
        lead_in = bytes([raw[0]]) * 16
        lead_out = bytes([raw[-1]]) * 16
        payload = lead_in + raw + lead_out
    else:
        payload = raw
    header = struct.pack("<HHI", 3, sample_rate, len(payload))
    return header + payload


def _build_synthetic_mus() -> bytes:
    """Build a synthetic MUS buffer exercising controllers, patch change, notes, and delays."""
    events = bytearray()
    # 1. Channel 0: change patch (controller 0) to instrument 24
    events.extend([0x40 | 0, 0, 24])
    # 2. Channel 0: change volume (controller 3 -> MIDI CC 7) to 100
    events.extend([0x40 | 0, 3, 100])
    # 3. Channel 0: press key 60 with new velocity 110, followed by delay 35
    events.extend([(0x10 | 0) | 0x80, 60 | 0x80, 110, 35])
    # 4. Channel 15 (percussion -> MIDI chan 9): press key 36 (kick), no delay
    events.extend([0x10 | 15, 36])
    # 5. Channel 0: pitch wheel bend (128 = center), followed by multi-byte delay (200 = 1*128 + 72)
    events.extend([(0x20 | 0) | 0x80, 140, 0x81, 72])
    # 6. Channel 0: release key 60, Channel 15: release key 36, followed by delay 10
    events.extend([0x00 | 0, 60, (0x00 | 15) | 0x80, 36, 10])
    # 7. Channel 0: system event 11 (all notes off)
    events.extend([0x30 | 0, 11])
    # 8. Score end
    events.append(0x60)

    instruments = struct.pack("<H", 24)
    score_start = 14 + len(instruments)
    header = struct.pack(
        "<4s5H",
        b"MUS\x1a",
        len(events),
        score_start,
        1,
        0,
        1,
    )
    return header + instruments + bytes(events)


class TestSfxDecode:
    def test_decode_and_resample_dmx(self) -> None:
        # 64 samples of a ramp from 0 to 255 at 11025 Hz with 16-byte DMX padding
        ramp = np.linspace(0, 255, 64, dtype=np.uint8)
        lump = _build_dmx_lump(ramp, sample_rate=11025, pad_16=True)

        decoded = decode_dmx_sfx(lump, target_rate=44100)
        assert decoded.sample_rate == 11025
        assert decoded.samples_u8.dtype == np.uint8
        assert decoded.samples_u8.shape == (64,)
        np.testing.assert_array_equal(decoded.samples_u8, ramp)

        # Resampled 4x from 11025 to 44100 -> 256 samples of signed int16
        assert decoded.samples_i16.dtype == np.int16
        assert decoded.samples_i16.shape == (256,)
        assert decoded.samples_i16[0] == -32768
        assert decoded.samples_i16[-1] == (255 - 128) * 256

    def test_decode_same_rate_and_short_unpadded(self) -> None:
        samples = np.array([128, 0, 255, 128], dtype=np.uint8)
        lump = _build_dmx_lump(samples, sample_rate=22050, pad_16=False)

        decoded = decode_dmx_sfx(lump, target_rate=22050)
        assert decoded.sample_rate == 22050
        np.testing.assert_array_equal(decoded.samples_u8, samples)
        expected_i16 = np.array([0, -32768, 32512, 0], dtype=np.int16)
        np.testing.assert_array_equal(decoded.samples_i16, expected_i16)

    def test_invalid_dmx_rejected(self) -> None:
        with pytest.raises(SfxDecodeError, match="too short"):
            decode_dmx_sfx(b"\x03\x00\x11")

        bad_fmt = struct.pack("<HHI", 1, 11025, 4) + b"\x80\x80\x80\x80"
        with pytest.raises(SfxDecodeError, match="format"):
            decode_dmx_sfx(bad_fmt)

        truncated = struct.pack("<HHI", 3, 11025, 100) + b"\x80" * 10
        with pytest.raises(SfxDecodeError, match="exceeds payload"):
            decode_dmx_sfx(truncated)

    def test_sfx_cache(self) -> None:
        wb = WadBuilder()
        wb.add_lump("SNDINFO", b"FighterGrunt fgtgrunt\n")
        wb.add_lump("FGTGRUNT", _build_dmx_lump(np.full(64, 160, dtype=np.uint8), 11025))
        stack = WadStack(wb.to_bytes())

        cache = SfxCache(stack, target_rate=22050)
        assert cache.resolve_lump_name("FighterGrunt") == "FGTGRUNT"
        d1 = cache.get_decoded("FighterGrunt")
        d2 = cache.get_decoded("FGTGRUNT")
        assert d1 is d2
        assert d1.samples_i16.shape == (128,)


class TestMusToMidi:
    def test_synthetic_mus_converts_to_valid_midi(self) -> None:
        mus_bytes = _build_synthetic_mus()
        midi_bytes = mus_to_midi(mus_bytes)

        assert midi_bytes.startswith(b"MThd")
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        assert mid.type == 0
        assert mid.ticks_per_beat == 70
        assert len(mid.tracks) == 1

        track = mid.tracks[0]
        msg_types = [m.type for m in track]
        assert "program_change" in msg_types
        assert "control_change" in msg_types
        assert "note_on" in msg_types
        assert "note_off" in msg_types
        assert "pitchwheel" in msg_types

        note_ons = [m for m in track if m.type == "note_on"]
        assert any(m.channel == 0 and m.note == 60 and m.velocity == 110 for m in note_ons)
        # Percussion channel 15 in MUS maps to MIDI channel 9
        assert any(m.channel == 9 and m.note == 36 and m.velocity == 127 for m in note_ons)

        total_ticks = sum(m.time for m in track)
        assert total_ticks == 35 + 200 + 10

    def test_invalid_mus_rejected(self) -> None:
        with pytest.raises(MusError, match="too short"):
            mus_to_midi(b"MUS\x1a")

        with pytest.raises(MusError, match="Invalid MUS magic"):
            mus_to_midi(b"BAD!" + b"\x00" * 16)
