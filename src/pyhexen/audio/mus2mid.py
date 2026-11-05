# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""MUS to Standard MIDI (Format 0) converter.

Port of src/mus2mid.c by Ben Ryves / Simon Howard from Chocolate Doom.
"""

from __future__ import annotations

import struct

NUM_CHANNELS = 16
MIDI_PERCUSSION_CHAN = 9
MUS_PERCUSSION_CHAN = 15

# MUS event codes
MUS_RELEASEKEY = 0x00
MUS_PRESSKEY = 0x10
MUS_PITCHWHEEL = 0x20
MUS_SYSTEMEVENT = 0x30
MUS_CHANGECONTROLLER = 0x40
MUS_SCOREEND = 0x60

# MIDI event codes
MIDI_RELEASEKEY = 0x80
MIDI_PRESSKEY = 0x90
MIDI_AFTERTOUCHKEY = 0xA0
MIDI_CHANGECONTROLLER = 0xB0
MIDI_CHANGEPATCH = 0xC0
MIDI_AFTERTOUCHCHANNEL = 0xD0
MIDI_PITCHWHEEL = 0xE0

_MUS_HEADER = struct.Struct("<4s5H")

# Standard MIDI type 0 header + MTrk prefix (18 bytes before 4-byte track length)
_MIDI_HEADER_PREFIX = bytes(
    [
        ord("M"),
        ord("T"),
        ord("h"),
        ord("d"),
        0x00,
        0x00,
        0x00,
        0x06,  # Header size (6)
        0x00,
        0x00,  # MIDI format 0
        0x00,
        0x01,  # 1 track
        0x00,
        0x46,  # 70 ticks per quarter note (140 Hz at 120 BPM)
        ord("M"),
        ord("T"),
        ord("r"),
        ord("k"),
    ]
)

_CONTROLLER_MAP: tuple[int, ...] = (
    0x00,
    0x20,
    0x01,
    0x07,
    0x0A,
    0x0B,
    0x5B,
    0x5D,
    0x40,
    0x43,
    0x78,
    0x7B,
    0x7E,
    0x7F,
    0x79,
)


class MusError(ValueError):
    """Raised when a MUS byte buffer is malformed or cannot be converted to MIDI."""


def mus_to_midi(mus_data: bytes | bytearray | memoryview) -> bytes:
    """Convert a Doom/Hexen `MUS\\x1a` byte buffer into a Format 0 MIDI byte buffer.

    Port of mus2mid (src/mus2mid.c).
    """
    buf = memoryview(mus_data)
    if len(buf) < _MUS_HEADER.size:
        raise MusError(f"MUS data too short for header ({len(buf)} bytes < 14)")

    (
        ident,
        _score_length,
        score_start,
        _primary_channels,
        _secondary_channels,
        _instrument_count,
    ) = _MUS_HEADER.unpack_from(buf, 0)

    if ident != b"MUS\x1a":
        raise MusError(f"Invalid MUS magic {ident!r} (expected b'MUS\\x1a')")

    if score_start > len(buf):
        raise MusError(f"MUS scorestart ({score_start}) exceeds buffer length ({len(buf)})")

    channel_velocities = [127] * NUM_CHANNELS
    channel_map = [-1] * NUM_CHANNELS
    queued_time = 0
    track = bytearray()
    pos = score_start
    total_len = len(buf)

    def read_u8() -> int:
        nonlocal pos
        if pos >= total_len:
            raise MusError("Unexpected EOF while reading MUS score data")
        b = buf[pos]
        pos += 1
        return b

    def write_time(time_val: int) -> None:
        nonlocal queued_time
        buffer = time_val & 0x7F
        while (time_val >> 7) != 0:
            time_val >>= 7
            buffer = (buffer << 8) | ((time_val & 0x7F) | 0x80)

        while True:
            track.append(buffer & 0xFF)
            if (buffer & 0x80) != 0:
                buffer >>= 8
            else:
                queued_time = 0
                return

    def write_end_track() -> None:
        write_time(queued_time)
        track.extend(b"\xff\x2f\x00")

    def write_press_key(channel: int, key: int, velocity: int) -> None:
        write_time(queued_time)
        track.append(MIDI_PRESSKEY | channel)
        track.append(key & 0x7F)
        track.append(velocity & 0x7F)

    def write_release_key(channel: int, key: int) -> None:
        write_time(queued_time)
        track.append(MIDI_RELEASEKEY | channel)
        track.append(key & 0x7F)
        track.append(0)

    def write_pitch_wheel(channel: int, wheel: int) -> None:
        write_time(queued_time)
        track.append(MIDI_PITCHWHEEL | channel)
        track.append(wheel & 0x7F)
        track.append((wheel >> 7) & 0x7F)

    def write_change_patch(channel: int, patch: int) -> None:
        write_time(queued_time)
        track.append(MIDI_CHANGEPATCH | channel)
        track.append(patch & 0x7F)

    def write_change_controller_valued(channel: int, control: int, value: int) -> None:
        write_time(queued_time)
        track.append(MIDI_CHANGECONTROLLER | channel)
        track.append(control & 0x7F)
        working = 0x7F if (value & 0x80) else value
        track.append(working)

    def write_change_controller_valueless(channel: int, control: int) -> None:
        write_change_controller_valued(channel, control, 0)

    def allocate_midi_channel() -> int:
        max_chan = max(channel_map)
        result = max_chan + 1
        if result == MIDI_PERCUSSION_CHAN:
            result += 1
        return result

    def get_midi_channel(mus_channel: int) -> int:
        if mus_channel == MUS_PERCUSSION_CHAN:
            return MIDI_PERCUSSION_CHAN
        if channel_map[mus_channel] == -1:
            channel_map[mus_channel] = allocate_midi_channel()
            write_change_controller_valueless(channel_map[mus_channel], 0x7B)
        return channel_map[mus_channel]

    hit_score_end = False

    while not hit_score_end:
        while not hit_score_end:
            event_descriptor = read_u8()
            channel = get_midi_channel(event_descriptor & 0x0F)
            event = event_descriptor & 0x70

            if event == MUS_RELEASEKEY:
                key = read_u8()
                write_release_key(channel, key)
            elif event == MUS_PRESSKEY:
                key = read_u8()
                if key & 0x80:
                    channel_velocities[channel] = read_u8() & 0x7F
                write_press_key(channel, key, channel_velocities[channel])
            elif event == MUS_PITCHWHEEL:
                key = read_u8()
                write_pitch_wheel(channel, key * 64)
            elif event == MUS_SYSTEMEVENT:
                controller_number = read_u8()
                if controller_number < 10 or controller_number > 14:
                    raise MusError(
                        f"Invalid MUS system event controller number {controller_number}"
                    )
                write_change_controller_valueless(channel, _CONTROLLER_MAP[controller_number])
            elif event == MUS_CHANGECONTROLLER:
                controller_number = read_u8()
                controller_value = read_u8()
                if controller_number == 0:
                    write_change_patch(channel, controller_value)
                else:
                    if controller_number < 1 or controller_number > 9:
                        raise MusError(f"Invalid MUS controller number {controller_number}")
                    write_change_controller_valued(
                        channel,
                        _CONTROLLER_MAP[controller_number],
                        controller_value,
                    )
            elif event == MUS_SCOREEND:
                hit_score_end = True
            else:
                raise MusError(f"Unknown MUS event code {event:#x}")

            if event_descriptor & 0x80:
                break

        if not hit_score_end:
            time_delay = 0
            while True:
                working = read_u8()
                time_delay = time_delay * 128 + (working & 0x7F)
                if (working & 0x80) == 0:
                    break
            queued_time += time_delay

    write_end_track()

    return _MIDI_HEADER_PREFIX + struct.pack(">I", len(track)) + bytes(track)
