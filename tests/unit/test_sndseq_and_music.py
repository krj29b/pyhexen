# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T11.3 (SoundSequencePlayer) and T11.5 (MusicManager / backends)."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import mido
import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.audio.music import MusicManager, NullMusicBackend
from pyhexen.audio.sndseq_player import SeqId, SeqType, SoundSequencePlayer
from pyhexen.core.events import EventBus, SoundEvent
from pyhexen.core.random import Random
from pyhexen.wad.sndseq import parse_sndseq
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def _build_synthetic_mus(note: int = 60, delay: int = 28) -> bytes:
    """Build a minimal valid MUS\\x1a lump."""
    events = bytearray()
    # Channel 0: play note with volume 100, followed by 1-byte delay
    events.extend([0x80 | 0x10, note | 0x80, 100, delay & 0x7F])
    # Channel 0: release note
    events.extend([0x00, note & 0x7F])
    # Score end (type 6)
    events.append(0x60)

    score_start = 16
    score_len = len(events)
    header = struct.pack(
        "<4sHHHHHH",
        b"MUS\x1a",
        score_len,
        score_start,
        1,
        0,
        1,
        0,
    ) + struct.pack("<H", 0)
    return header + bytes(events)


def test_sound_sequence_player_delay_volume_stopsound_end() -> None:
    sndseq_script = """
    :DoorNormal
        volume 50
        play DoorOpen
        delay 2
        play DoorSqueak
        stopsound DoorClunk
    end
    """
    sequences = parse_sndseq(sndseq_script)
    player = SoundSequencePlayer(sequences)
    rng = Random()
    bus = EventBus()
    emitted: list[SoundEvent] = []
    bus.subscribe(SoundEvent, emitted.append)

    origin = type("DummyOrigin", (), {"x": 100, "y": 200, "z": 0})()
    node = player.start_sequence(origin, "DoorNormal")
    assert node is not None
    assert len(player.active_sequences) == 1

    # Tic 1: executes SS_CMD_VOLUME(50) -> volume = (127 * 50) // 100 = 63
    assert player.tick(rng, bus) == []
    assert node.volume == 63

    # Tic 2: executes SS_CMD_PLAY("DoorOpen")
    events_t2 = player.tick(rng, bus)
    assert len(events_t2) == 1
    assert events_t2[0].sound_id == "DoorOpen"
    assert events_t2[0].volume == 63
    assert events_t2[0].x == 100
    assert events_t2[0].y == 200

    # Tic 3: executes SS_CMD_DELAY(2), setting delay_tics = 2 and current_sound = None
    assert player.tick(rng, bus) == []
    assert node.delay_tics == 2

    # Tics 4 and 5: count down delay_tics (2 -> 1 -> 0)
    assert player.tick(rng, bus) == []
    assert node.delay_tics == 1
    assert player.tick(rng, bus) == []
    assert node.delay_tics == 0

    # Tic 6: executes SS_CMD_PLAY("DoorSqueak")
    events_t6 = player.tick(rng, bus)
    assert len(events_t6) == 1
    assert events_t6[0].sound_id == "DoorSqueak"

    # Tic 7: hits SS_CMD_STOPSOUND right before SS_CMD_END -> waits until stop_sequence
    assert player.tick(rng, bus) == []
    assert len(player.active_sequences) == 1

    # Stopping sequence emits stop for DoorSqueak + plays stopsound DoorClunk
    stop_events = player.stop_sequence(origin, event_bus=bus)
    assert len(player.active_sequences) == 0
    assert [e.sound_id for e in stop_events] == ["DoorSqueak", "DoorClunk"]
    assert stop_events[0].stop is True
    assert stop_events[1].stop is False
    assert len(emitted) == 4


def test_sound_sequence_player_delayrand_playrepeat_and_wait() -> None:
    sndseq_script = """
    :PlatSeq
        playuntildone PlatStart
        delayrand 2 6
        playrepeat PlatMove
        stopsound PlatStop
    end
    """
    sequences = parse_sndseq(sndseq_script)
    player = SoundSequencePlayer(sequences)
    rng = Random()

    origin = "plat_1"
    node = player.start_sequence(origin, "PlatSeq")
    assert node is not None

    # Tic 1: playuntildone step 1 -> SS_CMD_PLAY("PlatStart")
    ev1 = player.tick(rng)
    assert len(ev1) == 1
    assert ev1[0].sound_id == "PlatStart"
    assert player.is_sound_playing(node)

    # Tic 2: at SS_CMD_WAIT while PlatStart is still marked playing -> stays at SS_CMD_WAIT
    assert player.tick(rng) == []
    assert node.ip == 1

    # Mark PlatStart finished -> next tic advances past SS_CMD_WAIT
    player.mark_sound_finished(origin, "PlatStart")
    assert not player.is_sound_playing(node)
    assert player.tick(rng) == []
    assert node.ip == 2

    # Next tic executes SS_CMD_DELAYRAND(2, 6)
    expected_rng = Random()
    expected_delay = 2 + (expected_rng.m_random() % (6 - 2))
    assert player.tick(rng) == []
    assert node.delay_tics == expected_delay

    # Count down delay_tics
    while node.delay_tics > 0:
        assert player.tick(rng) == []

    # Next tic executes SS_CMD_PLAYREPEAT("PlatMove")
    ev_repeat = player.tick(rng)
    assert len(ev_repeat) == 1
    assert ev_repeat[0].sound_id == "PlatMove"

    # While PlatMove is still playing, SS_CMD_PLAYREPEAT does not restart it
    assert player.tick(rng) == []

    # When PlatMove finishes, SS_CMD_PLAYREPEAT restarts it on the next tic
    player.mark_sound_finished(origin, "PlatMove")
    ev_repeat_again = player.tick(rng)
    assert len(ev_repeat_again) == 1
    assert ev_repeat_again[0].sound_id == "PlatMove"

    # Explicit stop_sequence emits stop for PlatMove and plays stopsound PlatStop
    stop_events = player.stop_sequence(origin)
    assert [e.sound_id for e in stop_events] == ["PlatMove", "PlatStop"]
    assert stop_events[0].stop is True
    assert stop_events[1].stop is False
    assert len(player.active_sequences) == 0


def test_sound_sequence_player_level_things(tmp_path: Path) -> None:
    sndseq_script = """
    :Water
        play WaterMove
    end
    :Wind
        playloop WindSound 20
    end
    """
    wb = WadBuilder("IWAD")
    wb.add_lump("SNDSEQ", sndseq_script.encode("ascii"))
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    # Thing 1403 overrides containing sector seq_type to 3 (SEQTYPE_CREAK)
    mb.add_thing(x=64, y=64, type=1403)
    # Thing 1410 starts ambient Wind sound sequence
    mb.add_thing(x=128, y=128, type=1410)
    wb.add_map(mb)

    wad_path = wb.write(tmp_path / "seq_level.wad")
    with WadStack(wad_path) as wad:
        level = load_level(wad, "MAP01")
        player = SoundSequencePlayer.from_wad(wad)

    ambient_things = player.apply_level_things(level)
    assert level.sectors[0].seq_type == SeqType.SEQTYPE_CREAK
    assert len(ambient_things) == 1
    assert ambient_things[0].type == 1410
    assert len(player.active_sequences) == 1
    assert player.active_sequences[0].sequence.name == "Wind"

    # Verify SeqId translation lookup
    resolved_creak = player.resolve_sequence(SeqId.SEQ_PLATFORM_CREAK)
    assert resolved_creak is None  # Creak not in this minimal script
    resolved_water = player.resolve_sequence(SeqId.SEQ_PLATFORM_WATER)
    assert resolved_water is not None
    assert resolved_water.name == "Water"


def test_music_manager_and_null_backend(tmp_path: Path) -> None:
    wb = WadBuilder("IWAD")
    wb.add_lump(
        "MAPINFO",
        b"""
        map 1 "HALL"
        cdtrack 1
        map 2 "KEEP"
        cdtrack 2
        """,
    )
    wb.add_lump(
        "SNDINFO",
        b"""
        $map 1 Winnowr
        $map 2 Jachr
        """,
    )
    wb.add_lump("WINNOWR", _build_synthetic_mus(note=60, delay=28))
    wb.add_lump("JACHR", _build_synthetic_mus(note=67, delay=56))

    wad_path = wb.write(tmp_path / "music_test.wad")
    with WadStack(wad_path) as wad:
        backend = NullMusicBackend()
        mgr = MusicManager(wad, backend=backend)

        assert mgr.resolve_map_music(1) == "WINNOWR"
        assert mgr.resolve_map_music(2) == "JACHR"
        assert mgr.resolve_map_music(99) is None

        # Play map 1 music
        assert mgr.play_map_music(1) is True
        assert mgr.current_lump == "WINNOWR"
        assert backend.is_playing is True
        assert backend.play_count == 1
        assert backend.current_midi is not None

        # Validate cached MIDI bytes with mido
        mid = mido.MidiFile(file=io.BytesIO(backend.current_midi))
        assert mid.type == 0

        # Calling play_map_music(1) again while already playing does not reload/restart
        assert mgr.play_map_music(1) is True
        assert backend.play_count == 1

        # Switching to map 2 loads and plays JACHR
        assert mgr.play_map_music(2) is True
        assert mgr.current_lump == "JACHR"
        assert backend.play_count == 2

        # Volume, pause, resume, stop
        mgr.set_volume(0.5)
        assert mgr.volume == pytest.approx(0.5)
        assert backend.volume == pytest.approx(0.5)

        mgr.pause()
        assert backend.is_paused is True
        mgr.resume()
        assert backend.is_paused is False

        mgr.stop()
        assert backend.is_playing is False
        assert mgr.current_lump is None
