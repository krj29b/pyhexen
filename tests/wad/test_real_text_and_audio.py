# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Real IWAD tests for MAPINFO, SNDINFO, SNDSEQ, ANIMDEFS, DMX SFX, and MUS lumps."""

from __future__ import annotations

import io
from pathlib import Path

import mido
import pytest

from pyhexen.audio.mus2mid import mus_to_midi
from pyhexen.audio.sfx import decode_dmx_sfx
from pyhexen.wad.animdefs import load_animdefs
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.sndinfo import load_sndinfo
from pyhexen.wad.sndseq import load_sndseq
from pyhexen.wad.wadfile import WadStack

pytestmark = pytest.mark.wad


def test_real_mapinfo(iwad_path: Path) -> None:
    with WadStack(iwad_path) as wad:
        maps = load_mapinfo(wad)
        assert len(maps) >= 31
        assert 1 in maps
        assert "MAP01" in maps
        m1 = maps[1]
        assert m1.name.upper() == "WINNOWING HALL"
        assert m1.cluster == 1
        assert m1.music_lump != ""


def test_real_sndinfo_sndseq_animdefs(iwad_path: Path) -> None:
    with WadStack(iwad_path) as wad:
        sndinfo = load_sndinfo(wad)
        assert len(sndinfo.sounds) > 100
        assert len(sndinfo.map_music) >= 31

        sndseq = load_sndseq(wad)
        assert len(sndseq) > 5
        assert "Platform" in sndseq or "DoorNormal" in sndseq

        animdefs = load_animdefs(wad)
        assert len(animdefs) > 5
        assert any(a.is_flat for a in animdefs)
        assert any(not a.is_flat for a in animdefs)


def test_real_sfx_and_mus_lumps(iwad_path: Path) -> None:
    with WadStack(iwad_path) as wad:
        sndinfo = load_sndinfo(wad)
        decoded_sfx = 0
        for lump_name in set(sndinfo.sounds.values()):
            if lump_name == "DEFAULT" or lump_name not in wad:
                continue
            raw = wad.get_bytes(lump_name)
            if len(raw) >= 8 and raw[0] == 3 and raw[1] == 0:
                sfx = decode_dmx_sfx(raw, target_rate=44100)
                assert sfx.sample_rate > 0
                assert sfx.samples_i16.size > 0
                decoded_sfx += 1
        assert decoded_sfx > 100

        converted_mus = 0
        for lump in wad.lumps:
            if lump.size >= 14:
                raw = wad.get_bytes(lump)
                if raw[:4] == b"MUS\x1a":
                    midi_bytes = mus_to_midi(raw)
                    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
                    assert len(mid.tracks) == 1
                    converted_mus += 1
        assert converted_mus >= 30
