# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for synthetic WAD/map builders, WadFile, WadStack, and namespaces."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from fixtures.wadbuilder import (
    ML_TWOSIDED,
    NF_SUBSECTOR,
    MapBuilder,
    PatchPlacement,
    TextureDef,
    WadBuilder,
    encode_picture,
)
from pyhexen.wad.namespaces import flat_lumps, patch_lumps, sprite_lumps
from pyhexen.wad.wadfile import (
    LumpNotFoundError,
    WadError,
    WadFile,
    WadStack,
    normalize_lump_name,
)


def test_normalize_lump_name() -> None:
    assert normalize_lump_name("playpal") == "PLAYPAL"
    assert normalize_lump_name(b"map01\x00\x00\x00") == "MAP01"
    assert normalize_lump_name(b"f1_start\x00garbage") == "F1_START"
    assert normalize_lump_name("longerthan8chars") == "LONGERTH"


def test_wadbuilder_and_wadfile_roundtrip(tmp_path: Path) -> None:
    wb = WadBuilder("IWAD")
    wb.add_playpal(num_palettes=14)
    wb.add_colormap(num_rows=33, include_fogmap=True)
    wb.add_flat("FLOOR01", fill=42)

    pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    wb.add_picture("PATCH01", pixels, mask=mask, left=2, top=3)

    wb.add_textures(
        textures=[
            TextureDef(
                name="WALL01",
                width=64,
                height=128,
                patches=[PatchPlacement(originx=0, originy=0, patch=0)],
            )
        ],
        pnames=["PATCH01"],
    )

    wad_path = wb.write(tmp_path / "synth.wad")
    with WadFile(wad_path) as wf:
        assert wf.wad_type == "IWAD"
        assert wf.source == "synth.wad"
        assert len(wf) == 7
        assert "PLAYPAL" in wf
        assert "COLORMAP" in wf
        assert "FOGMAP" in wf
        assert "FLOOR01" in wf
        assert "PATCH01" in wf
        assert "PNAMES" in wf
        assert "TEXTURE1" in wf

        playpal = wf.get_bytes("PLAYPAL")
        assert len(playpal) == 14 * 768

        colormap = wf.get_bytes("COLORMAP")
        assert len(colormap) == 33 * 256

        flat = wf.get_bytes("FLOOR01")
        assert len(flat) == 4096
        assert set(flat) == {42}

        pic = wf.get_bytes("PATCH01")
        w, h, left, top = struct.unpack_from("<4h", pic, 0)
        assert (w, h, left, top) == (4, 4, 2, 3)

        pnames = wf.get_bytes("PNAMES")
        (num_patches,) = struct.unpack_from("<i", pnames, 0)
        assert num_patches == 1
        assert normalize_lump_name(pnames[4:12]) == "PATCH01"

        tex1 = wf.get_bytes("TEXTURE1")
        num_tex, first_ofs = struct.unpack_from("<ii", tex1, 0)
        assert num_tex == 1
        assert first_ofs == 8
        raw_name, masked, tw, th, obsolete, patchcount = struct.unpack_from(
            "<8si2hih", tex1, first_ofs
        )
        assert normalize_lump_name(raw_name) == "WALL01"
        assert (masked, tw, th, obsolete, patchcount) == (0, 64, 128, 0, 1)


def test_encode_picture_columns_and_posts() -> None:
    pixels = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.uint8)
    mask = np.array([[True, False], [False, True], [True, True]], dtype=bool)
    blob = encode_picture(pixels, mask=mask, left=1, top=2)

    w, h, left, top = struct.unpack_from("<4h", blob, 0)
    assert (w, h, left, top) == (2, 3, 1, 2)
    col0_ofs, col1_ofs = struct.unpack_from("<2I", blob, 8)

    # Column 0 has two separate 1-pixel posts at y=0 and y=2
    assert blob[col0_ofs : col0_ofs + 5] == bytes([0, 1, 0, 10, 0])
    assert blob[col0_ofs + 5 : col0_ofs + 10] == bytes([2, 1, 0, 50, 0])
    assert blob[col0_ofs + 10] == 0xFF

    # Column 1 has one 2-pixel post starting at y=1
    assert blob[col1_ofs : col1_ofs + 6] == bytes([1, 2, 0, 40, 60, 0])
    assert blob[col1_ofs + 6] == 0xFF


def test_mapbuilder_single_and_two_sector_rooms() -> None:
    mb = MapBuilder("MAP01")
    sec1, sec2, portal_ld = mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=128,
        width2=128,
        height=128,
        portal_special=12,
        portal_args=(1, 16, 35, 0, 0),
    )
    assert (sec1, sec2) == (0, 1)
    assert portal_ld == 3

    mb.add_thing(
        x=64,
        y=64,
        type=1,
        tid=7,
        z=0,
        angle=90,
        flags=0x07E7,
        special=80,
        args=(1, 0, 2, 0, 0),
    )

    wb = WadBuilder("PWAD").add_map(mb)
    wf = WadFile.from_bytes(wb.to_bytes(), name="map01.wad")

    expected_order = [
        "MAP01",
        "THINGS",
        "LINEDEFS",
        "SIDEDEFS",
        "VERTEXES",
        "SEGS",
        "SSECTORS",
        "NODES",
        "SECTORS",
        "REJECT",
        "BLOCKMAP",
        "BEHAVIOR",
    ]
    assert [lump.name for lump in wf.lumps] == expected_order

    things_bytes = wf.get_bytes("THINGS")
    assert len(things_bytes) == 20
    tid, tx, ty, tz, angle, mtype, flags, spec, a1, a2, a3, a4, a5 = struct.unpack(
        "<7hB5B", things_bytes
    )
    assert (tid, tx, ty, tz, angle, mtype, flags) == (7, 64, 64, 0, 90, 1, 0x07E7)
    assert (spec, a1, a2, a3, a4, a5) == (80, 1, 0, 2, 0, 0)

    linedefs_bytes = wf.get_bytes("LINEDEFS")
    assert len(linedefs_bytes) == 7 * 16
    v1, v2, lflags, lspec, la1, la2, la3, la4, la5, s0, s1 = struct.unpack_from(
        "<3H6B2H", linedefs_bytes, portal_ld * 16
    )
    assert lflags & ML_TWOSIDED
    assert (lspec, la1, la2, la3, la4, la5) == (12, 1, 16, 35, 0, 0)
    assert s0 != 0xFFFF and s1 != 0xFFFF

    assert len(wf.get_bytes("SIDEDEFS")) == 8 * 30
    assert len(wf.get_bytes("VERTEXES")) == 6 * 4
    assert len(wf.get_bytes("SEGS")) == 8 * 12
    assert len(wf.get_bytes("SSECTORS")) == 2 * 4
    assert len(wf.get_bytes("SECTORS")) == 2 * 26
    assert wf.get_bytes("REJECT") == b"\x00"

    nodes_bytes = wf.get_bytes("NODES")
    assert len(nodes_bytes) == 28
    unpacked_node = struct.unpack("<12h2H", nodes_bytes)
    assert unpacked_node[-2] == (NF_SUBSECTOR | 0)
    assert unpacked_node[-1] == (NF_SUBSECTOR | 1)

    bmap_bytes = wf.get_bytes("BLOCKMAP")
    orgx, orgy, bw, bh = struct.unpack_from("<4h", bmap_bytes, 0)
    assert (orgx, orgy) == (0, 0)
    assert bw >= 1 and bh >= 1
    first_block_word_ofs = struct.unpack_from("<H", bmap_bytes, 8)[0]
    first_word = struct.unpack_from("<h", bmap_bytes, first_block_word_ofs * 2)[0]
    last_word = struct.unpack_from("<h", bmap_bytes, len(bmap_bytes) - 2)[0]
    assert first_word == 0
    assert last_word == -1

    behavior_bytes = wf.get_bytes("BEHAVIOR")
    assert behavior_bytes[:4] == b"ACS\x00"


def test_mapbuilder_single_and_multi_room_auto_nodes() -> None:
    mb1 = MapBuilder("MAP02")
    mb1.add_room(0, 0, 128, 128)
    wf1 = WadFile.from_bytes(WadBuilder("PWAD").add_map(mb1).to_bytes())
    assert len(wf1.get_bytes("NODES")) == 28

    mb3 = MapBuilder("MAP03")
    mb3.add_room(0, 0, 128, 128)
    mb3.add_room(256, 0, 128, 128)
    mb3.add_room(512, 0, 128, 128)
    wf3 = WadFile.from_bytes(WadBuilder("PWAD").add_map(mb3).to_bytes())
    assert len(wf3.get_bytes("SSECTORS")) == 3 * 4
    assert len(wf3.get_bytes("NODES")) == 2 * 28


def test_wadstack_pwad_override_and_find_all() -> None:
    iwad_bytes = (
        WadBuilder("IWAD")
        .add_lump("PLAYPAL", b"iwad_pal")
        .add_lump("TESTLUMP", b"from_iwad")
        .to_bytes()
    )
    pwad1_bytes = (
        WadBuilder("PWAD")
        .add_lump("TESTLUMP", b"from_pwad1")
        .add_lump("EXTRA", b"extra1")
        .to_bytes()
    )
    pwad2_bytes = WadBuilder("PWAD").add_lump("TESTLUMP", b"from_pwad2").to_bytes()

    with WadStack() as stack:
        stack.add_wad(iwad_bytes, source="hexen.wad")
        stack.add_wad(pwad1_bytes, source="mod1.wad")
        stack.add_wad(pwad2_bytes, source="mod2.wad")

        assert len(stack) == 5
        assert "PLAYPAL" in stack
        assert "MISSING" not in stack

        latest = stack.find("testlump")
        assert latest is not None
        assert latest.source == "mod2.wad"
        assert latest.index == 4
        assert stack.get_bytes("TESTLUMP") == b"from_pwad2"
        assert stack.get_bytes(latest) == b"from_pwad2"

        all_matches = stack.find_all("TESTLUMP")
        assert [m.source for m in all_matches] == ["hexen.wad", "mod1.wad", "mod2.wad"]
        assert [stack.get_bytes(m) for m in all_matches] == [
            b"from_iwad",
            b"from_pwad1",
            b"from_pwad2",
        ]

        with pytest.raises(LumpNotFoundError):
            stack.require("NO_SUCH")


def test_namespaces_s_ss_f_ff_p_pp_and_submarkers() -> None:
    iwad = (
        WadBuilder("IWAD")
        .add_marker("S_START")
        .add_lump("PLAYA1", b"spr1")
        .add_marker("S_END")
        .add_marker("F_START")
        .add_marker("F1_START")
        .add_flat("FLOOR01", fill=1)
        .add_flat("FLOOR02", fill=2)
        .add_marker("F1_END")
        .add_marker("F_END")
        .add_marker("P_START")
        .add_marker("P1_START")
        .add_lump("WALL01", b"patch1")
        .add_marker("P1_END")
        .add_marker("P_END")
        .to_bytes()
    )
    pwad = (
        WadBuilder("PWAD")
        .add_marker("SS_START")
        .add_lump("PLAYA2", b"spr2")
        .add_marker("SS_END")
        .add_marker("FF_START")
        .add_flat("FLOOR01", fill=99)
        .add_flat("FLOOR03", fill=3)
        .add_marker("FF_END")
        .add_marker("PP_START")
        .add_lump("WALL02", b"patch2")
        .add_marker("PP_END")
        .to_bytes()
    )

    with WadStack() as stack:
        stack.add_wad(iwad, source="iwad.wad")
        stack.add_wad(pwad, source="pwad.wad")

        sprites = sprite_lumps(stack)
        assert [s.name for s in sprites] == ["PLAYA1", "PLAYA2"]

        flats_all = flat_lumps(stack, deduplicate=False)
        assert [f.name for f in flats_all] == [
            "FLOOR01",
            "FLOOR02",
            "FLOOR01",
            "FLOOR03",
        ]

        flats_dedup = flat_lumps(stack, deduplicate=True)
        assert [f.name for f in flats_dedup] == ["FLOOR01", "FLOOR02", "FLOOR03"]
        assert stack.get_bytes(flats_dedup[0])[0] == 99

        patches = patch_lumps(stack)
        assert [p.name for p in patches] == ["WALL01", "WALL02"]


def test_malformed_wad_rejection() -> None:
    # Too short (< 12 bytes)
    with pytest.raises(WadError):
        WadFile.from_bytes(b"IWAD\x00")

    # Bad magic
    with pytest.raises(WadError):
        WadFile.from_bytes(struct.pack("<4sii", b"DOOM", 0, 12))

    # Negative lump count
    with pytest.raises(WadError):
        WadFile.from_bytes(struct.pack("<4sii", b"IWAD", -1, 12))

    # Directory out of bounds
    with pytest.raises(WadError):
        WadFile.from_bytes(struct.pack("<4sii", b"IWAD", 2, 12))

    # Lump data out of bounds
    bad_lump_wad = struct.pack("<4sii", b"PWAD", 1, 12) + struct.pack(
        "<ii8s", 100, 50, b"BADLUMP\x00"
    )
    with pytest.raises(WadError):
        WadFile.from_bytes(bad_lump_wad)
