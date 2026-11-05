# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for Scanner, MAPINFO, SNDINFO, SNDSEQ, and ANIMDEFS parsers (T2.6, T2.7)."""

from __future__ import annotations

import pytest
from tests.fixtures.wadbuilder import WadBuilder

from pyhexen.wad.animdefs import AnimDef, AnimFrame, load_animdefs, parse_animdefs
from pyhexen.wad.mapinfo import MapInfo, load_mapinfo, parse_mapinfo
from pyhexen.wad.scanner import Scanner, ScriptError
from pyhexen.wad.sndinfo import load_sndinfo, parse_sndinfo
from pyhexen.wad.sndseq import SeqOpcode, load_sndseq, parse_sndseq
from pyhexen.wad.wadfile import WadStack


class TestScanner:
    def test_tokens_quotes_and_comments(self) -> None:
        script = """
        ; Semicolon line comment
        // C++ style comment
        /* Multi-line
           block comment */
        map 1 "WINNOWING HALL" ; inline comment
        sky1 SKY2 0x10
        """
        sc = Scanner(script, name="TEST")
        assert sc.get_string() is True
        assert sc.string == "map"
        assert sc.quoted is False
        assert sc.compare("MAP") is True

        assert sc.must_get_number() == 1
        assert sc.crossed is False

        assert sc.must_get_string() == "WINNOWING HALL"
        assert sc.quoted is True

        assert sc.must_get_string() == "sky1"
        assert sc.crossed is True
        assert sc.must_get_string() == "SKY2"
        assert sc.must_get_number() == 16

        assert sc.get_string() is False
        assert sc.end is True

    def test_unget_and_matching(self) -> None:
        sc = Scanner("CLUSTER 1 NEXT 2", name="TEST")
        sc.must_get_string_name("cluster")
        idx = sc.must_match_string(["SKY1", "CLUSTER", "NEXT"])
        assert idx == 1

        sc.unget()
        assert sc.must_get_string() == "CLUSTER"
        assert sc.must_get_number() == 1
        sc.must_get_string_name("NEXT")
        assert sc.must_get_number() == 2

    def test_errors(self) -> None:
        sc = Scanner("notanumber", name="ERR")
        with pytest.raises(ScriptError, match="Bad numeric constant"):
            sc.must_get_number()

        sc_empty = Scanner("; only comment\n", name="EMPTY")
        with pytest.raises(ScriptError, match="Missing string"):
            sc_empty.must_get_string()

        sc_mismatch = Scanner("FOO", name="MISMATCH")
        with pytest.raises(ScriptError):
            sc_mismatch.must_get_string_name("BAR")


class TestMapInfo:
    def test_parse_and_load_mapinfo(self) -> None:
        mapinfo_text = """
        ; Hub 1 maps
        map 1 "WINNOWING HALL"
        warptrans 1
        next 2
        cluster 1
        sky1 SKY2 0
        sky2 SKY3 32
        lightning
        doublesky
        fadetable FOGMAP
        cdtrack 13

        map 2 "SEVEN PORTALS"
        warptrans 2
        next 3
        secretnext 4
        cluster 1
        sky1 SKY1 16
        cdtrack 2

        cd_start_track 1
        cd_intermission_track 5
        cd_title_track 6
        """
        infos = parse_mapinfo(mapinfo_text)
        assert len(infos) == 2

        m1 = infos[0]
        assert m1 == MapInfo(
            map_num=1,
            name="WINNOWING HALL",
            lump_name="MAP01",
            cluster=1,
            warp_trans=1,
            next_map=2,
            secret_map=1,
            sky1_texture="SKY2",
            sky1_scroll_delta=0,
            sky2_texture="SKY3",
            sky2_scroll_delta=32 << 8,
            double_sky=True,
            lightning=True,
            fadetable="FOGMAP",
            cd_track=13,
            music_lump="",
        )

        m2 = infos[1]
        assert m2.map_num == 2
        assert m2.name == "SEVEN PORTALS"
        assert m2.lump_name == "MAP02"
        assert m2.next_map == 3
        assert m2.secret_map == 4
        assert m2.sky1_scroll_delta == 16 << 8
        assert m2.double_sky is False
        assert m2.lightning is False
        assert m2.fadetable == "COLORMAP"

        wb = WadBuilder()
        wb.add_lump("MAPINFO", mapinfo_text.encode("latin1"))
        wb.add_lump("SNDINFO", b"$MAP 1 Winnowr\n$MAP 2 Jachr\n")
        stack = WadStack(wb.to_bytes())

        table = load_mapinfo(stack)
        assert 1 in table
        assert "MAP01" in table
        assert table[1].music_lump == "WINNOWR"
        assert table["MAP02"].music_lump == "JACHR"
        assert table.translate_warp(2) == 2
        assert table.translate_warp(99) == -1


class TestSndInfo:
    def test_parse_and_load_sndinfo(self) -> None:
        sndinfo_text = """
        ; Sound definitions
        $ARCHIVEPATH "o:\\sound\\archive\\"
        $registered
        $pitchshiftrange 4
        $MAP 1 Winnowr
        $MAP 2 Jachr
        PlayerFighterNormalDeath   fgtddth1
        PlayerFighterPain          fgtpain1
        UnassignedSound            ?default
        """
        wb = WadBuilder()
        wb.add_lump("SNDINFO", sndinfo_text.encode("latin1"))
        stack = WadStack(wb.to_bytes())

        info = load_sndinfo(stack)
        assert info.archive_path == "o:\\sound\\archive\\"
        assert info.pitch_shift_range == 4
        assert info.map_music == {1: "WINNOWR", 2: "JACHR"}
        assert info.sounds["PlayerFighterNormalDeath"] == "FGTDDTH1"
        assert info.sounds["PlayerFighterPain"] == "FGTPAIN1"
        assert info.sounds["UnassignedSound"] == "DEFAULT"
        assert info.get_lump("playerfighternormaldeath") == "FGTDDTH1"

        direct = parse_sndinfo(sndinfo_text)
        assert direct.sounds == info.sounds


class TestSndSeq:
    def test_parse_and_load_sndseq(self) -> None:
        sndseq_text = """
        :Platform
            play PlatformStart
            playrepeat StoneMove
            stopsound PlatformStop
        end

        :DoorNormal
            play DoorOpen
            delay 8
            playuntildone DoorCreak
            playtime StoneMove 15
            delayrand 5 20
            volume 80
            stopsound DoorCloseLight
        end
        """
        wb = WadBuilder()
        wb.add_lump("SNDSEQ", sndseq_text.encode("latin1"))
        stack = WadStack(wb.to_bytes())

        seqs = load_sndseq(stack)
        assert set(seqs.keys()) == {"Platform", "DoorNormal"}

        plat = seqs["Platform"]
        assert plat.name == "Platform"
        assert plat.stop_sound == "PlatformStop"
        assert plat.instructions == (
            (SeqOpcode.SS_CMD_PLAY, "PlatformStart"),
            (SeqOpcode.SS_CMD_PLAYREPEAT, "StoneMove"),
            (SeqOpcode.SS_CMD_STOPSOUND,),
            (SeqOpcode.SS_CMD_END,),
        )

        door = seqs["DoorNormal"]
        assert door.stop_sound == "DoorCloseLight"
        assert door.instructions == (
            (SeqOpcode.SS_CMD_PLAY, "DoorOpen"),
            (SeqOpcode.SS_CMD_DELAY, 8),
            (SeqOpcode.SS_CMD_PLAY, "DoorCreak"),
            (SeqOpcode.SS_CMD_WAIT,),
            (SeqOpcode.SS_CMD_PLAY, "StoneMove"),
            (SeqOpcode.SS_CMD_DELAY, 15),
            (SeqOpcode.SS_CMD_DELAYRAND, 5, 20),
            (SeqOpcode.SS_CMD_VOLUME, 80),
            (SeqOpcode.SS_CMD_STOPSOUND,),
            (SeqOpcode.SS_CMD_END,),
        )

    def test_nested_sequence_error(self) -> None:
        bad = ":Seq1\n play Foo\n:Seq2\n end\n"
        with pytest.raises(ScriptError, match="Nested Script Error"):
            parse_sndseq(bad)


class TestAnimDefs:
    def test_parse_and_load_animdefs(self) -> None:
        animdefs_text = """
        ; Water flat animation
        flat X_001
            pic 1 tics 4
            pic 2 tics 4
            pic 3 tics 4

        ; Portal texture with random tic range
        texture X_PORTAL
            pic 1 rand 4 12
            pic 2 rand 6 16
        """
        wb = WadBuilder()
        wb.add_lump("ANIMDEFS", animdefs_text.encode("latin1"))
        stack = WadStack(wb.to_bytes())

        defs = load_animdefs(stack)
        assert len(defs) == 2

        assert defs[0] == AnimDef(
            is_flat=True,
            name="X_001",
            frames=(
                AnimFrame(index=1, tics=4),
                AnimFrame(index=2, tics=4),
                AnimFrame(index=3, tics=4),
            ),
        )
        assert defs[1] == AnimDef(
            is_flat=False,
            name="X_PORTAL",
            frames=(
                AnimFrame(index=1, tics=0, rand_min=4, rand_max=12),
                AnimFrame(index=2, tics=0, rand_min=6, rand_max=16),
            ),
        )

    def test_single_frame_rejected(self) -> None:
        bad = "flat X_001\n pic 1 tics 4\n"
        with pytest.raises(ScriptError, match="framecount < 2"):
            parse_animdefs(bad)
