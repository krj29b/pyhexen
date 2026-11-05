# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T8.1: BEHAVIOR lump loader (`sim/acs/loader.py`)."""

from __future__ import annotations

import struct

import pytest

from pyhexen.sim.acs import OPEN_SCRIPTS_BASE, ACSError, parse_behavior


def _build_synthetic_behavior(
    scripts: list[tuple[int, bytes, int]],
    strings: list[str],
) -> bytes:
    """Build a synthetic Hexen BEHAVIOR lump from scripts and strings."""
    out = bytearray(b"ACS\x00\x00\x00\x00\x00")  # 8-byte header placeholder

    script_entries: list[tuple[int, int, int]] = []
    for num, code, argc in scripts:
        code_ofs = len(out)
        out.extend(code)
        script_entries.append((num, code_ofs, argc))

    string_offsets: list[int] = []
    for s in strings:
        str_ofs = len(out)
        out.extend(s.encode("latin1") + b"\x00")
        string_offsets.append(str_ofs)

    # Align directory to 4 bytes if needed
    while len(out) % 4 != 0:
        out.append(0)

    info_offset = len(out)
    struct.pack_into("<i", out, 4, info_offset)

    out.extend(struct.pack("<i", len(script_entries)))
    for num, code_ofs, argc in script_entries:
        out.extend(struct.pack("<3i", num, code_ofs, argc))

    out.extend(struct.pack("<i", len(string_offsets)))
    for str_ofs in string_offsets:
        out.extend(struct.pack("<i", str_ofs))

    return bytes(out)


def test_parse_empty_behavior() -> None:
    raw = b"ACS\x00" + struct.pack("<iii", 8, 0, 0)
    beh = parse_behavior(raw)
    assert beh.info_offset == 8
    assert len(beh.scripts) == 0
    assert len(beh.strings) == 0
    assert beh.open_scripts == ()


def test_parse_behavior_with_scripts_and_strings() -> None:
    # Script 1: normal script, 2 args
    # Script 1002: open script #2, 0 args
    # Script 5: normal script with >3 args (clamped to MAX_SCRIPT_ARGS=3)
    code1 = struct.pack("<ii", 55, 1)  # DelayDirect(35), Terminate
    code2 = struct.pack("<i", 1)  # Terminate
    raw = _build_synthetic_behavior(
        scripts=[
            (1, code1, 2),
            (OPEN_SCRIPTS_BASE + 2, code2, 0),
            (5, code2, 7),
        ],
        strings=["DOOR LOCKED", "GREETINGS MORTAL", ""],
    )

    beh = parse_behavior(raw)
    assert len(beh.scripts) == 3

    s1 = beh.get_script(1)
    assert s1 is not None
    assert s1.number == 1
    assert s1.arg_count == 2
    assert s1.is_open is False
    assert beh.code[s1.offset : s1.offset + len(code1)] == code1

    s2 = beh.get_script(2)
    assert s2 is not None
    assert s2.number == 2
    assert s2.arg_count == 0
    assert s2.is_open is True
    assert beh.open_scripts == (s2,)

    s5 = beh.get_script(5)
    assert s5 is not None
    assert s5.arg_count == 3  # clamped to MAX_SCRIPT_ARGS

    assert beh.strings == ("DOOR LOCKED", "GREETINGS MORTAL", "")


def test_parse_behavior_rejects_malformed() -> None:
    with pytest.raises(ACSError, match="too small"):
        parse_behavior(b"ACS")

    with pytest.raises(ACSError, match="Invalid BEHAVIOR header"):
        parse_behavior(b"BAD\x00\x08\x00\x00\x00\x00\x00\x00\x00")

    with pytest.raises(ACSError, match="infoOffset"):
        parse_behavior(b"ACS\x00" + struct.pack("<i", 999))
