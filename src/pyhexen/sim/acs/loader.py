# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""ACS BEHAVIOR bytecode loader.

Port of P_LoadACScripts (p_acs.c).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

MAX_SCRIPT_ARGS: int = 3
OPEN_SCRIPTS_BASE: int = 1000
MAX_ACS_SCRIPT_VARS: int = 10
MAX_ACS_MAP_VARS: int = 32
MAX_ACS_WORLD_VARS: int = 64
ACS_STACK_DEPTH: int = 32
MAX_ACS_STORE: int = 20


class ACSError(ValueError):
    """Raised when an ACS BEHAVIOR lump is malformed or fails an assertion."""


class ACScriptState(IntEnum):
    """Port of aste_t (p_spec.h)."""

    ASTE_INACTIVE = 0
    ASTE_RUNNING = 1
    ASTE_SUSPENDED = 2
    ASTE_WAITINGFORTAG = 3
    ASTE_WAITINGFORPOLY = 4
    ASTE_WAITINGFORSCRIPT = 5
    ASTE_TERMINATING = 6


@dataclass(slots=True, frozen=True)
class ACScriptDef:
    """Port of acsInfo_t directory entry in BEHAVIOR (p_spec.h / p_acs.c)."""

    number: int
    offset: int
    arg_count: int
    is_open: bool = False


@dataclass(slots=True)
class BehaviorData:
    """Parsed BEHAVIOR lump holding raw bytecode, script directory, and string table."""

    code: bytes
    info_offset: int
    scripts: tuple[ACScriptDef, ...]
    strings: tuple[str, ...]
    scripts_by_number: dict[int, ACScriptDef]

    def get_script(self, number: int) -> ACScriptDef | None:
        """Look up a script definition by its script number."""
        return self.scripts_by_number.get(number)

    @property
    def open_scripts(self) -> tuple[ACScriptDef, ...]:
        """Return all scripts flagged as OPEN (`is_open == True`)."""
        return tuple(s for s in self.scripts if s.is_open)


def _read_int32(data: bytes, offset: int) -> tuple[int, int]:
    if offset < 0 or offset + 4 > len(data):
        raise ACSError(
            f"Unexpectedly reached end of ACS lump at offset {offset} (size {len(data)})"
        )
    (val,) = struct.unpack_from("<i", data, offset)
    return val, offset + 4


def _read_offset(data: bytes, offset: int) -> tuple[int, int]:
    val, next_offset = _read_int32(data, offset)
    if val < 0 or val >= len(data):
        raise ACSError(f"Invalid ACS lump offset: {val} (size {len(data)})")
    return val, next_offset


def parse_behavior(data: bytes) -> BehaviorData:
    """Port of P_LoadACScripts (p_acs.c).

    Parses a Hexen `BEHAVIOR` lump:
    - 4-byte marker (`"ACS\\x00"`)
    - 4-byte `info_offset` (little-endian int32)
    - At `info_offset`: `script_count` (int32), followed by `script_count` entries of
      `(number, offset, arg_count)` (3 x int32). Scripts with `number >= 1000` are OPEN
      scripts with effective number `number - 1000`.
    - When `script_count > 0`, followed by `string_count` (int32) and `string_count`
      int32 byte offsets to NUL-terminated strings in `data`. (If `script_count == 0`
      and a string count is present in bounds, it is also parsed if valid.)
    """
    if len(data) < 8:
        raise ACSError(f"BEHAVIOR lump too small ({len(data)} bytes < 8)")

    marker = data[:4]
    if marker[:3] != b"ACS":
        raise ACSError(f"Invalid BEHAVIOR header marker: {marker!r} (expected b'ACS\\x00')")

    info_offset, _ = _read_int32(data, 4)
    if info_offset < 0 or info_offset + 4 > len(data):
        raise ACSError(f"Invalid BEHAVIOR infoOffset {info_offset} for lump size {len(data)}")

    script_count, cursor = _read_int32(data, info_offset)
    if script_count < 0:
        raise ACSError(f"Negative ACS script count: {script_count}")

    if script_count == 0:
        return BehaviorData(
            code=bytes(data),
            info_offset=info_offset,
            scripts=(),
            strings=(),
            scripts_by_number={},
        )

    scripts: list[ACScriptDef] = []
    scripts_by_number: dict[int, ACScriptDef] = {}

    for _ in range(script_count):
        raw_num, cursor = _read_int32(data, cursor)
        code_ofs, cursor = _read_offset(data, cursor)
        arg_count, cursor = _read_int32(data, cursor)

        if arg_count > MAX_SCRIPT_ARGS:
            arg_count = MAX_SCRIPT_ARGS
        elif arg_count < 0:
            raise ACSError(f"Negative ACS script argCount: {arg_count}")

        is_open = raw_num >= OPEN_SCRIPTS_BASE
        number = raw_num - OPEN_SCRIPTS_BASE if is_open else raw_num
        sdef = ACScriptDef(
            number=number,
            offset=code_ofs,
            arg_count=arg_count,
            is_open=is_open,
        )
        scripts.append(sdef)
        scripts_by_number[number] = sdef

    string_count, cursor = _read_int32(data, cursor)
    if string_count < 0:
        raise ACSError(f"Negative ACS string count: {string_count}")

    strings: list[str] = []
    for i in range(string_count):
        str_ofs, cursor = _read_offset(data, cursor)
        nul_pos = data.find(b"\x00", str_ofs)
        if nul_pos == -1:
            raise ACSError(f"ACS string #{i} at offset {str_ofs} is missing terminating NUL")
        strings.append(data[str_ofs:nul_pos].decode("latin1"))

    return BehaviorData(
        code=bytes(data),
        info_offset=info_offset,
        scripts=tuple(scripts),
        strings=tuple(strings),
        scripts_by_number=scripts_by_number,
    )


load_behavior = parse_behavior
