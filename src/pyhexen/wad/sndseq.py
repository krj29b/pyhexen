# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""SNDSEQ lump parser and sound sequence opcodes.

Port of SN_InitSequenceScript (sn_sonix.c).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from pyhexen.wad.scanner import Scanner
from pyhexen.wad.wadfile import WadFile, WadStack

SNDSEQ_SCRIPT_NAME = "SNDSEQ"


class SeqOpcode(IntEnum):
    """Sound sequence bytecode opcodes.

    Port of sscmds_t (sn_sonix.c).
    """

    SS_CMD_NONE = 0
    SS_CMD_PLAY = 1
    SS_CMD_WAIT = 2
    SS_CMD_WAITUNTILDONE = 2
    SS_CMD_PLAYTIME = 3
    SS_CMD_PLAYREPEAT = 4
    SS_CMD_PLAYLOOP = 5
    SS_CMD_DELAY = 6
    SS_CMD_DELAYRAND = 7
    SS_CMD_VOLUME = 8
    SS_CMD_STOPSOUND = 9
    SS_CMD_END = 10


# Standard sequence translation table from sn_sonix.c (SequenceTranslate[SEQ_NUMSEQ])
SEQUENCE_TRANSLATE: tuple[str, ...] = (
    "Platform",
    "Platform",  # heavy platform
    "PlatformMetal",
    "Platform",  # creak platform
    "Silence",
    "Lava",
    "Water",
    "Ice",
    "Earth",
    "PlatformMetal2",
    "DoorNormal",
    "DoorHeavy",
    "DoorMetal",
    "DoorCreak",
    "Silence",
    "Lava",
    "Water",
    "Ice",
    "Earth",
    "DoorMetal2",
    "Wind",
)

type SeqInstruction = tuple[SeqOpcode | str | int, ...]


@dataclass(slots=True)
class SoundSequence:
    """A parsed sound sequence definition from SNDSEQ.

    Port of SequenceData / SequenceTranslate entry (sn_sonix.c).
    """

    name: str
    instructions: tuple[SeqInstruction, ...]
    stop_sound: str | None = None


def parse_sndseq(text_or_bytes: str | bytes) -> dict[str, SoundSequence]:
    """Parse a SNDSEQ script into a dictionary of sequence name -> SoundSequence.

    Port of SN_InitSequenceScript (sn_sonix.c).
    """
    sc = Scanner(text_or_bytes, name=SNDSEQ_SCRIPT_NAME)
    sequences: dict[str, SoundSequence] = {}

    current_name: str | None = None
    current_instructions: list[SeqInstruction] = []
    current_stop_sound: str | None = None

    while sc.get_string():
        if sc.string.startswith(":"):
            if current_name is not None:
                sc.script_error("SN_InitSequenceScript:  Nested Script Error")
            current_name = sc.string[1:]
            if not current_name:
                sc.script_error("SN_InitSequenceScript:  Empty sequence name")
            current_instructions = []
            current_stop_sound = None
            continue

        if current_name is None:
            continue

        if sc.compare("playuntildone"):
            sound_name = sc.must_get_string()
            current_instructions.append((SeqOpcode.SS_CMD_PLAY, sound_name))
            current_instructions.append((SeqOpcode.SS_CMD_WAIT,))
        elif sc.compare("play"):
            sound_name = sc.must_get_string()
            current_instructions.append((SeqOpcode.SS_CMD_PLAY, sound_name))
        elif sc.compare("playtime"):
            sound_name = sc.must_get_string()
            tics = sc.must_get_number()
            current_instructions.append((SeqOpcode.SS_CMD_PLAY, sound_name))
            current_instructions.append((SeqOpcode.SS_CMD_DELAY, tics))
        elif sc.compare("playrepeat"):
            sound_name = sc.must_get_string()
            current_instructions.append((SeqOpcode.SS_CMD_PLAYREPEAT, sound_name))
        elif sc.compare("playloop"):
            sound_name = sc.must_get_string()
            tics = sc.must_get_number()
            current_instructions.append((SeqOpcode.SS_CMD_PLAYLOOP, sound_name, tics))
        elif sc.compare("delay"):
            tics = sc.must_get_number()
            current_instructions.append((SeqOpcode.SS_CMD_DELAY, tics))
        elif sc.compare("delayrand"):
            min_tics = sc.must_get_number()
            max_tics = sc.must_get_number()
            current_instructions.append((SeqOpcode.SS_CMD_DELAYRAND, min_tics, max_tics))
        elif sc.compare("volume"):
            vol = sc.must_get_number()
            current_instructions.append((SeqOpcode.SS_CMD_VOLUME, vol))
        elif sc.compare("stopsound"):
            current_stop_sound = sc.must_get_string()
            current_instructions.append((SeqOpcode.SS_CMD_STOPSOUND,))
        elif sc.compare("end"):
            current_instructions.append((SeqOpcode.SS_CMD_END,))
            sequences[current_name] = SoundSequence(
                name=current_name,
                instructions=tuple(current_instructions),
                stop_sound=current_stop_sound,
            )
            current_name = None
            current_instructions = []
            current_stop_sound = None
        else:
            sc.script_error("SN_InitSequenceScript:  Unknown commmand.")

    if current_name is not None:
        sc.script_error("SN_InitSequenceScript:  Unterminated sequence.")

    return sequences


def load_sndseq(wad: WadStack | WadFile) -> dict[str, SoundSequence]:
    """Load and parse the SNDSEQ lump from a WAD or WadStack."""
    return parse_sndseq(wad.get_bytes(SNDSEQ_SCRIPT_NAME))
