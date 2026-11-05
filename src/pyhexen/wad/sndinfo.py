# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""SNDINFO lump parser.

Port of S_InitScript (s_sound.c).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyhexen.wad.scanner import Scanner
from pyhexen.wad.wadfile import WadFile, WadStack, normalize_lump_name

SNDINFO_SCRIPT_NAME = "SNDINFO"


@dataclass(slots=True)
class SndInfo:
    """Parsed SNDINFO sound lump mappings and map music assignments.

    Port of S_InitScript data (s_sound.c).
    """

    sounds: dict[str, str] = field(default_factory=dict)
    map_music: dict[int, str] = field(default_factory=dict)
    pitch_shift_range: int = 7
    archive_path: str = ""

    def get_lump(self, logical_name: str) -> str | None:
        """Look up the uppercase WAD lump name for `logical_name` (case-insensitive fallback)."""
        if logical_name in self.sounds:
            return self.sounds[logical_name]
        lower = logical_name.lower()
        for k, v in self.sounds.items():
            if k.lower() == lower:
                return v
        return None


def parse_sndinfo(text_or_bytes: str | bytes) -> SndInfo:
    """Parse a SNDINFO script into a SndInfo instance.

    Port of S_InitScript (s_sound.c).
    """
    sc = Scanner(text_or_bytes, name=SNDINFO_SCRIPT_NAME)
    sounds: dict[str, str] = {}
    map_music: dict[int, str] = {}
    pitch_shift_range = 7
    archive_path = ""

    while sc.get_string():
        token = sc.string
        if token.startswith("$"):
            if sc.compare("$ARCHIVEPATH"):
                archive_path = sc.must_get_string()
            elif sc.compare("$MAP"):
                map_num = sc.must_get_number()
                song_name = sc.must_get_string()
                if map_num:
                    map_music[map_num] = normalize_lump_name(song_name)
            elif sc.compare("$PITCHSHIFTRANGE"):
                pitch_shift_range = sc.must_get_number()
            elif sc.compare("$REGISTERED"):
                pass
            continue

        logical_name = token
        lump_token = sc.must_get_string()
        if lump_token.startswith("?"):
            sounds[logical_name] = "DEFAULT"
        else:
            sounds[logical_name] = normalize_lump_name(lump_token)

    return SndInfo(
        sounds=sounds,
        map_music=map_music,
        pitch_shift_range=pitch_shift_range,
        archive_path=archive_path,
    )


def load_sndinfo(wad: WadStack | WadFile) -> SndInfo:
    """Load and parse the SNDINFO lump from a WAD or WadStack."""
    return parse_sndinfo(wad.get_bytes(SNDINFO_SCRIPT_NAME))
