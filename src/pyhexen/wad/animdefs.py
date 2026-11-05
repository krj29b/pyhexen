# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""ANIMDEFS lump parser for flat and wall texture animations.

Port of P_InitFTAnims (p_anim.c).
"""

from __future__ import annotations

from dataclasses import dataclass

from pyhexen.wad.scanner import Scanner
from pyhexen.wad.wadfile import WadFile, WadStack, normalize_lump_name

ANIM_SCRIPT_NAME = "ANIMDEFS"


@dataclass(slots=True)
class AnimFrame:
    """A single animation frame definition within an AnimDef.

    Port of frameDef_t (p_anim.c).
    """

    index: int | str
    tics: int = 0
    rand_min: int = 0
    rand_max: int = 0


@dataclass(slots=True)
class AnimDef:
    """A flat or texture animation definition from ANIMDEFS.

    Port of animDef_t (p_anim.c).
    """

    is_flat: bool
    name: str
    frames: tuple[AnimFrame, ...]


def parse_animdefs(text_or_bytes: str | bytes) -> list[AnimDef]:
    """Parse an ANIMDEFS script into a list of AnimDef entries.

    Port of P_InitFTAnims (p_anim.c).
    """
    sc = Scanner(text_or_bytes, name=ANIM_SCRIPT_NAME)
    anim_defs: list[AnimDef] = []

    while sc.get_string():
        if sc.compare("flat"):
            is_flat = True
        elif sc.compare("texture"):
            is_flat = False
        else:
            sc.script_error(None)

        name = normalize_lump_name(sc.must_get_string())
        frames: list[AnimFrame] = []

        while sc.get_string():
            if sc.compare("pic"):
                raw_index = sc.must_get_string()
                try:
                    frame_index: int | str = int(raw_index, 0)
                except ValueError:
                    frame_index = normalize_lump_name(raw_index)

                sc.must_get_string()
                if sc.compare("tics"):
                    tics = sc.must_get_number()
                    frames.append(AnimFrame(index=frame_index, tics=tics))
                elif sc.compare("rand"):
                    rand_min = sc.must_get_number()
                    rand_max = sc.must_get_number()
                    frames.append(
                        AnimFrame(
                            index=frame_index,
                            tics=0,
                            rand_min=rand_min,
                            rand_max=rand_max,
                        )
                    )
                else:
                    sc.script_error(None)
            else:
                sc.unget()
                break

        if len(frames) < 2:
            sc.script_error("P_InitFTAnims: AnimDef has framecount < 2.")

        anim_defs.append(AnimDef(is_flat=is_flat, name=name, frames=tuple(frames)))

    return anim_defs


def load_animdefs(wad: WadStack | WadFile) -> list[AnimDef]:
    """Load and parse the ANIMDEFS lump from a WAD or WadStack."""
    return parse_animdefs(wad.get_bytes(ANIM_SCRIPT_NAME))
