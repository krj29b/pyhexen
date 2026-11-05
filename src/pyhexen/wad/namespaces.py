# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""WAD namespace enumeration for sprites, flats, and patches.

Handles both IWAD markers (S_START/S_END, F_START/F_END, P_START/P_END) and
PWAD marker variants (SS_START/SS_END, FF_START/FF_END, PP_START/PP_END),
while filtering out internal sub-markers such as F1_START/F1_END and P1_START/P1_END.
"""

from __future__ import annotations

from collections.abc import Iterator

from pyhexen.wad.wadfile import LumpInfo, WadFile, WadStack

SPRITE_START_MARKERS: tuple[str, ...] = ("S_START", "SS_START")
SPRITE_END_MARKERS: tuple[str, ...] = ("S_END", "SS_END")

FLAT_START_MARKERS: tuple[str, ...] = ("F_START", "FF_START")
FLAT_END_MARKERS: tuple[str, ...] = ("F_END", "FF_END")

PATCH_START_MARKERS: tuple[str, ...] = ("P_START", "PP_START")
PATCH_END_MARKERS: tuple[str, ...] = ("P_END", "PP_END")


def is_submarker(lump: LumpInfo) -> bool:
    """Return True if `lump` is a namespace sub-marker (e.g. F1_START, P2_END) or empty marker."""
    return lump.size == 0 or lump.name.endswith(("_START", "_END"))


def _filter_namespace_lumps(
    raw_lumps: list[LumpInfo],
    *,
    deduplicate: bool = False,
) -> list[LumpInfo]:
    filtered = [lump for lump in raw_lumps if not is_submarker(lump)]
    if not deduplicate:
        return filtered
    # Keep last match per lump name while preserving relative order of final occurrences
    by_name: dict[str, LumpInfo] = {}
    for lump in filtered:
        by_name[lump.name] = lump
    return list(by_name.values())


def sprite_lumps(
    wad: WadFile | WadStack,
    *,
    deduplicate: bool = False,
) -> list[LumpInfo]:
    """Enumerate all sprite lumps between S_START/SS_START and S_END/SS_END.

    Port of R_InitSpriteLumps (r_data.c), extended for PWAD SS_* markers.
    """
    raw = wad.lumps_between(SPRITE_START_MARKERS, SPRITE_END_MARKERS)
    return _filter_namespace_lumps(raw, deduplicate=deduplicate)


def flat_lumps(
    wad: WadFile | WadStack,
    *,
    deduplicate: bool = False,
) -> list[LumpInfo]:
    """Enumerate all flat lumps between F_START/FF_START and F_END/FF_END.

    Excludes sub-markers such as F1_START, F1_END, F2_START, F2_END.
    Port of R_InitFlats (r_data.c), extended for PWAD FF_* markers.
    """
    raw = wad.lumps_between(FLAT_START_MARKERS, FLAT_END_MARKERS)
    return _filter_namespace_lumps(raw, deduplicate=deduplicate)


def patch_lumps(
    wad: WadFile | WadStack,
    *,
    deduplicate: bool = False,
) -> list[LumpInfo]:
    """Enumerate all patch lumps between P_START/PP_START and P_END/PP_END.

    Excludes sub-markers such as P1_START, P1_END, P2_START, P2_END, P3_START, P3_END.
    """
    raw = wad.lumps_between(PATCH_START_MARKERS, PATCH_END_MARKERS)
    return _filter_namespace_lumps(raw, deduplicate=deduplicate)


def iter_sprites(wad: WadFile | WadStack, *, deduplicate: bool = False) -> Iterator[LumpInfo]:
    """Yield sprite lumps from `wad`."""
    return iter(sprite_lumps(wad, deduplicate=deduplicate))


def iter_flats(wad: WadFile | WadStack, *, deduplicate: bool = False) -> Iterator[LumpInfo]:
    """Yield flat lumps from `wad`."""
    return iter(flat_lumps(wad, deduplicate=deduplicate))


def iter_patches(wad: WadFile | WadStack, *, deduplicate: bool = False) -> Iterator[LumpInfo]:
    """Yield patch lumps from `wad`."""
    return iter(patch_lumps(wad, deduplicate=deduplicate))


# Aliases for convenience
get_sprite_lumps = sprite_lumps
get_flat_lumps = flat_lumps
get_patch_lumps = patch_lumps
