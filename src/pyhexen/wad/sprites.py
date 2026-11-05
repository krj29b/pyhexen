# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sprite frame and rotation registry.

Port of R_InitSpriteDefs and R_InstallSpriteLump (r_things.c).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from pyhexen.sim.info_data import SPRNAMES
from pyhexen.wad.namespaces import sprite_lumps
from pyhexen.wad.wadfile import LumpInfo, WadFile, WadStack, normalize_lump_name

MAX_SPRITE_FRAMES = 30


@dataclass(slots=True, frozen=True)
class SpriteFrame:
    """One animation frame (A..Z, [, \\, ]) of a sprite across 8 viewing rotations.

    Port of spriteframe_t (r_local.h).
    `rotate` is 0 if a single lump is used for all 8 angles, or 1 if 8 distinct
    rotations are used.
    """

    rotate: int
    lumps: tuple[LumpInfo, ...]  # length 8
    flip: tuple[bool, ...]  # length 8


@dataclass(slots=True, frozen=True)
class SpriteDef:
    """All animation frames for a 4-character sprite prefix.

    Port of spritedef_t (r_local.h).
    """

    name: str
    frames: tuple[SpriteFrame, ...]

    @property
    def numframes(self) -> int:
        return len(self.frames)


class _MutableFrame:
    __slots__ = ("rotate", "lumps", "flip")

    def __init__(self) -> None:
        self.rotate: int = -1
        self.lumps: list[LumpInfo | None] = [None] * 8
        self.flip: list[bool] = [False] * 8


def _parse_frame_and_rotation(
    lump_name: str,
    frame_char: str,
    rot_char: str,
) -> tuple[int, int]:
    frame = ord(frame_char) - ord("A")
    rotation = ord(rot_char) - ord("0")
    if frame < 0 or frame >= MAX_SPRITE_FRAMES or rotation < 0 or rotation > 8:
        raise ValueError(
            f"R_InstallSpriteLump: Bad frame/rotation characters {frame_char + rot_char!r} "
            f"in lump {lump_name!r}"
        )
    return frame, rotation


def _install_sprite_lump(
    spritename: str,
    sprtemp: list[_MutableFrame],
    lump: LumpInfo,
    frame: int,
    rotation: int,
    flipped: bool,
) -> int:
    """Port of R_InstallSpriteLump (r_things.c)."""
    mf = sprtemp[frame]
    frame_letter = chr(ord("A") + frame)

    if rotation == 0:
        if mf.rotate == 0:
            raise ValueError(
                f"R_InitSprites: Sprite {spritename} frame {frame_letter} has multiple rot=0 lumps"
            )
        if mf.rotate == 1:
            raise ValueError(
                f"R_InitSprites: Sprite {spritename} frame {frame_letter} "
                "has rotations and a rot=0 lump"
            )
        mf.rotate = 0
        for r in range(8):
            mf.lumps[r] = lump
            mf.flip[r] = flipped
        return frame

    if mf.rotate == 0:
        raise ValueError(
            f"R_InitSprites: Sprite {spritename} frame {frame_letter} "
            "has rotations and a rot=0 lump"
        )
    mf.rotate = 1
    rot_idx = rotation - 1
    if mf.lumps[rot_idx] is not None:
        raise ValueError(
            f"R_InitSprites: Sprite {spritename} : {frame_letter} : {rotation} "
            "has two lumps mapped to it"
        )
    mf.lumps[rot_idx] = lump
    mf.flip[rot_idx] = flipped
    return frame


class SpriteRegistry:
    """Builds the sprite -> frames -> 8 rotations (lump, flip) table from WAD sprite lumps.

    Port of R_InitSpriteDefs (r_things.c).
    """

    def __init__(
        self,
        wad: WadStack | WadFile,
        sprite_names: Sequence[str] | None = None,
    ) -> None:
        # Deduplicate by lump name so PWAD sprite replacements override IWAD lumps cleanly
        lumps = sprite_lumps(wad, deduplicate=True)

        if sprite_names is None:
            names_list = [normalize_lump_name(n)[:4] for n in SPRNAMES]
            seen = set(names_list)
            for lump in lumps:
                if len(lump.name) >= 6:
                    prefix = lump.name[:4]
                    if prefix not in seen:
                        seen.add(prefix)
                        names_list.append(prefix)
        else:
            names_list = [normalize_lump_name(n)[:4] for n in sprite_names]

        # Group sprite lumps by 4-char prefix for O(1) per-sprite scanning
        by_prefix: dict[str, list[LumpInfo]] = {}
        for lump in lumps:
            if len(lump.name) < 6:
                raise ValueError(
                    f"R_InitSprites: Sprite lump name {lump.name!r} is shorter than 6 characters"
                )
            by_prefix.setdefault(lump.name[:4], []).append(lump)

        self._sprites: list[SpriteDef] = []
        self._by_name: dict[str, int] = {}

        for idx, spritename in enumerate(names_list):
            matching_lumps = by_prefix.get(spritename, [])
            if not matching_lumps:
                sdef = SpriteDef(name=spritename, frames=())
                self._sprites.append(sdef)
                self._by_name.setdefault(spritename, idx)
                continue

            sprtemp = [_MutableFrame() for _ in range(MAX_SPRITE_FRAMES)]
            maxframe = -1

            for lump in matching_lumps:
                lname = lump.name
                if len(lname) not in (6, 8):
                    raise ValueError(
                        f"R_InitSprites: Invalid sprite lump name length for {lname!r}"
                    )
                f1, r1 = _parse_frame_and_rotation(lname, lname[4], lname[5])
                maxframe = max(
                    maxframe,
                    _install_sprite_lump(spritename, sprtemp, lump, f1, r1, False),
                )
                if len(lname) == 8:
                    f2, r2 = _parse_frame_and_rotation(lname, lname[6], lname[7])
                    maxframe = max(
                        maxframe,
                        _install_sprite_lump(spritename, sprtemp, lump, f2, r2, True),
                    )

            frames: list[SpriteFrame] = []
            for frame_idx in range(maxframe + 1):
                mf = sprtemp[frame_idx]
                frame_letter = chr(ord("A") + frame_idx)
                if mf.rotate == -1:
                    raise ValueError(
                        f"R_InitSprites: No patches found for {spritename} frame {frame_letter}"
                    )
                if mf.rotate == 1:
                    for rot_idx in range(8):
                        if mf.lumps[rot_idx] is None:
                            raise ValueError(
                                f"R_InitSprites: Sprite {spritename} frame {frame_letter} "
                                f"is missing rotation {rot_idx + 1}"
                            )
                frames.append(
                    SpriteFrame(
                        rotate=mf.rotate,
                        lumps=tuple(lump for lump in mf.lumps if lump is not None),
                        flip=tuple(mf.flip),
                    )
                )

            sdef = SpriteDef(name=spritename, frames=tuple(frames))
            self._sprites.append(sdef)
            self._by_name.setdefault(spritename, idx)

    @property
    def sprites(self) -> tuple[SpriteDef, ...]:
        """All registered `SpriteDef` entries in order."""
        return tuple(self._sprites)

    def __len__(self) -> int:
        return len(self._sprites)

    def __contains__(self, name: str) -> bool:
        norm = normalize_lump_name(name)[:4]
        idx = self._by_name.get(norm, -1)
        return idx != -1 and len(self._sprites[idx].frames) > 0

    def __iter__(self) -> Iterator[SpriteDef]:
        return iter(self._sprites)

    def __getitem__(self, key: int | str) -> SpriteDef:
        return self.get(key)

    def get(self, index_or_name: int | str) -> SpriteDef:
        """Return `SpriteDef` by sprite index (e.g. `SpriteNum`) or 4-char sprite name."""
        if isinstance(index_or_name, int):
            if index_or_name < 0 or index_or_name >= len(self._sprites):
                raise IndexError(
                    f"Sprite index {index_or_name} out of range [0..{len(self._sprites)})"
                )
            return self._sprites[index_or_name]
        if isinstance(index_or_name, str):
            norm = normalize_lump_name(index_or_name)[:4]
            idx = self._by_name.get(norm, -1)
            if idx == -1:
                raise KeyError(f"Sprite {norm!r} not found in SpriteRegistry")
            return self._sprites[idx]
        raise TypeError(f"Unsupported sprite selector: {type(index_or_name)!r}")
