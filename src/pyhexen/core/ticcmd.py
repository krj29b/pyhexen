# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Per-tic player input command structure and bitfield helpers.

Port of d_ticcmd.h, h2def.h (BT_* and AFLAG_* constants), g_game.c, and p_user.c.
"""

from __future__ import annotations

from dataclasses import dataclass

# Button flags (h2def.h)
BT_ATTACK: int = 1
BT_USE: int = 2
BT_CHANGE: int = 4
BT_WEAPONMASK: int = 8 + 16 + 32
BT_WEAPONSHIFT: int = 3

BT_SPECIAL: int = 128
BTS_SAVEMASK: int = 4 + 8 + 16
BTS_SAVESHIFT: int = 2
BT_SPECIALMASK: int = 3
BTS_PAUSE: int = 1
BTS_SAVEGAME: int = 2

# Artifact & jump flags in ticcmd_t.arti (h2def.h)
AFLAG_MASK: int = 0x3F
AFLAG_SUICIDE: int = 0x40
AFLAG_JUMP: int = 0x80

# Special look/fly centering value (p_local.h)
TOCENTER: int = -8


def pack_lookfly(look: int, fly: int) -> int:
    """Pack signed 4-bit look and fly deltas (-8..7) into an unsigned byte.

    Port of lookfly packing in G_BuildTiccmd (g_game.c).
    """
    if not -8 <= look <= 7:
        raise ValueError(f"look delta must be in -8..7, got {look}")
    if not -8 <= fly <= 7:
        raise ValueError(f"fly delta must be in -8..7, got {fly}")
    look_nibble = (look + 16) if look < 0 else look
    fly_nibble = (fly + 16) if fly < 0 else fly
    return ((fly_nibble & 0x0F) << 4) | (look_nibble & 0x0F)


def unpack_lookfly(lookfly: int) -> tuple[int, int]:
    """Unpack an unsigned lookfly byte into signed 4-bit (look, fly) deltas (-8..7).

    Port of lookfly unpacking in P_MovePlayer (p_user.c).
    """
    look = lookfly & 0x0F
    if look > 7:
        look -= 16
    fly = (lookfly >> 4) & 0x0F
    if fly > 7:
        fly -= 16
    return look, fly


@dataclass(slots=True)
class TicCmd:
    """Input command for a single 35 Hz simulation tic.

    Port of ticcmd_t (d_ticcmd.h).
    """

    forwardmove: int = 0
    sidemove: int = 0
    angleturn: int = 0
    consistancy: int = 0
    chatchar: int = 0
    buttons: int = 0
    lookfly: int = 0
    arti: int = 0

    @property
    def look_delta(self) -> int:
        """Return signed 4-bit look delta (-8..7, where -8 is TOCENTER)."""
        look, _ = unpack_lookfly(self.lookfly)
        return look

    @property
    def fly_delta(self) -> int:
        """Return signed 4-bit fly delta (-8..7, where -8 is TOCENTER)."""
        _, fly = unpack_lookfly(self.lookfly)
        return fly

    @property
    def is_jump(self) -> bool:
        """Return True if the jump flag is set in arti."""
        return bool(self.arti & AFLAG_JUMP)

    @property
    def artifact_type(self) -> int:
        """Return the artifact ID (0..63) masked from arti."""
        return self.arti & AFLAG_MASK

    @property
    def weapon_change(self) -> int | None:
        """Return the 0-based weapon index if BT_CHANGE is set, else None."""
        if self.buttons & BT_CHANGE:
            return (self.buttons & BT_WEAPONMASK) >> BT_WEAPONSHIFT
        return None
