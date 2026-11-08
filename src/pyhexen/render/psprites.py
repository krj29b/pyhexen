# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""First-person weapon overlay (psprite) rendering (T10.6).

Port of PSpriteSY, R_DrawPSprite, and R_DrawPlayerSprites (r_things.c:660-831).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.fixed import FRACBITS, FRACUNIT
from pyhexen.render.overlay2d import VIRTUAL_HEIGHT, VIRTUAL_WIDTH, Overlay2D
from pyhexen.render.sprites import FF_FRAMEMASK, FF_FULLBRIGHT
from pyhexen.sim.info_data import STATES, MobjFlag, StateNum
from pyhexen.sim.player import PW_INVULNERABILITY, Player
from pyhexen.sim.psprites import NUMPSPRITES, PspDef
from pyhexen.ui.fonts import blit_picture
from pyhexen.wad.picture import Picture, decode_picture
from pyhexen.wad.sprites import SpriteRegistry
from pyhexen.wad.wadfile import WadFile, WadStack

if TYPE_CHECKING:
    pass

BASEYCENTER: int = 100
SBARHEIGHT: int = 39
VIEWHEIGHT_SBAR: int = VIRTUAL_HEIGHT - SBARHEIGHT

# Y-adjustment values for full screen (4 classes x 4 weapons)
# Port of PSpriteSY[NUMCLASSES][NUMWEAPONS] (r_things.c:661-665).
PSPRITE_SY: tuple[tuple[int, int, int, int], ...] = (
    (0, -12 * FRACUNIT, -10 * FRACUNIT, 10 * FRACUNIT),  # Fighter
    (-8 * FRACUNIT, 10 * FRACUNIT, 10 * FRACUNIT, 0),  # Cleric
    (9 * FRACUNIT, 20 * FRACUNIT, 20 * FRACUNIT, 20 * FRACUNIT),  # Mage
    (10 * FRACUNIT, 10 * FRACUNIT, 10 * FRACUNIT, 10 * FRACUNIT),  # Pig
)


def compute_psprite_draw_pos(
    psp: PspDef,
    player: Player,
    *,
    fullscreen: bool = True,
    alpha: float = 1.0,
) -> tuple[int, int]:
    """Compute 320x200 virtual screen `(x, y)` anchor for `psp` before patch offsets.

    Port of coordinate projection in `R_DrawPSprite` and `R_DrawVisSprite` (`r_things.c`).
    """
    a = max(0.0, min(1.0, float(alpha)))
    sx = psp.prev_sx + (psp.sx - psp.prev_sx) * a
    sy = psp.prev_sy + (psp.sy - psp.prev_sy) * a

    pclass = max(0, min(3, int(player.player_class)))
    readyweapon = max(0, min(3, int(player.readyweapon)))

    if fullscreen:
        sy += PSPRITE_SY[pclass][readyweapon]
        y_center_adj = 0
    else:
        y_center_adj = (VIEWHEIGHT_SBAR // 2) - BASEYCENTER

    x = int(sx) >> FRACBITS
    y = (int(sy) >> FRACBITS) + y_center_adj
    return x, y


def is_psprite_translucent(player: Player) -> bool:
    """Return True if the player's weapon sprite should be drawn translucent/shadowed.

    Port of shadow/invulnerability checks in `R_DrawPSprite` (`r_things.c:765-777`).
    """
    mo = player.mo
    if mo is None:
        return False
    if player.player_class == 1 and (
        len(player.powers) > PW_INVULNERABILITY
        and player.powers[PW_INVULNERABILITY] > 4 * 32
        or (len(player.powers) > PW_INVULNERABILITY and (player.powers[PW_INVULNERABILITY] & 8))
    ):
        return True
    return bool(mo.flags & (MobjFlag.MF_SHADOW | MobjFlag.MF_ALTSHADOW))


def is_psprite_fullbright(psp: PspDef) -> bool:
    """Return True if `psp` is currently on a fullbright state frame."""
    if psp.state is None or psp.state_num == int(StateNum.S_NULL):
        return False
    return bool(psp.state.frame & FF_FULLBRIGHT)


def resolve_psprite_picture(
    psp: PspDef,
    sprite_registry: SpriteRegistry,
    wad: WadStack | WadFile,
    cache: dict[str, Picture] | None = None,
) -> tuple[Picture, bool] | None:
    """Resolve the `Picture` and horizontal `flip` flag for `psp` from `sprite_registry`.

    Port of sprite/frame lookup in `R_DrawPSprite` (`r_things.c:687-700`).
    """
    if psp.state is None or psp.state_num == int(StateNum.S_NULL):
        return None
    st = STATES[psp.state_num] if 0 <= psp.state_num < len(STATES) else psp.state
    sprite_idx = int(st.sprite)
    frame_idx = int(st.frame) & FF_FRAMEMASK

    if sprite_idx < 0 or sprite_idx >= len(sprite_registry):
        return None
    sprdef = sprite_registry.get(sprite_idx)
    if frame_idx < 0 or frame_idx >= sprdef.numframes:
        return None

    sprframe = sprdef.frames[frame_idx]
    lump = sprframe.lumps[0]
    flip = bool(sprframe.flip[0])

    if cache is not None and lump.name in cache:
        return cache[lump.name], flip

    raw = wad.get_bytes(lump)
    pic = decode_picture(raw)
    if cache is not None:
        cache[lump.name] = pic
    return pic, flip


def draw_psprites_to_overlay(
    overlay: Overlay2D,
    player: Player,
    sprite_registry: SpriteRegistry,
    wad: WadStack | WadFile,
    *,
    fullscreen: bool = True,
    alpha: float = 1.0,
    picture_cache: dict[str, Picture] | None = None,
) -> int:
    """Draw active `ps_weapon` and `ps_flash` sprites for `player` onto `overlay`.

    Port of `R_DrawPlayerSprites` and `R_DrawPSprite` (`r_things.c:668-831`).
    Returns the number of psprites drawn.
    """
    mo = player.mo
    if mo is not None and (mo.flags2 & 0x00000080):  # MF2_DONTDRAW
        # In Hexen Cleric wraithverge invulnerability, MF2_DONTDRAW on player still draws
        # translucent weapon via viewangleoffset check, except when viewcamera is active.
        pass

    clip_h = VIRTUAL_HEIGHT if fullscreen else VIEWHEIGHT_SBAR
    translucent = is_psprite_translucent(player)
    drawn = 0

    for i in range(min(NUMPSPRITES, len(player.psprites))):
        psp = player.psprites[i]
        if psp.state is None or psp.state_num == int(StateNum.S_NULL):
            continue
        resolved = resolve_psprite_picture(psp, sprite_registry, wad, cache=picture_cache)
        if resolved is None:
            continue
        pic, flip = resolved
        x, y = compute_psprite_draw_pos(psp, player, fullscreen=fullscreen, alpha=alpha)
        blit_picture(
            overlay.indices[:clip_h, :VIRTUAL_WIDTH],
            overlay.mask[:clip_h, :VIRTUAL_WIDTH],
            pic,
            x,
            y,
            flip=flip,
            translucent=translucent,
        )
        drawn += 1

    return drawn


class PspriteRenderer:
    """Caches decoded weapon `Picture` patches and renders player psprites to an `Overlay2D`."""

    def __init__(
        self,
        wad: WadStack | WadFile,
        sprite_registry: SpriteRegistry | None = None,
    ) -> None:
        self.wad = wad
        self.sprite_registry = (
            sprite_registry if sprite_registry is not None else SpriteRegistry(wad)
        )
        self._cache: dict[str, Picture] = {}

    def draw(
        self,
        overlay: Overlay2D,
        player: Player,
        *,
        fullscreen: bool = True,
        alpha: float = 1.0,
    ) -> int:
        """Render `player`'s active psprites onto `overlay`."""
        return draw_psprites_to_overlay(
            overlay,
            player,
            self.sprite_registry,
            self.wad,
            fullscreen=fullscreen,
            alpha=alpha,
            picture_cache=self._cache,
        )
