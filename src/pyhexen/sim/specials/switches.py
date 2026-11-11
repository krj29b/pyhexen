# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Wall switch texture toggling and timed button reversion thinker.

Port of P_InitSwitchList, P_ChangeSwitchTexture, P_StartButton (p_switch.c)
and P_UpdateSpecials button countdown (p_spec.c).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.sim.info_data import SfxEnum
from pyhexen.sim.sound import start_sound
from pyhexen.sim.thinker import Thinker

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.world.level import Line

BUTTONTIME: int = 35
MAXBUTTONS: int = 16


class SwitchWhere(IntEnum):
    """Port of bwhere_e (p_spec.h)."""

    SWTCH_TOP = 0
    SWTCH_MIDDLE = 1
    SWTCH_BOTTOM = 2


@dataclass(slots=True, frozen=True)
class SwitchDef:
    """Port of switchlist_t (p_spec.h)."""

    name1: str
    name2: str
    sound_id: int


# Hexen switch pairs from alphSwitchListFull (p_switch.c) plus additional Hexen switch textures
SWITCH_LIST: tuple[SwitchDef, ...] = (
    SwitchDef("SW_1_UP", "SW_1_DN", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("SW_2_UP", "SW_2_DN", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("VALVE1", "VALVE2", int(SfxEnum.SFX_VALVE_TURN)),
    SwitchDef("SW51_OFF", "SW51_ON", int(SfxEnum.SFX_SWITCH2)),
    SwitchDef("SW52_OFF", "SW52_ON", int(SfxEnum.SFX_SWITCH2)),
    SwitchDef("SW53_UP", "SW53_DN", int(SfxEnum.SFX_ROPE_PULL)),
    SwitchDef("PUZZLE5", "PUZZLE9", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("PUZZLE6", "PUZZLE10", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("PUZZLE7", "PUZZLE11", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("PUZZLE8", "PUZZLE12", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("SW_EL1", "SW_EL2", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("SW_EL3", "SW_EL4", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("SW_EL5", "SW_EL1", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("SW_OL1", "SW_OL2", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("SW_OL3", "SW_OL4", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("SW_OL5", "SW_OL1", int(SfxEnum.SFX_SWITCH1)),
    SwitchDef("SW_STEEL1", "SW_STEEL2", int(SfxEnum.SFX_SWITCH2)),
    SwitchDef("SW_CHAIN1", "SW_CHAIN2", int(SfxEnum.SFX_ROPE_PULL)),
)

# Map from texture name (upper-case) to (partner_name, sound_id)
_SWITCH_LOOKUP: dict[str, tuple[str, int]] = {}
for _sw in SWITCH_LIST:
    _SWITCH_LOOKUP.setdefault(_sw.name1.upper(), (_sw.name2.upper(), _sw.sound_id))
    _SWITCH_LOOKUP.setdefault(_sw.name2.upper(), (_sw.name1.upper(), _sw.sound_id))


class ButtonThinker(Thinker):
    """Timed button thinker that reverts a repeatable switch texture after `BUTTONTIME` tics.

    Port of button_t / P_StartButton (p_switch.c) and P_UpdateSpecials (p_spec.c).
    """

    __slots__ = ("line", "where", "btexture", "btimer", "sound_id")

    def __init__(
        self,
        line: Line,
        where: SwitchWhere,
        btexture: str,
        btimer: int = BUTTONTIME,
        sound_id: int = int(SfxEnum.SFX_SWITCH1),
    ) -> None:
        super().__init__()
        self.line: Line = line
        self.where: SwitchWhere = where
        self.btexture: str = btexture
        self.btimer: int = btimer
        self.sound_id: int = sound_id

    def tick(self, sim: GameSim) -> None:
        if self.btimer > 0:
            self.btimer -= 1
            if self.btimer == 0:
                side = self.line.front_side
                if side is not None:
                    if self.where == SwitchWhere.SWTCH_TOP:
                        side.top_texture = self.btexture
                    elif self.where == SwitchWhere.SWTCH_MIDDLE:
                        side.mid_texture = self.btexture
                    elif self.where == SwitchWhere.SWTCH_BOTTOM:
                        side.bottom_texture = self.btexture
                # No sound here: the button-revert sound is commented out in
                # vanilla P_UpdateSpecials (p_spec.c).
                sim.thinkers.remove(self)

    def state_tuple(self) -> tuple[Any, ...]:
        return ("ButtonThinker", self.line.index, int(self.where), self.btexture, self.btimer)

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "ButtonThinker",
            "line_index": self.line.index,
            "where": int(self.where),
            "btexture": self.btexture,
            "btimer": self.btimer,
            "sound_id": self.sound_id,
        }


def p_change_switch_texture(sim: GameSim, line: Line, repeat: bool) -> bool:
    """Toggle the switch texture on `line`'s front side and start a `ButtonThinker` if `repeat`.

    Port of P_ChangeSwitchTexture (p_switch.c).
    """
    side = line.front_side
    if side is None:
        return False

    sec = line.front_sector  # sound origin: the front sector's soundorg

    top_upper = side.top_texture.upper()
    if top_upper in _SWITCH_LOOKUP:
        new_tex, sound_id = _SWITCH_LOOKUP[top_upper]
        old_tex = side.top_texture
        side.top_texture = new_tex
        start_sound(sim, sec, sound_id)
        if repeat:
            sim.thinkers.add(ButtonThinker(line, SwitchWhere.SWTCH_TOP, old_tex, BUTTONTIME))
        return True

    mid_upper = side.mid_texture.upper()
    if mid_upper in _SWITCH_LOOKUP:
        new_tex, sound_id = _SWITCH_LOOKUP[mid_upper]
        old_tex = side.mid_texture
        side.mid_texture = new_tex
        start_sound(sim, sec, sound_id)
        if repeat:
            sim.thinkers.add(ButtonThinker(line, SwitchWhere.SWTCH_MIDDLE, old_tex, BUTTONTIME))
        return True

    bot_upper = side.bottom_texture.upper()
    if bot_upper in _SWITCH_LOOKUP:
        new_tex, sound_id = _SWITCH_LOOKUP[bot_upper]
        old_tex = side.bottom_texture
        side.bottom_texture = new_tex
        start_sound(sim, sec, sound_id)
        if repeat:
            sim.thinkers.add(ButtonThinker(line, SwitchWhere.SWTCH_BOTTOM, old_tex, BUTTONTIME))
        return True

    return False


SPECIAL_HANDLERS: dict[int, Callable[..., bool]] = {}
