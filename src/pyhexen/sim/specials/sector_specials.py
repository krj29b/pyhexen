# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sector specials: secret areas, wind/scroll pushers, lava/damage floors, and phased lights.

Port of P_PlayerInSpecialSector, P_PlayerOnSpecialFlat, P_SpawnSpecials (p_spec.c),
and T_Phase, P_SpawnPhasedLight, P_SpawnLightSequence (p_lights.c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import ANG45, ANG90, ANG180, ANG270
from pyhexen.core.fixed import FRACBITS, FRACUNIT, fixed_div
from pyhexen.sim.info_data import SfxEnum
from pyhexen.sim.sound import start_sound
from pyhexen.sim.specials.dispatcher import get_next_sector, notify_sector_changed
from pyhexen.sim.specials.scroll import spawn_line_scrollers
from pyhexen.sim.thinker import Thinker

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.sim.player import Player
    from pyhexen.world.level import Sector

LIGHT_SEQUENCE_START: int = 2
LIGHT_SEQUENCE: int = 3
LIGHT_SEQUENCE_ALT: int = 4

PUSH_TAB: tuple[int, int, int] = (2048 * 5, 2048 * 10, 2048 * 25)

# PhaseTable[64] from p_lights.c
PHASE_TABLE: tuple[int, ...] = (
    128,
    112,
    96,
    80,
    64,
    48,
    32,
    32,
    16,
    16,
    16,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    16,
    16,
    16,
    32,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
)


class PhasedLightThinker(Thinker):
    """Phased sector light thinker.

    Port of phase_t / T_Phase (p_lights.c).
    """

    __slots__ = ("sector", "index", "base")

    def __init__(self, sector: Sector, base: int, index: int) -> None:
        super().__init__()
        self.sector: Sector = sector
        self.base: int = base & 255
        self.index: int = index & 63

    def tick(self, sim: GameSim) -> None:
        self.index = (self.index + 1) & 63
        self.sector.light_level = self.base + PHASE_TABLE[self.index]
        notify_sector_changed(sim, self.sector)

    def state_tuple(self) -> tuple[Any, ...]:
        return ("PhasedLightThinker", self.sector.index, self.base, self.index)

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "PhasedLightThinker",
            "sector_index": self.sector.index,
            "base": self.base,
            "index": self.index,
        }


def p_spawn_phased_light(
    sim: GameSim,
    sector: Sector,
    base: int,
    index: int = -1,
) -> PhasedLightThinker:
    """Spawn a `PhasedLightThinker` for `sector`.

    Port of P_SpawnPhasedLight (p_lights.c).
    """
    idx = (sector.light_level & 63) if index == -1 else (index & 63)
    b = base & 255
    sector.light_level = b + PHASE_TABLE[idx]
    sector.special = 0
    thinker = PhasedLightThinker(sector=sector, base=b, index=idx)
    sim.thinkers.add(thinker)
    return thinker


def p_spawn_light_sequence(sim: GameSim, sector: Sector, index_step: int = 1) -> None:
    """Spawn a linked sequence of phased lights across adjacent sectors.

    Port of P_SpawnLightSequence (p_lights.c).
    """
    seq_special = LIGHT_SEQUENCE
    sec: Sector | None = sector
    count = 1
    while sec is not None:
        next_sec: Sector | None = None
        sec.special = LIGHT_SEQUENCE_START
        for line in sec.lines:
            temp_sec = get_next_sector(line, sec)
            if temp_sec is None:
                continue
            if temp_sec.special == seq_special:
                seq_special = (
                    LIGHT_SEQUENCE_ALT if seq_special == LIGHT_SEQUENCE else LIGHT_SEQUENCE
                )
                next_sec = temp_sec
                count += 1
        sec = next_sec

    sec = sector
    count *= index_step
    index = 0
    index_delta = fixed_div(64 * FRACUNIT, count * FRACUNIT)
    base = sector.light_level
    while sec is not None:
        next_sec = None
        if sec.light_level:
            base = sec.light_level
        p_spawn_phased_light(sim, sec, base, index >> FRACBITS)
        index += index_delta
        for line in sec.lines:
            temp_sec = get_next_sector(line, sec)
            if temp_sec is not None and temp_sec.special == LIGHT_SEQUENCE_START:
                next_sec = temp_sec
        sec = next_sec


def p_player_in_special_sector(sim: GameSim, player: Player) -> None:
    """Port of P_PlayerInSpecialSector (p_spec.c).

    Called only when the player's sector has a special. Wind (40-51) is applied in
    P_XYMovement, stairs, lightning and sky specials elsewhere; vanilla has no damaging
    sector specials (lava hurts through the floor's terrain type instead).
    """
    from pyhexen.sim.player import p_thrust

    mo = player.mo
    if mo is None or mo.subsector is None:
        return
    sector = mo.subsector.sector
    if mo.z != sector.floor_height:
        return  # Player is not touching the floor

    special = sector.special
    if special == 9:  # SecretArea
        player.secret_count += 1
        sector.special = 0
    elif 201 <= special <= 203:  # Scroll_North_xxx
        p_thrust(player, ANG90, PUSH_TAB[special - 201])
    elif 204 <= special <= 206:  # Scroll_East_xxx
        p_thrust(player, 0, PUSH_TAB[special - 204])
    elif 207 <= special <= 209:  # Scroll_South_xxx
        p_thrust(player, ANG270, PUSH_TAB[special - 207])
    elif 210 <= special <= 212:  # Scroll_West_xxx
        p_thrust(player, ANG180, PUSH_TAB[special - 210])
    elif 213 <= special <= 215:  # Scroll_NorthWest_xxx
        p_thrust(player, (ANG90 + ANG45) & 0xFFFFFFFF, PUSH_TAB[special - 213])
    elif 216 <= special <= 218:  # Scroll_NorthEast_xxx
        p_thrust(player, ANG45, PUSH_TAB[special - 216])
    elif 219 <= special <= 221:  # Scroll_SouthEast_xxx
        p_thrust(player, (ANG270 + ANG45) & 0xFFFFFFFF, PUSH_TAB[special - 219])
    elif 222 <= special <= 224:  # Scroll_SouthWest_xxx
        p_thrust(player, (ANG180 + ANG45) & 0xFFFFFFFF, PUSH_TAB[special - 222])
    # Anything else: vanilla I_Errors on unknown specials; ignored here.


def p_player_on_special_flat(sim: GameSim, player: Player, floor_type: int) -> None:
    """Port of P_PlayerOnSpecialFlat (p_spec.c): lava burns every 32 tics.

    The damage goes through P_DamageMobj with LavaInflictor (MT_CIRCLEFLAME, fire
    damage, no thrust), so class armour saves part of it and pain chance draws P_Random.
    """
    from pyhexen.sim.actions.monsters_common import FLOOR_LAVA
    from pyhexen.sim.info_data import MobjFlag2, MobjType
    from pyhexen.sim.inter import damage_mobj
    from pyhexen.sim.mobj import Mobj

    mo = player.mo
    if mo is None or mo.z != mo.floorz:
        return  # Player is not touching the floor
    if floor_type == FLOOR_LAVA and not (sim.level_time & 31):
        lava_inflictor = Mobj(
            type=MobjType.MT_CIRCLEFLAME,
            flags2=int(MobjFlag2.MF2_FIREDAMAGE | MobjFlag2.MF2_NODMGTHRUST),
        )
        damage_mobj(sim, mo, lava_inflictor, None, 10)
        start_sound(sim, mo, int(SfxEnum.SFX_LAVA_SIZZLE))


def p_spawn_specials(sim: GameSim) -> None:
    """Scan level sectors and lines after map load to spawn sector/line thinkers.

    Port of P_SpawnSpecials (p_spec.c).
    """
    level = sim.level
    if level is None:
        return

    for sector in level.sectors:
        if sector.special == 1:
            p_spawn_phased_light(sim, sector, 80, -1)
        elif sector.special == 2:
            p_spawn_light_sequence(sim, sector, 1)

    sim.tagged_lines.clear()
    for line in level.lines:
        if line.special == 121:  # Line_SetIdentification
            if line.args[0] != 0:
                sim.tagged_lines.setdefault(line.args[0], []).append(line)
            line.special = 0

    # Init other misc stuff
    sim.active_ceilings = [None] * len(sim.active_ceilings)
    sim.active_plats = [None] * len(sim.active_plats)

    spawn_line_scrollers(sim)
