# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.
"""Player weapon framework and psprite state machine (Task T10.6).

Ports `WeaponInfo`, `WeaponManaUse`, `P_SetPsprite`, `P_SetPspriteNF`,
`P_SetupPsprites`, `P_MovePsprites`, `P_BringUpWeapon`, `P_CheckMana`,
`P_FireWeapon`, `P_DropWeapon`, `P_ActivateMorphWeapon`, `P_PostMorphWeapon`,
`AdjustPlayerAngle`, `A_WeaponReady`, `A_ReFire`, `A_Lower`, `A_Raise`,
`A_SnoutAttack`, and `A_Light0` from Chocolate Doom's `src/hexen/p_pspr.c`,
plus `P_GiveMana`, `TryPickupWeapon`, and `TryPickupWeaponPiece` (4th weapon
assembly from pieces) from `src/hexen/p_inter.c`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from pyhexen.core.angles import (
    ANG1,
    FINEANGLES,
    FINEMASK,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.events import PrintMessage, SoundEvent
from pyhexen.core.fixed import FRACBITS, FRACUNIT, fixed_mul, to_int32, to_uint32
from pyhexen.core.ticcmd import BT_ATTACK
from pyhexen.sim.actions import ActionCallable, action, get_action
from pyhexen.sim.attacks import aim_line_attack, line_attack
from pyhexen.sim.enemy import MELEERANGE, noise_alert
from pyhexen.sim.info_data import (
    STATES,
    MobjFlag2,
    MobjType,
    SfxEnum,
    State,
    StateNum,
)
from pyhexen.sim.mobj import Mobj, remove_mobj, set_mobj_state
from pyhexen.sim.specials.dispatcher import execute_line_special

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.sim.player import Player

# =============================================================================
# Constants & Enumerations (h2def.h, p_pspr.c)
# =============================================================================

LOWERSPEED: int = 6 * FRACUNIT
RAISESPEED: int = 6 * FRACUNIT
WEAPONBOTTOM: int = 128 * FRACUNIT
WEAPONTOP: int = 32 * FRACUNIT
MAX_MANA: int = 200
BONUSADD: int = 6
MAX_ANGADJUST: int = 5 * ANG1

WPIECE1: int = 1
WPIECE2: int = 2
WPIECE3: int = 4
WPIECE_ALL: int = WPIECE1 | WPIECE2 | WPIECE3

# Cooperative weapon piece translation table (p_inter.c:425)
PIECE_VALUE_TRANS: tuple[int, ...] = (
    0,  # 0: never
    WPIECE1 | WPIECE2 | WPIECE3,  # WPIECE1 (1)
    WPIECE2 | WPIECE3,  # WPIECE2 (2)
    0,  # 3: never
    WPIECE3,  # WPIECE3 (4)
)


class PsprNum(IntEnum):
    """Port of psprnum_t (h2def.h)."""

    PS_WEAPON = 0
    PS_FLASH = 1
    NUMPSPRITES = 2


ps_weapon: int = int(PsprNum.PS_WEAPON)
ps_flash: int = int(PsprNum.PS_FLASH)
PS_WEAPON: int = ps_weapon
PS_FLASH: int = ps_flash
NUMPSPRITES: int = int(PsprNum.NUMPSPRITES)


class WeaponType(IntEnum):
    """Port of weapontype_t (h2def.h)."""

    WP_FIRST = 0
    WP_SECOND = 1
    WP_THIRD = 2
    WP_FOURTH = 3
    NUMWEAPONS = 4
    WP_NOCHANGE = 5


WP_FIRST: int = int(WeaponType.WP_FIRST)
WP_SECOND: int = int(WeaponType.WP_SECOND)
WP_THIRD: int = int(WeaponType.WP_THIRD)
WP_FOURTH: int = int(WeaponType.WP_FOURTH)
NUMWEAPONS: int = int(WeaponType.NUMWEAPONS)
WP_NOCHANGE: int = int(WeaponType.WP_NOCHANGE)


class ManaType(IntEnum):
    """Port of manatype_t (h2def.h)."""

    MANA_1 = 0
    MANA_2 = 1
    NUMMANA = 2
    MANA_BOTH = 3
    MANA_NONE = 4


MANA_1: int = int(ManaType.MANA_1)
MANA_2: int = int(ManaType.MANA_2)
NUMMANA: int = int(ManaType.NUMMANA)
MANA_BOTH: int = int(ManaType.MANA_BOTH)
MANA_NONE: int = int(ManaType.MANA_NONE)


@dataclass(slots=True, frozen=True)
class WeaponInfo:
    """Port of weaponinfo_t (h2def.h)."""

    mana: ManaType
    upstate: StateNum
    downstate: StateNum
    readystate: StateNum
    atkstate: StateNum
    holdatkstate: StateNum
    flashstate: StateNum


@dataclass(slots=True)
class PspDef:
    """Player view sprite state definition.

    Port of pspdef_t (h2def.h).
    """

    state: State | None = None
    state_num: int = int(StateNum.S_NULL)
    tics: int = 0
    sx: int = 0
    sy: int = WEAPONBOTTOM
    prev_sx: int = 0
    prev_sy: int = WEAPONBOTTOM

    def capture_prev_state(self) -> None:
        """Copy current psprite coordinates into prev_* for sub-tic interpolation."""
        self.prev_sx = self.sx
        self.prev_sy = self.sy


# WeaponInfo[NUMWEAPONS][NUMCLASSES] from p_pspr.c:48-200
WEAPON_INFO: tuple[tuple[WeaponInfo, ...], ...] = (
    # WP_FIRST (First Weapons)
    (
        # Fighter - Spiked Gauntlets (Punch)
        WeaponInfo(
            mana=ManaType.MANA_NONE,
            upstate=StateNum.S_PUNCHUP,
            downstate=StateNum.S_PUNCHDOWN,
            readystate=StateNum.S_PUNCHREADY,
            atkstate=StateNum.S_PUNCHATK1_1,
            holdatkstate=StateNum.S_PUNCHATK1_1,
            flashstate=StateNum.S_NULL,
        ),
        # Cleric - Mace of Contrition
        WeaponInfo(
            mana=ManaType.MANA_NONE,
            upstate=StateNum.S_CMACEUP,
            downstate=StateNum.S_CMACEDOWN,
            readystate=StateNum.S_CMACEREADY,
            atkstate=StateNum.S_CMACEATK_1,
            holdatkstate=StateNum.S_CMACEATK_1,
            flashstate=StateNum.S_NULL,
        ),
        # Mage - Sapphire Wand
        WeaponInfo(
            mana=ManaType.MANA_NONE,
            upstate=StateNum.S_MWANDUP,
            downstate=StateNum.S_MWANDDOWN,
            readystate=StateNum.S_MWANDREADY,
            atkstate=StateNum.S_MWANDATK_1,
            holdatkstate=StateNum.S_MWANDATK_1,
            flashstate=StateNum.S_NULL,
        ),
        # Pig - Snout
        WeaponInfo(
            mana=ManaType.MANA_NONE,
            upstate=StateNum.S_SNOUTUP,
            downstate=StateNum.S_SNOUTDOWN,
            readystate=StateNum.S_SNOUTREADY,
            atkstate=StateNum.S_SNOUTATK1,
            holdatkstate=StateNum.S_SNOUTATK1,
            flashstate=StateNum.S_NULL,
        ),
    ),
    # WP_SECOND (Second Weapons)
    (
        # Fighter - Timon's Axe (MANA_NONE in WeaponInfo so it can still swing without mana)
        WeaponInfo(
            mana=ManaType.MANA_NONE,
            upstate=StateNum.S_FAXEUP,
            downstate=StateNum.S_FAXEDOWN,
            readystate=StateNum.S_FAXEREADY,
            atkstate=StateNum.S_FAXEATK_1,
            holdatkstate=StateNum.S_FAXEATK_1,
            flashstate=StateNum.S_NULL,
        ),
        # Cleric - Serpent Staff
        WeaponInfo(
            mana=ManaType.MANA_1,
            upstate=StateNum.S_CSTAFFUP,
            downstate=StateNum.S_CSTAFFDOWN,
            readystate=StateNum.S_CSTAFFREADY,
            atkstate=StateNum.S_CSTAFFATK_1,
            holdatkstate=StateNum.S_CSTAFFATK_1,
            flashstate=StateNum.S_NULL,
        ),
        # Mage - Frost Shards (Cone of Shards)
        WeaponInfo(
            mana=ManaType.MANA_1,
            upstate=StateNum.S_CONEUP,
            downstate=StateNum.S_CONEDOWN,
            readystate=StateNum.S_CONEREADY,
            atkstate=StateNum.S_CONEATK1_1,
            holdatkstate=StateNum.S_CONEATK1_3,
            flashstate=StateNum.S_NULL,
        ),
        # Pig - Snout
        WeaponInfo(
            mana=ManaType.MANA_NONE,
            upstate=StateNum.S_SNOUTUP,
            downstate=StateNum.S_SNOUTDOWN,
            readystate=StateNum.S_SNOUTREADY,
            atkstate=StateNum.S_SNOUTATK1,
            holdatkstate=StateNum.S_SNOUTATK1,
            flashstate=StateNum.S_NULL,
        ),
    ),
    # WP_THIRD (Third Weapons)
    (
        # Fighter - Hammer of Retribution (MANA_NONE in WeaponInfo so it can melee without mana)
        WeaponInfo(
            mana=ManaType.MANA_NONE,
            upstate=StateNum.S_FHAMMERUP,
            downstate=StateNum.S_FHAMMERDOWN,
            readystate=StateNum.S_FHAMMERREADY,
            atkstate=StateNum.S_FHAMMERATK_1,
            holdatkstate=StateNum.S_FHAMMERATK_1,
            flashstate=StateNum.S_NULL,
        ),
        # Cleric - Firestorm (Flame Strike)
        WeaponInfo(
            mana=ManaType.MANA_2,
            upstate=StateNum.S_CFLAMEUP,
            downstate=StateNum.S_CFLAMEDOWN,
            readystate=StateNum.S_CFLAMEREADY1,
            atkstate=StateNum.S_CFLAMEATK_1,
            holdatkstate=StateNum.S_CFLAMEATK_1,
            flashstate=StateNum.S_NULL,
        ),
        # Mage - Arc of Death (Lightning)
        WeaponInfo(
            mana=ManaType.MANA_2,
            upstate=StateNum.S_MLIGHTNINGUP,
            downstate=StateNum.S_MLIGHTNINGDOWN,
            readystate=StateNum.S_MLIGHTNINGREADY,
            atkstate=StateNum.S_MLIGHTNINGATK_1,
            holdatkstate=StateNum.S_MLIGHTNINGATK_1,
            flashstate=StateNum.S_NULL,
        ),
        # Pig - Snout
        WeaponInfo(
            mana=ManaType.MANA_NONE,
            upstate=StateNum.S_SNOUTUP,
            downstate=StateNum.S_SNOUTDOWN,
            readystate=StateNum.S_SNOUTREADY,
            atkstate=StateNum.S_SNOUTATK1,
            holdatkstate=StateNum.S_SNOUTATK1,
            flashstate=StateNum.S_NULL,
        ),
    ),
    # WP_FOURTH (Fourth Weapons)
    (
        # Fighter - Quietus (Rune Sword)
        WeaponInfo(
            mana=ManaType.MANA_BOTH,
            upstate=StateNum.S_FSWORDUP,
            downstate=StateNum.S_FSWORDDOWN,
            readystate=StateNum.S_FSWORDREADY,
            atkstate=StateNum.S_FSWORDATK_1,
            holdatkstate=StateNum.S_FSWORDATK_1,
            flashstate=StateNum.S_NULL,
        ),
        # Cleric - Wraithverge (Holy Symbol)
        WeaponInfo(
            mana=ManaType.MANA_BOTH,
            upstate=StateNum.S_CHOLYUP,
            downstate=StateNum.S_CHOLYDOWN,
            readystate=StateNum.S_CHOLYREADY,
            atkstate=StateNum.S_CHOLYATK_1,
            holdatkstate=StateNum.S_CHOLYATK_1,
            flashstate=StateNum.S_NULL,
        ),
        # Mage - Bloodscourge (Staff)
        WeaponInfo(
            mana=ManaType.MANA_BOTH,
            upstate=StateNum.S_MSTAFFUP,
            downstate=StateNum.S_MSTAFFDOWN,
            readystate=StateNum.S_MSTAFFREADY,
            atkstate=StateNum.S_MSTAFFATK_1,
            holdatkstate=StateNum.S_MSTAFFATK_1,
            flashstate=StateNum.S_NULL,
        ),
        # Pig - Snout
        WeaponInfo(
            mana=ManaType.MANA_NONE,
            upstate=StateNum.S_SNOUTUP,
            downstate=StateNum.S_SNOUTDOWN,
            readystate=StateNum.S_SNOUTREADY,
            atkstate=StateNum.S_SNOUTATK1,
            holdatkstate=StateNum.S_SNOUTATK1,
            flashstate=StateNum.S_NULL,
        ),
    ),
)

# WeaponManaUse[NUMCLASSES][NUMWEAPONS] from p_pspr.c:204-209
WEAPON_MANA_USE: tuple[tuple[int, ...], ...] = (
    (0, 2, 3, 14),  # Fighter
    (0, 1, 4, 18),  # Cleric
    (0, 3, 5, 15),  # Mage
    (0, 0, 0, 0),  # Pig
)

FOURTH_WEAPON_TEXT: tuple[str, ...] = (
    "QUIETUS ASSEMBLED",
    "WRAITHVERGE ASSEMBLED",
    "BLOODSCOURGE ASSEMBLED",
)

WEAPON_PIECE_TEXT: tuple[str, ...] = (
    "QUIETUS PIECE",
    "WRAITHVERGE PIECE",
    "BLOODSCOURGE PIECE",
)


def _emit_sound(
    sim: GameSim,
    actor: Mobj | None,
    sound_id: int | SfxEnum,
    *,
    full_volume: bool = False,
) -> None:
    """Emit a SoundEvent on sim.events."""
    sid = int(sound_id)
    if sid <= 0:
        return
    origin_id = None if (full_volume or actor is None) else actor.tid
    x = actor.x if actor is not None else 0
    y = actor.y if actor is not None else 0
    z = actor.z if actor is not None else 0
    sim.events.emit(
        SoundEvent(
            sound_id=sid,
            origin_id=origin_id,
            x=x,
            y=y,
            z=z,
        )
    )


def _set_player_message(sim: GameSim, player: Player, msg: str) -> None:
    """Set player HUD message and emit PrintMessage event."""
    player.message = msg
    player.message_tics = 4 * 35
    sim.events.emit(PrintMessage(text=msg, player_index=0, bold=False))


def _invoke_psp_action(
    act: ActionCallable,
    sim: GameSim,
    player: Player,
    psp: PspDef,
) -> None:
    """Invoke a psprite action callable, adapting to its parameter signature."""
    if getattr(act, "_is_stub", False):
        act(sim, player, psp)
        return
    try:
        sig = inspect.signature(act)
    except (ValueError, TypeError):
        act(sim, player, psp)
        return

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        act(sim, player, psp)
        return

    pos_params = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(pos_params) == 0:
        act()
    elif len(pos_params) == 1:
        name0 = pos_params[0].name
        if name0 in ("sim", "game_sim"):
            act(sim)
        elif name0 in ("player", "plr"):
            act(player)
        elif name0 in ("psp", "psprite"):
            act(psp)
        else:
            act(player.mo)
    elif len(pos_params) == 2:
        names = (pos_params[0].name, pos_params[1].name)
        if names[0] in ("sim", "game_sim") and names[1] in ("player", "plr"):
            act(sim, player)
        elif names[0] in ("player", "plr") and names[1] in ("psp", "psprite"):
            act(player, psp)
        elif names[0] in ("mobj", "mo", "actor", "thing"):
            act(player.mo, sim)
        else:
            act(sim, player.mo)
    elif len(pos_params) == 3:
        if pos_params[0].name in ("mobj", "mo", "actor", "obj"):
            act(player.mo, player, psp)
        else:
            act(sim, player, psp)
    else:
        act(sim, player.mo, player, psp)


# =============================================================================
# Core Psprite State Machine (p_pspr.c:219-684)
# =============================================================================


def p_set_psprite(
    sim: GameSim,
    player: Player,
    position: int,
    stnum: int | StateNum,
) -> None:
    """Port of P_SetPsprite (p_pspr.c:219).

    Transitions `player.psprites[position]` through `stnum` and any 0-tic chain,
    executing each state's action function.
    """
    psp = player.psprites[position]
    st_idx = int(stnum)
    while True:
        if st_idx == int(StateNum.S_NULL):
            psp.state = None
            psp.state_num = int(StateNum.S_NULL)
            break

        state = STATES[st_idx]
        psp.state = state
        psp.state_num = st_idx
        psp.tics = state.tics

        if state.misc1:
            psp.sx = state.misc1 << FRACBITS
        if state.misc2:
            psp.sy = state.misc2 << FRACBITS

        act = get_action(state.action_name)
        if act is not None:
            _invoke_psp_action(act, sim, player, psp)
            if psp.state is None:
                break

        st_idx = psp.state.next_state
        if psp.tics != 0:
            break


def p_set_psprite_nf(
    player: Player,
    position: int,
    stnum: int | StateNum,
) -> None:
    """Port of P_SetPspriteNF (p_pspr.c:263).

    Identical to `p_set_psprite`, without calling state action functions.
    """
    psp = player.psprites[position]
    st_idx = int(stnum)
    while True:
        if st_idx == int(StateNum.S_NULL):
            psp.state = None
            psp.state_num = int(StateNum.S_NULL)
            break

        state = STATES[st_idx]
        psp.state = state
        psp.state_num = st_idx
        psp.tics = state.tics

        if state.misc1:
            psp.sx = state.misc1 << FRACBITS
        if state.misc2:
            psp.sy = state.misc2 << FRACBITS

        st_idx = psp.state.next_state
        if psp.tics != 0:
            break


def p_activate_morph_weapon(sim: GameSim, player: Player) -> None:
    """Port of P_ActivateMorphWeapon (p_pspr.c:325)."""
    player.pendingweapon = WP_NOCHANGE
    player.psprites[ps_weapon].sy = WEAPONTOP
    player.readyweapon = WP_FIRST
    p_set_psprite(sim, player, ps_weapon, StateNum.S_SNOUTREADY)


def p_post_morph_weapon(
    sim: GameSim,
    player: Player,
    weapon: int | WeaponType,
) -> None:
    """Port of P_PostMorphWeapon (p_pspr.c:340)."""
    wp = int(weapon)
    pclass = player.player_class & 3
    player.pendingweapon = WP_NOCHANGE
    player.readyweapon = wp
    player.psprites[ps_weapon].sy = WEAPONBOTTOM
    p_set_psprite(sim, player, ps_weapon, WEAPON_INFO[wp][pclass].upstate)


def p_bring_up_weapon(sim: GameSim, player: Player) -> None:
    """Port of P_BringUpWeapon (p_pspr.c:357).

    Starts bringing the pending weapon up from the bottom of the screen.
    """
    if player.pendingweapon == WP_NOCHANGE:
        player.pendingweapon = player.readyweapon

    pclass = player.player_class & 3
    wp = player.pendingweapon & 3
    if (
        pclass == 0  # PCLASS_FIGHTER
        and wp == WP_SECOND
        and player.mana[MANA_1] > 0
    ):
        new_state = StateNum.S_FAXEUP_G
    else:
        new_state = WEAPON_INFO[wp][pclass].upstate

    player.pendingweapon = WP_NOCHANGE
    player.psprites[ps_weapon].sy = WEAPONBOTTOM
    p_set_psprite(sim, player, ps_weapon, new_state)


def p_check_mana(sim: GameSim, player: Player) -> bool:
    """Port of P_CheckMana (p_pspr.c:388).

    Returns True if there is enough mana to shoot. If not, selects the
    next weapon to use and lowers the current weapon.
    """
    pclass = player.player_class & 3
    wp = player.readyweapon & 3
    mana = WEAPON_INFO[wp][pclass].mana
    count = WEAPON_MANA_USE[pclass][wp]

    if mana == ManaType.MANA_BOTH:
        if player.mana[MANA_1] >= count and player.mana[MANA_2] >= count:
            return True
    elif mana == ManaType.MANA_NONE or player.mana[int(mana)] >= count:
        return True

    # Out of mana: pick a weapon to change to
    while True:
        if (
            player.weaponowned[WP_THIRD]
            and player.mana[MANA_2] >= WEAPON_MANA_USE[pclass][WP_THIRD]
        ):
            player.pendingweapon = WP_THIRD
        elif (
            player.weaponowned[WP_SECOND]
            and player.mana[MANA_1] >= WEAPON_MANA_USE[pclass][WP_SECOND]
        ):
            player.pendingweapon = WP_SECOND
        elif (
            player.weaponowned[WP_FOURTH]
            and player.mana[MANA_1] >= WEAPON_MANA_USE[pclass][WP_FOURTH]
            and player.mana[MANA_2] >= WEAPON_MANA_USE[pclass][WP_FOURTH]
        ):
            player.pendingweapon = WP_FOURTH
        else:
            player.pendingweapon = WP_FIRST

        if player.pendingweapon != WP_NOCHANGE:
            break

    p_set_psprite(sim, player, ps_weapon, WEAPON_INFO[wp][pclass].downstate)
    return False


def deduct_weapon_mana(player: Player, weapon: int | WeaponType | None = None) -> None:
    """Deduct the mana cost of `weapon` (defaulting to `player.readyweapon`)."""
    pclass = player.player_class & 3
    wp = (player.readyweapon if weapon is None else int(weapon)) & 3
    mana = WEAPON_INFO[wp][pclass].mana
    cost = WEAPON_MANA_USE[pclass][wp]
    if mana == ManaType.MANA_BOTH:
        player.mana[MANA_1] = max(0, player.mana[MANA_1] - cost)
        player.mana[MANA_2] = max(0, player.mana[MANA_2] - cost)
    elif mana in (ManaType.MANA_1, ManaType.MANA_2):
        idx = int(mana)
        player.mana[idx] = max(0, player.mana[idx] - cost)


def p_fire_weapon(sim: GameSim, player: Player) -> None:
    """Port of P_FireWeapon (p_pspr.c:445)."""
    if not p_check_mana(sim, player):
        return

    from pyhexen.sim.player import PSTATE_ATTACK

    pclass = player.player_class & 3
    wp = player.readyweapon & 3
    if player.mo is not None:
        set_mobj_state(sim, player.mo, PSTATE_ATTACK[pclass])

    if pclass == 0 and wp == WP_SECOND and player.mana[MANA_1] > 0:
        # Glowing axe
        attack_state = StateNum.S_FAXEATK_G1
    else:
        attack_state = (
            WEAPON_INFO[wp][pclass].holdatkstate
            if player.refire
            else WEAPON_INFO[wp][pclass].atkstate
        )

    p_set_psprite(sim, player, ps_weapon, attack_state)
    if player.mo is not None:
        noise_alert(sim, player.mo, player.mo)


def p_drop_weapon(sim: GameSim, player: Player) -> None:
    """Port of P_DropWeapon (p_pspr.c:477).

    The player died, so put the weapon away.
    """
    pclass = player.player_class & 3
    wp = player.readyweapon & 3
    p_set_psprite(sim, player, ps_weapon, WEAPON_INFO[wp][pclass].downstate)


def p_setup_psprites(sim: GameSim, player: Player) -> None:
    """Port of P_SetupPsprites (p_pspr.c:2431).

    Called at start of level for each player.
    """
    for i in range(NUMPSPRITES):
        player.psprites[i].state = None
        player.psprites[i].state_num = int(StateNum.S_NULL)
    player.pendingweapon = player.readyweapon
    p_bring_up_weapon(sim, player)


def p_move_psprites(sim: GameSim, player: Player) -> None:
    """Port of P_MovePsprites (p_pspr.c:2453).

    Called every tic by `p_player_think`.
    """
    for i in range(NUMPSPRITES):
        psp = player.psprites[i]
        psp.capture_prev_state()
        if psp.state is not None:
            if psp.tics != -1:
                psp.tics -= 1
                if psp.tics == 0:
                    p_set_psprite(sim, player, i, psp.state.next_state)

    player.psprites[ps_flash].sx = player.psprites[ps_weapon].sx
    player.psprites[ps_flash].sy = player.psprites[ps_weapon].sy


def adjust_player_angle(sim: GameSim, pmo: Mobj) -> None:
    """Port of AdjustPlayerAngle (p_pspr.c:668)."""
    linetarget = sim.map_state.linetarget
    if linetarget is None:
        return
    angle = point_to_angle2(pmo.x, pmo.y, linetarget.x, linetarget.y)
    difference = to_int32(angle - pmo.angle)
    if abs(difference) > MAX_ANGADJUST:
        if difference > 0:
            pmo.angle = to_uint32(pmo.angle + MAX_ANGADJUST)
        else:
            pmo.angle = to_uint32(pmo.angle - MAX_ANGADJUST)
    else:
        pmo.angle = to_uint32(angle)


# =============================================================================
# Mana, Weapon Pickup, and 4th Weapon Piece Assembly (p_inter.c:150-526)
# =============================================================================


def give_mana(
    sim: GameSim,
    player: Player,
    mana: int | ManaType,
    count: int,
) -> bool:
    """Port of P_GiveMana (p_inter.c:150).

    Adds `count` mana of type `mana` (or both if `MANA_BOTH`). Returns True if
    any mana was accepted.
    """
    mtype = int(mana)
    if mtype in (MANA_NONE, NUMMANA):
        return False
    if mtype == MANA_BOTH:
        g1 = give_mana(sim, player, MANA_1, count)
        g2 = give_mana(sim, player, MANA_2, count)
        return g1 or g2
    if mtype not in (MANA_1, MANA_2):
        raise ValueError(f"P_GiveMana: bad type {mtype}")

    if player.mana[mtype] >= MAX_MANA:
        return False

    if sim.skill in (0, 4):  # sk_baby or sk_nightmare
        count += count >> 1

    prev_mana = player.mana[mtype]
    player.mana[mtype] = min(MAX_MANA, player.mana[mtype] + count)

    if (
        (player.player_class & 3) == 0  # PCLASS_FIGHTER
        and player.readyweapon == WP_SECOND
        and mtype == MANA_1
        and prev_mana <= 0
    ):
        p_set_psprite(sim, player, ps_weapon, StateNum.S_FAXEREADY_G)

    return True


def give_weapon(
    sim: GameSim,
    player: Player,
    weapon_class: int,
    weapon_type: int | WeaponType,
    weapon_mobj: Mobj | None = None,
    message: str = "",
) -> bool:
    """Port of TryPickupWeapon (p_inter.c:191).

    Returns True if the weapon or its mana was picked up.
    """
    wp = int(weapon_type)
    mana_kind = MANA_1 if wp == WP_SECOND else MANA_2
    remove = True

    if player.player_class != weapon_class:
        if sim.netgame and not sim.deathmatch:
            return False
        if not give_mana(sim, player, mana_kind, 25):
            return False
    elif sim.netgame and not sim.deathmatch:
        if player.weaponowned[wp]:
            return False
        player.weaponowned[wp] = True
        give_mana(sim, player, mana_kind, 25)
        player.pendingweapon = wp
        remove = False
    else:
        gave_mana = give_mana(sim, player, mana_kind, 25)
        if player.weaponowned[wp]:
            gave_weapon = False
        else:
            gave_weapon = True
            player.weaponowned[wp] = True
            if wp > player.readyweapon:
                player.pendingweapon = wp
        if not (gave_weapon or gave_mana):
            return False

    if message:
        _set_player_message(sim, player, message)

    if weapon_mobj is not None:
        if weapon_mobj.special:
            execute_line_special(
                sim,
                weapon_mobj.special,
                weapon_mobj.args,
                None,
                player.mo,
                side=0,
            )
            weapon_mobj.special = 0
        if remove and not (sim.deathmatch and not (weapon_mobj.flags2 & MobjFlag2.MF2_DROPPED)):
            remove_mobj(sim, weapon_mobj)

    player.bonuscount += BONUSADD
    _emit_sound(sim, None, SfxEnum.SFX_PICKUP_WEAPON, full_volume=True)
    return True


def give_weapon_piece(
    sim: GameSim,
    player: Player,
    match_class: int,
    piece_value: int,
    piece_mobj: Mobj | None = None,
) -> tuple[bool, bool]:
    """Port of TryPickupWeaponPiece (p_inter.c:408).

    Returns `(picked_up, assembled_fourth_weapon)`. When all three pieces
    (`WPIECE1 | WPIECE2 | WPIECE3 == 7`) are collected for `player.player_class`,
    grants `WP_FOURTH`, sets `player.pendingweapon = WP_FOURTH`, and emits
    `SFX_WEAPON_BUILD`.
    """
    remove = True
    check_assembled = True
    gave_weapon = False

    if player.player_class != match_class:
        if sim.netgame and not sim.deathmatch:
            return False, False
        check_assembled = False
        g1 = give_mana(sim, player, MANA_1, 20)
        g2 = give_mana(sim, player, MANA_2, 20)
        if not (g1 or g2):
            return False, False
    elif sim.netgame and not sim.deathmatch:
        if player.pieces & piece_value:
            return False, False
        if 0 <= piece_value < len(PIECE_VALUE_TRANS):
            piece_value = PIECE_VALUE_TRANS[piece_value]
        give_mana(sim, player, MANA_1, 20)
        give_mana(sim, player, MANA_2, 20)
        remove = False
    else:
        g1 = give_mana(sim, player, MANA_1, 20)
        g2 = give_mana(sim, player, MANA_2, 20)
        if player.pieces & piece_value:
            if not (g1 or g2):
                return False, False
            check_assembled = False

    if piece_mobj is not None:
        if piece_mobj.special:
            execute_line_special(
                sim,
                piece_mobj.special,
                piece_mobj.args,
                None,
                player.mo,
                side=0,
            )
            piece_mobj.special = 0
        if remove and not (sim.deathmatch and not (piece_mobj.flags2 & MobjFlag2.MF2_DROPPED)):
            remove_mobj(sim, piece_mobj)

    player.bonuscount += BONUSADD

    if check_assembled:
        player.pieces |= piece_value
        if (player.pieces & WPIECE_ALL) == WPIECE_ALL:
            gave_weapon = True
            player.weaponowned[WP_FOURTH] = True
            player.pendingweapon = WP_FOURTH

    cls_idx = max(0, min(2, int(match_class)))
    if gave_weapon:
        _set_player_message(sim, player, FOURTH_WEAPON_TEXT[cls_idx])
        _emit_sound(sim, None, SfxEnum.SFX_WEAPON_BUILD, full_volume=True)
    else:
        _set_player_message(sim, player, WEAPON_PIECE_TEXT[cls_idx])
        _emit_sound(sim, None, SfxEnum.SFX_PICKUP_WEAPON, full_volume=True)

    return True, gave_weapon


# =============================================================================
# Core Psprite Action Functions (p_pspr.c:491-714, 2404)
# =============================================================================


def _resolve_player_and_psp(
    arg1: object,
    arg2: object = None,
    arg3: object = None,
) -> tuple[GameSim, Player, PspDef]:
    """Resolve `(sim, player, psp)` whether called from `p_set_psprite` or `set_mobj_state`."""
    sim: GameSim = arg1  # type: ignore[assignment]
    if isinstance(arg2, Mobj):
        player = arg2.player
        assert player is not None
        psp = player.psprites[ps_weapon]
        return sim, player, psp
    player = arg2  # type: ignore[assignment]
    psp = arg3 if isinstance(arg3, PspDef) else player.psprites[ps_weapon]
    return sim, player, psp


@action("A_WeaponReady")
def a_weaponready(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_WeaponReady (p_pspr.c:491)."""
    sim, player, psp_def = _resolve_player_and_psp(sim, player_or_mo, psp)
    from pyhexen.sim.player import PSTATE_ATTACK, PSTATE_ATTACK_END, PSTATE_NORMAL

    pclass = player.player_class & 3
    wp = player.readyweapon & 3

    # Change player from attack state back to normal state
    if player.mo is not None:
        if PSTATE_ATTACK[pclass] <= player.mo.state_num <= PSTATE_ATTACK_END[pclass]:
            set_mobj_state(sim, player.mo, PSTATE_NORMAL[pclass])

    # Put the weapon away if the player has a pending weapon or has died
    if player.pendingweapon != WP_NOCHANGE or player.health <= 0:
        p_set_psprite(sim, player, ps_weapon, WEAPON_INFO[wp][pclass].downstate)
        return

    # Check for fire button
    if player.cmd.buttons & BT_ATTACK:
        player.attackdown = True
        p_fire_weapon(sim, player)
        return
    player.attackdown = False

    if not player.morph_tics:
        # Bob the weapon based on movement speed
        angle = (128 * sim.level_time) & FINEMASK
        psp_def.sx = FRACUNIT + fixed_mul(player.bob, fine_cosine(angle))
        angle &= (FINEANGLES // 2) - 1
        psp_def.sy = WEAPONTOP + fixed_mul(player.bob, fine_sine(angle))


@action("A_ReFire")
def a_refire(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_ReFire (p_pspr.c:541)."""
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    if (
        (player.cmd.buttons & BT_ATTACK)
        and player.pendingweapon == WP_NOCHANGE
        and player.health > 0
    ):
        player.refire += 1
        p_fire_weapon(sim, player)
    else:
        player.refire = 0
        p_check_mana(sim, player)


@action("A_Lower")
def a_lower(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_Lower (p_pspr.c:562)."""
    sim, player, psp_def = _resolve_player_and_psp(sim, player_or_mo, psp)
    from pyhexen.sim.player import PlayerState

    if player.morph_tics:
        psp_def.sy = WEAPONBOTTOM
    else:
        psp_def.sy += LOWERSPEED

    if psp_def.sy < WEAPONBOTTOM:
        return

    if player.player_state == PlayerState.PST_DEAD:
        psp_def.sy = WEAPONBOTTOM
        return

    if player.health <= 0:
        p_set_psprite(sim, player, ps_weapon, StateNum.S_NULL)
        return

    player.readyweapon = player.pendingweapon
    p_bring_up_weapon(sim, player)


@action("A_Raise")
def a_raise(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_Raise (p_pspr.c:596)."""
    sim, player, psp_def = _resolve_player_and_psp(sim, player_or_mo, psp)
    psp_def.sy -= RAISESPEED
    if psp_def.sy > WEAPONTOP:
        return

    psp_def.sy = WEAPONTOP
    pclass = player.player_class & 3
    wp = player.readyweapon & 3
    if pclass == 0 and wp == WP_SECOND and player.mana[MANA_1] > 0:
        p_set_psprite(sim, player, ps_weapon, StateNum.S_FAXEREADY_G)
    else:
        p_set_psprite(sim, player, ps_weapon, WEAPON_INFO[wp][pclass].readystate)


@action("A_SnoutAttack")
def a_snoutattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_SnoutAttack (p_pspr.c:691)."""
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return
    damage = 3 + (sim.rng.p_random() & 3)
    angle = pmo.angle
    slope = aim_line_attack(sim, pmo, angle, MELEERANGE)
    puff = line_attack(sim, pmo, angle, MELEERANGE, slope, damage, puff_type=MobjType.MT_SNOUTPUFF)
    _emit_sound(sim, pmo, int(SfxEnum.SFX_PIG_ACTIVE1) + (sim.rng.p_random() & 1))
    if sim.map_state.linetarget is not None:
        adjust_player_angle(sim, pmo)
        if puff is not None:
            _emit_sound(sim, pmo, SfxEnum.SFX_PIG_ATTACK)


@action("A_Light0")
def a_light0(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_Light0 (p_pspr.c:2404)."""
    _, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    player.extralight = 0


__all__ = [
    "BONUSADD",
    "FOURTH_WEAPON_TEXT",
    "LOWERSPEED",
    "MANA_1",
    "MANA_2",
    "MANA_BOTH",
    "MANA_NONE",
    "MAX_ANGADJUST",
    "MAX_MANA",
    "NUMMANA",
    "NUMPSPRITES",
    "NUMWEAPONS",
    "PIECE_VALUE_TRANS",
    "RAISESPEED",
    "WEAPONBOTTOM",
    "WEAPONTOP",
    "WEAPON_INFO",
    "WEAPON_MANA_USE",
    "WEAPON_PIECE_TEXT",
    "WPIECE1",
    "WPIECE2",
    "WPIECE3",
    "WPIECE_ALL",
    "WP_FIRST",
    "WP_FOURTH",
    "WP_NOCHANGE",
    "WP_SECOND",
    "WP_THIRD",
    "ManaType",
    "PspDef",
    "PsprNum",
    "WeaponInfo",
    "WeaponType",
    "a_light0",
    "a_lower",
    "a_raise",
    "a_refire",
    "a_snoutattack",
    "a_weaponready",
    "adjust_player_angle",
    "deduct_weapon_mana",
    "give_mana",
    "give_weapon",
    "give_weapon_piece",
    "p_activate_morph_weapon",
    "p_bring_up_weapon",
    "p_check_mana",
    "p_drop_weapon",
    "p_fire_weapon",
    "p_move_psprites",
    "p_post_morph_weapon",
    "p_set_psprite",
    "p_set_psprite_nf",
    "p_setup_psprites",
    "ps_flash",
    "ps_weapon",
]
