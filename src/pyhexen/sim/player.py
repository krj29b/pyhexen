# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Player state, movement thrust, viewheight/bobbing, jumping, flying, lookdir, and thinker.

Port of P_PlayerThink, P_MovePlayer, P_CalcHeight, P_DeathThink, and P_Thrust (p_user.c).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from pyhexen.core.angles import (
    ANG1,
    ANG45,
    ANG90,
    ANG180,
    ANG_MAX,
    ANGLETOFINESHIFT,
    FINEANGLES,
    FINEMASK,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.fixed import FRACUNIT, abs32, c_div, fixed_mul, to_uint32
from pyhexen.core.ticcmd import (
    AFLAG_JUMP,
    BT_CHANGE,
    BT_SPECIAL,
    BT_USE,
    BT_WEAPONMASK,
    BT_WEAPONSHIFT,
    TOCENTER,
    TicCmd,
    unpack_lookfly,
)
from pyhexen.sim.info_data import MobjFlag, MobjFlag2, MobjType, StateNum
from pyhexen.sim.line_activation import p_use_lines
from pyhexen.sim.mobj import ONFLOORZ, Mobj, set_mobj_state, spawn_mobj
from pyhexen.sim.psprites import (
    NUMPSPRITES,
    WP_FIRST,
    WP_NOCHANGE,
    PspDef,
    p_move_psprites,
    p_setup_psprites,
)
from pyhexen.sim.specials.sector_specials import p_player_in_special_sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

VIEWHEIGHT: int = 48 * FRACUNIT
MAXBOB: int = 0x100000  # 16 * FRACUNIT
MINSPEED: int = 0x1000
MAXHEALTH: int = 100

# Powerup indices (h2def.h)
PW_NONE: int = 0
PW_INVULNERABILITY: int = 1
PW_ALLMAP: int = 2
PW_INFRARED: int = 3
PW_FLIGHT: int = 4
PW_SHIELD: int = 5
PW_HEALTH2: int = 6
PW_SPEED: int = 7
PW_MINOTAUR: int = 8
NUMPOWERS: int = 9

# Cheat flags (p_local.h)
CF_NOCLIP: int = 1
CF_GODMODE: int = 2
CF_NOMOMENTUM: int = 4


class PlayerState(IntEnum):
    """Port of playerstate_t (h2def.h)."""

    PST_LIVE = 0
    PST_DEAD = 1
    PST_REBORN = 2


class PlayerClass(IntEnum):
    """Port of pclass_t (h2def.h)."""

    PCLASS_FIGHTER = 0
    PCLASS_CLERIC = 1
    PCLASS_MAGE = 2
    PCLASS_PIG = 3


PSTATE_NORMAL: tuple[int, ...] = (
    int(StateNum.S_FPLAY),
    int(StateNum.S_CPLAY),
    int(StateNum.S_MPLAY),
    int(StateNum.S_PIGPLAY),
)

PSTATE_RUN: tuple[int, ...] = (
    int(StateNum.S_FPLAY_RUN1),
    int(StateNum.S_CPLAY_RUN1),
    int(StateNum.S_MPLAY_RUN1),
    int(StateNum.S_PIGPLAY_RUN1),
)

PSTATE_ATTACK: tuple[int, ...] = (
    int(StateNum.S_FPLAY_ATK1),
    int(StateNum.S_CPLAY_ATK1),
    int(StateNum.S_MPLAY_ATK1),
    int(StateNum.S_PIGPLAY_ATK1),
)

PSTATE_ATTACK_END: tuple[int, ...] = (
    int(StateNum.S_FPLAY_ATK2),
    int(StateNum.S_CPLAY_ATK3),
    int(StateNum.S_MPLAY_ATK2),
    int(StateNum.S_PIGPLAY_ATK1),
)

PLAYER_MOBJ_TYPES: tuple[int, ...] = (
    int(MobjType.MT_PLAYER_FIGHTER),
    int(MobjType.MT_PLAYER_CLERIC),
    int(MobjType.MT_PLAYER_MAGE),
    int(MobjType.MT_PIGPLAYER),
)


@dataclass(slots=True)
class Player:
    """Player state structure.

    Port of player_t (h2def.h).
    """

    mo: Mobj | None = None
    player_state: PlayerState = PlayerState.PST_LIVE
    player_class: int = int(PlayerClass.PCLASS_FIGHTER)
    cmd: TicCmd = field(default_factory=TicCmd)
    viewz: int = 0
    prev_viewz: int = 0
    viewheight: int = VIEWHEIGHT
    deltaviewheight: int = 0
    bob: int = 0
    health: int = MAXHEALTH
    armor_points: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    keys: int = 0
    mana: list[int] = field(default_factory=lambda: [0, 0])
    readyweapon: int = WP_FIRST
    pendingweapon: int = WP_NOCHANGE
    weaponowned: list[bool] = field(default_factory=lambda: [True, False, False, False])
    psprites: list[PspDef] = field(default_factory=lambda: [PspDef() for _ in range(NUMPSPRITES)])
    pieces: int = 0
    attackdown: bool = False
    usedown: bool = False
    refire: int = 0
    jump_tics: int = 0
    lookdir: int = 0
    centering: bool = False
    flyheight: int = 0
    powers: list[int] = field(default_factory=lambda: [0] * NUMPOWERS)
    message: str = ""
    message_tics: int = 0
    damagecount: int = 0
    bonuscount: int = 0
    poisoncount: int = 0
    extralight: int = 0
    fixedcolormap: int = 0
    attacker: Mobj | None = None
    morph_tics: int = 0
    secret_count: int = 0
    world_timer: int = 0
    cheats: int = 0


def p_thrust(player: Player, angle: int, move: int) -> None:
    """Move `player.mo` along 32-bit BAM `angle` by `move`.

    Port of P_Thrust (p_user.c).
    """
    mo = player.mo
    if mo is None:
        return

    fine = (angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT
    in_flight = bool(player.powers[PW_FLIGHT]) and mo.z > mo.floorz
    is_ice = mo.subsector is not None and (
        mo.subsector.sector.special == 15 or mo.subsector.sector.floor_pic.upper() == "F_033"
    )
    if not in_flight and is_ice:
        move >>= 1

    mo.momx += fixed_mul(move, fine_cosine(fine))
    mo.momy += fixed_mul(move, fine_sine(fine))


def _face_mobj(source: Mobj, target: Mobj) -> tuple[int, int]:
    """Return `(dir, delta)` where `dir == 1` is clockwise, `0` is counter-clockwise.

    Port of P_FaceMobj (p_mobj.c).
    """
    angle1 = source.angle & 0xFFFFFFFF
    angle2 = point_to_angle2(source.x, source.y, target.x, target.y) & 0xFFFFFFFF
    if angle2 > angle1:
        diff = (angle2 - angle1) & 0xFFFFFFFF
        if diff > ANG180:
            return 0, (ANG_MAX - diff) & 0xFFFFFFFF
        return 1, diff
    diff = (angle1 - angle2) & 0xFFFFFFFF
    if diff > ANG180:
        return 1, (ANG_MAX - diff) & 0xFFFFFFFF
    return 0, diff


def p_calc_height(sim: GameSim, player: Player) -> None:
    """Calculate walking/running viewheight and bobbing adjustment.

    Port of P_CalcHeight (p_user.c).
    """
    mo = player.mo
    if mo is None:
        return

    onground = (mo.z <= mo.floorz) or bool(mo.flags2 & MobjFlag2.MF2_ONMOBJ)

    player.bob = (fixed_mul(mo.momx, mo.momx) + fixed_mul(mo.momy, mo.momy)) >> 2
    if player.bob > MAXBOB:
        player.bob = MAXBOB
    if (mo.flags2 & MobjFlag2.MF2_FLY) and not onground:
        player.bob = FRACUNIT // 2

    if player.cheats & CF_NOMOMENTUM:
        player.viewz = mo.z + VIEWHEIGHT
        if player.viewz > mo.ceilingz - 4 * FRACUNIT:
            player.viewz = mo.ceilingz - 4 * FRACUNIT
        player.viewz = mo.z + player.viewheight
        return

    angle = ((FINEANGLES // 20) * sim.level_time) & FINEMASK
    bob = fixed_mul(c_div(player.bob, 2), fine_sine(angle))

    if player.player_state == PlayerState.PST_LIVE:
        player.viewheight += player.deltaviewheight
        if player.viewheight > VIEWHEIGHT:
            player.viewheight = VIEWHEIGHT
            player.deltaviewheight = 0
        if player.viewheight < VIEWHEIGHT // 2:
            player.viewheight = VIEWHEIGHT // 2
            if player.deltaviewheight <= 0:
                player.deltaviewheight = 1

        if player.deltaviewheight:
            player.deltaviewheight += FRACUNIT // 4
            if not player.deltaviewheight:
                player.deltaviewheight = 1

    if player.morph_tics:
        player.viewz = mo.z + player.viewheight - (20 * FRACUNIT)
    else:
        player.viewz = mo.z + player.viewheight + bob

    if mo.floorclip and player.player_state != PlayerState.PST_DEAD and mo.z <= mo.floorz:
        player.viewz -= mo.floorclip

    if player.viewz > mo.ceilingz - 4 * FRACUNIT:
        player.viewz = mo.ceilingz - 4 * FRACUNIT
    if player.viewz < mo.floorz + 4 * FRACUNIT:
        player.viewz = mo.floorz + 4 * FRACUNIT


def p_move_player(sim: GameSim, player: Player) -> None:
    """Apply `player.cmd` turning, thrust, lookdir, flying, and jumping to `player.mo`.

    Port of P_MovePlayer and jump handling in P_PlayerThink (p_user.c).
    """
    mo = player.mo
    if mo is None:
        return

    cmd = player.cmd
    mo.angle = to_uint32(mo.angle + (cmd.angleturn << 16))

    onground = (mo.z <= mo.floorz) or bool(mo.flags2 & MobjFlag2.MF2_ONMOBJ)

    if cmd.forwardmove:
        if onground or (mo.flags2 & MobjFlag2.MF2_FLY):
            p_thrust(player, mo.angle, cmd.forwardmove * 2048)
        else:
            p_thrust(player, mo.angle, FRACUNIT >> 8)

    if cmd.sidemove:
        if onground or (mo.flags2 & MobjFlag2.MF2_FLY):
            p_thrust(player, to_uint32(mo.angle - ANG90), cmd.sidemove * 2048)
        else:
            p_thrust(player, mo.angle, FRACUNIT >> 8)

    pclass = player.player_class & 3
    if (cmd.forwardmove or cmd.sidemove) and mo.state_num == PSTATE_NORMAL[pclass]:
        set_mobj_state(sim, mo, PSTATE_RUN[pclass])

    look, fly = unpack_lookfly(cmd.lookfly)
    if look:
        if look == TOCENTER:
            player.centering = True
        else:
            player.lookdir += 5 * look
            if player.lookdir > 90 or player.lookdir < -110:
                player.lookdir -= 5 * look

    if player.centering:
        if player.lookdir > 0:
            player.lookdir -= 8
        elif player.lookdir < 0:
            player.lookdir += 8
        if abs32(player.lookdir) < 8:
            player.lookdir = 0
            player.centering = False

    if fly and (player.powers[PW_FLIGHT] or (mo.flags2 & MobjFlag2.MF2_FLY)):
        if fly != TOCENTER:
            player.flyheight = fly * 2
            if not (mo.flags2 & MobjFlag2.MF2_FLY):
                mo.flags2 |= MobjFlag2.MF2_FLY
                mo.flags |= MobjFlag.MF_NOGRAVITY
        else:
            mo.flags2 &= ~MobjFlag2.MF2_FLY
            mo.flags &= ~MobjFlag.MF_NOGRAVITY

    if mo.flags2 & MobjFlag2.MF2_FLY:
        mo.momz = player.flyheight * FRACUNIT
        if player.flyheight:
            player.flyheight = c_div(player.flyheight, 2)

    if (cmd.arti & AFLAG_JUMP) and onground and player.jump_tics == 0:
        mo.momz = 6 * FRACUNIT if player.morph_tics else 9 * FRACUNIT
        mo.flags2 &= ~MobjFlag2.MF2_ONMOBJ
        player.jump_tics = 18


def p_death_think(sim: GameSim, player: Player) -> None:
    """Lower viewheight on death, face attacker, and trigger reborn on `BT_USE`.

    Port of P_DeathThink (p_user.c).
    """
    mo = player.mo
    if mo is None:
        return

    p_move_psprites(sim, player)

    onground = mo.z <= mo.floorz
    if mo.type in (MobjType.MT_BLOODYSKULL, MobjType.MT_ICECHUNK):
        player.viewheight = 6 * FRACUNIT
        player.deltaviewheight = 0
        if onground and player.lookdir < 60:
            look_delta = c_div(60 - player.lookdir, 8)
            if look_delta < 1 and (sim.level_time & 1):
                look_delta = 1
            elif look_delta > 6:
                look_delta = 6
            player.lookdir += look_delta
    elif not (mo.flags2 & MobjFlag2.MF2_ICEDAMAGE):
        player.deltaviewheight = 0
        if player.viewheight > 6 * FRACUNIT:
            player.viewheight -= FRACUNIT
        if player.viewheight < 6 * FRACUNIT:
            player.viewheight = 6 * FRACUNIT
        if player.lookdir > 0:
            player.lookdir -= 6
        elif player.lookdir < 0:
            player.lookdir += 6
        if abs32(player.lookdir) < 6:
            player.lookdir = 0

    p_calc_height(sim, player)

    if player.attacker is not None and player.attacker is not mo:
        dir_, delta = _face_mobj(mo, player.attacker)
        if delta < ANG1 * 10:
            if player.damagecount:
                player.damagecount -= 1
            if player.poisoncount:
                player.poisoncount -= 1
        delta = delta // 8
        if delta > ANG1 * 5:
            delta = ANG1 * 5
        if dir_:
            mo.angle = to_uint32(mo.angle + delta)
        else:
            mo.angle = to_uint32(mo.angle - delta)
    elif player.damagecount or player.poisoncount:
        if player.damagecount:
            player.damagecount -= 1
        else:
            player.poisoncount -= 1

    if player.cmd.buttons & BT_USE:
        player.player_state = PlayerState.PST_REBORN
        mo.special1 = player.player_class if player.player_class <= 2 else 0
        mo.special2 = 666


def p_player_think(sim: GameSim, player: Player) -> None:
    """Execute one tic of player logic.

    Port of P_PlayerThink (p_user.c).
    """
    mo = player.mo
    if mo is None:
        return

    player.prev_viewz = player.viewz

    if player.cheats & CF_NOCLIP:
        mo.flags |= MobjFlag.MF_NOCLIP
    else:
        mo.flags &= ~MobjFlag.MF_NOCLIP

    cmd = player.cmd
    if mo.flags & MobjFlag.MF_JUSTATTACKED:
        cmd.angleturn = 0
        cmd.forwardmove = 0xC800 // 512
        cmd.sidemove = 0
        mo.flags &= ~MobjFlag.MF_JUSTATTACKED

    player.message_tics -= 1
    player.world_timer += 1

    if player.player_state == PlayerState.PST_DEAD:
        p_death_think(sim, player)
        return

    if player.jump_tics > 0:
        player.jump_tics -= 1

    if mo.reactiontime > 0:
        mo.reactiontime -= 1
    else:
        p_move_player(sim, player)

    p_calc_height(sim, player)

    if mo.subsector is not None and (
        mo.subsector.sector.special or mo.subsector.sector.floor_pic.upper() == "X_001"
    ):
        p_player_in_special_sector(sim, player)

    if cmd.buttons & BT_SPECIAL:
        cmd.buttons = 0

    if (cmd.buttons & BT_CHANGE) and not player.morph_tics:
        newweapon = (cmd.buttons & BT_WEAPONMASK) >> BT_WEAPONSHIFT
        if (
            0 <= newweapon < len(player.weaponowned)
            and player.weaponowned[newweapon]
            and newweapon != player.readyweapon
        ):
            player.pendingweapon = newweapon

    if cmd.buttons & BT_USE:
        if not player.usedown:
            p_use_lines(sim, player)
            player.usedown = True
    else:
        player.usedown = False

    p_move_psprites(sim, player)

    if player.damagecount > 0:
        player.damagecount -= 1

    if player.bonuscount > 0:
        player.bonuscount -= 1


def spawn_player(
    sim: GameSim,
    player_num: int = 0,
    start_spot: int = 0,
    *,
    x: int | None = None,
    y: int | None = None,
    angle: int = 0,
) -> Player:
    """Spawn a `Player` and its underlying `Mobj` into `sim`.

    Port of P_SpawnPlayer (p_mobj.c).
    """
    mthing = sim.player_starts.get((player_num, start_spot)) or sim.player_starts.get(
        (player_num, 0)
    )
    if x is None or y is None:
        if mthing is not None:
            spawn_x = mthing.x << 16
            spawn_y = mthing.y << 16
            angle = to_uint32(ANG45 * c_div(mthing.angle, 45))
        else:
            spawn_x = 0
            spawn_y = 0
    else:
        spawn_x = x
        spawn_y = y

    pclass = sim.player_class & 3
    mtype = PLAYER_MOBJ_TYPES[pclass]
    mo = spawn_mobj(sim, spawn_x, spawn_y, ONFLOORZ, mtype)
    mo.angle = to_uint32(angle)

    player = Player(
        mo=mo,
        player_state=PlayerState.PST_LIVE,
        player_class=pclass,
        viewz=mo.z + VIEWHEIGHT,
        prev_viewz=mo.z + VIEWHEIGHT,
        viewheight=VIEWHEIGHT,
        health=mo.health,
    )
    mo.player = player
    p_setup_psprites(sim, player)
    mo.capture_prev_state()
    if player not in sim.players:
        sim.players.append(player)
    return player
