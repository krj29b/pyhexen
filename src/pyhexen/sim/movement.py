# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Map object XY/Z movement, friction, gravity, and per-tic thinker logic.

Port of P_XYMovement, P_ZMovement, P_MobjThinker, P_ExplodeMissile, and P_ThrustMobj (p_mobj.c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.angles import (
    ANG90,
    ANG180,
    ANG270,
    ANGLETOFINESHIFT,
    approx_distance,
    fine_cosine,
    fine_sine,
)
from pyhexen.core.fixed import FRACUNIT, c_div, fixed_mul
from pyhexen.sim.info_data import MOBJINFO, MobjFlag, MobjFlag2, MobjType, SfxEnum
from pyhexen.sim.map import check_on_mobj, slide_move, try_move
from pyhexen.sim.mobj import FLOAT_BOB_OFFSETS, Mobj, remove_mobj, set_mobj_state
from pyhexen.sim.sound import start_sound

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

MAXMOVE: int = 30 * FRACUNIT
GRAVITY: int = FRACUNIT
STOPSPEED: int = 0x1000
ORIG_FRICTION: int = 0xE800
FRICTION_NORMAL: int = 0xE800
FRICTION_LOW: int = 0xF900
FRICTION_FLY: int = 0xEB00
FLOATSPEED: int = 4 * FRACUNIT

WIND_TAB: tuple[int, int, int] = (2048 * 5, 2048 * 10, 2048 * 25)


def thrust_mobj(mo: Mobj, angle: int, move: int) -> None:
    """Add momentum `move` along 32-bit BAM `angle` to `mo`.

    Port of P_ThrustMobj (p_mobj.c).
    """
    fine = (angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT
    mo.momx += fixed_mul(move, fine_cosine(fine))
    mo.momy += fixed_mul(move, fine_sine(fine))


def explode_missile(sim: GameSim, mo: Mobj) -> None:
    """Zero missile momentum, clear `MF_MISSILE`, and enter `deathstate`.

    Port of P_ExplodeMissile (p_mobj.c).
    """
    mo.momx = 0
    mo.momy = 0
    mo.momz = 0
    set_mobj_state(sim, mo, MOBJINFO[mo.type].deathstate)
    mo.flags &= ~MobjFlag.MF_MISSILE
    if mo.type in (MobjType.MT_SORCBALL1, MobjType.MT_SORCBALL2, MobjType.MT_SORCBALL3):
        start_sound(sim, None, int(SfxEnum.SFX_SORCERER_BIGBALLEXPLODE))
    elif mo.type == MobjType.MT_SORCFX1:
        start_sound(sim, None, int(SfxEnum.SFX_SORCERER_HEADSCREAM))
    elif mo.info.deathsound:
        start_sound(sim, mo, int(mo.info.deathsound))


def p_xy_movement(sim: GameSim, mo: Mobj) -> None:
    """Advance `mo` horizontally, handling half-step splitting, wall sliding, and friction.

    Port of P_XYMovement (p_mobj.c).
    """
    if mo.momx == 0 and mo.momy == 0:
        if mo.flags & MobjFlag.MF_SKULLFLY:
            mo.flags &= ~MobjFlag.MF_SKULLFLY
            mo.momx = 0
            mo.momy = 0
            mo.momz = 0
            set_mobj_state(sim, mo, mo.info.seestate)
        return

    if mo.subsector is not None and (mo.flags2 & MobjFlag2.MF2_WINDTHRUST):
        special = mo.subsector.sector.special
        if 40 <= special <= 42:
            thrust_mobj(mo, 0, WIND_TAB[special - 40])
        elif 43 <= special <= 45:
            thrust_mobj(mo, ANG90, WIND_TAB[special - 43])
        elif 46 <= special <= 48:
            thrust_mobj(mo, ANG270, WIND_TAB[special - 46])
        elif 49 <= special <= 51:
            thrust_mobj(mo, ANG180, WIND_TAB[special - 49])

    if mo.momx > MAXMOVE:
        mo.momx = MAXMOVE
    elif mo.momx < -MAXMOVE:
        mo.momx = -MAXMOVE

    if mo.momy > MAXMOVE:
        mo.momy = MAXMOVE
    elif mo.momy < -MAXMOVE:
        mo.momy = -MAXMOVE

    xmove = mo.momx
    ymove = mo.momy

    while True:
        if (
            xmove > MAXMOVE // 2
            or ymove > MAXMOVE // 2
            or xmove < -(MAXMOVE // 2)
            or ymove < -(MAXMOVE // 2)
        ):
            ptryx = mo.x + c_div(xmove, 2)
            ptryy = mo.y + c_div(ymove, 2)
            xmove = c_div(xmove, 2)
            ymove = c_div(ymove, 2)
        else:
            ptryx = mo.x + xmove
            ptryy = mo.y + ymove
            xmove = 0
            ymove = 0

        if not try_move(sim, mo, ptryx, ptryy):
            # Blocked move
            if mo.flags2 & MobjFlag2.MF2_SLIDE:
                if sim.map_state.blocking_mobj is None:
                    slide_move(sim, mo)
                else:
                    if try_move(sim, mo, mo.x, ptryy):
                        mo.momx = 0
                    elif try_move(sim, mo, ptryx, mo.y):
                        mo.momy = 0
                    else:
                        mo.momx = 0
                        mo.momy = 0
            elif mo.flags & MobjFlag.MF_MISSILE:
                ceilingline = sim.map_state.ceilingline
                if (
                    ceilingline is not None
                    and ceilingline.back_sector is not None
                    and ceilingline.back_sector.ceiling_pic.startswith("F_SKY")
                ):
                    if mo.type == MobjType.MT_BLOODSPLATTER:
                        mo.momx = 0
                        mo.momy = 0
                        return
                    remove_mobj(sim, mo)
                    return
                explode_missile(sim, mo)
                return
            else:
                mo.momx = 0
                mo.momy = 0

        if xmove == 0 and ymove == 0:
            break

    if mo.removed:
        return

    # Friction
    if mo.flags & (MobjFlag.MF_MISSILE | MobjFlag.MF_SKULLFLY):
        return  # No friction for missiles ever

    if (
        mo.z > mo.floorz
        and not (mo.flags2 & MobjFlag2.MF2_FLY)
        and not (mo.flags2 & MobjFlag2.MF2_ONMOBJ)
    ):
        return  # No friction when airborne

    if mo.flags & MobjFlag.MF_CORPSE:
        if (
            mo.momx > FRACUNIT // 4
            or mo.momx < -(FRACUNIT // 4)
            or mo.momy > FRACUNIT // 4
            or mo.momy < -(FRACUNIT // 4)
        ):
            if mo.subsector is not None and mo.floorz != mo.subsector.sector.floor_height:
                return

    if -STOPSPEED < mo.momx < STOPSPEED and -STOPSPEED < mo.momy < STOPSPEED:
        mo.momx = 0
        mo.momy = 0
    else:
        if (
            (mo.flags2 & MobjFlag2.MF2_FLY)
            and mo.z > mo.floorz
            and not (mo.flags2 & MobjFlag2.MF2_ONMOBJ)
        ):
            mo.momx = fixed_mul(mo.momx, FRICTION_FLY)
            mo.momy = fixed_mul(mo.momy, FRICTION_FLY)
        elif mo.subsector is not None and mo.subsector.sector.special == 15:
            mo.momx = fixed_mul(mo.momx, FRICTION_LOW)
            mo.momy = fixed_mul(mo.momy, FRICTION_LOW)
        else:
            mo.momx = fixed_mul(mo.momx, ORIG_FRICTION)
            mo.momy = fixed_mul(mo.momy, ORIG_FRICTION)


def p_z_movement(sim: GameSim, mo: Mobj) -> None:
    """Advance `mo` vertically, applying gravity, float adjustments, and floor/ceiling contact.

    Port of P_ZMovement (p_mobj.c).
    """
    # Check for smooth step up
    if mo.player is not None and mo.z < mo.floorz:
        mo.player.viewheight -= mo.floorz - mo.z
        mo.player.deltaviewheight = ((48 * FRACUNIT) - mo.player.viewheight) >> 3

    mo.z += mo.momz

    if (mo.flags & MobjFlag.MF_FLOAT) and mo.target is not None:
        if not (mo.flags & MobjFlag.MF_SKULLFLY) and not (mo.flags & MobjFlag.MF_INFLOAT):
            dist = approx_distance(mo.x - mo.target.x, mo.y - mo.target.y)
            delta = (mo.target.z + (mo.height >> 1)) - mo.z
            if delta < 0 and dist < -(delta * 3):
                mo.z -= FLOATSPEED
            elif delta > 0 and dist < (delta * 3):
                mo.z += FLOATSPEED

    # Clip movement against floor
    if mo.z <= mo.floorz:
        if mo.flags & MobjFlag.MF_MISSILE:
            mo.z = mo.floorz
            if not (mo.flags2 & MobjFlag2.MF2_FLOORBOUNCE) and not (mo.flags & MobjFlag.MF_NOCLIP):
                explode_missile(sim, mo)
                return

        if mo.momz < 0:
            if mo.player is not None and mo.momz < -GRAVITY * 8:
                mo.player.deltaviewheight = mo.momz >> 3
            mo.momz = 0
        mo.z = mo.floorz

        if mo.flags & MobjFlag.MF_SKULLFLY:
            mo.momz = -mo.momz
    elif mo.flags2 & MobjFlag2.MF2_LOGRAV:
        if mo.momz == 0:
            mo.momz = -(GRAVITY >> 3) * 2
        else:
            mo.momz -= GRAVITY >> 3
    elif not (mo.flags & MobjFlag.MF_NOGRAVITY):
        if mo.momz == 0:
            mo.momz = -GRAVITY * 2
        else:
            mo.momz -= GRAVITY

    # Clip movement against ceiling
    if mo.z + mo.height > mo.ceilingz:
        if mo.momz > 0:
            mo.momz = 0
        mo.z = mo.ceilingz - mo.height
        if mo.flags & MobjFlag.MF_SKULLFLY:
            mo.momz = -mo.momz
        if (mo.flags & MobjFlag.MF_MISSILE) and not (mo.flags & MobjFlag.MF_NOCLIP):
            if mo.subsector is not None and mo.subsector.sector.ceiling_pic.startswith("F_SKY"):
                if mo.type == MobjType.MT_BLOODSPLATTER:
                    mo.momx = 0
                    mo.momy = 0
                    return
                remove_mobj(sim, mo)
                return
            explode_missile(sim, mo)
            return


def p_mobj_thinker(sim: GameSim, mobj: Mobj) -> None:
    """Execute one tic of `mobj` simulation (interpolation snapshot, XY/Z movement, state tics).

    Port of P_MobjThinker (p_mobj.c).
    """
    mobj.capture_prev_state()

    # Handle X and Y momentums
    if mobj.momx != 0 or mobj.momy != 0 or (mobj.flags & MobjFlag.MF_SKULLFLY):
        p_xy_movement(sim, mobj)
        if mobj.removed:
            return

    # Handle Z momentum and gravity
    if mobj.flags2 & MobjFlag2.MF2_FLOATBOB:
        base_offset = mobj.special1 if isinstance(mobj.special1, int) else 0
        mobj.z = mobj.floorz + base_offset + FLOAT_BOB_OFFSETS[mobj.health & 63]
        mobj.health = (mobj.health + 1) & 63
    elif (mobj.z != mobj.floorz) or mobj.momz != 0:
        if mobj.flags2 & MobjFlag2.MF2_PASSMOBJ:
            onmo = check_on_mobj(sim, mobj)
            if onmo is None:
                p_z_movement(sim, mobj)
                mobj.flags2 &= ~MobjFlag2.MF2_ONMOBJ
            else:
                if mobj.momz < 0:
                    mobj.momz = 0
                mobj.flags2 |= MobjFlag2.MF2_ONMOBJ
        else:
            p_z_movement(sim, mobj)
        if mobj.removed:
            return

    # Cycle through states, calling action functions at transitions
    if mobj.tics != -1:
        mobj.tics -= 1
        if mobj.tics == 0:
            set_mobj_state(sim, mobj, mobj.state.next_state)


def p_blaster_mobj_thinker(sim: GameSim, mobj: Mobj) -> None:
    """Port of P_BlasterMobjThinker (p_mobj.c:925).

    Advances high-speed missiles (MT_CFLAME_MISSILE, MT_MWAND_MISSILE) in 8
    sub-steps per tic to prevent tunneling through targets or geometry.
    """
    mobj.capture_prev_state()

    # Handle movement
    if mobj.momx != 0 or mobj.momy != 0 or (mobj.z != mobj.floorz) or mobj.momz != 0:
        xfrac = mobj.momx >> 3
        yfrac = mobj.momy >> 3
        zfrac = mobj.momz >> 3
        changexy = (xfrac != 0) or (yfrac != 0)

        for _ in range(8):
            if changexy:
                if not try_move(sim, mobj, mobj.x + xfrac, mobj.y + yfrac):
                    explode_missile(sim, mobj)
                    return
            mobj.z += zfrac
            if mobj.z <= mobj.floorz:
                mobj.z = mobj.floorz
                explode_missile(sim, mobj)
                return
            if mobj.z + mobj.height > mobj.ceilingz:
                mobj.z = mobj.ceilingz - mobj.height
                explode_missile(sim, mobj)
                return

            if changexy:
                if mobj.type == MobjType.MT_MWAND_MISSILE and (sim.rng.p_random() < 128):
                    z = mobj.z - 8 * FRACUNIT
                    if z < mobj.floorz:
                        z = mobj.floorz
                    from pyhexen.sim.mobj import spawn_mobj

                    spawn_mobj(sim, mobj.x, mobj.y, z, MobjType.MT_MWANDSMOKE)
                elif mobj.type != MobjType.MT_MWAND_MISSILE:
                    mobj.special1 = (mobj.special1 if isinstance(mobj.special1, int) else 0) - 1
                    if mobj.special1 <= 0:
                        mobj.special1 = 4
                        z = mobj.z - 12 * FRACUNIT
                        if z < mobj.floorz:
                            z = mobj.floorz
                        from pyhexen.sim.mobj import spawn_mobj

                        mo = spawn_mobj(sim, mobj.x, mobj.y, z, MobjType.MT_CFLAMEFLOOR)
                        if mo is not None:
                            mo.angle = mobj.angle

    # Advance the state
    if mobj.tics != -1:
        mobj.tics -= 1
        while mobj.tics == 0:
            if not set_mobj_state(sim, mobj, mobj.state.next_state):
                return
