# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Map object XY/Z movement, friction, gravity, and per-tic thinker logic.

Port of P_XYMovement, P_ZMovement, P_FloorBounceMissile, PlayerLandedOnThing, P_MobjThinker,
P_ExplodeMissile and P_ThrustMobj (p_mobj.c), and P_FakeZMovement (p_map.c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.angles import (
    ANG90,
    ANG180,
    ANG270,
    ANGLETOFINESHIFT,
    FINEANGLES,
    FINEMASK,
    approx_distance,
    fine_cosine,
    fine_sine,
)
from pyhexen.core.fixed import FRACUNIT, c_div, fixed_mul, to_int32
from pyhexen.sim.info_data import MOBJINFO, MobjFlag, MobjFlag2, MobjType, SfxEnum, StateNum
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
# (fixed_t)(-0.3 * FRACUNIT) and (-0.7 * FRACUNIT) in P_FloorBounceMissile
BOUNCE_SHARD: int = -19660
BOUNCE_NORMAL: int = -45875

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
    from pyhexen.sim.actions.monsters_common import FLOOR_ICE, get_thing_floor_type
    from pyhexen.sim.player import CF_NOMOMENTUM, PSTATE_NORMAL, PSTATE_RUN

    player = mo.player
    if player is not None and player.cheats & CF_NOMOMENTUM:
        mo.momx = 0  # debug option for no sliding at all
        mo.momy = 0
        return

    if mo.flags & (MobjFlag.MF_MISSILE | MobjFlag.MF_SKULLFLY):
        return  # No friction for missiles ever

    if (
        mo.z > mo.floorz
        and not (mo.flags2 & MobjFlag2.MF2_FLY)
        and not (mo.flags2 & MobjFlag2.MF2_ONMOBJ)
        and mo.type != MobjType.MT_BLASTEFFECT
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

    if (
        -STOPSPEED < mo.momx < STOPSPEED
        and -STOPSPEED < mo.momy < STOPSPEED
        and (player is None or (player.cmd.forwardmove == 0 and player.cmd.sidemove == 0))
    ):
        if player is not None:
            # If in a walking frame, stop moving (unsigned compare: 0..3 frames in)
            pclass = player.player_class
            if 0 <= mo.state_num - PSTATE_RUN[pclass] < 4:
                set_mobj_state(sim, mo, PSTATE_NORMAL[pclass])
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
        elif get_thing_floor_type(mo) == FLOOR_ICE:
            mo.momx = fixed_mul(mo.momx, FRICTION_LOW)
            mo.momy = fixed_mul(mo.momy, FRICTION_LOW)
        else:
            mo.momx = fixed_mul(mo.momx, FRICTION_NORMAL)
            mo.momy = fixed_mul(mo.momy, FRICTION_NORMAL)


def p_floor_bounce_missile(sim: GameSim, mo: Mobj) -> None:
    """Port of P_FloorBounceMissile (p_mobj.c)."""
    from pyhexen.sim.actions.monsters_common import FLOOR_LIQUID, hit_floor

    sorc_types = (
        MobjType.MT_SORCFX1,
        MobjType.MT_SORCBALL1,
        MobjType.MT_SORCBALL2,
        MobjType.MT_SORCBALL3,
    )
    if hit_floor(sim, mo) >= FLOOR_LIQUID and mo.type not in sorc_types:
        remove_mobj(sim, mo)
        return
    if mo.type == MobjType.MT_SORCFX1:
        mo.momz = -mo.momz  # no energy absorbed
    elif MobjType.MT_SGSHARD1 <= mo.type <= MobjType.MT_SGSHARD0:
        mo.momz = fixed_mul(mo.momz, BOUNCE_SHARD)
        if abs(mo.momz) < FRACUNIT // 2:
            set_mobj_state(sim, mo, StateNum.S_NULL)
            return
    else:
        mo.momz = fixed_mul(mo.momz, BOUNCE_NORMAL)
    mo.momx = c_div(2 * mo.momx, 3)
    mo.momy = c_div(2 * mo.momy, 3)
    if mo.info.seesound:
        if mo.type in sorc_types[1:]:
            if not mo.args[0]:
                start_sound(sim, mo, mo.info.seesound)
        else:
            start_sound(sim, mo, mo.info.seesound)
        start_sound(sim, mo, mo.info.seesound)  # vanilla plays it again


def monster_falling_damage(sim: GameSim, mo: Mobj) -> None:
    """Port of P_MonsterFallingDamage (p_mobj.c): a blasted monster's fall always kills it."""
    from pyhexen.sim.inter import damage_mobj

    damage_mobj(sim, mo, None, None, 10000)


def _player_landing(sim: GameSim, mo: Mobj, *, on_thing: bool) -> None:
    """The squat, damage and sounds of a hard landing.

    Shared by P_ZMovement (floor) and PlayerLandedOnThing (p_mobj.c); they differ only
    in the soft-landing sound, which the floor skips on liquids.
    """
    from pyhexen.sim.actions.monsters_common import FLOOR_LIQUID, get_thing_floor_type
    from pyhexen.sim.enemy import noise_alert
    from pyhexen.sim.inter import falling_damage
    from pyhexen.sim.player import PlayerClass

    player = mo.player
    assert player is not None
    player.deltaviewheight = mo.momz >> 3
    if mo.momz < -23 * FRACUNIT:
        falling_damage(sim, player)
        noise_alert(sim, mo, mo)
    elif mo.momz < -GRAVITY * 12 and not player.morph_tics:
        start_sound(sim, mo, int(SfxEnum.SFX_PLAYER_LAND))
        grunt = {
            int(PlayerClass.PCLASS_FIGHTER): SfxEnum.SFX_PLAYER_FIGHTER_GRUNT,
            int(PlayerClass.PCLASS_CLERIC): SfxEnum.SFX_PLAYER_CLERIC_GRUNT,
            int(PlayerClass.PCLASS_MAGE): SfxEnum.SFX_PLAYER_MAGE_GRUNT,
        }.get(int(player.player_class))
        if grunt is not None:
            start_sound(sim, mo, int(grunt))
    elif (on_thing or get_thing_floor_type(mo) < FLOOR_LIQUID) and not player.morph_tics:
        start_sound(sim, mo, int(SfxEnum.SFX_PLAYER_LAND))
    player.centering = True


def player_landed_on_thing(sim: GameSim, mo: Mobj, onmobj: Mobj) -> None:
    """Port of PlayerLandedOnThing (p_mobj.c)."""
    _ = onmobj
    _player_landing(sim, mo, on_thing=True)


def _z_adjust_height(sim: GameSim, mo: Mobj) -> None:
    """The "adjust height" part shared by P_ZMovement (p_mobj.c) and P_FakeZMovement (p_map.c)."""
    mo.z += mo.momz
    if (mo.flags & MobjFlag.MF_FLOAT) and mo.target is not None:
        # float down towards target if too close
        if not (mo.flags & MobjFlag.MF_SKULLFLY) and not (mo.flags & MobjFlag.MF_INFLOAT):
            dist = approx_distance(mo.x - mo.target.x, mo.y - mo.target.y)
            delta = (mo.target.z + (mo.height >> 1)) - mo.z
            if delta < 0 and dist < -(delta * 3):
                mo.z -= FLOATSPEED
            elif delta > 0 and dist < (delta * 3):
                mo.z += FLOATSPEED
    if (
        mo.player is not None
        and (mo.flags2 & MobjFlag2.MF2_FLY)
        and not (mo.z <= mo.floorz)
        and (sim.level_time & 2)
    ):
        mo.z += fine_sine(((FINEANGLES // 20 * sim.level_time) >> 2) & FINEMASK)


def _z_apply_gravity(mo: Mobj) -> None:
    if mo.flags2 & MobjFlag2.MF2_LOGRAV:
        if mo.momz == 0:
            mo.momz = -(GRAVITY >> 3) * 2
        else:
            mo.momz -= GRAVITY >> 3
    elif not (mo.flags & MobjFlag.MF_NOGRAVITY):
        if mo.momz == 0:
            mo.momz = -GRAVITY * 2
        else:
            mo.momz -= GRAVITY


def p_fake_z_movement(sim: GameSim, mo: Mobj) -> None:
    """Port of P_FakeZMovement (p_map.c): P_ZMovement's effect on `z` and `momz` only.

    P_CheckOnmobj uses it to see where the thing would be, then puts the thing back.
    """
    _z_adjust_height(sim, mo)

    # clip movement
    if mo.z <= mo.floorz:  # Hit the floor
        mo.z = mo.floorz
        if mo.momz < 0:
            mo.momz = 0
        if mo.flags & MobjFlag.MF_SKULLFLY:  # The skull slammed into something
            mo.momz = -mo.momz
        if mo.info.crashstate and (mo.flags & MobjFlag.MF_CORPSE):
            return
    else:
        _z_apply_gravity(mo)

    if mo.z + mo.height > mo.ceilingz:  # hit the ceiling
        if mo.momz > 0:
            mo.momz = 0
        mo.z = mo.ceilingz - mo.height
        if mo.flags & MobjFlag.MF_SKULLFLY:  # the skull slammed into something
            mo.momz = -mo.momz


def p_z_movement(sim: GameSim, mo: Mobj) -> None:
    """Advance `mo` vertically, applying gravity, float adjustments, and floor/ceiling contact.

    Port of P_ZMovement (p_mobj.c).
    """
    from pyhexen.sim.actions.monsters_common import hit_floor
    from pyhexen.sim.inter import damage_mobj
    from pyhexen.sim.player import VIEWHEIGHT

    # check for smooth step up
    if mo.player is not None and mo.z < mo.floorz:
        mo.player.viewheight -= mo.floorz - mo.z
        mo.player.deltaviewheight = (VIEWHEIGHT - mo.player.viewheight) >> 3

    _z_adjust_height(sim, mo)

    # clip movement
    if mo.z <= mo.floorz:  # Hit the floor
        if mo.flags & MobjFlag.MF_MISSILE:
            mo.z = mo.floorz
            if mo.flags2 & MobjFlag2.MF2_FLOORBOUNCE:
                p_floor_bounce_missile(sim, mo)
            elif mo.type == MobjType.MT_HOLY_FX:  # The spirit struck the ground
                mo.momz = 0
                hit_floor(sim, mo)
            elif mo.type in (MobjType.MT_MNTRFX2, MobjType.MT_LIGHTNING_FLOOR):
                pass  # Minotaur floor fire can go up steps
            else:
                hit_floor(sim, mo)
                explode_missile(sim, mo)
            return
        if mo.flags & MobjFlag.MF_COUNTKILL:  # Blasted mobj falling
            if mo.momz < -(23 * FRACUNIT):
                monster_falling_damage(sim, mo)
        if mo.z - mo.momz > mo.floorz:  # Spawn splashes, etc.
            hit_floor(sim, mo)
        mo.z = mo.floorz
        if mo.momz < 0:
            if (mo.flags2 & MobjFlag2.MF2_ICEDAMAGE) and mo.momz < -GRAVITY * 8:
                mo.tics = 1
                mo.momx = 0
                mo.momy = 0
                mo.momz = 0
                return
            if mo.player is not None:
                mo.player.jump_tics = 7  # delay any jumping for a short time
                if mo.momz < -GRAVITY * 8 and not (mo.flags2 & MobjFlag2.MF2_FLY):
                    _player_landing(sim, mo, on_thing=False)  # squat down
            elif MobjType.MT_POTTERY1 <= mo.type <= MobjType.MT_POTTERY3:
                damage_mobj(sim, mo, None, None, 25)
            mo.momz = 0
        if mo.flags & MobjFlag.MF_SKULLFLY:  # The skull slammed into something
            mo.momz = -mo.momz
        if (
            mo.info.crashstate
            and (mo.flags & MobjFlag.MF_CORPSE)
            and not (mo.flags2 & MobjFlag2.MF2_ICEDAMAGE)
        ):
            set_mobj_state(sim, mo, mo.info.crashstate)
            return
    else:
        _z_apply_gravity(mo)

    if mo.z + mo.height > mo.ceilingz:  # hit the ceiling
        if mo.momz > 0:
            mo.momz = 0
        mo.z = mo.ceilingz - mo.height
        if mo.flags2 & MobjFlag2.MF2_FLOORBOUNCE:
            # Maybe reverse momentum here for ceiling bounce. Currently won't happen
            if mo.info.seesound:
                start_sound(sim, mo, mo.info.seesound)
            return
        if mo.flags & MobjFlag.MF_SKULLFLY:  # the skull slammed into something
            mo.momz = -mo.momz
        if mo.flags & MobjFlag.MF_MISSILE:
            if mo.type == MobjType.MT_LIGHTNING_CEILING:
                return
            if mo.subsector is not None and mo.subsector.sector.ceiling_pic.startswith("F_SKY"):
                if mo.type == MobjType.MT_BLOODYSKULL:
                    mo.momx = 0
                    mo.momy = 0
                    mo.momz = -FRACUNIT
                elif mo.type == MobjType.MT_HOLY_FX:
                    explode_missile(sim, mo)
                else:
                    remove_mobj(sim, mo)
                return
            explode_missile(sim, mo)
            return


def reset_blasted(mo: Mobj) -> None:
    """Port of ResetBlasted (p_user.c)."""
    mo.flags2 &= ~MobjFlag2.MF2_BLASTED
    if not (mo.flags & MobjFlag.MF_ICECORPSE):
        mo.flags2 &= ~MobjFlag2.MF2_SLIDE


def p_mobj_thinker(sim: GameSim, mobj: Mobj) -> None:
    """Execute one tic of `mobj` simulation (interpolation snapshot, XY/Z movement, state tics).

    Port of P_MobjThinker (p_mobj.c).
    """
    from pyhexen.sim.player import VIEWHEIGHT

    mobj.capture_prev_state()
    st = sim.map_state

    # Handle X and Y momentums
    st.blocking_mobj = None
    if mobj.momx != 0 or mobj.momy != 0 or (mobj.flags & MobjFlag.MF_SKULLFLY):
        p_xy_movement(sim, mobj)
        if mobj.removed:
            return
    elif mobj.flags2 & MobjFlag2.MF2_BLASTED:
        reset_blasted(mobj)  # Reset to not blasted when momentums are gone

    if mobj.flags2 & MobjFlag2.MF2_FLOATBOB:
        # Floating item bobbing motion (special1 is height)
        base_offset = mobj.special1 if isinstance(mobj.special1, int) else 0
        mobj.z = mobj.floorz + base_offset + FLOAT_BOB_OFFSETS[mobj.health & 63]
        # Vanilla keeps the whole counter (health++), only the index is masked.
        mobj.health = to_int32(mobj.health + 1)
    elif (mobj.z != mobj.floorz) or mobj.momz != 0 or st.blocking_mobj is not None:
        # Handle Z momentum and gravity. A thing that blocked the XY move also brings
        # us here, which is how a player ends up standing on a low thing.
        if mobj.flags2 & MobjFlag2.MF2_PASSMOBJ:
            onmo = check_on_mobj(sim, mobj)
            if onmo is None:
                p_z_movement(sim, mobj)
                # Vanilla tests MF2_ONMOBJ in `flags`, where that bit is MF_PICKUP.
                # Players have MF_PICKUP, so this clears the flag for them as intended.
                if mobj.player is not None and (mobj.flags & int(MobjFlag2.MF2_ONMOBJ)):
                    mobj.flags2 &= ~MobjFlag2.MF2_ONMOBJ
            elif mobj.player is not None:
                if mobj.momz < -GRAVITY * 8 and not (mobj.flags2 & MobjFlag2.MF2_FLY):
                    player_landed_on_thing(sim, mobj, onmo)
                top = onmo.z + onmo.height
                if top - mobj.z <= 24 * FRACUNIT:
                    mobj.player.viewheight -= top - mobj.z
                    mobj.player.deltaviewheight = (VIEWHEIGHT - mobj.player.viewheight) >> 3
                    mobj.z = top
                    mobj.flags2 |= MobjFlag2.MF2_ONMOBJ
                    mobj.momz = 0
                else:  # hit the bottom of the blocking mobj
                    mobj.momz = 0
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
