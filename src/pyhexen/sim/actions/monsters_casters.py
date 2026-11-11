# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.
"""Caster monster actions: Dark Bishop, Reiver (Wraith), and floor raise/sink/dirt helpers.

Ports action functions from Chocolate Doom's `src/hexen/p_enemy.c` and
`src/hexen/a_action.c` for Task T10.5b:
- Dark Bishop (`MT_BISHOP`): `A_BishopAttack`, `A_BishopAttack2`,
  `A_BishopMissileWeave`, `A_BishopMissileSeek`, `A_BishopDecide`,
  `A_BishopDoBlur`, `A_BishopSpawnBlur`, `A_BishopChase`, `A_BishopPuff`,
  `A_BishopPainBlur`
- Reiver (`MT_WRAITH`, `MT_WRAITHB`): `A_WraithInit`, `A_WraithRaiseInit`,
  `A_WraithRaise`, `A_WraithMelee`, `A_WraithMissile`, `A_WraithFX2`,
  `A_WraithFX3`, `A_WraithFX4`, `A_WraithLook`, `A_WraithChase`
- Shared floor raise/sink and dirt helpers: `sink_mobj` (`A_SinkMobj`),
  `raise_mobj` (`A_RaiseMobj`), `spawn_dirt` (`P_SpawnDirt`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.angles import (
    ANG1,
    ANG90,
    ANGLETOFINESHIFT,
    FINEMASK,
    fine_cosine,
    fine_sine,
)
from pyhexen.core.fixed import FRACBITS, FRACUNIT, c_mod, fixed_mul
from pyhexen.sim.actions import action
from pyhexen.sim.actions.monsters_common import FLOAT_BOB_OFFSETS, hitdice
from pyhexen.sim.attacks import seeker_missile, spawn_missile
from pyhexen.sim.enemy import a_chase, a_look, check_melee_range
from pyhexen.sim.info_data import (
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
    StateNum,
)
from pyhexen.sim.inter import damage_mobj
from pyhexen.sim.map import try_move
from pyhexen.sim.mobj import Mobj, set_mobj_state, spawn_mobj
from pyhexen.sim.sound import start_sound

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


def thrust_mobj(mo: Mobj, angle: int, move: int) -> None:
    """Port of P_ThrustMobj (p_mobj.c). Adds momentum along `angle` by `move`."""
    fine = ((angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
    mo.momx += fixed_mul(move, fine_cosine(fine))
    mo.momy += fixed_mul(move, fine_sine(fine))


_DIRT_TYPES: tuple[MobjType, ...] = (
    MobjType.MT_DIRT1,
    MobjType.MT_DIRT2,
    MobjType.MT_DIRT3,
    MobjType.MT_DIRT4,
    MobjType.MT_DIRT5,
    MobjType.MT_DIRT6,
)


def _emit_sound(sim: GameSim, actor: Mobj, sound_id: int | SfxEnum) -> None:
    """Emit a SoundEvent from actor if sound_id is non-zero."""
    sid = int(sound_id)
    if sid:
        start_sound(sim, actor, sid)


# =============================================================================
# Floor Sink / Raise & Dirt Helpers (p_enemy.c, a_action.c)
# =============================================================================


def sink_mobj(actor: Mobj) -> bool:
    """Port of A_SinkMobj (p_enemy.c:3304).

    Incrementally sinks a mobj into the floor by increasing floorclip.
    Returns True once floorclip reaches actor.info.height, False otherwise.
    """
    if actor.floorclip < actor.info.height:
        if actor.type in (MobjType.MT_THRUSTFLOOR_DOWN, MobjType.MT_THRUSTFLOOR_UP):
            actor.floorclip += 6 * FRACUNIT
        else:
            actor.floorclip += FRACUNIT
        return False
    return True


def raise_mobj(actor: Mobj) -> bool:
    """Port of A_RaiseMobj (p_enemy.c:3328).

    Incrementally raises a mobj from the floor by decreasing floorclip.
    Returns True once floorclip reaches 0, False otherwise.
    """
    done = True
    if actor.floorclip > 0:
        if actor.type == MobjType.MT_WRAITHB:
            actor.floorclip -= 2 * FRACUNIT
        elif actor.type in (MobjType.MT_THRUSTFLOOR_DOWN, MobjType.MT_THRUSTFLOOR_UP):
            spec2 = actor.special2 if isinstance(actor.special2, int) else 0
            actor.floorclip -= spec2 * FRACUNIT
        else:
            actor.floorclip -= 2 * FRACUNIT

        if actor.floorclip <= 0:
            actor.floorclip = 0
            done = True
        else:
            done = False
    return done


def spawn_dirt(sim: GameSim, actor: Mobj, radius: int) -> Mobj:
    """Port of P_SpawnDirt (a_action.c:1024).

    Spawns a random dirt particle (MT_DIRT1..MT_DIRT6) at distance `radius`
    around `actor` with random upward velocity.
    """
    angle = (sim.rng.p_random() << 5) & FINEMASK  # <<24 >>19
    x = actor.x + fixed_mul(radius, fine_cosine(angle))
    y = actor.y + fixed_mul(radius, fine_sine(angle))
    z = actor.z + (sim.rng.p_random() << 9) + FRACUNIT
    dtype = _DIRT_TYPES[c_mod(sim.rng.p_random(), 6)]
    mo = spawn_mobj(sim, x, y, z, dtype)
    mo.momz = sim.rng.p_random() << 10
    return mo


# =============================================================================
# Dark Bishop (MT_BISHOP) Actions (p_enemy.c:2616-2829)
# =============================================================================


@action("A_BishopAttack")
def a_bishopattack(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BishopAttack (p_enemy.c:2616)."""
    if actor.target is None:
        return
    _emit_sound(sim, actor, actor.info.attacksound)
    if check_melee_range(sim, actor):
        damage_mobj(sim, actor.target, actor, actor, hitdice(sim, 4))
        return
    actor.special1 = (sim.rng.p_random() & 3) + 5


@action("A_BishopAttack2")
def a_bishopattack2(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BishopAttack2 (p_enemy.c:2638).

    Spawns one homing projectile in a volley of Bishop missiles, decrementing
    special1 until 0 and then returning to S_BISHOP_WALK1.
    """
    count = actor.special1 if isinstance(actor.special1, int) else 0
    if actor.target is None or count == 0:
        actor.special1 = 0
        set_mobj_state(sim, actor, StateNum.S_BISHOP_WALK1)
        return
    mo = spawn_missile(sim, actor, actor.target, MobjType.MT_BISH_FX)
    if mo is not None:
        mo.special1 = actor.target
        mo.special2 = 16  # High word == x/y, Low word == z
    actor.special1 = count - 1


@action("A_BishopMissileWeave")
def a_bishopmissileweave(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BishopMissileWeave (p_enemy.c:2663)."""
    spec2 = actor.special2 if isinstance(actor.special2, int) else 0
    weave_xy = (spec2 >> 16) & 0xFFFF
    weave_z = spec2 & 0xFFFF
    angle = (((actor.angle + ANG90) & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
    new_x = actor.x - fixed_mul(fine_cosine(angle), FLOAT_BOB_OFFSETS[weave_xy & 63] << 1)
    new_y = actor.y - fixed_mul(fine_sine(angle), FLOAT_BOB_OFFSETS[weave_xy & 63] << 1)
    weave_xy = (weave_xy + 2) & 63
    new_x += fixed_mul(fine_cosine(angle), FLOAT_BOB_OFFSETS[weave_xy] << 1)
    new_y += fixed_mul(fine_sine(angle), FLOAT_BOB_OFFSETS[weave_xy] << 1)
    try_move(sim, actor, new_x, new_y)
    actor.z -= FLOAT_BOB_OFFSETS[weave_z & 63]
    weave_z = (weave_z + 2) & 63
    actor.z += FLOAT_BOB_OFFSETS[weave_z]
    actor.special2 = weave_z + (weave_xy << 16)


@action("A_BishopMissileSeek")
def a_bishopmissileseek(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BishopMissileSeek (p_enemy.c:2692)."""
    seeker_missile(actor, (ANG1 * 2) & 0xFFFFFFFF, (ANG1 * 3) & 0xFFFFFFFF, sim)


@action("A_BishopDecide")
def a_bishopdecide(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BishopDecide (p_enemy.c:2703)."""
    if sim.rng.p_random() < 220:
        return
    set_mobj_state(sim, actor, StateNum.S_BISHOP_BLUR1)


@action("A_BishopDoBlur")
def a_bishopdoblur(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BishopDoBlur (p_enemy.c:2721)."""
    actor.special1 = (sim.rng.p_random() & 3) + 3  # Random number of blurs
    if sim.rng.p_random() < 120:
        thrust_mobj(actor, (actor.angle + ANG90) & 0xFFFFFFFF, 11 * FRACUNIT)
    elif sim.rng.p_random() > 125:
        thrust_mobj(actor, (actor.angle - ANG90) & 0xFFFFFFFF, 11 * FRACUNIT)
    else:
        thrust_mobj(actor, actor.angle, 11 * FRACUNIT)
    _emit_sound(sim, actor, SfxEnum.SFX_BISHOP_BLUR)


@action("A_BishopSpawnBlur")
def a_bishopspawnblur(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BishopSpawnBlur (p_enemy.c:2745)."""
    count = (actor.special1 if isinstance(actor.special1, int) else 0) - 1
    actor.special1 = count
    if not count:
        actor.momx = 0
        actor.momy = 0
        if sim.rng.p_random() > 96:
            set_mobj_state(sim, actor, StateNum.S_BISHOP_WALK1)
        else:
            set_mobj_state(sim, actor, StateNum.S_BISHOP_ATK1)
    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_BISHOPBLUR)
    mo.angle = actor.angle


@action("A_BishopChase")
def a_bishopchase(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BishopChase (p_enemy.c:2775)."""
    _ = sim
    spec2 = actor.special2 if isinstance(actor.special2, int) else 0
    actor.z -= FLOAT_BOB_OFFSETS[spec2 & 63] >> 1
    spec2 = (spec2 + 4) & 63
    actor.special2 = spec2
    actor.z += FLOAT_BOB_OFFSETS[spec2] >> 1


@action("A_BishopPuff")
def a_bishoppuff(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BishopPuff (p_enemy.c:2788)."""
    mo = spawn_mobj(sim, actor.x, actor.y, actor.z + 40 * FRACUNIT, MobjType.MT_BISHOP_PUFF)
    mo.momz = FRACUNIT // 2


@action("A_BishopPainBlur")
def a_bishoppainblur(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BishopPainBlur (p_enemy.c:2806)."""
    if sim.rng.p_random() < 64:
        set_mobj_state(sim, actor, StateNum.S_BISHOP_BLUR1)
        return

    r1 = sim.rng.p_subrandom()
    r2 = sim.rng.p_subrandom()
    r3 = sim.rng.p_subrandom()

    mo = spawn_mobj(
        sim,
        actor.x + (r3 << 12),
        actor.y + (r2 << 12),
        actor.z + (r1 << 11),
        MobjType.MT_BISHOPPAINBLUR,
    )
    mo.angle = actor.angle


# =============================================================================
# Reiver (MT_WRAITH, MT_WRAITHB) Actions (p_enemy.c:3373-3551)
# =============================================================================


@action("A_WraithInit")
def a_wraithinit(actor: Mobj, sim: GameSim) -> None:
    """Port of A_WraithInit (p_enemy.c:3373)."""
    _ = sim
    actor.z += 48 << FRACBITS
    actor.special1 = 0  # index into floatbob


@action("A_WraithRaiseInit")
def a_wraithraiseinit(actor: Mobj, sim: GameSim) -> None:
    """Port of A_WraithRaiseInit (p_enemy.c:3379)."""
    _ = sim
    actor.flags2 &= ~MobjFlag2.MF2_DONTDRAW
    actor.flags2 &= ~MobjFlag2.MF2_NONSHOOTABLE
    actor.flags |= MobjFlag.MF_SHOOTABLE | MobjFlag.MF_SOLID
    actor.floorclip = actor.info.height


@action("A_WraithRaise")
def a_wraithraise(actor: Mobj, sim: GameSim) -> None:
    """Port of A_WraithRaise (p_enemy.c:3387)."""
    if raise_mobj(actor):
        set_mobj_state(sim, actor, StateNum.S_WRAITH_CHASE1)
    spawn_dirt(sim, actor, actor.radius)


@action("A_WraithMelee")
def a_wraithmelee(actor: Mobj, sim: GameSim) -> None:
    """Port of A_WraithMelee (p_enemy.c:3399)."""
    if actor.target is None:
        return
    if check_melee_range(sim, actor) and sim.rng.p_random() < 220:
        amount = hitdice(sim, 2)
        damage_mobj(sim, actor.target, actor, actor, amount)
        actor.health += amount


@action("A_WraithMissile")
def a_wraithmissile(actor: Mobj, sim: GameSim) -> None:
    """Port of A_WraithMissile (p_enemy.c:3412)."""
    if actor.target is None:
        return
    mo = spawn_missile(sim, actor, actor.target, MobjType.MT_WRAITHFX1)
    if mo is not None:
        _emit_sound(sim, actor, SfxEnum.SFX_WRAITH_MISSILE_FIRE)


@action("A_WraithFX2")
def a_wraithfx2(actor: Mobj, sim: GameSim) -> None:
    """Port of A_WraithFX2 (p_enemy.c:3428). Spawns sparkle tail of missile."""
    for _ in range(2):
        mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_WRAITHFX2)
        if sim.rng.p_random() < 128:
            angle = (actor.angle + (sim.rng.p_random() << 22)) & 0xFFFFFFFF
        else:
            angle = (actor.angle - (sim.rng.p_random() << 22)) & 0xFFFFFFFF
        fine = (angle >> ANGLETOFINESHIFT) & FINEMASK
        mo.momz = 0
        mo.momx = fixed_mul((sim.rng.p_random() << 7) + FRACUNIT, fine_cosine(fine))
        mo.momy = fixed_mul((sim.rng.p_random() << 7) + FRACUNIT, fine_sine(fine))
        mo.target = actor
        mo.floorclip = 10 * FRACUNIT


@action("A_WraithFX3")
def a_wraithfx3(actor: Mobj, sim: GameSim) -> None:
    """Port of A_WraithFX3 (p_enemy.c:3460). Spawns FX3 droplets during melee attack."""
    numdropped = c_mod(sim.rng.p_random(), 15)
    for _ in range(numdropped):
        mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_WRAITHFX3)
        mo.x += (sim.rng.p_random() - 128) << 11
        mo.y += (sim.rng.p_random() - 128) << 11
        mo.z += sim.rng.p_random() << 10
        mo.target = actor


@action("A_WraithFX4")
def a_wraithfx4(actor: Mobj, sim: GameSim) -> None:
    """Port of A_WraithFX4 (p_enemy.c:3480). Spawns FX4/FX5 droplets during movement."""
    chance = sim.rng.p_random()
    if chance < 10:
        spawn4 = True
        spawn5 = False
    elif chance < 20:
        spawn4 = False
        spawn5 = True
    elif chance < 25:
        spawn4 = True
        spawn5 = True
    else:
        spawn4 = False
        spawn5 = False

    if spawn4:
        mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_WRAITHFX4)
        mo.x += (sim.rng.p_random() - 128) << 12
        mo.y += (sim.rng.p_random() - 128) << 12
        mo.z += sim.rng.p_random() << 10
        mo.target = actor
    if spawn5:
        mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_WRAITHFX5)
        mo.x += (sim.rng.p_random() - 128) << 11
        mo.y += (sim.rng.p_random() - 128) << 11
        mo.z += sim.rng.p_random() << 10
        mo.target = actor


@action("A_WraithLook")
def a_wraithlook(actor: Mobj, sim: GameSim) -> None:
    """Port of A_WraithLook (p_enemy.c:3532)."""
    a_look(actor, sim)


@action("A_WraithChase")
def a_wraithchase(actor: Mobj, sim: GameSim) -> None:
    """Port of A_WraithChase (p_enemy.c:3539)."""
    weaveindex = actor.special1 if isinstance(actor.special1, int) else 0
    actor.z += FLOAT_BOB_OFFSETS[weaveindex & 63]
    actor.special1 = (weaveindex + 2) & 63
    a_chase(actor, sim)
    a_wraithfx4(actor, sim)
