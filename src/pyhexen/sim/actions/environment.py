# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Environment actor actions and mechanics.

Port of environmental interactions from a_action.c and p_enemy.c:
destructibles (pottery, trees), ambient sound generators, fog/bat spawners,
leaves, thrust floor spikes, earthquakes, swinging bells, and bridges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.angles import (
    ANG1,
    ANG45,
    ANGLETOFINESHIFT,
    approx_distance,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.fixed import FRACBITS, FRACUNIT, abs32, c_div, fixed_mul
from pyhexen.sim.acs.vm import TRANSLATE_THING_TYPE
from pyhexen.sim.actions import action
from pyhexen.sim.attacks import spawn_missile_angle
from pyhexen.sim.info_data import (
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
    StateNum,
)
from pyhexen.sim.inter import damage_mobj
from pyhexen.sim.mobj import (
    FLOAT_BOB_OFFSETS,
    Mobj,
    remove_mobj,
    set_mobj_state,
    set_mobj_state_nf,
    spawn_mobj,
)
from pyhexen.sim.movement import thrust_mobj
from pyhexen.sim.sight import check_sight
from pyhexen.sim.sound import start_sound

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


def _is_sim(obj: object) -> bool:
    return hasattr(obj, "thinkers") and hasattr(obj, "level")


def _start_sound(sim: GameSim, mo: Mobj, sfx_id: int | SfxEnum) -> None:
    sid = int(sfx_id)
    if sid != int(SfxEnum.SFX_NONE) and hasattr(sim, "events"):
        start_sound(sim, mo, sid)


# =============================================================================
# Player Corpse Queue
# =============================================================================

_PLAYER_BODY_QUEUE_SIZE: int = 32
_player_body_queue: list[Mobj | None] = [None] * _PLAYER_BODY_QUEUE_SIZE
_player_body_queue_slot: int = 0


def p_init_player_body_queue() -> None:
    """Reset the player body queue."""
    global _player_body_queue, _player_body_queue_slot
    _player_body_queue = [None] * _PLAYER_BODY_QUEUE_SIZE
    _player_body_queue_slot = 0


@action("A_AddPlayerCorpse")
def a_addplayercorpse(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_AddPlayerCorpse (p_enemy.c:1982). Limits player corpses in world."""
    global _player_body_queue_slot
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if _player_body_queue_slot >= _PLAYER_BODY_QUEUE_SIZE:
        old = _player_body_queue[_player_body_queue_slot % _PLAYER_BODY_QUEUE_SIZE]
        if old is not None and sim is not None:
            remove_mobj(sim, old)
    _player_body_queue[_player_body_queue_slot % _PLAYER_BODY_QUEUE_SIZE] = actor
    _player_body_queue_slot += 1


# =============================================================================
# Pottery Destructibles
# =============================================================================


@action("A_PotteryExplode")
def a_potteryexplode(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_PotteryExplode (a_action.c:146). Shatters pottery into flying bits."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    last_bit: Mobj | None = None
    count = (sim.rng.p_random() & 3) + 3
    for _ in range(count):
        mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_POTTERYBIT1)
        if mo is not None:
            set_mobj_state(sim, mo, mo.info.spawnstate + (sim.rng.p_random() % 5))
            mo.momz = c_div(((sim.rng.p_random() & 7) + 5) * 3 * FRACUNIT, 4)
            mo.momx = sim.rng.p_subrandom() << (FRACBITS - 6)
            mo.momy = sim.rng.p_subrandom() << (FRACBITS - 6)
            last_bit = mo

    if last_bit is not None:
        _start_sound(sim, last_bit, SfxEnum.SFX_POTTERY_EXPLODE)

    arg0 = actor.args[0] if len(actor.args) > 0 else 0
    if arg0 and arg0 < len(TRANSLATE_THING_TYPE):
        spawn_type = TRANSLATE_THING_TYPE[arg0]
        nomonsters = getattr(sim, "nomonsters", False)
        if not nomonsters or not (spawn_type.info.flags & MobjFlag.MF_COUNTKILL):
            spawn_mobj(sim, actor.x, actor.y, actor.z, spawn_type)

    remove_mobj(sim, actor)


@action("A_PotteryChooseBit")
def a_potterychoosebit(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_PotteryChooseBit (a_action.c:179). Picks shard sprite and sets delay."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    set_mobj_state(sim, actor, actor.info.deathstate + (sim.rng.p_random() % 5) + 1)
    actor.tics = 256 + (sim.rng.p_random() << 1)


@action("A_PotteryCheck")
def a_potterycheck(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_PotteryCheck (a_action.c:191). Checks if players are looking at shard."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    for player in getattr(sim, "players", []):
        pmo = player.mo
        if pmo is None:
            continue
        if check_sight(sim, actor, pmo):
            angle_to_actor = point_to_angle2(pmo.x, pmo.y, actor.x, actor.y)
            diff = abs32((angle_to_actor - pmo.angle) & 0xFFFFFFFF)
            if diff <= ANG45 or (0x100000000 - diff) <= ANG45:
                # Revert to previous state (waiting state)
                set_mobj_state(sim, actor, actor.state_num - 1)
                return


# =============================================================================
# Corpse Gibs and Blood Drips
# =============================================================================


@action("A_CorpseBloodDrip")
def a_corpseblooddrip(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_CorpseBloodDrip (a_action.c:236). Spawns falling blood droplet."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    if sim.rng.p_random() > 128:
        return
    spawn_mobj(
        sim,
        actor.x,
        actor.y,
        actor.z + (actor.height >> 1),
        MobjType.MT_CORPSEBLOODDRIP,
    )


@action("A_CorpseExplode")
def a_corpseexplode(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_CorpseExplode (a_action.c:252). Explodes hanging corpse into gibs."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    count = (sim.rng.p_random() & 3) + 3
    for _ in range(count):
        mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_CORPSEBIT)
        if mo is not None:
            set_mobj_state(sim, mo, mo.info.spawnstate + (sim.rng.p_random() % 3))
            mo.momz = c_div(((sim.rng.p_random() & 7) + 5) * 3 * FRACUNIT, 4)
            mo.momx = sim.rng.p_subrandom() << (FRACBITS - 6)
            mo.momy = sim.rng.p_subrandom() << (FRACBITS - 6)

    # Spawn skull
    skull = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_CORPSEBIT)
    if skull is not None:
        set_mobj_state(sim, skull, StateNum.S_CORPSEBIT_4)
        skull.momz = c_div(((sim.rng.p_random() & 7) + 5) * 3 * FRACUNIT, 4)
        skull.momx = sim.rng.p_subrandom() << (FRACBITS - 6)
        skull.momy = sim.rng.p_subrandom() << (FRACBITS - 6)
        _start_sound(sim, skull, SfxEnum.SFX_FIRED_DEATH)

    remove_mobj(sim, actor)


# =============================================================================
# Leaves
# =============================================================================


@action("A_LeafSpawn")
def a_leafspawn(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_LeafSpawn (a_action.c:284). Spawns falling leaves."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    count = (sim.rng.p_random() & 3) + 1
    for _ in range(count):
        ltype = MobjType.MT_LEAF1 if (sim.rng.p_random() & 1) == 0 else MobjType.MT_LEAF2
        z = actor.z + (sim.rng.p_random() << 14)
        y = actor.y + (sim.rng.p_subrandom() << 14)
        x = actor.x + (sim.rng.p_subrandom() << 14)

        mo = spawn_mobj(sim, x, y, z, ltype)
        if mo is not None:
            thrust_mobj(mo, actor.angle, (sim.rng.p_random() << 9) + 3 * FRACUNIT)
            mo.target = actor
            mo.special1 = 0


@action("A_LeafThrust")
def a_leafthrust(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_LeafThrust (a_action.c:315). Adds upward flutter to leaves."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    if sim.rng.p_random() > 96:
        return
    actor.momz += (sim.rng.p_random() << 9) + FRACUNIT


@action("A_LeafCheck")
def a_leafcheck(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_LeafCheck (a_action.c:330). Checks leaf lifespan and motion."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    val = int(actor.special1) if isinstance(actor.special1, int) else 0
    val += 1
    actor.special1 = val
    if val >= 20:
        set_mobj_state(sim, actor, StateNum.S_NULL)
        return

    if sim.rng.p_random() > 64:
        if not actor.momx and not actor.momy and actor.target is not None:
            thrust_mobj(actor, actor.target.angle, (sim.rng.p_random() << 9) + FRACUNIT)
        return

    set_mobj_state(sim, actor, StateNum.S_LEAF1_8)
    actor.momz = (sim.rng.p_random() << 9) + FRACUNIT
    if actor.target is not None:
        thrust_mobj(actor, actor.target.angle, (sim.rng.p_random() << 9) + 2 * FRACUNIT)
    actor.flags |= MobjFlag.MF_MISSILE


# =============================================================================
# Bridges
# =============================================================================

ORBIT_RADIUS: int = 15 * FRACUNIT
ORBIT_TABLE_X: list[int] = [
    fixed_mul(ORBIT_RADIUS, fine_cosine(angle << 5)) for angle in range(256)
]
ORBIT_TABLE_Y: list[int] = [fixed_mul(ORBIT_RADIUS, fine_sine(angle << 5)) for angle in range(256)]


@action("A_BridgeOrbit")
def a_bridgeorbit(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_BridgeOrbit (a_action.c:406). Orbits bridge ball around center."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    if actor.target is not None and bool(actor.target.special1):
        set_mobj_state(sim, actor, StateNum.S_NULL)
        return

    cur_ang = (actor.args[0] + 3) & 255 if len(actor.args) > 0 else 0
    actor.args = (cur_ang, *actor.args[1:])

    if actor.target is not None:
        actor.x = actor.target.x + ORBIT_TABLE_X[cur_ang]
        actor.y = actor.target.y + ORBIT_TABLE_Y[cur_ang]
        actor.z = actor.target.z


@action("A_BridgeInit")
def a_bridgeinit(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_BridgeInit (a_action.c:419). Initializes invisible bridge triad."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    cx = actor.x
    cy = actor.y
    cz = actor.z
    start_angle = sim.rng.p_random()
    actor.special1 = 0

    ball1 = spawn_mobj(sim, cx, cy, cz, MobjType.MT_BRIDGEBALL)
    if ball1 is not None:
        ball1.args = (start_angle, *ball1.args[1:])
        ball1.target = actor

    ball2 = spawn_mobj(sim, cx, cy, cz, MobjType.MT_BRIDGEBALL)
    if ball2 is not None:
        ball2.args = ((start_angle + 85) & 255, *ball2.args[1:])
        ball2.target = actor

    ball3 = spawn_mobj(sim, cx, cy, cz, MobjType.MT_BRIDGEBALL)
    if ball3 is not None:
        ball3.args = ((start_angle + 170) & 255, *ball3.args[1:])
        ball3.target = actor

    if ball1 is not None:
        a_bridgeorbit(ball1, sim)
    if ball2 is not None:
        a_bridgeorbit(ball2, sim)
    if ball3 is not None:
        a_bridgeorbit(ball3, sim)


def a_bridgeremove(actor: Mobj, sim: GameSim) -> None:
    """Port of A_BridgeRemove (a_action.c:451). Removes bridge from world."""
    actor.special1 = 1
    actor.flags &= ~MobjFlag.MF_SOLID
    set_mobj_state(sim, actor, StateNum.S_FREE_BRIDGE1)


# =============================================================================
# Ambient Sounds & Weather Effects
# =============================================================================


@action("A_ESound")
def a_esound(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_ESound (a_action.c:592). Ambient environment sound trigger."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    if actor.type == MobjType.MT_SOUNDWIND:
        _start_sound(sim, actor, SfxEnum.SFX_WIND)


@action("A_FogSpawn")
def a_fogspawn(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_FogSpawn (a_action.c:670). Spawns moving fog patches."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    freq_count = int(actor.special1) if isinstance(actor.special1, int) else 0
    freq_count -= 1
    actor.special1 = freq_count
    if freq_count > 0:
        return

    actor.special1 = actor.args[2] if len(actor.args) > 2 else 0

    roll = sim.rng.p_random() % 3
    if roll == 0:
        ftype = MobjType.MT_FOGPATCHS
    elif roll == 1:
        ftype = MobjType.MT_FOGPATCHM
    else:
        ftype = MobjType.MT_FOGPATCHL

    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, ftype)
    if mo is not None:
        delta = actor.args[1] if (len(actor.args) > 1 and actor.args[1] != 0) else 1
        mo.angle = (
            actor.angle + (((sim.rng.p_random() % delta) - (delta >> 1)) << 24)
        ) & 0xFFFFFFFF
        mo.target = actor
        max_speed = actor.args[0] if (len(actor.args) > 0 and actor.args[0] >= 1) else 1
        spd = (sim.rng.p_random() % max_speed) + 1
        lifetime = actor.args[3] if len(actor.args) > 3 else 0
        mo.args = (spd, mo.args[1], mo.args[2], lifetime, 1)
        mo.special2 = sim.rng.p_random() & 63


@action("A_FogMove")
def a_fogmove(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_FogMove (a_action.c:711). Moves fog patch with wave motion."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    is_moving = actor.args[4] if len(actor.args) > 4 else 0
    if not is_moving:
        return

    # if (actor->args[3]-- <= 0): test the old value; args are bytes.
    life = actor.args[3] if len(actor.args) > 3 else 0
    remaining = (life - 1) & 0xFF
    actor.args = (actor.args[0], actor.args[1], actor.args[2], remaining, actor.args[4])
    if life <= 0:
        set_mobj_state_nf(sim, actor, actor.info.deathstate)
        return

    if (remaining % 4) == 0:
        weave = int(actor.special2) if isinstance(actor.special2, int) else 0
        actor.z += FLOAT_BOB_OFFSETS[weave & 63] >> 1
        actor.special2 = (weave + 1) & 63

    spd = actor.args[0] << FRACBITS if len(actor.args) > 0 else 0
    fine = (actor.angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT
    actor.momx = fixed_mul(spd, fine_cosine(fine))
    actor.momy = fixed_mul(spd, fine_sine(fine))


# =============================================================================
# Earthquake
# =============================================================================


@action("A_Quake")
def a_quake(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_Quake (a_action.c:888). Causes screen tremor and ground thrust."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    richters = actor.args[0] if len(actor.args) > 0 else 0
    duration = (actor.args[1] if len(actor.args) > 1 else 0) - 1
    dmg_rad = actor.args[2] if len(actor.args) > 2 else 0
    tremor_rad = actor.args[3] if len(actor.args) > 3 else 0
    actor.args = (richters, duration, dmg_rad, tremor_rad, actor.args[4])

    if duration > 0:
        for player in getattr(sim, "players", []):
            pmo = player.mo
            if pmo is None or pmo.health <= 0:
                continue
            dist = approx_distance(actor.x - pmo.x, actor.y - pmo.y) >> (FRACBITS + 6)
            if dist < tremor_rad:
                player.quake_intensity = richters
            if dist < dmg_rad and pmo.z <= pmo.floorz:
                if sim.rng.p_random() < 50:
                    damage = (sim.rng.p_random() & 7) + 1
                    damage_mobj(sim, pmo, None, None, damage)
                an = (pmo.angle + ANG1 * sim.rng.p_random()) & 0xFFFFFFFF
                thrust_mobj(pmo, an, richters << (FRACBITS - 1))
    else:
        for player in getattr(sim, "players", []):
            player.quake_intensity = 0
        set_mobj_state(sim, actor, StateNum.S_NULL)


# =============================================================================
# Thrust Floor Spikes
# =============================================================================


def a_sinkmobj(actor: Mobj) -> bool:
    """Sink a mobj incrementally into the floor. Port of A_SinkMobj (p_enemy.c:3304)."""
    if actor.floorclip < actor.info.height:
        if actor.type in (MobjType.MT_THRUSTFLOOR_DOWN, MobjType.MT_THRUSTFLOOR_UP):
            actor.floorclip += 6 * FRACUNIT
        else:
            actor.floorclip += FRACUNIT
        return False
    return True


def a_raisemobj(actor: Mobj) -> bool:
    """Raise a mobj incrementally from floor. Port of A_RaiseMobj (p_enemy.c:3328)."""
    if actor.floorclip > 0:
        if actor.type == MobjType.MT_WRAITHB:
            actor.floorclip -= 2 * FRACUNIT
        elif actor.type in (MobjType.MT_THRUSTFLOOR_DOWN, MobjType.MT_THRUSTFLOOR_UP):
            speed = int(actor.special2) if isinstance(actor.special2, int) else 5
            actor.floorclip -= speed * FRACUNIT
        else:
            actor.floorclip -= 2 * FRACUNIT
        if actor.floorclip <= 0:
            actor.floorclip = 0
            return True
        return False
    return True


def p_spawndirt(sim: GameSim, actor: Mobj, radius: int) -> None:
    """Port of P_SpawnDirt (a_action.c:1024). Spawns dirt chunks around spike."""
    angle = (sim.rng.p_random() << 5) & 0x1FFF
    x = actor.x + fixed_mul(radius, fine_cosine(angle))
    y = actor.y + fixed_mul(radius, fine_sine(angle))
    z = actor.z + (sim.rng.p_random() << 9) + FRACUNIT
    dtype = MobjType(int(MobjType.MT_DIRT1) + (sim.rng.p_random() % 6))
    mo = spawn_mobj(sim, x, y, z, dtype)
    if mo is not None:
        mo.momz = sim.rng.p_random() << 10


def pit_thrustspike(sim: GameSim, actor: Mobj) -> None:
    """Port of PIT_ThrustSpike (p_map.c:222). Impales shootable things in contact."""
    for th in list(sim.thinkers):
        if not isinstance(th, Mobj) or th is actor:
            continue
        if not (th.flags & MobjFlag.MF_SHOOTABLE):
            continue
        blockdist = th.radius + actor.radius
        if abs32(th.x - actor.x) >= blockdist or abs32(th.y - actor.y) >= blockdist:
            continue
        if th.z > actor.z + actor.height:
            continue
        damage_mobj(sim, th, actor, actor, 10001)
        actor.args = (actor.args[0], 1, *actor.args[2:])


@action("A_ThrustInitUp")
def a_thrustinitup(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_ThrustInitUp (a_action.c:1079). Marks spike as raised."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    actor.special2 = 5
    actor.args = (1, *actor.args[1:])
    actor.floorclip = 0
    actor.flags = MobjFlag.MF_SOLID
    actor.flags2 = MobjFlag2.MF2_NOTELEPORT | MobjFlag2.MF2_FLOORCLIP
    actor.special1 = None


@action("A_ThrustInitDn")
def a_thrustinitdn(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_ThrustInitDn (a_action.c:1089). Marks spike as lowered with dirt clump."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    actor.special2 = 5
    actor.args = (0, *actor.args[1:])
    actor.floorclip = actor.info.height
    actor.flags = 0
    actor.flags2 = MobjFlag2.MF2_NOTELEPORT | MobjFlag2.MF2_FLOORCLIP | MobjFlag2.MF2_DONTDRAW
    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_DIRTCLUMP)
    actor.special1 = mo


@action("A_ThrustRaise")
def a_thrustraise(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_ThrustRaise (a_action.c:1102). Raises floor spike towards target height."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    if a_raisemobj(actor):
        actor.args = (1, *actor.args[1:])
        st = StateNum.S_BTHRUSTINIT2_1 if actor.args[1] else StateNum.S_THRUSTINIT2_1
        set_mobj_state(sim, actor, st)

    # Lose the dirt clump
    if actor.floorclip < actor.height and actor.special1 is not None:
        if isinstance(actor.special1, Mobj):
            remove_mobj(sim, actor.special1)
        actor.special1 = None

    if sim.rng.p_random() < 40:
        p_spawndirt(sim, actor, actor.radius)

    speed = int(actor.special2) if isinstance(actor.special2, int) else 5
    actor.special2 = speed + 1


@action("A_ThrustLower")
def a_thrustlower(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_ThrustLower (a_action.c:1126). Lowers floor spike towards ground."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    if a_sinkmobj(actor):
        actor.args = (0, *actor.args[1:])
        st = StateNum.S_BTHRUSTINIT1_1 if actor.args[1] else StateNum.S_THRUSTINIT1_1
        set_mobj_state(sim, actor, st)


@action("A_ThrustBlock")
def a_thrustblock(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_ThrustBlock (a_action.c:1138). Restores solid collision flag."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    actor.flags |= MobjFlag.MF_SOLID


@action("A_ThrustImpale")
def a_thrustimpale(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_ThrustImpale (a_action.c:1143). Impales contacted actors."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    pit_thrustspike(sim, actor)


# =============================================================================
# Bells, Flames, Bats, and Trees
# =============================================================================


@action("A_BellReset1")
def a_bellreset1(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_BellReset1 (a_action.c:1195). Resets bell hanging flags."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    actor.flags |= MobjFlag.MF_NOGRAVITY
    actor.height <<= 2


@action("A_BellReset2")
def a_bellreset2(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_BellReset2 (a_action.c:1207). Resets bell shootability and health."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    actor.flags |= MobjFlag.MF_SHOOTABLE
    actor.flags &= ~MobjFlag.MF_CORPSE
    actor.health = 5


@action("A_FlameCheck")
def a_flamecheck(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_FlameCheck (a_action.c:1221). Flame burn duration countdown."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    val = (actor.args[0] if len(actor.args) > 0 else 0) - 1
    actor.args = (val, *actor.args[1:])
    if val < 0:
        set_mobj_state(sim, actor, StateNum.S_NULL)


@action("A_BatSpawnInit")
def a_batspawninit(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_BatSpawnInit (a_action.c:1245). Initializes bat spawner frequency."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    actor.special1 = 0


@action("A_BatSpawn")
def a_batspawn(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_BatSpawn (a_action.c:1250). Spawns flying bats from spawner."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    count = int(actor.special1) if isinstance(actor.special1, int) else 0
    count -= 1
    actor.special1 = count
    if count > 0:
        return

    actor.special1 = actor.args[0] if len(actor.args) > 0 else 0

    delta = actor.args[1] if (len(actor.args) > 1 and actor.args[1] != 0) else 1
    angle = (actor.angle + (((sim.rng.p_random() % delta) - (delta >> 1)) << 24)) & 0xFFFFFFFF
    mo = spawn_missile_angle(sim, actor, MobjType.MT_BAT, angle, 0)
    if mo is not None:
        duration_octics = actor.args[3] if len(actor.args) > 3 else 0
        turn_degs = actor.args[4] if len(actor.args) > 4 else 0
        mo.args = (sim.rng.p_random() & 63, mo.args[1], mo.args[2], mo.args[3], turn_degs)
        mo.special2 = duration_octics << 3
        mo.target = actor


@action("A_BatMove")
def a_batmove(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_BatMove (a_action.c:1276). Moves bat with random fluttering and squeaks."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    life = int(actor.special2) if isinstance(actor.special2, int) else 0
    if life < 0:
        set_mobj_state(sim, actor, actor.info.deathstate)
        return

    actor.special2 = life - 2

    turn_degs = actor.args[4] if len(actor.args) > 4 else 0
    turn = ANG1 * turn_degs
    if sim.rng.p_random() < 128:
        new_angle = (actor.angle + turn) & 0xFFFFFFFF
    else:
        new_angle = (actor.angle - turn) & 0xFFFFFFFF

    fine = new_angle >> ANGLETOFINESHIFT
    speed = fixed_mul(actor.info.speed, sim.rng.p_random() << 10)
    actor.momx = fixed_mul(speed, fine_cosine(fine))
    actor.momy = fixed_mul(speed, fine_sine(fine))

    if sim.rng.p_random() < 15:
        _start_sound(sim, actor, SfxEnum.SFX_BAT_SCREAM)

    if actor.target is not None:
        bob_idx = actor.args[0] if len(actor.args) > 0 else 0
        actor.z = actor.target.z + 2 * FLOAT_BOB_OFFSETS[bob_idx & 63]
        actor.args = ((bob_idx + 3) & 63, *actor.args[1:])


@action("A_TreeDeath")
def a_treedeath(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_TreeDeath (a_action.c:1316). Tree destruction: chops or burns."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    if not (actor.flags2 & MobjFlag2.MF2_FIREDAMAGE):
        actor.height <<= 2
        actor.flags |= MobjFlag.MF_SHOOTABLE
        actor.flags &= ~(MobjFlag.MF_CORPSE | MobjFlag.MF_DROPOFF)
        actor.health = 35
    else:
        set_mobj_state(sim, actor, actor.info.meleestate)


@action("A_NoGravity")
def a_nogravity(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_NoGravity (a_action.c:1338). Disables gravity for actor."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    actor.flags |= MobjFlag.MF_NOGRAVITY


__all__ = [
    "a_addplayercorpse",
    "a_batmove",
    "a_batspawn",
    "a_batspawninit",
    "a_bellreset1",
    "a_bellreset2",
    "a_bridgeinit",
    "a_bridgeorbit",
    "a_bridgeremove",
    "a_corpseblooddrip",
    "a_corpseexplode",
    "a_esound",
    "a_flamecheck",
    "a_fogmove",
    "a_fogspawn",
    "a_leafcheck",
    "a_leafspawn",
    "a_leafthrust",
    "a_nogravity",
    "a_potterycheck",
    "a_potterychoosebit",
    "a_potteryexplode",
    "a_quake",
    "a_raisemobj",
    "a_sinkmobj",
    "a_thrustblock",
    "a_thrustimpale",
    "a_thrustinitdn",
    "a_thrustinitup",
    "a_thrustlower",
    "a_thrustraise",
    "a_treedeath",
    "p_init_player_body_queue",
    "p_spawndirt",
    "pit_thrustspike",
]
