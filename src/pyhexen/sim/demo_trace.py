# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Per-tic demo traces for checking sync against Chocolate Hexen (T15.6).

The reference side is `tools/demo_trace/pyh_trace.c`, a hook called right after
`leveltime++` in P_Ticker. Both sides write the same text format, one line per tic:

    T <leveltime> <prndindex> <x> <y> <z> <angle> <momx> <momy> <momz> <health>
      <mobj count> <mobj hash> <sector hash>   (hashes as 8 hex digits)

The player fields are player 0's mobj; `health` is `player_t.health`. The mobj hash
folds every live mobj (P_MobjThinker or P_BlasterMobjThinker) in thinker order:
`h = h * 1000003 ^ (x + 3y + 7z + 11type + 13health + 17state)` in 32-bit unsigned
arithmetic. The sector hash folds `floor + 3ceiling + 7light` of every sector the
same way. At requested tics every mobj and sector is also dumped, before that tic's T
line:

    M <type> <x> <y> <z> <momx> <momy> <momz> <angle> <health> <state> <tics> <flags>
    S <index> <floor> <ceiling> <light>
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

from pyhexen.sim.mobj import Mobj

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

HASH_MUL = 1000003
MASK32 = 0xFFFFFFFF


@dataclass(frozen=True, slots=True)
class TraceTic:
    """One `T` line: the state after a tic of P_Ticker."""

    level_time: int
    prndindex: int
    x: int
    y: int
    z: int
    angle: int
    momx: int
    momy: int
    momz: int
    health: int
    mobj_count: int
    mobj_hash: int
    sector_hash: int = 0

    def format(self) -> str:
        return (
            f"T {self.level_time} {self.prndindex} {self.x} {self.y} {self.z} {self.angle} "
            f"{self.momx} {self.momy} {self.momz} {self.health} {self.mobj_count} "
            f"{self.mobj_hash:08x} {self.sector_hash:08x}"
        )


@dataclass(frozen=True, slots=True)
class MobjDump:
    """One `M` line: a mobj at a dumped tic."""

    type: int
    x: int
    y: int
    z: int
    momx: int
    momy: int
    momz: int
    angle: int
    health: int
    state: int
    tics: int
    flags: int

    def format(self) -> str:
        return "M " + " ".join(str(getattr(self, f.name)) for f in fields(self))


@dataclass(frozen=True, slots=True)
class SectorDump:
    """One `S` line: a sector at a dumped tic."""

    index: int
    floor: int
    ceiling: int
    light: int

    def format(self) -> str:
        return f"S {self.index} {self.floor} {self.ceiling} {self.light}"


@dataclass(slots=True)
class Trace:
    """A parsed trace: per-tic lines plus the mobj and sector dumps keyed by level time."""

    tics: list[TraceTic]
    dumps: dict[int, list[MobjDump]]
    sector_dumps: dict[int, list[SectorDump]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Divergence:
    """The first tic where two traces differ."""

    index: int  # position in the trace (0 = first tic)
    reference: TraceTic | None
    ours: TraceTic | None
    fields: tuple[str, ...]  # differing field names; empty when one trace is shorter

    def describe(self) -> str:
        if self.reference is None or self.ours is None:
            which = "reference" if self.reference is None else "our"
            return f"tic {self.index + 1}: the {which} trace ended first"
        parts = [
            f"{name} {getattr(self.reference, name)} != {getattr(self.ours, name)}"
            for name in self.fields
        ]
        return f"tic {self.reference.level_time}: " + ", ".join(parts) + " (reference != ours)"


def _live_mobjs(sim: GameSim) -> Iterable[Mobj]:
    """Mobjs in thinker order, skipping removed ones (vanilla: function == -1)."""
    for thinker in sim.thinkers:
        if isinstance(thinker, Mobj) and not thinker.removed:
            yield thinker


def mobj_hash(mobjs: Iterable[Mobj]) -> tuple[int, int]:
    """Return (count, hash) over `mobjs`, matching PYH_TraceTic."""
    h = 0
    count = 0
    for mo in mobjs:
        term = (
            mo.x + 3 * mo.y + 7 * mo.z + 11 * int(mo.type) + 13 * mo.health + 17 * mo.state_num
        ) & MASK32
        h = ((h * HASH_MUL) & MASK32) ^ term
        count += 1
    return count, h


def sector_hash(sim: GameSim) -> int:
    """Hash over every sector's floor, ceiling and light, matching PYH_TraceTic."""
    h = 0
    if sim.level is not None:
        for sec in sim.level.sectors:
            term = (sec.floor_height + 3 * sec.ceiling_height + 7 * sec.light_level) & MASK32
            h = ((h * HASH_MUL) & MASK32) ^ term
    return h


def dump_sectors(sim: GameSim) -> list[SectorDump]:
    """Every sector, as the reference hook dumps them."""
    if sim.level is None:
        return []
    return [
        SectorDump(i, sec.floor_height, sec.ceiling_height, sec.light_level)
        for i, sec in enumerate(sim.level.sectors)
    ]


def trace_tic(sim: GameSim) -> TraceTic:
    """Trace line for the tic `sim` just ran."""
    player = sim.players[0] if sim.players else None
    mo = player.mo if player is not None else None
    count, h = mobj_hash(_live_mobjs(sim))
    return TraceTic(
        level_time=sim.level_time,
        prndindex=sim.rng.prndindex,
        x=mo.x if mo else 0,
        y=mo.y if mo else 0,
        z=mo.z if mo else 0,
        angle=(mo.angle & MASK32) if mo else 0,
        momx=mo.momx if mo else 0,
        momy=mo.momy if mo else 0,
        momz=mo.momz if mo else 0,
        health=player.health if player is not None else 0,
        mobj_count=count,
        mobj_hash=h,
        sector_hash=sector_hash(sim),
    )


def dump_mobjs(sim: GameSim) -> list[MobjDump]:
    """Every live mobj, as the reference hook dumps them."""
    return [
        MobjDump(
            type=int(mo.type),
            x=mo.x,
            y=mo.y,
            z=mo.z,
            momx=mo.momx,
            momy=mo.momy,
            momz=mo.momz,
            angle=mo.angle & MASK32,
            health=mo.health,
            state=mo.state_num,
            tics=mo.tics,
            flags=mo.flags,
        )
        for mo in _live_mobjs(sim)
    ]


def parse_trace(text: str) -> Trace:
    """Parse a trace written by either side."""
    tics: list[TraceTic] = []
    dumps: dict[int, list[MobjDump]] = {}
    sector_dumps: dict[int, list[SectorDump]] = {}
    pending: list[MobjDump] = []
    pending_sectors: list[SectorDump] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            if parts[0] == "T" and len(parts) == 14:
                values = [int(p) for p in parts[1:12]]
                tic = TraceTic(
                    *values, mobj_hash=int(parts[12], 16), sector_hash=int(parts[13], 16)
                )
                tics.append(tic)
                if pending:
                    dumps[tic.level_time] = pending
                    pending = []
                if pending_sectors:
                    sector_dumps[tic.level_time] = pending_sectors
                    pending_sectors = []
            elif parts[0] == "M" and len(parts) == 13:
                pending.append(MobjDump(*(int(p) for p in parts[1:])))
            elif parts[0] == "S" and len(parts) == 5:
                pending_sectors.append(SectorDump(*(int(p) for p in parts[1:])))
            else:
                raise ValueError("unknown line")
        except ValueError as exc:
            raise ValueError(f"bad trace line {line_no}: {line!r}") from exc
    return Trace(tics=tics, dumps=dumps, sector_dumps=sector_dumps)


def first_divergence(reference: Sequence[TraceTic], ours: Sequence[TraceTic]) -> Divergence | None:
    """The first tic where the traces differ, or None if they match."""
    for i, (ref, mine) in enumerate(zip(reference, ours, strict=False)):
        if ref != mine:
            diff = tuple(
                f.name for f in fields(TraceTic) if getattr(ref, f.name) != getattr(mine, f.name)
            )
            return Divergence(index=i, reference=ref, ours=mine, fields=diff)
    if len(reference) != len(ours):
        i = min(len(reference), len(ours))
        return Divergence(
            index=i,
            reference=reference[i] if i < len(reference) else None,
            ours=ours[i] if i < len(ours) else None,
            fields=(),
        )
    return None
