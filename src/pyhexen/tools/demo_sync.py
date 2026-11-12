# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Compare PyHexen demo playback with Chocolate Hexen traces, tic by tic (T15.6).

Reference traces come from `tools/demo_trace/record_traces.py` (a Chocolate Hexen build
with a trace hook, run in Docker). They are derived from the IWAD's demo lumps, so they
live in the gitignored `out/demo_traces/` and are never committed.

    uv run python -m pyhexen.tools.demo_sync --iwad HEXEN.WAD DEMO1 DEMO2 DEMO3
    uv run python -m pyhexen.tools.demo_sync --iwad HEXEN.WAD DEMO1 --dump 120
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from difflib import SequenceMatcher
from pathlib import Path

from pyhexen.sim.demo import Demo, build_demo_sim, decode_vanilla_demo
from pyhexen.sim.demo_trace import (
    Divergence,
    MobjDump,
    SectorDump,
    Trace,
    TraceTic,
    dump_mobjs,
    dump_sectors,
    first_divergence,
    parse_trace,
    trace_tic,
)
from pyhexen.wad.mapinfo import MapInfoTable, load_mapinfo
from pyhexen.wad.wadfile import WadStack

DEFAULT_TRACE_DIR = Path("out/demo_traces")


@dataclass(slots=True)
class DemoRun:
    """Our trace of a demo, plus mobj dumps at the requested tics."""

    trace: list[TraceTic]
    dumps: dict[int, list[MobjDump]]
    sector_dumps: dict[int, list[SectorDump]] = field(default_factory=dict)


def run_demo(
    stack: WadStack, mapinfo: MapInfoTable, lump: str, dump_tics: Sequence[int] = ()
) -> DemoRun:
    """Play a vanilla demo lump and trace every tic."""
    demo = Demo.from_vanilla(decode_vanilla_demo(stack.get_bytes(lump)))
    sim = build_demo_sim(stack, mapinfo, demo)
    trace: list[TraceTic] = []
    dumps: dict[int, list[MobjDump]] = {}
    sector_dumps: dict[int, list[SectorDump]] = {}
    wanted = set(dump_tics)
    for cmds in demo.tics:
        sim.tick(cmds)
        if sim.level_time in wanted:
            dumps[sim.level_time] = dump_mobjs(sim)
            sector_dumps[sim.level_time] = dump_sectors(sim)
        trace.append(trace_tic(sim))
    return DemoRun(trace=trace, dumps=dumps, sector_dumps=sector_dumps)


def load_reference(trace_dir: Path, lump: str) -> Trace:
    path = trace_dir / f"{lump}.trace"
    return parse_trace(path.read_text(encoding="ascii"))


def compare_demo(
    stack: WadStack, mapinfo: MapInfoTable, lump: str, trace_dir: Path
) -> tuple[int, Divergence | None]:
    """Return (reference tic count, first divergence or None)."""
    reference = load_reference(trace_dir, lump)
    ours = run_demo(stack, mapinfo, lump)
    return len(reference.tics), first_divergence(reference.tics, ours.trace)


def _print_dump_diff(ref: list[MobjDump], ours: list[MobjDump], limit: int = 40) -> None:
    """Align the two mobj lists by (type, spawn position) and print the differences."""
    names = [f.name for f in fields(MobjDump)]

    def key(m: MobjDump) -> tuple[int, int, int]:
        return (m.type, m.x >> 16, m.y >> 16)

    matcher = SequenceMatcher(a=[key(m) for m in ref], b=[key(m) for m in ours], autojunk=False)
    lines: list[str] = []
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            for i, j in zip(range(a0, a1), range(b0, b1), strict=True):
                a, b = ref[i], ours[j]
                if a != b:
                    diff = ", ".join(
                        f"{n} {getattr(a, n)} != {getattr(b, n)}"
                        for n in names
                        if getattr(a, n) != getattr(b, n)
                    )
                    lines.append(f"  ref #{i} / ours #{j} type {a.type}: {diff}")
            continue
        lines.extend(f"  only in ref  #{i}: {ref[i].format()}" for i in range(a0, a1))
        lines.extend(f"  only in ours #{j}: {ours[j].format()}" for j in range(b0, b1))
    for line in lines[:limit]:
        print(line)
    if len(lines) > limit:
        print(f"  ... {len(lines) - limit} more")
    if not lines:
        print("  (mobj dumps identical)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument(
        "--dump",
        type=int,
        action="append",
        default=[],
        help="compare full mobj dumps at this level time (the reference trace must "
        "have been recorded with PYHEXEN_TRACE_DUMP including it)",
    )
    parser.add_argument("demos", nargs="*", default=["DEMO1", "DEMO2", "DEMO3"])
    args = parser.parse_args(argv)

    stack = WadStack(args.iwad)
    mapinfo = load_mapinfo(stack)
    all_ok = True
    for lump in args.demos:
        lump = lump.upper()
        reference = load_reference(args.traces, lump)
        ours = run_demo(stack, mapinfo, lump, args.dump)
        div = first_divergence(reference.tics, ours.trace)
        if div is None:
            print(f"{lump}: in sync for all {len(reference.tics)} tics")
        else:
            all_ok = False
            print(f"{lump}: diverges at {div.describe()}")
            if div.reference is not None:
                print(f"  ref:  {div.reference.format()}")
            if div.ours is not None:
                print(f"  ours: {div.ours.format()}")
        for tic in args.dump:
            print(f"{lump} mobj dump at tic {tic}:")
            ref_dump = reference.dumps.get(tic)
            if ref_dump is None:
                print("  (no reference dump at that tic)")
                continue
            _print_dump_diff(ref_dump, ours.dumps.get(tic, []))
            ref_sectors = reference.sector_dumps.get(tic, [])
            our_sectors = ours.sector_dumps.get(tic, [])
            sector_diffs = [
                f"  sector {a.index}: (floor, ceiling, light) "
                f"{(a.floor, a.ceiling, a.light)} != {(b.floor, b.ceiling, b.light)}"
                for a, b in zip(ref_sectors, our_sectors, strict=False)
                if a != b
            ]
            for line in sector_diffs[:20]:
                print(line)
            if len(sector_diffs) > 20:
                print(f"  ... {len(sector_diffs) - 20} more sectors")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
