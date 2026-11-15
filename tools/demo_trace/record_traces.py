# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Record Chocolate Hexen reference traces for DEMO1-3 in Docker (T15.6).

Builds `tools/demo_trace/Dockerfile`, compiles Chocolate Hexen from `reference/`
(run `tools/fetch_reference.*` first) with the `pyh_trace.c` hook, and plays the demos
with `-playdemo` under Xvfb. Traces land in `out/demo_traces/` (gitignored: they are
derived from the IWAD's demo lumps). Playback runs in real time, about 3 minutes for
all three demos. The build is cached in `out/demo_traces/.build`.

    uv run python tools/demo_trace/record_traces.py --iwad F:/games/HEXEN.WAD
    uv run python tools/demo_trace/record_traces.py --iwad HEXEN.WAD --dump 1,2 DEMO1

With `--file HEXDD.WAD` the demos are the PWAD's (Deathkings) and the traces go to
`out/demo_traces/hexdd/`. The demo header's player slot count (4 for v1.0, 8 for v1.1
and Deathkings) is read from the demo lumps and passed to the build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from pyhexen.sim.demo import detect_demo_slots
from pyhexen.wad.wadfile import WadStack

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "pyhexen-demo-trace"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iwad", type=Path, default=os.environ.get("PYHEXEN_IWAD"))
    parser.add_argument("--file", type=Path, default=None, help="PWAD to load (HEXDD.WAD)")
    parser.add_argument("--out", type=Path, default=None, help="default out/demo_traces[/pwad]")
    parser.add_argument("--dump", default="", help="comma-separated level times to dump")
    parser.add_argument("demos", nargs="*", default=["DEMO1", "DEMO2", "DEMO3"])
    args = parser.parse_args()
    if args.iwad is None or not Path(args.iwad).is_file():
        parser.error("give --iwad or set PYHEXEN_IWAD")
    reference = ROOT / "reference"
    if not (reference / "src" / "hexen" / "p_tick.c").is_file():
        parser.error("reference/ is missing; run tools/fetch_reference first")

    iwad = Path(args.iwad).resolve()
    pwad = args.file.resolve() if args.file is not None else None
    if pwad is not None and not pwad.is_file():
        parser.error(f"{pwad} not found")
    out = args.out or (ROOT / "out" / "demo_traces")
    if args.out is None and pwad is not None:
        out = out / pwad.stem.lower()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    demos = [d.upper() for d in args.demos]
    with WadStack(iwad) as stack:
        if pwad is not None:
            stack.add_wad(pwad)
        slots = {detect_demo_slots(stack.get_bytes(d)) for d in demos}
    if len(slots) != 1:
        parser.error(f"the demos have different player slot counts {sorted(slots)}")
    pwad_env: list[str] = []
    if pwad is not None:
        pwad_env = ["-e", f"PWAD_NAME={pwad.name}", "-v", f"{pwad.parent}:/pwad:ro"]
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(Path(__file__).parent)], check=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        f"IWAD_NAME={iwad.name}",
        "-e",
        f"PYHEXEN_TRACE_DUMP={args.dump}",
        "-e",
        f"PYHEXEN_MAXPLAYERS={slots.pop()}",
        *pwad_env,
        "-v",
        f"{reference}:/ref:ro",
        "-v",
        f"{iwad.parent}:/wad:ro",
        "-v",
        f"{out}:/out",
        IMAGE,
        *demos,
    ]
    # Git Bash would rewrite the /ref-style container paths otherwise.
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    result = subprocess.run(cmd, env=env).returncode
    # Which IWAD the traces belong to (v1.0 and v1.1 ship different demos).
    source: dict[str, str] = {"iwad_md5": hashlib.md5(iwad.read_bytes()).hexdigest()}
    if pwad is not None:
        source["pwad_md5"] = hashlib.md5(pwad.read_bytes()).hexdigest()
    (out / "source.json").write_text(json.dumps(source) + "\n", encoding="ascii")
    return result


if __name__ == "__main__":
    sys.exit(main())
