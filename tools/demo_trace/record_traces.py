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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "pyhexen-demo-trace"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iwad", type=Path, default=os.environ.get("PYHEXEN_IWAD"))
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "demo_traces")
    parser.add_argument("--dump", default="", help="comma-separated level times to dump")
    parser.add_argument("demos", nargs="*", default=["DEMO1", "DEMO2", "DEMO3"])
    args = parser.parse_args()
    if args.iwad is None or not Path(args.iwad).is_file():
        parser.error("give --iwad or set PYHEXEN_IWAD")
    reference = ROOT / "reference"
    if not (reference / "src" / "hexen" / "p_tick.c").is_file():
        parser.error("reference/ is missing; run tools/fetch_reference first")

    iwad = Path(args.iwad).resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(Path(__file__).parent)], check=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        f"IWAD_NAME={iwad.name}",
        "-e",
        f"PYHEXEN_TRACE_DUMP={args.dump}",
        "-v",
        f"{reference}:/ref:ro",
        "-v",
        f"{iwad.parent}:/wad:ro",
        "-v",
        f"{out}:/out",
        IMAGE,
        *[d.upper() for d in args.demos],
    ]
    # Git Bash would rewrite the /ref-style container paths otherwise.
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    result = subprocess.run(cmd, env=env).returncode
    # Which IWAD the traces belong to (v1.0 and v1.1 ship different demos).
    md5 = hashlib.md5(iwad.read_bytes()).hexdigest()
    (out / "source.json").write_text(json.dumps({"iwad_md5": md5}) + "\n", encoding="ascii")
    return result


if __name__ == "__main__":
    sys.exit(main())
