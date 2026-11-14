# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Run a headless netgame with several `pyhexen` processes on localhost (S2d)."""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

LINE = re.compile(
    r"NETGAME player=(\d+) tic=(\d+) map=(\d+) hash=([0-9a-f]{8}) desync=(-?\d+) state=(\w+)"
)


@dataclass
class NodeResult:
    returncode: int
    player: int
    tic: int
    map_num: int
    digest: str
    desync: int
    output: str


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_netgame(
    iwad: Path,
    players: int,
    tics: int,
    *,
    host_args: tuple[str, ...] = (),
    classes: tuple[str, ...] = ("fighter", "cleric", "mage", "fighter"),
    timeout: float = 240.0,
) -> list[NodeResult]:
    """Start a host and `players - 1` joiners with bot input; return each node's result."""
    port = free_udp_port()
    base = [
        sys.executable,
        "-m",
        "pyhexen",
        "--iwad",
        str(iwad),
        "--nosound",
        "--headless-frames",
        str(tics),
        "--net-bot",
    ]
    cmds = [
        base
        + [
            "--host",
            str(port),
            "--players",
            str(players),
            "--name",
            "N1",
            "--class",
            classes[0],
            *host_args,
        ]
    ]
    for i in range(1, players):
        cmds.append(
            base
            + [
                "--join",
                f"127.0.0.1:{port}",
                "--name",
                f"N{i + 1}",
                "--class",
                classes[i % len(classes)],
            ]
        )
    procs = []
    for i, cmd in enumerate(cmds):
        procs.append(
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        )
        if i == 0:
            time.sleep(1.0)  # let the host open its port first
    results = []
    deadline = time.monotonic() + timeout
    for proc in procs:
        try:
            out, _ = proc.communicate(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        m = LINE.search(out or "")
        if m is None:
            results.append(NodeResult(proc.returncode, -1, -1, -1, "", -1, out or ""))
        else:
            results.append(
                NodeResult(proc.returncode, int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5]), out)
            )
    return results
