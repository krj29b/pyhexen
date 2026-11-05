# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""CLI report of implemented vs stubbed state action functions."""

from __future__ import annotations

from pyhexen.sim.actions import get_action_report


def format_action_report() -> str:
    """Build a human-readable report of all unique actions referenced in STATES."""
    report = get_action_report()
    total = len(report)
    implemented = sum(1 for ok in report.values() if ok)
    stubs = total - implemented

    lines = [
        f"Total unique actions: {total}",
        f"Implemented:          {implemented}",
        f"Stubs:                {stubs}",
        "-" * 40,
    ]
    for name, is_impl in report.items():
        status = "IMPLEMENTED" if is_impl else "STUB"
        lines.append(f"{name:<30} {status}")
    return "\n".join(lines)


def main() -> None:
    print(format_action_report())


if __name__ == "__main__":
    main()
