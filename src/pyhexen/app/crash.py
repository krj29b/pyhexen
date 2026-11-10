# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Crash reporting and diagnostic log generator (T14.4).

Captures unhandled exceptions and writes a detailed crash.log containing
tracebacks, system environment, Python runtime metadata, and game state.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import platform
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pyhexen import __version__
from pyhexen.app.dialog import show_error_dialog

DEFAULT_CRASH_LOG = Path("crash.log")


def format_crash_report(
    exc: BaseException,
    context_info: dict[str, Any] | None = None,
) -> str:
    """Format an exception and system diagnostic metadata into a crash report string."""
    now_utc = datetime.datetime.now(datetime.UTC).isoformat()
    now_local = datetime.datetime.now().astimezone().isoformat()

    lines: list[str] = [
        "=" * 70,
        "PyHexen Fatal Crash Report",
        "=" * 70,
        f"Time (Local): {now_local}",
        f"Time (UTC):   {now_utc}",
        f"Version:      {__version__}",
        "",
        "--- System & Environment ---",
        f"OS / Platform:    {platform.platform()} ({os.name})",
        f"Architecture:     {platform.machine()}",
        f"Processor:        {platform.processor()}",
        f"Python Version:   {platform.python_version()} ({platform.python_implementation()})",
        f"Python Executable: {sys.executable}",
        f"Working Dir:      {os.getcwd()}",
    ]

    if context_info:
        lines.append("")
        lines.append("--- Game Context ---")
        for k, v in sorted(context_info.items()):
            lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("--- Traceback ---")
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    lines.extend("".join(tb_lines).splitlines())
    lines.append("=" * 70)
    lines.append("")
    return "\n".join(lines)


def write_crash_log(
    exc: BaseException,
    context_info: dict[str, Any] | None = None,
    log_path: Path | str | None = None,
) -> Path:
    """Write a crash report for `exc` to `log_path` (defaulting to crash.log)."""
    target = Path(log_path) if log_path is not None else DEFAULT_CRASH_LOG
    report = format_crash_report(exc, context_info=context_info)

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        f.write(report)

    return target


@contextlib.contextmanager
def crash_guard(
    context_info: dict[str, Any] | None = None,
    *,
    log_path: Path | str | None = None,
    show_dialog: bool = True,
) -> Iterator[None]:
    """Context manager that catches fatal errors and produces a crash.log file."""
    try:
        yield
    except SystemExit:
        raise
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        saved_path = write_crash_log(exc, context_info=context_info, log_path=log_path)
        err_msg = (
            f"PyHexen encountered an unexpected fatal error:\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"A diagnostic crash report has been saved to:\n"
            f"{saved_path.resolve()}\n\n"
            f"Please include this file when reporting bugs."
        )
        if show_dialog:
            show_error_dialog("PyHexen Crash", err_msg)
        else:
            print(f"\n[FATAL ERROR]\n{err_msg}\n", file=sys.stderr)
        raise SystemExit(1) from exc
