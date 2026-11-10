# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""First-run IWAD locator dialog and validation (T14.4).

Provides a graphical file dialog (Tkinter with CLI fallback) to locate
HEXEN.WAD when launching PyHexen without pre-configured IWAD paths.
Validates lump directory markers and persists the selection to config.toml.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pyhexen.app.config import DEFAULT_CONFIG_PATH, load_config, save_config

if TYPE_CHECKING:
    pass


def validate_hexen_iwad(path: Path | str) -> tuple[bool, str]:
    """Validate that `path` is a readable Hexen IWAD containing required lumps.

    Returns (is_valid, reason_or_description).
    """
    file_path = Path(path)
    if not file_path.is_file():
        return False, f"File does not exist: {file_path}"

    try:
        size = file_path.stat().st_size
        if size < 12:
            return False, "File is too small to be a WAD header (<12 bytes)"

        with file_path.open("rb") as f:
            header = f.read(12)
            magic = header[:4]
            if magic not in (b"IWAD", b"PWAD"):
                return False, f"Invalid WAD magic: {magic!r} (expected b'IWAD')"

            import struct

            num_lumps, dir_offset = struct.unpack("<II", header[4:12])
            if dir_offset + num_lumps * 16 > size:
                return False, "Corrupt WAD: directory offset extends beyond file"

            f.seek(dir_offset)
            dir_bytes = f.read(num_lumps * 16)
            lump_names = set()
            for i in range(num_lumps):
                entry = dir_bytes[i * 16 : (i + 1) * 16]
                raw_name = entry[8:16].split(b"\x00")[0].decode("ascii", errors="replace").upper()
                lump_names.add(raw_name)

        # A valid Hexen IWAD must have MAP01, PLAYPAL, and MAPINFO
        if "MAP01" not in lump_names:
            return False, "WAD does not contain MAP01"
        if "PLAYPAL" not in lump_names:
            return False, "WAD does not contain PLAYPAL"
        if "MAPINFO" not in lump_names:
            return False, "WAD does not contain MAPINFO (not a Hexen WAD)"

        return True, "Valid Hexen IWAD"
    except Exception as exc:
        return False, f"Failed to read WAD: {exc}"


def prompt_for_iwad(
    config_path: Path | str | None = None,
    *,
    use_gui: bool = True,
    initial_dir: str | Path | None = None,
) -> Path | None:
    """Prompt the user to locate HEXEN.WAD via graphical dialog or console.

    If a valid IWAD is selected, its path is saved into config.toml and returned.
    Returns None if the user cancels or no valid IWAD is provided.
    """
    selected: str | None = None

    if use_gui:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            chosen = filedialog.askopenfilename(
                title="PyHexen - Locate HEXEN.WAD",
                initialdir=str(initial_dir) if initial_dir else os.getcwd(),
                filetypes=[
                    ("Hexen IWAD", "hexen.wad;HEXEN.WAD;*.wad"),
                    ("WAD Files", "*.wad"),
                    ("All Files", "*.*"),
                ],
            )
            root.destroy()
            if chosen:
                selected = chosen
        except Exception:
            # Fall through if GUI cannot initialize (headless, no DISPLAY, etc.)
            selected = None

    if not selected:
        return None

    path = Path(selected).resolve()
    valid, msg = validate_hexen_iwad(path)
    if not valid:
        show_error_dialog(
            "Invalid Hexen IWAD",
            f"The selected file is not a valid Hexen IWAD:\n{msg}",
        )
        return None

    # Save to config.toml
    cfg_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    config = load_config(cfg_path)
    config.iwad = str(path)
    save_config(config, cfg_path)
    return path


def show_error_dialog(title: str, message: str) -> None:
    """Display an error message via GUI alert box or console stderr."""
    shown_gui = False
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror(title, message)
        root.destroy()
        shown_gui = True
    except Exception:
        pass

    if not shown_gui:
        print(f"\n[{title}]\n{message}\n", file=sys.stderr)
