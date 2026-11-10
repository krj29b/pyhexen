# -*- mode: python ; coding: utf-8 -*-
# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""PyInstaller specification file for PyHexen (T14.4).

Configures a standalone Windows one-folder distribution.
Bundles GLSL shaders and default configuration templates.
Strictly excludes all game data (*.wad, *.WAD).
"""

from pathlib import Path

block_cipher = None

project_root = Path.cwd()
shaders_src = project_root / "src" / "pyhexen" / "render" / "shaders"

datas = [
    (str(shaders_src / "*.vert"), "pyhexen/render/shaders"),
    (str(shaders_src / "*.frag"), "pyhexen/render/shaders"),
    (str(project_root / "config.example.toml"), "."),
]

excludes = [
    "*.wad",
    "*.WAD",
]

hiddenimports = [
    "moderngl",
    "pygame",
    "numpy",
    "tomli_w",
    "pyhexen.sim.actions",
    "pyhexen.sim.actions.monsters_common",
    "pyhexen.sim.actions.monsters_casters",
    "pyhexen.sim.actions.monsters_bosses",
    "pyhexen.sim.actions.weapons_fighter",
    "pyhexen.sim.actions.weapons_cleric",
    "pyhexen.sim.actions.weapons_mage",
    "pyhexen.sim.actions.environment",
    "pyhexen.sim.specials.doors",
    "pyhexen.sim.specials.floors",
    "pyhexen.sim.specials.ceilings",
    "pyhexen.sim.specials.plats",
    "pyhexen.sim.specials.pillars",
    "pyhexen.sim.specials.lights",
    "pyhexen.sim.specials.teleport",
    "pyhexen.sim.specials.switches",
    "pyhexen.sim.specials.sector_specials",
]

a = Analysis(
    ["src/pyhexen/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pyhexen",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="pyhexen",
)
