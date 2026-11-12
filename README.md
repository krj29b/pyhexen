# PyHexen

A Python reimplementation of the **Hexen: Beyond Heretic** (Raven Software, 1995) engine.
The game logic is a faithful port of the GPL-released source, running as a deterministic
35 Hz simulation in 16.16 fixed point. Drawing goes through a modern OpenGL 3.3 renderer
(pygame-ce + moderngl) that keeps the original's palette-indexed look.

> **PyHexen is a working codename.** The project is not affiliated with or endorsed by
> Raven Software, id Software, Activision or Bethesda. You need your own copy of the game
> data (`HEXEN.WAD`); none is included.

---

## Contents

- [Status](#status)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Getting the game data](#getting-the-game-data)
- [Running](#running)
- [Controls](#controls)
- [Configuration](#configuration)
- [Demos and determinism](#demos-and-determinism)
- [Tools](#tools)
- [Development](#development)
- [Project layout](#project-layout)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Packaging](#packaging)
- [Legal](#legal)

---

## Status

**Playable, still being verified against the original.** You can start a new game from
the title screen, pick a class and skill, play through maps and hubs, save and load, and
use the automap. Sound and music work too.

Verification work in progress:

| Area | State |
|---|---|
| Simulation (monsters, weapons, specials, ACS, polyobjects, hubs) | All 226 action functions ported; being checked against the reference tic by tic |
| Vanilla demo sync (DEMO1–3) | Measured against Chocolate Hexen: in sync for 419 / 2 / 170 tics so far (see [Demos](#demos-and-determinism)) |
| Line specials | 35 of vanilla's 81 do nothing yet, including stairs, some crushers and plats, and Thing_Spawn (task T15.8) |
| Full playthrough per class (Milestone M6) | Not yet done (task T15.7) |
| Animated textures, sky scrolling, lightning on screen | Simulated, but the renderer doesn't show them yet (task T15.6d) |

The live task list, with notes from every session, is in
[docs/PROGRESS.md](docs/PROGRESS.md).

## Features

**Simulation** (`src/pyhexen/sim/`)
- Ported from Chocolate Doom's `src/hexen/` with the original function names kept in
  docstrings (`Port of P_TryMove (p_map.c)`), so the reference is easy to follow.
- 16.16 fixed-point integers only. No floats, no `random` module, no wall clock. Uses the
  original 256-entry `P_Random` table and keeps the original thinker and RNG call order.
- All three classes, their weapons, artifacts, armor and the inventory.
- Monster AI, bosses, environment actors, sector and line specials, doors, floors,
  ceilings, lights, crushers, teleporters, scrollers and earthquakes.
- ACS bytecode VM (`BEHAVIOR` lumps) with world variables and deferred scripts.
- Polyobjects (sliding and swinging doors, rotating and moving walls).
- Hubs and clusters with persistent map state, MAPINFO, sound sequences (SNDSEQ) and
  ANIMDEFS.

**Rendering** (`src/pyhexen/render/`)
- OpenGL 3.3 core. Textures are uploaded as 8-bit palette indices, and a shader applies
  COLORMAP/FOGMAP and PLAYPAL, so light diminishing, fog maps and palette flashes look
  like the original.
- Walls, flats, sprites, player weapon sprites, skies (including double skies) and
  polyobjects. Map geometry is cached in chunks.
- Frames are interpolated between simulation tics, so motion stays smooth above 35 fps.
- Options for FOV, shear or true-pitch look, an authentic low-resolution mode, pixel-aspect
  correction and texture filtering.

**Interface and audio**
- Title loop, menus, skill and class selection, intermission, finale, status bar
  (full-screen toggle), automap and player messages.
- 6 save slots plus quicksave/quickload. Saves are compressed and versioned, and are checked
  against the IWAD they were made with.
- 8-channel sound effects with Hexen's distance and stereo curves, and MUS music played
  through MIDI (pygame, or FluidSynth with a SoundFont).

**Tooling**
- Demo recording and playback, a timedemo benchmark, and a demo-sync harness against
  Chocolate Hexen.
- A WAD inspector and PNG exporter, plus a 2D map viewer.

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependencies and running commands
- A GPU and driver with **OpenGL 3.3**
- **`HEXEN.WAD`** from the full game (v1.0 or v1.1). The shareware demo IWAD may load but
  isn't a target.
- Optional: **FluidSynth** and a General MIDI SoundFont (`.sf2`) for better music
- Optional, for demo-sync work only: **Docker**

Development happens on Windows 11. Nothing in the code is Windows-specific, but Linux and
macOS get less testing.

## Installation

```sh
git clone <this repository> pyhexen
cd pyhexen
uv sync
```

Optional extras:

```sh
uv sync --extra perf    # numba, for faster hot paths
uv sync --extra audio   # pyfluidsynth, for SoundFont music
```

## Getting the game data

PyHexen never ships game data. Use the `HEXEN.WAD` from your own copy of the game (for
example the Steam or GOG release). Tell PyHexen where it is in one of three ways, checked
in this order:

1. The command line: `--iwad C:\Games\Hexen\HEXEN.WAD`
2. The environment variable `PYHEXEN_IWAD`
3. `iwad = "..."` in `config.toml` (see [Configuration](#configuration))

If none of these finds it, PyHexen opens a file dialog so you can pick the IWAD.

## Running

```sh
# Title screen, then start a game from the menu
uv run python -m pyhexen --iwad path/to/HEXEN.WAD

# Jump straight into a map (name or number), with a skill and class
uv run python -m pyhexen --iwad HEXEN.WAD --warp MAP01 --skill 3 --class mage

# Load a PWAD on top of the IWAD
uv run python -m pyhexen --iwad HEXEN.WAD --file mymod.wad --warp 1
```

After `uv sync`, `uv run pyhexen ...` works as well.

| Option | Meaning |
|---|---|
| `--iwad PATH` | Path to `HEXEN.WAD` |
| `--file PWAD [PWAD ...]` | Extra WADs to load on top of the IWAD |
| `--warp MAP` | Start on a map, by name (`MAP01`) or number (`1`). Without it the game opens on the title screen |
| `--skill 1-5` | Skill level |
| `--class fighter\|cleric\|mage` | Player class |
| `--fly` | Free-fly camera, for looking around a map |
| `--nosound` | Turn off sound effects and music |
| `--record PATH` | Record a demo while playing |
| `--playdemo DEMO` | Play a demo file or a demo lump (`DEMO1`) |
| `--timedemo DEMO` | Play a demo as fast as possible, without a window |
| `--headless-frames N` | Run N frames without a visible window, then exit (for automated tests) |
| `--screenshot PATH` | Save a screenshot before exiting |
| `--version` | Print the version |

If the game crashes, the traceback goes to `crash.log` in the working directory.

## Controls

These are the defaults. Mouse sensitivity and inversion are set in `config.toml`.

**Movement and actions**

| Key | Action |
|---|---|
| `W` / `S` | Move forward / back |
| `A` / `D` | Strafe left / right |
| `←` / `→` | Turn |
| Mouse | Turn (X axis), move forward / back (Y axis) |
| `Left Shift` | Run |
| `Left Alt` + turn | Strafe |
| `Left Ctrl` / left mouse button | Attack |
| `E` | Use (doors, switches) |
| `Space` | Jump |
| `1`–`4` | Select a weapon |
| `Page Up` / `Page Down` / `Home` | Look up / down / centre |
| `Insert` / `Delete` / `End` | Fly up / down / stop (with Wings of Wrath) |
| `[` / `]` | Move the inventory cursor |
| `Enter` | Use the selected artifact |

**Interface**

| Key | Action |
|---|---|
| `Esc` | Menu |
| `Tab` | Automap |
| `F1` | Help |
| `F2` / `F3` | Save / load |
| `F4` | Volume |
| `F5` | Suicide |
| `F6` / `F9` | Quicksave / quickload |
| `F7` | End game |
| `F8` | Messages on / off |
| `F10` | Quit |
| `F11` | Full-screen status bar |
| `F12` | Screenshot (saved to `out/`, with a text file describing the view) |

**Automap**

| Key | Action |
|---|---|
| `+` / `-` | Zoom in / out |
| `F` | Follow the player on / off |
| Arrow keys | Pan (when follow is off) |
| `G` / `O` | Grid / transparent overlay (PyHexen extras, not in vanilla) |

**Free-fly mode (`--fly`)**: `W A S D` or the arrow keys to move, `E`/`Space` up,
`C`/`Ctrl` down, `Shift` faster.

## Configuration

Copy `config.example.toml` to `config.toml` (gitignored) and edit it:

```toml
iwad = "C:/Games/Hexen/HEXEN.WAD"

[video]
width = 1280
height = 800
fullscreen = false
vsync = true
fov = 90.0
authentic_aspect = true    # stretch to the original 320x200 pixel aspect
authentic_lowres = false   # render at low resolution for the original look
true_pitch = false         # true perspective look instead of the original y-shear
smooth_filter = false      # bilinear texture filtering

[audio]
sfx_volume = 0.8
music_volume = 0.7
music_backend = "pygame"   # or "fluidsynth"
soundfont = ""             # path to a .sf2 for the fluidsynth backend

[controls]
mouse_sensitivity = 1.0
invert_mouse = false
```

Saves go to `saves/` in the working directory.

## Demos and determinism

The simulation is meant to be exactly deterministic: the same inputs always produce the
same game, bit for bit. Two things depend on that:

- **PyHexen demos** (`--record`, `--playdemo`). Each one stores a hash of the final game
  state, and playback checks it.
- **The original demos** (`DEMO1`–`DEMO3` in the IWAD) should play back exactly as they did
  in 1995. Those demos carry no state hash, so PyHexen checks them against **per-tic traces
  from Chocolate Hexen**, a C port that plays them correctly.

The trace harness is in `tools/demo_trace/`. It builds Chocolate Hexen from the reference
source inside Docker, with a small logging hook. For every tic, the hook writes the
player's position and momentum, the RNG index, and a hash of every map object.

```sh
# 1. Fetch the reference source (once)
./tools/fetch_reference.sh            # or tools\fetch_reference.ps1

# 2. Record reference traces (needs Docker; plays in real time, about 3 minutes)
uv run python tools/demo_trace/record_traces.py --iwad HEXEN.WAD

# 3. Compare PyHexen tic by tic
uv run python -m pyhexen.tools.demo_sync --iwad HEXEN.WAD
#   DEMO1: diverges at tic 420: mobj_hash ... (reference != ours)

# Drill into one tic: record full object dumps there, then diff them
uv run python tools/demo_trace/record_traces.py --iwad HEXEN.WAD --dump 420 DEMO1
uv run python -m pyhexen.tools.demo_sync --iwad HEXEN.WAD --dump 420 DEMO1
```

Traces are derived from the IWAD, so they stay in the gitignored `out/demo_traces/`. The
known divergences are tracked as tasks T15.6a–c in [docs/PROGRESS.md](docs/PROGRESS.md).
`docs/DECISIONS.md` (ADR-011) explains the design.

## Tools

```sh
# Inspect a WAD
uv run python -m pyhexen.tools.wadinfo --iwad HEXEN.WAD list
uv run python -m pyhexen.tools.wadinfo --iwad HEXEN.WAD maps
uv run python -m pyhexen.tools.wadinfo --iwad HEXEN.WAD extract MAPINFO
uv run python -m pyhexen.tools.wadinfo --iwad HEXEN.WAD export-png flat X_005
uv run python -m pyhexen.tools.wadinfo --iwad HEXEN.WAD export-png sprite --all
#   kinds: texture, flat, sprite, patch, palette, colormap; output goes to out/

# 2D map viewer (window, or straight to a PNG)
uv run python -m pyhexen.tools.mapview2d --iwad HEXEN.WAD --warp MAP01
uv run python -m pyhexen.tools.mapview2d --iwad HEXEN.WAD --warp MAP01 --headless-out out/map01.png

# Demo sync (see above)
uv run python -m pyhexen.tools.demo_sync --help
```

Development scripts live in `tools/`: `profile_perf.py` (profiling), `report_actions.py`
(which action functions are ported), `playthrough_audit.py`, and the table/info generators
`gen_tables.py` and `gen_info.py`.

> Exported PNGs and extracted lumps are game data. Keep them out of the repository
> (`out/` is gitignored).

## Development

**Read [AGENTS.md](AGENTS.md) first.** It holds the project rules for human and AI
contributors alike: the session workflow, the determinism and layering rules, and the
legal constraints.

```sh
uv sync                              # install everything, including dev tools
uv run pytest                        # full test suite
uv run pytest -m wad                 # only tests that need a real IWAD
uv run ruff check .                  # lint
uv run ruff format .                 # format
./tools/fetch_reference.sh           # clone the reference source into reference/ (gitignored)
```

**Tests.** CI-safe tests build **synthetic WADs** in code (`tests/fixtures/wadbuilder.py`),
so they never need game data. Tests marked `@pytest.mark.wad` run against your IWAD when one
is configured (`--iwad`, `PYHEXEN_IWAD` or `config.toml`), and skip otherwise. Other markers
are `gl` (needs an OpenGL context) and `slow`. Test folders: `tests/unit`, `tests/render`,
`tests/e2e`, `tests/wad`.

**Ground rules** (full list in AGENTS.md):

- **Never commit game data**: no `.wad` files, extracted assets, PNG exports, or test
  fixtures derived from a real WAD.
- **No floats in `sim/`**. Use `core/fixed.py`, and `c_div` / `c_mod` for C-style
  division. Wrap angles and overflowing integers the way C does.
- **No `random` module and no wall-clock time in the simulation.** Keep the original order
  of thinkers and `P_Random` calls.
- **Layering**: `sim`, `world`, `wad` and `core` never import `render`, `audio`, `ui` or
  `app`. A test enforces this.
- **No per-pixel Python loops at runtime.** Use numpy, numba or shaders.
- The reference source (Chocolate Doom `src/hexen/`) defines correct behaviour.
- Every source file carries the GPL header shown in AGENTS.md.

## Project layout

```
src/pyhexen/
├── __main__.py      CLI entry point and main loop
├── core/            fixed point, angles and trig tables, P_Random, ticcmd, event bus
├── wad/             WAD reading, lumps, palettes, textures, flats, MAPINFO, ANIMDEFS, scanner
├── world/           level structures: map loading, BSP, blockmap, line and sector data
├── sim/             the game simulation (no floats)
│   ├── actions/       ported action functions (weapons, monsters, bosses, environment)
│   ├── acs/           ACS bytecode loader and VM
│   ├── specials/      doors, floors, ceilings, lights, teleports, animated surfaces...
│   ├── game_sim.py    per-level state and the P_Ticker loop
│   ├── map.py         movement and collision (P_TryMove, P_CheckPosition...)
│   ├── demo.py        demo recording and playback
│   └── demo_trace.py  per-tic traces for demo sync
├── render/          OpenGL renderer, shaders, sprites, sky, automap line visibility
├── audio/           sound channels, sound sequences, MUS-to-MIDI and music
├── ui/              menus, status bar, automap, intermission, finale, fonts, messages
├── app/             game flow controller, input, config, save games, crash log
└── tools/           wadinfo, mapview2d, demo_sync
tests/               unit, render, e2e and IWAD tests, plus the synthetic WAD builder
tools/               dev scripts, the reference fetcher and the demo trace harness
docs/                plan, progress log, decisions, audit and playthrough notes
```

## Architecture

```
          input (keyboard, mouse)                    WAD files
                   │                                     │
                   ▼                                     ▼
   app/ ── GameFlow (title, menus, levels, intermission, hubs, saves) ── wad/, world/
                   │ TicCmd per tic (35 Hz)
                   ▼
   sim/ ── GameSim: P_Ticker → players → thinkers → specials        (fixed point,
                   │         emits events (sounds, messages, map exit)   deterministic)
                   ▼
   render/  draws the current sim state, interpolated between tics
   audio/   plays sounds and music from sim events
   ui/      status bar, automap, menus, messages
```

The simulation knows nothing about drawing or sound. It emits events and exposes plain
state, and the upper layers read them. That keeps the simulation testable without a window,
and lets demos and saves replay exactly. Key design choices are recorded as ADRs in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Documentation

| Document | What it covers |
|---|---|
| [AGENTS.md](AGENTS.md) | Rules and workflow for contributors (start here) |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Architecture, phases and tasks with acceptance criteria |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Status of every task, notes and the session log |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Architecture decision records |
| [docs/AUDIT_2026-11-10.md](docs/AUDIT_2026-11-10.md) | Integration audit and the remediation tasks it produced |
| [docs/PLAYTHROUGH.md](docs/PLAYTHROUGH.md) | Hub-by-hub map and mechanics notes |
| [DEAD_ENDS.md](DEAD_ENDS.md) | Approaches that were tried and must not be repeated |

## Packaging

`pyhexen.spec` builds a Windows one-folder distribution with PyInstaller. It bundles the
shaders and the example config, and excludes all game data:

```sh
uv run --with pyinstaller pyinstaller pyhexen.spec
```

The packaged game finds the IWAD the same way as the source version, and asks with a file
dialog if it can't.

## Legal

- **License:** GPL-2.0-or-later (see [LICENSE](LICENSE)). Much of the code is translated
  from the Hexen source code released under the GPL by Raven Software and id Software,
  through Chocolate Doom, so the whole project is GPL.
- **Game data:** `HEXEN.WAD` and everything in it (maps, textures, sprites, sounds, music,
  demos) is copyrighted and **not** covered by this license. Use your own legally obtained
  copy, and never commit or distribute it, or anything extracted from it, with this project.
- **Trademarks:** "Hexen" and "Heretic" are trademarks of their respective owners. This
  project is not affiliated with or endorsed by them. "PyHexen" is a working codename;
  don't use "Hexen" in the name of a published build without legal review.

**Acknowledgements:** Raven Software and id Software for releasing the source;
[Chocolate Doom](https://github.com/chocolate-doom/chocolate-doom) for the accurate C port
used as the behavioural reference; the Doom Wiki and ZDoom Wiki for format documentation.
