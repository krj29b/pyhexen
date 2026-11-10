# Architecture Decision Records

Format: `ADR-NNN — Title` · Status (`ACCEPTED` | `PROPOSED` | `SUPERSEDED by ADR-XXX`) · Context, decision and consequences.
Add new ADRs at the bottom. Never delete one; supersede it instead.

---

## ADR-001 — Python + pygame-ce + moderngl stack
**Status:** ACCEPTED
**Context:** The project goal is a modern reimplementation in Python.
**Decision:** Python 3.12+, pygame-ce (window, input, mixer), moderngl (OpenGL 3.3 core), numpy, optional numba, and uv for the environment.
**Consequences:** Rendering must happen on the GPU, and no per-pixel CPU loops are allowed.

## ADR-002 — Chocolate Doom `src/hexen` as behavioural reference
**Status:** ACCEPTED
**Context:** We need an authoritative, GPL, portable version of the original source.
**Decision:** Use Chocolate Doom, pinned to a release tag, cloned to `reference/` (gitignored). Behaviour must match it; code structure may differ.
**Pinned tag:** `chocolate-doom-3.1.1` (commit `410d96855b5df5410ff591a90efeafa889119224`)
**Consequences:** The project is GPL-2.0-or-later.

## ADR-003 — Fixed-point integer simulation
**Status:** ACCEPTED
**Context:** Determinism, faithful behaviour, and possible future demo sync.
**Decision:** All simulation quantities are Python ints in 16.16 fixed point / 32-bit BAM angles, with C division semantics emulated. Floats are used only in render, audio and UI.
**Consequences:** Some speed cost; helpers in `core/fixed.py` are mandatory.

## ADR-004 — Per-sector mesh chunks with CPU rebuild
**Status:** ACCEPTED
**Context:** Floors and ceilings move; the GPU geometry must follow.
**Decision:** Group geometry by sector, and rebuild dirty sectors and their neighbouring walls on the CPU.
**Alternative (future):** a sector-height data texture read in the vertex shader (no rebuilds).
**Consequences:** Simple to implement; must meet the < 2 ms per sector rebuild target.

## ADR-005 — Explicit versioned serialisation (no pickle)
**Status:** ACCEPTED
**Context:** Save games and hub snapshots must survive code changes and be safe to load.
**Decision:** `to_state()` / `from_state()` producing plain data; references stored as indices; zlib-compressed with a version header.

## ADR-006 — Shear-based look up/down by default
**Status:** ACCEPTED
**Context:** The original uses y-shearing; modern players may prefer true perspective.
**Decision:** Projection-matrix shear by default, with a true-pitch option in settings.

## ADR-007 — Indexed-color rendering via palette shader
**Status:** ACCEPTED
**Context:** We want the authentic look, palette flashes and fog maps.
**Decision:** Upload textures as 8-bit indices; the shader applies the COLORMAP/FOGMAP row, then PLAYPAL. An optional RGBA "smooth" mode can come later.

## ADR-008 — Game flow lives in `app/GameFlow`, mirroring G_Ticker game actions
**Status:** ACCEPTED
**Context:** Map exits, intermissions, the finale, new games and reborn must switch the current `GameSim` between tics. The sim must not import `ui`, and nothing may swap the sim in the middle of a tic.
**Decision:** `app/game_flow.py` owns the current `GameSim` and `HubManager`. It drains the sim's event bus after every tic and turns `MapExit` into a deferred action that runs at the start of the next tic, as vanilla `gameaction` does (`ga_completed` → intermission → `ga_leavemap` → `SV_MapTeleport`). Reborn uses an in-memory copy of the vanilla reborn slot, written after every map change. Every map entry (new map, revisit, reborn) runs a full `GameSim.setup_level()` first, as vanilla `G_InitNew` does, and a revisit or reborn then unarchives over it.
**Consequences:** `__main__` only feeds tic commands and draws what `GameFlow.state` says. Later wiring (audio, messages) consumes the event list that `GameFlow.tick()` returns instead of subscribing to each new sim's bus.
