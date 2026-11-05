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
**Pinned tag:** _to be filled in by T0.2_
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
