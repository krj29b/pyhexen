# AGENTS.md — Onboarding for AI coding agents

You are working on **PyHexen**, a Python reimplementation of the Hexen (1995, Raven Software)
engine, using the GPL-released source as the behavioural reference and a modern GPU renderer.
This file is the **entry point for every agent session**. Read it fully before touching code.

## 1. Read in this order

1. `AGENTS.md` (this file): rules and workflow.
2. `docs/IMPLEMENTATION_PLAN.md`: architecture, decisions, phases, and the task list with
   acceptance criteria.
3. `docs/PROGRESS.md`: current status of each task, plus notes and gotchas from previous sessions.
4. `docs/DECISIONS.md`: architecture decision records (ADRs). Don't contradict an ADR without
   adding a new one that supersedes it.

## 2. Session protocol (follow every time)

1. **Pick a task.** In `docs/PROGRESS.md`, choose the first task with status `TODO` whose
   dependencies are all `DONE`. If the user assigned a specific task, do that one instead.
2. **Claim it.** Set its status to `IN PROGRESS (<agent>/<date>)` and commit that change alone,
   so parallel agents don't collide.
3. **Study the reference.** Each task lists reference files from Chocolate Doom's `src/hexen/`.
   Read them in `reference/` (fetched by `tools/fetch_reference.*`; never committed).
4. **Implement with tests.** Satisfy every acceptance criterion in the plan.
5. **Verify.** Run `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest`.
   All must pass. Tests marked `wad` run only when an IWAD is configured.
6. **Record.** Set the task to `DONE` in `docs/PROGRESS.md`, and add short notes covering
   deviations, gotchas and follow-ups. If you make a design choice, add an ADR to
   `docs/DECISIONS.md`.
7. **Commit.** Use a message like `T<id>: <summary>`, with one task per commit or PR where
   practical.

If a task is too large for one session, split it into `T<id>a`, `T<id>b` and so on in
`PROGRESS.md`, and explain why in the notes.

## 3. Hard rules

### Legal
- **Never commit game data.** That means no `*.wad`, and no extracted textures, sprites, sounds,
  music or maps, including PNG exports and test fixtures derived from a real WAD. `.gitignore`
  covers this, but you're still responsible.
- Tests that need real data use the `@pytest.mark.wad` marker and read the IWAD path from config
  or the environment. CI tests use **synthetic WADs** built by `tests/fixtures/wadbuilder.py`.
- Code ported or translated from the reference source is **GPL-2.0-or-later**. Keep the license
  header on every source file (see §6).
- Don't use "Hexen" in a published product name or binary without legal review. `PyHexen` is the
  working codename only.

### Determinism (simulation code, `src/pyhexen/sim/` and below)
- **No floats in the simulation.** Use 16.16 fixed point as Python `int`, via `core/fixed.py`.
  Floats are allowed only in rendering, audio and UI.
- **No `random` module.** Use `core/random.py` (the original 256-entry table, `P_Random` and
  `M_Random`).
- **No wall-clock time** in the simulation. Everything advances in tics (35 Hz).
- **Python ≠ C arithmetic.** C `/` and `%` truncate toward zero, while Python `//` and `%`
  floor. Use `fixed.c_div()` and `fixed.c_mod()`. Wrap angles with `& 0xFFFFFFFF`, and
  wrap int16/int32 fields where the original relies on overflow.
- Preserve the original **order** of thinker execution and `P_Random` calls. Any reordering
  breaks demo and determinism tests.

### Performance
- **No per-pixel or per-texel Python loops at runtime.** Use numpy vectorisation, numba, or a
  shader. Decoding assets at load time is fine.
- Budgets: average simulation tic under 10 ms, average CPU render frame under 8 ms on MAP01 on a
  mid-range PC. Profile before optimising (`py-spy`, `cProfile`).

### Code quality
- Python 3.12+, fully type-hinted public APIs, `ruff` clean, `dataclass(slots=True)` for
  hot objects.
- Keep the layering (see the plan, §3). `sim` must never import `render`, `audio` or `ui`.
  Communicate through events and interfaces instead.
- Keep the original names in docstrings when porting a function, for example
  `"""Port of P_TryMove (p_map.c)."""`, so the next agent can cross-reference it.

## 4. Commands

```powershell
uv sync                                  # install deps
uv run pytest                            # tests (wad tests auto-skip without IWAD)
uv run pytest -m wad                     # only tests needing real game data
uv run ruff check . ; uv run ruff format .
uv run python -m pyhexen --iwad C:\path\HEXEN.WAD --warp MAP01
uv run python -m pyhexen.tools.wadinfo --iwad C:\path\HEXEN.WAD list
./tools/fetch_reference.ps1              # clone reference source into ./reference (gitignored)
```

Find the IWAD with, in priority order: the `--iwad` flag, the `PYHEXEN_IWAD` environment
variable, then `iwad =` in `config.toml`.

## 5. When stuck

- Behaviour is ambiguous → the reference source (Chocolate Doom `src/hexen/`) is the truth.
  Behaviour should match it, though the structure doesn't have to.
- The reference is ambiguous → check the Doom Wiki or ZDoom Wiki format docs, and note the source
  in your PROGRESS notes.
- You'd need to change architecture → stop, write a proposed ADR in `docs/DECISIONS.md` with
  status `PROPOSED`, and ask the user.

## 6. Source file header

```python
# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.
```
