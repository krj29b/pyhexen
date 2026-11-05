# Project: PyHexen Phase 9 — Polyobjects

## Architecture

PyHexen is a Python reimplementation of the Hexen (1995) engine adhering to strict simulation invariants (16.16 fixed point, original lookup-table random, 35 Hz tics, zero floats in `sim/`, no python random, strict layering).

Phase 9 delivers complete support for **Polyobjects** across simulation, physics, ACS scripting, sound sequences, and dynamic OpenGL 3.3 GPU rendering:

```
[Level Loading / WAD]
         │ (Things 3000, 3001, 3002; Specials 1, 5)
         ▼
[sim/polyobj.py: PolyobjManager & Polyobj] ─── (Subsector poly link) ───► [world/level.py: Subsector]
         │                                 ─── (2D dynamic spatial hash) ──► [world/blockmap.py: PolyBlockMap]
         │
         ├─────────────────────────────────────────────────┐
         │ (Movement & Rotation Thinkers)                  │ (Dynamic Transform State)
         ▼                                                 ▼
[sim/thinker.py: Thinker Loop]                    [render/polyobj.py: Dynamic Poly Meshes]
   - RotatePolyThinker (speed, dist)                       │ (exclude from static ChunkManager)
   - MovePolyThinker (speed, dist, vector)                 │ (u_model GPU transformation)
   - PolyDoorThinker (slide, swing, wait, reverse)         │ (sub-tic alpha interpolation)
         │                                                 ▼
         ├──────────► [sim/map.py / physics: Mobj Blocking & Thrusting, Crushing 3 dmg]
         ├──────────► [core/events.py: SoundEvent SEQ_DOOR_STONE + seq_type]
         └──────────► [sim/acs/vm.py: PCD_POLYWAIT & poly_finished with PO_Busy guard]
```

---

## Feature Inventory

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `Polyobj` data structure | Core polyobject state (`tag`, `angle`, `start_spot`, `bbox`, `segs`, `original_pts`, `prev_pts`, `crush`, `seq_type`) | M1 | `r_local.h:153`, `po_man.c:56` |
| 2 | `PolyBlock` & `PolyBlockMap` | 2D dynamic spatial hash grid linking polyobjects to blockmap blocks | M1 | `r_local.h:170`, `po_man.c:55` |
| 3 | Anchor Spot (Thing 3000) | Recognizes thing 3000 in dummy area as reference origin; `angle` = tag | M1 | `po_man.c:1464` |
| 4 | Spawn Spot (Thing 3001) | Recognizes thing 3001 in playable map as start spot; `crush = False` | M1 | `po_man.c:1447` |
| 5 | Crush Spawn Spot (Thing 3002) | Recognizes thing 3002 in playable map as start spot; `crush = True` | M1 | `po_man.c:1447` |
| 6 | StartLine Chains (Special 1) | Automatic recursive closed perimeter seg collection (`IterFindPolySegs`) | M1 | `po_man.c:1216` |
| 7 | Explicit Lines (Special 5) | Manual indexed linedef collection with order numbers (1..63) | M1 | `po_man.c:1256` |
| 8 | Coordinate Translation | Translates segs/lines from anchor to spawn spot; initializes `original_pts` & `prev_pts` (shared vertex deduplicated) | M1 | `po_man.c:1333` |
| 9 | Subsector Linking | Finds subsector containing polyobject center of mass; sets `subsector.poly = po` | M1 | `po_man.c:1407` |
| 10 | Specials Dispatch Registration | Registers specials 2..4, 6..8, 90..93 in `sim/specials/dispatcher.py` | M2 | `po_man.c:114` |
| 11 | `RotatePolyThinker` | Thinker rotating polyobject at fixed angular rate; supports perpetual (255) and 360-deg (0) | M2 | `po_man.c:75` |
| 12 | `MovePolyThinker` | Thinker translating polyobject linearly; supports times-eight scaling | M2 | `po_man.c:210` |
| 13 | `PolyDoorThinker` (Slide) | Sliding door state machine (open, wait, close, reverse on obstacle) | M2 | `po_man.c:347` |
| 14 | `PolyDoorThinker` (Swing) | Swinging door state machine (open, wait, close, reverse on obstacle) | M2 | `po_man.c:397` |
| 15 | Override Specials (90..93) | Force movement/rotation preemption even when thinker is currently active | M2 | `po_man.c:126` |
| 16 | Mirror Polyobject Propagation | Propagates motion to linked mirror polyobjects (opposite translation/rotation) | M2 | `po_man.c:159` |
| 17 | Mobj Collision Detection | Detects solid mobjs/players crossing poly lines (`P_BoxOnLineSide == -1`) | M2 | `po_man.c:1048` |
| 18 | Mobj Thrust Physics | Imparts perpendicular momentum clamped to [1.0, 4.0] fixed-point | M2 | `po_man.c:599` |
| 19 | Crushing Damage | Deals 3 damage when mobj is pinned against geometry and `P_CheckPosition` fails | M2 | `po_man.c:644` |
| 20 | Sound Sequences | Emits `SoundEvent` for sequence start, loop, and stop (`SEQ_DOOR_STONE + seq_type`) | M2 | `po_man.c:156` |
| 21 | ACS `PCD_POLYWAIT` Hook | Suspends waiting scripts with `ASTE_WAITINGFORPOLY` | M2 | `p_acs.c:1555` |
| 22 | ACS `poly_finished` Wakeup | Resumes waiting scripts when thinker completes and `PO_Busy` is false | M2 | `p_acs.c:851` |
| 23 | Dynamic Polyobject Mesh | Builds local-coordinate wall quads from constituent segs/linedefs | M3 | `renderer.py` |
| 24 | Static Chunk Exclusion | Filters out polyobject linedefs from static sector chunk baking (`is_polyobj`) | M3 | `chunks.py` |
| 25 | GPU Model Matrix Transform | Computes and uploads 4x4 model matrix $M(\theta, ix, iy)$ to `u_model` | M3 | `world.vert` |
| 26 | Sub-Tic Frame Interpolation | Interpolates position and angle using render accumulator fraction $\alpha$ | M3 | `main_loop.py` |
| 27 | Dynamic Sector Lighting | Samples lighting level and floor/ceiling heights from currently occupied subsector | M3 | `renderer.py` |
| 28 | E2E Acceptance & Adversarial Hardening | 100% pass on Tiers 1-4 E2E tests + Tier 5 adversarial stress testing | M4 | Requirements |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Polyobject Spawning (T9.1) | `src/pyhexen/sim/polyobj.py`, thing parsing (3000, 3001, 3002), line collection (specials 1 & 5), anchor translation, subsector & blockmap linking, synthetic & real WAD tests | none | DONE (Verified clean passes; 0 errors; Auditor CLEAN; Reviewers/Challengers APPROVE) |
| 2 | M2: Polyobject Movement & Physical Interaction (T9.2) | Motion thinkers (`RotatePolyThinker`, `MovePolyThinker`, `PolyDoorThinker`), line specials (2..4, 6..8, 90..93), collision, thrust, 3-crush damage, mirror propagation, sound sequences, ACS wait & wake | M1 | PLANNED |
| 3 | M3: Polyobject Dynamic GPU Rendering (T9.3) | Static chunk line exclusion (`is_polyobj`), dynamic polyobject mesh generation, model matrix `u_model` in `world.vert`, sub-tic interpolation, dynamic sector lighting | M1 | PLANNED |
| 4 | M4: Final Integration & E2E Verification | Pass 100% of E2E test suite (Tiers 1-4) published by E2E Testing Track, followed by Phase 2 adversarial coverage hardening (Tier 5) | M1, M2, M3, TEST_READY.md | PLANNED |

---

## Interface Contracts

### 1. `sim.polyobj` ↔ `world.level` & `sim.game_sim`
- `PolyobjManager.init_polyobjs(level: Level) -> None`:
  Called after map loading (segs, linedefs, subsectors, things). Translates coordinates, marks `line.is_polyobj = True`, links `subsector.poly`, and populates `level.blockmap.poly_block_map`.
- `Level.polyobjs: dict[int, Polyobj]` and `Level.polyobjs_by_index: list[Polyobj]`:
  Provides tag-based and index-based lookup.
- `Polyobj.busy -> bool`:
  Returns `True` if `poly.specialdata is not None`.

### 2. `sim.polyobj` ↔ `sim.thinker`
- `RotatePolyThinker`, `MovePolyThinker`, `PolyDoorThinker` inherit from `Thinker`.
- `thinker.tick(sim: GameSim) -> None`:
  Advances motion by 1 tic (35 Hz). Handles blocking checks, sound sequence triggers, and completion.
- When motion finishes: clears `poly.specialdata = None`, stops sound sequence, calls `sim.poly_finished(poly.tag)`, and removes thinker via `thinker.remove()`.

### 3. `sim.polyobj` ↔ `sim.acs`
- `GameSim.poly_finished(tag: int)` -> `ACSVM.poly_finished(tag: int)`:
  Checks `if not poly.busy`: sets all scripts waiting with `ASTE_WAITINGFORPOLY && wait_value == tag` to `ASTE_RUNNING`.

### 4. `sim.polyobj` ↔ `core.events` / `audio.sndseq_player`
- Emits `SoundEvent(type=EventType.SOUND_SEQ, seq_id=SeqId.SEQ_DOOR_STONE + poly.seq_type, origin=(poly.start_spot.x, poly.start_spot.y))` upon motion start, loop, and stop without importing `audio`.

### 5. `sim.polyobj` ↔ `render.renderer` / `render.chunks`
- Polyobject linedefs flagged `line.is_polyobj = True` are skipped in `ChunkManager` / `build_sector_walls`.
- `WorldRenderer` maintains dynamic polyobject meshes rendered with local vertices and `u_model = mat4_translation(ix, iy, 0) * mat4_rotation_z(iangle) * mat4_translation(-local_ox, -local_oy, 0)`.
- Interpolation: `ix = prev_x + alpha * (curr_x - prev_x)`, `iy = prev_y + alpha * (curr_y - prev_y)`, `iangle = shortest_arc(prev_angle, curr_angle, alpha)`.

---

## Code Layout

- `src/pyhexen/sim/polyobj.py`: Core polyobject data models (`Polyobj`, `PolyBlock`), manager, line collection, translation, collision, and thinkers.
- `src/pyhexen/sim/specials/dispatcher.py`: Polyobject line specials (1..8, 90..93) dispatch.
- `src/pyhexen/sim/game_sim.py`: Polyobject initialization and tick integration.
- `src/pyhexen/sim/acs/vm.py`: ACS `PCD_POLYWAIT` and `poly_finished` verification.
- `src/pyhexen/world/level.py` & `world/blockmap.py`: `Subsector.poly` and `PolyBlockMap` integration.
- `src/pyhexen/render/renderer.py` & `render/chunks.py`: Dynamic mesh rendering and static chunk exclusion.
- `tests/unit/test_polyobj_spawning.py`: Unit and synthetic spawning tests.
- `tests/unit/test_polyobj_movement.py`: Unit and synthetic motion, collision, and ACS tests.
- `tests/render/test_polyobj_render.py`: Render dynamic mesh and interpolation tests.
- `tests/e2e/`: Opaque-box E2E test suite created by E2E Testing Track.
- `docs/PROGRESS.md`: Session task claims and status tracking.
