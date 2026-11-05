# PyHexen Phase 9 (Polyobjects) — Test Infrastructure Specification

**Document Version:** 1.0.0  
**Phase:** Phase 9 (Polyobjects)  
**Author:** Test Writer Track (`test_writer_e2e`)  
**Target Codebase:** PyHexen (`src/pyhexen/`)  
**Integrity Mode:** Development  

---

## 1. Test Philosophy & Design Principles

The PyHexen Phase 9 testing architecture adheres to strict opaque-box behavioral testing, deterministic simulation invariants, progressive testability, and adversarial hardening.

### 1.1 Opaque-Box Behavioral Testing
Tests interact exclusively through public APIs, event subscriptions, level lumps, and simulation ticks. Tests do not inspect internal private caches or rely on non-standard side-effects. Observable outputs (world coordinates, blockmap linkages, thinker states, sound events, ACS script states, and GPU matrices) form the contract.

### 1.2 Deterministic Simulation Invariants (AGENTS.md)
Polyobject simulation tests enforce the following core engine invariants:
- **16.16 Fixed-Point Arithmetic**: Positions, speeds, and dimensions are strictly computed with integer arithmetic via `core.fixed` (`FRACUNIT = 65536`). No floating-point math in `sim/`.
- **C-Style Integer Arithmetic**: Truncation toward zero via `c_div()` and `c_mod()`; BAM angles masked to 32-bit (`& 0xFFFFFFFF`).
- **Original Lookup Tables**: Angles, fine sine/cosine tables (`finesine`, `finecosine`), and random sequence generators.
- **Strict 35 Hz Tics**: No wall-clock time (`time.time()`) is referenced; simulation advances tic-by-tic.
- **Architectural Layering**: `sim/` never imports `render/`, `audio/`, or `ui/`. Audio communication occurs through `core.events.SoundEvent`.

### 1.3 Progressive Testability
During milestone execution (M1: Spawning, M2: Movement/Physics, M3: Dynamic Rendering, M4: Integration), test modules must execute without crashing or raising missing-dependency errors on milestones not yet implemented.
- Tests conditionally verify feature availability using clean skips when dependent modules/classes are under active implementation.
- As each milestone delivers its feature set, corresponding test suites activate and lock in regression prevention.
- Tests requiring real IWAD assets (`HEXEN.WAD`) use the `@pytest.mark.wad` marker and resolve paths via `iwad_path`. Synthetic tests run fully standalone without external data.

---

## 2. 4-Tier Test Suite Architecture

The Phase 9 E2E test suite (`tests/e2e/test_polyobj_e2e.py`) is structured into four distinct tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Phase 9 Polyobjects E2E Suite                     │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Feature Coverage (>=5 tests per feature, 28 features = 140+)   │
│   - M1 Spawning & Data Structures (Features 1..9)                      │
│   - M2 Thinkers, Physics, Specials, ACS, Audio (Features 10..22)       │
│   - M3 Dynamic GPU Meshes, Transforms & Lighting (Features 23..27)     │
│   - M4 End-to-End Synthetic Verification (Feature 28)                  │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Boundary & Corner Cases (Boundary, error detection, extremes)  │
│   - Angle extremes (0 -> 360°, 255 -> perpetual)                       │
│   - Explicit line sequence errors (skipped order, order 0, >64 segs)   │
│   - Geometry loop failures (non-closed loop, multi-poly subsector)     │
│   - Speed clamping when remaining distance < step speed                │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Cross-Feature Combinations (Pairwise & multi-subsystem flows)  │
│   - Rotate + Mirror motion propagation                                 │
│   - Move + Mirror translation propagation                              │
│   - Sliding door + Solid mobj collision blocking                       │
│   - Swinging door + Obstruction reversal (non-crush door bounces open) │
│   - Pinned crushing damage (3 HP damage when mobj cannot escape)       │
│   - PCD_POLYWAIT + Multi-script simultaneous wakeup                    │
│   - Override specials (90..93) preempting active thinkers              │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 4: Real-World Application Scenarios (Hexen IWAD MAP01 / MAP02)   │
│   - MAP01: All 12 polyobjects spawned, anchor translation verified     │
│   - MAP01: Iconic double swinging entrance doors (Tags 4 & 5) cycle    │
│   - MAP02: All 24 polyobjects spawned, crush tags verified             │
│   - MAP02: Sliding pillar / door operations verified                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Feature Inventory Mapping

The 28 features specified in `PROJECT.md § Feature Inventory` map directly to test classes and functions in `tests/e2e/test_polyobj_e2e.py`:

| # | Feature Name | Milestone | Target Module | Test Class / Functions in `test_polyobj_e2e.py` | Coverage Count |
|---|--------------|-----------|---------------|-------------------------------------------------|----------------|
| 1 | `Polyobj` data structure | M1 | `sim.polyobj` | `TestFeature01PolyobjDataStructure` | 5 |
| 2 | `PolyBlock` & `PolyBlockMap` | M1 | `world.blockmap`, `sim.polyobj` | `TestFeature02PolyBlockMap` | 5 |
| 3 | Anchor Spot (Thing 3000) | M1 | `sim.polyobj` | `TestFeature03AnchorSpot` | 5 |
| 4 | Spawn Spot (Thing 3001) | M1 | `sim.polyobj` | `TestFeature04SpawnSpot` | 5 |
| 5 | Crush Spawn Spot (Thing 3002) | M1 | `sim.polyobj` | `TestFeature05CrushSpawnSpot` | 5 |
| 6 | StartLine Chains (Special 1) | M1 | `sim.polyobj` | `TestFeature06StartLineChains` | 5 |
| 7 | Explicit Lines (Special 5) | M1 | `sim.polyobj` | `TestFeature07ExplicitLines` | 5 |
| 8 | Coordinate Translation | M1 | `sim.polyobj` | `TestFeature08CoordinateTranslation` | 5 |
| 9 | Subsector Linking | M1 | `world.level`, `sim.polyobj` | `TestFeature09SubsectorLinking` | 5 |
| 10 | Specials Dispatch Registration | M2 | `sim.specials.dispatcher` | `TestFeature10SpecialsDispatch` | 5 |
| 11 | `RotatePolyThinker` | M2 | `sim.polyobj` | `TestFeature11RotatePolyThinker` | 5 |
| 12 | `MovePolyThinker` | M2 | `sim.polyobj` | `TestFeature12MovePolyThinker` | 5 |
| 13 | `PolyDoorThinker` (Slide) | M2 | `sim.polyobj` | `TestFeature13PolyDoorSlide` | 5 |
| 14 | `PolyDoorThinker` (Swing) | M2 | `sim.polyobj` | `TestFeature14PolyDoorSwing` | 5 |
| 15 | Override Specials (90..93) | M2 | `sim.polyobj`, `sim.specials` | `TestFeature15OverrideSpecials` | 5 |
| 16 | Mirror Polyobject Propagation | M2 | `sim.polyobj` | `TestFeature16MirrorPropagation` | 5 |
| 17 | Mobj Collision Detection | M2 | `sim.polyobj`, `world.blockmap` | `TestFeature17MobjCollisionDetection` | 5 |
| 18 | Mobj Thrust Physics | M2 | `sim.polyobj` | `TestFeature18MobjThrustPhysics` | 5 |
| 19 | Crushing Damage | M2 | `sim.polyobj`, `sim.mobj` | `TestFeature19CrushingDamage` | 5 |
| 20 | Sound Sequences | M2 | `sim.polyobj`, `core.events` | `TestFeature20SoundSequences` | 5 |
| 21 | ACS `PCD_POLYWAIT` Hook | M2 | `sim.acs.vm` | `TestFeature21ACSPolyWait` | 5 |
| 22 | ACS `poly_finished` Wakeup | M2 | `sim.acs.vm`, `sim.game_sim` | `TestFeature22ACSPolyFinishedWakeup` | 5 |
| 23 | Dynamic Polyobject Mesh | M3 | `render.renderer` | `TestFeature23DynamicPolyMesh` | 5 |
| 24 | Static Chunk Exclusion | M3 | `render.chunks` | `TestFeature24StaticChunkExclusion` | 5 |
| 25 | GPU Model Matrix Transform | M3 | `render.renderer` | `TestFeature25GPUModelMatrixTransform` | 5 |
| 26 | Sub-Tic Frame Interpolation | M3 | `render.renderer` | `TestFeature26SubTicInterpolation` | 5 |
| 27 | Dynamic Sector Lighting | M3 | `render.renderer` | `TestFeature27DynamicSectorLighting` | 5 |
| 28 | E2E Acceptance & Adversarial Hardening | M4 | Full System | `TestFeature28E2EAcceptance` | 5 |
| T2 | Tier 2 Boundary & Corner Cases | M1-M4 | All Modules | `TestTier2BoundaryAndCornerCases` | 9 |
| T3 | Tier 3 Cross-Feature Combinations | M1-M4 | Multi-Subsystem | `TestTier3CrossFeatureCombinations` | 7 |
| T4 | Tier 4 Real-World Application Scenarios | M1-M4 | Real IWAD MAP01/02 | `TestTier4RealWorldScenarios` | 4 |

**Total Test Count:** >= 160 distinct test cases.

---

## 4. Coverage & Quality Thresholds

1. **Feature Coverage Requirement**: Every inventoried feature has at least 5 dedicated, verifiable test scenarios.
2. **Boundary Assurance**: Every mathematical and structural edge case documented in `po_man.c` (angle wrap, 64-seg limit, loop closure, order indexing, subsector uniqueness) has an adversarial validation test.
3. **Lint & Formatting Invariant**: All test files must pass `uv run ruff check .` and `uv run ruff format --check .` with zero warnings or errors.
4. **Execution Performance**: Synthetic E2E tests must execute within 15 seconds on a standard CI runner.

---

## 5. Test Runner Commands

```powershell
# Run the complete Phase 9 synthetic E2E test suite
uv run pytest tests/e2e/test_polyobj_e2e.py -v

# Run real IWAD tests (requires HEXEN.WAD path configured)
uv run pytest tests/e2e/test_polyobj_e2e.py -m wad -v

# Run lint and formatting verification
uv run ruff check tests/e2e/test_polyobj_e2e.py tests/fixtures/wadbuilder.py
uv run ruff format --check tests/e2e/test_polyobj_e2e.py tests/fixtures/wadbuilder.py
```
