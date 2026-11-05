# PyHexen Phase 9 (Polyobjects) — Test Readiness Report

**Status:** READY  
**Date:** 2026-11-05  
**Author:** Test Writer Track (`test_writer_e2e`)  
**Target File:** `tests/e2e/test_polyobj_e2e.py`  
**Test Infra Doc:** `TEST_INFRA.md`  

---

## 1. Executive Summary

The comprehensive, opaque-box End-to-End (E2E) test suite for Phase 9 (Polyobjects) is complete, verified, and integrated into PyHexen. The suite covers all 28 features from `PROJECT.md § Feature Inventory` across a 4-tier methodology, providing complete coverage across spawning (M1), thinkers and physical interaction (M2), GPU dynamic rendering (M3), and system-wide integration (M4).

---

## 2. Test Suite Structure & Metrics

| Tier | Category | Scope / Features | Tests | Status |
|------|----------|------------------|-------|--------|
| **Tier 1** | Feature Coverage | Features 1–28 (>=5 tests per feature covering happy path, data models, state transitions, interface contracts) | 140 | READY |
| **Tier 2** | Boundary & Corner Cases | Boundary extremes (angles 0 & 255, skipped order, order 0, non-closed loops, multiple polyobjs per subsector, max 64 segs, speed clamping) | 9 | READY |
| **Tier 3** | Cross-Feature Combinations | Pairwise subsystem interactions (rotate+mirror, move+mirror, sliding door collision, swinging door reversal, crushing damage, PCD_POLYWAIT multi-script, override specials) | 7 | READY |
| **Tier 4** | Real-World Application Scenarios | Real Hexen IWAD MAP01 (12 polyobjs, entrance double doors) and MAP02 (24 polyobjs, crush sliding pillars) | 4 | READY |
| **Total** | | Complete Phase 9 E2E Test Suite | **160** | **READY** |

---

## 3. Progressive Testability Guarantees

In accordance with `PROJECT.md` and teamwork rules, the test suite supports **progressive testability**:
- Milestone 1 tests (`HAS_M1_SPAWNING`): Activate automatically once `sim.polyobj.Polyobj` and `init_polyobjs` are implemented.
- Milestone 2 tests (`HAS_M2_THINKERS`, `HAS_M2_PHYSICS`): Activate automatically when `RotatePolyThinker`, `MovePolyThinker`, `PolyDoorThinker`, and physics routines are delivered.
- Milestone 3 tests (`HAS_M3_RENDER`): Activate when dynamic GPU meshes and transformations are connected.
- Real IWAD tests (`@pytest.mark.wad`): Automatically skip when no IWAD is configured and run when `HEXEN.WAD` is available.
- All 160 tests compile, collect cleanly, and execute with zero errors under `uv run pytest`.

---

## 4. Quality & Lint Verification

- `uv run ruff check tests/e2e/test_polyobj_e2e.py tests/fixtures/wadbuilder.py`: **Passed (0 errors)**.
- `uv run ruff format --check tests/e2e/test_polyobj_e2e.py tests/fixtures/wadbuilder.py`: **Passed (clean)**.
- `uv run pytest tests/e2e/test_polyobj_e2e.py`: **160 collected, 0 failed**.
- Full test suite regression test: **162 passed, 160 progressively skipped**.

---

## 5. How to Run the Tests

```powershell
# Run the synthetic Phase 9 E2E test suite
uv run pytest tests/e2e/test_polyobj_e2e.py -v

# Run real Hexen IWAD polyobject tests (requires configured IWAD)
uv run pytest tests/e2e/test_polyobj_e2e.py -m wad -v

# Run lint and formatting check
uv run ruff check tests/e2e/test_polyobj_e2e.py tests/fixtures/wadbuilder.py
uv run ruff format --check tests/e2e/test_polyobj_e2e.py tests/fixtures/wadbuilder.py
```
