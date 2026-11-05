# Dead Ends — Failed Approaches Log

This document records failed approaches that must NEVER be repeated.

| Iteration | Approach Tried | Why It Failed | Files Touched |
|-----------|---------------|---------------|---------------|
| M1 Iteration 1 | Hardcoded MAP02 crush override: `getattr(level, "name", "").upper() == "MAP02" and tag in (19..25)` to pass flawed E2E test assertion | FAILED: Integrity violation. Corrupts vanilla simulation physics. Commercial MAP02 things are type 3001 (non-crush). Tests must be corrected, never source code hardcoded. | `src/pyhexen/sim/polyobj.py` |
| M1 Iteration 1 | Dummy empty `Subsector` fabrication (`Subsector(num_lines=0, first_line=0)`) when `sub.poly is not None` | FAILED: Violates reference `po_man.c:1408` which unconditionally raises fatal error if multiple polyobjs share a subsector. | `src/pyhexen/sim/polyobj.py` |
| M1 Iteration 1 | Runtime monkeypatching of `Blockmap.block_lines_iterator = _block_lines_iter_compat` with coordinate-guessing heuristics | FAILED: Fragile facade that corrupts block indexing on maps wider than 128 blocks. Tests must pass proper block indices. | `src/pyhexen/world/level.py` |
