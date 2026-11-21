# Playthrough checklist (T15.7, Milestone M6)

M6 asks for a real playthrough of every hub with every class. That has two halves:

* **T15.7a, scripted (done, 2026-11-14):** `python -m pyhexen.tools.playthrough` plays the
  game through `GameFlow`, the same controller the window uses, with input given as
  ticcmds. It checks every exit, hub persistence, the cluster change, reborn, and a
  per-class bot run with save/load on every map. Results are below.
* **T15.7b, manual (open):** a person finishes each hub with each class: puzzles, bosses,
  weapon pieces, the finale. A script cannot judge that, and it cannot judge pictures or
  sound. The checklist is in section 4. **M6 is met when T15.7b is done.**

The old T14.5 audit (`tools/playthrough_audit.py`: each map run for 35 tics with no input,
hubs driven through `HubManager` directly) is gone; this replaces it.

## 1. Running it

```powershell
uv run python -m pyhexen.tools.playthrough --iwad HEXEN.WAD --render     # everything, ~7 min
uv run python -m pyhexen.tools.playthrough --iwad HEXEN.WAD --file HEXDD.WAD --render  # Deathkings, ~10 min
uv run python -m pyhexen.tools.playthrough --iwad HEXEN.WAD --only exits --maps 2,3
uv run python -m pyhexen.tools.playthrough --iwad HEXEN.WAD --only soak --classes 2 --soak-seconds 120
uv run pytest -m wad tests/wad/test_playthrough_iwad.py   # exits + hubs + reborn (base + Deathkings)
```

The exit code is 1 if any check fails. `WARN` lines are explained in section 3.

## 2. What each check does

| Check | How | Passes when |
|---|---|---|
| **exit** (every exit line) | Start the map with `init_new`, put the player 24 to 96 units in front of the line (front side, as `Teleport_NewMap` requires), walk forward, then sit through the intermission. | The map named by the line (`G_Completed(args[0], args[1])`; no warp translation) loads with the player on the start for `args[1]`, or the finale starts for `Teleport_EndGame`. The new map then runs for one second. |
| **exit** (ACS) | Scripts whose bytecode has an `LSPEC` pcode followed by an exit special are run with the player as activator. | The script leads to a real map (and to the right one when the bytecode names it). A candidate that never exits is a `WARN`: the scan can match operands. |
| **hub** return | In one map of each hub, kill every monster, give all keys, wait 5 s for death animations, leave by an exit line, come back by the other map's exit line. | No corpse is lost (dormant monsters cannot be killed and scripts may spawn more, so the test is that the dead count never drops), the keys are kept, and the player is on the right start. |
| **hub** cluster change | Leave with all keys through an exit into another cluster. | Keys are stripped (`G_PlayerExitMap`). |
| **reborn** | Leave MAP01 (so a reborn slot exists), die, press use. Once per class. | The same map reloads with a live player at full health. |
| **soak** | Every map, every class, 60 s of bot input with god mode, all weapons, mana and keys: walk, turn when stuck and every 5 s, press use every 12 tics, aim and fire at visible monsters, change weapon every 6 s. With `--render` a frame is drawn headlessly every 10 s. If the bot walks into an exit it follows it. | No exception; the player's sector link matches its position and it stays inside the sector under it (allowing vanilla's case of a solid thing over a step, see 3); the player moved. |
| **save/load** (after each soak) | Save through the flow, play 2 s, load, play 2 s more. | The loaded state equals the saved one in the demo-trace fields (position, momentum, health, mobj count and hash, sector hash). The RNG index is left out: vanilla does not save it (`SV_LoadGame` runs `G_InitNew`, which clears it), so a loaded game does not replay the same input identically, in vanilla or here. |

## 3. Results (v1.0 IWAD, 2026-11-14)

**181 checks: 170 pass, 11 warn, 0 fail** (441 s with `--render`).

| Map | Name | Cluster | Exit lines (pass / warn) | ACS exits | Fighter | Cleric | Mage |
|---|---|---|---|---|---|---|---|
| MAP01 | Winnowing Hall | 1 | 1 to 02 (1 / 0) | - | PASS | PASS | PASS |
| MAP02 | Seven Portals | 1 | 2 to 06, 13 (1 / 1) | - | PASS | PASS | PASS |
| MAP03 | Guardian of Ice | 1 | 2 to 04, 05 (0 / 2) | 1 ok, 1 unconfirmed | PASS | PASS | PASS |
| MAP04 | Guardian of Fire | 1 | 1 to 02 (1 / 0) | 1 ok | PASS | PASS | PASS |
| MAP05 | Guardian of Steel | 1 | 2 to 02 (2 / 0) | - | PASS | PASS | PASS |
| MAP06 | Bright Crucible | 1 | 1 to 02 (1 / 0) | - | PASS | PASS | PASS |
| MAP08 | Darkmere | 2 | 3 to 09, 10, 13 (3 / 0) | - | PASS | PASS | PASS |
| MAP09 | Caves of Circe | 2 | 3 to 08, 10, 13 (3 / 0) | - | PASS | PASS | PASS |
| MAP10 | Wastelands | 2 | 3 to 08, 09, 13 (2 / 1) | - | PASS | PASS | PASS |
| MAP11 | Sacred Grove | 2 | 1 to 13 (1 / 0) | - | PASS | PASS | PASS |
| MAP12 | Hypostyle | 2 | 2 to 13, 27 (2 / 0) | - | PASS | PASS | PASS |
| MAP13 | Shadow Wood | 2 | 5 to 08, 09, 10, 11, 12 (5 / 0) | - | PASS | PASS | PASS |
| MAP21 | Forsaken Outpost | 4 | 3 to 22, 24, 26 (3 / 0) | - | PASS | PASS | PASS |
| MAP22 | Castle of Grief | 4 | 2 to 21, 24 (2 / 0) | 1 ok | PASS | PASS | PASS |
| MAP23 | Gibbet | 4 | 3 to 22, 25, 35 (3 / 0) | 1 ok | PASS | PASS | PASS |
| MAP24 | Effluvium | 4 | 4 to 21, 22, 23, 25 (2 / 2) | - | PASS | PASS | PASS |
| MAP25 | Dungeons | 4 | 1 to 23 (1 / 0) | 1 ok | PASS | PASS | PASS |
| MAP26 | Desolate Garden | 4 | 1 to 21 (1 / 0) | - | PASS | PASS | PASS |
| MAP27 | Heresiarch's Seminary | 3 | 7 to 22, 28, 30, 31, 32, 33, 34 (7 / 0) | - | PASS | PASS | PASS |
| MAP28 | Dragon Chapel | 3 | 3 to 27, 30, 34 (3 / 0) | - | PASS | PASS | PASS |
| MAP30 | Griffin Chapel | 3 | 3 to 27, 28, 34 (3 / 0) | - | PASS | PASS | PASS |
| MAP31 | Deathwind Chapel | 3 | 1 to 27 (1 / 0) | - | PASS | PASS | PASS |
| MAP32 | Orchard of Lamentations | 3 | 1 to 27 (1 / 0) | - | PASS | PASS | PASS |
| MAP33 | Silent Refectory | 3 | 1 to 27 (1 / 0) | - | PASS | PASS | PASS |
| MAP34 | Wolf Chapel | 3 | 3 to 27, 28, 30 (3 / 0) | - | PASS | PASS | PASS |
| MAP35 | Necropolis | 5 | 5 to 36, 37, 38, 39, 40 (4 / 1) | - | PASS | PASS | PASS |
| MAP36 | Zedek's Tomb | 5 | 1 to 35 (1 / 0) | 1 ok | PASS | PASS | PASS |
| MAP37 | Menelkir's Tomb | 5 | 1 to 35 (0 / 1) | - | PASS | PASS | PASS |
| MAP38 | Traductus' Tomb | 5 | 1 to 35 (1 / 0) | - | PASS | PASS | PASS |
| MAP39 | Vivarium | 5 | 1 to 35 (1 / 0) | - | PASS | PASS | PASS |
| MAP40 | Dark Crucible | 5 | 1 to end (1 / 0) | - | PASS | PASS | PASS |
| MAP41 | Maze (deathmatch) | 0 | none | - | PASS | PASS | PASS |

Soak columns include the save/load check. Hub returns: MAP08 -> 09 -> 08, MAP21 -> 22 -> 21
and MAP27 -> 31 -> 27 pass; MAP02 -> 06 -> 02 and MAP35 -> 38 -> 35 pass with a `WARN` (an
exit had to be activated directly). Cluster change MAP02 -> MAP13 strips keys. Reborn passes
for all three classes.

**Why the 11 warnings are not bugs.** Nine exit lines (and with them two hub returns) cannot
be walked over from a fresh start: the line sits behind a closed door or a lowered ceiling
that the map opens later (MAP03 x2, MAP24 x2, MAP35, MAP37), behind a step higher than 24
units (MAP02 line 197), or out of reach from the front (MAP10 line 1040). Activated the way a
crossing activates them, they still lead to the right map and start. Whether the map opens them is part of
the manual checklist. The ACS candidate in MAP03 script 11 is a false match of the bytecode
scan (`Teleport_NewMap(32, 11)` is not a hub 1 destination).

**Bugs found and fixed (T15.7.x):**

* **T15.7.1** `A_MinotaurRoam` was a loose rewrite that called `P_NewChaseDir` without a
  target, so a Maulotaur with nothing to fight crashed the game (`RuntimeError: P_NewChaseDir:
  called with no target`, MAP10 and MAP35 soaks). Ported from p_enemy.c: random direction
  changes with `FaceMovementDirection`, turning on a blocked move. `A_MinotaurLook` lost a
  `CheckMinotaurAge` call that vanilla does not make. Regression test in
  `test_monsters_bosses.py`.

**Vanilla behaviour confirmed, not changed:** a player can stand up to 24 units below the floor
of a step it walked onto while a solid monster hovers over the step: `P_CheckOnmobj` finds the
monster and `P_MobjThinker` only stops the fall (MAP04, an afrit). It clears when the monster
moves. Loaded games do not continue with the saved RNG index (see save/load above).

**Not covered by the script:** puzzle items and switch puzzles, boss fights, the fourth
weapon pieces, polyobject doors that need a puzzle, intermission text, the finale screens,
music and sound, what frames look like (frames are drawn, not judged), menus, and the
deathmatch map with more than one player.

### 3b. Deathkings of the Dark Citadel (`HEXDD.WAD`, S3b, 2026-11-21)

**174 checks: 165 pass, 9 warn, 0 fail** (622 s with `--render`, v1.0 `HEXEN.WAD` + `HEXDD.WAD`).

| Map | Name | Cluster | Exit lines (pass / warn) | ACS exits | Fighter | Cleric | Mage |
|---|---|---|---|---|---|---|---|
| MAP33 | Transit (deathmatch) | 4 | none | 1 unconfirmed (arg-driven) | PASS | PASS | PASS |
| MAP34 | Over N Under (deathmatch) | 4 | none | 1 ok (to 33) | PASS | PASS | PASS |
| MAP35 | Deathfog (deathmatch) | 4 | none | 1 ok (to 33) | PASS | PASS | PASS |
| MAP36 | Castle of Pain (deathmatch) | 4 | none | 1 ok (to 33) | PASS | PASS | PASS |
| MAP37 | Sewer Pit (deathmatch) | 4 | none | 1 ok (to 33) | PASS | PASS | PASS |
| MAP38 | The Rose (deathmatch) | 4 | none | 1 ok (to 33) | PASS | PASS | PASS |
| MAP41 | Ruined Village | 1 | 5 to 42, 47 (5 / 0) | - | PASS | PASS | PASS |
| MAP42 | Blight | 1 | 8 to 43, 44, 45, 46 (8 / 0) | 2 ok (to 41) | PASS | PASS | PASS |
| MAP43 | Sump | 1 | 6 to 42, 44, 46 (6 / 0) | - | PASS | PASS | PASS |
| MAP44 | Catacomb | 1 | 6 to 42, 43, 45 (6 / 0) | - | PASS | PASS | PASS |
| MAP45 | Badlands | 1 | 6 to 42, 44, 46 (6 / 0) | - | PASS | PASS | PASS |
| MAP46 | Brackenwood | 1 | 6 to 42, 43, 45 (4 / 2) | - | PASS | PASS | PASS |
| MAP47 | Pyre | 1 | 1 to 41 (0 / 1) | - | PASS | PASS | PASS |
| MAP48 | Constable's Gate | 2 | 5 to 49, 50, 51, 52 (3 / 2) | 1 ok (to 54) | PASS | PASS | PASS |
| MAP49 | Treasury | 2 | 3 to 48, 53 (3 / 0) | - | PASS | PASS | PASS |
| MAP50 | Market Place | 2 | 1 to 48 (1 / 0) | - | PASS | PASS | PASS |
| MAP51 | Locus Requiescat | 2 | 2 to 48 (2 / 0) | - | PASS | PASS | PASS |
| MAP52 | Ordeal | 2 | 4 to 48, 49 (3 / 1) | - | PASS | PASS | PASS |
| MAP53 | Armory | 2 | 2 to 49 (2 / 0) | - | PASS | PASS | PASS |
| MAP54 | Nave | 3 | 12 to 55, 56, 57, 58, 59, 60 (12 / 0) | - | PASS | PASS | PASS |
| MAP55 | Chantry | 3 | none | 1 ok (to 54) | PASS | PASS | PASS |
| MAP56 | Abattoir | 3 | 3 to 54, 57 (3 / 0) | - | PASS | PASS | PASS |
| MAP57 | Dark Watch | 3 | 3 to 54, 58 (3 / 0) | - | PASS | PASS | PASS |
| MAP58 | Cloaca | 3 | 1 to 55 (1 / 0) | 1 ok (to 54) | PASS | PASS | PASS |
| MAP59 | Ice Hold | 3 | 2 to 54 (2 / 0) | - | PASS | PASS | PASS |
| MAP60 | Dark Citadel | 3 | 1 to end (1 / 0) | 1 unconfirmed (DM-only) | PASS | PASS | PASS |

Hub returns: `MAP48 -> 50 -> 48` and `MAP54 -> 56 -> 54` pass; `MAP41 -> 47 -> 41` passes
with a `WARN` (MAP47 line 1524 is blocked by a Bishop). Cluster change `MAP48 -> MAP54`
(via ACS script 13) passes and strips all keys. Reborn (leaving warp 1 = `MAP41` to `MAP42`)
passes for all three classes. `MAP83` ("TEST") is in `MAPINFO` without lumps in `HEXDD.WAD`
and is skipped automatically.

**Why the 9 warnings are not bugs:**
* `MAP46` lines 1435 & 1436 (Brackenwood -> Badlands): flagged `0x1204` (`SPAC_PUSH`) on a
  non-blocking two-sided archway in `HEXDD.WAD` itself, so vanilla Hexen does not trigger
  them by walking through either. Direct activation with `SPAC_PUSH` verifies they still
  lead to `MAP45` start 1.
* `MAP47` line 1524 (and hub return `MAP41 -> 47 -> 41`): blocked by a Dark Bishop (`MT_BISHOP`).
* `MAP48` line 880: blocked by a Chaos Serpent (`MT_DEMON`); line 881 beside it walks cleanly.
* `MAP48` line 1484: behind a closed door / lowered ceiling opened by the hub puzzle.
* `MAP52` line 1204: blocked by a statue (`MT_MISC50`); line 1205 beside it walks cleanly.
* `MAP33` script 1: deathmatch transit hub selector taking `arg0` in `1..5` (defaults to
  no-op when called with `0`; bots walked into the real switches during soak and transitioned
  to `MAP34`, `MAP35`, and `MAP37`).
* `MAP60` script 5: checks `GameType() == GAME_NET_DEATHMATCH` before calling
  `Teleport_NewMap(41, 0)`, so it rightly does nothing in single player.

## 4. Manual checklist (T15.7b)

Play each hub with each class at skill 3 (`--class fighter|cleric|mage`), from a new game,
without warping, saving where you like. Tick a box when it works; file anything else as
`T15.7.n` in PROGRESS.md with the map, position (F12 writes it next to the screenshot) and a
save if possible. Run the script after each fix.

| Hub | Fighter | Cleric | Mage | What to watch |
|---|---|---|---|---|
| 1 Seven Portals (MAP01-06) | [ ] | [ ] | [ ] | Seven Portals switch puzzle opening the Guardian portals; Guardian of Ice/Fire/Steel puzzles; the Bright Crucible; first weapon piece; leaving to Shadow Wood |
| 2 Shadow Wood (MAP08-13) | [ ] | [ ] | [ ] | Swamp key, Caves of Circe, Wastelands polyobjects, Sacred Grove; Hypostyle; Death Wyvern; weapon pieces; leaving to the Seminary |
| 3 Heresiarch's Seminary (MAP27-34) | [ ] | [ ] | [ ] | Chapel puzzles (gears, symbols); the Heresiarch fight (balls, spells, bishops); leaving to the Castle |
| 4 Castle of Grief (MAP21-26) | [ ] | [ ] | [ ] | Gibbet, Effluvium and Dungeons doors; Desolate Garden; leaving to the Necropolis |
| 5 Necropolis (MAP35-40) | [ ] | [ ] | [ ] | Three tombs; Vivarium; class boss fights in the Dark Crucible; Korax; the finale for that class |

Also once per class: the fourth weapon assembled from its three pieces and fired; every
artifact used at least once; death and "press use" reborn; save and load from the menu and
with F-keys, inside a hub and after a cluster change; music changes per map; automap; the
status bar and inventory bar; the intermission text between clusters.

### Multiplayer (S2d, needs two or more computers)

The netgame was checked with 2-4 processes on one machine (`tests/render/test_net_processes.py`,
`tests/wad/test_netgame_iwad.py`, and 5-minute 4-player coop and deathmatch runs with no
desync). Still to check by hand on a real LAN or over the internet:

- [ ] Host and join from two computers (`--host` / `--join`), lobby shows both, Enter starts
- [ ] Coop: both players see each other move, open doors, kill monsters; exit to the next map together
- [ ] Coop: one player dies and respawns at a start with weapons and keys kept
- [ ] Deathmatch with `--timer 2`: frags on the status bar, frag table at the intermission, next map
- [ ] One player quits mid-game: "PLAYER n LEFT THE GAME", the others go on
- [ ] Wrong IWAD on the joiner: refused with a message
- [ ] No `DESYNC AT TIC` message during a long game
