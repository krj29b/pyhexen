# Progress Tracker

Statuses: `TODO` · `IN PROGRESS (<agent>/<date>)` · `DONE` · `BLOCKED (<reason>)`
Rules: claim a task before you start it; when you finish, set it to `DONE` and add notes (gotchas, deviations, follow-ups).
Task details and acceptance criteria are in `IMPLEMENTATION_PLAN.md` §7.

| ID | Task | Deps | Status | Notes |
|---|---|---|---|---|
| T0.1 | Repository scaffold | – | DONE | Package structure, pyproject.toml, GPL-2.0 LICENSE, .gitignore, config.example.toml, version CLI + tests |
| T0.2 | Reference source fetch | T0.1 | DONE | Pinned `chocolate-doom-3.1.1`; use `powershell -ExecutionPolicy Bypass -File ./tools/fetch_reference.ps1` on Windows if execution policy is Restricted |
| T0.3 | Config & IWAD discovery | T0.1 | DONE | `app/config.py` (CLI > `PYHEXEN_IWAD` > `config.toml`) and `iwad_path` fixture in `tests/conftest.py` |
| T0.4 | CI | T0.1 | DONE | `.github/workflows/ci.yml` for Windows + Ubuntu Python 3.12 running ruff and pytest (excluding `wad` and `gl`) |
| T0.5 | Synthetic WAD builder | T0.1 | DONE | `tests/fixtures/wadbuilder.py` with `WadBuilder`, `MapBuilder` (rooms, portals, auto-BSP, blockmap, reject, behavior), picture/palette/texture encoders |
| T1.1 | Fixed-point math | T0.1 | DONE | `core/fixed.py` matching `m_fixed.c` 32-bit wrapping and `FixedDiv` overflow clamping |
| T1.2 | Trig tables & angles | T1.1, T0.2 | DONE | `tools/gen_tables.py` -> `core/tables_data.py` and `core/angles.py` (`point_to_angle2`, `approx_distance`, `slope_div`) |
| T1.3 | Random | T0.2 | DONE | Hexen flat distribution 256-entry `RNDTABLE` from `m_random.c`; `P_Random`/`M_Random` increment index before lookup |
| T1.4 | WAD reader | T0.5 | DONE | `wad/wadfile.py` (`WadFile`, `WadStack`) and `wad/namespaces.py` (`S_`/`SS_`, `F_`/`FF_`, `P_`/`PP_`) |
| T2.1 | Palette & colormaps | T1.4 | DONE | `wad/palette.py` (`load_playpal`, `load_colormap`, `apply_palette`, `save_png`) |
| T2.2 | Picture decoder | T1.4 | DONE | `wad/picture.py` (`decode_picture` with strict bounds validation) |
| T2.3 | Composite wall textures | T2.2 | DONE | `wad/textures.py` (`TextureRegistry` with index 0 = `"-"`, lazy patch composition & caching) |
| T2.4 | Flats | T1.4 | DONE | `wad/flats.py` (`FlatRegistry`; handles both 4096B and >=4096B scrolling Hexen flats) |
| T2.5 | Sprite frame registry | T2.2 | DONE | `wad/sprites.py` (`SpriteRegistry` using `SPRNAMES`, rotation 0 and mirrored 8-char lump names) |
| T2.6 | Scanner & MAPINFO | T1.4 | DONE | `wad/scanner.py` (`sc_man.c` port) and `wad/mapinfo.py` (`MapInfo`, `MapInfoTable`, merges `SNDINFO` `$map` songs) |
| T2.7 | SNDINFO, SNDSEQ, ANIMDEFS | T2.6 | DONE | `wad/sndinfo.py`, `wad/sndseq.py` (`SeqOpcode` instructions), `wad/animdefs.py` (`flat`/`texture` pic & rand frames) |
| T2.8 | wadinfo CLI | T2.1–T2.7 | DONE | `tools/wadinfo.py` (`list`, `extract`, `export-png`, `maps`) — M1 part 1 |
| T3.1 | Map lump parsers | T1.4 | DONE | `wad/maplumps.py` (structured numpy dtypes; rejects Doom maps missing `BEHAVIOR` with `MapFormatError`) |
| T3.2 | Level construction | T3.1, T1.1 | DONE | `world/level.py` (`Level`, `Sector`, `Line`, `Side`, `Seg`, `Subsector`, `Node`, `MapThing`, `P_GroupLines`) |
| T3.3 | BSP & blockmap | T3.2 | DONE | `world/bsp.py`, `world/blockmap.py` (`valid_count` deduplication), `world/reject.py` |
| T3.4 | 2D map viewer | T3.2, T2.6 | DONE | `tools/mapview2d.py` (colour-coded lines/things, pan/zoom, hover tooltips, headless surface export) — **M1 complete** |
| T4.1 | GL bootstrap & main loop | T0.3 | DONE | `render/context.py` (windowed + headless `standalone=True`), `render/camera.py` (y-shearing & true-pitch), `app/main_loop.py` (35 Hz accumulator) |
| T4.2 | GPU texture manager | T4.1, T2.1, T2.3, T2.4 | DONE | `render/textures.py` (`GpuTextureManager`, `R8` index + mask textures, `ANIMDEFS` translation tables) |
| T4.3 | Wall mesh builder | T4.2, T3.2 | DONE | `render/mesh_walls.py` (upper/mid/lower/masked pegging, offsets, `F_SKY` sky hack) |
| T4.4 | Flat mesh builder | T4.2, T3.3 | DONE | `render/mesh_flats.py` (loop tracing + `mapbox_earcut` with holes, BSP-clip fallback) |
| T4.5 | Palette/light shader | T4.2 | DONE | `render/shaders/world.vert/.frag` (wall `scalelight` & flat `zlight` curves, fogmap, palette flash, gamma) |
| T4.6 | Sky | T4.5, T2.6 | DONE | `render/sky.py` & `render/shaders/sky.vert/.frag` (single & double sky, scrolling, `LightningState`) |
| T4.7 | Sector chunks & dynamic rebuild | T4.3, T4.4 | DONE | `render/chunks.py` (`ChunkManager` with cached 2D flat triangulation, ~0.05 ms/sector dirty rebuild) |
| T4.8 | Fly mode & screenshots | T4.6, T4.7 | DONE | `render/renderer.py` (`WorldRenderer`), `__main__.py` (`--fly`, `--screenshot`, `--warp`) — **M2 complete** |
| T5.1 | Info table generator | T0.2 | DONE | `tools/gen_info.py` -> `sim/info_data.py` (300 sprites, 2846 states, 398 mobjs), `@action` registry (`sim/actions/__init__.py`), `tools/report_actions.py` (226 unique actions) |
| T5.2 | Thinkers & ticker | T5.1 | DONE | `core/events.py`, `sim/thinker.py` (circular doubly-linked list + deferred removal), `sim/game_sim.py` (`tick`, `state_hash`) |
| T5.3 | Map objects | T5.2, T3.3 | DONE | `sim/mobj.py` (`Mobj`, `spawn_mobj`, `set_mobj_state` zero-tic loop, `remove_mobj`, sector/blockmap/TID linking) |
| T5.4 | Map thing spawning | T5.3 | DONE | `sim/mobj.py` (`spawn_map_thing`, `spawn_map_things` with skill/class/mode filters, dormant/ambush, player starts) |
| T5.5 | Sprite rendering | T5.3, T2.5, T4.5 | DONE | `render/sprites.py` (`SpriteRenderer`, billboard quads, rotation selection, translucent/shadow sorting) |
| T6.1 | Input → TicCmd | T4.1 | DONE | `core/ticcmd.py` (`TicCmd`, `pack_lookfly`/`unpack_lookfly`) and `app/input.py` (`G_BuildTiccmd` port) |
| T6.2 | Map utilities | T3.3, T1.2 | DONE | `sim/maputl.py` (`point_on_line_side`, `box_on_line_side`, `line_opening`, `intercept_vector`, `path_traverse`) |
| T6.3 | Collision & movement | T6.2, T5.3 | DONE | `sim/map.py` & `sim/movement.py` (`check_position`, `try_move`, `slide_move`, `p_xy_movement`, `p_z_movement`) |
| T6.4 | Player thinker | T6.3, T6.1 | DONE | `sim/player.py` (`Player`, `p_player_think`, `p_move_player`, `p_calc_height`, viewbobbing, jump, fly) |
| T6.5 | Line activation | T6.3 | DONE | `sim/line_activation.py` (`p_activate_line`, `p_use_lines`, all 6 `SPAC_*` modes, repeatable lines) |
| T7.1 | Special dispatcher | T6.5 | DONE | `sim/specials/dispatcher.py` (`execute_line_special`, dynamic sibling handler merging, tag/line helpers) |
| T7.2 | Doors | T7.1 | DONE | `sim/specials/doors.py` (`VerticalDoor`, `Door_Close`, `Door_Open`, `Door_Raise`, `Door_LockedRaise` with key checks) |
| T7.3 | Floors & stairs | T7.1 | DONE | `sim/specials/floors.py` (`FloorMove`, `FloorWaggle`, `t_move_plane`, `Floor_LowerByValue`, `Floor_RaiseByValue`) |
| T7.4 | Ceilings & crushers | T7.1 | DONE | `sim/specials/ceilings.py` (`CeilingMove`, `Ceiling_LowerByValue`, `Ceiling_RaiseByValue`, `Ceiling_CrushAndRaise`) |
| T7.5 | Plats & pillars | T7.1 | DONE | `sim/specials/plats.py` (`Plat`, `Plat_DownWaitUpStay`) and `sim/specials/pillars.py` (`Pillar`, `Pillar_Build`) |
| T7.6 | Lights | T7.1 | DONE | `sim/specials/lights.py` (`LightFade`, `LightGlow`, `Light_ChangeToValue`, `Light_Fade`, `Light_Glow`) |
| T7.7 | Teleports & thing specials | T7.1 | DONE | `sim/specials/teleport.py` (`Teleport`, `Teleport_NewMap`, `Thing_Activate`, `Thing_Deactivate`, `Thing_Remove`) |
| T7.8 | Switches, animations, scrolling | T7.1, T4.2 | DONE | `sim/specials/switches.py` (`SWITCH_LIST`, `ButtonThinker`), `anim.py` (`SurfaceAnimator`), `scroll.py` (`ScrollThinker`) |
| T7.9 | Sector specials | T7.1 | DONE | `sim/specials/sector_specials.py` (damage floors, secrets, current pushers) — **M3 complete** |
| T8.1 | BEHAVIOR loader | T3.1 | DONE | `sim/acs/loader.py` (`parse_behavior`, `OPEN_SCRIPTS_BASE = 1000`, string table) |
| T8.2 | ACS VM core | T8.1, T5.2 | DONE | `sim/acs/vm.py` (`ACSVM`, `RunningScript`, stack machine, script/map/world vars, arithmetic/logic, flow, delay, instruction budget guard) |
| T8.3 | ACS game pcodes | T8.2, T7.1 | DONE | `sim/acs/vm.py` (all 102 Hexen PCD opcodes, line specials, thing counts, tag/script waits, print/sound events, line/sector modifiers) |
| T8.4 | Script lifecycle & ACS store | T8.3 | DONE | `sim/acs/store.py` (`ACSStore`), specials 80-83 in `dispatcher.py`, open script delays (35 tics), sector mover `tag_finished` wakeups |
| T9.1 | Polyobj spawn | T5.4, T7.1 | DONE | `sim/polyobj.py` (`PolyBlock`, `Polyobj`, 4-pass `init_polyobjs`, StartLine & ExplicitLine collection, vertex deduplication, dynamic blockmap linking, static wall exclusion) |
| T9.2 | Polyobj movement | T9.1, T6.3 | DONE | `sim/polyobj.py` (`RotatePolyThinker`, `MovePolyThinker`, `PolyDoorThinker`, `po_move_polyobj`, `po_rotate_polyobj`, `check_mobj_blocking`, `thrust_mobj`, pinned 3-pt crush damage, specials 2..4, 6..8, 90..93 in `dispatcher.py`, ACS `PCD_POLYWAIT` & `poly_finished` in `vm.py`) |
| T9.3 | Polyobj rendering | T9.2, T4.7 | DONE | `render/polyobj.py` (`PolyMesh`, local quads, `u_model` uniform in `world.vert`, 4x4 GL model matrix, shortest-arc sub-tic interpolation, dynamic sector lighting, `WorldRenderer` integration) |
| T10.1 | Damage & death | T6.3 | DONE | `sim/inter.py` (`damage_mobj`, `kill_mobj`, `falling_damage`, `poison_damage`, class armor degradation/absorption, momentum thrust, pain/death state branching) |
| T10.2 | Attacks (hitscan, missiles) | T10.1, T6.2 | DONE | `sim/attacks.py` (`aim_line_attack`, `line_attack`, `spawn_missile`, `spawn_missile_angle`, `seeker_missile`, `radius_attack`, puff/blood spawners), `sim/map.py` (ripper penetration `MF2_RIP`, missile damage collision), `sim/movement.py` (missile explosion audio) |
| T10.3 | Sight | T3.3 | DONE | `sim/sight.py` (`check_sight`, authoritative blockmap DDA raycaster, 75% eye-height, `RejectMatrix` fast sector rejection, 2-sided portal slope aperture clipping, 1-sided instant abort), replaced placeholder in `attacks.py`, `set_reject` in `wadbuilder.py` |
| T10.4 | Monster AI base | T10.2, T10.3 | DONE | `sim/enemy.py` (`noise_alert`, `recursive_sound`, `look_for_players`, `look_for_monsters`, `check_melee_range`, `check_missile_range`, 8-way `move`, `try_walk`, `new_chase_dir`, `A_Look`, `A_Chase`, `A_FastChase`, `A_FaceTarget`, `A_Pain`, `A_Scream`, corpse queue, ice death helpers, invulnerability/reflection flags) |
| T10.5a | Monster actions: common actors | T10.4 | DONE | `sim/actions/monsters_common.py` (Ettin, Afrit/FireDemon, Centaur/Slaughtaur, Chaos Serpent Demon1/2, Wendigo/IceGuy, Stalker/Serpent, floor splash/terrain helpers, visibility/shootability helpers; 68 registered actions total) |
| T10.5b | Monster actions: casters | T10.4 | TODO | |
| T10.5c | Monster actions: bosses | T10.4 | TODO | |
| T10.6 | Weapon framework & psprites | T10.2, T6.4 | TODO | |
| T10.7 | Fighter weapons | T10.6 | TODO | |
| T10.8 | Cleric weapons | T10.6 | TODO | |
| T10.9 | Mage weapons | T10.6 | TODO | |
| T10.10 | Items & inventory | T10.1 | TODO | |
| T10.11 | Environment actors | T10.1 | TODO | **M4** when all of P10 done |
| T11.1 | SFX decode | T1.4 | DONE | `audio/sfx.py` (`decode_dmx_sfx`, 16B lead-in/out pad handling, resampling, `SfxCache`) |
| T11.2 | Sound system | T11.1, T2.7, T5.3 | DONE | `audio/channels.py` (`SoundSystem`, channel limits, distance attenuation, stereo pan, moving origin tracking) |
| T11.3 | Sound sequences | T11.2 | DONE | `audio/sndseq_player.py` (`SoundSequencePlayer`, all `SeqOpcode`s, level override & wind things) |
| T11.4 | MUS→MIDI | T1.4 | DONE | `audio/mus2mid.py` (`mus_to_midi` Format 0 MIDI converter validated with `mido`) |
| T11.5 | Music backend | T11.4, T2.6 | DONE | `audio/music.py` (`MusicManager`, `PygameMidiBackend`, `FluidSynthBackend`, `NullMusicBackend`) |
| T12.1 | Fonts & 2D overlay | T4.2 | DONE | `ui/fonts.py` (`BitmapFont`, `FontSet`), `render/overlay2d.py` (`Overlay2D` 320x200 canvas + 4:3 GPU palette shader) |
| T12.2 | Status bar & inventory bar | T12.1, T10.10 | TODO | |
| T12.3 | Messages | T12.1 | DONE | `ui/messages.py` (`MessageSystem`, normal & yellow/bold messages, tic expiration, multi-line) |
| T12.4 | Menus | T12.1 | TODO | |
| T12.5 | Automap | T12.1, T3.2 | TODO | |
| T12.6 | Intermission & finale | T12.1 | TODO | |
| T13.1 | Map transitions | T7.7, T2.6 | TODO | |
| T13.2 | Serialization framework | P7–P10 | TODO | |
| T13.3 | Hub persistence | T13.1, T13.2, T8.4 | TODO | |
| T13.4 | Save/load UI & files | T13.2, T12.4 | TODO | **M5** |
| T14.1 | Demo record & determinism harness | T6.4 | TODO | can start early |
| T14.2 | Options | T12.4 | TODO | |
| T14.3 | Performance pass | M4 | TODO | |
| T14.4 | Packaging | M5 | TODO | |
| T14.5 | Full playthrough | all | TODO | **M6** |

## Session log
<!-- Append: date · agent · task(s) · one-line summary -->
- 2026-11-05 · Antigravity · T8.2–T8.4 · ACS VM interpreter, 102 game pcodes, lifecycle specials (80-83), ACSStore, open scripts, sector mover tag finished wakeups
- 2026-11-05 · worker_m1 · T9.1 · Polyobject data structures, 4-pass PO_Init, line collection (StartLine special 1 & ExplicitLine special 5), vertex deduplication, dynamic blockmap linking, subsector association, static mesh wall filtering
- 2026-11-05 · worker_m2 · T9.2 · Polyobject movement thinkers (Rotate, Move, Door slide/swing), collision/thrust/crushing physics, line specials 2-4/6-8/90-93 dispatch, mirror propagation, ACS PCD_POLYWAIT & wakeup
- 2026-11-05 · worker_m3_1 · T9.3 · Polyobject GPU rendering: PolyMesh local quads, u_model matrix in world.vert, shortest-arc sub-tic lerp, dynamic sector lighting, WorldRenderer integration, all 160 e2e tests & 450 total tests passing
- 2026-11-07 · worker_t10_1 · T10.1 · Combat damage & death simulation in sim/inter.py: P_DamageMobj, P_KillMobj, class armor tables/formulas (Fighter, Cleric, Mage), fixed-point momentum thrust, pain chance and wake-up aggro, death variants (normal, extreme gib, flame burn, ice freeze/shatter), 62 unit tests passing
- 2026-11-07 · worker_t10_2 · T10.2 · Combat attacks (hitscan, projectiles, rippers, homing seekers, radius damage splash) in sim/attacks.py, sim/map.py, and sim/movement.py, 58 comprehensive tests passing, full 605-test suite 100% clean
- 2026-11-08 · worker_t10_3 · T10.3 · Monster line-of-sight checking in sim/sight.py with RejectMatrix fast rejection, 75% eye-level, blockmap DDA traversal, 2-sided slope aperture clipping, 1-sided early-out; replaced placeholder in attacks.py; 36 tests in tests/unit/test_sight.py, 656 tests passing
- 2026-11-08 · worker_t10_4 · T10.4 · Monster AI base in sim/enemy.py porting p_enemy.c (sound propagation, target acquisition, 8-directional movement with 47000 fixed-point diagonal velocity, chase state machine, shared actions registered via @action), 72 tests in tests/unit/test_enemy_ai.py, full 779-test suite 100% passing
- 2026-11-08 · worker_t10_4_it2 · T10.4 · Remediated iteration 1 gate review defects: purged _fresh_threshold hack from mobj.py/enemy.py/tests, restored authentic C Hexen A_Chase threshold decrement, unconditional PRNG advancement in check_missile_range, SPAC_USE line activation in move, dead code cleanup in look_for_players, updated retaliation target assignment test, 74 tests in test_enemy_ai.py, full 781-test suite 100% clean
- 2026-11-08 · Antigravity · T10.5a · Implemented common monster actions in sim/actions/monsters_common.py (Ettin, Afrit/FireDemon, Centaur/Slaughtaur, Chaos Serpent Demon1/2, Wendigo/IceGuy, Stalker/Serpent, P_GetThingFloorType/P_HitFloor, shared visibility/shootability/sound helpers), 13 comprehensive unit tests in tests/unit/test_monsters_common.py, 794 tests passing
