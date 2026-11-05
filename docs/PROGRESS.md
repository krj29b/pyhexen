# Progress Tracker

Statuses: `TODO` · `IN PROGRESS (<agent>/<date>)` · `DONE` · `BLOCKED (<reason>)`
Rules: claim a task before you start it; when you finish, set it to `DONE` and add notes (gotchas, deviations, follow-ups).
Task details and acceptance criteria are in `IMPLEMENTATION_PLAN.md` §7.

| ID | Task | Deps | Status | Notes |
|---|---|---|---|---|
| T0.1 | Repository scaffold | – | DONE | Package structure, pyproject.toml, GPL-2.0 LICENSE, .gitignore, config.example.toml, version CLI + tests |
| T0.2 | Reference source fetch | T0.1 | IN PROGRESS (lead/2026-11-05) | |
| T0.3 | Config & IWAD discovery | T0.1 | TODO | |
| T0.4 | CI | T0.1 | TODO | |
| T0.5 | Synthetic WAD builder | T0.1 | TODO | |
| T1.1 | Fixed-point math | T0.1 | TODO | |
| T1.2 | Trig tables & angles | T1.1, T0.2 | TODO | |
| T1.3 | Random | T0.2 | TODO | |
| T1.4 | WAD reader | T0.5 | TODO | |
| T2.1 | Palette & colormaps | T1.4 | TODO | |
| T2.2 | Picture decoder | T1.4 | TODO | |
| T2.3 | Composite wall textures | T2.2 | TODO | |
| T2.4 | Flats | T1.4 | TODO | |
| T2.5 | Sprite frame registry | T2.2 | TODO | needs T5.1 sprite names (temporary list ok) |
| T2.6 | Scanner & MAPINFO | T1.4 | TODO | |
| T2.7 | SNDINFO, SNDSEQ, ANIMDEFS | T2.6 | TODO | |
| T2.8 | wadinfo CLI | T2.1–T2.7 | TODO | M1 part 1 |
| T3.1 | Map lump parsers | T1.4 | TODO | |
| T3.2 | Level construction | T3.1, T1.1 | TODO | |
| T3.3 | BSP & blockmap | T3.2 | TODO | |
| T3.4 | 2D map viewer | T3.2, T2.6 | TODO | **M1** |
| T4.1 | GL bootstrap & main loop | T0.3 | TODO | |
| T4.2 | GPU texture manager | T4.1, T2.1, T2.3, T2.4 | TODO | |
| T4.3 | Wall mesh builder | T4.2, T3.2 | TODO | |
| T4.4 | Flat mesh builder | T4.2, T3.3 | TODO | |
| T4.5 | Palette/light shader | T4.2 | TODO | |
| T4.6 | Sky | T4.5, T2.6 | TODO | |
| T4.7 | Sector chunks & dynamic rebuild | T4.3, T4.4 | TODO | |
| T4.8 | Fly mode & screenshots | T4.6, T4.7 | TODO | **M2** |
| T5.1 | Info table generator | T0.2 | TODO | |
| T5.2 | Thinkers & ticker | T5.1 | TODO | |
| T5.3 | Map objects | T5.2, T3.3 | TODO | |
| T5.4 | Map thing spawning | T5.3 | TODO | |
| T5.5 | Sprite rendering | T5.3, T2.5, T4.5 | TODO | |
| T6.1 | Input → TicCmd | T4.1 | TODO | |
| T6.2 | Map utilities | T3.3, T1.2 | TODO | |
| T6.3 | Collision & movement | T6.2, T5.3 | TODO | |
| T6.4 | Player thinker | T6.3, T6.1 | TODO | |
| T6.5 | Line activation | T6.3 | TODO | |
| T7.1 | Special dispatcher | T6.5 | TODO | |
| T7.2 | Doors | T7.1 | TODO | |
| T7.3 | Floors & stairs | T7.1 | TODO | |
| T7.4 | Ceilings & crushers | T7.1 | TODO | |
| T7.5 | Plats & pillars | T7.1 | TODO | |
| T7.6 | Lights | T7.1 | TODO | |
| T7.7 | Teleports & thing specials | T7.1 | TODO | |
| T7.8 | Switches, animations, scrolling | T7.1, T4.2 | TODO | |
| T7.9 | Sector specials | T7.1 | TODO | **M3** |
| T8.1 | BEHAVIOR loader | T3.1 | TODO | |
| T8.2 | ACS VM core | T8.1, T5.2 | TODO | |
| T8.3 | ACS game pcodes | T8.2, T7.1 | TODO | |
| T8.4 | Script lifecycle & ACS store | T8.3 | TODO | |
| T9.1 | Polyobj spawn | T5.4, T7.1 | TODO | |
| T9.2 | Polyobj movement | T9.1, T6.3 | TODO | |
| T9.3 | Polyobj rendering | T9.2, T4.7 | TODO | |
| T10.1 | Damage & death | T6.3 | TODO | |
| T10.2 | Attacks (hitscan, missiles) | T10.1, T6.2 | TODO | |
| T10.3 | Sight | T3.3 | TODO | |
| T10.4 | Monster AI base | T10.2, T10.3 | TODO | |
| T10.5a | Monster actions: common actors | T10.4 | TODO | |
| T10.5b | Monster actions: casters | T10.4 | TODO | |
| T10.5c | Monster actions: bosses | T10.4 | TODO | |
| T10.6 | Weapon framework & psprites | T10.2, T6.4 | TODO | |
| T10.7 | Fighter weapons | T10.6 | TODO | |
| T10.8 | Cleric weapons | T10.6 | TODO | |
| T10.9 | Mage weapons | T10.6 | TODO | |
| T10.10 | Items & inventory | T10.1 | TODO | |
| T10.11 | Environment actors | T10.1 | TODO | **M4** when all of P10 done |
| T11.1 | SFX decode | T1.4 | TODO | |
| T11.2 | Sound system | T11.1, T2.7, T5.3 | TODO | |
| T11.3 | Sound sequences | T11.2 | TODO | |
| T11.4 | MUS→MIDI | T1.4 | TODO | |
| T11.5 | Music backend | T11.4, T2.6 | TODO | |
| T12.1 | Fonts & 2D overlay | T4.2 | TODO | |
| T12.2 | Status bar & inventory bar | T12.1, T10.10 | TODO | |
| T12.3 | Messages | T12.1 | TODO | |
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
