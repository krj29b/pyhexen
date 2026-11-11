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

## ADR-009 — Sim sound requests carry their origin; audio is a Hexen s_sound.c port
**Status:** ACCEPTED
**Context:** Vanilla passes an `mobj_t *` (or a sector/polyobject `soundorg`) to S_StartSound and SN_StartSequence, and the sound code follows that object as it moves. PyHexen events had only a `tid` (0 for most things) and a copied position, so sounds could not follow their source and unrelated sounds replaced each other.
**Decision:** All sim sound requests go through `sim/sound.py`, which emits `SoundEvent(origin=...)` and `SequenceEvent(origin=..., sequence=...)`. `origin=None` means "at the listener", like `S_StartSound(NULL, ...)`. The audio layer (`audio/channels.py`, `audio/game_audio.py`) ports Hexen's s_sound.c: 8 channels, SNDCURVE volume, Hexen separation, one sound per non-player mobj, priority replacement. Sequence delays use an audio-only RNG.
**Consequences:** Audio cannot change the game. Deviation: vanilla shares `M_Random` between sound code and one call in p_user.c, so vanilla demo sync could depend on sounds; PyHexen's sim never sees audio RNG use. Pitch shifting is not implemented (pygame cannot pitch a channel cheaply).

## ADR-010 — HUD messages are player state; seen lines come from a CPU BSP pass
**Status:** ACCEPTED
**Context:** T12.3 built `ui/messages.py` as an event listener with separate normal and yellow slots, so the sim's `Player.message` fields were never shown, "ultimate" messages did not exist, and messages were not saved. Separately, the automap only draws lines flagged `ML_MAPPED`, which vanilla sets in R_StoreWallRange while drawing walls; the GPU renderer has no such step, so nothing was ever mapped.
**Decision:** Messages follow vanilla: `sim/message.py` ports P_SetMessage / P_SetYellowMessage / P_ClearMessage on the player (`message`, `message_tics`, `ultimate_message`, `yellow_message`), counted down in `p_player_think`; `sim.messages_on` is the menu's `messageson`. `ui/messages.py` only draws the console player's message (DrawMessage). `PrintMessage` events are still emitted for listeners. For the automap, `render/line_visibility.py` runs the CPU half of the vanilla wall pass (R_RenderBSPNode, R_CheckBBox, R_AddLine, the solid-seg clip list) at 320 columns once per tic and sets `ML_MAPPED`; nothing is drawn.
**Consequences:** Messages and mapped lines survive save/load and hub revisits, as in vanilla. The visibility pass uses the configured horizontal FOV, so a wider FOV maps lines at the screen edges that vanilla's 90° view would not. It costs about 0.2–0.7 ms per tic on IWAD maps.
