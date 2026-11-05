# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""ACS (Action Code Script) subsystem package."""

from __future__ import annotations

from pyhexen.sim.acs.loader import (
    ACS_STACK_DEPTH,
    MAX_ACS_MAP_VARS,
    MAX_ACS_SCRIPT_VARS,
    MAX_ACS_STORE,
    MAX_ACS_WORLD_VARS,
    MAX_SCRIPT_ARGS,
    OPEN_SCRIPTS_BASE,
    ACScriptDef,
    ACScriptState,
    ACSError,
    BehaviorData,
    load_behavior,
    parse_behavior,
)

__all__ = [
    "ACS_STACK_DEPTH",
    "ACSError",
    "ACScriptDef",
    "ACScriptState",
    "BehaviorData",
    "MAX_ACS_MAP_VARS",
    "MAX_ACS_SCRIPT_VARS",
    "MAX_ACS_STORE",
    "MAX_ACS_WORLD_VARS",
    "MAX_SCRIPT_ARGS",
    "OPEN_SCRIPTS_BASE",
    "load_behavior",
    "parse_behavior",
]
