# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Multiplayer netcode: client-server lockstep over UDP (ADR-013, S2b).

The host runs a `NetServer`; every player (the host included) runs a `NetClient`.
Clients send their ticcmds to the server, the server broadcasts complete tics, and a
`LockstepRunner` feeds them to a `GameSim`. `sim` never imports this package.
"""
