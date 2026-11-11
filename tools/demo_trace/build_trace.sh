#!/bin/sh
# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.
#
# Runs inside the trace container (T15.6). Mounts:
#   /ref  Chocolate Doom source (reference/), read-only
#   /wad  directory holding the IWAD, read-only (IWAD file name in $IWAD_NAME)
#   /out  trace output directory (also caches the build in /out/.build)
# Arguments: demo lump names (default DEMO1 DEMO2 DEMO3).
# PYHEXEN_TRACE_DUMP (comma-separated leveltimes) is passed through to the hook.
set -eu
SRC=/out/.build/src
BIN=/out/.build/bin
if [ ! -x "$BIN/src/chocolate-hexen" ] || [ /tools/pyh_trace.c -nt "$BIN/src/chocolate-hexen" ]; then
    rm -rf "$SRC"
    mkdir -p "$SRC" "$BIN"
    cp -r /ref/. "$SRC"
    rm -rf "$SRC/.git"
    # A Windows checkout has CRLF line endings; the sed patterns below need LF.
    find "$SRC" -type f \( -name '*.c' -o -name '*.h' \) -exec sed -i 's/\r$//' {} +
    TICK="$SRC/src/hexen/p_tick.c"
    grep -q 'leveltime++;' "$TICK"
    sed -i 's/^#include "p_local.h"$/#include "p_local.h"\nstatic void PYH_TraceTic(void);/' "$TICK"
    sed -i 's/^\(\s*\)leveltime++;$/\1leveltime++;\n\1PYH_TraceTic();/' "$TICK"
    grep -q 'PYH_TraceTic();' "$TICK"
    cat /tools/pyh_trace.c >> "$TICK"
    # Hexen v1.0 had 4 player slots (its demo headers are 3 + 4*2 bytes); v1.1 raised
    # MAXPLAYERS to 8. Chocolate only knows v1.1, so with the v1.0 IWAD it would read
    # tic bytes as extra players. Use 4 slots when the IWAD is v1.0.
    MAIN="$SRC/src/hexen/h2_main.c"
    grep -q '// The v1.0 IWAD file is missing' "$MAIN"
    sed -i '/\/\/ The v1.0 IWAD file is missing/i\
    if (gamemode != shareware \&\& W_CheckNumForName("CLUS1MSG") < 0)\
    {\
        maxplayers = 4;\
    }\
' "$MAIN"
    grep -q 'W_CheckNumForName("CLUS1MSG") < 0)$' "$MAIN"
    cmake -S "$SRC" -B "$BIN" -G Ninja -DCMAKE_BUILD_TYPE=Release >/out/.build/cmake.log
    ninja -C "$BIN" chocolate-hexen >/out/.build/ninja.log
fi
[ $# -gt 0 ] || set -- DEMO1 DEMO2 DEMO3
# The v1.0 IWAD lacks the cluster text lumps (CLUS1MSG...), which Chocolate Hexen
# refuses without -v10override. Demos never reach a cluster message, so allow it.
EXTRA_ARGS=""
if ! grep -q CLUS1MSG "/wad/$IWAD_NAME"; then
    EXTRA_ARGS="-v10override"
fi
export SDL_AUDIODRIVER=dummy SDL_RENDER_DRIVER=software HOME=/tmp
for demo in "$@"; do
    echo "tracing $demo"
    PYHEXEN_TRACE="/out/$demo.trace" \
        xvfb-run -a "$BIN/src/chocolate-hexen" -iwad "/wad/$IWAD_NAME" -playdemo "$demo" \
        -nosound -nomusic -nograbmouse $EXTRA_ARGS >"/out/$demo.log" 2>&1 || true
    tail -n 2 "/out/$demo.log"
    wc -l "/out/$demo.trace"
done
