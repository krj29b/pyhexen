# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.
#
# Build environment for the Chocolate Hexen trace build (T15.6). No game data in here:
# the reference source and the IWAD are mounted read-only at run time.
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config \
      libsdl2-dev libsdl2-mixer-dev libsdl2-net-dev \
      xvfb xauth gdb \
 && rm -rf /var/lib/apt/lists/*
COPY pyh_trace.c build_trace.sh /tools/
ENTRYPOINT ["/bin/sh", "/tools/build_trace.sh"]
