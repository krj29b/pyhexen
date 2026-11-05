#version 330 core
// PyHexen - Screen-space sky vertex shader (T4.6).
// Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
// released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

in vec3 in_pos;

uniform mat4 u_view;
uniform mat4 u_proj;

out vec4 v_clip_pos;

void main() {
    vec4 clip_pos = u_proj * u_view * vec4(in_pos, 1.0);
    gl_Position = clip_pos;
    v_clip_pos = clip_pos;
}
