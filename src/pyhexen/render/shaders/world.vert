#version 330 core
// PyHexen - World geometry vertex shader (T4.5).
// Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
// released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

in vec3 in_pos;
in vec2 in_uv;
in float in_light;
in float in_surface_type;
in float in_tex_id;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_light_override;

out vec2 v_uv;
out float v_light;
out float v_surface_type;
out float v_tex_id;
out float v_depth;

void main() {
    vec4 world_pos = u_model * vec4(in_pos, 1.0);
    vec4 view_pos = u_view * world_pos;
    gl_Position = u_proj * view_pos;
    v_uv = in_uv;
    v_light = (u_light_override >= 0.0) ? u_light_override : in_light;
    v_surface_type = in_surface_type;
    v_tex_id = in_tex_id;
    v_depth = max(1.0, -view_pos.z);
}
