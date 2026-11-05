#version 330 core
// PyHexen - Screen-space single/double-layer sky fragment shader (T4.6).
// Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
// released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

in vec4 v_clip_pos;

uniform sampler2D u_sky1_tex;
uniform sampler2D u_sky2_tex;
uniform sampler2D u_palette;

uniform float u_camera_yaw;
uniform float u_fov_x;
uniform float u_pitch_offset;
uniform float u_sky1_offset;
uniform float u_sky2_offset;
uniform int u_double_sky;
uniform int u_use_sky2;
uniform int u_palette_index;
uniform float u_gamma;

out vec4 f_color;

const float TWO_PI = 6.283185307179586;

int sample_sky_layer(sampler2D sky_tex, float col_base, float scroll_offset, float row_f) {
    ivec2 dims = textureSize(sky_tex, 0);
    float tx_f = mod(floor(col_base + scroll_offset), float(dims.x));
    int tx = clamp(int(tx_f), 0, dims.x - 1);
    int ty = clamp(int(floor(row_f)), 0, dims.y - 1);
    return int(round(texelFetch(sky_tex, ivec2(tx, ty), 0).r * 255.0));
}

void main() {
    vec2 ndc = v_clip_pos.xy / max(abs(v_clip_pos.w), 1e-6);

    // Compute horizontal view angle for this screen column (ANGLETOSKYSHIFT: 1024 cols / 2*pi)
    float half_fov = max(u_fov_x, 0.1) * 0.5;
    float ray_angle = u_camera_yaw - atan(ndc.x * tan(half_fov));
    float col_base = (-ray_angle / TWO_PI) * 1024.0;

    // Map vertical NDC [-1..1] (top=+1, bottom=-1) to 200p sky texture row
    float row_f = (1.0 - ndc.y) * 0.5 * 200.0 - u_pitch_offset;

    int color_index = 0;
    if (u_double_sky != 0) {
        // DoubleSky (r_plane.c): sky1 in front, index 0 is transparent revealing sky2
        int idx1 = sample_sky_layer(u_sky1_tex, col_base, u_sky1_offset, row_f);
        int idx2 = sample_sky_layer(u_sky2_tex, col_base, u_sky2_offset, row_f);
        color_index = (idx1 != 0) ? idx1 : idx2;
    } else if (u_use_sky2 != 0) {
        color_index = sample_sky_layer(u_sky2_tex, col_base, u_sky2_offset, row_f);
    } else {
        color_index = sample_sky_layer(u_sky1_tex, col_base, u_sky1_offset, row_f);
    }

    ivec2 pal_dims = textureSize(u_palette, 0);
    int pal_row = clamp(u_palette_index, 0, pal_dims.y - 1);
    vec3 rgb = texelFetch(u_palette, ivec2(color_index, pal_row), 0).rgb;

    if (u_gamma > 0.0 && u_gamma != 1.0) {
        rgb = pow(rgb, vec3(1.0 / u_gamma));
    }

    f_color = vec4(rgb, 1.0);
}
