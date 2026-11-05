#version 330 core
// PyHexen - Authentic 8-bit indexed palette & colormap world fragment shader (T4.5).
// Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
// released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

in vec2 v_uv;
in float v_light;
in float v_surface_type;
in float v_tex_id;
in float v_depth;

uniform sampler2D u_index_tex;
uniform sampler2D u_mask_tex;
uniform sampler2D u_palette;
uniform sampler2D u_colormap;
uniform sampler2D u_fogmap;

uniform vec2 u_tex_size;
uniform int u_palette_index;
uniform int u_extralight;
uniform int u_fixed_colormap;
uniform int u_use_fogmap;
uniform float u_gamma;

out vec4 f_color;

// Lighting constants from r_local.h / r_main.c:
// LIGHTLEVELS = 16, LIGHTSEGSHIFT = 4, MAXLIGHTSCALE = 48,
// MAXLIGHTZ = 128, LIGHTZSHIFT = 20 (16 map units), NUMCOLORMAPS = 32, DISTMAP = 2.
int compute_colormap_row(float light_level, float depth, float surface_type) {
    if (u_fixed_colormap >= 0) {
        return clamp(u_fixed_colormap, 0, 31);
    }
    int lightnum = clamp((int(light_level) >> 4) + u_extralight, 0, 15);
    int startmap = (15 - lightnum) * 4;
    int diminish = 0;

    if (surface_type > 0.5 && surface_type < 1.5) {
        // Flat distance curve (zlight in R_InitLightTables / R_MapPlane)
        int z_index = clamp(int(depth) >> 4, 0, 127);
        int flat_scale = 160 / (z_index + 1);
        diminish = flat_scale / 2;
    } else {
        // Wall distance curve (scalelight in R_ExecuteSetViewSize / R_RenderSegLoop)
        int wall_scale = clamp(int(2560.0 / max(depth, 1.0)), 0, 47);
        diminish = wall_scale / 2;
    }
    return clamp(startmap - diminish, 0, 31);
}

void main() {
    if (v_tex_id < -1e30) {
        discard;
    }
    ivec2 tex_dims = textureSize(u_index_tex, 0);
    vec2 size = (u_tex_size.x > 0.0 && u_tex_size.y > 0.0) ? u_tex_size : vec2(tex_dims);

    // Surface type 3.0 = masked mid-wall (no vertical repeat)
    if (v_surface_type > 2.5) {
        if (v_uv.y < 0.0 || v_uv.y >= size.y) {
            discard;
        }
    }

    float wrapped_u = mod(floor(v_uv.x), size.x);
    float wrapped_v = (v_surface_type > 2.5)
        ? clamp(floor(v_uv.y), 0.0, size.y - 1.0)
        : mod(floor(v_uv.y), size.y);

    ivec2 tex_coord = clamp(ivec2(int(wrapped_u), int(wrapped_v)), ivec2(0), tex_dims - ivec2(1));

    float mask_val = texelFetch(u_mask_tex, tex_coord, 0).r;
    if (mask_val < 0.5) {
        discard;
    }

    int raw_index = int(round(texelFetch(u_index_tex, tex_coord, 0).r * 255.0));
    int cmap_row = compute_colormap_row(v_light, v_depth, v_surface_type);

    float shaded_norm = (u_use_fogmap != 0)
        ? texelFetch(u_fogmap, ivec2(raw_index, cmap_row), 0).r
        : texelFetch(u_colormap, ivec2(raw_index, cmap_row), 0).r;
    int shaded_index = int(round(shaded_norm * 255.0));

    ivec2 pal_dims = textureSize(u_palette, 0);
    int pal_row = clamp(u_palette_index, 0, pal_dims.y - 1);
    vec3 rgb = texelFetch(u_palette, ivec2(shaded_index, pal_row), 0).rgb;

    if (u_gamma > 0.0 && u_gamma != 1.0) {
        rgb = pow(rgb, vec3(1.0 / u_gamma));
    }

    f_color = vec4(rgb, 1.0);
}
