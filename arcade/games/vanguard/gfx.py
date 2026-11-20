"""Vanguard's 3D view, drawn by the graphics card (OpenGL 3.3 through Qt's own classes, no other package) at the
window's own resolution into an image the game widget paints.

* The map is one buffer of coloured triangles built from its grid (floors, walls, slabs); every material's pattern
  (paving, planks, plaster with windows, rock strata) is drawn by the fragment shader from world coordinates, so
  there are no texture files and nothing to stretch.
* Everything else (heroes, weapons, the cart, scenery, shots, barriers) is *instances* of one shape: a rounded box
  that the vertex shader can also bend into a cylinder or a ball (models.py makes the rows).
* The sun casts shadows from two depth pictures: the map's (drawn once) and the moving things' (every frame).
* Heroes get a team-coloured outline from the stencil buffer; a revealed enemy shows through walls.
* Translucent things (barriers, zones, beams), additive glows, the sky, the player's own weapon and a vignette follow.

The picture is drawn upside down on purpose so it can be read back without a flip."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from PySide6.QtGui import (QImage, QMatrix4x4, QOffscreenSurface, QOpenGLContext, QOpenGLExtraFunctions,
                           QSurfaceFormat)
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
                              QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject)

from .maps import NO_SLAB, MapData
from .models import ROW

log = logging.getLogger(__name__)
F32 = np.float32
VERT = 13  # position 3, normal 3, colour 3, material / occlusion / wall top / wall bottom
MAX_LIGHTS = 8
GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_FLOAT = 0x0004, 0x0005, 0x1406
GL_DEPTH_TEST, GL_CULL_FACE, GL_BLEND, GL_STENCIL_TEST = 0x0B71, 0x0B44, 0x0BE2, 0x0B90
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_STENCIL_BUFFER_BIT = 0x4000, 0x0100, 0x0400
GL_LEQUAL, GL_LESS, GL_ALWAYS, GL_NOTEQUAL = 0x0203, 0x0201, 0x0207, 0x0205
GL_KEEP, GL_REPLACE = 0x1E00, 0x1E01
GL_ZERO, GL_ONE, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = 0, 1, 0x0302, 0x0303
GL_RGB8, GL_R32F = 0x8051, 0x822E
GL_TEXTURE_2D, GL_TEXTURE0 = 0x0DE1, 0x84C0
GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST = 0x2801, 0x2800, 0x2600
GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE = 0x2802, 0x2803, 0x812F

# --- shaders ---------------------------------------------------------------------------------------------------------
_LIGHT = """
uniform vec3 u_cam; uniform vec3 u_sun; uniform vec3 u_sun_col; uniform vec3 u_amb_sky; uniform vec3 u_amb_gnd;
uniform vec3 u_fog; uniform float u_fogd; uniform float u_shadows; uniform vec2 u_sh_size;
uniform mat4 u_lvp_s; uniform mat4 u_lvp_d; uniform sampler2D u_sh_s; uniform sampler2D u_sh_d;
uniform vec4 u_lpos[8]; uniform vec3 u_lcol[8]; uniform int u_nl;
float sh_tap(sampler2D t, vec2 uv, float z, float size) {
    vec2 st = uv * size - 0.5; vec2 f = fract(st); vec2 b = (floor(st) + 0.5) / size; float px = 1.0 / size;
    float s00 = step(z, texture(t, b).r), s10 = step(z, texture(t, b + vec2(px, 0.0)).r);
    float s01 = step(z, texture(t, b + vec2(0.0, px)).r), s11 = step(z, texture(t, b + vec2(px, px)).r);
    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}
float sh_map(sampler2D t, mat4 m, vec3 p, float bias, float size) {
    vec4 q = m * vec4(p, 1.0);
    vec3 c = q.xyz * 0.5 + 0.5;
    if (c.x <= 0.0 || c.x >= 1.0 || c.y <= 0.0 || c.y >= 1.0 || c.z >= 1.0) return 1.0;
    float z = c.z - bias, o = 0.8 / size;
    return 0.25 * (sh_tap(t, c.xy + vec2(-o, -o * 0.4), z, size) + sh_tap(t, c.xy + vec2(o, o * 0.4), z, size)
                 + sh_tap(t, c.xy + vec2(-o * 0.4, o), z, size) + sh_tap(t, c.xy + vec2(o * 0.4, -o), z, size));
}
vec3 shade(vec3 alb, vec3 p, vec3 n, float ao, float emis) {
    float nd = dot(n, u_sun);
    float sh = 1.0;
    if (u_shadows > 0.5 && nd > -0.05) {
        vec3 q = p + n * 0.06;
        sh = min(sh_map(u_sh_s, u_lvp_s, q, 0.0007, u_sh_size.x), sh_map(u_sh_d, u_lvp_d, q, 0.0012, u_sh_size.y));
    }
    float diff = smoothstep(-0.08, 0.3, nd) * 0.68 + max(nd, 0.0) * 0.32;
    vec3 c = alb * (mix(u_amb_gnd, u_amb_sky, n.y * 0.5 + 0.5) * ao + u_sun_col * diff * sh * (0.6 + 0.4 * ao));
    for (int i = 0; i < 8; i++) {
        if (i >= u_nl) break;
        vec3 d = u_lpos[i].xyz - p; float dist = length(d);
        float a = clamp(1.0 - dist / u_lpos[i].w, 0.0, 1.0);
        c += alb * u_lcol[i] * a * a * (0.35 + 0.65 * max(dot(n, d / max(dist, 1e-4)), 0.0));
    }
    vec3 v = normalize(u_cam - p);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    c += alb * u_amb_sky * rim * 0.4;
    return c + alb * emis;
}
vec3 finish(vec3 c, vec3 p) {
    float f = 1.0 - exp(-pow(length(p - u_cam) / u_fogd, 1.5));
    c = mix(c, u_fog, min(f, 0.92));
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(l), c, 1.14);
    vec3 e = max(c - 0.8, 0.0);
    return min(c, 0.8) + e / (1.0 + e * 2.6);
}
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
"""

_WORLD_VS = """#version 330 core
layout(location = 0) in vec3 a_pos; layout(location = 1) in vec3 a_nrm; layout(location = 2) in vec3 a_col;
layout(location = 3) in vec4 a_info;
uniform mat4 u_vp;
out vec3 v_pos; out vec3 v_nrm; out vec3 v_col; flat out vec4 v_info; out float v_ao;
void main() {
    v_pos = a_pos; v_nrm = a_nrm; v_col = a_col; v_info = a_info; v_ao = a_info.y;
    gl_Position = u_vp * vec4(a_pos, 1.0);
}
"""

_WORLD_FS = "#version 330 core\n" + _LIGHT + """
in vec3 v_pos; in vec3 v_nrm; in vec3 v_col; flat in vec4 v_info; in float v_ao;
uniform float u_dusk;
out vec4 o;
float line(float x, float w) {  // 1 on a line at every whole number of x
    float d = abs(fract(x - 0.5) - 0.5), fw = fwidth(x);
    return 1.0 - smoothstep(w - fw, w + fw, d);
}
void main() {
    vec3 n = normalize(v_nrm);
    vec3 p = v_pos;
    bool wall = abs(n.y) < 0.5;
    vec2 uv = wall ? (abs(n.x) > 0.5 ? vec2(p.z, p.y) : vec2(p.x, p.y)) : p.xz;
    int m = int(v_info.x + 0.5);
    float top = v_info.z, bot = v_info.w, dist = length(p - u_cam);
    float fade = clamp(1.0 - dist / 70.0, 0.0, 1.0);  // fine patterns fade out far away instead of shimmering
    vec3 c = v_col * (0.95 + 0.1 * vnoise(uv * 0.35));
    float emis = 0.0;
    float ao = v_ao;
    if (m == 0) {  // plaster: windows on tall walls, a plinth, a cornice
        if (wall) {
            float hgt = top - bot, y = p.y - bot;
            if (hgt > 4.4) {
                float fy = y / 3.0; vec2 cell = vec2(floor(uv.x / 2.4), floor(fy));
                vec2 q = vec2(fract(uv.x / 2.4) - 0.5, fract(fy) - 0.56);
                if (top - p.y > 0.9) {
                    vec2 e = fwidth(q);
                    float fr = (1.0 - smoothstep(0.27 - e.x, 0.27 + e.x, abs(q.x)))
                             * (1.0 - smoothstep(0.27 - e.y, 0.27 + e.y, abs(q.y)));
                    float gl = (1.0 - smoothstep(0.21 - e.x, 0.21 + e.x, abs(q.x)))
                             * (1.0 - smoothstep(0.215 - e.y, 0.215 + e.y, abs(q.y)));
                    float lit = step(mix(0.9, 0.45, u_dusk), hash(cell + 3.0));
                    vec3 glass = mix(vec3(0.20, 0.30, 0.42), vec3(0.55, 0.72, 0.86), clamp(q.y * 2.0 + 0.5, 0.0, 1.0));
                    glass = mix(glass, vec3(1.0, 0.82, 0.5), lit * 0.85);
                    glass *= 1.0 - 0.6 * max(line(q.x * 2.0 + 0.5, 0.06), line(q.y * 2.3 + 0.5, 0.05)) * fade;
                    c = mix(c, vec3(0.97, 0.95, 0.90), fr * fade);
                    c = mix(c, glass, gl * fade);
                    emis += lit * gl * fade * mix(0.1, 0.7, u_dusk);
                    c *= 1.0 - 0.10 * line(fy, 0.012) * fade;
                }
            }
            c *= mix(0.78, 1.0, smoothstep(0.42, 0.46, y));
            c = mix(c, vec3(0.98, 0.96, 0.92), (1.0 - smoothstep(0.26, 0.3, top - p.y)) * 0.75);
        }
    } else if (m == 1) {  // paving
        vec2 t = uv / 1.5;
        c *= 0.94 + 0.12 * hash(floor(t));
        c *= 1.0 - 0.2 * max(line(t.x, 0.02), line(t.y, 0.02)) * fade;
    } else if (m == 2) {  // cobbles
        vec2 t = vec2(uv.x / 0.7, uv.y / 0.35); t.x += 0.5 * mod(floor(t.y), 2.0);
        c *= 0.88 + 0.22 * hash(floor(t));
        c *= 1.0 - 0.24 * max(line(t.x, 0.05), line(t.y, 0.07)) * fade;
    } else if (m == 3) {  // planks
        vec2 t = vec2(uv.x / 2.2, uv.y / 0.28); t.x += 0.37 * floor(t.y);
        c *= 0.9 + 0.2 * hash(vec2(floor(t.y), floor(t.x)));
        c *= 1.0 - 0.26 * max(line(t.y, 0.035), line(t.x, 0.006)) * fade;
    } else if (m == 4) {  // grass
        c *= 0.86 + 0.26 * vnoise(uv * 1.7) + 0.1 * (hash(floor(uv * 14.0)) - 0.5) * fade;
    } else if (m == 5) {  // sand
        c *= 0.93 + 0.12 * vnoise(uv * 0.9) + 0.05 * sin(uv.x * 5.0 + vnoise(uv) * 6.0) * fade;
    } else if (m == 6) {  // metal plate
        vec2 t = uv / 2.0;
        c *= 0.95 + 0.08 * hash(floor(t));
        c *= 1.0 - 0.3 * max(line(t.x, 0.012), line(t.y, 0.012)) * fade;
        vec2 r = abs(fract(t) - 0.5);
        c *= 1.0 - 0.25 * (1.0 - smoothstep(0.02, 0.035, length(r - 0.42))) * fade;
    } else if (m == 7) {  // roof tiles
        vec2 t = vec2(uv.x / 0.5, uv.y / 0.5); t.x += 0.5 * mod(floor(t.y), 2.0);
        c *= 0.82 + 0.3 * fract(t.y) * mix(1.0, 0.6, 1.0 - fade) + 0.1 * hash(floor(t));
        c *= 1.0 - 0.15 * line(t.x, 0.04) * fade;
    } else if (m == 8) {  // crate
        vec2 q = fract(uv);
        float edge = min(min(q.x, 1.0 - q.x), min(q.y, 1.0 - q.y));
        float frame = 1.0 - smoothstep(0.09, 0.1, edge);
        float brace = wall ? (1.0 - smoothstep(0.05, 0.06, abs(q.x - q.y))) : 0.0;
        c *= 1.0 - 0.14 * line(q.y * 4.0, 0.03) * fade;
        c = mix(c, c * 0.7, max(frame, brace));
        c = mix(c, c * 1.25, (1.0 - smoothstep(0.015, 0.02, edge)) * 0.6);
    } else if (m == 9) {  // rock: strata on the faces
        if (wall) {
            float band = p.y * 0.9 + vnoise(uv * vec2(0.22, 0.0) + 4.0) * 1.6;
            c *= 0.84 + 0.3 * hash(vec2(floor(band), 7.0));
            c *= 1.0 - 0.2 * line(band, 0.03) * fade;
            c *= 0.94 + 0.1 * vnoise(uv * 2.2);
        } else {
            c *= 0.88 + 0.2 * vnoise(uv * 0.8);
        }
    } else if (m == 10) {  // concrete
        vec2 t = uv / 3.0;
        c *= 0.95 + 0.07 * hash(floor(t)) + 0.05 * (hash(floor(uv * 9.0)) - 0.5) * fade;
        c *= 1.0 - 0.22 * max(line(t.x, 0.008), line(t.y, 0.008)) * fade;
    } else if (m == 12) {  // hazard stripes
        c = mix(c, vec3(0.12), step(0.5, fract((uv.x + uv.y) * 0.8)));
    } else if (m == 13) {
        c *= 0.85 + 0.3 * hash(floor(uv * 6.0));
    }
    if (wall) {
        ao *= mix(0.66, 1.0, smoothstep(0.0, 0.8, p.y - bot));
        c *= 1.0 + 0.1 * (1.0 - smoothstep(0.04, 0.08, top - p.y));
    }
    o = vec4(finish(shade(c, p, n, ao, emis), p), 1.0);
}
"""

# one rounded box / cylinder per instance; the mesh is a cube whose outer bands bend round the edges
_PART_VS_HEAD = """#version 330 core
layout(location = 0) in vec3 a_uvw;
layout(location = 1) in vec4 i_pos; layout(location = 2) in vec4 i_ax; layout(location = 3) in vec4 i_ay;
layout(location = 4) in vec4 i_az; layout(location = 5) in vec4 i_col; layout(location = 6) in vec4 i_par;
uniform mat4 u_vp; uniform float u_grow; uniform vec3 u_eye; uniform float u_only;
out vec3 v_pos; out vec3 v_nrm; out vec4 v_col; out vec4 v_par; out vec3 v_loc;
void main() {
    vec3 hs = vec3(i_ax.w, i_ay.w, i_az.w) * 0.5;
    vec3 e = sign(a_uvw) * max(abs(a_uvw) - 0.5, 0.0) * 2.0;
    vec3 f = clamp(a_uvw * 2.0, -1.0, 1.0);
    vec3 loc, n;
    if (i_par.z < 0.5) {
        float r = min(i_pos.w, min(hs.x, min(hs.y, hs.z)));
        float l = length(e);
        n = l > 1e-5 ? e / l : vec3(0.0, 1.0, 0.0);
        loc = f * (hs - r) + n * r;
    } else {
        float rr = min(hs.x, hs.z);
        vec2 d = e.xz; float l = length(d);
        vec2 dir = l > 1e-5 ? d / l : vec2(0.0);
        if (abs(a_uvw.y) > 0.999) {
            float m = max(abs(d.x), abs(d.y));
            loc = vec3(dir.x * rr * m, sign(a_uvw.y) * hs.y, dir.y * rr * m);
            n = vec3(0.0, sign(a_uvw.y), 0.0);
        } else {
            loc = vec3(dir.x * rr, f.y * hs.y, dir.y * rr);
            n = vec3(dir.x, 0.0, dir.y);
        }
    }
    vec3 wn = normalize(i_ax.xyz * n.x + i_ay.xyz * n.y + i_az.xyz * n.z);
    vec3 wp = i_pos.xyz + i_ax.xyz * loc.x + i_ay.xyz * loc.y + i_az.xyz * loc.z;
    if (u_grow > 0.0) wp += wn * u_grow * (0.6 + length(wp - u_eye) * 0.045);
    v_pos = wp; v_nrm = wn; v_col = i_col; v_par = i_par; v_loc = loc / max(hs, vec3(1e-4));
    gl_Position = u_vp * vec4(wp, 1.0);
    if (u_only > 0.0 && (i_par.x < u_only - 0.5)) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}
"""

_PART_FS = "#version 330 core\n" + _LIGHT + """
in vec3 v_pos; in vec3 v_nrm; in vec4 v_col; in vec4 v_par; in vec3 v_loc;
uniform float u_shadowed;
out vec4 o;
void main() {
    vec3 n = normalize(v_nrm);
    vec3 c = v_col.rgb * (0.9 + 0.1 * n.y);
    if (v_par.w > 0.5 && v_par.w < 1.5) c = mix(c, vec3(0.1, 0.1, 0.12), step(0.5, fract((v_loc.x + v_loc.y + v_loc.z) * 1.6)));
    o = vec4(finish(shade(c, v_pos, n, 1.0, v_col.a), v_pos), 1.0);
}
"""

_OUTLINE_FS = """#version 330 core
in vec4 v_par; uniform vec3 u_ally; uniform vec3 u_enemy; uniform float u_alpha; out vec4 o;
void main() { o = vec4(v_par.x > 1.5 ? u_enemy : u_ally, u_alpha); }
"""

_DEPTH_FS = """#version 330 core
out vec4 o;
void main() { o = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0); }
"""

_DEPTH_WORLD_VS = """#version 330 core
layout(location = 0) in vec3 a_pos; uniform mat4 u_vp;
void main() { gl_Position = u_vp * vec4(a_pos, 1.0); }
"""

_ALPHA_FS = """#version 330 core
in vec3 v_pos; in vec3 v_nrm; in vec4 v_col; in vec4 v_par; in vec3 v_loc;
uniform vec3 u_cam; uniform float u_time; uniform float u_add;
out vec4 o;
void main() {
    vec3 n = normalize(v_nrm);
    vec3 v = normalize(u_cam - v_pos);
    float fres = pow(1.0 - abs(dot(n, v)), 2.0);
    float a = v_par.y;
    vec3 c = v_col.rgb * (1.0 + v_col.a);
    int pat = int(v_par.w + 0.5);
    if (pat == 1) {  // a hard-light barrier: honeycomb and a bright rim
        vec2 q = v_pos.xy * 1.6 + v_pos.zy * 1.6;
        q.x += q.y * 0.5;
        vec2 g = abs(fract(q) - 0.5);
        float hexl = smoothstep(0.40, 0.48, max(g.x, g.y));
        float sweep = 0.5 + 0.5 * sin(v_pos.y * 3.0 - u_time * 3.0);
        a = a * (0.45 + 0.8 * hexl + 0.9 * fres + 0.15 * sweep);
        c *= 0.8 + 0.6 * hexl + fres;
    } else if (pat == 2) {  // a ring on the ground
        float r = length(v_loc.xz);
        float wd = clamp(0.35 / max(v_par.z * 0.0 + length(fwidth(v_loc.xz)) * 60.0, 1e-3), 0.0, 1.0) * 0.0;
        float ring = smoothstep(0.93, 0.95, r) * (1.0 - smoothstep(0.985, 1.0, r)) + wd;
        float ang = atan(v_loc.z, v_loc.x);
        float dash = 0.7 + 0.3 * step(0.5, fract(ang * 3.82 + u_time * 0.25));
        a = a * (ring * dash + 0.10 * smoothstep(0.2, 0.95, r) * (1.0 - smoothstep(0.985, 1.0, r)));
        if (abs(n.y) < 0.5) a = 0.0;
    } else if (pat == 3) {  // a beam or a blast: soft and bright in the middle
        a = a * (0.35 + 0.65 * pow(abs(dot(n, v)), 1.5));
    } else if (pat == 4) {  // a spawn room's door
        float s = 0.5 + 0.5 * sin(v_pos.y * 9.0 - u_time * 4.0);
        a = a * (0.5 + 0.5 * s + fres);
    } else {
        a = a * (0.7 + 0.6 * fres);
    }
    a = clamp(a, 0.0, 1.0);
    o = u_add > 0.5 ? vec4(c * a, 1.0) : vec4(c, a);
}
"""

_GLOW_VS = """#version 330 core
layout(location = 0) in vec4 a_p; layout(location = 1) in vec4 a_c;
uniform mat4 u_vp; uniform vec3 u_right; uniform vec3 u_up; uniform vec3 u_cam;
out vec2 v_q; out vec4 v_c;
void main() {
    float u = float(gl_VertexID & 1) * 2.0 - 1.0, v = float(gl_VertexID >> 1) * 2.0 - 1.0;
    float dist = length(u_cam - a_p.xyz);
    vec3 c = a_p.xyz + normalize(u_cam - a_p.xyz + vec3(1e-4)) * min(a_p.w * 0.5, dist * 0.5);
    vec3 p = c + (u_right * u + u_up * v) * a_p.w;
    v_q = vec2(u, v);
    v_c = vec4(a_c.rgb, a_c.a * smoothstep(0.1, a_p.w * 1.5 + 0.2, dist));
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

_GLOW_FS = """#version 330 core
in vec2 v_q; in vec4 v_c; out vec4 o;
void main() {
    float r2 = dot(v_q, v_q);
    float k = exp(-r2 * 4.5) - exp(-4.5);
    if (k <= 0.0) discard;
    o = vec4(v_c.rgb * v_c.a * k, 1.0);
}
"""

_FULL_VS = """#version 330 core
out vec2 v_ndc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
    v_ndc = p; gl_Position = vec4(p, 1.0, 1.0);
}
"""

_SKY_FS = """#version 330 core
in vec2 v_ndc; out vec4 o;
uniform vec3 u_r; uniform vec3 u_u; uniform vec3 u_f; uniform vec3 u_zen; uniform vec3 u_hor; uniform vec3 u_sun;
uniform vec3 u_sun_col; uniform vec3 u_fog; uniform float u_kind; uniform float u_cloud; uniform float u_time;
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
float ridge(float az, float seed, float scale) {
    float x = az * scale;
    return vnoise(vec2(x, seed)) * 0.6 + vnoise(vec2(x * 2.7, seed + 9.0)) * 0.3 + vnoise(vec2(x * 7.0, seed)) * 0.1;
}
void main() {
    vec3 d = normalize(u_f + u_r * v_ndc.x - u_u * v_ndc.y);
    float el = d.y, az = atan(d.z, d.x);
    vec3 c = mix(u_hor, u_zen, pow(clamp(el, 0.0, 1.0), 0.5));
    float sd = max(dot(d, u_sun), 0.0);
    c += u_sun_col * (pow(sd, 350.0) * 2.0 + pow(sd, 24.0) * 0.22 + pow(sd, 4.0) * 0.12);
    if (el > 0.0 && u_cloud > 0.0) {
        vec2 cp = d.xz / (el + 0.12) * 0.9 + vec2(u_time * 0.012, u_time * 0.004);
        float n = vnoise(cp) * 0.55 + vnoise(cp * 2.3 + 5.0) * 0.3 + vnoise(cp * 5.1 + 9.0) * 0.15;
        float k = smoothstep(1.0 - u_cloud * 0.9, 1.15 - u_cloud * 0.9, n + 0.25) * smoothstep(0.0, 0.18, el);
        vec3 cc = mix(vec3(1.0), u_sun_col, 0.35) - vec3(0.14) * smoothstep(0.6, 0.95, n);
        c = mix(c, mix(cc, u_hor, 0.25), k * 0.85);
    }
    if (u_kind > 1.5) {  // the showroom: a soft studio glow
        c = mix(u_hor, u_zen, smoothstep(-0.3, 0.9, el)) + vec3(0.10, 0.08, 0.16) * pow(max(1.0 - abs(el), 0.0), 6.0);
    } else if (u_kind > 0.5) {  // far mesas
        float h1 = 0.03 + 0.12 * ridge(az, 3.0, 2.2), h2 = 0.015 + 0.07 * ridge(az, 11.0, 5.0);
        h1 = floor(h1 * 60.0) / 60.0;
        if (el < h1) c = mix(vec3(0.62, 0.34, 0.30), u_fog, 0.5);
        if (el < h2) c = mix(vec3(0.44, 0.22, 0.22), u_fog, 0.3);
    } else {  // far hills and a line of sea
        float h1 = 0.02 + 0.06 * ridge(az, 5.0, 3.0);
        if (el < h1) c = mix(vec3(0.42, 0.62, 0.56), u_fog, 0.55);
        if (el < 0.012) c = mix(vec3(0.30, 0.62, 0.78), u_fog, 0.3);
    }
    if (el < 0.0) c = u_fog;
    c = mix(c, u_fog, exp(-abs(el) * 26.0) * 0.5);
    o = vec4(c, 1.0);
}
"""

_OVER_FS = """#version 330 core
in vec2 v_ndc; out vec4 o;
uniform float u_amount; uniform vec4 u_tint;
void main() {
    float k = dot(v_ndc, v_ndc) * 0.5;
    float a = clamp(k * k * u_amount, 0.0, 1.0);
    float t = clamp(pow(k, 1.5) * u_tint.a, 0.0, 1.0);
    o = vec4(mix(vec3(0.0), u_tint.rgb, t / max(a + t, 1e-4)), clamp(a + t, 0.0, 0.92));
}
"""


# --- geometry --------------------------------------------------------------------------------------------------------
def _quads(p: np.ndarray, n: tuple, col: np.ndarray, mat: np.ndarray, ao: np.ndarray, top: np.ndarray,
           bot: np.ndarray) -> np.ndarray:
    """(k, 4, 3) corner positions and per-quad data -> (k * 6, VERT) triangle vertices."""
    k = len(p)
    out = np.zeros((k, 6, VERT), F32)
    for j, i in enumerate((0, 1, 2, 0, 2, 3)):
        out[:, j, 0:3] = p[:, i]
        out[:, j, 10] = ao[:, i]
    out[:, :, 3:6] = n
    out[:, :, 6:9] = (col / 255.0)[:, None, :]
    out[:, :, 9] = mat[:, None]
    out[:, :, 11] = top[:, None]
    out[:, :, 12] = bot[:, None]
    return out.reshape(-1, VERT)


def build_mesh(md: MapData) -> np.ndarray:
    """The map's floors, walls and slabs as triangles."""
    H, W = md.h, md.w
    fh = md.fh.astype(np.float64)
    pad = np.full((H + 2, W + 2), 30.0)
    pad[1:-1, 1:-1] = fh
    ys, xs = np.mgrid[0:H, 0:W]
    xs, ys = xs.reshape(-1).astype(np.float64), ys.reshape(-1).astype(np.float64)
    h = fh.reshape(-1)
    parts = []
    # floors, darker in corners under higher neighbours
    ao = np.empty((H, W, 4))
    for ci, (cx, cy) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        cnt = np.zeros((H, W))
        for dx, dy in ((cx - 1, cy - 1), (cx, cy - 1), (cx - 1, cy), (cx, cy)):
            if (dx, dy) == (0, 0):
                continue
            cnt += pad[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] > fh + 0.3
        ao[:, :, ci] = 1.0 - 0.17 * cnt
    p = np.stack([np.stack([xs + cx, h, ys + cy], -1) for cx, cy in ((0, 0), (1, 0), (1, 1), (0, 1))], 1)
    parts.append(_quads(p, (0, 1, 0), md.top_col.reshape(-1, 3), md.top_mat.reshape(-1), ao.reshape(-1, 4), h, h))
    # walls down to each lower neighbour
    one = np.ones((len(h), 4))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = pad[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx].reshape(-1)
        sel = nb < h - 1e-3
        if not sel.any():
            continue
        x, y, lo, hi = xs[sel], ys[sel], nb[sel], h[sel]
        if dx:
            X = x + (1 if dx > 0 else 0)
            c = [(X, lo, y), (X, lo, y + 1), (X, hi, y + 1), (X, hi, y)]
        else:
            Z = y + (1 if dy > 0 else 0)
            c = [(x, lo, Z), (x + 1, lo, Z), (x + 1, hi, Z), (x, hi, Z)]
        p = np.stack([np.stack(v, -1) for v in c], 1)
        parts.append(_quads(p, (dx, 0, dy), md.side_col.reshape(-1, 3)[sel], md.side_mat.reshape(-1)[sel],
                            one[:len(x)], hi, lo))
    # slabs
    sy, sx = np.nonzero(md.sb < NO_SLAB / 2)
    if len(sx):
        x, y = sx.astype(np.float64), sy.astype(np.float64)
        b, t = md.sb[sy, sx].astype(np.float64), md.st[sy, sx].astype(np.float64)
        col, mat = md.slab_col[sy, sx], md.slab_mat[sy, sx]
        one = np.ones((len(x), 4))
        p = np.stack([np.stack([x + cx, t, y + cy], -1) for cx, cy in ((0, 0), (1, 0), (1, 1), (0, 1))], 1)
        parts.append(_quads(p, (0, 1, 0), col, mat, one, t, t))
        p = np.stack([np.stack([x + cx, b, y + cy], -1) for cx, cy in ((0, 0), (1, 0), (1, 1), (0, 1))], 1)
        parts.append(_quads(p, (0, -1, 0), col, mat, one * 0.75, b, b))
        has = np.zeros((H + 2, W + 2), bool)
        has[1:-1, 1:-1] = md.sb < NO_SLAB / 2
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sel = ~has[1 + sy + dy, 1 + sx + dx] & (pad[1 + sy + dy, 1 + sx + dx] < t)
            if not sel.any():
                continue
            xx, yy, lo, hi = x[sel], y[sel], b[sel], t[sel]
            if dx:
                X = xx + (1 if dx > 0 else 0)
                c = [(X, lo, yy), (X, lo, yy + 1), (X, hi, yy + 1), (X, hi, yy)]
            else:
                Z = yy + (1 if dy > 0 else 0)
                c = [(xx, lo, Z), (xx + 1, lo, Z), (xx + 1, hi, Z), (xx, hi, Z)]
            p = np.stack([np.stack(v, -1) for v in c], 1)
            parts.append(_quads(p, (dx, 0, dy), col[sel], np.full(len(xx), 11.0), one[:len(xx)], hi + 9.0, lo - 9.0))
    return np.concatenate(parts).astype(F32)


def _unit_mesh() -> np.ndarray:
    """A cube of 6 x 7 x 7 squares in parameter space: the middle band of each side is its flat face, the outer
    bands bend round the edges."""
    g = np.array([-1.0, -5 / 6, -2 / 3, -0.5, 0.5, 2 / 3, 5 / 6, 1.0])
    tris = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            a, b = [i for i in range(3) if i != axis]
            for i in range(7):
                for j in range(7):
                    q = []
                    for u, v in ((g[i], g[j]), (g[i + 1], g[j]), (g[i + 1], g[j + 1]), (g[i], g[j + 1])):
                        pt = [0.0, 0.0, 0.0]
                        pt[axis], pt[a], pt[b] = sign, u, v
                        q.append(pt)
                    tris += [q[0], q[1], q[2], q[0], q[2], q[3]]
    return np.array(tris, F32)


class _Buf:
    __slots__ = ("count", "vao", "vbo")

    def __init__(self) -> None:
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.count = 0


def _ortho(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = 2 / (hi[0] - lo[0]), 2 / (hi[1] - lo[1]), -2 / (hi[2] - lo[2])
    m[0, 3] = -(hi[0] + lo[0]) / (hi[0] - lo[0])
    m[1, 3] = -(hi[1] + lo[1]) / (hi[1] - lo[1])
    m[2, 3] = -(hi[2] + lo[2]) / (hi[2] - lo[2])
    return m


_EMPTY = np.zeros((0, ROW), F32)


class Renderer:
    """Draws a map and the rows it is handed from a camera. Everything OpenGL is made the first time it is needed."""

    def __init__(self) -> None:
        self.ctx: QOpenGLContext | None = None
        self.surface: QOffscreenSurface | None = None
        self.f: QOpenGLExtraFunctions | None = None
        self.fbo = self.resolve = None
        self.sh_s = self.sh_d = None
        self.samples = 4
        self.shadow_quality = 2  # 0 off, 1 soft, 2 sharp
        self.outlines = True
        self.md: MapData | None = None
        self._map_dirty = False
        self._shadow_dirty = True
        self.fov = math.radians(90)
        self.view_w, self.view_h = 1280, 780
        self.cam = np.array([0.0, 1.6, 0.0])
        self.R = np.array([0.0, 0.0, 1.0])
        self.U = np.array([0.0, 1.0, 0.0])
        self.F = np.array([1.0, 0.0, 0.0])
        self.gpu = ""
        self.ms = 0.0
        self.lvp_s = np.eye(4)
        self._light_view = np.eye(4)
        self._light_z = (0.0, 1.0)

    # --- set-up -------------------------------------------------------------------------------------------------
    def _init(self) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            raise RuntimeError("Vanguard needs OpenGL 3.3 (no graphics context could be made)")
        surface = QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()
        if not ctx.makeCurrent(surface):
            raise RuntimeError("Vanguard could not use its graphics context")
        self.ctx, self.surface = ctx, surface
        f = self.f = QOpenGLExtraFunctions(ctx)
        f.initializeOpenGLFunctions()
        self.gpu = str(f.glGetString(0x1F01) or "")
        self.p_world = self._program(_WORLD_VS, _WORLD_FS)
        self.p_part = self._program(_PART_VS_HEAD, _PART_FS)
        self.p_line = self._program(_PART_VS_HEAD, _OUTLINE_FS)
        self.p_alpha = self._program(_PART_VS_HEAD, _ALPHA_FS)
        self.p_dpart = self._program(_PART_VS_HEAD, _DEPTH_FS)
        self.p_dworld = self._program(_DEPTH_WORLD_VS, _DEPTH_FS)
        self.p_glow = self._program(_GLOW_VS, _GLOW_FS)
        self.p_sky = self._program(_FULL_VS, _SKY_FS)
        self.p_over = self._program(_FULL_VS, _OVER_FS)
        self.loc = {id(p): {} for p in (self.p_world, self.p_part, self.p_line, self.p_alpha, self.p_dpart,
                                        self.p_dworld, self.p_glow, self.p_sky, self.p_over)}
        self.world = _Buf()
        self.world.vao.create()
        self.world.vao.bind()
        self.world.vbo.create()
        self.world.vbo.bind()
        for loc, off, size in ((0, 0, 3), (1, 12, 3), (2, 24, 3), (3, 36, 4)):
            f.glEnableVertexAttribArray(loc)
            self.p_world.setAttributeBuffer(loc, GL_FLOAT, off, size, VERT * 4)
        self.world.vao.release()
        self.world.vbo.release()
        self.unit = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.unit.create()
        self.unit.bind()
        raw = _unit_mesh()
        self.unit_count = len(raw)
        data = raw.tobytes()
        self.unit.allocate(data, len(data))
        self.unit.release()
        self.props, self.dyn, self.alpha, self.add, self.vm, self.cast = (self._inst_buffer() for _ in range(6))
        self.empty = QOpenGLVertexArrayObject()
        self.empty.create()
        self.glows = _Buf()
        self.glows.vao.create()
        self.glows.vao.bind()
        self.glows.vbo.create()
        self.glows.vbo.bind()
        for loc in (0, 1):
            f.glEnableVertexAttribArray(loc)
            self.p_glow.setAttributeBuffer(loc, GL_FLOAT, loc * 16, 4, 32)
            f.glVertexAttribDivisor(loc, 1)
        self.glows.vao.release()
        self.glows.vbo.release()

    def _inst_buffer(self) -> _Buf:
        f = self.f
        b = _Buf()
        b.vao.create()
        b.vao.bind()
        self.unit.bind()
        f.glEnableVertexAttribArray(0)
        self.p_part.setAttributeBuffer(0, GL_FLOAT, 0, 3, 12)
        b.vbo.create()
        b.vbo.bind()
        for i in range(6):
            f.glEnableVertexAttribArray(1 + i)
            self.p_part.setAttributeBuffer(1 + i, GL_FLOAT, i * 16, 4, ROW * 4)
            f.glVertexAttribDivisor(1 + i, 1)
        b.vao.release()
        b.vbo.release()
        return b

    def _program(self, vs: str, fs: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs) and ok
        if not (ok and prog.link()):
            raise RuntimeError(f"Vanguard shader did not compile: {prog.log()}")
        return prog

    def _u(self, prog: QOpenGLShaderProgram, name: str) -> int:
        cache = self.loc[id(prog)]
        loc = cache.get(name)
        if loc is None:
            loc = cache[name] = prog.uniformLocation(name)
        return loc

    def _mat(self, prog: QOpenGLShaderProgram, name: str, m: np.ndarray) -> None:
        prog.setUniformValue(self._u(prog, name), QMatrix4x4(*[float(v) for v in m.reshape(-1)]))

    def _current(self) -> bool:
        if self.ctx is None:
            self._init()
            return True
        return self.ctx.makeCurrent(self.surface)

    def set_map(self, md: MapData) -> None:
        """Show this map from now on (its mesh, scenery and shadows are made on the next frame or ``warm_up``)."""
        if md is not self.md:
            self.md = md
            self._map_dirty = True
            self._shadow_dirty = True

    def warm_up(self) -> None:
        if not self._current():
            return
        self._upload_map()
        self._shadow_static()
        self.ctx.doneCurrent()

    def close(self) -> None:
        """Give everything back to the graphics card."""
        if self.ctx is None:
            return
        try:
            if self.ctx.makeCurrent(self.surface):
                for b in (self.world, self.props, self.dyn, self.alpha, self.add, self.vm, self.cast, self.glows):
                    b.vao.destroy()
                    b.vbo.destroy()
                self.unit.destroy()
                self.empty.destroy()
                self.fbo = self.resolve = self.sh_s = self.sh_d = None
                self.ctx.doneCurrent()
        except Exception:  # noqa: BLE001 - closing must never fail
            log.debug("Vanguard renderer close", exc_info=True)
        self.ctx = None
        self.surface = None

    def _fill(self, b: _Buf, data: np.ndarray) -> None:
        raw = np.ascontiguousarray(data, F32).tobytes()
        b.vbo.bind()
        b.vbo.allocate(raw, len(raw))
        b.vbo.release()
        b.count = len(data)

    def _upload_map(self) -> None:
        if not self._map_dirty or self.md is None:
            return
        self._fill(self.world, build_mesh(self.md))
        props = np.array(self.md.props, F32).reshape(-1, ROW) if self.md.props else _EMPTY
        self._fill(self.props, props)
        self._map_dirty = False
        self._shadow_dirty = True

    def _size(self, w: int, h: int) -> None:
        if self.fbo is not None and self.fbo.width() == w and self.fbo.height() == h:
            return
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        fmt.setInternalTextureFormat(GL_RGB8)
        fmt.setSamples(self.samples)
        self.fbo = QOpenGLFramebufferObject(w, h, fmt)
        fmt2 = QOpenGLFramebufferObjectFormat()
        fmt2.setInternalTextureFormat(GL_RGB8)
        self.resolve = QOpenGLFramebufferObject(w, h, fmt2)

    # --- shadows --------------------------------------------------------------------------------------------------
    def _shadow_fbo(self, size: int) -> QOpenGLFramebufferObject | None:
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
        fmt.setInternalTextureFormat(GL_R32F)
        fbo = QOpenGLFramebufferObject(size, size, fmt)
        if not fbo.isValid():
            return None
        f = self.f
        f.glBindTexture(GL_TEXTURE_2D, fbo.texture())
        for name, val in ((GL_TEXTURE_MIN_FILTER, GL_NEAREST), (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                          (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
            f.glTexParameteri(GL_TEXTURE_2D, name, val)
        f.glBindTexture(GL_TEXTURE_2D, 0)
        return fbo

    def _shadow_static(self) -> None:
        """The map's own shadows, drawn once: its depth as the sun sees it."""
        if not self._shadow_dirty or self.md is None:
            return
        self._shadow_dirty = False
        want = (0, 2048, 4096)[self.shadow_quality]
        if want == 0:
            self.sh_s = self.sh_d = None
            return
        if self.sh_s is None or self.sh_s.width() != want:
            self.sh_s = self._shadow_fbo(want)
            self.sh_d = self._shadow_fbo(1024) if self.sh_s is not None else None
            if self.sh_s is None or self.sh_d is None:
                self.sh_s = self.sh_d = None
                self.shadow_quality = 0
                return
        md = self.md
        sun = np.array(md.sky["sun"], np.float64)
        sun /= np.linalg.norm(sun)
        fwd = -sun
        right = np.cross(fwd, (0.0, 1.0, 0.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = right, up, -fwd
        top = float(min(md.fh.max(), 16.0)) + 12.0
        corners = np.array([(x, y, z, 1.0) for x in (0, md.w) for y in (-1.0, top) for z in (0, md.h)])
        vc = corners @ view.T
        lo, hi = vc.min(0)[:3], vc.max(0)[:3]
        self._light_view = view
        self._light_z = (-hi[2], -lo[2])
        self.lvp_s = _ortho(np.array([lo[0], lo[1], -hi[2]]), np.array([hi[0], hi[1], -lo[2]])) @ view
        f = self.f
        self.sh_s.bind()
        f.glViewport(0, 0, want, want)
        f.glDisable(GL_BLEND)
        f.glDisable(GL_CULL_FACE)
        f.glDisable(GL_STENCIL_TEST)
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LESS)
        f.glDepthMask(True)
        f.glClearColor(1.0, 1.0, 1.0, 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        prog = self.p_dworld
        prog.bind()
        self._mat(prog, "u_vp", self.lvp_s)
        self.world.vao.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, self.world.count)
        self.world.vao.release()
        self._draw_parts(self.p_dpart, self.props, self.lvp_s)
        self.sh_s.release()

    def _shadow_dynamic(self, rows: np.ndarray) -> np.ndarray:
        """The moving things' shadows around the camera; returns the matrix that finds a point in that picture."""
        view = self._light_view
        c = view[:3, :3] @ self.cam
        half = 30.0
        zn, zf = self._light_z
        lvp = _ortho(np.array([c[0] - half, c[1] - half, zn]), np.array([c[0] + half, c[1] + half, zf])) @ view
        f = self.f
        self.sh_d.bind()
        f.glViewport(0, 0, 1024, 1024)
        f.glDisable(GL_BLEND)
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LESS)
        f.glDepthMask(True)
        f.glClearColor(1.0, 1.0, 1.0, 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if len(rows):
            self._fill(self.cast, rows)
            self._draw_parts(self.p_dpart, self.cast, lvp)
        self.sh_d.release()
        return lvp

    def _draw_parts(self, prog: QOpenGLShaderProgram, b: _Buf, vp: np.ndarray, grow: float = 0.0,
                    only: float = 0.0) -> None:
        if not b.count:
            return
        f = self.f
        prog.bind()
        self._mat(prog, "u_vp", vp)
        f.glUniform1f(self._u(prog, "u_grow"), grow)
        f.glUniform1f(self._u(prog, "u_only"), only)
        f.glUniform3f(self._u(prog, "u_eye"), *[float(v) for v in self.cam])
        b.vao.bind()
        f.glDrawArraysInstanced(GL_TRIANGLES, 0, self.unit_count, b.count)
        b.vao.release()

    # --- the camera ---------------------------------------------------------------------------------------------
    def set_camera(self, x: float, y: float, z: float, yaw: float, pitch: float) -> None:
        """The eye at map (x, y), height z, facing ``yaw``, looking ``pitch`` up."""
        ca, sa, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        self.cam = np.array([x, z, y])
        self.F = np.array([cp * ca, sp, cp * sa])
        self.R = np.array([-sa, 0.0, ca])
        self.U = np.cross(self.R, self.F)

    def _tans(self, fov: float) -> tuple[float, float]:
        tx = math.tan(fov / 2)
        return tx, tx * self.view_h / max(self.view_w, 1)

    def _vp(self, fov: float, near: float, far: float, at_eye: bool = False) -> np.ndarray:
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = self.R, self.U, -self.F
        view[:3, 3] = -(view[:3, :3] @ self.cam)
        tx, ty = self._tans(fov)
        proj = np.zeros((4, 4))
        proj[0, 0] = 1.0 / tx
        proj[1, 1] = -1.0 / ty  # upside down (see the module docstring)
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2 * far * near / (near - far)
        proj[3, 2] = -1.0
        return proj @ view

    @property
    def proj(self) -> float:
        """Pixels (of the widget) per unit of the view plane at distance 1."""
        return (self.view_w / 2) / math.tan(self.fov / 2)

    def project(self, x: float, y: float, z: float) -> tuple[float, float, float] | None:
        """Widget (sx, sy) and depth of the map point (x, y) at height z; None behind the camera."""
        dx, dy, dz = x - self.cam[0], z - self.cam[1], y - self.cam[2]
        F, R, U = self.F, self.R, self.U
        zc = dx * F[0] + dy * F[1] + dz * F[2]
        if zc < 0.05:
            return None
        k = self.proj / zc
        return (self.view_w / 2 + (dx * R[0] + dy * R[1] + dz * R[2]) * k,
                self.view_h / 2 - (dx * U[0] + dy * U[1] + dz * U[2]) * k, zc)

    # --- the frame ----------------------------------------------------------------------------------------------
    def _common(self, prog: QOpenGLShaderProgram, vp: np.ndarray, lights: list, lvp_d: np.ndarray) -> None:
        f, sky = self.f, self.md.sky
        prog.bind()
        self._mat(prog, "u_vp", vp)
        f.glUniform3f(self._u(prog, "u_cam"), *[float(v) for v in self.cam])
        sun = np.array(sky["sun"], np.float64)
        sun /= np.linalg.norm(sun)
        f.glUniform3f(self._u(prog, "u_sun"), *[float(v) for v in sun])
        f.glUniform3f(self._u(prog, "u_sun_col"), *sky["sun_col"])
        f.glUniform3f(self._u(prog, "u_amb_sky"), *sky["amb_sky"])
        f.glUniform3f(self._u(prog, "u_amb_gnd"), *sky["amb_gnd"])
        f.glUniform3f(self._u(prog, "u_fog"), *sky["fog"])
        f.glUniform1f(self._u(prog, "u_fogd"), float(sky["fog_d"]))
        on = self.sh_s is not None and self.sh_d is not None
        f.glUniform1f(self._u(prog, "u_shadows"), 1.0 if on else 0.0)
        if on:
            f.glUniform2f(self._u(prog, "u_sh_size"), float(self.sh_s.width()), 1024.0)
            self._mat(prog, "u_lvp_s", self.lvp_s)
            self._mat(prog, "u_lvp_d", lvp_d)
            prog.setUniformValue1i(self._u(prog, "u_sh_s"), 1)
            prog.setUniformValue1i(self._u(prog, "u_sh_d"), 2)
        f.glUniform1i(self._u(prog, "u_nl"), len(lights))
        for i, (X, Y, Z, col, rad) in enumerate(lights):
            f.glUniform4f(self._u(prog, f"u_lpos[{i}]"), float(X), float(Z), float(Y), float(rad))
            f.glUniform3f(self._u(prog, f"u_lcol[{i}]"), float(col[0]), float(col[1]), float(col[2]))

    def render(self, w: int, h: int, fov: float, t: float = 0.0, dpr: float = 1.0, rows: np.ndarray = _EMPTY,
               alpha: np.ndarray = _EMPTY, add: np.ndarray = _EMPTY, glows: list | None = None,
               lights: list | None = None, vm: np.ndarray = _EMPTY, casters: np.ndarray | None = None,
               hurt: float = 0.0, tint: tuple = (1.0, 0.1, 0.05), vignette: float = 0.55,
               ally: tuple = (0.25, 0.7, 1.0), enemy: tuple = (1.0, 0.25, 0.2), vm_fov: float = 1.15) -> QImage:
        """One frame for a widget of ``w`` x ``h`` (drawn at ``dpr`` times that) from the camera set with
        ``set_camera``. ``rows`` are the solid moving things, ``alpha`` / ``add`` the see-through and glowing ones,
        ``vm`` the player's own weapon; ``glows`` are (x, y, z, radius, r, g, b, strength); ``lights`` are
        (x, y, z, (r, g, b), radius)."""
        t0 = time.perf_counter()
        if not self._current():
            raise RuntimeError("Vanguard lost its graphics context")
        f = self.f
        pw, ph = max(int(w * dpr), 16), max(int(h * dpr), 16)
        self._size(pw, ph)
        self._upload_map()
        self._shadow_static()
        self.fov, self.view_w, self.view_h = fov, w, h
        lights = sorted(lights or [], key=lambda L: (L[0] - self.cam[0]) ** 2 + (L[1] - self.cam[2]) ** 2)[:MAX_LIGHTS]
        lvp_d = self.lvp_s
        shadows = self.sh_s is not None and self.sh_d is not None
        if shadows:
            lvp_d = self._shadow_dynamic(rows if casters is None else casters)
            f.glActiveTexture(GL_TEXTURE0 + 1)
            f.glBindTexture(GL_TEXTURE_2D, self.sh_s.texture())
            f.glActiveTexture(GL_TEXTURE0 + 2)
            f.glBindTexture(GL_TEXTURE_2D, self.sh_d.texture())
            f.glActiveTexture(GL_TEXTURE0)
        vp = self._vp(fov, 0.06, 400.0)
        self.fbo.bind()
        f.glViewport(0, 0, pw, ph)
        f.glDisable(GL_BLEND)
        f.glDisable(GL_CULL_FACE)
        f.glDisable(GL_STENCIL_TEST)
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LESS)
        f.glDepthMask(True)
        f.glStencilMask(0xFF)
        fog = self.md.sky["fog"]
        f.glClearColor(fog[0], fog[1], fog[2], 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        # the map and its scenery
        prog = self.p_world
        self._common(prog, vp, lights, lvp_d)
        f.glUniform1f(self._u(prog, "u_dusk"), 1.0 if self.md.sky["kind"] == 1.0 else 0.0)
        self.world.vao.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, self.world.count)
        self.world.vao.release()
        prog = self.p_part
        self._common(prog, vp, lights, lvp_d)
        self._draw_parts(prog, self.props, vp)
        # the moving things, marking the stencil so the outlines can go round them
        if len(rows):
            self._fill(self.dyn, rows)
            f.glEnable(GL_STENCIL_TEST)
            f.glStencilFunc(GL_ALWAYS, 1, 0xFF)
            f.glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)
            self._draw_parts(prog, self.dyn, vp)
            if self.outlines:
                f.glStencilFunc(GL_NOTEQUAL, 1, 0xFF)
                f.glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
                f.glDepthFunc(GL_LEQUAL)
                line = self.p_line
                line.bind()
                f.glUniform3f(self._u(line, "u_ally"), *ally)
                f.glUniform3f(self._u(line, "u_enemy"), *enemy)
                f.glUniform1f(self._u(line, "u_alpha"), 1.0)
                self._draw_parts(line, self.dyn, vp, grow=0.022, only=1.0)
                # revealed enemies show through everything
                f.glDisable(GL_DEPTH_TEST)
                f.glEnable(GL_BLEND)
                f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                f.glUniform1f(self._u(line, "u_alpha"), 0.5)
                self._draw_parts(line, self.dyn, vp, grow=0.012, only=3.0)
                f.glDisable(GL_BLEND)
                f.glEnable(GL_DEPTH_TEST)
            f.glDisable(GL_STENCIL_TEST)
        else:
            self.dyn.count = 0
        # the sky where nothing was drawn
        f.glDepthFunc(GL_LEQUAL)
        f.glDepthMask(False)
        prog = self.p_sky
        prog.bind()
        tx, ty = self._tans(fov)
        sky = self.md.sky
        sun = np.array(sky["sun"], np.float64)
        sun /= np.linalg.norm(sun)
        f.glUniform3f(self._u(prog, "u_r"), *[float(v) * tx for v in self.R])
        f.glUniform3f(self._u(prog, "u_u"), *[float(v) * ty for v in self.U])
        f.glUniform3f(self._u(prog, "u_f"), *[float(v) for v in self.F])
        f.glUniform3f(self._u(prog, "u_zen"), *sky["zenith"])
        f.glUniform3f(self._u(prog, "u_hor"), *sky["horizon"])
        f.glUniform3f(self._u(prog, "u_sun"), *[float(v) for v in sun])
        f.glUniform3f(self._u(prog, "u_sun_col"), *sky["sun_col"])
        f.glUniform3f(self._u(prog, "u_fog"), *sky["fog"])
        f.glUniform1f(self._u(prog, "u_kind"), float(sky["kind"]))
        f.glUniform1f(self._u(prog, "u_cloud"), float(sky["cloud"]))
        f.glUniform1f(self._u(prog, "u_time"), float(t))
        self.empty.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.empty.release()
        # see-through things, then the ones made of light
        f.glEnable(GL_BLEND)
        prog = self.p_alpha
        prog.bind()
        f.glUniform3f(self._u(prog, "u_cam"), *[float(v) for v in self.cam])
        f.glUniform1f(self._u(prog, "u_time"), float(t))
        if len(alpha):
            self._fill(self.alpha, alpha)
            f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            f.glUniform1f(self._u(prog, "u_add"), 0.0)
            self._draw_parts(prog, self.alpha, vp)
        if len(add):
            self._fill(self.add, add)
            f.glBlendFunc(GL_ONE, GL_ONE)
            f.glUniform1f(self._u(prog, "u_add"), 1.0)
            self._draw_parts(prog, self.add, vp)
        if glows:
            prog = self.p_glow
            prog.bind()
            self._mat(prog, "u_vp", vp)
            f.glUniform3f(self._u(prog, "u_right"), *[float(v) for v in self.R])
            f.glUniform3f(self._u(prog, "u_up"), *[float(v) for v in self.U])
            f.glUniform3f(self._u(prog, "u_cam"), *[float(v) for v in self.cam])
            g = np.array(glows, F32)
            g[:, [1, 2]] = g[:, [2, 1]]
            g[:, 4:7] *= 1.0 / 255.0
            self._fill(self.glows, g)
            f.glBlendFunc(GL_ONE, GL_ONE)
            self.glows.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(g))
            self.glows.vao.release()
        f.glDisable(GL_BLEND)
        f.glDepthMask(True)
        # the player's own weapon, over everything
        if len(vm):
            f.glClear(GL_DEPTH_BUFFER_BIT)
            f.glDepthFunc(GL_LESS)
            self._fill(self.vm, vm)
            prog = self.p_part
            prog.bind()
            self._draw_parts(prog, self.vm, self._vp(vm_fov, 0.02, 20.0))
        # darker corners, red when hurt
        f.glDisable(GL_DEPTH_TEST)
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        prog = self.p_over
        prog.bind()
        f.glUniform1f(self._u(prog, "u_amount"), float(vignette))
        f.glUniform4f(self._u(prog, "u_tint"), float(tint[0]), float(tint[1]), float(tint[2]), float(hurt))
        self.empty.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.empty.release()
        f.glDisable(GL_BLEND)
        self.fbo.release()
        QOpenGLFramebufferObject.blitFramebuffer(self.resolve, self.fbo)
        img = self.resolve.toImage(False)
        self.ctx.doneCurrent()
        if dpr != 1.0:
            img.setDevicePixelRatio(dpr)
        self.ms = (time.perf_counter() - t0) * 1000
        return img

    def vm_point(self, local: np.ndarray, vm_fov: float = 1.15, depth: float = 1.2) -> tuple[float, float, float]:
        """Where a point of the weapon model (eye axes: right, up, forward) is in the world as the main camera
        sees it: the map (x, y, z) at ``depth`` in front of the eye on the same line of sight."""
        k = math.tan(self.fov / 2) / math.tan(vm_fov / 2)
        z = max(float(local[2]), 0.05)
        sx, sy = float(local[0]) / z * k, float(local[1]) / z * k
        p = self.cam + (self.F + self.R * sx + self.U * sy) * depth
        return float(p[0]), float(p[2]), float(p[1])


__all__ = ["Renderer", "build_mesh"]
