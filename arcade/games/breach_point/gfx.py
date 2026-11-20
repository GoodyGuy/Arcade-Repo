"""The 3D view of BREACH POINT, drawn by the graphics card (OpenGL 3.3 through Qt's own classes) at the window's own
resolution into an image the widget paints.

Nothing is textured from a file: the map is one buffer of flat-coloured triangles whose brick courses, boards, tiles,
crate frames and painted stripes are drawn by the fragment shader from where the pixel is in the world, so every line
stays sharp at any size. The sun casts real shadows (a depth picture of the map from the sun, made once per map);
roofed rooms are lit by their lamps. Agents, guns and props are instanced boxes with an outline for enemies; smoke is
instanced spheres; ground effects are instanced discs; tracers and glows are additive. The picture is drawn upside
down on purpose so it can be read back without a flip."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from PySide6.QtGui import QImage, QMatrix4x4, QOffscreenSurface, QOpenGLContext, QOpenGLExtraFunctions, QSurfaceFormat
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat, QOpenGLShader,
                              QOpenGLShaderProgram, QOpenGLVertexArrayObject)

from .config import ROOF
from .maps import M_ROOF, GameMap

log = logging.getLogger(__name__)
F32 = np.float32
GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_FLOAT = 0x0004, 0x0005, 0x1406
GL_DEPTH_TEST, GL_CULL_FACE, GL_BLEND, GL_SAMPLE_ALPHA_TO_COVERAGE = 0x0B71, 0x0B44, 0x0BE2, 0x809E
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_LEQUAL, GL_LESS, GL_GREATER = 0x4000, 0x0100, 0x0203, 0x0201, 0x0204
GL_ZERO, GL_ONE, GL_DST_COLOR, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = 0, 1, 0x0306, 0x0302, 0x0303
GL_RGB8, GL_R32F, GL_RGBA32F, GL_FRONT, GL_BACK = 0x8051, 0x822E, 0x8814, 0x0404, 0x0405
GL_TEXTURE_2D, GL_TEXTURE0 = 0x0DE1, 0x84C0
SHADOW_SIZE = 2048
MAX_LAMPS = 12
VERT = 12  # position 3, normal 3, uv 2, material / occlusion / indoors / base height 4

_LIGHT = """
uniform vec3 u_cam; uniform vec3 u_sun; uniform vec3 u_sun_col; uniform vec3 u_amb_sky; uniform vec3 u_amb_gnd;
uniform vec3 u_amb_in; uniform vec3 u_fog; uniform float u_fogd;
uniform sampler2D u_shadow; uniform mat4 u_light; uniform float u_shadow_on;
uniform vec4 u_lpos[12]; uniform vec3 u_lcol[12]; uniform int u_nl;
float shadow(vec3 p, vec3 n) {
    if (u_shadow_on < 0.5) return 1.0;
    vec4 q = u_light * vec4(p + n * 0.07, 1.0);
    vec3 s = q.xyz * 0.5 + 0.5;
    if (s.x < 0.0 || s.x > 1.0 || s.y < 0.0 || s.y > 1.0) return 1.0;
    float lit = 0.0;
    float tx = 1.25 / 2048.0;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
        lit += (s.z - 0.0016 > texture(u_shadow, s.xy + vec2(i, j) * tx).r) ? 0.0 : 1.0;
    return lit / 9.0;
}
vec3 lighting(vec3 p, vec3 n, float ao, float indoor) {
    float sh = shadow(p, n);
    float nl = max(dot(n, -u_sun), 0.0);
    vec3 amb = mix(mix(u_amb_gnd, u_amb_sky, n.y * 0.5 + 0.5), u_amb_in, indoor);
    vec3 c = amb * ao + u_sun_col * nl * sh;
    c += u_sun_col * 0.1 * (1.0 - indoor) * max(-n.y * 0.5 + 0.5, 0.0) * ao;  // light off the ground
    for (int i = 0; i < 12; i++) {
        if (i >= u_nl) break;
        vec3 d = u_lpos[i].xyz - p; float dist = length(d);
        float a = clamp(1.0 - dist / u_lpos[i].w, 0.0, 1.0);
        c += u_lcol[i] * a * a * (max(dot(n, d / max(dist, 1e-4)), 0.0) * 0.75 + 0.25);
    }
    return c;
}
vec3 knee(vec3 c) { vec3 e = max(c - 0.8, 0.0); return min(c, 0.8) + e / (1.0 + e * 3.5); }
vec3 fogged(vec3 rgb, vec3 p) {
    float f = 1.0 - exp(-pow(length(p - u_cam) / u_fogd, 1.3));
    return mix(rgb, u_fog, min(f, 0.9));
}
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
"""

_WORLD_VS = """#version 330 core
layout(location = 0) in vec3 a_pos; layout(location = 1) in vec3 a_nrm; layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_info;
uniform mat4 u_vp;
out vec3 v_pos; out vec3 v_nrm; out vec2 v_uv; flat out float v_mat; out float v_ao; flat out float v_in;
flat out float v_base;
void main() {
    v_pos = a_pos; v_nrm = a_nrm; v_uv = a_uv; v_mat = a_info.x; v_ao = a_info.y; v_in = a_info.z; v_base = a_info.w;
    gl_Position = u_vp * vec4(a_pos, 1.0);
}
"""

_WORLD_FS = "#version 330 core\n" + _LIGHT + """
in vec3 v_pos; in vec3 v_nrm; in vec2 v_uv; flat in float v_mat; in float v_ao; flat in float v_in; flat in float v_base;
uniform vec3 u_pal[20]; uniform vec3 u_accent; uniform vec4 u_site[2];
out vec4 o;
float seam(float c, float period, float w) {
    float d = abs(fract(c / period - 0.5) - 0.5) * period;
    float aa = fwidth(c) * 1.2;
    return (1.0 - smoothstep(w, w + aa, d)) * clamp(1.0 - aa / (period * 0.3), 0.0, 1.0);
}
float band(float c, float lo, float hi) {
    float aa = fwidth(c);
    return smoothstep(lo - aa, lo + aa, c) * (1.0 - smoothstep(hi - aa, hi + aa, c));
}
void main() {
    int m = int(v_mat + 0.5);
    vec2 uv = v_uv;
    vec3 base = u_pal[m];
    vec3 c = base;
    vec3 n = normalize(v_nrm);
    float near = clamp(1.0 - fwidth(uv.x) * 9.0, 0.0, 1.0);
    if (m == 0) {
        c *= 1.0 - 0.15 * max(seam(uv.x, 2.0, 0.018), seam(uv.y, 2.0, 0.018));
        c *= 0.95 + 0.1 * hash(floor(uv / 2.0));
    } else if (m == 1) {
        vec2 t = floor(uv);
        c *= mod(t.x + t.y, 2.0) < 0.5 ? 1.0 : 0.9;
        c *= 1.0 - 0.14 * max(seam(uv.x, 1.0, 0.012), seam(uv.y, 1.0, 0.012));
    } else if (m == 2) {
        float pl = floor(uv.x / 0.25);
        c *= 0.88 + 0.22 * hash(vec2(pl, 3.0));
        c *= 1.0 - 0.22 * seam(uv.x, 0.25, 0.008);
        c *= 1.0 - 0.16 * seam(uv.y + hash(vec2(pl, 7.0)) * 2.0, 2.0, 0.008);
    } else if (m == 3) {
        c *= 1.0 - 0.2 * max(seam(uv.x, 1.0, 0.014), seam(uv.y, 1.0, 0.014));
        vec2 f = abs(fract(uv) - 0.5);
        c *= 1.0 + 0.3 * (1.0 - smoothstep(0.02, 0.04, length(f - vec2(0.4)))) * near;
        c *= 0.96 + 0.07 * hash(floor(uv));
    } else if (m == 4) {
        c *= 0.92 + 0.16 * hash(floor(uv * 8.0)) * near + 0.0;
        c *= 0.95 + 0.1 * hash(floor(uv / 3.0));
    } else if (m == 5) {
        c *= 1.0 - 0.13 * max(seam(uv.x, 1.5, 0.018), seam(uv.y, 1.5, 0.018));
        c *= 0.96 + 0.07 * hash(floor(uv / 1.5));
    } else if (m == 8) {
        float row = floor(uv.y / 0.25);
        float off = mod(row, 2.0) * 0.25;
        c *= 0.86 + 0.26 * hash(vec2(floor((uv.x + off) / 0.5), row));
        float mort = max(seam(uv.y, 0.25, 0.011), seam(uv.x + off, 0.5, 0.011));
        c = mix(c, base * 1.2 + 0.1, mort * 0.6);
    } else if (m == 9) {
        float dado = 1.0 - smoothstep(1.0, 1.0 + fwidth(uv.y) * 1.5, uv.y);
        c = mix(c, mix(c, u_accent, 0.42) * 0.82, dado * step(n.y, 0.5));
        c = mix(c, u_accent * 0.9, band(uv.y, 1.0, 1.07) * step(n.y, 0.5));
        c *= 0.97 + 0.06 * hash(floor(uv * 0.5));
        c *= 1.0 - 0.08 * seam(uv.x, 3.0, 0.012);
    } else if (m == 10) {
        float b = floor(uv.x / 0.33);
        c *= 0.88 + 0.22 * hash(vec2(b, 1.0));
        c *= 1.0 - 0.32 * seam(uv.x, 0.33, 0.012);
        c *= 1.0 - 0.25 * (band(uv.y, 0.42, 0.52) + band(uv.y, 1.9, 2.0)) * step(n.y, 0.5);
    } else if (m == 11) {
        vec2 f = fract(uv);
        vec2 e = min(f, 1.0 - f);
        float aa = fwidth(uv.x) * 1.5;
        float fr = 1.0 - smoothstep(0.075, 0.075 + aa, min(e.x, e.y));
        float brace = (1.0 - smoothstep(0.055, 0.055 + aa, abs(f.x - f.y))) * (1.0 - fr);
        c *= 1.0 - 0.13 * seam(uv.x, 0.2, 0.008) * (1.0 - fr);
        c = mix(c, c * 0.7, fr);
        c = mix(c, c * 0.8, brace);
        c *= 0.95 + 0.1 * hash(floor(uv));
    } else if (m == 12) {
        c *= 0.9 + 0.12 * sin(uv.x * 25.13) * near;
        c *= 1.0 - 0.3 * (1.0 - band(uv.y, 0.12, 100.0)) * step(n.y, 0.5);
        c = mix(c, u_accent, band(uv.y, 1.2, 1.32) * 0.85 * step(n.y, 0.5) * step(1.5, v_base + 99.0));
    } else if (m == 13) {
        float row = floor(uv.y / 0.5);
        float off = mod(row, 2.0) * 0.5;
        c *= 0.9 + 0.16 * hash(vec2(floor(uv.x + off), row));
        c *= 1.0 - 0.24 * max(seam(uv.y, 0.5, 0.014), seam(uv.x + off, 1.0, 0.014));
    } else if (m == 14) {
        c *= 0.9 + 0.16 * hash(vec2(floor(uv.y / 0.2), 5.0));
        c *= 1.0 - 0.24 * seam(uv.y, 0.2, 0.008);
    } else if (m == 15) {
        c *= 1.0 - 0.34 * seam(uv.x, 2.0, 0.09);
        c *= 1.0 - 0.12 * seam(uv.y, 0.5, 0.01);
    } else if (m == 16) {
        if (n.y < 0.5) {
            float top = smoothstep(0.8, 0.8 + fwidth(uv.y), uv.y);
            float st = step(0.5, fract((uv.x + uv.y) * 2.0));
            c = mix(c, mix(u_accent, vec3(0.12), st), top);
        }
    }
    float ao = v_ao;
    if (n.y < 0.5) {
        ao *= mix(0.66, 1.0, smoothstep(0.0, 0.9, v_pos.y - v_base));
        c *= mix(0.86, 1.0, smoothstep(0.0, 0.16, v_pos.y - v_base));
    } else if (n.y > 0.5) {
        for (int i = 0; i < 2; i++) {  // the painted edge of a site
            vec4 r = u_site[i];
            vec2 q = v_pos.xz;
            if (q.x > r.x - 0.2 && q.x < r.z + 0.2 && q.y > r.y - 0.2 && q.y < r.w + 0.2) {
                float e = min(min(abs(q.x - r.x), abs(q.x - r.z)), min(abs(q.y - r.y), abs(q.y - r.w)));
                float ln = 1.0 - smoothstep(0.07, 0.07 + fwidth(q.x) * 1.5, e);
                float dash = step(0.35, fract((q.x + q.y) * 0.5));
                c = mix(c, u_accent, ln * dash * 0.85 * step(v_pos.y, 1.6));
            }
        }
    }
    vec3 lit = c * lighting(v_pos, n, ao, v_in);
    o = vec4(fogged(knee(lit), v_pos), 1.0);
}
"""

_DEPTH_VS = """#version 330 core
layout(location = 0) in vec3 a_pos;
uniform mat4 u_vp;
void main() { gl_Position = u_vp * vec4(a_pos, 1.0); }
"""
_DEPTH_FS = """#version 330 core
out vec4 o;
void main() { o = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0); }
"""

_BOX_VS = """#version 330 core
layout(location = 0) in vec3 a_p; layout(location = 1) in vec3 a_n;
layout(location = 2) in vec3 i_c; layout(location = 3) in vec3 i_x; layout(location = 4) in vec3 i_y;
layout(location = 5) in vec3 i_z; layout(location = 6) in vec4 i_col;
uniform mat4 u_vp; uniform float u_grow;
out vec3 v_pos; out vec3 v_nrm; out vec4 v_col;
void main() {
    vec3 ax = i_x, ay = i_y, az = i_z;
    if (u_grow > 0.0) {
        ax += normalize(ax) * u_grow; ay += normalize(ay) * u_grow; az += normalize(az) * u_grow;
    }
    vec3 p = i_c + ax * a_p.x + ay * a_p.y + az * a_p.z;
    v_pos = p;
    v_nrm = normalize(i_x * a_n.x + i_y * a_n.y + i_z * a_n.z);
    v_col = i_col;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

_BOX_FS = "#version 330 core\n" + _LIGHT + """
in vec3 v_pos; in vec3 v_nrm; in vec4 v_col;
uniform float u_mode; uniform vec4 u_flat; uniform float u_indoor;
out vec4 o;
void main() {
    if (u_mode > 0.5) { o = u_flat; return; }
    vec3 n = normalize(v_nrm);
    if (v_col.a > 0.5) { o = vec4(v_col.rgb * 1.15, 1.0); return; }
    vec3 lit = v_col.rgb * lighting(v_pos, n, 0.92, u_indoor);
    lit += v_col.rgb * 0.06 * pow(1.0 - max(dot(n, normalize(u_cam - v_pos)), 0.0), 2.0);
    o = vec4(fogged(knee(lit), v_pos), 1.0);
}
"""

_DECAL_VS = """#version 330 core
layout(location = 0) in vec4 i_p; layout(location = 1) in vec4 i_col; layout(location = 2) in vec4 i_k;
uniform mat4 u_vp;
out vec2 v_q; out vec4 v_col; flat out vec4 v_k; out vec3 v_pos;
void main() {
    float u = float(gl_VertexID & 1) * 2.0 - 1.0, v = float(gl_VertexID >> 1) * 2.0 - 1.0;
    vec3 p = i_p.xyz + vec3(u, 0.0, v) * i_p.w;
    v_q = vec2(u, v); v_col = i_col; v_k = i_k; v_pos = p;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

_DECAL_FS = """#version 330 core
in vec2 v_q; in vec4 v_col; flat in vec4 v_k; in vec3 v_pos;
uniform float u_time;
out vec4 o;
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    float r = length(v_q);
    if (r > 1.0) discard;
    int kind = int(v_k.x + 0.5);
    float a = 0.0;
    vec3 c = v_col.rgb;
    if (kind == 0) {  // a soft shadow under a walker
        a = (1.0 - smoothstep(0.2, 1.0, r)) * v_col.a;
    } else if (kind == 1) {  // a ring going outwards
        float w = v_k.y;
        a = (1.0 - smoothstep(0.0, 0.06, abs(r - w))) * v_col.a * (1.0 - w * 0.6);
        a += (1.0 - smoothstep(0.0, w, r)) * 0.06 * v_col.a;
    } else if (kind == 2) {  // a field on the ground
        float edge = smoothstep(0.9, 0.97, r) * (1.0 - smoothstep(0.985, 1.0, r));
        float flow = vnoise(v_pos.xz * 1.3 + vec2(u_time * 0.5, -u_time * 0.3));
        float rings = 0.5 + 0.5 * sin(r * 16.0 - u_time * 2.5);
        a = (0.18 + 0.2 * flow + 0.1 * rings) * (1.0 - smoothstep(0.92, 1.0, r)) + edge * 0.8;
        a *= v_col.a;
    } else if (kind == 3) {  // fire
        float n1 = vnoise(v_pos.xz * 2.4 + vec2(0.0, u_time * 2.2));
        float n2 = vnoise(v_pos.xz * 5.0 - vec2(u_time * 1.6, 0.0));
        float f = n1 * 0.65 + n2 * 0.35;
        c = mix(vec3(1.0, 0.32, 0.05), vec3(1.0, 0.86, 0.3), smoothstep(0.35, 0.75, f));
        a = smoothstep(0.22, 0.5, f + (1.0 - r) * 0.35) * (1.0 - smoothstep(0.85, 1.0, r)) * v_col.a;
    } else {  // static: flickering cells
        vec2 g = floor(v_pos.xz * 3.0);
        float fl = step(0.55, hash(g + floor(u_time * 12.0)));
        a = (0.12 + 0.35 * fl) * (1.0 - smoothstep(0.9, 1.0, r)) * v_col.a;
        a += smoothstep(0.93, 0.97, r) * (1.0 - smoothstep(0.985, 1.0, r)) * 0.7 * v_col.a;
    }
    o = vec4(c, clamp(a, 0.0, 1.0));
}
"""

_SPHERE_VS = """#version 330 core
layout(location = 0) in vec3 a_p;
layout(location = 1) in vec4 i_p; layout(location = 2) in vec4 i_col;
uniform mat4 u_vp; uniform float u_time;
out vec3 v_pos; out vec3 v_nrm; out vec4 v_col; out vec3 v_loc;
void main() {
    float wob = 1.0 + 0.05 * sin(a_p.x * 5.0 + u_time * 0.9 + i_col.a) * sin(a_p.y * 4.0 - u_time * 0.7)
        + 0.04 * sin(a_p.z * 6.0 + u_time * 1.1);
    if (i_col.a < 0.0) wob = 1.0;
    vec3 p = i_p.xyz + a_p * i_p.w * wob;
    v_pos = p; v_nrm = a_p; v_col = i_col; v_loc = a_p;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

_SPHERE_FS = "#version 330 core\n" + _LIGHT + """
in vec3 v_pos; in vec3 v_nrm; in vec4 v_col; in vec3 v_loc;
uniform float u_time;
out vec4 o;
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec3 n = normalize(v_nrm);
    vec3 v = normalize(u_cam - v_pos);
    float facing = abs(dot(n, v));
    if (v_col.a < 0.0) {  // a shell of light: brightest at its rim
        float rim = pow(1.0 - facing, 2.2);
        float bandy = 0.5 + 0.5 * sin(v_loc.y * 22.0 - u_time * 3.0);
        o = vec4(v_col.rgb, clamp(rim * 0.85 + 0.05 + bandy * 0.05, 0.0, 1.0) * -v_col.a);
        return;
    }
    float puff = vnoise(v_loc.xy * 3.0 + v_loc.z * 2.0 + u_time * 0.12) * 0.6
        + vnoise(v_loc.yz * 6.5 - u_time * 0.1) * 0.4;
    float a = smoothstep(0.02, 0.4, facing + (puff - 0.5) * 0.35);
    if (a < 0.03) discard;
    if (!gl_FrontFacing) n = -n;
    vec3 base = v_col.rgb * (0.78 + 0.34 * puff);
    vec3 lit = base * (lighting(v_pos, n, 1.0, 0.0) * 0.75 + 0.2);
    o = vec4(fogged(knee(lit), v_pos), a);
}
"""

_BEAM_VS = """#version 330 core
layout(location = 0) in vec3 i_a; layout(location = 1) in vec3 i_b; layout(location = 2) in vec4 i_col;
uniform mat4 u_vp; uniform vec3 u_cam;
out vec2 v_q; out vec4 v_col;
void main() {
    float u = float(gl_VertexID & 1), v = float(gl_VertexID >> 1) * 2.0 - 1.0;
    vec3 p = mix(i_a, i_b, u);
    vec3 side = normalize(cross(i_b - i_a, u_cam - p) + vec3(1e-5));
    float dist = length(u_cam - p);
    p += side * v * i_col.a * (0.6 + dist * 0.03);
    v_q = vec2(u, v); v_col = i_col;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""
_BEAM_FS = """#version 330 core
in vec2 v_q; in vec4 v_col; out vec4 o;
void main() {
    float k = (1.0 - v_q.y * v_q.y) * smoothstep(0.0, 0.25, v_q.x);
    o = vec4(v_col.rgb * k, 1.0);
}
"""

_GLOW_VS = """#version 330 core
layout(location = 0) in vec4 a_p; layout(location = 1) in vec4 a_c;
uniform mat4 u_vp; uniform vec3 u_right; uniform vec3 u_up; uniform vec3 u_cam;
out vec2 v_q; out vec4 v_c;
void main() {
    float u = float(gl_VertexID & 1) * 2.0 - 1.0, v = float(gl_VertexID >> 1) * 2.0 - 1.0;
    vec3 p = a_p.xyz + (u_right * u + u_up * v) * a_p.w;
    v_q = vec2(u, v); v_c = a_c;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""
_GLOW_FS = """#version 330 core
in vec2 v_q; in vec4 v_c; out vec4 o;
void main() {
    float k = exp(-dot(v_q, v_q) * 4.5) - exp(-4.5);
    if (k <= 0.0) discard;
    o = vec4(v_c.rgb * v_c.a * k, 1.0);
}
"""

_BARRIER_VS = """#version 330 core
layout(location = 0) in vec3 a_pos; layout(location = 1) in vec2 a_uv;
uniform mat4 u_vp;
out vec2 v_uv;
void main() { v_uv = a_uv; gl_Position = u_vp * vec4(a_pos, 1.0); }
"""
_BARRIER_FS = """#version 330 core
in vec2 v_uv; out vec4 o;
uniform vec3 u_col; uniform float u_time;
void main() {
    float st = step(0.5, fract((v_uv.x + v_uv.y) * 1.2 - u_time * 0.25));
    float top = 1.0 - smoothstep(2.2, 3.0, v_uv.y);
    float foot = 1.0 - smoothstep(0.0, 0.5, v_uv.y);
    o = vec4(u_col, (0.1 + 0.16 * st + 0.3 * foot) * top);
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
uniform vec3 u_r; uniform vec3 u_u; uniform vec3 u_f; uniform float u_time; uniform vec3 u_top; uniform vec3 u_low;
uniform vec3 u_fog; uniform vec3 u_sundir; uniform vec3 u_disc;
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec3 d = normalize(u_f + u_r * v_ndc.x - u_u * v_ndc.y);
    float el = d.y;
    vec3 c = mix(u_low, u_top, pow(clamp(el, 0.0, 1.0), 0.5));
    float sd = max(dot(d, u_sundir), 0.0);
    c += u_disc * (pow(sd, 400.0) * 1.4 + pow(sd, 12.0) * 0.22);
    if (el > 0.01) {
        vec2 cp = d.xz / el * 0.9 + vec2(u_time * 0.012, u_time * 0.004);
        float n = vnoise(cp) * 0.6 + vnoise(cp * 2.3 + 5.0) * 0.3 + vnoise(cp * 5.1 + 9.0) * 0.1;
        float k = smoothstep(0.5, 0.8, n) * smoothstep(0.01, 0.22, el);
        c = mix(c, mix(u_low, vec3(1.0), 0.55) + u_disc * 0.12 * pow(sd, 3.0), k * 0.7);
    }
    float az = atan(d.z, d.x) * 9.5;
    float cell = floor(az);
    float h = 0.012 + hash(vec2(cell, 3.0)) * 0.045 + step(0.82, hash(vec2(cell, 9.0))) * 0.05;
    if (el < h) c = mix(u_fog * 0.62, u_fog, clamp(el / 0.12, 0.0, 1.0) * 0.5 + 0.2);
    if (el < 0.0) c = u_fog * 0.8;
    c = mix(c, u_fog, exp(-abs(el) * 26.0) * 0.5);
    o = vec4(c, 1.0);
}
"""
_VIGNETTE_FS = """#version 330 core
in vec2 v_ndc; out vec4 o;
uniform float u_amount;
void main() {
    vec2 v = v_ndc * 0.5;
    float k = 1.0 - dot(v, v) * u_amount;
    o = vec4(vec3(clamp(k, 0.0, 1.0)), 1.0);
}
"""


# --- geometry --------------------------------------------------------------------------------------------------------

def build_world(m: GameMap) -> np.ndarray:
    """The map's floors, walls, cover, decks and roofs as rows of VERT floats."""
    rows: list = []
    W, H = m.W, m.H
    hl = [[float(v) for v in row] for row in m.hgt.tolist()]
    mat = m.mat.tolist()
    floor = m.floor.tolist()
    roof = m.roof.tolist()

    def quad(p: list, n: tuple, uv: list, info: tuple, ao: tuple = (1.0, 1.0, 1.0, 1.0)) -> None:
        for i in (0, 1, 2, 0, 2, 3):
            rows.append((*p[i], *n, *uv[i], info[0], ao[i], info[1], info[2]))

    def hh(x: int, y: int) -> float:
        return hl[y][x] if 0 <= x < W and 0 <= y < H else 99.0

    for y in range(H):
        for x in range(W):
            h = hl[y][x]
            indoor = 1.0 if roof[y][x] else 0.0
            if h < 4.9:  # a top you can see: the floor, a crate, a deck
                mt = floor[y][x] if h == 0.0 else mat[y][x]
                ao = []
                for cx, cy in ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)):
                    n = 0
                    for ox, oy in ((-1, -1), (0, -1), (-1, 0), (0, 0)):
                        if hh(cx + ox, cy + oy) > h + 0.5:
                            n += 1
                    ao.append(1.0 - 0.15 * n)
                quad([(x, h, y), (x + 1, h, y), (x + 1, h, y + 1), (x, h, y + 1)], (0.0, 1.0, 0.0),
                     [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)], (mt, indoor, h), tuple(ao))
                if roof[y][x]:
                    quad([(x, ROOF, y), (x + 1, ROOF, y), (x + 1, ROOF, y + 1), (x, ROOF, y + 1)], (0.0, -1.0, 0.0),
                         [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)], (M_ROOF, 1.0, ROOF))
            if h <= 0.0:
                continue
            mt = mat[y][x]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                nh = hl[ny][nx]
                if nh >= h:
                    continue
                nin = 1.0 if roof[ny][nx] else 0.0
                top = min(h, ROOF) if roof[ny][nx] else h
                if top <= nh:
                    continue
                if dx:
                    X = x + (1 if dx > 0 else 0)
                    a, b = ((X, y + 1), (X, y)) if dx > 0 else ((X, y), (X, y + 1))
                    ua, ub = a[1], b[1]
                else:
                    Y = y + (1 if dy > 0 else 0)
                    a, b = ((x, Y), (x + 1, Y)) if dy > 0 else ((x + 1, Y), (x, Y))
                    ua, ub = a[0], b[0]
                quad([(a[0], nh, a[1]), (b[0], nh, b[1]), (b[0], top, b[1]), (a[0], top, a[1])], (dx, 0.0, dy),
                     [(ua, nh), (ub, nh), (ub, top), (ua, top)], (mt, nin, nh))
    return np.asarray(rows, F32) if rows else np.zeros((0, VERT), F32)


def build_barriers(m: GameMap) -> dict:
    """The buy-phase fences: for each side, the edges between where it may stand and where it may not."""
    out = {}
    hl = m.hl
    for side, mask in m.zones.items():
        rows: list = []
        for y in range(1, m.H - 1):
            for x in range(1, m.W - 1):
                if not mask[y, x] or hl[y][x] >= 2.0:
                    continue
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if mask[ny, nx] or hl[ny][nx] >= 2.0:
                        continue
                    if dx:
                        X = x + (1 if dx > 0 else 0)
                        a, b = (X, y), (X, y + 1)
                    else:
                        Y = y + (1 if dy > 0 else 0)
                        a, b = (x, Y), (x + 1, Y)
                    quad = [(a[0], 0.0, a[1], 0.0, 0.0), (b[0], 0.0, b[1], 1.0, 0.0), (b[0], 3.0, b[1], 1.0, 3.0),
                            (a[0], 3.0, a[1], 0.0, 3.0)]
                    for i in (0, 1, 2, 0, 2, 3):
                        rows.append(quad[i])
        out[side] = np.asarray(rows, F32) if rows else np.zeros((0, 5), F32)
    return out


def _cube() -> np.ndarray:
    faces = (((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
             ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0)))
    rows = []
    for n, u, v in faces:
        n, u, v = np.array(n, float), np.array(u, float), np.array(v, float)
        c = [n - u - v, n + u - v, n + u + v, n - u + v]
        for i in (0, 1, 2, 0, 2, 3):
            rows.append((*c[i], *n))
    return np.asarray(rows, F32)


def _sphere(seg: int = 22, rings: int = 14) -> np.ndarray:
    pts = []
    for j in range(rings + 1):
        th = math.pi * j / rings
        row = []
        for i in range(seg + 1):
            ph = 2 * math.pi * i / seg
            row.append((math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)))
        pts.append(row)
    rows = []
    for j in range(rings):
        for i in range(seg):
            a, b, c, d = pts[j][i], pts[j][i + 1], pts[j + 1][i + 1], pts[j + 1][i]
            rows += [a, b, c, a, c, d]
    return np.asarray(rows, F32)


class _Buf:
    __slots__ = ("count", "vao", "vbo")

    def __init__(self) -> None:
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.count = 0


class Frame:
    """What to draw this frame besides the map (lists of rows; see the shaders' instance layouts)."""

    def __init__(self) -> None:
        self.boxes: list = []  # 16: centre, half-axes x y z, colour, flag
        self.outline: list = []  # the enemies' boxes again
        self.xray: list = []  # marked enemies, seen through walls
        self.view: list = []  # the player's own hands and gun
        self.decals: list = []  # 12: x up y radius, r g b a, kind k 0 0
        self.smokes: list = []  # 8: x up y radius, r g b seed
        self.shells: list = []  # 8: x up y radius, r g b -alpha
        self.beams: list = []  # 10: a, b, r g b width
        self.glows: list = []  # 8: x up y radius, r g b strength
        self.barriers = False
        self.vignette = 0.75


class Renderer:
    def __init__(self, m: GameMap) -> None:
        self.m = m
        self.scene = m.scene
        self.ctx: QOpenGLContext | None = None
        self.surface: QOffscreenSurface | None = None
        self.f: QOpenGLExtraFunctions | None = None
        self.fbo = self.resolve = None
        self.shadow_fbo = None
        self.shadow_ok = False
        self.samples = 4
        self.fov = math.radians(90)
        self.vm_fov = math.radians(62)  # the player's own gun is drawn with a narrower lens
        self.view_w, self.view_h = 1280, 780
        self.cam = np.array([0.0, 1.6, 0.0])
        self.R = np.array([0.0, 0.0, 1.0])
        self.U = np.array([0.0, 1.0, 0.0])
        self.F = np.array([1.0, 0.0, 0.0])
        self.gpu = ""
        self.ms = 0.0
        self.flip_cull = False
        self.lamps = [(x, z, y, col, rad) for x, y, z, col, rad in m.lamps][:MAX_LAMPS]

    # --- set-up -------------------------------------------------------------------------------------------------

    def _init(self) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            raise RuntimeError("Breach Point needs OpenGL 3.3 (no graphics context could be made)")
        surface = QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()
        if not ctx.makeCurrent(surface):
            raise RuntimeError("Breach Point could not use its graphics context")
        self.ctx, self.surface = ctx, surface
        f = self.f = QOpenGLExtraFunctions(ctx)
        f.initializeOpenGLFunctions()
        self.gpu = str(f.glGetString(0x1F01) or "")
        P = self._program
        self.p_world = P(_WORLD_VS, _WORLD_FS)
        self.p_depth = P(_DEPTH_VS, _DEPTH_FS)
        self.p_box = P(_BOX_VS, _BOX_FS)
        self.p_decal = P(_DECAL_VS, _DECAL_FS)
        self.p_sphere = P(_SPHERE_VS, _SPHERE_FS)
        self.p_beam = P(_BEAM_VS, _BEAM_FS)
        self.p_glow = P(_GLOW_VS, _GLOW_FS)
        self.p_barrier = P(_BARRIER_VS, _BARRIER_FS)
        self.p_sky = P(_FULL_VS, _SKY_FS)
        self.p_vig = P(_FULL_VS, _VIGNETTE_FS)
        self.loc: dict = {}
        self.empty = QOpenGLVertexArrayObject()
        self.empty.create()
        # the map
        data = build_world(self.m)
        self.world = _Buf()
        self._layout(self.world, self.p_world, ((0, 0, 3), (1, 12, 3), (2, 24, 2), (3, 32, 4)), VERT * 4)
        self._fill(self.world, data, len(data))
        self.tris = len(data) // 3
        self.barrier = {}
        for side, rows in build_barriers(self.m).items():
            b = _Buf()
            self._layout(b, self.p_barrier, ((0, 0, 3), (1, 12, 2)), 20)
            self._fill(b, rows, len(rows))
            self.barrier[side] = b
        # instanced things
        self.cube = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.cube.create()
        self.cube.bind()
        raw = _cube().tobytes()
        self.cube.allocate(raw, len(raw))
        self.cube.release()
        self.ball = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.ball.create()
        self.ball.bind()
        sph = _sphere()
        raw = sph.tobytes()
        self.ball.allocate(raw, len(raw))
        self.ball.release()
        self.ball_n = len(sph)
        self.boxes = self._instanced(self.p_box, self.cube, ((0, 0, 3), (1, 12, 3)), 24,
                                     ((2, 0, 3), (3, 12, 3), (4, 24, 3), (5, 36, 3), (6, 48, 4)), 64)
        self.spheres = self._instanced(self.p_sphere, self.ball, ((0, 0, 3),), 12, ((1, 0, 4), (2, 16, 4)), 32)
        self.decals = self._instanced(self.p_decal, None, (), 0, ((0, 0, 4), (1, 16, 4), (2, 32, 4)), 48)
        self.beams = self._instanced(self.p_beam, None, (), 0, ((0, 0, 3), (1, 12, 3), (2, 24, 4)), 40)
        self.glows = self._instanced(self.p_glow, None, (), 0, ((0, 0, 4), (1, 16, 4)), 32)
        self._make_shadow()

    def _layout(self, b: _Buf, prog: QOpenGLShaderProgram, attrs: tuple, stride: int) -> None:
        f = self.f
        b.vao.create()
        b.vao.bind()
        b.vbo.create()
        b.vbo.bind()
        for loc, off, size in attrs:
            f.glEnableVertexAttribArray(loc)
            prog.setAttributeBuffer(loc, GL_FLOAT, off, size, stride)
        b.vao.release()
        b.vbo.release()

    def _instanced(self, prog: QOpenGLShaderProgram, mesh: QOpenGLBuffer | None, mesh_attrs: tuple, mesh_stride: int,
                   inst_attrs: tuple, inst_stride: int) -> _Buf:
        f = self.f
        b = _Buf()
        b.vao.create()
        b.vao.bind()
        if mesh is not None:
            mesh.bind()
            for loc, off, size in mesh_attrs:
                f.glEnableVertexAttribArray(loc)
                prog.setAttributeBuffer(loc, GL_FLOAT, off, size, mesh_stride)
        b.vbo.create()
        b.vbo.bind()
        for loc, off, size in inst_attrs:
            f.glEnableVertexAttribArray(loc)
            prog.setAttributeBuffer(loc, GL_FLOAT, off, size, inst_stride)
            f.glVertexAttribDivisor(loc, 1)
        b.vao.release()
        b.vbo.release()
        return b

    def _program(self, vs: str, fs: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs) and ok
        if not (ok and prog.link()):
            raise RuntimeError(f"Breach Point shader did not compile: {prog.log()}")
        return prog

    def _u(self, prog: QOpenGLShaderProgram, name: str) -> int:
        key = (id(prog), name)
        loc = self.loc.get(key)
        if loc is None:
            loc = self.loc[key] = prog.uniformLocation(name)
        return loc

    def _fill(self, b: _Buf, data, rows: int) -> None:  # noqa: ANN001
        raw = np.ascontiguousarray(data, F32).tobytes()
        b.vbo.bind()
        b.vbo.allocate(raw, len(raw))
        b.vbo.release()
        b.count = rows

    def _mat(self, prog: QOpenGLShaderProgram, name: str, m: np.ndarray) -> None:
        prog.setUniformValue(self._u(prog, name), QMatrix4x4(*[float(v) for v in m.reshape(-1)]))

    def _make_shadow(self) -> None:
        """A depth picture of the map from the sun, once: the shader compares against it."""
        f = self.f
        sc = self.scene
        d = np.array([sc["sun"][0], sc["sun"][1], sc["sun"][2]], float)
        d /= np.linalg.norm(d)
        r = np.cross(d, [0.0, 1.0, 0.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, d)
        W, H = self.m.W, self.m.H
        corners = np.array([[x, y, z] for x in (0, W) for y in (0, 14.0) for z in (0, H)], float)
        px, py, pz = corners @ r, corners @ u, corners @ d
        mat = np.eye(4)
        for i, (axis, vals) in enumerate(((r, px), (u, py), (d, pz))):
            lo, hi = vals.min() - 0.5, vals.max() + 0.5
            mat[i, :3] = axis * 2.0 / (hi - lo)
            mat[i, 3] = -(hi + lo) / (hi - lo)
        self.light = mat.astype(F32)
        self.shadow_ok = False
        for internal in (GL_R32F, GL_RGBA32F):
            fmt = QOpenGLFramebufferObjectFormat()
            fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
            fmt.setInternalTextureFormat(internal)
            fbo = QOpenGLFramebufferObject(SHADOW_SIZE, SHADOW_SIZE, fmt)
            if fbo.isValid():
                self.shadow_fbo = fbo
                self.shadow_ok = True
                break
        if not self.shadow_ok:
            log.warning("Breach Point: no float framebuffer, sun shadows are off")
            return
        self.shadow_fbo.bind()
        f.glViewport(0, 0, SHADOW_SIZE, SHADOW_SIZE)
        f.glDisable(GL_BLEND)
        f.glDisable(GL_CULL_FACE)
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LESS)
        f.glDepthMask(True)
        f.glClearColor(1.0, 1.0, 1.0, 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        prog = self.p_depth
        prog.bind()
        self._mat(prog, "u_vp", self.light)
        self.world.vao.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, self.world.count)
        self.world.vao.release()
        self.shadow_fbo.release()
        self.shadow_tex = self.shadow_fbo.texture()

    def warm_up(self, w: int = 0, h: int = 0) -> None:
        """Make everything now (behind the loading screen). With a size, also draw one frame holding one of each
        kind of thing, so the first smoke or tracer of the match does not cost a hitch."""
        if self.ctx is None:
            self._init()
            self.ctx.doneCurrent()
        if w > 16 and h > 16:
            cx, cy = self.m.W / 2, self.m.H / 2
            self.set_camera(cx, cy, 1.6, 0.3, -0.1, self.fov, w, h)
            fr = Frame()
            box = (cx + 2, 1.0, cy, 0.3, 0, 0, 0, 0.3, 0, 0, 0, 0.3, 0.5, 0.5, 0.5, 0.0)
            fr.boxes, fr.outline, fr.xray, fr.view = [box], [box], [box], [box]
            fr.decals = [(cx, 0.05, cy, 1.0, 1.0, 1.0, 1.0, 0.5, float(k), 0.5, 0.0, 0.0) for k in range(5)]
            fr.smokes = [(cx, 1.5, cy + 3, 1.0, 0.8, 0.8, 0.8, 1.0)]
            fr.shells = [(cx, 1.5, cy - 3, 1.0, 0.8, 0.8, 0.8, -0.5)]
            fr.beams = [(cx, 1.0, cy, cx + 3, 1.0, cy, 1.0, 1.0, 1.0, 0.01)]
            fr.glows = [(cx, 1.0, cy, 0.5, 1.0, 1.0, 1.0, 1.0)]
            fr.barriers = True
            self.render(fr, 0.0)

    def close(self) -> None:
        if self.ctx is None:
            return
        try:
            if self.ctx.makeCurrent(self.surface):
                bufs = [self.world, self.boxes, self.spheres, self.decals, self.beams, self.glows]
                for b in bufs + list(self.barrier.values()):
                    b.vao.destroy()
                    b.vbo.destroy()
                self.cube.destroy()
                self.ball.destroy()
                self.empty.destroy()
                self.fbo = self.resolve = self.shadow_fbo = None
                self.ctx.doneCurrent()
        except Exception:  # noqa: BLE001 - closing must never fail
            log.debug("Breach Point renderer close", exc_info=True)
        self.ctx = None
        self.surface = None

    def _size(self, w: int, h: int) -> None:
        if self.fbo is not None and self.fbo.width() == w and self.fbo.height() == h:
            return
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
        fmt.setInternalTextureFormat(GL_RGB8)
        fmt.setSamples(self.samples)
        self.fbo = QOpenGLFramebufferObject(w, h, fmt)
        fmt2 = QOpenGLFramebufferObjectFormat()
        fmt2.setInternalTextureFormat(GL_RGB8)
        self.resolve = QOpenGLFramebufferObject(w, h, fmt2)

    # --- the camera ---------------------------------------------------------------------------------------------

    def set_camera(self, x: float, y: float, eye: float, yaw: float, pitch: float, fov: float, w: int,
                   h: int) -> None:
        ca, sa, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        self.cam = np.array([x, eye, y])
        self.F = np.array([cp * ca, sp, cp * sa])
        self.R = np.array([-sa, 0.0, ca])
        self.U = np.cross(self.R, self.F)
        self.fov, self.view_w, self.view_h = fov, w, h

    def _tans(self, fov: float | None = None) -> tuple[float, float]:
        tx = math.tan((fov or self.fov) / 2)
        return tx, tx * self.view_h / max(self.view_w, 1)

    def _vp(self, fov: float | None = None) -> np.ndarray:
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = self.R, self.U, -self.F
        view[:3, 3] = -(view[:3, :3] @ self.cam)
        tx, ty = self._tans(fov)
        near, far = 0.05, 220.0
        proj = np.zeros((4, 4))
        proj[0, 0] = 1.0 / tx
        proj[1, 1] = -1.0 / ty  # upside down (see the module docstring)
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2 * far * near / (near - far)
        proj[3, 2] = -1.0
        return (proj @ view).astype(F32)

    @property
    def proj(self) -> float:
        """Pixels (of the widget) per unit of the view plane at distance 1."""
        return (self.view_w / 2) / math.tan(self.fov / 2)

    def project(self, x: float, y: float, z: float) -> tuple[float, float, float] | None:
        """Widget (sx, sy) and depth of the map point (x, y) at height z; None behind the eye."""
        d = np.array([x, z, y]) - self.cam
        zc = float(d @ self.F)
        if zc < 0.05:
            return None
        k = self.proj / zc
        return self.view_w / 2 + float(d @ self.R) * k, self.view_h / 2 - float(d @ self.U) * k, zc

    # --- the frame ----------------------------------------------------------------------------------------------

    def _common(self, prog: QOpenGLShaderProgram, vp: np.ndarray) -> None:
        f, sc = self.f, self.scene
        prog.bind()
        self._mat(prog, "u_vp", vp)
        u = self._u
        f.glUniform3f(u(prog, "u_cam"), *[float(v) for v in self.cam])
        sun = np.array(sc["sun"], float)
        sun /= np.linalg.norm(sun)
        f.glUniform3f(u(prog, "u_sun"), float(sun[0]), float(sun[1]), float(sun[2]))
        k = sc["sun_k"]
        f.glUniform3f(u(prog, "u_sun_col"), *[c * k for c in sc["sun_col"]])
        f.glUniform3f(u(prog, "u_amb_sky"), *sc["amb_sky"])
        f.glUniform3f(u(prog, "u_amb_gnd"), *sc["amb_gnd"])
        f.glUniform3f(u(prog, "u_amb_in"), *sc["amb_in"])
        f.glUniform3f(u(prog, "u_fog"), *sc["fog"])
        f.glUniform1f(u(prog, "u_fogd"), float(sc["fog_d"]))
        f.glUniform1f(u(prog, "u_shadow_on"), 1.0 if self.shadow_ok else 0.0)
        if self.shadow_ok:
            self._mat(prog, "u_light", self.light)
            prog.setUniformValue1i(u(prog, "u_shadow"), 1)
        f.glUniform1i(u(prog, "u_nl"), len(self.lamps))
        for i, (X, Y, Z, col, rad) in enumerate(self.lamps):
            f.glUniform4f(u(prog, f"u_lpos[{i}]"), float(X), float(Y), float(Z), float(rad))
            f.glUniform3f(u(prog, f"u_lcol[{i}]"), float(col[0]) * 0.55, float(col[1]) * 0.55, float(col[2]) * 0.55)

    def _draw_boxes(self, rows: list, vp: np.ndarray, mode: float = 0.0, flat: tuple = (0, 0, 0, 1),
                    grow: float = 0.0, indoor: float = 0.0, common: bool = True) -> None:
        if not rows:
            return
        f = self.f
        prog = self.p_box
        if common:
            self._common(prog, vp)
        else:
            prog.bind()
        f.glUniform1f(self._u(prog, "u_mode"), mode)
        f.glUniform4f(self._u(prog, "u_flat"), *[float(v) for v in flat])
        f.glUniform1f(self._u(prog, "u_grow"), grow)
        f.glUniform1f(self._u(prog, "u_indoor"), indoor)
        self._fill(self.boxes, np.asarray(rows, F32), len(rows))
        self.boxes.vao.bind()
        f.glDrawArraysInstanced(GL_TRIANGLES, 0, 36, len(rows))
        self.boxes.vao.release()

    def render(self, fr: Frame, t: float = 0.0, dpr: float = 1.0) -> QImage:
        """One frame from the camera last given to ``set_camera``."""
        t0 = time.perf_counter()
        if self.ctx is None:
            self._init()
        elif not self.ctx.makeCurrent(self.surface):
            raise RuntimeError("Breach Point lost its graphics context")
        f = self.f
        w, h = self.view_w, self.view_h
        pw, ph = max(int(w * dpr), 16), max(int(h * dpr), 16)
        self._size(pw, ph)
        vp = self._vp()
        sc = self.scene
        self.fbo.bind()
        f.glViewport(0, 0, pw, ph)
        f.glDisable(GL_BLEND)
        f.glDisable(GL_CULL_FACE)
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LESS)
        f.glDepthMask(True)
        fog = sc["fog"]
        f.glClearColor(fog[0], fog[1], fog[2], 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.shadow_ok:
            f.glActiveTexture(GL_TEXTURE0 + 1)
            f.glBindTexture(GL_TEXTURE_2D, self.shadow_tex)
            f.glActiveTexture(GL_TEXTURE0)
        # the map
        prog = self.p_world
        self._common(prog, vp)
        for i in range(20):
            col = sc["pal"].get(i, (0.5, 0.5, 0.5))
            f.glUniform3f(self._u(prog, f"u_pal[{i}]"), *col)
        f.glUniform3f(self._u(prog, "u_accent"), *sc["accent"])
        for i, s in enumerate(list(self.m.sites.values())[:2]):
            f.glUniform4f(self._u(prog, f"u_site[{i}]"), *[float(v) for v in s.rect])
        self.world.vao.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, self.world.count)
        self.world.vao.release()
        # agents, guns, props
        self._draw_boxes(fr.boxes, vp)
        if fr.outline:
            f.glEnable(GL_CULL_FACE)
            f.glCullFace(GL_BACK if self.flip_cull else GL_FRONT)
            self._draw_boxes(fr.outline, vp, 1.0, (1.0, 0.24, 0.55, 1.0), 0.022, common=False)
            f.glDisable(GL_CULL_FACE)
        # smoke
        if fr.smokes:
            prog = self.p_sphere
            self._common(prog, vp)
            f.glUniform1f(self._u(prog, "u_time"), float(t))
            f.glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE)
            self._fill(self.spheres, np.asarray(fr.smokes, F32), len(fr.smokes))
            self.spheres.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLES, 0, self.ball_n, len(fr.smokes))
            self.spheres.vao.release()
            f.glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE)
        # the sky where nothing was drawn
        f.glDepthFunc(GL_LEQUAL)
        f.glDepthMask(False)
        prog = self.p_sky
        prog.bind()
        tx, ty = self._tans()
        f.glUniform3f(self._u(prog, "u_r"), *[float(v) * tx for v in self.R])
        f.glUniform3f(self._u(prog, "u_u"), *[float(v) * ty for v in self.U])
        f.glUniform3f(self._u(prog, "u_f"), *[float(v) for v in self.F])
        f.glUniform1f(self._u(prog, "u_time"), float(t))
        f.glUniform3f(self._u(prog, "u_top"), *sc["sky_top"])
        f.glUniform3f(self._u(prog, "u_low"), *sc["sky_low"])
        f.glUniform3f(self._u(prog, "u_fog"), *fog)
        f.glUniform3f(self._u(prog, "u_disc"), *sc["sun_disc"])
        sun = np.array(sc["sun"], float)
        sun /= np.linalg.norm(sun)
        f.glUniform3f(self._u(prog, "u_sundir"), float(-sun[0]), float(-sun[1]), float(-sun[2]))
        self.empty.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.empty.release()
        # see-through things
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        if fr.decals:
            prog = self.p_decal
            prog.bind()
            self._mat(prog, "u_vp", vp)
            f.glUniform1f(self._u(prog, "u_time"), float(t))
            self._fill(self.decals, np.asarray(fr.decals, F32), len(fr.decals))
            self.decals.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(fr.decals))
            self.decals.vao.release()
        if fr.barriers:
            prog = self.p_barrier
            prog.bind()
            self._mat(prog, "u_vp", vp)
            f.glUniform1f(self._u(prog, "u_time"), float(t))
            f.glUniform3f(self._u(prog, "u_col"), *sc["accent"])
            for b in self.barrier.values():
                if b.count:
                    b.vao.bind()
                    f.glDrawArrays(GL_TRIANGLES, 0, b.count)
                    b.vao.release()
        if fr.shells:
            prog = self.p_sphere
            self._common(prog, vp)
            f.glUniform1f(self._u(prog, "u_time"), float(t))
            self._fill(self.spheres, np.asarray(fr.shells, F32), len(fr.shells))
            self.spheres.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLES, 0, self.ball_n, len(fr.shells))
            self.spheres.vao.release()
        if fr.xray:
            f.glDepthFunc(GL_GREATER)
            self._draw_boxes(fr.xray, vp, 1.0, (1.0, 0.24, 0.55, 0.5))
            f.glDepthFunc(GL_LEQUAL)
        f.glBlendFunc(GL_ONE, GL_ONE)
        if fr.beams:
            prog = self.p_beam
            prog.bind()
            self._mat(prog, "u_vp", vp)
            f.glUniform3f(self._u(prog, "u_cam"), *[float(v) for v in self.cam])
            self._fill(self.beams, np.asarray(fr.beams, F32), len(fr.beams))
            self.beams.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(fr.beams))
            self.beams.vao.release()
        if fr.glows:
            prog = self.p_glow
            prog.bind()
            self._mat(prog, "u_vp", vp)
            f.glUniform3f(self._u(prog, "u_right"), *[float(v) for v in self.R])
            f.glUniform3f(self._u(prog, "u_up"), *[float(v) for v in self.U])
            f.glUniform3f(self._u(prog, "u_cam"), *[float(v) for v in self.cam])
            self._fill(self.glows, np.asarray(fr.glows, F32), len(fr.glows))
            self.glows.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(fr.glows))
            self.glows.vao.release()
        f.glDisable(GL_BLEND)
        # the player's own hands and gun, over everything
        if fr.view:
            f.glDepthMask(True)
            f.glDepthFunc(GL_LESS)
            f.glClear(GL_DEPTH_BUFFER_BIT)
            cx, cy = int(self.cam[0]), int(self.cam[2])
            indoor = 1.0 if self.m.inside(cx, cy) and self.m.roofl[cy][cx] else 0.0
            self._draw_boxes(fr.view, self._vp(self.vm_fov), indoor=indoor)
            f.glDepthMask(False)
        # darker corners
        f.glDisable(GL_DEPTH_TEST)
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_DST_COLOR, GL_ZERO)
        prog = self.p_vig
        prog.bind()
        f.glUniform1f(self._u(prog, "u_amount"), float(fr.vignette))
        self.empty.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.empty.release()
        f.glDisable(GL_BLEND)
        f.glDepthMask(True)
        self.fbo.release()
        QOpenGLFramebufferObject.blitFramebuffer(self.resolve, self.fbo)
        img = self.resolve.toImage(False)
        self.ctx.doneCurrent()
        if dpr != 1.0:
            img.setDevicePixelRatio(dpr)
        self.ms = (time.perf_counter() - t0) * 1000
        return img


__all__ = ["Frame", "Renderer"]
