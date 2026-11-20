"""The 3D view, drawn by the graphics card (OpenGL 3.3 through Qt's own classes, no other package) at the window's
own resolution into an image the game widget paints.

The map becomes one static buffer of textured triangles (walls with their real heights, floors, ceilings, crates,
barriers, round tanks and drums), the see-through panes (boarded windows, glazed openings, fences) a small second
buffer rebuilt when planks come off. Textures are sharp texel-edged up close and mip-mapped far away; surfaces are lit
by the room's ambient, the sun or moon outdoors, up to 16 lamps and moving lights (kept in their own room) and the
torch on the player's weapon, then fogged. Billboards (zombies, soldiers, machines) are alpha-to-coverage cut-outs
in 4x multisampling; glows are additive quads; the sky is one full-screen pass. The picture is drawn upside down on
purpose so it can be read back without a flip."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtGui import (QImage, QMatrix4x4, QOffscreenSurface, QOpenGLContext, QOpenGLExtraFunctions,
                           QSurfaceFormat)
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
                              QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture, QOpenGLVertexArrayObject)

from .maps import FACES, WALL_H, MapData
from .textures import TS

log = logging.getLogger(__name__)
F32 = np.float32
EYE = 0.82  # standing eye height (cells)
MAX_LIGHTS = 16
VERT_FLOATS = 12  # position 3, uv 2, normal 3, layer / outdoors / room / occlusion 4
GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_FLOAT = 0x0004, 0x0005, 0x1406
GL_DEPTH_TEST, GL_CULL_FACE, GL_BLEND, GL_SAMPLE_ALPHA_TO_COVERAGE = 0x0B71, 0x0B44, 0x0BE2, 0x809E
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_LEQUAL, GL_LESS = 0x4000, 0x0100, 0x0203, 0x0201
GL_ZERO, GL_ONE, GL_DST_COLOR, GL_RGB8 = 0, 1, 0x0306, 0x8051


@dataclass
class Sprite:
    """A billboard standing at (x, y) of the map: its bottom ``z`` above the floor, ``w`` x ``h`` in cells."""
    x: float
    y: float
    z: float
    w: float
    h: float
    img: np.ndarray  # (ih, iw, 4) uint8 RGBA
    tint: tuple | None = None  # (r, g, b, amount) flash
    bright: bool = False  # lit from inside (a machine's sign): not darkened by the room
    eyes: tuple = ()  # ((u, v), ...) image fractions of glowing eyes
    eye_color: tuple = (150, 255, 170)
    flip: bool = False


# --- shaders ---------------------------------------------------------------------------------------------------------

_LIGHT = """
uniform vec3 u_cam; uniform vec3 u_fwd;
uniform vec3 u_amb_in; uniform vec3 u_amb_out; uniform vec4 u_sun; uniform vec3 u_sun_col;
uniform vec4 u_lpos[16]; uniform vec4 u_lcol[16]; uniform int u_nl; uniform float u_torch;
uniform vec3 u_fog; uniform float u_fogd;
vec3 lighting(vec3 p, vec3 n, float sky, float zone, float ao, float dirw) {
    vec3 c = mix(u_amb_in, u_amb_out, sky) * ao * (0.84 + 0.16 * abs(n.x) + 0.08 * n.y);
    if (sky > 0.0) {
        float sd = max(dot(n, -normalize(u_sun.xyz)), 0.0);
        c += u_sun_col * u_sun.w * mix(0.55, sd, dirw) * sky;
    }
    for (int i = 0; i < 16; i++) {
        if (i >= u_nl) break;
        vec4 L = u_lpos[i]; vec4 C = u_lcol[i];
        if (C.w >= 0.0 && zone >= 0.0 && abs(C.w - zone) > 0.5) continue;
        vec3 d = L.xyz - p; float dist = length(d);
        float a = clamp(1.0 - dist / L.w, 0.0, 1.0);
        a *= a;
        float nd = mix(1.0, max(dot(n, d / max(dist, 1e-4)), 0.0) * 0.8 + 0.2, dirw);
        c += C.rgb * a * nd;
    }
    if (u_torch > 0.0) {
        vec3 d = p - u_cam; float dist = length(d); vec3 dn = d / max(dist, 1e-4);
        float cone = smoothstep(0.80, 0.965, dot(dn, u_fwd));
        float att = clamp(1.0 - dist / 15.0, 0.0, 1.0) * (0.45 + 0.55 * smoothstep(0.4, 3.0, dist));
        float nd = mix(1.0, max(dot(n, -dn), 0.0), dirw);
        c += vec3(1.0, 0.94, 0.82) * u_torch * (cone * 0.9 + 0.1) * att * att * nd;
    }
    return c;
}
vec3 knee(vec3 c) {  // bright light rolls off instead of clipping flat
    vec3 e = max(c - 0.78, 0.0);
    return min(c, 0.78) + e / (1.0 + e * 4.5);
}
vec3 fogged(vec3 rgb, vec3 p) {
    float f = 1.0 - exp(-pow(length(p - u_cam) / u_fogd, 1.35));
    return mix(rgb, u_fog, min(f, 0.97));
}
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
"""

_WORLD_VS = """#version 330 core
layout(location = 0) in vec3 a_pos; layout(location = 1) in vec2 a_uv; layout(location = 2) in vec3 a_nrm;
layout(location = 3) in vec4 a_info;
uniform mat4 u_vp;
out vec3 v_pos; out vec2 v_uv; out vec3 v_nrm; flat out float v_layer; out float v_sky; flat out float v_zone;
out float v_ao;
void main() {
    v_pos = a_pos; v_uv = a_uv; v_nrm = a_nrm;
    v_layer = a_info.x; v_sky = a_info.y; v_zone = a_info.z; v_ao = a_info.w;
    gl_Position = u_vp * vec4(a_pos, 1.0);
}
"""

_WORLD_FS = "#version 330 core\n" + _LIGHT + f"const float TS = {float(TS * 2)};\n" + """
in vec3 v_pos; in vec2 v_uv; in vec3 v_nrm; flat in float v_layer; in float v_sky; flat in float v_zone; in float v_ao;
uniform sampler2DArray u_tex;
out vec4 o;
void main() {
    vec2 px = v_uv * TS;  // sharp texel edges up close, smooth when far (the mip-maps take over)
    vec2 seam = floor(px + 0.5);
    vec2 q = (seam + clamp((px - seam) / max(fwidth(px), vec2(1e-4)), -0.5, 0.5)) / TS;
    vec4 c = textureGrad(u_tex, vec3(q, v_layer), dFdx(v_uv), dFdy(v_uv));
    if (c.a < 0.5) discard;
    vec3 rgb = c.rgb / c.a;
    vec3 n = normalize(v_nrm);
    if (dot(n, u_cam - v_pos) < 0.0) n = -n;
    float dist = length(v_pos - u_cam);
    rgb *= 1.0 + (hash(floor(v_uv * TS * 2.0)) - 0.5) * 0.12 * clamp(1.0 - dist / 5.0, 0.0, 1.0);
    float side = 1.0 - abs(n.y);
    float contact = mix(1.0, 0.6 + 0.4 * smoothstep(0.0, 0.5, v_pos.y), side);
    vec3 lit = rgb * lighting(v_pos, n, v_sky, v_zone, v_ao * contact, 1.0);
    o = vec4(fogged(knee(lit), v_pos), 1.0);
}
"""

_SPRITE_VS = """#version 330 core
uniform mat4 u_vp; uniform vec3 u_base; uniform vec2 u_size; uniform vec3 u_right; uniform float u_flip;
out vec2 v_uv; out vec3 v_pos;
void main() {
    float u = float(gl_VertexID & 1), v = float(gl_VertexID >> 1);
    vec3 p = u_base + u_right * (u - 0.5) * u_size.x + vec3(0.0, v * u_size.y, 0.0);
    v_pos = p;
    v_uv = vec2(u_flip > 0.5 ? 1.0 - u : u, 1.0 - v);
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

_SPRITE_FS = "#version 330 core\n" + _LIGHT + """
in vec2 v_uv; in vec3 v_pos;
uniform sampler2D u_img; uniform vec4 u_tint; uniform float u_bright; uniform float u_sky; uniform float u_zone;
out vec4 o;
void main() {
    vec4 c = texture(u_img, v_uv);
    float a = clamp((c.a - 0.5) / max(fwidth(c.a), 1e-3) + 0.5, 0.0, 1.0);
    if (a < 0.02) discard;
    vec3 rgb = c.rgb / max(c.a, 1e-3);
    rgb = mix(rgb, u_tint.rgb, u_tint.a);
    vec3 n = normalize(vec3(u_cam.x - v_pos.x, 0.0, u_cam.z - v_pos.z) + vec3(1e-4));
    vec3 lit = rgb * lighting(v_pos, n, u_sky, u_zone, 1.0, 0.35);
    if (u_bright > 0.5) lit = max(lit, rgb * 0.95);
    o = vec4(fogged(knee(lit), v_pos), a);
}
"""

_GLOW_VS = """#version 330 core
layout(location = 0) in vec4 a_p; layout(location = 1) in vec4 a_c;
uniform mat4 u_vp; uniform vec3 u_right; uniform vec3 u_up; uniform vec3 u_cam;
out vec2 v_q; out vec4 v_c;
void main() {
    float u = float(gl_VertexID & 1) * 2.0 - 1.0, v = float(gl_VertexID >> 1) * 2.0 - 1.0;
    float dist = length(u_cam - a_p.xyz);
    vec3 c = a_p.xyz + normalize(u_cam - a_p.xyz + vec3(1e-4)) * min(a_p.w * 0.6, dist * 0.5);
    vec3 p = c + (u_right * u + u_up * v) * a_p.w;
    v_q = vec2(u, v);
    v_c = vec4(a_c.rgb, a_c.a * smoothstep(0.2, a_p.w * 2.5 + 0.3, dist));  // fades when the eye is inside it
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
uniform vec3 u_r; uniform vec3 u_u; uniform vec3 u_f; uniform float u_day; uniform float u_time; uniform vec3 u_fog;
uniform vec3 u_sundir;
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
float skyline(float az, float seed) {  // far-off sheds, gantries and stacks all round the horizon
    float x = az * 9.549 * 2.0;
    float cell = floor(x);
    float h = 0.018 + hash(vec2(cell, seed)) * 0.05;
    if (hash(vec2(cell, seed + 7.0)) > 0.8) h += 0.03;
    float f = fract(x);
    if (hash(vec2(cell, seed + 3.0)) > 0.78 && f > 0.42 && f < 0.52) h += 0.075;
    return h;
}
void main() {
    vec3 d = normalize(u_f + u_r * v_ndc.x - u_u * v_ndc.y);
    float el = d.y, az = atan(d.z, d.x);
    vec3 c;
    if (u_day < 0.5) {
        c = mix(vec3(0.10, 0.12, 0.19), vec3(0.015, 0.02, 0.05), pow(clamp(el, 0.0, 1.0), 0.45));
        c += vec3(0.30, 0.14, 0.05) * exp(-max(el, 0.0) * 12.0) * (0.55 + 0.45 * sin(az * 1.7 + 0.6));
        if (el > 0.0) {
            vec3 a = abs(d); float m = max(a.x, max(a.y, a.z));
            vec2 uv = m == a.y ? d.xz / m : (m == a.x ? d.yz / m : d.xy / m);
            float h = hash(floor(uv * 140.0) + (m == a.y ? 7.0 : (m == a.x ? 19.0 : 31.0)) * sign(dot(d, vec3(1.0))));
            if (h > 0.9965) c += vec3(0.8, 0.85, 1.0) * (0.5 + 0.5 * sin(u_time * 1.3 + h * 900.0))
                * clamp(el * 5.0, 0.0, 1.0);
            vec3 mdir = normalize(vec3(0.55, 0.42, -0.72));
            float md = dot(d, mdir);
            c += vec3(0.18, 0.2, 0.26) * pow(max(md, 0.0), 60.0);
            if (md > 0.99935) c = mix(vec3(0.82, 0.86, 0.95), vec3(0.6, 0.64, 0.72),
                                      vnoise(d.xy * 900.0) * 0.6);
            float cl = vnoise(d.xz / max(el, 0.05) * 1.6 + vec2(u_time * 0.01, 0.0));
            c = mix(c, vec3(0.05, 0.06, 0.09), smoothstep(0.55, 0.85, cl) * 0.7 * smoothstep(0.02, 0.25, el));
        }
        if (el < skyline(az, 3.0)) {
            c = vec3(0.012, 0.014, 0.024);
            vec2 w = vec2(az * 300.0, el * 300.0);
            if (hash(floor(w)) > 0.985 && el > 0.004) c = vec3(0.9, 0.62, 0.3) * 0.6;
        }
    } else {
        c = mix(vec3(0.70, 0.80, 0.88), vec3(0.24, 0.46, 0.80), pow(clamp(el, 0.0, 1.0), 0.55));
        float sd = dot(d, normalize(u_sundir));
        c += vec3(1.0, 0.86, 0.6) * (pow(max(sd, 0.0), 300.0) * 1.6 + pow(max(sd, 0.0), 8.0) * 0.18);
        if (el > 0.01) {
            vec2 cp = d.xz / el * 1.2 + vec2(u_time * 0.015, u_time * 0.004);
            float n = vnoise(cp) * 0.6 + vnoise(cp * 2.3 + 5.0) * 0.3 + vnoise(cp * 5.1 + 9.0) * 0.1;
            float k = smoothstep(0.52, 0.78, n) * smoothstep(0.01, 0.2, el);
            c = mix(c, vec3(0.96, 0.97, 0.98) - vec3(0.12) * smoothstep(0.7, 0.95, n), k * 0.85);
        }
        if (el < skyline(az, 11.0)) c = mix(vec3(0.52, 0.58, 0.64), u_fog / 255.0, 0.45);
    }
    if (el < 0.0) c = u_fog / 255.0;
    c = mix(c, u_fog / 255.0, exp(-abs(el) * 22.0) * 0.55);
    o = vec4(c, 1.0);
}
"""

_VIGNETTE_FS = """#version 330 core
in vec2 v_ndc; out vec4 o;
uniform float u_amount; uniform vec3 u_tint;
void main() {
    vec2 v = v_ndc * 0.5;
    float k = 1.0 - dot(v, v) * u_amount;
    o = vec4(u_tint * clamp(k, 0.0, 1.0), 1.0);
}
"""


# --- geometry --------------------------------------------------------------------------------------------------------

class _Builder:
    """Triangles as rows of VERT_FLOATS floats."""

    def __init__(self) -> None:
        self.rows: list = []

    def quad(self, p: list, uv: list, n: tuple, layer: int, sky: float, zone: float, ao=(1.0, 1.0, 1.0, 1.0)) -> None:
        """``p``: four (X, Y, Z) corners in order round the quad."""
        if isinstance(ao, float):
            ao = (ao, ao, ao, ao)
        for i in (0, 1, 2, 0, 2, 3):
            self.rows.append((*p[i], *uv[i], *n, float(layer), sky, zone, ao[i]))

    def tri(self, p: list, uv: list, n: list, layer: int, sky: float, zone: float) -> None:
        for i in range(3):
            self.rows.append((*p[i], *uv[i], *n[i], float(layer), sky, zone, 1.0))

    def array(self) -> np.ndarray:
        if not self.rows:
            return np.zeros((0, VERT_FLOATS), F32)
        return np.asarray(self.rows, F32)


def _blocking(m: MapData, x: int, y: int) -> tuple[float, float] | None:
    """The opaque block of a cell: (bottom, top) of its wall (or of the wall over its doorway / window)."""
    if not m.inside(x, y):
        return None
    hgt = float(m.hgt[y, x])
    if m.solid[y, x]:
        return 0.0, (hgt if hgt > 0 else WALL_H)
    if hgt > WALL_H:
        return WALL_H, hgt
    return None


def _spans(lo: float, hi: float, cut: tuple[float, float] | None) -> list[tuple[float, float]]:
    if cut is None:
        return [(lo, hi)]
    out = []
    if cut[0] > lo:
        out.append((lo, min(hi, cut[0])))
    if cut[1] < hi:
        out.append((max(lo, cut[1]), hi))
    return [(a, b) for a, b in out if b - a > 1e-4]


def build_static(m: MapData) -> np.ndarray:
    """Walls, floors, ceilings, boxes and round tanks of the map as it is now (doors open or not)."""
    B = _Builder()
    W, H = m.w, m.h
    occ = np.zeros((H + 2, W + 2), np.float32)  # what darkens the floor at a corner: walls and tall boxes
    occ[1:-1, 1:-1] = (m.solid | (m.top > 0.3)).astype(np.float32)
    corner = 1.0 - 0.17 * (occ[:-1, :-1] + occ[1:, :-1] + occ[:-1, 1:] + occ[1:, 1:])  # (H + 1, W + 1)
    door_cells = {c for d in m.doors for c in d.cells}
    for y in range(H):
        for x in range(W):
            solid = bool(m.solid[y, x])
            sky = 1.0 if m.ceil[y, x] < 0 else 0.0
            zone = float(m.zone[y, x]) if m.zone is not None else -1.0
            if not solid or (x, y) in door_cells:
                ao = (corner[y, x], corner[y, x + 1], corner[y + 1, x + 1], corner[y + 1, x])
                B.quad([(x, 0, y), (x + 1, 0, y), (x + 1, 0, y + 1), (x, 0, y + 1)],
                       [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)], (0, 1, 0), int(m.floor[y, x]), sky, zone,
                       ao)
                if m.ceil[y, x] >= 0:
                    B.quad([(x, WALL_H, y), (x + 1, WALL_H, y), (x + 1, WALL_H, y + 1), (x, WALL_H, y + 1)],
                           [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)], (0, -1, 0), int(m.ceil[y, x]), 0.0,
                           zone, ao)
            blk = _blocking(m, x, y)
            if blk is None:
                continue
            lo, hi = blk
            wall_tex = int(m.wall[y, x]) if m.wall[y, x] >= 0 else int(m.upper[y, x])
            up_tex = int(m.upper[y, x]) if m.upper[y, x] >= 0 else wall_tex
            if up_tex >= 0:  # the top of the wall
                B.quad([(x, hi, y), (x + 1, hi, y), (x + 1, hi, y + 1), (x, hi, y + 1)],
                       [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)], (0, 1, 0), up_tex, 1.0, -1.0, 0.9)
            for f, (dx, dy) in enumerate(FACES):
                nx, ny = x + dx, y + dy
                if not m.inside(nx, ny):
                    continue
                nsky = 1.0 if m.ceil[ny, nx] < 0 else 0.0
                nzone = float(m.zone[ny, nx]) if m.zone is not None else -1.0
                top = hi if nsky > 0 or m.solid[ny, nx] else min(hi, WALL_H)
                if top <= lo:
                    continue
                rx, rz = dy, -dx  # the viewer's right, facing the wall
                cx, cz = x + 0.5 + dx * 0.5, y + 0.5 + dy * 0.5
                ax, az = cx - rx * 0.5, cz - rz * 0.5
                bx, bz = cx + rx * 0.5, cz + rz * 0.5
                for a, b in _spans(lo, top, _blocking(m, nx, ny)):
                    for s0, s1, upper in ((a, min(b, WALL_H), False), (max(a, WALL_H), b, True)):
                        if s1 - s0 < 1e-4:
                            continue
                        if upper:
                            tex, base, unit = up_tex, WALL_H, WALL_H
                        else:
                            tex = int(m.face[f, y, x]) if solid and m.face[f, y, x] >= 0 else wall_tex
                            base, unit = 0.0, min(hi, WALL_H)
                        if tex < 0:
                            continue
                        v0, v1 = 1.0 - (s0 - base) / unit, 1.0 - (s1 - base) / unit
                        B.quad([(ax, s0, az), (bx, s0, bz), (bx, s1, bz), (ax, s1, az)],
                               [(0.0, v0), (1.0, v0), (1.0, v1), (0.0, v1)], (dx, 0, dy), tex, nsky, nzone)
    for x0, y0, x1, y1, h, side, top in m.boxes:
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        sky = 1.0 if m.ceil[cy, cx] < 0 else 0.0
        zone = float(m.zone[cy, cx]) if m.zone is not None else -1.0
        unit = h if h < 0.7 else h / max(round(h / 0.62), 1)
        B.quad([(x0, h, y0), (x1, h, y0), (x1, h, y1), (x0, h, y1)],
               [(0, 0), (x1 - x0, 0), (x1 - x0, y1 - y0), (0, y1 - y0)], (0, 1, 0), top, sky, zone)
        sides = (((x1, y1), (x1, y0), (1, 0)), ((x0, y0), (x0, y1), (-1, 0)),
                 ((x0, y1), (x1, y1), (0, 1)), ((x1, y0), (x0, y0), (0, -1)))
        for (ax, az), (bx, bz), (dx, dy) in sides:
            ln = abs(bx - ax) + abs(bz - az)
            v0, v1 = 1.0, 1.0 - h / unit
            B.quad([(ax, 0, az), (bx, 0, bz), (bx, h, bz), (ax, h, az)],
                   [(0, v0), (ln, v0), (ln, v1), (0, v1)], (dx, 0, dy), side, sky, zone)
    for cx, cy, rad, h, side, top in m.rounds:
        sky = 1.0 if m.ceil[int(cy), int(cx)] < 0 else 0.0
        zone = float(m.zone[int(cy), int(cx)]) if m.zone is not None else -1.0
        n = 28 if rad > 0.5 else 14
        reps = max(round(2 * math.pi * rad / 1.6), 1)
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            p0 = (cx + math.cos(a0) * rad, cy + math.sin(a0) * rad)
            p1 = (cx + math.cos(a1) * rad, cy + math.sin(a1) * rad)
            n0, n1 = (math.cos(a0), 0.0, math.sin(a0)), (math.cos(a1), 0.0, math.sin(a1))
            u0, u1 = reps * i / n, reps * (i + 1) / n
            B.tri([(p0[0], 0, p0[1]), (p1[0], 0, p1[1]), (p1[0], h, p1[1])], [(u0, 1), (u1, 1), (u1, 0)],
                  [n0, n1, n1], side, sky, zone)
            B.tri([(p0[0], 0, p0[1]), (p1[0], h, p1[1]), (p0[0], h, p0[1])], [(u0, 1), (u1, 0), (u0, 0)],
                  [n0, n1, n0], side, sky, zone)
            up = (0.0, 1.0, 0.0)
            B.tri([(cx, h, cy), (p0[0], h, p0[1]), (p1[0], h, p1[1])],
                  [(0.5, 0.5), (0.5 + math.cos(a0) * 0.5, 0.5 + math.sin(a0) * 0.5),
                   (0.5 + math.cos(a1) * 0.5, 0.5 + math.sin(a1) * 0.5)], [up, up, up], top, sky, zone)
    return B.array()


def build_panes(m: MapData) -> np.ndarray:
    """The see-through panes: one two-sided plane through the middle of each cell, along its wall."""
    B = _Builder()
    ys, xs = np.nonzero(m.glass >= 0)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        tex = int(m.glass[y, x])
        zone = -1.0
        along_y = (m.inside(x, y - 1) and m.solid[y - 1, x]) or (m.inside(x, y + 1) and m.solid[y + 1, x])
        sky = 1.0 if m.ceil[y, x] < 0 else 0.0
        if along_y:
            p = [(x + 0.5, 0, y), (x + 0.5, 0, y + 1), (x + 0.5, WALL_H, y + 1), (x + 0.5, WALL_H, y)]
            n = (1, 0, 0)
        else:
            p = [(x, 0, y + 0.5), (x + 1, 0, y + 0.5), (x + 1, WALL_H, y + 0.5), (x, WALL_H, y + 0.5)]
            n = (0, 0, 1)
        B.quad(p, [(0, 1), (1, 1), (1, 0), (0, 0)], n, tex, max(sky, 0.5), zone)
    return B.array()


def _mips(rgba: np.ndarray) -> list[np.ndarray]:
    """Premultiplied-alpha mip levels of (n, s, s, 4) images down to 1 pixel."""
    a = rgba[..., 3:4].astype(F32) / 255
    rgb = rgba[..., :3].astype(F32) * a
    out = [np.concatenate([rgb, a * 255], -1)]
    size = rgba.shape[1]
    lv = out[0]
    while size > 1:
        size //= 2
        lv = lv.reshape(len(lv), size, 2, size, 2, 4).mean((2, 4))
        out.append(lv)
    return [np.clip(x + 0.5, 0, 255).astype(np.uint8) for x in out]


class _Buf:
    __slots__ = ("count", "vao", "vbo")

    def __init__(self) -> None:
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.count = 0


# --- the renderer ----------------------------------------------------------------------------------------------------

class Renderer:
    """Draws a map, its billboards and lights from the player's eye. Everything OpenGL is made the first time it draws
    (or in ``warm_up``, behind the loading screen)."""

    def __init__(self, m: MapData, atlas) -> None:  # noqa: ANN001 - textures.Atlas
        self.m = m
        self.atlas = atlas
        self.scene = m.scene
        self.ctx: QOpenGLContext | None = None
        self.surface: QOffscreenSurface | None = None
        self.f: QOpenGLExtraFunctions | None = None
        self.fbo = self.resolve = None
        self.samples = 4
        self._static_dirty = True
        self._panes_dirty = True
        self._sprite_tex: dict[int, tuple] = {}
        self.lamps = [(x, z, y, (r / 255, g / 255, b / 255), rad, float(m.zone[int(y), int(x)]))
                      for x, y, z, (r, g, b), rad in m.lamps]
        # the camera of the last frame (``project`` uses it)
        self.fov = math.radians(85)
        self.view_w, self.view_h = 1280, 780
        self.cam = np.array([0.0, EYE, 0.0])
        self.R = np.array([0.0, 0.0, 1.0])
        self.U = np.array([0.0, 1.0, 0.0])
        self.F = np.array([1.0, 0.0, 0.0])
        self.gpu = ""
        self.ms = 0.0

    # --- set-up -------------------------------------------------------------------------------------------------

    def _init(self) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            raise RuntimeError("Shadow Ops needs OpenGL 3.3 (no graphics context could be made)")
        surface = QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()
        if not ctx.makeCurrent(surface):
            raise RuntimeError("Shadow Ops could not use its graphics context")
        self.ctx, self.surface = ctx, surface
        f = self.f = QOpenGLExtraFunctions(ctx)
        f.initializeOpenGLFunctions()
        self.gpu = str(f.glGetString(0x1F01) or "")
        self.p_world = self._program(_WORLD_VS, _WORLD_FS)
        self.p_sprite = self._program(_SPRITE_VS, _SPRITE_FS)
        self.p_glow = self._program(_GLOW_VS, _GLOW_FS)
        self.p_sky = self._program(_FULL_VS, _SKY_FS)
        self.p_vig = self._program(_FULL_VS, _VIGNETTE_FS)
        self.loc = {id(p): {} for p in (self.p_world, self.p_sprite, self.p_glow, self.p_sky, self.p_vig)}
        rgba = self.atlas.rgba
        n = len(rgba)
        levels = _mips(rgba)
        tex = QOpenGLTexture(QOpenGLTexture.Target.Target2DArray)
        tex.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        tex.setSize(rgba.shape[2], rgba.shape[1])  # the atlas is drawn at TS and doubled
        tex.setLayers(n)
        tex.setMipLevels(len(levels))
        tex.allocateStorage(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8)
        for level, images in enumerate(levels):  # each level in one call (one call per layer crashes PySide6)
            tex.setData(level, 0, n, QOpenGLTexture.CubeMapFace.CubeMapPositiveX, QOpenGLTexture.PixelFormat.RGBA,
                        QOpenGLTexture.PixelType.UInt8, np.ascontiguousarray(images).tobytes())
        tex.setMinMagFilters(QOpenGLTexture.Filter.LinearMipMapLinear, QOpenGLTexture.Filter.Linear)
        tex.setWrapMode(QOpenGLTexture.WrapMode.Repeat)
        try:
            tex.setMaximumAnisotropy(8.0)
        except Exception:  # noqa: BLE001 - not every driver has it
            pass
        self.tex = tex
        self.static = self._mesh_buffer()
        self.panes = self._mesh_buffer()
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

    def _mesh_buffer(self) -> _Buf:
        f = self.f
        b = _Buf()
        b.vao.create()
        b.vao.bind()
        b.vbo.create()
        b.vbo.bind()
        stride = VERT_FLOATS * 4
        for loc, off, size in ((0, 0, 3), (1, 12, 2), (2, 20, 3), (3, 32, 4)):
            f.glEnableVertexAttribArray(loc)
            self.p_world.setAttributeBuffer(loc, GL_FLOAT, off, size, stride)
        b.vao.release()
        b.vbo.release()
        return b

    def _program(self, vs: str, fs: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs) and ok
        if not (ok and prog.link()):
            raise RuntimeError(f"Shadow Ops shader did not compile: {prog.log()}")
        return prog

    def _u(self, prog: QOpenGLShaderProgram, name: str) -> int:
        cache = self.loc[id(prog)]
        loc = cache.get(name)
        if loc is None:
            loc = cache[name] = prog.uniformLocation(name)
        return loc

    def warm_up(self) -> None:
        """Make the context, shaders, textures and the map's geometry now (behind the loading screen)."""
        if self.ctx is None:
            self._init()
        elif not self.ctx.makeCurrent(self.surface):
            return
        self._upload_meshes()
        self.ctx.doneCurrent()

    def refresh(self) -> None:
        """The map changed (a door opened, planks came off)."""
        self._static_dirty = True
        self._panes_dirty = True

    def refresh_panes(self) -> None:
        self._panes_dirty = True

    def close(self) -> None:
        """Give everything back to the graphics card."""
        if self.ctx is None:
            return
        try:
            if self.ctx.makeCurrent(self.surface):
                for b in (self.static, self.panes, self.glows):
                    b.vao.destroy()
                    b.vbo.destroy()
                self.empty.destroy()
                self.tex.destroy()
                for t, _img in self._sprite_tex.values():
                    t.destroy()
                self.fbo = self.resolve = None
                self.ctx.doneCurrent()
        except Exception:  # noqa: BLE001 - closing must never fail
            log.debug("Shadow Ops renderer close", exc_info=True)
        self._sprite_tex.clear()
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

    def _fill(self, b: _Buf, data: np.ndarray, rows: int) -> None:
        raw = np.ascontiguousarray(data, F32).tobytes()
        b.vbo.bind()
        b.vbo.allocate(raw, len(raw))
        b.vbo.release()
        b.count = rows

    def _upload_meshes(self) -> None:
        if self._static_dirty:
            data = build_static(self.m)
            self._fill(self.static, data, len(data))
            self._static_dirty = False
        if self._panes_dirty:
            data = build_panes(self.m)
            self._fill(self.panes, data, len(data))
            self._panes_dirty = False

    def preload(self, images: list) -> None:
        """Upload billboard images now (behind the loading screen) so the first sight of one does not stutter."""
        if self.ctx is None:
            self._init()
        elif not self.ctx.makeCurrent(self.surface):
            return
        for img in images:
            self._sprite_texture(img)
        self.ctx.doneCurrent()

    def _sprite_texture(self, img: np.ndarray) -> QOpenGLTexture:
        entry = self._sprite_tex.get(id(img))
        if entry is not None and entry[1] is img:
            return entry[0]
        h, w = img.shape[:2]
        a = img[..., 3:4].astype(F32) / 255
        pre = np.concatenate([img[..., :3].astype(F32) * a, a * 255], -1)
        pre = np.clip(pre + 0.5, 0, 255).astype(np.uint8)
        t = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        t.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        t.setSize(w, h)
        t.setMipLevels(max(int(math.log2(max(w, h))) + 1, 1))
        t.allocateStorage(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8)
        t.setData(0, 0, 1, QOpenGLTexture.CubeMapFace.CubeMapPositiveX, QOpenGLTexture.PixelFormat.RGBA,
                  QOpenGLTexture.PixelType.UInt8, np.ascontiguousarray(pre).tobytes())
        t.generateMipMaps()
        t.setMinMagFilters(QOpenGLTexture.Filter.LinearMipMapLinear, QOpenGLTexture.Filter.Linear)
        t.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        self._sprite_tex[id(img)] = (t, img)
        return t

    # --- the camera ---------------------------------------------------------------------------------------------

    def _camera(self, x: float, y: float, eye: float, angle: float, look: float) -> None:
        p = math.atan(look)
        ca, sa, cp, sp = math.cos(angle), math.sin(angle), math.cos(p), math.sin(p)
        self.cam = np.array([x, eye, y])
        self.F = np.array([cp * ca, sp, cp * sa])
        self.R = np.array([-sa, 0.0, ca])
        self.U = np.cross(self.R, self.F)

    def _tans(self) -> tuple[float, float]:
        tx = math.tan(self.fov / 2)
        return tx, tx * self.view_h / max(self.view_w, 1)

    def _vp(self) -> np.ndarray:
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = self.R, self.U, -self.F
        view[:3, 3] = -(view[:3, :3] @ self.cam)
        tx, ty = self._tans()
        near, far = 0.04, 120.0
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
        """Widget (sx, sy) and depth of a map point (x, y) at height z; None behind the camera."""
        d = np.array([x, z, y]) - self.cam
        zc = float(d @ self.F)
        if zc < 0.05:
            return None
        k = self.proj / zc
        return self.view_w / 2 + float(d @ self.R) * k, self.view_h / 2 - float(d @ self.U) * k, zc

    def visible(self, x: float, y: float, z: float) -> tuple[float, float, float] | None:
        """``project`` when the point is on screen and no wall stands between it and the eye."""
        from .world import line_of_sight

        pt = self.project(x, y, z)
        if pt is None or not (0 <= pt[0] < self.view_w and 0 <= pt[1] < self.view_h):
            return None
        if not line_of_sight(self.m, float(self.cam[0]), float(self.cam[2]), x, y, self.m.solid):
            return None
        return pt

    # --- the frame ----------------------------------------------------------------------------------------------

    def _lights(self, dynamic: list) -> list:
        """The lamps and moving lights nearest the eye (at most MAX_LIGHTS)."""
        cam = self.cam
        out = []
        for X, Y, Z, col, rad, zone in self.lamps:
            out.append((X, Y, Z, col, rad, zone))
        for x, y, z, col, rad, strength in dynamic:
            k = max(min(strength, 1.5), 0.0)
            out.append((x, z, y, (col[0] / 255 * k, col[1] / 255 * k, col[2] / 255 * k), rad, -1.0))
        out.sort(key=lambda L: (L[0] - cam[0]) ** 2 + (L[2] - cam[2]) ** 2 - L[4] * L[4])
        return out[:MAX_LIGHTS]

    def _common(self, prog: QOpenGLShaderProgram, vp: np.ndarray, lights: list, torch: float) -> None:
        f, sc = self.f, self.scene
        prog.bind()
        prog.setUniformValue(self._u(prog, "u_vp"), QMatrix4x4(*[float(v) for v in vp.reshape(-1)]))
        f.glUniform3f(self._u(prog, "u_cam"), *[float(v) for v in self.cam])
        f.glUniform3f(self._u(prog, "u_fwd"), *[float(v) for v in self.F])
        f.glUniform3f(self._u(prog, "u_amb_in"), *sc["ambient_in"])
        f.glUniform3f(self._u(prog, "u_amb_out"), *sc["ambient_out"])
        f.glUniform4f(self._u(prog, "u_sun"), *sc["sun"])
        f.glUniform3f(self._u(prog, "u_sun_col"), *sc["sun_color"])
        f.glUniform3f(self._u(prog, "u_fog"), *[c / 255 for c in sc["fog"]])
        f.glUniform1f(self._u(prog, "u_fogd"), float(sc["fog_dist"]))
        f.glUniform1f(self._u(prog, "u_torch"), float(torch))
        f.glUniform1i(self._u(prog, "u_nl"), len(lights))
        for i, (X, Y, Z, col, rad, zone) in enumerate(lights):
            f.glUniform4f(self._u(prog, f"u_lpos[{i}]"), float(X), float(Y), float(Z), float(rad))
            f.glUniform4f(self._u(prog, f"u_lcol[{i}]"), float(col[0]), float(col[1]), float(col[2]), float(zone))

    def render(self, x: float, y: float, eye: float, angle: float, look: float, fov: float, w: int, h: int,
               sprites: list, glows: list, lights: list, torch: float = 0.0, t: float = 0.0,
               dpr: float = 1.0) -> QImage:
        """One frame for a widget of ``w`` x ``h`` (drawn at ``dpr`` times that). ``glows``: (x, y, z, (r, g, b),
        radius, strength) additive halos; ``lights``: the same tuples lighting the surfaces around them."""
        import time

        t0 = time.perf_counter()
        if self.ctx is None:
            self._init()
        elif not self.ctx.makeCurrent(self.surface):
            raise RuntimeError("Shadow Ops lost its graphics context")
        f = self.f
        pw, ph = max(int(w * dpr), 16), max(int(h * dpr), 16)
        self._size(pw, ph)
        self._upload_meshes()
        self.fov, self.view_w, self.view_h = fov, w, h
        self._camera(x, y, eye, angle, look)
        vp = self._vp()
        L = self._lights(lights)
        self.fbo.bind()
        f.glViewport(0, 0, pw, ph)
        f.glDisable(GL_BLEND)
        f.glDisable(GL_CULL_FACE)
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LESS)
        f.glDepthMask(True)
        fog = self.scene["fog"]
        f.glClearColor(fog[0] / 255, fog[1] / 255, fog[2] / 255, 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # the map
        prog = self.p_world
        self._common(prog, vp, L, torch)
        self.tex.bind(0)
        prog.setUniformValue1i(self._u(prog, "u_tex"), 0)
        for b in (self.static, self.panes):
            if b.count:
                b.vao.bind()
                f.glDrawArrays(GL_TRIANGLES, 0, b.count)
                b.vao.release()
        # billboards
        if sprites:
            prog = self.p_sprite
            self._common(prog, vp, L, torch)
            prog.setUniformValue1i(self._u(prog, "u_img"), 0)
            f.glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE)
            self.empty.bind()
            rx, rz = float(self.R[0]), float(self.R[2])
            nr = math.hypot(rx, rz) or 1.0
            f.glUniform3f(self._u(prog, "u_right"), rx / nr, 0.0, rz / nr)
            ub, us, ut, ubr, uf = (self._u(prog, k) for k in ("u_base", "u_size", "u_tint", "u_bright", "u_flip"))
            usky, uzone = self._u(prog, "u_sky"), self._u(prog, "u_zone")
            m = self.m
            for s in sprites:
                cx, cy = int(s.x), int(s.y)
                if not m.inside(cx, cy):
                    continue
                self._sprite_texture(s.img).bind(0)
                f.glUniform3f(ub, float(s.x), float(s.z), float(s.y))
                f.glUniform2f(us, float(s.w), float(s.h))
                if s.tint is not None:
                    f.glUniform4f(ut, s.tint[0] / 255, s.tint[1] / 255, s.tint[2] / 255, float(s.tint[3]))
                else:
                    f.glUniform4f(ut, 0.0, 0.0, 0.0, 0.0)
                f.glUniform1f(ubr, 1.0 if s.bright else 0.0)
                f.glUniform1f(uf, 1.0 if s.flip else 0.0)
                f.glUniform1f(usky, 1.0 if m.ceil[cy, cx] < 0 else 0.0)
                f.glUniform1f(uzone, float(m.zone[cy, cx]))
                f.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            self.empty.release()
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
        f.glUniform1f(self._u(prog, "u_day"), 1.0 if self.scene["kind"] == "day" else 0.0)
        f.glUniform1f(self._u(prog, "u_time"), float(t))
        f.glUniform3f(self._u(prog, "u_fog"), *[float(c) for c in fog])
        sun = self.scene["sun"]
        f.glUniform3f(self._u(prog, "u_sundir"), -sun[0], -sun[1], -sun[2])
        self.empty.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, 3)
        # glows (the lamps' own halos too)
        halos = [(X, Z, Y, (c[0] * 255, c[1] * 255, c[2] * 255), 0.32, 0.55) for X, Y, Z, c, _r, _z in self.lamps]
        rows = []
        for gx, gy, gz, col, rad, strength in list(glows) + halos:
            rows.append((gx, gz, gy, rad, col[0] / 255, col[1] / 255, col[2] / 255, max(min(strength, 2.0), 0.0)))
        if rows:
            prog = self.p_glow
            prog.bind()
            prog.setUniformValue(self._u(prog, "u_vp"), QMatrix4x4(*[float(v) for v in vp.reshape(-1)]))
            f.glUniform3f(self._u(prog, "u_right"), *[float(v) for v in self.R])
            f.glUniform3f(self._u(prog, "u_up"), *[float(v) for v in self.U])
            f.glUniform3f(self._u(prog, "u_cam"), *[float(v) for v in self.cam])
            self._fill(self.glows, np.array(rows, F32), len(rows))
            f.glEnable(GL_BLEND)
            f.glBlendFunc(GL_ONE, GL_ONE)
            self.glows.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(rows))
            self.glows.vao.release()
        # darker corners
        f.glDisable(GL_DEPTH_TEST)
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_DST_COLOR, GL_ZERO)
        prog = self.p_vig
        prog.bind()
        night = self.scene["kind"] != "day"
        f.glUniform1f(self._u(prog, "u_amount"), 1.25 if night else 0.8)
        f.glUniform3f(self._u(prog, "u_tint"), 1.0, 1.0, 1.0)
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

    def eye_glows(self, sprites: list) -> list:
        """Halos for the glowing eyes of the billboards that have them, a touch in front of the face."""
        out = []
        rx, rz = float(self.R[0]), float(self.R[2])
        nr = math.hypot(rx, rz) or 1.0
        rx, rz = rx / nr, rz / nr
        for s in sprites:
            if not s.eyes:
                continue
            dx, dy = float(self.cam[0]) - s.x, float(self.cam[2]) - s.y
            dist = math.hypot(dx, dy) or 1.0
            if dist > 22:
                continue
            for u, v in s.eyes:
                k = (1 - u) if s.flip else u
                ox = rx * (k - 0.5) * s.w + dx / dist * 0.06
                oy = rz * (k - 0.5) * s.w + dy / dist * 0.06
                out.append((s.x + ox, s.y + oy, s.z + (1 - v) * s.h, s.eye_color, 0.05, 1.6))
        return out


__all__ = ["EYE", "Renderer", "Sprite", "WALL_H"]
