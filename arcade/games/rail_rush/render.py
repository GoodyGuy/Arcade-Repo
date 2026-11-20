"""Rail Rush's renderer: the yard drawn by the graphics card (OpenGL 3.3 through Qt's own classes, no other package)
at the window's own resolution, into an image the game widget paints.

Everything is an *instance* of a few unit meshes (box, cylinder, wedge, pyramid, cone, ball, flat quad): one row of
24 floats gives its place and size (a 3 x 4 matrix), two colours, a pattern and its numbers. The patterns (hazard
stripes, windows, corrugated steel, planks, bricks, gravel, rock, water ...) are worked out per pixel by the shader,
so they stay crisp at any window size and need no texture files. Scenery is uploaded once per stretch of track and
drawn with an offset; trains, coins and the runners are sent every frame. The picture is drawn upside down on
purpose so it can be read back without a flip."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from PySide6.QtGui import (QImage, QMatrix4x4, QOffscreenSurface, QOpenGLContext, QOpenGLExtraFunctions,
                           QSurfaceFormat)
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
                              QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject)

log = logging.getLogger(__name__)
F32 = np.float32
INST = 24  # floats per instance: matrix rows (3 x 4), colour + alpha, second colour + pattern, 4 pattern numbers
BOX, CYL, WEDGE, PYR, CONE, BALL, QUAD = range(7)
MESHES = 7
# patterns (the fourth number of the second colour)
(FLAT, HAZARD, WINDOWS, RIBS, PLANKS, BRICKS, PANELS, GRAVEL, STRATA, CHEVRONS, DISC, WATER, GRASS, COIN, FADE,
 SOFTRECT, RING) = range(17)
# OpenGL names (Qt gives the functions, not the constants)
GL_TRIANGLES, GL_FLOAT, GL_DEPTH_TEST, GL_CULL_FACE, GL_BLEND = 0x0004, 0x1406, 0x0B71, 0x0B44, 0x0BE2
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = 0x4000, 0x0100, 0x0302, 0x0303
GL_LEQUAL, GL_RGB8 = 0x0203, 0x8051


class Env:
    """The light of the moment: sky colours, the sun (or moon), fog, and how much it is night (0..1)."""

    def __init__(self) -> None:
        self.zenith = (0.20, 0.50, 0.92)
        self.horizon = (0.74, 0.87, 0.96)
        self.sun = (0.42, 0.78, -0.46)
        self.sun_col = (0.78, 0.74, 0.66)
        self.amb_up = (0.52, 0.57, 0.66)
        self.amb_dn = (0.36, 0.34, 0.33)
        self.cloud = (1.0, 1.0, 1.0)
        self.hills = (0.45, 0.60, 0.72)
        self.night = 0.0
        self.fog = (110.0, 255.0)
        self.time = 0.0
        self.travel = 0.0  # metres run (far hills and clouds drift)
        self.tunnels: list[tuple[float, float]] = []  # (z0, z1) relative to the frame's origin


# --- the unit meshes -------------------------------------------------------------------------------------------------

def _unit_meshes() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Every mesh as triangles of (position, normal), all inside -1..1, and each one's (first vertex, count)."""
    out: list[list] = []
    ranges: list[tuple[int, int]] = []

    def tri(a, b, c, na, nb=None, nc=None) -> None:  # noqa: ANN001
        out.append([*a, *na])
        out.append([*b, *(nb or na)])
        out.append([*c, *(nc or na)])

    def quad(a, b, c, d, n) -> None:  # noqa: ANN001
        tri(a, b, c, n)
        tri(a, c, d, n)

    def close() -> None:
        first = ranges[-1][0] + ranges[-1][1] if ranges else 0
        ranges.append((first, len(out) - first))

    # box
    for axis in range(3):
        for sign in (-1.0, 1.0):
            n = [0.0, 0.0, 0.0]
            n[axis] = sign
            u, v = (axis + 1) % 3, (axis + 2) % 3
            pts = []
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis], p[u], p[v] = sign, su, sv
                pts.append(p)
            quad(*pts, n)
    close()
    # cylinder along y
    sides = 16
    ring = [(math.cos(i * math.tau / sides), math.sin(i * math.tau / sides)) for i in range(sides + 1)]
    for (c0, s0), (c1, s1) in zip(ring, ring[1:]):
        n0, n1 = (c0, 0.0, s0), (c1, 0.0, s1)
        tri((c0, -1, s0), (c1, -1, s1), (c1, 1, s1), n0, n1, n1)
        tri((c0, -1, s0), (c1, 1, s1), (c0, 1, s0), n0, n1, n0)
        tri((0, 1, 0), (c0, 1, s0), (c1, 1, s1), (0, 1, 0))
        tri((0, -1, 0), (c0, -1, s0), (c1, -1, s1), (0, -1, 0))
    close()
    # wedge: a slope rising from z = -1 to z = +1
    a, b, c, d, e, f = (-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1), (-1, 1, 1), (1, 1, 1)
    k = math.sqrt(0.5)
    quad(a, b, f, e, (0, k, -k))
    quad(d, c, f, e, (0, 0, 1))
    quad(a, b, c, d, (0, -1, 0))
    tri(a, d, e, (-1, 0, 0))
    tri(b, c, f, (1, 0, 0))
    close()
    # pyramid on the square base
    k = 1.0 / math.sqrt(5.0)
    top = (0, 1, 0)
    quad((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1), (0, -1, 0))
    tri((1, -1, -1), (1, -1, 1), top, (2 * k, k, 0))
    tri((-1, -1, -1), (-1, -1, 1), top, (-2 * k, k, 0))
    tri((-1, -1, 1), (1, -1, 1), top, (0, k, 2 * k))
    tri((-1, -1, -1), (1, -1, -1), top, (0, k, -2 * k))
    close()
    # cone
    sides = 10
    ring = [(math.cos(i * math.tau / sides), math.sin(i * math.tau / sides)) for i in range(sides + 1)]
    for (c0, s0), (c1, s1) in zip(ring, ring[1:]):
        n0, n1 = (2 * k * c0, k, 2 * k * s0), (2 * k * c1, k, 2 * k * s1)
        nm = (k * (c0 + c1), k, k * (s0 + s1))
        tri((c0, -1, s0), (c1, -1, s1), top, n0, n1, nm)
        tri((0, -1, 0), (c0, -1, s0), (c1, -1, s1), (0, -1, 0))
    close()
    # ball
    stacks, slices = 8, 14

    def on_ball(i: int, j: int) -> tuple:
        th, ph = math.pi * i / stacks, math.tau * j / slices
        return (math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph))

    for i in range(stacks):
        for j in range(slices):
            p00, p01, p10, p11 = on_ball(i, j), on_ball(i, j + 1), on_ball(i + 1, j), on_ball(i + 1, j + 1)
            if i > 0:
                tri(p00, p01, p11, p00, p01, p11)
            if i < stacks - 1:
                tri(p00, p11, p10, p00, p11, p10)
    close()
    # flat quad (in the x-z plane, facing up)
    quad((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1), (0, 1, 0))
    close()
    return np.array(out, F32), ranges


# --- shaders ---------------------------------------------------------------------------------------------------------

_NOISE = """
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
float aa(float edge, float x) { float w = fwidth(x) * 0.75 + 1e-6; return smoothstep(edge - w, edge + w, x); }
float tri(float x) { return abs(fract(x) - 0.5) * 2.0; }
"""

_VS = """#version 330 core
layout(location = 0) in vec3 a_pos; layout(location = 1) in vec3 a_nrm;
layout(location = 2) in vec4 a_m0; layout(location = 3) in vec4 a_m1; layout(location = 4) in vec4 a_m2;
layout(location = 5) in vec4 a_col; layout(location = 6) in vec4 a_col2; layout(location = 7) in vec4 a_par;
uniform mat4 u_vp; uniform vec3 u_off;
out vec3 v_unit; out vec3 v_ln; out vec3 v_n; out vec3 v_w;
flat out vec3 v_scale; flat out vec4 v_col; flat out vec4 v_col2; flat out vec4 v_par;
void main() {
    vec3 cx = vec3(a_m0.x, a_m1.x, a_m2.x), cy = vec3(a_m0.y, a_m1.y, a_m2.y), cz = vec3(a_m0.z, a_m1.z, a_m2.z);
    vec3 p = cx * a_pos.x + cy * a_pos.y + cz * a_pos.z + vec3(a_m0.w, a_m1.w, a_m2.w) + u_off;
    vec3 sc = vec3(length(cx), length(cy), length(cz));
    vec3 ln = a_nrm / max(sc * sc, vec3(1e-8));
    v_n = normalize(cx * ln.x + cy * ln.y + cz * ln.z);
    v_unit = a_pos; v_ln = a_nrm; v_scale = sc; v_w = p;
    v_col = a_col; v_col2 = a_col2; v_par = a_par;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

_FS = "#version 330 core\n" + _NOISE + """
in vec3 v_unit; in vec3 v_ln; in vec3 v_n; in vec3 v_w;
flat in vec3 v_scale; flat in vec4 v_col; flat in vec4 v_col2; flat in vec4 v_par;
uniform vec3 u_cam; uniform vec3 u_sun; uniform vec3 u_suncol; uniform vec3 u_ambup; uniform vec3 u_ambdn;
uniform vec3 u_fogcol; uniform vec2 u_fog; uniform vec4 u_tun0; uniform vec4 u_tun1;
uniform float u_time; uniform float u_night; uniform float u_zshift;
out vec4 o;
float inside(float z, vec4 t) { return step(t.x, z) * step(z, t.y); }
void main() {
    vec3 an = abs(v_ln);
    vec3 m = v_unit * v_scale;                      // metres in the instance's own frame
    vec2 uv, un; float side = 1.0;                  // side: 1 on an upright face, 0 on a level one
    if (an.y > an.x && an.y > an.z) { uv = m.xz; un = v_unit.xz; side = 0.0; }
    else if (an.x > an.z) { uv = m.zy; un = v_unit.zy; }
    else { uv = m.xy; un = v_unit.xy; }
    vec2 wxz = vec2(v_w.x, v_w.z + u_zshift);       // the ground's own coordinates (joined across stretches)
    vec3 c = v_col.rgb; vec3 c2 = v_col2.rgb; float alpha = v_col.a; float emis = v_par.z;
    int pat = int(v_col2.a + 0.5);
    float px = v_par.x, py = v_par.y;
    if (pat == 1) {                                 // hazard stripes
        c = mix(c, c2, aa(0.5, tri((uv.x + uv.y) / px)));
    } else if (pat == 2) {                          // rows of windows, some lit at night
        if (side > 0.5) {
            vec2 g = uv / vec2(px, py);
            vec2 f = abs(fract(g) - 0.5) * 2.0;
            float win = (1.0 - aa(0.62, f.x)) * (1.0 - aa(0.56, f.y));
            float lit = step(0.55, hash(floor(g) + v_par.w)) * u_night;
            c = mix(c, mix(c2, vec3(1.0, 0.86, 0.52), lit), win);
            emis = max(emis, win * lit * 0.9);
        }
    } else if (pat == 3) {                          // corrugated steel with a band
        if (side > 0.5) {
            float fw = clamp(fwidth(uv.x / px) * 1.6, 0.0, 1.0);
            c *= mix(0.82 + 0.18 * smoothstep(0.15, 0.85, tri(uv.x / px)), 0.91, fw);
            if (py < 5.0) c = mix(c, c2, aa(py - 0.13, un.y) * (1.0 - aa(py + 0.13, un.y)));
        }
    } else if (pat == 4) {                          // planks
        float q = (v_par.w > 0.5 ? uv.x : uv.y) / px;
        float fw = clamp(fwidth(q) * 2.0, 0.0, 1.0);
        c *= mix(0.86 + 0.28 * hash(vec2(floor(q), 7.0)), 1.0, fw);
        c = mix(c, c * 0.5, aa(0.9, tri(q)) * (1.0 - fw));
    } else if (pat == 5) {                          // bricks
        if (side > 0.5) {
            vec2 g = uv / vec2(px, px * 0.45);
            g.x += 0.5 * mod(floor(g.y), 2.0);
            float fw = clamp(fwidth(uv.y / (px * 0.45)) * 2.0, 0.0, 1.0);
            vec2 f = abs(fract(g) - 0.5) * 2.0;
            float mortar = max(aa(0.93, f.x), aa(0.84, f.y));
            c *= mix(0.84 + 0.32 * hash(floor(g)), 1.0, fw);
            c = mix(c, c2, mix(mortar, 0.18, fw));
        }
    } else if (pat == 6) {                          // wagon panels: seams and a band
        if (side > 0.5) {
            float q = uv.x / px;
            float fw = clamp(fwidth(q) * 2.0, 0.0, 1.0);
            c = mix(c, c * 0.68, aa(0.93, tri(q)) * (1.0 - fw));
            if (py < 5.0) c = mix(c, c2, aa(py - 0.09, un.y) * (1.0 - aa(py + 0.09, un.y)));
        }
    } else if (pat == 7) {                          // gravel
        vec2 q = side > 0.5 ? uv : wxz;
        float fw = 1.0 - clamp(fwidth(q.y) * 5.0, 0.0, 1.0);
        c *= 1.0 + (vnoise(q * 7.0) * 0.6 + vnoise(q * 19.0) * 0.4 - 0.5) * 0.6 * fw;
    } else if (pat == 8) {                          // rock in layers
        float n = vnoise(vec2(uv.x * 0.11 + v_par.w, uv.y * 0.8 + v_par.w));
        c *= 0.78 + 0.44 * floor(n * 5.0) / 5.0;
        c = mix(c, c2, 0.4 * step(0.58, vnoise(vec2(uv.x * 0.05 + v_par.w, uv.y * 0.27))));
    } else if (pat == 9) {                          // chevrons (px of them, py: +1 up, -1 down)
        c = mix(c, c2, aa(0.5, tri((un.y * py + abs(un.x) * 0.9) * px)));
    } else if (pat == 10) {                         // a soft disc: shadows and glows
        alpha *= 1.0 - smoothstep(px, 1.0, length(un));
    } else if (pat == 11) {                         // water
        float r1 = vnoise(wxz * 0.3 + vec2(u_time * 0.25, u_time * 0.15));
        float r2 = vnoise(wxz * vec2(0.8, 0.22) - vec2(u_time * 0.35, 0.0));
        c = mix(c, c2, r1 * 0.7);
        c += vec3(0.9) * smoothstep(0.6, 0.66, r1 * r2 * 2.0) * 0.55;
    } else if (pat == 12) {                         // ground in two tones
        float n = vnoise(wxz * 0.11) * 0.6 + vnoise(wxz * 0.53) * 0.4;
        c = mix(c, c2, smoothstep(0.38, 0.62, n));
        float fw = 1.0 - clamp(fwidth(wxz.y) * 3.0, 0.0, 1.0);
        c *= 1.0 + (vnoise(wxz * 5.0) - 0.5) * 0.22 * fw;
    } else if (pat == 13) {                         // the face of a coin
        if (side < 0.5) {
            float r = length(un);
            c = mix(c, c2, aa(0.66, r) * (1.0 - aa(0.84, r)));
            c = mix(c, c2, 1.0 - aa(0.42, abs(un.x) + abs(un.y)));
        }
    } else if (pat == 14) {                         // from the second colour at the bottom to the first on top
        c = mix(c2, c, clamp(v_unit.y * 0.5 + 0.5, 0.0, 1.0));
    } else if (pat == 15) {                         // a soft rectangle: the shadow of a wagon
        alpha *= (1.0 - smoothstep(px, 1.0, abs(un.x))) * (1.0 - smoothstep(py, 1.0, abs(un.y)));
    } else if (pat == 16) {                         // a ring
        float r = length(un);
        alpha *= aa(px, r) * (1.0 - aa(0.97, r));
    }
    vec3 N = normalize(v_n);
    vec3 light = mix(u_ambdn, u_ambup, N.y * 0.5 + 0.5) + u_suncol * max(dot(N, u_sun), 0.0);
    float tun = max(inside(v_w.z, u_tun0), inside(v_w.z, u_tun1)) * step(abs(v_w.x), 5.25) * step(v_w.y, 7.45)
        * step(-0.6, v_w.y);
    float d = length(v_w - u_cam);
    float fog = smoothstep(u_fog.x, u_fog.y, d);
    vec3 fogcol = u_fogcol;
    if (tun > 0.5) {                                // inside a tunnel: no sun, lamps along the roof, dark far away
        float tz = v_w.z - (inside(v_w.z, u_tun0) > 0.5 ? u_tun0.x : u_tun1.x);
        float lamp = pow(max(cos(tz * 0.5236), 0.0), 6.0);
        light = vec3(0.33, 0.31, 0.32) + vec3(1.0, 0.8, 0.52) * (0.2 + 0.5 * lamp);
        fog = smoothstep(30.0, 150.0, d) * 0.94;
        fogcol = vec3(0.03, 0.028, 0.035);
    }
    vec3 rgb = c * mix(light, vec3(1.0), clamp(emis, 0.0, 1.0)) + c * max(emis - 1.0, 0.0);
    o = vec4(mix(rgb, fogcol, fog), alpha);
}
"""

_SKY_VS = """#version 330 core
out vec2 v_ndc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
    v_ndc = p; gl_Position = vec4(p, 1.0, 1.0);
}
"""

_SKY_FS = "#version 330 core\n" + _NOISE + """
in vec2 v_ndc; out vec4 o;
uniform vec3 u_r; uniform vec3 u_u; uniform vec3 u_f; uniform vec3 u_zenith; uniform vec3 u_horizon;
uniform vec3 u_sun; uniform vec3 u_suncol; uniform vec3 u_cloud; uniform vec3 u_hills;
uniform float u_time; uniform float u_night; uniform float u_travel;
void main() {
    vec3 d = normalize(u_f + u_r * v_ndc.x - u_u * v_ndc.y);
    vec3 c = u_horizon;
    if (d.y > 0.0) {
        c = mix(u_horizon, u_zenith, pow(d.y, 0.6));
        vec2 sq = d.xz / (d.y + 0.25);
        if (u_night > 0.05) {                       // stars
            vec2 cell = floor(sq * 70.0);
            float h = hash(cell);
            c += vec3(0.9, 0.95, 1.0) * step(0.985, h) * u_night * (0.5 + 0.5 * sin(u_time * 2.0 + h * 600.0))
                * smoothstep(0.03, 0.25, d.y);
        }
        float sd = max(dot(d, u_sun), 0.0);         // the sun (or the moon) and its glow
        c += u_suncol * (smoothstep(0.9993, 0.9996, sd) * 1.6 + pow(sd, 40.0) * 0.3 + pow(sd, 6.0) * 0.12);
        vec2 q = d.xz / (d.y + 0.1) * 0.9 + vec2(u_time * 0.006, u_travel * 0.0004);
        float n = vnoise(q) * 0.55 + vnoise(q * 2.3 + 7.0) * 0.3 + vnoise(q * 5.1 + 3.0) * 0.15;
        c = mix(c, u_cloud, smoothstep(0.5, 0.74, n) * smoothstep(0.03, 0.22, d.y) * 0.92);
        float az = atan(d.x, d.z);                  // two ranges of far hills
        float far = 0.05 + 0.07 * vnoise(vec2(az * 2.6 + 11.0 + u_travel * 0.00006, 0.5))
            + 0.025 * vnoise(vec2(az * 9.0 + 3.0, 4.0));
        c = mix(c, mix(u_horizon, u_hills, 0.45), 1.0 - aa(far, d.y));
        float near = 0.012 + 0.045 * vnoise(vec2(az * 5.0 + 40.0 + u_travel * 0.0002, 2.0))
            + 0.012 * vnoise(vec2(az * 19.0, 9.0));
        c = mix(c, mix(u_horizon, u_hills, 0.8), 1.0 - aa(near, d.y));
    }
    o = vec4(c, 1.0);
}
"""


class _Part:
    """The instances of one mesh in one buffer on the card."""

    __slots__ = ("count", "vao", "vbo")

    def __init__(self) -> None:
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.count = 0

    def destroy(self) -> None:
        self.vao.destroy()
        self.vbo.destroy()


class Renderer:
    """Draws instance lists through a camera into a QImage. Everything OpenGL is made the first time it draws."""

    def __init__(self) -> None:
        self.ctx: QOpenGLContext | None = None
        self.surface: QOffscreenSurface | None = None
        self.f: QOpenGLExtraFunctions | None = None
        self.fbo: QOpenGLFramebufferObject | None = None
        self.resolve: QOpenGLFramebufferObject | None = None
        self.samples = 4
        self.gpu = ""
        self.chunks: dict = {}  # key -> ({mesh: _Part} solid, {mesh: _Part} see-through)
        self._pending: dict = {}  # key -> (solid arrays, see-through arrays) waiting for the next frame
        self._dead: list = []
        self._dyn: tuple[dict, dict] = ({}, {})
        self.stats = {"ms": 0.0, "instances": 0}

    # --- set-up -----------------------------------------------------------------------------------------------------

    def _init(self) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            raise RuntimeError("Rail Rush needs OpenGL 3.3 (no graphics context could be made)")
        surface = QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()
        if not ctx.makeCurrent(surface):
            raise RuntimeError("Rail Rush could not use its graphics context")
        self.ctx, self.surface = ctx, surface
        f = self.f = QOpenGLExtraFunctions(ctx)
        f.initializeOpenGLFunctions()
        self.gpu = str(f.glGetString(0x1F01) or "")
        self.p_main = self._program(_VS, _FS)
        self.p_sky = self._program(_SKY_VS, _SKY_FS)
        self.loc = {id(self.p_main): {}, id(self.p_sky): {}}
        verts, self.ranges = _unit_meshes()
        self.mesh_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.mesh_vbo.create()
        self.mesh_vbo.bind()
        raw = verts.tobytes()
        self.mesh_vbo.allocate(raw, len(raw))
        self.mesh_vbo.release()
        self.sky_vao = QOpenGLVertexArrayObject()
        self.sky_vao.create()

    def warm_up(self) -> None:
        """Make the graphics context and the shaders now rather than on the first frame."""
        if self.ctx is None:
            self._init()
            self.ctx.doneCurrent()

    def _program(self, vs: str, fs: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs) and ok
        if not (ok and prog.link()):
            raise RuntimeError(f"Rail Rush shader did not compile: {prog.log()}")
        return prog

    def _u(self, prog: QOpenGLShaderProgram, name: str) -> int:
        cache = self.loc[id(prog)]
        loc = cache.get(name)
        if loc is None:
            loc = cache[name] = prog.uniformLocation(name)
        return loc

    def _part(self) -> _Part:
        f, prog = self.f, self.p_main
        part = _Part()
        part.vao.create()
        part.vao.bind()
        self.mesh_vbo.bind()
        for loc, offset in ((0, 0), (1, 12)):
            f.glEnableVertexAttribArray(loc)
            prog.setAttributeBuffer(loc, GL_FLOAT, offset, 3, 24)
        part.vbo.create()
        part.vbo.bind()
        for i in range(6):
            f.glEnableVertexAttribArray(2 + i)
            prog.setAttributeBuffer(2 + i, GL_FLOAT, i * 16, 4, INST * 4)
            f.glVertexAttribDivisor(2 + i, 1)
        part.vao.release()
        part.vbo.release()
        return part

    def _fill(self, parts: dict, arrays: dict) -> None:
        """Put ``arrays`` ({mesh: (n, 24) float32}) into ``parts`` (buffers made as needed)."""
        for part in parts.values():
            part.count = 0
        for mesh, data in arrays.items():
            n = len(data)
            if not n:
                continue
            part = parts.get(mesh)
            if part is None:
                part = parts[mesh] = self._part()
            raw = np.ascontiguousarray(data, F32).tobytes()
            part.vbo.bind()
            part.vbo.allocate(raw, len(raw))
            part.vbo.release()
            part.count = n

    def _draw(self, parts: dict) -> int:
        f = self.f
        total = 0
        for mesh, part in parts.items():
            if part.count:
                first, count = self.ranges[mesh]
                part.vao.bind()
                f.glDrawArraysInstanced(GL_TRIANGLES, first, count, part.count)
                total += part.count
        return total

    # --- stretches of scenery ---------------------------------------------------------------------------------------

    def put_chunk(self, key, solid: dict, clear: dict | None = None) -> None:  # noqa: ANN001
        """Scenery that stays: uploaded at the start of the next frame."""
        self._pending[key] = (solid, clear or {})

    def drop_chunk(self, key) -> None:  # noqa: ANN001
        self._pending.pop(key, None)
        self._dead.append(key)

    def drop_all(self) -> None:
        self._pending.clear()
        self._dead.extend(self.chunks)

    def _sync(self) -> None:
        for key in self._dead:
            entry = self.chunks.pop(key, None)
            if entry is not None:
                for parts in entry:
                    for part in parts.values():
                        part.destroy()
        self._dead.clear()
        for key, (solid, clear) in self._pending.items():
            entry = self.chunks.get(key)
            if entry is None:
                entry = self.chunks[key] = ({}, {})
            self._fill(entry[0], solid)
            self._fill(entry[1], clear)
        self._pending.clear()

    def close(self) -> None:
        """Give everything back to the graphics card."""
        if self.ctx is None:
            return
        try:
            if self.ctx.makeCurrent(self.surface):
                for entry in list(self.chunks.values()) + [self._dyn]:
                    for parts in entry:
                        for part in parts.values():
                            part.destroy()
                self.mesh_vbo.destroy()
                self.sky_vao.destroy()
                self.fbo = self.resolve = None
                self.ctx.doneCurrent()
        except Exception:  # noqa: BLE001 - closing must never fail
            log.debug("Rail Rush renderer close", exc_info=True)
        self.chunks.clear()
        self._pending.clear()
        self._dead.clear()
        self._dyn = ({}, {})
        self.surface = None
        self.ctx = None

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

    # --- the frame --------------------------------------------------------------------------------------------------

    def render(self, w: int, h: int, eye: tuple, pitch: float, roll: float, fov: float, env: Env,
               offsets: dict, solid: dict, clear: dict) -> QImage:
        """One frame. ``eye``: the camera (it looks along +z, ``pitch`` radians down); ``offsets``: {chunk key: z of
        its start}; ``solid`` / ``clear``: {mesh: instances} of this frame's moving things (see-through ones are
        drawn last, in the order given). All positions are relative to the frame's origin."""
        t0 = time.perf_counter()
        if self.ctx is None:
            self._init()
        elif not self.ctx.makeCurrent(self.surface):
            raise RuntimeError("Rail Rush lost its graphics context")
        f = self.f
        w, h = max(w, 16), max(h, 16)
        self._size(w, h)
        self._sync()
        self._fill(self._dyn[0], solid)
        self._fill(self._dyn[1], clear)
        right, up, fwd = camera_basis(pitch, roll)
        vp = view_projection(right, up, fwd, eye, fov, w / h)
        self.fbo.bind()
        f.glViewport(0, 0, w, h)
        f.glDisable(GL_BLEND)
        f.glDisable(GL_CULL_FACE)
        f.glDisable(GL_DEPTH_TEST)
        f.glDepthMask(True)
        f.glClear(GL_DEPTH_BUFFER_BIT)
        # the sky
        prog = self.p_sky
        prog.bind()
        tan = math.tan(math.radians(fov) / 2)
        for name, v in (("u_r", [c * tan * w / h for c in right]), ("u_u", [c * tan for c in up]), ("u_f", fwd),
                        ("u_zenith", env.zenith), ("u_horizon", env.horizon), ("u_sun", env.sun),
                        ("u_suncol", env.sun_col), ("u_cloud", env.cloud), ("u_hills", env.hills)):
            f.glUniform3f(self._u(prog, name), float(v[0]), float(v[1]), float(v[2]))
        f.glUniform1f(self._u(prog, "u_time"), float(env.time))
        f.glUniform1f(self._u(prog, "u_night"), float(env.night))
        f.glUniform1f(self._u(prog, "u_travel"), float(env.travel))
        self.sky_vao.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.sky_vao.release()
        # everything solid
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LEQUAL)
        prog = self.p_main
        prog.bind()
        prog.setUniformValue(self._u(prog, "u_vp"), QMatrix4x4(*[float(v) for v in vp.reshape(-1)]))
        for name, v in (("u_cam", eye), ("u_sun", env.sun), ("u_suncol", env.sun_col), ("u_ambup", env.amb_up),
                        ("u_ambdn", env.amb_dn), ("u_fogcol", env.horizon)):
            f.glUniform3f(self._u(prog, name), float(v[0]), float(v[1]), float(v[2]))
        f.glUniform2f(self._u(prog, "u_fog"), float(env.fog[0]), float(env.fog[1]))
        tunnels = list(env.tunnels[:2]) + [(1.0, 0.0)] * 2
        for name, (z0, z1) in zip(("u_tun0", "u_tun1"), tunnels):
            f.glUniform4f(self._u(prog, name), float(z0), float(z1), 0.0, 0.0)
        f.glUniform1f(self._u(prog, "u_time"), float(env.time))
        f.glUniform1f(self._u(prog, "u_night"), float(env.night))
        f.glUniform1f(self._u(prog, "u_zshift"), float(env.travel % 4096.0))
        off = self._u(prog, "u_off")
        count = 0
        visible = [(self.chunks[key], z) for key, z in offsets.items() if key in self.chunks]
        for entry, z in visible:
            f.glUniform3f(off, 0.0, 0.0, float(z))
            count += self._draw(entry[0])
        f.glUniform3f(off, 0.0, 0.0, 0.0)
        count += self._draw(self._dyn[0])
        # see-through things last: shadows, glows, dust, the shield
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        f.glDepthMask(False)
        for entry, z in visible:
            f.glUniform3f(off, 0.0, 0.0, float(z))
            count += self._draw(entry[1])
        f.glUniform3f(off, 0.0, 0.0, 0.0)
        count += self._draw(self._dyn[1])
        f.glDepthMask(True)
        f.glDisable(GL_BLEND)
        self.sky_vao.bind()
        self.sky_vao.release()
        self.fbo.release()
        QOpenGLFramebufferObject.blitFramebuffer(self.resolve, self.fbo)
        img = self.resolve.toImage(False)
        self.ctx.doneCurrent()
        self.stats = {"ms": (time.perf_counter() - t0) * 1000, "instances": count}
        return img


def camera_basis(pitch: float, roll: float = 0.0) -> tuple[tuple, tuple, tuple]:
    """(right, up, forward) of a camera looking along +z, ``pitch`` down, with +x on the right of the picture."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    right, up, fwd = (1.0, 0.0, 0.0), (0.0, cp, sp), (0.0, -sp, cp)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        right, up = (tuple(r * c + u * s for r, u in zip(right, up)),
                     tuple(u * c - r * s for r, u in zip(right, up)))
    return right, up, fwd


def view_projection(right: tuple, up: tuple, fwd: tuple, eye: tuple, fov: float, aspect: float,
                    near: float = 0.3, far: float = 520.0) -> np.ndarray:
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, up, [-c for c in fwd]
    view[:3, 3] = -(view[:3, :3] @ np.array(eye, float))
    t = 1.0 / math.tan(math.radians(fov) / 2)
    proj = np.zeros((4, 4))
    proj[0, 0] = t / aspect
    proj[1, 1] = -t  # upside down (see the module docstring)
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2 * far * near / (near - far)
    proj[3, 2] = -1.0
    return (proj @ view).astype(F32)
