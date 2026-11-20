"""Storm Drop's renderer: the island drawn by the graphics card (OpenGL 3.3 through Qt's own classes, no other
package) at the window's own resolution, into an image the game widget paints.

The ground is one mesh coloured by the island's painted map. Everything else is an *instance* of a few unit meshes
(box, cylinder, pyramid, cone, ball, flat quad, dome, tube): 24 floats give its place and size (a 3 x 4 matrix), two
colours, a pattern and four numbers for it. The patterns (planks, bricks, plates, shingles, bark, leaves, faces, the
storm wall ...) are worked out per pixel by the shader, so they stay crisp at any window size and need no picture
files. Things that stay (buildings, trees, loot) live in ``Batch`` buffers that are only sent again when a row
changes; players, shots and effects are sent every frame. The picture is drawn upside down on purpose so it can be read
back without a flip."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from PySide6.QtGui import (QImage, QMatrix4x4, QOffscreenSurface, QOpenGLContext, QOpenGLExtraFunctions,
                           QSurfaceFormat)
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
                              QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture, QOpenGLVertexArrayObject)

log = logging.getLogger(__name__)
F32 = np.float32
INST = 24
BOX, CYL, PYR, CONE, BALL, QUAD, DOME, TUBE = range(8)
(FLAT, WOODP, BRICKP, METALP, SHINGLE, BARK, LEAF, ROCK, WATER, DISC, RING, FADE, FACE, STORM, CANOPY, SUIT, GHOST,
 RIBS, CHESTP, CARTOP, CRATE, BEAM, STRIPES, GLOW, ENVELOPE) = range(25)
SPIN = 64  # added to a pattern: the instance turns and bobs (loot)
SOLID, CLEAR, ADD = 0, 1, 2
GL_TRIANGLES, GL_FLOAT, GL_DEPTH_TEST, GL_CULL_FACE, GL_BLEND = 0x0004, 0x1406, 0x0B71, 0x0B44, 0x0BE2
GL_DEPTH_BUFFER_BIT, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE = 0x0100, 0x0302, 0x0303, 1
GL_LEQUAL, GL_RGB8 = 0x0203, 0x8051


class Env:
    """The light of the moment, the storm and the clock the shaders animate by."""

    def __init__(self) -> None:
        self.zenith = (0.16, 0.50, 0.95)
        self.horizon = (0.72, 0.90, 0.98)
        self.sun = (0.46, 0.76, 0.46)
        self.sun_col = (0.80, 0.74, 0.62)
        self.amb_up = (0.56, 0.62, 0.72)
        self.amb_dn = (0.42, 0.40, 0.36)
        self.fog = (240.0, 950.0)
        self.time = 0.0
        self.storm = (256.0, 256.0, 9999.0)  # centre x, z and radius of the eye
        self.in_storm = 0.0  # 0..1: the camera is out in it


class Camera:
    def __init__(self) -> None:
        self.eye = (0.0, 10.0, 0.0)
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = 80.0
        self.vp = np.eye(4, dtype=F32)
        self.size = (1280, 780)
        self.right = (1.0, 0.0, 0.0)
        self.up = (0.0, 1.0, 0.0)
        self.fwd = (0.0, 0.0, 1.0)

    def update(self, w: float, h: float, near: float = 0.25, far: float = 1500.0) -> None:
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        self.fwd = (sy * cp, sp, cy * cp)
        self.right = (cy, 0.0, -sy)
        self.up = (-sy * sp, cp, -cy * sp)
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = self.right, self.up, [-c for c in self.fwd]
        view[:3, 3] = -(view[:3, :3] @ np.array(self.eye, float))
        t = 1.0 / math.tan(math.radians(self.fov) / 2)
        proj = np.zeros((4, 4))
        proj[0, 0] = t / (w / max(h, 1.0))
        proj[1, 1] = -t  # upside down (see the module docstring)
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2 * far * near / (near - far)
        proj[3, 2] = -1.0
        self.vp64 = proj @ view
        self.vp = self.vp64.astype(F32)
        self.size = (w, h)

    def project(self, x: float, y: float, z: float) -> tuple[float, float, float] | None:
        """World to widget pixels: (x, y, distance along the view), None behind the camera."""
        m = self.vp64
        cw = m[3, 0] * x + m[3, 1] * y + m[3, 2] * z + m[3, 3]
        if cw <= 0.05:
            return None
        cx = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + m[0, 3]
        cy = m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + m[1, 3]
        return (cx / cw * 0.5 + 0.5) * self.size[0], (cy / cw * 0.5 + 0.5) * self.size[1], cw


class Batch:
    """Instances of one mesh that stay on the card; a row can be written or freed on its own."""

    def __init__(self, mesh: int, mode: int = SOLID, cap: int = 256) -> None:
        self.mesh, self.mode = mesh, mode
        self.data = np.zeros((cap, INST), F32)
        self.n = 0
        self.free_rows: list[int] = []
        self.dirty = True
        self.part = None

    def add(self, row) -> int:  # noqa: ANN001
        if self.free_rows:
            i = self.free_rows.pop()
        else:
            if self.n >= len(self.data):
                self.data = np.concatenate([self.data, np.zeros_like(self.data)])
            i = self.n
            self.n += 1
        self.data[i] = row
        self.dirty = True
        return i

    def set(self, i: int, row) -> None:  # noqa: ANN001
        self.data[i] = row
        self.dirty = True

    def free(self, i: int) -> None:
        self.data[i] = 0.0
        self.free_rows.append(i)
        self.dirty = True

    def clear(self) -> None:
        self.data[:] = 0.0
        self.n = 0
        self.free_rows.clear()
        self.dirty = True


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

    for axis in range(3):  # box
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
    sides = 14  # cylinder along y
    ring = [(math.cos(i * math.tau / sides), math.sin(i * math.tau / sides)) for i in range(sides + 1)]
    for (c0, s0), (c1, s1) in zip(ring, ring[1:]):
        n0, n1 = (c0, 0.0, s0), (c1, 0.0, s1)
        tri((c0, -1, s0), (c1, -1, s1), (c1, 1, s1), n0, n1, n1)
        tri((c0, -1, s0), (c1, 1, s1), (c0, 1, s0), n0, n1, n0)
        tri((0, 1, 0), (c0, 1, s0), (c1, 1, s1), (0, 1, 0))
        tri((0, -1, 0), (c0, -1, s0), (c1, -1, s1), (0, -1, 0))
    close()
    k = 1.0 / math.sqrt(5.0)  # pyramid on the square base
    top = (0, 1, 0)
    quad((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1), (0, -1, 0))
    tri((1, -1, -1), (1, -1, 1), top, (2 * k, k, 0))
    tri((-1, -1, -1), (-1, -1, 1), top, (-2 * k, k, 0))
    tri((-1, -1, 1), (1, -1, 1), top, (0, k, 2 * k))
    tri((-1, -1, -1), (1, -1, -1), top, (0, k, -2 * k))
    close()
    sides = 9  # cone, a little faceted on purpose
    ring = [(math.cos(i * math.tau / sides), math.sin(i * math.tau / sides)) for i in range(sides + 1)]
    for (c0, s0), (c1, s1) in zip(ring, ring[1:]):
        nm = (k * (c0 + c1), k, k * (s0 + s1))
        tri((c0, -1, s0), (c1, -1, s1), top, nm)
        tri((0, -1, 0), (c0, -1, s0), (c1, -1, s1), (0, -1, 0))
    close()

    def on_ball(i: int, j: int, stacks: int, slices: int) -> tuple:
        th, ph = math.pi * i / stacks, math.tau * j / slices
        return (math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph))

    stacks, slices = 7, 12  # ball
    for i in range(stacks):
        for j in range(slices):
            p00, p01 = on_ball(i, j, stacks, slices), on_ball(i, j + 1, stacks, slices)
            p10, p11 = on_ball(i + 1, j, stacks, slices), on_ball(i + 1, j + 1, stacks, slices)
            if i > 0:
                tri(p00, p01, p11, p00, p01, p11)
            if i < stacks - 1:
                tri(p00, p11, p10, p00, p11, p10)
    close()
    quad((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1), (0, 1, 0))  # flat quad facing up
    close()
    stacks, slices = 10, 16  # dome: the top half of a ball, open below
    for i in range(stacks // 2):
        for j in range(slices):
            p00, p01 = on_ball(i, j, stacks, slices), on_ball(i, j + 1, stacks, slices)
            p10, p11 = on_ball(i + 1, j, stacks, slices), on_ball(i + 1, j + 1, stacks, slices)
            if i > 0:
                tri(p00, p01, p11, p00, p01, p11)
            tri(p00, p11, p10, p00, p11, p10)
    close()
    sides = 120  # tube: the storm wall
    ring = [(math.cos(i * math.tau / sides), math.sin(i * math.tau / sides)) for i in range(sides + 1)]
    for (c0, s0), (c1, s1) in zip(ring, ring[1:]):
        n0, n1 = (c0, 0.0, s0), (c1, 0.0, s1)
        tri((c0, -1, s0), (c1, -1, s1), (c1, 1, s1), n0, n1, n1)
        tri((c0, -1, s0), (c1, 1, s1), (c0, 1, s0), n0, n1, n0)
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

_LIGHT = """
uniform vec3 u_cam; uniform vec3 u_sun; uniform vec3 u_suncol; uniform vec3 u_ambup; uniform vec3 u_ambdn;
uniform vec3 u_fogcol; uniform vec2 u_fog; uniform vec3 u_storm; uniform float u_instorm;
vec3 finish(vec3 rgb, vec3 w) {
    float d = length(w - u_cam);
    float fog = smoothstep(u_fog.x, u_fog.y, d);
    float st = smoothstep(-1.5, 1.5, length(w.xz - u_storm.xy) - u_storm.z);
    rgb = mix(rgb, rgb * vec3(0.62, 0.42, 0.86) + vec3(0.10, 0.02, 0.16), st * 0.72);
    vec3 fc = mix(u_fogcol, vec3(0.42, 0.22, 0.62), max(st * 0.6, u_instorm * 0.8));
    fog = max(fog, u_instorm * smoothstep(8.0, 120.0, d) * 0.85);
    return mix(rgb, fc, fog);
}
"""

_VS = """#version 330 core
layout(location = 0) in vec3 a_pos; layout(location = 1) in vec3 a_nrm;
layout(location = 2) in vec4 a_m0; layout(location = 3) in vec4 a_m1; layout(location = 4) in vec4 a_m2;
layout(location = 5) in vec4 a_col; layout(location = 6) in vec4 a_col2; layout(location = 7) in vec4 a_par;
uniform mat4 u_vp; uniform float u_time;
out vec3 v_unit; out vec3 v_ln; out vec3 v_n; out vec3 v_w;
flat out vec3 v_scale; flat out vec4 v_col; flat out vec4 v_col2; flat out vec4 v_par; flat out vec3 v_org;
void main() {
    vec3 cx = vec3(a_m0.x, a_m1.x, a_m2.x), cy = vec3(a_m0.y, a_m1.y, a_m2.y), cz = vec3(a_m0.z, a_m1.z, a_m2.z);
    vec3 t = vec3(a_m0.w, a_m1.w, a_m2.w);
    vec3 sc = vec3(length(cx), length(cy), length(cz));
    vec3 ln = a_nrm / max(sc * sc, vec3(1e-8));
    vec3 off = cx * a_pos.x + cy * a_pos.y + cz * a_pos.z;
    vec3 n = cx * ln.x + cy * ln.y + cz * ln.z;
    float pat = a_col2.a;
    if (pat >= 63.5) {                              // loot: turns and bobs
        pat -= 64.0;
        float ph = t.x * 1.7 + t.z * 2.3;
        float a = u_time * 1.5 + ph; float c = cos(a), s = sin(a);
        off = vec3(off.x * c + off.z * s, off.y, -off.x * s + off.z * c);
        n = vec3(n.x * c + n.z * s, n.y, -n.x * s + n.z * c);
        t.y += 0.09 * sin(u_time * 2.2 + ph);
    }
    vec3 p = off + t;
    if (abs(pat - 6.0) < 0.5) {                     // leaves sway
        float k = (a_pos.y * 0.5 + 0.5) * 0.07;
        p.x += sin(u_time * 1.3 + t.x * 0.7 + t.z * 0.9) * k;
        p.z += cos(u_time * 1.1 + t.x * 0.5 - t.z * 0.6) * k;
    }
    v_n = normalize(n);
    v_unit = a_pos; v_ln = a_nrm; v_scale = sc; v_w = p; v_org = t;
    v_col = a_col; v_col2 = vec4(a_col2.rgb, pat); v_par = a_par;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

_FS = "#version 330 core\n" + _NOISE + _LIGHT + """
in vec3 v_unit; in vec3 v_ln; in vec3 v_n; in vec3 v_w;
flat in vec3 v_scale; flat in vec4 v_col; flat in vec4 v_col2; flat in vec4 v_par; flat in vec3 v_org;
uniform float u_time;
out vec4 o;
void main() {
    vec3 an = abs(v_ln);
    vec3 m = v_unit * v_scale;                      // metres in the instance's own frame
    vec2 uv, un; float side = 1.0;                  // side: 1 on an upright face, 0 on a level one
    if (an.y > an.x && an.y > an.z) { uv = m.xz; un = v_unit.xz; side = 0.0; }
    else if (an.x > an.z) { uv = m.zy; un = v_unit.zy; }
    else { uv = m.xy; un = v_unit.xy; }
    vec3 c = v_col.rgb; vec3 c2 = v_col2.rgb; float alpha = v_col.a; float emis = 0.0;
    int pat = int(v_col2.a + 0.5);
    vec3 N = normalize(v_n);
    bool lit = true;
    if (pat >= 1 && pat <= 4) {                     // build pieces: planks, bricks, plates, shingles
        alpha = 1.0;
        float hp = v_par.x, var = v_par.w;
        vec2 q = uv + v_org.xz * 0.37 + v_org.y;    // so neighbours do not repeat each other
        if (side < 0.5 && v_scale.y < 0.3) q = v_w.xz;
        if (pat == 1) {
            float s = (var > 1.5) ? q.y : q.x;
            if (side < 0.5) s = q.x;
            float g = s / 0.5;
            float fw = clamp(fwidth(g) * 2.0, 0.0, 1.0);
            c *= mix(0.86 + 0.26 * hash(vec2(floor(g), 3.0)), 1.0, fw);
            c = mix(c, c2, aa(0.9, tri(g)) * (1.0 - fw) * 0.8);
            c = mix(c, c2, (1.0 - aa(0.07, abs(fract(((var > 1.5) ? q.x : q.y) / 1.5 + 0.5) - 0.5))) * 0.55 * (1.0 - fw));
        } else if (pat == 2) {
            vec2 g = q / vec2(0.62, 0.3);
            g.x += 0.5 * mod(floor(g.y), 2.0);
            float fw = clamp(fwidth(g.y) * 2.0, 0.0, 1.0);
            vec2 f = abs(fract(g) - 0.5) * 2.0;
            float mortar = max(aa(0.93, f.x), aa(0.86, f.y));
            c *= mix(0.84 + 0.30 * hash(floor(g)), 1.0, fw);
            c = mix(c, c2, mix(mortar, 0.16, fw));
        } else if (pat == 3) {
            vec2 g = q / vec2(1.0, 1.5);
            float fw = clamp(fwidth(g.x) * 2.0, 0.0, 1.0);
            vec2 f = abs(fract(g) - 0.5) * 2.0;
            c *= mix(0.92 + 0.12 * hash(floor(g)), 1.0, fw);
            c = mix(c, c2, max(aa(0.95, f.x), aa(0.965, f.y)) * (1.0 - fw));
            vec2 rv = abs(f - vec2(0.82));
            c = mix(c, c * 1.25, (1.0 - aa(0.06, length(rv))) * (1.0 - fw));
        } else {
            vec2 g = vec2(v_w.x + v_w.z, v_w.y) / vec2(0.7, 0.36);
            g.x += 0.5 * mod(floor(g.y), 2.0);
            float fw = clamp(fwidth(g.y) * 2.0, 0.0, 1.0);
            vec2 f = fract(g);
            c *= mix(0.86 + 0.26 * hash(floor(g)), 1.0, fw);
            c = mix(c, c2, max(1.0 - aa(0.10, f.y), 1.0 - aa(0.05, f.x)) * (1.0 - fw) * 0.7);
        }
        if (var > 0.5 && var < 1.5 && side > 0.5 && v_scale.y > 1.0) {   // a window
            vec2 wq = vec2(un.x, (m.y + v_scale.y - 3.0 + 1.5) / 1.5);    // measured down from the wall's top
            vec2 d = abs(wq - vec2(0.0, 0.18)) - vec2(0.42, 0.38);
            float frame = 1.0 - aa(0.07, max(d.x, d.y));
            float glass = 1.0 - aa(0.0, max(d.x, d.y));
            float bar = max(1.0 - aa(0.035, abs(wq.x)), 1.0 - aa(0.035, abs(wq.y - 0.18)));
            c = mix(c, c2 * 0.8 + 0.15, frame);
            c = mix(c, mix(vec3(0.50, 0.80, 0.96), vec3(0.86, 0.96, 1.0), wq.y * 0.5 + 0.5), glass * (1.0 - bar));
        }
        if (side > 0.5) c *= 0.88 + 0.12 * smoothstep(-1.0, -0.5, v_unit.y);
        float lost = 1.0 - hp;                      // damage: cracks spread out from a few spots, then holes of dark
        float cn = vnoise(q * 1.6 + 7.0);
        float crack = smoothstep(0.022, 0.0, abs(cn - 0.5)) * smoothstep(0.75 - lost, 0.95 - lost * 0.6, vnoise(q * 0.7 + 3.0) + lost * 0.5);
        c = mix(c, c * 0.42, clamp(crack * step(0.08, lost), 0.0, 1.0));
        c = mix(c, c * 0.6, smoothstep(0.62, 0.9, lost) * step(0.6, vnoise(q * 2.4 + 11.0)));
        c *= 1.0 - 0.18 * lost;
        float prog = v_par.z > 0.0 ? clamp((u_time - v_par.y) / v_par.z, 0.0, 1.0) : 1.0;
        if (prog < 1.0) {                           // still going up: a blueprint above the finished part
            float hgt = (side > 0.5) ? un.y * 0.5 + 0.5 : un.x * 0.5 + 0.5;
            float open = aa(prog, hgt);
            vec2 gl = abs(fract(uv / 0.5) - 0.5) * 2.0;
            float grid = max(aa(0.86, gl.x), aa(0.86, gl.y));
            vec3 blue = mix(vec3(0.30, 0.62, 0.95), vec3(0.85, 0.96, 1.0), grid);
            c = mix(c, blue, open * 0.82);
            c += vec3(0.5, 0.8, 1.0) * (1.0 - smoothstep(0.0, 0.05, abs(hgt - prog))) * 0.8;
            emis = open * 0.35;
        }
        c += vec3(0.5) * clamp(1.0 - (u_time - v_col.a) * 7.0, 0.0, 1.0) * step(0.0, v_col.a);
    } else if (pat == 5) {                          // bark
        float g = (atan(v_unit.z, v_unit.x) * 3.0 + m.y * 0.3);
        c = mix(c, c2, aa(0.55, tri(g)) * 0.7);
        c *= 0.9 + 0.2 * vnoise(vec2(g * 2.0, m.y * 3.0));
    } else if (pat == 6) {                          // leaves: blotches, lighter on top
        float n = vnoise(v_unit.xz * 2.6 + v_org.xz) * 0.6 + vnoise(v_unit.xy * 5.0 + v_org.zx) * 0.4;
        c = mix(c, c2, smoothstep(0.42, 0.58, n));
        c *= 0.82 + 0.3 * smoothstep(-0.6, 0.9, v_unit.y);
    } else if (pat == 7) {                          // rock: flat facets
        vec3 fn = normalize(cross(dFdx(v_w), dFdy(v_w)));
        float n = vnoise(v_unit.xz * 2.0 + v_org.xz) * 0.5 + vnoise(v_unit.xy * 3.0 + v_org.zx) * 0.5;
        c = mix(c, c2, smoothstep(0.4, 0.6, n));
        N = normalize(mix(N, faceforward(fn, -N, fn), 0.35));
    } else if (pat == 8) {                          // the sea
        vec2 wq = v_w.xz;
        float r1 = vnoise(wq * 0.22 + vec2(u_time * 0.22, u_time * 0.13));
        float r2 = vnoise(wq * vec2(0.6, 0.2) - vec2(u_time * 0.3, 0.0));
        float d = length(v_w - u_cam);
        float fw = 1.0 - clamp(d / 420.0, 0.0, 1.0);
        float fw2 = fw * fw * (1.0 - clamp(fwidth(wq.x) * 0.8, 0.0, 1.0));
        c = mix(c, c2, mix(0.4, r1 * 0.75, fw2));
        c += vec3(0.95) * smoothstep(0.62, 0.72, r1 * r2 * 2.0) * 0.32 * fw2;
        vec3 V = normalize(u_cam - v_w);
        float fres = pow(1.0 - max(V.y, 0.0), 3.0);
        c = mix(c, u_fogcol, fres * 0.55);
        vec3 wn = normalize(vec3((r1 - 0.5) * 0.3 * fw2, 1.0, (r2 - 0.5) * 0.3 * fw2));
        c += u_suncol * pow(max(dot(wn, normalize(u_sun + V)), 0.0), 90.0) * 0.55;
        alpha = mix(v_col.a, 1.0, clamp(fres + d / 300.0, 0.0, 1.0));
        lit = false;
    } else if (pat == 9) {                          // a soft disc: shadows and glows
        alpha *= 1.0 - smoothstep(v_par.x, 1.0, length(un));
        emis = v_par.z; lit = v_par.z < 0.5;
    } else if (pat == 10) {                         // a ring
        float r = length(un);
        alpha *= aa(v_par.x, r) * (1.0 - aa(0.97, r));
        emis = v_par.z; lit = false;
    } else if (pat == 11) {                         // from the second colour at the bottom to the first on top
        c = mix(c2, c, clamp(v_unit.y * 0.5 + 0.5, 0.0, 1.0));
        emis = v_par.z;
    } else if (pat == 12) {                         // a head: eyes on the front
        if (v_ln.z > 0.5) {
            vec2 e = abs(vec2(abs(un.x) - 0.42, un.y - 0.05)) - vec2(0.13, 0.2);
            float eye = 1.0 - aa(0.0, max(e.x, e.y));
            c = mix(c, vec3(0.12, 0.10, 0.14), eye);
            vec2 g = abs(vec2(abs(un.x) - 0.46, un.y - 0.14)) - vec2(0.045, 0.06);
            c = mix(c, vec3(1.0), (1.0 - aa(0.0, max(g.x, g.y))));
            vec2 mo = abs(vec2(un.x, un.y + 0.52)) - vec2(0.2, 0.035);
            c = mix(c, c * 0.55, 1.0 - aa(0.0, max(mo.x, mo.y)));
        }
        c = mix(c, c2, step(0.5, v_par.x) * aa(0.55, un.y) * step(0.5, side) + step(0.5, v_par.x) * (1.0 - side) * step(0.0, v_ln.y));
    } else if (pat == 13) {                         // the storm wall
        float ang = atan(v_unit.z, v_unit.x);
        float y = v_w.y * 0.045;
        float n = vnoise(vec2(ang * 34.0 + u_time * 0.35, y * 2.0 - u_time * 0.5)) * 0.6
            + vnoise(vec2(ang * 90.0 - u_time * 0.5, y * 6.0 - u_time * 1.1)) * 0.4;
        float d = length(v_w - u_cam);
        float top = 1.0 - smoothstep(40.0, 150.0, v_w.y);
        float rain = smoothstep(0.80, 0.98, vnoise(vec2(ang * 420.0, y * 1.5 + u_time * 2.6)))
            * (1.0 - smoothstep(30.0, 140.0, d));  // streaks of rain, close up only
        c = mix(c, c2, n) + vec3(0.8, 0.7, 1.0) * rain * 0.5;
        alpha = v_col.a * (0.6 + 0.4 * n) * smoothstep(0.0, 10.0, d) * top;
        lit = false;
    } else if (pat == 14) {                         // a canopy: gores in two colours
        float ang = atan(v_unit.z, v_unit.x);
        c = mix(c, c2, aa(0.5, tri(ang * 4.0 / 6.2832)));
        c *= 0.8 + 0.25 * v_unit.y;
    } else if (pat == 15) {                         // a suit: belt, collar and a chest mark
        if (side > 0.5) {
            float belt = (1.0 - aa(0.16, abs(un.y + 0.72)));
            c = mix(c, c2, belt);
            if (v_ln.z > 0.5) {
                c = mix(c, c2, (1.0 - aa(0.2, abs(un.x) + abs(un.y - 0.25))));
                c = mix(c, vec3(1.0, 0.9, 0.5), belt * (1.0 - aa(0.14, abs(un.x))));
            }
        }
    } else if (pat == 16) {                         // the build preview
        vec2 gl = abs(fract(uv / 1.0) - 0.5) * 2.0;
        float grid = max(aa(0.9, gl.x), aa(0.9, gl.y));
        float edge = max(aa(0.94, abs(un.x)), aa(0.94, abs(un.y)));
        alpha = v_col.a * (0.34 + 0.4 * max(grid, edge)) * (0.85 + 0.15 * sin(u_time * 6.0));
        lit = false;
    } else if (pat == 17) {                         // corrugated steel with a band
        if (side > 0.5) {
            float fw = clamp(fwidth(uv.x / v_par.x) * 1.6, 0.0, 1.0);
            c *= mix(0.80 + 0.2 * smoothstep(0.15, 0.85, tri(uv.x / v_par.x)), 0.9, fw);
            c = mix(c, c2, aa(v_par.y - 0.12, un.y) * (1.0 - aa(v_par.y + 0.12, un.y)));
        }
    } else if (pat == 18) {                         // a chest: bands and a clasp
        float band = max(1.0 - aa(0.12, abs(abs(un.x) - 0.62)), aa(0.84, abs(un.y)) * side);
        c = mix(c, c2, band);
        if (v_ln.z > 0.5) c = mix(c, vec3(1.0, 0.95, 0.6), 1.0 - aa(0.16, max(abs(un.x), abs(un.y - 0.5))));
        emis = v_par.z * band;
    } else if (pat == 19) {                         // a car's cabin: glass all round
        if (side > 0.5) {
            vec2 d = abs(un) - vec2(0.84, 0.62);
            c = mix(c2, c, aa(0.0, max(d.x, d.y)));
        }
    } else if (pat == 20) {                         // a crate: a frame round planks
        float g = (uv.x + uv.y) / 0.36;
        float fw = clamp(fwidth(g) * 2.0, 0.0, 1.0);
        c = mix(c, c * 0.8, aa(0.86, tri(g)) * (1.0 - fw));
        c = mix(c, c2, max(aa(0.76, abs(un.x)), aa(0.76, abs(un.y))));
    } else if (pat == 21) {                         // a beam of light: a tracer
        alpha *= (1.0 - smoothstep(0.2, 1.0, length(v_unit.xy))) * (0.35 + 0.65 * (v_unit.z * 0.5 + 0.5));
        emis = 1.0; lit = false;
    } else if (pat == 22) {                         // bands round an upright thing
        c = mix(c, c2, aa(0.5, tri(v_unit.y * v_par.x + v_par.y)) * step(0.5, side));
        emis = v_par.z;
    } else if (pat == 23) {                         // a ball of light: softer at the rim
        vec3 V = normalize(u_cam - v_w);
        alpha *= pow(max(dot(N, V), 0.0), v_par.x);
        emis = 1.0; lit = false;
    } else if (pat == 24) {                         // the ship's envelope: stripes from nose to tail
        float ang = atan(v_unit.y, v_unit.x);
        c = mix(c, c2, aa(0.5, tri(ang * 5.0 / 6.2832)));
        c = mix(c, c2 * 0.7, aa(0.78, abs(v_unit.z)));
    } else {
        emis = v_par.z;
    }
    vec3 rgb = c;
    if (lit) {
        vec3 light = mix(u_ambdn, u_ambup, N.y * 0.5 + 0.5) + u_suncol * max(dot(N, u_sun), 0.0);
        rgb = c * mix(light, vec3(1.0), clamp(emis, 0.0, 1.0)) + c * max(emis - 1.0, 0.0);
    }
    o = vec4((pat == 21 || pat == 23 || pat == 10) ? rgb : finish(rgb, v_w), alpha);
}
"""

_GROUND_VS = """#version 330 core
layout(location = 0) in vec3 a_pos; layout(location = 1) in vec3 a_nrm;
uniform mat4 u_vp;
out vec3 v_w; out vec3 v_n;
void main() { v_w = a_pos; v_n = a_nrm; gl_Position = u_vp * vec4(a_pos, 1.0); }
"""

_GROUND_FS = "#version 330 core\n" + _NOISE + _LIGHT + """
in vec3 v_w; in vec3 v_n; out vec4 o;
uniform sampler2D u_ground; uniform float u_size;
void main() {
    vec4 g = texture(u_ground, v_w.xz / u_size);
    vec3 c = g.rgb;
    float near = 1.0 - clamp(fwidth(v_w.x) * 1.2, 0.0, 1.0);
    float near2 = 1.0 - clamp(fwidth(v_w.x) * 6.0, 0.0, 1.0);
    c *= 1.0 + (aa(0.56, vnoise(v_w.xz * 0.8)) - 0.45) * 0.07 * near
        + (aa(0.60, vnoise(v_w.xz * 2.9 + 17.0)) - 0.35) * 0.07 * near
        + (aa(0.66, vnoise(v_w.xz * 11.0 + 5.0)) - 0.3) * 0.06 * near2
        + (vnoise(v_w.xz * 30.0) - 0.5) * 0.08 * near2;
    vec3 N = normalize(v_n);
    vec3 light = mix(u_ambdn, u_ambup, N.y * 0.5 + 0.5) * mix(0.84, 1.0, g.a) + u_suncol * max(dot(N, u_sun), 0.0) * g.a;
    c *= light;
    c = mix(c, c * vec3(0.55, 0.78, 0.86), smoothstep(0.0, -3.0, v_w.y));   // under the sea
    o = vec4(finish(c, v_w), 1.0);
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
uniform vec3 u_sun; uniform vec3 u_suncol; uniform float u_time; uniform float u_instorm;
void main() {
    vec3 d = normalize(u_f + u_r * v_ndc.x - u_u * v_ndc.y);
    float up = max(d.y, 0.0);
    vec3 c = mix(u_horizon, u_zenith, pow(up, 0.7));
    float sd = max(dot(d, u_sun), 0.0);
    c += u_suncol * (smoothstep(0.9990, 0.9994, sd) * 1.5 + pow(sd, 48.0) * 0.35 + pow(sd, 6.0) * 0.12);
    if (d.y > 0.0) {
        vec2 q = d.xz / (d.y + 0.12) * 0.8 + vec2(u_time * 0.008, u_time * 0.003);
        float n = vnoise(q) * 0.55 + vnoise(q * 2.3 + 7.0) * 0.3 + vnoise(q * 5.1 + 3.0) * 0.15;
        float cl = smoothstep(0.52, 0.70, n) * smoothstep(0.02, 0.2, d.y);
        c = mix(c, mix(vec3(1.0), vec3(0.86, 0.90, 0.98), smoothstep(0.6, 0.8, n)), cl * 0.92);
    } else {
        c = mix(u_horizon, vec3(0.10, 0.42, 0.66), smoothstep(0.0, -0.25, d.y));
    }
    float n2 = vnoise(d.xz * 3.0 / (abs(d.y) + 0.35) + vec2(u_time * 0.2, 0.0));
    c = mix(c, mix(vec3(0.30, 0.14, 0.48), vec3(0.62, 0.36, 0.80), n2), u_instorm * 0.85);
    o = vec4(c, 1.0);
}
"""


class _Part:
    __slots__ = ("count", "vao", "vbo")

    def __init__(self) -> None:
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.count = 0

    def destroy(self) -> None:
        self.vao.destroy()
        self.vbo.destroy()


class Renderer:
    """Draws the ground, the batches and this frame's instance lists through a camera into a QImage. Everything
    OpenGL is made the first time it draws."""

    def __init__(self) -> None:
        self.ctx: QOpenGLContext | None = None
        self.surface: QOffscreenSurface | None = None
        self.f: QOpenGLExtraFunctions | None = None
        self.fbo: QOpenGLFramebufferObject | None = None
        self.resolve: QOpenGLFramebufferObject | None = None
        self.samples = 4
        self.gpu = ""
        self.batches: list[Batch] = []
        self._dead_parts: list[_Part] = []
        self._dyn: tuple[dict, dict, dict] = ({}, {}, {})
        self._ground_src: tuple | None = None  # (mesh, rgba, size) waiting to be sent
        self._ground_n = 0
        self.ground_tex: QOpenGLTexture | None = None
        self.stats = {"ms": 0.0, "instances": 0}

    # --- set-up -----------------------------------------------------------------------------------------------------

    def _init(self) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            raise RuntimeError("Storm Drop needs OpenGL 3.3 (no graphics context could be made)")
        surface = QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()
        if not ctx.makeCurrent(surface):
            raise RuntimeError("Storm Drop could not use its graphics context")
        self.ctx, self.surface = ctx, surface
        f = self.f = QOpenGLExtraFunctions(ctx)
        f.initializeOpenGLFunctions()
        self.gpu = str(f.glGetString(0x1F01) or "")
        self.p_main = self._program(_VS, _FS)
        self.p_ground = self._program(_GROUND_VS, _GROUND_FS)
        self.p_sky = self._program(_SKY_VS, _SKY_FS)
        self.loc = {id(self.p_main): {}, id(self.p_sky): {}, id(self.p_ground): {}}
        verts, self.ranges = _unit_meshes()
        self.mesh_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.mesh_vbo.create()
        self.mesh_vbo.bind()
        raw = verts.tobytes()
        self.mesh_vbo.allocate(raw, len(raw))
        self.mesh_vbo.release()
        self.sky_vao = QOpenGLVertexArrayObject()
        self.sky_vao.create()
        self.ground_vao = QOpenGLVertexArrayObject()
        self.ground_vao.create()
        self.ground_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.ground_vbo.create()

    def warm_up(self) -> None:
        if self.ctx is None:
            self._init()
            self.ctx.doneCurrent()

    def _program(self, vs: str, fs: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs) and ok
        if not (ok and prog.link()):
            raise RuntimeError(f"Storm Drop shader did not compile: {prog.log()}")
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

    def _upload(self, part: _Part, data: np.ndarray) -> None:
        raw = np.ascontiguousarray(data, F32).tobytes()
        part.vbo.bind()
        part.vbo.allocate(raw, len(raw))
        part.vbo.release()
        part.count = len(data)

    def _fill(self, parts: dict, arrays: dict) -> None:
        for part in parts.values():
            part.count = 0
        for mesh, data in arrays.items():
            if len(data):
                part = parts.get(mesh)
                if part is None:
                    part = parts[mesh] = self._part()
                self._upload(part, data)

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

    # --- what stays ---------------------------------------------------------------------------------------------------

    def set_ground(self, mesh: np.ndarray, rgba: np.ndarray, size: float) -> None:
        """The island: triangles of (position, normal) and its painted map (sent at the next frame)."""
        self._ground_src = (mesh, rgba, size)
        self.ground_size = size

    def batch(self, mesh: int, mode: int = SOLID, cap: int = 256) -> Batch:
        b = Batch(mesh, mode, cap)
        self.batches.append(b)
        return b

    def drop_batches(self) -> None:
        for b in self.batches:
            if b.part is not None:
                self._dead_parts.append(b.part)
        self.batches = []

    def _sync(self) -> None:
        for part in self._dead_parts:
            part.destroy()
        self._dead_parts.clear()
        if self._ground_src is not None:
            mesh, rgba, _size = self._ground_src
            self._ground_src = None
            f, prog = self.f, self.p_ground
            raw = np.ascontiguousarray(mesh, F32).tobytes()
            self.ground_vao.bind()
            self.ground_vbo.bind()
            self.ground_vbo.allocate(raw, len(raw))
            for loc, offset in ((0, 0), (1, 12)):
                f.glEnableVertexAttribArray(loc)
                prog.setAttributeBuffer(loc, GL_FLOAT, offset, 3, 24)
            self.ground_vao.release()
            self.ground_vbo.release()
            self._ground_n = len(mesh)
            if self.ground_tex is not None:
                self.ground_tex.destroy()
            h, w = rgba.shape[:2]
            t = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            t.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
            t.setSize(w, h)
            t.setMipLevels(max(int(math.log2(max(w, h))) + 1, 1))
            t.allocateStorage(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8)
            t.setData(0, 0, 1, QOpenGLTexture.CubeMapFace.CubeMapPositiveX, QOpenGLTexture.PixelFormat.RGBA,
                      QOpenGLTexture.PixelType.UInt8, np.ascontiguousarray(rgba).tobytes())  # one call (see notes)
            t.generateMipMaps()
            t.setMinMagFilters(QOpenGLTexture.Filter.LinearMipMapLinear, QOpenGLTexture.Filter.Linear)
            t.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
            self.ground_tex = t
        for b in self.batches:
            if b.dirty:
                if b.part is None:
                    b.part = self._part()
                self._upload(b.part, b.data[:b.n])
                b.dirty = False

    def close(self) -> None:
        """Give everything back to the graphics card."""
        if self.ctx is None:
            return
        try:
            if self.ctx.makeCurrent(self.surface):
                for b in self.batches:
                    if b.part is not None:
                        b.part.destroy()
                        b.part = None
                for parts in self._dyn:
                    for part in parts.values():
                        part.destroy()
                for part in self._dead_parts:
                    part.destroy()
                self.mesh_vbo.destroy()
                self.ground_vbo.destroy()
                self.ground_vao.destroy()
                self.sky_vao.destroy()
                if self.ground_tex is not None:
                    self.ground_tex.destroy()
                    self.ground_tex = None
                self.fbo = self.resolve = None
                self.ctx.doneCurrent()
        except Exception:  # noqa: BLE001 - closing must never fail
            log.debug("Storm Drop renderer close", exc_info=True)
        self.batches = []
        self._dead_parts = []
        self._dyn = ({}, {}, {})
        self._ground_n = 0
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

    def _common(self, prog: QOpenGLShaderProgram, cam: Camera, env: Env) -> None:
        f = self.f
        prog.setUniformValue(self._u(prog, "u_vp"), QMatrix4x4(*[float(v) for v in cam.vp.reshape(-1)]))
        for name, v in (("u_cam", cam.eye), ("u_sun", env.sun), ("u_suncol", env.sun_col), ("u_ambup", env.amb_up),
                        ("u_ambdn", env.amb_dn), ("u_fogcol", env.horizon), ("u_storm", env.storm)):
            f.glUniform3f(self._u(prog, name), float(v[0]), float(v[1]), float(v[2]))
        f.glUniform2f(self._u(prog, "u_fog"), float(env.fog[0]), float(env.fog[1]))
        f.glUniform1f(self._u(prog, "u_instorm"), float(env.in_storm))

    def render(self, w: int, h: int, cam: Camera, env: Env, solid: dict, clear: dict, add: dict) -> QImage:
        """One frame: ``solid`` / ``clear`` / ``add``: {mesh: (n, 24) rows} of this frame's moving things."""
        t0 = time.perf_counter()
        if self.ctx is None:
            self._init()
        elif not self.ctx.makeCurrent(self.surface):
            raise RuntimeError("Storm Drop lost its graphics context")
        f = self.f
        w, h = max(w, 16), max(h, 16)
        self._size(w, h)
        self._sync()
        self._fill(self._dyn[0], solid)
        self._fill(self._dyn[1], clear)
        self._fill(self._dyn[2], add)
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
        tan = math.tan(math.radians(cam.fov) / 2)
        for name, v in (("u_r", [c * tan * w / h for c in cam.right]), ("u_u", [c * tan for c in cam.up]),
                        ("u_f", cam.fwd), ("u_zenith", env.zenith), ("u_horizon", env.horizon), ("u_sun", env.sun),
                        ("u_suncol", env.sun_col)):
            f.glUniform3f(self._u(prog, name), float(v[0]), float(v[1]), float(v[2]))
        f.glUniform1f(self._u(prog, "u_time"), float(env.time))
        f.glUniform1f(self._u(prog, "u_instorm"), float(env.in_storm))
        self.sky_vao.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.sky_vao.release()
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LEQUAL)
        # the ground
        if self._ground_n and self.ground_tex is not None:
            prog = self.p_ground
            prog.bind()
            self._common(prog, cam, env)
            f.glUniform1f(self._u(prog, "u_size"), float(self.ground_size))
            self.ground_tex.bind(0)
            prog.setUniformValue1i(self._u(prog, "u_ground"), 0)
            self.ground_vao.bind()
            f.glDrawArrays(GL_TRIANGLES, 0, self._ground_n)
            self.ground_vao.release()
        # everything solid
        prog = self.p_main
        prog.bind()
        self._common(prog, cam, env)
        f.glUniform1f(self._u(prog, "u_time"), float(env.time))
        count = 0
        ranges = self.ranges

        def draw_batches(mode: int) -> int:
            n = 0
            for b in self.batches:
                if b.mode == mode and b.part is not None and b.part.count:
                    first, cnt = ranges[b.mesh]
                    b.part.vao.bind()
                    f.glDrawArraysInstanced(GL_TRIANGLES, first, cnt, b.part.count)
                    n += b.part.count
            return n

        count += draw_batches(SOLID)
        count += self._draw(self._dyn[0])
        # see-through things: the sea and shadows first, then the rest, then lights
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        f.glDepthMask(False)
        count += draw_batches(CLEAR)
        count += self._draw(self._dyn[1])
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        count += draw_batches(ADD)
        count += self._draw(self._dyn[2])
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
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
