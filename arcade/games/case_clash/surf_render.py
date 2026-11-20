"""The surf view drawn by the graphics card: OpenGL 3.3 through Qt's own classes (no other package), into an
off-screen picture at the window's own resolution, which the game widget paints pixel for pixel.

Every map is one buffer of triangles drawn in a single call; the surfaces (ramp faces with their grid and ridge
trim, platform tiles, columns, the far scenery and the floor below) are drawn procedurally in the fragment shader,
so there are no texture images at all. The sky (gradient, sun, clouds or stars) is one full-screen pass; the zones
are glowing see-through walls drawn last. The picture is drawn upside down on purpose so it can be read back without
a flip (as in BlockCraft's renderer)."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from PySide6.QtGui import QImage, QMatrix4x4, QOffscreenSurface, QOpenGLContext, QOpenGLExtraFunctions, QSurfaceFormat
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
                              QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject)

from app.arcade.games.case_clash import surf_maps as SM

log = logging.getLogger(__name__)
F32 = np.float32
GL_TRIANGLES, GL_FLOAT = 0x0004, 0x1406
GL_DEPTH_TEST, GL_CULL_FACE, GL_BLEND = 0x0B71, 0x0B44, 0x0BE2
GL_DEPTH_BUFFER_BIT, GL_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_LESS, GL_RGB8 = 0x0100, 0x0302, 1, 0x0303, 0x0201, 0x8051
STYLE = {"dune": 0, "glacier": 1, "neon": 2}

_SKY_GLSL = """
uniform vec3 u_top; uniform vec3 u_hor; uniform vec3 u_sea; uniform vec3 u_sun; uniform vec3 u_suncol;
vec3 skyColor(vec3 d) {
    vec3 c = d.z >= 0.0 ? mix(u_hor, u_top, pow(clamp(d.z, 0.0, 1.0), 0.55)) : mix(u_hor, u_sea, smoothstep(0.0, 0.22, -d.z));
    float s = max(dot(d, u_sun), 0.0);
    c += u_suncol * (pow(s, 8.0) * 0.18 + pow(s, 64.0) * 0.25) * (d.z > -0.05 ? 1.0 : 0.3);
    return c;
}
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) { float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; } return s; }
"""

_WORLD_VS = """#version 330 core
layout(location = 0) in vec3 a_pos; layout(location = 1) in vec3 a_nrm;
layout(location = 2) in vec2 a_uv; layout(location = 3) in vec2 a_mat;
uniform mat4 u_vp;
out vec3 v_pos; out vec3 v_nrm; out vec2 v_uv; flat out int v_mat; out float v_slant;
void main() {
    v_pos = a_pos; v_nrm = a_nrm; v_uv = a_uv; v_mat = int(a_mat.x + 0.5); v_slant = a_mat.y;
    gl_Position = u_vp * vec4(a_pos, 1.0);
}
"""

_WORLD_FS = "#version 330 core\n" + _SKY_GLSL + """
in vec3 v_pos; in vec3 v_nrm; in vec2 v_uv; flat in int v_mat; in float v_slant;
uniform vec3 u_eye; uniform vec3 u_face; uniform vec3 u_trim; uniform vec3 u_under; uniform vec3 u_plata;
uniform vec3 u_platb; uniform vec3 u_pillar; uniform vec3 u_decor;
uniform float u_amb; uniform float u_fog; uniform float u_grid; uniform float u_glow; uniform float u_time;
uniform int u_style;
out vec4 o;
float lines(vec2 p, float cell, float w) {
    vec2 g = abs(fract(p / cell + 0.5) - 0.5) * cell;
    vec2 fw = max(fwidth(p), vec2(1e-3));
    vec2 l = 1.0 - smoothstep(vec2(w * 0.5), vec2(w * 0.5) + fw * 1.2, g);
    l *= clamp(cell / (fw * 6.0), 0.0, 1.0);  // fade out where the cells get smaller than a few pixels
    return max(l.x, l.y);
}
void main() {
    vec3 n = normalize(v_nrm);
    vec3 col; vec3 glow = vec3(0.0);
    if (v_mat == 0) {  // a ramp face: grid for the sense of speed, a bright trim along the ridge
        float v = v_uv.y;
        col = u_face * mix(1.06, 0.80, clamp(v / max(v_slant, 1.0), 0.0, 1.0));
        float g = lines(v_uv, 128.0, 3.0);
        float big = lines(vec2(v_uv.x, 0.0), 1024.0, 14.0);
        col = mix(col, col * 0.72, g * u_grid * (1.0 - u_glow));
        col = mix(col, col * 0.80, big * (1.0 - u_glow) * 0.6);
        glow += u_trim * (g * 0.55 + big * 0.9) * u_glow;
        float fw = max(fwidth(v), 1e-3);
        float trim = 1.0 - smoothstep(22.0, 22.0 + fw * 1.5, v);
        float foot = smoothstep(v_slant - 14.0 - fw, v_slant - 14.0, v);
        col = mix(col, u_trim, trim);
        col = mix(col, u_trim * 0.55, foot);
        glow += u_trim * (trim + foot * 0.5) * (0.25 + 0.9 * u_glow);
    } else if (v_mat == 1) {
        col = u_under;
    } else if (v_mat == 2) {  // platform: big tiles with joints; an edge band on the top
        vec2 c = floor(v_uv / 128.0);
        col = mod(c.x + c.y, 2.0) < 0.5 ? u_plata : u_platb;
        if (v_slant < 0.5) col *= 0.7;
        float j = lines(v_uv, 128.0, 4.0);
        col = mix(col, col * 0.7, j * (1.0 - u_glow));
        glow += u_trim * j * u_glow * 0.6;
    } else if (v_mat == 3) {  // a column
        col = u_pillar * (0.9 + 0.1 * sin(v_uv.y * 0.01));
        float band = lines(vec2(0.0, v_uv.y), 600.0, 30.0);
        col = mix(col, col * 0.75, band);
        glow += u_trim * band * u_glow * 0.5;
    } else if (v_mat == 4) {  // far scenery
        col = u_decor * (0.85 + 0.25 * vnoise(v_uv * 0.0015));
        if (u_style == 2 && abs(n.z) < 0.5) {  // lit windows of the towers
            vec2 cell = floor(v_uv / vec2(260.0, 340.0));
            vec2 f = fract(v_uv / vec2(260.0, 340.0));
            float on = step(0.55, hash(cell)) * step(0.2, f.x) * step(f.x, 0.8) * step(0.25, f.y) * step(f.y, 0.75);
            vec3 wc = mix(vec3(0.1, 0.9, 1.0), vec3(1.0, 0.25, 0.8), step(0.5, hash(cell + 7.0)));
            glow += wc * on * 0.8;
        }
    } else {  // the floor far below
        vec2 p = v_uv;
        if (u_style == 0) {
            float r = sin(p.x * 0.0011 + sin(p.y * 0.0007) * 2.0) * 0.5 + 0.5;
            col = u_sea * (0.82 + 0.22 * r);
        } else if (u_style == 1) {
            col = mix(u_sea * 0.9, vec3(1.0), smoothstep(0.35, 0.8, fbm(p * 0.00012 + u_time * 0.004)));
        } else {
            col = u_sea;
            glow += vec3(0.9, 0.15, 0.8) * lines(p, 2000.0, 40.0) * 0.8;
        }
    }
    float diff = clamp(dot(n, u_sun) * 0.6 + 0.4, 0.0, 1.0);  // wrapped: a face away from the sun stays readable
    float sky = 0.5 + 0.5 * n.z;
    vec3 lit = col * (u_amb * (0.8 + 0.3 * sky) + (1.0 - u_amb) * diff * diff * u_suncol);
    vec3 rgb = lit + glow;
    vec3 rel = v_pos - u_eye;
    float d = length(rel);
    float f = 1.0 - exp(-d * u_fog);
    if (v_mat == 5) f = max(f, 0.35);
    rgb = mix(rgb, skyColor(rel / max(d, 1e-3)), clamp(f, 0.0, 1.0));
    o = vec4(rgb, 1.0);
}
"""

_SKY_VS = """#version 330 core
out vec2 v_ndc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
    v_ndc = p; gl_Position = vec4(p, 1.0, 1.0);
}
"""

_SKY_FS = "#version 330 core\n" + _SKY_GLSL + """
in vec2 v_ndc; out vec4 o;
uniform vec3 u_r; uniform vec3 u_u; uniform vec3 u_f;
uniform float u_time; uniform float u_stars; uniform float u_clouds; uniform vec3 u_cloudcol; uniform int u_style;
void main() {
    vec3 d = normalize(u_f + u_r * v_ndc.x - u_u * v_ndc.y);
    vec3 c = skyColor(d);
    float s = dot(d, u_sun);
    if (s > 0.9994) c = mix(c, vec3(1.0) * 0.6 + u_suncol * 0.6, smoothstep(0.9994, 0.9997, s));
    if (u_clouds > 0.01 && d.z > 0.01) {
        vec2 q = d.xy / (d.z + 0.08) * 1.6 + vec2(u_time * 0.006, u_time * 0.002);
        float n = fbm(q);
        float a = smoothstep(0.5, 0.78, n) * u_clouds * smoothstep(0.01, 0.12, d.z);
        vec3 cc = u_cloudcol * (0.82 + 0.25 * fbm(q * 2.0 + 3.0)) + u_suncol * pow(max(s, 0.0), 6.0) * 0.25;
        c = mix(c, cc, a);
    }
    if (u_stars > 0.01 && d.z > -0.02) {
        vec3 a = abs(d); float m = max(a.x, max(a.y, a.z));
        vec2 uv = m == a.z ? d.xy / m : (m == a.x ? d.yz / m : d.xz / m);
        float h = hash(floor(uv * 150.0) + (m == a.z ? 7.0 : (m == a.x ? 19.0 : 31.0)) * sign(dot(d, vec3(1.0))));
        if (h > 0.993) c += vec3(0.85, 0.9, 1.0) * u_stars * (0.6 + 0.4 * sin(u_time * 1.9 + h * 900.0))
            * clamp(d.z * 5.0 + 0.2, 0.0, 1.0);
    }
    if (u_style == 2) c += vec3(0.6, 0.1, 0.55) * exp(-abs(d.z) * 14.0) * 0.35;  // the city's glow on the horizon
    o = vec4(c, 1.0);
}
"""

_ZONE_VS = """#version 330 core
layout(location = 0) in vec4 a_pos;
uniform mat4 u_vp;
out float v_h;
void main() { v_h = a_pos.w; gl_Position = u_vp * vec4(a_pos.xyz, 1.0); }
"""

_ZONE_FS = """#version 330 core
in float v_h; out vec4 o;
uniform vec4 u_col;
void main() { float a = pow(1.0 - v_h, 1.6) * u_col.a; o = vec4(u_col.rgb * a, a); }
"""


def basis(yaw: float, pitch: float) -> np.ndarray:
    """Rows: right, up, forward. ``yaw`` 0 looks along +x and grows to the left; positive ``pitch`` looks down."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    fwd = np.array([cp * cy, cp * sy, -sp])
    right = np.array([sy, -cy, 0.0])
    up = np.cross(right, fwd)
    return np.stack([right, up, fwd])


def view_projection(eye: tuple, yaw: float, pitch: float, fov: float, aspect: float) -> np.ndarray:
    M = basis(yaw, pitch)  # noqa: N806
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = M[0], M[1], -M[2]
    view[:3, 3] = -(view[:3, :3] @ np.array(eye, float))
    near, far = 6.0, 260000.0
    t = 1.0 / math.tan(math.radians(fov) / 2)
    proj = np.zeros((4, 4))
    proj[0, 0] = t / aspect
    proj[1, 1] = -t  # upside down (see the module docstring)
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2 * far * near / (near - far)
    proj[3, 2] = -1.0
    return proj @ view


def _mat(m: np.ndarray) -> QMatrix4x4:
    return QMatrix4x4(*[float(v) for v in m.reshape(-1)])


class _Mesh:
    def __init__(self) -> None:
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.count = 0
        self.ranges: list[tuple[int, int]] = []

    def destroy(self) -> None:
        self.vao.destroy()
        self.vbo.destroy()


class SurfRenderer:
    """Draws a surf map from a camera into a QImage; everything OpenGL is made the first time it draws."""

    def __init__(self) -> None:
        self.ctx: QOpenGLContext | None = None
        self.surface: QOffscreenSurface | None = None
        self.f: QOpenGLExtraFunctions | None = None
        self.fbo: QOpenGLFramebufferObject | None = None
        self.resolve: QOpenGLFramebufferObject | None = None
        self.samples = 4
        self.meshes: dict[str, tuple[_Mesh, _Mesh]] = {}
        self.error = ""
        self.gpu = ""
        self.ms = 0.0
        self._loc: dict = {}

    # --- set-up -------------------------------------------------------------------------------------------------

    def available(self) -> bool:
        """True when OpenGL 3.3 works here (tried once; ``error`` says why not)."""
        if self.ctx is None and not self.error:
            try:
                self._init()
                self.ctx.doneCurrent()
            except Exception as e:  # noqa: BLE001 - no OpenGL: the surf mode says so instead
                self.error = str(e) or "OpenGL 3.3 is not available"
                log.warning("Case Clash surf view unavailable: %s", self.error)
        return self.ctx is not None

    def _init(self) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            raise RuntimeError("Surf needs OpenGL 3.3 (no graphics context could be made)")
        surface = QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()
        if not ctx.makeCurrent(surface):
            raise RuntimeError("Surf could not use its graphics context")
        f = QOpenGLExtraFunctions(ctx)
        f.initializeOpenGLFunctions()
        self.p_world = self._program(_WORLD_VS, _WORLD_FS)
        self.p_sky = self._program(_SKY_VS, _SKY_FS)
        self.p_zone = self._program(_ZONE_VS, _ZONE_FS)
        self.sky_vao = QOpenGLVertexArrayObject()
        self.sky_vao.create()
        self.ctx, self.surface, self.f = ctx, surface, f
        self.gpu = str(f.glGetString(0x1F01) or "")

    def _program(self, vs: str, fs: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs) and ok
        if not (ok and prog.link()):
            raise RuntimeError(f"Surf shader did not compile: {prog.log()}")
        return prog

    def _u(self, prog: QOpenGLShaderProgram, name: str) -> int:
        key = (id(prog), name)
        loc = self._loc.get(key)
        if loc is None:
            loc = self._loc[key] = prog.uniformLocation(name)
        return loc

    def _buffer(self, data: np.ndarray, attrs: list[tuple[int, int, int]], stride: int) -> _Mesh:
        f = self.f
        m = _Mesh()
        m.vao.create()
        m.vao.bind()
        m.vbo.create()
        m.vbo.bind()
        raw = np.ascontiguousarray(data, F32).tobytes()
        m.vbo.allocate(raw, len(raw))
        for loc, offset, size in attrs:
            f.glEnableVertexAttribArray(loc)
            self.p_world.setAttributeBuffer(loc, GL_FLOAT, offset, size, stride)
        m.vao.release()
        m.vbo.release()
        return m

    def _meshes(self, m: SM.SurfMap) -> tuple[_Mesh, _Mesh]:
        got = self.meshes.get(m.id)
        if got is None:
            world = self._buffer(m.mesh, [(0, 0, 3), (1, 12, 3), (2, 24, 2), (3, 32, 2)], SM.VERTEX_FLOATS * 4)
            world.count = len(m.mesh)
            zones = self._buffer(np.concatenate(m.zones_mesh), [(0, 0, 4)], 16)
            first = 0
            for z in m.zones_mesh:
                zones.ranges.append((first, len(z)))
                first += len(z)
            got = self.meshes[m.id] = (world, zones)
        return got

    def close(self) -> None:
        """Give everything back to the graphics card."""
        if self.ctx is None:
            return
        try:
            if self.ctx.makeCurrent(self.surface):
                for a, b in self.meshes.values():
                    a.destroy()
                    b.destroy()
                self.sky_vao.destroy()
                self.fbo = self.resolve = None
                self.ctx.doneCurrent()
        except Exception:  # noqa: BLE001 - closing must never fail
            log.debug("surf renderer close", exc_info=True)
        self.meshes.clear()
        self.ctx = None
        self.surface = None

    def _size(self, w: int, h: int, samples: int) -> None:
        if self.fbo is not None and self.fbo.width() == w and self.fbo.height() == h and self.samples == samples:
            return
        self.samples = samples
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
        fmt.setInternalTextureFormat(GL_RGB8)
        fmt.setSamples(samples)
        self.fbo = QOpenGLFramebufferObject(w, h, fmt)
        if samples:
            fmt2 = QOpenGLFramebufferObjectFormat()
            fmt2.setInternalTextureFormat(GL_RGB8)
            self.resolve = QOpenGLFramebufferObject(w, h, fmt2)
        else:
            self.resolve = None

    # --- the frame ----------------------------------------------------------------------------------------------

    def render(self, m: SM.SurfMap, eye: tuple, yaw: float, pitch: float, fov: float, w: int, h: int, t: float,
               zones: list[tuple] = (), samples: int = 4) -> QImage:
        """One frame. ``zones``: (index into the map's zones, r, g, b, alpha) of the zones to draw."""
        t0 = time.perf_counter()
        if self.ctx is None:
            self._init()
        elif not self.ctx.makeCurrent(self.surface):
            raise RuntimeError("Surf lost its graphics context")
        f = self.f
        w, h = max(int(w), 16), max(int(h), 16)
        self._size(w, h, samples)
        world, zmesh = self._meshes(m)
        th = m.theme
        vp = view_projection(eye, yaw, pitch, fov, w / h)
        M = basis(yaw, pitch)  # noqa: N806
        sun = np.array(th.sun, float)
        sun /= np.linalg.norm(sun)
        style = STYLE.get(m.theme_id, 0)

        def sky_uniforms(prog: QOpenGLShaderProgram) -> None:
            f.glUniform3f(self._u(prog, "u_top"), *th.sky_top)
            f.glUniform3f(self._u(prog, "u_hor"), *th.sky_horizon)
            f.glUniform3f(self._u(prog, "u_sea"), *th.sea)
            f.glUniform3f(self._u(prog, "u_sun"), *(float(c) for c in sun))
            f.glUniform3f(self._u(prog, "u_suncol"), *th.sun_color)
            prog.setUniformValue1i(self._u(prog, "u_style"), style)

        self.fbo.bind()
        f.glViewport(0, 0, w, h)
        f.glDisable(GL_BLEND)
        f.glDisable(GL_CULL_FACE)
        f.glDisable(GL_DEPTH_TEST)
        f.glDepthMask(True)
        f.glClear(GL_DEPTH_BUFFER_BIT)
        # sky
        prog = self.p_sky
        prog.bind()
        sky_uniforms(prog)
        tan = math.tan(math.radians(fov) / 2)
        r, u = M[0] * (tan * w / h), M[1] * tan
        f.glUniform3f(self._u(prog, "u_r"), *(float(c) for c in r))
        f.glUniform3f(self._u(prog, "u_u"), *(float(c) for c in u))
        f.glUniform3f(self._u(prog, "u_f"), *(float(c) for c in M[2]))
        f.glUniform1f(self._u(prog, "u_time"), float(t))
        f.glUniform1f(self._u(prog, "u_stars"), float(th.stars))
        f.glUniform1f(self._u(prog, "u_clouds"), float(th.clouds))
        f.glUniform3f(self._u(prog, "u_cloudcol"), *th.cloud_color)
        self.sky_vao.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.sky_vao.release()
        # the map
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LESS)
        prog = self.p_world
        prog.bind()
        sky_uniforms(prog)
        prog.setUniformValue(self._u(prog, "u_vp"), _mat(vp))
        f.glUniform3f(self._u(prog, "u_eye"), *(float(c) for c in eye))
        for name, value in (("u_face", th.face), ("u_trim", th.trim), ("u_under", th.under), ("u_plata", th.plat_a),
                            ("u_platb", th.plat_b), ("u_pillar", th.pillar), ("u_decor", th.decor)):
            f.glUniform3f(self._u(prog, name), *value)
        f.glUniform1f(self._u(prog, "u_amb"), float(th.ambient))
        f.glUniform1f(self._u(prog, "u_fog"), float(th.fog))
        f.glUniform1f(self._u(prog, "u_grid"), float(th.grid))
        f.glUniform1f(self._u(prog, "u_glow"), float(th.glow))
        f.glUniform1f(self._u(prog, "u_time"), float(t))
        world.vao.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, world.count)
        world.vao.release()
        # the zones: glowing see-through walls, added on top
        if zones:
            prog = self.p_zone
            prog.bind()
            prog.setUniformValue(self._u(prog, "u_vp"), _mat(vp))
            f.glEnable(GL_BLEND)
            f.glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
            f.glDepthMask(False)
            zmesh.vao.bind()
            for k, cr, cg, cb, ca in zones:
                if 0 <= k < len(zmesh.ranges):
                    first, count = zmesh.ranges[k]
                    f.glUniform4f(self._u(prog, "u_col"), cr, cg, cb, ca)
                    f.glDrawArrays(GL_TRIANGLES, first, count)
            zmesh.vao.release()
            f.glDepthMask(True)
            f.glDisable(GL_BLEND)
        self.fbo.release()
        source = self.fbo
        if self.resolve is not None:
            QOpenGLFramebufferObject.blitFramebuffer(self.resolve, self.fbo)
            source = self.resolve
        img = source.toImage(False)
        self.ctx.doneCurrent()
        self.ms = (time.perf_counter() - t0) * 1000
        return img
