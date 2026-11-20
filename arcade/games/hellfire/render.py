"""The 3D view on the graphics card: OpenGL 3.3 through PySide6's own classes (an off-screen context and frame
buffer), drawn at the window's own resolution and read back as a QImage the game paints.

The level is a grid of 64-unit cells with their own floor and ceiling heights (``level.Grid``). Every open cell
gives a floor and a ceiling quad, and each of its four sides one wall instance that the vertex shader turns into
the lower step, the upper wall (doors are upper walls that rise), the full wall, or the sky curtain above a wall
under an open sky, from the heights of the two cells. Things are camera-facing billboards out of one sprite atlas;
the weapon is a screen quad. Lighting follows the 1993 game: the cell's light level, darker with distance, in 32
steps, with a little contrast between east-west and north-south walls.

The picture is drawn upside down (the projection flips y) so that ``toImage(False)`` needs no flip."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from PySide6.QtGui import QImage, QMatrix4x4, QOffscreenSurface, QOpenGLContext, QOpenGLExtraFunctions, QSurfaceFormat
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
                              QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture, QOpenGLVertexArrayObject)

from .assets import sky, sprites, tiles
from .level import CELL

log = logging.getLogger(__name__)

GL_FLOAT = 0x1406
GL_TRIANGLE_STRIP = 0x0005
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_CULL_FACE = 0x0B44
GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x0100
GL_LEQUAL = 0x0203
GL_RGB8 = 0x8051
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303

TAN_Y = 0.75  # vertical half field of view (90 degrees across a 4:3 picture; wider windows see more at the sides)
NEAR, FAR = 2.0, 12000.0

_COMMON = """
uniform vec3 u_eye;
uniform vec2 u_fwd;
uniform float u_extra;
uniform vec4 u_tint;
uniform float u_invuln;
uniform vec2 u_res;
uniform float u_yaw;
uniform vec2 u_tan;
uniform float u_time;
uniform sampler2D u_sky;

float lit(float light, float dist) {
    float l = clamp(light + u_extra, 0.0, 255.0);
    float startmap = (15.0 - floor(l / 16.0)) * 4.0;
    float cm = clamp(startmap - 1280.0 / max(dist, 1.0), 0.0, 31.0);
    return 1.0 - floor(cm) * (0.84 / 31.0);
}
vec3 post(vec3 c) {
    if (u_invuln > 0.5) {
        float g = dot(c, vec3(0.30, 0.59, 0.11));
        c = vec3(1.0 - g);
    }
    return mix(c, u_tint.rgb, u_tint.a);
}
vec3 skycol() {
    vec2 ndc = vec2(gl_FragCoord.x / u_res.x * 2.0 - 1.0, 1.0 - gl_FragCoord.y / u_res.y * 2.0);
    float yaw = u_yaw + atan(ndc.x * u_tan.x);
    float v = 0.70 - ndc.y * u_tan.y * 0.52;
    return texture(u_sky, vec2(yaw / 6.2831853, clamp(v, 0.002, 0.998))).rgb;
}
"""

_WALL_VS = """#version 330 core
layout(location = 0) in vec4 a_seg;   // x0 y0 x1 y1: the side, left to right as seen from the cell
layout(location = 1) in vec4 a_h;     // floor, ceiling of this cell; floor, ceiling of the cell beyond
layout(location = 2) in vec4 a_tex;   // lower, upper, full layer; light
layout(location = 3) in vec4 a_flag;  // beyond is solid, upper is sky, this cell under sky / beyond not, contrast
uniform mat4 u_vp;
uniform int u_part;
out vec3 v_uvl;
out float v_light;
out float v_sky;
out vec3 v_world;
void main() {
    float s = float(gl_VertexID & 1);
    float t = float(gl_VertexID >> 1);
    bool solid = a_flag.x > 0.5;
    float bot, top, layer, anchor;
    float skyf = 0.0;
    if (u_part == 0) {        // the step up to the floor beyond
        bot = a_h.x; top = solid ? a_h.x : a_h.z; layer = a_tex.x; anchor = a_h.z;
    } else if (u_part == 1) { // the wall down to the ceiling beyond (doors)
        bot = solid ? a_h.y : a_h.w; top = a_h.y; layer = a_tex.y; anchor = a_h.w + 128.0; skyf = a_flag.y;
    } else if (u_part == 2) { // a solid wall
        bot = a_h.x; top = solid ? a_h.y : a_h.x; layer = a_tex.z; anchor = a_h.y;
    } else {                  // the sky above a wall that borders an open sky
        bot = a_h.y; top = a_flag.z > 0.5 ? a_h.y + 4096.0 : a_h.y; layer = 0.0; anchor = 0.0; skyf = 1.0;
    }
    top = max(top, bot);
    vec2 p = mix(a_seg.xy, a_seg.zw, s);
    float z = mix(bot, top, t);
    vec2 dir = normalize(a_seg.zw - a_seg.xy);
    v_uvl = vec3(dot(p, dir) / 128.0, (anchor - z) / 128.0, layer);
    v_light = a_tex.w + a_flag.w;
    v_sky = skyf;
    v_world = vec3(p, z);
    gl_Position = u_vp * vec4(p, z, 1.0);
}
"""

_FLAT_VS = """#version 330 core
layout(location = 0) in vec4 a_cell;  // corner x, y; height; light
layout(location = 1) in vec4 a_info;  // layer, sky
uniform mat4 u_vp;
out vec3 v_uvl;
out float v_light;
out float v_sky;
out vec3 v_world;
void main() {
    vec2 p = a_cell.xy + vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 64.0;
    v_uvl = vec3(p / 128.0, a_info.x);
    v_light = a_cell.w;
    v_sky = a_info.y;
    v_world = vec3(p, a_cell.z);
    gl_Position = u_vp * vec4(p, a_cell.z, 1.0);
}
"""

_SURF_FS = """#version 330 core
in vec3 v_uvl;
in float v_light;
in float v_sky;
in vec3 v_world;
uniform sampler2DArray u_tiles;
out vec4 o;
""" + _COMMON + """
void main() {
    if (v_sky > 0.5) {
        o = vec4(post(skycol()), 1.0);
        return;
    }
    vec3 c = texture(u_tiles, v_uvl).rgb;
    float d = dot(v_world.xy - u_eye.xy, u_fwd);
    o = vec4(post(c * lit(v_light, d)), 1.0);
}
"""

_SPRITE_VS = """#version 330 core
layout(location = 0) in vec4 a_pos;   // x, y, z of the feet; light
layout(location = 1) in vec4 a_uv;    // atlas corners
layout(location = 2) in vec4 a_size;  // width, height, centre column, bottom height
layout(location = 3) in vec4 a_flag;  // full bright, mirrored, shadow (see-through), 0
uniform mat4 u_vp;
uniform vec3 u_right;
out vec2 v_uv;
out float v_light;
out vec3 v_flag;
out vec3 v_world;
void main() {
    float s = float(gl_VertexID & 1);
    float t = float(gl_VertexID >> 1);
    bool flip = a_flag.y > 0.5;
    float xo = flip ? a_size.x - a_size.z : a_size.z;
    vec3 p = vec3(a_pos.xy, a_pos.z + a_size.w + t * a_size.y) + u_right * (s * a_size.x - xo);
    float uu = flip ? 1.0 - s : s;
    v_uv = vec2(mix(a_uv.x, a_uv.z, uu), mix(a_uv.w, a_uv.y, t));
    v_light = a_pos.w;
    v_flag = a_flag.xyz;
    v_world = p;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

_SPRITE_FS = """#version 330 core
in vec2 v_uv;
in float v_light;
in vec3 v_flag;
in vec3 v_world;
uniform sampler2D u_atlas;
out vec4 o;
""" + _COMMON + """
void main() {
    vec4 c = texture(u_atlas, v_uv);
    if (c.a < 0.5) discard;
    float d = dot(v_world.xy - u_eye.xy, u_fwd);
    vec3 col = c.rgb * (v_flag.x > 0.5 ? 1.0 : lit(v_light, d));
    if (v_flag.z > 0.5) {
        vec2 q = floor(gl_FragCoord.xy / 2.0);
        float n = fract(sin(dot(q + vec2(floor(u_time * 24.0), 0.0), vec2(12.9898, 78.233))) * 43758.5453);
        if (n < 0.5) discard;
        col = col * 0.12 + vec3(0.02);
    }
    o = vec4(post(col), 1.0);
}
"""

_OVER_VS = """#version 330 core
layout(location = 0) in vec4 a_rect;  // left, bottom, right, top (-1..1, y up)
layout(location = 1) in vec4 a_uv;
layout(location = 2) in vec4 a_k;     // brightness
out vec2 v_uv;
out float v_k;
void main() {
    float s = float(gl_VertexID & 1);
    float t = float(gl_VertexID >> 1);
    vec2 p = mix(a_rect.xy, a_rect.zw, vec2(s, t));
    v_uv = vec2(mix(a_uv.x, a_uv.z, s), mix(a_uv.w, a_uv.y, t));
    v_k = a_k.x;
    gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
}
"""

_OVER_FS = """#version 330 core
in vec2 v_uv;
in float v_k;
uniform sampler2D u_atlas;
out vec4 o;
""" + _COMMON + """
void main() {
    vec4 c = texture(u_atlas, v_uv);
    if (c.a < 0.5) discard;
    o = vec4(post(c.rgb * v_k), 1.0);
}
"""

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))  # the face index of level.Grid.face_tex


class _Mesh:
    __slots__ = ("count", "vao", "vbo")

    def __init__(self) -> None:
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.count = 0


class Renderer:
    """Draws a ``World`` into a QImage. OpenGL is set up the first time it draws (or by ``warm_up``)."""

    def __init__(self) -> None:
        self.ctx: QOpenGLContext | None = None
        self.surface: QOffscreenSurface | None = None
        self.f: QOpenGLExtraFunctions | None = None
        self.fbo: QOpenGLFramebufferObject | None = None
        self.resolve: QOpenGLFramebufferObject | None = None
        self.samples = 4
        self.gpu = ""
        self._grid_id = None
        self._sig = None
        self._sky_tex: dict[str, QOpenGLTexture] = {}
        self.ms = 0.0
        self.tiles = tiles()
        self.sprites = sprites()

    # --- set-up -------------------------------------------------------------------------------------------------

    def _init(self) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            raise RuntimeError("Hellfire needs OpenGL 3.3 (no graphics context could be made)")
        surface = QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()
        if not ctx.makeCurrent(surface):
            raise RuntimeError("Hellfire could not use its graphics context")
        self.ctx, self.surface = ctx, surface
        f = self.f = QOpenGLExtraFunctions(ctx)
        f.initializeOpenGLFunctions()
        self.gpu = str(f.glGetString(0x1F01) or "")
        self.p_wall = self._program(_WALL_VS, _SURF_FS)
        self.p_flat = self._program(_FLAT_VS, _SURF_FS)
        self.p_sprite = self._program(_SPRITE_VS, _SPRITE_FS)
        self.p_over = self._program(_OVER_VS, _OVER_FS)
        self.loc = {id(p): {} for p in (self.p_wall, self.p_flat, self.p_sprite, self.p_over)}
        # the wall and floor tiles: one array texture, uploaded in one call (mipmaps made by the card)
        T = self.tiles
        data = np.ascontiguousarray(np.stack(T.layers), np.uint8)
        n, size = data.shape[0], data.shape[1]
        tex = QOpenGLTexture(QOpenGLTexture.Target.Target2DArray)
        tex.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        tex.setSize(size, size)
        tex.setLayers(n)
        tex.setMipLevels(int(math.log2(size)) + 1)
        tex.allocateStorage(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8)
        self._upload_layers(tex, data)
        tex.generateMipMaps()
        tex.setMinMagFilters(QOpenGLTexture.Filter.LinearMipMapLinear, QOpenGLTexture.Filter.Nearest)
        tex.setWrapMode(QOpenGLTexture.WrapMode.Repeat)
        try:
            tex.setMaximumAnisotropy(8.0)
        except Exception:  # noqa: BLE001 - not every driver has it
            pass
        self.tex_tiles = tex
        # the sprite atlas
        S = self.sprites
        atlas = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        atlas.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        atlas.setSize(S.size[0], S.size[1])
        atlas.setMipLevels(3)
        atlas.allocateStorage(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8)
        atlas.setData(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8,
                      np.ascontiguousarray(S.atlas).tobytes())
        atlas.generateMipMaps()
        atlas.setMinMagFilters(QOpenGLTexture.Filter.LinearMipMapNearest, QOpenGLTexture.Filter.Nearest)
        atlas.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        self.tex_atlas = atlas
        self.walls = self._mesh([0, 1, 2, 3])
        self.flats = self._mesh([0, 1])
        self.things = self._mesh([0, 1, 2, 3])
        self.over = self._mesh([0, 1, 2])

    @staticmethod
    def _upload_layers(tex: QOpenGLTexture, data: np.ndarray) -> None:
        """Every layer of mip level 0 in one call (calling setData once per layer, many times, is not safe)."""
        raw = data.tobytes()
        try:
            tex.setData(0, 0, data.shape[0], QOpenGLTexture.CubeMapFace.CubeMapPositiveX,
                        QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8, raw)
            return
        except TypeError:
            pass
        for layer in range(data.shape[0]):  # older bindings: one call per layer (a hundred or so, once)
            tex.setData(0, layer, QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8,
                        data[layer].tobytes())

    def warm_up(self) -> None:
        if self.ctx is None:
            self._init()
            self.ctx.doneCurrent()

    def prepare(self, sky_theme: str) -> None:
        """Make everything a map needs before it is played (so the first frames do not stall)."""
        if self.ctx is None:
            self._init()
        elif not self.ctx.makeCurrent(self.surface):
            return
        self._sky(sky_theme)
        self.ctx.doneCurrent()

    def _program(self, vs: str, fs: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs) and ok
        if not (ok and prog.link()):
            raise RuntimeError(f"Hellfire shader did not compile: {prog.log()}")
        return prog

    def _mesh(self, locs: list[int]) -> _Mesh:
        f = self.f
        m = _Mesh()
        m.vao.create()
        m.vao.bind()
        m.vbo.create()
        m.vbo.bind()
        stride = len(locs) * 16
        for loc in locs:
            f.glEnableVertexAttribArray(loc)
            self.p_wall.setAttributeBuffer(loc, GL_FLOAT, loc * 16, 4, stride)
            f.glVertexAttribDivisor(loc, 1)
        m.vao.release()
        m.vbo.release()
        return m

    @staticmethod
    def _fill(mesh: _Mesh, data: np.ndarray) -> None:
        raw = np.ascontiguousarray(data, np.float32).tobytes()
        mesh.vbo.bind()
        mesh.vbo.allocate(raw, len(raw))
        mesh.vbo.release()
        mesh.count = len(data)

    def _u(self, prog: QOpenGLShaderProgram, name: str) -> int:
        cache = self.loc[id(prog)]
        loc = cache.get(name)
        if loc is None:
            loc = cache[name] = prog.uniformLocation(name)
        return loc

    def _sky(self, theme: str) -> QOpenGLTexture:
        tex = self._sky_tex.get(theme)
        if tex is None:
            img = sky(theme)
            tex = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            tex.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
            tex.setSize(img.shape[1], img.shape[0])
            tex.setMipLevels(1)
            tex.allocateStorage(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8)
            tex.setData(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8,
                        np.ascontiguousarray(img).tobytes())
            tex.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
            tex.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionS, QOpenGLTexture.WrapMode.Repeat)
            tex.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionT, QOpenGLTexture.WrapMode.ClampToEdge)
            self._sky_tex[theme] = tex
        return tex

    def close(self) -> None:
        """Give everything back to the graphics card."""
        if self.ctx is None:
            return
        try:
            if self.ctx.makeCurrent(self.surface):
                for m in (self.walls, self.flats, self.things, self.over):
                    m.vao.destroy()
                    m.vbo.destroy()
                self.tex_tiles.destroy()
                self.tex_atlas.destroy()
                for t in self._sky_tex.values():
                    t.destroy()
                self._sky_tex.clear()
                self.fbo = self.resolve = None
                self.ctx.doneCurrent()
        except Exception:  # noqa: BLE001 - closing must never fail
            log.debug("Hellfire renderer close", exc_info=True)
        self.surface = None
        self.ctx = None
        self._grid_id = self._sig = None

    def _size(self, w: int, h: int) -> None:
        if self.fbo is not None and self.fbo.width() == w and self.fbo.height() == h:
            return
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
        fmt.setInternalTextureFormat(GL_RGB8)
        fmt.setSamples(self.samples)
        self.fbo = QOpenGLFramebufferObject(w, h, fmt)
        if self.samples:
            fmt2 = QOpenGLFramebufferObjectFormat()
            fmt2.setInternalTextureFormat(GL_RGB8)
            self.resolve = QOpenGLFramebufferObject(w, h, fmt2)
        else:
            self.resolve = None

    # --- the level's surfaces -----------------------------------------------------------------------------------

    def _build_static(self, g) -> None:  # noqa: ANN001 - a level.Grid
        """Which cell pairs make the wall instances and where their sides are (the parts that never change)."""
        open_ = np.flatnonzero(~g.solid)
        cy, cx = np.divmod(open_, g.w)
        a_l, b_l, d_l, seg_l, oob_l = [], [], [], [], []
        for d, (dx, dy) in enumerate(_DIRS):
            nx, ny = cx + dx, cy + dy
            inb = (nx >= 0) & (nx < g.w) & (ny >= 0) & (ny < g.h)
            b = np.where(inb, ny * g.w + nx, open_)
            x0 = cx * CELL
            y0 = cy * CELL
            if d == 0:
                seg = (x0 + CELL, y0, x0 + CELL, y0 + CELL)
            elif d == 1:
                seg = (x0, y0 + CELL, x0, y0)
            elif d == 2:
                seg = (x0 + CELL, y0 + CELL, x0, y0 + CELL)
            else:
                seg = (x0, y0, x0 + CELL, y0)
            a_l.append(open_)
            b_l.append(b)
            d_l.append(np.full(len(open_), d))
            seg_l.append(np.stack(seg, axis=1))
            oob_l.append(~inb)
        self.wa = np.concatenate(a_l)
        self.wb = np.concatenate(b_l)
        self.wd = np.concatenate(d_l)
        self.wseg = np.concatenate(seg_l).astype(np.float32)
        self.woob = np.concatenate(oob_l)
        self.wcontrast = np.where(self.wd < 2, 16.0, -16.0).astype(np.float32)
        self.fopen = open_
        self.fxy = np.stack([cx * CELL, cy * CELL], axis=1).astype(np.float32)

    def _build_dynamic(self, g, tick: int) -> None:  # noqa: ANN001
        layer = self.tiles.layer_of(tick)
        a, b = self.wa, self.wb
        solid = self.woob | g.solid[b]
        mid_id = np.where(self.woob, g.upper_tex[a], g.face_tex[self.wd, b])
        n = len(a)
        w = np.empty((n, 16), np.float32)
        w[:, 0:4] = self.wseg
        w[:, 4] = g.fh[a]
        w[:, 5] = g.ch[a]
        w[:, 6] = np.where(solid, g.fh[a], g.fh[b])
        w[:, 7] = np.where(solid, g.ch[a], g.ch[b])
        w[:, 8] = layer[g.lower_tex[b]]
        w[:, 9] = layer[g.upper_tex[b]]
        w[:, 10] = layer[mid_id]
        w[:, 11] = g.light[a]
        w[:, 12] = solid
        w[:, 13] = g.sky[a] & g.sky[b] & ~solid
        w[:, 14] = g.sky[a] & (solid | ~g.sky[b])
        w[:, 15] = self.wcontrast
        self._fill(self.walls, w)
        c = self.fopen
        m = len(c)
        fl = np.empty((2 * m, 8), np.float32)
        fl[:m, 0:2] = self.fxy
        fl[:m, 2] = g.fh[c]
        fl[:m, 3] = g.light[c]
        fl[:m, 4] = layer[g.ftex[c]]
        fl[:m, 5] = 0.0
        fl[m:, 0:2] = self.fxy
        fl[m:, 2] = g.ch[c]
        fl[m:, 3] = g.light[c]
        fl[m:, 4] = layer[g.ctex[c]]
        fl[m:, 5] = g.sky[c]
        fl[:, 6:8] = 0.0
        self._fill(self.flats, fl)

    # --- the frame ----------------------------------------------------------------------------------------------

    def render(self, grid, eye: tuple[float, float, float], angle: float, w: int, h: int, tick: int,  # noqa: ANN001
               things: list, weapon: list, sky_theme: str = "dusk", extra: float = 0.0,
               tint: tuple = (0.0, 0.0, 0.0, 0.0), invuln: bool = False, t: float = 0.0) -> QImage:
        """One frame. ``things``: (x, y, z, sprite id, mirrored, full bright, light, see-through);
        ``weapon``: (sprite id, left, top, brightness) on the 320 x 200 screen."""
        t0 = time.perf_counter()
        if self.ctx is None:
            self._init()
        elif not self.ctx.makeCurrent(self.surface):
            raise RuntimeError("Hellfire lost its graphics context")
        f = self.f
        w, h = max(int(w), 16), max(int(h), 16)
        self._size(w, h)
        if self._grid_id != id(grid):
            self._build_static(grid)
            self._grid_id = id(grid)
            self._sig = None
        sig = hash((grid.fh.tobytes(), grid.ch.tobytes(), grid.light.tobytes(), grid.face_tex.tobytes(),
                    grid.upper_tex.tobytes(), int(tick) // 8))
        if sig != self._sig:
            self._build_dynamic(grid, tick)
            self._sig = sig
        ex, ey, ez = (float(v) for v in eye)
        ca, sa = math.cos(angle), math.sin(angle)
        tan_y = TAN_Y
        tan_x = tan_y * w / h
        view = np.array([[-sa, ca, 0, 0], [0, 0, 1, 0], [-ca, -sa, 0, 0], [0, 0, 0, 1]], np.float64)
        view[:3, 3] = -(view[:3, :3] @ np.array([ex, ey, ez]))
        proj = np.zeros((4, 4))
        proj[0, 0] = 1.0 / tan_x
        proj[1, 1] = -1.0 / tan_y  # upside down: toImage(False) then needs no flip
        proj[2, 2] = (FAR + NEAR) / (NEAR - FAR)
        proj[2, 3] = 2 * FAR * NEAR / (NEAR - FAR)
        proj[3, 2] = -1.0
        vp = QMatrix4x4(*[float(v) for v in (proj @ view).reshape(-1)])
        sky_tex = self._sky(sky_theme)

        self.fbo.bind()
        f.glViewport(0, 0, w, h)
        f.glDisable(GL_BLEND)
        f.glDisable(GL_CULL_FACE)
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LEQUAL)
        f.glDepthMask(True)
        f.glClearColor(0.0, 0.0, 0.0, 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.tex_tiles.bind(0)
        self.tex_atlas.bind(1)
        sky_tex.bind(2)

        def common(prog: QOpenGLShaderProgram) -> None:
            prog.bind()
            u = self._u
            if u(prog, "u_vp") >= 0:
                prog.setUniformValue(u(prog, "u_vp"), vp)
            f.glUniform3f(u(prog, "u_eye"), ex, ey, ez)
            f.glUniform2f(u(prog, "u_fwd"), ca, sa)
            f.glUniform1f(u(prog, "u_extra"), float(extra))
            f.glUniform4f(u(prog, "u_tint"), *(float(v) for v in tint))
            f.glUniform1f(u(prog, "u_invuln"), 1.0 if invuln else 0.0)
            f.glUniform2f(u(prog, "u_res"), float(w), float(h))
            f.glUniform1f(u(prog, "u_yaw"), float(angle))
            f.glUniform2f(u(prog, "u_tan"), tan_x, tan_y)
            f.glUniform1f(u(prog, "u_time"), float(t))
            prog.setUniformValue1i(u(prog, "u_sky"), 2)

        # floors and ceilings
        prog = self.p_flat
        common(prog)
        prog.setUniformValue1i(self._u(prog, "u_tiles"), 0)
        if self.flats.count:
            self.flats.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, self.flats.count)
            self.flats.vao.release()
        # walls: steps, upper walls, full walls, sky curtains
        prog = self.p_wall
        common(prog)
        prog.setUniformValue1i(self._u(prog, "u_tiles"), 0)
        if self.walls.count:
            self.walls.vao.bind()
            for part in range(4):
                prog.setUniformValue1i(self._u(prog, "u_part"), part)
                f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, self.walls.count)
            self.walls.vao.release()
        # things
        if things:
            prog = self.p_sprite
            common(prog)
            prog.setUniformValue1i(self._u(prog, "u_atlas"), 1)
            f.glUniform3f(self._u(prog, "u_right"), -sa, ca, 0.0)
            self._fill(self.things, self._thing_instances(things))
            self.things.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, self.things.count)
            self.things.vao.release()
        # the weapon
        if weapon:
            f.glDisable(GL_DEPTH_TEST)
            prog = self.p_over
            common(prog)
            prog.setUniformValue1i(self._u(prog, "u_atlas"), 1)
            self._fill(self.over, self._weapon_instances(weapon, w, h))
            self.over.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, self.over.count)
            self.over.vao.release()
            f.glEnable(GL_DEPTH_TEST)
        self.fbo.release()
        source = self.fbo
        if self.resolve is not None:
            QOpenGLFramebufferObject.blitFramebuffer(self.resolve, self.fbo)
            source = self.resolve
        img = source.toImage(False)
        self.ctx.doneCurrent()
        self.ms = (time.perf_counter() - t0) * 1000
        return img

    def _thing_instances(self, things: list) -> np.ndarray:
        arr = np.array(things, np.float32).reshape(-1, 8)
        ids = arr[:, 3].astype(np.int32)
        r = self.sprites.rect[ids]
        out = np.zeros((len(arr), 16), np.float32)
        out[:, 0:3] = arr[:, 0:3]
        out[:, 3] = arr[:, 6]
        out[:, 4:8] = r[:, 0:4]
        out[:, 8:12] = r[:, 4:8]
        out[:, 12] = arr[:, 5]
        out[:, 13] = arr[:, 4]
        out[:, 14] = arr[:, 7]
        return out

    def _weapon_instances(self, weapon: list, w: int, h: int) -> np.ndarray:
        scale = h / 200.0
        out = np.zeros((len(weapon), 12), np.float32)
        for i, (sid, left, top, bright) in enumerate(weapon):
            r = self.sprites.rect[int(sid)]
            x0 = w / 2 + (left - 160.0) * scale
            y0 = top * scale
            x1 = x0 + r[4] * scale
            y1 = y0 + r[5] * scale
            out[i, 0:4] = (x0 / w * 2 - 1, 1 - y1 / h * 2, x1 / w * 2 - 1, 1 - y0 / h * 2)
            out[i, 4:8] = r[0:4]
            out[i, 8] = bright
        return out
