"""BlockCraft's renderer: the world drawn by the graphics card (OpenGL 3.3 through Qt, no other package), at the
window's own resolution, into an image the game widget paints.

Every chunk is one buffer of *face instances* (mesher.py) drawn in a single call; the texture atlas is an array
texture sampled nearest-pixel up close (crisp texels at any size) and mip-mapped far away (no shimmer). Sky, sun,
moon, stars and clouds are one full-screen pass; animals, dropped items, falling blocks and particles are instanced
boxes and quads; water is blended last. The picture is drawn upside down on purpose so it can be read back
without a flip."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from PySide6.QtGui import (QImage, QMatrix4x4, QOffscreenSurface, QOpenGLContext, QOpenGLExtraFunctions,
                           QSurfaceFormat)
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
                              QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture, QOpenGLVertexArrayObject)

from app.arcade.games.blockcraft import blocks as B
from app.arcade.games.blockcraft.mesher import FACE_BYTES, SHAPES
from app.arcade.games.blockcraft.worldgen import HEIGHT

log = logging.getLogger(__name__)
F32 = np.float32
WARM = np.array([1.0, 0.88, 0.66], F32)  # torch light tint
CLOUD_Y = 168.0
BOX_FLOATS = 28  # an entity box: base + yaw, pivot + angle, offset + roll, half + whiten, tint, 6 texture layers
# OpenGL names (Qt gives the functions, not the constants)
GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_UNSIGNED_BYTE, GL_FLOAT = 0x0004, 0x0005, 0x1401, 0x1406
GL_DEPTH_TEST, GL_CULL_FACE, GL_BLEND, GL_POLYGON_OFFSET_FILL = 0x0B71, 0x0B44, 0x0BE2, 0x8037
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_NEAREST = 0x4000, 0x0100, 0x2600
GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_CW, GL_LEQUAL, GL_LESS, GL_RGB8 = 0x0302, 0x0303, 0x0900, 0x0203, 0x0201, 0x8051

_GAMMA = [0.5]  # the Brightness option: 0 moody .. 1 bright


def set_gamma(gamma: float) -> None:
    _GAMMA[0] = float(min(max(gamma, 0.0), 1.0))


def brightness(level: float) -> float:
    """How bright a light level 0..15 is drawn (the same curve as the block shader), lifted by Brightness."""
    f = min(max(level / 15.0, 0.0), 1.0)
    b = f / (4.0 - 3.0 * f)
    g = _GAMMA[0]
    if g:
        inv = (1.0 - b) ** 2
        b += (1.0 - inv * inv - b) * g
    return b * 0.96 + 0.04


def basis(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """Rows: right, up, forward (yaw 0 faces +z / south; positive pitch looks down)."""
    y, p = math.radians(yaw), math.radians(pitch)
    fwd = np.array([-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])
    right = np.array([-math.cos(y), 0.0, -math.sin(y)])
    up = np.cross(right, fwd)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        right, up = right * c + up * s, up * c - right * s
    return np.stack([right, up, fwd]).astype(F32)


class Env:
    """What the sky looks like right now (made by the game from the time of day)."""

    def __init__(self, time_of_day: float = 0.25, t: float = 0.0, realm: str = "over") -> None:
        self.time = t  # seconds (water animation, clouds)
        self.realm = realm  # "over", "deep" (no sky: a red murk) or "isles" (a starlit dusk that never ends)
        self.set_time(time_of_day)
        self.underwater = False
        self.in_lava = False

    def set_time(self, tod: float) -> None:
        """``tod``: 0 sunrise, 0.25 noon, 0.5 sunset, 0.75 midnight."""
        self.tod = tod % 1.0
        a = self.tod * 2 * math.pi
        self.sun = np.array([math.cos(a), math.sin(a), 0.0], F32)
        day = min(max(math.sin(a) * 2.2 + 0.45, 0.0), 1.0)
        self.day = day
        self.daylight = F32(0.27 + 0.73 * day)  # how strongly sky light lights blocks
        top_day, top_night = np.array([98, 150, 255], F32), np.array([3, 5, 14], F32)
        fog_day, fog_night = np.array([186, 212, 255], F32), np.array([9, 11, 22], F32)
        self.sky_top = top_night + (top_day - top_night) * day
        self.fog = fog_night + (fog_day - fog_night) * day
        self.sunset = max(0.0, 1.0 - abs(math.sin(a)) / 0.38)
        self.stars = max(0.0, 1.0 - day * 2.2)
        self.ambient = 0.0  # the least light anything gets
        self.flat_sky = False  # one colour all round instead of a sky
        self.fog_range: tuple | None = None
        self.clouds = True
        if self.realm == "deep":
            self.day, self.daylight, self.sunset, self.stars = 0.0, F32(0.0), 0.0, 0.0
            self.sky_top = self.fog = np.array([52, 15, 8], F32)
            self.ambient, self.flat_sky, self.fog_range, self.clouds = 0.2, True, (5.0, 88.0), False
        elif self.realm == "isles":
            self.day, self.daylight, self.sunset, self.stars = 0.7, F32(0.82), 0.0, 0.9
            self.sky_top, self.fog = np.array([14, 18, 54], F32), np.array([70, 112, 152], F32)
            self.ambient, self.clouds = 0.13, False


# --- the texture layers ----------------------------------------------------------------------------------------------

class Layers:
    """Every 16 x 16 picture the 3D view uses, as the layers of one array texture: the block atlas, the crack
    stages, the item sprites and the faces of the animal models."""

    def __init__(self) -> None:
        from app.arcade.games.blockcraft import entities as E

        imgs = [B.ATLAS.astype(np.uint8)]
        n = len(B.ATLAS)
        cracks = np.zeros((10, 16, 16, 4), np.uint8)
        cracks[B.CRACKS] = (0, 0, 0, 96)
        if B.CRACK_RGBA is not None:  # the texture pack's
            cracks = B.CRACK_RGBA.copy()
            cracks[..., :3] = cracks[..., :3] // 3
            cracks[..., 3] = (cracks[..., 3] * 0.62).astype(np.uint8)
        self.crack0 = n
        imgs.append(cracks)
        n += 10
        blank = np.zeros((1, 16, 16, 4), np.uint8)
        self.blank = n
        imgs.append(blank)
        n += 1
        self.sprite: dict[int, int] = {}
        for item in sorted(B.NAMES):
            if not B.flat_item(item):
                continue
            if B.is_block(item):
                self.sprite[item] = int(B.FACE_TEX[item, 4])
            else:
                self.sprite[item] = n
                imgs.append(B.item_sprite(item)[None].astype(np.uint8))
                n += 1
        self.models: dict[str, np.ndarray] = {}  # kind -> box templates (parts, BOX_FLOATS)
        self.part_kinds: dict[str, np.ndarray] = {}
        seen: dict[int, int] = {}
        codes = {None: 0, "a": 1, "b": 2, "head": 3, "arm": 4}
        for kind, parts in E.MODELS.items():
            tmpl = np.zeros((len(parts), BOX_FLOATS), F32)
            for i, (size, pivot, offset, part, tex) in enumerate(parts):
                base = seen.get(id(tex))
                if base is None:
                    t = np.clip(np.asarray(tex), 0, 255).astype(np.uint8)
                    k = 16 // t.shape[1]
                    t = np.repeat(np.repeat(t, k, 1), k, 2)
                    rgba = np.full((6, 16, 16, 4), 255, np.uint8)
                    rgba[..., :t.shape[-1]] = t
                    base = seen[id(tex)] = n
                    imgs.append(rgba)
                    n += 6
                tmpl[i, 4:7] = np.array(pivot, F32) / 16
                tmpl[i, 8:11] = np.array(offset, F32) / 16
                tmpl[i, 12:15] = np.array(size, F32) / 32
                tmpl[i, 19] = 1.0
                tmpl[i, 20:26] = base + np.arange(6)
            self.models[kind] = tmpl
            self.part_kinds[kind] = np.array([codes[p[3]] for p in parts], np.intp)
        self.images = np.concatenate(imgs)
        self.count = n

    def mips(self) -> list[np.ndarray]:
        """The images at 16, 8, 4, 2 and 1 pixels (colour averaged over the solid texels only)."""
        a = self.images[..., 3:4].astype(F32) / 255
        rgb = self.images[..., :3].astype(F32) * a
        out = [self.images]
        size = 16
        while size > 1:
            size //= 2
            sh = (len(a), size, 2, size, 2)
            rgb = rgb.reshape(*sh, 3).sum((2, 4))
            a = a.reshape(*sh, 1).sum((2, 4))
            col = rgb / np.maximum(a, 1e-6)
            level = np.concatenate([col, a / 4 * 255], -1)
            out.append(np.clip(level + 0.5, 0, 255).astype(np.uint8))
            rgb, a = rgb / 4, a / 4
        return out


_layers: Layers | None = None


def layers() -> Layers:
    global _layers  # noqa: PLW0603 - built once, the first time the game draws
    if _layers is None:
        _layers = Layers()
    return _layers


# --- instances of the moving things ----------------------------------------------------------------------------------

def mob_boxes(m, a: float, light: np.ndarray, origin: tuple) -> np.ndarray:  # noqa: ANN001 - an entities.Mob
    """The boxes of an animal or zombie at tick fraction ``a``."""
    L = layers()
    if m.kind == "wyrm":  # a head and a tail of segments that follow where it has been
        parts = L.models["wyrm"]
        x, y, z = m.lerp(a)
        trail = m.trail
        pts = [(x, y, z)] + [trail[min(i * 3, len(trail) - 1)] if trail else (x, y, z) for i in range(1, 10)]
        inst = np.zeros((len(pts), BOX_FLOATS), F32)
        tint = light * np.array([1.0, 0.45, 0.45]) if (m.hurt > 0 or m.death > 0) else light
        for i, (px, py, pz) in enumerate(pts):
            row = parts[0 if i == 0 else 1].copy()
            row[0:3] = (px - origin[0], py, pz - origin[2])
            row[3] = math.radians(m.yaw) if i == 0 else math.atan2(-(pts[i - 1][0] - px), pts[i - 1][2] - pz)
            if i:
                row[12:15] *= max(0.35, 1.0 - i * 0.075)
            row[16:19] = tint
            inst[i] = row
        return inst
    inst = L.models[m.kind].copy()
    x, y, z = m.lerp(a)
    inst[:, 0], inst[:, 1], inst[:, 2] = x - origin[0], y, z - origin[2]
    inst[:, 3] = math.radians(m.yaw)
    sw = math.cos(m.swing * 0.6662) * 1.4 * m.swing_amount
    arm = -math.pi / 2 + math.sin(m.swing * 0.3) * 0.05 - m.attack_anim * 0.6
    inst[:, 7] = np.array([0.0, sw, -sw, m.head_pitch, arm], F32)[L.part_kinds[m.kind]]
    if m.death > 0:
        inst[:, 11] = min(m.death, 1.0) * math.pi / 2
    tint = light
    if m.hurt > 0 or m.death > 0:
        tint = tint * np.array([1.0, 0.45, 0.45])
    if m.burning > 0:
        tint = np.maximum(tint, np.array([1.0, 0.7, 0.3]))
    inst[:, 16:19] = tint
    return inst


def item_boxes(d, a: float, light: np.ndarray, origin: tuple, t: float) -> np.ndarray:  # noqa: ANN001 - ItemEntity
    """A dropped item: a little spinning block, or its flat sprite."""
    L = layers()
    x, y, z = d.lerp(a)
    bob = math.sin((d.age + t) * 2.5) * 0.06 + 0.12
    item = d.item[0]
    flat = B.flat_item(item)
    n = 1 if flat or d.item[1] == 1 else 2
    inst = np.zeros((n, BOX_FLOATS), F32)
    inst[:, 0], inst[:, 2] = x - origin[0], z - origin[2]
    inst[:, 3] = math.radians(d.spin + a * 4)
    inst[:, 16:19] = light
    inst[:, 19] = 1.0
    if flat:
        inst[:, 1] = y + bob + 0.2
        inst[:, 12:15] = (0.2, 0.2, 0.012)
        inst[:, 20:24] = L.blank
        inst[:, 24:26] = L.sprite.get(item, L.blank)
    else:
        inst[:, 1] = y + bob + 0.125
        inst[:, 12:15] = 0.125
        inst[:, 20:26] = B.FACE_TEX[item]
        if n == 2:
            inst[1, 0:3] += (0.06, 0.06, -0.06)
    return inst


def block_box(f, a: float, light: np.ndarray, origin: tuple) -> np.ndarray:  # noqa: ANN001 - an entities.BlockEntity
    """A falling block, or primed TNT flashing white and swelling before it goes off."""
    x, y, z = f.lerp(a)
    inst = np.zeros((1, BOX_FLOATS), F32)
    scale = 1.0
    if f.fuse:
        if int(f.fuse * 5) % 2 == 0:
            inst[0, 15] = 0.6
        if f.fuse < 0.5:
            scale += (0.5 - f.fuse) * 0.4
    inst[0, 0:3] = (x - origin[0], y + 0.49, z - origin[2])
    inst[0, 12:15] = 0.49 * scale
    inst[0, 16:19] = light
    inst[0, 19] = 1.0
    inst[0, 20:26] = B.FACE_TEX[f.block]
    return inst


# --- shaders ---------------------------------------------------------------------------------------------------------

SHADOW_SIZE = 2048  # Shaders: the sun's depth picture, and how far around the camera it reaches (blocks)
SHADOW_RANGE = 80.0


def _glsl_array(kind: str, name: str, rows: list) -> str:
    body = ", ".join(f"{kind}({', '.join(f'{float(v):.6f}' for v in r)})" for r in rows)
    return f"const {kind} {name}[{len(rows)}] = {kind}[{len(rows)}]({body});\n"


_SKY_GLSL = """
uniform vec3 u_zenith; uniform vec3 u_horizon; uniform vec3 u_sun; uniform float u_sunset; uniform vec4 u_flat;
vec3 skyColor(vec3 d) {
    if (u_flat.a > 0.5) return u_flat.rgb;
    float t = 1.0 - pow(1.0 - clamp(d.y, 0.0, 1.0), 2.4);
    vec3 c = mix(u_horizon, u_zenith, t);
    float g = max(dot(normalize(d.xz + vec2(1e-5)), normalize(u_sun.xz + vec2(1e-5))), 0.0);
    float low = 1.0 - clamp(abs(d.y) * 2.4, 0.0, 1.0);
    return mix(c, vec3(1.0, 0.46, 0.18), u_sunset * g * g * g * low * 0.8);
}
uniform vec2 u_fog; uniform vec3 u_cam;
vec3 fogged(vec3 rgb, vec3 rel) {
    float f = clamp((length(rel) - u_fog.x) / (u_fog.y - u_fog.x), 0.0, 1.0);
    f = f * f * (3.0 - 2.0 * f);
    return mix(rgb, skyColor(normalize(vec3(rel.x, 0.0, rel.z) + vec3(1e-5))), f);
}
"""

_LIGHT_GLSL = """
uniform float u_day; uniform float u_gamma; uniform vec3 u_skytint;
float curve(float f) {
    float b = f / (4.0 - 3.0 * f);
    float inv = (1.0 - b) * (1.0 - b);
    b += (1.0 - inv * inv - b) * u_gamma;
    return b * 0.96 + 0.04;
}
"""

_SHAPE_GLSL = (
    _glsl_array("vec3", "O", [s[0] for s in SHAPES]) + _glsl_array("vec3", "U", [s[1] for s in SHAPES]) +
    _glsl_array("vec3", "V", [s[2] for s in SHAPES]) + _glsl_array("vec4", "UV", [s[3] for s in SHAPES]) +
    f"const float SHADE[{len(SHAPES)}] = float[{len(SHAPES)}]({', '.join(str(float(s[4])) for s in SHAPES)});\n")

# Shaders: the world as the sun sees it, its depth kept in a texture (what the sun cannot see is in shadow)
_SHADOW_VS = "#version 330 core\n" + _SHAPE_GLSL + """
layout(location = 0) in vec4 a_pos; layout(location = 1) in vec4 a_dat;
uniform mat4 u_vp; uniform vec3 u_off; uniform float u_time;
out vec2 v_uv; flat out float v_layer;
const int FLIP[4] = int[4](1, 3, 0, 2);
void main() {
    vec4 P = floor(a_pos * 255.0 + 0.5);
    vec4 D = floor(a_dat * 255.0 + 0.5);
    int t = int(P.w), flags = int(D.y), id = gl_VertexID;
    if ((flags & 1) != 0) id = FLIP[id];
    float u = float(id & 1), v = float(id >> 1);
    vec3 p = P.xyz + O[t] + U[t] * u + V[t] * v + u_off;
    if ((flags & 4) != 0 && v > 0.5) {
        float ph = u_time * 1.7 + p.x * 0.7 + p.z * 0.9;
        p.x += sin(ph) * 0.05; p.z += cos(ph * 0.83) * 0.05;
    }
    gl_Position = u_vp * vec4(p, 1.0);
    vec4 r = UV[t];
    v_uv = vec2(r.x + u * r.z, 1.0 - (r.y + v * r.w));
    v_layer = D.x;
}
"""

_SHADOW_FS = """#version 330 core
in vec2 v_uv; flat in float v_layer; uniform sampler2DArray u_tex; out vec4 o;
void main() {
    if (texture(u_tex, vec3(v_uv, v_layer)).a < 0.35) discard;
    o = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0);
}
"""

_CHUNK_VS = "#version 330 core\n" + _LIGHT_GLSL + _SHAPE_GLSL + """
layout(location = 0) in vec4 a_pos; layout(location = 1) in vec4 a_dat;
layout(location = 2) in vec4 a_sky; layout(location = 3) in vec4 a_blk;
uniform mat4 u_vp; uniform vec3 u_off; uniform float u_time; uniform vec3 u_cam;
uniform float u_fx; uniform vec3 u_light; uniform float u_lightpow; uniform float u_amb; uniform mat4 u_shvp;
out vec2 v_uv; flat out float v_layer; out vec3 v_light; out vec3 v_rel; out vec3 v_sun; out vec3 v_sh;
const int FLIP[4] = int[4](1, 3, 0, 2);
const float AO[4] = float[4](0.42, 0.6, 0.8, 1.0);
void main() {
    vec4 P = floor(a_pos * 255.0 + 0.5);
    vec4 D = floor(a_dat * 255.0 + 0.5);
    int t = int(P.w), flags = int(D.y), id = gl_VertexID;
    if ((flags & 1) != 0) id = FLIP[id];
    float u = float(id & 1), v = float(id >> 1);
    vec3 p = P.xyz + O[t] + U[t] * u + V[t] * v + u_off;
    if ((flags & 4) != 0 && v > 0.5) {
        float ph = u_time * 1.7 + p.x * 0.7 + p.z * 0.9;
        p.x += sin(ph) * 0.05; p.z += cos(ph * 0.83) * 0.05;
    }
    gl_Position = u_vp * vec4(p, 1.0);
    v_rel = p - u_cam;
    vec4 r = UV[t];
    v_uv = vec2(r.x + u * r.z, 1.0 - (r.y + v * r.w));
    v_layer = D.x + (((flags & 2) != 0) ? 2.0 * mod(floor(u_time * 5.0), 8.0) : 0.0);
    v_sun = vec3(0.0);
    v_sh = vec3(0.5, 0.5, 0.0);
    if ((flags & 8) != 0) {
        v_light = vec3(1.0);
    } else {
        float s = curve(a_sky[id]) * u_day, b = curve(a_blk[id]);
        float ao = AO[(int(D.z) >> (2 * id)) & 3];
        float shade = SHADE[t];
        vec3 skyc = u_skytint;
        if (u_fx > 0.5) {  // shaders: the sun (or moon) lights what faces it and is not in shadow (the fragment
            vec3 n = normalize(cross(U[t], V[t]));  // shader asks the shadow map); the sky fills the rest in blue
            bool plant = t >= 6 && t <= 14;
            float lit = (plant ? 0.72 : max(dot(n, u_light), 0.0)) * smoothstep(0.45, 0.9, a_sky[id]) * u_lightpow;
            shade = plant ? 0.84 : 0.64 + 0.12 * n.y;
            skyc = u_skytint * vec3(0.70, 0.80, 1.0);
            v_sun = u_skytint * vec3(0.74, 0.60, 0.40) * lit * s * (0.55 + 0.45 * ao);
            v_sh = (u_shvp * vec4(p + (plant ? vec3(0.0, 0.05, 0.0) : n * 0.07), 1.0)).xyz * 0.5 + 0.5;
        }
        v_light = max(max(skyc * s, vec3(1.0, 0.88, 0.66) * b), vec3(u_amb)) * shade * ao;
    }
}
"""

_CHUNK_FS = "#version 330 core\n" + _SKY_GLSL + """
in vec2 v_uv; flat in float v_layer; in vec3 v_light; in vec3 v_rel; in vec3 v_sun; in vec3 v_sh;
uniform sampler2DArray u_tex; uniform float u_alpha;
uniform float u_fx; uniform vec3 u_light; uniform float u_lightpow; uniform float u_time;
uniform sampler2D u_shadow; uniform float u_shon; uniform float u_shpx;
out vec4 o;
float sunlit() {  // 1 in the sun, 0 in shadow: nine looks at the sun's depth picture around this spot
    if (u_shon < 0.5) return 1.0;
    vec3 q = v_sh;
    if (q.z > 1.0) return 1.0;
    vec2 e = abs(q.xy - 0.5) * 2.0;
    float edge = max(e.x, e.y);
    if (edge >= 1.0) return 1.0;
    float s = 0.0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            s += q.z - 0.00035 > texture(u_shadow, q.xy + vec2(float(i), float(j)) * u_shpx).r ? 0.0 : 1.0;
    return mix(s / 9.0, 1.0, smoothstep(0.82, 1.0, edge));
}
void main() {
    vec4 c = texture(u_tex, vec3(v_uv, v_layer));
    if (c.a < 0.35) discard;
    float sh = dot(v_sun, vec3(1.0)) > 0.001 ? sunlit() : 1.0;
    vec3 lightc = v_light + v_sun * sh;
    vec3 rgb = c.rgb * lightc;
    float alpha = u_alpha < 0.0 ? c.a : u_alpha;
    if (u_fx > 0.5 && u_alpha > 0.0 && u_alpha < 0.99) {  // shaders: water with waves, depth of colour, the sky
        vec3 e = normalize(v_rel);                         // mirrored in it and the sun glittering on it
        vec2 w = (v_rel.xz + u_cam.xz) * 0.3927;  // whole waves per 16 blocks: the pattern has no seam
        vec2 g = vec2(0.0);
        g += vec2(3.0, 1.0) * cos(w.x * 3.0 + w.y * 1.0 + u_time * 1.10) * 0.016;
        g += vec2(-2.0, 4.0) * cos(-w.x * 2.0 + w.y * 4.0 + u_time * 1.45) * 0.011;
        g += vec2(5.0, -3.0) * cos(w.x * 5.0 - w.y * 3.0 - u_time * 1.90) * 0.006;
        g += vec2(-7.0, -6.0) * cos(-w.x * 7.0 - w.y * 6.0 + u_time * 2.60) * 0.0034;
        g += vec2(11.0, 4.0) * cos(w.x * 11.0 + w.y * 4.0 - u_time * 3.30) * 0.0018;
        float far = clamp(length(v_rel) / 90.0, 0.0, 1.0);
        vec3 n = normalize(vec3(-g.x * (1.0 - 0.6 * far), 1.0, -g.y * (1.0 - 0.6 * far)));
        if (e.y > 0.0) n = -n;
        float facing = clamp(dot(-e, n), 0.0, 1.0);
        vec3 lit = clamp(lightc, 0.0, 1.25);
        vec3 body = mix(vec3(0.015, 0.105, 0.21), vec3(0.06, 0.34, 0.42), facing * 0.75) * (0.25 + 0.75 * lit);
        body = mix(body, c.rgb * lit, 0.18);
        float fres = 0.03 + 0.97 * pow(1.0 - facing, 5.0);
        vec3 back = reflect(e, n);
        vec3 up = vec3(back.x, abs(back.y), back.z);
        vec3 sky = skyColor(up) * (0.35 + 0.65 * clamp(dot(v_light, vec3(0.45)), 0.0, 1.0));
        float sd = max(dot(up, u_light), 0.0);
        float glint = (pow(sd, 900.0) * 3.0 + pow(sd, 90.0) * 0.22) * u_lightpow * sh * u_shon;
        glint += pow(sd, 600.0) * 0.9 * u_lightpow * (1.0 - u_shon);
        rgb = mix(body, sky, fres * 0.9) + vec3(1.0, 0.93, 0.78) * glint;
        alpha = mix(0.74, 0.98, fres);
    }
    o = vec4(fogged(rgb, v_rel), alpha);
}
"""

_BOX_VS = "#version 330 core\n" + """
layout(location = 0) in vec4 a_base; layout(location = 1) in vec4 a_pivot; layout(location = 2) in vec4 a_offs;
layout(location = 3) in vec4 a_half; layout(location = 4) in vec4 a_tint; layout(location = 5) in vec4 a_tex0;
layout(location = 6) in vec4 a_tex1;
uniform mat4 u_vp; uniform vec3 u_cam;
out vec2 v_uv; flat out float v_layer; out vec3 v_light; out vec3 v_rel; flat out float v_white;
const vec3 O[6] = vec3[6](vec3(1,-1,1), vec3(-1,-1,-1), vec3(-1,1,1), vec3(-1,-1,-1), vec3(-1,-1,1), vec3(1,-1,-1));
const vec3 U[6] = vec3[6](vec3(0,0,-2), vec3(0,0,2), vec3(2,0,0), vec3(2,0,0), vec3(2,0,0), vec3(-2,0,0));
const vec3 V[6] = vec3[6](vec3(0,2,0), vec3(0,2,0), vec3(0,0,-2), vec3(0,0,2), vec3(0,2,0), vec3(0,2,0));
const float SHADE[6] = float[6](0.6, 0.6, 1.0, 0.5, 0.8, 0.8);
const int QUAD[6] = int[6](0, 1, 2, 2, 1, 3);
void main() {
    int face = gl_VertexID / 6, id = QUAD[gl_VertexID - face * 6];
    float u = float(id & 1), v = float(id >> 1);
    vec3 p = a_offs.xyz + (O[face] + U[face] * u + V[face] * v) * a_half.xyz;
    float c = cos(a_pivot.w), s = sin(a_pivot.w);
    p = vec3(p.x, p.y * c + p.z * s, p.z * c - p.y * s) + a_pivot.xyz;
    c = cos(a_offs.w); s = sin(a_offs.w);
    p = vec3(p.x * c - p.y * s, p.x * s + p.y * c, p.z);
    c = cos(a_base.w); s = sin(a_base.w);
    p = vec3(p.x * c - p.z * s, p.y, p.x * s + p.z * c) + a_base.xyz;
    gl_Position = u_vp * vec4(p, 1.0);
    v_rel = p - u_cam;
    v_uv = vec2(u, 1.0 - v);
    v_layer = face < 4 ? a_tex0[face] : a_tex1[face - 4];
    v_light = a_tint.rgb * SHADE[face];
    v_white = a_half.w;
}
"""

_BOX_FS = "#version 330 core\n" + _SKY_GLSL + """
in vec2 v_uv; flat in float v_layer; in vec3 v_light; in vec3 v_rel; flat in float v_white;
uniform sampler2DArray u_tex;
out vec4 o;
void main() {
    vec4 c = texture(u_tex, vec3(v_uv, v_layer));
    if (c.a < 0.35) discard;
    o = vec4(fogged(mix(c.rgb * v_light, vec3(1.0), v_white), v_rel), 1.0);
}
"""

_DOT_VS = "#version 330 core\n" + """
layout(location = 0) in vec4 a_pos; layout(location = 1) in vec4 a_col;
uniform mat4 u_vp; uniform vec3 u_cam; uniform vec3 u_right; uniform vec3 u_up;
out vec3 v_col; out vec3 v_rel;
void main() {
    float u = float(gl_VertexID & 1) - 0.5, v = float(gl_VertexID >> 1) - 0.5;
    vec3 p = a_pos.xyz + (u_right * u + u_up * v) * a_pos.w;
    gl_Position = u_vp * vec4(p, 1.0);
    v_rel = p - u_cam; v_col = a_col.rgb;
}
"""

_DOT_FS = "#version 330 core\n" + _SKY_GLSL + """
in vec3 v_col; in vec3 v_rel; out vec4 o;
void main() { o = vec4(fogged(v_col, v_rel), 1.0); }
"""

_SKY_VS = """#version 330 core
out vec2 v_ndc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
    v_ndc = p; gl_Position = vec4(p, 1.0, 1.0);
}
"""

_SKY_FS = "#version 330 core\n" + _SKY_GLSL + f"const float CLOUD_Y = {CLOUD_Y};\n" + """
in vec2 v_ndc; out vec4 o;
uniform vec3 u_r; uniform vec3 u_u; uniform vec3 u_f; uniform vec3 u_eye;
uniform float u_time; uniform float u_stars; uniform float u_dayf; uniform float u_clouds;
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec3 d = normalize(u_f + u_r * v_ndc.x - u_u * v_ndc.y);
    vec3 c = skyColor(d);
    if (u_flat.a < 0.5) {
        if (u_stars > 0.01 && d.y > -0.02) {
            vec3 a = abs(d); float m = max(a.x, max(a.y, a.z));
            vec2 uv = m == a.y ? d.xz / m : (m == a.x ? d.yz / m : d.xy / m);
            float h = hash(floor(uv * 110.0) + (m == a.y ? 7.0 : (m == a.x ? 19.0 : 31.0)) * sign(dot(d, vec3(1.0))));
            if (h > 0.9935) c += vec3(0.9, 0.95, 1.0) * u_stars * (0.55 + 0.45 * sin(u_time * 1.7 + h * 900.0))
                * clamp(d.y * 6.0 + 0.2, 0.0, 1.0);
        }
        vec3 tng = vec3(-u_sun.y, u_sun.x, 0.0);
        float sd = dot(d, u_sun);
        vec2 q = floor(vec2(d.z, dot(d, tng)) / max(abs(sd), 1e-4) * 64.0 + 0.5);
        float r = length(q);
        if (sd > 0.0) {
            if (r < 5.6) c = mix(vec3(1.0, 0.86, 0.5), vec3(1.0, 0.98, 0.86), clamp(1.4 - r * 0.2, 0.0, 1.0));
            else c += vec3(1.0, 0.78, 0.45) * exp(-r * 0.16) * 0.45;
        } else if (r < 4.6 && length(q - vec2(2.0, 1.0)) > 3.7) {
            c = mix(c, vec3(0.86, 0.9, 1.0), 0.95);
        } else if (r < 4.6) {
            c = mix(c, vec3(0.2, 0.22, 0.3), 0.5 * u_stars);
        }
        if (u_clouds > 0.5 && d.y > 0.015) {
            float t = (CLOUD_Y - u_eye.y) / d.y;
            vec2 cell = floor((u_eye.xz + d.xz * t + vec2(u_time * 1.1, 0.0)) / 12.0);
            float n = vnoise(cell / 5.5) * 0.7 + vnoise(cell / 2.1 + 40.0) * 0.3;
            if (n > 0.56) {
                vec3 cc = mix(vec3(0.1, 0.11, 0.17), vec3(1.0), u_dayf);
                cc = mix(cc, vec3(1.0, 0.62, 0.42), u_sunset * 0.5);
                c = mix(c, cc, 0.84 * clamp(1.0 - t / 1100.0, 0.0, 1.0));
            }
        }
    }
    o = vec4(c, 1.0);
}
"""


_POST_FS = """#version 330 core
in vec2 v_ndc; out vec4 o;
uniform sampler2D u_src; uniform vec2 u_px; uniform vec3 u_sunpos; uniform float u_rays;
void main() {
    vec2 uv = v_ndc * 0.5 + 0.5;
    vec3 c = texture(u_src, uv).rgb;
    vec3 glow = vec3(0.0);  // light spills a little around what is bright
    for (int i = 0; i < 8; i++) {
        float a = 0.7854 * float(i);
        vec2 d = vec2(cos(a), sin(a)) * u_px;
        glow += max(texture(u_src, uv + d * 5.0).rgb - 0.74, 0.0) + max(texture(u_src, uv + d * 11.0).rgb - 0.8, 0.0);
    }
    c += glow * 0.085;
    if (u_rays > 0.01 && u_sunpos.z > 0.5) {  // rays from the sun through the gaps
        vec2 step_ = (u_sunpos.xy - uv) / 14.0;
        vec2 q = uv;
        float sum = 0.0;
        for (int i = 0; i < 14; i++) {
            q += step_;
            vec3 s = texture(u_src, q).rgb;
            sum += max(dot(s, vec3(0.3, 0.5, 0.2)) - 0.62, 0.0) * (1.0 - float(i) / 14.0);
        }
        float near = 1.0 - clamp(length((uv - u_sunpos.xy) * vec2(1.0, u_px.x / u_px.y)) * 1.3, 0.0, 1.0);
        c += vec3(1.0, 0.86, 0.62) * sum * 0.07 * u_rays * near;
    }
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(l), c, 1.16);  // a little more colour
    c = c * (1.06 - 0.06 * c);
    vec2 v = uv - 0.5;
    c *= 1.0 - dot(v, v) * 0.5;  // darker corners
    o = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""


class _Mesh:
    """One buffer of instances on the graphics card."""

    __slots__ = ("count", "vao", "vbo")

    def __init__(self) -> None:
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.count = 0

    def destroy(self) -> None:
        self.vao.destroy()
        self.vbo.destroy()


class Renderer:
    """Draws a ``World`` from a camera into a QImage. Everything OpenGL is made the first time it draws."""

    basis = staticmethod(basis)

    def __init__(self) -> None:
        self.ctx: QOpenGLContext | None = None
        self.surface: QOffscreenSurface | None = None
        self.f: QOpenGLExtraFunctions | None = None
        self.fbo: QOpenGLFramebufferObject | None = None
        self.resolve: QOpenGLFramebufferObject | None = None
        self.final: QOpenGLFramebufferObject | None = None  # the finished picture when Shaders are on
        self.shadow: QOpenGLFramebufferObject | None = None  # Shaders: the depth of the world seen from the sun
        self.samples = 4
        self.meshes: dict[tuple[int, int], list] = {}  # chunk -> [solid mesh, water mesh, top]
        self._keys: list = []
        self._coords = np.zeros((0, 3), F32)
        self._world_id = None
        self.gpu = ""
        self.stats = {"chunks": 0, "faces": 0, "drawn": 0, "ms": 0.0}
        self.smooth = True
        self.fov_scale = 1.0  # eased toward the sprint zoom by the game

    # --- set-up -------------------------------------------------------------------------------------------------

    def _init(self) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            raise RuntimeError("BlockCraft needs OpenGL 3.3 (no graphics context could be made)")
        surface = QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()
        if not ctx.makeCurrent(surface):
            raise RuntimeError("BlockCraft could not use its graphics context")
        self.ctx, self.surface = ctx, surface
        f = self.f = QOpenGLExtraFunctions(ctx)
        f.initializeOpenGLFunctions()
        self.gpu = str(f.glGetString(0x1F01) or "")
        self.p_chunk = self._program(_CHUNK_VS, _CHUNK_FS)
        self.p_box = self._program(_BOX_VS, _BOX_FS)
        self.p_dot = self._program(_DOT_VS, _DOT_FS)
        self.p_sky = self._program(_SKY_VS, _SKY_FS)
        self.p_post = self._program(_SKY_VS, _POST_FS)
        self.p_shadow = self._program(_SHADOW_VS, _SHADOW_FS)
        self.loc = {id(p): {} for p in (self.p_chunk, self.p_box, self.p_dot, self.p_sky, self.p_post, self.p_shadow)}
        # the textures
        L = layers()
        tex = QOpenGLTexture(QOpenGLTexture.Target.Target2DArray)
        tex.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        tex.setSize(16, 16)
        tex.setLayers(L.count)
        tex.setMipLevels(5)
        tex.allocateStorage(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8)
        for level, images in enumerate(L.mips()):  # every layer of a level in one call: thousands of single-layer
            tex.setData(level, 0, L.count, QOpenGLTexture.CubeMapFace.CubeMapPositiveX,  # calls crash PySide6
                        QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8,
                        np.ascontiguousarray(images[:L.count]).tobytes())
        tex.setMinMagFilters(QOpenGLTexture.Filter.LinearMipMapLinear, QOpenGLTexture.Filter.Nearest)
        tex.setWrapMode(QOpenGLTexture.WrapMode.Repeat)
        try:
            tex.setMaximumAnisotropy(8.0)
        except Exception:  # noqa: BLE001 - not every driver has it
            pass
        self.tex = tex
        # an empty vertex array for the sky, and the instance buffers of the moving things
        self.sky_vao = QOpenGLVertexArrayObject()
        self.sky_vao.create()
        self.boxes = self._instance_mesh([(i, i * 16) for i in range(7)], GL_FLOAT, BOX_FLOATS * 4)
        self.dots = self._instance_mesh([(0, 0), (1, 16)], GL_FLOAT, 32)
        self.crack = self._instance_mesh([(i, i * 4) for i in range(4)], GL_UNSIGNED_BYTE, FACE_BYTES)

    def warm_up(self) -> None:
        """Make the graphics context, shaders and textures now (behind the loading screen, not on the first frame)."""
        if self.ctx is None:
            self._init()
            self.ctx.doneCurrent()

    def _program(self, vs: str, fs: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
        ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs) and ok
        if not (ok and prog.link()):
            raise RuntimeError(f"BlockCraft shader did not compile: {prog.log()}")
        return prog

    def _instance_mesh(self, attrs: list, kind: int, stride: int, data: bytes = b"") -> _Mesh:
        f = self.f
        m = _Mesh()
        m.vao.create()
        m.vao.bind()
        m.vbo.create()
        m.vbo.bind()
        if data:
            m.vbo.allocate(data, len(data))
        for loc, offset in attrs:
            f.glEnableVertexAttribArray(loc)
            self.p_chunk.setAttributeBuffer(loc, kind, offset, 4, stride)
            f.glVertexAttribDivisor(loc, 1)
        m.vao.release()
        m.vbo.release()
        return m

    def _u(self, prog: QOpenGLShaderProgram, name: str) -> int:
        cache = self.loc[id(prog)]
        loc = cache.get(name)
        if loc is None:
            loc = cache[name] = prog.uniformLocation(name)
        return loc

    def close(self) -> None:
        """Give everything back to the graphics card."""
        if self.ctx is None:
            return
        try:
            if self.ctx.makeCurrent(self.surface):
                for solid, wet, _top in self.meshes.values():
                    for m in (solid, wet):
                        if m is not None:
                            m.destroy()
                for m in (self.boxes, self.dots, self.crack):
                    m.destroy()
                self.sky_vao.destroy()
                self.tex.destroy()
                self.fbo = self.resolve = self.final = self.shadow = None
                self.ctx.doneCurrent()
        except Exception:  # noqa: BLE001 - closing must never fail
            log.debug("BlockCraft renderer close", exc_info=True)
        self.meshes.clear()
        self.surface = None
        self.ctx = None

    def _size(self, w: int, h: int, samples: int, shaders: bool = False) -> None:
        if shaders and (self.final is None or self.final.width() != w or self.final.height() != h):
            fmt = QOpenGLFramebufferObjectFormat()
            fmt.setInternalTextureFormat(GL_RGB8)
            self.final = QOpenGLFramebufferObject(w, h, fmt)
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

    # --- chunk meshes -------------------------------------------------------------------------------------------

    def _upload(self, world, budget: float = 0.004) -> None:  # noqa: ANN001 - a World
        """Take the world's new chunk meshes (a few milliseconds' worth per frame)."""
        if self._world_id != world.wid:  # another world: drop everything of the last one
            for solid, wet, _top in self.meshes.values():
                for m in (solid, wet):
                    if m is not None:
                        m.destroy()
            self.meshes.clear()
            self._keys = []
            self._world_id = world.wid
        out = world.meshes_out
        if not out:
            return
        t0 = time.perf_counter()
        attrs = [(i, i * 4) for i in range(4)]
        changed = False
        while out:
            key, data = next(iter(out.items()))
            del out[key]
            entry = self.meshes.get(key)
            if data is None:
                if entry is not None:
                    for m in entry[:2]:
                        if m is not None:
                            m.destroy()
                    del self.meshes[key]
                    changed = True
                continue
            if entry is None:
                entry = self.meshes[key] = [None, None, 1]
                changed = True
            for i in (0, 1):
                raw = data[i]
                m = entry[i]
                if not raw:
                    if m is not None:
                        m.count = 0
                    continue
                if m is None:
                    entry[i] = m = self._instance_mesh(attrs, GL_UNSIGNED_BYTE, FACE_BYTES, raw)
                else:
                    m.vbo.bind()
                    m.vbo.allocate(raw, len(raw))
                    m.vbo.release()
                m.count = len(raw) // FACE_BYTES
            top = 1
            for raw in data:
                if raw:
                    top = max(top, int(np.frombuffer(raw, np.uint8)[1::FACE_BYTES].max()) + 2)
            if top != entry[2]:
                entry[2] = top
                changed = True
            if time.perf_counter() - t0 > budget:
                break
        if changed:
            self._keys = list(self.meshes)
            self._coords = np.array([(k[0] * 16, self.meshes[k][2], k[1] * 16) for k in self._keys], F32).reshape(-1, 3)

    def _visible(self, vp: np.ndarray, origin: tuple, cam: np.ndarray, max_dist: float) -> list:
        """The chunks inside the view (and the render distance), nearest first."""
        if not self._keys:
            return []
        lo = self._coords.copy()
        lo[:, 0] -= origin[0]
        lo[:, 2] -= origin[2]
        size = np.empty_like(lo)
        size[:, 0] = size[:, 2] = 16.0
        size[:, 1] = lo[:, 1]
        lo[:, 1] = 0.0
        cx, cz = lo[:, 0] + 8 - cam[0], lo[:, 2] + 8 - cam[2]
        d2 = cx * cx + cz * cz
        keep = d2 < (max_dist + 12) ** 2
        for a, b in ((3, 0), (3, 1), (3, 2)):
            for sign in (1.0, -1.0):
                pl = vp[a] + sign * vp[b]
                far = lo + size * (pl[:3] > 0)
                keep &= far @ pl[:3] + pl[3] >= 0
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(d2[idx])]
        return [(self._keys[i], float(lo[i, 0]), float(lo[i, 2])) for i in idx]

    # --- the frame ----------------------------------------------------------------------------------------------

    def render(self, world, eye: tuple, yaw: float, pitch: float, roll: float, fov: float, w: int, h: int,  # noqa: ANN001
               env: Env, chunks: int = 10, boxes: np.ndarray | None = None, dots: np.ndarray | None = None,
               crack: tuple | None = None, fancy: bool = True, samples: int = 4, shaders: bool = False) -> QImage:
        """One frame. ``chunks``: the render distance; ``boxes`` (n, 28) and ``dots`` (n, 8) float32 instances
        (positions relative to ``origin(eye)``); ``crack`` = (x, y, z, stage 0..9) of the block being broken."""
        t0 = time.perf_counter()
        if self.ctx is None:
            self._init()
        elif not self.ctx.makeCurrent(self.surface):
            raise RuntimeError("BlockCraft lost its graphics context")
        f = self.f
        self._size(max(w, 16), max(h, 16), samples, shaders)
        self._upload(world)
        ox, oz = self.origin(eye)[0], self.origin(eye)[2]
        cam = np.array([eye[0] - ox, eye[1], eye[2] - oz], F32)
        M = basis(yaw, pitch, roll)
        max_dist = chunks * 16.0
        vp = self._view_projection(M, cam, fov, w / max(h, 1), max_dist)
        # the sky and fog of this moment
        day = float(env.day)
        if env.in_lava:
            flat, fog = (0.9, 0.32, 0.05, 1.0), (0.0, 2.5)
        elif env.underwater:
            k = 0.25 + 0.75 * day
            flat, fog = (0.04 * k, 0.16 * k, 0.42 * k, 1.0), (0.0, 30.0)
        elif env.flat_sky:
            c = env.fog / 255.0
            flat = (float(c[0]), float(c[1]), float(c[2]), 1.0)
            fog = (env.fog_range[0], min(env.fog_range[1], max_dist - 6.0))
        else:
            flat, fog = (0.0, 0.0, 0.0, 0.0), (max_dist * 0.62, max_dist - 6.0)
        zen, hor = env.sky_top / 255.0, env.fog / 255.0
        tint = (0.55 + 0.45 * day, 0.62 + 0.38 * day, 1.0)

        def common(prog: QOpenGLShaderProgram) -> None:
            prog.bind()
            prog.setUniformValue(self._u(prog, "u_vp"), QMatrix4x4(*[float(v) for v in vp.reshape(-1)]))
            f.glUniform3f(self._u(prog, "u_cam"), float(cam[0]), float(cam[1]), float(cam[2]))
            f.glUniform3f(self._u(prog, "u_zenith"), float(zen[0]), float(zen[1]), float(zen[2]))
            f.glUniform3f(self._u(prog, "u_horizon"), float(hor[0]), float(hor[1]), float(hor[2]))
            f.glUniform3f(self._u(prog, "u_sun"), float(env.sun[0]), float(env.sun[1]), float(env.sun[2]))
            f.glUniform1f(self._u(prog, "u_sunset"), float(env.sunset))
            f.glUniform4f(self._u(prog, "u_flat"), *flat)
            f.glUniform2f(self._u(prog, "u_fog"), *fog)

        sun_up = float(env.sun[1]) > 0.0  # the light that falls: the sun by day, the moon by night
        light = env.sun if sun_up else -env.sun
        light = light + np.array([0.0, 0.25, 0.3], F32)
        light = light / float(np.linalg.norm(light))
        shvp = None
        if shaders and not (env.flat_sky or env.underwater or env.in_lava):
            shvp = self._shadow_pass(light, eye, ox, oz, float(env.time))
        self.fbo.bind()
        f.glViewport(0, 0, w, h)
        f.glDisable(GL_BLEND)
        f.glDisable(GL_DEPTH_TEST)
        f.glDisable(GL_CULL_FACE)
        f.glDepthMask(True)
        f.glClear(GL_DEPTH_BUFFER_BIT)
        # sky
        prog = self.p_sky
        common(prog)
        tan = math.tan(math.radians(fov) / 2)
        r, u = M[0] * (tan * w / max(h, 1)), M[1] * tan
        f.glUniform3f(self._u(prog, "u_r"), float(r[0]), float(r[1]), float(r[2]))
        f.glUniform3f(self._u(prog, "u_u"), float(u[0]), float(u[1]), float(u[2]))
        f.glUniform3f(self._u(prog, "u_f"), float(M[2][0]), float(M[2][1]), float(M[2][2]))
        f.glUniform3f(self._u(prog, "u_eye"), float(eye[0] % 12288.0), float(eye[1]), float(eye[2] % 12288.0))
        f.glUniform1f(self._u(prog, "u_time"), float(env.time))
        f.glUniform1f(self._u(prog, "u_stars"), float(env.stars))
        f.glUniform1f(self._u(prog, "u_dayf"), day)
        f.glUniform1f(self._u(prog, "u_clouds"), 1.0 if fancy and env.clouds else 0.0)
        self.sky_vao.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.sky_vao.release()
        # the world: solid faces, nearest chunks first
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LESS)
        f.glEnable(GL_CULL_FACE)
        f.glFrontFace(GL_CW)  # the picture is drawn upside down, which turns every triangle around
        prog = self.p_chunk
        common(prog)
        self.tex.bind(0)
        prog.setUniformValue1i(self._u(prog, "u_tex"), 0)
        f.glUniform1f(self._u(prog, "u_time"), float(env.time))
        f.glUniform1f(self._u(prog, "u_day"), float(env.daylight))
        f.glUniform1f(self._u(prog, "u_gamma"), _GAMMA[0])
        f.glUniform3f(self._u(prog, "u_skytint"), *tint)
        f.glUniform1f(self._u(prog, "u_alpha"), 1.0)
        power = (min(1.0, float(env.sun[1]) * 3.0 + 0.25) if sun_up else 0.25)
        f.glUniform1f(self._u(prog, "u_shon"), 1.0 if shvp is not None else 0.0)
        f.glUniform1f(self._u(prog, "u_shpx"), 1.0 / SHADOW_SIZE)
        prog.setUniformValue1i(self._u(prog, "u_shadow"), 1)
        if shvp is not None:
            prog.setUniformValue(self._u(prog, "u_shvp"), QMatrix4x4(*[float(v) for v in shvp.reshape(-1)]))
            f.glActiveTexture(0x84C1)
            f.glBindTexture(0x0DE1, self.shadow.texture())
            f.glActiveTexture(0x84C0)
        f.glUniform1f(self._u(prog, "u_amb"), float(env.ambient))
        f.glUniform1f(self._u(prog, "u_fx"), 1.0 if shaders else 0.0)
        f.glUniform3f(self._u(prog, "u_light"), float(light[0]), float(light[1]), float(light[2]))
        f.glUniform1f(self._u(prog, "u_lightpow"), power if not (env.underwater or env.in_lava) else power * 0.3)
        off = self._u(prog, "u_off")
        visible = self._visible(vp, (ox, 0.0, oz), cam, max_dist)
        meshes = self.meshes
        faces = 0
        for key, x, z in visible:
            m = meshes[key][0]
            if m is not None and m.count:
                f.glUniform3f(off, x, 0.0, z)
                m.vao.bind()
                f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, m.count)
                faces += m.count
        if crack is not None:
            self._draw_crack(prog, off, crack, ox, oz)
        # animals, items, falling blocks; then particles
        if boxes is not None and len(boxes):
            prog = self.p_box
            common(prog)
            prog.setUniformValue1i(self._u(prog, "u_tex"), 0)
            self._fill(self.boxes, boxes)
            f.glDisable(GL_CULL_FACE)
            self.boxes.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLES, 0, 36, len(boxes))
            f.glEnable(GL_CULL_FACE)
        if dots is not None and len(dots):
            prog = self.p_dot
            common(prog)
            f.glUniform3f(self._u(prog, "u_right"), float(M[0][0]), float(M[0][1]), float(M[0][2]))
            f.glUniform3f(self._u(prog, "u_up"), float(M[1][0]), float(M[1][1]), float(M[1][2]))
            self._fill(self.dots, dots)
            f.glDisable(GL_CULL_FACE)
            self.dots.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(dots))
            f.glEnable(GL_CULL_FACE)
        # water last, farthest first, seen from both sides
        prog = self.p_chunk
        prog.bind()
        f.glUniform1f(self._u(prog, "u_alpha"), 0.7)
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        f.glDepthMask(False)
        f.glDisable(GL_CULL_FACE)
        for key, x, z in reversed(visible):
            m = meshes[key][1]
            if m is not None and m.count:
                f.glUniform3f(off, x, 0.0, z)
                m.vao.bind()
                f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, m.count)
                faces += m.count
        f.glDepthMask(True)
        f.glDisable(GL_BLEND)
        self.sky_vao.bind()
        self.sky_vao.release()
        self.fbo.release()
        source = self.fbo
        if self.resolve is not None:
            QOpenGLFramebufferObject.blitFramebuffer(self.resolve, self.fbo)
            source = self.resolve
        if shaders and self.final is not None:
            source = self._finish(source, M, fov, w, h, env)
        img = source.toImage(False)
        self.ctx.doneCurrent()
        self.stats = {"chunks": len(meshes), "drawn": len(visible), "faces": faces,
                      "ms": (time.perf_counter() - t0) * 1000}
        return img

    def _shadow_pass(self, light: np.ndarray, eye: tuple, ox: float, oz: float, t: float) -> np.ndarray:
        """Shaders: draw the chunks around the camera as the sun (or moon) sees them, keeping only how far each
        point is from it. Returns the matrix that takes a position to that picture."""
        f = self.f
        if self.shadow is None:
            fmt = QOpenGLFramebufferObjectFormat()
            fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
            fmt.setInternalTextureFormat(0x822E)  # GL_R32F
            self.shadow = QOpenGLFramebufferObject(SHADOW_SIZE, SHADOW_SIZE, fmt)
        L = light.astype(np.float64)
        right = np.cross(np.array([0.0, 1.0, 0.0]), L)
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-4 else np.array([1.0, 0.0, 0.0])
        up = np.cross(L, right)
        R, D = SHADOW_RANGE, 260.0
        texel = 2.0 * R / SHADOW_SIZE  # the picture moves in whole texels, so shadows do not crawl as you walk
        e = np.array([eye[0], eye[1], eye[2]], np.float64)
        cx, cy = math.floor(float(right @ e) / texel) * texel, math.floor(float(up @ e) / texel) * texel
        c = right * cx + up * cy + L * float(L @ e) - np.array([ox, 0.0, oz])
        m = np.eye(4)
        m[0, :3], m[0, 3] = right / R, -float(right @ c) / R
        m[1, :3], m[1, 3] = up / R, -float(up @ c) / R
        m[2, :3], m[2, 3] = -L / D, float(L @ c) / D
        m = m.astype(F32)
        self.shadow.bind()
        f.glViewport(0, 0, SHADOW_SIZE, SHADOW_SIZE)
        f.glDisable(GL_BLEND)
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LESS)
        f.glDepthMask(True)
        f.glDisable(GL_CULL_FACE)
        f.glClearColor(1.0, 1.0, 1.0, 1.0)
        f.glClear(GL_DEPTH_BUFFER_BIT | 0x4000)
        prog = self.p_shadow
        prog.bind()
        prog.setUniformValue(self._u(prog, "u_vp"), QMatrix4x4(*[float(v) for v in m.reshape(-1)]))
        self.tex.bind(0)
        prog.setUniformValue1i(self._u(prog, "u_tex"), 0)
        f.glUniform1f(self._u(prog, "u_time"), t)
        off = self._u(prog, "u_off")
        reach = (R + 56.0) ** 2
        px, pz = eye[0] - 8.0, eye[2] - 8.0
        for key, entry in self.meshes.items():
            mesh = entry[0]
            if mesh is None or not mesh.count:
                continue
            x, z = key[0] * 16.0, key[1] * 16.0
            if (x - px) ** 2 + (z - pz) ** 2 > reach:
                continue
            f.glUniform3f(off, x - ox, 0.0, z - oz)
            mesh.vao.bind()
            f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, mesh.count)
        self.shadow.release()
        return m

    def _finish(self, source: QOpenGLFramebufferObject, M: np.ndarray, fov: float, w: int, h: int, env: Env,
                ) -> QOpenGLFramebufferObject:
        """Shaders: the last pass over the whole picture (glow, sun rays, colour, darker corners)."""
        f, prog = self.f, self.p_post
        self.final.bind()
        f.glViewport(0, 0, w, h)
        f.glDisable(GL_DEPTH_TEST)
        f.glDisable(GL_CULL_FACE)
        f.glDisable(GL_BLEND)
        prog.bind()
        f.glActiveTexture(0x84C0)
        f.glBindTexture(0x0DE1, source.texture())
        prog.setUniformValue1i(self._u(prog, "u_src"), 0)
        f.glUniform2f(self._u(prog, "u_px"), 1.0 / w, 1.0 / h)
        q = M @ env.sun
        tan = math.tan(math.radians(fov) / 2)
        seen = float(q[2]) > 0.05
        sx = (float(q[0]) / float(q[2]) / (tan * w / max(h, 1)) + 1) / 2 if seen else 0.5
        sy = (1 - float(q[1]) / float(q[2]) / tan) / 2 if seen else 0.5
        f.glUniform3f(self._u(prog, "u_sunpos"), sx, sy, 1.0 if seen else 0.0)
        rays = 0.0 if (env.underwater or env.in_lava) else float(env.day) * (0.55 + 0.45 * float(env.sunset))
        f.glUniform1f(self._u(prog, "u_rays"), rays)
        self.sky_vao.bind()
        f.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.sky_vao.release()
        f.glBindTexture(0x0DE1, 0)
        self.final.release()
        return self.final

    @staticmethod
    def origin(eye: tuple) -> tuple[float, float, float]:
        """The corner of the camera's chunk: positions are sent to the card relative to it (so far from the middle
        of the world nothing loses precision)."""
        return float(math.floor(eye[0] / 16) * 16), 0.0, float(math.floor(eye[2] / 16) * 16)

    @staticmethod
    def _view_projection(M: np.ndarray, cam: np.ndarray, fov: float, aspect: float, max_dist: float) -> np.ndarray:
        view = np.eye(4, dtype=np.float64)
        view[0, :3], view[1, :3], view[2, :3] = M[0], M[1], -M[2]
        view[:3, 3] = -(view[:3, :3] @ cam.astype(np.float64))
        near, far = 0.05, max(max_dist * 1.5, 96.0) + 64.0
        t = 1.0 / math.tan(math.radians(fov) / 2)
        proj = np.zeros((4, 4))
        proj[0, 0] = t / aspect
        proj[1, 1] = -t  # upside down (see the module docstring)
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2 * far * near / (near - far)
        proj[3, 2] = -1.0
        return (proj @ view).astype(F32)

    def _fill(self, mesh: _Mesh, data: np.ndarray) -> None:
        raw = np.ascontiguousarray(data, F32).tobytes()
        mesh.vbo.bind()
        mesh.vbo.allocate(raw, len(raw))
        mesh.vbo.release()

    def _draw_crack(self, prog: QOpenGLShaderProgram, off: int, crack: tuple, ox: float, oz: float) -> None:
        f = self.f
        x, y, z, stage = crack
        rec = np.zeros((6, FACE_BYTES), np.uint8)
        rec[:, 0], rec[:, 1], rec[:, 2] = x & 15, y, z & 15
        rec[:, 3] = np.arange(6)
        rec[:, 4] = layers().crack0 + min(max(int(stage), 0), 9)
        rec[:, 5] = 8
        raw = rec.tobytes()
        self.crack.vbo.bind()
        self.crack.vbo.allocate(raw, len(raw))
        self.crack.vbo.release()
        f.glUniform1f(self._u(prog, "u_alpha"), -1.0)
        f.glUniform3f(off, float((x >> 4) * 16 - ox), 0.0, float((z >> 4) * 16 - oz))
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        f.glEnable(GL_POLYGON_OFFSET_FILL)
        f.glPolygonOffset(-1.0, -2.0)
        f.glDepthFunc(GL_LEQUAL)
        f.glDepthMask(False)
        self.crack.vao.bind()
        f.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, 6)
        f.glDepthMask(True)
        f.glDepthFunc(GL_LESS)
        f.glDisable(GL_POLYGON_OFFSET_FILL)
        f.glDisable(GL_BLEND)
        f.glUniform1f(self._u(prog, "u_alpha"), 1.0)
