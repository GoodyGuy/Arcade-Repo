"""The first-person weapon (view model), drawn at full resolution with QPainter from the 3D models in
``vmodels``: faces lit by a warm key light and a teal fill (the game's grade) with specular highlights, lit
edges and rim light, gradient shading, decals (markings, rails, vents, grip texture), fabric gloves and camo
sleeves, and the Refit Press's animated camo. Idle sway, walk bob, recoil (the slide / bolt cycles), sprint and swap
poses, reloads (the magazine comes out, the pump racks), aiming down the sights, the knife, throws and drinking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF,
                           QRadialGradient, QTransform)

from . import vmodels as vm

KEY = vm._norm((-0.45, 0.78, -0.45))
FILL = vm._norm((0.85, -0.15, -0.35))
KEY_COL = (1.0, 0.86, 0.7)
FILL_COL = (0.38, 0.6, 0.66)
AMBIENT = (0.3, 0.31, 0.33)
NEAR = 0.03
CLIP = 0.085  # faces are cut here (closer parts are below the screen anyway)


@dataclass
class ViewState:
    ads: float = 0.0
    bob: float = 0.0
    moving: float = 0.0
    sprint: float = 0.0
    kick: float = 0.0
    flash: float = 0.0
    reload: float = -1.0  # 0..1 progress, <0 not reloading
    switch: float = 0.0  # 0..1 how far down
    knife: float = 0.0
    throw: float = 0.0
    drink: float = 0.0
    drink_color: tuple = (200, 40, 40)
    time: float = 0.0
    sway_x: float = 0.0
    sway_y: float = 0.0
    camo: bool = False
    tint: tuple = (255, 236, 214)
    crouch: float = 0.0
    name: str = ""


# --- images: decals and textures -----------------------------------------------------------------------------------

_IMAGES: dict = {}


def _img(w: int, h: int) -> QImage:
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    return img


def _from_array(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()


def image(key: str) -> QImage:
    img = _IMAGES.get(key)
    if img is None:
        img = _IMAGES[key] = _make_image(key)
    return img


def _make_image(key: str) -> QImage:  # noqa: C901 - one recipe per image
    if key.startswith("text:"):
        img = _img(320, 48)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        f = QFont("Bahnschrift")
        f.setPixelSize(30)
        f.setBold(True)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        p.setFont(f)
        p.setPen(QColor(230, 226, 214, 190))
        p.drawText(QRectF(4, 0, 316, 48), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, key[5:])
        p.end()
        return img
    rng = np.random.default_rng(sum(map(ord, key)))
    if key == "fabric":
        n = rng.random((64, 64))
        weave = ((np.arange(64)[:, None] + np.arange(64)[None, :]) % 4 < 2) * 0.12
        v = np.clip(200 + (n - 0.5) * 60 + weave * 255 - 20, 0, 255)
        arr = np.stack([v, v, v, np.full_like(v, 255)], -1)
        return _from_array(arr)

    if key == "pap":
        from .textures import value_noise

        a = value_noise(128, 4, 77, 4)
        b = value_noise(128, 8, 78, 3)
        h = (a * 2.0 + b) % 1.0
        r = 0.55 + 0.45 * np.sin(2 * np.pi * (h + 0.0))
        g = 0.25 + 0.35 * np.sin(2 * np.pi * (h + 0.35))
        bl = 0.75 + 0.25 * np.sin(2 * np.pi * (h + 0.7))
        streak = np.clip(1 - np.abs(((a * 6) % 1.0) - 0.5) * 8, 0, 1)
        rgb = np.stack([r * 200 + streak * 55, g * 120 + streak * 120, bl * 255], -1)
        arr = np.concatenate([np.clip(rgb, 0, 255), np.full((128, 128, 1), 255)], -1)
        return _from_array(arr)
    img = _img(128, 32)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    dark = QColor(12, 12, 14, 220)
    if key == "rail":
        for x in range(0, 128, 10):
            p.fillRect(QRectF(x, 0, 5, 32), dark)
            p.fillRect(QRectF(x + 5, 0, 1.5, 32), QColor(255, 255, 255, 50))
    elif key == "vents":
        for x in range(6, 124, 16):
            p.setBrush(dark)
            p.drawRoundedRect(QRectF(x, 8, 11, 16), 5, 5)
    elif key == "slots":
        for x in range(10, 120, 26):
            p.fillRect(QRectF(x, 4, 16, 8), dark)
            p.fillRect(QRectF(x, 20, 16, 8), dark)
    elif key == "holes":
        p.setBrush(dark)
        for x in range(8, 124, 14):
            for y in (8, 22):
                p.drawEllipse(QPointF(x + (y // 22) * 7, y), 4, 4)
    elif key == "grip_tex":
        p.setBrush(QColor(0, 0, 0, 120))
        for _ in range(260):
            p.drawEllipse(QPointF(rng.uniform(0, 128), rng.uniform(0, 32)), 0.9, 0.9)
    elif key == "serrations":
        for x in range(4, 128, 9):
            p.fillRect(QRectF(x, 0, 3, 32), dark)
            p.fillRect(QRectF(x + 3, 0, 1, 32), QColor(255, 255, 255, 60))
    elif key == "mag_ribs":
        for y in range(2, 32, 7):
            p.fillRect(QRectF(0, y, 128, 2), QColor(0, 0, 0, 120))
            p.fillRect(QRectF(0, y + 2, 128, 1), QColor(255, 255, 255, 40))
    elif key == "flutes":
        for x in range(0, 128, 21):
            p.fillRect(QRectF(x + 4, 0, 10, 32), QColor(0, 0, 0, 110))
    elif key == "port":
        p.fillRect(QRectF(20, 6, 70, 20), QColor(10, 10, 10, 230))
        p.fillRect(QRectF(26, 10, 40, 12), QColor(180, 140, 60, 220))
        p.fillRect(QRectF(20, 6, 70, 2), QColor(255, 255, 255, 60))
    elif key == "engrave":
        p.setPen(QPen(QColor(220, 200, 150, 140), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(4, 16)
        for k in range(8):
            path.cubicTo(8 + k * 15, 2, 14 + k * 15, 30, 20 + k * 15, 16)
        p.drawPath(path)
    elif key == "glass":
        g = QLinearGradient(0, 0, 128, 32)
        g.setColorAt(0, QColor(120, 200, 210, 50))
        g.setColorAt(0.45, QColor(220, 255, 255, 90))
        g.setColorAt(0.55, QColor(120, 200, 210, 40))
        g.setColorAt(1, QColor(80, 150, 160, 50))
        p.fillRect(QRectF(0, 0, 128, 32), g)
    elif key == "lens":
        p.end()
        img = _img(64, 64)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = QRadialGradient(QPointF(26, 24), 34)
        g.setColorAt(0, QColor(150, 110, 230))
        g.setColorAt(0.5, QColor(40, 60, 120))
        g.setColorAt(1, QColor(10, 14, 30))
        p.setBrush(g)
        p.setPen(QPen(QColor(10, 10, 10), 3))
        p.drawEllipse(QRectF(2, 2, 60, 60))
        p.setBrush(QColor(255, 255, 255, 170))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(16, 14, 12, 7))
    p.end()
    return img


def _camo() -> QImage:
    """An original grey-green digital camo (blocky pixels in four tones)."""
    rng = np.random.default_rng(11)
    from .textures import value_noise

    n = value_noise(64, 6, 12, 3)
    q = np.digitize(n + rng.random((64, 64)) * 0.08, [0.42, 0.52, 0.62])
    q = np.repeat(np.repeat(q[::4, ::4], 4, 0), 4, 1)
    pal = np.array([[96, 104, 86], [140, 146, 124], [70, 74, 62], [178, 176, 156]], float)
    rgb = pal[q]
    arr = np.concatenate([rgb, np.full((64, 64, 1), 255)], -1)
    return _from_array(arr)


def texture(key: str) -> QImage:
    if key == "camo":
        img = _IMAGES.get("camo")
        if img is None:
            img = _IMAGES["camo"] = _camo()
        return img
    return image(key)


# --- drawing -------------------------------------------------------------------------------------------------------

def _rot(yaw: float, pitch: float, roll: float):  # noqa: ANN202
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    def f(x: float, y: float, z: float) -> tuple[float, float, float]:
        x, y = x * cr - y * sr, x * sr + y * cr
        y, z = y * cp - z * sp, y * sp + z * cp
        x, z = x * cy + z * sy, -x * sy + z * cy
        return x, y, z
    return f


def _clamp(v: float) -> int:
    return int(0 if v < 0 else (255 if v > 255 else v))


def _matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    f = _rot(yaw, pitch, roll)
    return np.array([f(1, 0, 0), f(0, 1, 0), f(0, 0, 1)], np.float64)  # rows: the images of the axes


class _Mesh:
    """A model flattened into arrays: every face's own vertices, its normal, material and texture rect."""

    KINDS = {"glow": 1, "glowred": 1, "sleeve": 2, "decal": 3}

    def __init__(self, parts: list) -> None:
        tags = sorted({p.tag for p in parts})
        self.tags = tags
        ti = {t: i for i, t in enumerate(tags)}
        verts, vtag, start, count, normals, fpart, uv = [], [], [], [], [], [], []
        for pi, part in enumerate(parts):
            for face in part.faces:
                start.append(len(verts))
                count.append(len(face.pts))
                verts += face.pts
                vtag += [ti[part.tag]] * len(face.pts)
                normals.append(face.normal)
                fpart.append(pi)
                uv.append(face.uv)
        self.parts = parts
        self.verts = np.array(verts, np.float64)
        self.vtag = np.array(vtag, np.int32)
        self.start = np.array(start, np.int64)
        self.count = np.array(count, np.float64)
        self.normals = np.array(normals, np.float64)
        self.fpart = np.array(fpart, np.int32)
        self.uv = uv
        mats = [parts[i].mat for i in fpart]
        self.base = np.array([vm.MATS.get(m, vm.MATS["metal"])[0] if m != "decal" else (0, 0, 0) for m in mats],
                             np.float64)
        self.ks = np.array([vm.MATS.get(m, vm.MATS["metal"])[1] if m != "decal" else 0.0 for m in mats])
        self.shin = np.array([vm.MATS.get(m, vm.MATS["metal"])[2] if m != "decal" else 1.0 for m in mats])
        self.kind = np.array([self.KINDS.get(m, 4 if m in vm.GUN_MATS else 0) for m in mats], np.int32)
        self.bias = np.where(self.kind == 3, 0.002, 0.0)


_MESHES: dict = {}


def _mesh(parts: list) -> _Mesh:
    m = _MESHES.get(id(parts))
    if m is None or m.parts is not parts:
        m = _MESHES[id(parts)] = _Mesh(parts)
    return m


def _clip_near(poly3: np.ndarray, cx: float, cy: float, f: float) -> list:
    """A face cut at the near plane, projected."""
    out = []
    n = len(poly3)
    for i in range(n):
        a, b = poly3[i], poly3[(i + 1) % n]
        ina, inb = a[2] >= CLIP, b[2] >= CLIP
        if ina:
            out.append(a)
        if ina != inb:
            t = (CLIP - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * t)
    return [(cx + f * float(q[0]) / float(q[2]), cy - f * float(q[1]) / float(q[2])) for q in out]


def _affine(pts: list, rect: QRectF) -> QTransform:
    """Image space (``rect``) onto the parallelogram of a face's first, second and last corners."""
    (x0, y0), (x1, y1), (x3, y3) = pts[0], pts[1], pts[-1]
    w, h = max(rect.width(), 1e-6), max(rect.height(), 1e-6)
    m11, m12 = (x1 - x0) / w, (y1 - y0) / w
    m21, m22 = (x3 - x0) / h, (y3 - y0) / h
    dx = x0 - rect.x() * m11 - rect.y() * m21
    dy = y0 - rect.x() * m12 - rect.y() * m22
    return QTransform(m11, m12, m21, m22, dx, dy)


def draw_model(p: QPainter, w: int, h: int, parts: list, pos: tuple, rot: tuple, tint: tuple,
               camo_t: float | None = None, off: dict | None = None, flash: float = 0.0) -> None:
    mesh = _mesh(parts)
    f = h * 0.95
    cx, cy = w / 2, h / 2
    toff = np.zeros((len(mesh.tags), 3))
    for i, tag in enumerate(mesh.tags):
        if off and tag in off:
            toff[i] = off[tag]
    M = _matrix(*rot)
    vt = (mesh.verts + toff[mesh.vtag]) @ M + np.array(pos)
    z = vt[:, 2]
    zs = np.maximum(z, 1e-3)
    sx = cx + f * vt[:, 0] / zs
    sy = cy - f * vt[:, 1] / zs
    nrm = mesh.normals @ M
    cen = np.add.reduceat(vt, mesh.start, axis=0) / mesh.count[:, None]
    zmin = np.minimum.reduceat(z, mesh.start)
    zmax = np.maximum.reduceat(z, mesh.start)
    bw = np.maximum.reduceat(sx, mesh.start) - np.minimum.reduceat(sx, mesh.start)
    bh = np.maximum.reduceat(sy, mesh.start) - np.minimum.reduceat(sy, mesh.start)
    clipped = zmin < CLIP
    x0 = np.minimum.reduceat(sx, mesh.start)
    y0 = np.minimum.reduceat(sy, mesh.start)
    onscreen = (x0 < w) & (x0 + bw > 0) & (y0 < h) & (y0 + bh > 0)
    vis = (zmax > CLIP + 0.005) & ((nrm * cen).sum(1) < 0) & (((bw * bh > 4.0) & onscreen) | clipped)
    idx = np.flatnonzero(vis)
    if not len(idx):
        return
    n = nrm[idx]
    c = cen[idx]
    view = -c / np.linalg.norm(c, axis=1)[:, None]
    key = np.array(KEY)
    kd = np.clip(n @ key, 0, None)
    fd = np.clip(n @ np.array(FILL), 0, None)
    rim = (1.0 - np.clip((n * view).sum(1), 0, None)) ** 3 * 0.45
    hv = key[None, :] + view
    hv /= np.linalg.norm(hv, axis=1)[:, None]
    spec = mesh.ks[idx] * np.clip((n * hv).sum(1), 0, None) ** mesh.shin[idx] * 255
    tk = np.array([KEY_COL[i] * tint[i] / 255 for i in range(3)])
    fl = flash * 0.55 * np.maximum(0.3, -n[:, 2])
    lit = (np.array(AMBIENT)[None, :] + 0.95 * kd[:, None] * tk[None, :] + (0.5 * fd + rim)[:, None]
           * np.array(FILL_COL)[None, :] + fl[:, None] * np.array((1.0, 0.7, 0.35))[None, :])
    kind = mesh.kind[idx]
    col = np.where((kind == 1)[:, None], mesh.base[idx], mesh.base[idx] * lit + spec[:, None])
    top = np.clip(col * 1.14 + 6, 0, 255).astype(np.int32)
    bot = np.clip(col * 0.82, 0, 255).astype(np.int32)
    flat = np.clip(col, 0, 255).astype(np.int32)
    order = np.argsort(-(c[:, 2] - mesh.bias[idx]), kind="stable")
    big_camo: set = set()
    if camo_t is not None:
        cam = [int(i) for i in idx[kind == 4]]
        cam.sort(key=lambda i: -float(bw[i] * bh[i]))
        big_camo = set(cam[:4])
    outline = QPen(QColor(0, 0, 0, 120), 1.0)
    p.setPen(outline)
    starts = mesh.start
    counts = mesh.count
    for o in order:
        fi = int(idx[o])
        s0 = int(starts[fi])
        k = int(counts[fi])
        kk = int(kind[o])
        if clipped[fi]:
            pts = _clip_near(vt[s0:s0 + k], cx, cy, f)
            if len(pts) < 3 or min(x for x, _ in pts) > w or min(y for _, y in pts) > h or max(x for x, _ in pts) < 0:
                continue
            k = len(pts)
            if kk in (2, 3, 4):
                kk = 0  # no texture on a cut face
        else:
            pts = [(float(sx[s0 + j]), float(sy[s0 + j])) for j in range(k)]
        poly = QPolygonF([QPointF(x, y) for x, y in pts])
        if kk == 3:
            part = mesh.parts[mesh.fpart[fi]]
            img = image(part.decal)
            br = QBrush(img)
            br.setTransform(_affine(pts, QRectF(0, 0, img.width(), img.height())))
            p.setPen(Qt.PenStyle.NoPen)
            p.setOpacity(0.55 + 0.45 * float(kd[o]))
            p.setBrush(br)
            p.drawPolygon(poly)
            p.setOpacity(1.0)
            p.setPen(outline)
            continue
        if kk == 4 and camo_t is not None and fi not in big_camo:
            tt = camo_t * 0.35 + fi * 0.05
            g = QLinearGradient(poly.boundingRect().topLeft(), poly.boundingRect().bottomRight())
            lum = min(float(lit[o].mean()), 1.3)
            for j in range(3):
                hue = (0.68 + 0.16 * math.sin((tt + j * 0.33) * 2 * math.pi)) % 1.0
                g.setColorAt(j / 2, QColor.fromHsvF(hue, 0.8, min(0.25 + 0.6 * lum, 1.0)))
            p.setBrush(QBrush(g))
            p.drawPolygon(poly)
        elif kk == 2 or (kk == 4 and camo_t is not None):
            if kk == 2:
                img = texture("camo")
                u0, v0, du, dv = mesh.uv[fi]
                src = QRectF(u0 * img.width(), v0 * img.height(), du * img.width(), dv * img.height())
            else:
                img = texture("pap")
                tt = camo_t + fi * 0.013
                src = QRectF((tt * 22.0) % 80, (tt * 9.0 + 40) % 80, 48, 48)
            br = QBrush(img)
            br.setTransform(_affine(pts, src))
            p.setBrush(br)
            p.drawPolygon(poly)
            lum = float(lit[o].mean()) if kk == 4 else float((lit[o] * mesh.base[fi] / 150.0).mean())
            p.setPen(Qt.PenStyle.NoPen)
            if lum < 1.0:
                p.setBrush(QColor(8, 4, 14, _clamp((1.0 - lum) * (200 if kk == 4 else 230))))
            else:
                p.setBrush(QColor(255, 240, 220, _clamp((lum - 1.0) * 120)))
            p.drawPolygon(poly)
            p.setPen(outline)
        else:
            ys = [y for _, y in pts]
            y0, y1 = min(ys), max(ys)
            if y1 - y0 > 14:
                g = QLinearGradient(0, y0, 0, y1)
                g.setColorAt(0, QColor(*top[o]))
                g.setColorAt(1, QColor(*bot[o]))
                p.setBrush(QBrush(g))
            else:
                p.setBrush(QColor(*flat[o]))
            p.drawPolygon(poly)
        # a lit machined edge on hard surfaces
        if k == 4 and not clipped[fi] and mesh.ks[fi] >= 0.2 and kd[o] > 0.15:
            yv = vt[s0:s0 + 4, 1]
            best = max(range(4), key=lambda i: yv[i] + yv[(i + 1) % 4])
            a, b = pts[best], pts[(best + 1) % 4]
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 5:
                p.setPen(QPen(QColor(255, 236, 210, int(min(60 + 280 * float(kd[o]) * mesh.ks[fi], 210))), 1.2))
                p.drawLine(QPointF(*a), QPointF(*b))
                p.setPen(outline)

_CACHE: dict = {}


def model_for(kind: str, sight: str, name: str = "") -> vm.Model:
    key = (kind, sight, name)
    m = _CACHE.get(key)
    if m is None:
        m = _CACHE[key] = vm.build(kind, sight, name)
    return m


def draw_weapon(p: QPainter, w: int, h: int, model_kind: str, sight: str, s: ViewState) -> tuple[float, float] | None:
    """Draw the gun; returns the muzzle's screen point (for tracers), None when hidden by the scope."""
    if sight == "scope" and s.ads > 0.85:
        return None
    mdl = model_for(model_kind, sight, s.name)
    t = s.time
    ads = s.ads * s.ads * (3 - 2 * s.ads)
    hx, hy, hz = mdl.hip
    x = hx * (1 - ads)
    y = hy * (1 - ads)
    z = hz + (mdl.ads_z - hz) * ads
    calm = 1.0 - 0.88 * ads
    x += math.sin(t * 1.1) * 0.003 * calm + math.sin(s.bob) * 0.012 * s.moving * calm - s.sway_x * 0.02 * calm
    y += math.sin(t * 1.6) * 0.002 * calm - abs(math.cos(s.bob)) * 0.01 * s.moving * calm + s.sway_y * 0.015 * calm
    y -= s.crouch * 0.01 * calm
    small = model_kind in ("pistol", "revolver", "raygun")
    yaw = (-0.26 if small else -0.1) * (1 - ads) - s.sway_x * 0.04 * calm
    pitch = 0.0
    roll = 0.04 * (1 - ads)
    k = s.kick
    z -= k * 0.035 * (1 - 0.5 * ads)
    pitch += k * 0.09 * (1 - 0.6 * ads)
    y += k * 0.006
    sp = s.sprint
    x -= sp * 0.03
    y -= sp * 0.06
    yaw += sp * 0.55
    roll += sp * 0.45
    pitch -= sp * 0.25
    y -= s.switch * 0.25
    pitch -= s.switch * 0.6
    off: dict = {"slide": (0.0, 0.0, -0.03 * k), "bolt": (0.0, 0.0, -0.05 * k)}
    if 0.0 <= s.reload <= 1.0:
        r = s.reload
        tilt = math.sin(min(r * 1.2, 1.0) * math.pi) if r < 0.85 else 0.0
        roll -= 0.5 * tilt
        pitch += 0.12 * tilt
        yaw -= 0.15 * tilt
        y -= 0.02 * tilt
        if r < 0.3:
            mg = r / 0.3
        elif r < 0.6:
            mg = 1.0
        else:
            mg = max(1.0 - (r - 0.6) / 0.25, 0.0)
        off["mag"] = (0.0, -0.18 * mg, -0.02 * mg)
        off["hand_l"] = (0.02 * mg, -0.12 * mg, -0.12 * mg)
        off["sleeve_l"] = (0.02 * mg, -0.1 * mg, -0.1 * mg)
        if model_kind == "shotgun":
            pump = -0.07 * abs(math.sin(r * math.pi * 3))
            off["pump"] = (0.0, 0.0, pump)
            off["hand_l"] = (0.0, 0.0, pump)
            off["sleeve_l"] = (0.0, 0.0, pump)
            off["mag"] = (0.0, 0.0, 0.0)
        if model_kind == "sniper" and r > 0.75:
            off["bolt"] = (0.0, 0.0, -0.05 * math.sin((r - 0.75) / 0.25 * math.pi))
        if small:
            off["slide"] = (0.0, 0.0, -0.03 if r > 0.8 else 0.0)
    if model_kind == "shotgun" and s.kick > 0.2 and s.reload < 0:
        pump = -0.07 * math.sin(min((1 - s.kick) * 2.0, 1.0) * math.pi)
        off["pump"] = off["hand_l"] = off["sleeve_l"] = (0.0, 0.0, pump)
    draw_model(p, w, h, mdl.parts, (x, y, z), (yaw, pitch, roll), s.tint, s.time if s.camo else None, off,
               min(s.flash / 0.05, 1.0) if s.flash > 0 else 0.0)
    R = _rot(yaw, pitch, roll)
    mx, my, mz = R(*mdl.muzzle)
    mx, my, mz = mx + x, my + y, mz + z
    f = h * 0.95
    if mz <= NEAR:
        return None
    sx, sy = w / 2 + f * mx / mz, h / 2 - f * my / mz
    if s.flash > 0 and s.switch <= 0 and s.reload < 0:
        _flash(p, sx, sy, h * (0.055 + 0.035 * (1 - ads)) * (1.4 if model_kind in ("shotgun", "shotgun2", "sniper",
                                                                                   "launcher") else 1.0),
               model_kind, t)
    return sx, sy


def _flash(p: QPainter, x: float, y: float, r: float, kind: str, t: float) -> None:
    p.save()
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    green = kind == "raygun"
    p.setPen(Qt.PenStyle.NoPen)
    g = QRadialGradient(QPointF(x, y), r * 2.8)
    g.setColorAt(0, QColor(255, 250, 220, 230) if not green else QColor(200, 255, 200, 230))
    g.setColorAt(0.2, QColor(255, 180, 70, 170) if not green else QColor(80, 255, 120, 170))
    g.setColorAt(1, QColor(255, 110, 20, 0) if not green else QColor(20, 200, 60, 0))
    p.setBrush(g)
    p.drawEllipse(QPointF(x, y), r * 2.8, r * 2.8)
    rot = (t * 47.0) % (2 * math.pi)
    for layer, (scale, col) in enumerate(((1.0, QColor(255, 170, 60, 200)), (0.6, QColor(255, 240, 190, 230)))):
        path = QPainterPath()
        n = 5 + layer
        for i in range(n * 2):
            a = rot + layer * 0.4 + i * math.pi / n
            rr = r * scale * ((1.9 if i % 4 == 0 else 1.4) if i % 2 == 0 else 0.35)
            pt = QPointF(x + math.cos(a) * rr, y + math.sin(a) * rr * 0.75)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        path.closeSubpath()
        p.setBrush(col if not green else QColor(150, 255, 160, 210))
        p.drawPath(path)
    p.setBrush(QColor(255, 255, 255, 240))
    p.drawEllipse(QPointF(x, y), r * 0.35, r * 0.28)
    p.restore()


# --- other first-person actions ------------------------------------------------------------------------------------

def _knife_parts() -> list:
    return ([vm.beam((0, 0, 0.0), (0, 0, 0.17), 0.004, 0.028, "chrome"),
             vm.beam((0, 0.006, 0.02), (0, 0.006, 0.15), 0.005, 0.01, "metal"),
             vm.box(-0.014, 0.014, -0.022, 0.022, -0.012, 0.0, "dark"),
             vm.beam((0, 0, -0.012), (0, -0.005, -0.12), 0.024, 0.03, "poly")]
            + vm.right_hand(0.0, 0.025, -0.06, 0.0))


def draw_knife(p: QPainter, w: int, h: int, k: float, tint: tuple) -> None:
    """The knife slash (``k`` 0..1)."""
    if "knife" not in _CACHE:
        _CACHE["knife"] = _knife_parts()
    a = math.sin(k * math.pi)
    pos = (0.22 - 0.36 * k, -0.15 + 0.08 * a, 0.3)
    draw_model(p, w, h, _CACHE["knife"], pos, (-0.9 + 1.5 * k, 0.15 * a, -0.6 + 0.9 * k), tint)


def draw_throw(p: QPainter, w: int, h: int, k: float, tint: tuple) -> None:
    if "nade" not in _CACHE:
        _CACHE["nade"] = ([vm.cyl((0, -0.03, 0.0), (0, 0.03, 0.0), 0.028, "olive", n=10),
                           vm.box(-0.006, 0.006, 0.03, 0.045, -0.01, 0.01, "metal"),
                           vm.beam((0.006, 0.04, 0.0), (0.03, -0.02, -0.01), 0.008, 0.004, "metal")]
                          + vm.right_hand(0.0, 0.0, -0.03, 0.0))
    a = math.sin(min(k, 1) * math.pi)
    pos = (0.12 - 0.08 * k, -0.2 + 0.2 * a, 0.3 + 0.15 * k)
    draw_model(p, w, h, _CACHE["nade"], pos, (-0.2, 0.4 * a, -0.3), tint)


def draw_drink(p: QPainter, w: int, h: int, k: float, color: tuple, tint: tuple) -> None:
    """Drinking a tonic (``k`` 0..1)."""
    key = ("bottle", color)
    if key not in _CACHE:
        vm.MATS[f"soda{color}"] = (color, 0.7, 30, "")
        _CACHE[key] = ([vm.cyl((0, -0.07, 0), (0, 0.06, 0), 0.026, f"soda{color}", n=10),
                        vm.cyl((0, 0.06, 0), (0, 0.1, 0), 0.026, f"soda{color}", n=10, r2=0.01),
                        vm.cyl((0, 0.1, 0), (0, 0.115, 0), 0.011, "metal", n=8)]
                       + vm.right_hand(0.0, 0.03, -0.01, 0.0))
    up = math.sin(min(k, 1) * math.pi)
    pos = (0.08 - 0.06 * up, -0.26 + 0.18 * up, 0.28)
    draw_model(p, w, h, _CACHE[key], pos, (-0.2, -0.9 * up, -0.3 * up), tint)


def draw_scope_overlay(p: QPainter, w: int, h: int, t: float, sway: tuple[float, float]) -> None:
    """The sniper scope seen through: black around a lens with fine cross hairs and a mil-dot ladder."""
    r = h * 0.46
    cx, cy = w / 2 + sway[0], h / 2 + sway[1]
    path = QPainterPath()
    path.addRect(QRectF(0, 0, w, h))
    lens = QPainterPath()
    lens.addEllipse(QPointF(cx, cy), r, r)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(0, 0, 0))
    p.drawPath(path.subtracted(lens))
    g = QRadialGradient(QPointF(cx, cy), r)
    g.setColorAt(0.72, QColor(0, 0, 0, 0))
    g.setColorAt(0.93, QColor(0, 0, 0, 160))
    g.setColorAt(1.0, QColor(0, 0, 0, 255))
    p.setBrush(g)
    p.drawEllipse(QPointF(cx, cy), r, r)
    p.save()
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    gl = QRadialGradient(QPointF(cx - r * 0.45, cy - r * 0.5), r * 0.5)
    gl.setColorAt(0, QColor(60, 50, 90, 60))
    gl.setColorAt(1, QColor(0, 0, 0, 0))
    p.setBrush(gl)
    p.drawEllipse(QPointF(cx - r * 0.45, cy - r * 0.5), r * 0.5, r * 0.5)
    p.restore()
    p.setPen(QPen(QColor(0, 0, 0), 1.4))
    p.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))
    p.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))
    p.setPen(QPen(QColor(0, 0, 0), 5))
    for a, b in (((cx - r, cy), (cx - r * 0.35, cy)), ((cx + r * 0.35, cy), (cx + r, cy)),
                 ((cx, cy + r * 0.35), (cx, cy + r))):
        p.drawLine(QPointF(*a), QPointF(*b))
    p.setPen(QPen(QColor(0, 0, 0), 1))
    for k in range(1, 6):
        d = r * 0.06 * k
        p.drawLine(QPointF(cx - 4, cy + d), QPointF(cx + 4, cy + d))
        p.drawLine(QPointF(cx - d, cy - 3), QPointF(cx - d, cy + 3))
        p.drawLine(QPointF(cx + d, cy - 3), QPointF(cx + d, cy + 3))
    p.setPen(QPen(QColor(255, 40, 30), 1))
    p.setBrush(QColor(255, 40, 30))
    p.drawEllipse(QPointF(cx, cy), 1.6, 1.6)


def draw_red_dot(p: QPainter, x: float, y: float, h: int, acog: bool = False) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    if acog:
        p.setPen(QPen(QColor(255, 60, 40, 230), 2))
        s = h * 0.012
        p.drawLine(QPointF(x - s, y + s), QPointF(x, y))
        p.drawLine(QPointF(x, y), QPointF(x + s, y + s))
        return
    g = QRadialGradient(QPointF(x, y), h * 0.007)
    g.setColorAt(0, QColor(255, 120, 100, 255))
    g.setColorAt(0.4, QColor(255, 40, 20, 230))
    g.setColorAt(1, QColor(255, 20, 10, 0))
    p.setBrush(g)
    p.drawEllipse(QPointF(x, y), h * 0.007, h * 0.007)
