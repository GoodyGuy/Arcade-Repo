"""Case Clash's art: side-view weapon silhouettes (QPainter paths), procedural skin finishes (numpy) clipped to
them, wear scuffs, the weapon case art, and caches of the rendered pixmaps."""

from __future__ import annotations

import math
import re
from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient, QTransform)

from app.arcade.games.case_clash import data as D
from app.arcade.games.case_clash.shapes import SHAPES

# --- sharp at any display scaling ----------------------------------------------------------------------------------

_DPR = [0.0]  # device pixels per logical pixel the cached pictures are made for (0: not known yet)


def set_dpr(ratio: float) -> None:
    """The widget's device pixel ratio (the game sets it before painting): pictures are made at that many pixels."""
    _DPR[0] = max(float(ratio), 1.0)


def dpr() -> float:
    if not _DPR[0]:
        from PySide6.QtGui import QGuiApplication

        app = QGuiApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        _DPR[0] = max(float(screen.devicePixelRatio()) if screen is not None else 1.0, 1.0)
    return _DPR[0]


def _device(n: float, d: float) -> int:
    return max(math.ceil(n * d - 1e-6), 1)


def new_pixmap(w: float, h: float) -> QPixmap:
    """A pixmap of w x h logical pixels with as many device pixels as the screen shows (painted in logical units)."""
    d = dpr()
    pm = QPixmap(_device(w, d), _device(h, d))
    pm.setDevicePixelRatio(d)
    return pm


def new_image(w: float, h: float) -> QImage:
    """A transparent image of w x h logical pixels at the screen's pixel density."""
    d = dpr()
    img = QImage(_device(w, d), _device(h, d), QImage.Format.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(d)
    img.fill(Qt.GlobalColor.transparent)
    return img


def lsize(pm: QPixmap) -> tuple[float, float]:
    """A pixmap's size in logical pixels."""
    d = pm.devicePixelRatio() or 1.0
    return pm.width() / d, pm.height() / d


# --- silhouettes ---------------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"[MLQCZOW]|-?\d+(?:\.\d+)?")


def _parse(spec: str) -> QPainterPath:
    path = QPainterPath()
    tok = _TOKEN.findall(spec)
    i = 0
    if tok and tok[0] == "W":
        path.setFillRule(Qt.FillRule.WindingFill)
        i = 1

    def num(k: int) -> list[float]:
        return [float(tok[k]), float(tok[k + 1])]

    while i < len(tok):
        c = tok[i]
        if c == "M":
            path.moveTo(*num(i + 1))
            i += 3
        elif c == "L":
            path.lineTo(*num(i + 1))
            i += 3
        elif c == "Q":
            path.quadTo(*num(i + 1), *num(i + 3))
            i += 5
        elif c == "C":
            path.cubicTo(*num(i + 1), *num(i + 3), *num(i + 5))
            i += 7
        elif c == "Z":
            path.closeSubpath()
            i += 1
        elif c == "O":
            cx, cy, r = float(tok[i + 1]), float(tok[i + 2]), float(tok[i + 3])
            path.addEllipse(QPointF(cx, cy), r, r)
            i += 4
        else:  # a bare number after L: another point
            path.lineTo(*num(i))
            i += 2
    return path


_shape_cache: dict[tuple[str, bool], tuple[list[tuple[str, QPainterPath]], QRectF]] = {}


def shape(weapon: str, flip: bool = False) -> tuple[list[tuple[str, QPainterPath]], QRectF]:
    """The weapon's parts (muzzle / tip to the right; to the left when ``flip``) and their bounds."""
    got = _shape_cache.get((weapon, flip))
    if got is None:
        t = QTransform().scale(-1 if flip else 1, 1)
        parts = [(kind, t.map(_parse(spec))) for kind, spec in SHAPES.get(weapon, SHAPES["ak47"])]
        bounds = QRectF()
        for _k, path in parts:
            bounds = bounds.united(path.boundingRect())
        got = _shape_cache[(weapon, flip)] = (parts, bounds)
    return got


# --- procedural finishes -------------------------------------------------------------------------------------------

def _col(hx: str) -> np.ndarray:
    hx = hx.lstrip("#")
    if len(hx) == 3:
        hx = "".join(c * 2 for c in hx)
    return np.array([int(hx[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _noise(rng: np.random.Generator, h: int, w: int, cx: float, cy: float) -> np.ndarray:
    """Smooth value noise in [0, 1] with about ``cx`` x ``cy`` cells."""
    cx, cy = max(int(cx), 1), max(int(cy), 1)
    g = rng.random((cy + 2, cx + 2))
    ys = np.linspace(0, cy, h, endpoint=False)
    xs = np.linspace(0, cx, w, endpoint=False)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = ys - y0, xs - x0
    fy, fx = (fy * fy * (3 - 2 * fy))[:, None], fx * fx * (3 - 2 * fx)
    a, b = g[y0][:, x0], g[y0][:, x0 + 1]
    c, d = g[y0 + 1][:, x0], g[y0 + 1][:, x0 + 1]
    top, bot = a + (b - a) * fx, c + (d - c) * fx
    return top + (bot - top) * fy


def _fbm(rng: np.random.Generator, h: int, w: int, cx: float, cy: float, octaves: int = 4) -> np.ndarray:
    out, amp, tot = np.zeros((h, w)), 1.0, 0.0
    for o in range(octaves):
        out += amp * _noise(rng, h, w, cx * 2 ** o, cy * 2 ** o)
        tot += amp
        amp *= 0.5
    return out / tot


def _lerp(c0: np.ndarray, c1: np.ndarray, t: np.ndarray) -> np.ndarray:
    return c0 + (c1 - c0) * t[..., None]


def _ramp(t: np.ndarray, cols: list[np.ndarray]) -> np.ndarray:
    """Piecewise-linear colour ramp over t in [0, 1]."""
    t = np.clip(t, 0.0, 1.0) * (len(cols) - 1)
    i = np.minimum(t.astype(int), len(cols) - 2)
    f = t - i
    arr = np.array(cols)
    return arr[i] + (arr[i + 1] - arr[i]) * f[..., None]


def _levels(v: np.ndarray, cols: list[np.ndarray], qs: tuple[float, ...]) -> np.ndarray:
    idx = np.zeros(v.shape, dtype=int)
    for q in np.quantile(v, qs):
        idx += v > q
    return np.array(cols)[np.minimum(idx, len(cols) - 1)]


def _smooth(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


DOPPLER = {"Phase 1": ("#0b0710", "#3b1a49", "#c2408f"), "Phase 2": ("#1e0918", "#d6337f", "#ff86c0"),
           "Phase 3": ("#0a1916", "#2f6f5a", "#5fb3d8"), "Phase 4": ("#0a0f2a", "#2f5fe0", "#d65fbf"),
           "Garnet": ("#1a0204", "#8f0c14", "#ff2a3a"), "Azurite": ("#020822", "#1840c8", "#6fb0ff"),
           "Onyx": ("#040305", "#261634", "#6a4a8a")}


def finish_rgb(skin: D.Skin, seed: int, flt: float, w: int, h: int) -> np.ndarray:
    """The skin's finish as an (h, w, 3) float RGB array for a weapon spanning w x h pixels."""
    rng = np.random.default_rng(int(D._h("pat", skin.id, seed) * 2 ** 32))
    c = [_col(x) for x in skin.colors] + [_col("#000000")] * 4
    a = w / max(h, 1)
    xs = np.linspace(0.0, 1.0, w)[None, :]
    ys = np.linspace(0.0, 1.0, h)[:, None]
    X, Y = np.broadcast_to(xs, (h, w)), np.broadcast_to(ys, (h, w))
    px, py = X * w, Y * h
    ph = (seed % 97) / 97.0 * 6.283
    f = skin.finish
    if f == "camo":
        v = _fbm(rng, h, w, 4 * a + 2, 4, 3) * 0.75 + _fbm(rng, h, w, 9 * a, 8, 2) * 0.25
        out = _levels(v, c[:4], (0.3, 0.55, 0.8))
    elif f == "digital":
        b = max(3, h // 16)
        hs, ws = h // b + 2, w // b + 2
        v = _fbm(rng, hs, ws, 3 * a + 2, 3, 2) * 0.8 + rng.random((hs, ws)) * 0.2
        small = _levels(v, c[:4], (0.3, 0.55, 0.8))
        out = np.repeat(np.repeat(small, b, 0), b, 1)[:h, :w]
    elif f == "hydro":
        n = _fbm(rng, h, w, 3 * a + 2, 3, 3)
        v = np.sin(X * a * 5 + Y * 2.5 + n * 6 + ph)
        out = _ramp((v + 1) / 2, [c[0], c[1], c[2], c[1]])
    elif f == "tiles":
        s = h / 4.5
        ix, iy = (px + seed) // s, py // s
        fx, fy = (px + seed) % s / s, py % s / s
        k = ((ix + iy * 2) % 3).astype(int)
        base = np.array(c[:3])[k]
        inner = np.array(c[:3])[(k + 1) % 3]
        dia = (np.abs(fx - 0.5) + np.abs(fy - 0.5)) < 0.32
        out = np.where(dia[..., None], inner, base)
        grout = np.minimum(np.minimum(fx, fy), np.minimum(1 - fx, 1 - fy)) < 0.06
        out = np.where(grout[..., None], np.array([0.85, 0.8, 0.68]), out)
    elif f == "anodized":
        out = _lerp(c[0], c[1], _smooth(Y * 1.3 - 0.1))
        out += np.exp(-((Y - 0.28) / 0.09) ** 2)[..., None] * (c[2] - out) * 0.75
        out *= (0.94 + 0.06 * rng.random((h, 1, 1)))
    elif f == "graphic":
        v = _fbm(rng, h, w, 2.5 * a + 1, 2.5, 3) * 0.75 + _fbm(rng, h, w, 6 * a, 5, 2) * 0.25
        qs = np.quantile(v, (0.35, 0.6, 0.82))
        out = _levels(v, [c[0], c[2], c[3], c[0] * 0.55 + c[2] * 0.45], (0.35, 0.6, 0.82))
        edge = np.min(np.abs(v[..., None] - qs[None, None, :]), axis=2) < 0.008
        stripe = (((X * a * 2.5 + Y * 1.2 + ph) % 1.0) < 0.1) & (v > qs[1])
        out = np.where((edge | stripe)[..., None], c[1], out)
        eye = ((px - w * (0.25 + 0.1 * math.sin(ph))) ** 2 + (py - h * 0.42) ** 2) < (h * 0.13) ** 2
        out = np.where(eye[..., None], c[3] * 0.4 + 0.6, out)
        pupil = ((px - w * (0.25 + 0.1 * math.sin(ph))) ** 2 + (py - h * 0.42) ** 2) < (h * 0.06) ** 2
        out = np.where(pupil[..., None], c[1], out)
    elif f == "fade":
        pct = D.fade_percent(seed)
        t = np.clip(X * 1.25 - 0.2 + (pct - 90) / 60 + (Y - 0.5) * 0.15, 0, 1)
        out = _ramp(t, [c[0], c[1], c[2]])
        out += np.exp(-((Y - 0.3) / 0.12) ** 2)[..., None] * 0.12
    elif f == "snake":
        s = h / 7.0
        row = np.floor(py / s)
        du = ((px / s + (row % 2) * 0.5) % 1.0) - 0.5
        dv = (py / s % 1.0) - 0.5
        d = np.sqrt(du * du + dv * dv)
        n = _fbm(rng, h, w, 3 * a, 3, 2)
        out = _lerp(c[0], c[2], n) * (1.05 - d * 0.6)[..., None]
        dia = (np.abs((X * a * 1.4 + ph) % 1.0 - 0.5) * 1.6 + np.abs(Y - 0.5)) < 0.32
        out = np.where(dia[..., None], out * 0.45 + c[1] * 0.55, out)
        out = np.where((d > 0.47)[..., None], c[1], out)
    elif f == "hardened":
        b = D.blue_share(seed)
        n1 = _fbm(rng, h, w, 4 * a + 2, 4, 4)
        n2 = _fbm(rng, h, w, 10 * a, 9, 2)
        v = n1 * 0.8 + n2 * 0.2
        thr = np.quantile(v, 1.0 - b)
        tb = _smooth((v - thr) / 0.03 + 0.5)
        gold = _lerp(c[1] * 0.7, np.minimum(c[1] * 1.25, 1.0), n2)
        purple = (n2 > 0.6) & (v < thr - 0.02)
        gold = np.where(purple[..., None], gold * 0.4 + c[2] * 0.6, gold)
        blue = _lerp(c[0] * 0.55, np.minimum(c[0] * 1.35, 1.0), n2)
        out = gold + (blue - gold) * tb[..., None]
    elif f == "scifi":
        sh = (D._h("sf", seed) - 0.5) * 0.06
        out = _lerp(c[0] * 0.92, c[0], 1 - Y)
        orange = ((X > 0.06 + sh) & (X < 0.3 + sh)) | ((X > 0.63 + sh) & (X < 0.86 + sh) & (Y > 0.34))
        black = (X > 0.37 + sh + Y * 0.05) & (X < 0.56 + sh - Y * 0.06) & (Y > 0.2)
        out = np.where(orange[..., None], _lerp(c[1], c[1] * 0.8, Y), out)
        out = np.where(black[..., None], c[2], out)
        ticks = orange & (Y < 0.32) & ((px // 4) % 3 == 0)
        out = np.where(ticks[..., None], c[2], out)
        for lx in (0.33, 0.6, 0.9):
            out = np.where((np.abs(X - lx - sh) * w < 0.8)[..., None], out * 0.55, out)
        out = np.where(((np.abs(Y - 0.5) * h < 0.6) & ~orange & ~black)[..., None], out * 0.7, out)
    elif f == "stripes":
        d = (X * a + Y * 0.9) * 2.2 + ph
        out = _lerp(c[0], c[0] * 1.25, 1 - Y)
        out = np.where(((d % 1.0) < 0.2)[..., None], c[1], out)
        out = np.where((np.abs(d % 1.0 - 0.32) < 0.025)[..., None], c[2], out)
    elif f == "redline":
        chk = (((px // 2) + (py // 2)) % 2)[..., None]
        out = c[0] * (0.9 + 0.2 * chk)
        out = np.where((np.abs(Y - 0.36) < 0.04)[..., None], c[1], out)
        out = np.where((np.abs(Y - 0.45) < 0.014)[..., None], c[2], out)
        wedge = (X > 0.72) & (((X * a * 1.8 - Y * 1.1) % 1.0) < 0.22) & (Y > 0.4)
        out = np.where(wedge[..., None], c[1], out)
        wedge2 = (X < 0.25) & (((X * a * 2.0 + Y) % 1.0) < 0.12)
        out = np.where(wedge2[..., None], c[1] * 0.85, out)
    elif f == "halftone":
        n = _fbm(rng, h, w, 2.5 * a + 1, 2.5, 3)
        cell = max(4.0, h / 14)
        dx, dy = (px % cell) / cell - 0.5, (py % cell) / cell - 0.5
        r = np.sqrt(dx * dx + dy * dy)
        lo, hi = np.quantile(n, (0.42, 0.62))
        ink = (n > hi) | ((n > lo) & (r < (n - lo) / (hi - lo) * 0.55))
        out = np.where(ink[..., None], c[1], np.broadcast_to(c[0], (h, w, 3)))
        red = _fbm(rng, h, w, 3 * a, 3, 2) > 0.66
        out = np.where((red & ~ink)[..., None], c[2], out)
    elif f == "circuit":
        cs = max(5.0, h / 9)
        ix, iy = (px // cs).astype(int), (py // cs).astype(int)
        grid = rng.random((iy.max() + 2, ix.max() + 2))
        r = grid[iy, ix]
        fx, fy = (px % cs) / cs - 0.5, (py % cs) / cs - 0.5
        line = ((r < 0.5) & (np.abs(fy) < 0.09)) | ((r > 0.4) & (r < 0.85) & (np.abs(fx) < 0.09))
        node = (r > 0.8) & (fx * fx + fy * fy < 0.05)
        glow = ((r < 0.5) & (np.abs(fy) < 0.25)) | ((r > 0.4) & (r < 0.85) & (np.abs(fx) < 0.25))
        out = c[0] * (0.85 + 0.15 * _noise(rng, h, w, 20 * a, 20))[..., None]
        out = np.where(glow[..., None], out * 0.6 + c[1] * 0.35, out)
        out = np.where((line | node)[..., None], c[1], out)
    elif f == "carbon":
        s = max(3.0, h / 12)
        ix, iy = px // s, py // s
        fx, fy = (px % s) / s, (py % s) / s
        wv = np.where(((ix + iy) % 2) == 0, np.sin(np.pi * fx), np.sin(np.pi * fy))
        out = _lerp(c[1], c[0], wv)
        out += np.exp(-((Y - 0.3) / 0.15) ** 2)[..., None] * (c[2] - out) * 0.35
    elif f == "dragon":
        s = h / 8.0
        row = np.floor(py / s)
        du = ((px / s + (row % 2) * 0.5) % 1.0) - 0.5
        dv = (py / s % 1.0) - 0.5
        d = np.sqrt(du * du + dv * dv)
        base = _ramp(X, [c[1], c[0], c[1]])
        yc = 0.5 + 0.28 * np.sin(X * a * 2.2 + ph)
        band = np.abs(Y - yc) < 0.14
        out = np.where(band[..., None], c[3] * (1.1 - d * 0.7)[..., None], base * (1.05 - d * 0.5)[..., None])
        rim = band & (d > 0.42)
        out = np.where(rim[..., None], c[2], out)
        edge = np.abs(np.abs(Y - yc) - 0.14) < 0.012
        out = np.where(edge[..., None], c[2], out)
    elif f == "marble":
        n = _fbm(rng, h, w, 3 * a, 3, 5)
        v = np.sin(X * a * 4 + Y * 3 + n * 9 + ph)
        cols = [c[0], c[1], c[2], c[0]]
        if skin.special and D.frost_and_flame(seed):
            cols = [c[0], c[0] * 0.5 + c[2] * 0.2, c[2], c[0]]
        out = _ramp((v + 1) / 2, cols)
        out = np.where((np.abs(v) < 0.05)[..., None], out * 0.45, out)
    elif f == "web":
        cx, cy = w * (0.2 + 0.6 * D._h("wx", seed)), h * (0.2 + 0.6 * D._h("wy", seed))
        dx, dy = (px - cx) / h, (py - cy) / h
        ang = np.arctan2(dy, dx)
        r = np.sqrt(dx * dx + dy * dy)
        k = 14
        da = np.abs(((ang * k / (2 * np.pi)) % 1.0) - 0.5) * (2 * np.pi / k) * r * h
        spoke = np.abs(da - (np.pi / k) * r * h) < 0.7
        rr = r * 6 + 0.18 * np.cos(ang * k)
        ring = np.abs(rr % 1.0 - 0.5) * h / 6 > (0.5 / 6) * h - 0.7
        n = _fbm(rng, h, w, 3 * a, 3, 2)
        out = _lerp(c[0] * 0.7, c[0] * 1.15, n)
        out = np.where((spoke | ring)[..., None], c[1], out)
    elif f == "slaughter":
        n = _fbm(rng, h, w, 3 * a + 1, 3, 5)
        m = np.abs(_fbm(rng, h, w, 5 * a, 5, 3) - 0.5) * 2
        out = _ramp(n * 0.7 + (1 - m) * 0.45 - 0.1, [c[0] * 0.65, c[0], c[1], c[2]])
    elif f == "doppler":
        cols = [_col(x) for x in DOPPLER[D.doppler_phase(seed)]]
        n = _fbm(rng, h, w, 4 * a + 1, 4, 5)
        m = _fbm(rng, h, w, 14 * a, 12, 2)
        out = _ramp(n * 1.5 - 0.25 + (m - 0.5) * 0.5, cols)
        out += (rng.random((h, w)) > 0.985)[..., None] * 0.25
    elif f == "tiger":
        n = _fbm(rng, h, w, 2 * a, 2, 3)
        s = np.sin(X * a * 9 + n * 7 + Y * 2 + ph)
        out = _lerp(c[0] * 0.75, np.minimum(c[0] * 1.2, 1.0), 1 - Y)
        out = np.where((s > 0.55 + 0.35 * np.abs(Y - 0.5))[..., None], c[1], out)
    elif f == "damascus":
        n = _fbm(rng, h, w, 3 * a, 3, 3)
        v = np.sin(Y * 30 + n * 12 + X * a * 3)
        out = _lerp(c[1], c[0], (v + 1) / 2)
    elif f == "leather":
        grain = _noise(rng, h, w, 30 * a, 30)
        out = c[0] * (0.88 + 0.18 * grain)[..., None]
        band = (np.abs(X - 0.35) < 0.07) | (np.abs(Y - 0.62) < 0.05)
        out = np.where(band[..., None], c[2], out)
        out = np.where((Y > 0.8)[..., None], c[1], out)
        stitch = ((px // 3) % 2 == 0) & ((np.abs(Y - 0.55) * h < 0.6) | (np.abs(Y - 0.69) * h < 0.6))
        out = np.where(stitch[..., None], np.array([0.92, 0.9, 0.85]), out)
    elif f == "vanilla":
        out = _lerp(c[1], c[0], _smooth(1.2 - Y * 1.1))
        out *= (0.95 + 0.06 * rng.random((h, 1, 1)))
        out += np.exp(-((Y - 0.3) / 0.08) ** 2)[..., None] * 0.18
    else:  # solid
        out = _lerp(c[0], c[0] * 0.8, Y)
        out = np.where((Y > 0.66)[..., None], c[1] * 0.85 + out * 0.15, out)
        out = np.where((np.abs(Y - 0.64) * h < 0.8)[..., None], c[2], out)
        out *= (0.96 + 0.06 * _noise(rng, h, w, 25 * a, 25))[..., None]
    out = np.asarray(out, dtype=float)
    # wear: paint scuffed off to bare metal, and dirt
    cover = float(np.clip((flt - 0.07) * 0.55, 0.0, 0.36))
    if cover > 0.0 and skin.finish not in ("vanilla", "damascus"):
        wr = np.random.default_rng(int(D._h("wear", skin.id, seed) * 2 ** 32))
        edge = np.abs(Y - 0.5) * 2.0  # paint wears first along the top and bottom edges
        v = _fbm(wr, h, w, 26 * a, 26, 2) * 0.6 + _fbm(wr, h, w, 4 * a, 4, 2) * 0.25 + edge * 0.15
        thr = np.quantile(v, 1.0 - cover)
        mask = np.clip((v - thr) / 0.015, 0.0, 1.0)[..., None]
        metal = (0.4 + 0.16 * wr.random((h, w)))[..., None] * np.array([1.0, 1.0, 1.04])
        out = out * (1 - mask) + metal * mask
        out *= 1.0 - 0.22 * min(flt, 1.0)
    return np.clip(out, 0.0, 1.0)


def _to_image(rgb: np.ndarray) -> QImage:
    arr = np.ascontiguousarray((rgb * 255).astype(np.uint8))
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_RGB888)
    return img.copy()


# --- rendering and caches ------------------------------------------------------------------------------------------

class _LRU(OrderedDict):
    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap = cap

    def get_or(self, key, make):  # noqa: ANN001, ANN201
        if key in self:
            self.move_to_end(key)
            return self[key]
        val = self[key] = make()
        if len(self) > self.cap:
            self.popitem(last=False)
        return val


_art = _LRU(420)  # tile-sized renders
_big = _LRU(10)  # inspect / reveal / view model renders
_misc = _LRU(160)

_DARK = (QColor("#4a5058"), QColor("#1b1e22"))
_STEEL = (QColor("#d5dbe1"), QColor("#5d656d"))
_HANDLE = (QColor("#3a3532"), QColor("#121112"))


def _part_brush(kind: str, r: QRectF) -> QBrush:
    top, bot = {"d": _DARK, "m": _STEEL}.get(kind, _HANDLE)
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    g.setColorAt(0.0, top)
    g.setColorAt(1.0, bot)
    return QBrush(g)


def render_weapon(p: QPainter, rect: QRectF, skin: D.Skin, seed: int, flt: float, flip: bool = False) -> None:
    """Draw the skinned weapon fitted into ``rect`` (uncached)."""
    parts, b = shape(skin.weapon, flip)
    if b.width() <= 0 or b.height() <= 0:
        return
    s = min(rect.width() / b.width(), rect.height() / b.height())
    tx = rect.center().x() - b.center().x() * s
    ty = rect.center().y() - b.center().y() * s
    t = QTransform(s, 0, 0, s, tx, ty)
    fb = t.mapRect(b)
    dev = p.device()
    d = dev.devicePixelRatio() if dev is not None else 1.0  # the finish at the pixels it is shown with
    fw, fh = max(int(fb.width() * d) + 1, 2), max(int(fb.height() * d) + 1, 2)
    fin = _to_image(finish_rgb(skin, seed, flt, fw, fh))
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    outline = QPen(QColor(0, 0, 0, 170), max(0.8, s * 0.35))
    for kind, path in parts:
        tp = t.map(path)
        if kind == "s":  # a groove / serration / seam: a soft dark line, nothing else
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 95))
            p.drawPath(tp)
            continue
        if kind == "p" or (kind == "h" and skin.kind == "gloves"):
            p.save()
            p.setClipPath(tp)
            if kind == "h":
                p.fillRect(fb, QColor(skin.colors[1]).darker(160))
            else:
                p.drawImage(fb, fin)
            p.restore()
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_part_brush(kind, tp.boundingRect()))
            p.drawPath(tp)
        # shading: each part lit from above
        pb = tp.boundingRect()
        p.save()
        p.setClipPath(tp)
        g = QLinearGradient(pb.topLeft(), pb.bottomLeft())
        g.setColorAt(0.0, QColor(255, 255, 255, 75))
        g.setColorAt(0.3, QColor(255, 255, 255, 0))
        g.setColorAt(0.65, QColor(0, 0, 0, 0))
        g.setColorAt(1.0, QColor(0, 0, 0, 85))
        p.fillRect(pb, g)
        p.restore()
        p.setPen(outline)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(tp)
    p.restore()


def art_pixmap(skin_id: str, seed: int, flt: float, w: int, h: int, flip: bool = False) -> QPixmap:
    """The weapon render on a transparent pixmap (cached; the float is bucketed for the cache)."""
    w, h = max(int(w), 8), max(int(h), 8)
    fb = round(flt / 0.02) * 0.02

    def make() -> QPixmap:
        img = new_image(w, h)
        p = QPainter(img)
        render_weapon(p, QRectF(0, 0, w, h), D.skin(skin_id), seed, fb, flip)
        p.end()
        return QPixmap.fromImage(img)

    return (_art if w * h < 160_000 else _big).get_or((skin_id, seed, fb, w, h, flip, dpr()), make)


def tile_bg(rarity: int, w: int, h: int, glow: float = 1.0) -> QPixmap:
    """The Panorama item tile background: dark gradient, the rarity colour glowing from the bottom and its bar."""
    w, h = max(int(w), 8), max(int(h), 8)

    def make() -> QPixmap:
        pm = new_pixmap(w, h)
        pm.fill(QColor("#1c2128"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = QLinearGradient(0, 0, 0, h)
        g.setColorAt(0.0, QColor("#353b44"))
        g.setColorAt(1.0, QColor("#1d2127"))
        p.fillRect(0, 0, w, h, g)
        col = QColor(D.RARITY_COLORS[rarity])
        rg = QRadialGradient(QPointF(w / 2, h * 1.1), max(w * 0.62, h * 0.8))
        c0 = QColor(col)
        c0.setAlpha(min(255, int(62 * glow)))
        rg.setColorAt(0.0, c0)
        rg.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
        p.fillRect(0, 0, w, h, rg)
        bar = max(3, h // 32)
        p.fillRect(0, h - bar, w, bar, col)
        p.setPen(QColor(255, 255, 255, 22))
        p.drawRect(0, 0, w - 1, h - 1)
        p.end()
        return pm

    return _misc.get_or(("bg", rarity, w, h, glow, dpr()), make)


def star_tile(w: int, h: int) -> QPixmap:
    """The gold '★' placeholder (a rare special item, unrevealed)."""
    w, h = max(int(w), 8), max(int(h), 8)

    def make() -> QPixmap:
        pm = tile_bg(D.SPECIAL, w, h, 1.6).copy()
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rg = QRadialGradient(QPointF(w / 2, h * 0.45), h * 0.55)
        rg.setColorAt(0.0, QColor(255, 214, 120, 120))
        rg.setColorAt(1.0, QColor(228, 174, 57, 0))
        p.fillRect(0, 0, w, h, rg)
        p.setPen(QColor("#ffd479"))
        f = p.font()
        f.setPixelSize(max(int(h * 0.5), 6))
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(0, 0, w, h * 0.9), Qt.AlignmentFlag.AlignCenter, "★")
        p.end()
        return pm

    return _misc.get_or(("star", w, h, dpr()), make)


# --- cases ---------------------------------------------------------------------------------------------------------

def _emblem(p: QPainter, case_id: str, r: QRectF, col: QColor) -> None:
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    cx, cy, s = r.center().x(), r.center().y(), min(r.width(), r.height()) / 2
    p.setPen(QPen(col, max(1.0, s * 0.12)))
    p.setBrush(Qt.BrushStyle.NoBrush)
    if case_id == "sandstorm":  # a sun over dunes
        p.setBrush(col)
        p.drawEllipse(QPointF(cx, cy - s * 0.15), s * 0.38, s * 0.38)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(cx - s, cy + s * 0.55))
        path.quadTo(cx - s * 0.4, cy + s * 0.05, cx + s * 0.1, cy + s * 0.5)
        path.quadTo(cx + s * 0.5, cy + s * 0.2, cx + s, cy + s * 0.55)
        p.drawPath(path)
    elif case_id == "neon":  # a lightning bolt in a ring
        p.drawEllipse(QPointF(cx, cy), s * 0.85, s * 0.85)
        bolt = QPainterPath(QPointF(cx + s * 0.15, cy - s * 0.65))
        for x, y in ((-0.35, 0.05), (0.0, 0.05), (-0.15, 0.65), (0.35, -0.08), (0.02, -0.08)):
            bolt.lineTo(cx + s * x, cy + s * y)
        bolt.closeSubpath()
        p.setBrush(col)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(bolt)
    elif case_id == "glacier":  # a snowflake
        for k in range(6):
            a = k * math.pi / 3
            x1, y1 = cx + math.cos(a) * s * 0.85, cy + math.sin(a) * s * 0.85
            p.drawLine(QPointF(cx, cy), QPointF(x1, y1))
            mx, my = cx + math.cos(a) * s * 0.5, cy + math.sin(a) * s * 0.5
            for da in (0.6, -0.6):
                p.drawLine(QPointF(mx, my), QPointF(mx + math.cos(a + da) * s * 0.25, my + math.sin(a + da) * s * 0.25))
    else:  # a flame
        fl = QPainterPath(QPointF(cx, cy + s * 0.8))
        fl.cubicTo(cx - s * 0.9, cy + s * 0.5, cx - s * 0.3, cy - s * 0.2, cx - s * 0.05, cy - s * 0.85)
        fl.cubicTo(cx + s * 0.1, cy - s * 0.3, cx + s * 0.6, cy - s * 0.3, cx + s * 0.45, cy - s * 0.55)
        fl.cubicTo(cx + s * 0.95, cy + s * 0.1, cx + s * 0.6, cy + s * 0.65, cx, cy + s * 0.8)
        p.setBrush(col)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(fl)
    p.restore()


def case_pixmap(case_id: str, w: int, h: int) -> QPixmap:
    """A weapon case (front view, a touch of perspective): handle, ribs, latches and the case's art panel."""
    w, h = max(int(w), 16), max(int(h), 16)

    def make() -> QPixmap:
        case = D.CASES[case_id]
        c0, c1 = QColor(case.colors[0]), QColor(case.colors[1])
        img = new_image(w, h)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bw, bh = w * 0.9, h * 0.62
        body = QRectF((w - bw) / 2, h * 0.3, bw, bh)
        # handle
        hr = QRectF(w * 0.36, h * 0.12, w * 0.28, h * 0.24)
        p.setPen(QPen(QColor("#16181b"), max(2.0, h * 0.055)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(hr, h * 0.06, h * 0.06)
        # shadow
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 90))
        p.drawEllipse(QRectF(body.left() + bw * 0.05, body.bottom() - bh * 0.02, bw * 0.9, bh * 0.12))
        # body: top lid lighter, base darker
        g = QLinearGradient(body.topLeft(), body.bottomLeft())
        g.setColorAt(0.0, c1.lighter(170))
        g.setColorAt(0.45, c1.lighter(115))
        g.setColorAt(0.5, c1.darker(140))
        g.setColorAt(1.0, c1.darker(220))
        p.setBrush(g)
        p.setPen(QPen(QColor(0, 0, 0, 200), max(1.0, h * 0.012)))
        p.drawRoundedRect(body, bw * 0.06, bw * 0.06)
        # ribs
        p.setPen(QPen(QColor(255, 255, 255, 40), max(1.0, h * 0.01)))
        for k in range(4):
            y = body.top() + bh * (0.08 + 0.04 * k)
            p.drawLine(QPointF(body.left() + bw * 0.08, y), QPointF(body.right() - bw * 0.08, y))
        # art panel
        panel = QRectF(body.left() + bw * 0.18, body.top() + bh * 0.3, bw * 0.64, bh * 0.52)
        pg = QLinearGradient(panel.topLeft(), panel.bottomRight())
        pg.setColorAt(0.0, c0)
        pg.setColorAt(1.0, c0.darker(190))
        p.setBrush(pg)
        p.setPen(QPen(QColor(0, 0, 0, 160), max(1.0, h * 0.01)))
        p.drawRoundedRect(panel, bw * 0.03, bw * 0.03)
        _emblem(p, case_id, panel.adjusted(panel.width() * 0.25, panel.height() * 0.12, -panel.width() * 0.25,
                                           -panel.height() * 0.12), QColor(20, 16, 14, 220))
        # latches
        for x in (body.left() + bw * 0.06, body.right() - bw * 0.06 - bw * 0.08):
            lr = QRectF(x, body.top() + bh * 0.4, bw * 0.08, bh * 0.24)
            lg = QLinearGradient(lr.topLeft(), lr.bottomLeft())
            lg.setColorAt(0.0, QColor("#d9dde1"))
            lg.setColorAt(1.0, QColor("#5c6268"))
            p.setBrush(lg)
            p.setPen(QPen(QColor(0, 0, 0, 180), 1.0))
            p.drawRoundedRect(lr, 2, 2)
        # gloss
        gl = QLinearGradient(body.topLeft(), body.bottomLeft())
        gl.setColorAt(0.0, QColor(255, 255, 255, 60))
        gl.setColorAt(0.3, QColor(255, 255, 255, 0))
        p.setBrush(gl)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(body, bw * 0.06, bw * 0.06)
        p.end()
        return QPixmap.fromImage(img)

    return _misc.get_or(("case", case_id, w, h, dpr()), make)


def key_pixmap(w: int, h: int) -> QPixmap:
    """A case key."""
    w, h = max(int(w), 8), max(int(h), 8)

    def make() -> QPixmap:
        img = new_image(w, h)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = min(w, h)
        g = QLinearGradient(0, 0, 0, h)
        g.setColorAt(0.0, QColor("#f3e3a1"))
        g.setColorAt(1.0, QColor("#a8862f"))
        p.setBrush(g)
        p.setPen(QPen(QColor(0, 0, 0, 160), max(1.0, s * 0.03)))
        bow = QPainterPath()
        bow.addEllipse(QPointF(w * 0.27, h / 2), s * 0.24, s * 0.24)
        bow.addEllipse(QPointF(w * 0.27, h / 2), s * 0.1, s * 0.1)
        p.drawPath(bow)
        shaft = QPainterPath()
        shaft.addRect(QRectF(w * 0.27 + s * 0.22, h / 2 - s * 0.06, w * 0.5, s * 0.12))
        shaft.addRect(QRectF(w * 0.78, h / 2, s * 0.08, s * 0.2))
        shaft.addRect(QRectF(w * 0.66, h / 2, s * 0.07, s * 0.15))
        shaft.setFillRule(Qt.FillRule.WindingFill)
        p.drawPath(shaft.simplified())
        p.end()
        return QPixmap.fromImage(img)

    return _misc.get_or(("key", w, h, dpr()), make)


def clear_caches() -> None:
    _art.clear()
    _big.clear()
    _misc.clear()
