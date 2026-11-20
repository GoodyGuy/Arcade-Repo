"""2D sprite art in palette indices: pickups, decorations, barrels, projectiles and explosions, the first-person
weapons and the status bar face. Drawn with QPainter at the sprites' native size and quantised to the palette."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF, \
    QRadialGradient

from .models import Sprite, cap, humanoid, render, sph, add_pool, ZOMBIE, _human_head
from .palette import SPANS, quantize
from .weapons import WEAPON_ART, WSCALE


def qimage_rgba(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), np.uint8, count=img.bytesPerLine() * h).reshape(h, img.bytesPerLine())
    return arr[:, :w * 4].reshape(h, w, 4).copy()


def paint(w: int, h: int, fn, aa: bool = True, bright: bool = False, xoff: float | None = None,  # noqa: ANN001
          yoff: float = 0.0, scale: float = 1.0) -> Sprite:
    if scale != 1.0:
        w, h = int(round(w * scale)), int(round(h * scale))
        xoff = None if xoff is None else xoff * scale
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, aa)
    if scale != 1.0:
        p.scale(scale, scale)
    fn(p)
    p.end()
    rgba = qimage_rgba(img)
    a = rgba[..., 3:4].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(a > 0, rgba[..., :3] * 255.0 / np.maximum(a, 1), 0)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    idx, mask = quantize(rgba)
    return Sprite(idx, mask, w / 2 if xoff is None else xoff, yoff, bright)


def C(r: int, g: int, b: int, a: int = 255) -> QColor:  # noqa: N802 - short colour helper
    return QColor(r, g, b, a)


def lin(x0: float, y0: float, x1: float, y1: float, *stops) -> QBrush:  # noqa: ANN002
    g = QLinearGradient(x0, y0, x1, y1)
    for t, c in stops:
        g.setColorAt(t, c)
    return QBrush(g)


def rad(cx: float, cy: float, r: float, *stops) -> QBrush:  # noqa: ANN002
    g = QRadialGradient(cx, cy, r)
    for t, c in stops:
        g.setColorAt(t, c)
    return QBrush(g)


def poly(*pts) -> QPolygonF:  # noqa: ANN002
    return QPolygonF([QPointF(x, y) for x, y in pts])


DARK = C(20, 16, 12)


def outline(width: float = 1.0, c: QColor | None = None) -> QPen:
    return QPen(c or DARK, width)


# --- pickups -------------------------------------------------------------------------------------------------------

def _hbonus(frame: int) -> Sprite:
    glow = (0, 40, 80, 40)[frame % 4]

    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 14, 0, (0, C(80, 80, 200 + glow // 2)), (0.4, C(150 + glow, 150 + glow, 255)),
                       (1, C(20, 20, 110))))
        p.drawEllipse(QRectF(1, 7, 12, 11))
        p.drawRect(QRectF(4.5, 2, 5, 6))
        p.setBrush(C(200, 200, 210))
        p.drawRect(QRectF(4, 0.5, 6, 2.5))
    return paint(14, 18, f, bright=False)


def _abonus(frame: int) -> Sprite:
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 16, 14, (0, C(140, 230, 120)), (0.5, C(40, 140 + frame * 20, 40)), (1, C(10, 60, 10))))
        path = QPainterPath()
        path.moveTo(1, 13)
        path.cubicTo(0, 2, 16, 2, 15, 13)
        path.closeSubpath()
        p.drawPath(path)
        p.setBrush(C(20, 30, 20))
        p.drawRect(QRectF(4, 8, 8, 3))
    return paint(16, 14, f)


def _medkit(w: int, h: int) -> Sprite:
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 0, h, (0, C(250, 250, 250)), (0.6, C(200, 200, 205)), (1, C(120, 120, 130))))
        p.drawRect(QRectF(0.5, h * 0.25, w - 1, h * 0.75 - 1))
        p.setBrush(C(170, 170, 180))
        p.drawRect(QRectF(w * 0.35, 0.5, w * 0.3, h * 0.25))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(C(220, 20, 20))
        cx, cy, s = w / 2, h * 0.62, min(w, h) * 0.18
        p.drawRect(QRectF(cx - s * 2, cy - s * 0.7, s * 4, s * 1.4))
        p.drawRect(QRectF(cx - s * 0.7, cy - s * 2, s * 1.4, s * 4))
    return paint(w, h, f)


def _armor(blue: bool) -> Sprite:
    base = C(70, 90, 230) if blue else C(60, 170, 50)
    hi = C(170, 180, 255) if blue else C(170, 240, 140)
    lo = C(10, 10, 90) if blue else C(10, 60, 10)

    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 31, 17, (0, hi), (0.5, base), (1, lo)))
        p.drawPolygon(poly((4, 1), (11, 3), (20, 3), (27, 1), (30, 6), (27, 16), (4, 16), (1, 6)))
        p.setBrush(lo)
        p.drawRect(QRectF(13, 3, 5, 13))
        p.setPen(QPen(hi, 1))
        p.drawLine(QPointF(6, 8), QPointF(11, 8))
        p.drawLine(QPointF(20, 8), QPointF(25, 8))
    return paint(31, 17, f)


def _soulsphere(frame: int) -> Sprite:
    def f(p: QPainter) -> None:
        k = (0, 30, 60, 30)[frame % 4]
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(rad(10, 9, 14, (0, C(220, 230, 255)), (0.35, C(90 + k, 110 + k, 255)), (0.8, C(20, 30, 200)),
                       (1, C(0, 0, 90))))
        p.drawEllipse(QRectF(0.5, 0.5, 23, 23))
        p.setBrush(C(10, 10, 80, 160))
        p.drawEllipse(QRectF(7, 8, 3.5, 3.5))
        p.drawEllipse(QRectF(13.5, 8, 3.5, 3.5))
        p.drawEllipse(QRectF(9.5, 15, 5, 3))
    return paint(24, 24, f, bright=True, yoff=10)


def _clip() -> Sprite:
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 8, 0, (0, C(90, 90, 90)), (0.5, C(170, 170, 160)), (1, C(40, 40, 40))))
        p.drawRect(QRectF(1, 2, 6, 10))
        p.setBrush(C(210, 170, 60))
        p.drawRect(QRectF(2, 0.5, 4, 2))
    return paint(8, 12, f)


def _box(w: int, h: int, body: QColor, top: QColor, contents) -> Sprite:  # noqa: ANN001
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 0, h, (0, body.lighter(140)), (1, body.darker(220))))
        p.drawRect(QRectF(0.5, h * 0.35, w - 1, h * 0.65 - 1))
        p.setBrush(top)
        p.drawPolygon(poly((0.5, h * 0.35), (3, 0.5), (w - 3, 0.5), (w - 0.5, h * 0.35)))
        contents(p)
    return paint(w, h, f)


def _ammobox() -> Sprite:
    def bullets(p: QPainter) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(6):
            p.setBrush(C(230, 190, 70))
            p.drawEllipse(QRectF(4 + i * 3.5, 1.5, 2.6, 3))
    return _box(28, 16, C(60, 110, 50), C(40, 60, 30), bullets)


def _shells() -> Sprite:
    def f(p: QPainter) -> None:
        for i in range(4):
            x = 1 + i * 3.4
            p.setPen(outline(0.8))
            p.setBrush(lin(x, 0, x + 3, 0, (0, C(255, 80, 60)), (1, C(130, 0, 0))))
            p.drawRect(QRectF(x, 1, 3, 5))
            p.setBrush(C(220, 180, 60))
            p.drawRect(QRectF(x, 5, 3, 2))
    return paint(15, 8, f)


def _shellbox() -> Sprite:
    def shells(p: QPainter) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(7):
            p.setBrush(C(200, 30, 20))
            p.drawEllipse(QRectF(3 + i * 4, 1, 3, 3))
    return _box(32, 13, C(150, 40, 30), C(90, 30, 20), shells)


def _rocket(h: int = 25) -> Sprite:
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 7, 0, (0, C(80, 80, 80)), (0.5, C(200, 200, 190)), (1, C(50, 50, 50))))
        p.drawRect(QRectF(1.5, 5, 4, h - 9))
        p.setBrush(C(170, 30, 20))
        p.drawPolygon(poly((1.5, 5.5), (3.5, 0.5), (5.5, 5.5)))
        p.setBrush(C(90, 90, 90))
        p.drawPolygon(poly((0.5, h - 1), (1.5, h - 7), (5.5, h - 7), (6.5, h - 1)))
    return paint(7, h, f)


def _rocketbox() -> Sprite:
    def rockets(p: QPainter) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(5):
            p.setBrush(C(170, 30, 20))
            p.drawPolygon(poly((5 + i * 9, 8), (8 + i * 9, 1), (11 + i * 9, 8)))
    return _box(54, 21, C(90, 90, 70), C(60, 60, 50), rockets)


def _cell(w: int, h: int) -> Sprite:
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, w, 0, (0, C(60, 70, 60)), (0.5, C(150, 170, 150)), (1, C(30, 40, 30))))
        p.drawRect(QRectF(0.5, 2, w - 1, h - 2.5))
        p.setBrush(C(60, 230, 60))
        p.drawRect(QRectF(w * 0.2, h * 0.35, w * 0.6, h * 0.3))
        p.setBrush(C(200, 200, 200))
        p.drawRect(QRectF(w * 0.35, 0.5, w * 0.3, 2))
    return paint(w, h, f)


def _backpack() -> Sprite:
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 22, 20, (0, C(170, 130, 80)), (0.6, C(110, 80, 40)), (1, C(50, 35, 15))))
        p.drawRoundedRect(QRectF(1, 3, 20, 16), 4, 4)
        p.setBrush(C(90, 60, 30))
        p.drawRoundedRect(QRectF(4, 0.5, 14, 8), 3, 3)
        p.setBrush(C(200, 180, 120))
        p.drawRect(QRectF(9.5, 7, 3, 3))
    return paint(22, 20, f)


def _keycard(colour: QColor, frame: int) -> Sprite:
    def f(p: QPainter) -> None:
        c = colour.lighter(170) if frame else colour
        p.setPen(outline())
        p.setBrush(lin(0, 0, 14, 16, (0, c.lighter(130)), (1, c.darker(180))))
        p.drawRect(QRectF(1, 1, 12, 15))
        p.setBrush(C(240, 240, 240))
        p.drawRect(QRectF(3, 4, 8, 3))
        p.setBrush(C(30, 30, 30))
        p.drawRect(QRectF(3, 10, 8, 2))
    return paint(14, 17, f, bright=bool(frame))


def _floor_weapon(kind: str) -> Sprite:
    def f(p: QPainter) -> None:
        metal = lin(0, 0, 0, 16, (0, C(190, 190, 200)), (0.5, C(110, 110, 120)), (1, C(40, 40, 45)))
        wood = lin(0, 0, 0, 16, (0, C(170, 110, 60)), (1, C(70, 40, 15)))
        p.setPen(outline())
        if kind == "w_shotgun":
            p.setBrush(metal)
            p.drawRect(QRectF(16, 3, 45, 3))
            p.setBrush(wood)
            p.drawPolygon(poly((1, 9), (16, 4), (22, 4), (22, 9), (8, 12)))
            p.drawRect(QRectF(30, 5.5, 12, 3))
        elif kind == "w_chaingun":
            p.setBrush(metal)
            for i in range(3):
                p.drawRect(QRectF(20, 2 + i * 2.5, 33, 2))
            p.drawRect(QRectF(6, 3, 18, 9))
            p.setBrush(wood)
            p.drawRect(QRectF(1, 6, 8, 5))
        elif kind == "w_launcher":
            p.setBrush(lin(0, 0, 0, 16, (0, C(140, 150, 120)), (1, C(40, 45, 30))))
            p.drawRect(QRectF(4, 2, 56, 9))
            p.setBrush(C(30, 30, 30))
            p.drawEllipse(QRectF(56, 3, 5, 7))
            p.setBrush(metal)
            p.drawRect(QRectF(18, 10, 6, 5))
        elif kind == "w_plasma":
            p.setBrush(lin(0, 0, 0, 16, (0, C(140, 160, 210)), (1, C(30, 40, 80))))
            p.drawRect(QRectF(4, 3, 46, 9))
            p.setBrush(C(90, 160, 255))
            p.drawRect(QRectF(12, 5, 24, 3))
            p.setBrush(metal)
            p.drawRect(QRectF(48, 5, 6, 5))
        elif kind == "w_bfg":
            p.setBrush(lin(0, 0, 0, 30, (0, C(150, 160, 150)), (1, C(40, 50, 40))))
            p.drawRoundedRect(QRectF(2, 6, 56, 22), 5, 5)
            p.setBrush(C(60, 230, 60))
            p.drawEllipse(QRectF(40, 11, 14, 12))
            p.setBrush(C(30, 30, 30))
            p.drawRect(QRectF(8, 11, 26, 4))
    sizes = {"w_shotgun": (63, 13), "w_chaingun": (54, 16), "w_launcher": (62, 16), "w_plasma": (54, 16),
             "w_bfg": (61, 30)}
    w, h = sizes[kind]
    return paint(w, h, f)


# --- decorations ---------------------------------------------------------------------------------------------------

def _techlamp(frame: int) -> Sprite:
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 14, 0, (0, C(70, 70, 80)), (0.5, C(170, 170, 180)), (1, C(40, 40, 50))))
        p.drawRect(QRectF(5, 14, 4, 52))
        p.drawRect(QRectF(1, 63, 12, 5))
        p.setPen(Qt.PenStyle.NoPen)
        k = 30 if frame else 0
        p.setBrush(rad(7, 7, 8, (0, C(255, 255, 255)), (0.5, C(200 + k // 2, 220, 255)), (1, C(80, 120, 255))))
        p.drawEllipse(QRectF(0.5, 0.5, 13, 15))
    return paint(14, 68, f, bright=True)


def _lamp() -> Sprite:
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 18, 0, (0, C(60, 60, 60)), (0.5, C(150, 150, 150)), (1, C(40, 40, 40))))
        p.drawRect(QRectF(7, 12, 4, 20))
        p.drawPolygon(poly((2, 34), (9, 29), (16, 34)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(rad(9, 7, 8, (0, C(255, 255, 230)), (0.6, C(255, 230, 120)), (1, C(200, 120, 20))))
        p.drawEllipse(QRectF(2.5, 0.5, 13, 13))
    return paint(18, 35, f, bright=True)


def _torch(frame: int) -> Sprite:
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 12, 0, (0, C(70, 40, 20)), (0.5, C(150, 100, 50)), (1, C(40, 20, 10))))
        p.drawRect(QRectF(4, 22, 5, 46))
        p.drawRect(QRectF(2, 20, 9, 4))
        p.setPen(Qt.PenStyle.NoPen)
        rng = np.random.default_rng(frame + 3)
        for i in range(7):
            r = 5 - i * 0.55
            x = 6.5 + rng.normal(0, 1.2)
            y = 18 - i * 2.8
            p.setBrush(rad(x, y, r, (0, C(255, 240, 160)), (0.5, C(255, 120, 20)), (1, C(200, 20, 0, 0))))
            p.drawEllipse(QRectF(x - r, y - r, 2 * r, 2 * r))
    return paint(13, 68, f, bright=True)


def _deadguy() -> Sprite:
    body = dict(ZOMBIE, legs="green", body="green", vest="green", arms="green", forearms="green", hands="olive",
                head=_human_head("green", "green", helmet=True), weapon=lambda *a: [])
    pose = {"hip_l": 8, "hip_r": 14, "knee_l": 12, "knee_r": 6, "sh_l": 30, "sh_r": 45, "out_l": 70, "out_r": 60}
    return add_pool(render(humanoid(body, pose), 90, fall_deg=-88, seed=4), 46)


def _pool() -> Sprite:
    s = Sprite(np.zeros((1, 2), np.uint8), np.zeros((1, 2), bool), 1.0, 0.0)
    return add_pool(s, 40)


# --- the barrel ----------------------------------------------------------------------------------------------------

def _barrel(frame: int) -> Sprite:
    def f(p: QPainter) -> None:
        p.setPen(outline())
        p.setBrush(lin(0, 0, 23, 0, (0, C(60, 70, 60)), (0.35, C(170, 180, 160)), (0.7, C(100, 110, 95)),
                       (1, C(30, 35, 30))))
        p.drawRoundedRect(QRectF(1, 3, 21, 29), 3, 3)
        p.setPen(QPen(C(40, 45, 40), 1.5))
        p.drawLine(QPointF(1.5, 11), QPointF(21.5, 11))
        p.drawLine(QPointF(1.5, 24), QPointF(21.5, 24))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(C(70, 200, 50) if frame else C(110, 240, 80))
        p.drawEllipse(QRectF(2.5, 1, 18, 5))
        p.setBrush(C(60, 180, 40))
        p.drawRect(QRectF(6, 5, 3, 6 + frame))
        p.drawRect(QRectF(14, 5, 2, 4))
    return paint(23, 32, f)


# --- projectiles, explosions, puffs (radial noisy fireballs) -------------------------------------------------------

def ball(radius: float, ramp_name: str, seed: int, t_lo: float = 0.0, t_hi: float = 0.7, ragged: float = 0.35,
         size: int | None = None, yoff: float | None = None, bright: bool = True) -> Sprite:
    n = size or int(radius * 2 + 3)
    ys, xs = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    d = np.sqrt((xs - c) ** 2 + (ys - c) ** 2) / max(radius, 0.5)
    rng = np.random.default_rng(seed)
    ang = np.arctan2(ys - c, xs - c)
    wob = 1.0 + ragged * (0.5 * np.sin(ang * 5 + rng.random() * 6) + 0.5 * np.sin(ang * 9 + rng.random() * 6)) * 0.5
    noise = rng.random((n, n)) * 0.25
    mask = d < wob
    t = np.clip(t_lo + (d / wob) ** 1.3 * (t_hi - t_lo) + noise * 0.4, 0, 1)
    start, count = SPANS[ramp_name]
    img = (np.round(t * (count - 1)) + start).astype(np.uint8)
    return Sprite(img, mask, n / 2, -n / 2 if yoff is None else yoff, bright)


def _puff(i: int) -> Sprite:
    if i == 0:
        return ball(2.5, "fire", 1, 0.0, 0.4)
    return ball(2.5 + i, "grey", i + 2, 0.2 + i * 0.1, 0.6 + i * 0.1, bright=False)


def _blood(i: int) -> Sprite:
    rng = np.random.default_rng(i)
    n = 9 + i * 2
    img = np.zeros((n, n), np.uint8)
    mask = np.zeros((n, n), bool)
    start, count = SPANS["red"]
    for _ in range(6 - i):
        y, x = rng.integers(0, n, 2)
        r = rng.integers(1, 3)
        mask[max(y - r, 0):y + r, max(x - r, 0):x + r] = True
        img[max(y - r, 0):y + r, max(x - r, 0):x + r] = start + int(count * (0.35 + 0.1 * i))
    return Sprite(img, mask, n / 2, -n / 2)


# --- the status bar face -------------------------------------------------------------------------------------------

def face(level: int, look: int = 0, expr: str = "calm") -> Sprite:
    """The marine's face. ``level`` 0 (healthy) .. 4 (nearly dead); ``look`` -1 / 0 / 1; ``expr``: calm, grin,
    ouch, rage, hurt_l, hurt_r, dead, god."""
    def f(p: QPainter) -> None:
        dark = 1.0 - level * 0.08
        skin = C(int(205 * dark), int(140 * dark), int(100 * dark))
        p.setPen(outline(1.0))
        turn = {"hurt_l": -3, "hurt_r": 3}.get(expr, 0)
        p.setBrush(lin(2, 0, 22, 0, (0, skin.lighter(125)), (0.6, skin), (1, skin.darker(200))))
        p.drawRoundedRect(QRectF(3 + turn * 0.5, 3, 18, 25), 7, 8)
        p.setBrush(C(110, 70, 30))
        p.drawRoundedRect(QRectF(3 + turn * 0.5, 0.5, 18, 8), 4, 4)
        p.drawRect(QRectF(3 + turn * 0.5, 4, 2, 6))
        p.drawRect(QRectF(19 + turn * 0.5, 4, 2, 6))
        ex = 12 + turn
        if expr == "dead":
            p.setPen(QPen(C(40, 0, 0), 1.3))
            for x in (ex - 5, ex + 2):
                p.drawLine(QPointF(x, 10), QPointF(x + 3, 13))
                p.drawLine(QPointF(x + 3, 10), QPointF(x, 13))
            p.setBrush(C(60, 0, 0))
            p.drawEllipse(QRectF(ex - 3, 19, 6, 5))
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(C(240, 240, 240) if expr != "god" else C(255, 230, 80))
            p.drawRect(QRectF(ex - 6, 10, 4.5, 3))
            p.drawRect(QRectF(ex + 1.5, 10, 4.5, 3))
            if expr != "god":
                p.setBrush(C(30, 30, 60))
                lx = look if expr in ("calm", "rage") else (-1 if expr == "hurt_l" else 1 if expr == "hurt_r" else 0)
                p.drawRect(QRectF(ex - 4.8 + lx * 1.3, 10.5, 2, 2.4))
                p.drawRect(QRectF(ex + 2.7 + lx * 1.3, 10.5, 2, 2.4))
            p.setPen(QPen(C(70, 40, 20), 1.4))
            brow = 1.5 if expr in ("rage", "ouch", "hurt_l", "hurt_r") else 0
            p.drawLine(QPointF(ex - 6.5, 8.5 - brow * 0.3), QPointF(ex - 1.5, 9 + brow))
            p.drawLine(QPointF(ex + 1.5, 9 + brow), QPointF(ex + 6.5, 8.5 - brow * 0.3))
            p.setPen(QPen(C(80, 30, 20), 1.2))
            p.setBrush(C(60, 10, 10))
            if expr == "grin":
                p.setBrush(C(240, 240, 230))
                p.drawChord(QRectF(ex - 5, 17, 10, 7), 180 * 16, 180 * 16)
            elif expr == "ouch":
                p.drawEllipse(QRectF(ex - 3, 18, 6, 7))
            elif expr in ("rage", "hurt_l", "hurt_r"):
                p.setBrush(C(240, 240, 230))
                p.drawRect(QRectF(ex - 4, 19.5, 8, 3))
            else:
                p.drawLine(QPointF(ex - 3.5, 21.5 + level * 0.2), QPointF(ex + 3.5, 21.5))
        # blood and bruises with the damage taken
        rng = np.random.default_rng(level * 7 + 1)
        p.setPen(Qt.PenStyle.NoPen)
        for _ in range(level * 3 + (6 if expr == "dead" else 0)):
            x, y = rng.uniform(4, 20), rng.uniform(4, 26)
            p.setBrush(C(int(rng.uniform(120, 200)), 0, 0, 220))
            r = rng.uniform(0.8, 1.6 + level * 0.3)
            p.drawEllipse(QRectF(x, y, r * 1.2, r * 2.0))
    return paint(24, 29, f, aa=True)


# --- the registry --------------------------------------------------------------------------------------------------

class Art:
    """Every 2D sprite, built once."""

    def __init__(self) -> None:
        a: dict[str, list[Sprite]] = {}
        a["hbonus"] = [_hbonus(i) for i in range(4)]
        a["abonus"] = [_abonus(i) for i in range(2)]
        a["stim"] = [_medkit(16, 11)]
        a["medikit"] = [_medkit(28, 19)]
        a["garmor"] = [_armor(False)]
        a["barmor"] = [_armor(True)]
        a["soulsphere"] = [_soulsphere(i) for i in range(4)]
        a["clip"] = [_clip()]
        a["ammobox"] = [_ammobox()]
        a["shells"] = [_shells()]
        a["shellbox"] = [_shellbox()]
        a["rocket"] = [_rocket()]
        a["rocketbox"] = [_rocketbox()]
        a["cell"] = [_cell(15, 12)]
        a["cellpack"] = [_cell(32, 22)]
        a["backpack"] = [_backpack()]
        a["bluecard"] = [_keycard(C(40, 60, 230), i) for i in range(2)]
        a["yellowcard"] = [_keycard(C(230, 200, 30), i) for i in range(2)]
        a["redcard"] = [_keycard(C(220, 30, 30), i) for i in range(2)]
        for k in ("w_shotgun", "w_chaingun", "w_launcher", "w_plasma", "w_bfg"):
            a[k] = [_floor_weapon(k)]
        a["techlamp"] = [_techlamp(i) for i in range(2)]
        a["lamp"] = [_lamp()]
        a["torch"] = [_torch(i) for i in range(3)]
        a["deadguy"] = [_deadguy()]
        a["bloodpool"] = [_pool()]
        a["barrel"] = [_barrel(i) for i in range(2)]
        a["barrel_x"] = [ball(10 + i * 6, "fire", 30 + i, 0.0, 0.75, yoff=4 - i) for i in range(4)]
        a["imp_ball"] = [ball(5, "fire", 40 + i, 0.05, 0.55) for i in range(2)]
        a["imp_x"] = [ball(7 + i * 3, "fire", 45 + i, 0.15 + i * 0.1, 0.8) for i in range(3)]
        a["caco_ball"] = [ball(6, "purple", 50 + i, 0.0, 0.6) for i in range(2)]
        a["caco_x"] = [ball(8 + i * 3, "purple", 55 + i, 0.1 + i * 0.1, 0.9) for i in range(3)]
        a["baron_ball"] = [ball(6, "green", 60 + i, 0.0, 0.5) for i in range(2)]
        a["baron_x"] = [ball(8 + i * 3, "green", 65 + i, 0.1 + i * 0.1, 0.8) for i in range(3)]
        a["rocket_fly"] = [ball(3, "fire", 70, 0.1, 0.5)]
        a["rocket_x"] = [ball(14 + i * 7, "fire", 75 + i, 0.0 + i * 0.1, 0.8) for i in range(3)]
        a["plasma_ball"] = [ball(4, "blue", 80 + i, 0.0, 0.4) for i in range(2)]
        a["plasma_x"] = [ball(5 + i * 2, "blue", 85 + i, 0.1 * i, 0.6 + 0.1 * i) for i in range(4)]
        a["bfg_ball"] = [ball(12, "green", 90 + i, 0.0, 0.45) for i in range(2)]
        a["bfg_x"] = [ball(18 + i * 8, "green", 95 + i, 0.0 + i * 0.08, 0.7) for i in range(5)]
        a["tracer_x"] = [ball(6 + i * 3, "green", 101 + i, 0.1 * i, 0.7) for i in range(3)]
        a["puff"] = [_puff(i) for i in range(4)]
        a["blood"] = [_blood(i) for i in range(3)]
        self.sprites = a
        self.weapons = {
            name: [paint(w, h, (lambda p, fn=fn, i=i: fn(p, i)), xoff=cx, scale=WSCALE) for i in range(n)]
            for name, (fn, n, w, h, cx) in WEAPON_ART.items()}
        exprs = ("calm", "grin", "ouch", "rage", "hurt_l", "hurt_r")
        self.faces = {(lvl, look, e): face(lvl, look, e) for lvl in range(5) for look in (-1, 0, 1)
                      for e in exprs if look == 0 or e in ("calm",)}
        self.faces[("dead",)] = face(4, 0, "dead")
        self.faces[("god",)] = face(0, 0, "god")


_ART: Art | None = None


def art() -> Art:
    global _ART
    if _ART is None:
        _ART = Art()
    return _ART


__all__ = ["art", "Art", "ball", "paint", "cap", "sph"]
