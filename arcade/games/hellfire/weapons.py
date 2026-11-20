"""The first-person weapons, drawn the way the 1993 game shows them: seen from behind and a little above, the barrel
pointing into the screen (narrow and far at the top, wide and near at the bottom), lit from the upper left, with the
marine's hands and muzzle-flash fire frames. Coordinates are design units; ``WSCALE`` sizes them for the screen."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient

WSCALE = 1.3
# screen row of each weapon picture's top (its lower part hides behind the status bar, like the original)
WEAPON_TOP = {"fist": 11, "pistol": 36, "shotgun": 40, "chaingun": 22, "launcher": 24, "plasma": 18, "bfg": 16}


def C(r: int, g: int, b: int, a: int = 255) -> QColor:  # noqa: N802 - short colour helper
    return QColor(r, g, b, a)


INK = C(14, 10, 8)
STEEL = (C(26, 28, 34), C(88, 92, 104), C(168, 174, 188), C(228, 232, 240))
GUNMETAL = (C(14, 14, 18), C(52, 54, 62), C(112, 116, 128), C(176, 180, 190))
WOOD = (C(54, 26, 8), C(132, 78, 36), C(190, 124, 64), C(222, 170, 110))
OLIVE = (C(32, 36, 18), C(86, 94, 52), C(142, 152, 94), C(188, 196, 140))
BLUESTEEL = (C(20, 26, 48), C(70, 84, 128), C(140, 158, 206), C(206, 216, 246))
BFGGREEN = (C(24, 36, 24), C(72, 92, 70), C(132, 156, 126), C(196, 214, 188))
SKIN = (C(104, 54, 30), C(190, 122, 82), C(234, 180, 140), C(250, 214, 184))
SLEEVE = (C(16, 40, 14), C(52, 100, 44), C(96, 150, 76), C(140, 190, 110))


def poly(*pts) -> QPolygonF:  # noqa: ANN002
    return QPolygonF([QPointF(x, y) for x, y in pts])


def shade_h(x0: float, x1: float, tone: tuple, spec: bool = True) -> QLinearGradient:
    """A cylinder's shading across x: dark rim, highlight left of centre, mid, dark rim."""
    dark, mid, light, hi = tone
    g = QLinearGradient(x0, 0, x1, 0)
    g.setColorAt(0.0, dark)
    g.setColorAt(0.18, mid)
    g.setColorAt(0.32, hi if spec else light)
    g.setColorAt(0.42, light)
    g.setColorAt(0.7, mid)
    g.setColorAt(1.0, dark)
    return g


def shade_v(y0: float, y1: float, tone: tuple) -> QLinearGradient:
    dark, mid, light, _hi = tone
    g = QLinearGradient(0, y0, 0, y1)
    g.setColorAt(0.0, light)
    g.setColorAt(0.5, mid)
    g.setColorAt(1.0, dark)
    return g


def tube(p: QPainter, cx0: float, y0: float, w0: float, cx1: float, y1: float, w1: float, tone: tuple,
         bore: bool = True, bore_scale: float = 0.62) -> None:
    """A barrel seen along its length: a trapezoid shaded as a cylinder, its muzzle an ellipse at the top."""
    p.setPen(QPen(INK, 1.0))
    p.setBrush(shade_h(min(cx0 - w0 / 2, cx1 - w1 / 2), max(cx0 + w0 / 2, cx1 + w1 / 2), tone))
    p.drawPolygon(poly((cx0 - w0 / 2, y0), (cx0 + w0 / 2, y0), (cx1 + w1 / 2, y1), (cx1 - w1 / 2, y1)))
    p.setBrush(tone[1])
    p.drawEllipse(QRectF(cx0 - w0 / 2, y0 - w0 * 0.22, w0, w0 * 0.44))
    if bore:
        bw = w0 * bore_scale
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(INK)
        p.drawEllipse(QRectF(cx0 - bw / 2, y0 - bw * 0.2, bw, bw * 0.4))


def block(p: QPainter, pts: list, tone: tuple, x0: float, x1: float) -> None:
    p.setPen(QPen(INK, 1.0))
    p.setBrush(shade_h(x0, x1, tone, spec=False))
    p.drawPolygon(poly(*pts))


def rivet(p: QPainter, x: float, y: float, r: float = 1.3) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(C(20, 20, 24))
    p.drawEllipse(QPointF(x + 0.4, y + 0.4), r, r)
    p.setBrush(C(200, 204, 214))
    p.drawEllipse(QPointF(x - 0.3, y - 0.3), r * 0.55, r * 0.55)


def sleeve(p: QPainter, pts: list) -> None:
    xs = [x for x, _ in pts]
    block(p, pts, SLEEVE, min(xs), max(xs))


def flash(p: QPainter, cx: float, cy: float, r: float, blue: bool = False, seed: int = 0) -> None:
    """A jagged muzzle flash: white core, yellow, orange tips (blue for energy weapons)."""
    core = C(255, 255, 240) if not blue else C(235, 245, 255)
    mid = C(255, 214, 60) if not blue else C(110, 160, 255)
    tip = C(240, 90, 0, 0) if not blue else C(40, 40, 230, 0)
    p.setPen(Qt.PenStyle.NoPen)
    for layer, (rr, n) in enumerate(((r, 11), (r * 0.62, 9))):
        path = QPainterPath()
        for i in range(n * 2):
            a = i / (n * 2) * 2 * math.pi + seed * 0.37 + layer * 0.5
            k = 1.0 if i % 2 == 0 else 0.42 + 0.12 * math.sin(i * 2.3 + seed)
            pt = QPointF(cx + math.cos(a) * rr * k, cy + math.sin(a) * rr * k * 0.75)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        path.closeSubpath()
        g = QRadialGradient(cx, cy, rr)
        g.setColorAt(0.0, core)
        g.setColorAt(0.45, mid)
        g.setColorAt(1.0, tip)
        p.setBrush(g)
        p.drawPath(path)


# --- the hands --------------------------------------------------------------------------------------------------------

def fist_shape(p: QPainter, x: float, y: float, s: float) -> None:
    """A right fist from behind: four knuckles on top, the thumb across the front, the forearm below."""
    sleeve(p, [(x + 6 * s, y + 22 * s), (x + 30 * s, y + 22 * s), (x + 38 * s, y + 70 * s), (x + 2 * s, y + 70 * s)])
    p.setPen(QPen(INK, 1.0))
    p.setBrush(shade_h(x, x + 36 * s, SKIN, spec=False))
    p.drawRoundedRect(QRectF(x, y + 6 * s, 36 * s, 24 * s), 8 * s, 8 * s)
    for i in range(4):  # knuckles
        kx = x + 2 * s + i * 8.2 * s
        g = QRadialGradient(kx + 3 * s, y + 6 * s, 6 * s)
        g.setColorAt(0, SKIN[3])
        g.setColorAt(0.6, SKIN[2])
        g.setColorAt(1, SKIN[1])
        p.setBrush(g)
        p.drawEllipse(QRectF(kx, y + 1 * s, 8.6 * s, 10 * s))
    p.setPen(QPen(SKIN[0], 1.0))
    for i in range(1, 4):
        p.drawLine(QPointF(x + 2 * s + i * 8.2 * s, y + 9 * s), QPointF(x + 2 * s + i * 8.2 * s, y + 17 * s))
    p.setPen(QPen(INK, 1.0))
    p.setBrush(shade_v(y + 14 * s, y + 24 * s, SKIN))
    p.drawRoundedRect(QRectF(x + 3 * s, y + 15 * s, 24 * s, 8 * s), 4 * s, 4 * s)  # the thumb


def gripping_hand(p: QPainter, cx: float, y: float, s: float = 1.0) -> None:
    """The right hand round a pistol grip seen from behind: fingers wrapped on the right, thumb up the left."""
    sleeve(p, [(cx - 14 * s, y + 26 * s), (cx + 22 * s, y + 24 * s), (cx + 34 * s, y + 60 * s),
               (cx - 18 * s, y + 60 * s)])
    p.setPen(QPen(INK, 1.0))
    p.setBrush(shade_h(cx - 18 * s, cx + 26 * s, SKIN, spec=False))
    p.drawRoundedRect(QRectF(cx - 16 * s, y, 40 * s, 30 * s), 9 * s, 9 * s)  # back of the hand
    for i in range(3):  # finger tips curling round the right of the grip
        g = QRadialGradient(cx + 18 * s, y + 6 * s + i * 8 * s, 7 * s)
        g.setColorAt(0, SKIN[2])
        g.setColorAt(1, SKIN[1])
        p.setBrush(g)
        p.drawRoundedRect(QRectF(cx + 12 * s, y + 2 * s + i * 8 * s, 14 * s, 8.5 * s), 4 * s, 4 * s)
    p.setPen(QPen(SKIN[0], 1.0))
    p.drawLine(QPointF(cx - 10 * s, y + 10 * s), QPointF(cx + 8 * s, y + 9 * s))
    p.setPen(QPen(INK, 1.0))
    p.setBrush(shade_h(cx - 22 * s, cx - 8 * s, SKIN))
    p.drawPolygon(poly((cx - 20 * s, y + 18 * s), (cx - 13 * s, y - 14 * s), (cx - 6 * s, y - 15 * s),
                       (cx - 7 * s, y + 18 * s)))  # thumb along the frame
    p.setBrush(SKIN[3])
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QRectF(cx - 12.5 * s, y - 15 * s, 5 * s, 4 * s))  # thumbnail


def side_hand(p: QPainter, x: float, y: float, s: float = 1.0, left: bool = True) -> None:
    """A hand holding a big gun's side: knuckles toward the gun, the sleeve going down and out."""
    d = -1 if left else 1
    sleeve(p, [(x - 4 * s * d, y + 16 * s), (x + 18 * s * d, y + 14 * s), (x + 26 * s * d, y + 50 * s),
               (x - 14 * s * d, y + 50 * s)])
    p.setPen(QPen(INK, 1.0))
    x0 = min(x - 8 * s * d, x + 22 * s * d)
    p.setBrush(shade_h(x0, x0 + 30 * s, SKIN, spec=False))
    p.drawRoundedRect(QRectF(x0, y, 30 * s, 22 * s), 8 * s, 8 * s)
    p.setPen(QPen(SKIN[0], 1.0))
    for i in range(1, 4):
        fx = x0 + i * 7.5 * s
        p.drawLine(QPointF(fx, y + 2 * s), QPointF(fx, y + 9 * s))


# --- the weapons ------------------------------------------------------------------------------------------------------

def draw_fist(p: QPainter, frame: int) -> None:
    x, y, s = ((122, 80, 1.55), (88, 50, 1.8), (64, 26, 2.1))[frame]
    fist_shape(p, x, y, s)


def draw_pistol(p: QPainter, frame: int) -> None:
    cx = 60
    p.translate(0, (0, -9, -4)[frame])
    p.translate(0, 14)
    p.translate(cx, 32)
    p.scale(1.0, 0.8)
    p.translate(-cx, -32)
    if frame == 1:
        flash(p, cx, 22, 19, seed=1)
    # frame and grip under the slide
    block(p, [(cx - 15, 76), (cx + 15, 76), (cx + 17, 104), (cx - 17, 104)], GUNMETAL, cx - 17, cx + 17)
    # the slide: narrow far away, wide near the eye
    tube(p, cx, 32, 11, cx, 82, 30, GUNMETAL, bore=True, bore_scale=0.5)
    p.setPen(Qt.PenStyle.NoPen)
    top = QLinearGradient(0, 32, 0, 82)
    top.setColorAt(0, GUNMETAL[2])
    top.setColorAt(1, GUNMETAL[3])
    p.setBrush(top)
    p.drawPolygon(poly((cx - 3, 34), (cx + 3, 34), (cx + 7, 80), (cx - 7, 80)))  # the flat top of the slide
    p.setPen(QPen(GUNMETAL[0], 1.0))
    for i in range(6):  # rear serrations
        yy = 70 + i * 2
        w = 7 + (yy - 34) / 46 * 8
        p.drawLine(QPointF(cx - w, yy), QPointF(cx - w + 4, yy))
        p.drawLine(QPointF(cx + w - 4, yy), QPointF(cx + w, yy))
    p.setPen(QPen(INK, 1.0))
    p.setBrush(GUNMETAL[1])
    p.drawRect(QRectF(cx + 4, 52, 7, 10))  # ejection port
    p.setBrush(GUNMETAL[2])
    p.drawRect(QRectF(cx - 1.5, 31, 3, 3))  # front sight
    p.setBrush(GUNMETAL[0])
    p.drawPolygon(poly((cx - 10, 80), (cx - 3, 80), (cx - 3, 75), (cx - 7, 74)))  # rear sight, notch between
    p.drawPolygon(poly((cx + 10, 80), (cx + 3, 80), (cx + 3, 75), (cx + 7, 74)))
    gripping_hand(p, cx, 84, 1.05)


def draw_shotgun(p: QPainter, frame: int) -> None:
    cx = 70
    tilt = (0, 0, 6, 11)[frame]
    dy = (0, -7, 5, 9)[frame]
    p.translate(cx, 124 + dy)
    p.rotate(-tilt)
    p.translate(-cx, -124)
    if frame == 1:
        flash(p, cx + 1, 6, 30, seed=2)
    # the receiver near the eye
    block(p, [(cx - 22, 92), (cx + 22, 92), (cx + 30, 124), (cx - 30, 124)], STEEL, cx - 30, cx + 30)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(STEEL[3])
    p.drawPolygon(poly((cx - 6, 93), (cx + 6, 93), (cx + 8, 124), (cx - 8, 124)))
    p.setBrush(INK)
    p.drawRect(QRectF(cx + 9, 100, 9, 12))  # ejection port
    # the barrel and the magazine tube side by side
    tube(p, cx - 6, 12, 10, cx - 13, 96, 25, STEEL)
    tube(p, cx + 6, 16, 9, cx + 12, 96, 23, STEEL, bore=False)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(C(220, 200, 120))
    p.drawEllipse(QRectF(cx - 7.5, 9, 3, 3))  # the bead
    # the wooden pump round both tubes
    py0 = 52 if frame < 2 else 64
    w0, w1 = 30 + (py0 - 52) * 0.3, 42 + (py0 - 52) * 0.3
    block(p, [(cx - w0 / 2, py0), (cx + w0 / 2, py0), (cx + w1 / 2, py0 + 30), (cx - w1 / 2, py0 + 30)], WOOD,
          cx - w1 / 2, cx + w1 / 2)
    p.setPen(QPen(WOOD[0], 1.2))
    for i in range(7):
        yy = py0 + 3 + i * 4
        k = (yy - py0) / 30
        hw = (w0 + (w1 - w0) * k) / 2 - 2
        p.drawLine(QPointF(cx - hw, yy), QPointF(cx + hw, yy))
    if frame >= 2:
        side_hand(p, cx - 26, py0 + 4, 1.2, left=True)


def draw_chaingun(p: QPainter, frame: int) -> None:
    cx = 80
    p.translate(0, (0, -3, -2)[frame])
    if frame:
        flash(p, cx + (7 if frame == 1 else -7), 18, 24, seed=frame + 3)
    # the body near the eye
    block(p, [(cx - 30, 74), (cx + 30, 74), (cx + 40, 130), (cx - 40, 130)], STEEL, cx - 40, cx + 40)
    p.setPen(QPen(INK, 1.0))
    p.setBrush(shade_v(78, 96, STEEL))
    p.drawRoundedRect(QRectF(cx - 18, 66, 36, 18), 5, 5)  # the barrel housing
    for x in (cx - 26, cx + 26):
        rivet(p, x, 90)
        rivet(p, x + (2 if x > cx else -2), 116)
    # six barrels turning: four in view, their highlights shift between the firing frames
    spin = (0, 4, 2)[frame]
    for i in range(4):
        off = -13.5 + i * 9 + (spin if i % 2 else -spin) * 0.25
        tube(p, cx + off * 0.7, 24, 7, cx + off, 70, 11, STEEL, bore=True, bore_scale=0.55)
    p.setPen(QPen(INK, 1.0))
    p.setBrush(shade_h(cx - 22, cx + 22, STEEL, spec=False))
    p.drawRoundedRect(QRectF(cx - 18, 31, 36, 6), 2, 2)  # front clamp
    p.drawRoundedRect(QRectF(cx - 22, 52, 44, 7), 2, 2)  # rear clamp
    side_hand(p, cx - 48, 88, 1.25, left=True)
    side_hand(p, cx + 48, 90, 1.25, left=False)


def draw_launcher(p: QPainter, frame: int) -> None:
    cx = 70
    p.translate(0, (0, -6, -3)[frame])
    # the big tube
    tube(p, cx, 22, 30, cx, 125, 58, OLIVE, bore=True, bore_scale=0.72)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(C(60, 60, 60))
    p.drawEllipse(QRectF(cx - 8, 19, 16, 5))  # the rocket nose in the bore
    p.setBrush(C(190, 40, 20))
    p.drawEllipse(QRectF(cx - 4, 20, 8, 3))
    if frame == 1:
        flash(p, cx, 20, 34, seed=5)
    p.setPen(QPen(INK, 1.0))
    p.setBrush(shade_h(cx - 26, cx + 26, OLIVE, spec=False))
    p.drawRect(QRectF(cx - 21, 58, 42, 5))  # bands
    p.drawRect(QRectF(cx - 25, 92, 50, 6))
    p.setBrush(C(200, 70, 20))
    p.drawRect(QRectF(cx - 20, 71, 40, 3))
    p.setBrush(shade_h(cx - 6, cx + 6, STEEL))
    p.drawRect(QRectF(cx - 5, 30, 10, 24))  # the sight on top
    p.setBrush(INK)
    p.drawRect(QRectF(cx - 2, 33, 4, 5))
    for x in (cx - 16, cx + 16):
        rivet(p, x, 80)
    if frame == 2:
        p.setPen(Qt.PenStyle.NoPen)
        for i, (sx, sy, r) in enumerate(((cx - 6, 6, 9), (cx + 7, 0, 7), (cx, -6, 6))):
            p.setBrush(C(150, 150, 150, 150 - i * 30))
            p.drawEllipse(QPointF(sx, sy + 14), r, r)
    side_hand(p, cx + 30, 86, 1.25, left=False)


def draw_plasma(p: QPainter, frame: int) -> None:
    cx = 70
    p.translate(0, (0, -3, -1)[frame])
    if frame:
        flash(p, cx, 22, 20 + frame * 4, blue=True, seed=frame + 6)
    block(p, [(cx - 14, 26), (cx + 14, 26), (cx + 34, 128), (cx - 34, 128)], BLUESTEEL, cx - 34, cx + 34)
    p.setPen(QPen(INK, 1.0))
    p.setBrush(INK)
    p.drawRect(QRectF(cx - 9, 22, 18, 8))  # the emitter
    p.setBrush(C(90, 150, 255) if frame else C(40, 80, 200))
    p.drawRect(QRectF(cx - 6, 24, 12, 4))
    # the energy channel and the cooling fins
    glow = QLinearGradient(0, 34, 0, 120)
    glow.setColorAt(0, C(200, 230, 255) if frame else C(120, 170, 255))
    glow.setColorAt(1, C(30, 60, 200))
    p.setBrush(glow)
    p.drawPolygon(poly((cx - 3, 34), (cx + 3, 34), (cx + 6, 120), (cx - 6, 120)))
    for i in range(6):
        yy = 40 + i * 14
        hw = 15 + (yy - 26) / 102 * 19
        p.setBrush(shade_h(cx - hw, cx + hw, BLUESTEEL, spec=False))
        p.drawRect(QRectF(cx - hw, yy, hw - 8, 5))
        p.drawRect(QRectF(cx + 8, yy, hw - 8, 5))
    side_hand(p, cx - 42, 86, 1.25, left=True)
    side_hand(p, cx + 42, 88, 1.25, left=False)


def draw_bfg(p: QPainter, frame: int) -> None:
    cx = 90
    p.translate(0, (0, 0, -6)[frame])
    block(p, [(cx - 44, 24), (cx + 44, 24), (cx + 64, 130), (cx - 64, 130)], BFGGREEN, cx - 64, cx + 64)
    p.setPen(QPen(INK, 1.0))
    p.setBrush(shade_h(cx - 50, cx + 50, STEEL, spec=False))
    p.drawRect(QRectF(cx - 46, 64, 92, 8))
    p.drawRect(QRectF(cx - 52, 96, 104, 8))
    for x in (cx - 36, cx + 36):
        for y in (40, 84, 116):
            rivet(p, x + (x - cx) * (y - 24) / 300, y)
    core = (C(70, 150, 70), C(140, 255, 120), C(235, 255, 225))[frame]
    g = QRadialGradient(cx, 30, 30)
    g.setColorAt(0, core.lighter(130))
    g.setColorAt(0.55, core)
    g.setColorAt(1, C(10, 50, 10))
    p.setBrush(INK)
    p.drawEllipse(QRectF(cx - 32, 6, 64, 44))  # the emitter ring
    p.setBrush(g)
    p.drawEllipse(QRectF(cx - 27, 10, 54, 36))
    if frame == 2:
        flash(p, cx, 26, 40, blue=False, seed=9)
        p.setBrush(C(220, 255, 210, 200))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(cx - 18, 16, 36, 22))
    side_hand(p, cx - 66, 86, 1.35, left=True)
    side_hand(p, cx + 66, 88, 1.35, left=False)


# name: (painter, frames, canvas width, height, centre column)
WEAPON_ART = {
    "fist": (draw_fist, 3, 200, 130, 100), "pistol": (draw_pistol, 3, 120, 150, 60),
    "shotgun": (draw_shotgun, 4, 140, 130, 70), "chaingun": (draw_chaingun, 3, 160, 134, 80),
    "launcher": (draw_launcher, 3, 140, 130, 70), "plasma": (draw_plasma, 3, 140, 134, 70),
    "bfg": (draw_bfg, 3, 180, 134, 90),
}
