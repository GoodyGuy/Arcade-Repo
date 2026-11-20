"""Rail Rush 2D art, drawn with QPainter for the menus, the HUD and the home-screen card: the courier crew, the yard
drone, coins, power-up and item icons, and the logo. (The 3D view is in ``models.py``.) Everything is original artwork
made in code."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF,
                           QRadialGradient)

NOPEN = QPen(Qt.PenStyle.NoPen)
NOBRUSH = QBrush(Qt.BrushStyle.NoBrush)
ROUND = Qt.PenCapStyle.RoundCap

PAINTS = ("#ff3b6b", "#ffd23f", "#2fe0ff", "#7cff4f", "#b45cff", "#ff8a1f", "#3b7bff", "#ff5fd2", "#00e0a0")


def qc(spec: str, alpha: int = 255) -> QColor:
    c = QColor(spec)
    c.setAlpha(alpha)
    return c


_PENS: dict = {}


def pen(color: QColor | str, width: float, cap: Qt.PenCapStyle = ROUND) -> QPen:
    """A pen (cached: making QPens every frame is slow in Python)."""
    c = QColor(color)
    key = (c.rgba(), round(width, 3), cap)
    pn = _PENS.get(key)
    if pn is None:
        if len(_PENS) > 4000:
            _PENS.clear()
        pn = _PENS[key] = QPen(c, width)
        pn.setCapStyle(cap)
        pn.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pn


def _font(size: float, weight: QFont.Weight = QFont.Weight.Black, family: str = "Segoe UI") -> QFont:
    f = QFont(family)
    f.setPixelSize(max(1, int(size)))
    f.setWeight(weight)
    return f


def _image(w: int, h: int) -> tuple[QImage, QPainter]:
    img = QImage(max(1, w), max(1, h), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    return img, p


# --- the runner ------------------------------------------------------------------------------------------------------

@dataclass
class Pose:
    mode: str = "run"  # run, jump, roll, fly, board, stumble, dead, idle
    phase: float = 0.0  # run cycle (rad)
    lean: float = 0.0  # -1..1
    t: float = 0.0
    roll: float = 0.0  # 0..1 through a roll
    rising: bool = True
    front: bool = False
    board: bool = False
    jet: bool = False
    board_color: str = "#ffd23f"


def _poly(p: QPainter, pts) -> None:  # noqa: ANN001
    p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))


def _shoe(p: QPainter, ch: dict, x: float, y: float, u: float, b: float, front: bool) -> None:
    w = 0.17 * b
    if u > 0.22 and not front:  # the sole shows as the foot comes up behind
        h = 0.08 + 0.12 * u
        p.setPen(NOPEN)
        p.setBrush(qc(ch["shoes"]).darker(115))
        p.drawRoundedRect(QRectF(x - w / 2, y - h * 0.5, w, h), 0.05, 0.05)
        p.setBrush(qc("#f4f1ea"))
        p.drawRoundedRect(QRectF(x - w * 0.42, y - h * 0.5 + 0.02, w * 0.84, h * 0.72), 0.04, 0.04)
        p.setPen(pen("#b9b2a6", 0.012))
        for k in range(3):
            yy = y - h * 0.5 + 0.04 + k * h * 0.2
            p.drawLine(QPointF(x - w * 0.3, yy), QPointF(x + w * 0.3, yy))
    else:
        p.setPen(NOPEN)
        p.setBrush(qc(ch["shoes"]))
        p.drawRoundedRect(QRectF(x - w / 2, y - 0.07, w, 0.11), 0.05, 0.05)
        p.setBrush(qc("#f4f1ea"))
        p.drawRect(QRectF(x - w / 2 + 0.01, y + 0.015, w - 0.02, 0.025))


def _leg(p: QPainter, ch: dict, hip: tuple, sd: int, u: float, b: float, spread: float = 0.0,
         front: bool = False) -> None:
    hx, hy = hip
    knee = (hx + sd * (0.03 + 0.07 * u) + sd * spread * 0.5, hy + 0.42 - 0.2 * u)
    foot = (hx + sd * (0.03 + 0.02 * u) + sd * spread, -0.05 - 0.55 * u)
    p.setPen(pen(ch["pants"], 0.17 * b))
    _poly(p, (hip, knee, foot))
    _shoe(p, ch, foot[0], foot[1], u, b, front)


def _head_back(p: QPainter, ch: dict, hx: float, hy: float, r: float, pose: Pose) -> None:
    style, hair = ch["style"], ch["hair"]
    skin = qc(ch["skin"])
    p.setPen(NOPEN)
    p.setBrush(skin.darker(108))
    p.drawEllipse(QPointF(hx - r * 0.98, hy + 0.01), 0.04, 0.05)
    p.drawEllipse(QPointF(hx + r * 0.98, hy + 0.01), 0.04, 0.05)
    dark = qc("#2a1a10")
    if style in ("cap", "beanie"):
        p.setBrush(dark)
        p.drawEllipse(QPointF(hx, hy), r, r)
        dome = QPainterPath()
        top = 0.1 if style == "cap" else 0.25
        dome.moveTo(hx - r * 1.04, hy + r * top)
        dome.arcTo(QRectF(hx - r * 1.04, hy - r * 1.06, r * 2.08, r * 2.1), 180 + top * 40, -(180 + top * 80))
        dome.closeSubpath()
        g = QLinearGradient(hx, hy - r, hx, hy + r * 0.3)
        g.setColorAt(0, qc(hair).lighter(125))
        g.setColorAt(1, qc(hair).darker(115))
        p.setBrush(g)
        p.drawPath(dome)
        if style == "cap":
            p.setBrush(dark)
            p.drawChord(QRectF(hx - 0.06, hy + r * 0.05, 0.12, 0.12), 0, 180 * 16)
            p.setBrush(qc(hair).darker(135))
            p.drawEllipse(QRectF(hx - 0.15, hy + r * 0.12, 0.3, 0.1))
            p.setBrush(qc(ch["accent"]))
            p.drawEllipse(QPointF(hx, hy - r * 0.92), 0.025, 0.02)
        else:
            p.setBrush(qc(hair).darker(125))
            p.drawRoundedRect(QRectF(hx - r * 1.05, hy + r * 0.05, r * 2.1, r * 0.4), 0.04, 0.04)
            p.setBrush(qc(ch["accent"]))
            p.drawEllipse(QPointF(hx, hy - r * 1.12), 0.07, 0.07)
    elif style == "ponytail":
        sway = math.sin(pose.phase) * 0.06 + pose.lean * -0.08
        p.setBrush(qc(hair).darker(110))
        tail = QPainterPath()
        tail.moveTo(hx - 0.05, hy - r * 0.3)
        tail.quadTo(hx + sway - 0.09, hy + 0.25, hx + sway * 1.6, hy + 0.42)
        tail.quadTo(hx + sway + 0.1, hy + 0.2, hx + 0.05, hy - r * 0.3)
        p.drawPath(tail)
        g = QRadialGradient(hx - r * 0.3, hy - r * 0.4, r * 1.6)
        g.setColorAt(0, qc(hair).lighter(130))
        g.setColorAt(1, qc(hair).darker(120))
        p.setBrush(g)
        p.drawEllipse(QPointF(hx, hy - 0.01), r * 1.04, r * 1.02)
        p.setBrush(qc(ch["accent"]))
        p.drawEllipse(QPointF(hx, hy - r * 0.3), 0.04, 0.035)
    elif style == "puffs":
        p.setBrush(qc(hair))
        p.drawEllipse(QPointF(hx, hy), r * 1.02, r * 1.02)
        for sx in (-1, 1):
            p.drawEllipse(QPointF(hx + sx * r * 0.72, hy - r * 0.85), r * 0.62, r * 0.6)
        p.setBrush(NOBRUSH)
        p.setPen(pen(ch["accent"], 0.04))
        p.drawArc(QRectF(hx - r * 1.08, hy - r * 1.12, r * 2.16, r * 2.1), 10 * 16, 160 * 16)
        p.setPen(NOPEN)
        for sx in (-1, 1):
            p.setBrush(qc("#20242c"))
            p.drawEllipse(QPointF(hx + sx * r * 1.02, hy + 0.01), 0.07, 0.085)
            p.setBrush(qc(ch["accent"]))
            p.drawEllipse(QPointF(hx + sx * r * 1.06, hy + 0.01), 0.045, 0.06)
    else:  # mohawk
        p.setBrush(skin)
        p.drawEllipse(QPointF(hx, hy), r, r)
        p.setBrush(qc("#3a2a20", 120))
        p.drawEllipse(QPointF(hx, hy - 0.01), r * 0.95, r * 0.95)
        p.setBrush(qc(hair))
        spikes = QPainterPath()
        spikes.moveTo(hx - 0.05, hy + r * 0.8)
        n = 5
        for i in range(n + 1):
            yy = hy + r * 0.8 - i * (r * 1.9 / n)
            spikes.lineTo(hx - 0.05, yy)
        spikes.lineTo(hx, hy - r * 1.75)
        spikes.lineTo(hx + 0.05, hy - r * 0.9)
        spikes.lineTo(hx + 0.05, hy + r * 0.8)
        spikes.closeSubpath()
        p.drawPath(spikes)
        for i in range(4):
            yy = hy - r * 0.9 + i * 0.07
            p.drawPolygon(QPolygonF([QPointF(hx - 0.05, yy), QPointF(hx - 0.16 + i * 0.01, yy - 0.1),
                                     QPointF(hx + 0.05, yy + 0.02)]))


def _head_front(p: QPainter, ch: dict, hx: float, hy: float, r: float, pose: Pose) -> None:
    style, hair = ch["style"], ch["hair"]
    skin = qc(ch["skin"])
    dark = qc("#2a1a10")
    p.setPen(NOPEN)
    if style == "ponytail":
        p.setBrush(qc(hair).darker(110))
        p.drawEllipse(QPointF(hx + r * 1.05, hy + 0.08), 0.07, 0.17)
    if style == "puffs":
        p.setBrush(qc(hair))
        for sx in (-1, 1):
            p.drawEllipse(QPointF(hx + sx * r * 0.75, hy - r * 0.85), r * 0.62, r * 0.6)
    p.setBrush(skin.darker(106))
    p.drawEllipse(QPointF(hx - r * 0.98, hy + 0.02), 0.04, 0.05)
    p.drawEllipse(QPointF(hx + r * 0.98, hy + 0.02), 0.04, 0.05)
    p.setBrush(skin)
    p.drawEllipse(QPointF(hx, hy), r, r * 1.03)
    # hair / hat
    if style in ("cap", "beanie"):
        p.setBrush(dark)
        p.drawChord(QRectF(hx - r, hy - r * 1.02, r * 2, r * 1.3), 0, 180 * 16)
        g = QLinearGradient(hx, hy - r, hx, hy)
        g.setColorAt(0, qc(hair).lighter(125))
        g.setColorAt(1, qc(hair).darker(110))
        p.setBrush(g)
        top = 0.62 if style == "cap" else 0.85
        p.drawChord(QRectF(hx - r * 1.05, hy - r * 1.1, r * 2.1, r * 2 * top), 0, 180 * 16)
        p.setBrush(qc(hair).darker(130))
        if style == "cap":
            p.drawRoundedRect(QRectF(hx - 0.045, hy - r * 0.62, 0.09, 0.05), 0.02, 0.02)
        else:
            p.drawRoundedRect(QRectF(hx - r * 1.06, hy - r * 0.4, r * 2.12, r * 0.36), 0.03, 0.03)
            p.setBrush(qc(ch["accent"]))
            p.drawEllipse(QPointF(hx, hy - r * 1.18), 0.07, 0.07)
    elif style == "ponytail":
        p.setBrush(qc(hair))
        bang = QPainterPath()
        bang.moveTo(hx - r * 1.02, hy + r * 0.1)
        bang.arcTo(QRectF(hx - r * 1.05, hy - r * 1.08, r * 2.1, r * 2.0), 190, -200)
        bang.quadTo(hx + r * 0.2, hy - r * 0.2, hx - r * 0.3, hy - r * 0.55)
        bang.quadTo(hx - r * 0.6, hy - r * 0.2, hx - r * 1.02, hy + r * 0.1)
        p.drawPath(bang)
    elif style == "puffs":
        p.setBrush(qc(hair))
        p.drawChord(QRectF(hx - r * 1.03, hy - r * 1.05, r * 2.06, r * 1.2), 0, 180 * 16)
        p.setBrush(NOBRUSH)
        p.setPen(pen(ch["accent"], 0.04))
        p.drawArc(QRectF(hx - r * 1.1, hy - r * 1.15, r * 2.2, r * 2.1), 10 * 16, 160 * 16)
        p.setPen(NOPEN)
        for sx in (-1, 1):
            p.setBrush(qc("#20242c"))
            p.drawEllipse(QPointF(hx + sx * r * 1.02, hy + 0.02), 0.07, 0.085)
            p.setBrush(qc(ch["accent"]))
            p.drawEllipse(QPointF(hx + sx * r * 1.07, hy + 0.02), 0.045, 0.06)
    else:
        p.setBrush(qc("#3a2a20", 110))
        p.drawChord(QRectF(hx - r, hy - r * 1.02, r * 2, r * 1.1), 0, 180 * 16)
        p.setBrush(qc(hair))
        for i in range(5):
            xx = hx - 0.04 + i * 0.02
            p.drawPolygon(QPolygonF([QPointF(xx - 0.035, hy - r * 0.8), QPointF(xx, hy - r * 1.75 + abs(i - 2) * 0.05),
                                     QPointF(xx + 0.035, hy - r * 0.8)]))
    # face
    blink = (pose.t % 3.7) < 0.12
    for sx in (-1, 1):
        ex, ey = hx + sx * 0.062, hy + 0.012
        if blink:
            p.setPen(pen("#2a1a10", 0.014))
            p.drawLine(QPointF(ex - 0.025, ey), QPointF(ex + 0.025, ey))
            p.setPen(NOPEN)
        else:
            p.setBrush(qc("#ffffff"))
            p.drawEllipse(QPointF(ex, ey), 0.032, 0.042)
            p.setBrush(qc("#2a1a10"))
            p.drawEllipse(QPointF(ex + 0.008, ey + 0.004), 0.018, 0.024)
            p.setBrush(qc("#ffffff"))
            p.drawEllipse(QPointF(ex + 0.013, ey - 0.008), 0.006, 0.006)
        p.setPen(pen("#2a1a10", 0.016))
        p.drawLine(QPointF(ex - 0.03, ey - 0.06 + sx * 0.004), QPointF(ex + 0.03, ey - 0.065 - sx * 0.006))
        p.setPen(NOPEN)
    p.setBrush(qc("#ff7a7a", 90))
    p.drawEllipse(QPointF(hx - 0.095, hy + 0.06), 0.03, 0.018)
    p.drawEllipse(QPointF(hx + 0.095, hy + 0.06), 0.03, 0.018)
    p.setBrush(NOBRUSH)
    p.setPen(pen("#6a2a1a", 0.016))
    p.drawArc(QRectF(hx - 0.045, hy + 0.04, 0.1, 0.06), 200 * 16, 150 * 16)


def draw_runner(p: QPainter, x: float, y: float, s: float, ch: dict, pose: Pose) -> None:
    """The runner with the feet at screen ``(x, y)`` and ``s`` pixels per metre."""
    p.save()
    p.translate(x, y)
    p.scale(s, s)
    b = ch["build"]
    if pose.board:
        _board(p, pose)
        p.translate(0, -0.2)
    if pose.mode == "roll":
        _roll_ball(p, ch, pose, b)
        p.restore()
        return
    mode = pose.mode
    ph = pose.phase
    tilt = pose.lean * 9.0
    if mode == "stumble":
        tilt += math.sin(pose.t * 30) * 12
    p.rotate(tilt)
    if mode == "dead":
        p.scale(1.12, 0.9)
    bob = -0.045 * abs(math.cos(ph)) if mode == "run" else 0.0
    if mode == "idle":
        bob = -0.012 * math.sin(pose.t * 2.2)
    hip_y = -0.9 + bob
    sh_y = hip_y - 0.56
    # legs
    if mode in ("run", "stumble"):
        lifts = (max(0.0, math.sin(ph)), max(0.0, -math.sin(ph)))
        spread = 0.0
    elif mode == "jump":
        lifts = (0.6, 0.5) if pose.rising else (0.25, 0.18)
        spread = 0.04
    elif mode == "fly":
        lifts = (0.08 + 0.05 * math.sin(pose.t * 5), 0.08 - 0.05 * math.sin(pose.t * 5))
        spread = 0.02
    elif mode == "board":
        lifts = (0.1, 0.1)
        spread = 0.07
    elif mode == "dead":
        lifts = (0.0, 0.0)
        spread = 0.2
    else:
        lifts = (0.0, 0.0)
        spread = 0.02
    order = (0, 1) if lifts[0] <= lifts[1] else (1, 0)
    if pose.front:
        order = (1, 0) if lifts[0] <= lifts[1] else (0, 1)
    for i in order:
        sd = -1 if i == 0 else 1
        _leg(p, ch, (sd * 0.1 * b, hip_y + 0.08), sd, lifts[i], b, spread, pose.front)
    # hips and torso
    p.setPen(NOPEN)
    p.setBrush(qc(ch["pants"]))
    p.drawRoundedRect(QRectF(-0.19 * b, hip_y - 0.04, 0.38 * b, 0.18), 0.06, 0.06)
    twist = 0.025 * math.sin(ph) if mode == "run" else 0.0
    torso = QPainterPath()
    torso.moveTo(-0.19 * b, hip_y + 0.04)
    torso.lineTo(0.19 * b, hip_y + 0.04)
    torso.lineTo(0.235 * b + twist, sh_y + 0.11)
    torso.quadTo(0.25 * b + twist, sh_y - 0.01, 0.13 * b + twist, sh_y - 0.03)
    torso.lineTo(-0.13 * b + twist, sh_y - 0.03)
    torso.quadTo(-0.25 * b + twist, sh_y - 0.01, -0.235 * b + twist, sh_y + 0.11)
    torso.closeSubpath()
    g = QLinearGradient(0, sh_y, 0, hip_y)
    g.setColorAt(0, qc(ch["top"]).lighter(118))
    g.setColorAt(1, qc(ch["top"]).darker(112))
    p.setBrush(g)
    p.drawPath(torso)
    p.setBrush(qc(ch["top"]).darker(135))
    p.drawRoundedRect(QRectF(-0.19 * b, hip_y - 0.03, 0.38 * b, 0.07), 0.03, 0.03)
    p.setBrush(qc(ch["accent"]))
    p.drawRect(QRectF(-0.235 * b + twist, sh_y + 0.16, 0.03, 0.2))
    p.drawRect(QRectF(0.205 * b + twist, sh_y + 0.16, 0.03, 0.2))
    if pose.front:
        p.setBrush(qc(ch["top"]).darker(118))
        p.drawRoundedRect(QRectF(-0.13 * b, hip_y - 0.2, 0.26 * b, 0.13), 0.04, 0.04)
        p.setPen(pen(qc(ch["top"]).darker(150), 0.012))
        p.drawLine(QPointF(0, sh_y + 0.02), QPointF(0, hip_y))
        p.setPen(pen(ch["accent"], 0.016))
        p.drawLine(QPointF(-0.04, sh_y + 0.02), QPointF(-0.05, sh_y + 0.17))
        p.drawLine(QPointF(0.04, sh_y + 0.02), QPointF(0.05, sh_y + 0.17))
        p.setPen(pen(ch["pack"], 0.045))
        p.drawLine(QPointF(-0.15 * b, sh_y + 0.0), QPointF(-0.15 * b, sh_y + 0.32))
        p.drawLine(QPointF(0.15 * b, sh_y + 0.0), QPointF(0.15 * b, sh_y + 0.32))
        p.setPen(NOPEN)
    else:
        p.setBrush(qc(ch["top"]).darker(125))  # the hood, down
        p.drawEllipse(QRectF(-0.13, sh_y - 0.03, 0.26, 0.14))
    # arms
    for sd in (-1, 1):
        sx, sy = sd * 0.21 * b + twist, sh_y + 0.07
        if mode in ("run", "stumble"):
            a = math.sin(ph) * sd
            if mode == "stumble":
                a = math.sin(pose.t * 25 + sd) * 1.4
            elbow = (sd * 0.3 * b, sy + 0.24 - 0.08 * a)
            hand = (sd * (0.27 * b + 0.04 * (1 - a)), sy + 0.44 - 0.22 * a)
            if mode == "stumble":
                hand = (sd * 0.5, sy - 0.25 + 0.1 * a)
                elbow = (sd * 0.38, sy - 0.02)
        elif mode == "jump":
            elbow = (sd * 0.38 * b, sy - 0.02)
            hand = (sd * 0.48 * b, sy - 0.22) if pose.rising else (sd * 0.5 * b, sy + 0.05)
        elif mode == "fly":
            elbow = (sd * 0.3 * b, sy + 0.2)
            hand = (sd * 0.2 * b, sy + 0.32)
        elif mode == "board":
            sway = math.sin(pose.t * 3) * 0.05 * sd
            elbow = (sd * 0.36 * b, sy + 0.12)
            hand = (sd * 0.52 * b, sy + 0.1 + sway)
        elif mode == "dead":
            elbow = (sd * 0.4, sy - 0.1)
            hand = (sd * 0.58, sy - 0.3)
        else:
            elbow = (sd * 0.27 * b, sy + 0.24)
            hand = (sd * 0.26 * b, sy + 0.46 + 0.01 * math.sin(pose.t * 2.2))
        p.setPen(pen(ch["top"], 0.13 * b))
        _poly(p, ((sx, sy), elbow, hand))
        p.setPen(NOPEN)
        p.setBrush(qc(ch["skin"]))
        p.drawEllipse(QPointF(*hand), 0.055, 0.055)
    # back: backpack or jetpack
    if not pose.front:
        if pose.jet:
            _jetpack(p, sh_y, pose.t)
        else:
            bounce = 0.025 * math.sin(2 * ph) if mode == "run" else 0.0
            _backpack(p, ch, sh_y + 0.06 + bounce, b)
    # head
    p.setPen(NOPEN)
    p.setBrush(qc(ch["skin"]).darker(112))
    p.drawRect(QRectF(-0.05, sh_y - 0.1, 0.1, 0.1))
    hy = sh_y - 0.21
    if pose.front:
        _head_front(p, ch, twist, hy, 0.165, pose)
    else:
        _head_back(p, ch, twist, hy, 0.165, pose)
    p.restore()


def _backpack(p: QPainter, ch: dict, top: float, b: float) -> None:
    w, h = 0.34 * b, 0.4
    col = qc(ch["pack"])
    p.setPen(NOPEN)
    g = QLinearGradient(-w / 2, 0, w / 2, 0)
    g.setColorAt(0, col.darker(120))
    g.setColorAt(0.45, col.lighter(115))
    g.setColorAt(1, col.darker(130))
    p.setBrush(g)
    p.drawRoundedRect(QRectF(-w / 2, top, w, h), 0.08, 0.08)
    p.setBrush(col.darker(125))
    flap = QPainterPath()
    flap.moveTo(-w / 2, top + 0.08)
    flap.quadTo(-w / 2, top, -w / 2 + 0.08, top)
    flap.lineTo(w / 2 - 0.08, top)
    flap.quadTo(w / 2, top, w / 2, top + 0.08)
    flap.lineTo(w / 2, top + 0.13)
    flap.quadTo(0, top + 0.19, -w / 2, top + 0.13)
    flap.closeSubpath()
    p.drawPath(flap)
    p.setBrush(col.lighter(108))
    p.drawRoundedRect(QRectF(-w * 0.32, top + h * 0.55, w * 0.64, h * 0.36), 0.04, 0.04)
    p.setPen(pen("#f4f1ea", 0.012))
    p.drawLine(QPointF(-w * 0.26, top + h * 0.6), QPointF(w * 0.26, top + h * 0.6))
    p.setPen(NOPEN)
    p.setBrush(qc(ch["accent"]))
    p.drawEllipse(QPointF(w * 0.2, top + h * 0.7), 0.025, 0.025)


def _jetpack(p: QPainter, sh_y: float, t: float) -> None:
    top = sh_y + 0.02
    for sx in (-0.095, 0.095):
        g = QLinearGradient(sx - 0.08, 0, sx + 0.08, 0)
        g.setColorAt(0, qc("#7d8794"))
        g.setColorAt(0.45, qc("#f2f5f8"))
        g.setColorAt(1, qc("#6b7480"))
        p.setPen(NOPEN)
        p.setBrush(g)
        p.drawRoundedRect(QRectF(sx - 0.075, top, 0.15, 0.46), 0.07, 0.07)
        p.setBrush(qc("#e8323c"))
        p.drawChord(QRectF(sx - 0.075, top - 0.03, 0.15, 0.12), 0, 180 * 16)
        p.setBrush(qc("#3b414a"))
        p.drawRect(QRectF(sx - 0.05, top + 0.45, 0.1, 0.06))
        flick = 0.75 + 0.25 * math.sin(t * 47 + sx * 30)
        flame = QPainterPath()
        fy = top + 0.51
        flame.moveTo(sx - 0.06, fy)
        flame.quadTo(sx - 0.05, fy + 0.25 * flick, sx, fy + 0.52 * flick)
        flame.quadTo(sx + 0.05, fy + 0.25 * flick, sx + 0.06, fy)
        flame.closeSubpath()
        p.setBrush(qc("#ff7a1a", 230))
        p.drawPath(flame)
        inner = QPainterPath()
        inner.moveTo(sx - 0.03, fy)
        inner.quadTo(sx - 0.025, fy + 0.12, sx, fy + 0.28 * flick)
        inner.quadTo(sx + 0.025, fy + 0.12, sx + 0.03, fy)
        inner.closeSubpath()
        p.setBrush(qc("#fff3a0"))
        p.drawPath(inner)


def _board(p: QPainter, pose: Pose) -> None:
    hover = 0.03 * math.sin(pose.t * 5)
    glow = QRadialGradient(0, 0.0, 0.55)
    glow.setColorAt(0, qc("#7ffcff", 170))
    glow.setColorAt(1, qc("#7ffcff", 0))
    p.setPen(NOPEN)
    p.setBrush(glow)
    p.drawEllipse(QPointF(0, -0.02), 0.6, 0.2)
    p.save()
    p.rotate(pose.lean * -6)
    y0 = -0.24 + hover
    p.setBrush(qc("#20242c"))
    p.drawRoundedRect(QRectF(-0.36, y0 + 0.03, 0.72, 0.17), 0.08, 0.08)
    g = QLinearGradient(-0.36, 0, 0.36, 0)
    g.setColorAt(0, qc(pose.board_color).darker(115))
    g.setColorAt(0.5, qc(pose.board_color).lighter(125))
    g.setColorAt(1, qc(pose.board_color).darker(115))
    p.setBrush(g)
    p.drawRoundedRect(QRectF(-0.35, y0, 0.7, 0.16), 0.08, 0.08)
    p.setBrush(qc("#ffffff", 200))
    p.drawRect(QRectF(-0.04, y0 + 0.01, 0.08, 0.14))
    p.setBrush(qc("#e8323c"))
    p.drawRect(QRectF(-0.2, y0 + 0.01, 0.04, 0.14))
    p.drawRect(QRectF(0.16, y0 + 0.01, 0.04, 0.14))
    p.restore()


def _roll_ball(p: QPainter, ch: dict, pose: Pose, b: float) -> None:
    r = 0.4 * b
    cy = -r - 0.02
    th = pose.roll * math.tau * 1.25
    p.setPen(NOPEN)
    g = QRadialGradient(-r * 0.3, cy - r * 0.4, r * 1.6)
    g.setColorAt(0, qc(ch["top"]).lighter(125))
    g.setColorAt(1, qc(ch["top"]).darker(130))
    p.setBrush(g)
    p.drawEllipse(QPointF(0, cy), r, r * 0.98)
    # features on a ball tumbling forward (around the x axis): they slide up over the top and vanish
    feats = ((0.0, "pack"), (1.5, "head"), (math.pi, "pants"), (math.pi + 1.2, "shoes"))
    for a0, kind in feats:
        a = (a0 + th) % math.tau
        if a > math.pi:
            continue
        vis = math.sin(a)
        yy = cy + r * 0.85 * math.cos(a)
        hh = 0.25 * vis
        if kind == "pack":
            p.setBrush(qc(ch["pack"]))
            p.drawRoundedRect(QRectF(-0.16 * b, yy - hh / 2, 0.32 * b, hh), 0.05, 0.05)
        elif kind == "head":
            p.setBrush(qc(ch["hair"]) if ch["style"] != "mohawk" else qc(ch["skin"]))
            p.drawEllipse(QPointF(0, yy), 0.15, 0.15 * vis + 0.01)
        elif kind == "pants":
            p.setBrush(qc(ch["pants"]))
            p.drawRoundedRect(QRectF(-0.24 * b, yy - hh * 0.6, 0.48 * b, hh * 1.2), 0.06, 0.06)
        else:
            p.setBrush(qc(ch["shoes"]))
            p.drawEllipse(QPointF(-0.11, yy), 0.08, 0.07 * vis + 0.01)
            p.drawEllipse(QPointF(0.11, yy), 0.08, 0.07 * vis + 0.01)
    p.setBrush(qc(ch["accent"]))
    for sd in (-1, 1):
        p.drawEllipse(QPointF(sd * r * 0.92, cy), 0.035, r * 0.5)
    # the hat / hair peeking over the tuck, and swooshes for the spin
    p.setBrush(qc(ch["hair"] if ch["style"] != "mohawk" else ch["skin"]))
    p.drawChord(QRectF(-0.17, cy - r * 1.02, 0.34, 0.2), 0, 180 * 16)
    p.setBrush(NOBRUSH)
    for k in range(3):
        a = (pose.roll * 900 + k * 40) % 360
        p.setPen(pen(qc("#ffffff", 150 - k * 40), 0.035))
        p.drawArc(QRectF(-r * 1.25, cy - r * 1.25, r * 2.5, r * 2.5), int((a + 20) * 16), 50 * 16)
        p.drawArc(QRectF(-r * 1.25, cy - r * 1.25, r * 2.5, r * 2.5), int((a + 200) * 16), 50 * 16)
    p.setPen(NOPEN)


# --- the yard drone (2D, for the home card) --------------------------------------------------------------------------

def draw_drone(p: QPainter, x: float, y: float, s: float, t: float) -> None:
    """The yard security drone centred on screen ``(x, y)``, ``s`` pixels per metre."""
    p.save()
    p.translate(x, y)
    p.scale(s, s)
    p.setPen(NOPEN)
    for sx in (-1, 1):
        p.setBrush(qc("#2a2f37"))
        p.drawRect(QRectF(0.35 if sx > 0 else -0.9, -0.04, 0.55, 0.07))
        spin = abs(math.sin(t * 40 + sx))
        p.setBrush(qc("#d9e2ea", 110))
        p.drawEllipse(QPointF(sx * 0.85, -0.1), 0.32, 0.06 + 0.02 * spin)
    g = QRadialGradient(-0.12, -0.12, 0.6)
    g.setColorAt(0, qc("#7d8a9b"))
    g.setColorAt(1, qc("#2f3742"))
    p.setBrush(g)
    p.drawEllipse(QPointF(0, 0), 0.5, 0.22)
    p.setBrush(qc("#5f6b7b"))
    p.drawChord(QRectF(-0.27, -0.3, 0.54, 0.4), 0, 180 * 16)
    flash = 0.5 + 0.5 * math.sin(t * 12)
    p.setBrush(qc("#ff3b30", int(120 + 135 * flash)))
    p.drawEllipse(QPointF(-0.12, -0.3), 0.07, 0.07)
    p.setBrush(qc("#3b7bff", int(120 + 135 * (1 - flash))))
    p.drawEllipse(QPointF(0.12, -0.3), 0.07, 0.07)
    p.setBrush(qc("#111317"))
    p.drawEllipse(QPointF(0, 0.08), 0.12, 0.1)
    p.setBrush(qc("#ff3b30"))
    p.drawEllipse(QPointF(0, 0.08), 0.05, 0.04)
    p.restore()


# --- coins, icons --------------------------------------------------------------------------------------------------

def draw_coin(p: QPainter, cx: float, cy: float, r: float, spin: float = 0.0) -> None:
    """A gold coin; ``spin`` (rad) turns it about the vertical axis."""
    w = max(0.12, abs(math.cos(spin)))
    p.setPen(NOPEN)
    rim = qc("#b06a00")
    if w < 0.3:
        p.setBrush(rim)
        p.drawRoundedRect(QRectF(cx - r * 0.18, cy - r, r * 0.36, r * 2), r * 0.15, r * 0.15)
        p.setBrush(qc("#ffd84a"))
        p.drawRect(QRectF(cx - r * 0.06, cy - r * 0.9, r * 0.12, r * 1.8))
        return
    p.setBrush(rim)
    p.drawEllipse(QPointF(cx + r * 0.08 * (1 - w), cy), r * w, r)
    g = QRadialGradient(cx - r * 0.3 * w, cy - r * 0.35, r * 1.4)
    g.setColorAt(0, qc("#fff7b0"))
    g.setColorAt(0.45, qc("#ffd21f"))
    g.setColorAt(1, qc("#e08a00"))
    p.setBrush(g)
    p.drawEllipse(QPointF(cx, cy), r * w * 0.9, r * 0.9)
    p.setBrush(NOBRUSH)
    p.setPen(pen("#c77800", r * 0.1))
    p.drawEllipse(QPointF(cx, cy), r * w * 0.62, r * 0.62)
    p.setPen(NOPEN)
    p.setBrush(qc("#c77800"))
    star = QPolygonF()
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r * (0.42 if i % 2 == 0 else 0.18)
        star.append(QPointF(cx + math.cos(a) * rr * w, cy + math.sin(a) * rr))
    p.drawPolygon(star)
    p.setBrush(qc("#ffffff", 170))
    p.drawEllipse(QPointF(cx - r * 0.42 * w, cy - r * 0.45), r * 0.16 * w, r * 0.12)


def draw_sparkle(p: QPainter, cx: float, cy: float, r: float, alpha: int = 255) -> None:
    p.setPen(NOPEN)
    p.setBrush(qc("#ffffff", alpha))
    star = QPolygonF()
    for i in range(8):
        a = i * math.pi / 4
        rr = r if i % 2 == 0 else r * 0.2
        star.append(QPointF(cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    p.drawPolygon(star)


def draw_heart(p: QPainter, cx: float, cy: float, r: float) -> None:
    """A heart: what the revive costs."""
    p.save()
    p.translate(cx, cy)
    path = QPainterPath()
    path.moveTo(0, r * 0.95)
    path.cubicTo(-r * 1.25, r * 0.05, -r * 0.95, -r * 1.0, 0, -r * 0.42)
    path.cubicTo(r * 0.95, -r * 1.0, r * 1.25, r * 0.05, 0, r * 0.95)
    g = QLinearGradient(0, -r, 0, r)
    g.setColorAt(0, qc("#ff8aa0"))
    g.setColorAt(1, qc("#d9123c"))
    p.setPen(pen("#5a0a1a", max(1.0, r * 0.1)))
    p.setBrush(g)
    p.drawPath(path)
    p.setPen(NOPEN)
    p.setBrush(qc("#ffffff", 150))
    p.drawEllipse(QPointF(-r * 0.42, -r * 0.42), r * 0.18, r * 0.12)
    p.restore()


def draw_shield_icon(p: QPainter, cx: float, cy: float, r: float) -> None:
    """The bubble shield: a glassy cyan bubble with a spark inside."""
    p.save()
    p.translate(cx, cy)
    g = QRadialGradient(-r * 0.3, -r * 0.35, r * 1.4)
    g.setColorAt(0, qc("#eaffff", 230))
    g.setColorAt(0.6, qc("#7ffcff", 170))
    g.setColorAt(1, qc("#2aa8d8", 200))
    p.setPen(pen("#0b4a6a", max(1.0, r * 0.1)))
    p.setBrush(g)
    p.drawEllipse(QPointF(0, 0), r, r)
    p.setPen(NOPEN)
    draw_sparkle(p, r * 0.1, r * 0.1, r * 0.55, 230)
    p.setBrush(qc("#ffffff", 190))
    p.drawEllipse(QPointF(-r * 0.42, -r * 0.45), r * 0.22, r * 0.13)
    p.restore()


def draw_icon(p: QPainter, kind: str, cx: float, cy: float, r: float) -> None:
    """A power-up / item icon of radius ``r``."""
    p.save()
    p.translate(cx, cy)
    p.setPen(NOPEN)
    if kind == "jetpack":  # balloon lift
        p.setPen(pen("#ffffff", 0.06 * r))
        for bx, by in ((-0.42, -0.38), (0.42, -0.34), (0.0, -0.62)):
            p.drawLine(QPointF(0, 0.85 * r), QPointF(bx * r, (by + 0.3) * r))
        p.setPen(NOPEN)
        for (bx, by), col in (((-0.42, -0.38), "#3fc6ff"), ((0.42, -0.34), "#ffd23f"), ((0.0, -0.62), "#ff4f9a")):
            g = QRadialGradient((bx - 0.12) * r, (by - 0.12) * r, 0.5 * r)
            g.setColorAt(0, qc(col).lighter(150))
            g.setColorAt(1, qc(col))
            p.setBrush(g)
            p.drawEllipse(QPointF(bx * r, by * r), 0.34 * r, 0.4 * r)
    elif kind == "sneakers":  # spring soles
        p.rotate(-10)
        p.setBrush(qc("#e8323c"))
        shoe = QPainterPath()
        shoe.moveTo(-0.85 * r, 0.05 * r)
        shoe.lineTo(-0.8 * r, -0.7 * r)
        shoe.quadTo(-0.35 * r, -0.8 * r, -0.2 * r, -0.4 * r)
        shoe.quadTo(0.5 * r, -0.3 * r, 0.85 * r, -0.05 * r)
        shoe.lineTo(0.85 * r, 0.05 * r)
        shoe.closeSubpath()
        p.drawPath(shoe)
        p.setBrush(qc("#ffffff"))
        p.drawRoundedRect(QRectF(-0.9 * r, 0.0, 1.8 * r, 0.2 * r), 0.1 * r, 0.1 * r)
        p.setPen(pen("#7cff4f", 0.13 * r))
        p.setBrush(NOBRUSH)
        for i in range(3):
            p.drawLine(QPointF(-0.45 * r, (0.3 + i * 0.22) * r), QPointF(0.45 * r, (0.4 + i * 0.22) * r))
        p.setPen(NOPEN)
    elif kind == "magnet":
        p.rotate(-20)
        p.setPen(pen("#e8323c", 0.42 * r, Qt.PenCapStyle.FlatCap))
        p.setBrush(NOBRUSH)
        p.drawArc(QRectF(-0.6 * r, -0.75 * r, 1.2 * r, 1.2 * r), 180 * 16, 180 * 16)
        p.drawLine(QPointF(-0.6 * r, -0.15 * r), QPointF(-0.6 * r, -0.55 * r))
        p.drawLine(QPointF(0.6 * r, -0.15 * r), QPointF(0.6 * r, -0.55 * r))
        p.setPen(NOPEN)
        p.setBrush(qc("#e9edf2"))
        p.drawRect(QRectF(-0.81 * r, -0.92 * r, 0.42 * r, 0.38 * r))
        p.drawRect(QRectF(0.39 * r, -0.92 * r, 0.42 * r, 0.38 * r))
    elif kind == "double":  # the score gem
        top = QPolygonF([QPointF(-0.75 * r, -0.2 * r), QPointF(-0.4 * r, -0.75 * r), QPointF(0.4 * r, -0.75 * r),
                         QPointF(0.75 * r, -0.2 * r)])
        low = QPolygonF([QPointF(-0.75 * r, -0.2 * r), QPointF(0.75 * r, -0.2 * r), QPointF(0, 0.9 * r)])
        p.setBrush(qc("#c79bff"))
        p.drawPolygon(top)
        p.setBrush(qc("#7b3fd9"))
        p.drawPolygon(low)
        p.setBrush(qc("#ffffff", 120))
        p.drawPolygon(QPolygonF([QPointF(-0.4 * r, -0.75 * r), QPointF(-0.1 * r, -0.75 * r),
                                 QPointF(-0.3 * r, -0.2 * r)]))
        outline = QPainterPath()
        outline.addText(0, 0, _font(r * 0.62), "x2")
        br = outline.boundingRect()
        outline.translate(-br.center().x(), 0.25 * r - br.center().y())
        p.setPen(pen("#20242c", 0.08 * r))
        p.setBrush(qc("#ffffff"))
        p.drawPath(outline)
        p.setPen(NOPEN)
    elif kind == "mystery":  # a lost parcel
        p.setBrush(qc("#c58b4c"))
        p.drawRoundedRect(QRectF(-0.8 * r, -0.6 * r, 1.6 * r, 1.25 * r), 0.08 * r, 0.08 * r)
        p.setBrush(qc("#9a6a36"))
        p.drawRect(QRectF(-0.8 * r, -0.6 * r, 1.6 * r, 0.22 * r))
        p.setBrush(qc("#e8d6a0"))
        p.drawRect(QRectF(-0.13 * r, -0.6 * r, 0.26 * r, 1.25 * r))
        p.setBrush(qc("#ffffff"))
        p.drawRect(QRectF(0.25 * r, 0.05 * r, 0.42 * r, 0.3 * r))
        p.setPen(pen("#20242c", 0.05 * r))
        p.drawLine(QPointF(0.3 * r, 0.15 * r), QPointF(0.6 * r, 0.15 * r))
        p.drawLine(QPointF(0.3 * r, 0.25 * r), QPointF(0.5 * r, 0.25 * r))
        p.setPen(NOPEN)
    elif kind == "key":
        draw_heart(p, 0, 0, r * 0.85)
    elif kind == "hoverboard":
        draw_shield_icon(p, 0, 0, r * 0.9)
    elif kind == "coin":
        draw_coin(p, 0, 0, r)
    p.restore()


# --- the logo --------------------------------------------------------------------------------------------------------

_LOGO: dict = {}


def _logo_paths() -> list:
    paths = _LOGO.get("paths")
    if paths is None:
        paths = []
        for word in ("RAIL", "RUSH"):
            path = QPainterPath()
            path.addText(0, 0, _font(100, QFont.Weight.Black, "Impact"), word)
            paths.append(path)
        _LOGO["paths"] = paths
    return paths


def _logo_image(height: int) -> QImage:
    """RAIL RUSH drawn once per size (thick strokes are slow to draw every frame)."""
    key = ("img", height)
    img = _LOGO.get(key)
    if img is not None:
        return img
    a, b = _logo_paths()
    ba, bb = a.boundingRect(), b.boundingRect()
    gap = 24
    total_w = ba.width() + gap + bb.width()
    sc = height / max(ba.height(), 1)
    w, h = int(total_w * sc + 40 * sc + 8), int(ba.height() * sc * 1.5 + 8)
    img, p = _image(w, h)
    p.translate(w / 2, h / 2)
    p.scale(sc, sc)
    p.translate(-total_w / 2, ba.height() / 2)
    pieces = ((a.translated(-ba.left(), -ba.bottom()), ("#fff36b", "#ff9a1a")),
              (b.translated(-bb.left() + ba.width() + gap, -bb.bottom() + 6), ("#7ff9ff", "#2a7bff")))
    for path, _cols in pieces:
        p.setPen(pen("#14161b", 30))
        p.setBrush(qc("#14161b"))
        p.drawPath(path.translated(7, 9))
    for path, _cols in pieces:
        p.setPen(pen("#ffffff", 20))
        p.setBrush(NOBRUSH)
        p.drawPath(path)
    for path, (c1, c2) in pieces:
        br = path.boundingRect()
        g = QLinearGradient(0, br.top(), 0, br.bottom())
        g.setColorAt(0, qc(c1))
        g.setColorAt(1, qc(c2))
        p.setPen(pen("#14161b", 7))
        p.setBrush(g)
        p.drawPath(path)
        p.setPen(NOPEN)
        p.setBrush(qc("#ffffff", 120))
        p.drawRect(QRectF(br.left() + 6, br.top() + 10, br.width() - 12, 8))
    p.end()
    if len(_LOGO) > 12:
        _LOGO.clear()
    _LOGO[key] = img
    return img


def draw_logo(p: QPainter, cx: float, cy: float, height: float, t: float = 0.0) -> None:
    """RAIL RUSH in chunky letters, centred on ``(cx, cy)``."""
    img = _logo_image(max(4, int(height)))
    p.save()
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.translate(cx, cy)
    p.rotate(-5 + 0.8 * math.sin(t * 1.5))
    p.drawImage(QPointF(-img.width() / 2, -img.height() / 2), img)
    p.restore()
