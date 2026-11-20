"""Billboard sprite art, drawn once with QPainter at 4x into RGBA arrays and colour graded: the zombies (the works'
night shift in coveralls and hard hats; walk, attack, rise and death frames), the soldiers of both multiplayer teams
(front / back, run, aim, death), and the props (tonic machines, the Lucky Crate and its warning beacon, the Refit
Press, the breaker, power-ups, trees, barrels, supply crates, drones)."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF, \
    QRadialGradient

from .textures import grade, gun_path, sprite_from_painter


def _c(r: int, g: int, b: int, a: int = 255) -> QColor:
    return QColor(int(r), int(g), int(b), int(a))


def _poly(*pts: tuple[float, float]) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in pts])


def _graded(arr: np.ndarray, style: str) -> np.ndarray:
    out = arr.copy()
    out[..., :3] = grade(arr[..., :3].astype(np.float32), style).astype(np.uint8)
    return out


# --- zombies -------------------------------------------------------------------------------------------------------

ZW, ZH = 60, 100  # image size; world 0.6 x 1.0
ZOMBIE_LOOKS = (
    # coverall top, trousers, skin, hair / hard hat (the night shift of the works)
    ((214, 108, 34), (196, 98, 30), (150, 160, 132), (236, 196, 40)),
    ((52, 84, 128), (44, 70, 108), (162, 158, 128), (90, 70, 40)),
    ((120, 124, 112), (60, 62, 58), (140, 150, 128), (230, 230, 220)),
    ((200, 196, 180), (60, 60, 64), (156, 160, 138), (60, 50, 40)),
)
HARD_HAT = (0, 2)  # the looks that wear one
EYES = ((0.43, 0.165), (0.57, 0.165))


def _zombie(p: QPainter, w: int, h: int, look: int, pose: str) -> None:
    shirt, pants, skin, hair = (_c(*c) for c in ZOMBIE_LOOKS[look % len(ZOMBIE_LOOKS)])
    cx = w / 2
    step = {"walk0": 1, "walk1": -1}.get(pose, 0)
    dark = QColor(20, 14, 12)
    p.setPen(QPen(dark, 1.2))
    # legs
    for side in (-1, 1):
        lift = 5 if side == step else 0
        lean = 2 * side * (1 if side == step else 0)
        leg = QPainterPath()
        leg.moveTo(cx + side * 3, 56)
        leg.lineTo(cx + side * 12 + lean, 56)
        leg.lineTo(cx + side * 11 + lean, h - 6 - lift)
        leg.lineTo(cx + side * 3 + lean, h - 6 - lift)
        leg.closeSubpath()
        g = QLinearGradient(cx, 56, cx + side * 12, 56)
        g.setColorAt(0, pants.darker(130))
        g.setColorAt(1, pants)
        p.setBrush(g)
        p.drawPath(leg)
        p.setBrush(_c(36, 30, 26))
        p.drawRoundedRect(QRectF(cx + side * 3 + lean - (2 if side < 0 else 0), h - 8 - lift, 11, 6), 2, 2)
        # torn cuff with skin showing
        p.setBrush(skin.darker(120))
        p.drawRect(QRectF(cx + side * 4 + lean, h - 14 - lift, 6, 4))
    # torso, hunched
    torso = QPainterPath()
    torso.moveTo(cx - 14, 26)
    torso.cubicTo(cx - 17, 36, cx - 14, 50, cx - 13, 60)
    torso.lineTo(cx + 13, 60)
    torso.cubicTo(cx + 14, 50, cx + 17, 36, cx + 14, 26)
    torso.closeSubpath()
    g = QLinearGradient(cx - 16, 0, cx + 16, 0)
    g.setColorAt(0, shirt.darker(150))
    g.setColorAt(0.45, shirt)
    g.setColorAt(1, shirt.darker(170))
    p.setBrush(g)
    p.drawPath(torso)
    # rips and blood
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(skin.darker(125))
    p.drawPolygon(_poly((cx - 6, 44), (cx + 2, 40), (cx + 4, 50), (cx - 3, 52)))
    p.setBrush(_c(110, 14, 10, 230))
    p.drawEllipse(QRectF(cx + 3, 33, 8, 11))
    p.drawEllipse(QRectF(cx - 11, 50, 7, 6))
    p.setBrush(_c(70, 8, 6, 200))
    p.drawPolygon(_poly((cx - 4, 26), (cx + 4, 26), (cx + 2, 38), (cx - 1, 34)))
    # arms reaching out at the viewer (foreshortened), or a swipe
    p.setPen(QPen(dark, 1.0))
    for side in (-1, 1):
        if pose == "attack" and side == 1:
            arm = _poly((cx + 12, 28), (cx + 18, 26), (cx + 26, 6), (cx + 21, 4))
            hand = QRectF(cx + 18, -1, 10, 10)
        elif pose == "rise":
            arm = _poly((cx + side * 12, 28), (cx + side * 17, 30), (cx + side * 22, 8), (cx + side * 17, 6))
            hand = QRectF(cx + side * 19.5 - 5, 0, 10, 10)
        else:
            sway = 2 if (pose == "walk0") == (side > 0) else -1
            arm = _poly((cx + side * 12, 28), (cx + side * 18, 30), (cx + side * 15, 44 + sway),
                        (cx + side * 9, 42 + sway))
            hand = QRectF(cx + side * 12 - 6, 38 + sway, 12, 11)
        p.setBrush(shirt.darker(140))
        p.drawPolygon(arm)
        hg = QRadialGradient(hand.center(), 7)
        hg.setColorAt(0, skin.lighter(110))
        hg.setColorAt(1, skin.darker(150))
        p.setBrush(hg)
        p.drawEllipse(hand)
        p.setPen(QPen(skin.darker(200), 0.8))
        for k in range(3):
            fx = hand.left() + 3 + k * 3
            p.drawLine(QPointF(fx, hand.top() + 1), QPointF(fx, hand.top() - 2))
        p.setPen(QPen(dark, 1.0))
    # neck and head (bowed, tilted)
    p.setBrush(skin.darker(130))
    p.drawRect(QRectF(cx - 4, 21, 8, 7))
    head = QRectF(cx - 9, 4, 18, 21)
    hg = QRadialGradient(QPointF(cx - 2, 10), 14)
    hg.setColorAt(0, skin.lighter(115))
    hg.setColorAt(1, skin.darker(160))
    p.setBrush(hg)
    p.drawEllipse(head)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(hair)
    if look % len(ZOMBIE_LOOKS) in HARD_HAT:
        p.setPen(QPen(QColor(20, 14, 12), 0.8))
        p.drawChord(QRectF(cx - 10.5, 0.5, 21, 15), 0, 180 * 16)
        p.drawRect(QRectF(cx - 12.5, 7.5, 25, 2.2))  # the brim
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(hair.darker(140))
        p.drawRect(QRectF(cx - 1, 1, 2, 7))
    else:
        p.drawChord(QRectF(cx - 9.5, 3, 19, 14), 0, 180 * 16)
        p.setBrush(_c(*[int(v * 0.6) for v in (hair.red(), hair.green(), hair.blue())]))
        p.drawRect(QRectF(cx + 2, 6, 6, 3))
    # sunken sockets with glowing eyes
    p.setBrush(_c(30, 12, 10))
    for ex in EYES:
        p.drawEllipse(QPointF(ex[0] * w, ex[1] * h), 3.2, 2.6)
    p.setBrush(_c(190, 255, 200))
    for ex in EYES:
        p.drawEllipse(QPointF(ex[0] * w, ex[1] * h), 1.6, 1.3)
    # jaw and blood around the mouth
    p.setBrush(_c(40, 8, 8))
    p.drawEllipse(QRectF(cx - 4, 18, 8, 4))
    p.setBrush(_c(120, 14, 10, 200))
    p.drawPolygon(_poly((cx - 3, 21), (cx + 3, 21), (cx + 1, 27), (cx - 1, 26)))


def _zombie_dead(p: QPainter, w: int, h: int, look: int) -> None:
    shirt, pants, skin, hair = (_c(*c) for c in ZOMBIE_LOOKS[look % len(ZOMBIE_LOOKS)])
    p.setPen(QPen(QColor(20, 14, 12), 1))
    p.setBrush(_c(90, 10, 8, 220))
    p.drawEllipse(QRectF(w * 0.15, h * 0.55, w * 0.7, h * 0.4))
    p.setBrush(pants)
    p.drawRoundedRect(QRectF(w * 0.55, h * 0.35, w * 0.4, h * 0.35), 4, 4)
    p.setBrush(shirt)
    p.drawRoundedRect(QRectF(w * 0.2, h * 0.25, w * 0.42, h * 0.5), 5, 5)
    p.setBrush(skin)
    p.drawEllipse(QRectF(w * 0.04, h * 0.25, w * 0.17, h * 0.45))
    p.drawEllipse(QRectF(w * 0.25, h * 0.66, w * 0.12, h * 0.25))


# --- soldiers ------------------------------------------------------------------------------------------------------

SW_, SH_ = 60, 104
TEAMS = {
    "ally": dict(body=(54, 70, 82), vest=(40, 52, 58), helm=(46, 60, 64), skin=(196, 156, 122), mark=(80, 200, 255)),
    "enemy": dict(body=(112, 86, 60), vest=(70, 56, 42), helm=(40, 36, 34), skin=(150, 112, 86), mark=(255, 90, 40)),
}


def _soldier(p: QPainter, w: int, h: int, team: str, pose: str, back: bool) -> None:
    t = TEAMS[team]
    body, vest, helm, skin, mark = (_c(*t[k]) for k in ("body", "vest", "helm", "skin", "mark"))
    cx = w / 2
    dark = QColor(16, 16, 18)
    p.setPen(QPen(dark, 1.1))
    step = {"run0": 1, "run1": -1}.get(pose, 0)
    for side in (-1, 1):
        lift = 7 if side == step else 0
        g = QLinearGradient(cx, 0, cx + side * 12, 0)
        g.setColorAt(0, body.darker(140))
        g.setColorAt(1, body)
        p.setBrush(g)
        p.drawPolygon(_poly((cx + side * 2, 58), (cx + side * 13, 58), (cx + side * 12, h - 8 - lift),
                            (cx + side * 3, h - 8 - lift)))
        p.setBrush(body.darker(130))
        p.drawRoundedRect(QRectF(cx + side * 7.5 - 5, 76 - lift * 0.5, 10, 7), 2, 2)  # knee pad
        p.setBrush(_c(30, 28, 26))
        p.drawRoundedRect(QRectF(cx + side * 7.5 - 6.5, h - 10 - lift, 13, 8), 2, 2)
    # torso and vest
    p.setBrush(body)
    p.drawPolygon(_poly((cx - 15, 26), (cx + 15, 26), (cx + 14, 60), (cx - 14, 60)))
    vg = QLinearGradient(cx - 14, 0, cx + 14, 0)
    vg.setColorAt(0, vest.darker(140))
    vg.setColorAt(0.5, vest.lighter(110))
    vg.setColorAt(1, vest.darker(150))
    p.setBrush(vg)
    p.drawRoundedRect(QRectF(cx - 12, 29, 24, 26), 3, 3)
    if back:
        p.setBrush(vest.darker(120))
        p.drawRoundedRect(QRectF(cx - 10, 30, 20, 22), 4, 4)  # pack
    else:
        p.setBrush(vest.darker(125))
        for k in range(3):
            p.drawRect(QRectF(cx - 10 + k * 7, 42, 6, 9))  # mag pouches
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(mark)
    p.drawRect(QRectF(cx - 15, 31, 3, 4))  # team patch
    p.setPen(QPen(dark, 1.1))
    # arms and the rifle
    aim = pose in ("aim", "fire")
    if back:
        p.setBrush(body)
        p.drawPolygon(_poly((cx - 15, 28), (cx - 20, 30), (cx - 16, 48), (cx - 11, 46)))
        p.drawPolygon(_poly((cx + 15, 28), (cx + 20, 30), (cx + 16, 48), (cx + 11, 46)))
        p.setBrush(_c(28, 28, 30))
        p.drawRect(QRectF(cx + 8, 22, 4, 24))  # rifle across the back
    else:
        gun_y = 34 if aim else 44
        p.setBrush(body)
        p.drawPolygon(_poly((cx - 15, 28), (cx - 20, 32), (cx - 6, gun_y + 4), (cx - 3, gun_y - 1)))
        p.drawPolygon(_poly((cx + 15, 28), (cx + 19, 32), (cx + 8, gun_y + 6), (cx + 4, gun_y)))
        p.setBrush(_c(30, 30, 32))
        p.drawRect(QRectF(cx - 3, gun_y - 4, 9, 12))  # the rifle seen end-on
        p.setBrush(_c(14, 14, 16))
        p.drawEllipse(QPointF(cx + 1.5, gun_y - 1), 2.5, 2.5)
        p.setBrush(_c(60, 60, 62))
        p.drawRect(QRectF(cx - 1, gun_y - 9, 5, 5))  # sight
        p.setBrush(skin.darker(120))
        p.drawEllipse(QRectF(cx - 7, gun_y + 1, 6, 5))
        p.drawEllipse(QRectF(cx + 5, gun_y + 3, 6, 5))
        if pose == "fire":
            fg = QRadialGradient(QPointF(cx + 1.5, gun_y - 1), 11)
            fg.setColorAt(0, _c(255, 255, 220))
            fg.setColorAt(0.4, _c(255, 190, 60, 230))
            fg.setColorAt(1, _c(255, 120, 20, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(fg)
            p.drawEllipse(QPointF(cx + 1.5, gun_y - 1), 11, 11)
            p.setPen(QPen(dark, 1.1))
    # head
    p.setBrush(skin.darker(115))
    p.drawRect(QRectF(cx - 4, 20, 8, 8))
    hg = QRadialGradient(QPointF(cx - 2, 12), 12)
    hg.setColorAt(0, skin.lighter(110))
    hg.setColorAt(1, skin.darker(150))
    p.setBrush(hg)
    p.drawEllipse(QRectF(cx - 8, 6, 16, 18))
    p.setBrush(helm)
    p.drawChord(QRectF(cx - 10, 2, 20, 18), 0, 180 * 16)
    p.drawRect(QRectF(cx - 10, 10, 20, 3))
    if not back:
        if team == "enemy":
            p.setBrush(_c(26, 24, 22))  # balaclava
            p.drawRect(QRectF(cx - 7, 16, 14, 8))
        p.setBrush(_c(20, 22, 24))
        p.drawRoundedRect(QRectF(cx - 7, 12, 14, 4.5), 2, 2)  # goggles
        p.setBrush(mark.lighter(130))
        p.drawRect(QRectF(cx - 5, 13, 3, 2))


def _soldier_dead(p: QPainter, w: int, h: int, team: str) -> None:
    t = TEAMS[team]
    p.setPen(QPen(QColor(16, 16, 18), 1))
    p.setBrush(_c(90, 10, 8, 200))
    p.drawEllipse(QRectF(w * 0.2, h * 0.5, w * 0.6, h * 0.45))
    p.setBrush(_c(*t["body"]))
    p.drawRoundedRect(QRectF(w * 0.5, h * 0.35, w * 0.45, h * 0.35), 4, 4)
    p.setBrush(_c(*t["vest"]))
    p.drawRoundedRect(QRectF(w * 0.18, h * 0.22, w * 0.4, h * 0.55), 5, 5)
    p.setBrush(_c(*t["helm"]))
    p.drawEllipse(QRectF(w * 0.03, h * 0.25, w * 0.17, h * 0.5))


# --- props ---------------------------------------------------------------------------------------------------------

PERK_SYMBOL = {"jugg": "shield", "swift": "bolt", "double": "drops", "rally": "cross", "stamina": "boot",
               "deadeye": "sight"}


def _perk_machine(p: QPainter, w: int, h: int, color: tuple, symbol: str, on: bool) -> None:
    col = _c(*color)
    body = col if on else col.darker(250)
    g = QLinearGradient(0, 0, w, 0)
    g.setColorAt(0, body.darker(170))
    g.setColorAt(0.35, body)
    g.setColorAt(0.7, body.darker(120))
    g.setColorAt(1, body.darker(220))
    p.setPen(QPen(_c(14, 12, 12), 1.5))
    p.setBrush(g)
    p.drawRoundedRect(QRectF(4, 10, w - 8, h - 14), 4, 4)
    p.setBrush(body.darker(160))
    p.drawRoundedRect(QRectF(1, 2, w - 2, 14), 3, 3)  # the crown
    # the lit sign
    sign = QRectF(9, 20, w - 18, h * 0.38)
    sg = QLinearGradient(sign.topLeft(), sign.bottomLeft())
    hi = col.lighter(190) if on else col.darker(300)
    sg.setColorAt(0, hi)
    sg.setColorAt(1, col.lighter(120) if on else col.darker(350))
    p.setBrush(sg)
    p.drawRoundedRect(sign, 3, 3)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_c(255, 255, 255, 235) if on else _c(70, 70, 70))
    c = sign.center()
    s = sign.height() * 0.32
    if symbol == "shield":
        p.drawPolygon(_poly((c.x() - s, c.y() - s), (c.x() + s, c.y() - s), (c.x() + s, c.y()),
                            (c.x(), c.y() + s * 1.2), (c.x() - s, c.y())))
    elif symbol == "bolt":
        p.drawPolygon(_poly((c.x() + s * 0.3, c.y() - s * 1.2), (c.x() - s * 0.7, c.y() + s * 0.15),
                            (c.x(), c.y() + s * 0.15), (c.x() - s * 0.4, c.y() + s * 1.2),
                            (c.x() + s * 0.8, c.y() - s * 0.25), (c.x() + s * 0.05, c.y() - s * 0.25)))
    elif symbol == "boot":
        p.drawPolygon(_poly((c.x() - s * 0.5, c.y() - s), (c.x() + s * 0.1, c.y() - s), (c.x() + s * 0.1, c.y() + s * 0.2),
                            (c.x() + s, c.y() + s * 0.5), (c.x() + s, c.y() + s), (c.x() - s * 0.6, c.y() + s)))
    elif symbol == "sight":
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(255, 255, 255, 235) if on else _c(70, 70, 70), max(s * 0.25, 1.2)))
        p.drawEllipse(c, s * 0.75, s * 0.75)
        p.drawLine(QPointF(c.x() - s * 1.1, c.y()), QPointF(c.x() + s * 1.1, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - s * 1.1), QPointF(c.x(), c.y() + s * 1.1))
        p.setPen(Qt.PenStyle.NoPen)
    elif symbol == "drops":
        for dx in (-0.55, 0.55):
            p.drawEllipse(QPointF(c.x() + dx * s, c.y() + s * 0.25), s * 0.45, s * 0.55)
            p.drawPolygon(_poly((c.x() + dx * s - s * 0.4, c.y() + s * 0.1), (c.x() + dx * s, c.y() - s),
                                (c.x() + dx * s + s * 0.4, c.y() + s * 0.1)))
    else:
        p.drawRect(QRectF(c.x() - s * 0.3, c.y() - s, s * 0.6, s * 2))
        p.drawRect(QRectF(c.x() - s, c.y() - s * 0.3, s * 2, s * 0.6))
    # bottle window, slot, lights
    p.setBrush(_c(20, 18, 18))
    p.drawRoundedRect(QRectF(12, h * 0.62, w - 24, h * 0.16), 2, 2)
    for k in range(4):
        p.setBrush(col.lighter(150) if on else _c(40, 40, 40))
        p.drawRoundedRect(QRectF(15 + k * (w - 30) / 4, h * 0.64, (w - 34) / 4, h * 0.12), 2, 2)
    p.setBrush(_c(255, 240, 200) if on else _c(50, 50, 50))
    for k in range(5):
        p.drawEllipse(QPointF(8 + k * (w - 16) / 4, 9), 2, 2)


def _box(p: QPainter, w: int, h: int, open_: bool) -> None:
    """The Lucky Crate: a steel works crate, hazard-striped ends, a stencilled clover and LUCKY, the lid up when open."""
    steel = _c(64, 82, 60)
    body = QRectF(2, h * 0.42, w - 4, h * 0.56)
    g = QLinearGradient(body.topLeft(), body.bottomLeft())
    g.setColorAt(0, steel.lighter(130))
    g.setColorAt(1, steel.darker(170))
    p.setPen(QPen(_c(14, 18, 12), 1.4))
    p.setBrush(g)
    p.drawRect(body)
    for x in (2, w - 12):  # hazard-striped ends
        p.save()
        p.setClipRect(QRectF(x, body.top(), 10, body.height()))
        p.fillRect(QRectF(x, body.top(), 10, body.height()), _c(236, 186, 40))
        p.setPen(QPen(_c(20, 20, 20), 3))
        for k in range(-2, 8):
            y0 = body.top() + k * 6
            p.drawLine(QPointF(x, y0 + 10), QPointF(x + 10, y0))
        p.restore()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_c(232, 220, 170, 230))
    c = QPointF(w * 0.36, body.center().y())
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):  # a four-leaf clover
        p.drawEllipse(QPointF(c.x() + dx * 3.2, c.y() + dy * 3.2), 3.4, 3.4)
    f = QFont("Arial", 1)
    f.setPixelSize(int(h * 0.2))
    f.setBold(True)
    p.setFont(f)
    p.setPen(_c(232, 220, 170, 230))
    p.drawText(QRectF(w * 0.44, body.top(), w * 0.44, body.height()), Qt.AlignmentFlag.AlignCenter, "LUCKY")
    p.setPen(QPen(_c(14, 18, 12), 1.4))
    if open_:
        p.setBrush(steel.darker(140))
        p.drawPolygon(_poly((2, h * 0.42), (w - 2, h * 0.42), (w - 8, h * 0.06), (8, h * 0.06)))
        p.setBrush(_c(255, 226, 160))
        p.drawRect(QRectF(8, h * 0.36, w - 16, h * 0.06))
    else:
        p.setBrush(steel)
        p.drawRect(QRectF(2, h * 0.27, w - 4, h * 0.15))


def _pap(p: QPainter, w: int, h: int, on: bool) -> None:
    base = _c(70, 60, 90) if on else _c(40, 38, 46)
    g = QLinearGradient(0, 0, w, 0)
    g.setColorAt(0, base.darker(170))
    g.setColorAt(0.5, base.lighter(120))
    g.setColorAt(1, base.darker(190))
    p.setPen(QPen(_c(12, 10, 16), 1.5))
    p.setBrush(g)
    p.drawRect(QRectF(4, h * 0.3, w - 8, h * 0.68))
    p.setBrush(base.darker(140))
    p.drawRect(QRectF(0, h * 0.2, w, h * 0.14))
    p.setBrush(_c(30, 28, 34))
    p.drawRect(QRectF(10, h * 0.42, w - 20, h * 0.16))  # the slot
    for k in range(6):
        p.setBrush(_c(120, 110, 130))
        p.drawEllipse(QPointF(14 + k * (w - 28) / 5, h * 0.5), 3, 3)
    glow = _c(190, 90, 255) if on else _c(60, 50, 70)
    p.setBrush(glow)
    p.drawRect(QRectF(8, h * 0.66, w - 16, h * 0.08))
    p.setBrush(_c(200, 200, 210) if on else _c(80, 80, 86))
    p.drawEllipse(QRectF(w * 0.3, 0, w * 0.4, h * 0.24))  # the dome
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_c(255, 255, 255, 200) if on else _c(90, 90, 96))
    p.drawEllipse(QRectF(w * 0.36, h * 0.03, w * 0.12, h * 0.08))


def _power(p: QPainter, w: int, h: int, on: bool) -> None:
    p.setPen(QPen(_c(14, 14, 14), 1.5))
    g = QLinearGradient(0, 0, w, 0)
    g.setColorAt(0, _c(70, 74, 70))
    g.setColorAt(1, _c(120, 124, 118))
    p.setBrush(g)
    p.drawRect(QRectF(2, 2, w - 4, h - 4))
    p.setBrush(_c(230, 190, 40))
    for k in range(6):
        p.drawPolygon(_poly((2 + k * w / 6, h - 14), (2 + k * w / 6 + w / 12, h - 14), (2 + k * w / 6, h - 4)))
    p.setBrush(_c(30, 30, 30))
    p.drawRect(QRectF(w * 0.4, h * 0.2, w * 0.2, h * 0.5))
    ly = h * 0.24 if on else h * 0.62
    p.setPen(QPen(_c(200, 200, 200), 4))
    p.drawLine(QPointF(w / 2, h * 0.45), QPointF(w / 2 + (8 if on else -8), ly))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_c(220, 30, 20))
    p.drawEllipse(QPointF(w / 2 + (8 if on else -8), ly), 5, 5)
    p.setBrush(_c(80, 255, 100) if on else _c(255, 60, 40))
    p.drawEllipse(QPointF(w * 0.2, h * 0.15), 4, 4)


def _powerup(p: QPainter, w: int, h: int, kind: str) -> None:
    """A power-up token: an amber glowing disc with a dark steel rim and the icon of what it does."""
    glow = QRadialGradient(QPointF(w / 2, h / 2), w / 2)
    glow.setColorAt(0, _c(255, 236, 170, 255))
    glow.setColorAt(0.55, _c(250, 170, 50, 220))
    glow.setColorAt(1, _c(160, 80, 10, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(glow)
    p.drawEllipse(QRectF(0, 0, w, h))
    p.setPen(QPen(_c(40, 30, 20, 200), 1.2))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(w * 0.12, h * 0.12, w * 0.76, h * 0.76))
    ink = _c(60, 26, 4)
    p.setBrush(_c(255, 250, 235))
    p.setPen(QPen(ink, 1.5))
    s = w * 0.3
    cx, cy = w / 2, h / 2
    if kind == "max_ammo":
        for k in range(3):
            x = cx - s + k * s * 0.8
            p.drawRoundedRect(QRectF(x, cy - s * 0.6, s * 0.5, s * 1.5), 2, 2)
            p.drawEllipse(QRectF(x, cy - s * 1.1, s * 0.5, s * 0.8))
    elif kind == "insta_kill":
        p.drawEllipse(QRectF(cx - s, cy - s, s * 2, s * 1.6))
        p.drawRect(QRectF(cx - s * 0.6, cy + s * 0.4, s * 1.2, s * 0.6))
        p.setBrush(ink)
        p.drawEllipse(QRectF(cx - s * 0.6, cy - s * 0.4, s * 0.5, s * 0.5))
        p.drawEllipse(QRectF(cx + s * 0.1, cy - s * 0.4, s * 0.5, s * 0.5))
    elif kind == "double_points":
        f = QFont("Segoe UI", 1)
        f.setPixelSize(int(s * 1.7))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QPen(_c(255, 250, 235), 2))
        p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "x2")
    elif kind in ("fire_sale", "bonus_points"):
        f = QFont("Segoe UI", 1)
        f.setPixelSize(int(s * 1.9))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QPen(_c(255, 250, 235), 2))
        p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "%" if kind == "fire_sale" else "$")
    elif kind == "nuke":
        p.drawEllipse(QRectF(cx - s, cy - s * 0.6, s * 2, s * 1.6))
        p.drawRect(QRectF(cx - s * 0.3, cy - s * 1.1, s * 0.6, s * 0.6))
        p.setPen(QPen(_c(255, 200, 60), 2))
        p.drawLine(QPointF(cx, cy - s * 1.1), QPointF(cx + s * 0.5, cy - s * 1.5))
    else:  # carpenter: a hammer
        p.save()
        p.translate(cx, cy)
        p.rotate(-35)
        p.drawRect(QRectF(-s * 0.15, -s * 0.6, s * 0.3, s * 1.8))
        p.drawRect(QRectF(-s * 0.8, -s * 0.9, s * 1.6, s * 0.45))
        p.restore()


def _tree(p: QPainter, w: int, h: int, dead: bool) -> None:
    p.setPen(QPen(_c(26, 18, 12), 1.5))
    p.setBrush(_c(70, 50, 34))
    p.drawPolygon(_poly((w * 0.45, h), (w * 0.55, h), (w * 0.53, h * 0.4), (w * 0.47, h * 0.4)))
    if dead:
        p.setPen(QPen(_c(50, 36, 26), 3))
        for a, l in ((-50, 0.3), (40, 0.35), (-20, 0.4), (70, 0.25), (-80, 0.22)):
            r = math.radians(a - 90)
            x0, y0 = w * 0.5, h * 0.45 + l * 30
            p.drawLine(QPointF(x0, y0), QPointF(x0 + math.cos(r) * w * l, y0 + math.sin(r) * h * l * 0.8))
        return
    rng = np.random.default_rng(7)
    p.setPen(Qt.PenStyle.NoPen)
    for _k in range(26):
        x = w * (0.18 + 0.64 * rng.random())
        y = h * (0.06 + 0.42 * rng.random())
        r = w * (0.12 + 0.1 * rng.random())
        g = QRadialGradient(QPointF(x - r * 0.3, y - r * 0.3), r * 1.2)
        g.setColorAt(0, _c(110, 150, 70))
        g.setColorAt(1, _c(40, 70, 30))
        p.setBrush(g)
        p.drawEllipse(QPointF(x, y), r, r * 0.85)


def _barrel(p: QPainter, w: int, h: int, burning: bool) -> None:
    g = QLinearGradient(0, 0, w, 0)
    base = _c(60, 80, 110) if not burning else _c(90, 50, 30)
    g.setColorAt(0, base.darker(180))
    g.setColorAt(0.4, base.lighter(130))
    g.setColorAt(1, base.darker(200))
    p.setPen(QPen(_c(14, 14, 14), 1.2))
    p.setBrush(g)
    top = h * 0.3 if burning else 2
    p.drawRect(QRectF(2, top, w - 4, h - top - 1))
    for y in (0.45, 0.75):
        p.drawLine(QPointF(2, top + (h - top) * y), QPointF(w - 2, top + (h - top) * y))
    if burning:
        p.setPen(Qt.PenStyle.NoPen)
        for col, s in ((_c(255, 110, 20), 1.0), (_c(255, 190, 50), 0.7), (_c(255, 245, 190), 0.4)):
            path = QPainterPath()
            x0, x1 = w * (0.5 - 0.45 * s), w * (0.5 + 0.45 * s)
            path.moveTo(x0, top + 2)
            tongues = 4
            for i in range(tongues):
                a = x0 + (x1 - x0) * i / tongues
                b = x0 + (x1 - x0) * (i + 1) / tongues
                tip = top * (1.0 - s * (0.75 if i % 2 else 1.0)) + 1
                path.quadTo((a + b) / 2 - 2, top * 0.6, (a + b) / 2 + (1 if i % 2 else -1), tip)
                path.quadTo((a + b) / 2 + 2, top * 0.6, b, top + 2)
            path.closeSubpath()
            p.setBrush(col)
            p.drawPath(path)


def _gun(p: QPainter, w: int, h: int, model: str, camo: bool) -> None:
    path = gun_path(model, w, h)
    g = QLinearGradient(0, 0, 0, h)
    if camo:
        g = QLinearGradient(0, 0, w, h)
        for k, col in enumerate(((60, 30, 200), (230, 60, 210), (40, 200, 255), (150, 40, 255), (255, 120, 220))):
            g.setColorAt(k / 4, _c(*col))
    else:
        g.setColorAt(0, _c(120, 124, 130))
        g.setColorAt(0.5, _c(60, 62, 66))
        g.setColorAt(1, _c(30, 30, 34))
    p.setPen(QPen(_c(230, 240, 255, 230), 1.5))
    p.setBrush(QBrush(g))
    p.drawPath(path)


def _beacon(p: QPainter, w: int, h: int, lit: bool) -> None:
    """The red rotating warning light that rises out of the Lucky Crate before a crane lifts it away."""
    p.setPen(QPen(_c(16, 16, 16), 1.2))
    p.setBrush(_c(60, 62, 64))
    p.drawRect(QRectF(w * 0.2, h * 0.78, w * 0.6, h * 0.2))
    dome = QRectF(w * 0.18, h * 0.12, w * 0.64, h * 0.7)
    g = QRadialGradient(dome.center(), dome.width() * 0.6)
    g.setColorAt(0, _c(255, 200, 180) if lit else _c(150, 40, 30))
    g.setColorAt(1, _c(220, 20, 10) if lit else _c(90, 16, 12))
    p.setBrush(g)
    p.drawChord(dome.adjusted(0, 0, 0, dome.height() * 0.5), 0, 180 * 16)
    p.drawRect(QRectF(dome.x(), dome.center().y() + dome.height() * 0.25 - 1, dome.width(), dome.height() * 0.3))
    p.setPen(QPen(_c(40, 10, 8), 1))
    for k in range(3):
        y = dome.top() + dome.height() * (0.3 + k * 0.18)
        p.drawLine(QPointF(dome.left() + 3, y), QPointF(dome.right() - 3, y))


def _crate(p: QPainter, w: int, h: int) -> None:
    g = QLinearGradient(0, 0, 0, h)
    g.setColorAt(0, _c(92, 104, 70))
    g.setColorAt(1, _c(48, 56, 36))
    p.setPen(QPen(_c(20, 22, 14), 1.5))
    p.setBrush(g)
    p.drawRect(QRectF(2, h * 0.3, w - 4, h * 0.68))
    p.setBrush(_c(230, 200, 60))
    p.drawRect(QRectF(2, h * 0.55, w - 4, h * 0.08))
    p.setPen(QPen(_c(255, 60, 40), 2))
    p.drawLine(QPointF(w * 0.5, h * 0.3), QPointF(w * 0.5, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_c(255, 80, 50))
    p.drawEllipse(QPointF(w * 0.5, 3), 3, 3)


def _sentry(p: QPainter, w: int, h: int) -> None:
    p.setPen(QPen(_c(16, 16, 16), 1.4))
    p.setBrush(_c(60, 64, 60))
    for x in (0.2, 0.8):
        p.drawLine(QPointF(w * 0.5, h * 0.55), QPointF(w * x, h - 2))
    p.drawRect(QRectF(w * 0.42, h * 0.45, w * 0.16, h * 0.15))
    p.setBrush(_c(90, 96, 90))
    p.drawRoundedRect(QRectF(w * 0.18, h * 0.18, w * 0.64, h * 0.3), 4, 4)
    p.setBrush(_c(20, 20, 22))
    p.drawEllipse(QPointF(w * 0.5, h * 0.33), w * 0.1, w * 0.1)
    p.setBrush(_c(80, 200, 255))
    p.drawEllipse(QPointF(w * 0.3, h * 0.25), 2.5, 2.5)


def _drone(p: QPainter, w: int, h: int, hostile: bool) -> None:
    p.setPen(QPen(_c(14, 14, 16), 1.2))
    p.setBrush(_c(50, 52, 56))
    p.drawRect(QRectF(w * 0.1, h * 0.45, w * 0.8, h * 0.1))
    p.drawEllipse(QRectF(w * 0.32, h * 0.32, w * 0.36, h * 0.36))
    p.setBrush(_c(120, 120, 126, 200))
    for x in (0.0, 0.7):
        p.drawEllipse(QRectF(w * x, h * 0.38, w * 0.3, h * 0.12))
    p.setBrush(_c(255, 60, 40) if hostile else _c(80, 200, 255))
    p.drawEllipse(QPointF(w * 0.5, h * 0.5), 3, 3)


# --- the sprite set ------------------------------------------------------------------------------------------------

class SpriteSet:
    """Every sprite image for one colour grade (``style``: dusk / day), made on first use."""

    def __init__(self, style: str) -> None:
        self.style = style
        self._cache: dict = {}

    def _get(self, key: tuple, w: int, h: int, draw) -> np.ndarray:  # noqa: ANN001
        img = self._cache.get(key)
        if img is None:
            def big(p: QPainter, ww: int, hh: int) -> None:
                p.scale(4, 4)
                draw(p, w, h)
            img = self._cache[key] = _graded(sprite_from_painter(w * 4, h * 4, big), self.style)
        return img

    def zombie(self, look: int, pose: str) -> np.ndarray:
        if pose == "dead":
            return self._get(("zd", look), 96, 32, lambda p, w, h: _zombie_dead(p, w, h, look))
        return self._get(("z", look, pose), ZW, ZH, lambda p, w, h: _zombie(p, w, h, look, pose))

    def soldier(self, team: str, pose: str, back: bool = False) -> np.ndarray:
        if pose == "dead":
            return self._get(("sd", team), 96, 32, lambda p, w, h: _soldier_dead(p, w, h, team))
        return self._get(("s", team, pose, back), SW_, SH_, lambda p, w, h: _soldier(p, w, h, team, pose, back))

    def perk(self, color: tuple, symbol: str, on: bool) -> np.ndarray:
        return self._get(("perk", color, on), 56, 100, lambda p, w, h: _perk_machine(p, w, h, color, symbol, on))

    def box(self, open_: bool) -> np.ndarray:
        return self._get(("box", open_), 96, 48, lambda p, w, h: _box(p, w, h, open_))

    def pap(self, on: bool) -> np.ndarray:
        return self._get(("pap", on), 80, 88, lambda p, w, h: _pap(p, w, h, on))

    def power(self, on: bool) -> np.ndarray:
        return self._get(("power", on), 44, 64, lambda p, w, h: _power(p, w, h, on))

    def powerup(self, kind: str) -> np.ndarray:
        return self._get(("pu", kind), 48, 48, lambda p, w, h: _powerup(p, w, h, kind))

    def tree(self, dead: bool) -> np.ndarray:
        return self._get(("tree", dead), 96, 160, lambda p, w, h: _tree(p, w, h, dead))

    def barrel(self, burning: bool) -> np.ndarray:
        return self._get(("barrel", burning), 32, 56, lambda p, w, h: _barrel(p, w, h, burning))

    def gun(self, model: str, camo: bool = False) -> np.ndarray:
        return self._get(("gun", model, camo), 96, 40, lambda p, w, h: _gun(p, w, h, model, camo))

    def beacon(self, lit: bool) -> np.ndarray:
        return self._get(("beacon", lit), 40, 48, lambda p, w, h: _beacon(p, w, h, lit))

    def crate(self) -> np.ndarray:
        return self._get(("crate",), 48, 48, _crate)

    def sentry(self) -> np.ndarray:
        return self._get(("sentry",), 48, 56, _sentry)

    def drone(self, hostile: bool) -> np.ndarray:
        return self._get(("drone", hostile), 48, 24, lambda p, w, h: _drone(p, w, h, hostile))

    def grenade(self) -> np.ndarray:
        def draw(p: QPainter, w: int, h: int) -> None:
            p.setPen(QPen(_c(10, 10, 10), 1))
            p.setBrush(_c(60, 70, 50))
            p.drawEllipse(QRectF(2, 4, w - 4, h - 6))
            p.setBrush(_c(120, 120, 120))
            p.drawRect(QRectF(w * 0.4, 0, w * 0.2, 5))
        return self._get(("nade",), 16, 18, draw)


