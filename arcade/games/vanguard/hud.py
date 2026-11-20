"""Vanguard's head-up display, painted over the 3D picture: health, abilities, the ultimate, the objective, both
teams, the kill feed, hit markers, name plates, the scoreboard, the hero picker and the result screen. Also the
little pictures every screen shares: ability icons (drawn from a few shapes) and round hero portraits."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF

from .data import (ALLY_RGB, CART_RADIUS, ENEMY_RGB, GOLD, HERO_ORDER, HEROES, POINT_UNLOCK, ROLE_LABEL, ROLE_RGB,
                   ROUNDS_TO_WIN, SEGMENT)
from .maps import track_at

A = Qt.AlignmentFlag
NOPEN = Qt.PenStyle.NoPen
_fonts: dict = {}
_icons: dict = {}
PANEL = QColor(14, 18, 30, 170)


def fnt(size: float, bold: bool = True, italic: bool = False) -> QFont:
    key = (int(size), bold, italic)
    f = _fonts.get(key)
    if f is None:
        f = QFont("Bahnschrift")
        f.setStyleHint(QFont.StyleHint.SansSerif)
        f.setPixelSize(max(int(size), 6))
        f.setBold(bold)
        f.setItalic(italic)
        f.setStretch(88)
        _fonts[key] = f
    return f


def col(rgb: tuple, a: int = 255) -> QColor:
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), max(min(int(a), 255), 0))


def text(p: QPainter, r: QRectF, s: str, size: float, c: QColor, align=A.AlignLeft | A.AlignVCenter,  # noqa: ANN001
         bold: bool = True, italic: bool = False, shadow: bool = True) -> None:
    p.setFont(fnt(size, bold, italic))
    if shadow:
        p.setPen(QColor(0, 0, 0, min(c.alpha(), 170)))
        p.drawText(r.translated(max(size * 0.06, 1), max(size * 0.06, 1)), int(align), s)
    p.setPen(c)
    p.drawText(r, int(align), s)


def slant(r: QRectF, k: float = 0.25) -> QPolygonF:
    """A panel leaning to the right."""
    d = r.height() * k
    return QPolygonF([QPointF(r.left() + d, r.top()), QPointF(r.right(), r.top()),
                      QPointF(r.right() - d, r.bottom()), QPointF(r.left(), r.bottom())])


# --- ability icons ---------------------------------------------------------------------------------------------------
# shapes in a 0-1 square: ("p", points) filled polygon, ("c", x, y, r) disc, ("r", x, y, r, w) ring,
# ("l", x0, y0, x1, y1, w) line
_SHAPES = {
    "bulldoze": [("p", [(.55, .2), (.9, .2), (.8, .8), (.45, .8)]), ("l", .1, .35, .4, .35, .07),
                 ("l", .05, .5, .38, .5, .07), ("l", .1, .65, .36, .65, .07)],
    "barricade": [("p", [(.12, .2), (.88, .2), (.88, .8), (.12, .8)]), ("l", .37, .2, .37, .8, .05, 0),
                  ("l", .63, .2, .63, .8, .05, 0), ("l", .12, .5, .88, .5, .05, 0)],
    "pile_driver": [("p", [(.38, .1), (.62, .1), (.62, .5), (.78, .5), (.5, .78), (.22, .5), (.38, .5)]),
                    ("l", .08, .88, .92, .88, .08), ("l", .14, .7, .06, .58, .06), ("l", .86, .7, .94, .58, .06)],
    "heat_sink": [("r", .5, .5, .34, .09), ("p", [(.5, .3), (.64, .5), (.5, .7), (.36, .5)])],
    "slag_pool": [("p", [(.5, .1), (.66, .4), (.5, .56), (.34, .4)]), ("p", [(.1, .76), (.3, .64), (.7, .64),
                                                                            (.9, .76), (.7, .88), (.3, .88)])],
    "meltdown": [("p", [(.5, .06), (.74, .42), (.82, .66), (.66, .9), (.34, .9), (.18, .66), (.3, .4), (.42, .5)])],
    "vault_boots": [("p", [(.5, .08), (.82, .42), (.62, .42), (.62, .62), (.38, .62), (.38, .42), (.18, .42)]),
                    ("l", .25, .78, .75, .78, .08), ("l", .33, .92, .67, .92, .07)],
    "marker_flare": [("r", .5, .5, .3, .07), ("l", .5, .06, .5, .32, .07), ("l", .5, .68, .5, .94, .07),
                     ("l", .06, .5, .32, .5, .07), ("l", .68, .5, .94, .5, .07), ("c", .5, .5, .09)],
    "deadline": [("r", .5, .5, .36, .08), ("l", .5, .5, .5, .24, .08), ("l", .5, .5, .7, .58, .08),
                 ("c", .5, .5, .07)],
    "dash": [("p", [(.1, .2), (.34, .2), (.56, .5), (.34, .8), (.1, .8), (.32, .5)]),
             ("p", [(.46, .2), (.7, .2), (.92, .5), (.7, .8), (.46, .8), (.68, .5)])],
    "second_wind": [("p", [(.4, .14), (.6, .14), (.6, .4), (.86, .4), (.86, .6), (.6, .6), (.6, .86), (.4, .86),
                           (.4, .6), (.14, .6), (.14, .4), (.4, .4)])],
    "rush_hour": [("p", [(.58, .06), (.24, .54), (.46, .54), (.38, .94), (.76, .42), (.54, .42)])],
    "rocket_hop": [("p", [(.5, .06), (.76, .36), (.6, .36), (.6, .56), (.4, .56), (.4, .36), (.24, .36)]),
                   ("p", [(.5, .62), (.62, .76), (.84, .74), (.7, .88), (.5, .96), (.3, .88), (.16, .74),
                          (.38, .76)])],
    "prox_mine": [("c", .5, .56, .26), ("l", .5, .08, .5, .3, .08), ("l", .14, .34, .28, .44, .07),
                  ("l", .86, .34, .72, .44, .07), ("l", .1, .9, .9, .9, .07)],
    "carpet_charge": [("c", .2, .64, .13), ("c", .5, .5, .16), ("c", .8, .36, .13), ("l", .08, .9, .92, .9, .06)],
    "flask": [("p", [(.4, .08), (.6, .08), (.6, .34), (.84, .78), (.76, .92), (.24, .92), (.16, .78), (.4, .34)])],
    "adrenal": [("p", [(.5, .06), (.8, .4), (.62, .4), (.62, .7), (.38, .7), (.38, .4), (.2, .4)]),
                ("l", .26, .88, .74, .88, .09)],
    "monsoon": [("c", .34, .34, .18), ("c", .58, .28, .22), ("c", .72, .4, .15), ("l", .3, .6, .22, .86, .07),
                ("l", .52, .6, .44, .9, .07), ("l", .74, .6, .66, .86, .07)],
    "pylon": [("l", .5, .34, .5, .9, .1), ("c", .5, .26, .13), ("r", .5, .26, .3, .06), ("l", .3, .9, .7, .9, .1)],
    "hardlight": [("p", [(.5, .06), (.86, .3), (.86, .62), (.5, .94), (.14, .62), (.14, .3)]),
                  ("p", [(.5, .3), (.66, .46), (.5, .7), (.34, .46)], 0)],
    "lighthouse": [("p", [(.4, .4), (.6, .4), (.68, .92), (.32, .92)]), ("c", .5, .26, .14),
                   ("l", .5, .26, .06, .14, .06), ("l", .5, .26, .94, .14, .06), ("l", .5, .26, .06, .4, .06),
                   ("l", .5, .26, .94, .4, .06)],
    "melee": [("p", [(.2, .5), (.4, .3), (.8, .3), (.8, .7), (.4, .7)]), ("l", .08, .4, .2, .4, .06),
              ("l", .04, .5, .16, .5, .06), ("l", .08, .6, .2, .6, .06)],
    "tank": [("p", [(.5, .06), (.88, .22), (.88, .56), (.5, .94), (.12, .56), (.12, .22)])],
    "damage": [("p", [(.36, .9), (.36, .4), (.5, .08), (.64, .4), (.64, .9)]), ("l", .2, .9, .2, .52, .1),
               ("l", .8, .9, .8, .52, .1)],
    "support": [("p", [(.38, .1), (.62, .1), (.62, .38), (.9, .38), (.9, .62), (.62, .62), (.62, .9), (.38, .9),
                       (.38, .62), (.1, .62), (.1, .38), (.38, .38)])],
}


def icon(p: QPainter, name: str, r: QRectF, c: QColor) -> None:
    """Draw an ability's (or a role's) icon in ``r``."""
    size = int(max(r.width(), 4))
    key = (name, size, c.rgba())
    img = _icons.get(key)
    if img is None:
        img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        q = QPainter(img)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = QColor(10, 14, 24)
        for sh in _SHAPES.get(name, [("c", .5, .5, .3)]):
            kind = sh[0]
            if kind == "p":
                q.setPen(NOPEN)
                q.setBrush(dark if len(sh) > 2 else c)
                q.drawPolygon(QPolygonF([QPointF(x * size, y * size) for x, y in sh[1]]))
            elif kind == "c":
                q.setPen(NOPEN)
                q.setBrush(c)
                q.drawEllipse(QPointF(sh[1] * size, sh[2] * size), sh[3] * size, sh[3] * size)
            elif kind == "r":
                q.setBrush(Qt.BrushStyle.NoBrush)
                q.setPen(QPen(c, sh[4] * size))
                q.drawEllipse(QPointF(sh[1] * size, sh[2] * size), sh[3] * size, sh[3] * size)
            else:
                q.setPen(QPen(dark if len(sh) > 6 else c, sh[5] * size, Qt.PenStyle.SolidLine,
                              Qt.PenCapStyle.RoundCap))
                q.drawLine(QPointF(sh[1] * size, sh[2] * size), QPointF(sh[3] * size, sh[4] * size))
        q.end()
        _icons[key] = img
    p.drawImage(r.topLeft(), img)


def round_portrait(src: QImage, size: int, rgb: tuple, grey: bool = False) -> QImage:
    """A square render of a hero cut to a circle with a coloured rim."""
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    q = QPainter(img)
    q.setRenderHint(QPainter.RenderHint.Antialiasing)
    q.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    path = QPainterPath()
    path.addEllipse(QRectF(2, 2, size - 4, size - 4))
    q.setClipPath(path)
    s = src.convertToFormat(QImage.Format.Format_Grayscale8) if grey else src
    q.drawImage(QRectF(0, 0, size, size), s)
    if grey:
        q.fillRect(QRectF(0, 0, size, size), QColor(10, 12, 20, 120))
    q.setClipping(False)
    q.setBrush(Qt.BrushStyle.NoBrush)
    q.setPen(QPen(col(rgb), max(size * 0.06, 2)))
    q.drawEllipse(QRectF(2, 2, size - 4, size - 4))
    q.end()
    return img


def health_bar(p: QPainter, r: QRectF, a, thin: bool = False) -> None:  # noqa: ANN001 - sim.Actor
    """Health (white), armour (orange), shields (blue) and extra health (green), in segments of 25."""
    h = a.hero
    full = h.total + max(a.overhp, 0.0)
    k = r.width() / max(full, 1.0)
    p.setPen(NOPEN)
    p.setBrush(QColor(0, 0, 0, 150))
    p.drawRect(r)
    x = r.left()
    for cur, mx, c in ((a.hp if a.alive else 0.0, h.hp, QColor(245, 245, 245)),
                       (a.armor if a.alive else 0.0, h.armor, QColor(255, 170, 60)),
                       (a.shield if a.alive else 0.0, h.shield, QColor(90, 190, 255)),
                       (a.overhp if a.alive else 0.0, a.overhp, QColor(150, 240, 120))):
        if mx <= 0:
            continue
        if cur > 0:
            p.setBrush(c)
            p.drawRect(QRectF(x, r.top(), cur * k, r.height()))
        x += mx * k
    if not thin:
        p.setBrush(QColor(8, 10, 18, 200))
        n = int(full / SEGMENT)
        w = max(r.height() * 0.12, 1.0)
        for i in range(1, n + 1):
            xx = r.left() + i * SEGMENT * k
            if xx < r.right() - 1:
                p.drawRect(QRectF(xx - w / 2, r.top(), w, r.height()))


# --- the play HUD ----------------------------------------------------------------------------------------------------
def draw(p: QPainter, w: int, h: int, g) -> None:  # noqa: ANN001 - game.VanguardGame
    m = g.match
    me = m.player
    s = h / 780.0
    r = g.renderer
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    if g.settings["names"]:
        _plates(p, w, h, s, g, m, me, r)
    if m.phase != "over":
        _objective_marker(p, w, h, s, m, me, r)
    if me.alive and m.phase != "over":
        _crosshair(p, w, h, s, g, m, me)
    if m.phase == "over":
        return
    _damage_dirs(p, w, h, s, me)
    _top(p, w, h, s, g, m, me)
    _feed(p, w, h, s, m, me)
    _banners(p, w, h, s, m)
    if m.phase != "over":
        _vitals(p, w, h, s, g, m, me)
        _abilities(p, w, h, s, g, m, me)
        _ult(p, w, h, s, g, m, me)
        if not me.alive:
            _dead(p, w, h, s, m, me)
        elif m.phase == "prep":
            text(p, QRectF(0, h * 0.36, w, 60 * s), f"{max(int(m.phase_t) + 1, 1)}", 56 * s, QColor(255, 255, 255),
                 A.AlignCenter, italic=True)
            text(p, QRectF(0, h * 0.36 + 58 * s, w, 26 * s), "GET READY   ·   H: change hero", 18 * s,
                 QColor(255, 255, 255, 220), A.AlignCenter)
        elif me.protected:
            text(p, QRectF(0, h * 0.74, w, 24 * s), "H: change hero", 17 * s, QColor(255, 255, 255, 200),
                 A.AlignCenter)


def _plates(p, w, h, s, g, m, me, r) -> None:  # noqa: ANN001
    """Names and health bars over heads."""
    for a in m.actors:
        if a is me or not a.alive:
            continue
        mine = a.team == me.team
        if not mine and not ("reveal" in a.fx or (m.visible(me, a) and me.alive)):
            continue
        pt = r.project(a.x, a.y, a.z + a.height + 0.32)
        if pt is None or not (-40 < pt[0] < w + 40 and -20 < pt[1] < h):
            continue
        d = pt[2]
        if d > (70 if mine else 45):
            continue
        k = max(min(1.3 - d / 40.0, 1.0), 0.55) * s
        bw, bh = 64 * k, 6 * k
        x, y = pt[0] - bw / 2, pt[1] - bh
        health_bar(p, QRectF(x, y, bw, bh), a, True)
        rgb = ALLY_RGB if mine else ENEMY_RGB
        if mine or d < 22:
            text(p, QRectF(x - 40 * k, y - 17 * k, bw + 80 * k, 15 * k), a.name if mine else a.hero.name, 13 * k,
                 col(rgb), A.AlignCenter)
        if mine and a.ult_ready:
            p.setPen(NOPEN)
            p.setBrush(col(GOLD))
            p.drawEllipse(QPointF(x + bw + 7 * k, y + bh / 2), 3.5 * k, 3.5 * k)
        if mine and a.frac < 0.5 and me.hero.role == "support":
            text(p, QRectF(x - 40 * k, y - 34 * k, bw + 80 * k, 15 * k), "NEEDS HEALING", 11 * k, QColor(255, 220, 90),
                 A.AlignCenter)


def _objective_marker(p, w, h, s, m, me, r) -> None:  # noqa: ANN001
    ox, oy, oz = m.objective_pos()
    pt = r.project(ox, oy, oz + 3.2)
    d = math.hypot(ox - me.x, oy - me.y)
    if d < (6.5 if m.mode == "control" else CART_RADIUS + 1.0) or m.phase == "over":
        return
    if pt is None:
        return
    x = min(max(pt[0], 40 * s), w - 40 * s)
    y = min(max(pt[1], 150 * s), h - 200 * s)
    c = col(GOLD)
    p.setPen(QPen(QColor(0, 0, 0, 160), 3 * s))
    p.setBrush(c)
    k = 8 * s
    p.drawPolygon(QPolygonF([QPointF(x, y - k), QPointF(x + k, y), QPointF(x, y + k), QPointF(x - k, y)]))
    text(p, QRectF(x - 40 * s, y + 9 * s, 80 * s, 16 * s), f"{int(d)} m", 13 * s, c, A.AlignCenter)


def _crosshair(p, w, h, s, g, m, me) -> None:  # noqa: ANN001
    cx, cy = w / 2, h / 2
    c = col(g.cross_rgb())
    hero = me.hero
    if me.zoom > 0.6:
        p.setPen(QPen(QColor(0, 0, 0, 140), 3 * s))
        p.setBrush(Qt.BrushStyle.NoBrush)
        rr = h * 0.33
        p.drawEllipse(QPointF(cx, cy), rr, rr)
        p.setPen(QPen(QColor(140, 255, 220, 200), 1.6 * s))
        p.drawEllipse(QPointF(cx, cy), rr, rr)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p.drawLine(QPointF(cx + dx * 10 * s, cy + dy * 10 * s), QPointF(cx + dx * rr, cy + dy * rr))
        p.setPen(NOPEN)
        p.setBrush(QColor(255, 80, 70))
        p.drawEllipse(QPointF(cx, cy), 2 * s, 2 * s)
    elif hero.weapon.kind == "cone":
        p.setBrush(Qt.BrushStyle.NoBrush)
        for width, cc in ((4 * s, QColor(0, 0, 0, 120)), (2 * s, c)):
            p.setPen(QPen(cc, width))
            p.drawArc(QRectF(cx - 26 * s, cy - 26 * s, 52 * s, 52 * s), 30 * 16, 120 * 16)
            p.drawArc(QRectF(cx - 26 * s, cy - 26 * s, 52 * s, 52 * s), 210 * 16, 120 * 16)
        p.setPen(NOPEN)
        p.setBrush(c)
        p.drawEllipse(QPointF(cx, cy), 2.2 * s, 2.2 * s)
    else:
        gap = (5 + hero.weapon.spread * 4 + me.kick * 5) * s
        ln = 8 * s
        for width, cc in ((4.4 * s, QColor(0, 0, 0, 130)), (2.2 * s, c)):
            p.setPen(QPen(cc, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            p.drawLine(QPointF(cx - gap - ln, cy), QPointF(cx - gap, cy))
            p.drawLine(QPointF(cx + gap, cy), QPointF(cx + gap + ln, cy))
            p.drawLine(QPointF(cx, cy + gap), QPointF(cx, cy + gap + ln))
            if hero.weapon.kind == "hitscan":
                p.drawLine(QPointF(cx, cy - gap - ln), QPointF(cx, cy - gap))
        p.setPen(NOPEN)
        p.setBrush(c)
        p.drawEllipse(QPointF(cx, cy), 1.8 * s, 1.8 * s)
    if m.hit_t > 0.0 or m.kill_t > 0.0:
        kill = m.kill_t > 0.0
        k = (m.kill_t / 0.5) if kill else (m.hit_t / 0.2)
        cc = QColor(255, 70, 60, int(255 * min(k * 1.5, 1))) if (kill or m.hit_head) else \
            QColor(255, 255, 255, int(255 * min(k * 1.5, 1)))
        a0, a1 = (9 + (1 - k) * 4) * s, (17 + (1 - k) * 4 + (6 if kill else 0)) * s
        p.setPen(QPen(cc, (3.4 if kill else 2.4) * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            p.drawLine(QPointF(cx + dx * a0, cy + dy * a0), QPointF(cx + dx * a1, cy + dy * a1))
    if m.heal_t > 0.0:
        k = m.heal_t / 0.25
        p.setPen(QPen(QColor(120, 255, 170, int(230 * k)), 2.4 * s))
        p.drawLine(QPointF(cx + 22 * s, cy - 6 * s), QPointF(cx + 22 * s, cy + 6 * s))
        p.drawLine(QPointF(cx + 16 * s, cy), QPointF(cx + 28 * s, cy))


def _damage_dirs(p, w, h, s, me) -> None:  # noqa: ANN001
    if not me.hurt_dirs:
        return
    cx, cy = w / 2, h / 2
    rr = h * 0.2
    p.setBrush(Qt.BrushStyle.NoBrush)
    for ang, left in me.hurt_dirs:
        rel = math.degrees(ang - me.yaw)  # 0 ahead, positive to the right
        p.setPen(QPen(QColor(255, 60, 50, int(220 * min(left / 0.6, 1.0))), 7 * s, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(QRectF(cx - rr, cy - rr, rr * 2, rr * 2), int((90 - rel - 16) * 16), 32 * 16)


def _top(p, w, h, s, g, m, me) -> None:  # noqa: ANN001
    cx = w / 2
    y0 = 10 * s
    mine, theirs = me.team, 1 - me.team
    pw = 300 * s
    panel = QRectF(cx - pw / 2, y0, pw, 62 * s)
    p.setPen(NOPEN)
    p.setBrush(PANEL)
    p.drawPolygon(slant(panel, 0.12))
    ac, ec = col(ALLY_RGB), col(ENEMY_RGB)
    if m.mode == "control":
        text(p, QRectF(panel.left() + 18 * s, y0 + 4 * s, 90 * s, 40 * s), f"{int(m.score[mine])}%", 34 * s, ac,
             A.AlignLeft | A.AlignVCenter, italic=True)
        text(p, QRectF(panel.right() - 108 * s, y0 + 4 * s, 90 * s, 40 * s), f"{int(m.score[theirs])}%", 34 * s, ec,
             A.AlignRight | A.AlignVCenter, italic=True)
        for i in range(ROUNDS_TO_WIN):
            p.setBrush(ac if m.wins[mine] > i else QColor(255, 255, 255, 50))
            p.drawEllipse(QPointF(panel.left() + (26 + i * 14) * s, y0 + 51 * s), 4 * s, 4 * s)
            p.setBrush(ec if m.wins[theirs] > i else QColor(255, 255, 255, 50))
            p.drawEllipse(QPointF(panel.right() - (26 + i * 14) * s, y0 + 51 * s), 4 * s, 4 * s)
        # the point: a disc in its owner's colour with the capture going round it
        rr = 22 * s
        oc = QColor(230, 230, 230) if m.owner < 0 else (ac if m.owner == mine else ec)
        locked = m.phase != "live" or m.live_t < POINT_UNLOCK
        p.setBrush(QColor(60, 64, 80) if locked else oc)
        p.drawEllipse(QPointF(cx, y0 + 30 * s), rr, rr)
        if m.cap > 0.0 and m.cap_team >= 0:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(ac if m.cap_team == mine else ec, 5 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            p.drawArc(QRectF(cx - rr - 5 * s, y0 + 30 * s - rr - 5 * s, (rr + 5 * s) * 2, (rr + 5 * s) * 2), 90 * 16,
                      -int(m.cap * 360 * 16))
            p.setPen(NOPEN)
        if locked:
            left = (POINT_UNLOCK - m.live_t) + (m.phase_t if m.phase == "prep" else 0.0)
            text(p, QRectF(cx - rr, y0 + 30 * s - rr, rr * 2, rr * 2), f"{max(int(left) + 1, 0)}", 20 * s,
                 QColor(255, 255, 255), A.AlignCenter, shadow=False)
        else:
            text(p, QRectF(cx - rr, y0 + 30 * s - rr, rr * 2, rr * 2),
                 "!" if m.contested else ("" if m.owner < 0 else "A"), 22 * s, QColor(20, 24, 36), A.AlignCenter,
                 shadow=False)
        sub = "CONTESTED" if m.contested else ("" if locked else
                                               ("HOLD THE POINT" if m.owner == mine else "TAKE THE POINT"))
        if m.overtime:
            sub = "OVERTIME"
        text(p, QRectF(cx - 140 * s, y0 + 64 * s, 280 * s, 20 * s), f"ROUND {m.round}   {sub}", 14 * s,
             col(GOLD) if (m.overtime or m.contested) else QColor(255, 255, 255, 220), A.AlignCenter)
    else:
        attack = m.attack_team == mine
        tc = ac if attack else ec
        bar = QRectF(panel.left() + 26 * s, y0 + 36 * s, pw - 52 * s, 9 * s)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRect(bar)
        k = m.progress()
        p.setBrush(tc)
        p.drawRect(QRectF(bar.left(), bar.top(), bar.width() * k, bar.height()))
        for i, d in enumerate(m.md.checkpoints):
            x = bar.left() + bar.width() * d / m.cart_len
            p.setBrush(col(GOLD) if i >= m.cp else tc)
            p.drawPolygon(QPolygonF([QPointF(x, bar.top() - 5 * s), QPointF(x + 5 * s, bar.center().y()),
                                     QPointF(x, bar.bottom() + 5 * s), QPointF(x - 5 * s, bar.center().y())]))
        x = bar.left() + bar.width() * k
        p.setBrush(QColor(255, 255, 255))
        p.drawRect(QRectF(x - 4 * s, bar.top() - 4 * s, 8 * s, bar.height() + 8 * s))
        t = max(m.time_left, 0.0)
        text(p, QRectF(cx - 60 * s, y0 + 2 * s, 120 * s, 30 * s), f"{int(t) // 60}:{int(t) % 60:02d}", 26 * s,
             col(GOLD) if (m.overtime or t < 30) else QColor(255, 255, 255), A.AlignCenter, italic=True)
        text(p, QRectF(panel.left() + 22 * s, y0 + 4 * s, 90 * s, 24 * s), "ATTACK" if attack else "DEFEND", 15 * s,
             ac, A.AlignLeft | A.AlignVCenter)
        if m.pushers:
            text(p, QRectF(panel.right() - 112 * s, y0 + 4 * s, 90 * s, 24 * s), f"x{min(m.pushers, 3)}", 17 * s, tc,
                 A.AlignRight | A.AlignVCenter)
        sub = "OVERTIME" if m.overtime else ("CONTESTED" if m.contested else (
            ("MOVING" if m.cart_moving > 0 else ("ROLLING BACK" if m.cart_moving < 0 else
                                                 ("ESCORT THE CART" if attack else "STOP THE CART")))))
        text(p, QRectF(cx - 140 * s, y0 + 64 * s, 280 * s, 20 * s), sub, 14 * s,
             col(GOLD) if (m.overtime or m.contested) else QColor(255, 255, 255, 220), A.AlignCenter)
    # both teams
    size = 40 * s
    for team in (mine, theirs):
        lst = m.team_lists[team]
        for i, a in enumerate(lst):
            if team == mine:
                x = panel.left() - (len(lst) - i) * (size + 6 * s) - 8 * s
            else:
                x = panel.right() + 8 * s + i * (size + 6 * s)
            img = g.portrait(a.hero.id, team == mine, not a.alive)
            if img is not None:
                p.drawImage(QRectF(x, y0 + 4 * s, size, size), img)
            if a.is_player:
                p.setBrush(QColor(255, 255, 255))
                p.setPen(NOPEN)
                p.drawPolygon(QPolygonF([QPointF(x + size / 2 - 5 * s, y0 - 1 * s),
                                         QPointF(x + size / 2 + 5 * s, y0 - 1 * s),
                                         QPointF(x + size / 2, y0 + 5 * s)]))
            if not a.alive:
                text(p, QRectF(x, y0 + 4 * s, size, size), f"{max(int(a.respawn_t) + 1, 1)}", 18 * s,
                     QColor(255, 255, 255), A.AlignCenter)
            elif team == mine:
                health_bar(p, QRectF(x + 3 * s, y0 + size + 6 * s, size - 6 * s, 4 * s), a, True)
                if a.ult_ready:
                    p.setPen(NOPEN)
                    p.setBrush(col(GOLD))
                    p.drawEllipse(QPointF(x + size - 5 * s, y0 + 9 * s), 5 * s, 5 * s)


def _feed(p, w, h, s, m, me) -> None:  # noqa: ANN001
    y = 96 * s
    for e in reversed(m.feed):
        age = m.time - e["t"]
        if age > 7.0:
            continue
        al = int(255 * min((7.0 - age) / 1.0, 1.0))
        kc = col(ALLY_RGB if e["kt"] == me.team else ENEMY_RGB, al)
        vc = col(ALLY_RGB if e["vt"] == me.team else ENEMY_RGB, al)
        how = e["how"] or "eliminated"
        if e["head"]:
            how += " ◎"
        p.setFont(fnt(14 * s))
        fm = p.fontMetrics()
        wk, wh_, wv = fm.horizontalAdvance(e["k"]), fm.horizontalAdvance(how), fm.horizontalAdvance(e["v"])
        total = wk + wh_ + wv + 34 * s
        x = w - total - 14 * s
        p.setPen(NOPEN)
        p.setBrush(QColor(14, 18, 30, int(al * 0.62)))
        p.drawRect(QRectF(x - 8 * s, y, total + 16 * s, 22 * s))
        text(p, QRectF(x, y, wk + 4, 22 * s), e["k"], 14 * s, kc)
        text(p, QRectF(x + wk + 12 * s, y, wh_ + 4, 22 * s), how, 13 * s, QColor(255, 255, 255, int(al * 0.8)),
             bold=False)
        text(p, QRectF(x + wk + wh_ + 26 * s, y, wv + 8, 22 * s), e["v"], 14 * s, vc)
        y += 25 * s
    for cl in m.calls:
        al = int(255 * min(cl[2] / 0.5, 1.0))
        c = col(ALLY_RGB if cl[1] else ENEMY_RGB, al)
        r = QRectF(w - 330 * s, y, 316 * s, 26 * s)
        p.setPen(NOPEN)
        p.setBrush(QColor(14, 18, 30, int(al * 0.7)))
        p.drawPolygon(slant(r, 0.3))
        text(p, r.adjusted(14 * s, 0, -14 * s, 0), ("★ " if cl[1] else "⚠ ") + cl[0], 15 * s, c,
             A.AlignRight | A.AlignVCenter, italic=True)
        y += 29 * s


def _banners(p, w, h, s, m) -> None:  # noqa: ANN001
    if m.banners:
        t, sub, rgb, left, dur = m.banners[-1]
        al = int(255 * min(left / 0.4, 1.0, (dur - left) / 0.15 + 0.05))
        r = QRectF(w / 2 - 300 * s, h * 0.2, 600 * s, 46 * s)
        g = QLinearGradient(r.topLeft(), r.topRight())
        g.setColorAt(0, QColor(0, 0, 0, 0))
        g.setColorAt(0.5, QColor(10, 14, 26, int(al * 0.7)))
        g.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(r.adjusted(0, -4 * s, 0, 28 * s if sub else 4 * s), g)
        text(p, r, t, 34 * s, col(rgb, al), A.AlignCenter, italic=True)
        if sub:
            text(p, QRectF(r.left(), r.bottom() - 2 * s, r.width(), 24 * s), sub, 17 * s, QColor(255, 255, 255, al),
                 A.AlignCenter, bold=False)
    y = h * 0.62
    for t, left in m.elim_text:
        al = int(255 * min(left / 0.4, 1.0))
        text(p, QRectF(0, y, w, 26 * s), t, 20 * s, QColor(255, 255, 255, al) if t.startswith("ASSIST")
             else QColor(255, 90, 80, al), A.AlignCenter, italic=True)
        y += 26 * s


def _vitals(p, w, h, s, g, m, me) -> None:  # noqa: ANN001
    hero = me.hero
    x, y = 26 * s, h - 96 * s
    img = g.portrait(hero.id, True, not me.alive, 72)
    if img is not None:
        p.drawImage(QRectF(x, y, 72 * s, 72 * s), img)
    bx = x + 84 * s
    text(p, QRectF(bx, y + 2 * s, 300 * s, 22 * s), hero.name, 20 * s, QColor(255, 255, 255), italic=True)
    icon(p, hero.role, QRectF(bx + 300 * s - 20 * s, y + 3 * s, 18 * s, 18 * s), col(ROLE_RGB[hero.role]))
    bar = QRectF(bx, y + 28 * s, 300 * s, 20 * s)
    p.setPen(NOPEN)
    p.setBrush(PANEL)
    p.drawRect(bar.adjusted(-4 * s, -4 * s, 4 * s, 4 * s))
    health_bar(p, bar, me)
    cur = int(math.ceil(me.total)) if me.alive else 0
    text(p, QRectF(bx, y + 50 * s, 300 * s, 24 * s), f"{cur}", 24 * s,
         QColor(255, 90, 80) if me.alive and me.frac < 0.3 else QColor(255, 255, 255), italic=True)
    text(p, QRectF(bx + 62 * s, y + 54 * s, 100 * s, 20 * s), f"/ {hero.total}", 14 * s, QColor(255, 255, 255, 170),
         bold=False)
    tags = []
    fx = me.fx
    for key, label, rgb in (("burn", "BURNING", (255, 140, 50)), ("slow", "SLOWED", (150, 200, 255)),
                            ("mark", "MARKED", (140, 255, 220)), ("stun", "STAGGERED", (255, 230, 90)),
                            ("boost", "BOOSTED", (255, 120, 220)), ("haste", "FAST", (120, 240, 255)),
                            ("dr", "GUARDED", (255, 236, 150))):
        if key in fx:
            tags.append((label, rgb))
    if me.protected and me.alive:
        tags.append(("SAFE IN SPAWN", (160, 255, 190)))
    tx = bx + 130 * s
    for label, rgb in tags[:3]:
        p.setFont(fnt(12 * s))
        tw = p.fontMetrics().horizontalAdvance(label) + 12 * s
        p.setPen(NOPEN)
        p.setBrush(col(rgb, 220))
        p.drawPolygon(slant(QRectF(tx, y + 54 * s, tw, 17 * s), 0.3))
        text(p, QRectF(tx, y + 54 * s, tw, 17 * s), label, 12 * s, QColor(14, 18, 30), A.AlignCenter, shadow=False)
        tx += tw + 5 * s


def _slot(p, r, s, name, key, ready, cd, charges, active, hero_rgb) -> None:  # noqa: ANN001
    p.setPen(NOPEN)
    p.setBrush(col(hero_rgb, 235) if active else (QColor(240, 240, 246, 230) if ready else QColor(20, 24, 38, 200)))
    p.drawPolygon(slant(r, 0.14))
    ic = QRectF(r.center().x() - r.height() * 0.33, r.top() + r.height() * 0.1, r.height() * 0.66, r.height() * 0.66)
    icon(p, name, ic, QColor(20, 24, 38) if (ready or active) else QColor(255, 255, 255, 90))
    if not ready and cd > 0.05:
        text(p, r.adjusted(0, 0, 0, -r.height() * 0.2), f"{cd:.0f}" if cd >= 1 else f"{cd:.1f}", 22 * s,
             QColor(255, 255, 255), A.AlignCenter)
    text(p, QRectF(r.left(), r.bottom() - 17 * s, r.width(), 15 * s), key, 11 * s,
         QColor(20, 24, 38, 200) if (ready or active) else QColor(255, 255, 255, 170), A.AlignCenter, shadow=False)
    if charges is not None:
        for i in range(charges[1]):
            p.setBrush(col(GOLD) if i < charges[0] else QColor(0, 0, 0, 120))
            p.drawEllipse(QPointF(r.left() + 12 * s + i * 10 * s, r.top() - 6 * s), 3.5 * s, 3.5 * s)


def _abilities(p, w, h, s, g, m, me) -> None:  # noqa: ANN001
    hero = me.hero
    bw, bh = 62 * s, 62 * s
    x = w - 26 * s - bw
    y = h - 30 * s - bh
    fx = me.fx
    act = {"bulldoze": "bulldoze" in fx, "heat_sink": "sink" in fx}
    for i, (ab, key) in reversed(list(enumerate(((hero.a1, g.key_name("Ability 1")),
                                                (hero.a2, g.key_name("Ability 2")))))):
        ready = me.chg[i] > 0 and me.alive
        _slot(p, QRectF(x, y, bw, bh), s, ab.id, key, ready, me.cd[i], (me.chg[i], ab.charges) if ab.charges > 1
              else None, act.get(ab.id, False), hero.glow)
        x -= bw + 8 * s
    _slot(p, QRectF(x + 12 * s, y + 18 * s, bw - 18 * s, bh - 18 * s), s, "melee", g.key_name("Melee"),
          me.melee_cd <= 0 and me.alive, 0.0, None, False, hero.glow)
    # the weapon
    wpn = hero.weapon
    ax = w - 26 * s
    text(p, QRectF(ax - 300 * s, y - 34 * s, 300 * s, 20 * s), wpn.name.upper(), 13 * s, QColor(255, 255, 255, 200),
         A.AlignRight | A.AlignVCenter)
    if me.reload_t > 0.0:
        text(p, QRectF(ax - 300 * s, y - 76 * s, 300 * s, 42 * s), "RELOADING", 22 * s, col(GOLD),
             A.AlignRight | A.AlignVCenter, italic=True)
    else:
        inf = "deadline" in fx
        cur = "∞" if inf else f"{max(int(math.ceil(me.ammo)), 0)}"
        text(p, QRectF(ax - 300 * s, y - 80 * s, 252 * s, 46 * s), cur, 40 * s,
             QColor(255, 90, 80) if me.ammo < wpn.mag * 0.25 and not inf else QColor(255, 255, 255),
             A.AlignRight | A.AlignVCenter, italic=True)
        text(p, QRectF(ax - 44 * s, y - 66 * s, 44 * s, 26 * s), f"/{wpn.mag}", 15 * s, QColor(255, 255, 255, 170),
             A.AlignLeft | A.AlignVCenter, bold=False)


def _ult(p, w, h, s, g, m, me) -> None:  # noqa: ANN001
    hero = me.hero
    cx, cy = w / 2, h - 62 * s
    rr = 34 * s
    k = me.ult / hero.ult.cost
    active = None
    for key in ("melt", "deadline", "rush", "monsoon"):
        if key in me.fx:
            active = me.fx[key] / hero.ult.dur
    p.setPen(NOPEN)
    p.setBrush(PANEL)
    p.drawEllipse(QPointF(cx, cy), rr + 6 * s, rr + 6 * s)
    ready = me.ult_ready and me.alive
    pulse = 0.5 + 0.5 * math.sin(g.time * 6.0)
    if ready:
        p.setBrush(col(hero.glow, 150 + int(100 * pulse)))
        p.drawEllipse(QPointF(cx, cy), rr, rr)
        icon(p, hero.ult.id, QRectF(cx - rr * 0.62, cy - rr * 0.7, rr * 1.24, rr * 1.24), QColor(20, 24, 38))
        text(p, QRectF(cx - 60 * s, cy + rr + 6 * s, 120 * s, 16 * s), g.key_name("Ultimate"), 13 * s,
             col(GOLD), A.AlignCenter)
    else:
        text(p, QRectF(cx - rr, cy - rr, rr * 2, rr * 2), f"{int(k * 100)}", 26 * s, QColor(255, 255, 255),
             A.AlignCenter, italic=True)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(255, 255, 255, 50), 5 * s))
    p.drawEllipse(QPointF(cx, cy), rr + 1 * s, rr + 1 * s)
    p.setPen(QPen(col(GOLD if ready else hero.glow), 5 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
    frac = active if active is not None else k
    p.drawArc(QRectF(cx - rr - 1 * s, cy - rr - 1 * s, (rr + 1 * s) * 2, (rr + 1 * s) * 2), 90 * 16,
              -int(frac * 360 * 16))


def _dead(p, w, h, s, m, me) -> None:  # noqa: ANN001
    p.fillRect(QRectF(0, 0, w, h), QColor(20, 6, 10, 90))
    text(p, QRectF(0, h * 0.36, w, 44 * s), "ELIMINATED", 40 * s, QColor(255, 90, 80), A.AlignCenter, italic=True)
    if m.feed and m.feed[-1]["v"] == me.name and m.feed[-1]["k"]:
        e = m.feed[-1]
        text(p, QRectF(0, h * 0.36 + 44 * s, w, 24 * s), f"by {e['k']}  ({HEROES[e['kh']].name})", 17 * s,
             QColor(255, 255, 255, 220), A.AlignCenter, bold=False)
    if m.phase in ("live", "prep"):
        text(p, QRectF(0, h * 0.36 + 76 * s, w, 30 * s), f"Back in {max(int(me.respawn_t) + 1, 1)}", 24 * s,
             QColor(255, 255, 255), A.AlignCenter)
    nxt = me.next_hero.name if me.next_hero is not None else me.hero.name
    text(p, QRectF(0, h * 0.36 + 110 * s, w, 22 * s), f"H: change hero   ·   next: {nxt}", 15 * s,
         QColor(255, 255, 255, 190), A.AlignCenter, bold=False)


# --- overlays --------------------------------------------------------------------------------------------------------
def scoreboard(p: QPainter, w: int, h: int, g, top: float = 0.2) -> None:  # noqa: ANN001
    m = g.match
    me = m.player
    s = h / 780.0
    pw = 760 * s
    x0 = w / 2 - pw / 2
    y = h * top
    rowh = 30 * s
    p.setPen(NOPEN)
    p.setBrush(QColor(10, 14, 26, 215))
    p.drawRect(QRectF(x0, y, pw, rowh * 12 + 44 * s))
    cols = (("ELIMS", 330), ("ASSISTS", 410), ("DEATHS", 490), ("DAMAGE", 580), ("HEALING", 680))
    for label, cx in cols:
        text(p, QRectF(x0 + (cx - 50) * s, y + 4 * s, 100 * s, 22 * s), label, 12 * s, QColor(255, 255, 255, 170),
             A.AlignCenter)
    text(p, QRectF(x0 + 16 * s, y + 4 * s, 300 * s, 22 * s), f"{m.md.name}  ·  {g.match.difficulty}", 12 * s,
         QColor(255, 255, 255, 170))
    y += 30 * s
    for team in (me.team, 1 - me.team):
        rgb = ALLY_RGB if team == me.team else ENEMY_RGB
        p.setBrush(col(rgb, 200))
        p.setPen(NOPEN)
        p.drawRect(QRectF(x0, y, pw, 4 * s))
        y += 8 * s
        for a in m.team_lists[team]:
            if a.is_player:
                p.setBrush(QColor(255, 255, 255, 30))
                p.setPen(NOPEN)
                p.drawRect(QRectF(x0, y, pw, rowh))
            img = g.portrait(a.hero.id, team == me.team, not a.alive)
            if img is not None:
                p.drawImage(QRectF(x0 + 12 * s, y + 2 * s, rowh - 4 * s, rowh - 4 * s), img)
            icon(p, a.hero.role, QRectF(x0 + 48 * s, y + 7 * s, 16 * s, 16 * s), col(ROLE_RGB[a.hero.role]))
            text(p, QRectF(x0 + 72 * s, y, 110 * s, rowh), a.name, 16 * s, col(rgb))
            text(p, QRectF(x0 + 180 * s, y, 110 * s, rowh), a.hero.name, 14 * s, QColor(255, 255, 255, 200),
                 bold=False)
            st = a.stats
            for (label, cx), v in zip(cols, (st.elims, st.assists, st.deaths, int(st.damage), int(st.healing)),
                                      strict=True):
                text(p, QRectF(x0 + (cx - 50) * s, y, 100 * s, rowh), f"{v:,}", 16 * s, QColor(255, 255, 255),
                     A.AlignCenter)
            y += rowh
        y += 6 * s


def hero_cards(w: int, h: int) -> list:
    """[(hero id, rect)] of the hero picker."""
    s = h / 780.0
    cw, ch, gap = 140 * s, 190 * s, 12 * s
    total = len(HERO_ORDER) * cw + (len(HERO_ORDER) - 1) * gap
    x = w / 2 - total / 2
    y = h * 0.5 - ch / 2
    out = []
    for hid in HERO_ORDER:
        out.append((hid, QRectF(x, y, cw, ch)))
        x += cw + gap
    return out


def hero_picker(p: QPainter, w: int, h: int, g, current: str, hover: str | None, note: str) -> None:  # noqa: ANN001
    s = h / 780.0
    p.fillRect(QRectF(0, 0, w, h), QColor(8, 12, 24, 190))
    text(p, QRectF(0, h * 0.2, w, 44 * s), "CHOOSE YOUR HERO", 38 * s, QColor(255, 255, 255), A.AlignCenter,
         italic=True)
    text(p, QRectF(0, h * 0.2 + 46 * s, w, 22 * s), note, 15 * s, QColor(255, 255, 255, 200), A.AlignCenter,
         bold=False)
    taken = {a.hero.id for a in g.match.team_lists[g.match.player.team] if not a.is_player} if g.match else set()
    for i, (hid, r) in enumerate(hero_cards(w, h)):
        hero = HEROES[hid]
        sel = hid == current
        p.setPen(QPen(col(GOLD), 3 * s) if sel else (QPen(QColor(255, 255, 255, 160), 2 * s) if hid == hover
                                                     else NOPEN))
        p.setBrush(QColor(30, 36, 56, 235) if not sel else QColor(52, 60, 90, 245))
        p.drawPolygon(slant(r, 0.06))
        img = g.portrait(hid, True, False, 96)
        if img is not None:
            pr = r.width() * 0.7
            p.drawImage(QRectF(r.center().x() - pr / 2, r.top() + 12 * s, pr, pr), img)
        text(p, QRectF(r.left(), r.top() + 114 * s, r.width(), 24 * s), hero.name, 19 * s, QColor(255, 255, 255),
             A.AlignCenter, italic=True)
        icon(p, hero.role, QRectF(r.center().x() - 44 * s, r.top() + 143 * s, 15 * s, 15 * s),
             col(ROLE_RGB[hero.role]))
        text(p, QRectF(r.center().x() - 24 * s, r.top() + 140 * s, 90 * s, 20 * s), ROLE_LABEL[hero.role], 12 * s,
             col(ROLE_RGB[hero.role]))
        text(p, QRectF(r.left(), r.top() + 162 * s, r.width(), 18 * s),
             f"{i + 1}" + ("   on your team" if hid in taken else ""), 11 * s, QColor(255, 255, 255, 150),
             A.AlignCenter, bold=False)
    hero = HEROES[hover or current]
    text(p, QRectF(w * 0.15, h * 0.5 + 110 * s, w * 0.7, 50 * s), hero.blurb, 15 * s, QColor(255, 255, 255, 220),
         A.AlignHCenter | A.AlignTop | Qt.TextFlag.TextWordWrap, bold=False)
    text(p, QRectF(0, h * 0.5 + 170 * s, w, 20 * s), "Click or press 1-7   ·   H / Esc: close", 13 * s,
         QColor(255, 255, 255, 160), A.AlignCenter, bold=False)


def result(p: QPainter, w: int, h: int, g) -> None:  # noqa: ANN001
    m = g.match
    s = h / 780.0
    won = m.winner == m.player.team
    k = min(m.over_t / 0.6, 1.0)
    p.fillRect(QRectF(0, 0, w, h), QColor(8, 12, 24, int(150 * k)))
    c = col(ALLY_RGB if won else ENEMY_RGB, int(255 * k))
    text(p, QRectF(0, h * 0.06, w, 80 * s), "VICTORY" if won else "DEFEAT", 76 * s, c, A.AlignCenter, italic=True)
    if m.mode == "control":
        sub = f"{m.md.name}   ·   rounds {m.wins[m.player.team]} - {m.wins[1 - m.player.team]}"
    else:
        sub = f"{m.md.name}   ·   cart {int(m.progress() * 100)}% of the way"
    text(p, QRectF(0, h * 0.06 + 82 * s, w, 24 * s), sub, 17 * s, QColor(255, 255, 255, int(220 * k)), A.AlignCenter,
         bold=False)
    if m.over_t > 0.8:
        scoreboard(p, w, h, g, 0.24)
    if m.over_t > 2.0:
        text(p, QRectF(0, h - 44 * s, w, 24 * s), "Enter / click: continue", 16 * s, QColor(255, 255, 255, 220),
             A.AlignCenter)


def minimap_track(md) -> list:  # noqa: ANN001
    """Points along the track (for menus that draw the route)."""
    if not md.track:
        return []
    n = 40
    total = md.checkpoints[-1]
    return [track_at(md.track, total * i / n)[:2] for i in range(n + 1)]


__all__ = ["QBrush", "draw", "fnt", "health_bar", "hero_cards", "hero_picker", "icon", "result", "round_portrait",
           "scoreboard", "slant", "text"]
