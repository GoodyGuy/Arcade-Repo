"""BREACH POINT's heads-up display, painted over the 3D picture: score, clock and both rosters at the top, the
minimap, the kill feed, health and armour, the four abilities, ammo and credits, the crosshair, hit and damage marks,
plant and defuse bars, the planted charge's fuse, teammates' name tags, the scoreboard, the buy menu and the end of
the match. Icons (abilities, guns, the charge) are drawn in code and kept as pixmaps."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from . import config as C
from .config import WEAPONS
from .models import gun_parts

AL = Qt.AlignmentFlag
FAMILY = "Bahnschrift"
INK = QColor(*C.INK)
PAPER = QColor(*C.PAPER)
ACCENT = QColor(*C.ACCENT)
ALLY = QColor(*C.ALLY)
ENEMY = QColor(*C.ENEMY)
DIM = QColor(150, 152, 165)
CROSS_COLOURS = (("WHITE", (255, 255, 255)), ("GREEN", (80, 255, 120)), ("CYAN", (70, 235, 255)),
                 ("YELLOW", (255, 230, 60)), ("PINK", (255, 110, 200)))
_fonts: dict = {}
_pix: dict = {}


def fnt(size: float, bold: bool = True) -> QFont:
    key = (int(size), bold)
    f = _fonts.get(key)
    if f is None:
        f = QFont(FAMILY)
        f.setPixelSize(max(int(size), 1))
        f.setBold(bold)
        _fonts[key] = f
    return f


def text(p: QPainter, r: QRectF, s: str, size: float, col: QColor = PAPER, align=AL.AlignCenter,  # noqa: ANN001
         bold: bool = True, shadow: bool = False) -> None:
    p.setFont(fnt(size, bold))
    if shadow:
        p.setPen(QColor(0, 0, 0, 170))
        p.drawText(r.translated(1.5, 1.5), align, s)
    p.setPen(col)
    p.drawText(r, align, s)


def rgba(c: tuple, a: int = 255) -> QColor:
    return QColor(int(c[0]), int(c[1]), int(c[2]), a)


def panel(p: QPainter, r: QRectF, alpha: int = 170, edge: QColor | None = None) -> None:
    p.fillRect(r, QColor(INK.red(), INK.green(), INK.blue(), alpha))
    if edge is not None:
        p.fillRect(QRectF(r.x(), r.y(), 3, r.height()), edge)


# --- icons -----------------------------------------------------------------------------------------------------------

def _icon(p: QPainter, kind: str, s: float, col: QColor) -> None:
    """An ability's sign in a box of ``s`` pixels."""
    pen = QPen(col, max(s * 0.075, 1.4))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    c = s / 2

    def P(x: float, y: float) -> QPointF:
        return QPointF(x * s, y * s)

    if kind == "flash":
        for i in range(8):
            a = i * math.pi / 4
            r0, r1 = (0.14, 0.4) if i % 2 == 0 else (0.14, 0.28)
            p.drawLine(QPointF(c + math.cos(a) * r0 * s, c + math.sin(a) * r0 * s),
                       QPointF(c + math.cos(a) * r1 * s, c + math.sin(a) * r1 * s))
    elif kind == "buff":
        for y in (0.3, 0.5):
            p.drawPolyline(QPolygonF([P(0.25, y + 0.2), P(0.5, y), P(0.75, y + 0.2)]))
    elif kind == "dash":
        p.drawPolyline(QPolygonF([P(0.5, 0.25), P(0.8, 0.5), P(0.5, 0.75)]))
        for y in (0.38, 0.5, 0.62):
            p.drawLine(P(0.18, y), P(0.5 if y == 0.5 else 0.38, y))
    elif kind == "frag":
        p.drawEllipse(P(0.48, 0.58), 0.22 * s, 0.22 * s)
        p.drawPolyline(QPolygonF([P(0.6, 0.4), P(0.7, 0.26), P(0.82, 0.28)]))
    elif kind == "concuss":
        p.drawEllipse(P(0.5, 0.5), 0.07 * s, 0.07 * s)
        for r in (0.2, 0.33):
            p.drawArc(QRectF((0.5 - r) * s, (0.5 - r) * s, 2 * r * s, 2 * r * s), -40 * 16, 80 * 16)
            p.drawArc(QRectF((0.5 - r) * s, (0.5 - r) * s, 2 * r * s, 2 * r * s), 140 * 16, 80 * 16)
    elif kind == "chain":
        for i, x in enumerate((0.24, 0.5, 0.76)):
            p.drawEllipse(P(x, 0.62 - i * 0.12), (0.08 + i * 0.03) * s, (0.08 + i * 0.03) * s)
    elif kind == "slow":
        for y in (0.36, 0.52, 0.68):
            path = QPainterPath(P(0.2, y))
            path.cubicTo(P(0.35, y - 0.12), P(0.45, y + 0.12), P(0.6, y))
            path.cubicTo(P(0.68, y - 0.06), P(0.74, y - 0.02), P(0.8, y))
            p.drawPath(path)
    elif kind == "curtain":
        for x in (0.26, 0.42, 0.58, 0.74):
            p.drawLine(P(x, 0.24), P(x, 0.76))
    elif kind == "smoke":
        p.drawEllipse(P(0.38, 0.56), 0.17 * s, 0.17 * s)
        p.drawEllipse(P(0.62, 0.56), 0.15 * s, 0.15 * s)
        p.drawEllipse(P(0.5, 0.4), 0.16 * s, 0.16 * s)
    elif kind == "nearsight":
        path = QPainterPath(P(0.18, 0.5))
        path.quadTo(P(0.5, 0.2), P(0.82, 0.5))
        path.quadTo(P(0.5, 0.8), P(0.18, 0.5))
        p.drawPath(path)
        p.drawLine(P(0.25, 0.78), P(0.75, 0.22))
    elif kind == "static":
        p.drawPolyline(QPolygonF([P(0.58, 0.18), P(0.36, 0.52), P(0.56, 0.52), P(0.4, 0.84)]))
    elif kind == "sensor":
        p.drawEllipse(P(0.5, 0.66), 0.06 * s, 0.06 * s)
        for r in (0.2, 0.34):
            p.drawArc(QRectF((0.5 - r) * s, (0.66 - r) * s, 2 * r * s, 2 * r * s), 40 * 16, 100 * 16)
    elif kind == "sonar":
        for r in (0.1, 0.22, 0.34):
            p.drawEllipse(P(0.5, 0.5), r * s, r * s)
    elif kind == "sweep":
        p.drawEllipse(P(0.5, 0.5), 0.33 * s, 0.33 * s)
        p.drawLine(P(0.5, 0.5), P(0.74, 0.28))
        p.drawEllipse(P(0.36, 0.6), 0.03 * s, 0.03 * s)
    elif kind == "fire":
        path = QPainterPath(P(0.5, 0.16))
        path.cubicTo(P(0.78, 0.42), P(0.8, 0.84), P(0.5, 0.84))
        path.cubicTo(P(0.2, 0.84), P(0.26, 0.5), P(0.4, 0.42))
        path.cubicTo(P(0.44, 0.52), P(0.52, 0.4), P(0.5, 0.16))
        p.drawPath(path)
    elif kind == "heal":
        p.drawLine(P(0.5, 0.24), P(0.5, 0.76))
        p.drawLine(P(0.24, 0.5), P(0.76, 0.5))
    elif kind == "barricade":
        p.drawRect(QRectF(0.2 * s, 0.3 * s, 0.6 * s, 0.44 * s))
        p.drawLine(P(0.2, 0.52), P(0.8, 0.52))
        p.drawLine(P(0.5, 0.3), P(0.5, 0.52))
        p.drawLine(P(0.36, 0.52), P(0.36, 0.74))
        p.drawLine(P(0.64, 0.52), P(0.64, 0.74))
    elif kind == "stronghold":
        path = QPainterPath(P(0.5, 0.16))
        path.lineTo(P(0.8, 0.28))
        path.cubicTo(P(0.8, 0.6), P(0.66, 0.78), P(0.5, 0.86))
        path.cubicTo(P(0.34, 0.78), P(0.2, 0.6), P(0.2, 0.28))
        path.closeSubpath()
        p.drawPath(path)
    else:
        p.drawEllipse(P(0.5, 0.5), 0.25 * s, 0.25 * s)


def ability_pix(kind: str, size: int, col: tuple) -> QPixmap:
    key = ("ab", kind, size, col)
    pm = _pix.get(key)
    if pm is None:
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        q = QPainter(pm)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        _icon(q, kind, float(size), QColor(*col))
        q.end()
        _pix[key] = pm
    return pm


def weapon_pix(wid: str, w: int, h: int, col: tuple, accent: bool = True) -> QPixmap:
    """The gun from the side."""
    key = ("gun", wid, w, h, col, accent)
    pm = _pix.get(key)
    if pm is None:
        parts = gun_parts(wid)
        z0 = min(cz - hz for (cx, cy, cz), (hx, hy, hz), _c in parts)
        z1 = max(cz + hz for (cx, cy, cz), (hx, hy, hz), _c in parts)
        y0 = min(cy - hy for (cx, cy, cz), (hx, hy, hz), _c in parts)
        y1 = max(cy + hy for (cx, cy, cz), (hx, hy, hz), _c in parts)
        k = min((w - 4) / max(z1 - z0, 0.01), (h - 4) / max(y1 - y0, 0.01), w / 1.25)
        ox = (w - (z1 - z0) * k) / 2 - z0 * k
        oy = (h + (y1 - y0) * k) / 2 + y0 * k
        pm = QPixmap(w, h)
        pm.fill(Qt.GlobalColor.transparent)
        q = QPainter(pm)
        base = QColor(*col)
        for i, ((cx, cy, cz), (hx, hy, hz), c) in enumerate(parts):
            colr = base
            if accent and i == len(parts) - 1 or (accent and c[0] + c[1] + c[2] > 1.6 and c[1] != c[2]):
                colr = QColor(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            q.fillRect(QRectF(ox + (cz - hz) * k, oy - (cy + hy) * k, max(hz * 2 * k, 1.5), max(hy * 2 * k, 1.5)),
                       colr)
        q.end()
        _pix[key] = pm
    return pm


def charge_icon(p: QPainter, cx: float, cy: float, s: float, col: QColor) -> None:
    p.fillRect(QRectF(cx - s * 0.5, cy - s * 0.18, s, s * 0.5), col)
    p.fillRect(QRectF(cx - s * 0.28, cy - s * 0.36, s * 0.56, s * 0.2), col)
    p.fillRect(QRectF(cx - s * 0.4, cy - s * 0.62, s * 0.08, s * 0.44), col)


# --- the minimap -----------------------------------------------------------------------------------------------------

def minimap_pix(mp, px: int = 6) -> QPixmap:  # noqa: ANN001 - maps.GameMap
    key = ("map", mp.key, px)
    pm = _pix.get(key)
    if pm is not None:
        return pm
    pm = QPixmap(mp.W * px, mp.H * px)
    pm.fill(QColor(16, 15, 26, 235))
    q = QPainter(pm)
    hl, roof = mp.hl, mp.roofl
    for y in range(mp.H):
        for x in range(mp.W):
            h = hl[y][x]
            if h >= 4.9:
                continue
            if h >= 2.0:
                c = QColor(64, 62, 80)
            elif h > 0.4:
                c = QColor(112, 108, 124)
            else:
                c = QColor(196, 192, 200) if not roof[y][x] else QColor(166, 166, 184)
            q.fillRect(x * px, y * px, px, px, c)
    for k, s in mp.sites.items():
        x0, y0, x1, y1 = s.rect
        q.fillRect(QRectF(x0 * px, y0 * px, (x1 - x0) * px, (y1 - y0) * px), QColor(255, 198, 46, 60))
        q.setPen(QPen(QColor(255, 198, 46, 200), 1.5))
        q.drawRect(QRectF(x0 * px, y0 * px, (x1 - x0) * px, (y1 - y0) * px))
        q.setFont(fnt(px * 5.5))
        q.setPen(QColor(40, 30, 10, 200))
        q.drawText(QRectF(x0 * px, y0 * px, (x1 - x0) * px, (y1 - y0) * px), AL.AlignCenter, k)
    q.end()
    _pix[key] = pm
    return pm


def draw_minimap(p: QPainter, r: QRectF, g) -> None:  # noqa: ANN001
    m = g.match
    mp = m.map
    v = g.viewer()
    pm = minimap_pix(mp)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    p.drawPixmap(r, pm, QRectF(pm.rect()))
    kx, ky = r.width() / mp.W, r.height() / mp.H

    def at(x: float, y: float) -> QPointF:
        return QPointF(r.x() + x * kx, r.y() + y * ky)

    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    for s in m.smokes:
        if s[4] >= 0:
            p.setBrush(QColor(230, 230, 240, 150))
            p.drawEllipse(at(s[0], s[1]), s[3] * kx, s[3] * ky)
    for zn in m.zones:
        p.setBrush(rgba(zn["color"], 90))
        p.drawEllipse(at(zn["x"], zn["y"]), zn["r"] * kx, zn["r"] * ky)
    for x, y, age in g.noise_marks:
        a = max(0.0, 1.0 - age / 1.2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(ENEMY.red(), ENEMY.green(), ENEMY.blue(), int(220 * a)), 1.4))
        p.drawEllipse(at(x, y), (2 + age * 5) * kx, (2 + age * 5) * ky)
    p.setPen(Qt.PenStyle.NoPen)
    ch = m.charge
    if ch["state"] in ("dropped", "planted") and (ch["state"] == "planted" or v.side == "atk"):
        on = ch["state"] == "dropped" or int(g.time * 4) % 2 == 0
        charge_icon(p, r.x() + ch["x"] * kx, r.y() + ch["y"] * ky, 9.0, ACCENT if on else QColor(255, 255, 255))
    intel = m.team_ai[v.squad].intel if v.squad in m.team_ai else {}
    jam = v.near_t > 0 and v.near_r < 7.0
    for a in m.actors:
        if not a.alive:
            continue
        mine = a.squad == v.squad
        if not mine:
            it = intel.get(a.idx)
            if jam or not (a.revealed > 0 or (it is not None and it[3])):
                continue
        pt = at(a.x, a.y)
        col = QColor(255, 255, 255) if a is v else (ALLY if mine else ENEMY)
        if a is v or mine:
            yaw = a.yaw
            ca, sa = math.cos(yaw), math.sin(yaw)
            s = 6.5 if a is v else 5.0
            tri = QPolygonF([QPointF(pt.x() + ca * s * 1.5, pt.y() + sa * s * 1.5),
                             QPointF(pt.x() - ca * s - sa * s * 0.8, pt.y() - sa * s + ca * s * 0.8),
                             QPointF(pt.x() - ca * s + sa * s * 0.8, pt.y() - sa * s - ca * s * 0.8)])
            p.setBrush(col)
            p.drawPolygon(tri)
            if a.has_charge:
                p.setBrush(ACCENT)
                p.drawEllipse(pt, 2.4, 2.4)
        else:
            p.setBrush(col)
            p.drawEllipse(pt, 4.2, 4.2)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)


# --- the parts of the HUD --------------------------------------------------------------------------------------------

def draw_top(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    m = g.match
    cx = w / 2
    bw, bh = 250 * s, 46 * s
    panel(p, QRectF(cx - bw / 2, 8 * s, bw, bh), 190)
    text(p, QRectF(cx - bw / 2, 8 * s, 64 * s, bh), str(m.score[0]), 30 * s, ALLY)
    text(p, QRectF(cx + bw / 2 - 64 * s, 8 * s, 64 * s, bh), str(m.score[1]), 30 * s, ENEMY)
    if m.phase == "planted":
        left = max(m.charge.get("t", 0.0), 0.0)
        on = int(g.time * (2 if left > 10 else 5)) % 2 == 0
        charge_icon(p, cx, 30 * s, 22 * s, ACCENT if on else PAPER)
        fr = left / C.FUSE_TIME
        p.fillRect(QRectF(cx - 50 * s, 46 * s, 100 * s, 4 * s), QColor(0, 0, 0, 160))
        p.fillRect(QRectF(cx - 50 * s, 46 * s, 100 * s * fr, 4 * s), ACCENT if left > 10 else ENEMY)
    else:
        t = max(m.t, 0.0)
        label = f"{int(t) // 60}:{int(t) % 60:02d}"
        col = PAPER
        if m.phase == "buy":
            col = ACCENT
        elif m.phase == "live" and t < 10:
            col = ENEMY
        elif m.phase in ("end", "over"):
            label = "- -"
        text(p, QRectF(cx - 60 * s, 8 * s, 120 * s, bh), label, 26 * s, col)
    side = "ATTACK" if g.viewer().side == "atk" else "DEFENCE"
    sub = "BUY PHASE" if m.phase == "buy" else f"ROUND {min(m.round + 1, 99)} · {side}"
    text(p, QRectF(cx - 120 * s, 56 * s, 240 * s, 16 * s), sub, 11 * s, DIM, shadow=True)
    cs = 24 * s
    for sq, sign in ((0, -1), (1, 1)):
        for i, a in enumerate(a for a in m.actors if a.squad == sq):
            x = cx + sign * (bw / 2 + 8 * s + i * (cs + 4 * s)) - (cs if sign < 0 else 0)
            r = QRectF(x, 12 * s, cs, cs)
            col = rgba(a.agent.color)
            if a.alive:
                p.fillRect(r, col)
                text(p, r, a.agent.name[0], 14 * s, INK)
                p.fillRect(QRectF(x, 12 * s + cs, cs, 3 * s), ALLY if sq == 0 else ENEMY)
                if a.has_charge and sq == g.viewer().squad:
                    p.fillRect(QRectF(x + cs - 7 * s, 12 * s, 7 * s, 7 * s), ACCENT)
            else:
                p.fillRect(r, QColor(30, 28, 44, 200))
                text(p, r, "×", 16 * s, QColor(120, 120, 135))


def draw_feed(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    m = g.match
    y = 14 * s
    for e in reversed(m.feed):
        age = m.time - e["t"]
        if age > 7.0:
            continue
        a = int(255 * min(1.0, 7.0 - age))
        kn, vn = e["k"] or "—", e["v"]
        how = e["how"]
        wname = WEAPONS[how].name if how in WEAPONS else how.upper()
        if e["head"]:
            wname += "  ◎"
        if e["wall"]:
            wname += "  ▮"
        rw = (len(kn) + len(vn)) * 8.5 * s + len(wname) * 6.5 * s + 44 * s
        r = QRectF(w - rw - 14 * s, y, rw, 24 * s)
        p.fillRect(r, QColor(INK.red(), INK.green(), INK.blue(), int(a * 0.72)))
        mine = e["ks"] == g.viewer().squad
        p.fillRect(QRectF(r.x(), r.y(), 3 * s, r.height()), QColor(*(C.ALLY if mine else C.ENEMY), a))
        kc = QColor(*(C.ALLY if e["ks"] == 0 else C.ENEMY), a)
        vc = QColor(*(C.ALLY if e["vs"] == 0 else C.ENEMY), a)
        text(p, QRectF(r.x() + 10 * s, y, rw, 24 * s), kn, 13 * s, kc, AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(r.x(), y, rw - 10 * s, 24 * s), vn, 13 * s, vc, AL.AlignRight | AL.AlignVCenter)
        text(p, QRectF(r.x() + (len(kn) * 8.5 + 20) * s, y, len(wname) * 6.5 * s + 10 * s, 24 * s), wname, 10 * s,
             QColor(230, 228, 220, a), AL.AlignCenter, False)
        y += 28 * s


def draw_vitals(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    v = g.viewer()
    x, y = 22 * s, h - 78 * s
    panel(p, QRectF(x, y, 250 * s, 58 * s), 170, ALLY)
    hp = max(int(math.ceil(v.hp)), 0)
    col = PAPER if hp > 35 else ENEMY
    text(p, QRectF(x + 12 * s, y, 86 * s, 58 * s), str(hp), 40 * s, col, AL.AlignLeft | AL.AlignVCenter)
    p.fillRect(QRectF(x + 96 * s, y + 16 * s, 140 * s, 8 * s), QColor(0, 0, 0, 150))
    p.fillRect(QRectF(x + 96 * s, y + 16 * s, 140 * s * min(hp / 100, 1), 8 * s), col)
    p.fillRect(QRectF(x + 96 * s, y + 32 * s, 140 * s, 6 * s), QColor(0, 0, 0, 150))
    if v.armour_max > 0:
        p.fillRect(QRectF(x + 96 * s, y + 32 * s, 140 * s * min(v.armour / 50, 1), 6 * s), ALLY)
    text(p, QRectF(x + 96 * s, y + 40 * s, 140 * s, 16 * s), f"ARMOUR {int(v.armour)}", 10 * s, DIM,
         AL.AlignLeft | AL.AlignVCenter)
    tags = []
    if v.blind > 0.2:
        tags.append("BLINDED")
    if v.daze > 0:
        tags.append("DAZED")
    if v.near_t > 0:
        tags.append("NEARSIGHTED")
    if v.slow < 1.0:
        tags.append("SLOWED")
    if v.revealed > 0:
        tags.append("MARKED")
    if v.has_charge:
        tags.append("CARRYING THE CHARGE")
    for i, tg in enumerate(tags):
        r = QRectF(x, y - (22 + i * 22) * s, (len(tg) * 7.6 + 16) * s, 18 * s)
        p.fillRect(r, ACCENT if tg.startswith("CARRY") else QColor(ENEMY.red(), ENEMY.green(), ENEMY.blue(), 220))
        text(p, r, tg, 10.5 * s, INK)


def draw_abilities(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    v = g.viewer()
    bs = 54 * s
    gap = 8 * s
    x0 = w / 2 - (bs * 4 + gap * 3) / 2
    y = h - bs - 22 * s
    col = v.agent.color
    for i, slot in enumerate(C.AGENTS[v.agent.id].abilities):
        sl = slot.slot
        r = QRectF(x0 + i * (bs + gap), y, bs, bs)
        ready = v.can_use(sl)
        panel(p, r, 190 if ready else 130)
        if sl == "X" and ready:
            p.fillRect(r, QColor(col[0], col[1], col[2], 70 + int(40 * math.sin(g.time * 5))))
        tint = col if ready else (110, 110, 122)
        isz = int(bs * 0.7)
        p.drawPixmap(QPointF(r.x() + (bs - isz) / 2, r.y() + 3 * s), ability_pix(slot.kind, isz, tint))
        text(p, QRectF(r.x() + 4 * s, r.y() + 2 * s, 20 * s, 14 * s), g.key_label(sl), 10 * s,
             PAPER if ready else DIM, AL.AlignLeft | AL.AlignVCenter)
        if sl == "X":
            n = slot.points
            pw = (bs - 8 * s) / n
            for k in range(n):
                p.fillRect(QRectF(r.x() + 4 * s + k * pw, r.bottom() - 8 * s, pw - 2 * s, 5 * s),
                           rgba(col) if k < v.ult else QColor(70, 70, 84))
        else:
            for k in range(slot.charges):
                p.fillRect(QRectF(r.x() + 4 * s + k * 12 * s, r.bottom() - 8 * s, 9 * s, 5 * s),
                           rgba(col) if k < v.charges[sl] else QColor(70, 70, 84))
            if sl == "E" and v.charges["E"] < slot.charges and slot.recharge:
                fr = v.recharge / slot.recharge
                p.fillRect(QRectF(r.x(), r.bottom() - 2 * s, bs * fr, 2 * s), PAPER)


def draw_ammo(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    v = g.viewer()
    wp = v.weapon
    x, y = w - 272 * s, h - 78 * s
    panel(p, QRectF(x, y, 250 * s, 58 * s), 170)
    p.fillRect(QRectF(x + 247 * s, y, 3 * s, 58 * s), ACCENT)
    if wp.cls == "melee":
        text(p, QRectF(x + 10 * s, y, 150 * s, 58 * s), "—", 34 * s, PAPER, AL.AlignLeft | AL.AlignVCenter)
    else:
        mag = v.mag()
        col = PAPER if mag[0] > wp.mag * 0.25 else ENEMY
        text(p, QRectF(x + 12 * s, y, 80 * s, 58 * s), str(mag[0]), 40 * s, col, AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(x + 84 * s, y + 20 * s, 60 * s, 30 * s), f"/ {mag[1]}", 16 * s, DIM,
             AL.AlignLeft | AL.AlignVCenter)
    p.drawPixmap(QPointF(x + 138 * s, y + 6 * s), weapon_pix(v.wid, int(100 * s), int(30 * s), (236, 232, 222)))
    name = wp.name + ("  · RELOADING" if v.reload_t > 0 else "")
    text(p, QRectF(x + 100 * s, y + 38 * s, 140 * s, 16 * s), name, 10.5 * s, DIM, AL.AlignRight | AL.AlignVCenter)
    # the three slots and the purse
    yy = y - 22 * s
    slots = (("1", "main", WEAPONS[v.main].name if v.main else "—"), ("2", "side", WEAPONS[v.sidearm].name),
             ("3", "knife", "KNIFE"))
    xx = x + 250 * s
    for key, sl, name in reversed(slots):
        tw = (len(name) * 6.6 + 24) * s
        xx -= tw + 4 * s
        cur = v.cur == sl or (sl == "side" and v.cur == "main" and not v.main)
        r = QRectF(xx, yy, tw, 18 * s)
        p.fillRect(r, QColor(ACCENT.red(), ACCENT.green(), ACCENT.blue(), 230) if cur else QColor(20, 18, 31, 150))
        text(p, r, f"{key}  {name}", 10 * s, INK if cur else DIM)
    text(p, QRectF(x, y - 46 * s, 250 * s, 20 * s), f"¤ {v.credits:,}", 15 * s, ACCENT, AL.AlignRight | AL.AlignVCenter,
         shadow=True)


def draw_crosshair(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    v = g.viewer()
    st = g.settings
    if v.scoped() or not v.alive:
        return
    col = QColor(*CROSS_COLOURS[st["cross_colour"] % len(CROSS_COLOURS)][1])
    cx, cy = w / 2, h / 2
    ln = (4 + st["cross_size"] * 2) * s
    th = max(2.0 * s, 1.5) if st["cross_size"] < 4 else 3.0 * s
    spread = math.tan(math.radians(v.spread())) * g.renderer.proj if st["cross_dynamic"] else 0.0
    gap = (2 + st["cross_gap"] * 1.5) * s + spread
    if v.weapon.scoped:
        gap = max(gap, 14 * s)
    out = QColor(0, 0, 0, 150)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if dx:
            r = QRectF(cx + dx * gap if dx > 0 else cx - gap - ln, cy - th / 2, ln, th)
        else:
            r = QRectF(cx - th / 2, cy + dy * gap if dy > 0 else cy - gap - ln, th, ln)
        p.fillRect(r.adjusted(-1, -1, 1, 1), out)
        p.fillRect(r, col)
    if st["cross_dot"]:
        p.fillRect(QRectF(cx - th / 2 - 1, cy - th / 2 - 1, th + 2, th + 2), out)
        p.fillRect(QRectF(cx - th / 2, cy - th / 2, th, th), col)


def draw_marks(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    """Hit marks, where damage came from, and what blinds or dims the view."""
    v = g.viewer()
    cx, cy = w / 2, h / 2
    if g.hit_t > 0:
        k = g.hit_t / 0.22
        col = QColor(255, 80, 80, int(255 * k)) if g.hit_kill else (
            QColor(255, 220, 90, int(255 * k)) if g.hit_head else QColor(255, 255, 255, int(255 * k)))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(col, 2.4 * s))
        d0, d1 = (7 + 3 * (1 - k)) * s, (15 + 3 * (1 - k)) * s
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            p.drawLine(QPointF(cx + sx * d0, cy + sy * d0), QPointF(cx + sx * d1, cy + sy * d1))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    if v.hit_dirs:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for ang, left in v.hit_dirs:
            rel = ang - v.yaw
            a = int(230 * min(left / 0.5, 1.0))
            p.setPen(QPen(QColor(255, 60, 90, a), 6 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            rr = 92 * s
            deg = -math.degrees(rel) + 90  # screen: up is ahead
            p.drawArc(QRectF(cx - rr, cy - rr, rr * 2, rr * 2), int((deg - 16) * 16), int(32 * 16))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    if g.hurt_t > 0:
        a = int(90 * min(g.hurt_t / 0.4, 1.0))
        e = 60 * s
        for r in (QRectF(0, 0, w, e), QRectF(0, h - e, w, e), QRectF(0, 0, e, h), QRectF(w - e, 0, e, h)):
            p.fillRect(r, QColor(255, 30, 70, a))
    if g.in_smoke > 0.02:
        p.fillRect(QRectF(0, 0, w, h), QColor(200, 204, 212, int(238 * min(g.in_smoke, 1.0))))
    if v.near_t > 0:
        p.fillRect(QRectF(0, 0, w, h), QColor(24, 18, 44, int(150 * min(v.near_t / 0.4, 1.0))))
    if v.daze > 0:
        p.fillRect(QRectF(0, 0, w, h), QColor(255, 230, 150, int(60 * min(v.daze / 0.5, 1.0))))
    if v.blind > 0:
        p.fillRect(QRectF(0, 0, w, h), QColor(255, 255, 250, int(255 * min(v.blind / 0.7, 1.0))))


def draw_scope(p: QPainter, w: int, h: int, s: float) -> None:
    r = h * 0.44
    cx, cy = w / 2, h / 2
    path = QPainterPath()
    path.addRect(QRectF(0, 0, w, h))
    path.addEllipse(QPointF(cx, cy), r, r)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.fillPath(path, QColor(6, 6, 10))
    p.setPen(QPen(QColor(10, 10, 14), 1.6 * s))
    p.drawLine(QPointF(cx - r, cy), QPointF(cx - 12 * s, cy))
    p.drawLine(QPointF(cx + 12 * s, cy), QPointF(cx + r, cy))
    p.drawLine(QPointF(cx, cy - r), QPointF(cx, cy - 12 * s))
    p.drawLine(QPointF(cx, cy + 12 * s), QPointF(cx, cy + r))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 60, 60))
    p.drawEllipse(QPointF(cx, cy), 1.6 * s, 1.6 * s)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)


def draw_world_tags(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    """Teammates' names over their heads, and where the charge is."""
    m = g.match
    v = g.viewer()
    r = g.renderer
    for a in m.actors:
        if a.squad != v.squad or a is v or not a.alive:
            continue
        pt = r.project(a.x, a.y, a.top + 0.28)
        if pt is None or not (0 < pt[0] < w and 0 < pt[1] < h):
            continue
        sx, sy, d = pt
        k = max(0.55, min(1.0, 12.0 / max(d, 1.0)))
        tri = QPolygonF([QPointF(sx - 5 * s * k, sy - 7 * s * k), QPointF(sx + 5 * s * k, sy - 7 * s * k),
                         QPointF(sx, sy)])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(ACCENT if a.has_charge else ALLY)
        p.drawPolygon(tri)
        if d < 26:
            text(p, QRectF(sx - 60, sy - (24 * k + 6) * s, 120, 16 * s), a.name, 11 * s * k + 1, ALLY, shadow=True)
    ch = m.charge
    if ch["state"] in ("dropped", "planted") and (ch["state"] == "planted" or v.side == "atk"):
        pt = r.project(ch["x"], ch["y"], ch.get("z", 0.0) + 1.0)
        if pt is not None:
            sx = min(max(pt[0], 40 * s), w - 40 * s)
            sy = min(max(pt[1], 110 * s), h - 150 * s)
            charge_icon(p, sx, sy, 16 * s, ACCENT)
            text(p, QRectF(sx - 40, sy + 8 * s, 80, 14 * s), f"{int(pt[2])} m", 10 * s, ACCENT, shadow=True)
    if m.phase == "buy" and v.alive:
        for k, site in m.map.sites.items():
            cx, cy = site.centre
            pt = r.project(cx, cy, 2.6)
            if pt is not None and 0 < pt[0] < w and 60 * s < pt[1] < h and pt[2] > 9:
                text(p, QRectF(pt[0] - 30, pt[1] - 14 * s, 60, 28 * s), k, 20 * s, QColor(255, 198, 46, 190),
                     shadow=True)


def draw_prompts(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    m = g.match
    v = g.viewer()
    cx = w / 2
    y = h * 0.66

    def bar(label: str, fr: float, col: QColor, mark: float | None = None) -> None:
        bw = 260 * s
        text(p, QRectF(cx - bw / 2, y - 22 * s, bw, 18 * s), label, 12 * s, PAPER, shadow=True)
        p.fillRect(QRectF(cx - bw / 2 - 2, y - 2, bw + 4, 10 * s + 4), QColor(0, 0, 0, 170))
        p.fillRect(QRectF(cx - bw / 2, y, bw * min(fr, 1.0), 10 * s), col)
        if mark is not None:
            p.fillRect(QRectF(cx - bw / 2 + bw * mark - 1, y - 3, 2, 10 * s + 6), PAPER)

    ch = m.charge
    if v.plant_t > 0:
        bar("PLANTING", v.plant_t / C.PLANT_TIME, ACCENT)
        return
    if ch["state"] == "planted" and ch.get("defuser") is v:
        bar("DEFUSING", ch["defuse"] / C.DEFUSE_TIME, ALLY, C.DEFUSE_HALF / C.DEFUSE_TIME)
        return
    if ch["state"] == "planted" and ch.get("defuser") is not None and v.side == "atk":
        text(p, QRectF(cx - 200 * s, y - 20 * s, 400 * s, 20 * s), "THE CHARGE IS BEING DEFUSED", 14 * s, ENEMY,
             shadow=True)
    if not v.alive or v is not m.player:
        return
    msg = None
    use = g.key_label("use")
    if v.has_charge and m.phase == "live" and m.map.site_at(v.x, v.y):
        msg = f"HOLD [{use}] TO PLANT THE CHARGE"
    elif v.side == "def" and ch["state"] == "planted" and math.hypot(ch["x"] - v.x, ch["y"] - v.y) < 1.7:
        half = "  ·  HALF DONE" if ch["defuse"] >= C.DEFUSE_HALF else ""
        msg = f"HOLD [{use}] TO DEFUSE{half}"
    else:
        for d in m.drops:
            if math.hypot(d[0] - v.x, d[1] - v.y) < 1.5:
                msg = f"[{use}] PICK UP {WEAPONS[d[2]].name}"
                break
    if msg:
        text(p, QRectF(cx - 240 * s, y - 20 * s, 480 * s, 20 * s), msg, 13 * s, ACCENT, shadow=True)


def draw_banner(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    b = g.match.banner
    if b is None:
        return
    txt, sub, colour, left = b
    a = int(255 * min(left / 0.4, 1.0))
    col = {"accent": C.ACCENT, "ally": C.ALLY, "enemy": C.ENEMY}.get(colour, C.ACCENT)
    y = 96 * s
    if txt:
        bw = max(len(txt) * 24 * s + 80 * s, 300 * s)
        p.fillRect(QRectF(w / 2 - bw / 2, y, bw, 52 * s), QColor(INK.red(), INK.green(), INK.blue(), int(a * 0.8)))
        p.fillRect(QRectF(w / 2 - bw / 2, y + 50 * s, bw, 3 * s), QColor(*col, a))
        text(p, QRectF(w / 2 - bw / 2, y, bw, 52 * s), txt, 32 * s, QColor(*col, a))
        y += 58 * s
    if sub:
        text(p, QRectF(0, y, w, 22 * s), sub, 14 * s, QColor(240, 236, 226, a), shadow=True)


def draw_info(p: QPainter, w: int, h: int, g, s: float, map_rect: QRectF) -> None:  # noqa: ANN001
    """Under the minimap: where you are, what the team is doing, and the buy-phase reminder."""
    m = g.match
    v = g.viewer()
    y = map_rect.bottom() + 6 * s
    place = m.map.place(v.x, v.y)
    site = m.map.site_at(v.x, v.y)
    label = (f"SITE {site} · " if site else "") + place.upper()
    text(p, QRectF(map_rect.x(), y, 260 * s, 16 * s), label, 11 * s, PAPER, AL.AlignLeft | AL.AlignVCenter, shadow=True)
    ai = m.team_ai.get(v.squad)
    if ai is not None and m.phase in ("buy", "live"):
        if v.side == "atk":
            way = m.map.sites[ai.site].attack[ai.ways.get(v.idx, 0) % len(m.map.sites[ai.site].attack)][0]
            plan = f"TEAM PLAN · SITE {ai.site} VIA {way.upper()}"
        else:
            plan = "TEAM PLAN · HOLD BOTH SITES"
        text(p, QRectF(map_rect.x(), y + 16 * s, 320 * s, 16 * s), plan, 10.5 * s, ACCENT,
             AL.AlignLeft | AL.AlignVCenter, shadow=True)
        text(p, QRectF(map_rect.x(), y + 31 * s, 320 * s, 14 * s),
             f"[{g.key_label('call_a')}] CALL A   [{g.key_label('call_b')}] CALL B", 9.5 * s, DIM,
             AL.AlignLeft | AL.AlignVCenter, False, True)
    if m.phase == "buy" and v is m.player and g.overlay is None:
        text(p, QRectF(0, h * 0.74, w, 22 * s),
             f"[{g.key_label('buy')}] BUY      [{g.key_label('ready')}] READY", 15 * s, ACCENT, shadow=True)


def draw_spectate(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    v = g.viewer()
    r = QRectF(w / 2 - 190 * s, h - 150 * s, 380 * s, 30 * s)
    panel(p, r, 200, rgba(v.agent.color))
    text(p, r, f"SPECTATING  {v.name}  ·  {v.agent.name}      CLICK: NEXT", 12 * s, PAPER)


def draw_scoreboard(p: QPainter, w: int, h: int, g, s: float, final: bool = False) -> None:  # noqa: ANN001
    m = g.match
    pw, ph = 760 * s, 420 * s
    x, y = (w - pw) / 2, (h - ph) / 2 + (30 * s if final else 0)
    panel(p, QRectF(x, y, pw, ph), 232)
    p.fillRect(QRectF(x, y, pw, 3 * s), ACCENT)
    text(p, QRectF(x + 20 * s, y + 8 * s, pw, 30 * s), f"{m.map.name}   ·   {m.score[0]} - {m.score[1]}", 18 * s,
         PAPER, AL.AlignLeft | AL.AlignVCenter)
    cols = (("", 0.0), ("AGENT", 0.24), ("K", 0.44), ("D", 0.52), ("A", 0.6), ("HS%", 0.68), ("DMG", 0.78),
            ("ULT", 0.9))
    yy = y + 44 * s
    for sq in (0, 1):
        col = ALLY if sq == 0 else ENEMY
        side = "ATTACK" if m.sides[sq] == "atk" else "DEFENCE"
        p.fillRect(QRectF(x + 14 * s, yy, pw - 28 * s, 22 * s), QColor(col.red(), col.green(), col.blue(), 60))
        text(p, QRectF(x + 22 * s, yy, 200 * s, 22 * s), ("YOUR TEAM" if sq == 0 else "ENEMY TEAM") + " · " + side,
             11 * s, col, AL.AlignLeft | AL.AlignVCenter)
        for label, fx in cols[2:]:
            text(p, QRectF(x + pw * fx, yy, 60 * s, 22 * s), label, 10 * s, DIM)
        yy += 26 * s
        team = sorted((a for a in m.actors if a.squad == sq), key=lambda a: -a.kills)
        for a in team:
            if a is m.player:
                p.fillRect(QRectF(x + 14 * s, yy, pw - 28 * s, 26 * s), QColor(255, 255, 255, 22))
            p.fillRect(QRectF(x + 18 * s, yy + 4 * s, 18 * s, 18 * s), rgba(a.agent.color))
            text(p, QRectF(x + 18 * s, yy + 4 * s, 18 * s, 18 * s), a.agent.name[0], 11 * s, INK)
            nc = PAPER if a.alive or final else QColor(120, 120, 135)
            text(p, QRectF(x + 44 * s, yy, 150 * s, 26 * s), a.name, 13 * s, nc, AL.AlignLeft | AL.AlignVCenter)
            text(p, QRectF(x + pw * 0.24, yy, 120 * s, 26 * s), f"{a.agent.name} · {a.agent.role}", 10.5 * s, DIM,
                 AL.AlignLeft | AL.AlignVCenter, False)
            hs = f"{int(100 * a.stats['heads'] / a.kills)}" if a.kills else "-"
            ult = f"{a.ult}/{a.agent.ability('X').points}" if sq == 0 else "·"
            for val, fx in ((a.kills, 0.44), (a.deaths, 0.52), (a.assists, 0.6), (hs, 0.68), (a.stats["damage"], 0.78),
                            (ult, 0.9)):
                text(p, QRectF(x + pw * fx, yy, 60 * s, 26 * s), str(val), 13 * s, nc)
            yy += 28 * s
        yy += 8 * s


def draw_match_end(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    m = g.match
    p.fillRect(QRectF(0, 0, w, h), QColor(10, 9, 18, 150))
    won = m.match_winner == 0
    text(p, QRectF(0, 40 * s, w, 80 * s), "VICTORY" if won else "DEFEAT", 68 * s, ALLY if won else ENEMY, shadow=True)
    draw_scoreboard(p, w, h, g, s, final=True)
    if g.screen_t > 1.5:
        text(p, QRectF(0, h - 60 * s, w, 24 * s), "ENTER: BACK TO THE MENU", 14 * s, ACCENT, shadow=True)


# --- the buy menu ----------------------------------------------------------------------------------------------------

def draw_buy(p: QPainter, w: int, h: int, g, s: float) -> None:  # noqa: ANN001
    """The buy menu. Registers its cards with ``g.ui`` (left click buys, right click gives back)."""
    m = g.match
    v = m.player
    ui = g.ui
    p.fillRect(QRectF(0, 0, w, h), QColor(10, 9, 18, 205))
    pw, ph = min(1120 * s, w - 40), 600 * s
    x, y = (w - pw) / 2, (h - ph) / 2 + 8 * s
    text(p, QRectF(x, y - 44 * s, 400 * s, 40 * s), "ARMOURY", 30 * s, PAPER, AL.AlignLeft | AL.AlignVCenter)
    text(p, QRectF(x + pw - 400 * s, y - 44 * s, 400 * s, 40 * s), f"¤ {v.credits:,}", 28 * s, ACCENT,
         AL.AlignRight | AL.AlignVCenter)
    text(p, QRectF(x, y - 44 * s, pw, 40 * s), f"BUY PHASE  {max(int(m.t), 0)} s", 14 * s, DIM)
    colw = pw / 5
    hover_tip = None
    bought = {it for it, _pr in v.bought}

    def card(r: QRectF, item: str, title: str, sub: str, price: int, owned: bool, icon) -> None:  # noqa: ANN001
        nonlocal hover_tip
        why = m.can_buy(v, item)
        hov = ui.button(r, ("buy", item))
        can = not why
        bg = QColor(44, 40, 66, 235) if hov else QColor(28, 26, 44, 225)
        p.fillRect(r, bg)
        edge = ACCENT if owned else (PAPER if can else QColor(70, 70, 86))
        p.fillRect(QRectF(r.x(), r.y(), 3 * s, r.height()), edge)
        if icon is not None:
            icon(QRectF(r.x() + 10 * s, r.y() + 6 * s, r.width() - 20 * s, r.height() * 0.42))
        text(p, QRectF(r.x() + 12 * s, r.bottom() - 40 * s, r.width(), 18 * s), title, 13 * s,
             PAPER if can or owned else DIM, AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(r.x() + 12 * s, r.bottom() - 22 * s, r.width() - 20 * s, 16 * s), sub, 9.5 * s, DIM,
             AL.AlignLeft | AL.AlignVCenter, False)
        ptxt = "OWNED" if owned and item not in bought else ("BOUGHT · RMB RETURNS" if item in bought else
                                                              ("FREE" if price == 0 else f"¤ {price:,}"))
        text(p, QRectF(r.x(), r.bottom() - 40 * s, r.width() - 10 * s, 18 * s), ptxt, 11 * s,
             ACCENT if owned else (PAPER if can else ENEMY), AL.AlignRight | AL.AlignVCenter)
        if hov:
            hover_tip = item

    for ci, (title, ids) in enumerate(C.BUY_COLUMNS):
        cx = x + ci * colw
        text(p, QRectF(cx, y, colw, 22 * s), title, 12 * s, ACCENT, AL.AlignLeft | AL.AlignVCenter)
        chh = min(96 * s, (ph - 30 * s) / 5 - 8 * s)
        for i, wid in enumerate(ids):
            wp = WEAPONS[wid]
            r = QRectF(cx, y + 28 * s + i * (chh + 8 * s), colw - 12 * s, chh)
            owned = (v.sidearm if wp.group == "side" else v.main) == wid

            def icon(ir: QRectF, wid: str = wid, owned: bool = owned) -> None:
                pm = weapon_pix(wid, int(ir.width()), int(ir.height()), (236, 232, 222) if owned or not
                                m.can_buy(v, wid) else (200, 198, 205))
                p.drawPixmap(ir.topLeft(), pm)

            head = int(wp.dmg * wp.head)
            sub = f"{int(wp.dmg)} body · {head} head · {wp.pen.upper()} pen"
            if wp.pellets > 1:
                sub = f"{wp.pellets} × {int(wp.dmg)} · {wp.pen.upper()} pen"
            card(r, wid, wp.name, sub, wp.price, owned, icon)
    cx = x + 3 * colw
    text(p, QRectF(cx, y, colw, 22 * s), "ARMOUR", 12 * s, ACCENT, AL.AlignLeft | AL.AlignVCenter)
    for i, (key, (price, pool, name)) in enumerate(C.ARMOUR.items()):
        r = QRectF(cx, y + 28 * s + i * 104 * s, colw - 12 * s, 96 * s)

        def icon(ir: QRectF, pool: int = pool) -> None:
            q = ability_pix("stronghold", int(ir.height()), (63, 184, 255))
            p.drawPixmap(QPointF(ir.x() + 6 * s, ir.y()), q)
            text(p, QRectF(ir.x() + ir.height() + 12 * s, ir.y(), 80 * s, ir.height()), f"+{pool}", 22 * s, ALLY,
                 AL.AlignLeft | AL.AlignVCenter)

        card(r, key, name, f"soaks 66% until {pool} is gone", price, v.armour >= pool and v.armour_max == pool, icon)
    cx = x + 4 * colw
    text(p, QRectF(cx, y, colw, 22 * s), f"{v.agent.name}'S KIT", 12 * s, rgba(v.agent.color),
         AL.AlignLeft | AL.AlignVCenter)
    for i, slot in enumerate(("C", "Q")):
        ab = v.agent.ability(slot)
        r = QRectF(cx, y + 28 * s + i * 104 * s, colw - 12 * s, 96 * s)

        def icon(ir: QRectF, ab=ab) -> None:  # noqa: ANN001
            p.drawPixmap(QPointF(ir.x() + 6 * s, ir.y()), ability_pix(ab.kind, int(ir.height()), v.agent.color))
            for k in range(ab.charges):
                p.fillRect(QRectF(ir.x() + ir.height() + 16 * s + k * 16 * s, ir.y() + ir.height() / 2 - 4 * s,
                                  12 * s, 8 * s), rgba(v.agent.color) if k < v.charges[ab.slot] else QColor(70, 70, 84))

        card(r, slot, ab.name, f"[{g.key_label(slot)}] · {v.charges[slot]}/{ab.charges} held", ab.cost,
             v.charges[slot] >= ab.charges, icon)
    for i, slot in enumerate(("E", "X")):
        ab = v.agent.ability(slot)
        r = QRectF(cx, y + 28 * s + (2 + i) * 104 * s, colw - 12 * s, 60 * s)
        p.fillRect(r, QColor(24, 22, 38, 200))
        p.drawPixmap(QPointF(r.x() + 8 * s, r.y() + 8 * s), ability_pix(ab.kind, int(44 * s), v.agent.color))
        text(p, QRectF(r.x() + 60 * s, r.y() + 8 * s, r.width(), 20 * s), ab.name, 12 * s, PAPER,
             AL.AlignLeft | AL.AlignVCenter)
        note = f"free · back every {int(ab.recharge)} s" if slot == "E" else f"ultimate · {v.ult}/{ab.points} points"
        text(p, QRectF(r.x() + 60 * s, r.y() + 30 * s, r.width(), 18 * s), note, 9.5 * s, DIM,
             AL.AlignLeft | AL.AlignVCenter, False)
        if ui.hover(r):
            hover_tip = slot
    # what the pointer is on
    tip = ""
    if hover_tip in WEAPONS:
        wp = WEAPONS[hover_tip]
        tip = (f"{wp.name}  ·  {wp.blurb}   {wp.rps:g} shots/s · {wp.mag} rounds · reload {wp.reload:g} s · "
               f"moves at {int(wp.speed * 100)}%")
    elif hover_tip in C.ARMOUR:
        tip = "Armour soaks two thirds of each hit until it is used up. It carries over while you live."
    elif hover_tip:
        ab = v.agent.ability(hover_tip)
        tip = f"{ab.name}  ·  {ab.desc}"
    ty = y + ph - 46 * s
    p.fillRect(QRectF(x, ty, pw, 34 * s), QColor(24, 22, 38, 220))
    text(p, QRectF(x + 12 * s, ty, pw - 24 * s, 34 * s),
         tip or "LEFT CLICK: BUY     RIGHT CLICK: RETURN WHAT YOU BOUGHT THIS ROUND", 11.5 * s, PAPER if tip else DIM,
         AL.AlignLeft | AL.AlignVCenter, False)
    text(p, QRectF(x, ty + 38 * s, pw, 20 * s), f"[{g.key_label('buy')}] / ESC: CLOSE       "
         f"[{g.key_label('ready')}] READY", 11 * s, ACCENT)


def warm(p: QPainter, h: int, agent) -> None:  # noqa: ANN001
    """Draw once, behind the loading card, what would otherwise be made in the middle of a fight: every size of
    lettering with the odd signs of the kill feed, and the gun and ability pictures."""
    s = h / 780.0
    r = QRectF(0, 0, 600, 80)
    for size in (9.5, 10, 10.5, 11, 12, 13, 14, 15, 16, 18, 20, 22, 26, 30, 32, 40, 68):
        for bold in (True, False):
            text(p, r, "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789 ◎ ▮ ¤ × — · [ ] / : - , . ' %",
                 size * s, PAPER, AL.AlignLeft, bold, True)
    for wid in WEAPONS:
        weapon_pix(wid, int(100 * s), int(30 * s), (236, 232, 222))
    for ab in agent.abilities:
        for tint in (agent.color, (110, 110, 122)):
            ability_pix(ab.kind, int(54 * s * 0.7), tint)


# --- everything ------------------------------------------------------------------------------------------------------

def draw_hud(p: QPainter, w: int, h: int, g) -> None:  # noqa: ANN001
    s = h / 780.0
    m = g.match
    v = g.viewer()
    draw_marks(p, w, h, g, s)
    if v.alive and v.scoped():
        draw_scope(p, w, h, s)
    draw_world_tags(p, w, h, g, s)
    draw_crosshair(p, w, h, g, s)
    mw = 228 * s
    map_rect = QRectF(14 * s, 14 * s, mw, mw * m.map.H / m.map.W)
    draw_minimap(p, map_rect, g)
    draw_info(p, w, h, g, s, map_rect)
    draw_top(p, w, h, g, s)
    draw_feed(p, w, h, g, s)
    if v.alive:
        draw_vitals(p, w, h, g, s)
        draw_abilities(p, w, h, g, s)
        draw_ammo(p, w, h, g, s)
    draw_prompts(p, w, h, g, s)
    if v is not m.player and m.phase != "over":
        draw_spectate(p, w, h, g, s)
    if m.phase != "over":
        draw_banner(p, w, h, g, s)
    if g.kill_note is not None:
        txt, left = g.kill_note
        text(p, QRectF(0, h * 0.58, w, 24 * s), txt, 15 * s, QColor(255, 255, 255, int(255 * min(left / 0.4, 1.0))),
             shadow=True)
