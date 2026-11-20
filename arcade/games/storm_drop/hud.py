"""Storm Drop's heads-up display, drawn with QPainter over the 3D picture at the window's own resolution: health and
shield, materials and the build keys, the inventory bar, the minimap with the storm's eye, players left, the kill feed,
the compass, the crosshair with its bloom, prompts, damage numbers and the big map."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient

from app.arcade.base import font

from .data import (AMMO_COLORS, BUILD_COST, GRADES, MAT_COLORS, MAT_NAMES, MAX_HP, MAX_SHIELD, PIECE_NAMES, SIZE,
                   SLOTS, SUPPLIES, WEAPONS)

A = Qt.AlignmentFlag
INK = QColor(18, 22, 38)
PANEL = QColor(16, 20, 40, 170)
FAMILY = "Segoe UI"
HEAVY = "Segoe UI Black"


def text(p: QPainter, r: QRectF, s: str, size: float, color="#ffffff", align=A.AlignCenter, bold: bool = True,  # noqa: ANN001
         heavy: bool = False, shadow: bool = True, alpha: float = 1.0) -> None:
    p.setFont(font(size, bold, HEAVY if heavy else FAMILY))
    if shadow:
        p.setPen(QColor(8, 10, 24, int(200 * alpha)))
        d = max(size * 0.07, 1.0)
        p.drawText(r.translated(d, d), int(align), s)
    c = QColor(color)
    c.setAlphaF(c.alphaF() * alpha)
    p.setPen(c)
    p.drawText(r, int(align), s)


def panel(p: QPainter, r: QRectF, radius: float, fill: QColor = PANEL, edge: QColor | None = None,
          width: float = 2.0) -> None:
    p.setPen(QPen(edge, width) if edge is not None else Qt.PenStyle.NoPen)
    p.setBrush(fill)
    p.drawRoundedRect(r, radius, radius)


def icon(p: QPainter, r: QRectF, id_: str | None, col: QColor | None = None) -> None:
    """A weapon or supply drawn in the box ``r`` (the sledge for None)."""
    p.save()
    p.translate(r.center())
    s = min(r.width(), r.height()) / 100.0
    p.scale(s, s)
    p.setPen(QPen(INK, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    body = col or QColor("#dfe6f0")

    def poly(pts: list, c: QColor) -> None:
        p.setBrush(c)
        p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))

    dark = QColor(58, 64, 84)
    if id_ is None or id_ == "sledge":
        p.rotate(-35)
        poly([(-5, -34), (5, -34), (5, 42), (-5, 42)], QColor("#c98a4b"))
        poly([(-26, -44), (26, -44), (26, -22), (-26, -22)], QColor("#aeb6c4"))
    elif id_ in SUPPLIES:
        sup = SUPPLIES[id_]
        c = QColor(sup.color)
        if sup.heals == "hp":
            if id_ == "patch":
                p.setBrush(QColor("#fff4ee"))
                p.drawRoundedRect(QRectF(-30, -18, 60, 36), 10, 10)
            else:
                p.setBrush(QColor("#fff4ee"))
                p.drawRoundedRect(QRectF(-34, -28, 68, 56), 8, 8)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawRect(QRectF(-6, -16, 12, 32))
            p.drawRect(QRectF(-16, -6, 32, 12))
        else:
            big = id_ == "powercell"
            w, h = (26, 34) if big else (18, 24)
            poly([(-8, -h - 12), (8, -h - 12), (8, -h), (-8, -h)], QColor("#e8eef8"))
            p.setBrush(c)
            p.drawRoundedRect(QRectF(-w, -h, 2 * w, 2 * h), 10, 10)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 150))
            p.drawRoundedRect(QRectF(-w + 7, -h + 8, 8, h), 4, 4)
    else:
        w = WEAPONS[id_]
        if id_ == "pepper":
            poly([(-26, -14), (30, -14), (30, 2), (-8, 2), (-12, 26), (-28, 26), (-22, 2), (-26, 2)], body)
        elif id_ == "buzzer":
            poly([(-34, -12), (30, -12), (30, 2), (-34, 2)], body)
            poly([(-6, 2), (6, 2), (6, 30), (-6, 30)], dark)
            poly([(-34, -4), (-44, -4), (-44, 14), (-36, 14), (-34, 2)], dark)
            poly([(30, -9), (44, -9), (44, -2), (30, -2)], dark)
        elif id_ == "thumper":
            poly([(-10, -12), (46, -12), (46, -2), (-10, -2)], dark)
            poly([(-2, 0), (30, 0), (30, 8), (-2, 8)], body)
            poly([(-10, -12), (-46, -2), (-46, 14), (-28, 10), (-10, 2)], QColor("#c98a4b"))
        elif id_ == "ranger":
            poly([(-20, -12), (30, -12), (30, 0), (-20, 0)], body)
            poly([(30, -9), (48, -9), (48, -3), (30, -3)], dark)
            poly([(-20, -12), (-46, -8), (-46, 10), (-28, 4), (-20, 0)], dark)
            poly([(-2, 0), (10, 0), (6, 26), (-8, 26)], dark)
            poly([(-12, 0), (-18, 16), (-10, 16), (-6, 0)], dark)
        elif id_ == "hunter":
            poly([(-22, -8), (32, -8), (32, 2), (-22, 2)], body)
            poly([(32, -6), (50, -6), (50, -1), (32, -1)], dark)
            poly([(-22, -8), (-48, -4), (-48, 12), (-30, 6), (-22, 2)], QColor("#c98a4b"))
            poly([(-8, -20), (18, -20), (18, -11), (-8, -11)], dark)
            poly([(0, 2), (8, 2), (6, 18), (-2, 18)], dark)
        elif id_ == "longshot":
            poly([(-24, -6), (26, -6), (26, 3), (-24, 3)], body)
            poly([(26, -4), (52, -4), (52, 0), (26, 0)], dark)
            poly([(-24, -6), (-50, -2), (-50, 14), (-32, 8), (-24, 3)], QColor("#c98a4b"))
            poly([(-14, -22), (20, -22), (20, -10), (-14, -10)], dark)
            p.setBrush(QColor("#7fd4ff"))
            p.drawEllipse(QPointF(20, -16), 6, 6)
        elif w.id == "boomtube":
            poly([(-44, -12), (40, -12), (40, 8), (-44, 8)], body)
            poly([(40, -17), (50, -17), (50, 13), (40, 13)], dark)
            poly([(-50, -15), (-44, -15), (-44, 11), (-50, 11)], dark)
            poly([(-6, 8), (6, 8), (4, 26), (-8, 26)], dark)
        else:
            poly([(-30, -10), (36, -10), (36, 4), (-30, 4)], body)
            p.setBrush(dark)
            p.drawEllipse(QPointF(0, 10), 17, 17)
            poly([(-30, -10), (-46, -4), (-46, 12), (-34, 8), (-30, 4)], dark)
    p.restore()


def draw_hud(p: QPainter, g) -> None:  # noqa: ANN001 - the game widget
    m = g.match
    me = m.player
    w, h = g.width(), g.height()
    u = h / 780.0
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    now = m.time
    view = g.view_actor()
    _edges(p, g, w, h, u)
    if g.scoped:
        _scope(p, w, h)
    if me.alive and me.state == "ground" and not g.map_open:
        _numbers(p, g, u)
        _crosshair(p, g, w, h, u)
    if g.map_open:
        _big_map(p, g, w, h, u)
        return
    _compass(p, g.yaw, w, u)
    _minimap(p, g, w, u, view)
    _feed(p, m, u, now)
    if m.banner is not None and now < m.banner[0] and me.alive:
        k = min((m.banner[0] - now) / 0.6, 1.0)
        text(p, QRectF(0, 84 * u, w, 40 * u), m.banner[1], 26 * u, m.banner[2], heavy=True, alpha=k)
    if me.alive:
        _vitals(p, me, h, u)
        _inventory(p, g, me, w, h, u)
        if me.state == "ship":
            ready = m.ship_s >= m.ship_open
            text(p, QRectF(0, h * 0.68, w, 44 * u), "SPACE  ·  DROP" if ready else "Doors opening ...", 34 * u,
                 "#ffe14d" if ready else "#ffffff", heavy=True)
            text(p, QRectF(0, h * 0.68 + 44 * u, w, 26 * u), "Open the map with M and pick where to land", 16 * u,
                 "#e9eef5", bold=False)
        elif me.state in ("fall", "glide"):
            alt = me.y - max(m.island.height(me.x, me.z), 0.0)
            text(p, QRectF(0, h * 0.72, w, 34 * u), f"{alt:.0f} m", 28 * u, "#ffffff", heavy=True)
            hint = "W to dive at where you look  ·  the canopy opens by itself" if me.state == "fall" \
                else "Steer with W A S D"
            text(p, QRectF(0, h * 0.72 + 34 * u, w, 24 * u), hint, 15 * u, "#e9eef5", bold=False)
        else:
            _prompt(p, g, me, w, h, u)


def _edges(p: QPainter, g, w: float, h: float, u: float) -> None:  # noqa: ANN001
    m = g.match
    me = m.player
    if not me.alive:
        return
    now = m.time
    hurt = max(0.0, 1.0 - (now - me.hurt_t) / 0.5)
    out = m.storm_dist(me.x, me.z) > 0 and m.storm["state"] != "idle" and me.state != "ship"
    low = me.hp < 30
    if hurt > 0 or out or low:
        grad = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) * 0.72)
        grad.setColorAt(0.0, QColor(0, 0, 0, 0))
        grad.setColorAt(0.62, QColor(0, 0, 0, 0))
        if out:
            a = 110 + int(40 * math.sin(now * 5.0))
            grad.setColorAt(1.0, QColor(170, 70, 255, a))
        else:
            a = int(190 * max(hurt, 0.35 + 0.2 * math.sin(now * 7.0) if low else 0.0))
            grad.setColorAt(1.0, QColor(255, 40, 60, a))
        p.fillRect(QRectF(0, 0, w, h), QBrush(grad))
    # who hit me: a wedge towards them
    if me.attacker is not None and now - me.attacked_t < 1.2:
        a = me.attacker
        ang = math.atan2(a.x - me.x, a.z - me.z) - g.yaw
        k = 1.0 - (now - me.attacked_t) / 1.2
        p.save()
        p.translate(w / 2, h / 2)
        p.rotate(math.degrees(ang))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 70, 80, int(210 * k)))
        r = 120 * u
        p.drawPolygon(QPolygonF([QPointF(-26 * u, -r), QPointF(0, -r - 22 * u), QPointF(26 * u, -r),
                                 QPointF(0, -r - 8 * u)]))
        p.restore()


def _scope(p: QPainter, w: float, h: float) -> None:
    r = h * 0.44
    path = QPainterPath()
    path.addRect(QRectF(0, 0, w, h))
    hole = QPainterPath()
    hole.addEllipse(QPointF(w / 2, h / 2), r, r)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(6, 8, 14))
    p.drawPath(path.subtracted(hole))
    p.setPen(QPen(QColor(10, 12, 20, 230), 2))
    p.drawLine(QPointF(w / 2 - r, h / 2), QPointF(w / 2 - 14, h / 2))
    p.drawLine(QPointF(w / 2 + 14, h / 2), QPointF(w / 2 + r, h / 2))
    p.drawLine(QPointF(w / 2, h / 2 - r), QPointF(w / 2, h / 2 - 14))
    p.drawLine(QPointF(w / 2, h / 2 + 14), QPointF(w / 2, h / 2 + r))
    p.setPen(QPen(QColor(10, 12, 20), 6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(w / 2, h / 2), r, r)


def _crosshair(p: QPainter, g, w: float, h: float, u: float) -> None:  # noqa: ANN001
    m = g.match
    me = m.player
    cx, cy = w / 2, h / 2
    now = m.time
    if g.build_piece is not None:
        p.setPen(QPen(QColor(120, 200, 255), 2 * u))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(cx - 7 * u, cy - 7 * u, 14 * u, 14 * u))
    elif not g.scoped:
        wpn = me.weapon()
        spread = 0.0
        if wpn is not None and wpn.cls != "tool":
            spread = wpn.spread + me.bloom + (min(me.speed / 7.8, 1.2) * 0.014 if wpn.pellets == 1 else 0.0)
            if me.ads:
                spread *= 0.5
        gap = 5 * u + spread * (h * 0.5) / math.tan(math.radians(g.cam.fov) / 2)
        col = QColor(255, 90, 90) if g.aim_enemy else QColor(255, 255, 255)
        ln = 8 * u
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a = QPointF(cx + dx * gap, cy + dy * gap)
            b = QPointF(cx + dx * (gap + ln), cy + dy * (gap + ln))
            p.setPen(QPen(QColor(0, 0, 0, 170), 4 * u, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(a, b)
            p.setPen(QPen(col, 2 * u, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(a, b)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        p.drawEllipse(QPointF(cx, cy), 1.6 * u, 1.6 * u)
    t, head = m.hit_marker
    if now - t < 0.22:
        k = 1.0 - (now - t) / 0.22
        c = QColor(255, 80, 80) if head else QColor(255, 255, 255)
        c.setAlphaF(k)
        p.setPen(QPen(c, 3 * u, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        a, b = (9 + 5 * (1 - k)) * u, (17 + 5 * (1 - k)) * u
        for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            p.drawLine(QPointF(cx + dx * a, cy + dy * a), QPointF(cx + dx * b, cy + dy * b))
    # reloading / using: a bar under the crosshair
    bar, label = 0.0, ""
    if me.reload_t > 0 and me.reload_full > 0:
        bar, label = 1.0 - me.reload_t / me.reload_full, "Reloading"
    elif me.use_t > 0 and me.use_full > 0:
        bar, label = 1.0 - me.use_t / me.use_full, "Using"
    if label:
        r = QRectF(cx - 70 * u, cy + 46 * u, 140 * u, 9 * u)
        panel(p, r, 4 * u, QColor(10, 12, 24, 190))
        p.setBrush(QColor("#ffe14d"))
        p.drawRoundedRect(QRectF(r.x() + 2, r.y() + 2, (r.width() - 4) * bar, r.height() - 4), 3 * u, 3 * u)
        text(p, QRectF(cx - 100 * u, cy + 56 * u, 200 * u, 22 * u), label, 13 * u, "#ffffff")


def _numbers(p: QPainter, g, u: float) -> None:  # noqa: ANN001
    if not g.options.get("damage_numbers", True):
        return
    for x, y, z, amount, head, shield, age in g.match.numbers:
        s = g.cam.project(x, y + age * 1.4, z)
        if s is None:
            continue
        col = "#ffe14d" if head else ("#6fd0ff" if shield else "#ffffff")
        size = (22 if head else 18) * u * (1.25 - 0.25 * min(age / 0.15, 1.0))
        text(p, QRectF(s[0] - 60, s[1] - 16, 120, 32), f"{amount:.0f}", size, col, heavy=True,
             alpha=min(1.0, (0.9 - age) / 0.3))


def _vitals(p: QPainter, me, h: float, u: float) -> None:  # noqa: ANN001
    x, bw, bh = 28 * u, 330 * u, 22 * u
    y = h - 34 * u - bh
    for val, top, c1, c2, yy in ((me.shield, MAX_SHIELD, "#38a8ff", "#8fe0ff", y - bh - 8 * u),
                                 (me.hp, MAX_HP, "#35d06a", "#b6f58a", y)):
        r = QRectF(x, yy, bw, bh)
        panel(p, r, 7 * u, QColor(10, 14, 30, 200), QColor(255, 255, 255, 60), 1.5 * u)
        k = max(min(val / top, 1.0), 0.0)
        if k > 0:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(c1))
            p.drawRoundedRect(QRectF(x + 3 * u, yy + 3 * u, (bw - 6 * u) * k, bh - 6 * u), 5 * u, 5 * u)
            p.setBrush(QColor(c2))
            p.drawRoundedRect(QRectF(x + 3 * u, yy + 3 * u, (bw - 6 * u) * k, (bh - 6 * u) * 0.42), 5 * u, 5 * u)
        text(p, QRectF(x + bw + 10 * u, yy - 4 * u, 70 * u, bh + 8 * u), f"{math.ceil(max(val, 0)):.0f}", 22 * u,
             c2, A.AlignLeft | A.AlignVCenter, heavy=True)


def _inventory(p: QPainter, g, me, w: float, h: float, u: float) -> None:  # noqa: ANN001
    tile, gap = 66 * u, 8 * u
    n = SLOTS + 1
    x0 = w - 28 * u - n * tile - (n - 1) * gap
    y0 = h - 34 * u - tile
    building = g.build_piece is not None
    for i in range(n):
        s = None if i == 0 else me.inv[i - 1]
        sel = (me.sel == i - 1) and not building
        r = QRectF(x0 + i * (tile + gap), y0 - (8 * u if sel else 0), tile, tile)
        grade = GRADES[s.grade][1] if (s is not None and s.is_weapon) else None
        fill = QColor(16, 20, 40, 200)
        if s is not None and not s.is_weapon:
            fill = QColor(SUPPLIES[s.id].color).darker(260)
            fill.setAlpha(215)
        elif grade:
            fill = QColor(grade).darker(250)
            fill.setAlpha(215)
        panel(p, r, 10 * u, fill, QColor("#ffffff") if sel else QColor(255, 255, 255, 50), (3 if sel else 1.5) * u)
        if grade:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(grade))
            p.drawRoundedRect(QRectF(r.x() + 6 * u, r.bottom() - 9 * u, r.width() - 12 * u, 4 * u), 2 * u, 2 * u)
        if i == 0 or s is not None:
            icon(p, r.adjusted(7 * u, 5 * u, -7 * u, -9 * u), None if i == 0 else s.id,
                 QColor(grade).lighter(150) if grade else None)
        text(p, QRectF(r.x() + 6 * u, r.y() + 2 * u, 20 * u, 18 * u), "F" if i == 0 else str(i), 12 * u,
             "#c9d2e6", A.AlignLeft | A.AlignVCenter)
        if s is not None:
            if s.is_weapon:
                text(p, QRectF(r.x(), r.bottom() - 30 * u, r.width() - 6 * u, 18 * u), str(s.mag), 14 * u, "#ffffff",
                     A.AlignRight | A.AlignVCenter)
            elif s.count > 1:
                text(p, QRectF(r.x(), r.bottom() - 30 * u, r.width() - 6 * u, 18 * u), f"x{s.count}", 14 * u,
                     "#ffffff", A.AlignRight | A.AlignVCenter)
    # what is in the hands: name and ammo
    s = me.held()
    line_y = y0 - 44 * u
    if building:
        name, col = f"Build: {PIECE_NAMES[g.build_piece]}  ({MAT_NAMES[me.build_mat]})", "#8fd0ff"
    elif s is None:
        name, col = "Sledge", "#e9eef5"
    elif s.is_weapon:
        wd = WEAPONS[s.id]
        name, col = f"{GRADES[s.grade][0]} {wd.name}", GRADES[s.grade][1]
        text(p, QRectF(x0, line_y - 34 * u, n * tile + (n - 1) * gap, 36 * u),
             f"{s.mag}  /  {me.ammo[wd.ammo]}", 28 * u, "#ffffff", A.AlignRight | A.AlignVCenter, heavy=True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(AMMO_COLORS[wd.ammo]))
    else:
        name, col = SUPPLIES[s.id].name, SUPPLIES[s.id].color
    text(p, QRectF(x0, line_y, n * tile + (n - 1) * gap, 26 * u), name, 17 * u, col, A.AlignRight | A.AlignVCenter)
    # materials and the build keys, up the right edge
    mx = w - 28 * u - 86 * u
    my = y0 - 44 * u - 46 * u - 3 * 34 * u
    for i in range(3):
        r = QRectF(mx, my + i * 34 * u, 86 * u, 29 * u)
        on = building and me.build_mat == i
        short = me.mats[i] < BUILD_COST
        panel(p, r, 7 * u, QColor(16, 20, 40, 190), QColor("#ffffff") if on else None, 2 * u)
        p.setPen(QPen(INK, 1.5 * u))
        p.setBrush(QColor(MAT_COLORS[i]))
        p.drawRoundedRect(QRectF(r.x() + 6 * u, r.y() + 6 * u, 17 * u, 17 * u), 3 * u, 3 * u)
        text(p, QRectF(r.x() + 28 * u, r.y(), 52 * u, r.height()), str(me.mats[i]), 16 * u,
             "#ff8a8a" if short and building else "#ffffff", A.AlignRight | A.AlignVCenter)
    bx = mx - 4 * (40 * u) - 12 * u
    by = my + 2 * 34 * u - 6 * u
    for i, key in enumerate("ZXCV"):
        r = QRectF(bx + i * 40 * u, by, 35 * u, 35 * u)
        on = g.build_piece == i
        panel(p, r, 7 * u, QColor(40, 110, 200, 220) if on else QColor(16, 20, 40, 180),
              QColor("#ffffff") if on else QColor(255, 255, 255, 50), (2.5 if on else 1.2) * u)
        _piece_glyph(p, r.adjusted(7 * u, 9 * u, -7 * u, -5 * u), i)
        text(p, QRectF(r.x() + 3 * u, r.y(), 14 * u, 13 * u), key, 10 * u, "#dfe6f5", A.AlignLeft | A.AlignVCenter)


def _piece_glyph(p: QPainter, r: QRectF, i: int) -> None:
    p.setPen(QPen(QColor("#ffffff"), max(r.width() * 0.09, 1.0), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                  Qt.PenJoinStyle.RoundJoin))
    p.setBrush(QColor(255, 255, 255, 70))
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    if i == 0:
        p.drawRect(QRectF(x + w * 0.1, y, w * 0.8, h))
    elif i == 1:
        p.drawPolygon(QPolygonF([QPointF(x, y + h * 0.75), QPointF(x + w * 0.35, y + h * 0.35), QPointF(x + w, y + h * 0.35),
                                 QPointF(x + w * 0.65, y + h * 0.75)]))
    elif i == 2:
        p.drawPolygon(QPolygonF([QPointF(x, y + h), QPointF(x + w, y + h), QPointF(x + w, y)]))
    else:
        p.drawPolygon(QPolygonF([QPointF(x, y + h), QPointF(x + w / 2, y), QPointF(x + w, y + h)]))


def _compass(p: QPainter, yaw: float, w: float, u: float) -> None:
    cw, y = 430 * u, 22 * u
    x0 = w / 2 - cw / 2
    panel(p, QRectF(x0, y, cw, 30 * u), 8 * u, QColor(12, 16, 34, 150))
    deg = math.degrees(yaw) % 360.0
    span = 110.0
    p.save()
    p.setClipRect(QRectF(x0 + 4 * u, y, cw - 8 * u, 40 * u))
    first = int((deg - span / 2) // 15) * 15
    names = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}
    for d in range(first, first + int(span) + 30, 15):
        off = ((d - deg + 180) % 360) - 180
        if abs(off) > span / 2 - 5:
            continue
        x = w / 2 + off / span * cw
        lab = names.get(d % 360)
        if lab:
            text(p, QRectF(x - 30 * u, y + 2 * u, 60 * u, 26 * u), lab, (17 if len(lab) == 1 else 13) * u,
                 "#ffe14d" if lab == "N" else "#ffffff")
        else:
            p.setPen(QPen(QColor(255, 255, 255, 150), 1.5 * u))
            p.drawLine(QPointF(x, y + 10 * u), QPointF(x, y + 20 * u))
            text(p, QRectF(x - 20 * u, y + 17 * u, 40 * u, 12 * u), str(d % 360), 8 * u, "#c9d2e6", bold=False,
                 shadow=False)
    p.restore()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#ffffff"))
    p.drawPolygon(QPolygonF([QPointF(w / 2 - 6 * u, y - 5 * u), QPointF(w / 2 + 6 * u, y - 5 * u),
                             QPointF(w / 2, y + 4 * u)]))
    text(p, QRectF(w / 2 - 30 * u, y + 32 * u, 60 * u, 18 * u), f"{deg:.0f}°", 13 * u, "#ffffff")


def map_to_px(x: float, z: float, rect: QRectF, x0: float, z0: float, span: float) -> QPointF:
    """World (x, z) to a point in ``rect`` showing the square of ``span`` metres whose south-west corner is x0, z0."""
    return QPointF(rect.x() + (x - x0) / span * rect.width(), rect.bottom() - (z - z0) / span * rect.height())


def _storm_on_map(p: QPainter, m, rect: QRectF, x0: float, z0: float, span: float, u: float) -> None:  # noqa: ANN001
    s = m.storm
    if s["state"] == "idle":
        return
    k = rect.width() / span
    c = map_to_px(s["cx"], s["cz"], rect, x0, z0, span)
    path = QPainterPath()
    path.addRect(rect)
    hole = QPainterPath()
    hole.addEllipse(c, s["r"] * k, s["r"] * k)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(150, 60, 230, 120))
    p.drawPath(path.subtracted(hole))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(214, 140, 255), 2.2 * u))
    p.drawEllipse(c, s["r"] * k, s["r"] * k)
    if s["state"] != "done" and s["nr"] > 0.5:
        n = map_to_px(s["ncx"], s["ncz"], rect, x0, z0, span)
        p.setPen(QPen(QColor(255, 255, 255), 1.8 * u))
        p.drawEllipse(n, s["nr"] * k, s["nr"] * k)


def _arrow(p: QPainter, c: QPointF, yaw: float, size: float, col: str = "#ffffff") -> None:
    p.save()
    p.translate(c)
    p.rotate(math.degrees(yaw))
    p.setPen(QPen(INK, size * 0.16))
    p.setBrush(QColor(col))
    p.drawPolygon(QPolygonF([QPointF(0, -size), QPointF(size * 0.7, size * 0.8), QPointF(0, size * 0.4),
                             QPointF(-size * 0.7, size * 0.8)]))
    p.restore()


def _minimap(p: QPainter, g, w: float, u: float, view) -> None:  # noqa: ANN001
    m = g.match
    size = 196 * u
    rect = QRectF(w - 28 * u - size, 24 * u, size, size)
    span = 230.0
    x0 = min(max(view.x - span / 2, 0.0), SIZE - span)
    z0 = min(max(view.z - span / 2, 0.0), SIZE - span)
    img: QImage = g.map_image
    k = img.width() / SIZE
    src = QRectF(x0 * k, (SIZE - z0 - span) * k, span * k, span * k)
    p.save()
    clip = QPainterPath()
    clip.addRoundedRect(rect, 14 * u, 14 * u)
    p.setClipPath(clip)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    p.drawImage(rect, img, src)
    _storm_on_map(p, m, rect, x0, z0, span, u)
    s = m.storm
    c = map_to_px(view.x, view.z, rect, x0, z0, span)
    if s["state"] != "idle" and math.hypot(view.x - s["ncx"], view.z - s["ncz"]) > s["nr"] > 0.5:
        p.setPen(QPen(QColor(255, 255, 255, 200), 1.5 * u, Qt.PenStyle.DashLine))
        p.drawLine(c, map_to_px(s["ncx"], s["ncz"], rect, x0, z0, span))
    if m.ship_s < m.ship_len:
        sp = map_to_px(m.ship_a[0] + m.ship_dir[0] * m.ship_s, m.ship_a[1] + m.ship_dir[1] * m.ship_s, rect, x0, z0,
                       span)
        p.setPen(QPen(INK, 1.5 * u))
        p.setBrush(QColor("#ff8a3d"))
        p.drawEllipse(sp, 6 * u, 6 * u)
    _arrow(p, c, view.yaw if view is not m.player else g.yaw, 9 * u)
    p.restore()
    p.setPen(QPen(QColor(255, 255, 255, 200), 2.5 * u))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(rect, 14 * u, 14 * u)
    # under it: the storm clock, players left, eliminations
    y = rect.bottom() + 8 * u
    bar = QRectF(rect.x(), y, size, 30 * u)
    panel(p, bar, 8 * u, QColor(12, 16, 34, 180))
    if s["state"] == "wait":
        label, col = _clock(s["t"]), "#ffffff"
    elif s["state"] == "shrink":
        label, col = _clock(s["t"]), "#e9a6ff"
    else:
        label, col = "--:--", "#c9d2e6"
    cx = bar.x() + 8 * u
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#c88bff") if s["state"] == "shrink" else QColor("#ffffff"))
    p.drawEllipse(QPointF(cx + 8 * u, bar.center().y()), 7 * u, 7 * u)
    p.setBrush(QColor(12, 16, 34))
    p.drawEllipse(QPointF(cx + 8 * u, bar.center().y()), 3.5 * u, 3.5 * u)
    text(p, QRectF(cx + 20 * u, y, 60 * u, 30 * u), label, 16 * u, col, A.AlignLeft | A.AlignVCenter)
    px = bar.x() + 92 * u
    p.setBrush(QColor("#ffffff"))
    p.drawEllipse(QPointF(px + 7 * u, bar.center().y() - 5 * u), 4.2 * u, 4.2 * u)
    p.drawRoundedRect(QRectF(px + 1 * u, bar.center().y(), 12 * u, 8 * u), 3 * u, 3 * u)
    text(p, QRectF(px + 19 * u, y, 36 * u, 30 * u), str(m.alive), 16 * u, "#ffffff", A.AlignLeft | A.AlignVCenter)
    kx = bar.x() + 146 * u
    p.setPen(QPen(QColor("#ffe14d"), 2.2 * u, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    cy = bar.center().y()
    p.drawEllipse(QPointF(kx + 7 * u, cy), 5.5 * u, 5.5 * u)
    p.drawLine(QPointF(kx + 7 * u, cy - 9 * u), QPointF(kx + 7 * u, cy + 9 * u))
    p.drawLine(QPointF(kx - 2 * u, cy), QPointF(kx + 16 * u, cy))
    text(p, QRectF(kx + 20 * u, y, 30 * u, 30 * u), str(view.kills), 16 * u, "#ffe14d", A.AlignLeft | A.AlignVCenter)
    if s["state"] in ("wait", "shrink"):
        cap = "Storm moves in" if s["state"] == "wait" else "Storm closing"
        text(p, QRectF(rect.x(), y + 31 * u, size, 18 * u), cap, 12 * u, col, A.AlignLeft | A.AlignVCenter)
    place = g.match.island.place_at(view.x, view.z)
    if place:
        text(p, QRectF(rect.x(), y + 31 * u, size, 18 * u), place, 12 * u, "#ffffff", A.AlignRight | A.AlignVCenter)


def _clock(t: float) -> str:
    t = max(int(math.ceil(t)), 0)
    return f"{t // 60}:{t % 60:02d}"


def _feed(p: QPainter, m, u: float, now: float) -> None:  # noqa: ANN001
    rows = [f for f in m.feed if now - f[0] < 7.5][-6:]
    y = 26 * u
    p.setFont(font(14 * u, True))
    fm = QFontMetricsF(p.font())
    for t, s, col in rows:
        k = min((7.5 - (now - t)) / 0.8, 1.0)
        wd = fm.horizontalAdvance(s) + 20 * u
        fill = QColor(12, 16, 34, int(165 * k))
        panel(p, QRectF(24 * u, y, wd, 24 * u), 7 * u, fill)
        text(p, QRectF(34 * u, y, wd, 24 * u), s, 14 * u, col, A.AlignLeft | A.AlignVCenter, alpha=k, shadow=False)
        y += 28 * u


def _prompt(p: QPainter, g, me, w: float, h: float, u: float) -> None:  # noqa: ANN001
    m = g.match
    f = m.focus
    if f is None or g.build_piece is not None:
        return
    if hasattr(f, "opened"):
        label, col = ("Open the ammo box" if f.small else "Open the chest"), "#ffe14d"
    else:
        full = all(s is not None for s in me.inv)
        verb = "Swap for" if (full and f.kind in ("weapon", "supply")) else "Pick up"
        if f.kind == "weapon":
            label, col = f"{verb} {GRADES[f.grade][0]} {WEAPONS[f.id].name}", GRADES[f.grade][1]
        else:
            label, col = f"{verb} {f.label()}", "#ffffff"
    p.setFont(font(17 * u, True))
    wd = QFontMetricsF(p.font()).horizontalAdvance(label) + 70 * u
    r = QRectF(w / 2 - wd / 2, h * 0.62, wd, 36 * u)
    panel(p, r, 10 * u, QColor(12, 16, 34, 200), QColor(col), 1.5 * u)
    key = QRectF(r.x() + 7 * u, r.y() + 6 * u, 26 * u, 24 * u)
    panel(p, key, 5 * u, QColor("#ffffff"))
    text(p, key, "E", 15 * u, "#141a2c", shadow=False)
    text(p, QRectF(r.x() + 42 * u, r.y(), wd - 50 * u, r.height()), label, 17 * u, col, A.AlignLeft | A.AlignVCenter)


def _big_map(p: QPainter, g, w: float, h: float, u: float) -> None:  # noqa: ANN001
    m = g.match
    me = m.player
    p.fillRect(QRectF(0, 0, w, h), QColor(8, 10, 24, 150))
    size = min(h * 0.82, w * 0.6)
    rect = QRectF(w / 2 - size / 2, h / 2 - size / 2, size, size)
    p.save()
    clip = QPainterPath()
    clip.addRoundedRect(rect, 18 * u, 18 * u)
    p.setClipPath(clip)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    p.drawImage(rect, g.map_image)
    _storm_on_map(p, m, rect, 0.0, 0.0, SIZE, u)
    if m.ship_s < m.ship_len:
        a = map_to_px(*m.ship_a, rect, 0, 0, SIZE)
        b = map_to_px(*m.ship_b, rect, 0, 0, SIZE)
        p.setPen(QPen(QColor(255, 255, 255, 220), 3 * u, Qt.PenStyle.DashLine))
        p.drawLine(a, b)
        sp = map_to_px(m.ship_a[0] + m.ship_dir[0] * m.ship_s, m.ship_a[1] + m.ship_dir[1] * m.ship_s, rect, 0, 0,
                       SIZE)
        p.setPen(QPen(INK, 2 * u))
        p.setBrush(QColor("#ff8a3d"))
        p.drawEllipse(sp, 9 * u, 9 * u)
    for pl in m.island.places:
        c = map_to_px(pl.cx, pl.cz, rect, 0, 0, SIZE)
        text(p, QRectF(c.x() - 110 * u, c.y() - pl.d * 2.0 * size / SIZE - 24 * u, 220 * u, 22 * u), pl.name,
             15 * u, "#ffffff", heavy=True)
    view = g.view_actor()
    _arrow(p, map_to_px(view.x, view.z, rect, 0, 0, SIZE), view.yaw if view is not me else g.yaw, 11 * u,
           "#ffe14d")
    p.restore()
    p.setPen(QPen(QColor(255, 255, 255, 220), 3 * u))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(rect, 18 * u, 18 * u)
    text(p, QRectF(rect.x(), rect.y() - 40 * u, size, 34 * u), "BRIGHTWATER ISLE", 26 * u, "#ffffff", heavy=True)
    lx = rect.right() + 22 * u
    rows = (("#d68cff", "The storm's edge"), ("#ffffff", "The next eye"), ("#ffe14d", "You"), ("#ff8a3d", "The ship"))
    for i, (col, lab) in enumerate(rows):
        y = rect.y() + 12 * u + i * 28 * u
        p.setPen(QPen(INK, 1.5 * u))
        p.setBrush(QColor(col))
        p.drawEllipse(QPointF(lx + 8 * u, y + 10 * u), 7 * u, 7 * u)
        text(p, QRectF(lx + 24 * u, y, 220 * u, 20 * u), lab, 14 * u, "#ffffff", A.AlignLeft | A.AlignVCenter)
    text(p, QRectF(lx, rect.y() + 136 * u, 240 * u, 20 * u), "M or Tab: close the map", 13 * u, "#c9d2e6",
         A.AlignLeft | A.AlignVCenter, bold=False)
    g.ui.button(p, QRectF(lx, rect.bottom() - 46 * u, 200 * u, 44 * u), "LEAVE MATCH", "leave", 16 * u,
                base=QColor(150, 50, 70))
