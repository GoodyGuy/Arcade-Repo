"""The in-game HUD, drawn with QPainter at full resolution: crosshair, hit markers, the damage vignette and direction
arcs, ammo and weapon name, prompts, and for zombies the round on a hazard-striped shift board, points with pop-ups,
tonic icons, power-up timers and banners, the down / game over screens."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient

from .data import POWERUP_NAMES, ZPERKS

ORANGE = QColor(242, 107, 29)
HUD_FONT = "Bahnschrift"


_FONTS: dict = {}


def hfont(size: float, bold: bool = True, family: str = HUD_FONT) -> QFont:
    key = (max(int(size), 6), bold, family)
    f = _FONTS.get(key)
    if f is None:
        f = QFont(family)
        f.setPixelSize(key[0])
        f.setBold(bold)
        if family == HUD_FONT:
            f.setStretch(QFont.Stretch.SemiCondensed)
        f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        _FONTS[key] = f
    return f


def text(p: QPainter, rect: QRectF, s: str, size: float, color: QColor | tuple, align=Qt.AlignmentFlag.AlignLeft,
         bold: bool = True, shadow: int = 2, family: str = HUD_FONT) -> None:  # noqa: ANN001
    p.setFont(hfont(size, bold, family))
    flags = align | Qt.AlignmentFlag.AlignVCenter
    if shadow:
        p.setPen(QColor(0, 0, 0, 170))
        p.drawText(rect.translated(shadow, shadow), flags, s)
    p.setPen(color if isinstance(color, QColor) else QColor(*color))
    p.drawText(rect, flags, s)


# --- shared ----------------------------------------------------------------------------------------------------------

def draw_glows(p: QPainter, lights: list, project, scale_x: float, scale_y: float, proj: float) -> None:  # noqa: ANN001
    """Additive glows for lights at world points (``project`` gives render-pixel coordinates or None)."""
    p.save()
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    p.setPen(Qt.PenStyle.NoPen)
    for x, y, z, color, radius, strength in lights:
        pt = project(x, y, z)
        if pt is None:
            continue
        sx, sy, depth = pt
        r = radius * proj / max(depth, 0.3) * scale_x
        if r < 2:
            continue
        r = min(r, 900)
        a = int(max(min(strength, 1.0), 0.0) * 120 * min(1.0, 6.0 / max(depth, 1.0)) ** 0.5)
        g = QRadialGradient(QPointF(sx * scale_x, sy * scale_y), r)
        g.setColorAt(0, QColor(color[0], color[1], color[2], a))
        g.setColorAt(0.4, QColor(color[0], color[1], color[2], a // 3))
        g.setColorAt(1, QColor(color[0], color[1], color[2], 0))
        p.setBrush(g)
        p.drawEllipse(QPointF(sx * scale_x, sy * scale_y), r, r)
    p.restore()


def draw_eye_glows(p: QPainter, glows: list, scale_x: float, scale_y: float) -> None:
    p.save()
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    p.setPen(Qt.PenStyle.NoPen)
    for x, y, r, color, _depth in glows:
        rr = max(r * scale_x * 2.6, 2.0)
        g = QRadialGradient(QPointF(x * scale_x, y * scale_y), rr)
        g.setColorAt(0, QColor(color[0], color[1], color[2], 230))
        g.setColorAt(0.35, QColor(color[0], color[1], color[2], 110))
        g.setColorAt(1, QColor(color[0], color[1], color[2], 0))
        p.setBrush(g)
        p.drawEllipse(QPointF(x * scale_x, y * scale_y), rr, rr)
    p.restore()


def draw_crosshair(p: QPainter, w: int, h: int, spread_px: float, alpha: float, color: QColor | None = None) -> None:
    if alpha <= 0.02:
        return
    c = color or QColor(255, 255, 255)
    c.setAlpha(int(220 * alpha))
    cx, cy = w / 2, h / 2
    gap = max(spread_px, 6)
    ln = max(h * 0.014, 8)
    p.setPen(QPen(QColor(0, 0, 0, int(140 * alpha)), 4))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        p.drawLine(QPointF(cx + dx * gap, cy + dy * gap), QPointF(cx + dx * (gap + ln), cy + dy * (gap + ln)))
    p.setPen(QPen(c, 2))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        p.drawLine(QPointF(cx + dx * gap, cy + dy * gap), QPointF(cx + dx * (gap + ln), cy + dy * (gap + ln)))


def draw_hitmarker(p: QPainter, w: int, h: int, t: float, kill: bool) -> None:
    if t <= 0:
        return
    cx, cy = w / 2, h / 2
    a = min(t / 0.18, 1.0)
    s0, s1 = h * 0.012, h * 0.028
    col = QColor(255, 60, 40, int(255 * a)) if kill else QColor(255, 255, 255, int(255 * a))
    p.setPen(QPen(col, 2.4 if kill else 2.0))
    for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        p.drawLine(QPointF(cx + dx * s0, cy + dy * s0), QPointF(cx + dx * s1, cy + dy * s1))


def draw_damage(p: QPainter, w: int, h: int, health_frac: float, hurt_t: float, dirs: list, angle: float) -> None:
    """The red screen edge as health drops and red arcs towards whoever hit."""
    k = max(0.0, 1.0 - health_frac) * (1.0 if hurt_t < 3.0 else max(0.0, 1.0 - (hurt_t - 3.0)))
    if k > 0.01:
        g = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) * 0.65)
        g.setColorAt(0.45, QColor(120, 0, 0, 0))
        g.setColorAt(0.8, QColor(150, 0, 0, int(120 * k)))
        g.setColorAt(1.0, QColor(90, 0, 0, int(230 * k)))
        p.fillRect(QRectF(0, 0, w, h), g)
        if health_frac < 0.35:
            p.fillRect(QRectF(0, 0, w, h), QColor(60, 0, 0, int(60 * k)))
    cx, cy = w / 2, h / 2
    r = h * 0.17
    for ang, life in dirs:
        rel = ang - angle - math.pi / 2  # 0 = up on screen
        a = min(life, 1.0)
        x, y = cx + math.cos(rel) * r, cy + math.sin(rel) * r
        poly = QPolygonF()
        for k2 in range(-6, 7):
            aa = rel + k2 * 0.04
            poly.append(QPointF(cx + math.cos(aa) * r, cy + math.sin(aa) * r))
        for k2 in range(6, -7, -1):
            aa = rel + k2 * 0.028
            poly.append(QPointF(cx + math.cos(aa) * (r + h * 0.03), cy + math.sin(aa) * (r + h * 0.03)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(220, 20, 10, int(200 * a)))
        p.drawPolygon(poly)
        del x, y


def draw_tracers(p: QPainter, effects: list, project, sx: float, sy: float, muzzle: tuple | None) -> None:  # noqa: ANN001
    p.save()
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    for e in effects:
        k = 1 - e.t / e.life
        if e.kind == "tracer" and muzzle is not None:
            end = project(e.x2, e.y2, e.z2)
            if end is None:
                continue
            p.setPen(QPen(QColor(255, 210, 140, int(170 * k)), 2))
            mx, my = muzzle
            ex, ey = end[0] * sx, end[1] * sy
            # only the far part of the streak
            p.drawLine(QPointF(mx + (ex - mx) * 0.35, my + (ey - my) * 0.35), QPointF(ex, ey))
        elif e.kind in ("puff", "blood", "spark"):
            pt = project(e.x, e.y, e.z)
            if pt is None:
                continue
            r = max(10.0 / max(pt[2], 0.5), 1.5) * sx * (1 + e.t * 3)
            col = (255, 230, 180) if e.kind == "puff" else (200, 20, 10)
            if e.kind == "blood":
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            g = QRadialGradient(QPointF(pt[0] * sx, pt[1] * sy), r)
            g.setColorAt(0, QColor(*col, int(200 * k)))
            g.setColorAt(1, QColor(*col, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(g)
            p.drawEllipse(QPointF(pt[0] * sx, pt[1] * sy), r, r)
            if e.kind == "blood":
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    p.restore()


def draw_ammo(p: QPainter, w: int, h: int, name: str, mag: int, reserve: int, mag_size: int, grenades: int,
              tacticals: int = 0, upgraded: bool = False, accent: QColor = ORANGE) -> None:
    s = h / 780
    right = w - 30 * s
    base = h - 40 * s
    # slanted panel
    panel = QPolygonF([QPointF(right - 300 * s, base - 70 * s), QPointF(right + 10 * s, base - 70 * s),
                       QPointF(right + 10 * s, base + 18 * s), QPointF(right - 330 * s, base + 18 * s)])
    g = QLinearGradient(QPointF(right - 330 * s, 0), QPointF(right, 0))
    g.setColorAt(0, QColor(0, 0, 0, 0))
    g.setColorAt(0.5, QColor(0, 0, 0, 110))
    g.setColorAt(1, QColor(0, 0, 0, 150))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(g)
    p.drawPolygon(panel)
    low = mag <= max(mag_size // 4, 1)
    mag_col = QColor(255, 70, 50) if low else QColor(255, 255, 255)
    text(p, QRectF(right - 260 * s, base - 66 * s, 150 * s, 50 * s), str(mag), 46 * s, mag_col,
         Qt.AlignmentFlag.AlignRight)
    text(p, QRectF(right - 105 * s, base - 56 * s, 110 * s, 40 * s), f"/ {reserve}", 26 * s,
         QColor(210, 210, 210), Qt.AlignmentFlag.AlignLeft)
    text(p, QRectF(right - 420 * s, base - 12 * s, 420 * s, 26 * s), name.upper(), 18 * s,
         QColor(190, 140, 255) if upgraded else QColor(230, 230, 230), Qt.AlignmentFlag.AlignRight)
    # the bullets of the magazine as ticks
    n = min(mag_size, 40)
    if n > 0:
        filled = round(mag / max(mag_size, 1) * n)
        tw = min(190 * s / n, 8 * s)
        for i in range(n):
            x = right - 6 * s - (i + 1) * tw
            p.fillRect(QRectF(x, base - 78 * s, tw * 0.6, 10 * s),
                       accent if n - 1 - i < filled else QColor(255, 255, 255, 50))
    for i in range(grenades):
        cx = right - 6 * s - i * 20 * s
        p.setBrush(QColor(110, 130, 90))
        p.setPen(QPen(QColor(20, 20, 20), 1.2))
        p.drawEllipse(QPointF(cx, base - 98 * s), 7 * s, 8 * s)
        p.drawRect(QRectF(cx - 2 * s, base - 110 * s, 4 * s, 5 * s))
    for i in range(tacticals):
        cx = right - 6 * s - (grenades + i) * 20 * s - 6 * s
        p.setBrush(QColor(170, 170, 190))
        p.drawRoundedRect(QRectF(cx - 5 * s, base - 108 * s, 10 * s, 18 * s), 3, 3)


def draw_prompt(p: QPainter, w: int, h: int, s: str, flash: float = 0.0) -> None:
    if not s:
        return
    sz = h / 780
    r = QRectF(0, h * 0.62, w, 34 * sz)
    text(p, r, s, 21 * sz, QColor(255, 255, 255), Qt.AlignmentFlag.AlignHCenter, False)


def draw_low_ammo(p: QPainter, w: int, h: int, mag: int, reserve: int, mag_size: int, t: float) -> None:
    sz = h / 780
    if mag == 0 and reserve == 0:
        msg, col = "NO AMMO", QColor(255, 60, 40)
    elif mag == 0 or mag <= mag_size // 4:
        msg, col = ("RELOAD" if reserve > 0 else "LOW AMMO"), QColor(255, 230, 120)
    else:
        return
    if int(t * 3) % 2:
        return
    text(p, QRectF(0, h * 0.56, w, 26 * sz), msg, 18 * sz, col, Qt.AlignmentFlag.AlignHCenter)


# --- zombies ---------------------------------------------------------------------------------------------------------

def draw_round(p: QPainter, x: float, y: float, size: float, rnd: int, flash: float, t: float) -> None:
    """The round on a works shift board: a hazard-striped steel plate, SHIFT stencilled on it and the number in big
    amber digits, which pulse white while one round changes to the next."""
    if rnd <= 0:
        return
    digits = str(rnd)
    plate = QRectF(x, y, size * (0.9 + 0.62 * len(digits)), size * 1.18)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(14, 15, 16, 210))
    p.drawRoundedRect(plate, size * 0.08, size * 0.08)
    band = QRectF(plate.x(), plate.y(), plate.width(), size * 0.16)
    p.setClipRect(band)
    p.fillRect(band, QColor(236, 186, 40, 230))
    p.setBrush(QColor(18, 18, 18, 230))
    step = size * 0.2
    xx = band.x() - band.height()
    while xx < band.right():
        p.drawPolygon(QPolygonF([QPointF(xx, band.bottom()), QPointF(xx + step * 0.5, band.bottom()),
                                 QPointF(xx + step * 0.5 + band.height(), band.top()),
                                 QPointF(xx + band.height(), band.top())]))
        xx += step
    p.setClipping(False)
    p.setFont(hfont(size * 0.16, True))
    p.setPen(QColor(170, 170, 160))
    for i, ch in enumerate("SHIFT"):
        p.drawText(QRectF(plate.x() + size * 0.08, plate.y() + size * (0.24 + i * 0.17), size * 0.2, size * 0.18),
                   int(Qt.AlignmentFlag.AlignCenter), ch)
    if flash > 0:
        k = 0.5 + 0.5 * math.sin(t * 6)
        color = QColor(255, int(176 + 79 * k), int(60 + 195 * k))
    else:
        color = QColor(255, 176, 60)
    num = QRectF(plate.x() + size * 0.32, plate.y() + size * 0.12, plate.width() - size * 0.38, plate.height() - size * 0.1)
    text(p, num, digits, size * 0.98, color, Qt.AlignmentFlag.AlignHCenter, True, 3, "Impact")
    p.restore()


def draw_points(p: QPainter, w: int, h: int, points: int, popups: list) -> None:
    s = h / 780
    x0, y0 = w - 250 * s, h - 190 * s
    r = QRectF(x0, y0, 210 * s, 40 * s)
    g = QLinearGradient(r.topLeft(), r.topRight())
    g.setColorAt(0, QColor(0, 0, 0, 0))
    g.setColorAt(1, QColor(0, 0, 0, 140))
    p.fillRect(r, g)
    text(p, r.adjusted(0, 0, -10 * s, 0), f"{points:,}", 30 * s, QColor(255, 230, 120), Qt.AlignmentFlag.AlignRight)
    for pop in popups:
        k = pop.t / 1.4
        a = int(255 * (1 - k))
        col = QColor(*pop.color, a)
        text(p, QRectF(x0 - 120 * s - k * 60 * s, y0 - 4 * s + pop.dx * s - k * 26 * s, 110 * s, 30 * s), pop.text,
             22 * s, col, Qt.AlignmentFlag.AlignRight, True, 1)


def perk_icon(p: QPainter, r: QRectF, perk_id: str) -> None:
    perk = ZPERKS[perk_id]
    col = QColor(*perk.color)
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    g.setColorAt(0, col.lighter(150))
    g.setColorAt(1, col.darker(160))
    p.setPen(QPen(QColor(20, 20, 20), 1.5))
    p.setBrush(g)
    p.drawRoundedRect(r, r.width() * 0.2, r.width() * 0.2)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255, 235))
    c, s = r.center(), r.width() * 0.26
    if perk_id == "jugg":
        p.drawPolygon(QPolygonF([QPointF(c.x() - s, c.y() - s), QPointF(c.x() + s, c.y() - s),
                                 QPointF(c.x() + s, c.y()), QPointF(c.x(), c.y() + s * 1.2),
                                 QPointF(c.x() - s, c.y())]))
    elif perk_id == "swift":
        p.drawPolygon(QPolygonF([QPointF(c.x() + s * 0.3, c.y() - s * 1.2), QPointF(c.x() - s * 0.7, c.y() + s * 0.15),
                                 QPointF(c.x(), c.y() + s * 0.15), QPointF(c.x() - s * 0.4, c.y() + s * 1.2),
                                 QPointF(c.x() + s * 0.8, c.y() - s * 0.25),
                                 QPointF(c.x() + s * 0.05, c.y() - s * 0.25)]))
    elif perk_id == "double":
        for dx in (-0.55, 0.55):
            p.drawEllipse(QPointF(c.x() + dx * s, c.y() + s * 0.3), s * 0.42, s * 0.55)
    elif perk_id == "stamina":
        p.drawPolygon(QPolygonF([QPointF(c.x() - s * 0.5, c.y() - s), QPointF(c.x() + s * 0.1, c.y() - s),
                                 QPointF(c.x() + s * 0.1, c.y() + s * 0.2), QPointF(c.x() + s, c.y() + s * 0.5),
                                 QPointF(c.x() + s, c.y() + s), QPointF(c.x() - s * 0.6, c.y() + s)]))
    elif perk_id == "deadeye":
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 235), max(s * 0.25, 1.5)))
        p.drawEllipse(c, s * 0.75, s * 0.75)
        p.drawLine(QPointF(c.x() - s * 1.1, c.y()), QPointF(c.x() + s * 1.1, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - s * 1.1), QPointF(c.x(), c.y() + s * 1.1))
    else:
        p.drawRect(QRectF(c.x() - s * 0.3, c.y() - s, s * 0.6, s * 2))
        p.drawRect(QRectF(c.x() - s, c.y() - s * 0.3, s * 2, s * 0.6))


def draw_perks(p: QPainter, w: int, h: int, perks: list) -> None:
    s = h / 780
    for i, pid in enumerate(perks):
        perk_icon(p, QRectF(30 * s + i * 48 * s, h - 80 * s, 40 * s, 40 * s), pid)


def draw_powerups(p: QPainter, w: int, h: int, active: dict, t: float) -> None:
    s = h / 780
    items = list(active.items())
    for i, (kind, left) in enumerate(items):
        if left < 5 and int(t * 6) % 2:
            continue
        cx = w / 2 + (i - (len(items) - 1) / 2) * 70 * s
        g = QRadialGradient(QPointF(cx, h - 70 * s), 28 * s)
        g.setColorAt(0, QColor(255, 214, 120, 230))
        g.setColorAt(1, QColor(200, 110, 20, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawEllipse(QPointF(cx, h - 70 * s), 28 * s, 28 * s)
        label = {"double_points": "x2", "fire_sale": "%"}.get(kind, "☠")
        text(p, QRectF(cx - 30 * s, h - 92 * s, 60 * s, 44 * s), label, 26 * s, QColor(70, 30, 6),
             Qt.AlignmentFlag.AlignHCenter, True, 0, "Segoe UI Symbol" if label == "☠" else HUD_FONT)


def draw_banner(p: QPainter, w: int, h: int, banner: tuple) -> None:
    s, left = banner
    if not s or left <= 0:
        return
    sz = h / 780
    k = min(left / 0.4, 1.0) * min((2.6 - left) / 0.2 + 0.2, 1.0) if left < 2.6 else 1.0
    size = 54 * sz * (1.0 + 0.15 * max(0.0, (left - 2.3) / 0.3))
    a = int(255 * max(min(k, 1.0), 0.0))
    r = QRectF(0, h * 0.16, w, size * 1.6)
    p.save()
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    g = QRadialGradient(r.center(), w * 0.25)
    g.setColorAt(0, QColor(255, 170, 60, a // 3))
    g.setColorAt(1, QColor(0, 0, 0, 0))
    p.fillRect(r.adjusted(0, -40 * sz, 0, 40 * sz), g)
    p.restore()
    text(p, r, s, size, QColor(255, 236, 200, a), Qt.AlignmentFlag.AlignHCenter, True, 3, "Impact")


def draw_zombies_hud(p: QPainter, w: int, h: int, z, t: float) -> None:  # noqa: ANN001 - ZombiesMode
    s = h / 780
    draw_round(p, 36 * s, 26 * s, 78 * s, z.round, z.round_flash if z.between_rounds() or z.round_flash > 0 else 0, t)
    draw_points(p, w, h, z.points, z.popups)
    draw_perks(p, w, h, z.perks)
    draw_powerups(p, w, h, z.active, t)
    draw_banner(p, w, h, z.banner)
    gun = z.player.gun
    if gun is not None:
        draw_ammo(p, w, h, gun.name, gun.mag, gun.reserve, gun.mag_size(), z.player.grenades, 0, gun.upgraded,
                  QColor(255, 214, 80))
        draw_low_ammo(p, w, h, gun.mag, gun.reserve, gun.mag_size(), t)
    draw_prompt(p, w, h, z.prompt)
    if z.down_t > 0:
        k = min(z.down_t / 4.0, 1.0)
        p.fillRect(QRectF(0, 0, w, h), QColor(60, 60, 60, int(150 * k)))
        text(p, QRectF(0, h * 0.4, w, 50 * s), f"SECOND WIND  ({max(3 - z.rally_uses, 0)} left)", 40 * s,
             QColor(120, 190, 255), Qt.AlignmentFlag.AlignHCenter)
        bw = 300 * s
        p.fillRect(QRectF((w - bw) / 2, h * 0.5, bw, 10 * s), QColor(0, 0, 0, 160))
        p.fillRect(QRectF((w - bw) / 2, h * 0.5, bw * (1 - k), 10 * s), QColor(120, 190, 255))
    if z.repair_t > 0:
        bw = 180 * s
        p.fillRect(QRectF((w - bw) / 2, h * 0.67, bw, 6 * s), QColor(0, 0, 0, 150))
        p.fillRect(QRectF((w - bw) / 2, h * 0.67, bw * min(z.repair_t / 0.7, 1), 6 * s), QColor(255, 255, 255))


def draw_game_over(p: QPainter, w: int, h: int, z, t: float) -> None:  # noqa: ANN001
    s = h / 780
    k = min(z.over_t / 1.5, 1.0)
    p.fillRect(QRectF(0, 0, w, h), QColor(10, 0, 0, int(200 * k)))
    if z.over_t < 0.8:
        return
    a = int(255 * min((z.over_t - 0.8) / 0.8, 1.0))
    text(p, QRectF(0, h * 0.2, w, 90 * s), "GAME OVER", 84 * s, QColor(200, 20, 10, a), Qt.AlignmentFlag.AlignHCenter,
         True, 4, "Impact")
    rounds = max(z.round, 1)
    text(p, QRectF(0, h * 0.33, w, 40 * s), f"You Survived {rounds} Round{'s' if rounds != 1 else ''}", 30 * s,
         QColor(255, 255, 255, a), Qt.AlignmentFlag.AlignHCenter)
    st = z.stats
    rows = (("Kills", st.kills), ("Headshots", st.headshots), ("Points earned", st.points_total),
            ("Doors opened", st.doors), ("Tonics drunk", st.perks), ("Crates opened", st.box_uses),
            ("Downs", st.downs))
    y = h * 0.43
    for label, val in rows:
        text(p, QRectF(w / 2 - 230 * s, y, 230 * s, 30 * s), label, 20 * s, QColor(200, 200, 200, a),
             Qt.AlignmentFlag.AlignLeft, False)
        text(p, QRectF(w / 2, y, 230 * s, 30 * s), f"{val:,}", 20 * s, QColor(255, 214, 80, a),
             Qt.AlignmentFlag.AlignRight)
        y += 32 * s
    if z.over_t > 3.0 and int(t * 2) % 2 == 0:
        text(p, QRectF(0, h * 0.86, w, 30 * s), "Press SPACE or click to continue", 18 * s, QColor(255, 255, 255, a),
             Qt.AlignmentFlag.AlignHCenter, False)


def draw_beam(p: QPainter, project, xy: tuple, sx: float, sy: float, t: float) -> None:  # noqa: ANN001
    """The Lucky Crate's column of light (seen across the map, through walls, so the crate can be found)."""
    top = project(xy[0], xy[1], 6.0)
    bot = project(xy[0], xy[1], 0.5)
    if top is None or bot is None:
        return
    depth = bot[2]
    wid = max(24.0 / max(depth, 1.0), 2.0) * sx * (1 + 0.1 * math.sin(t * 3))
    x0, y0, x1, y1 = bot[0] * sx, bot[1] * sy, top[0] * sx, top[1] * sy
    p.save()
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    g = QLinearGradient(QPointF(x0, y0), QPointF(x1, y1))
    g.setColorAt(0, QColor(255, 196, 100, 90))
    g.setColorAt(1, QColor(255, 196, 100, 0))
    path = QPainterPath()
    path.moveTo(x0 - wid, y0)
    path.lineTo(x0 + wid, y0)
    path.lineTo(x1 + wid * 0.6, y1)
    path.lineTo(x1 - wid * 0.6, y1)
    path.closeSubpath()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(g)
    p.drawPath(path)
    p.restore()


__all__ = ["POWERUP_NAMES", "draw_zombies_hud", "draw_game_over"]
