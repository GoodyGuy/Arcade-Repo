"""The home screen's backgrounds: four themes (Synthwave, 80s Carpet, Cyber City, Spooky Night), each a cached static
layer (sky, skyline, floor) plus a few cheap animated extras drawn every frame (the scrolling grid, twinkling stars,
spotlights, rain, bats, fog), the flickering neon wall signs and the secret party's disco lights.

The floor shares the perspective of the cabinet carousel: ``horizon`` is the eye level (the vanishing line) and
``floor_c`` = (floor y - horizon) * depth is the same for every point of the floor, so ``y = horizon + floor_c / z``.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, \
    QRadialGradient, QTransform

from app.arcade import pixelfont

THEMES = ("synthwave", "carpet", "cyber", "spooky")


def _frac(x: float) -> float:
    return x - math.floor(x)


def _hash(*xs: float) -> float:
    """A repeatable pseudo-random number in [0, 1) from numbers (no state)."""
    v = math.sin(sum(x * k for x, k in zip(xs, (12.9898, 78.233, 37.719, 4.581), strict=False))) * 43758.5453
    return _frac(v)


def neon_text_pixmap(text: str, size: float, color: QColor, family: str = "Segoe Script", bold: bool = True,
                     lit: bool = True, italic: bool = False, pad: float = 0.5) -> QPixmap:
    """Text drawn as a glowing neon tube (or the dark, unlit tube); ``pad``: the margin for the glow, of ``size``."""
    f = QFont(family)
    f.setPixelSize(max(int(size), 4))
    f.setBold(bold)
    f.setItalic(italic)
    path = QPainterPath()
    path.addText(0, 0, f, text)
    box = path.boundingRect()
    pad = size * pad
    pm = QPixmap(int(box.width() + 2 * pad) + 2, int(box.height() + 2 * pad) + 2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.translate(pad - box.x(), pad - box.y())
    p.setBrush(Qt.BrushStyle.NoBrush)
    tube = max(size * 0.07, 1.5)
    if lit:
        for k, a in ((5.5, 18), (3.8, 30), (2.6, 55), (1.7, 110)):
            c = QColor(color)
            c.setAlpha(a)
            p.setPen(QPen(c, tube * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPath(path)
        p.setPen(QPen(QColor(color).lighter(150), tube, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.drawPath(path)
        p.setPen(QPen(QColor(255, 255, 255, 200), tube * 0.35))
        p.drawPath(path)
    else:
        c = QColor(color).darker(400)
        c.setAlpha(200)
        p.setPen(QPen(c, tube, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(path)
    p.end()
    return pm


class NeonSign:
    """A wall sign that is lit and, now and then, flickers."""

    def __init__(self, text: str, size: float, color: str, family: str, seed: int, flaky: float = 0.04) -> None:
        self.lit = neon_text_pixmap(text, size, QColor(color), family)
        self.dark = neon_text_pixmap(text, size, QColor(color), family, lit=False)
        self.seed = seed
        self.flaky = flaky

    def is_lit(self, t: float) -> bool:
        slot = math.floor(t * 12)
        burst = _hash(math.floor(t / 3.0), self.seed) < 0.35  # some 3 s windows have a flicker burst
        return not (burst and _hash(slot, self.seed + 1) < self.flaky * 8)

    def paint(self, p: QPainter, centre: QPointF, t: float, scale: float = 1.0) -> None:
        pm = self.lit if self.is_lit(t) else self.dark
        w, h = pm.width() * scale, pm.height() * scale
        p.drawPixmap(QRectF(centre.x() - w / 2, centre.y() - h / 2, w, h), pm, QRectF(pm.rect()))


SIGNS = {
    "synthwave": (("Open 24/7", "#29f3ff", "Segoe Script"), ("High Scores", "#ff3fb4", "Segoe Script")),
    "carpet": (("Game On!", "#ffe14d", "Segoe Script"), ("Open 24/7", "#ff4060", "Segoe Script")),
    "cyber": (("CYBER ZONE", "#00ffd5", "Impact"), ("OPEN 24/7", "#ff2bd6", "Impact")),
    "spooky": (("Open till dawn", "#ff8a1f", "Segoe Script"), ("Enter if you dare", "#7dff4a", "Segoe Script")),
}


def _dot_pixmap(color: QColor, radius: int) -> QPixmap:
    pm = QPixmap(radius * 2, radius * 2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    g = QRadialGradient(QPointF(radius, radius), radius)
    c0, c1 = QColor(color), QColor(color)
    c0.setAlpha(255)
    c1.setAlpha(0)
    g.setColorAt(0.0, QColor(255, 255, 255, 255))
    g.setColorAt(0.18, c0)
    g.setColorAt(1.0, c1)
    p.fillRect(pm.rect(), g)
    p.end()
    return pm


class Scene:
    """One theme's background for one window size."""

    def __init__(self, theme: str, w: int, h: int, horizon: float, floor_c: float, focal: float) -> None:
        self.theme = theme if theme in THEMES else "synthwave"
        self.w, self.h = max(w, 1), max(h, 1)
        self.horizon = horizon
        self.floor_c = floor_c
        self.focal = focal
        self.vx = self.w / 2
        self.unit = self.h / 780  # 1 at the reference size
        rng = np.random.default_rng(7)
        self.stars = [(float(rng.uniform(0, self.w)), float(rng.uniform(0, horizon * 0.92)), float(rng.uniform(0, 6.3)),
                       float(rng.uniform(0.8, 2.2))) for _ in range(28)]
        signs = SIGNS[self.theme]
        size = 30 * self.unit
        self.signs = [NeonSign(text, size, color, family, i * 7 + 3) for i, (text, color, family) in enumerate(signs)]
        self.dots = {name: _dot_pixmap(QColor(c), max(int(6 * self.unit), 3)) for name, c in
                     (("pink", "#ff3fb4"), ("cyan", "#29f3ff"), ("gold", "#ffd84d"), ("green", "#7dff4a"),
                      ("white", "#ffffff"), ("orange", "#ff8a1f"))}
        self.static = QPixmap(self.w, self.h)
        p = QPainter(self.static)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        getattr(self, f"_static_{self.theme}")(p)
        p.end()
        self._fog = self._make_fog() if self.theme == "spooky" else None

    # --- helpers --------------------------------------------------------------------------------------------------

    def floor_y(self, z: float) -> float:
        return self.horizon + self.floor_c / max(z, 1e-3)

    def floor_x(self, x: float, z: float) -> float:
        return self.vx + x * self.focal / max(z, 1e-3)

    def sign_positions(self) -> list[QPointF]:
        y = self.h * 0.215
        return [QPointF(self.w * 0.11, y), QPointF(self.w * 0.89, y)]

    def _vertical_grid(self, p: QPainter, color: QColor, spacing: float = 1.0, alpha_far: int = 30) -> None:
        z_bottom = self.floor_c / max(self.h - self.horizon, 1)
        span = (self.w / 2) * z_bottom / self.focal * 1.15
        g = QLinearGradient(0, self.horizon, 0, self.h)
        c0, c1 = QColor(color), QColor(color)
        c0.setAlpha(alpha_far)
        g.setColorAt(0.0, c0)
        g.setColorAt(1.0, c1)
        p.setPen(QPen(QBrush(g), max(1.2 * self.unit, 1.0)))
        n = int(span / spacing) + 1
        for k in range(-n, n + 1):
            p.drawLine(QPointF(self.vx, self.horizon), QPointF(self.floor_x(k * spacing, z_bottom), self.h))

    def _horizontal_grid(self, p: QPainter, t: float, color: QColor, speed: float = 0.7) -> None:
        phase = _frac(t * speed)
        pen = QPen(color, max(1.4 * self.unit, 1.0))
        z_bottom = self.floor_c / max(self.h - self.horizon, 1)
        for k in range(26):
            z = z_bottom + (k + 1 - phase) * 0.9
            y = self.floor_y(z)
            if y - self.horizon < 2:
                break
            a = min(1.0, (y - self.horizon) / (self.h - self.horizon) * 2.2)
            c = QColor(color)
            c.setAlphaF(color.alphaF() * a)
            pen.setColor(c)
            p.setPen(pen)
            p.drawLine(QPointF(0, y), QPointF(self.w, y))

    def _stars_static(self, p: QPainter, count: int, top: float, bottom: float, seed: int = 3) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(count):
            x, y = rng.uniform(0, self.w), rng.uniform(top, bottom)
            a = int(rng.uniform(40, 200))
            r = rng.choice((1.0, 1.0, 1.5, 2.0)) * max(self.unit, 0.8)
            p.fillRect(QRectF(x, y, r, r), QColor(255, 255, 255, a))

    def _twinkle(self, p: QPainter, t: float) -> None:
        for x, y, ph, sp in self.stars:
            a = 0.5 + 0.5 * math.sin(t * sp + ph)
            if a < 0.15:
                continue
            r = 1.6 * self.unit * (0.6 + a)
            p.fillRect(QRectF(x - r / 2, y - r / 2, r, r), QColor(255, 255, 255, int(255 * a)))
            if a > 0.85:
                c = QColor(255, 255, 255, int(120 * (a - 0.85) / 0.15))
                p.fillRect(QRectF(x - r * 2.5, y - 0.5, r * 5, 1), c)
                p.fillRect(QRectF(x - 0.5, y - r * 2.5, 1, r * 5), c)

    def _shooting_star(self, p: QPainter, t: float, period: float = 6.5) -> None:
        n = math.floor(t / period)
        local = t - n * period
        if local > 0.9 or _hash(n, 5) < 0.3:
            return
        x0 = self.w * (0.1 + 0.8 * _hash(n, 1))
        y0 = self.horizon * (0.05 + 0.4 * _hash(n, 2))
        d = -1 if _hash(n, 3) < 0.5 else 1
        k = local / 0.9
        length = self.w * 0.22
        head = QPointF(x0 + d * length * k, y0 + length * 0.35 * k)
        tail = QPointF(head.x() - d * length * 0.35, head.y() - length * 0.12)
        g = QLinearGradient(tail, head)
        g.setColorAt(0.0, QColor(255, 255, 255, 0))
        g.setColorAt(1.0, QColor(255, 255, 255, int(230 * math.sin(math.pi * k))))
        p.setPen(QPen(QBrush(g), 2.2 * self.unit, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(tail, head)

    def _motes(self, p: QPainter, t: float, n: int, dot: str, speed: float = 0.05, top: float = 0.2,
               drift: float = 0.02, size: float = 1.0) -> None:
        pm = self.dots[dot]
        for i in range(n):
            s1, s2, s3 = _hash(i, 1.7), _hash(i, 2.9), _hash(i, 4.1)
            yk = _frac(s2 - t * speed * (0.5 + s3))
            x = (_frac(s1 + drift * math.sin(t * 0.7 + i))) * self.w
            y = self.h * top + yk * self.h * (1 - top)
            a = math.sin(math.pi * yk)
            r = pm.width() * size * (0.5 + 0.6 * s3) / 2
            p.setOpacity(max(a, 0.0) * 0.8)
            p.drawPixmap(QRectF(x - r, y - r, 2 * r, 2 * r), pm, QRectF(pm.rect()))
        p.setOpacity(1.0)

    # --- synthwave ------------------------------------------------------------------------------------------------

    def _static_synthwave(self, p: QPainter) -> None:
        w, h, hy = self.w, self.h, self.horizon
        sky = QLinearGradient(0, 0, 0, hy)
        sky.setColorAt(0.0, QColor("#07011a"))
        sky.setColorAt(0.45, QColor("#1c0640"))
        sky.setColorAt(0.8, QColor("#4b0d5e"))
        sky.setColorAt(1.0, QColor("#a8256f"))
        p.fillRect(QRectF(0, 0, w, hy), sky)
        self._stars_static(p, 140, 0, hy * 0.8)
        # the sun, with a glow, striped near the horizon
        r = h * 0.25
        c = QPointF(self.vx, hy - r * 0.42)
        glow = QRadialGradient(c, r * 2.0)
        glow.setColorAt(0.0, QColor(255, 120, 90, 120))
        glow.setColorAt(0.5, QColor(255, 60, 140, 40))
        glow.setColorAt(1.0, QColor(255, 60, 140, 0))
        p.fillRect(QRectF(0, 0, w, hy), glow)
        sun = QLinearGradient(0, c.y() - r, 0, c.y() + r)
        sun.setColorAt(0.0, QColor("#fff27a"))
        sun.setColorAt(0.5, QColor("#ffa13d"))
        sun.setColorAt(1.0, QColor("#ff2e88"))
        disc = QPainterPath()
        disc.addEllipse(c, r, r)
        clip = QPainterPath()
        clip.addRect(QRectF(0, 0, w, hy))
        for k in range(8):
            y = c.y() + r * 0.05 + k * r * 0.12
            band = QPainterPath()
            band.addRect(QRectF(0, y, w, r * 0.012 + k * r * 0.011))
            clip = clip.subtracted(band)
        p.fillPath(disc.intersected(clip), QBrush(sun))
        # wireframe mountains on both sides
        rng = np.random.default_rng(11)
        for side in (-1, 1):
            pts = [QPointF(self.vx + side * w * 0.16, hy)]
            x = 0.16
            while x < 0.56:
                x += float(rng.uniform(0.03, 0.07))
                pts.append(QPointF(self.vx + side * w * x, hy - h * float(rng.uniform(0.05, 0.17)) * (0.6 + x)))
            pts.append(QPointF(self.vx + side * w * 0.62, hy))
            poly = QPolygonF(pts)
            g = QLinearGradient(0, hy - h * 0.15, 0, hy)
            g.setColorAt(0.0, QColor("#2a0a4a"))
            g.setColorAt(1.0, QColor("#12022a"))
            p.setBrush(g)
            p.setPen(QPen(QColor(41, 243, 255, 160), 1.6 * self.unit))
            p.drawPolygon(poly)
            p.setPen(QPen(QColor(41, 243, 255, 50), 1.0))
            for a, b in zip(pts[1:-2], pts[2:-1], strict=False):
                mid = QPointF((a.x() + b.x()) / 2, hy)
                p.drawLine(a, mid)
                p.drawLine(b, mid)
        # the floor and its static lines
        ground = QLinearGradient(0, hy, 0, h)
        ground.setColorAt(0.0, QColor("#2a0645"))
        ground.setColorAt(0.25, QColor("#12022a"))
        ground.setColorAt(1.0, QColor("#05000f"))
        p.fillRect(QRectF(0, hy, w, h - hy), ground)
        self._vertical_grid(p, QColor(255, 46, 200, 170), 1.0)
        haze = QLinearGradient(0, hy - h * 0.03, 0, hy + h * 0.05)
        haze.setColorAt(0.0, QColor(255, 80, 160, 0))
        haze.setColorAt(0.45, QColor(255, 120, 200, 140))
        haze.setColorAt(1.0, QColor(255, 80, 160, 0))
        p.fillRect(QRectF(0, hy - h * 0.03, w, h * 0.08), haze)

    def _dynamic_synthwave(self, p: QPainter, t: float) -> None:
        self._twinkle(p, t)
        self._shooting_star(p, t)
        self._horizontal_grid(p, t, QColor(255, 46, 200, 200))
        self._motes(p, t, 16, "pink", top=0.3, size=0.8)

    # --- 80s carpet -----------------------------------------------------------------------------------------------

    def _carpet_tile(self, size: int) -> QPixmap:
        pm = QPixmap(size, size)
        pm.fill(QColor("#07041a"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rng = np.random.default_rng(42)
        colors = [QColor(c) for c in ("#19e3ff", "#ff2fa0", "#ffe14d", "#8a5cff", "#38ff8a")]

        def wrapped(draw) -> None:  # noqa: ANN001 - draws a shape again across the edges (a seamless tile)
            for dx in (-size, 0, size):
                for dy in (-size, 0, size):
                    p.save()
                    p.translate(dx, dy)
                    draw()
                    p.restore()

        s = size / 256
        for _ in range(14):  # squiggles
            x, y = rng.uniform(0, size, 2)
            c = colors[int(rng.integers(len(colors)))]
            path = QPainterPath(QPointF(x, y))
            ang = rng.uniform(0, 6.28)
            for _k in range(4):
                x2, y2 = x + math.cos(ang) * 22 * s, y + math.sin(ang) * 22 * s
                path.quadTo(QPointF((x + x2) / 2 + rng.uniform(-14, 14) * s, (y + y2) / 2 + rng.uniform(-14, 14) * s),
                            QPointF(x2, y2))
                x, y = x2, y2
                ang += rng.uniform(-1.2, 1.2)
            pen = QPen(c, 3.2 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            wrapped(lambda path=path, pen=pen: (p.setPen(pen), p.setBrush(Qt.BrushStyle.NoBrush), p.drawPath(path)))
        for _ in range(7):  # planets with rings
            x, y = rng.uniform(0, size, 2)
            r = rng.uniform(6, 13) * s
            c = colors[int(rng.integers(len(colors)))]
            c2 = colors[int(rng.integers(len(colors)))]

            def planet(x=x, y=y, r=r, c=c, c2=c2) -> None:  # noqa: ANN001
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c)
                p.drawEllipse(QPointF(x, y), r, r)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(c2, 2 * s))
                p.drawEllipse(QPointF(x, y), r * 1.9, r * 0.55)

            wrapped(planet)
        for _ in range(10):  # triangles
            x, y = rng.uniform(0, size, 2)
            r = rng.uniform(6, 11) * s
            a0 = rng.uniform(0, 6.28)
            c = colors[int(rng.integers(len(colors)))]
            tri = QPolygonF([QPointF(x + r * math.cos(a0 + k * 2.094), y + r * math.sin(a0 + k * 2.094))
                             for k in range(3)])
            wrapped(lambda tri=tri, c=c: (p.setPen(QPen(c, 2.4 * s)), p.setBrush(Qt.BrushStyle.NoBrush),
                                          p.drawPolygon(tri)))
        for _ in range(40):  # specks
            x, y = rng.uniform(0, size, 2)
            c = colors[int(rng.integers(len(colors)))]
            wrapped(lambda x=x, y=y, c=c: p.fillRect(QRectF(x, y, 2.5 * s, 2.5 * s), c))
        p.end()
        return pm

    def _static_carpet(self, p: QPainter) -> None:
        w, h, hy = self.w, self.h, self.horizon
        z_wall = 11.0
        y_wall = self.floor_y(z_wall)
        wall = QLinearGradient(0, 0, 0, y_wall)
        wall.setColorAt(0.0, QColor("#05030c"))
        wall.setColorAt(0.35, QColor("#140a2c"))
        wall.setColorAt(1.0, QColor("#24123f"))
        p.fillRect(QRectF(0, 0, w, y_wall), wall)
        # wall panels and a neon stripe
        p.setPen(QPen(QColor(255, 255, 255, 14), 1))
        step = w / 14
        for k in range(15):
            p.drawLine(QPointF(k * step, h * 0.12), QPointF(k * step, y_wall))
        for y, c in ((y_wall - h * 0.085, "#ff2fa0"), (y_wall - h * 0.07, "#19e3ff")):
            for wdt, a in ((9, 30), (5, 60), (2, 230)):
                col = QColor(c)
                col.setAlpha(a)
                p.setPen(QPen(col, wdt * self.unit))
                p.drawLine(QPointF(0, y), QPointF(w, y))
        # ceiling with spotlight cans
        ceil = QLinearGradient(0, 0, 0, h * 0.12)
        ceil.setColorAt(0.0, QColor("#000000"))
        ceil.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(QRectF(0, 0, w, h * 0.12), ceil)
        # the carpet in perspective
        tile = self._carpet_tile(256)
        z_near = self.floor_c / max(h - hy, 1) * 0.98
        xr = (w / 2) * z_near / self.focal * 1.2
        xr = max(xr, (w / 2) * z_wall / self.focal * 1.05)
        src = QPolygonF([QPointF(0, 0), QPointF(2 * xr * 128, 0), QPointF(2 * xr * 128, (z_wall - z_near) * 128),
                         QPointF(0, (z_wall - z_near) * 128)])
        dst = QPolygonF([QPointF(self.floor_x(-xr, z_wall), y_wall), QPointF(self.floor_x(xr, z_wall), y_wall),
                         QPointF(self.floor_x(xr, z_near), self.floor_y(z_near)),
                         QPointF(self.floor_x(-xr, z_near), self.floor_y(z_near))])
        tf = QTransform()
        if QTransform.quadToQuad(src, dst, tf):
            p.save()
            p.setClipRect(QRectF(0, y_wall, w, h - y_wall))
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.setTransform(tf)
            p.fillRect(src.boundingRect(), QBrush(tile))
            p.restore()
        shade = QLinearGradient(0, y_wall, 0, h)
        shade.setColorAt(0.0, QColor(5, 2, 15, 235))
        shade.setColorAt(0.3, QColor(5, 2, 15, 120))
        shade.setColorAt(1.0, QColor(5, 2, 15, 60))
        p.fillRect(QRectF(0, y_wall, w, h - y_wall), shade)
        p.fillRect(QRectF(0, y_wall - 3 * self.unit, w, 6 * self.unit), QColor("#0a0614"))  # the skirting board
        for k in range(5):
            x = w * (0.1 + 0.2 * k)
            p.setBrush(QColor("#1d1b26"))
            p.setPen(QPen(QColor("#3a3748"), 1))
            p.drawRoundedRect(QRectF(x - 14 * self.unit, -4, 28 * self.unit, 22 * self.unit), 4, 4)

    CARPET_BEAMS = ("#ffe14d", "#19e3ff", "#ff2fa0", "#8a5cff", "#38ff8a")

    @staticmethod
    def _beam(p: QPainter, origin: QPointF, target: QPointF, half_top: float, half_end: float, color: QColor,
              alphas: tuple[int, ...]) -> None:
        """A light beam as a few solid trapezoids fading along its length (gradients and big transformed pixmaps
        are slow)."""
        dx, dy = target.x() - origin.x(), target.y() - origin.y()
        n = math.hypot(dx, dy) or 1.0
        px, py = -dy / n, dx / n
        steps = len(alphas)
        c = QColor(color)
        for i, a in enumerate(alphas):
            k0, k1 = i / steps, (i + 1) / steps
            h0 = half_top + (half_end - half_top) * k0
            h1 = half_top + (half_end - half_top) * k1
            c0 = QPointF(origin.x() + dx * k0, origin.y() + dy * k0)
            c1 = QPointF(origin.x() + dx * k1, origin.y() + dy * k1)
            c.setAlpha(a)
            p.setBrush(c)
            p.drawPolygon(QPolygonF([QPointF(c0.x() + px * h0, c0.y() + py * h0),
                                     QPointF(c1.x() + px * h1, c1.y() + py * h1),
                                     QPointF(c1.x() - px * h1, c1.y() - py * h1),
                                     QPointF(c0.x() - px * h0, c0.y() - py * h0)]))

    def _dynamic_carpet(self, p: QPainter, t: float) -> None:
        w, h = self.w, self.h
        top = 16 * self.unit
        spread = w * 0.07
        aa = p.testRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(Qt.PenStyle.NoPen)
        for k, color in enumerate(self.CARPET_BEAMS):
            x = w * (0.1 + 0.2 * k)
            swing = math.sin(t * (0.35 + 0.07 * k) + k * 1.7)
            target = QPointF(x + swing * w * 0.18, h * 0.97)
            c = QColor(color)
            self._beam(p, QPointF(x, top), target, 5 * self.unit, spread, c, (34, 24, 16, 10))
            c.setAlpha(40)
            p.setBrush(c)
            p.drawEllipse(target, spread * 1.1, spread * 0.25)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, aa)
        self._motes(p, t, 22, "gold", speed=0.02, top=0.1, drift=0.05, size=0.6)
    # --- cyber city -----------------------------------------------------------------------------------------------

    def _skyline(self, p: QPainter, base: float, rng: np.random.Generator, color: QColor, hmin: float, hmax: float,
                 wmin: float, wmax: float, windows: float, edge: QColor | None) -> list[QPointF]:
        x, antennas = -10.0, []
        win_colors = [QColor(c) for c in ("#ffd84d", "#29f3ff", "#ff3fb4", "#ffffff")]
        while x < self.w + 10:
            bw = float(rng.uniform(wmin, wmax)) * self.w
            bh = float(rng.uniform(hmin, hmax)) * self.h
            rect = QRectF(x, base - bh, bw, bh + 2)
            p.fillRect(rect, color)
            if edge is not None and rng.uniform() < 0.45:
                p.setPen(QPen(edge, 1.2 * self.unit))
                p.drawLine(rect.topLeft(), rect.topRight())
            ws = max(4.0 * self.unit, 2.5)
            cols = int(bw / (ws * 2.2))
            rows = int(bh / (ws * 2.6))
            wc = win_colors[int(rng.integers(len(win_colors)))]
            for r in range(rows):
                for c in range(cols):
                    if rng.uniform() < windows:
                        col = QColor(wc)
                        col.setAlpha(int(rng.uniform(90, 230)))
                        p.fillRect(QRectF(x + ws * 0.8 + c * ws * 2.2, base - bh + ws + r * ws * 2.6, ws, ws * 1.2),
                                   col)
            if rng.uniform() < 0.3:
                antennas.append(QPointF(x + bw / 2, base - bh - 14 * self.unit))
                p.setPen(QPen(color.lighter(130), 1.5 * self.unit))
                p.drawLine(QPointF(x + bw / 2, base - bh), QPointF(x + bw / 2, base - bh - 14 * self.unit))
            x += bw + float(rng.uniform(0, 0.01)) * self.w
        return antennas

    def _static_cyber(self, p: QPainter) -> None:
        w, h, hy = self.w, self.h, self.horizon
        upper = QPixmap(w, max(int(math.ceil(hy)), 1))
        q = QPainter(upper)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._cyber_city(q)
        q.end()
        p.drawPixmap(0, 0, upper)
        # a wet street that mirrors the city
        ground = QLinearGradient(0, hy, 0, h)
        ground.setColorAt(0.0, QColor("#1a0b33"))
        ground.setColorAt(1.0, QColor("#03020a"))
        p.fillRect(QRectF(0, hy, w, h - hy), ground)
        p.save()
        p.setOpacity(0.3)
        p.translate(0, hy)
        p.scale(1, -0.9)
        p.drawPixmap(QPointF(0, -upper.height()), upper)
        p.restore()
        fade = QLinearGradient(0, hy, 0, h)
        fade.setColorAt(0.0, QColor(3, 2, 10, 40))
        fade.setColorAt(0.6, QColor(3, 2, 10, 210))
        fade.setColorAt(1.0, QColor(3, 2, 10, 255))
        p.fillRect(QRectF(0, hy, w, h - hy), fade)
        self._vertical_grid(p, QColor(41, 243, 255, 70), 1.4, 10)

    def _cyber_city(self, p: QPainter) -> None:
        w, h, hy = self.w, self.h, self.horizon
        sky = QLinearGradient(0, 0, 0, hy)
        sky.setColorAt(0.0, QColor("#030112"))
        sky.setColorAt(0.5, QColor("#180a3a"))
        sky.setColorAt(1.0, QColor("#5a1660"))
        p.fillRect(QRectF(0, 0, w, hy), sky)
        self._stars_static(p, 50, 0, hy * 0.5, seed=8)
        moon = QPointF(w * 0.73, hy * 0.33)
        r = h * 0.11
        g = QRadialGradient(moon, r * 2.6)
        g.setColorAt(0.0, QColor(255, 60, 200, 110))
        g.setColorAt(1.0, QColor(255, 60, 200, 0))
        p.fillRect(QRectF(0, 0, w, hy), g)
        mg = QLinearGradient(0, moon.y() - r, 0, moon.y() + r)
        mg.setColorAt(0.0, QColor("#ffd1f3"))
        mg.setColorAt(1.0, QColor("#ff4fb8"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(mg)
        p.drawEllipse(moon, r, r)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(41, 243, 255, 150), 2 * self.unit))
        p.drawEllipse(moon, r * 1.9, r * 0.45)
        rng = np.random.default_rng(5)
        self._skyline(p, hy, rng, QColor("#1b0d3a"), 0.12, 0.3, 0.025, 0.06, 0.12, None)
        self.antennas = self._skyline(p, hy, rng, QColor("#0b0620"), 0.06, 0.2, 0.04, 0.09, 0.22,
                                      QColor(41, 243, 255, 200))
        for i, (text, col) in enumerate((("ARCADE", "#29f3ff"), ("NOODLES", "#ff3fb4"), ("24H", "#ffd84d"))):
            x = w * (0.08 + 0.38 * i + 0.06 * (i == 2))
            y = hy - h * (0.13 + 0.03 * i)
            px = 2.2 * self.unit
            rect = QRectF(x - 4 * px, y - 3 * px, pixelfont.width(text, px) + 8 * px, 13 * px)
            p.fillRect(rect, QColor(0, 0, 0, 200))
            p.setPen(QPen(QColor(col), 1.5 * self.unit))
            p.drawRect(rect)
            pixelfont.draw(p, text, x, y, px, col, glow=True)

    def _dynamic_cyber(self, p: QPainter, t: float) -> None:
        w, h, hy = self.w, self.h, self.horizon
        for i, a in enumerate(getattr(self, "antennas", [])):
            if _frac(t * 0.8 + i * 0.37) < 0.5:
                p.fillRect(QRectF(a.x() - 2 * self.unit, a.y() - 2 * self.unit, 4 * self.unit, 4 * self.unit),
                           QColor(255, 40, 60))
        for i in range(4):  # flying cars
            speed = 0.05 + 0.03 * _hash(i, 9)
            d = 1 if i % 2 else -1
            x = _frac(t * speed + _hash(i, 3)) * (w * 1.4) - w * 0.2
            x = x if d > 0 else w - x
            y = hy * (0.25 + 0.5 * _hash(i, 4))
            c = QColor("#ff3fb4") if i % 2 else QColor("#29f3ff")
            g = QLinearGradient(QPointF(x - d * 60 * self.unit, y), QPointF(x, y))
            c0 = QColor(c)
            c0.setAlpha(0)
            g.setColorAt(0.0, c0)
            g.setColorAt(1.0, c)
            p.setPen(QPen(QBrush(g), 2 * self.unit))
            p.drawLine(QPointF(x - d * 60 * self.unit, y), QPointF(x, y))
            p.fillRect(QRectF(x - 2 * self.unit, y - 1.5 * self.unit, 4 * self.unit, 3 * self.unit), QColor("white"))
        self._horizontal_grid(p, t, QColor(41, 243, 255, 60), 0.5)
        pen = QPen(QColor(160, 200, 255, 60), max(1.0 * self.unit, 1.0))
        p.setPen(pen)
        for i in range(70):  # rain
            x = _hash(i, 1) * w
            yk = _frac(_hash(i, 2) + t * (1.3 + 0.6 * _hash(i, 3)))
            y = yk * h * 1.1 - h * 0.05
            p.drawLine(QPointF(x, y), QPointF(x - 4 * self.unit, y + 18 * self.unit))

    # --- spooky night ---------------------------------------------------------------------------------------------

    def _tree(self, p: QPainter, x: float, y: float, length: float, angle: float, depth: int,
              rng: np.random.Generator) -> None:
        if depth == 0 or length < 2:
            return
        x2, y2 = x + math.cos(angle) * length, y - math.sin(angle) * length
        p.setPen(QPen(QColor("#07040c"), max(depth * 1.3 * self.unit, 1.0), Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(x, y), QPointF(x2, y2))
        for _ in range(2):
            self._tree(p, x2, y2, length * float(rng.uniform(0.6, 0.78)), angle + float(rng.uniform(-0.8, 0.8)),
                       depth - 1, rng)

    def _static_spooky(self, p: QPainter) -> None:
        w, h, hy = self.w, self.h, self.horizon
        sky = QLinearGradient(0, 0, 0, hy)
        sky.setColorAt(0.0, QColor("#030208"))
        sky.setColorAt(0.6, QColor("#15102a"))
        sky.setColorAt(1.0, QColor("#2c1d3a"))
        p.fillRect(QRectF(0, 0, w, hy), sky)
        self._stars_static(p, 70, 0, hy * 0.7, seed=13)
        moon = QPointF(w * 0.5, hy * 0.62)
        r = h * 0.19
        g = QRadialGradient(moon, r * 2.4)
        g.setColorAt(0.0, QColor(220, 255, 180, 90))
        g.setColorAt(1.0, QColor(220, 255, 180, 0))
        p.fillRect(QRectF(0, 0, w, hy), g)
        p.setPen(Qt.PenStyle.NoPen)
        mg = QRadialGradient(moon - QPointF(r * 0.3, r * 0.3), r * 1.4)
        mg.setColorAt(0.0, QColor("#fbffe0"))
        mg.setColorAt(1.0, QColor("#c9d99a"))
        p.setBrush(mg)
        p.drawEllipse(moon, r, r)
        rng = np.random.default_rng(21)
        p.setBrush(QColor(150, 170, 110, 70))
        for _ in range(7):
            a, d = rng.uniform(0, 6.28), rng.uniform(0, r * 0.7)
            cr = rng.uniform(0.06, 0.16) * r
            p.drawEllipse(moon + QPointF(math.cos(a) * d, math.sin(a) * d), cr, cr)
        p.setBrush(QColor(20, 14, 34, 200))  # clouds over the moon
        for _ in range(9):
            cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.1, 0.55) * hy
            p.drawEllipse(QPointF(cx, cy), rng.uniform(0.05, 0.12) * w, rng.uniform(0.01, 0.025) * h)
        # hills with tombstones, crosses, dead trees and the haunted house
        hill = QPainterPath(QPointF(0, hy + 2))
        for k in range(13):
            hill.lineTo(QPointF(w * k / 12, hy - h * (0.03 + 0.04 * math.sin(k * 1.3) ** 2)))
        hill.lineTo(QPointF(w, hy + 2))
        hill.closeSubpath()
        p.fillPath(hill, QColor("#0a0612"))
        house = QPointF(w * 0.86, hy - h * 0.06)
        s = h / 780
        p.setBrush(QColor("#07040c"))
        p.drawRect(QRectF(house.x() - 40 * s, house.y() - 50 * s, 80 * s, 52 * s))
        p.drawPolygon(QPolygonF([house + QPointF(-50 * s, -48 * s), house + QPointF(0, -95 * s),
                                 house + QPointF(50 * s, -48 * s)]))
        p.drawRect(QRectF(house.x() + 18 * s, house.y() - 110 * s, 14 * s, 40 * s))
        for dx, dy in ((-25, -38), (8, -38), (-8, -70)):
            p.fillRect(QRectF(house.x() + dx * s, house.y() + dy * s, 12 * s, 14 * s), QColor("#ffb43a"))
        for _ in range(12):
            x = float(rng.uniform(0.02, 0.98)) * w
            y = hy - h * 0.02 + float(rng.uniform(0, 0.02)) * h
            if rng.uniform() < 0.5:
                p.drawRoundedRect(QRectF(x, y - 18 * s, 12 * s, 20 * s), 6 * s, 6 * s)
            else:
                p.fillRect(QRectF(x + 4 * s, y - 22 * s, 4 * s, 24 * s), QColor("#07040c"))
                p.fillRect(QRectF(x - 1 * s, y - 15 * s, 14 * s, 4 * s), QColor("#07040c"))
        for x in (0.07, 0.3, 0.68):
            self._tree(p, w * x, hy, h * 0.08, math.pi / 2, 6, rng)
        ground = QLinearGradient(0, hy, 0, h)
        ground.setColorAt(0.0, QColor("#140d1f"))
        ground.setColorAt(1.0, QColor("#040208"))
        p.fillRect(QRectF(0, hy, w, h - hy), ground)
        p.setPen(QPen(QColor(90, 140, 60, 60), 1))
        for _ in range(240):  # grass
            x, y = float(rng.uniform(0, w)), float(rng.uniform(hy + 4, h))
            k = (y - hy) / (h - hy)
            p.drawLine(QPointF(x, y), QPointF(x + rng.uniform(-2, 2) * k, y - (3 + 8 * k) * self.unit))

    def _make_fog(self) -> QPixmap:
        pm = QPixmap(self.w, int(self.h * 0.3))
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rng = np.random.default_rng(3)
        for _ in range(26):
            c = QPointF(float(rng.uniform(0, self.w)), float(rng.uniform(0.3, 0.8)) * pm.height())
            rx = float(rng.uniform(0.08, 0.2)) * self.w
            g = QRadialGradient(c, rx)
            g.setColorAt(0.0, QColor(190, 180, 220, 40))
            g.setColorAt(1.0, QColor(190, 180, 220, 0))
            for dx in (-self.w, 0, self.w):
                p.save()
                p.translate(dx, 0)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(g)
                p.drawEllipse(c, rx, rx * 0.3)
                p.restore()
        p.end()
        return pm

    def _dynamic_spooky(self, p: QPainter, t: float) -> None:
        w, h, hy = self.w, self.h, self.horizon
        self._twinkle(p, t)
        for i in range(5):  # bats
            k = _frac(t * (0.04 + 0.015 * i) + i * 0.23)
            x = k * w * 1.3 - w * 0.15
            y = hy * (0.2 + 0.4 * _hash(i, 8)) + math.sin(t * 2 + i) * h * 0.03
            flap = math.sin(t * 14 + i * 2)
            s = (9 + 4 * _hash(i, 2)) * self.unit
            bat = QPainterPath(QPointF(x - s, y - flap * s * 0.6))
            bat.quadTo(QPointF(x - s * 0.5, y + s * 0.1), QPointF(x, y))
            bat.quadTo(QPointF(x + s * 0.5, y + s * 0.1), QPointF(x + s, y - flap * s * 0.6))
            bat.quadTo(QPointF(x + s * 0.5, y + s * 0.45), QPointF(x, y + s * 0.3))
            bat.quadTo(QPointF(x - s * 0.5, y + s * 0.45), QPointF(x - s, y - flap * s * 0.6))
            p.fillPath(bat, QColor("#050208"))
        if self._fog is not None:
            for layer, (speed, y0) in enumerate(((0.012, hy - h * 0.06), (-0.02, h * 0.68))):
                off = _frac(t * speed + layer * 0.4) * w
                p.setOpacity(0.9)
                p.drawPixmap(QPointF(-off, y0), self._fog)
                p.drawPixmap(QPointF(w - off, y0), self._fog)
            p.setOpacity(1.0)
        self._motes(p, t, 14, "green", speed=0.015, top=0.45, drift=0.06, size=0.55)
        n = math.floor(t / 11.0)
        local = t - n * 11.0
        if _hash(n, 77) < 0.5 and local < 0.35:  # lightning
            a = (1 - local / 0.35) * (0.6 if _frac(local * 9) < 0.5 else 0.25)
            p.fillRect(QRectF(0, 0, w, hy), QColor(220, 230, 255, int(110 * a)))

    # --- every frame ----------------------------------------------------------------------------------------------

    def paint(self, p: QPainter, t: float, party: float = 0.0) -> None:
        p.drawPixmap(0, 0, self.static)
        getattr(self, f"_dynamic_{self.theme}")(p, t)
        for sign, pos in zip(self.signs, self.sign_positions(), strict=False):
            sign.paint(p, pos, t + (5.0 if party else 0.0))
        if party > 0:
            self.paint_party(p, t, party)

    def paint_party(self, p: QPainter, t: float, strength: float) -> None:
        """Disco lights: coloured beams swinging below a mirror ball."""
        w, h = self.w, self.h
        ball = QPointF(w / 2, h * 0.02)
        aa = p.testRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(Qt.PenStyle.NoPen)
        length = math.hypot(w, h) * 0.9
        for k in range(8):
            ang = math.radians(78 * math.sin(t * 0.9 + k * 0.785))
            target = QPointF(ball.x() - math.sin(ang) * length, ball.y() + math.cos(ang) * length)
            c = QColor.fromHsvF((k / 8 + t * 0.1) % 1.0, 0.9, 1.0)
            self._beam(p, ball, target, 2.0, w * 0.06, c, tuple(int(a * strength) for a in (40, 26, 14)))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, aa)
        r = 16 * self.unit
        g = QRadialGradient(ball + QPointF(-r * 0.3, r * 0.6), r * 1.2)
        g.setColorAt(0.0, QColor("white"))
        g.setColorAt(1.0, QColor("#6b6b8a"))
        p.setBrush(g)
        p.drawEllipse(QPointF(ball.x(), ball.y() + r), r, r)
        p.setPen(QPen(QColor(0, 0, 0, 90), 1))
        for k in range(-2, 3):
            p.drawLine(QPointF(ball.x() - r, ball.y() + r + k * r * 0.35), QPointF(ball.x() + r, ball.y() + r + k * r *
                                                                                      0.35))
        p.setPen(QPen(QColor("#888"), 2))
        p.drawLine(QPointF(ball.x(), 0), QPointF(ball.x(), ball.y()))


def theme_swatch(theme: str, w: int, h: int) -> QPixmap:
    """A small picture of a theme (the prize counter)."""
    scene = Scene(theme, w * 4, h * 4, h * 4 * 0.55, (h * 4 * 0.45) * 5, h * 4 * 1.2)
    pm = QPixmap(w * 4, h * 4)
    p = QPainter(pm)
    scene.paint(p, 1.0)
    p.end()
    return pm.scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)


__all__ = ["THEMES", "NeonSign", "Scene", "neon_text_pixmap", "theme_swatch"]
