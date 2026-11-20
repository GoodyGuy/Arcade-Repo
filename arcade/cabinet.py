"""The arcade cabinets of the home screen's carousel.

``CabinetArt`` draws one game's cabinet once (front: a backlit marquee with the title, the screen bezel, a control panel
with a joystick and buttons, the coin door; side: the side art in the game's colour) into pixmaps. ``Carousel`` places
the cabinets in 3D - the chosen one big in the middle, the others turned and shrinking towards the sides - and gives
each one's perspective transforms. ``ScreenFeed`` keeps each cabinet's live screen: the game's ``paint_thumbnail`` drawn
into a pixmap (a thumbnail that raises shows an OUT OF ORDER screen; a slow one is refreshed less often).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, \
    QPolygonF, QRadialGradient, QTransform

from app.arcade import pixelfont

log = logging.getLogger(__name__)

ASPECT = 1.62  # cabinet height / width
DEPTH = 0.62  # cabinet depth / width
SIDE_X = 0.44  # the side panels, from the centre line (the control panel sticks out a little further)
PAD = 0.06  # glow margin around the front pixmap (of the width)

# fractions of the cabinet height (y) and width (x)
MARQUEE = (0.05, 0.012, 0.95, 0.118)
SCREEN = (0.14, 0.178, 0.86, 0.505)
COIN_SLOTS = ((0.355, 0.725, 0.455, 0.79), (0.545, 0.725, 0.645, 0.79))

TRIMS = {"chrome": ("#f2f6ff", "#8d99ae"), "gold": ("#fff1a8", "#c8901e"), "rainbow": ("#ffffff", "#c9c9d9")}


def _mix(a: QColor, b: QColor, k: float) -> QColor:
    return QColor(int(a.red() + (b.red() - a.red()) * k), int(a.green() + (b.green() - a.green()) * k),
                  int(a.blue() + (b.blue() - a.blue()) * k))


def front_outline(u: float, h: float) -> QPolygonF:
    """The cabinet's front silhouette in pixmap coordinates (u wide, h high)."""
    pts = [(0.03, 0.0), (0.97, 0.0), (0.97, 0.125), (0.94, 0.125), (0.94, 0.53), (0.965, 0.53), (1.0, 0.615),
           (1.0, 0.64), (0.94, 0.64), (0.94, 0.975), (0.92, 0.975), (0.92, 1.0), (0.08, 1.0), (0.08, 0.975),
           (0.06, 0.975), (0.06, 0.64), (0.0, 0.64), (0.0, 0.615), (0.035, 0.53), (0.06, 0.53), (0.06, 0.125),
           (0.03, 0.125)]
    return QPolygonF([QPointF(x * u, y * h) for x, y in pts])


def side_outline(d: float, h: float) -> QPolygonF:
    """The side panel's profile (d deep, h high; x = 0 is the front)."""
    pts = [(0.0, 0.0), (0.7, 0.0), (1.0, 0.07), (1.0, 1.0), (0.06, 1.0), (0.04, 0.975), (0.04, 0.64), (0.0, 0.64),
           (0.0, 0.53), (0.16, 0.525), (0.1, 0.16), (0.03, 0.155), (0.03, 0.125), (0.0, 0.125)]
    return QPolygonF([QPointF(x * d, y * h) for x, y in pts])


class CabinetArt:
    """One game's cabinet drawn into pixmaps ``width`` pixels wide (see the module docstring)."""

    def __init__(self, cls, width: int, trim: str = "neon", dpr: float = 1.0) -> None:  # noqa: ANN001
        self.cls = cls
        self.u = max(int(width), 40)
        self.h = int(self.u * ASPECT)
        self.pad = int(self.u * PAD)
        self.trim = trim
        self.dpr = dpr
        self.accent = QColor(getattr(cls, "accent", "#ff4081") or "#ff4081")
        if not self.accent.isValid():
            self.accent = QColor("#ff4081")
        if trim in TRIMS:
            self.trim_hi, self.trim_lo = (QColor(c) for c in TRIMS[trim])
        else:
            self.trim_hi, self.trim_lo = self.accent.lighter(150), self.accent
        self.front = self._front()
        self.side = self._side(False)  # seen from the left (a cabinet on the left shows its right side)
        self.side_mirror = self._side(True)
        d = DEPTH * self.u
        self.side_outline = side_outline(d, self.h)
        self.side_mirror_outline = QTransform(-1, 0, 0, 1, d, 0).map(self.side_outline)
        self.reflection = self._reflection()
        u, h = self.u, self.h
        self.screen_rect = QRectF(SCREEN[0] * u, SCREEN[1] * h, (SCREEN[2] - SCREEN[0]) * u,
                                  (SCREEN[3] - SCREEN[1]) * h)
        self.marquee_rect = QRectF(MARQUEE[0] * u, MARQUEE[1] * h, (MARQUEE[2] - MARQUEE[0]) * u,
                                   (MARQUEE[3] - MARQUEE[1]) * h)
        self.coin_rects = [QRectF(a * u, b * h, (c - a) * u, (d - b) * h) for a, b, c, d in COIN_SLOTS]
        self.outline = front_outline(u, h)
        self.glass = self._glass()

    # --- pixmaps --------------------------------------------------------------------------------------------------

    def _pixmap(self, w: float, h: float) -> QPixmap:
        pm = QPixmap(max(int(w * self.dpr), 1), max(int(h * self.dpr), 1))
        pm.setDevicePixelRatio(self.dpr)
        pm.fill(Qt.GlobalColor.transparent)
        return pm

    def _title_path(self, rect: QRectF, family: str = "Impact") -> QPainterPath:
        title = str(getattr(self.cls, "title", "GAME") or "GAME").upper()
        f = QFont(family)
        f.setPixelSize(100)
        path = QPainterPath()
        path.addText(0, 0, f, title)
        box = path.boundingRect()
        if box.width() <= 0 or box.height() <= 0:
            return QPainterPath()
        k = min(rect.width() / box.width(), rect.height() / box.height())
        tf = QTransform()
        tf.translate(rect.center().x() - box.center().x() * k, rect.center().y() - box.center().y() * k)
        tf.scale(k, k)
        return tf.map(path)

    def _front(self) -> QPixmap:
        u, h, pad = self.u, self.h, self.pad
        pm = self._pixmap(u + 2 * pad, h + 2 * pad)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(pad, pad)
        accent = self.accent
        body_dark = _mix(QColor("#07050c"), accent, 0.06)
        body_lit = _mix(QColor("#17122a"), accent, 0.13)
        outline = front_outline(u, h)
        # glow around the cabinet in the trim colour
        for k, a in ((pad * 0.9, 20), (pad * 0.55, 34), (pad * 0.3, 60)):
            c = QColor(self.trim_lo)
            c.setAlpha(a)
            p.setPen(QPen(c, k * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(outline)
        body = QLinearGradient(0, 0, u, 0)
        body.setColorAt(0.0, body_dark)
        body.setColorAt(0.18, body_lit)
        body.setColorAt(0.5, _mix(body_lit, QColor("#000"), 0.15))
        body.setColorAt(0.82, body_lit)
        body.setColorAt(1.0, body_dark)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(body)
        p.drawPolygon(outline)
        # marquee: backlit panel with the title
        mq = QRectF(MARQUEE[0] * u, MARQUEE[1] * h, (MARQUEE[2] - MARQUEE[0]) * u, (MARQUEE[3] - MARQUEE[1]) * h)
        p.fillRect(mq.adjusted(-u * 0.012, -u * 0.012, u * 0.012, u * 0.012), QColor("#050308"))
        g = QLinearGradient(mq.topLeft(), mq.bottomLeft())
        g.setColorAt(0.0, accent.lighter(170))
        g.setColorAt(0.5, accent)
        g.setColorAt(1.0, accent.darker(170))
        p.fillRect(mq, g)
        rg = QRadialGradient(mq.center(), mq.width() * 0.55)
        rg.setColorAt(0.0, QColor(255, 255, 255, 110))
        rg.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillRect(mq, rg)
        p.save()
        p.setClipRect(mq)
        p.setPen(QPen(QColor(0, 0, 0, 40), max(u * 0.012, 1)))
        for k in range(-12, 12):  # sunburst stripes
            x = mq.center().x() + k * u * 0.07
            p.drawLine(QPointF(mq.center().x(), mq.bottom() + mq.height()), QPointF(x, mq.top()))
        title = self._title_path(mq.adjusted(u * 0.06, mq.height() * 0.16, -u * 0.06, -mq.height() * 0.16))
        for wdt, a in ((u * 0.05, 60), (u * 0.03, 120)):
            p.setPen(QPen(QColor(0, 0, 0, a), wdt, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin))
            p.drawPath(title)
        p.setPen(QPen(accent.darker(250), u * 0.012))
        tg = QLinearGradient(0, title.boundingRect().top(), 0, title.boundingRect().bottom())
        tg.setColorAt(0.0, QColor("#ffffff"))
        tg.setColorAt(0.55, QColor("#fff6d0"))
        tg.setColorAt(0.56, accent.lighter(130))
        tg.setColorAt(1.0, QColor("#ffffff"))
        p.setBrush(tg)
        p.drawPath(title)
        p.restore()
        p.setPen(QPen(self.trim_hi, max(u * 0.008, 1)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(mq)
        # speaker grilles
        for x0 in (0.16, 0.6):
            p.fillRect(QRectF(x0 * u, 0.132 * h, 0.24 * u, 0.02 * h), QColor("#07050c"))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 28))
            for i in range(9):
                p.drawEllipse(QPointF((x0 + 0.015 + i * 0.026) * u, 0.142 * h), u * 0.006, u * 0.006)
        # bezel and screen
        bez = QRectF(0.09 * u, 0.16 * h, 0.82 * u, 0.365 * h)
        bg = QLinearGradient(bez.topLeft(), bez.bottomLeft())
        bg.setColorAt(0.0, QColor("#16121f"))
        bg.setColorAt(1.0, QColor("#07050c"))
        p.setBrush(bg)
        p.setPen(QPen(QColor(255, 255, 255, 30), 1))
        p.drawRoundedRect(bez, u * 0.03, u * 0.03)
        p.setPen(QPen(_mix(accent, QColor("#ffffff"), 0.3), max(u * 0.006, 1)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(bez.adjusted(u * 0.018, u * 0.018, -u * 0.018, -u * 0.018), u * 0.025, u * 0.025)
        scr = QRectF(SCREEN[0] * u, SCREEN[1] * h, (SCREEN[2] - SCREEN[0]) * u, (SCREEN[3] - SCREEN[1]) * h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#000000"))
        p.drawRoundedRect(scr.adjusted(-u * 0.012, -u * 0.012, u * 0.012, u * 0.012), u * 0.03, u * 0.03)
        p.fillRect(scr, QColor("#020306"))
        # control panel
        top = QPolygonF([QPointF(0.035 * u, 0.53 * h), QPointF(0.965 * u, 0.53 * h), QPointF(u, 0.615 * h),
                         QPointF(0, 0.615 * h)])
        cg = QLinearGradient(0, 0.53 * h, 0, 0.615 * h)
        cg.setColorAt(0.0, _mix(QColor("#1d1830"), accent, 0.35))
        cg.setColorAt(1.0, _mix(QColor("#0d0a16"), accent, 0.25))
        p.setBrush(cg)
        p.drawPolygon(top)
        p.setPen(QPen(QColor(255, 255, 255, 50), max(u * 0.004, 1)))
        p.drawLine(QPointF(0.035 * u, 0.53 * h), QPointF(0.965 * u, 0.53 * h))
        lip = QRectF(0, 0.615 * h, u, 0.025 * h)
        lg = QLinearGradient(lip.topLeft(), lip.bottomLeft())
        lg.setColorAt(0.0, self.trim_hi)
        lg.setColorAt(1.0, self.trim_lo.darker(160))
        p.fillRect(lip, lg)
        self._joystick(p, QPointF(0.25 * u, 0.582 * h), u)
        for i, color in enumerate((accent.lighter(120), QColor("#ffd84d"), QColor("#29f3ff"))):
            self._button(p, QPointF((0.5 + 0.13 * i) * u, (0.583 - 0.006 * i) * h), u * 0.04, color)
        for i, color in enumerate((QColor("#f5f5f5"), QColor("#f5f5f5"))):
            r = QRectF((0.43 + 0.1 * i) * u, 0.541 * h, 0.06 * u, 0.012 * h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(r, 2, 2)
        # lower body art: accent stripes either side of the coin door
        for x0, x1, sgn in ((0.075, 0.29, 1), (0.71, 0.925, -1)):
            area = QRectF(x0 * u, 0.665 * h, (x1 - x0) * u, 0.255 * h)
            p.save()
            p.setClipRect(area)
            p.setPen(Qt.PenStyle.NoPen)
            for i, (c, a) in enumerate(((accent, 150), (accent.lighter(150), 110), (QColor("#ffffff"), 60))):
                col = QColor(c)
                col.setAlpha(a)
                p.setBrush(col)
                y0 = area.top() + area.height() * (0.25 + 0.2 * i)
                band = QPolygonF([QPointF(area.left() - u * 0.2, y0 + sgn * area.width() * 0.6),
                                  QPointF(area.right() + u * 0.2, y0 - sgn * area.width() * 0.6),
                                  QPointF(area.right() + u * 0.2, y0 - sgn * area.width() * 0.6 + h * 0.022),
                                  QPointF(area.left() - u * 0.2, y0 + sgn * area.width() * 0.6 + h * 0.022)])
                p.drawPolygon(band)
            p.restore()
        # coin door
        door = QRectF(0.3 * u, 0.69 * h, 0.4 * u, 0.21 * h)
        dg = QLinearGradient(door.topLeft(), door.topRight())
        dg.setColorAt(0.0, QColor("#3b3946"))
        dg.setColorAt(0.5, QColor("#8a8798"))
        dg.setColorAt(1.0, QColor("#3b3946"))
        p.setBrush(dg)
        p.setPen(QPen(QColor("#111"), max(u * 0.006, 1)))
        p.drawRoundedRect(door, u * 0.015, u * 0.015)
        for r in [QRectF(a * u, b * h, (c - a) * u, (d - b) * h) for a, b, c, d in COIN_SLOTS]:
            p.setPen(QPen(QColor("#1a1a1a"), max(u * 0.004, 1)))
            lg = QLinearGradient(r.topLeft(), r.bottomLeft())
            lg.setColorAt(0.0, QColor("#ffb347"))
            lg.setColorAt(1.0, QColor("#ff4d1a"))
            p.setBrush(lg)
            p.drawRoundedRect(r, u * 0.008, u * 0.008)
            p.fillRect(QRectF(r.center().x() - u * 0.005, r.top() + r.height() * 0.15, u * 0.01, r.height() * 0.5),
                       QColor("#1a0500"))
            px = max(min(r.width() / 20, 0.028 * h / 7), 0.5)
            pixelfont.draw(p, "25¢", r.center().x(), r.bottom() + 0.006 * h, px, "#1a1a22",
                           Qt.AlignmentFlag.AlignHCenter)
        ret = QRectF(0.43 * u, 0.83 * h, 0.14 * u, 0.045 * h)
        p.setBrush(QColor("#111016"))
        p.setPen(QPen(QColor("#5b5966"), max(u * 0.004, 1)))
        p.drawRoundedRect(ret, u * 0.008, u * 0.008)
        # kick plate
        kick = QRectF(0.06 * u, 0.935 * h, 0.88 * u, 0.04 * h)
        p.fillRect(kick, QColor("#060409"))
        p.fillRect(QRectF(0.06 * u, 0.935 * h, 0.88 * u, max(u * 0.006, 1)), self.trim_lo)
        p.fillRect(QRectF(0.08 * u, 0.975 * h, 0.84 * u, 0.025 * h), QColor("#020103"))
        # T-moulding along the sides of the body
        p.setPen(QPen(self.trim_hi, max(u * 0.012, 1.2)))
        p.drawLine(QPointF(0.06 * u, 0.125 * h), QPointF(0.06 * u, 0.53 * h))
        p.drawLine(QPointF(0.94 * u, 0.125 * h), QPointF(0.94 * u, 0.53 * h))
        p.drawLine(QPointF(0.06 * u, 0.64 * h), QPointF(0.06 * u, 0.975 * h))
        p.drawLine(QPointF(0.94 * u, 0.64 * h), QPointF(0.94 * u, 0.975 * h))
        p.end()
        return pm

    @staticmethod
    def _joystick(p: QPainter, c: QPointF, u: float) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawEllipse(c, u * 0.07, u * 0.022)
        p.setBrush(QColor("#222"))
        p.drawEllipse(c, u * 0.05, u * 0.016)
        p.setPen(QPen(QColor("#bfc3cc"), u * 0.016, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        top = QPointF(c.x(), c.y() - u * 0.07)
        p.drawLine(c, top)
        g = QRadialGradient(top + QPointF(-u * 0.015, -u * 0.015), u * 0.05)
        g.setColorAt(0.0, QColor("#ffb3b3"))
        g.setColorAt(0.4, QColor("#ff2a2a"))
        g.setColorAt(1.0, QColor("#7a0000"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawEllipse(top, u * 0.042, u * 0.042)

    @staticmethod
    def _button(p: QPainter, c: QPointF, r: float, color: QColor) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawEllipse(c + QPointF(0, r * 0.25), r, r * 0.55)
        p.setBrush(color.darker(170))
        p.drawEllipse(c + QPointF(0, r * 0.12), r * 0.92, r * 0.5)
        g = QRadialGradient(c - QPointF(r * 0.3, r * 0.25), r)
        g.setColorAt(0.0, color.lighter(170))
        g.setColorAt(1.0, color)
        p.setBrush(g)
        p.drawEllipse(c, r * 0.85, r * 0.45)

    def _side(self, mirror: bool) -> QPixmap:
        """The side panel; ``mirror``: its front on the right (the art still reads the right way round)."""
        d, h = DEPTH * self.u, self.h
        pm = self._pixmap(d, h)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        outline = side_outline(d, h)
        if mirror:
            outline = QTransform(-1, 0, 0, 1, d, 0).map(outline)
        path = QPainterPath()
        path.addPolygon(outline)
        path.closeSubpath()
        accent = self.accent
        p.setClipPath(path)
        g = QLinearGradient(0, 0, d, h)
        g.setColorAt(0.0, _mix(QColor("#140f22"), accent, 0.25))
        g.setColorAt(1.0, QColor("#05030a"))
        p.fillRect(QRectF(0, 0, d, h), g)
        p.setPen(Qt.PenStyle.NoPen)
        for i, (k, c) in enumerate(((0.0, accent), (0.09, accent.lighter(140)), (0.15, QColor("#ffffff")))):
            band = QPolygonF([QPointF(-d, h * (0.98 - k)), QPointF(d * 2, h * (0.45 - k)),
                              QPointF(d * 2, h * (0.45 - k + 0.06 - 0.02 * i)),
                              QPointF(-d, h * (0.98 - k + 0.06 - 0.02 * i))])
            col = QColor(c)
            col.setAlpha(210 if i < 2 else 150)
            p.setBrush(col)
            p.drawPolygon(band)
        circle = QPointF(d * (0.45 if mirror else 0.55), h * 0.3)
        r = d * 0.36
        disc = QPainterPath()
        disc.addEllipse(circle, r, r)
        p.save()
        p.setClipPath(disc, Qt.ClipOperation.IntersectClip)
        try:
            self.cls.paint_thumbnail(p, QRectF(circle.x() - r * 1.3, circle.y() - r, r * 2.6, r * 2), 0.0)
        except Exception:  # noqa: BLE001 - a broken thumbnail just leaves the circle plain
            log.debug("side art", exc_info=True)
            p.fillRect(QRectF(circle.x() - r, circle.y() - r, 2 * r, 2 * r), accent.darker(200))
        p.restore()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(accent.lighter(150), d * 0.03))
        p.drawEllipse(circle, r, r)
        title = self._title_path(QRectF(0, 0, h * 0.42, d * 0.2))
        tf = QTransform()
        tf.translate(d * (0.08 if mirror else 0.7), h * 0.93)
        tf.rotate(-90)
        title = tf.map(title)
        p.setPen(QPen(QColor(0, 0, 0, 160), d * 0.04, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.drawPath(title)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("white"))
        p.drawPath(title)
        p.fillRect(QRectF(0, 0, d, h), QColor(0, 0, 0, 70))  # the side is in the shade
        p.setClipping(False)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(self.trim_hi, max(self.u * 0.014, 1.2), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.drawPolygon(outline)
        p.end()
        return pm

    def _reflection(self) -> QPixmap:
        """The front, upside down and fading out: a shine on the floor."""
        src = self.front
        out = QPixmap(src.size())
        out.setDevicePixelRatio(src.devicePixelRatio())
        out.fill(Qt.GlobalColor.transparent)
        p = QPainter(out)
        w, h = src.width() / src.devicePixelRatio(), src.height() / src.devicePixelRatio()
        p.translate(0, h)
        p.scale(1, -1)
        p.drawPixmap(0, 0, src)
        p.resetTransform()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        g = QLinearGradient(0, 0, 0, h * 0.4)
        g.setColorAt(0.0, QColor(0, 0, 0, 90))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(QRectF(0, 0, w, h), g)
        p.end()
        return out

    def _glass(self) -> QPixmap:
        """Over the live screen: scan lines, darker corners and a reflection."""
        r = QRectF(0, 0, (SCREEN[2] - SCREEN[0]) * self.u, (SCREEN[3] - SCREEN[1]) * self.h)
        pm = self._pixmap(r.width(), r.height())
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        step = max(r.height() / 70, 2.0)
        y = 0.0
        while y < r.height():
            p.fillRect(QRectF(0, y, r.width(), step * 0.45), QColor(0, 0, 0, 55))
            y += step
        v = QRadialGradient(r.center(), r.width() * 0.68)
        v.setColorAt(0.6, QColor(0, 0, 0, 0))
        v.setColorAt(1.0, QColor(0, 0, 0, 160))
        p.fillRect(r, v)
        shine = QPainterPath()
        shine.moveTo(r.width() * 0.04, r.height() * 0.04)
        shine.lineTo(r.width() * 0.55, r.height() * 0.04)
        shine.lineTo(r.width() * 0.2, r.height() * 0.5)
        shine.lineTo(r.width() * 0.04, r.height() * 0.5)
        shine.closeSubpath()
        g = QLinearGradient(0, 0, r.width() * 0.3, r.height() * 0.4)
        g.setColorAt(0.0, QColor(255, 255, 255, 46))
        g.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillPath(shine, g)
        p.end()
        return pm


# --- the live screens ----------------------------------------------------------------------------------------------

def _noise_frames(n: int = 4) -> list[QImage]:
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(n):
        v = rng.integers(0, 200, (36, 48), dtype=np.uint8)
        rgb = np.repeat(v[..., None], 3, axis=2)
        rgb = np.ascontiguousarray(rgb)
        frames.append(QImage(rgb.data, 48, 36, rgb.strides[0], QImage.Format.Format_RGB888).copy())
    return frames


class ScreenFeed:
    """A cabinet's live screen: the game's thumbnail, refreshed into a pixmap (see the module docstring)."""

    SLOW_S = 0.012  # a thumbnail slower than this is drawn less often
    _noise: list[QImage] | None = None

    def __init__(self, cls) -> None:  # noqa: ANN001
        self.cls = cls
        self.pixmap: QPixmap | None = None
        self.broken = False
        self.slow = False
        self.last = -1e9
        self.cost = 0.0

    def due(self, now: float, interval: float) -> bool:
        """Time to draw the thumbnail again? A slow one waits about three times its cost (it never takes more than
        about a third of the frames)."""
        if self.broken:
            return False
        return now - self.last >= max(interval, min(self.cost * 3.0, 0.5))

    def refresh(self, w: float, h: float, t: float, now: float, dpr: float = 1.0,
                glass: QPixmap | None = None, dim: float = 0.0) -> None:
        size = (max(int(w * dpr), 2), max(int(h * dpr), 2))
        if self.pixmap is None or (self.pixmap.width(), self.pixmap.height()) != size:
            self.pixmap = QPixmap(*size)
            self.pixmap.setDevicePixelRatio(dpr)
        self.pixmap.fill(QColor("#020306"))
        self.last = now
        start = time.perf_counter()
        p = QPainter(self.pixmap)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setClipRect(QRectF(0, 0, w, h))
            self.cls.paint_thumbnail(p, QRectF(0, 0, w, h), t)
        except Exception:  # noqa: BLE001 - a broken thumbnail shows OUT OF ORDER, never breaks the arcade
            log.warning("Arcade thumbnail of %s failed", getattr(self.cls, "game_id", "?"), exc_info=True)
            self.broken = True
        finally:
            p.end()
        self.cost = 0.7 * self.cost + 0.3 * (time.perf_counter() - start)
        self.slow = self.cost > self.SLOW_S
        if glass is not None and not self.broken:
            p = QPainter(self.pixmap)
            p.drawPixmap(QRectF(0, 0, w, h), glass, QRectF(0, 0, glass.width(), glass.height()))
            if dim > 0.01:
                p.fillRect(QRectF(0, 0, w, h), QColor(4, 2, 12, int(200 * min(dim, 0.8))))
            p.end()

    def paint(self, p: QPainter, rect: QRectF, t: float) -> None:
        if self.broken or self.pixmap is None:
            self.paint_out_of_order(p, rect, t, self.broken)
            return
        p.drawPixmap(rect, self.pixmap, QRectF(0, 0, self.pixmap.width(), self.pixmap.height()))

    @classmethod
    def paint_out_of_order(cls, p: QPainter, rect: QRectF, t: float, broken: bool = True) -> None:
        if cls._noise is None:
            cls._noise = _noise_frames()
        frame = cls._noise[int(t * 20) % len(cls._noise)]
        p.drawImage(rect, frame)
        if broken:
            px = max(rect.width() / 70, 1.0)
            band = QRectF(rect.x(), rect.center().y() - px * 6, rect.width(), px * 12)
            p.fillRect(band, QColor(0, 0, 0, 200))
            pixelfont.draw(p, "OUT OF ORDER", rect.center().x(), band.top() + px * 2.5, px, "#ff4d4d",
                           Qt.AlignmentFlag.AlignHCenter)


# --- placing the cabinets ------------------------------------------------------------------------------------------

@dataclass
class Pose:
    index: int
    k: float  # signed distance from the middle, in cabinets
    depth: float  # for the drawing order (far first)
    front: QTransform  # cabinet pixmap coordinates -> widget
    side: QTransform | None  # side pixmap coordinates -> widget (None: not seen)
    opacity: float
    dim: float
    quad: QPolygonF  # the front's outline on screen (hit test)
    side_quad: QPolygonF | None
    side_mirror: bool = False  # the side pixmap with its front on the right
    ref: Pose | None = None  # sliding: the pose at the nearest whole place, moved by ``affine`` (see ``moved``)
    affine: QTransform | None = None

    def moved(self, target: Pose) -> Pose:
        """This (whole-place) pose scaled and shifted onto ``target``'s place: a cabinet at least one place from the
        middle keeps its turn while it slides, so its picture only grows / shrinks and moves (a cheap affine copy of
        the perspective drawing made once)."""
        a, b = self.quad.boundingRect(), target.quad.boundingRect()
        sx, sy = b.width() / max(a.width(), 1e-6), b.height() / max(a.height(), 1e-6)
        affine = QTransform(sx, 0, 0, sy, b.left() - a.left() * sx, b.top() - a.top() * sy)
        return Pose(target.index, target.k, target.depth, self.front * affine,
                    None if self.side is None else self.side * affine, target.opacity, self.dim,
                    affine.map(self.quad), None if self.side_quad is None else affine.map(self.side_quad),
                    self.side_mirror, self, affine)


class Carousel:
    """The 3D layout of the cabinets for a widget size: see ``layout`` and ``poses``."""

    Z0 = 5.0  # the middle cabinet's distance (cabinet widths)
    EYE = 1.05  # eye height above the floor
    TURN = math.radians(40)

    def __init__(self) -> None:
        self.w = self.h = 1
        self.floor = self.horizon = self.s0 = self.focal = 0.0
        self.vx = 0.0
        self.cab_px = 40

    def layout(self, w: int, h: int, top: float, floor: float) -> None:
        """The middle cabinet stands on ``floor`` with its top at ``top`` (widget y)."""
        self.w, self.h = w, h
        self.floor = floor
        self.s0 = min((floor - top) / ASPECT, w * 0.25)
        self.focal = self.s0 * self.Z0
        self.horizon = floor - self.EYE * self.s0
        self.vx = w / 2
        self.cab_px = max(int(round(self.s0)), 40)

    @property
    def floor_c(self) -> float:
        return self.EYE * self.focal

    def place(self, k: float) -> tuple[float, float, float]:
        """World x, depth and turn of a cabinet ``k`` places from the middle."""
        a, b = min(abs(k), 1.0), max(abs(k) - 1.0, 0.0)
        s = 1.0 if k >= 0 else -1.0
        x = s * (1.38 * a + 0.92 * b)
        z = self.Z0 + 1.25 * a + 0.8 * b
        return x, z, s * self.TURN * a

    def project(self, x: float, y_top: float, z: float) -> QPointF:
        """World point (x across, y down from a cabinet's top, z depth) -> widget."""
        z = max(z, 0.2)
        return QPointF(self.vx + x * self.focal / z, self.horizon + (self.EYE - ASPECT + y_top) * self.focal / z)

    def pose(self, index: int, k: float, art: CabinetArt, lift: float = 0.0, opacity: float = 1.0) -> Pose:
        x0, z0, th = self.place(k)
        c, s = math.cos(th), math.sin(th)

        def world(lx: float, ly: float, lz: float) -> QPointF:  # cabinet-local (widths, from the front centre)
            return self.project(x0 + lx * c - lz * s, ly - lift, z0 + lx * s + lz * c)

        u, hpx = art.u, art.h
        hh = ASPECT
        src = QPolygonF([QPointF(0, 0), QPointF(u, 0), QPointF(u, hpx), QPointF(0, hpx)])
        dst = QPolygonF([world(-0.5, 0, 0), world(0.5, 0, 0), world(0.5, hh, 0), world(-0.5, hh, 0)])
        if abs(th) < 1e-6:  # facing us: a plain scale (Qt's fast path)
            tl, br = dst.at(0), dst.at(2)
            front = QTransform(( br.x() - tl.x()) / u, 0, 0, (br.y() - tl.y()) / hpx, tl.x(), tl.y())
        else:
            front = QTransform()
            QTransform.quadToQuad(src, dst, front)
        side = side_quad = None
        if abs(k) > 0.03:
            sx = -SIDE_X if k > 0 else SIDE_X
            d = DEPTH * u
            ssrc = QPolygonF([QPointF(0, 0), QPointF(d, 0), QPointF(d, hpx), QPointF(0, hpx)])
            if k > 0:  # the left side, seen from the right: its back is on the left
                sdst = QPolygonF([world(sx, 0, DEPTH), world(sx, 0, 0), world(sx, hh, 0), world(sx, hh, DEPTH)])
            else:
                sdst = QPolygonF([world(sx, 0, 0), world(sx, 0, DEPTH), world(sx, hh, DEPTH), world(sx, hh, 0)])
            side = QTransform()
            if not QTransform.quadToQuad(ssrc, sdst, side):
                side = None
            else:
                side_quad = sdst
        dim = min(abs(k), 1.0) * 0.55 + min(max(abs(k) - 1.0, 0.0), 2.0) * 0.12
        return Pose(index, k, z0, front, side, opacity, min(dim, 0.85), front.map(art.outline), side_quad, k > 0)

    def floor_point(self, k: float) -> tuple[QPointF, float]:
        """The middle of a cabinet's base on screen and its width there."""
        x0, z0, _th = self.place(k)
        left = self.project(x0 - 0.5, ASPECT, z0)
        right = self.project(x0 + 0.5, ASPECT, z0)
        return QPointF((left.x() + right.x()) / 2, left.y()), right.x() - left.x()


def _draw_body(p: QPainter, art: CabinetArt, pose: Pose, base: QTransform, reflection: float) -> None:
    """The cabinet's side and front (no screen) with the painter's opacity."""
    opacity = p.opacity()
    if pose.side is not None:
        p.setTransform(pose.side * base)
        p.drawPixmap(QPointF(0, 0), art.side_mirror if pose.side_mirror else art.side)
    p.setTransform(pose.front * base)
    if reflection > 0.02:
        p.setOpacity(opacity * reflection * 0.9)
        p.drawPixmap(QPointF(-art.pad, art.h - art.pad), art.reflection)
        p.setOpacity(opacity)
    p.drawPixmap(QPointF(-art.pad, -art.pad), art.front)
    if pose.dim > 0.01:  # further away, in the shade: a dark film over the cabinet (cheaper than a second pixmap)
        shade = QColor(4, 2, 12, int(255 * 0.85 * min(pose.dim, 0.75)))
        p.setTransform(base)  # polygons are drawn in device space (a perspective transform makes them slow)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(shade)
        p.drawPolygon(pose.quad)
        if pose.side is not None:
            p.drawPolygon(pose.side.map(art.side_mirror_outline if pose.side_mirror else art.side_outline))


def _body_bounds(art: CabinetArt, pose: Pose) -> QRectF:
    pad = art.pad
    box = pose.front.mapRect(QRectF(-pad, -pad, art.u + 2 * pad, art.h + 2 * pad))
    if pose.side_quad is not None:
        box = box.united(pose.side_quad.boundingRect())
    return box.adjusted(-2, -2, 2, 2)


def draw_cabinet(p: QPainter, art: CabinetArt, pose: Pose, feed: ScreenFeed | None, t: float,
                 trim_hue: float | None = None, marquee_glow: float = 0.0, coin_blink: bool = False,
                 reflection: float = 0.0, cache: dict | None = None, smooth: bool = True) -> None:
    """Paint one cabinet (side, front, live screen and overlays) at its pose (``reflection``: how strong its shine
    on the floor is, 0 for none). With ``cache`` (a cabinet standing still) the projected body is kept as a pixmap
    and only the live screen is drawn in perspective each frame."""
    base = p.transform()
    p.setOpacity(pose.opacity)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)  # the pixmaps have soft edges; AA here is slow
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
    if cache is not None and base.isIdentity():
        body = pose.ref or pose  # sliding: the picture at the nearest whole place, moved
        b = body.quad.boundingRect()
        key = (id(art), round(body.dim, 3), round(reflection, 3), p.device().devicePixelRatio(),
               round(b.x(), 1), round(b.y(), 1), round(b.width(), 1), round(b.height(), 1))
        slot = (pose.index, round(body.k))
        hit = cache.get(slot)
        if hit is None or hit[0] != key:
            box = QRectF(_body_bounds(art, body).toAlignedRect())  # whole pixels: a plain copy each frame
            dpr = p.device().devicePixelRatio()
            pm = QPixmap(max(int(box.width() * dpr) + 1, 1), max(int(box.height() * dpr) + 1, 1))
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            q.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            _draw_body(q, art, body, QTransform.fromTranslate(-box.left(), -box.top()), reflection)
            q.end()
            hit = (key, pm, box.topLeft())
            cache[slot] = hit
        if pose.affine is not None:
            p.setTransform(pose.affine)
            p.drawPixmap(hit[2], hit[1])
            p.setTransform(base)
        else:
            p.drawPixmap(hit[2].toPoint(), hit[1])
    else:
        _draw_body(p, art, pose, base, reflection)
    p.setTransform(pose.front * base)
    dim = pose.dim
    scr = art.screen_rect
    if feed is not None and not feed.broken and feed.pixmap is not None:
        feed.paint(p, scr, t)
    else:
        ScreenFeed.paint_out_of_order(p, scr, t, broken=feed is not None and feed.broken)
        if dim > 0.01:
            p.fillRect(scr, QColor(4, 2, 12, int(200 * min(dim, 0.8))))
    if marquee_glow > 0:
        p.fillRect(art.marquee_rect, QColor(255, 255, 255, int(70 * marquee_glow)))
    if coin_blink:
        for r in art.coin_rects:
            p.fillRect(r, QColor(255, 240, 180, 150))
    p.setTransform(base)
    if trim_hue is not None:  # a rainbow running round the outline, edge by edge (a gradient stroke is slow)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        quad = pose.quad
        n = quad.size()
        pen = QPen(QColor("white"), max(quad.boundingRect().width() * 0.016, 1.5), Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap)
        p.setOpacity(pose.opacity * (1 - 0.6 * min(pose.dim, 0.75)))
        for i in range(n):
            pen.setColor(QColor.fromHsvF((trim_hue + i / n) % 1.0, 0.8, 1.0))
            p.setPen(pen)
            p.drawLine(quad.at(i), quad.at((i + 1) % n))
    p.setOpacity(1.0)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
