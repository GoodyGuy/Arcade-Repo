"""Case Clash's Panorama-style UI kit: fonts, colours, panels, buttons, chips and the clickable-region registry
(immediate mode: every paint registers the regions the next click is tested against)."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap

from app.arcade.games.case_clash import data as D

BG_TOP = QColor("#2a3340")
BG_BOTTOM = QColor("#0e1217")
PANEL = QColor(28, 34, 42, 225)
PANEL_HEAD = QColor(255, 255, 255, 14)
BORDER = QColor(255, 255, 255, 34)
TEXT = QColor("#e9edf1")
DIM = QColor("#9aa6b2")
FAINT = QColor("#5f6b77")
GREEN = QColor("#4c9a2a")
GREEN_HI = QColor("#5fbf33")
MONEY = QColor("#8bd45b")
GOLD = QColor("#e4ae39")
ORANGE = QColor("#cf6a32")  # the kill counter's colour
BLUE = QColor("#5b9bd5")
RED = QColor("#d64545")

_fonts: dict[tuple, QFont] = {}


def F(size: float, weight: str = "semibold", cond: bool = False) -> QFont:  # noqa: N802 - short on purpose
    """Bahnschrift (a Stratum-like face), condensed for headers."""
    key = (int(max(size, 6)), weight, cond)
    f = _fonts.get(key)
    if f is None:
        f = QFont("Bahnschrift")
        f.setPixelSize(key[0])
        f.setWeight({"light": QFont.Weight.Light, "regular": QFont.Weight.Normal, "semibold": QFont.Weight.DemiBold,
                     "bold": QFont.Weight.Bold}[weight])
        if cond:
            f.setStretch(QFont.Stretch.Condensed)
        f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        _fonts[key] = f
    return f


_flags: dict = {}
_elided: dict = {}
_VERTICAL = Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignBottom


def text(p: QPainter, r: QRectF, s: str, size: float, color: QColor | str = TEXT, align=Qt.AlignmentFlag.AlignLeft,  # noqa: ANN001
         weight: str = "semibold", cond: bool = False, elide: bool = True, shadow: bool = False) -> None:
    f = F(size, weight, cond)
    p.setFont(f)
    if elide:
        key = (s, f.pixelSize(), weight, cond, int(r.width()))
        e = _elided.get(key)
        if e is None:
            if len(_elided) > 4000:
                _elided.clear()
            e = _elided[key] = QFontMetrics(f).elidedText(s, Qt.TextElideMode.ElideRight, int(max(r.width(), 1)))
        s = e
    flags = _flags.get(align)
    if flags is None:
        flags = _flags[align] = align if (align & _VERTICAL) else align | Qt.AlignmentFlag.AlignVCenter
    if shadow:
        p.setPen(QColor(0, 0, 0, 170))
        p.drawText(r.translated(1, 1), flags, s)
    p.setPen(color if isinstance(color, QColor) else QColor(color))
    p.drawText(r, flags, s)


def text_width(s: str, size: float, weight: str = "semibold", cond: bool = False) -> float:
    return QFontMetrics(F(size, weight, cond)).horizontalAdvance(s)


def panel(p: QPainter, r: QRectF, title: str = "", s: float = 1.0, alpha: int = 225) -> QRectF:
    """A Panorama panel; returns the body rect under the title bar."""
    c = QColor(PANEL)
    c.setAlpha(alpha)
    p.fillRect(r, c)
    p.setPen(QPen(BORDER, 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))
    if not title:
        return r
    hh = 30 * s
    head = QRectF(r.left(), r.top(), r.width(), hh)
    p.fillRect(head, PANEL_HEAD)
    p.setPen(QPen(QColor(255, 255, 255, 18), 1))
    p.drawLine(QPointF(r.left(), head.bottom()), QPointF(r.right(), head.bottom()))
    text(p, head.adjusted(12 * s, 0, -12 * s, 0), title.upper(), 14 * s, DIM, cond=True)
    return QRectF(r.left(), head.bottom(), r.width(), r.height() - hh)


def background(w: int, h: int, cache: dict) -> QPixmap:
    from app.arcade.games.case_clash import art as A

    key = ("bg", w, h, A.dpr())
    pm = cache.get(key)
    if pm is None:
        cache.clear()
        pm = A.new_pixmap(max(w, 1), max(h, 1))
        p = QPainter(pm)
        g = QLinearGradient(0, 0, w * 0.3, h)
        g.setColorAt(0.0, BG_TOP)
        g.setColorAt(1.0, BG_BOTTOM)
        p.fillRect(0, 0, w, h, g)
        # soft diagonal light and a vignette, like the Panorama menus
        g2 = QLinearGradient(0, 0, w, h * 0.6)
        g2.setColorAt(0.0, QColor(120, 150, 190, 30))
        g2.setColorAt(0.5, QColor(120, 150, 190, 0))
        p.fillRect(0, 0, w, h, g2)
        p.setPen(QPen(QColor(255, 255, 255, 6), 1))
        step = max(int(h / 40), 8)
        for y in range(0, h, step):
            p.drawLine(0, y, w, y)
        p.end()
        cache[key] = pm
    return pm


class Hits:
    """The clickable regions of the last paint."""

    def __init__(self) -> None:
        self.regions: list[tuple[QRectF, Callable[[], None], str, str]] = []  # rect, action, sound, label
        self.mouse = QPointF(-1000, -1000)
        self.live = True  # False while painting under a modal (no hover there)

    def begin(self) -> None:
        self.regions = []
        self.live = True

    def add(self, r: QRectF, action: Callable[[], None], sound: str = "click", label: str = "") -> None:
        self.regions.append((QRectF(r), action, sound, label))

    def hover(self, r: QRectF) -> bool:
        return self.live and r.contains(self.mouse)

    def find(self, pos: QPointF) -> tuple[Callable[[], None], str] | None:
        for r, action, sound, _label in reversed(self.regions):
            if r.contains(pos):
                return action, sound
        return None


def button(p: QPainter, hits: Hits, r: QRectF, label: str, action: Callable[[], None] | None, s: float = 1.0,
           style: str = "default", enabled: bool = True, size: float = 15, sound: str = "click") -> None:
    """A Panorama button: default (grey-blue), primary (green), gold, danger, ghost."""
    hov = enabled and hits.hover(r)
    base = {"primary": QColor("#3f7f24"), "gold": QColor("#a8791d"), "danger": QColor("#8a2a2a"),
            "ghost": QColor(255, 255, 255, 10), "blue": QColor("#2d5f8f")}.get(style, QColor("#3a4552"))
    if not enabled:
        base = QColor("#2a3038")
    elif hov:
        base = base.lighter(135) if style != "ghost" else QColor(255, 255, 255, 34)
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    g.setColorAt(0.0, base.lighter(118))
    g.setColorAt(1.0, base)
    p.fillRect(r, g)
    p.setPen(QPen(QColor(255, 255, 255, 50 if hov else 26), 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))
    text(p, r.adjusted(6 * s, 0, -6 * s, 0), label.upper(), size * s, TEXT if enabled else FAINT,
         Qt.AlignmentFlag.AlignHCenter, cond=True)
    if enabled and action is not None:
        hits.add(r, action, sound, label)


def chip(p: QPainter, hits: Hits, r: QRectF, label: str, active: bool, action: Callable[[], None],
         s: float = 1.0, color: QColor | None = None) -> None:
    hov = hits.hover(r)
    if active:
        p.fillRect(r, QColor(255, 255, 255, 40))
    elif hov:
        p.fillRect(r, QColor(255, 255, 255, 18))
    if active:
        p.fillRect(QRectF(r.left(), r.bottom() - 2 * s, r.width(), 2 * s), color or TEXT)
    text(p, r, label.upper(), 13 * s, (color or TEXT) if active else DIM, Qt.AlignmentFlag.AlignHCenter, cond=True)
    hits.add(r, action, "tab", label)


def bar(p: QPainter, r: QRectF, frac: float, color: QColor, back: QColor | None = None) -> None:
    p.fillRect(r, back if back is not None else QColor(255, 255, 255, 25))
    p.fillRect(QRectF(r.left(), r.top(), r.width() * max(0.0, min(frac, 1.0)), r.height()), color)


def money_color(cents: int) -> QColor:
    return MONEY if cents >= 0 else RED


def rarity_color(r: int) -> QColor:
    return QColor(D.RARITY_COLORS[r])


def sparkline(p: QPainter, r: QRectF, values: list[float], color: QColor) -> None:
    """A price line with a soft fill under it."""
    if len(values) < 2 or r.width() <= 2:
        return
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-6)
    path = QPainterPath()
    for i, v in enumerate(values):
        x = r.left() + r.width() * i / (len(values) - 1)
        y = r.bottom() - (v - lo) / span * r.height()
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    fill = QPainterPath(path)
    fill.lineTo(r.right(), r.bottom())
    fill.lineTo(r.left(), r.bottom())
    fill.closeSubpath()
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    c = QColor(color)
    c.setAlpha(60)
    g.setColorAt(0.0, c)
    g.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))
    p.fillPath(fill, g)
    p.setPen(QPen(color, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    p.restore()


def close_x(p: QPainter, hits: Hits, r: QRectF, action: Callable[[], None], s: float = 1.0) -> None:
    hov = hits.hover(r)
    if hov:
        p.fillRect(r, QColor(255, 255, 255, 30))
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(TEXT if hov else DIM, 2 * s))
    m = r.width() * 0.3
    p.drawLine(QPointF(r.left() + m, r.top() + m), QPointF(r.right() - m, r.bottom() - m))
    p.drawLine(QPointF(r.right() - m, r.top() + m), QPointF(r.left() + m, r.bottom() - m))
    p.restore()
    hits.add(r, action, "click", "close")
