"""Storm Drop's screens outside a match, drawn with QPainter over a live view of the island: the title menu, the
locker (outfits and canopies earned by playing), how to play, options, statistics, and the result of a match. ``Ui``
is the little toolkit they share: buttons that answer to the mouse and to the arrow keys."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF

from app.arcade.base import font

from .data import CANOPIES, GRADES, OUTFITS, STORM_PHASES, need_text, unlocked
from .hud import HEAVY, INK, icon, panel, text

A = Qt.AlignmentFlag
ACCENT = QColor("#ff8a3d")
PURPLE = QColor("#8a5cff")


class Ui:
    def __init__(self) -> None:
        self.hot: list[tuple[QRectF, str]] = []
        self.mouse = QPointF(-1.0, -1.0)
        self.focus = 0
        self.by_keys = True

    def begin(self) -> None:
        self.hot = []

    def hovered(self, r: QRectF, index: int) -> bool:
        return (index == self.focus) if self.by_keys else r.contains(self.mouse)

    def button(self, p: QPainter, r: QRectF, label: str, action: str, size: float, base: QColor | None = None,
               enabled: bool = True, selected: bool = False, align=A.AlignCenter) -> bool:  # noqa: ANN001
        index = len(self.hot)
        self.hot.append((r, action if enabled else ""))
        hot = enabled and self.hovered(r, index)
        if hot and not self.by_keys:
            self.focus = index
        c = QColor(base or QColor(26, 32, 62))
        if not enabled:
            c = QColor(40, 44, 60)
        fill = c.lighter(128) if hot else c
        fill.setAlpha(235 if hot else 205)
        rr = r.adjusted(-3, -2, 3, 2) if hot else r
        rad = r.height() * 0.28
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(6, 8, 20, 130))
        p.drawRoundedRect(rr.translated(0, r.height() * 0.1), rad, rad)
        panel(p, rr, rad, fill, QColor("#ffffff") if (hot or selected) else QColor(255, 255, 255, 60),
              3.0 if (hot or selected) else 1.5)
        text(p, rr.adjusted(r.height() * 0.4, 0, -r.height() * 0.4, 0), label, size,
             "#ffffff" if enabled else "#8a90a8", align, heavy=True)
        return hot

    def at(self, pos: QPointF) -> str:
        for r, action in self.hot:
            if r.contains(pos):
                return action
        return ""

    def move(self, d: int) -> bool:
        n = len(self.hot)
        if not n:
            return False
        self.by_keys = True
        i = self.focus
        for _ in range(n):
            i = (i + d) % n
            if self.hot[i][1]:
                self.focus = i
                return True
        return False

    def current(self) -> str:
        if 0 <= self.focus < len(self.hot):
            return self.hot[self.focus][1]
        return ""


def logo(p: QPainter, x: float, y: float, size: float, t: float = 0.0) -> None:
    """The game's name in chunky letters with a storm ring behind."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    f = QFont(HEAVY)
    f.setPixelSize(max(int(size), 4))
    f.setBold(True)
    f.setItalic(True)
    cx, cy = x + size * 1.55, y + size * 0.95
    for i in range(3):
        r = size * (1.25 + 0.16 * i)
        pen = QPen(QColor(170, 110, 255, 150 - 40 * i), size * 0.07)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        start = (t * (28 + 9 * i) + i * 70) % 360
        p.drawArc(QRectF(cx - r * 1.5, cy - r * 0.62, r * 3.0, r * 1.24), int(start * 16), int(250 * 16))
    for row, word in enumerate(("STORM", "DROP")):
        path = QPainterPath()
        path.addText(QPointF(x + row * size * 0.55, y + size * 0.86 + row * size * 0.92), f, word)
        p.setPen(QPen(INK, size * 0.16, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(INK)
        p.drawPath(path.translated(size * 0.05, size * 0.07))
        g = QLinearGradient(0, y + row * size * 0.92, 0, y + size + row * size * 0.92)
        if row == 0:
            g.setColorAt(0.0, QColor("#fff3a8"))
            g.setColorAt(1.0, QColor("#ff8a3d"))
        else:
            g.setColorAt(0.0, QColor("#d8c2ff"))
            g.setColorAt(1.0, QColor("#7a4dff"))
        p.setPen(QPen(QColor("#ffffff"), size * 0.035))
        p.setBrush(g)
        p.drawPath(path)
    p.restore()


def splash(p: QPainter, w: float, h: float, t: float) -> None:
    g = QLinearGradient(0, 0, 0, h)
    g.setColorAt(0.0, QColor("#2a7fe8"))
    g.setColorAt(0.6, QColor("#8fd6ff"))
    g.setColorAt(1.0, QColor("#2aa7c9"))
    p.fillRect(QRectF(0, 0, w, h), g)
    u = h / 780.0
    logo(p, w / 2 - 250 * u, h * 0.26, 110 * u, t)
    text(p, QRectF(0, h * 0.72, w, 30 * u), "Charting Brightwater Isle ...", 20 * u, "#ffffff")


def _title(p: QPainter, label: str, w: float, u: float) -> None:
    text(p, QRectF(60 * u, 36 * u, w - 120 * u, 56 * u), label, 44 * u, "#ffffff", A.AlignLeft | A.AlignVCenter,
         heavy=True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(ACCENT)
    p.drawRoundedRect(QRectF(62 * u, 96 * u, 120 * u, 6 * u), 3 * u, 3 * u)


def draw_menu(p: QPainter, g) -> None:  # noqa: ANN001 - the game widget
    w, h = g.width(), g.height()
    u = h / 780.0
    ui = g.ui
    grad = QLinearGradient(0, 0, w * 0.55, 0)
    grad.setColorAt(0.0, QColor(10, 14, 40, 190))
    grad.setColorAt(1.0, QColor(10, 14, 40, 0))
    p.fillRect(QRectF(0, 0, w * 0.55, h), grad)
    logo(p, 58 * u, 40 * u, 92 * u, g.time)
    text(p, QRectF(64 * u, 232 * u, 480 * u, 24 * u), "Drop in.  Loot up.  Build fast.  Be the last one standing.",
         16 * u, "#eaf2ff", A.AlignLeft | A.AlignVCenter, bold=False)
    rows = (("PLAY", "play"), ("LOCKER", "locker"), ("HOW TO PLAY", "help"), ("OPTIONS", "options"),
            ("STATISTICS", "stats"), ("BACK TO THE ARCADE", "quit"))
    y = 286 * u
    for i, (label, action) in enumerate(rows):
        big = i == 0
        hh = (64 if big else 48) * u
        ui.button(p, QRectF(62 * u, y, (360 if big else 320) * u, hh), label, action, (30 if big else 20) * u,
                  base=QColor("#ff7a2a") if big else None, align=A.AlignLeft | A.AlignVCenter)
        y += hh + 12 * u
    s = g.stats
    line = f"Wins {s['wins']}   ·   Eliminations {s['kills']}   ·   Matches {s['matches']}"
    text(p, QRectF(64 * u, h - 54 * u, 600 * u, 22 * u), line, 15 * u, "#ffffff", A.AlignLeft | A.AlignVCenter)
    o = OUTFITS[g.outfit]
    text(p, QRectF(w - 420 * u, h - 86 * u, 380 * u, 30 * u), o.name, 26 * u, "#ffffff", A.AlignRight | A.AlignVCenter,
         heavy=True)
    text(p, QRectF(w - 420 * u, h - 56 * u, 380 * u, 22 * u), f"{CANOPIES[g.canopy][0]} canopy  ·  change in the locker",
         14 * u, "#eaf2ff", A.AlignRight | A.AlignVCenter, bold=False)
    text(p, QRectF(64 * u, h - 30 * u, 700 * u, 18 * u), "↑ ↓ and Enter, or the mouse   ·   Esc: the arcade's menu",
         12 * u, "#c9d6f0", A.AlignLeft | A.AlignVCenter, bold=False)


def draw_locker(p: QPainter, g) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = h / 780.0
    ui = g.ui
    p.fillRect(QRectF(0, 0, w * 0.56, h), QColor(10, 14, 40, 170))
    _title(p, "LOCKER", w, u)
    text(p, QRectF(62 * u, 112 * u, 600 * u, 22 * u), "Outfits and canopies are earned by playing.", 15 * u,
         "#eaf2ff", A.AlignLeft | A.AlignVCenter, bold=False)
    cw, ch, gap = 300 * u, 84 * u, 12 * u
    for i, o in enumerate(OUTFITS):
        r = QRectF(62 * u + (i % 2) * (cw + gap), 150 * u + (i // 2) * (ch + gap), cw, ch)
        ok = unlocked(o.need, g.stats)
        sel = g.outfit == i
        hot = ui.button(p, r, "", f"outfit:{i}", 10, base=QColor(o.suit).darker(240) if ok else QColor(34, 38, 56),
                        selected=sel)
        rr = r.adjusted(-3, -2, 3, 2) if hot else r
        # the outfit in small: head and body in its colours
        bx, by = rr.x() + 18 * u, rr.y() + 12 * u
        p.setPen(QPen(INK, 2 * u))
        p.setBrush(QColor(o.suit) if ok else QColor(90, 94, 110))
        p.drawRoundedRect(QRectF(bx, by + 26 * u, 44 * u, 34 * u), 6 * u, 6 * u)
        p.setBrush(QColor(o.skin) if ok else QColor(120, 124, 140))
        p.drawRoundedRect(QRectF(bx + 7 * u, by, 30 * u, 28 * u), 6 * u, 6 * u)
        if o.hat_style and ok:
            p.setBrush(QColor(o.hat))
            p.drawRoundedRect(QRectF(bx + 5 * u, by - 4 * u, 34 * u, 10 * u), 4 * u, 4 * u)
        p.setBrush(QColor(o.trim) if ok else QColor(70, 74, 90))
        p.drawRect(QRectF(bx, by + 50 * u, 44 * u, 6 * u))
        text(p, QRectF(rr.x() + 78 * u, rr.y() + 12 * u, cw - 90 * u, 30 * u), o.name, 22 * u,
             "#ffffff" if ok else "#9aa0b8", A.AlignLeft | A.AlignVCenter, heavy=True)
        sub = "Wearing" if sel else ("Unlocked" if ok else f"Needs {need_text(o.need)}")
        text(p, QRectF(rr.x() + 78 * u, rr.y() + 44 * u, cw - 90 * u, 24 * u), sub, 14 * u,
             "#ffe14d" if sel else ("#bff5c8" if ok else "#ffb0b0"), A.AlignLeft | A.AlignVCenter, bold=False)
    y = 150 * u + 4 * (ch + gap) + 6 * u
    text(p, QRectF(62 * u, y, 400 * u, 26 * u), "CANOPY", 18 * u, "#ffffff", A.AlignLeft | A.AlignVCenter, heavy=True)
    y += 32 * u
    sw = (2 * cw + gap - 5 * 8 * u) / 6
    for i, (name, need, c1, c2) in enumerate(CANOPIES):
        r = QRectF(62 * u + i * (sw + 8 * u), y, sw, 78 * u)
        ok = unlocked(need, g.stats)
        hot = ui.button(p, r, "", f"canopy:{i}", 10, base=QColor(30, 36, 66), selected=g.canopy == i)
        rr = r.adjusted(-3, -2, 3, 2) if hot else r
        dome = QPainterPath()
        dome.moveTo(rr.x() + 12 * u, rr.y() + 40 * u)
        dome.cubicTo(rr.x() + 12 * u, rr.y() + 4 * u, rr.right() - 12 * u, rr.y() + 4 * u, rr.right() - 12 * u,
                     rr.y() + 40 * u)
        dome.closeSubpath()
        p.setPen(QPen(INK, 2 * u))
        gr = QLinearGradient(rr.x(), 0, rr.right(), 0)
        for k in range(6):
            col = QColor(c1 if k % 2 == 0 else c2) if ok else QColor(90 + 20 * (k % 2), 94, 110)
            gr.setColorAt(k / 6 + 0.001, col)
            gr.setColorAt((k + 1) / 6, col)
        p.setBrush(gr)
        p.drawPath(dome)
        text(p, QRectF(rr.x(), rr.y() + 42 * u, rr.width(), 18 * u), name, 12 * u, "#ffffff" if ok else "#9aa0b8")
        text(p, QRectF(rr.x(), rr.y() + 58 * u, rr.width(), 16 * u), "" if ok else need_text(need), 10 * u, "#ffb0b0",
             bold=False)
    ui.button(p, QRectF(62 * u, h - 78 * u, 200 * u, 48 * u), "BACK", "back", 20 * u)


def draw_help(p: QPainter, g) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = h / 780.0
    p.fillRect(QRectF(0, 0, w, h), QColor(10, 14, 40, 205))
    _title(p, "HOW TO PLAY", w, u)
    total = sum(a + b for a, b, _r, _d in STORM_PHASES) + 30
    cols = (
        (("THE MATCH", (
            "You and the bots ride the drop ship over Brightwater Isle.",
            "Press Space to jump. Look where you want to go and hold W to",
            "dive; the canopy opens by itself near the ground.",
            "Land, loot, fight. The last one standing wins.",
            f"A match lasts about {total // 60:.0f} minutes.")),
         ("LOOT", (
             "Chests glow gold: stand by one and press E.",
             "Ammo and materials are picked up by walking over them.",
             "Five grades, each a little stronger than the last:",
             "@grades",
             "Close, mid, long range and explosive weapons use their own",
             "ammo. Headshots hurt more. Shields soak damage first, but",
             "not the storm or a long fall.")),
         ("THE STORM", (
             "The white ring on the map is the next eye. When the clock",
             "runs out the storm closes in on it. Outside you lose health",
             "every second, and more with every ring."))),
        (("BUILDING", (
            "Hit trees for wood, rocks and walls for brick, cars, tanks",
            "and containers for metal with the sledge (F).",
            "Z wall, X floor, C ramp, V roof: a blue ghost shows where",
            "the piece will go. Left click places it for 10 materials;",
            "right click changes the material.",
            "Wood is finished fast, metal ends up strongest.",
            "Pieces need the ground or another piece under them: shoot",
            "out the bottom and the rest comes down.")),
         ("CONTROLS", (
             "@keys",))),
    )
    x = 62 * u
    cw = (w - 124 * u - 30 * u) / 2
    for col in cols:
        y = 128 * u
        for head, lines in col:
            text(p, QRectF(x, y, cw, 26 * u), head, 20 * u, "#ffb86b", A.AlignLeft | A.AlignVCenter, heavy=True)
            y += 30 * u
            for line in lines:
                if line == "@grades":
                    gx = x
                    for name, c in GRADES:
                        r = QRectF(gx, y + 2 * u, 86 * u, 22 * u)
                        panel(p, r, 6 * u, QColor(c))
                        text(p, r, name, 13 * u, "#141a2c", shadow=False)
                        gx += 92 * u
                    y += 30 * u
                elif line == "@keys":
                    keys = g.controls
                    half = (len(keys) + 1) // 2
                    for i, (k, what) in enumerate(keys):
                        kx = x + (i // half) * (cw / 2)
                        ky = y + (i % half) * 23 * u
                        text(p, QRectF(kx, ky, 120 * u, 22 * u), k, 13 * u, "#ffe14d", A.AlignLeft | A.AlignVCenter)
                        text(p, QRectF(kx + 122 * u, ky, cw / 2 - 124 * u, 22 * u), what, 13 * u, "#ffffff",
                             A.AlignLeft | A.AlignVCenter, bold=False)
                    y += half * 23 * u
                else:
                    text(p, QRectF(x, y, cw, 22 * u), line, 14.5 * u, "#f2f6ff", A.AlignLeft | A.AlignVCenter,
                         bold=False, shadow=False)
                    y += 22 * u
            y += 16 * u
        x += cw + 30 * u
    g.ui.button(p, QRectF(62 * u, h - 70 * u, 200 * u, 46 * u), "BACK", "back", 20 * u)


OPTION_ROWS = (  # (key, label, kind, choices)
    ("bots", "Bots in a match", "int", (16, 32, 2)),
    ("difficulty", "Bot skill", "choice", ("easy", "normal", "hard")),
    ("fov", "Field of view", "int", (60, 100, 5)),
    ("auto_pickup", "Pick weapons up by walking over them", "bool", None),
    ("hold_to_build", "Hold the button to keep building", "bool", None),
    ("damage_numbers", "Damage numbers", "bool", None),
)


def draw_options(p: QPainter, g) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = h / 780.0
    ui = g.ui
    p.fillRect(QRectF(0, 0, w, h), QColor(10, 14, 40, 195))
    _title(p, "OPTIONS", w, u)
    y = 140 * u
    for key, label, kind, _choices in OPTION_ROWS:
        r = QRectF(62 * u, y, 760 * u, 54 * u)
        panel(p, r, 12 * u, QColor(22, 28, 58, 215))
        text(p, r.adjusted(22 * u, 0, 0, 0), label, 19 * u, "#ffffff", A.AlignLeft | A.AlignVCenter)
        val = g.options[key]
        shown = ("On" if val else "Off") if kind == "bool" else (str(val).capitalize() if kind == "choice" else str(val))
        ui.button(p, QRectF(r.right() - 250 * u, y + 7 * u, 44 * u, 40 * u), "‹", f"opt:{key}:-1", 24 * u)
        text(p, QRectF(r.right() - 200 * u, y, 140 * u, 54 * u), shown, 20 * u, "#ffe14d", heavy=True)
        ui.button(p, QRectF(r.right() - 54 * u, y + 7 * u, 44 * u, 40 * u), "›", f"opt:{key}:1", 24 * u)
        y += 64 * u
    ui.button(p, QRectF(62 * u, y + 10 * u, 370 * u, 48 * u), "KEYS AND MOUSE ...", "controls", 19 * u,
              base=QColor(60, 50, 130))
    ui.button(p, QRectF(62 * u, h - 78 * u, 200 * u, 48 * u), "BACK", "back", 20 * u)
    text(p, QRectF(62 * u, y + 66 * u, 760 * u, 22 * u),
         "← → change the highlighted setting. Sound volume is in the arcade's own menu (Esc).", 13 * u, "#c9d6f0",
         A.AlignLeft | A.AlignVCenter, bold=False)


def draw_stats(p: QPainter, g) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = h / 780.0
    p.fillRect(QRectF(0, 0, w, h), QColor(10, 14, 40, 195))
    _title(p, "STATISTICS", w, u)
    s = g.stats
    played = max(s["matches"], 1)
    mins = s["seconds"] / 60.0
    rows = (
        ("Matches played", str(s["matches"])), ("Wins", str(s["wins"])),
        ("Win rate", f"{100.0 * s['wins'] / played:.0f} %"), ("Top five finishes", str(s["top5"])),
        ("Best place", f"#{s['best_place']}" if s["best_place"] else "-"), ("Eliminations", str(s["kills"])),
        ("Most in one match", str(s["best_kills"])), ("Per match", f"{s['kills'] / played:.1f}"),
        ("Headshots", str(s["headshots"])), ("Longest elimination", f"{s['longest']:.0f} m"),
        ("Damage dealt", f"{s['damage']:.0f}"), ("Pieces built", str(s["built"])),
        ("Materials harvested", str(s["harvested"])), ("Chests opened", str(s["chests"])),
        ("Time on the island", f"{mins:.0f} min"),
    )
    cw = 420 * u
    for i, (label, val) in enumerate(rows):
        r = QRectF(62 * u + (i % 2) * (cw + 20 * u), 136 * u + (i // 2) * 58 * u, cw, 48 * u)
        panel(p, r, 10 * u, QColor(22, 28, 58, 215))
        text(p, r.adjusted(18 * u, 0, 0, 0), label, 17 * u, "#dfe8ff", A.AlignLeft | A.AlignVCenter, bold=False)
        text(p, r.adjusted(0, 0, -18 * u, 0), val, 22 * u, "#ffe14d", A.AlignRight | A.AlignVCenter, heavy=True)
    g.ui.button(p, QRectF(62 * u, h - 78 * u, 200 * u, 48 * u), "BACK", "back", 20 * u)


def draw_result(p: QPainter, g) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = h / 780.0
    r = g.result
    ui = g.ui
    p.fillRect(QRectF(0, 0, w, h), QColor(10, 14, 40, 170))
    win = r["place"] == 1
    if win:
        for i in range(60):  # confetti
            a = (i * 0.618) % 1.0
            x = (a * w + math.sin(g.time * 1.3 + i) * 40 * u) % w
            y = ((g.time * (90 + i % 7 * 22) + i * 53) * u) % h
            p.save()
            p.translate(x, y)
            p.rotate((g.time * 180 + i * 40) % 360)
            p.fillRect(QRectF(-6 * u, -3 * u, 12 * u, 6 * u), QColor(GRADES[i % 5][1]))
            p.restore()
    head = "LAST ONE STANDING!" if win else f"YOU PLACED  #{r['place']}"
    text(p, QRectF(0, 70 * u, w, 80 * u), head, 68 * u, "#ffe14d" if win else "#ffffff", heavy=True)
    sub = f"of {r['total']}" + (f"   ·   out to {r['by']}" if r.get("by") else "")
    text(p, QRectF(0, 150 * u, w, 30 * u), "You outlasted everyone on Brightwater Isle." if win else sub, 20 * u,
         "#eaf2ff", bold=False)
    rows = (("Eliminations", str(r["kills"])), ("Damage dealt", f"{r['damage']:.0f}"),
            ("Time alive", f"{int(r['seconds']) // 60}:{int(r['seconds']) % 60:02d}"),
            ("Pieces built", str(r["built"])), ("Materials harvested", str(r["harvested"])),
            ("Chests opened", str(r["chests"])), ("Headshots", str(r["headshots"])),
            ("Longest elimination", f"{r['longest']:.0f} m"))
    cw = 300 * u
    x0 = w / 2 - (2 * cw + 20 * u) / 2
    for i, (label, val) in enumerate(rows):
        rr = QRectF(x0 + (i % 2) * (cw + 20 * u), 212 * u + (i // 2) * 60 * u, cw, 50 * u)
        panel(p, rr, 10 * u, QColor(22, 28, 58, 225))
        text(p, rr.adjusted(18 * u, 0, 0, 0), label, 16 * u, "#dfe8ff", A.AlignLeft | A.AlignVCenter, bold=False)
        text(p, rr.adjusted(0, 0, -18 * u, 0), val, 24 * u, "#ffe14d", A.AlignRight | A.AlignVCenter, heavy=True)
    y = 212 * u + 4 * 60 * u + 10 * u
    if r.get("unlocked"):
        rr = QRectF(x0, y, 2 * cw + 20 * u, 50 * u)
        panel(p, rr, 10 * u, QColor(90, 60, 190, 230), QColor("#ffffff"), 2 * u)
        text(p, rr, "Unlocked in the locker:  " + ",  ".join(r["unlocked"]), 18 * u, "#ffffff", heavy=True)
        y += 62 * u
    bw = 230 * u
    bx = w / 2 - (3 * bw + 2 * 16 * u) / 2
    for i, (label, action, base) in enumerate((("PLAY AGAIN", "again", QColor("#ff7a2a")), ("LOCKER", "locker", None),
                                               ("MENU", "back", None))):
        ui.button(p, QRectF(bx + i * (bw + 16 * u), y + 12 * u, bw, 54 * u), label, action, 22 * u, base=base)


def draw_out(p: QPainter, g) -> None:  # noqa: ANN001
    """Over the match once the player is out (watching) or has won."""
    w, h = g.width(), g.height()
    u = h / 780.0
    m = g.match
    me = m.player
    if m.over and m.winner is me:
        text(p, QRectF(0, h * 0.2, w, 90 * u), "LAST ONE STANDING!", 72 * u, "#ffe14d", heavy=True)
        text(p, QRectF(0, h * 0.2 + 92 * u, w, 30 * u), "Enter: the result", 18 * u, "#ffffff")
        return
    r = QRectF(w / 2 - 300 * u, h * 0.13, 600 * u, 112 * u)
    panel(p, r, 16 * u, QColor(12, 16, 34, 205), QColor(255, 255, 255, 80), 2 * u)
    text(p, QRectF(r.x(), r.y() + 8 * u, r.width(), 46 * u), f"OUT  ·  #{me.place} of {m.total}", 34 * u, "#ff8a8a",
         heavy=True)
    who = g.view_actor()
    line = f"Watching {who.name}" if who is not me else "Watching"
    if m.over and m.winner is not None:
        line = f"{m.winner.name} is the last one standing"
    text(p, QRectF(r.x(), r.y() + 54 * u, r.width(), 24 * u), line, 17 * u, "#ffffff")
    text(p, QRectF(r.x(), r.y() + 80 * u, r.width(), 22 * u), "Enter: the result   ·   click: watch someone else",
         13 * u, "#c9d6f0", bold=False)


def thumbnail(p: QPainter, rect: QRectF, t: float, accent: str) -> None:
    """The game's card on the arcade's home screen: the island, the storm ring and someone dropping in."""
    p.save()
    p.setClipRect(rect)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    sky = QLinearGradient(0, y, 0, y + h)
    sky.setColorAt(0.0, QColor("#2a7fe8"))
    sky.setColorAt(0.62, QColor("#9fdcff"))
    sky.setColorAt(0.63, QColor("#2fb6d6"))
    sky.setColorAt(1.0, QColor("#1b7fb8"))
    p.fillRect(rect, sky)
    p.setPen(Qt.PenStyle.NoPen)
    for i in range(4):  # clouds
        cx = x + ((i * 0.31 + t * 0.012 * (1 + i % 2)) % 1.2 - 0.1) * w
        cy = y + h * (0.12 + 0.09 * (i % 3))
        p.setBrush(QColor(255, 255, 255, 215))
        p.drawEllipse(QPointF(cx, cy), w * 0.07, h * 0.035)
        p.drawEllipse(QPointF(cx + w * 0.05, cy + h * 0.01), w * 0.06, h * 0.03)
    # the storm behind
    for side in (0, 1):
        sw = w * (0.2 + 0.03 * math.sin(t * (1.0 - 0.2 * side)))
        sg = QLinearGradient(x if side == 0 else x + w, 0, x + sw if side == 0 else x + w - sw, 0)
        sg.setColorAt(0.0, QColor(130, 60, 220, 190))
        sg.setColorAt(1.0, QColor(170, 110, 255, 0))
        p.fillRect(QRectF(x if side == 0 else x + w - sw, y, sw, h), sg)
    # the island
    iy = y + h * 0.64
    p.setBrush(QColor("#f0da9c"))
    p.drawEllipse(QPointF(x + w * 0.5, iy + h * 0.02), w * 0.36, h * 0.07)
    p.setBrush(QColor("#6fc457"))
    p.drawEllipse(QPointF(x + w * 0.5, iy), w * 0.31, h * 0.06)
    p.setBrush(QColor("#a09a96"))
    p.drawPolygon(QPolygonF([QPointF(x + w * 0.56, iy), QPointF(x + w * 0.66, iy - h * 0.2), QPointF(x + w * 0.76, iy)]))
    p.setBrush(QColor("#f2f4f8"))
    p.drawPolygon(QPolygonF([QPointF(x + w * 0.63, iy - h * 0.14), QPointF(x + w * 0.66, iy - h * 0.2),
                             QPointF(x + w * 0.69, iy - h * 0.14)]))
    for i, fx in enumerate((0.3, 0.36, 0.44, 0.5)):  # trees and a house
        bx = x + w * fx
        p.setBrush(QColor("#2c8a4a"))
        p.drawPolygon(QPolygonF([QPointF(bx - w * 0.018, iy), QPointF(bx, iy - h * (0.09 + 0.02 * (i % 2))),
                                 QPointF(bx + w * 0.018, iy)]))
    p.setBrush(QColor("#d1694f"))
    p.drawRect(QRectF(x + w * 0.39, iy - h * 0.05, w * 0.045, h * 0.05))
    p.setBrush(QColor("#f26d3d"))
    p.drawPolygon(QPolygonF([QPointF(x + w * 0.385, iy - h * 0.05), QPointF(x + w * 0.4125, iy - h * 0.09),
                             QPointF(x + w * 0.44, iy - h * 0.05)]))
    # someone under a canopy
    px = x + w * (0.3 + 0.03 * math.sin(t * 0.7))
    py = y + h * (0.3 + 0.03 * math.sin(t * 1.1))
    s = h * 0.1
    p.setPen(QPen(INK, max(s * 0.06, 1.0)))
    dome = QPainterPath()
    dome.moveTo(px - s * 1.3, py)
    dome.cubicTo(px - s * 1.3, py - s * 1.5, px + s * 1.3, py - s * 1.5, px + s * 1.3, py)
    dome.closeSubpath()
    gr = QLinearGradient(px - s * 1.3, 0, px + s * 1.3, 0)
    for k in range(6):
        c = QColor("#ff8a3d" if k % 2 == 0 else "#ffe14d")
        gr.setColorAt(k / 6 + 0.001, c)
        gr.setColorAt((k + 1) / 6, c)
    p.setBrush(gr)
    p.drawPath(dome)
    p.drawLine(QPointF(px - s * 1.2, py), QPointF(px - s * 0.2, py + s * 1.1))
    p.drawLine(QPointF(px + s * 1.2, py), QPointF(px + s * 0.2, py + s * 1.1))
    p.setBrush(QColor("#22c1c3"))
    p.drawRoundedRect(QRectF(px - s * 0.3, py + s * 1.2, s * 0.6, s * 0.6), s * 0.1, s * 0.1)
    p.setBrush(QColor("#f2c59b"))
    p.drawRoundedRect(QRectF(px - s * 0.22, py + s * 0.82, s * 0.44, s * 0.4), s * 0.08, s * 0.08)
    p.setBrush(QColor("#fdf3d0"))
    p.drawRect(QRectF(px - s * 0.25, py + s * 1.8, s * 0.2, s * 0.4))
    p.drawRect(QRectF(px + s * 0.05, py + s * 1.8, s * 0.2, s * 0.4))
    logo(p, x + w * 0.5, y + h * 0.08, h * 0.2, t)
    p.restore()


__all__ = ["Ui", "draw_help", "draw_locker", "draw_menu", "draw_options", "draw_out", "draw_result", "draw_stats",
           "icon", "splash", "thumbnail", "font", "PURPLE"]
