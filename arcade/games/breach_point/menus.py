"""BREACH POINT's menus: the title, match set-up (map, agent, difficulty, length, side), the agents explained, options,
the service record, the pause menu and the loading card. Every screen paints itself and registers what can be clicked
with ``UI``; the game widget turns a click (or Enter on the focused item) into an action."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPolygonF

from . import config as C
from .hud import ACCENT, ALLY, CROSS_COLOURS, DIM, ENEMY, INK, PAPER, ability_pix, minimap_pix, rgba, text, weapon_pix
from .maps import MAP_INFO, MAP_ORDER

AL = Qt.AlignmentFlag


class UI:
    """What was painted that can be clicked: (rectangle, action)."""

    def __init__(self) -> None:
        self.items: list = []
        self.mouse = QPointF(-10, -10)
        self.focus = -1
        self.use_keys = False

    def begin(self) -> None:
        self.items = []

    def hover(self, r: QRectF) -> bool:
        return not self.use_keys and r.contains(self.mouse)

    def button(self, r: QRectF, action: tuple) -> bool:
        self.items.append((QRectF(r), action))
        if self.use_keys:
            return len(self.items) - 1 == self.focus
        return r.contains(self.mouse)

    def at(self, pos: QPointF) -> tuple | None:
        for r, action in reversed(self.items):
            if r.contains(pos):
                return action
        return None

    def step(self, d: int) -> None:
        if self.items:
            self.use_keys = True
            self.focus = (self.focus + d) % len(self.items) if self.focus >= 0 else 0

    def focused(self) -> tuple | None:
        if self.use_keys and 0 <= self.focus < len(self.items):
            return self.items[self.focus][1]
        return None

    def moved(self, pos: QPointF) -> None:
        self.mouse = pos
        self.use_keys = False


def backdrop(p: QPainter, w: int, h: int, t: float) -> None:
    g = QLinearGradient(0, 0, w, h)
    g.setColorAt(0.0, QColor(26, 22, 44))
    g.setColorAt(1.0, QColor(12, 11, 22))
    p.fillRect(QRectF(0, 0, w, h), g)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    s = h / 780
    drift = (t * 14) % (160 * s)
    p.setBrush(QColor(255, 198, 46, 14))
    for i in range(-2, 14):
        x = i * 160 * s + drift
        p.drawPolygon(QPolygonF([QPointF(x, h), QPointF(x + 60 * s, h), QPointF(x + 60 * s + h * 0.5, 0),
                                 QPointF(x + h * 0.5, 0)]))
    p.setBrush(QColor(255, 198, 46, 235))
    p.drawPolygon(QPolygonF([QPointF(w * 0.6, 0), QPointF(w * 0.6 + 26 * s, 0), QPointF(w * 0.6 - h * 0.42 + 26 * s, h),
                             QPointF(w * 0.6 - h * 0.42, h)]))
    p.setBrush(QColor(255, 61, 139, 200))
    p.drawPolygon(QPolygonF([QPointF(w * 0.6 + 40 * s, 0), QPointF(w * 0.6 + 48 * s, 0),
                             QPointF(w * 0.6 - h * 0.42 + 48 * s, h), QPointF(w * 0.6 - h * 0.42 + 40 * s, h)]))
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)


def _button(p: QPainter, ui: UI, r: QRectF, label: str, action: tuple, s: float, big: bool = False,
            on: bool = False, sub: str = "") -> bool:
    hov = ui.button(r, action)
    bg = QColor(255, 198, 46) if hov else (QColor(60, 54, 92, 235) if on else QColor(30, 27, 48, 225))
    p.fillRect(r, bg)
    if on and not hov:
        p.fillRect(QRectF(r.x(), r.y(), 4 * s, r.height()), ACCENT)
    col = INK if hov else PAPER
    if sub:
        text(p, QRectF(r.x() + 16 * s, r.y() + 4 * s, r.width() - 24 * s, r.height() * 0.55), label,
             (20 if big else 14) * s, col, AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(r.x() + 16 * s, r.y() + r.height() * 0.52, r.width() - 24 * s, r.height() * 0.4), sub, 10 * s,
             INK if hov else DIM, AL.AlignLeft | AL.AlignVCenter, False)
    else:
        text(p, QRectF(r.x() + 16 * s, r.y(), r.width() - 24 * s, r.height()), label, (20 if big else 13) * s, col,
             AL.AlignLeft | AL.AlignVCenter)
    return hov


def _title(p: QPainter, w: int, s: float, label: str, sub: str = "") -> None:
    text(p, QRectF(60 * s, 34 * s, w, 50 * s), label, 38 * s, PAPER, AL.AlignLeft | AL.AlignVCenter)
    p.fillRect(QRectF(60 * s, 88 * s, 90 * s, 4 * s), ACCENT)
    if sub:
        text(p, QRectF(60 * s, 96 * s, w, 22 * s), sub, 12 * s, DIM, AL.AlignLeft | AL.AlignVCenter, False)


def paint_main(p: QPainter, w: int, h: int, g) -> None:  # noqa: ANN001
    s = h / 780
    ui = g.ui
    text(p, QRectF(60 * s, 120 * s, w, 96 * s), "BREACH", 92 * s, PAPER, AL.AlignLeft | AL.AlignVCenter)
    text(p, QRectF(60 * s, 204 * s, w, 96 * s), "POINT", 92 * s, ACCENT, AL.AlignLeft | AL.AlignVCenter)
    text(p, QRectF(64 * s, 300 * s, w, 24 * s), "FIVE AGAINST FIVE  ·  ONE LIFE A ROUND  ·  PLANT IT OR STOP IT", 13 * s,
         DIM, AL.AlignLeft | AL.AlignVCenter)
    rows = (("PLAY", ("page", "play"), "Pick a map, an agent and a side"),
            ("AGENTS", ("page", "agents"), "Five kits, explained"),
            ("OPTIONS", ("page", "options"), "View, crosshair, controls"),
            ("RECORD", ("page", "stats"), "What you have done so far"),
            ("BACK TO THE ARCADE", ("exit",), ""))
    y = 356 * s
    for label, action, sub in rows:
        _button(p, ui, QRectF(60 * s, y, 360 * s, 58 * s), label, action, s, True, sub=sub)
        y += 68 * s
    st = g.stats
    if st["matches"]:
        text(p, QRectF(w - 420 * s, h - 60 * s, 380 * s, 20 * s),
             f"{st['wins']} WON  ·  {st['losses']} LOST  ·  {st['kills']} KILLS", 12 * s, DIM,
             AL.AlignRight | AL.AlignVCenter)


def paint_play(p: QPainter, w: int, h: int, g) -> None:  # noqa: ANN001
    s = h / 780
    ui = g.ui
    ch = g.choice
    _title(p, w, s, "NEW MATCH", "First to the round count wins. Sides swap one round before that.")
    x = 60 * s
    y = 134 * s
    text(p, QRectF(x, y, 300 * s, 20 * s), "MAP", 12 * s, ACCENT, AL.AlignLeft | AL.AlignVCenter)
    cw = 330 * s
    for i, key in enumerate(MAP_ORDER):
        r = QRectF(x + i * (cw + 14 * s), y + 24 * s, cw, 150 * s)
        on = ch["map"] == key
        hov = ui.button(r, ("set", "map", key))
        p.fillRect(r, QColor(60, 54, 92, 235) if on else (QColor(44, 40, 66, 235) if hov else QColor(30, 27, 48, 225)))
        if on:
            p.fillRect(QRectF(r.x(), r.y(), r.width(), 4 * s), ACCENT)
        pm = g.map_preview(key)
        if pm is not None:
            ph = r.height() - 20 * s
            pw = ph * pm.width() / pm.height()
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawPixmap(QRectF(r.x() + 10 * s, r.y() + 12 * s, pw, ph), pm, QRectF(pm.rect()))
            tx = r.x() + pw + 22 * s
        else:
            tx = r.x() + 16 * s
        name, blurb = MAP_INFO[key]
        text(p, QRectF(tx, r.y() + 14 * s, r.right() - tx - 8 * s, 24 * s), name, 17 * s, PAPER,
             AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(tx, r.y() + 42 * s, r.right() - tx - 10 * s, 96 * s), blurb, 10.5 * s, DIM,
             AL.AlignLeft | AL.AlignTop | Qt.TextFlag.TextWordWrap, False)
    y += 190 * s
    text(p, QRectF(x, y, 300 * s, 20 * s), "AGENT", 12 * s, ACCENT, AL.AlignLeft | AL.AlignVCenter)
    aw = 200 * s
    for i, key in enumerate(C.AGENT_ORDER):
        ag = C.AGENTS[key]
        r = QRectF(x + i * (aw + 10 * s), y + 24 * s, aw, 132 * s)
        on = ch["agent"] == key
        hov = ui.button(r, ("set", "agent", key))
        p.fillRect(r, QColor(60, 54, 92, 235) if on else (QColor(44, 40, 66, 235) if hov else QColor(30, 27, 48, 225)))
        p.fillRect(QRectF(r.x(), r.y(), r.width(), 5 * s), rgba(ag.color, 255 if on else 150))
        text(p, QRectF(r.x() + 12 * s, r.y() + 12 * s, aw, 26 * s), ag.name, 20 * s, rgba(ag.color),
             AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(r.x() + 12 * s, r.y() + 38 * s, aw, 16 * s), ag.role.upper(), 10 * s, PAPER,
             AL.AlignLeft | AL.AlignVCenter)
        for k, ab in enumerate(ag.abilities):
            p.drawPixmap(QPointF(r.x() + 10 * s + k * 46 * s, r.y() + 60 * s),
                         ability_pix(ab.kind, int(40 * s), ag.color if on or hov else (150, 150, 165)))
        text(p, QRectF(r.x() + 12 * s, r.y() + 106 * s, aw - 20 * s, 18 * s), f'"{ag.line}"', 9.5 * s, DIM,
             AL.AlignLeft | AL.AlignVCenter, False)
    y += 176 * s
    groups = (("DIFFICULTY", "difficulty", [(k, v["label"]) for k, v in C.DIFFICULTY.items()]),
              ("FIRST TO", "rounds", [(n, f"{n} ROUNDS") for n in C.MATCH_LENGTHS]),
              ("START AS", "side", [("atk", "ATTACK"), ("def", "DEFENCE")]))
    gx = x
    for label, key, opts in groups:
        text(p, QRectF(gx, y, 300 * s, 20 * s), label, 12 * s, ACCENT, AL.AlignLeft | AL.AlignVCenter)
        bw = 104 * s
        for i, (val, name) in enumerate(opts):
            _button(p, ui, QRectF(gx + i * (bw + 6 * s), y + 24 * s, bw, 40 * s), name, ("set", key, val), s,
                    on=ch[key] == val)
        gx += len(opts) * (bw + 6 * s) + 26 * s
    y += 92 * s
    _button(p, ui, QRectF(x, y, 300 * s, 60 * s), "START MATCH", ("start",), s, True)
    _button(p, ui, QRectF(x + 316 * s, y, 160 * s, 60 * s), "BACK", ("page", "main"), s, True)


def paint_agents(p: QPainter, w: int, h: int, g) -> None:  # noqa: ANN001
    s = h / 780
    ui = g.ui
    _title(p, w, s, "AGENTS", "Two abilities you buy, one that comes back on its own, one ultimate earned by points.")
    x, y = 60 * s, 136 * s
    for i, key in enumerate(C.AGENT_ORDER):
        ag = C.AGENTS[key]
        r = QRectF(x, y + i * 62 * s, 230 * s, 54 * s)
        on = g.agent_page == key
        hov = ui.button(r, ("agent_page", key))
        p.fillRect(r, QColor(60, 54, 92, 235) if on else (QColor(44, 40, 66, 235) if hov else QColor(30, 27, 48, 225)))
        p.fillRect(QRectF(r.x(), r.y(), 5 * s, r.height()), rgba(ag.color))
        text(p, QRectF(r.x() + 18 * s, r.y() + 4 * s, 200 * s, 28 * s), ag.name, 19 * s, rgba(ag.color),
             AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(r.x() + 18 * s, r.y() + 30 * s, 200 * s, 18 * s), ag.role.upper(), 10 * s, DIM,
             AL.AlignLeft | AL.AlignVCenter)
    _button(p, ui, QRectF(x, y + 5 * 62 * s + 14 * s, 230 * s, 46 * s), "BACK", ("page", "main"), s, True)
    ag = C.AGENTS[g.agent_page]
    px = x + 260 * s
    pw = w - px - 60 * s
    p.fillRect(QRectF(px, y, pw, 560 * s), QColor(24, 22, 40, 225))
    p.fillRect(QRectF(px, y, pw, 5 * s), rgba(ag.color))
    text(p, QRectF(px + 24 * s, y + 16 * s, pw, 50 * s), ag.name, 44 * s, rgba(ag.color),
         AL.AlignLeft | AL.AlignVCenter)
    text(p, QRectF(px + 24 * s, y + 66 * s, pw, 20 * s), f"{ag.role.upper()}  ·  {C.ROLES[ag.role]}", 12 * s, PAPER,
         AL.AlignLeft | AL.AlignVCenter)
    text(p, QRectF(px + 24 * s, y + 92 * s, pw - 48 * s, 44 * s), ag.about, 11.5 * s, DIM,
         AL.AlignLeft | AL.AlignTop | Qt.TextFlag.TextWordWrap, False)
    yy = y + 150 * s
    for ab in ag.abilities:
        r = QRectF(px + 24 * s, yy, pw - 48 * s, 90 * s)
        p.fillRect(r, QColor(34, 31, 54, 235))
        p.drawPixmap(QPointF(r.x() + 12 * s, r.y() + 13 * s), ability_pix(ab.kind, int(64 * s), ag.color))
        text(p, QRectF(r.x() + 92 * s, r.y() + 10 * s, 400 * s, 24 * s), ab.name, 17 * s, PAPER,
             AL.AlignLeft | AL.AlignVCenter)
        if ab.slot == "X":
            note = f"ULTIMATE  ·  {ab.points} POINTS  (a kill, a plant, a defuse, and one each round)"
        elif ab.slot == "E":
            note = f"FREE  ·  RETURNS EVERY {int(ab.recharge)} S" + (f"  ·  HOLDS {ab.charges}" if ab.charges > 1 else "")
        else:
            note = f"BOUGHT  ·  ¤ {ab.cost}" + (f"  ·  UP TO {ab.charges}" if ab.charges > 1 else "")
        text(p, QRectF(r.x() + 92 * s, r.y() + 34 * s, r.width() - 100 * s, 18 * s),
             f"[{g.key_label(ab.slot)}]  {note}", 10 * s, rgba(ag.color), AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(r.x() + 92 * s, r.y() + 54 * s, r.width() - 104 * s, 34 * s), ab.desc, 11 * s, DIM,
             AL.AlignLeft | AL.AlignTop | Qt.TextFlag.TextWordWrap, False)
        yy += 98 * s


def option_rows(g) -> list:  # noqa: ANN001
    st = g.settings
    return [("FIELD OF VIEW", f"{st['fov']}°", "fov"),
            ("CROSSHAIR COLOUR", CROSS_COLOURS[st["cross_colour"] % len(CROSS_COLOURS)][0], "cross_colour"),
            ("CROSSHAIR SIZE", str(st["cross_size"]), "cross_size"),
            ("CROSSHAIR GAP", str(st["cross_gap"]), "cross_gap"),
            ("CENTRE DOT", "ON" if st["cross_dot"] else "OFF", "cross_dot"),
            ("CROSSHAIR SHOWS SPREAD", "ON" if st["cross_dynamic"] else "OFF", "cross_dynamic"),
            ("AIM DOWN SIGHTS", "TOGGLE" if st["ads_toggle"] else "HOLD", "ads_toggle"),
            ("CROUCH", "TOGGLE" if st["crouch_toggle"] else "HOLD", "crouch_toggle"),
            ("BUY PHASE", f"{st['buy_time']} S", "buy_time")]


def paint_options(p: QPainter, w: int, h: int, g, back: tuple) -> None:  # noqa: ANN001
    s = h / 780
    ui = g.ui
    _title(p, w, s, "OPTIONS", "Mouse speed and key bindings are in the arcade's Controls panel.")
    x, y = 60 * s, 136 * s
    for label, value, key in option_rows(g):
        r = QRectF(x, y, 560 * s, 42 * s)
        p.fillRect(r, QColor(30, 27, 48, 225))
        text(p, QRectF(x + 16 * s, y, 300 * s, 42 * s), label, 13 * s, PAPER, AL.AlignLeft | AL.AlignVCenter)
        _button(p, ui, QRectF(x + 330 * s, y + 5 * s, 36 * s, 32 * s), "‹", ("opt", key, -1), s)
        text(p, QRectF(x + 370 * s, y, 130 * s, 42 * s), value, 13 * s, ACCENT)
        _button(p, ui, QRectF(x + 504 * s, y + 5 * s, 36 * s, 32 * s), "›", ("opt", key, 1), s)
        y += 48 * s
    y += 10 * s
    _button(p, ui, QRectF(x, y, 270 * s, 50 * s), "CONTROLS…", ("controls",), s, True)
    _button(p, ui, QRectF(x + 290 * s, y, 270 * s, 50 * s), "BACK", back, s, True)
    # a crosshair to look at
    cx, cy = x + 760 * s, 300 * s
    p.fillRect(QRectF(cx - 130 * s, cy - 110 * s, 260 * s, 220 * s), QColor(110, 96, 90))
    p.fillRect(QRectF(cx - 130 * s, cy + 30 * s, 260 * s, 80 * s), QColor(70, 66, 72))
    st = g.settings
    col = QColor(*CROSS_COLOURS[st["cross_colour"] % len(CROSS_COLOURS)][1])
    ln = (4 + st["cross_size"] * 2) * s
    th = max(2.0 * s, 1.5) if st["cross_size"] < 4 else 3.0 * s
    gap = (2 + st["cross_gap"] * 1.5) * s
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if dx:
            r = QRectF(cx + dx * gap if dx > 0 else cx - gap - ln, cy - th / 2, ln, th)
        else:
            r = QRectF(cx - th / 2, cy + dy * gap if dy > 0 else cy - gap - ln, th, ln)
        p.fillRect(r.adjusted(-1, -1, 1, 1), QColor(0, 0, 0, 150))
        p.fillRect(r, col)
    if st["cross_dot"]:
        p.fillRect(QRectF(cx - th / 2, cy - th / 2, th, th), col)


def paint_stats(p: QPainter, w: int, h: int, g) -> None:  # noqa: ANN001
    s = h / 780
    ui = g.ui
    st = g.stats
    _title(p, w, s, "RECORD")
    x, y = 60 * s, 136 * s
    kd = st["kills"] / max(st["deaths"], 1)
    acc = 100 * st["hits"] / max(st["shots"], 1)
    hs = 100 * st["heads"] / max(st["kills"], 1)
    cells = (("MATCHES", st["matches"]), ("WON", st["wins"]), ("LOST", st["losses"]),
             ("ROUNDS WON", st["rounds_won"]), ("ROUNDS LOST", st["rounds_lost"]), ("KILLS", st["kills"]),
             ("DEATHS", st["deaths"]), ("ASSISTS", st["assists"]), ("K / D", f"{kd:.2f}"),
             ("HEADSHOT KILLS", f"{hs:.0f}%"), ("ACCURACY", f"{acc:.0f}%"), ("PLANTS", st["plants"]),
             ("DEFUSES", st["defuses"]), ("MOST KILLS IN A MATCH", st["best_kills"]), ("DAMAGE", f"{st['damage']:,}"))
    cw, chh = 222 * s, 84 * s
    for i, (label, val) in enumerate(cells):
        r = QRectF(x + (i % 5) * (cw + 10 * s), y + (i // 5) * (chh + 10 * s), cw, chh)
        p.fillRect(r, QColor(30, 27, 48, 225))
        text(p, QRectF(r.x() + 14 * s, r.y() + 8 * s, cw, 44 * s), str(val), 30 * s, PAPER,
             AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(r.x() + 14 * s, r.y() + 54 * s, cw, 20 * s), label, 10 * s, DIM, AL.AlignLeft | AL.AlignVCenter)
    y += 3 * (chh + 10 * s) + 16 * s
    text(p, QRectF(x, y, 300 * s, 20 * s), "BY AGENT  (played · won)", 12 * s, ACCENT, AL.AlignLeft | AL.AlignVCenter)
    for i, key in enumerate(C.AGENT_ORDER):
        ag = C.AGENTS[key]
        pl, wn = st["agents"].get(key, [0, 0])
        r = QRectF(x + i * (cw + 10 * s), y + 26 * s, cw, 54 * s)
        p.fillRect(r, QColor(30, 27, 48, 225))
        p.fillRect(QRectF(r.x(), r.y(), 4 * s, r.height()), rgba(ag.color))
        text(p, QRectF(r.x() + 14 * s, r.y(), cw, 54 * s), ag.name, 15 * s, rgba(ag.color),
             AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(r.x(), r.y(), cw - 14 * s, 54 * s), f"{pl} · {wn}", 15 * s, PAPER,
             AL.AlignRight | AL.AlignVCenter)
    y += 96 * s
    text(p, QRectF(x, y, 300 * s, 20 * s), "BY MAP  (played · won)", 12 * s, ACCENT, AL.AlignLeft | AL.AlignVCenter)
    for i, key in enumerate(MAP_ORDER):
        pl, wn = st["maps"].get(key, [0, 0])
        r = QRectF(x + i * (cw + 10 * s), y + 26 * s, cw, 44 * s)
        p.fillRect(r, QColor(30, 27, 48, 225))
        text(p, QRectF(r.x() + 14 * s, r.y(), cw, 44 * s), MAP_INFO[key][0], 13 * s, PAPER,
             AL.AlignLeft | AL.AlignVCenter)
        text(p, QRectF(r.x(), r.y(), cw - 14 * s, 44 * s), f"{pl} · {wn}", 14 * s, PAPER,
             AL.AlignRight | AL.AlignVCenter)
    _button(p, ui, QRectF(x, h - 86 * s, 230 * s, 50 * s), "BACK", ("page", "main"), s, True)


def paint_pause(p: QPainter, w: int, h: int, g) -> None:  # noqa: ANN001
    s = h / 780
    ui = g.ui
    p.fillRect(QRectF(0, 0, w, h), QColor(10, 9, 18, 200))
    text(p, QRectF(0, 150 * s, w, 60 * s), "PAUSED", 46 * s, PAPER)
    m = g.match
    text(p, QRectF(0, 210 * s, w, 24 * s), f"{m.map.name}  ·  {m.score[0]} - {m.score[1]}", 14 * s, DIM)
    rows = (("RESUME", ("resume",)), ("OPTIONS", ("overlay", "options")), ("CONTROLS…", ("controls",)),
            ("LEAVE THE MATCH", ("leave",)))
    y = 270 * s
    for label, action in rows:
        _button(p, ui, QRectF(w / 2 - 170 * s, y, 340 * s, 54 * s), label, action, s, True)
        y += 64 * s
    tips = ("Stand still to shoot straight. The crosshair opens when you move.",
            "Shift walks without a sound. Running is heard through walls.",
            "Thin boards, crates and barricades can be shot through; brick cannot.",
            "A defuse keeps its first half if you have to let go.")
    text(p, QRectF(0, h - 90 * s, w, 22 * s), tips[int(g.time / 5) % len(tips)], 12 * s, ACCENT)


def paint_loading(p: QPainter, w: int, h: int, g, ready: bool) -> None:  # noqa: ANN001
    s = h / 780
    backdrop(p, w, h, g.time)
    key = g.choice["map"]
    name, blurb = MAP_INFO[key]
    ag = C.AGENTS[g.choice["agent"]]
    text(p, QRectF(60 * s, 150 * s, w, 80 * s), name, 72 * s, PAPER, AL.AlignLeft | AL.AlignVCenter)
    text(p, QRectF(64 * s, 236 * s, w * 0.5, 50 * s), blurb, 14 * s, DIM,
         AL.AlignLeft | AL.AlignTop | Qt.TextFlag.TextWordWrap, False)
    text(p, QRectF(64 * s, 320 * s, w, 30 * s), f"YOU ARE  {ag.name}  ·  {ag.role.upper()}", 18 * s, rgba(ag.color),
         AL.AlignLeft | AL.AlignVCenter)
    lines = ("Attackers carry the charge to site A or B, plant it and keep it alive until it goes off.",
             "Defenders stop the plant, or defuse the charge in time.",
             "One life a round. Credits buy guns, armour and abilities before each round.",
             f"First to {g.choice['rounds']} rounds. Sides swap after round {g.choice['rounds'] - 1}.")
    for i, ln in enumerate(lines):
        text(p, QRectF(64 * s, (380 + i * 26) * s, w, 22 * s), "·  " + ln, 12.5 * s, PAPER,
             AL.AlignLeft | AL.AlignVCenter, False)
    dots = "." * (1 + int(g.time * 3) % 3)
    text(p, QRectF(64 * s, h - 110 * s, w, 30 * s), "READY" if ready else f"BUILDING THE MAP{dots}", 16 * s, ACCENT,
         AL.AlignLeft | AL.AlignVCenter)
    _ = (ALLY, ENEMY, math, minimap_pix, weapon_pix)
