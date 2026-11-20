"""The home screen's toys and effects: pixel sprites (critters and mascots), confetti, tickets flying into the counter,
floating texts, the animated title logo, the ticket counter with its rolling digits, the LED ticker, the CRT overlay,
and the insert-coin launch sequence played over a starting game (``LaunchIntro``)."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, \
    QPolygonF, QRadialGradient, QTransform
from PySide6.QtWidgets import QWidget

from app.arcade import pixelfont
from app.arcade.home_scene import neon_text_pixmap

_rng = random.Random()  # the arcade's own (the program's random numbers are not touched)
NEON = ("#ff3fb4", "#29f3ff", "#ffe14d", "#7dff4a", "#a66bff", "#ff8a1f")


def ease_out_back(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    c = 1.70158
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def ease_out_cubic(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_in_cubic(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x ** 3


# --- pixel sprites -------------------------------------------------------------------------------------------------

SPRITES = {
    "invader": (
        ("..X.....X..", "...X...X...", "..XXXXXXX..", ".XX.XXX.XX.", "XXXXXXXXXXX", "X.XXXXXXX.X", "X.X.....X.X",
         "...XX.XX..."),
        ("..X.....X..", "X..X...X..X", "X.XXXXXXX.X", "XXX.XXX.XXX", "XXXXXXXXXXX", ".XXXXXXXXX.", "..X.....X..",
         ".X.......X."),
    ),
    "squid": (
        ("...XX...", "..XXXX..", ".XXXXXX.", "XX.XX.XX", "XXXXXXXX", "..X..X..", ".X.XX.X.", "X.X..X.X"),
        ("...XX...", "..XXXX..", ".XXXXXX.", "XX.XX.XX", "XXXXXXXX", ".X.XX.X.", "X......X", ".X....X."),
    ),
    "ghost": (
        ("....XXXXXX....", "..XXXXXXXXXX..", ".XXXXXXXXXXXX.", ".XXWWXXXXWWXX.", "XXWWWWXXWWWWXX",
         "XXWWBBXXWWBBXX", "XXWWBBXXWWBBXX", "XXXWWXXXXWWXXX", "XXXXXXXXXXXXXX", "XXXXXXXXXXXXXX",
         "XXXXXXXXXXXXXX", "XXXXXXXXXXXXXX", "XX.XXX..XXX.XX", "X...XX..XX...X"),
        ("....XXXXXX....", "..XXXXXXXXXX..", ".XXXXXXXXXXXX.", ".XXWWXXXXWWXX.", "XXWWWWXXWWWWXX",
         "XXWWBBXXWWBBXX", "XXWWBBXXWWBBXX", "XXXWWXXXXWWXXX", "XXXXXXXXXXXXXX", "XXXXXXXXXXXXXX",
         "XXXXXXXXXXXXXX", "XXXXXXXXXXXXXX", "XXXX.XXXX.XXXX", ".XX...XX...XX."),
    ),
    "cat": (
        (".X........X.", "XOX......XOX", "XOOX....XOOX", "XOOOXXXXOOOX", "XOOOOOOOOOOX", "XOGGOOOOGGOX",
         "XOGKOOOOGKOX", "XOOOOPPOOOOX", "XROOOOOOOORX", ".XOOOWWOOOX.", "..XXXXXXXX..", ".PP......PP."),
        (".X........X.", "XOX......XOX", "XOOX....XOOX", "XOOOXXXXOOOX", "XOOOOOOOOOOX", "XOOOOOOOOOOX",
         "XOKKOOOOKKOX", "XOOOOPPOOOOX", "XROOOOOOOORX", ".XOOOWWOOOX.", "..XXXXXXXX..", ".PP......PP."),
    ),
}
PALETTES = {
    "invader": {"X": "#7dff4a"}, "squid": {"X": "#29f3ff"},
    "ghost": {"X": "#ff5ea8", "W": "#ffffff", "B": "#2a3cff"},
    "ghost_red": {"X": "#ff3030", "W": "#ffffff", "B": "#2a3cff"},
    "cat": {"X": "#2b1606", "O": "#ff9a2e", "G": "#7dff4a", "K": "#111111", "P": "#ff7ab8", "R": "#ff7ab8",
            "W": "#ffffff"},
}
_sprite_cache: dict = {}


def sprite_image(kind: str, frame: int = 0, palette: str | None = None) -> QImage:
    key = (kind, frame % 2, palette)
    image = _sprite_cache.get(key)
    if image is None:
        rows = SPRITES[kind][frame % len(SPRITES[kind])]
        colors = {k: QColor(v) for k, v in PALETTES[palette or kind].items()}
        image = QImage(len(rows[0]), len(rows), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch in colors:
                    image.setPixelColor(x, y, colors[ch])
        _sprite_cache[key] = image
    return image


def draw_sprite(p: QPainter, kind: str, centre: QPointF, px: float, frame: int = 0, palette: str | None = None,
                glow: QColor | None = None, flip: bool = False) -> QRectF:
    image = sprite_image(kind, frame, palette)
    w, h = image.width() * px, image.height() * px
    rect = QRectF(centre.x() - w / 2, centre.y() - h / 2, w, h)
    if glow is not None:
        g = QRadialGradient(centre, max(w, h) * 0.9)
        c = QColor(glow)
        c.setAlpha(70)
        g.setColorAt(0.0, c)
        c.setAlpha(0)
        g.setColorAt(1.0, c)
        p.fillRect(rect.adjusted(-w * 0.4, -h * 0.4, w * 0.4, h * 0.4), g)
    smooth = p.testRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    if flip:
        p.drawImage(rect, image.flipped(Qt.Orientation.Horizontal))
    else:
        p.drawImage(rect, image)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
    return rect


def speech_bubble(p: QPainter, text: str, anchor: QPointF, px: float, alpha: float = 1.0,
                  color: str = "#ffffff", side: bool = False) -> None:
    """A pixel-text speech bubble whose tail points at ``anchor`` (below it; ``side``: the bubble up and to the
    right of it)."""
    tw = pixelfont.width(text, px)
    if side:
        rect = QRectF(anchor.x() + px * 6, anchor.y() - px * 16, tw + px * 8, px * 13)
        tail = QPolygonF([QPointF(rect.left() + 1, rect.bottom() - px * 7), QPointF(rect.left() + 1,
                                                                                  rect.bottom() - px * 2),
                          QPointF(anchor.x() + px * 2, anchor.y() - px)])
    else:
        rect = QRectF(anchor.x() - tw / 2 - px * 4, anchor.y() - px * 18, tw + px * 8, px * 13)
        tail = QPolygonF([QPointF(anchor.x() - px * 3, rect.bottom() - 1),
                          QPointF(anchor.x() + px * 3, rect.bottom() - 1), QPointF(anchor.x(), anchor.y() - px)])
    p.setOpacity(alpha)
    path = QPainterPath()
    path.addRoundedRect(rect, px * 3, px * 3)
    path.addPolygon(tail)
    path = path.simplified()
    p.setPen(QPen(QColor("#1b0b33"), px))
    p.setBrush(QColor(color))
    p.drawPath(path)
    pixelfont.draw(p, text, rect.center().x(), rect.top() + px * 3, px, "#1b0b33", Qt.AlignmentFlag.AlignHCenter)
    p.setOpacity(1.0)


# --- confetti, floating text, flying tickets -----------------------------------------------------------------------

@dataclass
class Bit:
    x: float
    y: float
    vx: float
    vy: float
    color: QColor
    size: float
    spin: float
    life: float
    age: float = 0.0


class Confetti:
    """Paper confetti: bursts and rain."""

    MAX = 420

    def __init__(self) -> None:
        self.bits: list[Bit] = []

    def burst(self, at: QPointF, n: int = 60, speed: float = 520.0, scale: float = 1.0,
              colors: tuple[str, ...] = NEON + ("#ffffff",)) -> None:
        for _ in range(n):
            a = _rng.uniform(-math.pi * 0.95, -math.pi * 0.05)
            v = _rng.uniform(0.3, 1.0) * speed * scale
            self._add(Bit(at.x(), at.y(), math.cos(a) * v, math.sin(a) * v, QColor(_rng.choice(colors)),
                          _rng.uniform(5, 10) * scale, _rng.uniform(4, 14), _rng.uniform(1.6, 3.0)))

    def rain(self, w: float, n: int, scale: float = 1.0) -> None:
        for _ in range(n):
            self._add(Bit(_rng.uniform(0, w), _rng.uniform(-40, -5) * scale, _rng.uniform(-40, 40) * scale,
                          _rng.uniform(60, 160) * scale, QColor(_rng.choice(NEON + ("#ffffff",))),
                          _rng.uniform(5, 10) * scale, _rng.uniform(4, 14), _rng.uniform(3.0, 5.5)))

    def _add(self, bit: Bit) -> None:
        if len(self.bits) < self.MAX:
            self.bits.append(bit)

    def update(self, dt: float, scale: float = 1.0) -> None:
        keep = []
        for b in self.bits:
            b.age += dt
            if b.age >= b.life:
                continue
            b.vy += 600 * scale * dt
            b.vx *= 1 - 1.6 * dt
            b.vy = min(b.vy, 170 * scale + 40 * math.sin(b.age * 7 + b.spin))
            b.x += (b.vx + 30 * scale * math.sin(b.age * 5 + b.spin)) * dt
            b.y += b.vy * dt
            keep.append(b)
        self.bits = keep

    def paint(self, p: QPainter) -> None:
        for b in self.bits:
            flip = abs(math.cos(b.age * b.spin))
            fade = min(1.0, (b.life - b.age) / 0.6)
            c = b.color if fade >= 1.0 else QColor(b.color.red(), b.color.green(), b.color.blue(), int(255 * fade))
            w = b.size * (0.25 + 0.75 * flip)
            p.fillRect(QRectF(b.x - w / 2, b.y - b.size * 0.3, w, b.size * 0.6), c)

    def __len__(self) -> int:
        return len(self.bits)


@dataclass
class FloatText:
    text: str
    x: float
    y: float
    color: str
    px: float
    age: float = 0.0
    life: float = 1.6
    rise: float = 40.0  # pixels a second (at the reference size)


@dataclass
class FlyingTicket:
    start: QPointF
    ctrl: QPointF
    end: QPointF
    delay: float
    value: int
    dur: float = 0.75
    age: float = 0.0
    landed: bool = False


@dataclass
class Effects:
    """Short-lived things over the home screen."""

    confetti: Confetti = field(default_factory=Confetti)
    texts: list[FloatText] = field(default_factory=list)
    tickets: list[FlyingTicket] = field(default_factory=list)

    def text(self, text: str, at: QPointF, color: str = "#ffe14d", px: float = 3.0, life: float = 1.6,
             rise: float = 40.0) -> None:
        self.texts.append(FloatText(text, at.x(), at.y(), color, px, 0.0, life, rise))
        del self.texts[:-8]

    def send_tickets(self, n: int, start: QPointF, end: QPointF, spread: float = 30.0) -> None:
        count = max(1, min(n, 12))
        base, extra = divmod(n, count)
        for i in range(count):
            mid = QPointF((start.x() + end.x()) / 2 + _rng.uniform(-spread, spread) * 3,
                          min(start.y(), end.y()) - _rng.uniform(40, 160))
            s = QPointF(start.x() + _rng.uniform(-spread, spread), start.y() + _rng.uniform(-spread, spread) * 0.4)
            self.tickets.append(FlyingTicket(s, mid, end, i * 0.06, base + (1 if i < extra else 0),
                                             _rng.uniform(0.6, 0.85)))

    def update(self, dt: float, scale: float = 1.0) -> int:
        """Advance; returns the tickets that landed in the counter this step."""
        self.confetti.update(dt, scale)
        for t in self.texts:
            t.age += dt
            t.y -= t.rise * scale * dt
        self.texts = [t for t in self.texts if t.age < t.life]
        landed = 0
        for f in self.tickets:
            f.age += dt
            if not f.landed and f.age >= f.delay + f.dur:
                f.landed = True
                landed += f.value
        self.tickets = [f for f in self.tickets if not f.landed]
        return landed

    def pending_tickets(self) -> int:
        return sum(f.value for f in self.tickets)

    def busy(self) -> bool:
        return bool(len(self.confetti) or self.texts or self.tickets)

    def paint(self, p: QPainter, unit: float) -> None:
        for f in self.tickets:
            k = (f.age - f.delay) / f.dur
            if k < 0:
                continue
            k = ease_in_cubic(k) * 0.6 + k * 0.4
            a, b, c = f.start, f.ctrl, f.end
            x = (1 - k) ** 2 * a.x() + 2 * (1 - k) * k * b.x() + k * k * c.x()
            y = (1 - k) ** 2 * a.y() + 2 * (1 - k) * k * b.y() + k * k * c.y()
            draw_ticket_icon(p, QPointF(x, y), 13 * unit * (1.15 - 0.4 * k), f.age * 9)
        self.confetti.paint(p)
        for t in self.texts:
            k = t.age / t.life
            p.setOpacity(1.0 if k < 0.7 else max(0.0, (1 - k) / 0.3))
            pop = 1 + 0.35 * max(0.0, 1 - t.age / 0.15)
            pixelfont.draw(p, t.text, t.x, t.y, t.px * pop, t.color, Qt.AlignmentFlag.AlignHCenter, glow=True)
        p.setOpacity(1.0)


def draw_ticket_icon(p: QPainter, c: QPointF, w: float, angle: float = 0.0) -> None:
    """A little orange arcade ticket."""
    p.save()
    p.translate(c)
    p.rotate(math.degrees(math.sin(angle) * 0.5))
    h = w * 0.55
    path = QPainterPath()
    path.addRoundedRect(QRectF(-w / 2, -h / 2, w, h), w * 0.08, w * 0.08)
    for sx in (-w / 2, w / 2):
        notch = QPainterPath()
        notch.addEllipse(QPointF(sx, 0), h * 0.22, h * 0.22)
        path = path.subtracted(notch)
    p.setPen(QPen(QColor("#7a2a00"), max(w * 0.05, 1)))
    p.setBrush(QColor("#ff9a2e"))
    p.drawPath(path)
    p.setPen(QPen(QColor(122, 42, 0, 150), max(w * 0.04, 1), Qt.PenStyle.DotLine))
    p.drawLine(QPointF(-w * 0.22, -h * 0.4), QPointF(-w * 0.22, h * 0.4))
    p.restore()


# --- critters ------------------------------------------------------------------------------------------------------

@dataclass
class Critter:
    kind: str  # invader / squid / ghost / chomper / ufo
    x: float
    y: float
    vx: float
    hp: int
    value: int
    age: float = 0.0
    hit_flash: float = 0.0
    hop: float = 0.0
    dead: float = -1.0  # seconds since caught (-1: alive)
    seed: float = field(default_factory=_rng.random)

    @property
    def alive(self) -> bool:
        return self.dead < 0


class CritterPen:
    """Critters that wander over the home screen now and then; clicking them pays tickets."""

    KINDS = (("invader", 1, 4, 0.34), ("squid", 1, 5, 0.22), ("ghost", 2, 8, 0.22), ("chomper", 2, 8, 0.16),
             ("ufo", 3, 25, 0.06))

    def __init__(self, first_delay: float = 6.0) -> None:
        self.critters: list[Critter] = []
        self.next_spawn = first_delay

    def update(self, dt: float, w: float, h: float, lane_y: float, sky_y: float, unit: float,
               allow_spawn: bool = True) -> None:
        self.next_spawn -= dt
        if allow_spawn and self.next_spawn <= 0 and sum(c.alive for c in self.critters) < 2:
            self.spawn(w, lane_y, sky_y, unit)
            self.next_spawn = _rng.uniform(14, 30)
        for c in self.critters:
            c.age += dt
            c.hit_flash = max(0.0, c.hit_flash - dt)
            if not c.alive:
                c.dead += dt
                continue
            speed = c.vx * (1.8 if c.hp < self._hp(c.kind) else 1.0)
            c.x += speed * dt
            if c.kind == "ufo":
                c.y = sky_y + math.sin(c.age * 2.3 + c.seed * 6) * 26 * unit
            elif c.kind in ("ghost",):
                c.y = lane_y - 14 * unit + math.sin(c.age * 3 + c.seed * 6) * 8 * unit
            else:
                c.hop = abs(math.sin(c.age * 6)) * 10 * unit
                c.y = lane_y
        self.critters = [c for c in self.critters if (c.alive and -120 * unit < c.x < w + 120 * unit)
                         or (not c.alive and c.dead < 0.6)]

    def _hp(self, kind: str) -> int:
        return next(k[1] for k in self.KINDS if k[0] == kind)

    def spawn(self, w: float, lane_y: float, sky_y: float, unit: float, kind: str | None = None) -> Critter:
        if kind is None:
            r, acc = _rng.random(), 0.0
            kind = self.KINDS[0][0]
            for name, _hp, _v, weight in self.KINDS:
                acc += weight
                if r <= acc:
                    kind = name
                    break
        _name, hp, value, _w = next(k for k in self.KINDS if k[0] == kind)
        left = _rng.random() < 0.5
        speed = _rng.uniform(70, 120) * unit * (1.6 if kind == "ufo" else 1.0)
        c = Critter(kind, -60 * unit if left else w + 60 * unit, sky_y if kind == "ufo" else lane_y,
                    speed if left else -speed, hp, value)
        self.critters.append(c)
        return c

    def rect(self, c: Critter, unit: float) -> QRectF:
        s = {"ufo": 34, "chomper": 24, "ghost": 28}.get(c.kind, 24) * unit
        return QRectF(c.x - s, c.y - c.hop - s * 0.9, 2 * s, 1.8 * s)

    def hit(self, pos: QPointF, unit: float) -> tuple[Critter, bool] | None:
        """The critter under ``pos`` takes a hit: (critter, caught) or None."""
        for c in reversed(self.critters):
            if c.alive and self.rect(c, unit).adjusted(-8 * unit, -8 * unit, 8 * unit, 8 * unit).contains(pos):
                c.hp -= 1
                c.hit_flash = 0.15
                c.vx = -c.vx if c.kind != "ufo" else c.vx * 1.3  # it runs the other way
                if c.hp <= 0:
                    c.dead = 0.0
                    return c, True
                return c, False
        return None

    def paint(self, p: QPainter, t: float, unit: float) -> None:
        for c in self.critters:
            centre = QPointF(c.x, c.y - c.hop)
            if not c.alive:
                k = c.dead / 0.6
                p.setOpacity(1 - k)
                r = (14 + 40 * k) * unit
                p.setPen(QPen(QColor("#ffe14d"), 3 * unit))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(centre, r, r)
                p.setOpacity(1.0)
                continue
            frame = int(c.age * 4) % 2
            px = 3.6 * unit
            flip = c.vx < 0
            if c.hit_flash > 0:
                p.setOpacity(0.4)
            if c.kind in ("invader", "squid"):
                draw_sprite(p, c.kind, centre, px, frame, glow=QColor(PALETTES[c.kind]["X"]))
            elif c.kind == "ghost":
                draw_sprite(p, "ghost", centre, px * 0.85, frame, "ghost_red", glow=QColor("#ff3030"), flip=flip)
            elif c.kind == "chomper":
                self._chomper(p, centre, 21 * unit, c.age, flip)
            else:
                self._ufo(p, centre, 32 * unit, c.age)
            p.setOpacity(1.0)
            # a shadow on the floor
            if c.kind != "ufo":
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0, 90))
                p.drawEllipse(QPointF(c.x, c.y + 20 * unit), 16 * unit * (1 - c.hop / (30 * unit)), 4 * unit)

    @staticmethod
    def _chomper(p: QPainter, c: QPointF, r: float, age: float, flip: bool) -> None:
        mouth = 8 + 38 * abs(math.sin(age * 10))
        p.setPen(Qt.PenStyle.NoPen)
        g = QRadialGradient(c - QPointF(r * 0.3, r * 0.4), r * 1.3)
        g.setColorAt(0.0, QColor("#fff27a"))
        g.setColorAt(1.0, QColor("#ffc400"))
        p.setBrush(g)
        start = (180 + mouth) if flip else mouth
        p.drawPie(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), int(start * 16), int((360 - 2 * mouth) * 16))
        p.setBrush(QColor("#111"))
        p.drawEllipse(QPointF(c.x() + (-r * 0.15 if flip else r * 0.15), c.y() - r * 0.5), r * 0.12, r * 0.12)

    @staticmethod
    def _ufo(p: QPainter, c: QPointF, r: float, age: float) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        beam = QPolygonF([c + QPointF(-r * 0.3, r * 0.2), c + QPointF(r * 0.3, r * 0.2), c + QPointF(r * 0.9, r * 3),
                          c + QPointF(-r * 0.9, r * 3)])
        g = QLinearGradient(c, c + QPointF(0, r * 3))
        g.setColorAt(0.0, QColor(255, 240, 120, 90))
        g.setColorAt(1.0, QColor(255, 240, 120, 0))
        p.setBrush(g)
        p.drawPolygon(beam)
        p.setBrush(QColor(140, 255, 240, 200))
        p.drawEllipse(QPointF(c.x(), c.y() - r * 0.25), r * 0.45, r * 0.4)
        sg = QLinearGradient(c - QPointF(0, r * 0.3), c + QPointF(0, r * 0.3))
        sg.setColorAt(0.0, QColor("#fff3a6"))
        sg.setColorAt(1.0, QColor("#c88a12"))
        p.setBrush(sg)
        p.drawEllipse(c, r, r * 0.32)
        for i in range(5):
            on = int(age * 8 + i) % 2 == 0
            p.setBrush(QColor("#ff3fb4") if on else QColor("#29f3ff"))
            p.drawEllipse(QPointF(c.x() + (i - 2) * r * 0.38, c.y() + r * 0.05), r * 0.08, r * 0.08)


# --- title logo ----------------------------------------------------------------------------------------------------

class TitleLogo:
    """'Break Room' in pink neon script over a chrome 'ARCADE'; the letters pop in, a shine sweeps across. Everything
    it draws (the animation included) stays inside ``bounds``."""

    BOB = 0.025  # of the height, up and down
    OVERSHOOT = 0.06  # the pop-in grows a little past full size

    def __init__(self, height: float) -> None:
        self.height = max(height, 20.0)
        size = self.height * 0.78
        f = QFont("Impact")
        f.setPixelSize(int(size))
        f.setItalic(True)
        self.letters: list[tuple[QPixmap, float]] = []
        x = 0.0
        for ch in "ARCADE":
            path = QPainterPath()
            path.addText(0, 0, f, ch)
            box = path.boundingRect()
            pm = self._chrome_letter(path, box)
            self.letters.append((pm, x))
            x += box.width() * 1.02 + size * 0.02
        self.word_w = x
        self.letter_h = self.letters[0][0].height() if self.letters else 0
        self.script = neon_text_pixmap("Break Room", self.height * 0.42, QColor("#ff3fb4"), "Segoe Script", pad=0.24)
        self.script_dark = neon_text_pixmap("Break Room", self.height * 0.42, QColor("#ff3fb4"), "Segoe Script",
                                            lit=False, pad=0.24)

    def _chrome_letter(self, path: QPainterPath, box: QRectF) -> QPixmap:
        pad = self.height * 0.12
        pm = QPixmap(int(box.width() + 2 * pad) + 2, int(box.height() + 2 * pad) + 2)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(pad - box.x(), pad - box.y())
        top, bottom = box.top(), box.bottom()
        for wdt, a in ((self.height * 0.16, 40), (self.height * 0.1, 70)):
            p.setPen(QPen(QColor(41, 243, 255, a), wdt, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin))
            p.drawPath(path)
        p.setPen(QPen(QColor("#12002a"), self.height * 0.06, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.drawPath(path)
        g = QLinearGradient(0, top, 0, bottom)
        g.setColorAt(0.0, QColor("#ffffff"))
        g.setColorAt(0.25, QColor("#9fe8ff"))
        g.setColorAt(0.5, QColor("#1d3f9e"))
        g.setColorAt(0.52, QColor("#fff1d6"))
        g.setColorAt(0.75, QColor("#ff9a3c"))
        g.setColorAt(1.0, QColor("#ff2f8e"))
        p.setPen(QPen(QColor("#ffffff"), max(self.height * 0.012, 1)))
        p.setBrush(g)
        p.drawPath(path)
        p.end()
        return pm

    def size(self) -> tuple[float, float]:
        b = self.bounds(0.0, 0.0)
        return b.width(), b.height()

    def _script_transform(self, x0: float, top: float) -> QTransform:
        tf = QTransform()
        tf.translate(x0 + self.word_w * 0.02, top + self.height * 0.2)
        tf.rotate(-7)
        return tf

    def _script_rect(self) -> QRectF:
        pm = self.script
        return QRectF(-pm.width() * 0.12, -pm.height() * 0.55, pm.width(), pm.height())

    def bounds(self, centre_x: float, top: float) -> QRectF:
        """Where the logo draws when painted at (``centre_x``, ``top``), whatever the animation does."""
        x0 = centre_x - self.word_w / 2
        pad = self.height * 0.12
        grow = self.letter_h * self.OVERSHOOT / 2 + self.height * self.BOB
        letters = QRectF(x0 - pad, top + self.height * 0.3 - pad, self.word_w + 2 * pad, self.letter_h)
        letters = letters.adjusted(-grow, -grow, grow, grow)
        return letters.united(self._script_transform(x0, top).mapRect(self._script_rect()))

    def paint(self, p: QPainter, centre_x: float, top: float, t: float, intro: float, party: float = 0.0) -> None:
        """``intro``: seconds since the arcade appeared (the letters drop in)."""
        x0 = centre_x - self.word_w / 2
        base_y = top + self.height * 0.3
        pad = self.height * 0.12
        for i, (pm, x) in enumerate(self.letters):
            k = (intro - 0.12 * i) / 0.55
            if k <= 0:
                continue
            grow = min(ease_out_back(k), 1 + self.OVERSHOOT)  # pops in where it stands (never past the bounds)
            bob = math.sin(t * 2.2 + i * 0.7) * self.height * self.BOB
            p.setOpacity(min(1.0, k * 3))
            cx, cy = x0 + x - pad + pm.width() / 2, base_y + bob - pad + pm.height() / 2
            w, h = pm.width() * grow, pm.height() * grow
            p.drawPixmap(QRectF(cx - w / 2, cy - h / 2, w, h), pm, QRectF(pm.rect()))
        p.setOpacity(1.0)
        # the shine every 6 s
        sweep = (t % 6.0) / 0.9
        if 0 < sweep < 1 and intro > 1.2:
            sx = x0 - self.height + (self.word_w + 2 * self.height) * sweep
            g = QLinearGradient(QPointF(sx - self.height * 0.4, 0), QPointF(sx + self.height * 0.4, 0))
            g.setColorAt(0.0, QColor(255, 255, 255, 0))
            g.setColorAt(0.5, QColor(255, 255, 255, 120))
            g.setColorAt(1.0, QColor(255, 255, 255, 0))
            p.save()
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            band = QPolygonF([QPointF(sx - self.height * 0.2, base_y), QPointF(sx + self.height * 0.3, base_y),
                              QPointF(sx + self.height * 0.05, base_y + self.letter_h - 2 * pad),
                              QPointF(sx - self.height * 0.45, base_y + self.letter_h - 2 * pad)])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(g)
            p.drawPolygon(band)
            p.restore()
        # the neon script, tilted over the top left of the word
        k = (intro - 0.9) / 0.4
        if k > 0:
            flick = int(t * 13) % 37 == 0 or (intro < 1.6 and int(intro * 20) % 3 == 0)
            pm = self.script_dark if flick else self.script
            p.save()
            p.setTransform(self._script_transform(x0, top), True)
            if party > 0:
                p.setOpacity(1.0)
                hue = (t * 0.5) % 1.0
                tint = QPixmap(pm.size())
                tint.fill(Qt.GlobalColor.transparent)
                q = QPainter(tint)
                q.drawPixmap(0, 0, pm)
                q.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceAtop)
                q.fillRect(tint.rect(), QColor.fromHsvF(hue, 0.8, 1.0, 0.55))
                q.end()
                pm = tint
            p.setOpacity(min(1.0, k))
            p.drawPixmap(QPointF(-pm.width() * 0.12, -pm.height() * 0.55), pm)
            p.restore()
            p.setOpacity(1.0)


# --- the ticket counter --------------------------------------------------------------------------------------------

class TicketCounter:
    """The ticket stub top left: an odometer that rolls up to the balance."""

    def __init__(self) -> None:
        self.shown = 0.0
        self.target = 0
        self.bump = 0.0
        self._shape: tuple[tuple[int, int], QPixmap] | None = None
        self._still: tuple[tuple, QPixmap] | None = None

    def set(self, value: int, instant: bool = False) -> None:
        self.target = int(value)
        if instant:
            self.shown = float(value)

    def update(self, dt: float) -> int:
        """Roll towards the target; returns the whole tickets passed (for the tick sound)."""
        before = int(self.shown)
        diff = self.target - self.shown
        if abs(diff) < 0.01:
            self.shown = float(self.target)
        else:
            step = max(abs(diff) * 6 * dt, 14 * dt)
            self.shown += math.copysign(min(step, abs(diff)), diff)
        self.bump = max(0.0, self.bump - dt * 3)
        return abs(int(self.shown) - before)

    def _stub(self, w: float, h: float) -> QPixmap:
        key = (int(w), int(h))
        if self._shape is not None and self._shape[0] == key:
            return self._shape[1]
        pm = QPixmap(key[0] + 2, key[1] + 2)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(1, 1, w, h)
        path = QPainterPath()
        path.addRoundedRect(r, h * 0.12, h * 0.12)
        for y in (r.center().y(),):
            for x in (r.left(), r.right()):
                notch = QPainterPath()
                notch.addEllipse(QPointF(x, y), h * 0.16, h * 0.16)
                path = path.subtracted(notch)
        g = QLinearGradient(r.topLeft(), r.bottomLeft())
        g.setColorAt(0.0, QColor("#ffb54d"))
        g.setColorAt(1.0, QColor("#ff7a1a"))
        p.setPen(QPen(QColor("#7a2a00"), max(h * 0.04, 1.5)))
        p.setBrush(g)
        p.drawPath(path)
        p.setPen(QPen(QColor(122, 42, 0, 160), max(h * 0.03, 1), Qt.PenStyle.DashLine))
        x = r.left() + h * 0.98
        p.drawLine(QPointF(x, r.top() + h * 0.12), QPointF(x, r.bottom() - h * 0.12))
        px = h * 0.6 / 29
        cx = r.left() + h * 0.5
        pixelfont.draw(p, "ADMIT", cx, r.top() + h * 0.24, px, "#7a2a00", Qt.AlignmentFlag.AlignHCenter)
        pixelfont.draw(p, "ONE", cx, r.top() + h * 0.24 + px * 10, px, "#7a2a00", Qt.AlignmentFlag.AlignHCenter)
        pixelfont.draw(p, "★", cx, r.top() + h * 0.24 + px * 20, px, "#7a2a00", Qt.AlignmentFlag.AlignHCenter)
        p.end()
        self._shape = (key, pm)
        return pm

    def paint(self, p: QPainter, rect: QRectF, label: str = "TICKETS") -> None:
        still = self.bump <= 0 and self.shown == self.target
        if still:
            key = (int(self.shown), int(rect.width()), int(rect.height()), label)
            if self._still is None or self._still[0] != key:
                pm = QPixmap(int(rect.width()) + 4, int(rect.height()) + 4)
                pm.fill(Qt.GlobalColor.transparent)
                q = QPainter(pm)
                q.setRenderHint(QPainter.RenderHint.Antialiasing)
                self._paint(q, QRectF(2, 2, rect.width(), rect.height()), label)
                q.end()
                self._still = (key, pm)
            p.drawPixmap(rect.topLeft() - QPointF(2, 2), self._still[1])
            return
        self._paint(p, rect, label)

    def _paint(self, p: QPainter, rect: QRectF, label: str) -> None:
        scale = 1 + 0.08 * math.sin(self.bump * math.pi) if self.bump > 0 else 1.0
        p.save()
        p.translate(rect.center())
        p.scale(scale, scale)
        p.translate(-rect.center())
        p.drawPixmap(rect.topLeft() - QPointF(1, 1), self._stub(rect.width(), rect.height()))
        h = rect.height()
        win = QRectF(rect.left() + h * 1.1, rect.top() + h * 0.13, rect.width() - h * 1.3, h * 0.52)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#1a0800"))
        p.drawRoundedRect(win, h * 0.06, h * 0.06)
        digits = 6
        value = max(self.shown, 0.0)
        cell = win.width() / digits
        px = min(cell / 6.4, win.height() / 9)
        p.save()
        p.setClipRect(win)
        frac = value - math.floor(value)
        for i in range(digits):
            place = 10 ** (digits - 1 - i)
            d = int(value // place) % 10
            lower = value % place if place > 1 else 0.0
            rolling = place == 1 or lower >= place - 1  # this digit turns while the lower ones pass 9
            off = frac if rolling else 0.0
            if int(value) < place and place > 1 and d == 0 and not rolling:
                color = "#4a2410"
            else:
                color = "#ffd27a"
            cx = win.left() + cell * (i + 0.5)
            y = win.center().y() - px * 3.5 - off * win.height()
            pixelfont.draw(p, str(d), cx, y, px, color, Qt.AlignmentFlag.AlignHCenter, glow=True)
            if off > 0:
                pixelfont.draw(p, str((d + 1) % 10), cx, y + win.height(), px, color, Qt.AlignmentFlag.AlignHCenter,
                               glow=True)
        p.restore()
        pixelfont.draw(p, label, win.center().x(), win.bottom() + h * 0.075, h * 0.032, "#7a2a00",
                       Qt.AlignmentFlag.AlignHCenter)
        p.restore()


# --- the LED ticker ------------------------------------------------------------------------------------------------

class LedTicker:
    """A scrolling dot-matrix sign."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self._strip: QImage | None = None
        self._key = None
        self._grid: tuple[tuple, QPixmap] | None = None

    def set_messages(self, messages: list[tuple[str, str]]) -> None:
        if messages != self.messages:
            self.messages = list(messages)
            self._strip = None

    def _build(self, cell: int) -> QImage:
        key = (cell, tuple(self.messages))
        if self._strip is not None and self._key == key:
            return self._strip
        gap = "      "
        parts = []
        for text, color in self.messages:
            parts.append(pixelfont.text_image(text + gap, color, cell, dots=True))
        width = sum(img.width() for img in parts) or 1
        strip = QImage(width, 7 * cell, QImage.Format.Format_ARGB32_Premultiplied)
        strip.fill(Qt.GlobalColor.transparent)
        p = QPainter(strip)
        x = 0
        for img in parts:
            p.drawImage(x, 0, img)
            x += img.width()
        p.end()
        self._strip, self._key = strip, key
        return strip

    def _unlit(self, w: int, h: int, cell: int) -> QPixmap:
        key = (w, h, cell)
        if self._grid is not None and self._grid[0] == key:
            return self._grid[1]
        tile = QPixmap(cell, cell)
        tile.fill(Qt.GlobalColor.transparent)
        q = QPainter(tile)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        q.setPen(Qt.PenStyle.NoPen)
        q.setBrush(QColor(255, 80, 40, 26))
        r = cell * 0.36
        q.drawEllipse(QPointF(cell / 2, cell / 2), r, r)
        q.end()
        pm = QPixmap(w, h)
        pm.fill(Qt.GlobalColor.transparent)
        q = QPainter(pm)
        q.fillRect(QRectF(0, 0, w, h), QBrush(tile))
        q.end()
        self._grid = (key, pm)
        return pm

    def paint(self, p: QPainter, rect: QRectF, t: float, speed: float = 70.0) -> None:
        cell = max(int(rect.height() / 10), 2)
        p.setPen(QPen(QColor("#3a2f45"), 2))
        p.setBrush(QColor("#08040c"))
        p.drawRoundedRect(rect, 6, 6)
        inner = QRectF(rect.left() + cell, rect.top() + (rect.height() - 7 * cell) / 2, rect.width() - 2 * cell,
                       7 * cell)
        p.drawPixmap(inner.topLeft(), self._unlit(int(inner.width()), int(inner.height()), cell))
        if not self.messages:
            return
        strip = self._build(cell)
        span = strip.width()
        offset = (t * speed * cell / 3) % span
        x = inner.left() - offset
        x = inner.left() + math.floor((x - inner.left()) / cell) * cell  # whole LEDs: the dots stay on the grid
        p.save()
        p.setClipRect(inner)
        while x < inner.right():
            p.drawImage(QPointF(x, inner.top()), strip)
            x += span
        p.restore()


# --- CRT overlay ---------------------------------------------------------------------------------------------------

def crt_overlay(w: int, h: int) -> QPixmap:
    """Scan lines and a vignette over the whole screen."""
    pm = QPixmap(max(w, 1), max(h, 1))
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    line = QColor(0, 0, 0, 34)
    for y in range(0, h, 3):
        p.fillRect(0, y, w, 1, line)
    v = QRadialGradient(QPointF(w / 2, h / 2), math.hypot(w, h) * 0.56)
    v.setColorAt(0.0, QColor(0, 0, 0, 0))
    v.setColorAt(0.75, QColor(0, 0, 0, 30))
    v.setColorAt(1.0, QColor(0, 0, 0, 170))
    p.fillRect(QRectF(0, 0, w, h), v)
    p.end()
    return pm


# --- the insert-coin launch sequence -------------------------------------------------------------------------------

class LaunchIntro(QWidget):
    """Played over a game that just started: a coin drops into the cabinet's slot, PLAYER 1 READY flashes, the view
    zooms into the cabinet's screen and the game's CRT switches on. Any key or click skips it (and still reaches the
    game); it never takes the focus or a mouse click."""

    DURATION = 1.05
    COIN_END = 0.38
    ZOOM = (0.42, 0.74)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.snapshot: QPixmap | None = None
        self.screen = QRectF()
        self.slot = QRectF()
        self.accent = QColor("#ff3fb4")
        self.age = 0.0
        self._watched: QObject | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._last = 0.0
        self.on_coin = None  # called when the coin lands (the clink)
        self._clinked = False
        self.hide()

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    def start(self, snapshot: QPixmap, screen: QRectF, slot: QRectF, accent: QColor, watch: QObject | None) -> None:
        self.snapshot, self.screen, self.slot, self.accent = snapshot, QRectF(screen), QRectF(slot), QColor(accent)
        self.age = 0.0
        self._clinked = False
        self.setGeometry(self.parentWidget().rect())
        self._watch(watch)
        self._last = time.perf_counter()
        self.show()
        self.raise_()
        self._timer.start()

    def _watch(self, obj: QObject | None) -> None:
        if self._watched is not None:
            try:
                self._watched.removeEventFilter(self)
            except RuntimeError:
                pass
        self._watched = obj
        if obj is not None:
            obj.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.MouseButtonPress) and self.age > 0.08:
            self.finish()
        return False  # never takes the event from the game

    def finish(self) -> None:
        self._timer.stop()
        self._watch(None)
        self.snapshot = None
        self.hide()

    def _tick(self) -> None:
        now = time.perf_counter()
        self.age += min(now - self._last, 0.05)
        self._last = now
        if not self._clinked and self.age >= self.COIN_END:
            self._clinked = True
            if self.on_coin is not None:
                self.on_coin()
        if self.age >= self.DURATION:
            self.finish()
            return
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        a = self.age
        z0, z1 = self.ZOOM
        if a < z1 and self.snapshot is not None:
            k = ease_in_cubic((a - z0) / (z1 - z0)) if a > z0 else 0.0
            # zoom so that the cabinet's screen grows to fill the window
            scr = self.screen
            sx = w / max(scr.width(), 1)
            sy = h / max(scr.height(), 1)
            s = 1 + (max(sx, sy) - 1) * k
            cx = scr.center().x() + (w / 2 - scr.center().x()) * 0
            cy = scr.center().y()
            tf = QTransform()
            tf.translate(w / 2 * k + cx * (1 - k), h / 2 * k + cy * (1 - k))
            tf.scale(s, s)
            tf.translate(-cx, -cy)
            p.setTransform(tf)
            p.drawPixmap(QPointF(0, 0), self.snapshot)
            if a < self.COIN_END + 0.05:
                self._coin(p, a)
            flash = max(0.0, 1 - abs(a - self.COIN_END - 0.04) / 0.12)
            if flash > 0:
                p.fillRect(self.slot.adjusted(-6, -6, 6, 6), QColor(255, 240, 180, int(200 * flash)))
            p.resetTransform()
            if k > 0:
                p.fillRect(self.rect(), QColor(255, 255, 255, int(255 * k ** 2)))
        else:
            # the CRT switching on over the game
            k = (a - z1) / (self.DURATION - z1)
            band = h * ease_out_cubic(min(k * 1.6, 1.0))
            top = (h - band) / 2
            p.fillRect(QRectF(0, 0, w, top), QColor("black"))
            p.fillRect(QRectF(0, top + band, w, h - top - band), QColor("black"))
            line = max(0.0, 1 - k * 1.8)
            if line > 0:
                p.fillRect(QRectF(0, h / 2 - 2, w, 4), QColor(255, 255, 255, int(255 * line)))
            p.fillRect(QRectF(0, top, w, band), QColor(255, 255, 255, int(160 * max(0.0, 1 - k * 1.4))))
        # PLAYER 1 READY
        if 0.2 < a < 0.95:
            blink = int(a * 14) % 2 == 0 or a > 0.5
            if blink:
                px = max(min(w / 110, h / 60), 3.0)
                y = h * 0.5 - px * 3.5 if a > z1 else min(self.screen.top() - px * 12, h * 0.3)
                p.setOpacity(min(1.0, (0.95 - a) / 0.15))
                pixelfont.draw(p, "PLAYER 1 READY", w / 2, max(y, px * 2), px, "#ffe14d",
                               Qt.AlignmentFlag.AlignHCenter, glow=True)
                p.setOpacity(1.0)
        p.end()

    def _coin(self, p: QPainter, a: float) -> None:
        slot = self.slot
        k = min(a / self.COIN_END, 1.0)
        r = max(slot.width() * 0.95, 8.0)
        x = slot.center().x()
        y0 = slot.top() - r * 9
        y = y0 + (slot.center().y() - y0) * ease_in_cubic(k)
        spin = abs(math.cos(a * 22))
        p.save()
        if k > 0.85:
            p.setClipRect(QRectF(slot.left() - r * 3, slot.top() - r * 12, slot.width() + r * 6,
                                 slot.center().y() - slot.top() + r * 12))
        g = QLinearGradient(x - r, y - r, x + r, y + r)
        g.setColorAt(0.0, QColor("#fff6b0"))
        g.setColorAt(0.5, QColor("#ffc928"))
        g.setColorAt(1.0, QColor("#b8860b"))
        p.setPen(QPen(QColor("#7a5200"), max(r * 0.12, 1)))
        p.setBrush(g)
        p.drawEllipse(QPointF(x, y), r * max(spin, 0.12), r)
        if spin > 0.4:
            px = r * spin / 6
            pixelfont.draw(p, "¢", x, y - px * 3.5, px, "#7a5200", Qt.AlignmentFlag.AlignHCenter)
        p.restore()
