"""The screens around the game: the title (flames and the fiery logo), the skull-cursor menus, the help page, the
intermission tally, the story text and the column "melt" wipe. They are laid out on a 320 x 200 page that the
painter scales to the window (``page_transform``), drawn with vector text so they stay sharp at any size."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
                           QRadialGradient, QTransform)

from .info import MONSTER_NAMES, SKILLS

SW, SH = 320, 200
RED = QColor(232, 40, 24)
GREY = QColor(150, 140, 132)
GOLD = QColor(255, 200, 70)
WHITE = QColor(246, 236, 222)


def page_transform(w: int, h: int) -> tuple[QTransform, QRectF]:
    """The 320 x 200 page centred in a ``w`` x ``h`` window: (page -> window transform, the page in the window)."""
    s = min(w / SW, h / SH)
    x0, y0 = (w - SW * s) / 2, (h - SH * s) / 2
    return QTransform(s, 0, 0, s, x0, y0), QRectF(x0, y0, SW * s, SH * s)


def font(px: float, family: str = "Impact", bold: bool = False) -> QFont:
    f = QFont(family)
    f.setPixelSize(max(int(round(px)), 1))
    f.setBold(bold)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return f


def text(p: QPainter, x: float, y: float, s: str, colour: QColor, size: float = 12, family: str = "Impact",
         align: str = "left", shadow: float = 0.8, bold: bool = False) -> float:
    """Text with its baseline at ``y`` (page units); ``align``: left / centre / right of ``x``. Returns its width.
    The text is laid out at the window's own pixel size (a small font scaled up would misplace its letters)."""
    tr = p.transform()
    k = max(tr.m11(), 1e-3)
    p.save()
    p.resetTransform()
    p.setFont(font(size * k, family, bold))
    w = p.fontMetrics().horizontalAdvance(s)
    pt = tr.map(QPointF(x, y))
    px = pt.x() - (w / 2 if align == "centre" else w if align == "right" else 0.0)
    if shadow:
        p.setPen(QColor(0, 0, 0, 210))
        p.drawText(QPointF(px + shadow * k, pt.y() + shadow * k), s)
    p.setPen(colour)
    p.drawText(QPointF(px, pt.y()), s)
    p.restore()
    return w / k


def logo_path(size: float) -> QPainterPath:
    path = QPainterPath()
    path.addText(0, 0, font(size, "Impact"), "HELLFIRE")
    return path


def draw_logo(p: QPainter, cx: float, top: float, width: float, t: float = 0.0) -> None:
    path = logo_path(100)
    br = path.boundingRect()
    s = width / br.width()
    p.save()
    p.translate(cx - br.center().x() * s, top - br.top() * s)
    p.scale(s, s)
    p.translate(0, 2.0 * math.sin(t * 1.3) / s)
    p.save()
    p.translate(4, 6)
    p.fillPath(path, QColor(20, 0, 0, 220))
    p.restore()
    g = QLinearGradient(0, br.top(), 0, br.bottom())
    g.setColorAt(0.0, QColor(255, 252, 214))
    g.setColorAt(0.32, QColor(255, 206, 40))
    g.setColorAt(0.66, QColor(236, 76, 0))
    g.setColorAt(1.0, QColor(110, 0, 0))
    p.fillPath(path, g)
    p.strokePath(path, QPen(QColor(50, 0, 0), 2.2))
    p.restore()


def draw_flames(p: QPainter, rect: QRectF, t: float, strength: float = 1.0) -> None:
    """Layers of licking flames along the bottom of ``rect``."""
    w, h = rect.width(), rect.height()
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    for layer, (col, amp, base) in enumerate(((QColor(120, 6, 0, 210), 0.42, 0.70), (QColor(220, 60, 0, 200), 0.30,
                                                                                   0.80),
                                               (QColor(255, 170, 30, 210), 0.18, 0.89),
                                               (QColor(255, 240, 160, 200), 0.08, 0.95))):
        path = QPainterPath()
        path.moveTo(rect.left(), rect.bottom())
        n = 48
        for i in range(n + 1):
            x = rect.left() + w * i / n
            f = (math.sin(i * 1.7 + t * (4.2 + layer) + layer * 1.3) * 0.5
                 + math.sin(i * 0.63 - t * (2.9 + 0.4 * layer)) * 0.35 + math.sin(i * 3.1 + t * 7.0) * 0.15)
            y = rect.top() + h * (base - amp * strength * (0.6 + 0.4 * f) * (0.7 + 0.3 * math.sin(i * 2.3 + layer)))
            path.lineTo(x, y)
        path.lineTo(rect.right(), rect.bottom())
        path.closeSubpath()
        p.setBrush(col)
        p.drawPath(path)
    p.restore()


class Backdrops:
    """Page backgrounds made once from the game's own tiles (darkened, tiled)."""

    def __init__(self) -> None:
        self._cache: dict[str, QImage] = {}

    def tile(self, name: str, dark: float) -> QImage:
        key = f"{name}:{dark}"
        img = self._cache.get(key)
        if img is None:
            from .assets import rgba_to_qimage, tiles
            T = tiles()
            tid = T.flat_ids.get(name, T.wall_ids.get(name))
            if tid is None:
                arr = np.full((128, 128, 4), (30, 10, 8, 255), np.uint8)
            else:
                arr = T.layers[int(T.first[tid])].copy()
                arr[..., :3] = (arr[..., :3].astype(np.float32) * dark).astype(np.uint8)
            img = self._cache[key] = rgba_to_qimage(arr)
        return img

    def fill(self, p: QPainter, rect: QRectF, name: str, dark: float, scale: float = 0.5) -> None:
        b = QBrush(self.tile(name, dark))
        b.setTransform(QTransform.fromScale(scale, scale))
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        p.fillRect(rect, b)
        g = QRadialGradient(rect.center(), max(rect.width(), rect.height()) * 0.7)
        g.setColorAt(0.0, QColor(0, 0, 0, 0))
        g.setColorAt(1.0, QColor(0, 0, 0, 200))
        p.fillRect(rect, g)
        p.restore()


def draw_title(p: QPainter, w: int, h: int, t: float, logo: bool = True) -> None:
    g = QLinearGradient(0, 0, 0, h)
    g.setColorAt(0.0, QColor(10, 0, 0))
    g.setColorAt(0.55, QColor(70, 6, 0))
    g.setColorAt(1.0, QColor(150, 30, 0))
    p.fillRect(QRectF(0, 0, w, h), g)
    draw_flames(p, QRectF(0, h * 0.45, w, h * 0.55), t)
    if logo:
        tr, page = page_transform(w, h)
        p.save()
        p.setTransform(tr, True)
        draw_logo(p, SW / 2, 10, 236, t)
        p.restore()


# --- menus ---------------------------------------------------------------------------------------------------------

class Menu:
    """The skull-cursor menus. ``game`` supplies the actions."""

    def __init__(self, game) -> None:  # noqa: ANN001
        self.game = game
        self.stack: list[str] = []
        self.sel: dict[str, int] = {}
        self.confirm: tuple[str, object] | None = None
        self.message = ""
        self.on_title = False

    @property
    def open(self) -> bool:
        return bool(self.stack) or self.confirm is not None

    @property
    def current(self) -> str | None:
        return self.stack[-1] if self.stack else None

    def show(self, name: str = "main") -> None:
        self.stack = [name]
        self.sel.setdefault(name, 0)
        self.message = ""

    def push(self, name: str) -> None:
        self.stack.append(name)
        self.sel.setdefault(name, 0)
        self.message = ""
        self.game.sfx.play("swtchn")

    def close(self) -> None:
        self.stack = []
        self.confirm = None
        self.game.sfx.play("swtchx" if "swtchx" in self.game.sfx else "swtchn")

    def back(self) -> bool:
        """True when a menu level was closed; False when nothing was open."""
        if self.confirm is not None:
            self.confirm = None
            return True
        if not self.stack:
            return False
        self.stack.pop()
        self.game.sfx.play("swtchn")
        return True

    def items(self, name: str) -> list[tuple[str, object, bool]]:
        g = self.game
        if name == "main":
            return [("NEW GAME", "new", True), ("OPTIONS", "options", True), ("LOAD GAME", "load", True),
                    ("SAVE GAME", "save", g.world is not None and not g.world.player.dead),
                    ("READ THIS!", "help", True), ("BACK TO ARCADE", "quit", True)]
        if name == "skill":
            return [(s.upper(), ("skill", i + 1), True) for i, s in enumerate(SKILLS)]
        if name == "options":
            s = g.settings
            return [("CONTROLS...", "controls", True), ("ALWAYS RUN: " + ("ON" if s["run"] else "OFF"), "run", True),
                    ("MUSIC: " + ("ON" if s["music"] else "OFF"), "music", True),
                    ("MESSAGES: " + ("ON" if s["messages"] else "OFF"), "messages", True)]
        if name in ("load", "save"):
            out = []
            for i, slot in enumerate(g.slots):
                label = slot.get("name", "?") if slot else "EMPTY SLOT"
                out.append((label, (name + "_slot", i), name == "save" or slot is not None))
            return out
        return []

    def key(self, key: int, text_: str = "") -> None:
        if self.confirm is not None:
            if text_.lower() == "y" or key in (Qt.Key.Key_Y, Qt.Key.Key_Return, Qt.Key.Key_Enter):
                _, action = self.confirm
                self.confirm = None
                self.activate(action)
            elif text_.lower() == "n" or key in (Qt.Key.Key_N, Qt.Key.Key_Backspace):
                self.confirm = None
            return
        name = self.current
        if name is None:
            return
        if name == "help":
            self.back()
            return
        items = self.items(name)
        n = len(items)
        i = self.sel.get(name, 0) % max(n, 1)
        if key in (Qt.Key.Key_Up, Qt.Key.Key_W):
            self.sel[name] = (i - 1) % n
            self.game.sfx.play("pstop")
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_S):
            self.sel[name] = (i + 1) % n
            self.game.sfx.play("pstop")
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_A, Qt.Key.Key_Right, Qt.Key.Key_D) and name == "options":
            if items[i][1] != "controls":
                self.activate(items[i][1])
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space, Qt.Key.Key_E):
            if items[i][2]:
                self.activate(items[i][1])
            else:
                self.game.sfx.play("oof")
        elif key == Qt.Key.Key_Backspace:
            self.back()

    def activate(self, action) -> None:  # noqa: ANN001
        g = self.game
        if action == "new":
            self.push("skill")
        elif action == "options":
            self.push("options")
        elif action == "load":
            self.push("load")
        elif action == "save":
            self.push("save")
        elif action == "help":
            self.push("help")
        elif action == "quit":
            g.menu_quit()
        elif isinstance(action, tuple) and action[0] == "skill":
            if action[1] == 5 and self.confirm is None and not getattr(self, "_confirmed", False):
                self._confirmed = True
                self.confirm = ("ARE YOU SURE? THIS ONE IS NOT|EVEN REMOTELY FAIR.||PRESS Y OR N.", action)
                return
            self._confirmed = False
            g.new_game(action[1])
        elif action == "controls":
            g.open_controls()
        elif action in ("run", "music", "messages"):
            g.settings[action] = not g.settings[action]
            g.sfx.play("tink")
            if action == "music":
                g.update_music()
        elif isinstance(action, tuple) and action[0] == "save_slot":
            if g.save_slot(action[1]):
                self.message = "GAME SAVED."
                self.stack = []
                g.hud_message("Game saved.")
        elif isinstance(action, tuple) and action[0] == "load_slot":
            g.load_slot(action[1])

    def item_at(self, x: float, y: float) -> int | None:
        """The item under a point of the 320 x 200 page."""
        name = self.current
        if name is None or name == "help":
            return None
        items = self.items(name)
        top, step, left = self._layout(name, len(items))
        for i in range(len(items)):
            if top + i * step - step * 0.75 <= y < top + i * step + step * 0.25 and left - 24 <= x <= 304:
                return i
        return None

    def _layout(self, name: str, n: int) -> tuple[float, float, float]:
        """(baseline of the first item, item spacing, left edge) on the page."""
        if name in ("load", "save"):
            return 52, 20, 64
        if name == "skill":
            return 86, 18, 62
        if name == "options":
            return 76, 20, 74
        if self.on_title:
            return 200 - n * 16 - 2, 16, 104
        return 82, 17, 104

    def draw(self, p: QPainter, w: int, h: int, t: float) -> None:
        """Draw the open menu over whatever is behind it (the caller dims the play view)."""
        tr, _page = page_transform(w, h)
        p.save()
        p.setTransform(tr, True)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        if self.confirm is not None:
            lines = self.confirm[0].split("|")
            y = 90 - len(lines) * 6
            for ln in lines:
                text(p, SW / 2, y, ln, RED, 12, align="centre")
                y += 13
            p.restore()
            return
        name = self.current
        if name == "help":
            self._draw_help(p)
            p.restore()
            return
        if name is None:
            p.restore()
            return
        items = self.items(name)
        top, step, left = self._layout(name, len(items))
        if name == "main":
            if not self.on_title:
                draw_logo(p, SW / 2, 10, 200, t)
        else:
            if name == "skill":
                text(p, SW / 2, 40, "NEW GAME", RED, 26, align="centre", shadow=1.2)
                text(p, SW / 2, 62, "CHOOSE SKILL LEVEL:", GOLD, 13, align="centre")
            else:
                header = {"options": "OPTIONS", "load": "LOAD GAME", "save": "SAVE GAME"}[name]
                text(p, SW / 2, 34, header, RED, 26, align="centre", shadow=1.2)
        sel = self.sel.get(name, 0) % max(len(items), 1)
        size = 13 if name == "skill" else 16
        for i, (label, _action, enabled) in enumerate(items):
            y = top + i * step
            if name in ("load", "save"):
                box = QRectF(left - 4, y - 14, 200, 18)
                p.setPen(QPen(QColor(110, 30, 20) if i != sel else GOLD, 0.8))
                p.setBrush(QColor(10, 6, 6, 210))
                p.drawRoundedRect(box, 2, 2)
                text(p, left + 2, y, label, (RED if i != sel else GOLD) if enabled else GREY, 11, "Consolas",
                     bold=True)
            else:
                col = (GOLD if i == sel else RED) if enabled else GREY
                text(p, left, y, label, col, size, shadow=1.0)
        self._skull(p, left - 22, top + sel * step - 14, t)
        if self.message:
            text(p, SW / 2, 192, self.message, GOLD, 11, align="centre")
        p.restore()

    def _skull(self, p: QPainter, x: float, y: float, t: float) -> None:
        try:
            from .assets import sprites
            S = sprites()
            frames = S.monsters["soul"]["walk"]
            img = S.image(int(frames[int(t * 6) % len(frames)][0][0]))
        except Exception:  # noqa: BLE001 - the cursor is never worth an error
            img = QImage()
        if img.isNull():
            p.setBrush(GOLD)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(x + 3, y + 3, 12, 12))
            return
        s = 18.0 / max(img.width(), img.height())
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        p.drawImage(QRectF(x, y, img.width() * s, img.height() * s), img)
        p.restore()

    def _draw_help(self, p: QPainter) -> None:
        p.setPen(QPen(QColor(150, 34, 20), 0.8))
        p.setBrush(QColor(8, 5, 5, 235))
        p.drawRoundedRect(QRectF(8, 6, 304, 188), 4, 4)
        text(p, SW / 2, 26, "READ THIS!", RED, 18, align="centre")
        lines = (
            ("W A S D / ARROWS", "MOVE, STRAFE, TURN"), ("MOUSE", "TURN"), ("CTRL / LEFT CLICK", "FIRE"),
            ("SPACE / E", "USE: DOORS, SWITCHES, LIFTS"), ("1 - 7 / WHEEL / Q", "WEAPONS (1 TWICE: RIPSAW)"),
            ("SHIFT", "WALK (ALWAYS RUN IS ON)"), ("TAB  + / -", "AUTOMAP, ZOOM"), ("F2 / F3", "SAVE / LOAD GAME"),
            ("ESC", "MENU"),
        )
        y = 42
        for k, v in lines:
            text(p, 22, y, k, GOLD, 8, "Consolas", shadow=0.5, bold=True)
            text(p, 120, y, v, WHITE, 8, "Consolas", shadow=0.5)
            y += 10.5
        text(p, SW / 2, y + 6, "Find the exit switch. Coloured doors need the card of their colour.", RED, 7.5,
             "Segoe UI", align="centre", bold=True, shadow=0.5)
        y += 18
        names = ", ".join(MONSTER_NAMES[k] for k in ("zombie", "sergeant", "heavy", "imp", "demon", "soul", "caco"))
        text(p, SW / 2, y, "Known hostiles: " + names + ".", GREY, 5.6, "Segoe UI", align="centre", shadow=0.4)
        text(p, SW / 2, 186, "Sprites, walls and floors: the Freedoom project (BSD licence, see freedoom/COPYING.adoc).",
             GREY, 6, "Segoe UI", align="centre", shadow=0.4)


# --- intermission ------------------------------------------------------------------------------------------------

def _clock(s: int) -> str:
    s = max(int(s), 0)
    return f"{s // 60}:{s % 60:02d}"


class Intermission:
    """KILLS / ITEMS / SECRET percentages counting up, then TIME and PAR, then the next map's name on the route."""

    def __init__(self, name: str, next_name: str | None, stats: dict, index: int, keys: list | None = None,
                 next_index: int | None = None) -> None:
        self.name, self.next_name = name, next_name
        self.stats = stats
        self.index = index
        self.next_index = index + 1 if next_index is None else next_index
        self.keys = keys or []
        self.cnt = [-1, -1, -1]
        self.time_cnt = -1
        self.par_cnt = -1
        self.stage = 0  # 0 counting, 1 done, 2 "entering"
        self.tics = 0
        self.done = False

    def _pct(self, a: int, b: int) -> int:
        return 100 if b == 0 else a * 100 // b

    def tick(self, sfx) -> None:  # noqa: ANN001
        self.tics += 1
        if self.stage != 0 or self.tics < 20:
            if self.stage == 2 and self.tics > 120:
                self.done = True
            return
        targets = [self._pct(self.stats["kills"], self.stats["total_kills"]),
                   self._pct(self.stats["items"], self.stats["total_items"]),
                   self._pct(self.stats["secrets"], self.stats["total_secrets"])]
        for i in range(3):
            if self.cnt[i] < targets[i]:
                self.cnt[i] = min(self.cnt[i] + 2, targets[i])
                if self.tics % 3 == 0:
                    sfx.play("pistol", 0.5)
                if self.cnt[i] >= targets[i]:
                    sfx.play("barexp", 0.6)
                return
        if self.time_cnt < self.stats["time"]:
            self.time_cnt = min(self.time_cnt + 3, self.stats["time"])
            self.par_cnt = min(self.par_cnt + 3, self.stats["par"])
            if self.tics % 3 == 0:
                sfx.play("pistol", 0.5)
            if self.time_cnt >= self.stats["time"]:
                self.par_cnt = self.stats["par"]
                sfx.play("barexp", 0.6)
                self.stage = 1
            return
        self.par_cnt = self.stats["par"]
        self.stage = 1

    def skip(self, sfx) -> None:  # noqa: ANN001
        if self.stage == 0:
            self.cnt = [self._pct(self.stats["kills"], self.stats["total_kills"]),
                        self._pct(self.stats["items"], self.stats["total_items"]),
                        self._pct(self.stats["secrets"], self.stats["total_secrets"])]
            self.time_cnt, self.par_cnt = self.stats["time"], self.stats["par"]
            self.stage = 1
            sfx.play("barexp", 0.6)
        elif self.stage == 1:
            if self.next_name:
                self.stage = 2
                self.tics = 0
                sfx.play("sgcock")
            else:
                self.done = True
        else:
            self.done = True

    def draw(self, p: QPainter, w: int, h: int, back: Backdrops) -> None:
        back.fill(p, QRectF(0, 0, w, h), "CAVEFLR", 0.42, max(h / 780.0, 0.4))
        tr, _page = page_transform(w, h)
        p.save()
        p.setTransform(tr, True)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.stage < 2:
            text(p, SW / 2, 26, self.name.upper(), RED, 22, align="centre", shadow=1.2)
            text(p, SW / 2, 44, "CLEARED", GOLD, 14, align="centre")
            labels = ("KILLS", "ITEMS", "SECRETS")
            for i, lab in enumerate(labels):
                y = 76 + i * 22
                text(p, 60, y, lab, RED, 18, shadow=1.0)
                if self.cnt[i] >= 0:
                    text(p, 260, y, f"{self.cnt[i]}%", WHITE, 18, align="right", shadow=1.0)
            if self.time_cnt >= 0:
                text(p, 30, 158, "TIME", RED, 18, shadow=1.0)
                text(p, 150, 158, _clock(self.time_cnt), WHITE, 18, align="right", shadow=1.0)
                text(p, 176, 158, "PAR", RED, 18, shadow=1.0)
                text(p, 290, 158, _clock(self.par_cnt), WHITE, 18, align="right", shadow=1.0)
            if self.stage == 1 and (self.tics // 18) % 2:
                text(p, SW / 2, 190, "PRESS FIRE TO CONTINUE", GREY, 9, "Segoe UI", align="centre", bold=True)
            p.restore()
            return
        # the route through the base: a mark per map, a splat on every one cleared, the next one blinking
        marks = self.keys
        n = max(len(marks), 1)
        pts = [QPointF(34 + i * (252 / max(n - 1, 1)), 132 + 18 * math.sin(i * 1.9)) for i in range(n)]
        p.setPen(QPen(QColor(200, 170, 140, 160), 1.2, Qt.PenStyle.DashLine))
        for a, b in zip(pts, pts[1:]):
            p.drawLine(a, b)
        rng = np.random.default_rng(4)
        for i, (pt, key) in enumerate(zip(pts, marks)):
            p.setPen(QPen(QColor(20, 10, 8), 1.0))
            p.setBrush(QColor(110, 120, 130))
            p.drawRoundedRect(QRectF(pt.x() - 5, pt.y() - 5, 10, 10), 1.5, 1.5)
            text(p, pt.x(), pt.y() + 17, key, GREY, 8, "Consolas", align="centre", bold=True)
            if i <= self.index and i != self.next_index:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(170, 0, 0, 220))
                for _ in range(7):
                    dx, dy = rng.normal(0, 3.5, 2)
                    r = float(rng.uniform(1.2, 3.2))
                    p.drawEllipse(QPointF(pt.x() + dx, pt.y() + dy), r, r)
        if 0 <= self.next_index < len(pts) and (self.tics // 12) % 2:
            pt = pts[self.next_index]
            arrow = QPainterPath()
            arrow.moveTo(pt.x(), pt.y() - 8)
            arrow.lineTo(pt.x() - 7, pt.y() - 20)
            arrow.lineTo(pt.x() + 7, pt.y() - 20)
            arrow.closeSubpath()
            p.fillPath(arrow, GOLD)
            text(p, pt.x(), pt.y() - 25, "YOU ARE HERE", GOLD, 9, align="centre")
        text(p, SW / 2, 40, "ENTERING", RED, 18, align="centre", shadow=1.2)
        text(p, SW / 2, 66, (self.next_name or "").upper(), WHITE, 22, align="centre", shadow=1.2)
        p.restore()


# --- text screen -----------------------------------------------------------------------------------------------------

class TextScreen:
    """Story text typed out over a dark wall (3 characters per tic)."""

    def __init__(self, text_: str, bg: str = "text") -> None:
        self.text = text_
        self.count = 0
        self.tics = 0
        self.done = False
        self.bg = bg

    def tick(self) -> None:
        self.tics += 1
        if self.tics % 3 == 0:
            self.count += 1

    def skip(self) -> None:
        if self.count < len(self.text):
            self.count = len(self.text)
        else:
            self.done = True

    def draw(self, p: QPainter, w: int, h: int, back: Backdrops) -> None:
        back.fill(p, QRectF(0, 0, w, h), "SKULLFLR" if self.bg == "hell" else "ROCK", 0.32, max(h / 780.0, 0.4))
        tr, _page = page_transform(w, h)
        p.save()
        p.setTransform(tr, True)
        shown = self.text[:self.count]
        y = 22
        for line in shown.split("\n"):
            text(p, 22, y, line, WHITE, 9, "Consolas", shadow=0.6, bold=True)
            y += 12
        if self.count >= len(self.text) and (self.tics // 18) % 2:
            text(p, SW / 2, 190, "PRESS FIRE TO CONTINUE", RED, 10, align="centre")
        p.restore()


# --- the wipe --------------------------------------------------------------------------------------------------------

class Wipe:
    """The old picture melts down in uneven columns, revealing the new one."""

    COLS = 160

    def __init__(self, old: QImage, seed: int = 5) -> None:
        self.old = old.copy()
        rng = np.random.default_rng(seed)
        y = np.zeros(self.COLS, np.int32)
        y[0] = -int(rng.integers(0, 16))
        for i in range(1, self.COLS):
            y[i] = min(max(y[i - 1] + int(rng.integers(-1, 2)), -15), 0)
        self.y = y.astype(np.float64)  # in 200ths of the picture's height
        self.done = old.isNull()

    def tick(self) -> None:
        y = self.y
        dy = np.where(y < 0, 1, np.where(y < 16, y + 1, 8))
        self.y = np.minimum(y + dy, SH)
        if (self.y >= SH).all():
            self.done = True

    def draw(self, p: QPainter, w: int, h: int) -> None:
        if self.done:
            return
        ow, oh = self.old.width(), self.old.height()
        p.save()
        for i in range(self.COLS):
            off = max(self.y[i], 0.0) / SH
            if off >= 1.0:
                continue
            x0, x1 = w * i / self.COLS, w * (i + 1) / self.COLS
            sx0, sx1 = ow * i / self.COLS, ow * (i + 1) / self.COLS
            p.drawImage(QRectF(x0, h * off, x1 - x0 + 0.5, h * (1 - off)), self.old,
                        QRectF(sx0, 0, sx1 - sx0, oh * (1 - off)))
        p.restore()
