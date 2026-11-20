"""Vanguard's menus, painted over the showroom (the game draws the 3D behind them): the main menu, the match
set-up, the hero gallery with every kit explained, options, statistics and the pause menu."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen

from . import hud
from .data import (DIFFICULTIES, GOLD, HERO_ORDER, HEROES, MODE_BLURB, MODE_LABEL, MODES, ROLE_BLURB, ROLE_LABEL,
                   ROLE_RGB)
from .hud import A, NOPEN, col, icon, slant, text
from .maps import MAP_INFO, MAPS_BY_MODE

K = Qt.Key
SHADOWS = ("Off", "Soft", "Sharp")
CROSS = (("White", (255, 255, 255)), ("Green", (110, 255, 140)), ("Cyan", (110, 240, 255)),
         ("Yellow", (255, 230, 90)), ("Pink", (255, 130, 220)))
MAIN = (("play", "PLAY", "Pick a mode, a map and a hero"), ("heroes", "HEROES", "Every hero and what they do"),
        ("stats", "STATISTICS", "Your record"), ("options", "OPTIONS", "View and picture"),
        ("controls", "CONTROLS", "Keys and mouse"))
PAUSE = (("resume", "RESUME", ""), ("swap", "CHANGE HERO", "In your spawn room or while you wait to respawn"),
         ("options", "OPTIONS", ""), ("controls", "CONTROLS", ""), ("leave", "LEAVE MATCH", "It counts as a loss"))
PLAY_ROWS = ("mode", "map", "side", "difficulty", "hero", "start")
OPTION_ROWS = ("fov", "shadows", "outlines", "cross", "names")


class Menu:
    def __init__(self, g) -> None:  # noqa: ANN001 - game.VanguardGame
        self.g = g
        self.page = "main"
        self.sel = 0
        self.rects: list = []  # [(id, value, QRectF)] of what can be clicked, made while painting
        self.hover_id = None
        self.paused = False  # the pause menu of a match (not the main menu)
        self.back_to = "main"

    @property
    def active(self) -> bool:
        return self.g.screen == "menu" or self.paused

    def open(self, page: str) -> None:
        self.page = page
        self.sel = 0
        if page == "heroes":
            self.sel = HERO_ORDER.index(self.g.cfg["hero"])
        self.rects = []

    def open_pause(self) -> None:
        self.paused = True
        self.open("pause")

    def close(self) -> None:
        self.paused = False

    # --- what the rows are ----------------------------------------------------------------------------------------
    def _rows(self) -> tuple:
        if self.page == "main":
            return tuple(i[0] for i in MAIN)
        if self.page == "pause":
            return tuple(i[0] for i in PAUSE)
        if self.page == "play":
            return tuple(r for r in PLAY_ROWS if r != "side" or self.g.cfg["mode"] == "escort")
        if self.page == "options":
            return OPTION_ROWS
        if self.page == "heroes":
            return HERO_ORDER
        return ()

    def _choices(self, row: str) -> list:
        cfg = self.g.cfg
        if row == "mode":
            return list(MODES)
        if row == "map":
            return MAPS_BY_MODE[cfg["mode"]]
        if row == "side":
            return ["attack", "defend"]
        if row == "difficulty":
            return list(DIFFICULTIES)
        if row == "hero":
            return list(HERO_ORDER)
        return []

    def _set(self, row: str, value) -> None:  # noqa: ANN001
        g = self.g
        if row in ("mode", "map", "side", "difficulty", "hero"):
            g.cfg[row] = value
            if row == "mode" and g.cfg["map"] not in MAPS_BY_MODE[value]:
                g.cfg["map"] = MAPS_BY_MODE[value][0]
        elif row == "fov":
            g.settings["fov"] = int(min(max(value, 70), 110))
        elif row == "shadows":
            g.settings["shadows"] = int(value) % 3
        elif row == "outlines":
            g.settings["outlines"] = bool(value)
        elif row == "cross":
            g.settings["cross"] = int(value) % len(CROSS)
        elif row == "names":
            g.settings["names"] = bool(value)
        g.apply_settings()
        g.progress_changed.emit()

    def _step(self, row: str, d: int) -> None:
        g = self.g
        ch = self._choices(row)
        if ch:
            self._set(row, ch[(ch.index(g.cfg[row]) + d) % len(ch)])
        elif row == "fov":
            self._set(row, g.settings["fov"] + d * 5)
        elif row in ("shadows", "cross"):
            self._set(row, g.settings[row] + d)
        elif row in ("outlines", "names"):
            self._set(row, not g.settings[row])
        g.sfx.play("ui_move")

    def _activate(self, rid: str) -> None:
        g = self.g
        if rid in ("heroes", "stats", "options", "play"):
            self.back_to = self.page
            self.open(rid)
            g.sfx.play("ui_ok")
        elif rid == "controls":
            g.open_controls()
        elif rid == "start":
            g.sfx.play("ui_ok")
            g.start_match()
        elif rid == "resume":
            self.close()
            g.sfx.play("ui_back")
        elif rid == "swap":
            self.close()
            g.open_picker()
        elif rid == "leave":
            self.close()
            g.leave_match()
        elif rid == "back":
            self.escape()
        elif self.page in ("play", "options") and rid in self._rows():
            self._step(rid, 1)

    # --- input ----------------------------------------------------------------------------------------------------
    def escape(self) -> bool:
        """Esc: back one page; False at the main menu's top (the arcade then pauses)."""
        if self.page == "pause":
            self.close()
            self.g.sfx.play("ui_back")
            return True
        if self.page == "main":
            return False
        self.open("pause" if self.paused else "main")
        if self.back_to in ("play",) and not self.paused and self.page != self.back_to:
            pass
        self.g.sfx.play("ui_back")
        return True

    def key(self, key: int) -> None:
        rows = self._rows()
        g = self.g
        if key in (K.Key_Backspace,):
            self.escape()
            return
        if self.page == "stats":
            if key in (K.Key_Return, K.Key_Enter, K.Key_Space):
                self.escape()
            return
        if not rows:
            return
        if self.page == "heroes":
            if key in (K.Key_Up, K.Key_W, K.Key_Left, K.Key_A):
                self.sel = (self.sel - 1) % len(rows)
                g.sfx.play("ui_move")
            elif key in (K.Key_Down, K.Key_S, K.Key_Right, K.Key_D):
                self.sel = (self.sel + 1) % len(rows)
                g.sfx.play("ui_move")
            elif key in (K.Key_Return, K.Key_Enter, K.Key_Space):
                g.cfg["hero"] = rows[self.sel]
                g.sfx.play("ui_ok")
                self.open("play")
            return
        self.sel %= len(rows)
        if key in (K.Key_Up, K.Key_W):
            self.sel = (self.sel - 1) % len(rows)
            g.sfx.play("ui_move")
        elif key in (K.Key_Down, K.Key_S):
            self.sel = (self.sel + 1) % len(rows)
            g.sfx.play("ui_move")
        elif key in (K.Key_Left, K.Key_A) and self.page in ("play", "options"):
            self._step(rows[self.sel], -1)
        elif key in (K.Key_Right, K.Key_D) and self.page in ("play", "options"):
            self._step(rows[self.sel], 1)
        elif key in (K.Key_Return, K.Key_Enter, K.Key_Space):
            if self.page == "play" and rows[self.sel] != "start":
                self.sel = len(rows) - 1
                g.sfx.play("ui_move")
            else:
                self._activate(rows[self.sel])

    def _at(self, pos: QPointF):  # noqa: ANN202
        for rid, value, r in reversed(self.rects):
            if r.contains(pos):
                return rid, value
        return None

    def hover(self, pos: QPointF) -> None:
        hit = self._at(pos)
        self.hover_id = hit
        rows = self._rows()
        if hit is not None and hit[0] in rows and self.page != "heroes":
            self.sel = rows.index(hit[0])

    def click(self, pos: QPointF) -> None:
        hit = self._at(pos)
        if hit is None:
            return
        rid, value = hit
        g = self.g
        if rid == "pick":  # a hero in the gallery list
            self.sel = HERO_ORDER.index(value)
            g.sfx.play("ui_move")
        elif rid == "use":
            g.cfg["hero"] = HERO_ORDER[self.sel]
            g.progress_changed.emit()
            g.sfx.play("ui_ok")
            self.open("play")
        elif value is not None:
            self._set(rid, value)
            g.sfx.play("ui_move")
        else:
            self._activate(rid)

    def wheel(self, steps: int) -> None:
        if self.page == "heroes":
            self.sel = (self.sel - steps) % len(HERO_ORDER)

    def featured(self) -> str | None:
        """The hero the showroom should put forward (None: the whole line-up)."""
        if self.page == "heroes":
            return HERO_ORDER[self.sel % len(HERO_ORDER)]
        if self.page == "play":
            return self.g.cfg["hero"]
        return None

    # --- painting -------------------------------------------------------------------------------------------------
    def paint(self, p: QPainter, w: int, h: int, t: float) -> None:
        self.rects = []
        s = h / 780.0
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if self.paused:
            p.fillRect(QRectF(0, 0, w, h), QColor(8, 12, 24, 170))
        getattr(self, "_paint_" + self.page)(p, w, h, s, t)

    def _title(self, p: QPainter, s: float, title: str, sub: str = "") -> None:
        text(p, QRectF(54 * s, 36 * s, 900 * s, 60 * s), title, 52 * s, QColor(255, 255, 255), italic=True)
        p.setPen(NOPEN)
        p.setBrush(col(GOLD))
        p.drawPolygon(slant(QRectF(54 * s, 98 * s, 150 * s, 6 * s), 0.9))
        if sub:
            text(p, QRectF(56 * s, 108 * s, 900 * s, 24 * s), sub, 16 * s, QColor(255, 255, 255, 200), bold=False)

    def _button(self, p: QPainter, r: QRectF, s: float, label: str, on: bool, rid: str, value=None,  # noqa: ANN001
                size: float = 22, accent: tuple = GOLD) -> None:
        hov = self.hover_id == (rid, value)
        p.setPen(NOPEN)
        p.setBrush(col(accent, 240) if on else (QColor(60, 70, 100, 230) if hov else QColor(22, 28, 46, 215)))
        p.drawPolygon(slant(r, 0.18))
        text(p, r, label, size * s, QColor(16, 20, 34) if on else QColor(255, 255, 255), A.AlignCenter, italic=True,
             shadow=not on)
        self.rects.append((rid, value, r))

    def _back(self, p: QPainter, w: int, h: int, s: float) -> None:
        self._button(p, QRectF(54 * s, h - 66 * s, 150 * s, 38 * s), s, "BACK", False, "back", None, 18)
        text(p, QRectF(216 * s, h - 66 * s, 300 * s, 38 * s), "Esc", 13 * s, QColor(255, 255, 255, 150), bold=False)

    def _list(self, p: QPainter, w: int, h: int, s: float, items: tuple, y0: float) -> None:
        y = y0
        for i, (rid, label, sub) in enumerate(items):
            on = i == self.sel
            r = QRectF(54 * s, y, 330 * s, 52 * s)
            self._button(p, r, s, label, on, rid, None, 26)
            if sub and on:
                text(p, QRectF(r.right() + 16 * s, y, 420 * s, 52 * s), sub, 15 * s, QColor(255, 255, 255, 210),
                     bold=False)
            y += 62 * s

    def _paint_main(self, p: QPainter, w: int, h: int, s: float, t: float) -> None:
        g = QLinearGradient(0, 0, w * 0.6, 0)
        g.setColorAt(0, QColor(8, 12, 26, 210))
        g.setColorAt(1, QColor(8, 12, 26, 0))
        p.fillRect(QRectF(0, 0, w * 0.6, h), g)
        text(p, QRectF(50 * s, 50 * s, 900 * s, 110 * s), "VANGUARD", 104 * s, QColor(255, 255, 255), italic=True)
        p.setPen(NOPEN)
        p.setBrush(col(GOLD))
        p.drawPolygon(slant(QRectF(56 * s, 160 * s, 300 * s, 8 * s), 0.9))
        text(p, QRectF(58 * s, 174 * s, 700 * s, 26 * s), "FIVE AGAINST FIVE   ·   SEVEN HEROES   ·   TWO OBJECTIVES",
             16 * s, QColor(255, 255, 255, 210))
        self._list(p, w, h, s, MAIN, 250 * s)
        st = self.g.stats
        if st["matches"]:
            text(p, QRectF(58 * s, h - 80 * s, 600 * s, 22 * s),
                 f"{st['wins']} wins of {st['matches']} matches   ·   {st['elims']} eliminations", 15 * s,
                 QColor(255, 255, 255, 190), bold=False)
        text(p, QRectF(58 * s, h - 52 * s, 700 * s, 20 * s), "↑ ↓ / mouse: choose   ·   Enter: open   ·   Esc: "
             "the arcade's menu", 13 * s, QColor(255, 255, 255, 150), bold=False)

    def _paint_pause(self, p: QPainter, w: int, h: int, s: float, t: float) -> None:
        self._title(p, s, "PAUSED", f"{self.g.match.md.name}  ·  {MODE_LABEL[self.g.match.mode]}")
        self._list(p, w, h, s, PAUSE, 190 * s)

    def _paint_play(self, p: QPainter, w: int, h: int, s: float, t: float) -> None:
        g = QLinearGradient(0, 0, w * 0.72, 0)
        g.setColorAt(0, QColor(8, 12, 26, 225))
        g.setColorAt(0.8, QColor(8, 12, 26, 150))
        g.setColorAt(1, QColor(8, 12, 26, 0))
        p.fillRect(QRectF(0, 0, w * 0.72, h), g)
        self._title(p, s, "PLAY", "You and four bots against five bots")
        cfg = self.g.cfg
        rows = self._rows()
        y = 150 * s
        labels = {"mode": "MODE", "map": "MAP", "side": "SIDE", "difficulty": "BOTS", "hero": "HERO"}
        for i, row in enumerate(rows):
            on = i == self.sel
            if row == "start":
                self._button(p, QRectF(54 * s, h - 128 * s, 300 * s, 54 * s), s, "START MATCH", on, "start", None, 26)
                continue
            text(p, QRectF(54 * s, y, 110 * s, 40 * s), labels[row], 16 * s,
                 col(GOLD) if on else QColor(255, 255, 255, 190))
            x = 170 * s
            if row == "hero":
                for hid in HERO_ORDER:
                    hero = HEROES[hid]
                    r = QRectF(x, y - 4 * s, 78 * s, 96 * s)
                    cur = cfg["hero"] == hid
                    hov = self.hover_id == ("hero", hid)
                    p.setPen(QPen(col(GOLD), 3 * s) if cur else NOPEN)
                    p.setBrush(QColor(52, 60, 92, 240) if (cur or hov) else QColor(22, 28, 46, 215))
                    p.drawPolygon(slant(r, 0.07))
                    img = self.g.portrait(hid, True, False, 96)
                    if img is not None:
                        p.drawImage(QRectF(r.left() + 13 * s, r.top() + 6 * s, 52 * s, 52 * s), img)
                    text(p, QRectF(r.left(), r.top() + 58 * s, r.width(), 18 * s), hero.name, 13 * s,
                         QColor(255, 255, 255), A.AlignCenter, italic=True)
                    icon(p, hero.role, QRectF(r.center().x() - 7 * s, r.top() + 77 * s, 14 * s, 14 * s),
                         col(ROLE_RGB[hero.role]))
                    self.rects.append(("hero", hid, r))
                    x += 86 * s
                hero = HEROES[cfg["hero"]]
                text(p, QRectF(170 * s, y + 98 * s, 600 * s, 22 * s),
                     f"{hero.name}  ·  {hero.title}  ·  {ROLE_LABEL[hero.role]}: {ROLE_BLURB[hero.role]}", 14 * s,
                     QColor(255, 255, 255, 215), bold=False)
                y += 130 * s
                continue
            for ch in self._choices(row):
                label = {"mode": MODE_LABEL.get(ch, ""), "map": MAP_INFO.get(ch, ("",))[0]}.get(row, str(ch).upper())
                p.setFont(hud.fnt(18 * s, True, True))
                bw = max(p.fontMetrics().horizontalAdvance(label) + 40 * s, 110 * s)
                self._button(p, QRectF(x, y, bw, 40 * s), s, label, cfg[row] == ch, row, ch, 18,
                             (255, 255, 255) if not on else GOLD)
                x += bw + 8 * s
            note = {"mode": MODE_BLURB[cfg["mode"]], "map": MAP_INFO[cfg["map"]][2],
                    "side": "Attackers push the cart. Defenders hold it back until the clock runs out.",
                    "difficulty": "How well the enemy bots aim, react and use their abilities."}[row]
            text(p, QRectF(170 * s, y + 42 * s, 640 * s, 20 * s), note, 13 * s, QColor(255, 255, 255, 180),
                 bold=False)
            y += 74 * s
        self._back(p, w, h, s)
        text(p, QRectF(370 * s, h - 128 * s, 500 * s, 54 * s), "← →: change   ·   Enter: start", 13 * s,
             QColor(255, 255, 255, 150), bold=False)

    def _paint_heroes(self, p: QPainter, w: int, h: int, s: float, t: float) -> None:
        self._title(p, s, "HEROES")
        y = 150 * s
        for i, hid in enumerate(HERO_ORDER):
            hero = HEROES[hid]
            on = i == self.sel
            r = QRectF(54 * s, y, 210 * s, 50 * s)
            hov = self.hover_id == ("pick", hid)
            p.setPen(NOPEN)
            p.setBrush(col(GOLD, 240) if on else (QColor(60, 70, 100, 230) if hov else QColor(22, 28, 46, 215)))
            p.drawPolygon(slant(r, 0.16))
            img = self.g.portrait(hid, True, False, 96)
            if img is not None:
                p.drawImage(QRectF(r.left() + 14 * s, r.top() + 5 * s, 40 * s, 40 * s), img)
            text(p, QRectF(r.left() + 62 * s, r.top(), 110 * s, r.height()), hero.name, 20 * s,
                 QColor(16, 20, 34) if on else QColor(255, 255, 255), italic=True, shadow=not on)
            icon(p, hero.role, QRectF(r.right() - 38 * s, r.top() + 17 * s, 16 * s, 16 * s),
                 QColor(16, 20, 34) if on else col(ROLE_RGB[hero.role]))
            self.rects.append(("pick", hid, r))
            y += 58 * s
        # the kit
        hero = HEROES[HERO_ORDER[self.sel % len(HERO_ORDER)]]
        px = w - 560 * s
        panel = QRectF(px, 28 * s, 532 * s, h - 56 * s)
        p.setPen(NOPEN)
        p.setBrush(QColor(10, 14, 28, 225))
        p.drawRect(panel)
        p.setBrush(col(hero.color))
        p.drawRect(QRectF(px, 28 * s, 532 * s, 6 * s))
        x = px + 24 * s
        tw = 484 * s
        text(p, QRectF(x, 44 * s, tw, 50 * s), hero.name, 46 * s, QColor(255, 255, 255), italic=True)
        icon(p, hero.role, QRectF(x, 100 * s, 18 * s, 18 * s), col(ROLE_RGB[hero.role]))
        text(p, QRectF(x + 26 * s, 98 * s, tw, 22 * s), f"{ROLE_LABEL[hero.role]}   ·   {hero.title}", 15 * s,
             col(ROLE_RGB[hero.role]))
        stars = "●" * hero.stars + "○" * (3 - hero.stars)
        text(p, QRectF(x, 98 * s, tw, 22 * s), f"difficulty  {stars}", 13 * s, QColor(255, 255, 255, 190),
             A.AlignRight | A.AlignVCenter, bold=False)
        wrap = A.AlignLeft | A.AlignTop | Qt.TextFlag.TextWordWrap
        text(p, QRectF(x, 128 * s, tw, 64 * s), hero.blurb, 14 * s, QColor(255, 255, 255, 225), wrap, bold=False)
        # health
        y = 198 * s
        bar = QRectF(x, y, tw, 12 * s)
        k = tw / 700.0
        xx = bar.left()
        for amount, c in ((hero.hp, QColor(245, 245, 245)), (hero.armor, QColor(255, 170, 60)),
                          (hero.shield, QColor(90, 190, 255))):
            if amount:
                p.setBrush(c)
                p.drawRect(QRectF(xx, y, amount * k - 2 * s, 12 * s))
                xx += amount * k
        parts = [f"{hero.hp} health"] + ([f"{hero.armor} armour"] if hero.armor else []) + \
            ([f"{hero.shield} shields"] if hero.shield else [])
        text(p, QRectF(x, y + 14 * s, tw, 20 * s), "  +  ".join(parts) + f"   ·   speed {hero.speed:g} m/s", 13 * s,
             QColor(255, 255, 255, 200), bold=False)
        y += 44 * s
        g = self.g
        wpn = hero.weapon
        entries = [("melee" if False else None, wpn.name, "LMB", f"{wpn.mag} in the magazine" if wpn.kind != "cone"
                    else "tank of 100", wpn.desc),
                   (hero.a1.id, hero.a1.name, g.key_name("Ability 1"),
                    f"{hero.a1.cd:g} s" + (f"  ·  {hero.a1.charges} charges" if hero.a1.charges > 1 else ""),
                    hero.a1.desc),
                   (hero.a2.id, hero.a2.name, g.key_name("Ability 2"), f"{hero.a2.cd:g} s", hero.a2.desc),
                   (hero.ult.id, hero.ult.name, g.key_name("Ultimate"), f"ultimate  ·  {hero.ult.cost} charge",
                    hero.ult.desc)]
        for ic, name, key, meta, desc in entries:
            box = QRectF(x, y, 46 * s, 46 * s)
            ult = ic == hero.ult.id
            p.setPen(NOPEN)
            p.setBrush(col(GOLD) if ult else col(hero.glow))
            p.drawPolygon(slant(box, 0.14))
            if ic:
                icon(p, ic, QRectF(box.left() + 8 * s, box.top() + 8 * s, 30 * s, 30 * s), QColor(16, 20, 34))
            else:
                text(p, box, "◎", 26 * s, QColor(16, 20, 34), A.AlignCenter, shadow=False)
            text(p, QRectF(x + 58 * s, y - 2 * s, tw - 58 * s, 22 * s), name, 18 * s, QColor(255, 255, 255),
                 italic=True)
            text(p, QRectF(x + 58 * s, y - 2 * s, tw - 58 * s, 22 * s), f"[{key}]   {meta}", 12 * s,
                 col(GOLD) if ult else QColor(255, 255, 255, 170), A.AlignRight | A.AlignVCenter, bold=False)
            text(p, QRectF(x + 58 * s, y + 20 * s, tw - 58 * s, 54 * s), desc, 13 * s, QColor(255, 255, 255, 215),
                 wrap, bold=False)
            y += 78 * s
        if hero.tips:
            text(p, QRectF(x, y, tw, 18 * s), "TIPS", 12 * s, col(GOLD))
            text(p, QRectF(x, y + 18 * s, tw, 60 * s), "\n".join("·  " + tip for tip in hero.tips), 13 * s,
                 QColor(255, 255, 255, 215), wrap, bold=False)
        self._button(p, QRectF(panel.right() - 224 * s, panel.bottom() - 54 * s, 200 * s, 40 * s), s,
                     "PLAY THIS HERO", False, "use", None, 17)
        self._back(p, w, h, s)

    def _paint_options(self, p: QPainter, w: int, h: int, s: float, t: float) -> None:
        if not self.paused:
            p.fillRect(QRectF(0, 0, w, h), QColor(8, 12, 26, 150))
        self._title(p, s, "OPTIONS", "Mouse speed and keys are under Controls")
        st = self.g.settings
        rows = (("fov", "FIELD OF VIEW", f"{st['fov']}°", "How wide you see (70 to 110)."),
                ("shadows", "SHADOWS", SHADOWS[st["shadows"]], "Sharp looks best; Off is fastest."),
                ("outlines", "TEAM OUTLINES", "On" if st["outlines"] else "Off",
                 "Red round enemies, blue round your team."),
                ("cross", "CROSSHAIR", CROSS[st["cross"]][0], "Its colour."),
                ("names", "NAME PLATES", "On" if st["names"] else "Off", "Names and health over heads."))
        y = 170 * s
        for i, (rid, label, value, note) in enumerate(rows):
            on = i == self.sel
            r = QRectF(54 * s, y, 620 * s, 50 * s)
            p.setPen(NOPEN)
            p.setBrush(QColor(52, 60, 92, 235) if on else QColor(22, 28, 46, 215))
            p.drawPolygon(slant(r, 0.1))
            text(p, QRectF(r.left() + 26 * s, y, 280 * s, 50 * s), label, 19 * s, QColor(255, 255, 255), italic=True)
            text(p, QRectF(r.left() + 300 * s, y, 290 * s, 50 * s), f"◂  {value}  ▸", 19 * s,
                 col(CROSS[st["cross"]][1]) if rid == "cross" else col(GOLD), A.AlignRight | A.AlignVCenter)
            if on:
                text(p, QRectF(r.right() + 14 * s, y, 420 * s, 50 * s), note, 14 * s, QColor(255, 255, 255, 200),
                     bold=False)
            self.rects.append((rid, None, r))
            y += 60 * s
        self._back(p, w, h, s)

    def _paint_stats(self, p: QPainter, w: int, h: int, s: float, t: float) -> None:
        p.fillRect(QRectF(0, 0, w, h), QColor(8, 12, 26, 190))
        self._title(p, s, "STATISTICS")
        st = self.g.stats
        n = max(st["matches"], 1)
        mins = st["time"] / 60.0
        cells = (("MATCHES", f"{st['matches']}"), ("WINS", f"{st['wins']}"),
                 ("WIN RATE", f"{100 * st['wins'] // n}%" if st["matches"] else "-"),
                 ("ELIMINATIONS", f"{st['elims']:,}"), ("FINAL BLOWS", f"{st['final']:,}"),
                 ("DEATHS", f"{st['deaths']:,}"), ("DAMAGE", f"{st['damage']:,}"), ("HEALING", f"{st['healing']:,}"),
                 ("ULTIMATES", f"{st['ults']:,}"), ("BEST ELIMS", f"{st['best_elims']}"),
                 ("TIME PLAYED", f"{int(mins // 60)}h {int(mins % 60):02d}m"),
                 ("CONTROL W-L", f"{st['modes']['control'][0]}-{st['modes']['control'][1]}"),
                 ("ESCORT W-L", f"{st['modes']['escort'][0]}-{st['modes']['escort'][1]}"))
        x, y = 54 * s, 140 * s
        for i, (label, value) in enumerate(cells):
            r = QRectF(x + (i % 5) * 180 * s, y + (i // 5) * 84 * s, 168 * s, 72 * s)
            p.setPen(NOPEN)
            p.setBrush(QColor(22, 28, 46, 225))
            p.drawPolygon(slant(r, 0.1))
            text(p, QRectF(r.left() + 20 * s, r.top() + 6 * s, r.width(), 18 * s), label, 12 * s,
                 QColor(255, 255, 255, 170))
            text(p, QRectF(r.left() + 20 * s, r.top() + 24 * s, r.width(), 42 * s), value, 32 * s,
                 QColor(255, 255, 255), italic=True)
        y += 3 * 84 * s + 10 * s
        heads = (("HERO", 0), ("MATCHES", 250), ("WINS", 340), ("TIME", 430), ("ELIMS", 520), ("DEATHS", 610),
                 ("ELIMS / LIFE", 720))
        for label, dx in heads:
            text(p, QRectF(x + dx * s, y, 150 * s, 20 * s), label, 12 * s, QColor(255, 255, 255, 170))
        y += 22 * s
        for hid in HERO_ORDER:
            hero = HEROES[hid]
            hs = st["heroes"].get(hid, {})
            img = self.g.portrait(hid, True, False, 96)
            if img is not None:
                p.drawImage(QRectF(x, y, 26 * s, 26 * s), img)
            text(p, QRectF(x + 34 * s, y, 150 * s, 26 * s), hero.name, 16 * s, QColor(255, 255, 255), italic=True)
            icon(p, hero.role, QRectF(x + 150 * s, y + 6 * s, 14 * s, 14 * s), col(ROLE_RGB[hero.role]))
            tm = hs.get("time", 0.0) / 60.0
            kd = hs.get("elims", 0) / max(hs.get("deaths", 0), 1)
            for (label, dx), v in zip(heads[1:], (hs.get("matches", 0), hs.get("wins", 0), f"{int(tm)} min",
                                                  hs.get("elims", 0), hs.get("deaths", 0), f"{kd:.1f}"),
                                      strict=True):
                text(p, QRectF(x + dx * s, y, 150 * s, 26 * s), f"{v}", 15 * s, QColor(255, 255, 255, 225),
                     bold=False)
            y += 30 * s
        self._back(p, w, h, s)

    def paint_loading(self, p: QPainter, w: int, h: int, t: float, key: str, mode: str, tip: str) -> None:
        s = h / 780.0
        g = QLinearGradient(0, 0, 0, h)
        g.setColorAt(0, QColor(18, 28, 60))
        g.setColorAt(1, QColor(8, 10, 22))
        p.fillRect(QRectF(0, 0, w, h), g)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        name, _mode, blurb = MAP_INFO[key]
        text(p, QRectF(60 * s, h * 0.34, w, 30 * s), MODE_LABEL[mode], 24 * s, col(GOLD), italic=True)
        text(p, QRectF(58 * s, h * 0.34 + 30 * s, w, 80 * s), name, 72 * s, QColor(255, 255, 255), italic=True)
        text(p, QRectF(60 * s, h * 0.34 + 116 * s, w * 0.8, 26 * s), blurb, 18 * s, QColor(255, 255, 255, 220),
             bold=False)
        text(p, QRectF(60 * s, h * 0.34 + 150 * s, w * 0.8, 26 * s), MODE_BLURB[mode], 15 * s,
             QColor(255, 255, 255, 180), bold=False)
        text(p, QRectF(60 * s, h - 90 * s, w * 0.8, 24 * s), "TIP   " + tip, 15 * s, QColor(255, 255, 255, 200),
             bold=False)
        p.setPen(NOPEN)
        p.setBrush(col(GOLD))
        k = (t * 0.9) % 1.0
        p.drawPolygon(slant(QRectF(60 * s + k * 300 * s, h - 56 * s, 80 * s, 6 * s), 0.9))
