"""The menus: a boot splash, the main menu (dark panels with orange accents, scan lines and glitches), Zombies,
Multiplayer with ten-point Loadouts and scorestreaks, the Service Record, Options, the in-game pause menu and the
briefing loading screen. Everything is drawn with QPainter; mouse and keyboard both work."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen, QPolygonF, QRadialGradient

from . import hud
from .data import (ALL_PERKS, ATTACHMENTS, LETHALS, MAX_LEVEL, PERKS, PRIMARY_IDS, SECONDARY_IDS, STREAKS,
                   TACTICALS, WEAPONS, WILDCARDS, class_cost, level_progress, rank_name, unlocked, validate_class,
                   validate_streaks)
from .maps import MAP_DEFS, MP_MAP, ZOMBIES_MAP

K = Qt.Key
ORANGE = QColor(242, 107, 29)
TEXT = QColor(236, 232, 226)
DIM = QColor(150, 146, 140)


@dataclass
class Item:
    label: str
    action: str
    detail: str = ""
    enabled: bool = True
    value: str = ""  # shown on the right (current choice, setting)


@dataclass
class Page:
    key: str
    title: str
    items: list = field(default_factory=list)
    sel: int = 0
    note: str = ""


def slant(r: QRectF, cut: float) -> QPolygonF:
    """A panel with its top-left and bottom-right corners cut off."""
    return QPolygonF([QPointF(r.left() + cut, r.top()), QPointF(r.right(), r.top()),
                      QPointF(r.right(), r.bottom() - cut), QPointF(r.right() - cut, r.bottom()),
                      QPointF(r.left(), r.bottom()), QPointF(r.left(), r.top() + cut)])


class Menu:
    """The menu stack (``active`` while a page is shown)."""

    def __init__(self, game) -> None:  # noqa: ANN001
        self.g = game
        self.stack: list[Page] = []
        self.hover_i = -1
        self.t = 0.0
        self.glitch = 0.0
        self.message = ("", 0.0)
        self._bg: QImage | None = None
        self.edit_class = 0
        self.slot = ""

    # --- the stack ------------------------------------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return bool(self.stack)

    @property
    def page(self) -> Page | None:
        return self.stack[-1] if self.stack else None

    def close(self) -> None:
        self.stack = []

    def open_root(self) -> None:
        self.stack = [self._root()]
        self.glitch = 0.4

    def open_pause(self) -> None:
        m = self.g.mode
        items = [Item("RESUME", "resume", "Back into the fight."),
                 Item("CONTROLS...", "controls", "Mouse sensitivity, invert Y and key bindings."),
                 Item("OPTIONS", "options", "Field of view, ADS sensitivity.")]
        if m is not None and m.kind == "zombies":
            items.append(Item("END GAME", "end_game", "Leave the game. You can resume this round later from the "
                                                     "Zombies menu."))
        else:
            items.append(Item("LEAVE MATCH", "end_game", "Quit the match (it counts as a loss)."))
        items.append(Item("BACK TO ARCADE", "arcade", "Return to the Break Room Arcade."))
        self.stack = [Page("pause", "PAUSED", items)]

    def push(self, page: Page) -> None:
        self.stack.append(page)
        self.hover_i = -1
        self.glitch = 0.25
        self.g.sfx.play("menu_select")

    def pop(self) -> None:
        if len(self.stack) > 1:
            self.stack.pop()
            self.g.sfx.play("menu_back")
            self._refresh()

    def escape(self) -> bool:
        """Esc: back one page; False at the main menu (the arcade pauses)."""
        if not self.stack:
            return False
        if self.page.key == "pause":
            self.close()
            return True
        if len(self.stack) > 1:
            self.pop()
            return True
        return False

    def _refresh(self) -> None:
        """Rebuild the top page (values changed)."""
        p = self.page
        if p is None:
            return
        builders = {"root": self._root, "zombies": self._zombies, "mp": self._mp, "cac": self._cac,
                    "class": self._class_page, "streaks": self._streaks, "options": self._options,
                    "barracks": self._barracks}
        b = builders.get(p.key)
        if b is not None:
            sel = p.sel
            new = b()
            new.sel = min(sel, max(len(new.items) - 1, 0))
            self.stack[-1] = new

    # --- pages --------------------------------------------------------------------------------------------------

    def _root(self) -> Page:
        return Page("root", "", [
            Item("ZOMBIES", "zombies", "Survive the night shift at the tank works: boarded windows, the Lucky Crate, "
                                       "tonic machines and the Refit Press."),
            Item("MULTIPLAYER", "mp", "Team Deathmatch against bots in the CROSSDOCK freight yard. Ten-point loadouts, "
                                      "scorestreaks, "
                                      "55 levels and prestige."),
            Item("SERVICE RECORD", "barracks", "Your combat record."),
            Item("OPTIONS", "options", "Controls, field of view, ADS sensitivity."),
            Item("BACK TO ARCADE", "arcade", "Return to the Break Room Arcade."),
        ])

    def _zombies(self) -> Page:
        g = self.g
        items = []
        if g.zresume:
            items.append(Item(f"RESUME: ROUND {g.zresume.get('round', 1)}", "z_resume",
                              f"Pick up where you left: {g.zresume.get('points', 0):,} points, the round starts over."))
        items += [Item("NEW GAME: NIGHT SHIFT", "z_new", "Eight rooms of a tank works behind bought doors, the breaker "
                                                        "in the generator room. Solo survival from round 1."),
                  Item("BACK", "back", "")]
        return Page("zombies", "ZOMBIES", items, note=f"Best: round {g.zstats['best_round']}")

    def _mp(self) -> Page:
        g = self.g
        cls = g.profile["classes"][g.profile["cur_class"]]
        items = [Item("FIND MATCH: TEAM DEATHMATCH", "mp_play", "6 v 6 with bots in CROSSDOCK. First to 40 kills "
                                                                "or the most after 8 minutes."),
                 Item("SELECT CLASS", "select_class", "Which class you spawn with.", value=cls["name"]),
                 Item("LOADOUTS", "cac", "Spend 10 points on weapons, attachments, perks, equipment "
                                               "and wildcards."),
                 Item("SCORESTREAKS", "streaks", "Choose 3 scorestreaks.",
                      value=" / ".join(STREAKS[s].name for s in g.profile["streaks"]))]
        if g.can_prestige():
            items.append(Item("ENTER PRESTIGE", "prestige", "Reset to level 1 with a new prestige emblem."))
        items.append(Item("BACK", "back", ""))
        return Page("mp", "MULTIPLAYER", items)

    def _cac(self) -> Page:
        items = [Item(c["name"], f"edit:{i}", self._class_summary(c), value=f"{class_cost(c)}/10")
                 for i, c in enumerate(self.g.profile["classes"])]
        items.append(Item("BACK", "back", ""))
        return Page("cac", "LOADOUTS", items)

    def _class_summary(self, c: dict) -> str:
        parts = [WEAPONS[c["primary"]].name if c.get("primary") else "No primary"]
        if c.get("secondary"):
            parts.append(WEAPONS[c["secondary"]].name)
        perks = [ALL_PERKS[p].name for k in ("perk1", "perk2", "perk3") for p in c.get(k, [])]
        return " · ".join(parts + perks)

    def _class_page(self) -> Page:
        c = self.g.profile["classes"][self.edit_class]
        wild = c.get("wildcards", [])

        def name(table: dict, i: str | None) -> str:
            return table[i].name if i in table else "—"

        def nth(lst: list, n: int) -> str | None:
            return lst[n] if n < len(lst) else None

        items = [Item("PRIMARY", "slot:primary", "Your main weapon.", value=name(WEAPONS, c.get("primary")))]
        for n in range(3 if "primegunner" in wild else 2):
            items.append(Item(f"  ATTACHMENT {n + 1}", f"slot:primary_att:{n}", "Primary attachment.",
                              value=name(ATTACHMENTS, nth(c.get("primary_att", []), n))))
        items.append(Item("SECONDARY", "slot:secondary", "Your sidearm.", value=name(WEAPONS, c.get("secondary"))))
        for n in range(2 if "secgunner" in wild else 1):
            items.append(Item(f"  ATTACHMENT {n + 1}", f"slot:secondary_att:{n}", "Secondary attachment.",
                              value=name(ATTACHMENTS, nth(c.get("secondary_att", []), n))))
        for tier in (1, 2, 3):
            for n in range(2 if f"perk{tier}glutton" in wild else 1):
                items.append(Item(f"PERK {tier}" + (" (2)" if n else ""), f"slot:perk{tier}:{n}",
                                  "Perks change how you play.",
                                  value=name(ALL_PERKS, nth(c.get(f"perk{tier}", []), n))))
        for key, table, wc, label in (("lethal", LETHALS, "dangerclose", "LETHAL"),
                                      ("tactical", TACTICALS, "tactician", "TACTICAL")):
            for n in range(2 if wc in wild else 1):
                items.append(Item(label + (" (2)" if n else ""), f"slot:{key}:{n}", "Thrown with G / Q.",
                                  value=name(table, nth(c.get(key, []), n))))
        for n in range(3):
            items.append(Item(f"WILDCARD {n + 1}", f"slot:wildcards:{n}", "Break the rules for a point.",
                              value=name(WILDCARDS, nth(wild, n))))
        items.append(Item("RENAME (CYCLE)", "rename", "Cycle a name for this class."))
        items.append(Item("BACK", "back", ""))
        return Page("class", c["name"], items, note=f"{class_cost(c)} / 10 POINTS")

    def _chooser(self, slot: str) -> Page:
        g = self.g
        c = g.profile["classes"][self.edit_class]
        key, _, n = slot.partition(":")
        lv, pr = g.level, g.profile["prestige"]
        opts: list[tuple[str, str, int]] = []  # (id, name, unlock level)
        if key == "primary":
            ids = list(PRIMARY_IDS)
            opts = [(i, WEAPONS[i].name, WEAPONS[i].level) for i in ids]
        elif key == "secondary":
            ids = list(SECONDARY_IDS) + (list(PRIMARY_IDS) if "overkill" in c.get("wildcards", []) else [])
            opts = [(i, WEAPONS[i].name, WEAPONS[i].level) for i in ids]
        elif key in ("primary_att", "secondary_att"):
            wid = c.get("primary" if key == "primary_att" else "secondary")
            if wid:
                opts = [(a.id, a.name, a.level) for a in ATTACHMENTS.values() if WEAPONS[wid].cls in a.kinds]
        elif key.startswith("perk"):
            opts = [(p.id, p.name, p.level) for p in PERKS[int(key[4])].values()]
        elif key == "lethal":
            opts = [(i.id, i.name, i.level) for i in LETHALS.values()]
        elif key == "tactical":
            opts = [(i.id, i.name, i.level) for i in TACTICALS.values()]
        elif key == "wildcards":
            opts = [(i.id, i.name, i.level) for i in WILDCARDS.values()]
        items = []
        if key != "primary":
            items.append(Item("NONE", "pick:", "Empty the slot (frees a point)."))
        tables = (WEAPONS, ATTACHMENTS, ALL_PERKS, LETHALS, TACTICALS, WILDCARDS)
        for oid, oname, need in opts:
            ok = unlocked(lv, pr, need)
            desc = next((getattr(t[oid], "desc", "") for t in tables if oid in t and hasattr(t[oid], "desc")), "")
            if oid in WEAPONS:
                w = WEAPONS[oid]
                desc = f"{w.cls.upper()} · damage {w.dmg:.0f} · {w.rpm:.0f} rpm · mag {w.mag}"
            items.append(Item(oname if ok else f"{oname}  [LV {need}]", f"pick:{oid}", desc if ok else
                              f"Unlocks at level {need}.", ok))
        self.slot = slot
        return Page("chooser", key.upper().replace("_", " "), items)

    def _streaks(self) -> Page:
        g = self.g
        chosen = g.profile["streaks"]
        items = []
        for s in STREAKS.values():
            ok = unlocked(g.level, g.profile["prestige"], s.level)
            items.append(Item(s.name if ok else f"{s.name}  [LV {s.level}]", f"streak:{s.id}",
                              f"{s.cost} score. {s.desc}" if ok else f"Unlocks at level {s.level}.", ok,
                              value="✔" if s.id in chosen else ""))
        items.append(Item("BACK", "back", ""))
        return Page("streaks", "SCORESTREAKS", items, note="Pick 3")

    def _options(self) -> Page:
        s = self.g.settings
        return Page("options", "OPTIONS", [
            Item("CONTROLS...", "controls", "Mouse sensitivity, invert Y, swapped buttons, turn speed and key "
                                            "bindings (the arcade's Controls panel)."),
            Item("FIELD OF VIEW", "opt:fov", "Left / right to change.", value=f"{s['fov']}"),
            Item("ADS SENSITIVITY", "opt:ads", "Mouse look while aiming down the sights, on top of the mouse "
                                              "sensitivity. Left / right to change.", value=f"{s['ads_sens']:.2f}×"),
            Item("BACK", "back", ""),
        ])

    def _barracks(self) -> Page:
        return Page("barracks", "SERVICE RECORD", [Item("BACK", "back", "")])

    # --- input --------------------------------------------------------------------------------------------------

    def key(self, key: int, event) -> None:  # noqa: ANN001
        p = self.page
        if p is None:
            return
        n = len(p.items)
        if key in (K.Key_Down, K.Key_S, K.Key_Tab):
            p.sel = (p.sel + 1) % n
            self.g.sfx.play("menu_move")
        elif key in (K.Key_Up, K.Key_W, K.Key_Backtab):
            p.sel = (p.sel - 1) % n
            self.g.sfx.play("menu_move")
        elif key in (K.Key_Left, K.Key_A, K.Key_Right, K.Key_D):
            self.adjust(p.items[p.sel].action, 1 if key in (K.Key_Right, K.Key_D) else -1)
        elif key in (K.Key_Return, K.Key_Enter, K.Key_Space, K.Key_F):
            self.activate(p.sel)
        elif key == K.Key_Backspace:
            self.escape()

    def item_rects(self, w: int, h: int) -> list[QRectF]:
        p = self.page
        if p is None:
            return []
        s = h / 780
        n = len(p.items)
        row = min(46 * s, (h * 0.66) / max(n, 1))
        top = h * 0.25 if p.key != "pause" else h * 0.32
        x = w * 0.07
        return [QRectF(x, top + i * row, w * 0.36, row * 0.86) for i in range(n)]

    def hover(self, pos) -> None:  # noqa: ANN001
        rects = self.item_rects(self.g.width(), self.g.height())
        i = next((k for k, r in enumerate(rects) if r.contains(pos)), -1)
        if i != self.hover_i:
            self.hover_i = i
            if i >= 0 and self.page is not None:
                if self.page.sel != i:
                    self.g.sfx.play("menu_move")
                self.page.sel = i

    def click(self, pos, button) -> None:  # noqa: ANN001
        if button == Qt.MouseButton.RightButton:
            self.escape()
            return
        rects = self.item_rects(self.g.width(), self.g.height())
        i = next((k for k, r in enumerate(rects) if r.contains(pos)), -1)
        if i >= 0:
            self.page.sel = i
            self.activate(i)

    def wheel(self, steps: int) -> None:
        p = self.page
        if p is not None and p.items:
            p.sel = (p.sel - steps) % len(p.items)
            self.g.sfx.play("menu_move")

    def flash(self, msg: str) -> None:
        self.message = (msg, 2.5)

    def adjust(self, action: str, d: int) -> None:
        s = self.g.settings
        if action == "opt:fov":
            s["fov"] = int(min(max(s["fov"] + d * 5, 65), 110))
        elif action == "opt:ads":
            s["ads_sens"] = round(min(max(s["ads_sens"] + d * 0.05, 0.3), 1.5), 2)
        elif action == "select_class":
            pr = self.g.profile
            pr["cur_class"] = (pr["cur_class"] + d) % 5
        else:
            return
        self.g.sfx.play("menu_move")
        self._refresh()

    def activate(self, i: int) -> None:  # noqa: C901 - one switch over the actions
        p = self.page
        if p is None or not 0 <= i < len(p.items):
            return
        item = p.items[i]
        g = self.g
        a = item.action
        if not item.enabled:
            g.sfx.play("deny")
            self.flash(item.detail)
            return
        if a == "back":
            self.pop()
        elif a == "arcade":
            g.sfx.play("menu_back")
            if g.screen == "play":
                g.end_mode()
            self.close()
            g.exit_requested.emit()
        elif a == "zombies":
            self.push(self._zombies())
        elif a == "mp":
            self.push(self._mp())
        elif a == "barracks":
            self.push(self._barracks())
        elif a == "options":
            self.push(self._options())
        elif a in ("z_new", "z_resume"):
            g.start_mode("zombies", resume=a == "z_resume")
        elif a == "mp_play":
            g.start_mode("mp")
        elif a == "select_class":
            self.adjust(a, 1)
        elif a == "cac":
            self.push(self._cac())
        elif a == "streaks":
            self.push(self._streaks())
        elif a == "prestige":
            if g.prestige():
                self._refresh()
        elif a.startswith("edit:"):
            self.edit_class = int(a[5:])
            self.push(self._class_page())
        elif a.startswith("slot:"):
            self.push(self._chooser(a[5:]))
        elif a.startswith("pick:"):
            self._pick(a[5:] or None)
        elif a == "rename":
            names = ("ASSAULT", "SMG RUSH", "MARKSMAN", "RUN AND GUN", "OBJECTIVE", "HEAVY", "STEALTH", "CQB",
                     "SUPPORT", "LONG RANGE")
            c = g.profile["classes"][self.edit_class]
            c["name"] = names[(names.index(c["name"]) + 1) % len(names)] if c["name"] in names else names[0]
            self._refresh()
        elif a.startswith("streak:"):
            sid = a[7:]
            chosen = list(g.profile["streaks"])
            if sid in chosen:
                self.flash("Pick another scorestreak to swap out your oldest pick.")
            else:
                chosen.append(sid)
                if len(chosen) > 3:
                    chosen.pop(0)
                g.profile["streaks"] = chosen
            g.profile["streaks"].sort(key=lambda s: STREAKS[s].cost)
            g.sfx.play("menu_select")
            self._refresh()
        elif a.startswith("opt:"):
            self.adjust(a, 1)
        elif a == "controls":
            g.sfx.play("menu_select")
            g.open_controls()
        elif a == "resume":
            self.close()
            g.sfx.play("menu_select")
        elif a == "end_game":
            self.close()
            if g.mode is not None and g.mode.kind == "mp" and not g.mode.over:
                g.mode.forfeit()
                g.mode.record(g.mpstats)
            g.end_mode()

    def _pick(self, oid: str | None) -> None:
        g = self.g
        c = g.profile["classes"][self.edit_class]
        key, _, n = self.slot.partition(":")
        new = {k: (list(v) if isinstance(v, list) else v) for k, v in c.items()}
        if key in ("primary", "secondary"):
            new[key] = oid
            if oid is None or (c.get(key) != oid):
                new[key + "_att"] = [a for a in new.get(key + "_att", [])
                                     if oid and WEAPONS[oid].cls in ATTACHMENTS[a].kinds]
        else:
            lst = list(new.get(key, []))
            k = int(n or 0)
            if oid is None:
                if k < len(lst):
                    lst.pop(k)
            elif k < len(lst):
                lst[k] = oid
            else:
                lst.append(oid)
            new[key] = lst
            if key == "wildcards":
                # dropping a wildcard trims what it allowed
                w = lst
                if "primegunner" not in w:
                    new["primary_att"] = new.get("primary_att", [])[:2]
                if "secgunner" not in w:
                    new["secondary_att"] = new.get("secondary_att", [])[:1]
                for t in (1, 2, 3):
                    if f"perk{t}glutton" not in w:
                        new[f"perk{t}"] = new.get(f"perk{t}", [])[:1]
                if "dangerclose" not in w:
                    new["lethal"] = new.get("lethal", [])[:1]
                if "tactician" not in w:
                    new["tactical"] = new.get("tactical", [])[:1]
                if "overkill" not in w and new.get("secondary") in PRIMARY_IDS:
                    new["secondary"] = "service45"
                    new["secondary_att"] = []
        errs = validate_class(new, g.level, g.profile["prestige"])
        if errs:
            g.sfx.play("deny")
            self.flash(errs[0])
            return
        g.profile["classes"][self.edit_class] = new
        self.stack.pop()
        g.sfx.play("menu_select")
        self._refresh()

    def tick(self, dt: float) -> None:
        self.t += dt
        self.glitch = max(self.glitch - dt, 0.0)
        if self.message[1] > 0:
            self.message = (self.message[0], self.message[1] - dt)
        if random.random() < dt * 0.25:
            self.glitch = max(self.glitch, 0.12)
        if self.page is not None and self.page.key == "streaks":
            if len(self.g.profile["streaks"]) != 3 or validate_streaks(self.g.profile["streaks"], 99, 1):
                pass

    # --- painting -------------------------------------------------------------------------------------------------

    def background(self, w: int, h: int) -> QImage:
        if self._bg is None or self._bg.width() != w or self._bg.height() != h:
            self._bg = _menu_bg(w, h)
        return self._bg

    def paint_background(self, p: QPainter, w: int, h: int, t: float) -> None:
        p.drawImage(0, 0, self.background(w, h))
        p.save()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(22):
            k = (t * 0.05 + i / 22) % 1.0
            x = w * ((i * 0.173 + 0.03 * math.sin(t * 0.7 + i)) % 1.0)
            y = h * (1.05 - k * 1.1)
            p.setBrush(QColor(255, 130, 40, int(160 * math.sin(k * math.pi))))
            p.drawEllipse(QPointF(x, y), 1.8, 1.8)
        p.restore()

    def _scanlines(self, p: QPainter, w: int, h: int) -> None:
        p.save()
        p.setPen(QPen(QColor(0, 0, 0, 40), 1))
        for y in range(0, h, 3):
            p.drawLine(0, y, w, y)
        if self.glitch > 0:
            rng = random.Random(int(self.t * 30))
            for _ in range(4):
                y = rng.uniform(0, h)
                hh = rng.uniform(2, 14)
                p.fillRect(QRectF(0, y, w, hh), QColor(242, 107, 29, rng.randint(20, 70)))
        p.restore()

    def paint(self, p: QPainter, w: int, h: int, t: float) -> None:
        page = self.page
        if page is None:
            return
        s = h / 780
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if page.key == "pause":
            p.fillRect(QRectF(0, 0, w, h), QColor(0, 0, 0, 160))
        if page.key == "root":
            self._logo(p, w * 0.07, h * 0.06, h * 0.11, t)
        else:
            hud.text(p, QRectF(w * 0.07, h * 0.1, w * 0.6, 60 * s), page.title, 48 * s, TEXT,
                     Qt.AlignmentFlag.AlignLeft, True, 2, "Impact")
            p.fillRect(QRectF(w * 0.07, h * 0.1 + 62 * s, 90 * s, 4 * s), ORANGE)
            if page.note:
                hud.text(p, QRectF(w * 0.07 + 100 * s, h * 0.1 + 50 * s, w * 0.4, 26 * s), page.note, 18 * s, ORANGE,
                         Qt.AlignmentFlag.AlignLeft)
        rects = self.item_rects(w, h)
        for i, (item, r) in enumerate(zip(page.items, rects, strict=True)):
            on = i == page.sel
            if on:
                g = QLinearGradient(r.topLeft(), r.topRight())
                g.setColorAt(0, QColor(242, 107, 29, 230))
                g.setColorAt(1, QColor(242, 107, 29, 40))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(g)
                p.drawPolygon(slant(r, r.height() * 0.3))
                p.fillRect(QRectF(r.left() - 10 * s, r.top(), 4 * s, r.height()), ORANGE)
            col = QColor(20, 12, 6) if on else (TEXT if item.enabled else DIM)
            size = min(r.height() * 0.52, 24 * s)
            hud.text(p, r.adjusted(16 * s, 0, 0, 0), item.label, size, col, Qt.AlignmentFlag.AlignLeft, True,
                     0 if on else 2)
            if item.value:
                hud.text(p, r.adjusted(0, 0, -14 * s, 0), item.value, size * 0.85,
                         QColor(20, 12, 6) if on else ORANGE.lighter(120), Qt.AlignmentFlag.AlignRight, True,
                         0 if on else 1)
        # detail panel
        if page.items:
            item = page.items[page.sel]
            dr = QRectF(w * 0.52, h * 0.25, w * 0.41, h * 0.5)
            if page.key in ("class", "chooser"):
                self._paint_class_panel(p, dr, s)
            elif page.key == "barracks":
                self._paint_barracks(p, QRectF(w * 0.07, h * 0.25 + 60 * s, w * 0.86, h * 0.6), s)
            elif page.key == "mp":
                self._paint_class_preview(p, dr, s)
            elif item.detail:
                self._panel(p, QRectF(dr.x(), dr.y(), dr.width(), 120 * s))
                hud.text(p, QRectF(dr.x() + 20 * s, dr.y() + 12 * s, dr.width() - 40 * s, 30 * s), item.label.strip(),
                         22 * s, ORANGE, Qt.AlignmentFlag.AlignLeft)
                p.setFont(hud.hfont(16 * s, False))
                p.setPen(TEXT)
                p.drawText(QRectF(dr.x() + 20 * s, dr.y() + 46 * s, dr.width() - 40 * s, 70 * s),
                           int(Qt.TextFlag.TextWordWrap), item.detail)
            if page.key == "chooser" and item.detail:
                hud.text(p, QRectF(dr.x(), dr.bottom() + 10 * s, dr.width(), 40 * s), item.detail, 15 * s, DIM,
                         Qt.AlignmentFlag.AlignLeft, False)
        if self.message[1] > 0:
            hud.text(p, QRectF(0, h * 0.9, w, 30 * s), self.message[0], 18 * s, QColor(255, 90, 70),
                     Qt.AlignmentFlag.AlignHCenter)
        if page.key in ("root", "mp", "zombies", "cac", "barracks"):
            self._player_card(p, QRectF(w * 0.07, h - 92 * s, 360 * s, 62 * s), s)
        hud.text(p, QRectF(0, h - 34 * s, w - 24 * s, 24 * s), "ENTER select   ESC back   ← → change", 13 * s, DIM,
                 Qt.AlignmentFlag.AlignRight, False)
        self._scanlines(p, w, h)

    def _panel(self, p: QPainter, r: QRectF) -> None:
        p.setPen(QPen(QColor(242, 107, 29, 120), 1.2))
        p.setBrush(QColor(10, 8, 8, 200))
        p.drawPolygon(slant(r, 14))

    def _logo(self, p: QPainter, x: float, y: float, size: float, t: float) -> None:
        """The game's own logo: SHADOW OPS in heavy letters, a slashed orange II."""
        f = hud.hfont(size, True, "Impact")
        p.setFont(f)
        jitter = 3 if self.glitch > 0 else 0
        if jitter:
            p.setPen(QColor(0, 200, 255, 120))
            p.drawText(QPointF(x - jitter, y + size), "SHADOW OPS")
            p.setPen(QColor(255, 40, 40, 120))
            p.drawText(QPointF(x + jitter, y + size), "SHADOW OPS")
        p.setPen(QColor(0, 0, 0, 180))
        p.drawText(QPointF(x + 3, y + size + 3), "SHADOW OPS")
        p.setPen(TEXT)
        p.drawText(QPointF(x, y + size), "SHADOW OPS")
        wtxt = p.fontMetrics().horizontalAdvance("SHADOW OPS ")
        p.setPen(ORANGE)
        p.drawText(QPointF(x + wtxt, y + size), "II")
        p.setPen(QPen(QColor(10, 8, 8), size * 0.07))
        iw = p.fontMetrics().horizontalAdvance("II")
        p.drawLine(QPointF(x + wtxt - 4, y + size * 0.75), QPointF(x + wtxt + iw + 4, y + size * 0.45))
        p.fillRect(QRectF(x, y + size * 1.15, size * 4.5, 3), ORANGE)

    def _player_card(self, p: QPainter, r: QRectF, s: float) -> None:
        g = self.g
        lv, into, need = level_progress(g.profile["xp"])
        self._panel(p, r)
        _rank_icon(p, QRectF(r.x() + 10 * s, r.y() + 8 * s, 46 * s, 46 * s), lv, g.profile["prestige"])
        pr = g.profile["prestige"]
        label = f"LEVEL {lv}" + (f"  ·  PRESTIGE {pr}" if pr else "")
        hud.text(p, QRectF(r.x() + 66 * s, r.y() + 6 * s, r.width() - 76 * s, 24 * s), label, 18 * s, TEXT)
        hud.text(p, QRectF(r.x() + 66 * s, r.y() + 26 * s, r.width() - 76 * s, 18 * s), rank_name(lv), 13 * s, DIM,
                 bold=False)
        bar = QRectF(r.x() + 66 * s, r.y() + 47 * s, r.width() - 80 * s, 6 * s)
        p.fillRect(bar, QColor(255, 255, 255, 40))
        frac = 1.0 if lv >= MAX_LEVEL else into / max(need, 1)
        p.fillRect(QRectF(bar.x(), bar.y(), bar.width() * frac, bar.height()), ORANGE)

    def _paint_class_preview(self, p: QPainter, r: QRectF, s: float) -> None:
        c = self.g.profile["classes"][self.g.profile["cur_class"]]
        self._class_card(p, r, c, s)

    def _paint_class_panel(self, p: QPainter, r: QRectF, s: float) -> None:
        c = self.g.profile["classes"][self.edit_class]
        self._class_card(p, r, c, s)

    def _class_card(self, p: QPainter, r: QRectF, c: dict, s: float) -> None:
        from .sprites import SpriteSet  # noqa: F401 - only the silhouettes below
        from .textures import gun_path

        self._panel(p, r)
        hud.text(p, QRectF(r.x() + 20 * s, r.y() + 10 * s, r.width(), 30 * s), c["name"], 24 * s, ORANGE)
        cost = class_cost(c)
        for i in range(10):
            pip = QRectF(r.right() - 24 * s - (10 - i) * 16 * s, r.y() + 20 * s, 12 * s, 12 * s)
            p.fillRect(pip, ORANGE if i < cost else QColor(255, 255, 255, 40))
        y = r.y() + 52 * s
        for key in ("primary", "secondary"):
            wid = c.get(key)
            if not wid:
                continue
            w = WEAPONS[wid]
            p.save()
            p.translate(r.x() + 20 * s, y)
            path = gun_path(w.model, 150 * s, 60 * s)
            g = QLinearGradient(0, 0, 0, 60 * s)
            g.setColorAt(0, QColor(150, 150, 156))
            g.setColorAt(1, QColor(50, 50, 56))
            p.setBrush(g)
            p.setPen(QPen(QColor(255, 255, 255, 60), 1))
            p.drawPath(path)
            p.restore()
            atts = ", ".join(ATTACHMENTS[a].name for a in c.get(key + "_att", [])) or "No attachments"
            hud.text(p, QRectF(r.x() + 185 * s, y + 6 * s, r.width() - 200 * s, 26 * s), w.name, 19 * s, TEXT)
            hud.text(p, QRectF(r.x() + 185 * s, y + 30 * s, r.width() - 200 * s, 22 * s), atts, 14 * s, DIM,
                     bold=False)
            y += 66 * s
        colors = {1: QColor(70, 150, 255), 2: QColor(90, 210, 90), 3: QColor(230, 70, 60)}
        for tier in (1, 2, 3):
            for pid in c.get(f"perk{tier}", []):
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(colors[tier])
                p.drawRoundedRect(QRectF(r.x() + 20 * s, y + 4 * s, 16 * s, 16 * s), 3, 3)
                hud.text(p, QRectF(r.x() + 44 * s, y, r.width() - 60 * s, 24 * s),
                         f"{ALL_PERKS[pid].name} — {ALL_PERKS[pid].desc}", 14 * s, TEXT, bold=False)
                y += 24 * s
        eq = [LETHALS[i].name for i in c.get("lethal", [])] + [TACTICALS[i].name for i in c.get("tactical", [])]
        wc = [WILDCARDS[i].name for i in c.get("wildcards", [])]
        if eq:
            hud.text(p, QRectF(r.x() + 20 * s, y + 4 * s, r.width() - 40 * s, 22 * s), "Equipment: " + ", ".join(eq),
                     14 * s, DIM, bold=False)
            y += 22 * s
        if wc:
            hud.text(p, QRectF(r.x() + 20 * s, y + 4 * s, r.width() - 40 * s, 22 * s), "Wildcards: " + ", ".join(wc),
                     14 * s, ORANGE, bold=False)

    def _paint_barracks(self, p: QPainter, r: QRectF, s: float) -> None:
        g = self.g
        mp, z = g.mpstats, g.zstats
        kd = mp["kills"] / max(mp["deaths"], 1)
        acc = mp["hits"] / max(mp["shots"], 1) * 100
        cols = (("MULTIPLAYER", [("Kills", f"{mp['kills']:,}"), ("Deaths", f"{mp['deaths']:,}"), ("K/D", f"{kd:.2f}"),
                                 ("Accuracy", f"{acc:.1f}%"), ("Headshots", f"{mp['headshots']:,}"),
                                 ("Wins", f"{mp['wins']}"), ("Losses", f"{mp['losses']}"),
                                 ("Best streak", f"{mp['best_streak']}"), ("Score", f"{mp['score']:,}")]),
                ("ZOMBIES", [("Highest round", f"{z['best_round']}"), ("Games", f"{z['games']}"),
                             ("Kills", f"{z['kills']:,}"), ("Headshots", f"{z['headshots']:,}"),
                             ("Downs", f"{z['downs']}"), ("Points earned", f"{z['points']:,}")]),
                ("RANK", [("Level", f"{g.level}"), ("Prestige", f"{g.profile['prestige']}"),
                          ("Rank", rank_name(g.level)), ("Total XP", f"{g.profile['xp']:,}")]))
        cw = r.width() / 3 - 16 * s
        for i, (title, rows) in enumerate(cols):
            pr = QRectF(r.x() + i * (cw + 24 * s), r.y(), cw, min(r.height(), (len(rows) + 2) * 30 * s + 10 * s))
            self._panel(p, pr)
            hud.text(p, QRectF(pr.x() + 18 * s, pr.y() + 10 * s, cw, 30 * s), title, 22 * s, ORANGE)
            for k, (label, val) in enumerate(rows):
                y = pr.y() + 52 * s + k * 30 * s
                hud.text(p, QRectF(pr.x() + 18 * s, y, cw * 0.6, 26 * s), label, 16 * s, DIM, bold=False)
                hud.text(p, QRectF(pr.x(), y, cw - 18 * s, 26 * s), val, 17 * s, TEXT, Qt.AlignmentFlag.AlignRight)

    def paint_boot(self, p: QPainter, w: int, h: int, t: float) -> None:
        p.fillRect(QRectF(0, 0, w, h), QColor(4, 3, 3))
        s = h / 780
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if t < 1.2:
            a = int(255 * min(t / 0.4, 1.0) * min((1.2 - t) / 0.3, 1.0))
            hud.text(p, QRectF(0, h * 0.45, w, 30 * s), "BREAK ROOM ARCADE  PRESENTS", 18 * s, QColor(200, 200, 200, a),
                     Qt.AlignmentFlag.AlignHCenter, False)
            return
        k = min((t - 1.2) / 0.6, 1.0)
        g = QRadialGradient(QPointF(w / 2, h / 2), w * 0.5)
        g.setColorAt(0, QColor(242, 107, 29, int(110 * k)))
        g.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(QRectF(0, 0, w, h), g)
        size = h * 0.13
        p.setFont(hud.hfont(size, True, "Impact"))
        tw = p.fontMetrics().horizontalAdvance("SHADOW OPS II")
        self.glitch = 0.2 if t < 2.0 else self.glitch
        self._logo(p, (w - tw) / 2, h * 0.38, size, t)
        hud.text(p, QRectF(0, h * 0.62, w, 26 * s), "ZOMBIES  ·  MULTIPLAYER", 16 * s,
                 QColor(220, 220, 220, int(200 * k)), Qt.AlignmentFlag.AlignHCenter, False)
        self._scanlines(p, w, h)

    def paint_loading(self, p: QPainter, w: int, h: int, t: float, brief: tuple, ready: bool) -> None:
        """A briefing: the map name, the operation, typed lines, the map drawn as a plan, a progress bar."""
        title, op, lines = brief
        s = h / 780
        p.fillRect(QRectF(0, 0, w, h), QColor(6, 5, 5))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = QRadialGradient(QPointF(w * 0.7, h * 0.5), w * 0.6)
        g.setColorAt(0, QColor(70, 30, 10))
        g.setColorAt(1, QColor(6, 5, 5))
        p.fillRect(QRectF(0, 0, w, h), g)
        key = ZOMBIES_MAP if title == MAP_DEFS[ZOMBIES_MAP]["title"] else MP_MAP
        self._plan(p, QRectF(w * 0.5, h * 0.18, w * 0.44, h * 0.56), key, t)
        hud.text(p, QRectF(w * 0.07, h * 0.16, w * 0.5, 30 * s), op, 18 * s, ORANGE)
        hud.text(p, QRectF(w * 0.07, h * 0.2, w * 0.5, 80 * s), title, 64 * s, TEXT, Qt.AlignmentFlag.AlignLeft, True,
                 3, "Impact")
        shown = int(t * 45)
        y = h * 0.36
        for line in lines:
            part = line[:max(shown, 0)]
            shown -= len(line) + 8
            hud.text(p, QRectF(w * 0.07, y, w * 0.42, 26 * s), part, 17 * s, QColor(210, 205, 196), bold=False)
            y += 32 * s
        bar = QRectF(w * 0.07, h * 0.82, w * 0.86, 5 * s)
        p.fillRect(bar, QColor(255, 255, 255, 30))
        frac = min(t / 2.2, 1.0) if ready else min(t / 2.2, 0.6)
        p.fillRect(QRectF(bar.x(), bar.y(), bar.width() * frac, bar.height()), ORANGE)
        if ready and t > 0.6 and int(t * 2) % 2 == 0:
            hud.text(p, QRectF(0, h * 0.86, w, 30 * s), "PRESS ANY KEY TO DEPLOY", 18 * s, TEXT,
                     Qt.AlignmentFlag.AlignHCenter)
        self._scanlines(p, w, h)

    def _plan(self, p: QPainter, r: QRectF, key: str, t: float) -> None:
        rows = MAP_DEFS[key]["rows"]()
        mh, mw = len(rows), len(rows[0])
        cs = min(r.width() / mw, r.height() / mh)
        ox, oy = r.x() + (r.width() - cs * mw) / 2, r.y() + (r.height() - cs * mh) / 2
        reveal = min(t / 1.6, 1.0)
        for y, row in enumerate(rows):
            if y / mh > reveal:
                break
            for x, ch in enumerate(row):
                if ch in "#MbiCRwDX12345678":
                    col = QColor(242, 107, 29, 170) if ch not in "DX" else QColor(255, 220, 120, 200)
                elif ch in "Wr=oOB%@d":
                    col = QColor(242, 107, 29, 70)
                elif ch in "JSUQNG?K!":
                    col = QColor(120, 220, 255, 200)
                else:
                    continue
                p.fillRect(QRectF(ox + x * cs, oy + y * cs, cs * 0.92, cs * 0.92), col)
        sweep = oy + (t * 0.35 % 1.0) * cs * mh
        p.fillRect(QRectF(ox, sweep, cs * mw, 2), QColor(242, 107, 29, 90))


def _rank_icon(p: QPainter, r: QRectF, level: int, prestige: int) -> None:
    """Chevrons for the level, a star badge for prestige."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    if prestige:
        g = QRadialGradient(r.center(), r.width() / 2)
        g.setColorAt(0, QColor(255, 220, 120))
        g.setColorAt(1, QColor(180, 90, 20))
        p.setBrush(g)
        p.setPen(QPen(QColor(40, 20, 4), 1.5))
        poly = QPolygonF()
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = r.width() / 2 * (1.0 if i % 2 == 0 else 0.45)
            poly.append(QPointF(r.center().x() + math.cos(a) * rr, r.center().y() + math.sin(a) * rr))
        p.drawPolygon(poly)
        hud.text(p, r, str(prestige), r.height() * 0.32, QColor(40, 20, 4), Qt.AlignmentFlag.AlignHCenter, True, 0)
        p.restore()
        return
    n = 1 + min((level - 1) // 11, 4)
    p.setPen(Qt.PenStyle.NoPen)
    for i in range(n):
        y = r.bottom() - r.height() * 0.2 - i * r.height() * 0.17
        poly = QPolygonF([QPointF(r.left() + r.width() * 0.1, y), QPointF(r.center().x(), y - r.height() * 0.18),
                          QPointF(r.right() - r.width() * 0.1, y), QPointF(r.right() - r.width() * 0.1,
                                                                            y + r.height() * 0.08),
                          QPointF(r.center().x(), y - r.height() * 0.1), QPointF(r.left() + r.width() * 0.1,
                                                                                 y + r.height() * 0.08)])
        p.setBrush(QColor(242, 107, 29) if i == n - 1 else QColor(200, 200, 200))
        p.drawPolygon(poly)
    p.restore()


def _menu_bg(w: int, h: int) -> QImage:
    """Smoke (value noise), an orange glow and a burning town skyline."""
    from .textures import value_noise

    small_w, small_h = 192, max(int(192 * h / max(w, 1)), 24)
    n = value_noise(256, 6, 31, 5)[:small_h, :small_w]
    yy = np.linspace(0, 1, small_h)[:, None]
    xx = np.linspace(0, 1, small_w)[None, :]
    glow = np.exp(-(((xx - 0.72) / 0.35) ** 2 + ((yy - 0.55) / 0.45) ** 2))
    r = 10 + 150 * glow * (0.6 + 0.6 * n) + 30 * n
    g = 6 + 60 * glow * (0.5 + 0.6 * n) + 18 * n
    b = 6 + 18 * glow + 14 * n
    arr = np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    small = QImage(arr.data, small_w, small_h, small_w * 3, QImage.Format.Format_RGB888).copy()
    img = QImage(max(w, 1), max(h, 1), QImage.Format.Format_RGB32)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.drawImage(QRectF(0, 0, w, h), small)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(8, 5, 4))
    rng = random.Random(3)
    x = w * 0.45
    while x < w:
        bw = rng.uniform(0.04, 0.09) * w
        bh = rng.uniform(0.12, 0.3) * h
        p.drawRect(QRectF(x, h - bh, bw, bh))
        x += bw * rng.uniform(0.8, 1.05)
    p.fillRect(QRectF(0, h * 0.9, w, h * 0.1), QColor(8, 5, 4))
    g2 = QLinearGradient(0, 0, w * 0.55, 0)
    g2.setColorAt(0, QColor(0, 0, 0, 210))
    g2.setColorAt(1, QColor(0, 0, 0, 0))
    p.fillRect(QRectF(0, 0, w * 0.6, h), g2)
    p.end()
    return img
