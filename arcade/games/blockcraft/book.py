"""BlockCraft's two guides (mixed into the game widget):

* the **recipe book** (R, or the game menu): everything that can be crafted or smelted, a page of results on the
  left and how the chosen one is made on the right; what the player can make right now is marked green;
* the **advancements** (L, or the game menu): a list of things to do, roughly in the order a new player meets them,
  ticked off as they happen. They are kept with the world."""

from __future__ import annotations

import math

from PySide6.QtCore import QRect, QRectF
from PySide6.QtGui import QColor, QPainter

from app.arcade.games.blockcraft import blocks as B
from app.arcade.games.blockcraft import gui as G
from app.arcade.games.blockcraft import items as I

COLS, ROWS = 6, 7  # results on a page of the recipe book
PANEL = (252, 170)
ADV_ROWS = 7
STATIONS = {"hand": "Made in the inventory (2 x 2)", "table": "Made at a Crafting Table", "furnace": "Smelted in a Furnace"}


def _recipes() -> list[dict]:
    """Every recipe as {"result": (item, count), "cells": nine tuples of the items that fit a square, "station"}."""
    out = []
    for pattern, keys, result in I.SHAPED:
        cells: list[tuple] = [()] * 9
        for r, row in enumerate(pattern):
            for c, ch in enumerate(row):
                if ch != " ":
                    key = keys[ch]
                    cells[r * 3 + c] = tuple(sorted(key)) if isinstance(key, frozenset) else (key,)
        small = len(pattern) <= 2 and max(len(row) for row in pattern) <= 2
        out.append({"result": result, "cells": cells, "station": "hand" if small else "table"})
    for need, result in I.SHAPELESS:
        cells = [()] * 9
        for k, item in enumerate(need):
            cells[(k // 2) * 3 + k % 2 if len(need) <= 4 else k] = (item,)
        out.append({"result": result, "cells": cells, "station": "hand" if len(need) <= 4 else "table"})
    by_result: dict[int, list[int]] = {}
    for src, dst in I.SMELT.items():
        by_result.setdefault(dst, []).append(src)
    for dst, sources in by_result.items():
        cells = [()] * 9
        cells[4] = tuple(sorted(sources))
        out.append({"result": (dst, 1), "cells": cells, "station": "furnace"})
    out.sort(key=lambda e: (e["station"] == "furnace", e["result"][0] >= 256, e["result"][0]))
    return out


RECIPES = _recipes()

# key, title, what to do, icon, and (for the ones earned by owning something) the items that count
ADVANCEMENTS = [
    ("wood", "Timber", "Punch a tree until a log drops", B.OAK_LOG, tuple(I.LOGS)),
    ("table", "Workbench", "Craft a crafting table from planks", B.CRAFTING_TABLE, (B.CRAFTING_TABLE,)),
    ("pick", "First Tool", "Craft a wooden pickaxe", B.tool_id(0, 0), (B.tool_id(0, 0),)),
    ("stone", "Stone Tools", "Craft a stone pickaxe", B.tool_id(1, 0), (B.tool_id(1, 0),)),
    ("torch", "Light the Dark", "Craft torches from coal and a stick", B.TORCH, (B.TORCH,)),
    ("dawn", "Second Dawn", "Live through your first night", B.BED, ()),
    ("furnace", "Furnace Lit", "Craft a furnace from cobblestone", B.FURNACE, (B.FURNACE,)),
    ("iron", "Iron Worker", "Smelt iron ore into an ingot", B.IRON_INGOT, (B.IRON_INGOT,)),
    ("till", "Green Thumb", "Till the ground with a hoe", B.HOE0 + 1, ()),
    ("harvest", "Harvest Home", "Harvest ripe wheat", B.WHEAT_ITEM, ()),
    ("bread", "Fresh Bread", "Craft bread from three wheat", B.BREAD, (B.BREAD,)),
    ("sleep", "Good Night", "Sleep through a night in a bed", B.BED, ()),
    ("bucket", "Water Carrier", "Craft a bucket from iron", B.BUCKET, (B.BUCKET, B.WATER_BUCKET)),
    ("armor", "Well Armoured", "Wear a piece of armour", B.ARMOR0 + 1, ()),
    ("hunter", "Monster Hunter", "Defeat a monster", B.tool_id(2, 3), ()),
    ("trade", "Fair Trade", "Trade with a settler in a hamlet", B.GEM, ()),
    ("diamond", "Blue Treasure", "Find a diamond deep underground", B.DIAMOND, (B.DIAMOND,)),
    ("gate", "Gatewright", "Craft a Deep Gate: stone bricks round a diamond", B.DEEP_GATE, (B.DEEP_GATE,)),
    ("deep", "Into the Deep", "Step through a Deep Gate", B.DEEPSTONE, ()),
    ("star", "Star Finder", "Take starstone from a forge keep", B.STARSTONE, (B.STARSTONE,)),
    ("skygate", "Skywright", "Craft a Sky Gate from starstone", B.SKY_GATE, (B.SKY_GATE,)),
    ("isles", "Above the Clouds", "Step through a Sky Gate", B.MOONMOSS, ()),
    ("wyrm", "Wyrm Slayer", "Beat the Sky Wyrm on the middle isle", B.DEEP_CRYSTAL, ()),
]
_ADV = {a[0]: a for a in ADVANCEMENTS}


class BookMixin:
    """See the module docstring."""

    book_page = 0
    book_pick = 0
    adv_scroll = 0

    def open_book(self, name: str) -> None:
        """Show the recipe book ("recipes") or the advancements ("advancements")."""
        self.overlay = name
        self.mouse_buttons = 0
        self.mining = None
        self.set_mouse_captured(False)
        self.sfx.play("click", 0.4)

    # --- advancements -------------------------------------------------------------------------------------------

    def advancements_done(self) -> set:
        return set(self.record.get("adv") or [])

    def award(self, key: str) -> None:
        """Tick an advancement off (once): a line on screen, a sound, a little experience."""
        if key not in _ADV:
            return
        done = self.record.setdefault("adv", [])
        if not isinstance(done, list) or key in done:
            return
        done.append(key)
        self.say(f"Advancement: {_ADV[key][1]}", 5.0)
        self.sfx.play("levelup", 0.6)
        if not self.creative:
            self.gain_xp(3)

    def check_advancements(self) -> None:
        """The advancements earned by what the player carries, wears or has lived through (once a second)."""
        done = self.advancements_done()
        if len(done) >= len(ADVANCEMENTS):
            return
        have = {s[0] for s in self.inv.slots if s is not None}
        for key, _title, _text, _icon, items in ADVANCEMENTS:
            if key not in done and items and not have.isdisjoint(items):
                self.award(key)
        if "armor" not in done and any(s is not None for s in self.inv.armor):
            self.award("armor")
        if "dawn" not in done and self.day_number >= 2 and not self.creative:
            self.award("dawn")

    # --- recipes ------------------------------------------------------------------------------------------------

    def can_make(self, entry: dict) -> bool:
        """Does the inventory hold everything a recipe needs?"""
        need: dict[tuple, int] = {}
        for options in entry["cells"]:
            if options:
                need[options] = need.get(options, 0) + 1
        have = {}
        for s in self.inv.slots:
            if s is not None:
                have[s[0]] = have.get(s[0], 0) + s[1]
        return all(sum(have.get(i, 0) for i in options) >= n for options, n in need.items())

    def _book_origin(self) -> tuple[int, int]:
        gw, gh = self.gui_size()
        return (gw - PANEL[0]) // 2, (gh - PANEL[1]) // 2

    def book_click(self, bit: int, gx: float, gy: float) -> None:
        ox, oy = self._book_origin()
        if not (ox <= gx < ox + PANEL[0] and oy <= gy < oy + PANEL[1]):
            self.close_container()
            return
        if self.overlay != "recipes" or bit != 1:
            return
        c, r = int((gx - ox - 8) // 18), int((gy - oy - 20) // 18)
        if 0 <= c < COLS and 0 <= r < ROWS:
            k = self.book_page * COLS * ROWS + r * COLS + c
            if k < len(RECIPES):
                self.book_pick = k
                self.sfx.play("click", 0.4)

    def book_scroll(self, steps: int) -> None:
        if self.overlay == "recipes":
            pages = math.ceil(len(RECIPES) / (COLS * ROWS))
            self.book_page = min(max(self.book_page - steps, 0), pages - 1)
        else:
            self.adv_scroll = min(max(self.adv_scroll - steps, 0), max(len(ADVANCEMENTS) - ADV_ROWS, 0))

    # --- painting -----------------------------------------------------------------------------------------------

    def paint_book(self, p: QPainter, gw: int, gh: int) -> None:
        p.fillRect(QRect(0, 0, gw + 1, gh + 1), QColor(16, 16, 16, 176))
        ox, oy = self._book_origin()
        p.drawImage(QRectF(ox, oy, *PANEL), G.panel_image(*PANEL))
        if self.overlay == "recipes":
            self._paint_recipes(p, ox, oy, gw)
        else:
            self._paint_advancements(p, ox, oy)

    def _paint_recipes(self, p: QPainter, ox: int, oy: int, gw: int) -> None:
        dark, gs = (64, 64, 64), self.gs()
        per = COLS * ROWS
        pages = math.ceil(len(RECIPES) / per)
        self.book_page = min(self.book_page, pages - 1)
        G.draw_text(p, ox + 8, oy + 7, "Recipe Book", dark, False)
        G.draw_text(p, ox + 8 + COLS * 18, oy + 7, f"{self.book_page + 1}/{pages}", dark, False, align="right")
        mx, my = self.mouse_gui
        hover = None
        for k in range(per):
            i = self.book_page * per + k
            if i >= len(RECIPES):
                break
            entry = RECIPES[i]
            x, y = ox + 8 + (k % COLS) * 18, oy + 20 + (k // COLS) * 18
            G.draw_slot(p, x, y)
            if self.can_make(entry):
                p.fillRect(QRect(x + 1, y + 1, 16, 16), QColor(70, 190, 80, 110))
            if i == self.book_pick:
                p.fillRect(QRect(x + 1, y + 1, 16, 16), QColor(255, 255, 255, 110))
            item, count = entry["result"]
            G.draw_item(p, x + 1, y + 1, item, count, 0, gs)
            if x <= mx < x + 18 and y <= my < y + 18:
                hover = B.NAMES.get(item, "?")
        entry = RECIPES[min(self.book_pick, len(RECIPES) - 1)]
        item, count = entry["result"]
        rx = ox + 124
        G.draw_text(p, rx + 60, oy + 20, B.NAMES.get(item, "?")[:22], dark, False, align="center")
        gx0, gy0 = rx + 4, oy + 38
        turn = int(self.time * 0.8)
        for k, options in enumerate(entry["cells"]):
            x, y = gx0 + (k % 3) * 18, gy0 + (k // 3) * 18
            G.draw_slot(p, x, y)
            if options:
                shown = options[turn % len(options)]
                G.draw_item(p, x + 1, y + 1, shown, 1, 0, gs)
                if x <= mx < x + 18 and y <= my < y + 18:
                    hover = B.NAMES.get(shown, "?") + (" (or alike)" if len(options) > 1 else "")
        self._paint_arrow(p, gx0 + 58, gy0 + 19, 1.0, faint=True)
        G.draw_slot(p, gx0 + 86, gy0 + 14, 26)
        G.draw_item(p, gx0 + 91, gy0 + 19, item, count, 0, gs)
        G.draw_text(p, rx + 60, oy + 100, STATIONS[entry["station"]], dark, False, align="center")
        ready = self.can_make(entry)
        G.draw_text(p, rx + 60, oy + 112, "You have what it takes" if ready else "You lack something",
                    (30, 120, 40) if ready else (150, 60, 50), False, align="center")
        G.draw_text(p, rx + 60, oy + 136, "Green: can be made now", dark, False, align="center")
        G.draw_text(p, rx + 60, oy + 148, "Wheel: turn the page", dark, False, align="center")
        if hover:
            self._paint_tooltip(p, mx + 12, my - 12, hover, gw)

    def _paint_advancements(self, p: QPainter, ox: int, oy: int) -> None:
        dark, gs = (64, 64, 64), self.gs()
        done = self.advancements_done()
        G.draw_text(p, ox + 8, oy + 7, "Advancements", dark, False)
        G.draw_text(p, ox + PANEL[0] - 8, oy + 7, f"{len(done)} of {len(ADVANCEMENTS)}", dark, False, align="right")
        self.adv_scroll = min(self.adv_scroll, max(len(ADVANCEMENTS) - ADV_ROWS, 0))
        for row in range(ADV_ROWS):
            i = self.adv_scroll + row
            if i >= len(ADVANCEMENTS):
                break
            key, title, text, icon, _items = ADVANCEMENTS[i]
            y = oy + 20 + row * 20
            got = key in done
            p.fillRect(QRect(ox + 7, y, PANEL[0] - 14, 19), QColor(70, 150, 80, 70) if got else QColor(0, 0, 0, 28))
            G.draw_slot(p, ox + 8, y)
            G.draw_item(p, ox + 9, y + 1, icon, 1, 0, gs)
            G.draw_text(p, ox + 30, y + 1, title, (20, 96, 30) if got else dark, False)
            G.draw_text(p, ox + 30, y + 10, text, (96, 96, 96), False)
            G.draw_text(p, ox + PANEL[0] - 10, y + 5, "done" if got else "", (20, 96, 30), False, align="right")
        top = oy + 20 + int((ADV_ROWS * 20 - 24) * self.adv_scroll / max(len(ADVANCEMENTS) - ADV_ROWS, 1))
        p.fillRect(QRect(ox + PANEL[0] - 6, top, 2, 24), QColor(90, 90, 90))
