"""BlockCraft's screens: the studio splash, the title screen over a turning world panorama, world selection and
creation, options, the "Generating terrain" screen, the in-game HUD (hotbar, hearts, hunger, experience, crosshair,
debug overlay, the hand), the inventories (survival with 2x2 crafting, creative palette, crafting table, furnace),
the game menu and the death screen."""

from __future__ import annotations

import logging
import math
import os
import platform
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen, QPolygonF, QTransform

from app.arcade.games.blockcraft import blocks as B
from app.arcade.games.blockcraft import gui as G
from app.arcade.games.blockcraft import items as I
from app.arcade.games.blockcraft.render import (WARM, Env, Renderer, block_box, brightness, item_boxes, mob_boxes,
                                                set_gamma)

log = logging.getLogger(__name__)

SPLASHES = ("Also try Tank Design Studio!", "Now with 100% more blocks!", "Engineer approved!",
            "Hard hats required!", "Made of numpy!", "Now without edges!", "Punch trees!", "Coffee break certified!",
            "Seeds of change!", "Rated E for Excavation!", "Don't dig straight down!", "Block by block!",
            "Torches beat darkness!", "Zombies burn at sunrise!", "Hello, break room!", "Square sun, square moon!",
            "Pixel perfect!", "Stack to 64!", "Diamonds below 12!", "Also has lava!", "Moo!", "Oink!",
            "One more block!", "Mind the gap!", "Works offline!", "Contains no creepers!", "Voxel-tastic!")
RES_NAMES = {"auto": "Auto", "full": "Full", "half": "Half"}
AUTO_PIXELS = 2_600_000  # "Auto" draws the 3D view at the window's own pixels up to this many (then a little fewer)
MIN_CHUNKS, MAX_CHUNKS = 3, 20  # the render distance
PANO_CHUNKS = 7  # the land around the title screen's camera
# a box's faces: outward normal and corners (top-left, top-right, bottom-right, bottom-left seen from outside)
_BOX_FACES = {
    "+x": ((1, 0, 0), ((1, 1, 1), (1, 1, -1), (1, -1, -1), (1, -1, 1))),
    "-x": ((-1, 0, 0), ((-1, 1, -1), (-1, 1, 1), (-1, -1, 1), (-1, -1, -1))),
    "+z": ((0, 0, 1), ((-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1))),
    "-z": ((0, 0, -1), ((1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1))),
    "+y": ((0, 1, 0), ((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1))),
    "-y": ((0, -1, 0), ((-1, -1, 1), (1, -1, 1), (1, -1, -1), (-1, -1, -1))),
}


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Turn about y, then x, then z (degrees)."""
    a, b, c = (math.radians(v) for v in (yaw, pitch, roll))
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return ry @ rx @ rz


@dataclass
class Widget:
    kind: str  # button / slider / field / entry
    id: str
    rect: QRect
    label: str = ""
    enabled: bool = True
    value: float = 0.0


class ScreensMixin:
    """Painting and menu input (mixed into the game widget)."""

    # --- geometry -----------------------------------------------------------------------------------------------

    def gs(self) -> int:
        return G.gui_scale(self.width(), self.height(), int(self.opts.get("gui", 0)))

    def gui_size(self) -> tuple[int, int]:
        s = self.gs()
        return self.width() // s, self.height() // s

    def to_gui(self, pos: QPointF) -> tuple[float, float]:
        s = self.gs()
        return pos.x() / s, pos.y() / s

    def widgets(self) -> list[Widget]:
        gw, gh = self.gui_size()
        cx = gw // 2
        out: list[Widget] = []

        def button(id_: str, x: int, y: int, w: int, label: str, enabled: bool = True) -> None:
            out.append(Widget("button", id_, QRect(x, y, w, 20), label, enabled))

        scr = self.overlay if self.screen == "game" and self.overlay in ("menu", "dead", "options") else self.screen
        if scr == "title":
            y = gh // 4 + 48
            button("single", cx - 100, y, 200, "Singleplayer")
            button("multi", cx - 100, y + 24, 200, "Multiplayer", False)
            button("realms", cx - 100, y + 48, 200, "Arcade Realms", False)
            button("options", cx - 100, y + 84, 98, "Options...")
            button("exit", cx + 2, y + 84, 98, "Back to Arcade")
        elif scr == "select":
            top, bottom = 32, gh - 64
            for i, _rec in enumerate(self.worlds):
                y = top + 4 + i * 36 - int(self.list_scroll)
                if y + 32 < top or y > bottom:
                    continue
                out.append(Widget("entry", f"world:{i}", QRect(cx - 110, y, 220, 32)))
            sel = 0 <= self.sel_world < len(self.worlds)
            button("play", cx - 154, gh - 52, 150, "Play Selected World", sel)
            button("create", cx + 4, gh - 52, 150, "Create New World")
            button("delete", cx - 154, gh - 28, 150, "Delete", sel)
            button("cancel", cx + 4, gh - 28, 150, "Cancel")
        elif scr == "create":
            out.append(Widget("field", "name", QRect(cx - 100, 60, 200, 20), self.new_name))
            mode = "Creative" if self.new_mode == "creative" else "Survival"
            button("mode", cx - 75, 100, 150, f"Game Mode: {mode}")
            out.append(Widget("field", "seed", QRect(cx - 100, 160, 200, 20), self.new_seed))
            button("do_create", cx - 155, gh - 28, 150, "Create New World", bool(self.new_name.strip()))
            button("cancel_create", cx + 5, gh - 28, 150, "Cancel")
        elif scr == "delete":
            button("confirm_delete", cx - 155, gh // 6 + 96, 150, "Delete")
            button("cancel_delete", cx + 5, gh // 6 + 96, 150, "Cancel")
        elif scr == "options":
            y0 = gh // 6
            o = self.opts
            fov = int(o.get("fov", 70))
            out.append(Widget("slider", "fov", QRect(cx - 155, y0 - 12, 150, 20),
                              "FOV: Normal" if fov == 70 else f"FOV: {fov}", value=(fov - 30) / 80))
            ch = int(o.get("chunks", 10))
            out.append(Widget("slider", "dist", QRect(cx + 5, y0 - 12, 150, 20), f"Render Distance: {ch} chunks",
                              value=(ch - MIN_CHUNKS) / (MAX_CHUNKS - MIN_CHUNKS)))
            button("graphics", cx - 155, y0 + 12, 150, f"Graphics: {'Fancy' if o.get('fancy', True) else 'Fast'}")
            button("smooth", cx + 5, y0 + 12, 150, f"Smooth Lighting: {'ON' if o.get('smooth', True) else 'OFF'}")
            button("res", cx - 155, y0 + 36, 150, f"Resolution: {RES_NAMES.get(o.get('res', 'auto'), 'Auto')}")
            button("aa", cx + 5, y0 + 36, 150, f"Smooth Edges: {'ON' if o.get('aa', True) else 'OFF'}")
            button("bob", cx - 155, y0 + 60, 150, f"View Bobbing: {'ON' if o.get('bob', True) else 'OFF'}")
            button("music", cx + 5, y0 + 60, 150, f"Music: {'ON' if o.get('music', True) else 'OFF'}")
            gui = int(o.get("gui", 0))
            button("gui", cx - 155, y0 + 84, 150, f"GUI Scale: {'Auto' if not gui else gui}")
            parts = "All" if o.get("particles", True) else "Minimal"
            button("particles", cx + 5, y0 + 84, 150, f"Particles: {parts}")
            gamma = float(o.get("gamma", 0.5))
            glabel = "Moody" if gamma <= 0.01 else ("Bright" if gamma >= 0.99 else f"+{int(gamma * 100)}%")
            out.append(Widget("slider", "gamma", QRect(cx - 155, y0 + 108, 150, 20), f"Brightness: {glabel}",
                              value=gamma))
            button("shaders", cx + 5, y0 + 108, 150, f"Shaders: {'ON' if o.get('shaders', False) else 'OFF'}")
            button("controls", cx - 100, y0 + 136, 200, "Controls...")
            button("done", cx - 100, y0 + 168, 200, "Done")
        elif scr == "menu":
            y = gh // 4 + 8
            button("back", cx - 100, y + 16, 200, "Back to Game")
            button("adv", cx - 100, y + 40, 98, "Advancements")
            button("book", cx + 2, y + 40, 98, "Recipe Book")
            button("controls", cx - 100, y + 64, 98, "Controls...")
            button("options", cx + 2, y + 64, 98, "Options...")
            if getattr(self, "dim", "over") != "over":  # always a way home from the other realms
                button("home", cx - 100, y + 88, 200, "Return to the Overworld")
                y += 24
            button("quit_title", cx - 100, y + 96, 200, "Save and Quit to Title")
            button("exit", cx - 100, y + 120, 200, "Back to Arcade")
        elif scr == "dead":
            button("respawn", cx - 100, gh // 4 + 72, 200, "Respawn")
            button("title", cx - 100, gh // 4 + 96, 200, "Title Screen")
        return out

    def widget_at(self, gx: float, gy: float) -> Widget | None:
        for w in self.widgets():
            if w.rect.contains(int(gx), int(gy)):
                return w
        return None

    # --- painting -----------------------------------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        if self.screen == "splash":
            self._paint_splash(p)
            return
        if self.screen == "game" and self.player is not None:
            self._paint_world(p)
        elif self.screen == "title":
            self._paint_panorama(p)
        else:
            G.draw_dirt_background(p, w, h, self.gs())
        s = self.gs()
        p.save()
        p.scale(s, s)
        gw, gh = self.gui_size()
        if self.screen == "game" and self.player is not None:
            if not self.hide_hud and self.overlay not in ("dead",):
                self._paint_hud(p, gw, gh)
            if self.overlay in ("inventory", "crafting", "furnace"):
                self._paint_container(p, gw, gh)
            elif self.overlay in ("recipes", "advancements"):
                self.paint_book(p, gw, gh)
            elif self.overlay == "menu":
                p.fillRect(QRect(0, 0, gw + 1, gh + 1), QColor(16, 16, 16, 176))
                G.draw_text(p, gw / 2, 40, "Game Menu", align="center")
            elif self.overlay == "options":
                p.fillRect(QRect(0, 0, gw + 1, gh + 1), QColor(16, 16, 16, 176))
                G.draw_text(p, gw / 2, 15, "Options", align="center")
            if self.sleep is not None:  # eyes closing
                p.fillRect(QRect(0, 0, gw + 1, gh + 1), QColor(0, 0, 8, int(min(self.sleep / 1.3, 1.0) * 238)))
            if self.overlay == "dead":
                g = QLinearGradient(0, 0, 0, gh)
                g.setColorAt(0, QColor(80, 0, 0, 96))
                g.setColorAt(1, QColor(128, 48, 48, 160))
                p.fillRect(QRect(0, 0, gw + 1, gh + 1), g)
                G.draw_text(p, gw / 2, 50, "You died!", scale=2, align="center")
                sc = f"Score: {self.score}"
                x0 = gw / 2 - G.text_width(sc) / 2
                G.draw_text(p, x0, 90, "Score: ")
                G.draw_text(p, x0 + G.text_width("Score: "), 90, str(self.score), (255, 255, 85))
        elif self.screen == "title":
            self._paint_title(p, gw, gh)
        elif self.screen == "select":
            self._paint_select(p, gw, gh)
        elif self.screen == "create":
            G.draw_text(p, gw / 2, 20, "Create New World", align="center")
            G.draw_text(p, gw / 2 - 100, 47, "World Name", G.GREY)
            G.draw_text(p, gw / 2 - 100, 85, f"Will be saved as: {self.new_name.strip() or 'World'}", G.GREY)
            desc = ("Unlimited resources, free flying and", "destroy blocks instantly") if self.new_mode == \
                "creative" else ("Search for resources, crafting, gain", "levels, health and hunger")
            G.draw_text(p, gw / 2 - 75, 124, desc[0], G.GREY)
            G.draw_text(p, gw / 2 - 75, 134, desc[1], G.GREY)
            G.draw_text(p, gw / 2 - 100, 147, "Seed for the World Generator", G.GREY)
            G.draw_text(p, gw / 2 - 100, 184, "Leave blank for a random seed", G.GREY)
            G.draw_text(p, gw / 2, 208, "The world has no edge: it grows as you explore", G.GREY, align="center")
        elif self.screen == "delete":
            G.draw_text(p, gw / 2, gh // 6 + 36, "Are you sure you want to delete this world?", align="center")
            name = self.worlds[self.sel_world].get("name", "World") if 0 <= self.sel_world < len(self.worlds) else ""
            G.draw_text(p, gw / 2, gh // 6 + 56, f"'{name}' will be lost forever! (A long time!)", G.GREY,
                        align="center")
        elif self.screen == "options":
            G.draw_text(p, gw / 2, 15, "Options", align="center")
        elif self.screen == "generating":
            G.draw_text(p, gw / 2, gh / 2 - 30, "Loading world", align="center")
            pct = int(self.gen_progress * 100)
            G.draw_text(p, gw / 2, gh / 2 - 12, f"{self.gen_label or 'Building terrain'}: {pct}%", align="center")
            bar = QRect(int(gw / 2 - 50), int(gh / 2 + 4), 100, 2)
            p.fillRect(bar, QColor(128, 128, 128))
            p.fillRect(QRect(bar.x(), bar.y(), int(100 * self.gen_progress), 2), QColor(128, 255, 128))
        self._paint_widgets(p)
        if self.screen == "title":
            G.draw_text(p, 2, gh - 10, "BlockCraft 2.0 (Arcade)")
            if B.PACK_CREDIT:
                G.draw_text(p, 2, gh - 20, B.PACK_CREDIT, G.GREY)
            G.draw_text(p, gw - 2, gh - 10, "Break Room Arcade Edition. Not affiliated with Mojang.",
                        align="right")
        p.restore()

    def _paint_widgets(self, p: QPainter) -> None:
        mx, my = self.mouse_gui
        for wdg in self.widgets():
            hover = wdg.rect.contains(int(mx), int(my)) and wdg.enabled
            if wdg.kind == "button":
                G.draw_button(p, wdg.rect, wdg.label, hover, wdg.enabled)
            elif wdg.kind == "slider":
                G.draw_slider(p, wdg.rect, wdg.label, max(0.0, min(1.0, wdg.value)), hover)
            elif wdg.kind == "field":
                G.draw_field(p, wdg.rect, wdg.label, self.focus_field == wdg.id, int(time.time() * 3) % 2 == 0)

    # --- splash, title, panorama, world list --------------------------------------------------------------------

    def _paint_splash(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(30, 34, 48))
        s = self.gs()
        p.save()
        p.scale(s, s)
        gw, gh = self.gui_size()
        logo = self.studio_logo()
        lw = min(gw * 0.3, 72)
        lh = lw * logo.height() / logo.width()
        G.blit_scaled(p, logo, QRectF((gw - lw) / 2, gh / 2 - lh - 10, lw, lh))
        G.draw_text(p, gw / 2, gh / 2 + 2, "BREAK ROOM STUDIOS", (235, 235, 235), False, scale=2, align="center")
        bw = min(gw * 0.6, 200)
        r = QRectF((gw - bw) / 2, gh / 2 + 26, bw, 6)
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        p.fillRect(QRectF(r.x() + 2, r.y() + 2, (r.width() - 3) * self.load_progress, 3), QColor(255, 255, 255))
        p.restore()
        del w, h

    def studio_logo(self) -> QImage:
        if getattr(self, "_studio_logo", None) is None:
            self._studio_logo = G.to_qimage(G._iso_block(B.GRASS, 96))
        return self._studio_logo

    def _paint_panorama(self, p: QPainter) -> None:
        """The title screen's background: a slow turn over a world of its own (a plain sky until it is built)."""
        w, h = self.width(), self.height()
        img = None
        tw = self.title_world
        if tw is not None:
            sx, sy, sz = self.title_spawn
            if tw.loaded_fraction(sx, sz, 3) >= 1.0:
                if not self.title_since:
                    self.title_since = time.perf_counter()
                env = self.__dict__.setdefault("_title_env", Env(0.17, 0.0))
                env.time = self.time
                rw, rh = self.render_resolution()
                img = self.renderer.render(tw, (sx, sy + 15.0, sz), self.time * 2.4, 10.0, 0.0, 75.0, rw, rh, env,
                                           PANO_CHUNKS, fancy=True, samples=4 if self.opts.get("aa", True) else 0,
                                           shaders=bool(self.opts.get("shaders", False)))
        fade = min(1.0, (time.perf_counter() - self.title_since) / 0.8) if img is not None else 0.0
        if fade < 1.0:
            g = QLinearGradient(0, 0, 0, h)
            g.setColorAt(0, QColor(120, 167, 255))
            g.setColorAt(1, QColor(192, 216, 255))
            p.fillRect(self.rect(), g)
        if img is not None:
            p.setOpacity(fade)
            self._blit_view(p, img)
            p.setOpacity(1.0)
        g = QLinearGradient(0, 0, 0, h)
        g.setColorAt(0, QColor(255, 255, 255, 90))
        g.setColorAt(0.5, QColor(255, 255, 255, 0))
        g.setColorAt(1, QColor(0, 0, 0, 110))
        p.fillRect(self.rect(), g)
        del w

    def title_logo(self) -> QImage:
        if getattr(self, "_title_logo", None) is None:
            text = "BLOCKCRAFT"
            k = 4  # texture pixels per font pixel
            mask = np.zeros((8, G.text_width(text) + 1), bool)
            x = 0
            for ch in text:
                g = G.GLYPHS[ch]
                mask[:, x:x + g.shape[1]] |= g
                x += g.shape[1] + 1
            mask = mask[:7]
            big = np.repeat(np.repeat(mask, k, 0), k, 1)
            H, W = big.shape
            depth = 6
            img = np.zeros((H + depth + 2, W + depth + 2, 4), np.uint8)
            cob = B.ATLAS[B.TEX_INDEX["cobblestone"]][..., :3]
            tex = np.tile(cob, (H // 16 + 1, W // 16 + 1, 1))[:H, :W].astype(np.float32)
            shade = np.linspace(1.15, 0.8, H)[:, None, None]
            for d in range(depth, 0, -1):  # the extrusion, darker further back
                sub = img[1 + d:1 + d + H, 1 + d:1 + d + W]
                sub[big] = (40 + d * 4, 40 + d * 4, 44 + d * 4, 255)
            face = img[1:1 + H, 1:1 + W]
            col = np.clip(tex * shade, 0, 255).astype(np.uint8)
            face[big, :3] = col[big]
            face[big, 3] = 255
            edge = big & ~(np.roll(big, 1, 0) & np.roll(big, -1, 0) & np.roll(big, 1, 1) & np.roll(big, -1, 1))
            face[edge, :3] = (25, 25, 25)
            self._title_logo = G.to_qimage(img)
        return self._title_logo

    def _paint_title(self, p: QPainter, gw: int, gh: int) -> None:
        logo = self.title_logo()
        lw = min(274, gw - 20)
        lh = lw * logo.height() / logo.width()
        x0, y0 = (gw - lw) / 2, 30
        G.blit_scaled(p, logo, QRectF(x0, y0, lw, lh))
        # the "edition" banner
        ed = "ARCADE EDITION"
        ew = G.text_width(ed) + 8
        p.fillRect(QRectF((gw - ew) / 2, y0 + lh + 2, ew, 11), QColor(80, 80, 80))
        G.draw_text(p, gw / 2, y0 + lh + 4, ed, (235, 235, 235), False, align="center")
        # the splash text
        text = self.splash
        t = time.time()
        s = 1.8 - abs(math.sin((t % 1.0) * 2 * math.pi)) * 0.1
        s = s * 100 / (G.text_width(text) + 32)
        p.save()
        p.translate(x0 + lw * 0.88, y0 + lh * 0.85)
        p.rotate(-20)
        p.scale(s, s)
        G.draw_text(p, 0, -4, text, (255, 255, 0), align="center")
        p.restore()

    def _paint_select(self, p: QPainter, gw: int, gh: int) -> None:
        G.draw_text(p, gw / 2, 16, "Select World", align="center")
        top, bottom = 32, gh - 64
        p.fillRect(QRect(0, top, gw + 1, bottom - top), QColor(0, 0, 0, 140))
        p.fillRect(QRect(0, top, gw + 1, 1), QColor(0, 0, 0, 200))
        p.save()
        p.setClipRect(QRect(0, top, gw + 1, bottom - top))
        mx, my = self.mouse_gui
        if not self.worlds:
            G.draw_text(p, gw / 2, top + 20, "No worlds yet: create one!", G.GREY, align="center")
        for wdg in self.widgets():
            if wdg.kind != "entry":
                continue
            i = int(wdg.id.split(":")[1])
            rec = self.worlds[i]
            r = wdg.rect
            if i == self.sel_world:
                p.fillRect(r, QColor(128, 128, 128))
                p.fillRect(r.adjusted(1, 1, -1, -1), QColor(0, 0, 0))
            elif r.contains(int(mx), int(my)):
                p.fillRect(r, QColor(255, 255, 255, 30))
            icon = QRect(r.x() + 2, r.y() + 2, 28, 28)
            G.blit_scaled(p, G.item_icon(B.GRASS if rec.get("mode") != "creative" else B.DIAMOND_BLOCK, 56),
                          QRectF(icon))
            G.draw_text(p, r.x() + 34, r.y() + 2, str(rec.get("name", "World"))[:28])
            when = time.strftime("%m/%d/%y %I:%M %p", time.localtime(int(rec.get("last", rec.get("created", 0)))))
            G.draw_text(p, r.x() + 34, r.y() + 12, f"{rec.get('name', 'World')[:12]} ({when})", (128, 128, 128))
            mode = "Creative Mode" if rec.get("mode") == "creative" else "Survival Mode"
            days = int(float(rec.get("time", 0)) // 1200) + 1
            G.draw_text(p, r.x() + 34, r.y() + 22, f"{mode}, Day {days}", (128, 128, 128))
        p.restore()

    # --- the world view -----------------------------------------------------------------------------------------

    def render_resolution(self) -> tuple[int, int]:
        """The pixels the 3D view is drawn at: the widget's own (sharp at any size), fewer by option or on a very
        large screen."""
        dpr = self.devicePixelRatioF()
        w, h = max(int(self.width() * dpr), 64), max(int(self.height() * dpr), 48)
        mode = self.opts.get("res", "auto")
        scale = 1.0
        if mode == "half":
            scale = 0.5
        elif mode == "auto" and w * h > AUTO_PIXELS:
            scale = math.sqrt(AUTO_PIXELS / (w * h))
        return max(int(w * scale), 64), max(int(h * scale), 48)

    def _blit_view(self, p: QPainter, img: QImage) -> None:
        """The 3D picture on the widget: pixel for pixel when it was drawn at the widget's size."""
        exact = img.width() == int(self.width() * self.devicePixelRatioF())
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, not exact)
        p.drawImage(QRectF(0, 0, self.width(), self.height()), img)
        p.restore()

    def camera(self) -> tuple[tuple, float, float, float]:
        """Eye position, yaw, pitch and roll for this frame (interpolated, with view bobbing and the hurt tilt)."""
        pl = self.player
        a = min(max(self.acc / 0.05, 0.0), 1.0) if self.overlay not in ("menu", "options") else 1.0
        ex, ey, ez = self.eye(a)
        yaw, pitch, roll = pl.yaw, pl.pitch, 0.0
        if self.opts.get("bob", True) and not pl.flying:
            walk = pl.prev_walked + (pl.walked - pl.prev_walked) * a
            bob = pl.prev_bob + (pl.bob - pl.prev_bob) * a
            ph = walk * math.pi
            yr = math.radians(yaw)
            side = math.sin(ph) * bob * 0.05
            ex += -math.cos(yr) * side
            ez += -math.sin(yr) * side
            ey += -abs(math.cos(ph) * bob) * 0.07
            roll += math.radians(math.sin(ph) * bob * 1.5)
            pitch += abs(math.cos(ph - 0.2) * bob) * 1.2
        if self.hurt_anim > 0:
            roll += math.radians(math.sin(self.hurt_anim * math.pi) * 6)
        return (ex, ey, ez), yaw, pitch, roll

    def _paint_world(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        rw, rh = self.render_resolution()
        eye, yaw, pitch, roll = self.camera()
        now = time.perf_counter()
        r = self.renderer
        zoom = 1.1 if self.player.sprinting and not self.player.sneaking else 1.0  # the view widens into a sprint
        r.fov_scale += (zoom - r.fov_scale) * min(1.0, max(now - self._last_paint, 0.0) * 9.0)
        fov = float(self.opts.get("fov", 70)) * r.fov_scale
        env = self.env
        ex, ey, ez = (math.floor(v) for v in eye)
        b_eye = self.world.get(ex, ey, ez)
        env.underwater = bool(B.WATERY[b_eye])
        env.in_lava = b_eye == B.LAVA
        set_gamma(float(self.opts.get("gamma", 0.5)))
        crack = None
        if self.mining is not None and self.mine_progress > 0:
            crack = (*self.mining, int(self.mine_progress * 10))
        a = min(max(self.acc / 0.05, 0.0), 1.0)
        chunks = int(self.opts.get("chunks", 10))
        boxes = self._entity_boxes(a, eye, chunks * 16.0)
        dots = self._particle_dots(eye) if self.opts.get("particles", True) else None
        img = self.renderer.render(self.world, eye, yaw, pitch, roll, fov, rw, rh, env, chunks, boxes, dots, crack,
                                   fancy=bool(self.opts.get("fancy", True)),
                                   samples=4 if self.opts.get("aa", True) else 0,
                                   shaders=bool(self.opts.get("shaders", False)))
        self._blit_view(p, img)
        if self._last_paint:
            self.frame_ms = self.frame_ms * 0.85 + min((now - self._last_paint) * 1000, 250) * 0.15
        self._last_paint = now
        self._fps_frames += 1
        if now - self._fps_t0 >= 1.0:
            self.fps = self._fps_frames / (now - self._fps_t0)
            self._fps_frames = 0
            self._fps_t0 = now
        M = Renderer.basis(yaw, pitch, roll)
        if self.target is not None and self.overlay is None and not self.hide_hud:
            self._paint_outline(p, eye, M, fov, w, h)
        if not self.hide_hud:
            self._paint_hand(p, w, h)
        if self.fire > 0 and not env.underwater:
            self._paint_fire(p, w, h)
        if env.underwater:
            p.fillRect(self.rect(), QColor(20, 40, 120, 50))

    def _entity_boxes(self, a: float, eye: tuple, dist: float) -> np.ndarray | None:
        """The boxes of everything that moves (render.BOX_FLOATS numbers each), for the renderer."""
        origin = Renderer.origin(eye)
        ex, _ey, ez = eye
        out = []
        for m in self.mobs:
            if abs(m.x - ex) < dist and abs(m.z - ez) < dist:
                out.append(mob_boxes(m, a, self._light_rgb(m.x, m.y + m.height * 0.6, m.z), origin))
        for d in self.drops:
            if abs(d.x - ex) < 40 and abs(d.z - ez) < 40:
                out.append(item_boxes(d, a, self._light_rgb(d.x, d.y + 0.2, d.z), origin, self.time))
        for f in self.falling:
            out.append(block_box(f, a, self._light_rgb(f.x, f.y + 0.5, f.z), origin))
        return np.concatenate(out) if out else None

    def _particle_dots(self, eye: tuple) -> np.ndarray | None:
        """The particles as (x, y, z, size, r, g, b, 1) rows, lit by where they are."""
        items = self.particles.items
        if not items:
            return None
        ox, _oy, oz = Renderer.origin(eye)
        light_at, day = self.world.light_at, float(self.env.daylight)
        table = [brightness(i) for i in range(16)]
        out = np.empty((len(items), 8), np.float32)
        for i, q in enumerate(items):
            sky, block = light_at(math.floor(q[0]), math.floor(q[1]), math.floor(q[2]))
            k = max(table[sky] * day, table[block]) / 255.0
            c = q[7]
            out[i] = (q[0] - ox, q[1], q[2] - oz, q[8], c[0] * k, c[1] * k, c[2] * k, 1.0)
        return out

    def _light_rgb(self, x: float, y: float, z: float) -> np.ndarray:
        sky, block = self.world.light_at(math.floor(x), math.floor(y), math.floor(z))
        s = max(brightness(sky) * self.env.daylight, self.env.ambient)
        bl = brightness(block)
        return np.maximum(np.array([s, s, s]), bl * WARM).astype(np.float64)

    def _paint_outline(self, p: QPainter, eye: tuple, M: np.ndarray, fov: float, w: int, h: int) -> None:
        (bx, by, bz), _n, _d = self.target
        e = 0.002
        lo = np.array([bx - e, by - e, bz - e])
        hi = np.array([bx + 1 + e, by + 1 + e, bz + 1 + e])
        b = self.world.get(bx, by, bz)
        if B.KIND[b] == B.K_CROSS:  # plants: a smaller box
            lo = np.array([bx + 0.2, by, bz + 0.2])
            hi = np.array([bx + 0.8, by + (0.6 if b == B.TORCH else 0.8), bz + 0.8])
        tan = math.tan(math.radians(fov) / 2)
        aspect = w / h
        o = np.array(eye)

        def corner(i: int) -> np.ndarray:
            return np.array([hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]])

        def project(q: np.ndarray) -> QPointF:
            return QPointF((q[0] / q[2] / (tan * aspect) + 1) / 2 * w, (1 - q[1] / q[2] / tan) / 2 * h)

        # visible faces (facing the eye) and their edges
        faces = []
        for axis in range(3):
            if o[axis] < lo[axis]:
                faces.append((axis, 0))
            elif o[axis] > hi[axis]:
                faces.append((axis, 1))
        edges = set()
        for i in range(8):
            for axis in range(3):
                j = i | (1 << axis)
                if j == i:
                    continue
                # the edge i-j runs along `axis`; it borders the faces of the two other axes
                others = [ax for ax in range(3) if ax != axis]
                if any((ax, (i >> ax) & 1) in faces for ax in others):
                    edges.add((i, j))
        pen = QPen(QColor(0, 0, 0, 120), max(1.0, self.height() / 500))
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(pen)
        for i, j in edges:
            a = M @ (corner(i) - o)
            c = M @ (corner(j) - o)
            near = 0.05
            if a[2] < near and c[2] < near:
                continue
            if a[2] < near:
                a = a + (c - a) * ((near - a[2]) / (c[2] - a[2]))
            elif c[2] < near:
                c = c + (a - c) * ((near - c[2]) / (a[2] - c[2]))
            p.drawLine(project(a), project(c))
        p.restore()

    def _paint_hand(self, p: QPainter, w: int, h: int) -> None:
        """The held item - a little textured block, a tool / item sprite, or the bare arm - in 3D at the bottom right
        (camera space, the real game's fixed 70 degree hand view), bobbing with the walk and swinging when used."""
        if self.overlay is not None and self.overlay not in ("inventory", "crafting", "furnace"):
            return
        pl = self.player
        held = self.inv.held
        sw = 1.0 - self.swing if self.swing > 0 else 0.0  # 0 -> 1 through a swing
        s1 = math.sin(sw * math.pi)
        s2 = math.sin(math.sqrt(sw) * math.pi)
        walk = pl.walked * math.pi
        bob = pl.bob if self.opts.get("bob", True) else 0.0
        light = self._light_rgb(pl.x, pl.y + 1.6, pl.z)
        lvl = min(15, max(2, int(round(float(light.max()) * 15))))
        # where the hand is: right, down, forward (camera space), and how it turns
        x = 0.56 + math.sin(walk) * bob * 0.035 - s2 * 0.2
        y = -0.52 - abs(math.cos(walk)) * bob * 0.05 - (1 - self.equip) * 0.6
        y += math.sin(math.sqrt(sw) * 2 * math.pi) * 0.08
        z = 0.9 - s1 * 0.12
        yaw, pitch, roll = 0.0, 0.0, 0.0
        if held is None:
            boxes = [((x + 0.02, y - 0.12, z - 0.25), (0.12, 0.12, 0.45), "arm"),
                     ((x + 0.02, y - 0.12, z - 0.25), (0.135, 0.135, 0.14), "sleeve", (0, 0, -0.36))]
            yaw, pitch, roll = -22 - s2 * 20, -22 + s1 * 25, 0
        elif B.flat_item(held[0]):
            boxes = [((x + 0.02, y + 0.06, z), (0.27, 0.27, 0.0), "sprite")]
            yaw, pitch, roll = 62 + s2 * 10, 0 - s1 * 25, -20 + s2 * 25
            if self.eat_time > 0:
                bobble = abs(math.sin(self.eat_time * 14)) * 0.05
                boxes = [((x - 0.38, y + 0.18 - bobble, z - 0.25), (0.3, 0.3, 0.0), "sprite")]
                yaw, pitch, roll = 60, 0, -10
        else:
            boxes = [((x - 0.02, y + 0.14, z + 0.15), (0.145, 0.145, 0.145), "block")]
            yaw, pitch, roll = 45 + s2 * 15, -18 - s1 * 25, 0
        R = _rotation(yaw, pitch, roll)
        tan = math.tan(math.radians(35))
        aspect = w / max(h, 1)

        def project(v: np.ndarray) -> QPointF:
            zc = max(v[2], 0.05)
            return QPointF((v[0] / zc / (tan * aspect) + 1) / 2 * w, (1 - v[1] / zc / tan) / 2 * h)

        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        for centre, half, kind, *offset in boxes:
            c = np.array(centre) + (R @ np.array(offset[0], float) if offset else 0.0)
            hx, hy, hz = half
            faces = []
            for face, (normal, corners) in _BOX_FACES.items():
                if kind == "sprite" and face not in ("+z",):
                    continue
                nrm = R @ np.array(normal, float)
                pts = [c + R @ (np.array(k, float) * (hx, hy, hz)) for k in corners]
                centre_f = sum(pts) / 4
                if kind != "sprite" and float(nrm @ centre_f) >= 0:
                    continue  # facing away
                faces.append((float(np.linalg.norm(centre_f)), face, pts))
            for _, face, pts in sorted(faces, key=lambda f: -f[0]):
                img = self._hand_texture(kind, held[0] if held else 0, face, lvl)
                quad = QPolygonF([project(v) for v in pts])
                src = QPolygonF([QPointF(0, 0), QPointF(img.width(), 0), QPointF(img.width(), img.height()),
                                 QPointF(0, img.height())])
                t = QTransform()
                if QTransform.quadToQuad(src, quad, t):
                    p.setTransform(t)
                    p.drawImage(0, 0, img)
                    p.resetTransform()
        p.restore()

    def _hand_texture(self, kind: str, item: int, face: str, level: int) -> QImage:
        """A texture for the hand view, shaded for its face and darkened to the light level (cached)."""
        key = (kind, item if kind == "block" or kind == "sprite" else 0, face, level)
        cache = self.__dict__.setdefault("_hand_cache", {})
        img = cache.get(key)
        if img is None:
            shade = {"+y": 1.0, "-y": 0.5, "+z": 0.8, "-z": 0.8, "+x": 0.6, "-x": 0.6}[face]
            if kind == "sprite":
                rgba = B.item_sprite(item).astype(np.float32)[:, ::-1]
                shade = 1.0
            elif kind in ("arm", "sleeve"):
                rng = np.random.default_rng(5)
                rgba = np.zeros((16, 16, 4), np.float32)
                colour = (196, 150, 110) if kind == "arm" else (255, 120, 20)
                rgba[..., :3] = np.array(colour, np.float32) + rng.normal(0, 5, (16, 16, 1))
                rgba[..., 3] = 255
                if kind == "sleeve" and face in ("+x", "-x", "+y", "-y"):  # a reflective stripe
                    rgba[:, 6:9, :3] = (235, 235, 210)
            else:
                b = B.FURNACE + 2 if item == B.FURNACE else item
                tex = {"+y": 2, "-y": 3, "+z": 4, "-z": 5, "+x": 0, "-x": 1}[face]
                rgba = B.ATLAS[B.FACE_TEX[b, tex]].astype(np.float32)
            rgba = rgba.copy()
            rgba[..., :3] *= shade * level / 15
            img = G.to_qimage(np.clip(rgba, 0, 255).astype(np.uint8))
            if len(cache) > 200:
                cache.clear()
            cache[key] = img
        return img

    def _paint_fire(self, p: QPainter, w: int, h: int) -> None:
        t = time.time()
        p.save()
        for side in (0, 1):
            for k in range(7):
                x = (w * 0.04 + k * w * 0.05) if side == 0 else (w * 0.96 - k * w * 0.05)
                fh = h * (0.18 + 0.07 * math.sin(t * 9 + k * 1.7 + side))
                c = QColor(255, 140 + 60 * (k % 2), 20, 200)
                p.fillRect(QRectF(x - w * 0.025, h - fh, w * 0.05, fh), c)
                p.fillRect(QRectF(x - w * 0.012, h - fh * 0.7, w * 0.024, fh * 0.7), QColor(255, 230, 120, 220))
        p.restore()

    # --- the HUD ------------------------------------------------------------------------------------------------

    def _paint_hud(self, p: QPainter, gw: int, gh: int) -> None:
        cx = gw // 2
        # crosshair (inverts what is behind it)
        if self.overlay is None and not self.show_debug:
            p.save()
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Difference)
            p.fillRect(QRectF(cx - 4.5, gh / 2 - 0.5, 9, 1), QColor(230, 230, 230))
            p.fillRect(QRectF(cx - 0.5, gh / 2 - 4.5, 1, 4), QColor(230, 230, 230))
            p.fillRect(QRectF(cx - 0.5, gh / 2 + 0.5, 1, 4), QColor(230, 230, 230))
            p.restore()
        elif self.overlay is None:
            self._paint_axes(p, cx, gh / 2)
        x0 = cx - 91
        y0 = gh - 22
        p.drawImage(QRectF(x0, y0, 182, 22), G.hud_sprite("hotbar"))
        p.drawImage(QRectF(x0 - 1 + self.inv.selected * 20, y0 - 1, 24, 24), G.hud_sprite("selector"))
        for i in range(9):
            s = self.inv.slots[i]
            if s is not None:
                G.draw_item(p, x0 + 3 + i * 20, y0 + 3, s[0], s[1], s[2], self.gs())
        if not self.creative:
            self._paint_bars(p, cx, gh)
        if self.held_name_time > 0 and self.inv.held is not None:
            name = B.NAMES.get(self.inv.held[0], "")
            alpha = min(1.0, self.held_name_time)
            y = gh - 59 if not self.creative else gh - 35
            col = tuple(int(255 * alpha) for _ in range(3))
            if alpha > 0.05:
                G.draw_text(p, cx, y, name, col, align="center")
        if self.toast_time > 0 and self.toast:
            k = int(255 * min(1.0, self.toast_time))
            if k > 12:
                G.draw_text(p, cx, gh - 72 if not self.creative else gh - 48, self.toast, (k, k, min(k, 170)),
                            align="center")
        boss = next((m for m in self.mobs if m.kind == "wyrm" and not m.dead), None)
        if boss is not None:  # the Sky Wyrm's strength, across the top
            G.draw_text(p, cx, 6, "Sky Wyrm", (200, 236, 255), align="center")
            p.fillRect(QRect(cx - 91, 16, 182, 5), QColor(0, 0, 0))
            p.fillRect(QRect(cx - 90, 17, 180, 3), QColor(40, 46, 70))
            p.fillRect(QRect(cx - 90, 17, int(180 * max(boss.health, 0) / 160), 3), QColor(110, 214, 240))
        if self.show_debug:
            self._paint_debug(p, gw, gh)

    def _paint_bars(self, p: QPainter, cx: int, gh: int) -> None:
        x0 = cx - 91
        # experience
        lvl, frac = self.level()
        bar = QRect(x0, gh - 29, 182, 5)
        p.fillRect(bar, QColor(0, 0, 0))
        p.fillRect(bar.adjusted(1, 1, -1, -1), QColor(40, 40, 40))
        if frac > 0:
            p.fillRect(QRect(x0 + 1, gh - 28, int(180 * frac), 3), QColor(128, 255, 32))
        if lvl > 0:
            s = str(lvl)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                G.draw_text(p, cx + dx, gh - 36 + dy, s, (0, 0, 0), False, align="center")
            G.draw_text(p, cx, gh - 36, s, (128, 255, 32), False, align="center")
        # hearts
        hp = max(0.0, self.health)
        blink = self.hurt_cool > 0.3
        low = hp <= 4
        for i in range(10):
            x = x0 + i * 8
            y = gh - 39 + (int(np.random.default_rng(int(self.time * 10) + i).integers(0, 2)) if low else 0)
            p.drawImage(QRectF(x, y, 9, 9), G.hud_sprite("heart_empty"))
            if blink and int(self.time * 10) % 2 == 0:
                continue
            if hp >= i * 2 + 2:
                p.drawImage(QRectF(x, y, 9, 9), G.hud_sprite("heart"))
            elif hp >= i * 2 + 1:
                p.drawImage(QRectF(x, y, 9, 9), G.hud_sprite("heart_half"))
        # armour, over the hearts (each icon two points)
        armor = self.inv.armor_points()
        if armor:
            for i in range(10):
                name = "armor" if armor >= i * 2 + 2 else ("armor_half" if armor >= i * 2 + 1 else "armor_empty")
                p.drawImage(QRectF(x0 + i * 8, gh - 49, 9, 9), G.hud_sprite(name))
        # hunger (right to left)
        for i in range(10):
            x = cx + 91 - 9 - i * 8
            y = gh - 39
            p.drawImage(QRectF(x, y, 9, 9), G.hud_sprite("food_empty"))
            if self.food >= i * 2 + 2:
                p.drawImage(QRectF(x, y, 9, 9), G.hud_sprite("food"))
            elif self.food >= i * 2 + 1:
                p.drawImage(QRectF(x, y, 9, 9), G.hud_sprite("food_half"))
        # air
        if self.air < 300:
            full = math.ceil(max(self.air, 0) * 10 / 300)
            for i in range(full):
                p.drawImage(QRectF(cx + 91 - 9 - i * 8, gh - 49, 7, 7), G.hud_sprite("bubble"))

    def _paint_axes(self, p: QPainter, cx: float, cy: float) -> None:
        pl = self.player
        M = Renderer.basis(pl.yaw, pl.pitch)
        p.save()
        for axis, col in ((0, QColor(255, 60, 60)), (1, QColor(60, 255, 60)), (2, QColor(60, 120, 255))):
            v = np.zeros(3)
            v[axis] = 1
            q = M @ v
            p.setPen(QPen(col, 1))
            p.drawLine(QPointF(cx, cy), QPointF(cx + q[0] * 10, cy - q[1] * 10))
        p.restore()

    def _paint_debug(self, p: QPainter, gw: int, gh: int) -> None:
        pl = self.player
        rw, rh = self.render_resolution()
        facing = ("south", "Towards positive Z"), ("west", "Towards negative X"), ("north", "Towards negative Z"), \
            ("east", "Towards positive X")
        f = facing[int(((pl.yaw % 360) + 45) // 90) % 4]
        bx, by, bz = math.floor(pl.x), math.floor(pl.y), math.floor(pl.z)
        sky, block = self.world.light_at(bx, by, bz)
        yaw = (pl.yaw + 180) % 360 - 180
        hours = int((self.tod * 24 + 6) % 24)
        st = self.renderer.stats
        left = ["BlockCraft 2.0 (arcade/OpenGL)",
                f"{int(self.fps)} fps  frame {st['ms']:.1f} ms  T: {rw}x{rh}",
                f"C: {st['drawn']}/{st['chunks']} chunks  {st['faces']} faces  D: {self.opts.get('chunks', 10)}",
                f"E: {len(self.mobs)} mobs, {len(self.drops)} items",
                "",
                f"XYZ: {pl.x:.3f} / {pl.y:.5f} / {pl.z:.3f}",
                f"Block: {bx} {by} {bz}",
                f"Facing: {f[0]} ({f[1]}) ({yaw:.1f} / {pl.pitch:.1f})",
                f"Realm: {self.dim}  Biome: {self.world.biome_name(pl.x, pl.z)}",
                f"Light: {max(int(sky * self.env.daylight), block)} ({sky} sky, {block} block)",
                f"Day {self.day_number}, {hours:02d}:00  ({'flying' if pl.flying else self.mode})"]
        right = [f"Python {platform.python_version()} 64bit", f"CPU: {os.cpu_count()}x threads",
                 f"Display: {self.width()}x{self.height()}", self.renderer.gpu[:34],
                 f"Helpers: {self.pool.count} {'processes' if self.pool.processes else 'threads'}"
                 if self.pool is not None else "", f"Seed: {self.world.seed}"]
        if self.target is not None:
            (tx, ty, tz), _, _ = self.target
            right += ["", f"Targeted Block: {tx}, {ty}, {tz}",
                      "blockcraft:" + B.NAMES.get(B.base_block(self.world.get(tx, ty, tz)), "?").lower().replace(" ",
                                                                                                             "_")]
        for i, line in enumerate(left):
            if line:
                p.fillRect(QRectF(1, 2 + i * 9 - 1, G.text_width(line) + 2, 9), QColor(80, 80, 80, 144))
                G.draw_text(p, 2, 2 + i * 9, line, (224, 224, 224), False)
        for i, line in enumerate(right):
            if line:
                wdt = G.text_width(line)
                p.fillRect(QRectF(gw - wdt - 3, 2 + i * 9 - 1, wdt + 2, 9), QColor(80, 80, 80, 144))
                G.draw_text(p, gw - 2, 2 + i * 9, line, (224, 224, 224), False, align="right")

    # --- inventories --------------------------------------------------------------------------------------------

    def panel_size(self) -> tuple[int, int]:
        return (195, 136) if self.overlay == "inventory" and self.creative else (176, 166)

    def panel_origin(self) -> tuple[int, int]:
        gw, gh = self.gui_size()
        pw, ph = self.panel_size()
        return (gw - pw) // 2, (gh - ph) // 2

    def slot_layout(self) -> list[tuple[tuple, int, int]]:
        """(slot reference, x, y) in GUI px for the open container."""
        ox, oy = self.panel_origin()
        out = []
        if self.overlay == "inventory" and self.creative:
            first = int(self.creative_scroll) * 9
            for k in range(45):
                if first + k < len(B.CREATIVE_ITEMS):
                    out.append((("pal", first + k), ox + 9 + (k % 9) * 18, oy + 18 + (k // 9) * 18))
            for i in range(9):
                out.append((("inv", i), ox + 9 + i * 18, oy + 112))
            return out
        for i in range(9, 36):
            out.append((("inv", i), ox + 8 + (i - 9) % 9 * 18, oy + 84 + (i - 9) // 9 * 18))
        for i in range(9):
            out.append((("inv", i), ox + 8 + i * 18, oy + 142))
        if self.overlay == "inventory":
            for k in range(4):
                out.append((("arm", k), ox + 8, oy + 8 + k * 18))
                out.append((("grid", k), ox + 98 + (k % 2) * 18, oy + 18 + (k // 2) * 18))
            out.append((("out",), ox + 154, oy + 28))
        elif self.overlay == "crafting":
            for k in range(9):
                out.append((("grid", k), ox + 30 + (k % 3) * 18, oy + 17 + (k // 3) * 18))
            out.append((("out",), ox + 124, oy + 35))
        elif self.overlay == "furnace":
            out += [(("fur", 0), ox + 56, oy + 17), (("fur", 1), ox + 56, oy + 53), (("fur", 2), ox + 116, oy + 35)]
        return out

    def slot_at(self, gx: float, gy: float) -> tuple | None:
        for ref, x, y in self.slot_layout():
            if x - 1 <= gx < x + 17 and y - 1 <= gy < y + 17:
                return ref
        return None

    def _paint_container(self, p: QPainter, gw: int, gh: int) -> None:
        p.fillRect(QRect(0, 0, gw + 1, gh + 1), QColor(16, 16, 16, 176))
        ox, oy = self.panel_origin()
        pw, ph = self.panel_size()
        p.drawImage(QRectF(ox, oy, pw, ph), G.panel_image(pw, ph))
        dark = (64, 64, 64)
        creative = self.overlay == "inventory" and self.creative
        if creative:
            G.draw_text(p, ox + 8, oy + 6, "Creative Inventory", dark, False)
            sb = QRect(ox + 175, oy + 18, 12, 88)
            p.fillRect(sb, QColor(139, 139, 139))
            p.fillRect(QRect(sb.x(), sb.y(), 11, 1), QColor(55, 55, 55))
            rows = max(1, math.ceil(len(B.CREATIVE_ITEMS) / 9) - 5)
            ky = sb.y() + int((sb.height() - 15) * min(self.creative_scroll / rows, 1.0))
            p.drawImage(QRectF(sb.x(), ky, 12, 15), G.button_image(12, 15, "normal"))
        elif self.overlay == "inventory":
            G.draw_text(p, ox + 86, oy + 6, "Crafting", dark, False)
            box = QRect(ox + 26, oy + 8, 49, 70)
            p.fillRect(box, QColor(0, 0, 0))
            self._paint_player_figure(p, box)
            G.draw_slot(p, ox + 76, oy + 61)
            self._paint_arrow(p, ox + 134, oy + 28, 1.0, faint=True)
        elif self.overlay == "crafting":
            G.draw_text(p, ox + 28, oy + 6, "Crafting", dark, False)
            self._paint_arrow(p, ox + 90, oy + 35, 1.0, faint=True)
        elif self.overlay == "furnace":
            G.draw_text(p, ox + pw / 2, oy + 6, "Furnace", dark, False, align="center")
            f = self.furnaces.get(getattr(self, "furnace_pos", None))
            burn = f.burn / f.burn_total if f is not None and f.burn_total > 0 else 0.0
            prog = f.progress / I.SMELT_TIME if f is not None else 0.0
            self._paint_flame(p, ox + 57, oy + 37, burn)
            self._paint_arrow(p, ox + 79, oy + 34, prog)
        if not creative and self.overlay != "inventory":
            G.draw_text(p, ox + 8, oy + 72, "Inventory", dark, False)
        for ref, x, y in self.slot_layout():
            big = ref == ("out",)
            if big:
                G.draw_slot(p, x - 5, y - 5, 26)
            else:
                G.draw_slot(p, x - 1, y - 1)
            s = self.get_slot(ref)
            if s is not None:
                G.draw_item(p, x, y, s[0], s[1], s[2], self.gs())
        mx, my = self.mouse_gui
        hover = self.slot_at(mx, my)
        if hover is not None:
            for ref, x, y in self.slot_layout():
                if ref == hover:
                    p.fillRect(QRect(x, y, 16, 16), QColor(255, 255, 255, 128))
        if self.cursor is not None:
            G.draw_item(p, mx - 8, my - 8, self.cursor[0], self.cursor[1], self.cursor[2], self.gs())
        elif hover is not None:
            s = self.get_slot(hover)
            if s is not None:
                self._paint_tooltip(p, mx + 12, my - 12, B.NAMES.get(s[0], "?"), gw)

    def _paint_tooltip(self, p: QPainter, x: float, y: float, text: str, gw: int) -> None:
        w = G.text_width(text)
        x = min(x, gw - w - 6)
        r = QRectF(x - 3, y - 3, w + 6, 14)
        p.fillRect(r, QColor(16, 0, 16, 240))
        p.setPen(QPen(QColor(80, 0, 255, 160), 1))
        p.drawRect(r.adjusted(1, 1, -1, -1))
        G.draw_text(p, x, y + 1, text)

    def _paint_arrow(self, p: QPainter, x: int, y: int, frac: float, faint: bool = False) -> None:
        shape = [(0, 6), (15, 6), (15, 1), (22, 8), (15, 15), (15, 10), (0, 10)]
        poly = QPolygonF([QPointF(x + a, y + b) for a, b in shape])
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(139, 139, 139))
        p.drawPolygon(poly)
        if frac > 0 and not faint:
            p.setClipRect(QRectF(x, y, 23 * frac, 17))
            p.setBrush(QColor(255, 255, 255))
            p.drawPolygon(poly)
        p.restore()

    def _paint_flame(self, p: QPainter, x: int, y: int, frac: float) -> None:
        p.fillRect(QRect(x + 3, y + 2, 8, 12), QColor(139, 139, 139))
        p.fillRect(QRect(x + 1, y + 6, 12, 8), QColor(139, 139, 139))
        if frac > 0:
            hh = int(13 * frac)
            p.save()
            p.setClipRect(QRect(x, y + 14 - hh, 14, hh))
            p.fillRect(QRect(x + 3, y + 2, 8, 12), QColor(255, 140, 20))
            p.fillRect(QRect(x + 1, y + 6, 12, 8), QColor(255, 100, 10))
            p.fillRect(QRect(x + 5, y + 7, 4, 7), QColor(255, 230, 90))
            p.restore()

    def _paint_player_figure(self, p: QPainter, box: QRect) -> None:
        """A little character (hard hat, hi-vis vest) facing the mouse."""
        cx, top = box.center().x() + 1, box.y() + 6
        mx, _my = self.mouse_gui
        look = max(-2, min(2, int((mx - cx) / 20)))
        parts = [((-4, 0, 8, 8), (196, 150, 110)),  # head
                 ((-5, -2, 10, 3), (250, 200, 30)), ((-6, 0, 12, 1), (230, 180, 20)),  # hard hat
                 ((-4, 8, 8, 12), (60, 70, 90)), ((-4, 8, 8, 10), (255, 120, 20)),  # body, vest
                 ((-4, 14, 8, 1), (230, 230, 200)),  # reflective stripe
                 ((-8, 8, 4, 12), (255, 120, 20)), ((4, 8, 4, 12), (255, 120, 20)),  # arms
                 ((-8, 17, 4, 3), (196, 150, 110)), ((4, 17, 4, 3), (196, 150, 110)),  # hands
                 ((-4, 20, 4, 12), (50, 60, 110)), ((0, 20, 4, 12), (45, 55, 100)),  # legs
                 ((-4, 30, 4, 2), (60, 50, 40)), ((0, 30, 4, 2), (60, 50, 40))]  # boots
        k = 1.8
        for (x, y, w, h), col in parts:
            p.fillRect(QRectF(cx + x * k, top + y * k, w * k, h * k), QColor(*col))
        # eyes follow the mouse
        for ex in (-3, 1):
            p.fillRect(QRectF(cx + (ex + look * 0.5) * k, top + 4 * k, 2 * k, k), QColor(255, 255, 255))
            p.fillRect(QRectF(cx + (ex + 1 + look * 0.5) * k, top + 4 * k, k, k), QColor(60, 90, 200))
