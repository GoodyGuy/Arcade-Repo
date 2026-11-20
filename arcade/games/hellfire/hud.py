"""Hellfire's HUD at the window's own resolution (QPainter): the vitals panel with the marine's face, the arms
panel with the ammo table, keys and weapon slots, power-up timers, the pickup messages, and the automap."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF

from .info import AMMO_NAMES, MONSTER_NAMES, WEAPON_NAMES, WEAPONS
from .level import CELL

RED = QColor(226, 44, 30)
GOLD = QColor(255, 196, 64)
PANEL = QColor(12, 9, 8, 178)
EDGE = QColor(150, 34, 20, 220)
AMMO_LABEL = {"bullets": "BULLETS", "shells": "SHELLS", "rockets": "ROCKETS", "cells": "CELLS"}
KEY_ICON = {"blue": "bluecard", "yellow": "yellowcard", "red": "redcard"}
KEY_COLOUR = {"blue": QColor(70, 120, 255), "yellow": QColor(255, 210, 40), "red": QColor(255, 54, 40)}


def hfont(px: float, bold: bool = True, family: str = "Impact") -> QFont:
    f = QFont(family)
    f.setPixelSize(max(int(px), 1))
    f.setBold(bold and family != "Impact")
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return f


def shadow_text(p: QPainter, x: float, y: float, text: str, colour: QColor, off: float = 2.0) -> None:
    p.setPen(QColor(0, 0, 0, 200))
    p.drawText(QPointF(x + off, y + off), text)
    p.setPen(colour)
    p.drawText(QPointF(x, y), text)


def panel(p: QPainter, r: QRectF, k: float) -> None:
    p.setPen(QPen(EDGE, max(2.0 * k, 1.0)))
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    g.setColorAt(0, QColor(28, 18, 14, 190))
    g.setColorAt(1, PANEL)
    p.setBrush(g)
    p.drawRoundedRect(r, 10 * k, 10 * k)


class FaceState:
    """Picks the marine's face from what happened (looking around, pain toward the attacker, ouch, grin, rage)."""

    def __init__(self) -> None:
        self.key: tuple = (0, 0, "calm")
        self.timer = 0
        self.priority = 0
        self.last_health = 100
        self.look = 0
        self.rng = np.random.default_rng(3)

    def update(self, world) -> None:  # noqa: ANN001
        pl = world.player
        level = min(max((100 - min(pl.health, 100)) * 5 // 101, 0), 4)
        if pl.dead or pl.health <= 0:
            self.key = ("dead",)
            self.priority = 10
            return
        if pl.powers.get("invuln"):
            self.key = ("god",)
            self.last_health = pl.health
            return
        if self.timer > 0:
            self.timer -= 1
        new_key, prio = None, 0
        if pl.got_weapon > 60:
            new_key, prio = (level, 0, "grin"), 8
        elif pl.damagecount and pl.health < self.last_health:
            dmg = self.last_health - pl.health
            if dmg > 20:
                new_key, prio = (level, 0, "ouch"), 7
            elif pl.attacker is not None and pl.attacker is not world.pmo:
                an = math.atan2(pl.attacker.y - world.pmo.y, pl.attacker.x - world.pmo.x)
                d = (an - world.pmo.angle + math.pi) % (2 * math.pi) - math.pi
                if abs(d) < math.radians(30):
                    new_key, prio = (level, 0, "rage"), 6
                else:
                    new_key, prio = (level, 0, "hurt_r" if d > 0 else "hurt_l"), 6
            else:
                new_key, prio = (level, 0, "rage"), 6
        elif pl.fire_time > 70:
            new_key, prio = (level, 0, "rage"), 5
        self.last_health = pl.health
        if new_key is not None and (prio >= self.priority or self.timer <= 0):
            self.key, self.priority, self.timer = new_key, prio, 35
            return
        if self.timer <= 0:
            self.priority = 0
            self.look = int(self.rng.integers(-1, 2))
            self.key = (level, self.look, "calm")
            self.timer = 17


_FACES: dict = {}


def face_image(key: tuple) -> QImage:
    img = _FACES.get(key)
    if img is None:
        from .assets import rgba_to_qimage
        from .palette import PALETTE
        from .sprites import art
        faces = art().faces
        s = faces.get(key) or faces[(0, 0, "calm")]
        rgba = np.dstack([PALETTE[s.img], np.where(s.mask, 255, 0).astype(np.uint8)])
        img = _FACES[key] = rgba_to_qimage(rgba)
    return img


class Hud:
    def __init__(self) -> None:
        self.face = FaceState()
        self.messages: list[list] = []  # [text, tics left]

    def message(self, text: str) -> None:
        if text:
            self.messages = ([[text, 140]] + self.messages)[:3]

    def tick(self, world) -> None:  # noqa: ANN001
        self.face.update(world)
        for m in self.messages:
            m[1] -= 1
        self.messages = [m for m in self.messages if m[1] > 0]

    # --- drawing ----------------------------------------------------------------------------------------------------

    def draw(self, p: QPainter, w: int, h: int, world, sprites, tick: int, automap: bool = False) -> None:  # noqa: ANN001
        k = max(min(w / 1280.0, h / 780.0), 0.45)
        pl = world.player
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        if not automap:
            self._messages(p, k)
        self._vitals(p, w, h, k, pl, tick)
        self._arms(p, w, h, k, pl, sprites, tick)
        self._powers(p, w, k, pl, sprites, tick)
        if not pl.dead:
            c = QPointF(w / 2, h / 2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 230, 200, 150))
            for dx, dy in ((-7, 0), (7, 0), (0, -7), (0, 7)):
                p.drawEllipse(c + QPointF(dx * k, dy * k), 1.4 * k, 1.4 * k)
        p.restore()

    def _messages(self, p: QPainter, k: float) -> None:
        p.setFont(hfont(22 * k, True, "Segoe UI"))
        for i, (text, left) in enumerate(self.messages):
            a = min(255, left * 12)
            col = QColor(255, 220, 160, a) if i == 0 else QColor(220, 180, 140, int(a * 0.7))
            shadow_text(p, 18 * k, (34 + i * 28) * k, text, col, 2 * k)

    def _vitals(self, p: QPainter, w: int, h: int, k: float, pl, tick: int) -> None:  # noqa: ANN001
        r = QRectF(16 * k, h - 112 * k, 372 * k, 96 * k)
        panel(p, r, k)
        img = face_image(self.face.key)
        fh = 80 * k
        fw = fh * img.width() / max(img.height(), 1)
        fr = QRectF(r.left() + 10 * k, r.top() + 8 * k, fw, fh)
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        p.setBrush(QColor(0, 0, 0, 160))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(fr.adjusted(-3 * k, -3 * k, 3 * k, 3 * k), 6 * k, 6 * k)
        p.drawImage(fr, img)
        p.restore()
        x = fr.right() + 18 * k
        hp = max(pl.health, 0)
        col = RED if hp <= 25 and (tick // 10) % 2 else (QColor(255, 236, 214) if hp > 25 else RED)
        p.setFont(hfont(64 * k))
        shadow_text(p, x, r.top() + 70 * k, f"{hp}", col, 3 * k)
        tw = p.fontMetrics().horizontalAdvance(f"{hp}")
        p.setFont(hfont(30 * k))
        shadow_text(p, x + tw + 4 * k, r.top() + 70 * k, "%", col, 2 * k)
        p.setFont(hfont(15 * k, True, "Segoe UI"))
        shadow_text(p, x, r.top() + 88 * k, "HEALTH", QColor(200, 150, 120), 1 * k)
        # armour
        ax = r.left() + 268 * k
        acol = QColor(90, 220, 90) if pl.armortype == 1 else QColor(90, 150, 255) if pl.armortype == 2 \
            else QColor(140, 130, 120)
        shield = QPainterPath()
        sx, sy, ss = ax, r.top() + 22 * k, 22 * k
        shield.moveTo(sx, sy)
        shield.lineTo(sx + ss, sy - 4 * k)
        shield.lineTo(sx + 2 * ss, sy)
        shield.cubicTo(sx + 2 * ss, sy + ss * 1.2, sx + ss * 1.3, sy + ss * 1.6, sx + ss, sy + ss * 1.9)
        shield.cubicTo(sx + ss * 0.7, sy + ss * 1.6, sx, sy + ss * 1.2, sx, sy)
        p.setPen(QPen(QColor(0, 0, 0, 200), 2 * k))
        p.setBrush(acol)
        p.drawPath(shield)
        p.setFont(hfont(34 * k))
        shadow_text(p, ax + 52 * k, r.top() + 54 * k, f"{pl.armor}", QColor(255, 236, 214), 2 * k)
        p.setFont(hfont(15 * k, True, "Segoe UI"))
        shadow_text(p, ax, r.top() + 88 * k, "ARMOUR", QColor(200, 150, 120), 1 * k)

    def _arms(self, p: QPainter, w: int, h: int, k: float, pl, sprites, tick: int) -> None:  # noqa: ANN001
        r = QRectF(w - 388 * k, h - 112 * k, 372 * k, 96 * k)
        panel(p, r, k)
        info = WEAPONS[pl.weapon]
        ammo = info["ammo"]
        p.setFont(hfont(15 * k, True, "Segoe UI"))
        shadow_text(p, r.left() + 16 * k, r.top() + 24 * k, WEAPON_NAMES.get(pl.weapon, "").upper(),
                    QColor(200, 150, 120), 1 * k)
        p.setFont(hfont(64 * k))
        txt = str(pl.ammo[ammo]) if ammo else "--"
        low = ammo and pl.ammo[ammo] < max(info["use"] * 5, 5)
        shadow_text(p, r.left() + 16 * k, r.top() + 84 * k, txt, RED if low else QColor(255, 236, 214), 3 * k)
        # the ammo table
        p.setFont(hfont(15 * k, True, "Consolas"))
        for i, a in enumerate(AMMO_NAMES):
            y = r.top() + (22 + i * 20) * k
            cur = a == ammo
            col = GOLD if cur else QColor(180, 160, 140)
            shadow_text(p, r.left() + 170 * k, y, AMMO_LABEL[a], col, 1 * k)
            v = f"{pl.ammo[a]:>3}/{pl.maxammo[a]}"
            shadow_text(p, r.left() + 270 * k, y, v, col, 1 * k)
        # weapon slots and keys above the panel
        p.setFont(hfont(17 * k, True, "Consolas"))
        for slot in range(1, 8):
            owned = any(WEAPONS[n]["slot"] == slot for n in pl.weapons)
            current = WEAPONS[pl.weapon]["slot"] == slot
            x = r.left() + (8 + (slot - 1) * 26) * k
            box = QRectF(x, r.top() - 32 * k, 22 * k, 24 * k)
            p.setPen(QPen(GOLD if current else QColor(90, 60, 40, 200), 1.5 * k))
            p.setBrush(QColor(40, 20, 10, 200) if owned else QColor(10, 8, 8, 140))
            p.drawRoundedRect(box, 4 * k, 4 * k)
            p.setPen(GOLD if current else QColor(240, 210, 170) if owned else QColor(90, 80, 70))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, str(slot))
        for i, key in enumerate(("blue", "yellow", "red")):
            x = r.right() - (34 + (2 - i) * 34) * k
            box = QRectF(x, r.top() - 34 * k, 28 * k, 28 * k)
            p.setPen(QPen(QColor(90, 60, 40, 200), 1.5 * k))
            p.setBrush(QColor(10, 8, 8, 150))
            p.drawRoundedRect(box, 4 * k, 4 * k)
            if key in pl.keys:
                icon = sprites.icon(KEY_ICON[key], tick // 12)
                if not icon.isNull():
                    s = min(box.width() * 0.8 / icon.width(), box.height() * 0.8 / icon.height())
                    iw, ih = icon.width() * s, icon.height() * s
                    p.save()
                    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
                    p.drawImage(QRectF(box.center().x() - iw / 2, box.center().y() - ih / 2, iw, ih), icon)
                    p.restore()
                else:
                    p.fillRect(box.adjusted(6 * k, 6 * k, -6 * k, -6 * k), KEY_COLOUR[key])

    def _powers(self, p: QPainter, w: int, k: float, pl, sprites, tick: int) -> None:  # noqa: ANN001
        items = []
        for key, icon in (("invuln", "invuln"), ("suit", "suit"), ("berserk", "berserk"), ("scanner", "scanner")):
            v = pl.powers.get(key)
            if v:
                items.append((icon, f"{(v + 34) // 35}s" if key in ("invuln", "suit") else ""))
        y = 20 * k
        p.setFont(hfont(26 * k))
        for icon_key, label in items:
            icon = sprites.icon(icon_key, tick // 8)
            box = QRectF(w - 70 * k, y, 50 * k, 50 * k)
            p.setPen(QPen(EDGE, 1.5 * k))
            p.setBrush(PANEL)
            p.drawRoundedRect(box, 6 * k, 6 * k)
            if not icon.isNull():
                s = min(box.width() * 0.8 / icon.width(), box.height() * 0.8 / icon.height())
                iw, ih = icon.width() * s, icon.height() * s
                p.save()
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
                p.drawImage(QRectF(box.center().x() - iw / 2, box.center().y() - ih / 2, iw, ih), icon)
                p.restore()
            if label:
                fm = p.fontMetrics()
                shadow_text(p, box.left() - fm.horizontalAdvance(label) - 10 * k, box.bottom() - 12 * k, label,
                            GOLD, 2 * k)
            y += 58 * k


# --- automap ------------------------------------------------------------------------------------------------------

def draw_automap(p: QPainter, w: int, h: int, world, scale: float, tiles=None) -> None:  # noqa: ANN001
    """The map of what the player has seen (everything with the base scanner), centred on the player."""
    g = world.grid
    k = max(min(w / 1280.0, h / 780.0), 0.45)
    p.save()
    p.fillRect(QRectF(0, 0, w, h), QColor(6, 4, 4, 225))
    px, py = world.pmo.x, world.pmo.y
    s = scale * k / CELL  # pixels per map unit
    ox, oy = w / 2 - px * s, h / 2 - py * s
    seen = world.seen
    full = bool(world.player.powers.get("scanner"))
    solid = g.solid
    door = np.array([sp.get("kind") == "door" for sp in g.specs])
    secret = np.array([bool(sp.get("secretwall")) for sp in g.specs])
    keyed = {kk: np.array([sp.get("key") == kk for sp in g.specs]) for kk in ("blue", "yellow", "red")}
    lines: dict[str, list] = {"wall": [], "step": [], "ceil": [], "door": [], "unseen": [], "blue": [],
                              "yellow": [], "red": []}
    W, H = g.w, g.h
    for horiz in (True, False):
        if horiz:  # the boundary between (x, y) and (x + 1, y)
            a = np.arange(W * H).reshape(H, W)[:, :-1].ravel()
            b = a + 1
        else:
            a = np.arange(W * H).reshape(H, W)[:-1, :].ravel()
            b = a + W
        sa, sb = solid[a] | secret[a], solid[b] | secret[b]
        vis = seen[a] | seen[b]
        both_solid = sa & sb
        wall = (sa != sb) & ~both_solid
        opn = ~sa & ~sb
        dr = (door[a] | door[b]) & ~secret[a] & ~secret[b]
        step = opn & ~dr & (np.abs(g.fh[a] - g.fh[b]) > 0.5)
        ceil = opn & ~dr & ~step & (np.abs(g.ch[a] - g.ch[b]) > 0.5) & ~(g.sky[a] & g.sky[b])
        door_line = dr & ~both_solid
        ay, ax = np.divmod(a, W)
        if horiz:
            x0 = (ax + 1) * CELL
            seg = np.stack([x0, ay * CELL, x0, (ay + 1) * CELL], axis=1)
        else:
            y0 = (ay + 1) * CELL
            seg = np.stack([ax * CELL, y0, (ax + 1) * CELL, y0], axis=1)
        for name, m in (("wall", wall), ("step", step), ("ceil", ceil)):
            lines[name].append(seg[m & vis])
            if full:
                lines["unseen"].append(seg[m & ~vis])
        for kk in ("blue", "yellow", "red"):
            km = door_line & (keyed[kk][a] | keyed[kk][b])
            lines[kk].append(seg[km & (vis | full)])
            door_line = door_line & ~km
        lines["door"].append(seg[door_line & vis])
        if full:
            lines["unseen"].append(seg[door_line & ~vis])
    style = {"unseen": (QColor(110, 110, 120), 1.0), "ceil": (QColor(150, 130, 60), 1.2),
             "step": (QColor(170, 110, 60), 1.4), "door": (QColor(255, 220, 60), 2.2), "wall": (RED, 2.0),
             "blue": (KEY_COLOUR["blue"], 3.0), "yellow": (KEY_COLOUR["yellow"], 3.0), "red": (KEY_COLOUR["red"], 3.0)}
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for name in ("unseen", "ceil", "step", "door", "wall", "blue", "yellow", "red"):
        segs = np.concatenate(lines[name]) if lines[name] else np.zeros((0, 4))
        if not len(segs):
            continue
        sc = segs * s + np.array([ox, oy, ox, oy])
        col, width = style[name]
        p.setPen(QPen(col, max(width * k * scale / 7.0, 1.0)))
        p.drawLines([QLineF(*row) for row in sc.tolist()])
    # the player
    a = world.pmo.angle
    c = QPointF(w / 2, h / 2)
    r = 14 * k
    tri = QPolygonF([c + QPointF(math.cos(a) * r, math.sin(a) * r),
                     c + QPointF(math.cos(a + 2.5) * r * 0.75, math.sin(a + 2.5) * r * 0.75),
                     c + QPointF(math.cos(a - 2.5) * r * 0.75, math.sin(a - 2.5) * r * 0.75)])
    p.setPen(QPen(QColor(255, 255, 255), 2 * k))
    p.setBrush(QColor(80, 255, 80))
    p.drawPolygon(tri)
    # the title and the tally so far
    pl = world.player
    p.setFont(hfont(30 * k))
    m = world.mapdef
    shadow_text(p, 24 * k, 46 * k, f"{m.key}  {m.name.upper()}", GOLD, 2 * k)
    p.setFont(hfont(18 * k, True, "Segoe UI"))
    t = world.leveltime // 35
    tally = (f"KILLS {pl.killcount}/{world.total_kills}    ITEMS {pl.itemcount}/{world.total_items}    "
             f"SECRETS {pl.secretcount}/{world.total_secrets}    TIME {t // 60}:{t % 60:02d}")
    shadow_text(p, 24 * k, 76 * k, tally, QColor(230, 200, 170), 2 * k)
    p.restore()


__all__ = ["Hud", "FaceState", "draw_automap", "MONSTER_NAMES", "hfont", "shadow_text", "panel"]
