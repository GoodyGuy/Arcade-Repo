"""The Aim Arena: a pseudo-3D sunny desert courtyard (sand walls, arches, crates, blue sky) where bots pop up from
cover; aim with the cursor, headshots pay more. Drawn in a 1600 x 900 virtual scene scaled to cover the widget."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF, QRadialGradient)

from app.arcade.games.case_clash import art as A
from app.arcade.games.case_clash import data as D
from app.arcade.games.case_clash import ui as U
from app.arcade.games.case_clash.shapes import GRIPS

VW, VH = 1600.0, 900.0
ROUND_SECONDS = 45.0
BOT_NAMES = ("Ivan", "Rook", "Moss", "Dusty", "Vex", "Kilo", "Sable", "Brick", "Nomad", "Jinx", "Talon", "Gus",
             "Hex", "Mako", "Raz", "Oz", "Pike", "Zed")


def _poly(*pts: tuple[float, float]) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in pts])


def _path(*pts: tuple[float, float]) -> QPainterPath:
    path = QPainterPath()
    path.addPolygon(_poly(*pts))
    path.closeSubpath()
    return path


def _arch_path() -> QPainterPath:
    path = QPainterPath(QPointF(700, 520))
    path.lineTo(700, 330)
    path.arcTo(QRectF(700, 230, 200, 200), 180, -180)
    path.lineTo(900, 520)
    path.closeSubpath()
    return path


@dataclass(frozen=True)
class Spot:
    """Where a bot appears: its feet when fully out, size, the visible region and how it moves in."""
    x: float
    y: float
    scale: float
    clip: QPainterPath
    mode: str  # rise / left / right


def _rect_path(x0: float, y0: float, x1: float, y1: float) -> QPainterPath:
    path = QPainterPath()
    path.addRect(QRectF(x0, y0, x1 - x0, y1 - y0))
    return path


SPOTS = (
    Spot(610, 562 + 0.45 * 270, 0.9, _rect_path(430, 0, 790, 562), "rise"),  # behind the near crate
    Spot(1020, 440 + 0.45 * 126, 0.42, _rect_path(930, 0, 1110, 440), "rise"),  # behind the far crate
    Spot(1285, 612 + 0.42 * 405, 1.35, _rect_path(1100, 0, 1480, 612), "rise"),  # behind the big crate
    Spot(765, 520, 0.6, _arch_path(), "left"),  # peeking from the arch
    Spot(838, 520, 0.6, _arch_path(), "right"),
    Spot(545, 330 + 0.5 * 105, 0.35, _rect_path(500, 250, 590, 330), "rise"),  # the far window
    Spot(205, 735, 0.85, _path((150, 392), (250, 369), (250, 679), (150, 767)), "left"),  # the left doorway
    Spot(1360, 398 + 0.5 * 225, 0.75, _path((1300, 280), (1420, 285), (1420, 415), (1300, 381)), "rise"),
    Spot(900, 172 + 0.5 * 120, 0.4, _rect_path(430, 0, 1170, 172), "rise"),  # on the roof
)


@dataclass
class Bot:
    spot: int
    name: str
    stay: float
    t: float = 0.0  # s since it started to appear
    dead: float = -1.0  # s since it died (-1: alive)
    headshot: bool = False
    fired: bool = False
    look: int = 0  # which outfit

    @property
    def H(self) -> float:  # noqa: N802
        return 300.0 * SPOTS[self.spot].scale


@dataclass
class Popup:
    x: float
    y: float
    text: str
    color: str
    t: float = 0.0


@dataclass
class ArenaStats:
    kills: int = 0
    headshots: int = 0
    shots: int = 0
    hits: int = 0
    combo: int = 0
    best_combo: int = 0
    earned: int = 0
    damage_taken: int = 0
    feed: list = field(default_factory=list)


class Arena:
    """One Aim Arena session (several rounds) inside the game widget."""

    def __init__(self, game) -> None:  # noqa: ANN001 - CaseClashGame
        self.g = game
        self.rng = random.Random()
        skin, seed, flt, item = game.equipped_gun()
        self.skin, self.seed, self.flt, self.item = skin, seed, flt, item
        self.weapon = D.WEAPONS[skin.weapon]
        self._bg_key = None
        self._bg: QPixmap | None = None
        self._vm_key = None
        self._vm: QPixmap | None = None
        self._muzzle = QPointF()
        self._grip = QRectF()
        self.aim = QPointF(-1, -1)  # the crosshair (widget pixels; moved by the captured mouse)
        self.new_round()

    # --- round flow ---------------------------------------------------------------------------------------------

    def new_round(self) -> None:
        self.phase = "countdown"
        self.clock = 3.0
        self.left = ROUND_SECONDS
        self.health = 100
        self.bots: list[Bot] = []
        self.popups: list[Popup] = []
        self.decals: list[list[float]] = []
        self.spawn_in = 0.4
        self.stats = ArenaStats()
        self.ammo = self.weapon.mag
        self.reload = 0.0
        self.cool = 0.0
        self.trigger = False
        self.recoil = 0.0
        self.flash = 0.0
        self.hurt = 0.0
        self.shake = 0.0
        self.hitmark = 0.0
        self.hs_mark = 0.0
        self.result: dict = {}
        self._beeped = 3
        self.aim = QPointF(-1, -1)  # back to the middle

    def finish(self) -> None:
        if self.phase != "live":
            return
        self.phase = "results"
        self.trigger = False
        self.result = self.g.arena_done(self.stats, self.health > 0)
        self.g.set_arena_cursor(False)

    # --- input ----------------------------------------------------------------------------------------------------

    def to_virtual(self, pos: QPointF, w: float, h: float) -> QPointF:
        s = max(w / VW, h / VH)
        return QPointF((pos.x() - (w - VW * s) / 2) / s, (pos.y() - (h - VH * s) / 2) / s)

    def aim_point(self) -> QPointF:
        """The crosshair; the middle of the view until the mouse moves it."""
        if self.aim.x() < 0:
            self.aim = QPointF(self.g.width() / 2, self.g.height() / 2)
        return self.aim

    def steer(self) -> None:
        """Move the crosshair by the captured mouse's motion (the player's sensitivity and invert-Y included)."""
        if not self.g.mouse_captured:
            return
        dx, dy = self.g.take_mouse_delta()
        a = self.aim_point()
        self.aim = QPointF(min(max(a.x() + dx, 0.0), float(self.g.width())),
                           min(max(a.y() + dy, 0.0), float(self.g.height())))

    def press(self, pos: QPointF) -> None:
        if not self.g.mouse_captured:  # a free cursor aims where it is
            self.aim = QPointF(pos)
        if self.phase != "live":
            return
        self.trigger = True
        if self.cool <= 0.0:
            self.fire()

    def release(self) -> None:
        self.trigger = False

    def move(self, pos: QPointF) -> None:
        if not self.g.mouse_captured:
            self.aim = QPointF(pos)

    def key(self, key: int) -> None:
        if key == Qt.Key.Key_R and self.phase == "live":
            self.start_reload()

    def start_reload(self) -> None:
        if self.reload <= 0.0 and self.ammo < self.weapon.mag:
            self.reload = {"pistol": 1.2, "sniper": 2.2, "heavy": 2.0}.get(self.weapon.kind, 1.8)
            self.g.sfx.play("reload")

    # --- shooting -------------------------------------------------------------------------------------------------

    def fire(self) -> None:
        if self.reload > 0.0:
            return
        if self.ammo <= 0:
            self.g.sfx.play("empty")
            self.start_reload()
            return
        self.ammo -= 1
        self.cool = self.weapon.rate
        self.flash = 0.06
        self.shake = min(self.shake + (0.5 if self.weapon.kind == "sniper" else 0.22), 1.0)
        self.g.sfx.play({"sniper": "shot_sniper", "heavy": "shot_heavy", "pistol": "shot_pistol"}.get(
            self.weapon.kind, "shot_rifle"))
        self.stats.shots += 1
        w, h = self.g.width(), self.g.height()
        aim = self.to_virtual(self.aim_point(), w, h)
        spread = self.recoil * (18.0 if self.weapon.auto else 6.0)
        aim = QPointF(aim.x() + self.rng.gauss(0, spread), aim.y() + self.rng.gauss(0, spread))
        self.recoil = min(self.recoil + (0.35 if self.weapon.auto else 0.6), 1.0)
        pellets = [aim]
        if self.weapon.kind == "heavy" and not self.weapon.auto:  # a shotgun spreads
            pellets += [QPointF(aim.x() + self.rng.gauss(0, 14), aim.y() + self.rng.gauss(0, 14)) for _ in range(5)]
        hit_any = False
        for pt in pellets:
            if self.shoot_at(pt):
                hit_any = True
                break
        if not hit_any:
            self.stats.combo = 0
            if len(self.decals) > 40:
                self.decals.pop(0)
            self.decals.append([aim.x(), aim.y(), 0.0])
        if self.ammo == 0:
            self.start_reload()

    def bot_geometry(self, b: Bot) -> tuple[float, float, float]:
        """Feet x, feet y and height of the bot now (moving in / out of cover)."""
        sp = SPOTS[b.spot]
        H = b.H  # noqa: N806
        k = self.out_amount(b)
        x, y = sp.x, sp.y
        if sp.mode == "rise":
            y += (1 - k) * H * 0.8
        elif sp.mode == "left":
            x -= (1 - k) * H * 0.45
        else:
            x += (1 - k) * H * 0.45
        return x, y, H

    def out_amount(self, b: Bot) -> float:
        if b.dead >= 0:
            return 1.0
        if b.t < 0.16:
            k = b.t / 0.16
        elif b.t > b.stay + 0.16:
            k = 1 - (b.t - b.stay - 0.16) / 0.16
        else:
            k = 1.0
        k = max(0.0, min(k, 1.0))
        return k * k * (3 - 2 * k)

    def shoot_at(self, pt: QPointF) -> bool:
        """Test a shot against the bots (front ones first); True on a hit."""
        for b in sorted(self.bots, key=lambda b: -SPOTS[b.spot].scale):
            if b.dead >= 0 or not SPOTS[b.spot].clip.contains(pt):
                continue
            x, y, H = self.bot_geometry(b)  # noqa: N806
            hx, hy, hr = x, y - H * 0.9, H * 0.085
            head = (pt.x() - hx) ** 2 + (pt.y() - hy) ** 2 <= hr * hr
            body = QRectF(x - H * 0.15, y - H * 0.83, H * 0.3, H * 0.38).contains(pt) or \
                QRectF(x - H * 0.11, y - H * 0.46, H * 0.22, H * 0.46).contains(pt)
            if head or body:
                self.kill(b, head)
                return True
        return False

    def kill(self, b: Bot, head: bool) -> None:
        b.dead = 0.0
        b.headshot = head
        st = self.stats
        st.kills += 1
        st.hits += 1
        st.combo += 1
        st.best_combo = max(st.best_combo, st.combo)
        if head:
            st.headshots += 1
        mult = 1.0 + min(st.combo, 10) * 0.1
        cents = int(round((20 if head else 8) * mult * (1.0 + 0.4 * (SPOTS[b.spot].scale < 0.5))))
        st.earned += cents
        self.g.earn(cents, "arena")
        self.g.add_xp(16 if head else 10)
        if self.item is not None and self.item.counter:
            self.item.kills = (self.item.kills or 0) + 1
        x, y, H = self.bot_geometry(b)  # noqa: N806
        self.popups.append(Popup(x, y - H * 1.02, f"+{D.money(cents)}", "#8bd45b"))
        st.feed.insert(0, [b.name, head, 0.0])
        del st.feed[5:]
        self.hitmark = 0.18
        if head:
            self.hs_mark = 0.6
            self.g.sfx.play("ding")
        else:
            self.g.sfx.play("hit")

    # --- simulation -----------------------------------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        self.steer()
        for pp in self.popups:
            pp.t += dt
        self.popups = [pp for pp in self.popups if pp.t < 0.9]
        for e in self.stats.feed:
            e[2] += dt
        self.stats.feed = [e for e in self.stats.feed if e[2] < 5.0]
        for d in self.decals:
            d[2] += dt
        self.flash = max(0.0, self.flash - dt)
        self.hurt = max(0.0, self.hurt - dt * 1.6)
        self.shake = max(0.0, self.shake - dt * 4)
        self.hitmark = max(0.0, self.hitmark - dt)
        self.hs_mark = max(0.0, self.hs_mark - dt)
        self.recoil = max(0.0, self.recoil - dt * (2.5 if self.weapon.auto else 1.6))
        if self.phase == "countdown":
            self.clock -= dt
            if math.ceil(self.clock) < self._beeped and self.clock > 0:
                self._beeped = math.ceil(self.clock)
                self.g.sfx.play("beep")
            if self.clock <= 0:
                self.phase = "live"
                self.g.sfx.play("go")
            return
        if self.phase != "live":
            return
        self.left -= dt
        self.cool -= dt
        if self.reload > 0.0:
            self.reload -= dt
            if self.reload <= 0.0:
                self.reload = 0.0
                self.ammo = self.weapon.mag
        if self.trigger and self.weapon.auto and self.cool <= 0.0:
            self.fire()
        # bots
        for b in self.bots:
            if b.dead >= 0:
                b.dead += dt
                continue
            b.t += dt
            if not b.fired and b.t > b.stay + 0.16:
                b.fired = True
                dmg = self.rng.randint(12, 22)
                self.health = max(0, self.health - dmg)
                self.stats.damage_taken += dmg
                self.stats.combo = 0
                self.hurt = 1.0
                self.g.sfx.play("hurt")
        self.bots = [b for b in self.bots if (b.dead < 0.6 if b.dead >= 0 else b.t < b.stay + 0.34)]
        self.spawn_in -= dt
        elapsed = ROUND_SECONDS - self.left
        busy = {b.spot for b in self.bots}
        if self.spawn_in <= 0 and len(self.bots) < (2 if elapsed < 15 else 3):
            free = [i for i in range(len(SPOTS)) if i not in busy]
            if free:
                stay = max(0.75, 1.5 - elapsed * 0.012 - self.stats.kills * 0.008)
                self.bots.append(Bot(self.rng.choice(free), "Bot " + self.rng.choice(BOT_NAMES), stay,
                                     look=self.rng.randrange(len(OUTFITS))))
            self.spawn_in = max(0.35, 0.9 - elapsed * 0.01) * self.rng.uniform(0.7, 1.3)
        if self.left <= 0 or self.health <= 0:
            self.left = max(self.left, 0.0)
            self.finish()

    # --- drawing --------------------------------------------------------------------------------------------------

    def background(self, w: int, h: int) -> QPixmap:
        if self._bg_key != (w, h, A.dpr()) or self._bg is None:
            self._bg = scene_pixmap(w, h)
            self._bg_key = (w, h, A.dpr())
        return self._bg

    def paint(self, p: QPainter, w: int, h: int, hits: U.Hits) -> None:
        s = max(w / VW, h / VH)
        ox, oy = (w - VW * s) / 2, (h - VH * s) / 2
        sh = self.shake * 6
        jx, jy = (self.rng.uniform(-sh, sh), self.rng.uniform(-sh, sh)) if sh > 0.3 else (0.0, 0.0)
        p.drawPixmap(QPointF(jx, jy), self.background(w, h))
        p.save()
        p.translate(ox + jx, oy + jy)
        p.scale(s, s)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for d in self.decals:
            a = max(0, int(200 * (1 - d[2] / 4.0)))
            if a:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(40, 30, 20, a))
                p.drawEllipse(QPointF(d[0], d[1]), 3.2, 3.2)
                p.setBrush(QColor(255, 240, 200, a // 3))
                p.drawEllipse(QPointF(d[0] - 1, d[1] - 1), 1.2, 1.2)
        for b in sorted(self.bots, key=lambda b: SPOTS[b.spot].scale):
            self.paint_bot(p, b)
        for pp in self.popups:
            a = max(0, int(255 * (1 - pp.t / 0.9)))
            c = QColor(pp.color)
            c.setAlpha(a)
            U.text(p, QRectF(pp.x - 80, pp.y - 30 - pp.t * 40, 160, 30), pp.text, 26, c, Qt.AlignmentFlag.AlignHCenter,
                   "bold", True, False, True)
        p.restore()
        self.paint_viewmodel(p, w, h)
        self.paint_hud(p, w, h, hits)

    def paint_bot(self, p: QPainter, b: Bot) -> None:
        sp = SPOTS[b.spot]
        x, y, H = self.bot_geometry(b)  # noqa: N806
        p.save()
        p.setClipPath(sp.clip)
        if b.dead >= 0:
            k = min(b.dead / 0.45, 1.0)
            p.translate(x, y)
            p.rotate((-1 if sp.mode == "right" else 1) * -75 * k * k)
            p.translate(-x, -y + H * 0.15 * k)
            p.setOpacity(max(0.0, 1 - max(0.0, b.dead - 0.35) / 0.25))
        draw_soldier(p, x, y, H, aiming=b.dead < 0 and b.t > b.stay - 0.25, look=b.look)
        if b.dead < 0 and b.fired and b.t < b.stay + 0.26:
            fx, fy = x - H * 0.26, y - H * 0.66
            rg = QRadialGradient(QPointF(fx, fy), H * 0.12)
            rg.setColorAt(0.0, QColor(255, 250, 200, 240))
            rg.setColorAt(1.0, QColor(255, 170, 40, 0))
            p.setBrush(rg)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(fx, fy), H * 0.12, H * 0.12)
        if b.dead >= 0 and b.dead < 0.25:
            hx, hy = x, y - H * (0.9 if b.headshot else 0.65)
            p.setOpacity(1.0)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(170, 20, 20, int(200 * (1 - b.dead / 0.25))))
            for i in range(6):
                a = i * 1.05 + b.spot
                r = H * 0.03 + b.dead * H * 0.25
                p.drawEllipse(QPointF(hx + math.cos(a) * r, hy + math.sin(a) * r * 0.6), H * 0.018, H * 0.018)
        p.restore()

    def viewmodel(self, w: int, h: int) -> QPixmap:
        fw, fh = {"pistol": (0.27, 0.2), "smg": (0.33, 0.25)}.get(self.weapon.kind, (0.40, 0.30))
        vw, vh = int(w * fw), int(h * fh)
        if self._vm_key != (vw, vh, A.dpr()):
            self._vm = A.art_pixmap(self.skin.id, self.seed, self.flt, vw, vh, flip=True)
            parts, b = A.shape(self.skin.weapon, flip=True)
            sc = min(vw / b.width(), vh / b.height())
            front = min(parts, key=lambda kp: kp[1].boundingRect().left())[1].boundingRect()
            self._muzzle = QPointF(vw / 2 + (front.left() - b.center().x()) * sc,
                                   vh / 2 + (front.center().y() - b.center().y()) * sc)
            gx0, gy0, gx1, gy1 = GRIPS.get(self.skin.weapon, (28, 20, 36, 29))
            grip = QRectF(-gx1, gy0, gx1 - gx0, gy1 - gy0)  # mirrored like the view model
            self._grip = QRectF(vw / 2 + (grip.left() - b.center().x()) * sc,
                                vh / 2 + (grip.top() - b.center().y()) * sc, grip.width() * sc, grip.height() * sc)
            self._vm_key = (vw, vh, A.dpr())
        return self._vm

    def paint_viewmodel(self, p: QPainter, w: int, h: int) -> None:
        pm = self.viewmodel(w, h)
        pw, ph = A.lsize(pm)
        t = self.g.time
        kick = min(self.cool / max(self.weapon.rate, 0.05), 1.0) if self.flash > 0 or self.cool > 0 else 0.0
        kick = max(kick, 0.0) * (1.6 if self.weapon.kind == "sniper" else 1.0)
        rl = math.sin(min(self.reload, 1.0) * math.pi) if self.reload > 0 else 0.0
        lift = {"pistol": 1.45, "smg": 1.25}.get(self.weapon.kind, 1.12)
        bx = w - pw * 1.04 + math.sin(t * 1.7) * 3 + kick * 10
        by = h - ph * lift + math.cos(t * 3.4) * 2 + kick * 12 + rl * h * 0.18
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.translate(bx + pw / 2, by + ph / 2)
        p.rotate(11 - kick * 5 + rl * 25)
        p.translate(-pw / 2, -ph / 2)
        # the arm comes in from the bottom right; the gloved hand closes on the grip
        gloves = self.g.equipped_gloves()
        gr = self._grip
        hc = gr.center()
        hw = gr.width() * 1.25
        hh = gr.height() * 0.75
        sleeve = QPainterPath(QPointF(hc.x() - hw * 0.3, hc.y() + hh * 0.2))
        sleeve.lineTo(hc.x() + hw * 0.55, hc.y() - hh * 0.1)
        sleeve.lineTo(pw * 1.3, ph * 1.3)
        sleeve.lineTo(pw * 0.95, ph * 1.9)
        sleeve.closeSubpath()
        sg = QLinearGradient(hc, QPointF(pw * 1.2, ph * 1.6))
        sg.setColorAt(0.0, QColor("#6a6c52"))
        sg.setColorAt(1.0, QColor("#2c2e22"))
        p.setPen(QPen(QColor(0, 0, 0, 140), 1.5))
        p.setBrush(sg)
        p.drawPath(sleeve)
        p.drawPixmap(0, 0, pm)
        hand = QColor(gloves.colors[0]) if gloves else QColor("#5a5448")
        hg = QLinearGradient(QPointF(0, hc.y() - hh / 2), QPointF(0, hc.y() + hh / 2))
        hg.setColorAt(0.0, hand.lighter(125))
        hg.setColorAt(1.0, hand.darker(170))
        p.setBrush(hg)
        p.setPen(QPen(QColor(0, 0, 0, 170), 1.5))
        p.drawRoundedRect(QRectF(hc.x() - hw / 2, hc.y() - hh / 2, hw, hh), hh * 0.45, hh * 0.45)
        p.setPen(QPen(QColor(0, 0, 0, 90), 1.2))
        for i in range(1, 4):  # fingers
            fx = hc.x() - hw / 2 + hw * i / 4
            p.drawLine(QPointF(fx, hc.y() - hh * 0.1), QPointF(fx, hc.y() + hh * 0.45))
        if gloves:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(gloves.colors[2]))
            p.drawRect(QRectF(hc.x() + hw * 0.25, hc.y() - hh * 0.4, hw * 0.18, hh * 0.8))
        if self.flash > 0:
            m = self._muzzle
            r = ph * (0.35 if self.weapon.kind in ("rifle", "heavy", "sniper") else 0.25)
            rg = QRadialGradient(m, r)
            rg.setColorAt(0.0, QColor(255, 255, 230, 255))
            rg.setColorAt(0.3, QColor(255, 210, 90, 220))
            rg.setColorAt(1.0, QColor(255, 120, 20, 0))
            p.setBrush(rg)
            star = QPainterPath()
            for i in range(16):
                a = i * math.pi / 8 + self.g.time * 7
                rr = r * (1.0 if i % 2 == 0 else 0.4)
                pt = QPointF(m.x() + math.cos(a) * rr * 1.4, m.y() + math.sin(a) * rr * 0.8)
                star.lineTo(pt) if i else star.moveTo(pt)
            star.closeSubpath()
            p.drawPath(star)
        p.restore()

    def paint_hud(self, p: QPainter, w: int, h: int, hits: U.Hits) -> None:
        s = max(min(w / 1280, h / 780), 0.6)
        if self.hurt > 0:
            rg = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) * 0.7)
            rg.setColorAt(0.55, QColor(200, 0, 0, 0))
            rg.setColorAt(1.0, QColor(200, 0, 0, int(150 * self.hurt)))
            p.fillRect(QRectF(0, 0, w, h), rg)
        # crosshair (classic green lines, opening with recoil)
        if self.phase == "live" or self.phase == "countdown":
            m = self.aim_point()
            gap = (4 + self.recoil * 14) * s
            ln = 9 * s
            p.save()
            for col, wd in ((QColor(0, 0, 0, 180), 4 * s), (QColor("#4dff4d"), 2 * s)):
                p.setPen(QPen(col, wd))
                p.drawLine(QPointF(m.x() - gap - ln, m.y()), QPointF(m.x() - gap, m.y()))
                p.drawLine(QPointF(m.x() + gap, m.y()), QPointF(m.x() + gap + ln, m.y()))
                p.drawLine(QPointF(m.x(), m.y() - gap - ln), QPointF(m.x(), m.y() - gap))
                p.drawLine(QPointF(m.x(), m.y() + gap), QPointF(m.x(), m.y() + gap + ln))
            if self.hitmark > 0:
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setPen(QPen(QColor(255, 255, 255, int(255 * self.hitmark / 0.18)), 2.4 * s))
                for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                    p.drawLine(QPointF(m.x() + dx * 6 * s, m.y() + dy * 6 * s),
                               QPointF(m.x() + dx * 13 * s, m.y() + dy * 13 * s))
            p.restore()
        # top: timer and kills
        tb = QRectF(w / 2 - 110 * s, 10 * s, 220 * s, 46 * s)
        p.fillRect(tb, QColor(0, 0, 0, 120))
        secs = max(0, math.ceil(self.left))
        U.text(p, tb.adjusted(0, 0, 0, -16 * s), f"{secs // 60}:{secs % 60:02d}", 24 * s,
               U.RED if secs <= 10 and self.phase == "live" else U.TEXT, Qt.AlignmentFlag.AlignHCenter, "bold", True)
        U.text(p, tb.adjusted(0, 26 * s, 0, 0), f"KILLS {self.stats.kills}   HS {self.stats.headshots}", 12 * s,
               U.DIM, Qt.AlignmentFlag.AlignHCenter, cond=True)
        # earnings, top left
        eb = QRectF(14 * s, 12 * s, 230 * s, 40 * s)
        p.fillRect(eb, QColor(0, 0, 0, 110))
        U.text(p, eb.adjusted(10 * s, 0, 0, 0), f"ROUND  +{D.money(self.stats.earned)}", 20 * s, U.MONEY,
               weight="bold", cond=True)
        # kill feed, top right
        y = 12 * s
        for name, head, age in self.stats.feed:
            a = 1.0 if age < 4.5 else max(0.0, (5.0 - age) / 0.5)
            fw = 300 * s
            r = QRectF(w - fw - 14 * s, y, fw, 26 * s)
            p.setOpacity(a)
            p.fillRect(r, QColor(0, 0, 0, 150))
            p.setPen(QPen(QColor("#c8302f"), 1.5))
            p.drawRect(r)
            U.text(p, r.adjusted(8 * s, 0, 0, 0), "You", 14 * s, QColor("#7fb2e5"), cond=True)
            U.text(p, r.adjusted(48 * s, 0, 0, 0), self.weapon.name + ("  ◉" if head else ""), 13 * s, U.TEXT,
                   cond=True)
            U.text(p, r.adjusted(0, 0, -8 * s, 0), name, 14 * s, QColor("#e5c07f"), Qt.AlignmentFlag.AlignRight,
                   cond=True)
            p.setOpacity(1.0)
            y += 30 * s
        # bottom left: health; bottom right: ammo
        hb = QRectF(14 * s, h - 58 * s, 170 * s, 44 * s)
        p.fillRect(hb, QColor(0, 0, 0, 120))
        U.text(p, hb.adjusted(12 * s, 0, 0, 0), f"✚ {self.health}", 28 * s,
               U.RED if self.health <= 30 else QColor("#e8e3c8"), weight="bold", cond=True)
        ab = QRectF(w - 200 * s, h - 58 * s, 186 * s, 44 * s)
        p.fillRect(ab, QColor(0, 0, 0, 120))
        ammo = "RELOADING" if self.reload > 0 else f"{self.ammo} / ∞"
        U.text(p, ab.adjusted(0, 0, -12 * s, 0), ammo, (18 if self.reload > 0 else 28) * s, QColor("#e8e3c8"),
               Qt.AlignmentFlag.AlignRight, "bold", True)
        if self.stats.combo >= 3 and self.phase == "live":
            U.text(p, QRectF(0, h - 110 * s, w, 30 * s), f"COMBO ×{self.stats.combo}", 24 * s, U.ORANGE,
                   Qt.AlignmentFlag.AlignHCenter, "bold", True, shadow=True)
        if self.hs_mark > 0:
            c = QColor("#ffd24a")
            c.setAlpha(int(255 * min(1.0, self.hs_mark / 0.3)))
            U.text(p, QRectF(0, h * 0.36, w, 40 * s), "HEADSHOT", 30 * s, c, Qt.AlignmentFlag.AlignHCenter, "bold",
                   True, shadow=True)
        if self.phase == "countdown":
            n = math.ceil(self.clock)
            U.text(p, QRectF(0, h * 0.3, w, 120 * s), str(n) if n > 0 else "GO!", 110 * s, U.TEXT,
                   Qt.AlignmentFlag.AlignHCenter, "bold", True, shadow=True)
            reload_key = self.g.key_label(Qt.Key.Key_R).upper()
            U.text(p, QRectF(0, h * 0.3 + 120 * s, w, 30 * s),
                   f"SHOOT THE BOTS · HEADSHOTS PAY MORE · {reload_key} TO RELOAD",
                   16 * s, U.DIM, Qt.AlignmentFlag.AlignHCenter, cond=True, shadow=True)
        if self.phase == "results":
            self.paint_results(p, w, h, s, hits)

    def paint_results(self, p: QPainter, w: int, h: int, s: float, hits: U.Hits) -> None:
        p.fillRect(QRectF(0, 0, w, h), QColor(0, 0, 0, 150))
        pw, ph = min(560 * s, w * 0.9), min(440 * s, h * 0.9)
        r = QRectF((w - pw) / 2, (h - ph) / 2, pw, ph)
        body = U.panel(p, r, "Round complete" if self.health > 0 else "You were eliminated", s, 245)
        st, res = self.stats, self.result
        acc = st.hits / st.shots * 100 if st.shots else 0.0
        hs = st.headshots / max(st.kills, 1) * 100
        rows = (("Kills", str(st.kills)), ("Headshots", f"{st.headshots}  ({hs:.0f}%)"),
                ("Accuracy", f"{acc:.0f}%"), ("Best combo", f"×{st.best_combo}"),
                ("Bounty earned", D.money(st.earned)), ("Round bonus", D.money(res.get("bonus", 0))),
                ("XP", f"+{res.get('xp', 0)}"))
        y = body.top() + 14 * s
        rh = 30 * s
        for k, v in rows:
            U.text(p, QRectF(body.left() + 30 * s, y, pw * 0.5, rh), k.upper(), 15 * s, U.DIM, cond=True)
            U.text(p, QRectF(body.left(), y, pw - 30 * s, rh), v, 18 * s,
                   U.MONEY if "$" in v else U.TEXT, Qt.AlignmentFlag.AlignRight, "bold", True)
            y += rh
        drop = res.get("drop")
        if drop:
            cr = QRectF(body.left() + 30 * s, y + 4 * s, 70 * s, 50 * s)
            p.drawPixmap(cr.toRect(), A.case_pixmap(drop, int(cr.width()), int(cr.height())))
            U.text(p, QRectF(cr.right() + 10 * s, y + 4 * s, pw, 50 * s), f"CASE DROP: {D.CASES[drop].name}", 18 * s,
                   U.GOLD, weight="bold", cond=True)
        else:
            U.text(p, QRectF(body.left() + 30 * s, y + 4 * s, pw, 50 * s), "No case drop this round", 15 * s, U.FAINT,
                   cond=True)
        bw = (pw - 80 * s) / 3
        by = r.bottom() - 60 * s
        U.button(p, hits, QRectF(r.left() + 30 * s, by, bw, 40 * s), "Play again", self.again, s, "primary", size=17)
        U.button(p, hits, QRectF(r.left() + 40 * s + bw, by, bw, 40 * s), "Controls…", self.g.open_controls, s,
                 "ghost", size=16)
        U.button(p, hits, QRectF(r.right() - 30 * s - bw, by, bw, 40 * s), "Back to menu", self.g.close_arena, s,
                 size=17)

    def again(self) -> None:
        self.new_round()
        self.g.set_arena_cursor(True)


# --- the soldier -----------------------------------------------------------------------------------------------------

OUTFITS = (("#8a7653", "#6e6848", "#4c4a33"), ("#6b6a5e", "#4f5a44", "#3a3f30"),
           ("#9c8a6a", "#7a5f42", "#55402b"), ("#4d4c48", "#5d5547", "#2f2d29"))


def draw_soldier(p: QPainter, x: float, y: float, H: float, aiming: bool = False,  # noqa: N803
                 look: int = 0) -> None:
    """A desert-camo soldier with a balaclava and a rifle, feet at (x, y), ``H`` tall."""
    p.setPen(QPen(QColor(20, 16, 12, 200), max(1.0, H * 0.006)))
    u = H / 100.0
    # legs and boots
    pants, shirt, vest = (QColor(c) for c in OUTFITS[look % len(OUTFITS)])
    p.setBrush(pants)
    p.drawPolygon(_poly((x - 10 * u, y - 50 * u), (x - 1 * u, y - 50 * u), (x - 3 * u, y - 4 * u),
                        (x - 11 * u, y - 4 * u)))
    p.drawPolygon(_poly((x + 1 * u, y - 50 * u), (x + 10 * u, y - 50 * u), (x + 11 * u, y - 4 * u),
                        (x + 3 * u, y - 4 * u)))
    p.setBrush(QColor("#3b2f24"))
    p.drawRect(QRectF(x - 12 * u, y - 5 * u, 10 * u, 5 * u))
    p.drawRect(QRectF(x + 2 * u, y - 5 * u, 10 * u, 5 * u))
    # torso: shirt and vest
    p.setBrush(shirt)
    p.drawRoundedRect(QRectF(x - 15 * u, y - 83 * u, 30 * u, 36 * u), 5 * u, 5 * u)
    p.setBrush(vest)
    p.drawRoundedRect(QRectF(x - 11 * u, y - 79 * u, 22 * u, 26 * u), 3 * u, 3 * u)
    p.setBrush(QColor("#3c3a28"))
    for i in range(3):
        p.drawRect(QRectF(x - 10 * u + i * 7 * u, y - 66 * u, 6 * u, 8 * u))
    p.setBrush(QColor("#5a4a32"))
    p.drawRect(QRectF(x - 14 * u, y - 52 * u, 28 * u, 4 * u))  # belt
    # head: balaclava with an eye slit
    hx, hy, hr = x, y - 90 * u, 8.5 * u
    p.setBrush(QColor("#5a4a32"))
    p.drawRect(QRectF(x - 4 * u, y - 85 * u, 8 * u, 4 * u))
    p.setBrush(QColor("#2b2a27"))
    p.drawEllipse(QPointF(hx, hy), hr, hr * 1.08)
    p.setBrush(QColor("#c99b79"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(hx - 6 * u, hy - 3 * u, 12 * u, 4.2 * u), 2 * u, 2 * u)
    p.setBrush(QColor("#1a1410"))
    p.drawEllipse(QPointF(hx - 2.8 * u, hy - 0.9 * u), 1.1 * u, 1.1 * u)
    p.drawEllipse(QPointF(hx + 2.8 * u, hy - 0.9 * u), 1.1 * u, 1.1 * u)
    # arms and rifle (towards the player when aiming)
    p.setPen(QPen(QColor(20, 16, 12, 200), max(1.0, H * 0.006)))
    p.setBrush(shirt)
    if aiming:
        p.drawPolygon(_poly((x - 14 * u, y - 80 * u), (x - 6 * u, y - 80 * u), (x - 20 * u, y - 64 * u),
                            (x - 27 * u, y - 68 * u)))
        p.drawPolygon(_poly((x + 14 * u, y - 80 * u), (x + 8 * u, y - 72 * u), (x - 8 * u, y - 66 * u),
                            (x - 6 * u, y - 72 * u)))
        p.setBrush(QColor("#1d1d1f"))
        p.drawPolygon(_poly((x - 30 * u, y - 70 * u), (x + 6 * u, y - 70 * u), (x + 6 * u, y - 63 * u),
                            (x - 30 * u, y - 64 * u)))
        p.drawEllipse(QPointF(x - 28 * u, y - 67 * u), 3.5 * u, 3.5 * u)
    else:
        p.drawPolygon(_poly((x - 15 * u, y - 80 * u), (x - 9 * u, y - 80 * u), (x - 6 * u, y - 60 * u),
                            (x - 12 * u, y - 58 * u)))
        p.drawPolygon(_poly((x + 15 * u, y - 80 * u), (x + 9 * u, y - 80 * u), (x + 4 * u, y - 60 * u),
                            (x + 10 * u, y - 58 * u)))
        p.setBrush(QColor("#1d1d1f"))
        p.drawPolygon(_poly((x - 18 * u, y - 56 * u), (x + 20 * u, y - 72 * u), (x + 22 * u, y - 68 * u),
                            (x - 16 * u, y - 52 * u)))


# --- the scene -------------------------------------------------------------------------------------------------------

_grit: QImage | None = None


def _grit_brush() -> QBrush:
    global _grit
    if _grit is None:
        rng = np.random.default_rng(5)
        n = rng.random((128, 128))
        a = (np.abs(n - 0.5) * 2) ** 3 * 70
        arr = np.zeros((128, 128, 4), np.uint8)
        dark = n < 0.5
        arr[..., 0] = np.where(dark, 60, 255)
        arr[..., 1] = np.where(dark, 45, 245)
        arr[..., 2] = np.where(dark, 30, 220)
        arr[..., 3] = a.astype(np.uint8)
        _grit = QImage(arr.data, 128, 128, 512, QImage.Format.Format_RGBA8888).copy()
    return QBrush(_grit)


def _wall(p: QPainter, poly: QPolygonF, top: QColor, bottom: QColor) -> None:
    """A sandstone wall: a vertical gradient and grit."""
    path = QPainterPath()
    path.addPolygon(poly)
    path.closeSubpath()
    p.save()
    p.setClipPath(path)
    br = poly.boundingRect()
    g = QLinearGradient(br.topLeft(), br.bottomLeft())
    g.setColorAt(0.0, top)
    g.setColorAt(1.0, bottom)
    p.fillRect(br, g)
    p.fillRect(br, _grit_brush())
    p.restore()


def scene_pixmap(w: int, h: int) -> QPixmap:
    """The courtyard (no bots) drawn for a w x h widget."""
    pm = A.new_pixmap(max(w, 1), max(h, 1))
    pm.fill(QColor("#8fc6ee"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = max(w / VW, h / VH)
    p.translate((w - VW * s) / 2, (h - VH * s) / 2)
    p.scale(s, s)
    paint_scene(p)
    p.end()
    return pm


def _courses(p: QPainter, clip: QPainterPath, tl: QPointF, tr: QPointF, bl: QPointF, br: QPointF, rows: int,
             cols: int, color: QColor) -> None:
    """Mortar lines of a perspective wall quad (tl-tr top edge, bl-br bottom edge)."""
    p.save()
    p.setClipPath(clip)
    p.setPen(QPen(color, 1.6))
    for i in range(1, rows):
        f = i / rows
        a = QPointF(tl.x() + (bl.x() - tl.x()) * f, tl.y() + (bl.y() - tl.y()) * f)
        b = QPointF(tr.x() + (br.x() - tr.x()) * f, tr.y() + (br.y() - tr.y()) * f)
        p.drawLine(a, b)
        f0 = (i - 1) / rows
        for j in range(cols):
            g = (j + 0.5 * (i % 2)) / cols
            x = tl.x() + (tr.x() - tl.x()) * g
            ya = tl.y() + (tr.y() - tl.y()) * g
            yb = bl.y() + (br.y() - bl.y()) * g
            p.drawLine(QPointF(x, ya + (yb - ya) * f0), QPointF(x, ya + (yb - ya) * f))
    p.restore()


def _crate(p: QPainter, x0: float, y0: float, x1: float, y1: float, depth: float) -> None:
    """A wooden crate: front face with planks and an X brace, a lighter top face, and its shadow."""
    w = x1 - x0
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(90, 60, 30, 70))
    p.drawPolygon(_poly((x0 + w * 0.1, y1), (x1, y1), (x1 + w * 0.35, y1 + depth * 0.5), (x0 + w * 0.3,
                                                                                        y1 + depth * 0.5)))
    top = _poly((x0, y0), (x1, y0), (x1 - depth * 0.6, y0 - depth), (x0 + depth * 0.6, y0 - depth))
    p.setBrush(QColor("#c8a066"))
    p.setPen(QPen(QColor("#5a3d1e"), 2))
    p.drawPolygon(top)
    g = QLinearGradient(QPointF(0, y0), QPointF(0, y1))
    g.setColorAt(0.0, QColor("#b5874d"))
    g.setColorAt(1.0, QColor("#8a6236"))
    p.setBrush(g)
    face = QRectF(x0, y0, w, y1 - y0)
    p.drawRect(face)
    p.save()
    p.setClipRect(face)
    p.fillRect(face, _grit_brush())
    p.setPen(QPen(QColor("#6b4826"), 1.5))
    planks = 5
    for i in range(1, planks):
        yy = y0 + (y1 - y0) * i / planks
        p.drawLine(QPointF(x0, yy), QPointF(x1, yy))
    p.restore()
    fr = max(w * 0.07, 6)
    p.setPen(QPen(QColor("#5a3d1e"), 2))
    p.setBrush(QColor("#9c7040"))
    for r in (QRectF(x0, y0, w, fr), QRectF(x0, y1 - fr, w, fr), QRectF(x0, y0, fr, y1 - y0),
              QRectF(x1 - fr, y0, fr, y1 - y0)):
        p.drawRect(r)
    brace = QPainterPath(QPointF(x0 + fr, y0 + fr))
    brace.lineTo(x0 + fr * 2, y0 + fr)
    brace.lineTo(x1 - fr, y1 - fr * 2)
    brace.lineTo(x1 - fr, y1 - fr)
    brace.lineTo(x1 - fr * 2, y1 - fr)
    brace.lineTo(x0 + fr, y0 + fr * 2)
    brace.closeSubpath()
    p.drawPath(brace)


def paint_scene(p: QPainter) -> None:
    """The courtyard in virtual coordinates."""
    # sky and sun
    g = QLinearGradient(0, 0, 0, 520)
    g.setColorAt(0.0, QColor("#3f8ed8"))
    g.setColorAt(0.7, QColor("#8fc6ee"))
    g.setColorAt(1.0, QColor("#d8ecf6"))
    p.fillRect(QRectF(-400, -400, VW + 800, 920), g)
    sun = QRadialGradient(QPointF(330, 70), 260)
    sun.setColorAt(0.0, QColor(255, 255, 240, 255))
    sun.setColorAt(0.08, QColor(255, 250, 220, 230))
    sun.setColorAt(0.3, QColor(255, 240, 200, 70))
    sun.setColorAt(1.0, QColor(255, 240, 200, 0))
    p.fillRect(QRectF(0, -200, 700, 560), sun)
    # distant rooftops
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#d9c3a0"))
    p.drawPolygon(_poly((380, 520), (380, 230), (470, 230), (470, 200), (560, 200), (560, 240), (1250, 240),
                        (1250, 210), (1330, 210), (1330, 520)))
    # ground
    gg = QLinearGradient(0, 520, 0, 900)
    gg.setColorAt(0.0, QColor("#dcc394"))
    gg.setColorAt(1.0, QColor("#c49a5e"))
    p.fillRect(QRectF(-400, 520, VW + 800, 800), gg)
    p.fillRect(QRectF(-400, 520, VW + 800, 800), _grit_brush())
    p.setPen(QPen(QColor(150, 115, 70, 60), 3))
    for k in range(-6, 7):
        p.drawLine(QPointF(800 + k * 40, 520), QPointF(800 + k * 260, 900))
    # far building with the arch, a window and a roof lip
    _wall(p, _poly((430, 180), (1170, 180), (1170, 520), (430, 520)), QColor("#e2c79a"), QColor("#c9a874"))
    fb = _path((430, 180), (1170, 180), (1170, 520), (430, 520))
    _courses(p, fb, QPointF(430, 180), QPointF(1170, 180), QPointF(430, 520), QPointF(1170, 520), 9, 14,
             QColor(150, 115, 72, 90))
    p.setPen(QPen(QColor("#9c7a4c"), 2))
    p.setBrush(QColor("#ecd6ae"))
    p.drawRect(QRectF(420, 168, 760, 16))
    arch = _arch_path()
    ag = QLinearGradient(0, 230, 0, 520)
    ag.setColorAt(0.0, QColor("#3a2c1e"))
    ag.setColorAt(1.0, QColor("#6b5338"))
    p.setBrush(ag)
    p.setPen(QPen(QColor("#9c7a4c"), 6))
    p.drawPath(arch)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 230, 180, 40))  # light falling into the arch
    p.drawPolygon(_poly((700, 520), (900, 520), (860, 470), (735, 470)))
    p.setBrush(QColor("#3a2c1e"))
    p.setPen(QPen(QColor("#9c7a4c"), 5))
    p.drawRect(QRectF(500, 250, 90, 80))
    p.setPen(QPen(QColor("#7a5530"), 4))
    p.setBrush(QColor("#8a5f34"))
    p.drawRect(QRectF(470, 248, 28, 84))  # an open shutter
    p.drawRect(QRectF(1000, 280, 110, 120))  # a wooden door, closed
    p.setPen(QPen(QColor("#6b4826"), 2))
    for i in range(1, 5):
        p.drawLine(QPointF(1000 + i * 22, 282), QPointF(1000 + i * 22, 398))
    # left wall (in shade) with a doorway
    lw = _poly((0, 40), (430, 170), (430, 520), (0, 900))
    _wall(p, lw, QColor("#b4935f"), QColor("#8e7146"))
    _courses(p, _path((0, 40), (430, 170), (430, 520), (0, 900)), QPointF(0, 40), QPointF(430, 170),
             QPointF(0, 900), QPointF(430, 520), 12, 6, QColor(90, 68, 40, 110))
    p.setPen(QPen(QColor("#6b5434"), 5))
    dg = QLinearGradient(150, 0, 250, 0)
    dg.setColorAt(0.0, QColor("#1f170f"))
    dg.setColorAt(1.0, QColor("#4a3824"))
    p.setBrush(dg)
    p.drawPolygon(_poly((150, 392), (250, 369), (250, 679), (150, 767)))
    # right wall (in the sun) with a window
    rw = _poly((1170, 170), (1600, 30), (1600, 900), (1170, 520))
    _wall(p, rw, QColor("#f0d9ae"), QColor("#d4b27c"))
    _courses(p, _path((1170, 170), (1600, 30), (1600, 900), (1170, 520)), QPointF(1170, 170), QPointF(1600, 30),
             QPointF(1170, 520), QPointF(1600, 900), 12, 6, QColor(160, 125, 80, 100))
    p.setPen(QPen(QColor("#9c7a4c"), 5))
    p.setBrush(QColor("#3a2c1e"))
    p.drawPolygon(_poly((1300, 280), (1420, 285), (1420, 415), (1300, 381)))
    p.setPen(QPen(QColor("#b08a58"), 6))
    p.drawLine(QPointF(1290, 386), QPointF(1430, 422))  # sill
    # crates (far to near) and a few props
    _crate(p, 960, 450, 1080, 520, 10)
    _crate(p, 520, 580, 700, 730, 20)
    _crate(p, 1150, 640, 1420, 860, 28)
    _crate(p, 220, 790, 380, 905, 18)
    p.setPen(Qt.PenStyle.NoPen)
    shadow = QLinearGradient(0, 520, 0, 560)
    shadow.setColorAt(0.0, QColor(80, 60, 30, 70))
    shadow.setColorAt(1.0, QColor(80, 60, 30, 0))
    p.fillRect(QRectF(430, 520, 740, 40), shadow)
    # a hanging wire and a lamp for detail
    p.setPen(QPen(QColor(40, 30, 20, 180), 2))
    p.setBrush(Qt.BrushStyle.NoBrush)
    wire = QPainterPath(QPointF(430, 200))
    wire.quadTo(800, 260, 1170, 190)
    p.drawPath(wire)


def scene_preview(w: int, h: int) -> QPixmap:
    """The courtyard with a couple of bots, for the menu's Aim Arena card."""
    pm = scene_pixmap(w, h)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = max(w / VW, h / VH)
    p.translate((w - VW * s) / 2, (h - VH * s) / 2)
    p.scale(s, s)
    for i in (0, 3):
        sp = SPOTS[i]
        p.save()
        p.setClipPath(sp.clip)
        draw_soldier(p, sp.x, sp.y, 300 * sp.scale, aiming=True, look=i)
        p.restore()
    p.end()
    return pm

