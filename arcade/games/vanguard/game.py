"""VANGUARD: the arcade's five-against-five hero team shooter. This module is the game widget: menus over a 3D
showroom, the loading screen, the match (the 3D view drawn by the graphics card at the window's own resolution, the
HUD painted over it), input, sound, saving, and the card on the arcade's home screen."""

from __future__ import annotations

import math
import random
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QKeySequence, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF

from app.arcade.base import ArcadeGame

from . import hud, maps, scene
from . import models as M
from .data import ALLY_RGB, DIFFICULTIES, ENEMY_RGB, HERO_ORDER, HEROES, MODES
from .gfx import Renderer
from .menus import CROSS, Menu
from .sim import Controls, Match
from .sounds import build as build_sounds
from .world import World

K = Qt.Key
LOOK = 0.0022  # radians per pixel of mouse at the arcade's sensitivity 1.0
KEY_TURN = 2.4  # radians per second with the arrow keys at turn speed 1.0
TIPS = ("Stand on the objective: eliminations only matter because they clear it.",
        "Your ultimate charges from damage and from healing. Use it when the teams meet, not after.",
        "Hurt? Health packs sit on the side routes, and your spawn room heals you fast.",
        "Supports glow green when they heal you. Stay where they can see you.",
        "A barrier stops their shots, not yours.",
        "You can change hero in your spawn room: press H.",
        "Red outline: enemy. Blue outline: your team. A red shape through a wall has been revealed.")


def default_settings() -> dict:
    return {"fov": 95, "shadows": 2, "outlines": True, "cross": 0, "names": True}


def default_cfg() -> dict:
    return {"mode": "control", "map": "marigold", "side": "attack", "difficulty": "Regular", "hero": "dozer"}


def default_stats() -> dict:
    return {"matches": 0, "wins": 0, "losses": 0, "elims": 0, "final": 0, "assists": 0, "deaths": 0, "damage": 0,
            "healing": 0, "ults": 0, "best_elims": 0, "time": 0.0,
            "modes": {"control": [0, 0], "escort": [0, 0]},
            "heroes": {h: {"matches": 0, "wins": 0, "time": 0.0, "elims": 0, "deaths": 0} for h in HERO_ORDER}}


class VanguardGame(ArcadeGame):
    game_id = "vanguard"
    title = "VANGUARD"
    subtitle = "5 v 5 hero shooter"
    description = ("Pick one of seven heroes - tanks, damage, support - and fight with four bots against five. Hold the "
                   "bandstand of MARIGOLD MARKET in Control, or walk the tank car down SUNDOWN SPUR in Escort. Every "
                   "hero has a weapon, two abilities and an ultimate that charges from damage and healing.")
    controls = (("WASD", "Move"), ("Mouse", "Look / aim"), ("LMB", "Fire"), ("RMB", "Steady the sight (Flint)"),
                ("Shift", "Ability 1"), ("E", "Ability 2"), ("Q", "Ultimate"), ("Space", "Jump"), ("R", "Reload"),
                ("V", "Melee"), ("H", "Change hero (in spawn)"), ("Tab", "Scoreboard"), ("← →", "Turn"),
                ("Esc", "Menu (Controls...)"))
    accent = "#3fb0ff"
    uses_mouse_look = False  # captured while playing only (handled in tick)
    uses_key_turn = True
    bindings = (("Move forward", K.Key_W), ("Move back", K.Key_S), ("Strafe left", K.Key_A),
                ("Strafe right", K.Key_D), ("Turn left", K.Key_Left), ("Turn right", K.Key_Right),
                ("Jump", K.Key_Space), ("Ability 1", K.Key_Shift), ("Ability 2", K.Key_E), ("Ultimate", K.Key_Q),
                ("Reload", K.Key_R), ("Melee", K.Key_V), ("Change hero", K.Key_H), ("Scoreboard", K.Key_Tab))

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        build_sounds(self.sfx)
        self.settings = default_settings()
        self.cfg = default_cfg()
        self.stats = default_stats()
        self.screen = "menu"  # menu | loading | match
        self.screen_t = 0.0
        self.menu = Menu(self)
        self.renderer = Renderer()
        self.match: Match | None = None
        self.showroom = maps.build("showroom")
        self.autopilot = False  # tests: a bot plays the player's hero too
        self.seed: int | None = None
        self.picker = False
        self.picker_hover: str | None = None
        self.show_scores = False
        self.fire = self.alt = False
        self.sway = [0.0, 0.0]
        self.paint_ms = 0.0
        self._src: dict = {}  # hero id -> square render
        self._round: dict = {}
        self._counted = False
        self._hero_time: dict = {}
        self._tip = TIPS[0]
        self.apply_settings()

    # --- saving -----------------------------------------------------------------------------------------------
    def save_state(self) -> dict:
        return {"version": 1, "settings": dict(self.settings), "cfg": dict(self.cfg), "stats": self.stats}

    def load_state(self, data: dict) -> None:
        data = data if isinstance(data, dict) else {}
        sett = data.get("settings") if isinstance(data.get("settings"), dict) else {}
        for k, v in default_settings().items():
            val = sett.get(k, v)
            try:
                self.settings[k] = bool(val) if isinstance(v, bool) else int(val)
            except (TypeError, ValueError):
                self.settings[k] = v
        self.settings["fov"] = min(max(self.settings["fov"], 70), 110)
        self.settings["shadows"] = min(max(self.settings["shadows"], 0), 2)
        self.settings["cross"] = min(max(self.settings["cross"], 0), len(CROSS) - 1)
        cfg = data.get("cfg") if isinstance(data.get("cfg"), dict) else {}
        d = default_cfg()
        self.cfg = {
            "mode": cfg.get("mode") if cfg.get("mode") in MODES else d["mode"],
            "map": d["map"], "side": cfg.get("side") if cfg.get("side") in ("attack", "defend") else d["side"],
            "difficulty": cfg.get("difficulty") if cfg.get("difficulty") in DIFFICULTIES else d["difficulty"],
            "hero": cfg.get("hero") if cfg.get("hero") in HEROES else d["hero"],
        }
        ok = maps.MAPS_BY_MODE[self.cfg["mode"]]
        self.cfg["map"] = cfg.get("map") if cfg.get("map") in ok else ok[0]
        st = data.get("stats") if isinstance(data.get("stats"), dict) else {}
        fresh = default_stats()

        def num(src: dict, key: str, default):  # noqa: ANN001, ANN202
            try:
                v = type(default)(src.get(key, default))
            except (TypeError, ValueError):
                return default
            return v if v >= 0 and v == v else default

        for k, v in fresh.items():
            if isinstance(v, int | float):
                fresh[k] = num(st, k, v)
        modes = st.get("modes") if isinstance(st.get("modes"), dict) else {}
        for mname in fresh["modes"]:
            wl = modes.get(mname)
            if isinstance(wl, list) and len(wl) == 2 and all(isinstance(x, int) and x >= 0 for x in wl):
                fresh["modes"][mname] = list(wl)
        heroes = st.get("heroes") if isinstance(st.get("heroes"), dict) else {}
        for hid, hs in fresh["heroes"].items():
            src = heroes.get(hid) if isinstance(heroes.get(hid), dict) else {}
            for k, v in hs.items():
                hs[k] = num(src, k, v)
        self.stats = fresh
        self.apply_settings()

    @classmethod
    def summary(cls, data: dict) -> str:
        try:
            st = (data or {}).get("stats") or {}
            n, wins = int(st.get("matches", 0)), int(st.get("wins", 0))
        except (TypeError, ValueError, AttributeError):
            return "Continue"
        if not n:
            return "New recruit"
        return f"{wins} wins of {n} matches · {int(st.get('elims', 0))} eliminations"

    def apply_settings(self) -> None:
        r = self.renderer
        if r.shadow_quality != self.settings["shadows"]:
            r.shadow_quality = self.settings["shadows"]
            r._shadow_dirty = True
        r.outlines = bool(self.settings["outlines"])

    # --- what the menus and the HUD ask for -----------------------------------------------------------------------
    def cross_rgb(self) -> tuple:
        return CROSS[self.settings["cross"]][1]

    def key_name(self, action: str) -> str:
        """The key now bound to one of ``bindings``' actions, as the HUD shows it."""
        default = int(dict(self.bindings)[action])
        key = default
        for pressed, known in self._remap.items():
            if int(known) == default:
                key = int(pressed)
        names = {int(K.Key_Shift): "SHIFT", int(K.Key_Space): "SPACE", int(K.Key_Control): "CTRL",
                 int(K.Key_Tab): "TAB", int(K.Key_Alt): "ALT"}
        return names.get(key) or QKeySequence(key).toString().upper() or "?"

    def portrait(self, hid: str, ally: bool = True, grey: bool = False, size: int = 64) -> QImage | None:
        src = self._src.get(hid)
        if src is None:
            return None
        key = (hid, ally, grey, size)
        img = self._round.get(key)
        if img is None:
            img = self._round[key] = hud.round_portrait(src, size * 2, ALLY_RGB if ally else ENEMY_RGB, grey)
        return img

    def _make_portraits(self) -> None:
        """A head-and-shoulders render of every hero, on the showroom stage (once)."""
        if self._src:
            return
        r = self.renderer
        r.set_map(self.showroom)
        for hid, hero in HEROES.items():
            model = M.hero_model(hid, hero.scale)
            ang, bob = M.actor_pose(0.0, 0.0, 0.0, 0.0, False, 0.0, 0.5)
            rows = model.rows((12.0, 0.3, 12.0), M.yaw_basis(0.45), ang, bob, ALLY_RGB, 0.0)
            r.set_camera(12.0 + 1.75 * hero.scale, 12.0, 0.3 + hero.height * 0.9, math.pi, -0.06)
            was = r.outlines
            img = r.render(192, 192, math.radians(40), 0.0, 1.0, rows, vignette=0.0)
            r.outlines = was
            self._src[hid] = img.copy()

    # --- matches --------------------------------------------------------------------------------------------------
    def start_match(self) -> None:
        self.screen = "loading"
        self.screen_t = 0.0
        self.match = None
        self.menu.close()
        self.picker = False
        self._tip = random.choice(TIPS)

    def _build_match(self) -> None:
        cfg = self.cfg
        try:
            self._make_portraits()
        except RuntimeError:
            pass
        md = maps.build(cfg["map"])
        world = World(md)
        self.match = m = Match(md, world, md.mode, cfg["hero"], cfg["difficulty"], cfg["side"] == "attack",
                               self.seed, self.autopilot)
        # every route the bots will ask for, now, so no frame of the match has to work one out
        if md.mode == "control":
            ox, oy, oz = m.objective_pos()
            world.field(("point",), ox, oy, oz)
        else:
            for k in range(int(m.cart_len / 4.0) + 2):
                x, y, _yaw = maps.track_at(md.track, k * 4.0)
                world.field(("cart", k), x, y, 0.0)
        for i, pk in enumerate(m.packs):
            world.field(("pack", i), pk["x"], pk["y"], pk["z"])
        self.renderer.set_map(md)
        self.renderer.warm_up()
        self._counted = False
        self._hero_time = {}
        self.fire = self.alt = False
        self.keys.clear()
        self.screen = "match"
        self.screen_t = 0.0
        mine = (md.mode == "control") or cfg["side"] == "attack"
        m.banner(md.name, "Take the point" if md.mode == "control" else
                 ("Escort the cart" if mine else "Stop the cart"), (255, 255, 255), 4.0)

    def leave_match(self) -> None:
        """Back to the main menu; an unfinished match counts as a loss."""
        m = self.match
        if m is not None and not self._counted and m.time > 20.0:
            m.winner = 1 - m.player.team
            self._record()
        self.match = None
        self.picker = False
        self.show_scores = False
        self.set_mouse_captured(False)
        self.screen = "menu"
        self.menu.close()
        self.menu.open("main")
        self.sfx.play("ui_back")

    def _record(self) -> None:
        m = self.match
        self._counted = True
        res = m.result()
        st = self.stats
        won = bool(res["won"])
        st["matches"] += 1
        st["wins" if won else "losses"] += 1
        st["modes"][m.mode][0 if won else 1] += 1
        for k in ("elims", "final", "assists", "deaths", "damage", "healing", "ults"):
            st[k] += int(res[k])
        st["best_elims"] = max(st["best_elims"], int(res["elims"]))
        st["time"] = round(st["time"] + m.time, 1)
        for hid, t in self._hero_time.items():
            st["heroes"][hid]["time"] = round(st["heroes"][hid]["time"] + t, 1)
        if self._hero_time:
            main = max(self._hero_time, key=self._hero_time.get)
            hs = st["heroes"][main]
            hs["matches"] += 1
            hs["wins"] += 1 if won else 0
            hs["elims"] += int(res["elims"])
            hs["deaths"] += int(res["deaths"])
        self.progress_changed.emit()

    def can_swap(self) -> bool:
        m = self.match
        return m is not None and m.phase != "over" and (not m.player.alive or m.in_own_room(m.player))

    def open_picker(self) -> None:
        if self.can_swap():
            self.picker = True
            self.fire = self.alt = False
            self.set_mouse_captured(False)
        else:
            self.sfx.play("ui_back")

    def _pick(self, hid: str) -> None:
        m = self.match
        if m is not None and self.can_swap():
            m.swap_hero(m.player, hid)
            self.cfg["hero"] = hid
            self.sfx.play("ui_ok")
        self.picker = False

    def open_controls(self) -> None:
        """The arcade's Controls panel (it pauses the game and resumes it after)."""
        self.fire = self.alt = False
        self.keys.clear()
        self.set_mouse_captured(False)
        self.controls_requested.emit()

    def stop(self) -> None:
        super().stop()
        self.renderer.close()
        self._src.clear()
        self._round.clear()

    def on_pause(self) -> None:
        self.fire = self.alt = False

    # --- input ------------------------------------------------------------------------------------------------
    def playing(self) -> bool:
        m = self.match
        return self.screen == "match" and m is not None and not self.menu.paused and not self.picker \
            and m.phase != "over"

    def handle_escape(self) -> bool:
        if self.picker:
            self.picker = False
            return True
        if self.menu.active:
            return self.menu.escape()
        if self.screen == "match":
            if self.match is not None and self.match.phase == "over":
                self._finish()
            else:
                self.menu.open_pause()
                self.fire = self.alt = False
                self.sfx.play("ui_back")
            return True
        return self.screen == "loading"

    def _finish(self) -> None:
        if self.match is not None and not self._counted:
            self._record()
        self.match = None
        self.screen = "menu"
        self.menu.close()
        self.menu.open("play")
        self.set_mouse_captured(False)

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001
        if self.picker:
            if K.Key_1 <= key <= K.Key_7:
                self._pick(HERO_ORDER[key - K.Key_1])
            elif key in (K.Key_H, K.Key_Return, K.Key_Enter):
                self.picker = False
            return
        if self.menu.active:
            self.menu.key(key)
            return
        if self.screen != "match" or self.match is None:
            return
        m = self.match
        if m.phase == "over":
            if key in (K.Key_Return, K.Key_Enter, K.Key_Space) and m.over_t > 2.0:
                self._finish()
            return
        if key == K.Key_H:
            self.open_picker()
        elif key == K.Key_Tab:
            self.show_scores = True

    def key_released(self, key: int, event) -> None:  # noqa: ANN001
        if key == K.Key_Tab:
            self.show_scores = False

    def mouse_pressed(self, button, pos) -> None:  # noqa: ANN001
        if self.picker:
            for hid, r in hud.hero_cards(self.width(), self.height()):
                if r.contains(pos):
                    self._pick(hid)
                    return
            return
        if self.menu.active:
            if button == Qt.MouseButton.LeftButton:
                self.menu.click(pos)
            return
        if self.screen == "match" and self.match is not None:
            if self.match.phase == "over":
                if self.match.over_t > 2.0:
                    self._finish()
                return
            if button == Qt.MouseButton.LeftButton:
                self.fire = True
            elif button == Qt.MouseButton.RightButton:
                self.alt = True
            if not self.mouse_captured and self.isActiveWindow():
                self.set_mouse_captured(True)

    def mouse_released(self, button, pos) -> None:  # noqa: ANN001
        if button == Qt.MouseButton.LeftButton:
            self.fire = False
        elif button == Qt.MouseButton.RightButton:
            self.alt = False

    def mouse_moved(self, pos) -> None:  # noqa: ANN001
        if self.picker:
            self.picker_hover = None
            for hid, r in hud.hero_cards(self.width(), self.height()):
                if r.contains(pos):
                    self.picker_hover = hid
        elif self.menu.active:
            self.menu.hover(pos)

    def wheel(self, steps: int) -> None:
        if self.menu.active:
            self.menu.wheel(steps)

    def _drive(self, dt: float) -> None:
        """The player's keys and mouse into their hero's controls."""
        m = self.match
        p = m.player
        dx, dy = self.take_mouse_delta()  # already times the arcade's sensitivity, y flipped when inverted
        if p.brain is not None:
            return
        c = p.c
        if not self.playing() or not p.alive:
            p.c = Controls()
            return
        keys = self.keys
        c.mf = (1.0 if K.Key_W in keys or K.Key_Up in keys else 0.0) - \
            (1.0 if K.Key_S in keys or K.Key_Down in keys else 0.0)
        c.ms = (1.0 if K.Key_D in keys else 0.0) - (1.0 if K.Key_A in keys else 0.0)
        c.jump = K.Key_Space in keys
        c.a1, c.a2, c.ult = K.Key_Shift in keys, K.Key_E in keys, K.Key_Q in keys
        c.reload, c.melee = K.Key_R in keys, K.Key_V in keys
        c.fire, c.alt = self.fire, self.alt
        zoom = 1.0 - 0.5 * p.zoom
        turn = (1.0 if K.Key_Right in keys else 0.0) - (1.0 if K.Key_Left in keys else 0.0)
        p.yaw = (p.yaw + dx * LOOK * zoom + turn * KEY_TURN * self.key_turn_speed * dt * zoom) % (2 * math.pi)
        p.pitch = max(min(p.pitch - dy * LOOK * zoom, 1.45), -1.45)
        self.sway[0] += (dx * 0.02 - self.sway[0]) * min(dt * 8, 1)
        self.sway[1] += (dy * 0.02 - self.sway[1]) * min(dt * 8, 1)
        self.sway = [max(min(v, 1.5), -1.5) for v in self.sway]

    def _play_sounds(self) -> None:
        m = self.match
        p = m.player
        done: dict = {}
        for name, x, y, vol in m.sounds:
            if done.get(name, 0) >= 2:
                continue
            if x is not None:
                d = math.hypot(x - p.x, y - p.y)
                vol *= max(1.0 - d / 55.0, 0.0) ** 1.6
                if vol < 0.03:
                    continue
            done[name] = done.get(name, 0) + 1
            self.sfx.play(name, min(vol, 1.2))
        m.sounds.clear()

    def tick(self, dt: float) -> None:
        self.screen_t += dt
        want = self.playing() and self.isActiveWindow()
        if want != self.mouse_captured and (not want or self.isVisible()):
            self.set_mouse_captured(want)
        if self.screen == "loading":
            if self.screen_t > 0.08:
                self._build_match()
            return
        if self.screen != "match" or self.match is None:
            return
        m = self.match
        if self.menu.paused:
            self.take_mouse_delta()
            return
        if self.picker and not self.can_swap():
            self.picker = False
        self._drive(dt)
        before = m.phase
        if m.phase == "prep" and int(m.phase_t) != int(m.phase_t - dt) and m.phase_t < 3.5:
            self.sfx.play("tick")
        m.update(dt)
        self._play_sounds()
        if m.phase != "over":
            hid = m.player.hero.id
            self._hero_time[hid] = self._hero_time.get(hid, 0.0) + dt
        elif before != "over" or not self._counted:
            self.picker = False
            self.show_scores = False
            if not self._counted:
                self._record()

    # --- painting -----------------------------------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        t0 = time.perf_counter()
        p = QPainter(self)
        w, h = self.width(), self.height()
        try:
            if self.screen == "match" and self.match is not None:
                self._paint_match(p, w, h)
                if self.menu.paused:
                    self.menu.paint(p, w, h, self.time)
            elif self.screen == "loading":
                self.menu.paint_loading(p, w, h, self.screen_t, self.cfg["map"], self.cfg["mode"], self._tip)
            else:
                self._paint_showroom(p, w, h)
                self.menu.paint(p, w, h, self.time)
            if self.paused:
                p.fillRect(self.rect(), QColor(0, 0, 0, 90))
        finally:
            p.end()
        self.paint_ms = (time.perf_counter() - t0) * 1000

    def _dpr(self) -> float:
        return float(self.devicePixelRatioF()) if self.isVisible() else 1.0

    def _paint_showroom(self, p: QPainter, w: int, h: int) -> None:
        """The stage behind the menus: the whole line-up, or the hero being looked at."""
        r = self.renderer
        try:
            self._make_portraits()
            r.set_map(self.showroom)
            t = self.time
            feat = self.menu.featured()
            rows = []
            if feat is None:
                n = len(HERO_ORDER)
                for i, hid in enumerate(HERO_ORDER):
                    hero = HEROES[hid]
                    k = i - (n - 1) / 2
                    x, y = 12.0 - abs(k) * 0.28, 12.0 - k * 0.96
                    ang, bob = M.actor_pose(0.0, 0.0, 0.05 * math.sin(t * 0.7 + i), 0.0, False, 0.0, t + i * 1.7)
                    rows.append(M.hero_model(hid, hero.scale).rows((x, 0.3, y), M.yaw_basis(k * 0.1), ang, bob,
                                                                 ALLY_RGB, 0.0))
                cam = (12.0 + 9.4, 12.0 + 0.15 * math.sin(t * 0.2), 1.9, math.pi - 0.2, -0.08)
                fov = math.radians(62)
            else:
                hero = HEROES[feat]
                ang, bob = M.actor_pose(0.0, 0.0, 0.0, 0.0, False, 0.0, t)
                rows.append(M.hero_model(feat, hero.scale).rows((12.0, 0.3, 12.0), M.yaw_basis(0.5 * math.sin(t * 0.5)),
                                                               ang, bob, ALLY_RGB, 0.0))
                side = 0.12 if self.menu.page == "heroes" else -0.31
                cam = (12.0 + 6.4, 12.0, 1.55, math.pi + side, -0.05)
                fov = math.radians(56)
            r.set_camera(*cam)
            img = r.render(w, h, fov, t, self._dpr(), np.concatenate(rows), vignette=0.7)
            p.drawImage(QRectF(0, 0, w, h), img)
        except RuntimeError:
            g = QLinearGradient(0, 0, 0, h)
            g.setColorAt(0, QColor(24, 36, 80))
            g.setColorAt(1, QColor(8, 10, 22))
            p.fillRect(QRectF(0, 0, w, h), g)

    def _paint_match(self, p: QPainter, w: int, h: int) -> None:
        m = self.match
        me = m.player
        r = self.renderer
        fov = math.radians(self.settings["fov"])
        first = me.alive and m.phase != "over"
        if first:
            fov = 2 * math.atan(math.tan(fov / 2) * (1.0 - 0.5 * me.zoom))
            r.set_camera(me.x, me.y, me.view_z + me.eye, me.yaw, me.pitch)
        else:  # looking down at where they fell (or, at the end, at the objective)
            fx, fy, fz = (me.x, me.y, me.z) if m.phase != "over" else m.objective_pos()
            q = me.yaw + math.pi + (0.3 * m.over_t if m.phase == "over" else 0.25 * me.death_t)
            d = 5.5
            cx, cy = fx + math.cos(q) * d, fy + math.sin(q) * d
            cz = fz + 4.5
            wd = m.world.raycast(fx, fy, fz + 1.2, math.cos(q) * 0.78, math.sin(q) * 0.78, 0.62, 7.0)
            k = min(wd / 7.0, 1.0) * 0.9
            cx, cy, cz = fx + math.cos(q) * d * k, fy + math.sin(q) * d * k, fz + 1.2 + (cz - fz - 1.2) * k
            r.set_camera(cx, cy, cz, q + math.pi, -math.atan2(cz - fz - 1.0, max(d * k, 0.3)))
        fr = scene.build(m, me, m.time, r, first, tuple(self.sway))
        hurt = 0.0
        if me.alive:
            hurt = max(0.0, 0.3 - (m.time - me.hurt_t)) * 1.2 + max(0.0, 0.35 - me.frac) * 1.6
        img = r.render(w, h, fov, m.time, self._dpr(), fr.rows, fr.alpha, fr.add, fr.glows, fr.lights, fr.vm,
                       fr.casters, hurt=min(hurt, 1.0), vm_fov=scene.VM_FOV)
        p.drawImage(QRectF(0, 0, w, h), img)
        hud.draw(p, w, h, self)
        if m.phase == "over":
            hud.result(p, w, h, self)
        elif self.picker:
            note = "You change at once: you are in your spawn room" if me.alive else "You come back as the hero you pick"
            hud.hero_picker(p, w, h, self, (me.next_hero or me.hero).id, self.picker_hover, note)
        elif self.show_scores:
            hud.scoreboard(p, w, h, self)

    # --- the home card --------------------------------------------------------------------------------------------
    _thumb: dict = {}

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        w, h = int(rect.width()), int(rect.height())
        base = cls._thumb.get((w, h))
        if base is None:
            base = cls._thumb[(w, h)] = _thumb_base(max(w, 1), max(h, 1))
        p.drawImage(rect.topLeft(), base)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(rect.topLeft())
        p.setClipRect(QRectF(0, 0, w, h))
        # the point's ring, turning
        cx, cy = w * 0.5, h * 0.8
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(12):
            a0 = int((i * 30 + t * 30) * 16)
            p.setPen(QPen(QColor(255, 196, 64, 200 if i % 2 else 90), max(h * 0.012, 1.5)))
            p.drawArc(QRectF(cx - w * 0.42, cy - h * 0.09, w * 0.84, h * 0.18), a0, 18 * 16)
        # tracers
        k = (t * 1.3) % 1.0
        p.setPen(QPen(QColor(120, 220, 255, int(220 * (1 - k))), max(h * 0.01, 1.2)))
        p.drawLine(QPointF(w * 0.3, h * 0.47), QPointF(w * (0.3 + 0.42 * k), h * (0.47 + 0.02 * k)))
        k2 = (t * 1.3 + 0.5) % 1.0
        p.setPen(QPen(QColor(255, 120, 100, int(220 * (1 - k2))), max(h * 0.01, 1.2)))
        p.drawLine(QPointF(w * 0.74, h * 0.5), QPointF(w * (0.74 - 0.4 * k2), h * (0.5 + 0.03 * k2)))
        p.restore()


def _figure(p: QPainter, x: float, base: float, s: float, body: QColor, trim: QColor, bulk: float = 1.0,
            flip: bool = False) -> None:
    """A chunky hero in a few rounded boxes, ``s`` tall."""
    d = -1 if flip else 1
    p.setPen(Qt.PenStyle.NoPen)
    w = s * 0.26 * bulk
    p.setBrush(body.darker(150))
    p.drawRoundedRect(QRectF(x - w * 0.5, base - s * 0.46, w * 0.42, s * 0.46), s * 0.03, s * 0.03)
    p.drawRoundedRect(QRectF(x + w * 0.08, base - s * 0.46, w * 0.42, s * 0.46), s * 0.03, s * 0.03)
    p.setBrush(body)
    p.drawRoundedRect(QRectF(x - w * 0.62, base - s * 0.8, w * 1.24, s * 0.4), s * 0.05, s * 0.05)
    p.setBrush(trim)
    p.drawRoundedRect(QRectF(x - w * 0.36, base - s, w * 0.72, s * 0.22), s * 0.06, s * 0.06)
    p.setBrush(QColor(255, 255, 255, 230))
    p.drawRect(QRectF(x - w * 0.2 + d * w * 0.1, base - s * 0.93, w * 0.4, s * 0.05))
    p.setBrush(body.darker(125))
    p.drawRoundedRect(QRectF(x + (0 if d > 0 else -w * 1.5), base - s * 0.7, w * 1.5, s * 0.1), s * 0.03, s * 0.03)
    p.setBrush(trim.darker(140))
    p.drawRoundedRect(QRectF(x + (w * 0.5 if d > 0 else -w * 1.9), base - s * 0.72, w * 1.4, s * 0.12), s * 0.02,
                      s * 0.02)


def _thumb_base(w: int, h: int) -> QImage:
    """The still part of the card: a bright square, two teams of chunky heroes facing each other, the title."""
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    g = QLinearGradient(0, 0, 0, h)
    g.setColorAt(0, QColor(46, 120, 220))
    g.setColorAt(0.55, QColor(190, 220, 240))
    g.setColorAt(0.56, QColor(232, 206, 160))
    g.setColorAt(1, QColor(206, 160, 110))
    p.fillRect(QRectF(0, 0, w, h), g)
    p.setPen(Qt.PenStyle.NoPen)
    rng = random.Random(5)
    x = -w * 0.02
    cols = ((236, 178, 96), (232, 120, 96), (244, 214, 150), (150, 196, 190), (176, 150, 200))
    while x < w:
        bw = w * rng.uniform(0.1, 0.17)
        bh = h * rng.uniform(0.24, 0.42)
        c = QColor(*cols[rng.randrange(5)])
        p.setBrush(c)
        p.drawRect(QRectF(x, h * 0.56 - bh, bw, bh))
        p.setBrush(c.darker(118))
        p.drawRect(QRectF(x + bw * 0.8, h * 0.56 - bh, bw * 0.2, bh))
        p.setBrush(QColor(60, 90, 130, 200))
        for wy in range(int(bh / (h * 0.11))):
            for wx in range(2):
                p.drawRect(QRectF(x + bw * (0.14 + wx * 0.34), h * 0.56 - bh + h * (0.04 + wy * 0.11), bw * 0.2,
                                  h * 0.06))
        x += bw
    shade = QPainterPath()
    shade.addEllipse(QRectF(w * 0.08, h * 0.74, w * 0.84, h * 0.14))
    p.fillPath(shade, QColor(120, 80, 50, 60))
    for x, s, body, trim, bulk, flip in ((0.14, 0.4, (244, 186, 44), (62, 64, 74), 1.5, False),
                                         (0.27, 0.32, (118, 222, 190), (250, 250, 244), 1.0, False),
                                         (0.37, 0.3, (232, 72, 150), (244, 244, 250), 0.9, False),
                                         (0.88, 0.4, (72, 70, 82), (196, 112, 58), 1.5, True),
                                         (0.75, 0.33, (36, 150, 150), (214, 180, 120), 1.0, True),
                                         (0.65, 0.3, (96, 172, 74), (240, 150, 50), 1.05, True)):
        _figure(p, w * x, h * 0.84, h * s, QColor(*body), QColor(*trim), bulk, flip)
        oc = QColor(*(ALLY_RGB if not flip else ENEMY_RGB))
        p.setBrush(oc)
        p.drawPolygon(QPolygonF([QPointF(w * x - h * 0.02, h * (0.84 - s) - h * 0.06),
                                 QPointF(w * x + h * 0.02, h * (0.84 - s) - h * 0.06),
                                 QPointF(w * x, h * (0.84 - s) - h * 0.025)]))
    f = hud.fnt(h * 0.2, True, True)
    p.setFont(f)
    tr = QRectF(0, h * 0.03, w, h * 0.26)
    p.setPen(QColor(10, 20, 50, 170))
    p.drawText(tr.translated(h * 0.012, h * 0.012), int(Qt.AlignmentFlag.AlignCenter), "VANGUARD")
    p.setPen(QColor(255, 255, 255))
    p.drawText(tr, int(Qt.AlignmentFlag.AlignCenter), "VANGUARD")
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 196, 64))
    p.drawPolygon(hud.slant(QRectF(w * 0.34, h * 0.27, w * 0.32, max(h * 0.02, 2)), 0.9))
    p.end()
    return img
