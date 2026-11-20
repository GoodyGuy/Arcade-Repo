"""BREACH POINT: the arcade's five-against-five tactical shooter. This module is the game widget: the menus, the
loading card, play (the 3D view drawn by the graphics card at the window's own resolution, the HUD over it with
QPainter), input, spectating, sound, the service record and the home-screen card."""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen, QPolygonF

from app.arcade.base import ArcadeGame, font

from . import config as C
from . import hud, menus
from .gfx import Renderer
from .maps import MAP_ORDER, build
from .scene import build_frame, view_fov
from .sim import Cmd, Match
from .sounds import build as build_sounds

K = Qt.Key
LOOK = 0.0022  # radians a pixel of mouse at the arcade's sensitivity 1.0
KEY_TURN = 2.4  # radians a second with the arrow keys
ACTIONS = {"fwd": K.Key_W, "back": K.Key_S, "left": K.Key_A, "right": K.Key_D, "walk": K.Key_Shift,
           "crouch": K.Key_Control, "jump": K.Key_Space, "reload": K.Key_R, "use": K.Key_F, "buy": K.Key_B,
           "main": K.Key_1, "side": K.Key_2, "knife": K.Key_3, "C": K.Key_C, "Q": K.Key_Q, "E": K.Key_E,
           "X": K.Key_X, "scores": K.Key_Tab, "call_a": K.Key_Z, "call_b": K.Key_V, "ready": K.Key_Return}
MULTI = {2: "DOUBLE KILL", 3: "TRIPLE KILL", 4: "FOUR DOWN", 5: "ACE"}


def default_settings() -> dict:
    return {"fov": 90, "cross_colour": 0, "cross_size": 3, "cross_gap": 2, "cross_dot": False, "cross_dynamic": True,
            "ads_toggle": False, "crouch_toggle": False, "buy_time": 20}


def default_stats() -> dict:
    return {"matches": 0, "wins": 0, "losses": 0, "rounds_won": 0, "rounds_lost": 0, "kills": 0, "deaths": 0,
            "assists": 0, "heads": 0, "shots": 0, "hits": 0, "plants": 0, "defuses": 0, "damage": 0, "best_kills": 0,
            "agents": {}, "maps": {}}


OPTION_STEPS = {"fov": (70, 110, 5), "cross_colour": (0, len(hud.CROSS_COLOURS) - 1, 1), "cross_size": (1, 6, 1),
                "cross_gap": (0, 6, 1), "buy_time": (10, 30, 5)}


class BreachPointGame(ArcadeGame):
    game_id = "breach_point"
    title = "BREACH POINT"
    subtitle = "5 v 5 tactical shooter"
    description = ("Carry the charge to a site, plant it and hold, or stop the plant and defuse in time. One life a "
                   "round, a buy phase before each, five agents with their own kits, two maps, bots that buy, hold "
                   "angles, trade and rotate.")
    controls = (("WASD", "Move"), ("Mouse", "Look / aim"), ("LMB", "Fire"), ("RMB", "Sights / scope"),
                ("Shift", "Walk (silent)"), ("Ctrl", "Crouch"), ("Space", "Jump"), ("R", "Reload"),
                ("1 2 3 / Wheel", "Rifle, sidearm, knife"), ("C Q E X", "Abilities / ultimate"),
                ("F", "Plant, defuse, pick up"), ("B", "Buy menu"), ("Enter", "Ready (buy phase)"),
                ("Z / V", "Call site A / B"), ("Tab", "Scoreboard"), ("Esc", "Pause"))
    accent = "#ffc62e"
    uses_mouse_look = False  # the game takes and gives back the mouse itself (menus, the buy menu)
    uses_key_turn = True
    bindings = (("Move forward", K.Key_W), ("Move back", K.Key_S), ("Strafe left", K.Key_A),
                ("Strafe right", K.Key_D), ("Turn left", K.Key_Left), ("Turn right", K.Key_Right),
                ("Walk (silent)", K.Key_Shift), ("Crouch", K.Key_Control), ("Jump", K.Key_Space),
                ("Reload", K.Key_R), ("Plant / defuse / pick up", K.Key_F), ("Buy menu", K.Key_B),
                ("Main gun", K.Key_1), ("Sidearm", K.Key_2), ("Knife", K.Key_3), ("Ability 1", K.Key_C),
                ("Ability 2", K.Key_Q), ("Signature ability", K.Key_E), ("Ultimate", K.Key_X),
                ("Call site A", K.Key_Z), ("Call site B", K.Key_V), ("Ready", K.Key_Return),
                ("Scoreboard", K.Key_Tab))

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        build_sounds(self.sfx)
        self.settings = default_settings()
        self.stats = default_stats()
        self.choice = {"map": MAP_ORDER[0], "agent": C.AGENT_ORDER[0], "difficulty": "normal",
                       "rounds": C.WIN_ROUNDS, "side": "atk"}
        self.ui = menus.UI()
        self.screen = "menu"  # menu / loading / play
        self.page = "main"
        self.agent_page = C.AGENT_ORDER[0]
        self.screen_t = 0.0
        self.overlay: str | None = None  # buy / pause / options
        self.match: Match | None = None
        self.renderer: Renderer | None = None
        self.loading_ready = False
        self.recorded = False
        self.show_scores = False
        self.mouse_fire = self.mouse_ads = False
        self.ads_on = False
        self.crouch_on = False
        self.once: dict = {}  # one-shot commands from key presses
        self.spec = 0
        self.eye_z = 1.6
        self._eye_of = None
        self.sway = [0.0, 0.0]
        self.hit_t = 0.0
        self.hit_head = self.hit_kill = False
        self.hurt_t = 0.0
        self.in_smoke = 0.0
        self.kill_note: list | None = None
        self.noise_marks: list = []
        self.paint_ms = 0.0
        self._previews: dict = {}
        self._ult_was = False

    # --- saving -------------------------------------------------------------------------------------------------

    def save_state(self) -> dict:
        return {"version": 1, "settings": dict(self.settings), "stats": self.stats, "choice": dict(self.choice)}

    def load_state(self, data: dict) -> None:
        data = data if isinstance(data, dict) else {}
        sett = data.get("settings") if isinstance(data.get("settings"), dict) else {}
        for k, v in default_settings().items():
            val = sett.get(k, v)
            if isinstance(v, bool):
                self.settings[k] = bool(val)
            else:
                try:
                    val = int(val)
                except (TypeError, ValueError):
                    val = v
                lo, hi, _st = OPTION_STEPS.get(k, (val, val, 1))
                self.settings[k] = min(max(val, lo), hi)
        src = data.get("stats") if isinstance(data.get("stats"), dict) else {}
        st = default_stats()
        for k, v in st.items():
            got = src.get(k, v)
            if isinstance(v, dict):
                if isinstance(got, dict):
                    for name, pair in got.items():
                        try:
                            st[k][str(name)] = [max(int(pair[0]), 0), max(int(pair[1]), 0)]
                        except (TypeError, ValueError, IndexError):
                            pass
            else:
                try:
                    st[k] = max(int(got), 0)
                except (TypeError, ValueError):
                    pass
        self.stats = st
        ch = data.get("choice") if isinstance(data.get("choice"), dict) else {}
        if ch.get("map") in MAP_ORDER:
            self.choice["map"] = ch["map"]
        if ch.get("agent") in C.AGENTS:
            self.choice["agent"] = ch["agent"]
        if ch.get("difficulty") in C.DIFFICULTY:
            self.choice["difficulty"] = ch["difficulty"]
        if ch.get("rounds") in C.MATCH_LENGTHS:
            self.choice["rounds"] = ch["rounds"]
        if ch.get("side") in ("atk", "def"):
            self.choice["side"] = ch["side"]

    @classmethod
    def summary(cls, data: dict) -> str:
        try:
            st = (data or {}).get("stats") or {}
            n = int(st.get("matches", 0))
            if not n:
                return "New recruit"
            return f"{int(st.get('wins', 0))} won · {int(st.get('losses', 0))} lost · {int(st.get('kills', 0))} kills"
        except (TypeError, ValueError, AttributeError):
            return "Continue"

    # --- keys ---------------------------------------------------------------------------------------------------

    def key_label(self, action: str) -> str:
        """The key the player presses for an action (after any rebinding), for the HUD."""
        from app.arcade.controls import key_name

        key = ACTIONS.get(action)
        if key is None:
            return action
        for pressed, default in self._remap.items():
            if default == key:
                return key_name(pressed).upper()
        if key in self._unbound:
            return "—"
        return key_name(int(key)).upper()

    # --- matches ------------------------------------------------------------------------------------------------

    def map_preview(self, key: str):  # noqa: ANN201 - QPixmap of the map from above
        pm = self._previews.get(key)
        if pm is None:
            pm = self._previews[key] = hud.minimap_pix(build(key))
        return pm

    def start_match(self, seed: int | None = None, autopilot: bool = False) -> None:
        """Build the chosen map, a match on it and the renderer (the loading screen calls this)."""
        ch = self.choice
        self._close_renderer()
        gmap = build(ch["map"])
        self.match = Match(gmap, ch["agent"], ch["difficulty"], ch["rounds"], ch["side"], seed=seed,
                           autopilot=autopilot, buy_time=float(self.settings["buy_time"]))
        self.renderer = Renderer(gmap)
        self.renderer.warm_up(self.width(), self.height())
        hud.minimap_pix(gmap)
        self.recorded = False
        self.overlay = None
        self.spec = 0
        self.noise_marks = []
        self.kill_note = None
        self.hit_t = self.hurt_t = self.in_smoke = 0.0
        self.mouse_fire = self.mouse_ads = self.ads_on = self.crouch_on = False
        self.once = {}
        w, h = max(self.width(), 320), max(self.height(), 200)
        img = QImage(w, h, QImage.Format.Format_RGB32)  # one whole frame now, so the first real one is not slow
        p = QPainter(img)
        try:
            self._eye_of, self.eye_z = self.match.player, self.match.player.eye
            self._paint_play(p, w, h)
            hud.draw_scoreboard(p, w, h, self, h / 780.0)
            hud.warm(p, h, self.match.player.agent)
        finally:
            p.end()
        self.loading_ready = True

    def leave_match(self) -> None:
        self.match = None
        self._close_renderer()
        self.overlay = None
        self.screen, self.page, self.screen_t = "menu", "main", 0.0
        self.set_mouse_captured(False)
        self.progress_changed.emit()

    def _record(self) -> None:
        m = self.match
        if m is None or self.recorded:
            return
        self.recorded = True
        res = m.result()
        st = self.stats
        st["matches"] += 1
        st["wins" if res["won"] else "losses"] += 1
        st["rounds_won"] += res["score"][0]
        st["rounds_lost"] += res["score"][1]
        for k in ("kills", "deaths", "assists", "heads", "shots", "hits", "plants", "defuses", "damage"):
            st[k] += int(res[k])
        st["best_kills"] = max(st["best_kills"], res["kills"])
        for group, key in (("agents", res["agent"]), ("maps", res["map"])):
            pair = st[group].setdefault(key, [0, 0])
            pair[0] += 1
            pair[1] += 1 if res["won"] else 0
        self.progress_changed.emit()

    def viewer(self):  # noqa: ANN201 - sim.Actor: whose eyes the screen shows
        m = self.match
        p = m.player
        if p.alive or m.phase == "over":
            return p
        mates = [a for a in m.actors if a.squad == 0 and a.alive]
        if not mates:
            return p
        return mates[self.spec % len(mates)]

    def stop(self) -> None:
        super().stop()
        self._close_renderer()

    def _close_renderer(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # --- actions (menus and the buy menu) -----------------------------------------------------------------------

    def do(self, action: tuple | None, button=Qt.MouseButton.LeftButton) -> None:  # noqa: ANN001
        if not action:
            return
        kind = action[0]
        sfx = self.sfx
        if kind == "page":
            self.page = action[1]
            self.ui.focus = -1
            sfx.play("menu_select" if action[1] != "main" else "menu_back")
        elif kind == "set":
            self.choice[action[1]] = action[2]
            sfx.play("menu_move")
        elif kind == "agent_page":
            self.agent_page = action[1]
            sfx.play("menu_move")
        elif kind == "start":
            self.screen, self.screen_t, self.loading_ready = "loading", 0.0, False
            sfx.play("menu_select")
        elif kind == "exit":
            self.exit_requested.emit()
        elif kind == "opt":
            key, d = action[1], action[2]
            st = self.settings
            if isinstance(st[key], bool):
                st[key] = not st[key]
            else:
                lo, hi, step = OPTION_STEPS[key]
                v = st[key] + d * step
                st[key] = lo if v > hi and key == "cross_colour" else (hi if v < lo and key == "cross_colour" else
                                                                      min(max(v, lo), hi))
            sfx.play("menu_move")
            self.progress_changed.emit()
        elif kind == "controls":
            self.mouse_fire = self.mouse_ads = False
            self.keys.clear()
            self.set_mouse_captured(False)
            self.controls_requested.emit()
        elif kind == "resume":
            self.overlay = None
            sfx.play("menu_back")
        elif kind == "overlay":
            self.overlay = action[1]
            self.ui.focus = -1
            sfx.play("menu_select")
        elif kind == "leave":
            sfx.play("menu_back")
            self.leave_match()
        elif kind == "buy" and self.match is not None:
            m = self.match
            if button == Qt.MouseButton.RightButton:
                ok = m.sell(m.player, action[1])
            else:
                ok = m.buy(m.player, action[1])
            sfx.play("buy" if ok else "deny")

    def handle_escape(self) -> bool:
        if self.screen == "play":
            if self.match is not None and self.match.phase == "over":
                return True
            if self.overlay == "options":
                self.overlay = "pause"
            elif self.overlay in ("pause", "buy"):
                self.overlay = None
            else:
                self.overlay = "pause"
                self.ui.focus = -1
            self.sfx.play("menu_back")
            return True
        if self.screen == "loading":
            return True
        if self.page != "main":
            self.do(("page", "main"))
            return True
        return False

    # --- input --------------------------------------------------------------------------------------------------

    def _menu_active(self) -> bool:
        return self.screen == "menu" or self.overlay in ("pause", "options", "buy")

    def playing(self) -> bool:
        return self.screen == "play" and self.match is not None and self.overlay is None

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001
        if self.screen == "loading":
            if self.loading_ready and self.screen_t > 0.3:
                self._enter_play()
            return
        m = self.match
        if self.screen == "play" and m is not None and m.phase == "over":
            if key in (K.Key_Return, K.Key_Enter, K.Key_Space) and self.screen_t > 1.5:
                self.leave_match()
            return
        if self._menu_active():
            if self.overlay == "buy" and key in (K.Key_B, K.Key_Return, K.Key_Enter):
                if key != K.Key_B and m is not None:
                    m.set_ready()
                self.overlay = None
                return
            if key in (K.Key_Down, K.Key_Right, K.Key_S, K.Key_D):
                self.ui.step(1)
                self.sfx.play("menu_move")
            elif key in (K.Key_Up, K.Key_Left, K.Key_W, K.Key_A):
                self.ui.step(-1)
                self.sfx.play("menu_move")
            elif key in (K.Key_Return, K.Key_Enter, K.Key_Space):
                self.do(self.ui.focused())
            return
        if self.screen != "play" or m is None:
            return
        if key == K.Key_Tab:
            self.show_scores = True
        elif key == K.Key_B:
            if m.phase == "buy" and m.player.alive:
                self.overlay = "buy"
                self.mouse_fire = self.mouse_ads = False
                self.sfx.play("menu_select")
            else:
                self.sfx.play("deny")
        elif key in (K.Key_Return, K.Key_Enter):
            m.set_ready()
        elif key == K.Key_R:
            self.once["reload"] = True
        elif key in (K.Key_1, K.Key_2, K.Key_3):
            self.once["slot"] = {K.Key_1: "main", K.Key_2: "side", K.Key_3: "knife"}[key]
        elif key in (K.Key_C, K.Key_Q, K.Key_E, K.Key_X):
            self.once["ability"] = {K.Key_C: "C", K.Key_Q: "Q", K.Key_E: "E", K.Key_X: "X"}[key]
        elif key in (K.Key_Z, K.Key_V):
            site = "A" if key == K.Key_Z else "B"
            ai = m.team_ai.get(0)
            if ai is not None and m.player.alive:
                ai.set_call(site)
                m.say("", f"YOU CALLED SITE {site}", "ally", 1.6)
                self.sfx.play("menu_select")
        elif key == K.Key_Control and self.settings["crouch_toggle"]:
            self.crouch_on = not self.crouch_on

    def key_released(self, key: int, event) -> None:  # noqa: ANN001
        if key == K.Key_Tab:
            self.show_scores = False

    def mouse_pressed(self, button, pos) -> None:  # noqa: ANN001
        if self.screen == "loading":
            if self.loading_ready and self.screen_t > 0.3:
                self._enter_play()
            return
        m = self.match
        if self.screen == "play" and m is not None and m.phase == "over":
            return
        if self._menu_active():
            self.ui.moved(pos)
            self.do(self.ui.at(pos), button)
            return
        if self.screen != "play" or m is None:
            return
        if not m.player.alive:
            if button == Qt.MouseButton.LeftButton:
                self.spec += 1
            return
        if button == Qt.MouseButton.LeftButton:
            self.mouse_fire = True
        elif button == Qt.MouseButton.RightButton:
            self.mouse_ads = True
            if self.settings["ads_toggle"]:
                self.ads_on = not self.ads_on
        if not self.mouse_captured and self.isActiveWindow():
            self.set_mouse_captured(True)

    def mouse_released(self, button, pos) -> None:  # noqa: ANN001
        if button == Qt.MouseButton.LeftButton:
            self.mouse_fire = False
        elif button == Qt.MouseButton.RightButton:
            self.mouse_ads = False

    def mouse_moved(self, pos) -> None:  # noqa: ANN001
        self.ui.moved(pos)

    def wheel(self, steps: int) -> None:
        if not self.playing() or self.match is None:
            return
        p = self.match.player
        order = [s for s in ("main", "side", "knife") if s != "main" or p.main]
        cur = p.cur if p.cur in order else "side"
        self.once["slot"] = order[(order.index(cur) - steps) % len(order)]

    def _command(self, dt: float) -> Cmd:
        keys = self.keys
        m = self.match
        p = m.player
        c = Cmd()
        c.fwd = (1.0 if K.Key_W in keys or K.Key_Up in keys else 0.0) - (1.0 if K.Key_S in keys or K.Key_Down in keys
                                                                         else 0.0)
        c.side = (1.0 if K.Key_D in keys else 0.0) - (1.0 if K.Key_A in keys else 0.0)
        c.walk = K.Key_Shift in keys
        c.crouch = self.crouch_on if self.settings["crouch_toggle"] else K.Key_Control in keys
        c.jump = K.Key_Space in keys
        c.fire = self.mouse_fire
        c.ads = self.ads_on if self.settings["ads_toggle"] else self.mouse_ads
        c.use = K.Key_F in keys
        c.reload = bool(self.once.pop("reload", False))
        c.slot = self.once.pop("slot", None)
        c.ability = self.once.pop("ability", None)
        if c.slot or p.reload_t > 0:
            self.ads_on = False
        dx, dy = self.take_mouse_delta()
        zoom = 1.0 + (p.weapon.zoom - 1.0) * p.ads
        sens = LOOK / zoom
        turn = (1.0 if K.Key_Right in keys else 0.0) - (1.0 if K.Key_Left in keys else 0.0)
        slow = 0.45 if p.daze > 0 else 1.0
        p.yaw = (p.yaw + (dx * sens + turn * KEY_TURN * self.key_turn_speed * dt / zoom) * slow + math.pi) % (
            2 * math.pi) - math.pi
        p.pitch = min(max(p.pitch - dy * sens * slow, -1.45), 1.45)
        self.sway[0] += (dx * 0.03 - self.sway[0]) * min(dt * 9, 1)
        self.sway[1] += (dy * 0.03 - self.sway[1]) * min(dt * 9, 1)
        self.sway = [max(min(v, 1.2), -1.2) for v in self.sway]
        return c

    # --- the loop -----------------------------------------------------------------------------------------------

    def _enter_play(self) -> None:
        self.screen, self.screen_t = "play", 0.0
        self.mouse_fire = self.mouse_ads = False
        self.take_mouse_delta()

    def tick(self, dt: float) -> None:
        self.screen_t += dt
        m = self.match
        want = (self.playing() and m is not None and m.phase != "over" and self.isActiveWindow())
        if want != self.mouse_captured and (not want or self.isVisible()):
            self.set_mouse_captured(want)
        if self.screen == "loading":
            if not self.loading_ready and self.screen_t > 0.05:
                self.start_match()
            if self.loading_ready and self.screen_t > 1.6:
                self._enter_play()
            return
        if self.screen != "play" or m is None:
            return
        if self.overlay in ("pause", "options"):
            self.take_mouse_delta()
            return
        if self.overlay == "buy" and (m.phase != "buy" or not m.player.alive):
            self.overlay = None
        cmd = None
        if m.player.alive and m.phase != "over" and not m.autopilot:
            if self.overlay is None:
                cmd = self._command(dt)
            else:
                self.take_mouse_delta()
        was_over = m.phase == "over"
        m.update(dt, cmd)
        if m.phase == "over" and not was_over:
            self.screen_t = 0.0
            self.overlay = None
            self._record()
        self._after_update(dt)

    def _after_update(self, dt: float) -> None:
        m = self.match
        v = self.viewer()
        sfx = self.sfx
        played = 0
        for name, x, y, loud in m.sounds:
            d = math.hypot(x - v.x, y - v.y)
            vol = loud / (1.0 + (d / 9.0) ** 2)
            if vol > 0.05 and played < 10:
                sfx.play(name, min(vol, 1.0))
                played += 1
        m.sounds.clear()
        for fx in m.fx:
            if fx[0] == "hit":
                self.hit_t, self.hit_head, self.hit_kill = 0.22, fx[1], fx[2]
                sfx.play("headshot" if fx[1] else "hit")
            elif fx[0] == "hurt":
                self.hurt_t = min(self.hurt_t + 0.25 + fx[1] / 120.0, 0.8)
            elif fx[0] == "kill":
                note = "ELIMINATED  " + fx[1] + ("  ·  HEADSHOT" if fx[2] else "")
                if fx[3] in MULTI:
                    note += "  ·  " + MULTI[fx[3]]
                self.kill_note = [note, 2.4]
                sfx.play("kill")
        m.fx.clear()
        self.hit_t = max(self.hit_t - dt, 0.0)
        self.hurt_t = max(self.hurt_t - dt, 0.0)
        if self.kill_note is not None:
            self.kill_note[1] -= dt
            if self.kill_note[1] <= 0:
                self.kill_note = None
        ready = m.player.alive and m.player.can_use("X")
        if ready and not self._ult_was and m.phase in ("live", "planted"):
            sfx.play("ult_ready")
        self._ult_was = ready
        if m.phase == "buy" and m.t < 3.2 and int(m.t + dt) != int(m.t):
            sfx.play("tick10")
        # what the watched agent hears of the other team, drawn on the map
        for mk in self.noise_marks:
            mk[2] += dt
        self.noise_marks = [mk for mk in self.noise_marks if mk[2] < 1.2]
        if v.alive:
            for x, y, r, squad, kind, _who in m.noises:
                if squad == v.squad or kind not in ("step", "shot", "plant", "defuse"):
                    continue
                if (x - v.x) ** 2 + (y - v.y) ** 2 < r * r and not any(
                        abs(mk[0] - x) < 2.5 and abs(mk[1] - y) < 2.5 and mk[2] < 0.5 for mk in self.noise_marks):
                    self.noise_marks.append([x, y, 0.0])
        # the eye: smoothed over steps and crouching
        if self._eye_of is not v:
            self._eye_of, self.eye_z = v, v.eye
        self.eye_z += (v.eye - self.eye_z) * min(dt * 18.0, 1.0)
        inside = 0.0
        for s in m.smokes:
            if s[4] > 0.2 and (s[0] - v.x) ** 2 + (s[1] - v.y) ** 2 + (s[2] - self.eye_z) ** 2 < (s[3] * 0.88) ** 2:
                inside = 1.0
        self.in_smoke += (inside - self.in_smoke) * min(dt * 9.0, 1.0)

    # --- painting -----------------------------------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        t0 = time.perf_counter()
        p = QPainter(self)
        w, h = self.width(), self.height()
        try:
            self.ui.begin()
            if self.screen == "play" and self.match is not None and self.renderer is not None:
                self._paint_play(p, w, h)
            elif self.screen == "loading":
                menus.paint_loading(p, w, h, self, self.loading_ready)
            else:
                menus.backdrop(p, w, h, self.time)
                if self.page == "play":
                    menus.paint_play(p, w, h, self)
                elif self.page == "agents":
                    menus.paint_agents(p, w, h, self)
                elif self.page == "options":
                    menus.paint_options(p, w, h, self, ("page", "main"))
                elif self.page == "stats":
                    menus.paint_stats(p, w, h, self)
                else:
                    menus.paint_main(p, w, h, self)
            if self.paused and self.screen == "play":
                p.fillRect(self.rect(), QColor(0, 0, 0, 90))
        finally:
            p.end()
        self.paint_ms = (time.perf_counter() - t0) * 1000

    def _paint_play(self, p: QPainter, w: int, h: int) -> None:
        m = self.match
        r = self.renderer
        v = self.viewer()
        dpr = float(self.devicePixelRatioF()) if self.isVisible() else 1.0
        if v.alive:
            yaw, pitch = v.view()
            if v.daze > 0:
                k = min(v.daze, 1.0) * 0.03
                yaw += math.sin(self.time * 7.0) * k
                pitch += math.cos(self.time * 5.3) * k
            r.set_camera(v.x, v.y, self.eye_z, yaw, pitch, view_fov(self.settings["fov"], v), w, h)
            first = True
        else:  # everyone on the team is down: look at where you fell
            ca, sa = math.cos(v.yaw), math.sin(v.yaw)
            bx, by = v.x - ca * 1.6, v.y - sa * 1.6
            if m.map.height(bx, by) > 1.0:
                bx, by = v.x, v.y
            r.set_camera(bx, by, v.z + 2.3, v.yaw, -0.85, math.radians(self.settings["fov"]), w, h)
            first = False
        fr = build_frame(m, v, r, self.time, self.eye_z, self.sway, first)
        if v.near_t > 0:
            fr.vignette = 2.6
        img = r.render(fr, self.time, dpr)
        p.drawImage(QRectF(0, 0, w, h), img)
        hud.draw_hud(p, w, h, self)
        s = h / 780.0
        if m.phase == "over":
            hud.draw_match_end(p, w, h, self, s)
        elif self.overlay == "buy":
            hud.draw_buy(p, w, h, self, s)
        elif self.overlay == "pause":
            menus.paint_pause(p, w, h, self)
        elif self.overlay == "options":
            p.fillRect(QRectF(0, 0, w, h), QColor(10, 9, 18, 225))
            menus.paint_options(p, w, h, self, ("overlay", "pause"))
        elif self.show_scores:
            hud.draw_scoreboard(p, w, h, self, s)

    # --- the home-screen card -----------------------------------------------------------------------------------

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        p.save()
        p.setClipRect(rect)
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        g = QLinearGradient(rect.topLeft(), rect.bottomRight())
        g.setColorAt(0.0, QColor(34, 28, 58))
        g.setColorAt(1.0, QColor(12, 11, 22))
        p.fillRect(rect, g)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        # a wall with a doorway, and the light through it
        p.setBrush(QColor(60, 34, 40))
        p.drawRect(QRectF(x, y + h * 0.2, w * 0.36, h * 0.8))
        p.drawRect(QRectF(x + w * 0.62, y + h * 0.2, w * 0.38, h * 0.8))
        p.setBrush(QColor(255, 198, 46, 50))
        p.drawPolygon(QPolygonF([QPointF(x + w * 0.36, y + h * 0.2), QPointF(x + w * 0.62, y + h * 0.2),
                                 QPointF(x + w * 0.8, y + h), QPointF(x + w * 0.18, y + h)]))
        # the charge, beeping
        cx, cy = x + w * 0.49, y + h * 0.7
        pulse = (t * 1.4) % 1.0
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 198, 46, int(255 * (1 - pulse))), max(h * 0.012, 1.0)))
        p.drawEllipse(QPointF(cx, cy), h * (0.06 + 0.34 * pulse), h * (0.03 + 0.17 * pulse))
        p.setPen(Qt.PenStyle.NoPen)
        hud.charge_icon(p, cx, cy, h * 0.13, QColor(255, 198, 46))
        # two agents either side of the door
        for fx, col, flip in ((0.2, C.ALLY, 1), (0.8, C.ENEMY, -1)):
            ax, ay, u = x + w * fx, y + h * 0.86, h * 0.05
            p.setBrush(QColor(28, 28, 38))
            p.drawRect(QRectF(ax - u * 1.1, ay - u * 6.4, u * 2.2, u * 3.4))
            p.drawRect(QRectF(ax - u * 1.0, ay - u * 3.0, u * 0.9, u * 3.0))
            p.drawRect(QRectF(ax + u * 0.1, ay - u * 3.0, u * 0.9, u * 3.0))
            p.setBrush(QColor(*col))
            p.drawRect(QRectF(ax - u * 0.75, ay - u * 8.0, u * 1.5, u * 1.5))
            p.drawRect(QRectF(ax - u * 0.8, ay - u * 6.0, u * 1.6, u * 1.6))
            p.setBrush(QColor(20, 20, 26))
            p.drawRect(QRectF(ax + (u * 0.6 if flip > 0 else -u * 4.0), ay - u * 5.4, u * 3.4, u * 0.6))
        # the slash and the name
        p.setBrush(QColor(255, 198, 46))
        p.drawPolygon(QPolygonF([QPointF(x + w * 0.9, y), QPointF(x + w * 0.96, y), QPointF(x + w * 0.8, y + h * 0.34),
                                 QPointF(x + w * 0.74, y + h * 0.34)]))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        f = font(h * 0.15, True, hud.FAMILY)
        p.setFont(f)
        p.setPen(QColor(0, 0, 0, 170))
        p.drawText(QRectF(x + w * 0.05 + 2, y + h * 0.04 + 2, w, h * 0.2), Qt.AlignmentFlag.AlignLeft, "BREACH")
        p.setPen(QColor(240, 236, 226))
        p.drawText(QRectF(x + w * 0.05, y + h * 0.04, w, h * 0.2), Qt.AlignmentFlag.AlignLeft, "BREACH")
        p.setPen(QColor(255, 198, 46))
        p.drawText(QRectF(x + w * 0.05, y + h * 0.19, w, h * 0.2), Qt.AlignmentFlag.AlignLeft, "POINT")
        p.restore()
