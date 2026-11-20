"""SHADOW OPS II: the arcade's zombies-and-multiplayer shooter. This module is the game widget: the boot splash,
menus, briefing loading screens, play (the 3D view drawn by the graphics card at the window's own resolution, then the
view model and HUD over it with QPainter), input, saving, and the home-screen card."""

from __future__ import annotations

import math
import random
import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient

from app.arcade.base import ArcadeGame

from . import hud
from .data import (DEFAULT_STREAKS, MAX_LEVEL, MAX_PRESTIGE, default_classes, level_for_xp, rank_name, unlocks_at,
                   validate_class, validate_streaks)
from .gfx import Renderer
from .maps import MAP_DEFS, MP_MAP, ZOMBIES_MAP, parse
from .menus import Menu
from .sounds import build as build_sounds
from .sprites import SpriteSet
from .textures import Atlas
from .viewmodel import ViewState, draw_drink, draw_knife, draw_red_dot, draw_scope_overlay, draw_throw, draw_weapon
from .world import Controls

K = Qt.Key
LOOK_RAD_PER_PX = 0.0022  # mouse look at the arcade's sensitivity 1.0
KEY_TURN = 2.4  # rad / s turning with the arrow keys at turn speed 1.0
BRIEFINGS = {
    "zombies": ("NIGHT SHIFT", "PLANT 9  ·  23:10", [
        "The night shift at the tank works never clocked out.",
        "Hold the break room. Board the windows. Buy every door.",
        "The breaker is in the generator room. The tonics need it.",
        "Somewhere in the test bay the Refit Press still runs."]),
    "mp": ("CROSSDOCK", "TEAM DEATHMATCH", [
        "A freight yard by day: containers north, tanks south.",
        "VANGUARD versus ONYX: first team to the score limit wins.",
        "Earn scorestreaks without dying. Call them in with 3, 4, 5.",
        "Watch the minimap. Every unsuppressed shot gives you away."]),
}
TORCH = 0.7  # the weapon light on the night map


def default_settings() -> dict:
    return {"fov": 85, "ads_sens": 1.0}


class ShadowOpsGame(ArcadeGame):
    game_id = "shadow_ops"
    title = "SHADOW OPS II"
    subtitle = "Zombies & Multiplayer"
    description = ("Round-based zombie survival in NIGHT SHIFT's tank works and Team Deathmatch against bots in the "
                   "CROSSDOCK freight yard. The Lucky Crate, tonics, the Refit Press, ten-point loadouts, "
                   "scorestreaks and prestige.")
    controls = (("WASD", "Move"), ("Mouse", "Look / aim"), ("LMB", "Fire"), ("RMB", "Aim down sights"),
                ("Shift", "Sprint"), ("C / Ctrl", "Crouch"), ("Space", "Jump"), ("R", "Reload"),
                ("1 / 2 / Wheel", "Switch weapon"), ("F", "Use / buy / rebuild"), ("V", "Knife"),
                ("G", "Lethal"), ("Q", "Tactical"), ("3 4 5", "Scorestreaks"), ("Tab", "Scoreboard"),
                ("← →", "Turn"), ("Esc", "Menu (Controls...)"))
    accent = "#f26b1d"
    uses_mouse_look = False
    uses_key_turn = True  # the arrow keys turn the view
    bindings = (("Move forward", K.Key_W), ("Move back", K.Key_S), ("Strafe left", K.Key_A),
                ("Strafe right", K.Key_D), ("Turn left", K.Key_Left), ("Turn right", K.Key_Right),
                ("Sprint", K.Key_Shift), ("Crouch (toggle)", K.Key_C), ("Crouch (hold)", K.Key_Control),
                ("Jump", K.Key_Space), ("Reload", K.Key_R), ("Use / buy / rebuild", K.Key_F), ("Knife", K.Key_V),
                ("Lethal grenade", K.Key_G), ("Tactical", K.Key_Q), ("Primary weapon", K.Key_1),
                ("Secondary weapon", K.Key_2), ("Scorestreak 1", K.Key_3), ("Scorestreak 2", K.Key_4),
                ("Scorestreak 3", K.Key_5), ("Scoreboard", K.Key_Tab))

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        build_sounds(self.sfx)
        self.settings = default_settings()
        self.profile = {"xp": 0, "prestige": 0, "classes": default_classes(), "streaks": list(DEFAULT_STREAKS),
                        "cur_class": 0}
        self.zstats = {"best_round": 0, "games": 0, "kills": 0, "headshots": 0, "downs": 0, "points": 0}
        self.mpstats = {"kills": 0, "deaths": 0, "wins": 0, "losses": 0, "shots": 0, "hits": 0, "headshots": 0,
                        "matches": 0, "score": 0, "best_streak": 0}
        self.zresume: dict | None = None
        self.screen = "boot"
        self.screen_t = 0.0
        self.menu = Menu(self)
        self.mode = None
        self.renderer: Renderer | None = None
        self._worlds: dict = {}
        self._sprites: dict = {}
        self.pending: tuple | None = None
        self.loading_ready = False
        self.paint_ms = 0.0
        self.crouch_toggle = False
        self.mouse_fire = False
        self.mouse_ads = False
        self.use_tap = False
        self.sway = [0.0, 0.0]
        self.rank_popup: tuple | None = None  # (text, sub, time left)
        self.music_on = False
        self.ambient_on = False
        self._last_tracer_muzzle = None
        self.show_scores = False

    # --- saving -----------------------------------------------------------------------------------------------

    def save_state(self) -> dict:
        data = {"profile": self.profile, "settings": self.settings, "zstats": self.zstats, "mpstats": self.mpstats,
                "version": 2}
        if self.mode is not None and self.mode.kind == "zombies" and not self.mode.over and self.mode.round >= 1:
            data["zresume"] = self.mode.snapshot()
        elif self.zresume:
            data["zresume"] = self.zresume
        return data

    def load_state(self, data: dict) -> None:
        data = data if isinstance(data, dict) else {}
        prof = data.get("profile") if isinstance(data.get("profile"), dict) else {}
        try:
            self.profile["xp"] = max(int(prof.get("xp", 0)), 0)
            self.profile["prestige"] = min(max(int(prof.get("prestige", 0)), 0), MAX_PRESTIGE)
            self.profile["cur_class"] = min(max(int(prof.get("cur_class", 0)), 0), 4)
        except (TypeError, ValueError):
            pass
        classes = prof.get("classes")
        if isinstance(classes, list) and len(classes) == 5 and all(isinstance(c, dict) for c in classes):
            fixed = []
            for c, default in zip(classes, default_classes(), strict=True):
                c = {**default, **c}
                fixed.append(c if not validate_class(c, 55, 1) else default)
            self.profile["classes"] = fixed
        streaks = prof.get("streaks")
        if isinstance(streaks, list) and not validate_streaks(streaks, 55, 1):
            self.profile["streaks"] = list(streaks)
        sett = data.get("settings") if isinstance(data.get("settings"), dict) else {}
        for k, v in default_settings().items():
            val = sett.get(k, v)
            if isinstance(v, bool):
                self.settings[k] = bool(val)
            else:
                try:
                    self.settings[k] = type(v)(val)
                except (TypeError, ValueError):
                    self.settings[k] = v
        self.settings["ads_sens"] = round(min(max(float(self.settings["ads_sens"]), 0.3), 1.5), 2)
        self.settings["fov"] = int(min(max(self.settings["fov"], 65), 110))
        for store, key in ((self.zstats, "zstats"), (self.mpstats, "mpstats")):
            src = data.get(key) if isinstance(data.get(key), dict) else {}
            for k in store:
                try:
                    store[k] = max(int(src.get(k, store[k])), 0)
                except (TypeError, ValueError):
                    pass
        zr = data.get("zresume")
        self.zresume = zr if isinstance(zr, dict) and zr.get("round") else None

    @classmethod
    def summary(cls, data: dict) -> str:
        if not data:
            return "New recruit"
        try:
            prof = data.get("profile", {}) or {}
            lv = level_for_xp(int(prof.get("xp", 0)))
            pr = int(prof.get("prestige", 0))
            best = int((data.get("zstats") or {}).get("best_round", 0))
        except (TypeError, ValueError, AttributeError):
            return "Continue"
        rank = f"Prestige {pr} Lv {lv}" if pr else f"Lv {lv}"
        if best:
            return f"Zombies best: round {best} · {rank}"
        return f"{rank} · {rank_name(lv)}"

    # --- profile ------------------------------------------------------------------------------------------------

    @property
    def level(self) -> int:
        return level_for_xp(self.profile["xp"])

    def add_xp(self, n: int) -> None:
        before = self.level
        self.profile["xp"] += max(int(n), 0)
        after = self.level
        if after > before:
            got = []
            for lv in range(before + 1, after + 1):
                got += unlocks_at(lv)
            sub = ("Unlocked: " + ", ".join(got[:4])) if got else rank_name(after)
            self.rank_popup = (f"PROMOTED: LEVEL {after}", sub, 4.0)
            self.sfx.play("rank_up")
            self.progress_changed.emit()

    def can_prestige(self) -> bool:
        return self.level >= MAX_LEVEL and self.profile["prestige"] < MAX_PRESTIGE

    def prestige(self) -> bool:
        if not self.can_prestige():
            return False
        self.profile["prestige"] += 1
        self.profile["xp"] = 0
        self.rank_popup = (f"PRESTIGE {self.profile['prestige']}", "Your rank starts over. Your unlocks stay.", 4.0)
        self.sfx.play("rank_up")
        self.progress_changed.emit()
        return True

    # --- worlds ---------------------------------------------------------------------------------------------------

    def sprite_set(self, style: str) -> SpriteSet:
        if style not in self._sprites:
            self._sprites[style] = SpriteSet(style)
        return self._sprites[style]

    def world(self, key: str) -> tuple:
        """A fresh map (doors closed) with its atlas (cached across games)."""
        m = parse(key)
        if key not in self._worlds:
            self._worlds[key] = Atlas(m.tex_names, MAP_DEFS[key]["grade"])
        return m, self._worlds[key]

    def start_mode(self, kind: str, resume: bool = False) -> None:
        self.pending = (kind, resume)
        self.loading_ready = False
        self.screen = "loading"
        self.screen_t = 0.0
        self.menu.close()
        self.stop_loops()
        self.sfx.play("menu_select")

    def _build_pending(self) -> None:
        kind, resume = self.pending
        if kind == "zombies":
            from .zombies import ZombiesMode

            m, atlas = self.world(ZOMBIES_MAP)
            snap = self.zresume if resume else None
            self.mode = ZombiesMode(m, self.sprite_set("dusk"), self.play_sound, snap)
            self.zresume = None
        else:
            from .mp import MPMode

            m, atlas = self.world(MP_MAP)
            cls = self.profile["classes"][self.profile["cur_class"]]
            self.mode = MPMode(m, self.sprite_set("day"), self.play_sound, cls, list(self.profile["streaks"]),
                               self.level, self.profile["prestige"])
        self._close_renderer()
        self.renderer = Renderer(m, atlas)
        self.renderer.warm_up()
        self.renderer.preload(self.mode.preload_images())
        self.loading_ready = True
        self._counted = False

    def end_mode(self) -> None:
        """Back to the main menu from a game."""
        if self.mode is not None and self.mode.kind == "zombies":
            if not self.mode.over and self.mode.round >= 1:
                self.zresume = self.mode.snapshot()
            else:
                self.zresume = None
        self.mode = None
        self._close_renderer()
        self.stop_loops()
        self.set_mouse_captured(False)
        self.screen = "menu"
        self.menu.open_root()
        self.progress_changed.emit()

    def _record_zombies(self) -> None:
        z = self.mode
        zs = self.zstats
        zs["games"] += 1
        zs["kills"] += z.stats.kills
        zs["headshots"] += z.stats.headshots
        zs["downs"] += z.stats.downs
        zs["points"] += z.stats.points_total
        if z.round > zs["best_round"]:
            zs["best_round"] = z.round
        self.zresume = None
        self.add_xp(z.stats.kills * 25 + z.round * 150)
        self.progress_changed.emit()

    # --- sound --------------------------------------------------------------------------------------------------

    def play_sound(self, name: str, volume: float = 1.0) -> None:
        self.sfx.play(name, volume)

    def stop_loops(self) -> None:
        self.sfx.stop("menu_music")
        self.sfx.stop("ambient_z")
        self.music_on = self.ambient_on = False

    def _update_loops(self) -> None:
        want_music = self.screen in ("menu", "boot") and not self.paused
        want_amb = self.screen == "play" and self.mode is not None and self.mode.kind == "zombies" and not self.paused
        if want_music != self.music_on:
            self.music_on = want_music
            (self.sfx.play if want_music else self.sfx.stop)("menu_music")
        if want_amb != self.ambient_on:
            self.ambient_on = want_amb
            (self.sfx.play if want_amb else self.sfx.stop)("ambient_z")

    def on_pause(self) -> None:
        self.stop_loops()
        self.mouse_fire = self.mouse_ads = False

    def on_resume(self) -> None:
        self._update_loops()

    def stop(self) -> None:
        """The arcade closes the game: give the graphics card its memory back too."""
        super().stop()
        self._close_renderer()

    def _close_renderer(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # --- input --------------------------------------------------------------------------------------------------

    def playing(self) -> bool:
        return self.screen == "play" and self.mode is not None and not self.menu.active

    def handle_escape(self) -> bool:
        if self.menu.active:
            return self.menu.escape()
        if self.screen == "play":
            self.menu.open_pause()
            self.sfx.play("menu_back")
            return True
        if self.screen in ("loading", "boot"):
            return True
        return False

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001
        if self.menu.active:
            self.menu.key(key, event)
            return
        if self.screen == "boot":
            self._boot_done()
            return
        if self.screen == "loading":
            if self.loading_ready and self.screen_t > 0.6:
                self._enter_play()
            return
        if self.screen != "play" or self.mode is None:
            return
        m = self.mode
        if getattr(m, "over", False):
            if key in (K.Key_Space, K.Key_Return, K.Key_Enter) and m.over_t > 2.0:
                self._finish_game()
            return
        p = m.player
        if key == K.Key_R:
            if p.start_reload():
                self.sfx.play("reload")
        elif key in (K.Key_1, K.Key_2):
            p.switch(0 if key == K.Key_1 else 1)
        elif key == K.Key_F:
            self.use_tap = True
        elif key == K.Key_V:
            m.knife()
        elif key == K.Key_G:
            m.throw_grenade()
        elif key == K.Key_Q and hasattr(m, "throw_tactical"):
            m.throw_tactical()
        elif key == K.Key_C:
            self.crouch_toggle = not self.crouch_toggle
        elif key in (K.Key_3, K.Key_4, K.Key_5) and hasattr(m, "use_streak"):
            m.use_streak({K.Key_3: 0, K.Key_4: 1, K.Key_5: 2}[key])
        elif key == K.Key_Tab:
            self.show_scores = True
        elif key == K.Key_Space and p.crouched:
            self.crouch_toggle = False

    def key_released(self, key: int, event) -> None:  # noqa: ANN001
        if key == K.Key_Tab:
            self.show_scores = False

    def mouse_pressed(self, button, pos) -> None:  # noqa: ANN001
        if self.menu.active:
            self.menu.click(pos, button)
            return
        if self.screen == "boot":
            self._boot_done()
            return
        if self.screen == "loading":
            if self.loading_ready and self.screen_t > 0.6:
                self._enter_play()
            return
        if self.screen == "play" and self.mode is not None:
            if getattr(self.mode, "over", False):
                if self.mode.over_t > 2.0:
                    self._finish_game()
                return
            if button == Qt.MouseButton.LeftButton:
                self.mouse_fire = True
            elif button == Qt.MouseButton.RightButton:
                self.mouse_ads = True
            if not self.mouse_captured and self.isActiveWindow():
                self.set_mouse_captured(True)

    def mouse_released(self, button, pos) -> None:  # noqa: ANN001
        if button == Qt.MouseButton.LeftButton:
            self.mouse_fire = False
        elif button == Qt.MouseButton.RightButton:
            self.mouse_ads = False

    def mouse_moved(self, pos) -> None:  # noqa: ANN001
        if self.menu.active:
            self.menu.hover(pos)

    def wheel(self, steps: int) -> None:
        if self.menu.active:
            self.menu.wheel(steps)
            return
        if self.playing():
            p = self.mode.player
            if len(p.guns) > 1:
                p.switch((p.cur - steps) % len(p.guns))

    def _controls(self, dt: float) -> Controls:
        keys = self.keys
        c = Controls()
        c.fwd = (1.0 if K.Key_W in keys or K.Key_Up in keys else 0.0) - (1.0 if K.Key_S in keys or K.Key_Down in keys
                                                                         else 0.0)
        c.side = (1.0 if K.Key_D in keys else 0.0) - (1.0 if K.Key_A in keys else 0.0)
        c.sprint = K.Key_Shift in keys
        hold = K.Key_Control in keys
        if c.sprint and c.fwd > 0:
            self.crouch_toggle = False
        c.crouch = self.crouch_toggle or hold
        c.jump = K.Key_Space in keys
        c.fire = self.mouse_fire
        c.ads = self.mouse_ads
        dx, dy = self.take_mouse_delta()  # already times the arcade's sensitivity, y flipped when inverted
        p = self.mode.player
        zoom = 1 + (p.gun.w.zoom - 1) * p.ads if p.gun else 1.0
        ads_mult = 1.0 + (self.settings["ads_sens"] - 1.0) * p.ads
        sens = LOOK_RAD_PER_PX / zoom * ads_mult
        turn_keys = (1.0 if K.Key_Right in keys else 0.0) - (1.0 if K.Key_Left in keys else 0.0)
        c.turn = dx * sens + turn_keys * KEY_TURN * self.key_turn_speed * dt / zoom
        c.look = -dy * sens
        self.sway[0] += (dx * 0.02 - self.sway[0]) * min(dt * 8, 1)
        self.sway[1] += (dy * 0.02 - self.sway[1]) * min(dt * 8, 1)
        self.sway = [max(min(v, 1.5), -1.5) for v in self.sway]
        return c

    # --- flow ---------------------------------------------------------------------------------------------------

    def open_controls(self) -> None:
        """The arcade's Controls panel (it pauses the game and resumes it after; the menu stays where it was)."""
        self.mouse_fire = self.mouse_ads = False
        self.keys.clear()
        self.set_mouse_captured(False)
        self.controls_requested.emit()

    def _boot_done(self) -> None:
        if self.screen == "boot":
            self.screen = "menu"
            self.screen_t = 0.0
            self.menu.open_root()

    def _enter_play(self) -> None:
        self.screen = "play"
        self.screen_t = 0.0
        self.mouse_fire = self.mouse_ads = False
        if self.mode is not None and self.mode.kind == "mp":
            self.sfx.play("spawn")

    def _finish_game(self) -> None:
        m = self.mode
        if m is None:
            return
        if m.kind == "zombies":
            if not getattr(self, "_counted", False):
                self._record_zombies()
                self._counted = True
        self.end_mode()

    def tick(self, dt: float) -> None:
        self.screen_t += dt
        want_capture = self.playing() and self.isActiveWindow() and not getattr(self.mode, "over", False)
        if want_capture != self.mouse_captured and (not want_capture or self.isVisible()):
            self.set_mouse_captured(want_capture)
        if self.rank_popup is not None:
            t, sub, left = self.rank_popup
            self.rank_popup = (t, sub, left - dt) if left > dt else None
        self._update_loops()
        self.menu.tick(dt)
        if self.screen == "boot":
            if self.screen_t > 3.2:
                self._boot_done()
            return
        if self.screen == "loading":
            if self.pending is not None and not self.loading_ready and self.screen_t > 0.05:
                self._build_pending()
            if self.loading_ready and self.screen_t > 4.5:
                self._enter_play()
            return
        if self.screen != "play" or self.mode is None:
            return
        m = self.mode
        if self.menu.active:
            self.take_mouse_delta()
            return
        c = self._controls(dt)
        use = self.use_tap or K.Key_F in self.keys
        self.use_tap = False
        m.update(dt, c, use)
        if self.renderer is not None:
            if getattr(m, "dirty", False):
                self.renderer.refresh()
                m.dirty = False
            if getattr(m, "panes_dirty", False):
                self.renderer.refresh_panes()
                m.panes_dirty = False
        if m.kind == "zombies" and m.over and not getattr(self, "_counted", False):
            self._record_zombies()
            self._counted = True
        if m.kind == "mp":
            for xp in m.take_xp():
                self.add_xp(xp)
            if m.over and not getattr(self, "_counted", False):
                self._counted = True
                m.record(self.mpstats)
                self.progress_changed.emit()

    # --- painting -----------------------------------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        t0 = time.perf_counter()
        p = QPainter(self)
        w, h = self.width(), self.height()
        try:
            if self.screen in ("play",) and self.mode is not None and self.renderer is not None:
                self._paint_play(p, w, h)
            elif self.screen == "loading":
                self.menu.paint_loading(p, w, h, self.screen_t, BRIEFINGS["zombies" if (self.pending or ("",))[0] ==
                                                                          "zombies" else "mp"], self.loading_ready)
            elif self.screen == "boot":
                self.menu.paint_boot(p, w, h, self.screen_t)
            else:
                self.menu.paint_background(p, w, h, self.time)
            if self.menu.active:
                self.menu.paint(p, w, h, self.time)
            if self.rank_popup is not None:
                self._paint_rank_popup(p, w, h)
            if self.paused:
                p.fillRect(self.rect(), QColor(0, 0, 0, 90))
        finally:
            p.end()
        self.paint_ms = (time.perf_counter() - t0) * 1000

    def _view_fov(self) -> float:
        pl = self.mode.player
        gun = pl.gun
        base = math.radians(self.settings["fov"])
        zoom = 1 + ((gun.w.zoom if gun else 1.0) - 1) * pl.ads
        if gun is not None and gun.sight() == "acog":
            zoom = 1 + (2.2 - 1) * pl.ads
        return 2 * math.atan(math.tan(base / 2) / zoom)

    def _paint_play(self, p: QPainter, w: int, h: int) -> None:
        m = self.mode
        r = self.renderer
        pl = m.player
        gun = pl.gun
        # the 3D view: the graphics card draws it at the window's own resolution
        sprites = m.sprites_list()
        lit = m.lights()
        lights = [(x, y, z, c, rad * 2.6, s * 0.9) for x, y, z, c, rad, s in lit]
        if pl.flash_t > 0 and not getattr(m, "over", False):
            ca, sa = math.cos(pl.angle), math.sin(pl.angle)
            lights.append((pl.x + ca * 0.5, pl.y + sa * 0.5, pl.view_eye(), (255, 196, 120), 5.5, 1.3))
        glows = list(lit)
        dpr = float(self.devicePixelRatioF()) if self.isVisible() else 1.0
        torch = TORCH if m.m.scene.get("kind") == "night" else 0.0
        r.fov = self._view_fov()
        r._camera(pl.x, pl.y, pl.view_eye(), pl.angle, pl.look)  # for the eye halos
        glows += r.eye_glows(sprites)
        img = r.render(pl.x, pl.y, pl.view_eye(), pl.angle, pl.look, r.fov, w, h, sprites, glows, lights, torch,
                       self.time, dpr)
        p.drawImage(QRectF(0, 0, w, h), img)
        for e in m.effects:
            if e.kind == "nuke":
                p.fillRect(QRectF(0, 0, w, h), QColor(255, 240, 200, int(255 * (1 - e.t / e.life))))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        tint = (255, 226, 200) if m.kind == "zombies" else (255, 246, 236)
        scoped = gun is not None and gun.sight() == "scope" and pl.ads > 0.85
        muzzle = None
        if gun is not None and not getattr(m, "over", False):
            vs = ViewState(ads=pl.ads, bob=pl.bob, moving=min(pl.moving, 1.2) if pl.z <= 0 else 0.0,
                           sprint=1.0 if pl.sprinting else 0.0, kick=pl.kick, flash=pl.flash_t,
                           reload=(1 - pl.reload_t / pl.reload_total) if pl.reload_t > 0 and pl.reload_total else -1,
                           switch=math.sin(min(pl.switch_t / 0.5, 1.0) * math.pi) if pl.switch_t > 0 else 0.0,
                           time=self.time, sway_x=self.sway[0], sway_y=self.sway[1], camo=gun.upgraded, tint=tint,
                           crouch=1.0 if pl.crouched else 0.0, name=gun.name)
            drink = getattr(m, "drink", None)
            if pl.knife_t > 0 and not drink:
                vs.knife = 1.0
                draw_knife(p, w, h, 1 - pl.knife_t / 0.45, tint)
            elif pl.throw_t > 0:
                vs.throw = 1.0
                draw_throw(p, w, h, 1 - pl.throw_t / 0.5, tint)
            elif drink:
                vs.drink = 1.0
                from .data import ZPERKS

                draw_drink(p, w, h, 1 - drink[1] / 1.6, ZPERKS[drink[0]].color, tint)
            if vs.knife == 0 and vs.throw == 0 and vs.drink == 0:
                muzzle = draw_weapon(p, w, h, gun.w.model, gun.sight(), vs)
            if scoped:
                draw_scope_overlay(p, w, h, self.time, (self.sway[0] * 6, self.sway[1] * 6))
            elif pl.ads > 0.7 and gun.sight() in ("reddot", "acog"):
                draw_red_dot(p, w / 2, h / 2, h, gun.sight() == "acog")
        hud.draw_tracers(p, m.effects, r.project, 1.0, 1.0, muzzle)
        beam = m.light_beam() if hasattr(m, "light_beam") else None
        if beam is not None:
            hud.draw_beam(p, r.project, beam, 1.0, 1.0, self.time)
        if not scoped and pl.ads < 0.5 and not pl.sprinting:
            spread_px = math.tan(pl.spread()) * r.proj
            hud.draw_crosshair(p, w, h, spread_px, 1 - pl.ads * 2)
        hud.draw_hitmarker(p, w, h, m.hit_marker, m.hit_marker_kill)
        hud.draw_damage(p, w, h, pl.health / max(pl.max_health, 1), pl.hurt_t, pl.damage_dirs, pl.angle)
        if m.kind == "zombies":
            hud.draw_zombies_hud(p, w, h, m, self.time)
            if m.over:
                hud.draw_game_over(p, w, h, m, self.time)
        else:
            from . import mphud

            mphud.draw_mp_hud(p, w, h, m, self, self.time)
            if self.show_scores or m.over:
                mphud.draw_scoreboard(p, w, h, m, self)

    def _paint_rank_popup(self, p: QPainter, w: int, h: int) -> None:
        title, sub, left = self.rank_popup
        s = h / 780
        a = int(255 * min(left / 0.5, 1.0, (4.0 - left) / 0.3 + 0.01))
        a = max(min(a, 255), 0)
        r = QRectF(w / 2 - 260 * s, h * 0.24, 520 * s, 84 * s)
        g = QLinearGradient(r.topLeft(), r.topRight())
        g.setColorAt(0, QColor(0, 0, 0, 0))
        g.setColorAt(0.5, QColor(30, 14, 4, int(a * 0.85)))
        g.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(r, g)
        hud.text(p, QRectF(r.x(), r.y() + 6 * s, r.width(), 40 * s), title, 30 * s, QColor(242, 140, 50, a),
                 Qt.AlignmentFlag.AlignHCenter)
        hud.text(p, QRectF(r.x(), r.y() + 46 * s, r.width(), 28 * s), sub, 17 * s, QColor(255, 255, 255, a),
                 Qt.AlignmentFlag.AlignHCenter, False)

    # --- the home card --------------------------------------------------------------------------------------------

    _thumb_cache: dict = {}

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        w, h = int(rect.width()), int(rect.height())
        key = (w, h)
        base = cls._thumb_cache.get(key)
        if base is None:
            base = cls._thumb_cache[key] = _thumb_base(w, h)
        p.drawImage(rect.topLeft(), base)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(rect.topLeft())
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setPen(Qt.PenStyle.NoPen)
        # glowing eyes of the night shift, flickering
        rng = random.Random(4)
        for i in range(5):
            ex = w * (0.08 + 0.16 * i + rng.uniform(-0.02, 0.02))
            ey = h * (0.52 + rng.uniform(-0.04, 0.05))
            flick = 0.6 + 0.4 * math.sin(t * (3 + i) + i * 1.7)
            for dx in (-1, 1):
                g = QRadialGradient(QPointF(ex + dx * w * 0.012, ey), h * 0.025)
                g.setColorAt(0, QColor(160, 255, 180, int(230 * flick)))
                g.setColorAt(1, QColor(40, 200, 90, 0))
                p.setBrush(g)
                p.drawEllipse(QPointF(ex + dx * w * 0.012, ey), h * 0.025, h * 0.025)
        # embers drifting up
        for i in range(14):
            k = (t * 0.12 + i / 14) % 1.0
            x = w * ((i * 0.137 + 0.05 * math.sin(t + i)) % 1.0)
            y = h * (1.0 - k)
            p.setBrush(QColor(255, 140, 40, int(200 * (1 - k))))
            p.drawEllipse(QPointF(x, y), 1.6, 1.6)
        p.restore()


def _thumb_base(w: int, h: int) -> QImage:
    """The static layer of the card art: an orange haze, a town skyline, a soldier and a row of the dead, tally marks."""
    img = QImage(max(w, 1), max(h, 1), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor(8, 6, 6))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    g = QRadialGradient(QPointF(w * 0.62, h * 0.55), w * 0.75)
    g.setColorAt(0, QColor(242, 107, 29))
    g.setColorAt(0.35, QColor(150, 50, 14))
    g.setColorAt(1, QColor(10, 6, 6))
    p.fillRect(QRectF(0, 0, w, h), g)
    # skyline
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(30, 14, 8))
    path = QPainterPath()
    path.moveTo(0, h * 0.62)
    for x0, top in ((0.0, 0.5), (0.12, 0.42), (0.2, 0.5), (0.34, 0.38), (0.46, 0.47), (0.55, 0.4), (0.7, 0.46),
                    (0.82, 0.36), (0.9, 0.44), (1.0, 0.5)):
        path.lineTo(w * x0, h * top)
        path.lineTo(w * (x0 + 0.07), h * top)
    path.lineTo(w, h)
    path.lineTo(0, h)
    path.closeSubpath()
    p.drawPath(path)
    # the horde
    for i in range(5):
        cx = w * (0.08 + 0.16 * i)
        top = h * 0.47
        p.setBrush(QColor(14, 8, 6))
        p.drawEllipse(QPointF(cx, top + h * 0.05), h * 0.05, h * 0.06)
        body = QPainterPath()
        body.moveTo(cx - h * 0.07, h)
        body.lineTo(cx - h * 0.06, top + h * 0.1)
        body.lineTo(cx - h * 0.13, top + h * 0.17)  # arms out
        body.lineTo(cx - h * 0.12, top + h * 0.2)
        body.lineTo(cx, top + h * 0.15)
        body.lineTo(cx + h * 0.12, top + h * 0.2)
        body.lineTo(cx + h * 0.13, top + h * 0.17)
        body.lineTo(cx + h * 0.06, top + h * 0.1)
        body.lineTo(cx + h * 0.07, h)
        body.closeSubpath()
        p.drawPath(body)
    # the soldier, rifle up, in front
    p.setBrush(QColor(4, 3, 3))
    s = h / 100
    sx = w * 0.72
    sol = QPainterPath()
    sol.moveTo(sx - 16 * s, h)
    sol.lineTo(sx - 13 * s, 62 * s)
    sol.lineTo(sx - 15 * s, 44 * s)
    sol.lineTo(sx - 8 * s, 34 * s)
    sol.lineTo(sx + 9 * s, 34 * s)
    sol.lineTo(sx + 14 * s, 44 * s)
    sol.lineTo(sx + 13 * s, 62 * s)
    sol.lineTo(sx + 16 * s, h)
    sol.closeSubpath()
    p.drawPath(sol)
    p.drawEllipse(QPointF(sx, 27 * s), 7.5 * s, 8.5 * s)
    p.drawChord(QRectF(sx - 9 * s, 17 * s, 18 * s, 14 * s), 0, 180 * 16)
    p.save()
    p.translate(sx - 4 * s, 45 * s)
    p.rotate(-28)
    p.drawRect(QRectF(-34 * s, -2.5 * s, 46 * s, 5 * s))
    p.drawRect(QRectF(-4 * s, 2 * s, 5 * s, 9 * s))
    p.restore()
    # red tally marks
    p.setPen(QPen(QColor(236, 186, 40, 230), max(h * 0.02, 2), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for i in range(4):
        x = w * 0.06 + i * h * 0.05
        p.drawLine(QPointF(x, h * 0.07), QPointF(x - h * 0.01, h * 0.25))
    p.drawLine(QPointF(w * 0.06 - h * 0.03, h * 0.21), QPointF(w * 0.06 + h * 0.19, h * 0.1))
    # title
    f = hud.hfont(h * 0.15, True, "Impact")
    p.setFont(f)
    p.setPen(QColor(0, 0, 0, 160))
    tr = QRectF(0, h * 0.74, w * 0.96, h * 0.2)
    p.drawText(tr.translated(2, 2), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "SHADOW OPS II")
    p.setPen(QColor(255, 236, 220))
    p.drawText(tr, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "SHADOW OPS II")
    p.fillRect(QRectF(w * 0.55, h * 0.93, w * 0.41, max(h * 0.012, 1)), QColor(242, 107, 29))
    p.end()
    return img
