"""STORM DROP: the arcade's third-person battle royale. This module is the game widget: the title screens, a match
(the 3D view drawn by the graphics card at the window's own resolution, the HUD over it with QPainter), the player's
controls and camera, saving, and the home-screen card."""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter

from app.arcade.base import ArcadeGame

from . import hud, menus
from .data import (CANOPIES, DIFFICULTY, OUTFITS, SLOTS, WEAPONS, default_options, default_stats, unlocked)
from .render import Camera, Env, Renderer
from .scene import Scene, _Fake, ghost_row
from .sounds import build as build_sounds

K = Qt.Key
LOOK = 0.0023  # rad per pixel of mouse at the arcade's sensitivity 1.0
KEY_TURN = 2.3  # rad / s with the arrow keys at turn speed 1.0
MENUS = ("menu", "locker", "help", "options", "stats")


class StormDropGame(ArcadeGame):
    game_id = "storm_drop"
    title = "STORM DROP"
    subtitle = "Battle royale"
    description = ("Ride the drop ship over Brightwater Isle, pick your landing, loot chests, harvest wood, brick and "
                   "metal, and throw up walls and ramps in the middle of a fight. The storm closes ring by ring: be "
                   "the last one standing against the bots.")
    controls = (("W A S D", "Move"), ("Mouse", "Look / aim"), ("LMB", "Fire / place / use"),
                ("RMB", "Aim  (building: material)"), ("Space", "Jump / leave the ship"), ("Shift", "Sprint"),
                ("Ctrl", "Crouch"), ("E", "Open / pick up / swap"), ("R", "Reload"), ("F", "Sledge (harvest)"),
                ("1 - 5 / Wheel", "Inventory"), ("Z  X  C  V", "Wall, floor, ramp, roof"), ("B", "Change material"),
                ("T", "Drop the held item"), ("M / Tab", "Map (leave match)"), ("← → ↑ ↓", "Turn / look"),
                ("Esc", "Arcade menu"))
    accent = "#ff8a3d"
    uses_mouse_look = False  # the game takes and frees the mouse itself (menus need the pointer)
    uses_key_turn = True
    bindings = (("Move forward", K.Key_W), ("Move back", K.Key_S), ("Move left", K.Key_A), ("Move right", K.Key_D),
                ("Jump / leave the ship", K.Key_Space), ("Sprint", K.Key_Shift), ("Crouch", K.Key_Control),
                ("Open / pick up", K.Key_E), ("Reload", K.Key_R), ("Sledge", K.Key_F), ("Build wall", K.Key_Z),
                ("Build floor", K.Key_X), ("Build ramp", K.Key_C), ("Build roof", K.Key_V),
                ("Change material", K.Key_B), ("Drop item", K.Key_T), ("Map", K.Key_M), ("Slot 1", K.Key_1),
                ("Slot 2", K.Key_2), ("Slot 3", K.Key_3), ("Slot 4", K.Key_4), ("Slot 5", K.Key_5),
                ("Turn left", K.Key_Left), ("Turn right", K.Key_Right), ("Look up", K.Key_Up),
                ("Look down", K.Key_Down))

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        build_sounds(self.sfx)
        self.options = default_options()
        self.stats = default_stats()
        self.outfit = 0
        self.canopy = 0
        self.screen = "boot"
        self._boot = 0
        self.island = None
        self.renderer: Renderer | None = None
        self.scene: Scene | None = None
        self.cam = Camera()
        self.env = Env()
        self.ui = menus.Ui()
        self.match = None
        self.result: dict | None = None
        self.map_image: QImage | None = None
        self.yaw = self.pitch = 0.0
        self.cam_dist = 4.3
        self.fov = 80.0
        self.scoped = False
        self.build_piece: int | None = None
        self.preview = None
        self.map_open = False
        self.fire_held = self.fire_edge = self.alt_held = False
        self.last_built = None
        self.aim_enemy = False
        self.spectate = None
        self.out_t = -1.0
        self.end_t = -1.0
        self.gl_error = ""
        self._frame_dt = 0.0
        self._wind = False
        self._in_storm = 0.0
        self._hero = None
        self._bound = "none"
        self.autoplay = False  # tests and demos: a bot plays the player's part
        self.debug_seed: int | None = None

    # --- saving -------------------------------------------------------------------------------------------------------

    def save_state(self) -> dict:
        return {"v": 1, "stats": dict(self.stats), "options": dict(self.options), "outfit": OUTFITS[self.outfit].id,
                "canopy": CANOPIES[self.canopy][0]}

    def load_state(self, data: dict) -> None:
        data = data if isinstance(data, dict) else {}
        stats = default_stats()
        saved = data.get("stats") if isinstance(data.get("stats"), dict) else {}
        for key, base in stats.items():
            v = saved.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v >= 0:
                stats[key] = float(v) if isinstance(base, float) else int(v)
        self.stats = stats
        opts = default_options()
        saved = data.get("options") if isinstance(data.get("options"), dict) else {}
        for key, base in opts.items():
            v = saved.get(key)
            if isinstance(base, bool):
                if isinstance(v, bool):
                    opts[key] = v
            elif isinstance(base, int):
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
                    opts[key] = int(v)
            elif isinstance(v, str):
                opts[key] = v
        opts["bots"] = min(max(opts["bots"], 16), 32)
        opts["fov"] = min(max(opts["fov"], 60), 100)
        if opts["difficulty"] not in DIFFICULTY:
            opts["difficulty"] = "normal"
        self.options = opts
        ids = [o.id for o in OUTFITS]
        self.outfit = ids.index(data["outfit"]) if data.get("outfit") in ids else 0
        if not unlocked(OUTFITS[self.outfit].need, self.stats):
            self.outfit = 0
        names = [c[0] for c in CANOPIES]
        self.canopy = names.index(data["canopy"]) if data.get("canopy") in names else 0
        if not unlocked(CANOPIES[self.canopy][1], self.stats):
            self.canopy = 0

    @classmethod
    def summary(cls, data: dict) -> str:
        s = data.get("stats") if isinstance(data, dict) and isinstance(data.get("stats"), dict) else {}
        if not s.get("matches"):
            return "New game"
        wins, kills = int(s.get("wins", 0)), int(s.get("kills", 0))
        return f"{wins} win{'' if wins == 1 else 's'} · {kills} eliminations"

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        menus.thumbnail(p, rect, t, cls.accent)

    # --- the world ------------------------------------------------------------------------------------------------------

    def _make_world(self) -> None:
        from .island import Island

        self.island = Island.get()
        self.renderer = Renderer()
        self.scene = Scene(self.renderer, self.island)
        rgbm = self.island.map_rgb
        self.map_image = QImage(rgbm.data, rgbm.shape[1], rgbm.shape[0], rgbm.strides[0],
                                QImage.Format.Format_RGB888).copy()
        quay = next(p for p in self.island.places if p.kind == "quay")
        self._hero_at = ((quay.gx + 8) * 4.0, quay.gy * 3.0 + 0.06, (quay.gz - 7.3) * 4.0)
        self._bind_menu()

    def _bind_menu(self) -> None:
        if self._bound != "menu":
            self.scene.bind(None)
            self._bound = "menu"
        self.match = None

    def stop(self) -> None:
        super().stop()
        self.sfx.stop("wind")
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            self.scene = None
            self._bound = "none"
            self.screen = "boot"
            self._boot = 0

    def on_pause(self) -> None:
        self.sfx.stop("wind")
        self._wind = False
        self.fire_held = self.alt_held = False

    def on_resume(self) -> None:
        if self.screen == "match" and not self.map_open:
            self.set_mouse_captured(True)

    def handle_escape(self) -> bool:
        if self.gl_error:
            self.exit_requested.emit()
            return True
        if self.screen in ("locker", "help", "options", "stats", "result"):
            self._do("back")
            return True
        if self.screen == "match" and self.map_open:
            self._set_map(False)
            return True
        return False

    # --- flow -----------------------------------------------------------------------------------------------------------

    def _go(self, screen: str) -> None:
        self.screen = screen
        self.ui.focus = 0
        self.ui.by_keys = True
        self.ui.hot = []

    def _do(self, action: str) -> None:
        if not action:
            return
        sfx = self.sfx
        if action == "play" or action == "again":
            sfx.play("menu_select")
            self.start_match()
        elif action in ("locker", "help", "options", "stats"):
            sfx.play("menu_select")
            if self.screen == "result":
                self._bind_menu()
            self._go(action)
        elif action == "back":
            sfx.play("menu_back")
            if self.screen in ("result", "match"):
                self._bind_menu()
            self._go("menu")
            self.progress_changed.emit()
        elif action == "quit":
            self.exit_requested.emit()
        elif action == "controls":
            self.controls_requested.emit()
        elif action == "leave":
            self._leave_match()
        elif action.startswith("outfit:"):
            i = int(action[7:])
            if unlocked(OUTFITS[i].need, self.stats):
                self.outfit = i
                sfx.play("menu_select")
                self.progress_changed.emit()
            else:
                sfx.play("deny")
        elif action.startswith("canopy:"):
            i = int(action[7:])
            if unlocked(CANOPIES[i][1], self.stats):
                self.canopy = i
                sfx.play("menu_select")
                self.progress_changed.emit()
            else:
                sfx.play("deny")
        elif action.startswith("opt:"):
            _o, key, d = action.split(":")
            self._option(key, int(d))

    def _option(self, key: str, d: int) -> None:
        for k, _label, kind, choices in menus.OPTION_ROWS:
            if k != key:
                continue
            v = self.options[key]
            if kind == "bool":
                self.options[key] = not v
            elif kind == "choice":
                self.options[key] = choices[(choices.index(v) + d) % len(choices)]
            else:
                lo, hi, step = choices
                self.options[key] = min(max(v + d * step, lo), hi)
            self.sfx.play("menu_move")
            self.progress_changed.emit()

    def start_match(self) -> None:
        from .world import Match

        if self.renderer is None:
            self._make_world()
        seed = self.debug_seed if self.debug_seed is not None else random.randrange(1 << 30)
        self.match = Match(self.island, self.options, self.outfit, self.canopy, "You", seed=seed,
                           autoplay=self.autoplay)
        self.scene.bind(self.match)
        self._bound = "match"
        self.yaw, self.pitch = self.match.ship_yaw, -0.45
        self.cam_dist = 30.0
        self.build_piece, self.preview, self.last_built = None, None, None
        self.map_open = False
        self.fire_held = self.fire_edge = self.alt_held = False
        self.spectate = None
        self.out_t = self.end_t = -1.0
        self.result = None
        self._in_storm = 0.0
        self.screen = "match"
        self.ui.hot = []
        self.set_mouse_captured(True)

    def _set_map(self, on: bool) -> None:
        self.map_open = on
        self.fire_held = self.alt_held = False
        self.set_mouse_captured(not on)
        self.ui.by_keys = False

    def _leave_match(self) -> None:
        m = self.match
        if m is not None and m.player.alive and not m.over:
            m.eliminate(m.player, None, "left")
        self._finish_match()

    def _finish_match(self) -> None:
        m = self.match
        me = m.player
        if me.alive and not m.over:
            return
        win = m.winner is me
        place = 1 if win else max(me.place, 1)
        secs = (m.over_t if win and m.over_t > 0 else (me.died_t if me.died_t > 0 else m.time))
        before = self._unlocked_names()
        s = self.stats
        s["matches"] += 1
        s["wins"] += 1 if win else 0
        s["deaths"] += 0 if win else 1
        s["kills"] += me.kills
        s["top5"] += 1 if place <= 5 else 0
        s["best_kills"] = max(s["best_kills"], me.kills)
        s["damage"] += int(me.damage)
        s["built"] += me.stats["built"]
        s["harvested"] += me.stats["harvested"]
        s["chests"] += me.stats["chests"]
        s["headshots"] += me.stats["headshots"]
        s["longest"] = max(s["longest"], round(me.stats["longest"], 1))
        s["seconds"] = round(s["seconds"] + secs, 1)
        s["best_place"] = place if not s["best_place"] else min(s["best_place"], place)
        new = [n for n in self._unlocked_names() if n not in before]
        by = me.attacker.name if (not win and me.attacker is not None) else ""
        self.result = {"place": place, "total": m.total, "kills": me.kills, "damage": me.damage, "seconds": secs,
                       "built": me.stats["built"], "harvested": me.stats["harvested"], "chests": me.stats["chests"],
                       "headshots": me.stats["headshots"], "longest": me.stats["longest"], "unlocked": new, "by": by}
        self.sfx.stop("wind")
        self._wind = False
        self.set_mouse_captured(False)
        self.map_open = False
        self._go("result")
        if new:
            self.sfx.play("unlock")
        self.progress_changed.emit()

    def _unlocked_names(self) -> list[str]:
        out = [o.name for o in OUTFITS if unlocked(o.need, self.stats)]
        out += [c[0] + " canopy" for c in CANOPIES if unlocked(c[1], self.stats)]
        return out

    def view_actor(self):  # noqa: ANN201 - whose eyes the camera follows
        m = self.match
        me = m.player
        if me.alive:
            return me
        s = self.spectate
        if s is None or not s.alive:
            nxt = s.attacker if (s is not None and s.attacker is not None and s.attacker.alive) else None
            if nxt is None and me.attacker is not None and me.attacker.alive and s is None:
                nxt = me.attacker
            if nxt is None:
                ref = s or me
                alive = [a for a in m.actors if a.alive]
                nxt = min(alive, key=lambda a: (a.x - ref.x) ** 2 + (a.z - ref.z) ** 2) if alive else None
            self.spectate = s = nxt
        return s or me

    # --- the loop -------------------------------------------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        self._frame_dt += dt
        if self.screen == "boot":
            self._boot += 1
            if self._boot >= 2:
                self._make_world()
                self._go("menu")
            return
        if self.screen == "match":
            self._tick_match(dt)

    def _tick_match(self, dt: float) -> None:
        m = self.match
        me = m.player
        keys = self.keys
        mdx, mdy = self.take_mouse_delta()
        zoom = max(self.options["fov"] / max(self.fov, 1.0), 1.0)
        turn = KEY_TURN * self.key_turn_speed * dt / zoom
        self.yaw += mdx * LOOK / zoom + ((K.Key_Right in keys) - (K.Key_Left in keys)) * turn
        self.pitch -= mdy * LOOK / zoom - ((K.Key_Up in keys) - (K.Key_Down in keys)) * turn * 0.7
        self.pitch = min(max(self.pitch, -1.45), 1.45)
        self.preview = None
        if me.alive and not me.bot:
            f = (K.Key_W in keys) - (K.Key_S in keys)
            r = (K.Key_D in keys) - (K.Key_A in keys)
            sy, cy = math.sin(self.yaw), math.cos(self.yaw)
            wx, wz = sy * f + cy * r, cy * f - sy * r
            n = math.hypot(wx, wz)
            me.wish_x, me.wish_z = (wx / n, wz / n) if n > 0 else (0.0, 0.0)
            me.sprint = K.Key_Shift in keys and f > 0
            me.crouch = K.Key_Control in keys
            me.yaw, me.pitch = self.yaw, self.pitch
            if K.Key_Space in keys and me.on_ground:
                me.want_jump = True
            w = me.weapon()
            building = self.build_piece is not None and me.state == "ground"
            me.ads = self.alt_held and not building and w is not None and w.cls != "tool" and not self.map_open
            self._aim()
            if building:
                tgt = m.build_target(me, self.build_piece, self.yaw, self.pitch)
                ok = m.structs.can_place(*tgt) and me.mats[me.build_mat] >= 10
                self.preview = ghost_row(*tgt, ok)
                me.firing = False
                if self.fire_held and (self.fire_edge or (self.options["hold_to_build"] and tgt != self.last_built)):
                    if m.try_build(me, *tgt):
                        self.last_built = tgt
                    elif self.fire_edge:
                        self.sfx.play("deny", 0.6)
            else:
                me.firing = self.fire_held and not self.map_open
        elif me.alive:  # a bot is playing for the player (tests, demos)
            d = (me.yaw - self.yaw + math.pi) % math.tau - math.pi
            self.yaw += d * min(1.0, 4.0 * dt)
            self.pitch += (-0.18 - self.pitch) * min(1.0, 3.0 * dt)
        self.fire_edge = False
        m.listener = self.cam.eye
        m.step(dt)
        self._camera(dt)
        if m.sounds:
            best: dict[str, float] = {}
            for name, vol in m.sounds:
                if vol > best.get(name, 0.0):
                    best[name] = vol
            for name, vol in list(best.items())[:10]:
                self.sfx.play(name, vol)
            m.sounds.clear()
        falling = me.alive and me.state in ("fall", "glide")
        if falling != self._wind:
            self._wind = falling
            if falling:
                self.sfx.play("wind")
            else:
                self.sfx.stop("wind")
        # out, or won
        if not me.alive and self.out_t < 0:
            self.out_t = self.time
            self.build_piece = None
            self.sfx.play("lose")
        if m.over and self.end_t < 0:
            self.end_t = self.time
            if m.winner is me:
                self.sfx.play("win")
        if self.end_t > 0 and self.time - self.end_t > 7.0:
            self._finish_match()

    def _aim(self) -> None:
        """Where the middle of the screen points: the player's shots go there from the muzzle."""
        m = self.match
        me = m.player
        ex, ey, ez = self.cam.eye
        cp = math.cos(self.pitch)
        fx, fy, fz = math.sin(self.yaw) * cp, math.sin(self.pitch), math.cos(self.yaw) * cp
        t0 = max((me.x - ex) * fx + (me.y + 1.5 - ey) * fy + (me.z - ez) * fz, 0.0)
        sx, sy, sz = ex + fx * t0, ey + fy * t0, ez + fz * t0
        w = me.weapon()
        reach = max(w.reach if w is not None else 4.0, 8.0)
        t, _what, _obj = m.structs.ray(sx, sy, sz, fx, fy, fz, reach)
        v, tv, _head = m._ray_actors(sx, sy, sz, fx, fy, fz, t, me)
        self.aim_enemy = v is not None
        if v is not None:
            t = tv
        me.aim = (sx + fx * t, sy + fy * t, sz + fz * t)

    def _camera(self, dt: float) -> None:
        m = self.match
        a = self.view_actor()
        me = m.player
        cam = self.cam
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        sy, cy = math.sin(self.yaw), math.cos(self.yaw)
        fwd = (sy * cp, sp, cy * cp)
        right = (cy, 0.0, -sy)
        want_fov = float(self.options["fov"])
        self.scoped = False
        shoulder = 0.0
        if a.state == "ship":
            sx = m.ship_a[0] + m.ship_dir[0] * m.ship_s
            sz = m.ship_a[1] + m.ship_dir[1] * m.ship_s
            pivot = (sx, a.y + 16.0, sz)
            want = 52.0
        elif a.state != "ground":
            pivot = (a.x, a.y + 1.6, a.z)
            want = 9.0
        else:
            pivot = (a.x, a.y + (1.3 if a.crouch else 1.72), a.z)
            want, shoulder = 4.3, 0.72
            if a is me and me.ads:
                w = me.weapon()
                zoom = w.zoom if w is not None else 1.0
                want_fov /= zoom
                want, shoulder = 2.4, 0.58
                if zoom > 2.5:
                    want, shoulder = 0.0, 0.0
                    self.scoped = abs(self.fov - want_fov) < want_fov * 0.25
        self.fov += (want_fov - self.fov) * min(1.0, 14.0 * dt)
        py = pivot[1] + 0.2
        if shoulder > 0.0:  # the shoulder the camera looks over must not be inside a wall
            t, what, _obj = m.structs.ray(pivot[0], py, pivot[2], right[0], 0.0, right[2], shoulder + 0.3)
            if what is not None:
                shoulder = max(t - 0.3, 0.0)
        px = pivot[0] + right[0] * shoulder
        pz = pivot[2] + right[2] * shoulder
        dist = want
        if want > 0.1 and a.state != "ship":
            t, what, _obj = m.structs.ray(px, py, pz, -fwd[0], -fwd[1], -fwd[2], want + 0.4)
            if what is not None:
                dist = max(t - 0.4, 0.2)
        if dist < self.cam_dist:
            self.cam_dist = dist
        else:
            self.cam_dist += (dist - self.cam_dist) * min(1.0, 6.0 * dt)
        d = self.cam_dist
        ex, ey, ez = px - fwd[0] * d, py - fwd[1] * d, pz - fwd[2] * d
        if want <= 0.1:
            ex, ey, ez = a.x + fwd[0] * 0.2, a.y + 1.58, a.z + fwd[2] * 0.2
        g = m.island.height(ex, ez)
        if ey < g + 0.25:
            ey = g + 0.25
        cam.eye = (ex, ey, ez)
        cam.yaw, cam.pitch, cam.fov = self.yaw, self.pitch, self.fov
        out = 1.0 if (m.storm["state"] != "idle" and m.storm_dist(ex, ez) > 0) else 0.0
        self._in_storm += (out - self._in_storm) * min(1.0, 3.0 * dt)

    # --- input ----------------------------------------------------------------------------------------------------------

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001
        if self.gl_error:
            return
        if self.screen == "match" and not self.map_open:
            self._match_key(key)
            return
        ui = self.ui
        if self.screen == "match":  # the big map is open
            if key in (K.Key_M, K.Key_Tab):
                self._set_map(False)
            elif key in (K.Key_Return, K.Key_Enter) and ui.by_keys:
                self._do(ui.current())
            return
        if key in (K.Key_Up, K.Key_W, K.Key_Down, K.Key_S):
            d = -1 if key in (K.Key_Up, K.Key_W) else 1
            moved = ui.move(d)
            if self.screen == "options" and ui.current().endswith(":1" if d > 0 else ":-1")                     and ui.current().startswith("opt:"):
                moved = ui.move(d) or moved  # one row of the list at a time, not one arrow button
            if moved:
                self.sfx.play("menu_move")
        elif key in (K.Key_Left, K.Key_A, K.Key_Right, K.Key_D):
            d = -1 if key in (K.Key_Left, K.Key_A) else 1
            cur = ui.current()
            if self.screen == "options" and cur.startswith("opt:"):
                self._option(cur.split(":")[1], d)
            elif ui.move(d):
                self.sfx.play("menu_move")
        elif key in (K.Key_Return, K.Key_Enter, K.Key_Space):
            ui.by_keys = True
            self._do(ui.current())

    def _match_key(self, key: int) -> None:
        m = self.match
        me = m.player
        if key in (K.Key_M, K.Key_Tab):
            self._set_map(True)
            return
        if not me.alive or m.over:
            if key in (K.Key_Return, K.Key_Enter):
                self._finish_match()
            return
        if me.bot:
            return
        if key == K.Key_Space:
            if me.state == "ship":
                m.leave_ship(me)
            else:
                me.want_jump = True
        elif key == K.Key_E:
            m.interact(me)
        elif key == K.Key_R:
            m.reload(me)
        elif key == K.Key_F:
            self.build_piece = None
            m.select(me, -1)
        elif K.Key_1 <= key <= K.Key_5:
            self.build_piece = None
            m.select(me, key - K.Key_1)
        elif key in (K.Key_Z, K.Key_X, K.Key_C, K.Key_V):
            i = (K.Key_Z, K.Key_X, K.Key_C, K.Key_V).index(key)
            self.build_piece = None if self.build_piece == i else i
            self.last_built = None
            self.sfx.play("switch", 0.7)
        elif key == K.Key_B:
            self._next_material()
        elif key == K.Key_T:
            if me.sel >= 0 and me.inv[me.sel] is not None:
                m.drop_slot(me, me.sel)
                self.sfx.play("switch", 0.7)

    def _next_material(self) -> None:
        me = self.match.player
        for i in range(1, 4):
            k = (me.build_mat + i) % 3
            if me.mats[k] >= 10 or i == 3:
                me.build_mat = k
                break
        self.sfx.play("menu_move", 0.7)

    def mouse_pressed(self, button: Qt.MouseButton, pos: QPointF) -> None:
        if self.gl_error:
            return
        if self.screen != "match" or self.map_open:
            if button == Qt.MouseButton.LeftButton:
                self.ui.by_keys = False
                self.ui.mouse = QPointF(pos)
                self._do(self.ui.at(pos))
            return
        if not self.mouse_captured:
            self.set_mouse_captured(True)
        m = self.match
        me = m.player
        if not me.alive:
            if button == Qt.MouseButton.LeftButton:  # watch someone else
                alive = [a for a in m.actors if a.alive]
                if alive:
                    cur = self.spectate
                    i = alive.index(cur) if cur in alive else -1
                    self.spectate = alive[(i + 1) % len(alive)]
            return
        if button == Qt.MouseButton.LeftButton:
            self.fire_held = self.fire_edge = True
        elif button == Qt.MouseButton.RightButton:
            if self.build_piece is not None:
                self._next_material()
            else:
                self.alt_held = True

    def mouse_released(self, button: Qt.MouseButton, pos: QPointF) -> None:
        if button == Qt.MouseButton.LeftButton:
            self.fire_held = False
        elif button == Qt.MouseButton.RightButton:
            self.alt_held = False

    def mouse_moved(self, pos: QPointF) -> None:
        if self.screen != "match" or self.map_open:
            if (pos - self.ui.mouse).manhattanLength() > 1.5:
                self.ui.by_keys = False
            self.ui.mouse = QPointF(pos)

    def wheel(self, steps: int) -> None:
        if self.screen != "match" or self.map_open:
            return
        me = self.match.player
        if not me.alive or me.bot:
            return
        if self.build_piece is not None:
            self.build_piece = (self.build_piece - steps) % 4
            self.last_built = None
            return
        order = [-1] + [i for i in range(SLOTS) if me.inv[i] is not None]
        i = order.index(me.sel) if me.sel in order else 0
        self.match.select(me, order[(i - steps) % len(order)])

    # --- drawing --------------------------------------------------------------------------------------------------------

    def _hero_view(self, w: float, h: float) -> list:
        """The camera and the stand-in player for the screens outside a match."""
        x, y, z = self._hero_at
        t = self.time
        close = self.screen == "locker"
        hero = self._hero
        if hero is None:
            hero = self._hero = _Fake(self.outfit, x, y, z, math.pi)
        hero.outfit = self.outfit
        hero.yaw = math.pi - 0.25 + 0.2 * math.sin(t * 0.5) + (t * 0.9 if close else 0.0)
        hero.swing_t = math.floor(t / 4.0) * 4.0 + 2.2 if not close else -9.0
        dist = 4.2 if close else 5.6
        off = 1.25 if close else 1.9
        ex, ey, ez = x - off + 0.25 * math.sin(t * 0.21), y + 1.55, z - dist
        cam = self.cam
        cam.eye = (ex, ey, ez)
        cam.yaw = 0.035 * math.sin(t * 0.3)
        cam.pitch = -0.04
        cam.fov = 62.0
        return [hero]

    def _view_image(self, w: int, h: int) -> QImage:
        dt = self._frame_dt
        self._frame_dt = 0.0
        env = self.env
        cam = self.cam
        scene = self.scene
        m = self.match
        if m is not None and self.screen in ("match", "result"):
            if self.screen == "result":  # a slow turn high over the island
                ang = self.time * 0.08
                cam.eye = (256.0 - math.sin(ang) * 330.0, 170.0, 256.0 - math.cos(ang) * 330.0)
                cam.yaw, cam.pitch, cam.fov = ang, -0.5, 60.0
            env.time = m.time
            s = m.storm
            env.storm = (s["cx"], s["cz"], s["r"])
            env.in_storm = self._in_storm
            hide = m.player if (self.scoped and self.screen == "match") else None
            solid, clear, add = scene.frame(dt, m.time, hide=hide, preview=self.preview)
        else:
            extra = self._hero_view(w, h)
            env.time = self.time
            env.storm = (256.0, 256.0, 9999.0)
            env.in_storm = 0.0
            solid, clear, add = scene.frame(dt, self.time, extra=extra)
        cam.update(float(w), float(h))
        dpr = float(self.devicePixelRatioF()) if self.isVisible() else 1.0
        return self.renderer.render(int(w * dpr), int(h * dpr), cam, env, solid, clear, add)

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        p = QPainter(self)
        w, h = self.width(), self.height()
        if self.gl_error:
            p.fillRect(self.rect(), QColor(16, 20, 40))
            self.draw_center_text(p, "Storm Drop needs a graphics card with OpenGL 3.3", h * 0.42, 24)
            self.draw_center_text(p, self.gl_error[:110], h * 0.5, 13, "#c9d2e6", bold=False)
            self.draw_center_text(p, "Esc: back to the arcade", h * 0.58, 16, "#ffe14d")
            p.end()
            return
        if self.screen == "boot" or self.renderer is None:
            menus.splash(p, w, h, self.time)
            p.end()
            return
        try:
            img = self._view_image(w, h)
        except RuntimeError as exc:
            import os

            if os.environ.get("TDS_ARCADE_STRICT"):
                raise
            self.gl_error = str(exc)
            self.set_mouse_captured(False)
            p.end()
            self.update()
            return
        p.drawImage(QRectF(0, 0, w, h), img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        ui = self.ui
        ui.begin()
        screen = self.screen
        if screen == "match":
            m = self.match
            hud.draw_hud(p, self)
            if not m.player.alive or m.over:
                menus.draw_out(p, self)
        elif screen == "menu":
            menus.draw_menu(p, self)
        elif screen == "locker":
            menus.draw_locker(p, self)
        elif screen == "help":
            menus.draw_help(p, self)
        elif screen == "options":
            menus.draw_options(p, self)
        elif screen == "stats":
            menus.draw_stats(p, self)
        elif screen == "result":
            menus.draw_result(p, self)
        if ui.hot and ui.focus >= len(ui.hot):
            ui.focus = 0
        p.end()


__all__ = ["StormDropGame", "WEAPONS"]
