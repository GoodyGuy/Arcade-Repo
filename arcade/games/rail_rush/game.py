"""Rail Rush: the arcade game widget - title screen, runs, the revive offer, game over, shop, missions and stats.
The 3D view is drawn by the graphics card (``scene.py`` / ``render.py``) at the window's own resolution; the HUD and
the menus are painted over it."""

from __future__ import annotations

import math
import random
import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPolygonF, QRadialGradient

from app.arcade.base import ArcadeGame
from app.arcade.games.rail_rush import art, sounds, ui
from app.arcade.games.rail_rush.art import NOPEN, Pose, qc
from app.arcade.games.rail_rush.data import (BOX_PRICE, CHARACTERS, HOVERBOARD_PRICE, KEY_PRICE, UPGRADE_COSTS,
                                             UPGRADE_LEVELS, Profile, character, mission_text)
from app.arcade.games.rail_rush.scene import Scene
from app.arcade.games.rail_rush.world import FLY_H, World

K = Qt.Key
MENU_STATES = ("title", "shop", "missions", "stats", "over", "box")
READY_TIME = 3.0  # s of "3 - 2 - 1" after a revive


def box_reward(rng: random.Random) -> tuple[str, int, str]:
    """A lost parcel: (kind, amount, label)."""
    roll = rng.random()
    if roll < 0.03:
        return "coins", 5000, "JACKPOT! 5,000 coins"
    if roll < 0.2:
        n = rng.choice((1, 1, 2))
        return "hoverboard", n, f"{n} shield{'s' if n > 1 else ''}"
    if roll < 0.3:
        return "key", 1, "1 heart"
    n = rng.choice((150, 250, 250, 400, 500, 750, 1000))
    return "coins", n, f"{n:,} coins"


class RailRushGame(ArcadeGame):
    """An endless runner down a freight line (see the module docstring)."""

    game_id = "rail_rush"
    title = "Rail Rush"
    subtitle = "Endless freight-line runner"
    description = ("Sprint a parcel down three tracks of a busy freight line: dodge parked wagons and oncoming "
                   "locomotives, leap barriers, roll under gantries, ride balloons and spring soles, grab coins - and "
                   "keep ahead of the yard's security drone.")
    controls = (("Left / Right, A / D", "Change lane"), ("Up / W / Space", "Jump"),
                ("Down / S", "Roll / drop fast"), ("H", "Bubble shield (1 crash)"),
                ("Mouse drag", "Swipe to move"), ("Enter / arrows", "Menus"), ("Esc", "Pause / back"))
    accent = "#ffb21f"
    # rebindable in the arcade's Controls panel (the base class maps a rebound key back to these defaults)
    bindings = (("Left", Qt.Key.Key_Left), ("Right", Qt.Key.Key_Right), ("Jump", Qt.Key.Key_Up),
                ("Roll", Qt.Key.Key_Down), ("Shield", Qt.Key.Key_H))

    SAVE_TIME = 4.5

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.profile = Profile()
        self.scene = Scene()
        self.ready_t = 0.0
        self.cam_pitch = 0.0
        sounds.build(self.sfx)
        self.rng = random.Random()
        self.state = "title"
        self.world = World(self.rng.randrange(1 << 30))
        self.anim_t = 0.0
        self.buttons: list[ui.Button] = []
        self.focus = 0
        self.toasts: list[dict] = []
        self.shop_tab = "upgrades"
        self.shop_from = "title"
        self.box: dict = {}
        self.saveme_t = 0.0
        self.saves_used = 0
        self.hs_banner = 0.0
        self.hs_shown = False
        self.run_best = 0
        self._banked = (0, 0.0)
        self.last_run = {"score": 0, "coins": 0, "distance": 0, "mult": 1, "new_best": False}
        self.particles: list[list] = []
        self.shake = 0.0
        self.dying_t = 0.0
        self.lane_count = 0
        self.cam_x, self.cam_y, self.cam_back = 0.0, 1.3, 5.5
        self.chaser_y = 0.0
        self._coin_snd = 0.0
        self._swipe: QPointF | None = None
        self._swiped = False
        self._music = False
        self._jet = False
        self._frame_ms = 0.0

    @property
    def char(self) -> dict:
        return character(self.profile.selected)

    # --- save / load ------------------------------------------------------------------------------------------------

    def save_state(self) -> dict:
        if self.state in ("run", "dying", "saveme", "ready"):
            self._bank_run()
        return self.profile.to_json()

    def _bank_run(self) -> None:
        """Move what the current run has earned so far into the saved stats (only the part not banked yet)."""
        wd, pr = self.world, self.profile
        st = pr.stats
        dist = int(wd.distance) - self._banked[0]
        st["distance"] += dist
        st["play_seconds"] += wd.run_time - self._banked[1]
        self._banked = (self._banked[0] + dist, wd.run_time)
        st["best_distance"] = max(st["best_distance"], int(wd.distance))
        st["best_coins"] = max(st["best_coins"], wd.run_coins)
        pr.best = max(pr.best, int(wd.score))

    def load_state(self, data: dict) -> None:
        self.profile = Profile.from_json(data or {})
        self._new_world()

    @classmethod
    def summary(cls, data: dict) -> str:
        if not data:
            return "New game"
        pr = Profile.from_json(data)
        return f"Best {pr.best:,} · {pr.coins:,} coins · x{pr.multiplier}"

    # --- flow -------------------------------------------------------------------------------------------------------

    def _new_world(self) -> None:
        self.world = World(self.rng.randrange(1 << 30), self.profile.upgrades)
        self.world.multiplier = self.profile.multiplier
        self.particles.clear()
        self.cam_x, self.cam_y, self.cam_back, self.cam_pitch = 0.0, 1.3, 5.5, 0.0

    def stop(self) -> None:
        super().stop()
        self.scene.close()

    def _set_state(self, state: str) -> None:
        self.state = state
        self.focus = 0
        self.buttons = []
        self.update()

    def play(self) -> None:
        if self.world.running:
            self._new_world()
        self.world.upgrades = dict(self.profile.upgrades)
        self.world.multiplier = self.profile.multiplier
        self.profile.new_run()
        self.world.start()
        self.saves_used = 0
        self._banked = (0, 0.0)
        self.run_best = self.profile.best
        self.hs_shown = False
        self.hs_banner = 0.0
        self.lane_count = 0
        self._set_state("run")
        self.sfx.play("alarm")
        self.sfx.play("siren", 0.8)
        self._music_on(True)

    def _music_on(self, on: bool) -> None:
        if on and not self._music:
            self.sfx.play("music")
        elif not on and self._music:
            self.sfx.stop("music")
        self._music = on

    def _jet_on(self, on: bool) -> None:
        if on and not self._jet:
            self.sfx.play("jetpack")
        elif not on and self._jet:
            self.sfx.stop("jetpack")
        self._jet = on

    @property
    def save_cost(self) -> int:
        return 2 ** min(self.saves_used, 10)

    def save_me(self) -> None:
        if self.state != "saveme":
            return
        cost = self.save_cost
        if self.profile.keys < cost:
            self.sfx.play("deny")
            return
        self.profile.keys -= cost
        self.saves_used += 1
        self.world.revive()
        self.ready_t = READY_TIME
        self._set_state("ready")
        self.sfx.play("key")
        self.progress_changed.emit()

    def end_run(self) -> None:
        wd = self.world
        pr = self.profile
        st = pr.stats
        score = int(wd.score)
        new_best = score > self.run_best
        self._bank_run()
        st["runs"] += 1
        self.last_run = {"score": score, "coins": wd.run_coins, "distance": int(wd.distance),
                         "mult": wd.multiplier, "new_best": new_best}
        self._jet_on(False)
        self._music_on(False)
        self._set_state("over")
        self.progress_changed.emit()

    def to_title(self) -> None:
        self._new_world()
        self._set_state("title")

    def activate(self, action: str) -> None:
        """A button (or its key) was used."""
        pr = self.profile
        self.sfx.play("click")
        if action == "play" or action == "again":
            self.play()
        elif action == "shop":
            self.shop_from = self.state
            self._set_state("shop")
        elif action == "missions":
            self._set_state("missions")
        elif action == "stats":
            self._set_state("stats")
        elif action == "arcade":
            self.exit_requested.emit()
        elif action == "controls":
            self.controls_requested.emit()
        elif action == "back":
            if self.state == "shop" and self.shop_from == "over":
                self._set_state("over")
            else:
                self._set_state("title")
        elif action == "menu":
            self.to_title()
        elif action.startswith("tab:"):
            self.shop_tab = action[4:]
            focus = self.focus
            self._set_state("shop")
            self.focus = focus
        elif action.startswith("up:"):
            kind = action[3:]
            lvl = pr.upgrades.get(kind, 0)
            if lvl < UPGRADE_LEVELS and pr.coins >= UPGRADE_COSTS[lvl]:
                pr.coins -= UPGRADE_COSTS[lvl]
                pr.upgrades[kind] = lvl + 1
                self.sfx.play("buy")
                self.progress_changed.emit()
            else:
                self.sfx.play("deny")
        elif action.startswith("buy:"):
            self._buy(action[4:])
        elif action.startswith("char:"):
            cid = action[5:]
            ch = character(cid)
            if cid in pr.unlocked:
                pr.selected = cid
                self.sfx.play("buy")
            elif pr.coins >= ch["price"]:
                pr.coins -= ch["price"]
                pr.unlocked.append(cid)
                pr.selected = cid
                self.sfx.play("powerup")
                self.toast(f"{ch['name']} joined the courier crew!", "#35a853")
            else:
                self.sfx.play("deny")
            self.progress_changed.emit()
        elif action == "saveme":
            self.save_me()
        elif action == "skip":
            if self.state == "saveme":
                self.end_run()
        elif action == "box_close":
            if self.box.get("t", 0) > 1.3:
                self.box = {}
                self._set_state("shop")

    def _buy(self, item: str) -> None:
        pr = self.profile
        price = {"hoverboard": HOVERBOARD_PRICE, "key": KEY_PRICE, "mystery": BOX_PRICE}.get(item)
        if price is None or pr.coins < price:
            self.sfx.play("deny")
            return
        pr.coins -= price
        if item == "hoverboard":
            pr.hoverboards += 1
            self.sfx.play("buy")
        elif item == "key":
            pr.keys += 1
            self.sfx.play("key")
        else:
            kind, n, label = box_reward(self.rng)
            self._grant(kind, n)
            pr.stats["boxes"] += 1
            self.box = {"t": 0.0, "kind": kind, "label": label}
            self._set_state("box")
            self.sfx.play("rattle")
        self.progress_changed.emit()

    def _grant(self, kind: str, n: int) -> None:
        pr = self.profile
        if kind == "coins":
            pr.coins += n
        elif kind == "hoverboard":
            pr.hoverboards += n
        elif kind == "key":
            pr.keys += n

    def toast(self, text: str, color: str = "#2c6fd8", dur: float = 2.4) -> None:
        self.toasts.append({"text": text, "t": dur, "dur": dur, "color": color})
        del self.toasts[:-3]

    # --- input ------------------------------------------------------------------------------------------------------

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001
        st = self.state
        if st == "run":
            wd = self.world
            if key in (K.Key_Left, K.Key_A):
                wd.move(-1)
            elif key in (K.Key_Right, K.Key_D):
                wd.move(1)
            elif key in (K.Key_Up, K.Key_W, K.Key_Space):
                wd.jump()
            elif key in (K.Key_Down, K.Key_S):
                wd.duck()
            elif key == K.Key_H:
                self.use_hoverboard()
            return
        if st == "saveme":
            if key in (K.Key_Space, K.Key_Return, K.Key_Enter, K.Key_K):
                self.save_me()
            elif key in (K.Key_N, K.Key_Backspace, K.Key_Delete):
                self.end_run()
            return
        if st in ("dying", "ready"):
            return
        if st == "box":
            if key in (K.Key_Space, K.Key_Return, K.Key_Enter):
                self.activate("box_close")
            return
        if key == K.Key_Space and st in ("title", "over"):
            self.activate("play")
            return
        if key == K.Key_M and st == "over":
            self.activate("menu")
            return
        if key == K.Key_C and st in ("title", "over"):
            self.activate("controls")
            return
        if key == K.Key_Backspace and st in ("shop", "missions", "stats"):
            self.activate("back")
            return
        n = len(self.buttons)
        if key in (K.Key_Left, K.Key_Up, K.Key_A, K.Key_W) and n:
            self.focus = (self.focus - 1) % n
            self.sfx.play("click", 0.5)
        elif key in (K.Key_Right, K.Key_Down, K.Key_D, K.Key_S, K.Key_Tab) and n:
            self.focus = (self.focus + 1) % n
            self.sfx.play("click", 0.5)
        elif key in (K.Key_Return, K.Key_Enter) and n:
            b = self.buttons[self.focus % n]
            if b.enabled:
                self.activate(b.action)
            else:
                self.sfx.play("deny")
        self.update()

    def use_hoverboard(self) -> None:
        if self.profile.hoverboards <= 0 or self.world.board > 0:
            if self.profile.hoverboards <= 0:
                self.toast("No shields left - buy some in the shop", "#7a1212", 1.6)
            return
        if self.world.use_board():
            self.profile.hoverboards -= 1

    def handle_escape(self) -> bool:
        if self.state in ("shop", "missions", "stats"):
            self.activate("back")
            return True
        if self.state == "box":
            self.activate("box_close")
            return True
        if self.state == "over":
            self.to_title()
            return True
        return False

    def mouse_pressed(self, button: Qt.MouseButton, pos: QPointF) -> None:
        if button != Qt.MouseButton.LeftButton:
            return
        if self.state in ("run",):
            self._swipe = QPointF(pos)
            self._swiped = False
            return
        for i, b in enumerate(self.buttons):
            if b.rect.contains(pos):
                self.focus = i
                if b.enabled:
                    self.activate(b.action)
                else:
                    self.sfx.play("deny")
                return

    def mouse_moved(self, pos: QPointF) -> None:
        if self.state == "run":
            if self._swipe is not None and not self._swiped:
                d = pos - self._swipe
                lim = self.swipe_threshold()
                if abs(d.x()) > lim or abs(d.y()) > lim:
                    self._swiped = True
                    if abs(d.x()) > abs(d.y()):
                        self.world.move(1 if d.x() > 0 else -1)
                    elif d.y() < 0:
                        self.world.jump()
                    else:
                        self.world.duck()
            return
        for i, b in enumerate(self.buttons):
            if b.rect.contains(pos) and self.focus != i:
                self.focus = i
                self.update()
                return

    def swipe_threshold(self) -> float:
        """Pixels a mouse drag must travel to count as a swipe (shorter with a higher mouse sensitivity)."""
        base = max(24.0, min(self.width(), self.height()) * 0.045)
        return max(6.0, base / max(self.mouse_sensitivity, 0.1))

    def mouse_released(self, button: Qt.MouseButton, pos: QPointF) -> None:
        self._swipe = None

    def on_pause(self) -> None:
        self.sfx.stop("music")
        self.sfx.stop("jetpack")
        self._music = False
        self._jet = False
        self._swipe = None

    def on_resume(self) -> None:
        if self.state in ("run", "ready"):
            self._music_on(True)
            self._jet_on(self.world.flying)

    # --- the game loop ----------------------------------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        self.anim_t += dt
        for toast in self.toasts:
            toast["t"] -= dt
        self.toasts[:] = [t for t in self.toasts if t["t"] > 0]
        self.hs_banner = max(0.0, self.hs_banner - dt)
        self._coin_snd = max(0.0, self._coin_snd - dt)
        wd = self.world
        st = self.state
        if st == "run":
            wd.update(dt)
            self._events()
            pr = self.profile
            for done in pr.advance("score_run", value=wd.score) + pr.advance("distance_run", value=wd.distance):
                self._mission_done(done)
            if not self.hs_shown and self.run_best > 0 and wd.score > self.run_best:
                self.hs_shown = True
                self.hs_banner = 2.8
                self.sfx.play("highscore")
                self.progress_changed.emit()
            if wd.dead:
                self.dying_t = 0.0
                self._jet_on(False)
                self._music_on(False)
                self._set_state("dying")
        elif st == "dying":
            wd.update(dt)
            self._events()
            self.dying_t += dt
            if self.dying_t > 1.25:
                self.saveme_t = self.SAVE_TIME if self.profile.keys >= self.save_cost else 2.5
                self._set_state("saveme")
        elif st == "ready":
            before = math.ceil(self.ready_t)
            self.ready_t -= dt
            if math.ceil(self.ready_t) != before:
                self.sfx.play("click" if self.ready_t > 0 else "whistle")
            if self.ready_t <= 0:
                self._set_state("run")
                self._music_on(True)
        elif st == "saveme":
            self.saveme_t -= dt
            if self.saveme_t <= 0:
                self.end_run()
        elif st == "box":
            self.box["t"] = self.box.get("t", 0.0) + dt
            if self.box["t"] - dt < 1.3 <= self.box["t"]:
                self.sfx.play("pop")
        elif st == "title":
            wd.update(dt)
        self._camera(dt)
        self._particles(dt)
        self.shake = max(0.0, self.shake - dt * 2.5)

    def _camera(self, dt: float) -> None:
        wd = self.world
        k = 1.0 - math.exp(-dt * 6.0)
        if self.state == "title" or not wd.running:
            tx, ty, tb, tp = 0.0, 1.3, 5.5, 0.0
        else:
            ref = wd.y * 0.8 if wd.flying or wd.y > FLY_H * 0.5 else wd.ground
            if not wd.flying and not wd.grounded:
                ref = min(wd.y, max(wd.ground, self.cam_y - 4.4))
            tx, ty, tb, tp = wd.x * 0.75, 4.1 + ref * 0.92, 6.3, 0.22
            if wd.flying:
                ty, tb, tp = 3.2 + ref, 7.2, 0.12
        self.cam_x += (tx - self.cam_x) * k
        self.cam_y += (ty - self.cam_y) * (1.0 - math.exp(-dt * 3.0))
        self.cam_back += (tb - self.cam_back) * (1.0 - math.exp(-dt * 2.5))
        self.cam_pitch += (tp - self.cam_pitch) * (1.0 - math.exp(-dt * 2.5))
        gy = 0.0 if wd.flying else wd.ground
        self.chaser_y += (gy - self.chaser_y) * min(1.0, dt * 5)

    def _mission_done(self, m: dict) -> None:
        self.toast(f"Mission complete: {mission_text(m)}", "#7b3fd9")
        self.sfx.play("mission")
        pr = self.profile
        if pr.set_complete():
            pr.next_set()
            self.world.multiplier = pr.multiplier
            self.toast(f"Score multiplier x{pr.multiplier}!", "#f0801a", 3.0)
        self.progress_changed.emit()

    def _events(self) -> None:
        wd = self.world
        pr = self.profile
        st = pr.stats
        done: list[dict] = []
        for ev in wd.events:
            name = ev[0]
            if name == "coin":
                pr.coins += 1
                st["coins"] += 1
                done += pr.advance("coins") + pr.advance("coins_run", value=wd.run_coins)
                if ev[1]:
                    done += pr.advance("magnet_coins")
                if self._coin_snd <= 0:
                    self.sfx.play("coin")
                    self._coin_snd = 0.05
                self._spark(wd.x, wd.y + 1.0, wd.pz + 0.5, "#fff36b", 3)
            elif name == "jump":
                st["jumps"] += 1
                done += pr.advance("jumps")
                self.sfx.play("sjump" if ev[1] else "jump")
            elif name == "roll":
                st["rolls"] += 1
                done += pr.advance("rolls")
                self.sfx.play("roll")
            elif name == "lane":
                self.lane_count += 1
                done += pr.advance("lanes_run", value=self.lane_count)
                self.sfx.play("lane")
            elif name == "land":
                if ev[1]:
                    self.sfx.play("land")
                    self._dust(wd.x, wd.ground, wd.pz)
            elif name == "roof":
                done += pr.advance("roofs")
            elif name == "barrier":
                done += pr.advance("barriers")
            elif name == "bump":
                self.sfx.play("land")
                self.shake = max(self.shake, 0.2)
            elif name == "stumble":
                st["stumbles"] += 1
                done += pr.advance("stumbles")
                self.shake = max(self.shake, 0.5)
                self.sfx.play("stumble")
                self.sfx.play("alarm", 0.8)
                self.sfx.play("siren", 0.7)
            elif name == "crash":
                st["crashes"] += 1
                self.shake = 1.0
                self.sfx.play("crash")
                self.sfx.play("alarm", 0.7)
            elif name == "caught":
                st["crashes"] += 1
                self.shake = 0.6
                self.sfx.play("stumble")
                self.sfx.play("alarm")
                self.sfx.play("siren")
            elif name == "pickup":
                kind = ev[1]
                if kind in ("jetpack", "sneakers", "magnet", "double"):
                    st["powerups"] += 1
                    done += pr.advance("powerups")
                    self.sfx.play("powerup")
                    if kind == "jetpack":
                        st["jetpacks"] += 1
                        done += pr.advance("jetpack")
                        self._jet_on(True)
                    self._spark(wd.x, wd.y + 1.0, wd.pz + 0.5, "#9ff3ff", 10)
                elif kind == "mystery":
                    k2, n, label = box_reward(self.rng)
                    self._grant(k2, n)
                    st["boxes"] += 1
                    self.toast(f"Lost parcel: {label}!", "#b0702c")
                    self.sfx.play("pop")
                    self._spark(wd.x, wd.y + 1.0, wd.pz + 0.5, "#d9a8ff", 12)
                elif kind == "key":
                    pr.keys += 1
                    st["keys_found"] += 1
                    self.toast("You found a heart!", "#d92b5b")
                    self.sfx.play("key")
                    self._spark(wd.x, wd.y + 1.0, wd.pz + 0.5, "#9ff3ff", 12)
                self.progress_changed.emit()
            elif name == "power_end":
                if ev[1] == "jetpack":
                    self._jet_on(False)
            elif name == "board":
                st["hoverboards"] += 1
                done += pr.advance("hoverboard")
                self.sfx.play("board")
                self._spark(wd.x, wd.y + 0.3, wd.pz + 0.3, "#7ffcff", 12)
            elif name == "board_break":
                self.shake = 0.7
                self.sfx.play("board_break")
                self.toast("Shield popped - you're still running!", "#1f6fd6", 1.8)
                for _ in range(14):
                    self._spark(wd.x, wd.y + 0.3, wd.pz + 0.6, self.rng.choice(("#ffd23f", "#e8323c", "#7ffcff")), 1,
                                kind="shard")
            elif name == "revive":
                for _ in range(26):
                    self._spark(wd.x, wd.y + 1.0, wd.pz + 1.0, self.rng.choice(art.PAINTS), 1, kind="shard")
        wd.events.clear()
        for m in done:
            self._mission_done(m)

    # --- particles ---

    def _spark(self, x: float, y: float, z: float, color: str, n: int, kind: str = "spark") -> None:
        sp = self.world.speed if self.world.running else 0.0
        for _ in range(n):
            self.particles.append([x, y, z, self.rng.uniform(-2.5, 2.5), self.rng.uniform(1.0, 5.0),
                                   sp + self.rng.uniform(-1.5, 3.0), 0.0, self.rng.uniform(0.35, 0.7), kind, color])
        del self.particles[:-90]

    def _dust(self, x: float, y: float, z: float) -> None:
        sp = self.world.speed
        for i in range(8):
            a = i / 8 * math.tau
            self.particles.append([x + math.cos(a) * 0.3, y + 0.1, z, math.cos(a) * 2.0, 0.6, sp * 0.8, 0.0, 0.45,
                                   "dust", "#d9cdb8"])

    def _particles(self, dt: float) -> None:
        wd = self.world
        if wd.flying and self.state == "run" and self.rng.random() < 0.5:
            self.particles.append([wd.x + self.rng.choice((-0.1, 0.1)), wd.y + 0.45, wd.pz - 0.3,
                                   self.rng.uniform(-0.3, 0.3), -2.5, wd.speed * 0.85, 0.0, 0.35, "smoke", "#ffffff"])
        keep = []
        for pt in self.particles:
            pt[6] += dt
            if pt[6] >= pt[7]:
                continue
            pt[0] += pt[3] * dt
            pt[1] += pt[4] * dt
            pt[2] += pt[5] * dt
            if pt[8] in ("spark", "shard"):
                pt[4] -= 12 * dt
            keep.append(pt)
        self.particles = keep

    # --- drawing ----------------------------------------------------------------------------------------------------

    # --- drawing ----------------------------------------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        p = QPainter(self)
        t0 = time.perf_counter()
        try:
            self._paint(p)
        finally:
            p.end()
        self._frame_ms = (time.perf_counter() - t0) * 1000.0

    def _paint(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        if w < 40 or h < 40:
            p.fillRect(self.rect(), QColor("#2f9bf0"))
            return
        wd = self.world
        dpr = self.devicePixelRatioF()
        img = self.scene.render(self, max(int(w * dpr), 16), max(int(h * dpr), 16))
        if img is None:
            p.fillRect(self.rect(), QColor("#1d2a44"))
            ui.outlined(p, QRectF(0, h * 0.4, w, h * 0.2), Qt.AlignmentFlag.AlignCenter,
                        "Rail Rush needs OpenGL 3.3 graphics", h * 0.04, "#ffb21f")
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, img.width() != int(w * dpr))
        p.drawImage(QRectF(0, 0, w, h), img)
        p.restore()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        if wd.flying:
            g = QLinearGradient(0, 0, 0, h)
            g.setColorAt(0, qc("#ffffff", 0))
            g.setColorAt(1, qc("#ffd9f0", 34))
            p.fillRect(QRectF(0, 0, w, h), g)
        self.buttons = []
        st = self.state
        if st in ("run", "dying", "ready"):
            ui.hud(self, p)
            if st == "dying" and self.dying_t > 0.15:
                k = min(1.0, (self.dying_t - 0.15) * 3)
                text = "CRASH!" if wd.death == "crash" else "CAUGHT!"
                p.setOpacity(k)
                ui.outlined(p, QRectF(0, h * 0.32, w, h * 0.12), Qt.AlignmentFlag.AlignCenter, text,
                            h * 0.1 * (0.8 + 0.2 * k), "#ff6b5b", ow=4)
                p.setOpacity(1.0)
            elif st == "ready":
                n = max(1, math.ceil(self.ready_t))
                f = self.ready_t - math.floor(self.ready_t)
                size = h * (0.16 + 0.06 * f)
                ui.outlined(p, QRectF(0, h * 0.3, w, h * 0.2), Qt.AlignmentFlag.AlignCenter, str(n), size,
                            "#fff36b", ow=5)
                ui.outlined(p, QRectF(0, h * 0.48, w, h * 0.06), Qt.AlignmentFlag.AlignCenter, "GET READY",
                            h * 0.035, ow=2)
        elif st == "saveme":
            ui.hud(self, p)
            ui.save_me(self, p)
        elif st == "over":
            ui.game_over(self, p)
        elif st == "title":
            ui.title(self, p)
        elif st == "shop":
            ui.shop(self, p)
        elif st == "box":
            ui.box_overlay(self, p)
        elif st == "missions":
            ui.missions(self, p)
        elif st == "stats":
            ui.stats(self, p)
        ui.toasts(self, p)
        if self.buttons:
            self.focus = min(self.focus, len(self.buttons) - 1)
        if self.paused and st in ("run", "ready"):
            p.fillRect(QRectF(0, 0, w, h), qc("#000000", 60))

    # --- the home screen card ---------------------------------------------------------------------------------------

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        p.save()
        p.setClipRect(rect)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x0, y0, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        hy = y0 + h * 0.42
        g = QLinearGradient(0, y0, 0, hy)
        g.setColorAt(0, QColor("#2f86e0"))
        g.setColorAt(0.75, QColor("#9ad2f7"))
        g.setColorAt(1, QColor("#e8f4fb"))
        p.fillRect(QRectF(x0, y0, w, hy - y0), g)
        sun = QRadialGradient(x0 + w * 0.84, y0 + h * 0.13, h * 0.2)
        sun.setColorAt(0, qc("#fffbe0"))
        sun.setColorAt(0.2, qc("#fff3b0", 200))
        sun.setColorAt(1, qc("#ffffff", 0))
        p.setPen(NOPEN)
        p.setBrush(sun)
        p.drawEllipse(QPointF(x0 + w * 0.84, y0 + h * 0.13), h * 0.2, h * 0.2)
        # far hills and pines
        for k, (col, amp, base) in enumerate((("#8fb6c4", 0.07, 0.0), ("#5f9e6a", 0.05, 0.02))):
            poly = QPolygonF([QPointF(x0, hy)])
            for i in range(25):
                xx = x0 + w * i / 24
                poly.append(QPointF(xx, hy - h * (base + amp * (0.6 + 0.4 * math.sin(i * 1.7 + k * 2.1)))))
            poly.append(QPointF(x0 + w, hy))
            p.setBrush(QColor(col))
            p.drawPolygon(poly)
        p.fillRect(QRectF(x0, hy, w, y0 + h - hy), QColor("#6fa64e"))
        cx = x0 + w / 2
        f = h * 0.62
        cam_y = 4.2

        def pt(x: float, y: float, d: float) -> QPointF:
            return QPointF(cx + x * f / d, hy - (y - cam_y) * f / d)

        def quad(pts) -> QPolygonF:  # noqa: ANN001
            return QPolygonF([pt(*q) for q in pts])

        p.setBrush(QColor("#8a7f70"))
        p.drawPolygon(quad(((-4.3, 0, 1.2), (4.3, 0, 1.2), (4.3, 0, 300), (-4.3, 0, 300))))
        off = (t * 16) % 1.5
        p.setBrush(QColor("#5e4532"))
        for k in range(24, 0, -1):
            d = 1.5 + k * 1.5 - off
            if d < 2.5:
                continue
            sc = f / d
            th = max(1.0, 0.28 * cam_y * sc / d)
            yy = hy + cam_y * sc - th / 2
            for lx in (-2.6, 0.0, 2.6):
                p.drawRect(QRectF(cx + (lx - 1.15) * sc, yy, 2.3 * sc, th))
        p.setBrush(QColor("#d9dee4"))
        for lx in (-2.6, 0.0, 2.6):
            for rx in (lx - 0.72, lx + 0.72):
                p.drawPolygon(quad(((rx - 0.06, 0.15, 2.5), (rx + 0.06, 0.15, 2.5), (rx + 0.06, 0.15, 300),
                                    (rx - 0.06, 0.15, 300))))
        # a parked boxcar on the left, an oncoming locomotive (lights on) on the right
        for lx, d, body, front in ((-2.6, 7.5, "#a8432e", None), (2.6, 30.0 - (t * 6) % 22, "#e9e4d8", "#f2c12e")):
            xs = lx + (1.15 if lx < 0 else -1.15)
            p.setBrush(QColor(body).darker(118))
            p.drawPolygon(quad(((xs, 0.9, d), (xs, 3.5, d), (xs, 3.5, d + 40), (xs, 0.9, d + 40))))
            p.setBrush(QColor("#2a2c31"))
            p.drawPolygon(quad(((xs, 0.2, d), (xs, 0.9, d), (xs, 0.9, d + 40), (xs, 0.2, d + 40))))
            tl, br = pt(lx - 1.15, 3.5, d), pt(lx + 1.15, 0.9, d)
            p.setBrush(QColor(body))
            p.drawRect(QRectF(tl, br))
            if front:
                s = f / d
                p.setBrush(QColor("#1c2a38"))
                p.drawRect(QRectF(pt(lx - 1.0, 3.3, d), pt(lx + 1.0, 2.6, d)))
                p.setBrush(QColor(front))
                p.drawRect(QRectF(pt(lx - 1.1, 2.2, d), pt(lx + 1.1, 1.0, d)))
                for lxx in (lx - 0.78, lx + 0.78):
                    c = pt(lxx, 1.6, d)
                    gl = QRadialGradient(c, 1.2 * s)
                    gl.setColorAt(0, qc("#ffffff", 230))
                    gl.setColorAt(0.25, qc("#fff2a8", 140))
                    gl.setColorAt(1, qc("#fff2a8", 0))
                    p.setBrush(gl)
                    p.drawEllipse(c, 1.2 * s, 1.2 * s)
            else:
                p.setPen(QColor(body).darker(150))
                for i in range(1, 8):
                    xx = lx - 1.15 + i * 2.3 / 8
                    p.drawLine(pt(xx, 3.5, d), pt(xx, 0.9, d))
                p.setPen(NOPEN)
        # the jump: an arc of coins and the runner in the air over a barrier
        phase = (t * 0.9) % 1.0
        hop = math.sin(phase * math.pi)
        tl, br = pt(-1.1, 1.0, 9.0), pt(1.1, 0.0, 9.0)
        p.setBrush(QColor("#9a9a94"))
        p.drawRect(QRectF(pt(-1.1, 0.4, 9.0), br))
        p.setBrush(QColor("#f4f4f4"))
        p.drawRect(QRectF(tl, pt(1.1, 0.4, 9.0)))
        p.setBrush(QColor("#e23b2e"))
        for i in range(4):
            a = pt(-1.1 + i * 0.6, 1.0, 9.0)
            p.drawRect(QRectF(a.x(), a.y(), 0.3 * f / 9.0, 0.6 * f / 9.0))
        for i in range(7):
            u = i / 6
            d = 6.0 + u * 7.0
            c = pt(0, 0.95 + 1.9 * 4 * u * (1 - u), d)
            art.draw_coin(p, c.x(), c.y(), 0.34 * f / d, t * 3 + i)
        d_h = 3.9
        s = f / d_h
        feet = pt(0.0, 0.0, d_h)
        p.setBrush(qc("#000000", 70))
        p.drawEllipse(feet, 0.45 * s * (1 - 0.4 * hop), 0.14 * s * (1 - 0.4 * hop))
        pose = Pose(mode="jump", phase=t * 12, t=t, rising=phase < 0.5)
        art.draw_runner(p, feet.x(), feet.y() - hop * 1.6 * s, s, CHARACTERS[0], pose)
        # the yard drone on the chase
        c = pt(-1.6, 2.6 + 0.15 * math.sin(t * 2), 3.4)
        art.draw_drone(p, c.x(), c.y(), f / 3.4, t)
        art.draw_logo(p, cx, y0 + h * 0.15, h * 0.16, t)
        p.restore()
