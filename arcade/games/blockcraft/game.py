"""BlockCraft: a first-person voxel sandbox for the Break Room Arcade. Worlds (survival or creative) are endless,
generated from a seed around the player and saved as a compact diff; the graphics card draws them (render.py).

The game widget is put together from mixins: screens.py (menus, the HUD, the container screens), book.py (recipes,
advancements), life.py (growth, farming, beds, hamlets, the other realms), extras.py (chests, breeding, riding,
weather, the rune bench, signals) and play.py (the world session and its tick)."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter

from app.arcade.base import ArcadeGame
from app.arcade.games.blockcraft import blocks as B
from app.arcade.games.blockcraft import gui as G
from app.arcade.games.blockcraft import items as I
from app.arcade.games.blockcraft import sounds
from app.arcade.games.blockcraft.chunkpool import ChunkPool
from app.arcade.games.blockcraft.book import BookMixin
from app.arcade.games.blockcraft.extras import ExtrasMixin
from app.arcade.games.blockcraft.life import LifeMixin
from app.arcade.games.blockcraft.play import PlayMixin
from app.arcade.games.blockcraft.render import Env, Renderer, layers
from app.arcade.games.blockcraft.screens import MAX_CHUNKS, MIN_CHUNKS, PANO_CHUNKS, SPLASHES, ScreensMixin
from app.arcade.games.blockcraft.world import World

PANO_SEED = 20261119  # the world turning behind the title screen
PARALLEL = True  # build the world with helper processes (tests turn this off: helper threads)
DEFAULT_OPTS = {"fov": 70, "chunks": 10, "fancy": True, "smooth": True, "res": "auto", "aa": True, "shaders": False,
                "bob": True, "music": True, "gui": 0, "gamma": 0.5, "particles": True}
RES_MODES = ("auto", "full", "half")
SAVE_FORMAT = 2  # 1: the first BlockCraft's walled worlds (their terrain is made anew)
CONTAINERS = ("inventory", "crafting", "furnace", "chest", "runes")  # overlays with slots to click
_thumb_cache: dict = {}
log = logging.getLogger(__name__)


class BlockCraftGame(ScreensMixin, BookMixin, LifeMixin, ExtrasMixin, PlayMixin, ArcadeGame):
    """The game widget: menus, a world being played, and its saved worlds."""

    game_id = "blockcraft"
    title = "BlockCraft"
    subtitle = "Mine, craft, build and survive the night"
    description = ("A first-person block sandbox: punch trees, craft tools, mine for diamonds, build anything, and "
                   "survive the zombies of the night - or fly free in creative mode.")
    controls = (("WASD", "Walk"), ("Space", "Jump (double: fly)"), ("Shift", "Sneak / fly down"),
                ("Ctrl / WW", "Sprint"), ("Mouse", "Look"), ("Left click", "Break / attack"),
                ("Right click", "Place / use / eat / trade / feed / ride"), ("Middle click", "Pick block"),
                ("1-9 / wheel", "Hotbar"),
                ("E", "Inventory"), ("R / L", "Recipes / goals"), ("Q", "Drop item"), ("F3 / F1", "Debug / hide HUD"), ("Esc", "Game menu"))
    accent = "#6aa83c"
    uses_mouse_look = False
    # the keys the arcade's Controls panel may rebind (the game always sees these default keys)
    bindings = (("Walk forward", Qt.Key.Key_W), ("Walk backward", Qt.Key.Key_S), ("Strafe left", Qt.Key.Key_A),
                ("Strafe right", Qt.Key.Key_D), ("Jump / fly up", Qt.Key.Key_Space),
                ("Sneak / fly down", Qt.Key.Key_Shift), ("Sprint", Qt.Key.Key_Control),
                ("Inventory", Qt.Key.Key_E), ("Recipe book", Qt.Key.Key_R), ("Advancements", Qt.Key.Key_L),
                ("Drop item", Qt.Key.Key_Q), ("Debug screen", Qt.Key.Key_F3),
                ("Hide HUD", Qt.Key.Key_F1))

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.sfx = sounds.Bank()  # recorded takes of each sound, the synthesised ones where a file is missing
        self.opts = dict(DEFAULT_OPTS)
        self.worlds: list[dict] = []
        self.screen = "splash"
        self.overlay: str | None = None
        self.options_back = "title"
        self.sel_world = -1
        self.list_scroll = 0.0
        self.new_name = "New World"
        self.new_seed = ""
        self.new_mode = "survival"
        self.focus_field: str | None = None
        self.mouse_gui = (0.0, 0.0)
        self.mouse_buttons = 0
        self.drag_slider: str | None = None
        self.cursor: list | None = None
        self.grid: list = [None] * 9
        self.creative_scroll = 0.0
        self.furnace_pos = None
        self.hide_hud = False
        self.show_debug = False
        self.fps = 0.0
        self.frame_ms = 30.0
        self._fps_frames = 0
        self._fps_t0 = time.perf_counter()
        self._last_paint = 0.0
        self.renderer = Renderer()
        self.pool: ChunkPool | None = None  # the helpers that build the world (chunkpool.py)
        self.title_world = None  # the world turning behind the title screen
        self.title_spawn = (0.0, 80.0, 0.0)
        self.title_since = 0.0
        self.load_progress = 0.0
        self.splash = SPLASHES[0]
        self._last_click = (0.0, None)
        self._loader = self._loading()
        self._reset_session()
        self.env = Env(0.2, 0.0)

    # --- arcade hooks -------------------------------------------------------------------------------------------

    def save_state(self) -> dict:
        if self.screen == "game":
            self.store_world()
        return {"v": 1, "opts": dict(self.opts), "worlds": [dict(r) for r in self.worlds]}

    def load_state(self, data: dict) -> None:
        data = data if isinstance(data, dict) else {}
        opts = data.get("opts")
        if isinstance(opts, dict):
            for k, v in DEFAULT_OPTS.items():
                if k in opts and isinstance(opts[k], type(v)) or k in opts and isinstance(v, float) and \
                        isinstance(opts[k], int):
                    self.opts[k] = opts[k]
        self.opts["fov"] = int(min(max(self.opts["fov"], 30), 110))
        self.opts["chunks"] = int(min(max(self.opts["chunks"], MIN_CHUNKS), MAX_CHUNKS))
        if self.opts["res"] not in RES_MODES:
            self.opts["res"] = "auto"
        self.worlds = []
        for r in data.get("worlds", []) or []:
            rec = _clean_record(r)
            if rec is not None:
                self.worlds.append(rec)
        self.worlds.sort(key=lambda r: -int(r.get("last", 0)))

    def handle_escape(self) -> bool:
        if self.screen == "game":
            if self.overlay in CONTAINERS or self.overlay in ("recipes", "advancements"):
                self.close_container()
            elif self.overlay == "options":
                self.overlay = "menu"
            elif self.overlay == "menu":
                self.overlay = None
                self.capture_if_playing()
            elif self.overlay is None:
                self.open_menu()
            return True
        if self.screen in ("select", "options"):
            self.screen = "title" if self.screen == "select" else self.options_back
            return True
        if self.screen in ("create", "delete"):
            self.screen = "select"
            return True
        return self.screen == "generating"

    def on_pause(self) -> None:
        self.mouse_buttons = 0
        self.drag_slider = None
        if self.screen == "game" and self.overlay is None:
            self.overlay = "menu"
        self.sfx.stop_all()

    def on_resume(self) -> None:
        self.capture_if_playing()

    def start_helpers(self) -> None:
        """Start the helpers that build the world (they get ready while the splash screen shows)."""
        if self.pool is None:
            self.pool = ChunkPool(processes=None if PARALLEL else False)

    def stop(self) -> None:
        super().stop()
        self.title_world = None
        self.renderer.close()
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    @classmethod
    def summary(cls, data: dict) -> str:
        worlds = [w for w in (data or {}).get("worlds", []) or [] if isinstance(w, dict)]
        if not worlds:
            return "New game"
        last = max(worlds, key=lambda w: int(w.get("last", 0)) if isinstance(w.get("last", 0), (int, float)) else 0)
        mode = "Creative" if last.get("mode") == "creative" else "Survival"
        try:
            day = int(float(last.get("time", 0)) // 1200) + 1
        except (TypeError, ValueError):
            day = 1
        n = len(worlds)
        return f"{n} world{'s' if n != 1 else ''} · {mode} day {day}"

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        w, h = max(int(rect.width()), 8), max(int(rect.height()), 8)
        sky, island = _thumbnail_layers(w, h)
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        p.setClipRect(rect)
        p.drawImage(QRectF(rect.x(), rect.y(), w, h), sky)
        bs = h / 7
        for k, (cy, cw) in enumerate(((0.1, 2.4), (0.24, 1.6), (0.36, 1.2))):
            x = (k * 0.41 * w + t * (7 + k * 3)) % (w + cw * bs) - cw * bs
            p.fillRect(QRectF(rect.x() + x, rect.y() + h * cy, cw * bs, bs * 0.32), QColor(255, 255, 255, 230))
            p.fillRect(QRectF(rect.x() + x + bs * 0.3, rect.y() + h * cy - bs * 0.16, cw * bs * 0.5, bs * 0.17),
                       QColor(255, 255, 255, 230))
        bob = math.sin(t * 1.3) * h * 0.015
        p.drawImage(QRectF(rect.x(), rect.y() + bob, w, h), island)
        p.restore()

    # --- the loop -----------------------------------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        if self.screen == "splash":
            self.step_loading()
        elif self.screen == "generating":
            self.step_generation()
        elif self.screen == "game" and self.player is not None:
            self.update_world(dt)
        elif self.screen == "title" and self.title_world is not None:
            self.title_world.pump(self.title_spawn[0], self.title_spawn[2], PANO_CHUNKS)

    def step_loading(self) -> None:
        try:
            self.load_progress = next(self._loader)
        except StopIteration:
            self.load_progress = 1.0
            self.go_title()

    def finish_loading(self) -> None:
        """Do the whole start-up now (tests)."""
        while self.screen == "splash":
            self.step_loading()

    def _loading(self):  # noqa: ANN202 - a generator of progress
        yield 0.1
        self.start_helpers()
        yield 0.3
        sounds.build(self.sfx)
        yield 0.6
        layers()  # the 3D view's textures
        yield 0.7
        self.renderer.warm_up()
        yield 0.85
        for item in B.CREATIVE_ITEMS[:12]:
            G.item_icon(item, 16 * self.gs())
        yield 1.0

    def _start_title_world(self) -> None:
        """The land that turns behind the title screen (built by the helpers while the menu shows)."""
        if self.title_world is not None or self.pool is None:
            return
        try:
            from app.arcade.games.blockcraft.world import World

            self.title_world = World(PANO_SEED, self.pool)
            self.title_spawn = self.title_world.gen.find_spawn()
            self.title_since = 0.0
        except Exception:  # noqa: BLE001 - the title just keeps its plain sky
            log.info("BlockCraft title world not made", exc_info=True)
            self.title_world = None

    def go_title(self) -> None:
        self.screen = "title"
        self.overlay = None
        self.splash = SPLASHES[int(time.time()) % len(SPLASHES)]
        self.set_mouse_captured(False)
        self._start_title_world()

    def capture_if_playing(self) -> None:
        playing = self.screen == "game" and self.overlay is None and not self.paused and self.health > 0
        self.set_mouse_captured(playing)

    def open_menu(self) -> None:
        self.overlay = "menu"
        self.mouse_buttons = 0
        self.set_mouse_captured(False)

    def open_overlay(self, name: str) -> None:
        self.overlay = name
        self.grid = [None] * 9
        self.cursor = None
        self.mouse_buttons = 0
        self.mining = None
        self.set_mouse_captured(False)
        self.sfx.play("door" if name != "inventory" else "click", 0.4)

    def close_container(self) -> None:
        for s in self.grid + [self.cursor] + self.rune_slots:
            if s is not None:
                left = self.inv.add(*s)
                if left and self.player is not None:
                    self._spawn_drop(self.player.x, self.player.y + 1.2, self.player.z, [s[0], left, s[2]])
        self.grid = [None] * 9
        self.rune_slots = [None, None]
        self.cursor = None
        self.overlay = None
        self.capture_if_playing()

    # --- input --------------------------------------------------------------------------------------------------

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001
        if self.screen == "create" and self.focus_field:
            self._type(key, event)
            return
        if self.screen != "game" or self.player is None:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.screen == "select" and \
                    0 <= self.sel_world < len(self.worlds):
                self.play_world(self.sel_world)
            return
        if key == Qt.Key.Key_F3:
            self.show_debug = not self.show_debug
        elif key == Qt.Key.Key_F1:
            self.hide_hud = not self.hide_hud
        if self.overlay in ("menu", "options", "dead"):
            return
        if key in (Qt.Key.Key_R, Qt.Key.Key_L):
            name = "recipes" if key == Qt.Key.Key_R else "advancements"
            if self.overlay == name:
                self.close_container()
            elif self.overlay is None:
                self.open_book(name)
            return
        if key == Qt.Key.Key_E:
            if self.overlay is None:
                self.open_overlay("inventory")
            else:
                self.close_container()
            return
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            n = key - Qt.Key.Key_1
            if self.overlay is not None:
                ref = self.slot_at(*self.mouse_gui)
                if ref is not None and ref[0] in ("inv", "grid", "fur", "che") and ref != ("inv", n):
                    a, b = self.get_slot(ref), self.inv.slots[n]
                    if ref == ("fur", 2) and b is not None:
                        return
                    self.set_slot(ref, b)
                    self.inv.slots[n] = a
                return
            self.inv.selected = n
            return
        if self.overlay is not None:
            return
        now = self.time
        if key == Qt.Key.Key_Space and self.creative:
            if now - self.last_space < 0.3:
                self.player.flying = not self.player.flying
                self.player.vy = 0.0
                self.last_space = -1.0
            else:
                self.last_space = now
        elif key == Qt.Key.Key_W:
            if now - self.last_w < 0.3:
                self.sprint_key = True
            self.last_w = now
        elif key == Qt.Key.Key_Q:
            self.drop_held(bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier))

    def keyPressEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        """Typing in a text field takes the typed characters as they are (rebinding a key must not stop a letter
        being typed); everything else goes through the arcade's (rebindable) key handling."""
        ch = event.text()
        if self.screen == "create" and self.focus_field and not self.paused and ch and ch.isprintable():
            self._type_text(ch)
            return
        super().keyPressEvent(event)

    def _type_text(self, ch: str) -> None:
        field = self.focus_field
        text = self.new_name if field == "name" else self.new_seed
        if len(text) < 32 and all(c in G.GLYPHS or c == " " for c in ch):
            text += ch
        if field == "name":
            self.new_name = text
        else:
            self.new_seed = text

    def _type(self, key: int, event) -> None:  # noqa: ANN001
        field = self.focus_field
        text = self.new_name if field == "name" else self.new_seed
        if key == Qt.Key.Key_Backspace:
            text = text[:-1]
        elif key == Qt.Key.Key_Tab:
            self.focus_field = "seed" if field == "name" else "name"
            return
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.create_world()
            return
        if field == "name":
            self.new_name = text
        else:
            self.new_seed = text

    def mouse_pressed(self, button: Qt.MouseButton, pos: QPointF) -> None:
        self.mouse_gui = self.to_gui(pos)
        bit = {Qt.MouseButton.LeftButton: 1, Qt.MouseButton.RightButton: 2, Qt.MouseButton.MiddleButton: 4}.get(button,
                                                                                                                0)
        if self.screen == "game" and self.player is not None and self.overlay is None:
            if not self.mouse_captured:
                self.capture_if_playing()
                return
            self.mouse_buttons |= bit
            if bit == 1:
                self.swing = 1.0
                if self.target_mob is not None:
                    self.attack()
            elif bit == 4:
                self.pick_block()
            elif bit == 2:
                self.use_click = True
            return
        if self.screen == "game" and self.overlay in ("recipes", "advancements"):
            self.book_click(bit, *self.mouse_gui)
            return
        if self.screen == "game" and self.overlay in CONTAINERS:
            self.container_click(bit, pos)
            return
        if bit != 1:
            return
        gx, gy = self.mouse_gui
        w = self.widget_at(gx, gy)
        self.focus_field = w.id if w is not None and w.kind == "field" else None
        if w is None or not w.enabled:
            return
        if w.kind == "slider":
            self.drag_slider = w.id
            self._set_slider(w, gx)
            return
        if w.kind in ("button", "entry"):
            self.sfx.play("click")
            self.on_widget(w.id)

    def mouse_released(self, button: Qt.MouseButton, pos: QPointF) -> None:
        bit = {Qt.MouseButton.LeftButton: 1, Qt.MouseButton.RightButton: 2, Qt.MouseButton.MiddleButton: 4}.get(button,
                                                                                                                0)
        self.mouse_buttons &= ~bit
        self.drag_slider = None

    def mouse_moved(self, pos: QPointF) -> None:
        self.mouse_gui = self.to_gui(pos)
        if self.drag_slider:
            for w in self.widgets():
                if w.id == self.drag_slider:
                    self._set_slider(w, self.mouse_gui[0])

    def wheel(self, steps: int) -> None:
        if self.screen == "game" and self.overlay is None and self.player is not None:
            self.inv.selected = (self.inv.selected - steps) % 9
        elif self.screen == "game" and self.overlay == "inventory" and self.creative:
            rows = max(0, math.ceil(len(B.CREATIVE_ITEMS) / 9) - 5)
            self.creative_scroll = min(max(self.creative_scroll - steps, 0), rows)
        elif self.screen == "game" and self.overlay in ("recipes", "advancements"):
            self.book_scroll(steps)
        elif self.screen == "select":
            self.list_scroll = max(0.0, self.list_scroll - steps * 20)

    def _set_slider(self, w, gx: float) -> None:  # noqa: ANN001
        v = min(max((gx - w.rect.x() - 4) / (w.rect.width() - 8), 0.0), 1.0)
        if w.id == "fov":
            self.opts["fov"] = int(round(30 + v * 80))
        elif w.id == "dist":
            self.opts["chunks"] = int(round(MIN_CHUNKS + v * (MAX_CHUNKS - MIN_CHUNKS)))
        elif w.id == "gamma":
            self.opts["gamma"] = round(v, 2)

    # --- menu actions -------------------------------------------------------------------------------------------

    def on_widget(self, wid: str) -> None:
        if wid == "single":
            self.screen = "select"
            self.sel_world = 0 if self.worlds else -1
        elif wid == "options":
            if self.screen == "game":
                self.overlay = "options"
            else:
                self.options_back = self.screen
                self.screen = "options"
        elif wid == "exit":
            if self.screen == "game":
                self.store_world()
            self.exit_requested.emit()
        elif wid.startswith("world:"):
            i = int(wid.split(":")[1])
            now = time.time()
            if self._last_click[1] == wid and now - self._last_click[0] < 0.4:
                self.play_world(i)
                return
            self._last_click = (now, wid)
            self.sel_world = i
        elif wid == "play":
            self.play_world(self.sel_world)
        elif wid == "create":
            n = len(self.worlds) + 1
            self.new_name = "New World" if n == 1 else f"New World ({n})"
            self.new_seed = ""
            self.new_mode = "survival"
            self.focus_field = "name"
            self.screen = "create"
        elif wid == "delete":
            self.screen = "delete"
        elif wid == "cancel":
            self.screen = "title"
        elif wid == "mode":
            self.new_mode = "creative" if self.new_mode == "survival" else "survival"
        elif wid == "do_create":
            self.create_world()
        elif wid == "cancel_create":
            self.screen = "select"
        elif wid == "confirm_delete":
            if 0 <= self.sel_world < len(self.worlds):
                self.worlds.pop(self.sel_world)
                self.progress_changed.emit()
            self.sel_world = -1
            self.screen = "select"
        elif wid == "cancel_delete":
            self.screen = "select"
        elif wid in ("graphics", "smooth", "bob", "music", "particles", "aa", "shaders"):
            key = "fancy" if wid == "graphics" else wid
            self.opts[key] = not self.opts.get(key, DEFAULT_OPTS.get(key, True))
            if wid == "smooth":
                for world in (self.world, self.title_world):
                    if world is not None:
                        world.remesh_all(bool(self.opts["smooth"]))
        elif wid in ("adv", "book"):
            self.open_book("advancements" if wid == "adv" else "recipes")
        elif wid == "home":
            if self.dim != "over" and not self.travelling:
                self.overlay = None
                self.travel("over")
        elif wid == "controls":
            self.controls_requested.emit()
        elif wid == "res":
            i = RES_MODES.index(self.opts["res"]) if self.opts.get("res") in RES_MODES else 0
            self.opts["res"] = RES_MODES[(i + 1) % len(RES_MODES)]
        elif wid == "gui":
            self.opts["gui"] = (int(self.opts.get("gui", 0)) + 1) % 5
        elif wid == "done":
            if self.screen == "game":
                self.overlay = "menu"
            else:
                self.screen = self.options_back if self.options_back in ("title",) else "title"
        elif wid == "back":
            self.overlay = None
            self.capture_if_playing()
        elif wid == "quit_title":
            self.leave_world()
            self.progress_changed.emit()
            self.go_title()
        elif wid == "respawn":
            self.respawn()
        elif wid == "title":
            self.respawn()
            self.leave_world()
            self.progress_changed.emit()
            self.go_title()

    def create_world(self) -> None:
        name = self.new_name.strip() or "New World"
        text = self.new_seed.strip()
        if not text:
            seed = int(np.random.default_rng().integers(0, 2 ** 31))
        else:
            try:
                seed = int(text)
            except ValueError:
                seed = _java_hash(text)
        rec = {"name": name[:32], "seed": seed, "mode": self.new_mode, "fmt": SAVE_FORMAT,
               "created": int(time.time()), "last": int(time.time()), "time": 0.0}
        self.focus_field = None
        self.start_world(rec)

    def play_world(self, i: int) -> None:
        if not 0 <= i < len(self.worlds):
            return
        rec = self.worlds.pop(i)
        rec["last"] = int(time.time())
        self.worlds.insert(0, rec)
        self.sel_world = 0
        self.start_world(rec)

    # --- containers ---------------------------------------------------------------------------------------------

    def grid_width(self) -> int:
        return 3 if self.overlay == "crafting" else 2

    def crafting_result(self) -> list | None:
        n = self.grid_width() ** 2
        res = I.match([s[0] if s else 0 for s in self.grid[:n]], self.grid_width())
        return I.stack(*res) if res else None

    def get_slot(self, ref: tuple) -> list | None:
        kind = ref[0]
        if kind == "inv":
            return self.inv.slots[ref[1]]
        if kind == "grid":
            return self.grid[ref[1]]
        if kind == "out":
            return self.crafting_result()
        if kind == "arm":
            return self.inv.armor[ref[1]]
        if kind == "fur":
            f = self.furnaces.get(self.furnace_pos)
            return f.slots[ref[1]] if f is not None else None
        if kind == "che":
            slots = self.chest_slots(self.chest_pos)
            return slots[ref[1]] if slots is not None else None
        if kind == "run":
            return self.rune_slots[ref[1]]
        if kind == "pal":
            item = B.CREATIVE_ITEMS[ref[1]]
            return I.stack(item, 1)
        return None

    def set_slot(self, ref: tuple, s: list | None) -> None:
        if s is not None and s[1] <= 0:
            s = None
        kind = ref[0]
        if kind == "inv":
            self.inv.slots[ref[1]] = s
        elif kind == "grid":
            self.grid[ref[1]] = s
        elif kind == "arm":
            self.inv.armor[ref[1]] = s
        elif kind == "fur":
            f = self.furnaces.get(self.furnace_pos)
            if f is not None:
                f.slots[ref[1]] = s
        elif kind == "che":
            slots = self.chest_slots(self.chest_pos)
            if slots is not None:
                slots[ref[1]] = s
        elif kind == "run":
            self.rune_slots[ref[1]] = s

    def container_click(self, bit: int, pos: QPointF) -> None:
        gx, gy = self.mouse_gui
        ref = self.slot_at(gx, gy)
        shift = Qt.Key.Key_Shift in self.keys
        if ref is None and self.overlay == "runes" and bit == 1:
            row = self.rune_row_at(gx, gy)
            if row is not None:
                self.apply_rune(row)
                return
        if ref is None:
            ox, oy = self.panel_origin()
            pw, ph = self.panel_size()
            inside = ox <= gx < ox + pw and oy <= gy < oy + ph
            if self.overlay == "inventory" and self.creative and ox + 175 <= gx < ox + 187 and oy + 18 <= gy < oy + 106:
                rows = max(0, math.ceil(len(B.CREATIVE_ITEMS) / 9) - 5)
                self.creative_scroll = round((gy - oy - 18) / 88 * rows)
                return
            if not inside and self.cursor is not None and bit in (1, 2):
                if not self.creative:
                    drop = self.cursor if bit == 1 else [self.cursor[0], 1, self.cursor[2]]
                    eye, d = self.eye(), self.look_dir()
                    self._spawn_drop(eye[0], eye[1] - 0.3, eye[2], list(drop), v=(d[0] * 0.3, 0.1, d[2] * 0.3),
                                     delay=2.0)
                if bit == 1:
                    self.cursor = None
                else:
                    self.cursor[1] -= 1
                    if self.cursor[1] <= 0:
                        self.cursor = None
            return
        self.click_slot(ref, bit, shift)

    def click_slot(self, ref: tuple, bit: int, shift: bool = False) -> None:
        """Minecraft's slot clicking: pick up / put down / swap / merge stacks, right click halves or places one,
        shift moves a stack across."""
        kind = ref[0]
        if kind == "pal":
            item = B.CREATIVE_ITEMS[ref[1]]
            full = I.stack(item, B.max_stack(item))
            if shift:
                self.inv.add(item, B.max_stack(item))
            elif self.cursor is not None:
                self.cursor = None
            else:
                self.cursor = full if bit != 2 else I.stack(item, 1)
            return
        if kind == "out":
            self._take_output(shift)
            return
        s = self.get_slot(ref)
        if shift and s is not None:
            self._shift_move(ref, s)
            return
        cur = self.cursor
        if kind == "run" and cur is not None and not self.rune_slot_takes(ref[1], cur[0]):
            return  # the bench takes a tool, a weapon or armour, and gems
        if kind == "arm":  # an armour slot takes only its own piece
            info = B.armor_info(cur[0]) if cur else None
            if cur is None or (info is not None and info[1] == ref[1]):
                self.inv.armor[ref[1]], self.cursor = cur, s
                if cur is not None:
                    self.sfx.play("click", 0.5)
            return
        if ref == ("fur", 2):  # the furnace output only gives
            if s is not None and (cur is None or (I.same_kind(cur, s) and cur[1] + s[1] <= B.max_stack(s[0]))):
                if cur is None:
                    self.cursor = s
                else:
                    cur[1] += s[1]
                self.set_slot(ref, None)
                f = self.furnaces.get(self.furnace_pos)
                if f is not None and f.xp >= 1:
                    self.gain_xp(int(f.xp))
                    f.xp -= int(f.xp)
            return
        if bit == 2:
            if cur is None:
                if s is not None:
                    half = (s[1] + 1) // 2
                    self.cursor = [s[0], half, s[2]]
                    s[1] -= half
                    self.set_slot(ref, s)
            elif s is None:
                self.set_slot(ref, [cur[0], 1, cur[2]])
                cur[1] -= 1
            elif I.same_kind(cur, s) and s[1] < B.max_stack(s[0]):
                s[1] += 1
                cur[1] -= 1
            else:
                self.set_slot(ref, cur)
                self.cursor = s
            if self.cursor is not None and self.cursor[1] <= 0:
                self.cursor = None
            return
        if cur is None:
            self.cursor = s
            self.set_slot(ref, None)
        elif s is None:
            self.set_slot(ref, cur)
            self.cursor = None
        elif I.same_kind(cur, s):
            n = min(B.max_stack(s[0]) - s[1], cur[1])
            s[1] += n
            cur[1] -= n
            if cur[1] <= 0:
                self.cursor = None
        else:
            self.set_slot(ref, cur)
            self.cursor = s

    def _take_output(self, shift: bool) -> None:
        res = self.crafting_result()
        if res is None:
            return
        n = self.grid_width() ** 2
        for _ in range(64 if shift else 1):
            res = self.crafting_result()
            if res is None:
                break
            if shift:
                if self.inv.add(*res):
                    break
            else:
                cur = self.cursor
                if cur is None:
                    self.cursor = res
                elif I.same_kind(cur, res) and cur[1] + res[1] <= B.max_stack(res[0]):
                    cur[1] += res[1]
                else:
                    return
            for k in range(n):
                g = self.grid[k]
                if g is not None:
                    g[1] -= 1
                    if g[1] <= 0:
                        self.grid[k] = None
        self.sfx.play("click", 0.5)

    def _shift_move(self, ref: tuple, s: list) -> None:
        if ref[0] in ("grid", "fur", "arm", "che", "run"):
            left = self.inv.add(*s)
            self.set_slot(ref, None if left <= 0 else [s[0], left, s[2]])
            return
        i = ref[1]
        info = B.armor_info(s[0])
        if info is not None and self.overlay == "inventory" and not self.creative and self.inv.armor[info[1]] is None:
            self.inv.armor[info[1]] = s  # armour goes on
            self.inv.slots[i] = None
            self.sfx.play("click", 0.5)
            return
        if self.overlay == "furnace":
            f = self.furnaces.get(self.furnace_pos)
            if f is not None:
                target = 0 if s[0] in I.SMELT else (1 if I.fuel_time(s[0]) > 0 else None)
                if target is not None:
                    dst = f.slots[target]
                    if dst is None:
                        f.slots[target] = s
                        self.inv.slots[i] = None
                        return
                    if I.same_kind(dst, s):
                        n = min(B.max_stack(s[0]) - dst[1], s[1])
                        dst[1] += n
                        s[1] -= n
                        if s[1] <= 0:
                            self.inv.slots[i] = None
                        return
        if self.overlay == "runes":  # onto the bench: the thing to mark, or the gems
            k = 1 if s[0] == B.GEM else 0
            dst = self.rune_slots[k]
            if self.rune_slot_takes(k, s[0]):
                if dst is None:
                    self.rune_slots[k] = s
                    self.inv.slots[i] = None
                elif I.same_kind(dst, s) and dst[1] < B.max_stack(s[0]):
                    n = min(B.max_stack(s[0]) - dst[1], s[1])
                    dst[1] += n
                    s[1] -= n
                    if s[1] <= 0:
                        self.inv.slots[i] = None
            return
        if self.overlay == "chest":  # into the chest: onto stacks of its kind first, then the first free slot
            slots = self.chest_slots(self.chest_pos)
            if slots is not None:
                for d in slots:
                    if d is not None and I.same_kind(d, s) and d[1] < B.max_stack(s[0]):
                        n = min(B.max_stack(s[0]) - d[1], s[1])
                        d[1] += n
                        s[1] -= n
                        if s[1] <= 0:
                            self.inv.slots[i] = None
                            return
                for j, d in enumerate(slots):
                    if d is None:
                        slots[j] = s
                        self.inv.slots[i] = None
                        return
            return
        rng = range(9, 36) if i < 9 else range(0, 9)
        if self.creative and self.overlay == "inventory":
            self.inv.slots[i] = None
            return
        for j in rng:
            d = self.inv.slots[j]
            if d is not None and I.same_kind(d, s) and d[1] < B.max_stack(s[0]):
                n = min(B.max_stack(s[0]) - d[1], s[1])
                d[1] += n
                s[1] -= n
                if s[1] <= 0:
                    self.inv.slots[i] = None
                    return
        for j in rng:
            if self.inv.slots[j] is None:
                self.inv.slots[j] = s
                self.inv.slots[i] = None
                return


def _java_hash(text: str) -> int:
    h = 0
    for ch in text:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h


def _clean_record(r) -> dict | None:  # noqa: ANN001
    """A saved world record made safe (None when it is not one)."""
    if not isinstance(r, dict):
        return None
    try:
        seed = int(r.get("seed", 0))
    except (TypeError, ValueError):
        return None
    rec = dict(r)
    rec["seed"] = seed
    rec["name"] = str(r.get("name", "World"))[:32] or "World"
    rec["mode"] = "creative" if r.get("mode") == "creative" else "survival"
    if rec.get("fmt") != SAVE_FORMAT:  # a walled world of the first BlockCraft: the same seed, endless now. Its
        for key in ("edits", "player", "spawn", "furn", "mobs", "visited", "size", "height"):  # land is made anew
            rec.pop(key, None)  # (the inventory, the game mode and the day are kept)
        rec["fmt"] = SAVE_FORMAT
    for key in ("edits", "edits_deep", "edits_isles"):
        if not isinstance(rec.get(key, ""), str):
            rec[key] = ""
    if rec.get("dim") not in ("deep", "isles"):
        rec["dim"] = "over"
    for k in ("last", "created"):
        if not isinstance(rec.get(k, 0), (int, float)):
            rec[k] = 0
    try:
        rec["time"] = float(r.get("time", 0.0))
    except (TypeError, ValueError):
        rec["time"] = 0.0
    return rec


def _thumbnail_layers(w: int, h: int) -> tuple[QImage, QImage]:
    """The card art: a sky with a square sun, and a floating island of isometric blocks with a tree, flowers,
    water and ore."""
    key = (w, h)
    if key in _thumb_cache:
        return _thumb_cache[key]
    sky = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    p = QPainter(sky)
    g = QLinearGradient(0, 0, 0, h)
    g.setColorAt(0, QColor(110, 160, 255))
    g.setColorAt(1, QColor(200, 222, 255))
    p.fillRect(0, 0, w, h, g)
    s = h / 9
    p.fillRect(QRectF(w * 0.08 - s * 0.3, h * 0.08 - s * 0.3, s * 1.6, s * 1.6), QColor(255, 255, 200, 60))
    p.fillRect(QRectF(w * 0.08, h * 0.08, s, s), QColor(255, 255, 225))
    p.end()
    n = max(int(min(w * 0.16, h * 0.175)), 8)
    m = n * 0.06
    bw, hh = n / 2 - m, (n / 2 - m) * 0.5
    sh = n - 2 * m - 2 * hh
    layers = [
        (-2, [(x, z) for x in range(1, 4) for z in range(1, 4)], B.STONE),
        (-1, [(x, z) for x in range(5) for z in range(5) if 0 < x + z < 8 and abs(x - z) < 4], B.DIRT),
        (0, [(x, z) for x in range(5) for z in range(5) if (x, z) not in ((0, 0), (4, 4), (0, 4))], B.GRASS),
    ]
    blocks: list[tuple[int, int, int, int]] = []
    for y, cells, b in layers:
        for x, z in cells:
            blocks.append((x, y, z, b))
    blocks = [(x, y, z, B.DIAMOND_ORE if (x, y, z) == (3, -2, 2) else b) for x, y, z, b in blocks]
    blocks = [(x, y, z, B.WATER if (y, x, z) in ((0, 3, 2), (0, 2, 3), (0, 3, 3)) else b) for x, y, z, b in blocks]
    for y in (1, 2, 3):
        blocks.append((1, y, 1, B.OAK_LOG))
    for x in range(0, 3):
        for z in range(0, 3):
            for y in (3, 4):
                if (x, z) != (1, 1) or y == 4:
                    blocks.append((x, y, z, B.OAK_LEAVES))
    blocks.append((1, 5, 1, B.OAK_LEAVES))
    sprites = [(3, 1, 1, B.POPPY), (1, 1, 3, B.DANDELION), (4, 1, 1, B.TALL_GRASS), (4, 1, 3, B.TALL_GRASS)]
    island = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    island.fill(0)
    p = QPainter(island)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    placed = []
    for x, y, z, b in sorted(blocks + sprites, key=lambda t: (t[0] + t[2], t[1])):
        placed.append(((x - z) * bw - n / 2, (x + z) * hh - y * sh - n / 2, b))
    top = min(sy for _, sy, _ in placed)
    bottom = max(sy for _, sy, _ in placed) + n
    cx, oy = w * 0.55, (h - (bottom - top)) / 2 - top + h * 0.03
    icons: dict[int, QImage] = {}
    for sx, sy, b in placed:
        sx += cx
        sy += oy
        if b not in icons:
            if B.KIND[b] == B.K_CROSS:
                icons[b] = G.to_qimage(np.repeat(np.repeat(B.ATLAS[B.FACE_TEX[b, 4]], 4, 0), 4, 1))
            else:
                cube = G._iso_block(b, n)
                if b == B.WATER:
                    cube[..., 3] = np.where(cube[..., 3] > 0, 200, 0)
                icons[b] = G.to_qimage(cube)
        if B.KIND[b] == B.K_CROSS:
            p.drawImage(QRectF(sx + n * 0.25, sy - n * 0.05, n * 0.5, n * 0.5), icons[b])
        else:
            p.drawImage(QRectF(sx, sy, n, n), icons[b])
    p.end()
    if len(_thumb_cache) > 8:
        _thumb_cache.clear()
    _thumb_cache[key] = (sky, island)
    return sky, island
