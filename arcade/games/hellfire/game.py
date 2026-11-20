"""HELLFIRE - the arcade's 1993-style first-person shooter: a marine on a Martian moon base overrun by demons from
a teleporter experiment gone wrong. Six maps (one of them secret), ten monsters, eight weapons, keycards,
power-ups, the HUD, the automap, the intermission tally and the melt wipe.

The 3D view is drawn by the graphics card at the window's own resolution (``render.py``); the HUD, menus and
screens are painted over it with QPainter, also at full resolution. The game itself runs at 35 tics per second
(``world.py``) and the view is interpolated between tics, so it moves smoothly at the display's frame rate."""

from __future__ import annotations

import logging
import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient)

from app.arcade.base import ArcadeGame

from .assets import side_index, sprites
from .hud import Hud, draw_automap, hfont, shadow_text
from .info import ITEM_INFO, MISSILE_INFO, SKILL_SHORT, SKILLS, VIEWHEIGHT, WEAPONS
from .level import CELL
from .maps import EPISODE, MAIN_ROUTE, MAPS, STORY
from .render import Renderer
from .screens import Backdrops, Intermission, Menu, TextScreen, Wipe, draw_title, page_transform
from .sounds import build_sounds
from .world import Player, World

log = logging.getLogger(__name__)

TIC = 1.0 / 35.0
SAVE_VERSION = 2
NSLOTS = 6
ART_ALIAS = {"clip_drop": "clip", "w_shotgun_drop": "w_shotgun", "w_chaingun_drop": "w_chaingun"}
FX_ART = {"puff": "puff", "blood": "blood", "tracer": "tracer_x"}


class HellfireGame(ArcadeGame):
    game_id = "hellfire"
    title = "HELLFIRE"
    subtitle = "Rip and tear through the moon base"
    description = ("A 1993-style first-person shooter. You are the last marine on a Martian moon base after a "
                   "teleporter experiment tore a hole into hell. Six maps, eight weapons, keycards, secrets, "
                   "power-ups, and a siege titan waiting at the gate.")
    controls = (
        ("W A S D / Arrows", "Move, strafe, turn"),
        ("Mouse", "Turn"),
        ("Ctrl / Left click", "Fire"),
        ("Space / E", "Open doors, use switches"),
        ("1-7 / Wheel", "Weapons"),
        ("Shift", "Walk"),
        ("Tab", "Automap"),
        ("Esc", "Menu"),
    )
    accent = "#d8361c"
    uses_mouse_look = False  # captured while playing only (handled in tick)
    uses_key_turn = True
    bindings = (
        ("Move forward", Qt.Key.Key_W), ("Move back", Qt.Key.Key_S), ("Strafe left", Qt.Key.Key_A),
        ("Strafe right", Qt.Key.Key_D), ("Turn left", Qt.Key.Key_Left), ("Turn right", Qt.Key.Key_Right),
        ("Fire", Qt.Key.Key_Control), ("Use (doors, switches)", Qt.Key.Key_Space), ("Walk (hold)", Qt.Key.Key_Shift),
        ("Automap", Qt.Key.Key_Tab), ("Next weapon", Qt.Key.Key_PageDown), ("Previous weapon", Qt.Key.Key_Q),
    )

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        build_sounds(self.sfx)
        self.sprites = sprites()
        self.renderer = Renderer()
        self.back = Backdrops()
        self.settings = {"run": True, "music": True, "messages": True}
        self.screen = "title"
        self.menu = Menu(self)
        self.menu.show("main")
        self.world: World | None = None
        self.hud = Hud()
        self.skill = 3
        self.map_index = 0
        self.slots: list[dict | None] = [None] * NSLOTS
        self.career_kills = 0
        self.episode_done = False
        self.acc = 0.0
        self.wipe: Wipe | None = None
        self.wipe_acc = 0.0
        self.inter: Intermission | None = None
        self.text: TextScreen | None = None
        self.ticks = 0
        self.automap = False
        self.automap_scale = 16.0  # pixels per map cell (at 1280 x 780)
        self.prev = (0.0, 0.0, 0.0, 0.0)
        self.turnheld = 0
        self.weapon_req: object = None
        self.mouse_fire = False
        self.music_playing = False
        self.dead_tics = 0
        self.last_render_ms = 0.0
        self.text_next = "title"  # after the text screen: "start" (the first map) or "title"
        self.pending_player = Player()
        self.carry = Player()  # the inventory carried to the next map
        self.next_index = 0
        self.mouse_turn = 0.0  # mouse turning since the last tic (applied to the view at once)

    # --- the arcade's hooks -----------------------------------------------------------------------------------------

    @classmethod
    def summary(cls, data: dict) -> str:
        if not isinstance(data, dict) or not data:
            return "New game"
        try:
            kills = int(data.get("career_kills", 0) or 0)
            game = data.get("game")
            if isinstance(game, dict) and 0 <= int(game.get("map", 0) or 0) < len(MAPS):
                m = MAPS[int(game.get("map", 0) or 0)]
                skill = int(game.get("skill", 3) or 3)
                return f"{m.key} · {SKILL_SHORT[min(max(skill, 1), 5) - 1]} · {kills} kills"
            if data.get("episode_done"):
                return f"Episode cleared · {kills} kills"
            return f"{kills} kills" if kills else "New game"
        except (TypeError, ValueError, AttributeError):
            return "Continue"

    _thumb_cache: dict = {}

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        rect = QRectF(rect)
        key = (int(rect.width()), int(rect.height()))
        base = cls._thumb_cache.get(key)
        if base is None:
            base = cls._thumb_base(key[0], key[1])
            if len(cls._thumb_cache) > 6:
                cls._thumb_cache.clear()
            cls._thumb_cache[key] = base
        p.save()
        p.setClipRect(rect)
        p.drawPixmap(rect.topLeft(), base)
        w, h = rect.width(), rect.height()
        # animated flames along the bottom
        p.setPen(Qt.PenStyle.NoPen)
        for layer, (col, amp, base_y) in enumerate(((QColor(150, 10, 0, 200), 0.30, 0.80),
                                                    (QColor(240, 80, 0, 210), 0.22, 0.86),
                                                    (QColor(255, 210, 60, 220), 0.12, 0.92))):
            path = QPainterPath()
            path.moveTo(rect.left(), rect.bottom())
            n = 18
            for i in range(n + 1):
                x = rect.left() + w * i / n
                f = math.sin(i * 1.7 + t * (4 + layer) + layer) * 0.5 + math.sin(i * 0.9 - t * 3.1) * 0.5
                y = rect.top() + h * (base_y - amp * (0.55 + 0.45 * f) * (0.6 + 0.4 * math.sin(i * 2.3 + layer)))
                path.lineTo(x, y)
            path.lineTo(rect.right(), rect.bottom())
            path.closeSubpath()
            p.setBrush(col)
            p.drawPath(path)
        # the demon's glowing eye
        cx, cy = rect.left() + w * 0.72, rect.top() + h * 0.43
        glow = 0.6 + 0.4 * math.sin(t * 3.0)
        g = QRadialGradient(cx, cy, h * 0.09)
        g.setColorAt(0, QColor(220, 255, 160, int(255 * glow)))
        g.setColorAt(1, QColor(40, 200, 40, 0))
        p.setBrush(g)
        p.drawEllipse(QPointF(cx, cy), h * 0.09, h * 0.09)
        p.restore()

    @classmethod
    def _thumb_base(cls, w: int, h: int) -> QPixmap:
        pm = QPixmap(max(w, 1), max(h, 1))
        pm.fill(QColor(0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        g = QLinearGradient(0, 0, 0, h)
        g.setColorAt(0, QColor(30, 0, 0))
        g.setColorAt(0.6, QColor(120, 16, 0))
        g.setColorAt(1, QColor(255, 90, 0))
        p.fillRect(QRectF(0, 0, w, h), g)
        # mountains
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(60, 10, 5))
        path = QPainterPath()
        path.moveTo(0, h * 0.75)
        for i in range(13):
            path.lineTo(w * i / 12, h * (0.55 + 0.12 * math.sin(i * 2.1) + 0.06 * math.sin(i * 5.3)))
        path.lineTo(w, h)
        path.lineTo(0, h)
        p.drawPath(path)
        # the floating red demon with one eye and a toothy grin
        cx, cy, r = w * 0.72, h * 0.47, h * 0.26
        rg = QRadialGradient(cx - r * 0.35, cy - r * 0.4, r * 1.3)
        rg.setColorAt(0, QColor(255, 120, 100))
        rg.setColorAt(0.5, QColor(210, 20, 10))
        rg.setColorAt(1, QColor(60, 0, 0))
        p.setBrush(rg)
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.setBrush(QColor(230, 220, 200))
        for side in (-1, 1):
            horn = QPainterPath()
            horn.moveTo(cx + side * r * 0.55, cy - r * 0.75)
            horn.lineTo(cx + side * r * 0.85, cy - r * 1.15)
            horn.lineTo(cx + side * r * 0.75, cy - r * 0.6)
            p.drawPath(horn)
        p.setBrush(QColor(40, 0, 0))
        mouth = QPainterPath()
        mouth.moveTo(cx - r * 0.6, cy + r * 0.15)
        mouth.quadTo(cx, cy + r * 0.95, cx + r * 0.6, cy + r * 0.15)
        mouth.quadTo(cx, cy + r * 0.45, cx - r * 0.6, cy + r * 0.15)
        p.drawPath(mouth)
        p.setPen(QPen(QColor(240, 240, 230), max(r * 0.05, 1)))
        for i in range(7):
            x = cx - r * 0.45 + i * r * 0.15
            p.drawLine(QPointF(x, cy + r * 0.28), QPointF(x, cy + r * 0.42))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(QPointF(cx, cy - r * 0.38), r * 0.22, r * 0.18)
        p.setBrush(QColor(40, 170, 40))
        p.drawEllipse(QPointF(cx, cy - r * 0.38), r * 0.1, r * 0.1)
        # the logo
        f = QFont("Impact")
        f.setPixelSize(max(int(h * 0.24), 8))
        text = QPainterPath()
        text.addText(0, 0, f, "HELLFIRE")
        br = text.boundingRect()
        scale = min(w * 0.62 / max(br.width(), 1), 1.0 * h * 0.3 / max(br.height(), 1))
        p.translate(w * 0.05 - br.x() * scale, h * 0.06 - br.y() * scale)
        p.scale(scale, scale)
        lg = QLinearGradient(0, br.top(), 0, br.bottom())
        lg.setColorAt(0, QColor(255, 250, 200))
        lg.setColorAt(0.35, QColor(255, 200, 30))
        lg.setColorAt(0.7, QColor(230, 70, 0))
        lg.setColorAt(1, QColor(110, 0, 0))
        p.save()
        p.translate(2 / scale, 3 / scale)
        p.fillPath(text, QColor(40, 0, 0))
        p.restore()
        p.fillPath(text, lg)
        p.strokePath(text, QPen(QColor(50, 0, 0), 1.2 / scale))
        p.end()
        return pm


    def save_state(self) -> dict:
        return {"version": SAVE_VERSION, "settings": dict(self.settings), "slots": list(self.slots),
                "career_kills": self.career_kills, "episode_done": self.episode_done,
                "game": self.snapshot()}

    def load_state(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        try:
            s = data.get("settings") or {}
            for k in ("run", "music", "messages"):
                if k in s:
                    self.settings[k] = bool(s[k])
            slots = data.get("slots") or []
            self.slots = [x if isinstance(x, dict) else None for x in list(slots)[:NSLOTS]]
            self.slots += [None] * (NSLOTS - len(self.slots))
            self.career_kills = int(data.get("career_kills", 0) or 0)
            self.episode_done = bool(data.get("episode_done", False))
            game = data.get("game")
            if isinstance(game, dict):
                self.restore(game)
        except Exception:  # noqa: BLE001 - a damaged save never stops the game
            log.warning("Hellfire save not restored", exc_info=True)
            self.world = None
            self.screen = "title"
            self.menu.show("main")

    def handle_escape(self) -> bool:
        if self.menu.open:
            if self.menu.current == "main" and len(self.menu.stack) == 1 and self.menu.confirm is None:
                if self.screen == "title":
                    return False
                self.menu.close()
                return True
            self.menu.back()
            return True
        self.menu.show("main")
        self.sfx.play("swtchn")
        return True

    def on_pause(self) -> None:
        self.sfx.stop("music")
        self.music_playing = False
        self.mouse_fire = False

    def on_resume(self) -> None:
        self.acc = 0.0
        self.update_music()

    def stop(self) -> None:
        super().stop()
        self.renderer.close()

    # --- game flow ---------------------------------------------------------------------------------------------------

    def new_game(self, skill: int) -> None:
        self.skill = min(max(int(skill), 1), 5)
        self.menu.close()
        self.start_wipe()
        self.map_index = 0
        intro = STORY.get("intro")
        if intro:
            self.text = TextScreen(intro)
            self.text_next = "start"
            self.screen = "text"
            self.world = None
            self.pending_player = Player()
        else:
            self.start_map(0, Player())

    def start_map(self, index: int, player: Player, state: dict | None = None) -> None:
        self.map_index = index
        m = MAPS[index]
        self.renderer.prepare(m.sky)  # the graphics context, textures and this map's sky, made now, not mid-play
        self.world = World(m, self.skill, player, seed=index * 101 + self.skill)
        if state:
            self.world.from_json(state)
        self.world.reveal(force=True)
        self.hud = Hud()
        self.screen = "play"
        self.menu.stack = []
        self.menu.confirm = None
        self.automap = False
        self.acc = 0.0
        self.dead_tics = 0
        pm = self.world.pmo
        self.prev = (pm.x, pm.y, self.world.player.viewz, pm.angle)
        if not state:
            self.hud_message(f"{m.key}: {m.name}")
        self.update_music()

    def restart_level(self) -> None:
        """Death: the map again with a pistol, like the original's single player."""
        self.start_wipe()
        self.start_map(self.map_index, Player())

    def _next_map(self, secret: bool) -> int:
        m = MAPS[self.map_index]
        if secret and m.secret_next is not None:
            return m.secret_next
        if m.back_to is not None:
            return m.back_to
        i = MAIN_ROUTE.index(self.map_index) if self.map_index in MAIN_ROUTE else len(MAIN_ROUTE) - 1
        return MAIN_ROUTE[i + 1] if i + 1 < len(MAIN_ROUTE) else len(MAPS)

    def finish_level(self) -> None:
        w = self.world
        pl = w.player
        self.career_kills += pl.killcount
        stats = {"kills": pl.killcount, "total_kills": w.total_kills, "items": pl.itemcount,
                 "total_items": w.total_items, "secrets": pl.secretcount, "total_secrets": w.total_secrets,
                 "time": w.leveltime // 35, "par": w.mapdef.par}
        nxt = self._next_map(w.secret_exit)
        self.carry = Player()
        self.carry.from_json(pl.to_json())
        self.carry.keys = set()
        self.next_index = nxt
        self.start_wipe()
        route = [MAPS[i].key for i in MAIN_ROUTE]
        pos = MAIN_ROUTE.index(self.map_index) if self.map_index in MAIN_ROUTE else len(MAIN_ROUTE)
        npos = MAIN_ROUTE.index(nxt) if nxt in MAIN_ROUTE else (len(MAIN_ROUTE) if nxt < len(MAPS) else -1)
        keys = route + ([MAPS[nxt].key] if nxt < len(MAPS) and nxt not in MAIN_ROUTE else []) + \
            ([MAPS[self.map_index].key] if self.map_index not in MAIN_ROUTE else [])
        if self.map_index not in MAIN_ROUTE:
            pos = len(keys) - 1
            back = MAPS[self.map_index].back_to
            npos = MAIN_ROUTE.index(back) if back in MAIN_ROUTE else -1
        self.inter = Intermission(w.mapdef.name, MAPS[nxt].name if nxt < len(MAPS) else None, stats, pos, keys,
                                  npos)
        self.screen = "inter"
        self.world = None
        self.sfx.stop("music")
        self.music_playing = False
        self.progress_changed.emit()

    def after_intermission(self) -> None:
        self.start_wipe()
        if self.next_index < len(MAPS):
            self.start_map(self.next_index, self.carry)
            return
        self.episode_done = True
        self.text = TextScreen(STORY.get("end", "THE END"), bg="hell")
        self.text_next = "title"
        self.screen = "text"
        self.progress_changed.emit()

    def after_text(self) -> None:
        self.start_wipe()
        if self.text_next == "start":
            self.start_map(0, self.pending_player)
            return
        self.text = None
        self.world = None
        self.screen = "title"
        self.menu.show("main")
        self.update_music()

    def open_controls(self) -> None:
        """The arcade's Controls panel over the game (it pauses the game and resumes it after, back in this menu)."""
        self.set_mouse_captured(False)
        self.mouse_fire = False
        self.controls_requested.emit()

    def menu_quit(self) -> None:
        self.menu.close()
        self.sfx.stop_all()
        self.music_playing = False
        self.exit_requested.emit()

    def hud_message(self, text: str) -> None:
        if self.settings.get("messages", True) or text.endswith("saved."):
            self.hud.message(text)

    def start_wipe(self) -> None:
        """Keep the picture on screen now; it melts away over whatever comes next."""
        w, h = max(self.width(), 16), max(self.height(), 16)
        img = QImage(w, h, QImage.Format.Format_RGB32)
        img.fill(QColor(0, 0, 0))
        p = QPainter(img)
        try:
            self.compose(p, w, h, with_wipe=False)
        except Exception:  # noqa: BLE001 - no picture to melt is fine
            log.debug("Hellfire wipe picture", exc_info=True)
        p.end()
        self.wipe = Wipe(img, seed=self.ticks)
        self.wipe_acc = 0.0

    # --- saving ----------------------------------------------------------------------------------------------------

    def snapshot(self) -> dict | None:
        if self.screen == "play" and self.world is not None:
            if self.world.player.dead:
                return {"map": self.map_index, "skill": self.skill, "player": Player().to_json(), "world": None,
                        "career_kills": self.career_kills}
            return {"map": self.map_index, "skill": self.skill, "player": self.world.player.to_json(),
                    "world": self.world.to_json(), "career_kills": self.career_kills}
        if self.screen == "inter" and self.next_index < len(MAPS):
            return {"map": self.next_index, "skill": self.skill, "player": self.carry.to_json(), "world": None}
        if self.screen == "text" and self.text_next == "start":
            return {"map": 0, "skill": self.skill, "player": Player().to_json(), "world": None}
        return None

    def restore(self, snap: dict) -> None:
        idx = int(snap.get("map", 0) or 0)
        if not 0 <= idx < len(MAPS):
            raise ValueError("bad map")
        self.skill = min(max(int(snap.get("skill", 3) or 3), 1), 5)
        pl = Player()
        if isinstance(snap.get("player"), dict):
            pl.from_json(snap["player"])
        state = snap.get("world") if isinstance(snap.get("world"), dict) else None
        self.menu.close()
        self.start_map(idx, pl, state)
        if state is not None:
            # the player's inventory as saved (World() reset the level state)
            self.world.player.from_json(snap["player"])
            self.world.player.keys = {k for k in (state.get("player") or {}).get("keys", [])
                                      if k in ("blue", "yellow", "red")}
            self.world.pmo.hp = self.world.player.health
            if self.world.player.health <= 0:
                self.world.player.health = 1
            pm = self.world.pmo
            self.prev = (pm.x, pm.y, self.world.player.viewz, pm.angle)
            self.hud_message("Game resumed.")

    def save_slot(self, i: int) -> bool:
        snap = self.snapshot()
        if snap is None or self.world is None:
            return False
        m = MAPS[self.map_index]
        snap["name"] = f"{m.key}: {m.name.upper()} - {time.strftime('%H:%M')}"
        self.slots[i] = snap
        self.progress_changed.emit()
        return True

    def load_slot(self, i: int) -> None:
        snap = self.slots[i] if 0 <= i < len(self.slots) else None
        if not snap:
            return
        self.menu.close()
        self.start_wipe()
        try:
            self.restore(snap)
            self.hud_message("Game loaded.")
        except Exception:  # noqa: BLE001
            log.warning("Hellfire slot not loaded", exc_info=True)
            self.screen = "title"
            self.menu.show("main")

    # --- input -------------------------------------------------------------------------------------------------------

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001
        text = event.text() if event is not None else ""
        if self.wipe is not None and self.screen != "play":
            return
        if self.menu.open:
            self.menu.key(key, text)
            return
        if self.screen == "title":
            self.menu.show("main")
            return
        if self.screen == "inter" and self.inter is not None:
            self.inter.skip(self.sfx)
            return
        if self.screen == "text" and self.text is not None:
            self.text.skip()
            return
        if self.screen != "play" or self.world is None:
            return
        if key == Qt.Key.Key_Tab:
            self.automap = not self.automap
        elif self.automap and key in (Qt.Key.Key_Equal, Qt.Key.Key_Plus, Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            zoom = 1.25 if key in (Qt.Key.Key_Equal, Qt.Key.Key_Plus) else 0.8
            self.automap_scale = min(max(self.automap_scale * zoom, 3.0), 40.0)
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_7:
            self.weapon_req = int(key - Qt.Key.Key_0)
        elif key == Qt.Key.Key_F2:
            self.menu.show("save")
        elif key == Qt.Key.Key_F3:
            self.menu.show("load")
        elif key in (Qt.Key.Key_Q, Qt.Key.Key_PageUp):
            self.weapon_req = "prev"
        elif key in (Qt.Key.Key_PageDown,):
            self.weapon_req = "next"

    def mouse_pressed(self, button: Qt.MouseButton, pos: QPointF) -> None:
        if self.menu.open:
            x, y = self._to_page(pos)
            i = self.menu.item_at(x, y)
            if self.menu.current == "help":
                self.menu.back()
                return
            if i is not None:
                name = self.menu.current
                self.menu.sel[name] = i
                if button == Qt.MouseButton.LeftButton:
                    items = self.menu.items(name)
                    if items[i][2]:
                        self.menu.activate(items[i][1])
            elif button == Qt.MouseButton.RightButton:
                self.menu.back()
            return
        if self.screen == "title":
            self.menu.show("main")
            return
        if self.screen in ("inter", "text"):
            self.key_pressed(Qt.Key.Key_Space, None)
            return
        if button == Qt.MouseButton.LeftButton:
            self.mouse_fire = True
            if not self.mouse_captured and self.isActiveWindow():
                self.set_mouse_captured(True)
        elif button == Qt.MouseButton.RightButton:
            self.weapon_req = "next"

    def mouse_released(self, button: Qt.MouseButton, pos: QPointF) -> None:
        if button == Qt.MouseButton.LeftButton:
            self.mouse_fire = False

    def mouse_moved(self, pos: QPointF) -> None:
        if self.menu.open:
            x, y = self._to_page(pos)
            i = self.menu.item_at(x, y)
            if i is not None and self.menu.current:
                self.menu.sel[self.menu.current] = i

    def wheel(self, steps: int) -> None:
        if self.menu.open and self.menu.current:
            n = len(self.menu.items(self.menu.current)) or 1
            name = self.menu.current
            self.menu.sel[name] = (self.menu.sel.get(name, 0) - steps) % n
            return
        if self.screen == "play":
            self.weapon_req = "prev" if steps > 0 else "next"

    def _to_page(self, pos: QPointF) -> tuple[float, float]:
        tr, _page = page_transform(max(self.width(), 1), max(self.height(), 1))
        inv, _ok = tr.inverted()
        q = inv.map(QPointF(pos))
        return q.x(), q.y()

    def build_cmd(self) -> dict:
        k = self.keys
        K = Qt.Key
        shift = K.Key_Shift in k
        run = self.settings["run"] != shift
        fwd = side = 0.0
        if K.Key_W in k or K.Key_Up in k:
            fwd += 50 if run else 25
        if K.Key_S in k or K.Key_Down in k:
            fwd -= 50 if run else 25
        if K.Key_D in k:
            side += 40 if run else 24
        if K.Key_A in k:
            side -= 40 if run else 24
        turn = 0.0
        left, right = K.Key_Left in k, K.Key_Right in k
        if left or right:
            self.turnheld += 1
            amount = (1280 if run and self.turnheld > 6 else 640) / 65536 * 2 * math.pi * self.key_turn_speed
            if K.Key_Alt in k:
                side += (40 if run else 24) * (1 if right else -1)
            elif left != right:
                turn += amount if right else -amount
        else:
            self.turnheld = 0
        cmd = {"forward": fwd, "side": side, "turn": turn,
               "fire": self.mouse_fire or K.Key_Control in k,
               "use": K.Key_Space in k or K.Key_E in k}
        if self.weapon_req is not None:
            cmd["weapon"] = self.weapon_req
            self.weapon_req = None
        return cmd

    # --- the loop ----------------------------------------------------------------------------------------------

    def update_music(self) -> None:
        want = self.settings.get("music", True) and not self.paused and self.screen in ("play", "title", "text")
        if want and not self.music_playing:
            self.sfx.play("music")
            self.music_playing = True
        elif not want and self.music_playing:
            self.sfx.stop("music")
            self.music_playing = False

    def tick(self, dt: float) -> None:
        self.ticks += 1
        want_capture = self.screen == "play" and not self.menu.open and self.isActiveWindow()
        if want_capture != self.mouse_captured and (not want_capture or self.isVisible()):
            self.set_mouse_captured(want_capture)
        if self.wipe is not None:
            self.wipe_acc += dt
            while self.wipe_acc >= TIC and self.wipe is not None:
                self.wipe_acc -= TIC
                self.wipe.tick()
                if self.wipe.done:
                    self.wipe = None
        if self.screen == "title" or (self.menu.open and self.screen != "play"):
            return
        if self.menu.open:
            self.take_mouse_delta()
            return
        if self.screen == "inter":
            self._tic_loop(dt, self._inter_tic)
        elif self.screen == "text":
            self._tic_loop(dt, self._text_tic)
        elif self.screen == "play" and self.world is not None:
            dx, _ = self.take_mouse_delta()
            if dx and not self.world.player.dead:
                # mouse turning is applied at once (not per tic), so the view follows the hand smoothly
                a = dx * 0.0032
                self.world.pmo.angle = (self.world.pmo.angle + a) % (2 * math.pi)
                self.prev = (self.prev[0], self.prev[1], self.prev[2], self.prev[3] + a)
            self._tic_loop(dt, self._play_tic)

    def _tic_loop(self, dt: float, fn) -> None:  # noqa: ANN001
        self.acc += dt
        n = 0
        while self.acc >= TIC and n < 4:
            self.acc -= TIC
            n += 1
            fn()
            if self.screen != "play" and fn == self._play_tic:
                break
        if self.acc > TIC * 4:
            self.acc = 0.0

    def _inter_tic(self) -> None:
        if self.inter is None:
            return
        self.inter.tick(self.sfx)
        if self.inter.done:
            self.after_intermission()
            self.inter = None

    def _text_tic(self) -> None:
        if self.text is None:
            return
        self.text.tick()
        if self.text.done:
            self.after_text()

    def _play_tic(self) -> None:
        w = self.world
        if w is None or self.screen != "play":
            return
        pm = w.pmo
        self.prev = (pm.x, pm.y, w.player.viewz, pm.angle)
        cmd = self.build_cmd()
        if w.player.dead:
            self.dead_tics += 1
            if self.dead_tics > 35 and (cmd["use"] or cmd["fire"]):
                self.restart_level()
                return
        w.tick(cmd)
        w.reveal()
        self.hud.tick(w)
        self._events()
        if w.exited:
            self.finish_level()

    def _events(self) -> None:
        played = set()
        for ev in self.world.events:
            if ev[0] == "sound":
                if ev[1] not in played:
                    played.add(ev[1])
                    self.sfx.play(ev[1], ev[2])
            elif ev[0] == "msg":
                self.hud_message(ev[1])
            elif ev[0] == "died":
                self.dead_tics = 0
                self.automap = False
        self.world.events.clear()

    # --- drawing -------------------------------------------------------------------------------------------------

    def compose(self, p: QPainter, w: int, h: int, with_wipe: bool = True) -> None:
        t = self.ticks / 60.0
        if self.screen == "play" and self.world is not None:
            self.draw_play(p, w, h)
            if self.menu.open:
                p.fillRect(QRectF(0, 0, w, h), QColor(0, 0, 0, 150))
        elif self.screen == "inter" and self.inter is not None:
            self.inter.draw(p, w, h, self.back)
        elif self.screen == "text" and self.text is not None:
            self.text.draw(p, w, h, self.back)
        else:
            draw_title(p, w, h, t, logo=not (self.menu.open and self.menu.current not in (None, "main")))
            if not self.menu.open or self.menu.current == "main":
                p.save()
                p.setFont(hfont(max(h / 60, 9), False, "Segoe UI"))
                p.setPen(QColor(255, 226, 190, 220))
                p.drawText(QRectF(0, h - h / 30 - 4, w - 12, h / 30), Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter, "Art: the Freedoom project (BSD licence)")
                p.restore()
        if self.menu.open:
            self.menu.on_title = self.screen == "title"
            self.menu.draw(p, w, h, t)
        if with_wipe and self.wipe is not None:
            self.wipe.draw(p, w, h)

    def view(self) -> tuple[float, float, float, float]:
        """The camera between the last two tics."""
        w = self.world
        pm, pl = w.pmo, w.player
        alpha = min(max(self.acc / TIC, 0.0), 1.0) if not self.menu.open and not self.paused else 1.0
        x = self.prev[0] + (pm.x - self.prev[0]) * alpha
        y = self.prev[1] + (pm.y - self.prev[1]) * alpha
        z = self.prev[2] + (pl.viewz - self.prev[2]) * alpha
        da = (pm.angle - self.prev[3] + math.pi) % (2 * math.pi) - math.pi
        return x, y, z, self.prev[3] + da * alpha

    def draw_play(self, p: QPainter, w: int, h: int) -> None:
        wd = self.world
        pl = wd.player
        if self.automap:
            p.fillRect(QRectF(0, 0, w, h), QColor(0, 0, 0))
            draw_automap(p, w, h, wd, self.automap_scale)
        else:
            x, y, z, a = self.view()
            invuln = pl.powers.get("invuln", 0)
            # the view at the screen's real pixels (a 150 % display scale draws 1.5 x as many, so it stays sharp)
            dpr = max(float(p.device().devicePixelRatioF()) if p.device() is not None else 1.0, 1.0)
            img = self.renderer.render(
                wd.grid, (x, y, z), a, int(round(w * dpr)), int(round(h * dpr)), wd.leveltime,
                self.collect_sprites(x, y), self.weapon_quads(), sky_theme=wd.mapdef.sky,
                extra=pl.extralight * 16.0, tint=self.tint(),
                invuln=invuln > 4 * 35 or (invuln and (invuln // 8) % 2 == 1), t=wd.leveltime / 35.0)
            self.last_render_ms = self.renderer.ms
            img.setDevicePixelRatio(dpr)
            p.drawImage(QPointF(0, 0), img)
        self.hud.draw(p, w, h, wd, self.sprites, wd.leveltime, self.automap)
        if pl.dead and self.dead_tics > 35 and (self.ticks // 30) % 2:
            p.save()
            p.setFont(hfont(h / 18))
            fm = p.fontMetrics()
            msg = "PRESS USE TO TRY AGAIN"
            shadow_text(p, (w - fm.horizontalAdvance(msg)) / 2, h * 0.62, msg, QColor(230, 40, 24), 3)
            p.restore()

    def collect_sprites(self, px: float, py: float) -> list:
        """Every visible thing as (x, y, z, sprite id, mirrored, full bright, light, see-through)."""
        w = self.world
        g = w.grid
        S = self.sprites
        out = []
        t = w.leveltime
        things = S.things
        for mo in w.mobjs:
            if mo.removed:
                continue
            light = int(g.light[g.cell_at(mo.x, mo.y)])
            cls = mo.cls
            if cls == "monster":
                groups = S.monsters.get(mo.kind)
                fk = mo.frame
                if groups is None or not isinstance(fk, tuple):
                    continue
                group, i = fk
                lst = groups.get(group) or groups.get("death" if group == "gib" else "walk") or groups["walk"]
                ids, flips = lst[min(i, len(lst) - 1)]
                r = side_index(math.atan2(py - mo.y, px - mo.x) - mo.angle)
                out.append((mo.x, mo.y, mo.z, int(ids[r]), bool(flips[r]), bool(mo.bright), light,
                            bool(mo.info.get("shadow"))))
                continue
            if cls == "missile":
                if mo.source is w.pmo and mo.state != "boom" and math.hypot(mo.x - px, mo.y - py) < 40:
                    continue  # the player's own shot, still in front of the lens
                info = mo.info or MISSILE_INFO.get(mo.kind, {})
                if mo.state == "boom":
                    lst = things[info["boom"]]
                    ids, flips = lst[min(mo.si, len(lst) - 1)]
                else:
                    lst = things[info["fly"]]
                    ids, flips = lst[int(mo.frame) % len(lst)]
                r = side_index(math.atan2(py - mo.y, px - mo.x) - mo.angle)
                out.append((mo.x, mo.y, mo.z + mo.height / 2, int(ids[r]), bool(flips[r]), True, 255, False))
                continue
            if cls == "fx":
                key = FX_ART.get(mo.kind)
                if key is None:
                    continue
                lst = things[key]
                ids, _ = lst[min(mo.si, len(lst) - 1)]
                out.append((mo.x, mo.y, mo.z, int(ids[0]), False, mo.kind != "blood", light, False))
                continue
            if cls == "barrel":
                if mo.state == "boom":
                    lst = things["barrel_x"]
                    ids, _ = lst[min(mo.si + 1, len(lst) - 1)]
                    out.append((mo.x, mo.y, mo.z + 24, int(ids[0]), False, True, 255, False))
                else:
                    lst = things["barrel"]
                    ids, _ = lst[(t // 10) % len(lst)]
                    out.append((mo.x, mo.y, mo.z, int(ids[0]), False, False, light, False))
                continue
            key = ART_ALIAS.get(mo.kind, mo.kind)
            lst = things.get(key)
            if not lst:
                continue
            ids, _ = lst[(t // 6) % len(lst)]
            bright = key in S.bright or ITEM_INFO.get(mo.kind, {}).get("bright", False)
            out.append((mo.x, mo.y, mo.z, int(ids[0]), False, bool(bright), light, False))
        return out

    def weapon_quads(self) -> list:
        """The first-person weapon (and its muzzle flash) as (sprite id, left, top, brightness)."""
        w = self.world
        pl = w.player
        if pl.dead and pl.wy >= 96:
            return []
        frames, flashes = self.sprites.weapons.get(pl.weapon, ([], []))
        if not frames:
            return []
        fi = pl.wframe
        if pl.weapon == "saw" and pl.wstate != "fire":
            fi = (w.leveltime // 4) % 2
        sid, left, top, _ = frames[min(fi, len(frames) - 1)]
        bx = by = 0.0
        if pl.wstate == "ready":
            ang = w.leveltime * 128 / 8192 * 2 * math.pi
            bx = pl.bob * math.cos(ang)
            by = pl.bob * abs(math.sin(ang))
        dy = by + pl.wy
        g = w.grid
        light = int(g.light[g.cell_at(w.pmo.x, w.pmo.y)]) + pl.extralight * 16
        bright = min(max(0.38 + 0.62 * light / 224.0, 0.3), 1.0)
        out = []
        flash = None
        if pl.wflash >= 0 and flashes and pl.wstate == "fire":
            flash = flashes[min(pl.wflash, len(flashes) - 1)]
        if flash is not None and flash[3]:
            out.append((flash[0], flash[1] + bx, flash[2] + dy, 1.0))
        else:
            out.append((sid, left + bx, top + dy, 1.0 if flash is not None else bright))
            if flash is not None:
                out.append((flash[0], flash[1] + bx, flash[2] + dy, 1.0))
        return out

    def tint(self) -> tuple[float, float, float, float]:
        if self.screen != "play" or self.world is None:
            return (0.0, 0.0, 0.0, 0.0)
        pl = self.world.player
        red = pl.damagecount
        if pl.powers.get("berserk"):
            red = max(red, 12 - (self.world.leveltime >> 6))
        if red > 0:
            return (1.0, 0.0, 0.0, min(red, 100) / 100.0 * 0.62 + 0.06)
        if pl.dead:
            return (1.0, 0.0, 0.0, 0.25)
        if pl.bonuscount:
            return (0.85, 0.73, 0.27, min(pl.bonuscount, 40) * 0.009)
        suit = pl.powers.get("suit", 0)
        if suit > 4 * 35 or (suit and (suit // 8) % 2):
            return (0.0, 1.0, 0.0, 0.14)
        return (0.0, 0.0, 0.0, 0.0)

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        p = QPainter(self)
        try:
            w, h = self.width(), self.height()
            p.fillRect(self.rect(), QColor(0, 0, 0))
            self.compose(p, w, h)
        except Exception:  # noqa: BLE001 - a drawing error must never close the arcade
            if __import__("os").environ.get("TDS_ARCADE_STRICT"):
                p.end()
                raise
            log.warning("Hellfire frame not drawn", exc_info=True)
        p.end()


__all__ = ["HellfireGame", "EPISODE", "SKILLS", "WEAPONS", "VIEWHEIGHT", "CELL"]
