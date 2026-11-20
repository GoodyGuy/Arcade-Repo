"""The arcade's home screen: a neon arcade room with one cabinet per game.

* A 3D carousel of cabinets (the chosen one big in the middle, its live screen showing the game); arrows / A-D / the
  mouse wheel slide it, clicking a side cabinet brings it to the middle, Enter / Space / clicking the middle one inserts
  a coin and plays (``play`` is emitted at once; the window plays the coin / PLAYER 1 READY / CRT sequence over the
  starting game, see ``launch_fx``).
* Tickets (top left) for visiting, playing and catching the critters that wander by, spent at the Prize Counter (P)
  on themes, cabinet trims, mascots and toys; a Hall of Fame board, the chosen game's info and how to play, and an LED
  ticker along the bottom.
* After 20 s without input the cabinets cycle by themselves (attract mode). A chiptune plays while the home screen is
  shown in the active window (never offscreen or muted). Somewhere there is a secret code.
"""

from __future__ import annotations

import logging
import math
import random
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient, QTextOption
from PySide6.QtWidgets import QWidget

from app.arcade import audio, pixelfont
from app.arcade.base import font
from app.arcade.cabinet import TRIMS, CabinetArt, Carousel, Pose, ScreenFeed, draw_cabinet
from app.arcade.home_fx import NEON, CritterPen, Effects, LedTicker, TicketCounter, TitleLogo, crt_overlay, \
    draw_sprite, draw_ticket_icon, ease_out_cubic, speech_bubble
from app.arcade.home_scene import Scene, theme_swatch
from app.arcade.music import HomeAudio
from app.arcade.saves import SaveStore
from app.arcade.tickets import PRIZES, Prize, TicketBook

log = logging.getLogger(__name__)
_rng = random.Random()  # the arcade's own (the program's random numbers are not touched)

IDLE_ATTRACT_S = 20.0
ATTRACT_STEP_S = 3.4
PARTY_S = 20.0
KONAMI = (Qt.Key.Key_Up, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right,
          Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_B, Qt.Key.Key_A)
MASCOT_QUIPS = {
    "invader": ("BLEEP BLOOP!", "1UP!", "PICK ME!", "PEW PEW!", "HI-SCORE TIME"),
    "ghost": ("BOO!", "WOOO...", "HAUNT ME", "I'M FRIENDLY!", "SPOOKY FUN"),
    "cat": ("MEOW!", "PURRR...", "ONE MORE GAME?", "FEED ME TICKETS", "*YAWN*"),
}
PRIZE_COLS = 4
KIND_TITLES = {"theme": "THEMES", "trim": "CABINET TRIM", "mascot": "MASCOTS", "toy": "TOYS"}


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60} h {minutes % 60:02d} min played" if minutes >= 60 else f"{minutes} min played"


def _clip_text(text: str, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[:max(n - 2, 1)] + ".."


class HomeScreen(QWidget):
    """The game selector (see the module docstring)."""

    play = Signal(int)
    reset = Signal(int)
    close_arcade = Signal()
    controls = Signal(int)  # the Controls panel of a game (K or the CONTROLS button)
    toggle_mute = Signal()

    def __init__(self, games: list, store: SaveStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.games = list(games)
        self.store = store
        self.tickets = TicketBook(store)
        self.tickets.on_award = self._on_award
        self.audio = HomeAudio()
        self.selected = 0
        self.hover = -1
        self.pos = 0.0  # the carousel, in cabinets (not wrapped round)
        self.target = 0.0
        self.t = 0.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._t0 = time.perf_counter()
        self._last_frame = self._t0
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._animate)
        self.carousel = Carousel()
        self.arts: dict[int, CabinetArt] = {}
        self._art_key = None
        self.feeds = [ScreenFeed(cls) for cls in self.games]
        self.scene: Scene | None = None
        self._scene_key = None
        self._logo: TitleLogo | None = None
        self._caches: dict = {}
        self._swatches: dict[str, QPixmap] = {}
        self.fx = Effects()
        self.critters = CritterPen()
        self.counter = TicketCounter()
        self.counter.set(self.tickets.balance, instant=True)
        self.ticker = LedTicker()
        self._awards: list[tuple[int, str, QPointF | None]] = []
        self._award_origin: QPointF | None = None
        self.prizes_open = False
        self.prize_sel = 0
        self.prize_msg = ("", "#ffffff", 0.0)
        self.prize_shake = 0.0
        self.party_until = -1.0
        self.attract = False
        self._last_input = 0.0
        self._attract_next = 0.0
        self._keys: list[int] = []
        self.intro_t = 0.0  # when the logo intro / power-on started
        self.land = 0.0  # the middle cabinet's little hop when it arrives
        self.mascot_hop = 0.0
        self.bubble = ("", 0.0)  # mascot speech: text, until
        self._next_quip = 12.0
        self.board_page = 0
        self._board_next = 7.0
        self._data_version = 0
        self._frame = 0
        self.paint_ms = 0.0
        self._wheel = 0
        self.launch_fx: tuple[QPixmap, QRectF, QRectF, QColor] | None = None
        self._poses: list[Pose] = []
        self._errors: set = set()
        self._body_cache: dict = {}
        self._power_on = -10.0  # when the screen last switched on (the CRT effect)
        self._mid_pose: Pose | None = None
        self._mascot_rect = QRectF()
        self._logo_at = QPointF()
        self._hover_controls = False

    # --- life cycle -----------------------------------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        super().showEvent(event)
        now = time.perf_counter()
        self._last_frame = now
        self._last_input = self.t
        self.attract = False
        self._timer.start()
        self.setFocus()
        self.update_music()

    def hideEvent(self, event) -> None:  # noqa: N802, ANN001
        self._timer.stop()
        self.audio.set_music(False)
        super().hideEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().changeEvent(event)
        if event.type() == event.Type.ActivationChange:
            self.update_music()

    def _window_active(self) -> bool:
        w = self.window()
        return w.isActiveWindow() and not w.isMinimized()

    def update_music(self) -> None:
        """The music plays while the home screen is shown in the active arcade window (and not muted)."""
        self.audio.set_music(self.isVisible() and self._window_active() and not audio.is_muted())

    def stop_sounds(self) -> None:
        self.audio.stop_all()

    def on_opened(self) -> None:
        """The arcade window was opened: the daily bonus, the logo and the CRT switch on again."""
        self.intro_t = self.t
        self._power_on = self.t
        self.audio.play("power", 0.7)
        self.tickets.daily_visit()
        self.data_changed()

    def on_returned(self) -> None:
        """Back from a game: play time pays tickets."""
        self.intro_t = self.t - 1.4  # the logo is already there; the screen switches on
        self._power_on = self.t
        self.tickets.pay_play_time()
        self.data_changed()
        self.store.flush()

    def data_changed(self) -> None:
        """Progress or statistics changed: the panels are drawn again."""
        self._data_version += 1
        self.update()

    def game_launched(self, cls) -> None:  # noqa: ANN001
        """A game started (any way): a coin was inserted."""
        self.tickets.launch_bonus(cls.game_id)
        self.data_changed()

    # --- the clock ------------------------------------------------------------------------------------------------

    def _busy(self) -> bool:
        return (abs(self.target - self.pos) > 0.001 or self.fx.busy() or self.t - self.intro_t < 2.0
                or self.party_until > self.t or self.land > 0 or abs(self.counter.shown - self.counter.target) > 0.01
                or self.prize_shake > 0)

    def _animate(self) -> None:
        now = time.perf_counter()
        dt = min(now - self._last_frame, 0.1)
        self._last_frame = now
        try:
            self.step(dt)
        except Exception as exc:  # noqa: BLE001 - an animation bug never closes the arcade
            self._log_once("step", exc)
        active = self._window_active()  # in the background (the user is working): a few frames a second
        self._timer.setInterval(16 if self._busy() and active else (33 if active else 150))
        self.update()

    def _log_once(self, where: str, exc: Exception) -> None:
        key = (where, type(exc).__name__, str(exc))
        if key not in self._errors:
            self._errors.add(key)
            log.warning("Arcade home screen %s", where, exc_info=True)

    def step(self, dt: float) -> None:
        """Advance every animation by ``dt`` seconds (the timer; tests call it directly)."""
        self.t += dt
        n = len(self.games)
        unit = self.unit
        # the carousel
        if n:
            diff = self.target - self.pos
            if abs(diff) < 0.0015:
                if self.pos != self.target:
                    self.land = 1.0
                self.pos = self.target
            else:
                self.pos += diff * (1 - math.exp(-dt * 10.0))
        self.land = max(0.0, self.land - dt / 0.35)
        self.mascot_hop = max(0.0, self.mascot_hop - dt / 0.45)
        # attract mode
        idle = self.t - self._last_input
        if self.overlay_open():  # the Controls panel is over the home screen: not idle
            self._last_input = self.t
            self.attract = False
        if not self.attract and idle > IDLE_ATTRACT_S and not self.prizes_open and n > 1:
            self.attract = True
            self._attract_next = self.t + 0.5
        if self.attract and self.t >= self._attract_next:
            self._attract_next = self.t + ATTRACT_STEP_S
            self.move(1, sound=False, user=False)
        # tickets, effects, critters
        for row, (n_t, reason, origin) in enumerate(self._awards):
            self._show_award(n_t, reason, origin, row)
        self._awards.clear()
        landed = self.fx.update(dt, unit)
        if landed:
            self.counter.bump = 1.0
        self.counter.set(self.tickets.balance - self.fx.pending_tickets())
        if self.counter.update(dt) and self._frame % 3 == 0:
            self.audio.play("ticket", 0.5)
        lay = self.layout_rects()
        self.critters.update(dt, self.width(), self.height(), lay["lane"], lay["sky"], unit,
                             allow_spawn=not self.prizes_open and not self.overlay_open())
        if self.party_until > self.t:
            self.fx.confetti.rain(self.width(), 2 if self._frame % 2 else 1, unit)
        self.prize_shake = max(0.0, self.prize_shake - dt * 2.5)
        # the board and the mascot
        if self.t >= self._board_next:
            self._board_next = self.t + 7.0
            self.board_page = (self.board_page + 1) % len(self.board_pages())
        if self.t >= self._next_quip:
            self._next_quip = self.t + _rng.uniform(14, 26)
            mascot = self.tickets.equipped("mascot")
            if mascot in MASCOT_QUIPS:
                self.say(_rng.choice(MASCOT_QUIPS[mascot]))
        self._frame += 1

    # --- selection ------------------------------------------------------------------------------------------------

    def side_limits(self) -> tuple[float, float]:
        """The furthest places shown on the left and on the right at rest (the carousel goes round: a cabinet leaving
        one end fades out while it fades in at the other)."""
        n = len(self.games)
        if n <= 1:
            return 0.0, 0.0
        left = -float(math.ceil(n / 2) - 1)
        return left, left + n - 1

    def offset(self, index: int) -> float:
        """How many places cabinet ``index`` is from the middle (negative: on the left)."""
        n = len(self.games)
        if n <= 1:
            return index - self.pos
        left, _right = self.side_limits()
        return (index - self.pos - left) % n + left

    def move(self, step: int, sound: bool = True, user: bool = True) -> None:
        n = len(self.games)
        if not n:
            return
        if user:
            self._touch()
        if n == 1:
            if sound:
                self.audio.play("bump")
            self.mascot_hop = 1.0
            return
        self.target += step
        self.selected = int(round(self.target)) % n
        self.mascot_hop = 1.0
        if sound:
            self.audio.play("move")
        self.update()

    def select(self, index: int, k: float | None = None) -> None:
        """Slide cabinet ``index`` (shown ``k`` places from the middle) to the middle."""
        n = len(self.games)
        if not 0 <= index < n or index == self.selected:
            return
        if k is None:
            k = self.offset(index)
        self.target += round(k + (self.pos - self.target))
        self.selected = int(round(self.target)) % n
        self.mascot_hop = 1.0
        self.audio.play("move")
        self.update()

    def overlay_open(self) -> bool:
        """A panel of the window (the Controls panel) is over the home screen."""
        panel = getattr(self.window(), "controls_panel", None)
        return panel is not None and panel.isVisible()

    def request_controls(self) -> None:
        """The chosen game's Controls panel (the window opens it)."""
        if self.games:
            self._touch()
            self.audio.play("open")
            self.controls.emit(self.selected)

    def _touch(self) -> None:
        self._last_input = self.t
        if self.attract:
            self.attract = False

    def request_play(self, index: int | None = None) -> None:
        """Insert a coin: the selected game starts (``play``), with the launch sequence ready in ``launch_fx``."""
        if not self.games:
            return
        if index is not None and index != self.selected:
            self.select(index)
        self.pos = self.target  # the snapshot shows the cabinet in the middle
        self.launch_fx = None
        try:
            snap = self.grab()
            art = self.arts.get(self.selected)
            pose = next((ps for ps in self._poses if ps.index == self.selected), None)
            if art is not None and pose is not None:
                screen = pose.front.mapRect(art.screen_rect)
                slot = pose.front.mapRect(art.coin_rects[0])
                self.launch_fx = (snap, screen, slot, QColor(art.accent))
        except Exception:  # noqa: BLE001 - the launch never depends on the show
            log.debug("launch snapshot", exc_info=True)
        self.audio.play("coin")
        self.play.emit(self.selected)

    # --- tickets --------------------------------------------------------------------------------------------------

    def _on_award(self, n: int, reason: str) -> None:
        self._awards.append((n, reason, self._award_origin))

    def _show_award(self, n: int, reason: str, origin: QPointF | None, row: int = 0) -> None:
        lay = self.layout_rects()
        counter = lay["counter"]
        end = QPointF(counter.left() + counter.height() * 1.6, counter.center().y())
        start = origin or QPointF(self.width() / 2, self.height() * 0.62)  # out of the coin door
        self.fx.send_tickets(n, start, end, 26 * self.unit)
        # the reason under the ticket stub (never over a cabinet's screen), stacked when several come at once
        px = max(1.8 * self.unit, 1.2)
        self.fx.text(f"+{n} {reason}", QPointF(counter.center().x(), counter.bottom() + 10 * self.unit + row * px * 11),
                     "#ffe14d", px, 2.2 + 0.4 * row, rise=6.0)
        self.audio.play("bonus" if n >= 20 else "pop", 0.8)

    def say(self, text: str, seconds: float = 2.6) -> None:
        self.bubble = (text, self.t + seconds)

    def party(self, bonus: bool = True) -> None:
        """The secret: rainbow neon, confetti, disco lights (and tickets for the code)."""
        self.party_until = self.t + PARTY_S
        self.fx.confetti.rain(self.width(), 120, self.unit)
        for x in (0.25, 0.5, 0.75):
            self.fx.confetti.burst(QPointF(self.width() * x, self.height() * 0.6), 50, 700, self.unit)
        self.audio.play("party")
        self.say("PARTY TIME!", 3.0)
        if bonus:
            self._award_origin = QPointF(self.width() / 2, self.height() * 0.35)
            self.tickets.party()
            self._award_origin = None
            self.store.flush()

    @property
    def party_level(self) -> float:
        return min(max((self.party_until - self.t) / 1.0, 0.0), 1.0)

    # --- layout ---------------------------------------------------------------------------------------------------

    @property
    def unit(self) -> float:
        return max(min(self.width() / 1280, self.height() / 780), 0.5)

    def layout_rects(self) -> dict:
        key = (self.width(), self.height())
        cached = self._caches.get("layout")
        if cached is not None and cached[0] == key:
            return cached[1]
        w, h, u = self.width(), self.height(), self.unit
        panels_top = h * 0.79
        panels_bottom = h * 0.925
        lay = {
            "counter": QRectF(18 * u, 16 * u, 230 * u, 66 * u),
            "mute": QRectF(w - 62 * u, 20 * u, 44 * u, 44 * u),
            "prizes": QRectF(w - 62 * u - 168 * u, 20 * u, 156 * u, 44 * u),
            "logo_top": h * 0.012,
            "logo_h": min(h * 0.11, w * 0.08),  # the most: _fit_logo makes it fit the header
            "carousel_top": h * 0.175,
            "floor": h * 0.765,
            "board": QRectF(w * 0.012, panels_top, w * 0.3, panels_bottom - panels_top),
            "info": QRectF(w * 0.322, panels_top, w * 0.356, panels_bottom - panels_top),
            "howto": QRectF(w * 0.688, panels_top, w * 0.3, panels_bottom - panels_top),
            "controls": self._controls_rect(QRectF(w * 0.688, panels_top, w * 0.3, panels_bottom - panels_top), u),
            "ticker": QRectF(w * 0.012, h * 0.937, w * 0.976, h * 0.05),
            "lane": panels_top - 22 * u,
            "sky": h * 0.2,
        }
        self._caches["layout"] = (key, lay)
        return lay

    def _ensure_caches(self) -> None:
        w, h = self.width(), self.height()
        lay = self.layout_rects()
        self.carousel.layout(w, h, lay["carousel_top"], lay["floor"])
        theme = self.tickets.equipped("theme")
        key = (w, h, theme)
        if self._scene_key != key:
            self._scene_key = key
            self.scene = Scene(theme, w, h, self.carousel.horizon, self.carousel.floor_c, self.carousel.focal)
            q = QPainter(self.scene.static)  # the CRT look baked into the background (no full-screen pass a frame)
            q.drawPixmap(0, 0, crt_overlay(w, h))
            q.end()
            self._fit_logo(lay)
        trim = self.tickets.equipped("trim")
        dpr = self.devicePixelRatioF()
        art_key = (self.carousel.cab_px, trim, dpr, len(self.games))
        if self._art_key != art_key:
            self._art_key = art_key
            self.arts = {}
            self._body_cache = {}
        for i, cls in enumerate(self.games):
            if i not in self.arts:
                try:
                    self.arts[i] = CabinetArt(cls, self.carousel.cab_px, trim, dpr)
                except Exception:  # noqa: BLE001 - a game whose cabinet cannot be drawn is left out
                    log.warning("Arcade cabinet of %s not drawn", getattr(cls, "game_id", "?"), exc_info=True)
                    self.arts[i] = None  # type: ignore[assignment]

    def _fit_logo(self, lay: dict) -> None:
        """The title logo as big as the header allows: all of it (its animation included) inside the window with a
        margin, between the ticket stub and the buttons and above the middle cabinet."""
        w, u = self.width(), self.unit
        margin = 8 * u
        avail_h = lay["carousel_top"] - 4 * u - margin
        avail_w = 2 * min(w / 2 - lay["counter"].right(), lay["prizes"].left() - w / 2) - 2 * margin
        logo = TitleLogo(lay["logo_h"])
        b = logo.bounds(0.0, 0.0)
        scale = min(1.0, avail_h / max(b.height(), 1.0), max(avail_w, 40.0) / max(b.width(), 1.0))
        if scale < 0.999:
            logo = TitleLogo(lay["logo_h"] * scale)
            b = logo.bounds(0.0, 0.0)
        self._logo = logo
        self._logo_at = QPointF(w / 2 - b.center().x(), margin - b.top())

    def logo_bounds(self) -> QRectF:
        """Where the title logo may draw (tests)."""
        self._ensure_caches()
        return self._logo.bounds(self._logo_at.x(), self._logo_at.y())

    def poses(self) -> list[Pose]:
        """The cabinets to draw, far first (a cabinet going round the end is drawn at both ends, fading)."""
        n = len(self.games)
        out = []
        left, right = self.side_limits()
        for i in range(n):
            art = self.arts.get(i)
            if art is None:
                continue
            first = self.offset(i)
            for k in ((first, first - n) if n > 1 else (first,)):
                a = abs(k)
                opacity = 1.0
                if n > 1:
                    opacity = min(max(k - (left - 1), 0.0), max((right + 1) - k, 0.0), 1.0)
                opacity *= min(max((4.6 - a) / 0.8, 0.0), 1.0)
                if opacity <= 0.01:
                    continue
                lift = 0.0
                if a < 0.5:
                    hop = 0.035 * math.sin(math.pi * (1 - self.land)) if self.land > 0 else 0.0
                    lift = hop + 0.008 * math.sin(self.t * 2.2)
                    if self.hover == i:
                        lift += 0.02
                pose = self.carousel.pose(i, k, art, lift, opacity)
                whole = round(k)
                if a >= 1.0 and abs(k - whole) > 1e-6:  # sliding at the side: the whole-place picture, moved
                    pose = self.carousel.pose(i, whole, art, 0.0, opacity).moved(pose)
                box = pose.quad.boundingRect()
                if box.right() < -20 or box.left() > self.width() + 20:
                    continue
                out.append(pose)
        out.sort(key=lambda ps: -ps.depth)
        return out

    # --- drawing --------------------------------------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        start = time.perf_counter()
        self._ensure_caches()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            self._paint(p)
        except Exception as exc:  # noqa: BLE001 - a drawing bug never closes the arcade
            self._log_once("paint", exc)
        p.end()
        ms = (time.perf_counter() - start) * 1000
        self.paint_ms = ms if not self.paint_ms else self.paint_ms * 0.9 + ms * 0.1

    def _paint(self, p: QPainter) -> None:
        w, h, u = self.width(), self.height(), self.unit
        lay = self.layout_rects()
        party = self.party_level
        self.scene.paint(p, self.t, party)
        self._poses = self.poses()
        self._paint_cabinets(p, party)
        self._paint_header(p, lay, party)
        if self._mid_pose is not None:
            self._paint_mascot(p, self._mid_pose)
        self._paint_board(p, lay["board"])
        self._paint_info(p, lay["info"])
        self._paint_howto(p, lay["howto"])
        self._update_ticker()
        self.ticker.paint(p, lay["ticker"], self.t)
        self.critters.paint(p, self.t, u)
        self.fx.paint(p, u)
        if self.attract:
            self._paint_attract(p)
        if not self.games:
            self._paint_empty(p)
        if self.prizes_open:
            self._paint_prizes(p)
        on = self.t - self._power_on
        if on < 0.35:
            k = ease_out_cubic(on / 0.35)
            band = h * k
            p.fillRect(QRectF(0, 0, w, (h - band) / 2), QColor("black"))
            p.fillRect(QRectF(0, (h + band) / 2, w, (h - band) / 2), QColor("black"))
            p.fillRect(QRectF(0, (h - band) / 2, w, band), QColor(255, 255, 255, int(120 * (1 - k))))

    def _paint_cabinets(self, p: QPainter, party: float) -> None:
        now = self.t
        trim = self.tickets.equipped("trim")
        rainbow = trim == "rainbow" or party > 0
        mid = None
        for pose in self._poses:
            art = self.arts[pose.index]
            # a pool of light on the floor
            base, width = self.carousel.floor_point(pose.k)
            pool = self._pool(art.accent)
            p.setOpacity((0.9 if abs(pose.k) < 0.5 else 0.45) * pose.opacity)
            p.drawPixmap(QRectF(base.x() - width * 0.95, base.y() - width * 0.19, width * 1.9, width * 0.38), pool,
                         QRectF(pool.rect()))
            p.setOpacity(1.0)
            feed = self.feeds[pose.index]
            a = abs(pose.k)
            # the live screens: 30 fps in the middle, 15 next to it (and while sliding), 4 further away
            sliding = self.pos != self.target
            interval = (1 / 15 if sliding else 1 / 30) if a < 0.5 else (1 / 15 if a < 1.5 else 0.25)
            if feed.due(now, interval):
                art_scr = art.screen_rect
                feed.refresh(art_scr.width(), art_scr.height(), now, now, art.dpr, art.glass, pose.dim)
            middle = a < 0.5
            glow = 0.5 + 0.5 * math.sin(now * 3.1 + pose.index) if middle else 0.0
            blink = middle and int(now * 2.5) % 2 == 0
            hue = (now * 0.35 + pose.index * 0.13) % 1.0 if rainbow else None
            still = self.pos == self.target or pose.ref is not None
            draw_cabinet(p, art, pose, feed, now, hue, glow * 0.6, blink, reflection=max(0.0, 1.0 - 2 * a),
                         cache=self._body_cache if still and not middle else None, smooth=still or middle)
            if middle:
                mid = pose
            if pose.index == self.hover and not middle:
                p.setPen(QPen(QColor(255, 255, 255, int(120 + 100 * math.sin(now * 8))), 2.5 * self.unit))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(pose.quad)
        if mid is not None:
            art = self.arts[mid.index]
            scr = mid.front.mapRect(art.screen_rect)
            if int(self.t * 1.6) % 2 == 0:
                px = max(scr.width() / 95, 1.2)
                label = "PRESS START" if self.attract else "INSERT COIN"
                r = pixelfont.draw(p, label, scr.center().x(), scr.bottom() - px * 11, px, "#ffffff",
                                   Qt.AlignmentFlag.AlignHCenter, glow=True)
                del r
            self._mid_pose = mid
        else:
            self._mid_pose = None

    def _pool(self, color: QColor) -> QPixmap:
        """A soft round pool of light in ``color`` (drawn squashed on the floor)."""
        key = ("pool", QColor(color).rgb())
        pm = self._caches.get(key)
        if pm is None:
            pm = QPixmap(128, 128)
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            c = QColor(color)
            c.setAlpha(120)
            q.fillRect(pm.rect(), _radial(QPointF(64, 64), 64, c))
            q.end()
            self._caches[key] = pm
        return pm

    def _paint_mascot(self, p: QPainter, pose: Pose) -> None:
        mascot = self.tickets.equipped("mascot")
        q = pose.quad
        if q.size() < 2:
            return
        a, b = q.at(0), q.at(1)
        top = QPointF(a.x() + (b.x() - a.x()) * 0.78, a.y() + (b.y() - a.y()) * 0.78)
        px = 3.2 * self.unit
        hop = math.sin(math.pi * self.mascot_hop) * 22 * self.unit
        if mascot != "none":
            kind = {"invader": "invader", "ghost": "ghost", "cat": "cat"}[mascot]
            img_h = {"invader": 8, "ghost": 14, "cat": 12}[kind]
            frame = int(self.t * 2) % 2 if kind != "cat" else (1 if int(self.t * 10) % 37 == 0 else 0)
            bob = math.sin(self.t * 3) * 2 * self.unit if kind != "cat" else 0.0
            centre = QPointF(top.x(), top.y() - img_h * px / 2 + px * (3 if kind == "cat" else 0) - hop + bob)
            draw_sprite(p, kind, centre, px, frame, glow=QColor(NEON[0]) if self.party_level else None)
            self._mascot_rect = QRectF(centre.x() - 8 * px, centre.y() - img_h * px / 2, 16 * px, img_h * px)
            anchor = QPointF(centre.x(), centre.y() - img_h * px / 2 - 2 * px)
        else:
            self._mascot_rect = QRectF()
            anchor = QPointF(top.x(), top.y() - 4 * px)
        text, until = self.bubble
        if text and self.t < until and mascot != "none":
            alpha = min(1.0, (until - self.t) / 0.3)
            speech_bubble(p, text, anchor, max(2.0 * self.unit, 1.2), alpha, side=True)

    def _paint_header(self, p: QPainter, lay: dict, party: float) -> None:
        u = self.unit
        logo = self._logo
        logo.paint(p, self._logo_at.x(), self._logo_at.y(), self.t, self.t - self.intro_t, party)
        self.counter.paint(p, lay["counter"])
        # prize counter button (its face cached, the border pulses)
        r = lay["prizes"]
        self._button(p, "prizes", r, self._draw_prizes_button)
        pulse = 0.5 + 0.5 * math.sin(self.t * 3)
        p.setPen(QPen(QColor(255, 225, 77, int(150 + 100 * pulse)), 2 * u))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, 10 * u, 10 * u)
        m = lay["mute"]
        muted = audio.is_muted()
        self._button(p, f"mute{muted}", m, self._paint_mute)
        if not muted and self.audio.music_on:  # the speaker pulses with the music
            beat = max(0.0, math.sin(self.t * 11.7))
            p.setPen(QPen(QColor(41, 243, 255, int(160 * beat)), 2 * u))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(m.adjusted(-2 * u, -2 * u, 2 * u, 2 * u), 11 * u, 11 * u)

    def _button(self, p: QPainter, name: str, r: QRectF, draw) -> None:  # noqa: ANN001
        key = ("button", name, int(r.width()), int(r.height()))
        pm = self._caches.get(key)
        if pm is None:
            dpr = self.devicePixelRatioF()
            pm = QPixmap(int((r.width() + 8) * dpr), int((r.height() + 8) * dpr))
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            draw(q, QRectF(4, 4, r.width(), r.height()))
            q.end()
            self._caches[key] = pm
        p.drawPixmap(r.topLeft() - QPointF(4, 4), pm)

    def _draw_prizes_button(self, p: QPainter, r: QRectF) -> None:
        u = self.unit
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 8, 30, 210))
        p.drawRoundedRect(r, 10 * u, 10 * u)
        self._gift_icon(p, QPointF(r.left() + 24 * u, r.center().y()), 14 * u)
        pixelfont.draw(p, "PRIZES", r.left() + 46 * u, r.center().y() - 7 * u, 2.0 * u, "#ffe14d", glow=True)
        pixelfont.draw(p, "P", r.right() - 12 * u, r.center().y() - 5 * u, 1.4 * u, "#9a8aa8",
                       Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _gift_icon(p: QPainter, c: QPointF, s: float) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ff3fb4"))
        p.drawRect(QRectF(c.x() - s, c.y() - s * 0.4, 2 * s, s * 1.4))
        p.setBrush(QColor("#ff6fc8"))
        p.drawRect(QRectF(c.x() - s * 1.12, c.y() - s * 0.75, 2.24 * s, s * 0.4))
        p.setBrush(QColor("#ffe14d"))
        p.drawRect(QRectF(c.x() - s * 0.18, c.y() - s * 0.75, s * 0.36, s * 1.75))
        p.setPen(QPen(QColor("#ffe14d"), s * 0.18))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(c.x() - s * 0.35, c.y() - s * 0.95), s * 0.35, s * 0.22)
        p.drawEllipse(QPointF(c.x() + s * 0.35, c.y() - s * 0.95), s * 0.35, s * 0.22)

    def _paint_mute(self, p: QPainter, m: QRectF) -> None:
        u = self.unit
        p.setPen(QPen(QColor(255, 255, 255, 200), 2 * u))
        p.setBrush(QColor(20, 8, 30, 210))
        p.drawRoundedRect(m, 10 * u, 10 * u)
        cx, cy = m.center().x() - 3 * u, m.center().y()
        speaker = QPainterPath()
        speaker.moveTo(cx - 11 * u, cy - 5 * u)
        speaker.lineTo(cx - 5 * u, cy - 5 * u)
        speaker.lineTo(cx + 2 * u, cy - 11 * u)
        speaker.lineTo(cx + 2 * u, cy + 11 * u)
        speaker.lineTo(cx - 5 * u, cy + 5 * u)
        speaker.lineTo(cx - 11 * u, cy + 5 * u)
        speaker.closeSubpath()
        p.fillPath(speaker, QColor("white"))
        if audio.is_muted():
            p.setPen(QPen(QColor("#ff4d4d"), 3 * u))
            p.drawLine(QPointF(cx + 6 * u, cy - 7 * u), QPointF(cx + 16 * u, cy + 7 * u))
            p.drawLine(QPointF(cx + 16 * u, cy - 7 * u), QPointF(cx + 6 * u, cy + 7 * u))
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("white"), 2 * u))
            p.drawArc(QRectF(cx - 4 * u, cy - 8 * u, 14 * u, 16 * u), -60 * 16, 120 * 16)
            p.drawArc(QRectF(cx - 4 * u, cy - 13 * u, 22 * u, 26 * u), -60 * 16, 120 * 16)

    def _panel(self, p: QPainter, r: QRectF, color: QColor, title: str | None = None) -> None:
        u = self.unit
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(8, 4, 18, 215))
        p.drawRoundedRect(r, 10 * u, 10 * u)
        for k, a in ((5, 30), (2.5, 70), (1.2, 230)):
            c = QColor(color)
            c.setAlpha(a)
            p.setPen(QPen(c, k * u))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r, 10 * u, 10 * u)
        if title:
            px = max(1.7 * u, 1.0)
            tw = pixelfont.width(title, px)
            tab = QRectF(r.left() + 14 * u, r.top() - px * 5, tw + px * 6, px * 10)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(color))
            p.drawRoundedRect(tab, 4 * u, 4 * u)
            pixelfont.draw(p, title, tab.left() + px * 3, tab.top() + px * 1.5, px, "#12002a")

    def _cached(self, p: QPainter, name: str, key: tuple, rect: QRectF, draw,  # noqa: ANN001
                frame: tuple | None = None) -> None:
        """Draw a panel (``frame``: its colour and title) and its contents once into a pixmap while ``key`` stays the
        same, then paint that pixmap at ``rect``."""
        cached = self._caches.get(name)
        full = (key, frame, int(rect.width()), int(rect.height()), self._data_version)
        m = int(16 * self.unit) + 2
        if cached is None or cached[0] != full:
            dpr = self.devicePixelRatioF()
            pm = QPixmap(max(int((rect.width() + 2 * m) * dpr), 1), max(int((rect.height() + 2 * m) * dpr), 1))
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            q.translate(m, m)
            local = QRectF(0, 0, rect.width(), rect.height())
            try:
                if frame is not None:
                    self._panel(q, local, QColor(frame[0]), frame[1])
                draw(q, local)
            except Exception:  # noqa: BLE001
                log.debug("arcade panel %s", name, exc_info=True)
            q.end()
            cached = (full, pm)
            self._caches[name] = cached
        p_rect = rect.adjusted(-m, -m, m, m)
        p.drawPixmap(p_rect.topLeft(), cached[1])

    def _summary(self, cls) -> str:  # noqa: ANN001
        data = self.store.game(cls.game_id)
        try:
            return str(cls.summary(data))
        except Exception:  # noqa: BLE001
            return "Continue" if data else "New game"

    def _played(self, cls) -> float:  # noqa: ANN001
        return float(self.store.data["stats"]["play_seconds"].get(cls.game_id, 0.0))

    # Hall of Fame -----------------------------------------------------------------------------------------------

    def _board_rows(self) -> int:
        """Games per page of the board (more games: more pages)."""
        u = self.unit
        line = max(1.45 * u, 1.0) * 10.5
        return max(int((self.layout_rects()["board"].height() - 22 * u) / line) - 1, 1)

    def board_pages(self) -> list[tuple[str, int]]:
        """The pages the board turns through: (what, first game) - high scores and most played split over as many
        pages as the games need, then the arcade records."""
        per, n = self._board_rows(), len(self.games)
        firsts = list(range(0, max(n, 1), per))
        return [("scores", i) for i in firsts] + [("played", i) for i in firsts] + [("records", 0)]

    def _paint_board(self, p: QPainter, r: QRectF) -> None:
        pages = self.board_pages()
        self.board_page %= len(pages)
        self._cached(p, "board", (pages[self.board_page], len(self.games)), r, self._draw_board,
                     ("#29f3ff", "HALL OF FAME"))
        u = self.unit
        px = max(1.5 * u, 1.0)
        if int(self.t * 2) % 2 == 0:
            pixelfont.draw(p, "1UP", r.right() - 12 * u, r.top() + 9 * u, px, "#ff4d4d", Qt.AlignmentFlag.AlignRight,
                           glow=True)
        n = len(pages)
        for i in range(n):
            c = QColor("#29f3ff") if i == self.board_page else QColor(255, 255, 255, 60)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawEllipse(QPointF(r.right() - 14 * u - (n - 1 - i) * 9 * u, r.bottom() - 9 * u), 2.6 * u, 2.6 * u)

    def _draw_board(self, p: QPainter, r: QRectF) -> None:
        u = self.unit
        px = max(1.45 * u, 1.0)
        line = px * 10.5
        x0, y = r.left() + 14 * u, r.top() + 15 * u
        per = self._board_rows()
        rows_fit = per + 1
        max_chars = int((r.width() - 28 * u) / (px * 6))
        what, first = self.board_pages()[self.board_page % len(self.board_pages())]
        page = {"scores": 0, "played": 1}.get(what, 2)
        stats = self.store.data["stats"]
        if page == 0:
            pixelfont.draw(p, "HIGH SCORES", x0, y, px, "#ffe14d")
            y += line
            if not self.games:
                pixelfont.draw(p, "NO GAMES YET", x0, y, px, "#ffffff")
            for i, cls in enumerate(self.games[first:first + per], first):
                color = _bright(cls.accent)
                rank = ("1ST", "2ND", "3RD")[i] if i < 3 else f"{i + 1}TH"
                pixelfont.draw(p, rank, x0, y, px, ("#ffe14d", "#c9d4e0", "#ff9a3c")[i] if i < 3 else "#9a8aa8")
                name = _clip_text(cls.title, 11)
                pixelfont.draw(p, name, x0 + px * 6 * 4.5, y, px, color)
                rest = max_chars - 17
                if rest > 4:
                    pixelfont.draw(p, _clip_text(self._summary(cls), rest), r.right() - 14 * u, y, px, "#ffffff",
                                   Qt.AlignmentFlag.AlignRight)
                y += line
        elif page == 1:
            pixelfont.draw(p, "MOST PLAYED", x0, y, px, "#ffe14d")
            y += line
            ranked = sorted(self.games, key=self._played, reverse=True)
            top = max((self._played(c) for c in ranked), default=0.0) or 1.0
            if not ranked:
                pixelfont.draw(p, "NO GAMES YET", x0, y, px, "#ffffff")
            for cls in ranked[first:first + per]:
                played = self._played(cls)
                pixelfont.draw(p, _clip_text(cls.title, 11), x0, y, px, _bright(cls.accent))
                bar_x = x0 + px * 6 * 12
                minutes = int(played // 60)
                label = f"{minutes // 60}H{minutes % 60:02d}" if minutes >= 60 else f"{minutes}M"
                bar_w = max(r.right() - 14 * u - bar_x - pixelfont.width("0H00 ", px), 4)
                p.fillRect(QRectF(bar_x, y + px, bar_w, px * 5), QColor(255, 255, 255, 25))
                p.fillRect(QRectF(bar_x, y + px, bar_w * played / top, px * 5), QColor(cls.accent))
                pixelfont.draw(p, label, r.right() - 14 * u, y, px, "#ffffff", Qt.AlignmentFlag.AlignRight)
                y += line
        else:
            pixelfont.draw(p, "ARCADE RECORDS", x0, y, px, "#ffe14d")
            y += line
            total = sum(float(v) for v in stats.get("play_seconds", {}).values())
            rows = (("VISITS", stats.get("opened", 0)), ("COINS IN", stats.get("coins", 0)),
                    ("TICKETS WON", stats.get("tickets_earned", 0)), ("CRITTERS", stats.get("critters", 0)),
                    ("DAY STREAK", stats.get("best_streak", 0)), ("PLAY TIME", f"{int(total // 3600)}H"
                                                                   f"{int(total % 3600 // 60):02d}"))
            half = (r.width() - 28 * u) / 2
            for i, (label, value) in enumerate(rows):
                col, row = divmod(i, max(rows_fit - 1, 1))
                if col > 1:
                    break
                xx = x0 + col * (half + 8 * u)
                pixelfont.draw(p, label, xx, y + row * line, px, "#9fe8ff")
                pixelfont.draw(p, str(value), xx + half - 8 * u, y + row * line, px, "#ffffff",
                               Qt.AlignmentFlag.AlignRight)

    # the chosen game ---------------------------------------------------------------------------------------------

    def _selected_cls(self):  # noqa: ANN202
        return self.games[self.selected] if 0 <= self.selected < len(self.games) else None

    def _paint_info(self, p: QPainter, r: QRectF) -> None:
        cls = self._selected_cls()
        if cls is None:
            self._cached(p, "info", (None,), r, lambda q, rr: None, ("#ff3fb4", None))
            return
        self._cached(p, "info", (self.selected, id(cls)), r, self._draw_info, (QColor(cls.accent).name(), None))
        u = self.unit
        if int(self.t * 2.2) % 3 != 2:
            px = max(1.6 * u, 1.0)
            pixelfont.draw(p, "▶ PRESS ENTER TO PLAY ◀", r.center().x(), r.bottom() - px * 7 - 6 * u, px, "#ffe14d",
                           Qt.AlignmentFlag.AlignHCenter, glow=True)

    def _draw_info(self, p: QPainter, r: QRectF) -> None:
        cls = self._selected_cls()
        u = self.unit
        accent = QColor(cls.accent)
        title_px = min(r.height() * 0.26, 30 * u)
        f = font(title_px, True, "Impact")
        p.setFont(f)
        wide = p.fontMetrics().horizontalAdvance(str(cls.title).upper())
        if wide > r.width() - 24 * u:  # a long title is made to fit
            title_px *= (r.width() - 24 * u) / wide
            f = font(title_px, True, "Impact")
            p.setFont(f)
        tr = QRectF(r.left(), r.top() + 4 * u, r.width(), title_px * 1.3)
        glow = QColor(accent)
        glow.setAlpha(110)
        p.setPen(glow)
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            p.drawText(tr.translated(dx * u, dy * u), Qt.AlignmentFlag.AlignCenter, str(cls.title).upper())
        p.setPen(QColor("white"))
        p.drawText(tr, Qt.AlignmentFlag.AlignCenter, str(cls.title).upper())
        y = tr.bottom()
        p.setFont(font(max(12 * u, 9)))
        p.setPen(QColor(255, 255, 255, 190))
        sub = str(getattr(cls, "subtitle", "") or "")
        p.drawText(QRectF(r.left(), y - 2 * u, r.width(), 16 * u), Qt.AlignmentFlag.AlignCenter, sub)
        y += 15 * u
        px = max(1.45 * u, 1.0)
        line = _clip_text(self._summary(cls), int(r.width() / (px * 6)) - 2)
        played = self._played(cls)
        pixelfont.draw(p, line, r.center().x(), y + 2 * u, px, _bright(accent, 0.68),
                       Qt.AlignmentFlag.AlignHCenter)
        if played >= 60:
            pixelfont.draw(p, fmt_time(played).upper(), r.center().x(), y + 2 * u + px * 10, px * 0.85, "#9a8aa8",
                           Qt.AlignmentFlag.AlignHCenter)

    def _paint_howto(self, p: QPainter, r: QRectF) -> None:
        cls = self._selected_cls()
        draw = self._draw_howto if cls is not None else (lambda q, rr: None)
        self._cached(p, "howto", (self.selected, id(cls)), r, draw, ("#ffe14d", "HOW TO PLAY"))
        if cls is not None:
            self._paint_controls_button(p, self.layout_rects()["controls"])

    @staticmethod
    def _controls_rect(card: QRectF, u: float) -> QRectF:
        """The CONTROLS button, on the top edge of the How-to-play card (like its title tab)."""
        px = max(1.9 * u, 1.2)
        w = pixelfont.width("K CONTROLS", px) + px * 14
        h = px * 11
        return QRectF(card.right() - 14 * u - w, card.top() - h / 2, w, h)

    def _paint_controls_button(self, p: QPainter, r: QRectF) -> None:
        u = self.unit
        px = r.height() / 11
        hot = self._hover_controls
        pulse = 0.5 + 0.5 * math.sin(self.t * 3.3)
        glow = QColor(41, 243, 255, int(60 + 80 * pulse) if not hot else 200)
        p.setPen(QPen(glow, 5 * u))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setPen(QPen(QColor("#29f3ff"), 1.5 * u))
        p.setBrush(QColor("#29f3ff") if hot else QColor(8, 30, 44, 240))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        key = QRectF(r.left() + px * 4, r.top() + px * 1.5, px * 9, px * 8)  # a little keycap with K
        p.setPen(QPen(QColor("#12002a") if hot else QColor("#29f3ff"), 1.2 * u))
        p.setBrush(QColor(255, 255, 255, 40))
        p.drawRoundedRect(key, px * 1.5, px * 1.5)
        color = "#12002a" if hot else "#e9fdff"
        pixelfont.draw(p, "K", key.center().x(), key.top() + px * 0.5, px, color, Qt.AlignmentFlag.AlignHCenter)
        pixelfont.draw(p, "CONTROLS", key.right() + px * 3, r.top() + px * 2, px, color)

    def _draw_howto(self, p: QPainter, r: QRectF) -> None:
        cls = self._selected_cls()
        u = self.unit
        x0, y = r.left() + 14 * u, r.top() + 13 * u
        width = r.width() - 28 * u
        size = max(11.5 * u, 8.5)
        f = font(size)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255, 215))
        desc = str(getattr(cls, "description", "") or getattr(cls, "subtitle", "") or "")
        controls = list(getattr(cls, "controls", ()) or ())
        lines = 2 if controls else 4
        text_rect = QRectF(x0, y, width, size * 1.3 * lines)
        opt = QTextOption()
        opt.setWrapMode(QTextOption.WrapMode.WordWrap)
        p.drawText(text_rect, _clip_text(desc, int(width / (size * 0.48)) * lines), opt)
        y = text_rect.bottom() + 4 * u
        if not controls:
            return
        rows = max(int((r.bottom() - 6 * u - y) / (size * 1.55)), 1)
        col_w = width / 2
        small = font(max(10 * u, 8), True)
        for i, (keys, action) in enumerate(controls[:rows * 2]):
            col, row = divmod(i, rows)
            xx = x0 + col * col_w
            yy = y + row * size * 1.55
            p.setFont(small)
            label = p.fontMetrics().elidedText(str(keys), Qt.TextElideMode.ElideRight, int(col_w * 0.5))
            kw = p.fontMetrics().horizontalAdvance(label) + 10 * u
            key_rect = QRectF(xx, yy, kw, size * 1.3)
            p.setPen(QPen(QColor("#ffe14d"), 1.2 * u))
            p.setBrush(QColor(255, 225, 77, 30))
            p.drawRoundedRect(key_rect, 4 * u, 4 * u)
            p.setPen(QColor("#ffe14d"))
            p.drawText(key_rect, Qt.AlignmentFlag.AlignCenter, label)
            p.setFont(font(max(10.5 * u, 8)))
            p.setPen(QColor(255, 255, 255, 200))
            p.drawText(QRectF(key_rect.right() + 6 * u, yy, col_w - kw - 10 * u, size * 1.3),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       p.fontMetrics().elidedText(str(action), Qt.TextElideMode.ElideRight,
                                                  int(col_w - kw - 10 * u)))

    # the ticker ---------------------------------------------------------------------------------------------------

    def _update_ticker(self) -> None:
        key = (self._data_version, len(self.games))
        if self._caches.get("ticker") == key:
            return
        self._caches["ticker"] = key
        msgs = [("★ WELCOME TO THE BREAK ROOM ARCADE ★", "#ffe14d")]
        total = 0.0
        for cls in self.games:
            played = self._played(cls)
            total += played
            line = f"{cls.title}: {self._summary(cls)}"
            if played >= 60:
                line += f" · {int(played // 60)} MIN"
            msgs.append((line, _bright(cls.accent)))
        stats = self.store.data["stats"]
        msgs += [
            (f"TOTAL PLAY TIME {int(total // 3600)}H {int(total % 3600 // 60):02d}M", "#29f3ff"),
            (f"TICKETS WON {stats.get('tickets_earned', 0)} · COINS IN {stats.get('coins', 0)}", "#ff9a2e"),
            ("TIP: HOLD CTRL+6+7+9 ANYWHERE TO COME BACK", "#7dff4a"),
            ("TIP: CLICK THE CRITTERS FOR TICKETS", "#ff3fb4"),
            ("TIP: SPEND TICKETS AT THE PRIZE COUNTER (P)", "#ffe14d"),
            ("TIP: PRESS K (OR CLICK CONTROLS) TO SET UP A GAME'S KEYS AND MOUSE", "#29f3ff"),
            ("TIP: COME BACK EVERY DAY FOR A BIGGER BONUS", "#29f3ff"),
            ("LEGEND SAYS AN OLD SECRET CODE STILL WORKS HERE...", "#a66bff"),
        ]
        self.ticker.set_messages(msgs)

    def _paint_attract(self, p: QPainter) -> None:
        if int(self.t * 2) % 2:
            return
        px = max(4.2 * self.unit, 2.0)
        pixelfont.draw(p, "PRESS START", self.width() / 2, self.height() * 0.70, px, "#ffffff",
                       Qt.AlignmentFlag.AlignHCenter, glow=True)

    def _paint_empty(self, p: QPainter) -> None:
        px = max(3.0 * self.unit, 1.5)
        pixelfont.draw(p, "THE GAMES ARE BEING INSTALLED", self.width() / 2, self.height() * 0.4, px, "#ffe14d",
                       Qt.AlignmentFlag.AlignHCenter, glow=True)
        pixelfont.draw(p, "COME BACK SOON!", self.width() / 2, self.height() * 0.4 + px * 12, px, "#29f3ff",
                       Qt.AlignmentFlag.AlignHCenter, glow=True)

    # --- the prize counter ----------------------------------------------------------------------------------------

    def open_prizes(self) -> None:
        self._touch()
        self.prizes_open = True
        self.prize_msg = ("", "#ffffff", 0.0)
        self.audio.play("open")
        self.update()

    def close_prizes(self) -> None:
        self.prizes_open = False
        self.audio.play("close")
        self.update()

    def prize_rects(self) -> list[tuple[Prize, QRectF]]:
        w, h, u = self.width(), self.height(), self.unit
        panel = self._prize_panel()
        kinds = ("theme", "trim", "mascot", "toy")
        row_h = (panel.height() - 120 * u) / len(kinds)
        card_w = (panel.width() - 190 * u - (PRIZE_COLS - 1) * 12 * u) / PRIZE_COLS
        out = []
        for r_i, kind in enumerate(kinds):
            items = [pz for pz in PRIZES if pz.kind == kind]
            for c_i, pz in enumerate(items):
                x = panel.left() + 170 * u + c_i * (card_w + 12 * u)
                y = panel.top() + 72 * u + r_i * row_h
                out.append((pz, QRectF(x, y, card_w, row_h - 10 * u)))
        del w, h
        return out

    def _prize_panel(self) -> QRectF:
        w, h = self.width(), self.height()
        pw, ph = min(w * 0.86, 1240 * self.unit), h * 0.8
        return QRectF((w - pw) / 2, (h - ph) / 2 + h * 0.03, pw, ph)

    def _swatch(self, theme: str, w: int, h: int) -> QPixmap:
        key = f"{theme}:{w}x{h}"
        pm = self._swatches.get(key)
        if pm is None:
            pm = self._swatches[key] = theme_swatch(theme, max(w, 8), max(h, 8))
        return pm

    def _paint_prizes(self, p: QPainter) -> None:
        w, h, u = self.width(), self.height(), self.unit
        p.fillRect(self.rect(), QColor(0, 0, 0, 170))
        panel = self._prize_panel()
        shake = math.sin(self.prize_shake * 40) * 8 * u * self.prize_shake
        p.save()
        p.translate(shake, 0)
        self._panel(p, panel, QColor("#ffe14d"))
        title_px = max(3.4 * u, 2.0)
        pixelfont.draw(p, "PRIZE COUNTER", panel.center().x(), panel.top() + 18 * u, title_px, "#ffe14d",
                       Qt.AlignmentFlag.AlignHCenter, glow=True)
        draw_ticket_icon(p, QPointF(panel.right() - 150 * u, panel.top() + 30 * u), 30 * u)
        pixelfont.draw(p, f"{self.tickets.balance}", panel.right() - 128 * u, panel.top() + 21 * u, 2.6 * u, "#ffd27a",
                       glow=True)
        rects = self.prize_rects()
        kinds_done = set()
        for i, (pz, r) in enumerate(rects):
            if pz.kind not in kinds_done:
                kinds_done.add(pz.kind)
                pixelfont.draw(p, KIND_TITLES[pz.kind], panel.left() + 24 * u, r.center().y() - 6 * u, 1.9 * u,
                               "#29f3ff", glow=True)
            self._prize_card(p, pz, r, i == self.prize_sel)
        text, color, until = self.prize_msg
        footer_y = panel.bottom() - 34 * u
        if text and self.t < until:
            pixelfont.draw(p, text, panel.center().x(), footer_y, 2.4 * u, color, Qt.AlignmentFlag.AlignHCenter,
                           glow=True)
        else:
            pixelfont.draw(p, "← → ↑ ↓ CHOOSE   ENTER BUY / USE   ESC BACK", panel.center().x(), footer_y, 1.8 * u,
                           "#9a8aa8", Qt.AlignmentFlag.AlignHCenter)
        p.restore()
        del w, h

    def _prize_card(self, p: QPainter, pz: Prize, r: QRectF, selected: bool) -> None:
        u = self.unit
        owned = self.tickets.owns(pz.prize_id)
        in_use = self.tickets.in_use(pz.prize_id)
        border = QColor("#ffe14d") if selected else (QColor("#7dff4a") if in_use else QColor(255, 255, 255, 60))
        if selected:
            glow = QColor(border)
            glow.setAlpha(60)
            p.setPen(QPen(glow, 8 * u))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r, 8 * u, 8 * u)
        p.setPen(QPen(border, 2 * u))
        p.setBrush(QColor(30, 14, 48, 235) if selected else QColor(18, 8, 30, 235))
        p.drawRoundedRect(r, 8 * u, 8 * u)
        pic = QRectF(r.left() + 6 * u, r.top() + 6 * u, r.height() * 1.25, r.height() - 12 * u)
        if pic.width() > r.width() * 0.42:
            pic.setWidth(r.width() * 0.42)
        self._prize_picture(p, pz, pic)
        tx = pic.right() + 8 * u
        tw = r.right() - 6 * u - tx
        px = max(min(1.6 * u, tw / (6 * max(len(pz.name), 1))), 0.9)
        pixelfont.draw(p, pz.name, tx, r.top() + 9 * u, px, "#ffffff")
        p.setFont(font(max(10 * u, 7.5)))
        p.setPen(QColor(255, 255, 255, 150))
        p.drawText(QRectF(tx, r.top() + 9 * u + px * 9, tw, r.height() * 0.4),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, pz.blurb)
        by = r.bottom() - 18 * u
        if in_use:
            pixelfont.draw(p, "IN USE" if pz.kind != "toy" else "OWNED", tx, by, 1.6 * u, "#7dff4a", glow=True)
        elif owned:
            pixelfont.draw(p, "OWNED - USE", tx, by, 1.6 * u, "#29f3ff")
        else:
            draw_ticket_icon(p, QPointF(tx + 9 * u, by + 5 * u), 18 * u)
            short = self.tickets.balance < pz.price
            pixelfont.draw(p, str(pz.price), tx + 22 * u, by, 1.8 * u, "#ff6b6b" if short else "#ffd27a", glow=True)

    def _prize_picture(self, p: QPainter, pz: Prize, r: QRectF) -> None:
        u = self.unit
        p.save()
        p.setClipRect(r)
        p.fillRect(r, QColor("#05020c"))
        if pz.kind == "theme":
            p.drawPixmap(r, self._swatch(pz.value, int(r.width()), int(r.height())), QRectF())
        elif pz.kind == "trim":
            if pz.value in TRIMS:
                hi, lo = (QColor(c) for c in TRIMS[pz.value])
            else:
                hi, lo = QColor("#ff6fc8"), QColor("#29f3ff")
            body = QRectF(r.center().x() - r.width() * 0.22, r.top() + r.height() * 0.1, r.width() * 0.44,
                          r.height() * 0.85)
            g = QLinearGradient(body.topLeft(), body.bottomRight())
            if pz.value == "rainbow":
                for i in range(7):
                    g.setColorAt(i / 6, QColor.fromHsvF(((self.t * 0.3) + i / 6) % 1.0, 0.85, 1.0))
            else:
                g.setColorAt(0.0, hi)
                g.setColorAt(1.0, lo)
            p.setPen(QPen(g, 4 * u))
            p.setBrush(QColor("#1c1630"))
            p.drawRoundedRect(body, 3 * u, 3 * u)
            p.fillRect(QRectF(body.left() + body.width() * 0.15, body.top() + body.height() * 0.15,
                              body.width() * 0.7, body.height() * 0.35), QColor("#29f3ff" if pz.value == "neon" else
                                                                                   "#0a2a3a"))
        elif pz.kind == "mascot":
            if pz.value == "none":
                pixelfont.draw(p, "×", r.center().x(), r.center().y() - 10 * u, 3 * u, "#6b5a7a",
                               Qt.AlignmentFlag.AlignHCenter)
            else:
                kind = pz.value
                draw_sprite(p, kind, r.center(), min(r.width() / 16, r.height() / 16), int(self.t * 2) % 2)
        else:
            if pz.value == "popper":
                for i in range(14):
                    a = i * 0.45 + self.t
                    c = QColor(NEON[i % len(NEON)])
                    p.fillRect(QRectF(r.center().x() + math.cos(a) * r.width() * 0.3,
                                      r.center().y() + math.sin(a * 1.3) * r.height() * 0.3, 5 * u, 3 * u), c)
            else:
                rr = min(r.width(), r.height()) * 0.3
                g = QLinearGradient(r.center() - QPointF(rr, rr), r.center() + QPointF(rr, rr))
                g.setColorAt(0.0, QColor("white"))
                g.setColorAt(1.0, QColor("#6b6b8a"))
                p.setPen(QPen(QColor(0, 0, 0, 80), 1))
                p.setBrush(g)
                p.drawEllipse(r.center(), rr, rr)
        p.restore()

    def _prize_key(self, key: int) -> None:
        n = len(PRIZES)
        rects = self.prize_rects()
        cur = rects[self.prize_sel][0]
        kinds = ("theme", "trim", "mascot", "toy")
        rows = [[i for i, (pz, _r) in enumerate(rects) if pz.kind == k] for k in kinds]
        r_i = kinds.index(cur.kind)
        c_i = rows[r_i].index(self.prize_sel)
        if key in (Qt.Key.Key_Right, Qt.Key.Key_D, Qt.Key.Key_Tab):
            self.prize_sel = (self.prize_sel + 1) % n
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_A, Qt.Key.Key_Backtab):
            self.prize_sel = (self.prize_sel - 1) % n
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_S):
            row = rows[(r_i + 1) % len(rows)]
            self.prize_sel = row[min(c_i, len(row) - 1)]
        elif key in (Qt.Key.Key_Up, Qt.Key.Key_W):
            row = rows[(r_i - 1) % len(rows)]
            self.prize_sel = row[min(c_i, len(row) - 1)]
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.buy(PRIZES[self.prize_sel].prize_id)
            return
        elif key in (Qt.Key.Key_Escape, Qt.Key.Key_P):
            self.close_prizes()
            return
        else:
            return
        self.audio.play("move", 0.7)

    def buy(self, prize_id: str) -> str:
        """Buy (or use) a prize from the counter."""
        pz = next(p for p in PRIZES if p.prize_id == prize_id)
        before = self.tickets.owns(prize_id)
        result = self.tickets.buy(prize_id)
        if result == "short":
            need = pz.price - self.tickets.balance
            self.prize_msg = (f"NEED {need} MORE TICKETS", "#ff6b6b", self.t + 2.2)
            self.prize_shake = 1.0
            self.audio.play("deny")
        elif result == "bought":
            self.prize_msg = (f"SOLD! {pz.name} IS YOURS", "#7dff4a", self.t + 2.4)
            self.fx.confetti.burst(QPointF(self.width() / 2, self.height() * 0.5), 90, 760, self.unit)
            self.audio.play("prize")
            self.store.flush()
        elif result == "equipped":
            if pz.kind == "toy":
                self.prize_msg = ("PRESS C TO POP" if pz.value == "popper" else "PRESS B TO PARTY", "#29f3ff",
                                  self.t + 2.0)
            else:
                self.prize_msg = (f"{pz.name} IN USE", "#29f3ff", self.t + 1.6)
            self.audio.play("open")
            self.store.flush()
        if result in ("bought", "equipped") and pz.kind == "mascot" and pz.value != "none":
            self.mascot_hop = 1.0
            self.say("HELLO!" if not before else _rng.choice(MASCOT_QUIPS[pz.value]))
        self.update()
        return result

    # --- input ----------------------------------------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        key = event.key()
        self._touch()
        if not event.isAutoRepeat() and self._konami(key):
            return
        if self.prizes_open:
            self._prize_key(key)
            self.update()
            return
        n = len(self.games)
        if key in (Qt.Key.Key_Right, Qt.Key.Key_D, Qt.Key.Key_Tab):
            self.move(1)
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_A, Qt.Key.Key_Backtab):
            self.move(-1)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            if n:
                self.request_play()
        elif key == Qt.Key.Key_Delete:
            if n:
                self.reset.emit(self.selected)
        elif key == Qt.Key.Key_M:
            self.toggle_mute.emit()
        elif key == Qt.Key.Key_K:
            self.request_controls()
        elif key == Qt.Key.Key_P:
            self.open_prizes()
        elif key == Qt.Key.Key_C and self.tickets.owns("toy.popper"):
            self.fx.confetti.burst(QPointF(self.width() / 2, self.height() * 0.75), 80, 820, self.unit)
            self.audio.play("confetti")
        elif key == Qt.Key.Key_B and self.tickets.owns("toy.disco") and not self._konami_pending():
            self.party(bonus=False)
        elif key == Qt.Key.Key_Escape:
            self.close_arcade.emit()
        else:
            super().keyPressEvent(event)
            return
        self.update()

    def _konami_pending(self) -> bool:
        return tuple(self._keys[-8:]) == KONAMI[:8]

    def _konami(self, key: int) -> bool:
        """True when ``key`` completed the code (the key then does nothing else)."""
        self._keys.append(key)
        del self._keys[:-len(KONAMI)]
        if tuple(self._keys) == KONAMI:
            self._keys.clear()
            self.party(bonus=True)
            return True
        return False

    def wheelEvent(self, event) -> None:  # noqa: N802, ANN001
        self._touch()
        if self.prizes_open:
            return
        self._wheel += event.angleDelta().y()
        while abs(self._wheel) >= 120:
            step = -1 if self._wheel > 0 else 1
            self._wheel += 120 * step
            self.move(step)

    def pose_at(self, pos: QPointF) -> Pose | None:
        """The cabinet under ``pos`` (the nearest one)."""
        for pose in reversed(self._poses or self.poses()):
            if pose.opacity < 0.3:
                continue
            if pose.quad.containsPoint(pos, Qt.FillRule.OddEvenFill) or (
                    pose.side_quad is not None and pose.side_quad.containsPoint(pos, Qt.FillRule.OddEvenFill)):
                return pose
        return None

    def cabinet_at(self, pos: QPointF) -> int:
        """The index of the cabinet under ``pos``, -1 for none."""
        pose = self.pose_at(pos)
        return -1 if pose is None else pose.index

    def mouseMoveEvent(self, event) -> None:  # noqa: N802, ANN001
        self._touch()
        if self.prizes_open:
            hit = next((i for i, (_pz, r) in enumerate(self.prize_rects()) if r.contains(event.position())), -1)
            self.setCursor(Qt.CursorShape.PointingHandCursor if hit >= 0 else Qt.CursorShape.ArrowCursor)
            return
        hover = self.cabinet_at(event.position())
        lay = self.layout_rects()
        self._hover_controls = bool(self.games) and lay["controls"].contains(event.position())
        clickable = hover >= 0 or self._hover_controls or any(lay[k].contains(event.position())
                                                              for k in ("mute", "prizes", "counter"))
        if hover != self.hover:
            self.hover = hover
            self.update()
        self.setCursor(Qt.CursorShape.PointingHandCursor if clickable else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802, ANN001
        self._touch()
        pos = event.position()
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.prizes_open:
            hit = next((i for i, (_pz, r) in enumerate(self.prize_rects()) if r.contains(pos)), -1)
            if hit >= 0:
                self.prize_sel = hit
                self.buy(PRIZES[hit].prize_id)
            elif not self._prize_panel().contains(pos):
                self.close_prizes()
            return
        lay = self.layout_rects()
        if lay["mute"].contains(pos):
            self.toggle_mute.emit()
            return
        if lay["prizes"].contains(pos) or lay["counter"].contains(pos):
            self.open_prizes()
            return
        if lay["controls"].contains(pos) and self.games:
            self.request_controls()
            return
        hit = self.critters.hit(pos, self.unit)
        if hit is not None:
            critter, caught = hit
            if caught:
                self._award_origin = QPointF(critter.x, critter.y)
                self.tickets.critter(critter.value)
                self._award_origin = None
                self.fx.confetti.burst(pos, 24, 420, self.unit)
                self.audio.play("pop")
                self.store.flush()
            else:
                self.fx.text("OUCH!", QPointF(critter.x, critter.y - 30 * self.unit), "#ffffff", 2 * self.unit, 0.8)
                self.audio.play("hit")
            return
        if self._mascot_rect.contains(pos):
            self.mascot_hop = 1.0
            mascot = self.tickets.equipped("mascot")
            self.say(_rng.choice(MASCOT_QUIPS.get(mascot, ("HI!",))))
            self.audio.play("meow" if mascot == "cat" else "pop", 0.7)
            return
        if lay["board"].contains(pos):
            self.board_page = (self.board_page + 1) % len(self.board_pages())
            self._board_next = self.t + 10.0
            self.audio.play("move", 0.6)
            self.update()
            return
        if lay["info"].contains(pos) and self.games:
            self.request_play()
            return
        pose = self.pose_at(pos)
        if pose is None:
            return
        if pose.index == self.selected and abs(pose.k) < 0.5:
            self.request_play()
        else:
            self.select(pose.index, pose.k)


def _bright(color: QColor | str, lightness: float = 0.62) -> str:
    """A game's accent, light enough to read as text on the dark panels."""
    hue, sat, light, _alpha = QColor(color).getHslF()
    return QColor.fromHslF(max(hue, 0.0), sat, max(light, lightness)).name()


def _radial(centre: QPointF, radius: float, color: QColor) -> QRadialGradient:
    g = QRadialGradient(centre, radius)
    g.setColorAt(0.0, color)
    c = QColor(color)
    c.setAlpha(0)
    g.setColorAt(1.0, c)
    return g


__all__ = ["HomeScreen", "fmt_time"]
