"""The base of every arcade game: a QWidget with a 60 fps game loop, held-key and mouse-look input, a sound bank, and
the save / load hooks the arcade window calls.

A game subclasses ``ArcadeGame`` and overrides:

* ``tick(dt)`` - advance the game by ``dt`` seconds (the window repaints after each tick);
* ``paintEvent`` - draw it (``blit`` puts a numpy RGB frame on screen, pixelated);
* ``save_state()`` / ``load_state(data)`` - its progress as plain JSON data;
* the input hooks ``key_pressed`` / ``key_released`` / ``mouse_pressed`` / ``mouse_released`` / ``mouse_moved`` /
  ``wheel`` (``self.keys`` holds the keys down; ``take_mouse_delta()`` the mouse-look motion while captured);
* ``handle_escape()`` - True when Esc closed something of the game's own (an inventory); otherwise the arcade pauses;
* the class attributes and ``paint_thumbnail`` / ``summary`` for its card on the arcade's home screen.
"""

from __future__ import annotations

import functools
import logging
import os
import time

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget

from app.arcade.audio import SoundBank

FRAME_MS = 16
MAX_DT = 0.05  # s: a slow frame (or a debugger stop) never moves the game further than this
# a game's own methods that Qt or the loop calls: an error in one stops the game (logged) and the arcade goes home;
# without this the program's error dialog would come up for every frame. TDS_ARCADE_STRICT=1 (the tests) re-raises.
_GUARDED = (
    "tick", "key_pressed", "key_released", "mouse_pressed", "mouse_released", "mouse_moved", "wheel", "on_pause",
    "on_resume", "handle_escape", "paintEvent", "resizeEvent", "showEvent", "hideEvent", "keyPressEvent",
    "keyReleaseEvent", "mousePressEvent", "mouseReleaseEvent", "mouseDoubleClickEvent", "mouseMoveEvent",
    "wheelEvent", "focusInEvent", "focusOutEvent", "enterEvent", "leaveEvent", "timerEvent",
)

log = logging.getLogger(__name__)


def _guard(fn):  # noqa: ANN001, ANN202
    @functools.wraps(fn)
    def guarded(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if getattr(self, "_failed", False):
            return None
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            if os.environ.get("TDS_ARCADE_STRICT"):
                raise
            self._fail(fn.__name__)
            return None

    guarded.arcade_guarded = True
    return guarded


def blit(painter: QPainter, frame: np.ndarray, target: QRect | QRectF, smooth: bool = False) -> None:
    """Draw a (h, w, 3) RGB or (h, w, 4) RGBA uint8 array scaled into ``target`` (nearest pixel unless ``smooth``)."""
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    h, w = frame.shape[:2]
    fmt = QImage.Format.Format_RGB888 if frame.shape[2] == 3 else QImage.Format.Format_RGBA8888
    image = QImage(frame.data, w, h, frame.strides[0], fmt)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
    painter.drawImage(QRectF(target), image)
    painter.restore()


def letterbox(area: QRect, aspect: float) -> QRect:
    """The largest rectangle of the given width / height ratio centred in ``area``."""
    w, h = area.width(), area.height()
    if w / max(h, 1) > aspect:
        nw = int(h * aspect)
        return QRect(area.x() + (w - nw) // 2, area.y(), nw, h)
    nh = int(w / aspect)
    return QRect(area.x(), area.y() + (h - nh) // 2, w, nh)


def font(size: float, bold: bool = False, family: str = "Segoe UI") -> QFont:
    f = QFont(family)
    f.setPixelSize(max(int(size), 1))
    f.setBold(bold)
    return f


class ArcadeGame(QWidget):
    """One game of the arcade (see the module docstring)."""

    game_id = "game"
    title = "Game"
    subtitle = ""
    description = ""
    controls: tuple[tuple[str, str], ...] = ()  # (keys, action) shown on the home screen and the pause menu
    accent = "#ff4081"
    uses_mouse_look = False  # the window then captures the mouse when the game starts / resumes
    # the arcade's Controls panel: (action, default Qt key) rows the player may rebind; uses_key_turn offers a keyboard
    # turn speed (read ``self.key_turn_speed``) for games that turn the view with keys
    bindings: tuple[tuple[str, int], ...] = ()
    uses_key_turn = False

    pause_requested = Signal()  # Esc (not taken by the game): the arcade shows its pause menu
    exit_requested = Signal()  # the game wants to go back to the arcade's home screen
    progress_changed = Signal()  # something worth saving now happened (a new best score)
    controls_requested = Signal()  # the game's own menu asks for the arcade's Controls panel

    def __init_subclass__(cls, **kwargs) -> None:  # noqa: ANN003
        super().__init_subclass__(**kwargs)
        for name in _GUARDED:
            fn = cls.__dict__.get(name)
            if callable(fn) and not getattr(fn, "arcade_guarded", False):
                setattr(cls, name, _guard(fn))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._failed = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.keys: set[int] = set()
        self.sfx = SoundBank()
        self.time = 0.0  # s of game time (paused time not counted)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._step)
        self._last = 0.0
        self._paused = True
        self._captured = False
        self._mouse_dx = 0.0
        self._mouse_dy = 0.0
        # the player's controls (the arcade's Controls panel; apply_controls)
        self.mouse_sensitivity = 1.0
        self.invert_y = False
        self.swap_buttons = False
        self.key_turn_speed = 1.0
        self._remap: dict[int, int] = {}  # key pressed -> the default key the game knows
        self._unbound: set[int] = set()  # default keys moved to another key

    def apply_controls(self, cfg: dict) -> None:
        """The player's controls (``controls.normalise`` data): sensitivities, invert / swap, and rebound keys."""
        from app.arcade.controls import key_maps, normalise

        cfg = normalise(cfg, self.bindings)
        self.mouse_sensitivity = cfg["mouse_sensitivity"]
        self.invert_y = cfg["invert_y"]
        self.swap_buttons = cfg["swap_buttons"]
        self.key_turn_speed = cfg["key_turn_speed"]
        self._remap, self._unbound = key_maps(cfg["keys"], self.bindings)
        self.keys.clear()

    def map_key(self, key: int) -> int | None:
        """The key the game sees for a key pressed (None: a default key the player moved to another key)."""
        if key in self._remap:
            return self._remap[key]
        return None if key in self._unbound else key

    def _map_button(self, button: Qt.MouseButton) -> Qt.MouseButton:
        if self.swap_buttons:
            if button == Qt.MouseButton.LeftButton:
                return Qt.MouseButton.RightButton
            if button == Qt.MouseButton.RightButton:
                return Qt.MouseButton.LeftButton
        return button

    # --- for the game to override -------------------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        """Advance the game by ``dt`` seconds."""

    def save_state(self) -> dict:
        """The progress to keep (JSON data)."""
        return {}

    def load_state(self, data: dict) -> None:
        """Restore ``save_state()``'s data ({} for a first game). Called once, before ``start``."""

    def handle_escape(self) -> bool:
        """True when Esc was used by the game itself (closing its inventory); False lets the arcade pause."""
        return False

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001 - QKeyEvent
        """A key went down (auto-repeats not included)."""

    def key_released(self, key: int, event) -> None:  # noqa: ANN001
        """A key came up."""

    def mouse_pressed(self, button: Qt.MouseButton, pos: QPointF) -> None:
        """A mouse button went down."""

    def mouse_released(self, button: Qt.MouseButton, pos: QPointF) -> None:
        """A mouse button came up."""

    def mouse_moved(self, pos: QPointF) -> None:
        """The mouse moved (not captured)."""

    def wheel(self, steps: int) -> None:
        """Mouse wheel: +1 per notch up, -1 per notch down."""

    def on_pause(self) -> None:
        """The game was paused (stop looping sounds)."""

    def on_resume(self) -> None:
        """The game continues."""

    @classmethod
    def summary(cls, data: dict) -> str:
        """One line for the home screen from the saved progress (``data`` is {} for a game never played)."""
        return "New game" if not data else "Continue"

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        """The game's card art on the home screen (``t``: seconds, for an animation)."""
        g = QLinearGradient(rect.topLeft(), rect.bottomRight())
        g.setColorAt(0.0, QColor(cls.accent))
        g.setColorAt(1.0, QColor(cls.accent).darker(300))
        p.fillRect(rect, g)
        p.setPen(QColor("white"))
        p.setFont(font(rect.height() * 0.14, True))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, cls.title)

    # --- lifecycle (the arcade window calls these) --------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self._paused

    def start(self) -> None:
        self.resume()

    def pause(self) -> None:
        if self._paused:
            return
        self._paused = True
        self._timer.stop()
        self.keys.clear()
        self.set_mouse_captured(False)
        self.on_pause()
        self.update()

    def resume(self) -> None:
        if not self._paused:
            return
        self._paused = False
        self._last = time.perf_counter()
        self._timer.start()
        self.setFocus()
        if self.uses_mouse_look:
            self.set_mouse_captured(True)
        self.on_resume()

    def stop(self) -> None:
        """The game is closing: stop the loop and every sound."""
        self.pause()
        self.sfx.stop_all()

    @property
    def failed(self) -> bool:
        """The game hit an error and stopped."""
        return self._failed

    def _fail(self, where: str) -> None:
        log.exception("Arcade game %s stopped after an error in %s", self.game_id, where)
        self._failed = True
        self._timer.stop()
        self._paused = True
        try:
            self.set_mouse_captured(False)
            self.sfx.stop_all()
        except Exception:  # noqa: BLE001
            pass
        QTimer.singleShot(0, self.exit_requested.emit)

    def _step(self) -> None:
        now = time.perf_counter()
        dt = min(now - self._last, MAX_DT)
        self._last = now
        if dt <= 0.0 or self._failed:
            return
        self.time += dt
        self.tick(dt)
        if not self._failed:
            self.update()

    def step(self, dt: float = FRAME_MS / 1000) -> None:
        """Advance one frame of ``dt`` seconds by hand (tests)."""
        self.time += dt
        self.tick(dt)

    # --- input ------------------------------------------------------------------------------------------------

    def set_mouse_captured(self, captured: bool) -> None:
        """Mouse look: the cursor hidden and kept in the middle of the game; ``take_mouse_delta`` gives the motion."""
        if captured == self._captured:
            return
        self._captured = captured
        self._mouse_dx = self._mouse_dy = 0.0
        if captured:
            self.setCursor(Qt.CursorShape.BlankCursor)
            self._recentre()
        else:
            self.unsetCursor()

    @property
    def mouse_captured(self) -> bool:
        return self._captured

    def _recentre(self) -> None:
        if self.isVisible():
            QCursor.setPos(self.mapToGlobal(self.rect().center()))

    def take_mouse_delta(self) -> tuple[float, float]:
        """Mouse-look motion (pixels, times the player's sensitivity; y flipped when inverted) since the last call."""
        dx, dy = self._mouse_dx, self._mouse_dy
        self._mouse_dx = self._mouse_dy = 0.0
        return dx * self.mouse_sensitivity, dy * self.mouse_sensitivity * (-1.0 if self.invert_y else 1.0)

    def keyPressEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if not event.isAutoRepeat() and not self.handle_escape():
                self.pause_requested.emit()
            return
        if event.isAutoRepeat():
            return
        key = self.map_key(key)
        if key is None:
            return
        self.keys.add(key)
        if not self._paused:
            self.key_pressed(key, event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802, ANN001
        if event.isAutoRepeat():
            return
        key = self.map_key(event.key())
        if key is None:
            return
        self.keys.discard(key)
        if not self._paused:
            self.key_released(key, event)

    def mousePressEvent(self, event) -> None:  # noqa: N802, ANN001
        self.setFocus()
        if not self._paused:
            if self.uses_mouse_look and not self._captured:
                self.set_mouse_captured(True)
            self.mouse_pressed(self._map_button(event.button()), event.position())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802, ANN001
        if not self._paused:
            self.mouse_released(self._map_button(event.button()), event.position())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802, ANN001
        if self._captured and not self._paused:
            if self.isActiveWindow():
                centre = QPointF(self.rect().center())
                d = event.position() - centre
                if abs(d.x()) + abs(d.y()) > 0.5:
                    self._mouse_dx += d.x()
                    self._mouse_dy += d.y()
                    self._recentre()
            return
        if not self._paused:
            self.mouse_moved(event.position())

    def wheelEvent(self, event) -> None:  # noqa: N802, ANN001
        if not self._paused:
            steps = event.angleDelta().y() // 120
            if steps:
                self.wheel(int(steps))

    def focusOutEvent(self, event) -> None:  # noqa: N802, ANN001
        self.keys.clear()
        super().focusOutEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().resizeEvent(event)
        if self._captured:
            self._recentre()

    # --- drawing helpers ----------------------------------------------------------------------------------------

    def draw_center_text(self, p: QPainter, text: str, y: float, size: float, color: str = "white",
                         bold: bool = True, shadow: bool = True) -> None:
        p.setFont(font(size, bold))
        r = QRectF(0, y - size, self.width(), size * 2)
        if shadow:
            p.setPen(QColor(0, 0, 0, 180))
            p.drawText(r.translated(2, 2), Qt.AlignmentFlag.AlignCenter, text)
        p.setPen(QColor(color))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
