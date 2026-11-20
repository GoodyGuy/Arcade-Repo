"""The player's controls, per game: mouse sensitivity, invert Y, swap the mouse buttons, keyboard turn speed, and the
keys rebound to other keys (``ArcadeGame.bindings`` lists what may be rebound). Kept in the arcade save
(``settings["controls"][game_id]``) and applied by ``ArcadeGame.apply_controls``; ``ControlsPanel`` edits them."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from app.arcade.base import font

SENSITIVITY = (0.1, 4.0)
TURN_SPEED = (0.25, 3.0)
STEP = 0.05
DEFAULTS = {"mouse_sensitivity": 1.0, "invert_y": False, "swap_buttons": False, "key_turn_speed": 1.0, "keys": {}}
_NAMES = {
    Qt.Key.Key_Shift: "Shift", Qt.Key.Key_Control: "Ctrl", Qt.Key.Key_Alt: "Alt", Qt.Key.Key_Meta: "Win",
    Qt.Key.Key_Space: "Space", Qt.Key.Key_Return: "Enter", Qt.Key.Key_Enter: "Enter", Qt.Key.Key_Tab: "Tab",
    Qt.Key.Key_Backspace: "Backspace", Qt.Key.Key_CapsLock: "Caps Lock", Qt.Key.Key_Up: "↑", Qt.Key.Key_Down: "↓",
    Qt.Key.Key_Left: "←", Qt.Key.Key_Right: "→",
}
_NEVER = {Qt.Key.Key_Escape, Qt.Key.Key_F11, Qt.Key.Key_unknown}  # Esc pauses, F11 is the window's full screen


def _clamp(value, lo: float, hi: float, default: float) -> float:  # noqa: ANN001
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else min(max(v, lo), hi)


def normalise(cfg, bindings) -> dict:  # noqa: ANN001
    """Saved controls made valid (missing or bad values -> the defaults; keys only for the game's actions)."""
    cfg = cfg if isinstance(cfg, dict) else {}
    actions = {action for action, _key in bindings}
    keys = cfg.get("keys") if isinstance(cfg.get("keys"), dict) else {}
    clean = {}
    for action, key in keys.items():
        if action in actions and isinstance(key, int | float) and not isinstance(key, bool):
            k = int(key)
            if k > 0 and k not in _NEVER:
                clean[action] = k
    return {
        "mouse_sensitivity": _clamp(cfg.get("mouse_sensitivity"), *SENSITIVITY, 1.0),
        "invert_y": bool(cfg.get("invert_y", False)),
        "swap_buttons": bool(cfg.get("swap_buttons", False)),
        "key_turn_speed": _clamp(cfg.get("key_turn_speed"), *TURN_SPEED, 1.0),
        "keys": clean,
    }


def key_maps(keys: dict, bindings) -> tuple[dict[int, int], set[int]]:  # noqa: ANN001
    """(key pressed -> the default key the game knows, the default keys moved away to another key)."""
    defaults = dict(bindings)
    remap: dict[int, int] = {}
    moved: set[int] = set()
    for action, key in keys.items():
        default = defaults.get(action)
        if default is None or key == default:
            continue
        remap[key] = default
        moved.add(default)
    return remap, moved - set(remap)


def key_name(key: int) -> str:
    if key in _NAMES:
        return _NAMES[key]
    text = QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)
    return text or f"Key {key}"


def rebind(cfg: dict, bindings, action: str, key: int) -> dict:  # noqa: ANN001
    """``action`` on ``key``; an action that had ``key`` takes the old key of ``action`` (a swap, never a clash)."""
    defaults = dict(bindings)
    current = {a: cfg["keys"].get(a, d) for a, d in defaults.items()}
    old = current[action]
    for other, k in current.items():
        if other != action and k == key:
            current[other] = old
    current[action] = key
    cfg["keys"] = {a: k for a, k in current.items() if k != defaults[a]}
    return cfg


class ControlsPanel(QWidget):
    """Over the arcade: the controls of one game. ``closed`` gives the new controls (normalised)."""

    closed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.game_cls = None
        self.cfg = normalise({}, ())
        self.selected = 0
        self.hover = -1
        self.waiting: str | None = None  # the action whose new key is awaited
        self.scroll = 0
        self._drag = -1
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.hide()

    # --- rows: ("slider", key, label, range) / ("toggle", key, label) / ("bind", action) / ("button", id, label) ------

    def rows(self) -> list[tuple]:
        rows: list[tuple] = [("slider", "mouse_sensitivity", "Mouse sensitivity", SENSITIVITY),
                             ("toggle", "invert_y", "Invert mouse Y"),
                             ("toggle", "swap_buttons", "Swap left / right mouse buttons")]
        if self.game_cls is not None and self.game_cls.uses_key_turn:
            rows.append(("slider", "key_turn_speed", "Keyboard turn speed", TURN_SPEED))
        for action, _key in self._bindings():
            rows.append(("bind", action))
        rows += [("button", "reset", "RESET TO DEFAULTS"), ("button", "done", "DONE")]
        return rows

    def _bindings(self) -> tuple:
        return tuple(self.game_cls.bindings) if self.game_cls is not None else ()

    def open(self, game_cls, cfg: dict) -> None:  # noqa: ANN001
        self.game_cls = game_cls
        self.cfg = normalise(cfg, self._bindings())
        self.selected = 0
        self.scroll = 0
        self.waiting = None
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.setFocus()

    def finish(self) -> None:
        self.waiting = None
        self.hide()
        self.closed.emit(normalise(self.cfg, self._bindings()))

    # --- layout ---------------------------------------------------------------------------------------------------

    def _panel(self) -> QRectF:
        w, h = self.width(), self.height()
        pw, ph = min(w * 0.72, 820), min(h * 0.9, 860)
        return QRectF((w - pw) / 2, (h - ph) / 2, pw, ph)

    def _row_h(self) -> float:
        return max(min(self.height() * 0.05, 40), 26)

    def _visible(self) -> int:
        panel = self._panel()
        return max(int((panel.height() - self._row_h() * 3.2) / self._row_h()), 3)

    def row_rects(self) -> dict[int, QRectF]:
        """Rectangles of the rows on screen (index -> rect); the list scrolls to keep the selection in view."""
        panel, rh = self._panel(), self._row_h()
        n, vis = len(self.rows()), self._visible()
        self.scroll = min(max(self.scroll, self.selected - vis + 1, 0), max(n - vis, 0))
        self.scroll = min(self.scroll, self.selected)
        top = panel.top() + rh * 2.4
        return {i: QRectF(panel.left() + rh * 0.6, top + (i - self.scroll) * rh, panel.width() - rh * 1.2, rh * 0.9)
                for i in range(self.scroll, min(n, self.scroll + vis))}

    def _value_rect(self, r: QRectF) -> QRectF:
        return QRectF(r.left() + r.width() * 0.55, r.top() + r.height() * 0.15, r.width() * 0.43, r.height() * 0.7)

    # --- drawing --------------------------------------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        g = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) * 0.7)
        g.setColorAt(0.0, QColor(0, 0, 0, 170))
        g.setColorAt(1.0, QColor(0, 0, 0, 230))
        p.fillRect(self.rect(), g)
        accent = QColor(self.game_cls.accent if self.game_cls is not None else "#ff4081")
        panel, rh = self._panel(), self._row_h()
        p.setPen(QPen(accent, 2))
        p.setBrush(QColor(14, 12, 26, 240))
        p.drawRoundedRect(panel, 16, 16)
        p.setPen(QColor("white"))
        p.setFont(font(rh * 0.8, True, "Impact"))
        p.drawText(QRectF(panel.left(), panel.top() + rh * 0.3, panel.width(), rh), Qt.AlignmentFlag.AlignCenter,
                   "CONTROLS")
        p.setFont(font(rh * 0.42))
        p.setPen(accent.lighter(140))
        title = self.game_cls.title if self.game_cls is not None else ""
        hint = "press the new key   (Esc cancels)" if self.waiting else "↑ ↓ choose   ← → adjust   Enter change   Esc done"
        p.drawText(QRectF(panel.left(), panel.top() + rh * 1.25, panel.width(), rh * 0.8),
                   Qt.AlignmentFlag.AlignCenter, f"{title}   ·   {hint}")
        rows = self.rows()
        for i, r in self.row_rects().items():
            self._paint_row(p, rows[i], r, i, accent)
        if self.scroll > 0 or self.scroll + self._visible() < len(rows):
            p.setPen(QColor(255, 255, 255, 120))
            p.setFont(font(rh * 0.4))
            p.drawText(QRectF(panel.left(), panel.bottom() - rh * 0.75, panel.width() - rh * 0.6, rh * 0.6),
                       Qt.AlignmentFlag.AlignRight, "more ↓" if self.scroll + self._visible() < len(rows) else "")
        p.end()

    def _paint_row(self, p: QPainter, row: tuple, r: QRectF, i: int, accent: QColor) -> None:
        on = i == self.selected or i == self.hover
        if row[0] == "button":
            p.setPen(QPen(accent if on else QColor(255, 255, 255, 90), 2))
            p.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 110) if on else QColor(30, 28, 44))
            b = QRectF(r.center().x() - r.width() * 0.22, r.top(), r.width() * 0.44, r.height())
            p.drawRoundedRect(b, b.height() / 2, b.height() / 2)
            p.setPen(QColor("white"))
            p.setFont(font(r.height() * 0.42, True))
            p.drawText(b, Qt.AlignmentFlag.AlignCenter, row[2])
            return
        if on:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 55))
            p.drawRoundedRect(r, 8, 8)
        label = row[1] if row[0] == "bind" else row[2]
        p.setPen(QColor("white"))
        p.setFont(font(r.height() * 0.42, row[0] != "bind"))
        p.drawText(r.adjusted(r.height() * 0.4, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   label)
        v = self._value_rect(r)
        if row[0] == "slider":
            lo, hi = row[3]
            value = self.cfg[row[1]]
            frac = (value - lo) / (hi - lo)
            track = QRectF(v.left(), v.center().y() - 3, v.width() * 0.72, 6)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 50))
            p.drawRoundedRect(track, 3, 3)
            p.setBrush(accent)
            p.drawRoundedRect(QRectF(track.left(), track.top(), track.width() * frac, track.height()), 3, 3)
            p.setBrush(QColor("white"))
            p.drawEllipse(QPointF(track.left() + track.width() * frac, track.center().y()), v.height() * 0.32,
                          v.height() * 0.32)
            p.setPen(QColor("white"))
            p.setFont(font(r.height() * 0.42, True))
            p.drawText(QRectF(track.right() + 8, v.top(), v.width() * 0.28 - 8, v.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, f"{value:.2f}×")
        elif row[0] == "toggle":
            on_value = self.cfg[row[1]]
            sw = QRectF(v.right() - v.height() * 1.9, v.top(), v.height() * 1.9, v.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(accent if on_value else QColor(255, 255, 255, 60))
            p.drawRoundedRect(sw, sw.height() / 2, sw.height() / 2)
            p.setBrush(QColor("white"))
            d = sw.height() * 0.8
            x = sw.right() - d - sw.height() * 0.1 if on_value else sw.left() + sw.height() * 0.1
            p.drawEllipse(QRectF(x, sw.top() + sw.height() * 0.1, d, d))
        else:
            default = dict(self._bindings())[row[1]]
            key = self.cfg["keys"].get(row[1], default)
            waiting = self.waiting == row[1]
            cap = QRectF(v.right() - v.width() * 0.6, v.top(), v.width() * 0.6, v.height())
            p.setPen(QPen(accent if waiting else QColor(255, 255, 255, 120), 2))
            p.setBrush(QColor(40, 38, 60) if not waiting else QColor(accent.red(), accent.green(), accent.blue(), 90))
            p.drawRoundedRect(cap, 6, 6)
            p.setPen(QColor("white") if key == default else accent.lighter(160))
            p.setFont(font(r.height() * 0.42, True))
            p.drawText(cap, Qt.AlignmentFlag.AlignCenter, "press a key…" if waiting else key_name(key))

    # --- input ----------------------------------------------------------------------------------------------------

    def _adjust(self, row: tuple, direction: int) -> None:
        if row[0] == "slider":
            lo, hi = row[3]
            self.cfg[row[1]] = round(min(max(self.cfg[row[1]] + STEP * direction, lo), hi), 2)
        elif row[0] == "toggle":
            self.cfg[row[1]] = not self.cfg[row[1]]

    def _activate(self, i: int) -> None:
        row = self.rows()[i]
        if row[0] == "toggle":
            self._adjust(row, 1)
        elif row[0] == "bind":
            self.waiting = row[1]
        elif row[0] == "button":
            if row[1] == "reset":
                self.cfg = normalise({}, self._bindings())
            else:
                self.finish()
                return
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802, ANN001
        key = event.key()
        if self.waiting is not None:
            if key != Qt.Key.Key_Escape and key not in _NEVER:
                self.cfg = rebind(self.cfg, self._bindings(), self.waiting, key)
            self.waiting = None
            self.update()
            return
        rows = self.rows()
        if key in (Qt.Key.Key_Down, Qt.Key.Key_S, Qt.Key.Key_Tab):
            self.selected = (self.selected + 1) % len(rows)
        elif key in (Qt.Key.Key_Up, Qt.Key.Key_W, Qt.Key.Key_Backtab):
            self.selected = (self.selected - 1) % len(rows)
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_A):
            self._adjust(rows[self.selected], -1)
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_D):
            self._adjust(rows[self.selected], 1)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._activate(self.selected)
            return
        elif key == Qt.Key.Key_Escape:
            self.finish()
            return
        self.update()

    def _row_at(self, pos: QPointF) -> int:
        return next((i for i, r in self.row_rects().items() if r.contains(pos)), -1)

    def _slide(self, i: int, pos: QPointF) -> None:
        row = self.rows()[i]
        r = self.row_rects().get(i)
        if row[0] != "slider" or r is None:
            return
        v = self._value_rect(r)
        lo, hi = row[3]
        frac = min(max((pos.x() - v.left()) / (v.width() * 0.72), 0.0), 1.0)
        self.cfg[row[1]] = round((lo + frac * (hi - lo)) / STEP) * STEP
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802, ANN001
        if self.waiting is not None:
            self.waiting = None
            self.update()
            return
        i = self._row_at(event.position())
        if i < 0:
            if not self._panel().contains(event.position()):
                self.finish()
            return
        self.selected = i
        if self.rows()[i][0] == "slider":
            self._drag = i
            self._slide(i, event.position())
        else:
            self._activate(i)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802, ANN001
        if self._drag >= 0:
            self._slide(self._drag, event.position())
            return
        hover = self._row_at(event.position())
        if hover != self.hover:
            self.hover = hover
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802, ANN001
        self._drag = -1

    def wheelEvent(self, event) -> None:  # noqa: N802, ANN001
        steps = event.angleDelta().y() // 120
        if steps:
            self.selected = min(max(self.selected - int(steps), 0), len(self.rows()) - 1)
            self.update()
