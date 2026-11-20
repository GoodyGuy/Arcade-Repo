"""Ctrl + 6 + 7 + 9 held together opens the arcade. An application event filter that only looks at the key presses: it
never consumes an event and never raises (the program behaves exactly as without it)."""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import QEvent, QObject, Qt, QTimer

log = logging.getLogger(__name__)

# the three digits, by Qt key (top row and keypad) and by Windows virtual key (any layout: AZERTY digits need Shift)
_DIGITS = {Qt.Key.Key_6: 6, Qt.Key.Key_7: 7, Qt.Key.Key_9: 9}
_NATIVE = {0x36: 6, 0x37: 7, 0x39: 9, 0x66: 6, 0x67: 7, 0x69: 9}
TRIGGER = frozenset((6, 7, 9))
STALE_S = 2.0  # a digit "held" longer than this without a release seen (focus moved) no longer counts


class ArcadeHotkey(QObject):
    """Application event filter: opens (or raises) the arcade window when Ctrl and 6, 7 and 9 are down at once."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._held: dict[int, float] = {}
        self.window = None  # the arcade window, made the first time

    @staticmethod
    def _digit(event) -> int | None:  # noqa: ANN001 - QKeyEvent
        digit = _DIGITS.get(event.key())
        if digit is None:
            digit = _NATIVE.get(event.nativeVirtualKey())
        return digit

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        try:
            kind = event.type()
            if kind == QEvent.Type.KeyPress:
                digit = self._digit(event)
                if digit is not None:
                    now = time.monotonic()
                    self._held.setdefault(digit, now)
                    if event.modifiers() & Qt.KeyboardModifier.ControlModifier and self._complete(now):
                        self._held.clear()
                        QTimer.singleShot(0, self.open_arcade)
            elif kind == QEvent.Type.KeyRelease:
                digit = self._digit(event)
                if digit is not None and not event.isAutoRepeat():
                    self._held.pop(digit, None)
            elif kind == QEvent.Type.ApplicationDeactivate:
                self._held.clear()
        except Exception:  # noqa: BLE001 - the easter egg must never disturb the program
            log.debug("arcade hotkey", exc_info=True)
        return False

    def _complete(self, now: float) -> bool:
        return all(d in self._held and now - self._held[d] <= STALE_S for d in TRIGGER)

    def open_arcade(self) -> None:
        try:
            if self.window is None:
                from app.arcade.window import ArcadeWindow

                self.window = ArcadeWindow()
                self.window.destroyed.connect(self._forget)
            self.window.show_arcade()
        except Exception:  # noqa: BLE001
            log.warning("The arcade could not open", exc_info=True)
            self.window = None

    def _forget(self, *_args) -> None:  # noqa: ANN002
        self.window = None


def install_hotkey(app) -> ArcadeHotkey | None:  # noqa: ANN001 - QApplication
    """Watch the application's key presses for the arcade hotkey; None (and the program unchanged) on any failure."""
    try:
        hotkey = ArcadeHotkey(app)
        app.installEventFilter(hotkey)
        return hotkey
    except Exception:  # noqa: BLE001
        log.debug("arcade hotkey not installed", exc_info=True)
        return None
