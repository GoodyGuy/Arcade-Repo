"""The arcade window: a home screen with the game selector (app/arcade/home.py), the game being played, and its pause
menu. Progress is saved when a game is left, every 30 s while playing, and when the window closes. Closing the window
only hides it (and stops every sound)."""

from __future__ import annotations

import logging
import math
import time

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QRadialGradient, QShortcut
from PySide6.QtWidgets import QMessageBox, QStackedLayout, QWidget

from app.arcade import audio
from app.arcade.base import ArcadeGame, font
from app.arcade.controls import ControlsPanel
from app.arcade.games import load_games
from app.arcade.home import HomeScreen, fmt_time
from app.arcade.home_fx import LaunchIntro
from app.arcade.saves import SaveStore

log = logging.getLogger(__name__)

AUTOSAVE_MS = 30_000
TITLE = "BREAK ROOM ARCADE"
_fmt_time = fmt_time  # the old name

__all__ = ["ArcadeWindow", "GamePage", "HomeScreen", "PauseMenu"]


class PauseMenu(QWidget):
    """Drawn over the paused game: Resume, Save, Quit to the arcade, and the game's controls."""

    resume = Signal()
    controls = Signal()
    save = Signal()
    quit_to_home = Signal()

    ITEMS = ("RESUME", "CONTROLS", "SAVE GAME", "QUIT TO ARCADE")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.selected = 0
        self.hover = -1
        self.game_cls = None
        self.message = ""
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.hide()

    def open(self, game_cls) -> None:  # noqa: ANN001
        self.game_cls = game_cls
        self.selected = 0
        self.message = ""
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.setFocus()

    def item_rects(self) -> list[QRectF]:
        w, h = self.width(), self.height()
        bw, bh = min(w * 0.3, 380), max(h * 0.07, 40)
        y0 = h * 0.36
        return [QRectF((w - bw) / 2, y0 + i * bh * 1.3, bw, bh) for i in range(len(self.ITEMS))]

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        g = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) * 0.7)
        g.setColorAt(0.0, QColor(0, 0, 0, 150))
        g.setColorAt(1.0, QColor(0, 0, 0, 220))
        p.fillRect(self.rect(), g)
        accent = QColor(self.game_cls.accent if self.game_cls else "#ff4081")
        p.setPen(QColor("white"))
        p.setFont(font(h * 0.07, True, "Impact"))
        p.drawText(QRectF(0, h * 0.12, w, h * 0.1), Qt.AlignmentFlag.AlignCenter, "PAUSED")
        if self.game_cls is not None:
            p.setFont(font(h * 0.028))
            p.setPen(accent.lighter(140))
            p.drawText(QRectF(0, h * 0.22, w, h * 0.05), Qt.AlignmentFlag.AlignCenter, self.game_cls.title)
        for i, (label, r) in enumerate(zip(self.ITEMS, self.item_rects(), strict=True)):
            on = i == self.selected or i == self.hover
            p.setPen(QPen(accent if on else QColor(255, 255, 255, 90), 2))
            p.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 90) if on else QColor(20, 20, 30, 200))
            p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
            p.setPen(QColor("white"))
            p.setFont(font(r.height() * 0.4, True))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, label)
        if self.message:
            p.setPen(QColor("#7CFC00"))
            p.setFont(font(h * 0.024, True))
            last = self.item_rects()[-1]
            p.drawText(QRectF(0, last.bottom() + h * 0.02, w, h * 0.04), Qt.AlignmentFlag.AlignCenter, self.message)
        if self.game_cls is not None and self.game_cls.controls:
            rows = self.game_cls.controls
            size = max(min(h * 0.022, 16), 11)
            top = self.item_rects()[-1].bottom() + h * 0.08
            p.setFont(font(size))
            col_w = min(w * 0.22, 300)
            per_col = math.ceil(len(rows) / 2)
            for k, (keys, action) in enumerate(rows):
                c, r = divmod(k, per_col)
                x = w / 2 - col_w - 20 + c * (col_w + 40)
                y = top + r * size * 1.6
                p.setPen(accent.lighter(150))
                p.drawText(QRectF(x, y, col_w * 0.42, size * 1.5), Qt.AlignmentFlag.AlignRight, keys)
                p.setPen(QColor(255, 255, 255, 200))
                p.drawText(QRectF(x + col_w * 0.46, y, col_w * 0.6, size * 1.5), Qt.AlignmentFlag.AlignLeft, action)
        p.end()

    def _activate(self, i: int) -> None:
        (self.resume, self.controls, self.save, self.quit_to_home)[i].emit()

    def keyPressEvent(self, event) -> None:  # noqa: N802, ANN001
        key = event.key()
        if key in (Qt.Key.Key_Down, Qt.Key.Key_S, Qt.Key.Key_Tab):
            self.selected = (self.selected + 1) % len(self.ITEMS)
        elif key in (Qt.Key.Key_Up, Qt.Key.Key_W, Qt.Key.Key_Backtab):
            self.selected = (self.selected - 1) % len(self.ITEMS)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._activate(self.selected)
            return
        elif key == Qt.Key.Key_Escape:
            self.resume.emit()
            return
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802, ANN001
        hover = next((i for i, r in enumerate(self.item_rects()) if r.contains(event.position())), -1)
        if hover != self.hover:
            self.hover = hover
            self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802, ANN001
        i = next((i for i, r in enumerate(self.item_rects()) if r.contains(event.position())), -1)
        if i >= 0:
            self._activate(i)


class GamePage(QWidget):
    """Holds the game being played, the insert-coin sequence over it as it starts, and the pause menu."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.game: ArcadeGame | None = None
        self.intro = LaunchIntro(self)
        self.menu = PauseMenu(self)

    def set_game(self, game: ArcadeGame | None) -> None:
        self.intro.finish()
        if self.game is not None:
            self.game.setParent(None)
            self.game.deleteLater()
        self.game = game
        if game is not None:
            game.setParent(self)
            game.setGeometry(self.rect())
            game.show()
            self.intro.raise_()
            self.menu.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().resizeEvent(event)
        if self.game is not None:
            self.game.setGeometry(self.rect())
        self.menu.setGeometry(self.rect())
        self.intro.setGeometry(self.rect())


class ArcadeWindow(QWidget):
    """The arcade (see the module docstring)."""

    def __init__(self, store: SaveStore | None = None) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("Break Room Arcade")
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)  # never keeps the program running
        self.setStyleSheet("background: #07021a; color: white;")
        self.resize(1280, 780)
        self.setMinimumSize(900, 560)
        self.store = store or SaveStore()
        audio.set_muted(self.store.settings.get("muted", False))
        audio.set_volume(self.store.settings.get("volume", 0.7))
        self.games = load_games()
        self.home = HomeScreen(self.games, self.store, self)
        self.page = GamePage(self)
        self.stack = QStackedLayout(self)
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.page)
        self.game: ArcadeGame | None = None
        self._played_since = 0.0
        self.home.play.connect(self._play_from_home)
        self.home.reset.connect(self.reset_progress)
        self.home.close_arcade.connect(self.close)
        self.home.toggle_mute.connect(self.toggle_mute)
        self.page.menu.resume.connect(self.resume_game)
        self.page.menu.controls.connect(lambda: self.open_controls("pause"))
        self.page.menu.save.connect(self._save_from_menu)
        self.page.menu.quit_to_home.connect(self.quit_to_home)
        self.controls_panel = ControlsPanel(self)
        self.controls_panel.closed.connect(self._controls_closed)
        self._controls_cls = None
        self._controls_from = ""
        if hasattr(self.home, "controls"):  # the home screen's Controls button for the chosen game
            self.home.controls.connect(self._controls_from_home)
        self.page.intro.on_coin = lambda: self.home.audio.play("ready")
        self._autosave = QTimer(self)
        self._autosave.setInterval(AUTOSAVE_MS)
        self._autosave.timeout.connect(self.save_progress)
        QShortcut(QKeySequence("F11"), self, self.toggle_full_screen)

    def show_arcade(self) -> None:
        self.store.data["stats"]["opened"] = self.store.data["stats"].get("opened", 0) + 1
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        self.home.on_opened()
        if self.game is None:
            self.home.setFocus()
            self.home.update_music()

    # --- games --------------------------------------------------------------------------------------------------

    def _play_from_home(self, index: int) -> None:
        """A coin inserted on the home screen: the game starts at once, the launch sequence plays over it."""
        self.launch(index)
        fx, self.home.launch_fx = self.home.launch_fx, None
        if self.game is not None and fx is not None:
            snapshot, screen, slot, accent = fx
            self.page.intro.start(snapshot, screen, slot, accent, self.game)

    def launch(self, index: int) -> None:
        if not 0 <= index < len(self.games):
            return
        cls = self.games[index]
        try:
            game = cls()
            game.load_state(self.store.game(cls.game_id))
            game.apply_controls(self.controls_for(cls))
        except Exception:  # noqa: BLE001
            log.warning("Arcade game %s did not start", cls.game_id, exc_info=True)
            QMessageBox.warning(self, "Arcade", f"{cls.title} could not start.")
            return
        self.game = game
        game.pause_requested.connect(self.pause_game)
        game.exit_requested.connect(self.quit_to_home)
        game.progress_changed.connect(self.save_progress)
        game.controls_requested.connect(lambda: self.open_controls("game"))
        self.page.set_game(game)
        self.stack.setCurrentWidget(self.page)
        self.page.menu.hide()
        self._played_since = time.monotonic()
        self._autosave.start()
        try:
            self.home.game_launched(cls)
        except Exception:  # noqa: BLE001 - tickets never stop a game
            log.debug("arcade tickets", exc_info=True)
        game.start()
        game.setFocus()

    def pause_game(self) -> None:
        if self.game is None or self.page.menu.isVisible():
            return
        self.page.intro.finish()
        self.game.pause()
        self._count_play_time()
        self.save_progress()
        self.page.menu.open(type(self.game))

    def resume_game(self) -> None:
        if self.game is None:
            return
        self.page.menu.hide()
        self._played_since = time.monotonic()
        self.game.resume()
        self.game.setFocus()

    def _save_from_menu(self) -> None:
        ok = self.save_progress()
        self.page.menu.message = "Game saved" if ok else "Could not save (see the log)"
        self.page.menu.update()

    # --- controls -----------------------------------------------------------------------------------------------

    def controls_for(self, cls) -> dict:  # noqa: ANN001
        """The player's saved controls of a game ({}: the defaults)."""
        return dict((self.store.settings.get("controls") or {}).get(cls.game_id) or {})

    def open_controls(self, origin: str, cls=None) -> None:  # noqa: ANN001
        """The Controls panel for ``cls`` (the game being played by default), from the pause menu ("pause"), the
        game's own menu ("game") or the home screen ("home")."""
        cls = cls or (type(self.game) if self.game is not None else None)
        if cls is None:
            return
        if origin == "game" and self.game is not None and not self.game.paused:
            self.game.pause()
            self._count_play_time()
        self._controls_cls, self._controls_from = cls, origin
        self.controls_panel.open(cls, self.controls_for(cls))

    def _controls_from_home(self, index: int) -> None:
        if 0 <= index < len(self.games):
            self.open_controls("home", self.games[index])

    def _controls_closed(self, cfg: dict) -> None:
        cls, origin = self._controls_cls, self._controls_from
        self._controls_cls = None
        if cls is not None:
            self.store.settings.setdefault("controls", {})[cls.game_id] = cfg
            self.store.flush()
            if self.game is not None and type(self.game) is cls:
                self.game.apply_controls(cfg)
        if origin == "pause" and self.page.menu.isVisible():
            self.page.menu.setFocus()
        elif origin == "game" and self.game is not None:
            self.resume_game()
        else:
            self.home.setFocus()

    def quit_to_home(self) -> None:
        if self.game is None:
            return
        self._count_play_time()
        self.save_progress()
        self.game.stop()
        self._autosave.stop()
        self.page.menu.hide()
        self.page.set_game(None)
        self.game = None
        try:
            self.home.on_returned()
        except Exception:  # noqa: BLE001
            log.debug("arcade tickets", exc_info=True)
        self.stack.setCurrentWidget(self.home)
        self.home.setFocus()

    def _count_play_time(self) -> None:
        if self.game is not None and not self.game.paused and self._played_since:
            self.store.add_play_time(self.game.game_id, time.monotonic() - self._played_since)
        self._played_since = time.monotonic()

    def save_progress(self) -> bool:
        if self.game is not None:
            try:
                self.store.put(self.game.game_id, self.game.save_state())
            except Exception:  # noqa: BLE001
                log.warning("Arcade game %s not saved", self.game.game_id, exc_info=True)
                return False
            if not self.game.paused:
                self._count_play_time()
        return self.store.flush()

    def reset_progress(self, index: int) -> None:
        if not 0 <= index < len(self.games):
            return
        cls = self.games[index]
        if not self.store.game(cls.game_id):
            return
        answer = QMessageBox.question(self, "Reset progress", f"Erase all saved progress of {cls.title}?")
        if answer == QMessageBox.StandardButton.Yes:
            self.store.reset(cls.game_id)
            self.home.tickets.game_reset(cls.game_id)
            self.store.flush()
            self.home.data_changed()

    def toggle_mute(self) -> None:
        muted = not audio.is_muted()
        audio.set_muted(muted)
        self.store.settings["muted"] = muted
        self.store.flush()
        if muted:
            self.home.stop_sounds()
            if self.game is not None:
                self.game.sfx.stop_all()
        self.home.update_music()
        self.home.update()

    def toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # --- window -------------------------------------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        super().resizeEvent(event)
        self.controls_panel.setGeometry(self.rect())

    def changeEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        super().changeEvent(event)
        if event.type() in (QEvent.Type.ActivationChange, QEvent.Type.WindowStateChange):
            if not self.isActiveWindow() and self.game is not None and not self.game.paused:
                self.pause_game()  # alt-tab: the game waits (and the mouse is let go)
            self.home.update_music()  # the home screen's music only plays in the active window

    def hideEvent(self, event) -> None:  # noqa: N802, ANN001
        self.home.stop_sounds()
        self.page.intro.finish()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001
        if self.controls_panel.isVisible():  # keep the controls being edited, without resuming the game
            self._controls_from = "home"
            self.controls_panel.finish()
        self.quit_to_home()
        self.home.stop_sounds()
        self.store.flush()
        event.accept()


