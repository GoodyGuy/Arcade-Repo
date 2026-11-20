"""Ashworth Hall: the cabinet around ``ashworth_hall.html`` (one self-contained page: its own 3D engine, art and
sound, nothing loaded from the internet).

The page runs in Qt's own web view (Qt WebEngine, part of PySide6) laid over this widget. Esc opens the arcade's
pause menu; while paused the view is hidden, so the page stops drawing and goes quiet. Where this PySide6 has no
web view, or the view cannot start, the cabinet offers to open the page in the browser instead.

The page keeps no progress of its own, so the cabinet remembers the best share of the lawn mown in one sitting."""

from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QGuiApplication, QKeySequence, QLinearGradient, QPainter, \
    QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import QApplication

from app.arcade import audio
from app.arcade.base import ArcadeGame, font
from app.arcade.saves import arcade_dir

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("ashworth_hall.html")
GOLD = QColor("#e9c98a")
SERIF = "Times New Roman"
_PROGRESS_JS = "(function(){var e=document.getElementById('pct');return e?e.textContent:''})()"
_UNLOCK_JS = "document.exitPointerLock&&document.exitPointerLock()"

_web: dict = {}  # Qt's web classes and the one profile, made the first time the game starts


def web_classes() -> dict | None:
    """Qt's web view classes ({} once loaded), None when this PySide6 has no web view."""
    if "failed" in _web:
        return None
    if _web:
        return _web
    try:
        from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except Exception:  # noqa: BLE001 - no web view in this install: the browser plays the page
        log.info("Ashworth Hall: no Qt web view here, the page opens in the browser")
        _web["failed"] = True
        return None

    class Page(QWebEnginePage):
        """The game's page: it stays on its own file and opens nothing else."""

        def acceptNavigationRequest(self, url, kind, is_main_frame) -> bool:  # noqa: N802, ANN001 - Qt API
            return url.isLocalFile() and Path(url.toLocalFile()) == PAGE

        def createWindow(self, kind):  # noqa: N802, ANN001, ANN201 - Qt API: no pop-ups
            return None

        def javaScriptConsoleMessage(self, level, message, line, source) -> None:  # noqa: N802, ANN001 - Qt API
            log.debug("Ashworth Hall page: %s (line %s)", message[:300], line)

    _web.update(Page=Page, Profile=QWebEngineProfile, Settings=QWebEngineSettings, View=QWebEngineView)
    return _web


def profile():  # noqa: ANN201 - the QWebEngineProfile (kept for the life of the program: pages must go before it)
    if "profile" not in _web:
        store = arcade_dir() / "ashworth_hall"
        prof = _web["Profile"]("ashworth_hall", QApplication.instance())
        prof.setPersistentStoragePath(str(store / "storage"))
        prof.setCachePath(str(store / "cache"))
        prof.setHttpCacheType(_web["Profile"].HttpCacheType.MemoryHttpCache)
        prof.setPersistentCookiesPolicy(_web["Profile"].PersistentCookiesPolicy.NoPersistentCookies)
        _web["profile"] = prof
    return _web["profile"]


def open_in_browser(path: Path) -> bool:
    """Open the page in the default browser. Nothing is started on the offscreen platform (tests)."""
    if QGuiApplication.platformName().startswith("offscreen"):
        return True
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class AshworthHallGame(ArcadeGame):
    game_id = "ashworth_hall"
    title = "ASHWORTH HALL"
    subtitle = "The great lawn restoration"
    description = ("Forty years of grass, one tractor. Drive the mower over the great lawn of Ashworth Hall, with a "
                   "cold beer and a fine cigar within reach, until the estate looks cared for again.")
    controls = (("W A S D", "Drive"), ("Mouse", "Look around"), ("Space", "Blades"), ("Shift", "Full throttle"),
                ("Q / E", "Drink / smoke"), ("F", "Cheers"), ("C", "Camera"), ("H", "Horn"), ("Esc", "Pause"))
    accent = "#86c04a"

    ITEMS = ("PLAY IN BROWSER", "BACK TO ARCADE")

    def __init__(self) -> None:
        super().__init__()
        self.launches = 0
        self.last_played = ""
        self.best = 0.0  # % of the lawn mown, best sitting
        self.mown = 0.0  # this sitting
        self.view = None  # the web view, while the game is open
        self.page = None
        self.state = "idle"  # idle, loading, ready, browser (no web view: the page opens in the browser)
        self.message = ""
        self.selected = 0
        self.hover = -1
        self.allow_mouse_lock = True  # tests switch it off: nothing may take the mouse there
        self.mouse_lock_requests = 0
        self.opener = open_in_browser  # tests replace it
        self._poll = 0.0
        self._counted = False
        self._escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._escape.activated.connect(self._escape_pressed)

    # --- progress -------------------------------------------------------------------------------------------------

    def save_state(self) -> dict:
        return {"launches": self.launches, "last_played": self.last_played, "best": round(self.best, 1)}

    def load_state(self, data: dict) -> None:
        data = data if isinstance(data, dict) else {}
        try:
            self.launches = max(int(data.get("launches", 0)), 0)
            self.best = min(max(float(data.get("best", 0.0)), 0.0), 100.0)
        except (TypeError, ValueError):
            self.launches, self.best = 0, 0.0
        self.last_played = str(data.get("last_played", ""))

    @classmethod
    def summary(cls, data: dict) -> str:
        data = data if isinstance(data, dict) else {}
        try:
            best = float(data.get("best", 0.0))
        except (TypeError, ValueError):
            best = 0.0
        if best > 0.0:
            return f"Best sitting: {best:.1f}% of the lawn mown"
        return "The lawn is waiting" if not data.get("launches") else "Back on the tractor"

    # --- the web view ---------------------------------------------------------------------------------------------

    def _count_launch(self) -> None:
        if not self._counted:
            self._counted = True
            self.launches += 1
            self.last_played = dt.date.today().isoformat()
            self.progress_changed.emit()

    def _open_view(self) -> None:
        """Make the web view and load the page (once per time the game is opened)."""
        if self.view is not None or self.state == "browser":
            return
        if not PAGE.is_file():
            self._use_browser("The game's page (ashworth_hall.html) is missing from the arcade folder.")
            return
        web = None if QGuiApplication.platformName().startswith("offscreen") else web_classes()
        if web is None:
            self._use_browser("This computer's install has no built-in web view, so the game opens in your browser.")
            return
        try:
            page = web["Page"](profile(), self)
            s = page.settings()
            attr = web["Settings"].WebAttribute
            s.setAttribute(attr.LocalStorageEnabled, True)
            s.setAttribute(attr.LocalContentCanAccessRemoteUrls, False)
            s.setAttribute(attr.LocalContentCanAccessFileUrls, False)
            s.setAttribute(attr.JavascriptCanOpenWindows, False)
            s.setAttribute(attr.ScrollAnimatorEnabled, False)
            page.setBackgroundColor(QColor("#050604"))
            page.setAudioMuted(audio.is_muted())
            self._connect_permissions(page)
            page.renderProcessTerminated.connect(self._page_died)
            view = web["View"](self)
            view.setPage(page)
            view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            view.setGeometry(self.rect())
            view.loadFinished.connect(self._loaded)
            view.installEventFilter(self)
            self.view, self.page, self.state = view, page, "loading"
            view.load(QUrl.fromLocalFile(str(PAGE)))
            view.show()
            view.setFocus()
            self._watch_focus_proxy()
            self._count_launch()
        except Exception:  # noqa: BLE001 - the web view did not start: the browser still plays the page
            log.warning("Ashworth Hall: the web view did not start", exc_info=True)
            self._close_view()
            self._use_browser("The built-in web view did not start, so the game opens in your browser.")

    def _connect_permissions(self, page) -> None:  # noqa: ANN001
        """Mouse look: the page asks to hold the mouse pointer when the player clicks into it. Nothing else is
        allowed (no camera, microphone, location or notifications)."""
        if hasattr(page, "permissionRequested"):  # Qt 6.8 and later
            def asked(permission) -> None:  # noqa: ANN001
                wanted = "MouseLock" in str(permission.permissionType())
                self.mouse_lock_requests += int(wanted)
                if wanted and self.allow_mouse_lock:
                    permission.grant()
                else:
                    permission.deny()

            page.permissionRequested.connect(asked)
        else:
            def asked_feature(url, feature) -> None:  # noqa: ANN001
                wanted = feature == type(page).Feature.MouseLock
                self.mouse_lock_requests += int(wanted)
                policy = type(page).PermissionPolicy
                page.setFeaturePermission(url, feature, policy.PermissionGrantedByUser
                                          if wanted and self.allow_mouse_lock else policy.PermissionDeniedByUser)

            page.featurePermissionRequested.connect(asked_feature)

    def _watch_focus_proxy(self) -> None:
        """The view hands its keys to an inner widget: Esc is caught there too."""
        proxy = self.view.focusProxy() if self.view is not None else None
        if proxy is not None:
            proxy.installEventFilter(self)

    def _loaded(self, ok: bool) -> None:
        if self.view is None:
            return
        if not ok:
            self._close_view()
            self._use_browser("The page did not load in the built-in web view, so the game opens in your browser.")
            return
        self.state = "ready"
        self._watch_focus_proxy()
        if not self.paused:
            self.view.setFocus()

    def _page_died(self, *args) -> None:  # noqa: ANN002 - the web view's own process stopped
        log.warning("Ashworth Hall: the web view's process stopped (%s)", args[:2])
        self._close_view()
        self._use_browser("The built-in web view stopped. You can keep playing in your browser.")

    def _close_view(self) -> None:
        view, page, self.view, self.page = self.view, self.page, None, None
        if page is not None:
            try:
                page.setAudioMuted(True)
            except RuntimeError:
                pass
        if view is not None:
            view.hide()
            view.setParent(None)
            view.deleteLater()
        if page is not None:
            page.deleteLater()
        if self.state != "browser":
            self.state = "idle"

    def _use_browser(self, why: str) -> None:
        self.state = "browser"
        self.message = why
        self.setFocus()
        self.update()

    def _play_in_browser(self) -> None:
        if not PAGE.is_file():
            self.message = "The game's page (ashworth_hall.html) is missing from the arcade folder."
        elif self.opener(PAGE):
            self._count_launch()
            self.message = "Opened in your browser: the game runs there. Come back any time."
        else:
            self.message = f"Could not start a browser. Open this file yourself: {PAGE}"
        self.update()

    # --- lifecycle ------------------------------------------------------------------------------------------------

    def on_resume(self) -> None:
        if self.view is None:
            if self.isVisible():  # a game built but never shown (tests) loads nothing
                self._open_view()
        else:
            self.page.setAudioMuted(audio.is_muted())
            self.view.setGeometry(self.rect())
            self.view.show()
            self.view.setFocus()

    def on_pause(self) -> None:
        if self.view is not None:  # hidden: the page stops drawing and its sound is cut
            self.page.runJavaScript(_UNLOCK_JS)
            self.page.setAudioMuted(True)
            self.view.hide()
            self.setFocus()

    def stop(self) -> None:
        super().stop()
        self._close_view()

    def tick(self, dt_: float) -> None:
        self._poll += dt_
        if self._poll < 1.0 or self.page is None:
            return
        self._poll = 0.0
        self.page.setAudioMuted(audio.is_muted())  # the arcade's mute key
        if self.state == "ready":
            self.page.runJavaScript(_PROGRESS_JS, 0, self._progress)

    def _progress(self, text) -> None:  # noqa: ANN001 - the page's "% mown" readout
        try:
            self.mown = min(max(float(text), 0.0), 100.0)
        except (TypeError, ValueError):
            return
        if self.mown > self.best:
            whole = int(self.mown) > int(self.best)
            self.best = self.mown
            if whole:  # saved at every whole per cent (and when the game is left)
                self.progress_changed.emit()

    # --- input ----------------------------------------------------------------------------------------------------

    def _escape_pressed(self) -> None:
        if self.paused:
            return
        if self.state == "browser":
            self.exit_requested.emit()
        else:
            self.pause_requested.emit()

    def handle_escape(self) -> bool:
        if self.state == "browser":
            self.exit_requested.emit()
            return True
        return False

    def eventFilter(self, obj, event) -> bool:  # noqa: N802, ANN001 - Qt API
        kind = event.type()
        if kind == QEvent.Type.ChildAdded and obj is self.view:
            self._watch_focus_proxy()
        elif kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape and self.view is not None:
            if not event.isAutoRepeat():
                self._escape_pressed()
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        super().resizeEvent(event)
        if self.view is not None:
            self.view.setGeometry(self.rect())

    def _item_rects(self) -> list[QRectF]:
        w, h = self.width(), self.height()
        bw, bh = min(w * 0.34, 400), max(h * 0.065, 38)
        return [QRectF((w - bw) / 2, h * 0.6 + i * bh * 1.3, bw, bh) for i in range(len(self.ITEMS))]

    def _activate(self, i: int) -> None:
        if i == 0:
            self._play_in_browser()
        else:
            self.exit_requested.emit()

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001
        if self.state != "browser":
            return
        if key in (Qt.Key.Key_Down, Qt.Key.Key_S, Qt.Key.Key_Tab):
            self.selected = (self.selected + 1) % len(self.ITEMS)
        elif key in (Qt.Key.Key_Up, Qt.Key.Key_W, Qt.Key.Key_Backtab):
            self.selected = (self.selected - 1) % len(self.ITEMS)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._activate(self.selected)

    def mouse_moved(self, pos: QPointF) -> None:
        if self.state == "browser":
            self.hover = next((i for i, r in enumerate(self._item_rects()) if r.contains(pos)), -1)

    def mouse_pressed(self, button: Qt.MouseButton, pos: QPointF) -> None:
        if self.state != "browser" or button != Qt.MouseButton.LeftButton:
            return
        i = next((i for i, r in enumerate(self._item_rects()) if r.contains(pos)), -1)
        if i >= 0:
            self.selected = i
            self._activate(i)

    # --- drawing --------------------------------------------------------------------------------------------------

    @classmethod
    def _estate(cls, p: QPainter, rect: QRectF, t: float) -> None:
        """The backdrop: an evening sky, the park wall, the long grass, a mown stripe and the tractor on it."""
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        sky = QLinearGradient(rect.topLeft(), QPointF(x, y + h * 0.5))
        sky.setColorAt(0.0, QColor("#232a3c"))
        sky.setColorAt(0.6, QColor("#6b6a78"))
        sky.setColorAt(1.0, QColor("#d9b98a"))
        p.fillRect(QRectF(x, y, w, h * 0.5), sky)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(22, 32, 20))
        for k in range(7):  # trees behind the wall
            cx = x + w * (0.04 + k * 0.16 + 0.03 * math.sin(k * 2.1))
            r = h * (0.13 + 0.04 * math.sin(k * 1.7))
            p.drawEllipse(QPointF(cx, y + h * 0.36), r * 1.3, r)
        p.fillRect(QRectF(x, y + h * 0.4, w, h * 0.1), QColor("#6f6a5e"))  # the wall, with its coping
        p.fillRect(QRectF(x, y + h * 0.4, w, h * 0.012), QColor("#8d8778"))
        p.setPen(QPen(QColor(0, 0, 0, 40), 1))
        for k in range(1, 14):
            p.drawLine(QPointF(x + w * k / 14, y + h * 0.412), QPointF(x + w * k / 14, y + h * 0.5))
        lawn = QLinearGradient(QPointF(x, y + h * 0.5), rect.bottomLeft())
        lawn.setColorAt(0.0, QColor("#3f7a25"))
        lawn.setColorAt(1.0, QColor("#16330f"))
        p.fillRect(QRectF(x, y + h * 0.5, w, h * 0.5), lawn)
        drive = (t * 0.06) % 1.3 - 0.15  # the tractor crosses, leaving a mown stripe behind it
        tx = x + w * drive
        stripe = QPainterPath()
        stripe.moveTo(x, y + h * 0.74)
        stripe.lineTo(max(tx, x), y + h * 0.74)
        stripe.lineTo(max(tx, x), y + h * 0.9)
        stripe.lineTo(x, y + h * 0.9)
        stripe.closeSubpath()
        p.fillPath(stripe, QBrush(QColor(134, 192, 74, 150)))
        p.setPen(QPen(QColor(150, 205, 90, 190), max(h * 0.006, 1.0)))
        for k in range(46):  # long grass, swaying
            gx = x + w * ((k * 0.0731) % 1.0)
            gy = y + h * (0.62 + ((k * 0.377) % 1.0) * 0.38)
            if x <= gx < tx and y + h * 0.72 < gy < y + h * 0.92:
                continue  # already mown
            sway = math.sin(t * 1.6 + k) * h * 0.012
            p.drawLine(QPointF(gx, gy), QPointF(gx + sway, gy - h * (0.07 + 0.03 * ((k * 13) % 5) / 5)))
        u = h * 0.011  # the tractor
        by = y + h * 0.8
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#b3261e"))
        p.drawRoundedRect(QRectF(tx - 9 * u, by - 6 * u, 15 * u, 6 * u), u, u)
        p.drawRoundedRect(QRectF(tx + 3 * u, by - 5 * u, 6 * u, 5 * u), u, u)
        p.setBrush(QColor("#e4d7b0"))
        p.drawRoundedRect(QRectF(tx - 8 * u, by - 12 * u, 3 * u, 7 * u), u * 0.6, u * 0.6)  # seat back
        p.setBrush(QColor("#4b6ea0"))
        p.drawRoundedRect(QRectF(tx - 5.5 * u, by - 12.5 * u, 4 * u, 7 * u), u, u)  # the driver
        p.setBrush(QColor("#e0b48c"))
        p.drawEllipse(QPointF(tx - 3.5 * u, by - 14.5 * u), 1.9 * u, 1.9 * u)
        p.setBrush(QColor("#6b5a44"))
        p.drawRoundedRect(QRectF(tx - 5.6 * u, by - 16.8 * u, 4.6 * u, 1.5 * u), u * 0.5, u * 0.5)  # flat cap
        p.setBrush(QColor("#15140f"))
        p.drawEllipse(QPointF(tx - 5 * u, by), 4 * u, 4 * u)
        p.drawEllipse(QPointF(tx + 6 * u, by + u), 3 * u, 3 * u)
        p.setBrush(QColor("#8c8a80"))
        p.drawEllipse(QPointF(tx - 5 * u, by), 1.6 * u, 1.6 * u)
        p.drawEllipse(QPointF(tx + 6 * u, by + u), 1.2 * u, 1.2 * u)

    @classmethod
    def _crest(cls, p: QPainter, rect: QRectF, size: float) -> None:
        p.setFont(font(size, False, SERIF))
        f = p.font()
        f.setLetterSpacing(f.SpacingType.PercentageSpacing, 118)
        p.setFont(f)
        p.setPen(QColor(0, 0, 0, 170))
        p.drawText(rect.translated(size * 0.04, size * 0.04), Qt.AlignmentFlag.AlignCenter, cls.title)
        p.setPen(QColor("#fff6df"))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, cls.title)

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        p.save()
        p.setClipRect(rect)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cls._estate(p, rect, t)
        size = min(rect.height() * 0.15, rect.width() * 0.085)
        cls._crest(p, QRectF(rect.x(), rect.y() + rect.height() * 0.06, rect.width(), size * 1.5), size)
        p.setPen(GOLD)
        p.setFont(font(max(rect.height() * 0.055, 6), True))
        p.drawText(QRectF(rect.x(), rect.y() + rect.height() * 0.06 + size * 1.5, rect.width(), rect.height() * 0.09),
                   Qt.AlignmentFlag.AlignCenter, "L A W N   R E S T O R A T I O N")
        p.restore()

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        if self.view is not None and self.view.isVisible():
            return  # the page covers the whole cabinet
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        full = QRectF(0, 0, w, h)
        self._estate(p, full, self.time)
        p.fillRect(full, QColor(0, 0, 0, 110))
        size = min(h * 0.11, w * 0.07)
        self._crest(p, QRectF(0, h * 0.14, w, size * 1.5), size)
        p.setPen(GOLD)
        p.setFont(font(max(h * 0.02, 10), True))
        p.drawText(QRectF(0, h * 0.14 + size * 1.5, w, h * 0.05), Qt.AlignmentFlag.AlignCenter,
                   "T H E   G R E A T   L A W N   R E S T O R A T I O N")
        p.setPen(QColor(243, 234, 216, 220))
        p.setFont(font(max(h * 0.022, 12)))
        if self.state == "browser":
            p.drawText(QRectF(w * 0.15, h * 0.4, w * 0.7, h * 0.16),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                       self.message)
            for i, (label, r) in enumerate(zip(self.ITEMS, self._item_rects(), strict=True)):
                on = i == self.selected or i == self.hover
                p.setPen(QPen(GOLD if on else QColor(255, 255, 255, 80), 2))
                p.setBrush(QColor(60, 96, 34, 220) if on else QColor(12, 18, 10, 210))
                p.drawRoundedRect(r, 6, 6)
                p.setPen(QColor("white"))
                p.setFont(font(r.height() * 0.38, True))
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, label)
        else:
            line = "Waking the groundskeeper…" if self.state in ("idle", "loading") else \
                f"The tractor is waiting  ·  {self.mown:.1f}% mown this sitting"
            p.drawText(QRectF(0, h * 0.44, w, h * 0.06), Qt.AlignmentFlag.AlignCenter, line)
        p.end()
