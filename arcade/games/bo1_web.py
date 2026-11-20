"""BO1 Zombies (web): a cabinet that starts the browser version of Black Ops 1 Zombies (a third-party website, not part
of this program) in Chrome or Edge, where the co-op extension in ``Desktop\\BO1 Zombies`` can be loaded. The link is
read from that folder's ``.url`` shortcut. Nothing runs inside the program: the browser is started on its own."""

from __future__ import annotations

import datetime as dt
import math
import os
from pathlib import Path

from PySide6.QtCore import QPointF, QProcess, QRectF, Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QGuiApplication, QLinearGradient, QPainter, QPainterPath, \
    QPen, QRadialGradient

from app.arcade.base import ArcadeGame, font

DEFAULT_URL = "https://vel.gg/bsz"
FOLDER_NAME = "BO1 Zombies"
RED = QColor("#b3121b")
GREEN = QColor("#7dff6a")


def game_folder() -> Path:
    """``Desktop\\BO1 Zombies`` (``TDS_ARCADE_BO1_DIR`` overrides it)."""
    override = os.environ.get("TDS_ARCADE_BO1_DIR")
    if override:
        return Path(override)
    return Path.home() / "Desktop" / FOLDER_NAME


def game_url(folder: Path | None = None) -> str:
    """The link in the folder's ``.url`` shortcut (``URL=...``), else the game's address."""
    folder = folder or game_folder()
    try:
        for shortcut in sorted(folder.glob("*.url")):
            for line in shortcut.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip().upper().startswith("URL="):
                    url = line.split("=", 1)[1].strip()
                    if url.startswith(("https://", "http://")):
                        return url
    except OSError:
        pass
    return DEFAULT_URL


def extension_folder(folder: Path | None = None) -> Path | None:
    """The co-op extension (the folder with its manifest.json), None when it is not there."""
    folder = folder or game_folder()
    for candidate in (folder / "bo1-zombies-coop" / "extension", folder / "bo1-zombies-coop", folder / "extension"):
        if (candidate / "manifest.json").is_file():
            return candidate
    return None


def find_browser() -> tuple[str, str | None]:
    """(name, executable) of Chrome, else Edge (both load the extension); ("your browser", None) for the default."""
    roots = [os.environ.get(v) for v in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
    roots += [r"C:\Program Files", r"C:\Program Files (x86)", str(Path.home() / "AppData" / "Local")]
    for name, rel in (("Google Chrome", r"Google\Chrome\Application\chrome.exe"),
                      ("Microsoft Edge", r"Microsoft\Edge\Application\msedge.exe")):
        for root in filter(None, roots):
            exe = Path(root) / rel
            if exe.is_file():
                return name, str(exe)
    return "your browser", None


def open_in_browser(url: str) -> str:
    """Start the browser on ``url`` (detached: it is not a child of the program); the browser's name. Nothing is
    started on the offscreen platform (tests)."""
    if QGuiApplication.platformName().startswith("offscreen"):
        return "offscreen"
    name, exe = find_browser()
    if exe is not None and QProcess.startDetached(exe, [url]):
        return name
    QDesktopServices.openUrl(QUrl(url))
    return "your browser"


def open_folder(path: Path) -> None:
    if not QGuiApplication.platformName().startswith("offscreen"):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class Bo1WebGame(ArcadeGame):
    """The launcher screen: Play in browser, Co-op setup, Back."""

    game_id = "bo1_web"
    title = "BO1 ZOMBIES"
    subtitle = "Black Ops 1 Zombies, in your browser"
    description = ("Kino der Toten and the other Black Ops 1 Zombies maps, played on a website in your browser (up to "
                   "4 players with the co-op extension). Not part of this program: your progress stays in the browser.")
    controls = (("Enter", "Play in browser"), ("C", "Co-op setup"), ("Esc", "Back to the arcade"))
    accent = "#b3121b"

    ITEMS = ("PLAY IN BROWSER", "CO-OP SETUP", "BACK TO ARCADE")

    def __init__(self) -> None:
        super().__init__()
        self.launches = 0
        self.last_played = ""
        self.browser = ""
        self.selected = 0
        self.hover = -1
        self.setup_open = False
        self.message = ""
        self.message_age = 0.0
        self.url = game_url()
        self.extension = extension_folder()
        self.opener = open_in_browser  # tests replace it
        self.folder_opener = open_folder

    # --- progress -----------------------------------------------------------------------------------------------

    def save_state(self) -> dict:
        return {"launches": self.launches, "last_played": self.last_played, "browser": self.browser}

    def load_state(self, data: dict) -> None:
        data = data if isinstance(data, dict) else {}
        try:
            self.launches = max(int(data.get("launches", 0)), 0)
        except (TypeError, ValueError):
            self.launches = 0
        self.last_played = str(data.get("last_played", ""))
        self.browser = str(data.get("browser", ""))

    @classmethod
    def summary(cls, data: dict) -> str:
        n = data.get("launches", 0) if isinstance(data, dict) else 0
        return f"Opens in your browser · played {n}×" if n else "Opens in your browser · co-op ready"

    # --- actions ------------------------------------------------------------------------------------------------

    def play(self) -> None:
        name = self.opener(self.url)
        self.launches += 1
        self.last_played = dt.date.today().isoformat()
        self.browser = name
        self.message = f"Opened in {name} — the game runs there. Come back any time."
        self.message_age = 0.0
        self.progress_changed.emit()

    def open_extension_folder(self) -> None:
        if self.extension is not None:
            self.folder_opener(self.extension)
            self.message = "Extension folder opened: pick it with “Load unpacked”."
        else:
            self.message = f"No co-op extension found in {game_folder()}"
        self.message_age = 0.0

    def _activate(self, i: int) -> None:
        if i == 0:
            self.play()
        elif i == 1:
            self.setup_open = not self.setup_open
        else:
            self.exit_requested.emit()
        self.update()

    def handle_escape(self) -> bool:
        if self.setup_open:
            self.setup_open = False
        else:
            self.exit_requested.emit()
        self.update()
        return True

    def tick(self, dt_: float) -> None:
        self.message_age += dt_

    # --- input --------------------------------------------------------------------------------------------------

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001
        if key in (Qt.Key.Key_Down, Qt.Key.Key_S, Qt.Key.Key_Tab):
            self.selected = (self.selected + 1) % len(self.ITEMS)
        elif key in (Qt.Key.Key_Up, Qt.Key.Key_W, Qt.Key.Key_Backtab):
            self.selected = (self.selected - 1) % len(self.ITEMS)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._activate(self.selected)
        elif key == Qt.Key.Key_C:
            self._activate(1)
        elif key == Qt.Key.Key_O and self.setup_open:
            self.open_extension_folder()

    def _item_rects(self) -> list[QRectF]:
        w, h = self.width(), self.height()
        bw, bh = min(w * 0.3, 380), max(h * 0.065, 38)
        x = w * 0.08
        return [QRectF(x, h * 0.5 + i * bh * 1.3, bw, bh) for i in range(len(self.ITEMS))]

    def _folder_button(self) -> QRectF:
        panel = self._setup_rect()
        return QRectF(panel.left() + panel.width() * 0.06, panel.bottom() - panel.height() * 0.24,
                      panel.width() * 0.6, panel.height() * 0.11)

    def _setup_rect(self) -> QRectF:
        w, h = self.width(), self.height()
        return QRectF(w * 0.46, h * 0.18, w * 0.48, h * 0.66)

    def mouse_moved(self, pos: QPointF) -> None:
        hover = next((i for i, r in enumerate(self._item_rects()) if r.contains(pos)), -1)
        if hover != self.hover:
            self.hover = hover
            self.update()

    def mouse_pressed(self, button: Qt.MouseButton, pos: QPointF) -> None:
        if button != Qt.MouseButton.LeftButton:
            return
        if self.setup_open and self._folder_button().contains(pos):
            self.open_extension_folder()
            return
        i = next((i for i, r in enumerate(self._item_rects()) if r.contains(pos)), -1)
        if i >= 0:
            self.selected = i
            self._activate(i)

    # --- drawing ------------------------------------------------------------------------------------------------

    @classmethod
    def _theatre(cls, p: QPainter, rect: QRectF, t: float) -> None:
        """The backdrop: a dark theatre, red curtains, a flickering projector beam, green fog and eyes in the dark."""
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        bg = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        bg.setColorAt(0.0, QColor("#0b0606"))
        bg.setColorAt(1.0, QColor("#1a0c08"))
        p.fillRect(rect, bg)
        flicker = 0.75 + 0.25 * math.sin(t * 23.0) * math.sin(t * 7.3)
        beam = QPainterPath()
        beam.moveTo(x + w * 0.5, y + h * 0.02)
        beam.lineTo(x + w * 0.18, y + h * 0.9)
        beam.lineTo(x + w * 0.82, y + h * 0.9)
        beam.closeSubpath()
        g = QLinearGradient(QPointF(x + w * 0.5, y), QPointF(x + w * 0.5, y + h * 0.9))
        g.setColorAt(0.0, QColor(255, 236, 190, int(110 * flicker)))
        g.setColorAt(1.0, QColor(255, 236, 190, 0))
        p.fillPath(beam, QBrush(g))
        for side in (0, 1):  # curtains, gathered in folds
            for k in range(6):
                fx = x + (k * w * 0.035 if side == 0 else w - (k + 1) * w * 0.035)
                c = QLinearGradient(QPointF(fx, 0), QPointF(fx + w * 0.035, 0))
                c.setColorAt(0.0, QColor("#4a0508"))
                c.setColorAt(0.5, QColor("#9e0f16"))
                c.setColorAt(1.0, QColor("#3b0306"))
                p.fillRect(QRectF(fx, y, w * 0.036, h), c)
        p.fillRect(QRectF(x, y, w, h * 0.07), QColor("#5c070b"))
        fog = QRadialGradient(QPointF(x + w * 0.5, y + h * 1.05), w * 0.55)
        fog.setColorAt(0.0, QColor(90, 255, 90, 70))
        fog.setColorAt(1.0, QColor(90, 255, 90, 0))
        p.fillRect(rect, fog)
        p.setPen(Qt.PenStyle.NoPen)
        for k in range(7):  # pairs of glowing eyes in the dark
            ex = x + w * (0.2 + 0.1 * k + 0.02 * math.sin(t * 0.7 + k))
            ey = y + h * (0.72 + 0.05 * ((k * 37) % 5) / 5)
            blink = 0.0 if (t * 0.6 + k * 0.37) % 4.0 < 0.12 else 1.0
            a = int(220 * blink)
            r = max(h * 0.012, 1.5)
            p.setBrush(QColor(255, 210, 60, a) if k % 3 else QColor(GREEN.red(), GREEN.green(), GREEN.blue(), a))
            p.drawEllipse(QPointF(ex, ey), r, r * 0.7)
            p.drawEllipse(QPointF(ex + r * 3.2, ey), r, r * 0.7)

    @classmethod
    def _logo(cls, p: QPainter, rect: QRectF, size: float) -> None:
        p.setFont(font(size, True, "Impact"))
        p.setPen(QColor(0, 0, 0, 200))
        p.drawText(rect.translated(3, 3), Qt.AlignmentFlag.AlignCenter, "ZOMBIES")
        p.setPen(RED.lighter(130))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "ZOMBIES")
        p.setFont(font(size * 0.3, True))
        p.setPen(QColor(235, 225, 210))
        cy = rect.center().y()
        p.drawText(QRectF(rect.left(), cy - size * 1.05, rect.width(), size * 0.4), Qt.AlignmentFlag.AlignCenter,
                   "B L A C K   O P S   I")

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        p.save()
        p.setClipRect(rect)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cls._theatre(p, rect, t)
        size = rect.height() * 0.2
        cls._logo(p, QRectF(rect.x(), rect.y() + rect.height() * 0.3, rect.width(), size * 1.3), size)
        p.setPen(QPen(RED, max(rect.height() * 0.012, 1.5)))  # round tally marks
        for k in range(4):
            bx = rect.x() + rect.width() * (0.08 + k * 0.035)
            p.drawLine(QPointF(bx, rect.y() + rect.height() * 0.1), QPointF(bx + 2, rect.y() + rect.height() * 0.24))
        p.drawLine(QPointF(rect.x() + rect.width() * 0.06, rect.y() + rect.height() * 0.2),
                   QPointF(rect.x() + rect.width() * 0.22, rect.y() + rect.height() * 0.13))
        p.setPen(QColor(235, 225, 210, 200))
        p.setFont(font(rect.height() * 0.06, True))
        p.drawText(rect.adjusted(0, 0, 0, -rect.height() * 0.06),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, "PLAYS IN YOUR BROWSER")
        p.restore()

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        self._theatre(p, QRectF(0, 0, w, h), self.time)
        p.fillRect(QRectF(0, 0, w, h), QColor(0, 0, 0, 90))
        size = min(h * 0.14, w * 0.09)
        logo = QRectF(w * 0.05, h * 0.16, min(w * 0.4, size * 4.2), size * 1.4)
        self._logo(p, logo, size)
        p.setFont(font(max(h * 0.022, 12)))
        p.setPen(QColor(235, 225, 210, 210))
        p.drawText(QRectF(w * 0.08, logo.bottom() + h * 0.01, w * 0.36, h * 0.12),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                   self.description)
        for i, (label, r) in enumerate(zip(self.ITEMS, self._item_rects(), strict=True)):
            on = i == self.selected or i == self.hover
            p.setPen(QPen(RED.lighter(150) if on else QColor(255, 255, 255, 80), 2))
            p.setBrush(QColor(179, 18, 27, 170) if on else QColor(20, 10, 10, 200))
            p.drawRoundedRect(r, 6, 6)
            p.setPen(QColor("white"))
            p.setFont(font(r.height() * 0.4, True))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, label)
        last = self._item_rects()[-1]
        p.setFont(font(max(h * 0.018, 11)))
        p.setPen(QColor(235, 225, 210, 150))
        info = f"Link: {self.url}"
        if self.launches:
            info += f"   ·   played {self.launches}× (last {self.last_played or '-'})"
        p.drawText(QRectF(w * 0.08, last.bottom() + h * 0.03, w * 0.9, h * 0.04), Qt.AlignmentFlag.AlignLeft, info)
        if self.message and self.message_age < 8.0:
            p.setPen(GREEN)
            p.setFont(font(max(h * 0.022, 12), True))
            p.drawText(QRectF(w * 0.08, last.bottom() + h * 0.07, w * 0.86, h * 0.05), Qt.AlignmentFlag.AlignLeft,
                       self.message)
        if self.setup_open:
            self._paint_setup(p)
        p.setFont(font(max(h * 0.016, 10)))
        p.setPen(QColor(255, 255, 255, 110))
        p.drawText(QRectF(0, h * 0.95, w - w * 0.02, h * 0.04), Qt.AlignmentFlag.AlignRight,
                   "Third-party website: it runs in your browser, not in this program.")
        p.end()

    def _paint_setup(self, p: QPainter) -> None:
        panel = self._setup_rect()
        p.setPen(QPen(RED, 2))
        p.setBrush(QColor(12, 8, 8, 235))
        p.drawRoundedRect(panel, 12, 12)
        pad = panel.width() * 0.06
        size = max(panel.height() * 0.042, 11)
        p.setPen(QColor("white"))
        p.setFont(font(size * 1.5, True))
        p.drawText(QRectF(panel.left() + pad, panel.top() + pad * 0.6, panel.width(), size * 2),
                   Qt.AlignmentFlag.AlignLeft, "CO-OP SETUP (once, every player)")
        browser = "chrome://extensions" if find_browser()[0] == "Google Chrome" else "edge://extensions"
        steps = (
            f"1.  Open {browser} in the browser the game opens in.\n"
            "2.  Switch on Developer mode.\n"
            "3.  Click “Load unpacked” and pick the extension folder (the button below).\n"
            "4.  Play: the Co-op card shows top right. Host a game and send the link; friends open the link and "
            "press a key to load the map."
        )
        p.setFont(font(size))
        p.setPen(QColor(235, 225, 210))
        top = panel.top() + pad * 0.6 + size * 3.0
        p.drawText(QRectF(panel.left() + pad, top, panel.width() - 2 * pad, self._folder_button().top() - top - pad),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, steps)
        b = self._folder_button()
        p.setPen(QPen(GREEN, 2))
        p.setBrush(QColor(30, 70, 30, 200) if self.extension is not None else QColor(50, 50, 50, 200))
        p.drawRoundedRect(b, 6, 6)
        p.setPen(QColor("white"))
        p.setFont(font(min(b.height() * 0.38, b.width() * 0.045), True))
        p.drawText(b, Qt.AlignmentFlag.AlignCenter,
                   "OPEN EXTENSION FOLDER (O)" if self.extension is not None else "EXTENSION NOT FOUND")
        if self.extension is not None:
            p.setFont(font(size * 0.75))
            p.setPen(QColor(235, 225, 210, 150))
            text = QRectF(panel.left() + pad, b.bottom() + size * 0.3, panel.width() - 2 * pad, size * 1.2)
            p.drawText(text, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                       p.fontMetrics().elidedText(str(self.extension), Qt.TextElideMode.ElideMiddle, int(text.width())))
