"""BlockCraft's GUI kit: a blocky pixel font with drop shadows, grey beveled buttons, sliders, text fields, inventory
panels and slots, the hotbar, hearts / hunger / bubbles, and isometric item icons. Everything is drawn in GUI pixels
(the painter is scaled by the GUI scale), nearest-pixel, so it stays crisp like the real thing."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter

from app.arcade.games.blockcraft import blocks as B

# --- the pixel font ----------------------------------------------------------------------------------------------
_G = {
    "A": ".###./#...#/#...#/#####/#...#/#...#/#...#", "B": "####./#...#/####./#...#/#...#/#...#/####.",
    "C": ".###./#...#/#..../#..../#..../#...#/.###.", "D": "####./#...#/#...#/#...#/#...#/#...#/####.",
    "E": "#####/#..../###../#..../#..../#..../#####", "F": "#####/#..../###../#..../#..../#..../#....",
    "G": ".####/#..../#..##/#...#/#...#/#...#/.###.", "H": "#...#/#...#/#####/#...#/#...#/#...#/#...#",
    "I": "###/.#./.#./.#./.#./.#./###", "J": "....#/....#/....#/....#/....#/#...#/.###.",
    "K": "#...#/#..#./###../#..#./#...#/#...#/#...#", "L": "#..../#..../#..../#..../#..../#..../#####",
    "M": "#...#/##.##/#.#.#/#...#/#...#/#...#/#...#", "N": "#...#/##..#/#.#.#/#..##/#...#/#...#/#...#",
    "O": ".###./#...#/#...#/#...#/#...#/#...#/.###.", "P": "####./#...#/####./#..../#..../#..../#....",
    "Q": ".###./#...#/#...#/#...#/#...#/#..#./.##.#", "R": "####./#...#/####./#...#/#...#/#...#/#...#",
    "S": ".####/#..../.###./....#/....#/#...#/.###.", "T": "#####/..#../..#../..#../..#../..#../..#..",
    "U": "#...#/#...#/#...#/#...#/#...#/#...#/.###.", "V": "#...#/#...#/#...#/#...#/#...#/.#.#./..#..",
    "W": "#...#/#...#/#...#/#...#/#.#.#/##.##/#...#", "X": "#...#/.#.#./..#../.#.#./#...#/#...#/#...#",
    "Y": "#...#/.#.#./..#../..#../..#../..#../..#..", "Z": "#####/....#/...#./..#../.#.../#..../#####",
    "a": "...../...../.###./....#/.####/#...#/.####", "b": "#..../#..../#.##./##..#/#...#/#...#/####.",
    "c": "...../...../.###./#..../#..../#...#/.###.", "d": "....#/....#/.##.#/#..##/#...#/#...#/.####",
    "e": "...../...../.###./#...#/#####/#..../.####", "f": "..##/.#../####/.#../.#../.#../.#..",
    "g": "...../...../.####/#...#/#...#/.####/....#/####.", "h": "#..../#..../#.##./##..#/#...#/#...#/#...#",
    "i": "#/./#/#/#/#/#", "j": "....#/...../....#/....#/....#/....#/#...#/.###.",
    "k": "#.../#.../#..#/#.#./##../#.#./#..#", "l": "#./#./#./#./#./#./.#",
    "m": "...../...../##.#./#.#.#/#.#.#/#...#/#...#", "n": "...../...../####./#...#/#...#/#...#/#...#",
    "o": "...../...../.###./#...#/#...#/#...#/.###.", "p": "...../...../#.##./##..#/#...#/####./#..../#....",
    "q": "...../...../.##.#/#..##/#...#/.####/....#/....#", "r": "...../...../#.##./##..#/#..../#..../#....",
    "s": "...../...../.####/#..../.###./....#/####.", "t": ".#../.#../####/.#../.#../.#../..##",
    "u": "...../...../#...#/#...#/#...#/#...#/.####", "v": "...../...../#...#/#...#/#...#/.#.#./..#..",
    "w": "...../...../#...#/#...#/#.#.#/#.#.#/.####", "x": "...../...../#...#/.#.#./..#../.#.#./#...#",
    "y": "...../...../#...#/#...#/#...#/.####/....#/####.", "z": "...../...../#####/...#./..#../.#.../#####",
    "0": ".###./#...#/#..##/#.#.#/##..#/#...#/.###.", "1": "..#../.##../..#../..#../..#../..#../#####",
    "2": ".###./#...#/....#/..##./.#.../#...#/#####", "3": ".###./#...#/....#/..##./....#/#...#/.###.",
    "4": "...##/..#.#/.#..#/#...#/#####/....#/....#", "5": "#####/#..../####./....#/....#/#...#/.###.",
    "6": "..##./.#.../#..../####./#...#/#...#/.###.", "7": "#####/#...#/....#/...#./..#../..#../..#..",
    "8": ".###./#...#/#...#/.###./#...#/#...#/.###.", "9": ".###./#...#/#...#/.####/....#/...#./.##..",
    " ": "..././././././.", ".": "./././././././#", ",": "./././././#/#/#", ":": "./././#/./././#",
    ";": "./././#/././#/#", "!": "#/#/#/#/#/./#", "?": ".###./#...#/....#/...#./..#../...../..#..",
    "'": "#/#/./././././.", '"': "#.#/#.#/.../.../.../.../...", "-": "...../...../...../#####/...../...../.....",
    "+": "...../..#../..#../#####/..#../..#../.....", "=": "...../...../#####/...../...../#####/.....",
    "/": "....#/...#./...#./..#../.#.../.#.../#....", "\\": "#..../.#.../.#.../..#../...#./...#./....#",
    "(": "..#/.#./#../#../#../.#./..#", ")": "#../.#./..#/..#/..#/.#./#..", "[": "###/#../#../#../#../#../###",
    "]": "###/..#/..#/..#/..#/..#/###", "<": "...#/..#./.#../#.../.#../..#./...#",
    ">": "#.../.#../..#./...#/..#./.#../#...", "_": "...../...../...../...../...../...../...../#####",
    "%": "#...#/#..#./...#./..#../.#.../.#..#/#...#", "#": ".#.#./.#.#./#####/.#.#./#####/.#.#./.#.#.",
    "*": "...../...../#..#./.##../#..#./...../.....", "&": ".##../#..#./.##../.##.#/#..#./#..#./.##.#",
    "@": ".###./#...#/#.###/#.#.#/#.###/#..../.###.", "|": "#/#/#/#/#/#/#",
    "°": ".#./#.#/.#./.../.../.../...",
}
GLYPHS: dict[str, np.ndarray] = {}
for _ch, _rows in _G.items():
    rows = _rows.split("/")
    w = max(len(r) for r in rows)
    g = np.zeros((8, w), bool)
    for _y, _r in enumerate(rows[:8]):
        for _x, _c in enumerate(_r):
            g[_y, _x] = _c == "#"
    GLYPHS[_ch] = g

WHITE, GREY, YELLOW, DARK = (255, 255, 255), (160, 160, 160), (255, 255, 160), (64, 64, 64)
_text_cache: dict = {}


def text_width(text: str) -> int:
    """Width of a string in GUI pixels (each glyph plus one pixel of spacing)."""
    return sum(GLYPHS.get(ch, GLYPHS["?"]).shape[1] + 1 for ch in text)


def _text_image(text: str, colour: tuple, shadow: bool) -> QImage:
    key = (text, colour, shadow)
    img = _text_cache.get(key)
    if img is not None:
        return img
    w = max(text_width(text), 1) + 1
    a = np.zeros((9, w), bool)
    x = 0
    for ch in text:
        g = GLYPHS.get(ch, GLYPHS["?"])
        a[:8, x:x + g.shape[1]] |= g
        x += g.shape[1] + 1
    rgba = np.zeros((9, w, 4), np.uint8)
    if shadow:
        sh = np.zeros_like(a)
        sh[1:, 1:] = a[:-1, :-1]
        rgba[sh] = (colour[0] // 4, colour[1] // 4, colour[2] // 4, 255)
    rgba[a] = (*colour, 255)
    img = QImage(rgba.data, w, 9, w * 4, QImage.Format.Format_RGBA8888).copy()
    if len(_text_cache) > 600:
        _text_cache.clear()
    _text_cache[key] = img
    return img


def draw_text(p: QPainter, x: float, y: float, text: str, colour: tuple = WHITE, shadow: bool = True,
              scale: int = 1, align: str = "left") -> int:
    """Draw text at (x, y) in GUI pixels (top-left; ``align`` left / center / right); returns its width."""
    if not text:
        return 0
    img = _text_image(text, tuple(colour), shadow)
    w = text_width(text) * scale
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    p.drawImage(QRectF(round(x), round(y), img.width() * scale, img.height() * scale), img)
    return w


# --- textures for widgets ----------------------------------------------------------------------------------------

def to_qimage(rgba: np.ndarray) -> QImage:
    rgba = np.ascontiguousarray(rgba, np.uint8)
    h, w = rgba.shape[:2]
    if rgba.shape[2] == 3:
        return QImage(rgba.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()


_widget_cache: dict = {}


def button_image(w: int, h: int, state: str) -> QImage:
    """A Minecraft-style button (``state``: normal / hover / disabled), w x h GUI pixels."""
    key = ("button", w, h, state)
    if key in _widget_cache:
        return _widget_cache[key]
    rng = np.random.default_rng(3)
    base = {"normal": 111, "hover": 126, "disabled": 44}[state]
    noise = rng.integers(-7, 8, (h, w))
    g = np.clip(base + noise, 0, 255)
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 0] = g
    img[..., 1] = g
    img[..., 2] = g + (40 if state == "hover" else 0)
    img[..., 3] = 255
    light = {"normal": (170, 170, 170), "hover": (190, 200, 255), "disabled": (80, 80, 80)}[state]
    dark = {"normal": (86, 86, 86), "hover": (90, 96, 150), "disabled": (30, 30, 30)}[state]
    img[1, 1:w - 1, :3] = light
    img[1:h - 1, 1, :3] = light
    img[h - 2, 1:w - 1, :3] = dark
    img[h - 3, 1:w - 1, :3] = dark
    img[1:h - 1, w - 2, :3] = dark
    img[0, :, :3] = img[h - 1, :, :3] = 0
    img[:, 0, :3] = img[:, w - 1, :3] = 0
    if state == "hover":
        img[0, :, :3] = img[h - 1, :, :3] = 255
        img[:, 0, :3] = img[:, w - 1, :3] = 255
    q = to_qimage(img)
    _widget_cache[key] = q
    return q


def draw_button(p: QPainter, r: QRect, label: str, hover: bool = False, enabled: bool = True) -> None:
    state = "disabled" if not enabled else ("hover" if hover else "normal")
    p.drawImage(QRectF(r), button_image(r.width(), r.height(), state))
    colour = GREY if not enabled else (YELLOW if hover else WHITE)
    draw_text(p, r.x() + r.width() / 2, r.y() + (r.height() - 8) / 2, label, colour, align="center")


def draw_slider(p: QPainter, r: QRect, label: str, value: float, hover: bool = False) -> None:
    """A slider: the dark track, a button-like knob at ``value`` (0..1) and the label over it."""
    p.drawImage(QRectF(r), button_image(r.width(), r.height(), "disabled"))
    kx = r.x() + int(value * (r.width() - 8))
    p.drawImage(QRectF(kx, r.y(), 8, r.height()), button_image(8, r.height(), "hover" if hover else "normal"))
    draw_text(p, r.x() + r.width() / 2, r.y() + (r.height() - 8) / 2, label, YELLOW if hover else WHITE,
              align="center")


def draw_field(p: QPainter, r: QRect, text: str, focused: bool, blink: bool) -> None:
    """A text field: black box with a grey (white when focused) border."""
    p.fillRect(r, QColor(160, 160, 160) if not focused else QColor(255, 255, 255))
    p.fillRect(r.adjusted(1, 1, -1, -1), QColor(0, 0, 0))
    shown = text
    while text_width(shown) > r.width() - 10 and shown:
        shown = shown[1:]
    w = draw_text(p, r.x() + 4, r.y() + (r.height() - 8) / 2, shown, (224, 224, 224))
    if focused and blink:
        draw_text(p, r.x() + 4 + w, r.y() + (r.height() - 8) / 2, "_", (224, 224, 224))


def panel_image(w: int, h: int) -> QImage:
    """The light grey inventory panel with its rounded, beveled border."""
    key = ("panel", w, h)
    if key in _widget_cache:
        return _widget_cache[key]
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3] = 198
    img[..., 3] = 255
    img[0, 2:w - 3, :3] = img[h - 1, 3:w - 2, :3] = 0
    img[2:h - 3, 0, :3] = img[3:h - 2, w - 1, :3] = 0
    img[1, 1:w - 3, :3] = 255
    img[2, 1:w - 4, :3] = 255
    img[1:h - 3, 1, :3] = 255
    img[1:h - 4, 2, :3] = 255
    img[h - 2, 3:w - 1, :3] = 85
    img[h - 3, 4:w - 2, :3] = 85
    img[3:h - 1, w - 2, :3] = 85
    img[4:h - 2, w - 3, :3] = 85
    for (y, x) in ((0, 0), (0, 1), (1, 0), (0, w - 1), (0, w - 2), (0, w - 3), (1, w - 1), (1, w - 2),
                   (h - 1, 0), (h - 1, 1), (h - 1, 2), (h - 2, 0), (h - 2, 1), (h - 1, w - 1), (h - 1, w - 2),
                   (h - 2, w - 1)):
        img[y, x, 3] = 0
    img[1, w - 3, :3] = img[2, w - 2, :3] = 0
    img[h - 3, 1, :3] = img[h - 2, 2, :3] = 0
    img[1, w - 2, 3] = img[h - 2, 1, 3] = 0
    q = to_qimage(img)
    _widget_cache[key] = q
    return q


def draw_slot(p: QPainter, x: int, y: int, size: int = 18) -> None:
    """An inventory slot (18 x 18 GUI px; the item goes at x + 1, y + 1)."""
    p.fillRect(QRect(x, y, size, size), QColor(139, 139, 139))
    p.fillRect(QRect(x, y, size - 1, 1), QColor(55, 55, 55))
    p.fillRect(QRect(x, y, 1, size - 1), QColor(55, 55, 55))
    p.fillRect(QRect(x + 1, y + size - 1, size - 1, 1), QColor(255, 255, 255))
    p.fillRect(QRect(x + size - 1, y + 1, 1, size - 1), QColor(255, 255, 255))


def draw_dirt_background(p: QPainter, w: int, h: int, gs: int, scroll: float = 0.0, dim: int = 64) -> None:
    """The options-screen background: dark dirt tiles (32 GUI px each)."""
    key = ("dirt", dim)
    img = _widget_cache.get(key)
    if img is None:
        t = B.ATLAS[B.TEX_INDEX["dirt"]][..., :3].astype(np.float32) * (dim / 255)
        img = to_qimage(t.astype(np.uint8))
        _widget_cache[key] = img
    tile = 32 * gs
    oy = int(scroll * gs) % tile
    p.save()
    p.resetTransform()
    for y in range(-tile, h + tile, tile):
        for x in range(0, w + tile, tile):
            p.drawImage(QRectF(x, y + oy, tile, tile), img)
    p.restore()


# --- HUD sprites -------------------------------------------------------------------------------------------------

def _sprite(rows: list[str], colours: dict[str, tuple]) -> QImage:
    h, w = len(rows), max(len(r) for r in rows)
    img = np.zeros((h, w, 4), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in colours:
                img[y, x] = (*colours[ch], 255)
    return to_qimage(img)


_HEART = ["..##.##..", ".#rr#rr#.", "#rwrrrrr#", "#rrrrrrr#", "#rrrrrrd#", ".#rrrrd#.", "..#rdd#..",
          "...#d#...", "....#...."]
_HEART_COLOURS = {"#": (0, 0, 0), "r": (220, 20, 20), "w": (255, 200, 200), "d": (150, 10, 10)}
_FOOD = ["....##...", "...#ww#..", "..#ww#...", ".#bb##...", "#bbbbb#..", "#bbbbbb#.", "#bbbbbb#.",
         ".#bbbb#..", "..####..."]
_FOOD_COLOURS = {"#": (40, 20, 10), "w": (240, 230, 220), "b": (170, 90, 40)}
_BUBBLE = ["..###..", ".#www#.", "#wbbbb#", "#wbbbb#", "#bbbbb#", ".#bbb#.", "..###.."]
_BUBBLE_COLOURS = {"#": (20, 60, 160), "w": (240, 250, 255), "b": (100, 160, 255)}
_ARMOR = ["#.#####.#", "#########", "#aaaaaaa#", "#awaaaaa#", "#aaaaaad#", ".#aaaad#.", ".#aaaad#.", "..#add#..",
          "...###..."]
_ARMOR_COLOURS = {"#": (20, 20, 24), "a": (200, 204, 212), "w": (250, 250, 255), "d": (130, 134, 146)}

_hud: dict = {}


def hud_sprite(name: str) -> QImage:
    if name in _hud:
        return _hud[name]
    if name == "heart":
        img = _sprite(_HEART, _HEART_COLOURS)
    elif name == "heart_half":
        rows = [r[:5] + "".join("x" if c in "rwd" else c for c in r[5:]) for r in _HEART]
        img = _sprite(rows, {**_HEART_COLOURS, "x": (60, 10, 10)})
    elif name == "heart_empty":
        rows = ["".join("x" if c in "rwd" else c for c in r) for r in _HEART]
        img = _sprite(rows, {**_HEART_COLOURS, "x": (60, 10, 10)})
    elif name == "food":
        img = _sprite(_FOOD, _FOOD_COLOURS)
    elif name == "food_half":
        rows = [r[:4] + "".join("x" if c in "wb" else c for c in r[4:]) for r in _FOOD]
        img = _sprite(rows, {**_FOOD_COLOURS, "x": (60, 40, 30)})
    elif name == "food_empty":
        rows = ["".join("x" if c in "wb" else c for c in r) for r in _FOOD]
        img = _sprite(rows, {**_FOOD_COLOURS, "x": (60, 40, 30)})
    elif name == "bubble":
        img = _sprite(_BUBBLE, _BUBBLE_COLOURS)
    elif name == "armor":
        img = _sprite(_ARMOR, _ARMOR_COLOURS)
    elif name == "armor_half":
        rows = [r[:5] + "".join("x" if c in "awd" else c for c in r[5:]) for r in _ARMOR]
        img = _sprite(rows, {**_ARMOR_COLOURS, "x": (58, 60, 68)})
    elif name == "armor_empty":
        rows = ["".join("x" if c in "awd" else c for c in r) for r in _ARMOR]
        img = _sprite(rows, {**_ARMOR_COLOURS, "x": (58, 60, 68)})
    elif name == "hotbar":
        a = np.zeros((22, 182, 4), np.uint8)
        a[..., 3] = 255
        a[..., :3] = 20
        a[1:21, 1:181, :3] = 115
        for i in range(9):
            x = 1 + i * 20
            a[2:20, x + 1:x + 19, :3] = 40
            a[2:20, x + 1:x + 19, 3] = 150
            a[2, x + 1:x + 19, :3] = 30
            a[19, x + 1:x + 19, :3] = 70
        a[1, 1:181, :3] = 150
        img = to_qimage(a)
    elif name == "selector":
        a = np.zeros((24, 24, 4), np.uint8)
        a[0, :, :] = a[23, :, :] = (0, 0, 0, 255)
        a[:, 0, :] = a[:, 23, :] = (0, 0, 0, 255)
        a[1, 1:23] = a[22, 1:23] = (255, 255, 255, 255)
        a[1:23, 1] = a[1:23, 22] = (255, 255, 255, 255)
        a[2, 2:22] = a[21, 2:22] = (160, 160, 160, 255)
        a[2:22, 2] = a[2:22, 21] = (160, 160, 160, 255)
        img = to_qimage(a)
    else:
        img = QImage(1, 1, QImage.Format.Format_RGBA8888)
        img.fill(0)
    _hud[name] = img
    return img


# --- item icons --------------------------------------------------------------------------------------------------
_icons: dict = {}


def _iso_block(b: int, n: int) -> np.ndarray:
    """An isometric little block (top, front-left and front-right faces) like the inventory icons."""
    if b == B.FURNACE:
        b = B.FURNACE + 2  # facing south: the front shows
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) + 0.5
    m = n * 0.06
    cx = n / 2
    w = n / 2 - m
    hh = w * 0.5
    sh = n - 2 * m - 2 * hh
    T = np.array([cx, m])
    R = np.array([cx + w, m + hh])
    C = np.array([cx, m + 2 * hh])
    L = np.array([cx - w, m + hh])
    out = np.zeros((n, n, 4), np.float32)

    def face(origin, eu, ev, tex: int, shade: float) -> None:  # noqa: ANN001
        Mi = np.linalg.inv(np.array([[eu[0], ev[0]], [eu[1], ev[1]]]))
        px, py = xx - origin[0], yy - origin[1]
        u = Mi[0, 0] * px + Mi[0, 1] * py
        v = Mi[1, 0] * px + Mi[1, 1] * py
        inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        ui = np.clip((u * 16).astype(int), 0, 15)
        vi = np.clip((v * 16).astype(int), 0, 15)
        t = B.ATLAS[tex]
        col = t[vi, ui].astype(np.float32)
        ok = inside & (col[..., 3] > 127)
        out[ok, :3] = col[ok, :3] * shade
        out[ok, 3] = 255

    face(T, R - T, L - T, int(B.FACE_TEX[b, 2]), 1.0)
    face(L, C - L, np.array([0, sh]), int(B.FACE_TEX[b, 4]), 0.8)
    face(C, R - C, np.array([0, sh]), int(B.FACE_TEX[b, 0]), 0.6)
    return out.astype(np.uint8)


def item_icon(item: int, n: int = 32) -> QImage:
    """The icon of an item, n x n pixels: a little block, or the flat sprite."""
    key = (item, n)
    img = _icons.get(key)
    if img is None:
        if B.flat_item(item):
            s = B.item_sprite(item)
            k = max(n // 16, 1)
            img = to_qimage(np.repeat(np.repeat(s, k, 0), k, 1))
        else:
            img = to_qimage(_iso_block(item, n))
        _icons[key] = img
    return img


def draw_item(p: QPainter, x: float, y: float, item: int, count: int = 1, damage: int = 0, gs: int = 2,
              size: int = 16) -> None:
    """An item in a slot at (x, y) GUI px: its icon, the stack count (bottom right) and the durability bar."""
    if not item:
        return
    p.drawImage(QRectF(x, y, size, size), item_icon(item, size * max(gs, 1)))
    if count > 1:
        draw_text(p, x + size + 1, y + size - 7, str(count), WHITE, align="right")
    md = B.max_damage(item)
    if md and damage > 0:
        frac = max(0.0, 1 - damage / md)
        p.fillRect(QRectF(x + 2, y + 13, 13, 2), QColor(0, 0, 0))
        col = QColor.fromHsvF(frac / 3, 1.0, 1.0)
        p.fillRect(QRectF(x + 2, y + 13, max(round(13 * frac), 1), 1), col)


def gui_scale(w: int, h: int, setting: int = 0) -> int:
    """Minecraft's automatic GUI scale: the largest that leaves at least 320 x 240 GUI pixels."""
    auto = max(1, min(w // 320, h // 240))
    return min(setting, auto) if setting else auto


def blit_scaled(p: QPainter, img: QImage, target: QRectF) -> None:
    p.save()
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    p.drawImage(target, img)
    p.restore()


ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
