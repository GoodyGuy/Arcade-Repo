"""A 5 x 7 pixel font for the arcade's scoreboards, LED ticker and counters (no font files ship with the program).

``text_image`` gives the text as a small QImage (one image pixel per font pixel, optionally drawn as round LED dots and
with a soft glow); ``draw`` paints it scaled with hard pixel edges. Images are cached, so drawing the same text every
frame costs one ``drawImage``."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter

GLYPH_W, GLYPH_H = 5, 7
ADVANCE = GLYPH_W + 1  # one blank column between letters

_ROWS = {
    "A": "01110 10001 10001 11111 10001 10001 10001", "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110", "D": "11100 10010 10001 10001 10001 10010 11100",
    "E": "11111 10000 10000 11110 10000 10000 11111", "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110", "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001", "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001", "N": "10001 10001 11001 10101 10011 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101", "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110", "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110", "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 10101 01010", "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 10001 01010 00100 00100 00100", "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    " ": "00000 00000 00000 00000 00000 00000 00000", ".": "00000 00000 00000 00000 00000 01100 01100",
    ",": "00000 00000 00000 00000 01100 00100 01000", ":": "00000 01100 01100 00000 01100 01100 00000",
    ";": "00000 01100 01100 00000 01100 00100 01000", "!": "00100 00100 00100 00100 00100 00000 00100",
    "?": "01110 10001 00001 00010 00100 00000 00100", "-": "00000 00000 00000 11111 00000 00000 00000",
    "+": "00000 00100 00100 11111 00100 00100 00000", "/": "00000 00001 00010 00100 01000 10000 00000",
    "'": "00100 00100 01000 00000 00000 00000 00000", '"': "01010 01010 00000 00000 00000 00000 00000",
    "(": "00010 00100 01000 01000 01000 00100 00010", ")": "01000 00100 00010 00010 00010 00100 01000",
    "%": "11000 11001 00010 00100 01000 10011 00011", "#": "01010 01010 11111 01010 11111 01010 01010",
    "*": "00000 00100 10101 01110 10101 00100 00000", "=": "00000 00000 11111 00000 11111 00000 00000",
    "<": "00010 00100 01000 10000 01000 00100 00010", ">": "01000 00100 00010 00001 00010 00100 01000",
    "_": "00000 00000 00000 00000 00000 00000 11111", "&": "01100 10010 10100 01000 10101 10010 01101",
    "@": "01110 10001 00001 01101 10101 10101 01110", "$": "00100 01111 10100 01110 00101 11110 00100",
    "[": "01110 01000 01000 01000 01000 01000 01110", "]": "01110 00010 00010 00010 00010 00010 01110",
    "↑": "00100 01110 10101 00100 00100 00100 00100", "↓": "00100 00100 00100 00100 10101 01110 00100",
    "←": "00000 00100 01000 11111 01000 00100 00000", "→": "00000 00100 00010 11111 00010 00100 00000",
    "★": "00100 01110 11111 01110 01010 10001 00000", "♥": "00000 01010 11111 11111 01110 00100 00000",
    "▶": "01000 01100 01110 01111 01110 01100 01000", "◀": "00010 00110 01110 11110 01110 00110 00010",
    "·": "00000 00000 00000 00100 00000 00000 00000", "¢": "00100 01110 10100 10100 10101 01110 00100",
    "×": "00000 10001 01010 00100 01010 10001 00000", "♪": "00110 00101 00100 00100 01100 11100 11000",
    "|": "00100 00100 00100 00100 00100 00100 00100", "~": "00000 00000 01000 10101 00010 00000 00000",
}
GLYPHS: dict[str, np.ndarray] = {
    ch: np.array([[c == "1" for c in row] for row in rows.split()], dtype=bool) for ch, rows in _ROWS.items()
}


def normalise(text: str) -> str:
    """Upper case; characters the font does not have become '?'."""
    return "".join(ch if ch in GLYPHS else "?" for ch in str(text).upper())


def bitmap(text: str) -> np.ndarray:
    """The text as a (7, n) bool array."""
    text = normalise(text)
    if not text:
        return np.zeros((GLYPH_H, 0), dtype=bool)
    out = np.zeros((GLYPH_H, len(text) * ADVANCE - 1), dtype=bool)
    for i, ch in enumerate(text):
        out[:, i * ADVANCE:i * ADVANCE + GLYPH_W] = GLYPHS[ch]
    return out


def width(text: str, px: float = 1.0) -> float:
    """The drawn width of ``text`` with font pixels of ``px`` screen pixels."""
    return max(len(normalise(text)) * ADVANCE - 1, 0) * px


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return a
    k = np.ones(2 * r + 1) / (2 * r + 1)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, a)


def _to_image(rgba: np.ndarray) -> QImage:
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()


_cache: OrderedDict = OrderedDict()
_CACHE_SIZE = 400


def text_image(text: str, color: QColor | str, cell: int = 1, dots: bool = False, glow: bool = False) -> QImage:
    """The text as an RGBA image, ``cell`` image pixels per font pixel (``dots``: round LED dots; ``glow``: a soft
    halo of the colour around the lit pixels, ``cell`` pixels wide)."""
    c = QColor(color)
    key = (text, c.rgba(), cell, dots, glow)
    image = _cache.get(key)
    if image is not None:
        _cache.move_to_end(key)
        return image
    bits = bitmap(text).astype(float)
    pad = cell if glow else 0
    if cell > 1:
        if dots:
            yy, xx = np.mgrid[0:cell, 0:cell]
            r = (cell - 1) / 2
            kernel = np.clip(r + 0.5 - np.hypot(yy - r, xx - r) * 1.12, 0.0, 1.0)
        else:
            kernel = np.ones((cell, cell))
        bits = np.kron(bits, kernel)
    if pad:
        bits = np.pad(bits, pad)
    alpha = bits
    if glow:
        halo = _box_blur(bits, max(cell // 2, 1))
        halo = _box_blur(halo, max(cell // 2, 1))
        alpha = np.maximum(bits, np.clip(halo * 0.9, 0.0, 1.0) * 0.55)
    rgba = np.zeros(alpha.shape + (4,), dtype=np.uint8)
    core = bits[..., None]
    base = np.array([c.red(), c.green(), c.blue()], dtype=float)
    hot = base + (255.0 - base) * 0.45  # lit pixels a little whiter than the halo
    rgba[..., :3] = np.clip(base * (1 - core) + hot * core, 0, 255).astype(np.uint8) if glow else base.astype(np.uint8)
    rgba[..., 3] = np.clip(alpha * c.alpha(), 0, 255).astype(np.uint8)
    image = _to_image(rgba)
    _cache[key] = image
    if len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return image


def draw(p: QPainter, text: str, x: float, y: float, px: float, color: QColor | str = "white",
         align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft, glow: bool = False, dots: bool = False) -> QRectF:
    """Paint ``text`` with its top at ``y``; ``x`` is the left edge, the centre (AlignHCenter) or the right edge
    (AlignRight). ``px``: screen pixels per font pixel. Returns the text's rectangle."""
    w = width(text, px)
    if align & Qt.AlignmentFlag.AlignHCenter:
        x -= w / 2
    elif align & Qt.AlignmentFlag.AlignRight:
        x -= w
    rect = QRectF(x, y, w, GLYPH_H * px)
    if not w:
        return rect
    cell = max(int(round(px)), 1) if (dots or glow) else 1
    image = text_image(text, color, cell, dots, glow)
    pad = px if glow else 0.0
    smooth = p.testRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, cell > 1)
    p.drawImage(rect.adjusted(-pad, -pad, pad, pad), image)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
    return rect


def fit_px(text: str, max_width: float, max_px: float, min_px: float = 1.0) -> float:
    """The largest pixel size (up to ``max_px``) at which ``text`` fits in ``max_width``."""
    w = width(text, 1.0)
    return max(min(max_px, max_width / w if w else max_px), min_px)
