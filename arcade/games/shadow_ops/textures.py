"""Procedural textures (64 x 64, made with numpy and QPainter at map load), the game's colour grade (teal shadows,
warm highlights, a soft S curve) and the atlas: every texture of a map as the layers the graphics card samples."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen

TS = 64  # texture size
TT = TS * TS


# --- helpers -------------------------------------------------------------------------------------------------------

def value_noise(size: int, cells: int, seed: int, octaves: int = 3) -> np.ndarray:
    """Tileable smooth noise in 0..1."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = cells * (2 ** o)
        grid = rng.random((n, n))
        t = np.arange(size) * n / size
        i0 = np.floor(t).astype(int)
        f = t - i0
        f = f * f * (3 - 2 * f)
        i1 = (i0 + 1) % n
        i0 %= n
        top = grid[i0][:, i0] * (1 - f[None, :]) + grid[i0][:, i1] * f[None, :]
        bot = grid[i1][:, i0] * (1 - f[None, :]) + grid[i1][:, i1] * f[None, :]
        out += amp * (top * (1 - f[:, None]) + bot * f[:, None])
        total += amp
        amp *= 0.5
    return out / total


def _rgb(base, n: np.ndarray, amount: float) -> np.ndarray:  # noqa: ANN001
    b = np.array(base, float)
    return np.clip(b[None, None, :] * (1.0 + (n[..., None] - 0.5) * amount), 0, 255)


def _qimage_to_array(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8, img.sizeInBytes()).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return arr[:, :img.width()].copy()


def _array_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def grade(rgb: np.ndarray, style: str) -> np.ndarray:
    """The colour grade: teal in the shadows, orange in the highlights, an S curve, a little more saturation."""
    x = np.clip(rgb.astype(np.float32) / 255.0, 0, 1)
    lum = (x @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    if style == "dusk":
        teal, warm, sat, lift = (-0.02, 0.05, 0.08), (0.16, 0.05, -0.06), 1.05, 0.02
    else:
        teal, warm, sat, lift = (-0.03, 0.04, 0.09), (0.10, 0.05, -0.04), 1.18, 0.0
    x = x + (1 - lum) ** 2 * np.array(teal, np.float32) + lum ** 2 * np.array(warm, np.float32)
    x = np.clip(x, 0, 1)
    x = x * x * (3 - 2 * x) * 0.4 + x * 0.6 + lift
    lum = (x @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    x = lum + (x - lum) * sat
    return np.clip(x * 255.0, 0, 255)


# --- the textures --------------------------------------------------------------------------------------------------

def _boards(base, seed: int, board: int = 8, vertical: bool = False) -> np.ndarray:  # noqa: ANN001
    n = value_noise(TS, 4, seed) * 0.6 + value_noise(TS, 16, seed + 1, 1) * 0.4
    img = _rgb(base, n, 0.5)
    y = np.arange(TS)
    shade = 1.0 - 0.35 * ((y % board) / board) ** 3
    shade[(y % board) == board - 1] = 0.45
    shade[(y % board) == 0] = 1.12
    s = shade[:, None, None] if not vertical else shade[None, :, None]
    return np.clip(img * s, 0, 255)


def _siding(base, seed: int) -> np.ndarray:  # noqa: ANN001
    img = _boards(base, seed, 8)
    dirt = value_noise(TS, 3, seed + 5)
    grime = np.clip((np.arange(TS)[:, None] / TS) ** 3 * 0.5 + (dirt - 0.6) * 0.8, 0, 0.6)
    img = img * (1 - grime[..., None] * 0.6)
    img[:3] = img[:3] * 0.55 + np.array([235, 232, 220]) * 0.45  # trim under the roof
    img[-4:] = img[-4:] * 0.5
    return img


def _wallpaper(seed: int) -> np.ndarray:
    n = value_noise(TS, 4, seed)
    img = _rgb((150, 140, 112), n, 0.35)
    x = np.arange(TS)
    stripe = ((x // 4) % 4 == 0)
    img[:, stripe] *= 0.82
    dots = ((np.arange(TS)[:, None] % 8 == 4) & (x[None, :] % 8 == 2))
    img[dots] = img[dots] * 0.6 + np.array([140, 60, 50]) * 0.4
    wain = _boards((96, 64, 40), seed + 3, 16, vertical=True)
    img[40:] = wain[40:]
    img[39:41] = (60, 40, 26)
    stain = value_noise(TS, 2, seed + 9)
    img *= (0.7 + 0.3 * stain)[..., None]
    return img


def _concrete(seed: int, base=(124, 122, 118)) -> np.ndarray:  # noqa: ANN001
    n = value_noise(TS, 8, seed, 4)
    img = _rgb(base, n, 0.45)
    img[:, :2] *= 0.7
    img[31:33] *= 0.75
    return img


def _brick(seed: int) -> np.ndarray:
    n = value_noise(TS, 16, seed, 2)
    img = _rgb((150, 66, 48), n, 0.6)
    y, x = np.mgrid[0:TS, 0:TS]
    row = y // 8
    mortar = (y % 8 == 7) | (((x + (row % 2) * 8) % 16) == 15)
    img[mortar] = _rgb((150, 145, 135), n, 0.3)[mortar]
    tint = np.random.default_rng(seed).random((8, 4)) * 0.35 + 0.8
    img *= tint[row, ((x + (row % 2) * 8) // 16) % 4][..., None]
    return img


def _fence(seed: int) -> np.ndarray:
    img = _boards((120, 86, 52), seed, 8, vertical=True)
    img[10:13] *= 0.6
    img[50:53] *= 0.6
    return img


def _door(seed: int) -> np.ndarray:
    img = _boards((112, 70, 40), seed, 64)
    img = np.clip(img * 1.0, 0, 255)
    img[:, :4] = (70, 60, 50)
    img[:, -4:] = (70, 60, 50)
    img[:4] = (70, 60, 50)
    for y0, y1 in ((8, 28), (34, 58)):
        img[y0:y1, 10:28] *= 0.82
        img[y0:y1, 36:54] *= 0.82
    img[32:36, 48:52] = (210, 180, 90)  # the knob
    # boards nailed across: it is locked
    for y0 in (18, 44):
        img[y0:y0 + 6, 2:62] = _rgb((150, 110, 70), value_noise(TS, 8, seed + y0), 0.4)[y0:y0 + 6, 2:62]
        img[y0 + 5, 2:62] *= 0.5
    return img


def _debris(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = value_noise(TS, 8, seed)
    img = _rgb((70, 58, 48), n, 0.6) * 0.6
    for _ in range(18):
        x0, y0 = rng.integers(-10, 60), rng.integers(4, 62)
        w, h = rng.integers(14, 40), rng.integers(3, 8)
        col = rng.choice([(120, 90, 60), (90, 90, 96), (140, 120, 90), (60, 50, 44), (150, 40, 30)])
        img[y0:y0 + h, max(x0, 0):x0 + w] = np.array(col) * (0.7 + 0.5 * rng.random())
        img[min(y0 + h, 63), max(x0, 0):x0 + w] *= 0.4
    return img


def _floor_wood(seed: int) -> np.ndarray:
    img = _boards((122, 84, 52), seed, 8, vertical=True)
    stain = value_noise(TS, 2, seed + 4)
    return img * (0.7 + 0.35 * stain)[..., None]


def _tile(seed: int) -> np.ndarray:
    y, x = np.mgrid[0:TS, 0:TS]
    chk = ((x // 16) + (y // 16)) % 2 == 0
    n = value_noise(TS, 8, seed)
    img = np.where(chk[..., None], _rgb((200, 196, 176), n, 0.2), _rgb((70, 120, 120), n, 0.3))
    grout = (x % 16 == 0) | (y % 16 == 0)
    img[grout] = (60, 58, 50)
    return img * (0.7 + 0.3 * value_noise(TS, 2, seed + 2))[..., None]


def _grass(seed: int) -> np.ndarray:
    n = value_noise(TS, 8, seed, 4)
    blades = np.random.default_rng(seed).random((TS, TS))
    img = _rgb((70, 104, 44), n, 0.7)
    img[blades > 0.85] *= 1.3
    img[blades < 0.1] *= 0.7
    return img


def _asphalt(seed: int) -> np.ndarray:
    n = value_noise(TS, 16, seed, 3)
    grit = np.random.default_rng(seed).random((TS, TS))
    img = _rgb((66, 66, 68), n, 0.3)
    img[grit > 0.93] *= 1.18
    img[grit < 0.05] *= 0.88
    crack = value_noise(TS, 3, seed + 7)
    img[np.abs(crack - 0.5) < 0.008] *= 0.75
    return img


def _sidewalk(seed: int) -> np.ndarray:
    img = _concrete(seed, (150, 148, 140))
    img[:, 31:33] *= 0.7
    return img


def _dirt(seed: int) -> np.ndarray:
    n = value_noise(TS, 8, seed, 4)
    img = _rgb((96, 72, 50), n, 0.7)
    rocks = np.random.default_rng(seed).random((TS, TS)) > 0.96
    img[rocks] = (130, 118, 100)
    return img


def _ceiling(seed: int, dark: bool = False) -> np.ndarray:
    n = value_noise(TS, 3, seed, 4)
    img = _rgb((54, 50, 46) if dark else (120, 112, 100), n, 0.5)
    stain = value_noise(TS, 2, seed + 3)
    img *= (0.75 + 0.35 * stain)[..., None]
    img[:, 0:2] *= 0.8  # panel seams
    img[0:2] *= 0.8
    return img


# see-through textures: (rgb, alpha)

def _window(base_rgb: np.ndarray, planks: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    img = base_rgb.copy()
    alpha = np.ones((TS, TS), bool)
    x0, x1, y0, y1 = 10, 54, 12, 44
    alpha[y0:y1, x0:x1] = False
    img[y0 - 3:y0, x0 - 3:x1 + 3] = (88, 70, 54)
    img[y1:y1 + 4, x0 - 4:x1 + 4] = (104, 84, 62)
    img[y0 - 3:y1 + 3, x0 - 3:x0] = (88, 70, 54)
    img[y0 - 3:y1 + 3, x1:x1 + 3] = (88, 70, 54)
    rng = np.random.default_rng(seed)
    wood = value_noise(TS, 8, seed + 1)
    for k in range(planks):
        yc = y0 + 3 + k * (y1 - y0 - 6) / 5.0
        slope = (rng.random() - 0.5) * 0.35
        xs = np.arange(x0 - 2, x1 + 2)
        yy = (yc + (xs - 32) * slope).astype(int)
        col = np.array((150, 112, 72)) * (0.75 + 0.4 * rng.random())
        for dy in range(-2, 3):
            ys = np.clip(yy + dy, 0, TS - 1)
            shade = 0.65 if dy == 2 else (1.1 if dy == -2 else 1.0)
            img[ys, xs] = col * shade * (0.8 + 0.4 * wood[ys, xs])[:, None]
            alpha[ys, xs] = True
        for nx in (x0 + 1, x1 - 2):  # nails
            img[np.clip(int(yc + (nx - 32) * slope), 0, 63), nx] = (40, 40, 40)
    return img, alpha


def _chainlink(seed: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:TS, 0:TS]
    wire = ((x + y) % 8 == 0) | ((x - y) % 8 == 0)
    alpha = wire.copy()
    alpha[:, :3] = True
    alpha[:3] = True
    img = np.zeros((TS, TS, 3))
    img[:] = (150, 156, 160)
    img[:, :3] = (110, 112, 116)
    img[:3] = (120, 124, 128)
    return img, alpha


def _glass(base_rgb: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A glazed opening in a wall: a steel frame, one mullion and transom, a glint across the glass."""
    img = base_rgb.copy()
    alpha = np.ones((TS, TS), bool)
    x0, x1, y0, y1 = 8, 56, 12, 46
    alpha[y0:y1, x0:x1] = False
    frame = (66, 70, 74)
    img[y0 - 3:y0, x0 - 3:x1 + 3] = frame
    img[y1:y1 + 3, x0 - 3:x1 + 3] = frame
    img[y0 - 3:y1 + 3, x0 - 3:x0] = frame
    img[y0 - 3:y1 + 3, x1:x1 + 3] = frame
    for sl in ((slice(y0, y1), slice(31, 33)), (slice(28, 30), slice(x0, x1))):
        img[sl] = frame
        alpha[sl] = True
    y, x = np.mgrid[0:TS, 0:TS]
    glint = (np.abs((x - y) - 4) < 1) & ~alpha
    alpha |= glint
    img[glint] = (236, 246, 252)
    return img, alpha


# --- weapon shapes -----------------------------------------------------------------------------------------------

def gun_path(model: str, w: float, h: float) -> QPainterPath:
    """A weapon's side silhouette fitted into ``w`` x ``h`` (wall racks, menus)."""
    pts = {
        "pistol": [(0.15, 0.3), (0.85, 0.3), (0.85, 0.45), (0.45, 0.45), (0.42, 0.8), (0.28, 0.8), (0.3, 0.45),
                   (0.15, 0.45)],
        "revolver": [(0.1, 0.32), (0.8, 0.32), (0.8, 0.42), (0.5, 0.42), (0.48, 0.55), (0.42, 0.8), (0.28, 0.78),
                     (0.32, 0.5), (0.1, 0.45)],
        "smg": [(0.05, 0.35), (0.7, 0.35), (0.72, 0.42), (0.95, 0.42), (0.95, 0.5), (0.62, 0.5), (0.6, 0.8),
                (0.52, 0.8), (0.5, 0.5), (0.35, 0.5), (0.33, 0.72), (0.25, 0.72), (0.25, 0.5), (0.05, 0.55)],
        "smg2": [(0.05, 0.38), (0.75, 0.38), (0.95, 0.44), (0.95, 0.52), (0.55, 0.52), (0.52, 0.85),
                 (0.44, 0.85), (0.44, 0.52), (0.3, 0.52), (0.26, 0.7), (0.18, 0.7), (0.2, 0.52), (0.05, 0.52)],
        "ar": [(0.0, 0.4), (0.25, 0.36), (0.6, 0.36), (0.62, 0.42), (0.98, 0.42), (0.98, 0.48), (0.6, 0.5),
               (0.56, 0.78), (0.48, 0.78), (0.46, 0.5), (0.36, 0.5), (0.34, 0.68), (0.26, 0.68), (0.26, 0.52),
               (0.0, 0.6)],
        "ar2": [(0.0, 0.42), (0.3, 0.36), (0.66, 0.36), (0.98, 0.42), (0.98, 0.48), (0.62, 0.5), (0.58, 0.8),
                (0.5, 0.8), (0.48, 0.5), (0.36, 0.5), (0.34, 0.7), (0.26, 0.7), (0.26, 0.52), (0.0, 0.58)],
        "dmr": [(0.0, 0.42), (0.2, 0.38), (0.98, 0.4), (0.98, 0.46), (0.55, 0.5), (0.5, 0.7), (0.42, 0.7),
                (0.4, 0.52), (0.25, 0.52), (0.0, 0.6)],
        "shotgun": [(0.0, 0.42), (0.25, 0.38), (0.98, 0.38), (0.98, 0.46), (0.7, 0.46), (0.7, 0.54), (0.4, 0.54),
                    (0.35, 0.68), (0.27, 0.68), (0.28, 0.52), (0.0, 0.62)],
        "shotgun2": [(0.0, 0.44), (0.3, 0.4), (0.98, 0.4), (0.98, 0.5), (0.35, 0.52), (0.3, 0.66), (0.22, 0.64),
                     (0.24, 0.52), (0.0, 0.64)],
        "lmg": [(0.0, 0.4), (0.25, 0.34), (0.7, 0.34), (0.98, 0.4), (0.98, 0.46), (0.6, 0.5), (0.58, 0.62),
                (0.45, 0.72), (0.38, 0.6), (0.34, 0.7), (0.26, 0.7), (0.26, 0.52), (0.0, 0.6)],
        "sniper": [(0.0, 0.44), (0.2, 0.4), (0.3, 0.4), (0.32, 0.3), (0.6, 0.3), (0.62, 0.4), (0.98, 0.42),
                   (0.98, 0.46), (0.5, 0.5), (0.46, 0.66), (0.38, 0.66), (0.36, 0.52), (0.0, 0.6)],
        "launcher": [(0.0, 0.36), (0.98, 0.36), (0.98, 0.56), (0.55, 0.56), (0.52, 0.76), (0.44, 0.76),
                     (0.42, 0.56), (0.0, 0.56)],
        "raygun": [(0.1, 0.3), (0.6, 0.26), (0.9, 0.36), (0.9, 0.46), (0.6, 0.5), (0.45, 0.5), (0.42, 0.82),
                   (0.3, 0.82), (0.32, 0.5), (0.1, 0.46)],
    }.get(model, [(0.0, 0.4), (1.0, 0.4), (1.0, 0.55), (0.0, 0.55)])
    path = QPainterPath()
    path.moveTo(pts[0][0] * w, pts[0][1] * h)
    for x, y in pts[1:]:
        path.lineTo(x * w, y * h)
    path.closeSubpath()
    return path


def _rack(base: np.ndarray, model: str, cost: int, seed: int) -> np.ndarray:
    """A wall rack: a steel pegboard with the weapon's shape hung on it and its price on a plate."""
    img = _array_to_qimage(base)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    board = QRectF(5, 14, 54, 30)
    p.setPen(QPen(QColor(18, 20, 22), 1.2))
    p.setBrush(QColor(46, 52, 56))
    p.drawRoundedRect(board, 2, 2)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(20, 22, 24))
    for py in range(17, 43, 4):
        for px in range(8, 58, 4):
            p.drawRect(QRectF(px, py, 1, 1))
    p.setBrush(QColor(242, 196, 60))
    p.drawRect(QRectF(5, 12, 54, 2))
    p.drawRect(QRectF(5, 44, 54, 2))
    p.save()
    p.translate(9, 13)
    path = gun_path(model, 46, 32)
    p.setPen(QPen(QColor(8, 8, 10), 1.0))
    p.setBrush(QColor(150, 156, 160))
    p.drawPath(path)
    p.restore()
    plate = QRectF(20, 47, 24, 9)
    p.setPen(QPen(QColor(18, 18, 18), 0.8))
    p.setBrush(QColor(242, 196, 60))
    p.drawRect(plate)
    f = QFont("Arial")
    f.setPixelSize(8)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(20, 16, 8))
    p.drawText(plate, int(Qt.AlignmentFlag.AlignCenter), str(cost))
    p.end()
    return _qimage_to_array(img)[..., :3].astype(float)


def _corrugated(base, seed: int, pitch: int = 8, vertical: bool = True) -> np.ndarray:  # noqa: ANN001
    """Ribbed sheet steel: round ribs, a little rust running down, a darker foot."""
    n = value_noise(TS, 6, seed, 3)
    img = _rgb(base, n, 0.22)
    t = np.arange(TS)
    rib = 0.74 + 0.34 * np.sin((t % pitch) / pitch * np.pi) ** 0.8
    img *= rib[None, :, None] if vertical else rib[:, None, None]
    rust = np.clip((value_noise(TS, 5, seed + 3, 2) - 0.62) * 3.5, 0, 1) * (np.arange(TS)[:, None] / TS) ** 0.7
    img = img * (1 - rust[..., None] * 0.7) + np.array([124, 66, 34]) * rust[..., None] * 0.7
    img[-5:] *= 0.62
    return img


def _metal_wall(seed: int) -> np.ndarray:
    img = _corrugated((98, 122, 128), seed)
    img[:4] = np.array([52, 58, 62]) * (0.8 + 0.4 * value_noise(TS, 8, seed + 1)[:4, :, None])  # the eaves rail
    img[4] *= 0.5
    img[30:32] *= 0.72  # a sheet joint
    return img


def _container(base, seed: int) -> np.ndarray:  # noqa: ANN001
    img = _corrugated(base, seed, 8)
    for y0, y1 in ((0, 4), (60, 64)):
        img[y0:y1] = np.array(base) * 0.55
    img[:, :3] = np.array(base) * 0.5
    img[:, -3:] = np.array(base) * 0.5
    img[22:30, 20:44] = np.array([226, 226, 216]) * (0.75 + 0.3 * value_noise(TS, 8, seed + 2)[22:30, 20:44, None])
    img[25:27, 23:41] = np.array(base) * 0.6  # a line of lettering
    return img


def _drywall(seed: int) -> np.ndarray:
    """A works wall: pale paint over a dark green dado, a rubber skirting, scuffs."""
    n = value_noise(TS, 4, seed, 3)
    img = _rgb((176, 172, 150), n, 0.16)
    img[38:] = _rgb((58, 92, 80), n, 0.22)[38:]
    img[36:38] = (226, 190, 70)
    img[-5:] = (34, 34, 36)
    scuff = np.clip((value_noise(TS, 7, seed + 5, 2) - 0.66) * 4, 0, 1)
    img *= (1 - 0.3 * scuff)[..., None]
    stain = value_noise(TS, 2, seed + 9)
    return img * (0.78 + 0.26 * stain)[..., None]


def _shopfloor(seed: int) -> np.ndarray:
    """Sealed concrete: slab joints, tyre marks, grit."""
    n = value_noise(TS, 8, seed, 4)
    img = _rgb((112, 112, 110), n, 0.3)
    img[:, :1] *= 0.6
    img[:1] *= 0.6
    marks = np.clip((value_noise(TS, 3, seed + 4, 2) - 0.6) * 3, 0, 1)
    img *= (1 - 0.35 * marks)[..., None]
    grit = np.random.default_rng(seed).random((TS, TS))
    img[grit > 0.95] *= 1.14
    return img


def _truss(seed: int) -> np.ndarray:
    """A shed roof seen from below: dark sheet with purlins across and one truss chord along."""
    img = _corrugated((46, 50, 54), seed, 8, vertical=False) * 0.8
    img[:, 28:36] = np.array([70, 72, 74]) * (0.8 + 0.4 * value_noise(TS, 8, seed + 2)[:, 28:36, None])
    img[:, 28] *= 0.5
    img[:, 35] *= 0.5
    img[30:34] = np.array([58, 60, 62]) * (0.8 + 0.4 * value_noise(TS, 8, seed + 3)[30:34, :, None])
    return img


def _crate(seed: int, top: bool = False) -> np.ndarray:
    img = _boards((150, 114, 70), seed, 16, vertical=top)
    frame = np.zeros((TS, TS), bool)
    frame[:6] = frame[-6:] = True
    frame[:, :6] = frame[:, -6:] = True
    y, x = np.mgrid[0:TS, 0:TS]
    if not top:
        frame |= np.abs(x - y) < 4  # the brace
    img[frame] = _rgb((122, 88, 52), value_noise(TS, 8, seed + 2), 0.4)[frame]
    edge = np.zeros((TS, TS), bool)
    edge[6] = edge[-7] = True
    edge[:, 6] = edge[:, -7] = True
    img[edge & ~frame] *= 0.55
    for cx, cy in ((3, 3), (60, 3), (3, 60), (60, 60)):
        img[cy - 1:cy + 1, cx - 1:cx + 1] = (40, 40, 42)
    return img


def _barrier(seed: int) -> np.ndarray:
    img = _concrete(seed, (150, 148, 142))
    y, x = np.mgrid[0:TS, 0:TS]
    stripe = ((x + y) // 8) % 2 == 0
    band = (y >= 6) & (y < 18)
    img[band & stripe] = img[band & stripe] * 0.3 + np.array([236, 190, 40]) * 0.7
    img[band & ~stripe] = img[band & ~stripe] * 0.35
    img[-6:] *= 0.7
    return img


def _tank(seed: int) -> np.ndarray:
    """A storage tank's shell: painted plate courses with weld seams, a band, rust weeping from the seams."""
    n = value_noise(TS, 6, seed, 3)
    img = _rgb((188, 192, 186), n, 0.14)
    for y0 in (0, 21, 42, 63):
        img[max(y0 - 1, 0):y0 + 1] *= 0.6
    for x0 in (15, 47):
        img[:21, x0] *= 0.7
        img[42:, x0] *= 0.7
    img[21:42, 31] *= 0.7
    img[24:30] = img[24:30] * 0.25 + np.array([226, 96, 30]) * 0.75
    rust = np.clip((value_noise(TS, 7, seed + 5, 2) - 0.6) * 3, 0, 1)
    drip = np.maximum.accumulate(rust * (np.arange(TS)[:, None] % 21 < 9), axis=0) * 0.5
    img = img * (1 - drip[..., None]) + np.array([120, 70, 40]) * drip[..., None]
    img[-4:] *= 0.6
    return img


def _steel(seed: int) -> np.ndarray:
    n = value_noise(TS, 10, seed, 3)
    img = _rgb((120, 124, 126), n, 0.2)
    y, x = np.mgrid[0:TS, 0:TS]
    ring = np.abs(np.hypot(x - 31.5, y - 31.5) - 24) < 1.2
    img[ring] *= 0.6
    return img


def _drum(seed: int) -> np.ndarray:
    n = value_noise(TS, 6, seed, 3)
    img = _rgb((44, 92, 132), n, 0.25)
    for y0 in (2, 21, 42, 60):
        img[y0:y0 + 2] *= 1.3
        img[y0 + 2] *= 0.6
    img[26:36, 22:42] = (232, 200, 60)
    img[29:33, 26:38] = (30, 30, 30)
    return img


# --- the atlas -----------------------------------------------------------------------------------------------------

def make_texture(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(rgb float (64, 64, 3), opaque mask (64, 64)) of a texture by its map name."""
    opaque = np.ones((TS, TS), bool)
    simple = {
        "concrete": lambda: _concrete(seed), "siding_blue": lambda: _siding((92, 142, 186), seed),
        "brick": lambda: _brick(seed), "fence_wood": lambda: _fence(seed), "door": lambda: _door(seed),
        "debris": lambda: _debris(seed), "wood": lambda: _floor_wood(seed), "tile": lambda: _tile(seed),
        "grass": lambda: _grass(seed), "asphalt": lambda: _asphalt(seed), "sidewalk": lambda: _sidewalk(seed),
        "dirt": lambda: _dirt(seed), "ceiling": lambda: _ceiling(seed), "metal": lambda: _metal_wall(seed),
        "drywall": lambda: _drywall(seed), "container_blue": lambda: _container((52, 96, 140), seed),
        "container_red": lambda: _container((150, 62, 44), seed), "shopfloor": lambda: _shopfloor(seed),
        "ceiling_truss": lambda: _truss(seed), "crate": lambda: _crate(seed),
        "crate_top": lambda: _crate(seed, True), "barrier": lambda: _barrier(seed), "tank": lambda: _tank(seed),
        "steel": lambda: _steel(seed), "drum": lambda: _drum(seed),
    }
    if name in simple:
        return simple[name](), opaque
    parts = name.split(":")
    if parts[0] == "rack":
        from .data import weapon

        base, _ = make_texture(parts[1], seed)
        w = weapon(parts[2])
        return _rack(base, w.model, w.cost, seed), opaque
    if parts[0] == "window":
        base, _ = make_texture(parts[1], seed)
        return _window(base, int(parts[2]), seed + int(parts[2]) * 13)
    if parts[0] == "glass":
        base, _ = make_texture(parts[1] if len(parts) > 1 else "brick", seed)
        return _glass(base, seed)
    if name == "chainlink":
        return _chainlink(seed)
    return np.full((TS, TS, 3), 255.0) * np.array([1, 0, 1]), opaque


class Atlas:
    """Every texture of a map, colour graded: ``rgba`` (n, 128, 128, 4) uint8 (drawn at 64, doubled smoothly), the alpha
    0 where a pane is open."""

    def __init__(self, tex_names: list[str], style: str) -> None:
        self.names = list(tex_names)
        n = len(self.names)
        self.n = n
        rgb = np.zeros((n, TS, TS, 3), np.float32)
        alpha = np.zeros((n, TS, TS), bool)
        for i, name in enumerate(self.names):
            r, a = make_texture(name, 1000 + i * 7 + sum(map(ord, name)))
            rgb[i], alpha[i] = r, a
        rgb = grade(rgb, style)
        self.avg = rgb.reshape(n, -1, 3).mean(axis=1)
        rgb = _double(_double(rgb, 1), 2)  # 128 x 128: smooth between the drawn texels, crisp edges in the shader
        alpha = np.repeat(np.repeat(alpha, 2, 1), 2, 2)
        self.rgba = np.concatenate([np.clip(rgb + 0.5, 0, 255).astype(np.uint8),
                                    (alpha[..., None] * 255).astype(np.uint8)], axis=-1)


def _double(a: np.ndarray, axis: int) -> np.ndarray:
    """Twice the samples along ``axis``, linearly between the old ones (wrapping, as the textures tile)."""
    prev, nxt = np.roll(a, 1, axis), np.roll(a, -1, axis)
    lo, hi = a * 0.75 + prev * 0.25, a * 0.75 + nxt * 0.25
    out = np.stack([lo, hi], axis=axis + 1)
    shape = list(a.shape)
    shape[axis] *= 2
    return out.reshape(shape)


def sprite_from_painter(w: int, h: int, draw) -> np.ndarray:  # noqa: ANN001
    """An RGBA sprite drawn by ``draw(painter, w, h)``."""
    img = QImage(w, h, QImage.Format.Format_RGBA8888)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw(p, w, h)
    p.end()
    return _qimage_to_array(img)


__all__ = ["Atlas", "QPointF", "QRectF", "TS", "TT", "grade", "gun_path", "sprite_from_painter"]
