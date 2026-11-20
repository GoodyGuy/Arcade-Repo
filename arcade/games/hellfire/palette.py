"""The 256-colour palette (ramps of shades, like the 1993 game's), the 32 light-level colour maps, the red / gold /
green screen tints, and the quantiser that turns RGB pictures into palette indices."""

from __future__ import annotations

import numpy as np

# (name, count, colour stops bright -> dark)
RAMPS: tuple[tuple[str, int, tuple[tuple[int, int, int], ...]], ...] = (
    ("black", 1, ((0, 0, 0),)),
    ("grey", 31, ((244, 244, 244), (128, 128, 128), (10, 10, 10))),
    ("flesh", 24, ((255, 196, 184), (206, 110, 96), (120, 46, 36), (36, 8, 6))),
    ("tan", 16, ((246, 204, 144), (176, 132, 80), (66, 44, 20))),
    ("brown", 32, ((214, 156, 104), (138, 88, 48), (70, 42, 20), (16, 10, 4))),
    ("green", 24, ((150, 255, 120), (52, 170, 36), (22, 80, 14), (6, 20, 4))),
    ("olive", 16, ((176, 168, 112), (108, 102, 60), (28, 26, 12))),
    ("red", 32, ((255, 140, 124), (255, 20, 16), (150, 0, 0), (28, 0, 0))),
    ("blue", 16, ((210, 210, 255), (64, 64, 236), (0, 0, 120), (0, 0, 36))),
    ("fire", 24, ((255, 255, 220), (255, 236, 60), (255, 140, 0), (200, 60, 0), (90, 20, 0))),
    ("steel", 16, ((176, 186, 204), (96, 104, 122), (18, 22, 30))),
    ("purple", 8, ((214, 120, 230), (110, 40, 130), (36, 8, 44))),
    ("rust", 16, ((188, 92, 52), (110, 44, 20), (18, 6, 2))),
)


def _build() -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    colours: list[tuple[float, float, float]] = []
    spans: dict[str, tuple[int, int]] = {}
    for name, count, stops in RAMPS:
        start = len(colours)
        stops_a = np.array(stops, dtype=float)
        for i in range(count):
            t = i / max(count - 1, 1) * (len(stops_a) - 1)
            j = min(int(t), len(stops_a) - 2) if len(stops_a) > 1 else 0
            f = t - j
            c = stops_a[j] if len(stops_a) == 1 else stops_a[j] * (1 - f) + stops_a[j + 1] * f
            colours.append(tuple(c))
        spans[name] = (start, count)
    pal = np.clip(np.round(np.array(colours)), 0, 255).astype(np.uint8)
    assert len(pal) == 256, len(pal)
    return pal, spans


PALETTE, SPANS = _build()
BLACK = 0


def ramp(name: str, t: float | np.ndarray) -> np.ndarray | int:
    """The palette index ``t`` (0 = brightest, 1 = darkest) along a ramp."""
    start, count = SPANS[name]
    idx = np.clip(np.round(np.asarray(t, dtype=float) * (count - 1)), 0, count - 1).astype(np.int32) + start
    return idx if np.ndim(idx) else int(idx)


def _nearest(rgb: np.ndarray) -> np.ndarray:
    """Nearest palette index of each (n, 3) colour (weighted for the eye): |c - p|^2 = |c|^2 - 2 c.p + |p|^2."""
    w = np.sqrt(np.array([0.30, 0.59, 0.11], np.float32) * 3)
    pal = PALETTE.astype(np.float32) * w
    p2 = (pal * pal).sum(axis=1)
    out = np.empty(len(rgb), dtype=np.uint8)
    for s in range(0, len(rgb), 8192):
        chunk = rgb[s:s + 8192].astype(np.float32) * w
        out[s:s + 8192] = (p2[None, :] - 2.0 * chunk @ pal.T).argmin(axis=1)
    return out

def _colormaps() -> np.ndarray:
    f = 1.0 - np.arange(32) * (0.85 / 31)  # the darkest map keeps 15 % (never a flat black band)
    maps = _nearest((PALETTE.astype(float)[None, :, :] * f[:, None, None]).reshape(-1, 3)).reshape(32, 256)
    maps[0] = np.arange(256)
    return maps


COLORMAP = _colormaps()  # (32, 256): light level 0 (full bright) .. 31 (darkest)
COLORMAP_FLAT = COLORMAP.ravel()


def _tints() -> np.ndarray:
    """14 screen palettes: 0 normal, 1-8 red (pain / berserk), 9-12 gold (pickups), 13 green (suit)."""
    base = PALETTE.astype(float)
    out = [base]
    for i in range(1, 9):
        f = i / 9.0
        out.append(base + (np.array([255, 0, 0]) - base) * f)
    for i in range(1, 5):
        f = i * 0.125
        out.append(base + (np.array([215, 186, 69]) - base) * f)
    out.append(base + (np.array([0, 256, 0]) - base) * 0.125)
    return np.clip(np.round(np.array(out)), 0, 255).astype(np.uint8)


TINTS = _tints()

_LUT: np.ndarray | None = None


def _lut() -> np.ndarray:
    """RGB (32 levels per channel) -> palette index."""
    global _LUT
    if _LUT is None:
        g = (np.arange(32) * 255 / 31.0)
        r, gg, b = np.meshgrid(g, g, g, indexing="ij")
        _LUT = _nearest(np.stack([r.ravel(), gg.ravel(), b.ravel()], axis=1)).reshape(32, 32, 32)
    return _LUT


def quantize(rgba: np.ndarray, alpha_cut: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """An (h, w, 4) RGBA picture as (palette indices, opaque mask)."""
    lut = _lut()
    q = (rgba[..., :3].astype(np.int32) * 31 + 127) // 255
    idx = lut[q[..., 0], q[..., 1], q[..., 2]]
    mask = rgba[..., 3] >= alpha_cut
    return idx.astype(np.uint8), mask


def to_rgb(frame: np.ndarray, tint: int = 0) -> np.ndarray:
    return TINTS[tint][frame]
