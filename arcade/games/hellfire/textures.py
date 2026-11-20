"""Procedural wall textures (64 x 128), flats (64 x 64) and the sky (256 x 128), all in palette indices, in the style
of the 1993 game's tech base, stone, computer, door and hell textures."""

from __future__ import annotations

import numpy as np

from .palette import ramp
from .pixfont import GLYPH

TW, TH = 64, 128  # wall texture width / height (map units)
FS = 64  # flat size


def vnoise(h: int, w: int, cell: int, seed: int, octaves: int = 3) -> np.ndarray:
    """Tileable value noise in [0, 1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        gh, gw = max(h // cell, 1), max(w // cell, 1)
        g = rng.random((gh, gw))
        ys = np.arange(h) * gh / h
        xs = np.arange(w) * gw / w
        y0, x0 = ys.astype(int), xs.astype(int)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        y1, x1 = (y0 + 1) % gh, (x0 + 1) % gw
        a = g[y0][:, x0] * (1 - fx) + g[y0][:, x1] * fx
        b = g[y1][:, x0] * (1 - fx) + g[y1][:, x1] * fx
        out += (a * (1 - fy) + b * fy) * amp
        total += amp
        amp *= 0.5
        cell = max(cell // 2, 1)
    return out / total


def grain(h: int, w: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((h, w))


def shade(name: str, t: np.ndarray) -> np.ndarray:
    return ramp(name, np.clip(t, 0.0, 1.0)).astype(np.uint8)


def bevel_rect(t: np.ndarray, y0: int, x0: int, y1: int, x1: int, light: float = -0.15, dark: float = 0.2,
               fill: float | None = None) -> None:
    """Raised panel: lighter top/left edge, darker bottom/right edge (in place on a shade map)."""
    if fill is not None:
        t[y0:y1, x0:x1] = fill + (t[y0:y1, x0:x1] - t[y0:y1, x0:x1].mean()) * 0.6
    t[y0, x0:x1] += light
    t[y0:y1, x0] += light
    t[y1 - 1, x0:x1] += dark
    t[y0:y1, x1 - 1] += dark


def rivet(t: np.ndarray, y: int, x: int) -> None:
    h, w = t.shape
    t[y % h, x % w] -= 0.25
    t[(y + 1) % h, (x + 1) % w] += 0.25
    t[y % h, (x + 1) % w] += 0.05


def stamp_text(tex: np.ndarray, text: str, x: int, y: int, colour: int, scale: int = 1) -> None:
    for i, ch in enumerate(text):
        g = GLYPH.get(ch)
        if g is None:
            continue
        gs = np.repeat(np.repeat(g, scale, 0), scale, 1)
        x0 = x + i * 6 * scale
        region = tex[y:y + gs.shape[0], x0:x0 + gs.shape[1]]
        region[gs[:region.shape[0], :region.shape[1]]] = colour


# --- walls ---------------------------------------------------------------------------------------------------------

def _tech(ramp_name: str, seed: int, band: str = "steel") -> np.ndarray:
    t = 0.32 + (vnoise(TH, TW, 16, seed) - 0.5) * 0.12 + (grain(TH, TW, seed + 1) - 0.5) * 0.08
    for y0, y1 in ((0, 40), (40, 52), (76, 128)):
        bevel_rect(t, y0, 0, y1, 32, -0.12, 0.18)
        bevel_rect(t, y0, 32, y1, 64, -0.12, 0.18)
    for x in (6, 26, 38, 58):
        for y in (6, 34, 82, 120):
            rivet(t, y, x)
    tex = shade(ramp_name, t)
    b = 0.35 + (grain(24, TW, seed + 2) - 0.5) * 0.06
    b[0, :] -= 0.2
    b[1, :] -= 0.1
    b[-1, :] += 0.3
    b[-2, :] += 0.15
    b[10:14, :] += 0.25
    b[14, :] -= 0.12
    for x in range(4, 64, 8):
        b[5:8, x:x + 3] += 0.18
    tex[52:76] = shade(band, b)
    return tex


def tex_techtan(seed: int = 1) -> np.ndarray:
    return _tech("tan", seed)


def tex_techgrey(seed: int = 2) -> np.ndarray:
    return _tech("grey", seed, band="tan")


def tex_stone(seed: int = 3, name: str = "grey", base: float = 0.42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = base + (vnoise(TH, TW, 8, seed) - 0.5) * 0.22 + (grain(TH, TW, seed) - 0.5) * 0.1
    y = 0
    row = 0
    while y < TH:
        h = int(rng.integers(14, 26)) if TH - y > 30 else TH - y
        off = int(rng.integers(0, 32))
        xs = sorted({(off + int(v)) % TW for v in rng.integers(0, TW, 2)} | {off % TW})
        t[y:y + h] += (rng.random() - 0.5) * 0.12
        t[y, :] -= 0.12
        t[min(y + h - 1, TH - 1), :] += 0.22
        for x in xs:
            t[y:y + h, x] += 0.24
            t[y:y + h, (x + 1) % TW] -= 0.1
        y += h
        row += 1
    return shade(name, t)


def tex_brown(seed: int = 4) -> np.ndarray:
    t = 0.45 + (vnoise(TH, TW, 16, seed, 4) - 0.5) * 0.35 + (grain(TH, TW, seed) - 0.5) * 0.12
    cracks = vnoise(TH, TW, 8, seed + 9, 2)
    t[np.abs(cracks - 0.5) < 0.02] += 0.3
    for y in range(0, TH, 32):
        t[y, :] += 0.25
        t[y + 1, :] -= 0.1
    return shade("brown", t)


def tex_brownpanel(seed: int = 5) -> np.ndarray:
    t = 0.38 + (grain(TH, TW, seed) - 0.5) * 0.08
    wood = vnoise(TH, TW // 8, 4, seed, 2)
    t += np.repeat(wood, 8, axis=1)[:, :TW] * 0.1
    t += (np.sin(np.arange(TH)[:, None] * 0.7 + np.arange(TW)[None, :] * 0.05) * 0.03)
    for x0 in (0, 16, 32, 48):
        bevel_rect(t, 0, x0, 128, x0 + 16, -0.15, 0.25)
    t[0:4, :] = 0.55
    t[124:, :] = 0.55
    return shade("brown", t)


def tex_compute(frame: int, seed: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = 0.72 + (grain(TH, TW, seed) - 0.5) * 0.06
    for y0 in range(4, 124, 24):
        bevel_rect(t, y0, 2, y0 + 20, 62, -0.3, 0.15)
    tex = shade("grey", t)
    blink = np.random.default_rng(seed + frame * 31)
    colours = [ramp("red", 0.1), ramp("green", 0.05), ramp("fire", 0.2), ramp("blue", 0.25)]
    for y0 in range(4, 124, 24):
        for x in range(6, 58, 6):
            for yy in (y0 + 5, y0 + 12):
                on = blink.random() < 0.55
                c = colours[int(rng.integers(0, 4))] if on else ramp("grey", 0.85)
                tex[yy:yy + 3, x:x + 3] = c
                if on:
                    tex[yy, x] = ramp("grey", 0.0)
    return tex


def tex_compsta(frame: int, seed: int = 7) -> np.ndarray:
    t = 0.6 + (grain(TH, TW, seed) - 0.5) * 0.06
    bevel_rect(t, 0, 0, 128, 64, -0.15, 0.2)
    tex = shade("steel", t)
    rng = np.random.default_rng(seed + frame)
    for y0 in (14, 56):
        tex[y0:y0 + 32, 6:58] = ramp("grey", 0.95)
        scr = np.full((28, 48), ramp("green", 0.9), np.uint8)
        for line in range(0, 28, 4):
            n = int(rng.integers(8, 44))
            scr[line + 1:line + 3, 2:2 + n] = ramp("green", 0.15 + rng.random() * 0.3)
        tex[y0 + 2:y0 + 30, 8:56] = scr
    tex[96:116, 6:58] = shade("grey", 0.55 + (grain(20, 52, seed) > 0.5) * 0.25)
    for x in range(8, 56, 5):
        tex[100:103, x:x + 3] = ramp("grey", 0.2)
        tex[107:110, x + 2:x + 5] = ramp("grey", 0.25)
    tex[120:124, 10:16] = ramp("red", 0.1 if frame % 2 else 0.6)
    tex[120:124, 48:54] = ramp("green", 0.1 if frame % 2 == 0 else 0.6)
    return tex


def tex_metal(seed: int = 8) -> np.ndarray:
    t = 0.55 + (vnoise(TH, TW, 16, seed) - 0.5) * 0.15 + (grain(TH, TW, seed) - 0.5) * 0.1
    for y0 in range(0, TH, 32):
        bevel_rect(t, y0, 0, y0 + 32, 64, -0.15, 0.25)
        for x in (4, 58):
            rivet(t, y0 + 4, x)
            rivet(t, y0 + 26, x)
    stains = vnoise(TH, TW, 32, seed + 3, 2)
    t += np.clip(stains - 0.6, 0, 1) * 0.6
    return shade("steel", t)


def tex_support(seed: int = 9) -> np.ndarray:
    t = 0.55 + (grain(TH, TW, seed) - 0.5) * 0.08
    xs = np.arange(TW)
    t += (np.abs(xs - 31.5) / 32.0)[None, :] * 0.3
    t[:, 14:18] += 0.25
    t[:, 46:50] += 0.25
    for y in range(8, TH, 16):
        rivet(t, y, 22)
        rivet(t, y, 40)
    return shade("steel", t)


def tex_door(seed: int = 10, stripe: str | None = None) -> np.ndarray:
    t = 0.5 + (vnoise(TH, TW, 16, seed) - 0.5) * 0.1 + (grain(TH, TW, seed) - 0.5) * 0.08
    bevel_rect(t, 0, 0, 128, 64, -0.2, 0.3)
    for y0, y1 in ((8, 40), (48, 80), (88, 120)):
        bevel_rect(t, y0, 6, y1, 30, 0.2, -0.15)
        bevel_rect(t, y0, 34, y1, 58, 0.2, -0.15)
    t[:, 31:33] += 0.35
    for y in (12, 36, 52, 76, 92, 116):
        rivet(t, y, 10)
        rivet(t, y, 54)
    tex = shade("grey", t)
    haz = ((np.arange(64)[None, :] + np.arange(10)[:, None]) // 5) % 2
    tex[0:10] = np.where(haz, ramp("fire", 0.35), ramp("grey", 0.92))
    if stripe:
        c0, c1 = ramp(stripe, 0.1), ramp(stripe, 0.5)
        for x0 in (14, 44):
            tex[24:104, x0:x0 + 6] = c1
            tex[24:104, x0 + 1:x0 + 5] = c0
        tex[60:68, 8:56] = c1
    else:
        tex[58:70, 26:38] = ramp("fire", 0.25)
        tex[60:68, 28:36] = ramp("fire", 0.05)
    return tex


def tex_doortrak(seed: int = 11) -> np.ndarray:
    t = 0.75 + (grain(TH, TW, seed) - 0.5) * 0.06
    t[:, 24:40] = 0.45
    t[:, 24] -= 0.2
    t[:, 39] += 0.2
    for y in range(0, TH, 8):
        t[y:y + 3, 26:38] = 0.3
    return shade("grey", t)


def tex_hazard(seed: int = 12) -> np.ndarray:
    ys, xs = np.mgrid[0:TH, 0:TW]
    stripes = ((xs + ys) // 12) % 2
    t = 0.3 + (grain(TH, TW, seed) - 0.5) * 0.08
    tex = np.where(stripes, shade("fire", t + 0.05), shade("grey", 0.9 + t * 0.1))
    return tex.astype(np.uint8)


def tex_switch(on: bool, exit_: bool = False, seed: int = 13) -> np.ndarray:
    tex = tex_techgrey(seed) if not exit_ else tex_metal(seed)
    t = np.full((40, 28), 0.5) + (grain(40, 28, seed) - 0.5) * 0.06
    bevel_rect(t, 0, 0, 40, 28, -0.25, 0.3)
    bevel_rect(t, 6, 8, 34, 20, 0.25, -0.2, fill=0.8)
    plate = shade("steel" if not exit_ else "grey", t)
    if exit_:
        haz = ((np.arange(28)[None, :] + np.arange(40)[:, None]) // 4) % 2
        border = np.zeros((40, 28), bool)
        border[:3] = border[-3:] = True
        border[:, :3] = border[:, -3:] = True
        plate[border & (haz == 1)] = ramp("fire", 0.3)
        plate[border & (haz == 0)] = ramp("grey", 0.9)
    if on:
        plate[24:32, 11:17] = ramp("grey", 0.3)
        plate[28:34, 12:16] = ramp("green", 0.1)
    else:
        plate[8:16, 11:17] = ramp("grey", 0.3)
        plate[6:12, 12:16] = ramp("red", 0.1)
    tex[44:84, 18:46] = plate
    return tex


def tex_exitsign(seed: int = 14) -> np.ndarray:
    tex = tex_techgrey(seed)
    tex[0:22, :] = ramp("grey", 0.93)
    tex[1:21, 1:63] = ramp("grey", 0.88)
    stamp_text(tex, "EXIT", 9, 4, ramp("red", 0.25), scale=2)
    for x in range(10, 56):
        for y in range(4, 18):
            if tex[y, x] == ramp("red", 0.25):
                tex[y, x] = ramp("red", 0.05) if (y - 4) < 6 else ramp("red", 0.25)
    return tex


def tex_lite(seed: int = 15) -> np.ndarray:
    t = 0.65 + (grain(TH, TW, seed) - 0.5) * 0.06
    bevel_rect(t, 0, 0, 128, 64, -0.15, 0.2)
    tex = shade("steel", t)
    for y0 in (16, 72):
        tex[y0:y0 + 40, 12:52] = ramp("grey", 0.6)
        tex[y0 + 2:y0 + 38, 14:50] = ramp("blue", 0.0)
        tex[y0 + 4:y0 + 36, 18:46] = ramp("grey", 0.0)
    return tex


def tex_marble(seed: int = 16) -> np.ndarray:
    n = vnoise(TH, TW, 32, seed, 5)
    veins = np.abs(np.sin(n * 18.0))
    t = 0.45 + (1 - veins) * 0.25 * (veins < 0.25) + (grain(TH, TW, seed) - 0.5) * 0.06
    t -= (veins < 0.08) * 0.2
    for y0 in range(0, TH, 64):
        bevel_rect(t, y0, 0, y0 + 64, 64, -0.12, 0.25)
    return shade("green", 0.15 + t * 0.85)


def tex_skin(seed: int = 17) -> np.ndarray:
    n = vnoise(TH, TW, 16, seed, 4)
    veins = np.abs(np.sin(vnoise(TH, TW, 32, seed + 5, 3) * 25.0))
    t = 0.35 + (n - 0.5) * 0.4 + (grain(TH, TW, seed) - 0.5) * 0.08
    tex = shade("flesh", t)
    tex[veins < 0.12] = shade("red", 0.55 + n[veins < 0.12] * 0.3)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        y, x = int(rng.integers(8, 120)), int(rng.integers(6, 58))
        yy, xx = np.mgrid[-5:6, -5:6]
        d = np.sqrt(yy * yy + xx * xx)
        region = tex[y - 5:y + 6, x - 5:x + 6]
        m = d < 5
        region[m] = shade("red", 0.5 + d[m] / 10.0)[...]
        region[d < 2] = ramp("fire", 0.4)
    return tex


def tex_gstone(seed: int = 18) -> np.ndarray:
    return tex_stone(seed, "green", 0.55)


def tex_slime(seed: int = 19) -> np.ndarray:
    tex = tex_brown(seed)
    rng = np.random.default_rng(seed)
    drip = np.zeros(TW)
    for x in range(TW):
        drip[x] = 10 + rng.random() * 12 + (30 * rng.random() if rng.random() < 0.2 else 0)
    drip = np.convolve(drip, np.ones(3) / 3, mode="same")
    ys = np.arange(TH)[:, None]
    m = ys < drip[None, :]
    g = vnoise(TH, TW, 8, seed + 1, 2)
    tex[m] = shade("green", 0.35 + g[m] * 0.4 + (ys / 80.0 * np.ones((1, TW)))[m])
    tex[TH - 12:, :] = shade("green", 0.45 + g[TH - 12:, :] * 0.3)
    return tex


def tex_pipes(seed: int = 20) -> np.ndarray:
    t = np.full((TH, TW), 0.8)
    for x0, w in ((2, 14), (18, 10), (30, 18), (50, 12)):
        xs = np.arange(w)
        prof = (np.abs(xs - (w - 1) / 2) / (w / 2)) ** 2 * 0.45 + 0.25
        t[:, x0:x0 + w] = prof[None, :]
        for y in range(10 + x0, TH, 40):
            t[y:y + 4, x0:x0 + w] -= 0.1
            t[y + 4, x0:x0 + w] += 0.2
    t += (grain(TH, TW, seed) - 0.5) * 0.06
    return shade("rust", t)


def tex_step(seed: int = 21) -> np.ndarray:
    t = 0.45 + (grain(TH, TW, seed) - 0.5) * 0.1
    for y0 in range(0, TH, 16):
        t[y0:y0 + 2, :] -= 0.2
        t[y0 + 13:y0 + 16, :] += 0.2
    tex = shade("grey", t)
    for y0 in range(0, TH, 16):
        haz = ((np.arange(64)[None, :] + np.arange(3)[:, None]) // 4) % 2
        tex[y0:y0 + 3] = np.where(haz, ramp("fire", 0.35), ramp("grey", 0.9))
    return tex


def tex_tekwall(seed: int = 22) -> np.ndarray:
    t = 0.7 + (grain(TH, TW, seed) - 0.5) * 0.05
    rng = np.random.default_rng(seed)
    tex = shade("steel", t)
    for _ in range(26):
        y, x = int(rng.integers(0, TH)), int(rng.integers(0, TW))
        ln = int(rng.integers(6, 30))
        if rng.random() < 0.5:
            tex[y, x:x + ln] = ramp("green", 0.55)
        else:
            tex[y:y + ln, x] = ramp("green", 0.55)
        tex[y:y + 2, x:x + 2] = ramp("fire", 0.3)
    bevel = np.zeros((TH, TW))
    bevel_rect(bevel, 0, 0, 64, 64, -0.2, 0.2)
    bevel_rect(bevel, 64, 0, 128, 64, -0.2, 0.2)
    tex[bevel < 0] = ramp("steel", 0.35)
    tex[bevel > 0] = ramp("steel", 0.95)
    return tex


def tex_lift(seed: int = 23) -> np.ndarray:
    tex = tex_metal(seed)
    haz = ((np.arange(64)[None, :] + np.arange(8)[:, None]) // 6) % 2
    band = np.where(haz, ramp("fire", 0.35), ramp("grey", 0.9)).astype(np.uint8)
    for y0 in range(0, TH, 32):
        tex[y0:y0 + 8] = band
    return tex


def tex_redbrick(seed: int = 24) -> np.ndarray:
    t = 0.4 + (grain(TH, TW, seed) - 0.5) * 0.15 + (vnoise(TH, TW, 8, seed) - 0.5) * 0.15
    for y0 in range(0, TH, 16):
        t[y0, :] = 0.85
        off = 0 if (y0 // 16) % 2 == 0 else 16
        for x in range(off, TW + 32, 32):
            t[y0:y0 + 16, x % TW] = 0.85
    return shade("rust", t)


def tex_lavafall(frame: int, seed: int = 25) -> np.ndarray:
    n = vnoise(TH * 2, TW, 16, seed, 3)
    roll = np.roll(n, frame * 16, axis=0)[:TH]
    t = 0.15 + roll * 0.5
    return shade("fire", t)


def tex_skull_wall(seed: int = 26) -> np.ndarray:
    tex = tex_marble(seed)
    yy, xx = np.mgrid[0:24, 0:20]
    cy, cx = 10, 9.5
    skull = ((yy - cy) ** 2 / 100 + (xx - cx) ** 2 / 90) < 1
    skull |= (yy > 16) & (yy < 23) & (np.abs(xx - cx) < 5)
    eyes = (((yy - 11) ** 2 + (xx - 6) ** 2) < 6) | (((yy - 11) ** 2 + (xx - 13) ** 2) < 6)
    region = tex[40:64, 22:42]
    region[skull] = shade("tan", 0.15 + yy[skull] / 60.0)
    region[eyes] = ramp("red", 0.1)
    region[(yy > 19) & (yy < 22) & (xx % 2 == 0) & (np.abs(xx - cx) < 5)] = ramp("grey", 0.8)
    return tex


# --- flats ---------------------------------------------------------------------------------------------------------

def flat_floor4(seed: int = 40) -> np.ndarray:
    t = 0.45 + (grain(FS, FS, seed) - 0.5) * 0.1 + (vnoise(FS, FS, 16, seed) - 0.5) * 0.1
    for y0 in range(0, FS, 16):
        for x0 in range(0, FS, 16):
            bevel_rect(t, y0, x0, y0 + 16, x0 + 16, -0.1, 0.2)
    return shade("grey", t)


def flat_floor5(seed: int = 41) -> np.ndarray:
    t = 0.35 + (grain(FS, FS, seed) - 0.5) * 0.12
    for y0 in range(0, FS, 32):
        for x0 in range(0, FS, 32):
            bevel_rect(t, y0, x0, y0 + 32, x0 + 32, -0.15, 0.25)
            for dy, dx in ((4, 4), (4, 27), (27, 4), (27, 27)):
                rivet(t, y0 + dy, x0 + dx)
    t[np.add.outer(np.arange(FS), np.arange(FS)) % 8 == 0] += 0.06
    return shade("tan", t)


def flat_compflr(seed: int = 42) -> np.ndarray:
    t = 0.5 + (grain(FS, FS, seed) - 0.5) * 0.08
    ys, xs = np.mgrid[0:FS, 0:FS]
    t += ((xs % 8 < 2) | (ys % 8 < 2)) * 0.25
    bevel_rect(t, 0, 0, 64, 64, -0.15, 0.2)
    return shade("steel", t)


def flat_ceil3(seed: int = 43) -> np.ndarray:
    t = 0.3 + (grain(FS, FS, seed) - 0.5) * 0.1 + (vnoise(FS, FS, 16, seed) - 0.5) * 0.08
    for y0 in range(0, FS, 32):
        for x0 in range(0, FS, 32):
            bevel_rect(t, y0, x0, y0 + 32, x0 + 32, -0.12, 0.2)
    return shade("tan", t)


def flat_tlite(seed: int = 44) -> np.ndarray:
    t = 0.5 + (grain(FS, FS, seed) - 0.5) * 0.08
    bevel_rect(t, 0, 0, 64, 64, -0.15, 0.25)
    tex = shade("grey", t)
    ys, xs = np.mgrid[0:FS, 0:FS]
    d = np.sqrt((ys - 31.5) ** 2 + (xs - 31.5) ** 2)
    tex[d < 22] = ramp("grey", 0.75)
    tex[d < 20] = shade("grey", (d[d < 20] / 20.0) ** 2 * 0.25)
    return tex


def flat_lightsq(seed: int = 45) -> np.ndarray:
    t = 0.55 + (grain(FS, FS, seed) - 0.5) * 0.08
    bevel_rect(t, 0, 0, 64, 64, -0.15, 0.25)
    tex = shade("steel", t)
    tex[8:56, 8:56] = ramp("grey", 0.7)
    tex[10:54, 10:54] = ramp("grey", 0.0)
    tex[30:34, 10:54] = ramp("grey", 0.5)
    tex[10:54, 30:34] = ramp("grey", 0.5)
    return tex


def _liquid(name: str, frame: int, seed: int, base: float, amp: float) -> np.ndarray:
    n1 = vnoise(FS, FS, 16, seed, 3)
    n2 = vnoise(FS, FS, 16, seed + 7, 3)
    ph = frame / 3.0 * 2 * np.pi
    n = n1 * (0.5 + 0.5 * np.cos(ph)) + n2 * (0.5 - 0.5 * np.cos(ph))
    n = np.roll(n, frame * 3, axis=1)
    t = base + (n - 0.5) * amp * 2
    blobs = np.abs(np.sin(n * 14.0)) < 0.15
    t[blobs] -= 0.15
    return shade(name, t)


def flat_nukage(frame: int) -> np.ndarray:
    return _liquid("green", frame, 46, 0.3, 0.22)


def flat_blood(frame: int) -> np.ndarray:
    return _liquid("red", frame, 47, 0.45, 0.2)


def flat_lava(frame: int) -> np.ndarray:
    return _liquid("fire", frame, 48, 0.35, 0.25)


def flat_rock(seed: int = 49) -> np.ndarray:
    t = 0.45 + (vnoise(FS, FS, 16, seed, 4) - 0.5) * 0.4 + (grain(FS, FS, seed) - 0.5) * 0.15
    cr = vnoise(FS, FS, 8, seed + 3, 2)
    t[np.abs(cr - 0.5) < 0.02] += 0.25
    return shade("brown", t)


def flat_dirt(seed: int = 50) -> np.ndarray:
    t = 0.35 + (vnoise(FS, FS, 8, seed, 4) - 0.5) * 0.3 + (grain(FS, FS, seed) - 0.5) * 0.2
    return shade("rust", t)


def flat_steptop(seed: int = 51) -> np.ndarray:
    t = 0.4 + (grain(FS, FS, seed) - 0.5) * 0.1
    ys, xs = np.mgrid[0:FS, 0:FS]
    t += (((xs + ys) % 6 == 0) | ((xs - ys) % 6 == 0)) * 0.12
    return shade("steel", t)


def flat_exitpad(seed: int = 52) -> np.ndarray:
    t = 0.45 + (grain(FS, FS, seed) - 0.5) * 0.08
    bevel_rect(t, 0, 0, 64, 64, -0.15, 0.25)
    tex = shade("grey", t)
    ys, xs = np.mgrid[0:FS, 0:FS]
    d = np.maximum(np.abs(ys - 31.5), np.abs(xs - 31.5))
    ring = (d > 18) & (d < 26)
    haz = ((xs + ys) // 4) % 2
    tex[ring & (haz == 1)] = ramp("fire", 0.3)
    tex[ring & (haz == 0)] = ramp("grey", 0.9)
    tex[d < 14] = ramp("red", 0.4)
    return tex


def flat_marbflr(seed: int = 53) -> np.ndarray:
    n = vnoise(FS, FS, 16, seed, 4)
    t = 0.45 + (n - 0.5) * 0.4
    t[np.abs(np.sin(n * 20)) < 0.1] += 0.25
    for y0 in range(0, FS, 32):
        for x0 in range(0, FS, 32):
            bevel_rect(t, y0, x0, y0 + 32, x0 + 32, -0.1, 0.2)
    return shade("rust", t)


def flat_metal(seed: int = 54) -> np.ndarray:
    t = 0.6 + (grain(FS, FS, seed) - 0.5) * 0.08 + (vnoise(FS, FS, 16, seed) - 0.5) * 0.12
    bevel_rect(t, 0, 0, 64, 64, -0.15, 0.25)
    for y, x in ((6, 6), (6, 57), (57, 6), (57, 57)):
        rivet(t, y, x)
    return shade("steel", t)


def flat_hellrock(seed: int = 55) -> np.ndarray:
    t = 0.5 + (vnoise(FS, FS, 16, seed, 4) - 0.5) * 0.45
    return shade("red", 0.35 + t * 0.6)


# --- sky -----------------------------------------------------------------------------------------------------------

def make_sky(seed: int = 60) -> np.ndarray:
    h, w = 128, 256
    ys = np.arange(h)[:, None] * np.ones((1, w))
    clouds = vnoise(h, w, 32, seed, 4)
    t = 0.85 - ys / h * 0.55 + (clouds - 0.5) * 0.25
    sky = shade("red", np.clip(t, 0.45, 0.95))
    glow = ys > 60
    sky[glow] = shade("fire", np.clip(0.95 - (ys[glow] - 60) / 120 + (clouds[glow] - 0.5) * 0.1, 0.55, 0.97))
    x = np.arange(w)
    ridge1 = 78 + 14 * np.sin(x / w * 2 * np.pi * 3 + 1.0) + 8 * np.sin(x / w * 2 * np.pi * 7 + 2.0) \
        + (vnoise(1, w, 8, seed + 1, 3)[0] - 0.5) * 18
    ridge2 = 96 + 8 * np.sin(x / w * 2 * np.pi * 5 + 0.3) + (vnoise(1, w, 4, seed + 2, 3)[0] - 0.5) * 14
    rock = vnoise(h, w, 8, seed + 3, 4)
    far = ys > ridge1[None, :]
    sky[far] = shade("rust", 0.5 + rock[far] * 0.3 + (ys[far] - ridge1[None, :].repeat(h, 0)[far]) / 200.0)
    near = ys > ridge2[None, :]
    sky[near] = shade("brown", 0.55 + rock[near] * 0.35)
    edge = np.abs(ys - ridge1[None, :]) < 1.0
    sky[edge] = ramp("rust", 0.3)
    return sky


# --- the registry --------------------------------------------------------------------------------------------------

def _wall_defs() -> list[tuple[str, list[np.ndarray]]]:
    return [
        ("TECHTAN", [tex_techtan()]), ("TECHGRY", [tex_techgrey()]), ("STONE", [tex_stone()]),
        ("BROWN", [tex_brown()]), ("BRNPANEL", [tex_brownpanel()]),
        ("COMPUTE", [tex_compute(f) for f in range(3)]), ("COMPSTA", [tex_compsta(f) for f in range(2)]),
        ("METAL", [tex_metal()]), ("SUPPORT", [tex_support()]), ("DOOR", [tex_door()]),
        ("DOORBLU", [tex_door(30, "blue")]), ("DOORYEL", [tex_door(31, "fire")]), ("DOORRED", [tex_door(32, "red")]),
        ("DOORTRAK", [tex_doortrak()]), ("HAZARD", [tex_hazard()]), ("SWOFF", [tex_switch(False)]),
        ("SWON", [tex_switch(True)]), ("EXITOFF", [tex_switch(False, True)]), ("EXITON", [tex_switch(True, True)]),
        ("EXITSIGN", [tex_exitsign()]), ("LITE", [tex_lite()]), ("MARBLE", [tex_marble()]), ("SKIN", [tex_skin()]),
        ("GSTONE", [tex_gstone()]), ("SLIME", [tex_slime()]), ("PIPES", [tex_pipes()]), ("STEP", [tex_step()]),
        ("TEKWALL", [tex_tekwall()]), ("LIFT", [tex_lift()]), ("REDBRICK", [tex_redbrick()]),
        ("LAVAFALL", [tex_lavafall(f) for f in range(4)]), ("SKULLWAL", [tex_skull_wall()]),
    ]


def _flat_defs() -> list[tuple[str, list[np.ndarray]]]:
    return [
        ("FLOOR4", [flat_floor4()]), ("FLOOR5", [flat_floor5()]), ("COMPFLR", [flat_compflr()]),
        ("CEIL3", [flat_ceil3()]), ("TLITE", [flat_tlite()]), ("LIGHTSQ", [flat_lightsq()]),
        ("NUKAGE", [flat_nukage(f) for f in range(3)]), ("BLOOD", [flat_blood(f) for f in range(3)]),
        ("LAVA", [flat_lava(f) for f in range(3)]), ("ROCK", [flat_rock()]), ("DIRT", [flat_dirt()]),
        ("STEPTOP", [flat_steptop()]), ("EXITPAD", [flat_exitpad()]), ("MARBFLR", [flat_marbflr()]),
        ("METAL", [flat_metal()]), ("HELLROCK", [flat_hellrock()]),
    ]


class TextureSet:
    """Every texture in one flat atlas (a pixel gather per screen pixel): walls, then flats, then the sky.
    Animated textures have several frames; ``frame_ids(tick)`` maps a texture id to its current frame."""

    def __init__(self) -> None:
        walls = _wall_defs()
        flats = _flat_defs()
        self.wall_ids: dict[str, int] = {}
        self.flat_ids: dict[str, int] = {}
        wall_frames: list[np.ndarray] = []
        flat_frames: list[np.ndarray] = []
        self.wall_anim: list[tuple[int, int]] = []  # per base id: (first frame, count)
        self.flat_anim: list[tuple[int, int]] = []
        for name, frames in walls:
            self.wall_ids[name] = len(self.wall_anim)
            self.wall_anim.append((len(wall_frames), len(frames)))
            wall_frames.extend(frames)
        for name, frames in flats:
            self.flat_ids[name] = len(self.flat_anim)
            self.flat_anim.append((len(flat_frames), len(frames)))
            flat_frames.extend(frames)
        self.walls = np.stack(wall_frames).astype(np.uint8)
        self.flats = np.stack(flat_frames).astype(np.uint8)
        self.sky = make_sky()
        self.wall_base = 0
        self.flat_base = self.walls.size
        self.sky_base = self.flat_base + self.flats.size
        self.atlas = np.concatenate([self.walls.ravel(), self.flats.ravel(), self.sky.ravel()])
        self._wall_first = np.array([a for a, _ in self.wall_anim])
        self._wall_count = np.array([c for _, c in self.wall_anim])
        self._flat_first = np.array([a for a, _ in self.flat_anim])
        self._flat_count = np.array([c for _, c in self.flat_anim])

    def wall_frames(self, tick: int) -> np.ndarray:
        """Base wall id -> atlas offset of its current frame (speeds: 8 tics per frame)."""
        f = self._wall_first + (tick // 8) % self._wall_count
        return (f * (TW * TH)).astype(np.int32)

    def flat_frames(self, tick: int) -> np.ndarray:
        f = self._flat_first + (tick // 8) % self._flat_count
        return (self.flat_base + f * (FS * FS)).astype(np.int32)


_TEXTURES: TextureSet | None = None


def textures() -> TextureSet:
    global _TEXTURES
    if _TEXTURES is None:
        _TEXTURES = TextureSet()
    return _TEXTURES
