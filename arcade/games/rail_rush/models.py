"""Rail Rush 3D models, built from instances of the renderer's unit meshes (see ``render.py``): freight wagons and
locomotives, ramps and barriers, coins, pickups in their bubbles, the courier runner (animated every frame) and the
yard drone that chases. All original designs made in code."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtGui import QColor

from app.arcade.games.rail_rush.render import (BALL, BOX, CHEVRONS, COIN, CONE, CYL, DISC, F32, FADE, HAZARD, INST,
                                               MESHES, PANELS, PLANKS, PYR, QUAD, RIBS, SOFTRECT, WEDGE, WINDOWS)
from app.arcade.games.rail_rush.world import BARRIERS, CAR_GAP, CAR_LEN, RAMP_LEN, TRAIN_H, TRAIN_HALF_W

_RGB: dict = {}


def rgb(spec: str, k: float = 1.0) -> tuple[float, float, float]:
    """A colour as floats (``k`` > 1 lighter, < 1 darker)."""
    key = (spec, k)
    c = _RGB.get(key)
    if c is None:
        q = QColor(spec)
        c = _RGB[key] = (min(1.0, q.redF() * k), min(1.0, q.greenF() * k), min(1.0, q.blueF() * k))
    return c


def rot_x(a: float) -> tuple:
    c, s = math.cos(a), math.sin(a)
    return (1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)


def rot_y(a: float) -> tuple:
    c, s = math.cos(a), math.sin(a)
    return (c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)


def rot_z(a: float) -> tuple:
    c, s = math.cos(a), math.sin(a)
    return (c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)


def mul(a: tuple, b: tuple) -> tuple:
    return tuple(tuple(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] for j in range(3)) for i in range(3))


def app(r: tuple, v: tuple) -> tuple:
    return (r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2], r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2])


def add(a: tuple, b: tuple) -> tuple:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


ROT_Z90 = rot_z(math.pi / 2)
ROT_X180 = rot_x(math.pi)
FACE_BACK = rot_x(-math.pi / 2)  # a flat quad turned to face -z (toward the camera)


def row(c: tuple, h: tuple, col: tuple, col2: tuple = (0.0, 0.0, 0.0), pat: int = 0, px: float = 1.0,
        py: float = 9.0, emis: float = 0.0, alpha: float = 1.0, pw: float = 0.0, r: tuple | None = None) -> list:
    """One instance: centre ``c``, half sizes ``h``, rotation ``r`` (3 x 3 rows), colours, pattern and numbers."""
    hx, hy, hz = h
    if r is None:
        m = (hx, 0.0, 0.0, c[0], 0.0, hy, 0.0, c[1], 0.0, 0.0, hz, c[2])
    else:
        (a, b, d), (e, f, g), (i, j, k) = r
        m = (a * hx, b * hy, d * hz, c[0], e * hx, f * hy, g * hz, c[1], i * hx, j * hy, k * hz, c[2])
    return [*m, col[0], col[1], col[2], alpha, col2[0], col2[1], col2[2], float(pat), px, py, emis, pw]


class Batch:
    """Instances gathered per mesh, solid and see-through."""

    def __init__(self) -> None:
        self.rows = ([[] for _ in range(MESHES)], [[] for _ in range(MESHES)])
        self.blocks = ([[] for _ in range(MESHES)], [[] for _ in range(MESHES)])

    def add(self, mesh: int, r: list, clear: bool = False) -> None:
        self.rows[clear][mesh].append(r)

    def box(self, *a, clear: bool = False, **k) -> None:  # noqa: ANN002, ANN003
        self.rows[clear][BOX].append(row(*a, **k))

    def cyl(self, *a, clear: bool = False, **k) -> None:  # noqa: ANN002, ANN003
        self.rows[clear][CYL].append(row(*a, **k))

    def ball(self, *a, clear: bool = False, **k) -> None:  # noqa: ANN002, ANN003
        self.rows[clear][BALL].append(row(*a, **k))

    def cone(self, *a, clear: bool = False, **k) -> None:  # noqa: ANN002, ANN003
        self.rows[clear][CONE].append(row(*a, **k))

    def pyr(self, *a, clear: bool = False, **k) -> None:  # noqa: ANN002, ANN003
        self.rows[clear][PYR].append(row(*a, **k))

    def wedge(self, *a, clear: bool = False, **k) -> None:  # noqa: ANN002, ANN003
        self.rows[clear][WEDGE].append(row(*a, **k))

    def quad(self, *a, clear: bool = False, **k) -> None:  # noqa: ANN002, ANN003
        self.rows[clear][QUAD].append(row(*a, **k))

    def put(self, mesh: int, arr: np.ndarray, clear: bool = False) -> None:
        if len(arr):
            self.blocks[clear][mesh].append(arr)

    def put_model(self, model: tuple, x: float, y: float, z: float) -> None:
        """A model made by ``freeze`` moved to (x, y, z)."""
        for clear in (0, 1):
            for mesh, arr in model[clear].items():
                a = arr.copy()
                a[:, 3] += x
                a[:, 7] += y
                a[:, 11] += z
                self.blocks[clear][mesh].append(a)

    def arrays(self, clear: bool = False) -> dict:
        out = {}
        for m in range(MESHES):
            parts = list(self.blocks[clear][m])
            if self.rows[clear][m]:
                parts.append(np.asarray(self.rows[clear][m], F32))
            if parts:
                out[m] = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return out

    def freeze(self) -> tuple[dict, dict]:
        return self.arrays(False), self.arrays(True)

    def count(self) -> int:
        return sum(len(r) for side in self.rows for r in side) + sum(len(a) for side in self.blocks for b in side
                                                                     for a in b)


# --- wagons and locomotives ------------------------------------------------------------------------------------------

# parked trains (style 0..5): (kind, body, second colour); oncoming (6, 7): the locomotive's colours
LIVERIES = (
    ("boxcar", "#a8432e", "#e8dcc0"),
    ("container", "#2f6fb5", "#e0782c"),
    ("hopper", "#c9c3b5", "#3b6fa8"),
    ("boxcar", "#3d7d5a", "#f2d24a"),
    ("container", "#e0782c", "#35a07a"),
    ("hopper", "#d9b23a", "#5b6069"),
    ("loco", "#e9e4d8", "#d8392c"),
    ("loco", "#2b3a55", "#f2c12e"),
)
DARK = "#2a2c31"
STEEL = "#3a3d44"


def _undercarriage(b: Batch, z0: float, L: float) -> None:
    hw = TRAIN_HALF_W
    b.box((0, 0.86, z0 + L / 2), (hw - 0.04, 0.1, L / 2 - 0.05), rgb(DARK))
    for zb in (z0 + 2.3, z0 + L - 2.3):
        b.box((0, 0.52, zb), (0.92, 0.24, 1.05), rgb(STEEL))
        for dz in (-0.62, 0.62):
            for x in (-0.72, 0.72):
                b.cyl((x, 0.38, zb + dz), (0.36, 0.07, 0.36), rgb("#1c1d21"), r=ROT_Z90)
    for zc in (z0 - 0.2, z0 + L + 0.2):
        b.box((0, 0.86, zc), (0.16, 0.09, 0.22), rgb("#4a4d55"))


def _wagon(b: Batch, kind: str, z0: float, body: str, second: str, variant: int) -> None:
    hw, L = TRAIN_HALF_W, CAR_LEN
    zc = z0 + L / 2
    _undercarriage(b, z0, L)
    if kind == "boxcar":
        b.box((0, 2.24, zc), (hw, 1.26, L / 2), rgb(body), rgb(second), RIBS, px=0.62, py=0.62)
        b.box((0, 3.47, zc), (hw + 0.05, 0.05, L / 2 + 0.03), rgb(body, 0.7))
        b.box((0, 3.53, zc), (0.32, 0.03, L / 2 - 0.3), rgb("#7d858f"), rgb("#5b636d"), PLANKS, px=0.25, pw=1)
        for sg in (-1, 1):
            b.box((sg * (hw + 0.03), 2.0, zc), (0.04, 0.98, 1.15), rgb(body, 0.82), rgb(body, 0.6), PLANKS,
                  px=0.32, pw=0)
            b.box((sg * (hw + 0.07), 3.0, zc), (0.03, 0.05, 1.5), rgb("#8d939c"))
    elif kind == "container":
        b.box((0, 1.05, zc), (hw - 0.02, 0.1, L / 2), rgb(DARK))
        half = (L - 0.9) / 4
        for i, col in enumerate((body, second) if variant % 2 == 0 else (second, body)):
            cz = z0 + 0.3 + half + i * (2 * half + 0.3)
            b.box((0, 2.33, cz), (hw, 1.17, half), rgb(col), rgb(col, 0.75), RIBS, px=0.26, py=0.82)
            b.box((0, 3.49, cz), (hw - 0.02, 0.03, half - 0.02), rgb(col, 1.1))
            for sx in (-1, 1):
                for ez in (-1, 1):
                    b.box((sx * (hw - 0.06), 2.33, cz + ez * (half - 0.06)), (0.08, 1.18, 0.08), rgb(col, 0.6))
    else:  # covered hopper
        b.box((0, 2.6, zc), (hw, 0.9, L / 2), rgb(body), rgb(second), PANELS, px=1.45, py=0.5)
        b.box((0, 3.47, zc), (hw + 0.04, 0.05, L / 2 + 0.02), rgb(body, 0.85))
        for i in range(3):
            pz = z0 + L * (i + 0.5) / 3
            b.pyr((0, 1.33, pz), (hw - 0.12, 0.38, L / 6 - 0.15), rgb(body, 0.78), r=ROT_X180)
            b.box((0, 3.56, pz), (0.45, 0.05, 0.7), rgb(body, 0.7))
    # end ladders and a lamp on the last car
    for ez in (z0 + 0.02, z0 + L - 0.02):
        b.box((0.8, 2.2, ez), (0.04, 1.0, 0.04), rgb("#e8e3d6"))


def _loco(b: Batch, z0: float, body: str, second: str) -> None:
    """A diesel locomotive whose front (at ``z0``) faces the runner."""
    hw, L = TRAIN_HALF_W, CAR_LEN
    _undercarriage(b, z0, L)
    b.box((0, 2.24, z0 + 1.3), (hw, 1.26, 1.3), rgb(body), rgb(second), PANELS, px=9.0, py=-0.25)
    b.box((0, 2.24, z0 + 2.6 + (L - 2.6) / 2), (hw - 0.12, 1.26, (L - 2.6) / 2), rgb(body), rgb(second), PANELS,
          px=1.3, py=-0.25)
    b.box((0, 3.53, z0 + 6.5), (0.6, 0.06, 2.0), rgb("#5b636d"), rgb("#2a2c31"), RIBS, px=0.2, py=9)
    for sg in (-1, 1):
        b.box((sg * (hw - 0.06), 1.0, z0 + 7.3), (0.1, 0.05, 4.6), rgb("#d9dde3"))
    # the cab front: windscreen, warning chevrons, lights
    b.box((0, 2.95, z0 - 0.02), (0.98, 0.38, 0.03), rgb("#1c2a38"), emis=0.1)
    b.box((0, 2.95, z0 - 0.03), (0.05, 0.4, 0.035), rgb(body, 0.6))
    b.box((0, 1.6, z0 - 0.04), (hw - 0.02, 0.62, 0.04), rgb("#f2c12e"), rgb("#1b1d22"), CHEVRONS, px=1.1, py=1.0)
    for sx in (-0.78, 0.78):
        b.ball((sx, 1.62, z0 - 0.1), (0.14, 0.14, 0.08), rgb("#fff6d0"), emis=1.8)
        b.quad((sx, 1.62, z0 - 0.25), (1.25, 1.0, 1.25), rgb("#fff1b8"), pat=DISC, px=0.0, alpha=0.5, emis=1.2,
               r=FACE_BACK, clear=True)
        b.ball((sx * 1.15, 3.36, z0 - 0.05), (0.06, 0.06, 0.05), rgb("#ff4a3a"), emis=1.5)
    b.ball((0, 3.42, z0 - 0.06), (0.12, 0.12, 0.07), rgb("#fff6d0"), emis=1.8)
    b.box((0, 0.6, z0 - 0.18), (hw - 0.1, 0.18, 0.16), rgb("#3b3f46"), rgb("#f2c12e"), HAZARD, px=0.35)


_TRAINS: dict = {}


def train_model(style: int, cars: int, oncoming: bool) -> tuple:
    key = (style, cars, oncoming)
    m = _TRAINS.get(key)
    if m is None:
        b = Batch()
        for i in range(cars):
            z0 = i * (CAR_LEN + CAR_GAP)
            if oncoming and i == 0:
                kind, body, second = LIVERIES[6 + style % 2]
                _loco(b, z0, body, second)
            else:
                st = (style + i * 3) % 6 if oncoming else style
                kind, body, second = LIVERIES[st]
                if i % 2 == 1 and kind != "container":
                    body = second if kind == "boxcar" else body
                _wagon(b, kind, z0, body, second, i + style)
        total = cars * CAR_LEN + (cars - 1) * CAR_GAP
        b.quad((0, 0.3, total / 2), (1.55, 1.0, total / 2 + 0.6), (0, 0, 0), pat=SOFTRECT, px=0.55, py=0.96,
               alpha=0.5, clear=True)
        m = _TRAINS[key] = b.freeze()
    return m


# --- ramps and barriers ----------------------------------------------------------------------------------------------

_PROPS: dict = {}


def ramp_model() -> tuple:
    m = _PROPS.get("ramp")
    if m is None:
        b = Batch()
        hw = TRAIN_HALF_W
        b.wedge((0, TRAIN_H / 2, RAMP_LEN / 2), (hw, TRAIN_H / 2, RAMP_LEN / 2), rgb("#8e98a6"), rgb("#5c6a7d"),
                PLANKS, px=0.42, pw=0)
        for sg in (-1, 1):
            ang = math.atan2(TRAIN_H, RAMP_LEN)
            ln = math.hypot(TRAIN_H, RAMP_LEN) / 2
            b.box((sg * (hw - 0.05), TRAIN_H / 2 + 0.06, RAMP_LEN / 2), (0.07, 0.06, ln), rgb("#f2c12e"),
                  rgb("#1b1d22"), HAZARD, px=0.3, r=rot_x(-ang))
        b.box((0, 0.07, 0.12), (hw, 0.07, 0.14), rgb("#f2c12e"), rgb("#1b1d22"), HAZARD, px=0.35)
        b.quad((0, 0.3, RAMP_LEN / 2), (1.5, 1.0, RAMP_LEN / 2 + 0.3), (0, 0, 0), pat=SOFTRECT, px=0.5, py=0.9,
               alpha=0.4, clear=True)
        m = _PROPS["ramp"] = b.freeze()
    return m


def barrier_model(kind: str) -> tuple:
    m = _PROPS.get(kind)
    if m is not None:
        return m
    b = Batch()
    y0, y1, hw, depth = BARRIERS[kind]
    zc = depth / 2
    red, white, yellow, black = rgb("#e23b2e"), rgb("#f4f4f4"), rgb("#f2c12e"), rgb("#1b1d22")
    if kind == "low":  # a solid block: jump it
        b.box((0, 0.2, zc), (hw, 0.2, depth / 2), rgb("#9a9a94"), rgb("#7d7d77"), PANELS, px=0.55, py=9)
        b.box((0, 0.7, zc), (hw, 0.3, depth / 2 - 0.04), white, red, HAZARD, px=0.42)
        for sx in (-1, 1):
            b.ball((sx * (hw - 0.12), 1.04, zc), (0.07, 0.07, 0.07), rgb("#ffb21f"), emis=1.2)
    elif kind == "mid":  # a board on posts: jump it or roll under
        for sx in (-1, 1):
            b.cyl((sx * (hw - 0.08), y1 / 2, zc), (0.06, y1 / 2, 0.06), rgb("#d9dde3"))
            b.box((sx * (hw - 0.08), 0.04, zc), (0.16, 0.04, 0.2), rgb("#3b3f46"))
        b.box((0, (y0 + y1) / 2, zc), (hw, (y1 - y0) / 2, 0.06), yellow, black, HAZARD, px=0.4)
        b.ball((0, y1 + 0.06, zc), (0.08, 0.06, 0.08), rgb("#ff7a1a"), emis=1.2)
    elif kind == "high":  # a gantry with a sign: roll under
        for sx in (-1, 1):
            b.box((sx * (hw - 0.08), y1 / 2, zc), (0.08, y1 / 2, 0.08), rgb("#b9bfc7"))
            b.box((sx * (hw - 0.08), 0.04, zc), (0.18, 0.04, 0.2), rgb("#3b3f46"))
        b.box((0, y0 + 0.28, zc), (hw, 0.28, 0.06), white, red, HAZARD, px=0.4)
        b.box((0, y0 + 0.62 + 0.8, zc), (hw - 0.16, 0.8, 0.05), rgb("#2c6fd8"), white, CHEVRONS, px=1.6, py=-1.0)
        b.box((0, y1 - 0.06, zc), (hw, 0.06, 0.09), rgb("#5b6069"))
        for sx in (-0.6, 0.6):
            b.ball((sx, y1 + 0.02, zc), (0.08, 0.08, 0.08), rgb("#ffb21f"), emis=1.2)
    else:  # a signal mast between the rails: change lane
        b.box((0, 0.15, zc), (0.32, 0.15, 0.3), rgb("#6b717b"))
        b.cyl((0, 1.4, zc), (0.08, 1.3, 0.08), rgb("#3b3f46"))
        b.box((0, 2.75, zc), (0.3, 0.55, 0.16), rgb("#16181c"))
        for i, (col, on) in enumerate((("#ff3b30", 1.6), ("#ffd23f", 0.0), ("#3bff6a", 0.0))):
            b.ball((0, 3.1 - i * 0.34, zc - 0.17), (0.11, 0.11, 0.05), rgb(col, 1.0 if on else 0.35), emis=on)
        b.box((0, 1.6, zc - 0.09), (0.3, 0.07, 0.02), white, red, HAZARD, px=0.2)
    b.quad((0, 0.3, zc), (hw + 0.3, 1.0, 0.5), (0, 0, 0), pat=SOFTRECT, px=0.4, py=0.3, alpha=0.4, clear=True)
    m = _PROPS[kind] = b.freeze()
    return m


# --- coins and pickups -----------------------------------------------------------------------------------------------

def coins(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, t: float) -> np.ndarray:
    """Spinning coins standing up (one row each)."""
    n = len(xs)
    a = t * 4.0 + zs * 0.35
    c, s = np.cos(a), np.sin(a)
    r, th = 0.36, 0.06
    out = np.zeros((n, INST), F32)
    out[:, 0], out[:, 1], out[:, 3] = c * r, s * th, xs
    out[:, 6], out[:, 7] = -r, ys
    out[:, 8], out[:, 9], out[:, 11] = -s * r, c * th, zs
    out[:, 12:16] = (1.0, 0.8, 0.18, 1.0)
    out[:, 16:20] = (0.82, 0.5, 0.04, COIN)
    out[:, 20], out[:, 22] = 1.0, 0.3
    return out


TINTS = {"jetpack": "#ff9ad5", "sneakers": "#7cff4f", "magnet": "#ff8a6a", "double": "#c79bff", "mystery": "#ffcf7a",
         "key": "#ff6b8a"}


def pickup(b: Batch, kind: str, x: float, y: float, z: float, t: float) -> None:
    """A pickup turning inside its glowing bubble."""
    rr = rot_y(t * 2.0)
    s = 0.42

    def part(mesh: int, c: tuple, h: tuple, col: tuple, r: tuple | None = None, **k) -> None:  # noqa: ANN003
        rot = rr if r is None else mul(rr, r)
        b.add(mesh, row(add((x, y, z), app(rr, (c[0] * s, c[1] * s, c[2] * s))), (h[0] * s, h[1] * s, h[2] * s),
                        col, r=rot, **k))

    if kind == "jetpack":  # balloons
        for (cx, cy, cz), col in (((0, 0.45, 0), "#ff4f9a"), ((-0.42, 0.25, 0.1), "#3fc6ff"),
                                  ((0.42, 0.28, -0.1), "#ffd23f")):
            part(BALL, (cx, cy, cz), (0.34, 0.4, 0.34), rgb(col), emis=0.2)
        part(CYL, (0, -0.4, 0), (0.02, 0.5, 0.02), rgb("#ffffff"))
    elif kind == "sneakers":  # a spring boot
        part(BOX, (0, 0.1, 0.1), (0.3, 0.32, 0.55), rgb("#e8323c"))
        part(BOX, (0, -0.28, 0.1), (0.33, 0.08, 0.6), rgb("#ffffff"))
        part(CYL, (0, -0.62, 0.1), (0.2, 0.26, 0.2), rgb("#7cff4f"), col2=rgb("#2a8f2a"), pat=RIBS, px=0.15,
             emis=0.3)
    elif kind == "magnet":
        part(BOX, (0, -0.5, 0), (0.6, 0.18, 0.2), rgb("#e8323c"))
        for sx in (-1, 1):
            part(BOX, (sx * 0.45, 0.05, 0), (0.17, 0.4, 0.2), rgb("#e8323c"))
            part(BOX, (sx * 0.45, 0.58, 0), (0.18, 0.15, 0.21), rgb("#e9edf2"))
    elif kind == "double":  # a purple gem
        part(PYR, (0, 0.3, 0), (0.5, 0.45, 0.5), rgb("#b45cff"), emis=0.3)
        part(PYR, (0, -0.6, 0), (0.5, 0.45, 0.5), rgb("#7b3fd9"), r=ROT_X180, emis=0.3)
    elif kind == "mystery":  # a lost parcel
        part(BOX, (0, 0, 0), (0.6, 0.5, 0.5), rgb("#c58b4c"), col2=rgb("#9a6a36"), pat=PLANKS, px=0.5)
        part(BOX, (0, 0, 0), (0.12, 0.52, 0.52), rgb("#e8d6a0"))
        part(BOX, (0, 0.2, 0.505), (0.3, 0.12, 0.01), rgb("#ffffff"))
    else:  # a heart
        for sx in (-1, 1):
            part(BALL, (sx * 0.27, 0.25, 0), (0.36, 0.36, 0.25), rgb("#ff3b5c"), emis=0.3)
        part(PYR, (0, -0.25, 0), (0.6, 0.55, 0.24), rgb("#ff3b5c"), r=ROT_X180, emis=0.3)
    b.ball((x, y, z), (0.62, 0.62, 0.62), rgb(TINTS.get(kind, "#ffffff")), alpha=0.22, emis=0.8, clear=True)
    b.quad((x, 0.32 if y < 3 else TRAIN_H + 0.32, z), (0.6, 1.0, 0.6), (0, 0, 0), pat=DISC, px=0.1, alpha=0.3,
           clear=True)


# --- the runner ------------------------------------------------------------------------------------------------------

@dataclass
class Pose:
    mode: str = "run"  # run, jump, roll, fly, stumble, dead, caught, idle
    phase: float = 0.0  # run cycle (rad)
    lean: float = 0.0  # -1..1 into a lane change
    t: float = 0.0
    roll: float = 0.0  # 0..1 through a roll
    rising: bool = True
    yaw: float = 0.0  # 0: running away from the camera; pi: facing it
    fall: float = 0.0  # 0..1 falling over after a crash
    shield: bool = False
    springs: bool = False
    balloons: bool = False


def runner(b: Batch, ch: dict, pose: Pose, x: float, y: float, z: float) -> None:
    """The courier with the feet at (x, y, z)."""
    bld = ch["build"]
    skin, top, accent, pants = rgb(ch["skin"]), rgb(ch["top"]), rgb(ch["accent"]), rgb(ch["pants"])
    shoes, pack, hair = rgb(ch["shoes"]), rgb(ch["pack"]), rgb(ch["hair"])
    mode, ph, t = pose.mode, pose.phase, pose.t
    tilt = -pose.lean * 0.22
    if mode == "stumble":
        tilt += math.sin(t * 30) * 0.22
    R = mul(rot_y(pose.yaw), rot_z(tilt))
    P = (x, y, z)
    if mode == "roll":
        pv = (0.0, 0.5, 0.0)
        R = mul(R, rot_x(pose.roll * math.tau))
        P = add(add(P, pv), tuple(-v for v in app(R, pv)))
    elif mode == "dead":
        R = mul(R, rot_x(-1.4 * pose.fall))

    def put(mesh: int, c: tuple, h: tuple, col: tuple, r: tuple | None = None, **k) -> None:  # noqa: ANN003
        b.add(mesh, row(add(P, app(R, c)), h, col, r=R if r is None else mul(R, r), **k))

    # the pose: hip height, thigh / knee angles, upper-arm / elbow angles, forward lean
    hip_y, lean_f = 0.9, 0.0
    s = math.sin(ph)
    if mode == "run" or mode == "stumble":
        hip_y = 0.9 - 0.05 * abs(math.cos(ph))
        th = (-0.8 * s, 0.8 * s)
        kn = (0.25 + 1.4 * max(0.0, math.cos(ph)), 0.25 + 1.4 * max(0.0, -math.cos(ph)))
        arm = (0.75 * s, -0.75 * s)
        elb = (-1.3, -1.3)
        lean_f = 0.16
        if mode == "stumble":
            arm = (math.sin(t * 25) * 1.6 - 1.4, math.sin(t * 25 + 2) * 1.6 - 1.4)
            elb = (-0.3, -0.3)
    elif mode == "jump":
        th, kn = ((-1.25, -0.35), (1.7, 0.9)) if pose.rising else ((-0.7, 0.05), (1.1, 0.5))
        arm, elb = ((-2.5, -2.2), (-0.3, -0.3)) if pose.rising else ((-1.4, -1.1), (-0.6, -0.6))
        lean_f = 0.1
    elif mode == "roll":
        hip_y = 0.62
        th, kn, arm, elb, lean_f = (-2.1, -2.1), (2.5, 2.5), (-0.9, -0.9), (-2.0, -2.0), 0.6
    elif mode == "fly":
        sw = math.sin(t * 3.0) * 0.25
        th, kn, arm, elb = (0.1 + sw, 0.1 - sw), (0.35, 0.35), (-3.0, -3.0), (0.0, 0.0)
    elif mode == "dead":
        th, kn, arm, elb = (-0.25, 0.15), (0.2, 0.2), (-2.7, -2.5), (0.0, 0.0)
    elif mode == "caught":
        th, kn, arm, elb = (0.0, 0.0), (0.05, 0.05), (-2.9, -2.9), (-0.35, -0.35)
    else:  # idle
        hip_y = 0.9 + 0.008 * math.sin(t * 2.2)
        th, kn = (0.0, 0.0), (0.04, 0.04)
        arm, elb = (0.05 * math.sin(t * 2.2), -0.05 * math.sin(t * 2.2)), (-0.25, -0.25)
    # legs
    for i, sd in enumerate((-1, 1)):
        hip = (sd * 0.11 * bld, hip_y - 0.02, 0.0)
        rt = mul(rot_z(sd * 0.04), rot_x(th[i]))
        put(BOX, add(hip, app(rt, (0, -0.22, 0))), (0.085 * bld, 0.23, 0.09), pants, rt)
        knee = add(hip, app(rt, (0, -0.44, 0)))
        rs = mul(rt, rot_x(kn[i]))
        put(BOX, add(knee, app(rs, (0, -0.2, 0))), (0.075 * bld, 0.21, 0.08), pants, rs)
        ankle = add(knee, app(rs, (0, -0.4, 0)))
        put(BOX, add(ankle, app(rs, (0, -0.03, 0.06))), (0.085, 0.06, 0.15), shoes, rs)
        put(BOX, add(ankle, app(rs, (0, -0.08, 0.06))), (0.088, 0.02, 0.155), rgb("#f4f1ea"), rs)
        if pose.springs:
            put(BOX, add(ankle, app(rs, (0, -0.11, 0.06))), (0.09, 0.025, 0.16), rgb("#7cff4f"), rs, emis=0.9)
    # hips, torso, bag
    rtor = rot_x(lean_f)
    hip_c = (0.0, hip_y, 0.0)
    put(BOX, hip_c, (0.2 * bld, 0.09, 0.12), pants)
    put(BOX, add(hip_c, app(rtor, (0, 0.29, 0))), (0.22 * bld, 0.27, 0.14), top, rtor, col2=rgb(ch["top"], 0.72),
        pat=FADE)
    put(BOX, add(hip_c, app(rtor, (0, 0.08, 0.0))), (0.225 * bld, 0.05, 0.145), rgb(ch["top"], 0.6), rtor)
    put(BOX, add(hip_c, app(rtor, (0, 0.33, 0.001))), (0.03, 0.2, 0.146), accent, rtor)
    if not pose.balloons:
        put(BOX, add(hip_c, app(rtor, (0, 0.34, -0.22))), (0.19 * bld, 0.2, 0.09), pack, rtor)
        put(BOX, add(hip_c, app(rtor, (0, 0.5, -0.225))), (0.195 * bld, 0.06, 0.095), rgb(ch["pack"], 0.7), rtor)
        put(BOX, add(hip_c, app(rtor, (0, 0.3, -0.315))), (0.12, 0.035, 0.005), accent, rtor)
    # arms
    for i, sd in enumerate((-1, 1)):
        sh = add(hip_c, app(rtor, (sd * 0.27 * bld, 0.5, 0)))
        ra = mul(rtor, mul(rot_z(sd * (0.12 if mode != "fly" else -0.15)), rot_x(arm[i])))
        put(BOX, add(sh, app(ra, (0, -0.15, 0))), (0.07 * bld, 0.16, 0.075), top, ra)
        elbow = add(sh, app(ra, (0, -0.3, 0)))
        rf = mul(ra, rot_x(elb[i]))
        put(BOX, add(elbow, app(rf, (0, -0.13, 0))), (0.062, 0.14, 0.065), top, rf)
        put(BALL, add(elbow, app(rf, (0, -0.3, 0))), (0.07, 0.07, 0.07), skin)
    # head
    neck = add(hip_c, app(rtor, (0, 0.6, 0.0)))
    put(CYL, neck, (0.07, 0.06, 0.07), rgb(ch["skin"], 0.9))
    hc = add(neck, app(rtor, (0, 0.24, 0.02)))
    put(BALL, hc, (0.2, 0.2, 0.2), skin)
    for sd in (-1, 1):
        put(BALL, add(hc, (sd * 0.19, -0.01, 0)), (0.04, 0.055, 0.04), rgb(ch["skin"], 0.9))
        put(BALL, add(hc, (sd * 0.075, 0.03, 0.17)), (0.045, 0.05, 0.03), rgb("#ffffff"))
        put(BALL, add(hc, (sd * 0.075, 0.025, 0.195)), (0.024, 0.03, 0.012), rgb("#2a1a10"))
    put(BOX, add(hc, (0, -0.09, 0.185)), (0.05, 0.012, 0.01), rgb("#7a2a1a"))
    style = ch["style"]
    if style == "cap":
        put(BALL, add(hc, (0, 0.07, -0.005)), (0.21, 0.15, 0.21), hair)
        put(BOX, add(hc, (0, 0.06, -0.22)), (0.13, 0.015, 0.09), rgb(ch["hair"], 0.75))
        put(BALL, add(hc, (0, 0.215, 0)), (0.03, 0.02, 0.03), accent)
    elif style == "beanie":
        put(BALL, add(hc, (0, 0.06, 0)), (0.215, 0.19, 0.215), hair)
        put(CYL, add(hc, (0, 0.02, 0)), (0.218, 0.05, 0.218), rgb(ch["hair"], 0.75))
        put(BALL, add(hc, (0, 0.27, 0)), (0.07, 0.07, 0.07), accent)
    elif style == "ponytail":
        put(BALL, add(hc, (0, 0.035, -0.012)), (0.208, 0.19, 0.205), hair)
        sway = math.sin(ph) * 0.25 + pose.lean * 0.3
        put(BALL, add(hc, (0, -0.02, -0.25)), (0.07, 0.17, 0.07), hair, mul(rot_z(sway), rot_x(0.7)))
        put(BALL, add(hc, (0, 0.06, -0.19)), (0.05, 0.04, 0.04), accent)
    elif style == "puffs":
        put(BALL, add(hc, (0, 0.03, -0.01)), (0.208, 0.2, 0.208), hair)
        for sd in (-1, 1):
            put(BALL, add(hc, (sd * 0.15, 0.17, -0.03)), (0.12, 0.12, 0.12), hair)
            put(CYL, add(hc, (sd * 0.2, 0.0, 0)), (0.075, 0.03, 0.075), accent, ROT_Z90)
        put(BOX, add(hc, (0, 0.205, 0)), (0.2, 0.02, 0.03), rgb("#20242c"))
    else:  # mohawk
        put(BALL, add(hc, (0, 0.01, -0.01)), (0.203, 0.19, 0.203), rgb("#3a2a20"), alpha=1.0)
        put(BOX, add(hc, (0, 0.2, -0.01)), (0.04, 0.1, 0.17), hair)
        for k in range(4):
            put(PYR, add(hc, (0, 0.3, -0.13 + k * 0.085)), (0.04, 0.06, 0.04), hair)
    # extras
    if pose.balloons:
        hand = add(hip_c, (0, 1.9, 0))
        put(CYL, add(hand, (0, 0.2, 0)), (0.012, 0.22, 0.012), rgb("#ffffff"))
        for i, ((bx, by, bz), col) in enumerate((((0, 0.7, 0), "#ff4f9a"), ((-0.4, 0.6, 0.1), "#3fc6ff"),
                                                  ((0.4, 0.62, -0.05), "#ffd23f"), ((-0.16, 1.05, -0.12), "#7cff4f"),
                                                  ((0.22, 1.0, 0.12), "#b45cff"))):
            sw = math.sin(t * 2.0 + i) * 0.06
            put(BALL, add(hand, (bx + sw, by, bz)), (0.27, 0.32, 0.27), rgb(col), emis=0.15)
    if pose.shield:
        b.ball(add(P, (0, 0.95, 0)), (0.85, 1.12, 0.85), rgb("#7ffcff"), alpha=0.18 + 0.05 * math.sin(t * 6),
               emis=0.9, clear=True)


def shadow(b: Batch, x: float, y: float, z: float, size: float, strength: float = 0.45) -> None:
    b.quad((x, y + 0.04, z), (size, 1.0, size * 0.8), (0, 0, 0), pat=DISC, px=0.15, alpha=strength, clear=True)


# --- the yard drone --------------------------------------------------------------------------------------------------

def drone(b: Batch, x: float, y: float, z: float, t: float, beam_to: float | None = None, alarm: bool = False) -> None:
    """The yard security drone (seen from behind), its siren flashing when ``alarm``."""
    b.ball((x, y, z), (0.5, 0.2, 0.5), rgb("#39424e"))
    b.ball((x, y + 0.12, z), (0.27, 0.16, 0.27), rgb("#5f6b7b"))
    b.box((x, y - 0.03, z - 0.35), (0.2, 0.05, 0.12), rgb("#f2c12e"), rgb("#1b1d22"), HAZARD, px=0.12)
    flash = 0.5 + 0.5 * math.sin(t * 12) if alarm else 0.0
    b.ball((x - 0.12, y + 0.28, z), (0.07, 0.07, 0.07), rgb("#ff3b30"), emis=0.3 + 1.5 * flash)
    b.ball((x + 0.12, y + 0.28, z), (0.07, 0.07, 0.07), rgb("#3b7bff"), emis=0.3 + 1.5 * (1 - flash) * alarm)
    b.ball((x, y - 0.06, z + 0.44), (0.12, 0.12, 0.1), rgb("#111317"))
    b.ball((x, y - 0.06, z + 0.53), (0.05, 0.05, 0.03), rgb("#ff3b30"), emis=1.6)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        dx, dz = math.sin(a), math.cos(a)
        b.box((x + dx * 0.55, y + 0.02, z + dz * 0.55), (0.05, 0.035, 0.32), rgb("#2a2f37"), r=rot_y(a))
        cx, cz = x + dx * 0.88, z + dz * 0.88
        b.cyl((cx, y + 0.07, cz), (0.05, 0.05, 0.05), rgb("#1b1d22"))
        b.cyl((cx, y + 0.11, cz), (0.3, 0.008, 0.3), rgb("#d9e2ea"), alpha=0.32, clear=True)
        b.box((cx, y + 0.12, cz), (0.3, 0.008, 0.025), rgb("#9aa3ad"), r=rot_y(t * 40 + k))
    if beam_to is not None and y - beam_to > 0.3:
        hh = (y - beam_to) / 2
        b.cone((x, beam_to + hh, z), (1.0, hh, 1.0), rgb("#fff6b0"), alpha=0.16, emis=1.2, clear=True)


# --- scenery helpers (used by scene.py) ------------------------------------------------------------------------------

def pine(b: Batch, x: float, z: float, h: float, shade: float = 1.0) -> None:
    b.cyl((x, h * 0.12, z), (0.16, h * 0.12, 0.16), rgb("#6b4a2f"))
    b.cone((x, h * 0.42, z), (h * 0.3, h * 0.24, h * 0.3), rgb("#2f7d32", shade), rgb("#245e27", shade), FADE)
    b.cone((x, h * 0.72, z), (h * 0.22, h * 0.22, h * 0.22), rgb("#3f9a3a", shade), rgb("#2f7d32", shade), FADE)


def round_tree(b: Batch, x: float, z: float, h: float) -> None:
    b.cyl((x, h * 0.25, z), (0.14, h * 0.25, 0.14), rgb("#6b4a2f"))
    b.ball((x, h * 0.68, z), (h * 0.3, h * 0.28, h * 0.3), rgb("#4fae3c"), rgb("#2f7d32"), FADE)


def building(b: Batch, x: float, z: float, hw: float, h: float, hd: float, col: str, glass: str, seed: int,
             shop: str | None = None) -> None:
    """A block of flats or offices (x: the centre of its footprint)."""
    b.box((x, h / 2, z), (hw, h / 2, hd), rgb(col), rgb(glass), WINDOWS, px=2.1, py=3.0, pw=float(seed % 97))
    b.box((x, h + 0.25, z), (hw + 0.15, 0.25, hd + 0.15), rgb(col, 0.8))
    if shop:
        sg = -1 if x < 0 else 1
        b.box((x - sg * (hw + 0.01), 1.4, z), (0.02, 1.2, hd - 0.4), rgb("#26323f"), emis=0.15)
        b.box((x - sg * (hw + 0.5), 2.85, z), (0.5, 0.12, hd - 0.2), rgb(shop), rgb("#ffffff"), RIBS, px=0.5,
              r=rot_z(sg * 0.3))
