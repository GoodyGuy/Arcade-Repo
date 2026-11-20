"""What Storm Drop draws: the instance rows for build pieces, trees and other props, loot, chests, the players'
chunky bodies (worked out for everyone at once with numpy), the drop ship, shots, sparks and the storm wall.

``Scene`` keeps the renderer's standing batches in step with a match (a piece placed or broken rewrites one row) and
hands back each frame's moving things."""

from __future__ import annotations

import math

import numpy as np

from .build import RAMP_DX, SLAB, Piece
from .data import (AMMO_COLORS, BRICK, CANOPIES, CELL, FLOOR, GRADES, LEVEL, MAT_COLORS, METAL, OUTFITS, RAMP, ROOF,
                   SEA, SHIP_ALT, SIZE, SUPPLIES, WALLX, WALLZ, WEAPONS, WOOD)
from .render import (ADD, BALL, BARK, BEAM, BOX, CANOPY, CARTOP, CHESTP, CLEAR, CONE, CRATE, CYL, DISC, DOME,
                     ENVELOPE, F32, FACE, FADE, FLAT, GHOST, GLOW, INST, LEAF, PYR, QUAD, RIBS, RING, ROCK, SHINGLE,
                     SOLID, SPIN, STORM, STRIPES, SUIT, TUBE, WATER, Renderer)


def rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


def inst(X, Y, Z, T, col, col2=(0.0, 0.0, 0.0), pat: float = FLAT, par=(0.0, 0.0, 0.0, 0.0),  # noqa: ANN001
         alpha: float = 1.0) -> list:
    """One instance: its three axes (each already as long as that half size), its centre, colours and pattern."""
    return [X[0], Y[0], Z[0], T[0], X[1], Y[1], Z[1], T[1], X[2], Y[2], Z[2], T[2],
            col[0], col[1], col[2], alpha, col2[0], col2[1], col2[2], pat, par[0], par[1], par[2], par[3]]


def box(cx: float, cy: float, cz: float, hx: float, hy: float, hz: float, col, col2=(0.0, 0.0, 0.0),  # noqa: ANN001
        pat: float = FLAT, par=(0.0, 0.0, 0.0, 0.0), yaw: float = 0.0, alpha: float = 1.0) -> list:
    if yaw:
        c, s = math.cos(yaw), math.sin(yaw)
        return inst((c * hx, 0.0, -s * hx), (0.0, hy, 0.0), (s * hz, 0.0, c * hz), (cx, cy, cz), col, col2, pat, par,
                    alpha)
    return [hx, 0.0, 0.0, cx, 0.0, hy, 0.0, cy, 0.0, 0.0, hz, cz,
            col[0], col[1], col[2], alpha, col2[0], col2[1], col2[2], pat, par[0], par[1], par[2], par[3]]


MAT_COL = {WOOD: ((0.84, 0.61, 0.36), (0.50, 0.33, 0.19)), BRICK: ((0.82, 0.45, 0.35), (0.94, 0.89, 0.80)),
           METAL: ((0.68, 0.76, 0.84), (0.38, 0.46, 0.56))}
ROOF_COL = {WOOD: ((0.95, 0.43, 0.25), (0.60, 0.20, 0.14)), BRICK: ((0.34, 0.56, 0.70), (0.20, 0.34, 0.48)),
            METAL: ((0.78, 0.84, 0.90), (0.50, 0.57, 0.66))}
CLASS_COL = {"close": (0.96, 0.56, 0.20), "mid": (0.36, 0.42, 0.52), "long": (0.22, 0.52, 0.38),
             "blast": (0.88, 0.26, 0.26), "tool": (0.62, 0.64, 0.68)}
GUN_LEN = {"pepper": 0.15, "buzzer": 0.21, "thumper": 0.33, "ranger": 0.36, "hunter": 0.42, "longshot": 0.52,
           "boomtube": 0.46, "lobber": 0.30}
GRADE_COL = [rgb(c) for _n, c in GRADES]
CAR_COLS = ((0.93, 0.26, 0.24), (0.98, 0.80, 0.22), (0.20, 0.72, 0.74), (0.95, 0.95, 0.92), (0.98, 0.55, 0.20),
            (0.42, 0.52, 0.92))
BOX_COLS = ((0.86, 0.30, 0.26), (0.24, 0.50, 0.82), (0.26, 0.66, 0.44), (0.95, 0.60, 0.20), (0.56, 0.40, 0.76))
_HATS = (  # style: (offset, half size) for the hat's two boxes, from the neck
    (((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0))),
    (((0, 0.445, 0), (0.225, 0.05, 0.225)), ((0, 0.405, 0.29), (0.2, 0.025, 0.12))),
    (((0, 0.43, 0), (0.235, 0.075, 0.225)), ((0, 0.555, 0), (0.075, 0.06, 0.075))),
    (((0, 0.385, -0.02), (0.245, 0.105, 0.24)), ((0, 0.265, 0.218), (0.205, 0.05, 0.03))),
    (((0, 0.46, 0), (0.2, 0.055, 0.2)), ((0, 0.55, 0), (0.06, 0.05, 0.06))),
    (((0, 0.335, 0), (0.24, 0.045, 0.23)), ((0, 0.36, 0.26), (0.18, 0.02, 0.1))),
)


def piece_row(p: Piece) -> tuple[int, list]:
    """(mesh, row) of a build piece."""
    k, mat = p.kind, p.mat
    col, col2 = MAT_COL[mat]
    if p.owner >= 0:
        col = (min(col[0] * 1.06, 1.0), min(col[1] * 1.06, 1.0), min(col[2] * 1.04, 1.0))
    # how whole it is next to what it would be by now if nobody had hit it (a piece still going up is not damaged)
    at = p.hit_t if p.hit_t > p.t0 else p.t0
    due = p.hp_full - p.rate * max(p.t0 + p.dur - at, 0.0)
    par = (min(max(p.hp / max(due, 1.0), 0.0), 1.0), p.t0, p.dur, float(p.var))
    pat = 1.0 + mat
    if k == WALLX:
        h = (p.top - p.base) * 0.5
        return BOX, box((p.gx + 0.5) * CELL, p.base + h, p.gz * CELL, CELL / 2, h, 0.11, col, col2, pat, par,
                        alpha=p.hit_t)
    if k == WALLZ:
        h = (p.top - p.base) * 0.5
        return BOX, box(p.gx * CELL, p.base + h, (p.gz + 0.5) * CELL, 0.11, h, CELL / 2, col, col2, pat, par,
                        alpha=p.hit_t)
    cx, cz = (p.gx + 0.5) * CELL, (p.gz + 0.5) * CELL
    if k == FLOOR:
        return BOX, box(cx, p.gy * LEVEL + SLAB - 0.09, cz, CELL / 2, 0.09, CELL / 2, col, col2, pat,
                        (par[0], par[1], par[2], 0.0), alpha=p.hit_t)
    if k == RAMP:
        dx, dz = RAMP_DX[p.dir]
        A = (dx * 0.8, 0.6, dz * 0.8)
        Nn = (-dx * 0.6, 0.8, -dz * 0.6)
        T = (cx - Nn[0] * 0.09, p.gy * LEVEL + 1.5 + SLAB - Nn[1] * 0.09, cz - Nn[2] * 0.09)
        return BOX, inst((dz * 2.0, 0.0, -dx * 2.0), (Nn[0] * 0.09, Nn[1] * 0.09, Nn[2] * 0.09),
                         (A[0] * 2.5, A[1] * 2.5, A[2] * 2.5), T, col, col2, pat, (par[0], par[1], par[2], 2.0),
                         alpha=p.hit_t)
    col, col2 = ROOF_COL[mat]
    return PYR, box(cx, p.gy * LEVEL + 0.8, cz, CELL / 2, 0.8, CELL / 2, col, col2, SHINGLE,
                    (par[0], par[1], par[2], 0.0), alpha=p.hit_t)


def ghost_row(kind: int, gx: int, gy: int, gz: int, d: int, ok: bool, base: float | None = None) -> tuple[int, list]:
    col = (0.35, 0.75, 1.0) if ok else (1.0, 0.3, 0.3)
    a = 0.9
    if kind == WALLX:
        b = gy * LEVEL if base is None else base
        h = (gy * LEVEL + LEVEL - b) * 0.5
        return BOX, box((gx + 0.5) * CELL, b + h, gz * CELL, CELL / 2, h, 0.12, col, pat=GHOST, alpha=a)
    if kind == WALLZ:
        b = gy * LEVEL if base is None else base
        h = (gy * LEVEL + LEVEL - b) * 0.5
        return BOX, box(gx * CELL, b + h, (gz + 0.5) * CELL, 0.12, h, CELL / 2, col, pat=GHOST, alpha=a)
    cx, cz = (gx + 0.5) * CELL, (gz + 0.5) * CELL
    if kind == FLOOR:
        return BOX, box(cx, gy * LEVEL + SLAB - 0.08, cz, CELL / 2, 0.1, CELL / 2, col, pat=GHOST, alpha=a)
    if kind == RAMP:
        dx, dz = RAMP_DX[d]
        return BOX, inst((dz * 2.0, 0.0, -dx * 2.0), (-dx * 0.06, 0.08, -dz * 0.06), (dx * 2.0, 1.5, dz * 2.0),
                         (cx, gy * LEVEL + 1.5 + SLAB, cz), col, pat=GHOST, alpha=a)
    return PYR, box(cx, gy * LEVEL + 0.82, cz, CELL / 2, 0.82, CELL / 2, col, pat=GHOST, alpha=a)


def prop_rows(p) -> list[tuple[int, list]]:  # noqa: ANN001
    """The instances of a prop (a tree, a rock, a car ...)."""
    out: list[tuple[int, list]] = []
    x, y, z, s, sd, yaw = p.x, p.y, p.z, p.scale, p.seed, p.yaw
    c, sn = math.cos(yaw), math.sin(yaw)
    k = p.kind

    def at(lx: float, lz: float) -> tuple[float, float]:
        return x + c * lx + sn * lz, z - sn * lx + c * lz

    def B(mesh: int, lx: float, ly: float, lz: float, hx: float, hy: float, hz: float, col, col2=(0, 0, 0),  # noqa: ANN001, N802
          pat: float = FLAT, par=(0.0, 0.0, 0.0, 0.0), turn: float = 0.0) -> None:
        wx, wz = at(lx, lz)
        out.append((mesh, box(wx, y + ly, wz, hx, hy, hz, col, col2, pat, par, yaw + turn)))

    def lying(lx: float, ly: float, lz: float, r: float, half: float, col, col2=(0, 0, 0), pat: float = FLAT,  # noqa: ANN001
              along_x: bool = False, par=(0.0, 0.0, 0.0, 0.0)) -> None:
        wx, wz = at(lx, lz)
        ax = (c, 0.0, -sn) if along_x else (sn, 0.0, c)
        side = (sn, 0.0, c) if along_x else (c, 0.0, -sn)
        out.append((CYL, inst((side[0] * r, 0.0, side[2] * r), (ax[0] * half, 0.0, ax[2] * half), (0.0, r, 0.0),
                              (wx, y + ly, wz), col, col2, pat, par)))

    if k == "pine":
        g = (0.13 + 0.10 * sd, 0.50 + 0.14 * sd, 0.30 + 0.06 * sd)
        g2 = (g[0] * 0.6, g[1] * 0.72, g[2] * 0.7)
        B(CYL, 0, 1.0 * s, 0, 0.2 * s, 1.0 * s, 0.2 * s, (0.50, 0.34, 0.22), (0.34, 0.22, 0.15), BARK)
        for r, cy, hy in ((1.9, 2.7, 1.3), (1.5, 4.2, 1.2), (1.0, 5.6, 1.1)):
            B(CONE, 0, cy * s, 0, r * s, hy * s, r * s, g, g2, LEAF)
    elif k == "oak":
        if sd < 0.09:
            g, g2 = (0.99, 0.64, 0.76), (0.92, 0.46, 0.64)
        elif sd < 0.17:
            g, g2 = (0.98, 0.64, 0.22), (0.90, 0.44, 0.16)
        else:
            g = (0.34 + 0.14 * sd, 0.70 + 0.08 * sd, 0.24)
            g2 = (g[0] * 0.66, g[1] * 0.8, g[2] * 0.8)
        B(CYL, 0, 1.3 * s, 0, 0.28 * s, 1.3 * s, 0.28 * s, (0.52, 0.36, 0.24), (0.36, 0.24, 0.16), BARK)
        B(BALL, 0, 4.1 * s, 0, 2.1 * s, 1.75 * s, 2.1 * s, g, g2, LEAF)
        B(BALL, 1.2 * s, 3.4 * s, 0.5 * s, 1.45 * s, 1.2 * s, 1.45 * s, g, g2, LEAF)
        B(BALL, -1.0 * s, 3.5 * s, -0.8 * s, 1.5 * s, 1.25 * s, 1.5 * s, g, g2, LEAF)
    elif k == "palm":
        lx, lz = (sd - 0.5) * 1.4 * s, 0.5 * s
        top = (x + c * lx * 2 + sn * lz * 2, y + 5.2 * s, z - sn * lx * 2 + c * lz * 2)
        out.append((CYL, inst((0.17 * s, 0.0, 0.0), (c * lx + sn * lz, 2.6 * s, -sn * lx + c * lz), (0.0, 0.0, 0.17 * s),
                              ((x + top[0]) / 2, y + 2.6 * s, (z + top[2]) / 2), (0.70, 0.56, 0.38),
                              (0.52, 0.40, 0.28), BARK)))
        g, g2 = (0.30, 0.74, 0.30), (0.20, 0.58, 0.26)
        for i in range(6):
            a = yaw + i * math.tau / 6
            ca, sa = math.cos(a), math.sin(a)
            out.append((BALL, inst((ca * 0.34 * s, 0.0, -sa * 0.34 * s), (0.0, 0.07 * s, 0.0),
                                   (sa * 1.5 * s, -0.55 * s, ca * 1.5 * s),
                                   (top[0] + sa * 1.3 * s, top[1] - 0.35 * s, top[2] + ca * 1.3 * s), g, g2, LEAF)))
        out.append((BALL, box(top[0], top[1] - 0.2 * s, top[2], 0.3 * s, 0.26 * s, 0.3 * s, (0.50, 0.34, 0.20))))
    elif k == "rock":
        g = (0.64 - 0.08 * sd, 0.62 - 0.08 * sd, 0.60 - 0.04 * sd)
        g2 = (g[0] * 0.76, g[1] * 0.76, g[2] * 0.8)
        B(BALL, 0, 0.45 * s, 0, 1.35 * s, 1.05 * s, 1.15 * s, g, g2, ROCK)
        B(BALL, 0.8 * s, 0.2 * s, 0.5 * s, 0.75 * s, 0.6 * s, 0.7 * s, g, g2, ROCK, turn=1.0)
    elif k == "car":
        col = CAR_COLS[int(sd * 5.999)]
        B(BOX, 0, 0.66, 0, 1.0, 0.36, 2.2, col)
        B(BOX, 0, 1.24, -0.25, 0.88, 0.26, 1.1, col, (0.20, 0.30, 0.42), CARTOP)
        B(BOX, 0, 0.52, 2.22, 0.9, 0.12, 0.06, (0.9, 0.9, 0.92))
        B(BOX, 0, 0.52, -2.22, 0.9, 0.12, 0.06, (0.3, 0.3, 0.34))
        for wx in (-0.98, 0.98):
            for wz in (-1.35, 1.4):
                lying(wx, 0.38, wz, 0.38, 0.16, (0.16, 0.16, 0.19), along_x=True)
            B(BOX, wx * 0.62, 0.74, 2.21, 0.16, 0.09, 0.05, (1.0, 0.95, 0.6), par=(0, 0, 0.9, 0))
    elif k == "container":
        col = BOX_COLS[int(sd * 4.999)]
        B(BOX, 0, 1.3, 0, 1.25, 1.3, 3.1, col, (0.96, 0.94, 0.86), RIBS, (0.32, 0.5, 0.0, 0.0))
        B(BOX, 0, 1.3, 3.12, 1.1, 1.15, 0.03, (col[0] * 0.8, col[1] * 0.8, col[2] * 0.8))
    elif k == "tank":
        col = ((0.92, 0.94, 0.95), (0.86, 0.90, 0.80), (0.80, 0.90, 0.94))[int(sd * 2.999)]
        band = BOX_COLS[int(sd * 4.999)]
        B(CYL, 0, 1.8, 0, 1.7, 1.8, 1.7, col, band, STRIPES, (0.5, 0.25, 0.0, 0.0))
        B(DOME, 0, 3.6, 0, 1.7, 0.8, 1.7, col)
        B(CYL, 1.85, 1.8, 0, 0.12, 1.8, 0.12, (0.4, 0.46, 0.52))
        B(CYL, 0, 0.12, 0, 1.85, 0.12, 1.85, (0.46, 0.48, 0.52))
    elif k == "machine":
        col = ((0.24, 0.62, 0.66), (0.92, 0.72, 0.22), (0.56, 0.60, 0.68))[int(sd * 2.999)]
        B(BOX, 0, 0.8, 0, 1.1, 0.8, 1.1, col, (0.2, 0.24, 0.3), RIBS, (0.4, -0.6, 0.0, 0.0))
        B(BOX, 0.2, 1.9, -0.2, 0.7, 0.3, 0.7, (0.36, 0.40, 0.48))
        B(CYL, -0.6, 2.0, 0.5, 0.2, 0.6, 0.2, (0.3, 0.32, 0.38))
        B(BALL, 0.55, 2.3, 0.35, 0.14, 0.14, 0.14, (1.0, 0.4, 0.3) if sd < 0.5 else (0.4, 1.0, 0.5),
          par=(0, 0, 1.0, 0))
    elif k == "crate":
        B(BOX, 0, 0.65, 0, 0.7, 0.65, 0.7, (0.84, 0.64, 0.38), (0.56, 0.38, 0.20), CRATE)
    elif k == "barrel":
        col = BOX_COLS[int(sd * 4.999)]
        B(CYL, 0, 0.575, 0, 0.48, 0.575, 0.48, col, (0.24, 0.26, 0.3), STRIPES, (1.5, 0.0, 0.0, 0.0))
    elif k == "hay":
        B(BOX, 0, 0.6, 0, 0.9, 0.6, 1.3, (0.95, 0.80, 0.36), (0.80, 0.62, 0.24), STRIPES, (3.0, 0.0, 0.0, 0.0))
    elif k == "logs":
        for lx, ly in ((-0.56, 0.5), (0.56, 0.5), (0.0, 1.38)):
            lying(lx, ly * 0.72, 0.0, 0.5, 2.0, (0.56, 0.38, 0.24), (0.38, 0.25, 0.16), BARK)
            B(BOX, lx, ly * 0.72, 2.0, 0.3, 0.3, 0.02, (0.88, 0.72, 0.48))
    elif k == "stall":
        B(BOX, 0, 0.5, 0, 0.9, 0.5, 1.4, (0.84, 0.64, 0.38), (0.56, 0.38, 0.20), CRATE)
        for i, lz in enumerate((-0.9, -0.3, 0.3, 0.9)):
            B(BALL, 0.1, 1.15, lz, 0.24, 0.2, 0.24, BOX_COLS[(i + int(sd * 5)) % 5])
    elif k == "pillar":
        B(CYL, 0, 1.6, 0, 0.5, 1.6, 0.5, (0.78, 0.74, 0.68), (0.62, 0.58, 0.54), STRIPES, (2.5, 0.0, 0.0, 0.0))
        B(BOX, 0, 3.3, 0, 0.62, 0.12, 0.62, (0.72, 0.68, 0.62))
    return out


def decor_rows(d: tuple) -> list[tuple[int, list]]:
    kind, x, y, z, yaw, s, sd = d
    if kind == "bush":
        g = (0.26 + 0.16 * sd, 0.66 + 0.1 * sd, 0.28)
        return [(BALL, box(x, y + 0.3 * s, z, 0.8 * s, 0.55 * s, 0.75 * s, g, (g[0] * 0.7, g[1] * 0.8, g[2] * 0.8),
                           LEAF, yaw=yaw))]
    if kind == "flower":
        col = ((1.0, 0.86, 0.3), (1.0, 0.5, 0.6), (0.96, 0.96, 1.0), (0.7, 0.56, 1.0))[int(sd * 3.999)]
        return [(BALL, box(x, y + 0.3 * s, z, 0.13, 0.11, 0.13, col, par=(0, 0, 0.3, 0))),
                (CONE, box(x, y + 0.13 * s, z, 0.06, 0.16 * s, 0.06, (0.3, 0.66, 0.3)))]
    if kind == "tuft":
        return [(CONE, box(x, y + 0.22 * s, z, 0.2 * s, 0.3 * s, 0.2 * s, (0.36, 0.72, 0.3)))]
    if kind == "post":
        h = (s - y) * 0.5
        return [(CYL, box(x, y + h, z, sd * 0.6, h, sd * 0.6, (0.50, 0.36, 0.24), (0.36, 0.25, 0.17), BARK))]
    if kind == "lantern":
        return [(CYL, box(x, y + 0.6, z, 0.16, 0.6, 0.16, (0.3, 0.32, 0.38))),
                (BALL, box(x, y + 1.6, z, 0.6, 0.6, 0.6, (1.0, 0.9, 0.5), par=(0, 0, 1.0, 0)))]
    return []


def item_rows(it) -> tuple[list[list], list]:  # noqa: ANN001
    """(spinning boxes, the glow on the ground) of something lying about."""
    x, y, z = it.x, it.y + 0.55, it.z
    P = float(SPIN)
    rows = []

    def part(ox: float, oy: float, oz: float, hx: float, hy: float, hz: float, col, emis: float = 0.2) -> None:  # noqa: ANN001
        rows.append(box(x, y + oy, z, hx, hy, hz, col, pat=P, par=(ox, oz, emis, 0.0)))

    if it.kind == "weapon":
        w = WEAPONS[it.id]
        col = CLASS_COL[w.cls]
        g = GRADE_COL[it.grade]
        ln = GUN_LEN.get(it.id, 0.3) * 1.25
        fat = 0.1 if w.cls == "blast" else 0.06
        part(0, 0.04, 0.0, fat, fat + 0.02, ln, col)
        part(0, -0.1, -ln * 0.45, 0.045, 0.11, 0.07, (0.2, 0.2, 0.24))
        part(0, 0.05, -ln * 0.9, 0.05, 0.07, 0.12, g, 0.5)
        glow = g
    elif it.kind == "supply":
        sup = SUPPLIES[it.id]
        col = rgb(sup.color)
        big = sup.amount >= 50
        part(0, 0, 0, 0.2 if big else 0.14, 0.26 if big else 0.18, 0.2 if big else 0.14, col, 0.4)
        part(0, 0.3 if big else 0.21, 0, 0.08, 0.05, 0.08, (0.95, 0.95, 0.98))
        glow = col
    elif it.kind == "ammo":
        col = rgb(AMMO_COLORS[it.id])
        part(0, -0.2, 0, 0.2, 0.13, 0.14, col, 0.25)
        part(0, -0.05, 0, 0.21, 0.03, 0.15, (0.25, 0.26, 0.3))
        glow = (col[0] * 0.5, col[1] * 0.5, col[2] * 0.5)
    else:
        col = rgb(MAT_COLORS[it.id])
        part(0, -0.2, 0, 0.26, 0.1, 0.26, col, 0.2)
        part(0.04, 0.0, 0.03, 0.26, 0.1, 0.26, (col[0] * 0.85, col[1] * 0.85, col[2] * 0.85), 0.2)
        glow = (col[0] * 0.5, col[1] * 0.5, col[2] * 0.5)
    disc = box(it.x, it.y + 0.05, it.z, 0.85, 1.0, 0.85, glow, pat=DISC, par=(0.1, 0.0, 1.0, 0.0), alpha=0.5)
    return rows, disc


def chest_rows(c) -> tuple[list[list], list | None]:  # noqa: ANN001
    cs, sn = math.cos(c.yaw), math.sin(c.yaw)
    if c.small:
        w, h, d, col, col2 = 0.42, 0.2, 0.28, (0.30, 0.52, 0.34), (0.94, 0.86, 0.42)
    else:
        w, h, d, col, col2 = 0.62, 0.27, 0.4, (0.56, 0.34, 0.18), (1.0, 0.80, 0.26)
    rows = [box(c.x, c.y + h, c.z, w, h, d, col, col2, CHESTP, (0, 0, 0.0 if c.opened else 0.5, 0), yaw=c.yaw)]
    lid_h = h * 0.45
    if not c.opened:
        rows.append(box(c.x, c.y + 2 * h + lid_h, c.z, w, lid_h, d, col, col2, CHESTP, (0, 0, 0.5, 0), yaw=c.yaw))
        glow = None if c.small else box(c.x, c.y + 0.06, c.z, 1.5, 1.0, 1.5, (1.0, 0.8, 0.3), pat=DISC,
                                        par=(0.1, 0.0, 1.0, 0.0), alpha=0.55)
        return rows, glow
    # open: the lid thrown back on its hinge
    X = (cs * w, 0.0, -sn * w)
    Y = (-sn * lid_h * 0.94, lid_h * 0.34, -cs * lid_h * 0.94)
    Z = (sn * d * 0.34, d * 0.94, cs * d * 0.34)
    T = (c.x - sn * (d + 0.02), c.y + 2 * h + d * 0.9, c.z - cs * (d + 0.02))
    rows.append(inst(X, Y, Z, T, col, col2, CHESTP))
    return rows, None


class _Fake:
    """A stand-in player for the locker and the home card: just what the body needs."""

    def __init__(self, outfit: int, x: float, y: float, z: float, yaw: float) -> None:
        self.outfit, self.canopy = outfit, 0
        self.x, self.y, self.z, self.yaw, self.pitch = x, y, z, yaw, 0.0
        self.state, self.alive, self.crouch, self.swim, self.on_ground = "ground", True, False, False, True
        self.walk, self.speed, self.swing_t, self.hurt_t, self.use_t, self.sprint = 0.0, 0.0, -9.0, -9.0, 0.0, False
        self.held_id = None
        self.sel = -1

    def held(self):  # noqa: ANN201
        return None


class Scene:
    def __init__(self, renderer: Renderer, island) -> None:  # noqa: ANN001
        self.r = renderer
        self.island = island
        self.match = None
        self.b: dict = {}
        o = OUTFITS
        self._suit = np.array([rgb(x.suit) for x in o], F32)
        self._trim = np.array([rgb(x.trim) for x in o], F32)
        self._skin = np.array([rgb(x.skin) for x in o], F32)
        self._pack = np.array([rgb(x.pack) for x in o], F32)
        self._hat = np.array([rgb(x.hat) for x in o], F32)
        self._hat_style = [x.hat_style for x in o]
        self._hat_o = np.array([[h[0][0], h[1][0]] for h in _HATS], F32)  # (style, 2, 3)
        self._hat_s = np.array([[h[0][1], h[1][1]] for h in _HATS], F32)
        # particles
        n = 700
        self.p_pos = np.zeros((n, 3), F32)
        self.p_vel = np.zeros((n, 3), F32)
        self.p_col = np.zeros((n, 3), F32)
        self.p_life = np.zeros(n, F32)
        self.p_life0 = np.ones(n, F32)
        self.p_size = np.zeros(n, F32)
        self.p_kind = np.zeros(n, np.int8)  # 0 debris (falls), 1 puff (drifts, see-through)
        self.p_n = 0
        self.booms: list[list] = []  # [x, y, z, r, age]
        self.rng = np.random.default_rng(5)
        self.ground_sent = False

    # --- standing things ----------------------------------------------------------------------------------------------

    def bind(self, match) -> None:  # noqa: ANN001
        """Start over for a new match (None: just the island, for the menus)."""
        r = self.r
        if not self.ground_sent:
            r.set_ground(self.island.terrain_mesh(), self.island.ground, SIZE)
            self.ground_sent = True
        r.drop_batches()
        self.match = match
        b = self.b = {}
        b["water"] = r.batch(QUAD, CLEAR, 4)
        b["water"].add(box(SIZE / 2, SEA, SIZE / 2, 6000.0, 1.0, 6000.0, (0.16, 0.66, 0.82), (0.10, 0.50, 0.78),
                           WATER, alpha=0.72))
        b["piece_box"] = r.batch(BOX, SOLID, 2048)
        b["piece_pyr"] = r.batch(PYR, SOLID, 512)
        for mesh, cap in ((BOX, 512), (CYL, 1024), (CONE, 2048), (BALL, 1024), (DOME, 16)):
            b[("prop", mesh)] = r.batch(mesh, SOLID, cap)
        b["loot"] = r.batch(BOX, SOLID, 1024)
        b["glow"] = r.batch(QUAD, ADD, 512)
        self.p_n = 0
        self.booms = []
        for d in self.island.decor:
            for mesh, row in decor_rows(d):
                b[("prop", mesh)].add(row)
        if match is None:
            from .build import Structures

            self._menu_structs = Structures(self.island)
            pieces = self._menu_structs.pieces.values()
            for p in self.island.props:
                p.alive, p.slots = True, []
                self._add_prop(p)
        else:
            self._menu_structs = None
            pieces = ()
            for p in self.island.props:
                p.slots = []
                self._add_prop(p)
            for c in match.chests:
                self._add_chest(c)
        for p in pieces:
            self._put_piece(p)
        if match is not None:
            self.sync()

    def _add_prop(self, p) -> None:  # noqa: ANN001
        for mesh, row in prop_rows(p):
            batch = self.b[("prop", mesh)]
            p.slots.append((batch, batch.add(row)))

    def _put_piece(self, p: Piece) -> None:
        mesh, row = piece_row(p)
        batch = self.b["piece_box" if mesh == BOX else "piece_pyr"]
        if p.slot is None:
            p.slot = (batch, batch.add(row))
        else:
            p.slot[0].set(p.slot[1], row)

    def _add_chest(self, c) -> None:  # noqa: ANN001
        if c.slot:
            for batch, i in c.slot:
                batch.free(i)
        rows, glow = chest_rows(c)
        c.slot = [(self.b["loot"], self.b["loot"].add(row)) for row in rows]
        if glow is not None:
            c.slot.append((self.b["glow"], self.b["glow"].add(glow)))

    def sync(self) -> None:
        """Bring the batches up to date with what the match changed."""
        m = self.match
        if m is None:
            return
        st = m.structs
        if st.gone:
            for p in st.gone:
                if p.slot is not None:
                    p.slot[0].free(p.slot[1])
                    p.slot = None
            st.gone.clear()
        if st.dirty:
            for p in st.dirty:
                if not p.dead:
                    self._put_piece(p)
            st.dirty.clear()
        if m.prop_events:
            for p in m.prop_events:
                if not p.alive and p.slots:
                    for batch, i in p.slots:
                        batch.free(i)
                    p.slots = []
            m.prop_events.clear()
        if m.item_events:
            loot, glow = self.b["loot"], self.b["glow"]
            for what, it in m.item_events:
                if what == "add":
                    if it.alive:
                        rows, disc = item_rows(it)
                        it.slot = [(loot, loot.add(row)) for row in rows]
                        it.slot.append((glow, glow.add(disc)))
                elif it.slot:
                    for batch, i in it.slot:
                        batch.free(i)
                    it.slot = None
            m.item_events.clear()
        if m.chest_events:
            for c in m.chest_events:
                self._add_chest(c)
            m.chest_events.clear()
        if m.fx:
            for e in m.fx:
                kind = e[0]
                if kind == "burst":
                    self.burst(e[1], e[2], e[3], e[4], e[5], e[6])
                elif kind == "poof":
                    self.burst(e[1], e[2], e[3], rgb(e[4]), e[5], e[6], size=0.16, life=0.9)
                    self.burst(e[1], e[2], e[3], (1.0, 1.0, 1.0), 10, 3.0, size=0.1, life=0.7)
                    self.booms.append([e[1], e[2], e[3], 1.6, 0.0, 1])
                elif kind == "boom":
                    self.booms.append([e[1], e[2], e[3], e[4], 0.0, 0])
                    self.burst(e[1], e[2], e[3], (1.0, 0.7, 0.25), 22, 9.0, size=0.16, life=0.8)
                    self.burst(e[1], e[2], e[3], (0.35, 0.33, 0.33), 14, 5.0, size=0.3, life=1.1, kind=1)
                elif kind == "smoke":
                    self.burst(e[1], e[2], e[3], (0.9, 0.88, 0.85), 1, 0.4, size=0.2, life=0.6, kind=1)
            m.fx.clear()

    # --- particles ------------------------------------------------------------------------------------------------------

    def burst(self, x: float, y: float, z: float, col, n: int, speed: float, size: float = 0.09,  # noqa: ANN001
              life: float = 0.6, kind: int = 0) -> None:
        cap = len(self.p_life)
        n = min(n, cap - self.p_n)
        if n <= 0:
            return
        i0, i1 = self.p_n, self.p_n + n
        rng = self.rng
        v = rng.normal(0.0, 1.0, (n, 3)).astype(F32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
        v *= rng.uniform(0.3, 1.0, (n, 1)).astype(F32) * speed
        if kind == 0:
            v[:, 1] = np.abs(v[:, 1]) * 0.9 + speed * 0.25
        else:
            v[:, 1] = np.abs(v[:, 1]) * 0.5 + 0.6
        self.p_pos[i0:i1] = (x, y, z)
        self.p_vel[i0:i1] = v
        self.p_col[i0:i1] = np.array(col, F32) * rng.uniform(0.8, 1.1, (n, 1)).astype(F32)
        self.p_life[i0:i1] = self.p_life0[i0:i1] = life * rng.uniform(0.6, 1.0, n).astype(F32)
        self.p_size[i0:i1] = size * rng.uniform(0.6, 1.2, n).astype(F32)
        self.p_kind[i0:i1] = kind
        self.p_n = i1

    def _particles(self, dt: float) -> tuple[np.ndarray | None, np.ndarray | None]:
        n = self.p_n
        if not n:
            return None, None
        life = self.p_life[:n]
        life -= dt
        keep = life > 0
        k = int(keep.sum())
        if k < n:
            for arr in (self.p_pos, self.p_vel, self.p_col, self.p_life, self.p_life0, self.p_size, self.p_kind):
                arr[:k] = arr[:n][keep]
            n = self.p_n = k
            if not n:
                return None, None
        kind = self.p_kind[:n]
        vel = self.p_vel[:n]
        vel[:, 1] -= np.where(kind == 0, 16.0, -0.4).astype(F32) * dt
        vel[kind == 1] *= max(1.0 - 2.0 * dt, 0.0)
        self.p_pos[:n] += vel * dt
        frac = self.p_life[:n] / self.p_life0[:n]
        rows = np.zeros((n, INST), F32)
        size = self.p_size[:n] * np.where(kind == 0, 0.4 + 0.6 * frac, 1.0 + (1.0 - frac) * 2.2)
        rows[:, 0] = rows[:, 5] = rows[:, 10] = size
        rows[:, 3], rows[:, 7], rows[:, 11] = self.p_pos[:n, 0], self.p_pos[:n, 1], self.p_pos[:n, 2]
        rows[:, 12:15] = self.p_col[:n]
        rows[:, 15] = np.where(kind == 0, 1.0, frac * 0.55)
        solid = rows[kind == 0]
        puffs = rows[kind == 1]
        if len(puffs):
            puffs[:, 19] = GLOW
            puffs[:, 20] = 0.6
        return (solid if len(solid) else None), (puffs if len(puffs) else None)

    # --- bodies ---------------------------------------------------------------------------------------------------------

    def bodies(self, actors: list, time: float) -> tuple[np.ndarray | None, list, list]:
        """(box rows of every body, dome rows of open canopies, shadow rows)."""
        n = len(actors)
        if not n:
            return None, [], []
        pos = np.empty((n, 3), F32)
        yaw = np.empty(n, F32)
        lean = np.zeros(n, F32)
        hip = np.full(n, 0.78, F32)
        ang = np.zeros((n, 6), F32)  # leg l, leg r, arm l, arm r, torso, head
        oi = np.empty(n, np.intp)
        flash = np.zeros(n, F32)
        gun_s = np.zeros((n, 2, 3), F32)
        gun_o = np.zeros((n, 2, 3), F32)
        gun_c = np.zeros((n, 2, 3), F32)
        domes, shadows = [], []
        isl = self.island
        for i, a in enumerate(actors):
            pos[i] = (a.x, a.y, a.z)
            yaw[i] = a.yaw
            oi[i] = a.outfit
            st = a.state
            if time - a.hurt_t < 0.12:
                flash[i] = 0.55
            if st == "fall":
                lean[i] = 1.25
                wob = math.sin(time * 9.0 + i) * 0.12
                ang[i] = (0.3 + wob, -0.15 - wob, -2.6, -2.75, 0.0, -0.9)
                continue
            if st == "glide":
                lean[i] = 0.12
                ang[i] = (0.2, -0.12, -2.95, -2.95, 0.0, -0.2)
                c1, c2 = rgb(CANOPIES[a.canopy % len(CANOPIES)][2]), rgb(CANOPIES[a.canopy % len(CANOPIES)][3])
                domes.append(box(a.x, a.y + 3.3, a.z, 2.5, 1.25, 1.8, c1, c2, CANOPY, yaw=a.yaw))
                continue
            ph = a.walk * 2.3
            amp = min(a.speed / 4.5, 1.0) * (0.8 if a.on_ground else 0.3)
            sw = math.sin(ph) * amp
            ll, lr, al, ar, tor = sw, -sw, -sw * 0.8, sw * 0.8, 0.0
            if not a.on_ground:
                ll, lr = 0.5, -0.35
            if a.sprint and a.speed > 6:
                lean[i] = 0.2
            if a.crouch:
                hip[i] = 0.5
                tor = 0.32
            if a.swim:
                lean[i] = 1.0
                ll, lr = math.sin(time * 6.0) * 0.4, -math.sin(time * 6.0) * 0.4
            s = a.held()
            if a.use_t > 0:
                al = ar = -2.0 + math.sin(time * 10.0) * 0.1
                if s is not None:
                    gun_s[i, 0] = (0.07, 0.1, 0.07)
                    gun_o[i, 0] = (0.0, -0.64, 0.06)
                    gun_c[i, 0] = rgb(SUPPLIES[s.id].color) if s.id in SUPPLIES else (1, 1, 1)
            elif s is None:  # the sledge
                t = (time - a.swing_t) / 0.34
                if 0.0 <= t < 1.0:
                    ar = -0.4 - 2.3 * math.sin(t * math.pi)
                else:
                    ar = ar * 0.5 - 0.35
                gun_s[i] = ((0.032, 0.42, 0.032), (0.11, 0.09, 0.2))
                gun_o[i] = ((0.0, -0.86, 0.02), (0.0, -1.24, 0.02))
                gun_c[i] = ((0.62, 0.44, 0.26), (0.62, 0.66, 0.72))
            elif s.id in WEAPONS:
                w = WEAPONS[s.id]
                aim = -(math.pi / 2 + a.pitch)
                ar, al = aim, aim * 0.94
                ln = GUN_LEN.get(s.id, 0.3)
                fat = 0.085 if w.cls == "blast" else 0.05
                gun_s[i] = ((fat, ln, fat + 0.02), (0.04, 0.09, 0.06))
                gun_o[i] = ((-0.06, -0.6 - ln * 0.4, 0.06), (-0.06, -0.52, -0.06))
                gun_c[i] = (CLASS_COL[w.cls], GRADE_COL[s.grade])
            else:
                gun_s[i, 0] = (0.07, 0.1, 0.07)
                gun_o[i, 0] = (0.0, -0.66, 0.06)
                gun_c[i, 0] = rgb(SUPPLIES[s.id].color)
                ar = -0.9
            ang[i] = (ll, lr, al, ar, tor, -a.pitch * 0.5)
            g = isl.height(a.x, a.z)
            if a.y - g < 0.25 and g > SEA:
                nx, ny, nz = isl.normal(a.x, a.z)
                ln2 = math.hypot(ny, nx) or 1.0
                X = (ny / ln2 * 0.62, -nx / ln2 * 0.62, 0.0)
                Z = (X[1] * nz - 0.0, 0.0 * nx - X[0] * nz, (X[0] * ny - X[1] * nx))
                shadows.append(inst(X, (nx, ny, nz), Z, (a.x, g + 0.05, a.z), (0.0, 0.0, 0.0), pat=DISC,
                                    par=(0.25, 0, 0, 0), alpha=0.38))
            elif a.on_ground and not a.swim:
                shadows.append(box(a.x, a.y + 0.03, a.z, 0.62, 1.0, 0.62, (0.0, 0.0, 0.0), pat=DISC,
                                   par=(0.25, 0, 0, 0), alpha=0.38))
        cy, sy = np.cos(yaw), np.sin(yaw)
        cl, sl = np.cos(lean), np.sin(lean)
        Rb = np.empty((n, 3, 3), F32)  # columns: right, up (leaned), forward (leaned)
        Rb[:, 0, 0], Rb[:, 1, 0], Rb[:, 2, 0] = cy, 0.0, -sy
        Rb[:, 0, 1], Rb[:, 1, 1], Rb[:, 2, 1] = sy * sl, cl, cy * sl
        Rb[:, 0, 2], Rb[:, 1, 2], Rb[:, 2, 2] = sy * cl, -sl, cy * cl
        up = np.array((0.0, 1.0, 0.0), F32)
        suit, trim, skin, pack, hat = (self._suit[oi], self._trim[oi], self._skin[oi], self._pack[oi],
                                       self._hat[oi])
        fl = flash[:, None]
        zeros = np.zeros(n, F32)
        out = []

        def part(P, O, S, th, col, col2=None, pat: float = FLAT, par0: float = 0.0) -> None:  # noqa: ANN001, N803
            c, s = np.cos(th), np.sin(th)
            L = np.zeros((n, 3, 3), F32)
            L[:, 0, 0] = S[..., 0]
            L[:, 1, 1], L[:, 2, 1] = c * S[..., 1], s * S[..., 1]
            L[:, 1, 2], L[:, 2, 2] = -s * S[..., 2], c * S[..., 2]
            cen = np.empty((n, 3), F32)
            cen[:, 0] = P[..., 0] + O[..., 0]
            cen[:, 1] = P[..., 1] + c * O[..., 1] - s * O[..., 2]
            cen[:, 2] = P[..., 2] + s * O[..., 1] + c * O[..., 2]
            W = Rb @ L
            cw = pos + up + (Rb @ (cen - up)[..., None])[..., 0]
            rows = np.zeros((n, INST), F32)
            rows[:, 0:12] = np.concatenate([W, cw[..., None]], axis=2).reshape(n, 12)
            rows[:, 12:15] = col + (1.0 - col) * fl
            rows[:, 15] = 1.0
            if col2 is not None:
                rows[:, 16:19] = col2
            rows[:, 19] = pat
            rows[:, 20] = par0
            rows[:, 22] = flash * 0.6
            out.append(rows)

        def v3(x, y, z) -> np.ndarray:  # noqa: ANN001
            a = np.empty((n, 3), F32)
            a[:, 0], a[:, 1], a[:, 2] = x, y, z
            return a

        legh = hip * 0.5
        leg_s = v3(0.12, legh, 0.13)
        leg_o = v3(0.0, -legh, 0.0)
        part(v3(-0.145, hip, 0.0), leg_o, leg_s, ang[:, 0], trim)
        part(v3(0.145, hip, 0.0), leg_o, leg_s, ang[:, 1], trim)
        part(v3(0.0, hip, 0.0), v3(0.0, 0.31, 0.0), v3(0.29, 0.31, 0.18), ang[:, 4], suit, trim, SUIT)
        tl_c, tl_s = np.cos(ang[:, 4]), np.sin(ang[:, 4])
        neck_y, neck_z = hip + 0.62 * tl_c, 0.62 * tl_s  # the torso's top after its lean
        style = np.array([self._hat_style[o] for o in oi], np.intp)
        part(v3(0.0, neck_y, neck_z), v3(0.0, 0.21, 0.0), v3(0.21, 0.21, 0.21), ang[:, 5], skin, hat, FACE)
        sh_y, sh_z = hip + 0.52 * tl_c, 0.52 * tl_s
        arm_s, arm_o = v3(0.095, 0.29, 0.105), v3(0.0, -0.25, 0.0)
        part(v3(-0.385, sh_y, sh_z), arm_o, arm_s, ang[:, 2], suit)
        part(v3(0.385, sh_y, sh_z), arm_o, arm_s, ang[:, 3], suit)
        part(v3(0.0, hip + 0.32 * tl_c, 0.32 * tl_s - 0.27), v3(0.0, 0.0, 0.0), v3(0.2, 0.24, 0.1), ang[:, 4], pack)
        for j in (0, 1):
            part(v3(0.0, neck_y, neck_z), self._hat_o[style, j], self._hat_s[style, j], ang[:, 5],
                 hat if j == 0 else np.where((style == 3)[:, None], np.array((0.12, 0.14, 0.2), F32), hat))
        for j in (0, 1):
            part(v3(0.385, sh_y, sh_z), gun_o[:, j], gun_s[:, j], ang[:, 3], gun_c[:, j])
        _ = zeros
        return np.concatenate(out), domes, shadows

    # --- the frame ----------------------------------------------------------------------------------------------------------

    def frame(self, dt: float, time: float, hide=None, preview=None, storm: bool = True,  # noqa: ANN001
              extra=None) -> tuple[dict, dict, dict]:  # noqa: ANN001
        """This frame's moving things: ({mesh: rows} solid, see-through, light)."""
        m = self.match
        boxes: list = []
        solid: dict = {}
        clear_q: list = []
        clear: dict = {}
        add: dict = {}
        add_box: list = []
        balls_add: list = []
        if m is not None:
            self.sync()
            actors = [a for a in m.actors if a.alive and a.state != "ship" and a is not hide]
            if extra:
                actors = actors + list(extra)
            body, domes, shadows = self.bodies(actors, time)
            if domes:
                solid[DOME] = np.array(domes, F32)
            clear_q += shadows
            # the ship
            if m.ship_s < m.ship_len:
                sx = m.ship_a[0] + m.ship_dir[0] * m.ship_s
                sz = m.ship_a[1] + m.ship_dir[1] * m.ship_s
                ship_boxes, ship_balls = ship_rows(sx, SHIP_ALT, sz, m.ship_yaw, time)
                boxes += ship_boxes
                solid[BALL] = np.array(ship_balls, F32)
            for p in m.projectiles:
                sp = math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz) or 1.0
                d = (p.vx / sp, p.vy / sp, p.vz / sp)
                X, Y = _across(d)
                if p.weapon.blast > 0:
                    boxes.append(inst(_mul(X, 0.09), _mul(Y, 0.09), _mul(d, 0.32), (p.x, p.y, p.z),
                                      (0.9, 0.3, 0.25), par=(0, 0, 0.4, 0)))
                    balls_add.append(box(p.x - d[0] * 0.4, p.y - d[1] * 0.4, p.z - d[2] * 0.4, 0.3, 0.3, 0.3,
                                         (1.0, 0.7, 0.3), pat=GLOW, par=(1.0, 0, 0, 0), alpha=0.9))
                else:
                    add_box.append(inst(_mul(X, 0.05), _mul(Y, 0.05), _mul(d, 2.6),
                                        (p.x - d[0] * 2.6, p.y - d[1] * 2.6, p.z - d[2] * 2.6), (1.0, 0.95, 0.7),
                                        pat=BEAM, alpha=0.9))
            for t in m.tracers:
                dx, dy, dz = t[3] - t[0], t[4] - t[1], t[5] - t[2]
                ln = math.sqrt(dx * dx + dy * dy + dz * dz)
                if ln < 0.5:
                    continue
                d = (dx / ln, dy / ln, dz / ln)
                X, Y = _across(d)
                k = t[6] / 0.07
                seg = min(ln, 9.0 + ln * 0.25) * 0.5  # the bright part runs out along the shot
                mid = min(ln - seg, ln * (1.0 - k) + seg) if ln > 2 * seg else ln * 0.5
                add_box.append(inst(_mul(X, 0.035), _mul(Y, 0.035), _mul(d, seg),
                                    (t[0] + d[0] * mid, t[1] + d[1] * mid, t[2] + d[2] * mid), (1.0, 0.92, 0.6),
                                    pat=BEAM, alpha=0.55 + 0.4 * k))
                if k > 0.6:
                    balls_add.append(box(t[0], t[1], t[2], 0.22, 0.22, 0.22, (1.0, 0.85, 0.45), pat=GLOW,
                                         par=(1.0, 0, 0, 0), alpha=0.9))
        else:
            body = None
            if extra:
                body, domes, shadows = self.bodies(list(extra), time)
                clear_q += shadows
        debris, puffs = self._particles(dt)
        parts = [np.array(boxes, F32)] if boxes else []
        if body is not None:
            parts.append(body)
        if debris is not None:
            parts.append(debris)
        if parts:
            solid[BOX] = np.concatenate(parts) if len(parts) > 1 else parts[0]
        if self.booms:
            for b in self.booms:
                b[4] += dt
                life = 0.45 if b[5] == 0 else 0.3
                k = b[4] / life
                if k < 1.0:
                    r = b[3] * (0.35 + 0.75 * k)
                    col = (1.0, 0.75 - 0.3 * k, 0.3) if b[5] == 0 else (1.0, 1.0, 1.0)
                    balls_add.append(box(b[0], b[1], b[2], r, r, r, col, pat=GLOW, par=(1.3, 0, 0, 0),
                                         alpha=(1.0 - k) * 0.95))
            self.booms = [b for b in self.booms if b[4] < 0.5]
        clear_box: list = []
        clear_pyr: list = []
        if preview is not None:
            mesh, row = preview
            (clear_box if mesh == BOX else clear_pyr).append(row)
        if clear_q:
            clear[QUAD] = np.array(clear_q, F32)
        if clear_box:
            clear[BOX] = np.array(clear_box, F32)
        if clear_pyr:
            clear[PYR] = np.array(clear_pyr, F32)
        if puffs is not None:
            clear[BALL] = puffs
        if m is not None and storm:
            s = m.storm
            if s["r"] < 520:
                clear[TUBE] = np.array([box(s["cx"], 60.0, s["cz"], s["r"], 90.0, s["r"], (0.50, 0.20, 0.86),
                                            (0.86, 0.50, 1.0), STORM, alpha=0.5)], F32)
        if add_box:
            add[BOX] = np.array(add_box, F32)
        if balls_add:
            add[BALL] = np.array(balls_add, F32)
        return solid, clear, add


def _mul(v: tuple, k: float) -> tuple:
    return v[0] * k, v[1] * k, v[2] * k


def _across(d: tuple) -> tuple[tuple, tuple]:
    """Two unit directions across ``d``."""
    if abs(d[1]) > 0.95:
        X = (1.0, 0.0, 0.0)
    else:
        ln = math.hypot(d[0], d[2])
        X = (d[2] / ln, 0.0, -d[0] / ln)
    Y = (d[1] * X[2] - d[2] * X[1], d[2] * X[0] - d[0] * X[2], d[0] * X[1] - d[1] * X[0])
    return X, Y


def ship_rows(x: float, y: float, z: float, yaw: float, time: float) -> tuple[list, list]:
    """The drop ship: a striped airship with a gondola (boxes, balls)."""
    c, s = math.cos(yaw), math.sin(yaw)
    bob = math.sin(time * 0.8) * 0.4

    def at(lx: float, ly: float, lz: float) -> tuple[float, float, float]:
        return x + c * lx + s * lz, y + ly + bob, z - s * lx + c * lz

    balls = [inst((c * 6.5, 0.0, -s * 6.5), (0.0, 5.6, 0.0), (s * 18.0, 0.0, c * 18.0), at(0, 9.0, 0),
                  (1.0, 0.56, 0.22), (1.0, 0.95, 0.80), ENVELOPE)]
    boxes = []

    def B(lx: float, ly: float, lz: float, hx: float, hy: float, hz: float, col, col2=(0, 0, 0), pat: float = FLAT,  # noqa: ANN001, N802
          par=(0.0, 0.0, 0.0, 0.0)) -> None:
        cx, cy, cz = at(lx, ly, lz)
        boxes.append(box(cx, cy, cz, hx, hy, hz, col, col2, pat, par, yaw))

    B(0, 1.2, 0, 2.3, 1.3, 6.5, (0.30, 0.44, 0.62), (0.96, 0.92, 0.80), RIBS, (0.5, 0.3, 0.0, 0.0))
    B(0, -0.2, 0, 2.5, 0.15, 6.8, (0.96, 0.80, 0.30))
    B(0, 9.0, -16.6, 0.15, 4.4, 1.8, (0.86, 0.40, 0.16))
    B(0, 9.0, -16.6, 4.4, 0.15, 1.8, (0.86, 0.40, 0.16))
    for lx in (-3.4, 3.4):
        B(lx, 1.6, -3.5, 0.7, 0.7, 1.4, (0.9, 0.9, 0.92))
        spin = time * 22.0
        cx, cy, cz = at(lx, 1.6, -5.1)
        boxes.append(inst((c * math.cos(spin) * 1.5, math.sin(spin) * 1.5, -s * math.cos(spin) * 1.5),
                          (-c * math.sin(spin) * 0.16, math.cos(spin) * 0.16, s * math.sin(spin) * 0.16),
                          (s * 0.05, 0.0, c * 0.05), (cx, cy, cz), (0.2, 0.2, 0.24)))
    for lz in (-4.0, 0.0, 4.0):
        for lx in (-1.9, 1.9):
            B(lx, 4.2, lz, 0.05, 2.0, 0.05, (0.3, 0.3, 0.34))
    return boxes, balls


__all__ = ["Scene", "_Fake", "ghost_row", "rgb", "FADE", "RING"]
