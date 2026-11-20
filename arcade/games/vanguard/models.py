"""Vanguard's shapes: everything is built from rounded boxes and cylinders, which the renderer draws as instances.

One instance is a row of 24 floats (in the renderer's axes: x, height, y):
``centre(3) bevel | right(3) width | up(3) height | forward(3) depth | r g b emissive | flag alpha mode pattern``.

``inst`` makes one row from map coordinates. The map's scenery (trees, lamps, rails) is lists of rows made once. A
``Model`` is a jointed set of parts (a hero, a weapon, the cart) posed every frame."""

from __future__ import annotations

import math

import numpy as np

F32 = np.float32
ROW = 24
TEAM = (-1, -1, -1)  # a part in this colour takes the viewer's ally / enemy colour and glows
VIEW_SCALE = 0.44  # the player's own weapon is drawn smaller and nearer than it is
VIEW_POS = (0.2, -0.17, 0.5)  # where it sits from the eye: right, up, forward
BODY, HEAD, ARM_L, ARM_R, LEG_L, LEG_R, GUN = range(7)


def inst(x: float, y: float, z: float, sx: float, sy: float, sz: float, col: tuple, yaw: float = 0.0,
         pitch: float = 0.0, roll: float = 0.0, emis: float = 0.0, rad: float = 0.05, mode: float = 0.0,
         flag: float = 0.0, alpha: float = 1.0, pat: float = 0.0) -> list:
    """A box centred at map (x, y), height z: ``sx`` wide, ``sy`` tall, ``sz`` deep along its facing ``yaw``."""
    ca, sa = math.cos(yaw), math.sin(yaw)
    f = (ca, 0.0, sa)
    r = (-sa, 0.0, ca)
    u = (0.0, 1.0, 0.0)
    if pitch:
        cp, sp = math.cos(pitch), math.sin(pitch)
        f, u = ((f[0] * cp + u[0] * sp, f[1] * cp + u[1] * sp, f[2] * cp + u[2] * sp),
                (u[0] * cp - f[0] * sp, u[1] * cp - f[1] * sp, u[2] * cp - f[2] * sp))
    if roll:
        cq, sq = math.cos(roll), math.sin(roll)
        r, u = ((r[0] * cq + u[0] * sq, r[1] * cq + u[1] * sq, r[2] * cq + u[2] * sq),
                (u[0] * cq - r[0] * sq, u[1] * cq - r[1] * sq, u[2] * cq - r[2] * sq))
    return [x, z, y, rad, r[0], r[1], r[2], sx, u[0], u[1], u[2], sy, f[0], f[1], f[2], sz,
            col[0] / 255, col[1] / 255, col[2] / 255, emis, flag, alpha, mode, pat]


def mirror_row(row: list, w: float) -> list:
    r = list(row)
    r[0] = w - r[0]
    r[4], r[8], r[12] = -r[4], -r[8], -r[12]
    return r


def shade(col: tuple, k: float) -> tuple:
    return tuple(max(min(int(c * k), 255), 0) for c in col)


# --- scenery ---------------------------------------------------------------------------------------------------------
def tree(x: float, y: float, z: float, h: float, col: tuple) -> list:
    rows = [inst(x, y, z + h * 0.3, 0.32, h * 0.6, 0.32, (124, 88, 62), mode=1)]
    for dx, dy, dz, s, k in ((0.0, 0.0, 0.0, 2.3, 1.0), (0.8, 0.3, -0.35, 1.5, 0.9), (-0.7, -0.4, -0.2, 1.7, 1.1),
                             (0.1, -0.7, 0.45, 1.4, 1.15), (-0.2, 0.7, 0.2, 1.3, 0.95)):
        rows.append(inst(x + dx, y + dy, z + h * 0.8 + dz, s, s * 0.86, s, shade(col, k), rad=9.0))
    return rows


def cactus(x: float, y: float, z: float, s: float = 1.0) -> list:
    g = (96, 150, 96)
    return [inst(x, y, z + 1.1 * s, 0.5 * s, 2.2 * s, 0.5 * s, g, rad=9.0),
            inst(x + 0.5 * s, y, z + 1.2 * s, 0.6 * s, 0.26 * s, 0.26 * s, shade(g, 0.92), rad=9.0),
            inst(x + 0.72 * s, y, z + 1.55 * s, 0.26 * s, 0.9 * s, 0.26 * s, shade(g, 1.05), rad=9.0),
            inst(x - 0.45 * s, y, z + 0.9 * s, 0.5 * s, 0.24 * s, 0.24 * s, shade(g, 0.92), rad=9.0),
            inst(x - 0.62 * s, y, z + 1.2 * s, 0.24 * s, 0.7 * s, 0.24 * s, shade(g, 1.05), rad=9.0)]


def awning(x: float, y: float, z: float, yaw: float, col: tuple) -> list:
    """A striped shop awning on a wall at (x, y), sloping down toward ``yaw``."""
    rows = []
    rx, ry = -math.sin(yaw), math.cos(yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    for i in range(5):
        k = (i - 2) * 0.56
        c = col if i % 2 == 0 else (248, 244, 232)
        rows.append(inst(x + rx * k + fx * 0.75, y + ry * k + fy * 0.75, z + 0.1, 0.56, 0.07, 1.6, c, yaw,
                         pitch=-0.34, rad=0.02))
    return rows


def lamp(x: float, y: float, z: float, col: tuple) -> list:
    return [inst(x, y, z + 1.7, 0.14, 3.4, 0.14, (58, 60, 70), mode=1),
            inst(x, y, z + 0.12, 0.34, 0.24, 0.34, (58, 60, 70), mode=1),
            inst(x, y, z + 3.52, 0.4, 0.4, 0.4, col, emis=1.3, rad=0.12),
            inst(x, y, z + 3.78, 0.52, 0.1, 0.52, (58, 60, 70), mode=1)]


def bunting(x0: float, y0: float, x1: float, y1: float, z: float, n: int) -> list:
    cols = ((250, 96, 86), (255, 204, 70), (70, 180, 220), (120, 210, 130), (250, 250, 240))
    ang = math.atan2(y1 - y0, x1 - x0)
    rows = []
    prev = None
    for i in range(n + 1):
        t = i / n
        px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        pz = z - 1.1 * 4 * t * (1 - t)
        if prev is not None:
            mx, my, mz = (px + prev[0]) / 2, (py + prev[1]) / 2, (pz + prev[2]) / 2
            ln = math.dist((px, py, pz), prev)
            rows.append(inst(mx, my, mz, 0.03, 0.03, ln, (240, 236, 220), ang,
                             pitch=math.asin((pz - prev[2]) / max(ln, 1e-6)), rad=0.0))
            rows.append(inst(mx, my, mz - 0.2, 0.34, 0.36, 0.03, cols[i % 5], ang + math.pi / 2, rad=0.02))
        prev = (px, py, pz)
    return rows


def stall(x: float, y: float, yaw: float, col: tuple) -> list:
    rows = []
    rx, ry = -math.sin(yaw), math.cos(yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    for a, b in ((-1.1, -0.9), (1.1, -0.9), (-1.1, 0.9), (1.1, 0.9)):
        rows.append(inst(x + rx * a + fx * b, y + ry * a + fy * b, 1.3, 0.1, 2.6, 0.1, (150, 110, 80), mode=1))
    for i in range(4):
        k = (i - 1.5) * 0.62
        rows.append(inst(x + rx * k, y + ry * k, 2.66, 0.62, 0.08, 2.2, col if i % 2 else (250, 246, 234), yaw,
                         rad=0.03))
    return rows


def monument(x: float, y: float, z: float) -> list:
    rows = [inst(x, y, z + 0.5, 0.7, 1.0, 0.7, (240, 232, 214), rad=0.08),
            inst(x, y, z + 1.5, 0.26, 1.2, 0.26, (120, 170, 110), mode=1),
            inst(x, y, z + 2.4, 0.9, 0.9, 0.9, (255, 190, 50), emis=0.25, rad=9.0)]
    for i in range(8):
        a = i * math.pi / 4
        rows.append(inst(x + math.cos(a) * 0.62, y + math.sin(a) * 0.62, z + 2.4, 0.5, 0.5, 0.5, (255, 150, 40),
                         rad=9.0))
    return rows


def windmill(x: float, y: float, z: float, body: tuple, cap: tuple) -> list:
    rows = [inst(x, y, z + 3.0, 4.2, 6.0, 4.2, body, mode=1),
            inst(x, y, z + 7.4, 3.4, 3.0, 3.4, shade(body, 0.95), mode=1),
            inst(x, y, z + 9.3, 3.8, 1.6, 3.8, cap, rad=9.0),
            inst(x, y + 2.1, z + 8.6, 0.5, 0.5, 0.8, (90, 70, 60), math.pi / 2, rad=0.2)]
    for i in range(4):
        q = i * math.pi / 2 + 0.5
        cx, cz = math.cos(q) * 3.4, math.sin(q) * 3.4
        rows.append(inst(x + cx, y + 2.45, z + 8.6 + cz, 6.6, 1.3, 0.08, (244, 238, 226), math.pi / 2, roll=q,
                         rad=0.03))
    return rows


def water_tower(x: float, y: float, z: float) -> list:
    rows = []
    for a, b in ((-1.3, -1.3), (1.3, -1.3), (-1.3, 1.3), (1.3, 1.3)):
        rows.append(inst(x + a, y + b, z + 2.2, 0.26, 4.4, 0.26, (110, 84, 66), mode=1))
    rows += [inst(x, y, z + 4.5, 3.6, 0.24, 3.6, (120, 92, 70), rad=0.05),
             inst(x, y, z + 6.3, 3.8, 3.4, 3.8, (204, 150, 84), mode=1),
             inst(x, y, z + 5.3, 3.92, 0.18, 3.92, (120, 96, 80), mode=1),
             inst(x, y, z + 7.3, 3.92, 0.18, 3.92, (120, 96, 80), mode=1),
             inst(x, y, z + 8.2, 4.2, 1.2, 4.2, (150, 84, 60), rad=9.0)]
    return rows


def chimney(x: float, y: float, z: float, s: float = 1.0) -> list:
    return [inst(x, y, z + 4.5 * s, 2.0 * s, 9.0 * s, 2.0 * s, (170, 96, 80), mode=1),
            inst(x, y, z + 9.2 * s, 2.3 * s, 0.6 * s, 2.3 * s, (70, 64, 70), mode=1),
            inst(x, y, z + 9.6 * s, 1.5 * s, 0.3 * s, 1.5 * s, (255, 140, 60), emis=1.4, mode=1)]


def rails(track: list) -> list:
    rows = []
    for (ax, ay), (bx, by) in zip(track, track[1:], strict=False):
        ln = math.hypot(bx - ax, by - ay)
        ang = math.atan2(by - ay, bx - ax)
        nx, ny = -math.sin(ang), math.cos(ang)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        for s in (-0.72, 0.72):
            rows.append(inst(mx + nx * s, my + ny * s, 0.09, 0.09, 0.12, ln + 0.1, (150, 152, 164), ang, rad=0.02))
        n = max(int(ln / 1.25), 1)
        for i in range(n + 1):
            t = i / n
            rows.append(inst(ax + (bx - ax) * t, ay + (by - ay) * t, 0.03, 2.1, 0.07, 0.3, (120, 86, 62), ang,
                             rad=0.02))
    return rows


def gate(x: float, y: float, yaw: float, width: float) -> list:
    nx, ny = -math.sin(yaw), math.cos(yaw)
    rows = []
    for s in (-1, 1):
        rows.append(inst(x + nx * s * width / 2, y + ny * s * width / 2, 2.2, 0.42, 4.4, 0.42, (70, 74, 88),
                         mode=1))
        rows.append(inst(x + nx * s * width / 2, y + ny * s * width / 2, 3.2, 0.5, 0.3, 0.5, (255, 196, 64),
                         emis=1.2, mode=1))
    rows.append(inst(x, y, 4.5, width + 0.9, 0.5, 0.5, (70, 74, 88), yaw, rad=0.08))
    rows.append(inst(x, y, 4.5, width * 0.5, 0.2, 0.56, (255, 196, 64), yaw, emis=1.2, rad=0.05))
    return rows


def end_pad(x: float, y: float) -> list:
    return [inst(x, y, 0.03, 6.4, 0.06, 6.4, (80, 84, 96), mode=1),
            inst(x, y, 0.05, 5.4, 0.06, 5.4, (255, 196, 64), emis=0.5, mode=1),
            inst(x, y, 0.07, 4.6, 0.06, 4.6, (80, 84, 96), mode=1)]


# --- jointed models --------------------------------------------------------------------------------------------------
def _euler(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """A part's own turn in the model's axes (right, up, forward)."""
    cp, sp, cq, sq, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll), math.cos(yaw), \
        math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cq, -sq, 0], [sq, cq, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return ry @ rx @ rz


class Model:
    """Parts on joints. A part is (joint, (right, up, forward) offset from the joint, (width, height, depth), colour)
    and the options emis, rad, mode, rot=(pitch, roll, yaw), pat."""

    def __init__(self, parts: list, pivots: list, scale: float = 1.0) -> None:
        n = len(parts)
        self.n = n
        self.joint = np.array([p[0] for p in parts], np.intp)
        self.off = np.array([p[1] for p in parts], np.float64) * scale
        self.piv = np.array(pivots, np.float64) * scale
        self.rot = np.tile(np.eye(3), (n, 1, 1))
        self.team = np.zeros(n, bool)
        self.base = np.zeros((n, ROW), F32)
        for i, p in enumerate(parts):
            opt = p[4] if len(p) > 4 else {}
            size = p[2]
            col = p[3]
            if col == TEAM:
                self.team[i] = True
                col = (255, 255, 255)
            self.base[i, 3] = opt.get("rad", 0.04) * scale
            self.base[i, 7], self.base[i, 11], self.base[i, 15] = size[0] * scale, size[1] * scale, size[2] * scale
            self.base[i, 16:19] = (col[0] / 255, col[1] / 255, col[2] / 255)
            self.base[i, 19] = opt.get("emis", 1.6 if self.team[i] else 0.0)
            self.base[i, 21] = 1.0
            self.base[i, 22] = opt.get("mode", 0)
            self.base[i, 23] = opt.get("pat", 0)
            if "rot" in opt:
                self.rot[i] = _euler(*opt["rot"])
        self.plain = not any("rot" in (p[4] if len(p) > 4 else {}) for p in parts)
        self._rx = np.zeros((n, 3, 3))
        self._rx[:, 0, 0] = 1.0
        self.muzzle = np.zeros(3)

    def rows(self, origin: tuple, basis: np.ndarray, ang: list | None = None, bob: float = 0.0,
             team_rgb: tuple | None = None, flag: float = 0.0, flash: float = 0.0, alpha: float = 1.0) -> np.ndarray:
        """The parts as instance rows. ``basis``: 3x3 whose columns are the model's right, up and forward in the
        renderer's axes; ``ang``: the joints' turn about the right axis (children already include the body's)."""
        out = self.base.copy()
        off = self.off
        if ang is None:
            C = self.piv[self.joint] + off
            A = np.broadcast_to(basis, (self.n, 3, 3)) if self.plain else basis @ self.rot
        else:
            a = np.asarray(ang, np.float64)
            piv = self.piv.copy()
            if a[0] != 0.0:  # the body leans: what hangs from it moves with it
                c0, s0 = math.cos(a[0]), math.sin(a[0])
                d = piv[[HEAD, ARM_L, ARM_R, GUN]] - piv[BODY]
                piv[[HEAD, ARM_L, ARM_R, GUN], 1] = piv[BODY, 1] + d[:, 1] * c0 - d[:, 2] * s0
                piv[[HEAD, ARM_L, ARM_R, GUN], 2] = piv[BODY, 2] + d[:, 1] * s0 + d[:, 2] * c0
            if bob:
                piv[[BODY, HEAD, ARM_L, ARM_R, GUN], 1] += bob
            c, s = np.cos(a)[self.joint], np.sin(a)[self.joint]
            P = piv[self.joint]
            C = np.empty((self.n, 3))
            C[:, 0] = P[:, 0] + off[:, 0]
            C[:, 1] = P[:, 1] + off[:, 1] * c - off[:, 2] * s
            C[:, 2] = P[:, 2] + off[:, 1] * s + off[:, 2] * c
            rx = self._rx
            rx[:, 1, 1] = c
            rx[:, 1, 2] = -s
            rx[:, 2, 1] = s
            rx[:, 2, 2] = c
            A = basis @ (rx if self.plain else rx @ self.rot)
        out[:, 0:3] = np.asarray(origin) + C @ basis.T
        out[:, 4:7] = A[:, :, 0]
        out[:, 8:11] = A[:, :, 1]
        out[:, 12:15] = A[:, :, 2]
        if team_rgb is not None and self.team.any():
            out[self.team, 16:19] = (team_rgb[0] / 255, team_rgb[1] / 255, team_rgb[2] / 255)
        if flash > 0.0:
            out[:, 16:19] += (1.0 - out[:, 16:19]) * min(flash, 1.0) * 0.4
            out[:, 19] += flash * 0.2
        out[:, 20] = flag
        if alpha != 1.0:
            out[:, 21] = alpha
        return out


def yaw_basis(yaw: float, tilt: float = 0.0) -> np.ndarray:
    """Columns right, up, forward (renderer axes) of something standing on the map facing ``yaw``; ``tilt`` tips
    it over backwards about its right axis."""
    ca, sa = math.cos(yaw), math.sin(yaw)
    r = (-sa, 0.0, ca)
    u = (0.0, 1.0, 0.0)
    f = (ca, 0.0, sa)
    if tilt:
        ct, st = math.cos(tilt), math.sin(tilt)
        u, f = ((u[0] * ct - f[0] * st, u[1] * ct - f[1] * st, u[2] * ct - f[2] * st),
                (f[0] * ct + u[0] * st, f[1] * ct + u[1] * st, f[2] * ct + u[2] * st))
    return np.array([[r[0], u[0], f[0]], [r[1], u[1], f[1]], [r[2], u[2], f[2]]])


SKIN = ((232, 184, 150), (176, 124, 92), (244, 206, 170), (120, 84, 66))
PIV = [(0, 0.95, 0), (0, 1.56, 0), (-0.29, 1.46, 0), (0.29, 1.46, 0), (-0.12, 0.95, 0), (0.12, 0.95, 0),
       (0.17, 1.34, 0.2)]


def _limbs(arm: tuple, glove: tuple, leg: tuple, boot: tuple, aw: float = 0.13, lw: float = 0.17,
           two_hands: bool = True) -> list:
    parts = []
    for j in (ARM_L, ARM_R):
        parts += [(j, (0, -0.16, 0), (aw, 0.36, aw + 0.01), arm, {"rad": 0.05}),
                  (j, (0, -0.46, 0), (aw - 0.01, 0.3, aw), glove, {"rad": 0.05}),
                  (j, (0, -0.64, 0), (aw + 0.02, 0.12, aw + 0.03), shade(glove, 0.8), {"rad": 0.05})]
    for j in (LEG_L, LEG_R):
        parts += [(j, (0, -0.23, 0), (lw, 0.48, lw + 0.01), leg, {"rad": 0.05}),
                  (j, (0, -0.66, 0), (lw - 0.02, 0.42, lw - 0.01), boot, {"rad": 0.05}),
                  (j, (0, -0.89, 0.05), (lw + 0.01, 0.12, lw + 0.13), shade(boot, 0.8), {"rad": 0.04})]
    return parts


def _hero_parts(hid: str) -> tuple[list, list, list, tuple]:
    """(body parts, gun parts, pivots, muzzle) of a hero. Gun parts hang from the GUN joint and point forward."""
    piv = [list(p) for p in PIV]
    if hid == "dozer":
        y, st, dk, og = (244, 186, 44), (62, 64, 74), (38, 40, 48), (255, 150, 40)
        piv[ARM_L][0], piv[ARM_R][0] = -0.42, 0.42
        piv[GUN] = [0.44, 1.26, 0.2]
        body = [(BODY, (0, 0.05, 0), (0.56, 0.26, 0.4), st, {"rad": 0.06}),
                (BODY, (0, 0.4, 0), (0.78, 0.56, 0.5), y, {"rad": 0.1}),
                (BODY, (0, 0.38, 0.26), (0.5, 0.3, 0.06), dk, {"rad": 0.03}),
                (BODY, (0, 0.38, 0.29), (0.3, 0.12, 0.04), og, {"emis": 1.5}),
                (BODY, (0, 0.46, -0.3), (0.5, 0.5, 0.16), st, {"rad": 0.06}),
                (BODY, (-0.18, 0.8, -0.3), (0.1, 0.34, 0.1), dk, {"mode": 1}),
                (BODY, (0.18, 0.8, -0.3), (0.1, 0.34, 0.1), dk, {"mode": 1}),
                (BODY, (-0.5, 0.66, 0), (0.34, 0.26, 0.46), y, {"rad": 0.08, "pat": 1}),
                (BODY, (0.5, 0.66, 0), (0.34, 0.26, 0.46), y, {"rad": 0.08, "pat": 1}),
                (BODY, (-0.5, 0.82, 0), (0.2, 0.06, 0.2), TEAM), (BODY, (0.5, 0.82, 0), (0.2, 0.06, 0.2), TEAM),
                (HEAD, (0, 0.1, 0.02), (0.26, 0.24, 0.26), SKIN[1], {"rad": 0.1}),
                (HEAD, (0, 0.22, 0.0), (0.34, 0.14, 0.36), y, {"rad": 0.07}),
                (HEAD, (0, 0.17, 0.16), (0.36, 0.035, 0.16), y, {"rad": 0.015}),
                (HEAD, (0, 0.1, 0.135), (0.22, 0.07, 0.03), (60, 220, 255), {"emis": 1.2}),
                # the blade, folded on the left arm
                (ARM_L, (-0.06, -0.5, 0.14), (0.5, 0.56, 0.08), st, {"rad": 0.03}),
                (ARM_L, (-0.06, -0.76, 0.17), (0.52, 0.08, 0.1), y, {"rad": 0.02, "pat": 1})]
        body += _limbs(st, dk, st, dk, 0.2, 0.24)
        gun = [(GUN, (0, 0, 0.1), (0.2, 0.22, 0.5), st, {"rad": 0.05}),
               (GUN, (0, 0, 0.5), (0.13, 0.13, 0.4), dk, {"rad": 0.05}),
               (GUN, (0, 0.0, 0.72), (0.16, 0.16, 0.06), y, {"rad": 0.03}),
               (GUN, (0, 0.15, 0.12), (0.1, 0.1, 0.3), y, {"rad": 0.03}),
               (GUN, (0, -0.17, 0.0), (0.14, 0.2, 0.16), og, {"rad": 0.04, "emis": 0.5})]
        return body, gun, piv, (0, 0, 0.78)
    if hid == "caldera":
        ir, cu, gl = (72, 70, 82), (196, 112, 58), (255, 120, 30)
        piv[ARM_L][0], piv[ARM_R][0] = -0.4, 0.4
        piv[GUN] = [0.36, 1.2, 0.22]
        body = [(BODY, (0, 0.04, 0), (0.54, 0.26, 0.4), ir, {"rad": 0.08}),
                (BODY, (0, 0.38, 0), (0.74, 0.6, 0.56), ir, {"rad": 0.2}),
                (BODY, (0, 0.36, 0.27), (0.36, 0.3, 0.06), (30, 26, 30), {"rad": 0.03}),
                (BODY, (0, 0.44, 0.3), (0.3, 0.05, 0.03), gl, {"emis": 2.0}),
                (BODY, (0, 0.36, 0.3), (0.3, 0.05, 0.03), gl, {"emis": 2.0}),
                (BODY, (0, 0.28, 0.3), (0.3, 0.05, 0.03), gl, {"emis": 2.0}),
                (BODY, (-0.2, 0.74, -0.24), (0.18, 0.6, 0.18), cu, {"mode": 1}),
                (BODY, (0.2, 0.74, -0.24), (0.18, 0.6, 0.18), cu, {"mode": 1}),
                (BODY, (-0.2, 1.05, -0.24), (0.14, 0.05, 0.14), gl, {"emis": 1.6, "mode": 1}),
                (BODY, (0.2, 1.05, -0.24), (0.14, 0.05, 0.14), gl, {"emis": 1.6, "mode": 1}),
                (BODY, (-0.46, 0.64, 0), (0.3, 0.3, 0.4), cu, {"rad": 0.14}),
                (BODY, (0.46, 0.64, 0), (0.3, 0.3, 0.4), cu, {"rad": 0.14}),
                (BODY, (-0.46, 0.8, 0), (0.16, 0.05, 0.16), TEAM), (BODY, (0.46, 0.8, 0), (0.16, 0.05, 0.16), TEAM),
                (HEAD, (0, 0.13, 0), (0.34, 0.34, 0.34), cu, {"rad": 0.16}),
                (HEAD, (0, 0.13, 0.16), (0.24, 0.06, 0.04), gl, {"emis": 2.2}),
                (HEAD, (0, 0.32, 0), (0.1, 0.08, 0.1), ir, {"mode": 1})]
        body += _limbs(ir, cu, ir, (40, 38, 46), 0.19, 0.23)
        gun = [(GUN, (0, 0, 0.1), (0.16, 0.18, 0.5), cu, {"rad": 0.06}),
               (GUN, (0, 0, 0.5), (0.1, 0.1, 0.34), ir, {"rad": 0.04}),
               (GUN, (0, 0, 0.7), (0.17, 0.17, 0.1), ir, {"rad": 0.05}),
               (GUN, (0, 0, 0.75), (0.1, 0.1, 0.03), gl, {"emis": 2.0}),
               (GUN, (0, -0.16, 0.05), (0.12, 0.2, 0.2), ir, {"rad": 0.05}),
               (GUN, (0, 0.12, 0.1), (0.06, 0.06, 0.3), gl, {"emis": 1.0, "rad": 0.03})]
        return body, gun, piv, (0, 0, 0.78)
    if hid == "flint":
        tl, tn, dk, gl = (36, 150, 150), (214, 180, 120), (44, 60, 72), (140, 255, 220)
        body = [(BODY, (0, 0.04, 0), (0.34, 0.2, 0.22), dk, {"rad": 0.05}),
                (BODY, (0, 0.36, 0), (0.44, 0.46, 0.26), tl, {"rad": 0.08}),
                (BODY, (-0.11, -0.2, -0.08), (0.2, 0.46, 0.07), tl, {"rad": 0.03, "rot": (0.12, 0, 0)}),
                (BODY, (0.11, -0.2, -0.08), (0.2, 0.46, 0.07), tl, {"rad": 0.03, "rot": (0.12, 0, 0)}),
                (BODY, (0, 0.15, 0), (0.46, 0.07, 0.28), tn, {"rad": 0.02}),
                (BODY, (0, 0.15, 0.14), (0.08, 0.08, 0.03), gl, {"emis": 1.4}),
                (BODY, (0, 0.6, 0.02), (0.36, 0.12, 0.32), tn, {"rad": 0.05}),
                (BODY, (-0.3, 0.56, 0), (0.12, 0.05, 0.14), TEAM), (BODY, (0.3, 0.56, 0), (0.12, 0.05, 0.14), TEAM),
                (HEAD, (0, 0.14, 0), (0.24, 0.27, 0.25), SKIN[0], {"rad": 0.11}),
                (HEAD, (0, 0.27, 0), (0.66, 0.035, 0.66), shade(tn, 0.8), {"mode": 1}),
                (HEAD, (0, 0.35, 0), (0.3, 0.14, 0.3), shade(tn, 0.8), {"mode": 1}),
                (HEAD, (0, 0.3, 0), (0.31, 0.04, 0.31), tl, {"mode": 1}),
                (HEAD, (0.07, 0.15, 0.125), (0.09, 0.07, 0.03), gl, {"emis": 1.6})]
        body += _limbs(tl, tn, dk, shade(tn, 0.7), 0.12, 0.16)
        gun = [(GUN, (0, 0, 0.3), (0.06, 0.09, 1.0), dk, {"rad": 0.02}),
               (GUN, (0, -0.03, -0.26), (0.07, 0.15, 0.34), (150, 104, 70), {"rad": 0.03}),
               (GUN, (0, 0.1, 0.16), (0.06, 0.06, 0.34), tl, {"rad": 0.03}),
               (GUN, (0, 0.1, 0.34), (0.05, 0.05, 0.02), gl, {"emis": 1.5}),
               (GUN, (0, -0.1, 0.08), (0.05, 0.14, 0.08), dk, {"rad": 0.02}),
               (GUN, (0, 0, 0.82), (0.085, 0.085, 0.1), tn, {"rad": 0.03})]
        return body, gun, piv, (0, 0, 0.9)
    if hid == "wren":
        mg, wh, dk, gl = (232, 72, 150), (244, 244, 250), (60, 50, 80), (90, 240, 255)
        piv[ARM_L][0], piv[ARM_R][0] = -0.26, 0.26
        body = [(BODY, (0, 0.04, 0), (0.32, 0.2, 0.22), dk, {"rad": 0.06}),
                (BODY, (0, 0.34, 0), (0.4, 0.42, 0.24), mg, {"rad": 0.1}),
                (BODY, (0, 0.42, 0.13), (0.16, 0.2, 0.03), wh, {"rad": 0.02}),
                (BODY, (-0.12, 0.3, -0.18), (0.26, 0.3, 0.14), wh, {"rad": 0.05, "rot": (0, 0.3, 0)}),
                (BODY, (-0.27, 0.54, 0), (0.1, 0.05, 0.12), TEAM), (BODY, (0.27, 0.54, 0), (0.1, 0.05, 0.12), TEAM),
                (HEAD, (0, 0.14, 0), (0.27, 0.27, 0.28), wh, {"rad": 0.13}),
                (HEAD, (0, 0.14, 0.12), (0.22, 0.09, 0.06), gl, {"emis": 1.5, "rad": 0.03}),
                (HEAD, (0, 0.3, -0.06), (0.04, 0.12, 0.26), mg, {"rad": 0.02, "rot": (-0.3, 0, 0)})]
        body += _limbs(mg, wh, dk, wh, 0.11, 0.15)
        body += [(LEG_L, (0, -0.95, 0.02), (0.2, 0.07, 0.36), gl, {"emis": 1.3, "rad": 0.03}),
                 (LEG_R, (0, -0.95, 0.02), (0.2, 0.07, 0.36), gl, {"emis": 1.3, "rad": 0.03})]
        gun = [(GUN, (0, 0, 0.12), (0.07, 0.12, 0.4), wh, {"rad": 0.03}),
               (GUN, (0, 0.0, 0.38), (0.05, 0.05, 0.16), dk, {"rad": 0.02}),
               (GUN, (0, -0.15, 0.1), (0.05, 0.22, 0.07), mg, {"rad": 0.02}),
               (GUN, (0, -0.1, -0.02), (0.05, 0.14, 0.07), dk, {"rad": 0.02}),
               (GUN, (0, 0.08, 0.1), (0.03, 0.03, 0.2), gl, {"emis": 1.4})]
        return body, gun, piv, (0, 0, 0.48)
    if hid == "nitra":
        gr, og, dk, gl = (96, 172, 74), (240, 150, 50), (58, 66, 58), (255, 210, 60)
        piv[ARM_L][0], piv[ARM_R][0] = -0.32, 0.32
        piv[GUN] = [0.22, 1.4, 0.16]
        body = [(BODY, (0, 0.04, 0), (0.42, 0.22, 0.3), dk, {"rad": 0.06}),
                (BODY, (0, 0.36, 0), (0.52, 0.46, 0.34), gr, {"rad": 0.12}),
                (BODY, (0, 0.36, 0.17), (0.5, 0.08, 0.04), og, {"rad": 0.02, "rot": (0, 0.5, 0)}),
                (BODY, (0, 0.34, -0.24), (0.4, 0.44, 0.2), og, {"rad": 0.07}),
                (BODY, (-0.33, 0.58, 0), (0.12, 0.05, 0.14), TEAM), (BODY, (0.33, 0.58, 0), (0.12, 0.05, 0.14), TEAM),
                (HEAD, (0, 0.14, 0), (0.34, 0.32, 0.34), gr, {"rad": 0.15}),
                (HEAD, (0, 0.12, 0.14), (0.26, 0.12, 0.08), (40, 44, 50), {"rad": 0.04}),
                (HEAD, (-0.07, 0.12, 0.185), (0.09, 0.08, 0.02), gl, {"emis": 1.5}),
                (HEAD, (0.07, 0.12, 0.185), (0.09, 0.08, 0.02), gl, {"emis": 1.5}),
                (HEAD, (0, 0.3, 0), (0.36, 0.05, 0.1), og, {"rad": 0.02})]
        for i in range(4):
            body.append((BODY, (-0.18 + i * 0.12, 0.3 + i * 0.045, 0.2), (0.07, 0.14, 0.07), og, {"mode": 1}))
        body += _limbs(gr, dk, gr, dk, 0.15, 0.19)
        gun = [(GUN, (0, 0, 0.2), (0.2, 0.2, 0.7), dk, {"rad": 0.08}),
               (GUN, (0, 0, 0.56), (0.24, 0.24, 0.1), og, {"rad": 0.06}),
               (GUN, (0, 0, 0.6), (0.14, 0.14, 0.04), (20, 20, 24), {"rad": 0.05}),
               (GUN, (0, -0.02, -0.12), (0.26, 0.26, 0.2), gr, {"rad": 0.1}),
               (GUN, (0, -0.2, 0.2), (0.06, 0.18, 0.08), dk, {"rad": 0.02}),
               (GUN, (0, 0.13, 0.2), (0.04, 0.05, 0.2), gl, {"emis": 1.3})]
        return body, gun, piv, (0, 0, 0.66)
    if hid == "tonic":
        mt, wh, pu, gl = (118, 222, 190), (250, 250, 244), (150, 104, 230), (150, 255, 210)
        piv[ARM_L][0], piv[ARM_R][0] = -0.27, 0.27
        body = [(BODY, (0, 0.04, 0), (0.34, 0.2, 0.24), pu, {"rad": 0.06}),
                (BODY, (0, 0.34, 0), (0.42, 0.44, 0.26), wh, {"rad": 0.1}),
                (BODY, (0, 0.14, 0.13), (0.34, 0.6, 0.04), mt, {"rad": 0.03}),
                (BODY, (-0.12, 0.38, -0.22), (0.17, 0.5, 0.17), gl, {"mode": 1, "emis": 0.9}),
                (BODY, (0.12, 0.38, -0.22), (0.17, 0.5, 0.17), gl, {"mode": 1, "emis": 0.9}),
                (BODY, (-0.12, 0.65, -0.22), (0.19, 0.05, 0.19), pu, {"mode": 1}),
                (BODY, (0.12, 0.65, -0.22), (0.19, 0.05, 0.19), pu, {"mode": 1}),
                (BODY, (-0.28, 0.55, 0), (0.1, 0.05, 0.12), TEAM), (BODY, (0.28, 0.55, 0), (0.1, 0.05, 0.12), TEAM),
                (HEAD, (0, 0.14, 0), (0.25, 0.27, 0.26), SKIN[2], {"rad": 0.12}),
                (HEAD, (0, 0.24, -0.02), (0.29, 0.14, 0.3), pu, {"rad": 0.07}),
                (HEAD, (0, 0.36, -0.1), (0.17, 0.17, 0.17), pu, {"rad": 9.0}),
                (HEAD, (-0.065, 0.15, 0.13), (0.1, 0.1, 0.03), gl, {"emis": 1.3, "mode": 0, "rad": 0.04}),
                (HEAD, (0.065, 0.15, 0.13), (0.1, 0.1, 0.03), gl, {"emis": 1.3, "mode": 0, "rad": 0.04})]
        body += _limbs(wh, mt, pu, wh, 0.11, 0.15)
        gun = [(GUN, (0, 0, 0.14), (0.09, 0.11, 0.42), wh, {"rad": 0.04}),
               (GUN, (0, 0, 0.44), (0.05, 0.05, 0.24), pu, {"rad": 0.02}),
               (GUN, (0, 0, 0.6), (0.13, 0.13, 0.08), mt, {"rad": 0.05}),
               (GUN, (0, 0.13, 0.1), (0.12, 0.16, 0.12), gl, {"emis": 1.0, "rad": 0.05}),
               (GUN, (0, -0.14, 0.04), (0.05, 0.18, 0.07), pu, {"rad": 0.02})]
        return body, gun, piv, (0, 0, 0.66)
    # beacon
    nv, gd, wh, gl = (52, 76, 170), (242, 202, 92), (240, 240, 246), (255, 240, 170)
    body = [(BODY, (0, 0.04, 0), (0.34, 0.2, 0.22), nv, {"rad": 0.05}),
            (BODY, (0, 0.38, 0), (0.42, 0.5, 0.25), nv, {"rad": 0.08}),
            (BODY, (0, 0.4, 0.13), (0.06, 0.4, 0.02), gd, {"rad": 0.01}),
            (BODY, (0, 0.16, 0), (0.44, 0.06, 0.27), gd, {"rad": 0.02}),
            (BODY, (-0.24, 0.62, 0), (0.2, 0.06, 0.22), gd, {"rad": 0.02}),
            (BODY, (0.24, 0.62, 0), (0.2, 0.06, 0.22), gd, {"rad": 0.02}),
            (BODY, (-0.24, 0.67, 0), (0.12, 0.04, 0.14), TEAM), (BODY, (0.24, 0.67, 0), (0.12, 0.04, 0.14), TEAM),
            (BODY, (-0.16, 0.7, -0.17), (0.05, 1.3, 0.05), (70, 76, 96), {"mode": 1}),
            (BODY, (-0.16, 1.42, -0.17), (0.22, 0.22, 0.22), gl, {"emis": 2.2, "rad": 9.0}),
            (BODY, (-0.16, 1.28, -0.17), (0.14, 0.04, 0.14), gd, {"mode": 1}),
            (HEAD, (0, 0.14, 0), (0.24, 0.27, 0.25), SKIN[3], {"rad": 0.11}),
            (HEAD, (0, 0.28, 0), (0.3, 0.1, 0.3), nv, {"mode": 1}),
            (HEAD, (0, 0.24, 0.15), (0.26, 0.03, 0.14), (30, 36, 60), {"rad": 0.01}),
            (HEAD, (0, 0.29, 0.13), (0.08, 0.06, 0.03), gd, {"emis": 0.6})]
    body += _limbs(nv, wh, nv, (40, 44, 70), 0.12, 0.16)
    gun = [(GUN, (0, 0, 0.2), (0.07, 0.11, 0.66), wh, {"rad": 0.03}),
           (GUN, (0, 0, 0.58), (0.1, 0.1, 0.14), gd, {"rad": 0.04}),
           (GUN, (0, 0, 0.66), (0.06, 0.06, 0.03), gl, {"emis": 2.0}),
           (GUN, (0, -0.04, -0.2), (0.07, 0.14, 0.26), nv, {"rad": 0.03}),
           (GUN, (0, -0.13, 0.1), (0.05, 0.16, 0.07), nv, {"rad": 0.02}),
           (GUN, (0, 0.09, 0.14), (0.03, 0.04, 0.3), gl, {"emis": 1.2})]
    return body, gun, piv, (0, 0, 0.7)


_hero_cache: dict = {}
_view_cache: dict = {}


def hero_model(hid: str, scale: float) -> Model:
    m = _hero_cache.get(hid)
    if m is None:
        body, gun, piv, muzzle = _hero_parts(hid)
        m = _hero_cache[hid] = Model(body + gun, piv, scale)
        m.muzzle = (np.array(piv[GUN]) + np.array(muzzle)) * scale
        m.gun_piv = np.array(piv[GUN]) * scale
        m.muzzle_off = np.array(muzzle) * scale
    return m


def view_model(hid: str) -> Model:
    """The hero's weapon as its player sees it: the same gun, with two gloves on it, in the eye's own axes."""
    m = _view_cache.get(hid)
    if m is None:
        _body, gun, _piv, muzzle = _hero_parts(hid)
        glove = {"dozer": (38, 40, 48), "caldera": (196, 112, 58), "flint": (214, 180, 120), "wren": (244, 244, 250),
                 "nitra": (58, 66, 58), "tonic": (118, 222, 190), "beacon": (240, 240, 246)}[hid]
        parts = [(0, p[1], p[2], p[3], p[4] if len(p) > 4 else {}) for p in gun]
        parts += [(0, (0.0, -0.13, 0.02), (0.1, 0.12, 0.13), glove, {"rad": 0.045}),
                  (0, (0.05, -0.22, -0.12), (0.09, 0.09, 0.3), shade(glove, 0.9), {"rad": 0.04, "rot": (0.6, 0, 0)}),
                  (0, (-0.06, -0.08, 0.3), (0.1, 0.1, 0.12), glove, {"rad": 0.045})]
        m = _view_cache[hid] = Model(parts, [(0, 0, 0)], VIEW_SCALE)
        m.muzzle = np.array(muzzle, np.float64) * VIEW_SCALE
    return m


_cart = None


def cart_model() -> Model:
    """The escort cart: a brass tank on a hover flatcar."""
    global _cart  # noqa: PLW0603
    if _cart is None:
        st, br, dk, gl = (84, 90, 108), (214, 160, 84), (50, 54, 66), (255, 200, 90)
        half = math.pi / 2
        parts = [(0, (0, 0.5, 0), (2.0, 0.26, 3.8), st, {"rad": 0.08}),
                 (0, (0, 0.34, 0), (1.5, 0.2, 3.2), dk, {"rad": 0.06}),
                 (0, (0, 1.35, 0), (1.5, 2.6, 1.5), br, {"mode": 1, "rot": (half, 0, 0)}),
                 (0, (0, 1.35, 1.3), (1.5, 1.5, 0.7), shade(br, 0.94), {"rad": 9.0}),
                 (0, (0, 1.35, -1.3), (1.5, 1.5, 0.7), shade(br, 0.94), {"rad": 9.0}),
                 (0, (0, 1.35, 0.6), (1.58, 0.12, 1.58), dk, {"mode": 1, "rot": (half, 0, 0)}),
                 (0, (0, 1.35, -0.6), (1.58, 0.12, 1.58), dk, {"mode": 1, "rot": (half, 0, 0)}),
                 (0, (0, 2.2, 0), (0.4, 0.3, 0.4), dk, {"mode": 1}),
                 (0, (0, 2.42, 0), (0.26, 0.2, 0.26), TEAM, {"rad": 9.0, "emis": 2.2}),
                 (0, (0.78, 1.5, 0.2), (0.06, 0.3, 0.3), gl, {"emis": 1.4, "rad": 0.03}),
                 (0, (-0.78, 1.5, 0.2), (0.06, 0.3, 0.3), gl, {"emis": 1.4, "rad": 0.03})]
        for sx in (-0.8, 0.8):
            for sz in (-1.3, 1.3):
                parts += [(0, (sx, 0.26, sz), (0.5, 0.22, 0.9), dk, {"rad": 0.1}),
                          (0, (sx, 0.14, sz), (0.3, 0.06, 0.6), (120, 220, 255), {"emis": 1.8, "rad": 0.03})]
            parts.append((0, (sx * 1.2, 0.66, 0), (0.08, 0.08, 3.6), gl, {"emis": 0.7, "rad": 0.03}))
        _cart = Model(parts, [(0, 0, 0)], 1.0)
    return _cart


def actor_pose(walk: float, move: float, pitch: float, kick: float, air: bool, stun: float, idle_t: float,
               cast: float = 0.0) -> tuple[list, float]:
    """Joint angles (and the body's bob) for a hero walking at phase ``walk`` with strength ``move`` (0-1), looking
    ``pitch`` up, the gun kicked ``kick``."""
    swing = math.sin(walk) * 0.7 * move
    lean = 0.1 * move + 0.35 * stun + 0.03 * math.sin(idle_t * 1.7)
    if air:
        swing = 0.35
        lean -= 0.06
    arm = -(1.12 + pitch * 0.85) - kick * 0.12
    gun = -pitch - kick * 0.2
    head = -pitch * 0.55
    ang = [lean, lean * 0.4 + head, lean * 0.3 + arm * 0.72 - cast * 1.1, lean * 0.3 + arm + cast * 0.2, swing,
           -swing, gun]
    bob = abs(math.sin(walk)) * 0.05 * move + 0.012 * math.sin(idle_t * 2.1)
    return ang, bob
