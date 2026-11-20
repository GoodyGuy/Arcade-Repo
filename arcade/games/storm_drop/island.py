"""Brightwater Isle: the one island of Storm Drop, made once from a fixed seed (numpy only, no Qt).

The ground is a height grid on the same 4 m cells the build pieces snap to. Nine places (a quay, a ruined fort, a
sawmill, a works, beach huts, a lantern tower on the peak, a farm, a scrapyard and a market) stand on levelled pads;
their buildings are ordinary build pieces, so they can be harvested and shot down like anything a player builds.
Trees give wood, rocks brick, and cars / containers / tanks metal. ``ground`` is the painted ground (0.5 m a texel:
sand, two grasses, rock, paths, fields, paving, with soft shadows in its alpha) and ``map_rgb`` the minimap."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .data import BRICK, CELL, FLOOR, GRID, LEVEL, METAL, RAMP, ROOF, SIZE, WALLX, WALLZ, WOOD

SEED = 7719
N = GRID + 1
TEX = 1024
SUN = (0.46, 0.76, 0.46)  # the light of the island (towards the sun); the baked shadows fall the other way
F32 = np.float32


@dataclass
class Place:
    name: str
    gx: int  # the pad: first cell, size in cells, level
    gz: int
    w: int
    d: int
    gy: int = 0
    kind: str = "town"
    paving: tuple = (176, 170, 160)

    @property
    def cx(self) -> float:
        return (self.gx + self.w / 2) * CELL

    @property
    def cz(self) -> float:
        return (self.gz + self.d / 2) * CELL

    @property
    def radius(self) -> float:
        return max(self.w, self.d) * CELL * 0.6


@dataclass
class Prop:
    kind: str
    x: float
    y: float
    z: float
    yaw: float
    shape: str  # "cyl" (r, h) or "box" (hx, hz, h)
    a: float
    b: float
    h: float
    mat: int
    hp: float
    seed: float = 0.0
    scale: float = 1.0
    alive: bool = True
    slots: list = field(default_factory=list)  # where its instances sit in the renderer's buffers
    hit_t: float = -9.0  # the time it was last hit (a shake)
    hp_max: float = 0.0


PROP_DEFS = {  # kind: (shape, a, b, h, material, health)
    "pine": ("cyl", 0.42, 0.0, 6.5, WOOD, 150), "oak": ("cyl", 0.5, 0.0, 5.5, WOOD, 170),
    "palm": ("cyl", 0.34, 0.0, 5.5, WOOD, 110), "rock": ("cyl", 1.25, 0.0, 1.5, BRICK, 220),
    "car": ("box", 1.0, 2.2, 1.5, METAL, 260), "container": ("box", 1.25, 3.1, 2.6, METAL, 420),
    "tank": ("cyl", 1.7, 0.0, 3.6, METAL, 380), "machine": ("box", 1.1, 1.1, 2.2, METAL, 240),
    "crate": ("box", 0.7, 0.7, 1.3, WOOD, 90), "barrel": ("cyl", 0.48, 0.0, 1.15, METAL, 100),
    "hay": ("box", 0.9, 1.3, 1.2, WOOD, 80), "logs": ("box", 1.1, 2.0, 1.0, WOOD, 160),
    "stall": ("box", 0.9, 1.4, 1.0, WOOD, 100), "pillar": ("cyl", 0.5, 0.0, 3.2, BRICK, 200),
}


def _vnoise(rng: np.random.Generator, n: int, cells: int) -> np.ndarray:
    g = rng.random((cells + 2, cells + 2)).astype(F32)
    t = np.linspace(0.0, cells, n, endpoint=False)
    i = t.astype(int)
    f = (t - i).astype(F32)
    f = f * f * (3.0 - 2.0 * f)
    fx, fz = f[None, :], f[:, None]
    a, b = g[i][:, i], g[i][:, i + 1]
    c, d = g[i + 1][:, i], g[i + 1][:, i + 1]
    return (a * (1 - fx) + b * fx) * (1 - fz) + (c * (1 - fx) + d * fx) * fz


def _fbm(rng: np.random.Generator, n: int, cells: int, octaves: int) -> np.ndarray:
    out = np.zeros((n, n), F32)
    amp, total = 1.0, 0.0
    for k in range(octaves):
        out += amp * _vnoise(rng, n, cells * 2 ** k)
        total += amp
        amp *= 0.5
    return out / total


def _smooth(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _upsample(a: np.ndarray, n: int) -> np.ndarray:
    """Bilinear: an (m, m) grid over the island to (n, n) texels (texel centres)."""
    m = a.shape[0] - 1
    t = (np.arange(n) + 0.5) / n * m
    i = np.minimum(t.astype(int), m - 1)
    f = (t - i).astype(F32)
    fx, fz = f[None, :], f[:, None]
    return ((a[i][:, i] * (1 - fx) + a[i][:, i + 1] * fx) * (1 - fz)
            + (a[i + 1][:, i] * (1 - fx) + a[i + 1][:, i + 1] * fx) * fz)


class Kit:
    """Puts build pieces on the grid for the island's own buildings and notes where loot goes."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.pieces: dict[tuple, tuple] = {}  # (kind, gx, gy, gz) -> (kind, gx, gy, gz, dir, mat, variant, anchor)
        self.chests: list[tuple] = []
        self.floor_loot: list[tuple] = []
        self.ammo: list[tuple] = []
        self.shadows: list[tuple] = []  # (x0, z0, x1, z1, height m)

    def add(self, kind: int, gx: int, gy: int, gz: int, mat: int, d: int = 0, var: int = 0,
            anchor: bool = False) -> None:
        self.pieces[(kind, gx, gy, gz)] = (kind, gx, gy, gz, d, mat, var, anchor)

    def wall(self, gx: int, gy: int, gz: int, side: str, mat: int, var: int = 0) -> None:
        if side == "S":
            self.add(WALLX, gx, gy, gz, mat, var=var)
        elif side == "N":
            self.add(WALLX, gx, gy, gz + 1, mat, var=var)
        elif side == "W":
            self.add(WALLZ, gx, gy, gz, mat, var=var)
        else:
            self.add(WALLZ, gx + 1, gy, gz, mat, var=var)

    def spot(self, gx: int, gy: int, gz: int, ox: float = 0.5, oz: float = 0.5) -> tuple:
        return ((gx + ox) * CELL, gy * LEVEL + 0.06, (gz + oz) * CELL)

    def house(self, gx: int, gz: int, w: int, d: int, gy: int, storeys: int, mat: int, roof_mat: int = WOOD,
              doors: str = "SN", roof: str = "peak", ground_floor: bool = True, stairs: bool = True,
              ruin: float = 0.0, floor_mat: int = WOOD, chests: int = 1, windows: float = 0.4,
              hollow: bool = False) -> None:
        """A box of walls ``storeys`` high with floors, a stair ramp per storey, doors and a roof. ``hollow``: one
        tall room (floors only round a gallery)."""
        rng = self.rng
        keep = (lambda: rng.random() >= ruin) if ruin else (lambda: True)
        door_cells = {"S": (gx + w // 2, gz), "N": (gx + (w - 1) // 2, gz + d - 1), "W": (gx, gz + d // 2),
                      "E": (gx + w - 1, gz + (d - 1) // 2)}
        wells: set[tuple] = set()
        for s in range(storeys):
            y = gy + s
            for i in range(w):
                for side, cz in (("S", gz), ("N", gz + d - 1)):
                    if s == 0 and side in doors and door_cells[side] == (gx + i, cz):
                        continue
                    if keep():
                        self.wall(gx + i, y, cz, side, mat, 1 if rng.random() < windows else 0)
            for j in range(d):
                for side, cx in (("W", gx), ("E", gx + w - 1)):
                    if s == 0 and side in doors and door_cells[side] == (cx, gz + j):
                        continue
                    if keep():
                        self.wall(cx, y, gz + j, side, mat, 1 if rng.random() < windows else 0)
            # the stair of this storey and the opening over it
            last = s == storeys - 1
            if stairs and d >= 2 and (not last or roof == "flat"):
                if s % 2 == 0:
                    rx, rz, rd = gx + w - 1, gz, 0
                else:
                    rx, rz, rd = gx, gz + d - 1, 2
                if (rx, rz) in (door_cells[c] for c in doors):
                    rx = gx + w - 1 - (rx - gx) if w > 1 else rx
                self.add(RAMP, rx, y, rz, floor_mat, d=rd)
                wells.add((rx, y + 1, rz))
            if s > 0 or ground_floor:
                for i in range(w):
                    for j in range(d):
                        if (gx + i, y, gz + j) in wells:
                            continue
                        if hollow and s > 0 and 0 < i < w - 1 and j < d - 1:
                            continue
                        if keep():
                            self.add(FLOOR, gx + i, y, gz + j, floor_mat if s > 0 or mat != METAL else METAL)
        top = gy + storeys
        for i in range(w):
            for j in range(d):
                if (gx + i, top, gz + j) in wells:
                    continue
                if roof == "peak" and keep():
                    self.add(ROOF, gx + i, top, gz + j, roof_mat)
                elif roof == "flat" and keep():
                    self.add(FLOOR, gx + i, top, gz + j, roof_mat)
        self.shadows.append((gx * CELL, gz * CELL, (gx + w) * CELL, (gz + d) * CELL, storeys * LEVEL + 1.0))
        # loot: chests against a wall, floor finds elsewhere
        cells = [(gx + i, gy + s, gz + j) for s in range(storeys) for i in range(w) for j in range(d)
                 if (gx + i, gy + s + 1, gz + j) not in wells and (gx + i, gy + s, gz + j) not in wells
                 and not (s == 0 and (gx + i, gz + j) in door_cells.values())
                 and not (hollow and s > 0 and 0 < i < w - 1 and j < d - 1)]
        rng.shuffle(cells)
        for k, (cx, cy, cz) in enumerate(cells[:chests + 2]):
            ox = 0.24 if cx == gx else (0.76 if cx == gx + w - 1 else 0.5)
            oz = 0.24 if cz == gz else (0.76 if cz == gz + d - 1 else 0.5)
            if k < chests:
                yaw = math.atan2((gx + w / 2) - (cx + ox), (gz + d / 2) - (cz + oz))
                self.chests.append((*self.spot(cx, cy, cz, ox, oz), yaw))
            elif k == chests:
                self.floor_loot.append(self.spot(cx, cy, cz, 1 - ox if ox != 0.5 else 0.4, 1 - oz if oz != 0.5 else 0.6))
            else:
                self.ammo.append(self.spot(cx, cy, cz, 0.5, 0.5 if oz != 0.5 else 0.3))

    def tower(self, gx: int, gz: int, gy: int, levels: int, mat: int, floor_mat: int = WOOD) -> None:
        """A two-cell lookout: a ramp in one cell, a landing in the other, turn and turn about."""
        for s in range(levels):
            y = gy + s
            if s % 2 == 0:
                self.add(RAMP, gx, y, gz, floor_mat, d=1)  # rising east
                self.add(FLOOR, gx + 1, y + 1, gz, floor_mat)
            else:
                self.add(RAMP, gx + 1, y, gz, floor_mat, d=3)  # rising west
                self.add(FLOOR, gx, y + 1, gz, floor_mat)
            self.wall(gx, y, gz, "N", mat)
            self.wall(gx + 1, y, gz, "N", mat)
        top = gy + levels
        self.add(FLOOR, gx, top, gz, floor_mat)
        self.add(FLOOR, gx + 1, top, gz, floor_mat)
        self.add(ROOF, gx, top + 1, gz, floor_mat)
        self.add(ROOF, gx + 1, top + 1, gz, floor_mat)
        for cx, side in ((gx, "W"), (gx + 1, "E")):
            self.wall(cx, top, gz, side, mat, 1)
        self.wall(gx, top, gz, "N", mat, 1)
        self.wall(gx + 1, top, gz, "N", mat, 1)
        self.chests.append((*self.spot(gx if levels % 2 == 0 else gx + 1, top, gz, 0.5, 0.7), math.pi))
        self.shadows.append((gx * CELL, gz * CELL, (gx + 2) * CELL, (gz + 1) * CELL, (levels + 1) * LEVEL))


class Island:
    _cached: "Island | None" = None

    @classmethod
    def get(cls) -> "Island":
        if cls._cached is None:
            cls._cached = cls()
        return cls._cached

    def __init__(self) -> None:
        rng = np.random.default_rng(SEED)
        self.places: list[Place] = [
            Place("Copper Quay", 56, 17, 16, 9, kind="quay", paving=(178, 172, 166)),
            Place("Tumbledown", 27, 47, 9, 9, kind="fort", paving=(160, 150, 132)),
            Place("Sawmill Hollow", 35, 84, 10, 8, kind="mill", paving=(170, 138, 96)),
            Place("Gizmo Works", 86, 52, 12, 10, kind="works", paving=(150, 156, 166)),
            Place("Palm Pocket", 87, 26, 9, 7, kind="beach", paving=(238, 216, 152)),
            Place("High Lantern", 80, 84, 5, 5, kind="peak", paving=(168, 164, 160)),
            Place("Clover Farm", 59, 67, 11, 9, kind="farm", paving=(176, 146, 100)),
            Place("Old Circuit", 37, 27, 10, 9, kind="scrap", paving=(140, 134, 128)),
            Place("Crossways", 59, 44, 8, 7, kind="market", paving=(196, 178, 150)),
        ]
        self._terrain(rng)
        self.kit = Kit(rng)
        self.props: list[Prop] = []
        self.decor: list[tuple] = []  # (kind, x, y, z, yaw, scale, seed): looks only
        self._build_places(rng)
        self._scatter(rng)
        self._paint(rng)
        self.pieces = list(self.kit.pieces.values())
        self.chest_spots = self.kit.chests
        self.floor_spots = self.kit.floor_loot
        self.ammo_spots = self.kit.ammo
        self.H = [float(v) for v in self.heights.reshape(-1)]
        self.prop_cols: dict[tuple, list[Prop]] = {}
        for p in self.props:
            p.hp_max = p.hp
            r = p.a if p.shape == "cyl" else max(p.a, p.b)
            for cx in range(int((p.x - r) // CELL), int((p.x + r) // CELL) + 1):
                for cz in range(int((p.z - r) // CELL), int((p.z + r) // CELL) + 1):
                    self.prop_cols.setdefault((cx, cz), []).append(p)

    # --- the ground ---------------------------------------------------------------------------------------------------

    def _terrain(self, rng: np.random.Generator) -> None:
        c = SIZE / 2
        xs = np.arange(N, dtype=F32) * CELL
        X, Z = np.meshgrid(xs, xs)  # [j (z), i (x)]
        dx, dz = X - c, Z - c
        dist = np.hypot(dx, dz)
        ang = np.arctan2(dz, dx)
        coast = (198 + 20 * np.sin(3 * ang + 1.0) + 13 * np.sin(5 * ang + 2.3) + 8 * np.sin(8 * ang + 0.7)
                 + 16 * (_fbm(rng, N, 5, 3) - 0.5))
        edge = coast - dist
        hills = _fbm(rng, N, 5, 4)
        rough = _fbm(rng, N, 22, 2)
        h = np.where(edge > 0,
                     0.5 + 5.5 * _smooth(0, 80, edge) + _smooth(8, 70, edge) * ((hills - 0.42) * 17 + rough * 1.6)
                     + _smooth(0, 14, edge) * 0.9,
                     np.maximum(edge * 0.22, -7.0) + 0.5 * np.exp(edge / 6.0))
        h = np.maximum(h, np.where(edge > 6, 1.3, -99.0))
        # the peak in the north-east and the fort's plateau in the west
        h += 34.0 * np.exp(-((X - 330) ** 2 + (Z - 346) ** 2) / (2 * 44.0 ** 2)) * _smooth(-10, 40, edge)
        h += 7.0 * np.exp(-((X - 126) ** 2 + (Z - 206) ** 2) / (2 * 40.0 ** 2)) * _smooth(0, 40, edge)
        # a pond north-west of the middle
        pond = np.exp(-((X - 196) ** 2 + (Z - 292) ** 2) / (2 * 15.0 ** 2))
        h = h * (1 - pond) + (-1.6) * pond
        self.base = h.astype(F32).copy()
        # small places away from the big ones: a shack or a lookout each on its own little pad
        taken = [(p.cx, p.cz, p.radius + 26) for p in self.places]
        self.shacks: list[Place] = []
        tries = 0
        while len(self.shacks) < 11 and tries < 4000:
            tries += 1
            gx, gz = int(rng.integers(14, GRID - 16)), int(rng.integers(14, GRID - 16))
            x, z = (gx + 1) * CELL, (gz + 1) * CELL
            if any(math.hypot(x - tx, z - tz) < tr for tx, tz, tr in taken):
                continue
            patch = h[gz - 1:gz + 4, gx - 1:gx + 4]
            if patch.min() < 2.4 or patch.max() - patch.min() > 5.0 or abs(x - 196) + abs(z - 292) < 60:
                continue
            self.shacks.append(Place(f"shack{len(self.shacks)}", gx, gz, 2, 2, kind="shack"))
            taken.append((x, z, 44))
        for p in self.places + self.shacks:
            patch = h[p.gz:p.gz + p.d + 1, p.gx:p.gx + p.w + 1]
            p.gy = max(int(round(float(np.median(patch)) / LEVEL)), 1)
            target = p.gy * LEVEL
            x0, x1, z0, z1 = p.gx * CELL, (p.gx + p.w) * CELL, p.gz * CELL, (p.gz + p.d) * CELL
            out = np.hypot(np.maximum(np.maximum(x0 - X, X - x1), 0), np.maximum(np.maximum(z0 - Z, Z - z1), 0))
            wgt = 1.0 - _smooth(0.0, 20.0 if p.kind != "shack" else 10.0, out)
            h = h * (1 - wgt) + target * wgt
        self.heights = h.astype(F32)
        gz_, gx_ = np.gradient(self.heights, CELL)
        nrm = np.stack([-gx_, np.ones_like(gx_), -gz_], axis=-1)
        self.normals = (nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)).astype(F32)
        self.forest = _fbm(rng, N, 6, 3)

    def height(self, x: float, z: float) -> float:
        if x < 0.0 or z < 0.0:
            return -8.0
        fx, fz = x * 0.25, z * 0.25
        i, j = int(fx), int(fz)
        if i >= GRID or j >= GRID:
            return -8.0
        fx -= i
        fz -= j
        H = self.H
        k = j * N + i
        if fx + fz <= 1.0:
            h00 = H[k]
            return h00 + fx * (H[k + 1] - h00) + fz * (H[k + N] - h00)
        h11 = H[k + N + 1]
        return h11 + (1.0 - fx) * (H[k + N] - h11) + (1.0 - fz) * (H[k + 1] - h11)

    def normal(self, x: float, z: float) -> tuple[float, float, float]:
        e = 0.6
        dx = self.height(x + e, z) - self.height(x - e, z)
        dz = self.height(x, z + e) - self.height(x, z - e)
        n = math.sqrt(dx * dx + dz * dz + 4 * e * e)
        return (-dx / n, 2 * e / n, -dz / n)

    def terrain_mesh(self) -> np.ndarray:
        """Triangles of (position, normal) for the renderer; each cell is cut along the same diagonal ``height`` uses."""
        xs = np.arange(N, dtype=F32) * CELL
        X, Z = np.meshgrid(xs, xs)
        P = np.stack([X, self.heights, Z], axis=-1)
        V = np.concatenate([P, self.normals], axis=-1)  # (N, N, 6)
        a, b, c, d = V[:-1, :-1], V[:-1, 1:], V[1:, :-1], V[1:, 1:]  # 00, 10 (x+1), 01 (z+1), 11
        tris = np.stack([a, b, c, d, c, b], axis=2)  # (G, G, 6, 6)
        return np.ascontiguousarray(tris.reshape(-1, 6), F32)

    # --- places -------------------------------------------------------------------------------------------------------

    def _prop(self, kind: str, x: float, z: float, yaw: float = 0.0, scale: float = 1.0, y: float | None = None,
              rng: np.random.Generator | None = None) -> Prop:
        shape, a, b, h, mat, hp = PROP_DEFS[kind]
        if shape == "box" and abs(math.sin(yaw)) > 0.7:
            a, b = b, a
        gy = float(self._h(x, z)) if y is None else y
        p = Prop(kind, x, gy, z, yaw, shape, a * scale, b * scale, h * scale, mat, hp * scale,
                 seed=float(rng.random()) if rng is not None else 0.37, scale=scale)
        self.props.append(p)
        return p

    def _h(self, x: float, z: float) -> float:
        fx, fz = min(max(x / CELL, 0.0), GRID - 1e-3), min(max(z / CELL, 0.0), GRID - 1e-3)
        i, j = int(fx), int(fz)
        fx -= i
        fz -= j
        h = self.heights
        if fx + fz <= 1.0:
            return float(h[j, i] + fx * (h[j, i + 1] - h[j, i]) + fz * (h[j + 1, i] - h[j, i]))
        return float(h[j + 1, i + 1] + (1 - fx) * (h[j + 1, i] - h[j + 1, i + 1])
                     + (1 - fz) * (h[j, i + 1] - h[j + 1, i + 1]))

    def _build_places(self, rng: np.random.Generator) -> None:
        k = self.kit
        P = {p.kind: p for p in self.places}
        half = math.pi / 2

        def at(p: Place, cx: float, cz: float) -> tuple[float, float]:
            return (p.gx + cx) * CELL, (p.gz + cz) * CELL

        # Copper Quay: brick houses, a wooden warehouse, containers and a pier out to sea
        p = P["quay"]
        g = p.gy
        k.house(p.gx + 1, p.gz + 5, 3, 2, g, 2, BRICK, doors="SE", chests=1)
        k.house(p.gx + 5, p.gz + 6, 2, 2, g, 2, BRICK, doors="SW", chests=1)
        k.house(p.gx + 12, p.gz + 5, 3, 3, g, 2, BRICK, roof="flat", doors="SW", chests=2)
        k.house(p.gx + 7, p.gz + 5, 4, 3, g, 1, WOOD, roof="flat", doors="SN", windows=0.15, chests=1)
        for i, (cx, cz, yw) in enumerate(((1.4, 2.2, half), (3.6, 2.2, half), (2.5, 3.6, half), (9.5, 1.6, 0.0),
                                          (10.9, 1.6, 0.0), (13.4, 2.4, half), (6.3, 3.0, 0.0))):
            x, z = at(p, cx, cz)
            self._prop("container", x, z, yw, rng=rng)
            if i in (0, 3):
                x2, z2 = at(p, cx + (0.0 if yw else 0.05), cz)
                self._prop("container", x2, z2, yw, y=g * LEVEL + 2.6, rng=rng)
        for cx, cz in ((5.2, 4.3), (5.9, 4.4), (11.6, 3.8), (15.2, 6.5), (0.6, 4.4)):
            self._prop("barrel" if rng.random() < 0.5 else "crate", *at(p, cx, cz), rng=rng)
        pier_x = p.gx + 7
        for j in range(1, 9):
            k.add(FLOOR, pier_x, g, p.gz - j, WOOD, anchor=True)
            k.add(FLOOR, pier_x + 1, g, p.gz - j, WOOD, anchor=True)
            if j % 2 == 0:
                for ox in (0.0, 2.0):
                    self.decor.append(("post", (pier_x + ox) * CELL, -2.0, (p.gz - j + 0.5) * CELL, 0.0,
                                       g * LEVEL + 2.0, 0.3))
        k.chests.append((*k.spot(pier_x, g, p.gz - 8, 0.5, 0.4), 0.0))
        k.floor_loot.append(k.spot(pier_x + 1, g, p.gz - 5))
        k.floor_loot.append(k.spot(p.gx + 4, g, p.gz + 3))

        # Tumbledown: a ring of broken brick walls, corner towers and a keep
        p = P["fort"]
        g = p.gy
        x0, z0, n = p.gx + 1, p.gz + 1, 7
        for s in range(2):
            for i in range(n):
                for side, cz in (("S", z0), ("N", z0 + n - 1)):
                    if not (s == 0 and i == n // 2) and rng.random() > (0.2 + 0.3 * s):
                        k.wall(x0 + i, g + s, cz, side, BRICK)
                for side, cx in (("W", x0), ("E", x0 + n - 1)):
                    if not (s == 0 and i == n // 2) and rng.random() > (0.2 + 0.3 * s):
                        k.wall(cx, g + s, z0 + i, side, BRICK)
        for cx, cz in ((x0, z0), (x0 + n - 1, z0), (x0, z0 + n - 1), (x0 + n - 1, z0 + n - 1)):
            k.house(cx, cz, 1, 1, g, 3, BRICK, roof="peak", roof_mat=BRICK, doors="SNWE"[int(rng.integers(4))],
                    stairs=False, ground_floor=False, ruin=0.12, windows=0.5, chests=0)
            k.floor_loot.append(k.spot(cx, g, cz))
        k.house(x0 + 2, z0 + 2, 3, 3, g, 2, BRICK, roof="flat", roof_mat=BRICK, doors="SN", ruin=0.1, chests=2,
                floor_mat=BRICK)
        for cx, cz in ((1.6, 4.5), (5.4, 1.7), (7.3, 6.2), (3.3, 7.4)):
            self._prop("pillar", *at(p, cx, cz), rng=rng)
        k.ammo.append(k.spot(x0 + 1, g, z0 + 5))

        # Sawmill Hollow: cabins, an open mill shed, log piles
        p = P["mill"]
        g = p.gy
        k.house(p.gx + 1, p.gz + 5, 2, 2, g, 1, WOOD, doors="S", chests=1)
        k.house(p.gx + 7, p.gz + 4, 2, 2, g, 2, WOOD, doors="W", chests=1)
        k.house(p.gx + 3, p.gz + 1, 4, 2, g, 1, WOOD, doors="SN", roof="peak", windows=0.0, chests=1)
        for cx, cz, yw in ((1.5, 2.5, 0.0), (8.4, 1.6, half), (5.0, 5.4, half), (4.6, 6.6, half), (8.6, 7.2, 0.0)):
            self._prop("logs", *at(p, cx, cz), yw, rng=rng)
        for cx, cz in ((0.7, 4.3), (6.4, 4.2), (3.4, 3.7)):
            self._prop("crate", *at(p, cx, cz), rng=rng)
        self._prop("machine", *at(p, 4.5, 2.0), rng=rng)
        k.floor_loot.append(k.spot(p.gx + 5, g, p.gz + 4))
        k.ammo.append(k.spot(p.gx + 2, g, p.gz + 3))

        # Gizmo Works: a tall metal hall with a gallery, tanks, machines, an office
        p = P["works"]
        g = p.gy
        k.house(p.gx + 1, p.gz + 4, 5, 4, g, 2, METAL, roof="flat", roof_mat=METAL, doors="SE", hollow=True,
                windows=0.2, chests=2, floor_mat=METAL)
        k.house(p.gx + 8, p.gz + 6, 3, 2, g, 2, BRICK, roof="flat", doors="SW", chests=1)
        for cx, cz in ((8.0, 1.8), (10.2, 1.8), (8.0, 4.0), (10.2, 4.0)):
            self._prop("tank", *at(p, cx, cz), rng=rng)
        for cx, cz in ((2.4, 5.5), (3.6, 5.5), (2.4, 6.6), (4.4, 6.4), (4.9, 5.2)):
            self._prop("machine", *at(p, cx, cz), rng=rng)
        for cx, cz, yw in ((2.0, 1.6, half), (4.6, 1.6, half), (6.9, 2.6, 0.0)):
            self._prop("container", *at(p, cx, cz), yw, rng=rng)
        for cx, cz in ((6.6, 4.5), (7.2, 8.6), (0.6, 2.8), (11.3, 5.4)):
            self._prop("barrel", *at(p, cx, cz), rng=rng)
        k.floor_loot.append(k.spot(p.gx + 9, g, p.gz + 3))
        k.floor_loot.append(k.spot(p.gx + 6, g, p.gz + 1))
        k.ammo.append(k.spot(p.gx + 7, g, p.gz + 5))

        # Palm Pocket: beach huts and a lookout
        p = P["beach"]
        g = p.gy
        for cx, cz, door in ((1, 1, "N"), (3, 1, "N"), (6, 1, "N"), (1, 4, "S")):
            k.house(p.gx + cx, p.gz + cz, 1, 1, g, 1, WOOD, doors=door, ground_floor=True, stairs=False,
                    chests=1 if cx != 3 else 0, windows=0.6)
            if cx == 3:
                k.floor_loot.append(k.spot(p.gx + cx, g, p.gz + cz))
        k.house(p.gx + 4, p.gz + 4, 2, 2, g, 1, WOOD, doors="SW", roof="flat", chests=1, windows=0.7)
        k.tower(p.gx + 7, p.gz + 5, g, 3, WOOD)
        for cx, cz in ((2.5, 3.2), (5.6, 3.1), (0.5, 3.0)):
            self._prop("crate", *at(p, cx, cz), rng=rng)
        k.ammo.append(k.spot(p.gx + 3, g, p.gz + 4))

        # High Lantern: a brick tower on the peak
        p = P["peak"]
        g = p.gy
        k.house(p.gx + 1, p.gz + 1, 2, 2, g, 4, BRICK, roof="flat", roof_mat=BRICK, doors="S", chests=2,
                floor_mat=WOOD, windows=0.5)
        top = g + 4
        self.decor.append(("lantern", (p.gx + 2) * CELL, top * LEVEL, (p.gz + 2) * CELL, 0.0, 1.0, 0.5))
        k.chests.append((*k.spot(p.gx + 1, top, p.gz + 2, 0.4, 0.5), half))
        k.floor_loot.append(k.spot(p.gx + 3, g, p.gz + 4))
        k.ammo.append(k.spot(p.gx + 4, g, p.gz + 1))

        # Clover Farm: a barn with a loft, a farmhouse, a silo and bales
        p = P["farm"]
        g = p.gy
        k.house(p.gx + 1, p.gz + 4, 4, 3, g, 2, WOOD, doors="SE", hollow=True, windows=0.1, chests=2)
        k.house(p.gx + 7, p.gz + 5, 2, 2, g, 2, BRICK, doors="SW", chests=1)
        self._prop("tank", *at(p, 6.0, 7.6), rng=rng)
        for cx, cz, yw in ((2.0, 2.6, 0.0), (3.2, 2.5, half), (6.3, 3.2, 0.0), (8.8, 3.0, half), (9.6, 1.4, 0.0),
                           (5.2, 1.2, half)):
            self._prop("hay", *at(p, cx, cz), yw, rng=rng)
        for cx, cz in ((0.6, 3.6), (9.8, 4.4)):
            self._prop("crate", *at(p, cx, cz), rng=rng)
        k.floor_loot.append(k.spot(p.gx + 5, g, p.gz + 3))
        k.ammo.append(k.spot(p.gx + 8, g, p.gz + 2))

        # Old Circuit: rows of cars, a metal shed, containers
        p = P["scrap"]
        g = p.gy
        k.house(p.gx + 6, p.gz + 5, 3, 3, g, 1, METAL, roof="flat", roof_mat=METAL, doors="SW", windows=0.1,
                chests=2, floor_mat=METAL)
        for i in range(4):
            for j in range(2):
                if rng.random() < 0.85:
                    x, z = at(p, 1.3 + i * 1.25, 1.6 + j * 1.7)
                    self._prop("car", x + rng.uniform(-0.3, 0.3), z + rng.uniform(-0.3, 0.3),
                               rng.choice([0.0, 0.0, half]) if j else 0.0, rng=rng)
        for cx, cz, yw in ((1.6, 5.6, half), (2.6, 7.4, 0.0), (7.6, 2.0, half)):
            self._prop("container", *at(p, cx, cz), yw, rng=rng)
        for cx, cz in ((4.4, 5.4), (4.2, 7.6), (8.9, 3.6), (0.6, 4.2)):
            self._prop("car", *at(p, cx, cz), half if rng.random() < 0.5 else 0.0, rng=rng)
        for cx, cz in ((5.4, 4.0), (6.6, 3.6)):
            self._prop("barrel", *at(p, cx, cz), rng=rng)
        k.floor_loot.append(k.spot(p.gx + 3, g, p.gz + 4))
        k.floor_loot.append(k.spot(p.gx + 8, g, p.gz + 1))
        k.ammo.append(k.spot(p.gx + 5, g, p.gz + 6))

        # Crossways: a shop, market stalls round a well
        p = P["market"]
        g = p.gy
        k.house(p.gx + 1, p.gz + 4, 3, 2, g, 2, BRICK, doors="SE", chests=2)
        k.house(p.gx + 5, p.gz + 4, 2, 2, g, 1, WOOD, doors="SW", roof="flat", chests=1)
        for cx, cz in ((1, 1), (3, 1), (6, 1)):
            k.wall(p.gx + cx, g, p.gz + cz, "N", WOOD)
            k.add(ROOF, p.gx + cx, g + 1, p.gz + cz, WOOD)
            self._prop("stall", *at(p, cx + 0.5, cz + 0.45), half, rng=rng)
            k.floor_loot.append(k.spot(p.gx + cx, g, p.gz + cz, 0.5, 0.75))
        self._prop("pillar", *at(p, 4.6, 2.9), rng=rng)
        for cx, cz in ((5.5, 3.4), (0.6, 3.0), (7.4, 2.6)):
            self._prop("crate" if rng.random() < 0.6 else "barrel", *at(p, cx, cz), rng=rng)
        k.ammo.append(k.spot(p.gx + 4, g, p.gz + 5))

        # the small places
        for i, s in enumerate(self.shacks):
            mat = (WOOD, WOOD, BRICK, METAL)[i % 4]
            if i % 3 == 2:
                k.tower(s.gx, s.gz, s.gy, 2 + i % 2, mat)
            else:
                k.house(s.gx, s.gz, 2 if i % 2 else 1, 2 if i % 4 == 0 else 1, s.gy, 1 + (i % 5 == 0), mat,
                        doors="SNEW"[i % 4], chests=1, stairs=True, windows=0.5)
            x, z = (s.gx + 2.4) * CELL, (s.gz - 0.3) * CELL
            self._prop(("crate", "barrel", "hay", "logs")[i % 4], x, z, rng=rng)

    def _near_built(self, x: float, z: float, margin: float) -> bool:
        for p in self.places + self.shacks:
            if (p.gx * CELL - margin < x < (p.gx + p.w) * CELL + margin
                    and p.gz * CELL - margin < z < (p.gz + p.d) * CELL + margin):
                return True
        return False

    def _scatter(self, rng: np.random.Generator) -> None:
        """Trees, rocks, a few wrecks, and the loot that lies in the open."""
        self.paths = self._paths(rng)
        taken: dict[tuple, bool] = {}

        def free(x: float, z: float, cell: float = 3.2) -> bool:
            key = (int(x / cell), int(z / cell))
            if key in taken:
                return False
            taken[key] = True
            return True

        def on_path(x: float, z: float, w: float = 3.5) -> bool:
            for ax, az, bx, bz in self.paths:
                dx, dz = bx - ax, bz - az
                t = min(max(((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz + 1e-6), 0.0), 1.0)
                if (x - ax - t * dx) ** 2 + (z - az - t * dz) ** 2 < w * w:
                    return True
            return False

        count = 0
        for _ in range(9000):
            if count >= 520:
                break
            x, z = float(rng.uniform(20, SIZE - 20)), float(rng.uniform(20, SIZE - 20))
            h = self._h(x, z)
            if h < 1.7:
                continue
            fi, fj = int(x / CELL), int(z / CELL)
            dens = float(self.forest[fj, fi])
            slope = 1.0 - float(self.normals[fj, fi, 1])
            if slope > 0.16 or self._near_built(x, z, 2.5) or on_path(x, z):
                continue
            want = 0.06 + 1.5 * max(dens - 0.46, 0.0) * 4.0
            if h < 3.0:
                want = 0.10
            if rng.random() > want or not free(x, z):
                continue
            kind = "palm" if h < 3.1 else ("pine" if dens > 0.56 or h > 13 else "oak")
            self._prop(kind, x, z, float(rng.uniform(0, math.tau)), float(rng.uniform(0.82, 1.25)), rng=rng)
            count += 1
        count = 0
        for _ in range(4000):
            if count >= 110:
                break
            x, z = float(rng.uniform(16, SIZE - 16)), float(rng.uniform(16, SIZE - 16))
            h = self._h(x, z)
            if h < 0.6 or self._near_built(x, z, 3.0) or on_path(x, z, 4.0):
                continue
            want = 0.03 + (0.25 if h > 15 else 0.0) + (0.08 if h < 2.0 else 0.0)
            if rng.random() > want or not free(x, z, 4.0):
                continue
            self._prop("rock", x, z, float(rng.uniform(0, math.tau)), float(rng.uniform(0.7, 1.5)), rng=rng)
            count += 1
        # wrecks along the paths (metal away from the works and the scrapyard) and finds beside them
        for i, (ax, az, bx, bz) in enumerate(self.paths):
            if i % 3 != 1:
                continue
            t = float(rng.uniform(0.3, 0.7))
            nx, nz = -(bz - az), bx - ax
            ln = math.hypot(nx, nz) + 1e-6
            x, z = ax + (bx - ax) * t + nx / ln * 4.2, az + (bz - az) * t + nz / ln * 4.2
            if self._h(x, z) > 1.6 and not self._near_built(x, z, 3.0) and free(x, z, 5.0):
                self._prop("car", x, z, 0.0 if abs(bx - ax) < abs(bz - az) else math.pi / 2, rng=rng)
                self.kit.floor_loot.append((x + 2.6, self._h(x + 2.6, z + 1.0) + 0.06, z + 1.0))
        # loot in the open: by trees and rocks, so the wilds are worth crossing
        n = 0
        for p in self.props:
            if p.kind in ("oak", "rock", "pine") and rng.random() < 0.12 and n < 72:
                a = float(rng.uniform(0, math.tau))
                x, z = p.x + math.cos(a) * 2.4, p.z + math.sin(a) * 2.4
                if self._h(x, z) > 1.6 and not self._near_built(x, z, 1.0):
                    if n % 3 == 0:
                        self.kit.chests.append((x, self._h(x, z) + 0.06, z, a + math.pi / 2))
                    elif n % 3 == 1:
                        self.kit.floor_loot.append((x, self._h(x, z) + 0.06, z))
                    else:
                        self.kit.ammo.append((x, self._h(x, z) + 0.06, z))
                    n += 1
        # looks only: bushes, flowers, reeds by the pond
        for _ in range(900):
            x, z = float(rng.uniform(20, SIZE - 20)), float(rng.uniform(20, SIZE - 20))
            h = self._h(x, z)
            if h < 1.9 or self._near_built(x, z, 1.0) or on_path(x, z, 2.5):
                continue
            kind = "bush" if rng.random() < 0.45 else ("flower" if rng.random() < 0.7 else "tuft")
            self.decor.append((kind, x, h, z, float(rng.uniform(0, math.tau)), float(rng.uniform(0.7, 1.3)),
                               float(rng.random())))

    def _paths(self, rng: np.random.Generator) -> list[tuple]:
        P = {p.kind: p for p in self.places}
        links = (("quay", "market"), ("market", "farm"), ("market", "scrap"), ("market", "works"), ("farm", "mill"),
                 ("scrap", "fort"), ("works", "beach"), ("farm", "peak"), ("fort", "mill"), ("quay", "beach"),
                 ("works", "peak"), ("quay", "scrap"))
        segs: list[tuple] = []
        for a, b in links:
            ax, az, bx, bz = P[a].cx, P[a].cz, P[b].cx, P[b].cz
            mx, mz = (ax + bx) / 2 + float(rng.uniform(-22, 22)), (az + bz) / 2 + float(rng.uniform(-22, 22))
            pts = []
            for i in range(9):  # a quadratic curve in eight pieces
                t = i / 8
                pts.append(((1 - t) ** 2 * ax + 2 * t * (1 - t) * mx + t * t * bx,
                            (1 - t) ** 2 * az + 2 * t * (1 - t) * mz + t * t * bz))
            segs += [(*p0, *p1) for p0, p1 in zip(pts, pts[1:])]
        return segs

    # --- the painted ground -------------------------------------------------------------------------------------------

    def _paint(self, rng: np.random.Generator) -> None:
        T = TEX
        h = _upsample(self.heights, T)
        gz_, gx_ = np.gradient(h, SIZE / T)
        slope = np.hypot(gx_, gz_)
        n1 = _fbm(rng, T, 10, 3)
        n2 = _fbm(rng, T, 48, 2)
        n3 = _vnoise(rng, T, 300)
        forest = _upsample(self.forest, T)

        def col(c: tuple) -> np.ndarray:
            return np.array(c, F32)[None, None, :]

        grass = col((112, 196, 84)) * (1 - n1[..., None]) + col((74, 164, 80)) * n1[..., None]
        meadow = _smooth(0.55, 0.75, n2)[..., None]
        grass = grass * (1 - 0.5 * meadow) + col((176, 204, 92)) * 0.5 * meadow
        wood_floor = _smooth(0.52, 0.64, forest)[..., None]
        grass = grass * (1 - 0.55 * wood_floor) + col((62, 138, 74)) * 0.55 * wood_floor
        sand = col((240, 218, 156)) * (1 - 0.25 * n2[..., None]) + col((222, 196, 134)) * 0.25 * n2[..., None]
        rgb = sand * (1 - _smooth(1.5, 2.7, h + (n2 - 0.5) * 0.8)[..., None]) \
            + grass * _smooth(1.5, 2.7, h + (n2 - 0.5) * 0.8)[..., None]
        rock = col((158, 152, 148)) * (1 - 0.3 * n2[..., None]) + col((120, 116, 118)) * 0.3 * n2[..., None]
        rk = np.maximum(_smooth(0.55, 0.95, slope), _smooth(19, 27, h + (n1 - 0.5) * 6))[..., None]
        rgb = rgb * (1 - rk) + rock * rk
        cap = _smooth(33, 39, h + (n2 - 0.5) * 3)[..., None]
        rgb = rgb * (1 - cap) + col((236, 238, 242)) * cap
        wet = (1 - _smooth(0.0, 0.7, h))[..., None]
        rgb = rgb * (1 - 0.3 * wet) + col((176, 160, 116)) * 0.3 * wet
        deep = _smooth(0.0, 5.0, -h)[..., None]
        under = (h < 0)[..., None]
        sea = col((190, 214, 170)) * (1 - deep) + col((30, 104, 150)) * deep
        rgb = np.where(under, sea, rgb)
        # paths: dirt lanes between the places (worked out at half size, then doubled)
        half = T // 2
        u = (np.arange(half, dtype=F32) + 0.5) * (SIZE / half)
        X, Z = np.meshgrid(u, u)
        best = np.full((half, half), 1e9, F32)
        for ax, az, bx, bz in self.paths:
            x0, x1, z0, z1 = min(ax, bx) - 8, max(ax, bx) + 8, min(az, bz) - 8, max(az, bz) + 8
            i0, i1 = max(int(x0 / SIZE * half), 0), min(int(x1 / SIZE * half) + 1, half)
            j0, j1 = max(int(z0 / SIZE * half), 0), min(int(z1 / SIZE * half) + 1, half)
            xs, zs = X[j0:j1, i0:i1], Z[j0:j1, i0:i1]
            dx, dz = bx - ax, bz - az
            t = np.clip(((xs - ax) * dx + (zs - az) * dz) / (dx * dx + dz * dz + 1e-6), 0, 1)
            d = np.hypot(xs - ax - t * dx, zs - az - t * dz)
            best[j0:j1, i0:i1] = np.minimum(best[j0:j1, i0:i1], d)
        pd = np.repeat(np.repeat(best, 2, axis=0), 2, axis=1) + (n3 - 0.5) * 1.4
        lane = ((1 - _smooth(1.5, 2.9, pd)) * (h > 0.9))[..., None]
        dirt = col((206, 172, 118)) * (1 - 0.3 * n2[..., None]) + col((178, 142, 96)) * 0.3 * n2[..., None]
        rgb = rgb * (1 - lane * 0.92) + dirt * lane * 0.92
        # pads: each place's own ground
        tx = (np.arange(T, dtype=F32) + 0.5) * (SIZE / T)
        TXX, TZZ = np.meshgrid(tx, tx)
        for p in self.places + self.shacks:
            x0, x1, z0, z1 = p.gx * CELL, (p.gx + p.w) * CELL, p.gz * CELL, (p.gz + p.d) * CELL
            m = 7
            i0, i1 = int((x0 - m) / SIZE * T), int((x1 + m) / SIZE * T) + 1
            j0, j1 = int((z0 - m) / SIZE * T), int((z1 + m) / SIZE * T) + 1
            xs, zs = TXX[j0:j1, i0:i1], TZZ[j0:j1, i0:i1]
            out = np.hypot(np.maximum(np.maximum(x0 - xs, xs - x1), 0), np.maximum(np.maximum(z0 - zs, zs - z1), 0))
            wgt = (1 - _smooth(0.0, 3.5, out + (n3[j0:j1, i0:i1] - 0.5) * 3.0))[..., None]
            base = col(p.paving if p.kind != "shack" else (196, 166, 116))
            tone = base * (0.9 + 0.2 * n2[j0:j1, i0:i1][..., None])
            if p.kind in ("quay", "works", "market", "peak", "scrap", "fort"):  # paving stones
                gx_l = ((xs / 2.0) % 1.0)
                gz_l = ((zs / 2.0) % 1.0)
                joint = ((gx_l < 0.06) | (gz_l < 0.06))[..., None]
                tone = np.where(joint, tone * 0.82, tone)
            rgb[j0:j1, i0:i1] = rgb[j0:j1, i0:i1] * (1 - wgt) + tone * wgt
        # the farm's fields: rows of crops west and north of the yard
        farm = next(p for p in self.places if p.kind == "farm")
        for fx0, fz0, fw, fd, c1, c2, along in (
                (farm.cx - 62, farm.cz - 14, 34, 30, (150, 196, 70), (112, 160, 62), 0),
                (farm.cx - 18, farm.cz + 24, 40, 26, (226, 196, 92), (196, 160, 70), 1)):
            i0, i1 = int(fx0 / SIZE * T), int((fx0 + fw) / SIZE * T)
            j0, j1 = int(fz0 / SIZE * T), int((fz0 + fd) / SIZE * T)
            xs, zs = TXX[j0:j1, i0:i1], TZZ[j0:j1, i0:i1]
            rows = (((zs if along else xs) / 1.6) % 1.0 < 0.5)[..., None]
            edge = np.minimum(np.minimum(xs - fx0, fx0 + fw - xs), np.minimum(zs - fz0, fz0 + fd - zs))
            wgt = (_smooth(0.0, 1.5, edge) * (h[j0:j1, i0:i1] > 2.0) * (lane[j0:j1, i0:i1, 0] < 0.3))[..., None]
            tone = np.where(rows, col(c1), col(c2))
            rgb[j0:j1, i0:i1] = rgb[j0:j1, i0:i1] * (1 - wgt * 0.9) + tone * wgt * 0.9
        rgb *= (0.94 + 0.12 * n3)[..., None]
        # shade (alpha): soft blobs under trees and rocks, a slanted block behind each building
        shade = np.ones((T, T), F32)
        k = T / SIZE
        sx, sz = -SUN[0] / SUN[1], -SUN[2] / SUN[1]
        yy, xx = np.mgrid[-12:13, -12:13].astype(F32)
        for p in self.props:
            if p.kind in ("pine", "oak", "palm"):
                r, reach, dark = 2.3 * p.scale, 2.4 * p.scale, 0.36
            elif p.kind in ("rock", "car", "container", "tank", "hay", "logs", "machine", "stall", "crate", "barrel",
                            "pillar"):
                r, reach, dark = max(p.a, p.b, 0.6) * 1.25, p.h * 0.45, 0.34
            else:
                continue
            cx, cz = (p.x + sx * reach) * k, (p.z + sz * reach) * k
            ic, jc = int(cx), int(cz)
            if not (13 <= ic < T - 13 and 13 <= jc < T - 13):
                continue
            rr = r * k
            blob = 1 - dark * np.clip(1.25 - np.hypot(xx - (cx - ic), yy - (cz - jc)) / rr, 0, 1) ** 1.0
            shade[jc - 12:jc + 13, ic - 12:ic + 13] *= np.minimum(blob, 1.0)
        for x0, z0, x1, z1, hh in self.kit.shadows:
            ox, oz = sx * hh * 0.55, sz * hh * 0.55
            i0, i1 = int((min(x0, x0 + ox) - 1) * k), int((max(x1, x1 + ox) + 1) * k)
            j0, j1 = int((min(z0, z0 + oz) - 1) * k), int((max(z1, z1 + oz) + 1) * k)
            shade[j0:j1, i0:i1] *= 0.66
        for _ in range(3):  # soften
            shade = (shade + np.roll(shade, 1, 0) + np.roll(shade, -1, 0) + np.roll(shade, 1, 1)
                     + np.roll(shade, -1, 1)) / 5.0
        self.ground = np.concatenate([np.clip(rgb, 0, 255), np.clip(shade * 255, 0, 255)[..., None]],
                                     axis=-1).astype(np.uint8)
        # the minimap: the same paint with sea, relief and roofs, north up
        M = 512
        small = rgb.reshape(M, 2, M, 2, 3).mean(axis=(1, 3))
        hs = h.reshape(M, 2, M, 2).mean(axis=(1, 3))
        gz2, gx2 = np.gradient(hs, SIZE / M)
        relief = np.clip(1.0 + (-gx2 * 0.5 - gz2 * 0.5) * 0.9, 0.7, 1.3)[..., None]
        small = small * relief
        depth = _smooth(0.0, 6.0, -hs)[..., None]
        water = col((84, 190, 214)) * (1 - depth) + col((36, 120, 190)) * depth
        small = np.where((hs < 0)[..., None], water, small)
        roof_cols = {WOOD: (196, 120, 70), BRICK: (188, 92, 76), METAL: (130, 150, 170)}
        km = M / SIZE
        for (kind, gx, gy, gz, _d, mat, _v, _a) in self.kit.pieces.values():
            if kind in (ROOF, FLOOR):
                i0, i1 = int(gx * CELL * km), int((gx + 1) * CELL * km)
                j0, j1 = int(gz * CELL * km), int((gz + 1) * CELL * km)
                small[j0:j1, i0:i1] = roof_cols[mat]
                small[j0:j1, i0] = np.array(roof_cols[mat]) * 0.7
                small[j0, i0:i1] = np.array(roof_cols[mat]) * 0.7
        self.map_rgb = np.ascontiguousarray(np.clip(small, 0, 255).astype(np.uint8)[::-1])

    # --- helpers the match uses ---------------------------------------------------------------------------------------

    def on_land(self, x: float, z: float) -> bool:
        return 8 < x < SIZE - 8 and 8 < z < SIZE - 8 and self.height(x, z) > 0.4

    def place_at(self, x: float, z: float) -> str:
        best, name = 1e9, ""
        for p in self.places:
            d = math.hypot(x - p.cx, z - p.cz)
            if d < p.radius + 14 and d < best:
                best, name = d, p.name
        return name
