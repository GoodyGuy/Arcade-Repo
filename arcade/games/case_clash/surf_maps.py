"""Case Clash's surf maps, without Qt: a small kit (ramps swept along a ridge line, platforms, zones) and three
courses of the game's own design, themed on its weapon cases.

A ramp is a triangular prism: a ridge line on top and one or two faces falling away from it at the ramp's slope
(steeper than 45.6 degrees, so they are surfed, never stood on). It is built from one convex solid per piece of the
ridge; the pieces share their end points exactly, so there are no gaps or steps between them. Every map is staged:
each stage starts on a flat platform (a checkpoint), a fall sends the player back to the platform of the stage,
and the clock keeps running (except in the first stage, where a fall restarts the run)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.arcade.games.case_clash.surf_physics import HULL, World, hull_planes

F32 = np.float32
# materials (the shader draws each its own way)
M_FACE, M_UNDER, M_PLAT, M_PILLAR, M_DECOR, M_FLOOR = 0, 1, 2, 3, 4, 5
VERTEX_FLOATS = 10  # position, normal, uv, material, slant length of the ramp face
RIDE = 0.45  # how far down a face (as a share of its width) the course expects the player to ride
UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Theme:
    sky_top: tuple
    sky_horizon: tuple
    sea: tuple  # the colour far below (the fall)
    sun: tuple  # direction to the sun
    sun_color: tuple
    ambient: float
    fog: float  # density per unit
    face: tuple  # ramp faces
    trim: tuple  # the ridge trim (readable at speed)
    under: tuple  # ramp ends and undersides
    plat_a: tuple
    plat_b: tuple
    pillar: tuple
    decor: tuple
    grid: float  # how strong the face grid is
    glow: float  # neon: the grid and trims glow
    stars: float
    clouds: float
    cloud_color: tuple


THEMES = {
    "dune": Theme(sky_top=(0.23, 0.42, 0.78), sky_horizon=(0.98, 0.78, 0.55), sea=(0.86, 0.66, 0.42),
                  sun=(0.55, -0.45, 0.42), sun_color=(1.0, 0.86, 0.66), ambient=0.62, fog=1.0 / 52000,
                  face=(0.86, 0.76, 0.60), trim=(0.80, 0.20, 0.12), under=(0.52, 0.40, 0.30),
                  plat_a=(0.78, 0.66, 0.48), plat_b=(0.66, 0.54, 0.38), pillar=(0.80, 0.68, 0.52),
                  decor=(0.80, 0.56, 0.36), grid=0.10, glow=0.0, stars=0.0, clouds=0.55,
                  cloud_color=(1.0, 0.93, 0.85)),
    "glacier": Theme(sky_top=(0.30, 0.52, 0.86), sky_horizon=(0.82, 0.90, 0.98), sea=(0.86, 0.92, 0.98),
                     sun=(-0.35, 0.6, 0.55), sun_color=(1.0, 0.98, 0.94), ambient=0.64, fog=1.0 / 46000,
                     face=(0.88, 0.93, 0.98), trim=(0.16, 0.48, 0.92), under=(0.45, 0.58, 0.72),
                     plat_a=(0.80, 0.88, 0.95), plat_b=(0.62, 0.74, 0.86), pillar=(0.70, 0.82, 0.92),
                     decor=(0.76, 0.86, 0.95), grid=0.12, glow=0.0, stars=0.0, clouds=0.85,
                     cloud_color=(1.0, 1.0, 1.0)),
    "neon": Theme(sky_top=(0.02, 0.01, 0.06), sky_horizon=(0.20, 0.05, 0.28), sea=(0.03, 0.01, 0.07),
                  sun=(0.3, 0.8, 0.25), sun_color=(0.75, 0.55, 1.0), ambient=0.5, fog=1.0 / 60000,
                  face=(0.07, 0.07, 0.11), trim=(0.10, 0.95, 1.00), under=(0.05, 0.04, 0.08),
                  plat_a=(0.10, 0.10, 0.16), plat_b=(0.06, 0.06, 0.10), pillar=(0.08, 0.07, 0.12),
                  decor=(0.07, 0.06, 0.12), grid=0.9, glow=1.0, stars=1.0, clouds=0.0,
                  cloud_color=(0.6, 0.2, 0.8)),
}


@dataclass
class Ramp:
    """A ramp the course is meant to be ridden on: its ridge joints and which face (side +1 = the face on the left
    of the ridge looking along it)."""
    ridge: np.ndarray
    side: int
    height: float
    width: float


@dataclass
class Zone:
    """A rotated box over a platform: centre, heading (radians), half length, half width, floor height."""
    centre: tuple
    heading: float
    half: tuple
    floor: float
    top: float = 220.0

    def contains(self, p: tuple) -> bool:
        dx, dy = p[0] - self.centre[0], p[1] - self.centre[1]
        c, s = math.cos(self.heading), math.sin(self.heading)
        a, b = dx * c + dy * s, -dx * s + dy * c
        foot = p[2] - HULL[2]
        return abs(a) <= self.half[0] and abs(b) <= self.half[1] and self.floor - 8 <= foot <= self.floor + self.top

    def corners(self, z0: float, z1: float) -> np.ndarray:
        c, s = math.cos(self.heading), math.sin(self.heading)
        out = []
        for z in (z0, z1):
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                x, y = a * self.half[0], b * self.half[1]
                out.append((self.centre[0] + x * c - y * s, self.centre[1] + x * s + y * c, z))
        return np.array(out)


@dataclass
class Stage:
    spawn: tuple  # box centre
    yaw: float  # radians
    zone: Zone
    ramps: list = field(default_factory=list)
    kill_z: float = 0.0


@dataclass
class SurfMap:
    id: str
    name: str
    tier: int
    blurb: str
    theme_id: str
    case_id: str  # the case the first clear drops
    par: float  # s: finishing under it pays
    gold: float  # s: a fast time pays double
    reward: int  # cents for a run under par
    stages: list
    end: Zone
    solids: list
    mesh: np.ndarray  # (n, VERTEX_FLOATS) float32 triangles
    zones_mesh: list  # per zone (stage zones, then the end): (n, 3) float32 triangles of its walls
    sea_z: float
    bounds: tuple
    _world: World | None = None

    @property
    def theme(self) -> Theme:
        return THEMES[self.theme_id]

    @property
    def world(self) -> World:
        if self._world is None:
            self._world = World(self.solids)
        return self._world

    def zone_list(self) -> list:
        return [s.zone for s in self.stages] + [self.end]


# --- building ------------------------------------------------------------------------------------------------------

def _fwd(h: float) -> np.ndarray:
    return np.array([math.cos(h), math.sin(h), 0.0])


def _left(h: float) -> np.ndarray:
    return np.array([-math.sin(h), math.cos(h), 0.0])


def ridge_path(start: np.ndarray, heading: float, pieces: list, step: float = 220.0) -> tuple[np.ndarray, float]:
    """Ridge joints from ``start``: each piece is (length, turn in degrees, drop); a turning piece is cut into
    short straight bits (a few degrees each)."""
    pts = [np.asarray(start, float)]
    h = heading
    for length, turn, drop in pieces:
        n = max(1, math.ceil(length / step)) if turn else 1
        dt, dl, dz = math.radians(turn) / n, length / n, drop / n
        for _ in range(n):
            h += dt / 2
            pts.append(pts[-1] + _fwd(h) * dl - UP * dz)
            h += dt / 2
    return np.array(pts), h


class Course:
    """Builds a staged map piece by piece: ``start`` a platform, then ``ramp``s and ``land``ings."""

    def __init__(self, height: float, slope: float, seed: int, speed: float = 800.0) -> None:
        self.H, self.slope = height, slope
        self.speed = speed  # the slowest speed a gap is built for (u/s)
        self.solids: list[np.ndarray] = []
        self.parts: list[tuple] = []  # (points, attribute table or None, stage index) for the mesh
        self.decor: list[tuple] = []  # render only: (points, material)
        self.stages: list[Stage] = []
        self.end: Zone | None = None
        self.rng = np.random.default_rng(seed)
        self.h = 0.0
        self.line = np.zeros(3)  # where the expected riding line meets the face level at the course's end
        self.plat: tuple | None = None  # (far edge centre, top) of the platform the next ramp starts from

    def drop_for(self, gap: float, drop: float) -> float:
        """At least the fall across ``gap`` at the course's design speed (leaving a ramp already sinking at about
        350 u/s), plus room: arriving lower than the next face would hit its end wall."""
        t = gap / self.speed
        return max(drop, 350.0 * t + 400.0 * t * t + 120.0)

    def width(self, height: float | None = None, slope: float | None = None) -> float:
        return (height or self.H) / math.tan(math.radians(slope or self.slope))

    def _platform(self, centre: np.ndarray, top: float, h: float, length: float, width: float) -> Zone:
        zone = Zone((float(centre[0]), float(centre[1])), h, (length / 2, width / 2), top)
        pts = zone.corners(top - 96.0, top)
        self.solids.append(pts)
        self.parts.append((pts, None, len(self.stages)))
        return zone

    def start(self, centre: tuple, top: float, heading: float, size: tuple = (700.0, 700.0)) -> None:
        h = math.radians(heading)
        c = np.array([centre[0], centre[1], top])
        zone = self._platform(c, top, h, *size)
        self._new_stage(c, top, h, zone)
        self.plat = (c + _fwd(h) * size[0] / 2, top)

    def _new_stage(self, c: np.ndarray, top: float, h: float, zone: Zone) -> None:
        self.stages.append(Stage((float(c[0]), float(c[1]), top + HULL[2] + 1.0), h, zone))
        self.h = h

    def ramp(self, pieces: list, side: int, gap: float = 0.0, drop: float = 0.0, shift: float = 0.0,
             height: float | None = None, slope: float | None = None) -> None:
        H = height or self.H  # noqa: N806
        W = self.width(H, slope)  # noqa: N806
        h = self.h
        if self.plat is not None:
            edge, top = self.plat
            meet = edge + _fwd(h) * 60.0
            meet[2] = top - 210.0
            self.plat = None
        else:
            meet = self.line + _fwd(h) * gap + _left(h) * shift - UP * self.drop_for(gap, drop)
        start = meet - side * _left(h) * RIDE * W + UP * RIDE * H
        ridge, h_end = ridge_path(start, h, pieces)
        self._ramp_solids(ridge, H, W, side)
        self.stages[-1].ramps.append(Ramp(ridge, side, H, W))
        self.h = h_end
        self.line = ridge[-1] + side * _left(h_end) * RIDE * W - UP * RIDE * H

    def _ramp_solids(self, ridge: np.ndarray, H: float, W: float, side: int) -> None:  # noqa: N803
        n = len(ridge)
        dirs = []
        for i in range(n - 1):
            d = ridge[i + 1] - ridge[i]
            d[2] = 0.0
            dirs.append(d / np.linalg.norm(d))
        lefts, scales = [], []
        for i in range(n):
            a = dirs[max(i - 1, 0)]
            b = dirs[min(i, n - 2)]
            t = a + b
            t /= np.linalg.norm(t)
            lefts.append(np.array([-t[1], t[0], 0.0]))
            scales.append(1.0 / max(float(t @ b), 0.5))  # mitred joints keep the slope of both pieces
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ridge, axis=0), axis=1))])
        slant = math.hypot(H, W)
        bottom_l = [ridge[i] + lefts[i] * W * scales[i] - UP * H for i in range(n)]
        bottom_r = [ridge[i] - lefts[i] * W * scales[i] - UP * H for i in range(n)]
        for i in range(n - 1):
            pts = np.array([ridge[i], ridge[i + 1], bottom_l[i], bottom_l[i + 1], bottom_r[i], bottom_r[i + 1]])
            uv = [(along[i], 0.0), (along[i + 1], 0.0), (along[i], slant), (along[i + 1], slant),
                  (along[i], slant), (along[i + 1], slant)]
            self.solids.append(pts)
            self.parts.append((pts, (uv, slant), len(self.stages) - 1))
            if i % 3 == 1:  # a column holding the ramp up, into the depths (seen, never hit)
                c = (ridge[i] + ridge[i + 1]) / 2 - UP * (H * 0.9)
                self._column(c, min(W * 0.28, 110.0), 5200.0)

    def _column(self, top: np.ndarray, r: float, length: float) -> None:
        pts = []
        for z in (top[2] - length, top[2]):
            for k in range(8):
                a = k * math.pi / 4 + math.pi / 8
                pts.append((top[0] + math.cos(a) * r, top[1] + math.sin(a) * r, z))
        self.decor.append((np.array(pts), M_PILLAR))

    def land(self, gap: float, drop: float, length: float = 1400.0, width: float = 1100.0, turn: float = 0.0,
             shift: float = 0.0, final: bool = False) -> None:
        h = self.h
        meet = self.line + _fwd(h) * gap + _left(h) * shift - UP * self.drop_for(gap, drop)
        top = float(meet[2])
        h2 = h + math.radians(turn)
        centre = meet + _fwd(h) * length * 0.5
        zone = self._platform(centre, top, h2, length, width)
        if final:
            self.end = zone
            return
        self._new_stage(centre, top, h2, zone)
        self.plat = (centre + _fwd(h2) * length / 2, top)

    def scenery(self, kind: str, count: int, spread: tuple) -> None:
        """Big shapes far around the course, lost in the haze (seen, never hit)."""
        lo, hi = self.bounds()
        mid = (lo + hi) / 2
        for _ in range(count):
            a = self.rng.uniform(0, math.tau)
            dist = self.rng.uniform(*spread)
            x, y = mid[0] + math.cos(a) * dist, mid[1] + math.sin(a) * dist
            base = self.sea_z()
            size = self.rng.uniform(1800, 5200)
            tall = self.rng.uniform(0.6, 1.6) * size
            if kind == "mesa":
                top = base + tall * 1.4
                pts = [(x + math.cos(k * 1.2566) * size, y + math.sin(k * 1.2566) * size, base) for k in range(5)]
                pts += [(x + math.cos(k * 1.2566 + 0.3) * size * 0.7, y + math.sin(k * 1.2566 + 0.3) * size * 0.7,
                         top) for k in range(5)]
            elif kind == "spike":
                top = base + tall * 2.4
                pts = [(x + math.cos(k * 2.094) * size * 0.6, y + math.sin(k * 2.094) * size * 0.6, base)
                       for k in range(3)]
                pts += [(x + self.rng.uniform(-200, 200), y + self.rng.uniform(-200, 200), top)]
            else:  # tower
                top = base + tall * 3.0
                r = size * 0.35
                pts = [(x + dx * r, y + dy * r, z) for z in (base, top) for dx, dy in ((-1, -1), (1, -1), (1, 1),
                                                                                      (-1, 1))]
            self.decor.append((np.array(pts, float), M_DECOR))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        allp = np.concatenate(self.solids)
        return allp.min(0), allp.max(0)

    def sea_z(self) -> float:
        return float(self.bounds()[0][2]) - 2600.0

    def finish(self, **info) -> SurfMap:  # noqa: ANN003
        lo, hi = self.bounds()
        sea = self.sea_z()
        for i, st in enumerate(self.stages):  # a fall: under the stage's lowest part
            z = [float(p[:, 2].min()) for p, _a, k in self.parts if k == i]
            nxt = self.stages[i + 1].zone.floor if i + 1 < len(self.stages) else self.end.floor
            st.kill_z = min(min(z), nxt) - 420.0
        mesh = _mesh(self.parts, self.decor, sea, lo, hi)
        zones = [_zone_walls(z) for z in [s.zone for s in self.stages] + [self.end]]
        return SurfMap(stages=self.stages, end=self.end, solids=self.solids, mesh=mesh, zones_mesh=zones,
                       sea_z=sea, bounds=(tuple(lo), tuple(hi)), **info)


# --- meshes --------------------------------------------------------------------------------------------------------

def _faces(pts: np.ndarray) -> list[tuple[np.ndarray, list[int]]]:
    """The faces of a convex solid: (outward normal, its corner indices in order)."""
    out = []
    for n, d in hull_planes(pts):
        idx = [i for i in range(len(pts)) if abs(float(pts[i] @ n) - d) < 0.05]
        if len(idx) < 3:
            continue
        c = pts[idx].mean(0)
        u = pts[idx[0]] - c
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        idx.sort(key=lambda i: math.atan2(float((pts[i] - c) @ v), float((pts[i] - c) @ u)))
        out.append((n, idx))
    return out


def _mesh(parts: list, decor: list, sea: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []

    def tri(a: tuple, b: tuple, c: tuple) -> None:
        rows.extend((a, b, c))

    for pts, info, _stage in parts:
        for n, idx in _faces(pts):
            vs = []
            for i in idx:
                p = pts[i]
                if info is not None and 0.05 < n[2] < 0.75:  # a surfable face of a ramp
                    (u, v), slant = info[0][i], info[1]
                    vs.append((*p, *n, u, v, M_FACE, slant))
                elif info is not None:
                    vs.append((*p, *n, p[0] + p[1], p[2], M_UNDER, 0.0))
                else:  # a platform
                    vs.append((*p, *n, p[0], p[1] if abs(n[2]) > 0.5 else p[2], M_PLAT, 1.0 if n[2] > 0.5 else 0.0))
            for k in range(1, len(vs) - 1):
                tri(vs[0], vs[k], vs[k + 1])
    for pts, mat in decor:
        for n, idx in _faces(pts):
            vs = [(*pts[i], *n, pts[i][0] + pts[i][1], pts[i][2], mat, 0.0) for i in idx]
            for k in range(1, len(vs) - 1):
                tri(vs[0], vs[k], vs[k + 1])
    # the floor far below, out to the haze
    r = 160000.0
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    q = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    fl = [(x, y, sea, 0.0, 0.0, 1.0, x, y, M_FLOOR, 0.0) for x, y in q]
    tri(fl[0], fl[1], fl[2])
    tri(fl[0], fl[2], fl[3])
    return np.array(rows, F32)


def _zone_walls(zone: Zone) -> np.ndarray:
    """The four glowing walls of a zone (x, y, z, height share 0..1)."""
    c = zone.corners(zone.floor + 2.0, zone.floor + 150.0)
    rows = []
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        at, bt = c[k + 4], c[(k + 1) % 4 + 4]
        quad = [(*a, 0.0), (*b, 0.0), (*bt, 1.0), (*a, 0.0), (*bt, 1.0), (*at, 1.0)]
        rows.extend(quad)
    return np.array(rows, F32)


# --- the maps ------------------------------------------------------------------------------------------------------

MAP_ORDER = ("dune", "glacier", "neon")
# what the menus show without building a map; par / gold from a scripted surfer's clean runs in the game
# (25.9 s, 36.5 s, 38.9 s): gold is a little slower than that, par leaves plenty of room
INFO = {
    "dune": dict(name="Dune Drift", tier=1, theme_id="dune", case_id="sandstorm", par=60.0, gold=32.0, reward=150,
                 blurb="Wide sandstone ramps over a sunset desert. Learn to hold a ramp: strafe into it and look "
                       "along it."),
    "glacier": dict(name="Glacier Run", tier=3, theme_id="glacier", case_id="glacier", par=85.0, gold=42.0,
                    reward=350, blurb="Ice-blue ramps above the clouds. Ninety-degree bends and transfers from one "
                                      "face to the other."),
    "neon": dict(name="Neon Spiral", tier=4, theme_id="neon", case_id="neon", par=95.0, gold=44.0, reward=700,
                 blurb="Narrow glowing rails over a night city. A falling spiral, long gaps and sideways "
                       "transfers."),
}


def meta(map_id: str) -> dict:
    return dict(id=map_id, **INFO[map_id])


def _dune_drift() -> SurfMap:
    """Tier 1: wide, gently falling ramps, short gaps, one long sweeping bend; the same face all the way."""
    c = Course(height=700.0, slope=55.0, seed=11, speed=650.0)
    c.start((0.0, 0.0), 9000.0, 0.0)
    c.ramp([(3200, 0, 520)], side=1)
    c.ramp([(2800, 0, 460)], side=1, gap=420, drop=330)
    c.land(gap=420, drop=330, turn=0.0)
    c.ramp([(700, 0, 110), (2600, 70, 470), (700, 0, 110)], side=-1)
    c.land(gap=420, drop=330)
    c.ramp([(2600, -45, 430)], side=1)
    c.ramp([(2600, 0, 430)], side=1, gap=450, drop=340)
    c.land(gap=450, drop=340, length=1800, final=True)
    c.scenery("mesa", 26, (16000, 52000))
    return c.finish(**meta("dune"))


def _glacier_run() -> SurfMap:
    """Tier 3: steeper and longer, bends of 90 degrees, switching faces across the gaps, longer flights."""
    c = Course(height=760.0, slope=58.0, seed=23, speed=850.0)
    c.start((0.0, 0.0), 12000.0, 0.0)
    c.ramp([(2600, 0, 420)], side=1)
    c.ramp([(900, 0, 140), (2600, 90, 480)], side=-1, gap=640, drop=420)
    c.land(gap=560, drop=380)
    c.ramp([(1100, 0, 170), (2700, -90, 470)], side=1)
    c.ramp([(2800, 0, 460)], side=1, gap=760, drop=470)
    c.land(gap=600, drop=400, turn=30.0)
    c.ramp([(2000, 50, 330)], side=-1)
    c.ramp([(2300, -70, 380)], side=1, gap=720, drop=460)
    c.ramp([(2400, 0, 400)], side=1, gap=780, drop=480)
    c.land(gap=640, drop=420, length=1800, final=True)
    c.scenery("spike", 30, (15000, 52000))
    return c.finish(**meta("glacier"))


def _neon_spiral() -> SurfMap:
    """Tier 4: narrow ramps, a falling spiral, long gaps with a sideways transfer, tight landings."""
    c = Course(height=560.0, slope=60.0, seed=37, speed=1000.0)
    c.start((0.0, 0.0), 15000.0, 0.0)
    c.ramp([(900, 0, 140), (4200, 180, 700)], side=-1)
    c.ramp([(1100, 0, 180), (3000, 120, 480)], side=-1, gap=880, drop=520)
    c.land(gap=700, drop=440)
    c.ramp([(1000, 0, 160), (3200, -150, 520)], side=1)
    c.ramp([(2400, 90, 400)], side=-1, gap=950, drop=540, shift=-160)
    c.ramp([(2300, -90, 380)], side=1, gap=950, drop=540, shift=160)
    c.land(gap=700, drop=440)
    c.ramp([(800, 0, 120), (4300, 270, 760)], side=-1)
    c.ramp([(2600, 0, 420)], side=-1, gap=1050, drop=580)
    c.land(gap=750, drop=460, length=1800, final=True)
    c.scenery("tower", 40, (14000, 50000))
    return c.finish(**meta("neon"))


_MAPS: dict[str, SurfMap] = {}
_BUILDERS = {"dune": _dune_drift, "glacier": _glacier_run, "neon": _neon_spiral}


def get_map(map_id: str) -> SurfMap:
    """The map (built the first time it is asked for)."""
    m = _MAPS.get(map_id)
    if m is None:
        m = _MAPS[map_id] = _BUILDERS[map_id]()
    return m
