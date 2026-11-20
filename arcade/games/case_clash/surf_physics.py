"""Surf movement, without Qt: a box-shaped player sliding over convex solids.

The numbers and the order of the steps are the classic ones for this kind of movement (units of about one inch):
a fixed 100 Hz tick; gravity applied half before and half after the move; on the ground friction and a ground
acceleration; in the air an acceleration that only looks at the speed *along the wish direction* and caps that at
30 u/s (so pressing a strafe key at right angles to the motion keeps adding speed and turns it); and a slide move
that removes the part of the velocity pointing into whatever plane was hit. A surface steeper than normal.z = 0.7
is never "ground": the player stays in the air state while sliding on it, without friction - that is surfing.

Collision is exact for a box against convex solids: every solid is a set of planes (its faces, plus the bevel
planes that make the box-expanded shape exact), and a move is clipped against the planes pushed out by the box's
size."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

TICK = 0.01  # s
GRAVITY = 800.0
MAX_SPEED = 250.0  # wish speed from the keys
GROUND_ACCEL = 5.5
AIR_ACCEL = 150.0
AIR_CAP = 30.0  # the cap on speed along the wish direction in the air
FRICTION = 4.0
STOP_SPEED = 75.0
JUMP_SPEED = 301.99  # sqrt(2 * 800 * 57)
MAX_VEL = 3500.0  # per axis
GROUND_NZ = 0.7  # a plane with normal.z >= this can be stood on
NO_GROUND_VZ = 180.0  # rising faster than this is never "on the ground"
HULL = (16.0, 16.0, 36.0)  # half extents of the player box (32 x 32 x 72)
EYE = 28.0  # the eye above the box centre (64 above the feet)
EPS = 0.03125  # how far off a surface a move stops
MAX_BUMPS = 4
MAX_PLANES = 5
_PAD = (np.zeros(3), 1.0e9)  # a plane nothing is ever in front of (pads the plane lists to one length)


# --- convex solids ---------------------------------------------------------------------------------------------------

def hull_planes(points: np.ndarray, eps: float = 1e-3) -> list[tuple[np.ndarray, float]]:
    """The face planes (outward unit normal, distance) of the convex hull of a few points."""
    pts = np.asarray(points, float)
    tri = np.array(list(combinations(range(len(pts)), 3)))
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1)
    ok = length > 1e-7
    n = n[ok] / length[ok, None]
    d = (n * a[ok]).sum(1)
    side = pts @ n.T - d  # (points, candidates)
    above, below = side.max(0) > eps, side.min(0) < -eps
    face = ~(above & below)  # every point on one side: a face (pointing away from them)
    sign = np.where(above, -1.0, 1.0)
    planes: list[tuple[np.ndarray, float]] = []
    for nn, dd in zip(n[face] * sign[face, None], d[face] * sign[face], strict=True):
        if any(float(nn @ q) > 1.0 - 1e-9 and abs(dd - e) < 2 * eps for q, e in planes):
            continue
        planes.append((nn, float(dd)))
    return planes


def bevel_planes(points: np.ndarray, faces: list[tuple[np.ndarray, float]], eps: float = 1e-3,
                 ) -> list[tuple[np.ndarray, float]]:
    """The extra planes that make "faces pushed out by the box" the exact shape a box centre cannot enter: the six
    axis planes of the solid's bounds, and for every edge the planes through it along each axis."""
    pts = np.asarray(points, float)
    out: list[tuple[np.ndarray, float]] = []

    def add(n: np.ndarray) -> None:
        d = float((pts @ n).max())
        for q, e in faces + out:
            if float(n @ q) > 1.0 - 1e-9 and abs(d - e) < 2 * eps:
                return
        out.append((n, d))

    for axis in range(3):
        for sign in (1.0, -1.0):
            n = np.zeros(3)
            n[axis] = sign
            add(n)
    on = [np.abs(pts @ n - d) < 4 * eps for n, d in faces]
    for a, b in combinations(range(len(faces)), 2):
        both = np.flatnonzero(on[a] & on[b])
        if len(both) < 2:
            continue
        edge = pts[both]
        span = edge - edge[0]
        far = int(np.argmax((span * span).sum(1)))
        e = span[far]
        length = float(np.linalg.norm(e))
        if length < 1e-6:
            continue
        e = e / length
        for axis in range(3):
            unit = np.zeros(3)
            unit[axis] = 1.0
            n = np.cross(e, unit)
            length = float(np.linalg.norm(n))
            if length < 1e-6:
                continue
            n = n / length
            for cand in (n, -n):
                if float((pts @ cand).max()) - float(cand @ edge[0]) < eps:  # the whole solid is behind it
                    add(cand)
    return out


class World:
    """The solids of a map, ready for box traces."""

    def __init__(self, solids: list[np.ndarray], hull: tuple[float, float, float] = HULL) -> None:
        self.hull = hull
        h = np.array(hull)
        rows = []
        for pts in solids:
            pts = np.asarray(pts, float)
            faces = hull_planes(pts)
            planes = faces + bevel_planes(pts, faces)
            rows.append((pts, planes, len(faces)))
        self.count = len(rows)
        most = max((len(p) for _pts, p, _f in rows), default=1)
        self.N = np.zeros((self.count, most, 3))
        self.D = np.full((self.count, most), _PAD[1])
        self.raw = np.full((self.count, most), _PAD[1])  # the planes of the solid itself (no box added)
        self.faces = np.zeros(self.count, np.intp)
        self.lo = np.zeros((self.count, 3))
        self.hi = np.zeros((self.count, 3))
        for b, (pts, planes, n_faces) in enumerate(rows):
            for k, (n, d) in enumerate(planes):
                self.N[b, k] = n
                self.raw[b, k] = d
                self.D[b, k] = d + float(np.abs(n) @ h)
            self.faces[b] = n_faces
            self.lo[b] = pts.min(0) - h - 1.0
            self.hi[b] = pts.max(0) + h + 1.0
        self.traces = 0

    def near(self, lo: tuple, hi: tuple) -> np.ndarray:
        """The solids whose (box-expanded) bounds touch the box lo..hi."""
        a, b = self.lo, self.hi
        return np.flatnonzero((a[:, 0] <= hi[0]) & (b[:, 0] >= lo[0]) & (a[:, 1] <= hi[1]) & (b[:, 1] >= lo[1])
                              & (a[:, 2] <= hi[2]) & (b[:, 2] >= lo[2]))

    def trace(self, s: tuple, e: tuple) -> tuple[float, tuple | None, tuple | None]:
        """Move the box centre from ``s`` towards ``e``: (the fraction of the way it gets, the normal of the plane
        that stopped it or None, (push normal, depth) when the start is already inside a solid or None)."""
        self.traces += 1
        idx = self.near((min(s[0], e[0]), min(s[1], e[1]), min(s[2], e[2])),
                        (max(s[0], e[0]), max(s[1], e[1]), max(s[2], e[2])))
        if idx.size == 0:
            return 1.0, None, None
        N = self.N[idx]  # noqa: N806
        D = self.D[idx]  # noqa: N806
        d1 = N @ np.array(s) - D
        d2 = N @ np.array(e) - D
        front = d1 > 0.0
        # a plane is missed when the move ends clear of it or does not really go towards it (a velocity just clipped
        # parallel to a plane comes out a hair into it after rounding: that must not count as a new hit)
        miss = (front & ((d2 >= EPS) | (d2 >= d1 - 1e-4))).any(1)
        outside = front.any(1)
        if not ((~miss) & outside).any():
            inside = ~miss & ~outside
            if inside.any():
                return self._inside(N, d1, inside)
            return 1.0, None, None
        with np.errstate(divide="ignore", invalid="ignore"):
            span = d1 - d2
            enter = np.where(front & (span > 0.0), np.maximum((d1 - EPS) / span, 0.0), -1.0)
            leave = np.where(~front & (d2 > 0.0), np.minimum((d1 + EPS) / span, 1.0), 1.0)
        first = enter.max(1)
        last = leave.min(1)
        hit = ~miss & outside & (first < last) & (first > -1.0)
        inside = ~miss & ~outside
        if inside.any():
            return self._inside(N, d1, inside)
        if not hit.any():
            return 1.0, None, None
        first = np.where(hit, first, 2.0)
        k = int(first.argmin())
        frac = float(first[k])
        if frac >= 1.0:
            return 1.0, None, None
        n = N[k, int(enter[k].argmax())]
        return frac, (float(n[0]), float(n[1]), float(n[2])), None

    @staticmethod
    def _inside(N: np.ndarray, d1: np.ndarray, inside: np.ndarray) -> tuple:  # noqa: N803
        """The start is in a solid: the shortest way out of the deepest one."""
        k = int(np.flatnonzero(inside)[0])
        p = int(d1[k].argmax())
        n = N[k, p]
        return 0.0, None, ((float(n[0]), float(n[1]), float(n[2])), float(-d1[k, p]))

    def clear(self, p: tuple) -> bool:
        """True when the box can be at ``p``."""
        idx = self.near(p, p)
        if idx.size == 0:
            return True
        d = self.N[idx] @ np.array(p) - self.D[idx]
        return not (d <= 0.0).all(1).any()


# --- the player ------------------------------------------------------------------------------------------------------

def clip_velocity(v: tuple, n: tuple, overbounce: float = 1.0) -> tuple:
    """Remove the part of ``v`` that points into the plane with normal ``n``."""
    back = (v[0] * n[0] + v[1] * n[1] + v[2] * n[2]) * overbounce
    x, y, z = v[0] - n[0] * back, v[1] - n[1] * back, v[2] - n[2] * back
    adjust = x * n[0] + y * n[1] + z * n[2]
    if adjust < 0.0:
        x, y, z = x - n[0] * adjust, y - n[1] * adjust, z - n[2] * adjust
    return x, y, z


class Player:
    """The surfer: position of the box centre, velocity, and one ``tick`` of movement."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.pos = (0.0, 0.0, 0.0)
        self.prev = self.pos  # the position one tick ago (the view is drawn between the two)
        self.vel = (0.0, 0.0, 0.0)
        self.on_ground = False
        self.surfing = False  # touching a surface too steep to stand on
        self.surf_normal: tuple | None = None
        self.landed = 0.0  # the speed into the surface of the hardest hit this tick (for the sound)
        self.jumped = False
        self.rescues = 0  # times a move that would have stopped dead on a seam was let through

    def place(self, pos: tuple, vel: tuple = (0.0, 0.0, 0.0)) -> None:
        self.pos = self.prev = (float(pos[0]), float(pos[1]), float(pos[2]))
        self.vel = (float(vel[0]), float(vel[1]), float(vel[2]))
        self.on_ground = False
        self.surfing = False
        self.surf_normal = None

    @property
    def speed(self) -> float:
        """Horizontal speed (what a speedometer shows)."""
        return math.hypot(self.vel[0], self.vel[1])

    @property
    def eye(self) -> tuple:
        return self.pos[0], self.pos[1], self.pos[2] + EYE

    def lerp(self, a: float) -> tuple:
        p, q = self.prev, self.pos
        return p[0] + (q[0] - p[0]) * a, p[1] + (q[1] - p[1]) * a, p[2] + (q[2] - p[2]) * a

    # --- one tick ---------------------------------------------------------------------------------------------------

    def tick(self, forward: float, side: float, jump: bool, yaw: float, dt: float = TICK) -> None:
        """``forward`` / ``side``: -1..1 from the keys (side +1 = strafe right); ``yaw`` in radians, 0 along +x,
        growing to the left."""
        self.prev = self.pos
        self.landed = 0.0
        self.jumped = False
        cy, sy = math.cos(yaw), math.sin(yaw)
        wx = cy * forward + sy * side
        wy = sy * forward - cy * side
        length = math.hypot(wx, wy)
        if length > 1e-9:
            wx, wy = wx / length, wy / length
            wish = MAX_SPEED * min(length, 1.0)
        else:
            wx = wy = wish = 0.0
        vx, vy, vz = self.vel
        ground = self._ground(vz)
        vz -= GRAVITY * dt * 0.5
        if ground:
            if jump:
                vz = JUMP_SPEED - GRAVITY * dt * 0.5
                ground = False
                self.jumped = True
            else:
                vz = 0.0
                speed = math.hypot(vx, vy)
                if speed > 0.1:
                    drop = max(speed, STOP_SPEED) * FRICTION * dt
                    k = max(speed - drop, 0.0) / speed
                    vx, vy = vx * k, vy * k
        self.on_ground = ground
        if ground:
            add = wish - (vx * wx + vy * wy)
            if add > 0.0:
                a = min(GROUND_ACCEL * wish * dt, add)
                vx, vy = vx + wx * a, vy + wy * a
            self.vel = (vx, vy, 0.0)
            self._slide(dt)
            self._stay_on_ground()
            vx, vy, vz = self.vel
            self.vel = (vx, vy, 0.0)
        else:
            add = min(wish, AIR_CAP) - (vx * wx + vy * wy)
            if add > 0.0:
                a = min(AIR_ACCEL * wish * dt, add)
                vx, vy = vx + wx * a, vy + wy * a
            self.vel = (vx, vy, vz)
            self._slide(dt)
            vx, vy, vz = self.vel
            vz -= GRAVITY * dt * 0.5
            self.vel = (vx, vy, vz)
        self.vel = tuple(min(max(c, -MAX_VEL), MAX_VEL) for c in self.vel)

    def _ground(self, vz: float) -> bool:
        """Standing on something? (a plane flat enough, 2 units under the box, and not shooting upwards)"""
        self.surfing = False
        self.surf_normal = None
        p = self.pos
        frac, n, stuck = self.world.trace(p, (p[0], p[1], p[2] - 2.0))
        if stuck is not None:
            self._unstick(stuck)
            return False
        if n is None:
            return False
        if n[2] < GROUND_NZ:
            self.surfing = True
            self.surf_normal = n
            return False
        return vz <= NO_GROUND_VZ

    def _stay_on_ground(self) -> None:
        """Keep the feet on the floor over small drops (the edge of a platform lets go)."""
        p = self.pos
        frac, n, stuck = self.world.trace(p, (p[0], p[1], p[2] - 18.0))
        if stuck is None and n is not None and n[2] >= GROUND_NZ and frac > 0.0:
            self.pos = (p[0], p[1], p[2] - 18.0 * frac)

    def _unstick(self, stuck: tuple) -> None:
        """Out of a solid by the shortest way (a safety net; a normal move never ends inside one)."""
        for _ in range(4):
            n, depth = stuck
            push = depth + 4 * EPS
            self.pos = (self.pos[0] + n[0] * push, self.pos[1] + n[1] * push, self.pos[2] + n[2] * push)
            _f, _n, stuck = self.world.trace(self.pos, self.pos)
            if stuck is None:
                return

    def _slide(self, dt: float) -> None:
        """Move by the velocity, sliding along up to four planes (and following the crease between two)."""
        trace = self.world.trace
        pos, v = self.pos, self.vel
        primal = original = v
        planes: list[tuple] = []
        left = dt
        moved = 0.0
        dead = False
        for _bump in range(MAX_BUMPS):
            if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0:
                break
            end = (pos[0] + v[0] * left, pos[1] + v[1] * left, pos[2] + v[2] * left)
            frac, n, stuck = trace(pos, end)
            if stuck is not None:
                self.pos = pos
                self._unstick(stuck)
                pos = self.pos
                continue
            moved += frac
            if frac > 0.0:
                pos = (pos[0] + (end[0] - pos[0]) * frac, pos[1] + (end[1] - pos[1]) * frac,
                       pos[2] + (end[2] - pos[2]) * frac)
                original = v
                planes = []
            if n is None:
                break
            into = -(v[0] * n[0] + v[1] * n[1] + v[2] * n[2])
            if into > self.landed:
                self.landed = into
            if n[2] < GROUND_NZ:
                self.surfing = True
                self.surf_normal = n
            left -= left * frac
            if len(planes) >= MAX_PLANES:
                dead = True
                break
            planes.append(n)
            if len(planes) == 1 and not self.on_ground:
                v = original = clip_velocity(original, n)
            else:
                for i, pi in enumerate(planes):
                    v = clip_velocity(original, pi)
                    if all(v[0] * pj[0] + v[1] * pj[1] + v[2] * pj[2] >= 0.0
                           for j, pj in enumerate(planes) if j != i):
                        break
                else:
                    if len(planes) != 2:
                        dead = True
                        break
                    a, b = planes
                    cx, cy, cz = a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
                    length = math.sqrt(cx * cx + cy * cy + cz * cz) or 1.0
                    cx, cy, cz = cx / length, cy / length, cz / length
                    d = cx * v[0] + cy * v[1] + cz * v[2]
                    v = (cx * d, cy * d, cz * d)
                if v[0] * primal[0] + v[1] * primal[1] + v[2] * primal[2] <= 0.0:
                    dead = True
                    break
        if moved == 0.0 and not dead and any(v):
            dead = planes != []
        if dead:
            v = self._rescue(primal, planes)
        self.pos, self.vel = pos, v

    def _rescue(self, primal: tuple, planes: list[tuple]) -> tuple:
        """The slide gave up (the classic code stops the player dead here). On the ground, or when the planes
        really form a corner, that is right. On a surf ramp it is a seam between two pieces of the same slope:
        keep the speed along the planes instead of losing all of it."""
        if self.on_ground or not planes:
            return 0.0, 0.0, 0.0
        if math.sqrt(primal[0] ** 2 + primal[1] ** 2 + primal[2] ** 2) < 120.0:
            return 0.0, 0.0, 0.0
        if any(n[2] >= GROUND_NZ for n in planes):
            return 0.0, 0.0, 0.0
        v = primal
        for _ in range(2):
            for n in planes:
                v = clip_velocity(v, n)
        if any(v[0] * n[0] + v[1] * n[1] + v[2] * n[2] < -1e-6 for n in planes):
            return 0.0, 0.0, 0.0
        if v[0] * primal[0] + v[1] * primal[1] + v[2] * primal[2] <= 0.0:
            return 0.0, 0.0, 0.0
        self.rescues += 1
        return v
