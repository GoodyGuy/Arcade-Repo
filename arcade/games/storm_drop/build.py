"""The build grid: walls, floors, ramps and roofs on 4 x 3 x 4 m cells, for the island's own buildings and for
everything a player or a bot puts up in a fight.

A piece has health (it starts part built and finishes over a few seconds, sooner in wood than in metal), can be shot or
hammered down, and must be held up: by the ground under it or by a piece it touches. When a piece goes, whatever it
alone was holding up falls after it. This module also answers the questions movement and bullets ask: what is under my
feet here, what is in my way, what does this ray hit."""

from __future__ import annotations

import math

from .data import CELL, FLOOR, LEVEL, MAT_BUILD, PIECE_HP_SCALE, RAMP, ROOF, SEA, SIZE, WALLX, WALLZ
from .island import Island

ROOF_H = 1.6
SLAB = 0.06  # a floor's top sits this far above its level (never fights the ground it lies on)
RAMP_DX = ((0, 1), (1, 0), (0, -1), (-1, 0))  # dir -> (dx, dz) it rises towards


class Piece:
    __slots__ = ("kind", "gx", "gy", "gz", "dir", "mat", "hp", "hp_full", "rate", "t0", "dur", "owner", "base",
                 "top", "slot", "var", "anchor", "key", "verts", "grounded", "dead", "hit_t", "world")

    def centre(self) -> tuple[float, float, float]:
        k = self.kind
        if k == WALLX:
            return (self.gx + 0.5) * CELL, self.gy * LEVEL + 1.5, self.gz * CELL
        if k == WALLZ:
            return self.gx * CELL, self.gy * LEVEL + 1.5, (self.gz + 0.5) * CELL
        y = self.gy * LEVEL + (1.5 if k == RAMP else (0.6 if k == ROOF else 0.0))
        return (self.gx + 0.5) * CELL, y, (self.gz + 0.5) * CELL


def piece_verts(kind: int, gx: int, gy: int, gz: int, d: int) -> tuple:
    """The lattice points a piece touches (two pieces that share one hold each other up)."""
    if kind == WALLX:
        return ((gx, gy, gz), (gx + 1, gy, gz), (gx, gy + 1, gz), (gx + 1, gy + 1, gz))
    if kind == WALLZ:
        return ((gx, gy, gz), (gx, gy, gz + 1), (gx, gy + 1, gz), (gx, gy + 1, gz + 1))
    if kind == RAMP:
        lo = {0: ((gx, gz), (gx + 1, gz)), 1: ((gx, gz), (gx, gz + 1)), 2: ((gx, gz + 1), (gx + 1, gz + 1)),
              3: ((gx + 1, gz), (gx + 1, gz + 1))}[d]
        hi = {0: ((gx, gz + 1), (gx + 1, gz + 1)), 1: ((gx + 1, gz), (gx + 1, gz + 1)), 2: ((gx, gz), (gx + 1, gz)),
              3: ((gx, gz), (gx, gz + 1))}[d]
        return tuple((x, gy, z) for x, z in lo) + tuple((x, gy + 1, z) for x, z in hi)
    return ((gx, gy, gz), (gx + 1, gy, gz), (gx, gy, gz + 1), (gx + 1, gy, gz + 1))


class Structures:
    def __init__(self, island: Island) -> None:
        self.island = island
        self.pieces: dict[tuple, Piece] = {}
        self.cols: dict[tuple, int] = {}  # (gx, gz) -> how many pieces matter to someone standing in that column
        self.verts: dict[tuple, set] = {}
        self.growing: set[Piece] = set()
        self.dirty: set[Piece] = set()  # for the renderer: rows to rewrite
        self.gone: list[Piece] = []  # for the renderer: rows to free
        self.broken: list[Piece] = []  # for sounds and debris this frame
        self.falling: list[tuple[float, Piece]] = []
        self.time = 0.0
        for kind, gx, gy, gz, d, mat, var, anchor in island.pieces:
            self.place(kind, gx, gy, gz, d, mat, owner=-1, instant=True, var=var, anchor=anchor, check=False,
                       world=True)

    # --- placing and breaking -------------------------------------------------------------------------------------------

    def _cols_of(self, kind: int, gx: int, gz: int) -> tuple:
        if kind == WALLX:
            return tuple((gx + i, gz + j) for i in (-1, 0, 1) for j in (-1, 0))
        if kind == WALLZ:
            return tuple((gx + i, gz + j) for i in (-1, 0) for j in (-1, 0, 1))
        return ((gx, gz),)

    def _ground_under(self, verts: tuple) -> float:
        h = self.island.height
        low = min(v[1] for v in verts)
        return max(h(v[0] * CELL, v[2] * CELL) - v[1] * LEVEL for v in verts if v[1] == low)

    def supported(self, kind: int, gx: int, gy: int, gz: int, d: int) -> bool:
        verts = piece_verts(kind, gx, gy, gz, d)
        if self._ground_under(verts) > -1.7:
            return True
        return any(self.verts.get(v) for v in verts)

    def can_place(self, kind: int, gx: int, gy: int, gz: int, d: int = 0) -> bool:
        if (kind, gx, gy, gz) in self.pieces or gy < 0 or gy > 40:
            return False
        if not (1 <= gx < SIZE / CELL - 1 and 1 <= gz < SIZE / CELL - 1):
            return False
        if kind == RAMP and ((FLOOR, gx, gy, gz) in self.pieces and False):
            return False
        return self.supported(kind, gx, gy, gz, d)

    def place(self, kind: int, gx: int, gy: int, gz: int, d: int, mat: int, owner: int = -1,
              instant: bool = False, var: int = 0, anchor: bool = False, check: bool = True,
              world: bool = False) -> Piece | None:
        if check and not self.can_place(kind, gx, gy, gz, d):
            return None
        p = Piece()
        p.kind, p.gx, p.gy, p.gz, p.dir, p.mat, p.owner, p.var = kind, gx, gy, gz, d, mat, owner, var
        p.key = (kind, gx, gy, gz)
        p.anchor, p.dead, p.slot, p.hit_t, p.world = anchor, False, None, -9.0, world
        start, full, secs = MAT_BUILD[mat]
        scale = PIECE_HP_SCALE[kind]
        p.hp_full = full * scale
        p.t0, p.dur = self.time, secs
        if instant:
            p.hp, p.rate, p.dur = p.hp_full, 0.0, 0.0
        else:
            p.hp = start * scale
            p.rate = (p.hp_full - p.hp) / secs
            self.growing.add(p)
        p.verts = piece_verts(kind, gx, gy, gz, d)
        under = self._ground_under(p.verts)
        p.grounded = anchor or under > -1.7
        p.top = gy * LEVEL + LEVEL
        p.base = gy * LEVEL
        if kind in (WALLX, WALLZ) and -2.4 < under:
            h = self.island.height
            (ax, _y, az), (bx, _y2, bz) = p.verts[0], p.verts[1]
            low = min(h(ax * CELL, az * CELL), h(bx * CELL, bz * CELL), h((ax + bx) * 2.0, (az + bz) * 2.0))
            if low < p.base:
                p.base = max(low - 0.15, p.base - 3.0)
        self.pieces[p.key] = p
        for c in self._cols_of(kind, gx, gz):
            self.cols[c] = self.cols.get(c, 0) + 1
        for v in p.verts:
            self.verts.setdefault(v, set()).add(p)
        self.dirty.add(p)
        return p

    def remove(self, p: Piece, collapse: bool = True) -> None:
        if p.dead:
            return
        p.dead = True
        del self.pieces[p.key]
        for c in self._cols_of(p.kind, p.gx, p.gz):
            n = self.cols.get(c, 0) - 1
            if n <= 0:
                self.cols.pop(c, None)
            else:
                self.cols[c] = n
        near: set[Piece] = set()
        for v in p.verts:
            s = self.verts.get(v)
            if s is not None:
                s.discard(p)
                near |= s
                if not s:
                    del self.verts[v]
        self.growing.discard(p)
        self.dirty.discard(p)
        self.gone.append(p)
        self.broken.append(p)
        if collapse and near:
            self._collapse(near)

    def _collapse(self, starts: set) -> None:
        safe: set[Piece] = set()
        for s in starts:
            if s in safe or s.dead:
                continue
            seen = {s}
            order = [(s, 0)]
            held = False
            i = 0
            while i < len(order):
                q, depth = order[i]
                i += 1
                if q.grounded or q in safe or len(order) > 500:
                    held = True
                    break
                for v in q.verts:
                    for n in self.verts.get(v, ()):
                        if n not in seen:
                            seen.add(n)
                            order.append((n, depth + 1))
            if held:
                safe |= seen
            else:
                for q, depth in order:
                    self.falling.append((self.time + 0.12 + 0.1 * depth, q))

    def damage(self, p: Piece, amount: float) -> bool:
        """True when it broke."""
        if p.dead:
            return False
        p.hp -= amount
        p.hit_t = self.time
        if p.hp <= 0.0:
            self.remove(p)
            return True
        self.dirty.add(p)
        return False

    def update(self, dt: float) -> None:
        self.time += dt
        if self.growing:
            done = []
            for p in self.growing:
                p.hp += p.rate * dt
                if self.time >= p.t0 + p.dur:
                    p.hp = min(p.hp, p.hp_full)
                    done.append(p)
            for p in done:
                self.growing.discard(p)
        if self.falling:
            due = [q for t, q in self.falling if t <= self.time]
            if due:
                self.falling = [(t, q) for t, q in self.falling if t > self.time]
                for q in due:
                    self.remove(q, collapse=False)

    def blast(self, x: float, y: float, z: float, radius: float, amount: float) -> None:
        r2 = (radius + 1.6) ** 2
        g = int(radius // CELL) + 1
        cx, cy, cz = int(x // CELL), int(y // LEVEL), int(z // CELL)
        hit = []
        for ix in range(cx - g, cx + g + 2):
            for iz in range(cz - g, cz + g + 2):
                if (ix, iz) not in self.cols:
                    continue
                for iy in range(cy - 2, cy + 3):
                    for kind in (FLOOR, WALLX, WALLZ, RAMP, ROOF):
                        p = self.pieces.get((kind, ix, iy, iz))
                        if p is not None:
                            px, py, pz = p.centre()
                            d2 = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2
                            if d2 < r2:
                                hit.append((p, d2))
        for p, d2 in hit:
            self.damage(p, amount * (1.0 - 0.5 * min(math.sqrt(d2) / (radius + 1.6), 1.0)))

    # --- movement ---------------------------------------------------------------------------------------------------------

    @staticmethod
    def surface(p: Piece, x: float, z: float) -> float:
        k = p.kind
        if k == FLOOR:
            return p.gy * LEVEL + SLAB
        if k == RAMP:
            d = p.dir
            if d == 0:
                t = (z - p.gz * CELL) / CELL
            elif d == 1:
                t = (x - p.gx * CELL) / CELL
            elif d == 2:
                t = 1.0 - (z - p.gz * CELL) / CELL
            else:
                t = 1.0 - (x - p.gx * CELL) / CELL
            return p.gy * LEVEL + LEVEL * min(max(t, 0.0), 1.0) + SLAB
        u = abs((x - p.gx * CELL) / (CELL / 2) - 1.0)
        v = abs((z - p.gz * CELL) / (CELL / 2) - 1.0)
        return p.gy * LEVEL + ROOF_H * (1.0 - min(max(u, v), 1.0))

    def support(self, x: float, z: float, limit: float) -> float | None:
        """The highest slab top at (x, z) that is not above ``limit`` (feet plus a step)."""
        cx, cz = int(x * 0.25), int(z * 0.25)
        if (cx, cz) not in self.cols:
            return None
        get = self.pieces.get
        g1 = int(limit // LEVEL)
        best = None
        for gy in (g1, g1 - 1, g1 - 2):
            for kind in (FLOOR, RAMP, ROOF):
                p = get((kind, cx, gy, cz))
                if p is not None:
                    y = self.surface(p, x, z)
                    if y <= limit and (best is None or y > best):
                        best = y
            if best is not None and gy < g1:
                break
        return best

    def ceiling(self, x: float, z: float, lo: float, hi: float) -> float | None:
        """The lowest slab that cuts through the span lo..hi at (x, z)."""
        cx, cz = int(x * 0.25), int(z * 0.25)
        if (cx, cz) not in self.cols:
            return None
        get = self.pieces.get
        best = None
        for gy in range(int(lo // LEVEL) - 1, int(hi // LEVEL) + 1):
            for kind in (FLOOR, RAMP, ROOF):
                p = get((kind, cx, gy, cz))
                if p is not None:
                    y = self.surface(p, x, z)
                    if lo < y < hi and (best is None or y < best):
                        best = y
        return best

    def push(self, x: float, z: float, feet: float, head: float, r: float) -> tuple[float, float]:
        """Out of the walls: the position moved so a body of radius ``r`` no longer cuts one."""
        cx, cz = int(x * 0.25), int(z * 0.25)
        if (cx, cz) not in self.cols:
            return x, z
        lx, lz = x - cx * CELL, z - cz * CELL
        zi = cz if lz < r else (cz + 1 if lz > CELL - r else None)
        xi = cx if lx < r else (cx + 1 if lx > CELL - r else None)
        if zi is None and xi is None:
            return x, z
        get = self.pieces.get
        r2 = r * r
        for gy in range(int(feet // LEVEL), int(head // LEVEL) + 2):
            if zi is not None:
                zw = zi * CELL
                for gxw in (cx - 1, cx, cx + 1):
                    p = get((WALLX, gxw, gy, zi))
                    if p is not None and feet < p.top - 0.05 and head > p.base:
                        px = min(max(x, gxw * CELL), gxw * CELL + CELL)
                        ddx, ddz = x - px, z - zw
                        d2 = ddx * ddx + ddz * ddz
                        if d2 < r2:
                            if d2 > 1e-8:
                                d = math.sqrt(d2)
                                x, z = px + ddx / d * r, zw + ddz / d * r
                            else:
                                z = zw + r
            if xi is not None:
                xw = xi * CELL
                for gzw in (cz - 1, cz, cz + 1):
                    p = get((WALLZ, xi, gy, gzw))
                    if p is not None and feet < p.top - 0.05 and head > p.base:
                        pz = min(max(z, gzw * CELL), gzw * CELL + CELL)
                        ddx, ddz = x - xw, z - pz
                        d2 = ddx * ddx + ddz * ddz
                        if d2 < r2:
                            if d2 > 1e-8:
                                d = math.sqrt(d2)
                                x, z = xw + ddx / d * r, pz + ddz / d * r
                            else:
                                x = xw + r
        return x, z

    # --- rays ---------------------------------------------------------------------------------------------------------------

    def ray(self, ox: float, oy: float, oz: float, dx: float, dy: float, dz: float, maxd: float,
            props: bool = True) -> tuple[float, str | None, object]:
        """What a ray hits first: (distance, "ground" / "piece" / "prop" / None, the piece or prop)."""
        isl = self.island
        height = isl.height
        cols, get, prop_cols = self.cols, self.pieces.get, isl.prop_cols
        lim = SIZE - 0.01
        if not (0.0 < ox < lim and 0.0 < oz < lim):
            return maxd, None, None
        cx, cz = int(ox * 0.25), int(oz * 0.25)
        inf = 1e18
        if dx > 1e-9:
            sx, tmx, tdx = 1, ((cx + 1) * CELL - ox) / dx, CELL / dx
        elif dx < -1e-9:
            sx, tmx, tdx = -1, (cx * CELL - ox) / dx, -CELL / dx
        else:
            sx, tmx, tdx = 0, inf, inf
        if dz > 1e-9:
            sz, tmz, tdz = 1, ((cz + 1) * CELL - oz) / dz, CELL / dz
        elif dz < -1e-9:
            sz, tmz, tdz = -1, (cz * CELL - oz) / dz, -CELL / dz
        else:
            sz, tmz, tdz = 0, inf, inf
        t0 = 0.0
        gmax = int(SIZE / CELL)
        while t0 < maxd:
            if tmx < tmz:
                t1, axis = tmx, 0
            else:
                t1, axis = tmz, 1
            crossing = t1 < maxd
            if not crossing:
                t1 = maxd
            best, what, obj = inf, None, None
            y0, y1 = oy + dy * t0, oy + dy * t1
            # the ground (and the sea)
            ylo = y0 if y0 < y1 else y1
            if ylo < 46.0:
                for tt in ((t0 + t1) * 0.5, t1):
                    yy = oy + dy * tt
                    h = height(ox + dx * tt, oz + dz * tt)
                    if h < SEA:
                        h = SEA
                    if yy < h:
                        a, b = t0, tt
                        for _ in range(7):
                            m = (a + b) * 0.5
                            hm = height(ox + dx * m, oz + dz * m)
                            if hm < SEA:
                                hm = SEA
                            if oy + dy * m < hm:
                                b = m
                            else:
                                a = m
                        best, what = b, "ground"
                        break
            key = (cx, cz)
            if key in cols:
                yhi = y1 if y0 < y1 else y0
                for gy in range(int(ylo // LEVEL) - 1, int(yhi // LEVEL) + 1):
                    p = get((FLOOR, cx, gy, cz))
                    if p is not None and (dy > 1e-9 or dy < -1e-9):
                        t = (gy * LEVEL + SLAB - oy) / dy
                        if t0 <= t <= t1 and t < best:
                            best, what, obj = t, "piece", p
                    p = get((RAMP, cx, gy, cz))
                    if p is not None:
                        d = p.dir
                        k = LEVEL / CELL
                        if d == 0:
                            num, den = gy * LEVEL - oy + k * (oz - cz * CELL), dy - k * dz
                        elif d == 1:
                            num, den = gy * LEVEL - oy + k * (ox - cx * CELL), dy - k * dx
                        elif d == 2:
                            num, den = gy * LEVEL + LEVEL - oy - k * (oz - cz * CELL), dy + k * dz
                        else:
                            num, den = gy * LEVEL + LEVEL - oy - k * (ox - cx * CELL), dy + k * dx
                        if den > 1e-9 or den < -1e-9:
                            t = num / den
                            if t0 <= t <= t1 and t < best:
                                best, what, obj = t, "piece", p
                    p = get((ROOF, cx, gy, cz))
                    if p is not None:
                        prev_t = t0
                        prev = oy + dy * t0 - self.surface(p, ox + dx * t0, oz + dz * t0)
                        for i in (1, 2, 3, 4):
                            tt = t0 + (t1 - t0) * i / 4
                            cur = oy + dy * tt - self.surface(p, ox + dx * tt, oz + dz * tt)
                            if (prev > 0) != (cur > 0) and oy + dy * tt > gy * LEVEL - 0.01:
                                tm = prev_t + (tt - prev_t) * (prev / (prev - cur)) if prev != cur else tt
                                if tm < best:
                                    best, what, obj = tm, "piece", p
                                break
                            prev, prev_t = cur, tt
                if crossing and t1 < best:
                    if axis == 0:
                        wk, wx, wz = WALLZ, (cx + 1 if sx > 0 else cx), cz
                    else:
                        wk, wx, wz = WALLX, cx, (cz + 1 if sz > 0 else cz)
                    g = int(y1 // LEVEL)
                    for gy in (g, g + 1):
                        p = get((wk, wx, gy, wz))
                        if p is not None and p.base <= y1 < p.top:
                            best, what, obj = t1, "piece", p
                            break
            if props:
                plist = prop_cols.get(key)
                if plist:
                    for pr in plist:
                        if pr.alive:
                            t = _ray_prop(pr, ox, oy, oz, dx, dy, dz)
                            if t is not None and t < best and t <= maxd:
                                best, what, obj = t, "prop", pr
            if what is not None and best <= t1 + 1e-6:
                return best, what, obj
            if what == "prop" and best <= maxd:  # a prop reaching over the column's edge: still the nearest thing
                return best, what, obj
            if not crossing:
                break
            t0 = t1
            if axis == 0:
                cx += sx
                tmx += tdx
                if cx < 0 or cx >= gmax:
                    break
            else:
                cz += sz
                tmz += tdz
                if cz < 0 or cz >= gmax:
                    break
        return maxd, None, None

    def clear_line(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> bool:
        dx, dy, dz = bx - ax, by - ay, bz - az
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < 1e-3:
            return True
        return self.ray(ax, ay, az, dx / d, dy / d, dz / d, d)[1] is None


def _ray_prop(p, ox: float, oy: float, oz: float, dx: float, dy: float, dz: float) -> float | None:  # noqa: ANN001
    y0, y1 = p.y, p.y + p.h
    if p.shape == "cyl":
        fx, fz = ox - p.x, oz - p.z
        a = dx * dx + dz * dz
        r2 = p.a * p.a
        if a < 1e-9:
            if fx * fx + fz * fz > r2 or dy == 0:
                return None
            t = ((y1 if dy < 0 else y0) - oy) / dy
            return t if t >= 0 else None
        b = fx * dx + fz * dz
        c = fx * fx + fz * fz - r2
        disc = b * b - a * c
        if disc < 0:
            return None
        s = math.sqrt(disc)
        t = (-b - s) / a
        if t < 0:
            if c > 0:
                return None
            t = 0.0  # starting inside
        y = oy + dy * t
        if y0 <= y <= y1:
            return t
        if y > y1 and dy < 0:  # down through the top
            t2 = (y1 - oy) / dy
            if t2 <= (-b + s) / a:
                return t2
        return None
    tmin, tmax = 0.0, 1e18
    for o, d, lo, hi in ((ox, dx, p.x - p.a, p.x + p.a), (oy, dy, y0, y1), (oz, dz, p.z - p.b, p.z + p.b)):
        if -1e-9 < d < 1e-9:
            if o < lo or o > hi:
                return None
            continue
        ta, tb = (lo - o) / d, (hi - o) / d
        if ta > tb:
            ta, tb = tb, ta
        if ta > tmin:
            tmin = ta
        if tb < tmax:
            tmax = tb
        if tmin > tmax:
            return None
    return tmin
