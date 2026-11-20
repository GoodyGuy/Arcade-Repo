"""The map as the match feels it: what a hero can walk on and into, what a shot hits, and the way from anywhere to
the places bots go (no Qt, plain lists for speed).

Walking: a cell stops a hero when its floor is more than a step above the hero's feet, when its slab is in the way of
the body, or when it belongs to a spawn room that is not the hero's team's (leaving is always allowed).

Routes: every cell's floor (and every slab's top) is a place; places are joined when a hero can walk, step, hop up
(under a metre) or drop (a few metres) from one to the next. ``field`` gives the walking distance from every place
to one goal; a bot just walks downhill in it."""

from __future__ import annotations

import heapq
import math

from .data import STEP
from .maps import NO_SLAB, MapData

INF = float("inf")
BODY = 2.2  # head room a route needs
HOP = 0.95
DROP = 7.0


class World:
    def __init__(self, md: MapData) -> None:
        self.md = md
        self.w, self.h = md.w, md.h
        self.fh: list = md.fh.astype(float).tolist()
        self.sb: list = md.sb.astype(float).tolist()
        self.st: list = md.st.astype(float).tolist()
        self.room: list = md.room.astype(int).tolist()
        self.door: list = md.door.astype(int).tolist()
        self.has_slab = bool((md.sb < NO_SLAB / 2).any())
        self.room_owner: dict[int, int | None] = {}
        self.closed: set[int] = set()  # rooms whose doors are shut
        self.fields: dict = {}
        self._build_nav()

    # --- walking --------------------------------------------------------------------------------------------------
    def room_at(self, x: float, y: float) -> int:
        cx, cy = int(x), int(y)
        if 0 <= cx < self.w and 0 <= cy < self.h:
            return self.room[cy][cx]
        return 0

    def free(self, x: float, y: float, z: float, r: float, hgt: float, team: int, cur_room: int,
             step: float = STEP) -> bool:
        """Can a body of radius ``r`` and height ``hgt`` with its feet at ``z`` stand at (x, y)?"""
        if x - r < 0.0 or y - r < 0.0:
            return False
        x0, x1, y0, y1 = int(x - r), int(x + r), int(y - r), int(y + r)
        if x1 >= self.w or y1 >= self.h:
            return False
        top = z + step
        head = z + hgt
        fh, sb, st, room, door = self.fh, self.sb, self.st, self.room, self.door
        owner, closed = self.room_owner, self.closed
        for cy in range(y0, y1 + 1):
            rh, rb, rr = fh[cy], sb[cy], room[cy]
            for cx in range(x0, x1 + 1):
                if rh[cx] > top:
                    return False
                if rb[cx] < head and st[cy][cx] > top:
                    return False
                rm = rr[cx]
                if rm:
                    if closed and door[cy][cx] in closed:
                        return False
                    if rm != cur_room and owner.get(rm) != team:
                        return False
        return True

    def ground(self, x: float, y: float, z: float, r: float, step: float = STEP) -> float:
        """The highest thing under the body that its feet (at ``z``) can be on."""
        x0, x1, y0, y1 = int(x - r), int(x + r), int(y - r), int(y + r)
        top = z + step
        g = -50.0
        fh, st = self.fh, self.st
        for cy in range(max(y0, 0), min(y1, self.h - 1) + 1):
            rh, rt = fh[cy], st[cy]
            for cx in range(max(x0, 0), min(x1, self.w - 1) + 1):
                t = rt[cx]
                v = t if -1e8 < t <= top else rh[cx]
                if v > g and v <= top:
                    g = v
        return g

    def ceiling(self, x: float, y: float, z: float, r: float) -> float:
        """The lowest slab above the feet at ``z`` over the body."""
        if not self.has_slab:
            return INF
        x0, x1, y0, y1 = int(x - r), int(x + r), int(y - r), int(y + r)
        c = INF
        sb = self.sb
        for cy in range(max(y0, 0), min(y1, self.h - 1) + 1):
            rb = sb[cy]
            for cx in range(max(x0, 0), min(x1, self.w - 1) + 1):
                b = rb[cx]
                if z < b < c:
                    c = b
        return c

    def floor_at(self, x: float, y: float, z: float = 99.0) -> float:
        """The floor (or slab top) under the point (x, y, z)."""
        cx, cy = int(x), int(y)
        if not (0 <= cx < self.w and 0 <= cy < self.h):
            return 0.0
        t = self.st[cy][cx]
        return t if -1e8 < t <= z + 0.3 else self.fh[cy][cx]

    # --- shots ----------------------------------------------------------------------------------------------------
    def raycast(self, ox: float, oy: float, oz: float, dx: float, dy: float, dz: float, maxd: float) -> float:
        """How far a ray from (ox, oy, oz) along the unit vector (dx, dy, dz) goes before the map stops it."""
        W, H = self.w, self.h
        fh, sb, st = self.fh, self.sb, self.st
        cx, cy = int(ox), int(oy)
        if ox < 0 or oy < 0:
            return 0.0
        if dx > 1e-9:
            sx, tmx, tdx = 1, (cx + 1 - ox) / dx, 1.0 / dx
        elif dx < -1e-9:
            sx, tmx, tdx = -1, (cx - ox) / dx, -1.0 / dx
        else:
            sx, tmx, tdx = 0, INF, INF
        if dy > 1e-9:
            sy, tmy, tdy = 1, (cy + 1 - oy) / dy, 1.0 / dy
        elif dy < -1e-9:
            sy, tmy, tdy = -1, (cy - oy) / dy, -1.0 / dy
        else:
            sy, tmy, tdy = 0, INF, INF
        t = 0.0
        slabs = self.has_slab
        while True:
            if cx < 0 or cy < 0 or cx >= W or cy >= H:
                return t
            t1 = tmx if tmx < tmy else tmy
            if t1 > maxd:
                t1 = maxd
            h = fh[cy][cx]
            z0 = oz + dz * t
            if z0 < h:
                return t
            z1 = oz + dz * t1
            if z1 < h:
                return (h - oz) / dz
            if slabs:
                b = sb[cy][cx]
                if b < 1e8:
                    tp = st[cy][cx]
                    if z0 < b:
                        if z1 > b:
                            return (b - oz) / dz
                    elif z0 > tp:
                        if z1 < tp:
                            return (tp - oz) / dz
                    else:
                        return t
            if t1 >= maxd:
                return maxd
            t = t1
            if tmx < tmy:
                tmx += tdx
                cx += sx
            else:
                tmy += tdy
                cy += sy

    def los(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> bool:
        dx, dy, dz = bx - ax, by - ay, bz - az
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < 1e-6:
            return True
        return self.raycast(ax, ay, az, dx / d, dy / d, dz / d, d) >= d - 1e-4

    # --- routes ---------------------------------------------------------------------------------------------------
    def _build_nav(self) -> None:
        W, H = self.w, self.h
        N = W * H
        fh, sb, st, room = self.fh, self.sb, self.st, self.room
        surf: list = [None] * (2 * N)
        for y in range(H):
            for x in range(W):
                i = y * W + x
                h = fh[y][x]
                b = sb[y][x]
                if b > 1e8:
                    surf[i] = h
                else:
                    if b - h >= BODY:
                        surf[i] = h
                    surf[N + i] = st[y][x]
        self.surf = surf
        # how cramped a place is (routes keep off the walls a little)
        tight = [0.0] * (2 * N)
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                for lv in (0, N):
                    s = surf[lv + y * W + x]
                    if s is None:
                        continue
                    n = 0
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if fh[y + dy][x + dx] > s + HOP:
                            n += 1
                    tight[lv + y * W + x] = 0.7 * n
        adj: list = [None] * (2 * N)
        radj: list = [[] for _ in range(2 * N)]

        def step(a: int, sa: float, x: int, y: int, nx: int, ny: int) -> list:
            """The places next door a hero at place ``a`` can get to: [(place, cost, flat)]."""
            out = []
            if not (0 < nx < W - 1 and 0 < ny < H - 1):
                return out
            j = ny * W + nx
            if room[ny][nx] and room[ny][nx] != room[y][x]:
                return out
            for lv in (0, N):
                sn = surf[lv + j]
                if sn is None:
                    continue
                dz = sn - sa
                if dz > HOP or dz < -DROP:
                    continue
                # the body has to fit past the neighbour's slab on the way
                b = sb[ny][nx]
                if lv == 0 and b < 1e8 and (b < max(sa, sn) + BODY - 0.3):
                    continue
                cost = 1.0 + (0.8 if dz > STEP else 0.0) + (0.5 if dz < -STEP else 0.0) + tight[lv + j]
                out.append((lv + j, cost, abs(dz) <= STEP))
            return out

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                for lv in (0, N):
                    a = lv + y * W + x
                    sa = surf[a]
                    if sa is None:
                        continue
                    if sa > 6.5 and lv == 0 and sb[y][x] > 1e8 and fh[y][x] > 6.5:
                        continue  # roof tops: nobody goes there
                    edges = []
                    flat = {}
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        for b, cost, is_flat in step(a, sa, x, y, x + dx, y + dy):
                            edges.append((b, cost))
                            if is_flat:
                                flat[(dx, dy)] = surf[b]
                    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        if (dx, 0) in flat and (0, dy) in flat:
                            for b, cost, is_flat in step(a, sa, x, y, x + dx, y + dy):
                                if is_flat and abs(flat[(dx, 0)] - sa) < 0.1 and abs(flat[(0, dy)] - sa) < 0.1:
                                    edges.append((b, cost * 1.414))
                    adj[a] = edges
                    for b, cost in edges:
                        radj[b].append((a, cost))
        self.adj = adj
        self.radj = radj

    def place(self, x: float, y: float, z: float) -> int:
        """The place a hero at (x, y) with feet at ``z`` is on."""
        cx, cy = min(max(int(x), 0), self.w - 1), min(max(int(y), 0), self.h - 1)
        i = cy * self.w + cx
        t = self.st[cy][cx]
        if t > -1e8 and z >= t - 0.7:
            return self.w * self.h + i
        return i

    def field(self, key, x: float, y: float, z: float = 0.0) -> list:  # noqa: ANN001
        """Walking distance from every place to (x, y, z), made once per ``key``."""
        f = self.fields.get(key)
        if f is not None:
            return f
        radj = self.radj
        pop, push = heapq.heappop, heapq.heappush
        dist: list = []
        # the goal itself may be somewhere nobody can walk to (the top of a monument, a crate): then the nearest
        # place that most of the map can reach stands in for it
        for goal in self._places_near(x, y, z):
            dist = [INF] * (2 * self.w * self.h)
            dist[goal] = 0.0
            heap = [(0.0, goal)]
            reached = 0
            while heap:
                d, a = pop(heap)
                if d > dist[a]:
                    continue
                reached += 1
                for b, cost in radj[a]:
                    nd = d + cost
                    if nd < dist[b]:
                        dist[b] = nd
                        push(heap, (nd, b))
            if reached > 200:
                break
        self.fields[key] = dist
        return dist

    def _places_near(self, x: float, y: float, z: float) -> list:
        """The place at (x, y, z), then the places round it, nearest first."""
        out = []
        cx, cy = int(x), int(y)
        n = self.w * self.h
        for yy in range(max(cy - 5, 1), min(cy + 6, self.h - 1)):
            for xx in range(max(cx - 5, 1), min(cx + 6, self.w - 1)):
                for lv in (0, n):
                    i = lv + yy * self.w + xx
                    if self.surf[i] is not None and self.radj[i]:
                        out.append(((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 + abs(self.surf[i] - z) * 0.5, i))
        out.sort()
        first = self.place(x, y, z)
        return [first] * (self.surf[first] is not None) + [i for _d, i in out if i != first]

    def downhill(self, dist: list, x: float, y: float, z: float) -> tuple[float, float, float, float] | None:
        """From (x, y, z), the next spot toward the field's goal: (x, y, its height, distance left); None when
        there is no way from here."""
        a = self.place(x, y, z)
        edges = self.adj[a]
        if not edges:
            a = a % (self.w * self.h)
            edges = self.adj[a]
            if not edges:
                return None
        here = dist[a]
        best, bd = -1, here
        for b, cost in edges:
            d = dist[b]
            if d + cost * 0.01 < bd:
                best, bd = b, d
        if best < 0:
            return None if here == INF else (x, y, z, here)
        # look one more step ahead when the way is flat: smoother corners
        n = self.w * self.h
        j = best % n
        return (j % self.w) + 0.5, (j // self.w) + 0.5, self.surf[best], bd
