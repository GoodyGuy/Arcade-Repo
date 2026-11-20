"""The two maps of BREACH POINT, built in code: KILNYARD (a brickworks at sunset) and COLD HARBOR (a fish dock on a
grey morning). A map is a height field of one-metre cells (0: open floor, up to WALL: a full wall; low values are cover,
steps and decks you stand on), a material per cell, roofs, two sites, spawns, the buy-phase zones, named places, and
what the bots need: paths (a flood-filled distance field per goal), lines of sight, and the spots to hold."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import EYE, PEN_POWER, ROOF, STEP, WALL

# floors
F_CONC, F_TILE, F_WOOD, F_PLATE, F_GRAVEL, F_SITE = 0, 1, 2, 3, 4, 5
# solids
M_WALL, M_WALL2, M_PANEL, M_CRATE, M_STEEL, M_TRIM, M_DECK, M_ROOF, M_LOW, M_BARRICADE = 8, 9, 10, 11, 12, 13, 14, 15, \
    16, 17
INF = 1e9
RESIST = {M_WALL: INF, M_WALL2: INF, M_PANEL: 0.55, M_CRATE: 0.65, M_STEEL: 1.7, M_TRIM: INF, M_DECK: 1.2, M_LOW: 1.6,
          M_BARRICADE: 2.2}
DIRS8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
FAR = 30000


@dataclass
class Spot:
    x: float
    y: float
    wx: float  # what it watches
    wy: float


@dataclass
class Site:
    key: str
    name: str
    rect: tuple  # the plant zone, cells (x0, y0, x1, y1)
    attack: list  # (name, x, y, stage x, stage y): the ways in from the attackers' side
    defend: list  # (name, x, y): the ways in from the defenders' side
    smokes: list  # (x, y) the attackers smoke off
    holds: list = field(default_factory=list)  # Spot: where defenders stand
    posts: list = field(default_factory=list)  # Spot: where attackers stand after the plant
    plants: list = field(default_factory=list)  # (x, y) places to plant

    @property
    def centre(self) -> tuple:
        x0, y0, x1, y1 = self.rect
        return (x0 + x1) / 2, (y0 + y1) / 2

    def inside(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= x < x1 and y0 <= y < y1


class GameMap:
    def __init__(self, key: str, name: str, w: int, h: int) -> None:
        self.key, self.name, self.W, self.H = key, name, w, h
        self.hgt = np.full((h, w), WALL, np.float32)
        self.mat = np.full((h, w), M_WALL, np.uint8)
        self.floor = np.full((h, w), F_CONC, np.uint8)
        self.roof = np.zeros((h, w), bool)
        self.sites: dict[str, Site] = {}
        self.spawns: dict[str, list] = {}
        self.spawn_yaw: dict[str, float] = {}
        self.zones: dict[str, np.ndarray] = {}
        self.places: list = []  # (name, x0, y0, x1, y1)
        self.lamps: list = []  # (x, y, z, (r, g, b), radius)
        self.scene: dict = {}
        self.mid: tuple = (0.0, 0.0)  # where a spare defender waits
        self.blurb = ""
        self.version = 0  # goes up when a barricade changes the ground
        self._fields: dict = {}
        self.budget = 2  # new distance fields allowed this tick (the match resets it)
        self._links: list | None = None
        self.hl: list = []
        self.rl: list = []
        self.roofl: list = []

    # --- building -----------------------------------------------------------------------------------------------

    def room(self, x0: int, y0: int, x1: int, y1: int, floor: int = F_CONC, roof: bool = False,
             name: str = "") -> None:
        self.hgt[y0:y1, x0:x1] = 0.0
        self.floor[y0:y1, x0:x1] = floor
        self.roof[y0:y1, x0:x1] = roof
        if name:
            self.places.append((name, x0, y0, x1, y1))

    def block(self, x0: int, y0: int, x1: int, y1: int, h: float, mat: int = M_CRATE) -> None:
        self.hgt[y0:y1, x0:x1] = h
        self.mat[y0:y1, x0:x1] = mat

    def skin(self, x0: int, y0: int, x1: int, y1: int, mat: int) -> None:
        """The walls round a room: the full-height cells of the rectangle take ``mat``."""
        x0, y0 = max(x0, 0), max(y0, 0)
        sel = self.hgt[y0:y1, x0:x1] >= WALL
        self.mat[y0:y1, x0:x1][sel] = mat

    def stairs(self, cells: list, heights: list, mat: int = M_DECK) -> None:
        for (x0, y0, x1, y1), h in zip(cells, heights, strict=True):
            self.block(x0, y0, x1, y1, h, mat)

    def finish(self) -> None:
        """Lists for the fast loops, the bots' spots, and the top of the outer wall varied a little."""
        h = self.hgt
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = WALL
        rng = np.random.default_rng(len(self.name) * 7 + self.W)
        tall = (h >= WALL)
        bump = rng.random((self.H // 4 + 1, self.W // 4 + 1)).repeat(4, 0).repeat(4, 1)[:self.H, :self.W]
        h[tall] = WALL + np.round(bump[tall] * 3.0) * 0.8
        self._sync()
        for site in self.sites.values():
            self._make_spots(site)

    def _sync(self) -> None:
        self.hl = self.hgt.tolist()
        self.roofl = self.roof.tolist()
        res = np.zeros(self.hgt.shape, np.float64)
        for m, r in RESIST.items():
            res[self.mat == m] = r
        self.rl = res.tolist()
        self._links = None
        self._fields.clear()
        self.version += 1

    # --- the ground ---------------------------------------------------------------------------------------------

    def inside(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.W and 0 <= cy < self.H

    def height(self, x: float, y: float) -> float:
        cx, cy = int(x), int(y)
        if 0 <= cx < self.W and 0 <= cy < self.H:
            return self.hl[cy][cx]
        return WALL

    def ground(self, x: float, y: float, z: float, r: float = 0.22) -> float:
        """The floor under a walker at (x, y) whose feet are at z: the highest cell top under it that it can step
        onto."""
        hl, g, lim = self.hl, 0.0, z + STEP
        for ox, oy in ((-r, -r), (r, -r), (-r, r), (r, r)):
            cx, cy = int(x + ox), int(y + oy)
            if 0 <= cx < self.W and 0 <= cy < self.H:
                hh = hl[cy][cx]
                if g < hh <= lim:
                    g = hh
        return g

    def walkable(self, cx: int, cy: int) -> bool:
        return self.inside(cx, cy) and self.hl[cy][cx] < 2.0

    def place(self, x: float, y: float) -> str:
        for name, x0, y0, x1, y1 in self.places:
            if x0 <= x < x1 and y0 <= y < y1:
                return name
        return ""

    def site_at(self, x: float, y: float) -> str | None:
        for k, s in self.sites.items():
            if s.inside(x, y):
                return k
        return None

    def set_cells(self, cells: list, h: float, mat: int) -> list:
        """Raise ``cells`` (a barricade): gives back what was there for ``restore_cells``."""
        old = [(cx, cy, float(self.hgt[cy, cx]), int(self.mat[cy, cx])) for cx, cy in cells]
        for cx, cy in cells:
            self.hgt[cy, cx] = h
            self.mat[cy, cx] = mat
        self._sync()
        return old

    def restore_cells(self, old: list) -> None:
        for cx, cy, h, m in old:
            self.hgt[cy, cx] = h
            self.mat[cy, cx] = m
        self._sync()

    # --- rays ---------------------------------------------------------------------------------------------------

    def trace(self, ox: float, oy: float, oz: float, dx: float, dy: float, dz: float, maxd: float) -> tuple:
        """A bullet's way: (distance where it stops for good, [(t in, t out, resistance per metre), ...] the thin
        things it crossed before that)."""
        hl, rl, roofl, W, H = self.hl, self.rl, self.roofl, self.W, self.H
        cx, cy = int(ox), int(oy)
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        tdx = abs(1.0 / dx) if dx else INF
        tdy = abs(1.0 / dy) if dy else INF
        tmx = ((cx + 1 - ox) if dx > 0 else (ox - cx)) * tdx if dx else INF
        tmy = ((cy + 1 - oy) if dy > 0 else (oy - cy)) * tdy if dy else INF
        t, solids = 0.0, []
        while t < maxd:
            if not (0 <= cx < W and 0 <= cy < H):
                return t, solids
            tn = tmx if tmx < tmy else tmy
            if tn > maxd:
                tn = maxd
            h = hl[cy][cx]
            z1 = oz + dz * tn
            if h > 0.0:
                a, b = t, tn
                if dz < -1e-9:
                    th = (h - oz) / dz
                    if th > a:
                        a = th
                elif dz > 1e-9:
                    th = (h - oz) / dz
                    if th < b:
                        b = th
                elif oz >= h:
                    b = a
                if b - a > 1e-6:
                    r = rl[cy][cx]
                    if r >= INF:
                        return a, solids
                    if solids and abs(solids[-1][1] - a) < 1e-6 and solids[-1][2] == r:
                        solids[-1] = (solids[-1][0], b, r)
                    else:
                        solids.append((a, b, r))
            if z1 < 0.0:
                return (-oz / dz) if dz < 0 else tn, solids
            if z1 > ROOF and (roofl[cy][cx] or z1 > 14.0):
                return ((ROOF - oz) / dz) if roofl[cy][cx] and dz > 0 else tn, solids
            t = tn
            if tmx < tmy:
                tmx += tdx
                cx += sx
            else:
                tmy += tdy
                cy += sy
        return maxd, solids

    def los(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> bool:
        """Nothing of the map stands between two points."""
        dx, dy, dz = bx - ax, by - ay, bz - az
        hl, W, H = self.hl, self.W, self.H
        cx, cy = int(ax), int(ay)
        ex, ey = int(bx), int(by)
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        tdx = abs(1.0 / dx) if dx else INF
        tdy = abs(1.0 / dy) if dy else INF
        tmx = ((cx + 1 - ax) if dx > 0 else (ax - cx)) * tdx if dx else INF
        tmy = ((cy + 1 - ay) if dy > 0 else (ay - cy)) * tdy if dy else INF
        t = 0.0
        for _ in range(abs(ex - cx) + abs(ey - cy) + 1):
            if not (0 <= cx < W and 0 <= cy < H):
                return False
            tn = tmx if tmx < tmy else tmy
            if tn > 1.0:
                tn = 1.0
            h = hl[cy][cx]
            if h > 0.0:
                z0, z1 = az + dz * t, az + dz * tn
                if (z0 if z0 < z1 else z1) < h:
                    return False
            if cx == ex and cy == ey:
                return True
            t = tn
            if tmx < tmy:
                tmx += tdx
                cx += sx
            else:
                tmy += tdy
                cy += sy
        return True

    def clear_walk(self, ax: float, ay: float, bx: float, by: float, r: float = 0.38) -> bool:
        """A walker can go straight from a to b (no wall, no drop or step too high on the way)."""
        dx, dy = bx - ax, by - ay
        d = math.hypot(dx, dy)
        n = max(int(d / 0.45), 1)
        hl, W, H = self.hl, self.W, self.H
        prev = self.height(ax, ay)
        for i in range(1, n + 1):
            x, y = ax + dx * i / n, ay + dy * i / n
            top = 0.0
            for ox, oy in ((-r, -r), (r, -r), (-r, r), (r, r)):
                cx, cy = int(x + ox), int(y + oy)
                if not (0 <= cx < W and 0 <= cy < H):
                    return False
                hh = hl[cy][cx]
                if hh > prev + STEP:
                    return False
                if hh > top:
                    top = hh
            here = hl[int(y)][int(x)]
            if abs(here - prev) > STEP:
                return False
            prev = here
        return True

    # --- paths --------------------------------------------------------------------------------------------------

    def _make_links(self) -> list:
        h = self.hgt
        pad = np.full((self.H + 2, self.W + 2), 99.0, np.float32)
        pad[1:-1, 1:-1] = h
        ok = pad < 2.0
        links = []
        c = pad[1:-1, 1:-1]
        for dx, dy in DIRS8:
            n = pad[1 + dy:self.H + 1 + dy, 1 + dx:self.W + 1 + dx]
            lk = ok[1:-1, 1:-1] & (n < 2.0) & (np.abs(n - c) <= STEP)
            if dx and dy:  # a diagonal only past two open corners at the same level
                a = pad[1:-1, 1 + dx:self.W + 1 + dx]
                b = pad[1 + dy:self.H + 1 + dy, 1:-1]
                lk &= (np.abs(a - c) <= 0.05) & (np.abs(b - c) <= 0.05) & (np.abs(n - c) <= 0.05)
            links.append((dx, dy, lk))
        return links

    def field(self, x: float, y: float) -> np.ndarray:
        """Steps to the cell of (x, y) from every cell a walker can reach it from (FAR elsewhere)."""
        cx, cy = min(max(int(x), 0), self.W - 1), min(max(int(y), 0), self.H - 1)
        got = self._fields.get((cx, cy))
        if got is not None:
            return got
        if self._links is None:
            self._links = self._make_links()
        W, H = self.W, self.H
        dist = np.full((H, W), FAR, np.int16)
        front = np.zeros((H + 2, W + 2), bool)
        if self.hl[cy][cx] >= 2.0:  # a goal inside something solid: start from the open cells round it
            for dx, dy in DIRS8:
                if self.walkable(cx + dx, cy + dy):
                    front[cy + dy + 1, cx + dx + 1] = True
                    dist[cy + dy, cx + dx] = 0
        else:
            front[cy + 1, cx + 1] = True
            dist[cy, cx] = 0
        step = 0
        links = self._links
        while True:
            step += 1
            new = np.zeros((H, W), bool)
            for dx, dy, lk in links:  # a cell is reached from its neighbour in the front when the link allows it
                new |= front[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] & lk
            new &= dist == FAR
            if not new.any():
                break
            dist[new] = step
            front[1:-1, 1:-1] = new
        if len(self._fields) > 400:
            self._fields.clear()
        self._fields[(cx, cy)] = dist
        return dist

    def try_field(self, x: float, y: float) -> np.ndarray | None:
        """``field`` when it is known already or this tick may still work one out; else None (ask again)."""
        cx, cy = min(max(int(x), 0), self.W - 1), min(max(int(y), 0), self.H - 1)
        got = self._fields.get((cx, cy))
        if got is not None:
            return got
        if self.budget <= 0:
            return None
        self.budget -= 1
        return self.field(x, y)

    def next_point(self, dist: np.ndarray, x: float, y: float, ahead: int = 7) -> tuple | None:
        """Where to walk next along ``dist`` from (x, y): the furthest of the next cells reachable in a straight
        line."""
        cx, cy = int(x), int(y)
        if not self.inside(cx, cy):
            return None
        d = int(dist[cy, cx])
        if d >= FAR:  # off the walkable ground (on a crate): any neighbour that is on it
            best = None
            for dx, dy in DIRS8:
                nx, ny = cx + dx, cy + dy
                if self.inside(nx, ny) and dist[ny, nx] < FAR and (best is None or dist[ny, nx] < best[0]):
                    best = (int(dist[ny, nx]), nx + 0.5, ny + 0.5)
            return (best[1], best[2]) if best else None
        if d == 0:
            return None
        hl = self.hl
        h0 = hl[cy][cx]
        out = None
        for i in range(ahead):
            best = None
            for dx, dy in DIRS8:
                nx, ny = cx + dx, cy + dy
                if not self.inside(nx, ny):
                    continue
                nd = int(dist[ny, nx])
                if nd >= d or abs(hl[ny][nx] - hl[cy][cx]) > STEP:
                    continue
                if dx and dy and (hl[cy][nx] != hl[cy][cx] or hl[ny][cx] != hl[cy][cx]):
                    continue
                if best is None or nd < best[0] or (nd == best[0] and not (dx and dy)):
                    best = (nd, nx, ny)
            if best is None:
                break
            d, cx, cy = best
            px, py = cx + 0.5, cy + 0.5
            if i == 0 or (hl[cy][cx] == h0 and self.clear_walk(x, y, px, py)):
                out = (px, py)
            else:
                break
            if d == 0:
                break
        return out

    def path_len(self, ax: float, ay: float, bx: float, by: float) -> int:
        return int(self.field(bx, by)[min(max(int(ay), 0), self.H - 1), min(max(int(ax), 0), self.W - 1)])

    # --- where the bots stand -----------------------------------------------------------------------------------

    def _cover(self, cx: int, cy: int) -> int:
        n = 0
        for dx, dy in DIRS8:
            if self.inside(cx + dx, cy + dy) and self.hl[cy + dy][cx + dx] >= 1.0:
                n += 1
        return n

    def _pick_spots(self, cells: list, watch: list, lo: float, hi: float, want: int, seed: int) -> list:
        rng = np.random.default_rng(seed)
        cand = []
        for cx, cy in cells:
            if self.hl[cy][cx] > 1.3 or self.field(*watch[0])[cy, cx] >= FAR:
                continue
            cov = self._cover(cx, cy)
            if cov == 0:
                continue
            x, y = cx + 0.5, cy + 0.5
            z = self.hl[cy][cx] + EYE
            for wi, (wx, wy) in enumerate(watch):
                d = math.hypot(wx - x, wy - y)
                if lo <= d <= hi and self.los(x, y, z, wx, wy, 1.3):
                    cand.append((cov + rng.random() * 2.5 + (1.5 if self.hl[cy][cx] > 0.5 else 0.0), x, y, wi))
        cand.sort(reverse=True)
        out: list = []
        used = [0] * len(watch)
        for _ in range(2):  # first one spot per way in, then the best that are left
            for score, x, y, wi in cand:
                if len(out) >= want:
                    break
                if any(math.hypot(x - s.x, y - s.y) < 3.2 for s in out):
                    continue
                if used[wi] > min(used):
                    continue
                used[wi] += 1
                out.append(Spot(x, y, watch[wi][0], watch[wi][1]))
        return out

    def _make_spots(self, site: Site) -> None:
        x0, y0, x1, y1 = site.rect
        near = [(cx, cy) for cy in range(max(y0 - 4, 1), min(y1 + 4, self.H - 1))
                for cx in range(max(x0 - 4, 1), min(x1 + 4, self.W - 1)) if self.hl[cy][cx] < 1.4]
        ways_in = [(x, y) for _n, x, y, _sx, _sy in site.attack]
        ways_def = [(x, y) for _n, x, y in site.defend] + ways_in
        inside = [(cx, cy) for cx, cy in near if x0 - 1 <= cx < x1 + 1 and y0 - 1 <= cy < y1 + 1]
        site.holds = self._pick_spots(inside, ways_in, 5.0, 30.0, 7, 11 + len(site.name))
        site.posts = self._pick_spots(near, ways_def, 4.0, 26.0, 7, 23 + len(site.name))
        cx, cy = site.centre
        plants = []
        for px, py in inside:
            if x0 <= px < x1 and y0 <= py < y1 and self.hl[py][px] == 0.0 and self._cover(px, py) >= 1:
                plants.append((math.hypot(px + 0.5 - cx, py + 0.5 - cy), px + 0.5, py + 0.5))
        plants.sort()
        site.plants = [(x, y) for _d, x, y in plants[:6]] or [(cx, cy)]

    def zone_mask(self, rects: list) -> np.ndarray:
        m = np.zeros((self.H, self.W), bool)
        for x0, y0, x1, y1 in rects:
            m[y0:y1, x0:x1] = True
        return m


# --- KILNYARD --------------------------------------------------------------------------------------------------------

def _kilnyard() -> GameMap:
    m = GameMap("kilnyard", "KILNYARD", 60, 50)
    m.blurb = "A brickworks at closing time. Two long lanes, a clock yard in the middle, and a kiln to plant under."
    R = m.room
    R(24, 1, 36, 7, F_TILE, True, "Office Row")
    R(11, 2, 24, 6, F_TILE, True, "West Hall")
    R(36, 2, 49, 6, F_TILE, True, "East Hall")
    R(12, 6, 16, 8, F_TILE, True, "Kiln Door")
    R(44, 6, 48, 8, F_TILE, True, "Depot Door")
    R(3, 8, 20, 20, F_SITE, False, "Kiln Court")
    R(40, 8, 57, 20, F_SITE, False, "Depot")
    R(28, 7, 32, 12, F_WOOD, True, "Stair Hall")
    R(24, 12, 36, 30, F_GRAVEL, False, "Clock Yard")
    R(20, 12, 24, 15, F_CONC, True, "Kiln Link")
    R(36, 12, 40, 15, F_CONC, True, "Depot Link")
    R(28, 30, 32, 43, F_CONC, False, "Cart Lane")
    R(21, 43, 39, 49, F_PLATE, False, "Loading Bay")
    R(3, 42, 21, 46, F_CONC, False, "West Road")
    R(3, 20, 8, 42, F_GRAVEL, False, "Furnace Lane")
    R(15, 33, 28, 37, F_CONC, False, "Brick Alley")
    R(15, 20, 19, 33, F_CONC, False, "Brick Alley")
    R(39, 42, 57, 46, F_CONC, False, "East Road")
    R(52, 20, 57, 42, F_GRAVEL, False, "Rail Siding")
    R(32, 33, 45, 37, F_PLATE, True, "Coal Tunnel")
    R(41, 20, 45, 33, F_PLATE, True, "Coal Tunnel")
    m.skin(10, 0, 50, 8, M_WALL2)
    m.skin(19, 11, 41, 16, M_WALL2)
    m.skin(31, 32, 46, 38, M_TRIM)
    m.skin(40, 20, 46, 34, M_TRIM)
    B = m.block
    # the clock yard
    B(28, 17, 32, 22, WALL + 4.0, M_TRIM)  # the clock tower
    B(25, 24, 27, 26, 1.2, M_CRATE)
    B(33, 25, 35, 27, 1.0, M_CRATE)
    B(24, 16, 26, 17, 1.1, M_LOW)
    B(34, 16, 36, 17, 1.1, M_LOW)
    B(28, 30, 29, 31, WALL, M_WALL)
    B(31, 30, 32, 31, WALL, M_WALL)
    B(31, 36, 32, 38, 1.2, M_CRATE)
    # the kiln court (A)
    B(9, 12, 13, 16, WALL + 5.0, M_TRIM)  # the kiln
    B(15, 9, 17, 11, 1.3, M_CRATE)
    B(16, 16, 18, 18, 2.2, M_CRATE)
    B(8, 18, 12, 19, 2.4, M_PANEL)
    B(3, 16, 4, 18, 1.0, M_CRATE)
    B(4, 8, 8, 10, 1.2, M_DECK)
    m.stairs([(8, 8, 9, 10), (9, 8, 10, 10), (10, 8, 11, 10)], [0.9, 0.6, 0.3])
    B(13, 13, 14, 15, 1.1, M_LOW)
    # the depot (B)
    B(50, 8, 57, 11, 0.9, M_DECK)
    m.stairs([(49, 8, 50, 11), (48, 8, 49, 11)], [0.6, 0.3])
    B(46, 13, 50, 15, 2.6, M_STEEL)
    B(42, 10, 44, 12, 1.2, M_CRATE)
    B(53, 16, 55, 18, 1.3, M_CRATE)
    B(45, 17, 46, 20, 2.4, M_PANEL)
    B(51, 13, 52, 14, 1.0, M_CRATE)
    B(40, 17, 41, 19, 1.0, M_CRATE)
    # the lanes
    B(6, 30, 8, 32, 1.2, M_CRATE)
    B(3, 24, 5, 25, 1.0, M_CRATE)
    B(3, 36, 4, 38, 1.4, M_STEEL)
    B(55, 27, 57, 31, 1.6, M_STEEL)
    B(52, 36, 54, 37, 1.0, M_CRATE)
    B(10, 44, 12, 46, 1.2, M_CRATE)
    B(46, 42, 48, 44, 1.2, M_CRATE)
    B(15, 26, 16, 28, 1.0, M_CRATE)
    B(22, 33, 24, 34, 1.2, M_CRATE)
    B(38, 36, 40, 37, 1.1, M_CRATE)
    B(44, 26, 45, 28, 1.0, M_CRATE)
    B(19, 28, 20, 32, WALL, M_PANEL)  # a boarded-up bay in the alley's east wall
    B(24, 44, 25, 45, WALL, M_TRIM)
    B(35, 44, 36, 45, WALL, M_TRIM)
    B(12, 3, 13, 5, 1.0, M_CRATE)
    B(47, 3, 48, 5, 1.0, M_CRATE)
    B(29, 2, 31, 3, 1.1, M_LOW)
    m.sites["A"] = Site("A", "Kiln Court", (4, 9, 19, 19),
                        [("Furnace Lane", 5.5, 21.5, 5.5, 31.0), ("Brick Alley", 17.0, 21.5, 17.0, 31.0),
                         ("Kiln Link", 21.5, 13.5, 26.0, 27.5)],
                        [("Kiln Door", 14.0, 7.0)], [(14.0, 9.6), (19.0, 13.5)])
    m.sites["B"] = Site("B", "Depot", (41, 9, 56, 19),
                        [("Rail Siding", 54.5, 21.5, 54.5, 32.5), ("Coal Tunnel", 43.0, 21.5, 43.0, 31.0),
                         ("Depot Link", 38.5, 13.5, 34.0, 27.5)],
                        [("Depot Door", 46.0, 7.0)], [(46.0, 9.6), (41.0, 13.5)])
    m.spawns = {"atk": [(26.5, 47.0), (28.5, 46.0), (30.0, 47.5), (31.5, 46.0), (33.5, 47.0)],
                "def": [(27.0, 3.5), (28.5, 4.5), (30.0, 3.5), (31.5, 4.5), (33.0, 3.5)]}
    m.spawn_yaw = {"atk": -math.pi / 2, "def": math.pi / 2}
    m.zones = {"atk": m.zone_mask([(21, 43, 39, 49), (12, 42, 21, 46), (39, 42, 48, 46), (28, 38, 32, 43)]),
               "def": m.zone_mask([(0, 0, 60, 21), (24, 12, 36, 24)])}
    m.mid = (30.0, 13.5)
    m.lamps = [(30, 4, 3.6, (1.0, 0.86, 0.66), 9), (17, 4, 3.6, (1.0, 0.82, 0.6), 8), (43, 4, 3.6, (1.0, 0.82, 0.6), 8),
               (30, 9.5, 3.4, (1.0, 0.8, 0.55), 6), (22, 13.5, 3.2, (1.0, 0.7, 0.4), 5),
               (38, 13.5, 3.2, (1.0, 0.7, 0.4), 5), (38, 35, 3.4, (1.0, 0.62, 0.3), 7),
               (43, 27, 3.4, (1.0, 0.62, 0.3), 7), (14, 7, 3.2, (1.0, 0.8, 0.55), 4), (46, 7, 3.2, (1.0, 0.8, 0.55), 4)]
    m.scene = dict(
        sun=(-0.5, -0.74, -0.45), sun_col=(1.0, 0.76, 0.54), sun_k=1.12, amb_sky=(0.6, 0.57, 0.7),
        amb_gnd=(0.46, 0.36, 0.32), amb_in=(0.4, 0.36, 0.36), fog=(0.86, 0.62, 0.5), fog_d=150.0,
        sky_top=(0.3, 0.36, 0.62), sky_low=(0.98, 0.66, 0.44), sun_disc=(1.0, 0.86, 0.62), accent=(1.0, 0.72, 0.2),
        pal={F_CONC: (0.56, 0.52, 0.5), F_TILE: (0.66, 0.56, 0.46), F_WOOD: (0.5, 0.36, 0.26),
             F_PLATE: (0.4, 0.4, 0.43), F_GRAVEL: (0.55, 0.47, 0.4), F_SITE: (0.62, 0.55, 0.5),
             M_WALL: (0.64, 0.33, 0.25), M_WALL2: (0.84, 0.75, 0.62), M_PANEL: (0.56, 0.4, 0.25),
             M_CRATE: (0.74, 0.54, 0.3), M_STEEL: (0.25, 0.3, 0.36), M_TRIM: (0.36, 0.31, 0.31),
             M_DECK: (0.44, 0.4, 0.4), M_ROOF: (0.3, 0.25, 0.24), M_LOW: (0.6, 0.57, 0.54),
             M_BARRICADE: (0.35, 0.6, 1.0)})
    m.finish()
    return m


# --- COLD HARBOR -----------------------------------------------------------------------------------------------------

def _cold_harbor() -> GameMap:
    m = GameMap("cold_harbor", "COLD HARBOR", 64, 52)
    m.blurb = "A fish dock before the boats come in. An open slipway to the north-west, an icehouse to the east."
    R = m.room
    R(10, 42, 26, 50, F_WOOD, False, "Ferry Ramp")
    R(4, 42, 10, 46, F_WOOD, False, "Ferry Ramp")
    R(30, 1, 46, 7, F_TILE, True, "Customs")
    R(4, 6, 23, 20, F_SITE, False, "Slipway")
    R(44, 18, 61, 32, F_SITE, True, "Icehouse")
    R(4, 20, 9, 42, F_WOOD, False, "The Quay")
    R(26, 44, 52, 48, F_CONC, False, "Container Row")
    R(46, 32, 51, 44, F_CONC, False, "Reefer Alley")
    R(24, 34, 29, 44, F_CONC, True, "Net Loft")
    R(22, 22, 38, 34, F_TILE, True, "Fish Market")
    R(17, 24, 22, 28, F_CONC, False, "Chain Walk")
    R(13, 20, 17, 28, F_CONC, False, "Chain Walk")
    R(38, 25, 44, 29, F_PLATE, True, "Cold Door")
    R(23, 3, 30, 8, F_PLATE, False, "Crane Deck")
    R(30, 7, 34, 22, F_WOOD, False, "Winch Stair")
    R(44, 7, 49, 18, F_TILE, True, "Scale House")
    m.skin(43, 17, 62, 33, M_WALL2)
    m.skin(21, 21, 39, 35, M_WALL2)
    m.skin(29, 0, 50, 19, M_TRIM)
    m.skin(23, 33, 30, 45, M_WALL2)
    B = m.block
    # the slipway (A)
    B(10, 10, 17, 13, 2.8, M_STEEL)  # the hull on its cradle
    B(4, 6, 8, 9, 1.2, M_DECK)
    m.stairs([(8, 6, 9, 9), (9, 6, 10, 9), (10, 6, 11, 9)], [0.9, 0.6, 0.3])
    B(19, 10, 21, 12, 1.3, M_CRATE)
    B(6, 15, 8, 17, 1.2, M_CRATE)
    B(18, 16, 20, 17, 2.3, M_PANEL)
    B(12, 17, 13, 19, 1.0, M_CRATE)
    B(20, 14, 22, 15, 1.1, M_LOW)
    B(14, 7, 16, 8, 1.1, M_LOW)
    # the icehouse (B)
    for px, py in ((49, 22), (55, 22), (49, 27), (55, 27)):
        B(px, py, px + 1, py + 1, WALL, M_TRIM)
    B(52, 24, 54, 26, 1.2, M_CRATE)
    B(57, 20, 58, 26, WALL, M_PANEL)
    B(45, 29, 47, 31, 1.3, M_CRATE)
    B(58, 29, 60, 31, 1.0, M_CRATE)
    B(51, 19, 53, 20, 1.1, M_LOW)
    B(45, 20, 46, 22, 1.0, M_CRATE)
    # the market
    B(26, 25, 28, 27, 1.1, M_CRATE)
    B(32, 29, 34, 31, 1.1, M_CRATE)
    B(30, 24, 31, 28, 2.2, M_PANEL)
    B(35, 23, 37, 24, 1.0, M_CRATE)
    B(23, 30, 24, 32, 1.0, M_CRATE)
    # the lanes
    B(4, 30, 6, 32, 1.0, M_CRATE)
    B(7, 36, 9, 37, 1.3, M_CRATE)
    B(4, 24, 5, 26, 1.4, M_STEEL)
    B(30, 44, 34, 46, 2.6, M_STEEL)
    B(40, 46, 44, 48, 2.6, M_STEEL)
    B(28, 47, 30, 48, 1.1, M_CRATE)
    B(46, 37, 48, 38, 1.2, M_CRATE)
    B(50, 40, 51, 42, 1.0, M_CRATE)
    B(24, 38, 25, 40, 1.0, M_CRATE)
    B(15, 23, 16, 25, 1.0, M_CRATE)
    B(26, 3, 28, 4, 1.2, M_CRATE)
    B(25, 6, 27, 8, 2.6, M_STEEL)
    B(30, 14, 31, 16, 1.1, M_LOW)
    B(33, 18, 34, 20, 1.1, M_LOW)
    B(44, 11, 45, 13, 1.0, M_CRATE)
    B(37, 5, 39, 6, 1.1, M_LOW)
    B(40, 25, 41, 26, 1.0, M_CRATE)
    B(14, 45, 15, 47, WALL, M_TRIM)
    B(21, 45, 22, 47, WALL, M_TRIM)
    m.sites["A"] = Site("A", "Slipway", (5, 7, 22, 19),
                        [("The Quay", 6.5, 21.5, 6.5, 33.0), ("Chain Walk", 15.0, 21.5, 26.5, 37.0)],
                        [("Crane Deck", 24.5, 7.0)], [(22.0, 7.0), (17.0, 8.5)])
    m.sites["B"] = Site("B", "Icehouse", (45, 19, 60, 31),
                        [("Reefer Alley", 48.5, 33.5, 48.5, 42.0), ("Cold Door", 42.5, 27.0, 34.0, 27.5)],
                        [("Scale House", 46.5, 16.5)], [(46.5, 19.5), (46.0, 25.0)])
    m.spawns = {"atk": [(13.0, 46.5), (16.0, 48.0), (18.0, 46.5), (20.0, 48.0), (23.0, 46.5)],
                "def": [(34.0, 3.5), (36.0, 4.5), (38.0, 3.5), (40.0, 4.5), (42.0, 3.5)]}
    m.spawn_yaw = {"atk": -math.pi / 2, "def": math.pi / 2}
    m.zones = {"atk": m.zone_mask([(10, 42, 26, 50), (4, 42, 10, 46), (26, 44, 34, 48), (24, 40, 29, 44)]),
               "def": m.zone_mask([(0, 0, 64, 21), (44, 18, 62, 32), (30, 21, 34, 24)])}
    m.mid = (32.0, 11.0)
    m.lamps = [(38, 4, 3.6, (0.8, 0.92, 1.0), 10), (46.5, 12, 3.6, (0.8, 0.92, 1.0), 7),
               (48, 22, 3.6, (0.7, 0.9, 1.0), 9), (57, 28, 3.6, (0.7, 0.9, 1.0), 9), (52, 28, 3.6, (0.7, 0.9, 1.0), 8),
               (27, 26, 3.6, (1.0, 0.92, 0.75), 9), (34, 31, 3.6, (1.0, 0.92, 0.75), 9),
               (26.5, 39, 3.4, (1.0, 0.86, 0.6), 7), (41, 27, 3.2, (0.7, 0.9, 1.0), 5)]
    m.scene = dict(
        sun=(0.45, -0.72, 0.53), sun_col=(0.92, 0.95, 1.0), sun_k=0.85, amb_sky=(0.56, 0.62, 0.68),
        amb_gnd=(0.3, 0.34, 0.36), amb_in=(0.36, 0.4, 0.44), fog=(0.72, 0.78, 0.82), fog_d=110.0,
        sky_top=(0.5, 0.58, 0.66), sky_low=(0.82, 0.86, 0.88), sun_disc=(1.0, 1.0, 0.96), accent=(1.0, 0.84, 0.16),
        pal={F_CONC: (0.5, 0.53, 0.55), F_TILE: (0.7, 0.76, 0.78), F_WOOD: (0.46, 0.39, 0.32),
             F_PLATE: (0.36, 0.42, 0.46), F_GRAVEL: (0.48, 0.5, 0.5), F_SITE: (0.57, 0.6, 0.62),
             M_WALL: (0.6, 0.66, 0.68), M_WALL2: (0.2, 0.46, 0.5), M_PANEL: (0.76, 0.78, 0.72),
             M_CRATE: (0.84, 0.64, 0.24), M_STEEL: (0.15, 0.34, 0.52), M_TRIM: (0.33, 0.37, 0.41),
             M_DECK: (0.42, 0.44, 0.46), M_ROOF: (0.26, 0.3, 0.33), M_LOW: (0.58, 0.6, 0.6),
             M_BARRICADE: (0.35, 0.6, 1.0)})
    m.finish()
    return m


MAPS = {"kilnyard": _kilnyard, "cold_harbor": _cold_harbor}
MAP_ORDER = ("kilnyard", "cold_harbor")
MAP_INFO = {"kilnyard": ("KILNYARD", "A brickworks at closing time. Two long lanes and a clock yard between them."),
            "cold_harbor": ("COLD HARBOR", "A fish dock on a grey morning. An open slipway and a roofed icehouse.")}


def build(key: str) -> GameMap:
    return MAPS[key]()
