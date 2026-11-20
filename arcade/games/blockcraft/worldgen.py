"""Seeded gradient noise (vectorised numpy) and the BlockCraft terrain generator.

The world has no edge: every column is a pure function of the seed and its coordinates, so terrain is generated a
*region* (4 x 4 chunks) at a time, wherever the player goes, and always comes out the same. Continents and oceans,
plains, forests, deserts, taiga, snowy plains and mountains, rivers, caves, ores, lava, trees and plants, sea
grass and kelp, and here and there a hamlet: a well, a few huts and a field."""

from __future__ import annotations

import numpy as np

from app.arcade.games.blockcraft import blocks as B

CHUNK = 16  # blocks along x and z
HEIGHT = 128  # blocks of height (y 0..127)
SEA = 62  # the water level (the highest water block)
REGION = 4  # chunks along a region's side
RSIZE = CHUNK * REGION
MARGIN = 4  # columns computed around a region (trees rooted outside it reach in)

PLAINS, FOREST, BIRCH_FOREST, DESERT, MOUNTAINS, BEACH, OCEAN, TAIGA, SNOWY = range(9)
BIOME_NAMES = ("plains", "forest", "birch_forest", "desert", "mountains", "beach", "ocean", "taiga", "snowy_plains")

_U = np.uint64


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def hash01(x: np.ndarray, z: np.ndarray, salt: int) -> np.ndarray:
    """A random number in [0, 1) for every (x, z) integer pair (the same every time)."""
    with np.errstate(over="ignore"):
        h = (np.asarray(x).astype(np.int64).astype(_U) * _U(0x9E3779B97F4A7C15)) ^ \
            (np.asarray(z).astype(np.int64).astype(_U) * _U(0xC2B2AE3D27D4EB4F)) ^ _U(salt & 0xFFFFFFFFFFFFFFFF)
        h ^= h >> _U(29)
        h *= _U(0xBF58476D1CE4E5B9)
        h ^= h >> _U(32)
        h *= _U(0x94D049BB133111EB)
        h ^= h >> _U(29)
    return (h >> _U(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


class Noise:
    """Perlin gradient noise in 2D and 3D from a seed."""

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        p = rng.permutation(256).astype(np.int32)
        self.p = np.concatenate([p, p])
        ang = rng.uniform(0, 2 * np.pi, 256)
        self.g2 = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
        v = rng.normal(size=(256, 3))
        self.g3 = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    def n2(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0, y0 = np.floor(x), np.floor(y)
        xf, yf = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
        xi, yi = x0.astype(np.int64) & 255, y0.astype(np.int64) & 255
        p, g = self.p, self.g2
        a, b = p[xi], p[xi + 1]
        aa, ab, ba, bb = p[a + yi], p[a + yi + 1], p[b + yi], p[b + yi + 1]

        def dot(h: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
            return g[h, 0] * dx + g[h, 1] * dy

        u, v = _fade(xf), _fade(yf)
        x1 = dot(aa, xf, yf) + u * (dot(ba, xf - 1, yf) - dot(aa, xf, yf))
        x2 = dot(ab, xf, yf - 1) + u * (dot(bb, xf - 1, yf - 1) - dot(ab, xf, yf - 1))
        return (x1 + v * (x2 - x1)) * 1.4

    def n3(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
        xf, yf, zf = (x - x0).astype(np.float32), (y - y0).astype(np.float32), (z - z0).astype(np.float32)
        xi, yi, zi = x0.astype(np.int64) & 255, y0.astype(np.int64) & 255, z0.astype(np.int64) & 255
        p, g = self.p, self.g3

        def h(i: int, j: int, k: int) -> np.ndarray:
            return p[p[p[xi + i] + yi + j] + zi + k]

        def dot(hh: np.ndarray, dx, dy, dz) -> np.ndarray:  # noqa: ANN001
            return g[hh, 0] * dx + g[hh, 1] * dy + g[hh, 2] * dz

        u, v, w = _fade(xf), _fade(yf), _fade(zf)
        c = {}
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    c[i, j, k] = dot(h(i, j, k), xf - i, yf - j, zf - k)
        x00 = c[0, 0, 0] + u * (c[1, 0, 0] - c[0, 0, 0])
        x10 = c[0, 1, 0] + u * (c[1, 1, 0] - c[0, 1, 0])
        x01 = c[0, 0, 1] + u * (c[1, 0, 1] - c[0, 0, 1])
        x11 = c[0, 1, 1] + u * (c[1, 1, 1] - c[0, 1, 1])
        y0_ = x00 + v * (x10 - x00)
        y1_ = x01 + v * (x11 - x01)
        return (y0_ + w * (y1_ - y0_)) * 1.2

    def fbm2(self, x: np.ndarray, y: np.ndarray, octaves: int = 4, gain: float = 0.5) -> np.ndarray:
        total = np.zeros(np.broadcast(x, y).shape, np.float32)
        amp, freq, norm = 1.0, 1.0, 0.0
        for o in range(octaves):
            total += amp * self.n2(x * freq + o * 31.7, y * freq - o * 17.3)
            norm += amp
            amp *= gain
            freq *= 2.0
        return total / norm


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _upsample(coarse: np.ndarray, step: tuple[int, int, int], shape: tuple[int, int, int]) -> np.ndarray:
    """Trilinear interpolation of a coarse lattice (every ``step`` blocks) to the full grid."""
    out = coarse
    for axis, (s, n) in enumerate(zip(step, shape, strict=True)):
        pos = np.arange(n) / s
        i0 = np.minimum(pos.astype(int), out.shape[axis] - 2)
        f = (pos - i0).astype(np.float32)
        a = np.take(out, i0, axis=axis)
        b = np.take(out, i0 + 1, axis=axis)
        sh = [1, 1, 1]
        sh[axis] = n
        out = a + (b - a) * f.reshape(sh)
    return out


# --- trees: (offsets (K, 3) of x, y, z from the block above the root, blocks (K,), is a log (K,)) -------------------

def _round_tree(height: int, log: int, leaves: int, rng: np.random.Generator) -> tuple:
    cells: dict[tuple[int, int, int], int] = {}
    top = height - 1
    for ly in range(top - 3, top + 2):
        rad = 2 if ly <= top - 2 else 1
        for dx in range(-rad, rad + 1):
            for dz in range(-rad, rad + 1):
                corner = abs(dx) == rad and abs(dz) == rad
                if corner and (ly == top + 1 or rng.random() < 0.5):
                    continue
                if ly == top + 1 and abs(dx) + abs(dz) > 1:
                    continue
                cells[dx, ly, dz] = leaves
    for y in range(height):
        cells[0, y, 0] = log
    return _template(cells, log)


def _cone_tree(height: int, log: int, leaves: int, rng: np.random.Generator) -> tuple:
    cells: dict[tuple[int, int, int], int] = {}
    top = height - 1
    cells[0, top + 1, 0] = leaves
    y, wide = top, False
    radii = []
    while y >= 2 + int(rng.integers(0, 2)):
        radii.append((y, 1 if not wide else 2))
        wide = not wide
        y -= 1
    for y, rad in radii:
        for dx in range(-rad, rad + 1):
            for dz in range(-rad, rad + 1):
                if abs(dx) == rad and abs(dz) == rad and (rad > 1 or rng.random() < 0.6):
                    continue
                cells[dx, y, dz] = leaves
    for y in range(height):
        cells[0, y, 0] = log
    return _template(cells, log)


def _template(cells: dict, log: int) -> tuple:
    offs = np.array(list(cells), np.int64)
    blocks = np.array(list(cells.values()), np.uint8)
    return offs, blocks, blocks == log


def _tree_templates() -> dict:
    rng = np.random.default_rng(4242)
    return {
        "oak": [_round_tree(h, B.OAK_LOG, B.OAK_LEAVES, rng) for h in (4, 5, 5, 6, 6, 7)],
        "birch": [_round_tree(h, B.BIRCH_LOG, B.BIRCH_LEAVES, rng) for h in (5, 6, 6, 7, 7, 8)],
        "spruce": [_cone_tree(h, B.SPRUCE_LOG, B.SPRUCE_LEAVES, rng) for h in (6, 7, 7, 8, 9, 10)],
    }


TREES = _tree_templates()
TREE_KINDS = ("oak", "birch", "spruce")
_SOFT = np.zeros(256, bool)  # what a tree's leaves may grow into
_SOFT[[B.AIR, B.TALL_GRASS, B.FERN, B.DEAD_BUSH, B.RED_MUSHROOM, B.BROWN_MUSHROOM, *B.FLOWERS]] = True
HAMLET_CELL = 160  # one place a hamlet may stand in every square of this many blocks
HAMLET_REACH = 22  # blocks from its well to its farthest wall
_HUT_SPOTS = ((-11, -9), (10, -10), (-10, 10), (11, 9))
_FIELD_SPOT = (0, -15)


def make_generator(seed: int, dim: str = "over"):  # noqa: ANN201
    """The generator of a realm of a seed: the overworld ("over"), the Deep ("deep") or the Sky Isles ("isles")."""
    if dim in ("deep", "isles"):
        from app.arcade.games.blockcraft.realms import realm_generator

        return realm_generator(seed, dim)
    return Generator(seed)


class Generator:
    """The terrain of one seed (the overworld)."""

    dim = "over"
    sea = SEA
    names = BIOME_NAMES

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.noise = Noise(self.seed)
        self.salt = (self.seed * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFFFFFF
        self._sites: dict[tuple[int, int], dict | None] = {}

    # --- columns: everything about the surface, for any x and z ------------------------------------------------------

    def columns(self, X: np.ndarray, Z: np.ndarray) -> dict:
        """Surface height, biome and soil for the given block coordinates (float32 arrays of one shape)."""
        n = self.noise
        c = n.fbm2(X / 560 + 0.5, Z / 560 + 0.5, 4) * 1.2 + 0.17  # continents: < 0 is ocean
        land = np.clip(c * 6, 0, 1)
        hills = n.fbm2(X / 120 + 100, Z / 120 + 100, 3)
        detail = n.fbm2(X / 24 + 200, Z / 24 + 200, 2)
        mnt = _smoothstep(0.06, 0.4, n.fbm2(X / 310 + 300, Z / 310 + 300, 3)) * _smoothstep(0.04, 0.28, c)
        ridge = np.clip(1 - np.abs(n.fbm2(X / 90 + 400, Z / 90 + 400, 4)) * 2.2, 0, 1)
        h = np.where(c < 0, SEA + c * 62, SEA + 1 + c * 24)
        h = h + hills * (2 + 9 * land) + detail * 1.6 + (ridge * ridge * 50 + hills * 10 + 6) * mnt
        # rivers: narrow valleys along the zero lines of a noise, cut down below the sea
        river = np.abs(n.fbm2(X / 230 + 1200, Z / 230 + 1200, 3))
        cut = _smoothstep(0.05, 0.0, river) * land * (1 - _smoothstep(0.25, 0.7, mnt))
        h = h + (np.minimum(h, SEA - 3) - h) * cut
        h = np.clip(np.round(h), 6, HEIGHT - 14).astype(np.int32)
        temp = n.fbm2(X / 440 + 500, Z / 440 + 500, 3) - (h - SEA) * 0.0025
        humid = n.fbm2(X / 350 + 600, Z / 350 + 600, 3)
        biome = np.full(h.shape, PLAINS, np.uint8)
        biome[humid > 0.07] = FOREST
        biome[(humid > 0.07) & (temp < 0.0) & (hills > 0)] = BIRCH_FOREST
        biome[(temp > 0.15) & (humid < 0.05)] = DESERT
        biome[temp < -0.17] = TAIGA
        biome[(temp < -0.17) & (humid < -0.02)] = SNOWY
        biome[(mnt > 0.45) & (h > SEA + 20)] = MOUNTAINS
        biome[(h <= SEA + 2) & ((c < 0.08) | (cut > 0.3))] = BEACH
        biome[h < SEA] = OCEAN
        # the soil
        gx = np.gradient(h.astype(np.float32)) if h.ndim == 2 and min(h.shape) > 1 else (np.zeros_like(X),) * 2
        steep = (np.abs(gx[0]) + np.abs(gx[1])) > 2.2
        top = np.full(h.shape, B.GRASS, np.uint8)
        filler = np.full(h.shape, B.DIRT, np.uint8)
        sandy = (biome == DESERT) | (biome == BEACH)
        top[sandy], filler[sandy] = B.SAND, B.SAND
        cold = temp < -0.17
        top[(biome == SNOWY) | ((biome == TAIGA) & (temp < -0.27))] = B.SNOWY_GRASS
        under = h < SEA
        floor = n.fbm2(X / 9 + 700, Z / 9 + 700, 2)
        top[under] = np.where(h[under] >= SEA - 5, B.SAND,
                              np.where(floor[under] > 0.2, B.GRAVEL, np.where(floor[under] < -0.25, B.CLAY, B.DIRT)))
        filler[under] = np.where(h[under] >= SEA - 5, B.SAND, B.DIRT)
        rocky = (biome == MOUNTAINS) & (steep | (h > SEA + 34))
        top[rocky], filler[rocky] = B.STONE, B.STONE
        top[(biome == MOUNTAINS) & (h > SEA + 44 - (cold * 14))] = B.SNOW
        depth = (3 + (detail > 0)).astype(np.int32)
        return {"h": h, "biome": biome, "top": top, "filler": filler, "depth": depth, "sandy": sandy,
                "frozen": cold & under & (temp < -0.2)}

    def find_spawn(self) -> tuple[float, float, float]:
        """Dry, green land near the middle of the world."""
        xs = np.arange(-320, 321, 8, dtype=np.float32)
        X, Z = np.meshgrid(xs, xs, indexing="ij")
        col = self.columns(X, Z)
        ok = (col["h"] > SEA + 1) & (col["top"] == B.GRASS) & (col["biome"] != MOUNTAINS)
        if not ok.any():
            return 0.5, float(SEA + 40), 0.5
        d = X * X + Z * Z + np.where(ok, 0, 1e12)
        i = np.unravel_index(int(np.argmin(d)), d.shape)
        return float(X[i]) + 0.5, float(col["h"][i]) + 1.0, float(Z[i]) + 0.5

    # --- a region of 4 x 4 chunks ------------------------------------------------------------------------------------

    def region(self, rx: int, rz: int) -> tuple[np.ndarray, np.ndarray]:
        """The blocks (64, 64, HEIGHT) indexed [x, z, y] and the biomes (64, 64) of region (rx, rz)."""
        S, M, H = RSIZE, MARGIN, HEIGHT
        x0, z0 = rx * S, rz * S
        xs = np.arange(x0 - M, x0 + S + M, dtype=np.float32)
        zs = np.arange(z0 - M, z0 + S + M, dtype=np.float32)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        col = self.columns(X, Z)
        inner = (slice(M, M + S), slice(M, M + S))
        h, top, filler = col["h"][inner], col["top"][inner], col["filler"][inner]
        depth = col["depth"][inner]
        Y = np.arange(H, dtype=np.int32)[None, None, :]
        rel = Y - h[:, :, None]
        d3 = depth[:, :, None]
        blk = np.where(rel < -d3, B.STONE,
                       np.where(rel < 0, filler[:, :, None],
                                np.where(rel == 0, top[:, :, None], np.where(Y <= SEA, B.WATER, B.AIR)))).astype(np.uint8)
        blk[col["sandy"][inner][:, :, None] & (rel >= -d3 - 3) & (rel < -d3)] = B.SANDSTONE
        ice = col["frozen"][inner]
        blk[:, :, SEA][ice & (h < SEA)] = B.ICE
        xi = np.arange(x0, x0 + S, dtype=np.int64)[:, None]
        zi = np.arange(z0, z0 + S, dtype=np.int64)[None, :]
        blk[:, :, 0] = B.BEDROCK
        for y in (1, 2, 3):
            blk[:, :, y][hash01(xi, zi, self.salt + 100 + y) < (0.75 - y * 0.2)] = B.BEDROCK
        self._caves(blk, x0, z0, h)
        self._ores(blk, rx, rz)
        self._trees(blk, x0, z0, col)
        self._plants(blk, xi, zi, col)
        for site in self.hamlets(x0 - HAMLET_REACH, z0 - HAMLET_REACH, x0 + S + HAMLET_REACH, z0 + S + HAMLET_REACH):
            self._build_hamlet(blk, x0, z0, site)
        return blk, col["biome"][inner].copy()

    def _caves(self, blk: np.ndarray, x0: int, z0: int, h: np.ndarray) -> None:
        S, H = RSIZE, HEIGHT
        n = self.noise
        step = (4, 4, 4)
        cx = np.arange(x0, x0 + S + 4, 4, dtype=np.float32)
        cz = np.arange(z0, z0 + S + 4, 4, dtype=np.float32)
        cy = np.arange(0, H + 4, 4, dtype=np.float32)
        CX, CZ, CY = np.meshgrid(cx, cz, cy, indexing="ij")
        n1 = n.n3(CX / 38 + 11.3, CY / 22 + 3.1, CZ / 38 + 5.7)
        n2 = n.n3(CX / 38 + 91.3, CY / 22 + 43.1, CZ / 38 + 25.7)
        n3 = n.n3(CX / 78 + 51.3, CY / 34 + 13.1, CZ / 78 + 75.7)
        sp = _upsample(n1 * n1 + n2 * n2, step, (S, S, H))
        ch = _upsample(n3, step, (S, S, H))
        Yf = np.arange(H, dtype=np.float32)[None, None, :]
        # tunnels everywhere underground, big caverns deep down; never right under the sea floor
        limit = np.where(h < SEA + 1, h - 5, h + 1)[:, :, None]
        cave = ((sp < 0.011 + 0.006 * (Yf < SEA - 14)) | ((ch > 0.4) & (Yf < SEA - 12))) & (Yf >= 1) & (Yf < limit)
        cave &= (blk != B.BEDROCK) & (blk != B.WATER) & (blk != B.ICE)
        blk[cave] = B.AIR
        blk[cave & (Yf < 11)] = B.LAVA

    def _ores(self, blk: np.ndarray, rx: int, rz: int) -> None:
        S, H = RSIZE, HEIGHT
        rng = np.random.default_rng([self.seed & 0xFFFFFFFF, rx & 0xFFFFFFFF, rz & 0xFFFFFFFF, 7])
        chunks = REGION * REGION
        for ore, per_chunk, size, ymax in ((B.COAL_ORE, 18, 12, 112), (B.IRON_ORE, 15, 8, 62),
                                           (B.GOLD_ORE, 2.4, 7, 30), (B.REDSTONE_ORE, 6, 7, 17),
                                           (B.DIAMOND_ORE, 1.3, 6, 15), (B.GRAVEL, 5, 26, 108),
                                           (B.DIRT, 5, 26, 108)):
            n = int(per_chunk * chunks)
            centre = np.stack([rng.uniform(0, S, n), rng.uniform(1, ymax, n), rng.uniform(0, S, n)], 1)
            pts = centre[:, None, :] + np.cumsum(rng.normal(0, 0.8, (n, size, 3)), 1)
            pts = np.floor(pts.reshape(-1, 3)).astype(np.int64)
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < S) & (pts[:, 2] >= 0) & (pts[:, 2] < S) & (pts[:, 1] >= 1) & \
                (pts[:, 1] < H)
            pts = pts[ok]
            pts = pts[blk[pts[:, 0], pts[:, 2], pts[:, 1]] == B.STONE]
            blk[pts[:, 0], pts[:, 2], pts[:, 1]] = ore

    def _trees(self, blk: np.ndarray, x0: int, z0: int, col: dict) -> None:
        """One tree candidate in every 4 x 4 cell of the world; the ones rooted just outside the region reach in."""
        S, M, H = RSIZE, MARGIN, HEIGHT
        cells = (S + 2 * M) // 4
        gx = (np.arange(cells, dtype=np.int64) + (x0 - M) // 4)[:, None] + np.zeros((1, cells), np.int64)
        gz = (np.arange(cells, dtype=np.int64) + (z0 - M) // 4)[None, :] + np.zeros((cells, 1), np.int64)
        s = self.salt
        x = gx * 4 + (hash01(gx, gz, s + 1) * 4).astype(np.int64)
        z = gz * 4 + (hash01(gx, gz, s + 2) * 4).astype(np.int64)
        roll, kind_roll = hash01(gx, gz, s + 3), hash01(gx, gz, s + 4)
        variant = (hash01(gx, gz, s + 5) * 6).astype(np.int64)
        ix, iz = x - (x0 - M), z - (z0 - M)
        bm = col["biome"][ix, iz]
        h = col["h"][ix, iz]
        top = col["top"][ix, iz]
        density = np.zeros(bm.shape)
        for b, dens in ((PLAINS, 0.03), (FOREST, 0.5), (BIRCH_FOREST, 0.5), (MOUNTAINS, 0.05), (TAIGA, 0.42),
                        (SNOWY, 0.03)):
            density[bm == b] = dens
        ok = (roll < density) & ((top == B.GRASS) | (top == B.SNOWY_GRASS)) & (h >= SEA) & (h < H - 16)
        kind = np.zeros(bm.shape, np.int64)  # 0 oak, 1 birch, 2 spruce
        kind[((bm == BIRCH_FOREST) & (kind_roll < 0.8)) | ((bm == FOREST) & (kind_roll < 0.2))] = 1
        kind[(bm == TAIGA) | (bm == SNOWY) | ((bm == MOUNTAINS) & (kind_roll < 0.5))] = 2
        for k, name in enumerate(TREE_KINDS):
            for v in range(6):
                sel = ok & (kind == k) & (variant == v)
                if not sel.any():
                    continue
                offs, blocks, is_log = TREES[name][v]
                px = (x[sel] - x0)[:, None] + offs[None, :, 0]
                py = (h[sel] + 1)[:, None] + offs[None, :, 1]
                pz = (z[sel] - z0)[:, None] + offs[None, :, 2]
                b = np.broadcast_to(blocks, px.shape)
                lg = np.broadcast_to(is_log, px.shape)
                inside = (px >= 0) & (px < S) & (pz >= 0) & (pz < S) & (py < H)
                px, py, pz, b, lg = px[inside], py[inside], pz[inside], b[inside], lg[inside]
                free = lg | _SOFT[blk[px, pz, py]]
                blk[px[free], pz[free], py[free]] = b[free]
                rx_, rz_, ry_ = x[sel] - x0, z[sel] - z0, h[sel]
                under = (rx_ >= 0) & (rx_ < S) & (rz_ >= 0) & (rz_ < S)
                blk[rx_[under], rz_[under], ry_[under]] = B.DIRT

    def _plants(self, blk: np.ndarray, xi: np.ndarray, zi: np.ndarray, col: dict) -> None:
        S, M, H = RSIZE, MARGIN, HEIGHT
        inner = (slice(M, M + S), slice(M, M + S))
        hf = col["h"]
        h, biome = hf[inner], col["biome"][inner]
        ax, az = np.arange(S)[:, None], np.arange(S)[None, :]
        ground = blk[ax, az, np.minimum(h, H - 3)]
        above = blk[ax, az, np.minimum(h + 1, H - 2)]
        above_air = above == B.AIR
        s = self.salt
        roll, kind = hash01(xi, zi, s + 11), hash01(xi, zi, s + 12)
        plant = np.zeros((S, S), np.uint8)
        green = (biome == FOREST) | (biome == BIRCH_FOREST)
        taiga = biome == TAIGA
        p_grass = np.where(biome == PLAINS, 0.16, np.where(green, 0.09, np.where(taiga, 0.07, 0.03)))
        tuft = roll < p_grass
        plant[tuft] = B.TALL_GRASS
        plant[tuft & ((taiga & (kind < 0.7)) | (green & (kind < 0.28)))] = B.FERN
        flower = roll >= 1 - np.where(biome == PLAINS, 0.026, 0.011)
        kinds = np.array(B.FLOWERS, np.uint8)
        plant[flower] = kinds[np.minimum((kind[flower] * len(kinds)).astype(np.int64), len(kinds) - 1)]
        mush = (green | taiga) & (roll > 0.5) & (roll < 0.505)
        plant[mush] = np.where(kind[mush] < 0.5, B.RED_MUSHROOM, B.BROWN_MUSHROOM)
        plant[(biome == PLAINS) & (roll > 0.6) & (roll < 0.6007)] = B.PUMPKIN
        plant[~((ground == B.GRASS) & above_air)] = 0
        sand = (biome == DESERT) & above_air & (ground == B.SAND)
        plant[sand & (roll < 0.01)] = B.DEAD_BUSH
        xs, zs = np.nonzero(plant)
        blk[xs, zs, h[xs, zs] + 1] = plant[xs, zs]
        cactus = sand & (roll > 0.992)
        xs, zs = np.nonzero(cactus)
        tall = 1 + (kind[xs, zs] * 3).astype(np.int64)
        for k in range(3):
            m = tall > k
            blk[xs[m], zs[m], h[xs[m], zs[m]] + 1 + k] = B.CACTUS
        # sugar cane on the shore: ground level with the water, water beside it
        wet = hf < SEA
        shore = wet[M - 1:M + S - 1, M:M + S] | wet[M + 1:M + S + 1, M:M + S] | wet[M:M + S, M - 1:M + S - 1] | \
            wet[M:M + S, M + 1:M + S + 1]
        cane = shore & (h == SEA) & above_air & ((ground == B.GRASS) | (ground == B.SAND) | (ground == B.DIRT)) & \
            (hash01(xi, zi, s + 13) < 0.3) & (plant == 0)
        xs, zs = np.nonzero(cane)
        tall = 1 + (kind[xs, zs] * 3).astype(np.int64)
        for k in range(3):
            m = tall > k
            blk[xs[m], zs[m], SEA + 1 + k] = B.SUGAR_CANE
        # the sea floor: seagrass in the shallows, kelp reaching up where it is deep
        r2 = hash01(xi, zi, s + 14)
        sunk = (h < SEA - 1) & (above == B.WATER) & (blk[ax, az, np.minimum(h + 2, H - 1)] == B.WATER)
        xs, zs = np.nonzero(sunk & (r2 < 0.17))
        blk[xs, zs, h[xs, zs] + 1] = B.SEAGRASS
        xs, zs = np.nonzero(sunk & (h < SEA - 6) & (r2 > 0.95))
        room = SEA - 2 - h[xs, zs]
        tall = 2 + (kind[xs, zs] * np.minimum(room - 1, 10)).astype(np.int64)
        for k in range(12):
            m = (tall > k) & (h[xs, zs] + 1 + k < SEA - 1)
            px, pz, py = xs[m], zs[m], h[xs[m], zs[m]] + 1 + k
            ok = blk[px, pz, py] == B.WATER
            blk[px[ok], pz[ok], py[ok]] = B.KELP

    # --- hamlets -----------------------------------------------------------------------------------------------------

    def hamlets(self, x0: float, z0: float, x1: float, z1: float) -> list[dict]:
        """The hamlets whose wells stand inside a rectangle of the world: for each its ``id``, the well ``x``,
        ``y`` (ground), ``z``, the ``huts`` [(x, y, z, door side)] and the ``field`` (x, y, z)."""
        out = []
        c = HAMLET_CELL
        for i in range(int(x0 // c), int(x1 // c) + 1):
            for j in range(int(z0 // c), int(z1 // c) + 1):
                site = self._site(i, j)
                if site is not None and x0 <= site["x"] <= x1 and z0 <= site["z"] <= z1:
                    out.append(site)
        return out

    def _site(self, i: int, j: int) -> dict | None:
        key = (i, j)
        if key in self._sites:
            return self._sites[key]
        site = None
        ci, cj = np.array([i], np.int64), np.array([j], np.int64)
        if float(hash01(ci, cj, self.salt + 21)[0]) < 0.8:
            c = HAMLET_CELL
            x = i * c + 30 + int(float(hash01(ci, cj, self.salt + 22)[0]) * (c - 60))
            z = j * c + 30 + int(float(hash01(ci, cj, self.salt + 23)[0]) * (c - 60))
            n_huts = 2 + int(float(hash01(ci, cj, self.salt + 24)[0]) * 3)
            spots = [(0, 0), (-8, 0), (8, 0), (0, -8), (0, 8), _FIELD_SPOT, *_HUT_SPOTS[:n_huts]]
            X = np.array([x + dx for dx, _ in spots], np.float32)[:, None]
            Z = np.array([z + dz for _, dz in spots], np.float32)[:, None]
            col = self.columns(X, Z)
            h = col["h"][:, 0]
            flat = int(h[:5].max() - h[:5].min()) <= 4 and int(h.max() - h.min()) <= 8
            if flat and int(h.min()) > SEA and bool((col["top"][:5, 0] == B.GRASS).all()):
                huts = [(x + dx, int(h[6 + k]), z + dz, 0 if abs(dx) > abs(dz) and dx < 0 else
                         1 if abs(dx) > abs(dz) else 2 if dz < 0 else 3)
                        for k, (dx, dz) in enumerate(_HUT_SPOTS[:n_huts])]
                site = {"id": (i * 73856093) ^ (j * 19349663), "x": x, "y": int(h[0]), "z": z, "huts": huts,
                        "field": (x + _FIELD_SPOT[0], int(h[5]), z + _FIELD_SPOT[1])}
        if len(self._sites) > 400:
            self._sites.clear()
        self._sites[key] = site
        return site

    def _build_hamlet(self, blk: np.ndarray, x0: int, z0: int, site: dict) -> None:
        """Stamp the part of a hamlet that lies in this region."""
        S, H = RSIZE, HEIGHT
        cells: dict[tuple[int, int, int], int] = {}

        def box(ax: int, ay: int, az: int, bx: int, by: int, bz: int, block: int) -> None:
            for x in range(ax, bx + 1):
                for z in range(az, bz + 1):
                    for y in range(ay, by + 1):
                        cells[x, y, z] = block

        x, y, z = site["x"], site["y"], site["z"]
        # the well: a ring of cobblestone around three blocks of water, under a little roof on four posts
        box(x - 2, y - 3, z - 2, x + 2, y, z + 2, B.COBBLE)
        box(x - 1, y + 1, z - 1, x + 1, y + 1, z + 1, B.COBBLE)
        box(x - 2, y + 1, z - 2, x + 2, y + 5, z + 2, B.AIR)
        box(x - 1, y + 1, z - 1, x + 1, y + 1, z + 1, B.COBBLE)
        box(x, y - 2, z, x, y + 1, z, B.WATER)
        for dx, dz in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            box(x + dx, y + 2, z + dz, x + dx, y + 3, z + dz, B.OAK_LOG)
        box(x - 1, y + 4, z - 1, x + 1, y + 4, z + 1, B.OAK_PLANKS)
        for k, (hx, hy, hz, door) in enumerate(site["huts"]):
            wood = (B.OAK_PLANKS, B.SPRUCE_PLANKS, B.BIRCH_PLANKS)[(site["id"] + k) % 3]
            box(hx - 2, hy - 3, hz - 2, hx + 2, hy - 1, hz + 2, B.COBBLE)
            box(hx - 3, hy + 1, hz - 3, hx + 3, hy + 6, hz + 3, B.AIR)
            box(hx - 2, hy, hz - 2, hx + 2, hy, hz + 2, wood)
            box(hx - 2, hy + 1, hz - 2, hx + 2, hy + 3, hz + 2, wood)
            box(hx - 1, hy + 1, hz - 1, hx + 1, hy + 3, hz + 1, B.AIR)
            for dx, dz in ((-2, -2), (2, -2), (-2, 2), (2, 2)):
                box(hx + dx, hy, hz + dz, hx + dx, hy + 3, hz + dz, B.OAK_LOG)
            box(hx - 3, hy + 4, hz - 3, hx + 3, hy + 4, hz + 3, B.COBBLE)
            box(hx - 2, hy + 5, hz - 2, hx + 2, hy + 5, hz + 2, B.COBBLE)
            box(hx - 1, hy + 4, hz - 1, hx + 1, hy + 4, hz + 1, B.AIR)
            ddx, ddz = ((-2, 0), (2, 0), (0, -2), (0, 2))[door]
            box(hx + ddx, hy + 1, hz + ddz, hx + ddx, hy + 2, hz + ddz, B.AIR)  # the doorway
            cells[hx - ddx, hy + 2, hz - ddz] = B.GLASS  # a window across from it
            side = (0, 2) if ddx else (2, 0)
            cells[hx + side[0], hy + 2, hz + side[1]] = B.GLASS
            cells[hx - side[0], hy + 2, hz - side[1]] = B.GLASS
            inx, inz = -ddx // 2, -ddz // 2  # the corner away from the door
            cells[hx + inx + (1 if not ddx else 0), hy + 1, hz + inz + (1 if not ddz else 0)] = B.BED
            cells[hx + inx - (1 if not ddx else 0), hy + 1, hz + inz - (1 if not ddz else 0)] = \
                (B.CRAFTING_TABLE, B.FURNACE + door, B.BOOKSHELF, B.HAY)[k % 4]
            cells[hx, hy + 5, hz] = B.GLOWSTONE  # a lamp in the roof
        # the field: wheat on wet farmland around a ditch, fenced with logs; hay beside it
        fx, fy, fz = site["field"]
        box(fx - 4, fy - 2, fz - 3, fx + 4, fy, fz + 3, B.DIRT)
        box(fx - 4, fy + 1, fz - 3, fx + 4, fy + 5, fz + 3, B.AIR)
        box(fx - 4, fy, fz - 3, fx + 4, fy, fz + 3, B.OAK_LOG)
        box(fx - 3, fy, fz - 2, fx + 3, fy, fz + 2, B.FARMLAND)
        box(fx - 3, fy, fz, fx + 3, fy, fz, B.WATER)
        for dx in range(-3, 4):
            for dz in (-2, -1, 1, 2):
                stage = int(float(hash01(np.array([fx + dx]), np.array([fz + dz]), self.salt + 31)[0]) * 4)
                cells[fx + dx, fy + 1, fz + dz] = B.WHEAT + min(stage + 1, 3)
        box(fx + 5, fy + 1, fz - 1, fx + 5, fy + 2, fz, B.HAY)
        cells[fx + 5, fy + 2, fz - 1] = B.AIR
        if not cells:
            return
        pos = np.array(list(cells), np.int64)
        val = np.array(list(cells.values()), np.uint8)
        px, py, pz = pos[:, 0] - x0, pos[:, 1], pos[:, 2] - z0
        ok = (px >= 0) & (px < S) & (pz >= 0) & (pz < S) & (py >= 1) & (py < H)
        blk[px[ok], pz[ok], py[ok]] = val[ok]
