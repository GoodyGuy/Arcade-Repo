"""Seeded gradient noise (vectorised numpy) and the BlockCraft terrain generator.

The world has no edge: every column is a pure function of the seed and its coordinates, so terrain is generated a
*region* (4 x 4 chunks) at a time, wherever the player goes, and always comes out the same. Continents and oceans,
plains, forests, deserts, taiga, snowy plains and mountains, rivers, caves, ores, lava, trees and plants, sea
grass and kelp, and here and there a hamlet: a well, a few huts and a field.

The climate (two slow noises, warmth and wetness) picks the biome; a few more slow noises pick its variants and
raise their landforms: sunflower meadows, fields of ice spires, savanna with high plateaus and crags, the painted
mesa (banded clay, with woods on top or eroded to spires), marshes, rainforest, dark old-growth forest, frozen
seas. Under the ground the caves have kinds of their own: dripstone, moss, sulfur and, deepest, gloom."""

from __future__ import annotations

import numpy as np

from app.arcade.games.blockcraft import blocks as B

CHUNK = 16  # blocks along x and z
HEIGHT = 128  # blocks of height (y 0..127)
SEA = 62  # the water level (the highest water block)
REGION = 4  # chunks along a region's side
RSIZE = CHUNK * REGION
MARGIN = 4  # columns computed around a region (trees rooted outside it reach in)

(PLAINS, FOREST, BIRCH_FOREST, DESERT, MOUNTAINS, BEACH, OCEAN, TAIGA, SNOWY, MEADOW, ICE_SPIRES, SAVANNA,
 SAVANNA_PLATEAU, SAVANNA_CRAGS, MESA, MESA_WOODLAND, MESA_SPIRES, MARSH, RAINFOREST, OLD_GROWTH,
 FROZEN_OCEAN) = range(21)
BIOME_NAMES = ("plains", "forest", "birch_forest", "desert", "mountains", "beach", "ocean", "taiga", "snowy_plains",
               "sunflower_meadow", "ice_spires", "savanna", "savanna_plateau", "savanna_crags", "painted_mesa",
               "mesa_woodland", "mesa_spires", "marsh", "rainforest", "old_growth_forest", "frozen_ocean")
CAVE_NAMES = ("", "dripstone_cave", "moss_cave", "sulfur_cave", "gloom_cave")
# the beds of the painted mesa, bottom to top (they repeat)
_BANDS = np.array([B.ORANGE_TERRACOTTA, B.ORANGE_TERRACOTTA, B.TERRACOTTA, B.TERRACOTTA, B.YELLOW_TERRACOTTA,
                   B.TERRACOTTA, B.RED_TERRACOTTA, B.RED_TERRACOTTA, B.ORANGE_TERRACOTTA, B.WHITE_TERRACOTTA,
                   B.TERRACOTTA, B.TERRACOTTA, B.BROWN_TERRACOTTA, B.ORANGE_TERRACOTTA, B.ORANGE_TERRACOTTA,
                   B.GRAY_TERRACOTTA, B.TERRACOTTA, B.RED_TERRACOTTA, B.YELLOW_TERRACOTTA, B.TERRACOTTA,
                   B.ORANGE_TERRACOTTA, B.BROWN_TERRACOTTA, B.TERRACOTTA, B.WHITE_TERRACOTTA], np.uint8)

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


def _disc(cells: dict, cx: float, y: int, cz: float, rad: float, block: int, rng: np.random.Generator,
          ragged: float = 0.0) -> None:
    for dx in range(int(np.floor(cx - rad)), int(np.ceil(cx + rad)) + 1):
        for dz in range(int(np.floor(cz - rad)), int(np.ceil(cz + rad)) + 1):
            d = np.hypot(dx - cx, dz - cz)
            if d <= rad and max(abs(dx), abs(dz)) <= 4 and not (d > rad - 1 and rng.random() < ragged):
                cells[dx, y, dz] = block


def _vines(cells: dict, rng: np.random.Generator, chance: float, longest: int) -> None:
    """Vines under the outer leaves of a crown."""
    leaves = [k for k, b in cells.items() if B.KIND[b] == B.K_CUTOUT]
    for (dx, y, dz) in leaves:
        if (dx, y - 1, dz) in cells or max(abs(dx), abs(dz)) < 2 or rng.random() > chance:
            continue
        for k in range(1, int(rng.integers(1, longest + 1)) + 1):
            if (dx, y - k, dz) in cells or y - k < 1:
                break
            cells[dx, y - k, dz] = B.VINE


def _flat_tree(height: int, log: int, leaves: int, rng: np.random.Generator) -> tuple:
    """A savanna tree: a trunk that leans near the top under a wide flat crown."""
    cells: dict[tuple[int, int, int], int] = {}
    lx, lz = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(0, 4))]
    top = height - 1
    _disc(cells, lx, top, lz, 3.2, leaves, rng, 0.35)
    _disc(cells, lx, top + 1, lz, 1.9, leaves, rng, 0.2)
    for y in range(height):
        bent = y >= height - 2
        cells[lx if bent else 0, y, lz if bent else 0] = log
    cells[0, height - 2, 0] = log
    return _template(cells, log)


def _tall_tree(height: int, log: int, leaves: int, rng: np.random.Generator) -> tuple:
    """A rainforest giant: a long bare trunk, a round crown, a tuft of leaves half way up, vines."""
    cells: dict[tuple[int, int, int], int] = {}
    top = height - 1
    for ly, rad in ((top - 2, 2.6), (top - 1, 3.4), (top, 3.2), (top + 1, 2.2), (top + 2, 1.0)):
        _disc(cells, 0, ly, 0, rad, leaves, rng, 0.3)
    mid = height // 2 + int(rng.integers(-1, 2))
    sx, sz = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(0, 4))]
    _disc(cells, sx * 1.5, mid, sz * 1.5, 1.6, leaves, rng, 0.2)
    _vines(cells, rng, 0.3, 5)
    for y in range(height):
        cells[0, y, 0] = log
    cells[sx, mid, sz] = log
    return _template(cells, log)


def _broad_tree(height: int, log: int, leaves: int, rng: np.random.Generator) -> tuple:
    """An old-growth tree: a trunk of four logs under a wide, heavy dome."""
    cells: dict[tuple[int, int, int], int] = {}
    top = height - 1
    for ly, rad in ((top - 1, 3.3), (top, 3.9), (top + 1, 3.3), (top + 2, 2.0)):
        _disc(cells, 0.5, ly, 0.5, rad, leaves, rng, 0.25)
    for y in range(height):
        for dx in (0, 1):
            for dz in (0, 1):
                cells[dx, y, dz] = log
    return _template(cells, log)


def _stilt_tree(height: int, log: int, leaves: int, rng: np.random.Generator) -> tuple:
    """A marsh tree: prop roots at its foot, a round crown hung with vines."""
    cells: dict[tuple[int, int, int], int] = {}
    top = height - 1
    for ly, rad in ((top - 1, 2.5), (top, 2.5), (top + 1, 1.5)):
        _disc(cells, 0, ly, 0, rad, leaves, rng, 0.35)
    _vines(cells, rng, 0.4, 3)
    for y in range(height):
        cells[0, y, 0] = log
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if rng.random() < 0.6:
            cells[dx, 0, dz] = log
            cells[dx, -1, dz] = log
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
        "acacia": [_flat_tree(h, B.ACACIA_LOG, B.ACACIA_LEAVES, rng) for h in (5, 5, 6, 6, 7, 8)],
        "kapok": [_tall_tree(h, B.KAPOK_LOG, B.KAPOK_LEAVES, rng) for h in (9, 11, 12, 14, 16, 19)],
        "blackwood": [_broad_tree(h, B.BLACKWOOD_LOG, B.BLACKWOOD_LEAVES, rng) for h in (6, 7, 7, 8, 9, 10)],
        "mangrove": [_stilt_tree(h, B.MANGROVE_LOG, B.MANGROVE_LEAVES, rng) for h in (5, 6, 6, 7, 8, 9)],
    }


TREES = _tree_templates()
TREE_KINDS = ("oak", "birch", "spruce", "acacia", "kapok", "blackwood", "mangrove")
_SOFT = np.zeros(256, bool)  # what a tree's leaves may grow into
_SOFT[[B.AIR, B.TALL_GRASS, B.FERN, B.DEAD_BUSH, B.RED_MUSHROOM, B.BROWN_MUSHROOM, *B.FLOWERS, B.DRY_TUFT, B.VINE,
       B.SUNFLOWER, B.SUNFLOWER_TOP, B.LILY_PAD]] = True
_CAVE_ROCK = np.zeros(256, bool)  # what the floor and roof of a cave are made of (and may be dressed)
_CAVE_ROCK[[B.STONE, B.DIRT, B.GRAVEL, B.GLOOMSTONE]] = True
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

    def _spikes(self, X: np.ndarray, Z: np.ndarray, cell: int, salt: int, chance: float, radius: tuple,
                height: tuple, tall: float) -> np.ndarray:
        """Cones scattered over the land, one at most in every square of ``cell`` blocks: how high each column
        stands in one (0 outside them). A few (``tall``) are giants."""
        gx, gz = np.floor(X / cell).astype(np.int64), np.floor(Z / cell).astype(np.int64)
        best = np.zeros(X.shape, np.float32)
        s = self.salt + salt
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                cx, cz = gx + di, gz + dj
                r = [hash01(cx, cz, s + k) for k in range(5)]
                px, pz = (cx + 0.2 + 0.6 * r[0]) * cell, (cz + 0.2 + 0.6 * r[1]) * cell
                giant = r[3] > 1 - tall
                rad = (radius[0] + (radius[1] - radius[0]) * r[2]) * np.where(giant, 1.25, 1.0)
                hh = np.where(giant, height[1] * 1.7, height[0] + (height[1] - height[0]) * r[3] * r[3])
                d = np.hypot(X - px, Z - pz)
                v = np.where(r[4] < chance, hh * np.clip(1 - d / rad, 0, 1) ** 0.7, 0)
                best = np.maximum(best, v.astype(np.float32))
        return best

    def cave_zones(self, X: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The two slow noises that pick the kind of the caves under a column."""
        n = self.noise
        return n.fbm2(X / 130 + 4000, Z / 130 + 4000, 2), n.fbm2(X / 150 + 4300, Z / 150 + 4300, 2)

    @staticmethod
    def _cave_levels(a, b, surface) -> tuple:  # noqa: ANN001 - arrays of one shape
        """For every column: the kind of its caves above the deep kinds (0 plain, 1 dripstone, 2 moss), the height
        under which caves have a kind at all, and the heights under which they are sulfur and gloom (0: nowhere)."""
        lim = np.minimum(surface - 6, 56)
        base = np.where(a > 0.19, 1, np.where(a < -0.19, 2, 0)).astype(np.uint8)
        sulfur = np.where(b > 0.27, np.minimum(46, lim), 0)
        gloom = np.where(b < -0.19, np.minimum(np.floor(24 + a * 12), lim), 0)
        return base, lim, sulfur, gloom

    @classmethod
    def _cave_kind(cls, a, b, y, surface):  # noqa: ANN001, ANN206 - numbers or arrays of one shape
        """0 plain, 1 dripstone, 2 moss, 3 sulfur, 4 gloom (see CAVE_NAMES)."""
        base, lim, sulfur, gloom = cls._cave_levels(a, b, surface)
        return np.where(y < gloom, 4, np.where(y < sulfur, 3, np.where(y < lim, base, 0)))

    def cave_biome(self, x: float, y: float, z: float) -> str:
        """The kind of cave at a place under the ground ("" for plain rock)."""
        X, Z = np.array([[x]], np.float32), np.array([[z]], np.float32)
        a, b = self.cave_zones(X, Z)
        return CAVE_NAMES[int(self._cave_kind(a, b, y, self.columns(X, Z)["h"])[0, 0])]

    def columns(self, X: np.ndarray, Z: np.ndarray) -> dict:
        """Surface height, biome and soil for the given block coordinates (float32 arrays of one shape)."""
        n = self.noise
        S_ = _smoothstep
        c = n.fbm2(X / 560 + 0.5, Z / 560 + 0.5, 4) * 1.2 + 0.17  # continents: < 0 is ocean
        land = np.clip(c * 6, 0, 1)
        hills = n.fbm2(X / 120 + 100, Z / 120 + 100, 3)
        detail = n.fbm2(X / 24 + 200, Z / 24 + 200, 2)
        mnt = _smoothstep(0.06, 0.4, n.fbm2(X / 310 + 300, Z / 310 + 300, 3)) * _smoothstep(0.04, 0.28, c)
        ridge = np.clip(1 - np.abs(n.fbm2(X / 90 + 400, Z / 90 + 400, 4)) * 2.2, 0, 1)
        h = np.where(c < 0, SEA + c * 62, SEA + 1 + c * 24)
        h = h + hills * (2 + 9 * land) + detail * 1.6 + (ridge * ridge * 50 + hills * 10 + 6) * mnt
        # the climate (of the plain land, before any landform below is raised on it)
        temp = n.fbm2(X / 440 + 500, Z / 440 + 500, 3) - (h - SEA) * 0.0025
        humid = n.fbm2(X / 350 + 600, Z / 350 + 600, 3)
        v1 = n.fbm2(X / 260 + 900, Z / 260 + 900, 2)  # variants: meadow, ice spires, mesa
        calm = (1 - S_(0.12, 0.4, mnt)) * S_(0.03, 0.09, c)  # inland and no mountain
        zero = np.zeros(h.shape, np.float32)
        # savanna: here a high plateau, there crags and cliffs
        sav = S_(0.10, 0.14, temp) * S_(0.05, 0.09, humid) * S_(0.18, 0.14, humid) * calm
        mesa = S_(0.15, 0.19, temp) * S_(0.05, 0.02, humid) * S_(0.08, 0.12, v1) * calm
        # ... plateau and crags, eroded mesa (worked out only where there is savanna or mesa)
        v2 = n.fbm2(X / 200 + 1900, Z / 200 + 1900, 2) if float(sav.max()) > 0.01 or float(mesa.max()) > 0.01 else zero
        plateau = sav * S_(0.02, 0.08, v2)
        crags = sav * S_(-0.02, -0.09, v2)
        h = h + plateau * 17
        if float(crags.max()) > 0.01:
            jag = np.clip(1 - np.abs(n.fbm2(X / 70 + 2200, Z / 70 + 2200, 3)) * 2.4, 0, 1)
            step = S_(0.0, 0.035, n.fbm2(X / 38 + 2300, Z / 38 + 2300, 2))
            h = h + crags * (jag * jag * 27 + step * 10 + 4)
        # the painted mesa: two steps of table land; where it is eroded, a low floor that spires stand on
        eroded, upper = zero, zero
        if float(mesa.max()) > 0.01:
            eroded = mesa * S_(0.10, 0.15, v2)
            t = n.fbm2(X / 95 + 2400, Z / 95 + 2400, 3)
            upper = mesa * S_(0.12, 0.15, t) * (1 - eroded)
            h = h + mesa * 5 + (mesa * 11 * S_(-0.03, 0.0, t)) * (1 - eroded) + upper * 11
        # marsh: the land sinks to the water level
        marsh = S_(-0.08, -0.03, temp) * S_(0.13, 0.08, temp) * S_(0.08, 0.13, humid) * (1 - S_(0.1, 0.3, mnt)) * \
            S_(-0.01, 0.04, c)
        if float(marsh.max()) > 0.01:
            marsh = marsh * S_(-0.02, 0.05, n.fbm2(X / 240 + 1700, Z / 240 + 1700, 2))
            level = SEA + 0.35 + np.clip(n.fbm2(X / 13 + 2900, Z / 13 + 2900, 2) * 4.5, -2.2, 1.7)
            h = h + (level - h) * marsh
        # rivers: narrow valleys along the zero lines of a noise, cut down below the sea
        river = np.abs(n.fbm2(X / 230 + 1200, Z / 230 + 1200, 3))
        cut = _smoothstep(0.05, 0.0, river) * land * (1 - _smoothstep(0.25, 0.7, mnt))
        h = h + (np.minimum(h, SEA - 3) - h) * cut
        h = np.clip(np.round(h), 6, HEIGHT - 14).astype(np.int32)
        # spires: of packed ice on the coldest, driest plains, of banded clay on the eroded mesa
        icy = S_(0.10, 0.14, v1) * S_(-0.17, -0.2, temp) * S_(-0.02, -0.05, humid) * calm * (1 - cut)
        spike = np.zeros(h.shape, np.int32)
        if float(icy.max()) > 0.01:
            spike = np.round(self._spikes(X, Z, 8, 300, 0.6, (1.4, 3.3), (8, 25), 0.06) * icy).astype(np.int32)
        if float(eroded.max()) > 0.01:
            spike = np.maximum(spike, np.round(self._spikes(X, Z, 10, 320, 0.55, (1.7, 3.6), (12, 30), 0.07) *
                                               eroded * (1 - cut)).astype(np.int32))
        spike[h < SEA + 1] = 0
        h = np.minimum(h + spike, HEIGHT - 10)
        # the biomes
        biome = np.full(h.shape, PLAINS, np.uint8)
        biome[humid > 0.07] = FOREST
        biome[(humid > 0.07) & (temp < 0.0) & (hills > 0)] = BIRCH_FOREST
        biome[(biome == PLAINS) & (v1 > 0.12) & (temp > -0.12)] = MEADOW
        biome[(humid > 0.23) & (temp > -0.12) & (temp <= 0.10)] = OLD_GROWTH
        biome[(temp > 0.10) & (humid < 0.18)] = SAVANNA
        biome[(temp > 0.10) & (humid >= 0.18)] = RAINFOREST
        dry = (temp > 0.15) & (humid < 0.05)
        biome[dry] = DESERT
        biome[(biome == SAVANNA) & (plateau > 0.6)] = SAVANNA_PLATEAU
        biome[(biome == SAVANNA) & (crags > 0.5)] = SAVANNA_CRAGS
        painted = dry & (v1 > 0.08)
        biome[painted] = MESA
        biome[painted & (upper > 0.7)] = MESA_WOODLAND
        biome[painted & (eroded > 0.5)] = MESA_SPIRES
        biome[temp < -0.17] = TAIGA
        biome[(temp < -0.17) & (humid < -0.02)] = SNOWY
        biome[(biome == SNOWY) & (v1 > 0.10)] = ICE_SPIRES
        wet = (marsh > 0.6) & (h >= SEA - 2)
        biome[wet] = MARSH
        biome[(mnt > 0.45) & (h > SEA + 20)] = MOUNTAINS
        biome[(h <= SEA + 2) & ((c < 0.08) | (cut > 0.3)) & ~wet] = BEACH
        cold = temp < -0.17
        under = h < SEA
        frozen = cold & under & (temp < -0.2)
        biome[under & ~wet] = OCEAN
        biome[frozen & (c < 0.02)] = FROZEN_OCEAN
        # the soil
        gx = np.gradient(h.astype(np.float32)) if h.ndim == 2 and min(h.shape) > 1 else (np.zeros_like(X),) * 2
        steep = (np.abs(gx[0]) + np.abs(gx[1])) > 2.2
        top = np.full(h.shape, B.GRASS, np.uint8)
        filler = np.full(h.shape, B.DIRT, np.uint8)
        sandy = (biome == DESERT) | (biome == BEACH)
        top[sandy], filler[sandy] = B.SAND, B.SAND
        top[(biome == SNOWY) | ((biome == TAIGA) & (temp < -0.27))] = B.SNOWY_GRASS
        dryland = (biome == SAVANNA) | (biome == SAVANNA_PLATEAU) | (biome == SAVANNA_CRAGS)
        top[dryland] = B.DRY_GRASS
        cliff = (biome == SAVANNA_CRAGS) & steep
        top[cliff], filler[cliff] = B.STONE, B.STONE
        floor = n.fbm2(X / 9 + 700, Z / 9 + 700, 2)
        top[biome == MARSH] = B.MARSH_GRASS
        top[(biome == MARSH) & (h <= SEA) & (floor > 0.05)] = B.MUD
        top[under] = np.where(h[under] >= SEA - 5, B.SAND,
                              np.where(floor[under] > 0.2, B.GRAVEL, np.where(floor[under] < -0.25, B.CLAY, B.DIRT)))
        filler[under] = np.where(h[under] >= SEA - 5, B.SAND, B.DIRT)
        sunk = wet & under
        top[sunk], filler[sunk] = B.MUD, B.DIRT
        rocky = (biome == MOUNTAINS) & (steep | (h > SEA + 34))
        top[rocky], filler[rocky] = B.STONE, B.STONE
        top[(biome == MOUNTAINS) & (h > SEA + 44 - (cold * 14))] = B.SNOW
        depth = (3 + (detail > 0)).astype(np.int32)
        # the mesa is clay in coloured beds down to the plain it stands on: red sand lies on its level ground
        band = (biome == MESA) | (biome == MESA_WOODLAND) | (biome == MESA_SPIRES)
        top[band] = np.where(steep[band] | (spike[band] > 0), B.TERRACOTTA, B.RED_SAND)
        top[(biome == MESA_WOODLAND) & ~steep] = B.DRY_GRASS
        filler[band] = B.TERRACOTTA
        depth[band] = np.clip(h[band] - SEA + 5, 4, 70)
        spires = biome == ICE_SPIRES
        top[spires], filler[spires] = B.SNOW, B.DIRT
        ice = spires & (spike > 0)
        top[ice], filler[ice] = B.PACKED_ICE, B.PACKED_ICE
        depth[ice] = spike[ice] + 2
        berg = np.zeros(h.shape, np.int8)
        sea_ice = (biome == FROZEN_OCEAN) & (h < SEA - 3)
        if sea_ice.any():
            lump = n.fbm2(X / 12 + 3300, Z / 12 + 3300, 2)
            berg[sea_ice] = (lump[sea_ice] > 0.24).astype(np.int8) + (lump[sea_ice] > 0.34) + (lump[sea_ice] > 0.44)
        return {"h": h, "biome": biome, "top": top, "filler": filler, "depth": depth, "sandy": sandy,
                "frozen": frozen, "band": band, "bandoff": np.round(detail * 2.5).astype(np.int32), "berg": berg}

    def find_spawn(self) -> tuple[float, float, float]:
        """Dry, green land near the middle of the world."""
        xs = np.arange(-320, 321, 8, dtype=np.float32)
        X, Z = np.meshgrid(xs, xs, indexing="ij")
        col = self.columns(X, Z)
        ok = (col["h"] > SEA + 1) & (col["top"] == B.GRASS) & (col["biome"] != MOUNTAINS) & \
            (col["biome"] != OLD_GROWTH)
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
        band = col["band"][inner]
        if band.any():  # the beds of the mesa
            beds = _BANDS[(Y + col["bandoff"][inner][:, :, None]) % len(_BANDS)]
            m = band[:, :, None] & (rel >= -d3) & ((rel < 0) | ((rel == 0) & (top == B.TERRACOTTA)[:, :, None]))
            blk[m] = beds[m]
        ice = col["frozen"][inner]
        blk[:, :, SEA][ice & (h < SEA)] = B.ICE
        berg = col["berg"][inner]
        if berg.any():  # floes and small bergs of packed ice on the frozen sea
            for y in range(SEA - 2, SEA + 3):
                blk[:, :, y][(berg > 0) & (berg >= y - SEA + 1)] = B.PACKED_ICE
        xi = np.arange(x0, x0 + S, dtype=np.int64)[:, None]
        zi = np.arange(z0, z0 + S, dtype=np.int64)[None, :]
        blk[:, :, 0] = B.BEDROCK
        for y in (1, 2, 3):
            blk[:, :, y][hash01(xi, zi, self.salt + 100 + y) < (0.75 - y * 0.2)] = B.BEDROCK
        self._caves(blk, x0, z0, h)
        self._ores(blk, rx, rz)
        self._cave_kinds(blk, x0, z0, X[inner], Z[inner], h)
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

    def _cave_kinds(self, blk: np.ndarray, x0: int, z0: int, X: np.ndarray, Z: np.ndarray, h: np.ndarray) -> None:
        """Dress the caves by their kind: dripstone with its spikes, moss with lantern vines, yellow sulfur, and
        deep down the gloom, dark stone with a faint glow; small pools stand in the hollows of the first three."""
        S, H = RSIZE, 58  # (no cave has a kind above this height)
        a, b = self.cave_zones(X, Z)
        base, lim, sulfur, gloom = self._cave_levels(a, b, h)
        if not (base.any() or sulfur.any() or gloom.any()):
            return
        blk = blk[:, :, :H]
        Y = np.arange(H, dtype=np.int16)[None, None, :]
        kind = np.where(Y < lim[:, :, None].astype(np.int16), base[:, :, None], np.uint8(0))
        if sulfur.any():
            kind[Y < sulfur[:, :, None].astype(np.int16)] = 3
        if gloom.any():
            kind[Y < gloom[:, :, None].astype(np.int16)] = 4
            blk[(kind == 4) & (blk == B.STONE)] = B.GLOOMSTONE
        air = blk == B.AIR
        s = self.salt
        ax, az, ay = np.nonzero(air & (kind > 0))  # the air of every cave that has a kind
        if not len(ax):
            return
        low = _CAVE_ROCK[blk[ax, az, ay - 1]]
        fx, fz, fy = ax[low], az[low], ay[low] - 1  # (fy: the floor block)
        high = _CAVE_ROCK[blk[ax, az, ay + 1]]
        cx, cz, cy = ax[high], az[high], ay[high]  # (cy: the air under a roof)
        fk = kind[fx, fz, fy + 1]
        # pools: hollows one block deep, where the floor around is whole
        pool2 = self.noise.fbm2(X / 7 + 4600, Z / 7 + 4600, 2) > 0.12
        wet = pool2[fx, fz] & (fk < 4) & (fy > 1) & (fx > 0) & (fx < S - 1) & (fz > 0) & (fz < S - 1)
        px, pz, py = fx[wet], fz[wet], fy[wet]
        if len(px):
            def firm(b: np.ndarray) -> np.ndarray:
                return (b != B.AIR) & (b != B.LAVA)

            held = firm(blk[px - 1, pz, py]) & firm(blk[px + 1, pz, py]) & firm(blk[px, pz - 1, py]) & \
                firm(blk[px, pz + 1, py]) & firm(blk[px, pz, py - 1])
            px, pz, py = px[held], pz[held], py[held]
            pk = kind[px, pz, py + 1]
            bed = np.array([B.STONE, B.DRIPSTONE, B.CLAY, B.SULFUR, B.STONE], np.uint8)[pk]
            under = blk[px, pz, py - 1]
            blk[px, pz, py - 1] = np.where(_CAVE_ROCK[under], bed, under)
            blk[px, pz, py] = B.WATER
        if len(fx):
            dry = blk[fx, fz, fy] != B.WATER
            fx, fz, fy, fk = fx[dry], fz[dry], fy[dry], fk[dry]
            r1 = hash01(fx + x0 + fy * 131, fz + z0 - fy * 71, s + 41)
            r2 = hash01(fx + x0 - fy * 57, fz + z0 + fy * 193, s + 42)
            ground = blk[fx, fz, fy]
            ground = np.where((fk == 1) & (r2 < 0.55), B.DRIPSTONE, ground)
            ground = np.where((fk == 2) & (r2 < 0.9), B.MOSS, ground)
            ground = np.where(fk == 3, B.SULFUR, ground)
            ground = np.where((fk == 4) & (r2 < 0.1), B.GLOOM_LICHEN, ground)
            blk[fx, fz, fy] = ground
            grow = np.zeros(len(fx), np.uint8)
            grow[(fk == 1) & (r1 < 0.16)] = B.STALAGMITE
            grow[(fk == 2) & (r1 < 0.14)] = B.TALL_GRASS
            grow[(fk == 2) & (r1 < 0.05)] = B.FERN
            grow[(fk == 3) & (r1 < 0.09)] = B.SULFUR_CRYSTAL
            grow[(fk == 4) & (r1 < 0.05)] = B.WISP_BLOOM
            g = grow > 0
            blk[fx[g], fz[g], fy[g] + 1] = grow[g]
        if len(cx):
            ck = kind[cx, cz, cy]
            r1 = hash01(cx + x0 + cy * 131, cz + z0 - cy * 71, s + 43)
            r2 = hash01(cx + x0 - cy * 57, cz + z0 + cy * 193, s + 44)
            roof = blk[cx, cz, cy + 1]
            roof = np.where((ck == 1) & (r2 < 0.55), B.DRIPSTONE, roof)
            roof = np.where((ck == 2) & (r2 < 0.5), B.MOSS, roof)
            roof = np.where(ck == 3, B.SULFUR, roof)
            roof = np.where((ck == 4) & (r2 < 0.07), B.GLOOM_LICHEN, roof)
            blk[cx, cz, cy + 1] = roof
            free = blk[cx, cz, cy] == B.AIR
            spike = (ck == 1) & (r1 < 0.2) & free
            blk[cx[spike], cz[spike], cy[spike]] = B.STALACTITE
            vine = (ck == 2) & (r1 < 0.13) & free
            vx, vz, vy, long = cx[vine], cz[vine], cy[vine], 1 + (r2[vine] * 3).astype(np.int64)
            for k in range(3):
                m = (long > k) & (vy - k > 1)
                m[m] = blk[vx[m], vz[m], vy[m] - k] == B.AIR
                if k:
                    m &= blk[vx, vz, vy - k + 1] == B.LANTERN_VINE
                blk[vx[m], vz[m], vy[m] - k] = B.LANTERN_VINE
        # sulfur also colours the walls
        if sulfur.any():
            yellow = kind[ax, az, ay] == 3
            ax, az, ay = ax[yellow], az[yellow], ay[yellow]
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                wx, wz = ax + dx, az + dz
                ok = (wx >= 0) & (wx < S) & (wz >= 0) & (wz < S)
                wx, wz, wy = wx[ok], wz[ok], ay[ok]
                stone = blk[wx, wz, wy] == B.STONE
                blk[wx[stone], wz[stone], wy[stone]] = B.SULFUR

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
                        (SNOWY, 0.03), (MEADOW, 0.008), (SAVANNA, 0.075), (SAVANNA_PLATEAU, 0.06),
                        (SAVANNA_CRAGS, 0.035), (MESA_WOODLAND, 0.34), (MARSH, 0.2), (RAINFOREST, 0.48),
                        (OLD_GROWTH, 0.6)):
            density[bm == b] = dens
        soil = (top == B.GRASS) | (top == B.SNOWY_GRASS) | (top == B.DRY_GRASS) | (top == B.MARSH_GRASS)
        marsh = bm == MARSH
        ok = (roll < density) & (soil | (marsh & (top == B.MUD))) & (h >= SEA - 2 * marsh) & (h < H - 16)
        kind = np.zeros(bm.shape, np.int64)  # an index into TREE_KINDS
        kind[((bm == BIRCH_FOREST) & (kind_roll < 0.8)) | ((bm == FOREST) & (kind_roll < 0.2))] = 1
        kind[(bm == TAIGA) | (bm == SNOWY) | ((bm == MOUNTAINS) & (kind_roll < 0.5))] = 2
        kind[((bm == SAVANNA) | (bm == SAVANNA_PLATEAU) | (bm == SAVANNA_CRAGS)) & (kind_roll < 0.9)] = 3
        kind[(bm == RAINFOREST) & (kind_roll < 0.55)] = 4
        kind[(bm == OLD_GROWTH) & (kind_roll < 0.78)] = 5
        kind[marsh] = 6
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
        dry = (biome == SAVANNA) | (biome == SAVANNA_PLATEAU) | (biome == SAVANNA_CRAGS) | (biome == MESA_WOODLAND)
        meadow, marsh, rain, old = biome == MEADOW, biome == MARSH, biome == RAINFOREST, biome == OLD_GROWTH
        p_grass = np.where(biome == PLAINS, 0.16, np.where(green, 0.09, np.where(taiga, 0.07, 0.03)))
        p_grass = np.where(dry, 0.3, np.where(marsh, 0.2, np.where(rain, 0.4, np.where(meadow, 0.2, np.where(
            old, 0.12, p_grass)))))
        tuft = roll < p_grass
        plant[tuft] = B.TALL_GRASS
        plant[tuft & dry] = B.DRY_TUFT
        plant[tuft & ((taiga & (kind < 0.7)) | (green & (kind < 0.28)) | (rain & (kind < 0.5)) |
                      (old & (kind < 0.6)) | (marsh & (kind < 0.3)))] = B.FERN
        flower = roll >= 1 - np.where(biome == PLAINS, 0.026, np.where(meadow, 0.05, np.where(dry | old, 0.002, 0.011)))
        kinds = np.array(B.FLOWERS, np.uint8)
        plant[flower] = kinds[np.minimum((kind[flower] * len(kinds)).astype(np.int64), len(kinds) - 1)]
        mush = (green | taiga | marsh | old) & (roll > 0.5) & (roll < np.where(old, 0.535, 0.505))
        plant[mush] = np.where(kind[mush] < 0.5, B.RED_MUSHROOM, B.BROWN_MUSHROOM)
        plant[(biome == PLAINS) & (roll > 0.6) & (roll < 0.6007)] = B.PUMPKIN
        plant[meadow & (roll > 0.7) & (roll < 0.77)] = B.SUNFLOWER
        plant[rain & (roll > 0.6) & (roll < 0.64)] = B.KAPOK_LEAVES  # a bush
        grassy = (ground == B.GRASS) | (ground == B.DRY_GRASS) | (ground == B.MARSH_GRASS)
        plant[~(grassy & above_air)] = 0
        sand = (biome == DESERT) & above_air & (ground == B.SAND)
        red = above_air & (ground == B.RED_SAND)
        plant[(sand & (roll < 0.01)) | (red & (roll < 0.022))] = B.DEAD_BUSH
        xs, zs = np.nonzero(plant)
        blk[xs, zs, h[xs, zs] + 1] = plant[xs, zs]
        xs, zs = np.nonzero((plant == B.SUNFLOWER) & (blk[ax, az, np.minimum(h + 2, H - 1)] == B.AIR))
        blk[xs, zs, h[xs, zs] + 2] = B.SUNFLOWER_TOP
        pads = marsh & (h < SEA) & (blk[:, :, SEA] == B.WATER) & (blk[:, :, SEA + 1] == B.AIR) & (roll < 0.09)
        blk[:, :, SEA + 1][pads] = B.LILY_PAD
        cactus = (sand & (roll > 0.992)) | (red & (roll > 0.994))
        xs, zs = np.nonzero(cactus)
        tall = 1 + (kind[xs, zs] * 3).astype(np.int64)
        for k in range(3):
            m = tall > k
            blk[xs[m], zs[m], h[xs[m], zs[m]] + 1 + k] = B.CACTUS
        # sugar cane on the shore: ground level with the water, water beside it
        wet = hf < SEA
        shore = wet[M - 1:M + S - 1, M:M + S] | wet[M + 1:M + S + 1, M:M + S] | wet[M:M + S, M - 1:M + S - 1] | \
            wet[M:M + S, M + 1:M + S + 1]
        cane = shore & (h == SEA) & above_air & ((ground == B.GRASS) | (ground == B.SAND) | (ground == B.DIRT) |
                                                 (ground == B.MARSH_GRASS) | (ground == B.MUD)) & \
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
            if flat and int(h.min()) > SEA and bool(np.isin(col["top"][:5, 0], (B.GRASS, B.DRY_GRASS)).all()):
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
