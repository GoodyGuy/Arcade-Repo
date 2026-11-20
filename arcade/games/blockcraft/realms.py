"""BlockCraft's two other realms, generated like the overworld (a region of 4 x 4 chunks at a time, every block a
pure function of the seed):

* **the Deep**: a world of fire under the world. A sea of lava between a floor and a roof of deepstone held apart by
  great pillars; ash wastes with charred trunks, ember forests of glowing caps, crystal hollows, basalt spires,
  rime hollows (a pocket of cold: pale rimestone, spikes of ice with a frostlight on top) and tar flats (black
  tarstone with amber in it); and here and there a ruined forge keep that guards starstone.
* **the Sky Isles**: islands of pale stone over nothing, moonmoss and pale trees, clouds to walk on, a storm altar on
  the middle isle, and lone spires with treasure at the top; further out, dawn groves (rosemoss under trees the
  colour of sunrise) and star barrens (bare stardust, pillars of windstone).

Both are reached through gates the player crafts (see life.py)."""

from __future__ import annotations

import numpy as np

from app.arcade.games.blockcraft import blocks as B
from app.arcade.games.blockcraft.worldgen import (HEIGHT, MARGIN, RSIZE, Noise, _round_tree, _template, _upsample,
                                                  hash01)

DEEP_SEA = 30  # the lava level
ASH_WASTES, EMBER_FOREST, CRYSTAL_HOLLOWS, BASALT_SPIRES, RIME_HOLLOWS, TAR_FLATS = range(6)
SKY_ISLES, DAWN_GROVES, STAR_BARRENS = range(3)
_DEEP_TOP = np.array([B.ASH, B.EMBER_MOSS, B.DEEPSTONE, B.BASALT, B.RIMESTONE, B.TARSTONE], np.uint8)
_DEEP_UNDER = np.array([B.ASH, 0, 0, B.BASALT, B.RIMESTONE, B.TARSTONE], np.uint8)  # under the top block (0: rock)
_ISLE_TOP = np.array([B.MOONMOSS, B.ROSEMOSS, B.STARDUST], np.uint8)
KEEP_CELL = 176  # one place a forge keep may stand in every square of this many blocks
SPIRE_CELL = 144
ISLE_TOP = 70  # the height of the middle isle


def _stamp(blk: np.ndarray, x0: int, z0: int, x: np.ndarray, y: np.ndarray, z: np.ndarray, template: tuple) -> None:
    """Put one template at each (x, y, z) (world coordinates), into air only, cut to the region."""
    S = blk.shape[0]
    offs, blocks, _ = template
    px = (x - x0)[:, None] + offs[None, :, 0]
    py = y[:, None] + offs[None, :, 1]
    pz = (z - z0)[:, None] + offs[None, :, 2]
    b = np.broadcast_to(blocks, px.shape)
    ok = (px >= 0) & (px < S) & (pz >= 0) & (pz < S) & (py >= 1) & (py < HEIGHT - 1)
    px, py, pz, b = px[ok], py[ok], pz[ok], b[ok]
    free = blk[px, pz, py] == B.AIR
    blk[px[free], pz[free], py[free]] = b[free]


def _cells(blk: np.ndarray, x0: int, z0: int, cells: dict) -> None:
    """Write a dict of {(x, y, z): block} (world coordinates) into the region it overlaps."""
    if not cells:
        return
    S = blk.shape[0]
    pos = np.array(list(cells), np.int64)
    val = np.array(list(cells.values()), np.uint8)
    px, py, pz = pos[:, 0] - x0, pos[:, 1], pos[:, 2] - z0
    ok = (px >= 0) & (px < S) & (pz >= 0) & (pz < S) & (py >= 1) & (py < HEIGHT - 1)
    blk[px[ok], pz[ok], py[ok]] = val[ok]


def _box(cells: dict, ax: int, ay: int, az: int, bx: int, by: int, bz: int, block: int) -> None:
    for x in range(ax, bx + 1):
        for z in range(az, bz + 1):
            for y in range(ay, by + 1):
                cells[x, y, z] = block


def _glowcap(h: int) -> tuple:
    cells = {(0, y, 0): B.GLOWSTEM for y in range(h)}
    for dx in range(-2, 3):
        for dz in range(-2, 3):
            if abs(dx) + abs(dz) < 4:
                cells[dx, h, dz] = B.GLOWCAP
            if max(abs(dx), abs(dz)) < 2:
                cells[dx, h + 1, dz] = B.GLOWCAP
    return _template(cells, B.GLOWSTEM)


def _trunk(h: int, block: int, arms: tuple = ()) -> tuple:
    cells = {(0, y, 0): block for y in range(h)}
    for dx, dy, dz in arms:
        cells[dx, dy, dz] = block
    return _template(cells, block)


_FEATURES = {
    EMBER_FOREST: [_glowcap(h) for h in (3, 4, 4, 5, 6, 8)],
    ASH_WASTES: [_trunk(2, B.CHARRED_LOG), _trunk(3, B.CHARRED_LOG, ((1, 2, 0),)), _trunk(4, B.CHARRED_LOG),
                 _trunk(4, B.CHARRED_LOG, ((0, 3, 1), (-1, 2, 0))), _trunk(5, B.CHARRED_LOG, ((1, 4, 0),)),
                 _trunk(3, B.CHARRED_LOG, ((0, 2, -1),))],
    CRYSTAL_HOLLOWS: [_trunk(1, B.DEEP_CRYSTAL), _trunk(2, B.DEEP_CRYSTAL), _trunk(3, B.DEEP_CRYSTAL, ((1, 0, 0),)),
                      _trunk(4, B.DEEP_CRYSTAL, ((0, 0, 1), (-1, 0, 0), (0, 1, 1))), _trunk(2, B.DEEP_CRYSTAL, ((0, 0, -1),)),
                      _trunk(5, B.DEEP_CRYSTAL, ((1, 0, 0), (1, 1, 0), (0, 0, 1)))],
    BASALT_SPIRES: [_trunk(h, B.BASALT, arms) for h, arms in ((4, ()), (6, ((1, 0, 0), (1, 1, 0))), (8, ((0, 0, 1),)),
                                                             (10, ((1, 0, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1))),
                                                             (13, ((-1, 0, 0), (-1, 1, 0), (-1, 2, 0))), (16, ((0, 0, -1),)))],
}
def _spike(h: int, lit: bool, arms: tuple = ()) -> tuple:
    """A spike of ice (a frostlight on its tip)."""
    cells = {(0, y, 0): B.PACKED_ICE for y in range(h)}
    for dx, dy, dz in arms:
        cells[dx, dy, dz] = B.PACKED_ICE
    if lit:
        cells[0, h, 0] = B.FROSTLIGHT
    return _template(cells, B.PACKED_ICE)


def _lump(cells: dict) -> tuple:
    return _template(cells, B.TARSTONE)


_FEATURES[RIME_HOLLOWS] = [_spike(2, False), _spike(3, True), _spike(4, False, ((1, 0, 0),)), _spike(5, True, ((0, 0, 1),)),
                           _spike(7, True, ((1, 0, 0), (0, 0, -1), (1, 1, 0))), _spike(1, True)]
_FEATURES[TAR_FLATS] = [_lump({(0, 0, 0): B.AMBER}), _lump({(0, 0, 0): B.TARSTONE, (0, 1, 0): B.AMBER, (1, 0, 0): B.TARSTONE}),
                        _trunk(3, B.CHARRED_LOG), _lump({(0, 0, 0): B.AMBER, (1, 0, 0): B.AMBER, (0, 0, 1): B.TARSTONE}),
                        _lump({(dx, 0, dz): B.TARSTONE for dx in (0, 1) for dz in (0, 1)} | {(0, 1, 0): B.TARSTONE,
                                                                                            (0, 2, 0): B.AMBER}),
                        _trunk(2, B.CHARRED_LOG, ((1, 1, 0),))]
_PILLARS = [_trunk(h, B.WINDSTONE, arms) for h, arms in ((3, ()), (5, ((1, 0, 0),)), (7, ((0, 0, 1), (0, 1, 1))),
                                                         (9, ((1, 0, 0), (-1, 0, 0))), (4, ((0, 0, -1),)),
                                                         (12, ((1, 0, 0), (0, 0, 1), (1, 1, 0))))]
_DENSITY = {EMBER_FOREST: 0.2, ASH_WASTES: 0.05, CRYSTAL_HOLLOWS: 0.32, BASALT_SPIRES: 0.14, RIME_HOLLOWS: 0.17,
            TAR_FLATS: 0.18}
_PALE_TREES = None
_DAWN_TREES = None


def _pale_trees() -> list:
    global _PALE_TREES  # noqa: PLW0603 - made once
    if _PALE_TREES is None:
        rng = np.random.default_rng(909)
        _PALE_TREES = [_round_tree(h, B.PALE_LOG, B.PALE_LEAVES, rng) for h in (4, 5, 5, 6, 7, 8)]
    return _PALE_TREES


def _dawn_trees() -> list:
    global _DAWN_TREES  # noqa: PLW0603 - made once
    if _DAWN_TREES is None:
        rng = np.random.default_rng(910)
        _DAWN_TREES = [_round_tree(h, B.PALE_LOG, B.DAWN_LEAVES, rng) for h in (4, 5, 6, 6, 7, 9)]
    return _DAWN_TREES


class _Realm:
    """What the two realms share: no hamlets, features on a 4 x 4 grid, sites on a wide grid."""

    dim = ""
    sea = -64
    names: tuple = ()

    def __init__(self, seed: int, mix: int) -> None:
        self.seed = int(seed)
        self.noise = Noise((self.seed ^ mix) & 0xFFFFFFFF)
        self.salt = (self.seed * 0x9E3779B1 + mix) & 0xFFFFFFFFFFFF
        self.phase = ((self.salt % 6283) / 1000.0, ((self.salt >> 13) % 6283) / 1000.0)  # where the newer lands lie
        self._sites: dict = {}

    def hamlets(self, *_args) -> list:  # noqa: ANN002 - only the overworld has them
        return []

    def _grid(self, rx: int, rz: int) -> tuple:
        S, M = RSIZE, MARGIN
        x0, z0 = rx * S, rz * S
        xs = np.arange(x0 - M, x0 + S + M, dtype=np.float32)
        zs = np.arange(z0 - M, z0 + S + M, dtype=np.float32)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        return x0, z0, X, Z, (slice(M, M + S), slice(M, M + S))

    def _candidates(self, x0: int, z0: int) -> tuple:
        """One spot in every 4 x 4 cell of the region and its margin: (x, z, grid x, grid z, roll, variant)."""
        S, M = RSIZE, MARGIN
        n = (S + 2 * M) // 4
        gx = (np.arange(n, dtype=np.int64) + (x0 - M) // 4)[:, None] + np.zeros((1, n), np.int64)
        gz = (np.arange(n, dtype=np.int64) + (z0 - M) // 4)[None, :] + np.zeros((n, 1), np.int64)
        s = self.salt
        x = gx * 4 + (hash01(gx, gz, s + 1) * 4).astype(np.int64)
        z = gz * 4 + (hash01(gx, gz, s + 2) * 4).astype(np.int64)
        return x, z, x - (x0 - M), z - (z0 - M), hash01(gx, gz, s + 3), (hash01(gx, gz, s + 5) * 6).astype(np.int64)

    def _site_cells(self, x0: int, z0: int, cell: int, reach: int) -> list:
        S = RSIZE
        return [(i, j) for i in range((x0 - reach) // cell, (x0 + S + reach) // cell + 1)
                for j in range((z0 - reach) // cell, (z0 + S + reach) // cell + 1)]

    def _site_spot(self, i: int, j: int, cell: int, chance: float) -> tuple | None:
        ci, cj = np.array([i], np.int64), np.array([j], np.int64)
        if float(hash01(ci, cj, self.salt + 21)[0]) >= chance:
            return None
        return (i * cell + 24 + int(float(hash01(ci, cj, self.salt + 22)[0]) * (cell - 48)),
                j * cell + 24 + int(float(hash01(ci, cj, self.salt + 23)[0]) * (cell - 48)))

    def _bedrock(self, blk: np.ndarray, x0: int, z0: int) -> None:
        S = RSIZE
        xi = np.arange(x0, x0 + S, dtype=np.int64)[:, None]
        zi = np.arange(z0, z0 + S, dtype=np.int64)[None, :]
        blk[:, :, 0] = blk[:, :, HEIGHT - 1] = B.BEDROCK
        for y in (1, 2):
            thin = hash01(xi, zi, self.salt + 100 + y) < (0.7 - y * 0.25)
            blk[:, :, y][thin] = B.BEDROCK
            blk[:, :, HEIGHT - 1 - y][thin] = B.BEDROCK


class DeepGenerator(_Realm):
    dim = "deep"
    sea = DEEP_SEA
    names = ("ash_wastes", "ember_forest", "crystal_hollows", "basalt_spires", "rime_hollows", "tar_flats")

    def __init__(self, seed: int) -> None:
        super().__init__(seed, 0x5F3759DF)

    def columns(self, X: np.ndarray, Z: np.ndarray) -> dict:
        n = self.noise
        floor = 33 + n.fbm2(X / 70 + 3, Z / 70 + 3, 3) * 15 + n.fbm2(X / 17 + 50, Z / 17 + 50, 2) * 2.5
        roof = 100 + n.fbm2(X / 60 + 200, Z / 60 + 200, 3) * 14 + n.fbm2(X / 15 + 90, Z / 15 + 90, 2) * 3
        b1 = n.fbm2(X / 170 + 300, Z / 170 + 300, 3)
        b2 = n.fbm2(X / 150 + 500, Z / 150 + 500, 3)
        biome = np.full(X.shape, ASH_WASTES, np.uint8)
        biome[b2 > 0.13] = BASALT_SPIRES
        biome[b1 > 0.1] = EMBER_FOREST
        biome[b1 < -0.12] = CRYSTAL_HOLLOWS
        # the two newer lands: where a slow wave (bent by the noise there already is) runs high or low
        b3 = np.sin(X / 83 + self.phase[0] + b2 * 5.0) * np.cos(Z / 97 + self.phase[1] + b1 * 5.0)
        biome[b3 > 0.62] = RIME_HOLLOWS
        biome[b3 < -0.64] = TAR_FLATS
        return {"floor": np.clip(np.round(floor), 8, 62).astype(np.int32),
                "roof": np.clip(np.round(roof), 78, 122).astype(np.int32), "biome": biome}

    def find_spawn(self) -> tuple[float, float, float]:
        return 0.5, 64.0, 0.5

    def region(self, rx: int, rz: int) -> tuple[np.ndarray, np.ndarray]:
        S, H = RSIZE, HEIGHT
        x0, z0, X, Z, inner = self._grid(rx, rz)
        col = self.columns(X, Z)
        floor, roof, biome = col["floor"][inner], col["roof"][inner], col["biome"][inner]
        n = self.noise
        cx = np.arange(x0, x0 + S + 4, 4, dtype=np.float32)
        cz = np.arange(z0, z0 + S + 4, 4, dtype=np.float32)
        cy = np.arange(0, H + 4, 4, dtype=np.float32)
        CX, CZ, CY = np.meshgrid(cx, cz, cy, indexing="ij")
        dens = _upsample(n.n3(CX / 46 + 7.1, CY / 40 + 2.3, CZ / 46 + 9.7), (4, 4, 4), (S, S, H))
        Y = np.arange(H, dtype=np.int32)[None, None, :]
        f3, r3 = floor[:, :, None], roof[:, :, None]
        solid = (Y <= f3) | (Y >= r3) | (dens > 0.37)
        blk = np.where(solid, B.DEEPSTONE, np.where(Y <= DEEP_SEA, B.LAVA, B.AIR)).astype(np.uint8)
        top = _DEEP_TOP[biome]
        dry = f3 > DEEP_SEA
        blk = np.where((Y == f3) & dry, top[:, :, None], blk)
        below = _DEEP_UNDER[biome][:, :, None]
        blk = np.where((Y < f3) & (Y >= f3 - 2) & dry & (below != 0), below, blk).astype(np.uint8)
        self._bedrock(blk, x0, z0)
        rng = np.random.default_rng([self.seed & 0xFFFFFFFF, rx & 0xFFFFFFFF, rz & 0xFFFFFFFF, 11])
        for ore, per_chunk, size, lo, hi in ((B.GOLD_ORE, 7, 8, 4, 122), (B.DIAMOND_ORE, 2.5, 6, 4, 60),
                                             (B.REDSTONE_ORE, 6, 7, 4, 122), (B.COAL_ORE, 8, 10, 4, 122),
                                             (B.IRON_ORE, 8, 8, 4, 122), (B.GLOWSTONE, 5, 12, 84, 124)):
            k = int(per_chunk * 16)
            centre = np.stack([rng.uniform(0, S, k), rng.uniform(lo, hi, k), rng.uniform(0, S, k)], 1)
            pts = np.floor((centre[:, None, :] + np.cumsum(rng.normal(0, 0.8, (k, size, 3)), 1)).reshape(-1, 3))
            pts = pts.astype(np.int64)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < S) & (pts[:, 2] >= 0) & (pts[:, 2] < S) & (pts[:, 1] >= 2) &
                      (pts[:, 1] < H - 2)]
            pts = pts[blk[pts[:, 0], pts[:, 2], pts[:, 1]] == B.DEEPSTONE]
            blk[pts[:, 0], pts[:, 2], pts[:, 1]] = ore
        x, z, ix, iz, roll, variant = self._candidates(x0, z0)
        cb, cf = col["biome"][ix, iz], col["floor"][ix, iz]
        for bm, templates in _FEATURES.items():
            for v in range(6):
                sel = (cb == bm) & (variant == v) & (roll < _DENSITY[bm]) & (cf > DEEP_SEA + 1)
                if sel.any():
                    _stamp(blk, x0, z0, x[sel], cf[sel] + 1, z[sel], templates[v])
        for i, j in self._site_cells(x0, z0, KEEP_CELL, 8):
            self._keep(blk, x0, z0, i, j)
        return blk, biome.copy()

    def _keep(self, blk: np.ndarray, x0: int, z0: int, i: int, j: int) -> None:
        """A ruined forge keep: broken walls of stone brick on a platform, lamps on obsidian corners, and on the
        dais in the middle four starstones around gold."""
        key = (i, j)
        if key not in self._sites:
            spot = self._site_spot(i, j, KEEP_CELL, 0.75)
            site = None
            if spot is not None:
                f = int(self.columns(np.array([[spot[0]]], np.float32), np.array([[spot[1]]], np.float32))["floor"][0, 0])
                if f > DEEP_SEA + 2:
                    site = (spot[0], f, spot[1])
            if len(self._sites) > 300:
                self._sites.clear()
            self._sites[key] = site
        site = self._sites[key]
        if site is None:
            return
        x, y, z = site
        cells: dict = {}
        _box(cells, x - 5, y - 3, z - 5, x + 5, y - 1, z + 5, B.DEEPSTONE)
        _box(cells, x - 5, y, z - 5, x + 5, y, z + 5, B.STONE_BRICKS)
        _box(cells, x - 5, y + 1, z - 5, x + 5, y + 8, z + 5, B.AIR)
        for d in range(-5, 6):
            for wx, wz in ((x + d, z - 5), (x + d, z + 5), (x - 5, z + d), (x + 5, z + d)):
                h = int(float(hash01(np.array([wx]), np.array([wz]), self.salt + 41)[0]) * 6)  # broken to 0..5 high
                if abs(d) > 1:  # a gap in the middle of every side is the way in
                    for k in range(h):
                        cells[wx, y + 1 + k, wz] = B.MOSSY_COBBLE if (wx + wz + k) % 5 == 0 else B.STONE_BRICKS
        for dx, dz in ((-5, -5), (5, -5), (-5, 5), (5, 5)):
            _box(cells, x + dx, y + 1, z + dz, x + dx, y + 5, z + dz, B.OBSIDIAN)
            cells[x + dx, y + 6, z + dz] = B.GLOWSTONE
        _box(cells, x - 1, y + 1, z - 1, x + 1, y + 1, z + 1, B.STONE_BRICKS)
        for dx, dz in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cells[x + dx, y + 2, z + dz] = B.STARSTONE
        cells[x, y + 2, z] = B.GOLD_BLOCK
        cells[x, y + 1, z] = B.DIAMOND_BLOCK
        cells[x + 3, y + 1, z + 3] = B.OLD_CHEST  # what the smiths left
        _cells(blk, x0, z0, cells)


class IslesGenerator(_Realm):
    dim = "isles"
    names = ("sky_isles", "dawn_groves", "star_barrens")

    def __init__(self, seed: int) -> None:
        super().__init__(seed, 0x1B873593)

    def columns(self, X: np.ndarray, Z: np.ndarray) -> dict:
        n = self.noise
        r = np.sqrt(X * X + Z * Z)
        m = n.fbm2(X / 120 + 7, Z / 120 + 7, 4) * 1.1 + n.fbm2(X / 34 + 90, Z / 34 + 90, 2) * 0.22 - 0.2 + \
            np.clip(1 - r / 90, 0, 1) * 1.2
        hills = n.fbm2(X / 55 + 300, Z / 55 + 300, 3)
        top = ISLE_TOP + hills * 8 + np.clip(m, 0, 1) * 3
        flat = np.clip((r - 26) / 14, 0, 1)  # the middle of the middle isle is level
        top = ISLE_TOP + (top - ISLE_TOP) * flat
        thick = 2 + np.sqrt(np.clip(m, 0, 1.3)) * 24 + n.fbm2(X / 12 + 400, Z / 12 + 400, 2) * 3
        # away from the middle isle: groves and barrens, where a slow wave (bent by the hills) runs high or low
        b = np.sin(X / 71 + self.phase[0] + hills * 4.0) * np.cos(Z / 89 + self.phase[1] + m * 2.0)
        biome = np.zeros(X.shape, np.uint8)
        biome[(b > 0.5) & (r > 130)] = DAWN_GROVES
        biome[(b < -0.55) & (r > 130)] = STAR_BARRENS
        return {"m": m, "top": np.round(top).astype(np.int32), "thick": np.maximum(np.round(thick), 1).astype(np.int32),
                "r": r, "biome": biome}

    def find_spawn(self) -> tuple[float, float, float]:
        col = self.columns(np.array([[40.0]], np.float32), np.array([[0.0]], np.float32))
        return 40.5, float(col["top"][0, 0]) + 1.0, 0.5

    def region(self, rx: int, rz: int) -> tuple[np.ndarray, np.ndarray]:
        S, H = RSIZE, HEIGHT
        x0, z0, X, Z, inner = self._grid(rx, rz)
        col = self.columns(X, Z)
        m, top, thick, r = (col[k][inner] for k in ("m", "top", "thick", "r"))
        Y = np.arange(H, dtype=np.int32)[None, None, :]
        land = (m > 0)[:, :, None]
        t3 = top[:, :, None]
        blk = np.where(land & (Y <= t3) & (Y >= t3 - thick[:, :, None]), B.VOIDSTONE, B.AIR).astype(np.uint8)
        biome = col["biome"][inner]
        li, lj = np.nonzero((m > 0) & (top >= 0) & (top < H))  # the land, column by column: what lies on top
        lt, lb = top[li, lj], biome[li, lj]
        blk[li, lj, lt] = _ISLE_TOP[lb]
        for k in (1, 2):  # ... and stardust goes two blocks down (where the isle is that thick)
            deep = (lb == STAR_BARRENS) & (thick[li, lj] >= k) & (lt >= k)
            if deep.any():
                blk[li[deep], lj[deep], lt[deep] - k] = B.STARDUST
        cloud = (self.noise.fbm2(X[inner] / 45 + 700, Z[inner] / 45 + 700, 2) > 0.34) & (r > 60) & (m < 0.05)
        blk[:, :, 98:100][cloud] = B.CLOUD
        x, z, ix, iz, roll, variant = self._candidates(x0, z0)
        cm, ct, cr, cbm = col["m"][ix, iz], col["top"][ix, iz], col["r"][ix, iz], col["biome"][ix, iz]
        trees, dawn = _pale_trees(), _dawn_trees()
        wild = bool(cbm.any())  # (most regions near the middle have none of the newer lands)
        for v in range(6):
            sel = (variant == v) & (cm > 0.12) & (cr > 32) & (roll < 0.09) & (cbm == SKY_ISLES)
            if sel.any():
                _stamp(blk, x0, z0, x[sel], ct[sel] + 1, z[sel], trees[v])
            sel = (variant == v) & (cm > 0.05) & (cr > 30) & (roll > 0.955) & (cbm != DAWN_GROVES)
            if sel.any():
                _stamp(blk, x0, z0, x[sel], ct[sel] + 1, z[sel], _FEATURES[CRYSTAL_HOLLOWS][v])
            if not wild:
                continue
            sel = (variant == v) & (cm > 0.1) & (roll < 0.15) & (cbm == DAWN_GROVES)
            if sel.any():
                _stamp(blk, x0, z0, x[sel], ct[sel] + 1, z[sel], dawn[v])
            sel = (variant == v) & (cm > 0.08) & (roll < 0.06) & (cbm == STAR_BARRENS)
            if sel.any():
                _stamp(blk, x0, z0, x[sel], ct[sel] + 1, z[sel], _PILLARS[v])
        cells: dict = {}
        if abs(x0) <= S and abs(z0) <= S:  # the arena: an altar between four lit pillars
            cells[0, ISLE_TOP + 1, 0] = B.STORM_ALTAR
            for dx, dz in ((-16, -16), (16, -16), (-16, 16), (16, 16)):
                _box(cells, dx, ISLE_TOP + 1, dz, dx, ISLE_TOP + 8, dz, B.VOIDSTONE)
                cells[dx, ISLE_TOP + 9, dz] = B.GLOWSTONE
        for i, j in self._site_cells(x0, z0, SPIRE_CELL, 6):
            self._spire(cells, i, j)
        _cells(blk, x0, z0, cells)
        return blk, biome.copy()

    def _spire(self, cells: dict, i: int, j: int) -> None:
        """A lone tower: a stair winding up inside to a lit room with treasure."""
        key = (i, j)
        if key not in self._sites:
            spot = self._site_spot(i, j, SPIRE_CELL, 0.8)
            site = None
            if spot is not None:
                c = self.columns(np.array([[spot[0]]], np.float32), np.array([[spot[1]]], np.float32))
                if float(c["m"][0, 0]) > 0.22 and float(c["r"][0, 0]) > 110:
                    site = (spot[0], int(c["top"][0, 0]), spot[1])
            if len(self._sites) > 300:
                self._sites.clear()
            self._sites[key] = site
        site = self._sites[key]
        if site is None:
            return
        x, y, z = site
        height = 22
        _box(cells, x - 3, y - 4, z - 3, x + 3, y, z + 3, B.VOIDSTONE)
        _box(cells, x - 3, y + 1, z - 3, x + 3, y + height + 4, z + 3, B.STONE_BRICKS)
        _box(cells, x - 2, y + 1, z - 2, x + 2, y + height + 3, z + 2, B.AIR)
        ring = [(d, -2) for d in range(-2, 3)] + [(2, d) for d in range(-1, 3)] + [(d, 2) for d in range(1, -3, -1)] + \
            [(-2, d) for d in range(1, -2, -1)]
        for k in range(height - 1):  # the stair: one step up for every step round
            dx, dz = ring[k % len(ring)]
            cells[x + dx, y + 1 + k, z + dz] = B.VOIDSTONE
        _box(cells, x - 2, y + height, z - 2, x + 2, y + height, z + 2, B.STONE_BRICKS)  # the floor of the top room
        dx, dz = ring[(height - 1) % len(ring)]
        cells[x + dx, y + height, z + dz] = B.AIR  # ... where the stair comes through
        dx, dz = ring[(height - 2) % len(ring)]
        cells[x + dx, y + height, z + dz] = B.AIR
        cells[x, y + height + 3, z] = B.GLOWSTONE
        cells[x - 1, y + height + 1, z + 1] = B.GOLD_BLOCK
        cells[x + 1, y + height + 1, z + 1] = B.DIAMOND_BLOCK
        cells[x, y + height + 1, z + 1] = B.IRON_BLOCK
        cells[x - 1, y + height + 1, z - 1] = B.OLD_CHEST
        for k in (6, 12, 18, height + 2):  # windows
            for wx, wz in ((x - 3, z), (x + 3, z), (x, z - 3), (x, z + 3)):
                cells[wx, y + k, wz] = B.GLASS
        _box(cells, x + 3, y + 1, z, x + 3, y + 2, z, B.AIR)  # the door


def realm_generator(seed: int, dim: str):  # noqa: ANN201
    return DeepGenerator(seed) if dim == "deep" else IslesGenerator(seed)
