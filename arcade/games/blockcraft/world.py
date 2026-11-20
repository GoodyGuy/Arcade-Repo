"""The BlockCraft world: endless terrain kept as 16 x 16 x 128 chunks around the player.

Regions of terrain are generated and chunks are lit and meshed by the helpers (chunkpool.py); ``pump`` asks for
what is missing near the player, takes what is finished and drops what is far away. A block the player changes is
relit and remeshed at once, here, so it shows on the next frame. Only the changes are saved: the rest of the world
is the seed."""

from __future__ import annotations

import base64
import itertools
import logging
import math
import struct
import zlib

import numpy as np

from app.arcade.games.blockcraft import blocks as B
from app.arcade.games.blockcraft.mesher import build_mesh, light_volume, sky_columns
from app.arcade.games.blockcraft.worldgen import CHUNK, HEIGHT, REGION, make_generator

log = logging.getLogger(__name__)
LIGHT_RANGE = 16  # blocks around a change whose light may change
KEEP = 3  # chunks past the render distance that stay loaded
_ids = itertools.count(1)
_AIR, _BORDER = B.AIR, B.BORDER
_TARGET_SKIP = frozenset({B.AIR, B.LAVA, B.BORDER} | {int(v) for v in np.flatnonzero(B.FLUID)})  # looked through
_TARGET_AIR = frozenset({B.AIR, B.BORDER} | set(range(B.FLOW + 1, B.FLOW + 8)))  # ... when a bucket is held


class Chunk:
    """One 16 x 16 column of the world: blocks [x, z, y], and its light once a helper has lit it."""

    __slots__ = ("asked", "biome", "bl", "blk", "mesh_ver", "raw", "sky", "top", "ver")

    def __init__(self, blk: np.ndarray, biome: np.ndarray) -> None:
        self.blk = blk
        self.raw = memoryview(blk).cast("B")  # the same bytes, quick to read one at a time
        self.biome = biome
        self.sky: np.ndarray | None = None
        self.bl: np.ndarray | None = None
        self.top = 1  # one above the highest block
        self.ver = 1  # bumps when the chunk must be meshed again
        self.mesh_ver = 0  # the version its mesh shows
        self.asked = 0  # the version a helper is working on

    def find_top(self) -> None:
        ys = np.flatnonzero(self.blk.any(axis=(0, 1)))
        self.top = int(ys[-1]) + 1 if len(ys) else 1


def spiral(radius: int) -> list[tuple[int, int]]:
    """Chunk offsets within ``radius``, nearest first."""
    pts = [(dx, dz) for dx in range(-radius, radius + 1) for dz in range(-radius, radius + 1)]
    pts.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    return pts


class World:
    """An endless world of one seed (x, z horizontal; y up, 0..127)."""

    def __init__(self, seed: int, pool, edits: str = "", smooth: bool = True,  # noqa: ANN001 - a ChunkPool
                 dim: str = "over") -> None:
        self.seed = int(seed)
        self.dim = dim  # the realm: "over", "deep" or "isles"
        self.height = HEIGHT
        self.pool = pool
        self.wid = next(_ids)
        self.gen = make_generator(self.seed, dim)
        self.sea = self.gen.sea
        self.smooth = smooth
        self.chunks: dict[tuple[int, int], Chunk] = {}
        self.regions: set[tuple[int, int]] = set()  # loaded
        self.regions_asked: set[tuple[int, int]] = set()
        self.edits: dict[tuple[int, int], dict[int, int]] = {}
        self.meshes_out: dict[tuple[int, int], tuple[bytes, bytes] | None] = {}  # for the renderer (None: gone)
        self.version = 0  # bumps on every change
        self.centre: tuple[int, int] | None = None
        self.radius = 0
        self._order: list[tuple[int, int]] = []
        self._idle = False  # nothing left to ask for until something changes
        self.ready = True
        if edits:
            self.apply_edits(edits)

    # --- streaming -----------------------------------------------------------------------------------------------

    def pump(self, x: float, z: float, radius: int) -> None:
        """Once a frame: take finished chunks, ask for the nearest missing ones around (x, z), drop the far ones."""
        pool = self.pool
        results = pool.poll()
        if pool.lost:
            pool.lost = False
            self.regions_asked.clear()
            for c in self.chunks.values():
                c.asked = 0
            self._idle = False
        for r in results:
            if r[1] != self.wid:
                continue
            self._idle = False
            if r[0] == "gen":
                self._take_region(r[2], r[3], r[4], r[5])
            elif r[0] == "mesh":
                self._take_mesh(*r[2:])
            else:
                log.warning("BlockCraft helper failed a %s job %s: %s", r[2], r[3], r[4])
                if r[2] == "gen":
                    self.regions_asked.discard((r[3][1], r[3][2]))
        centre = (math.floor(x) >> 4, math.floor(z) >> 4)
        if centre != self.centre or radius != self.radius:
            if radius != self.radius:
                self._order = spiral(radius + 1)
            self.centre, self.radius = centre, radius
            self._unload()
            self._idle = False
        if not self._idle and pool.free > 0:
            self._request()

    def _request(self) -> None:
        pool, chunks = self.pool, self.chunks
        cx0, cz0 = self.centre
        r2 = (self.radius + 0.5) ** 2
        free = pool.free
        waiting = False
        for dx, dz in self._order:
            key = (cx0 + dx, cz0 + dz)
            c = chunks.get(key)
            if c is None:
                rk = (key[0] >> 2, key[1] >> 2)
                if rk not in self.regions_asked:
                    self.regions_asked.add(rk)
                    pool.submit(("gen", self.wid, self.seed, rk[0], rk[1], self.dim))
                    free -= 1
                waiting = True
            elif c.mesh_ver != c.ver and dx * dx + dz * dz <= r2:
                if c.asked == c.ver:
                    waiting = True
                    continue
                vol = self._volume(key)
                if vol is None:
                    waiting = True
                    continue
                c.asked = c.ver
                pool.submit(("mesh", self.wid, key[0], key[1], c.ver, vol, self.smooth))
                free -= 1
                waiting = True
            if free <= 0:
                return
        self._idle = not waiting

    def _volume(self, key: tuple[int, int]) -> np.ndarray | None:
        """The blocks of the 3 x 3 chunks around one (48, 48, as high as anything stands); None until all are in."""
        cx, cz = key
        chunks = self.chunks
        near = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = chunks.get((cx + dx, cz + dz))
                if c is None:
                    return None
                near.append(c)
        top = min(HEIGHT, max(c.top for c in near) + 2)
        vol = np.empty((48, 48, top), np.uint8)
        i = 0
        for ax in (0, 16, 32):
            for az in (0, 16, 32):
                vol[ax:ax + 16, az:az + 16] = near[i].blk[:, :, :top]
                i += 1
        return vol

    def _take_region(self, rx: int, rz: int, blk: np.ndarray, biome: np.ndarray) -> None:
        rk = (rx, rz)
        if rk not in self.regions_asked or rk in self.regions:
            return
        self.regions.add(rk)
        for i in range(REGION):
            for j in range(REGION):
                key = (rx * REGION + i, rz * REGION + j)
                c = Chunk(np.ascontiguousarray(blk[i * 16:i * 16 + 16, j * 16:j * 16 + 16]),
                          np.ascontiguousarray(biome[i * 16:i * 16 + 16, j * 16:j * 16 + 16]))
                e = self.edits.get(key)
                if e:
                    flat = c.blk.reshape(-1)
                    flat[np.fromiter(e.keys(), np.int64, len(e))] = np.fromiter(e.values(), np.uint8, len(e))
                c.find_top()
                self.chunks[key] = c

    def _take_mesh(self, cx: int, cz: int, ver: int, sky: np.ndarray, bl: np.ndarray | None, solid: bytes,
                   wet: bytes) -> None:
        c = self.chunks.get((cx, cz))
        if c is None or c.ver != ver:
            return  # changed (or dropped) since it was asked for
        top = sky.shape[2]
        full = np.full((CHUNK, CHUNK, HEIGHT), 15, np.uint8)
        full[:, :, :top] = sky
        c.sky = full
        if bl is not None:
            c.bl = np.zeros((CHUNK, CHUNK, HEIGHT), np.uint8)
            c.bl[:, :, :top] = bl
        else:
            c.bl = None
        c.mesh_ver = ver
        self.meshes_out[(cx, cz)] = (solid, wet)

    def _unload(self) -> None:
        cx0, cz0 = self.centre
        far = self.radius + KEEP
        for rk in [rk for rk in self.regions if
                   rk[0] * REGION > cx0 + far or rk[0] * REGION + REGION - 1 < cx0 - far or
                   rk[1] * REGION > cz0 + far or rk[1] * REGION + REGION - 1 < cz0 - far]:
            self.regions.discard(rk)
            self.regions_asked.discard(rk)
            for i in range(REGION):
                for j in range(REGION):
                    key = (rk[0] * REGION + i, rk[1] * REGION + j)
                    if self.chunks.pop(key, None) is not None:
                        self.meshes_out[key] = None
        # regions asked for and no longer wanted come back anyway; they are kept until they fall out of range

    def loaded_fraction(self, x: float, z: float, radius: int) -> float:
        """How much of the land within ``radius`` chunks of (x, z) is ready to be seen (0..1)."""
        cx0, cz0 = math.floor(x) >> 4, math.floor(z) >> 4
        n = done = 0
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                n += 1
                c = self.chunks.get((cx0 + dx, cz0 + dz))
                if c is not None and c.mesh_ver:
                    done += 1
        return done / n

    def remesh_all(self, smooth: bool) -> None:
        self.smooth = smooth
        for c in self.chunks.values():
            c.ver += 1
        self._idle = False

    # --- access ------------------------------------------------------------------------------------------------

    def inside(self, x: int, y: int, z: int) -> bool:
        return 0 <= y < HEIGHT and (x >> 4, z >> 4) in self.chunks

    def get(self, x: int, y: int, z: int) -> int:
        """The block at a position; BORDER where the land is not loaded yet, AIR above the top and below the
        bottom."""
        if y < 0 or y >= HEIGHT:
            return _AIR
        c = self.chunks.get((x >> 4, z >> 4))
        if c is None:
            return _BORDER
        return c.raw[((x & 15) << 11) | ((z & 15) << 7) | y]

    def light_at(self, x: int, y: int, z: int) -> tuple[int, int]:
        """(sky, block) light of a voxel (full sky light above the world and where it is not lit yet)."""
        if y < 0:
            return 0, 0
        c = self.chunks.get((x >> 4, z >> 4))
        if y >= HEIGHT or c is None or c.sky is None:
            return 15, 0
        lx, lz = x & 15, z & 15
        return int(c.sky[lx, lz, y]), (int(c.bl[lx, lz, y]) if c.bl is not None else 0)

    def top_y(self, x: int, z: int) -> int:
        """The y of the highest solid or liquid block of a column (-1 when none, or not loaded)."""
        c = self.chunks.get((x >> 4, z >> 4))
        if c is None:
            return -1
        col = c.blk[x & 15, z & 15, :c.top]
        ys = np.flatnonzero(B.SOLID[col] | B.FLUID[col] | (col == B.LAVA))
        return int(ys[-1]) if len(ys) else -1

    def biome_name(self, x: float, z: float) -> str:
        xi, zi = math.floor(x), math.floor(z)
        c = self.chunks.get((xi >> 4, zi >> 4))
        return self.gen.names[int(c.biome[xi & 15, zi & 15])] if c is not None else "?"

    # --- changes -----------------------------------------------------------------------------------------------

    def set(self, x: int, y: int, z: int, block: int) -> bool:
        """Change one block (light and meshes follow at once). False where the world is not loaded."""
        return self.set_many([(x, y, z, block)])

    def set_many(self, changes: list[tuple[int, int, int, int]]) -> bool:
        done = []
        for x, y, z, b in changes:
            if not 0 <= y < HEIGHT:
                continue
            key = (x >> 4, z >> 4)
            c = self.chunks.get(key)
            if c is None:
                continue
            i = ((x & 15) << 11) | ((z & 15) << 7) | y
            if c.raw[i] == b:
                continue
            c.raw[i] = b
            self.edits.setdefault(key, {})[i] = int(b)
            if b != _AIR and y >= c.top:
                c.top = y + 1
            done.append((x, y, z))
        if not done:
            return False
        self._relight(done)
        self.version += 1
        self._idle = False
        return True

    def _relight(self, done: list[tuple[int, int, int]]) -> None:
        """New light around changed blocks, and new meshes for the chunks they are in (and touch)."""
        xs, _ys, zs = zip(*done, strict=True)
        bx0, bx1 = min(xs) - LIGHT_RANGE, max(xs) + LIGHT_RANGE
        bz0, bz1 = min(zs) - LIGHT_RANGE, max(zs) + LIGHT_RANGE
        keys = [(cx, cz) for cx in range(bx0 >> 4, (bx1 >> 4) + 1) for cz in range(bz0 >> 4, (bz1 >> 4) + 1)]
        near = [(k, self.chunks.get(k)) for k in keys]
        top = min(HEIGHT, max((c.top for _, c in near if c is not None), default=1) + 2)
        W, D = bx1 - bx0 + 1, bz1 - bz0 + 1
        blk = np.full((W, D, top), _BORDER, np.uint8)
        sky0 = np.zeros((W, D, top), np.uint8)
        bl0 = np.zeros((W, D, top), np.uint8)
        spans = []
        for (cx, cz), c in near:
            if c is None:
                continue
            ax0, ax1 = max(cx * 16, bx0), min(cx * 16 + 15, bx1)
            az0, az1 = max(cz * 16, bz0), min(cz * 16 + 15, bz1)
            box = (slice(ax0 - bx0, ax1 - bx0 + 1), slice(az0 - bz0, az1 - bz0 + 1))
            own = (slice(ax0 - cx * 16, ax1 - cx * 16 + 1), slice(az0 - cz * 16, az1 - cz * 16 + 1), slice(0, top))
            blk[box] = c.blk[own]
            if c.sky is not None:
                sky0[box] = c.sky[own]
                if c.bl is not None:
                    bl0[box] = c.bl[own]
                spans.append((c, box, own))
            else:
                sky0[box] = sky_columns(c.blk[own])
        sky, bl = light_volume(blk, sky0, bl0)
        dirty = set()
        for c, box, own in spans:
            new = sky[box]
            if not np.array_equal(new, c.sky[own]):
                c.sky[own] = new
                dirty.add(id(c))
            new = bl[box]
            if c.bl is None:
                if new.any():
                    c.bl = np.zeros((CHUNK, CHUNK, HEIGHT), np.uint8)
                    c.bl[own] = new
                    dirty.add(id(c))
            elif not np.array_equal(new, c.bl[own]):
                c.bl[own] = new
                dirty.add(id(c))
        now = set()
        for x, _y, z in done:
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    now.add(((x + dx) >> 4, (z + dz) >> 4))
        for key, c in near:
            if c is not None and (key in now or id(c) in dirty):
                c.ver += 1
        for key in now:
            self.mesh_now(key)

    def mesh_now(self, key: tuple[int, int]) -> bool:
        """Mesh a chunk here and now from the light the world has; False when a helper must light it first."""
        cx, cz = key
        chunks = self.chunks
        c = chunks.get(key)
        if c is None or c.sky is None:
            return False
        near = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                n = chunks.get((cx + dx, cz + dz))
                if n is None or n.sky is None:
                    return False
                near.append(n)
        top = min(HEIGHT, max(n.top for n in near) + 2)
        blk = np.empty((18, 18, top), np.uint8)
        sky = np.empty((18, 18, top), np.uint8)
        bl = np.zeros((18, 18, top), np.uint8)
        spans = ((slice(0, 1), slice(15, 16)), (slice(1, 17), slice(0, 16)), (slice(17, 18), slice(0, 1)))
        i = 0
        for bx, ox in spans:
            for bz, oz in spans:
                n = near[i]
                i += 1
                blk[bx, bz] = n.blk[ox, oz, :top]
                sky[bx, bz] = n.sky[ox, oz, :top]
                if n.bl is not None:
                    bl[bx, bz] = n.bl[ox, oz, :top]
        self.meshes_out[key] = build_mesh(blk, sky, bl, 1, 1, self.smooth)
        c.mesh_ver = c.ver
        return True

    # --- edits (saved as a diff against the regenerated terrain) ------------------------------------------------

    def encode_edits(self) -> str:
        parts = []
        for (cx, cz), e in self.edits.items():
            if not e:
                continue
            idx = np.fromiter(e.keys(), np.uint16, len(e))
            val = np.fromiter(e.values(), np.uint8, len(e))
            parts.append(struct.pack("<iiI", cx, cz, len(e)) + idx.tobytes() + val.tobytes())
        if not parts:
            return ""
        return base64.b64encode(zlib.compress(b"BC2" + b"".join(parts), 6)).decode("ascii")

    def apply_edits(self, text: str) -> int:
        """Take an ``encode_edits`` string (before any chunk is loaded); the number of changed blocks."""
        try:
            raw = zlib.decompress(base64.b64decode(text.encode("ascii")))
        except (ValueError, TypeError, zlib.error, base64.binascii.Error):
            return 0
        if raw[:3] != b"BC2":
            return 0
        pos, total, edits = 3, 0, {}
        try:
            while pos < len(raw):
                cx, cz, n = struct.unpack_from("<iiI", raw, pos)
                pos += 12
                if n > CHUNK * CHUNK * HEIGHT or pos + 3 * n > len(raw):
                    return 0
                idx = np.frombuffer(raw, np.uint16, n, pos)
                val = np.frombuffer(raw, np.uint8, n, pos + 2 * n)
                pos += 3 * n
                ok = (idx < CHUNK * CHUNK * HEIGHT) & (val != _BORDER)
                edits[(cx, cz)] = dict(zip(idx[ok].tolist(), val[ok].tolist(), strict=True))
                total += int(ok.sum())
        except struct.error:
            return 0
        self.edits = edits
        return total

    # --- queries ------------------------------------------------------------------------------------------------

    def raycast(self, origin: tuple[float, float, float], direction: tuple[float, float, float], reach: float,
                liquids: bool = False) -> tuple[tuple[int, int, int], tuple[int, int, int], float] | None:
        """The first targetable block along a ray: (block position, face normal, distance), None if none. Water
        and lava are looked through unless ``liquids`` (a bucket is held)."""
        skip = _TARGET_AIR if liquids else _TARGET_SKIP
        ox, oy, oz = origin
        dx, dy, dz = direction
        x, y, z = math.floor(ox), math.floor(oy), math.floor(oz)
        step = [1 if d > 0 else -1 for d in (dx, dy, dz)]
        tmax, tdelta = [], []
        for o, d, v in ((ox, dx, x), (oy, dy, y), (oz, dz, z)):
            if abs(d) < 1e-9:
                tmax.append(math.inf)
                tdelta.append(math.inf)
            else:
                nxt = v + (1 if d > 0 else 0)
                tmax.append((nxt - o) / d)
                tdelta.append(abs(1 / d))
        normal = (0, 0, 0)
        t = 0.0
        get = self.get
        while t <= reach:
            if get(x, y, z) not in skip:
                return (x, y, z), normal, t
            axis = 0 if tmax[0] < tmax[1] and tmax[0] < tmax[2] else (1 if tmax[1] < tmax[2] else 2)
            t = tmax[axis]
            tmax[axis] += tdelta[axis]
            if axis == 0:
                x += step[0]
                normal = (-step[0], 0, 0)
            elif axis == 1:
                y += step[1]
                normal = (0, -step[1], 0)
            else:
                z += step[2]
                normal = (0, 0, -step[2])
        return None

    def solid_boxes(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> list:
        """The block boxes (x, y, z, x+1, y+1, z+1) that collide in a region (land not loaded yet is a wall)."""
        out = []
        get, solid = self.get, B.SOLID
        for x in range(math.floor(x0), math.floor(x1) + 1):
            for z in range(math.floor(z0), math.floor(z1) + 1):
                for y in range(math.floor(y0), math.floor(y1) + 1):
                    if solid[get(x, y, z)]:
                        out.append((x, y, z, x + 1, y + 1, z + 1))
        return out

    def blocks_in(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> set[int]:
        found = set()
        get = self.get
        for x in range(math.floor(x0), math.floor(x1) + 1):
            for z in range(math.floor(z0), math.floor(z1) + 1):
                for y in range(math.floor(y0), math.floor(y1) + 1):
                    found.add(get(x, y, z))
        return found
