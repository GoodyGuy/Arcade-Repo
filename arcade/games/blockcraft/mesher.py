"""Light and geometry of BlockCraft chunks (numpy only: this also runs in the helper processes).

``light_volume`` floods sky light and block light (0..15) through a box of blocks; ``build_mesh`` turns one chunk of
it into the renderer's *face instances*: 16 bytes per visible block face (position, face shape, texture, and the
smooth light and ambient occlusion of its four corners). Faces between two solid blocks are never made."""

from __future__ import annotations

import numpy as np

from app.arcade.games.blockcraft import blocks as B

FACE_BYTES = 16
# --- face shapes: (origin, U edge, V edge, (u0, v0, du, dv) of the texture, shade) -----------------------------------
# faces 0..5 are a cube's (+x, -x, top, bottom, +z, -z), with U x V pointing out of the block
_t, _h, _l = 1 / 16, 10 / 16, 14 / 16
SHAPES = [
    ((1, 0, 1), (0, 0, -1), (0, 1, 0), (0, 0, 1, 1), 0.6),
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 0, 1, 1), 0.6),
    ((0, 1, 1), (1, 0, 0), (0, 0, -1), (0, 0, 1, 1), 1.0),
    ((0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, 1, 1), 0.5),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1, 1), 0.8),
    ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1, 1), 0.8),
    # 6..9: the two crossed planes of a plant, each seen from both sides
    ((0.15, 0, 0.15), (0.7, 0, 0.7), (0, 1, 0), (0, 0, 1, 1), 0.95),
    ((0.85, 0, 0.85), (-0.7, 0, -0.7), (0, 1, 0), (1, 0, -1, 1), 0.95),
    ((0.15, 0, 0.85), (0.7, 0, -0.7), (0, 1, 0), (0, 0, 1, 1), 0.95),
    ((0.85, 0, 0.15), (-0.7, 0, 0.7), (0, 1, 0), (1, 0, -1, 1), 0.95),
    # 10..14: a torch (a 2 x 10 x 2 pixel post): +x, -x, +z, -z, top
    ((9 * _t, 0, 9 * _t), (0, 0, -2 * _t), (0, _h, 0), (7 * _t, 0, 2 * _t, _h), 1.0),
    ((7 * _t, 0, 7 * _t), (0, 0, 2 * _t), (0, _h, 0), (7 * _t, 0, 2 * _t, _h), 1.0),
    ((7 * _t, 0, 9 * _t), (2 * _t, 0, 0), (0, _h, 0), (7 * _t, 0, 2 * _t, _h), 1.0),
    ((9 * _t, 0, 7 * _t), (-2 * _t, 0, 0), (0, _h, 0), (7 * _t, 0, 2 * _t, _h), 1.0),
    ((7 * _t, _h, 9 * _t), (2 * _t, 0, 0), (0, 0, -2 * _t), (7 * _t, 8 * _t, 2 * _t, 2 * _t), 1.0),
    # 15: the surface of water (a little below the block's top); 16..19: its sides, as low
    ((0, _l, 1), (1, 0, 0), (0, 0, -1), (0, 0, 1, 1), 1.0),
    ((1, 0, 1), (0, 0, -1), (0, _l, 0), (0, 0, 1, _l), 0.6),
    ((0, 0, 0), (0, 0, 1), (0, _l, 0), (0, 0, 1, _l), 0.6),
    ((0, 0, 1), (1, 0, 0), (0, _l, 0), (0, 0, 1, _l), 0.8),
    ((1, 0, 0), (-1, 0, 0), (0, _l, 0), (0, 0, 1, _l), 0.8),
]
# 20..49: running water, 1/8 to 6/8 of a block deep: its surface and four sides for each depth
for _lv in range(1, 7):
    _d = _lv / 8
    SHAPES += [((0, _d, 1), (1, 0, 0), (0, 0, -1), (0, 0, 1, 1), 1.0),
               ((1, 0, 1), (0, 0, -1), (0, _d, 0), (0, 0, 1, _d), 0.6),
               ((0, 0, 0), (0, 0, 1), (0, _d, 0), (0, 0, 1, _d), 0.6),
               ((0, 0, 1), (1, 0, 0), (0, _d, 0), (0, 0, 1, _d), 0.8),
               ((1, 0, 0), (-1, 0, 0), (0, _d, 0), (0, 0, 1, _d), 0.8)]
# 50, 51: a leaf lying flat on water, seen from above and from below
SHAPES += [((0, -0.11, 1), (1, 0, 0), (0, 0, -1), (0, 0, 1, 1), 1.0),
           ((0, -0.11, 0), (1, 0, 0), (0, 0, 1), (0, 0, 1, 1), 0.6)]
F_FLIP, F_ANIM, F_WAVE, F_GLOW = 1, 2, 4, 8  # flag bits of a face

DIRS = ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0))  # (dx, dz, dy) the faces 0..5 look
_KU, _KV, _KD = np.array([1, 7, 1, 7]), np.array([3, 3, 5, 5]), np.array([0, 6, 2, 8])  # patch cells beside a corner
# the shape of a water face by the water's depth in eighths (8: under more water, a whole block) and the face 0..5
_WSHAPE = np.zeros((9, 6), np.uint8)
_WSHAPE[8] = (0, 1, 2, 3, 4, 5)
_WSHAPE[7] = (16, 17, 15, 3, 18, 19)
for _lv in range(1, 7):
    _s = 20 + (_lv - 1) * 5
    _WSHAPE[_lv] = (_s + 1, _s + 2, _s, 3, _s + 3, _s + 4)
_DEPTH = np.minimum(B.LEVEL, 7).astype(np.int8)  # eighths of a block the water stands (with air above it)
_DEPTH[[B.SEAGRASS, B.KELP]] = 7

OPAQUE = B.OPAQUE.copy()
OPAQUE[B.BORDER] = True
LIGHT_OPAQUE = B.OPAQUE.copy()
LIGHT_OPAQUE[B.LAVA] = True
LIGHT_OPAQUE[B.BORDER] = True
CUBE = (B.KIND == B.K_OPAQUE) | (B.KIND == B.K_CUTOUT) | (B.KIND == B.K_WATER)
CUBE[B.BORDER] = False
SELF_CULL = np.zeros(256, bool)  # no faces between two of the same block
SELF_CULL[[B.GLASS, B.ICE, B.PACKED_ICE, *B.DENSE_LEAVES]] = True
_OCC16 = (OPAQUE.astype(np.uint16) << 8)
_OCC16[B.BORDER] = 0
_FLAGS = np.zeros(256, np.uint8)
_FLAGS[B.FLUID] = F_ANIM
_FLAGS[B.LAVA] = F_ANIM
_FLAGS[list(B.GLOWING)] = F_GLOW
_FLAGS[[B.LAVA, B.GLOWSTONE]] |= F_GLOW
_COST = (B.LIGHT_ATT.astype(np.int16) + 1)
_PLANT_SHAPES = np.array([6, 7, 8, 9], np.uint8)
_TORCH_SHAPES = np.array([10, 11, 12, 13, 14], np.uint8)
_FLAT_SHAPES = np.array([50, 51], np.uint8)
_PLANT_FLAGS = np.full(256, F_WAVE, np.uint8)  # plants sway, but for the stiff ones; some glow
_PLANT_FLAGS[list(B.STIFF)] = 0
_PLANT_FLAGS[list(B.GLOW_PLANTS)] |= F_GLOW
_PLANT_FLAGS[B.TORCH] = F_GLOW
_PLANT_GROUP = np.zeros(256, np.uint8)  # 0 crossed planes, 1 a torch, 2 flat
_PLANT_GROUP[B.TORCH] = 1
_PLANT_GROUP[list(B.FLAT)] = 2


# --- light -----------------------------------------------------------------------------------------------------------

def sky_columns(blk: np.ndarray) -> np.ndarray:
    """Sunlight straight down each column (last axis is y, up): 15 until the first opaque block, less through
    leaves and water."""
    op = LIGHT_OPAQUE[blk][..., ::-1]
    blocked = np.logical_or.accumulate(op, axis=-1)[..., ::-1]
    att = np.cumsum(B.LIGHT_ATT[blk][..., ::-1], axis=-1, dtype=np.int16)[..., ::-1]
    return np.where(blocked, 0, np.clip(15 - att, 0, 15)).astype(np.uint8)


def _flood(light: np.ndarray, trans: np.ndarray, cost: np.ndarray) -> None:
    """Spread light through a padded volume in place: a voxel gets its brightest neighbour's light minus its own
    cost. Level by level from the brightest, so every voxel is written once with its final value."""
    L16 = light.astype(np.int16)
    want = np.where(trans, L16 + cost, 127)  # a neighbour brighter than this has light to give
    need = np.zeros(light.shape, bool)
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(1, None), slice(None, -1)
        lo, hi = tuple(lo), tuple(hi)
        need[lo] |= L16[lo] > want[hi]
        need[hi] |= L16[hi] > want[lo]
    front = np.flatnonzero(need)
    if not front.size:
        return
    sx, sz = light.shape[1] * light.shape[2], light.shape[2]
    offs = (sx, -sx, sz, -sz, 1, -1)
    flat, tflat, cflat = light.reshape(-1), trans.reshape(-1), cost.reshape(-1)
    lv = flat[front]
    buckets: list[list[np.ndarray]] = [[] for _ in range(16)]
    for level in range(2, 16):
        sel = front[lv == level]
        if sel.size:
            buckets[level].append(sel)
    for level in range(15, 1, -1):
        parts = buckets[level]
        if not parts:
            continue
        src = parts[0] if len(parts) == 1 else np.concatenate(parts)
        src = src[flat[src] == level]
        if not src.size:
            continue
        for off in offs:
            nb = src + off
            new = level - cflat[nb]
            ok = tflat[nb] & (flat[nb] < new)
            if not ok.any():
                continue
            nb, new = nb[ok], new[ok]
            flat[nb] = new
            lo_, hi_ = int(new.min()), int(new.max())
            if lo_ == hi_:
                if lo_ > 1:
                    buckets[lo_].append(nb)
            else:
                for v in range(max(lo_, 2), hi_ + 1):
                    sel = nb[new == v]
                    if sel.size:
                        buckets[v].append(sel)


def light_volume(blk: np.ndarray, sky0: np.ndarray | None = None, bl0: np.ndarray | None = None,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(sky light, block light) of a box of blocks indexed [x, z, y] (the top of the box is open sky).

    With ``sky0`` / ``bl0`` (the light the box had), the outer ring of columns keeps that light and feeds it in:
    for relighting the middle of a box after a block changed there."""
    X, Z, Y = blk.shape
    shape = (X + 2, Z + 2, Y + 2)
    inner = (slice(1, X + 1), slice(1, Z + 1), slice(1, Y + 1))
    trans = np.zeros(shape, bool)
    trans[inner] = ~LIGHT_OPAQUE[blk]
    cost = np.ones(shape, np.int16)
    cost[inner] = _COST[blk]
    sky = np.zeros(shape, np.uint8)
    sky[inner] = sky_columns(blk)
    emit = B.EMIT[blk]
    has_block_light = bool(emit.any()) or (bl0 is not None and bool(bl0.any()))
    bl = np.zeros(shape, np.uint8)
    if has_block_light:
        bl[inner] = emit
    if sky0 is not None:
        ring = np.zeros((X, Z), bool)
        ring[[0, -1], :] = True
        ring[:, [0, -1]] = True
        sky[inner][ring] = sky0[ring]
        if has_block_light and bl0 is not None:
            bl[inner][ring] = bl0[ring]
        trans[inner][ring] = False  # the ring gives light and takes none
    _flood(sky, trans, cost)
    if has_block_light:
        _flood(bl, trans, cost)
    return sky[inner], bl[inner]


# --- geometry --------------------------------------------------------------------------------------------------------

def build_mesh(blk: np.ndarray, sky: np.ndarray, bl: np.ndarray, x0: int, z0: int, smooth: bool = True,
               ) -> tuple[bytes, bytes]:
    """The face instances of the chunk at [x0:x0+16, z0:z0+16] of a box of blocks (which reaches at least one
    block further on every side): (everything solid, the water)."""
    Y = blk.shape[2]
    SY = Y + 2
    P = np.zeros((18, 18, SY), np.uint8)
    P[:, :, 1:Y + 1] = blk[x0 - 1:x0 + 17, z0 - 1:z0 + 17]
    P[:, :, 0] = B.BEDROCK
    packed = np.zeros((18, 18, SY), np.uint16)
    packed[:, :, 1:Y + 1] = sky[x0 - 1:x0 + 17, z0 - 1:z0 + 17] | (bl[x0 - 1:x0 + 17, z0 - 1:z0 + 17].astype(np.uint16)
                                                                 << 4)
    packed[:, :, Y + 1] = 15
    packed |= _OCC16[P]
    C = P[1:17, 1:17, 1:Y + 1]
    fluid = B.FLUID[C]
    cube = CUBE[C] & ~fluid
    self_cull = SELF_CULL[C]
    wet = B.WATERY[P]
    depth = _DEPTH[P]  # how high water stands in every voxel: 8 under more water
    depth[:, :, :-1][wet[:, :, 1:] & (depth[:, :, :-1] > 0)] = 8
    dC = depth[1:17, 1:17, 1:Y + 1]
    vis = np.empty((6, 16, 16, Y), bool)
    for d, (dx, dz, dy) in enumerate(DIRS):
        box = (slice(1 + dx, 17 + dx), slice(1 + dz, 17 + dz), slice(1 + dy, Y + 1 + dy))
        N = P[box]
        open_ = ~OPAQUE[N]
        # water shows a side where the water beside it stands lower (or there is none), its top and bottom where
        # no water follows
        seen = (depth[box] < dC) if dy == 0 else ~wet[box]
        vis[d] = (cube & open_ & ~(self_cull & (N == C))) | (fluid & open_ & seen)
    face, xs, zs, ys = np.nonzero(vis)
    SX = 18 * SY
    pflat = packed.reshape(-1)
    n = len(face)
    out = np.zeros((n, FACE_BYTES), np.uint8)
    b = C[xs, zs, ys]
    if n:
        step = np.array([SX, -SX, 1, -1, SY, -SY], np.int64)  # to the voxel each face looks into
        u_step = np.array([-SY, SY, SX, SX, SX, -SX], np.int64)
        v_step = np.array([1, 1, -SY, SY, 1, 1], np.int64)
        ci = (xs + 1) * SX + (zs + 1) * SY + (ys + 1)
        ni = ci + step[face]
        du = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], np.int64)
        dv = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], np.int64)
        patch = u_step[face][:, None] * du[None, :] + v_step[face][:, None] * dv[None, :]
        pp = pflat[ni[:, None] + patch]  # (n, 9) light words around every face
        s9 = (pp & 15).astype(np.uint16)
        b9 = ((pp >> 4) & 15).astype(np.uint16)
        if smooth:
            occ = (pp >> 8).astype(bool)
            ou, ov, od = occ[:, _KU], occ[:, _KV], occ[:, _KD]
            ao = np.where(ou & ov, 0, 3 - (ou.astype(np.uint8) + ov + od)).astype(np.uint8)
            blocked = od | (ou & ov)

            def corners(L: np.ndarray) -> np.ndarray:
                c = L[:, 4:5]
                total = c + np.where(ou, c, L[:, _KU]) + np.where(ov, c, L[:, _KV]) + np.where(blocked, c, L[:, _KD])
                return (total * 17 + 2) // 4

            cs, cb = corners(s9), corners(b9)
            m = ao.astype(np.uint16) * 24 + np.maximum(cs, cb)
            flip = (m[:, 0] + m[:, 3]) > (m[:, 1] + m[:, 2])
        else:
            cs = np.repeat(s9[:, 4:5] * 17, 4, 1)
            cb = np.repeat(b9[:, 4:5] * 17, 4, 1)
            ao = np.full((n, 4), 3, np.uint8)
            flip = np.zeros(n, bool)
        shape = face.astype(np.uint8)
        water = B.FLUID[b]
        if water.any():
            shape[water] = _WSHAPE[dC[xs[water], zs[water], ys[water]], face[water]]
        out[:, 0], out[:, 1], out[:, 2], out[:, 3] = xs, ys, zs, shape
        out[:, 4] = B.FACE_TEX[b, face]
        out[:, 5] = _FLAGS[b] | flip
        out[:, 6] = ao[:, 0] | (ao[:, 1] << 2) | (ao[:, 2] << 4) | (ao[:, 3] << 6)
        out[:, 8:12] = cs
        out[:, 12:16] = cb
        solid, wet = out[~water], out[water]
    else:
        solid, wet = out, out
    # plants and torches
    kind = B.KIND[C]
    px, pz, py = np.nonzero(kind == B.K_CROSS)
    if len(px):
        pb = C[px, pz, py]
        w = pflat[(px + 1) * SX + (pz + 1) * SY + (py + 1)]
        ls = ((w & 15) * 17).astype(np.uint8)
        lb = (((w >> 4) & 15) * 17).astype(np.uint8)
        parts = [solid]
        group = _PLANT_GROUP[pb]
        for g, shapes in enumerate((_PLANT_SHAPES, _TORCH_SHAPES, _FLAT_SHAPES)):
            sel = group == g
            k = int(sel.sum())
            if not k:
                continue
            m = len(shapes)
            rec = np.zeros((k, m, FACE_BYTES), np.uint8)
            rec[:, :, 0], rec[:, :, 1], rec[:, :, 2] = px[sel][:, None], py[sel][:, None], pz[sel][:, None]
            rec[:, :, 3] = shapes[None, :]
            rec[:, :, 4] = B.FACE_TEX[pb[sel], 4][:, None]
            rec[:, :, 5] = _PLANT_FLAGS[pb[sel]][:, None]
            rec[:, :, 6] = 255
            rec[:, :, 8:12] = ls[sel][:, None, None]
            rec[:, :, 12:16] = lb[sel][:, None, None]
            parts.append(rec.reshape(-1, FACE_BYTES))
        solid = np.concatenate(parts)
    return solid.tobytes(), wet.tobytes()


def light_and_mesh(vol: np.ndarray, smooth: bool = True) -> tuple[np.ndarray, np.ndarray | None, bytes, bytes]:
    """A helper's whole job for one chunk: ``vol`` is the 3 x 3 chunks around it (48, 48, Y). Gives the chunk's sky
    light and block light (None when there is none) and its two meshes."""
    sky, bl = light_volume(vol)
    solid, wet = build_mesh(vol, sky, bl, 16, 16, smooth)
    csky = np.ascontiguousarray(sky[16:32, 16:32])
    cbl = np.ascontiguousarray(bl[16:32, 16:32])
    return csky, (cbl if cbl.any() else None), solid, wet
