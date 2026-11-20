"""Monster sprites made from 3D "clay" models: spheres and tapered capsules in model space (x: the creature's right,
y: up, z: forward, 1 unit = 1 map unit = 1 sprite pixel), rendered orthographically from any side, Lambert-shaded
into palette ramps. Each monster has walk / attack / pain frames from 5 sides (the other 3 mirrored) and
rotation-less death and gib frames."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .palette import SPANS

# material: (ramp, brightest t, darkest t, noise, emissive)
MATERIALS = {
    "skin": ("flesh", 0.15, 0.85, 0.05, False),
    "pale": ("tan", 0.05, 0.9, 0.04, False),
    "hair": ("brown", 0.55, 0.97, 0.08, False),
    "olive": ("olive", 0.1, 0.95, 0.06, False),
    "green": ("green", 0.35, 0.97, 0.06, False),
    "dkgrey": ("grey", 0.45, 0.97, 0.05, False),
    "grey": ("grey", 0.2, 0.9, 0.05, False),
    "black": ("grey", 0.7, 1.0, 0.03, False),
    "boot": ("brown", 0.6, 0.98, 0.04, False),
    "gun": ("steel", 0.3, 1.0, 0.04, False),
    "impskin": ("brown", 0.12, 0.9, 0.09, False),
    "bone": ("tan", 0.0, 0.6, 0.05, False),
    "pink": ("flesh", 0.0, 0.75, 0.07, False),
    "pinkdark": ("red", 0.35, 0.95, 0.06, False),
    "red": ("red", 0.2, 0.97, 0.05, False),
    "blood": ("red", 0.45, 0.98, 0.1, False),
    "gore": ("red", 0.3, 0.95, 0.12, False),
    "mouth": ("red", 0.8, 1.0, 0.03, False),
    "teeth": ("grey", 0.0, 0.45, 0.04, False),
    "eye_red": ("red", 0.05, 0.05, 0.0, True),
    "eye_glow": ("fire", 0.15, 0.15, 0.0, True),
    "eye_green": ("green", 0.05, 0.05, 0.0, True),
    "eye_white": ("grey", 0.0, 0.0, 0.0, True),
    "eye_dark": ("grey", 0.95, 0.95, 0.0, True),
    "flame": ("fire", 0.2, 0.7, 0.12, True),
    "flash": ("fire", 0.0, 0.15, 0.0, True),
    "baron": ("flesh", 0.05, 0.8, 0.06, False),
    "fur": ("brown", 0.3, 0.95, 0.12, False),
    "horn": ("tan", 0.2, 0.85, 0.03, False),
    "plasma_g": ("green", 0.0, 0.3, 0.05, True),
    "caco": ("red", 0.12, 0.95, 0.05, False),
    "cacoblue": ("blue", 0.45, 0.95, 0.05, False),
}
LIGHT = np.array([-0.45, 0.65, 0.62])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


@dataclass
class Sprite:
    img: np.ndarray  # palette indices
    mask: np.ndarray
    xoff: float  # image column of the thing's centre
    yoff: float  # height of the image's bottom above the thing's feet
    bright: bool = False


def sph(c, r, mat, fn=None):  # noqa: ANN001, ANN201 - tiny constructors
    return ("s", np.asarray(c, float), float(r), mat, fn)


def cap(a, b, r1, r2, mat):  # noqa: ANN001, ANN201
    return ("c", np.asarray(a, float), np.asarray(b, float), float(r1), float(r2), mat)


def _expand(prims: list) -> list:
    out = []
    for p in prims:
        if p[0] == "s":
            out.append(p)
            continue
        _, a, b, r1, r2, mat = p
        ln = float(np.linalg.norm(b - a))
        n = max(2, int(math.ceil(ln / max(min(r1, r2) * 0.7, 0.6))) + 1)
        for i in range(n):
            t = i / (n - 1)
            out.append(("s", a + (b - a) * t, r1 + (r2 - r1) * t, mat, None))
    return out


def render(prims: list, rel_deg: float = 0.0, fall_deg: float = 0.0, pitch_deg: float = 0.0, seed: int = 1,
           bright: bool = False) -> Sprite:
    """Render the model seen from ``rel_deg`` (0: from the front, 90: from its right side); ``fall_deg`` tips it
    over backwards about its feet (deaths); ``pitch_deg`` looks down on it a little."""
    spheres = _expand(prims)
    rel = math.radians(rel_deg)
    right = np.array([-math.cos(rel), 0.0, math.sin(rel)])
    toward = np.array([math.sin(rel), 0.0, math.cos(rel)])
    up = np.array([0.0, 1.0, 0.0])
    if pitch_deg:
        pr = math.radians(pitch_deg)
        up, toward = up * math.cos(pr) - toward * math.sin(pr), toward * math.cos(pr) + up * math.sin(pr)
    fall = math.radians(fall_deg)
    cf, sf = math.cos(fall), math.sin(fall)
    pts = []
    for _, c, r, mat, fn in spheres:
        x, y, z = c
        if fall:
            y, z = c[1] * cf - c[2] * sf, c[1] * sf + c[2] * cf
        p = np.array([x, y, z])
        pts.append((p @ right, p @ up, p @ toward, r, mat, fn))
    if fall:
        low = min(vy - r for _, vy, _, r, _, _ in pts)
        pts = [(vx, vy - low, vz, r, m, f) for vx, vy, vz, r, m, f in pts]
    x0 = math.floor(min(vx - r for vx, _, _, r, _, _ in pts)) - 1
    x1 = math.ceil(max(vx + r for vx, _, _, r, _, _ in pts)) + 1
    y0 = math.floor(min(vy - r for _, vy, _, r, _, _ in pts)) - 1
    y1 = math.ceil(max(vy + r for _, vy, _, r, _, _ in pts)) + 1
    w, h = x1 - x0, y1 - y0
    n = len(pts)
    vx = np.array([p[0] for p in pts]) - x0
    vy = y1 - np.array([p[1] for p in pts])
    vz = np.array([p[2] for p in pts])
    rad = np.array([p[3] for p in pts])
    mat_names = sorted({p[4] for p in pts})
    mat_id = np.array([mat_names.index(p[4]) for p in pts])
    # every sphere's candidate pixels at once, grouped by size so the boxes stay tight
    rows_l, cols_l, depth_l, sid_l, nx_l, ny_l, nz_l = [], [], [], [], [], [], []
    size_class = np.ceil(rad).astype(int)
    for s in np.unique(np.minimum(size_class, 40)):
        sel = np.nonzero(np.minimum(size_class, 40) == s)[0]
        span = 2 * int(s) + 3
        off = np.arange(span) - (s + 1)
        c0 = np.floor(vx[sel]).astype(int)
        r0 = np.floor(vy[sel]).astype(int)
        shape3 = (len(sel), span, span)
        px = np.broadcast_to(c0[:, None, None] + off[None, None, :], shape3)
        py = np.broadcast_to(r0[:, None, None] + off[None, :, None], shape3)
        dx = px + 0.5 - vx[sel][:, None, None]
        dy = vy[sel][:, None, None] - (py + 0.5)
        r2 = (rad[sel] ** 2)[:, None, None]
        d2 = dx * dx + dy * dy
        inside = (d2 < r2) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        dz = np.sqrt(np.maximum(r2 - d2, 0.0))
        ii = np.nonzero(inside)
        rr = rad[sel][ii[0]]
        rows_l.append(py[ii])
        cols_l.append(px[ii])
        depth_l.append(vz[sel][ii[0]] + dz[ii])
        sid_l.append(sel[ii[0]])
        nx_l.append(dx[ii] / rr)
        ny_l.append(dy[ii] / rr)
        nz_l.append(dz[ii] / rr)
    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    depth = np.concatenate(depth_l)
    sid = np.concatenate(sid_l)
    nx, ny, nz = np.concatenate(nx_l), np.concatenate(ny_l), np.concatenate(nz_l)
    order = np.argsort(depth, kind="stable")
    flat = rows[order] * w + cols[order]
    win = np.full(h * w, -1, np.int64)
    win[flat] = order  # the nearest candidate is written last
    covered = win >= 0
    pick = win[covered]
    lam = np.maximum(nx[pick] * LIGHT[0] + ny[pick] * LIGHT[1] + nz[pick] * LIGHT[2], 0.0)
    inten = 0.3 + 0.7 * lam
    rng = np.random.default_rng(seed)
    noise = rng.random(len(pick)) - 0.5
    psid = sid[pick]
    names = np.array(mat_names, dtype=object)[mat_id[psid]]
    fn_ids = [i for i in range(n) if pts[i][5] is not None]
    if fn_ids:
        has_fn = np.isin(psid, fn_ids)
        if has_fn.any():
            q = pick[has_fn]
            nmx = nx[q] * right[0] + ny[q] * up[0] + nz[q] * toward[0]
            nmy = nx[q] * right[1] + ny[q] * up[1] + nz[q] * toward[1]
            nmz = nx[q] * right[2] + ny[q] * up[2] + nz[q] * toward[2]
            names = names.copy()
            names[has_fn] = pts[fn_ids[0]][5](nmx, nmy, nmz)
    vals = np.zeros(len(pick), np.uint8)
    for m in set(names.tolist()):
        selm = names == m
        vals[selm] = _shade(m, inten[selm], noise[selm])
    out = np.zeros(h * w, np.uint8)
    out[covered] = vals
    return Sprite(out.reshape(h, w), covered.reshape(h, w), float(-x0), float(y0), bright)


_MAT_CACHE: dict[str, tuple[int, int, float, float, float, bool]] = {}


def _shade(mat: str, inten: np.ndarray, noise: np.ndarray) -> np.ndarray:
    ramp_name, lo, hi, namp, emissive = MATERIALS[mat]
    start, count = SPANS[ramp_name]
    if emissive:
        t = lo + (1.0 - np.clip(inten, 0, 1)) * (hi - lo) * 0.3 + noise * namp
    else:
        t = lo + (1.0 - np.clip(inten, 0, 1)) * (hi - lo) + noise * namp
    return (np.clip(np.round(np.clip(t, 0, 1) * (count - 1)), 0, count - 1) + start).astype(np.uint8)


def add_pool(s: Sprite, width: float, colour_mat: str = "blood", seed: int = 3) -> Sprite:
    """A flat pool of blood under a corpse (widens the image if needed)."""
    h, w = s.img.shape
    pw = int(width)
    nw = max(w, pw + 2)
    pad_l = (nw - w) // 2
    img = np.zeros((h + 2, nw), np.uint8)
    mask = np.zeros((h + 2, nw), bool)
    img[:h, pad_l:pad_l + w] = s.img
    mask[:h, pad_l:pad_l + w] = s.mask
    cx = s.xoff + pad_l
    ys, xs = np.mgrid[0:h + 2, 0:nw]
    rng = np.random.default_rng(seed)
    wob = 1.0 + 0.15 * np.sin(xs * 0.7 + rng.random() * 6)
    ell = ((xs + 0.5 - cx) / (pw / 2)) ** 2 + ((ys + 0.5 - (h - 0.5)) / (2.6 * wob)) ** 2 < 1
    pool = ell & ~mask
    ramp_name, lo, hi, _, _ = MATERIALS[colour_mat]
    start, count = SPANS[ramp_name]
    t = np.clip(0.55 + (rng.random((h + 2, nw)) - 0.5) * 0.25, 0, 1)
    img[pool] = (np.round(t[pool] * (count - 1)) + start).astype(np.uint8)
    mask |= pool
    return Sprite(img, mask, cx, s.yoff - 2, s.bright)


# --- humanoid skeleton -------------------------------------------------------------------------------------------

def _rot_x(v: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([v[0], v[1] * math.cos(a) - v[2] * math.sin(a), v[1] * math.sin(a) + v[2] * math.cos(a)])


def _limb_dir(swing: float, out: float, side: float) -> np.ndarray:
    """A limb hanging down, swung forward by ``swing`` degrees and out sideways by ``out`` degrees."""
    s, o = math.radians(swing), math.radians(out)
    return np.array([side * math.sin(o), -math.cos(s) * math.cos(o), math.sin(s) * math.cos(o)])


def humanoid(b: dict, pose: dict) -> list:
    """A two-legged creature. ``b``: body (sizes, materials, extras); ``pose``: joint angles in degrees."""
    prims: list = []
    hip_h = b["leg"] * 2
    lean = pose.get("lean", 0.0)
    crouch = pose.get("crouch", 0.0)
    pelvis = np.array([0.0, hip_h - crouch, 0.0])
    # legs
    for side, key in ((-1, "l"), (1, "r")):
        hip = pelvis + np.array([side * b["hipw"], 0, 0])
        sw = pose.get("hip_" + key, 0.0)
        kn = pose.get("knee_" + key, 0.0)
        thigh = _limb_dir(sw, 4, side) * b["leg"]
        knee = hip + thigh
        shin = _limb_dir(sw - kn, 2, side) * b["leg"]
        ankle = knee + shin
        prims.append(cap(hip, knee, b["thigh_r"], b["thigh_r"] * 0.85, b["legs"]))
        prims.append(cap(knee, ankle, b["thigh_r"] * 0.85, b["thigh_r"] * 0.7, b.get("shins", b["legs"])))
        foot = ankle + np.array([0, -1.0, b["foot"]])
        prims.append(cap(ankle, foot, b["thigh_r"] * 0.7, b["thigh_r"] * 0.6, b["boots"]))
        if b.get("hooves"):
            prims.append(sph(foot + np.array([0, 0, 1.5]), b["thigh_r"] * 0.7, "black"))

    def up(p: np.ndarray) -> np.ndarray:  # the upper body leans about the pelvis
        return pelvis + _rot_x(p - pelvis, lean) if lean else p

    chest = np.array([0.0, hip_h - crouch + b["torso"], 0.0])
    prims.append(cap(up(pelvis + np.array([0, 2, 0])), up(chest), b["waist_r"], b["chest_r"], b["body"]))
    prims.append(sph(up(chest + np.array([0, -b["torso"] * 0.35, b["chest_r"] * 0.25])), b["chest_r"] * 0.95,
                     b.get("vest", b["body"])))
    neck = chest + np.array([0, b["chest_r"] * 0.55, 0])
    head = neck + np.array([0, b["head_r"] * 1.05, b.get("head_fwd", 0.0)])
    tilt = pose.get("head", 0.0)
    head = pelvis + _rot_x(head - pelvis, lean + tilt * 0.2) if (lean or tilt) else head
    for item in b["head"](head, lean + tilt, pose):
        prims.append(item)
    # arms
    for side, key in ((-1, "l"), (1, "r")):
        shoulder = up(chest + np.array([side * (b["chest_r"] * 0.95), b["chest_r"] * 0.15, 0]))
        sw = pose.get("sh_" + key, 0.0) + lean
        out = pose.get("out_" + key, 10.0)
        el = pose.get("el_" + key, 10.0)
        upper = _limb_dir(sw, out, side) * b["arm"]
        elbow = shoulder + upper
        lower = _limb_dir(sw + el, out * 0.5, side) * b["arm"]
        hand = elbow + lower
        prims.append(cap(shoulder, elbow, b["arm_r"], b["arm_r"] * 0.85, b.get("arms", b["body"])))
        prims.append(cap(elbow, hand, b["arm_r"] * 0.85, b["arm_r"] * 0.7, b.get("forearms", b.get("arms",
                                                                                               b["body"]))))
        prims.append(sph(hand, b["arm_r"] * 0.8, b.get("hands", "skin")))
        if "extras_arm" in b:
            prims.extend(b["extras_arm"](shoulder, elbow, hand, side, pose))
        if side == 1 and "weapon" in b:
            prims.extend(b["weapon"](hand, elbow, pose))
    if "extras" in b:
        prims.extend(b["extras"](up, chest, pelvis, pose))
    return prims


def _human_head(skin: str, hair: str, helmet: bool = False):  # noqa: ANN202
    def make(c: np.ndarray, tilt: float, pose: dict) -> list:
        r = 5.2
        items = [sph(c, r, skin)]
        items.append(sph(c + np.array([0, 1.6, -1.2]), r * 0.92, hair))
        fwd = _rot_x(np.array([0, 0, r * 0.82]), tilt * 0.5)
        items.append(sph(c + fwd + np.array([-1.9, 0.6, 0]), 1.05, "eye_dark"))
        items.append(sph(c + fwd + np.array([1.9, 0.6, 0]), 1.05, "eye_dark"))
        items.append(sph(c + fwd + np.array([0, -2.4, -0.4]), 1.3, "mouth" if pose.get("shout") else skin))
        if helmet:
            items.append(sph(c + np.array([0, 2.2, -0.4]), r * 0.95, "dkgrey"))
        return items
    return make


def _rifle(hand: np.ndarray, elbow: np.ndarray, pose: dict) -> list:
    d = hand - elbow
    d = d / (np.linalg.norm(d) + 1e-6)
    aim = pose.get("aim", False)
    fwd = np.array([0, 0, 1.0]) if aim else np.array([-0.5, 0.45, 0.74])
    tip = hand + fwd * 16 + np.array([-2, 0, 0])
    items = [cap(hand - fwd * 8, tip, 1.6, 1.3, "gun"), cap(hand - fwd * 2, hand - fwd * 2 + np.array([0, -4, 0]),
                                                            1.2, 1.0, "boot")]
    if pose.get("flash"):
        items.append(sph(tip + fwd * 3, 4.0, "flash"))
        items.append(sph(tip + fwd * 6, 2.5, "flash"))
    return items


def _shotgun(hand: np.ndarray, elbow: np.ndarray, pose: dict) -> list:
    aim = pose.get("aim", False)
    fwd = np.array([0, 0, 1.0]) if aim else np.array([-0.5, 0.45, 0.74])
    tip = hand + fwd * 18 + np.array([-2, 0, 0])
    items = [cap(hand - fwd * 9, tip, 1.8, 1.6, "gun"), cap(hand - fwd * 9, hand - fwd * 3, 2.2, 2.2, "boot"),
             cap(hand + fwd * 5, hand + fwd * 10, 2.0, 2.0, "boot")]
    if pose.get("flash"):
        items.append(sph(tip + fwd * 3, 5.0, "flash"))
        items.append(sph(tip + fwd * 7, 3.0, "flash"))
    return items


def _imp_spikes(up, chest, pelvis, pose) -> list:  # noqa: ANN001
    items = []
    for side in (-1, 1):
        for dy, dz in ((8, -4), (2, -6), (-6, -6)):
            base = up(chest + np.array([side * 6, dy - 4, dz]))
            items.append(cap(base, base + np.array([side * 2.5, 2.0, -4.5]), 1.6, 0.4, "bone"))
    for i in range(3):
        base = up(chest + np.array([0, -2 - i * 6, -7.5]))
        items.append(cap(base, base + np.array([0, 1.5, -4.5]), 1.6, 0.4, "bone"))
    return items


def _imp_arm_spikes(shoulder, elbow, hand, side, pose) -> list:  # noqa: ANN001
    items = []
    for t in (0.4, 0.8):
        p = elbow + (hand - elbow) * t
        items.append(cap(p, p + np.array([side * 3, 1.5, -2]), 1.2, 0.3, "bone"))
    p = shoulder + np.array([side * 1.5, 2.5, 0])
    items.append(cap(p, p + np.array([side * 2.5, 4.5, -1]), 1.8, 0.4, "bone"))
    if pose.get("fireball"):
        items.append(sph(hand + np.array([0, 2, 2]), 5.0, "flame"))
        items.append(sph(hand + np.array([0, 3, 2]), 3.2, "flash"))
    return items


def _imp_head(c, tilt, pose) -> list:  # noqa: ANN001
    r = 5.6
    fwd = _rot_x(np.array([0, 0, r * 0.8]), tilt * 0.5)
    items = [sph(c, r, "impskin"), sph(c + np.array([0, -2.2, 1.6]), r * 0.75, "impskin")]
    items.append(sph(c + fwd + np.array([-2.0, 0.8, 0]), 1.2, "eye_glow"))
    items.append(sph(c + fwd + np.array([2.0, 0.8, 0]), 1.2, "eye_glow"))
    items.append(sph(c + fwd * 1.05 + np.array([0, -3.0, 0.4]), 1.8 if pose.get("shout") else 1.1, "mouth"))
    for side in (-1, 1):
        items.append(cap(c + np.array([side * 3.5, 3.5, -1]), c + np.array([side * 4.5, 7.5, -3]), 1.3, 0.3, "bone"))
    return items


def _baron_head(c, tilt, pose) -> list:  # noqa: ANN001
    r = 6.2
    fwd = _rot_x(np.array([0, 0, r * 0.8]), tilt * 0.5)
    items = [sph(c, r, "baron"), sph(c + np.array([0, -2.4, 1.8]), r * 0.72, "baron")]
    items.append(sph(c + fwd + np.array([-2.2, 0.9, 0]), 1.2, "eye_green"))
    items.append(sph(c + fwd + np.array([2.2, 0.9, 0]), 1.2, "eye_green"))
    items.append(sph(c + fwd * 1.05 + np.array([0, -3.5, 0.6]), 1.6, "mouth"))
    for side in (-1, 1):
        a = c + np.array([side * 4, 3.5, -1])
        b = a + np.array([side * 6, 3.0, -1])
        items.append(cap(a, b, 1.8, 1.2, "horn"))
        items.append(cap(b, b + np.array([side * 1.0, 5.0, 1.0]), 1.2, 0.3, "horn"))
    return items


def _baron_hand(shoulder, elbow, hand, side, pose) -> list:  # noqa: ANN001
    if pose.get("fireball") and side == 1:
        return [sph(hand + np.array([0, 3, 2]), 6.0, "plasma_g"), sph(hand + np.array([0, 3, 3]), 3.5, "flash")]
    return []


ZOMBIE = {
    "leg": 10.5, "hipw": 3.9, "thigh_r": 3.6, "foot": 3.5, "torso": 15, "waist_r": 6.0, "chest_r": 7.8,
    "arm": 9.5, "arm_r": 3.1, "head_r": 5.2, "legs": "olive", "boots": "boot", "body": "olive", "vest": "green",
    "arms": "olive", "forearms": "skin", "hands": "skin", "head": _human_head("skin", "hair"), "weapon": _rifle,
    "walk_arms": {"sh_r": 30, "el_r": 60, "out_r": 0, "sh_l": 45, "el_l": 55, "out_l": -15},
}
SERGEANT = dict(ZOMBIE, legs="dkgrey", body="dkgrey", vest="black", arms="black", forearms="pale", hands="pale",
                head=_human_head("pale", "black", helmet=False), weapon=_shotgun, chest_r=7.6)
IMP = {
    "leg": 11, "hipw": 4.4, "thigh_r": 4.1, "foot": 4.5, "torso": 15, "waist_r": 6.4, "chest_r": 8.8,
    "arm": 10, "arm_r": 3.5, "head_r": 5.6, "legs": "impskin", "boots": "impskin", "body": "impskin",
    "hands": "impskin", "head": _imp_head, "extras": _imp_spikes, "extras_arm": _imp_arm_spikes, "head_fwd": 1.0,
}
BARON = {
    "leg": 14, "hipw": 5.0, "thigh_r": 4.6, "foot": 5.0, "torso": 20, "waist_r": 7.0, "chest_r": 10.5,
    "arm": 13, "arm_r": 3.8, "head_r": 6.2, "legs": "fur", "shins": "fur", "boots": "black", "body": "baron",
    "hands": "baron", "head": _baron_head, "extras_arm": _baron_hand, "hooves": True, "head_fwd": 1.5,
}


def _walk_pose(i: int) -> dict:
    s = (24, 0, -24, 0)[i % 4]
    k = (6, 30, 6, 30)[i % 4]
    return {"hip_l": s, "hip_r": -s, "knee_l": k if s >= 0 else 10, "knee_r": k if s <= 0 else 10,
            "sh_l": -s * 0.8, "sh_r": s * 0.8, "el_l": 20, "el_r": 20, "lean": 4}


# --- the frames of each monster ----------------------------------------------------------------------------------

ROTS = (0, 45, 90, 135, 180)


def _views(prims: list, bright: bool = False, seed: int = 1) -> list[Sprite]:
    return [render(prims, r, seed=seed, bright=bright) for r in ROTS]


def _death_frames(make, n_fall: int = 4, pool: float = 40) -> list[Sprite]:  # noqa: ANN001
    frames = []
    for i in range(n_fall + 1):
        t = i / n_fall
        pose = {"hip_l": 8 * t, "hip_r": 14 * t, "knee_l": 12 * t, "knee_r": 6 * t, "sh_l": 30 * t,
                "sh_r": 45 * t, "out_l": 20 + 50 * t, "out_r": 20 + 40 * t, "el_l": 10, "el_r": 20,
                "head": -30 * t, "shout": i < 2, "lean": -10 * t}
        s = render(make(pose), 90, fall_deg=-88 * t ** 1.4, seed=7 + i)
        if i >= n_fall - 1:
            s = add_pool(s, pool * (0.6 if i < n_fall else 1.0))
        frames.append(s)
    return frames


def _gib_frames(make, scale: float = 1.0, pool: float = 46) -> list[Sprite]:  # noqa: ANN001
    rng = np.random.default_rng(11)
    frames = []
    base_pose = {"sh_l": 40, "sh_r": 40, "out_l": 50, "out_r": 50}
    body = make(base_pose)
    for i in range(7):
        t = i / 6
        prims = []
        if i < 2:
            # the body bursting: shrinking upper half, a red cloud
            for p in body:
                if p[0] == "s" and p[1][1] > 26 * scale * (1.0 - 0.3 * i):
                    continue
                if p[0] == "c" and max(p[1][1], p[2][1]) > 26 * scale * (1.0 - 0.3 * i):
                    continue
                prims.append(p)
            for _ in range(10 + 6 * i):
                c = np.array([rng.normal(0, 7), 28 * scale + rng.normal(4 * i, 6), rng.normal(0, 5)])
                prims.append(sph(c, rng.uniform(2, 5), "gore"))
        else:
            height = max(30 * scale * (1 - t) ** 1.5, 2)
            for _ in range(int(26 - 2 * i)):
                c = np.array([rng.normal(0, 8 + 3 * t), abs(rng.normal(height * 0.5, height * 0.4)) + 1,
                              rng.normal(0, 5)])
                prims.append(sph(c, rng.uniform(1.8, 4.2), rng.choice(["gore", "blood", "pink"])))
            for _ in range(3):
                a = np.array([rng.normal(0, 6), rng.uniform(1, 3), rng.normal(0, 4)])
                prims.append(cap(a, a + rng.normal(0, 4, 3) * np.array([1, 0.2, 1]), 1.1, 0.9, "bone"))
        s = render(prims, 30, seed=20 + i)
        if i >= 3:
            s = add_pool(s, pool * min(1.0, 0.5 + t))
        frames.append(s)
    return frames


def _humanoid_set(body: dict, attack: list[dict], bright_attack: tuple[int, ...], pool: float = 40) -> dict:
    def make(pose: dict) -> list:
        return humanoid(body, pose)

    walk = [_views(make(dict(_walk_pose(i), **body.get("walk_arms", {}))), seed=i + 1) for i in range(4)]
    atk = [_views(make(p), bright=(j in bright_attack), seed=j + 5) for j, p in enumerate(attack)]
    pain = [_views(make({"lean": -14, "head": -30, "sh_l": 50, "sh_r": 30, "out_l": 40, "out_r": 40, "el_l": 40,
                         "el_r": 30, "shout": True, "knee_l": 10, "knee_r": 10}), seed=9)]
    unarmed = {k: v for k, v in body.items() if k != "weapon"}  # the gun is dropped as the body falls

    def make_dead(pose: dict) -> list:
        return humanoid(unarmed if pose.get("lean", 0) < -2 else body, pose)

    return {"walk": walk, "attack": atk, "pain": pain, "death": _death_frames(make_dead, pool=pool),
            "gib": _gib_frames(make, scale=body["leg"] / 10.5, pool=pool + 6)}


def build_zombie(body: dict = ZOMBIE) -> dict:
    aim = {"sh_r": 80, "el_r": 5, "sh_l": 70, "el_l": 20, "out_l": -20, "out_r": 5, "aim": True, "lean": 2}
    return _humanoid_set(body, [aim, dict(aim, flash=True), aim], (1,))


def build_sergeant() -> dict:
    return build_zombie(SERGEANT)


def build_imp() -> dict:
    a = {"sh_r": -50, "el_r": 70, "sh_l": 20, "lean": -6, "head": -10, "fireball": True, "shout": True}
    b = {"sh_r": 10, "el_r": 90, "sh_l": 20, "out_r": 10, "fireball": True, "shout": True}
    c = {"sh_r": 90, "el_r": 0, "sh_l": 30, "lean": 12, "shout": True}
    return _humanoid_set(IMP, [a, b, c], (0, 1))


def build_baron() -> dict:
    a = {"sh_r": -60, "el_r": 60, "sh_l": 30, "lean": -8, "fireball": True, "shout": True}
    b = {"sh_r": 20, "el_r": 80, "sh_l": 30, "fireball": True, "shout": True}
    c = {"sh_r": 95, "el_r": 0, "sh_l": 40, "lean": 14, "shout": True}
    return _humanoid_set(BARON, [a, b, c], (0, 1), pool=52)


# --- the demon (a hunched, big-jawed brute) ----------------------------------------------------------------------

def _demon(pose: dict) -> list:
    prims = []
    jaw = pose.get("jaw", 0.0)
    step = pose.get("step", 0.0)
    lean = pose.get("lean", 0.0)
    for side in (-1, 1):
        sw = step * side
        hip = np.array([side * 7.0, 22.0, -4.0])
        knee = hip + _limb_dir(sw + 35, 6, side) * 11
        ankle = knee + _limb_dir(sw - 35, 2, side) * 11
        prims.append(cap(hip, knee, 6.0, 4.6, "pink"))
        prims.append(cap(knee, ankle, 4.6, 3.0, "pink"))
        prims.append(cap(ankle, ankle + np.array([0, -1.0, 6.0]), 3.0, 2.4, "pinkdark"))
    body_c = _rot_x(np.array([0, 10.0, 4.0]), lean) + np.array([0, 24.0, 0])
    prims.append(cap(np.array([0, 22.0, -5.0]), body_c, 11.0, 13.0, "pink"))
    prims.append(sph(body_c + np.array([0, 2, 2]), 13.5, "pink"))
    for i in range(4):
        p = body_c + np.array([0, 6 - i * 5, -11 + i * 1.5])
        prims.append(sph(p, 3.0, "pinkdark"))
    head = body_c + np.array([0, 3.0, 13.0])
    prims.append(sph(head, 9.5, "pink"))
    prims.append(sph(head + np.array([0, 5.0, -2.0]), 7.5, "pink"))
    for side in (-1, 1):
        prims.append(sph(head + np.array([side * 4.0, 5.0, 6.0]), 1.6, "eye_red"))
        prims.append(cap(head + np.array([side * 6, 6, -2]), head + np.array([side * 8, 10, -5]), 2.0, 0.6, "bone"))
    # jaws with teeth
    upper = head + np.array([0, -1.0, 8.0])
    lower = head + _rot_x(np.array([0, -6.0, 8.0]), jaw)
    prims.append(sph(upper, 6.5, "pink"))
    prims.append(sph(head + np.array([0, -3.5, 8.0]), 5.0 + jaw * 0.05, "mouth"))
    prims.append(sph(lower, 5.5, "pinkdark"))
    for i in range(5):
        x = -4 + i * 2
        prims.append(cap(upper + np.array([x, -4.5, 3.5]), upper + np.array([x, -7.5 - jaw * 0.05, 3.8]), 0.9, 0.2,
                         "teeth"))
        prims.append(cap(lower + np.array([x, 4.0, 3.0]), lower + np.array([x, 6.5, 3.2]), 0.9, 0.2, "teeth"))
    for side in (-1, 1):
        sh = body_c + np.array([side * 11.0, 2.0, 6.0])
        reach = pose.get("reach", 0.0)
        el = sh + _limb_dir(50 + reach, 20, side) * 8
        hand = el + _limb_dir(90 + reach, 10, side) * 7
        prims.append(cap(sh, el, 3.4, 2.8, "pink"))
        prims.append(cap(el, hand, 2.8, 2.2, "pink"))
        prims.append(sph(hand, 2.6, "pinkdark"))
    return prims


def build_demon() -> dict:
    walk = [_views(_demon({"step": (22, 0, -22, 0)[i], "jaw": 10 + i * 4}), seed=i + 1) for i in range(4)]
    atk = [_views(_demon({"jaw": j, "lean": ln, "reach": r}), seed=5 + n)
           for n, (j, ln, r) in enumerate(((30, -10, 0), (55, 5, 20), (8, 20, 40)))]
    pain = [_views(_demon({"jaw": 45, "lean": -20}), seed=9)]

    def make(pose: dict) -> list:
        return _demon({"jaw": 40 if pose.get("shout") else 15})

    death = []
    for i in range(6):
        t = i / 5
        s = render(_demon({"jaw": 50 * (1 - t) + 10, "lean": -20 * t}), 90, fall_deg=-80 * t ** 1.3, seed=30 + i)
        if i >= 4:
            s = add_pool(s, 58)
        death.append(s)
    return {"walk": walk, "attack": atk, "pain": pain, "death": death, "gib": _gib_frames(make, 1.3, 60)}


# --- the cacodemon (a floating red ball with one green eye and a big mouth) --------------------------------------

def _caco_surface(open_: float, hurt: bool):  # noqa: ANN202
    def fn(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray) -> np.ndarray:
        out = np.full(nx.shape, "caco", dtype=object)
        curve = 0.25 * nx * nx
        mouth = (nz > 0.3) & (ny < -0.08 + curve) & (ny > -0.22 - 0.42 * open_ + curve * 0.6) & (np.abs(nx) < 0.66)
        out[mouth] = "mouth"
        edge = (ny > -0.14 + curve) | (ny < -0.17 - 0.42 * open_ + curve * 0.6)
        teeth = mouth & edge & ((np.floor(nx * 22) % 2) == 0)
        out[teeth] = "teeth"
        eye = ((nx ** 2 + (ny - 0.32) ** 2) < 0.07) & (nz > 0)
        out[eye] = "eye_white"
        iris = ((nx ** 2 + (ny - 0.32) ** 2) < 0.022) & (nz > 0)
        out[iris] = "eye_green" if not hurt else "red"
        pupil = ((nx ** 2 + (ny - 0.32) ** 2) < 0.005) & (nz > 0)
        out[pupil] = "eye_dark"
        spots = (np.sin(nx * 23) * np.sin(ny * 19) * np.sin(nz * 17) > 0.7) & ~mouth & ~eye
        out[spots] = "pinkdark"
        out[mouth & ~teeth & (ny < -0.14 - 0.3 * open_ + curve * 0.6)] = "cacoblue"
        return out
    return fn


def _caco(open_: float = 0.2, hurt: bool = False, y: float = 6.0) -> list:
    c = np.array([0, 31.0 + y, 0])
    prims = [sph(c, 30.0, "caco", _caco_surface(open_, hurt))]
    for side in (-1, 1):
        for a in (40, 80):
            base = c + np.array([side * 26 * math.sin(math.radians(a)), 26 * math.cos(math.radians(a)), -6])
            tip = base + np.array([side * 6, 5, -3])
            prims.append(cap(base, tip, 3.0, 0.6, "horn"))
    prims.append(cap(c + np.array([0, 28, -4]), c + np.array([0, 36, -8]), 3.0, 0.6, "horn"))
    return prims


def build_caco() -> dict:
    walk = [_views(_caco(0.2, y=6 + 3 * i), seed=1) for i in range(2)]
    atk = [_views(_caco(0.5), seed=2), _views(_caco(0.8), seed=3), _views(_caco(0.95), bright=True, seed=4)]
    pain = [_views(_caco(0.7, hurt=True), seed=5)]
    death = []
    rng = np.random.default_rng(5)
    for i in range(6):
        t = i / 5
        prims = []
        if i < 3:
            prims = _caco(0.9, hurt=True, y=6 - 20 * t)
            for _ in range(i * 6):
                prims.append(sph([rng.normal(0, 18), 30 + rng.normal(0, 14), 20], rng.uniform(2, 5), "gore"))
        else:
            for _ in range(22):
                prims.append(sph([rng.normal(0, 16), abs(rng.normal(4, 4)) + 2, rng.normal(0, 8)], rng.uniform(3, 7),
                                 rng.choice(["caco", "gore", "cacoblue"])))
            prims.append(sph([0, 8, 6], 4, "eye_white"))
            prims.append(sph([0, 8, 9], 1.8, "eye_green"))
        s = render(prims, 0, seed=40 + i)
        if i >= 3:
            s = add_pool(s, 64, "blood")
        death.append(s)
    return {"walk": walk, "attack": atk, "pain": pain, "death": death, "gib": death}


# --- the lost soul (a flaming flying skull) ----------------------------------------------------------------------

def _soul(phase: int, charge: bool = False) -> list:
    c = np.array([0, 14.0, 0])
    prims = [sph(c, 8.5, "bone"), sph(c + np.array([0, -4.5, 3.0]), 6.0, "bone")]
    for side in (-1, 1):
        prims.append(sph(c + np.array([side * 3.4, 0.5, 6.6]), 2.4, "eye_dark"))
        prims.append(sph(c + np.array([side * 3.4, 0.5, 7.6]), 1.0, "eye_red"))
        prims.append(cap(c + np.array([side * 5, 6, -2]), c + np.array([side * 8, 11, -5]), 1.4, 0.3, "bone"))
    prims.append(sph(c + np.array([0, -7.5, 6.0]), 2.6 if charge else 1.6, "mouth"))
    for i in range(5):
        x = -2.5 + i * 1.25
        prims.append(sph(c + np.array([x, -6.0, 7.5]), 0.8, "teeth"))
    rng = np.random.default_rng(phase)
    for _ in range(14):
        p = c + np.array([rng.normal(0, 4), rng.uniform(4, 16), rng.normal(-7, 3)])
        prims.append(sph(p, rng.uniform(2.0, 4.0), "flame"))
    return prims


def build_soul() -> dict:
    walk = [_views(_soul(i), bright=True, seed=i) for i in range(2)]
    atk = [_views(_soul(2, True), bright=True), _views(_soul(3, True), bright=True)]
    pain = [_views(_soul(4, True), bright=True)]
    death = []
    rng = np.random.default_rng(9)
    for i in range(6):
        prims = []
        for _ in range(16 - i * 2):
            prims.append(sph([rng.normal(0, 6 + i * 3), 14 + rng.normal(0, 5 + i * 2), rng.normal(0, 4)],
                             rng.uniform(2, 5) * (1 - i * 0.12), "flame"))
        if i < 3:
            prims.extend(_soul(5)[:2])
        death.append(render(prims, 0, seed=50 + i, bright=True))
    return {"walk": walk, "attack": atk, "pain": pain, "death": death, "gib": death}


BUILDERS = {
    "zombie": build_zombie, "sergeant": build_sergeant, "imp": build_imp, "demon": build_demon,
    "soul": build_soul, "caco": build_caco, "baron": build_baron,
}

_CACHE: dict[str, dict] = {}


def monster_frames(kind: str) -> dict:
    """{"walk": [[5 views] x frames], "attack": ..., "pain": ..., "death": [frames], "gib": [frames]}."""
    if kind not in _CACHE:
        _CACHE[kind] = BUILDERS[kind]()
    return _CACHE[kind]


def view_sprite(views: list[Sprite], rel: float) -> tuple[Sprite, bool]:
    """The view of a 5-view frame for the viewer at ``rel`` radians off the monster's facing (+: its right)."""
    deg = math.degrees(rel)
    deg = (deg + 180.0) % 360.0 - 180.0
    i = min(int(round(abs(deg) / 45.0)), 4)
    return views[i], deg < 0
