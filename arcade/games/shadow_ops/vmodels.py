"""View-model geometry: boxes, beams, tapered cylinders and decals in gun space (x right, y up, z forward, the line of
sight at y = 0), the materials, and a detailed model per weapon class with gloved hands and camo sleeves."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# material: (base rgb, specular strength, shininess, texture key or "")
MATS = {
    "metal": ((62, 64, 70), 0.55, 24, ""),
    "dark": ((40, 41, 45), 0.35, 18, ""),
    "poly": ((50, 51, 50), 0.22, 10, ""),
    "tan": ((140, 120, 86), 0.18, 8, ""),
    "olive": ((84, 92, 64), 0.2, 8, ""),
    "wood": ((118, 74, 40), 0.3, 12, ""),
    "chrome": ((150, 156, 164), 0.9, 40, ""),
    "brass": ((176, 140, 60), 0.8, 30, ""),
    "red": ((168, 40, 30), 0.6, 26, ""),
    "glove": ((44, 44, 42), 0.15, 8, "fabric"),
    "pad": ((34, 34, 34), 0.3, 12, ""),
    "knuckle": ((132, 116, 88), 0.2, 8, ""),
    "sleeve": ((90, 96, 80), 0.05, 4, "camo"),
    "cuff": ((44, 46, 40), 0.08, 4, ""),
    "lens": ((40, 60, 90), 1.0, 60, ""),
    "glow": ((120, 255, 140), 0.0, 1, ""),
    "glowred": ((255, 70, 50), 0.0, 1, ""),
    "glass": ((120, 170, 180), 0.6, 30, ""),
}
GUN_MATS = {"metal", "dark", "poly", "tan", "olive", "wood", "red"}


@dataclass
class Face:
    pts: list  # 3D points (gun space)
    normal: tuple
    uv: tuple = (0.0, 0.0, 1.0, 1.0)  # texture sub-rect


@dataclass
class Part:
    faces: list
    mat: str
    tag: str = "body"
    decal: str = ""  # a decal image key (single-face parts)


def _norm(v: tuple) -> tuple:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return v[0] / n, v[1] / n, v[2] / n


def _cross(a: tuple, b: tuple) -> tuple:
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def box(x0, x1, y0, y1, z0, z1, mat="metal", tag="body") -> Part:  # noqa: ANN001
    c = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    quads = (((0, 2, 3, 1), (0, 0, -1)), ((4, 5, 7, 6), (0, 0, 1)), ((0, 1, 5, 4), (0, -1, 0)),
             ((2, 6, 7, 3), (0, 1, 0)), ((0, 4, 6, 2), (-1, 0, 0)), ((1, 3, 7, 5), (1, 0, 0)))
    return Part([Face([c[i] for i in q], n) for q, n in quads], mat, tag)


def _frame(a: tuple, b: tuple, up: tuple = (0.0, 1.0, 0.0)) -> tuple:
    v = _norm((b[0] - a[0], b[1] - a[1], b[2] - a[2]))
    u = _cross(up, v)
    if math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2) < 1e-6:
        u = (1.0, 0.0, 0.0)
    u = _norm(u)
    w = _norm(_cross(v, u))
    return v, u, w


def beam(a, b, wdt, hgt, mat="poly", tag="body", up=(0.0, 1.0, 0.0)) -> Part:  # noqa: ANN001
    """A box from ``a`` to ``b``: ``wdt`` across (sideways), ``hgt`` the other way."""
    v, u, w = _frame(a, b, up)
    c = []
    for p in (a, b):
        for k in (-0.5, 0.5):
            for j in (-0.5, 0.5):
                c.append(tuple(p[i] + u[i] * j * wdt + w[i] * k * hgt for i in range(3)))
    neg = tuple(-x for x in v)
    quads = (((0, 2, 3, 1), neg), ((4, 5, 7, 6), v), ((0, 1, 5, 4), tuple(-x for x in w)), ((2, 6, 7, 3), w),
             ((0, 4, 6, 2), tuple(-x for x in u)), ((1, 3, 7, 5), u))
    return Part([Face([c[i] for i in q], n) for q, n in quads], mat, tag)


def cyl(a, b, r, mat="metal", tag="body", n=8, r2=None, caps=True) -> Part:  # noqa: ANN001
    """A prism of ``n`` sides from ``a`` (radius ``r``) to ``b`` (radius ``r2``)."""
    r2 = r if r2 is None else r2
    v, u, w = _frame(a, b, (0.0, 1.0, 0.0) if abs(b[1] - a[1]) < 0.9 * math.dist(a, b) else (0.0, 0.0, 1.0))
    ring = []
    for k in range(n):
        ang = 2 * math.pi * (k + 0.5) / n
        ring.append((math.cos(ang), math.sin(ang)))
    pa = [tuple(a[i] + (u[i] * cx + w[i] * sy) * r for i in range(3)) for cx, sy in ring]
    pb = [tuple(b[i] + (u[i] * cx + w[i] * sy) * r2 for i in range(3)) for cx, sy in ring]
    faces = []
    for k in range(n):
        k2 = (k + 1) % n
        ang = 2 * math.pi * (k + 1) / n
        nrm = _norm(tuple(u[i] * math.cos(ang) + w[i] * math.sin(ang) for i in range(3)))
        faces.append(Face([pa[k], pa[k2], pb[k2], pb[k]], nrm, (k / n, 0.0, 1.0 / n, 1.0)))
    if caps:
        faces.append(Face(list(reversed(pa)), tuple(-x for x in v)))
        faces.append(Face(pb, v))
    return Part(faces, mat, tag)


def decal(p0, p1, p2, p3, key, normal, tag="body") -> Part:  # noqa: ANN001
    """A flat image on a surface (corners in reading order: top-left, top-right, bottom-right, bottom-left)."""
    off = tuple(x * 0.0006 for x in normal)
    pts = [tuple(p[i] + off[i] for i in range(3)) for p in (p0, p1, p2, p3)]
    return Part([Face(pts, normal)], "decal", tag, key)


def side_decal(x, y0, y1, z0, z1, key, tag="body") -> Part:  # noqa: ANN001
    """A decal on a left-facing side (x = ``x``), read from the left."""
    return decal((x, y1, z1), (x, y1, z0), (x, y0, z0), (x, y0, z1), key, (-1.0, 0.0, 0.0), tag)


def top_decal(y, x0, x1, z0, z1, key, tag="body") -> Part:  # noqa: ANN001
    return decal((x0, y, z1), (x1, y, z1), (x1, y, z0), (x0, y, z0), key, (0.0, 1.0, 0.0), tag)


# --- hands -----------------------------------------------------------------------------------------------------------

def right_hand(gx: float, gy: float, gz: float, slant: float = 0.3) -> list[Part]:
    """A gloved shooting hand round a pistol grip at (gx, gy, gz) (the top of the grip), its sleeve off screen."""
    parts = [beam((gx + 0.02, gy - 0.005, gz - 0.02), (gx + 0.022, gy - 0.085, gz - 0.02 - slant * 0.08), 0.022,
                  0.06, "glove", "hand_r")]
    for k in range(4):  # fingers round the front of the grip
        y = gy - 0.018 - k * 0.019
        z = gz + 0.026 - (y - gy) * -slant * 0.6
        parts.append(beam((gx + 0.026, y, z - 0.006), (gx - 0.026, y - 0.003, z - 0.004), 0.017, 0.018, "glove",
                          "hand_r"))
    parts.append(beam((gx + 0.01, gy + 0.004, gz - 0.035), (gx - 0.022, gy - 0.004, gz + 0.03), 0.018, 0.018, "glove",
                      "hand_r"))  # thumb along the left side
    parts.append(box(gx + 0.03, gx + 0.04, gy - 0.07, gy - 0.01, gz - 0.04, gz + 0.01, "knuckle", "hand_r"))
    parts.append(cyl((gx + 0.028, gy - 0.05, gz - 0.05), (gx + 0.05, gy - 0.1, gz - 0.12), 0.034, "cuff", "hand_r",
                     n=6, r2=0.037, caps=False))
    parts.append(cyl((gx + 0.05, gy - 0.1, gz - 0.12), (gx + 0.15, gy - 0.35, gz - 0.42), 0.042, "sleeve",
                     "sleeve_r", n=6, r2=0.05, caps=False))
    return parts


def left_hand_under(fx: float, fy: float, fz: float) -> list[Part]:
    """The support hand under a handguard whose bottom is at (fx, fy, fz)."""
    parts = [box(fx - 0.03, fx + 0.03, fy - 0.03, fy, fz - 0.04, fz + 0.04, "glove", "hand_l")]
    for k in range(4):  # fingers up the left side
        z = fz - 0.03 + k * 0.02
        parts.append(beam((fx - 0.03, fy - 0.01, z), (fx - 0.032, fy + 0.032, z + 0.004), 0.016, 0.016, "glove",
                          "hand_l", up=(0.0, 0.0, 1.0)))
    parts.append(beam((fx + 0.02, fy - 0.012, fz - 0.03), (fx + 0.03, fy + 0.025, fz + 0.03), 0.017, 0.017, "glove",
                      "hand_l"))
    parts.append(box(fx - 0.036, fx - 0.028, fy + 0.008, fy + 0.036, fz - 0.036, fz + 0.04, "knuckle", "hand_l"))
    parts.append(cyl((fx - 0.005, fy - 0.03, fz - 0.03), (fx - 0.04, fy - 0.08, fz - 0.1), 0.034, "cuff", "hand_l",
                     n=6, r2=0.037, caps=False))
    parts.append(cyl((fx - 0.04, fy - 0.08, fz - 0.1), (fx - 0.2, fy - 0.33, fz - 0.4), 0.042, "sleeve", "sleeve_l",
                     n=6, r2=0.05, caps=False))
    return parts


def left_hand_cup(gx: float, gy: float, gz: float) -> list[Part]:
    """The support hand cupping the shooting hand (pistols)."""
    parts = [beam((gx - 0.03, gy - 0.03, gz - 0.01), (gx - 0.03, gy - 0.1, gz - 0.03), 0.024, 0.065, "glove",
                  "hand_l")]
    for k in range(3):
        y = gy - 0.05 - k * 0.018
        parts.append(beam((gx - 0.04, y, gz + 0.03), (gx + 0.02, y - 0.004, gz + 0.045), 0.016, 0.016, "glove",
                          "hand_l"))
    parts.append(cyl((gx - 0.035, gy - 0.09, gz - 0.04), (gx - 0.07, gy - 0.14, gz - 0.11), 0.033, "cuff", "hand_l",
                     n=6, r2=0.036, caps=False))
    parts.append(cyl((gx - 0.07, gy - 0.14, gz - 0.11), (gx - 0.2, gy - 0.38, gz - 0.42), 0.042, "sleeve",
                     "sleeve_l", n=6, r2=0.05, caps=False))
    return parts


# --- sights ----------------------------------------------------------------------------------------------------------

def iron(z_rear: float, z_front: float, base: float) -> list[Part]:
    """A rear aperture and a winged front post on the line of sight (y = 0)."""
    return [box(-0.014, -0.004, base, 0.006, z_rear, z_rear + 0.012, "dark", "sight"),
            box(0.004, 0.014, base, 0.006, z_rear, z_rear + 0.012, "dark", "sight"),
            box(-0.014, 0.014, base, base + 0.006, z_rear - 0.01, z_rear + 0.02, "dark", "sight"),
            box(-0.0022, 0.0022, base, 0.0, z_front, z_front + 0.006, "dark", "sight"),
            box(-0.012, -0.008, base, 0.004, z_front - 0.004, z_front + 0.01, "dark", "sight"),
            box(0.008, 0.012, base, 0.004, z_front - 0.004, z_front + 0.01, "dark", "sight"),
            box(-0.012, 0.012, base - 0.012, base, z_front - 0.01, z_front + 0.016, "dark")]


def reflex(z: float, base: float) -> list[Part]:
    return [box(-0.017, 0.017, base, base + 0.008, z - 0.04, z + 0.03, "dark"),
            box(-0.02, -0.015, base + 0.008, 0.024, z, z + 0.014, "dark"),
            box(0.015, 0.02, base + 0.008, 0.024, z, z + 0.014, "dark"),
            box(-0.02, 0.02, 0.024, 0.03, z - 0.005, z + 0.016, "dark"),
            box(-0.008, 0.008, base + 0.008, base + 0.014, z - 0.035, z - 0.01, "metal"),
            decal((-0.015, 0.024, z + 0.007), (0.015, 0.024, z + 0.007), (0.015, base + 0.008, z + 0.007),
                  (-0.015, base + 0.008, z + 0.007), "glass", (0.0, 0.0, -1.0), "glass")]


def scope(z0: float, z1: float, base: float, r: float = 0.017, big: float = 0.027) -> list[Part]:
    parts = [cyl((0, 0, z0 + 0.05), (0, 0, z1 - 0.07), r, "dark", "scope", n=10),
             cyl((0, 0, z1 - 0.07), (0, 0, z1 - 0.03), r, "dark", "scope", n=10, r2=big, caps=False),
             cyl((0, 0, z1 - 0.03), (0, 0, z1), big, "dark", "scope", n=10),
             cyl((0, 0, z0), (0, 0, z0 + 0.05), r * 1.3, "dark", "scope", n=10),
             cyl((0, r * 0.6, (z0 + z1) / 2 - 0.01), (0, r + 0.012, (z0 + z1) / 2 - 0.01), 0.009, "metal", n=8),
             cyl((r * 0.6, 0, (z0 + z1) / 2 - 0.01), (r + 0.012, 0, (z0 + z1) / 2 - 0.01), 0.009, "metal", n=8)]
    for zr in (z0 + 0.07, z1 - 0.1):
        parts.append(cyl((0, 0, zr), (0, 0, zr + 0.012), r + 0.003, "metal", n=10))
        parts.append(box(-0.009, 0.009, base, -r, zr, zr + 0.012, "metal"))
    parts.append(decal((-big * 0.7, big * 0.7, z1 + 0.001), (big * 0.7, big * 0.7, z1 + 0.001),
                       (big * 0.7, -big * 0.7, z1 + 0.001), (-big * 0.7, -big * 0.7, z1 + 0.001), "lens",
                       (0.0, 0.0, 1.0)))
    return parts


# --- the weapons -----------------------------------------------------------------------------------------------------

@dataclass
class Model:
    parts: list = field(default_factory=list)
    muzzle: tuple = (0.0, -0.05, 0.45)
    eject: tuple = (0.03, -0.04, 0.0)
    hip: tuple = (0.11, -0.085, 0.34)
    ads_z: float = 0.26


def _rifle(kind: str, sight: str, name: str) -> Model:
    m = Model()
    P = m.parts
    by = -0.05  # the bore line below the line of sight (the sights ride on a rail)
    long_ = {"ar": 0.44, "ar2": 0.44, "smg": 0.3, "smg2": 0.26, "dmr": 0.52, "lmg": 0.5, "sniper": 0.62}.get(kind, 0.44)
    stock_mat = "wood" if kind == "dmr" else ("tan" if kind == "sniper" else "poly")
    hg_mat = "wood" if kind == "dmr" else ("tan" if kind in ("ar2", "smg2") else "poly")
    # receiver (upper and lower), dust cover, charging handle, markings
    P += [box(-0.022, 0.022, by - 0.018, by + 0.016, -0.13, 0.09, "metal"),
          box(-0.02, 0.02, by - 0.05, by - 0.018, -0.11, 0.05, "dark"),
          box(-0.006, 0.006, by + 0.016, by + 0.022, -0.14, -0.1, "metal"),
          side_decal(-0.0221, by - 0.046, by - 0.024, -0.1, 0.02, f"text:{name}"),
          side_decal(-0.0225, by - 0.012, by + 0.01, -0.06, 0.03, "port"),
          box(-0.028, -0.02, by - 0.042, by - 0.032, -0.07, -0.05, "metal")]  # the selector
    # top rail
    if kind not in ("dmr",):
        P += [box(-0.012, 0.012, by + 0.016, by + 0.026, -0.12, 0.3 if kind != "smg2" else 0.2, "dark"),
              top_decal(by + 0.0262, -0.012, 0.012, -0.12, 0.3 if kind != "smg2" else 0.2, "rail")]
    # handguard
    hz1 = long_ * 0.72
    if kind in ("ar", "lmg", "smg"):
        P += [box(-0.025, 0.025, by - 0.026, by + 0.018, 0.09, hz1, hg_mat),
              side_decal(-0.0251, by - 0.02, by + 0.012, 0.11, hz1 - 0.02, "vents")]
    elif kind in ("ar2", "smg2"):
        P += [cyl((0, by - 0.004, 0.09), (0, by - 0.004, hz1), 0.026, hg_mat, n=8),
              side_decal(-0.024, by - 0.012, by + 0.006, 0.11, hz1 - 0.02, "grip_tex")]
    elif kind == "dmr":
        P += [box(-0.024, 0.024, by - 0.03, by + 0.008, -0.36, hz1, "wood"),
              box(-0.013, 0.013, by + 0.008, by + 0.016, 0.0, hz1, "wood")]
    elif kind == "sniper":
        P += [box(-0.024, 0.024, by - 0.03, by + 0.004, 0.09, hz1 - 0.05, "tan")]
    # barrel, gas block, muzzle device
    P += [cyl((0, by, hz1), (0, by, long_), 0.0085 if kind != "lmg" else 0.012, "metal", n=8),
          cyl((0, by, long_), (0, by, long_ + 0.05), 0.014, "dark", n=8),
          side_decal(-0.0105, by - 0.008, by + 0.008, long_ + 0.008, long_ + 0.044, "slots")]
    if kind == "lmg":
        P += [cyl((0, by, hz1), (0, by, long_ - 0.04), 0.02, "dark", n=8, caps=False),
              side_decal(-0.0201, by - 0.012, by + 0.012, hz1 + 0.01, long_ - 0.06, "holes"),
              box(-0.03, 0.03, by + 0.016, by + 0.04, -0.12, 0.08, "dark"),  # feed cover
              box(-0.006, 0.006, by + 0.04, by + 0.062, -0.04, 0.06, "dark"),  # carry handle
              beam((-0.006, by - 0.02, long_ - 0.06), (-0.016, by - 0.03, long_ - 0.24), 0.008, 0.008, "metal"),
              beam((0.006, by - 0.02, long_ - 0.06), (0.016, by - 0.03, long_ - 0.24), 0.008, 0.008, "metal")]
    if kind == "sniper":
        P += [cyl((0, by, long_ - 0.06), (0, by, long_ - 0.01), 0.013, "dark", n=8),
              beam((0.02, by + 0.004, -0.06), (0.06, by - 0.02, -0.06), 0.008, 0.008, "metal", "bolt"),
              cyl((0.058, by - 0.018, -0.068), (0.058, by - 0.018, -0.05), 0.012, "metal", "bolt", n=8)]
    # magazine
    mz = 0.0 if kind not in ("lmg",) else -0.03
    if kind in ("ar", "ar2", "smg"):
        curve = {"ar": 0.02, "ar2": 0.05, "smg": 0.035}[kind]
        wdt = 0.024 if kind != "smg" else 0.02
        dep = 0.062 if kind != "smg" else 0.036
        P += [beam((0, by - 0.045, mz + 0.03), (0, by - 0.11, mz + 0.03 + curve * 0.4), wdt, dep, "dark", "mag"),
              beam((0, by - 0.11, mz + 0.03 + curve * 0.4), (0, by - 0.17, mz + 0.03 + curve * 1.4), wdt, dep, "dark",
                   "mag"),
              side_decal(-wdt / 2 - 0.0003, by - 0.1, by - 0.06, mz + 0.006, mz + 0.054, "mag_ribs", "mag")]
    elif kind == "smg2":
        P += [beam((0, by - 0.045, 0.06), (0, by - 0.16, 0.08), 0.02, 0.03, "dark", "mag")]
    elif kind == "lmg":
        P += [box(-0.04, 0.03, by - 0.13, by - 0.05, mz - 0.01, mz + 0.07, "olive", "mag"),
              side_decal(-0.0403, by - 0.12, by - 0.06, mz, mz + 0.06, "text:7.62 LINKED", "mag")]
    else:
        P += [box(-0.014, 0.014, by - 0.12, by - 0.045, 0.0, 0.05, "dark", "mag")]
    # grip, trigger guard, stock
    P += [beam((0, by - 0.048, -0.075), (0, by - 0.13, -0.1), 0.027, 0.04, "poly" if kind != "dmr" else "wood"),
          side_decal(-0.0136, by - 0.12, by - 0.07, -0.115, -0.08, "grip_tex"),
          box(-0.004, 0.004, by - 0.08, by - 0.074, -0.06, 0.0, "dark")]
    if kind in ("ar", "lmg", "smg"):
        P += [cyl((0, by - 0.01, -0.13), (0, by - 0.01, -0.24), 0.015, "dark", n=8),
              box(-0.02, 0.02, by - 0.07, by + 0.006, -0.38, -0.2, stock_mat)]
    elif kind in ("ar2", "smg2"):
        P += [beam((0, by - 0.005, -0.13), (0, by - 0.03, -0.36), 0.03, 0.05, stock_mat)]
    elif kind == "dmr":
        P += [beam((0, by - 0.01, -0.13), (0, by - 0.05, -0.42), 0.034, 0.06, "wood")]
    elif kind == "sniper":
        P += [box(-0.022, 0.022, by - 0.075, by + 0.01, -0.42, -0.13, "tan"),
              box(-0.018, 0.018, by + 0.01, by + 0.024, -0.38, -0.2, "tan")]
    # sights
    if sight == "scope" or kind == "sniper":
        P += scope(-0.16, 0.16, by + 0.026)
    elif sight == "acog":
        P += scope(-0.08, 0.09, by + 0.026, 0.014, 0.02)
    elif sight == "reddot":
        P += reflex(-0.03, by + 0.026)
    else:
        P += iron(-0.11, long_ - 0.06, by + 0.022 if kind != "dmr" else by + 0.016)
    # hands
    P += right_hand(0.0, by - 0.05, -0.085)
    P += left_hand_under(0.0, by - 0.028 if kind not in ("ar2", "smg2") else by - 0.03, min(hz1 - 0.07, 0.24))
    m.muzzle = (0.0, by, long_ + 0.06)
    if kind in ("smg", "smg2"):
        m.hip = (0.105, -0.08, 0.32)
    return m


def _shotgun(kind: str, sight: str) -> Model:
    m = Model()
    P = m.parts
    by = -0.03
    if kind == "shotgun":
        P += [box(-0.022, 0.022, by - 0.04, by + 0.012, -0.12, 0.07, "dark"),
              side_decal(-0.0221, by - 0.03, by + 0.004, -0.1, 0.05, "text:RIDGEBACK 12GA"),
              cyl((0, by, 0.07), (0, by, 0.46), 0.011, "metal"),
              cyl((0, by - 0.026, 0.07), (0, by - 0.026, 0.42), 0.012, "dark"),
              cyl((0, by - 0.02, 0.16), (0, by - 0.02, 0.28), 0.024, "poly", "pump", n=10),
              side_decal(-0.0241, by - 0.03, by - 0.01, 0.17, 0.27, "grip_tex", "pump"),
              beam((0, by - 0.035, -0.07), (0, by - 0.13, -0.1), 0.026, 0.04, "poly"),
              box(-0.004, 0.004, by - 0.07, by - 0.064, -0.05, 0.0, "dark"),
              beam((0, by - 0.01, -0.12), (0, by - 0.04, -0.4), 0.032, 0.055, "poly"),
              box(-0.003, 0.003, by + 0.011, by + 0.016, 0.44, 0.446, "brass", "sight"),
              box(-0.008, 0.008, by + 0.012, by + 0.016, -0.12, 0.42, "dark")]
        P += left_hand_under(0.0, by - 0.044, 0.22)
    else:
        P += [box(-0.022, 0.022, by - 0.04, by + 0.012, -0.12, 0.06, "metal"),
              side_decal(-0.0221, by - 0.035, by + 0.004, -0.1, 0.04, "engrave"),
              cyl((-0.0105, by, 0.06), (-0.0105, by, 0.48), 0.0105, "dark"),
              cyl((0.0105, by, 0.06), (0.0105, by, 0.48), 0.0105, "dark"),
              box(-0.002, 0.002, by + 0.0105, by + 0.014, 0.06, 0.48, "dark"),
              box(-0.02, 0.02, by - 0.036, by - 0.01, 0.06, 0.26, "wood"),
              beam((0, by - 0.02, -0.12), (0, by - 0.07, -0.42), 0.034, 0.06, "wood"),
              box(-0.004, 0.004, by - 0.06, by - 0.054, -0.05, 0.0, "dark"),
              box(-0.002, 0.002, by + 0.014, by + 0.02, 0.47, 0.474, "brass", "sight")]
        P += left_hand_under(0.0, by - 0.036, 0.18)
    P += right_hand(0.0, by - 0.04, -0.08)
    m.muzzle = (0.0, by, 0.49)
    return m


def _pistol(kind: str, name: str) -> Model:
    m = Model(hip=(0.095, -0.065, 0.28), ads_z=0.3)
    P = m.parts
    if kind == "pistol":
        P += [box(-0.0135, 0.0135, -0.03, -0.006, -0.07, 0.12, "metal", "slide"),
              side_decal(-0.0136, -0.026, -0.01, -0.065, -0.03, "serrations", "slide"),
              side_decal(-0.0136, -0.024, -0.012, 0.0, 0.09, f"text:{name}", "slide"),
              box(-0.0125, 0.0125, -0.046, -0.03, -0.06, 0.11, "poly"),
              cyl((0, -0.02, 0.11), (0, -0.02, 0.125), 0.006, "dark", "slide", n=8),
              box(-0.004, 0.004, -0.064, -0.058, -0.04, 0.0, "poly"),
              beam((0, -0.044, -0.045), (0, -0.13, -0.075), 0.025, 0.037, "poly"),
              side_decal(-0.0126, -0.12, -0.06, -0.09, -0.05, "grip_tex"),
              box(-0.012, 0.012, -0.142, -0.13, -0.09, -0.05, "dark", "mag"),
              box(-0.011, -0.003, -0.008, 0.003, -0.068, -0.058, "dark", "slide"),
              box(0.003, 0.011, -0.008, 0.003, -0.068, -0.058, "dark", "slide"),
              box(-0.002, 0.002, -0.008, 0.0, 0.105, 0.112, "dark", "slide")]
        m.muzzle = (0.0, -0.02, 0.13)
    elif kind == "revolver":
        P += [box(-0.012, 0.012, -0.04, -0.008, -0.06, 0.03, "metal"),
              cyl((0, -0.026, -0.03), (0, -0.026, 0.035), 0.024, "metal", n=10),
              side_decal(-0.0241, -0.04, -0.012, -0.025, 0.03, "flutes"),
              cyl((0, -0.016, 0.035), (0, -0.016, 0.21), 0.01, "metal", n=8),
              box(-0.004, 0.004, -0.008, -0.002, 0.03, 0.21, "metal"),
              box(-0.002, 0.002, -0.004, 0.0, 0.2, 0.206, "brass", "sight"),
              box(-0.004, 0.004, -0.012, 0.004, -0.07, -0.055, "dark"),
              beam((0, -0.04, -0.05), (0, -0.13, -0.085), 0.026, 0.04, "wood"),
              side_decal(-0.0131, -0.12, -0.06, -0.1, -0.06, "engrave"),
              box(-0.004, 0.004, -0.06, -0.055, -0.04, 0.0, "dark")]
        m.muzzle = (0.0, -0.016, 0.22)
    else:  # the Arc Caster: an original arc-welder pistol
        P += [cyl((0, -0.03, -0.06), (0, -0.03, 0.06), 0.03, "red", n=12),
              cyl((0, -0.03, 0.06), (0, -0.03, 0.17), 0.022, "chrome", n=10, r2=0.012),
              cyl((0, -0.03, -0.075), (0, -0.03, -0.06), 0.024, "chrome", n=12),
              box(-0.003, 0.003, 0.0, 0.03, -0.04, 0.05, "red"),
              box(-0.003, 0.003, -0.075, -0.06, -0.03, 0.04, "red"),
              cyl((0, 0.002, -0.03), (0, 0.002, 0.03), 0.009, "glow", "glow", n=8),
              beam((0, -0.05, -0.04), (0, -0.14, -0.07), 0.026, 0.038, "brass"),
              side_decal(-0.0131, -0.13, -0.07, -0.08, -0.04, "grip_tex"),
              box(-0.004, 0.004, -0.07, -0.064, -0.035, 0.0, "chrome"),
              box(-0.002, 0.002, -0.004, 0.0, 0.14, 0.146, "chrome", "sight"),
              box(-0.008, -0.002, -0.004, 0.004, -0.07, -0.062, "chrome", "sight"),
              box(0.002, 0.008, -0.004, 0.004, -0.07, -0.062, "chrome", "sight"),
              side_decal(-0.0301, -0.045, -0.015, -0.04, 0.04, "text:ZAPPER MK II")]
        for z in (0.085, 0.115, 0.145):
            P.append(cyl((0, -0.03, z), (0, -0.03, z + 0.008), 0.025 - (z - 0.07) * 0.1, "glow", "glow", n=10))
        m.muzzle = (0.0, -0.03, 0.18)
    P += right_hand(0.0, -0.044, -0.06, 0.4)
    P += left_hand_cup(0.0, -0.044, -0.06)
    return m


def _launcher(name: str) -> Model:
    m = Model(hip=(0.1, -0.06, 0.3), ads_z=0.26)
    P = m.parts
    tx, ty = 0.06, -0.06
    P += [cyl((tx, ty, -0.38), (tx, ty, 0.36), 0.045, "olive", n=12),
          cyl((tx, ty, 0.36), (tx, ty, 0.42), 0.045, "dark", n=12, r2=0.052),
          cyl((tx, ty, 0.42), (tx, ty, 0.52), 0.04, "olive", "body", n=10, r2=0.012),
          cyl((tx, ty, -0.42), (tx, ty, -0.38), 0.05, "dark", n=12),
          side_decal(tx - 0.0455, ty - 0.02, ty + 0.02, -0.2, 0.2, f"text:{name}"),
          box(-0.012, 0.012, -0.03, 0.0, -0.05, 0.0, "dark", "sight"),
          box(-0.003, 0.003, -0.02, 0.0, 0.2, 0.206, "dark", "sight"),
          box(-0.004, 0.004, -0.03, -0.02, -0.05, 0.21, "dark"),
          beam((tx - 0.01, ty - 0.04, -0.06), (tx - 0.01, ty - 0.12, -0.09), 0.026, 0.04, "poly")]
    P += right_hand(tx - 0.01, ty - 0.04, -0.075)
    P += left_hand_under(tx, ty - 0.045, 0.16)
    m.muzzle = (tx, ty, 0.53)
    return m


def build(kind: str, sight: str, name: str = "") -> Model:
    nm = name.upper()[:18]
    if kind in ("pistol", "revolver", "raygun"):
        return _pistol(kind, nm)
    if kind in ("shotgun", "shotgun2"):
        return _shotgun(kind, sight)
    if kind == "launcher":
        return _launcher(nm)
    return _rifle(kind if kind in ("ar", "ar2", "smg", "smg2", "dmr", "lmg", "sniper") else "ar", sight, nm)
