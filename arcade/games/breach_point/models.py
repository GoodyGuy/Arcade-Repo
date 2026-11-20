"""Everything in BREACH POINT that is not the map is built from boxes: the ten agents (posed from their walk, crouch,
aim and death), the guns in their hands and in the player's own view, the charge, sensors and thrown things. A box is
one row of 16 numbers for the renderer's instanced cube: centre, three half-axes, colour, and a flag (0 lit, 1 glowing).
The same gun parts draw the flat icons of the HUD and the buy menu. Coordinates here are the renderer's: (x, up, y)."""

from __future__ import annotations

import math

from .config import WEAPONS

DARK = (0.13, 0.14, 0.17)
STEEL = (0.3, 0.32, 0.36)
POLY = (0.2, 0.21, 0.24)
WOOD = (0.5, 0.34, 0.2)
SUIT = (0.2, 0.22, 0.27)
SUIT2 = (0.28, 0.3, 0.36)
SKIN = (0.86, 0.66, 0.52)
BOOT = (0.1, 0.1, 0.12)
GUN_ACCENT = {"tack": (0.75, 0.75, 0.72), "vesper": (0.35, 0.8, 0.75), "maul": (0.85, 0.6, 0.25),
              "wasp": (0.95, 0.8, 0.2), "lynx": (0.4, 0.75, 0.45), "brace": (0.8, 0.35, 0.25),
              "lancer": (0.55, 0.7, 0.9), "sentry": (0.9, 0.55, 0.2), "kestrel": (0.9, 0.3, 0.3),
              "mistral": (0.35, 0.6, 0.95), "bastion": (0.6, 0.62, 0.3), "longbow": (0.7, 0.45, 0.9),
              "knife": (0.8, 0.82, 0.85)}


def _parts(wid: str) -> list:
    """(centre right, up, forward, half sizes, colour) in the gun's own frame: the grip at the origin, metres."""
    w = WEAPONS[wid]
    acc = GUN_ACCENT.get(wid, STEEL)
    c = w.cls
    if c == "melee":
        return [((0, 0.0, 0.03), (0.013, 0.016, 0.055), POLY), ((0, 0.005, 0.17), (0.004, 0.02, 0.09), acc),
                ((0, 0.0, 0.085), (0.016, 0.026, 0.008), STEEL)]
    if c == "sidearm":
        big = wid == "maul"
        ln = 0.11 if big else 0.085
        out = [((0, 0.055, 0.04 + ln * 0.4), (0.015, 0.02 if not big else 0.024, ln), DARK if not big else STEEL),
               ((0, -0.02, 0.0), (0.014, 0.055, 0.02), POLY), ((0, 0.03, 0.03), (0.016, 0.006, 0.05), acc)]
        if wid == "vesper":
            out.append(((0, 0.055, 0.04 + ln * 1.4 + 0.05), (0.014, 0.014, 0.06), DARK))
        if big:
            out.append(((0, 0.04, 0.03), (0.022, 0.022, 0.03), DARK))
        return out
    if c == "smg":
        out = [((0, 0.05, 0.12), (0.022, 0.035, 0.17), DARK), ((0, -0.03, 0.0), (0.015, 0.05, 0.022), POLY),
               ((0, -0.07, 0.13), (0.013, 0.09, 0.022), STEEL), ((0, 0.055, 0.36), (0.012, 0.012, 0.08), STEEL),
               ((0, 0.05, -0.14), (0.014, 0.022, 0.1), POLY), ((0, 0.092, 0.1), (0.008, 0.008, 0.06), acc)]
        if wid == "lynx":
            out.append(((0, 0.055, 0.46), (0.018, 0.018, 0.08), DARK))
        return out
    if c == "shotgun":
        return [((0, 0.05, 0.1), (0.024, 0.036, 0.18), DARK), ((0, 0.06, 0.48), (0.014, 0.014, 0.22), STEEL),
                ((0, 0.025, 0.4), (0.022, 0.02, 0.1), WOOD), ((0, -0.03, 0.0), (0.015, 0.05, 0.022), WOOD),
                ((0, 0.03, -0.2), (0.016, 0.034, 0.13), WOOD), ((0, 0.09, 0.05), (0.008, 0.006, 0.05), acc)]
    if c in ("rifle", "marksman"):
        long = c == "marksman"
        out = [((0, 0.05, 0.1), (0.024, 0.04, 0.2), DARK), ((0, 0.05, 0.4), (0.022, 0.03, 0.12), POLY),
               ((0, 0.055, 0.62 if not long else 0.7), (0.01, 0.01, 0.12 if not long else 0.2), STEEL),
               ((0, -0.03, 0.0), (0.015, 0.05, 0.022), POLY), ((0, 0.04, -0.2), (0.016, 0.036, 0.13), POLY),
               ((0.0, -0.085, 0.16), (0.013, 0.085, 0.03), STEEL), ((0, 0.1, 0.1), (0.008, 0.012, 0.07), acc)]
        if wid == "mistral":
            out.append(((0, 0.055, 0.78), (0.02, 0.02, 0.1), DARK))
        if long:
            out.append(((0, 0.125, 0.14), (0.016, 0.016, 0.09), DARK))
        return out
    if c == "sniper":
        heavy = wid == "longbow"
        return [((0, 0.05, 0.08), (0.024, 0.04, 0.22), POLY if heavy else WOOD),
                ((0, 0.058, 0.62), (0.012 if heavy else 0.009, 0.012 if heavy else 0.009, 0.36), STEEL),
                ((0, -0.03, 0.0), (0.015, 0.05, 0.022), POLY),
                ((0, 0.035, -0.24), (0.018, 0.045, 0.15), POLY if heavy else WOOD),
                ((0, 0.125, 0.1), (0.02, 0.02, 0.14), DARK), ((0, 0.125, 0.25), (0.024, 0.024, 0.02), acc),
                ((0, -0.06, 0.14), (0.013, 0.04, 0.035), STEEL)] + (
                    [((0, 0.058, 1.0), (0.02, 0.018, 0.04), DARK)] if heavy else [])
    # lmg
    return [((0, 0.05, 0.12), (0.03, 0.05, 0.24), DARK), ((0, 0.055, 0.62), (0.013, 0.013, 0.28), STEEL),
            ((0, -0.03, 0.0), (0.015, 0.05, 0.022), POLY), ((0, 0.04, -0.24), (0.018, 0.04, 0.13), POLY),
            ((0.0, -0.08, 0.16), (0.05, 0.06, 0.07), acc), ((0, 0.11, 0.05), (0.01, 0.01, 0.06), STEEL),
            ((0, 0.0, 0.5), (0.008, 0.05, 0.008), STEEL)]


_CACHE: dict[str, list] = {}


def gun_parts(wid: str) -> list:
    got = _CACHE.get(wid)
    if got is None:
        got = _CACHE[wid] = _parts(wid)
    return got


def add_gun(rows: list, o: tuple, R: tuple, U: tuple, F: tuple, wid: str, s: float = 1.0) -> None:
    """The gun ``wid`` with its grip at ``o``, pointing along F."""
    ox, oy, oz = o
    for (cx, cy, cz), (hx, hy, hz), col in gun_parts(wid):
        cx *= s
        cy *= s
        cz *= s
        hx *= s
        hy *= s
        hz *= s
        rows.append((ox + R[0] * cx + U[0] * cy + F[0] * cz, oy + R[1] * cx + U[1] * cy + F[1] * cz,
                     oz + R[2] * cx + U[2] * cy + F[2] * cz,
                     R[0] * hx, R[1] * hx, R[2] * hx, U[0] * hy, U[1] * hy, U[2] * hy,
                     F[0] * hz, F[1] * hz, F[2] * hz, col[0], col[1], col[2], 0.0))


def _box(rows: list, c: tuple, R: tuple, U: tuple, F: tuple, hx: float, hy: float, hz: float, col: tuple,
         flag: float = 0.0) -> None:
    rows.append((c[0], c[1], c[2], R[0] * hx, R[1] * hx, R[2] * hx, U[0] * hy, U[1] * hy, U[2] * hy,
                 F[0] * hz, F[1] * hz, F[2] * hz, col[0], col[1], col[2], flag))


def _seg(rows: list, p0: tuple, p1: tuple, w: float, t: float, ref: tuple, col: tuple) -> None:
    """A box from p0 to p1, ``w`` wide across ``ref``'s side."""
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    ln = math.sqrt(dx * dx + dy * dy + dz * dz) or 1e-6
    dx, dy, dz = dx / ln, dy / ln, dz / ln
    sx, sy, sz = dy * ref[2] - dz * ref[1], dz * ref[0] - dx * ref[2], dx * ref[1] - dy * ref[0]
    sn = math.sqrt(sx * sx + sy * sy + sz * sz)
    if sn < 1e-4:
        sx, sy, sz, sn = 1.0, 0.0, 0.0, 1.0
    sx, sy, sz = sx / sn, sy / sn, sz / sn
    tx, ty, tz = dy * sz - dz * sy, dz * sx - dx * sz, dx * sy - dy * sx  # the same hand as the other boxes
    h = ln / 2
    rows.append(((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2, sx * w, sy * w, sz * w,
                 dx * h, dy * h, dz * h, tx * t, ty * t, tz * t, col[0], col[1], col[2], 0.0))


def _at(p: tuple, a: tuple, s: float, b: tuple = (0, 0, 0), t: float = 0.0, c: tuple = (0, 0, 0),
        u: float = 0.0) -> tuple:
    return (p[0] + a[0] * s + b[0] * t + c[0] * u, p[1] + a[1] * s + b[1] * t + c[1] * u,
            p[2] + a[2] * s + b[2] * t + c[2] * u)


def add_character(rows: list, a, team_col: tuple, t: float) -> None:  # noqa: ANN001 - sim.Actor
    """The boxes of one agent, alive (walking, crouched, aiming) or down."""
    yaw = a.yaw
    ca, sa = math.cos(yaw), math.sin(yaw)
    f = (ca, 0.0, sa)
    r = (-sa, 0.0, ca)
    up = (0.0, 1.0, 0.0)
    base = (a.x, a.z, a.y)
    k = a.crouch
    col = tuple(c / 255 for c in a.agent.color)
    dead = not a.alive
    if dead:
        th = min(a.dead_t / 0.35, 1.0) * 1.5
        ct, st = math.cos(th), math.sin(th)
        up, f = (-f[0] * st, ct, -f[2] * st), (f[0] * ct, st, f[2] * ct)
        base = (base[0], base[1] + 0.12 * min(a.dead_t / 0.35, 1.0), base[2])
        k = 0.0
    sp = 0.0 if dead else min(a.speed() / 5.0, 1.0)
    sw = math.sin(a.phase) * 0.62 * sp
    bend = 0.95 * k
    hh = 0.46 * (math.cos(bend) + math.cos(-bend)) * (1.0 - 0.02 * sp)
    hip = _at(base, up, hh)
    for side, s in ((1.0, sw), (-1.0, -sw)):
        A = s + bend
        B = A - 2.0 * bend - max(-s, 0.0) * 0.9
        pv = _at(hip, r, 0.1 * side)
        d1 = (-up[0] * math.cos(A) + f[0] * math.sin(A), -up[1] * math.cos(A) + f[1] * math.sin(A),
              -up[2] * math.cos(A) + f[2] * math.sin(A))
        knee = _at(pv, d1, 0.46)
        d2 = (-up[0] * math.cos(B) + f[0] * math.sin(B), -up[1] * math.cos(B) + f[1] * math.sin(B),
              -up[2] * math.cos(B) + f[2] * math.sin(B))
        foot = _at(knee, d2, 0.44)
        _seg(rows, pv, knee, 0.085, 0.09, r, SUIT)
        _seg(rows, knee, foot, 0.07, 0.075, r, SUIT2)
        _box(rows, _at(foot, f, 0.05, up, 0.03), r, up, f, 0.06, 0.04, 0.12, BOOT)
    lean = 0.06 + 0.3 * k
    cl, sl = math.cos(lean), math.sin(lean)
    ut = (up[0] * cl + f[0] * sl, up[1] * cl + f[1] * sl, up[2] * cl + f[2] * sl)
    ft = (f[0] * cl - up[0] * sl, f[1] * cl - up[1] * sl, f[2] * cl - up[2] * sl)
    _box(rows, _at(hip, ut, 0.05), r, ut, ft, 0.17, 0.09, 0.11, SUIT)
    chest = _at(hip, ut, 0.36)
    wide = 0.235 if a.agent.id == "bulwark" else 0.205
    _box(rows, chest, r, ut, ft, wide, 0.23, 0.125, SUIT2)
    _box(rows, _at(chest, ut, 0.03, ft, 0.115), r, ut, ft, wide * 0.8, 0.15, 0.025, col)  # the chest plate
    _box(rows, _at(chest, ut, 0.18, r, wide + 0.03), r, ut, ft, 0.06, 0.05, 0.09, team_col)
    _box(rows, _at(chest, ut, 0.18, r, -wide - 0.03), r, ut, ft, 0.06, 0.05, 0.09, team_col)
    pitch = 0.0 if dead else max(min(a.pitch + a.rec_p * 0.0175, 1.2), -1.2)
    cp, spt = math.cos(pitch * 0.7), math.sin(pitch * 0.7)
    fh = (ft[0] * cp + ut[0] * spt, ft[1] * cp + ut[1] * spt, ft[2] * cp + ut[2] * spt)
    uh = (ut[0] * cp - ft[0] * spt, ut[1] * cp - ft[1] * spt, ut[2] * cp - ft[2] * spt)
    neck = _at(chest, ut, 0.25)
    head = _at(neck, uh, 0.13)
    _box(rows, head, r, uh, fh, 0.105, 0.12, 0.115, SKIN)
    ag = a.agent.id
    if ag == "murk":  # a hood
        _box(rows, _at(head, uh, 0.03, fh, -0.03), r, uh, fh, 0.125, 0.125, 0.125, col)
    else:
        _box(rows, _at(head, uh, 0.09, fh, -0.01), r, uh, fh, 0.112, 0.045, 0.122, SUIT if ag != "bulwark" else col)
    if ag in ("echo", "fuse"):  # a visor
        _box(rows, _at(head, uh, 0.02, fh, 0.11), r, uh, fh, 0.1, 0.028, 0.012, col, 1.0 if ag == "echo" else 0.0)
    if ag == "echo":
        _box(rows, _at(chest, ut, 0.42, r, -0.16, ft, -0.13), r, ut, ft, 0.008, 0.22, 0.008, STEEL)
    elif ag == "kite":
        _box(rows, _at(neck, ut, 0.0, ft, -0.1), r, ut, ft, 0.12, 0.05, 0.08, col)
        _box(rows, _at(neck, ut, -0.2, ft, -0.17, r, 0.06), r, ut, ft, 0.04, 0.18, 0.012, col)
    elif ag == "fuse":
        _box(rows, _at(chest, ut, 0.0, ft, -0.19), r, ut, ft, 0.14, 0.17, 0.07, (0.3, 0.27, 0.2))
    elif ag == "bulwark":
        _box(rows, _at(chest, ut, 0.21, r, 0.27), r, ut, ft, 0.085, 0.06, 0.11, col)
        _box(rows, _at(chest, ut, 0.21, r, -0.27), r, ut, ft, 0.085, 0.06, 0.11, col)
    # arms and the gun
    sr = _at(chest, ut, 0.17, r, wide + 0.02)
    sl_ = _at(chest, ut, 0.17, r, -wide - 0.02)
    if dead:
        _seg(rows, sr, _at(sr, r, 0.3, ut, -0.35), 0.05, 0.05, ft, SUIT)
        _seg(rows, sl_, _at(sl_, r, -0.3, ut, -0.35), 0.05, 0.05, ft, SUIT)
        return
    cpp, spp = math.cos(pitch), math.sin(pitch)
    ad = (f[0] * cpp + up[0] * spp, f[1] * cpp + up[1] * spp, f[2] * cpp + up[2] * spp)
    au = (up[0] * cpp - f[0] * spp, up[1] * cpp - f[1] * spp, up[2] * cpp - f[2] * spp)
    wid = a.wid
    low = a.reload_t > 0 or a.plant_t > 0 or a.defusing or a.equip_t > 0.1
    grip = _at(chest, ad, 0.3, au, 0.02 - (0.16 if low else 0.0), r, 0.12)
    w = WEAPONS[wid]
    reach = 0.1 if w.cls in ("sidearm", "melee") else 0.3
    fore = _at(grip, ad, reach, au, 0.0 if w.cls in ("sidearm", "melee") else 0.02)
    er = _at(sr, ad, 0.12, up, -0.2, r, 0.03)
    el = _at(sl_, ad, 0.2, up, -0.2, r, -0.02)
    _seg(rows, sr, er, 0.05, 0.05, r, SUIT)
    _seg(rows, er, grip, 0.042, 0.042, r, SUIT2)
    _seg(rows, sl_, el, 0.05, 0.05, r, SUIT)
    _seg(rows, el, fore, 0.042, 0.042, r, SUIT2)
    _box(rows, grip, r, au, ad, 0.035, 0.04, 0.04, SKIN)
    _box(rows, fore, r, au, ad, 0.035, 0.035, 0.04, SKIN)
    if a.plant_t > 0 or a.defusing:
        return
    add_gun(rows, grip, r, au, ad, wid)
    if a.has_charge:  # the charge on its back
        _box(rows, _at(chest, ut, 0.0, ft, -0.17), r, ut, ft, 0.1, 0.13, 0.05, (0.16, 0.17, 0.2))
        _box(rows, _at(chest, ut, 0.06, ft, -0.225), r, ut, ft, 0.03, 0.03, 0.01, (1.0, 0.75, 0.15), 1.0)


def add_viewmodel(rows: list, cam: tuple, R: tuple, U: tuple, F: tuple, a, t: float, sway: tuple,  # noqa: ANN001
                  fov_k: float) -> tuple | None:
    """The watched agent's own hands and gun in front of the eye. Gives the muzzle's place (for the flash)."""
    w = a.weapon
    wid = a.wid
    if a.plant_t > 0 or a.defusing:
        # both hands down on the charge
        base = _at(cam, F, 0.5, U, -0.085)
        bob = math.sin(t * 9.0) * 0.012
        _box(rows, _at(base, U, bob - 0.05), R, U, F, 0.11, 0.035, 0.07, (0.16, 0.17, 0.2))
        _box(rows, _at(base, U, bob - 0.012, R, 0.04), R, U, F, 0.022, 0.006, 0.022,
             (1.0, 0.75, 0.15) if int(t * 6) % 2 else (0.3, 0.2, 0.05), 1.0)
        for s in (1.0, -1.0):
            _seg(rows, _at(cam, R, 0.32 * s, U, -0.42, F, 0.2), _at(base, R, 0.1 * s, U, bob), 0.035, 0.035, U, SUIT2)
        return None
    ads = a.ads
    scoped_in = w.scoped and ads > 0.6
    if scoped_in:
        return None
    sp = min(a.speed() / 5.5, 1.0) if a.on_ground else 0.0
    bobx = math.sin(a.phase) * 0.012 * sp
    boby = abs(math.cos(a.phase)) * 0.012 * sp
    kick = a.kick
    drop = 0.0
    roll = 0.0
    if a.equip_t > 0:
        drop = min(a.equip_t / max(w.equip, 0.2), 1.0) * 0.3
    if a.reload_t > 0:
        ph = 1.0 - a.reload_t / max(w.reload, 0.1)
        drop = max(drop, math.sin(min(ph, 1.0) * math.pi) * 0.16)
        roll = math.sin(min(ph, 1.0) * math.pi) * 0.5
    small = w.cls in ("sidearm", "melee")
    side = (0.17 if small else 0.2) * (1 - ads * 0.92) + bobx + sway[0] * 0.02
    down = (0.15 if small else 0.175) * (1 - ads * 0.42) + boby + drop - sway[1] * 0.02
    fwd = (0.5 if small else 0.52) - kick * 0.05 * (1.6 if w.cls in ("sniper", "shotgun") else 1.0) + ads * 0.04
    o = _at(cam, R, side, U, -down, F, fwd)
    pk = kick * 0.06 + roll * 0.35
    cpk, spk = math.cos(pk), math.sin(pk)
    cr, sr = math.cos(roll), math.sin(roll)
    F2 = (F[0] * cpk + U[0] * spk, F[1] * cpk + U[1] * spk, F[2] * cpk + U[2] * spk)
    U2 = (U[0] * cpk - F[0] * spk, U[1] * cpk - F[1] * spk, U[2] * cpk - F[2] * spk)
    R2 = (R[0] * cr + U2[0] * sr, R[1] * cr + U2[1] * sr, R[2] * cr + U2[2] * sr)
    U3 = (U2[0] * cr - R[0] * sr, U2[1] * cr - R[1] * sr, U2[2] * cr - R[2] * sr)
    if w.cls == "melee":
        sw = kick
        o = _at(o, F, sw * 0.18, R, -sw * 0.12)
    vs = 0.74  # the gun in the hand is drawn a little smaller than life: it sits close to the eye
    add_gun(rows, o, R2, U3, F2, wid, vs)
    shoulder = _at(cam, R, 0.36, U, -0.42, F, 0.1)
    _seg(rows, shoulder, _at(o, U3, -0.03), 0.032, 0.036, U, SUIT2)
    _box(rows, _at(o, U3, -0.012), R2, U3, F2, 0.021, 0.034, 0.026, SKIN)
    if w.cls not in ("sidearm", "melee"):
        fg = _at(o, F2, 0.3 * vs, U3, 0.008)
        _seg(rows, _at(cam, R, -0.34, U, -0.42, F, 0.12), fg, 0.032, 0.036, U, SUIT2)
        _box(rows, fg, R2, U3, F2, 0.026, 0.023, 0.034, SKIN)
    tip = max((cz + hz) for (cx, cy, cz), (hx, hy, hz), _c in gun_parts(wid))
    return _at(o, F2, (tip + 0.03) * vs, U3, 0.055 * vs)


def add_charge(rows: list, x: float, y: float, z: float, t: float, armed: bool, defusing: bool) -> None:
    R, U, F = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    _box(rows, (x, z + 0.09, y), R, U, F, 0.2, 0.09, 0.13, (0.16, 0.17, 0.2))
    _box(rows, (x, z + 0.2, y), R, U, F, 0.12, 0.03, 0.07, (0.26, 0.27, 0.3))
    on = int(t * (6 if armed else 1.5)) % 2 == 0
    col = (0.3, 0.7, 1.0) if defusing else ((1.0, 0.75, 0.15) if on else (0.35, 0.22, 0.05))
    _box(rows, (x + 0.1, z + 0.24, y), R, U, F, 0.03, 0.02, 0.03, col, 1.0)
    _box(rows, (x - 0.12, z + 0.3, y - 0.05), R, U, F, 0.008, 0.1, 0.008, (0.5, 0.5, 0.55))


def add_small(rows: list, x: float, y: float, z: float, s: float, col: tuple, spin: float = 0.0,
              glow: bool = True) -> None:
    c, sn = math.cos(spin), math.sin(spin)
    _box(rows, (x, z, y), (c, 0.0, sn), (0.0, 1.0, 0.0), (-sn, 0.0, c), s, s, s, col, 1.0 if glow else 0.0)


def add_block(rows: list, x0: float, y0: float, x1: float, y1: float, z0: float, z1: float, col: tuple,
              flag: float = 0.0) -> None:
    _box(rows, ((x0 + x1) / 2, (z0 + z1) / 2, (y0 + y1) / 2), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
         (x1 - x0) / 2, (z1 - z0) / 2, (y1 - y0) / 2, col, flag)
