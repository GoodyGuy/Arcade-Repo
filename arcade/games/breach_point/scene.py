"""Turns the state of a match into what the renderer draws this frame: the agents' boxes (with the outline for the
watcher's enemies and the see-through copy of marked ones), the watcher's own hands and gun, smoke, ground effects,
the charge, things thrown and dropped, barricades, tracers and glows."""

from __future__ import annotations

import math

from . import config as C
from .gfx import Frame
from .models import add_block, add_character, add_charge, add_gun, add_small, add_viewmodel

ALLY = tuple(c / 255 for c in C.ALLY)
ENEMY = tuple(c / 255 for c in C.ENEMY)
GREY = (0.78, 0.8, 0.82)


def view_fov(base_deg: float, a) -> float:  # noqa: ANN001 - sim.Actor
    zoom = 1.0 + (a.weapon.zoom - 1.0) * a.ads
    return 2 * math.atan(math.tan(math.radians(base_deg) / 2) / zoom)


def build_frame(m, viewer, r, t: float, eye_z: float, sway: tuple, first_person: bool = True) -> Frame:  # noqa: ANN001
    """``m``: sim.Match, ``viewer``: the actor watched, ``r``: the Renderer (its camera already set)."""
    fr = Frame()
    boxes, decals, glows = fr.boxes, fr.decals, fr.glows
    mp = m.map
    squad = viewer.squad
    cam = (float(r.cam[0]), float(r.cam[1]), float(r.cam[2]))
    R, U, F = tuple(map(float, r.R)), tuple(map(float, r.U)), tuple(map(float, r.F))
    muzzle = None
    for a in m.actors:
        if a is viewer and first_person and a.alive:
            continue
        foe = a.squad != squad
        n0 = len(boxes)
        add_character(boxes, a, ENEMY if foe else ALLY, t)
        if a.alive:
            decals.append((a.x, a.z + 0.03, a.y, 0.5, 0.0, 0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 0.0))
            if foe:
                fr.outline.extend(boxes[n0:])
                if a.revealed > 0:
                    fr.xray.extend(boxes[n0:])
            if a.since_shot < 0.05 and not a.weapon.quiet and a.weapon.cls != "melee":
                yaw, pitch = a.view()
                glows.append((a.x + math.cos(yaw) * 0.9, a.eye - 0.12 + math.sin(pitch) * 0.9,
                              a.y + math.sin(yaw) * 0.9, 0.42, 1.0, 0.8, 0.45, 1.3))
    if viewer.alive and first_person:
        muzzle = add_viewmodel(fr.view, cam, R, U, F, viewer, t, sway, 1.0)
        if muzzle is not None:  # where that point is on screen, through the world's lens
            k = math.tan(r.fov / 2) / math.tan(r.vm_fov / 2)
            d = (muzzle[0] - cam[0], muzzle[1] - cam[1], muzzle[2] - cam[2])
            dx = (d[0] * R[0] + d[1] * R[1] + d[2] * R[2]) * k
            dy = (d[0] * U[0] + d[1] * U[1] + d[2] * U[2]) * k
            dz = d[0] * F[0] + d[1] * F[1] + d[2] * F[2]
            muzzle = (cam[0] + R[0] * dx + U[0] * dy + F[0] * dz, cam[1] + R[1] * dx + U[1] * dy + F[1] * dz,
                      cam[2] + R[2] * dx + U[2] * dy + F[2] * dz)
        if muzzle is not None and viewer.since_shot < 0.045 and viewer.weapon.cls != "melee":
            k = 0.5 if viewer.weapon.quiet else 1.0
            glows.append((muzzle[0], muzzle[1], muzzle[2], 0.075 * k + 0.035, 1.0, 0.82, 0.5, 1.3 * k))
    # the charge
    ch = m.charge
    st = ch["state"]
    if st == "dropped":
        z = mp.ground(ch["x"], ch["y"], 2.0)
        add_charge(boxes, ch["x"], ch["y"], z, t, False, False)
        glows.append((ch["x"], z + 0.5, ch["y"], 0.5, 1.0, 0.75, 0.2, 0.5 + 0.3 * math.sin(t * 5)))
    elif st in ("planted", "defused"):
        add_charge(boxes, ch["x"], ch["y"], ch["z"], t, st == "planted", ch.get("defuser") is not None)
        if st == "planted":
            left = ch["t"]
            rate = 1.0 if left > 20 else (2.0 if left > 10 else (4.0 if left > 5 else 7.0))
            pulse = (t * rate) % 1.0
            col = (0.3, 0.7, 1.0) if ch.get("defuser") is not None else (1.0, 0.72, 0.15)
            glows.append((ch["x"] + 0.1, ch["z"] + 0.3, ch["y"], 0.35 + 0.5 * (1 - pulse), *col, 1.2 * (1 - pulse)))
            decals.append((ch["x"], ch["z"] + 0.04, ch["y"], 0.4 + pulse * 2.6, *col, 0.7 * (1 - pulse),
                           1.0, 0.92, 0.0, 0.0))
    elif st == "boom":
        bt = ch.get("boom_t", 0.0)
        k = min(bt / 0.9, 1.0)
        rad = C.BLAST_RADIUS * (0.15 + 0.85 * k)
        fr.shells.append((ch["x"], ch["z"] + 0.5, ch["y"], rad, 1.0, 0.75, 0.3, -(1.0 - k * 0.6)))
        fr.shells.append((ch["x"], ch["z"] + 0.5, ch["y"], rad * 0.6, 1.0, 0.95, 0.8, -(1.0 - k)))
        glows.append((ch["x"], ch["z"] + 1.5, ch["y"], 6.0 + 16.0 * k, 1.0, 0.7, 0.3, 2.0 * (1.0 - k)))
    for d in m.drops:
        z = mp.ground(d[0], d[1], 2.0) + 0.04
        add_gun(boxes, (d[0], z, d[1]), (0.0, 1.0, 0.0), (0.7, 0.0, 0.7), (0.7, 0.0, -0.7), d[2])
    for pr in m.projectiles:
        col = tuple(c / 255 for c in pr["color"])
        add_small(boxes, pr["x"], pr["y"], pr["z"], 0.07, col, pr["age"] * 9.0)
        glows.append((pr["x"], pr["z"], pr["y"], 0.3, *col, 0.8))
    for s in m.smokes:
        age = s[4]
        if age < 0:
            continue
        k = min(age / 0.5, 1.0) * min((s[5] - age) / 0.8, 1.0)
        tint = tuple(0.82 * g + 0.18 * c / 255 for g, c in zip(GREY, s[7], strict=True))
        fr.smokes.append((s[0], s[2], s[1], s[3] * (0.25 + 0.75 * k), *tint, (s[0] * 7.1 + s[1] * 3.3) % 6.28))
    for zn in m.zones:
        kind = zn["kind"]
        fade = min(zn["t"] / 0.6, 1.0) * min((zn["dur"] - zn["t"]) / 0.3, 1.0)
        col = tuple(c / 255 for c in zn["color"])
        z = zn["z"] + 0.05
        if kind == "fire":
            decals.append((zn["x"], z, zn["y"], zn["r"], 1.0, 0.5, 0.1, fade, 3.0, 0.0, 0.0, 0.0))
            glows.append((zn["x"], z + 0.6, zn["y"], zn["r"] * 1.3, 1.0, 0.45, 0.1, 0.35 * fade))
        elif kind == "static":
            decals.append((zn["x"], z, zn["y"], zn["r"], *col, fade, 4.0, 0.0, 0.0, 0.0))
        else:
            decals.append((zn["x"], z, zn["y"], zn["r"], *col, fade * (0.8 if kind == "slow" else 0.55), 2.0, 0.0,
                           0.0, 0.0))
            if kind == "stronghold":
                fr.shells.append((zn["x"], zn["z"], zn["y"], zn["r"], *col, -0.3 * fade))
    for s in m.sensors:
        mine = s["squad"] == squad
        col = ALLY if mine else ENEMY
        add_small(boxes, s["x"], s["y"], s["z"] + 0.12, 0.08, (0.2, 0.22, 0.26), 0.0, False)
        add_small(boxes, s["x"], s["y"], s["z"] + 0.26, 0.035, col, t * 2.0)
        glows.append((s["x"], s["z"] + 0.3, s["y"], 0.3, *col, 0.6 + 0.3 * math.sin(t * 4)))
        if mine:
            decals.append((s["x"], s["z"] + 0.04, s["y"], s["r"], *col, 0.5, 1.0, 0.985, 0.0, 0.0))
    for pg in m.pings:
        col = ALLY if pg["squad"] == squad else ENEMY
        add_small(boxes, pg["x"], pg["y"], pg["z"] + 0.1, 0.08, col, t * 3.0)
        if pg["pulse"] < 0.9:
            w = pg["pulse"] / 0.9
            decals.append((pg["x"], mp.ground(pg["x"], pg["y"], 2.0) + 0.05, pg["y"], pg["r"], *col, 0.9, 1.0, w, 0.0,
                           0.0))
    for b in m.barricades:
        rise = min(b[3] / 0.3, 1.0) * min(b[0] / 0.3, 1.0)
        for cx, cy in b[2]:
            add_block(boxes, cx + 0.04, cy + 0.04, cx + 0.96, cy + 0.96, 0.0, 2.7 * rise, (0.3, 0.52, 0.9))
            add_block(boxes, cx + 0.02, cy + 0.02, cx + 0.98, cy + 0.98, 2.7 * rise - 0.08, 2.7 * rise + 0.02,
                      (0.55, 0.8, 1.0), 1.0)
    for pf in m.puffs:
        x, y, z, age, kind = pf
        if kind == "dust":
            if age < 0.12:
                glows.append((x, z, y, 0.1 + age * 0.8, 1.0, 0.85, 0.6, 1.0 - age / 0.12))
        elif kind == "blood":
            if age < 0.16:
                glows.append((x, z, y, 0.16 + age, 1.0, 0.25, 0.4, 1.2 * (1.0 - age / 0.16)))
        elif kind == "flash":
            if age < 0.3:
                glows.append((x, z, y, 3.0 + age * 12, 1.0, 1.0, 0.95, 2.0 * (1.0 - age / 0.3)))
        elif kind == "blast":
            k = age / 0.7
            glows.append((x, z + 0.6, y, 1.5 + k * 4.5, 1.0, 0.6, 0.2, 2.0 * (1.0 - k)))
            decals.append((x, mp.ground(x, y, 2.0) + 0.05, y, 5.0, 1.0, 0.6, 0.2, 0.9 * (1.0 - k), 1.0, min(k * 1.3, 1.0),
                           0.0, 0.0))
            if k < 0.5:
                fr.shells.append((x, z, y, 1.0 + k * 6.0, 1.0, 0.7, 0.3, -(1.0 - k * 2.0)))
        elif kind == "shock":
            k = age / 0.7
            decals.append((x, mp.ground(x, y, 2.0) + 0.05, y, 4.6, 1.0, 0.9, 0.4, 0.9 * (1.0 - k), 1.0, min(k * 1.4, 1.0),
                           0.0, 0.0))
    for tr in m.tracers:
        x0, y0, z0, x1, y1, z1, age, who = tr
        if who == viewer.idx and muzzle is not None:
            a0 = muzzle
        else:
            a0 = (x0, z0 - 0.12, y0)
        k = 1.0 - age / 0.12
        fr.beams.append((*a0, x1, z1, y1, 1.0 * k, 0.86 * k, 0.5 * k, 0.012))
    for X, Y, Z, col, _rad in r.lamps:
        add_block(boxes, X - 0.18, Z - 0.18, X + 0.18, Z + 0.18, Y + 0.3, Y + 0.42, col, 1.0)
        glows.append((X, Y + 0.25, Z, 0.7, col[0], col[1], col[2], 0.5))
    fr.barriers = m.phase == "buy"
    return fr
