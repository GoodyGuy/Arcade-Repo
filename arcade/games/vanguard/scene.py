"""Turns a match into what the renderer draws this frame: rows for the solid things (heroes, the cart, gadgets,
shots in flight), rows for the see-through and glowing ones (barriers, rings, tracers, blasts), glows, lights and
the player's own weapon."""

from __future__ import annotations

import math

import numpy as np

from . import models as M
from .data import ALLY_RGB, CART_RADIUS, ENEMY_RGB, GOLD
from .maps import track_at

F32 = np.float32
_EMPTY = np.zeros((0, M.ROW), F32)
VM_FOV = 1.15


def _line(a: tuple, b: tuple, thick: float, col: tuple, alpha: float, emis: float = 1.0) -> list:
    """A row for a glowing line from map point ``a`` to ``b`` (x, y, z)."""
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    ln = math.sqrt(dx * dx + dy * dy + dz * dz) or 1e-6
    return M.inst((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2, thick, thick, ln, col,
                  math.atan2(dy, dx), math.asin(max(min(dz / ln, 1.0), -1.0)), emis=emis, rad=thick / 2, alpha=alpha,
                  pat=3)


class Frame:
    __slots__ = ("add", "alpha", "casters", "glows", "lights", "rows", "vm")


def build(m, viewer, t: float, renderer, first_person: bool = True, sway: tuple = (0.0, 0.0)) -> Frame:  # noqa: ANN001
    """Everything of match ``m`` as ``viewer`` sees it at time ``t``."""
    rows: list = []
    own = None
    alpha: list = []
    add: list = []
    glows: list = []
    lights: list = []
    vteam = viewer.team
    for a in m.actors:
        mine = a.team == vteam
        rgb = ALLY_RGB if mine else ENEMY_RGB
        model = M.hero_model(a.hero.id, a.hero.scale)
        if a.alive:
            fx = a.fx
            ang, bob = M.actor_pose(a.walk, a.move_amt, a.pitch, a.kick, not a.on_ground, 1.0 if "stun" in fx else 0.0,
                                    t + a.idx * 1.3, max(a.cast_anim, a.melee_anim * 2.0))
            flag = 1.0 if mine else (3.0 if "reveal" in fx else 2.0)
            r = model.rows((a.x, a.z, a.y), M.yaw_basis(a.yaw), ang, bob, rgb, flag, a.flash / 0.14)
            if a is viewer and first_person:
                own = r
            else:
                rows.append(r)
            cz = a.z + a.height * 0.6
            if "burn" in fx:
                glows.append((a.x + math.sin(t * 9 + a.idx) * 0.3, a.y + math.cos(t * 7) * 0.3, cz + 0.3 *
                              math.sin(t * 11), 0.7, 255, 120, 30, 0.8))
            if "mark" in fx:
                alpha.append(M.inst(a.x, a.y, a.z + a.height + 0.55, 0.34, 0.34, 0.34, (140, 255, 220), t * 3.0,
                                    0.6, 0.8, emis=0.1, rad=0.02, alpha=0.9))
            if "boost" in fx:
                glows.append((a.x, a.y, a.z + a.height * 0.75, 0.9, 255, 110, 220, 0.5))
            if m.time - a.healed_t < 0.4 and a is not viewer:
                glows.append((a.x, a.y, cz, 1.0, 110, 255, 170, 0.35))
            if "sink" in fx and not (a is viewer and first_person):
                k = min(a.sink_store / 150.0, 1.0)
                alpha.append(M.inst(a.x, a.y, cz, 2.3, 2.6, 2.3, (255, 150 - 60 * k, 50), emis=0.4 + k, rad=9.0,
                                    alpha=0.3 + 0.3 * k, pat=1))
            if "melt" in fx:
                alpha.append(M.inst(a.x, a.y, a.z + 0.06, 18.0, 0.04, 18.0, (255, 110, 30), mode=1, alpha=0.55, pat=2))
                glows.append((a.x, a.y, cz, 2.4, 255, 120, 30, 0.9))
                lights.append((a.x, a.y, cz, (0.5, 0.22, 0.06), 9.0))
            if "monsoon" in fx:
                alpha.append(M.inst(a.x, a.y, a.z + 0.06, 20.0, 0.04, 20.0, (120, 240, 210), mode=1, alpha=0.8,
                                    pat=2))
            if "bulldoze" in fx:
                cy, sy = math.cos(a.yaw), math.sin(a.yaw)
                alpha.append(M.inst(a.x + cy * 0.9, a.y + sy * 0.9, a.z + 1.0, 2.0, 1.9, 0.14, rgb, a.yaw, emis=0.5,
                                    rad=0.05, alpha=0.5, pat=1))
            if "deadline" in fx or "rush" in fx:
                glows.append((*m.muzzle(a), 0.5, 255, 255, 255, 0.8))
            if a.overhp > 1.0 and a is not viewer:
                alpha.append(M.inst(a.x, a.y, cz, 1.5 * a.hero.scale, a.height * 1.1, 1.5 * a.hero.scale,
                                    (255, 236, 150), rad=9.0, alpha=0.16, pat=1))
            if "stun" in fx:
                for i in range(3):
                    q = t * 6.0 + i * 2.09
                    glows.append((a.x + math.cos(q) * 0.4, a.y + math.sin(q) * 0.4, a.z + a.height + 0.3, 0.2, 255,
                                  230, 90, 1.0))
            if a.firing and a.hero.weapon.kind == "cone":  # the stream of slag or mist
                wp = a.hero.weapon
                c = wp.tracer
                dx, dy, dz = a.aim()
                ex, ey, ez = a.eye_pos()
                reach = m.world.raycast(ex, ey, ez, dx, dy, dz, wp.rng)
                if a is viewer and first_person:
                    mx, my, mz = renderer.vm_point(M.view_model(a.hero.id).muzzle + np.array(M.VIEW_POS), VM_FOV)
                else:
                    mx, my, mz = m.muzzle(a)
                for i in range(8):
                    k = (i + (t * 7.0) % 1.0) / 8.0
                    d = 0.3 + k * (reach - 0.3)
                    sz = 0.16 + d * 0.15
                    j = 0.04 * d
                    px = ex + dx * d + (mx - ex) * (1.0 - k) + math.sin(t * 31 + i * 2.4) * j
                    py = ey + dy * d + (my - ey) * (1.0 - k) + math.cos(t * 27 + i * 1.9) * j
                    pz = ez + dz * d + (mz - ez) * (1.0 - k) + math.sin(t * 23 + i * 3.1) * j
                    add.append(M.inst(px, py, pz, sz, sz, sz, c, rad=9.0, alpha=0.5 * (1.0 - k * 0.7), pat=3))
                glows.append((mx, my, mz, 0.3, c[0], c[1], c[2], 0.7))
                lights.append((ex + dx * 2.5, ey + dy * 2.5, ez + dz * 2.5, (c[0] / 500, c[1] / 500, c[2] / 500), 6.0))
            if a.kick > 0.75 and a.hero.weapon.kind != "cone" and not (a is viewer and first_person):
                mx, my, mz = m.muzzle(a)
                c = a.hero.weapon.tracer
                glows.append((mx, my, mz, 0.5, c[0], c[1], c[2], 0.9))
                lights.append((mx, my, mz, (c[0] / 340, c[1] / 340, c[2] / 340), 5.0))
        elif a.death_t < 2.4 and not (a is viewer and first_person):
            k = min(a.death_t / 0.45, 1.0)
            sink = max(a.death_t - 1.6, 0.0) * 1.4
            ang, bob = M.actor_pose(0.0, 0.0, 0.0, 0.0, False, 0.0, 0.0)
            rows.append(model.rows((a.x, a.z + 0.16 * k - sink, a.y), M.yaw_basis(a.yaw, -k * math.pi / 2), ang, 0.0,
                                   rgb, 0.0, 0.0))
    # the objective
    if m.mode == "control":
        px, py, pr = m.md.point
        pz = m.world.fh[int(py) - 4][int(px)] + 0.56
        col = (240, 240, 240) if m.owner < 0 else (ALLY_RGB if m.owner == vteam else ENEMY_RGB)
        if m.contested:
            col = GOLD
        locked = m.phase != "live" or m.live_t < 12.0
        alpha.append(M.inst(px, py, pz, pr * 2, 0.04, pr * 2, col, mode=1, alpha=0.35 if locked else 0.95, pat=2))
        if m.owner >= 0:
            add.append(M.inst(px, py, pz + 9.0, 0.5, 18.0, 0.5, col, mode=1, alpha=0.3, pat=3))
    elif m.md.track:
        cx, cy, cyaw = track_at(m.md.track, m.cart_d)
        att_mine = m.attack_team == vteam
        rgb = ALLY_RGB if att_mine else ENEMY_RGB
        hover = 0.06 * math.sin(t * 2.4)
        rows.append(M.cart_model().rows((cx, 0.05 + hover, cy), M.yaw_basis(cyaw), None, 0.0, rgb))
        col = GOLD if m.contested else (rgb if m.pushers else (230, 230, 230))
        alpha.append(M.inst(cx, cy, 0.07, CART_RADIUS * 2, 0.04, CART_RADIUS * 2, col, mode=1, alpha=0.9, pat=2))
        glows.append((cx, cy, 2.5 + hover, 1.2, rgb[0], rgb[1], rgb[2], 0.7))
        for sx in (-0.8, 0.8):
            for sz in (-1.3, 1.3):
                gx = cx + math.cos(cyaw) * sz - math.sin(cyaw) * sx
                gy = cy + math.sin(cyaw) * sz + math.cos(cyaw) * sx
                glows.append((gx, gy, 0.12, 0.6, 110, 210, 255, 0.5))
    # spawn-room doors
    for rid, room in m.md.rooms.items():
        owner = m.world.room_owner.get(rid)
        if owner is None and m.md.mode != "control" and rid not in m.world.closed:
            col, al = (200, 200, 210), 0.5
        else:
            col = ALLY_RGB if owner == vteam else ENEMY_RGB
            al = 0.75 if rid in m.world.closed else 0.4
        for x0, y0, x1, y1, h in room["doors"]:
            wide_x = (x1 - x0) > (y1 - y0)
            alpha.append(M.inst((x0 + x1) / 2, (y0 + y1) / 2, h + 1.5, (x1 - x0) if wide_x else 0.1, 3.0,
                                0.1 if wide_x else (y1 - y0), col, 0.0, emis=0.3, rad=0.02, alpha=al, pat=4))
    # health packs
    for pk in m.packs:
        x, y, z = pk["x"], pk["y"], pk["z"]
        big = pk["big"]
        col = (255, 200, 70) if big else (110, 240, 150)
        rows.append([M.inst(x, y, z + 0.04, 1.5, 0.08, 1.5, (70, 76, 92), mode=1)])
        if pk["t"] <= 0.0:
            s = 0.62 if big else 0.44
            zz = z + 0.75 + 0.08 * math.sin(t * 2.5 + x)
            rows.append([M.inst(x, y, zz, s, s * 0.34, s * 0.34, col, t * 1.6, emis=0.9, rad=0.04),
                         M.inst(x, y, zz, s * 0.34, s, s * 0.34, col, t * 1.6, emis=0.9, rad=0.04)])
            glows.append((x, y, zz, 0.9, col[0], col[1], col[2], 0.45))
        else:
            k = 1.0 - pk["t"] / (15.0 if big else 10.0)
            alpha.append(M.inst(x, y, z + 0.1, 1.4 * k + 0.05, 0.03, 1.4 * k + 0.05, col, mode=1, alpha=0.5))
    # shots in flight
    for p in m.projectiles:
        sp = math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz) or 1.0
        yaw, pitch = math.atan2(p.vy, p.vx), math.asin(max(min(p.vz / sp, 1.0), -1.0))
        if p.kind == "rivet":
            rows.append([M.inst(p.x, p.y, p.z, 0.07, 0.07, 0.6, (255, 200, 90), yaw, pitch, emis=2.0, rad=0.03)])
            glows.append((p.x, p.y, p.z, 0.35, 255, 180, 70, 0.7))
        elif p.kind == "charge":
            rows.append([M.inst(p.x, p.y, p.z, 0.3, 0.3, 0.3, (60, 64, 70), p.age * 9, p.age * 7, rad=9.0),
                         M.inst(p.x, p.y, p.z, 0.32, 0.1, 0.32, (255, 170, 60), p.age * 9, p.age * 7, emis=1.6,
                                mode=1)])
            glows.append((p.x, p.y, p.z, 0.5, 255, 160, 60, 0.6))
        elif p.kind == "slag":
            rows.append([M.inst(p.x, p.y, p.z, 0.44, 0.44, 0.44, (255, 130, 40), emis=1.8, rad=9.0)])
            glows.append((p.x, p.y, p.z, 0.9, 255, 120, 30, 0.9))
        elif p.kind == "flare":
            rows.append([M.inst(p.x, p.y, p.z, 0.12, 0.12, 0.4, (160, 255, 230), yaw, pitch, emis=2.5, rad=0.05)])
            glows.append((p.x, p.y, p.z, 1.0, 140, 255, 220, 1.0))
        elif p.kind == "mine":
            rows.append([M.inst(p.x, p.y, p.z, 0.36, 0.16, 0.36, (70, 90, 70), p.age * 8, p.age * 5, mode=1)])
        else:
            rows.append([M.inst(p.x, p.y, p.z, 0.26, 0.34, 0.26, (130, 255, 200), p.age * 6, p.age * 8, emis=1.2,
                                rad=9.0)])
            glows.append((p.x, p.y, p.z, 0.5, 120, 255, 190, 0.5))
    # barriers, gadgets, zones
    for b in m.barriers:
        mine = b.team == vteam
        col = (120, 200, 255) if mine else (255, 130, 110)
        k = 0.3 + 0.5 * (b.hp / b.max_hp) + b.flash * 3.0
        fade = min(b.life / 0.4, 1.0)
        alpha.append(M.inst(b.x, b.y, b.z + b.h / 2, b.w, b.h, 0.12, col, b.yaw, emis=0.3, rad=0.05,
                            alpha=min(k, 1.0) * fade * 0.7, pat=1))
        rx, ry = -b.ny * b.w / 2, b.nx * b.w / 2
        for s in (-1, 1):
            rows.append([M.inst(b.x + rx * s, b.y + ry * s, b.z + 0.25, 0.3, 0.5, 0.3, (60, 64, 76), mode=1),
                         M.inst(b.x + rx * s, b.y + ry * s, b.z + 0.56, 0.2, 0.12, 0.2, col, emis=1.6, mode=1)])
    for g in m.gadgets:
        mine = g.team == vteam
        rgb = ALLY_RGB if mine else ENEMY_RGB
        fl = (255, 255, 255) if g.flash > 0 else None
        if g.kind == "mine":
            blink = 1.8 if (g.age > 0.7 and int(t * 4) % 2 == 0) else 0.2
            rows.append([M.inst(g.x, g.y, g.z + 0.07, 0.5, 0.14, 0.5, (64, 84, 64), mode=1),
                         M.inst(g.x, g.y, g.z + 0.17, 0.16, 0.08, 0.16, rgb, emis=blink, mode=1)])
        elif g.kind == "pylon":
            pulse = 0.5 + 0.5 * math.sin(t * 5.0)
            rows.append([M.inst(g.x, g.y, g.z + 0.1, 0.6, 0.2, 0.6, (52, 76, 170), mode=1),
                         M.inst(g.x, g.y, g.z + 0.8, 0.12, 1.3, 0.12, fl or (240, 240, 246), mode=1),
                         M.inst(g.x, g.y, g.z + 1.6, 0.36, 0.36, 0.36, fl or (255, 240, 170), t * 2.0, 0.6, 0.6,
                                emis=1.5 + pulse, rad=0.04),
                         M.inst(g.x, g.y, g.z + 1.25, 0.26, 0.06, 0.26, rgb, emis=1.6, mode=1)])
            alpha.append(M.inst(g.x, g.y, g.z + 0.06, 16.0, 0.04, 16.0, (255, 240, 170) if mine else ENEMY_RGB,
                                mode=1, alpha=0.45, pat=2))
            glows.append((g.x, g.y, g.z + 1.6, 0.8, 255, 236, 150, 0.6))
        else:  # the lighthouse
            rows.append([M.inst(g.x, g.y, g.z + 0.15, 1.0, 0.3, 1.0, (52, 76, 170), mode=1),
                         M.inst(g.x, g.y, g.z + 1.6, 0.3, 2.8, 0.3, fl or (240, 240, 246), mode=1),
                         M.inst(g.x, g.y, g.z + 3.3, 0.8, 0.8, 0.8, fl or (255, 240, 170), emis=1.6, rad=9.0),
                         M.inst(g.x, g.y, g.z + 2.8, 0.5, 0.08, 0.5, rgb, emis=1.6, mode=1),
                         M.inst(g.x, g.y, g.z + 3.8, 0.6, 0.12, 0.6, (242, 202, 92), mode=1)])
            alpha.append(M.inst(g.x, g.y, g.z + 0.06, 26.0, 0.04, 26.0, (255, 236, 150) if mine else ENEMY_RGB,
                                mode=1, alpha=0.7, pat=2))
            for i in range(2):
                q = t * 1.6 + i * math.pi
                add.append(_line((g.x, g.y, g.z + 3.3), (g.x + math.cos(q) * 9, g.y + math.sin(q) * 9, g.z + 3.0),
                                 0.3, (255, 236, 150), 0.1))
            glows.append((g.x, g.y, g.z + 3.3, 1.5, 255, 236, 150, 0.6))
            lights.append((g.x, g.y, g.z + 3.0, (0.3, 0.26, 0.14), 13.0))
    for z in m.zones:
        fade = min((z.life - z.age) / 0.5, 1.0, z.age / 0.2 + 0.2)
        alpha.append(M.inst(z.x, z.y, z.z + 0.05, z.r * 2, 0.04, z.r * 2, (255, 120, 30), mode=1, alpha=fade, pat=2))
        alpha.append(M.inst(z.x, z.y, z.z + 0.04, z.r * 1.9, 0.03, z.r * 1.9, (255, 90, 20), mode=1, emis=0.4,
                            alpha=0.2 * fade))
        for i in range(3):
            q = t * (1.1 + i * 0.4) + i * 2.1 + z.x
            rr = z.r * (0.3 + 0.25 * i)
            glows.append((z.x + math.cos(q) * rr, z.y + math.sin(q) * rr, z.z + 0.3, 0.8, 255, 130, 30, 0.6 * fade))
        lights.append((z.x, z.y, z.z + 0.8, (0.8 * fade, 0.3 * fade, 0.08 * fade), z.r + 3.0))
    # things only seen
    muzzle = None
    for f in m.fx:
        k = f.t / f.life
        c = f.col
        kind = f.kind
        if kind in ("tracer", "ptracer", "streak", "rain"):
            a0 = f.a
            if kind == "ptracer" and first_person:
                if muzzle is None:
                    muzzle = renderer.vm_point(M.view_model(viewer.hero.id).muzzle + np.array(M.VIEW_POS),
                                               VM_FOV)
                a0 = muzzle
            add.append(_line(a0, f.b, f.size * (1.0 - 0.5 * k), c, (1.0 - k) * (0.9 if kind != "rain" else 0.5)))
        elif kind == "boom":
            s = f.size * 2.0 * (0.35 + 0.65 * (1.0 - (1.0 - k) ** 3))
            add.append(M.inst(f.a[0], f.a[1], f.a[2], s, s, s, c, rad=9.0, alpha=(1.0 - k) ** 1.5 * 0.55, pat=3))
            if k < 0.5:
                glows.append((f.a[0], f.a[1], f.a[2], f.size * 1.3, c[0], c[1], c[2], 1.2 * (1.0 - k * 2)))
                lights.append((f.a[0], f.a[1], f.a[2] + 0.5, (c[0] / 200 * (1 - k * 2), c[1] / 200 * (1 - k * 2),
                                                             c[2] / 200 * (1 - k * 2)), f.size * 2.5 + 3.0))
        elif kind == "ring":
            s = f.size * 2.0 * (1.0 - (1.0 - k) ** 2)
            alpha.append(M.inst(f.a[0], f.a[1], f.a[2], s, 0.04, s, c, mode=1, alpha=(1.0 - k), pat=2))
        elif kind == "warn":
            pulse = 0.55 + 0.45 * math.sin(f.t * 16.0)
            alpha.append(M.inst(f.a[0], f.a[1], f.a[2], f.size * 2, 0.04, f.size * 2, c, mode=1, alpha=pulse, pat=2))
            alpha.append(M.inst(f.a[0], f.a[1], f.a[2], f.size * 2 * k, 0.03, f.size * 2 * k, c, mode=1, emis=0.5,
                                alpha=0.35))
        elif kind == "flare":
            glows.append((f.a[0], f.a[1], f.a[2] + 0.3, f.size * (1.0 - k * 0.5), c[0], c[1], c[2], 1.2 * (1 - k)))
            lights.append((f.a[0], f.a[1], f.a[2] + 0.6, (c[0] / 300 * (1 - k), c[1] / 300 * (1 - k),
                                                         c[2] / 300 * (1 - k)), 8.0))
        elif kind == "heal":
            glows.append((f.a[0], f.a[1], f.a[2] + k * 0.7, f.size, c[0], c[1], c[2], 1.0 - k))
        else:  # spark, puff
            glows.append((f.a[0], f.a[1], f.a[2] + (k * 0.4 if kind == "puff" else 0.0), f.size * (0.6 + k), c[0],
                          c[1], c[2], (1.0 - k) * (0.7 if kind == "puff" else 1.0)))
    out = Frame()
    flat = [r if isinstance(r, np.ndarray) else np.array(r, F32).reshape(-1, M.ROW) for r in rows]
    out.rows = np.concatenate(flat) if flat else _EMPTY
    out.casters = np.concatenate([out.rows, own]) if own is not None else out.rows
    out.alpha = np.array(alpha, F32).reshape(-1, M.ROW) if alpha else _EMPTY
    out.add = np.array(add, F32).reshape(-1, M.ROW) if add else _EMPTY
    out.glows = glows
    out.lights = lights
    out.vm = _view_model(viewer, renderer, t, sway) if (first_person and viewer.alive) else _EMPTY
    if first_person and viewer.alive and viewer.kick > 0.6 and viewer.hero.weapon.kind != "cone":
        if muzzle is None:
            muzzle = renderer.vm_point(M.view_model(viewer.hero.id).muzzle + np.array(M.VIEW_POS), VM_FOV)
        c = viewer.hero.weapon.tracer
        glows.append((*muzzle, 0.2, c[0], c[1], c[2], 1.0))
        lights.append((viewer.x, viewer.y, viewer.z + viewer.eye, (c[0] / 400, c[1] / 400, c[2] / 400), 6.0))
    return out


def _view_model(a, renderer, t: float, sway: tuple) -> np.ndarray:  # noqa: ANN001
    """The player's own weapon: in the eye's axes, bobbing as they walk, kicking as it fires."""
    model = M.view_model(a.hero.id)
    w = a.hero.weapon
    z = a.zoom
    ox = M.VIEW_POS[0] * (1.0 - z) + sway[0] * 0.01
    oy = M.VIEW_POS[1] + 0.07 * z + sway[1] * 0.01
    oz = M.VIEW_POS[2] - 0.04 * z
    bobk = a.move_amt * (1.0 - z * 0.8)
    ox += math.cos(a.walk) * 0.012 * bobk
    oy += -abs(math.sin(a.walk)) * 0.014 * bobk + 0.004 * math.sin(t * 1.8)
    tilt = a.kick * 0.07
    oz -= a.kick * 0.045
    if a.reload_t > 0.0:
        k = math.sin(min(max(1.0 - a.reload_t / max(w.reload, 0.05), 0.0), 1.0) * math.pi)
        oy -= 0.2 * k
        tilt += 0.22 * k
    dip = max(a.cast_anim, 1.0 if "bulldoze" in a.fx or "stun" in a.fx else 0.0)
    if dip > 0.0:
        oy -= 0.13 * dip
        ox -= 0.05 * dip
        tilt -= 0.25 * dip
    if a.melee_anim > 0.0:
        k = math.sin(a.melee_anim / 0.3 * math.pi)
        oz += 0.2 * k
        ox -= 0.16 * k
    if not a.on_ground:
        oy += max(min(a.vz * 0.004, 0.03), -0.03)
    R, U, F = renderer.R, renderer.U, renderer.F
    if tilt:
        ct, st = math.cos(tilt), math.sin(tilt)
        U, F = U * ct - F * st, F * ct + U * st
    origin = renderer.cam + renderer.R * ox + renderer.U * oy + renderer.F * oz
    basis = np.stack([R, U, F], 1)
    rows = model.rows(origin, basis)
    if "deadline" in a.fx or "rush" in a.fx:
        rows[:, 19] += 0.5
    return rows
