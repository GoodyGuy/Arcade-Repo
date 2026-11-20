"""Rail Rush scene: what the graphics card draws each frame (see ``render.py``). The line is cut into 40 m stretches;
each stretch's scenery (track, walls, buildings, woods, cliffs, tunnels, bridges, masts and wires) is built once and
kept on the card, drawn with an offset. Trains, ramps, barriers, coins, pickups, the runner, the drone and the
particles are gathered every frame. Positions are sent relative to the runner, so nothing loses precision on a long
run. The light changes from day to sunset to night to dawn as the run goes on."""

from __future__ import annotations

import logging
import math
import random

import numpy as np
from PySide6.QtGui import QImage

from app.arcade.games.rail_rush import models as M
from app.arcade.games.rail_rush.models import Batch, Pose, rgb, rot_x, rot_y
from app.arcade.games.rail_rush.render import (BRICKS, GRASS, GRAVEL, PANELS, PLANKS, STRATA, WATER, Env,
                                               Renderer)
from app.arcade.games.rail_rush.world import ROLL_TIME, lane_x

log = logging.getLogger(__name__)
CHUNK = 40.0
FAR = 262.0
BEHIND = 45.0
POLE_GAP = 42.0
TUNNEL_PIECE = 12.0  # tunnel lamps every 12 m (the shader's lamp light repeats at the same spacing)
FOV = 60.0

BUILDING_COLS = ("#e0a46e", "#c96b4f", "#d9c7a3", "#8fb3c9", "#b5866b", "#e7d4b5", "#9fc79a", "#d98b8b", "#c4b5d9",
                 "#f0c987")
GLASS = ("#2f4a66", "#3d5a78", "#28405a")
AWNINGS = ("#e8323c", "#2c6fd8", "#35a853", "#f2c12e")


def _norm(v: tuple) -> tuple:
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


# day, sunset, night, dawn: what Env holds at each
PALETTES = (
    dict(zenith=(0.22, 0.52, 0.92), horizon=(0.76, 0.88, 0.97), sun=_norm((0.42, 0.78, -0.46)),
         sun_col=(0.72, 0.68, 0.6), amb_up=(0.56, 0.6, 0.7), amb_dn=(0.4, 0.38, 0.36), cloud=(1.0, 1.0, 1.0),
         hills=(0.42, 0.58, 0.62), night=0.0, fog=(120.0, 255.0)),
    dict(zenith=(0.26, 0.3, 0.6), horizon=(1.0, 0.68, 0.45), sun=_norm((0.7, 0.32, -0.62)),
         sun_col=(0.95, 0.6, 0.36), amb_up=(0.52, 0.46, 0.56), amb_dn=(0.36, 0.3, 0.3), cloud=(1.0, 0.74, 0.62),
         hills=(0.46, 0.36, 0.46), night=0.3, fog=(110.0, 245.0)),
    dict(zenith=(0.02, 0.03, 0.1), horizon=(0.12, 0.15, 0.28), sun=_norm((-0.4, 0.7, -0.5)),
         sun_col=(0.3, 0.35, 0.48), amb_up=(0.36, 0.4, 0.54), amb_dn=(0.19, 0.19, 0.25), cloud=(0.2, 0.22, 0.32),
         hills=(0.07, 0.09, 0.16), night=1.0, fog=(85.0, 235.0)),
    dict(zenith=(0.36, 0.46, 0.76), horizon=(0.98, 0.8, 0.72), sun=_norm((-0.6, 0.35, -0.6)),
         sun_col=(0.88, 0.7, 0.6), amb_up=(0.52, 0.52, 0.62), amb_dn=(0.35, 0.32, 0.34), cloud=(1.0, 0.86, 0.82),
         hills=(0.5, 0.48, 0.58), night=0.15, fog=(115.0, 250.0)),
)
PHASE_M = 1500.0  # metres of each part of the day


def light_at(distance: float) -> dict:
    """The palette ``distance`` metres into a run (the last quarter of each part blends into the next)."""
    u = distance / PHASE_M
    i = int(u) % 4
    f = u - math.floor(u)
    k = max(0.0, (f - 0.75) / 0.25)
    k = k * k * (3 - 2 * k)
    a, b = PALETTES[i], PALETTES[(i + 1) % 4]
    if k <= 0.0:
        return a
    out = {}
    for key, va in a.items():
        vb = b[key]
        out[key] = (va + (vb - va) * k) if isinstance(va, float) else tuple(x + (y - x) * k for x, y in zip(va, vb))
    out["sun"] = _norm(out["sun"])
    return out


class Scene:
    """Draws the game's world through its camera (the renderer and the stretches on the card live here)."""

    def __init__(self) -> None:
        self.r = Renderer()
        self.env = Env()
        self._wid: int | None = None
        self.built: set[int] = set()
        self.error = ""
        self.ms = 0.0

    def close(self) -> None:
        self.r.close()
        self.built.clear()
        self._wid = None

    # --- stretches of scenery -------------------------------------------------------------------------------------

    def _sync_chunks(self, wd) -> dict:  # noqa: ANN001 - a World
        if self._wid != id(wd):
            self.r.drop_all()
            self.built.clear()
            self._wid = id(wd)
        pz = wd.pz
        k0, k1 = math.floor((pz - BEHIND) / CHUNK), math.floor((pz + FAR) / CHUNK)
        ready = min(min(wd.side_z), wd.zone_z)
        made = 0
        offsets = {}
        for k in range(k0, k1 + 1):
            if k not in self.built:
                if (k + 1) * CHUNK > ready or made >= 2:
                    continue
                self.r.put_chunk(k, *build_chunk(wd, k))
                self.built.add(k)
                made += 1
            offsets[k] = k * CHUNK - pz
        for k in [k for k in self.built if k < k0 - 1]:
            self.built.discard(k)
            self.r.drop_chunk(k)
        return offsets

    # --- the frame ------------------------------------------------------------------------------------------------

    def render(self, g, w: int, h: int) -> QImage | None:  # noqa: ANN001 - the RailRushGame
        """The picture of this moment at ``w`` x ``h`` pixels (None when the graphics card cannot draw it)."""
        if self.error:
            return None
        try:
            return self._render(g, w, h)
        except RuntimeError as exc:  # no OpenGL 3.3: the game says so instead of stopping
            log.warning("Rail Rush cannot draw: %s", exc)
            self.error = str(exc)
            return None

    def _render(self, g, w: int, h: int) -> QImage:  # noqa: ANN001
        wd = g.world
        pz = wd.pz
        offsets = self._sync_chunks(wd)
        env = self.env
        pal = light_at(wd.distance if wd.running else 0.0)
        for key, v in pal.items():
            setattr(env, key, v)
        env.time = g.anim_t
        env.travel = pz
        env.tunnels = [(zn.z0 - pz, zn.z1 - pz) for zn in wd.zones
                       if zn.kind == "tunnel" and zn.z1 > pz - 20 and zn.z0 < pz + FAR][:2]
        b = Batch()
        self._obstacles(b, wd, pz)
        self._coins(b, wd, pz, g.anim_t)
        for pk in wd.pickups:
            if not pk.taken and pz - 2 < pk.z < pz + FAR * 0.7:
                bob = math.sin(g.anim_t * 3.2 + pk.z) * 0.15
                M.pickup(b, pk.kind, pk.x, pk.y + bob, pk.z - pz, g.anim_t + pk.z)
        self._runner(b, g, wd)
        self._drone(b, g, wd)
        self._particles(b, g, pz)
        sx = sy = 0.0
        if g.shake > 0:
            amp = g.shake * 0.12
            sx, sy = math.sin(g.anim_t * 61) * amp, math.cos(g.anim_t * 53) * amp
        eye = (g.cam_x + sx, g.cam_y + sy, -g.cam_back)
        img = self.r.render(w, h, eye, g.cam_pitch, -wd.lean * 0.025 if wd.running else 0.0, FOV, env, offsets,
                            b.arrays(False), b.arrays(True))
        self.ms = self.r.stats["ms"]
        return img

    def _obstacles(self, b: Batch, wd, pz: float) -> None:  # noqa: ANN001
        for o in wd.obstacles:
            if o.far < pz - 12 or o.z > pz + FAR:
                continue
            if o.kind == "train":
                model = M.train_model(o.style % 6 if not o.vz else o.style, o.cars, bool(o.vz))
            elif o.kind == "ramp":
                model = M.ramp_model()
            else:
                model = M.barrier_model(o.kind)
            b.put_model(model, o.x, 0.0, o.z - pz)

    def _coins(self, b: Batch, wd, pz: float, t: float) -> None:  # noqa: ANN001
        vis = [(c.x, c.y, c.z - pz) for c in wd.coins if not c.taken and pz - 3 < c.z < pz + FAR * 0.75]
        if vis:
            a = np.array(vis, np.float32)
            b.put(M.CYL, M.coins(a[:, 0], a[:, 1], a[:, 2], t))

    def pose(self, g, wd) -> Pose:  # noqa: ANN001
        p = Pose(phase=wd.anim, lean=wd.lean, t=g.anim_t, shield=wd.board > 0 and not wd.dead,
                 springs=wd.power.get("sneakers", 0.0) > 0, balloons=wd.flying)
        if g.state == "title" or not wd.running:
            p.mode, p.yaw = "idle", math.pi
            return p
        if wd.run_time < 0.4:
            u = wd.run_time / 0.4
            p.yaw = math.pi * (1.0 - u * u * (3 - 2 * u))
        if wd.dead:
            p.mode = "dead" if wd.death == "crash" else "caught"
            p.fall = min(1.0, wd.dead_time * 3.0)
        elif wd.flying:
            p.mode = "fly"
        elif wd.rolling:
            p.mode, p.roll = "roll", 1.0 - wd.roll / ROLL_TIME
        elif not wd.grounded:
            p.mode, p.rising = "jump", wd.vy > 0
        elif wd.stumble_anim > 0:
            p.mode = "stumble"
        return p

    def _runner(self, b: Batch, g, wd) -> None:  # noqa: ANN001
        if wd.invuln > 0 and wd.running and not wd.dead and int(g.anim_t * 14) % 2:
            return
        ground = wd.support() if wd.flying or not wd.grounded else wd.ground
        M.shadow(b, wd.x, ground, 0.0, 0.5, max(0.12, 0.45 - (wd.y - ground) * 0.04))
        M.runner(b, g.char, self.pose(g, wd), wd.x, wd.y, 0.0)

    def _drone(self, b: Batch, g, wd) -> None:  # noqa: ANN001
        t = g.anim_t
        if g.state == "title" or not wd.running:
            M.drone(b, 1.9, 2.6 + 0.12 * math.sin(t * 1.7), 3.2, t)
            return
        chase = wd.chase
        if chase < 0.02:
            return
        bob = 0.12 * math.sin(t * 2.3)
        if wd.dead:
            k = min(1.0, wd.dead_time * 2.0)
            x = wd.x * k + (wd.x - 1.0) * (1 - k)
            z = -1.6 * (1 - k) - 0.2 * k
            y = wd.y + 2.7 + bob
            M.drone(b, x, y, z, t, beam_to=wd.y if k > 0.5 else None, alarm=True)
            return
        z = -2.4 - (1.0 - chase) * 9.0
        x = max(-3.6, min(3.6, wd.x - 1.35))
        y = g.chaser_y + 2.85 + bob
        M.shadow(b, x, g.chaser_y, z, 0.7, 0.3)
        M.drone(b, x, y, z, t, alarm=chase > 0.6)

    def _particles(self, b: Batch, g, pz: float) -> None:  # noqa: ANN001
        for x, y, z, _vx, _vy, _vz, age, life, kind, color in g.particles:
            a = max(0.0, 1.0 - age / life)
            zz = z - pz
            if kind == "spark":
                r = 0.06 + 0.06 * a
                b.ball((x, y, zz), (r, r, r), rgb(color), emis=1.5)
            elif kind == "dust":
                r = 0.15 + age * 0.8
                b.ball((x, y, zz), (r, r * 0.6, r), rgb(color), alpha=0.45 * a, clear=True)
            elif kind == "smoke":
                r = 0.1 + age * 0.5
                b.ball((x, y, zz), (r, r, r), rgb(color), alpha=0.4 * a, clear=True)
            else:
                r = 0.08
                b.box((x, y, zz), (r, r, r), rgb(color), emis=0.6, r=rot_y(age * 9))


# --- building a stretch ----------------------------------------------------------------------------------------------

def _segments(wd, z0: float, z1: float) -> list:  # noqa: ANN001
    """(kind, a, b) of the zones over [z0, z1) (open land where none is known)."""
    out = []
    for zn in wd.zones:
        a, c = max(z0, zn.z0), min(z1, zn.z1)
        if a < c:
            out.append((zn.kind, a, c))
    if not out:
        out.append(("city", z0, z1))
    return out


def build_chunk(wd, k: int) -> tuple[dict, dict]:  # noqa: ANN001
    """The scenery of stretch ``k`` (z from 40 k to 40 k + 40), positions relative to its start."""
    z0, z1 = k * CHUNK, (k + 1) * CHUNK
    b = Batch()
    segs = _segments(wd, z0, z1)
    for kind, a, c in segs:
        la, lc = a - z0, c - z0
        mid, half = (la + lc) / 2, (lc - la) / 2
        if kind == "bridge":
            b.quad((0, -12.0, mid), (300, 1, half), rgb("#2f7fb8"), rgb("#5fb8e8"), WATER)
            b.box((0, -0.6, mid), (4.6, 0.6, half), rgb("#8b8780"), rgb("#6f6a66"), PANELS, px=6.0)
        elif kind != "tunnel":
            if kind == "country":
                b.quad((0, -0.02, mid), (300, 1, half), rgb("#5f9e45"), rgb("#86b955"), GRASS)
            else:
                b.quad((0, -0.02, mid), (300, 1, half), rgb("#8f8a7c"), rgb("#a19a86"), GRASS)
        else:
            b.quad((0, -0.02, mid), (40, 1, half), rgb("#3f3a35"))
        # the track: ballast, beds, sleepers, rails
        b.box((0, 0.03, mid), (4.3, 0.05, half), rgb("#7d7366"), pat=GRAVEL)
        for lane in range(3):
            lx = lane_x(lane)
            b.box((lx, 0.09, mid), (1.42, 0.03, half), rgb("#6c6358"), pat=GRAVEL)
            for rx in (lx - 0.72, lx + 0.72):
                b.box((rx, 0.24, mid), (0.045, 0.06, half), rgb("#b8bec6"), emis=0.08)
                b.box((rx, 0.17, mid), (0.07, 0.02, half), rgb("#5d5752"))
        s0 = math.ceil(a / 0.75)
        while s0 * 0.75 < c:
            sz = s0 * 0.75 - z0
            for lane in range(3):
                b.box((lane_x(lane), 0.135, sz), (1.18, 0.05, 0.12), rgb("#5e4532"))
            s0 += 1
        if kind != "tunnel":
            for lane in range(3):  # overhead wires
                b.box((lane_x(lane), 6.6, mid), (0.02, 0.02, half), rgb("#30343a"))
            for sx in (-4.2, 4.2):
                b.box((sx, 7.35, mid), (0.02, 0.02, half), rgb("#30343a"))
    # masts every 42 m (not in tunnels or on bridges)
    p = math.ceil(z0 / POLE_GAP)
    while p * POLE_GAP < z1:
        zp = p * POLE_GAP
        p += 1
        zn = wd.zone_at(zp)
        if zn is not None and zn.kind in ("tunnel", "bridge"):
            continue
        lz = zp - z0
        for sx in (-4.75, 4.75):
            b.cyl((sx, 3.8, lz), (0.15, 3.8, 0.15), rgb("#6b717b"))
            b.box((sx, 0.15, lz), (0.3, 0.15, 0.3), rgb("#8a8780"))
        b.box((0, 7.55, lz), (4.95, 0.17, 0.12), rgb("#5b6169"))
        for lane in range(3):
            b.box((lane_x(lane), 7.1, lz), (0.04, 0.45, 0.04), rgb("#c9cdd6"))
    # tunnels and bridges, in 12 m pieces
    for zn in wd.zones:
        if zn.kind not in ("tunnel", "bridge") or zn.z1 <= z0 or zn.z0 >= z1:
            continue
        if z0 <= zn.z0 < z1:
            (_portal if zn.kind == "tunnel" else _bank)(b, zn.z0 - z0, -1)
        if z0 <= zn.z1 < z1:
            (_portal if zn.kind == "tunnel" else _bank)(b, zn.z1 - z0, 1)
        i = max(0, math.floor((z0 - zn.z0) / TUNNEL_PIECE))
        while True:
            pa = zn.z0 + i * TUNNEL_PIECE
            if pa >= min(z1, zn.z1):
                break
            pb = min(pa + TUNNEL_PIECE, zn.z1)
            if pa >= z0:
                (_tunnel_piece if zn.kind == "tunnel" else _bridge_piece)(b, pa - z0, pb - z0, i)
            i += 1
    # the sides
    for sgi, side in enumerate(wd.sides):
        sg = -1 if sgi == 0 else 1
        for s in side:
            if z0 <= s.z0 < z1:
                zn = wd.zone_at(s.z0)
                _side(b, s, sg, s.z0 - z0, s.z1 - z0, zn is None or zn.kind == "city")
    return b.freeze()


def _portal(b: Batch, z: float, way: int) -> None:
    """The rock face of a tunnel mouth (``way`` -1: the way in at ``z``, +1: the way out)."""
    zc = z + way * 0.6
    rock, rock2 = rgb("#8a7560"), rgb("#6f8f4a")
    for sg in (-1, 1):
        b.box((sg * 23.0, 8.0, zc), (17.6, 8.0, 0.6), rock, rock2, STRATA, pw=3.0)
    b.box((0, 11.7, zc), (5.6, 4.3, 0.6), rock, rock2, STRATA, pw=3.0)
    b.box((0, 16.2, zc + way * 2.0), (40.0, 0.25, 2.6), rgb("#5a9a40"))
    frame = rgb("#c9bba3")
    for sg in (-1, 1):
        b.box((sg * 5.55, 3.6, zc - way * 0.1), (0.35, 3.6, 0.65), frame, rgb("#a89a84"), BRICKS, px=0.55)
    b.box((0, 7.45, zc - way * 0.1), (5.9, 0.3, 0.66), frame, rgb("#a89a84"), BRICKS, px=0.55)


def _tunnel_piece(b: Batch, a: float, c: float, i: int) -> None:
    mid, half = (a + c) / 2, (c - a) / 2
    wall = rgb("#6a5d52")
    for sg in (-1, 1):
        b.box((sg * 5.45, 3.5, mid), (0.25, 3.5, half), wall, rgb("#4d443c"), BRICKS, px=0.6)
        b.box((sg * 22.8, 8.0, mid), (17.1, 8.0, half), rgb("#6f8f4a"))
        b.box((sg * 5.0, 0.4, mid), (0.25, 0.4, half), rgb("#4a4440"))
    b.box((0, 7.3, mid), (5.7, 0.3, half), rgb("#3e3732"))
    b.box((0, 12.0, mid), (5.7, 4.4, half), rgb("#6f8f4a"))
    b.box((0, 6.95, a), (0.7, 0.05, 0.25), rgb("#ffe9a0"), emis=2.0)
    for sg in (-1, 1):
        b.box((sg * 5.15, 5.2, a), (0.06, 0.3, 0.12), rgb("#ffe9a0"), emis=1.6)


def _bank(b: Batch, z: float, way: int) -> None:
    """The river bank at either end of a bridge."""
    zc = z + way * 2.0
    for sg in (-1, 1):
        b.box((sg * 152.3, -6.0, zc), (147.7, 6.0, 2.0), rgb("#8a7560"), rgb("#6f8f4a"), STRATA, pw=7.0)
    b.box((0, -6.0, zc), (4.6, 6.0, 2.0), rgb("#9a9690"), rgb("#7f7b75"), PANELS, px=1.2)


def _bridge_piece(b: Batch, a: float, c: float, i: int) -> None:
    mid, half = (a + c) / 2, (c - a) / 2
    red = rgb("#c9532b")
    for sg in (-1, 1):
        x = sg * 4.85
        b.box((x, 0.18, mid), (0.16, 0.18, half), red)
        b.box((x, 6.0, mid), (0.16, 0.2, half), red)
        b.box((x, 3.0, a), (0.14, 3.0, 0.14), red)
        ln = math.hypot(c - a, 5.6) / 2
        ang = math.atan2(c - a, 5.6)
        b.box((x, 3.0, mid), (0.1, ln, 0.1), red, r=rot_x(ang * (1 if i % 2 else -1)))
        b.box((sg * 4.55, 1.0, mid), (0.04, 0.04, half), rgb("#d9dde3"))
    if i % 2 == 0:
        b.box((0, -6.6, a + 1.0), (3.2, 5.4, 1.0), rgb("#9a9690"), rgb("#7f7b75"), PANELS, px=1.2)
    if half > 2.0:
        b.box((0, 6.0, mid), (4.85, 0.1, 0.1), red)


def _side(b: Batch, s, sg: int, a: float, c: float, city: bool = True) -> None:  # noqa: ANN001 - a world.Side
    rng = random.Random(s.seed)
    mid, half = (a + c) / 2, (c - a) / 2
    if s.kind in ("tunnel", "bridge"):
        return
    if s.kind == "wall":
        h = s.height
        brick = rng.random() < 0.5
        col = rgb(rng.choice(("#b0573f", "#a8664e", "#9c4a36"))) if brick else rgb(rng.choice(("#cbbda5", "#bfb39d")))
        b.box((sg * 5.65, h / 2, mid), (0.25, h / 2, half), col, rgb("#d8cdb8") if brick else rgb("#a89c86"),
              BRICKS if brick else PANELS, px=0.5 if brick else 3.0)
        b.box((sg * 5.65, h + 0.1, mid), (0.35, 0.1, half), rgb("#d9cdb8"))
    elif s.kind == "fence":
        _fence(b, sg, a, c, mid, half)
        for _ in range(int(half / 3)):
            zt = rng.uniform(a, c)
            M.round_tree(b, sg * rng.uniform(7.0, 9.0), zt, rng.uniform(4.0, 6.0))
    elif s.kind == "trees":
        if rng.random() < 0.5:
            _fence(b, sg, a, c, mid, half)
        zt = a + rng.uniform(0, 3)
        while zt < c:
            M.pine(b, sg * rng.uniform(6.8, 10.0), zt, rng.uniform(5.0, 8.5), rng.uniform(0.85, 1.1))
            zt += rng.uniform(3.0, 6.0)
        zt = a + rng.uniform(0, 5)
        while zt < c:
            M.pine(b, sg * rng.uniform(14.0, 45.0), zt, rng.uniform(7.0, 13.0), rng.uniform(0.75, 1.0))
            zt += rng.uniform(4.0, 9.0)
    elif s.kind == "cliff":
        h = s.height
        b.box((sg * 9.6, h / 2, mid), (4.0, h / 2, half), rgb("#8a7560"), rgb("#6f8f4a"), STRATA, pw=float(s.seed % 50))
        b.box((sg * 9.6, h + 0.2, mid), (4.0, 0.2, half), rgb("#5a9a40"))
        for _ in range(int(half / 4)):
            M.pine(b, sg * rng.uniform(7.0, 12.5), rng.uniform(a, c), rng.uniform(4.0, 6.5))
            # (pines on the top read as the wood going on above the cutting)
    elif s.kind == "building":
        hw = 4.0
        M.building(b, sg * (5.5 + hw), mid, hw, s.height, max(1.0, half - 0.5), rng.choice(BUILDING_COLS),
                   rng.choice(GLASS), s.seed, shop=rng.choice(AWNINGS))
        b.box((sg * 5.6, 0.5, mid), (0.15, 0.5, half), rgb("#bfb39d"))
    if s.back > 0 and s.kind in ("wall", "fence"):
        hw = rng.uniform(4.0, 6.0)
        M.building(b, sg * (rng.uniform(10.0, 13.0) + hw), mid, hw, s.back, max(1.0, half - 1.0),
                   rng.choice(BUILDING_COLS), rng.choice(GLASS), s.seed + 1)
    if city and s.kind in ("wall", "building", "fence") and rng.random() < 0.6:  # the skyline further off
        hw = rng.uniform(5, 9)
        M.building(b, sg * rng.uniform(30, 60), mid, hw, rng.uniform(18, 40), max(2.0, half), rng.choice(BUILDING_COLS),
                   rng.choice(GLASS), s.seed + 2)


def _fence(b: Batch, sg: int, a: float, c: float, mid: float, half: float) -> None:
    post = rgb("#6b717b")
    zp = math.ceil(a / 3.0) * 3.0
    while zp < c:
        b.cyl((sg * 5.4, 1.15, zp), (0.05, 1.15, 0.05), post)
        zp += 3.0
    for yy in (2.2, 1.4, 0.6):
        b.box((sg * 5.4, yy, mid), (0.025, 0.025, half), post)
    b.box((sg * 5.4, 1.15, mid), (0.005, 1.0, half), rgb("#9aa3ad"), rgb("#c9cdd6"), PLANKS, px=0.08, alpha=1.0)

