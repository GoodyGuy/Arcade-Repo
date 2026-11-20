"""The bots: nine of the ten heroes in a match (all ten when a test lets one drive the player's hero too).

A bot does what a player does: it fills in a hero's ``Controls`` and turns its view. Several times a second it
*thinks*: who can I see, who do I shoot or heal, where should I be (the objective, beside the tank, a health pack),
is an ability worth using. Every frame it *steers* along the map's routes toward that place, strafing and keeping its
hero's range in a fight, and *aims* with a reaction time, a turning speed and an error that depend on the difficulty.

Team play comes from a few rules: tanks stand on the objective and the rest fan out round it; supports stay by the
tank, heal whoever is lowest and save their ultimate for a real fight; a bot that respawns waits a moment for
team-mates who are about to; a badly hurt bot goes for a health pack or its healer."""

from __future__ import annotations

import math
import random

from .data import STEP
from .maps import track_at


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def lob_pitch(dh: float, dz: float, v: float, g: float) -> float:
    """The (low) angle to throw at ``v`` m/s to land ``dh`` away and ``dz`` up."""
    if g <= 0.0 or dh < 0.1:
        return math.atan2(dz, max(dh, 0.1))
    disc = v ** 4 - g * (g * dh * dh + 2.0 * dz * v * v)
    if disc < 0.0:
        return math.pi / 4
    return math.atan((v * v - math.sqrt(disc)) / (g * dh))


class Brain:
    def __init__(self, a, skill: tuple, rng: random.Random) -> None:  # noqa: ANN001 - sim.Actor
        self.a = a
        self.err, self.react, self.turn, self.gate, _dmg = skill
        self.rng = random.Random(rng.random())
        self.think_t = self.rng.random() * 0.2
        self.target = None
        self.heal_target = None
        self.seen = 0.0
        self.goal: tuple | None = None  # (field key, x, y, z)
        self.spot = (0.0, 0.0)  # where exactly to stand when near the goal
        self.off = (0.0, 0.0)
        self.off_t = 0.0
        self.strafe = 1.0
        self.strafe_t = 0.0
        self.stuck_t = 0.0
        self.last = (0.0, 0.0)
        self.unstick = 0.0
        self.hold_t = 0.0
        self.snap_t = 0.0
        self.w1 = self.w2 = self.wu = self.wm = False
        self.dash_fwd = 0.0
        self.slot = self.rng.uniform(0, 6.283)
        self.retreat = False
        self.follow = False
        self.seen_list: list = []
        self.look = (1.0, 0.0)

    def respawned(self, m) -> None:  # noqa: ANN001 - sim.Match
        """Back from the dead: wait a moment for team-mates who are about to respawn, and go together."""
        self.target = self.heal_target = None
        self.seen = 0.0
        self.retreat = False
        wait = [t.respawn_t for t in m.allies(self.a) if not t.alive and t is not self.a and 0.0 < t.respawn_t < 3.5]
        self.hold_t = max(wait) + 0.2 if wait and m.phase == "live" else 0.0
        self.last = (self.a.x, self.a.y)

    # --- several times a second -----------------------------------------------------------------------------------
    def think(self, m) -> None:  # noqa: ANN001
        a = self.a
        hero = a.hero
        seen = []
        for e in m.enemies(a):
            if e.alive and not e.protected:
                d = math.dist((a.x, a.y, a.z), (e.x, e.y, e.z))
                if d < 55.0 and m.visible(a, e):
                    seen.append((d, e))
        self.seen_list = seen
        ox, oy, oz = m.objective_pos()
        # who to shoot
        best, bs = None, 1e9
        for d, e in seen:
            s = d + 14.0 * e.frac
            if m.time - a.damagers.get(e.idx, -9.0) < 2.0:
                s -= 8.0
            if math.hypot(e.x - ox, e.y - oy) < 6.0:
                s -= 6.0
            if e is self.target:
                s -= 5.0
            if "mark" in e.fx:
                s -= 6.0
            if hero.id in ("flint", "wren") and e.hero.role == "support":
                s -= 6.0
            if s < bs:
                best, bs = e, s
        if best is not self.target:
            self.seen = 0.0 if self.target is None else self.react * 0.5
        self.target = best
        # who to heal
        self.heal_target = None
        if hero.weapon.heal > 0.0:
            rng = hero.weapon.rng if hero.weapon.kind == "cone" else 45.0
            low, lf = None, 0.97
            for t in m.allies(a):
                if t is not a and t.alive and t.frac < lf and math.dist(a.chest(), t.chest()) < rng + 6.0 \
                        and m.visible(a, t):
                    low, lf = t, t.frac
            if low is not None and (best is None or lf < (0.9 if hero.id == "tonic" else 0.72)):
                self.heal_target = low
        self._place(m, ox, oy, oz)
        if self.rng.random() < self.gate and m.phase == "live":
            self._abilities(m, ox, oy)

    def _place(self, m, ox: float, oy: float, oz: float) -> None:  # noqa: ANN001
        """Where this bot should be."""
        a = self.a
        role = a.hero.role
        w = m.world
        self.retreat = False
        # badly hurt and nobody to heal me: a health pack
        if a.frac < 0.36 and m.time - a.healed_t > 1.5:
            best, bd = None, 30.0
            for i, pk in enumerate(m.packs):
                d = math.hypot(pk["x"] - a.x, pk["y"] - a.y)
                if pk["t"] <= 0.0 and d < bd:
                    best, bd = i, d
            if best is not None:
                pk = m.packs[best]
                self.goal = (("pack", best), pk["x"], pk["y"], pk["z"])
                self.spot = (pk["x"], pk["y"])
                self.retreat = True
                return
        if m.mode == "control":
            rad = {"tank": 1.9, "damage": 3.6, "support": 4.3}[role]
            self.goal = (("point",), ox, oy, oz)
            self.spot = (ox + math.cos(self.slot) * rad, oy + math.sin(self.slot) * rad)
        else:
            attack = a.team == m.attack_team
            if attack or role == "tank":
                d = m.cart_d
                rad = 3.0 if role == "tank" else 3.9
            else:  # defenders stand in its way, close enough to stop it
                d = min(m.cart_d + 2.5 + (a.idx % 3) * 1.5, m.cart_len)
                rad = 2.2
            key = int(d / 4.0)
            x, y, _yaw = track_at(m.md.track, key * 4.0)
            self.goal = (("cart", key), x, y, 0.0)
            cx, cy, _yaw = track_at(m.md.track, d)
            self.spot = (cx + math.cos(self.slot) * rad, cy + math.sin(self.slot) * rad)
        # supports keep near their tank when it is close to where the team should be
        if role == "support":
            anchor = None
            for t in m.allies(a):
                if t.alive and t.hero.role == "tank" and t is not a:
                    anchor = t
            if anchor is not None and math.hypot(anchor.x - a.x, anchor.y - a.y) < 26.0 and m.visible(a, anchor) \
                    and math.hypot(anchor.x - self.goal[1], anchor.y - self.goal[2]) < 22.0:
                back = math.atan2(a.y - anchor.y, a.x - anchor.x)
                self.spot = (anchor.x + math.cos(back) * 4.0, anchor.y + math.sin(back) * 4.0)
                self.follow = True
                return
        self.follow = False
        cx, cy = int(self.spot[0]), int(self.spot[1])
        if not (0 < cx < w.w and 0 < cy < w.h) or w.fh[cy][cx] > self.goal[3] + 0.9:
            self.spot = (self.goal[1], self.goal[2])

    def _near(self, m, r: float) -> int:  # noqa: ANN001
        a = self.a
        return sum(1 for e in m.enemies(a) if e.alive and not e.protected and abs(e.z - a.z) < 4.0
                   and math.hypot(e.x - a.x, e.y - a.y) < r)

    def _snap(self, x: float, y: float, z: float, lob: tuple | None = None) -> None:
        """Turn straight to a point (for an ability thrown or aimed at it)."""
        a = self.a
        ex, ey, ez = a.eye_pos()
        dh = math.hypot(x - ex, y - ey)
        a.yaw = math.atan2(y - ey, x - ex)
        a.pitch = lob_pitch(dh, z - ez, *lob) if lob else math.atan2(z - ez, max(dh, 0.1))
        self.snap_t = 0.12

    def _abilities(self, m, ox: float, oy: float) -> None:  # noqa: ANN001
        a = self.a
        hid = a.hero.id
        t = self.target
        d = math.dist((a.x, a.y, a.z), (t.x, t.y, t.z)) if t is not None else 99.0
        hurt = m.time - a.hurt_t < 1.0
        r1, r2, ru = a.chg[0] > 0, a.chg[1] > 0, a.ult_ready
        seen = self.seen_list
        allies_near = [x for x in m.allies(a) if x.alive and math.hypot(x.x - a.x, x.y - a.y) < 11.0]
        if hid == "dozer":
            if r1 and t is not None and 4.0 < d < 13.0 and a.frac > 0.35 and abs(t.z - a.z) < 1.0:
                self._snap(t.x, t.y, t.z + 1.0)
                self.snap_t = 0.6
                self.w1 = True
            elif r2 and t is not None and d < 30.0 and (hurt or len(seen) >= 2) and \
                    not any(b.owner is a for b in m.barriers):
                self._snap(t.x, t.y, t.z + 1.0)
                self.w2 = True
            if ru and (self._near(m, 9.0) >= 2 or (self._near(m, 7.0) >= 1 and a.frac < 0.4)):
                self.wu = True
        elif hid == "caldera":
            if "sink" in a.fx:
                if a.sink_store > 90.0 and self._near(m, 6.5) >= 1:
                    self.w1 = True
            elif r1 and hurt and self._near(m, 10.0) >= 1:
                self.w1 = True
            if r2 and t is not None and 5.0 < d < 20.0:
                self._snap(t.x, t.y, t.z, (20.0, 14.0))
                self.w2 = True
            if ru and (self._near(m, 8.0) >= 2 or (self._near(m, 6.0) >= 1 and a.frac < 0.5)):
                self.wu = True
        elif hid == "flint":
            if r1 and ((hurt and d < 7.0) or self.stuck_t > 1.2):
                self.w1 = True
            if r2 and t is not None and d < 40.0:
                self._snap(t.x, t.y, t.z + 0.3, (46.0, 6.0))
                self.w2 = True
            if ru and t is not None and d < 42.0 and (len(seen) >= 2 or t.frac > 0.5):
                self.wu = True
        elif hid == "wren":
            if r1 and t is not None and ((9.0 < d < 20.0 and a.frac > 0.4) or (a.frac < 0.3 and d < 8.0)):
                self.dash_fwd = 1.0 if a.frac > 0.4 else -1.0
                self.w1 = True
            if r2 and a.frac < 0.5:
                self.w2 = True
            if ru and t is not None and d < 13.0:
                self.wu = True
        elif hid == "nitra":
            if r1 and self._near(m, 3.6) >= 1:
                self.w1 = True
            if r2 and t is not None and 4.0 < d < 17.0:
                self._snap(t.x, t.y, t.z, (16.0, 16.0))
                self.w2 = True
            if ru and t is not None and 8.0 < d < 34.0:
                bunch = sum(1 for e in m.enemies(a) if e.alive and math.hypot(e.x - t.x, e.y - t.y) < 6.0)
                if bunch >= 2:
                    self._snap(t.x, t.y, t.z)
                    self.wu = True
        elif hid == "tonic":
            if r1:
                low = min((x for x in m.allies(a) if x.alive and x is not a and m.visible(a, x)
                           and 3.0 < math.hypot(x.x - a.x, x.y - a.y) < 18.0), key=lambda x: x.frac, default=None)
                if low is not None and low.frac < 0.6:
                    self._snap(low.x, low.y, low.z, (19.0, 14.0))
                    self.w1 = True
                elif t is not None and 5.0 < d < 18.0 and len(seen) >= 2:
                    self._snap(t.x, t.y, t.z, (19.0, 14.0))
                    self.w1 = True
            if r2 and seen:
                pal = min((x for x in allies_near if x is not a and x.hero.role != "support" and m.visible(a, x)),
                          key=lambda x: math.hypot(x.x - a.x, x.y - a.y), default=None)
                if pal is not None:
                    self._snap(*pal.chest())
                    self.w2 = True
            if ru and len(allies_near) >= 2 and (
                    sum(x.frac for x in allies_near) / len(allies_near) < 0.7 or self._near(m, 12.0) >= 2):
                self.wu = True
        elif hid == "beacon":
            if r1 and (seen or math.hypot(ox - a.x, oy - a.y) < 9.0) and \
                    not any(g.owner is a and g.kind == "pylon" for g in m.gadgets) and len(allies_near) >= 2:
                self.w1 = True
            if r2:
                low = min((x for x in m.allies(a) if x.alive and x is not a and m.visible(a, x)
                           and math.hypot(x.x - a.x, x.y - a.y) < 22.0), key=lambda x: x.frac, default=None)
                if low is not None and low.frac < 0.55:
                    self._snap(*low.chest())
                    self.w2 = True
                elif a.frac < 0.4 and hurt:
                    a.pitch = -1.2
                    self.snap_t = 0.12
                    self.w2 = True
            if ru and len(allies_near) >= 3 and len(seen) >= 2:
                self.wu = True

    # --- every frame --------------------------------------------------------------------------------------------------
    def update(self, m, dt: float) -> None:  # noqa: ANN001
        a = self.a
        c = a.c
        self.think_t -= dt
        if self.think_t <= 0.0:
            self.think_t = 0.14 + self.rng.random() * 0.08
            self.think(m)
        if self.target is not None and not self.target.alive:
            self.target = None
        if self.heal_target is not None and not self.heal_target.alive:
            self.heal_target = None
        c.mf = c.ms = 0.0
        c.jump = c.fire = c.reload = c.alt = False
        self._steer(m, dt)
        self._aim(m, dt)
        c.a1, c.a2, c.ult, c.melee = self.w1, self.w2, self.wu, self.wm
        if self.w1 and a.hero.id == "wren":
            c.mf, c.ms = self.dash_fwd, 0.0
        self.w1 = self.w2 = self.wu = self.wm = False

    def _go(self, vx: float, vy: float) -> None:
        """Walk along the map direction (vx, vy), whichever way the hero is facing."""
        n = math.hypot(vx, vy)
        if n < 1e-6:
            return
        vx, vy = vx / n, vy / n
        a = self.a
        cy, sy = math.cos(a.yaw), math.sin(a.yaw)
        a.c.mf = vx * cy + vy * sy
        a.c.ms = -vx * sy + vy * cy
        self.look = (vx, vy)

    def _route(self, m, dt: float) -> bool:  # noqa: ANN001
        """Walk toward the goal along the map's routes; False when already there."""
        a = self.a
        if self.goal is None:
            return False
        key, gx, gy, gz = self.goal
        sx, sy = self.spot
        ds = math.hypot(sx - a.x, sy - a.y)
        if ds < 0.7:
            return False
        w = m.world
        if ds < 5.0 and abs(w.floor_at(sx, sy, a.z + 1.0) - a.z) < 1.2 and w.los(a.x, a.y, a.z + 0.5, sx, sy, a.z + 0.5):
            self._go(sx - a.x, sy - a.y)
            return True
        nxt = w.downhill(w.field(key, gx, gy, gz), a.x, a.y, a.z)
        if nxt is None:
            self._go(sx - a.x, sy - a.y)
            return True
        nx, ny, nz, left = nxt
        if left < 1.5:
            self._go(sx - a.x, sy - a.y)
            return True
        self._go(nx - a.x, ny - a.y)
        if nz > a.z + STEP - 0.05 and a.on_ground and math.hypot(nx - a.x, ny - a.y) < 1.5:
            a.c.jump = True
        return True

    def _steer(self, m, dt: float) -> None:  # noqa: ANN001
        a = self.a
        c = a.c
        if m.phase == "over":
            return
        if self.hold_t > 0.0:
            self.hold_t -= dt
            if self.target is None and m.time - a.hurt_t > 1.0:
                return
            self.hold_t = 0.0
        t = self.target
        moving = False
        if self.retreat or t is None or self.goal is None:
            moving = self._route(m, dt)
        else:
            key, gx, gy, gz = self.goal
            d = math.hypot(t.x - a.x, t.y - a.y) or 1.0
            pref = a.hero.pref_range
            far = math.hypot(self.spot[0] - a.x, self.spot[1] - a.y)
            leash = 5.0 if a.hero.role == "tank" else (9.0 if a.hero.role == "support" else 12.0 + pref * 0.4)
            if far > leash or (a.hero.role != "damage" and far > 2.5 and d > pref):
                moving = self._route(m, dt)  # the objective (or the tank) first
            else:
                ux, uy = (t.x - a.x) / d, (t.y - a.y) / d
                radial = 0.0
                if d > pref + 3.0 and far < leash - 2.0:
                    radial = 1.0
                elif d < pref - 3.0 and pref > 8.0:
                    radial = -1.0
                self.strafe_t -= dt
                if self.strafe_t <= 0.0 or a.blocked:
                    self.strafe = -self.strafe
                    self.strafe_t = self.rng.uniform(0.5, 1.5)
                pull_x, pull_y = (self.spot[0] - a.x) * 0.12, (self.spot[1] - a.y) * 0.12
                self._go(ux * radial - uy * self.strafe * 0.9 + pull_x, uy * radial + ux * self.strafe * 0.9 + pull_y)
                moving = True
                if d < 2.4 and abs(t.z - a.z) < 1.5 and self.rng.random() < 0.05:
                    self.wm = True
        # stuck: hop and sidestep
        if moving:
            self.stuck_t += dt
            if self.stuck_t > 0.45:
                if math.hypot(a.x - self.last[0], a.y - self.last[1]) < 0.3:
                    self.unstick = 0.5
                    self.strafe = -self.strafe
                self.last = (a.x, a.y)
                self.stuck_t = 0.0
        else:
            self.stuck_t = 0.0
            self.last = (a.x, a.y)
        if self.unstick > 0.0:
            self.unstick -= dt
            c.jump = a.on_ground
            lx, ly = self.look
            self._go(lx - ly * self.strafe, ly + lx * self.strafe)

    def _aim(self, m, dt: float) -> None:  # noqa: ANN001
        a = self.a
        c = a.c
        w = a.hero.weapon
        if self.snap_t > 0.0:
            self.snap_t -= dt
            return
        ex, ey, ez = a.eye_pos()
        t = self.heal_target or self.target
        healing = self.heal_target is not None
        if t is None or m.phase == "over":
            lx, ly = self.look
            dyaw = wrap(math.atan2(ly, lx) - a.yaw)
            a.yaw += max(min(dyaw * min(dt * 8.0, 1.0), self.turn * dt), -self.turn * dt)
            a.pitch += (0.0 - a.pitch) * min(dt * 6.0, 1.0)
            self.seen = 0.0
            if a.ammo < w.mag * 0.5 and a.reload_t <= 0.0:
                c.reload = True
            return
        tx, ty, tz = t.chest()
        if not healing and w.kind == "hitscan" and w.head > 1.6 and self.err < 0.04:
            tz = t.z + t.height * 0.84
        dh = math.hypot(tx - ex, ty - ey)
        d = math.sqrt(dh * dh + (tz - ez) ** 2)
        if w.kind == "proj" and not healing:  # lead the shot
            ft = d / w.speed
            tx, ty = tx + (t.vx + t.kx) * ft, ty + (t.vy + t.ky) * ft
            if w.gravity > 0.0:
                tz = t.z + 0.3
            dh = math.hypot(tx - ex, ty - ey)
        want_pitch = lob_pitch(dh, tz - ez, w.speed, w.gravity) if (w.kind == "proj" and w.gravity > 0.0) \
            else math.atan2(tz - ez, max(dh, 0.1))
        self.off_t -= dt
        if self.off_t <= 0.0:
            self.off_t = self.rng.uniform(0.22, 0.5)
            k = 1.0 + math.hypot(t.vx, t.vy) * 0.1
            self.off = (self.rng.gauss(0.0, self.err * k), self.rng.gauss(0.0, self.err * 0.6))
        dyaw = wrap(math.atan2(ty - ey, tx - ex) + self.off[0] - a.yaw)
        dpitch = want_pitch + self.off[1] - a.pitch
        step = self.turn * dt
        a.yaw += max(min(dyaw * min(dt * 14.0, 1.0), step), -step)
        a.pitch += max(min(dpitch * min(dt * 14.0, 1.0), step), -step)
        a.pitch = max(min(a.pitch, 1.4), -1.4)
        if not m.visible(a, t) or (not healing and t.protected):
            self.seen = max(self.seen - dt * 2.0, 0.0)
            return
        self.seen += dt
        if self.seen < self.react or m.phase != "live":
            return
        if w.kind == "cone":
            if d < w.rng + 0.5 and abs(dyaw) < 0.3:
                c.fire = True
            elif self.target is not None and healing is False and d < w.rng + 0.5:
                c.fire = True
        else:
            reach = w.far * 1.25 if w.far < 100 else w.rng
            tol = 0.05 + (t.hit_r + 0.25) / max(d, 1.0)
            if d < reach and abs(dyaw) < tol and abs(dpitch) < tol + 0.05:
                c.fire = True
                if a.hero.id == "flint" and d > 14.0:
                    c.alt = True
