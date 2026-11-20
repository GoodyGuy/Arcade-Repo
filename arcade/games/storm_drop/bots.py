"""The bots of Storm Drop. Each has a ``Brain`` that picks a landing spot, loots chests and floor finds, tops up
health and shield, harvests wood when short, walks ahead of the storm, and fights what it can see: it strafes, picks the
weapon that suits the range, fires in bursts with an aim error, throws up a wall when hit and ramps up to a target above
it. They drive the same controls and the same match verbs as the player."""

from __future__ import annotations

import math

from .data import BUILD_COST, SIZE, SLOTS, SUPPLIES, WEAPONS

SIGHT = 88.0
ENGAGE = {"close": 30.0, "mid": 60.0, "long": 88.0, "blast": 52.0, "tool": 0.0}
PREF = {"close": 7.0, "mid": 22.0, "long": 52.0, "blast": 26.0, "tool": 1.4}


class Brain:
    def __init__(self, match, actor) -> None:  # noqa: ANN001
        self.m, self.a = match, actor
        rng = match.rng
        err, react, _dmg, keen = match.diff
        self.skill = rng.uniform(0.75, 1.25)
        self.aim_err = err / self.skill
        self.react = react / self.skill
        self.keen = min(keen * self.skill, 0.97)
        self.think_t = rng.uniform(0.0, 0.3)
        self.goal = "roam"
        self.tx, self.tz = SIZE / 2, SIZE / 2
        self.arrive = 1.3
        self.target = None
        self.seen = self.first_seen = -9.0
        self.visible = False
        self.thing = None  # the chest / item / prop the goal is about
        self.skip: set[int] = set()  # things given up on
        self.strafe, self.strafe_t = 1.0, 0.0
        self.err_x = self.err_y = 0.0
        self.err_t = 0.0
        self.burst_t, self.rest_t = 0.0, 0.0
        self.build_cd = rng.uniform(0.0, 2.0)
        self.hold_t = 0.0
        self.push_t = 0.0
        self.lx = self.lz = 0.0
        self.stuck_acc, self.stuck_n = 0.0, 0
        self.detour_t, self.detour = 0.0, 0.0
        self.smash_t = 0.0
        self.jump_s = 200.0
        self.land_x, self.land_z = SIZE / 2, SIZE / 2
        self.roam_t = 0.0
        self.head_aim = False

    # --- the drop ---------------------------------------------------------------------------------------------------

    def plan_drop(self) -> None:
        m, rng = self.m, self.m.rng
        spots = [c[:3] for c in m.island.chest_spots] + list(m.island.floor_spots)
        ax, az = m.ship_a
        dx, dz = m.ship_dir
        claims = m.drop_claims
        for _ in range(30):
            x, _y, z = rng.choice(spots)
            s = (x - ax) * dx + (z - az) * dz
            side = abs((x - ax) * dz - (z - az) * dx)
            if side < 150 and m.ship_open + 4 < s < m.ship_end - 10                     and all((x - cx) ** 2 + (z - cz) ** 2 > 22 * 22 for cx, cz in claims):
                break
        else:
            s = rng.uniform(m.ship_open + 30, m.ship_end - 60)
            x, z = ax + dx * s, az + dz * s
        self.jump_s = max(s - rng.uniform(0, 30) - side * 0.12, m.ship_open + 1)
        ang = rng.uniform(0, math.tau)
        self.land_x, self.land_z = x + math.cos(ang) * 5.0, z + math.sin(ang) * 5.0
        claims.append((x, z))
        self.tx, self.tz = x, z

    # --- every frame --------------------------------------------------------------------------------------------------

    def update(self, dt: float) -> None:
        m, a = self.m, self.a
        if a.state == "ship":
            if m.ship_s >= self.jump_s:
                m.leave_ship(a)
            return
        if a.state != "ground":
            dx, dz = self.land_x - a.x, self.land_z - a.z
            d = math.hypot(dx, dz)
            if d > 2.5:
                a.wish_x, a.wish_z = dx / d, dz / d
                a.yaw = math.atan2(dx, dz)
                a.pitch = -0.5 if d > 60 else -0.15
            else:
                a.wish_x = a.wish_z = 0.0
            return
        self.think_t -= dt
        self.build_cd -= dt
        self.hold_t -= dt
        self.push_t -= dt
        if self.think_t <= 0:
            self.think_t = 0.2 + m.rng.random() * 0.14
            self.think()
        a.firing = False
        a.crouch = False
        if self.smash_t > 0:
            self.smash_t -= dt
            self._smash()
        elif self.goal == "fight":
            self._fight(dt)
        elif self.goal == "heal":
            self._heal()
        elif self.goal == "harvest":
            self._harvest(dt)
        else:
            self._travel(dt, shoot=True)

    # --- deciding -------------------------------------------------------------------------------------------------------

    def _safe_spot(self) -> tuple[float, float] | None:
        """Where to go to be out of the storm's way, or None when there is no hurry."""
        m, a = self.m, self.a
        s = m.storm
        if s["state"] == "idle":
            return None
        cur = math.hypot(a.x - s["cx"], a.z - s["cz"]) - s["r"]
        dn = math.hypot(a.x - s["ncx"], a.z - s["ncz"])
        out_next = dn - s["nr"]
        hurry = cur > -4.0
        if out_next > -5.0:
            need = out_next / 6.0 + 10.0
            left = s["t"] + (0.0 if s["state"] == "shrink" else STORM_GRACE)
            if s["state"] == "shrink" or left < need:
                hurry = True
        if not hurry:
            return None
        k = max(s["nr"] * 0.55, 1.0)
        if dn < 1e-3:
            return s["ncx"], s["ncz"]
        j = (a.id * 0.7) % 1.0 - 0.5  # each bot keeps its own lane into the eye
        ux, uz = (a.x - s["ncx"]) / dn, (a.z - s["ncz"]) / dn
        ca, sa = math.cos(j), math.sin(j)
        return s["ncx"] + (ux * ca - uz * sa) * k, s["ncz"] + (ux * sa + uz * ca) * k

    def _inside_next(self, x: float, z: float) -> bool:
        s = self.m.storm
        if s["state"] == "idle":
            return True
        return math.hypot(x - s["ncx"], z - s["ncz"]) < s["nr"] - 3.0 and math.hypot(x - s["cx"], z - s["cz"]) < s["r"]

    def think(self) -> None:
        m, a, rng = self.m, self.a, self.m.rng
        now = m.time
        # who can I see?
        tgt = self.target
        if tgt is not None and not tgt.alive:
            tgt = self.target = None
        ex, ey, ez = a.eye()
        near = []
        for b in m.actors:
            if b.alive and b is not a and b.state != "ship":
                d2 = (b.x - a.x) ** 2 + (b.z - a.z) ** 2
                if d2 < SIGHT * SIGHT and (b.state == "ground" or d2 < 1600):
                    near.append((d2, b.id, b))
        near.sort()
        self.visible = False
        for d2, _i, b in near[:3]:
            if m.structs.clear_line(ex, ey, ez, b.x, b.y + 1.15, b.z):
                if b is not tgt:
                    self.first_seen = now
                tgt = self.target = b
                self.seen = now
                self.visible = True
                break
        if not self.visible:
            if a.attacker is not None and a.attacker.alive and now - a.attacked_t < 1.2 and a.attacker is not tgt:
                tgt = self.target = a.attacker
                self.seen = self.first_seen = now
            elif tgt is not None and now - self.seen > 3.5:
                tgt = self.target = None
        # the small stuff underfoot
        for it in m.items_near(a.x, a.z, 2.4):
            if it.alive and abs(it.y - a.y) < 2.2 and self._wants(it):
                m.give(a, it)
        self._pick_weapon(math.hypot(tgt.x - a.x, tgt.z - a.z) if tgt is not None else 20.0)
        armed = a.has_gun()
        if tgt is None:
            s = a.held()
            if s is not None and s.is_weapon and s.mag < WEAPONS[s.id].mag * 0.5:
                m.reload(a)
        # what now?
        safe = self._safe_spot()
        hurt = a.hp < 62 or a.shield < 38
        if safe is not None:
            self.goal = "rotate"
            self.tx, self.tz = safe
            self.arrive = 3.0
            self.thing = None
            return
        if self.hold_t > 0 and hurt and self._supply_slot() >= 0:
            self.goal = "heal"
            return
        if tgt is not None:
            d = math.hypot(tgt.x - a.x, tgt.z - a.z)
            s = a.held()
            cls = WEAPONS[s.id].cls if (s is not None and s.is_weapon) else "tool"
            calm = 0.5 if tgt.bot else 1.0  # bots leave each other be at range
            hit = now - a.attacked_t < 2.5
            if tgt.bot and m.truce:  # the match is ahead of its pace: bots break off from each other
                fight = False
            else:
                fight = d < 4.5 or (armed and (d < ENGAGE[cls] * calm or hit))
            if fight:
                self.goal = "fight"
                return
        if tgt is None and hurt and self._supply_slot() >= 0:
            self.goal = "heal"
            return
        if self.goal in ("loot", "harvest", "roam") and self.thing is not None and self._thing_ok():
            return  # carry on with it
        self.thing = None
        if self._wants_loot() or tgt is not None:
            best, bd = None, 1e9
            for it in m.items_near(a.x, a.z, 26.0):
                if it.alive and id(it) not in self.skip and self._wants(it) and self._inside_next(it.x, it.z):
                    d = (it.x - a.x) ** 2 + (it.z - a.z) ** 2 + (it.y - a.y) ** 2 * 6
                    if d < bd:
                        best, bd = it, d
            if best is None:
                for c in m.chests:
                    if not c.opened and id(c) not in self.skip:
                        d = (c.x - a.x) ** 2 + (c.z - a.z) ** 2 + (c.y - a.y) ** 2 * 6
                        if d < bd and d < 110 * 110 and self._inside_next(c.x, c.z):
                            best, bd = c, d
            if best is not None:
                self.goal, self.thing = "loot", best
                self.tx, self.tz, self.arrive = best.x, best.z, 1.5
                return
        if sum(a.mats) < 50:
            best, bd = None, 30.0 * 30.0
            cx, cz = int(a.x // 4), int(a.z // 4)
            for ix in range(cx - 6, cx + 7, 2):
                for iz in range(cz - 6, cz + 7, 2):
                    for p in m.island.prop_cols.get((ix, iz), ()):
                        if p.alive and id(p) not in self.skip and p.kind in ("pine", "oak", "palm", "crate", "logs"):
                            d = (p.x - a.x) ** 2 + (p.z - a.z) ** 2
                            if d < bd and self._inside_next(p.x, p.z):
                                best, bd = p, d
            if best is not None:
                self.goal, self.thing = "harvest", best
                self.tx, self.tz, self.arrive = best.x, best.z, best.a + 1.25
                return
        self.roam_t -= 0.3
        if self.goal != "roam" or self.roam_t <= 0 or math.hypot(self.tx - a.x, self.tz - a.z) < 4.0:
            s = m.storm
            self.goal = "roam"
            self.roam_t = rng.uniform(8.0, 16.0)
            self.arrive = 3.0
            for _ in range(8):
                if s["state"] == "idle":
                    ang, d = rng.uniform(0, math.tau), rng.uniform(15, 70)
                    x, z = a.x + math.cos(ang) * d, a.z + math.sin(ang) * d
                else:
                    ang, d = rng.uniform(0, math.tau), s["nr"] * 0.8 * math.sqrt(rng.random())
                    x, z = s["ncx"] + math.cos(ang) * d, s["ncz"] + math.sin(ang) * d
                if m.island.on_land(x, z):
                    self.tx, self.tz = x, z
                    break

    def _thing_ok(self) -> bool:
        t = self.thing
        if self.goal == "loot":
            ok = (not t.opened) if hasattr(t, "opened") else (t.alive and self._wants(t))
            return ok and self._inside_next(t.x, t.z)
        if self.goal == "harvest":
            return t.alive and sum(self.a.mats) < 110
        return False

    def _wants_loot(self) -> bool:
        a = self.a
        guns = sum(1 for s in a.inv if s is not None and s.is_weapon)
        return guns < 2 or a.shield < 50 or sum(1 for s in a.inv if s is None) >= 2

    def _wants(self, it) -> bool:  # noqa: ANN001
        a = self.a
        if it.kind == "mats":
            return a.mats[it.id] < 400
        if it.kind == "ammo":
            return a.ammo[it.id] < 150
        free = any(s is None for s in a.inv)
        if it.kind == "supply":
            sup = SUPPLIES[it.id]
            return free or any(s is not None and s.id == it.id and s.count < sup.stack for s in a.inv)
        if not free:
            return False
        have = [s for s in a.inv if s is not None and s.id == it.id]
        return not have or it.grade > max(s.grade for s in have)

    def _supply_slot(self) -> int:
        a = self.a
        best, score = -1, 0.0
        for i, s in enumerate(a.inv):
            if s is None or s.is_weapon:
                continue
            sup = SUPPLIES[s.id]
            cur = a.hp if sup.heals == "hp" else a.shield
            gain = min(cur + sup.amount, sup.cap) - cur
            if gain <= 0:
                continue
            sc = gain / sup.time + (4.0 if sup.heals == "hp" and a.hp < 45 else 0.0)
            if sc > score:
                best, score = i, sc
        return best

    def _pick_weapon(self, dist: float) -> None:
        a = self.a
        if a.reload_t > 0 or a.use_t > 0:
            return
        best, score = -1, 0.0
        for i, s in enumerate(a.inv):
            if s is None or not s.is_weapon:
                continue
            w = WEAPONS[s.id]
            if s.mag <= 0 and a.ammo[w.ammo] <= 0:
                continue
            c = w.cls
            if c == "close":
                sc = 10.0 if dist < 11 else (5.0 if dist < 20 else 1.0)
            elif c == "mid":
                sc = 8.0 if 7 < dist < 75 else 4.0
            elif c == "long":
                sc = 10.0 if dist > 45 else (5.0 if dist > 24 else 1.5)
            else:
                sc = 6.5 if 13 < dist < 60 else 0.0
            sc += s.grade * 0.5 + (0.0 if s.mag > 0 else -2.5)
            if sc > score:
                best, score = i, sc
        if best != a.sel and (best >= 0 or (a.held() is not None)):
            self.m.select(a, best)

    # --- doing ----------------------------------------------------------------------------------------------------------

    def _turn(self, yaw: float, dt: float, rate: float = 7.0) -> float:
        a = self.a
        d = (yaw - a.yaw + math.pi) % math.tau - math.pi
        step = rate * dt
        a.yaw += d if abs(d) < step else math.copysign(step, d)
        return abs(d)

    def _travel(self, dt: float, shoot: bool = False) -> None:
        m, a = self.m, self.a
        dx, dz = self.tx - a.x, self.tz - a.z
        d = math.hypot(dx, dz)
        tgt = self.target
        if d < self.arrive:
            a.wish_x = a.wish_z = 0.0
            self._arrive()
        else:
            ux, uz = dx / d, dz / d
            if self.detour_t > 0:
                self.detour_t -= dt
                ca, sa = math.cos(self.detour), math.sin(self.detour)
                ux, uz = ux * ca - uz * sa, ux * sa + uz * ca
            a.wish_x, a.wish_z = ux, uz
            a.sprint = d > 5.0
            self._unstick(dt, ux, uz)
        if shoot and tgt is not None and self.visible and a.has_gun() \
                and (tgt.x - a.x) ** 2 + (tgt.z - a.z) ** 2 < 50 * 50:
            self._aim_fire(dt, tgt)
        elif d >= self.arrive:
            self._turn(math.atan2(a.wish_x, a.wish_z), dt)
            a.pitch *= 0.9
            a.aim = None

    def _unstick(self, dt: float, ux: float, uz: float) -> None:
        m, a = self.m, self.a
        self.stuck_acc += dt
        if self.stuck_acc < 0.55:
            return
        self.stuck_acc = 0.0
        moved = math.hypot(a.x - self.lx, a.z - self.lz)
        self.lx, self.lz = a.x, a.z
        if moved > 0.9:
            self.stuck_n = 0
            return
        self.stuck_n += 1
        n = self.stuck_n
        rng = m.rng
        if n == 1:
            a.want_jump = True
        elif n == 2:
            self.detour, self.detour_t = rng.choice((-1.3, 1.3)), 1.2
        elif n == 3:
            ex, ey, ez = a.x, a.y + 0.9, a.z
            t, what, _obj = m.structs.ray(ex, ey, ez, ux, 0.0, uz, 2.4)
            if what in ("piece", "prop"):
                self.smash_t = 1.7
                self.smash_dir = (ux, uz)
            else:
                self.detour, self.detour_t = rng.choice((-1.7, 1.7)), 1.6
                a.want_jump = True
        elif n == 4:
            if sum(a.mats) >= BUILD_COST:
                yaw = math.atan2(ux, uz)
                m.try_build(a, *m.build_target(a, 2, yaw, 0.0))
            self.detour, self.detour_t = rng.choice((-0.9, 0.9)), 1.0
        else:
            self.stuck_n = 0
            if self.thing is not None:
                self.skip.add(id(self.thing))
                self.thing = None
            self.goal = "roam"
            self.roam_t = 0.0
            self.detour, self.detour_t = rng.uniform(2.0, 4.2), 2.0

    def _arrive(self) -> None:
        m, a = self.m, self.a
        t = self.thing
        if self.goal == "loot" and t is not None:
            if abs(t.y - a.y) > 2.4:  # it is on another floor: give up on it for now
                self.skip.add(id(t))
            elif hasattr(t, "opened"):
                m.open_chest(t, a)
            elif t.alive:
                if not m.give(a, t):
                    self.skip.add(id(t))
            self.thing = None
        elif self.goal == "rotate":
            self.goal = "roam"
            self.roam_t = 0.0

    def _smash(self) -> None:
        a = self.a
        ux, uz = getattr(self, "smash_dir", (math.sin(a.yaw), math.cos(a.yaw)))
        a.yaw = math.atan2(ux, uz)
        a.pitch = 0.0
        a.wish_x, a.wish_z = ux * 0.3, uz * 0.3
        if a.sel != -1:
            self.m.select(a, -1)
        a.aim = (a.x + ux * 3, a.y + 1.0, a.z + uz * 3)
        a.firing = True

    def _harvest(self, dt: float) -> None:
        a = self.a
        p = self.thing
        if p is None or not p.alive or sum(a.mats) >= 110:
            self.goal, self.thing = "roam", None
            self.roam_t = 0.0
            return
        d = math.hypot(p.x - a.x, p.z - a.z)
        if d > self.arrive + 0.2:
            self._travel(dt)
            return
        a.wish_x = a.wish_z = 0.0
        self._turn(math.atan2(p.x - a.x, p.z - a.z), dt)
        if a.sel != -1:
            self.m.select(a, -1)
        a.aim = (p.x, a.y + 1.0, p.z)
        a.firing = True

    def _heal(self) -> None:
        m, a = self.m, self.a
        a.wish_x = a.wish_z = 0.0
        a.crouch = True
        a.aim = None
        if a.use_t > 0:
            return
        i = self._supply_slot()
        if i < 0:
            self.goal = "roam"
            self.roam_t = 0.0
            return
        if a.sel != i:
            m.select(a, i)
        m.use(a)

    def _aim_fire(self, dt: float, t) -> float:  # noqa: ANN001
        """Turn to the target, set the aim point (with this bot's error) and hold the trigger in bursts."""
        m, a = self.m, self.a
        rng = m.rng
        dx, dz = t.x - a.x, t.z - a.z
        dist = math.hypot(dx, dz) or 0.01
        off = self._turn(math.atan2(dx, dz), dt, 9.0)
        self.err_t -= dt
        if self.err_t <= 0:
            self.err_t = rng.uniform(0.25, 0.5)
            k = self.aim_err * (1.0 + 0.5 * min(t.speed / 7.0, 1.3)) * (1.0 + 0.4 * min(a.speed / 7.0, 1.0))
            if t.bot:
                k *= 1.8  # bots are kinder to each other, so more of them are left for the player to meet
            self.err_x, self.err_y = rng.gauss(0, k), rng.gauss(0, k * 0.7)
            self.head_aim = rng.random() < 0.18 * self.skill
        ty = t.y + (1.55 if self.head_aim else 1.05) - (0.4 if t.crouch else 0.0)
        s = a.held()
        w = WEAPONS[s.id] if s is not None and s.is_weapon else None
        if w is not None and w.drop > 0 and w.speed > 0:
            ty += 0.5 * w.drop * (dist / w.speed) ** 2
        px, pz = dz / dist, -dx / dist
        a.aim = (t.x + px * self.err_x * dist, ty + self.err_y * dist, t.z + pz * self.err_x * dist)
        a.pitch = math.atan2(ty - (a.y + 1.5), dist)
        ready = self.visible and off < 0.3 and m.time - self.first_seen > self.react
        if w is not None and w.blast > 0 and dist < 9:
            ready = False
        if ready:
            if self.rest_t > 0:
                self.rest_t -= dt
            else:
                a.firing = True
                self.burst_t -= dt
                if self.burst_t <= 0:
                    auto = w is not None and w.auto
                    self.burst_t = rng.uniform(0.5, 1.1) if auto else rng.uniform(0.2, 0.6)
                    self.rest_t = rng.uniform(0.25, 0.7) / self.skill
        return dist

    def _fight(self, dt: float) -> None:
        m, a, rng = self.m, self.a, self.m.rng
        t = self.target
        if t is None or not t.alive:
            self.goal = "roam"
            self.roam_t = 0.0
            return
        dist = self._aim_fire(dt, t)
        s = a.held()
        w = WEAPONS[s.id] if s is not None and s.is_weapon else WEAPONS["sledge"]
        if s is not None and not s.is_weapon:
            w = WEAPONS["sledge"]
        if w.cls == "tool":
            if a.sel != -1 and not a.has_gun():
                m.select(a, -1)
            a.firing = dist < 2.6
        pref = PREF[w.cls]
        ux, uz = (t.x - a.x) / dist, (t.z - a.z) / dist
        radial = 1.0 if dist > pref * 1.25 else (-1.0 if dist < pref * 0.55 else 0.0)
        if not self.visible:
            radial = 1.0  # go and look
        if self.push_t > 0:
            radial, side = 1.0, 0.0
        else:
            self.strafe_t -= dt
            if self.strafe_t <= 0:
                self.strafe_t = rng.uniform(0.6, 1.5)
                self.strafe = rng.choice((-1.0, 1.0)) if rng.random() < 0.85 else 0.0
                if dist < 12 and rng.random() < 0.3 * self.skill:
                    a.want_jump = True
            side = self.strafe * 0.85
        wx, wz = ux * radial + uz * side, uz * radial - ux * side
        n = math.hypot(wx, wz)
        if n > 1e-3:
            a.wish_x, a.wish_z = wx / n, wz / n
        else:
            a.wish_x = a.wish_z = 0.0
        a.sprint = radial > 0 and dist > pref * 2.0
        if radial > 0 and not self.visible:
            self._unstick(dt, ux, uz)
        # building under fire: a wall towards whoever hit me, a ramp up to a target above me
        if self.build_cd <= 0 and sum(a.mats) >= BUILD_COST:
            now = m.time
            if now - a.hurt_t < 0.5 and a.attacker is not None and a.attacker.alive and rng.random() < self.keen:
                att = a.attacker
                yaw = math.atan2(att.x - a.x, att.z - a.z)
                if m.try_build(a, *m.build_target(a, 0, yaw, 0.0)):
                    self.build_cd = rng.uniform(2.2, 4.5)
                    if (a.hp < 62 or a.shield < 30) and self._supply_slot() >= 0:
                        self.hold_t = 5.0
                    if rng.random() < 0.35 * self.keen:  # and box in a side as well
                        m.try_build(a, *m.build_target(a, 0, yaw + math.pi / 2, 0.0))
                else:
                    self.build_cd = 0.8
            elif t.y - a.y > 2.3 and 4.0 < dist < 30.0 and rng.random() < self.keen:
                yaw = math.atan2(ux, uz)
                if m.try_build(a, *m.build_target(a, 2, yaw, 0.2)):
                    self.push_t = 1.1
                    self.build_cd = 0.75
                else:
                    self.build_cd = 1.5


STORM_GRACE = 30.0  # s of the shrink a bot counts on when the eye is only marked
__all__ = ["Brain", "SLOTS"]
