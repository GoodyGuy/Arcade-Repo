"""The bots of BREACH POINT. ``TeamAI`` is a side's shared head: what the team has seen and heard, which site the
attack goes to and when, who holds what on defence, who rotates, who defuses. ``Brain`` is one bot: it buys, walks the
map's distance fields, holds an angle, reacts after a human delay with imperfect aim, fires in bursts, uses its agent's
abilities and plants or defuses."""

from __future__ import annotations

import math

from . import config as C
from .config import WEAPONS
from .maps import FAR
from .sim import DEG, Actor, Cmd

TWO_PI = 2 * math.pi
LOBBED = ("flash", "frag", "smoke", "slow", "fire", "static", "sonar", "concuss")


def _wrap(a: float) -> float:
    return (a + math.pi) % TWO_PI - math.pi


class TeamAI:
    def __init__(self, match, squad: int) -> None:  # noqa: ANN001
        self.m = match
        self.squad = squad
        self.side = match.sides[squad]
        self.rng = match.rng
        self.intel: dict[int, list] = {}  # enemy index -> [x, y, time, seen now]
        self.sees: dict[int, list] = {}  # member index -> enemies in its sight
        self.scan_t = 0.0
        self.plan_t = 0.0
        self.live_t = 0.0
        self.spots: dict[int, object] = {}
        self.assign: dict[int, str] = {}
        self.defuser: Actor | None = None
        self.fetcher: Actor | None = None
        self.plant_heard: tuple | None = None
        self.defuse_heard = -99.0
        self.call: str | None = None  # the player's call: a site
        self.call_t = -99.0
        self.switched = False
        self.execute = False
        self.quiet_t = 0.0
        mp = match.map
        keys = list(mp.sites)
        self.site = self.rng.choice(keys)
        self.go_time = self.rng.choice((2.0, 8.0, 12.0, 16.0, 22.0))
        self.stealth = self.rng.random() < 0.35
        self.ways: dict[int, int] = {}
        self.plant_spot: tuple | None = None
        self._plan_round()

    # --- plans --------------------------------------------------------------------------------------------------

    def members(self, alive: bool = True) -> list:
        return [a for a in self.m.actors if a.squad == self.squad and (a.alive or not alive)]

    def bots(self) -> list:
        return [a for a in self.members() if a.brain is not None and (not a.human or self.m.autopilot)]

    def _plan_round(self) -> None:
        mp = self.m.map
        mem = self.members(alive=False)
        if self.side == "atk":
            self._plan_attack()
        else:
            order = ["A", "B", "A", "B", "mid"]
            bots = [a for a in mem if not a.human or self.m.autopilot]
            bots.sort(key=lambda a: (a.agent.role != "Anchor", a.agent.role == "Recon", a.idx))
            used: dict[str, int] = {}
            for a, k in zip(bots, order, strict=False):
                self.assign[a.idx] = k
                if k != "mid":
                    site = mp.sites[k]
                    i = used.get(k, 0)
                    used[k] = i + 1
                    pool = site.holds
                    if pool:
                        self.spots[a.idx] = pool[(i * 2 + self.rng.randrange(2)) % len(pool)]

    def _plan_attack(self) -> None:
        mp = self.m.map
        site = mp.sites[self.site]
        n = len(site.attack)
        main = self.rng.randrange(n)
        split = self.rng.random() < 0.45
        for i, a in enumerate(self.members(alive=False)):
            self.ways[a.idx] = (main + 1) % n if (split and i % 5 in (3, 4)) else main
        pool = list(site.posts) + list(site.holds)
        self.rng.shuffle(pool)
        self.spots = {}
        for i, a in enumerate(self.members(alive=False)):
            if pool:
                self.spots[a.idx] = pool[i % len(pool)]
        self.plant_spot = self.rng.choice(site.plants[:4])

    def set_call(self, site: str) -> None:
        """The player asks the team to go to a site."""
        self.call, self.call_t = site, self.m.time
        if self.side == "atk" and self.m.charge["state"] != "planted" and site != self.site:
            self.site = site
            self._plan_attack()
        elif self.side == "def":
            self._rotate(site, keep=1)

    # --- knowing ------------------------------------------------------------------------------------------------

    def on_kill(self, victim: Actor, killer: Actor | None) -> None:
        if victim.squad == self.squad:
            if killer is not None and killer.squad != self.squad:
                self.intel[killer.idx] = [killer.x, killer.y, self.m.time, False]
            if self.defuser is victim:
                self.defuser = None
        else:
            self.intel.pop(victim.idx, None)

    def on_plant(self, site: str) -> None:
        mp = self.m.map
        s = mp.sites[site]
        if self.side == "atk":
            pool = sorted(s.posts, key=lambda sp: self.rng.random())
            for i, a in enumerate(self.members()):
                if pool:
                    self.spots[a.idx] = pool[i % len(pool)]
        else:
            self.spots = {}
            for a in self.members():
                self.assign[a.idx] = site

    def known(self, a: Actor, max_age: float = 3.0, max_dist: float = 40.0) -> tuple | None:
        """The nearest enemy position the team knows of that ``a`` cannot see right now."""
        best = None
        now = self.m.time
        for idx, (x, y, t, _seen) in self.intel.items():
            if now - t > max_age or not self.m.actors[idx].alive:
                continue
            d = math.hypot(x - a.x, y - a.y)
            if d < max_dist and (best is None or d < best[2]):
                best = (x, y, d, idx)
        return best

    def update(self, dt: float) -> None:
        m = self.m
        now = m.time
        self.scan_t -= dt
        if self.scan_t <= 0:
            self.scan_t = 0.12
            foes = [b for b in m.actors if b.squad != self.squad and b.alive]
            self.sees = {}
            for b in foes:
                it = self.intel.get(b.idx)
                if it is not None:
                    it[3] = False
            for a in self.members():
                row = []
                for b in foes:
                    d2 = (b.x - a.x) ** 2 + (b.y - a.y) ** 2
                    if d2 > 70 * 70:
                        continue
                    if (d2 < 9.0 or m._in_fov(a, b, 1.08)) and m.can_see(a, b):
                        row.append(b)
                        self.intel[b.idx] = [b.x, b.y, now, True]
                self.sees[a.idx] = row
            for b in foes:
                if b.revealed > 0:
                    self.intel[b.idx] = [b.x, b.y, now, self.intel.get(b.idx, [0, 0, 0, False])[3]]
        for x, y, r, squad, kind, who in m.heard:
            if squad == self.squad or not who.alive:
                continue
            for a in self.members():
                if (a.x - x) ** 2 + (a.y - y) ** 2 < r * r:
                    it = self.intel.get(who.idx)
                    if it is None or not it[3]:
                        j = 1.5 if kind == "step" else 0.6
                        self.intel[who.idx] = [x + self.rng.uniform(-j, j), y + self.rng.uniform(-j, j), now, False]
                    if kind == "plant":
                        self.plant_heard = (x, y, now)
                    elif kind == "defuse":
                        self.defuse_heard = now
                    break
        if m.phase not in ("live", "planted"):
            return
        self.live_t += dt
        self.plan_t -= dt
        if self.plan_t > 0:
            return
        self.plan_t = 0.5
        if self.side == "atk":
            self._think_attack()
        else:
            self._think_defence()

    def _think_attack(self) -> None:
        m, mp = self.m, self.m.map
        ch = m.charge
        mem = self.members()
        if not mem:
            return
        if ch["state"] == "dropped":
            bots = self.bots()
            if bots:
                self.fetcher = min(bots, key=lambda a: mp.path_len(a.x, a.y, ch["x"], ch["y"]))
        else:
            self.fetcher = None
        if ch["state"] == "planted":
            return
        # the team follows a human who goes the other way
        human = next((a for a in mem if a.human and not m.autopilot), None)
        if human is not None and m.time - self.call_t > 20.0:
            dists = {k: mp.path_len(human.x, human.y, *s.centre) for k, s in mp.sites.items()}
            other = min(dists, key=dists.get)
            if other != self.site and dists[other] + 12 < dists[self.site] and dists[other] < 28:
                self.site = other
                self._plan_attack()
        contact = any(self.sees.get(a.idx) for a in mem)
        if not self.execute:
            if self.live_t > self.go_time or m.t < 42.0 or (contact and self.live_t > 4.0):
                self.execute = True
        elif (not self.switched and m.t > 48.0 and len(mem) <= 3 and len(self.m.enemies(mem[0])) >= len(mem)
              and not any(mp.site_at(a.x, a.y) for a in mem) and self.rng.random() < 0.3):
            self.switched = True  # the push died at the door: try the other site
            self.site = next(k for k in mp.sites if k != self.site)
            self._plan_attack()
            self.execute = False
            self.go_time = self.live_t + 9.0

    def _think_defence(self) -> None:
        m, mp = self.m, self.m.map
        ch = m.charge
        now = m.time
        if ch["state"] == "planted":
            bots = self.bots()
            self.defuser = min(bots, key=lambda a: mp.path_len(a.x, a.y, ch["x"], ch["y"])) if bots else None
            return
        threat = {k: 0.0 for k in mp.sites}
        for idx, (x, y, t, _seen) in self.intel.items():
            b = m.actors[idx]
            if now - t > 6.0 or not b.alive:
                continue
            d = {k: mp.path_len(x, y, *s.centre) for k, s in mp.sites.items()}
            k = min(d, key=d.get)
            if d[k] < 34 and abs(d[k] - max(d.values())) > 6:
                threat[k] += 2.0 if b.has_charge else 1.0
        if self.plant_heard and now - self.plant_heard[2] < 5.0:
            k = mp.site_at(self.plant_heard[0], self.plant_heard[1])
            if k:
                threat[k] += 4.0
        hot = max(threat, key=threat.get)
        cold = sum(v for k, v in threat.items() if k != hot)
        if threat[hot] >= 1.0 and cold < 1.0:
            self.quiet_t = 0.0
            if threat[hot] >= 2.0:
                self._rotate(hot, keep=1 if threat[hot] < 3.0 else 0)
            else:  # one seen: the spare goes to look
                self._rotate(hot, keep=9)
        else:
            self.quiet_t += 0.5
            if self.quiet_t > 14.0 and now - self.call_t > 20.0:
                self.quiet_t = 0.0
                self.assign = {}
                self.spots = {}
                self._plan_round()

    def _rotate(self, site: str, keep: int) -> None:
        """Send the spare and the other site's defenders to ``site`` (all but ``keep`` of them)."""
        mp = self.m.map
        away = [a for a in self.bots() if self.assign.get(a.idx) not in (site, None)]
        away.sort(key=lambda a: (self.assign.get(a.idx) != "mid", mp.path_len(a.x, a.y, *mp.sites[site].centre)))
        stay = [a for a in away if self.assign.get(a.idx) != "mid"]
        movers = [a for a in away if self.assign.get(a.idx) == "mid"] + stay[:max(len(stay) - keep, 0)]
        holds = mp.sites[site].holds
        for a in movers:
            self.assign[a.idx] = site
            taken = {id(s) for i, s in self.spots.items() if i != a.idx}
            free = [s for s in holds if id(s) not in taken]
            if free:
                self.spots[a.idx] = min(free, key=lambda s: math.hypot(s.x - a.x, s.y - a.y))

    # --- what one bot should do ---------------------------------------------------------------------------------

    def task(self, a: Actor) -> dict:
        m, mp = self.m, self.m.map
        ch = m.charge
        if self.side == "atk":
            site = mp.sites[self.site]
            if m.phase == "buy":
                return {"goal": None, "watch": None}
            if ch["state"] == "dropped" and self.fetcher is a:
                return {"goal": (ch["x"], ch["y"]), "watch": None, "run": True}
            if ch["state"] == "planted":
                alarm = m.time - self.defuse_heard < 4.0 or ch.get("defuser") is not None
                sp = self.spots.get(a.idx)
                if alarm or sp is None:
                    return {"goal": (ch["x"], ch["y"]), "watch": (ch["x"], ch["y"]), "near": 5.0, "run": True}
                return {"goal": (sp.x, sp.y), "watch": (sp.wx, sp.wy), "hold": True}
            way = site.attack[self.ways.get(a.idx, 0) % len(site.attack)]
            if not self.execute:
                j = (a.idx % 5 - 2) * 0.8
                return {"goal": (way[3] + j * 0.4, way[4] + abs(j)), "watch": (way[1], way[2]), "hold": True,
                        "walk": self.stealth}
            if a.has_charge:
                others = [b for b in self.members() if b is not a]
                ahead = any(site.inside(b.x, b.y) for b in others)
                hot = any(m.time - t < 2.5 and math.hypot(x - self.plant_spot[0], y - self.plant_spot[1]) < 14
                          for x, y, t, _s in self.intel.values())
                if ahead or not others or m.t < 30.0 or (site.inside(a.x, a.y) and not hot):
                    return {"goal": self.plant_spot, "watch": None, "act": "plant", "run": True}
                return {"goal": (way[1], way[2]), "watch": site.centre, "hold": True}
            sp = self.spots.get(a.idx)
            if sp is None:
                return {"goal": site.centre, "watch": None}
            return {"goal": (sp.x, sp.y), "watch": (sp.wx, sp.wy), "hold": True, "enter": True}
        # defence
        if ch["state"] == "planted":
            cx, cy = ch["x"], ch["y"]
            if self.defuser is a or not self.m.team(1 - self.squad):
                return {"goal": (cx, cy), "watch": None, "act": "defuse", "run": True}
            return {"goal": (cx, cy), "watch": (cx, cy), "near": 6.0, "run": True}
        k = self.assign.get(a.idx, "mid")
        if k == "mid":
            return {"goal": mp.mid, "watch": (mp.mid[0], mp.mid[1] + 12.0), "hold": True}
        sp = self.spots.get(a.idx)
        if sp is None:
            return {"goal": mp.sites[k].centre, "watch": None, "hold": True}
        return {"goal": (sp.x, sp.y), "watch": (sp.wx, sp.wy), "hold": True}


class Brain:
    def __init__(self, match, actor: Actor) -> None:  # noqa: ANN001
        self.m, self.a = match, actor
        self.rng = match.rng
        skill = match.difficulty if actor.squad == 1 else "normal"
        self.D = C.DIFFICULTY[skill]
        self.t = self.rng.random()
        self.new_round()

    def new_round(self) -> None:
        r = self.rng
        self.bought = False
        self.buy_delay = 1.0 + r.random() * 5.0
        self.target: Actor | None = None
        self.react_at = 0.0
        self.err = [0.0, 0.0]
        self.aim_head = False
        self.burst = 0
        self._shots0 = 0
        self.pause = 0.0
        self.strafe = 0.0
        self.strafe_t = 0.0
        self.crouch_fight = False
        self.last_seen: tuple | None = None
        self.alert: tuple | None = None
        self.goal_cell: tuple | None = None
        self.field = None
        self.steer_t = 0.0
        self.point: tuple | None = None
        self.stuck_t = 0.0
        self.stuck_pos = (0.0, 0.0)
        self.unstick = 0.0
        self.util_t = 2.0 + r.random() * 3.0
        self.used: set = set()
        self.scan = r.random() * 6.0
        self.tap = False
        self.arrived = False
        self.pick_t = 0.0

    def hurt(self, src: Actor) -> None:
        self.alert = (src.x, src.y, self.t)
        ai = self.m.team_ai.get(self.a.squad)
        if ai is not None and src.squad != self.a.squad:
            it = ai.intel.get(src.idx)
            if it is None or not it[3]:
                ai.intel[src.idx] = [src.x, src.y, self.m.time, False]

    # --- buying -------------------------------------------------------------------------------------------------

    def buy(self) -> None:
        m, a, r = self.m, self.a, self.rng
        cr = a.credits
        role = a.agent.role
        full = 3900

        def get(item: str) -> bool:
            return m.buy(a, item)

        if a.main is None:
            if cr >= full:
                pick = "kestrel" if r.random() < 0.5 else "mistral"
                x = r.random()
                if role == "Anchor" and cr >= 5900 and x < 0.3 and a.side == "def":
                    pick = "longbow"
                elif x < 0.1:
                    pick = "sentry"
                elif x < 0.17 and cr >= 4300:
                    pick = "bastion"
                get(pick)
                get("heavy") or get("light")
            elif cr >= 2700 and (m.streak[a.squad] >= 2 or cr + C.CREDIT_LOSS[0] < full or r.random() < 0.3):
                get(r.choice(("lynx", "lynx", "wasp", "brace", "lancer")))
                get("light")
            elif cr <= 900:  # the pistol round
                if r.random() < 0.5:
                    get("vesper")
                else:
                    get("light")
            elif cr >= 1700 and cr + C.CREDIT_LOSS[0] >= full + 800:
                get(r.choice(("vesper", "maul")))
        else:
            if a.armour < 50 and a.credits >= 1000:
                get("heavy")
            elif a.armour < 25:
                get("light")
            if WEAPONS[a.main].price < 2000 and a.credits >= full:
                get("kestrel" if r.random() < 0.5 else "mistral")
        spend = cr <= 900 or (a.main is not None and WEAPONS[a.main].price >= 1500)
        for slot in ("C", "Q"):
            ab = a.agent.ability(slot)
            while spend and a.charges[slot] < ab.charges and a.credits >= ab.cost:
                if not get(slot):
                    break

    # --- thinking -----------------------------------------------------------------------------------------------

    def think(self, dt: float) -> Cmd:
        m, a = self.m, self.a
        c = Cmd()
        self.t += dt
        ai = m.team_ai[a.squad]
        if m.phase == "buy" and not self.bought and m.t < m.buy_time - min(self.buy_delay, m.buy_time * 0.4):
            self.bought = True
            self.buy()
        if m.phase in ("end", "over"):
            tgt = self._perceive(ai)
            if tgt is not None:
                self._fight(c, tgt, dt)
            return c
        task = ai.task(a)
        tgt = self._perceive(ai) if m.phase != "buy" else None
        act = task.get("act")
        if a.blind > 0.35:
            tgt = None
        using = False
        if act == "plant" and a.has_charge and m.map.site_at(a.x, a.y) and m.map.height(a.x, a.y) == 0.0:
            gx, gy = task["goal"]
            if math.hypot(gx - a.x, gy - a.y) < 1.2 or m.t < 12.0:
                if tgt is None or a.plant_t > 1.8 or m.t < 9.0:
                    c.use = using = True
        elif act == "defuse" and m.charge["state"] == "planted":
            ch = m.charge
            if math.hypot(ch["x"] - a.x, ch["y"] - a.y) < 1.5:
                left = C.DEFUSE_TIME - ch["defuse"]
                if tgt is None or ch["t"] < left + 1.6 or ch["defuse"] > C.DEFUSE_TIME - 1.5:
                    c.use = using = True
        if using:
            self._look(c, task, dt, None)
            return c
        if tgt is not None:
            self._fight(c, tgt, dt)
            if self.t < self.react_at or a.reload_t > 0:
                self._walk(c, task, dt, slow=True)
        else:
            self._walk(c, task, dt)
            self._look(c, task, dt, ai)
            self._idle_gun(c)
        if m.phase in ("live", "planted"):
            self.util_t -= dt
            if self.util_t <= 0:
                self.util_t = 0.45 + self.rng.random() * 0.3
                self._utility(c, ai, task, tgt)
        return c

    def _perceive(self, ai: TeamAI) -> Actor | None:
        a = self.a
        seen = ai.sees.get(a.idx) or ()
        cur = self.target
        if cur is not None and (not cur.alive or cur not in seen):
            if cur.alive:
                self.last_seen = (cur.x, cur.y, self.t)
            cur = None
        if cur is None and seen:
            cur = min(seen, key=lambda b: (b.x - a.x) ** 2 + (b.y - a.y) ** 2)
            D = self.D
            ang = abs(_wrap(math.atan2(cur.y - a.y, cur.x - a.x) - a.yaw))
            fresh = self.last_seen is None or self.t - self.last_seen[2] > 1.2
            react = D["react"] * self.rng.uniform(0.75, 1.3) * (1.0 if fresh else 0.45)
            react += ang * 0.12 + (0.35 if a.daze > 0 else 0.0) + (0.2 if a.blind > 0 else 0.0)
            self.react_at = self.t + react
            e = D["err"] * DEG * self.rng.uniform(0.6, 1.5) * (1.8 if a.daze > 0 else 1.0)
            th = self.rng.random() * TWO_PI
            self.err = [math.cos(th) * e, math.sin(th) * e * 0.6]
            self.aim_head = self.rng.random() < D["head"]
            self.burst = 0
            self.pause = 0.0
            self.crouch_fight = self.rng.random() < 0.35
        self.target = cur
        return cur

    def _turn(self, yaw: float, pitch: float, dt: float, fast: float = 1.0) -> float:
        """Turn the view towards (yaw, pitch); gives the angle still to go."""
        a = self.a
        dy = _wrap(yaw - a.yaw)
        dp = pitch - a.pitch
        off = math.hypot(dy, dp)
        rate = self.D["turn"] * DEG * (0.35 + min(off / 0.5, 1.6)) * fast * (0.5 if a.daze > 0 else 1.0)
        k = min(rate * dt / off, 1.0) if off > 1e-5 else 1.0
        a.yaw = _wrap(a.yaw + dy * k)
        a.pitch = min(max(a.pitch + dp * k, -1.2), 1.2)
        return off * (1 - k)

    def _fight(self, c: Cmd, b: Actor, dt: float) -> None:
        a, D = self.a, self.D
        w = a.weapon
        lag = D["lag"]
        dx, dy = b.x - b.vx * lag - a.x, b.y - b.vy * lag - a.y  # the eye is a little behind a mover
        dist = math.hypot(dx, dy) or 0.01
        tz = (b.top - 0.14) if self.aim_head else (b.z + (b.top - b.z) * 0.62)
        k = math.exp(-D["settle"] * dt)
        self.err[0] *= k
        self.err[1] *= k
        wob = D["err"] * DEG * 0.12
        yaw = math.atan2(dy, dx) + self.err[0] + math.sin(self.t * 2.3) * wob
        pitch = math.atan2(tz - a.eye, dist) + self.err[1] + math.cos(self.t * 1.7) * wob * 0.5
        yaw -= a.rec_y * DEG * D["control"]
        pitch -= a.rec_p * DEG * D["control"]
        off = self._turn(yaw, pitch, dt, 1.4)
        if a.cur != "main" and a.main and a.ammo.get(a.main, [0, 0])[0] > 0:
            c.slot = "main"
        mag = a.mag()
        if w.cls != "melee" and mag[0] == 0:
            other = "side" if a.cur == "main" else "main"
            owid = a.sidearm if other == "side" else a.main
            if mag[1] == 0 and owid and sum(a.ammo.get(owid, [0, 0])) > 0:
                c.slot = other
            elif mag[1] == 0:
                c.slot = "knife"
            elif dist < 9 and other == "side" and a.ammo.get(a.sidearm, [0, 0])[0] > 0 and a.reload_t <= 0:
                c.slot = "side"
            else:
                c.reload = True
        if w.cls == "melee":
            c.fwd = 1.0
            c.fire = dist < 1.8
            return
        if w.scoped:
            c.ads = dist > 6.0
        if self.t < self.react_at:
            return
        thr = 0.03 + min(0.3 / dist, 0.06)
        ready = off < thr and (not w.scoped or dist <= 6.0 or a.ads > 0.92)
        self.pause -= dt
        if self.pause > 0:
            if dist > 4 and self.D is not C.DIFFICULTY["easy"]:
                self.strafe_t -= dt
                if self.strafe_t <= 0:
                    self.strafe = self.rng.choice((-1.0, 1.0))
                    self.strafe_t = 0.3
                c.side = self.strafe
            return
        if ready:
            if self.burst <= 0:
                if not w.auto:
                    self.burst = 1 if dist > 14 else 2
                elif dist < 8:
                    self.burst = 10
                elif dist < 20:
                    self.burst = self.rng.randint(3, 6)
                else:
                    self.burst = self.rng.randint(1, 3)
                self._shots0 = a.stats["shots"]
            if w.auto:
                c.fire = True
            else:
                self.tap = not self.tap
                c.fire = self.tap and a.cool <= 0
            fired = a.stats["shots"] - self._shots0
            if fired >= self.burst:
                self.burst = 0
                c.fire = False
                self.pause = (0.16 + self.rng.random() * 0.22) if dist > 8 else 0.05
                if not w.auto:
                    self.pause = 0.12 + self.rng.random() * (0.25 if dist > 14 else 0.1)
        if dist > 2.5:
            c.fwd = c.side = 0.0
        c.crouch = self.crouch_fight and dist < 16 and w.auto

    def _idle_gun(self, c: Cmd) -> None:
        a = self.a
        if a.cur == "knife" or (a.cur == "side" and a.main and sum(a.ammo.get(a.main, [0, 0])) > 0):
            c.slot = "main"
        w = a.weapon
        if w.cls != "melee":
            mag = a.mag()
            if mag[0] < w.mag * 0.45 and mag[1] > 0 and a.reload_t <= 0:
                c.reload = True
        self.pick_t -= 1
        if self.pick_t <= 0:
            self.pick_t = 20
            if self.m.drops and self.m.phase in ("live", "planted"):
                self.m.pick_up(a, only_better=True)

    def _walk(self, c: Cmd, task: dict, dt: float, slow: bool = False) -> None:
        a, mp = self.a, self.m.map
        goal = task.get("goal")
        if goal is None or a.dash is not None:
            return
        gx, gy = goal
        d = math.hypot(gx - a.x, gy - a.y)
        near = task.get("near", 0.55)
        if d < near and (near < 1.0 or mp.los(a.x, a.y, a.eye, gx, gy, 0.6)):
            self.arrived = True
            if task.get("hold") and self.target is None:
                c.crouch = (int(self.t / 3.0 + a.idx) % 3 == 0) and a.side == "def"
            return
        self.arrived = False
        cell = (int(gx), int(gy))
        if cell != self.goal_cell or self.field is None or getattr(self, "_ver", -1) != mp.version:
            got = mp.try_field(gx, gy)
            if got is None:  # the map has worked out enough paths this tick: next tick
                if self.field is None or getattr(self, "_ver", -1) != mp.version:
                    return
            else:
                self.goal_cell = cell
                self._ver = mp.version
                self.field = got
                self.steer_t = 0.0
        self.steer_t -= dt
        if self.steer_t <= 0 or self.point is None:
            self.steer_t = 0.1
            if d < 1.6 and mp.clear_walk(a.x, a.y, gx, gy, 0.3):
                self.point = (gx, gy)
            else:
                self.point = mp.next_point(self.field, a.x, a.y)
                if self.point is None and self.field[int(a.y), int(a.x)] >= FAR:
                    return
                if self.point is None:
                    self.point = (gx, gy)
        px, py = self.point
        wx, wy = px - a.x, py - a.y
        n = math.hypot(wx, wy)
        if n < 0.05:
            return
        wx, wy = wx / n, wy / n
        if self.unstick > 0:
            self.unstick -= dt
            wx, wy = -wy * self.strafe - wx * 0.3, wx * self.strafe - wy * 0.3
            c.jump = self.unstick > 0.3
        fx, fy = math.cos(a.yaw), math.sin(a.yaw)
        c.fwd = wx * fx + wy * fy
        c.side = -wx * fy + wy * fx
        c.walk = bool(task.get("walk")) or slow
        self.stuck_t += dt
        if self.stuck_t > 0.8:
            moved = math.hypot(a.x - self.stuck_pos[0], a.y - self.stuck_pos[1])
            if moved < 0.3 and self.m.phase != "buy":
                self.unstick = 0.5
                self.strafe = self.rng.choice((-1.0, 1.0))
                self.point = None
            self.stuck_t = 0.0
            self.stuck_pos = (a.x, a.y)

    def _look(self, c: Cmd, task: dict, dt: float, ai: TeamAI | None) -> None:
        a = self.a
        look = None
        if self.alert is not None and self.t - self.alert[2] < 2.2:
            look = (self.alert[0], self.alert[1], 1.3)
        elif self.last_seen is not None and self.t - self.last_seen[2] < 2.5:
            look = (self.last_seen[0], self.last_seen[1], 1.3)
        elif ai is not None:
            kn = ai.known(a, 2.5, 30.0)
            if kn is not None:
                look = (kn[0], kn[1], 1.3)
        if look is None and not self.arrived and self.point is not None and (abs(c.fwd) + abs(c.side)) > 0.05:
            px, py = self.point
            if math.hypot(px - a.x, py - a.y) > 0.6:
                look = (px, py, a.z + 1.4)
        if look is None:
            watch = task.get("watch")
            if watch is not None:
                sway = math.sin(self.t * 0.6 + self.scan) * 0.3
                dx, dy = watch[0] - a.x, watch[1] - a.y
                yaw = math.atan2(dy, dx) + sway
                self._turn(yaw, math.atan2(1.4 - a.eye, max(math.hypot(dx, dy), 1.0)), dt, 0.6)
            return
        dx, dy = look[0] - a.x, look[1] - a.y
        self._turn(math.atan2(dy, dx), math.atan2(look[2] - a.eye + a.z * 0, max(math.hypot(dx, dy), 0.5)), dt, 0.8)

    # --- abilities ----------------------------------------------------------------------------------------------

    def _utility(self, c: Cmd, ai: TeamAI, task: dict, tgt: Actor | None) -> None:
        m, a, r = self.m, self.a, self.rng
        if a.plant_t > 0 or a.defusing or a.equip_t > 0 or a.blind > 0.3:
            return
        mp = m.map
        ch = m.charge
        known = ai.known(a, 1.8, 24.0)
        util = self.D["util"]
        entering = a.side == "atk" and ai.execute and ch["state"] != "planted"
        site = mp.sites[ai.site] if a.side == "atk" else None
        for slot in ("X", "E", "C", "Q"):
            if not a.can_use(slot):
                continue
            ab = a.agent.ability(slot)
            kind = ab.kind
            key = (slot, kind)
            lob = None
            go = False
            if kind in ("slow", "fire", "static", "concuss", "frag"):
                hot = None
                if a.side == "def" and ai.plant_heard and m.time - ai.plant_heard[2] < 3.0 and ch["state"] != "planted":
                    hot = ai.plant_heard[:2]
                elif a.side == "atk" and ch["state"] == "planted" and (m.time - ai.defuse_heard < 2.5
                                                                         or ch.get("defuser") is not None):
                    hot = (ch["x"], ch["y"])
                if hot is not None and 4.0 < math.hypot(hot[0] - a.x, hot[1] - a.y) < 24.0:
                    lob, go = hot, True
                elif known is not None and tgt is None and 5.0 < known[2] < 20.0 and r.random() < 0.5 * util:
                    lob, go = (known[0], known[1]), True
                elif tgt is not None and kind == "frag" and 7.0 < math.hypot(tgt.x - a.x, tgt.y - a.y) < 18.0 \
                        and r.random() < 0.15 * util:
                    lob, go = (tgt.x, tgt.y), True
            elif kind == "flash":
                if known is not None and tgt is None and 5.0 < known[2] < 22.0 and r.random() < 0.6 * util:
                    lob, go = self._toward(known[0], known[1], min(known[2] * 0.7, 9.0)), True
                elif entering and site is not None and key not in self.used and r.random() < 0.5 * util:
                    cx, cy = site.centre
                    d = math.hypot(cx - a.x, cy - a.y)
                    if 6.0 < d < 20.0:
                        lob, go = self._toward(cx, cy, min(d * 0.7, 9.0)), True
            elif kind == "smoke":
                if entering and site is not None and site.smokes:
                    pts = sorted(site.smokes, key=lambda p: math.hypot(p[0] - a.x, p[1] - a.y))
                    free = [p for p in pts if not any(math.hypot(s[0] - p[0], s[1] - p[1]) < 3.0 for s in m.smokes)]
                    if free and math.hypot(free[0][0] - a.x, free[0][1] - a.y) < 26.0 and r.random() < util:
                        lob, go = free[0], True
                elif a.side == "def" and known is not None and tgt is None and 7.0 < known[2] < 24.0 \
                        and ch["state"] != "planted" and r.random() < 0.35 * util:
                    lob, go = self._toward(known[0], known[1], known[2] * 0.75), True
                elif a.side == "def" and ch["state"] == "planted" and ai.defuser is a and key not in self.used:
                    d = math.hypot(ch["x"] - a.x, ch["y"] - a.y)
                    if d < 14.0:
                        lob, go = (ch["x"], ch["y"]), True
            elif kind == "sonar":
                if entering and site is not None and key not in self.used:
                    cx, cy = site.centre
                    if math.hypot(cx - a.x, cy - a.y) < 26.0:
                        lob, go = (cx, cy), True
                elif known is not None and tgt is None and known[2] < 24.0 and r.random() < 0.4 * util:
                    lob, go = (known[0], known[1]), True
                elif a.side == "def" and ch["state"] == "planted" and math.hypot(ch["x"] - a.x, ch["y"] - a.y) < 24:
                    lob, go = (ch["x"], ch["y"]), True
            elif kind == "curtain":
                if entering and site is not None and key not in self.used:
                    cx, cy = site.centre
                    d = math.hypot(cx - a.x, cy - a.y)
                    if 8.0 < d < 20.0 and abs(_wrap(math.atan2(cy - a.y, cx - a.x) - a.yaw)) < 0.5:
                        go = r.random() < util
            elif kind == "sensor":
                go = self.arrived and bool(task.get("hold")) and key not in self.used and m.phase in ("live", "planted")
            elif kind == "sweep":
                go = (entering or (a.side == "def" and ch["state"] == "planted")) and r.random() < 0.5 * util \
                    and len(m.enemies(a)) >= 2
            elif kind == "nearsight":
                go = (entering or (a.side == "def" and ch["state"] == "planted") or tgt is not None) \
                    and r.random() < 0.4 * util and len(m.enemies(a)) >= 2
            elif kind == "stronghold":
                if a.side == "atk" and ch["state"] == "planted":
                    go = math.hypot(ch["x"] - a.x, ch["y"] - a.y) < 8.0
                elif a.side == "def":
                    go = known is not None and known[2] < 13.0 and mp.site_at(a.x, a.y) is not None
            elif kind == "chain":
                if tgt is not None:
                    d = math.hypot(tgt.x - a.x, tgt.y - a.y)
                    go = 5.0 < d < 20.0 and abs(_wrap(math.atan2(tgt.y - a.y, tgt.x - a.x) - a.yaw)) < 0.12
            elif kind == "dash":
                if entering and task.get("enter") and not self.arrived and a.speed() > 3.0 and tgt is None:
                    go = r.random() < 0.12 * util and mp.site_at(a.x, a.y) is None
                elif tgt is None and self.alert is not None and self.t - self.alert[2] < 0.6 and a.hp < 70:
                    go = a.speed() > 2.0
            elif kind == "buff":
                if "heal" in ab.p:
                    go = a.hp < 60 and tgt is None
                else:
                    go = tgt is not None and self.t >= self.react_at and math.hypot(tgt.x - a.x, tgt.y - a.y) < 28
            elif kind == "heal":
                if tgt is None:
                    go = a.hp < 55 or any(b.hp < 55 and math.hypot(b.x - a.x, b.y - a.y) < ab.p["range"] and b is not a
                                          for b in m.team(a.squad))
            elif kind == "barricade":
                if known is not None and tgt is None and 4.0 < known[2] < 13.0 and not entering \
                        and abs(_wrap(math.atan2(known[1] - a.y, known[0] - a.x) - a.yaw)) < 0.35:
                    go = r.random() < 0.5 * util
            if go:
                c.ability, c.lob = slot, lob
                if kind in LOBBED and lob is None:
                    c.ability = None
                    continue
                self.used.add(key)
                self.util_t = 1.4 + r.random()
                return

    def _toward(self, x: float, y: float, dist: float) -> tuple:
        a = self.a
        dx, dy = x - a.x, y - a.y
        n = math.hypot(dx, dy) or 1.0
        return a.x + dx / n * dist, a.y + dy / n * dist
