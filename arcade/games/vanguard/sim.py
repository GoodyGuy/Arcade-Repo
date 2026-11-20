"""A Vanguard match: ten heroes, their weapons and health, shots, barriers, zones and gadgets, the objective (the
control point or the escort cart), spawn rooms and respawns, the kill feed and the score. No Qt and no drawing here:
the game widget reads this state to draw it, and plays the sounds listed in ``Match.sounds``."""

from __future__ import annotations

import math
import random

from . import kits
from .data import (ARMOR_BLOCK, ASSIST_WINDOW, BOT_NAMES, CAPTURE_TIME, CART_RADIUS, CART_ROLLBACK, CART_SPEED,
                   CONTROL_RATE, ESCORT_BONUS, ESCORT_TIME, GRAVITY, HEROES, JUMP, MELEE_CD, MELEE_DMG, MELEE_RANGE,
                   OVERHEALTH_DECAY, PACK_BIG, PACK_SMALL, POINT_UNLOCK, PREP_TIME, REGEN_DELAY, REGEN_RATE, RESPAWN,
                   ROUNDS_TO_WIN, SHIELD_DELAY, SHIELD_RATE, SKILL, STEP, ULT_PASSIVE, Hero, pick_team)
from .maps import MapData, track_at, track_length
from .world import World

RADIUS = 0.4


class Controls:
    """What a hero is being told to do this frame (by the player's keys or by a bot)."""
    __slots__ = ("a1", "a2", "alt", "fire", "jump", "melee", "mf", "ms", "reload", "ult")

    def __init__(self) -> None:
        self.mf = self.ms = 0.0
        self.fire = self.alt = self.a1 = self.a2 = self.ult = self.jump = self.reload = self.melee = False


class Stats:
    __slots__ = ("assists", "damage", "deaths", "elims", "final", "healing", "obj_time", "ults")

    def __init__(self) -> None:
        self.elims = self.final = self.assists = self.deaths = self.ults = 0
        self.damage = self.healing = self.obj_time = 0.0


class Actor:
    def __init__(self, idx: int, team: int, name: str, hero: Hero, is_player: bool = False) -> None:
        self.idx, self.team, self.name, self.is_player = idx, team, name, is_player
        self.c = Controls()
        self.p_a1 = self.p_a2 = self.p_ult = self.p_melee = False
        self.brain = None
        self.stats = Stats()
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.kx = self.ky = 0.0
        self.yaw = self.pitch = 0.0
        self.on_ground = True
        self.blocked = False
        self.alive = False
        self.respawn_t = 0.0
        self.death_t = 0.0
        self.protected = False
        self.next_hero: Hero | None = None
        self.ult = 0.0
        self.dmg_mult = 1.0
        self.view_z = 0.0
        self.hurt_dirs: list = []  # [world angle the hit came from, time left]
        self.set_hero(hero)

    def set_hero(self, hero: Hero) -> None:
        keep = min(self.ult / max(getattr(self, "hero", hero).ult.cost, 1), 0.15) if hasattr(self, "hero") else 0.0
        self.hero = hero
        self.radius = RADIUS + (0.05 if hero.role == "tank" else 0.0)
        self.hit_r = 0.42 * hero.scale
        self.height = hero.height
        self.eye = hero.height * 0.9
        self.ult = keep * hero.ult.cost
        self.reset_life()

    def reset_life(self) -> None:
        h = self.hero
        self.hp, self.armor, self.shield, self.overhp = float(h.hp), float(h.armor), float(h.shield), 0.0
        self.overhp_t = 0.0
        self.ammo = float(h.weapon.mag)
        self.reload_t = self.fire_cd = self.melee_cd = 0.0
        self.burst_left = 0
        self.burst_t = self.spray_t = 0.0
        self.cd = [0.0, 0.0]
        self.chg = [h.a1.charges, h.a2.charges]
        self.fx: dict[str, float] = {}
        self.slow_f = self.boost_m = self.haste_m = self.dr_m = 1.0
        self.burn_dps = self.hot_hps = 0.0
        self.burn_src = self.hot_src = None
        self.sink_store = 0.0
        self.hitset: set = set()
        self.damagers: dict[int, float] = {}
        self.hurt_t = -99.0
        self.healed_t = -99.0
        self.dash_t = 0.0
        self.dash_vx = self.dash_vy = 0.0
        self.zoom = 0.0
        self.firing = False
        self.walk = self.move_amt = self.kick = self.flash = self.cast_anim = 0.0
        self.melee_anim = 0.0
        self.tick_t = self.aura_t = 0.0

    # --- numbers the HUD and the bots ask for ---------------------------------------------------------------------
    @property
    def max_total(self) -> float:
        return float(self.hero.total)

    @property
    def total(self) -> float:
        return self.hp + self.armor + self.shield + self.overhp

    @property
    def frac(self) -> float:
        return (self.hp + self.armor + self.shield) / max(self.hero.total, 1)

    @property
    def ult_ready(self) -> bool:
        return self.ult >= self.hero.ult.cost

    def eye_pos(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z + self.eye

    def chest(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z + self.height * 0.62

    def aim(self) -> tuple[float, float, float]:
        cp = math.cos(self.pitch)
        return cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)


class Projectile:
    __slots__ = ("age", "dmg", "g", "kind", "life", "owner", "team", "vx", "vy", "vz", "x", "y", "z")

    def __init__(self, owner: Actor, kind: str, pos: tuple, vel: tuple, g: float, dmg: float, life: float = 6.0):
        self.owner, self.team, self.kind = owner, owner.team, kind
        self.x, self.y, self.z = pos
        self.vx, self.vy, self.vz = vel
        self.g, self.dmg, self.life, self.age = g, dmg, life, 0.0


class Barrier:
    """A hard-light wall: stops enemy shots until it runs out of health or time."""

    def __init__(self, owner: Actor, x: float, y: float, z: float, yaw: float, w: float, h: float, hp: float,
                 life: float) -> None:
        self.owner, self.team = owner, owner.team
        self.x, self.y, self.z, self.yaw, self.w, self.h = x, y, z, yaw, w, h
        self.hp = self.max_hp = hp
        self.life = life
        self.flash = 0.0
        self.nx, self.ny = math.cos(yaw), math.sin(yaw)

    def ray(self, ox: float, oy: float, oz: float, dx: float, dy: float, dz: float, limit: float) -> float | None:
        den = dx * self.nx + dy * self.ny
        if abs(den) < 1e-6:
            return None
        t = ((self.x - ox) * self.nx + (self.y - oy) * self.ny) / den
        if t < 0.0 or t > limit:
            return None
        px, py, pz = ox + dx * t, oy + dy * t, oz + dz * t
        if abs((px - self.x) * -self.ny + (py - self.y) * self.nx) > self.w / 2 or not self.z <= pz <= self.z + self.h:
            return None
        return t


class Gadget:
    """Something a hero planted: a pylon, a lighthouse, a mine."""

    def __init__(self, owner: Actor, kind: str, x: float, y: float, z: float, hp: float, life: float,
                 radius: float, height: float) -> None:
        self.owner, self.team, self.kind = owner, owner.team, kind
        self.x, self.y, self.z = x, y, z
        self.hp = self.max_hp = hp
        self.life, self.age = life, 0.0
        self.hit_r, self.height = radius, height
        self.alive = True
        self.tick_t = self.aura_t = 0.0
        self.flash = 0.0


class Zone:
    __slots__ = ("age", "kind", "life", "owner", "r", "team", "tick_t", "x", "y", "z")

    def __init__(self, owner: Actor, kind: str, x: float, y: float, z: float, r: float, life: float) -> None:
        self.owner, self.team, self.kind = owner, owner.team, kind
        self.x, self.y, self.z, self.r, self.life = x, y, z, r, life
        self.age = self.tick_t = 0.0


class Fx:
    """Something only seen: a tracer, a blast, a ring."""
    __slots__ = ("a", "b", "col", "kind", "life", "size", "t")

    def __init__(self, kind: str, a: tuple, b: tuple | None, col: tuple, life: float, size: float = 1.0) -> None:
        self.kind, self.a, self.b, self.col, self.life, self.size, self.t = kind, a, b, col, life, size, 0.0


class Match:
    def __init__(self, md: MapData, world: World, mode: str, player_hero: str, difficulty: str = "Regular",
                 attack: bool = True, seed: int | None = None, autopilot: bool = False) -> None:
        from .bots import Brain

        self.md, self.world, self.mode = md, world, mode
        self.rng = random.Random(seed)
        self.difficulty = difficulty
        self.time = 0.0
        self.attack_team = 0 if attack else 1  # escort: who pushes the cart
        self.actors: list[Actor] = []
        names = list(BOT_NAMES)
        self.rng.shuffle(names)
        teams = (pick_team(self.rng, player_hero), pick_team(self.rng))
        skill_enemy = SKILL[difficulty]
        skill_ally = SKILL["Veteran" if difficulty == "Elite" else "Regular"]
        if autopilot:
            skill_ally = skill_enemy
        for team in (0, 1):
            for i, hid in enumerate(teams[team]):
                is_player = team == 0 and i == 0
                a = Actor(len(self.actors), team, "YOU" if is_player else names.pop(), HEROES[hid], is_player)
                if not is_player or autopilot:
                    sk = skill_ally if team == 0 else skill_enemy
                    a.brain = Brain(a, sk, self.rng)
                    a.dmg_mult = sk[4] if team == 1 else 1.0
                self.actors.append(a)
        self.player = self.actors[0]
        self.team_lists = ([a for a in self.actors if a.team == 0], [a for a in self.actors if a.team == 1])
        self.projectiles: list[Projectile] = []
        self.barriers: list[Barrier] = []
        self.gadgets: list[Gadget] = []
        self.zones: list[Zone] = []
        self.fx: list[Fx] = []
        self.delayed: list = []  # [time, function]
        self.sounds: list = []  # (name, x, y, volume) - x None: heard everywhere
        self.feed: list = []  # {"k", "kt", "v", "vt", "how", "head", "t"}
        self.banners: list = []  # [text, sub, (r, g, b), time left]
        self.hit_t = self.kill_t = 0.0
        self.hit_head = False
        self.heal_t = 0.0
        self.elim_text: list = []  # [text, time left] for the player
        self.calls: list = []  # [text, ours, time left]: ultimates called out
        self._vis: dict = {}
        self.packs = [{"x": x, "y": y, "z": world.fh[int(y)][int(x)], "big": big, "t": 0.0}
                      for x, y, big in md.packs]
        self.phase = "prep"
        self.phase_t = PREP_TIME + (6.0 if mode == "escort" else 0.0)  # escort: the defenders get a start
        self.winner: int | None = None
        self.over_t = 0.0
        self.live_t = 0.0
        # control
        self.owner = -1
        self.cap_team = -1
        self.cap = 0.0
        self.score = [0.0, 0.0]
        self.wins = [0, 0]
        self.round = 1
        self.overtime = False
        self.ot_t = 0.0
        self.touch = [99.0, 99.0]
        self.on_point = [0, 0]
        self.contested = False
        # escort
        self.cart_d = 0.0
        self.cart_len = track_length(md.track) if md.track else 0.0
        self.cp = 0
        self.time_left = ESCORT_TIME
        self.pushers = 0
        self.no_push = 0.0
        self.cart_moving = 0.0
        self.best_d = 0.0
        self._set_rooms()
        for a in self.actors:
            self.spawn(a)

    # --- rooms and spawning ---------------------------------------------------------------------------------------
    def team_room(self, team: int) -> int:
        if self.mode == "control":
            return self.md.control_rooms[team]
        att, dfn = self.md.phases[min(self.cp, len(self.md.phases) - 1)]
        return att if team == self.attack_team else dfn

    def _set_rooms(self) -> None:
        w = self.world
        w.room_owner = {}
        if self.mode == "control":
            w.room_owner = {self.md.control_rooms[0]: 0, self.md.control_rooms[1]: 1}
            w.closed = set(self.md.control_rooms) if self.phase == "prep" else set()
        else:
            cp = min(self.cp, len(self.md.phases) - 1)
            for i in range(cp + 1):
                w.room_owner[self.md.phases[i][0]] = self.attack_team
            w.room_owner[self.md.phases[cp][1]] = 1 - self.attack_team
            w.closed = {self.md.phases[0][0]} if self.phase == "prep" else set()

    def spawn(self, a: Actor) -> None:
        if a.next_hero is not None:
            a.set_hero(a.next_hero)
            a.next_hero = None
        else:
            a.reset_life()
        room = self.md.rooms[self.team_room(a.team)]
        spots = room["spawns"]
        x, y, yaw = spots[self.team_lists[a.team].index(a) % len(spots)]
        a.x, a.y, a.z, a.yaw, a.pitch = x, y, room["h"], yaw, 0.0
        a.view_z = a.z
        a.vx = a.vy = a.vz = a.kx = a.ky = 0.0
        a.alive = True
        a.on_ground = True
        a.respawn_t = 0.0
        a.damagers.clear()
        if a.brain is not None:
            a.brain.respawned(self)

    def in_own_room(self, a: Actor) -> bool:
        rm = self.world.room_at(a.x, a.y)
        return bool(rm) and self.world.room_owner.get(rm) == a.team

    def swap_hero(self, a: Actor, hid: str) -> bool:
        """Change hero: at once inside the team's room, at the next respawn when dead."""
        hero = HEROES[hid]
        if hero is a.hero and a.next_hero is None:
            return False
        if not a.alive:
            a.next_hero = hero
            return True
        if not self.in_own_room(a):
            return False
        pos = (a.x, a.y, a.z, a.yaw, a.pitch)
        a.set_hero(hero)
        a.x, a.y, a.z, a.yaw, a.pitch = pos
        self.sound("swap")
        return True

    # --- small helpers --------------------------------------------------------------------------------------------
    def sound(self, name: str, x: float | None = None, y: float | None = None, vol: float = 1.0) -> None:
        self.sounds.append((name, x, y, vol))

    def banner(self, text: str, sub: str = "", col: tuple = (255, 255, 255), dur: float = 3.0) -> None:
        self.banners.append([text, sub, col, dur, dur])
        del self.banners[:-3]

    def add_fx(self, kind: str, a: tuple, b: tuple | None = None, col: tuple = (255, 255, 255), life: float = 0.3,
               size: float = 1.0) -> None:
        if len(self.fx) < 220:
            self.fx.append(Fx(kind, a, b, col, life, size))

    def visible(self, a: Actor, b) -> bool:  # noqa: ANN001 - an Actor or a Gadget
        """Can ``a`` see ``b``? Remembered for a moment: bots and auras ask this a lot."""
        key = (a.idx, id(b)) if not isinstance(b, Actor) else ((a.idx, b.idx) if a.idx < b.idx else (b.idx, a.idx))
        hit = self._vis.get(key)
        if hit is not None and self.time - hit[0] < 0.14:
            return hit[1]
        ax, ay, az = a.eye_pos()
        ok = self.world.los(ax, ay, az, b.x, b.y, b.z + b.height * 0.7)
        self._vis[key] = (self.time + self.rng.random() * 0.04, ok)
        return ok

    def enemies(self, a: Actor) -> list[Actor]:
        return self.team_lists[1 - a.team]

    def allies(self, a: Actor) -> list[Actor]:
        return self.team_lists[a.team]

    def objective_pos(self) -> tuple[float, float, float]:
        if self.mode == "control":
            x, y, _r = self.md.point
            return x, y, self.world.floor_at(x, y)
        x, y, _yaw = track_at(self.md.track, self.cart_d)
        return x, y, 0.0

    # --- damage and healing ---------------------------------------------------------------------------------------
    def damage(self, src: Actor | None, tgt: Actor, amt: float, kind: str = "hit", head: bool = False,
               how: str = "", self_ok: bool = False) -> float:
        if amt <= 0.0 or not tgt.alive or self.phase != "live" or tgt.protected:
            return 0.0
        if src is not None:
            if src.team == tgt.team and not (self_ok and src is tgt):
                return 0.0
            if "boost" in src.fx:
                amt *= src.boost_m
            amt *= src.dmg_mult
        if "mark" in tgt.fx:
            amt *= 1.25
        if "dr" in tgt.fx:
            amt *= tgt.dr_m
        if "windup" in tgt.fx:
            amt *= 0.7
        if "bulldoze" in tgt.fx and src is not None:
            dx, dy = src.x - tgt.x, src.y - tgt.y
            d = math.hypot(dx, dy) or 1.0
            if (dx * math.cos(tgt.yaw) + dy * math.sin(tgt.yaw)) / d > 0.3:
                amt *= 0.4
        if "sink" in tgt.fx:
            tgt.sink_store += amt * 0.5
            amt *= 0.5
        left = amt
        if tgt.overhp > 0.0:
            d = min(tgt.overhp, left)
            tgt.overhp -= d
            left -= d
        if left > 0.0 and tgt.shield > 0.0:
            d = min(tgt.shield, left)
            tgt.shield -= d
            left -= d
            if tgt.shield <= 0.0:
                self.sound("shield_break", tgt.x, tgt.y, 0.7)
        if left > 0.0 and tgt.armor > 0.0:
            left -= min(ARMOR_BLOCK, left * 0.5) if kind == "hit" else left * 0.25
            d = min(tgt.armor, left)
            tgt.armor -= d
            left -= d
        if left > 0.0:
            tgt.hp -= left
        tgt.hurt_t = self.time
        tgt.flash = 0.14
        if src is not None and src is not tgt:
            tgt.damagers[src.idx] = self.time
            src.stats.damage += amt
            if not kits.ULT_ACTIVE.intersection(src.fx):
                src.ult = min(src.ult + amt, src.hero.ult.cost)
            if src.is_player:
                self.hit_t = 0.2
                self.hit_head = head
                self.sound("hit_head" if head else "hit", None, None, 0.6 if kind == "dot" else 1.0)
            if tgt.is_player:
                tgt.hurt_dirs.append([math.atan2(src.y - tgt.y, src.x - tgt.x), 1.2])
                del tgt.hurt_dirs[:-6]
                if kind != "dot" or self.rng.random() < 0.3:
                    self.sound("hurt", None, None, 0.7)
        if tgt.hp <= 0.0:
            self.kill(src, tgt, how, head)
        return amt

    def heal(self, src: Actor | None, tgt: Actor, amt: float) -> float:
        if amt <= 0.0 or not tgt.alive:
            return 0.0
        h = tgt.hero
        d1 = min(h.hp - tgt.hp, amt)
        tgt.hp += d1
        d2 = min(h.armor - tgt.armor, amt - d1)
        tgt.armor += d2
        done = d1 + d2
        if done > 0.0:
            tgt.healed_t = self.time
            if src is not None and src is not tgt:
                src.stats.healing += done
                if not kits.ULT_ACTIVE.intersection(src.fx):
                    src.ult = min(src.ult + done, src.hero.ult.cost)
                if src.is_player:
                    self.heal_t = 0.25
            if self.rng.random() < 0.08 + done * 0.01:
                self.add_fx("heal", (tgt.x + self.rng.uniform(-0.3, 0.3), tgt.y + self.rng.uniform(-0.3, 0.3),
                                     tgt.z + self.rng.uniform(0.6, 1.6)), None, (120, 255, 170), 0.7, 0.16)
        return done

    def kill(self, src: Actor | None, tgt: Actor, how: str, head: bool) -> None:
        tgt.alive = False
        tgt.hp = 0.0
        tgt.death_t = 0.0
        tgt.respawn_t = RESPAWN + (0.0 if self.mode != "escort" else (1.5 if tgt.team == self.attack_team else -1.5))
        tgt.stats.deaths += 1
        tgt.fx.clear()
        tgt.firing = False
        tgt.c = Controls()
        credited = set()
        if src is not None and src is not tgt:
            src.stats.final += 1
            src.stats.elims += 1
            credited.add(src.idx)
            if "rush" in src.fx:
                src.fx["rush"] += 1.5
        for idx, t in tgt.damagers.items():
            if idx not in credited and self.time - t < ASSIST_WINDOW:
                self.actors[idx].stats.assists += 1
                self.actors[idx].stats.elims += 1
                credited.add(idx)
                if self.actors[idx].is_player:
                    self.elim_text.append([f"ASSIST  {tgt.name}", 2.4])
        self.feed.append({"k": src.name if src else "", "kt": src.team if src else -1, "kh": src.hero.id if src else "",
                          "v": tgt.name, "vt": tgt.team, "vh": tgt.hero.id, "how": how, "head": head,
                          "t": self.time})
        del self.feed[:-6]
        col = (255, 120, 90) if tgt.team else (110, 190, 255)
        for _ in range(8):
            self.add_fx("spark", (tgt.x + self.rng.uniform(-0.4, 0.4), tgt.y + self.rng.uniform(-0.4, 0.4),
                                  tgt.z + self.rng.uniform(0.4, 1.7)), None, col, 0.7, 0.3)
        if src is not None and src.is_player:
            self.kill_t = 0.5
            self.elim_text.append([f"ELIMINATED  {tgt.name}", 2.6])
            self.sound("elim")
        if tgt.is_player:
            self.sound("death")
            tgt.hurt_dirs.clear()
        else:
            self.sound("down", tgt.x, tgt.y, 0.8)
        del self.elim_text[:-3]

    def explode(self, src: Actor, x: float, y: float, z: float, radius: float, dmg: float, how: str,
                knock: float = 0.0, self_mult: float = 0.0, col: tuple = (255, 170, 60), skip: Actor | None = None,
                quiet: bool = False) -> None:
        """A blast: full damage at its middle, a third at its edge, to enemies it can see."""
        w = self.world
        for t in self.actors:
            if not t.alive or t is skip:
                continue
            own = t is src
            if (t.team == src.team and not own) or (own and self_mult <= 0.0 and knock <= 0.0):
                continue
            tx, ty, tz = t.chest()
            d = math.sqrt((tx - x) ** 2 + (ty - y) ** 2 + (tz - z) ** 2)
            if d > radius + t.hit_r or not w.los(x, y, z + 0.15, tx, ty, tz):
                continue
            k = 1.0 - 0.67 * min(d / radius, 1.0)
            if own:
                if self_mult > 0.0:
                    self.damage(src, t, dmg * k * self_mult, "hit", False, how, True)
            else:
                self.damage(src, t, dmg * k, "hit", False, how)
            if knock > 0.0 and not own:
                hd = math.hypot(tx - x, ty - y) or 1.0
                t.kx += (tx - x) / hd * knock * k
                t.ky += (ty - y) / hd * knock * k
                t.vz = max(t.vz, knock * 0.5 * k)
                t.on_ground = False
        for g in self.gadgets:
            if g.alive and g.team != src.team and math.dist((g.x, g.y, g.z), (x, y, z)) < radius + 0.5:
                self.hurt_gadget(g, dmg * 0.6)
        for b in self.barriers:
            if b.team != src.team and math.dist((b.x, b.y, b.z + 1.0), (x, y, z)) < radius + b.w * 0.4:
                b.hp -= dmg * 0.6
                b.flash = 0.15
        self.add_fx("boom", (x, y, z), None, col, 0.42, radius)
        if not quiet:
            self.sound("boom", x, y, min(0.6 + radius * 0.1, 1.2))

    def hurt_gadget(self, g: Gadget, dmg: float) -> None:
        if self.phase != "live":
            return
        g.hp -= dmg
        g.flash = 0.12
        if g.hp <= 0.0 and g.alive:
            g.alive = False
            self.add_fx("boom", (g.x, g.y, g.z + 0.6), None, (255, 240, 180), 0.3, 1.2)
            self.sound("break", g.x, g.y, 0.8)

    # --- shots ----------------------------------------------------------------------------------------------------
    @staticmethod
    def ray_body(t, ox: float, oy: float, oz: float, dx: float, dy: float, dz: float, limit: float,  # noqa: ANN001
                 fat: float = 0.0) -> tuple[float, bool] | None:
        """Where a ray meets a standing body (an Actor or a Gadget): (distance, head shot)."""
        rx, ry = t.x - ox, t.y - oy
        dh2 = dx * dx + dy * dy
        if dh2 < 1e-9:
            return None
        tca = (rx * dx + ry * dy) / dh2
        if tca < 0.0:
            return None
        R = t.hit_r + fat
        perp2 = rx * rx + ry * ry - tca * tca * dh2
        if perp2 > R * R:
            return None
        t0 = tca - math.sqrt((R * R - perp2) / dh2)
        if t0 > limit:
            return None
        t0 = max(t0, 0.0)
        zc = oz + dz * tca
        top = t.z + t.height
        if not (t.z - 0.05 - fat <= zc <= top + 0.05 + fat):
            z0 = oz + dz * t0
            if not (t.z <= z0 <= top):
                return None
            zc = z0
        return t0, (zc > top - 0.4 * (t.height / 1.8) and perp2 < (R * 0.62) ** 2 and fat == 0.0)

    def hitscan(self, a: Actor, ox: float, oy: float, oz: float, dx: float, dy: float, dz: float) -> float:
        """Fire one round of ``a``'s weapon along the ray; returns how far it went."""
        w = a.hero.weapon
        limit = self.world.raycast(ox, oy, oz, dx, dy, dz, w.rng)
        wall = None
        for b in self.barriers:
            if b.team != a.team:
                t = b.ray(ox, oy, oz, dx, dy, dz, limit)
                if t is not None:
                    limit, wall = t, b
        pierce = "deadline" in a.fx
        hits = []
        for t in self.actors:
            if t is a or not t.alive:
                continue
            ally = t.team == a.team
            if ally and (w.heal <= 0.0 or t.frac >= 1.0):
                continue
            res = self.ray_body(t, ox, oy, oz, dx, dy, dz, limit, 0.3 if ally else 0.0)
            if res is not None:
                hits.append((res[0], t.idx, t, res[1]))
        for g in self.gadgets:
            if g.alive and g.team != a.team and g.kind != "mine":
                res = self.ray_body(g, ox, oy, oz, dx, dy, dz, limit)
                if res is not None:
                    hits.append((res[0], 100 + id(g) % 1000, g, False))
        hits.sort(key=lambda h: h[0])
        if not pierce:
            hits = hits[:1]
        for d, _i, t, head in hits:
            if isinstance(t, Gadget):
                self.hurt_gadget(t, w.dmg)
                if a.is_player:
                    self.hit_t, self.hit_head = 0.2, False
                    self.sound("hit")
            elif t.team == a.team:
                self.heal(a, t, w.heal)
            else:
                k = 1.0 if d <= w.near else (w.far_mult if d >= w.far else
                                             1.0 + (w.far_mult - 1.0) * (d - w.near) / (w.far - w.near))
                dmg = w.dmg * k * (w.head if head else (1.5 if pierce else 1.0))
                self.damage(a, t, dmg, "hit", head, w.name)
                self.add_fx("spark", (ox + dx * d, oy + dy * d, oz + dz * d), None, (255, 240, 200), 0.18, 0.22)
        if hits and not pierce:
            return hits[0][0]
        if wall is not None:
            wall.hp -= w.dmg
            wall.flash = 0.12
            self.add_fx("spark", (ox + dx * limit, oy + dy * limit, oz + dz * limit), None, (160, 220, 255), 0.2, 0.4)
        elif limit < w.rng:
            self.add_fx("spark", (ox + dx * limit, oy + dy * limit, oz + dz * limit), None, (255, 220, 170), 0.14,
                        0.16)
        return limit

    def muzzle(self, a: Actor) -> tuple[float, float, float]:
        """Roughly where ``a``'s gun ends (shots are drawn from here)."""
        dx, dy, dz = a.aim()
        s = a.hero.scale
        rx, ry = -math.sin(a.yaw), math.cos(a.yaw)
        return (a.x + rx * 0.2 * s + dx * 0.8 * s, a.y + ry * 0.2 * s + dy * 0.8 * s,
                a.z + a.height * 0.74 + dz * 0.8 * s)

    def launch(self, a: Actor, kind: str, speed: float, g: float, dmg: float, up: float = 0.0,
               spread: float = 0.0) -> Projectile:
        dx, dy, dz = a.aim()
        if spread:
            yaw = a.yaw + self.rng.gauss(0.0, spread)
            pitch = a.pitch + self.rng.gauss(0.0, spread)
            cp = math.cos(pitch)
            dx, dy, dz = cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)
        ox, oy, oz = a.eye_pos()
        p = Projectile(a, kind, (ox + dx * 0.4, oy + dy * 0.4, oz - 0.12 + dz * 0.4),
                       (dx * speed, dy * speed, dz * speed + up), g, dmg)
        self.projectiles.append(p)
        return p

    def _fire_round(self, a: Actor) -> None:
        w = a.hero.weapon
        if "deadline" not in a.fx:
            a.ammo -= 1
        a.kick = 1.0
        spread = math.radians(w.spread) * (0.15 if a.zoom > 0.6 else 1.0)
        if w.kind == "proj":
            self.launch(a, "charge" if w.splash else "rivet", w.speed, w.gravity, w.dmg, spread=spread)
        else:
            yaw = a.yaw + self.rng.gauss(0.0, spread)
            pitch = a.pitch + self.rng.gauss(0.0, spread)
            cp = math.cos(pitch)
            dx, dy, dz = cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)
            ox, oy, oz = a.eye_pos()
            d = self.hitscan(a, ox, oy, oz, dx, dy, dz)
            col = (255, 255, 255) if "deadline" in a.fx else w.tracer
            self.add_fx("tracer", self.muzzle(a), (ox + dx * d, oy + dy * d, oz + dz * d), col, 0.09,
                        0.05 if w.dmg < 20 else 0.08)
            if a.is_player:
                self.fx[-1].kind = "ptracer"
        self.sound(w.sound, a.x, a.y, 1.0 if a.is_player else 0.75)

    def _cone_tick(self, a: Actor, dt: float) -> None:
        """The slag thrower and the mist sprayer: everyone in the cone in front, each tick."""
        w = a.hero.weapon
        dx, dy, dz = a.aim()
        ox, oy, oz = a.eye_pos()
        cosc = math.cos(math.radians(w.cone))
        for t in self.actors:
            if t is a or not t.alive:
                continue
            ally = t.team == a.team
            if ally and (w.heal <= 0.0 or t.frac >= 1.0):
                continue
            tx, ty, tz = t.chest()
            vx, vy, vz = tx - ox, ty - oy, tz - oz
            d = math.sqrt(vx * vx + vy * vy + vz * vz)
            if d > w.rng + t.hit_r:
                continue
            # the cone is generous close up: a body's width counts
            if d > 0.3 and (vx * dx + vy * dy + vz * dz) / d < math.cos(math.radians(w.cone) + math.atan2(t.hit_r, d)):
                continue
            if not self.visible(a, t):
                continue
            if ally:
                self.heal(a, t, w.heal * dt)
                continue
            blocked = False
            for b in self.barriers:
                if b.team != a.team and b.ray(ox, oy, oz, vx / d, vy / d, vz / d, d) is not None:
                    b.hp -= w.dmg * dt
                    b.flash = 0.1
                    blocked = True
            if blocked:
                continue
            self.damage(a, t, w.dmg * dt, "dot", False, w.name)
            if a.hero.id == "caldera":
                kits.burn(self, a, t, 20.0, 2.0)
        for g in self.gadgets:
            if g.alive and g.team != a.team and g.kind != "mine":
                vx, vy, vz = g.x - ox, g.y - oy, g.z + 0.6 - oz
                d = math.sqrt(vx * vx + vy * vy + vz * vz)
                if 0.1 < d < w.rng and (vx * dx + vy * dy + vz * dz) / d > cosc:
                    self.hurt_gadget(g, w.dmg * dt)

    def _weapon(self, a: Actor, dt: float, can_act: bool) -> None:
        w = a.hero.weapon
        c = a.c
        rush = "rush" in a.fx
        rate = 1.5 if rush else (1.6 if "deadline" in a.fx else 1.0)
        a.fire_cd -= dt
        if a.reload_t > 0.0:
            a.reload_t -= dt
            if a.reload_t <= 0.0:
                a.ammo = float(w.mag)
        a.firing = False
        if w.kind == "cone":
            if c.fire and can_act and a.ammo > 0.0 and a.reload_t <= 0.0:
                a.ammo -= w.use * dt
                a.firing = True
                a.spray_t += dt
                while a.spray_t >= 0.1:
                    a.spray_t -= 0.1
                    self._cone_tick(a, 0.1)
                a.tick_t -= dt
                if a.tick_t <= 0.0:
                    a.tick_t = 0.22
                    self.sound(w.sound, a.x, a.y, 0.9 if a.is_player else 0.6)
        elif a.burst_left > 0:
            a.burst_t -= dt * rate
            while a.burst_left > 0 and a.burst_t <= 0.0:
                a.burst_left -= 1
                a.burst_t += w.burst_gap
                if a.ammo > 0.0:
                    self._fire_round(a)
        elif c.fire and can_act and a.fire_cd <= 0.0 and a.reload_t <= 0.0 and a.ammo > 0.0:
            a.fire_cd = 1.0 / (w.rate * rate)
            self._fire_round(a)
            if w.burst > 1:
                a.burst_left = w.burst - 1
                a.burst_t = w.burst_gap
        if a.reload_t <= 0.0 and a.burst_left == 0 and (a.ammo <= 0.0 or (c.reload and a.ammo < w.mag)):
            a.ammo = max(a.ammo, 0.0)
            a.reload_t = 0.05 if rush else w.reload
            if not rush:
                self.sound("reload", a.x, a.y, 0.8 if a.is_player else 0.35)

    def _melee(self, a: Actor) -> None:
        a.melee_cd = MELEE_CD
        a.melee_anim = 0.3
        dx, dy = math.cos(a.yaw), math.sin(a.yaw)
        best, bd = None, MELEE_RANGE
        for t in self.enemies(a):
            if not t.alive:
                continue
            vx, vy = t.x - a.x, t.y - a.y
            d = math.hypot(vx, vy)
            if d < bd + t.hit_r and abs(t.z - a.z) < 2.0 and (d < 0.6 or (vx * dx + vy * dy) / d > 0.6):
                best, bd = t, d
        self.sound("melee", a.x, a.y, 0.8)
        if best is not None:
            self.damage(a, best, MELEE_DMG, "hit", False, "Melee")
            best.kx += dx * 2.5
            best.ky += dy * 2.5
            return
        for g in self.gadgets:
            if g.alive and g.team != a.team and math.hypot(g.x - a.x, g.y - a.y) < MELEE_RANGE:
                self.hurt_gadget(g, MELEE_DMG)
                return

    # --- one hero, one frame --------------------------------------------------------------------------------------
    def _update_actor(self, a: Actor, dt: float) -> None:
        if not a.alive:
            a.death_t += dt
            a.respawn_t -= dt
            if a.respawn_t <= 0.0 and self.phase in ("live", "prep"):
                self.spawn(a)
                if a.is_player:
                    self.sound("respawn")
            return
        c = a.c
        fx = a.fx
        for k in list(fx):
            fx[k] -= dt
            if fx[k] <= 0.0:
                del fx[k]
                kits.ended(self, a, k)
                if not a.alive:
                    return
        h = a.hero
        live = self.phase == "live"
        a.protected = self.in_own_room(a)
        # cooldowns, charges, the ultimate
        for i, ab in enumerate((h.a1, h.a2)):
            if a.chg[i] < ab.charges:
                a.cd[i] -= dt * (4.0 if "rush" in fx and ab.id == "dash" else 1.0)
                if a.cd[i] <= 0.0:
                    a.chg[i] += 1
                    a.cd[i] = ab.cd if a.chg[i] < ab.charges else 0.0
                    if a.is_player:
                        self.sound("ready", None, None, 0.5)
        if live:
            before = a.ult_ready
            a.ult = min(a.ult + ULT_PASSIVE * dt, h.ult.cost)
            if a.ult_ready and not before and a.is_player:
                self.sound("ult_ready")
        # health over time
        if "burn" in fx:
            self.damage(a.burn_src, a, a.burn_dps * dt, "dot", False, "Burn")
            if not a.alive:
                return
        if "hot" in fx:
            self.heal(a.hot_src, a, a.hot_hps * dt)
        since = self.time - a.hurt_t
        if a.protected:
            self.heal(None, a, 200.0 * dt)
            a.shield = min(a.shield + 100.0 * dt, h.shield)
        else:
            if since > REGEN_DELAY:
                self.heal(None, a, REGEN_RATE * dt)
            if since > SHIELD_DELAY and a.shield < h.shield:
                a.shield = min(a.shield + SHIELD_RATE * dt, h.shield)
        if a.overhp > 0.0:
            a.overhp_t -= dt
            if a.overhp_t <= 0.0:
                a.overhp = max(a.overhp - OVERHEALTH_DECAY * dt, 0.0)
        # abilities
        stunned = "stun" in fx
        can_act = not stunned and "windup" not in fx
        if can_act:
            if c.a1 and not a.p_a1:
                kits.cast(self, a, 0)
            if c.a2 and not a.p_a2:
                kits.cast(self, a, 1)
            if c.ult and not a.p_ult and live:
                kits.cast(self, a, 2)
            a.melee_cd -= dt
            if c.melee and not a.p_melee and a.melee_cd <= 0.0 and "bulldoze" not in fx:
                self._melee(a)
        a.p_a1, a.p_a2, a.p_ult, a.p_melee = c.a1, c.a2, c.ult, c.melee
        kits.tick(self, a, dt)
        if not a.alive:
            return
        want_zoom = 1.0 if (c.alt and h.id == "flint" and can_act) else 0.0
        a.zoom += (want_zoom - a.zoom) * min(dt * 14.0, 1.0)
        self._weapon(a, dt, can_act and "bulldoze" not in fx)
        self._move(a, dt, stunned or "windup" in fx)
        # how it looks
        sp = math.hypot(a.vx, a.vy)
        a.move_amt += (min(sp / 5.0, 1.0) * (1.0 if a.on_ground else 0.3) - a.move_amt) * min(dt * 10.0, 1.0)
        a.walk += sp * dt * 2.1
        a.kick = max(a.kick - dt * 7.0, 0.0)
        a.flash = max(a.flash - dt, 0.0)
        a.cast_anim = max(a.cast_anim - dt * 2.5, 0.0)
        a.melee_anim = max(a.melee_anim - dt, 0.0)
        a.view_z += (a.z - a.view_z) * min(dt * 18.0, 1.0)
        if abs(a.z - a.view_z) > 1.2:
            a.view_z = a.z
        if a.hurt_dirs:
            for hd in a.hurt_dirs:
                hd[1] -= dt
            a.hurt_dirs = [hd for hd in a.hurt_dirs if hd[1] > 0.0]

    def _move(self, a: Actor, dt: float, frozen: bool) -> None:
        c = a.c
        fx = a.fx
        sp = a.hero.speed
        if "slow" in fx and "melt" not in fx:
            sp *= a.slow_f
        if "haste" in fx:
            sp *= a.haste_m
        if "rush" in fx:
            sp *= 1.35
        if "melt" in fx:
            sp *= 1.25
        mf, ms = c.mf, c.ms
        if "bulldoze" in fx:
            sp *= 1.7
            mf, ms = 1.0, ms * 0.3
        if a.zoom > 0.5:
            sp *= 0.7
        if frozen:
            sp = 0.0
        cy, sy = math.cos(a.yaw), math.sin(a.yaw)
        wx, wy = cy * mf - sy * ms, sy * mf + cy * ms
        n = math.hypot(wx, wy)
        if n > 1.0:
            wx, wy = wx / n, wy / n
        if mf < 0.0:
            sp *= 0.9
        acc = (70.0 if a.on_ground else 16.0) * dt
        dvx, dvy = wx * sp - a.vx, wy * sp - a.vy
        dn = math.hypot(dvx, dvy)
        if dn > acc:
            dvx, dvy = dvx / dn * acc, dvy / dn * acc
        a.vx += dvx
        a.vy += dvy
        if c.jump and a.on_ground and not frozen:
            a.vz = JUMP
            a.on_ground = False
        a.vz = max(a.vz - GRAVITY * dt, -30.0)
        k = math.exp(-(5.0 if a.on_ground else 1.2) * dt)
        a.kx *= k
        a.ky *= k
        mvx, mvy = a.vx + a.kx, a.vy + a.ky
        if a.dash_t > 0.0:
            a.dash_t -= dt
            mvx, mvy = a.dash_vx, a.dash_vy
            a.vz = max(a.vz, 0.0)
        w = self.world
        r, hgt, team = a.radius, a.height, a.team
        n = int(max(abs(mvx), abs(mvy), abs(a.vz)) * dt / 0.4) + 1
        hdt = dt / n
        a.blocked = False
        was_ground = a.on_ground
        for _ in range(n):
            room = w.room_at(a.x, a.y)
            nx = a.x + mvx * hdt
            if w.free(nx, a.y, a.z, r, hgt, team, room):
                a.x = nx
            else:
                a.vx = a.kx = 0.0
                a.blocked = True
            ny = a.y + mvy * hdt
            if w.free(a.x, ny, a.z, r, hgt, team, room):
                a.y = ny
            else:
                a.vy = a.ky = 0.0
                a.blocked = True
            g = w.ground(a.x, a.y, a.z, r)
            nz = a.z + a.vz * hdt
            if a.vz > 0.0:
                top = w.ceiling(a.x, a.y, a.z, r)
                if nz + hgt > top:
                    nz = top - hgt
                    a.vz = 0.0
            if nz <= g:
                nz = g
                a.vz = 0.0
                a.on_ground = True
            elif was_ground and a.vz <= 0.0 and nz - g <= STEP + 0.05:
                nz = g  # walking down a step
                a.vz = 0.0
                a.on_ground = True
            else:
                a.on_ground = False
            a.z = nz

    # --- shots in flight, gadgets, zones ----------------------------------------------------------------------------
    def _update_projectiles(self, dt: float) -> None:
        w = self.world
        keep = []
        for p in self.projectiles:
            p.age += dt
            alive = p.age < p.life
            speed = math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz)
            n = int(speed * dt / 0.9) + 1
            hdt = dt / n
            for _ in range(n):
                if not alive:
                    break
                p.vz -= p.g * hdt
                sx, sy, sz = p.vx * hdt, p.vy * hdt, p.vz * hdt
                d = math.sqrt(sx * sx + sy * sy + sz * sz)
                if d < 1e-9:
                    break
                dx, dy, dz = sx / d, sy / d, sz / d
                limit = w.raycast(p.x, p.y, p.z, dx, dy, dz, d)
                target = None
                head = False
                wall = None
                for b in self.barriers:
                    if b.team != p.team:
                        t = b.ray(p.x, p.y, p.z, dx, dy, dz, limit)
                        if t is not None:
                            limit, wall = t, b
                for t in self.enemies(p.owner):
                    if t.alive:
                        res = self.ray_body(t, p.x, p.y, p.z, dx, dy, dz, limit, 0.08)
                        if res is not None:
                            limit, target, wall = res[0], t, None
                            head = res[1]
                for g in self.gadgets:
                    if g.alive and g.team != p.team and g.kind != "mine":
                        res = self.ray_body(g, p.x, p.y, p.z, dx, dy, dz, limit)
                        if res is not None:
                            limit, target, wall = res[0], g, None
                if limit < d - 1e-6 or target is not None or wall is not None:
                    back = 0.05
                    p.x, p.y, p.z = p.x + dx * (limit - back), p.y + dy * (limit - back), p.z + dz * (limit - back)
                    kits.impact(self, p, target, wall, head)
                    alive = False
                else:
                    p.x, p.y, p.z = p.x + sx, p.y + sy, p.z + sz
            if alive:
                keep.append(p)
        self.projectiles = keep

    def _update_things(self, dt: float) -> None:
        for b in self.barriers:
            b.life -= dt
            b.flash = max(b.flash - dt, 0.0)
            if b.hp <= 0.0 and b.life > 0.0:
                b.life = 0.0
                self.sound("break", b.x, b.y)
                self.add_fx("boom", (b.x, b.y, b.z + 1.5), None, (150, 210, 255), 0.3, 2.5)
        self.barriers = [b for b in self.barriers if b.life > 0.0]
        for g in self.gadgets:
            g.age += dt
            g.flash = max(g.flash - dt, 0.0)
            if g.age >= g.life:
                g.alive = False
            if g.alive:
                kits.gadget_tick(self, g, dt)
        self.gadgets = [g for g in self.gadgets if g.alive]
        for z in self.zones:
            z.age += dt
            z.tick_t -= dt
            if z.tick_t <= 0.0:
                z.tick_t += 0.25
                kits.zone_tick(self, z, 0.25)
        self.zones = [z for z in self.zones if z.age < z.life]
        if self.delayed:
            due = [d for d in self.delayed if d[0] <= self.time]
            self.delayed = [d for d in self.delayed if d[0] > self.time]
            for _t, fn in due:
                fn()
        for f in self.fx:
            f.t += dt
        self.fx = [f for f in self.fx if f.t < f.life]
        for pk in self.packs:
            if pk["t"] > 0.0:
                pk["t"] -= dt
                continue
            for a in self.actors:
                if a.alive and a.hp + a.armor < a.hero.hp + a.hero.armor - 1.0 and \
                        abs(a.x - pk["x"]) < 1.1 and abs(a.y - pk["y"]) < 1.1 and abs(a.z - pk["z"]) < 1.5:
                    amount, back = PACK_BIG if pk["big"] else PACK_SMALL
                    self.heal(None, a, amount)
                    pk["t"] = back
                    self.sound("pack", a.x, a.y)
                    break

    def _separate(self) -> None:
        """Enemies cannot stand in each other, and nobody stands in the cart."""
        w = self.world
        if self.mode == "escort":
            cx, cy, _yaw = track_at(self.md.track, self.cart_d)
            for a in self.actors:
                if a.alive and a.z < 2.2:
                    dx, dy = a.x - cx, a.y - cy
                    d2 = dx * dx + dy * dy
                    if 1e-6 < d2 < 2.56:
                        d = math.sqrt(d2)
                        nx, ny = a.x + dx / d * (1.6 - d), a.y + dy / d * (1.6 - d)
                        if w.free(nx, ny, a.z, a.radius, a.height, a.team, w.room_at(a.x, a.y)):
                            a.x, a.y = nx, ny
        for a in self.team_lists[0]:
            if not a.alive:
                continue
            for b in self.team_lists[1]:
                if not b.alive or abs(a.z - b.z) > 1.6:
                    continue
                dx, dy = b.x - a.x, b.y - a.y
                d2 = dx * dx + dy * dy
                m = a.radius + b.radius
                if 1e-6 < d2 < m * m:
                    d = math.sqrt(d2)
                    push = (m - d) * 0.5
                    ux, uy = dx / d * push, dy / d * push
                    if w.free(a.x - ux, a.y - uy, a.z, a.radius, a.height, a.team, w.room_at(a.x, a.y)):
                        a.x, a.y = a.x - ux, a.y - uy
                    if w.free(b.x + ux, b.y + uy, b.z, b.radius, b.height, b.team, w.room_at(b.x, b.y)):
                        b.x, b.y = b.x + ux, b.y + uy

    # --- the objective ----------------------------------------------------------------------------------------------
    def _near(self, x: float, y: float, z: float, r: float) -> list[int]:
        n = [0, 0]
        for a in self.actors:
            if a.alive and (a.x - x) ** 2 + (a.y - y) ** 2 <= r * r and abs(a.z - z) < 3.0:
                n[a.team] += 1
                a.stats.obj_time += 1.0 / 60.0
        return n

    def _control(self, dt: float) -> None:
        px, py, pr = self.md.point
        on = self._near(px, py, self.world.floor_at(px, py), pr)
        self.on_point = on
        for t in (0, 1):
            self.touch[t] = 0.0 if on[t] else self.touch[t] + dt
        if self.live_t < POINT_UNLOCK:
            return
        if self.live_t - dt < POINT_UNLOCK:
            self.banner("THE POINT IS OPEN", "Take it and hold it", (255, 196, 64))
            self.sound("unlock")
        self.contested = on[0] > 0 and on[1] > 0
        if not self.contested and (on[0] or on[1]):
            t = 0 if on[0] else 1
            if t != self.owner:
                if self.cap_team != t:
                    self.cap_team, self.cap = t, 0.0
                self.cap += dt / CAPTURE_TIME * (1.0, 1.5, 2.0)[min(on[t], 3) - 1]
                if self.cap >= 1.0:
                    self.owner, self.cap, self.cap_team = t, 0.0, -1
                    mine = t == self.player.team
                    self.banner("POINT CAPTURED" if mine else "POINT LOST",
                                "Hold it" if mine else "Take it back", (110, 190, 255) if mine else (255, 110, 90))
                    self.sound("capture_us" if mine else "capture_them")
            else:
                self.cap = max(self.cap - dt / 1.5, 0.0)
        elif not self.contested:
            self.cap = max(self.cap - dt / 5.0, 0.0)
        if self.owner >= 0:
            o = self.owner
            fuse = max(0.6, 3.0 - self.ot_t * 0.12)
            pressed = self.touch[1 - o] < fuse
            if self.score[o] < 99.0:
                self.score[o] = min(self.score[o] + CONTROL_RATE * dt, 99.0)
            elif pressed:
                if not self.overtime:
                    self.overtime = True
                    self.banner("OVERTIME", "", (255, 196, 64), 2.0)
                    self.sound("overtime")
                self.ot_t += dt
            else:
                self.score[o] = 100.0
                self._round_won(o)
                return
        if self.owner < 0 or self.score[self.owner] < 99.0:
            self.overtime = False
            self.ot_t = 0.0

    def _round_won(self, team: int) -> None:
        self.wins[team] += 1
        self.overtime = False
        mine = team == self.player.team
        if self.wins[team] >= ROUNDS_TO_WIN:
            self._finish(team)
            return
        self.phase, self.phase_t = "between", 6.0
        self.banner("ROUND WON" if mine else "ROUND LOST", f"{self.wins[0]} - {self.wins[1]}",
                    (110, 190, 255) if mine else (255, 110, 90), 5.0)
        self.sound("round_win" if mine else "round_lose")

    def _next_round(self) -> None:
        self.round += 1
        self.owner, self.cap_team, self.cap = -1, -1, 0.0
        self.score = [0.0, 0.0]
        self.ot_t = 0.0
        self.live_t = 0.0
        self.touch = [99.0, 99.0]
        self.projectiles.clear()
        self.barriers.clear()
        self.gadgets.clear()
        self.zones.clear()
        self.delayed.clear()
        self.phase, self.phase_t = "prep", 5.0
        self._set_rooms()
        for a in self.actors:
            ult = a.ult
            self.spawn(a)
            a.ult = ult
        self.banner(f"ROUND {self.round}", "Take the point", (255, 255, 255), 3.0)

    def _escort(self, dt: float) -> None:
        cx, cy, _yaw = track_at(self.md.track, self.cart_d)
        near = self._near(cx, cy, 0.0, CART_RADIUS)
        att, dfn = near[self.attack_team], near[1 - self.attack_team]
        self.on_point = near
        self.pushers = att
        self.contested = att > 0 and dfn > 0
        self.touch[self.attack_team] = 0.0 if att else self.touch[self.attack_team] + dt
        last_cp = self.md.checkpoints[self.cp - 1] if self.cp > 0 else 0.0
        self.cart_moving = 0.0
        if att > 0 and dfn == 0:
            self.cart_moving = CART_SPEED[min(att, 3)]
            self.cart_d += self.cart_moving * dt
            self.no_push = 0.0
            for a in self.team_lists[self.attack_team]:
                if a.alive and (a.x - cx) ** 2 + (a.y - cy) ** 2 <= CART_RADIUS ** 2:
                    self.heal(None, a, 6.0 * dt)
        elif att == 0:
            self.no_push += dt
            if self.no_push > 6.0 and self.cart_d > last_cp:
                self.cart_moving = -CART_ROLLBACK
                self.cart_d = max(self.cart_d - CART_ROLLBACK * dt, last_cp)
        self.best_d = max(self.best_d, self.cart_d)
        if self.cart_d >= self.md.checkpoints[self.cp]:
            self.cp += 1
            if self.cp >= len(self.md.checkpoints):
                self.cart_d = self.cart_len
                self.cp = len(self.md.checkpoints) - 1
                self._finish(self.attack_team)
                return
            self.time_left += ESCORT_BONUS
            self.overtime = False
            self.ot_t = 0.0
            self._set_rooms()
            mine = self.attack_team == self.player.team
            self.banner("CHECKPOINT REACHED" if mine else "CHECKPOINT LOST", f"+{int(ESCORT_BONUS)} seconds",
                        (110, 190, 255) if mine else (255, 110, 90))
            self.sound("checkpoint")
        if not self.overtime:
            self.time_left -= dt
        if self.time_left <= 0.0:
            self.time_left = 0.0
            fuse = max(0.6, 3.0 - self.ot_t * 0.1)
            if self.touch[self.attack_team] < fuse:
                if not self.overtime:
                    self.overtime = True
                    self.banner("OVERTIME", "", (255, 196, 64), 2.0)
                    self.sound("overtime")
                self.ot_t += dt
            else:
                self._finish(1 - self.attack_team)

    def _finish(self, team: int) -> None:
        self.winner = team
        self.phase = "over"
        self.over_t = 0.0
        self.overtime = False
        self.world.closed = set()
        self.sound("victory" if team == self.player.team else "defeat")
        for a in self.actors:
            a.c = Controls()
            a.firing = False

    # --- one frame ----------------------------------------------------------------------------------------------------
    def update(self, dt: float) -> None:
        self.time += dt
        if self.hit_t > 0.0:
            self.hit_t -= dt
        if self.kill_t > 0.0:
            self.kill_t -= dt
        if self.heal_t > 0.0:
            self.heal_t -= dt
        for e in self.elim_text:
            e[1] -= dt
        self.elim_text = [e for e in self.elim_text if e[1] > 0.0]
        for b in self.banners:
            b[3] -= dt
        for cl in self.calls:
            cl[2] -= dt
        self.calls = [cl for cl in self.calls if cl[2] > 0.0]
        self.banners = [b for b in self.banners if b[3] > 0.0]
        if self.phase == "over":
            self.over_t += dt
            for f in self.fx:
                f.t += dt
            self.fx = [f for f in self.fx if f.t < f.life]
            return
        if self.phase == "prep":
            self.phase_t -= dt
            if self.phase_t <= 0.0:
                self.phase = "live"
                self.world.closed = set()
                self.sound("go")
                if self.mode == "control":
                    self.banner("DOORS OPEN", "The point opens in a moment", (255, 255, 255), 2.5)
                else:
                    self.banner("ESCORT THE CART" if self.attack_team == self.player.team else "STOP THE CART",
                                "", (255, 255, 255), 2.5)
        elif self.phase == "between":
            self.phase_t -= dt
            if self.phase_t <= 0.0:
                self._next_round()
        for a in self.actors:
            if a.brain is not None and a.alive:
                a.brain.update(self, dt)
        for a in self.actors:
            self._update_actor(a, dt)
        self._separate()
        self._update_projectiles(dt)
        self._update_things(dt)
        if self.phase == "live":
            self.live_t += dt
            if self.mode == "control":
                self._control(dt)
            else:
                self._escort(dt)

    # --- for the scoreboard and the save --------------------------------------------------------------------------------
    def progress(self) -> float:
        """Escort: how far along the whole track the cart is (0-1)."""
        return self.cart_d / max(self.cart_len, 1e-6)

    def result(self) -> dict:
        p = self.player
        return {"won": self.winner == p.team, "mode": self.mode, "map": self.md.key, "hero": p.hero.id,
                "elims": p.stats.elims, "final": p.stats.final, "deaths": p.stats.deaths,
                "assists": p.stats.assists, "damage": int(p.stats.damage), "healing": int(p.stats.healing),
                "time": self.time, "ults": p.stats.ults}
