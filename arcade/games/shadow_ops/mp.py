"""MULTIPLAYER: Team Deathmatch in the CROSSDOCK freight yard, the player and five VANGUARD bots against six ONYX bots. Bots
patrol between waypoints on flow fields, spot enemies in their view cone (or by sound), react, strafe and fire
with weapon-accurate rates and damage. Kills score for the team; the player earns score pop-ups, medals,
multi-kills, XP and three chosen scorestreaks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from .data import MP_KNIFE_DAMAGE, STREAKS, XP, weapon
from .maps import MapData
from .gfx import Sprite
from .sprites import SpriteSet
from .world import (Controls, Effect, Grenade, Gun, Player, Target, angle_diff, falloff, flow_dir, flow_field,
                    line_of_sight, open_facing, shoot_ray, slide, throw)

SCORE_LIMIT = 40
TIME_LIMIT = 8 * 60.0
BOT_GUNS = ("saber", "kestrel", "volley16", "tempest", "mantis", "ridgeback", "arcturus", "hammerfall", "oldguard")
NAMES = ("Rook", "Viper", "Mako", "Brick", "Havoc", "Nomad", "Jinx", "Dozer", "Kodiak", "Wraith", "Ember", "Tusk",
         "Cinder", "Pike", "Gauge", "Talon", "Rivet", "Sable")
TEAM_NAMES = {"ally": "VANGUARD", "enemy": "ONYX"}


@dataclass
class Bot:
    name: str
    team: str
    x: float
    y: float
    angle: float
    wid: str
    hp: float = 100.0
    dead: bool = False
    respawn: float = 0.0
    target: object = None
    react: float = 0.0
    seen_t: float = 0.0
    goal: int = 0
    mag: int = 30
    reload_t: float = 0.0
    fire_cd: float = 0.0
    burst: int = 0
    strafe: float = 1.0
    strafe_t: float = 0.0
    phase: float = 0.0
    flash_t: float = 0.0
    fired_t: float = 99.0  # since last shot (minimap)
    blind: float = 0.0
    stun: float = 0.0
    kills: int = 0
    deaths: int = 0
    score: int = 0
    streak: int = 0
    hit_flash: float = 0.0
    dead_t: float = 0.0
    moving: bool = False
    last_x: float = 0.0
    last_y: float = 0.0
    stuck: float = 0.0
    last_hit_by: object = None


@dataclass
class ScorePop:
    label: str
    points: int
    t: float = 0.0


@dataclass
class Feed:
    killer: str
    kteam: str
    victim: str
    vteam: str
    weapon: str
    head: bool
    t: float = 0.0


class MPMode:
    kind = "mp"

    def __init__(self, m: MapData, sprites: SpriteSet, sfx, cls: dict, streaks: list, level: int,  # noqa: ANN001
                 prestige: int) -> None:
        self.m = m
        self.sprites = sprites
        self.sfx = sfx or (lambda name, volume=1.0: None)
        self.cls = cls
        self.streak_ids = sorted([s for s in streaks if s in STREAKS], key=lambda s: STREAKS[s].cost)[:3]
        self.perks = set(cls.get("perk1", []) + cls.get("perk2", []) + cls.get("perk3", []))
        self.time = 0.0
        self.time_left = TIME_LIMIT
        self.scores = {"ally": 0, "enemy": 0}
        self.over = False
        self.over_t = 0.0
        self.won: bool | None = None
        self.effects: list[Effect] = []
        self.grenades: list[Grenade] = []
        self.feed: list[Feed] = []
        self.pops: list[ScorePop] = []
        self.medal = ("", 0.0)
        self.message = ("", 0.0)
        self.hit_marker = 0.0
        self.hit_marker_kill = False
        self.xp_pending: list[int] = []
        self.walk = ~m.block
        self.player = Player(0, 0, 0)
        self.me = Bot("You", "ally", 0, 0, 0, cls.get("primary") or "saber")
        self.p_kills = self.p_deaths = self.p_score = self.p_assists = self.p_headshots = 0
        self.life_score = 0
        self.life_kills = 0
        self.best_streak = 0
        self.multi = []  # times of recent kills
        self.dead_t = 0.0
        self.killed_by = ""
        self.first_blood = False
        self.earned: list[str] = []  # streaks ready to call in
        self.got_this_life: set[str] = set()
        self.active: dict[str, float] = {}  # recon / counter / swarm / enemy_recon timers
        self.sentries: list[list[float]] = []  # [x, y, time left, fire cooldown]
        self.crates: list[list[float]] = []  # [x, y, z, landed]
        self.missiles: list[list] = []  # [target bot, time left, kind]
        self.last_killer: Bot | None = None
        # waypoints spread over the map and their flow fields (made on demand)
        self.waypoints = []
        for y in range(2, m.h - 1, 4):
            for x in range(2, m.w - 1, 4):
                cell = next(((x + dx, y + dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1))
                             if m.walkable(x + dx, y + dy)), None)
                if cell:
                    self.waypoints.append(cell)
        self._fields: dict = {}
        names = list(NAMES)
        random.shuffle(names)
        self.bots: list[Bot] = []
        for i in range(11):
            team = "ally" if i < 5 else "enemy"
            b = Bot(names[i], team, 0, 0, 0, random.choice(BOT_GUNS))
            self.bots.append(b)
            self._spawn_bot(b, initial=True)
        self._loadout()
        self._spawn_player()

    # --- setup ---------------------------------------------------------------------------------------------------

    def _loadout(self) -> None:
        c = self.cls
        p = self.player
        p.guns = []
        if c.get("primary"):
            p.guns.append(Gun(c["primary"], atts=list(c.get("primary_att", []))))
        if c.get("secondary"):
            p.guns.append(Gun(c["secondary"], atts=list(c.get("secondary_att", []))))
        if not p.guns:
            p.guns.append(Gun("service45"))
        p.cur = 0
        lethal = c.get("lethal", [])
        tact = c.get("tactical", [])
        p.grenades = len(lethal)
        p.tacticals = len(tact)
        self.lethal_kind = lethal[0] if lethal else "frag"
        self.tactical_kind = tact[0] if tact else "flash"
        perks = self.perks
        p.speed_mult = 1.07 if "featherstep" in perks else 1.0
        p.infinite_sprint = "endurance" in perks
        p.nimble = "nimble" in perks
        p.flinch_mult = 0.3 if "steelnerve" in perks else 1.0
        p.quick_hands = "quickhands" in perks
        p.regen_delay = 4.0
        p.max_health = 100.0

    def _spawn_points(self, team: str) -> list:
        return self.m.spawns["ally" if team == "ally" else "enemy"]

    def _pick_spawn(self, team: str) -> tuple[float, float]:
        foes = [b for b in self.bots if b.team != team and not b.dead]
        if team != "ally" and not self.player.dead:
            foes_xy = [(b.x, b.y) for b in foes] + [(self.player.x, self.player.y)]
        else:
            foes_xy = [(b.x, b.y) for b in foes]
        best, bd = None, -1.0
        for sx, sy in self._spawn_points(team):
            d = min((math.hypot(sx - fx, sy - fy) for fx, fy in foes_xy), default=99.0)
            d += random.uniform(0, 3)
            if d > bd:
                best, bd = (sx, sy), d
        return best or (self.m.w / 2, self.m.h / 2)

    def _spawn_bot(self, b: Bot, initial: bool = False) -> None:
        pts = self._spawn_points(b.team)
        x, y = random.choice(pts) if initial else self._pick_spawn(b.team)
        b.x, b.y = x + random.uniform(-0.2, 0.2), y + random.uniform(-0.2, 0.2)
        b.angle = open_facing(self.m, b.x, b.y, math.atan2(self.m.h / 2 - b.y, self.m.w / 2 - b.x))
        b.hp = 100.0
        b.dead = False
        b.target = None
        b.react = 0.0
        b.mag = weapon(b.wid).mag
        b.reload_t = 0.0
        b.streak = 0
        b.goal = random.randrange(len(self.waypoints))
        b.blind = b.stun = 0.0

    def _spawn_player(self) -> None:
        p = self.player
        x, y = self._pick_spawn("ally")
        p.x, p.y = x, y
        p.angle = open_facing(self.m, x, y, math.atan2(self.m.h / 2 - y, self.m.w / 2 - x))
        p.look = 0.0
        p.health = p.max_health
        p.dead = False
        p.hurt_t = 99.0
        p.damage_dirs = []
        self._loadout()
        self.life_score = 0
        self.life_kills = 0
        self.got_this_life = set()

    def _field(self, i: int) -> np.ndarray:
        f = self._fields.get(i)
        if f is None:
            f = self._fields[i] = flow_field(self.m, [self.waypoints[i]], self.walk)
        return f

    # --- the frame -----------------------------------------------------------------------------------------------

    def update(self, dt: float, c: Controls, use_held: bool) -> None:
        self.time += dt
        if self.over:
            self.over_t += dt
            self._tick_lists(dt)
            return
        self.time_left -= dt
        if self.time_left <= 0:
            self._end(self.scores["ally"] > self.scores["enemy"] if self.scores["ally"] != self.scores["enemy"]
                      else None)
            return
        p = self.player
        if p.dead:
            self.dead_t += dt
            if self.dead_t > 3.2:
                self._spawn_player()
                self.sfx("spawn", 0.6)
        else:
            p.move(self.m, c, dt)
            shots = p.update_weapon(dt, c.fire)
            if shots < 0:
                self.sfx("dry")
            for _ in range(max(shots, 0)):
                self._fire()
            self.me.x, self.me.y, self.me.angle = p.x, p.y, p.angle
            if shots > 0 and not (p.gun and "suppressor" in p.gun.atts):
                self.me.fired_t = 0.0
            if use_held:
                self._use_crate()
        self.me.fired_t += dt
        for b in self.bots:
            self._update_bot(b, dt)
        self._update_grenades(dt)
        self._update_streaks(dt)
        self._tick_lists(dt)

    def _tick_lists(self, dt: float) -> None:
        for e in self.effects:
            e.t += dt
        self.effects = [e for e in self.effects if e.t < e.life]
        for f in self.feed:
            f.t += dt
        self.feed = [f for f in self.feed if f.t < 7.0][-6:]
        for s in self.pops:
            s.t += dt
        self.pops = [s for s in self.pops if s.t < 2.2]
        self.hit_marker = max(self.hit_marker - dt, 0.0)
        if self.medal[1] > 0:
            self.medal = (self.medal[0], self.medal[1] - dt)
        if self.message[1] > 0:
            self.message = (self.message[0], self.message[1] - dt)
        for k in list(self.active):
            self.active[k] -= dt
            if self.active[k] <= 0:
                del self.active[k]

    # --- the player's weapons ------------------------------------------------------------------------------------

    def targets(self) -> list[Target]:
        return [Target(b, b.x, b.y, 0.25, 0.0, 1.04, 0.2) for b in self.bots if b.team == "enemy" and not b.dead]

    def _fire(self) -> None:
        p = self.player
        gun = p.gun
        w = gun.w
        self.sfx(w.sound, 0.8 if "suppressor" in gun.atts else 1.0)
        eye = p.view_eye()
        targets = self.targets()
        hit_any = False
        for yaw, slope in p.pellet_dirs():
            hit = shoot_ray(self.m, p.x, p.y, eye, yaw, slope, targets)
            self.effects.append(Effect("tracer", p.x, p.y, eye - 0.08, 0.0, 0.05, hit.x, hit.y, hit.z))
            if w.splash > 0:
                self._explode(hit.x - math.cos(yaw) * 0.15, hit.y - math.sin(yaw) * 0.15, w.splash, w.dmg, self.me,
                              w.name)
                continue
            if hit.target is None:
                self.effects.append(Effect("puff", hit.x, hit.y, hit.z, 0.0, 0.25))
                continue
            b = hit.target.obj
            dmg = falloff(w, hit.dist, "long" in gun.atts) * (1.15 if "fmj" in gun.atts else 1.0)
            if hit.head:
                dmg *= w.head
            self.effects.append(Effect("blood", hit.x, hit.y, hit.z, 0.0, 0.3))
            self._hurt_bot(b, dmg, self.me, w.name, hit.head)
            hit_any = True
        if hit_any:
            p.hits += 1

    def _hurt_bot(self, b: Bot, dmg: float, by: Bot, wname: str, head: bool = False) -> None:
        if b.dead:
            return
        b.hp -= dmg
        b.hit_flash = 0.15
        b.last_hit_by = by
        if by is self.me:
            self.hit_marker = 0.18
            self.hit_marker_kill = b.hp <= 0
            self.sfx("hitmarker", 0.7)
        if b.target is None and not b.dead:
            b.target = by
            b.react = 0.25
        if b.hp <= 0:
            self._kill(b, by, wname, head)

    def _kill(self, b: Bot, by: Bot, wname: str, head: bool) -> None:
        b.dead = True
        b.dead_t = 0.0
        b.respawn = 3.0
        b.deaths += 1
        b.streak = 0
        team = "enemy" if b.team == "ally" else "ally"
        self.scores[team] += 1
        self.feed.append(Feed(by.name, by.team, b.name, b.team, wname, head))
        if by is self.me:
            self._player_kill(b, head)
        else:
            by.kills += 1
            by.score += 100
            by.streak += 1
            if by.team == "enemy" and by.streak == 3 and "enemy_recon" not in self.active:
                self.active["enemy_recon"] = 20.0
                self.message = ("Enemy Recon Drone spotted", 3.0)
                self.sfx("uav", 0.6)
            if by.team == "enemy" and b.last_hit_by is self.me and b.hp > -200:
                self.p_assists += 1
                self._score("Assist", XP["assist"])
        if self.scores[team] >= SCORE_LIMIT:
            self._end(team == "ally")

    def _player_kill(self, b: Bot, head: bool) -> None:
        self.p_kills += 1
        self.me.kills += 1
        self.life_kills += 1
        self.best_streak = max(self.best_streak, self.life_kills)
        self.sfx("kill_ding", 0.8)
        self._score("Enemy Killed", XP["kill"])
        if head:
            self.p_headshots += 1
            self.player.headshots += 1
            self._score("Headshot", XP["headshot"])
            self.sfx("headshot", 0.7)
        if not self.first_blood:
            self.first_blood = True
            self._score("First Blood", XP["first_blood"])
        if self.last_killer is b:
            self._score("Revenge", XP["revenge"])
            self.last_killer = None
        self.multi = [t for t in self.multi if self.time - t < 4.0] + [self.time]
        n = len(self.multi)
        if n >= 2:
            name, pts = {2: ("Double Kill", XP["double"]), 3: ("Triple Kill", XP["triple"])}.get(
                n, ("Multi Kill", XP["multi"]))
            self._score(name, pts)
            self.medal = (name.upper(), 2.0)
        if self.life_kills in (5, 10, 15, 20):
            self.medal = ({5: "KILLSTREAK 5", 10: "KILLSTREAK 10", 15: "RAMPAGE", 20: "UNSTOPPABLE"}[self.life_kills],
                          2.2)
        if "scrounger" in self.perks and self.player.gun is not None:
            g = self.player.gun
            g.reserve = min(g.reserve + g.mag_size(), g.reserve_max())

    def _score(self, label: str, pts: int) -> None:
        self.pops.append(ScorePop(label, pts))
        self.p_score += pts
        self.me.score += pts
        self.life_score += pts
        self.xp_pending.append(pts)
        mult = 0.85 if "hardbargain" in self.perks else 1.0
        for sid in self.streak_ids:
            cost = int(STREAKS[sid].cost * mult)
            if self.life_score >= cost and sid not in self.got_this_life:
                self.got_this_life.add(sid)
                self.earned.append(sid)
                self.message = (f"{STREAKS[sid].name} ready! Press {self.streak_ids.index(sid) + 3}", 3.0)
                self.sfx("streak_ready")

    def take_xp(self) -> list[int]:
        out, self.xp_pending = self.xp_pending, []
        return out

    def knife(self) -> None:
        p = self.player
        if p.knife_t > 0 or p.dead or self.over:
            return
        p.knife_t = 0.45
        p.reload_t = 0.0
        best, bd = None, 1.5
        for b in self.bots:
            if b.team != "enemy" or b.dead:
                continue
            d = math.hypot(b.x - p.x, b.y - p.y)
            if d < bd and abs(angle_diff(math.atan2(b.y - p.y, b.x - p.x), p.angle)) < 0.7:
                best, bd = b, d
        self.sfx("knife_hit" if best else "knife_swing")
        if best is not None:
            self._hurt_bot(best, MP_KNIFE_DAMAGE, self.me, "Knife")

    def throw_grenade(self) -> None:
        p = self.player
        if p.grenades <= 0 or p.throw_t > 0 or p.dead or self.over:
            return
        p.grenades -= 1
        p.throw_t = 0.5 if not getattr(p, "quick_hands", False) else 0.3
        kind = self.lethal_kind
        if kind == "hatchet":
            hit = shoot_ray(self.m, p.x, p.y, p.view_eye(), p.angle, p.look, self.targets(), 16.0)
            if hit.target is not None:
                self._hurt_bot(hit.target.obj, 200, self.me, "Hatchet", hit.head)
            self.sfx("knife_swing")
            return
        g = throw(p, kind, "ally")
        g.owner = self.me
        self.grenades.append(g)
        self.sfx("throw")

    def throw_tactical(self) -> None:
        p = self.player
        if p.tacticals <= 0 or p.throw_t > 0 or p.dead or self.over:
            return
        p.tacticals -= 1
        p.throw_t = 0.5
        g = throw(p, self.tactical_kind, "ally")
        g.owner = self.me
        self.grenades.append(g)
        self.sfx("throw")

    def _update_grenades(self, dt: float) -> None:
        for g in list(self.grenades):
            if not g.update(self.m, dt):
                continue
            self.grenades.remove(g)
            if g.kind in ("flash", "stun"):
                self.sfx("flashbang", 0.8)
                for b in self.bots:
                    if not b.dead and math.hypot(b.x - g.x, b.y - g.y) < 6 and line_of_sight(self.m, g.x, g.y, b.x,
                                                                                              b.y):
                        if g.kind == "flash":
                            b.blind = 3.0
                        else:
                            b.stun = 4.0
                self.effects.append(Effect("boom", g.x, g.y, 0.4, 0.0, 0.25, color=(255, 255, 255), size=2.5))
                continue
            self._explode(g.x, g.y, 2.8, 150.0, g.owner if isinstance(g.owner, Bot) else self.me, "Frag")

    def _explode(self, x: float, y: float, radius: float, dmg: float, by: Bot, wname: str) -> None:
        self.effects.append(Effect("boom", x, y, 0.3, 0.0, 0.6, color=(255, 170, 60), size=radius))
        d = math.hypot(self.player.x - x, self.player.y - y)
        self.sfx("explosion", max(0.2, 1 - d / 25))
        for b in self.bots:
            if b.dead or (b.team == by.team and b is not by):
                continue
            dd = math.hypot(b.x - x, b.y - y)
            if dd < radius and line_of_sight(self.m, x, y, b.x, b.y):
                self._hurt_bot(b, dmg * (1 - 0.6 * dd / radius), by, wname)
        p = self.player
        if not p.dead and (by is self.me or by.team == "enemy") and d < radius and line_of_sight(self.m, x, y, p.x,
                                                                                                    p.y):
            k = 0.5 if "blastvest" in self.perks else 1.0
            self._hurt_player(dmg * (1 - 0.6 * d / radius) * k * (0.5 if by is self.me else 1.0), by, wname)

    def _hurt_player(self, dmg: float, by: Bot, wname: str, head: bool = False) -> None:
        p = self.player
        if p.dead:
            return
        p.hurt(dmg, (by.x, by.y))
        self.sfx("player_hit", 0.6)
        if p.dead:
            self.p_deaths += 1
            self.me.deaths += 1
            self.dead_t = 0.0
            self.killed_by = f"Killed by {by.name}  ·  {wname}" if by is not self.me else "You killed yourself"
            if by is not self.me:
                by.kills += 1
                by.streak += 1
                by.score += 100
                self.last_killer = by
                self.scores["enemy"] += 1
                if self.scores["enemy"] >= SCORE_LIMIT:
                    self._end(False)
            self.feed.append(Feed(by.name, by.team, "You", "ally", wname, head))
            self.life_kills = 0
            self.multi = []

    # --- bots ----------------------------------------------------------------------------------------------------

    def _foes(self, b: Bot) -> list[Bot]:
        out = [o for o in self.bots if o.team != b.team and not o.dead]
        if b.team == "enemy" and not self.player.dead:
            out.append(self.me)
        return out

    def _update_bot(self, b: Bot, dt: float) -> None:
        b.hit_flash = max(b.hit_flash - dt, 0.0)
        b.flash_t = max(b.flash_t - dt, 0.0)
        b.fired_t += dt
        if b.dead:
            b.dead_t += dt
            b.respawn -= dt
            if b.respawn <= 0:
                self._spawn_bot(b)
            return
        b.blind = max(b.blind - dt, 0.0)
        b.stun = max(b.stun - dt, 0.0)
        w = weapon(b.wid)
        # reload
        if b.reload_t > 0:
            b.reload_t -= dt
            if b.reload_t <= 0:
                b.mag = w.mag
        b.fire_cd = max(b.fire_cd - dt, 0.0)
        # look for enemies (a few times a second)
        b.seen_t -= dt
        if b.seen_t <= 0:
            b.seen_t = 0.15
            self._scan(b)
        tgt = b.target
        speed = 2.3 * w.speed * (0.45 if b.stun > 0 else 1.0)
        mvx = mvy = 0.0
        if tgt is not None and b.blind <= 0:
            tx, ty = tgt.x, tgt.y
            want = math.atan2(ty - b.y, tx - b.x)
            turn = 6.0 * dt * (0.4 if b.stun > 0 else 1.0)
            b.angle += max(min(angle_diff(want, b.angle), turn), -turn)
            b.react -= dt
            dist = math.hypot(tx - b.x, ty - b.y)
            # strafe while fighting
            b.strafe_t -= dt
            if b.strafe_t <= 0:
                b.strafe = random.choice((-1.0, 1.0, 0.0))
                b.strafe_t = random.uniform(0.5, 1.4)
            mvx, mvy = -math.sin(want) * b.strafe * 0.55, math.cos(want) * b.strafe * 0.55
            if dist > 9:
                mvx += math.cos(want) * 0.6
                mvy += math.sin(want) * 0.6
            if b.react <= 0 and abs(angle_diff(want, b.angle)) < 0.25:
                self._bot_fire(b, tgt, dist, w)
        else:
            if b.blind > 0:
                b.angle += dt * 3.0
            # patrol to the goal waypoint; head for a recon-spotted player when the enemy has a drone up
            goal = self.waypoints[b.goal]
            if math.hypot(goal[0] + 0.5 - b.x, goal[1] + 0.5 - b.y) < 1.2:
                b.goal = self._next_goal(b)
            d = flow_dir(self.m, self._field(b.goal), b.x, b.y, self.walk)
            if d is not None:
                mvx, mvy = d
                b.angle += max(min(angle_diff(math.atan2(d[1], d[0]), b.angle), 4 * dt), -4 * dt)
        n = math.hypot(mvx, mvy)
        b.moving = n > 0.05
        if n > 1:
            mvx, mvy = mvx / n, mvy / n
        # keep apart
        for o in self.bots:
            if o is b or o.dead:
                continue
            ex, ey = b.x - o.x, b.y - o.y
            e2 = ex * ex + ey * ey
            if 1e-6 < e2 < 0.36:
                mvx += ex * 1.5
                mvy += ey * 1.5
        nx, ny = slide(self.m, b.x, b.y, mvx * speed * dt, mvy * speed * dt, 0.22)
        moved = math.hypot(nx - b.x, ny - b.y)
        b.x, b.y = nx, ny
        b.phase += moved * 3.2
        if b.moving and moved < speed * dt * 0.2 and tgt is None:
            b.stuck += dt
            if b.stuck > 1.5:
                b.goal = random.randrange(len(self.waypoints))
                b.stuck = 0.0
        else:
            b.stuck = 0.0

    def _next_goal(self, b: Bot) -> int:
        p = self.player
        if b.team == "enemy" and "enemy_recon" in self.active and "phantom" not in self.perks and not p.dead:
            return min(range(len(self.waypoints)), key=lambda i: math.hypot(self.waypoints[i][0] - p.x,
                                                                           self.waypoints[i][1] - p.y))
        # mostly towards the other side of the map
        mid = self.m.w / 2
        side = [i for i, (x, _y) in enumerate(self.waypoints) if (x > mid) == (b.team == "ally")]
        pool = side if side and random.random() < 0.6 else range(len(self.waypoints))
        return random.choice(list(pool))

    def _scan(self, b: Bot) -> None:
        best, bd = None, 1e9
        cold = 0.3 if "coldfront" in self.perks else 0.0
        for o in self._foes(b):
            dx, dy = o.x - b.x, o.y - b.y
            d = math.hypot(dx, dy)
            if d > 24:
                continue
            in_cone = abs(angle_diff(math.atan2(dy, dx), b.angle)) < 1.05
            heard = d < (2.5 if o is self.me and "hush" in self.perks else 5.0)
            heard = heard or (o.fired_t < 0.5 and d < 14)
            if not (in_cone or heard):
                continue
            if not line_of_sight(self.m, b.x, b.y, o.x, o.y):
                continue
            if d < bd:
                best, bd = o, d
        if best is not None:
            if b.target is not best:
                base = random.uniform(0.3, 0.55)
                if best is self.me:
                    base += cold
                if b.team == "enemy" and "counter" in self.active:
                    base *= 1.5
                b.react = base
            b.target = best
        elif b.target is not None:
            t = b.target
            if (t is self.me and self.player.dead) or (isinstance(t, Bot) and t.dead) or \
                    not line_of_sight(self.m, b.x, b.y, t.x, t.y):
                if isinstance(t, Bot) and not t.dead:
                    b.goal = min(range(len(self.waypoints)), key=lambda i: math.hypot(self.waypoints[i][0] - t.x,
                                                                                     self.waypoints[i][1] - t.y))
                b.target = None

    def _bot_fire(self, b: Bot, tgt: Bot, dist: float, w) -> None:  # noqa: ANN001
        if b.reload_t > 0 or b.fire_cd > 0:
            return
        if b.mag <= 0:
            b.reload_t = w.reload
            return
        b.mag -= 1
        b.fire_cd = 60.0 / w.rpm * (1.0 if w.mode == "auto" else 1.6) * random.uniform(1.0, 1.4)
        if w.mode == "auto" and random.random() < 0.12:
            b.fire_cd += random.uniform(0.2, 0.5)  # bursts with pauses
        b.flash_t = 0.06
        b.fired_t = 0.0
        pd = math.hypot(b.x - self.player.x, b.y - self.player.y)
        if pd < 22:
            self.sfx(w.sound, max(0.12, 0.55 - pd / 40))
        acc = 0.62 - dist * 0.018
        if tgt is self.me:
            p = self.player
            acc -= 0.12 * min(p.moving, 1.0) + (0.08 if p.crouched else 0.0) + (0.1 if p.sprinting else 0.0)
            if self.active.get("enemy_recon") and "phantom" not in self.perks:
                acc += 0.05
            acc *= 0.72  # the player gets a fair chance
        if b.team == "ally" and "recon" in self.active:
            acc += 0.08
        hits = 0
        for _ in range(w.pellets):
            if random.random() < max(acc, 0.06):
                hits += 1
        if not hits:
            return
        dmg = falloff(w, dist) * hits
        head = random.random() < 0.08
        if head:
            dmg *= w.head
        if tgt is self.me:
            self._hurt_player(dmg, b, w.name, head)
        else:
            self._hurt_bot(tgt, dmg, b, w.name, head)

    # --- scorestreaks --------------------------------------------------------------------------------------------

    def use_streak(self, slot: int) -> bool:
        if self.over or self.player.dead or not 0 <= slot < len(self.streak_ids):
            return False
        sid = self.streak_ids[slot]
        if sid not in self.earned:
            self.sfx("deny")
            return False
        self.earned.remove(sid)
        self.call_in(sid)
        return True

    def call_in(self, sid: str) -> None:
        p = self.player
        self.sfx("call_in")
        self.message = (f"{STREAKS[sid].name} inbound", 2.5)
        if sid == "recon":
            self.active["recon"] = 30.0
            self.sfx("uav")
        elif sid == "counter":
            self.active["counter"] = 30.0
            self.active.pop("enemy_recon", None)
        elif sid in ("hunter", "hellstorm"):
            foes = [b for b in self.bots if b.team == "enemy" and not b.dead]
            if not foes:
                self.earned.append(sid)
                return
            if sid == "hunter":
                tgt = min(foes, key=lambda b: math.hypot(b.x - p.x, b.y - p.y))
                self.missiles.append([tgt, 1.6, sid])
            else:
                tgt = max(foes, key=lambda b: sum(1 for o in foes if math.hypot(o.x - b.x, o.y - b.y) < 4))
                self.missiles.append([tgt, 2.4, sid])
            self.sfx("missile")
        elif sid == "supply":
            ca, sa = math.cos(p.angle), math.sin(p.angle)
            x, y = p.x + ca * 1.6, p.y + sa * 1.6
            if not self.m.walkable(int(x), int(y)):
                x, y = p.x, p.y
            self.crates.append([x, y, 6.0, 0.0])
        elif sid == "sentry":
            ca, sa = math.cos(p.angle), math.sin(p.angle)
            x, y = p.x + ca * 1.0, p.y + sa * 1.0
            if not self.m.walkable(int(x), int(y)):
                x, y = p.x, p.y
            self.sentries.append([x, y, 45.0, 0.0])
        elif sid == "swarm":
            self.active["swarm"] = 25.0
            self._swarm_t = 2.0

    def _update_streaks(self, dt: float) -> None:
        for mis in list(self.missiles):
            mis[1] -= dt
            if mis[1] <= 0:
                self.missiles.remove(mis)
                tgt, _, kind = mis
                x, y = tgt.x, tgt.y
                radius = 2.5 if kind == "hunter" else 4.0
                self._explode(x, y, radius, 250.0, self.me, STREAKS[kind].name)
        for crate in self.crates:
            if crate[2] > 0:
                crate[2] = max(crate[2] - dt * 4.0, 0.0)
                if crate[2] == 0:
                    self.sfx("box_land", 0.7)
        for s in list(self.sentries):
            s[2] -= dt
            s[3] -= dt
            if s[2] <= 0:
                self.sentries.remove(s)
                continue
            if s[3] <= 0:
                foes = [b for b in self.bots if b.team == "enemy" and not b.dead and
                        math.hypot(b.x - s[0], b.y - s[1]) < 12 and line_of_sight(self.m, s[0], s[1], b.x, b.y)]
                if foes:
                    tgt = min(foes, key=lambda b: math.hypot(b.x - s[0], b.y - s[1]))
                    s[3] = 0.12
                    self.effects.append(Effect("tracer", s[0], s[1], 0.5, 0.0, 0.05, tgt.x, tgt.y, 0.6))
                    if random.random() < 0.5:
                        self._hurt_bot(tgt, 20.0, self.me, "Sentry Turret")
                    if math.hypot(s[0] - self.player.x, s[1] - self.player.y) < 18:
                        self.sfx("smg", 0.35)
        if "swarm" in self.active:
            self._swarm_t = getattr(self, "_swarm_t", 2.0) - dt
            if self._swarm_t <= 0:
                self._swarm_t = 2.6
                foes = [b for b in self.bots if b.team == "enemy" and not b.dead]
                if foes:
                    tgt = random.choice(foes)
                    self._explode(tgt.x, tgt.y, 1.5, 300.0, self.me, "Drone Swarm")

    def _use_crate(self) -> None:
        p = self.player
        for crate in list(self.crates):
            if crate[2] == 0 and math.hypot(crate[0] - p.x, crate[1] - p.y) < 1.2:
                self.crates.remove(crate)
                pool = [s for s in STREAKS if s != "supply"]
                if random.random() < 0.3:
                    for g in p.guns:
                        g.refill()
                    self.message = ("Ammo refilled", 2.0)
                    self.sfx("weapon_take")
                else:
                    sid = random.choice(pool)
                    self.message = (f"Supply Drop: {STREAKS[sid].name}", 2.5)
                    self.sfx("streak_ready")
                    self.call_in(sid)
                return

    # --- the end -------------------------------------------------------------------------------------------------

    def _end(self, won: bool | None) -> None:
        if self.over:
            return
        self.over = True
        self.over_t = 0.0
        self.won = won
        self.player.reload_t = 0
        bonus = XP["win"] if won else XP["loss"]
        self.xp_pending.append(bonus)
        self.match_bonus = bonus
        self.sfx("match_win" if won else "match_lose")

    def forfeit(self) -> None:
        self._end(False)

    def record(self, st: dict) -> None:
        if getattr(self, "_recorded", False):
            return
        self._recorded = True
        st["kills"] += self.p_kills
        st["deaths"] += self.p_deaths
        st["shots"] += self.player.shots
        st["hits"] += self.player.hits
        st["headshots"] += self.p_headshots
        st["matches"] += 1
        st["score"] += self.p_score
        st["best_streak"] = max(st["best_streak"], self.best_streak)
        if self.won:
            st["wins"] += 1
        else:
            st["losses"] += 1

    # --- drawing data --------------------------------------------------------------------------------------------

    def sprites_list(self) -> list[Sprite]:
        S = self.sprites
        p = self.player
        out: list[Sprite] = []
        for b in self.bots:
            if b.dead:
                if b.dead_t < 2.6:
                    out.append(Sprite(b.x, b.y, -0.02, 0.95, 0.32, S.soldier(b.team, "dead")))
                continue
            to_me = math.atan2(p.y - b.y, p.x - b.x)
            back = abs(angle_diff(to_me, b.angle)) > 1.9
            if b.flash_t > 0 and not back:
                pose = "fire"
            elif b.target is not None and not back:
                pose = "aim"
            elif b.moving:
                pose = "run0" if math.sin(b.phase) > 0 else "run1"
            else:
                pose = "aim"
            tint = (255, 255, 255, 0.5) if b.hit_flash > 0 else None
            out.append(Sprite(b.x, b.y, 0.0, 0.6, 1.04, S.soldier(b.team, pose, back), tint=tint))
        for kind, x, y in self.m.objects:
            if kind == "tree":
                out.append(Sprite(x + 0.5, y + 0.5, 0.0, 1.6, 2.6, S.tree(False)))
            elif kind == "barrel":
                out.append(Sprite(x + 0.5, y + 0.5, 0.0, 0.36, 0.62, S.barrel(False)))
        for g in self.grenades:
            out.append(Sprite(g.x, g.y, g.z, 0.1, 0.11, S.grenade()))
        for c in self.crates:
            out.append(Sprite(c[0], c[1], c[2], 0.6, 0.6, S.crate()))
        for s in self.sentries:
            out.append(Sprite(s[0], s[1], 0.0, 0.5, 0.6, S.sentry()))
        for mis in self.missiles:
            tgt = mis[0]
            out.append(Sprite(tgt.x, tgt.y, 0.6 + mis[1] * 4.0, 0.3, 0.15, S.drone(False), bright=True))
        if "swarm" in self.active:
            for k in range(3):
                a = self.time * 0.5 + k * 2.1
                out.append(Sprite(self.m.w / 2 + math.cos(a) * 8, self.m.h / 2 + math.sin(a) * 6, 3.5, 0.5, 0.25,
                                  S.drone(False), bright=True))
        return out

    def preload_images(self) -> list:
        """Every billboard this mode can show (uploaded behind the loading screen)."""
        S = self.sprites
        out = [S.soldier(team, pose, back) for team in ("ally", "enemy") for pose in ("run0", "run1", "aim", "fire")
               for back in (False, True)]
        out += [S.soldier(team, "dead") for team in ("ally", "enemy")]
        out += [S.crate(), S.sentry(), S.drone(False), S.grenade(), S.tree(False), S.barrel(False)]
        return out

    def lights(self) -> list[tuple]:
        out = []
        for b in self.bots:
            if not b.dead and b.flash_t > 0:
                out.append((b.x, b.y, 0.75, (255, 200, 120), 0.8, 0.9))
        for e in self.effects:
            if e.kind == "boom":
                out.append((e.x, e.y, e.z, e.color, e.size * 1.3, 1 - e.t / e.life))
        for c in self.crates:
            out.append((c[0], c[1], c[2] + 0.6, (255, 80, 50), 0.3, 0.8))
        return out

    def light_beam(self) -> None:
        return None

    def enemy_dots(self) -> list[tuple[float, float, bool]]:
        """(x, y, from the recon drone) of the enemies the minimap shows."""
        out = []
        for b in self.bots:
            if b.team != "enemy" or b.dead:
                continue
            if "recon" in self.active:
                out.append((b.x, b.y, True))
            elif b.fired_t < 1.6:
                out.append((b.x, b.y, False))
            elif "sharpears" in self.perks and math.hypot(b.x - self.player.x, b.y - self.player.y) < 8:
                out.append((b.x, b.y, False))
        return out

    def team_rows(self) -> dict:
        rows = {"ally": [], "enemy": []}
        for b in self.bots:
            rows[b.team].append((b.name, b.score, b.kills, b.deaths, False))
        rows["ally"].append(("You", self.p_score, self.p_kills, self.p_deaths, True))
        for k in rows:
            rows[k].sort(key=lambda r: -r[1])
        return rows
