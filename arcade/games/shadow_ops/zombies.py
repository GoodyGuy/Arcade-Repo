"""ZOMBIES on NIGHT SHIFT: round-based survival in a tank works after dark. The dead come over the yard, tear the
planks off the windows and climb in, or rise out of the floor of the rooms that are open. The player earns points for
hits and kills, buys doors, weapons off the wall racks, a random weapon from the Lucky Crate, tonics (at most four,
lost on going down) and the Refit Press upgrade, throws the breaker, rebuilds barriers and grabs timed power-ups.

Numbers (the genre's standard): 10 points a hit, 50 a body kill, 60 a neck kill, 100 a headshot, 130 a knife kill,
10 a plank rebuilt (at most 500 a round); zombie health 150 on round 1, +100 a round to round 9, then x1.1 a round."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .data import (BOX_COST, BOX_POOL, FIRE_SALE_COST, KNIFE_DAMAGE, PAP_COST, PAP_DAMAGE, PERK_LIMIT, POWERUP_LIFE,
                   POWERUP_NAMES, POWERUP_TIME, POWERUPS, REPAP_COST, STARTER, ZPERKS, spawn_delay, weapon,
                   zombie_count, zombie_health, zombie_speeds)
from .gfx import Sprite
from .maps import MapData, start_zone
from .sprites import EYES, PERK_SYMBOL, SpriteSet
from .world import (Controls, Effect, Grenade, Gun, Player, Target, angle_diff, falloff, flow_dir, flow_field,
                    line_of_sight, shoot_ray, slide, throw)

MAX_ALIVE = 24
ZOMBIE_HIT = 50.0  # two hits down a player of 100 health, five one with Bulwark Brew
EYE_COLOR = (150, 255, 170)
SPEEDS = {"walk": 0.95, "run": 1.75, "sprint": 2.45}


@dataclass
class Zombie:
    x: float
    y: float
    hp: float
    speed: str
    look: int
    state: str = "rise"  # rise / outside / window / climb / chase / dead
    t: float = 0.0
    window: int = -1
    phase: float = 0.0
    flash: float = 0.0
    attack_t: float = 0.0  # >0: swiping
    attack_cd: float = 0.0
    tear_t: float = 0.0
    unseen: float = 0.0
    z: float = 0.0
    vmul: float = 1.0
    from_xy: tuple = (0.0, 0.0)
    to_xy: tuple = (0.0, 0.0)


@dataclass
class PowerUp:
    kind: str
    x: float
    y: float
    t: float = 0.0


@dataclass
class Popup:
    text: str
    t: float = 0.0
    color: tuple = (255, 214, 80)
    dx: float = 0.0


@dataclass
class ZStats:
    kills: int = 0
    headshots: int = 0
    downs: int = 0
    doors: int = 0
    revives: int = 0
    points_total: int = 0
    perks: int = 0
    box_uses: int = 0


@dataclass
class Box:
    spots: list
    spot: int = 0
    state: str = "idle"  # idle / spin / offer / leave / away
    t: float = 0.0
    wid: str = ""
    uses: int = 0
    shown: str = "saber"
    next_spot: int = 0


@dataclass
class Machine:
    kind: str  # perk:<id> / pap / power / tree / barrel
    x: float
    y: float
    state: str = "idle"
    t: float = 0.0
    gun: dict = field(default_factory=dict)


class ZombiesMode:
    """One game of zombies (``sfx(name, volume)`` plays sounds; the game draws it)."""

    kind = "zombies"

    def __init__(self, m: MapData, sprites: SpriteSet, sfx=None, resume: dict | None = None) -> None:  # noqa: ANN001
        self.m = m
        self.sprites = sprites
        self.sfx = sfx or (lambda name, volume=1.0: None)
        sx, sy = m.spawns["player"][0]
        self.player = Player(sx, sy, -math.pi / 2)
        self.player.guns = [Gun(STARTER)]
        self.player.grenades = 2
        self.player.stamina_max = self.player.stamina = 5.0
        self.points = 500
        self.round = 0
        self.zombies: list[Zombie] = []
        self.to_spawn = 0
        self.spawn_t = 0.0
        self.break_t = 2.5  # s before the next round starts
        self.power = False
        self.perks: list[str] = []
        self.rally_uses = 0
        self.down_t = 0.0  # >0: going down / reviving
        self.powerups: list[PowerUp] = []
        self.active: dict[str, float] = {}
        self.drops_this_round = 0
        self.kills_since_drop = 0
        self.drop_at = random.randint(16, 28)  # kills before a drop is due
        self.deck: list[str] = []
        self.popups: list[Popup] = []
        self.banner = ("", 0.0)
        self.effects: list[Effect] = []
        self.grenades: list[Grenade] = []
        self.stats = ZStats()
        self.over = False
        self.over_t = 0.0
        self.time = 0.0
        self.groan_t = 2.0
        self.repair_t = 0.0
        self.repair_points_round = 0
        self.round_flash = 0.0
        self.machines: list[Machine] = []
        spots = []
        for kind, x, y in m.objects:
            if kind == "box":
                spots.append((x + 0.5, y + 0.5))
            else:
                self.machines.append(Machine(kind, x + 0.5, y + 0.5))
        self.box = Box(spots)
        self._zone_open = {start_zone(m)}
        self._field = None
        self._field_cell = (-1, -1)
        self._field_t = 0.0
        self._window_fields: dict[int, np.ndarray] = {}
        self._outside = np.isin(m.zone, list(m.outside_zones)) if m.outside_zones else np.zeros_like(m.block)
        self.dirty = True  # the renderer must rebuild the map (a door opened)
        self.panes_dirty = True  # ... or only the windows (planks)
        self.prompt = ""
        self.hit_marker = 0.0
        self.hit_marker_kill = False
        self.in_break = False
        self.drink = None
        self._use_was = False
        self._passable = ~m.block
        if resume:
            self._resume(resume)
        self._apply_perks()

    # --- saving -------------------------------------------------------------------------------------------------

    def snapshot(self) -> dict:
        """The game in progress between rounds (zombies alive are not kept: the round starts over)."""
        p = self.player
        return {"round": max(self.round, 1), "points": int(self.points), "guns": [g.to_json() for g in p.guns],
                "cur": p.cur, "perks": list(self.perks), "rally_uses": self.rally_uses,
                "doors": [i for i, d in enumerate(self.m.doors) if d.opened], "power": self.power,
                "box_spot": self.box.spot, "x": p.x, "y": p.y, "angle": p.angle, "grenades": p.grenades,
                "stats": self.stats.__dict__.copy(), "map": self.m.key}

    def _resume(self, d: dict) -> None:
        try:
            p = self.player
            guns = [Gun.from_json(g) for g in d.get("guns", []) if isinstance(g, dict)][:2]
            p.guns = guns or [Gun(STARTER)]
            p.cur = min(max(int(d.get("cur", 0)), 0), len(p.guns) - 1)
            self.points = max(int(d.get("points", 500)), 0)
            self.perks = list(dict.fromkeys(k for k in d.get("perks", []) if k in ZPERKS))[:PERK_LIMIT]
            self.rally_uses = min(max(int(d.get("rally_uses", 0)), 0), 3)
            if d.get("map") == self.m.key:  # a game saved on another map keeps its round, points, weapons and tonics
                for i in d.get("doors", []):
                    if 0 <= int(i) < len(self.m.doors):
                        self._open_door(self.m.doors[int(i)], free=True)
                if d.get("power"):
                    self.power = True
                self.box.spot = min(max(int(d.get("box_spot", 0)), 0), max(len(self.box.spots) - 1, 0))
                x, y = float(d.get("x", p.x)), float(d.get("y", p.y))
                if self.m.walkable(int(x), int(y)):
                    p.x, p.y = x, y
                p.angle = float(d.get("angle", p.angle))
            p.grenades = min(max(int(d.get("grenades", 2)), 0), 4)
            for k, v in (d.get("stats") or {}).items():
                if hasattr(self.stats, k):
                    setattr(self.stats, k, int(v))
            self.round = max(int(d.get("round", 1)), 1) - 1
            if self.round > 0:
                self.in_break = True  # the saved round starts again after a short lead-in
                self.break_t = 3.0
                self.round_flash = 3.0
        except (TypeError, ValueError, AttributeError):
            pass

    # --- helpers ------------------------------------------------------------------------------------------------

    def _apply_perks(self) -> None:
        p = self.player
        perks = self.perks
        p.max_health = 250.0 if "jugg" in perks else 100.0
        p.health = min(p.health, p.max_health) if p.health > 0 else p.max_health
        p.reload_mult = 0.5 if "swift" in perks else 1.0
        p.rate_mult = 1.33 if "double" in perks else 1.0
        p.damage_mult = 2.0 if "double" in perks else 1.0
        p.speed_mult = 1.07 if "stamina" in perks else 1.0
        p.stamina_max = 10.0 if "stamina" in perks else 5.0
        p.stamina = min(p.stamina, p.stamina_max)
        p.hip_mult = 0.6 if "deadeye" in perks else 1.0
        p.recoil_mult = 0.6 if "deadeye" in perks else 1.0

    def add_points(self, n: int, popup: bool = True) -> None:
        if n > 0 and "double_points" in self.active:
            n *= 2
        self.points += n
        if n > 0:
            self.stats.points_total += n
        if popup and n:
            self.popups.append(Popup(f"+{n}" if n > 0 else str(n), 0.0,
                                     (255, 214, 80) if n > 0 else (255, 90, 70), random.uniform(-8, 8)))

    def _zone_active(self, zone: int) -> bool:
        return zone in self._zone_open

    def _open_door(self, door, free: bool = False) -> None:  # noqa: ANN001
        self.m.open_door(door)
        for z in (door.zone_a, door.zone_b):
            if z >= 0 and z not in self.m.outside_zones:
                self._zone_open.add(z)
        self.dirty = True
        self._field_cell = (-1, -1)
        if not free:
            self.stats.doors += 1

    def active_windows(self) -> list[int]:
        out = []
        for i, w in enumerate(self.m.windows):
            ix, iy = w.inside
            if self._zone_active(int(self.m.zone[iy, ix])):
                out.append(i)
        return out

    def _set_planks(self, i: int, n: int) -> None:
        w = self.m.windows[i]
        w.planks = min(max(n, 0), 6)
        x, y = w.cell
        self.m.glass[y, x] = w.tex0 + w.planks
        self.panes_dirty = True

    # --- rounds and spawning ------------------------------------------------------------------------------------

    def _start_round(self) -> None:
        self.round += 1
        self.to_spawn = zombie_count(self.round)
        self.spawn_t = 1.0
        self.drops_this_round = 0
        self.repair_points_round = 0
        self.round_flash = 2.5
        self.sfx("round_start")

    def _spawn(self) -> None:
        r = self.round
        walk, run, _ = zombie_speeds(r)
        roll = random.random()
        speed = "walk" if roll < walk else ("run" if roll < walk + run else "sprint")
        hp = zombie_health(r)
        look = random.randrange(4)
        p = self.player
        risers = []
        for x, y in self.m.spawns["zombie"]:
            zone = int(self.m.zone[int(y), int(x)])
            if zone in self.m.outside_zones or not self._zone_active(zone):
                continue
            d = math.hypot(x - p.x, y - p.y)
            if 5.0 < d < 18.0:
                risers.append((x, y))
        windows = self.active_windows()
        if risers and (not windows or random.random() < 0.4):
            x, y = random.choice(risers)
            x += random.uniform(-0.3, 0.3)
            y += random.uniform(-0.3, 0.3)
            z = Zombie(x, y, hp, speed, look, "rise", z=-1.0)
            self.sfx("rise", 0.6)
        elif windows:
            # the window nearest the player most of the time
            windows.sort(key=lambda i: math.hypot(self.m.windows[i].inside[0] + 0.5 - p.x,
                                                  self.m.windows[i].inside[1] + 0.5 - p.y))
            wi = windows[0] if random.random() < 0.45 else random.choice(windows)
            win = self.m.windows[wi]
            ox, oy = win.outside
            outside = [(x, y) for x, y in self.m.spawns["zombie"]
                       if int(self.m.zone[int(y), int(x)]) in self.m.outside_zones]
            outside.sort(key=lambda s: math.hypot(s[0] - ox - 0.5, s[1] - oy - 0.5))
            x, y = random.choice(outside[:2]) if outside else (ox + 0.5, oy + 0.5)
            z = Zombie(x, y, hp, speed, look, "rise", window=wi, z=-1.0)
        else:
            return
        z.vmul = random.uniform(0.9, 1.1)
        z.phase = random.uniform(0, 6)
        self.zombies.append(z)
        self.to_spawn -= 1

    def _window_field(self, i: int) -> np.ndarray:
        f = self._window_fields.get(i)
        if f is None:
            w = self.m.windows[i]
            f = self._window_fields[i] = flow_field(self.m, [w.outside], ~self.m.block)
        return f

    def _player_field(self) -> np.ndarray:
        p = self.player
        cell = (int(p.x), int(p.y))
        if self._field is None or (cell != self._field_cell and self.time - self._field_t > 0.25):
            passable = ~self.m.block & ~self._outside
            self._field = flow_field(self.m, [cell], passable)
            self._field_cell = cell
            self._field_t = self.time
            self._passable = passable
        return self._field

    # --- the frame ----------------------------------------------------------------------------------------------

    def update(self, dt: float, c: Controls, use_held: bool) -> None:
        self.time += dt
        p = self.player
        if self.over:
            self.over_t += dt
            self._update_effects(dt)
            return
        if self.down_t > 0:
            self._update_down(dt)
            return
        p.move(self.m, c, dt)
        shots = p.update_weapon(dt, c.fire)
        if shots < 0:
            self.sfx("dry")
        for _ in range(max(shots, 0)):
            self._fire()
        if p.reload_t > 0 and abs(p.reload_t - p.reload_total) < dt * 1.01:
            self.sfx("reload")
        self._update_round(dt)
        self._update_zombies(dt)
        self._update_grenades(dt)
        self._update_powerups(dt)
        self._update_machines(dt)
        self._update_use(dt, use_held)
        self._update_effects(dt)
        self.hit_marker = max(self.hit_marker - dt, 0.0)
        for k in list(self.active):
            self.active[k] -= dt
            if self.active[k] <= 0:
                del self.active[k]
                self.sfx("powerup_end")
        for pop in self.popups:
            pop.t += dt
        self.popups = [q for q in self.popups if q.t < 1.4]
        if self.banner[1] > 0:
            self.banner = (self.banner[0], self.banner[1] - dt)
        self.round_flash = max(self.round_flash - dt, 0.0)
        if p.dead:
            self._go_down()

    def _update_round(self, dt: float) -> None:
        if self.round == 0:
            self.break_t -= dt
            if self.break_t <= 0:
                self._start_round()
            return
        if self.between_rounds():
            if not self.in_break:
                self.in_break = True
                self.break_t = 9.0
                self.round_flash = 9.0
                self.sfx("round_end")
            self.break_t -= dt
            if self.break_t <= 0:
                self.in_break = False
                self._start_round()
            return
        alive = sum(1 for z in self.zombies if z.state != "dead")
        self.spawn_t -= dt
        if self.to_spawn > 0 and self.spawn_t <= 0 and alive < MAX_ALIVE:
            self._spawn()
            self.spawn_t = spawn_delay(self.round) * random.uniform(0.7, 1.3)

    def between_rounds(self) -> bool:
        return self.round == 0 or (self.to_spawn <= 0 and not any(z.state != "dead" for z in self.zombies))

    def _update_zombies(self, dt: float) -> None:
        p = self.player
        field_ = self._player_field()
        passable = self._passable
        alive = [z for z in self.zombies if z.state != "dead"]
        for z in self.zombies:
            z.flash = max(z.flash - dt * 4, 0.0)
            if z.state == "dead":
                z.t += dt
                if z.t > 2.5:
                    z.z = -(z.t - 2.5) * 0.25
                continue
            sp = SPEEDS[z.speed] * z.vmul
            z.phase += dt * sp * 3.0
            z.attack_cd = max(z.attack_cd - dt, 0.0)
            if z.state == "rise":
                z.z = min(z.z + dt * 0.9, 0.0)
                if z.z >= 0:
                    z.state = "outside" if z.window >= 0 else "chase"
                continue
            if z.state == "outside":
                win = self.m.windows[z.window]
                ox, oy = win.outside[0] + 0.5, win.outside[1] + 0.5
                if math.hypot(ox - z.x, oy - z.y) < 0.35:
                    z.state = "window"
                    z.tear_t = 1.0
                    continue
                d = flow_dir(self.m, self._window_field(z.window), z.x, z.y, ~self.m.block)
                if d is None:
                    d = ((ox - z.x), (oy - z.y))
                    n = math.hypot(*d) or 1
                    d = (d[0] / n, d[1] / n)
                z.x, z.y = slide(self.m, z.x, z.y, d[0] * sp * dt, d[1] * sp * dt, 0.2)
                continue
            if z.state == "window":
                win = self.m.windows[z.window]
                if win.planks > 0:
                    z.tear_t -= dt
                    z.attack_t = max(z.attack_t - dt, 0.0)
                    if z.tear_t <= 0:
                        self._set_planks(z.window, win.planks - 1)
                        z.tear_t = 1.1 if z.speed == "walk" else 0.8
                        z.attack_t = 0.4
                        self._sound_at("plank_tear", z.x, z.y)
                else:
                    z.state = "climb"
                    z.t = 0.0
                    z.from_xy = (z.x, z.y)
                    z.to_xy = (win.inside[0] + 0.5, win.inside[1] + 0.5)
                continue
            if z.state == "climb":
                z.t += dt / 0.9
                k = min(z.t, 1.0)
                z.x = z.from_xy[0] + (z.to_xy[0] - z.from_xy[0]) * k
                z.y = z.from_xy[1] + (z.to_xy[1] - z.from_xy[1]) * k
                z.z = math.sin(k * math.pi) * 0.25
                if k >= 1:
                    z.state = "chase"
                    z.z = 0.0
                continue
            # chase
            dx, dy = p.x - z.x, p.y - z.y
            dist = math.hypot(dx, dy)
            if z.attack_t > 0:
                z.attack_t -= dt
                if z.attack_t <= 0:
                    if dist < 1.0 and not p.dead:
                        p.hurt(ZOMBIE_HIT, (z.x, z.y))
                        self.sfx("player_hit")
                    z.attack_cd = 0.9
                continue
            if dist < 0.75 and z.attack_cd <= 0:
                z.attack_t = 0.42
                self._sound_at("swipe", z.x, z.y)
                continue
            if dist < 2.2 and line_of_sight(self.m, z.x, z.y, p.x, p.y, self.m.block):
                d = (dx / (dist or 1), dy / (dist or 1))
            else:
                d = flow_dir(self.m, field_, z.x, z.y, passable)
                if d is None:
                    d = (dx / (dist or 1), dy / (dist or 1))
            # keep apart from the others
            sx = sy = 0.0
            for o in alive:
                if o is z:
                    continue
                ex, ey = z.x - o.x, z.y - o.y
                e2 = ex * ex + ey * ey
                if 1e-6 < e2 < 0.25:
                    k = (0.5 - math.sqrt(e2)) / 0.5
                    sx += ex * k * 2
                    sy += ey * k * 2
            mvx, mvy = d[0] + sx, d[1] + sy
            n = math.hypot(mvx, mvy) or 1.0
            step = sp * dt if dist > 0.6 else 0.0
            z.x, z.y = slide(self.m, z.x, z.y, mvx / n * step, mvy / n * step, 0.2, self.m.block)
            # stuck far away out of sight: come back in somewhere else
            z.unseen = z.unseen + dt if dist > 16 else 0.0
            if z.unseen > 20:
                z.state = "dead"
                z.t = 99
                self.to_spawn += 1
        self.zombies = [z for z in self.zombies if not (z.state == "dead" and z.t > 5.0)]
        self.groan_t -= dt
        if self.groan_t <= 0 and alive:
            z = random.choice(alive)
            self._sound_at(random.choice(("groan1", "groan2", "groan3")) if z.speed == "walk" else "sprint_groan",
                           z.x, z.y)
            self.groan_t = random.uniform(0.6, 2.2) * (1.5 if len(alive) < 4 else 1.0)

    def _sound_at(self, name: str, x: float, y: float) -> None:
        d = math.hypot(x - self.player.x, y - self.player.y)
        if d < 18:
            self.sfx(name, max(0.15, 1.0 - d / 18))

    # --- shooting -----------------------------------------------------------------------------------------------

    def targets(self) -> list[Target]:
        return [Target(z, z.x, z.y, 0.27, z.z, 1.0, 0.24) for z in self.zombies
                if z.state not in ("dead",) and z.z > -0.7]

    def _fire(self) -> None:
        p = self.player
        gun = p.gun
        w = gun.w
        self.sfx(w.sound)
        eye = p.view_eye()
        targets = self.targets()
        any_hit = False
        for yaw, slope in p.pellet_dirs():
            hit = shoot_ray(self.m, p.x, p.y, eye, yaw, slope, targets)
            self.effects.append(Effect("tracer", p.x, p.y, eye - 0.08, 0.0, 0.06, hit.x, hit.y, hit.z))
            if w.splash > 0:
                self._explode(hit.x - math.cos(yaw) * 0.1, hit.y - math.sin(yaw) * 0.1, max(hit.z, 0.1),
                              w.splash, w.dmg * (PAP_DAMAGE if gun.upgraded else 1.0) * p.damage_mult,
                              raygun=w.id == "zapper", upgraded=gun.upgraded)
                any_hit = any_hit or hit.target is not None
                continue
            if hit.target is None:
                self.effects.append(Effect("puff", hit.x, hit.y, hit.z, 0.0, 0.25))
                continue
            z = hit.target.obj
            dmg = falloff(w, hit.dist, "long" in gun.atts) * p.damage_mult
            if gun.upgraded:
                dmg *= PAP_DAMAGE
            if hit.head:
                dmg *= max(w.head, 1.4) * (2.0 if "deadeye" in self.perks else 1.5)
            t = hit.target
            neck = not hit.head and hit.z >= t.z + t.h * 0.66
            self.effects.append(Effect("blood", hit.x, hit.y, hit.z, 0.0, 0.3))
            self._damage(z, dmg, hit.head, "neck" if neck else "gun")
            any_hit = True
        if any_hit:
            p.hits += 1
        p.flash_t = 0.06 if w.id != "zapper" else 0.1

    def _damage(self, z: Zombie, dmg: float, head: bool, how: str) -> None:
        if z.state == "dead":
            return
        if "insta_kill" in self.active:
            dmg = z.hp + 1
        z.hp -= dmg
        z.flash = 1.0
        self.hit_marker = 0.18
        self.hit_marker_kill = False
        if z.hp > 0:
            self.add_points(10)
            self.sfx("hitmarker", 0.7)
            return
        neck = how == "neck"
        self.hit_marker_kill = True
        z.state = "dead"
        z.t = 0.0
        z.attack_t = 0.0
        self.stats.kills += 1
        if how == "knife":
            self.add_points(130)
        elif how == "nuke":
            pass
        elif head:
            self.add_points(100)
            self.stats.headshots += 1
            self.player.headshots += 1
            self.sfx("headshot", 0.8)
        else:
            self.add_points(60 if neck else 50)
        self._sound_at("zombie_die", z.x, z.y)
        if how != "nuke":
            self._maybe_drop(z.x, z.y)

    def _explode(self, x: float, y: float, z: float, radius: float, dmg: float, raygun: bool = False,
                 upgraded: bool = False, from_player: bool = True) -> None:
        self.effects.append(Effect("ray" if raygun else "boom", x, y, z, 0.0, 0.35 if raygun else 0.6,
                                   color=((120, 255, 120) if not upgraded else (255, 90, 90)) if raygun
                                   else (255, 170, 60), size=radius))
        self._sound_at("ray_hit" if raygun else "explosion", x, y)
        for zz in self.zombies:
            if zz.state == "dead":
                continue
            d = math.hypot(zz.x - x, zz.y - y)
            if d <= radius and line_of_sight(self.m, x, y, zz.x, zz.y):
                self._damage(zz, dmg * (1.0 - 0.5 * d / radius), False, "boom")
        p = self.player
        d = math.hypot(p.x - x, p.y - y)
        if not raygun and d < radius * 0.8 and line_of_sight(self.m, x, y, p.x, p.y):
            p.hurt(30 * (1 - d / radius), (x, y))  # your own grenades hurt a little

    def knife(self) -> None:
        p = self.player
        if p.knife_t > 0 or p.switch_t > 0 or self.over or self.down_t > 0:
            return
        p.knife_t = 0.45
        p.reload_t = 0.0
        best, bd = None, 1.4
        for z in self.zombies:
            if z.state in ("dead", "rise") or z.z < -0.3:
                continue
            d = math.hypot(z.x - p.x, z.y - p.y)
            if d < bd and abs(angle_diff(math.atan2(z.y - p.y, z.x - p.x), p.angle)) < 0.7:
                best, bd = z, d
        if best is None:
            self.sfx("knife_swing")
            return
        self.sfx("knife_hit")
        dmg = KNIFE_DAMAGE * (2 if p.gun is not None and p.gun.w.id == "judge5" and False else 1)
        self._damage(best, dmg, False, "knife")

    def throw_grenade(self) -> None:
        p = self.player
        if p.grenades <= 0 or p.throw_t > 0 or p.knife_t > 0 or self.over or self.down_t > 0:
            return
        p.grenades -= 1
        p.throw_t = 0.5
        p.reload_t = 0.0
        self.grenades.append(throw(p))
        self.sfx("throw")

    def _update_grenades(self, dt: float) -> None:
        for g in list(self.grenades):
            if g.update(self.m, dt):
                self.grenades.remove(g)
                self._explode(g.x, g.y, 0.3, 2.6, 260.0)

    # --- power-ups ----------------------------------------------------------------------------------------------

    def _maybe_drop(self, x: float, y: float) -> None:
        """A power-up now and then: a small chance on every kill, and one for sure when enough kills have passed
        since the last (at most four a round)."""
        self.kills_since_drop += 1
        due = self.kills_since_drop >= self.drop_at
        if self.drops_this_round >= 4 or not (due or random.random() < 0.025):
            return
        if not self.m.walkable(int(x), int(y)) or self._outside[int(y), int(x)]:
            return
        if not self.deck:
            self.deck = list(POWERUPS) + ["max_ammo", "double_points", "insta_kill"]
            if len(self.box.spots) < 2:
                self.deck.remove("fire_sale")
            random.shuffle(self.deck)
        self.powerups.append(PowerUp(self.deck.pop(), x, y))
        self.drops_this_round += 1
        self.kills_since_drop = 0
        self.drop_at = random.randint(16, 28) + self.round // 3
        self.sfx("powerup_spawn", 0.8)

    def _update_powerups(self, dt: float) -> None:
        p = self.player
        for pu in list(self.powerups):
            pu.t += dt
            if pu.t > POWERUP_LIFE:
                self.powerups.remove(pu)
            elif math.hypot(pu.x - p.x, pu.y - p.y) < 0.65:
                self.powerups.remove(pu)
                self.grab(pu.kind)

    def grab(self, kind: str) -> None:
        self.banner = (POWERUP_NAMES.get(kind, kind.upper()), 2.6)
        self.sfx("powerup_grab")
        self.sfx("announce_" + kind, 1.0)
        p = self.player
        if kind == "max_ammo":
            for g in p.guns:
                g.reserve = g.reserve_max()
            p.grenades = max(p.grenades, 4)
        elif kind == "insta_kill":
            self.active["insta_kill"] = POWERUP_TIME
        elif kind == "double_points":
            self.active["double_points"] = POWERUP_TIME
        elif kind == "fire_sale":
            self.active["fire_sale"] = POWERUP_TIME
            if self.box.state in ("leave", "away"):
                self.box.state, self.box.t = "idle", 0.0
        elif kind == "bonus_points":
            self.add_points(random.choice((300, 400, 500)))
        elif kind == "nuke":
            self.effects.append(Effect("nuke", p.x, p.y, 1.0, 0.0, 1.2))
            for z in self.zombies:
                if z.state != "dead":
                    self._damage(z, 1e12, False, "nuke")
            self.add_points(400)
            self.sfx("explosion")
        elif kind == "carpenter":
            for i in range(len(self.m.windows)):
                self._set_planks(i, 6)
            self.add_points(200)
            self.sfx("plank_fix")

    # --- machines, the box, doors, repairs ----------------------------------------------------------------------

    def _update_machines(self, dt: float) -> None:
        b = self.box
        if b.state == "spin":
            b.t += dt
            period = 0.06 + 0.25 * (b.t / 4.2) ** 2
            if int(b.t / period) != int((b.t - dt) / period):
                b.shown = random.choice(BOX_POOL)
            if b.t >= 4.2:
                if b.wid == "leave":
                    b.state, b.t = "leave", 0.0
                    self.sfx("bear")
                    self.banner = ("THE CRATE MOVES ON", 2.5)
                else:
                    b.state, b.t = "offer", 0.0
                    b.shown = b.wid
        elif b.state == "offer":
            b.t += dt
            if b.t > 12.0:
                b.state, b.t = "idle", 0.0
                self.sfx("box_close")
        elif b.state == "leave":
            b.t += dt
            if b.t > 3.0:
                b.state, b.t = "away", 0.0
                choices = [i for i in range(len(b.spots)) if i != b.spot]
                b.next_spot = random.choice(choices) if choices else b.spot
                self.sfx("box_move")
        elif b.state == "away":
            b.t += dt
            if b.t > 6.0:
                b.spot = b.next_spot
                b.state, b.t, b.uses = "idle", 0.0, 0
                self.sfx("box_land")
        for mc in self.machines:
            if mc.kind == "pap" and mc.state != "idle":
                mc.t += dt
                if mc.state == "work" and mc.t > 3.6:
                    mc.state, mc.t = "ready", 0.0
                    self.sfx("pap_ding")
                elif mc.state == "ready" and mc.t > 12.0:  # not taken in time: it comes back to the hand anyway
                    mc.state, mc.t = "idle", 0.0
                    g = Gun.from_json(mc.gun)
                    g.upgraded = True
                    g.refill()
                    self.player.give(g)

    def interactables(self) -> list[tuple]:
        """(distance, kind, data, prompt text) of everything the player can use, nearest first."""
        p = self.player
        out = []
        fx, fy = math.cos(p.angle), math.sin(p.angle)

        def near(x: float, y: float, reach: float) -> float | None:
            dx, dy = x - p.x, y - p.y
            d = math.hypot(dx, dy)
            if d > reach:
                return None
            if d > 0.5 and (dx * fx + dy * fy) / d < 0.2:
                return None
            return d

        for door in self.m.doors:
            if door.opened:
                continue
            ds = [near(x + 0.5, y + 0.5, 1.6) for x, y in door.cells]
            ds = [d for d in ds if d is not None]
            if ds:
                what = "Clear Debris" if door.kind == "debris" else "Open Door"
                out.append((min(ds), "door", door, f"Hold F to {what} [Cost: {door.cost}]"))
        for (cx, cy), wid, fronts in self.m.wallbuys:
            if not any(int(p.x) == x and int(p.y) == y for x, y in fronts) and \
                    math.hypot(cx + 0.5 - p.x, cy + 0.5 - p.y) > 1.3:
                continue
            d = near(cx + 0.5, cy + 0.5, 1.6)
            if d is None:
                continue
            w = weapon(wid)
            owned = next((g for g in p.guns if g.wid == wid), None)
            if owned is None:
                out.append((d, "wallbuy", wid, f"Hold F for {w.name} [Cost: {w.cost}]"))
            else:
                cost = 4500 if owned.upgraded else w.cost // 2
                out.append((d, "ammo", wid, f"Hold F for Ammo [Cost: {cost}]"))
        for mc in self.machines:
            if mc.kind.startswith("perk:"):
                d = near(mc.x, mc.y, 1.5)
                if d is None:
                    continue
                perk = ZPERKS[mc.kind[5:]]
                if perk.id in self.perks:
                    continue
                if perk.power and not self.power:
                    out.append((d, "nopower", None, "Throw the breaker first!"))
                elif perk.id == "rally" and self.rally_uses >= 3:
                    continue
                elif len(self.perks) >= PERK_LIMIT:
                    out.append((d, "perkfull", None, f"You can carry {PERK_LIMIT} tonics"))
                else:
                    out.append((d, "perk", perk.id, f"Hold F to buy {perk.name} [Cost: {perk.cost}]"))
            elif mc.kind == "pap":
                d = near(mc.x, mc.y, 1.6)
                if d is None:
                    continue
                if not self.power:
                    out.append((d, "nopower", None, "Throw the breaker first!"))
                elif mc.state == "ready":
                    out.append((d, "pap_take", mc, f"Hold F to take {weapon(mc.gun.get('wid', 'service45')).pap}"))
                elif mc.state == "idle" and p.gun is not None and not p.gun.upgraded:
                    out.append((d, "pap", mc, f"Hold F to use the Refit Press [Cost: {PAP_COST}]"))
                elif mc.state == "idle" and p.gun is not None:
                    out.append((d, "repap", mc, f"Hold F to Refit again: full ammo [Cost: {REPAP_COST}]"))
            elif mc.kind == "power" and not self.power:
                d = near(mc.x, mc.y, 1.5)
                if d is not None:
                    out.append((d, "power", mc, "Hold F to throw the Breaker"))
        b = self.box
        if b.spots and b.state != "away":
            bx, by = b.spots[b.spot]
            d = near(bx, by, 1.7)
            if d is not None:
                if b.state == "idle":
                    out.append((d, "box", None, f"Hold F to open the Lucky Crate [Cost: {self.box_cost()}]"))
                elif b.state == "offer":
                    out.append((d, "box_take", None, f"Hold F for {weapon(b.wid).name}"))
        for i, win in enumerate(self.m.windows):
            if win.planks >= 6:
                continue
            ix, iy = win.inside
            if not self._zone_active(int(self.m.zone[iy, ix])):
                continue
            d = math.hypot(ix + 0.5 - p.x, iy + 0.5 - p.y)
            if d < 1.35:
                out.append((d + 0.3, "repair", i, "Hold F to Rebuild Barrier"))
        out.sort(key=lambda o: o[0])
        return out

    def _update_use(self, dt: float, held: bool) -> None:
        items = self.interactables()
        self.prompt = items[0][3] if items else ""
        if not items:
            self.repair_t = 0.0
            return
        _, kind, data, _ = items[0]
        if kind == "repair":
            if not held:
                self.repair_t = 0.0
                return
            self.repair_t += dt
            if self.repair_t >= 0.7:
                self.repair_t = 0.0
                win = self.m.windows[data]
                self._set_planks(data, win.planks + 1)
                self.sfx("plank_fix")
                if self.repair_points_round < 500:
                    self.add_points(10)
                    self.repair_points_round += 10
            return
        if held and not getattr(self, "_use_was", False):
            self.use(kind, data)
        self._use_was = held

    def use(self, kind: str, data) -> bool:  # noqa: ANN001
        """Buy / take / switch on what is in front of the player."""
        p = self.player
        self._use_was = True
        if kind == "door":
            if self.points < data.cost:
                self.sfx("deny")
                return False
            self.add_points(-data.cost)
            self._open_door(data)
            self.sfx("door_open")
            self.sfx("purchase")
            return True
        if kind == "wallbuy":
            w = weapon(data)
            if self.points < w.cost:
                self.sfx("deny")
                return False
            self.add_points(-w.cost)
            p.give(Gun(data))
            self.sfx("purchase")
            return True
        if kind == "ammo":
            g = next((g for g in p.guns if g.wid == data), None)
            cost = 4500 if g and g.upgraded else weapon(data).cost // 2
            if g is None or self.points < cost or g.reserve >= g.reserve_max():
                self.sfx("deny")
                return False
            self.add_points(-cost)
            g.reserve = g.reserve_max()
            self.sfx("purchase")
            return True
        if kind == "perk":
            perk = ZPERKS[data]
            if self.points < perk.cost or data in self.perks or len(self.perks) >= PERK_LIMIT:
                self.sfx("deny")
                return False
            self.add_points(-perk.cost)
            self.perks.append(data)
            if data == "rally":
                self.rally_uses += 1
            self.stats.perks += 1
            self._apply_perks()
            p.health = p.max_health
            p.knife_t = 1.0  # the drinking animation
            self.drink = (data, 1.6)
            self.sfx("drink")
            self.sfx("jingle_" + data)
            return True
        if kind == "power":
            self.power = True
            self.sfx("power_on")
            self.banner = ("BREAKER ON", 2.5)
            return True
        if kind == "box":
            b = self.box
            cost = self.box_cost()
            if self.points < cost or b.state != "idle":
                self.sfx("deny")
                return False
            self.add_points(-cost)
            b.uses += 1
            self.stats.box_uses += 1
            b.state, b.t = "spin", 0.0
            owned = {g.wid for g in p.guns}
            pool = [w for w in BOX_POOL if w not in owned]
            if len(b.spots) > 1 and b.uses >= 4 and random.random() < 0.2 and "fire_sale" not in self.active:
                b.wid = "leave"
                self.add_points(cost, popup=False)
            else:
                b.wid = random.choice(pool or list(BOX_POOL))
            self.sfx("box_open")
            self.sfx("box_jingle")
            return True
        if kind == "box_take":
            b = self.box
            p.give(Gun(b.wid))
            b.state, b.t = "idle", 0.0
            self.sfx("weapon_take")
            return True
        if kind == "pap":
            if self.points < PAP_COST or p.gun is None or p.gun.upgraded:
                self.sfx("deny")
                return False
            self.add_points(-PAP_COST)
            data.gun = p.gun.to_json()
            data.state, data.t = "work", 0.0
            p.guns.pop(p.cur)
            if not p.guns:
                p.guns.append(Gun(STARTER, mag=0, reserve=0))
            p.cur = 0
            p.switch_t = 0.4
            self.sfx("pap_work")
            return True
        if kind == "repap":
            g = p.gun
            if self.points < REPAP_COST or g is None or not g.upgraded or g.reserve >= g.reserve_max():
                self.sfx("deny")
                return False
            self.add_points(-REPAP_COST)
            g.refill()
            self.sfx("pap_ding")
            return True
        if kind == "pap_take":
            g = Gun.from_json(data.gun)
            g.upgraded = True
            g.refill()
            if len(p.guns) == 1 and p.guns[0].wid == STARTER and p.guns[0].mag == 0 and p.guns[0].reserve == 0:
                p.guns = []
            p.give(g)
            data.state, data.t = "idle", 0.0
            self.sfx("weapon_take")
            return True
        if kind in ("nopower", "perkfull"):
            self.sfx("deny")
        return False

    def box_cost(self) -> int:
        return FIRE_SALE_COST if "fire_sale" in self.active else BOX_COST

    # --- going down ---------------------------------------------------------------------------------------------

    def _go_down(self) -> None:
        p = self.player
        self.stats.downs += 1
        self.sfx("down")
        if "rally" in self.perks:
            self.down_t = 4.0
            self.perks = []
            self._apply_perks()
            self.stats.revives += 1
            return
        self.over = True
        self.over_t = 0.0
        self.sfx("game_over")
        p.dead = True

    def _update_down(self, dt: float) -> None:
        p = self.player
        self.down_t -= dt
        # the zombies back off while you get up
        for z in self.zombies:
            if z.state == "chase":
                z.attack_t = 0.0
                z.attack_cd = 1.5
        self._update_effects(dt)
        if self.down_t <= 0:
            self.down_t = 0.0
            p.dead = False
            p.health = p.max_health
            p.hurt_t = 99.0
            self.sfx("revive")
            for z in self.zombies:
                if z.state == "chase" and math.hypot(z.x - p.x, z.y - p.y) < 2.5:
                    self._damage(z, 1e12, False, "nuke")

    def _update_effects(self, dt: float) -> None:
        for e in self.effects:
            e.t += dt
        self.effects = [e for e in self.effects if e.t < e.life]
        drink = getattr(self, "drink", None)
        if drink:
            self.drink = (drink[0], drink[1] - dt) if drink[1] > dt else None

    # --- drawing data -------------------------------------------------------------------------------------------

    def sprites_list(self) -> list[Sprite]:
        S = self.sprites
        out: list[Sprite] = []
        for z in self.zombies:
            if z.state == "dead":
                if z.t < 99:
                    out.append(Sprite(z.x, z.y, z.z - 0.02, 0.95, 0.32, S.zombie(z.look, "dead")))
                continue
            if z.state == "rise":
                pose = "rise"
            elif z.attack_t > 0 or (z.state == "window" and z.attack_t > 0):
                pose = "attack"
            else:
                pose = "walk0" if math.sin(z.phase) > 0 else "walk1"
            tint = (255, 255, 255, 0.55 * z.flash) if z.flash > 0 else None
            if "insta_kill" in self.active and tint is None:
                tint = None
            out.append(Sprite(z.x, z.y, z.z, 0.6, 1.0, S.zombie(z.look, pose), tint=tint, eyes=EYES,
                              eye_color=EYE_COLOR))
        for mc in self.machines:
            if mc.kind.startswith("perk:"):
                perk = ZPERKS[mc.kind[5:]]
                on = self.power or not perk.power
                out.append(Sprite(mc.x, mc.y, 0.0, 0.62, 1.1, S.perk(perk.color, PERK_SYMBOL[perk.id], on),
                                  bright=on))
            elif mc.kind == "pap":
                out.append(Sprite(mc.x, mc.y, 0.0, 0.9, 1.0, S.pap(self.power), bright=self.power))
                if mc.state in ("work", "ready") and mc.gun:
                    w = weapon(mc.gun.get("wid", "service45"))
                    bob = math.sin(self.time * 3) * 0.03
                    if mc.state == "ready" or mc.t > 2.0:
                        gx, gy = self._toward_player(mc.x, mc.y, 0.5)
                        out.append(Sprite(gx, gy, 0.66 + bob, 0.55, 0.23, S.gun(w.model, True), bright=True))
            elif mc.kind == "power":
                out.append(Sprite(mc.x, mc.y, 0.2, 0.42, 0.62, S.power(self.power)))
            elif mc.kind == "tree":
                out.append(Sprite(mc.x, mc.y, 0.0, 1.5, 2.4, S.tree(True)))
            elif mc.kind == "barrel":
                out.append(Sprite(mc.x, mc.y, 0.0, 0.35, 0.62, S.barrel(True), bright=True))
        b = self.box
        if b.spots and b.state != "away":
            bx, by = b.spots[b.spot]
            out.append(Sprite(bx, by, 0.0, 0.95, 0.48, S.box(b.state != "idle")))
            if b.state in ("spin", "offer"):
                rise = min(b.t / 1.5, 1.0) * 0.35 if b.state == "spin" else 0.35 - min(b.t / 12, 1) * 0.3
                gx, gy = self._toward_player(bx, by, 0.12)
                out.append(Sprite(gx, gy, 0.45 + rise, 0.6, 0.25, S.gun(weapon(b.shown).model), bright=True))
            elif b.state == "leave":
                out.append(Sprite(bx, by, 0.48 + b.t * 0.3, 0.34, 0.4, S.beacon(int(b.t * 6) % 2 == 0),
                                  bright=True))
        for pu in self.powerups:
            if pu.t > 21 and int(pu.t * 8) % 2:
                continue
            out.append(Sprite(pu.x, pu.y, 0.35 + math.sin(self.time * 3 + pu.x) * 0.05, 0.42, 0.42,
                              S.powerup(pu.kind), bright=True))
        for g in self.grenades:
            out.append(Sprite(g.x, g.y, g.z, 0.1, 0.11, S.grenade()))
        return out

    def _toward_player(self, x: float, y: float, d: float) -> tuple[float, float]:
        """A point ``d`` from (x, y) towards the player (a weapon shown in front of its machine)."""
        p = self.player
        dx, dy = p.x - x, p.y - y
        n = math.hypot(dx, dy) or 1.0
        return x + dx / n * d, y + dy / n * d

    def preload_images(self) -> list:
        """Every billboard this mode can show (uploaded behind the loading screen)."""
        S = self.sprites
        out = [S.zombie(look, pose) for look in range(4) for pose in ("walk0", "walk1", "attack", "rise", "dead")]
        out += [S.powerup(k) for k in POWERUPS] + [S.beacon(True), S.beacon(False), S.box(True), S.box(False),
                                                   S.grenade()]
        out += [S.gun(weapon(w).model, camo) for w in set(BOX_POOL) | {g for _c, g, _f in self.m.wallbuys} | {STARTER}
                for camo in (False, True)]
        out += [s.img for s in self.sprites_list()]
        return out

    def lights(self) -> list[tuple]:
        """(x, y, z, (r, g, b), world radius, strength) glows drawn over the view."""
        out = []
        for mc in self.machines:
            if mc.kind.startswith("perk:"):
                perk = ZPERKS[mc.kind[5:]]
                if self.power or not perk.power:
                    out.append((mc.x, mc.y, 0.75, perk.color, 1.0, 0.55))
            elif mc.kind == "pap" and self.power:
                out.append((mc.x, mc.y, 0.6, (190, 90, 255), 0.8, 0.45))
            elif mc.kind == "barrel":
                out.append((mc.x, mc.y, 0.75, (255, 140, 40), 1.1, 0.7))
            elif mc.kind == "power":
                out.append((mc.x, mc.y, 0.65, (80, 255, 100) if self.power else (255, 60, 40), 0.3, 0.6))
        b = self.box
        if b.spots and b.state != "away":
            bx, by = b.spots[b.spot]
            out.append((bx, by, 0.6, (255, 196, 90), 0.8, 0.45 if b.state == "idle" else 0.7))
        if b.state == "leave":
            out.append((b.spots[b.spot][0], b.spots[b.spot][1], 0.8, (255, 40, 30), 1.2,
                        0.9 if int(b.t * 6) % 2 == 0 else 0.2))
        for pu in self.powerups:
            out.append((pu.x, pu.y, 0.4, (255, 200, 90), 0.7, 0.7))
        for e in self.effects:
            if e.kind in ("boom", "ray"):
                k = 1 - e.t / e.life
                out.append((e.x, e.y, e.z, e.color, e.size * 1.2, k))
        return out

    def light_beam(self) -> tuple[float, float] | None:
        """Where the Lucky Crate's light column stands."""
        b = self.box
        if b.spots and b.state != "away":
            return b.spots[b.spot]
        return None


__all__ = ["ZombiesMode", "Zombie", "MAX_ALIVE", "field"]
