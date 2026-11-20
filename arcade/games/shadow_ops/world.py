"""The shared first-person core of both modes: the player (shooter movement: walk, sprint with stamina, crouch, jump,
aim down the sights), weapon handling (semi / burst / auto fire, magazines, reloads, recoil, spread, weapon swaps,
the knife and grenades), grid collision, hitscan traces, line of sight and grid flow fields for the AI."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .data import ATTACHMENTS, WEAPONS, Weapon, weapon
from .maps import MapData
from .gfx import EYE

RADIUS = 0.22
WALK = 2.6  # cells / s
SPRINT = 1.55
CROUCH_EYE = 0.52
GRAVITY = 11.0


# --- weapons ---------------------------------------------------------------------------------------------------------

@dataclass
class Gun:
    """A weapon the player carries."""
    wid: str
    mag: int = -1
    reserve: int = -1
    upgraded: bool = False
    atts: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.wid not in WEAPONS:
            self.wid = "service45"
        if self.mag < 0:
            self.mag = self.mag_size()
        if self.reserve < 0:
            self.reserve = self.reserve_max()

    @property
    def w(self) -> Weapon:
        return weapon(self.wid)

    @property
    def name(self) -> str:
        return (self.w.pap or self.w.name) if self.upgraded else self.w.name

    def mag_size(self) -> int:
        n = self.w.mag
        if "ext" in self.atts:
            n = int(n * 1.5)
        if self.upgraded:
            n = int(math.ceil(n * 1.5)) if self.w.mag > 2 else n
        return max(n, 1)

    def reserve_max(self) -> int:
        n = self.w.reserve
        if self.upgraded:
            n = int(n * 1.6)
        return n

    def refill(self) -> None:
        self.mag = self.mag_size()
        self.reserve = self.reserve_max()

    def to_json(self) -> dict:
        return {"wid": self.wid, "mag": self.mag, "reserve": self.reserve, "upgraded": self.upgraded,
                "atts": list(self.atts)}

    @classmethod
    def from_json(cls, d: dict) -> Gun:
        try:
            return cls(str(d.get("wid", "service45")), int(d.get("mag", -1)), int(d.get("reserve", -1)),
                       bool(d.get("upgraded", False)), [a for a in d.get("atts", []) if a in ATTACHMENTS])
        except (TypeError, ValueError, AttributeError):
            return cls("service45")

    def sight(self) -> str:
        if self.w.sight == "scope":
            return "scope"
        if "acog" in self.atts:
            return "acog"
        if "reflex" in self.atts or self.w.sight == "reddot":
            return "reddot"
        return "iron"


# --- grid helpers ----------------------------------------------------------------------------------------------------

def blocked(m: MapData, x: float, y: float, r: float = RADIUS, grid: np.ndarray | None = None) -> bool:
    g = m.block if grid is None else grid
    x0, x1 = int(math.floor(x - r)), int(math.floor(x + r))
    y0, y1 = int(math.floor(y - r)), int(math.floor(y + r))
    if x0 < 0 or y0 < 0 or x1 >= m.w or y1 >= m.h:
        return True
    return bool(g[y0:y1 + 1, x0:x1 + 1].any())


def slide(m: MapData, x: float, y: float, dx: float, dy: float, r: float = RADIUS,
          grid: np.ndarray | None = None) -> tuple[float, float]:
    """Move by (dx, dy) with axis-separated collision (sliding along walls); big moves in small steps."""
    steps = max(1, int(max(abs(dx), abs(dy)) / 0.2) + 1)
    sx, sy = dx / steps, dy / steps
    for _ in range(steps):
        if sx and not blocked(m, x + sx, y, r, grid):
            x += sx
        if sy and not blocked(m, x, y + sy, r, grid):
            y += sy
    return x, y


def trace(m: MapData, x: float, y: float, dx: float, dy: float, maxd: float = 64.0,
          grid: np.ndarray | None = None, z0: float | None = None, slope: float = 0.0) -> float:
    """Distance along the unit direction (dx, dy) to the first blocking cell (``m.bullet`` by default). With ``z0`` (the
    height the line starts at, rising ``slope`` per cell) low cover stops it too when it passes below the top."""
    tops = m.top if z0 is not None else None
    g = m.bullet if grid is None else grid
    mx, my = int(x), int(y)
    ddx = abs(1.0 / dx) if dx else 1e30
    ddy = abs(1.0 / dy) if dy else 1e30
    stx, sty = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    sdx = ((mx + 1 - x) if dx >= 0 else (x - mx)) * ddx
    sdy = ((my + 1 - y) if dy >= 0 else (y - my)) * ddy
    d = 0.0
    while d < maxd:
        if sdx < sdy:
            d = sdx
            sdx += ddx
            mx += stx
        else:
            d = sdy
            sdy += ddy
            my += sty
        if not (0 <= mx < m.w and 0 <= my < m.h) or g[my, mx]:
            return min(d, maxd)
        if tops is not None and tops[my, mx] > 0.0 and d > 0.4 and z0 + slope * d < tops[my, mx]:
            return min(d, maxd)
    return maxd


def line_of_sight(m: MapData, x0: float, y0: float, x1: float, y1: float, grid: np.ndarray | None = None) -> bool:
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return True
    return trace(m, x0, y0, dx / dist, dy / dist, dist, grid) >= dist - 1e-3


def flow_field(m: MapData, goals: list[tuple[int, int]], passable: np.ndarray, limit: int = 4000) -> np.ndarray:
    """Steps from every cell to the nearest goal over ``passable`` cells (4-neighbour BFS); unreachable = big."""
    dist = np.full((m.h, m.w), 9999, np.int32)
    todo: deque = deque()
    for gx, gy in goals:
        if 0 <= gx < m.w and 0 <= gy < m.h:
            dist[gy, gx] = 0
            todo.append((gx, gy))
    n = 0
    while todo and n < limit * 4:
        x, y = todo.popleft()
        n += 1
        d = dist[y, x] + 1
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < m.w and 0 <= ny < m.h and passable[ny, nx] and dist[ny, nx] > d:
                dist[ny, nx] = d
                todo.append((nx, ny))
    return dist


def flow_dir(m: MapData, field_: np.ndarray, x: float, y: float, passable: np.ndarray) -> tuple[float, float] | None:
    """The direction to walk from (x, y) down the flow field (towards the middle of the best neighbour)."""
    cx, cy = int(x), int(y)
    if not (0 <= cx < m.w and 0 <= cy < m.h):
        return None
    best, bd = None, field_[cy, cx]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        nx, ny = cx + dx, cy + dy
        if not (0 <= nx < m.w and 0 <= ny < m.h) or not passable[ny, nx]:
            continue
        if dx and dy and not (passable[cy, nx] and passable[ny, cx]):
            continue
        d = field_[ny, nx] + (0.4 if dx and dy else 0.0)
        if d < bd:
            best, bd = (nx, ny), d
    if best is None:
        return None
    tx, ty = best[0] + 0.5, best[1] + 0.5
    vx, vy = tx - x, ty - y
    n = math.hypot(vx, vy) or 1.0
    return vx / n, vy / n


# --- targets and shots -----------------------------------------------------------------------------------------------

@dataclass
class Target:
    obj: object
    x: float
    y: float
    r: float = 0.26
    z: float = 0.0
    h: float = 1.0
    head: float = 0.22  # top fraction that is the head


@dataclass
class Hit:
    target: Target | None
    dist: float
    x: float
    y: float
    z: float
    head: bool


def shoot_ray(m: MapData, x: float, y: float, eye: float, yaw: float, slope: float, targets: list[Target],
              maxd: float = 60.0) -> Hit:
    """The first thing a bullet from (x, y, eye) along ``yaw`` rising ``slope`` per cell hits."""
    dx, dy = math.cos(yaw), math.sin(yaw)
    wall = trace(m, x, y, dx, dy, maxd, None, eye, slope)
    best: Hit | None = None
    for t in targets:
        ox, oy = t.x - x, t.y - y
        along = ox * dx + oy * dy
        if along <= 0.05 or along >= wall or (best is not None and along >= best.dist):
            continue
        perp = abs(ox * dy - oy * dx)
        if perp > t.r:
            continue
        hit_at = along - math.sqrt(max(t.r * t.r - perp * perp, 0.0)) * 0.5
        z = eye + slope * hit_at
        if t.z <= z <= t.z + t.h:
            head = z >= t.z + t.h * (1 - t.head) and abs(perp) < t.r * 0.75
            best = Hit(t, hit_at, x + dx * hit_at, y + dy * hit_at, z, head)
    if best is not None:
        return best
    return Hit(None, wall, x + dx * wall, y + dy * wall, eye + slope * wall, False)


def falloff(w: Weapon, dist: float, long_barrel: bool = False) -> float:
    near, far = w.near, w.far
    if long_barrel:
        near, far = near * 1.4, far * 1.4
    if dist <= near:
        return w.dmg
    if dist >= far:
        return w.dmg_min
    return w.dmg + (w.dmg_min - w.dmg) * (dist - near) / (far - near)


# --- the player ------------------------------------------------------------------------------------------------------

@dataclass
class Controls:
    """What the player is asking for this frame."""
    fwd: float = 0.0
    side: float = 0.0
    sprint: bool = False
    crouch: bool = False
    jump: bool = False
    fire: bool = False
    ads: bool = False
    turn: float = 0.0  # radians
    look: float = 0.0  # slope change


class Player:
    def __init__(self, x: float, y: float, angle: float = 0.0) -> None:
        self.x, self.y, self.angle = x, y, angle
        self.look = 0.0  # slope of the view (tan of the pitch)
        self.z = 0.0
        self.vz = 0.0
        self.crouched = False
        self.eye = EYE
        self.health = 100.0
        self.max_health = 100.0
        self.hurt_t = 99.0  # s since last hurt
        self.regen_delay = 3.0
        self.dead = False
        self.guns: list[Gun] = []
        self.cur = 0
        self.max_guns = 2
        self.ads = 0.0
        self.sprinting = False
        self.stamina = 4.0
        self.stamina_max = 4.0
        self.sprint_cool = 0.0  # s before firing after a sprint
        self.reload_t = 0.0
        self.reload_total = 0.0
        self.switch_t = 0.0
        self.switch_to = -1
        self.fire_cd = 0.0
        self.burst = 0
        self.trigger_was = False
        self.knife_t = 0.0
        self.throw_t = 0.0
        self.kick = 0.0  # visual gun recoil 0..1
        self.flash_t = 0.0  # muzzle flash
        self.bob = 0.0
        self.moving = 0.0  # 0..1 how fast
        self.vx = self.vy = 0.0
        self.recoil_pitch = 0.0
        self.recoil_yaw = 0.0
        self.damage_dirs: list[list[float]] = []  # [world angle, time left]
        self.speed_mult = 1.0
        self.reload_mult = 1.0
        self.rate_mult = 1.0
        self.damage_mult = 1.0
        self.grenades = 2
        self.tacticals = 0
        self.shots = 0
        self.hits = 0
        self.headshots = 0
        self.ads_speed_mult = 1.0
        self.nimble = False
        self.infinite_sprint = False
        self.recoil_mult = 1.0
        self.flinch_mult = 1.0
        self.flinch = 0.0
        self.quick_hands = False
        self.reload_hint = False
        self.hip_mult = 1.0  # Steady Hand: tighter hip fire

    @property
    def gun(self) -> Gun | None:
        return self.guns[self.cur] if 0 <= self.cur < len(self.guns) else None

    @property
    def busy(self) -> bool:
        return self.reload_t > 0 or self.switch_t > 0 or self.knife_t > 0 or self.throw_t > 0

    # --- movement ---------------------------------------------------------------------------------------------

    def move(self, m: MapData, c: Controls, dt: float, grid: np.ndarray | None = None) -> None:
        self.angle = (self.angle + c.turn) % (2 * math.pi)
        self.look = min(max(self.look + c.look, -0.9), 0.9)
        # recoil recovers
        rec = min(dt * 6.0, 1.0)
        self.look = min(max(self.look - self.recoil_pitch * rec, -0.9), 0.9)
        self.recoil_pitch -= self.recoil_pitch * rec
        self.angle += self.recoil_yaw * rec
        self.recoil_yaw -= self.recoil_yaw * rec
        if c.crouch != self.crouched:
            self.crouched = c.crouch
        moving = abs(c.fwd) + abs(c.side) > 0.01
        want_sprint = c.sprint and c.fwd > 0.3 and not self.crouched and (self.stamina > 0.2 or self.infinite_sprint)
        if want_sprint and not self.sprinting and self.stamina < 1.0 and not self.infinite_sprint:
            want_sprint = False
        if want_sprint and (c.ads or self.knife_t > 0):
            want_sprint = False
        if want_sprint and self.reload_t > 0:
            self.reload_t = 0.0  # sprinting cancels the reload
        if self.sprinting and not want_sprint:
            self.sprint_cool = 0.05 if self.nimble else 0.22
        self.sprinting = want_sprint
        if self.sprinting and not self.infinite_sprint:
            self.stamina = max(self.stamina - dt, 0.0)
        else:
            self.stamina = min(self.stamina + dt * 0.8, self.stamina_max)
        self.sprint_cool = max(self.sprint_cool - dt, 0.0)
        gun = self.gun
        speed = WALK * self.speed_mult * (gun.w.speed if gun else 1.0)
        if self.sprinting:
            speed *= SPRINT
        elif self.crouched:
            speed *= 0.5
        speed *= 1.0 - 0.45 * self.ads * (0.6 if gun and "stock" in gun.atts else 1.0)
        fx, fy = math.cos(self.angle), math.sin(self.angle)
        rx, ry = -fy, fx
        wx = fx * c.fwd + rx * c.side
        wy = fy * c.fwd + ry * c.side
        n = math.hypot(wx, wy)
        if n > 1:
            wx, wy = wx / n, wy / n
        # a little acceleration so it does not feel rigid
        k = min(dt * (14.0 if self.z <= 0 else 3.0), 1.0)
        self.vx += (wx * speed - self.vx) * k
        self.vy += (wy * speed - self.vy) * k
        self.x, self.y = slide(m, self.x, self.y, self.vx * dt, self.vy * dt, RADIUS, grid)
        sp = math.hypot(self.vx, self.vy)
        self.moving = min(sp / WALK, 1.6) if moving else min(sp / WALK, 1.6) * 0.5
        if self.z <= 0:
            self.bob += dt * sp * 3.1
        # jump and crouch heights
        if c.jump and self.z <= 0 and not self.crouched:
            self.vz = 2.6
        if self.z > 0 or self.vz > 0:
            self.vz -= GRAVITY * dt
            self.z = max(self.z + self.vz * dt, 0.0)
            if self.z == 0:
                self.vz = 0.0
        target = CROUCH_EYE if self.crouched else EYE
        self.eye += (target - self.eye) * min(dt * 10, 1.0)
        # ADS
        can_ads = c.ads and not self.sprinting and self.reload_t <= 0 and self.switch_t <= 0 and self.knife_t <= 0
        t = gun.w.ads_time if gun else 0.2
        if gun and "quick" in gun.atts:
            t *= 0.6
        step = dt / max(t * self.ads_speed_mult, 0.05)
        self.ads = min(self.ads + step, 1.0) if can_ads else max(self.ads - step * 1.3, 0.0)
        # health
        self.hurt_t += dt
        if self.hurt_t > self.regen_delay and self.health < self.max_health and not self.dead:
            self.health = min(self.health + self.max_health * 0.6 * dt, self.max_health)
        for d in self.damage_dirs:
            d[1] -= dt
        self.damage_dirs = [d for d in self.damage_dirs if d[1] > 0]
        self.kick = max(self.kick - dt * 7.0, 0.0)
        self.flash_t = max(self.flash_t - dt, 0.0)
        self.flinch = max(self.flinch - dt * 4, 0.0)

    def view_eye(self) -> float:
        bob = math.sin(self.bob * 2) * 0.012 * min(self.moving, 1.2) if self.z <= 0 else 0.0
        return self.eye + self.z + bob

    # --- weapons ----------------------------------------------------------------------------------------------

    def spread(self) -> float:
        """Current cone half-angle (radians)."""
        gun = self.gun
        if gun is None:
            return 0.0
        w = gun.w
        hip = w.hip * (0.6 if "laser" in gun.atts else 1.0) * self.hip_mult
        ads = w.ads if gun.sight() not in ("scope",) else 0.05
        s = hip + (ads - hip) * self.ads
        s *= 1.0 + 0.6 * min(self.moving, 1.0) * (1 - self.ads * 0.7) + (0.8 if self.z > 0 else 0.0)
        if self.crouched:
            s *= 0.8
        return math.radians(s) * 0.5

    def fire_interval(self) -> float:
        gun = self.gun
        if gun is None:
            return 1.0
        return 60.0 / (gun.w.rpm * self.rate_mult)

    def reload_time(self) -> float:
        gun = self.gun
        if gun is None:
            return 1.0
        t = gun.w.reload * self.reload_mult
        if "fastmag" in gun.atts:
            t *= 0.65
        return t

    def start_reload(self) -> bool:
        gun = self.gun
        if gun is None or self.busy or gun.mag >= gun.mag_size() or gun.reserve <= 0:
            return False
        self.reload_total = self.reload_t = self.reload_time()
        self.burst = 0
        return True

    def switch(self, index: int) -> bool:
        if not (0 <= index < len(self.guns)) or index == self.cur or self.knife_t > 0 or self.throw_t > 0:
            return False
        self.reload_t = 0.0
        self.switch_to = index
        self.switch_t = 0.3 if self.quick_hands else 0.5
        self.burst = 0
        return True

    def give(self, gun: Gun) -> Gun | None:
        """Take a weapon: into a free slot, else in place of the one in hand. Returns the dropped weapon."""
        for i, g in enumerate(self.guns):
            if g.wid == gun.wid:
                self.guns[i] = gun
                self.cur = i
                self.switch_t = 0.45
                self.switch_to = i
                return None
        dropped = None
        if len(self.guns) < self.max_guns:
            self.guns.append(gun)
            self.cur = len(self.guns) - 1
        else:
            dropped = self.guns[self.cur]
            self.guns[self.cur] = gun
        self.reload_t = 0.0
        self.switch_t = 0.45
        self.switch_to = self.cur
        return dropped

    def update_weapon(self, dt: float, trigger: bool) -> int:
        """Advance reload / swap / cooldown timers; returns how many shots go off this frame (the mode fires
        them)."""
        gun = self.gun
        self.fire_cd = max(self.fire_cd - dt, 0.0)
        self.knife_t = max(self.knife_t - dt, 0.0)
        self.throw_t = max(self.throw_t - dt, 0.0)
        if self.switch_t > 0:
            before = self.switch_t
            self.switch_t = max(self.switch_t - dt, 0.0)
            if before > 0.22 >= self.switch_t and 0 <= self.switch_to < len(self.guns):
                self.cur = self.switch_to
            self.trigger_was = trigger
            return 0
        if self.reload_t > 0:
            self.reload_t = max(self.reload_t - dt, 0.0)
            if self.reload_t == 0 and gun is not None:
                need = gun.mag_size() - gun.mag
                take = min(need, gun.reserve)
                gun.mag += take
                gun.reserve -= take
            self.trigger_was = trigger
            return 0
        if gun is None or self.knife_t > 0 or self.throw_t > 0 or self.sprinting or self.sprint_cool > 0:
            self.trigger_was = trigger
            self.burst = 0 if not trigger else self.burst
            return 0
        pressed = trigger and not self.trigger_was
        self.trigger_was = trigger
        shots = 0
        mode = gun.w.mode
        if mode == "burst" and pressed and self.burst == 0 and self.fire_cd <= 0:
            self.burst = 3
        want = (mode == "auto" and trigger) or (mode == "semi" and pressed) or (mode == "burst" and self.burst > 0)
        if want and self.fire_cd <= 0:
            if gun.mag <= 0:
                self.burst = 0
                if pressed or mode == "auto":
                    if not self.start_reload():
                        return -1  # dry fire
                return 0
            gun.mag -= 1
            shots = 1
            if mode == "burst":
                self.burst -= 1
                self.fire_cd = 0.065 if self.burst > 0 else max(self.fire_interval() * 3.2, 0.25)
            else:
                self.fire_cd = self.fire_interval()
            self.shots += 1
            self.kick = 1.0
            self.flash_t = 0.05
            mult = self.recoil_mult * (0.65 if "grip" in gun.atts else 1.0) * (0.7 if self.ads > 0.5 else 1.0)
            self.recoil_pitch += math.tan(math.radians(gun.w.recoil)) * mult * 0.6
            self.look = min(self.look + math.tan(math.radians(gun.w.recoil)) * mult, 0.9)
            self.recoil_yaw += random.uniform(-1, 1) * math.radians(gun.w.recoil) * 0.35 * mult
            if gun.mag == 0 and gun.reserve > 0 and mode != "burst":
                self.reload_hint = True
        elif gun.mag <= 0 and self.fire_cd <= 0 and gun.reserve > 0 and self.burst == 0 and not trigger:
            self.start_reload()  # auto reload on empty
        return shots

    def pellet_dirs(self) -> list[tuple[float, float]]:
        """(yaw, slope) of every pellet of one shot."""
        gun = self.gun
        if gun is None:
            return []
        s = self.spread()
        out = []
        for _ in range(gun.w.pellets):
            a = random.uniform(0, 2 * math.pi)
            r = s * math.sqrt(random.random()) if gun.w.pellets == 1 else s * (0.3 + 0.7 * random.random())
            yaw = self.angle + math.cos(a) * r
            slope = math.tan(math.atan(self.look) + math.sin(a) * r)
            out.append((yaw, slope))
        return out

    def hurt(self, amount: float, src: tuple[float, float] | None = None) -> None:
        if self.dead:
            return
        self.health -= amount
        self.hurt_t = 0.0
        self.flinch = min(self.flinch + 0.6 * self.flinch_mult, 1.0)
        self.look += random.uniform(-0.03, 0.05) * self.flinch_mult
        self.angle += random.uniform(-0.03, 0.03) * self.flinch_mult
        if src is not None:
            self.damage_dirs.append([math.atan2(src[1] - self.y, src[0] - self.x), 1.6])
        if self.health <= 0:
            self.health = 0.0
            self.dead = True


@dataclass
class Grenade:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    fuse: float
    owner: object = None
    kind: str = "frag"
    team: str = ""
    stuck: bool = False

    def update(self, m: MapData, dt: float) -> bool:
        """Fly and bounce; True when it goes off."""
        self.fuse -= dt
        if not self.stuck:
            self.vz -= GRAVITY * dt
            nx, ny = self.x + self.vx * dt, self.y + self.vy * dt
            if blocked(m, nx, self.y, 0.05, m.bullet):
                if self.kind == "sticky":
                    self.stuck = True
                self.vx = -self.vx * 0.45
            else:
                self.x = nx
            if blocked(m, self.x, ny, 0.05, m.bullet):
                if self.kind == "sticky":
                    self.stuck = True
                self.vy = -self.vy * 0.45
            else:
                self.y = ny
            self.z += self.vz * dt
            if self.z <= 0.05:
                self.z = 0.05
                if self.kind == "sticky":
                    self.stuck = True
                self.vz = -self.vz * 0.35
                self.vx *= 0.6
                self.vy *= 0.6
            if self.stuck:
                self.vx = self.vy = self.vz = 0.0
        return self.fuse <= 0


def throw(p: Player, kind: str = "frag", team: str = "") -> Grenade:
    ca, sa = math.cos(p.angle), math.sin(p.angle)
    pitch = math.atan(p.look) + 0.25
    speed = 7.5
    return Grenade(p.x + ca * 0.3, p.y + sa * 0.3, p.view_eye(), ca * speed * math.cos(pitch),
                   sa * speed * math.cos(pitch), speed * math.sin(pitch), 2.2 if kind != "flash" else 1.4, p, kind,
                   team)


@dataclass
class Effect:
    """A short-lived world effect: impact puff, blood, explosion, tracer."""
    kind: str
    x: float
    y: float
    z: float
    t: float
    life: float
    x2: float = 0.0
    y2: float = 0.0
    z2: float = 0.0
    color: tuple = (255, 220, 150)
    size: float = 1.0


def open_facing(m: MapData, x: float, y: float, prefer: float) -> float:
    """A spawn facing: the direction with the longest clear view, leaning towards ``prefer``."""
    best, bs = prefer, -1.0
    for k in range(16):
        a = k * math.pi / 8
        d = trace(m, x, y, math.cos(a), math.sin(a), 20.0, m.solid)
        score = min(d, 12.0) * (1.0 + 0.35 * math.cos(a - prefer))
        if score > bs:
            best, bs = a, score
    return best


def angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi
