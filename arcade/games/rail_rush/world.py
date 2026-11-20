"""Rail Rush simulation (no Qt): the three tracks ahead of the runner with their trains, ramps, barriers, coins and
power-ups, the scenery zones (city, tunnels, bridges), and the runner's physics and collisions.

World units are metres: ``x`` across the tracks (lane centres -2.6, 0, +2.6), ``y`` up, ``z`` along the line (the
runner is at ``z = pz`` and runs toward +z). The game layer reads ``events`` after every ``update``."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from app.arcade.games.rail_rush.data import HOVERBOARD_TIME, powerup_duration

LANE_W = 2.6
TRAIN_HALF_W = 1.15
TRAIN_H = 3.5
CAR_LEN = 12.0
CAR_GAP = 0.8
RAMP_LEN = 9.0
GRAVITY = 46.0
JUMP_H = 2.15
SNEAKER_JUMP_H = 4.7
ROLL_TIME = 0.62
FLY_H = 8.5
HALF_W = 0.35  # the runner's half width / half depth (m)
HALF_D = 0.35
STAND_H = 1.75
ROLL_H = 0.75
STEP = 0.6  # the runner steps up onto a surface this much higher than the feet
LANE_TIME = 0.12  # s for a lane change
COIN_Y = 0.95
GEN_AHEAD = 230.0
START_SPEED = 12.0
MAX_SPEED = 24.0
TOP_SPEED_AT = 180.0  # s of running to reach the top speed, which then holds
STUMBLE_WINDOW = 4.0

BARRIERS = {  # kind -> (y bottom, y top, half width, depth)
    "low": (0.0, 1.0, 1.1, 0.4),
    "mid": (0.85, 1.5, 1.1, 0.4),
    "high": (1.0, 3.3, 1.15, 0.4),
    "signal": (0.0, 3.3, 0.38, 0.5),
}
POWER_KINDS = ("jetpack", "sneakers", "magnet", "double")


def lane_x(lane: int) -> float:
    return (lane - 1) * LANE_W


def run_speed(t: float) -> float:
    """m/s after ``t`` seconds of running: an easing-out climb that reaches the top speed after three minutes."""
    u = min(max(t / TOP_SPEED_AT, 0.0), 1.0)
    return START_SPEED + (MAX_SPEED - START_SPEED) * (1.0 - (1.0 - u) ** 2)


@dataclass(slots=True)
class Obstacle:
    kind: str  # train, ramp, low, mid, high, signal
    lane: int
    z: float  # the near end (m)
    length: float
    vz: float = 0.0  # m/s along z (negative: an oncoming train)
    style: int = 0
    cars: int = 1
    active: bool = False
    hit: bool = False
    passed: bool = False

    @property
    def x(self) -> float:
        return lane_x(self.lane)

    @property
    def half_w(self) -> float:
        if self.kind in ("train", "ramp"):
            return TRAIN_HALF_W
        return BARRIERS[self.kind][2]

    @property
    def far(self) -> float:
        return self.z + self.length

    def yrange(self, z: float) -> tuple[float, float]:
        if self.kind == "train":
            return 0.0, TRAIN_H
        if self.kind == "ramp":
            return 0.0, TRAIN_H * min(1.0, max(0.0, (z - self.z) / self.length))
        b = BARRIERS[self.kind]
        return b[0], b[1]


@dataclass(slots=True)
class Coin:
    x: float
    y: float
    z: float
    taken: bool = False
    pulled: bool = False


@dataclass(slots=True)
class Pickup:
    kind: str  # jetpack, sneakers, magnet, double, mystery, key
    x: float
    y: float
    z: float
    taken: bool = False


@dataclass(slots=True)
class Zone:
    kind: str  # city, tunnel, bridge
    z0: float
    z1: float


@dataclass(slots=True)
class Side:
    kind: str  # wall, building, fence, tunnel, bridge
    z0: float
    z1: float
    seed: int
    height: float
    back: float  # height of the building behind (0: none)


class World:
    """One run (or the still scene behind the title screen)."""

    def __init__(self, seed: int | None = None, upgrades: dict | None = None) -> None:
        self.rng = random.Random(seed)
        self.upgrades = dict(upgrades or {})
        self.obstacles: list[Obstacle] = []
        self.coins: list[Coin] = []
        self.pickups: list[Pickup] = []
        self.zones: list[Zone] = []
        self.sides: tuple[list[Side], list[Side]] = ([], [])
        self.events: list[tuple] = []
        # the runner
        self.pz = 0.0
        self.speed = 0.0
        self.lane = 1
        self.prev_lane = 1
        self.x = 0.0
        self.lean = 0.0
        self.y = 0.0
        self.vy = 0.0
        self.grounded = True
        self.ground = 0.0
        self.roll = 0.0
        self.roll_queued = False
        self.jump_buffer = 0.0
        self.coyote = 0.0
        self.fast_drop = False
        self.power: dict[str, float] = {}
        self.board = 0.0
        self.invuln = 0.0
        self.stumble = 0.0  # s left of the window in which a second stumble gets you caught
        self.stumble_anim = 0.0
        self.chase = 0.0  # 0..1: how close the yard drone is
        self.running = False
        self.dead = False
        self.death = ""
        self.dead_time = 0.0
        self.run_time = 0.0
        self.score = 0.0
        self.distance = 0.0
        self.run_coins = 0
        self.multiplier = 1
        self.clear: tuple[float, float] = (0.0, 0.0)
        self.anim = 0.0  # run cycle phase
        self.air_time = 0.0
        # generators
        self.gen_z = 50.0
        self.zone_z = -60.0
        self.side_z = [-40.0, -40.0]
        self._start_scene()
        self._fill()

    # --- the scene and its generation ---------------------------------------------------------------------------

    def _start_scene(self) -> None:
        """Parked trains on both sides of the start (the runner sets off between them)."""
        length = 3 * CAR_LEN + 2 * CAR_GAP
        self.obstacles.append(Obstacle("train", 0, 5.0 - length, length, style=1, cars=3))
        self.obstacles.append(Obstacle("train", 2, 9.0 - length, length, style=3, cars=3))
        for i in range(8):
            self.coins.append(Coin(0.0, COIN_Y, 14.0 + i * 2.4))

    def _fill(self) -> None:
        while self.zone_z < self.pz + GEN_AHEAD + 160:
            self._gen_zone()
        for side in (0, 1):
            while self.side_z[side] < self.pz + GEN_AHEAD + 40:
                self._gen_side(side)
        while self.gen_z < self.pz + GEN_AHEAD:
            self._gen_block()

    def _gen_zone(self) -> None:
        z = self.zone_z
        if z < 200:
            length, kind = 260.0 - z, "city"
        elif self.zones and self.zones[-1].kind not in ("city", "country"):
            open_land = [zn.kind for zn in self.zones if zn.kind in ("city", "country")]
            kind = "country" if open_land and open_land[-1] == "city" else "city"
            length = self.rng.uniform(260, 520)
        else:
            kind = "tunnel" if self.rng.random() < 0.55 else "bridge"
            length = self.rng.uniform(70, 120) if kind == "tunnel" else self.rng.uniform(80, 130)
        self.zones.append(Zone(kind, z, z + length))
        self.zone_z = z + length

    def zone_at(self, z: float) -> Zone | None:
        for zone in self.zones:
            if zone.z0 <= z < zone.z1:
                return zone
        return None

    def _gen_side(self, side: int) -> None:
        z = self.side_z[side]
        zone = self.zone_at(z)
        rng = self.rng
        if zone is None or zone.kind in ("city", "country"):
            end = zone.z1 if zone else z + 30
            if zone is not None and zone.kind == "country":
                kind = rng.choices(("trees", "cliff", "fence"), (0.55, 0.3, 0.15))[0]
            else:
                kind = rng.choices(("wall", "building", "fence"), (0.5, 0.28, 0.22))[0]
            length = min(rng.uniform(14, 34) if kind != "building" else rng.uniform(18, 30), end - z)
            if end - (z + length) < 6:
                length = end - z
            height = {"wall": rng.uniform(3.0, 4.2), "building": rng.uniform(9, 17), "fence": 2.3, "trees": 2.0,
                      "cliff": rng.uniform(6.0, 12.0)}[kind]
            back = rng.uniform(8, 26) if kind in ("wall", "fence") and rng.random() < 0.8 else 0.0
            if zone is not None and zone.kind == "country":
                back = 0.0
        else:
            kind, length, height, back = zone.kind, zone.z1 - z, 0.0, 0.0
        length = max(length, 1.0)
        self.sides[side].append(Side(kind, z, z + length, rng.randrange(1 << 30), height, back))
        self.side_z[side] = z + length

    def _free(self, lane: int, z0: float, z1: float, kinds: tuple = ("train", "ramp", "low", "mid", "high",
                                                                        "signal")) -> bool:
        for o in self.obstacles:
            if o.lane == lane and o.kind in kinds and o.z < z1 and o.far > z0:
                return False
        return True

    def _line(self, lane: int, z0: float, n: int, y: float = COIN_Y, gap: float = 2.4) -> float:
        for i in range(n):
            self.coins.append(Coin(lane_x(lane), y, z0 + i * gap))
        return z0 + n * gap

    def _arc(self, lane: int, zc: float, n: int = 7, base: float = 0.0, h: float = 1.9, span: float = 13.0) -> None:
        for i in range(n):
            u = i / (n - 1)
            self.coins.append(Coin(lane_x(lane), base + COIN_Y + h * 4 * u * (1 - u), zc - span / 2 + u * span))

    def _train(self, lane: int, z: float, cars: int, moving: bool = False) -> Obstacle:
        o = Obstacle("train", lane, z, cars * CAR_LEN + (cars - 1) * CAR_GAP, style=self.rng.randrange(6), cars=cars)
        if moving:
            o.vz = -self.rng.uniform(8.0, 11.0)
            o.style = 6 + self.rng.randrange(2)
        self.obstacles.append(o)
        return o

    def _pickup(self, z0: float, z1: float, kinds: tuple, weights: tuple) -> None:
        lanes = [ln for ln in range(3) if self._free(ln, z0 - 2, z0 + 2)]
        if lanes:
            lane = self.rng.choice(lanes)
            self.pickups.append(Pickup(self.rng.choices(kinds, weights)[0], lane_x(lane), 1.1, z0))

    def _gen_block(self) -> None:
        z = self.gen_z
        rng = self.rng
        d = min(1.0, z / 5000.0)  # difficulty
        kinds = ("barriers", "trains", "ramp", "oncoming", "coins")
        weights = (3.0, 2.6 + d, 2.0, 0.6 + 2.2 * d if z > 400 else 0.0, 1.6 - d)
        kind = rng.choices(kinds, weights)[0]
        if kind == "barriers":
            end = self._block_barriers(z, d)
        elif kind == "trains":
            end = self._block_trains(z, d)
        elif kind == "ramp":
            end = self._block_ramp(z, d)
        elif kind == "oncoming":
            end = self._block_oncoming(z, d)
        else:
            end = self._block_coins(z)
        if rng.random() < 0.16 and z > 150:
            self._pickup(z + rng.uniform(3, 8), end, ("jetpack", "sneakers", "magnet", "double", "mystery"),
                         (1.0, 1.2, 1.4, 1.2, 0.6))
        elif rng.random() < 0.025 and z > 600:
            self._pickup(z + 4, end, ("key",), (1.0,))
        self._apply_clear()
        self.gen_z = end + rng.uniform(9.0, 16.0) * (1.0 - 0.3 * d)

    def _block_barriers(self, z: float, d: float) -> float:
        rng = self.rng
        rows = rng.randint(2, 3)
        coin_lane = rng.randrange(3)
        zz = z
        for _ in range(rows):
            n = rng.choices((1, 2, 3), (0.45, 0.4, 0.15 + 0.2 * d))[0]
            lanes = rng.sample(range(3), n)
            for lane in lanes:
                kind = rng.choices(("low", "mid", "high", "signal"), (1.0, 0.9, 0.9, 0.35 if n < 3 else 0.0))[0]
                if kind == "signal" and lane == coin_lane:
                    kind = "low"
                self.obstacles.append(Obstacle(kind, lane, zz + rng.uniform(-0.3, 0.3) * (n == 1), BARRIERS[kind][3]))
                if lane == coin_lane:
                    if kind in ("low", "mid"):
                        self._arc(lane, zz, 7)
                    elif kind == "high":
                        for i in range(5):
                            self.coins.append(Coin(lane_x(lane), 0.55, zz - 4.8 + i * 2.4))
            if coin_lane not in lanes:
                self._line(coin_lane, zz - 5, 5)
            zz += rng.uniform(17, 25) * (1.0 - 0.25 * d)
        return zz - 8

    def _block_trains(self, z: float, d: float) -> float:
        rng = self.rng
        free = rng.randrange(3)
        end = z
        for lane in range(3):
            if lane == free:
                continue
            if rng.random() < 0.25 and d < 0.5 and end > z:
                continue
            start = z + rng.uniform(0, 10)
            cars = rng.choices((1, 2, 3), (0.35, 0.4, 0.25 + 0.2 * d))[0]
            o = self._train(lane, start, cars)
            end = max(end, o.far)
        end = max(end, z + 24)
        # the free lane: coins, and a barrier now and then
        zz = z + 2
        if rng.random() < 0.5 + 0.4 * d:
            kind = rng.choice(("low", "mid", "high"))
            bz = z + (end - z) * rng.uniform(0.35, 0.7)
            self.obstacles.append(Obstacle(kind, free, bz, BARRIERS[kind][3]))
            self._line(free, zz, max(1, int((bz - 7 - zz) / 2.4)))
            if kind != "high":
                self._arc(free, bz, 7)
        else:
            self._line(free, zz, int((end - zz) / 2.4))
        return end

    def _block_ramp(self, z: float, d: float) -> float:
        rng = self.rng
        lane = rng.randrange(3)
        self.obstacles.append(Obstacle("ramp", lane, z, RAMP_LEN, style=rng.randrange(2)))
        cars = rng.choice((2, 3))
        roof = self._train(lane, z + RAMP_LEN, cars)
        # coins up the ramp and along the roof
        for i in range(4):
            u = (i + 0.5) / 4
            self.coins.append(Coin(lane_x(lane), TRAIN_H * u + COIN_Y, z + u * RAMP_LEN))
        self._line(lane, roof.z + 1.5, int((roof.length - 3) / 2.4), TRAIN_H + COIN_Y)
        end = roof.far
        others = [ln for ln in range(3) if ln != lane]
        rng.shuffle(others)
        for i, ln in enumerate(others):
            if i == 0 or rng.random() < 0.45 + 0.4 * d:
                start = z + RAMP_LEN + rng.uniform(-2, 14)
                o = self._train(ln, start, rng.choice((1, 2, 3)))
                end = max(end, o.far)
                if rng.random() < 0.5:
                    self._line(ln, o.z + 3, 4, TRAIN_H + COIN_Y)
            else:
                self._line(ln, z + 4, int((end - z - 8) / 2.4))
        self._arc(lane, roof.far + 4, 6, base=TRAIN_H * 0.5, h=1.6, span=8)
        if rng.random() < 0.25:
            self.pickups.append(Pickup(rng.choice(("sneakers", "magnet", "double")), lane_x(lane), TRAIN_H + 1.1,
                                       roof.z + roof.length * 0.6))
        return end

    def _block_oncoming(self, z: float, d: float) -> float:
        rng = self.rng
        lane = rng.randrange(3)
        cars = rng.choice((1, 2))
        o = self._train(lane, z + 68, cars, moving=True)
        end = o.far
        others = [ln for ln in range(3) if ln != lane]
        rng.shuffle(others)
        coin_lane = others[0]
        self._line(coin_lane, z + 6, 12)
        other = others[1]
        if rng.random() < 0.55:
            t = self._train(other, z + rng.uniform(5, 30), rng.choice((1, 2)))
            end = max(end, t.far)
        elif rng.random() < 0.6:
            kind = rng.choice(("low", "mid", "high"))
            self.obstacles.append(Obstacle(kind, other, z + 30, BARRIERS[kind][3]))
        if rng.random() < 0.3 + 0.4 * d:
            kind = rng.choice(("low", "mid"))
            self.obstacles.append(Obstacle(kind, coin_lane, z + 40, BARRIERS[kind][3]))
            self.coins[:] = [c for c in self.coins if not (c.x == lane_x(coin_lane) and abs(c.z - (z + 40)) < 4)]
            self._arc(coin_lane, z + 40, 7)
        return end

    def _block_coins(self, z: float) -> float:
        rng = self.rng
        lane = rng.randrange(3)
        zz = z
        for _ in range(rng.randint(2, 3)):
            zz = self._line(lane, zz, 5)
            nxt = max(0, min(2, lane + rng.choice((-1, 1))))
            lane = nxt
        if rng.random() < 0.5:
            kind = rng.choice(("low", "mid"))
            ln = rng.randrange(3)
            if ln != lane:
                self.obstacles.append(Obstacle(kind, ln, zz + 4, BARRIERS[kind][3]))
        return zz + 4

    def _apply_clear(self) -> None:
        z0, z1 = self.clear
        if z1 <= z0:
            return
        self.obstacles[:] = [o for o in self.obstacles if not (o.z < z1 and o.far > z0 and o.z > self.pz + 1)]

    # --- the runner ---------------------------------------------------------------------------------------------

    @property
    def rolling(self) -> bool:
        return self.roll > 0.0

    @property
    def flying(self) -> bool:
        return self.power.get("jetpack", 0.0) > 0.0

    def duration(self, kind: str) -> float:
        return powerup_duration(kind, self.upgrades.get(kind, 0))

    def start(self) -> None:
        self.running = True
        self.chase = 1.0
        self.speed = START_SPEED

    def move(self, step: int) -> None:
        """Change lane left (-1) or right (+1)."""
        if not self.running or self.dead:
            return
        target = self.lane + step
        if not 0 <= target <= 2:  # a bump against the wall: a wobble, no penalty
            self.events.append(("bump", step))
            self.stumble_anim = max(self.stumble_anim, 0.3)
            return
        self.prev_lane = self.lane
        self.lane = target
        self.events.append(("lane", step))

    def jump(self) -> None:
        if not self.running or self.dead:
            return
        if self.flying:
            return
        if self.grounded or self.coyote > 0.0:
            h = SNEAKER_JUMP_H if self.power.get("sneakers", 0.0) > 0 else JUMP_H
            self.vy = math.sqrt(2 * GRAVITY * h)
            self.grounded = False
            self.coyote = 0.0
            self.roll = 0.0
            self.roll_queued = False
            self.fast_drop = False
            self.air_time = 0.0
            self.events.append(("jump", h > JUMP_H))
        else:
            self.jump_buffer = 0.14

    def duck(self) -> None:
        if not self.running or self.dead:
            return
        if self.flying:
            return
        if self.grounded:
            if self.roll <= 0.0:
                self.events.append(("roll",))
            self.roll = ROLL_TIME
        else:
            self.vy = min(self.vy, -30.0)
            self.fast_drop = True
            self.roll_queued = True
            self.jump_buffer = 0.0

    def use_board(self) -> bool:
        if not self.running or self.dead or self.board > 0.0:
            return False
        self.board = HOVERBOARD_TIME
        self.events.append(("board",))
        return True

    def support(self) -> float:
        """The height the runner stands on (0, a ramp, a train roof)."""
        best = 0.0
        lane = min(2, max(0, round(self.x / LANE_W) + 1))
        for o in self.obstacles:
            if o.lane != lane or o.kind not in ("train", "ramp"):
                continue
            if o.z - 0.25 <= self.pz <= o.far + 0.25:
                top = o.yrange(min(max(self.pz, o.z), o.far))[1]
                if self.y >= top - STEP and top > best:
                    best = top
        return best

    def revive(self) -> None:
        """Save me: back on the tracks, everything just ahead cleared away."""
        self.dead = False
        self.death = ""
        self.obstacles[:] = [o for o in self.obstacles if not (o.z < self.pz + 32 and o.far > self.pz - 3)]
        self.y = self.support()
        self.vy = 0.0
        self.grounded = True
        self.roll = 0.0
        self.invuln = 3.0
        self.stumble = 0.0
        self.stumble_anim = 0.0
        self.lane = min(2, max(0, round(self.x / LANE_W) + 1))
        self.prev_lane = self.lane
        self.events.append(("revive",))

    def _crash(self, o: Obstacle | None) -> None:
        if self.invuln > 0.0:
            return
        if self.board > 0.0:
            self.board = 0.0
            self.invuln = 2.0
            self.events.append(("board_break",))
            return
        self.dead = True
        self.death = "crash"
        self.dead_time = 0.0
        self.chase = max(self.chase, 0.2)
        if o is not None:
            o.hit = True
        self.events.append(("crash",))

    def _stumble_side(self, o: Obstacle | None) -> None:
        if o is not None:
            o.hit = True
        if self.lane != self.prev_lane:
            self.lane, self.prev_lane = self.prev_lane, self.lane
        if self.invuln > 0.0 or self.stumble_anim > 0.25:
            return
        self.stumble_anim = 0.5
        if self.stumble > 0.0:
            self.dead = True
            self.death = "caught"
            self.dead_time = 0.0
            self.events.append(("caught",))
            return
        self.stumble = STUMBLE_WINDOW
        self.events.append(("stumble",))

    def collect_jetpack(self) -> None:
        dur = self.duration("jetpack")
        self.power["jetpack"] = dur
        self.grounded = False
        self.roll = 0.0
        span = max(self.speed, START_SPEED) * dur
        z = self.pz + 14
        lane = min(2, max(0, round(self.x / LANE_W) + 1))
        end = self.pz + span
        while z < end - 8:
            n = self.rng.randint(6, 10)
            for i in range(n):
                if z + i * 2.6 < end - 6:
                    self.coins.append(Coin(lane_x(lane), FLY_H + 0.7, z + i * 2.6))
            z += n * 2.6 + 1.5
            lane = max(0, min(2, lane + self.rng.choice((-1, 1))))
        self.clear = (end - 4, end + 40)
        self._apply_clear()

    def update(self, dt: float) -> None:
        if self.dead:
            self.dead_time += dt
            self.chase = min(1.0, self.chase + dt * 1.6)
            self.stumble_anim = max(0.0, self.stumble_anim - dt)
            return
        if not self.running:
            self.anim += dt * 2.0
            return
        self.run_time += dt
        self.speed = run_speed(self.run_time)
        for k in list(self.power):
            self.power[k] -= dt
            if self.power[k] <= 0.0:
                del self.power[k]
                self.events.append(("power_end", k))
                if k == "jetpack":
                    self.invuln = max(self.invuln, 1.2)
        if self.board > 0.0:
            self.board = max(0.0, self.board - dt)
            if self.board == 0.0:
                self.events.append(("board_end",))
        self.invuln = max(0.0, self.invuln - dt)
        self.stumble = max(0.0, self.stumble - dt)
        self.stumble_anim = max(0.0, self.stumble_anim - dt)
        self.jump_buffer = max(0.0, self.jump_buffer - dt)
        self.coyote = max(0.0, self.coyote - dt)
        if self.run_time < 2.5 or self.stumble > 0.0:
            self.chase = min(1.0, self.chase + dt * 2.5)
        else:
            self.chase = max(0.0, self.chase - dt * 0.7)
        # forward
        step = self.speed * dt
        self.pz += step
        self.distance += step
        self.anim += dt * self.speed * 0.30
        for o in self.obstacles:
            if o.vz:
                if not o.active and o.z - self.pz < 95.0:
                    o.active = True
                if o.active:
                    o.z += o.vz * dt
        # across
        tx = lane_x(self.lane)
        dx = tx - self.x
        mv = LANE_W / LANE_TIME * dt
        self.x = tx if abs(dx) <= mv else self.x + math.copysign(mv, dx)
        if self.x == tx:
            self.prev_lane = self.lane
        target_lean = max(-1.0, min(1.0, dx / LANE_W * 2.0))
        self.lean += (target_lean - self.lean) * min(1.0, dt * 14.0)
        # up and down
        if self.flying:
            self.y += (FLY_H - self.y) * min(1.0, dt * 3.0)
            self.vy = 0.0
            self.grounded = False
        else:
            ground = self.support()
            if self.grounded and self.y - ground > 0.05:
                self.grounded = False
                self.vy = 0.0
                self.coyote = 0.12
            if not self.grounded:
                g = GRAVITY * (1.6 if self.fast_drop else 1.0)
                self.vy -= g * dt
                self.y += self.vy * dt
                self.air_time += dt
                ground = self.support()
                if self.y <= ground:
                    self._land(ground)
            else:
                self.y = ground
            self.ground = ground
        if self.roll > 0.0:
            self.roll = max(0.0, self.roll - dt)
        self._collide(dt)
        if self.dead:
            return
        self._collect(dt)
        mult = self.multiplier * (2 if self.power.get("double", 0.0) > 0 else 1)
        self.score += step * mult
        # generation and clean-up
        self._fill()
        behind = self.pz - 30.0
        self.obstacles[:] = [o for o in self.obstacles if o.far > behind]
        self.coins[:] = [c for c in self.coins if c.z > self.pz - 6 and not c.taken]
        self.pickups[:] = [p for p in self.pickups if p.z > self.pz - 6 and not p.taken]
        self.zones[:] = [zn for zn in self.zones if zn.z1 > behind - 60]
        for side in self.sides:
            side[:] = [s for s in side if s.z1 > behind - 20]

    def _land(self, ground: float) -> None:
        hard = self.vy < -14.0
        self.y = ground
        self.vy = 0.0
        self.grounded = True
        self.fast_drop = False
        if ground >= TRAIN_H - 0.01 and self.air_time > 0.15:
            self.events.append(("roof",))
        self.events.append(("land", hard))
        self.air_time = 0.0
        if self.roll_queued:
            self.roll_queued = False
            self.roll = ROLL_TIME
            self.events.append(("roll",))
        if self.jump_buffer > 0.0:
            self.jump_buffer = 0.0
            self.jump()

    def _collide(self, dt: float) -> None:
        if self.flying:
            return
        ptop = self.y + (ROLL_H if self.rolling else STAND_H)
        for o in self.obstacles:
            if o.far < self.pz - 2 or o.z > self.pz + 2:
                continue
            if o.kind in ("low", "mid") and not o.passed and o.far < self.pz - HALF_D:
                o.passed = True
                if abs(self.x - o.x) < 1.2 and self.y > 0.3:
                    self.events.append(("barrier",))
            if o.hit or self.invuln > 0.0:
                continue
            off = abs(self.x - o.x)
            if off >= o.half_w + HALF_W:
                continue
            if not (self.pz + HALF_D > o.z and self.pz - HALF_D < o.far):
                continue
            y0, y1 = o.yrange(self.pz)
            if self.y >= y1 - 0.04 or ptop <= y0:
                continue
            if o.kind in ("train", "ramp") and self.y >= y1 - STEP:
                continue  # stepped up onto it
            depth = self.pz + HALF_D - o.z
            frontal = o.kind not in ("ramp",) and depth <= (self.speed - o.vz) * dt + 0.12
            if o.kind in BARRIERS:
                frontal = True
            if frontal and off < o.half_w * 0.55 + 0.25:
                self._crash(o)
                return
            self._stumble_side(o)
            return

    def _collect(self, dt: float) -> None:
        magnet = self.power.get("magnet", 0.0) > 0.0
        top = self.y + (ROLL_H + 0.5 if self.rolling else STAND_H + 0.4)
        cy = self.y + 0.9
        for c in self.coins:
            if c.taken or c.z > self.pz + 26:
                continue
            if magnet and not c.pulled and c.z < self.pz + 20 and c.z > self.pz - 1:
                c.pulled = True
            if c.pulled:
                k = min(1.0, dt * 14.0)
                c.x += (self.x - c.x) * k
                c.y += (cy - c.y) * k
                c.z += (self.pz + 0.3 - c.z) * k + self.speed * dt * 0.5
                if abs(c.z - self.pz) < 1.0 and abs(c.x - self.x) < 0.8:
                    c.taken = True
                    self.run_coins += 1
                    self.events.append(("coin", True))
                continue
            if abs(c.z - self.pz) < 0.9 and abs(c.x - self.x) < 1.0 and self.y - 0.5 < c.y < top:
                c.taken = True
                self.run_coins += 1
                self.events.append(("coin", False))
        for pk in self.pickups:
            if pk.taken or abs(pk.z - self.pz) > 1.0 or abs(pk.x - self.x) > 1.1:
                continue
            if self.y - 0.6 < pk.y < top + 0.3:
                pk.taken = True
                if pk.kind == "jetpack":
                    self.collect_jetpack()
                elif pk.kind in POWER_KINDS:
                    self.power[pk.kind] = self.duration(pk.kind)
                self.events.append(("pickup", pk.kind))
