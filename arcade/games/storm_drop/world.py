"""One match of Storm Drop: the drop ship, every player (you and the bots), movement, weapons, loot, harvesting,
building and the storm. No Qt and no drawing here: the game widget feeds the player's controls in, steps the match, and
reads what to draw and which sounds to play back out."""

from __future__ import annotations

import math
import random

from .build import RAMP_DX, Structures
from .data import (AMMO, AMMO_CAP, AMMO_PICKUP, BOT_FIRST, BOT_LAST, BUILD_COST, CELL, CHEST_GRADE_WEIGHTS, CHUTE_ALT,
                   CHUTE_GLIDE, CHUTE_V, CROUCH, DIFFICULTY, DIVE_GLIDE, DIVE_V, EYE, FALL_DMG, FLOOR, GRADE_DAMAGE,
                   GRADE_WEIGHTS, GRAVITY, GUNS, HEIGHT, JUMP_V, LEVEL, MAT_CAP, MAX_HP, MAX_SHIELD, OUTFITS, RADIUS,
                   RAMP, ROOF, SAFE_FALL_V, SEA, SHIP_ALT, SHIP_SPEED, SIZE, SLOTS, SPRINT, STEP_UP, STORM_PHASES,
                   STORM_START_R, SUPPLIES, SUPPLY_FOUND, SWIM, WALK, WALLX, WALLZ, WEAPONS, WOOD)
from .island import Island

SWIM_DEPTH = 1.15
BOT_VS_BOT = 0.42  # bots hurt each other less, so the field thins at the storm's pace, not in the first minute
PACE = ((60.0, 0.86), (150.0, 0.62), (270.0, 0.40), (360.0, 0.22), (425.0, 0.10), (445.0, 0.0))  # (s, share left)
HEAR = 95.0  # m: sounds further than this from the listener are not played


class Slot:
    __slots__ = ("id", "grade", "mag", "count")

    def __init__(self, id_: str, grade: int = 0, mag: int = 0, count: int = 1) -> None:
        self.id, self.grade, self.mag, self.count = id_, grade, mag, count

    @property
    def is_weapon(self) -> bool:
        return self.id in WEAPONS


class Item:
    """Something lying on the ground: a weapon, a supply, ammo or materials."""

    __slots__ = ("kind", "id", "grade", "count", "mag", "x", "y", "z", "alive", "slot", "no_auto", "born", "bucket")

    def __init__(self, kind: str, id_, grade: int, count: int, x: float, y: float, z: float, mag: int = 0) -> None:
        self.kind, self.id, self.grade, self.count, self.mag = kind, id_, grade, count, mag
        self.x, self.y, self.z = x, y, z
        self.alive, self.slot, self.no_auto, self.born, self.bucket = True, None, -1.0, 0.0, None

    def label(self) -> str:
        if self.kind == "weapon":
            return WEAPONS[self.id].name
        if self.kind == "supply":
            return f"{SUPPLIES[self.id].name} x{self.count}"
        if self.kind == "ammo":
            return f"{self.id.capitalize()} ammo x{self.count}"
        return f"{('Wood', 'Brick', 'Metal')[self.id]} x{self.count}"


class Chest:
    __slots__ = ("x", "y", "z", "yaw", "opened", "slot", "small")

    def __init__(self, x: float, y: float, z: float, yaw: float, small: bool = False) -> None:
        self.x, self.y, self.z, self.yaw, self.opened, self.slot, self.small = x, y, z, yaw, False, None, small


class Actor:
    __slots__ = (
        "id", "name", "bot", "outfit", "canopy", "x", "y", "z", "vx", "vy", "vz", "yaw", "pitch", "on_ground",
        "state", "hp", "shield", "alive", "inv", "sel", "mats", "ammo", "fire_cd", "reload_t", "reload_full",
        "use_t", "use_full", "use_slot", "bloom", "crouch", "sprint", "ads", "kills", "walk", "swing_t", "hurt_t",
        "attacker", "attacked_t", "storm_acc", "place", "brain", "wish_x", "wish_z", "want_jump", "firing",
        "aim", "build_mat", "damage", "swim", "speed", "died_t", "shot_t", "land_t", "stats", "last_dir")

    def __init__(self, id_: int, name: str, bot: bool, outfit: int, canopy: int) -> None:
        self.id, self.name, self.bot, self.outfit, self.canopy = id_, name, bot, outfit, canopy
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.yaw = self.pitch = 0.0
        self.on_ground = False
        self.state = "ship"  # ship / fall / glide / ground
        self.hp, self.shield, self.alive = float(MAX_HP), 0.0, True
        self.inv: list[Slot | None] = [None] * SLOTS
        self.sel = -1  # -1: the sledge
        self.mats = [0, 0, 0]
        self.ammo = dict.fromkeys(AMMO, 0)
        self.fire_cd = self.reload_t = self.reload_full = 0.0
        self.use_t = self.use_full = 0.0
        self.use_slot = -1
        self.bloom = 0.0
        self.crouch = self.sprint = self.ads = False
        self.kills = 0
        self.walk = 0.0
        self.swing_t = self.hurt_t = self.attacked_t = self.shot_t = self.land_t = -9.0
        self.attacker: Actor | None = None
        self.storm_acc = 0.0
        self.place = 0
        self.brain = None
        self.wish_x = self.wish_z = 0.0
        self.want_jump = self.firing = False
        self.aim: tuple | None = None
        self.build_mat = WOOD
        self.damage = 0.0
        self.swim = False
        self.speed = 0.0
        self.died_t = -1.0
        self.stats = {"built": 0, "harvested": 0, "chests": 0, "headshots": 0, "longest": 0.0, "shots": 0}
        self.last_dir = (0.0, 0.0, 1.0)

    # --- what is in the hands ---------------------------------------------------------------------------------------

    def held(self) -> Slot | None:
        return self.inv[self.sel] if 0 <= self.sel < SLOTS else None

    def weapon(self):  # noqa: ANN201 - the Weapon in the hands (the sledge when none)
        s = self.held()
        if s is not None and s.is_weapon:
            return WEAPONS[s.id]
        return WEAPONS["sledge"] if s is None else None

    def eye(self) -> tuple[float, float, float]:
        return self.x, self.y + (EYE - 0.5 if self.crouch else EYE), self.z

    def head_h(self) -> float:
        return 1.3 if self.crouch else HEIGHT

    def has_gun(self) -> bool:
        for s in self.inv:
            if s is not None and s.is_weapon and (s.mag > 0 or self.ammo[WEAPONS[s.id].ammo] > 0):
                return True
        return False


class Projectile:
    __slots__ = ("x", "y", "z", "vx", "vy", "vz", "owner", "weapon", "grade", "life", "drop", "ox", "oy", "oz")


class Match:
    def __init__(self, island: Island, options: dict, outfit: int = 0, canopy: int = 0, name: str = "You",
                 seed: int | None = None, autoplay: bool = False) -> None:
        self.island = island
        for p in island.props:  # the island is shared between matches: stand everything back up
            p.alive, p.hp, p.hit_t, p.slots = True, p.hp_max, -9.0, []
        self.rng = random.Random(seed)
        self.options = options
        self.structs = Structures(island)
        self.time = 0.0
        self.diff = DIFFICULTY.get(options.get("difficulty", "normal"), DIFFICULTY["normal"])
        # what the widget reads back
        self.sounds: list[tuple[str, float]] = []
        self.fx: list[tuple] = []  # ("burst", x, y, z, colour, n, speed) / ("boom", x, y, z, r) / ("poof", ...)
        self.tracers: list[list] = []
        self.numbers: list[list] = []  # damage numbers: [x, y, z, amount, head, shield, age]
        self.feed: list[tuple] = []  # (time, text, colour)
        self.banner: tuple[float, str, str] | None = None  # (until, text, colour)
        self.hit_marker = (-9.0, False)
        self.item_events: list[tuple[str, Item]] = []
        self.chest_events: list[Chest] = []
        self.prop_events: list = []
        self.projectiles: list[Projectile] = []
        self.listener = (SIZE / 2, 0.0, SIZE / 2)
        # everyone
        rng = self.rng
        n_bots = int(min(max(options.get("bots", 24), 1), 40))
        self.actors: list[Actor] = []
        self.player = Actor(0, name, autoplay, outfit, canopy)
        self.actors.append(self.player)
        names: set[str] = set()
        from .bots import Brain

        for i in range(n_bots):
            while True:
                nm = rng.choice(BOT_FIRST) + rng.choice(BOT_LAST)
                if nm not in names:
                    names.add(nm)
                    break
            a = Actor(i + 1, nm, True, rng.randrange(len(OUTFITS)), rng.randrange(6))
            a.brain = Brain(self, a)
            self.actors.append(a)
        if autoplay:
            self.player.brain = Brain(self, self.player)
        self.alive = len(self.actors)
        self.total = self.alive
        self.over = False
        self.winner: Actor | None = None
        self.over_t = -1.0
        # the ship: a straight line over the island
        ang = rng.uniform(0, math.tau)
        off = rng.uniform(-60, 60)
        c = SIZE / 2
        dx, dz = math.sin(ang), math.cos(ang)
        px, pz = c + dz * off, c - dx * off
        self.ship_a = (px - dx * 330, pz - dz * 330)
        self.ship_b = (px + dx * 330, pz + dz * 330)
        self.ship_dir = (dx, dz)
        self.ship_yaw = ang
        self.ship_len = 660.0
        self.ship_s = 0.0  # metres flown
        self.ship_open = 62.0  # doors open once over the coast
        self.ship_end = 470.0  # everyone still aboard is put out here, over the far shore
        self.drop_claims: list[tuple[float, float]] = []
        for a in self.actors:
            a.yaw = ang
            a.x, a.z = self.ship_a
            a.y = SHIP_ALT
            if a.brain is not None:
                a.brain.plan_drop()
        # loot
        self.items: list[Item] = []
        self.buckets: dict[tuple, list[Item]] = {}
        self.chests: list[Chest] = []
        for x, y, z, yaw in island.chest_spots:
            self.chests.append(Chest(x, y, z, yaw))
        for x, y, z in island.ammo_spots:
            self.chests.append(Chest(x, y, z, rng.uniform(0, math.tau), small=True))
        for x, y, z in island.floor_spots:
            self._floor_find(x, y, z)
        # the storm
        self.storm = {"cx": c, "cz": c, "r": STORM_START_R, "ncx": c, "ncz": c, "nr": STORM_START_R, "phase": -1,
                      "state": "idle", "t": 0.0, "dps": 1.0, "fx": c, "fz": c, "fr": STORM_START_R, "total": 1.0}
        self._storm_delay = 26.0  # s after the doors open before the first circle shows
        self.focus = None  # what the player would interact with: a Chest or an Item
        self.truce = False  # bots are leaving each other alone for now (see ``_pace``)
        self._focus_t = 0.0

    # --- small helpers ------------------------------------------------------------------------------------------------

    def sound(self, name: str, x: float | None = None, y: float = 0.0, z: float = 0.0, volume: float = 1.0) -> None:
        if x is None:
            self.sounds.append((name, volume))
            return
        lx, ly, lz = self.listener
        d = math.sqrt((x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2)
        if d < HEAR:
            self.sounds.append((name, volume * max(0.06, (1.0 - d / HEAR) ** 2)))

    def say(self, text: str, colour: str = "#ffffff") -> None:
        self.feed.append((self.time, text, colour))
        if len(self.feed) > 40:
            del self.feed[:-40]

    def show_banner(self, text: str, colour: str = "#ffffff", secs: float = 4.0) -> None:
        self.banner = (self.time + secs, text, colour)

    def ground_at(self, x: float, z: float, feet: float) -> float:
        """The surface under a body whose feet are at ``feet``: ground, a floor / ramp / roof, the top of a prop."""
        g = self.island.height(x, z)
        s = self.structs.support(x, z, feet + STEP_UP)
        if s is not None and s > g:
            g = s
        plist = self.island.prop_cols.get((int(x * 0.25), int(z * 0.25)))
        if plist:
            lim = feet + STEP_UP
            for p in plist:
                if p.alive:
                    top = p.y + p.h
                    if g < top <= lim:
                        if p.shape == "cyl":
                            if (x - p.x) ** 2 + (z - p.z) ** 2 < p.a * p.a:
                                g = top
                        elif abs(x - p.x) < p.a and abs(z - p.z) < p.b:
                            g = top
        return g

    def storm_dist(self, x: float, z: float) -> float:
        """How far outside the storm's eye a point is (negative inside)."""
        s = self.storm
        return math.hypot(x - s["cx"], z - s["cz"]) - s["r"]

    # --- loot ---------------------------------------------------------------------------------------------------------

    def _roll_grade(self, weights: tuple, floor: int = 0) -> int:
        g = self.rng.choices(range(5), weights)[0]
        return max(g, floor)

    def _roll_weapon(self, weights: tuple) -> tuple[str, int]:
        wid = self.rng.choices(GUNS, [WEAPONS[w].weight for w in GUNS])[0]
        return wid, self._roll_grade(weights, WEAPONS[wid].min_grade)

    def add_item(self, kind: str, id_, grade: int, count: int, x: float, y: float, z: float, mag: int = 0,
                 no_auto: float = -1.0) -> Item:
        x = min(max(x, 3.0), SIZE - 3.0)
        z = min(max(z, 3.0), SIZE - 3.0)
        it = Item(kind, id_, grade, count, x, max(self.ground_at(x, z, y + 0.9), SEA - 0.2) + 0.02, z, mag)
        it.no_auto, it.born = no_auto, self.time
        self.items.append(it)
        it.bucket = (int(x) >> 3, int(z) >> 3)
        self.buckets.setdefault(it.bucket, []).append(it)
        self.item_events.append(("add", it))
        return it

    def remove_item(self, it: Item) -> None:
        if not it.alive:
            return
        it.alive = False
        b = self.buckets.get(it.bucket)
        if b is not None and it in b:
            b.remove(it)
        self.item_events.append(("del", it))

    def items_near(self, x: float, z: float, r: float) -> list[Item]:
        out = []
        bx0, bx1 = int(x - r) >> 3, int(x + r) >> 3
        bz0, bz1 = int(z - r) >> 3, int(z + r) >> 3
        r2 = r * r
        for bx in range(bx0, bx1 + 1):
            for bz in range(bz0, bz1 + 1):
                for it in self.buckets.get((bx, bz), ()):
                    if (it.x - x) ** 2 + (it.z - z) ** 2 < r2:
                        out.append(it)
        return out

    def _weapon_item(self, wid: str, grade: int, x: float, y: float, z: float) -> None:
        w = WEAPONS[wid]
        self.add_item("weapon", wid, grade, 1, x, y, z, mag=w.mag)

    def _floor_find(self, x: float, y: float, z: float) -> None:
        rng = self.rng
        roll = rng.random()
        if roll < 0.56:
            wid, grade = self._roll_weapon(GRADE_WEIGHTS)
            self._weapon_item(wid, grade, x, y, z)
            self.add_item("ammo", WEAPONS[wid].ammo, 0, AMMO_PICKUP[WEAPONS[wid].ammo], x + 0.8, y, z + 0.4)
        elif roll < 0.80:
            sid = rng.choices(list(SUPPLIES), [s.weight for s in SUPPLIES.values()])[0]
            self.add_item("supply", sid, 0, SUPPLY_FOUND[sid], x, y, z)
        elif roll < 0.92:
            for i, am in enumerate(rng.sample(AMMO[:4], 2)):
                self.add_item("ammo", am, 0, AMMO_PICKUP[am], x + i * 0.8, y, z)
        else:
            self.add_item("mats", rng.randrange(3), 0, 30, x, y, z)

    def open_chest(self, c: Chest, who: Actor | None = None) -> None:
        if c.opened:
            return
        c.opened = True
        self.chest_events.append(c)
        rng = self.rng
        fx, fz = math.sin(c.yaw), math.cos(c.yaw)
        rx, rz = fz, -fx
        drops: list[tuple] = []
        if c.small:
            for am in rng.sample(AMMO[:4], 2):
                drops.append(("ammo", am, 0, AMMO_PICKUP[am], 0))
            if rng.random() < 0.25:
                drops.append(("ammo", "rockets", 0, 2, 0))
            self.sound("ammo_box", c.x, c.y, c.z)
        else:
            wid, grade = self._roll_weapon(CHEST_GRADE_WEIGHTS)
            w = WEAPONS[wid]
            drops.append(("weapon", wid, grade, 1, w.mag))
            drops.append(("ammo", w.ammo, 0, AMMO_PICKUP[w.ammo] * 2, 0))
            sid = rng.choices(list(SUPPLIES), [s.weight + (6 if s.heals == "shield" else 0)
                                               for s in SUPPLIES.values()])[0]
            drops.append(("supply", sid, 0, SUPPLY_FOUND[sid], 0))
            drops.append(("mats", rng.randrange(3), 0, 30, 0))
            if rng.random() < 0.3:
                sid = rng.choice(list(SUPPLIES))
                drops.append(("supply", sid, 0, SUPPLY_FOUND[sid], 0))
            self.sound("chest", c.x, c.y, c.z)
            self.fx.append(("burst", c.x, c.y + 0.8, c.z, (1.0, 0.85, 0.3), 14, 3.0))
        n = len(drops)
        for i, (kind, id_, grade, count, mag) in enumerate(drops):
            side = (i - (n - 1) / 2) * 0.75
            self.add_item(kind, id_, grade, count, c.x + fx * 1.25 + rx * side, c.y, c.z + fz * 1.25 + rz * side,
                          mag=mag)
        if who is not None:
            who.stats["chests"] += 1

    def give(self, a: Actor, it: Item, swap: bool = False) -> bool:
        """``a`` takes ``it`` if there is room (``swap``: the held slot is dropped for it when there is not)."""
        if not it.alive:
            return False
        if it.kind == "ammo":
            room = AMMO_CAP[it.id] - a.ammo[it.id]
            if room <= 0:
                return False
            take = min(room, it.count)
            a.ammo[it.id] += take
            it.count -= take
            if it.count <= 0:
                self.remove_item(it)
            return True
        if it.kind == "mats":
            room = MAT_CAP - a.mats[it.id]
            if room <= 0:
                return False
            a.mats[it.id] += min(room, it.count)
            self.remove_item(it)
            return True
        if it.kind == "supply":
            sup = SUPPLIES[it.id]
            for s in a.inv:
                if s is not None and s.id == it.id and s.count < sup.stack:
                    take = min(sup.stack - s.count, it.count)
                    s.count += take
                    it.count -= take
                    if it.count <= 0:
                        self.remove_item(it)
                        return True
        free = next((i for i, s in enumerate(a.inv) if s is None), -1)
        if free < 0:
            if not swap:
                return False
            free = a.sel if a.sel >= 0 else SLOTS - 1
            self.drop_slot(a, free)
        a.inv[free] = Slot(it.id, it.grade, it.mag, it.count)
        if a.sel < 0 and it.kind == "weapon" and not a.bot:
            self.select(a, free)
        self.remove_item(it)
        return True

    def drop_slot(self, a: Actor, i: int) -> None:
        s = a.inv[i]
        if s is None:
            return
        a.inv[i] = None
        fx, fz = math.sin(a.yaw), math.cos(a.yaw)
        self.add_item("weapon" if s.is_weapon else "supply", s.id, s.grade, s.count, a.x + fx * 1.2, a.y,
                      a.z + fz * 1.2, mag=s.mag, no_auto=self.time + 4.0)
        if a.sel == i:
            a.reload_t = a.use_t = 0.0

    def select(self, a: Actor, i: int) -> None:
        if i == a.sel:
            return
        a.sel = i
        a.reload_t = a.use_t = 0.0
        a.fire_cd = max(a.fire_cd, 0.22)
        a.bloom = 0.0
        if not a.bot:
            self.sound("switch")

    # --- the player's verbs (bots call the same ones) ---------------------------------------------------------------------

    def interact(self, a: Actor) -> bool:
        target = self.focus if a is self.player and not a.bot else None
        if target is None:
            return False
        if isinstance(target, Chest):
            self.open_chest(target, a)
            return True
        if self.give(a, target, swap=True):
            self.sound("pickup_weapon" if target.kind == "weapon" else "pickup")
            return True
        return False

    def reload(self, a: Actor) -> bool:
        s = a.held()
        if s is None or not s.is_weapon or a.reload_t > 0:
            return False
        w = WEAPONS[s.id]
        if s.mag >= w.mag or a.ammo[w.ammo] <= 0:
            return False
        a.reload_t = a.reload_full = w.reload
        a.use_t = 0.0
        self.sound("reload", a.x, a.y, a.z, 0.8)
        return True

    def use(self, a: Actor) -> bool:
        """Start using the supply in the hands."""
        s = a.held()
        if s is None or s.is_weapon or a.use_t > 0 or a.state != "ground":
            return False
        sup = SUPPLIES[s.id]
        cur = a.hp if sup.heals == "hp" else a.shield
        if cur >= sup.cap:
            return False
        a.use_t = a.use_full = sup.time
        a.use_slot = a.sel
        self.sound("use_start", a.x, a.y, a.z, 0.7)
        return True

    def _finish_use(self, a: Actor) -> None:
        s = a.inv[a.use_slot] if 0 <= a.use_slot < SLOTS else None
        if s is None or s.is_weapon:
            return
        sup = SUPPLIES[s.id]
        if sup.heals == "hp":
            a.hp = min(max(a.hp, min(a.hp + sup.amount, sup.cap)), MAX_HP)
        else:
            a.shield = min(max(a.shield, min(a.shield + sup.amount, sup.cap)), MAX_SHIELD)
        s.count -= 1
        if s.count <= 0:
            a.inv[a.use_slot] = None
            if a.sel == a.use_slot and not a.bot:
                nxt = next((i for i, q in enumerate(a.inv) if q is not None and q.is_weapon), -1)
                a.sel = nxt
        self.sound("heal" if sup.heals == "hp" else "shield", a.x, a.y, a.z)
        self.fx.append(("burst", a.x, a.y + 1.0, a.z, (0.4, 1.0, 0.5) if sup.heals == "hp" else (0.35, 0.75, 1.0),
                        10, 2.0))

    def build_target(self, a: Actor, piece: int, yaw: float, pitch: float) -> tuple[int, int, int, int, int]:
        """Where the build piece (0 wall, 1 floor, 2 ramp, 3 roof) would go for someone standing at ``a`` looking
        along yaw / pitch: (kind, gx, gy, gz, dir)."""
        fx, fz = math.sin(yaw), math.cos(yaw)
        east_west = abs(fx) > abs(fz)
        lev = int((a.y + 1.2) // LEVEL)
        cx, cz = int(a.x // CELL), int(a.z // CELL)
        if piece == 0:
            gy = lev + (1 if pitch > 0.62 else (-1 if pitch < -1.0 else 0))
            if east_west:
                gxw = int((a.x + 0.55) // CELL) + 1 if fx > 0 else int(math.ceil((a.x - 0.55) / CELL)) - 1
                return WALLZ, gxw, gy, cz, 0
            gzw = int((a.z + 0.55) // CELL) + 1 if fz > 0 else int(math.ceil((a.z - 0.55) / CELL)) - 1
            return WALLX, cx, gy, gzw, 0
        reach = 2.6 * max(math.cos(pitch), 0.0) if pitch < 0.9 else 0.0
        if pitch < -1.05:
            reach = 0.0
        tx, tz = int((a.x + fx * reach) // CELL), int((a.z + fz * reach) // CELL)
        if piece == 1:
            gy = int(round(a.y / LEVEL)) + (1 if pitch > 0.5 else 0)
            return FLOOR, tx, gy, tz, 0
        if piece == 2:
            d = (1 if fx > 0 else 3) if east_west else (0 if fz > 0 else 2)
            gy = int((a.y + STEP_UP) // LEVEL) + (1 if pitch > 0.75 else 0)
            return RAMP, tx, gy, tz, d
        return ROOF, tx, lev + (0 if pitch < -0.35 else 1), tz, 0

    def try_build(self, a: Actor, kind: int, gx: int, gy: int, gz: int, d: int, mat: int | None = None) -> bool:
        mat = a.build_mat if mat is None else mat
        if a.mats[mat] < BUILD_COST:
            alt = next((m for m in range(3) if a.mats[m] >= BUILD_COST), None)
            if alt is None or not a.bot:
                return False
            mat = alt
        if a.state != "ground" or not self.structs.can_place(kind, gx, gy, gz, d):
            return False
        p = self.structs.place(kind, gx, gy, gz, d, mat, owner=a.id, check=False)
        if p is None:
            return False
        a.mats[mat] -= BUILD_COST
        a.stats["built"] += 1
        px, py, pz = p.centre()
        self.sound(("build_wood", "build_brick", "build_metal")[mat], px, py, pz)
        return True

    # --- hurting ------------------------------------------------------------------------------------------------------------

    def hurt(self, v: Actor, amount: float, by: Actor | None, cause: str, head: bool = False,
             direct: bool = False, dist: float = 0.0) -> None:
        if not v.alive or v.state == "ship" or amount <= 0:
            return
        if by is not None and by.bot and by is not v and not direct:
            amount *= (BOT_VS_BOT * (0.2 if self.truce else 1.0)) if v.bot else self.diff[2]
        took_shield = 0.0
        if not direct:
            took_shield = min(v.shield, amount)
            v.shield -= took_shield
        v.hp -= amount - took_shield
        v.hurt_t = self.time
        if by is not None and by is not v:
            v.attacker, v.attacked_t = by, self.time
            by.damage += amount
            if by is self.player:
                self.hit_marker = (self.time, head)
                self.numbers.append([v.x, v.y + 1.9, v.z, amount, head, took_shield >= amount * 0.5, 0.0])
                self.sound("headshot" if head else "hit", volume=0.8)
                if head:
                    by.stats["headshots"] += 1
        if v is self.player and not direct:
            self.sound("hurt", volume=0.9)
        if v.hp <= 0.0:
            self.eliminate(v, by if by is not v else None, cause, dist)

    def eliminate(self, v: Actor, by: Actor | None, cause: str, dist: float = 0.0) -> None:
        if not v.alive:
            return
        v.alive = False
        v.hp = 0.0
        v.died_t = self.time
        v.place = self.alive
        self.alive -= 1
        if by is None and v.attacker is not None and self.time - v.attacked_t < 6.0 and v.attacker.alive:
            by = v.attacker  # finished off by the storm or a fall: the last one to hit them takes it
            cause = cause + "+"
        if by is not None:
            by.kills += 1
            if dist > by.stats["longest"]:
                by.stats["longest"] = dist
        me = self.player
        col = "#ffd166" if (v is me or by is me) else "#e9eef5"
        mine = v is me and not me.bot
        vn = "you" if mine else v.name
        if cause.startswith("storm"):
            text = (("You were" if mine else f"{v.name} was") + " lost to the storm") if by is None                 else f"{by.name} left {vn} to the storm"
        elif cause.startswith("fall"):
            text = (("You" if mine else v.name) + " fell too far") if by is None else f"{by.name} knocked {vn} off"
        elif by is None:
            text = ("You are" if mine else f"{v.name} is") + " out"
        else:
            wname = WEAPONS[cause.rstrip("+")].name if cause.rstrip("+") in WEAPONS else cause
            text = f"{by.name} eliminated {vn} with {wname}" + (f" ({dist:.0f} m)" if dist >= 30 else "")
        self.say(text, col)
        self.sound("elim" if by is me else "elim_far", v.x, v.y, v.z)
        self.fx.append(("poof", v.x, v.y + 1.0, v.z, OUTFITS[v.outfit].suit, 26, 5.0))
        # what they carried stays on the ground
        rng = self.rng
        k = 0
        for s in v.inv:
            if s is not None:
                ang = k * 1.26
                self.add_item("weapon" if s.is_weapon else "supply", s.id, s.grade, s.count,
                              v.x + math.cos(ang) * 1.0, v.y, v.z + math.sin(ang) * 1.0, mag=s.mag)
                k += 1
        v.inv = [None] * SLOTS
        for am, n in sorted(v.ammo.items(), key=lambda kv: -kv[1])[:3]:
            if n > 0:
                ang = rng.uniform(0, math.tau)
                self.add_item("ammo", am, 0, n, v.x + math.cos(ang) * 1.6, v.y, v.z + math.sin(ang) * 1.6)
        for m in range(3):
            if v.mats[m] > 0:
                ang = rng.uniform(0, math.tau)
                self.add_item("mats", m, 0, v.mats[m], v.x + math.cos(ang) * 1.9, v.y, v.z + math.sin(ang) * 1.9)
        v.mats = [0, 0, 0]
        if self.alive <= 1 and not self.over:
            self.over = True
            self.over_t = self.time
            self.winner = next((a for a in self.actors if a.alive), by or v)
            self.winner.place = 1
            won = self.winner is me and not me.bot
            self.say(("You are" if won else f"{self.winner.name} is") + " the last one standing!", "#ffd166")

    # --- shooting -------------------------------------------------------------------------------------------------------------

    def _ray_actors(self, ox: float, oy: float, oz: float, dx: float, dy: float, dz: float, maxd: float,
                    skip: Actor | None) -> tuple[Actor | None, float, bool]:
        best, bt, bhead = None, maxd, False
        for b in self.actors:
            if not b.alive or b is skip or b.state == "ship":
                continue
            vx, vy, vz = b.x - ox, b.y + 0.9 - oy, b.z - oz
            tp = vx * dx + vy * dy + vz * dz
            if tp < -1.0 or tp > bt + 1.5:
                continue
            if vx * vx + vy * vy + vz * vz - tp * tp > 1.7:
                continue
            body_top = b.y + (1.02 if b.crouch else 1.45)
            # the head: a ball
            hy = body_top + 0.2
            fx, fy, fz = ox - b.x, oy - hy, oz - b.z
            bq = fx * dx + fy * dy + fz * dz
            cq = fx * fx + fy * fy + fz * fz - 0.075
            disc = bq * bq - cq
            if disc >= 0:
                t = -bq - math.sqrt(disc)
                if 0 <= t < bt:
                    best, bt, bhead = b, t, True
            # the body: an upright cylinder
            a2 = dx * dx + dz * dz
            if a2 > 1e-9:
                fx, fz = ox - b.x, oz - b.z
                bq = fx * dx + fz * dz
                cq = fx * fx + fz * fz - 0.185
                disc = bq * bq - a2 * cq
                if disc >= 0:
                    t = (-bq - math.sqrt(disc)) / a2
                    if 0 <= t < bt:
                        y = oy + dy * t
                        if b.y <= y <= body_top:
                            best, bt, bhead = b, t, False
        return best, bt, bhead

    def _hit_world(self, a: Actor, what: str | None, obj, w, x: float, y: float, z: float) -> None:  # noqa: ANN001
        if what == "piece":
            self.structs.damage(obj, w.struct)
            self.fx.append(("burst", x, y, z, _MAT_FX[obj.mat], 4, 2.5))
            self.sound(("thud_wood", "thud_brick", "thud_metal")[obj.mat], x, y, z, 0.55)
        elif what == "prop":
            self._hit_prop(a, obj, w.struct, False)
            self.fx.append(("burst", x, y, z, _MAT_FX[obj.mat], 4, 2.5))
            self.sound(("thud_wood", "thud_brick", "thud_metal")[obj.mat], x, y, z, 0.55)
        elif what == "ground":
            self.fx.append(("burst", x, y + 0.05, z, (0.75, 0.68, 0.5) if y > SEA + 0.05 else (0.8, 0.95, 1.0), 3,
                            2.0))

    def _hit_prop(self, a: Actor | None, p, amount: float, harvest: bool) -> None:  # noqa: ANN001
        if not p.alive:
            return
        p.hp -= amount
        p.hit_t = self.time
        if harvest and a is not None:
            gain = 8 if p.mat == WOOD else 7
            if p.hp <= 0:
                gain += 10
            a.mats[p.mat] = min(a.mats[p.mat] + gain, MAT_CAP)
            a.stats["harvested"] += gain
        if p.hp <= 0:
            p.alive = False
            self.fx.append(("burst", p.x, p.y + min(p.h, 3.0) * 0.5, p.z, _MAT_FX[p.mat], 18, 5.0))
            self.sound(("break_wood", "break_brick", "break_metal")[p.mat], p.x, p.y, p.z)
        self.prop_events.append(p)

    def muzzle(self, a: Actor) -> tuple[float, float, float]:
        fx, fz = math.sin(a.yaw), math.cos(a.yaw)
        return a.x + fz * 0.28 + fx * 0.45, a.y + (0.95 if a.crouch else 1.38), a.z - fx * 0.28 + fz * 0.45

    def fire(self, a: Actor) -> bool:
        """One pull of the trigger (or swing of the sledge) if the weapon is ready."""
        if a.fire_cd > 0 or a.reload_t > 0 or a.use_t > 0 or a.state != "ground":
            return False
        s = a.held()
        if s is not None and not s.is_weapon:
            return self.use(a)
        w = WEAPONS[s.id] if s is not None else WEAPONS["sledge"]
        ex, ey, ez = a.eye()
        if a.aim is not None:
            dx, dy, dz = a.aim[0] - ex, a.aim[1] - ey, a.aim[2] - ez
            d = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            dx, dy, dz = dx / d, dy / d, dz / d
        else:
            cp = math.cos(a.pitch)
            dx, dy, dz = math.sin(a.yaw) * cp, math.sin(a.pitch), math.cos(a.yaw) * cp
        a.last_dir = (dx, dy, dz)
        a.fire_cd = 1.0 / w.rate
        if w.cls == "tool":
            a.swing_t = self.time
            self.sound("swing", a.x, a.y, a.z, 0.6)
            t, what, obj = self.structs.ray(ex, ey, ez, dx, dy, dz, w.reach + 0.4)
            v, tv, _head = self._ray_actors(ex, ey, ez, dx, dy, dz, min(t, w.reach + 0.4), a)
            if v is not None:
                self.hurt(v, w.damage, a, "sledge", dist=tv)
                self.fx.append(("burst", v.x, v.y + 1.1, v.z, (1.0, 0.9, 0.5), 5, 3.0))
            elif what == "piece":
                x, y, z = ex + dx * t, ey + dy * t, ez + dz * t
                mat = obj.mat
                gain = 6 + (4 if obj.hp <= w.struct else 0)
                self.structs.damage(obj, w.struct)
                a.mats[mat] = min(a.mats[mat] + gain, MAT_CAP)
                a.stats["harvested"] += gain
                self.fx.append(("burst", x, y, z, _MAT_FX[mat], 7, 3.5))
                self.sound(("chop_wood", "chop_brick", "chop_metal")[mat], x, y, z)
            elif what == "prop":
                x, y, z = ex + dx * t, ey + dy * t, ez + dz * t
                self._hit_prop(a, obj, w.struct, True)
                self.fx.append(("burst", x, y, z, _MAT_FX[obj.mat], 7, 3.5))
                self.sound(("chop_wood", "chop_brick", "chop_metal")[obj.mat], x, y, z)
            return True
        if s.mag <= 0:
            if not self.reload(a):
                a.fire_cd = 0.3
                if a is self.player and not a.bot:
                    self.sound("empty")
            return False
        s.mag -= 1
        a.shot_t = self.time
        a.stats["shots"] += 1
        mx, my, mz = self.muzzle(a)
        self.sound(w.sound, mx, my, mz)
        spread = w.spread + a.bloom
        if w.pellets == 1:
            spread += min(a.speed / SPRINT, 1.2) * 0.014 + (0.0 if a.on_ground else 0.02)
            if a.crouch:
                spread *= 0.72
            if a.ads:
                spread *= 0.5 if w.zoom < 2.5 else 0.05
            elif w.zoom > 2.5:
                spread += 0.05  # a scoped rifle from the hip
        a.bloom = min(a.bloom + w.bloom, w.bloom_max)
        grade = s.grade
        rng = self.rng
        # the shot leaves the muzzle towards where the eye is looking
        if a.aim is not None:
            ax, ay, az = a.aim
        else:
            ax, ay, az = ex + dx * 60.0, ey + dy * 60.0, ez + dz * 60.0
        bx, by, bz = ax - mx, ay - my, az - mz
        d = math.sqrt(bx * bx + by * by + bz * bz) or 1.0
        if d < 2.0:  # something right against the muzzle: shoot from the eye instead
            mx, my, mz = ex, ey, ez
            bx, by, bz, d = dx, dy, dz, 1.0
        bx, by, bz = bx / d, by / d, bz / d
        for _ in range(w.pellets):
            px, py, pz = bx, by, bz
            if spread > 0:
                ang, rad = rng.uniform(0, math.tau), spread * math.sqrt(rng.random())
                # two directions across the shot
                ux, uy, uz = -bz, 0.0, bx
                ul = math.sqrt(ux * ux + uz * uz) or 1.0
                ux, uz = ux / ul, uz / ul
                vx, vy, vz = by * uz - bz * uy, bz * ux - bx * uz, bx * uy - by * ux
                ca, sa = math.cos(ang) * rad, math.sin(ang) * rad
                px, py, pz = bx + ux * ca + vx * sa, by + uy * ca + vy * sa, bz + uz * ca + vz * sa
                n = math.sqrt(px * px + py * py + pz * pz)
                px, py, pz = px / n, py / n, pz / n
            if w.speed > 0:
                p = Projectile()
                p.x, p.y, p.z = mx, my, mz
                p.ox, p.oy, p.oz = mx, my, mz
                p.vx, p.vy, p.vz = px * w.speed, py * w.speed, pz * w.speed
                p.owner, p.weapon, p.grade, p.life, p.drop = a, w, grade, w.reach / w.speed + 1.0, w.drop
                self.projectiles.append(p)
                continue
            t, what, obj = self.structs.ray(mx, my, mz, px, py, pz, w.reach)
            v, tv, head = self._ray_actors(mx, my, mz, px, py, pz, t, a)
            if v is not None:
                t = tv
                fall = 1.0 if t <= w.near else (w.floor if t >= w.far else
                                                1.0 - (1.0 - w.floor) * (t - w.near) / (w.far - w.near))
                self.hurt(v, w.damage * GRADE_DAMAGE[grade] * fall * (w.head if head else 1.0), a, w.id, head,
                          dist=t)
            else:
                self._hit_world(a, what, obj, w, mx + px * t, my + py * t, mz + pz * t)
            self.tracers.append([mx, my, mz, mx + px * t, my + py * t, mz + pz * t, 0.07, 0])
        return True

    def _explode(self, p: Projectile, x: float, y: float, z: float, direct: Actor | None) -> None:
        w = p.weapon
        self.fx.append(("boom", x, y, z, w.blast))
        self.sound("boom", x, y, z, 1.0)
        self.structs.blast(x, y, z, w.blast, w.struct)
        r = w.blast
        for pr in list(self.island.props):
            if pr.alive and abs(pr.x - x) < r + 2 and abs(pr.z - z) < r + 2 and abs(pr.y - y) < r + 6:
                if math.hypot(pr.x - x, pr.z - z) < r + max(pr.a, pr.b):
                    self._hit_prop(p.owner, pr, w.struct * 0.6, False)
        dmg = w.damage * GRADE_DAMAGE[p.grade]
        for b in self.actors:
            if not b.alive or b.state == "ship":
                continue
            d = math.sqrt((b.x - x) ** 2 + (b.y + 0.9 - y) ** 2 + (b.z - z) ** 2)
            if d < r + 0.5:
                if b is not direct and not self.structs.clear_line(x, y + 0.3, z, b.x, b.y + 1.0, b.z):
                    continue  # behind cover
                amount = dmg * (1.0 if b is direct else max(0.25, 1.0 - d / (r + 0.5)))
                if b is p.owner:
                    amount *= 0.5
                self.hurt(b, amount, p.owner, w.id, dist=math.hypot(p.ox - x, p.oz - z))

    def _projectiles(self, dt: float) -> None:
        keep = []
        for p in self.projectiles:
            p.life -= dt
            p.vy -= p.drop * dt
            sp = math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz)
            step = sp * dt
            dx, dy, dz = p.vx / sp, p.vy / sp, p.vz / sp
            t, what, obj = self.structs.ray(p.x, p.y, p.z, dx, dy, dz, step)
            v, tv, head = self._ray_actors(p.x, p.y, p.z, dx, dy, dz, t if what else step, p.owner)
            w = p.weapon
            if v is not None:
                x, y, z = p.x + dx * tv, p.y + dy * tv, p.z + dz * tv
                if w.blast > 0:
                    self._explode(p, x, y, z, v)
                else:
                    dist = math.sqrt((x - p.ox) ** 2 + (y - p.oy) ** 2 + (z - p.oz) ** 2)
                    self.hurt(v, w.damage * GRADE_DAMAGE[p.grade] * (w.head if head else 1.0), p.owner, w.id, head,
                              dist=dist)
                continue
            if what is not None:
                x, y, z = p.x + dx * t, p.y + dy * t, p.z + dz * t
                if w.blast > 0:
                    self._explode(p, x - dx * 0.15, y - dy * 0.15, z - dz * 0.15, None)
                else:
                    self._hit_world(p.owner, what, obj, w, x, y, z)
                continue
            p.x += dx * step
            p.y += dy * step
            p.z += dz * step
            if p.life > 0 and 0 < p.x < SIZE and 0 < p.z < SIZE and p.y > -2:
                keep.append(p)
                if w.blast > 0:
                    self.fx.append(("smoke", p.x, p.y, p.z))
            elif w.blast > 0 and p.y <= SEA:
                self._explode(p, p.x, SEA, p.z, None)
        self.projectiles = keep

    # --- movement -------------------------------------------------------------------------------------------------------------

    def leave_ship(self, a: Actor) -> None:
        if a.state != "ship" or self.ship_s < self.ship_open:
            return
        a.state = "fall"
        a.vx, a.vz = self.ship_dir[0] * 8.0, self.ship_dir[1] * 8.0
        a.vy = -6.0
        a.y = SHIP_ALT - 4.0
        if a is self.player:
            self.sound("jump_out")

    def _air(self, a: Actor, dt: float) -> None:
        """Falling from the ship, then under the canopy."""
        wx, wz = a.wish_x, a.wish_z
        if a.state == "fall":
            dive = max(-math.sin(a.pitch), 0.0) if (wx or wz) else 0.0
            glide = DIVE_GLIDE * (1.0 - 0.55 * dive)
            target_vy = -(DIVE_V + 16.0 * dive)
            k = min(1.0, 2.2 * dt)
        else:
            glide, target_vy, k = CHUTE_GLIDE, -CHUTE_V, min(1.0, 3.5 * dt)
        a.vx += (wx * glide - a.vx) * k
        a.vz += (wz * glide - a.vz) * k
        a.vy += (target_vy - a.vy) * min(1.0, 3.0 * dt)
        nx = min(max(a.x + a.vx * dt, 4.0), SIZE - 4.0)
        nz = min(max(a.z + a.vz * dt, 4.0), SIZE - 4.0)
        ny = a.y + a.vy * dt
        feet_probe = ny + 1.0
        g = max(self.ground_at(nx, nz, feet_probe), SEA - SWIM_DEPTH)
        if a.state == "fall" and ny - g < CHUTE_ALT:
            a.state = "glide"
            self.sound("canopy", a.x, a.y, a.z)
        if a.state == "glide":
            nx, nz = self.structs.push(nx, nz, ny, ny + HEIGHT, RADIUS)
            plist = self.island.prop_cols.get((int(nx * 0.25), int(nz * 0.25)))
            if plist:
                nx, nz = _push_props(plist, nx, nz, ny, ny + HEIGHT)
            g = max(self.ground_at(nx, nz, feet_probe), SEA - SWIM_DEPTH)
        if ny <= g:
            ny = g
            a.state = "ground"
            a.on_ground = True
            a.vy = 0.0
            a.land_t = self.time
            self.sound("land", a.x, a.y, a.z)
        a.x, a.y, a.z = nx, ny, nz
        a.speed = math.hypot(a.vx, a.vz)

    def _move(self, a: Actor, dt: float) -> None:
        st = self.structs
        feet = a.y
        head_h = 1.3 if a.crouch else HEIGHT
        head = feet + head_h
        # something built through me: stand on it
        c = st.ceiling(a.x, a.z, feet + STEP_UP, head)
        stuck = c is not None
        if stuck and c - feet < 1.35:
            feet = a.y = c
            head = feet + head_h
            a.vy = 0.0
            a.on_ground = True
            stuck = False
        if a.swim:
            top = SWIM
        elif a.crouch:
            top = CROUCH
        elif a.sprint and not a.ads and a.use_t <= 0:
            top = SPRINT
        else:
            top = WALK
        if a.use_t > 0:
            top *= 0.6
        k = min(1.0, (13.0 if a.on_ground else 2.6) * dt)
        a.vx += (a.wish_x * top - a.vx) * k
        a.vz += (a.wish_z * top - a.vz) * k
        nx, nz = a.x + a.vx * dt, a.z + a.vz * dt
        nx, nz = st.push(nx, nz, feet, head, RADIUS)
        plist = self.island.prop_cols.get((int(nx * 0.25), int(nz * 0.25)))
        if plist:
            nx, nz = _push_props(plist, nx, nz, feet, head)
        if not stuck and st.ceiling(nx, nz, feet + STEP_UP, head) is not None:
            if st.ceiling(nx, a.z, feet + STEP_UP, head) is None:
                nz = a.z
                a.vz = 0.0
            elif st.ceiling(a.x, nz, feet + STEP_UP, head) is None:
                nx = a.x
                a.vx = 0.0
            else:
                nx, nz = a.x, a.z
                a.vx = a.vz = 0.0
        nx = min(max(nx, 3.0), SIZE - 3.0)
        nz = min(max(nz, 3.0), SIZE - 3.0)
        moved = math.hypot(nx - a.x, nz - a.z)
        a.speed = moved / dt if dt > 0 else 0.0
        a.x, a.z = nx, nz
        g = self.ground_at(nx, nz, feet)
        swim_y = SEA - SWIM_DEPTH
        a.swim = g < swim_y and feet <= swim_y + 0.3
        if g < swim_y:
            g = swim_y
        if a.on_ground:
            if a.want_jump and not a.swim and a.use_t <= 0:
                a.vy = JUMP_V
                a.on_ground = False
                if a is self.player:
                    self.sound("jump", volume=0.5)
            elif g >= feet - 0.8:
                a.y = g
                a.vy = 0.0
            else:
                a.on_ground = False
        if not a.on_ground:
            a.vy -= GRAVITY * dt
            ny = feet + a.vy * dt
            if a.vy > 0:
                c = st.ceiling(nx, nz, head - 0.05, ny + head_h)
                if c is not None:
                    ny = c - head_h - 0.02
                    a.vy = 0.0
            if ny <= g:
                impact = -a.vy
                ny = g
                a.on_ground = True
                a.vy = 0.0
                if impact > 7.0:
                    a.land_t = self.time
                    if a is self.player:
                        self.sound("land", volume=min(1.0, impact / 20.0))
                if impact > SAFE_FALL_V and not a.swim and g > swim_y + 0.01:
                    self.hurt(a, (impact - SAFE_FALL_V) * FALL_DMG, None, "fall", direct=True)
            a.y = ny
        a.walk += a.speed * dt

    # --- the storm ------------------------------------------------------------------------------------------------------------

    def _next_circle(self) -> None:
        s = self.storm
        i = s["phase"] + 1
        if i >= len(STORM_PHASES):
            s["state"] = "done"
            return
        wait, shrink, radius, dps = STORM_PHASES[i]
        s["phase"] = i
        s["state"] = "wait"
        s["t"] = s["total"] = wait
        s["dps"] = dps
        room = max(s["r"] - radius, 0.0) * (1.0 if i < 4 else 0.85)
        best = (s["cx"], s["cz"])
        for _ in range(40):
            ang, d = self.rng.uniform(0, math.tau), room * math.sqrt(self.rng.random())
            x, z = s["cx"] + math.cos(ang) * d, s["cz"] + math.sin(ang) * d
            if self.island.on_land(x, z) and self.island.height(x, z) > 1.5:
                best = (x, z)
                break
        s["ncx"], s["ncz"], s["nr"] = best[0], best[1], radius
        self.show_banner("The storm's eye is marked" if i == 0 else "A new eye is marked", "#d7a6ff")
        self.sound("storm_mark")

    def _storm(self, dt: float) -> None:
        s = self.storm
        if s["state"] == "idle":
            if self.ship_s >= self.ship_open:
                self._storm_delay -= dt
                if self._storm_delay <= 0:
                    self._next_circle()
            return
        if s["state"] == "wait":
            s["t"] -= dt
            if s["t"] <= 0:
                s["state"] = "shrink"
                s["t"] = s["total"] = STORM_PHASES[s["phase"]][1]
                s["fx"], s["fz"], s["fr"] = s["cx"], s["cz"], s["r"]
                self.show_banner("The storm is closing in", "#ff8ad8")
                self.sound("storm_move")
        elif s["state"] == "shrink":
            s["t"] -= dt
            k = 1.0 - max(s["t"], 0.0) / s["total"]
            s["cx"] = s["fx"] + (s["ncx"] - s["fx"]) * k
            s["cz"] = s["fz"] + (s["ncz"] - s["fz"]) * k
            s["r"] = s["fr"] + (s["nr"] - s["fr"]) * k
            if s["t"] <= 0:
                self._next_circle()
        dps = s["dps"]
        for a in self.actors:
            if a.alive and a.state != "ship" and math.hypot(a.x - s["cx"], a.z - s["cz"]) > s["r"]:
                a.storm_acc += dps * dt
                if a.storm_acc >= 1.0 and self.alive > 1:
                    n = int(a.storm_acc)
                    a.storm_acc -= n
                    self.hurt(a, n, None, "storm", direct=True)
                    if a is self.player:
                        self.sound("storm_tick", volume=0.6)

    def _pace(self) -> None:
        """The match's pace: when fewer players are left than this point of the match calls for, the bots stop picking
        fights with each other (never with the player) until the storm has caught up."""
        t = self.time
        want = 0.0
        prev_t, prev_f = 0.0, 1.0
        for pt, pf in PACE:
            if t < pt:
                want = (prev_f + (pf - prev_f) * (t - prev_t) / (pt - prev_t)) * self.total
                break
            prev_t, prev_f = pt, pf
        self.truce = self.alive <= want and self.alive > 2

    # --- one step ---------------------------------------------------------------------------------------------------------------

    def step(self, dt: float) -> None:
        self.time += dt
        self.structs.update(dt)
        # the ship
        if self.ship_s < self.ship_len:
            self.ship_s += SHIP_SPEED * dt
            sx = self.ship_a[0] + self.ship_dir[0] * self.ship_s
            sz = self.ship_a[1] + self.ship_dir[1] * self.ship_s
            eject = self.ship_s >= self.ship_end
            for a in self.actors:
                if a.state == "ship":
                    a.x, a.z, a.y = sx, sz, SHIP_ALT
                    if eject:
                        a.x = min(max(a.x, 30.0), SIZE - 30.0)
                        a.z = min(max(a.z, 30.0), SIZE - 30.0)
                        self.leave_ship(a)
        self._storm(dt)
        self._pace()
        for a in self.actors:
            if not a.alive:
                continue
            if a.brain is not None:
                a.brain.update(dt)
            if a.fire_cd > 0:
                a.fire_cd -= dt
            if a.bloom > 0:
                a.bloom = max(a.bloom - (0.06 + a.bloom * 5.0) * dt, 0.0)
            if a.reload_t > 0:
                a.reload_t -= dt
                if a.reload_t <= 0:
                    s = a.held()
                    if s is not None and s.is_weapon:
                        w = WEAPONS[s.id]
                        take = min(w.mag - s.mag, a.ammo[w.ammo])
                        s.mag += take
                        a.ammo[w.ammo] -= take
            if a.use_t > 0:
                a.use_t -= dt
                if a.use_t <= 0:
                    self._finish_use(a)
            if a.state == "ground":
                self._move(a, dt)
                if a.firing and a.alive:
                    self.fire(a)
            elif a.state != "ship":
                self._air(a, dt)
            a.want_jump = False
        self._projectiles(dt)
        if self.tracers:
            for t in self.tracers:
                t[6] -= dt
            self.tracers = [t for t in self.tracers if t[6] > 0]
        if self.numbers:
            for n in self.numbers:
                n[6] += dt
            self.numbers = [n for n in self.numbers if n[6] < 0.9]
        for p in self.structs.broken:
            x, y, z = p.centre()
            self.fx.append(("burst", x, y, z, _MAT_FX[p.mat], 14, 5.0))
            self.sound(("break_wood", "break_brick", "break_metal")[p.mat], x, y, z)
        self.structs.broken.clear()
        me = self.player
        if me.alive and not me.bot and me.state == "ground":
            self._player_loot()

    def _player_loot(self) -> None:
        me = self.player
        fx, fz = math.sin(me.yaw), math.cos(me.yaw)
        best, score = None, 1e9
        for c in self.chests:
            if not c.opened:
                dx, dz = c.x - me.x, c.z - me.z
                d2 = dx * dx + dz * dz
                if d2 < 7.5 and abs(c.y - me.y) < 2.2:
                    sc = d2 - (dx * fx + dz * fz) * 0.8
                    if sc < score:
                        best, score = c, sc
        auto = self.options.get("auto_pickup", True)
        for it in self.items_near(me.x, me.z, 2.6):
            if not it.alive or abs(it.y - me.y) > 2.0:
                continue
            dx, dz = it.x - me.x, it.z - me.z
            d2 = dx * dx + dz * dz
            if it.kind in ("ammo", "mats"):
                if d2 < 2.9 and self.give(me, it):
                    self.sound("pickup", volume=0.7)
                continue
            if auto and d2 < 2.0 and self.time > it.no_auto:
                if it.kind == "weapon" and any(s is not None and s.id == it.id and s.grade >= it.grade
                                               for s in me.inv):
                    room = False  # one of these is enough: E still takes it
                else:
                    room = any(s is None for s in me.inv)
                stack = it.kind == "supply" and any(
                    s is not None and s.id == it.id and s.count < SUPPLIES[it.id].stack for s in me.inv)
                if (room or stack) and self.give(me, it):
                    self.sound("pickup_weapon" if it.kind == "weapon" else "pickup")
                    continue
            sc = d2 - (dx * fx + dz * fz) * 0.8 + 0.4
            if sc < score:
                best, score = it, sc
        self.focus = best


_MAT_FX = ((0.80, 0.56, 0.30), (0.78, 0.44, 0.34), (0.62, 0.70, 0.78))


def _push_props(plist: list, x: float, z: float, feet: float, head: float) -> tuple[float, float]:
    for p in plist:
        if not p.alive:
            continue
        top = p.y + p.h
        if feet + STEP_UP >= top or head <= p.y:
            continue
        if p.shape == "cyl":
            dx, dz = x - p.x, z - p.z
            rr = p.a + RADIUS
            d2 = dx * dx + dz * dz
            if d2 < rr * rr:
                d = math.sqrt(d2)
                if d > 1e-6:
                    x, z = p.x + dx / d * rr, p.z + dz / d * rr
                else:
                    x = p.x + rr
        else:
            px = min(max(x, p.x - p.a), p.x + p.a)
            pz = min(max(z, p.z - p.b), p.z + p.b)
            dx, dz = x - px, z - pz
            d2 = dx * dx + dz * dz
            if d2 < RADIUS * RADIUS:
                if d2 > 1e-10:
                    d = math.sqrt(d2)
                    x, z = px + dx / d * RADIUS, pz + dz / d * RADIUS
                else:  # inside the box: out through the nearest side
                    ex, ez = p.a - abs(x - p.x), p.b - abs(z - p.z)
                    if ex < ez:
                        x = p.x + (p.a + RADIUS) * (1 if x >= p.x else -1)
                    else:
                        z = p.z + (p.b + RADIUS) * (1 if z >= p.z else -1)
    return x, z


__all__ = ["Actor", "Chest", "Item", "Match", "Slot", "RAMP_DX", "ROOF", "RAMP", "WALLX", "WALLZ", "FLOOR"]
