"""More of a BlockCraft world (mixed into the game widget beside play.py and life.py): chests and what is found in
the old ones, feeding and breeding animals, riding a saddled horse, the weather, and the rune bench."""

from __future__ import annotations

import math

import numpy as np

from app.arcade.games.blockcraft import blocks as B
from app.arcade.games.blockcraft import items as I
from app.arcade.games.blockcraft.entities import (BABY_TIME, BREED_REST, COLD_BIOMES, DRY_BIOMES, FOOD_OF, LOVE_TIME,
                                                  TICK, Mob)
from app.arcade.games.blockcraft.life import realm_key
from app.arcade.games.blockcraft.signals import Signals

CHEST_SLOTS = 27
WEATHERS = ("clear", "rain", "storm")
_SKY = {"clear": 0.0, "rain": 0.72, "storm": 1.0}  # how far each closes the sky
_SPELL = {"clear": (480.0, 1200.0), "rain": (180.0, 420.0), "storm": (120.0, 300.0)}  # seconds each may last
_NEXT = {"clear": (("rain", 0.72), ("storm", 0.28)), "rain": (("clear", 0.7), ("storm", 0.3)),
         "storm": (("rain", 0.55), ("clear", 0.45))}
DROPS = 1500  # rain drops (or snow flakes) drawn around the camera in a storm
FALL_REACH = 12.0  # ... within this many blocks of it
_COVER = B.SOLID | B.FLUID  # what rain does not fall through
_COVER[B.LAVA] = True
_LIVE = [b in B.DUST_LEVELS[1:] for b in range(256)]  # signal dust with a signal in it
# what an old chest may hold, by realm: (item, fewest, most, chance)
LOOT = {
    "over": ((B.BREAD, 1, 4, 0.8), (B.WHEAT_ITEM, 2, 8, 0.6), (B.WHEAT_SEEDS, 2, 6, 0.5), (B.APPLE, 1, 3, 0.5),
             (B.COAL, 2, 8, 0.6), (B.IRON_INGOT, 1, 4, 0.45), (B.STRING, 1, 4, 0.4), (B.LEATHER, 1, 3, 0.4),
             (B.GEM, 1, 2, 0.3), (B.SADDLE, 1, 1, 0.3), (B.TORCH, 2, 8, 0.5), (B.REDSTONE, 2, 6, 0.3)),
    "deep": ((B.GOLD_INGOT, 2, 6, 0.8), (B.IRON_INGOT, 2, 8, 0.7), (B.COAL, 4, 12, 0.6), (B.DIAMOND, 1, 3, 0.4),
             (B.REDSTONE, 4, 12, 0.7), (B.GEM, 1, 4, 0.5), (B.SADDLE, 1, 1, 0.25), (B.COOKED_PORK, 1, 4, 0.5),
             (B.OBSIDIAN, 1, 4, 0.3), (B.tool_id(3, 0), 1, 1, 0.3), (B.ARMOR0 + 4, 1, 1, 0.2)),
    "isles": ((B.DIAMOND, 1, 4, 0.6), (B.GEM, 2, 6, 0.7), (B.GOLD_INGOT, 2, 8, 0.7), (B.FEATHER, 2, 8, 0.6),
              (B.DEEP_CRYSTAL, 1, 4, 0.4), (B.GLOWSTONE, 1, 4, 0.4), (B.BREAD, 2, 6, 0.6), (B.SADDLE, 1, 1, 0.2),
              (B.tool_id(4, 3), 1, 1, 0.15), (B.ARMOR0 + 11, 1, 1, 0.15)),
}


def is_chest(b: int) -> bool:
    return B.CHEST <= b <= B.OLD_CHEST


def chest_loot(seed: int, x: int, y: int, z: int, dim: str) -> list:
    """The 27 slots of an old chest: the same every time for one place of one world."""
    rng = np.random.default_rng([seed & 0xFFFFFFFF, x & 0xFFFFFFFF, y & 0xFF, z & 0xFFFFFFFF, 77])
    slots: list = [None] * CHEST_SLOTS
    free = list(rng.permutation(CHEST_SLOTS))
    for item, lo, hi, chance in LOOT.get(dim, LOOT["over"]):
        if rng.random() < chance and free:
            n = int(rng.integers(lo, hi + 1))
            slots[int(free.pop())] = I.stack(item, min(n, B.max_stack(item)))
    return slots


def slots_json(slots: list) -> list:
    return [[i, s[0], s[1], s[2]] for i, s in enumerate(slots) if s is not None]


def slots_load(data, size: int) -> list:  # noqa: ANN001 - whatever the save holds
    slots: list = [None] * size
    for entry in data if isinstance(data, list) else []:
        try:
            i, item, count, dmg = (int(v) for v in entry[:4])
        except (TypeError, ValueError):
            continue
        if 0 <= i < size and item in B.NAMES and count > 0:
            slots[i] = I.stack(item, min(count, B.max_stack(item)), max(dmg, 0))
    return slots


class ExtrasMixin:
    """See the module docstring."""

    def _reset_extras(self) -> None:
        self.chests: dict[tuple, list] = {}  # (x, y, z) -> 27 slots
        self.chest_pos: tuple | None = None  # the open one
        self.weather = "clear"
        self.weather_left = 600.0  # seconds until it changes
        self.rain = 0.0  # 0..1: how closed the sky is over the player right now (eased)
        self.flash = 0.0  # lightning, fading
        self.thunder_in = 8.0
        self.exposed = False  # the player stands under the open sky where something falls from it
        self._fall_seed = np.random.default_rng(31).random((DROPS, 4))
        self.riding: Mob | None = None  # the horse under the player
        self.rune_slots: list = [None, None]  # on the open rune bench: the thing to mark, and the gems to pay with
        self.signals = Signals()
        self._hoof = 0.0
        self._fall_maps: tuple | None = None
        self._col_cache: dict = {}

    # --- chests -------------------------------------------------------------------------------------------------

    def chest_slots(self, pos: tuple | None) -> list | None:
        """The slots of the chest at a place (an old chest gets its loot the first time anyone looks)."""
        if pos is None or self.world is None:
            return None
        slots = self.chests.get(pos)
        if slots is None and is_chest(self.world.get(*pos)):
            if self.world.get(*pos) == B.OLD_CHEST:
                slots = chest_loot(self.world.seed, *pos, self.dim)
            else:
                slots = [None] * CHEST_SLOTS
            self.chests[pos] = slots
        return slots

    def open_chest(self, pos: tuple) -> None:
        b = self.world.get(*pos)
        self.chest_slots(pos)
        if b == B.OLD_CHEST:  # found: from now on it is a chest like any other, its latch towards the finder
            self._set_blocks([(*pos, B.CHEST + int(round(self.player.yaw / 90)) % 4)])
            self.award("loot")
        self.chest_pos = pos
        self.open_overlay("chest")

    def spill_chest(self, pos: tuple, block: int) -> None:
        """A chest was broken (``block``: what stood there): what it held falls out."""
        slots = self.chests.pop(pos, None)
        if slots is None and block == B.OLD_CHEST:
            slots = chest_loot(self.world.seed, *pos, self.dim)
        for s in slots or ():
            if s is not None:
                self._spawn_drop(pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5, s, 0.15)
        if self.chest_pos == pos:
            self.chest_pos = None

    def _load_chests(self, r: dict) -> None:
        self.chests = {}
        for entry in r.get(realm_key("chests", self.dim), []) or []:
            try:
                pos = (int(entry[0]), int(entry[1]), int(entry[2]))
                self.chests[pos] = slots_load(entry[3], CHEST_SLOTS)
            except (TypeError, ValueError, IndexError):
                continue

    def _store_extras(self, r: dict) -> None:
        w = self.world
        r["weather"] = [self.weather, round(self.weather_left)]
        r["riding"] = self.riding is not None and self.dim == "over"
        r[realm_key("chests", self.dim)] = [[*pos, slots_json(slots)] for pos, slots in self.chests.items()
                                            if is_chest(w.get(*pos)) or w.get(*pos) == B.BORDER]

    def _load_extras(self, r: dict) -> None:
        """What a realm keeps beside its blocks (after its land stands)."""
        self._load_chests(r)
        wth = r.get("weather")
        if isinstance(wth, list) and len(wth) >= 2 and wth[0] in WEATHERS:
            try:
                self.weather, self.weather_left = wth[0], min(max(float(wth[1]), 1.0), 1200.0)
            except (TypeError, ValueError):
                pass
        else:  # a world from before there was weather (or a new one): fair for a while
            self.weather, self.weather_left = "clear", float(self.rng.uniform(300, 900))
        self.rain = _SKY[self.weather] if self.dim == "over" else 0.0

    def _load_animals(self, r: dict) -> None:
        """Back in the overworld from another realm: the animals that were left there (a saddled horse, the young
        ones) are where they were."""
        from app.arcade.games.blockcraft.entities import ANIMALS

        for m in r.get("mobs") or []:
            try:
                kind, x, y, z, yaw, hp = m[0], float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])
            except (TypeError, ValueError, IndexError):
                continue
            if kind in ANIMALS and hp > 0:
                mob = Mob(kind, x, y, z, self.rng)
                mob.yaw, mob.health = yaw, hp
                mob.load_more(m[6:])
                self.mobs.append(mob)
        self._load_riders({})

    def _load_riders(self, r: dict) -> None:
        """After the saved animals are back: saddled horses wait where they were left; the one the player sat on
        when the world was saved carries them again."""
        p = self.player
        for m in self.mobs:
            if m.saddled:
                m.home, m.leash = (m.x, m.y, m.z, -1), 6.0
        if r.get("riding") and self.dim == "over":
            near = [m for m in self.mobs if m.saddled and not m.dead and math.dist((m.x, m.z), (p.x, p.z)) < 4]
            if near:
                self.mount(min(near, key=lambda m: math.dist((m.x, m.z), (p.x, p.z))), quiet=True)

    def extras_tick(self) -> None:
        t = self.tick_count
        self._weather_tick()
        if t % 2 == 0 and (self.signals.queue or self.signals.timers):
            self._signal_tick()
        if t % 10 == 6:
            self._breed_tick()

    # --- signals: levers, buttons, dust, lamps and doors (signals.py) -------------------------------------------

    def _signal_tick(self) -> None:
        w = self.world
        changes, events = self.signals.step(w)
        if changes:
            get = w.get
            same = [_LIVE[c[3]] and _LIVE[get(c[0], c[1], c[2])] for c in changes]
            if any(same):  # live dust growing weaker or stronger: nothing to see, nothing to light or draw again
                w.set_alike([c for c, s in zip(changes, same, strict=True) if s])
                changes = [c for c, s in zip(changes, same, strict=True) if not s]
            if changes:
                w.set_many(changes)
            self.edit_time = self.world_time
        for ev in events:
            x, y, z = ev[1]
            if ev[0] == "drop":
                if not self.creative:
                    self._spawn_drop(x + 0.5, y + 0.3, z + 0.5, I.stack(ev[2], 1))
            elif ev[0] == "door":
                self.water.poke(x, y, z)
                self.sfx.play("door", 0.7)
            elif ev[0] == "lamp":
                if ev[2]:
                    self.award("signal")
            elif ev[0] == "click":
                self.sfx.play("click", 0.4)

    def use_signal(self, pos: tuple, b: int, fresh: bool) -> bool:
        """The use button on a lever, a button or a door: True when that was it."""
        get = self.world.get
        if b in (B.LEVER, B.LEVER_ON):
            if fresh:
                self._set_blocks([(*pos, B.LEVER_ON if b == B.LEVER else B.LEVER)])
                self.signals.edge(get, *pos)
                self.sfx.play("click", 0.7)
                self.swing = 1.0
            return True
        if b in (B.BUTTON, B.BUTTON_ON):
            if fresh:
                self._set_blocks([(*pos, B.BUTTON_ON)])
                self.signals.press(pos)
                self.signals.edge(get, *pos)
                self.sfx.play("click", 0.7)
                self.swing = 1.0
            return True
        if b in B.DOORS:
            if fresh:  # by hand: both halves swing
                k = (b - B.DOOR) ^ 2
                oy = pos[1] - 1 if k & 1 else pos[1] + 1
                changes = [(*pos, B.DOOR + k)]
                if get(pos[0], oy, pos[2]) in B.DOORS:
                    changes.append((pos[0], oy, pos[2], B.DOOR + (k ^ 1)))
                if self.world.set_many(changes):  # (not poked: a door set by hand stays as it was set)
                    self.edit_time = self.world_time
                    self.water.poke(*pos)
                self.sfx.play("door", 0.7)
                self.swing = 1.0
            return True
        return False

    def place_signal(self, pos: tuple, normal: tuple, item: int) -> bool:
        """Lay signal dust, or stand a door (two blocks high, across the way the player faces)."""
        w = self.world
        from app.arcade.games.blockcraft.play import REPLACEABLE

        x, y, z = pos if w.get(*pos) in REPLACEABLE else (pos[0] + normal[0], pos[1] + normal[1], pos[2] + normal[2])
        if not w.inside(x, y, z) or w.get(x, y, z) not in REPLACEABLE or B.FLUID[w.get(x, y, z)] or \
                not B.SOLID[w.get(x, y - 1, z)]:
            return False
        if item == B.REDSTONE:
            changes = [(x, y, z, B.DUST)]
        else:
            if w.get(x, y + 1, z) not in REPLACEABLE or B.FLUID[w.get(x, y + 1, z)] or y + 1 >= w.height:
                return False
            box = (x, y, z, x + 1, y + 2, z + 1)
            pb = self.player.box()
            if pb[0] < box[3] and pb[3] > box[0] and pb[1] < box[4] and pb[4] > box[1] and pb[2] < box[5] and pb[5] > box[2]:
                return False
            axis = 0 if int(round(self.player.yaw / 90)) % 2 == 0 else 1  # facing along z: the leaf lies along x
            changes = [(x, y, z, B.DOOR + axis * 4), (x, y + 1, z, B.DOOR + axis * 4 + 1)]
        self._set_blocks(changes)
        self.sfx.play(f"place_{B.MATERIAL.get(changes[0][3], 'stone')}")
        self.swing = 1.0
        if not self.creative:
            self.inv.take_held(1)
        return True

    # --- the weather --------------------------------------------------------------------------------------------

    def set_weather(self, kind: str, seconds: float | None = None) -> None:
        """Clear, rain or storm from now on, for ``seconds`` (or as long as such weather lasts)."""
        if kind not in WEATHERS:
            return
        self.weather = kind
        lo, hi = _SPELL[kind]
        self.weather_left = float(seconds) if seconds is not None else float(self.rng.uniform(lo, hi))
        self.thunder_in = float(self.rng.uniform(3, 9))

    def fall_kind(self, x: float, z: float) -> int:
        """What falls from a wet sky at a place: 0 rain, 1 snow, 2 nothing (the dry lands)."""
        name = self.world.biome_name(x, z)
        return 1 if name in COLD_BIOMES else (2 if name in DRY_BIOMES else 0)

    def raining_at(self, x: int, y: int, z: int) -> bool:
        """Does rain (or snow) reach this block right now?"""
        if self.dim != "over" or self.weather == "clear" or self.fall_kind(x, z) == 2:
            return False
        return self.world.top_y(x, z) < y

    def _weather_tick(self) -> None:
        self.weather_left -= TICK
        if self.weather_left <= 0:
            roll, acc = float(self.rng.random()), 0.0
            for kind, chance in _NEXT[self.weather]:
                acc += chance
                if roll < acc:
                    self.set_weather(kind)
                    break
            else:
                self.set_weather("clear")
        if self.dim != "over":
            self.exposed = False
            return
        p, t = self.player, self.tick_count
        if t % 10 == 4:
            x, y, z = math.floor(p.x), math.floor(p.y + 1.7), math.floor(p.z)
            self.exposed = self.rain > 0.3 and self.raining_at(x, y, z)
            if self.exposed:
                if self.fire > 0:
                    self.fire = 0.0  # the rain puts the player out
                self.award("rain")
            if self.rain > 0.3:
                for m in self.mobs:  # ... and whatever else was burning
                    if m.burning > 0 and self.raining_at(math.floor(m.x), math.floor(m.y + m.height), math.floor(m.z)):
                        m.burning = 0.0
        if self.rain > 0.2 and t % 44 == 9 and self.fall_kind(p.x, p.z) == 0:
            self.sfx.play("rain", (0.3 + 0.7 * self.rain) * (1.0 if self.exposed else 0.35))
        if self.weather == "storm" and self.rain > 0.6:
            self.thunder_in -= TICK
            if self.thunder_in <= 0:
                self.thunder_in = float(self.rng.uniform(5, 22))
                self.flash = 1.0
                self.sfx.play("thunder", float(self.rng.uniform(0.5, 1.0)))

    def _weather_frame(self, dt: float) -> None:
        """Ease the sky towards the weather (thinner over the dry lands), fade the lightning."""
        want = 0.0
        if self.dim == "over":
            want = _SKY[self.weather]
            if want and self.fall_kind(self.player.x, self.player.z) == 2:
                want *= 0.4
        step = dt / 6.0  # a change of weather takes a few seconds to come over
        self.rain += min(max(want - self.rain, -step), step)
        if self.flash > 0:
            before = self.flash
            self.flash = max(0.0, self.flash - dt * (5.0 if self.flash > 0.5 else 1.6))
            if before > 0.5 >= self.flash and self.rng.random() < 0.4:
                self.flash = 0.85  # now and then a second stroke

    def _column_maps(self, key: tuple) -> tuple | None:
        """For a chunk: the height of what stands in each column, and what falls there (0 rain, 1 snow, 2 none)."""
        c = self.world.chunks.get(key)
        if c is None:
            return None
        hit = self._col_cache.get(key)
        if hit is not None and hit[0] == c.ver and hit[3] is c:
            return hit
        cover = _COVER[c.blk[:, :, :c.top]]
        tops = np.where(cover.any(2), c.top - 1 - np.argmax(cover[:, :, ::-1], 2), -1).astype(np.int16)
        names = self.world.gen.names
        table = np.array([1 if n in COLD_BIOMES else (2 if n in DRY_BIOMES else 0) for n in names], np.uint8)
        if len(self._col_cache) > 64:
            self._col_cache.clear()
        hit = self._col_cache[key] = (c.ver, tops, table[c.biome], c)
        return hit

    def weather_dots(self, eye: tuple) -> np.ndarray | None:
        """The rain drops and snow flakes around the camera, as rows for the dots the renderer draws (None:
        nothing falls). Nothing is kept from frame to frame: where a drop is follows from the time alone."""
        if self.rain < 0.05 or self.dim != "over" or self.world is None:
            return None
        ex, ey, ez = eye
        cx, cz = math.floor(ex) >> 4, math.floor(ez) >> 4
        cols = [self._column_maps((cx + i, cz + j)) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        if any(c is None for c in cols):
            return None
        vers = (cx, cz, *(c[0] for c in cols))
        maps = self._fall_maps
        if maps is None or maps[0] != vers:
            tops = np.block([[cols[i * 3 + j][1] for j in range(3)] for i in range(3)])
            kinds = np.block([[cols[i * 3 + j][2] for j in range(3)] for i in range(3)])
            maps = self._fall_maps = (vers, tops, kinds)
        _vers, tops, kinds = maps
        n = int(DROPS * min(self.rain, 1.0) * (1.0 if self.opts.get("particles", True) else 0.45))
        if n <= 0:
            return None
        seed = self._fall_seed[:n]
        reach, span = FALL_REACH, FALL_REACH * 2
        x = ex - reach + np.mod(seed[:, 0] * span - (ex - reach), span)  # each drop keeps its place in the world
        z = ez - reach + np.mod(seed[:, 1] * span - (ez - reach), span)
        xi = np.clip(np.floor(x).astype(np.intp) - (cx - 1) * 16, 0, 47)
        zi = np.clip(np.floor(z).astype(np.intp) - (cz - 1) * 16, 0, 47)
        kind, top = kinds[xi, zi], tops[xi, zi]
        snow = kind == 1
        t, tall, y0 = self.time, 26.0, ey - 12.0
        y = y0 + np.mod(seed[:, 2] * tall - t * np.where(snow, 2.4, 19.0) - y0, tall)
        keep = (kind < 2) & (y > top + 1.0) & ((np.abs(x - ex) > 1.0) | (np.abs(z - ez) > 1.0) | snow)
        if not keep.any():
            return None
        x, y, z, snow, ph = x[keep], y[keep], z[keep], snow[keep], seed[keep, 3]
        drift = np.where(snow, np.sin(t * 1.3 + ph * 40.0) * 0.35, 0.0)
        origin = math.floor(ex / 16) * 16, math.floor(ez / 16) * 16
        light = 0.3 + 0.7 * float(self.env.daylight)
        out = np.empty((len(x), 8), np.float32)
        out[:, 0], out[:, 1], out[:, 2] = x + drift - origin[0], y, z + drift * 0.6 - origin[1]
        out[:, 3] = np.where(snow, 0.075, 0.022)
        out[:, 4] = np.where(snow, 0.95, 0.62) * light
        out[:, 5] = np.where(snow, 0.96, 0.68) * light
        out[:, 6] = np.where(snow, 1.0, 0.8) * light
        out[:, 7] = np.where(snow, 1.0, 32.0)  # a drop is a streak, many times as long as it is wide
        return out

    # --- feeding and breeding -----------------------------------------------------------------------------------

    def feeds_mob(self, item: int) -> bool:
        """Is the animal looked at one that eats this (then the use button feeds it; the player does not eat)?"""
        m = self.target_mob
        return m is not None and not m.dead and item in FOOD_OF.get(m.kind, ())

    def use_mob(self, m: Mob, fresh: bool) -> bool:
        """The use button on an animal: True when that was it."""
        held = self.inv.held
        if held is not None and not m.dead and held[0] in FOOD_OF.get(m.kind, ()):
            if fresh:
                self._feed(m)
            return True
        if m.kind != "horse" or m.dead:
            return False
        if held is not None and held[0] == B.SADDLE and not m.saddled:
            if fresh:
                if m.baby > 0:
                    self.say("This horse is too young for a saddle")
                    return True
                m.saddled = True
                m.home, m.leash = (m.x, m.y, m.z, -1), 6.0  # a saddled horse waits where it is left
                if not self.creative:
                    self.inv.take_held(1)
                self.sfx.play("place_cloth")
                self.swing = 1.0
                self.say("Saddled. Use the horse again to mount it")
            return True
        if m.saddled and self.riding is None:
            if fresh:
                self.mount(m)
            return True
        if not m.saddled and held is None:
            if fresh:
                self.say("A horse needs a saddle to be ridden")
            return True
        return False

    # --- the rune bench -----------------------------------------------------------------------------------------
    # One rune may be cut into a tool, a weapon or a piece of armour, for experience levels and a trade gem. The
    # rune is part of the item from then on (blocks.py: runes); a new one replaces the old.

    def held_rune(self) -> int:
        s = self.inv.held
        return B.rune_of(s[2]) if s else 0

    def worn_runes(self, rune: int) -> int:
        """How many pieces of the armour worn bear this rune."""
        return sum(1 for s in self.inv.armor if s is not None and B.rune_of(s[2]) == rune)

    def rune_slot_takes(self, k: int, item: int) -> bool:
        return item == B.GEM if k == 1 else bool(B.runes_for(item))

    def rune_rows(self) -> list:
        """What the open bench offers for the item on it: (rune, levels, can it be paid for now)."""
        s, gem = self.rune_slots
        if s is None:
            return []
        lvl = self.level()[0]
        paid = gem is not None and gem[0] == B.GEM
        return [(r, B.RUNE_LEVELS[r], self.creative or (paid and lvl >= B.RUNE_LEVELS[r])) for r in B.runes_for(s[0])]

    def apply_rune(self, k: int) -> bool:
        """Cut the k-th offered rune into the item on the bench."""
        rows = self.rune_rows()
        if not 0 <= k < len(rows):
            return False
        rune, levels, ok = rows[k]
        s = self.rune_slots[0]
        if not ok or B.rune_of(s[2]) == rune:
            self.sfx.play("click", 0.4)
            return False
        if not self.creative:
            self.spend_levels(levels)
            gem = self.rune_slots[1]
            gem[1] -= 1
            if gem[1] <= 0:
                self.rune_slots[1] = None
        s[2] = B.with_rune(s[2], rune)
        if B.wear_of(s[2]) >= B.max_wear(s[0], s[2]) > 0:  # (a Lasting rune swapped off a tool worn past its own life)
            s[2] = B.with_rune(B.max_wear(s[0], s[2]) - 1, rune)
        self.sfx.play("levelup", 0.7)
        self.award("rune")
        return True

    def spend_levels(self, n: int) -> None:
        """Take whole experience levels away (what was gathered towards the next one is kept in proportion)."""
        from app.arcade.games.blockcraft.play import xp_for_level

        lvl, frac = self.level()
        lvl = max(0, lvl - n)
        self.xp_total = sum(xp_for_level(k) for k in range(lvl)) + int(frac * xp_for_level(lvl))

    # --- riding -------------------------------------------------------------------------------------------------

    def mount(self, horse: Mob, quiet: bool = False) -> None:
        p = self.player
        self.riding = horse
        horse.ridden, horse.panic, horse.moving = True, 0.0, False
        horse.vx = horse.vz = 0.0
        p.flying = False
        p.vx = p.vy = p.vz = 0.0
        self._seat()
        p.remember()
        self.mining = None
        if not quiet:
            self.sfx.play("horse", 0.7)
            self.say("WASD to ride, Space to jump, Shift to get down", 5.0)
            self.award("rider")

    def dismount(self) -> None:
        horse, p = self.riding, self.player
        if horse is None:
            return
        self.riding = None
        horse.ridden = False
        horse.home, horse.leash = (horse.x, horse.y, horse.z, -1), 6.0
        horse.vx = horse.vz = 0.0
        a = math.radians(horse.yaw)
        for side in (1.0, -1.0, 0.0):  # down on its left, else its right, else where it stands
            x, z = horse.x - math.cos(a) * 1.1 * side, horse.z - math.sin(a) * 1.1 * side
            if not self.world.solid_boxes(x - 0.3, horse.y + 0.05, z - 0.3, x + 0.3, horse.y + 1.75, z + 0.3):
                break
        p.x, p.y, p.z = x, horse.y + 0.02, z
        p.vx = p.vy = p.vz = 0.0
        p.fall = 0.0
        self._unbury(p)
        p.remember()

    def _seat(self) -> None:
        horse, p = self.riding, self.player
        p.x, p.y, p.z = horse.x, horse.y + horse.height - 0.45, horse.z
        p.vx = p.vy = p.vz = 0.0
        p.fall = 0.0

    def ride_tick(self, forward: float, strafe: float, jump: bool, sneak: bool) -> None:
        """One tick in the saddle (instead of the player walking): the horse goes where the player looks."""
        horse, p, w = self.riding, self.player, self.world
        if horse.dead or sneak or self.health <= 0 or horse not in self.mobs:
            self.dismount()
            return
        p.remember()
        horse.remember()
        p.prev_walked, p.prev_bob = p.walked, p.bob
        p.bob *= 0.6
        p.sprinting = p.sneaking = p.jumped = p.on_ground = False
        p.landed_fall = 0.0
        horse.age += TICK
        if horse.hurt > 0:
            horse.hurt -= TICK
        horse.check_liquids(w)
        turn = (p.yaw - horse.yaw + 180) % 360 - 180
        horse.yaw += max(-14.0, min(14.0, turn))
        horse.target_yaw = horse.yaw
        fwd = forward if forward > 0 else forward * 0.4  # backwards at a walk
        if horse.in_water or horse.in_lava:
            horse.vy += 0.04 if horse.vy < 0.1 else 0.0
            horse.move_relative(strafe * 0.3, fwd, 0.03)
            horse.move(w, horse.vx, horse.vy, horse.vz)
            horse.vx *= 0.8
            horse.vy = horse.vy * 0.8 - 0.02
            horse.vz *= 0.8
            if horse.hit_wall and fwd:
                horse.vy = 0.3  # up the bank
            horse.fall = 0.0
        else:
            if horse.on_ground and jump:
                horse.vy = 0.56  # nearly two blocks high
            elif horse.on_ground and horse.hit_wall and fwd > 0:
                horse.vy = 0.42  # a step of one block is taken in stride
            slip = 0.546 if horse.on_ground else 0.91
            horse.move_relative(strafe * 0.4, fwd, 0.2 if horse.on_ground else 0.035)
            horse.move(w, horse.vx, horse.vy, horse.vz)
            if horse.on_ground:
                if horse.fall > 5:
                    horse.health -= math.ceil(horse.fall - 5)
                    horse.hurt = 0.5
                    if not self.creative:
                        self.damage(math.ceil((horse.fall - 5) / 2), "fall")
                horse.fall = 0.0
            horse.vy = (horse.vy - 0.08) * 0.98
            horse.vx *= slip
            horse.vz *= slip
        if horse.in_lava:
            horse.health -= 0.2
        moved = math.hypot(horse.x - horse.px, horse.z - horse.pz)
        horse.swing_amount += (min(moved * 3, 1.0) - horse.swing_amount) * 0.4
        horse.swing += moved * 6
        if horse.on_ground:
            self._hoof += moved
            if self._hoof > 2.2:
                self._hoof = 0.0
                below = w.get(math.floor(horse.x), math.floor(horse.y - 0.2), math.floor(horse.z))
                if below not in (B.AIR, B.BORDER):
                    self.sfx.play(f"step_{B.MATERIAL.get(below, 'stone')}", 0.9)
        if horse.y < -20:
            horse.health = 0
        self._seat()
        p.check_liquids(w)

    def _feed(self, m: Mob) -> None:
        if m.baby > 0:
            m.baby = max(1.0, m.baby - BABY_TIME * 0.1)  # the young grow faster for it
        elif m.love > 0 or m.breed_rest > 0:
            self.say(f"This {m.kind.replace('_', ' ')} is not hungry now")
            return
        else:
            m.love = LOVE_TIME
        if not self.creative:
            self.inv.take_held(1)
        self.particles.hearts(m.x, m.y + m.height, m.z, self.player.yaw)
        self.sfx.play("eat", 0.6)
        self.swing = 1.0

    def _breed_tick(self) -> None:
        """Two fed animals of a kind find each other and have a young one; the young keep to an adult."""
        mobs = self.mobs
        lovers = [m for m in mobs if m.love > 0 and not m.dead and m.baby <= 0]
        for a in lovers:
            b = a.mate
            if b is None or b.dead or b.love <= 0:
                near = [m for m in lovers if m is not a and m.kind == a.kind and m.mate in (None, a) and
                        math.dist((a.x, a.y, a.z), (m.x, m.y, m.z)) < 12]
                b = a.mate = min(near, key=lambda m: math.dist((a.x, a.z), (m.x, m.z)), default=None)
                if b is not None:
                    b.mate = a
            if b is None or a.love <= 0 or b.love <= 0 or math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) > 1.5 + a.width:
                continue
            young = Mob(a.kind, (a.x + b.x) / 2, max(a.y, b.y), (a.z + b.z) / 2, self.rng)
            young.make_baby()
            young.parent = a
            mobs.append(young)
            for m in (a, b):
                m.love, m.breed_rest, m.mate = 0.0, BREED_REST, None
            self.particles.hearts(young.x, young.y + a.height, young.z, self.player.yaw, 4)
            self.sfx.play(a.kind, 0.7)
            self.gain_xp(int(self.rng.integers(1, 8)))
            self.award("breed")
        if self.tick_count % 40 == 6:
            for m in mobs:
                if m.baby > 0 and (m.parent is None or m.parent.dead):
                    m.parent = min((o for o in mobs if o.kind == m.kind and o.baby <= 0 and not o.dead and
                                    math.dist((o.x, o.z), (m.x, m.z)) < 14),
                                   key=lambda o: math.dist((o.x, o.z), (m.x, m.z)), default=None)

    def extras_frame(self, dt: float) -> None:
        self._weather_frame(dt)


__all__ = ["CHEST_SLOTS", "LOOT", "WEATHERS", "ExtrasMixin", "chest_loot", "is_chest"]
