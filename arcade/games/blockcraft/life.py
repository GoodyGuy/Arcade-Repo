"""The living side of a BlockCraft world (mixed into the game widget beside play.py): plants that grow, farming,
beds and sleep, the people of the hamlets and their trades, and who appears where (fish in the sea, settlers and
their guardian at home)."""

from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt

from app.arcade.games.blockcraft import blocks as B
from app.arcade.games.blockcraft import items as I
from app.arcade.games.blockcraft.entities import Mob
from app.arcade.games.blockcraft.sounds import material_of
from app.arcade.games.blockcraft.worldgen import HEIGHT, SEA, TREES

REALM_NAMES = {"over": "the Overworld", "deep": "the Deep", "isles": "the Sky Isles"}


def realm_key(name: str, dim: str) -> str:
    """The key of a saved world record that holds ``name`` ("edits", "furn") of a realm."""
    return name if dim == "over" else f"{name}_{dim}"


def _vec(v) -> bool:  # noqa: ANN001
    try:
        return len(v) >= 3 and all(math.isfinite(float(x)) for x in v[:3])
    except (TypeError, ValueError):
        return False

RANDOM_TICKS = 24  # blocks looked at per chunk and game tick (a block is reached about once a minute)
GROW_CHUNKS = 2  # chunks around the player where things grow
GROWING = frozenset({B.GRASS, B.OAK_SAPLING, B.SUGAR_CANE, B.KELP, B.FARMLAND, B.WHEAT, B.WHEAT + 1, B.WHEAT + 2})
_SOFT_PLANTS = (B.TALL_GRASS, B.FERN, B.DEAD_BUSH)
_SIDES = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_FISH = 10
# what the people of a hamlet trade: (item wanted, how many, item given, how many)
JOBS = {
    "Farmer": [(B.WHEAT_ITEM, 10, B.GEM, 1), (B.GEM, 1, B.BREAD, 4), (B.GEM, 2, B.HOE0 + 2, 1),
               (B.GEM, 1, B.WHEAT_SEEDS, 8)],
    "Smith": [(B.COAL, 12, B.GEM, 1), (B.IRON_INGOT, 4, B.GEM, 1), (B.GEM, 4, B.tool_id(2, 0), 1),
              (B.GEM, 6, B.ARMOR0 + 1, 1), (B.GEM, 3, B.ARMOR0, 1)],
    "Fisher": [(B.FISH, 6, B.GEM, 1), (B.STRING, 6, B.GEM, 1), (B.GEM, 1, B.BUCKET, 1), (B.GEM, 2, B.COOKED_FISH, 6)],
    "Shepherd": [(B.WOOL, 8, B.GEM, 1), (B.FEATHER, 8, B.GEM, 1), (B.GEM, 2, B.BED, 1), (B.GEM, 1, B.STEAK, 3)],
}


class LifeMixin:
    """See the module docstring."""

    def _reset_life(self) -> None:
        self.toast = ""  # a line of text over the hotbar
        self.toast_time = 0.0
        self.sleep: float | None = None  # seconds asleep (the screen fades, then it is morning)
        self.use_click = False  # the use button was just pressed (not held)
        self.hamlets_alive: set[int] = set()

    def say(self, text: str, seconds: float = 3.5) -> None:
        self.toast, self.toast_time = text, seconds

    # --- every frame, every tick --------------------------------------------------------------------------------

    def life_frame(self, dt: float) -> None:
        if self.toast_time > 0:
            self.toast_time -= dt
        if self.sleep is not None:
            self.sleep += dt
            if self.sleep >= 2.6:
                from app.arcade.games.blockcraft.play import DAY_SECONDS

                self.world_time += ((1.02 - self.tod) % 1.0) * DAY_SECONDS
                self.tod = 0.02
                self.sleep = None
                self.health = min(20.0, self.health + 4)
                self.mobs = [m for m in self.mobs if not m.hostile]  # the night's creatures are gone
                self.say("Good morning")
                self.award("sleep")

    def life_tick(self) -> None:
        t = self.tick_count
        self._random_ticks()
        if t % 40 == 7:
            self._spawn_fish()
        elif t % 40 == 17:
            self._settle_hamlets()
        if t % 10 == 3:
            self._guard()
        if t % 20 == 13:
            self.check_advancements()
        if self.dim == "isles" and t % 20 == 9:
            self._wake_wyrm()

    # --- growth -------------------------------------------------------------------------------------------------

    def _random_ticks(self) -> None:
        """A few blocks of every chunk near the player, picked at random: the ones that grow get their turn."""
        w, p = self.world, self.player
        cx0, cz0 = math.floor(p.x) >> 4, math.floor(p.z) >> 4
        g = GROW_CHUNKS
        picks = self.rng.integers(0, 16 * 16 * HEIGHT, ((2 * g + 1) ** 2, RANDOM_TICKS)).tolist()
        changes: list = []
        k = 0
        for cx in range(cx0 - g, cx0 + g + 1):
            for cz in range(cz0 - g, cz0 + g + 1):
                row = picks[k]
                k += 1
                c = w.chunks.get((cx, cz))
                if c is None:
                    continue
                raw = c.raw
                for i in row:
                    b = raw[i]
                    if b in GROWING:
                        self._grow((cx << 4) | (i >> 11), i & 127, (cz << 4) | ((i >> 7) & 15), b, changes)
        if changes:
            self._set_blocks(changes)

    def _light(self, x: int, y: int, z: int) -> int:
        sky, block = self.world.light_at(x, y, z)
        return max(sky if self.env.day > 0.4 else sky // 3, block)

    def _wet(self, x: int, y: int, z: int) -> bool:
        """Water within four blocks of a field block (at its own height)."""
        get, fluid = self.world.get, B.FLUID
        return any(fluid[get(x + dx, y, z + dz)] for dx in range(-4, 5) for dz in range(-4, 5))

    def _grow(self, x: int, y: int, z: int, b: int, changes: list) -> None:
        w, rnd = self.world, self.rng.random
        if B.WHEAT <= b < B.WHEAT + 3:
            if w.get(x, y - 1, z) == B.FARMLAND and self._light(x, y, z) >= 8 and \
                    rnd() < (0.85 if self._wet(x, y - 1, z) else 0.3):
                changes.append((x, y, z, b + 1))
        elif b == B.OAK_SAPLING:
            if self._light(x, y, z) >= 8 and rnd() < 0.3:
                self._grow_tree(x, y, z, changes)
        elif b == B.SUGAR_CANE:
            if w.get(x, y + 1, z) == B.AIR and rnd() < 0.5:
                h = 1
                while w.get(x, y - h, z) == B.SUGAR_CANE:
                    h += 1
                if h < 3:
                    changes.append((x, y + 1, z, B.SUGAR_CANE))
        elif b == B.KELP:
            if w.get(x, y + 1, z) == B.WATER and B.WATERY[w.get(x, y + 2, z)] and rnd() < 0.2:
                changes.append((x, y + 1, z, B.KELP))
        elif b == B.FARMLAND:
            if w.get(x, y + 1, z) not in B.CROPS and rnd() < 0.15 and not self._wet(x, y, z):
                changes.append((x, y, z, B.DIRT))
        elif b == B.GRASS:
            up = w.get(x, y + 1, z)
            if B.OPAQUE[up] or B.FLUID[up]:
                if rnd() < 0.5:
                    changes.append((x, y, z, B.DIRT))  # grass dies in the dark
            elif self._light(x, y + 1, z) >= 9:  # ... and spreads onto lit dirt beside it
                nx, ny, nz = (int(v) + d for v, d in zip((x, y, z), self.rng.integers(-1, 2, 3), strict=True))
                if w.get(nx, ny, nz) == B.DIRT:
                    over = w.get(nx, ny + 1, nz)
                    if not (B.OPAQUE[over] or B.FLUID[over]) and self._light(nx, ny + 1, nz) >= 8:
                        changes.append((nx, ny, nz, B.GRASS))

    def _grow_tree(self, x: int, y: int, z: int, changes: list) -> None:
        w = self.world
        offs, blocks, is_log = TREES["oak"][int(self.rng.integers(0, 6))]
        cells = []
        for (dx, dy, dz), block, log in zip(offs.tolist(), blocks.tolist(), is_log.tolist(), strict=True):
            px, py, pz = x + dx, y + dy, z + dz
            here = w.get(px, py, pz)
            free = here == B.AIR or here in B.LEAVES or here in _SOFT_PLANTS or (px, py, pz) == (x, y, z)
            if log and not free:
                return  # no room for the trunk
            if free and py < HEIGHT:
                cells.append((px, py, pz, block))
        changes.extend(cells)

    # --- using things -------------------------------------------------------------------------------------------

    def use_special(self, fresh: bool) -> bool:
        """The use button on a settler or a bed, or with a hoe or seeds in hand: True when that was it."""
        held, w = self.inv.held, self.world
        m = self.target_mob
        if m is not None and m.kind == "settler":
            if fresh:
                self._trade(m)
            return True
        t = self.target
        if t is None:
            return False
        (x, y, z), _normal, _ = t
        b = w.get(x, y, z)
        if b == B.BED and not (Qt.Key.Key_Shift in self.keys and held):
            if fresh:
                self._use_bed(x, y, z)
            return True
        if b in (B.DEEP_GATE, B.SKY_GATE) and not (Qt.Key.Key_Shift in self.keys and held):
            if fresh:  # a gate leads to its realm, and from there back to the overworld
                home = "deep" if b == B.DEEP_GATE else "isles"
                self.travel("over" if self.dim == home else home)
            return True
        if held is None:
            return False
        if B.hoe_material(held[0]) is not None:
            up = w.get(x, y + 1, z)
            if b in (B.GRASS, B.DIRT) and (up == B.AIR or up in _SOFT_PLANTS):
                changes = [(x, y, z, B.FARMLAND)]
                if up != B.AIR:
                    changes.append((x, y + 1, z, B.AIR))
                self._set_blocks(changes)
                self.award("till")
                self.sfx.play("place_gravel")
                self.swing = 1.0
                if not self.creative and self.inv.damage_held(1):
                    self.sfx.play("break_tool")
            return True
        if held[0] == B.WHEAT_SEEDS:
            if b == B.FARMLAND and w.get(x, y + 1, z) == B.AIR:
                self._set_blocks([(x, y + 1, z, B.WHEAT)])
                self.sfx.play("place_grass")
                self.swing = 1.0
                if not self.creative:
                    self.inv.take_held(1)
            return True
        return False

    def plant_ok(self, block: int, x: int, y: int, z: int, pos: tuple) -> bool:
        """May this plant (or torch) stand at (x, y, z)? ``pos``: the block that was clicked."""
        w = self.world
        below, here = w.get(x, y - 1, z), w.get(x, y, z)
        if block == B.TORCH:
            return bool(B.SOLID[below] or B.SOLID[w.get(*pos)])
        if block in (B.SEAGRASS, B.KELP):
            return here == B.WATER and bool(B.SOLID[below] or below == block)
        if here != B.AIR and here not in _SOFT_PLANTS:
            return False
        if block == B.SUGAR_CANE:
            if below == B.SUGAR_CANE:
                return True
            return below in (B.GRASS, B.DIRT, B.SAND) and any(B.FLUID[w.get(x + dx, y - 1, z + dz)]
                                                              for dx, dz in _SIDES)
        if block == B.DEAD_BUSH:
            return below in (B.SAND, B.DIRT, B.GRASS)
        if block in (B.RED_MUSHROOM, B.BROWN_MUSHROOM):
            return bool(B.SOLID[below])
        return below in (B.GRASS, B.DIRT, B.SNOWY_GRASS, B.FARMLAND)

    def _use_bed(self, x: int, y: int, z: int) -> None:
        p = self.player
        self.spawn = (x + 0.5, y + 1.0, z + 0.5)
        if self.env.day > 0.3:
            self.say("Respawn point set. You can only sleep at night")
        elif any(m.hostile and not m.dead and math.dist((m.x, m.y, m.z), (p.x, p.y, p.z)) < 10 for m in self.mobs):
            self.say("You cannot rest now: monsters are near")
        else:
            self.sleep = 0.0
            self.mouse_buttons = 0
            self.say("Respawn point set. Sleeping...", 2.6)

    def _trade(self, m: Mob) -> None:
        """Hold what a settler wants and click to trade; any other click shows the next thing they offer."""
        held, p = self.inv.held, self.player
        if not m.offers:
            return
        if held is not None:
            for give, n, get, k in m.offers:
                if held[0] == give and held[1] >= n:
                    self.inv.take_held(n)
                    left = self.inv.add(get, k)
                    if left:
                        self._spawn_drop(p.x, p.y + 1.0, p.z, I.stack(get, left))
                    self.say(f"{m.job}: {n} {B.NAMES[give]} for {k} {B.NAMES[get]}. Done!")
                    self.sfx.play("orb", 0.6)
                    self.swing = 1.0
                    self.award("trade")
                    return
        m.offer_shown = (m.offer_shown + 1) % len(m.offers)
        give, n, get, k = m.offers[m.offer_shown]
        self.say(f"{m.job}: I give {k} {B.NAMES[get]} for {n} {B.NAMES[give]}", 5.0)
        self.sfx.play("click", 0.5)

    # --- the other realms ---------------------------------------------------------------------------------------

    def travel(self, dim: str) -> None:
        """Leave this realm for another of the same world: everything the player carries and is goes along; the
        realm left is saved, the other one loads around where the player will stand."""
        from app.arcade.games.blockcraft.fluids import Water
        from app.arcade.games.blockcraft.world import World

        p, r = self.player, self.record
        self.store_world()
        r[f"pos_{self.dim}"] = [round(p.x, 2), round(p.y, 2), round(p.z, 2)]
        self.dim = r["dim"] = dim
        self.mobs, self.drops, self.falling, self.furnaces = [], [], [], {}
        self.water = Water()
        self.hamlets_alive = set()
        self.mining = self.target = self.target_mob = self.sleep = None
        self.mouse_buttons = 0
        self.particles.items.clear()
        self.world = World(int(r.get("seed", 0)), self.pool, str(r.get(realm_key("edits", dim), "") or ""),
                           bool(self.opts.get("smooth", True)), dim)
        self.gen_spawn = self.world.gen.find_spawn()
        pos = r.get(f"pos_{dim}")
        if _vec(pos):
            x, z = float(pos[0]), float(pos[2])
        elif dim == "deep":
            x, z = p.x, p.z  # straight down
        else:
            x, z = (self.gen_spawn if dim == "isles" else self.spawn)[::2]
        p.x, p.z = x, z
        self.gen_at = (x, z)
        self.gen_progress = 0.0
        self.gen_label = f"Travelling to {REALM_NAMES[dim]}"
        self.gen_since = time.perf_counter()
        self.travelling = True
        self.screen, self.overlay = "generating", None
        self.set_mouse_captured(False)

    def _arrive(self) -> None:
        """The other realm stands: put the player where they were last time there, or at a new gate."""
        p, r = self.player, self.record
        pos = r.get(f"pos_{self.dim}")
        if _vec(pos):
            p.x, p.y, p.z = float(pos[0]), float(pos[1]), float(pos[2])
        else:
            x, y, z = self._landing(math.floor(p.x), math.floor(p.z))
            p.x, p.y, p.z = x + 0.5, float(y), z + 0.5
            gate = {"deep": B.DEEP_GATE, "isles": B.SKY_GATE}.get(self.dim)
            if gate is not None:  # the way back, under the player's feet
                self._set_blocks([(x, y - 1, z, gate), (x, y, z, B.AIR), (x, y + 1, z, B.AIR)])
        p.vx = p.vy = p.vz = 0.0
        p.fall = 0.0
        p.flying = p.flying and self.creative
        self._unbury(p)
        p.remember()
        self._load_furnaces(r)
        self.env.realm = self.dim
        self.env.set_time(self.tod)
        self.travelling = False
        self.screen, self.overlay, self.acc = "game", None, 0.0
        self.capture_if_playing()
        self.progress_changed.emit()
        self.say(f"You enter {REALM_NAMES[self.dim]}", 4.5)
        self.award("deep" if self.dim == "deep" else ("isles" if self.dim == "isles" else ""))

    def _landing(self, x: int, z: int) -> tuple[int, int, int]:
        """Somewhere to stand on arriving in a realm for the first time: (x, y of the feet, z)."""
        w = self.world
        if self.dim != "deep":
            sx, sy, sz = self.gen_spawn if self.dim == "isles" else self.spawn
            return math.floor(sx), int(sy), math.floor(sz)
        get, solid = w.get, B.SOLID
        for radius in range(0, 32, 2):  # the nearest dry floor with room above it
            for dx in range(-radius, radius + 1, 2):
                for dz in ((-radius, radius) if abs(dx) != radius else range(-radius, radius + 1, 2)):
                    for y in range(w.sea + 2, 92):
                        if solid[get(x + dx, y - 1, z + dz)] and get(x + dx, y, z + dz) == B.AIR and \
                                get(x + dx, y + 1, z + dz) == B.AIR:
                            return x + dx, y, z + dz
        y = 64  # nothing but rock or lava: a ledge of its own
        self._set_blocks([(x + dx, y - 1, z + dz, B.DEEPSTONE) for dx in (-1, 0, 1) for dz in (-1, 0, 1)] +
                         [(x + dx, y + k, z + dz, B.AIR) for dx in (-1, 0, 1) for dz in (-1, 0, 1) for k in (0, 1, 2)])
        return x, y, z

    def _load_furnaces(self, r: dict) -> None:
        self.furnaces = {}
        for f in r.get(realm_key("furn", self.dim), []) or []:
            try:
                pos = (int(f[0]), int(f[1]), int(f[2]))
                fur = I.Furnace()
                fur.load(f[3:])
                if B.FURNACE <= self.world.get(*pos) < B.FURNACE + 8:
                    self.furnaces[pos] = fur
            except (TypeError, ValueError, IndexError):
                continue

    def _wake_wyrm(self) -> None:
        """The Sky Wyrm rises when the player steps onto the middle of the middle isle (until it is beaten)."""
        from app.arcade.games.blockcraft.realms import ISLE_TOP

        p = self.player
        if self.record.get("wyrm_dead") or math.hypot(p.x, p.z) > 34 or any(m.kind == "wyrm" for m in self.mobs):
            return
        wyrm = Mob("wyrm", 0.5, ISLE_TOP + 34.0, 30.5, self.rng)
        wyrm.home = (0.5, ISLE_TOP + 1.0, 0.5, 0)
        self.mobs.append(wyrm)
        self.say("The Sky Wyrm wakes!", 5.0)
        self.sfx.play("guardian")

    # --- who appears where --------------------------------------------------------------------------------------

    def _spawn_fish(self) -> None:
        p, w = self.player, self.world
        if sum(1 for m in self.mobs if m.kind == "fish") >= MAX_FISH:
            return
        ang, r = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(8, 30)
        x, z = math.floor(p.x + math.cos(ang) * r), math.floor(p.z + math.sin(ang) * r)
        for y in range(SEA - 1, SEA - 14, -2):
            if w.get(x, y, z) == B.WATER and w.get(x, y - 1, z) == B.WATER:
                for _ in range(int(self.rng.integers(2, 5))):
                    self.mobs.append(Mob("fish", x + self.rng.uniform(0.2, 0.8), y - self.rng.uniform(0.2, 0.9),
                                         z + self.rng.uniform(0.2, 0.8), self.rng))
                return

    def _settle_hamlets(self) -> None:
        """The people (one a hut) and the guardian of every hamlet near the player, once its land stands."""
        p, w = self.player, self.world
        jobs = list(JOBS)
        for site in w.gen.hamlets(p.x - 72, p.z - 72, p.x + 72, p.z + 72):
            sid = site["id"]
            if sid in self.hamlets_alive or (site["x"] >> 4, site["z"] >> 4) not in w.chunks:
                continue
            self.hamlets_alive.add(sid)
            home = (site["x"], site["y"], site["z"], sid)
            for k, (hx, hy, hz, door) in enumerate(site["huts"]):
                dx, dz = ((-3, 0), (3, 0), (0, -3), (0, 3))[door]
                m = Mob("settler", hx + dx + 0.5, hy + 1.0, hz + dz + 0.5, self.rng)
                m.home, m.leash = home, 14.0
                m.job = jobs[(sid + k) % len(jobs)]
                m.offers = JOBS[m.job]
                self._unbury(m)
                self.mobs.append(m)
            keeper = Mob("guardian", site["x"] + 3.5, site["y"] + 1.0, site["z"] + 3.5, self.rng)
            keeper.home, keeper.leash = home, 16.0
            self._unbury(keeper)
            self.mobs.append(keeper)
        if self.tick_count % 400 == 17 and self.hamlets_alive:  # hamlets left far behind fill again on return
            homes = {m.home[3] for m in self.mobs if m.home is not None}
            near = {s["id"] for s in w.gen.hamlets(p.x - 110, p.z - 110, p.x + 110, p.z + 110)}
            self.hamlets_alive = {i for i in self.hamlets_alive if i in homes or i in near}

    def _guard(self) -> None:
        """Every guardian picks the nearest monster within reach of its post."""
        monsters = [m for m in self.mobs if m.hostile and not m.dead]
        for g in self.mobs:
            if g.kind != "guardian":
                continue
            g.foe = min((m for m in monsters if math.hypot(m.x - g.x, m.z - g.z) < 14),
                        key=lambda m: math.hypot(m.x - g.x, m.z - g.z), default=None)


__all__ = ["JOBS", "LifeMixin", "material_of"]
