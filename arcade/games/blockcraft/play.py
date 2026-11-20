"""BlockCraft gameplay: a world session (player, inventory, survival stats, mobs, dropped items, furnaces, time of
day), the 20 Hz game tick, breaking and placing blocks, and the world's saved record."""

from __future__ import annotations

import logging
import math
import random
import time

import numpy as np
from PySide6.QtCore import Qt

from app.arcade.games.blockcraft import blocks as B
from app.arcade.games.blockcraft import items as I
from app.arcade.games.blockcraft.entities import (ANIMALS, MOB_DAMAGE, TICK, BlockEntity, ItemEntity, Mob, Particles,
                                                  Player)
from app.arcade.games.blockcraft.fluids import SOFT, Water
from app.arcade.games.blockcraft.life import realm_key
from app.arcade.games.blockcraft.render import Env
from app.arcade.games.blockcraft.sounds import material_of
from app.arcade.games.blockcraft.world import World

log = logging.getLogger(__name__)
DAY_SECONDS = 1200.0  # a whole day and night, like the real one
REACH = 4.5
CREATIVE_REACH = 5.0
MAX_MOBS = 60
MAX_ANIMALS = 14  # near the player
MAX_MONSTERS = 8
ENTER_CHUNKS = 3  # the land that must stand around the player before the world opens
SKY_LIMIT = 220.0  # no flying above this
REPLACEABLE = {B.AIR, B.WATER, B.TALL_GRASS, B.FERN, B.LAVA, B.DEAD_BUSH, B.SEAGRASS} | \
    set(range(B.FLOW + 1, B.FLOW + 8))
PLANTS = {B.TORCH, B.TALL_GRASS, B.FERN, B.DEAD_BUSH, B.SUGAR_CANE, B.OAK_SAPLING, B.RED_MUSHROOM, B.BROWN_MUSHROOM,
          B.SEAGRASS, B.KELP, *B.FLOWERS, *B.CROPS}  # they go when the block under them goes
WATER_STEP = 5  # game ticks between two steps of running water
ORE_XP = {B.COAL_ORE: (0, 2), B.DIAMOND_ORE: (3, 7), B.REDSTONE_ORE: (1, 5)}


def xp_for_level(level: int) -> int:
    """Experience points from one level to the next (the Java Edition formula)."""
    if level < 16:
        return 2 * level + 7
    if level < 31:
        return 5 * level - 38
    return 9 * level - 158


class PlayMixin:
    """The world being played (mixed into the game widget)."""

    # --- starting and stopping ----------------------------------------------------------------------------------

    def _reset_session(self) -> None:
        self.world: World | None = None
        self.gen_progress = 0.0
        self.gen_label = ""
        self.gen_at = (0.0, 0.0)  # where the world is being loaded (x, z)
        self.gen_since = 0.0
        self.player: Player | None = None
        self.inv = I.Inventory()
        self.mode = "survival"
        self.mobs: list[Mob] = []
        self.drops: list[ItemEntity] = []
        self.falling: list[BlockEntity] = []
        self.particles = Particles()
        self.furnaces: dict[tuple, I.Furnace] = {}
        self.water = Water()
        self.tod = 0.02
        self.world_time = 0.0
        self.health = 20.0
        self.food = 20
        self.saturation = 5.0
        self.exhaustion = 0.0
        self.xp_total = 0
        self.air = 300
        self.fire = 0.0
        self.hurt_cool = 0.0
        self.hurt_anim = 0.0
        self.regen_timer = 0.0
        self.spawn = (0.0, 0.0, 0.0)
        self.acc = 0.0
        self.tick_count = 0
        self.mining = None  # (x, y, z) being broken
        self.mine_progress = 0.0
        self.mine_cool = 0.0
        self.place_cool = 0.0
        self.eat_time = 0.0
        self.swing = 0.0  # 0..1 hand swing
        self.equip = 1.0  # 0..1 the held item coming up
        self.last_held = None
        self.held_name_time = 0.0
        self.target = None  # ((x, y, z), normal, dist) the block looked at
        self.target_mob: Mob | None = None
        self.last_space = -1.0
        self.last_w = -1.0
        self.sprint_key = False
        self.step_dist = 0.0
        self.was_in_water = False
        self.left_water = -999  # the tick of the last splash on coming out of the water
        self.edit_time = 0.0
        self.rng = np.random.default_rng()
        self.music_timer = 20.0
        self.music_queue: list[tuple[float, int]] = []
        self.score = 0
        self.record: dict = {}
        self.dim = "over"  # the realm being played
        self.travelling = False
        self._reset_life()

    def start_world(self, record: dict) -> None:
        """Begin loading a saved (or new) world record; the game starts when the land around the player stands."""
        self._reset_session()
        self.record = record
        if not any(r is record for r in self.worlds):
            self.worlds.insert(0, record)
        self.mode = "creative" if record.get("mode") == "creative" else "survival"
        self.title_world = None  # the title screen's land makes room
        self.start_helpers()
        self.dim = record.get("dim") if record.get("dim") in ("deep", "isles") else "over"
        self.world = World(int(record.get("seed", 0)), self.pool,
                           str(record.get(realm_key("edits", self.dim), "") or ""),
                           bool(self.opts.get("smooth", True)), self.dim)
        self.gen_spawn = self.world.gen.find_spawn()
        pos = record.get("player")
        at = pos if _is_vec(pos) else self.gen_spawn
        self.gen_at = (float(at[0]), float(at[2]))
        self.gen_progress = 0.0
        self.gen_label = "Building terrain"
        self.gen_since = time.perf_counter()
        self.screen = "generating"
        self.overlay = None

    def step_generation(self) -> None:
        """Enter the world once the land around the player is built."""
        w = self.world
        if w is None:
            self.go_title()
            return
        w.pump(self.gen_at[0], self.gen_at[1], ENTER_CHUNKS + 1)
        self.gen_progress = w.loaded_fraction(self.gen_at[0], self.gen_at[1], ENTER_CHUNKS)
        if self.gen_progress >= 1.0:
            if self.travelling:
                self._arrive()
            else:
                self._enter_world()
        elif time.perf_counter() - self.gen_since > 90:  # the helpers are stuck: back to the title
            log.warning("BlockCraft world did not load")
            self._reset_session()
            self.go_title()

    def finish_generation(self) -> None:
        """Wait for the world and enter it (tests)."""
        t0 = time.perf_counter()
        while self.screen == "generating" and time.perf_counter() - t0 < 120:
            self.step_generation()
            time.sleep(0.002)

    def _enter_world(self) -> None:
        r = self.record
        w = self.world
        sx, sy, sz = self.gen_spawn
        spawn = r.get("spawn")
        self.spawn = tuple(float(v) for v in spawn[:3]) if _is_vec(spawn) else (sx, sy, sz)
        p = r.get("player")
        if isinstance(p, list) and len(p) >= 5:
            try:
                self.player = Player(float(p[0]), float(p[1]), float(p[2]))
                self.player.yaw, self.player.pitch = float(p[3]), float(p[4])
                self.health = float(p[5]) if len(p) > 5 else 20.0
                self.food = int(p[6]) if len(p) > 6 else 20
                self.saturation = float(p[7]) if len(p) > 7 else 5.0
                self.xp_total = int(p[8]) if len(p) > 8 else 0
                self.air = int(p[9]) if len(p) > 9 else 300
                self.player.flying = bool(p[10]) if len(p) > 10 else False
            except (TypeError, ValueError):
                self.player = None
        if self.player is None or not all(math.isfinite(v) for v in (self.player.x, self.player.y, self.player.z)):
            self.player = Player(*self.spawn)
            self.player.yaw = 0.0
        self._unbury(self.player)
        self.player.remember()
        if self.health <= 0:
            self.health = 20.0
        self.inv.load(r.get("inv", []))
        self.inv.load_armor(r.get("armor"))
        self.inv.selected = int(r.get("sel", 0)) % 9 if isinstance(r.get("sel", 0), int) else 0
        self.tod = float(r.get("tod", 0.02)) % 1.0
        self.world_time = float(r.get("time", 0.0))
        self._load_furnaces(r)
        mobs = r.get("mobs")
        if isinstance(mobs, list) and r.get("visited"):
            for m in mobs:
                try:
                    kind, x, y, z, yaw, hp = m[0], float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])
                except (TypeError, ValueError, IndexError):
                    continue
                if kind in ANIMALS and hp > 0:
                    mob = Mob(kind, x, y, z, self.rng)
                    mob.yaw, mob.health = yaw, hp
                    self.mobs.append(mob)
        else:
            for _ in range(8):
                self._spawn_animals(8.0, 44.0)
        self.record["visited"] = True
        self.screen = "game"
        self.overlay = None
        self.acc = 0.0
        self.env = Env(self.tod, self.time, self.dim)
        self.capture_if_playing()
        self.progress_changed.emit()

    def _unbury(self, body) -> None:  # noqa: ANN001 - a Body
        """Lift a body out of anything solid (a tree grown on the spawn point, a block placed there)."""
        w = self.world
        for _ in range(w.height):
            box = body.box()
            if not w.solid_boxes(box[0] + 0.01, box[1] + 0.01, box[2] + 0.01, box[3] - 0.01, box[4] - 0.01,
                                 box[5] - 0.01):
                break
            body.y = math.floor(body.y) + 1.0

    def _spawn_animals(self, near: float = 24.0, far: float = 60.0) -> None:
        """A little herd on lit grass somewhere around the player (never right beside them)."""
        p, w = self.player, self.world
        if sum(1 for m in self.mobs if not m.hostile) >= MAX_ANIMALS:
            return
        ang = self.rng.uniform(0, 2 * math.pi)
        r = self.rng.uniform(near, far)
        x, z = p.x + math.cos(ang) * r, p.z + math.sin(ang) * r
        kind = ANIMALS[int(self.rng.integers(0, len(ANIMALS)))]
        for _ in range(int(self.rng.integers(2, 5))):
            mx, mz = x + self.rng.uniform(-3, 3), z + self.rng.uniform(-3, 3)
            bx, bz = math.floor(mx), math.floor(mz)
            top = w.top_y(bx, bz)
            if top < 0 or w.get(bx, top, bz) not in (B.GRASS, B.SNOWY_GRASS) or w.light_at(bx, top + 1, bz)[0] < 9:
                continue
            self.mobs.append(Mob(kind, mx, top + 1.0, mz, self.rng))

    def leave_world(self) -> None:
        """Save the world's record and unload it."""
        self.store_world()
        self._reset_session()
        self.set_mouse_captured(False)

    # --- the saved record ---------------------------------------------------------------------------------------

    def store_world(self) -> None:
        """Write the world being played into its saved record."""
        if self.world is None or not self.world.ready or self.player is None:
            return
        p = self.player
        r = self.record
        r[realm_key("edits", self.dim)] = self.world.encode_edits()
        r["dim"] = self.dim
        r["player"] = [round(p.x, 3), round(p.y, 3), round(p.z, 3), round(p.yaw % 360, 2), round(p.pitch, 2),
                       round(self.health, 1), int(self.food), round(self.saturation, 1), int(self.xp_total),
                       int(self.air), bool(p.flying)]
        r["spawn"] = [round(v, 2) for v in self.spawn]
        r["inv"] = self.inv.to_json()
        r["sel"] = self.inv.selected
        r["tod"] = round(self.tod, 5)
        r["time"] = round(self.world_time, 1)
        r[realm_key("furn", self.dim)] = [[*pos, *f.to_json()] for pos, f in self.furnaces.items()]
        if self.dim == "over":
            r["mobs"] = [m.to_json() for m in self.mobs if m.kind in ANIMALS and not m.dead][:MAX_MOBS]
        r["armor"] = self.inv.armor_json()
        r["last"] = int(time.time())
        r["visited"] = True
        r["fmt"] = 2

    # --- the tick -----------------------------------------------------------------------------------------------

    @property
    def creative(self) -> bool:
        return self.mode == "creative"

    @property
    def day_number(self) -> int:
        return int((self.world_time + DAY_SECONDS * 0.02) // DAY_SECONDS) + 1

    def daylight(self) -> float:
        return self.env.day if hasattr(self, "env") else 1.0

    def update_world(self, dt: float) -> None:
        """Per frame: mouse look, mining / placing / eating, then the fixed 20 Hz ticks."""
        p = self.player
        if p is None:
            return
        self.world.pump(p.x, p.z, int(self.opts.get("chunks", 10)))
        if self.mouse_captured:
            dx, dy = self.take_mouse_delta()
            # (the arcade's Controls already applied the player's sensitivity and invert-Y)
            p.yaw = (p.yaw + dx * 0.15) % 360
            p.pitch = max(-90.0, min(90.0, p.pitch + dy * 0.15))
        else:
            self.take_mouse_delta()
        paused = self.overlay in ("menu", "options")
        if paused:
            return
        self.swing = max(0.0, self.swing - dt * 3.3) if self.swing > 0 else 0.0
        held = self.inv.held[0] if self.inv.held else 0
        if held != self.last_held:
            self.last_held = held
            self.equip = 0.0
            self.held_name_time = 2.0 if held else 0.0
        self.equip = min(1.0, self.equip + dt * 5)
        self.held_name_time = max(0.0, self.held_name_time - dt)
        self.life_frame(dt)
        self.hurt_anim = max(0.0, self.hurt_anim - dt * 2.5)
        if self.overlay is None and self.health > 0 and self.sleep is None:
            self._update_target()
            self._update_hands(dt)
        self.acc += dt
        n = 0
        while self.acc >= TICK and n < 5:
            self.acc -= TICK
            self.game_tick()
            n += 1
        if n == 5:
            self.acc = 0.0
        self.particles.tick(self.world, dt)
        self.env.time = self.time
        self.env.set_time(self.tod)

    def game_tick(self) -> None:
        p, w = self.player, self.world
        self.tick_count += 1
        playing = self.overlay is None or self.overlay in ("inventory", "crafting", "furnace", "dead")
        keys = self.keys if self.overlay is None and self.health > 0 and self.sleep is None else set()
        forward = (Qt.Key.Key_W in keys) - (Qt.Key.Key_S in keys)
        strafe = (Qt.Key.Key_A in keys) - (Qt.Key.Key_D in keys)
        jump = Qt.Key.Key_Space in keys
        sneak = Qt.Key.Key_Shift in keys
        sprint = (Qt.Key.Key_Control in keys or self.sprint_key) and (self.creative or self.food > 6)
        if forward <= 0:
            self.sprint_key = False
        was_ground = p.on_ground
        p.tick(w, forward, strafe, jump, sneak, sprint, self.creative)
        if p.y > SKY_LIMIT:  # no flying to the clouds
            p.y = SKY_LIMIT
            p.vy = min(p.vy, 0.0)
        if p.y < -12:
            if self.creative:
                self._teleport_spawn()
            else:
                self.damage(4, "void")
        # sounds of moving
        if p.in_water and not self.was_in_water and p.vy < -0.15:
            self.sfx.play("splash")
            self.particles.puff(p.x, p.y + 0.3, p.z, (200, 220, 255), 12, 0.08)
        elif self.was_in_water and not p.in_water and p.vy > 0.1 and self.tick_count - self.left_water > 40:
            self.left_water = self.tick_count  # climbing or jumping out (not bobbing at the surface)
            self.sfx.play("splash_out", 0.6)
        self.was_in_water = p.in_water
        if p.on_ground and not p.sneaking:
            self.step_dist += math.hypot(p.x - p.px, p.z - p.pz)
            if self.step_dist > 1.6:
                self.step_dist = 0.0
                below = w.get(math.floor(p.x), math.floor(p.y - 0.2), math.floor(p.z))
                if below not in (B.AIR, B.BORDER):
                    mat = material_of(below)
                    self.sfx.play(f"step_grass{int(self.rng.integers(1, 3))}" if mat == "grass" and
                                  self.rng.random() < 0.66 else f"step_{mat}")
        elif p.in_water and (forward or strafe) and self.tick_count % 14 == 0:
            self.sfx.play("swim")
        if p.on_ground and not was_ground and p.landed_fall > 0:
            under = w.get(math.floor(p.x), math.floor(p.y - 0.2), math.floor(p.z))
            fall = p.landed_fall * (0.0 if under == B.CLOUD else (0.2 if under == B.HAY else 1.0))  # soft landings
            if fall > 3 and not self.creative:
                self.damage(math.ceil(fall - 3), "fall")
            elif p.landed_fall > 1.2:
                below = w.get(math.floor(p.x), math.floor(p.y - 0.2), math.floor(p.z))
                if below not in (B.AIR, B.BORDER):
                    self.sfx.play(f"step_{material_of(below)}")
        if not self.creative:
            self._survival_tick(forward, jump)
        self._mob_tick()
        self._drops_tick()
        self._falling_tick()
        self._furnace_tick()
        if self.tick_count % WATER_STEP == 0:
            self._water_tick()
        self.life_tick()
        self._music_tick()
        self.tod = (self.tod + TICK / DAY_SECONDS) % 1.0
        self.world_time += TICK
        if self.hurt_cool > 0:
            self.hurt_cool -= TICK
        del playing

    def _teleport_spawn(self) -> None:
        p = self.player
        p.x, p.y, p.z = self.spawn if self.dim == "over" else self.gen_spawn
        p.vx = p.vy = p.vz = 0.0
        p.fall = 0.0
        p.remember()

    def _survival_tick(self, forward: float, jump: bool) -> None:
        p = self.player
        moved = math.hypot(p.x - p.px, p.z - p.pz)
        if p.sprinting:
            self.exhaustion += 0.1 * moved
        if p.jumped:
            self.exhaustion += 0.2 if p.sprinting else 0.05
        if self.exhaustion >= 4:
            self.exhaustion -= 4
            if self.saturation > 0:
                self.saturation = max(0.0, self.saturation - 1)
            else:
                self.food = max(0, self.food - 1)
        self.regen_timer += TICK
        if self.food >= 18 and self.health < 20 and self.regen_timer >= 4:
            self.regen_timer = 0.0
            self.health = min(20.0, self.health + 1)
            self.exhaustion += 6
        elif self.food <= 0 and self.regen_timer >= 4:
            self.regen_timer = 0.0
            if self.health > 1:
                self.damage(1, "starve")
        elif self.regen_timer >= 4:
            self.regen_timer = 0.0
        # air
        eye = self.eye()
        if B.WATERY[self.world.get(math.floor(eye[0]), math.floor(eye[1]), math.floor(eye[2]))]:
            self.air -= 1
            if self.air <= -20:
                self.air = 0
                self.damage(2, "drown")
        else:
            self.air = min(300, self.air + 5)
        if p.in_lava:
            self.fire = 8.0
            if self.tick_count % 10 == 0:
                self.damage(4, "lava")
        if self.fire > 0:
            if p.in_water:
                self.fire = 0.0
            else:
                self.fire -= TICK
                if self.tick_count % 20 == 0:
                    self.damage(1, "fire")
                if self.tick_count % 8 == 0:
                    self.sfx.play("fire")

    def damage(self, amount: float, cause: str = "") -> None:
        if self.creative and cause != "void" or self.health <= 0:
            return
        if self.hurt_cool > 0 and cause not in ("void", "drown", "starve"):
            return
        if cause in ("mob", "explosion"):  # armour takes 4 % off a blow for each of its points, and wears
            points = self.inv.armor_points()
            if points:
                amount *= 1 - 0.04 * points
                self.inv.wear_armor()
        if self.sleep is not None:
            self.sleep = None
        self.health -= amount
        self.hurt_cool = 0.5
        self.hurt_anim = 1.0
        self.exhaustion += 0.1
        self.sfx.play("hurt")
        if self.health <= 0:
            self.health = 0.0
            self._die()

    def _die(self) -> None:
        p = self.player
        self.score = self.xp_total
        for i, s in enumerate(self.inv.slots):
            if s is not None:
                self._spawn_drop(p.x, p.y + 1.0, p.z, s, scatter=0.3)
                self.inv.slots[i] = None
        for i, s in enumerate(self.inv.armor):
            if s is not None:
                self._spawn_drop(p.x, p.y + 1.0, p.z, s, scatter=0.3)
                self.inv.armor[i] = None
        self.xp_total = 0
        self.mining = None
        self.overlay = "dead"
        self.set_mouse_captured(False)

    def respawn(self) -> None:
        self.health = 20.0
        self.food = 20
        self.saturation = 5.0
        self.air = 300
        self.fire = 0.0
        self.exhaustion = 0.0
        if self.dim != "over":  # home is in the overworld
            self.record["pos_over"] = list(self.spawn)
            self.overlay = None
            self.travel("over")
            return
        self._teleport_spawn()
        self.overlay = None
        self.capture_if_playing()

    # --- mobs, drops, falling blocks, furnaces ------------------------------------------------------------------

    def _mob_tick(self) -> None:
        p, w = self.player, self.world
        day = self.env.day if hasattr(self, "env") else 1.0
        keep = []
        chunks = w.chunks
        for m in self.mobs:
            d = math.hypot(m.x - p.x, m.z - p.z)
            if m.hostile and (d > 72 or m.age > 600):
                continue
            if (math.floor(m.x) >> 4, math.floor(m.z) >> 4) not in chunks or (d > 96 and not m.hostile) or \
                    (d > 48 and m.kind == "fish"):
                continue  # walked (or was left) out of the loaded land
            if d < 56 or m.dead:
                for ev in m.tick(w, p, day):
                    if ev == "smash" and m.foe is not None:  # a guardian's blow
                        m.foe.damage(14, m.x, m.z)
                        if d < 24:
                            self.sfx.play("attack", max(0.2, 1 - d / 24))
                    elif ev == "attack" and self.health > 0 and not self.creative:
                        self.damage(MOB_DAMAGE.get(m.kind, 2), "mob")
                        if m.kind == "cinder":
                            self.fire = max(self.fire, 3.0)
                        if self.health > 0:
                            dx, dz = p.x - m.x, p.z - m.z
                            dd = math.hypot(dx, dz) or 1.0
                            p.vx += dx / dd * 0.4
                            p.vz += dz / dd * 0.4
                            p.vy = 0.3
                    elif ev == "say" and d < 16:
                        self.sfx.play(m.kind, max(0.15, 1 - d / 16))
            if m.dead and m.death > 1.0:
                self.particles.puff(m.x, m.y + 0.3, m.z, (235, 235, 235), 14, 0.15)
                self._mob_drops(m)
                continue
            keep.append(m)
        self.mobs = keep
        if self.tick_count % 50 == 25:
            self._spawn_animals()
        # night: zombies come out of the dark
        if not self.creative and self.dim != "isles" and self.tick_count % 40 == 0:
            hostile = sum(1 for m in self.mobs if m.hostile)
            if hostile < MAX_MONSTERS and (day < 0.35 or self.rng.random() < 0.15):
                self._try_spawn_zombie(day)

    def _try_spawn_zombie(self, day: float) -> None:
        p, w = self.player, self.world
        for _ in range(6):
            ang = self.rng.uniform(0, 2 * math.pi)
            r = self.rng.uniform(20, 38)
            x, z = math.floor(p.x + math.cos(ang) * r), math.floor(p.z + math.sin(ang) * r)
            # a dark floor: the surface at night, or a cave
            top = w.top_y(x, z)
            ys = [top + 1] if day < 0.35 else []
            ys += [int(p.y) + dy for dy in (-6, -3, 0, 3)]
            for y in ys:
                if not (1 <= y < w.height - 2):
                    continue
                if w.get(x, y, z) != B.AIR or w.get(x, y + 1, z) != B.AIR or not B.SOLID[w.get(x, y - 1, z)]:
                    continue
                sky, block = w.light_at(x, y, z)
                if block == 0 and sky * self.env.daylight < 7:  # only where no lamp reaches
                    roll = self.rng.random()
                    kind = "zombie" if roll < 0.55 else ("spider" if roll < 0.82 else "slime")
                    if self.dim == "deep":
                        kind = "cinder" if roll < 0.55 else ("slime" if roll < 0.8 else "spider")
                    self.mobs.append(Mob(kind, x + 0.5, float(y), z + 0.5, self.rng))
                    return

    def _mob_drops(self, m: Mob) -> None:
        from app.arcade.games.blockcraft.entities import MOB_DROPS

        for item, lo, hi in MOB_DROPS.get(m.kind, []):
            n = int(self.rng.integers(lo, hi + 1))
            if n > 0:
                self._spawn_drop(m.x, m.y + 0.5, m.z, I.stack(item, n))
        self.gain_xp(5 if m.hostile else int(self.rng.integers(1, 4)))
        if m.hostile:
            self.award("hunter")
        if m.kind == "wyrm":
            self.record["wyrm_dead"] = True
            self.gain_xp(150)
            self.sfx.play("levelup")
            self.say("The Sky Wyrm is beaten. The isles are quiet.", 8.0)
            self.award("wyrm")

    def _spawn_drop(self, x: float, y: float, z: float, s: list, scatter: float = 0.1, delay: float = 0.4,
                    v: tuple | None = None) -> None:
        if v is None:
            v = (self.rng.uniform(-scatter, scatter), 0.2, self.rng.uniform(-scatter, scatter))
        self.drops.append(ItemEntity(x, y, z, list(s), v[0], v[1], v[2], delay))
        if len(self.drops) > 200:
            self.drops.pop(0)

    def _drops_tick(self) -> None:
        p = self.player
        keep = []
        for d in self.drops:
            d.tick(self.world)
            if d.age > 300 or d.y < -20:
                continue
            if d.delay <= 0 and self.health > 0 and abs(d.x - p.x) < 1.3 and abs(d.z - p.z) < 1.3 and \
                    p.y - 0.5 < d.y < p.y + 2.2:
                left = self.inv.add(d.item[0], d.item[1], d.item[2])
                if left < d.item[1]:
                    self.sfx.play("pop", 0.6)
                if left <= 0:
                    continue
                d.item[1] = left
            # merge with a nearby identical item
            merged = False
            for o in keep:
                if o.item[0] == d.item[0] and o.item[2] == d.item[2] and abs(o.x - d.x) < 0.5 and \
                        abs(o.y - d.y) < 0.5 and abs(o.z - d.z) < 0.5 and \
                        o.item[1] + d.item[1] <= B.max_stack(d.item[0]):
                    o.item[1] += d.item[1]
                    merged = True
                    break
            if not merged:
                keep.append(d)
        self.drops = keep

    def _falling_tick(self) -> None:
        keep = []
        for f in self.falling:
            f.tick(self.world)
            if f.fuse:
                if f.fuse <= 0.001:
                    self.explode(f.x, f.y + 0.5, f.z, 4.0)
                    continue
            elif f.on_ground or f.age > 20:
                x, y, z = math.floor(f.x), math.floor(f.y + 0.5), math.floor(f.z)
                if self.world.get(x, y, z) in REPLACEABLE and self.world.inside(x, y, z):
                    self._set_blocks([(x, y, z, f.block)])
                else:
                    self._spawn_drop(f.x, f.y + 0.5, f.z, I.stack(f.block, 1))
                continue
            keep.append(f)
        self.falling = keep

    def _furnace_tick(self) -> None:
        w = self.world
        for pos, f in list(self.furnaces.items()):
            b = w.get(*pos)
            if not B.FURNACE <= b < B.FURNACE + 8:
                del self.furnaces[pos]
                continue
            lit = f.tick(TICK)
            facing = (b - B.FURNACE) % 4
            want = (B.FURNACE_LIT if lit else B.FURNACE) + facing
            if want != b:
                self._set_blocks([(*pos, want)])

    def _music_tick(self) -> None:
        """Calm music now and then (a piece, then a long quiet), and the sounds of the place the player is in."""
        self._ambience_tick()
        if not self.opts.get("music", True):
            if self.music_queue:
                self.music_queue.clear()
                for k in range(4):
                    self.sfx.stop(f"music{k}")
            return
        self.music_timer -= TICK
        if self.music_timer <= 0:
            piece = int(self.rng.integers(0, 4))
            self.sfx.play(f"music{piece}")
            self.music_queue = [(0.0, piece)]
            self.music_timer = float(self.rng.uniform(75, 150))

    def _ambience_tick(self) -> None:
        if self.tick_count % 20 != 11:
            return
        p, w, rnd = self.player, self.world, self.rng.random
        x, y, z = math.floor(p.x), math.floor(p.y + 1.5), math.floor(p.z)
        sky, _block = w.light_at(x, y, z)
        day = self.env.day
        if sky < 4 and (y < w.sea - 6 or self.dim == "deep"):  # underground
            if rnd() < 0.09:
                self.sfx.play(f"drip{int(rnd() * 2)}", 0.5 + 0.5 * rnd())
            elif random.random() < 0.035:  # the hollow air of a cave, now and then (not the world's dice)
                self.sfx.play("cave", 0.6 + 0.4 * random.random())
            return
        if self._eye_in_water():
            if random.random() < 0.3:
                self.sfx.play("bubbles", 0.5 + 0.5 * random.random())
            return
        if day > 0.6 and rnd() < 0.13 and w.biome_name(p.x, p.z) in ("forest", "birch_forest", "plains", "taiga"):
            self.sfx.play(f"bird{int(rnd() * 4)}", 0.35 + 0.65 * rnd())
        elif day < 0.25 and rnd() < 0.16:
            self.sfx.play("crickets", 0.5 + 0.5 * rnd())
        if rnd() < 0.2 and any(B.FLUID[w.get(x + dx, w.sea, z + dz)] for dx, dz in ((5, 0), (-5, 0), (0, 5), (0, -5),
                                                                                    (9, 9), (-9, -9))) \
                and abs(p.y - w.sea) < 8:
            self.sfx.play("shore", 0.5 + 0.5 * rnd())
        elif p.y > w.sea + 34 and rnd() < 0.12:
            self.sfx.play("wind", 0.5 + 0.5 * rnd())

    # --- looking, breaking, placing -----------------------------------------------------------------------------

    def eye(self, a: float = 1.0) -> tuple[float, float, float]:
        p = self.player
        x, y, z = p.lerp(a)
        return x, y + p.eye_height(), z

    def look_dir(self) -> tuple[float, float, float]:
        p = self.player
        y, pt = math.radians(p.yaw), math.radians(p.pitch)
        return (-math.sin(y) * math.cos(pt), -math.sin(pt), math.cos(y) * math.cos(pt))

    def _update_target(self) -> None:
        eye, d = self.eye(), self.look_dir()
        reach = CREATIVE_REACH if self.creative else REACH
        self.target = self.world.raycast(eye, d, reach)
        best = self.target[2] if self.target else reach
        self.target_mob = None
        for m in self.mobs:
            if m.dead:
                continue
            t = _ray_box(eye, d, m.box())
            if t is not None and t < min(best, 3.0 if not self.creative else 5.0):
                best = t
                self.target_mob = m

    def _update_hands(self, dt: float) -> None:
        lmb = self.mouse_buttons & 1
        rmb = self.mouse_buttons & 2
        self.mine_cool = max(0.0, self.mine_cool - dt)
        self.place_cool = max(0.0, self.place_cool - dt)
        # breaking
        if lmb and self.target_mob is None and self.target is not None:
            pos = self.target[0]
            b = self.world.get(*pos)
            if self.creative:
                if self.mine_cool <= 0:
                    self.break_block(*pos)
                    self.mine_cool = 0.25
                    self.swing = 1.0
            else:
                if pos != self.mining:
                    self.mining = pos
                    self.mine_progress = 0.0
                held = self.inv.held[0] if self.inv.held else 0
                p = self.player
                t = B.dig_time(b, held, p.on_ground or p.flying, self._eye_in_water())
                if self.mine_cool <= 0:
                    self.mine_progress += dt / t if t > 0 else 1.0
                    if self.swing < 0.2:
                        self.swing = 1.0
                    if int((self.mine_progress - dt / max(t, 1e-3)) * 4 * t) != int(self.mine_progress * 4 * t):
                        self.sfx.play(f"hit_{material_of(b)}", 0.6)
                if self.mine_progress >= 1.0:
                    self.break_block(*pos)
                    self.mining = None
                    self.mine_progress = 0.0
                    self.mine_cool = 0.25
        else:
            self.mining = None
            self.mine_progress = 0.0
        # using / placing / eating
        held = self.inv.held
        food = held is not None and held[0] in B.FOOD
        if rmb and food and (self.food < 20) and not self.creative:
            self.eat_time += dt
            if int((self.eat_time - dt) * 4) != int(self.eat_time * 4):
                self.sfx.play("eat", 0.7)
                p = self.player
                eye = self.eye()
                d = self.look_dir()
                self.particles.puff(eye[0] + d[0] * 0.5, eye[1] - 0.3, eye[2] + d[2] * 0.5,
                                    tuple(int(v) for v in B.item_sprite(held[0])[8, 8, :3]), 2, 0.06)
                del p
            if self.eat_time >= 1.6:
                self.eat_time = 0.0
                value, sat = B.FOOD[held[0]]
                self.food = min(20, self.food + value)
                self.saturation = min(float(self.food), self.saturation + sat)
                self.inv.take_held(1)
                self.sfx.play("burp")
        else:
            self.eat_time = 0.0
            if rmb and self.place_cool <= 0 and not food:
                fresh, self.use_click = self.use_click, False
                self.use_item(fresh)
                self.place_cool = 0.2

    def _eye_in_water(self) -> bool:
        e = self.eye()
        return bool(B.WATERY[self.world.get(math.floor(e[0]), math.floor(e[1]), math.floor(e[2]))])

    def attack(self) -> None:
        """Left click on a mob."""
        m = self.target_mob
        if m is None:
            return
        held = self.inv.held[0] if self.inv.held else 0
        info = B.tool_info(held)
        dmg = 1.0
        if info is not None:
            mat, kind = info
            dmg = {3: B.SWORD_DAMAGE[mat], 1: B.SWORD_DAMAGE[mat] - 1, 0: 2 + mat * 0.5, 2: 1.5 + mat * 0.5}[kind]
            if not self.creative:
                self.inv.damage_held(1 if kind == 3 else 2)
        if self.player.vy < -0.1 and not self.player.on_ground:
            dmg *= 1.5  # a critical hit
            self.particles.puff(m.x, m.y + m.height * 0.6, m.z, (255, 255, 160), 6, 0.06)
        m.damage(dmg, self.player.x, self.player.z)
        self.sfx.play("attack")
        self.sfx.play(f"{m.kind}_hurt", 0.8)  # its cry of pain (its usual voice when it has none)
        self.exhaustion += 0.1
        self.swing = 1.0

    def _set_blocks(self, changes: list) -> None:
        if self.world.set_many(changes):
            self.edit_time = self.world_time
            for c in changes:
                self.water.poke(c[0], c[1], c[2])

    def _water_tick(self) -> None:
        """Water runs on a step (see fluids.py); the plants it reaches are washed out as items."""
        changes, washed = self.water.step(self.world)
        if not changes:
            return
        self.world.set_many(changes)
        if not self.creative:
            for x, y, z, b in washed[:24]:
                for item, n in B.drops(b, 0, self.rng):
                    self._spawn_drop(x + 0.5, y + 0.3, z + 0.5, I.stack(item, n))

    def break_block(self, x: int, y: int, z: int) -> None:
        w = self.world
        b = w.get(x, y, z)
        if b in (B.AIR, B.BORDER) or (B.HARDNESS[b] < 0 and not self.creative):
            return
        if b == B.BEDROCK and not self.creative:
            return
        held = self.inv.held[0] if self.inv.held else 0
        new = B.WATER if b in (B.SEAGRASS, B.KELP) else B.AIR  # (water runs into the gap by itself: fluids.py)
        changes = [(x, y, z, new)]
        above = w.get(x, y + 1, z)
        top, yy = above, y + 1
        while top in PLANTS:  # what stood on it goes too (a whole stalk of cane or kelp)
            changes.append((x, yy, z, B.WATER if top in (B.SEAGRASS, B.KELP) else B.AIR))
            if not self.creative:
                for item, n in B.drops(top, 0, self.rng):
                    self._spawn_drop(x + 0.5, yy + 0.3, z + 0.5, I.stack(item, n))
            yy += 1
            if top not in (B.SUGAR_CANE, B.KELP) or w.get(x, yy, z) != top:
                break
        self._set_blocks(changes)
        if above in B.GRAVITY:
            self._start_fall(x, y + 1, z)
        self.particles.burst(x, y, z, b)
        self.sfx.play(f"dig_{material_of(b)}")
        if b == B.WHEAT + 3:
            self.award("harvest")
        if B.FURNACE <= b < B.FURNACE + 8:
            f = self.furnaces.pop((x, y, z), None)
            if f is not None:
                for s in f.slots:
                    if s is not None:
                        self._spawn_drop(x + 0.5, y + 0.5, z + 0.5, s, 0.15)
        if self.creative:
            return
        for item, n in B.drops(b, held, self.rng):
            self._spawn_drop(x + 0.5, y + 0.3, z + 0.5, I.stack(item, n), 0.1)
        if b in ORE_XP and B.drops(b, held, np.random.default_rng(0)):
            lo, hi = ORE_XP[b]
            self.gain_xp(int(self.rng.integers(lo, hi + 1)))
        if B.HARDNESS[b] > 0 and self.inv.damage_held(2 if B.tool_info(held) and B.tool_info(held)[1] == 3 else 1):
            self.sfx.play("break_tool")
        self.exhaustion += 0.005

    def _start_fall(self, x: int, y: int, z: int) -> None:
        """Sand and gravel with nothing below fall (the whole column above too)."""
        w = self.world
        changes = []
        while w.get(x, y, z) in B.GRAVITY and w.get(x, y - 1, z) in REPLACEABLE | {B.AIR}:
            self.falling.append(BlockEntity(x + 0.5, float(y), z + 0.5, w.get(x, y, z)))
            changes.append((x, y, z, B.AIR))
            y += 1
            if w.get(x, y, z) not in B.GRAVITY:
                break
        if changes:
            self._set_blocks(changes)

    def use_item(self, fresh: bool = True) -> None:
        """Right click: trade, sleep, till, sow, fill or pour a bucket, open a crafting table / furnace, light TNT,
        or place the held block (``fresh``: the button was just pressed, not held)."""
        t = self.target
        held = self.inv.held
        if self.use_special(fresh):
            return
        if held is not None and held[0] in (B.BUCKET, B.WATER_BUCKET):
            if fresh:
                self._use_bucket(held[0])
            return
        if t is not None:
            pos, normal, _ = t
            b = self.world.get(*pos)
            if b == B.CRAFTING_TABLE and not (Qt.Key.Key_Shift in self.keys and held):
                self.open_overlay("crafting")
                return
            if B.FURNACE <= b < B.FURNACE + 8 and not (Qt.Key.Key_Shift in self.keys and held):
                self.furnace_pos = pos
                self.furnaces.setdefault(pos, I.Furnace())
                self.open_overlay("furnace")
                return
            if b == B.TNT and held is not None and held[0] == B.FLINT_STEEL:
                self._set_blocks([(*pos, B.AIR)])
                self.falling.append(BlockEntity(pos[0] + 0.5, float(pos[1]), pos[2] + 0.5, B.TNT, fuse=4.0))
                self.sfx.play("fuse")
                if not self.creative:
                    self.inv.damage_held(1)
                self.swing = 1.0
                return
            if held is not None and B.is_block(held[0]) and held[0] in B.PLACEABLE:
                self.place_block(pos, normal, held[0])

    def _use_bucket(self, item: int) -> None:
        """An empty bucket lifts a block of still water; a full one pours it out where the player looks."""
        w = self.world
        reach = CREATIVE_REACH if self.creative else REACH
        if item == B.BUCKET:
            hit = w.raycast(self.eye(), self.look_dir(), reach, liquids=True)
            if hit is None or w.get(*hit[0]) != B.WATER:
                return
            self._set_blocks([(*hit[0], B.AIR)])
            if not self.creative:
                self.inv.take_held(1)
                if self.inv.add(B.WATER_BUCKET, 1):
                    self._spawn_drop(self.player.x, self.player.y + 1.0, self.player.z, I.stack(B.WATER_BUCKET, 1))
        else:
            hit = self.target
            if hit is None:
                return
            pos, normal, _ = hit
            x, y, z = pos if SOFT[w.get(*pos)] else (pos[0] + normal[0], pos[1] + normal[1], pos[2] + normal[2])
            if not w.inside(x, y, z) or not (SOFT[w.get(x, y, z)] or B.FLUID[w.get(x, y, z)]):
                return
            self._set_blocks([(x, y, z, B.WATER)])
            if not self.creative:
                self.inv.slots[self.inv.selected] = I.stack(B.BUCKET, 1)
        self.sfx.play("splash", 0.6)
        self.swing = 1.0

    def place_block(self, pos: tuple, normal: tuple, item: int) -> bool:
        w = self.world
        target = w.get(*pos)
        if target in (B.TALL_GRASS, B.FERN, B.DEAD_BUSH, B.SEAGRASS):
            x, y, z = pos
        else:
            x, y, z = pos[0] + normal[0], pos[1] + normal[1], pos[2] + normal[2]
        if not w.inside(x, y, z) or w.get(x, y, z) not in REPLACEABLE:
            return False
        block = item
        if item == B.FURNACE:
            block = B.FURNACE + int(round(self.player.yaw / 90)) % 4
        if B.KIND[block] == B.K_CROSS and not self.plant_ok(block, x, y, z, pos):
            return False
        if B.SOLID[block]:
            box = (x, y, z, x + 1, y + 1, z + 1)
            if _box_hit(self.player.box(), box) or any(_box_hit(m.box(), box) for m in self.mobs if not m.dead):
                return False
        self._set_blocks([(x, y, z, block)])
        self.sfx.play(f"place_{material_of(block)}")
        self.swing = 1.0
        if not self.creative:
            self.inv.take_held(1)
        if block in B.GRAVITY and w.get(x, y - 1, z) in REPLACEABLE:
            self._start_fall(x, y, z)
        return True

    def pick_block(self) -> None:
        """Middle click: the looked-at block into the hand."""
        if self.target is None:
            return
        b = B.base_block(self.world.get(*self.target[0]))
        if b not in B.NAMES or b in (B.WATER, B.LAVA):
            return
        for i in range(9):
            s = self.inv.slots[i]
            if s is not None and s[0] == b:
                self.inv.selected = i
                return
        if self.creative:
            empty = next((i for i in range(9) if self.inv.slots[i] is None), self.inv.selected)
            self.inv.slots[empty] = I.stack(b, 1 if B.max_stack(b) == 1 else 64)
            self.inv.selected = empty

    def drop_held(self, whole: bool = False) -> None:
        s = self.inv.take_held(64 if whole else 1)
        if s is None:
            return
        eye, d = self.eye(), self.look_dir()
        self._spawn_drop(eye[0], eye[1] - 0.3, eye[2], s, v=(d[0] * 0.3, d[1] * 0.3 + 0.1, d[2] * 0.3), delay=2.0)
        self.swing = 1.0

    def explode(self, x: float, y: float, z: float, power: float) -> None:
        w = self.world
        r = int(math.ceil(power))
        changes = []
        drops = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx * dx + dy * dy + dz * dz > power * power * self.rng.uniform(0.55, 0.9):
                        continue
                    bx, by, bz = math.floor(x) + dx, math.floor(y) + dy, math.floor(z) + dz
                    b = w.get(bx, by, bz)
                    if b in (B.AIR, B.BORDER, B.BEDROCK, B.OBSIDIAN, B.WATER, B.LAVA) or not w.inside(bx, by, bz):
                        continue
                    if b == B.TNT:
                        self.falling.append(BlockEntity(bx + 0.5, float(by), bz + 0.5, B.TNT,
                                                        fuse=float(self.rng.uniform(0.5, 1.5))))
                    elif self.rng.random() < 0.25 and not self.creative:
                        for item, n in B.drops(b, B.tool_id(4, 0), self.rng):
                            drops.append((bx + 0.5, by + 0.5, bz + 0.5, I.stack(item, n)))
                    changes.append((bx, by, bz, B.AIR))
        if changes:
            self._set_blocks(changes)
        for d in drops[:40]:
            self._spawn_drop(*d[:3], d[3], 0.3)
        self.sfx.play("explode")
        for _ in range(4):
            self.particles.puff(x, y, z, (230, 230, 230), 10, 0.5)
        p = self.player
        dist = math.dist((p.x, p.y + 0.9, p.z), (x, y, z))
        if dist < power * 2:
            f = 1 - dist / (power * 2)
            self.damage(int((f * f + f) / 2 * 7 * power * 2 + 1) * 0.5, "explosion")
            p.vx += (p.x - x) / max(dist, 0.5) * f
            p.vy += 0.5 * f
            p.vz += (p.z - z) / max(dist, 0.5) * f
        for m in self.mobs:
            dm = math.dist((m.x, m.y + 0.5, m.z), (x, y, z))
            if dm < power * 2:
                f = 1 - dm / (power * 2)
                m.damage((f * f + f) / 2 * 7 * power * 2 + 1, x, z)

    def gain_xp(self, points: int) -> None:
        if points <= 0:
            return
        before = self.level()[0]
        self.xp_total += points
        self.sfx.play("orb", 0.6)
        if self.level()[0] > before and self.level()[0] % 5 == 0:
            self.sfx.play("levelup")

    def level(self) -> tuple[int, float]:
        """(level, progress 0..1 to the next)."""
        pts, lvl = self.xp_total, 0
        while pts >= xp_for_level(lvl):
            pts -= xp_for_level(lvl)
            lvl += 1
        return lvl, pts / xp_for_level(lvl)


def _box_hit(a: list, b: tuple) -> bool:
    return a[0] < b[3] - 1e-6 and a[3] > b[0] + 1e-6 and a[1] < b[4] - 1e-6 and a[4] > b[1] + 1e-6 and \
        a[2] < b[5] - 1e-6 and a[5] > b[2] + 1e-6


def _ray_box(o: tuple, d: tuple, box: list) -> float | None:
    tmin, tmax = 0.0, 1e9
    for k in range(3):
        if abs(d[k]) < 1e-9:
            if not box[k] <= o[k] <= box[k + 3]:
                return None
            continue
        t1, t2 = (box[k] - o[k]) / d[k], (box[k + 3] - o[k]) / d[k]
        tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
    return tmin if tmin <= tmax else None


def _is_vec(v) -> bool:  # noqa: ANN001
    try:
        return len(v) >= 3 and all(math.isfinite(float(x)) for x in v[:3])
    except (TypeError, ValueError):
        return False
