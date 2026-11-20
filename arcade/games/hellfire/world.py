"""The game world at 35 tics per second: the player (momentum, steps, weapons, use), monsters (the 1993 game's
look / chase / attack AI with pain, sound alerts and infighting), missiles, barrels, pickups, doors, switches,
lifts, secrets, damaging floors and light effects. ``events`` collects sounds and messages for the game to play."""

from __future__ import annotations

import math
import random

import numpy as np

from .info import (AMMO_MAX, AUTO_ORDER, DECOR_INFO, ITEM_INFO, MELEERANGE, MISSILE_INFO, MISSILERANGE,
                   MONSTER_INFO, MAXSTEP, PLAYER_HEIGHT, PLAYER_RADIUS, POWER_TICS, USERANGE, VIEWHEIGHT,
                   WEAPON_NAMES, WEAPON_ORDER, WEAPONS)
from .level import CELL, MONSTERS, Grid, MapDef, parse

DIRS = [(1.0, 0.0), (0.7071, -0.7071), (0.0, -1.0), (-0.7071, -0.7071), (-1.0, 0.0), (-0.7071, 0.7071),
        (0.0, 1.0), (0.7071, 0.7071)]
NODIR = 8
OPPOSITE = [4, 5, 6, 7, 0, 1, 2, 3, NODIR]
FRICTION = 0.90625
SKULLSPEED = 20.0
FLOATSPEED = 4.0
GRAVITY = 1.0
DOORSPEED = 2.0
DOORWAIT = 150
LIFTSPEED = 4.0
LIFTWAIT = 105

PICKUP_MSG = {
    "hbonus": "Picked up a health vial.", "abonus": "Picked up a shield shard.", "stim": "Picked up a stim pack.",
    "medikit": "Picked up a field medkit.", "garmor": "Picked up the combat vest.",
    "barmor": "Picked up the heavy plate armor!", "soulsphere": "Soul orb! Your spirit swells!",
    "clip": "Picked up an ammo clip.", "clip_drop": "Picked up an ammo clip.",
    "ammobox": "Picked up a crate of bullets.", "shells": "Picked up 4 buckshot shells.",
    "shellbox": "Picked up a box of buckshot.", "rocket": "Picked up a rocket.",
    "rocketbox": "Picked up a crate of rockets.", "cell": "Picked up a power cell.",
    "cellpack": "Picked up a power cell pack.", "backpack": "Picked up a heavy-duty rucksack!",
    "bluecard": "Picked up a blue access card.", "yellowcard": "Picked up a yellow access card.",
    "redcard": "Picked up a red access card.", "w_shotgun": "You got the pump shotgun!",
    "w_shotgun_drop": "You got the pump shotgun!", "w_chaingun": "You got the rotary cannon!",
    "w_launcher": "You got the rocket launcher!", "w_plasma": "You got the plasma rifle!",
    "w_bfg": "You got the HELLFIRE CANNON! Oh, yes.", "w_chaingun_drop": "You got the rotary cannon!",
    "w_saw": "A ripsaw! Find some meat!", "berserk": "Blood rage! Your fists are sledgehammers.",
    "invuln": "Aegis field! Nothing can touch you.", "suit": "Hazmat suit.",
    "scanner": "Base scanner: the whole map is yours.",
}


class Mobj:
    """A thing in the world: the player's body, a monster, missile, pickup, decoration, barrel or effect."""

    _ids = 0

    def __init__(self, kind: str, cls: str, x: float, y: float, z: float = 0.0) -> None:
        Mobj._ids += 1
        self.id = Mobj._ids
        self.kind, self.cls = kind, cls
        self.x, self.y, self.z = x, y, z
        self.momx = self.momy = self.momz = 0.0
        self.angle = 0.0
        self.radius, self.height = 20.0, 56.0
        self.hp = 0
        self.state, self.si, self.tics = "spawn", 0, 0
        self.target: Mobj | None = None
        self.source: Mobj | None = None
        self.movedir, self.movecount = NODIR, 0
        self.reactiontime = 0
        self.threshold = 0
        self.solid = self.shootable = self.countkill = False
        self.alive = True
        self.floats = self.nogravity = self.skullfly = False
        self.justattacked = self.justhit = False
        self.dropped = False
        self.removed = False
        self.ambush = False
        self.frame: object = 0  # current sprite frame key
        self.bright = False
        self.info: dict = {}


class Mover:
    def __init__(self, cell: int, kind: str, **kw) -> None:  # noqa: ANN003
        self.cell = cell
        self.kind = kind  # door / lift / floor
        self.state = kw.get("state", "idle")  # idle / up / down / wait
        self.wait = 0
        self.top = kw.get("top", 0.0)
        self.bottom = kw.get("bottom", 0.0)
        self.speed = kw.get("speed", DOORSPEED)
        self.stay = kw.get("stay", False)


class Player:
    def __init__(self) -> None:
        self.health = 100
        self.armor = 0
        self.armortype = 0
        self.ammo = {"bullets": 50, "shells": 0, "rockets": 0, "cells": 0}
        self.maxammo = dict(AMMO_MAX)
        self.backpack = False
        self.weapons = {"fist", "pistol"}
        self.weapon = "pistol"
        self.pending: str | None = None
        self.keys: set[str] = set()
        self.reset_level_state()

    def reset_level_state(self) -> None:
        self.damagecount = 0
        self.bonuscount = 0
        self.attacker: Mobj | None = None
        self.extralight = 0
        self.dead = False
        self.viewheight = VIEWHEIGHT
        self.deltaviewheight = 0.0
        self.viewz = VIEWHEIGHT
        self.bob = 0.0
        self.wstate = "raise"  # ready / fire / lower / raise
        self.wsi = 0
        self.wtics = 0
        self.wframe = 0
        self.wflash = -1
        self.powers: dict[str, int] = {}  # invuln / suit: tics left; berserk / scanner: 1 for the level
        self.wy = 96.0  # weapon lowered by this many pixels
        self.refire = 0
        self.attackdown = False
        self.usedown = False
        self.killcount = self.itemcount = self.secretcount = 0
        self.fire_time = 0
        self.got_weapon = 0
        self.flash = 0

    def to_json(self) -> dict:
        return {"health": self.health, "armor": self.armor, "armortype": self.armortype, "ammo": dict(self.ammo),
                "maxammo": dict(self.maxammo), "backpack": self.backpack, "weapons": sorted(self.weapons),
                "weapon": self.weapon, "keys": sorted(self.keys)}

    def from_json(self, d: dict) -> None:
        try:
            self.health = int(d.get("health", 100))
            self.armor = int(d.get("armor", 0))
            self.armortype = int(d.get("armortype", 0))
            for k in self.ammo:
                self.ammo[k] = int((d.get("ammo") or {}).get(k, self.ammo[k]))
                self.maxammo[k] = int((d.get("maxammo") or {}).get(k, self.maxammo[k]))
            self.backpack = bool(d.get("backpack", False))
            self.weapons = {w for w in d.get("weapons", ["fist", "pistol"]) if w in WEAPONS} | {"fist"}
            w = d.get("weapon", "pistol")
            self.weapon = w if w in self.weapons else "fist"
            self.keys = {k for k in d.get("keys", []) if k in ("blue", "yellow", "red")}
        except (TypeError, ValueError, AttributeError):
            pass


class World:
    def __init__(self, mapdef: MapDef, skill: int = 3, player: Player | None = None, seed: int = 1) -> None:
        self.mapdef = mapdef
        self.grid: Grid = parse(mapdef)
        self.init_fh = self.grid.fh.copy()
        self.init_ch = self.grid.ch.copy()
        self.skill = skill
        self.rng = random.Random(seed)
        self.player = player or Player()
        self.player.reset_level_state()
        self.player.keys = set()
        self.events: list[tuple] = []
        self.mobjs: list[Mobj] = []
        self.movers: dict[int, Mover] = {}
        self.leveltime = 0
        self.total_kills = self.total_items = self.total_secrets = 0
        self.switched: set[int] = set()
        self.secrets_found: set[int] = set()
        self.triggered: set[int] = set()
        self.exit_pending = 0
        self.exited = False
        self.secret_exit = False
        self.god = False  # tests only: the player takes no damage
        self.boss_done = False
        self.sound_zone: np.ndarray | None = None
        self._zone_dirty = True
        g = self.grid
        self.total_secrets = int(self._secret_groups())
        sx, sy = g.start
        self.pmo = Mobj("player", "player", sx, sy, float(g.fh[g.cell_at(sx, sy)]))
        self.pmo.radius, self.pmo.height = PLAYER_RADIUS, PLAYER_HEIGHT
        self.pmo.solid = self.pmo.shootable = True
        self.pmo.hp = self.player.health
        self.pmo.angle = math.radians(mapdef.angle)
        self.player.viewz = self.pmo.z + VIEWHEIGHT
        for i, t in enumerate(g.things):
            if t.kind in MONSTERS and not self._keep_monster(t, i):
                continue
            self.spawn(t.kind, t.x, t.y)
        self._init_movers()
        self.seen = np.zeros(g.w * g.h, bool)  # cells the player has seen (the automap)
        self._reveal_tic = -99

    def reveal(self, force: bool = False) -> None:
        """Mark the cells in sight of the player as seen (rays all around, stopped by walls and closed doors)."""
        if not force and self.leveltime - self._reveal_tic < 4:
            return
        self._reveal_tic = self.leveltime
        g = self.grid
        n_rays, step, n_steps = 256, 12.0, 200
        ang = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)
        d = (np.arange(1, n_steps + 1) * step)[None, :]
        xs = self.pmo.x + np.cos(ang)[:, None] * d
        ys = self.pmo.y + np.sin(ang)[:, None] * d
        cx = np.clip((xs // CELL).astype(np.int32), 0, g.w - 1)
        cy = np.clip((ys // CELL).astype(np.int32), 0, g.h - 1)
        c = cy * g.w + cx
        block = g.solid[c] | ((g.ch[c] - g.fh[c]) < 1)
        first = np.where(block.any(axis=1), block.argmax(axis=1), n_steps - 1)
        keep = np.arange(n_steps)[None, :] <= first[:, None]
        self.seen[c[keep]] = True
        self.seen[g.cell_at(self.pmo.x, self.pmo.y)] = True

    # --- setup ---------------------------------------------------------------------------------------------------

    def _keep_monster(self, t, i: int) -> bool:  # noqa: ANN001
        if MONSTER_INFO[t.kind].get("boss"):
            return True
        keep = {1: 0.6, 2: 0.6, 3: 0.85}.get(self.skill, 1.0)
        h = ((int(t.x) * 7919 + int(t.y) * 104729 + i * 31) % 1000) / 1000.0
        return h < keep

    def _secret_groups(self) -> int:
        g = self.grid
        seen = set()
        groups = 0
        for i in np.nonzero(g.secret)[0].tolist():
            if i in seen:
                continue
            groups += 1
            stack = [i]
            seen.add(i)
            while stack:
                c = stack.pop()
                cy, cx = divmod(c, g.w)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (cy + dy) * g.w + cx + dx
                    if g.in_bounds(cx + dx, cy + dy) and g.secret[n] and n not in seen:
                        seen.add(n)
                        stack.append(n)
        return groups

    def _init_movers(self) -> None:
        g = self.grid
        for i, s in enumerate(g.specs):
            kind = s.get("kind")
            if kind == "door":
                self.movers[i] = Mover(i, "door", top=s["open_height"], bottom=float(g.fh[i]),
                                       speed=DOORSPEED * (4 if s.get("fast") else 1), stay=bool(s.get("stay")))
            elif kind == "lift":
                low = s.get("low")
                if low is None:
                    cy, cx = divmod(i, g.w)
                    fl = [g.fh[(cy + dy) * g.w + cx + dx] for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                          if g.in_bounds(cx + dx, cy + dy) and not g.solid[(cy + dy) * g.w + cx + dx]
                          and g.specs[(cy + dy) * g.w + cx + dx].get("kind") != "lift"]
                    low = min(fl) if fl else g.fh[i] - 64
                self.movers[i] = Mover(i, "lift", top=float(g.fh[i]), bottom=float(low), speed=LIFTSPEED)

    def rnd(self) -> int:
        return self.rng.randrange(256)

    def spawn(self, kind: str, x: float, y: float, z: float | None = None) -> Mobj:
        g = self.grid
        fz = float(g.fh[g.cell_at(x, y)])
        if kind in MONSTER_INFO:
            info = MONSTER_INFO[kind]
            mo = Mobj(kind, "monster", x, y, fz)
            mo.info = info
            mo.radius, mo.height = float(info["radius"]), float(info["height"])
            mo.hp = info["hp"]
            mo.solid = mo.shootable = True
            mo.countkill = info.get("countkill", True)
            mo.floats = bool(info.get("float"))
            mo.nogravity = bool(info.get("nogravity"))
            if mo.floats:
                mo.z = fz + (24.0 if kind == "caco" else 16.0)
            mo.reactiontime = info["reaction"] if self.skill < 5 else 0
            mo.angle = (self.rng.randrange(8)) * math.pi / 4
            self.total_kills += 1 if mo.countkill else 0
            self.set_state(mo, "spawn")
            mo.tics = 1 + self.rng.randrange(10)
        elif kind == "barrel":
            mo = Mobj(kind, "barrel", x, y, fz)
            mo.radius, mo.height, mo.hp = 10.0, 42.0, 20
            mo.solid = mo.shootable = True
        elif kind in ITEM_INFO:
            mo = Mobj(kind, "item", x, y, fz)
            mo.radius, mo.height = 20.0, 16.0
            if ITEM_INFO[kind].get("count"):
                self.total_items += 1
        elif kind in DECOR_INFO:
            info = DECOR_INFO[kind]
            mo = Mobj(kind, "decor", x, y, fz)
            mo.radius, mo.height = float(info.get("radius", 16)), 16.0
            mo.solid = bool(info.get("solid"))
            mo.height = 56.0 if mo.solid else 16.0
        else:
            mo = Mobj(kind, "fx", x, y, fz)
            mo.tics = 4
        if z is not None:
            mo.z = z
        self.mobjs.append(mo)
        return mo

    # --- events ---------------------------------------------------------------------------------------------------

    def sound(self, name: str | None, at: Mobj | None = None) -> None:
        if not name:
            return
        vol = 1.0
        if at is not None and at is not self.pmo:
            d = math.hypot(at.x - self.pmo.x, at.y - self.pmo.y)
            if d > 1200:
                return
            vol = 1.0 if d < 200 else (1200 - d) / 1000.0
        self.events.append(("sound", name, vol))

    def message(self, text: str) -> None:
        self.events.append(("msg", text))

    # --- geometry --------------------------------------------------------------------------------------------------

    def cells_under(self, x: float, y: float, r: float) -> list[int]:
        g = self.grid
        x0, x1 = int((x - r) // CELL), int((x + r - 0.001) // CELL)
        y0, y1 = int((y - r) // CELL), int((y + r - 0.001) // CELL)
        out = []
        for cy in range(max(y0, 0), min(y1, g.h - 1) + 1):
            for cx in range(max(x0, 0), min(x1, g.w - 1) + 1):
                out.append(cy * g.w + cx)
        if x0 < 0 or y0 < 0 or x1 >= g.w or y1 >= g.h:
            out.append(-1)
        return out

    def check_position(self, mo: Mobj, x: float, y: float) -> tuple[bool, float, float, float, object]:
        """(fits, floorz, ceilingz, dropoffz, blocker): blocker is a cell index, a Mobj, or None."""
        g = self.grid
        floorz, ceilz, dropz = -1e9, 1e9, 1e9
        for c in self.cells_under(x, y, mo.radius):
            if c < 0 or g.solid[c]:
                return False, 0.0, 0.0, 0.0, c
            floorz = max(floorz, float(g.fh[c]))
            dropz = min(dropz, float(g.fh[c]))
            ceilz = min(ceilz, float(g.ch[c]))
        if mo.solid or mo.cls == "missile":
            for o in self.mobjs:
                if o is mo or not o.solid or o.removed or o is mo.source:
                    continue
                if mo.cls == "missile" and not o.shootable and o.cls != "decor":
                    continue
                blockdist = o.radius + mo.radius
                if abs(o.x - x) >= blockdist or abs(o.y - y) >= blockdist:
                    continue
                if mo.cls == "missile" or mo.skullfly:
                    if mo.z > o.z + o.height or mo.z + mo.height < o.z:
                        continue
                    return False, floorz, ceilz, dropz, o
                if o.floats or mo.floats:
                    if mo.z >= o.z + o.height or mo.z + mo.height <= o.z:
                        continue
                return False, floorz, ceilz, dropz, o
            if mo is not self.pmo and self.pmo.solid and not self.player.dead and mo.cls != "missile":
                o = self.pmo
                blockdist = o.radius + mo.radius
                if abs(o.x - x) < blockdist and abs(o.y - y) < blockdist:
                    if not (mo.z >= o.z + o.height or mo.z + mo.height <= o.z):
                        return False, floorz, ceilz, dropz, o
        return True, floorz, ceilz, dropz, None

    def try_move(self, mo: Mobj, x: float, y: float) -> tuple[bool, object]:
        ok, floorz, ceilz, dropz, blocker = self.check_position(mo, x, y)
        if not ok:
            return False, blocker
        if ceilz - floorz < mo.height:
            return False, self._door_in(x, y, mo.radius)
        if not mo.floats and mo.cls != "missile":
            if floorz - mo.z > MAXSTEP:
                return False, None
            if mo.cls == "monster" and floorz - dropz > MAXSTEP and not mo.skullfly:
                return False, None
        if mo.floats and (mo.z < floorz or mo.z + mo.height > ceilz):
            return False, None
        if mo.cls == "missile" and (mo.z < floorz or mo.z + mo.height > ceilz):
            return False, None
        mo.x, mo.y = x, y
        if not mo.floats and mo.cls != "missile":
            if mo is not self.pmo or mo.z <= floorz:
                if mo.z < floorz:
                    if mo is self.pmo:
                        self.player.viewheight -= floorz - mo.z
                        self.player.deltaviewheight = (VIEWHEIGHT - self.player.viewheight) / 8.0
                    mo.z = floorz
            if mo is not self.pmo and mo.z > floorz:
                mo.z = floorz
        return True, None

    def _door_in(self, x: float, y: float, r: float) -> int | None:
        for c in self.cells_under(x, y, r):
            if c >= 0 and self.grid.specs[c].get("kind") == "door":
                return c
        return None

    def floor_ceiling(self, mo: Mobj) -> tuple[float, float]:
        g = self.grid
        cells = [c for c in self.cells_under(mo.x, mo.y, mo.radius) if c >= 0 and not g.solid[c]]
        if not cells:
            c = g.cell_at(mo.x, mo.y)
            return float(g.fh[c]), float(g.ch[c])
        return max(float(g.fh[c]) for c in cells), min(float(g.ch[c]) for c in cells)

    def check_sight(self, a: Mobj, b: Mobj) -> bool:
        """Line of sight from ``a``'s eyes to ``b`` through open cells (heights checked)."""
        z1 = a.z + a.height * 0.75
        for zt in (b.z + b.height * 0.5, b.z + b.height * 0.9):
            if self._trace_clear(a.x, a.y, z1, b.x, b.y, zt):
                return True
        return False

    def _trace_clear(self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> bool:
        g = self.grid
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1:
            return True
        cx, cy = int(x1 // CELL), int(y1 // CELL)
        tx, ty = int(x2 // CELL), int(y2 // CELL)
        rdx, rdy = dx / dist, dy / dist
        stepx = 1 if rdx > 0 else -1
        stepy = 1 if rdy > 0 else -1
        ddx = abs(CELL / rdx) if rdx else 1e30
        ddy = abs(CELL / rdy) if rdy else 1e30
        sdx = ((cx + 1) * CELL - x1) / rdx if rdx > 0 else (x1 - cx * CELL) / -rdx if rdx < 0 else 1e30
        sdy = ((cy + 1) * CELL - y1) / rdy if rdy > 0 else (y1 - cy * CELL) / -rdy if rdy < 0 else 1e30
        slope = (z2 - z1) / dist
        for _ in range(256):
            if cx == tx and cy == ty:
                return True
            if sdx < sdy:
                t = sdx
                sdx += ddx
                cx += stepx
            else:
                t = sdy
                sdy += ddy
                cy += stepy
            if t > dist:
                return True
            if not g.in_bounds(cx, cy):
                return False
            c = cy * g.w + cx
            if g.solid[c]:
                return False
            z = z1 + slope * t
            if z < g.fh[c] or z > g.ch[c] or g.ch[c] - g.fh[c] < 1:
                return False
        return False

    # --- the tic ----------------------------------------------------------------------------------------------------

    def tick(self, cmd: dict) -> None:
        self.leveltime += 1
        self.player_think(cmd)
        for mo in list(self.mobjs):
            if not mo.removed:
                self.mobj_think(mo)
        if any(mo.removed for mo in self.mobjs):
            self.mobjs = [m for m in self.mobjs if not m.removed]
        self.movers_think()
        self.sector_think()
        if self.exit_pending:
            self.exit_pending -= 1
            if self.exit_pending == 0:
                self.exited = True

    # --- the player ------------------------------------------------------------------------------------------------

    def player_think(self, cmd: dict) -> None:
        pl, mo = self.player, self.pmo
        if pl.damagecount:
            pl.damagecount -= 1
        if pl.bonuscount:
            pl.bonuscount -= 1
        if pl.got_weapon:
            pl.got_weapon -= 1
        for k in ("invuln", "suit"):
            if pl.powers.get(k):
                pl.powers[k] -= 1
                if pl.powers[k] == 4 * 35:
                    self.message("Your " + ("aegis field" if k == "invuln" else "hazmat suit") + " is failing!")
                if pl.powers[k] <= 0:
                    del pl.powers[k]
        if pl.dead:
            self.death_think(cmd)
            self.xy_movement(mo)
            self.z_movement(mo)
            return
        mo.angle = (mo.angle + cmd.get("turn", 0.0)) % (2 * math.pi)
        onground = mo.z <= self.floor_ceiling(mo)[0] + 0.01
        fwd, side = cmd.get("forward", 0.0), cmd.get("side", 0.0)
        if onground and (fwd or side):
            ca, sa = math.cos(mo.angle), math.sin(mo.angle)
            mo.momx += (fwd * ca - side * sa) / 32.0
            mo.momy += (fwd * sa + side * ca) / 32.0
        self.xy_movement(mo)
        self.z_movement(mo)
        self.calc_height(cmd)
        self.check_specials()
        self.touch_items()
        if cmd.get("use"):
            if not pl.usedown:
                self.use_lines()
                pl.usedown = True
        else:
            pl.usedown = False
        if cmd.get("weapon"):
            self.request_weapon(cmd["weapon"])
        self.weapon_think(bool(cmd.get("fire")))
        mo.hp = pl.health

    def calc_height(self, cmd: dict) -> None:
        pl, mo = self.player, self.pmo
        pl.bob = min((mo.momx ** 2 + mo.momy ** 2) / 4.0, 16.0)
        ang = (self.leveltime * 2 * math.pi / 20.0)
        bob = pl.bob / 2 * math.sin(ang)
        pl.viewheight += pl.deltaviewheight
        if pl.viewheight > VIEWHEIGHT:
            pl.viewheight = VIEWHEIGHT
            pl.deltaviewheight = 0.0
        if pl.viewheight < VIEWHEIGHT / 2:
            pl.viewheight = VIEWHEIGHT / 2
            if pl.deltaviewheight <= 0:
                pl.deltaviewheight = 1.0
        if pl.deltaviewheight:
            pl.deltaviewheight += 0.25
            if not pl.deltaviewheight:
                pl.deltaviewheight = 1.0
        _, ceilz = self.floor_ceiling(mo)
        pl.viewz = min(mo.z + pl.viewheight + bob, ceilz - 4)

    def death_think(self, cmd: dict) -> None:
        pl, mo = self.player, self.pmo
        if pl.viewheight > 6:
            pl.viewheight -= 1
        pl.viewz = mo.z + pl.viewheight
        pl.deltaviewheight = 0
        if pl.attacker is not None and pl.attacker is not mo:
            want = math.atan2(pl.attacker.y - mo.y, pl.attacker.x - mo.x)
            d = (want - mo.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(d) < math.radians(5):
                mo.angle = want
                if pl.damagecount:
                    pl.damagecount -= 1
            else:
                mo.angle += math.copysign(math.radians(5), d)
        elif pl.damagecount:
            pl.damagecount -= 1
        pl.wy = min(pl.wy + 6, 96)

    def xy_movement(self, mo: Mobj) -> None:
        if not mo.momx and not mo.momy:
            return
        mo.momx = max(min(mo.momx, 30.0), -30.0)
        mo.momy = max(min(mo.momy, 30.0), -30.0)
        xmove, ymove = mo.momx, mo.momy
        while xmove or ymove:
            if abs(xmove) > 15 or abs(ymove) > 15:
                px, py = xmove / 2, ymove / 2
            else:
                px, py = xmove, ymove
            xmove -= px
            ymove -= py
            ok, blocker = self.try_move(mo, mo.x + px, mo.y + py)
            if ok:
                continue
            if mo.skullfly:
                self.skull_hit(mo, blocker)
                return
            if mo is self.pmo:
                ok1, _ = self.try_move(mo, mo.x + px, mo.y)
                ok2, _ = self.try_move(mo, mo.x, mo.y + py)
                if not ok1:
                    mo.momx = 0.0
                if not ok2:
                    mo.momy = 0.0
                if not ok1 and not ok2:
                    break
            else:
                mo.momx = mo.momy = 0.0
                break
        if mo.skullfly:
            return
        fz, _ = self.floor_ceiling(mo)
        if mo.z > fz and not mo.floats and mo is not self.pmo:
            return
        if mo is self.pmo and mo.z > fz:
            return
        if abs(mo.momx) < 0.0625 and abs(mo.momy) < 0.0625 and mo is self.pmo:
            mo.momx = mo.momy = 0.0
        else:
            mo.momx *= FRICTION
            mo.momy *= FRICTION
            if abs(mo.momx) < 0.01 and abs(mo.momy) < 0.01:
                mo.momx = mo.momy = 0.0

    def z_movement(self, mo: Mobj) -> None:
        fz, cz = self.floor_ceiling(mo)
        if mo.floats and mo.target is not None and not mo.skullfly and mo.alive:
            dist = math.hypot(mo.x - mo.target.x, mo.y - mo.target.y)
            delta = (mo.target.z + mo.height / 2) - mo.z
            if delta < 0 and dist < -(delta * 3):
                mo.z -= FLOATSPEED
            elif delta > 0 and dist < delta * 3:
                mo.z += FLOATSPEED
        mo.z += mo.momz
        if mo.z <= fz:
            if mo.skullfly:
                mo.momz = -mo.momz
            if mo is self.pmo and mo.momz < -8:
                self.player.deltaviewheight = mo.momz / 8.0
                self.sound("oof")
            mo.z = fz
            if not mo.skullfly:
                mo.momz = 0.0
        elif not mo.nogravity:
            mo.momz = -GRAVITY * 2 if mo.momz == 0 else mo.momz - GRAVITY
        if mo.z + mo.height > cz:
            if mo.momz > 0:
                mo.momz = 0.0
            mo.z = cz - mo.height
            if mo.skullfly:
                mo.momz = -mo.momz
        if not mo.alive and mo.floats and mo.z > fz:
            mo.z = max(fz, mo.z - 4.0)

    # --- specials, pickups ----------------------------------------------------------------------------------------

    def check_specials(self) -> None:
        g, mo = self.grid, self.pmo
        c = g.cell_at(mo.x, mo.y)
        if g.secret[c] and c not in self.secrets_found:
            self._find_secret(c)
        trig = g.specs[c].get("trigger")
        if trig and c not in self.triggered:
            self.triggered.add(c)
            self.do_action(trig)
        dmg = int(g.damage[c])
        if dmg and mo.z <= g.fh[c] + 0.5 and self.leveltime % 32 == 0:
            if not self.player.powers.get("suit") or (dmg >= 20 and self.rnd() < 5):
                self.damage(mo, None, None, dmg)
        if g.specs[c].get("exitfloor") and not self.exit_pending:
            self.exit_pending = 1
            self.secret_exit = g.specs[c].get("exitfloor") == "secret"

    def _find_secret(self, c: int) -> None:
        g = self.grid
        stack, group = [c], {c}
        while stack:
            k = stack.pop()
            cy, cx = divmod(k, g.w)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (cy + dy) * g.w + cx + dx
                if g.in_bounds(cx + dx, cy + dy) and g.secret[n] and n not in group:
                    group.add(n)
                    stack.append(n)
        self.secrets_found |= group
        self.player.secretcount += 1
        self.message("You found a secret stash!")
        self.sound("secret")

    def touch_items(self) -> None:
        mo = self.pmo
        for o in self.mobjs:
            if o.cls != "item" or o.removed:
                continue
            if abs(o.x - mo.x) < o.radius + mo.radius and abs(o.y - mo.y) < o.radius + mo.radius \
                    and abs(o.z - mo.z) < 24:
                if self.give_item(o.kind, o.dropped):
                    o.removed = True
                    if ITEM_INFO.get(o.kind, {}).get("count"):
                        self.player.itemcount += 1

    def give_ammo(self, ammo: str, num: int, dropped: bool = False) -> bool:
        pl = self.player
        if pl.ammo[ammo] >= pl.maxammo[ammo]:
            return False
        if dropped:
            num = max(num // 2, 1)
        if self.skill in (1, 5):
            num *= 2
        old = pl.ammo[ammo]
        pl.ammo[ammo] = min(pl.ammo[ammo] + num, pl.maxammo[ammo])
        if old == 0 and pl.weapon in ("fist", "pistol") and not pl.pending:
            cands = {"bullets": ("chaingun", "pistol"), "shells": ("shotgun",), "rockets": ("launcher",),
                     "cells": ("plasma",)}[ammo]
            for w in cands:
                if w in pl.weapons and w != pl.weapon and (ammo != "rockets" or pl.weapon == "fist"):
                    pl.pending = w
                    break
        return True

    def give_weapon(self, w: str, dropped: bool = False) -> bool:
        pl = self.player
        ammo = WEAPONS[w]["ammo"]
        amount = {"shotgun": 8, "chaingun": 20, "launcher": 2, "plasma": 40, "bfg": 40}.get(w, 0)
        got_ammo = self.give_ammo(ammo, amount, dropped) if ammo else False
        if w not in pl.weapons:
            pl.weapons.add(w)
            pl.pending = w
            pl.got_weapon = 70
            return True
        return got_ammo

    def give_item(self, kind: str, dropped: bool = False) -> bool:
        pl = self.player
        snd = "itemup"
        msg = PICKUP_MSG.get(kind, "")
        if kind == "hbonus":
            pl.health = min(pl.health + 1, 200)
        elif kind == "abonus":
            pl.armor = min(pl.armor + 1, 200)
            if not pl.armortype:
                pl.armortype = 1
        elif kind == "stim":
            if pl.health >= 100:
                return False
            pl.health = min(pl.health + 10, 100)
        elif kind == "medikit":
            if pl.health >= 100:
                return False
            if pl.health < 25:
                msg = "Picked up a field medkit. Just in time!"
            pl.health = min(pl.health + 25, 100)
        elif kind == "soulsphere":
            pl.health = min(pl.health + 100, 200)
            snd = "getpow"
        elif kind in ("garmor", "barmor"):
            amount, t = (100, 1) if kind == "garmor" else (200, 2)
            if pl.armor >= amount:
                return False
            pl.armor, pl.armortype = amount, t
        elif kind in ("clip", "clip_drop"):
            if not self.give_ammo("bullets", 10, dropped or kind == "clip_drop"):
                return False
        elif kind == "ammobox":
            if not self.give_ammo("bullets", 50):
                return False
        elif kind == "shells":
            if not self.give_ammo("shells", 4):
                return False
        elif kind == "shellbox":
            if not self.give_ammo("shells", 20):
                return False
        elif kind == "rocket":
            if not self.give_ammo("rockets", 1):
                return False
        elif kind == "rocketbox":
            if not self.give_ammo("rockets", 5):
                return False
        elif kind == "cell":
            if not self.give_ammo("cells", 20):
                return False
        elif kind == "cellpack":
            if not self.give_ammo("cells", 100):
                return False
        elif kind == "backpack":
            if not pl.backpack:
                for a in pl.maxammo:
                    pl.maxammo[a] = AMMO_MAX[a] * 2
                pl.backpack = True
            for a, n in (("bullets", 10), ("shells", 4), ("rockets", 1), ("cells", 20)):
                self.give_ammo(a, n)
        elif kind == "berserk":
            pl.powers["berserk"] = 1
            pl.health = max(pl.health, 100)
            if pl.weapon != "fist":
                pl.pending = "fist"
            pl.damagecount = max(pl.damagecount, 60)
            snd = "getpow"
        elif kind in ("invuln", "suit"):
            pl.powers[kind] = POWER_TICS[kind]
            snd = "getpow"
        elif kind == "scanner":
            pl.powers["scanner"] = 1
            snd = "getpow"
        elif kind in ("bluecard", "yellowcard", "redcard"):
            key = ITEM_INFO[kind]["key"]
            if key in pl.keys:
                return False
            pl.keys.add(key)
        elif kind.startswith("w_"):
            w = {"w_shotgun": "shotgun", "w_shotgun_drop": "shotgun", "w_chaingun": "chaingun",
                 "w_chaingun_drop": "chaingun", "w_launcher": "launcher", "w_plasma": "plasma", "w_bfg": "bfg",
                 "w_saw": "saw"}[kind]
            if not self.give_weapon(w, dropped or kind.endswith("_drop")):
                return False
            snd = "wpnup"
        else:
            return False
        pl.bonuscount = min(pl.bonuscount + 6, 40)
        self.message(msg)
        self.sound(snd)
        return True

    # --- weapons ---------------------------------------------------------------------------------------------------

    def has_ammo(self, w: str) -> bool:
        info = WEAPONS[w]
        return info["ammo"] is None or self.player.ammo[info["ammo"]] >= info["use"]

    def request_weapon(self, slot_or_name) -> None:  # noqa: ANN001
        pl = self.player
        if isinstance(slot_or_name, int):
            cands = [w for w in WEAPON_ORDER if WEAPONS[w]["slot"] == slot_or_name and w in pl.weapons]
            if slot_or_name == 1:
                cands.sort(key=lambda w: 0 if w == "saw" else 1)
            if not cands:
                return
            cur = pl.pending or pl.weapon
            w = cands[(cands.index(cur) + 1) % len(cands)] if cur in cands else cands[0]
        elif slot_or_name in ("next", "prev"):
            owned = [w for w in WEAPON_ORDER if w in pl.weapons]
            cur = pl.pending or pl.weapon
            i = owned.index(cur) if cur in owned else 0
            step = 1 if slot_or_name == "next" else -1
            for k in range(1, len(owned) + 1):
                cand = owned[(i + step * k) % len(owned)]
                if self.has_ammo(cand) or cand == "fist":
                    w = cand
                    break
            else:
                return
        else:
            w = slot_or_name
        if w in pl.weapons and w != pl.weapon:
            pl.pending = w

    def set_wstate(self, state: str) -> None:
        pl = self.player
        pl.wstate = state
        pl.wsi = 0
        if state == "fire":
            self._enter_fire_state()
        elif state == "ready":
            pl.wframe = 0
            pl.wflash = -1
            pl.wtics = 1

    def _enter_fire_state(self) -> None:
        pl = self.player
        seqs = WEAPONS[pl.weapon]["fire"]
        while True:
            if pl.wsi >= len(seqs):
                self.set_wstate("ready")
                return
            frame, tics, action, flash = seqs[pl.wsi]
            pl.wframe = frame
            pl.wflash = flash
            pl.wtics = tics
            if action:
                jumped = self.weapon_action(action)
                if jumped:
                    return
            if tics > 0:
                return
            pl.wsi += 1

    def weapon_think(self, fire: bool) -> None:
        pl = self.player
        self._fire_input = fire
        if pl.extralight and pl.flash <= 0:
            pl.extralight = 0
        pl.flash -= 1
        if fire:
            pl.fire_time += 1
        else:
            pl.fire_time = 0
        if pl.wstate == "lower":
            pl.wflash = -1
            pl.wy += 8
            if pl.wy >= 96:
                pl.wy = 96
                if pl.pending:
                    pl.weapon = pl.pending
                    pl.pending = None
                pl.wstate = "raise"
                pl.wframe = 0
            return
        if pl.wstate == "raise":
            pl.wy -= 8
            if pl.wy <= 0:
                pl.wy = 0
                self.set_wstate("ready")
            return
        if pl.wstate == "ready":
            pl.wframe = 0
            if pl.pending or not pl.health:
                pl.wstate = "lower"
                return
            if fire:
                if self.has_ammo(pl.weapon):
                    if not pl.attackdown or pl.weapon not in ("launcher", "bfg"):
                        pl.attackdown = True
                        self.set_wstate("fire")
                else:
                    self.out_of_ammo()
            else:
                pl.attackdown = False
            return
        if pl.wstate == "fire":
            pl.wtics -= 1
            if pl.wtics <= 0:
                pl.wsi += 1
                self._enter_fire_state()

    _fire_input = False

    def out_of_ammo(self) -> None:
        pl = self.player
        for w in AUTO_ORDER:
            if w in pl.weapons and self.has_ammo(w) and w != pl.weapon:
                pl.pending = w
                return

    def weapon_action(self, action: str) -> bool:
        """Run a weapon action; True when it jumped to another state."""
        pl, mo = self.player, self.pmo
        info = WEAPONS[pl.weapon]
        if action == "refire":
            if self._fire_input and not pl.pending and pl.health > 0:
                if self.has_ammo(pl.weapon):
                    pl.refire += 1
                    self.set_wstate("fire")
                    return True
                self.out_of_ammo()
            pl.refire = 0
            if not self.has_ammo(pl.weapon):
                self.out_of_ammo()
            return False
        if action == "saw":
            dmg = 2 * (self.rnd() % 10 + 1)
            ang = mo.angle + (self.rnd() - self.rnd()) / 256.0 * math.radians(5.6)
            slope = self.aim_line(mo, ang, MELEERANGE + 1)[1]
            hit = self.line_attack(mo, ang, MELEERANGE + 1, slope, dmg)
            if hit is None:
                self.sound("sawful")
                return False
            self.sound("sawhit")
            want = math.atan2(hit.y - mo.y, hit.x - mo.x)
            d = (want - mo.angle + math.pi) % (2 * math.pi) - math.pi
            mo.angle = (mo.angle + max(min(d, 0.06), -0.06)) % (2 * math.pi) if abs(d) > 0.06 else want
            self.noise_alert()
            return False
        if action == "punch":
            dmg = (self.rnd() % 10 + 1) * 2
            if pl.powers.get("berserk"):
                dmg *= 10
            ang = mo.angle + (self.rnd() - self.rnd()) / 256.0 * math.radians(5.6)
            slope = self.aim_line(mo, ang, MELEERANGE)[1]
            hit = self.line_attack(mo, ang, MELEERANGE, slope, dmg)
            if hit is not None and hit.cls in ("monster", "barrel"):
                self.sound("punch")
                mo.angle = math.atan2(hit.y - mo.y, hit.x - mo.x)
            self.noise_alert()
            return False
        if action == "pump":
            self.sound("sgcock")
            return False
        if action == "bfgsound":
            self.sound("bfg")
            return False
        if action == "flash":
            self._flash(2)
            return False
        if info["ammo"]:
            pl.ammo[info["ammo"]] = max(pl.ammo[info["ammo"]] - info["use"], 0)
        if action in ("pistol", "cgun"):
            self.sound("pistol")
            self._flash(1)
            slope = self.bullet_slope(mo)
            self.gunshot(mo, slope, accurate=(pl.refire == 0))
        elif action == "shotgun":
            self.sound("shotgn")
            self._flash(1)
            slope = self.bullet_slope(mo)
            for _ in range(7):
                self.gunshot(mo, slope, accurate=False)
        elif action == "rocket":
            self.player_missile("rocket")
        elif action == "plasma":
            self._flash(1)
            self.player_missile("plasma")
        elif action == "bfg":
            self._flash(2)
            self.player_missile("bfg")
        self.noise_alert()
        return False

    def _flash(self, level: int) -> None:
        self.player.extralight = level
        self.player.flash = 4

    def bullet_slope(self, mo: Mobj) -> float:
        for da in (0.0, math.radians(5.625), -math.radians(5.625)):
            target, slope = self.aim_line(mo, mo.angle + da, 1024.0)
            if target is not None:
                return slope
        return 0.0

    def gunshot(self, mo: Mobj, slope: float, accurate: bool) -> None:
        dmg = 5 * (self.rnd() % 3 + 1)
        ang = mo.angle
        if not accurate:
            ang += (self.rnd() - self.rnd()) / 256.0 * math.radians(5.625)
        self.line_attack(mo, ang, MISSILERANGE, slope, dmg)

    # --- hitscan ---------------------------------------------------------------------------------------------------

    def _wall_distance(self, x: float, y: float, z: float, ang: float, slope: float, maxd: float) -> tuple[float, bool]:
        """Distance along the shot to the first wall / step / floor hit; and whether it hit the sky."""
        g = self.grid
        rdx, rdy = math.cos(ang), math.sin(ang)
        cx, cy = int(x // CELL), int(y // CELL)
        stepx = 1 if rdx > 0 else -1
        stepy = 1 if rdy > 0 else -1
        ddx = abs(CELL / rdx) if abs(rdx) > 1e-9 else 1e30
        ddy = abs(CELL / rdy) if abs(rdy) > 1e-9 else 1e30
        sdx = ((cx + 1) * CELL - x) / rdx if rdx > 1e-9 else (x - cx * CELL) / -rdx if rdx < -1e-9 else 1e30
        sdy = ((cy + 1) * CELL - y) / rdy if rdy > 1e-9 else (y - cy * CELL) / -rdy if rdy < -1e-9 else 1e30
        cur = cy * g.w + cx
        t0 = 0.0
        for _ in range(512):
            if sdx < sdy:
                t = sdx
                sdx += ddx
                cx += stepx
            else:
                t = sdy
                sdy += ddy
                cy += stepy
            # within the current cell: floor or ceiling hit?
            if slope < 0:
                tf = (g.fh[cur] - z) / slope
                if t0 <= tf < t:
                    return min(tf, maxd), False
            elif slope > 0:
                tc = (g.ch[cur] - z) / slope
                if t0 <= tc < t:
                    return min(tc, maxd), bool(g.sky[cur])
            if t >= maxd:
                return maxd, False
            if not g.in_bounds(cx, cy):
                return t, False
            nxt = cy * g.w + cx
            zt = z + slope * t
            if g.solid[nxt] or zt < g.fh[nxt] or zt > g.ch[nxt]:
                sky = bool(g.sky[cur]) and not g.solid[nxt] and zt > g.ch[nxt]
                if g.solid[nxt] and g.sky[cur] and zt > g.ch[cur] - 1:
                    sky = True
                return t, sky
            cur = nxt
            t0 = t
        return maxd, False

    def _things_on_line(self, src: Mobj, ang: float, maxd: float) -> list[tuple[float, Mobj]]:
        ca, sa = math.cos(ang), math.sin(ang)
        out = []
        for o in self.mobjs:
            if o is src or not o.shootable or o.removed or not o.alive:
                continue
            dx, dy = o.x - src.x, o.y - src.y
            t = dx * ca + dy * sa
            if t <= 0 or t > maxd + o.radius:
                continue
            perp = abs(-dx * sa + dy * ca)
            if perp < o.radius:
                out.append((max(t - math.sqrt(max(o.radius ** 2 - perp ** 2, 0.0)), 1.0), o))
        if src is not self.pmo and not self.player.dead:
            o = self.pmo
            dx, dy = o.x - src.x, o.y - src.y
            t = dx * ca + dy * sa
            perp = abs(-dx * sa + dy * ca)
            if 0 < t <= maxd + o.radius and perp < o.radius:
                out.append((max(t - math.sqrt(max(o.radius ** 2 - perp ** 2, 0.0)), 1.0), o))
        out.sort(key=lambda it: it[0])
        return out

    def shoot_z(self, mo: Mobj) -> float:
        return mo.z + mo.height / 2 + 8 if mo is self.pmo else mo.z + mo.height / 2 + 8

    def aim_line(self, mo: Mobj, ang: float, maxd: float) -> tuple[Mobj | None, float]:
        z = self.shoot_z(mo)
        wall, _ = self._wall_distance(mo.x, mo.y, z, ang, 0.0, maxd)
        for t, o in self._things_on_line(mo, ang, maxd):
            if t > wall:
                break
            if o is mo.source:
                continue
            slope = ((o.z + o.height / 2) - z) / max(t, 1.0)
            slope = max(min(slope, 100 / 160), -100 / 160)
            if not self._trace_clear(mo.x, mo.y, z, o.x, o.y, o.z + o.height / 2):
                if not self._trace_clear(mo.x, mo.y, z, o.x, o.y, o.z + o.height * 0.9):
                    continue
            return o, slope
        return None, 0.0

    def line_attack(self, mo: Mobj, ang: float, maxd: float, slope: float, dmg: int) -> Mobj | None:
        z = self.shoot_z(mo)
        wall, sky = self._wall_distance(mo.x, mo.y, z, ang, slope, maxd)
        for t, o in self._things_on_line(mo, ang, maxd):
            if t > wall:
                break
            zt = z + slope * t
            if zt < o.z or zt > o.z + o.height:
                continue
            x, y = mo.x + math.cos(ang) * (t - 4), mo.y + math.sin(ang) * (t - 4)
            if o.cls == "monster" or o is self.pmo:
                self.spawn_fx("blood", x, y, zt - 4)
            else:
                self.spawn_fx("puff", x, y, zt - 4)
            self.damage(o, mo, mo, dmg)
            return o
        if wall < maxd and not sky:
            t = wall - 4
            x, y = mo.x + math.cos(ang) * t, mo.y + math.sin(ang) * t
            self.spawn_fx("puff", x, y, z + slope * t - 4)
        return None

    def spawn_fx(self, kind: str, x: float, y: float, z: float) -> Mobj:
        fx = Mobj(kind, "fx", x, y, z)
        fx.si = 0
        fx.tics = 4
        fx.frame = 0
        fx.bright = kind == "puff"
        self.mobjs.append(fx)
        return fx

    # --- missiles --------------------------------------------------------------------------------------------------

    def player_missile(self, kind: str) -> None:
        mo = self.pmo
        ang = mo.angle
        slope = 0.0
        for da in (0.0, math.radians(5.625), -math.radians(5.625)):
            target, s = self.aim_line(mo, mo.angle + da, 1024.0)
            if target is not None:
                ang, slope = mo.angle + da, s
                break
        self.spawn_missile(mo, kind, ang, slope, z=mo.z + 32)

    def spawn_missile(self, src: Mobj, kind: str, ang: float, slope: float, z: float | None = None) -> Mobj:
        info = MISSILE_INFO[kind]
        m = Mobj(kind, "missile", src.x, src.y, src.z + 32 if z is None else z)
        m.info = info
        m.radius, m.height = float(info["radius"]), float(info["height"])
        m.source = src
        m.angle = ang
        speed = info["speed"] * (1.6 if self.skill == 5 and src is not self.pmo else 1.0)
        m.momx, m.momy = math.cos(ang) * speed, math.sin(ang) * speed
        m.momz = slope * speed
        m.tics = 4
        m.frame = 0
        m.bright = True
        self.mobjs.append(m)
        self.sound(info["sound"], src)
        # spawned a little forward so it does not hit its shooter
        m.x += m.momx / 2
        m.y += m.momy / 2
        m.z += m.momz / 2
        ok, _, _, _, blocker = self.check_position(m, m.x, m.y)
        if not ok:
            self.explode_missile(m, blocker if isinstance(blocker, Mobj) else None)
        return m

    def missile_think(self, m: Mobj) -> None:
        if m.state == "boom":
            m.tics -= 1
            if m.tics <= 0:
                m.si += 1
                m.tics = 4 if m.kind != "bfg" else 6
                if m.kind == "bfg" and m.si == 2:
                    self.bfg_spray(m)
                n = {"bfg": 5}.get(m.kind, 3)
                if m.si >= n:
                    m.removed = True
            return
        m.tics -= 1
        if m.tics <= 0:
            m.tics = 4
            m.frame = 1 - int(m.frame)
        steps = max(1, int(math.ceil(max(abs(m.momx), abs(m.momy)) / 8.0)))
        g = self.grid
        for _ in range(steps):
            nx, ny, nz = m.x + m.momx / steps, m.y + m.momy / steps, m.z + m.momz / steps
            ok, floorz, ceilz, _, blocker = self.check_position(m, nx, ny)
            hit_thing = blocker if isinstance(blocker, Mobj) else None
            if hit_thing is None and ok and m.source is not self.pmo and not self.player.dead:
                p = self.pmo
                if abs(p.x - nx) < p.radius + m.radius and abs(p.y - ny) < p.radius + m.radius \
                        and p.z <= nz + m.height and nz <= p.z + p.height:
                    hit_thing, ok = p, False
            if ok and (nz < floorz or nz + m.height > ceilz):
                c = g.cell_at(nx, ny)
                if nz + m.height > ceilz and g.sky[c]:
                    m.removed = True  # into the sky: gone
                    return
                ok = False
            if not ok:
                if hit_thing is None:
                    c = g.cell_at(nx, ny)
                    if g.sky[c] and not g.solid[c] and nz + m.height > g.ch[c] - 2:
                        m.removed = True
                        return
                self.explode_missile(m, hit_thing)
                return
            m.x, m.y, m.z = nx, ny, nz

    def explode_missile(self, m: Mobj, hit: Mobj | None) -> None:
        info = m.info
        if hit is not None and hit.shootable:
            dmg = (self.rnd() % 8 + 1) * info["damage"]
            self.damage(hit, m, m.source, dmg)
        m.momx = m.momy = m.momz = 0.0
        m.state = "boom"
        m.si = 0
        m.tics = 4 if m.kind != "bfg" else 6
        m.x -= math.cos(m.angle) * 4
        m.y -= math.sin(m.angle) * 4
        self.sound(info["hit"], m)
        if info.get("splash"):
            self.radius_attack(m, m.source, info["splash"])

    def radius_attack(self, spot: Mobj, source: Mobj | None, damage: int) -> None:
        for o in [self.pmo] + list(self.mobjs):
            if not o.shootable or o.removed or not o.alive or o is spot:
                continue
            if o is self.pmo and self.player.dead:
                continue
            if o.info.get("noradius"):
                continue
            dist = max(abs(o.x - spot.x), abs(o.y - spot.y)) - o.radius
            dist = max(dist, 0.0)
            if dist >= damage:
                continue
            if not self._trace_clear(spot.x, spot.y, spot.z + 4, o.x, o.y, o.z + o.height / 2):
                continue
            self.damage(o, spot, source, int(damage - dist))

    def bfg_spray(self, m: Mobj) -> None:
        src = m.source or self.pmo
        for i in range(40):
            ang = m.angle - math.pi / 4 + math.pi / 2 / 40 * i
            target, _ = self.aim_line(src, ang, 16 * 64.0)
            if target is None:
                continue
            fx = self.spawn_fx("tracer", target.x, target.y, target.z + target.height / 4)
            fx.bright = True
            dmg = sum(self.rnd() % 8 + 1 for _ in range(15))
            self.damage(target, src, src, dmg)

    # --- damage ----------------------------------------------------------------------------------------------------

    def damage(self, target: Mobj, inflictor: Mobj | None, source: Mobj | None, dmg: int) -> None:
        if not target.shootable or not target.alive:
            return
        if target.skullfly:
            target.momx = target.momy = target.momz = 0.0
            target.skullfly = False
        # thrust away from the inflictor
        if inflictor is not None and target.cls != "barrel":
            mass = target.info.get("mass", 100) if target is not self.pmo else 100
            ang = math.atan2(target.y - inflictor.y, target.x - inflictor.x)
            thrust = dmg * 12.5 / mass
            if dmg < 40 and dmg > target.hp and target.z - (inflictor.z if inflictor else 0) > 64 \
                    and self.rnd() & 1:
                ang += math.pi
                thrust *= 4
            target.momx += math.cos(ang) * thrust
            target.momy += math.sin(ang) * thrust
        if target is self.pmo:
            pl = self.player
            if pl.powers.get("invuln") or self.god:
                return
            if self.skill == 1:
                dmg = max(dmg >> 1, 1)
            if pl.armortype:
                saved = dmg // 3 if pl.armortype == 1 else dmg // 2
                if pl.armor <= saved:
                    saved = pl.armor
                    pl.armortype = 0
                pl.armor -= saved
                dmg -= saved
            pl.health = max(pl.health - dmg, 0)
            pl.attacker = source
            pl.damagecount = min(pl.damagecount + dmg, 100)
            target.hp = pl.health
            if pl.health <= 0:
                self.kill_player(source)
            elif dmg > 0:
                self.sound("plpain")
            return
        target.hp -= dmg
        if target.hp <= 0:
            self.kill(target, source)
            return
        if target.cls != "monster":
            return
        if self.rnd() < target.info["painchance"] and not target.skullfly:
            target.justhit = True
            self.set_state(target, "pain")
        target.reactiontime = 0
        if source is not None and source is not target and (not target.threshold or target.kind == "vile") \
                and source.cls in ("player", "monster") and source.alive:
            if source is self.pmo or source.kind != target.kind:
                target.target = source
                target.threshold = 100
                if target.state == "spawn" and "see" in target.info["states"]:
                    self.set_state(target, "see")

    def kill_player(self, source: Mobj | None) -> None:
        pl = self.player
        pl.dead = True
        pl.health = 0
        self.pmo.solid = False
        self.pmo.momz = 0.0
        self.sound("pldeth" if pl.damagecount < 60 else "pdiehi")
        pl.wstate = "lower"
        pl.pending = None
        self.events.append(("died",))

    def kill(self, mo: Mobj, source: Mobj | None) -> None:
        mo.alive = False
        mo.shootable = False
        mo.skullfly = False
        if mo.cls == "barrel":
            mo.target = source
            mo.state = "boom"
            mo.si = 0
            mo.tics = 5
            return
        if mo.countkill:
            self.player.killcount += 1
        mo.floats = mo.floats and mo.kind == "soul"
        mo.nogravity = mo.kind == "soul"
        states = mo.info["states"]
        if "xdeath" in states and mo.hp < -mo.info["hp"]:
            self.set_state(mo, "xdeath")
        else:
            self.set_state(mo, "death")
        mo.tics = max(1, mo.tics - (self.rnd() & 3))
        drop = mo.info.get("drop")
        if drop:
            d = self.spawn(drop, mo.x, mo.y)
            d.dropped = True
            d.z = mo.z
        if mo.info.get("boss"):
            self._check_boss_death()

    def _check_boss_death(self) -> None:
        if self.boss_done:
            return
        if any(o.cls == "monster" and o.alive and o.info.get("boss") for o in self.mobjs):
            return
        self.boss_done = True
        tag = getattr(self.mapdef, "boss_action", None)
        if tag:
            self.do_action(tag)

    # --- monsters -------------------------------------------------------------------------------------------------

    def set_state(self, mo: Mobj, state: str, si: int = 0) -> None:
        states = mo.info["states"]
        while True:
            if state == "remove":
                mo.removed = True
                return
            if state not in states:
                state = "see" if "see" in states else "spawn"
            seqd = states[state]
            mo.state, mo.si = state, si
            if si >= len(seqd["states"]):
                nxt = seqd["next"]
                if nxt is None:
                    mo.si = len(seqd["states"]) - 1
                    mo.tics = -1
                    return
                state, si = nxt, 0
                continue
            frame, tics, action, bright = seqd["states"][si]
            mo.frame, mo.tics, mo.bright = frame, tics, bright
            if self.skill == 5 and state in ("see", "missile", "melee") and tics > 1:
                mo.tics = max(1, tics // 2) if mo.kind in ("demon",) else tics
            if action:
                self.monster_action(mo, action)
                if mo.removed or mo.state != state or mo.si != si:
                    return
            if tics != 0:
                return
            si += 1

    def mobj_think(self, mo: Mobj) -> None:
        if mo.cls == "missile":
            self.missile_think(mo)
            return
        if mo.cls == "fx":
            mo.tics -= 1
            if mo.kind == "puff":
                mo.z += 0.5
            if mo.tics <= 0:
                mo.si += 1
                mo.frame = mo.si
                mo.tics = 4
                if mo.si >= {"puff": 4, "blood": 3, "tracer": 3}.get(mo.kind, 3):
                    mo.removed = True
            return
        if mo.cls == "barrel":
            if mo.state == "boom":
                mo.tics -= 1
                if mo.tics <= 0:
                    mo.si += 1
                    mo.tics = 5
                    if mo.si == 2:
                        self.sound("barexp", mo)
                        self.radius_attack(mo, mo.target, 128)
                    if mo.si >= 4:
                        mo.removed = True
            return
        if mo.cls in ("item", "decor"):
            return
        # monsters
        if mo.momx or mo.momy or mo.skullfly:
            self.xy_movement(mo)
            if mo.removed:
                return
        if mo.alive or mo.momz or not mo.nogravity:
            self.z_movement(mo)
        if mo.tics == -1:
            return
        mo.tics -= 1
        if mo.tics <= 0:
            self.set_state(mo, mo.state, mo.si + 1)

    def monster_action(self, mo: Mobj, action: str) -> None:
        getattr(self, "a_" + action)(mo)

    def a_look(self, mo: Mobj) -> None:
        mo.threshold = 0
        p = self.pmo
        if self.player.dead:
            return
        heard = getattr(mo, "heard", False)
        if heard and not mo.ambush:
            mo.target = p
        elif not self.look_for_player(mo, all_around=False):
            return
        mo.target = p
        snd = mo.info["sounds"].get("sight")
        if snd:
            self.sound(snd, mo if not mo.info.get("boss") else None)
        self.set_state(mo, "see")

    def look_for_player(self, mo: Mobj, all_around: bool) -> bool:
        p = self.pmo
        if self.player.dead:
            return False
        if not self.check_sight(mo, p):
            return False
        if not all_around:
            an = math.atan2(p.y - mo.y, p.x - mo.x)
            d = abs((an - mo.angle + math.pi) % (2 * math.pi) - math.pi)
            if d > math.pi / 2 and math.hypot(p.x - mo.x, p.y - mo.y) > MELEERANGE:
                return False
        mo.target = p
        return True

    def a_face(self, mo: Mobj) -> None:
        t = mo.target
        if t is None:
            return
        mo.angle = math.atan2(t.y - mo.y, t.x - mo.x)

    def a_pain(self, mo: Mobj) -> None:
        self.sound(mo.info["sounds"].get("pain"), mo)

    def a_scream(self, mo: Mobj) -> None:
        self.sound(mo.info["sounds"].get("death"), mo if not mo.info.get("boss") else None)

    def a_xscream(self, mo: Mobj) -> None:
        self.sound("slop", mo)

    def a_fall(self, mo: Mobj) -> None:
        mo.solid = False

    def _target_alive(self, t: Mobj | None) -> bool:
        if t is None:
            return False
        if t is self.pmo:
            return not self.player.dead
        return t.alive and not t.removed

    def in_melee_range(self, mo: Mobj) -> bool:
        t = mo.target
        if not self._target_alive(t):
            return False
        dist = math.hypot(t.x - mo.x, t.y - mo.y)
        if dist >= MELEERANGE - 20 + t.radius:
            return False
        return self.check_sight(mo, t)

    def check_missile_range(self, mo: Mobj) -> bool:
        t = mo.target
        if not self.check_sight(mo, t):
            return False
        if mo.justhit:
            mo.justhit = False
            return True
        if mo.reactiontime:
            return False
        dist = math.hypot(t.x - mo.x, t.y - mo.y) - 64
        if "melee" not in mo.info["states"]:
            dist -= 128
        if mo.kind == "soul":
            dist /= 2
        dist = min(dist, 200)
        return self.rnd() >= dist

    def a_chase(self, mo: Mobj) -> None:
        if mo.reactiontime:
            mo.reactiontime -= 1
        if mo.threshold:
            if not self._target_alive(mo.target):
                mo.threshold = 0
            else:
                mo.threshold -= 1
        if mo.movedir < 8:
            want = math.atan2(DIRS[mo.movedir][1], DIRS[mo.movedir][0])
            d = (want - mo.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(d) > 0.01:
                mo.angle += math.copysign(min(abs(d), math.pi / 4), d)
        if not self._target_alive(mo.target):
            if mo.target is not self.pmo and self.look_for_player(mo, True):
                return
            mo.target = None
            self.set_state(mo, "spawn")
            return
        if mo.justattacked:
            mo.justattacked = False
            if self.skill != 5:
                self.new_chase_dir(mo)
            return
        st = mo.info["states"]
        if "melee" in st and self.in_melee_range(mo):
            self.sound(mo.info["sounds"].get("attack"), mo)
            self.set_state(mo, "melee")
            return
        if "missile" in st and (self.skill == 5 or not mo.movecount):
            if self.check_missile_range(mo):
                self.set_state(mo, "missile")
                mo.justattacked = True
                return
        mo.movecount -= 1
        if mo.movecount < 0 or not self.monster_move(mo):
            self.new_chase_dir(mo)
        if self.rnd() < 3:
            self.sound(mo.info["sounds"].get("active"), mo)

    def monster_move(self, mo: Mobj) -> bool:
        if mo.movedir == NODIR:
            return False
        dx, dy = DIRS[mo.movedir]
        speed = mo.info["speed"]
        nx, ny = mo.x + dx * speed, mo.y + dy * speed
        ok, blocker = self.try_move(mo, nx, ny)
        if not ok:
            if mo.floats and blocker is None:
                fz, cz = self._floor_ceil_at(nx, ny, mo.radius)
                if fz is not None and cz - fz >= mo.height:
                    mo.z += FLOATSPEED if mo.z < fz else -FLOATSPEED
                    return True
            if isinstance(blocker, int) and blocker >= 0:
                door = self.grid.specs[blocker]
                if door.get("kind") == "door" and not door.get("key") and not door.get("tag") \
                        and not door.get("secretwall"):
                    self.open_door(blocker, by_monster=True)
            return False
        return True

    def _floor_ceil_at(self, x: float, y: float, r: float) -> tuple[float | None, float]:
        g = self.grid
        fz, cz = -1e9, 1e9
        for c in self.cells_under(x, y, r):
            if c < 0 or g.solid[c]:
                return None, 0.0
            fz, cz = max(fz, float(g.fh[c])), min(cz, float(g.ch[c]))
        return fz, cz

    def new_chase_dir(self, mo: Mobj) -> None:
        t = mo.target
        if t is None:
            return
        olddir = mo.movedir
        turnaround = OPPOSITE[olddir]
        deltax, deltay = t.x - mo.x, t.y - mo.y
        d1 = 0 if deltax > 10 else 4 if deltax < -10 else NODIR
        d2 = 6 if deltay > 10 else 2 if deltay < -10 else NODIR
        if d1 != NODIR and d2 != NODIR:
            diag = {(0, 2): 1, (4, 2): 3, (4, 6): 5, (0, 6): 7}[(d1, d2)]
            if diag != turnaround and self._try_dir(mo, diag):
                return
        if self.rnd() > 200 or abs(deltay) > abs(deltax):
            d1, d2 = d2, d1
        if d1 == turnaround:
            d1 = NODIR
        if d2 == turnaround:
            d2 = NODIR
        for d in (d1, d2):
            if d != NODIR and self._try_dir(mo, d):
                return
        if olddir != NODIR and self._try_dir(mo, olddir):
            return
        order = range(8) if self.rnd() & 1 else range(7, -1, -1)
        for d in order:
            if d != turnaround and self._try_dir(mo, d):
                return
        if turnaround != NODIR and self._try_dir(mo, turnaround):
            return
        mo.movedir = NODIR

    def _try_dir(self, mo: Mobj, d: int) -> bool:
        mo.movedir = d
        if not self.monster_move(mo):
            return False
        mo.movecount = self.rnd() & 15
        return True

    def _aim_slope(self, mo: Mobj, t: Mobj) -> float:
        dist = max(math.hypot(t.x - mo.x, t.y - mo.y), 1.0)
        return ((t.z + t.height / 2) - (mo.z + mo.height / 2 + 8)) / dist

    def a_posattack(self, mo: Mobj) -> None:
        t = mo.target
        if t is None:
            return
        self.a_face(mo)
        self.sound("pistol", mo)
        ang = mo.angle + (self.rnd() - self.rnd()) / 256.0 * math.radians(22.5)
        self.line_attack(mo, ang, MISSILERANGE, self._aim_slope(mo, t), ((self.rnd() % 5) + 1) * 3)

    def a_sposattack(self, mo: Mobj) -> None:
        t = mo.target
        if t is None:
            return
        self.a_face(mo)
        self.sound("shotgn", mo)
        slope = self._aim_slope(mo, t)
        for _ in range(3):
            ang = mo.angle + (self.rnd() - self.rnd()) / 256.0 * math.radians(22.5)
            self.line_attack(mo, ang, MISSILERANGE, slope, ((self.rnd() % 5) + 1) * 3)

    def _missile_at(self, mo: Mobj, kind: str) -> None:
        t = mo.target
        if t is None:
            return
        ang = math.atan2(t.y - mo.y, t.x - mo.x)
        dist = max(math.hypot(t.x - mo.x, t.y - mo.y), 1.0)
        slope = ((t.z + t.height / 2) - (mo.z + 32)) / dist
        self.spawn_missile(mo, kind, ang, slope)

    def a_cposattack(self, mo: Mobj) -> None:
        t = mo.target
        if t is None:
            return
        self.a_face(mo)
        self.sound("pistol", mo)
        ang = mo.angle + (self.rnd() - self.rnd()) / 256.0 * math.radians(22.5)
        self.line_attack(mo, ang, MISSILERANGE, self._aim_slope(mo, t), ((self.rnd() % 5) + 1) * 3)

    def a_cposrefire(self, mo: Mobj) -> None:
        self.a_face(mo)
        if self.rnd() < 40:
            return
        if not self._target_alive(mo.target) or not self.check_sight(mo, mo.target):
            self.set_state(mo, "see")
            return
        self.set_state(mo, "missile", 1)

    def a_cyberattack(self, mo: Mobj) -> None:
        t = mo.target
        if t is None:
            return
        self.a_face(mo)
        ang = math.atan2(t.y - mo.y, t.x - mo.x)
        dist = max(math.hypot(t.x - mo.x, t.y - mo.y), 1.0)
        z = mo.z + 64
        slope = ((t.z + t.height / 2) - z) / dist
        self.spawn_missile(mo, "rocket", ang, slope, z=z)

    def a_troopattack(self, mo: Mobj) -> None:
        self.a_face(mo)
        if self.in_melee_range(mo):
            self.sound("claw", mo)
            self.damage(mo.target, mo, mo, (self.rnd() % 8 + 1) * 3)
            return
        self._missile_at(mo, "imp_ball")

    def a_sargattack(self, mo: Mobj) -> None:
        self.a_face(mo)
        if self.in_melee_range(mo):
            self.damage(mo.target, mo, mo, (self.rnd() % 10 + 1) * 4)

    def a_headattack(self, mo: Mobj) -> None:
        self.a_face(mo)
        if self.in_melee_range(mo):
            self.damage(mo.target, mo, mo, (self.rnd() % 6 + 1) * 10)
            return
        self._missile_at(mo, "caco_ball")

    def a_bruisattack(self, mo: Mobj) -> None:
        self.a_face(mo)
        if self.in_melee_range(mo):
            self.sound("claw", mo)
            self.damage(mo.target, mo, mo, (self.rnd() % 8 + 1) * 10)
            return
        self._missile_at(mo, "baron_ball")

    def a_skullattack(self, mo: Mobj) -> None:
        t = mo.target
        if t is None:
            return
        mo.skullfly = True
        self.sound("sklatk", mo)
        self.a_face(mo)
        mo.momx = math.cos(mo.angle) * SKULLSPEED
        mo.momy = math.sin(mo.angle) * SKULLSPEED
        dist = max(math.hypot(t.x - mo.x, t.y - mo.y) / SKULLSPEED, 1.0)
        mo.momz = ((t.z + t.height / 2) - mo.z) / dist

    def skull_hit(self, mo: Mobj, blocker: object) -> None:
        if isinstance(blocker, Mobj) and blocker.shootable:
            self.damage(blocker, mo, mo, (self.rnd() % 8 + 1) * 3)
        mo.skullfly = False
        mo.momx = mo.momy = mo.momz = 0.0
        if mo.alive:
            self.set_state(mo, "see")

    def noise_alert(self) -> None:
        """Gunfire wakes every monster that can hear it (through open cells; closed doors stop the sound)."""
        if self._zone_dirty or self.sound_zone is None:
            self._compute_zones()
        g = self.grid
        pz = self.sound_zone[g.cell_at(self.pmo.x, self.pmo.y)]
        for o in self.mobjs:
            if o.cls == "monster" and o.alive and not getattr(o, "heard", False):
                if self.sound_zone[g.cell_at(o.x, o.y)] == pz:
                    o.heard = True

    def _compute_zones(self) -> None:
        g = self.grid
        n = g.w * g.h
        zone = np.full(n, -1, np.int32)
        open_ = (~g.solid) & ((g.ch - g.fh) > 0.5)
        label = 0
        openl = open_.tolist()
        zl = zone.tolist()
        w = g.w
        for start in range(n):
            if not openl[start] or zl[start] >= 0:
                continue
            stack = [start]
            zl[start] = label
            while stack:
                c = stack.pop()
                for nb in (c - 1, c + 1, c - w, c + w):
                    if 0 <= nb < n and openl[nb] and zl[nb] < 0:
                        zl[nb] = label
                        stack.append(nb)
            label += 1
        self.sound_zone = np.array(zl, np.int32)
        self._zone_dirty = False

    # --- use: doors, switches, lifts ------------------------------------------------------------------------------

    def use_lines(self) -> None:
        mo = self.pmo
        g = self.grid
        ca, sa = math.cos(mo.angle), math.sin(mo.angle)
        here = g.cell_at(mo.x, mo.y)
        target = None
        for i in range(1, 21):
            t = i * (USERANGE + 8) / 20.0
            c = g.cell_at(mo.x + ca * t, mo.y + sa * t)
            if c == here:
                continue
            s = g.specs[c]
            if g.solid[c] or s.get("kind") in ("door", "lift") or g.ch[c] - g.fh[c] < 1:
                target = c
                break
            if abs(g.fh[c] - g.fh[here]) > MAXSTEP or s.get("kind") == "lift":
                target = c
                break
        if target is None and g.specs[here].get("kind") == "lift":
            target = here
        if target is None:
            return
        s = g.specs[target]
        kind = s.get("kind")
        if kind == "door":
            if s.get("tag") and not s.get("usable"):
                self.sound("noway")
                return
            key = s.get("key")
            if key and key not in self.player.keys:
                self.message(f"You need a {key} access card for this door.")
                self.sound("noway")
                return
            self.open_door(target)
        elif kind == "lift":
            self.activate_lift(target)
        elif s.get("action"):
            if target in self.switched and not s.get("repeat"):
                return
            self.switched.add(target)
            self.flip_switch(target)
            self.sound("swtchn")
            self.do_action(s["action"])
        else:
            self.sound("noway")

    def flip_switch(self, c: int) -> None:
        from .assets import tiles
        ts = tiles()
        g = self.grid
        cur = g.face_tex[0, c]
        names = {v: k for k, v in ts.wall_ids.items()}
        name = names.get(int(cur), "")
        swap = {"SWOFF": "SWON", "EXITOFF": "EXITON"}.get(name)
        if swap:
            g.face_tex[:, c] = ts.wall_ids[swap]

    def do_action(self, action) -> None:  # noqa: ANN001
        if isinstance(action, str):
            action = (action,)
        kind = action[0]
        if kind in ("exit", "secret_exit"):
            self.exit_pending = 20
            self.secret_exit = kind == "secret_exit"
        elif kind == "open":
            for c, s in enumerate(self.grid.specs):
                if s.get("tag") == action[1] and s.get("kind") == "door":
                    self.open_door(c, stay=True)
        elif kind in ("raise", "lower"):
            g = self.grid
            for c, s in enumerate(g.specs):
                if s.get("tag") == action[1] and s.get("kind") != "door":
                    m = Mover(c, "floor", top=float(action[2]), bottom=float(g.fh[c]), speed=2.0)
                    m.state = "move"
                    self.movers[c] = m
            self.sound("pstart")
        elif kind == "message":
            self.message(action[1])

    def open_door(self, c: int, by_monster: bool = False, stay: bool = False) -> None:
        m = self.movers.get(c)
        g = self.grid
        if m is None or m.kind != "door":
            return
        if m.state in ("up",):
            return
        if m.state == "wait" and not by_monster and not m.stay and not stay:
            m.state = "down"
            self.sound("dorcls", self._at_cell(c))
            return
        if m.state == "wait":
            return
        if g.ch[c] >= m.top and m.state == "idle" and not by_monster:
            if not m.stay:
                m.state = "down"
                self.sound("dorcls", self._at_cell(c))
            return
        if stay:
            m.stay = True
        m.state = "up"
        self.sound("doropn", self._at_cell(c))
        self._zone_dirty = True

    def _at_cell(self, c: int) -> Mobj:
        cy, cx = divmod(c, self.grid.w)
        m = Mobj("snd", "fx", (cx + 0.5) * CELL, (cy + 0.5) * CELL)
        Mobj._ids -= 1
        return m

    def activate_lift(self, c: int) -> None:
        m = self.movers.get(c)
        if m is None or m.state != "idle":
            return
        m.state = "down"
        self.sound("pstart", self._at_cell(c))

    def _cell_blocked(self, c: int, top: float) -> bool:
        cy, cx = divmod(c, self.grid.w)
        x0, y0 = cx * CELL, cy * CELL
        for o in [self.pmo] + self.mobjs:
            if not o.solid and o is not self.pmo:
                continue
            if o is self.pmo and self.player.dead:
                continue
            if x0 - o.radius < o.x < x0 + CELL + o.radius and y0 - o.radius < o.y < y0 + CELL + o.radius:
                if o.z + o.height > top - 0.01 or o.cls == "monster" or o is self.pmo:
                    return True
        return False

    def movers_think(self) -> None:
        g = self.grid
        for c, m in self.movers.items():
            if m.kind == "door":
                if m.state == "up":
                    g.ch[c] = min(g.ch[c] + m.speed, m.top)
                    if g.ch[c] >= m.top:
                        m.state = "idle" if m.stay else "wait"
                        m.wait = DOORWAIT
                elif m.state == "wait":
                    m.wait -= 1
                    if m.wait <= 0:
                        m.state = "down"
                        self.sound("dorcls", self._at_cell(c))
                elif m.state == "down":
                    nh = g.ch[c] - m.speed
                    if self._cell_blocked(c, nh + 0.0) and self._door_crush(c, nh):
                        m.state = "up"
                        self.sound("doropn", self._at_cell(c))
                        continue
                    g.ch[c] = max(nh, m.bottom)
                    if g.ch[c] <= m.bottom:
                        m.state = "idle"
                        self._zone_dirty = True
            elif m.kind == "lift":
                if m.state == "down":
                    g.fh[c] = max(g.fh[c] - m.speed, m.bottom)
                    self._carry(c)
                    if g.fh[c] <= m.bottom:
                        m.state = "wait"
                        m.wait = LIFTWAIT
                        self.sound("pstop", self._at_cell(c))
                elif m.state == "wait":
                    m.wait -= 1
                    if m.wait <= 0:
                        m.state = "up"
                        self.sound("pstart", self._at_cell(c))
                elif m.state == "up":
                    nh = min(g.fh[c] + m.speed, m.top)
                    if self._lift_blocked(c, nh):
                        m.state = "down"
                        continue
                    g.fh[c] = nh
                    self._carry(c)
                    if g.fh[c] >= m.top:
                        m.state = "idle"
                        self.sound("pstop", self._at_cell(c))
            elif m.kind == "floor" and m.state == "move":
                step = m.speed if m.top > g.fh[c] else -m.speed
                g.fh[c] = g.fh[c] + step
                if (step > 0 and g.fh[c] >= m.top) or (step < 0 and g.fh[c] <= m.top):
                    g.fh[c] = m.top
                    m.state = "done"
                    self.sound("pstop", self._at_cell(c))
                    self._zone_dirty = True
                self._carry(c)

    def _door_crush(self, c: int, nh: float) -> bool:
        cy, cx = divmod(c, self.grid.w)
        x0, y0 = cx * CELL, cy * CELL
        for o in [self.pmo] + self.mobjs:
            if o is self.pmo and self.player.dead:
                continue
            if not (o.solid or o is self.pmo):
                continue
            if x0 - o.radius < o.x < x0 + CELL + o.radius and y0 - o.radius < o.y < y0 + CELL + o.radius:
                if o.z + o.height > nh:
                    return True
        return False

    def _lift_blocked(self, c: int, nh: float) -> bool:
        cy, cx = divmod(c, self.grid.w)
        x0, y0 = cx * CELL, cy * CELL
        for o in [self.pmo] + self.mobjs:
            if o is self.pmo and self.player.dead:
                continue
            if not (o.solid or o is self.pmo) or o.floats:
                continue
            if x0 - o.radius < o.x < x0 + CELL + o.radius and y0 - o.radius < o.y < y0 + CELL + o.radius:
                if o.z < nh - 0.01 and self.grid.cell_at(o.x, o.y) != c:
                    return True
        return False

    def _carry(self, c: int) -> None:
        """Things standing on a moving floor move with it."""
        g = self.grid
        for o in [self.pmo] + self.mobjs:
            if o.cls in ("missile",) or o.floats:
                continue
            if c in self.cells_under(o.x, o.y, o.radius):
                fz, _ = self.floor_ceiling(o)
                if o.z < fz or (abs(o.z - fz) < LIFTSPEED + 1 and o.cls != "fx"):
                    if o is self.pmo and o.z < fz:
                        self.player.viewheight -= fz - o.z
                        self.player.deltaviewheight = (VIEWHEIGHT - self.player.viewheight) / 8.0
                    o.z = fz
        del g

    def sector_think(self) -> None:
        g = self.grid
        t = self.leveltime
        for c, s in self._effects():
            eff = s.get("effect")
            base = int(g.base_light[c])
            if eff == "blink":
                g.light[c] = base if (t // 8 + c * 7) % 5 else max(base - 96, 0)
            elif eff == "strobe":
                g.light[c] = base if (t // 17) % 2 else max(base - 112, 0)
            elif eff == "flicker":
                if self.rng.random() < 0.08:
                    g.light[c] = base if self.rng.random() < 0.6 else max(base - 64, 0)
            elif eff == "glow":
                g.light[c] = int(base - 48 + 48 * math.sin(t / 20.0))

    def _effects(self) -> list[tuple[int, dict]]:
        if not hasattr(self, "_effect_cells"):
            self._effect_cells = [(c, s) for c, s in enumerate(self.grid.specs) if s.get("effect")]
        return self._effect_cells

    # --- save / restore --------------------------------------------------------------------------------------------

    def to_json(self) -> dict:
        g = self.grid
        changed = np.nonzero((g.fh != self.init_fh) | (g.ch != self.init_ch))[0].tolist()
        saved = [m for m in self.mobjs if m.cls != "fx"]
        ids = {id(m): i for i, m in enumerate(saved)}

        def ref(m: Mobj | None) -> object:
            if m is None:
                return None
            if m is self.pmo:
                return "p"
            return ids.get(id(m))

        things = []
        for m in saved:
            things.append({
                "k": m.kind, "c": m.cls, "x": round(m.x, 2), "y": round(m.y, 2), "z": round(m.z, 2),
                "mx": round(m.momx, 3), "my": round(m.momy, 3), "mz": round(m.momz, 3), "a": round(m.angle, 4),
                "hp": m.hp, "st": m.state, "si": m.si, "ti": m.tics, "t": ref(m.target), "s": ref(m.source),
                "md": m.movedir, "mc": m.movecount, "rt": m.reactiontime, "th": m.threshold,
                "f": [int(m.solid), int(m.shootable), int(m.alive), int(m.skullfly), int(m.dropped),
                      int(getattr(m, "heard", False)), int(m.floats), int(m.nogravity)],
                "fr": list(m.frame) if isinstance(m.frame, tuple) else m.frame,
            })
        pm = self.pmo
        return {
            "time": self.leveltime,
            "player": {"x": round(pm.x, 2), "y": round(pm.y, 2), "z": round(pm.z, 2), "a": round(pm.angle, 4),
                       "kills": self.player.killcount, "items": self.player.itemcount,
                       "secrets": self.player.secretcount, "keys": sorted(self.player.keys),
                       "dead": self.player.dead, "powers": dict(self.player.powers)},
            "seen": np.flatnonzero(self.seen).tolist(),
            "things": things,
            "cells": [[c, float(g.fh[c]), float(g.ch[c])] for c in changed],
            "movers": [[c, m.state, m.wait, int(m.stay)] for c, m in self.movers.items() if m.state != "idle"
                       or m.stay],
            "floors": [[c, m.top, m.state] for c, m in self.movers.items() if m.kind == "floor"],
            "switched": sorted(self.switched), "secrets": sorted(self.secrets_found),
            "triggered": sorted(self.triggered), "totals": [self.total_kills, self.total_items],
            "boss_done": self.boss_done,
        }

    def from_json(self, d: dict) -> None:
        """Restore a saved level state (the world was built from the same map); bad data is skipped."""
        g = self.grid
        try:
            for c, fh, ch in d.get("cells", []):
                g.fh[int(c)], g.ch[int(c)] = float(fh), float(ch)
            for c, top, state in d.get("floors", []):
                m = Mover(int(c), "floor", top=float(top), speed=2.0)
                m.state = state
                self.movers[int(c)] = m
            for c, state, wait, stay in d.get("movers", []):
                m = self.movers.get(int(c))
                if m is not None:
                    m.state, m.wait, m.stay = str(state), int(wait), bool(stay)
            for c in d.get("switched", []):
                self.switched.add(int(c))
                self.flip_switch(int(c))
            self.secrets_found = {int(c) for c in d.get("secrets", [])}
            self.triggered = {int(c) for c in d.get("triggered", [])}
            self.leveltime = int(d.get("time", 0))
            self.boss_done = bool(d.get("boss_done", False))
            tk, ti = d.get("totals", [self.total_kills, self.total_items])
            self.total_kills, self.total_items = int(tk), int(ti)
            p = d.get("player") or {}
            pm = self.pmo
            pm.x, pm.y = float(p.get("x", pm.x)), float(p.get("y", pm.y))
            pm.z = float(p.get("z", pm.z))
            pm.angle = float(p.get("a", pm.angle))
            self.player.killcount = int(p.get("kills", 0))
            self.player.itemcount = int(p.get("items", 0))
            self.player.secretcount = int(p.get("secrets", 0))
            self.player.keys = {k for k in p.get("keys", []) if k in ("blue", "yellow", "red")}
            pw = p.get("powers")
            if isinstance(pw, dict):
                self.player.powers = {k: int(v) for k, v in pw.items()
                                      if k in ("invuln", "suit", "berserk", "scanner") and int(v) > 0}
            for c in d.get("seen", []) or []:
                if 0 <= int(c) < len(self.seen):
                    self.seen[int(c)] = True
            things = d.get("things")
            if isinstance(things, list):
                self._restore_things(things)
        except (TypeError, ValueError, KeyError, IndexError):
            pass
        self.player.viewz = self.pmo.z + VIEWHEIGHT
        self._zone_dirty = True

    def _restore_things(self, things: list) -> None:
        made: list[Mobj] = []
        for t in things:
            kind, cls = t.get("k"), t.get("c")
            if cls == "monster" and kind in MONSTER_INFO:
                m = self.spawn(kind, float(t["x"]), float(t["y"]))
                self.total_kills -= 1
            elif cls == "missile" and kind in MISSILE_INFO:
                m = Mobj(kind, "missile", float(t["x"]), float(t["y"]))
                info = MISSILE_INFO[kind]
                m.info = info
                m.radius, m.height = float(info["radius"]), float(info["height"])
                m.bright = True
                self.mobjs.append(m)
            elif cls in ("item", "decor", "barrel") and (kind in ITEM_INFO or kind in DECOR_INFO or kind == "barrel"):
                m = self.spawn(kind, float(t["x"]), float(t["y"]))
                if cls == "item" and ITEM_INFO.get(kind, {}).get("count"):
                    self.total_items -= 1
            else:
                continue
            m.z = float(t.get("z", m.z))
            m.momx, m.momy, m.momz = float(t.get("mx", 0)), float(t.get("my", 0)), float(t.get("mz", 0))
            m.angle = float(t.get("a", 0))
            m.hp = int(t.get("hp", m.hp))
            m.state, m.si, m.tics = str(t.get("st", "spawn")), int(t.get("si", 0)), int(t.get("ti", 1))
            m.movedir, m.movecount = int(t.get("md", NODIR)), int(t.get("mc", 0))
            m.reactiontime, m.threshold = int(t.get("rt", 0)), int(t.get("th", 0))
            f = t.get("f", [])
            if len(f) >= 8:
                m.solid, m.shootable, m.alive, m.skullfly, m.dropped = map(bool, f[:5])
                m.heard, m.floats, m.nogravity = bool(f[5]), bool(f[6]), bool(f[7])
            fr = t.get("fr", 0)
            m.frame = tuple(fr) if isinstance(fr, list) else fr
            if m.cls == "monster":
                st = m.info["states"].get(m.state)
                if st is None or not st["states"]:
                    self.set_state(m, "see" if m.alive else "death")
                else:
                    m.si = min(max(m.si, 0), len(st["states"]) - 1)
                    m.frame = st["states"][m.si][0]
                    m.bright = st["states"][m.si][3]
            made.append(m)
        # the map's things were replaced by the saved ones
        keep = set(id(m) for m in made)
        self.mobjs = [m for m in self.mobjs if id(m) in keep]
        for m, t in zip(made, [t for t in things if _restorable(t)], strict=False):
            for attr, key in (("target", "t"), ("source", "s")):
                r = t.get(key)
                if r == "p":
                    setattr(m, attr, self.pmo)
                elif isinstance(r, int) and 0 <= r < len(things):
                    # refs are indices into the saved list; map them to the made list
                    idx = _ref_index(things, r)
                    setattr(m, attr, made[idx] if idx is not None and idx < len(made) else None)


def _restorable(t: dict) -> bool:
    kind, cls = t.get("k"), t.get("c")
    if cls == "monster":
        return kind in MONSTER_INFO
    if cls == "missile":
        return kind in MISSILE_INFO
    if cls in ("item", "decor", "barrel"):
        return kind in ITEM_INFO or kind in DECOR_INFO or kind == "barrel"
    return False


def _ref_index(things: list, r: int) -> int | None:
    if not _restorable(things[r]):
        return None
    return sum(1 for t in things[:r] if _restorable(t))
