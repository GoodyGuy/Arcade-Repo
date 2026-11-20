"""The simulation of BREACH POINT: ten actors on a map, their movement and guns, abilities and what they leave in the
world, the charge, the rounds and the economy. The player and the bots drive their actors through the same ``Cmd``;
nothing here draws or plays a sound (``Match.sounds`` and ``Match.fx`` are read by the game widget)."""

from __future__ import annotations

import math
import random

from . import config as C
from .config import AGENTS, WEAPONS
from .maps import M_BARRICADE, GameMap

DEG = math.pi / 180.0
SLOTS = ("C", "Q", "E", "X")


class Cmd:
    """One frame of what an actor wants to do."""
    __slots__ = ("ability", "ads", "crouch", "fire", "fwd", "jump", "lob", "reload", "side", "slot", "use", "walk")

    def __init__(self) -> None:
        self.fwd = self.side = 0.0
        self.walk = self.crouch = self.jump = self.fire = self.ads = self.reload = self.use = False
        self.slot: str | None = None  # main / side / knife
        self.ability: str | None = None  # C / Q / E / X
        self.lob: tuple | None = None  # (x, y) a bot throws to


class Actor:
    def __init__(self, idx: int, squad: int, agent: str, name: str, human: bool = False) -> None:
        self.idx, self.squad, self.agent, self.name, self.human = idx, squad, AGENTS[agent], name, human
        self.side = "atk"
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.yaw = self.pitch = 0.0
        self.on_ground = True
        self.crouch = 0.0
        self.hp, self.armour, self.armour_max = C.HP, 0.0, 0.0
        self.alive = True
        self.credits = C.CREDITS_START
        self.main: str | None = None
        self.sidearm = "tack"
        self.cur = "side"
        self.ammo: dict[str, list] = {}
        self.cool = self.reload_t = self.equip_t = 0.0
        self.bloom = self.rec_p = self.rec_y = 0.0  # degrees
        self.shots = 0
        self.since_shot = 9.0
        self.held = False
        self.ads = 0.0
        self.charges = {"C": 0, "Q": 0, "E": 1, "X": 0}
        self.recharge = 0.0
        self.ult = 0
        self.blind = self.daze = self.revealed = self.tag = self.land_t = 0.0
        self.near_t, self.near_r = 0.0, 99.0
        self.buffs: dict[str, list] = {}  # ability id -> [time left, numbers]
        self.heals: list = []  # [per second, time left]
        self.dash: list | None = None  # [time left, vx, vy]
        self.has_charge = False
        self.plant_t = 0.0
        self.defusing = False
        self.kills = self.deaths = self.assists = 0
        self.round_kills = 0
        self.stats = {"shots": 0, "hits": 0, "heads": 0, "damage": 0, "plants": 0, "defuses": 0, "abilities": 0}
        self.bought: list = []  # (item, price) this buy phase
        self.hurt_by: dict[int, float] = {}  # actor index -> time of its last damage
        self.dead_t = 0.0
        self.phase = 0.0  # walk cycle
        self.step_acc = 0.0
        self.kick = 0.0  # view-model kick
        self.hit_dirs: list = []  # [angle, time left] where damage came from (the HUD)
        self.brain = None  # bots.Brain
        self.slow = 1.0
        self.spectating = False

    # --- what it holds ------------------------------------------------------------------------------------------

    @property
    def wid(self) -> str:
        if self.cur == "main" and self.main:
            return self.main
        if self.cur == "knife":
            return "knife"
        return self.sidearm

    @property
    def weapon(self) -> C.Weapon:
        return WEAPONS[self.wid]

    def mag(self) -> list:
        return self.ammo.setdefault(self.wid, [0, 0])

    def give(self, wid: str) -> None:
        w = WEAPONS[wid]
        self.ammo[wid] = [w.mag, w.mag * C.MAG_RESERVE]
        if w.group == "side":
            self.sidearm = wid
        else:
            self.main = wid

    def best_slot(self) -> str:
        return "main" if self.main else "side"

    @property
    def eye(self) -> float:
        return self.z + C.EYE - C.CROUCH_DROP * self.crouch

    @property
    def top(self) -> float:
        return self.z + C.HEIGHT - C.CROUCH_DROP * self.crouch

    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def top_speed(self) -> float:
        return C.RUN * self.weapon.speed

    def buff(self, key: str, default: float = 1.0) -> float:
        v = default
        for _t, p in self.buffs.values():
            if key in p:
                v *= p[key]
        return v

    def scoped(self) -> bool:
        return self.weapon.scoped and self.ads > 0.9

    def spread(self) -> float:
        """Degrees: how wide the next shot may go."""
        w = self.weapon
        if w.cls == "melee":
            return 0.0
        steady = self.buff("steady")
        frac = self.speed() / max(self.top_speed(), 0.1)
        move = 0.0 if frac < C.STILL else w.move * min((frac - C.STILL) / (1 - C.STILL), 1.2) * steady
        base = w.spread
        if w.scoped:
            base = 0.0 if self.ads > 0.9 else w.spread
        elif self.ads > 0.5:
            base *= 0.85
        s = (base + self.bloom) * (1.0 - 0.18 * self.crouch) + move
        if not self.on_ground:
            s += (4.0 + w.move * 0.6) * steady
        elif self.land_t > 0:
            s += 2.2
        if self.daze > 0:
            s += 1.5
        return s

    def view(self) -> tuple:
        """(yaw, pitch) the eye looks along: the aim plus the recoil."""
        return self.yaw + self.rec_y * DEG, min(max(self.pitch + self.rec_p * DEG, -1.5), 1.5)

    def can_use(self, slot: str) -> bool:
        ab = self.agent.ability(slot)
        return self.ult >= ab.points if slot == "X" else self.charges[slot] > 0


def view_dir(yaw: float, pitch: float) -> tuple:
    cp = math.cos(pitch)
    return cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)


class Match:
    """A whole game: first to ``win_rounds``. Squad 0 is the player's team."""

    def __init__(self, gmap: GameMap, agent: str, difficulty: str = "normal", win_rounds: int = C.WIN_ROUNDS,
                 player_side: str = "atk", seed: int | None = None, player_name: str = "YOU",
                 autopilot: bool = False, buy_time: float = C.BUY_TIME) -> None:
        self.map = gmap
        self.rng = random.Random(seed)
        self.difficulty = difficulty
        self.win_rounds = win_rounds
        self.buy_time = buy_time
        self.sides = {0: player_side, 1: "def" if player_side == "atk" else "atk"}
        names = list(C.BOT_NAMES)
        self.rng.shuffle(names)
        self.actors: list[Actor] = []
        mates = [agent] + [a for a in C.AGENT_ORDER if a != agent]
        for i, ag in enumerate(mates):
            self.actors.append(Actor(i, 0, ag, player_name if i == 0 else names.pop(), human=(i == 0)))
        for i, ag in enumerate(C.AGENT_ORDER):
            self.actors.append(Actor(5 + i, 1, ag, names.pop()))
        self.player = self.actors[0]
        self.score = [0, 0]
        self.streak = [0, 0]  # losses in a row
        self.round = 0  # rounds played
        self.phase = "buy"  # buy / live / planted / end / over
        self.t = 0.0  # s left in the phase
        self.time = 0.0
        self.charge: dict = {"state": "none"}
        self.smokes: list = []  # [x, y, z, r, age, dur]
        self.zones: list = []  # dict
        self.sensors: list = []
        self.pings: list = []
        self.projectiles: list = []
        self.blasts: list = []  # pending [delay, x, y, z, r, dmg, owner]
        self.barricades: list = []  # [time left, old cells, cells, age]
        self.drops: list = []  # [x, y, weapon id, mag, reserve]
        self.tracers: list = []  # [x0, y0, z0, x1, y1, z1, age, squad]
        self.puffs: list = []  # [x, y, z, age, kind]
        self.sounds: list = []  # (name, x, y, loudness) for the widget
        self.noises: list = []  # (x, y, radius, squad, kind, actor) made this tick
        self.heard: list = []  # the tick before's: what the bots hear
        self.used_abilities: dict[str, int] = {}
        self.feed: list = []  # dict, newest last
        self.banner: list | None = None  # [text, sub, colour key, time left]
        self.fx: list = []  # things for the HUD of the watched actor: ("hit", head, kill) ...
        self.winner: int | None = None  # squad of the round just ended
        self.win_how = ""
        self.match_winner: int | None = None
        self.half_swapped = False
        self.ready = False
        self.geometry_dirty = False  # a barricade went up or down: the renderer rebuilds what it must
        self.team_ai: dict = {}  # squad -> bots.TeamAI
        self.round_log: list = []  # (winner squad, how)
        self.autopilot = autopilot  # the player's actor driven by a bot brain (tests, demo)
        self._new_round(first=True)

    # --- helpers ------------------------------------------------------------------------------------------------

    def side(self, a: Actor) -> str:
        return self.sides[a.squad]

    def squad_of(self, side: str) -> int:
        return 0 if self.sides[0] == side else 1

    def team(self, squad: int, alive: bool = True) -> list:
        return [a for a in self.actors if a.squad == squad and (a.alive or not alive)]

    def enemies(self, a: Actor) -> list:
        return [b for b in self.actors if b.squad != a.squad and b.alive]

    def sound(self, name: str, x: float, y: float, loud: float = 1.0) -> None:
        self.sounds.append((name, x, y, loud))

    def say(self, text: str, sub: str = "", colour: str = "accent", t: float = 2.6) -> None:
        self.banner = [text, sub, colour, t]

    # --- rounds -------------------------------------------------------------------------------------------------

    def _new_round(self, first: bool = False) -> None:
        m = self.map
        decider = self.score[0] == self.score[1] == self.win_rounds - 1
        for b in self.barricades:
            m.restore_cells(b[1])
            self.geometry_dirty = True
        self.barricades.clear()
        for lst in (self.smokes, self.zones, self.sensors, self.pings, self.projectiles, self.blasts, self.drops,
                    self.tracers, self.puffs):
            lst.clear()
        order = {0: list(range(5)), 1: list(range(5))}
        for a in self.actors:
            a.side = self.sides[a.squad]
            sp = m.spawns[a.side][order[a.squad].pop(0)]
            a.x, a.y = sp
            a.z = m.ground(a.x, a.y, 0.0)
            a.vx = a.vy = a.vz = 0.0
            a.yaw, a.pitch = m.spawn_yaw[a.side], 0.0
            if not a.alive or first:
                a.main, a.sidearm, a.armour, a.armour_max = None, "tack", 0.0, 0.0
                a.ammo = {}
            for wid in (a.main, a.sidearm):
                if wid:
                    w = WEAPONS[wid]
                    a.ammo[wid] = [w.mag, w.mag * C.MAG_RESERVE]
            a.alive, a.hp = True, C.HP
            a.cur = a.best_slot()
            a.cool = a.reload_t = a.bloom = a.rec_p = a.rec_y = 0.0
            a.equip_t = 0.0
            a.blind = a.daze = a.revealed = a.tag = a.near_t = a.plant_t = a.land_t = 0.0
            a.buffs.clear()
            a.heals.clear()
            a.dash = None
            a.has_charge = a.defusing = False
            a.crouch = a.ads = 0.0
            a.round_kills = 0
            a.bought = []
            a.hurt_by.clear()
            a.hit_dirs.clear()
            a.spectating = False
            a.on_ground = True
            a.charges["E"] = a.agent.ability("E").charges
            a.recharge = 0.0
            if not first:
                a.ult = min(a.ult + C.ULT_ROUND, a.agent.ability("X").points)
            if decider:
                a.credits = max(a.credits, C.CREDITS_DECIDER)
        atk = self.team(self.squad_of("atk"))
        carrier = self.rng.choice(atk)
        carrier.has_charge = True
        self.charge = {"state": "carried", "by": carrier}
        self.phase, self.t, self.ready = "buy", self.buy_time, False
        self.winner, self.win_how = None, ""
        from . import bots

        for sq in (0, 1):
            self.team_ai[sq] = bots.TeamAI(self, sq)
        for a in self.actors:
            if not a.human or self.autopilot:
                if a.brain is None:
                    a.brain = bots.Brain(self, a)
                a.brain.new_round()
        n = self.round + 1
        self.say(f"ROUND {n}", "MATCH POINT" if max(self.score) == self.win_rounds - 1 else
                 ("LAST ROUND OF THE HALF" if self.round == self.win_rounds - 2 else "BUY PHASE"), "accent", 3.0)
        self.sound("round_start", self.player.x, self.player.y, 9.0)

    def set_ready(self) -> None:
        if self.phase == "buy" and self.t > 3.0:
            self.t = 3.0
            self.ready = True

    def _end_round(self, squad: int, how: str) -> None:
        if self.phase in ("end", "over"):
            return
        planted = self.charge["state"] in ("planted", "boom", "defused")
        self.winner, self.win_how = squad, how
        self.round_log.append((squad, how, self.sides[squad]))
        self.score[squad] += 1
        loser = 1 - squad
        self.streak[squad] = 0
        self.streak[loser] += 1
        for a in self.actors:
            if a.squad == squad:
                gain = C.CREDIT_WIN
            else:
                gain = C.CREDIT_LOSS[min(self.streak[loser], 3) - 1]
                failed = (a.side == "atk" and not planted) or (a.side == "def" and how == "detonation")
                if a.alive and failed:
                    gain = C.CREDIT_SURVIVED_LOSS
            a.credits = min(a.credits + gain, C.CREDITS_CAP)
        self.round += 1
        self.phase, self.t = "end", C.END_TIME
        mine = squad == 0
        text = {"elimination": "TEAM ELIMINATED", "detonation": "CHARGE DETONATED", "defuse": "CHARGE DEFUSED",
                "time": "TIME RAN OUT"}[how]
        self.say("ROUND WON" if mine else "ROUND LOST", text, "ally" if mine else "enemy", C.END_TIME)
        self.sound("round_win" if mine else "round_loss", self.player.x, self.player.y, 9.0)

    def _after_round(self) -> None:
        if max(self.score) >= self.win_rounds:
            self.match_winner = 0 if self.score[0] > self.score[1] else 1
            self.phase, self.t = "over", 0.0
            self.say("VICTORY" if self.match_winner == 0 else "DEFEAT", f"{self.score[0]} - {self.score[1]}",
                     "ally" if self.match_winner == 0 else "enemy", 99.0)
            self.sound("match_win" if self.match_winner == 0 else "match_loss", self.player.x, self.player.y, 9.0)
            return
        if not self.half_swapped and self.round >= self.win_rounds - 1:
            self.half_swapped = True
            self.sides = {0: self.sides[1], 1: self.sides[0]}
            self.streak = [0, 0]
            for a in self.actors:
                a.alive = False  # nothing is carried over a half
                a.credits = C.CREDITS_START
                a.charges["C"] = a.charges["Q"] = 0
            self._new_round(first=True)
            self.say("SIDES SWAPPED", f"YOU NOW {'ATTACK' if self.sides[0] == 'atk' else 'DEFEND'}", "accent", 3.5)
            return
        self._new_round()

    # --- buying -------------------------------------------------------------------------------------------------

    def price(self, a: Actor, item: str) -> int:
        if item in WEAPONS:
            return WEAPONS[item].price
        if item in C.ARMOUR:
            return C.ARMOUR[item][0]
        return a.agent.ability(item).cost

    def can_buy(self, a: Actor, item: str) -> str:
        """'' when the purchase is possible, else why not."""
        if self.phase != "buy" or not a.alive:
            return "Buy phase only"
        if item in WEAPONS:
            w = WEAPONS[item]
            if (a.sidearm if w.group == "side" else a.main) == item:
                return "Owned"
            cost = w.price - self._refund_for(a, w.group)
        elif item in C.ARMOUR:
            cost, pool, _n = C.ARMOUR[item]
            if a.armour >= pool:
                return "Owned"
            cost -= self._refund_for(a, "armour")
        else:
            ab = a.agent.ability(item)
            if a.charges[item] >= ab.charges:
                return "Full"
            cost = ab.cost
        return "" if a.credits >= cost else "Not enough credits"

    def _refund_for(self, a: Actor, group: str) -> int:
        """What comes back for the thing of this group bought earlier this buy phase."""
        for item, price in a.bought:
            g = "armour" if item in C.ARMOUR else (("side" if WEAPONS[item].group == "side" else "main")
                                                   if item in WEAPONS else item)
            if g == ("main" if group in ("light", "heavy") else group):
                return price
        return 0

    def buy(self, a: Actor, item: str) -> bool:
        if self.can_buy(a, item):
            return False
        if item in WEAPONS:
            w = WEAPONS[item]
            grp = "side" if w.group == "side" else "main"
            self._drop_bought(a, grp)
            a.credits -= w.price
            a.give(item)
            a.cur = a.best_slot()
            a.bought.append((item, w.price))
        elif item in C.ARMOUR:
            cost, pool, _n = C.ARMOUR[item]
            self._drop_bought(a, "armour")
            a.credits -= cost
            a.armour = a.armour_max = float(pool)
            a.bought.append((item, cost))
        else:
            ab = a.agent.ability(item)
            a.credits -= ab.cost
            a.charges[item] += 1
            a.bought.append((item, ab.cost))
        return True

    def _drop_bought(self, a: Actor, group: str) -> None:
        for i, (item, price) in enumerate(a.bought):
            g = "armour" if item in C.ARMOUR else (("side" if WEAPONS[item].group == "side" else "main")
                                                   if item in WEAPONS else item)
            if g == group:
                a.credits += price
                del a.bought[i]
                return

    def sell(self, a: Actor, item: str) -> bool:
        """Give back something bought this buy phase."""
        if self.phase != "buy":
            return False
        for i, (it, price) in enumerate(a.bought):
            if it != item:
                continue
            a.credits += price
            del a.bought[i]
            if item in WEAPONS:
                if WEAPONS[item].group == "side":
                    a.sidearm = "tack"
                    a.give("tack")
                else:
                    a.main = None
                a.cur = a.best_slot()
            elif item in C.ARMOUR:
                a.armour = a.armour_max = 0.0
            else:
                a.charges[item] -= 1
            return True
        return False

    # --- seeing -------------------------------------------------------------------------------------------------

    def smoked(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> bool:
        """A smoke stands on the line between two points."""
        dx, dy, dz = bx - ax, by - ay, bz - az
        ll = dx * dx + dy * dy + dz * dz
        if ll < 1e-6:
            return False
        for s in self.smokes:
            r = s[3] * min(s[4] / 0.6, 1.0)
            fx, fy, fz = s[0] - ax, s[1] - ay, s[2] - az
            k = (fx * dx + fy * dy + fz * dz) / ll
            k = 0.0 if k < 0 else (1.0 if k > 1 else k)
            ex, ey, ez = fx - dx * k, fy - dy * k, fz - dz * k
            if ex * ex + ey * ey + ez * ez < r * r * 0.86:
                return True
        return False

    def can_see(self, a: Actor, b: Actor) -> bool:
        """``a`` has a clear line to ``b`` (its field of view is the caller's business)."""
        if a.blind > 0.25:
            return False
        dx, dy = b.x - a.x, b.y - a.y
        if a.near_t > 0 and dx * dx + dy * dy > a.near_r * a.near_r:
            return False
        ez = a.eye
        m = self.map
        for tz in (b.top - 0.15, b.z + 0.9):
            if m.los(a.x, a.y, ez, b.x, b.y, tz):
                return not self.smoked(a.x, a.y, ez, b.x, b.y, tz)
        return False

    def visible_to_squad(self, b: Actor, squad: int) -> bool:
        if b.revealed > 0:
            return True
        for a in self.actors:
            if a.squad == squad and a.alive and self._in_fov(a, b, 1.05) and self.can_see(a, b):
                return True
        return False

    @staticmethod
    def _in_fov(a: Actor, b: Actor, half: float) -> bool:
        ang = math.atan2(b.y - a.y, b.x - a.x) - a.yaw
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        return abs(ang) < half

    # --- the tick -----------------------------------------------------------------------------------------------

    def update(self, dt: float, cmd: Cmd | None) -> None:
        """Advance by ``dt``. ``cmd``: the player's command (its view angles already set on the actor)."""
        self.time += dt
        self.map.budget = 2
        self.heard, self.noises = self.noises, []
        if self.banner is not None:
            self.banner[3] -= dt
            if self.banner[3] <= 0:
                self.banner = None
        if self.phase == "over":
            self._effects(dt)
            return
        self.t -= dt
        if self.phase == "buy" and self.t <= 0:
            self.phase, self.t = "live", C.ROUND_TIME
            self.say("GO", "PLANT THE CHARGE" if self.player.side == "atk" else "HOLD THE SITES", "accent", 2.0)
            self.sound("go", self.player.x, self.player.y, 9.0)
        elif self.phase == "live" and self.t <= 0:
            self._end_round(self.squad_of("def"), "time")
        elif self.phase == "end" and self.t <= 0:
            self._after_round()
            return
        from . import bots

        for ai in self.team_ai.values():
            ai.update(dt)
        for a in self.actors:
            if not a.alive:
                a.dead_t += dt
                continue
            if a.human and not self.autopilot:
                c = cmd or Cmd()
            else:
                if a.brain is None:
                    a.brain = bots.Brain(self, a)
                c = a.brain.think(dt)
            self._actor(a, c, dt)
        self._separate()
        self._charge(dt)
        self._effects(dt)
        if self.phase in ("live", "planted"):
            self._check_end()

    def _check_end(self) -> None:
        atk, dfn = self.squad_of("atk"), self.squad_of("def")
        if not self.team(dfn):
            self._end_round(atk, "elimination")
        elif not self.team(atk) and self.charge["state"] != "planted":
            self._end_round(dfn, "elimination")

    # --- one actor ----------------------------------------------------------------------------------------------

    def _actor(self, a: Actor, c: Cmd, dt: float) -> None:
        # timers
        for name in ("blind", "daze", "revealed", "tag", "land_t", "near_t", "cool", "equip_t"):
            v = getattr(a, name)
            if v > 0:
                setattr(a, name, max(v - dt, 0.0))
        a.since_shot += dt
        a.kick = max(a.kick - dt * 7.0, 0.0)
        for h in a.hit_dirs:
            h[1] -= dt
        if a.hit_dirs and a.hit_dirs[0][1] <= 0:
            a.hit_dirs = [h for h in a.hit_dirs if h[1] > 0]
        for key in list(a.buffs):
            b = a.buffs[key]
            b[0] -= dt
            if "heal" in b[1]:
                a.hp = min(a.hp + b[1]["heal"] / b[1]["dur"] * dt, C.HP)
            if b[0] <= 0:
                del a.buffs[key]
        for h in a.heals:
            h[1] -= dt
            a.hp = min(a.hp + h[0] * dt, C.HP)
        if a.heals and a.heals[0][1] <= 0:
            a.heals = [h for h in a.heals if h[1] > 0]
        ab = a.agent.ability("E")
        if a.charges["E"] < ab.charges and self.phase != "buy":
            a.recharge += dt
            if a.recharge >= ab.recharge:
                a.recharge = 0.0
                a.charges["E"] += 1
        w = a.weapon
        gap = a.since_shot - 1.0 / max(w.rps, 0.5)
        if gap > 0.06:  # the gun settles
            k = math.exp(-dt * 3.2 / max(w.recover, 0.1))
            a.bloom *= k
            a.rec_p *= k
            a.rec_y *= k
            if a.rec_p < 0.25 and a.bloom < 0.12:
                a.shots = 0
        # stance
        a.crouch += ((1.0 if c.crouch else 0.0) - a.crouch) * min(dt * 11.0, 1.0)
        busy = a.plant_t > 0 or a.defusing
        want_ads = c.ads and w.zoom > 1.0 and a.reload_t <= 0 and a.equip_t <= 0 and not busy
        a.ads += ((1.0 if want_ads else 0.0) - a.ads) * min(dt * (9.0 if w.scoped else 14.0), 1.0)
        self._move(a, c, dt, busy)
        # weapons
        if c.slot and c.slot != a.cur and not busy and (c.slot != "main" or a.main):
            a.cur = c.slot
            a.equip_t = a.weapon.equip
            a.reload_t = 0.0
            a.ads = 0.0
            self.sound("equip", a.x, a.y, 0.35)
            w = a.weapon
        if a.reload_t > 0:
            a.reload_t -= dt
            if a.reload_t <= 0:
                mag = a.mag()
                take = min(w.mag - mag[0], mag[1])
                mag[0] += take
                mag[1] -= take
        elif c.reload and not busy:
            self.start_reload(a)
        if c.ability and not busy and self.phase in ("live", "planted"):
            self.use_ability(a, c.ability, c.lob)
        if not busy and self.phase in ("live", "planted", "end"):
            self._trigger(a, c)
        a.held = c.fire
        # the charge and what lies on the floor
        self._objective(a, c, dt)

    def start_reload(self, a: Actor) -> bool:
        w = a.weapon
        if w.cls == "melee" or a.reload_t > 0 or a.equip_t > 0:
            return False
        mag = a.mag()
        if mag[0] >= w.mag or mag[1] <= 0:
            return False
        a.reload_t = w.reload * a.buff("reload")
        a.ads = 0.0
        self.sound("reload", a.x, a.y, 0.5)
        return True

    def _move(self, a: Actor, c: Cmd, dt: float, rooted: bool) -> None:
        m = self.map
        top = a.top_speed() * a.buff("speed")
        if a.scoped():
            top *= C.SCOPED_SPEED
        mode = 1.0
        if a.crouch > 0.5:
            mode = C.CROUCH
        elif c.walk:
            mode = C.WALK
        slow = a.slow
        if a.tag > 0:
            slow *= C.TAG_SLOW + (1 - C.TAG_SLOW) * (1 - a.tag / C.TAG_TIME)
        if a.daze > 0:
            slow *= 0.7
        top *= mode * slow
        fx, fy = math.cos(a.yaw), math.sin(a.yaw)
        wx = fx * c.fwd - fy * c.side
        wy = fy * c.fwd + fx * c.side
        n = math.hypot(wx, wy)
        if n > 1.0:
            wx, wy = wx / n, wy / n
        if rooted:
            wx = wy = 0.0
        if a.dash is not None:
            a.dash[0] -= dt
            a.vx, a.vy = a.dash[1], a.dash[2]
            if a.dash[0] <= 0:
                a.dash = None
                a.vx *= 0.3
                a.vy *= 0.3
        elif a.on_ground:
            rate = C.ACCEL if n > 0.01 else C.BRAKE
            k = min(rate * dt, 1.0)
            a.vx += (wx * top - a.vx) * k
            a.vy += (wy * top - a.vy) * k
            if c.jump and not rooted and a.daze <= 0:
                a.vz = C.JUMP
                a.on_ground = False
                self.noises.append((a.x, a.y, 14.0, a.squad, "step", a))
        else:
            a.vx += wx * top * C.AIR_CONTROL * dt
            a.vy += wy * top * C.AIR_CONTROL * dt
            sp = math.hypot(a.vx, a.vy)
            lim = max(top, 0.1) * 1.05
            if sp > lim and a.dash is None:
                a.vx, a.vy = a.vx / sp * lim, a.vy / sp * lim
        sp = math.hypot(a.vx, a.vy)
        steps = max(int(sp * dt / 0.28) + 1, 1)
        sdt = dt / steps
        blocked = None if self.phase != "buy" else m.zones[a.side]
        for _ in range(steps):
            a.x += a.vx * sdt
            a.y += a.vy * sdt
            self._collide(a, blocked)
        g = m.ground(a.x, a.y, a.z)
        if a.on_ground:
            if g < a.z - C.STEP:  # walked off an edge
                a.on_ground = False
                a.vz = 0.0
            else:
                a.z = g
        if not a.on_ground:
            a.vz -= C.GRAVITY * dt
            a.z += a.vz * dt
            g = m.ground(a.x, a.y, a.z)
            if a.z <= g and a.vz <= 0:
                if a.vz < -4.0:
                    a.land_t = 0.25
                    self.noises.append((a.x, a.y, 16.0, a.squad, "step", a))
                    self.sound("land", a.x, a.y, 0.6)
                a.z, a.vz, a.on_ground = g, 0.0, True
        # footsteps
        if a.on_ground and sp > 0.5:
            a.phase += sp * dt * 1.9
            a.step_acc += sp * dt
            if a.step_acc > 2.3:
                a.step_acc = 0.0
                if sp > a.top_speed() * 0.62 and a.crouch < 0.5:
                    self.noises.append((a.x, a.y, 20.0, a.squad, "step", a))
                    self.sound("step", a.x, a.y, 0.55)

    def _collide(self, a: Actor, zone) -> None:  # noqa: ANN001
        hl, W, H = self.map.hl, self.map.W, self.map.H
        r = C.RADIUS
        lim = a.z + C.STEP
        for _ in range(2):
            cx0, cy0 = int(a.x), int(a.y)
            moved = False
            for cy in (cy0 - 1, cy0, cy0 + 1):
                for cx in (cx0 - 1, cx0, cx0 + 1):
                    if 0 <= cx < W and 0 <= cy < H:
                        if hl[cy][cx] <= lim and (zone is None or zone[cy, cx] or hl[cy][cx] >= 2.0):
                            continue
                    px = cx if a.x < cx else (cx + 1 if a.x > cx + 1 else a.x)
                    py = cy if a.y < cy else (cy + 1 if a.y > cy + 1 else a.y)
                    dx, dy = a.x - px, a.y - py
                    d2 = dx * dx + dy * dy
                    if d2 >= r * r:
                        continue
                    moved = True
                    if d2 > 1e-10:
                        d = math.sqrt(d2)
                        a.x += dx / d * (r - d)
                        a.y += dy / d * (r - d)
                    else:  # inside the cell: out through the nearest side
                        ex, ey = a.x - (cx + 0.5), a.y - (cy + 0.5)
                        if abs(ex) > abs(ey):
                            a.x = cx + (1 + r if ex > 0 else -r)
                        else:
                            a.y = cy + (1 + r if ey > 0 else -r)
            if not moved:
                break

    def _separate(self) -> None:
        acts = [a for a in self.actors if a.alive]
        for i, a in enumerate(acts):
            for b in acts[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                d2 = dx * dx + dy * dy
                if 1e-6 < d2 < 0.36 and abs(a.z - b.z) < 1.2:
                    d = math.sqrt(d2)
                    push = (0.6 - d) * 0.25
                    ux, uy = dx / d * push, dy / d * push
                    for act, s in ((a, -1.0), (b, 1.0)):
                        nx, ny = act.x + ux * s, act.y + uy * s
                        if self.map.height(nx, ny) <= act.z + C.STEP:
                            act.x, act.y = nx, ny

    # --- shooting -----------------------------------------------------------------------------------------------

    def _trigger(self, a: Actor, c: Cmd) -> None:
        w = a.weapon
        if not c.fire or a.cool > 0 or a.equip_t > 0 or a.reload_t > 0:
            return
        if not w.auto and a.held:
            return
        if w.cls == "melee":
            self._melee(a)
            return
        mag = a.mag()
        if mag[0] <= 0:
            if not self.start_reload(a) and not a.held:
                self.sound("dry", a.x, a.y, 0.4)
            return
        mag[0] -= 1
        a.stats["shots"] += 1
        spread = a.spread() * DEG
        yaw, pitch = a.view()
        ox, oy, oz = a.x, a.y, a.eye
        rng = self.rng
        hit_any = False
        for _ in range(w.pellets):
            r = spread * math.sqrt(rng.random()) if w.pellets == 1 else (spread + w.cone * DEG) * math.sqrt(rng.random())
            th = rng.random() * 2 * math.pi
            dx, dy, dz = view_dir(yaw + r * math.cos(th), pitch + r * math.sin(th))
            hit_any |= self._bullet(a, ox, oy, oz, dx, dy, dz, w)
        if hit_any:
            a.stats["hits"] += 1
        a.cool = 1.0 / (w.rps * a.buff("rate"))
        a.since_shot = 0.0
        a.shots += 1
        a.bloom = min(a.bloom + w.bloom, w.bloom_max)
        if a.rec_p < w.rmax:
            a.rec_p += w.rv * (0.55 if a.shots == 1 else 1.0)
        else:
            a.rec_p += w.rv * 0.08 * (rng.random() - 0.5)
            a.rec_y += w.rh * (1.0 if math.sin(a.shots * 0.55) > 0 else -1.0) * (0.6 + rng.random() * 0.6)
        a.rec_y += w.rh * 0.22 * (rng.random() - 0.5)
        a.kick = 1.0
        if w.scoped:
            a.ads = 0.0
        loud = 0.55 if w.quiet else 1.0
        self.sound("shot_" + w.cls + ("_q" if w.quiet else ""), a.x, a.y, loud * 2.2)
        self.noises.append((a.x, a.y, 26.0 if w.quiet else 70.0, a.squad, "shot", a))

    def _hit_actor(self, b: Actor, ox: float, oy: float, oz: float, dx: float, dy: float, dz: float) -> tuple | None:
        fx, fy = ox - b.x, oy - b.y
        aa = dx * dx + dy * dy
        if aa < 1e-9:
            return None
        bq = fx * dx + fy * dy
        cc = fx * fx + fy * fy - 0.1024  # radius 0.32
        disc = bq * bq - aa * cc
        if disc < 0:
            return None
        sq = math.sqrt(disc)
        t1 = (-bq + sq) / aa
        if t1 < 0:
            return None
        t0 = max((-bq - sq) / aa, 0.0)
        top = b.top
        z0, z1 = oz + dz * t0, oz + dz * t1
        if (z0 > top and z1 > top) or (z0 < b.z and z1 < b.z):
            return None
        t = t0
        if z0 > top:
            t = (top - oz) / dz
        elif z0 < b.z:
            t = (b.z - oz) / dz
        tm = min((t + t1) * 0.5, t + 0.25)  # a little way inside: what the bullet is in
        z = oz + dz * tm
        perp2 = (fx * fx + fy * fy) - bq * bq / aa  # the line's distance from the middle, squared
        if z > top - 0.3:
            zone = "head" if perp2 < 0.2 * 0.2 else "body"
        elif z < b.z + (top - b.z) * 0.42:
            zone = "leg"
        else:
            zone = "body"
        return t, zone

    def _bullet(self, a: Actor, ox: float, oy: float, oz: float, dx: float, dy: float, dz: float,
                w: C.Weapon) -> bool:
        end, solids = self.map.trace(ox, oy, oz, dx, dy, dz, 90.0)
        power = C.PEN_POWER[w.pen]
        cost = 0.0
        stop = end
        for t0, t1, r in solids:
            need = (t1 - t0) * r
            if cost + need > power:
                stop = t0 + (power - cost) / r
                break
            cost += need
        # smokes do not stop bullets
        best = None
        for b in self.actors:
            if b is a or not b.alive or b.squad == a.squad:
                continue
            ex, ey = b.x - ox, b.y - oy
            along = ex * dx + ey * dy
            if along < -0.5 or along > stop + 0.5:
                continue
            got = self._hit_actor(b, ox, oy, oz, dx, dy, dz)
            if got is not None and got[0] < stop and (best is None or got[0] < best[0]):
                best = (got[0], got[1], b)
        tend = stop
        hit = False
        if best is not None:
            t, zone, b = best
            through = 0.0
            for t0, t1, r in solids:
                if t0 < t:
                    through += (min(t1, t) - t0) * r
            mult = w.head if zone == "head" else (w.leg if zone == "leg" else 1.0)
            fall = 1.0
            for dist, k in w.falloff:
                if t >= dist:
                    fall = k
            dmg = w.dmg * mult * fall * (1.0 - C.PEN_LOSS * through / power)
            self.damage(b, dmg, a, w.id, zone == "head", through > 0.01)
            tend, hit = t, True
            self.puffs.append([ox + dx * t, oy + dy * t, oz + dz * t, 0.0, "blood"])
        else:
            self.puffs.append([ox + dx * tend, oy + dy * tend, oz + dz * tend, 0.0, "dust"])
            for t0, _t1, _r in solids:
                if t0 < tend:
                    self.puffs.append([ox + dx * t0, oy + dy * t0, oz + dz * t0, 0.0, "dust"])
        self.tracers.append([ox, oy, oz, ox + dx * tend, oy + dy * tend, oz + dz * tend, 0.0, a.idx])
        return hit

    def _melee(self, a: Actor) -> None:
        a.cool = 0.62
        a.kick = 1.0
        self.sound("knife", a.x, a.y, 0.6)
        fx, fy = math.cos(a.yaw), math.sin(a.yaw)
        for b in self.enemies(a):
            dx, dy = b.x - a.x, b.y - a.y
            d = math.hypot(dx, dy)
            if d < 2.0 and (dx * fx + dy * fy) / max(d, 0.01) > 0.6 and abs(b.z - a.z) < 1.4:
                back = (math.cos(b.yaw) * fx + math.sin(b.yaw) * fy) > 0.4
                self.damage(b, 150.0 if back else 50.0, a, "knife", False, False)
                self.sound("knife_hit", b.x, b.y, 0.9)
                break

    def damage(self, b: Actor, dmg: float, src: Actor | None, how: str, head: bool = False,
               wall: bool = False) -> None:
        if not b.alive or dmg <= 0 or self.phase == "buy":
            return
        soak = min(b.armour, dmg * C.ARMOUR_ABSORB)
        b.armour -= soak
        if b.armour < 0.5:
            b.armour = 0.0
        b.hp -= dmg - soak
        b.tag = C.TAG_TIME
        if src is not None and src is not b:
            b.hurt_by[src.idx] = self.time
            src.stats["damage"] += int(dmg)
            b.hit_dirs.append([math.atan2(src.y - b.y, src.x - b.x), 1.2])
            if src.human or src is self.view_actor():
                self.fx.append(("hit", head, b.hp <= 0))
            if b.brain is not None:
                b.brain.hurt(src)
        if b.human:
            self.fx.append(("hurt", dmg))
        self.sound("hurt", b.x, b.y, 0.7)
        if b.hp <= 0:
            self._kill(b, src, how, head, wall)

    def view_actor(self) -> Actor:
        return self.player

    def _kill(self, b: Actor, src: Actor | None, how: str, head: bool, wall: bool) -> None:
        b.alive, b.hp, b.dead_t = False, 0, 0.0
        b.deaths += 1
        b.plant_t = 0.0
        b.defusing = False
        if b.has_charge:
            b.has_charge = False
            self.charge = {"state": "dropped", "x": b.x, "y": b.y}
        if b.main:
            mag = b.ammo.get(b.main, [0, 0])
            self.drops.append([b.x, b.y, b.main, mag[0], mag[1]])
        self.sound("death", b.x, b.y, 1.2)
        if src is not None and src is not b:
            src.kills += 1
            src.round_kills += 1
            src.credits = min(src.credits + C.CREDIT_KILL, C.CREDITS_CAP)
            src.ult = min(src.ult + C.ULT_KILL, src.agent.ability("X").points)
            if head:
                src.stats["heads"] += 1
            for key in list(src.buffs):
                ext = src.buffs[key][1].get("extend")
                if ext:
                    src.buffs[key][0] = min(src.buffs[key][0] + ext, 12.0)
            if src.human:
                self.fx.append(("kill", b.name, head, src.round_kills))
        for i, t in b.hurt_by.items():
            if (src is None or i != src.idx) and self.time - t < 8.0:
                self.actors[i].assists += 1
        self.feed.append({"k": src.name if src else "", "ks": src.squad if src else -1, "v": b.name, "vs": b.squad,
                          "how": how, "head": head, "wall": wall, "t": self.time})
        del self.feed[:-6]
        for ai in self.team_ai.values():
            ai.on_kill(b, src)

    # --- the charge ---------------------------------------------------------------------------------------------

    def _objective(self, a: Actor, c: Cmd, dt: float) -> None:
        ch = self.charge
        was_planting = a.plant_t > 0
        a.plant_t = 0.0 if not was_planting else a.plant_t
        planting = False
        if a.side == "atk":
            if ch["state"] == "dropped" and math.hypot(ch["x"] - a.x, ch["y"] - a.y) < 1.1:
                self.charge = {"state": "carried", "by": a}
                a.has_charge = True
                self.sound("pickup", a.x, a.y, 0.6)
                if a.squad == 0:
                    self.say("", "You have the charge" if a.human else f"{a.name} has the charge", "ally", 1.6)
            if (a.has_charge and c.use and self.phase == "live" and a.on_ground
                    and self.map.site_at(a.x, a.y) and self.map.height(a.x, a.y) == 0.0):
                planting = True
                if not was_planting:
                    self.sound("plant_start", a.x, a.y, 1.6)
                    self.noises.append((a.x, a.y, 30.0, a.squad, "plant", a))
                a.plant_t += dt
                a.vx = a.vy = 0.0
                if a.plant_t >= C.PLANT_TIME:
                    self._planted(a)
        if not planting:
            a.plant_t = 0.0
        if a.side == "def" and ch["state"] == "planted":
            near = math.hypot(ch["x"] - a.x, ch["y"] - a.y) < 1.7 and abs(a.z - ch.get("z", 0.0)) < 1.2
            who = ch.get("defuser")
            if near and c.use and a.on_ground and (who is None or who is a):
                if who is None:
                    ch["defuser"] = a
                    self.sound("defuse_start", ch["x"], ch["y"], 1.6)
                    self.noises.append((a.x, a.y, 30.0, a.squad, "defuse", a))
                a.defusing = True
                a.vx = a.vy = 0.0
                ch["defuse"] += dt
                if ch["defuse"] >= C.DEFUSE_TIME:
                    ch["state"] = "defused"
                    a.defusing = False
                    a.stats["defuses"] += 1
                    a.credits = min(a.credits + C.CREDIT_DEFUSE, C.CREDITS_CAP)
                    a.ult = min(a.ult + C.ULT_OBJECTIVE, a.agent.ability("X").points)
                    self.sound("defused", ch["x"], ch["y"], 9.0)
                    self._end_round(a.squad, "defuse")
            elif who is a:
                a.defusing = False
                ch["defuser"] = None
                ch["defuse"] = C.DEFUSE_HALF if ch["defuse"] >= C.DEFUSE_HALF else 0.0
        elif a.defusing:
            a.defusing = False
        # a better gun on the floor
        if c.use and a.plant_t <= 0 and not a.defusing and self.drops and a.human and not self.autopilot:
            self.pick_up(a)

    def pick_up(self, a: Actor, only_better: bool = False) -> bool:
        for d in self.drops:
            if math.hypot(d[0] - a.x, d[1] - a.y) < 1.5:
                if only_better and a.main and WEAPONS[a.main].price >= WEAPONS[d[2]].price:
                    continue
                old = a.main
                oldmag = a.ammo.get(old, [0, 0]) if old else None
                a.main = d[2]
                a.ammo[d[2]] = [d[3], d[4]]
                if old:
                    d[2], d[3], d[4] = old, oldmag[0], oldmag[1]
                else:
                    self.drops.remove(d)
                a.cur = "main"
                a.equip_t = a.weapon.equip
                self.sound("pickup", a.x, a.y, 0.6)
                return True
        return False

    def _planted(self, a: Actor) -> None:
        a.plant_t = 0.0
        a.has_charge = False
        a.stats["plants"] += 1
        site = self.map.site_at(a.x, a.y) or "A"
        self.charge = {"state": "planted", "x": a.x, "y": a.y, "z": a.z, "site": site, "t": C.FUSE_TIME, "defuse": 0.0,
                       "defuser": None, "beep": 0.0, "by": a}
        self.phase, self.t = "planted", C.FUSE_TIME
        for b in self.team(a.squad, alive=False):
            b.credits = min(b.credits + C.CREDIT_PLANT, C.CREDITS_CAP)
        a.ult = min(a.ult + C.ULT_OBJECTIVE, a.agent.ability("X").points)
        self.say("CHARGE PLANTED", f"SITE {site}  ·  {self.map.sites[site].name.upper()}", "accent", 2.6)
        self.sound("planted", a.x, a.y, 9.0)
        for ai in self.team_ai.values():
            ai.on_plant(site)

    def _charge(self, dt: float) -> None:
        ch = self.charge
        if ch["state"] != "planted" or self.phase != "planted":
            return
        ch["t"] -= dt
        left = ch["t"]
        rate = 1.0 if left > 20 else (2.0 if left > 10 else (4.0 if left > 5 else 7.0))
        ch["beep"] -= dt
        if ch["beep"] <= 0:
            ch["beep"] = 1.0 / rate
            self.sound("beep", ch["x"], ch["y"], 2.4)
        if left <= 0:
            ch["state"] = "boom"
            ch["boom_t"] = 0.0
            self.sound("boom", ch["x"], ch["y"], 9.0)
            for a in self.actors:
                if a.alive and math.hypot(a.x - ch["x"], a.y - ch["y"]) < C.BLAST_RADIUS:
                    a.hp = 0
                    self._kill(a, None, "charge", False, False)
            self._end_round(self.squad_of("atk"), "detonation")

    # --- abilities ----------------------------------------------------------------------------------------------

    def use_ability(self, a: Actor, slot: str, lob: tuple | None = None) -> bool:
        if not a.alive or not a.can_use(slot) or a.equip_t > 0.2:
            return False
        ab = a.agent.ability(slot)
        p = ab.p
        kind = ab.kind
        yaw, pitch = a.yaw, a.pitch
        fx, fy = math.cos(yaw), math.sin(yaw)
        if kind in ("flash", "frag", "smoke", "slow", "fire", "static", "sonar", "concuss"):
            self._throw(a, ab, lob)
        elif kind == "buff":
            a.buffs[ab.id] = [p["dur"], p]
        elif kind == "dash":
            sp = a.speed()
            dx, dy = (a.vx / sp, a.vy / sp) if sp > 1.0 else (fx, fy)
            v = p["dist"] / p["time"]
            a.dash = [p["time"], dx * v, dy * v]
        elif kind == "curtain":
            n = int(p["length"] / 1.45)
            for i in range(n):
                x, y = a.x + fx * (2.4 + i * 1.45), a.y + fy * (2.4 + i * 1.45)
                if self.map.height(x, y) < 2.0:
                    self.smokes.append([x, y, 1.5, p["r"], -i * 0.03, p["dur"], a.squad, a.agent.color])
        elif kind == "nearsight":
            for b in self.enemies(a):
                b.near_t, b.near_r = p["dur"], p["range"]
        elif kind == "sweep":
            for b in self.enemies(a):
                b.revealed = max(b.revealed, p["dur"])
        elif kind == "sensor":
            self.sensors.append({"x": a.x + fx * 0.6, "y": a.y + fy * 0.6, "z": a.z, "r": p["radius"], "t": p["dur"],
                                 "squad": a.squad, "seen": set(), "age": 0.0})
        elif kind == "barricade":
            cells = self._barricade_cells(a, p["width"])
            if len(cells) < 2:
                return False
            old = self.map.set_cells(cells, p["h"], M_BARRICADE)
            self.barricades.append([p["dur"], old, cells, 0.0])
            self.geometry_dirty = True
        elif kind == "heal":
            target = a
            for b in self.team(a.squad):
                if (b is not a and b.hp < target.hp - 5 and math.hypot(b.x - a.x, b.y - a.y) < p["range"]
                        and self.map.los(a.x, a.y, a.eye, b.x, b.y, b.z + 1.0)):
                    target = b
            if target.hp >= C.HP:
                return False
            target.heals.append([p["heal"] / p["dur"], p["dur"]])
        elif kind == "stronghold":
            self.zones.append({"kind": "stronghold", "x": a.x, "y": a.y, "z": a.z, "r": p["radius"], "t": p["dur"],
                               "dur": p["dur"], "squad": a.squad, "owner": a, "p": p, "color": a.agent.color})
        elif kind == "chain":
            x, y = a.x, a.y
            for i in range(p["count"]):
                nx, ny = a.x + fx * (4.0 + i * p["step"]), a.y + fy * (4.0 + i * p["step"])
                if self.map.height(nx, ny) >= C.WALL or not self.map.los(x, y, 1.0, nx, ny, 1.0):
                    break
                x, y = nx, ny
                self.blasts.append([p["delay"] + i * p["gap"], x, y, self.map.height(x, y) + 0.4, p["radius"],
                                    p["dmg"], a])
        if slot == "X":
            a.ult = 0
            self.sound("ult", a.x, a.y, 4.0)
            mine = a.squad == 0
            self.say(ab.name, f"{a.name}  ·  {'ALLY' if mine else 'ENEMY'} ULTIMATE", "ally" if mine else "enemy", 2.2)
        else:
            a.charges[slot] -= 1
        a.stats["abilities"] += 1
        self.used_abilities[ab.id] = self.used_abilities.get(ab.id, 0) + 1
        self.sound("ability", a.x, a.y, 0.9)
        self.noises.append((a.x, a.y, 16.0, a.squad, "ability", a))
        a.equip_t = max(a.equip_t, 0.35)
        return True

    def _throw(self, a: Actor, ab: C.Ability, lob: tuple | None) -> None:
        z0 = a.eye - 0.1
        yaw, pitch = a.view()
        if lob is not None:
            tx, ty = lob
            d = math.hypot(tx - a.x, ty - a.y)
            T = min(max(d / 13.0, 0.45), 1.5)
            if ab.kind == "flash":
                T = ab.p["fuse"] * 1.05
            tz = self.map.height(tx, ty) if self.map.height(tx, ty) < 2.0 else 0.0
            if ab.kind == "flash":
                tz += 2.0
            vx, vy = (tx - a.x) / T, (ty - a.y) / T
            vz = (tz - z0 + 0.5 * C.THROW_GRAVITY * T * T) / T
        else:
            dx, dy, dz = view_dir(yaw, pitch + 0.1)
            vx, vy, vz = dx * C.THROW_SPEED + a.vx * 0.5, dy * C.THROW_SPEED + a.vy * 0.5, dz * C.THROW_SPEED
        self.projectiles.append({"x": a.x + math.cos(yaw) * 0.4, "y": a.y + math.sin(yaw) * 0.4, "z": z0, "vx": vx,
                                 "vy": vy, "vz": vz, "ab": ab, "owner": a, "age": 0.0, "color": a.agent.color})
        self.sound("throw", a.x, a.y, 0.6)

    def _barricade_cells(self, a: Actor, width: int) -> list:
        fx, fy = math.cos(a.yaw), math.sin(a.yaw)
        cx, cy = int(a.x + fx * 2.3), int(a.y + fy * 2.3)
        along_y = abs(fx) > abs(fy)
        cells = []
        m = self.map
        for i in range(-(width // 2), width // 2 + 1):
            x, y = (cx, cy + i) if along_y else (cx + i, cy)
            if not m.inside(x, y) or m.hl[y][x] != 0.0:
                continue
            if any(b.alive and abs(b.x - x - 0.5) < 0.9 and abs(b.y - y - 0.5) < 0.9 for b in self.actors):
                continue
            ch = self.charge
            if ch["state"] in ("planted", "dropped") and int(ch["x"]) == x and int(ch["y"]) == y:
                continue
            cells.append((x, y))
        return cells

    def _pop(self, pr: dict) -> None:
        ab: C.Ability = pr["ab"]
        p, a = ab.p, pr["owner"]
        x, y, z = pr["x"], pr["y"], pr["z"]
        kind = ab.kind
        if kind == "flash":
            self.sound("flash", x, y, 3.0)
            self.puffs.append([x, y, z, 0.0, "flash"])
            for b in self.actors:
                if not b.alive:
                    continue
                d = math.hypot(b.x - x, b.y - y)
                if d > 34 or not self.map.los(x, y, z, b.x, b.y, b.eye) or self.smoked(x, y, z, b.x, b.y, b.eye):
                    continue
                vy_, vp_ = b.view()
                dx, dy, dz = view_dir(vy_, vp_)
                ex, ey, ez = x - b.x, y - b.y, z - b.eye
                n = math.sqrt(ex * ex + ey * ey + ez * ez) or 1.0
                cos = (dx * ex + dy * ey + dz * ez) / n
                if cos < -0.1:
                    k = 0.12
                else:
                    k = 0.35 + 0.65 * min(max(cos, 0.0) * 1.3, 1.0)
                k *= 1.0 - 0.5 * min(d / 34, 1.0)
                if b.squad == a.squad:
                    k *= 0.4
                b.blind = max(b.blind, p["blind"] * k)
        elif kind == "frag":
            self._blast(x, y, z + 0.3, p["radius"], p["dmg"], a)
        elif kind == "smoke":
            self.smokes.append([x, y, max(z, self.map.height(x, y)) + 1.3, p["radius"], 0.0, p["dur"], a.squad,
                                a.agent.color])
            self.sound("smoke", x, y, 1.4)
        elif kind in ("slow", "fire", "static"):
            self.zones.append({"kind": kind, "x": x, "y": y, "z": self.map.ground(x, y, z + 0.5), "r": p["radius"],
                               "t": p["dur"], "dur": p["dur"], "squad": a.squad, "owner": a, "p": p,
                               "color": a.agent.color})
            self.sound("fire" if kind == "fire" else "zone", x, y, 1.6)
        elif kind == "sonar":
            self.pings.append({"x": x, "y": y, "z": z, "r": p["radius"], "left": p["pulses"], "t": 0.35,
                               "gap": p["gap"], "reveal": p["reveal"], "squad": a.squad, "pulse": 9.0})
        elif kind == "concuss":
            self.sound("concuss", x, y, 2.6)
            self.puffs.append([x, y, z + 0.4, 0.0, "shock"])
            for b in self.actors:
                if b.alive and math.hypot(b.x - x, b.y - y) < p["radius"] and self.map.los(x, y, z + 0.5, b.x, b.y,
                                                                                         b.z + 1.0):
                    b.daze = max(b.daze, p["daze"] * (0.5 if b.squad == a.squad else 1.0))

    def _blast(self, x: float, y: float, z: float, r: float, dmg: float, owner: Actor) -> None:
        self.sound("blast", x, y, 5.0)
        self.puffs.append([x, y, z, 0.0, "blast"])
        self.noises.append((x, y, 60.0, owner.squad, "blast", owner))
        for b in list(self.actors):
            if not b.alive or (b.squad == owner.squad and b is not owner):
                continue
            d = math.sqrt((b.x - x) ** 2 + (b.y - y) ** 2 + (b.z + 0.9 - z) ** 2)
            if d < r and self.map.los(x, y, z, b.x, b.y, b.z + 0.9):
                k = (1.0 - d / r) * (0.5 if b is owner else 1.0)
                self.damage(b, dmg * max(k, 0.12), owner, "blast")

    def _effects(self, dt: float) -> None:
        m = self.map
        # things in the air
        g = C.THROW_GRAVITY
        for pr in list(self.projectiles):
            pr["age"] += dt
            ab = pr["ab"]
            fused = ab.kind in ("flash", "frag")
            pr["vz"] -= g * dt
            nx, ny, nz = pr["x"] + pr["vx"] * dt, pr["y"] + pr["vy"] * dt, pr["z"] + pr["vz"] * dt
            landed = False
            if m.height(nx, pr["y"]) > nz:
                pr["vx"] *= -0.45
                nx = pr["x"]
                self.sound("bounce", nx, ny, 0.3)
            if m.height(nx, ny) > nz:
                if m.height(nx, pr["y"]) <= nz:
                    pr["vy"] *= -0.45
                    ny = pr["y"]
                elif pr["vz"] < 0 and m.height(nx, ny) < pr["z"] + 0.3:
                    nz = m.height(nx, ny)
                    landed = True
                else:
                    pr["vx"] *= -0.45
                    pr["vy"] *= -0.45
                    nx, ny = pr["x"], pr["y"]
            if nz <= m.height(nx, ny) + 0.02 and pr["vz"] <= 0:
                nz = m.height(nx, ny) + 0.02
                landed = True
            pr["x"], pr["y"], pr["z"] = nx, ny, nz
            if landed:
                pr["vz"] = -pr["vz"] * 0.35 if abs(pr["vz"]) > 2.0 else 0.0
                pr["vx"] *= 0.55
                pr["vy"] *= 0.55
            if (fused and pr["age"] >= ab.p["fuse"]) or (not fused and (landed or pr["age"] > 2.4)):
                self.projectiles.remove(pr)
                self._pop(pr)
        for b in list(self.blasts):
            b[0] -= dt
            if b[0] <= 0:
                self.blasts.remove(b)
                self._blast(b[1], b[2], b[3], b[4], b[5], b[6])
        # smokes
        for s in list(self.smokes):
            s[4] += dt
            if s[4] > s[5]:
                self.smokes.remove(s)
        # ground zones
        for a in self.actors:
            a.slow = 1.0
        for zn in list(self.zones):
            zn["t"] -= dt
            if zn["t"] <= 0:
                self.zones.remove(zn)
                continue
            r2 = zn["r"] ** 2
            p, kind = zn["p"], zn["kind"]
            for a in self.actors:
                if not a.alive or (a.x - zn["x"]) ** 2 + (a.y - zn["y"]) ** 2 > r2 or abs(a.z - zn["z"]) > 2.2:
                    continue
                foe = a.squad != zn["squad"]
                if kind == "slow" and foe:
                    a.slow = min(a.slow, p["slow"])
                elif kind == "fire" and (foe or a is zn["owner"]):
                    self._burn(a, p["dps"] * dt, zn["owner"])
                elif kind == "static" and foe:
                    a.near_t, a.near_r = max(a.near_t, 0.3), min(p["range"], a.near_r if a.near_t > 0.3 else 99)
                elif kind == "stronghold":
                    if foe:
                        a.slow = min(a.slow, p["slow"])
                        a.revealed = max(a.revealed, 0.4)
                    elif a.armour_max > 0:
                        a.armour = min(a.armour + p["armour"] * dt, a.armour_max)
        for s in list(self.sensors):
            s["t"] -= dt
            s["age"] += dt
            if s["t"] <= 0:
                self.sensors.remove(s)
                continue
            for a in self.actors:
                if a.alive and a.squad != s["squad"] and math.hypot(a.x - s["x"], a.y - s["y"]) < s["r"]:
                    a.revealed = max(a.revealed, 0.5)
                    if a.idx not in s["seen"]:
                        s["seen"].add(a.idx)
                        self.sound("sensor", s["x"], s["y"], 2.0)
                else:
                    s["seen"].discard(a.idx)
        for pg in list(self.pings):
            pg["t"] -= dt
            pg["pulse"] += dt
            if pg["t"] <= 0:
                pg["left"] -= 1
                pg["t"] = pg["gap"]
                pg["pulse"] = 0.0
                self.sound("ping", pg["x"], pg["y"], 2.0)
                for a in self.actors:
                    if a.alive and a.squad != pg["squad"] and math.hypot(a.x - pg["x"], a.y - pg["y"]) < pg["r"]:
                        a.revealed = max(a.revealed, pg["reveal"])
            if pg["left"] <= 0 and pg["pulse"] > 1.0:
                self.pings.remove(pg)
        for b in list(self.barricades):
            b[0] -= dt
            b[3] += dt
            if b[0] <= 0:
                m.restore_cells(b[1])
                self.barricades.remove(b)
                self.geometry_dirty = True
                self.sound("barricade_down", b[2][0][0], b[2][0][1], 1.2)
        for t in self.tracers:
            t[6] += dt
        if self.tracers and self.tracers[0][6] > 0.12:
            self.tracers = [t for t in self.tracers if t[6] <= 0.12]
        for pf in self.puffs:
            pf[3] += dt
        if self.puffs and self.puffs[0][3] > 0.7:
            self.puffs = [pf for pf in self.puffs if pf[3] <= 0.7]
        if self.charge["state"] == "boom":
            self.charge["boom_t"] += dt

    def _burn(self, a: Actor, dmg: float, owner: Actor) -> None:
        if self.phase == "buy":
            return
        soak = min(a.armour, dmg * 0.5)
        a.armour -= soak
        a.hp -= dmg - soak
        if a.human and self.rng.random() < 0.1:
            self.fx.append(("hurt", dmg))
        if a.hp <= 0 and a.alive:
            self._kill(a, owner if owner is not a else None, "fire", False, False)

    # --- what a match leaves behind -----------------------------------------------------------------------------

    def result(self) -> dict:
        p = self.player
        return {"won": self.match_winner == 0, "score": list(self.score), "kills": p.kills, "deaths": p.deaths,
                "assists": p.assists, "map": self.map.key, "agent": p.agent.id, "rounds": self.round, **p.stats}
