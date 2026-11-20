"""What every hero's two abilities and ultimate do (the numbers a player reads about them are in data.py).

``cast`` is called when the key goes down; ``tick`` every frame for the abilities that last; ``ended`` when a timed
effect runs out; ``impact`` when a thrown or fired thing lands; ``gadget_tick`` / ``zone_tick`` for what was left on
the map."""

from __future__ import annotations

import math

ULT_ACTIVE = frozenset({"melt", "deadline", "rush", "monsoon"})  # no ultimate charge while your own is running
HEAL_COL = (120, 255, 180)


# --- effects on a hero -----------------------------------------------------------------------------------------------
def slow(t, factor: float, dur: float) -> None:  # noqa: ANN001 - sim.Actor
    if "melt" in t.fx or "rush" in t.fx:
        return
    if "slow" in t.fx:
        t.slow_f = min(t.slow_f, factor)
        t.fx["slow"] = max(t.fx["slow"], dur)
    else:
        t.slow_f = factor
        t.fx["slow"] = dur


def burn(m, src, t, dps: float, dur: float) -> None:  # noqa: ANN001
    t.fx["burn"] = dur
    t.burn_dps, t.burn_src = dps, src


def boost(t, mult: float, dur: float) -> None:  # noqa: ANN001
    t.boost_m = max(t.boost_m, mult) if "boost" in t.fx else mult
    t.fx["boost"] = max(t.fx.get("boost", 0.0), dur)


def pick_ally(m, a, rng: float, cone_deg: float, low_first: bool = False):  # noqa: ANN001, ANN201
    """The ally ``a`` is looking at (within the cone), or None."""
    dx, dy, dz = a.aim()
    ox, oy, oz = a.eye_pos()
    best, bs = None, -1e9
    cosc = math.cos(math.radians(cone_deg))
    for t in m.allies(a):
        if t is a or not t.alive:
            continue
        tx, ty, tz = t.chest()
        vx, vy, vz = tx - ox, ty - oy, tz - oz
        d = math.sqrt(vx * vx + vy * vy + vz * vz)
        if d > rng or d < 1e-3:
            continue
        dot = (vx * dx + vy * dy + vz * dz) / d
        if dot < cosc or not m.visible(a, t):
            continue
        s = (1.0 - t.frac) * 3.0 + dot if low_first else dot
        if s > bs:
            best, bs = t, s
    return best


def _move_dir(a) -> tuple[float, float]:  # noqa: ANN001
    """Where ``a`` is trying to go (unit vector), or where it faces."""
    cy, sy = math.cos(a.yaw), math.sin(a.yaw)
    wx, wy = cy * a.c.mf - sy * a.c.ms, sy * a.c.mf + cy * a.c.ms
    n = math.hypot(wx, wy)
    if n < 0.1:
        return cy, sy
    return wx / n, wy / n


# --- the casts -------------------------------------------------------------------------------------------------------
def _bulldoze(m, a) -> bool:  # noqa: ANN001
    a.fx["bulldoze"] = a.hero.a1.dur
    a.hitset = set()
    m.sound("engine", a.x, a.y)
    return True


def _barricade(m, a) -> bool:  # noqa: ANN001
    from .sim import Barrier

    cy, sy = math.cos(a.yaw), math.sin(a.yaw)
    ex, ey, ez = a.eye_pos()
    reach = min(m.world.raycast(ex, ey, ez - 0.4, cy, sy, 0.0, 3.4) - 0.3, 3.2)
    px, py = a.x + cy * max(reach, 0.6), a.y + sy * max(reach, 0.6)
    z = m.world.floor_at(px, py, a.z + 0.8)
    if abs(z - a.z) > 1.6:
        z = a.z
    m.barriers = [b for b in m.barriers if b.owner is not a]
    m.barriers.append(Barrier(a, px, py, z, a.yaw, 6.0, 3.2, 550.0, a.hero.a2.dur))
    m.sound("barrier", a.x, a.y)
    return True


def _pile_driver(m, a) -> bool:  # noqa: ANN001
    a.fx["windup"] = 0.45
    m.sound("windup", a.x, a.y)

    def land() -> None:
        if not a.alive:
            return
        ax, ay, az = a.chest()
        for t in m.enemies(a):
            if not t.alive:
                continue
            tx, ty, tz = t.chest()
            if math.dist((ax, ay, az), (tx, ty, tz)) < 11.0 and abs(t.z - a.z) < 4.0 and \
                    m.world.los(ax, ay, az + 0.6, tx, ty, tz + 0.4):
                m.damage(a, t, 80.0, "hit", False, "Pile Driver")
                if t.alive:
                    t.vz = 6.5
                    t.on_ground = False
                    t.fx["stun"] = max(t.fx.get("stun", 0.0), 1.0)
                    slow(t, 0.5, 3.0)
        m.add_fx("ring", (a.x, a.y, a.z + 0.1), None, (255, 170, 60), 0.6, 11.0)
        m.add_fx("boom", (a.x, a.y, a.z + 0.5), None, (255, 190, 90), 0.4, 4.0)
        m.sound("quake", a.x, a.y, 1.3)

    m.delayed.append([m.time + 0.45, land])
    return True


def _heat_sink(m, a) -> bool:  # noqa: ANN001
    a.fx["sink"] = a.hero.a1.dur
    a.sink_store = 0.0
    m.sound("absorb", a.x, a.y)
    return True


def _slag_pool(m, a) -> bool:  # noqa: ANN001
    m.launch(a, "slag", 20.0, 14.0, 0.0)
    m.sound("lob", a.x, a.y)
    return True


def _meltdown(m, a) -> bool:  # noqa: ANN001
    a.fx["melt"] = a.hero.ult.dur
    a.fx.pop("slow", None)
    a.overhp += 300.0
    a.overhp_t = a.hero.ult.dur
    m.add_fx("ring", (a.x, a.y, a.z + 0.1), None, (255, 110, 30), 0.5, 9.0)
    return True


def _vault_boots(m, a) -> bool:  # noqa: ANN001
    a.vz = 11.5
    a.on_ground = False
    wx, wy = _move_dir(a) if abs(a.c.mf) + abs(a.c.ms) > 0.1 else (0.0, 0.0)
    a.kx += wx * 3.0
    a.ky += wy * 3.0
    m.add_fx("ring", (a.x, a.y, a.z + 0.1), None, (140, 255, 220), 0.35, 2.0)
    m.sound("spring", a.x, a.y)
    return True


def _marker_flare(m, a) -> bool:  # noqa: ANN001
    m.launch(a, "flare", 46.0, 6.0, 30.0)
    m.sound("flare", a.x, a.y)
    return True


def _deadline(m, a) -> bool:  # noqa: ANN001
    a.fx["deadline"] = a.hero.ult.dur
    a.ammo = float(a.hero.weapon.mag)
    a.reload_t = 0.0
    return True


def _dash(m, a) -> bool:  # noqa: ANN001
    wx, wy = _move_dir(a)
    a.dash_t = 0.16
    a.dash_vx, a.dash_vy = wx * 44.0, wy * 44.0
    m.add_fx("streak", (a.x, a.y, a.z + 0.9), (a.x + wx * 7.0, a.y + wy * 7.0, a.z + 0.9), (90, 240, 255), 0.3, 0.5)
    m.sound("dash", a.x, a.y)
    return True


def _second_wind(m, a) -> bool:  # noqa: ANN001
    a.fx["hot"] = 1.5
    a.hot_hps, a.hot_src = 50.0, a
    for k in ("slow", "burn", "mark", "reveal"):
        a.fx.pop(k, None)
    a.fx["haste"] = a.hero.a2.dur
    a.haste_m = 1.3
    m.add_fx("ring", (a.x, a.y, a.z + 0.1), None, HEAL_COL, 0.4, 1.6)
    m.sound("wind", a.x, a.y)
    return True


def _rush_hour(m, a) -> bool:  # noqa: ANN001
    a.fx["rush"] = a.hero.ult.dur
    a.fx.pop("slow", None)
    a.ammo = float(a.hero.weapon.mag)
    a.reload_t = 0.0
    a.chg[0] = a.hero.a1.charges
    return True


def _rocket_hop(m, a) -> bool:  # noqa: ANN001
    m.explode(a, a.x, a.y, a.z + 0.3, 4.0, 50.0, "Rocket Hop", knock=9.0)
    wx, wy = _move_dir(a)
    a.vz = 9.5
    a.on_ground = False
    a.kx += wx * 8.0
    a.ky += wy * 8.0
    return True


def _prox_mine(m, a) -> bool:  # noqa: ANN001
    m.launch(a, "mine", 16.0, 16.0, 0.0)
    m.sound("lob", a.x, a.y)
    return True


def _carpet_charge(m, a) -> bool:  # noqa: ANN001
    dx, dy, dz = a.aim()
    ox, oy, oz = a.eye_pos()
    d = m.world.raycast(ox, oy, oz, dx, dy, dz, 38.0)
    px, py, pz = ox + dx * (d - 0.4), oy + dy * (d - 0.4), oz + dz * (d - 0.4)
    cy, sy = math.cos(a.yaw), math.sin(a.yaw)
    z = m.world.floor_at(px, py, pz + 0.5)
    for i in range(8):
        x, y = px + cy * i * 3.0, py + sy * i * 3.0
        fz = m.world.floor_at(x, y, z + 1.0)
        if fz > z + 2.5 or not (1 < x < m.world.w - 1 and 1 < y < m.world.h - 1):
            break
        z = fz
        delay = 0.9 + i * 0.18
        m.add_fx("warn", (x, y, z + 0.06), None, (255, 90, 50), delay, 4.5)

        def blast(x: float = x, y: float = y, z: float = z) -> None:
            m.explode(a, x, y, z + 0.4, 4.5, 120.0, "Carpet Charge", knock=4.0, col=(255, 140, 50))

        m.delayed.append([m.time + delay, blast])
    m.sound("whistle", px, py, 1.2)
    return True


def _flask(m, a) -> bool:  # noqa: ANN001
    m.launch(a, "flask", 19.0, 14.0, 0.0)
    m.sound("lob", a.x, a.y)
    return True


def _adrenal(m, a) -> bool:  # noqa: ANN001
    t = pick_ally(m, a, 22.0, 25.0) or a
    boost(t, 1.3, a.hero.a2.dur)
    t.fx["haste"] = a.hero.a2.dur
    t.haste_m = max(t.haste_m if "haste" in t.fx else 1.0, 1.2)
    m.add_fx("ring", (t.x, t.y, t.z + 0.1), None, (255, 120, 220), 0.5, 1.8)
    if t is not a:
        m.add_fx("streak", m.muzzle(a), t.chest(), (255, 120, 220), 0.3, 0.12)
    m.sound("buff", t.x, t.y)
    return True


def _monsoon(m, a) -> bool:  # noqa: ANN001
    a.fx["monsoon"] = a.hero.ult.dur
    return True


def _pylon(m, a) -> bool:  # noqa: ANN001
    from .sim import Gadget

    cy, sy = math.cos(a.yaw), math.sin(a.yaw)
    x, y = a.x + cy * 1.4, a.y + sy * 1.4
    if not m.world.free(x, y, a.z, 0.3, 1.5, a.team, m.world.room_at(a.x, a.y)):
        x, y = a.x, a.y
    for g in m.gadgets:
        if g.owner is a and g.kind == "pylon":
            g.alive = False
    m.gadgets.append(Gadget(a, "pylon", x, y, m.world.floor_at(x, y, a.z + 0.6), 150.0, a.hero.a1.dur, 0.35, 1.7))
    m.sound("deploy", a.x, a.y)
    return True


def _hardlight(m, a) -> bool:  # noqa: ANN001
    t = pick_ally(m, a, 24.0, 30.0, True) or a
    t.overhp = min(t.overhp + 125.0, 250.0)
    t.overhp_t = a.hero.a2.dur
    t.fx["dr"] = 2.0
    t.dr_m = 0.7
    m.add_fx("ring", (t.x, t.y, t.z + 0.1), None, (255, 236, 150), 0.5, 1.8)
    if t is not a:
        m.add_fx("streak", m.muzzle(a), t.chest(), (255, 236, 150), 0.3, 0.12)
    m.sound("tag", t.x, t.y)
    return True


def _lighthouse(m, a) -> bool:  # noqa: ANN001
    from .sim import Gadget

    cy, sy = math.cos(a.yaw), math.sin(a.yaw)
    x, y = a.x + cy * 1.3, a.y + sy * 1.3
    if not m.world.free(x, y, a.z, 0.4, 2.0, a.team, m.world.room_at(a.x, a.y)):
        x, y = a.x, a.y
    m.gadgets.append(Gadget(a, "lighthouse", x, y, m.world.floor_at(x, y, a.z + 0.6), 350.0, a.hero.ult.dur, 0.5,
                            3.4))
    m.add_fx("ring", (x, y, a.z + 0.1), None, (255, 236, 150), 0.7, 13.0)
    return True


CASTS = {
    "bulldoze": _bulldoze, "barricade": _barricade, "pile_driver": _pile_driver,
    "heat_sink": _heat_sink, "slag_pool": _slag_pool, "meltdown": _meltdown,
    "vault_boots": _vault_boots, "marker_flare": _marker_flare, "deadline": _deadline,
    "dash": _dash, "second_wind": _second_wind, "rush_hour": _rush_hour,
    "rocket_hop": _rocket_hop, "prox_mine": _prox_mine, "carpet_charge": _carpet_charge,
    "flask": _flask, "adrenal": _adrenal, "monsoon": _monsoon,
    "pylon": _pylon, "hardlight": _hardlight, "lighthouse": _lighthouse,
}


def cast(m, a, slot: int) -> bool:  # noqa: ANN001
    """Use ability 0 (Shift), 1 (E) or 2 (the ultimate) if it is ready."""
    h = a.hero
    ab = (h.a1, h.a2, h.ult)[slot]
    if slot == 2:
        if not a.ult_ready:
            return False
    elif a.chg[slot] <= 0:
        if ab.id == "heat_sink" and "sink" in a.fx:  # vent early
            del a.fx["sink"]
            ended(m, a, "sink")
            return True
        return False
    if "bulldoze" in a.fx and ab.id not in ("pile_driver", "barricade"):
        return False
    if not CASTS[ab.id](m, a):
        return False
    a.cast_anim = 1.0
    if slot == 2:
        a.ult = 0.0
        a.stats.ults += 1
        mine = a.team == m.player.team
        m.calls.append([f"{a.hero.name}  {ab.name.upper()}", mine, 3.0])
        del m.calls[:-3]
        m.sound("ult_ally" if mine else "ult_enemy", None, None, 1.0)
    else:
        a.chg[slot] -= 1
        if a.cd[slot] <= 0.0:
            a.cd[slot] = ab.cd
        if a.is_player:
            m.sound("cast", None, None, 0.6)
    return True


def ended(m, a, key: str) -> None:  # noqa: ANN001
    if key == "sink":
        dmg = min(40.0 + 0.8 * a.sink_store, 180.0)
        a.sink_store = 0.0
        m.explode(a, a.x, a.y, a.z + 1.0, 7.0, dmg, "Heat Sink", knock=7.0, col=(255, 130, 40))
    elif key == "boost":
        a.boost_m = 1.0
    elif key == "haste":
        a.haste_m = 1.0
    elif key == "slow":
        a.slow_f = 1.0
    elif key == "dr":
        a.dr_m = 1.0


def tick(m, a, dt: float) -> None:  # noqa: ANN001
    """The abilities that last, every frame."""
    fx = a.fx
    if "bulldoze" in fx:
        cy, sy = math.cos(a.yaw), math.sin(a.yaw)
        for t in m.enemies(a):
            if not t.alive or t.idx in a.hitset or abs(t.z - a.z) > 1.8:
                continue
            vx, vy = t.x - a.x, t.y - a.y
            d = math.hypot(vx, vy)
            if d < 1.5 + t.hit_r and (d < 0.5 or (vx * cy + vy * sy) / d > 0.2):
                a.hitset.add(t.idx)
                m.damage(a, t, 55.0, "hit", False, "Bulldoze")
                side = 1.0 if (vx * -sy + vy * cy) >= 0 else -1.0
                t.kx += (cy * 0.6 - sy * side * 0.8) * 10.0
                t.ky += (sy * 0.6 + cy * side * 0.8) * 10.0
                t.vz = max(t.vz, 4.5)
                t.on_ground = False
                m.sound("clang", t.x, t.y)
    if "melt" in fx or "monsoon" in fx:
        a.aura_t -= dt
        if a.aura_t <= 0.0:
            a.aura_t += 0.25
            if "melt" in fx:
                for t in m.enemies(a):
                    if t.alive and math.dist(a.chest(), t.chest()) < 9.0 and m.visible(a, t):
                        m.damage(a, t, 55.0 * 0.25, "dot", False, "Meltdown")
                for _ in range(3):
                    q = m.rng.uniform(0, 6.283)
                    r = m.rng.uniform(1.0, 8.5)
                    m.add_fx("puff", (a.x + math.cos(q) * r, a.y + math.sin(q) * r, a.z + m.rng.uniform(0.2, 1.6)),
                             None, (255, 120, 30), 0.5, 0.7)
            else:
                for t in m.actors:
                    if not t.alive or math.hypot(t.x - a.x, t.y - a.y) > 10.0 or abs(t.z - a.z) > 4.0:
                        continue
                    if t.team == a.team:
                        m.heal(a, t, 85.0 * 0.25)
                    else:
                        m.damage(a, t, 20.0 * 0.25, "dot", False, "Monsoon")
                        slow(t, 0.85, 0.5)
                for _ in range(4):
                    q = m.rng.uniform(0, 6.283)
                    r = m.rng.uniform(0.5, 9.5)
                    x, y = a.x + math.cos(q) * r, a.y + math.sin(q) * r
                    m.add_fx("rain", (x, y, a.z + 5.0), (x, y, a.z + 0.2), (120, 240, 210), 0.35, 0.05)


def impact(m, p, target, wall, head: bool) -> None:  # noqa: ANN001
    """A projectile hit ``target`` (a hero or a gadget), a barrier (``wall``) or the map."""
    from .sim import Actor, Gadget, Zone

    a = p.owner
    k = p.kind
    hit_actor = isinstance(target, Actor)
    if k == "rivet":
        if hit_actor:
            m.damage(a, target, p.dmg * (a.hero.weapon.head if head else 1.0), "hit", head, a.hero.weapon.name)
        elif target is not None:
            m.hurt_gadget(target, p.dmg)
            if a.is_player:
                m.hit_t, m.hit_head = 0.2, False
        elif wall is not None:
            wall.hp -= p.dmg
            wall.flash = 0.12
        m.add_fx("spark", (p.x, p.y, p.z), None, (255, 210, 120), 0.16, 0.3)
    elif k == "charge":
        w = a.hero.weapon
        if hit_actor:
            m.damage(a, target, w.dmg, "hit", False, w.name)
        elif target is not None:
            m.hurt_gadget(target, w.dmg)
        elif wall is not None:
            wall.hp -= w.dmg
            wall.flash = 0.12
        m.explode(a, p.x, p.y, p.z, w.splash, w.splash_dmg, w.name, knock=3.0, self_mult=0.4,
                  skip=target if hit_actor else None)
    elif k == "slag":
        z = m.world.floor_at(p.x, p.y, p.z + 0.4)
        m.zones.append(Zone(a, "slag", p.x, p.y, z, 4.5, a.hero.a2.dur))
        m.add_fx("boom", (p.x, p.y, z + 0.3), None, (255, 120, 30), 0.35, 3.0)
        m.sound("splash", p.x, p.y)
    elif k == "flare":
        if hit_actor:
            m.damage(a, target, p.dmg, "hit", False, "Marker Flare")
        for t in m.enemies(a):
            if t.alive and math.dist((p.x, p.y, p.z), t.chest()) < 7.0:
                t.fx["mark"] = 5.0
                t.fx["reveal"] = 5.0
        m.add_fx("ring", (p.x, p.y, m.world.floor_at(p.x, p.y, p.z + 0.4) + 0.1), None, (140, 255, 220), 0.6, 7.0)
        m.add_fx("flare", (p.x, p.y, p.z), None, (160, 255, 230), 1.6, 1.6)
        m.sound("mark", p.x, p.y)
    elif k == "mine":
        if hit_actor:
            m.explode(a, p.x, p.y, p.z, 3.5, 80.0, "Prox Mine", knock=4.0)
            return
        mines = [g for g in m.gadgets if g.owner is a and g.kind == "mine" and g.alive]
        for g in mines[:-1]:
            g.alive = False
        m.gadgets.append(Gadget(a, "mine", p.x, p.y, m.world.floor_at(p.x, p.y, p.z + 0.4), 1.0, 25.0, 0.3, 0.25))
        m.sound("deploy", p.x, p.y, 0.7)
    elif k == "flask":
        for t in m.actors:
            if not t.alive or math.dist((p.x, p.y, p.z), t.chest()) > 5.0 + t.hit_r:
                continue
            if t.team == a.team:
                m.heal(a, t, 60.0)
                t.fx["hot"] = 3.0
                t.hot_hps, t.hot_src = 25.0, a
            else:
                m.damage(a, t, 30.0, "hit", False, "Draught Flask")
                slow(t, 0.65, 2.5)
        m.add_fx("boom", (p.x, p.y, p.z), None, HEAL_COL, 0.4, 5.0)
        m.add_fx("ring", (p.x, p.y, m.world.floor_at(p.x, p.y, p.z + 0.4) + 0.1), None, HEAL_COL, 0.5, 5.0)
        m.sound("splash", p.x, p.y)


def gadget_tick(m, g, dt: float) -> None:  # noqa: ANN001
    a = g.owner
    if g.kind == "mine":
        if g.age < 0.7:
            return
        for t in m.enemies(a):
            if t.alive and not t.protected and math.dist((g.x, g.y, g.z + 0.8), t.chest()) < 2.9 and \
                    m.world.los(g.x, g.y, g.z + 0.4, *t.chest()):
                g.alive = False
                m.explode(a, g.x, g.y, g.z + 0.3, 4.2, 120.0, "Prox Mine", knock=5.0)
                return
        return
    g.tick_t -= dt
    if g.tick_t > 0.0:
        return
    g.tick_t += 0.25
    if g.kind == "pylon":
        for t in m.allies(a):
            if t.alive and math.hypot(t.x - g.x, t.y - g.y) < 8.0 and abs(t.z - g.z) < 4.0:
                m.heal(a, t, 32.0 * 0.25)
    elif g.kind == "lighthouse":
        for t in m.actors:
            if not t.alive or math.hypot(t.x - g.x, t.y - g.y) > 13.0 or abs(t.z - g.z) > 5.0:
                continue
            if t.team == a.team:
                m.heal(a, t, 45.0 * 0.25)
                boost(t, 1.25, 0.5)
            else:
                t.fx["reveal"] = max(t.fx.get("reveal", 0.0), 0.5)


def zone_tick(m, z, dt: float) -> None:  # noqa: ANN001
    if z.kind == "slag":
        for t in m.enemies(z.owner):
            if t.alive and math.hypot(t.x - z.x, t.y - z.y) < z.r and abs(t.z - z.z) < 2.0:
                m.damage(z.owner, t, 45.0 * dt, "dot", False, "Slag Pool")
                slow(t, 0.75, 0.5)
