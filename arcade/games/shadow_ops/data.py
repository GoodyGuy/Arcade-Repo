"""SHADOW OPS II game data: the weapons (multiplayer and zombies), attachments, ten-point loadout perks / equipment /
wildcards, scorestreaks, the zombies tonics and power-ups, ranks and the loadout validation. All names are the game's
own; the ids stay as they were so old saves load."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Weapon:
    id: str
    name: str
    cls: str  # pistol / smg / ar / shotgun / lmg / sniper / launcher / special
    dmg: float  # per bullet (pellet) at close range; multiplayer health is 100
    dmg_min: float  # beyond ``far``
    near: float  # cells of full damage
    far: float  # cells where the damage reaches ``dmg_min``
    rpm: float
    mag: int
    reserve: int
    reload: float  # s
    hip: float  # degrees of spread from the hip
    ads: float  # degrees of spread aiming down the sights
    recoil: float  # degrees of kick per shot
    zoom: float = 1.3  # field of view divided by this while aiming
    speed: float = 1.0  # movement speed multiplier
    mode: str = "auto"  # auto / semi / burst
    pellets: int = 1
    head: float = 1.4
    splash: float = 0.0  # cells of explosion radius (launchers, the ray gun)
    sight: str = "iron"  # iron / reddot / scope
    model: str = "ar"  # the view model drawn
    sound: str = "ar"
    cost: int = 0  # zombies wall-buy price (0: box only)
    pap: str = ""  # its name after the Refit Press
    level: int = 1  # multiplayer unlock level
    slot: str = "primary"
    ads_time: float = 0.22


W = Weapon
WEAPONS: dict[str, Weapon] = {w.id: w for w in (
    # sidearms
    W("service45", "Service .45", "pistol", 32, 20, 4, 12, 380, 8, 80, 1.4, 3.0, 0.4, 2.2, 1.15, 1.05, "semi",
      head=1.6, model="pistol", sound="pistol", pap="Brimstone Brothers", slot="secondary", ads_time=0.15),
    W("br9", "Tri-Burst 9", "pistol", 34, 22, 5, 14, 900, 36, 144, 1.6, 3.2, 0.7, 2.0, 1.2, 1.05, "burst",
      head=1.6, model="pistol", sound="pistol", cost=900, pap="Triple Threat", slot="secondary", ads_time=0.15),
    W("judge5", "Judge-5", "pistol", 18, 6, 2, 6, 240, 5, 40, 1.8, 6.0, 4.0, 3.6, 1.15, 1.0, "semi", pellets=6,
      head=1.2, model="revolver", sound="shotgun", pap="Final Verdict", level=24, slot="secondary"),
    W("kap", "Hornet MP", "pistol", 25, 16, 3, 9, 1000, 20, 120, 1.5, 4.0, 2.0, 1.2, 1.15, 1.05, "auto",
      head=1.3, model="pistol", sound="smg", pap="Killer Bees", level=12, slot="secondary", ads_time=0.15),
    W("rook", "Rook Launcher", "launcher", 160, 160, 2, 30, 60, 1, 4, 3.6, 1.0, 1.0, 6.0, 1.3, 0.9, "semi",
      head=1.0, splash=2.0, sight="iron", model="launcher", sound="launcher", pap="Checkmate", level=6,
      slot="secondary"),
    # SMGs
    W("kestrel", "KS-9 Kestrel", "smg", 30, 18, 4, 11, 800, 30, 120, 2.2, 4.5, 0.9, 0.8, 1.25, 1.07, "auto",
      head=1.3, model="smg", sound="smg", cost=1000, pap="Kestrel Overdrive", level=1),
    W("mantis", "Mantis-X", "smg", 24, 14, 3, 9, 1200, 32, 192, 1.9, 4.0, 1.5, 1.2, 1.2, 1.08, "auto",
      head=1.3, model="smg2", sound="smg", pap="Praying Mantis", level=4),
    W("rattler", "Rattler", "smg", 27, 17, 4, 12, 880, 50, 200, 2.6, 4.5, 0.9, 0.7, 1.25, 1.06, "auto",
      head=1.3, model="smg2", sound="smg", pap="Rattlesnake Nest", level=16),
    # assault rifles
    W("oldguard", "Old Guard DMR", "ar", 45, 35, 8, 22, 430, 10, 120, 2.4, 2.8, 0.15, 1.6, 1.4, 0.95, "semi",
      head=1.5, model="dmr", sound="rifle", cost=500, pap="Iron Warden", level=8, ads_time=0.25),
    W("volley16", "Volley R3", "ar", 40, 30, 8, 20, 820, 30, 150, 2.2, 3.5, 0.3, 0.9, 1.35, 0.97, "burst",
      head=1.4, model="ar", sound="rifle", cost=1200, pap="Hailstorm", level=2, sight="iron"),
    W("saber", "Saber AR", "ar", 33, 25, 7, 18, 740, 30, 180, 3.0, 3.5, 0.25, 0.7, 1.35, 0.97, "auto",
      head=1.4, model="ar", sound="ar", pap="Sabertooth", level=1, sight="reddot"),
    W("tempest", "Tempest AR", "ar", 30, 22, 6, 16, 900, 30, 210, 3.0, 3.8, 0.3, 0.8, 1.35, 0.97, "auto",
      head=1.4, model="ar2", sound="ar", pap="Eye of the Storm", level=10, sight="iron"),
    W("arcturus", "Arcturus R", "ar", 36, 26, 7, 19, 600, 30, 180, 3.0, 3.6, 0.2, 0.6, 1.35, 0.97, "auto",
      head=1.4, model="ar2", sound="rifle", pap="North Star", level=20, sight="reddot"),
    # shotguns
    W("twinoak", "Twin Oak", "shotgun", 22, 6, 2, 7, 260, 2, 60, 1.9, 7.0, 6.0, 4.0, 1.15, 1.0, "semi", pellets=8,
      head=1.1, model="shotgun2", sound="shotgun", cost=500, pap="Oak Splitter", level=14),
    W("ridgeback", "Ridgeback Pump", "shotgun", 20, 5, 2, 7, 75, 6, 54, 3.0, 7.0, 5.0, 3.5, 1.15, 1.0, "semi",
      pellets=8, head=1.1, model="shotgun", sound="shotgun", cost=1500, pap="Ridge Ripper", level=5),
    W("breaker", "Breaker", "shotgun", 18, 5, 2, 8, 110, 13, 104, 3.2, 6.5, 4.5, 3.0, 1.15, 1.0, "semi",
      pellets=8, head=1.1, model="shotgun", sound="shotgun", pap="Earth Breaker", level=18),
    # light machine guns
    W("hammerfall", "Hammerfall LMG", "lmg", 38, 30, 8, 22, 720, 100, 400, 4.2, 4.5, 0.6, 0.9, 1.35, 0.88, "auto",
      head=1.3, model="lmg", sound="lmg", pap="Foundry Hammer", level=22, ads_time=0.35),
    W("stampede", "Stampede LMG", "lmg", 36, 30, 8, 22, 820, 75, 450, 4.6, 4.8, 0.7, 0.8, 1.35, 0.88, "auto",
      head=1.3, model="lmg", sound="lmg", pap="Thundering Herd", level=30, ads_time=0.35, sight="reddot"),
    # snipers
    W("longbow", "Longbow .50", "sniper", 110, 90, 20, 60, 50, 5, 40, 2.6, 15.0, 0.0, 4.0, 3.6, 0.92, "semi",
      head=2.0, sight="scope", model="sniper", sound="sniper", pap="Longbow Prime", level=3, ads_time=0.4),
    W("reaper", "Reaper .50", "sniper", 100, 80, 20, 60, 160, 10, 60, 3.0, 15.0, 0.1, 3.0, 3.4, 0.9, "semi",
      head=2.0, sight="scope", model="sniper", sound="sniper", pap="Grim Harvest", level=26, ads_time=0.42),
    # zombies-only wonders (from the Lucky Crate)
    W("zapper", "Arc Caster", "special", 900, 900, 4, 30, 200, 20, 160, 2.6, 2.0, 0.5, 1.4, 1.25, 1.0, "semi",
      head=1.0, splash=1.2, model="raygun", sound="raygun", pap="Arc Caster Overload", level=99, slot="secondary"),
    W("drum", "Rivet Cannon", "launcher", 300, 300, 2, 30, 300, 6, 30, 3.0, 2.0, 1.5, 3.0, 1.2, 0.95, "semi",
      head=1.0, splash=1.8, model="launcher", sound="launcher", pap="Rivet Storm", level=99),
)}

STARTER = "service45"
KNIFE_DAMAGE = 150.0  # zombies: kills on round 1 in one stab
MP_KNIFE_DAMAGE = 135.0
BOX_POOL = ("zapper", "drum", "mantis", "rattler", "saber", "tempest", "arcturus", "breaker", "hammerfall",
            "stampede", "longbow", "reaper", "judge5", "kap", "rook")
BOX_COST = 950
FIRE_SALE_COST = 10
PAP_COST = 5000
REPAP_COST = 2000
PERK_LIMIT = 4
PAP_DAMAGE = 2.0
PAP_AMMO = 1.5


def weapon(wid: str) -> Weapon:
    return WEAPONS.get(wid) or WEAPONS[STARTER]


def weapon_name(wid: str, upgraded: bool = False) -> str:
    w = weapon(wid)
    return (w.pap or w.name) if upgraded else w.name


# --- multiplayer: attachments, perks, equipment, wildcards, scorestreaks ----------------------------------------

@dataclass(frozen=True)
class Item:
    id: str
    name: str
    desc: str
    level: int = 1
    kinds: tuple[str, ...] = ()  # attachments: the weapon classes that take it
    cost: int = 0  # scorestreaks: score needed


ATTACHMENTS = {a.id: a for a in (
    Item("reflex", "Reflex Sight", "Red dot sight.", 1, ("pistol", "smg", "ar", "shotgun", "lmg")),
    Item("ext", "Extended Mags", "50% more rounds per magazine.", 2, ("pistol", "smg", "ar", "shotgun", "lmg")),
    Item("fastmag", "Fast Mag", "Reload much faster.", 4, ("smg", "ar", "lmg", "pistol")),
    Item("grip", "Fore Grip", "Less recoil.", 3, ("smg", "ar", "shotgun", "lmg")),
    Item("laser", "Laser Sight", "Tighter hip fire.", 2, ("pistol", "smg", "ar", "shotgun", "lmg")),
    Item("quick", "Quickdraw", "Aim down the sights faster.", 5, ("smg", "ar", "shotgun", "lmg", "sniper")),
    Item("long", "Long Barrel", "Full damage further away.", 6, ("pistol", "smg", "ar", "shotgun", "lmg")),
    Item("fmj", "FMJ", "More damage through targets.", 7, ("pistol", "smg", "ar", "lmg", "sniper")),
    Item("stock", "Stock", "Move faster while aiming.", 8, ("smg", "ar", "shotgun", "sniper")),
    Item("suppressor", "Suppressor", "Firing does not show on the enemy minimap.", 9,
         ("pistol", "smg", "ar", "shotgun", "lmg", "sniper")),
    Item("acog", "Scope 3x", "Magnified scope.", 10, ("ar", "lmg")),
)}

PERKS = {
    1: {p.id: p for p in (
        Item("featherstep", "Featherstep", "Move faster."),
        Item("hardbargain", "Hard Bargain", "Scorestreaks cost 15% less.", 4),
        Item("ghosteye", "Low Profile", "Enemy drones never target you.", 6),
        Item("blastvest", "Blast Vest", "Half damage from explosives.", 2),
        Item("phantom", "Phantom", "Invisible to the enemy Recon Drone.", 12),
    )},
    2: {p.id: p for p in (
        Item("steelnerve", "Steel Nerve", "Less flinch when shot."),
        Item("quickhands", "Slick Swap", "Swap weapons and throw faster.", 3),
        Item("scrounger", "Scrounger", "Take ammo from fallen enemies.", 5),
        Item("coldfront", "Cold Front", "Bots spot you later.", 9),
    )},
    3: {p.id: p for p in (
        Item("nimble", "Nimble", "Aim straight out of a sprint."),
        Item("endurance", "Endurance", "Sprint without end.", 7),
        Item("hush", "Hush", "Silent footsteps.", 11),
        Item("sharpears", "Sharp Ears", "Hear enemies clearly; they show on the minimap nearby.", 14),
    )},
}
ALL_PERKS = {pid: p for tier in PERKS.values() for pid, p in tier.items()}

LETHALS = {i.id: i for i in (
    Item("frag", "Frag", "Cookable grenade."),
    Item("sticky", "Sticky Grenade", "Sticks, then blows.", 3),
    Item("hatchet", "Hatchet", "Thrown blade: one-hit kill.", 8),
)}
TACTICALS = {i.id: i for i in (
    Item("flash", "Flashbang", "Blinds enemies."),
    Item("stun", "Stun Grenade", "Slows enemies.", 2),
)}

WILDCARDS = {i.id: i for i in (
    Item("perk1glutton", "Extra Perk 1", "A second Perk 1."),
    Item("perk2glutton", "Extra Perk 2", "A second Perk 2.", 4),
    Item("perk3glutton", "Extra Perk 3", "A second Perk 3.", 6),
    Item("primegunner", "Primary Plus", "A third primary attachment.", 2),
    Item("secgunner", "Sidearm Plus", "A second secondary attachment.", 5),
    Item("overkill", "Two Primaries", "A primary weapon as your secondary.", 10),
    Item("dangerclose", "Extra Lethal", "A second lethal.", 3),
    Item("tactician", "Extra Tactical", "A second tactical.", 7),
)}

STREAKS = {i.id: i for i in (
    Item("recon", "Recon Drone", "Enemies show on the minimap for 30 s.", 1, cost=425),
    Item("counter", "Jammer Drone", "Scrambles enemy awareness for 30 s.", 4, cost=500),
    Item("hunter", "Seeker Drone", "A seeker missile hits the nearest enemy.", 1, cost=525),
    Item("supply", "Supply Drop", "A crate with a random scorestreak or a refill.", 1, cost=550),
    Item("hellstorm", "Skyhammer Missile", "A heavy missile on the biggest group of enemies.", 8, cost=700),
    Item("sentry", "Auto Turret", "An automated turret for 45 s.", 12, cost=800),
    Item("swarm", "Wasp Pack", "Little attack drones hunt enemies for 25 s.", 20, cost=1200),
)}

PRIMARY_IDS = ("kestrel", "mantis", "rattler", "volley16", "saber", "tempest", "arcturus", "oldguard", "ridgeback",
               "twinoak", "breaker", "hammerfall", "stampede", "longbow", "reaper")
SECONDARY_IDS = ("service45", "br9", "kap", "judge5", "rook")

DEFAULT_CLASSES = [
    {"name": "ASSAULT", "primary": "saber", "primary_att": ["reflex", "grip"], "secondary": "service45",
     "secondary_att": [], "perk1": ["featherstep"], "perk2": ["steelnerve"], "perk3": ["nimble"],
     "lethal": ["frag"], "tactical": ["flash"], "wildcards": []},
    {"name": "SMG RUSH", "primary": "kestrel", "primary_att": ["laser", "ext"], "secondary": "br9",
     "secondary_att": [], "perk1": ["featherstep"], "perk2": ["quickhands"], "perk3": ["endurance"],
     "lethal": ["frag"], "tactical": ["stun"], "wildcards": []},
    {"name": "MARKSMAN", "primary": "longbow", "primary_att": ["quick"], "secondary": "service45",
     "secondary_att": [], "perk1": ["ghosteye"], "perk2": ["steelnerve"], "perk3": ["nimble"],
     "lethal": ["frag"], "tactical": ["flash"], "wildcards": []},
    {"name": "CUSTOM 4", "primary": "volley16", "primary_att": [], "secondary": "service45",
     "secondary_att": [], "perk1": ["featherstep"], "perk2": ["steelnerve"], "perk3": [],
     "lethal": ["frag"], "tactical": [], "wildcards": []},
    {"name": "CUSTOM 5", "primary": "ridgeback", "primary_att": [], "secondary": "service45",
     "secondary_att": [], "perk1": ["blastvest"], "perk2": [], "perk3": ["nimble"],
     "lethal": ["frag"], "tactical": [], "wildcards": []},
]
DEFAULT_STREAKS = ["recon", "hunter", "supply"]


def default_classes() -> list[dict]:
    return copy.deepcopy(DEFAULT_CLASSES)


def class_cost(c: dict) -> int:
    """Loadout points the class uses: every weapon, attachment, perk, piece of equipment and wildcard is one."""
    n = (1 if c.get("primary") else 0) + (1 if c.get("secondary") else 0)
    for key in ("primary_att", "secondary_att", "perk1", "perk2", "perk3", "lethal", "tactical", "wildcards"):
        n += len(c.get(key) or [])
    return n


def unlocked(level: int, prestige: int, need: int) -> bool:
    return prestige > 0 or level >= need


def validate_class(c: dict, level: int = 55, prestige: int = 0) -> list[str]:
    """The reasons a loadout is not allowed ([] when it is)."""
    errs: list[str] = []
    wild = list(c.get("wildcards") or [])
    if len(wild) > 3:
        errs.append("At most 3 wildcards")
    if len(set(wild)) != len(wild):
        errs.append("A wildcard is picked twice")
    for w in wild:
        if w not in WILDCARDS:
            errs.append(f"Unknown wildcard {w}")
        elif not unlocked(level, prestige, WILDCARDS[w].level):
            errs.append(f"{WILDCARDS[w].name} is locked")
    if class_cost(c) > 10:
        errs.append(f"Uses {class_cost(c)} of 10 points")
    prim, sec = c.get("primary"), c.get("secondary")
    for wid, slot, atts, cap in ((prim, "primary", c.get("primary_att") or [], 3 if "primegunner" in wild else 2),
                                 (sec, "secondary", c.get("secondary_att") or [], 2 if "secgunner" in wild else 1)):
        if wid is None:
            if atts:
                errs.append(f"Attachments without a {slot} weapon")
            continue
        if wid not in WEAPONS or WEAPONS[wid].level > 60:
            errs.append(f"Unknown weapon {wid}")
            continue
        w = WEAPONS[wid]
        allowed = PRIMARY_IDS if slot == "primary" else (
            SECONDARY_IDS + PRIMARY_IDS if "overkill" in wild else SECONDARY_IDS)
        if wid not in allowed:
            errs.append(f"{w.name} cannot be the {slot}")
        if not unlocked(level, prestige, w.level):
            errs.append(f"{w.name} is locked")
        if len(atts) > cap:
            errs.append(f"At most {cap} {slot} attachments")
        if len(set(atts)) != len(atts):
            errs.append("An attachment is picked twice")
        for a in atts:
            item = ATTACHMENTS.get(a)
            if item is None or w.cls not in item.kinds:
                errs.append(f"{a} does not fit the {w.name}")
            elif not unlocked(level, prestige, item.level):
                errs.append(f"{item.name} is locked")
    for tier in (1, 2, 3):
        perks = c.get(f"perk{tier}") or []
        cap = 2 if f"perk{tier}glutton" in wild else 1
        if len(perks) > cap:
            errs.append(f"At most {cap} Perk {tier}")
        if len(set(perks)) != len(perks):
            errs.append("A perk is picked twice")
        for p in perks:
            if p not in PERKS[tier]:
                errs.append(f"{p} is not a Perk {tier}")
            elif not unlocked(level, prestige, PERKS[tier][p].level):
                errs.append(f"{PERKS[tier][p].name} is locked")
    for key, table, wc in (("lethal", LETHALS, "dangerclose"), ("tactical", TACTICALS, "tactician")):
        items = c.get(key) or []
        cap = 2 if wc in wild else 1
        if len(items) > cap:
            errs.append(f"At most {cap} {key}")
        for i in items:
            if i not in table:
                errs.append(f"Unknown {key} {i}")
            elif not unlocked(level, prestige, table[i].level):
                errs.append(f"{table[i].name} is locked")
    return errs


def validate_streaks(ids: list[str], level: int = 55, prestige: int = 0) -> list[str]:
    errs = []
    if len(ids) != 3 or len(set(ids)) != 3:
        errs.append("Pick 3 different scorestreaks")
    for s in ids:
        if s not in STREAKS:
            errs.append(f"Unknown scorestreak {s}")
        elif not unlocked(level, prestige, STREAKS[s].level):
            errs.append(f"{STREAKS[s].name} is locked")
    return errs


# --- ranks ---------------------------------------------------------------------------------------------------------

MAX_LEVEL = 55
MAX_PRESTIGE = 10
RANK_NAMES = ("Private", "Private First Class", "Specialist", "Corporal", "Sergeant", "Staff Sergeant",
              "Sergeant First Class", "Master Sergeant", "First Sergeant", "Sergeant Major", "Command Sgt. Major",
              "2nd Lieutenant", "1st Lieutenant", "Captain", "Major", "Lt. Colonel", "Colonel", "Brigadier General",
              "Major General", "Lt. General", "General", "Commander")


def _xp_table() -> list[int]:
    out, total = [0], 0
    for lv in range(1, MAX_LEVEL):
        total += 800 + lv * 260 + (lv // 10) * 900
        out.append(total)
    return out


XP_TABLE = _xp_table()  # XP_TABLE[n]: total XP needed for level n + 1


def level_for_xp(xp: int) -> int:
    lv = 1
    for i, need in enumerate(XP_TABLE):
        if xp >= need:
            lv = i + 1
    return min(lv, MAX_LEVEL)


def level_progress(xp: int) -> tuple[int, int, int]:
    """(level, XP into the level, XP the level needs)."""
    lv = level_for_xp(xp)
    if lv >= MAX_LEVEL:
        return lv, 1, 1
    return lv, xp - XP_TABLE[lv - 1], XP_TABLE[lv] - XP_TABLE[lv - 1]


def rank_name(level: int) -> str:
    return RANK_NAMES[min((max(level, 1) - 1) * len(RANK_NAMES) // MAX_LEVEL, len(RANK_NAMES) - 1)]


def unlocks_at(level: int) -> list[str]:
    """What reaching ``level`` unlocks (names)."""
    out = [w.name for w in WEAPONS.values() if w.level == level and w.level <= 60]
    for table in (ATTACHMENTS, ALL_PERKS, LETHALS, TACTICALS, WILDCARDS, STREAKS):
        out += [i.name for i in table.values() if i.level == level and level > 1]
    return out


XP = {"kill": 100, "headshot": 50, "assist": 50, "double": 50, "triple": 75, "multi": 100, "revenge": 50,
      "first_blood": 100, "longshot": 25, "win": 1500, "loss": 500, "streak_kill": 50}


# --- zombies -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Perk:
    id: str
    name: str
    cost: int
    color: tuple[int, int, int]
    desc: str
    power: bool = True


# the tonic machines of the works (the ids are the save format)
ZPERKS = {p.id: p for p in (
    Perk("jugg", "Bulwark Brew", 2500, (206, 44, 40), "Health 100 to 250: five hits to go down instead of two."),
    Perk("swift", "Gearshift Soda", 3000, (64, 196, 96), "Reload in half the time."),
    Perk("double", "Twin Turbine", 2000, (236, 150, 36), "Fire a third faster, and every bullet hits twice as hard."),
    Perk("rally", "Second Wind", 500, (70, 150, 240), "Get back up by yourself when you go down (3 times a game).",
         power=False),
    Perk("stamina", "Long Haul", 2000, (232, 210, 60), "Move faster and sprint twice as long."),
    Perk("deadeye", "Steady Hand", 1500, (150, 90, 220), "Tighter hip fire, less kick, headshots hit harder."),
)}

POWERUPS = ("max_ammo", "insta_kill", "double_points", "nuke", "carpenter", "fire_sale", "bonus_points")
POWERUP_NAMES = {"max_ammo": "RESUPPLY", "insta_kill": "ONE SHOT", "double_points": "DOUBLE PAY",
                 "nuke": "KABOOM!", "carpenter": "REPAIR CREW", "fire_sale": "CLEARANCE", "bonus_points": "BONUS"}
POWERUP_COLORS = {"max_ammo": (120, 255, 120), "insta_kill": (255, 255, 255), "double_points": (255, 220, 80),
                  "nuke": (255, 120, 40), "carpenter": (180, 200, 255), "fire_sale": (255, 120, 200),
                  "bonus_points": (255, 230, 120)}
POWERUP_TIME = 30.0  # s a timed power-up lasts
POWERUP_LIFE = 26.0  # s a drop waits on the floor (it blinks for the last 5)


def zombie_health(rnd: int) -> float:
    """150 at round 1, +100 a round to round 9, then x1.1 a round."""
    rnd = max(int(rnd), 1)
    if rnd < 10:
        return 150.0 + 100.0 * (rnd - 1)
    return 950.0 * (1.1 ** min(rnd - 9, 400))


def zombie_count(rnd: int) -> int:
    rnd = max(int(rnd), 1)
    if rnd <= 5:
        return (6, 8, 13, 18, 24)[rnd - 1]
    k = rnd - 5
    return int(24 + 1.6 * k + 0.06 * k * k)


def zombie_speeds(rnd: int) -> tuple[float, float, float]:
    """Chances (walker, runner, sprinter) of a new zombie of the round."""
    run = min(max((rnd - 3) * 0.18, 0.0), 0.85)
    sprint = min(max((rnd - 7) * 0.12, 0.0), 0.7)
    walk = max(1.0 - run - sprint, 0.0)
    total = walk + run + sprint
    return walk / total, run / total, sprint / total


def spawn_delay(rnd: int) -> float:
    return max(2.0 * (0.95 ** (max(rnd, 1) - 1)), 0.3)


@dataclass
class Profile:
    """The player's multiplayer and zombies record."""
    xp: int = 0
    prestige: int = 0
    classes: list = field(default_factory=default_classes)
    streaks: list = field(default_factory=lambda: list(DEFAULT_STREAKS))
    stats: dict = field(default_factory=dict)

    @property
    def level(self) -> int:
        return level_for_xp(self.xp)
