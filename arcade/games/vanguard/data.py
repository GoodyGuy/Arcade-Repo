"""Vanguard's heroes, tuning numbers and difficulty levels (data only: no Qt, no numpy)."""

from __future__ import annotations

from dataclasses import dataclass

# --- movement and combat rules ---------------------------------------------------------------------------------------
GRAVITY = 17.5  # m/s2
JUMP = 5.8  # m/s
STEP = 0.55  # m a walker climbs without jumping
MELEE_DMG, MELEE_RANGE, MELEE_CD = 35.0, 2.6, 1.0
REGEN_DELAY, REGEN_RATE = 6.0, 20.0  # health comes back out of combat
SHIELD_DELAY, SHIELD_RATE = 3.0, 30.0
ARMOR_BLOCK = 5.0  # each hit loses this much to armour (never more than half)
OVERHEALTH_DECAY = 30.0  # per second once its timer ran out
ULT_PASSIVE = 5.0  # points per second
RESPAWN = 7.0
ASSIST_WINDOW = 6.0
PACK_SMALL, PACK_BIG = (75.0, 10.0), (250.0, 15.0)  # (health, seconds to come back)
SEGMENT = 25  # health bar segment

# control
POINT_UNLOCK = 12.0  # s after the doors open
CAPTURE_TIME = 5.0  # s for one hero to take the point (faster with more, up to three)
CONTROL_RATE = 1.15  # % per second for the owner
ROUNDS_TO_WIN = 2
# escort
CART_RADIUS = 5.5
CART_SPEED = (0.0, 0.9, 1.15, 1.35)  # by attackers pushing (three or more: the last)
CART_ROLLBACK = 0.5
ESCORT_TIME = 150.0
ESCORT_BONUS = 70.0
PREP_TIME = 8.0

TEAM_NAMES = ("TIDE", "EMBER")
ALLY_RGB = (64, 176, 255)
ENEMY_RGB = (255, 82, 70)
GOLD = (255, 196, 64)

BOT_NAMES = ("Mako", "Juniper", "Pike", "Sable", "Tansy", "Quill", "Brio", "Vesper", "Orrin", "Lark", "Dune",
             "Fable", "Cobalt", "Nettle", "Riven", "Sorrel", "Kip", "Odessa", "Thorn", "Marlo")


@dataclass(frozen=True)
class Weapon:
    name: str
    kind: str  # "hitscan" | "proj" | "cone"
    dmg: float  # per round (per second for a cone)
    rate: float  # shots (or bursts) per second
    mag: int
    reload: float
    desc: str
    spread: float = 0.0  # degrees
    head: float = 2.0
    near: float = 200.0  # damage falls from ``near`` to ``far`` metres down to ``far_mult``
    far: float = 200.0
    far_mult: float = 1.0
    burst: int = 1
    burst_gap: float = 0.06
    speed: float = 0.0  # projectile m/s
    gravity: float = 0.0
    splash: float = 0.0  # radius
    splash_dmg: float = 0.0
    rng: float = 80.0
    cone: float = 0.0  # half angle, degrees
    heal: float = 0.0  # to allies: per round (hitscan) or per second (cone)
    use: float = 0.0  # ammo per second (cone)
    tracer: tuple = (255, 230, 150)
    sound: str = "shot"
    auto: bool = True


@dataclass(frozen=True)
class Ability:
    id: str
    name: str
    cd: float
    desc: str
    charges: int = 1
    cost: int = 0  # ultimate points
    dur: float = 0.0


@dataclass(frozen=True)
class Hero:
    id: str
    name: str
    role: str
    title: str
    blurb: str
    hp: int
    armor: int
    shield: int
    speed: float
    weapon: Weapon
    a1: Ability  # Shift
    a2: Ability  # E
    ult: Ability  # Q
    color: tuple
    accent: tuple
    glow: tuple
    pref_range: float  # where its bots like to fight from
    stars: int  # how hard it is to play well (1-3)
    tips: tuple = ()
    height: float = 1.8
    scale: float = 1.0

    @property
    def total(self) -> int:
        return self.hp + self.armor + self.shield


HEROES: dict[str, Hero] = {}


def _add(h: Hero) -> None:
    HEROES[h.id] = h


_add(Hero(
    "dozer", "DOZER", "tank", "The Foreman",
    "A site foreman in a load-lifter frame. Dozer makes ground and keeps it: a wall where the team needs one, a "
    "plough through whoever stands in the way.",
    350, 250, 0, 5.0,
    Weapon("Rivet Driver", "proj", 24, 5.5, 50, 1.8, "Heavy automatic rivets. Steady damage at any range.",
           spread=0.7, head=1.5, speed=75.0, tracer=(255, 190, 70), sound="rivet"),
    Ability("bulldoze", "Bulldoze", 9.0,
            "Lower the blade and run: much faster for 2.2 s, shots from the front mostly bounce off, and enemies "
            "you hit are thrown aside.", dur=2.2),
    Ability("barricade", "Barricade", 12.0,
            "Stand up a hard-light wall ahead of you. It stops enemy fire for 7 s or until it breaks.", dur=7.0),
    Ability("pile_driver", "Pile Driver", 0.0,
            "Slam the frame into the ground: enemies within 11 m are hurt, thrown up and left staggered.",
            cost=1700),
    (244, 186, 44), (62, 64, 74), (255, 150, 40), 9.0, 1,
    ("Put the Barricade between your team and their guns, not on top of yourself.",
     "Bulldoze into a group first, then Pile Driver while they are still bunched."),
    height=2.15, scale=1.2))

_add(Hero(
    "caldera", "CALDERA", "tank", "The Smelter",
    "A furnace keeper who wears the furnace. Caldera wins the close fight: soak a volley, hand it back as a blast, "
    "and leave the floor too hot to stand on.",
    400, 150, 0, 5.2,
    Weapon("Slag Thrower", "cone", 95, 0, 100, 1.6,
           "A short stream of molten slag. Burns everyone in front of you and sets them alight.",
           rng=11.0, cone=10.0, use=20.0, tracer=(255, 130, 40), sound="flame"),
    Ability("heat_sink", "Heat Sink", 10.0,
            "For 2.5 s take half damage and bank what you soaked, then vent it as a blast around you. Press again "
            "to vent early.", dur=2.5),
    Ability("slag_pool", "Slag Pool", 11.0,
            "Lob a crucible. It leaves a burning pool for 6 s that hurts and slows enemies standing in it.", dur=6.0),
    Ability("meltdown", "Meltdown", 0.0,
            "Run the furnace past red for 7 s: enemies near you burn, you gain 300 extra health and move faster.",
            cost=1800, dur=7.0),
    (72, 70, 82), (196, 112, 58), (255, 120, 30), 6.0, 2,
    ("Start Heat Sink just before the enemy fires, not after.",
     "Slag Pool on the objective forces the other team to step off it."),
    height=2.1, scale=1.18))

_add(Hero(
    "flint", "FLINT", "damage", "The Surveyor",
    "A long-range marksman who measures twice and fires once. Flint picks a perch, marks the target and makes "
    "every round count.",
    225, 0, 0, 5.5,
    Weapon("Transit Rifle", "hitscan", 42, 3.2, 10, 1.5,
           "A precise semi-automatic rifle. Head shots do double. Hold the right button to steady the sight.",
           spread=0.25, head=2.0, near=32.0, far=55.0, far_mult=0.6, tracer=(150, 255, 230), sound="rifle"),
    Ability("vault_boots", "Vault Boots", 6.0, "Spring-heeled boots throw you high into the air: reach a roof "
            "or leave a bad spot."),
    Ability("marker_flare", "Marker Flare", 12.0,
            "Fire a flare. Enemies within 7 m of where it lands are marked for 5 s: seen through walls and hurt "
            "25% more by everyone.", dur=5.0),
    Ability("deadline", "Deadline", 0.0,
            "For 6 s the rifle fires faster, never runs dry, and every round passes through everyone in its path "
            "for extra damage.", cost=1600, dur=6.0),
    (36, 150, 150), (214, 180, 120), (140, 255, 220), 24.0, 3,
    ("High ground wins: Vault Boots reach most ledges.",
     "Mark a tank before your team shoots it."),
    height=1.86))

_add(Hero(
    "wren", "WREN", "damage", "The Courier",
    "The fastest pair of skates in the league. Wren gets in behind, empties a magazine, and is gone before anyone "
    "turns round.",
    150, 0, 50, 6.1,
    Weapon("Stitch SMG", "hitscan", 8.5, 2.6, 32, 1.2,
           "Four-round bursts. Deadly close, weak at range.",
           spread=1.5, head=2.0, near=12.0, far=24.0, far_mult=0.35, burst=4, burst_gap=0.055,
           tracer=(255, 150, 220), sound="smg"),
    Ability("dash", "Slipstream", 3.5, "Dash 7 m the way you are moving. Holds two charges.", charges=2),
    Ability("second_wind", "Second Wind", 11.0,
            "Catch your breath: heal 75, shake off burns, slows and marks, and skate faster for a moment.", dur=2.5),
    Ability("rush_hour", "Rush Hour", 0.0,
            "For 7 s you fire and move much faster, reload at once and Slipstream comes back four times as quickly. "
            "Each elimination adds time.", cost=1400, dur=7.0),
    (232, 72, 150), (244, 244, 250), (90, 240, 255), 7.0, 3,
    ("Keep one Slipstream to get out.", "Pick off supports first: the rest of their team falls apart."),
    height=1.68, scale=0.94))

_add(Hero(
    "nitra", "NITRA", "damage", "The Blaster",
    "A quarry blaster with a launcher and a head for angles. Nitra owns doorways and corners and anyone who "
    "stands too close together.",
    250, 0, 0, 5.5,
    Weapon("Charge Launcher", "proj", 70, 1.4, 6, 1.8,
           "Lobbed charges that burst on contact. Aim above far targets.",
           head=1.0, speed=30.0, gravity=9.0, splash=3.2, splash_dmg=60.0, tracer=(255, 170, 60),
           sound="launcher"),
    Ability("rocket_hop", "Rocket Hop", 7.0,
            "Set off a charge under your feet: you fly up and away, and enemies beside you are hurt and thrown."),
    Ability("prox_mine", "Prox Mine", 9.0,
            "Throw a mine. It arms, waits, and goes off when an enemy comes near. Two can be out at once."),
    Ability("carpet_charge", "Carpet Charge", 0.0,
            "Mark a line on the ground: eight heavy blasts walk along it a second later.", cost=1650),
    (96, 172, 74), (240, 150, 50), (255, 210, 60), 15.0, 2,
    ("A direct hit does more than twice the splash.", "Mines guard the flank while you watch the front."),
    height=1.74, scale=1.02))

_add(Hero(
    "tonic", "TONIC", "support", "The Chemist",
    "A field chemist with a tank of something that fixes friends and stings everyone else. Tonic lives in the "
    "middle of the fight and keeps it going.",
    225, 0, 0, 5.5,
    Weapon("Mist Sprayer", "cone", 30, 0, 100, 1.4,
           "A cone of mist: heals every ally it touches and hurts every enemy, both at once.",
           rng=12.0, cone=12.0, use=14.0, heal=65.0, tracer=(130, 255, 200), sound="mist"),
    Ability("flask", "Draught Flask", 10.0,
            "Lob a flask that bursts: allies are healed at once and for 3 s after, enemies are hurt and slowed."),
    Ability("adrenal", "Adrenal Shot", 13.0,
            "The ally you are looking at (or you) deals 30% more damage and moves faster for 5 s.", dur=5.0),
    Ability("monsoon", "Monsoon", 0.0,
            "A healing downpour follows you for 8 s: allies inside are healed fast, enemies are hurt and slowed.",
            cost=1900, dur=8.0),
    (118, 222, 190), (250, 250, 244), (170, 120, 255), 9.0, 1,
    ("Stand where the mist reaches both your tank and their front line.",
     "Adrenal Shot on a damage hero as the fight starts."),
    height=1.72, scale=0.98))

_add(Hero(
    "beacon", "BEACON", "support", "The Signaller",
    "A signaller who fights with light. Beacon's carbine mends allies and hurts enemies, and the gear Beacon "
    "plants turns a patch of ground into the team's ground.",
    175, 0, 75, 5.5,
    Weapon("Lumen Carbine", "hitscan", 15, 1.8, 24, 1.4,
           "Three-round bursts of hard light: damage to enemies, healing to allies.",
           spread=0.4, head=1.5, burst=3, burst_gap=0.07, heal=21.0, near=30.0, far=50.0, far_mult=0.6,
           tracer=(255, 236, 150), sound="carbine"),
    Ability("pylon", "Uplink Pylon", 14.0,
            "Plant a pylon that heals allies within 8 m for 14 s. Enemies can break it.", dur=14.0),
    Ability("hardlight", "Hardlight Tag", 10.0,
            "Wrap the ally you are looking at (or you) in 125 extra health and a moment of damage resistance.",
            dur=5.0),
    Ability("lighthouse", "Lighthouse", 0.0,
            "Raise a great lamp for 10 s: allies within 13 m are healed and deal 25% more damage, and enemies "
            "inside are shown through walls.", cost=1800, dur=10.0),
    (52, 76, 170), (242, 202, 92), (255, 240, 170), 15.0, 2,
    ("Shoot your own team: every round heals.", "Plant the Pylon behind cover so it lasts."),
    height=1.9, scale=1.0))

ROLES = ("tank", "damage", "support")
ROLE_LABEL = {"tank": "TANK", "damage": "DAMAGE", "support": "SUPPORT"}
ROLE_BLURB = {"tank": "Leads the way, soaks damage and makes space.",
              "damage": "Wins duels and finishes fights.",
              "support": "Keeps the team alive and stronger."}
ROLE_RGB = {"tank": (255, 190, 70), "damage": (255, 100, 90), "support": (110, 230, 170)}
TEAM_SHAPE = {"tank": 1, "damage": 2, "support": 2}
HERO_ORDER = tuple(HEROES)
BY_ROLE = {r: [h.id for h in HEROES.values() if h.role == r] for r in ROLES}

# --- bots ------------------------------------------------------------------------------------------------------------
DIFFICULTIES = ("Recruit", "Regular", "Veteran", "Elite")
# aim error (rad), reaction (s), turn rate (rad/s), chance to use an ability when it would help, damage dealt
SKILL = {
    "Recruit": (0.085, 0.60, 5.0, 0.35, 0.70),
    "Regular": (0.052, 0.40, 7.0, 0.60, 0.85),
    "Veteran": (0.032, 0.27, 9.0, 0.85, 1.0),
    "Elite": (0.018, 0.18, 12.0, 1.0, 1.0),
}

MODES = ("control", "escort")
MODE_LABEL = {"control": "CONTROL", "escort": "ESCORT"}
MODE_BLURB = {
    "control": "Take the point and hold it to 100%. Best of three rounds. A contested point goes to overtime.",
    "escort": "Attackers walk the cart along its track through two checkpoints before time runs out. "
              "Defenders stop it.",
}


def pick_team(rng, taken: str | None = None) -> list[str]:  # noqa: ANN001 - random.Random
    """Five heroes, one tank, two damage, two support, around the hero already ``taken`` (the player's)."""
    need = dict(TEAM_SHAPE)
    team = []
    if taken is not None:
        team.append(taken)
        need[HEROES[taken].role] = max(need[HEROES[taken].role] - 1, 0)
    for role in ROLES:
        pool = [h for h in BY_ROLE[role] if h not in team]
        rng.shuffle(pool)
        team += pool[:need[role]]
    return team
