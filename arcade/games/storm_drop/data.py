"""Storm Drop's numbers: the build grid, materials, grades, weapons, supplies, the storm's phases, outfits and names.
Everything here is plain data (no Qt), so the simulation and the tests can read it on their own."""

from __future__ import annotations

from dataclasses import dataclass

# --- the world -------------------------------------------------------------------------------------------------------
CELL = 4.0  # m: a build cell is CELL x LEVEL x CELL
LEVEL = 3.0
GRID = 128  # terrain cells per side
SIZE = GRID * CELL  # m: the island's square of sea
SEA = 0.0

GRAVITY = 24.0
JUMP_V = 8.0
WALK, SPRINT, CROUCH, SWIM = 5.4, 7.8, 2.7, 3.4
RADIUS, HEIGHT, EYE = 0.38, 1.8, 1.58
STEP_UP = 0.62
SAFE_FALL_V = 17.5  # m/s: landing faster than this hurts
FALL_DMG = 6.0  # health per m/s above it

SHIP_ALT = 175.0
SHIP_SPEED = 30.0
DIVE_V = 31.0
CHUTE_ALT = 42.0  # m above the ground: the canopy opens by itself
CHUTE_V = 6.0
CHUTE_GLIDE = 13.5
DIVE_GLIDE = 17.0

MAX_HP = 100
MAX_SHIELD = 100
SLOTS = 5

# --- materials and build pieces ----------------------------------------------------------------------------------------
WOOD, BRICK, METAL = 0, 1, 2
MAT_NAMES = ("Wood", "Brick", "Metal")
MAT_COLORS = ("#c98a4b", "#c96a52", "#8fa3b5")
MAT_BUILD = ((90, 150, 4.0), (99, 300, 11.5), (110, 500, 24.5))  # health when placed, when finished, seconds between
BUILD_COST = 10
MAT_CAP = 999
FLOOR, WALLX, WALLZ, RAMP, ROOF = range(5)  # WALLX runs along x (the south edge of its cell), WALLZ along z (west edge)
PIECE_NAMES = ("Wall", "Floor", "Ramp", "Roof")  # the order of the build keys
PIECE_HP_SCALE = {FLOOR: 0.93, WALLX: 1.0, WALLZ: 1.0, RAMP: 0.93, ROOF: 0.93}

# --- grades (how good a find is) ---------------------------------------------------------------------------------------
GRADES = (("Scrap", "#a3acb8"), ("Tuned", "#45d483"), ("Prime", "#3aa0ff"), ("Royal", "#c06bff"), ("Solar", "#ffb22e"))
GRADE_DAMAGE = (1.0, 1.05, 1.10, 1.15, 1.21)
GRADE_WEIGHTS = (46, 28, 15, 8, 3)  # floor finds
CHEST_GRADE_WEIGHTS = (24, 33, 24, 13, 6)

AMMO = ("light", "medium", "heavy", "shells", "rockets")
AMMO_NAMES = {"light": "Light rounds", "medium": "Medium rounds", "heavy": "Heavy rounds", "shells": "Shells",
              "rockets": "Rockets"}
AMMO_COLORS = {"light": "#ffd966", "medium": "#ff9d4d", "heavy": "#b98cff", "shells": "#ff6b6b", "rockets": "#7fd4ff"}
AMMO_PICKUP = {"light": 18, "medium": 20, "heavy": 6, "shells": 5, "rockets": 2}
AMMO_CAP = {"light": 300, "medium": 300, "heavy": 60, "shells": 60, "rockets": 12}


@dataclass(frozen=True)
class Weapon:
    id: str
    name: str
    cls: str  # close / mid / long / blast / tool
    ammo: str
    damage: float  # per pellet (or the blast)
    rate: float  # shots per second
    mag: int
    reload: float  # s
    head: float = 1.5
    pellets: int = 1
    spread: float = 0.012  # rad at rest
    bloom: float = 0.004  # rad added per shot
    bloom_max: float = 0.05
    near: float = 40.0  # full damage out to here ...
    far: float = 90.0  # ... falling to ``floor`` of it here
    floor: float = 0.6
    reach: float = 220.0
    struct: float = 30.0  # damage to build pieces and props
    auto: bool = True
    speed: float = 0.0  # > 0: a projectile (m/s)
    drop: float = 0.0  # its gravity (m/s2)
    blast: float = 0.0  # > 0: explodes with this radius
    zoom: float = 1.25  # aiming down the sights
    min_grade: int = 0
    weight: int = 10  # how often it turns up
    sound: str = "shot_mid"


WEAPONS: dict[str, Weapon] = {w.id: w for w in (
    Weapon("sledge", "Sledge", "tool", "", 20, 2.1, 0, 0.0, head=1.0, struct=50, reach=2.7, weight=0,
           sound="swing"),
    Weapon("pepper", "Pepper Pistol", "mid", "light", 24, 6.5, 16, 1.4, head=2.0, spread=0.008, bloom=0.006,
           bloom_max=0.04, near=28, far=60, struct=22, auto=False, weight=14, sound="shot_light"),
    Weapon("buzzer", "Buzzer SMG", "close", "light", 17, 12.0, 30, 2.0, head=1.5, spread=0.022, bloom=0.0035,
           bloom_max=0.06, near=16, far=42, floor=0.45, struct=17, weight=13, sound="shot_smg"),
    Weapon("thumper", "Thumper Shotgun", "close", "shells", 9.5, 0.95, 5, 4.2, head=1.5, pellets=10, spread=0.062,
           bloom=0.0, bloom_max=0.062, near=7, far=24, floor=0.2, reach=60, struct=9, auto=False, weight=13,
           sound="shot_shell"),
    Weapon("ranger", "Ranger Rifle", "mid", "medium", 30, 5.5, 30, 2.3, head=1.5, spread=0.010, bloom=0.0042,
           bloom_max=0.05, near=50, far=110, floor=0.66, struct=30, weight=15, sound="shot_mid"),
    Weapon("hunter", "Hunter Carbine", "long", "heavy", 46, 2.6, 10, 2.5, head=1.75, spread=0.004, bloom=0.012,
           bloom_max=0.04, near=90, far=180, floor=0.8, reach=300, struct=46, auto=False, zoom=1.9, weight=8,
           sound="shot_heavy"),
    Weapon("longshot", "Longshot", "long", "heavy", 105, 0.72, 3, 3.0, head=2.5, spread=0.0, bloom=0.0,
           bloom_max=0.03, near=400, far=500, floor=1.0, reach=420, struct=105, auto=False, speed=320.0, drop=7.0,
           zoom=3.4, min_grade=1, weight=6, sound="shot_sniper"),
    Weapon("boomtube", "Boom Tube", "blast", "rockets", 110, 0.7, 1, 2.6, head=1.0, spread=0.004, bloom=0.0,
           bloom_max=0.02, reach=300, struct=320, auto=False, speed=46.0, drop=0.0, blast=5.0, min_grade=2,
           weight=4, sound="shot_rocket"),
    Weapon("lobber", "Lobber", "blast", "rockets", 72, 1.0, 6, 3.2, head=1.0, spread=0.01, bloom=0.0,
           bloom_max=0.02, reach=120, struct=210, auto=False, speed=30.0, drop=16.0, blast=4.2, min_grade=2,
           weight=3, sound="shot_lob"),
)}
GUNS = tuple(w for w in WEAPONS if w != "sledge")
CLASS_NAMES = {"close": "Close range", "mid": "Mid range", "long": "Long range", "blast": "Explosive", "tool": "Tool"}


@dataclass(frozen=True)
class Supply:
    id: str
    name: str
    heals: str  # "hp" / "shield"
    amount: int
    cap: int  # cannot lift the bar above this
    time: float  # s to use
    stack: int
    weight: int
    color: str


SUPPLIES: dict[str, Supply] = {s.id: s for s in (
    Supply("patch", "Patches", "hp", 15, 75, 2.6, 15, 16, "#ff7a8a"),
    Supply("medpack", "Med Pack", "hp", 100, 100, 7.0, 3, 6, "#ff4d6d"),
    Supply("spark", "Spark Cell", "shield", 25, 50, 2.0, 6, 14, "#5ad1ff"),
    Supply("powercell", "Power Cell", "shield", 50, 100, 4.5, 3, 8, "#3a8dff"),
)}
SUPPLY_FOUND = {"patch": 5, "medpack": 1, "spark": 3, "powercell": 1}  # how many lie in one find

# --- the storm: (wait s, shrink s, radius it ends at in m, health per second outside) -------------------------------
STORM_START_R = 340.0
STORM_PHASES = (
    (50.0, 45.0, 150.0, 1.0),
    (42.0, 40.0, 96.0, 1.0),
    (38.0, 34.0, 60.0, 2.0),
    (32.0, 30.0, 36.0, 4.0),
    (26.0, 26.0, 19.0, 7.0),
    (18.0, 24.0, 8.0, 10.0),
    (8.0, 30.0, 0.0, 12.0),
)

# --- outfits: (name, how it unlocks, (kind, count), colours) --------------------------------------------------------
# colours: suit, trim, skin, pack, hat; hat styles: 0 none, 1 cap, 2 beanie, 3 helmet, 4 crown, 5 visor


@dataclass(frozen=True)
class Outfit:
    id: str
    name: str
    need: tuple[str, int]  # ("free", 0) / ("wins", n) / ("kills", n) / ("matches", n)
    suit: str
    trim: str
    skin: str
    pack: str
    hat: str
    hat_style: int


OUTFITS = (
    Outfit("rookie", "Rookie", ("free", 0), "#ff8a3d", "#2d3a55", "#f2c59b", "#3c4b6e", "#2d3a55", 1),
    Outfit("lagoon", "Lagoon", ("free", 0), "#22c1c3", "#fdf3d0", "#c68a5d", "#f0a04b", "#fdf3d0", 2),
    Outfit("clover", "Clover", ("matches", 3), "#57c84d", "#274b2a", "#f0d0b0", "#8a5a36", "#274b2a", 1),
    Outfit("ember", "Ember", ("kills", 10), "#e5383b", "#ffd166", "#8d5a3c", "#33272a", "#ffd166", 5),
    Outfit("volt", "Volt", ("wins", 1), "#ffe14d", "#30343f", "#e8b990", "#30343f", "#30343f", 3),
    Outfit("orchid", "Orchid", ("kills", 40), "#b667f1", "#ffe8f7", "#f6d3b8", "#4a2a6a", "#ffe8f7", 2),
    Outfit("midnight", "Midnight", ("wins", 3), "#2b2f4a", "#59e3ff", "#b98562", "#16182a", "#59e3ff", 5),
    Outfit("sovereign", "Sovereign", ("wins", 7), "#f7f1e3", "#ffb22e", "#e2b08a", "#b8862b", "#ffb22e", 4),
)
CANOPIES = (  # (name, need, colour, second colour)
    ("Sunrise", ("free", 0), "#ff8a3d", "#ffe14d"),
    ("Seafoam", ("free", 0), "#22c1c3", "#f4fff8"),
    ("Berry", ("kills", 5), "#e84a8a", "#5b2a86"),
    ("Checker", ("matches", 5), "#f7f1e3", "#30343f"),
    ("Aurora", ("wins", 1), "#59e3ff", "#b667f1"),
    ("Solar", ("wins", 5), "#ffb22e", "#fff3c4"),
)

BOT_FIRST = ("Bolt", "Mossy", "Jolly", "Rusty", "Zippy", "Sandy", "Lucky", "Dizzy", "Sunny", "Pixel", "Turbo", "Cozy",
             "Salty", "Fizzy", "Nimble", "Brassy", "Wobbly", "Peppy", "Dusty", "Snappy", "Mellow", "Chunky", "Breezy",
             "Sly", "Gusty", "Plucky", "Tidal", "Copper", "Maple", "Jumpy", "Quirky", "Stormy")
BOT_LAST = ("Otter", "Badger", "Gecko", "Puffin", "Mantis", "Heron", "Crab", "Lynx", "Newt", "Marmot", "Finch",
            "Yak", "Beetle", "Stoat", "Pelican", "Moth", "Walrus", "Shrew", "Ibis", "Tapir", "Wombat", "Koi", "Hare",
            "Gull", "Mole", "Orca", "Wren", "Bison", "Toad", "Lemur", "Ferret", "Kiwi")

DIFFICULTY = {  # aim error (rad), reaction (s), damage they deal, how keen they are to build
    "easy": (0.060, 0.75, 0.55, 0.35),
    "normal": (0.042, 0.50, 0.75, 0.6),
    "hard": (0.028, 0.32, 0.95, 0.85),
}


def default_options() -> dict:
    return {"bots": 24, "difficulty": "normal", "fov": 80, "auto_pickup": True, "damage_numbers": True,
            "hold_to_build": True}


def default_stats() -> dict:
    return {"matches": 0, "wins": 0, "kills": 0, "deaths": 0, "top5": 0, "best_kills": 0, "damage": 0,
            "built": 0, "harvested": 0, "chests": 0, "headshots": 0, "longest": 0.0, "seconds": 0.0,
            "best_place": 0}


def unlocked(need: tuple[str, int], stats: dict) -> bool:
    kind, n = need
    return kind == "free" or int(stats.get(kind, 0)) >= n


def need_text(need: tuple[str, int]) -> str:
    kind, n = need
    if kind == "free":
        return "Unlocked"
    word = {"wins": "win", "kills": "elimination", "matches": "match"}[kind]
    plural = "es" if word == "match" and n != 1 else ("s" if n != 1 else "")
    return f"{n} {word}{plural}"
