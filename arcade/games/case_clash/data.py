"""Case Clash's catalogue and rules, without Qt: rarities and odds, the cases and their skins, wear, patterns, prices,
the trade up contract and the player's items. Every case and skin name is original."""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import random
from dataclasses import dataclass, field

# --- rarities ------------------------------------------------------------------------------------------------------

CONSUMER, INDUSTRIAL, MILSPEC, RESTRICTED, CLASSIFIED, COVERT, SPECIAL = range(7)
RARITY_NAMES = ("Consumer Grade", "Industrial Grade", "Mil-Spec Grade", "Restricted", "Classified", "Covert",
                "Rare Special Item")
RARITY_SHORT = ("Consumer", "Industrial", "Mil-Spec", "Restricted", "Classified", "Covert", "★ Special")
RARITY_COLORS = ("#b0c3d9", "#5e98d9", "#4b69ff", "#8847ff", "#d32ce6", "#eb4b4b", "#e4ae39")
CASE_ODDS = {MILSPEC: 0.7992, RESTRICTED: 0.1598, CLASSIFIED: 0.0320, COVERT: 0.0064, SPECIAL: 0.0026}
COUNTER_CHANCE = 0.10
COUNTER_NAME = "KillTally"  # the kill counter variant (our own name)

KEY_PRICE = 249  # cents
MARKET_FEE = 0.15
START_CENTS = 2500

WEARS = (("Factory New", "FN", 0.07), ("Minimal Wear", "MW", 0.15), ("Field-Tested", "FT", 0.38),
         ("Well-Worn", "WW", 0.45), ("Battle-Scarred", "BS", 1.0))
WEAR_PRICE = (1.0, 0.74, 0.5, 0.42, 0.34)


def wear_index(f: float) -> int:
    for i, (_n, _s, top) in enumerate(WEARS[:-1]):
        if f < top:
            return i
    return len(WEARS) - 1


def wear_name(f: float) -> str:
    return WEARS[wear_index(f)][0]


# --- weapons -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Weapon:
    id: str
    name: str
    kind: str  # pistol / smg / rifle / sniper / heavy / knife / gloves
    rate: float = 0.12  # s between shots in the Aim Arena
    mag: int = 30
    auto: bool = True


WEAPONS = {w.id: w for w in (
    Weapon("ak47", "AK-47", "rifle", 0.1, 30), Weapon("m4a4", "M4A4", "rifle", 0.09, 30),
    Weapon("m4a1s", "M4A1-S", "rifle", 0.1, 20), Weapon("galil", "Galil AR", "rifle", 0.09, 35),
    Weapon("famas", "FAMAS", "rifle", 0.09, 25), Weapon("awp", "AWP", "sniper", 1.2, 5, False),
    Weapon("ssg08", "SSG 08", "sniper", 0.9, 10, False), Weapon("deagle", "Desert Eagle", "pistol", 0.3, 7, False),
    Weapon("usps", "USP-S", "pistol", 0.17, 12, False), Weapon("glock", "Glock-18", "pistol", 0.15, 20, False),
    Weapon("p250", "P250", "pistol", 0.15, 13, False), Weapon("fiveseven", "Five-SeveN", "pistol", 0.15, 20, False),
    Weapon("mp9", "MP9", "smg", 0.07, 30), Weapon("mac10", "MAC-10", "smg", 0.075, 30),
    Weapon("p90", "P90", "smg", 0.07, 50), Weapon("ump45", "UMP-45", "smg", 0.09, 25),
    Weapon("nova", "Nova", "heavy", 0.9, 8, False), Weapon("xm1014", "XM1014", "heavy", 0.35, 7, False),
    Weapon("negev", "Negev", "heavy", 0.06, 150),
    Weapon("karambit", "Karambit", "knife"), Weapon("butterfly", "Butterfly Knife", "knife"),
    Weapon("m9", "M9 Bayonet", "knife"), Weapon("bayonet", "Bayonet", "knife"), Weapon("flip", "Flip Knife", "knife"),
    Weapon("gut", "Gut Knife", "knife"), Weapon("huntsman", "Huntsman Knife", "knife"),
    Weapon("falchion", "Falchion Knife", "knife"), Weapon("talon", "Talon Knife", "knife"),
    Weapon("sport", "Sport Gloves", "gloves"), Weapon("driver", "Driver Gloves", "gloves"),
    Weapon("wraps", "Hand Wraps", "gloves"), Weapon("moto", "Moto Gloves", "gloves"),
    Weapon("specialist", "Specialist Gloves", "gloves"),
)}
KNIFE_BASE = {"karambit": 420.0, "butterfly": 520.0, "m9": 340.0, "bayonet": 230.0, "flip": 170.0, "gut": 105.0,
              "huntsman": 150.0, "falchion": 125.0, "talon": 390.0}
GLOVE_BASE = {"sport": 520.0, "driver": 240.0, "wraps": 190.0, "moto": 260.0, "specialist": 300.0}
KIND_LABEL = {"pistol": "Pistol", "smg": "SMG", "rifle": "Rifle", "sniper": "Sniper Rifle", "heavy": "Heavy",
              "knife": "Knife", "gloves": "Gloves"}

# --- skins ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Skin:
    id: str
    case: str
    weapon: str
    name: str  # "" for a plain (vanilla) knife
    rarity: int
    finish: str  # the procedural finish (art.py)
    colors: tuple[str, ...]
    fmin: float = 0.0
    fmax: float = 1.0
    mult: float = 1.0  # price multiplier

    @property
    def weapon_name(self) -> str:
        return WEAPONS[self.weapon].name

    @property
    def kind(self) -> str:
        return WEAPONS[self.weapon].kind

    @property
    def special(self) -> bool:
        return self.rarity == SPECIAL

    @property
    def title(self) -> str:
        """'AK-47 | Desert Wraith' / '★ Karambit | Afterglow' / '★ Karambit'."""
        star = "★ " if self.special else ""
        return f"{star}{self.weapon_name}" + (f" | {self.name}" if self.name else "")

    @property
    def grade(self) -> str:
        """'Covert Rifle', '★ Covert Knife', '★ Extraordinary Gloves'."""
        if self.kind == "knife":
            return "★ Covert Knife"
        if self.kind == "gloves":
            return "★ Extraordinary Gloves"
        return f"{RARITY_NAMES[self.rarity]} {KIND_LABEL[self.kind]}"


# Knife finishes shared by the cases: id -> (name, finish, colours, float min, float max, price multiplier)
KNIFE_FINISHES = {
    "vanilla": ("", "vanilla", ("#a9b0b8", "#6d747c"), 0.0, 1.0, 0.95),
    "afterglow": ("Afterglow", "fade", ("#f6d743", "#f05aa8", "#7b4bff"), 0.0, 0.08, 1.55),
    "nebula": ("Nebula", "doppler", ("#2a0f3c", "#d63fa0", "#3f7bff"), 0.0, 0.08, 1.45),
    "firestorm": ("Firestorm Marble", "marble", ("#e3271c", "#f7c21b", "#2357d9"), 0.0, 0.08, 1.35),
    "heat": ("Heat Treated", "hardened", ("#2e57c9", "#c7912f", "#7a5a9e"), 0.0, 1.0, 1.1),
    "amber": ("Amber Claw", "tiger", ("#f2a81d", "#1a1104"), 0.0, 0.08, 1.2),
    "lattice": ("Red Lattice", "web", ("#b3121b", "#14080a"), 0.06, 0.8, 1.3),
    "carnage": ("Carnage", "slaughter", ("#c0121f", "#f06a82", "#f7d4d8"), 0.01, 0.26, 1.2),
    "midnight": ("Midnight Violet", "solid", ("#3b2a5e", "#151020"), 0.06, 0.8, 0.85),
    "dune": ("Dune Stalker", "camo", ("#b99b6b", "#7d6a48", "#4e4636", "#d8c39a"), 0.06, 0.8, 0.7),
    "frostbite": ("Frostbite", "digital", ("#c9d9ea", "#5f7fa8", "#25354d", "#eef4fb"), 0.06, 0.8, 0.75),
    "damask": ("Damask Steel", "damascus", ("#8c949c", "#3f454b"), 0.02, 0.5, 0.95),
    "weave": ("Graphite Weave", "carbon", ("#2b2f35", "#121417", "#4a525c"), 0.0, 0.12, 0.85),
}
# Glove finishes: (glove, name, colours, price multiplier)
GLOVE_SKINS = (
    ("sport", "Voltage", ("#d7f21a", "#1d1f22", "#3ad1ff"), 1.4),
    ("sport", "Tropic Night", ("#1b6b5a", "#f04f8c", "#111111"), 1.1),
    ("driver", "Saddle Brown", ("#7a4a25", "#2b1b10", "#c9a46b"), 0.9),
    ("driver", "Racing Stripe", ("#1d3f8f", "#e6e6e6", "#c21d2c"), 1.0),
    ("wraps", "Desert Scarf", ("#c9b18a", "#7b6648", "#4a3a28"), 0.85),
    ("wraps", "Indigo Bandage", ("#2b3f8f", "#d9d4c4", "#151a33"), 0.95),
    ("moto", "Hazard", ("#f2c11b", "#16181b", "#e23b2e"), 1.1),
    ("moto", "Polar", ("#e9eef4", "#2c6fd6", "#20242a"), 1.0),
    ("specialist", "Night Ops", ("#23272d", "#596370", "#a6ff3a"), 1.0),
    ("specialist", "Tigerline", ("#f2961d", "#16120c", "#f6e7c8"), 1.3),
)

# Each case: id, name, price (cents), colours, the knife types and finishes (or gloves), and its gun skins:
# (weapon, name, rarity, finish, colours, float min, float max, price multiplier)
_M, _R, _C, _V = MILSPEC, RESTRICTED, CLASSIFIED, COVERT
_CASES = (
    ("sandstorm", "Sandstorm Case", 85, ("#d9a441", "#6b4a1e"),
     ("karambit", "m9", "bayonet", "flip", "gut"), ("vanilla", "heat", "lattice", "carnage", "dune", "afterglow",
                                                    "damask"),
     (("glock", "Sunbleach", _M, "camo", ("#d8c49b", "#b0905d", "#7d6440", "#efe2c4"), 0.0, 0.8, 1.0),
      ("p250", "Dust Devil", _M, "camo", ("#a88a5c", "#6e5a3c", "#c9b28a", "#3e3428"), 0.06, 0.8, 0.9),
      ("mp9", "Sandpiper", _M, "digital", ("#c2a878", "#8a7350", "#e2d3b0", "#5a4a33"), 0.0, 0.6, 0.9),
      ("nova", "Rust Belt", _M, "hydro", ("#8b4a26", "#c97a3d", "#3a2a20"), 0.1, 1.0, 0.8),
      ("galil", "Mirage Tiles", _M, "tiles", ("#2f7fa8", "#e6d6b0", "#b5552c"), 0.0, 0.7, 1.1),
      ("ssg08", "Dry Creek", _M, "hydro", ("#c9a66b", "#8d6b40", "#e8d4a8"), 0.0, 0.5, 1.0),
      ("fiveseven", "Copperhead", _R, "anodized", ("#c8642c", "#5a2410", "#f0a868"), 0.0, 0.4, 1.0),
      ("ump45", "Scorpion King", _R, "graphic", ("#d9a441", "#1a1510", "#8a2b14", "#f2e1b4"), 0.0, 0.9, 1.0),
      ("famas", "Heatwave", _R, "fade", ("#f5d23a", "#f07a1c", "#b8141c"), 0.0, 0.3, 1.1),
      ("m4a1s", "Sand Viper", _R, "snake", ("#c9a66b", "#3d2f1c", "#8a6b3c"), 0.0, 0.8, 1.2),
      ("deagle", "Sunforged", _C, "hardened", ("#2e57c9", "#d9a441", "#8a5a2e"), 0.0, 1.0, 1.0),
      ("awp", "Mirage Prophet", _C, "scifi", ("#efe7d6", "#e07b22", "#1c1c1c"), 0.18, 1.0, 1.2),
      ("p90", "Pharaoh's Gold", _C, "graphic", ("#e4ae39", "#14305a", "#0d0d0d", "#2fa3a0"), 0.0, 0.7, 0.9),
      ("ak47", "Desert Wraith", _V, "graphic", ("#e07b22", "#121212", "#f2d9a0", "#7a1d10"), 0.0, 1.0, 1.1),
      ("m4a4", "Sun Sentinel", _V, "scifi", ("#f3efe4", "#ff7a1a", "#151515"), 0.18, 1.0, 1.0))),
    ("neon", "Neon Riot Case", 140, ("#ff3fbf", "#2b0f4a"),
     ("butterfly", "talon", "huntsman", "falchion"), ("vanilla", "afterglow", "firestorm", "amber", "nebula", "weave",
                                                      "midnight"),
     (("mac10", "Glowstick", _M, "stripes", ("#1a1a24", "#39ff9c", "#ff3fbf"), 0.0, 0.7, 1.0),
      ("p250", "Static", _M, "halftone", ("#e8e8e8", "#151515", "#ff2b6a"), 0.0, 0.8, 0.9),
      ("ump45", "Synthwave", _M, "fade", ("#ff3fbf", "#7b3fff", "#21d4fd"), 0.0, 0.5, 1.1),
      ("xm1014", "Circuitry", _M, "circuit", ("#0f1d1a", "#2cf2c2", "#0b0b0b"), 0.0, 0.6, 0.9),
      ("negev", "Bubblegum", _M, "solid", ("#ff8fc8", "#ffd1ea", "#6b2b52"), 0.0, 0.7, 0.8),
      ("mp9", "Night Market", _M, "camo", ("#221a3a", "#ff3fbf", "#29d0ff", "#3d2c66"), 0.0, 0.8, 1.0),
      ("fiveseven", "Glitch", _M, "digital", ("#141420", "#ff2bd6", "#2bf0ff", "#40406a"), 0.0, 0.6, 1.0),
      ("glock", "Afterburn", _R, "graphic", ("#ff5a1f", "#1b1035", "#ffd23f", "#ff2b8a"), 0.0, 0.8, 1.2),
      ("famas", "Laser Grid", _R, "circuit", ("#13072b", "#ff3fbf", "#ff3fbf"), 0.0, 0.6, 0.9),
      ("galil", "Riot Paint", _R, "graphic", ("#29d0ff", "#151515", "#ff3fbf", "#f9f871"), 0.0, 0.9, 1.0),
      ("usps", "Night Racer", _R, "stripes", ("#101418", "#ff3b3b", "#f2f2f2"), 0.0, 0.6, 1.3),
      ("m4a1s", "Overdrive", _C, "graphic", ("#7b3fff", "#ff3fbf", "#0f0f1a", "#39ff9c"), 0.0, 0.8, 1.1),
      ("awp", "Chromatic", _C, "fade", ("#21d4fd", "#b721ff", "#ff3fbf"), 0.0, 0.08, 1.2),
      ("deagle", "Ink Storm", _C, "halftone", ("#f4f1ea", "#111111", "#e0162b"), 0.0, 0.7, 1.0),
      ("ak47", "Neon Revenant", _V, "graphic", ("#39ff9c", "#141414", "#ff3fbf", "#7b3fff"), 0.0, 1.0, 1.3),
      ("awp", "Hyperion", _V, "scifi", ("#d9f3ff", "#7b3fff", "#101018"), 0.05, 0.7, 1.1))),
    ("glacier", "Glacier Case", 60, ("#6fd3ff", "#183a5c"),
     ("karambit", "butterfly", "m9", "flip"), ("vanilla", "nebula", "frostbite", "heat", "amber", "firestorm",
                                                "weave"),
     (("glock", "Permafrost", _M, "digital", ("#d6e6f5", "#7fa6cc", "#3f5f85", "#f4f9ff"), 0.0, 0.8, 1.0),
      ("nova", "Ice Shelf", _M, "hydro", ("#e6f2fb", "#7fb6e0", "#2f5f8f"), 0.0, 0.6, 0.9),
      ("mac10", "Snowdrift", _M, "camo", ("#eef3f7", "#b9c7d4", "#7d8fa1", "#ffffff"), 0.0, 0.8, 0.9),
      ("p90", "Frostline", _M, "stripes", ("#1d2a3a", "#8fd8ff", "#eaf6ff"), 0.0, 0.7, 1.0),
      ("ssg08", "Polar Night", _M, "solid", ("#16243b", "#2d4a72", "#8fb8e6"), 0.0, 0.5, 1.0),
      ("negev", "Avalanche", _M, "camo", ("#c9d2da", "#8a96a3", "#5a6672", "#eef2f5"), 0.06, 0.8, 0.8),
      ("ump45", "Crevasse", _M, "digital", ("#2f5f8f", "#9cc8ea", "#0f2238", "#dcefff"), 0.0, 0.7, 0.9),
      ("fiveseven", "Aurora", _R, "fade", ("#38f2a6", "#21b4fd", "#7b3fff"), 0.0, 0.3, 1.2),
      ("mp9", "Cold Fusion", _R, "carbon", ("#10243c", "#06101c", "#3fa0ff"), 0.0, 0.5, 0.9),
      ("xm1014", "Icebreaker", _R, "scifi", ("#eaf6ff", "#2f8fff", "#0f1820"), 0.0, 0.7, 0.9),
      ("usps", "Frozen Tide", _R, "hydro", ("#bfe6ff", "#2f7fd9", "#0d2a4f"), 0.0, 0.5, 1.2),
      ("m4a4", "Glacial Sentinel", _C, "scifi", ("#f4f8fc", "#2f8fff", "#121820"), 0.0, 0.8, 1.2),
      ("deagle", "Northern Lights", _C, "fade", ("#39ff9c", "#21b4fd", "#b721ff"), 0.0, 0.2, 1.1),
      ("galil", "Blizzard", _C, "graphic", ("#9cd8ff", "#0f2238", "#ffffff", "#2f8fff"), 0.0, 0.9, 0.8),
      ("awp", "Frost Wyrm", _V, "dragon", ("#8fd8ff", "#16324f", "#eaf6ff", "#2f8fff"), 0.0, 0.7, 1.4),
      ("ak47", "Cryo Core", _V, "anodized", ("#2fb8ff", "#0a2a5c", "#bff0ff"), 0.0, 0.5, 1.0))),
    ("ember", "Ember Glove Case", 210, ("#ff6a1f", "#3a0d06"),
     (), (),
     (("p250", "Cinder", _M, "camo", ("#2a1a16", "#7a2414", "#c2461e", "#141010"), 0.0, 0.8, 0.9),
      ("mp9", "Firebrand", _M, "stripes", ("#1a1210", "#ff6a1f", "#ffd23f"), 0.0, 0.7, 1.0),
      ("glock", "Charcoal", _M, "carbon", ("#2a2420", "#121010", "#ff6a1f"), 0.0, 0.5, 1.0),
      ("nova", "Kindling", _M, "solid", ("#9b4a1e", "#e07b22", "#3a1a0a"), 0.0, 0.8, 0.8),
      ("famas", "Ashfall", _M, "digital", ("#6b6560", "#2e2a28", "#ff6a1f", "#a39e98"), 0.0, 0.8, 0.9),
      ("mac10", "Lava Lamp", _M, "marble", ("#ff3d1f", "#ffb01f", "#5a0d2a"), 0.0, 0.6, 1.0),
      ("ump45", "Magma Flow", _R, "hydro", ("#ff6a1f", "#b8141c", "#2a0806"), 0.0, 0.7, 1.0),
      ("galil", "Firewatch", _R, "stripes", ("#1f1a18", "#ff3b1f", "#ffb01f"), 0.0, 0.8, 1.0),
      ("p90", "Smolder", _R, "fade", ("#ffd23f", "#ff6a1f", "#3a0d06"), 0.0, 0.4, 1.0),
      ("ssg08", "Phoenix Feather", _R, "dragon", ("#ff6a1f", "#3a0d06", "#ffd23f", "#b8141c"), 0.0, 0.8, 1.1),
      ("m4a1s", "Inferno Core", _C, "scifi", ("#f2ede6", "#e0162b", "#131313"), 0.0, 0.8, 1.2),
      ("usps", "Blood Moon", _C, "web", ("#8a0d14", "#120507"), 0.0, 0.6, 1.0),
      ("fiveseven", "Hellhound", _C, "graphic", ("#ff3b1f", "#101010", "#ffd23f", "#7a1d10"), 0.0, 0.9, 0.9),
      ("ak47", "Ember Line", _V, "redline", ("#141414", "#d1121f", "#f2f2f2"), 0.1, 0.7, 1.4),
      ("deagle", "Molten", _V, "marble", ("#ff3d1f", "#ffd23f", "#1a0505"), 0.0, 0.08, 1.0))),
)


@dataclass(frozen=True)
class Case:
    id: str
    name: str
    price: int  # cents
    colors: tuple[str, str]
    skins: tuple[Skin, ...] = field(repr=False)

    def tier(self, rarity: int) -> tuple[Skin, ...]:
        return tuple(s for s in self.skins if s.rarity == rarity)

    @property
    def has_gloves(self) -> bool:
        return any(s.kind == "gloves" for s in self.skins)


def _build() -> tuple[dict[str, Case], dict[str, Skin]]:
    cases, skins = {}, {}
    for cid, name, price, colors, knives, finishes, guns in _CASES:
        mine = []
        for weapon, sname, rarity, finish, cols, fmin, fmax, mult in guns:
            sid = f"{cid}.{weapon}.{sname.lower().replace(' ', '_').replace(chr(39), '')}"
            mine.append(Skin(sid, cid, weapon, sname, rarity, finish, cols, fmin, fmax, mult))
        for knife in knives:
            for fid in finishes:
                fname, finish, cols, fmin, fmax, mult = KNIFE_FINISHES[fid]
                mine.append(Skin(f"{cid}.{knife}.{fid}", cid, knife, fname, SPECIAL, finish, cols, fmin, fmax, mult))
        if not knives:
            for glove, gname, cols, mult in GLOVE_SKINS:
                sid = f"{cid}.{glove}.{gname.lower().replace(' ', '_')}"
                mine.append(Skin(sid, cid, glove, gname, SPECIAL, "leather", cols, 0.06, 0.8, mult))
        mine.sort(key=lambda s: (s.rarity, s.weapon))
        cases[cid] = Case(cid, name, price, colors, tuple(mine))
        skins.update({s.id: s for s in mine})
    return cases, skins


CASES, SKINS = _build()
CASE_ORDER = tuple(CASES)
# the plain pistol the Aim Arena hands out when no gun is equipped (not in any case, never an item)
DEFAULT_GUN = Skin("default.usps", "", "usps", "", MILSPEC, "solid", ("#3d4249", "#25292e", "#5a6068"), 0.0, 0.0)


def skin(skin_id: str) -> Skin:
    return SKINS.get(skin_id) or DEFAULT_GUN

# --- patterns ------------------------------------------------------------------------------------------------------


def _h(*parts) -> float:  # noqa: ANN002
    """A stable pseudo-random number in [0, 1) from the parts."""
    d = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(d, "little") / 2 ** 64


BLUE_GEM_SEEDS = frozenset((48, 151, 179, 321, 387, 555, 661, 760, 868, 955))


def blue_share(seed: int) -> float:
    """How much of a heat treated finish is blue (0..1): 0.8+ is a blue gem."""
    if seed in BLUE_GEM_SEEDS:
        return 0.8 + 0.15 * _h("gem", seed)
    return 0.08 + 0.5 * _h("blue", seed) ** 1.6


DOPPLER_PHASES = ("Phase 1", "Phase 2", "Phase 3", "Phase 4")
GEMS = {"Garnet": "#d1121f", "Azurite": "#1f5bff", "Onyx": "#3a1f4f"}


def doppler_phase(seed: int) -> str:
    """Phase 1-4, or a rare gem: Garnet (red), Azurite (blue), Onyx (black)."""
    s = seed % 1000
    if s < 20:
        return "Garnet"
    if s < 35:
        return "Azurite"
    if s < 50:
        return "Onyx"
    return DOPPLER_PHASES[(s * 7) % 4]


def fade_percent(seed: int) -> float:
    return round(80.0 + 20.0 * _h("fade", seed) ** 0.7, 1)


def frost_and_flame(seed: int) -> bool:
    """A marble pattern with no yellow (red and blue only), about 1 seed in 16."""
    return _h("marble", seed) < 0.06


def pattern_note(skin: Skin, seed: int) -> str:
    """The pattern's collector note ('' for most)."""
    if skin.finish == "hardened":
        b = blue_share(seed)
        return f"Blue Gem · {b * 100:.0f}% blue" if b >= 0.8 else f"{b * 100:.0f}% blue"
    if skin.finish == "doppler":
        return doppler_phase(seed)
    if skin.finish == "fade":
        return f"{fade_percent(seed):.1f}% fade"
    if skin.finish == "marble" and skin.special and frost_and_flame(seed):
        return "Frost & Flame"
    return ""


def pattern_multiplier(skin: Skin, seed: int) -> float:
    if skin.finish == "hardened":
        b = blue_share(seed)
        return 6.0 + 20.0 * (b - 0.8) if b >= 0.8 else 1.0 + max(0.0, b - 0.4) * 2.0
    if skin.finish == "doppler":
        return {"Garnet": 6.0, "Azurite": 5.0, "Onyx": 3.0, "Phase 2": 1.25}.get(doppler_phase(seed), 1.0)
    if skin.finish == "fade":
        return 1.0 + max(0.0, fade_percent(seed) - 94.0) * 0.08
    if skin.finish == "marble" and skin.special and frost_and_flame(seed):
        return 2.2
    return 1.0


# --- items ---------------------------------------------------------------------------------------------------------


@dataclass
class Item:
    uid: int
    skin_id: str
    float: float
    pattern: int
    kills: int | None = None  # None: no kill counter
    t: int = 0  # when it was unboxed (unix seconds)

    @property
    def skin(self) -> Skin:
        return SKINS[self.skin_id]

    @property
    def rarity(self) -> int:
        return self.skin.rarity

    @property
    def counter(self) -> bool:
        return self.kills is not None

    @property
    def wear(self) -> str:
        return wear_name(self.float)

    @property
    def wear_short(self) -> str:
        return WEARS[wear_index(self.float)][1]

    @property
    def full_name(self) -> str:
        """'KillTally AK-47 | Desert Wraith (Field-Tested)' - a plain knife has no wear in its name only if vanilla."""
        s = self.skin
        base = s.title
        if self.counter:
            base = base.replace("★ ", "★ " + COUNTER_NAME + " ") if s.special else f"{COUNTER_NAME} {base}"
        return base if s.finish == "vanilla" else f"{base} ({self.wear})"

    def to_json(self) -> dict:
        s = self.skin
        d = {"u": self.uid, "c": s.case, "s": self.skin_id, "w": s.weapon, "r": s.rarity, "f": round(self.float, 6),
             "p": self.pattern, "t": self.t}
        if self.kills is not None:
            d["k"] = self.kills
        return d

    @classmethod
    def from_json(cls, d: dict) -> Item | None:
        try:
            sid = str(d["s"])
            if sid not in SKINS:
                return None
            k = d.get("k")
            return cls(int(d["u"]), sid, float(min(max(float(d.get("f", 0.2)), 0.0), 1.0)),
                       int(d.get("p", 0)) % 1000, None if k is None else max(int(k), 0), int(d.get("t", 0)))
        except (KeyError, TypeError, ValueError):
            return None


def roll_rarity(rng: random.Random, odds: dict[int, float] | None = None) -> int:
    odds = odds or CASE_ODDS
    x = rng.random() * sum(odds.values())
    for rarity, p in odds.items():
        x -= p
        if x < 0:
            return rarity
    return next(iter(odds))


def roll_item(case_id: str, rng: random.Random, uid: int = 0, now: int = 0) -> Item:
    """Unbox one item from the case: the rarity by the odds, a skin of that tier, a float, a pattern and maybe a
    kill counter (never on gloves)."""
    case = CASES[case_id]
    rarity = roll_rarity(rng)
    skin = rng.choice(case.tier(rarity) or case.tier(MILSPEC))
    return make_item(skin, rng, uid, now)


def make_item(skin: Skin, rng: random.Random, uid: int = 0, now: int = 0, counter: bool | None = None,
              fval: float | None = None) -> Item:
    f = rng.uniform(skin.fmin, skin.fmax) if fval is None else fval
    pattern = rng.randrange(1000)
    if counter is None:
        counter = skin.kind != "gloves" and rng.random() < COUNTER_CHANCE
    return Item(uid, skin.id, f, pattern, 0 if counter and skin.kind != "gloves" else None, now)


# --- prices --------------------------------------------------------------------------------------------------------

GUN_BASE = {MILSPEC: 0.14, RESTRICTED: 0.62, CLASSIFIED: 3.1, COVERT: 13.5}
CASE_RARITY = {"sandstorm": 1.0, "neon": 1.3, "glacier": 0.85, "ember": 1.15}


def today_index(day: _dt.date | None = None) -> int:
    return (day or _dt.date.today()).toordinal()


def _smooth_noise(key: str, x: float) -> float:
    """1D value noise in [-0.5, 0.5], smooth in x."""
    k = math.floor(x)
    f = x - k
    f = f * f * (3 - 2 * f)
    a, b = _h("n", key, k) - 0.5, _h("n", key, k + 1) - 0.5
    return a + (b - a) * f


def drift(key: str, day: int) -> float:
    """The day's market drift of a price: a slow trend, swings over a few days and a little daily noise."""
    return (1.0 + 0.14 * _smooth_noise(key + "t", day / 19.0) + 0.09 * _smooth_noise(key, day / 4.0)
            + 0.03 * (_h("dj", key, day) - 0.5))


def skin_base(skin: Skin) -> float:
    if skin.kind == "knife":
        return KNIFE_BASE[skin.weapon] * skin.mult
    if skin.kind == "gloves":
        return GLOVE_BASE[skin.weapon] * skin.mult
    return GUN_BASE[skin.rarity] * skin.mult * CASE_RARITY.get(skin.case, 1.0) * (0.75 + 0.6 * _h("p", skin.id))


def item_value(item: Item, day: int | None = None) -> int:
    """Market price in cents: rarity x wear x kill counter x special pattern x the day's drift."""
    return max(3, int(round(_raw_value(item, today_index() if day is None else day))))


def _raw_value(item: Item, day: int) -> float:
    s = item.skin
    v = skin_base(s)
    if s.finish != "vanilla":
        v *= WEAR_PRICE[wear_index(item.float)]
        if item.float < 0.01:
            v *= 1.25
    if item.counter:
        v *= 1.35 if s.special else 2.1
    v *= pattern_multiplier(s, item.pattern)
    return v * drift(s.id, day) * 100


def sale_proceeds(cents: int) -> int:
    """What the seller receives after the market fee."""
    return max(1, int(math.floor(cents * (1.0 - MARKET_FEE) + 1e-9)))


def case_price(case_id: str, day: int | None = None) -> int:
    return max(5, int(round(CASES[case_id].price * drift("case." + case_id, today_index() if day is None else day))))


def price_history(key_item: Item | str, days: int = 30, day: int | None = None) -> list[float]:
    """The last ``days`` daily prices (cents, unrounded) of an item, or of a case by id."""
    d0 = today_index() if day is None else day
    if isinstance(key_item, str):
        return [CASES[key_item].price * drift("case." + key_item, d0 - days + 1 + i) for i in range(days)]
    return [_raw_value(key_item, d0 - days + 1 + i) for i in range(days)]


# --- trade up ------------------------------------------------------------------------------------------------------

TRADE_RARITIES = (MILSPEC, RESTRICTED, CLASSIFIED)


def trade_problem(items: list[Item]) -> str:
    """Why these items cannot go into one contract ('' when they can, given ten)."""
    if not items:
        return ""
    if any(i.skin.special for i in items):
        return "Knives and gloves cannot be traded up"
    r = items[0].rarity
    if r not in TRADE_RARITIES:
        return "Covert items cannot be traded up"
    if any(i.rarity != r for i in items):
        return "All items must be the same rarity"
    if any(i.counter != items[0].counter for i in items):
        return f"All items must be {COUNTER_NAME} or none"
    if len(items) > 10:
        return "A contract takes ten items"
    return ""


def trade_outcomes(items: list[Item]) -> list[tuple[Skin, float]]:
    """Each possible result and its chance: every input adds one ticket per next-rarity skin of its case; a skin wins
    with the tickets of the inputs from its case over all tickets."""
    if not items or trade_problem(items):
        return []
    r = items[0].rarity + 1
    tickets: dict[str, int] = {}
    total = 0
    for it in items:
        pool = CASES[it.skin.case].tier(r)
        total += len(pool)
        for s in pool:
            tickets[s.id] = tickets.get(s.id, 0) + 1
    if not total:
        return []
    return [(SKINS[sid], n / total) for sid, n in tickets.items()]


def trade_float(items: list[Item], out: Skin) -> float:
    """The real formula: the average input float mapped into the output skin's float range."""
    avg = sum(i.float for i in items) / len(items)
    return out.fmin + (out.fmax - out.fmin) * avg


def trade_up(items: list[Item], rng: random.Random, uid: int = 0, now: int = 0) -> Item:
    if len(items) != 10 or trade_problem(items):
        raise ValueError(trade_problem(items) or "A contract takes ten items")
    outcomes = trade_outcomes(items)
    x = rng.random()
    skin = outcomes[-1][0]
    for s, p in outcomes:
        x -= p
        if x < 0:
            skin = s
            break
    return make_item(skin, rng, uid, now, counter=items[0].counter, fval=trade_float(items, skin))


# --- levels, ranks, achievements -----------------------------------------------------------------------------------

RANKS = ("Recruit", "Private", "Private First Class", "Corporal", "Sergeant", "Staff Sergeant", "Sergeant First Class",
         "Master Sergeant", "Sergeant Major", "Second Lieutenant", "First Lieutenant", "Captain", "Major",
         "Lieutenant Colonel", "Colonel", "Brigadier General", "Major General", "Lieutenant General", "General",
         "Field Marshal")


def xp_for_level(level: int) -> int:
    """XP from this level to the next."""
    return 400 + 120 * (level - 1)


def level_of(xp: int) -> tuple[int, int, int]:
    """(level, xp into it, xp needed for the next) from total XP; levels 1-40."""
    level = 1
    while level < 40 and xp >= xp_for_level(level):
        xp -= xp_for_level(level)
        level += 1
    return level, xp, xp_for_level(level)


def rank_name(level: int) -> str:
    return RANKS[min((level - 1) // 2, len(RANKS) - 1)]


ACHIEVEMENTS = (
    ("first_case", "Fresh Cut", "Open your first case"),
    ("first_covert", "Red Alert", "Unbox a Covert item"),
    ("first_knife", "Cold Steel", "Unbox a ★ Rare Special item"),
    ("blue_gem", "Deep Blue", "Own a Blue Gem heat treated knife or pistol"),
    ("gem", "Gemstone", "Unbox a Garnet, Azurite or Onyx Nebula"),
    ("counter", "Tally Ho", f"Unbox a {COUNTER_NAME} item"),
    ("cases_25", "Collector", "Open 25 cases"),
    ("cases_100", "Centurion", "Open 100 cases"),
    ("trade_up", "Contractor", "Complete a trade up contract"),
    ("one_tap", "One Tap", "Land 10 headshots in one Aim Arena round"),
    ("arena_50", "Marksman", "Get 50 kills in one Aim Arena round"),
    ("rich", "High Roller", "Hold $100 in your wallet"),
    ("level_10", "Veteran", "Reach level 10"),
    ("fn_low", "Mint Condition", "Own an item with a float under 0.01"),
    ("surf_first", "Ramp Rat", "Finish a surf map"),
    ("surf_gold", "Slipstream", "Finish a surf map under its gold time"),
    ("surf_all", "Tour of Ramps", "Finish all three surf maps"),
)
ACHIEVEMENT_IDS = tuple(a[0] for a in ACHIEVEMENTS)


def money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(cents) / 100:,.2f}"
