"""Rail Rush progress: the crew of characters, power-up upgrades, shop prices, mission sets and the saved profile
(plain JSON data, tolerant of missing or old fields)."""

from __future__ import annotations

import random

MAX_MULTIPLIER = 30
UPGRADE_LEVELS = 6
UPGRADE_COSTS = (500, 1000, 2000, 4000, 8000, 16000)
HOVERBOARD_PRICE = 300
KEY_PRICE = 1500
BOX_PRICE = 500
HOVERBOARD_TIME = 30.0

# kind -> (name, base seconds, seconds per level, blurb). The ids stay as the first version saved them.
POWERUPS = {
    "jetpack": ("Balloon Lift", 7.0, 1.5, "Float high over the wagons along a trail of coins in the sky."),
    "sneakers": ("Spring Soles", 10.0, 3.0, "Bounce high enough to land on top of the wagons."),
    "magnet": ("Coin Magnet", 10.0, 3.0, "Pulls in every coin from all three tracks."),
    "double": ("Score Gem", 10.0, 3.0, "Doubles the score multiplier while it lasts."),
}

# The courier crew (original characters). Colours: skin, top, accent, pants, shoes, bag, hair / hat.
CHARACTERS = (
    {"id": "jax", "name": "Jax", "price": 0, "style": "cap", "build": 1.0,
     "blurb": "Fastest courier on the line. Cap backwards, parcel always on time.",
     "skin": "#e9b48a", "top": "#2f7de1", "accent": "#ffd23f", "pants": "#3a3f5c", "shoes": "#ef4136",
     "pack": "#f7a823", "hair": "#e8323c"},
    {"id": "roxy", "name": "Roxy", "price": 2500, "style": "ponytail", "build": 0.94,
     "blurb": "Night-shift runner. Knows every shortcut through the yards.",
     "skin": "#f3c9a5", "top": "#8e44d9", "accent": "#3fe0c5", "pants": "#232a3d", "shoes": "#ffffff",
     "pack": "#1fb5a0", "hair": "#ff5fae"},
    {"id": "tank", "name": "Tank", "price": 6000, "style": "beanie", "build": 1.16,
     "blurb": "Carries the heavy boxes. Big, quiet, unstoppable.",
     "skin": "#a8714a", "top": "#2fae5c", "accent": "#ffffff", "pants": "#2b2b2b", "shoes": "#f2f2f2",
     "pack": "#555b66", "hair": "#ff8a1f"},
    {"id": "nova", "name": "Nova", "price": 12000, "style": "puffs", "build": 0.97,
     "blurb": "Beats in the headphones, route map in the head.",
     "skin": "#7a4a2e", "top": "#ff7a1a", "accent": "#fff3e0", "pants": "#1e5aa8", "shoes": "#ffd23f",
     "pack": "#e8323c", "hair": "#1c120c"},
    {"id": "ziggy", "name": "Ziggy", "price": 25000, "style": "mohawk", "build": 1.0,
     "blurb": "Green mohawk, express deliveries only.",
     "skin": "#f0c09a", "top": "#25262b", "accent": "#c9cdd6", "pants": "#7a1f2b", "shoes": "#25262b",
     "pack": "#c9cdd6", "hair": "#59e84a"},
)
CHARACTER_IDS = tuple(c["id"] for c in CHARACTERS)


def character(char_id: str) -> dict:
    for c in CHARACTERS:
        if c["id"] == char_id:
            return c
    return CHARACTERS[0]


def powerup_duration(kind: str, level: int) -> float:
    _name, base, step, _blurb = POWERUPS[kind]
    return base + step * max(0, min(int(level), UPGRADE_LEVELS))


# --- missions ------------------------------------------------------------------------------------------------------

# kind -> (text with {n}, base target, one run only)
MISSION_KINDS = {
    "coins_run": ("Collect {n} coins in one run", 120, True),
    "coins": ("Collect {n} coins", 500, False),
    "jumps": ("Jump {n} times", 25, False),
    "rolls": ("Roll {n} times", 15, False),
    "score_run": ("Score {n} points in one run", 2500, True),
    "hoverboard": ("Use {n} bubble shields", 1, False),
    "powerups": ("Pick up {n} power-ups", 3, False),
    "jetpack": ("Ride the balloons {n} times", 1, False),
    "lanes_run": ("Change lanes {n} times in one run", 50, True),
    "distance_run": ("Run {n} m in one run", 900, True),
    "roofs": ("Land on {n} wagon roofs", 6, False),
    "stumbles": ("Bump into the side of {n} wagons", 3, False),
    "magnet_coins": ("Collect {n} coins with the magnet", 80, False),
    "barriers": ("Jump over {n} barriers", 12, False),
}


def _nice(n: float) -> int:
    if n < 10:
        return max(1, round(n))
    if n < 100:
        return int(round(n / 5.0) * 5)
    if n < 1000:
        return int(round(n / 10.0) * 10)
    return int(round(n / 100.0) * 100)


def mission_set(index: int) -> list[dict]:
    """The three missions of set ``index`` (0 = the first set), harder as the sets go on."""
    rng = random.Random(9173 + index * 7919)
    kinds = list(MISSION_KINDS)
    if index == 0:
        picks = ["coins_run", "jumps", "rolls"]
    else:
        picks = rng.sample(kinds, 3)
    out = []
    for kind in picks:
        _text, base, _per_run = MISSION_KINDS[kind]
        scale = 1.0 + 0.32 * index
        if kind in ("hoverboard", "jetpack"):
            scale = 1.0 + 0.5 * index
        out.append({"kind": kind, "target": _nice(base * scale), "progress": 0})
    return out


def mission_text(m: dict) -> str:
    text = MISSION_KINDS.get(m.get("kind", ""), ("{n} things", 1, False))[0]
    return text.format(n=f"{int(m.get('target', 1)):,}")


def mission_per_run(m: dict) -> bool:
    return MISSION_KINDS.get(m.get("kind", ""), ("", 1, False))[2]


def mission_done(m: dict) -> bool:
    return int(m.get("progress", 0)) >= int(m.get("target", 1))


# --- the saved profile ---------------------------------------------------------------------------------------------

STAT_KEYS = ("runs", "distance", "coins", "jumps", "rolls", "crashes", "stumbles", "hoverboards", "powerups",
             "jetpacks", "boxes", "keys_found", "best_distance", "best_coins", "missions", "play_seconds")


def _int(v, default: int = 0, lo: int = 0, hi: int = 10 ** 12) -> int:  # noqa: ANN001
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


class Profile:
    """Everything Rail Rush keeps between runs."""

    def __init__(self) -> None:
        self.best = 0
        self.coins = 0
        self.keys = 1
        self.hoverboards = 3
        self.upgrades = {k: 0 for k in POWERUPS}
        self.unlocked = ["jax"]
        self.selected = "jax"
        self.mission_set = 0
        self.missions = mission_set(0)
        self.stats = {k: 0 for k in STAT_KEYS}

    @property
    def multiplier(self) -> int:
        return min(MAX_MULTIPLIER, 1 + self.mission_set)

    # --- missions ---

    def advance(self, kind: str, amount: float = 1.0, value: float | None = None) -> list[dict]:
        """Count ``amount`` toward missions of ``kind`` (or set the run value ``value``); returns the missions just
        completed."""
        done = []
        for m in self.missions:
            if m["kind"] != kind or mission_done(m):
                continue
            if value is not None:
                m["progress"] = max(int(m["progress"]), int(value))
            else:
                m["progress"] = int(m["progress"]) + int(amount)
            if mission_done(m):
                m["progress"] = int(m["target"])
                done.append(m)
                self.stats["missions"] += 1
        return done

    def set_complete(self) -> bool:
        return all(mission_done(m) for m in self.missions)

    def next_set(self) -> None:
        self.mission_set += 1
        self.missions = mission_set(self.mission_set)

    def new_run(self) -> None:
        """A run starts: one-run missions not yet done start again from zero."""
        for m in self.missions:
            if mission_per_run(m) and not mission_done(m):
                m["progress"] = 0

    # --- json ---

    def to_json(self) -> dict:
        return {
            "version": 1,
            "best": int(self.best),
            "coins": int(self.coins),
            "keys": int(self.keys),
            "hoverboards": int(self.hoverboards),
            "upgrades": {k: int(v) for k, v in self.upgrades.items()},
            "unlocked": list(self.unlocked),
            "selected": self.selected,
            "mission_set": int(self.mission_set),
            "missions": [{"kind": m["kind"], "target": int(m["target"]), "progress": int(m["progress"])}
                         for m in self.missions],
            "stats": {k: (round(float(v), 1) if k == "play_seconds" else int(v)) for k, v in self.stats.items()},
        }

    @classmethod
    def from_json(cls, data) -> Profile:  # noqa: ANN001
        p = cls()
        if not isinstance(data, dict):
            return p
        p.best = _int(data.get("best"), 0)
        p.coins = _int(data.get("coins"), 0)
        p.keys = _int(data.get("keys"), 1)
        p.hoverboards = _int(data.get("hoverboards"), 3)
        ups = data.get("upgrades")
        if isinstance(ups, dict):
            for k in POWERUPS:
                p.upgrades[k] = _int(ups.get(k), 0, 0, UPGRADE_LEVELS)
        unlocked = data.get("unlocked")
        if isinstance(unlocked, list):
            p.unlocked = ["jax"] + [c for c in CHARACTER_IDS if c in unlocked and c != "jax"]
        sel = data.get("selected")
        if sel in p.unlocked:
            p.selected = sel
        p.mission_set = _int(data.get("mission_set"), 0, 0, 10 ** 6)
        p.missions = mission_set(p.mission_set)
        saved = data.get("missions")
        if isinstance(saved, list) and len(saved) == 3:
            ok = all(isinstance(m, dict) and m.get("kind") in MISSION_KINDS for m in saved)
            if ok:
                p.missions = [{"kind": m["kind"], "target": _int(m.get("target"), 1, 1),
                               "progress": _int(m.get("progress"), 0)} for m in saved]
        for m in p.missions:  # a one-run mission cannot carry over a run that was not finished
            if mission_per_run(m) and not mission_done(m):
                m["progress"] = 0
        stats = data.get("stats")
        if isinstance(stats, dict):
            for k in STAT_KEYS:
                if k == "play_seconds":
                    try:
                        p.stats[k] = max(0.0, float(stats.get(k, 0.0)))
                    except (TypeError, ValueError):
                        p.stats[k] = 0.0
                else:
                    p.stats[k] = _int(stats.get(k), 0)
        return p
