"""Map parsing: an ASCII grid (one character per 64-unit cell) and a legend turn into the cell arrays the renderer and
the world share (heights, flats, textures, light, specials), plus the things placed on the map."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assets import tiles

CELL = 64.0

# things: monsters, items, weapons, keys, decorations (lower case, digits and a few symbols)
THING_CHARS = {
    "@": "player",
    "z": "zombie", "s": "sergeant", "i": "imp", "d": "demon", "l": "soul", "c": "caco", "b": "baron",
    "o": "barrel",
    "h": "hbonus", "a": "abonus", "t": "stim", "m": "medikit", "g": "garmor", "u": "barmor", "q": "soulsphere",
    "e": "clip", "x": "ammobox", "v": "shells", "w": "shellbox", "r": "rocket", "y": "rocketbox",
    "n": "cell", "f": "cellpack", "k": "backpack",
    "3": "w_shotgun", "4": "w_chaingun", "5": "w_launcher", "6": "w_plasma", "7": "w_bfg",
    "!": "bluecard", "$": "yellowcard", "&": "redcard",
    "|": "techlamp", "*": "lamp", ";": "deadguy", ":": "bloodpool", '"': "torch", ",": "gibs",
    "j": "heavy", "p": "shade", "?": "titan", "'": "berserk", "`": "invuln",
}

MONSTERS = ("zombie", "sergeant", "heavy", "imp", "demon", "shade", "soul", "caco", "baron", "titan")

# geometry defaults (a map's legend adds to / overrides these)
BASE_LEGEND: dict[str, dict] = {
    " ": {"kind": "wall", "tex": "@wall"},
    "#": {"kind": "wall", "tex": "@wall"},
    ".": {"kind": "floor"},
    "D": {"kind": "door", "tex": "DOOR"},
    "B": {"kind": "door", "tex": "DOORBLU", "key": "blue"},
    "Y": {"kind": "door", "tex": "DOORYEL", "key": "yellow"},
    "R": {"kind": "door", "tex": "DOORRED", "key": "red"},
    "X": {"kind": "wall", "tex": "EXITOFF", "action": "exit"},
    "E": {"kind": "wall", "tex": "EXITSIGN"},
    "C": {"kind": "wall", "tex": "COMPUTE"},
    "K": {"kind": "wall", "tex": "COMPSTA"},
    "L": {"kind": "wall", "tex": "LITE"},
    "H": {"kind": "wall", "tex": "HAZARD"},
    "P": {"kind": "wall", "tex": "SUPPORT"},
    "M": {"kind": "wall", "tex": "METAL"},
    "~": {"kind": "floor", "floor": -16, "ftex": "NUKAGE", "damage": 5, "lower": "SLIME"},
}


@dataclass
class Thing:
    kind: str
    x: float
    y: float
    cell: int


@dataclass
class MapDef:
    """A hand-made map: its grid, legend and data."""

    key: str  # "E1M1"
    name: str
    grid: str
    legend: dict = field(default_factory=dict)
    wall: str = "TECHTAN"
    floor: float = 0.0
    ceil: float = 128.0
    ftex: str = "FLOOR4"
    ctex: str = "CEIL3"
    light: int = 160
    par: int = 30  # s
    angle: float = 0.0  # player start angle (degrees, 0 = east, 90 = south)
    boss_action: tuple | None = None  # done when every boss is dead
    text: str = ""  # the story shown after the map (episode end)
    sky: str = "dusk"  # the sky panorama (assets.SKIES)
    secret_next: int | None = None  # the map the secret exit leads to
    back_to: int | None = None  # a secret map: the map that follows it


@dataclass
class Grid:
    """The level's cells as flat arrays (index = y * w + x)."""

    w: int
    h: int
    solid: np.ndarray
    fh: np.ndarray
    ch: np.ndarray
    ftex: np.ndarray
    ctex: np.ndarray
    sky: np.ndarray
    light: np.ndarray
    base_light: np.ndarray
    lower_tex: np.ndarray
    upper_tex: np.ndarray
    face_tex: np.ndarray  # (4, n): faces hit moving +x, -x, +y, -y
    damage: np.ndarray
    secret: np.ndarray
    chars: list[str]
    specs: list[dict]
    things: list[Thing]
    start: tuple[float, float]

    def idx(self, cx: int, cy: int) -> int:
        return cy * self.w + cx

    def cell_at(self, x: float, y: float) -> int:
        cx = min(max(int(x // CELL), 0), self.w - 1)
        cy = min(max(int(y // CELL), 0), self.h - 1)
        return cy * self.w + cx

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.w and 0 <= cy < self.h


def grid_rows(text: str) -> list[str]:
    rows = [r for r in text.strip("\n").split("\n")]
    width = max(len(r) for r in rows)
    return [r.ljust(width) for r in rows]


def parse(m: MapDef) -> Grid:
    ts = tiles()
    rows = grid_rows(m.grid)
    h, w = len(rows), len(rows[0])
    n = w * h
    legend = dict(BASE_LEGEND)
    legend.update(m.legend)
    chars = [c for r in rows for c in r]

    def spec_of(c: str) -> dict:
        s = legend.get(c)
        if s is None:
            raise ValueError(f"{m.key}: no legend for {c!r}")
        return s

    # things inherit the sector of their neighbours
    things: list[Thing] = []
    resolved = list(chars)
    start = None
    for i, c in enumerate(chars):
        if c in THING_CHARS and c not in legend:
            cy, cx = divmod(i, w)
            votes: dict[str, int] = {}
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    nc = chars[ny * w + nx]
                    if nc not in THING_CHARS or nc in legend:
                        s = legend.get(nc, {})
                        if s.get("kind") in ("floor", "lift"):
                            votes[nc] = votes.get(nc, 0) + (2 if abs(dx) + abs(dy) == 1 else 1)
            resolved[i] = max(votes, key=votes.get) if votes else "."
            kind = THING_CHARS[c]
            pos = ((cx + 0.5) * CELL, (cy + 0.5) * CELL)
            if kind == "player":
                start = pos
            else:
                things.append(Thing(kind, pos[0], pos[1], i))
    if start is None:
        raise ValueError(f"{m.key}: no player start")
    # legend entries may carry a thing on their own sector (an item on a one-cell walkway)
    for i, c in enumerate(chars):
        s = legend.get(c)
        if s and s.get("thing"):
            cy, cx = divmod(i, w)
            things.append(Thing(s["thing"], (cx + 0.5) * CELL, (cy + 0.5) * CELL, i))

    solid = np.zeros(n, bool)
    fh = np.zeros(n, np.float32)
    ch = np.zeros(n, np.float32)
    ftex = np.zeros(n, np.int32)
    ctex = np.zeros(n, np.int32)
    sky = np.zeros(n, bool)
    light = np.zeros(n, np.int32)
    lower = np.zeros(n, np.int32)
    upper = np.zeros(n, np.int32)
    face = np.zeros((4, n), np.int32)
    damage = np.zeros(n, np.int32)
    secret = np.zeros(n, bool)
    specs: list[dict] = []

    def wid(name: str) -> int:
        name = m.wall if name == "@wall" else name
        return ts.wall_ids[name]

    for i, c in enumerate(resolved):
        s = spec_of(c)
        specs.append(s)
        kind = s.get("kind", "floor")
        floor_h = float(s.get("floor", m.floor))
        ceil_h = float(s.get("ceil", m.ceil))
        fh[i] = floor_h
        ch[i] = ceil_h
        ftex[i] = ts.flat_ids[s.get("ftex", m.ftex)]
        ctex[i] = ts.flat_ids[s.get("ctex", m.ctex)]
        sky[i] = bool(s.get("sky", False))
        light[i] = int(s.get("light", m.light))
        lower[i] = wid(s.get("lower", "@wall"))
        upper[i] = wid(s.get("upper", "@wall"))
        damage[i] = int(s.get("damage", 0))
        secret[i] = bool(s.get("secret", False))
        if kind == "wall":
            solid[i] = True
            face[:, i] = wid(s.get("tex", "@wall"))
        elif kind == "door":
            upper[i] = wid(s.get("tex", "DOOR"))
            ch[i] = fh[i]  # closed
    # doors: open height, track textures on the jambs, light of the lit side
    trak = ts.wall_ids["DOORTRAK"]
    for i in range(n):
        s = specs[i]
        if s.get("kind") != "door":
            continue
        cy, cx = divmod(i, w)
        neigh = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                neigh.append((dx, dy, ny * w + nx))
        open_ceil = [ch[j] for _, _, j in neigh if not solid[j] and specs[j].get("kind") != "door"]
        s_open = (min(open_ceil) if open_ceil else fh[i] + 128.0) - 4.0
        specs[i] = dict(s, open_height=float(max(s_open, fh[i] + 64.0)))
        lit = [light[j] for _, _, j in neigh if not solid[j]]
        if lit:
            light[i] = max(lit)
        if not s.get("secretwall"):
            for dx, dy, j in neigh:
                if solid[j]:
                    # the jamb's face looking at the door: hit moving -x when the jamb is east of the door, ...
                    f = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3}[(dx, dy)]
                    face[f, j] = trak
    return Grid(w, h, solid, fh, ch, ftex, ctex, sky, light, light.copy(), lower, upper, face, damage, secret,
                resolved, specs, things, start)
