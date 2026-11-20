"""The maps, both the game's own: NIGHT SHIFT (zombies: Plant 9, a tank works after dark, eight rooms behind doors,
boarded windows onto the yard, the breaker in the generator room) and CROSSDOCK (multiplayer: a freight yard by day,
a warehouse in the middle, container rows to the north, a tank farm to the south, an office block at each end).

A map is drawn with a few rectangle helpers into rows of characters, then ``parse`` turns it into ``MapData``: the
grids the game collides and aims with, what the renderer builds its geometry from (wall heights, see-through panes,
low boxes, round tanks, lamps) and the things on it (doors, windows, wall buys, machines, spawns)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

WALL_H = 1.5  # the height of a room (cells)
# solid walls: texture, height (taller than a room: the outside of a building)
WALLS = {"#": ("concrete", 1.9), "M": ("metal", 3.0), "b": ("brick", 3.0), "i": ("drywall", 3.0),
         "C": ("container_blue", 1.3), "R": ("container_red", 1.3), "w": ("fence_wood", 1.5),
         "D": ("door", 1.5), "X": ("debris", 1.5)}
# see-through panes that block walking: texture, stops bullets
GLASS = {"W": ("window6", False), "=": ("chainlink", False), "r": ("glass", False)}
# floors: texture, indoors (a ceiling)
FLOORS = {".": ("wood", True), "t": ("tile", True), "g": ("shopfloor", True), ",": ("grass", False),
          "_": ("asphalt", False), ":": ("sidewalk", False), ";": ("dirt", False)}
# low boxes: side texture, top texture, height (bullets pass over them; the tall stack stops everything)
BOXES = {"o": ("crate", "crate_top", 0.62), "O": ("crate", "crate_top", 1.24), "B": ("barrier", "barrier", 0.5),
         "%": ("barrier", "barrier", 0.22)}
# round things: side texture, top texture, radius, height
ROUNDS = {"@": ("tank", "steel", 1.36, 2.5), "d": ("drum", "steel", 0.27, 0.6)}
OBJECTS = {"J": "perk:jugg", "S": "perk:swift", "U": "perk:double", "Q": "perk:rally", "N": "perk:stamina",
           "G": "perk:deadeye", "?": "box", "K": "pap", "!": "power", "T": "tree", "x": "barrel"}
LAMPS = {"*": ((255, 176, 96), 5.2), "+": ((190, 220, 255), 5.0), "~": ((255, 70, 50), 3.6)}
MARKERS = {"P": "player", "z": "zombie", "A": "ally", "E": "enemy"}
CEILING = {"wood": "ceiling", "tile": "ceiling", "shopfloor": "ceiling_truss"}
FACES = ((1, 0), (-1, 0), (0, 1), (0, -1))  # the cell beside each face: east, west, south, north


# --- drawing -------------------------------------------------------------------------------------------------------

class Canvas:
    def __init__(self, w: int, h: int, fill: str) -> None:
        self.w, self.h = w, h
        self.g = [[fill] * w for _ in range(h)]

    def box(self, x0: int, y0: int, x1: int, y1: int, ch: str) -> None:
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.g[y][x] = ch

    def frame(self, x0: int, y0: int, x1: int, y1: int, ch: str) -> None:
        self.box(x0, y0, x1, y0, ch)
        self.box(x0, y1, x1, y1, ch)
        self.box(x0, y0, x0, y1, ch)
        self.box(x1, y0, x1, y1, ch)

    def put(self, x: int, y: int, ch: str) -> None:
        self.g[y][x] = ch

    def puts(self, ch: str, *cells: tuple[int, int]) -> None:
        for x, y in cells:
            self.g[y][x] = ch

    def tank(self, x: int, y: int) -> None:
        """A storage tank on its 3 x 3 slab, the middle at (x, y)."""
        self.box(x - 1, y - 1, x + 1, y + 1, "%")
        self.g[y][x] = "@"

    def mirror(self, swap: dict) -> None:
        """Copy the west half onto the east half, left to right."""
        for row in self.g:
            for x in range(self.w // 2):
                ch = row[x]
                row[self.w - 1 - x] = swap.get(ch, ch)

    def rows(self) -> list[str]:
        return ["".join(r) for r in self.g]


def _night_shift() -> list[str]:
    c = Canvas(46, 34, ";")
    c.frame(0, 0, 45, 33, "#")
    c.frame(4, 4, 41, 29, "M")  # the works
    c.box(5, 5, 13, 13, "t")  # the break room (start)
    c.box(15, 5, 31, 18, "g")  # the shop floor
    c.box(5, 15, 13, 22, ".")  # the locker hall
    c.box(33, 5, 40, 13, ".")  # the drawing office
    c.box(5, 24, 13, 28, "g")  # the stores
    c.box(15, 20, 31, 28, "g")  # the loading dock
    c.box(33, 15, 40, 22, "g")  # the generator room
    c.box(33, 24, 40, 28, "t")  # the test bay
    for x0, y0, x1, y1 in ((14, 5, 14, 28), (32, 5, 32, 28), (5, 14, 13, 14), (5, 23, 13, 23), (15, 19, 31, 19),
                           (33, 14, 40, 14), (33, 23, 40, 23)):
        c.box(x0, y0, x1, y1, "i")
    # doors and rubble (the prices are in MAP_DEFS)
    c.box(14, 9, 14, 10, "D")
    c.box(8, 14, 9, 14, "D")
    c.box(32, 9, 32, 10, "D")
    c.box(8, 23, 9, 23, "D")
    c.box(22, 19, 23, 19, "X")
    c.put(14, 26, "D")
    c.box(36, 14, 37, 14, "D")
    c.box(32, 25, 32, 26, "D")
    c.box(36, 23, 37, 23, "X")
    # boarded windows onto the yard
    c.puts("W", (4, 7), (4, 11), (9, 4), (19, 4), (27, 4), (4, 17), (4, 21), (41, 7), (41, 11), (36, 4), (4, 26),
           (9, 29), (18, 29), (28, 29), (41, 17), (41, 20), (41, 26), (36, 29))
    # the shop floor: a vessel on its slab, crates, drums
    c.tank(23, 11)
    c.puts("o", (17, 7), (17, 8), (29, 15), (28, 15), (19, 16), (35, 8), (38, 11), (12, 21), (6, 27), (24, 22),
           (20, 26))
    c.puts("O", (12, 25), (25, 22), (39, 16), (39, 17), (30, 27))
    c.puts("d", (30, 6), (16, 12), (26, 17), (16, 21), (34, 28), (2, 14), (43, 23), (24, 31))
    c.box(17, 23, 18, 23, "C")
    c.box(27, 26, 28, 26, "R")
    # machines, the crate's three places, wall racks
    c.put(13, 5, "Q")
    c.put(5, 22, "J")
    c.put(40, 5, "S")
    c.put(15, 28, "U")
    c.put(31, 5, "N")
    c.put(5, 28, "G")
    c.puts("?", (15, 18), (31, 20), (33, 13))
    c.put(39, 27, "K")
    c.put(40, 21, "!")
    for ch, (x, y) in zip("12345678", ((14, 6), (11, 14), (23, 4), (4, 19), (41, 9), (24, 29), (40, 14), (34, 29)),
                          strict=True):
        c.put(x, y, ch)
    # lamps
    c.puts("+", (7, 7), (11, 11), (9, 18), (36, 9), (36, 26))
    c.puts("*", (18, 8), (22, 15), (28, 8), (28, 13), (9, 26), (19, 24), (27, 23), (10, 20))
    c.puts("~", (36, 18))
    c.put(9, 9, "P")
    c.puts("z", (2, 7), (2, 11), (2, 17), (2, 21), (2, 26), (9, 2), (19, 2), (27, 2), (36, 2), (43, 7), (43, 11),
           (43, 17), (43, 20), (43, 26), (9, 31), (18, 31), (28, 31), (36, 31),
           (20, 14), (27, 9), (21, 25), (29, 23), (7, 19), (37, 7), (35, 19), (11, 27), (36, 27))
    c.puts("T", (2, 3), (43, 30), (31, 2), (2, 31), (43, 2), (13, 31))
    return c.rows()


def _crossdock() -> list[str]:
    c = Canvas(44, 30, "_")
    c.frame(0, 0, 43, 29, "#")
    c.box(1, 1, 4, 28, ",")  # the verge the west team starts on
    c.box(5, 1, 5, 28, ":")
    # the west office block
    c.frame(7, 11, 12, 18, "b")
    c.box(8, 12, 11, 17, ".")
    c.box(7, 14, 7, 15, ".")
    c.puts(".", (12, 13), (12, 16), (10, 11), (9, 18))
    c.puts("r", (7, 12), (7, 17), (12, 14), (12, 15))
    c.put(11, 12, "o")
    # the north lane: container rows
    c.box(8, 3, 10, 3, "C")
    c.box(13, 5, 13, 7, "R")
    c.box(16, 2, 18, 2, "C")
    c.box(16, 6, 18, 6, "R")
    c.puts("o", (11, 7), (14, 2))
    # the south lane: the tank farm
    c.tank(10, 24)
    c.tank(17, 25)
    c.box(13, 21, 14, 21, "B")
    c.puts("B", (7, 21), (7, 22))
    c.puts("d", (13, 26), (6, 26), (14, 12), (14, 17))
    c.puts("T", (2, 5), (3, 23))
    # the warehouse in the middle (its east half comes from the mirror)
    c.frame(16, 9, 27, 20, "M")
    c.box(17, 10, 26, 19, "g")
    c.box(16, 11, 16, 12, "g")
    c.box(16, 17, 16, 18, "g")
    c.puts("r", (16, 14), (16, 15), (19, 9), (19, 20))
    c.puts("g", (21, 9), (21, 20))
    c.puts("o", (19, 13), (18, 16), (20, 18))
    c.puts("O", (21, 14), (21, 15))
    c.puts("+", (19, 11), (19, 18))
    # the middle of the lanes
    c.box(20, 4, 21, 4, "R")
    c.box(20, 25, 21, 25, "B")
    c.puts("A", (2, 3), (2, 8), (2, 14), (2, 20), (2, 26), (4, 11), (4, 17))
    c.mirror({"A": "E"})
    return c.rows()


# zombies wall racks by digit, door prices by the door's first cell (row-major)
MAP_DEFS = {
    "night_shift": dict(
        title="NIGHT SHIFT", rows=_night_shift,
        wallbuys={"1": "oldguard", "2": "twinoak", "3": "kestrel", "4": "br9", "5": "volley16", "6": "ridgeback",
                  "7": "saber", "8": "rattler"},
        doors={(14, 9): 750, (8, 14): 750, (32, 9): 1000, (8, 23): 1000, (22, 19): 1250, (14, 26): 1000,
               (36, 14): 1250, (32, 25): 1250, (36, 23): 1500},
        grade="dusk", scene=dict(kind="night", fog=(16, 22, 34), fog_dist=13.0, ambient_out=(0.34, 0.40, 0.56),
                                 ambient_in=(0.30, 0.31, 0.34), sun=(0.35, -0.5, 0.8, 0.16),
                                 sun_color=(0.55, 0.65, 0.9))),
    "crossdock": dict(
        title="CROSSDOCK", rows=_crossdock, wallbuys={}, doors={},
        grade="day", scene=dict(kind="day", fog=(182, 200, 212), fog_dist=30.0, ambient_out=(0.80, 0.84, 0.92),
                                ambient_in=(0.60, 0.60, 0.62), sun=(-0.45, -0.6, 0.66, 0.42),
                                sun_color=(1.0, 0.93, 0.78))),
}
ZOMBIES_MAP = "night_shift"
MP_MAP = "crossdock"


# --- parsing -------------------------------------------------------------------------------------------------------

@dataclass
class Door:
    cells: list
    cost: int
    kind: str  # door / debris
    zone_a: int = -1
    zone_b: int = -1
    opened: bool = False


@dataclass
class Window:
    cell: tuple
    inside: tuple
    outside: tuple
    planks: int = 6
    tex0: int = 0  # the texture with no planks (+ planks: the current one)


@dataclass
class MapData:
    key: str
    title: str
    w: int
    h: int
    rows: list
    tex_names: list
    solid: np.ndarray  # a wall: nothing is seen or walked through it
    wall: np.ndarray
    face: np.ndarray  # the texture of each face of a wall
    glass: np.ndarray  # the texture of a see-through pane (-1: none)
    floor: np.ndarray
    ceil: np.ndarray
    block: np.ndarray  # no walking
    bullet: np.ndarray  # stops every bullet
    top: np.ndarray  # low cover: bullets below this height stop here (0: none)
    hgt: np.ndarray  # how high the wall (or the wall over a doorway / window) goes
    upper: np.ndarray  # the texture above a room's height (-1: nothing up there)
    grade: str
    scene: dict
    objects: list = field(default_factory=list)  # (kind, x, y)
    doors: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    wallbuys: list = field(default_factory=list)  # (cell, weapon id, the floor cells in front)
    boxes: list = field(default_factory=list)  # (x0, y0, x1, y1, height, side texture, top texture)
    rounds: list = field(default_factory=list)  # (x, y, radius, height, side texture, top texture)
    lamps: list = field(default_factory=list)  # (x, y, z, (r, g, b), radius)
    spawns: dict = field(default_factory=dict)
    zone: np.ndarray | None = None  # zone id per walkable cell (-1 elsewhere)
    outside_zones: set = field(default_factory=set)  # zombie-only zones

    def tex(self, name: str) -> int:
        if name not in self.tex_names:
            self.tex_names.append(name)
        return self.tex_names.index(name)

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def walkable(self, x: int, y: int) -> bool:
        return self.inside(x, y) and not self.block[y, x]

    def open_door(self, door: Door) -> None:
        door.opened = True
        for x, y in door.cells:
            self.solid[y, x] = False
            self.block[y, x] = False
            self.bullet[y, x] = False
            self.wall[y, x] = -1
            self.face[:, y, x] = -1


def _neighbour_floor(rows: list[str], x: int, y: int) -> str:
    """The floor most of a cell's walkable neighbours have (for machines, doors and markers)."""
    counts: dict[str, int] = {}
    for dx, dy in FACES + ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        nx, ny = x + dx, y + dy
        if 0 <= ny < len(rows) and 0 <= nx < len(rows[0]) and rows[ny][nx] in FLOORS:
            ch = rows[ny][nx]
            counts[ch] = counts.get(ch, 0) + 1
    return max(counts, key=lambda k: (counts[k], FLOORS[k][1])) if counts else ","


def _base_wall(rows: list[str], x: int, y: int) -> str:
    """The wall a door, window or rack is set in: the character of a plain wall beside it."""
    for dx, dy in FACES:
        nx, ny = x + dx, y + dy
        if 0 <= ny < len(rows) and 0 <= nx < len(rows[0]) and rows[ny][nx] in WALLS and rows[ny][nx] not in "DX":
            return rows[ny][nx]
    return "i"


def parse(key: str) -> MapData:
    spec = MAP_DEFS[key]
    rows = spec["rows"]()
    h, w = len(rows), len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError(f"map {key}: ragged rows")
    m = MapData(key, spec["title"], w, h, rows, [],
                solid=np.zeros((h, w), bool), wall=np.full((h, w), -1, np.int32),
                face=np.full((4, h, w), -1, np.int32), glass=np.full((h, w), -1, np.int32),
                floor=np.zeros((h, w), np.int32), ceil=np.full((h, w), -1, np.int32),
                block=np.zeros((h, w), bool), bullet=np.zeros((h, w), bool), top=np.zeros((h, w), np.float32),
                hgt=np.zeros((h, w), np.float32), upper=np.full((h, w), -1, np.int32), grade=spec["grade"],
                scene=dict(spec["scene"]))
    m.spawns = {"player": [], "zombie": [], "ally": [], "enemy": []}
    digits = spec["wallbuys"]
    door_cells: list[tuple[int, int]] = []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            fch = ch if ch in FLOORS else _neighbour_floor(rows, x, y)
            fname, indoor = FLOORS[fch]
            m.floor[y, x] = m.tex(fname)
            if indoor:
                m.ceil[y, x] = m.tex(CEILING[fname])
            if ch in WALLS or ch in digits:
                base = _base_wall(rows, x, y) if (ch in digits or ch in "DX") else ch
                bname, bh = WALLS[base]
                tex = m.tex(bname if ch in digits else WALLS[ch][0])
                m.solid[y, x] = m.block[y, x] = m.bullet[y, x] = True
                m.wall[y, x] = tex
                m.face[:, y, x] = tex
                m.hgt[y, x] = bh
                if bh > WALL_H:
                    m.upper[y, x] = m.tex(bname)
                if ch in "DX":
                    door_cells.append((x, y))
                    m.ceil[y, x] = m.tex("ceiling")  # the lintel once it is open
                if ch in digits:
                    wid = digits[ch]
                    fronts = []
                    for f, (dx, dy) in enumerate(FACES):
                        nx, ny = x + dx, y + dy
                        if m.inside(nx, ny) and rows[ny][nx] not in WALLS and rows[ny][nx] not in digits:
                            m.face[f, y, x] = m.tex(f"rack:{bname}:{wid}")
                            fronts.append((nx, ny))
                    m.wallbuys.append(((x, y), wid, fronts))
            elif ch in GLASS:
                name, stops = GLASS[ch]
                m.block[y, x] = True
                m.bullet[y, x] = stops
                base = _base_wall(rows, x, y)
                bname, bh = WALLS[base]
                if ch == "W":
                    tex0 = m.tex(f"window:{bname}:0")
                    for k in range(1, 7):
                        m.tex(f"window:{bname}:{k}")
                    m.windows.append(Window((x, y), (x, y), (x, y), 6, tex0))
                    m.glass[y, x] = tex0 + 6
                elif ch == "r":
                    m.glass[y, x] = m.tex(f"glass:{bname}")
                else:
                    m.glass[y, x] = m.tex(name)
                if ch in "Wr" and bh > WALL_H:  # the wall goes on over the window
                    m.hgt[y, x] = bh
                    m.upper[y, x] = m.tex(bname)
                    m.ceil[y, x] = m.tex("ceiling")
            elif ch in BOXES:
                side, topn, bh = BOXES[ch]
                m.block[y, x] = True
                m.top[y, x] = bh
                if bh > 1.15:
                    m.bullet[y, x] = True
                if ch != "%" or all(rows[y + dy][x + dx] != "@" for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                    m.boxes.append((x, y, x + 1, y + 1, bh, m.tex(side), m.tex(topn)))
            elif ch in ROUNDS:
                side, topn, rad, bh = ROUNDS[ch]
                m.rounds.append((x + 0.5, y + 0.5, rad, bh, m.tex(side), m.tex(topn)))
                if rad > 0.5:  # a tank: its whole slab is closed
                    x0, y0, x1, y1 = x - 1, y - 1, x + 2, y + 2
                    m.block[y0:y1, x0:x1] = m.bullet[y0:y1, x0:x1] = True
                    m.boxes.append((x0, y0, x1, y1, BOXES["%"][2], m.tex("barrier"), m.tex("barrier")))
                else:
                    m.block[y, x] = True
                    m.top[y, x] = bh
            elif ch in OBJECTS:
                m.objects.append((OBJECTS[ch], x, y))
                m.block[y, x] = True
            elif ch in LAMPS:
                col, rad = LAMPS[ch]
                m.lamps.append((x + 0.5, y + 0.5, WALL_H - 0.12 if indoor else 1.7, col, rad))
            elif ch in MARKERS:
                m.spawns[MARKERS[ch]].append((x + 0.5, y + 0.5))
    for kind, x, y in m.objects:
        if kind == "barrel":
            m.lamps.append((x + 0.5, y + 0.5, 0.9, (255, 140, 50), 4.2))
    _make_doors(m, door_cells, spec["doors"])
    _make_zones(m)
    return m


def _make_doors(m: MapData, cells: list, costs: dict) -> None:
    left = set(cells)
    for cell in cells:
        if cell not in left:
            continue
        group, todo = [], [cell]
        left.discard(cell)
        while todo:
            cx, cy = todo.pop()
            group.append((cx, cy))
            for dx, dy in FACES:
                n = (cx + dx, cy + dy)
                if n in left and m.rows[n[1]][n[0]] == m.rows[cy][cx]:
                    left.discard(n)
                    todo.append(n)
        group.sort(key=lambda c: (c[1], c[0]))
        kind = "debris" if m.rows[group[0][1]][group[0][0]] == "X" else "door"
        m.doors.append(Door(group, int(costs.get(group[0], 1000)), kind))


def _flood(m: MapData, start: tuple[int, int], passable) -> set:  # noqa: ANN001
    seen = {start}
    todo = deque([start])
    while todo:
        x, y = todo.popleft()
        for dx, dy in FACES:
            n = (x + dx, y + dy)
            if n not in seen and m.inside(*n) and passable(*n):
                seen.add(n)
                todo.append(n)
    return seen


def _make_zones(m: MapData) -> None:
    """Zones: the walkable areas between doors and windows; the zombie-only ones (not reachable through doors)."""
    walk = ~m.block
    for _kind, x, y in m.objects:
        walk[y, x] = True  # a machine stands inside its zone
    for y in range(m.h):
        for x in range(m.w):
            if m.top[y, x] > 0:
                walk[y, x] = True  # so does a crate
    m.zone = np.full((m.h, m.w), -1, np.int32)
    zid = 0
    for y in range(m.h):
        for x in range(m.w):
            if walk[y, x] and m.zone[y, x] < 0:
                for cx, cy in _flood(m, (x, y), lambda a, b: bool(walk[b, a])):
                    m.zone[cy, cx] = zid
                zid += 1
    starts = m.spawns["player"] or m.spawns["ally"]
    if not starts:
        return
    sx, sy = int(starts[0][0]), int(starts[0][1])
    door_set = {c for d in m.doors for c in d.cells}
    playable = _flood(m, (sx, sy), lambda a, b: bool(walk[b, a]) or (a, b) in door_set)
    zones_play = {int(m.zone[y, x]) for x, y in playable if m.zone[y, x] >= 0}
    m.outside_zones = {z for z in range(zid) if z not in zones_play}
    for d in m.doors:
        sides = []
        for x, y in d.cells:
            for dx, dy in FACES:
                nx, ny = x + dx, y + dy
                if m.inside(nx, ny) and m.zone[ny, nx] >= 0:
                    sides.append(int(m.zone[ny, nx]))
        sides = sorted(set(sides))
        d.zone_a, d.zone_b = (sides + [-1, -1])[:2]
    for win in m.windows:
        x, y = win.cell
        for dx, dy in ((1, 0), (0, 1)):
            a, b = (x - dx, y - dy), (x + dx, y + dy)
            if m.inside(*a) and m.inside(*b) and m.zone[a[1], a[0]] >= 0 and m.zone[b[1], b[0]] >= 0:
                if int(m.zone[a[1], a[0]]) in m.outside_zones:
                    a, b = b, a
                win.inside, win.outside = a, b
    # a wall rack only faces the rooms the player can reach
    fixed = []
    for (x, y), wid, fronts in m.wallbuys:
        keep = []
        for f, (dx, dy) in enumerate(FACES):
            n = (x + dx, y + dy)
            if n in fronts:
                if int(m.zone[n[1], n[0]]) in m.outside_zones:
                    m.face[f, y, x] = m.wall[y, x]
                else:
                    keep.append(n)
        fixed.append(((x, y), wid, keep))
    m.wallbuys = fixed


def start_zone(m: MapData) -> int:
    starts = m.spawns["player"] or m.spawns["ally"]
    x, y = starts[0]
    return int(m.zone[int(y), int(x)])
