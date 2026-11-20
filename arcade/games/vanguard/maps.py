"""Vanguard's maps, built in code on a grid of one-metre cells.

Every cell has a floor height (a building or a cliff is just a very high floor) and may carry one slab above it (a
bridge, a roof, an arch: walk under it or on it). Spawn rooms are marked cells with door cells; a room belongs to one
team at a time. On top of the grid sit the things that are only looked at (trees, awnings, lamps, rails: ``props``,
rows for the renderer's instanced shapes) and the things the match needs (spawn points, health packs, the control
point, the cart's track)."""

from __future__ import annotations

import math
import random

import numpy as np

from . import models as M

# materials (the world shader draws each one's pattern)
PLASTER, PAVING, COBBLE, WOOD, GRASS, SAND, METAL, ROOF, CRATE, ROCK, CONCRETE, TRIM, STRIPE, GRAVEL = range(14)
NO_SLAB = 1.0e9


class MapData:
    def __init__(self, key: str, name: str, mode: str, w: int, h: int) -> None:
        self.key, self.name, self.mode, self.w, self.h = key, name, mode, w, h
        self.blurb = ""
        self.fh = np.zeros((h, w), np.float32)
        self.top_mat = np.zeros((h, w), np.uint8)
        self.top_col = np.zeros((h, w, 3), np.uint8)
        self.side_mat = np.zeros((h, w), np.uint8)
        self.side_col = np.zeros((h, w, 3), np.uint8)
        self.sb = np.full((h, w), NO_SLAB, np.float32)
        self.st = np.full((h, w), -NO_SLAB, np.float32)
        self.slab_mat = np.zeros((h, w), np.uint8)
        self.slab_col = np.zeros((h, w, 3), np.uint8)
        self.room = np.zeros((h, w), np.uint8)
        self.door = np.zeros((h, w), np.uint8)
        self.rooms: dict[int, dict] = {}  # id -> {"spawns": [(x, y, yaw)], "doors": [(x0, y0, x1, y1, axis)]}
        self.packs: list[tuple] = []  # (x, y, big)
        self.props: list = []  # instance rows
        self.point: tuple | None = None  # (x, y, radius)
        self.track: list[tuple] = []
        self.checkpoints: list[float] = []  # distances along the track (the last is its end)
        self.phases: list[tuple] = []  # (attackers' room, defenders' room) per stretch of the track
        self.control_rooms = (1, 2)
        self.sky: dict = {}
        self.overview = (0.0, 0.0, 30.0)  # where the menu's camera looks: (x, y, distance)

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h


class Builder:
    def __init__(self, md: MapData, seed: int = 1) -> None:
        self.md = md
        self.rng = random.Random(seed)

    # --- ground ---------------------------------------------------------------------------------------------------
    def fill(self, x0: int, y0: int, x1: int, y1: int, h: float, top: tuple, side: tuple | None = None) -> None:
        """Cells [x0, x1) x [y0, y1) at height ``h``; ``top`` / ``side`` are (material, (r, g, b))."""
        m = self.md
        m.fh[y0:y1, x0:x1] = h
        m.top_mat[y0:y1, x0:x1] = top[0]
        m.top_col[y0:y1, x0:x1] = top[1]
        s = side or top
        m.side_mat[y0:y1, x0:x1] = s[0]
        m.side_col[y0:y1, x0:x1] = s[1]

    def stairs(self, x0: int, y0: int, x1: int, y1: int, h0: float, h1: float, axis: str, top: tuple) -> None:
        """Steps from ``h0`` (the level before the first cell) to ``h1`` (the last cell) along +x or +y."""
        n = (x1 - x0) if "x" in axis else (y1 - y0)
        for i in range(n):
            k = (i + 1) / n
            h = h0 + (h1 - h0) * k
            if "x" in axis:
                self.fill(x0 + i, y0, x0 + i + 1, y1, h, top)
            else:
                self.fill(x0, y0 + i, x1, y0 + i + 1, h, top)

    def slab(self, x0: int, y0: int, x1: int, y1: int, bottom: float, top: float, mat: tuple) -> None:
        m = self.md
        m.sb[y0:y1, x0:x1] = bottom
        m.st[y0:y1, x0:x1] = top
        m.slab_mat[y0:y1, x0:x1] = mat[0]
        m.slab_col[y0:y1, x0:x1] = mat[1]

    def path(self, pts: list, width: float, h: float, top: tuple) -> None:
        """Open ground within ``width / 2`` of the line through ``pts``."""
        m = self.md
        r = width / 2
        for (ax, ay), (bx, by) in zip(pts, pts[1:], strict=False):
            x0, x1 = int(min(ax, bx) - r), int(max(ax, bx) + r) + 1
            y0, y1 = int(min(ay, by) - r), int(max(ay, by) + r) + 1
            for y in range(max(y0, 1), min(y1, m.h - 1)):
                for x in range(max(x0, 1), min(x1, m.w - 1)):
                    px, py = x + 0.5, y + 0.5
                    dx, dy = bx - ax, by - ay
                    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9)))
                    if math.hypot(px - ax - dx * t, py - ay - dy * t) <= r:
                        self.fill(x, y, x + 1, y + 1, h, top)

    def crate(self, x: int, y: int, h: float = 1.1, col: tuple = (196, 150, 90)) -> None:
        base = float(self.md.fh[y, x])
        self.fill(x, y, x + 1, y + 1, base + h, (CRATE, col))

    def room(self, rid: int, x0: int, y0: int, x1: int, y1: int, h: float, top: tuple, spawns: list) -> None:
        self.fill(x0, y0, x1, y1, h, top)
        self.md.room[y0:y1, x0:x1] = rid
        self.md.rooms[rid] = {"spawns": spawns, "doors": [], "rect": (x0, y0, x1, y1), "h": h}

    def door(self, rid: int, x0: int, y0: int, x1: int, y1: int, top: tuple) -> None:
        h = self.md.rooms[rid]["h"]
        self.fill(x0, y0, x1, y1, h, top)
        self.md.room[y0:y1, x0:x1] = rid
        self.md.door[y0:y1, x0:x1] = rid
        self.md.rooms[rid]["doors"].append((x0, y0, x1, y1, h))

    def pack(self, x: float, y: float, big: bool = False) -> None:
        self.md.packs.append((x, y, big))

    def add(self, rows: list) -> None:
        self.md.props += rows

    # --- whole-map helpers ----------------------------------------------------------------------------------------
    def mirror_x(self) -> None:
        """Copy the left half over the right half, mirrored (rooms 1 -> 2)."""
        m = self.md
        half = m.w // 2
        for arr in (m.fh, m.top_mat, m.top_col, m.side_mat, m.side_col, m.sb, m.st, m.slab_mat, m.slab_col):
            arr[:, half:] = arr[:, :half][:, ::-1]
        for arr in (m.room, m.door):
            right = arr[:, :half][:, ::-1].copy()
            right[right == 1] = 2
            arr[:, half:] = right
        left = m.rooms[1]
        m.rooms[2] = {
            "spawns": [(m.w - x, y, math.pi - yaw) for x, y, yaw in left["spawns"]],
            "doors": [(m.w - x1, y0, m.w - x0, y1, h) for x0, y0, x1, y1, h in left["doors"]],
            "rect": (m.w - left["rect"][2], left["rect"][1], m.w - left["rect"][0], left["rect"][3]), "h": left["h"],
        }
        m.packs += [(m.w - x, y, big) for x, y, big in m.packs if abs(x - m.w / 2) > 0.6]
        m.props += [M.mirror_row(r, m.w) for r in m.props if abs(r[0] - m.w / 2) > 0.6]


def _h01(*v: int) -> float:
    n = 1469598103
    for i in v:
        n = ((n ^ (i & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    n ^= n >> 15
    n = (n * 2246822519) & 0xFFFFFFFF
    n ^= n >> 13
    return (n & 0xFFFF) / 65535.0


# =====================================================================================================================
# MARIGOLD MARKET - control
# =====================================================================================================================
def marigold_market() -> MapData:
    md = MapData("marigold", "MARIGOLD MARKET", "control", 64, 44)
    md.blurb = ("A market square on a bright morning. The bandstand in the middle is the point; the covered arcade "
                "to the north and the terrace to the south are the ways round.")
    b = Builder(md, 7)
    walls = ((236, 178, 96), (232, 120, 96), (244, 214, 150), (150, 196, 190), (238, 154, 120), (226, 200, 170),
             (176, 150, 200))
    roofs = ((70, 150, 150), (196, 96, 70), (90, 110, 160))
    # every lot starts as a building
    for y in range(md.h):
        for x in range(32):
            lx, ly = (x + (3 if (y // 6) % 2 else 0)) // 6, y // 6
            k = _h01(lx, ly, 11)
            b.fill(x, y, x + 1, y + 1, 7.0 + round(k * 5), (ROOF, roofs[int(_h01(lx, ly, 5) * 2.99)]),
                   (PLASTER, walls[int(_h01(lx, ly, 3) * 6.99)]))
    plaza = (PAVING, (222, 204, 168))
    street = (COBBLE, (182, 166, 148))
    tile = (PAVING, (202, 132, 100))
    wood = (WOOD, (176, 126, 84))
    stone = (PAVING, (230, 220, 198))
    grass = (GRASS, (124, 176, 92))
    # the team's room and its three doors
    b.room(1, 2, 17, 9, 27, 0.0, wood, [(4.5, 19.5, 0.0), (4.5, 22.0, 0.0), (4.5, 24.5, 0.0), (6.5, 20.7, 0.0),
                                         (6.5, 23.3, 0.0)])
    b.door(1, 9, 20, 10, 24, wood)
    b.door(1, 4, 16, 7, 17, wood)
    b.door(1, 4, 27, 7, 28, wood)
    b.fill(10, 18, 24, 26, 0.0, street)  # the main street
    b.fill(3, 10, 8, 16, 0.0, street)  # north alley
    b.fill(3, 5, 32, 10, 0.0, tile)  # the arcade
    b.fill(15, 10, 18, 18, 0.0, street)
    b.fill(22, 10, 26, 12, 0.0, tile)
    b.fill(30, 10, 32, 12, 0.0, tile)
    b.fill(3, 28, 8, 33, 0.0, street)  # south alley
    b.fill(3, 33, 14, 39, 0.0, grass)  # the garden court
    b.fill(11, 26, 14, 33, 0.0, street)
    b.stairs(14, 35, 20, 38, 0.0, 3.0, "x", stone)
    b.fill(20, 33, 32, 40, 3.0, stone, (PLASTER, (226, 200, 170)))  # the terrace
    b.fill(24, 12, 32, 32, 0.0, plaza)  # the square
    b.stairs(24, 27, 26, 33, 0.0, 3.0, "y", stone)
    # the bandstand: a low round stage with a monument in the middle
    for y in range(16, 28):
        for x in range(26, 32):
            if math.hypot(x + 0.5 - 32, y + 0.5 - 22) < 4.4:
                b.fill(x, y, x + 1, y + 1, 0.5, (WOOD, (206, 150, 96)), (TRIM, (240, 232, 214)))
    b.fill(31, 21, 32, 23, 1.6, (TRIM, (240, 232, 214)))
    # cover
    b.fill(27, 14, 28, 16, 1.1, (GRASS, (96, 160, 84)), (TRIM, (232, 222, 200)))
    b.fill(27, 28, 28, 30, 1.1, (GRASS, (96, 160, 84)), (TRIM, (232, 222, 200)))
    for x, y, h in ((25, 20, 1.1), (25, 21, 1.7), (25, 23, 1.1), (16, 19, 1.2), (16, 20, 1.2), (19, 24, 1.2),
                    (20, 24, 1.7), (12, 6, 1.2), (20, 8, 1.2), (21, 8, 1.2), (6, 35, 1.1), (9, 37, 1.1),
                    (29, 9, 1.2), (22, 36, 1.2), (30, 34, 1.2)):
        b.crate(x, y, h, (200, 152, 92) if (x + y) % 2 else (150, 170, 190))
    b.fill(26, 33, 29, 34, 4.1, (TRIM, (236, 226, 204)))  # a parapet on the terrace
    b.slab(21, 18, 23, 26, 3.3, 4.0, (PLASTER, (240, 214, 150)))  # the street gate
    b.slab(3, 5, 32, 10, 3.3, 3.7, (ROOF, (196, 96, 70)))  # the arcade roof
    b.pack(16.5, 7.5)
    b.pack(8.5, 36.0)
    b.pack(32.0, 7.5, True)
    b.pack(32.0, 37.0, True)
    # things to look at
    for x, y in ((27.5, 15.0), (27.5, 29.0)):
        b.add(M.tree(x, y, 1.1, 3.4, (86, 170, 90)))
    for x, y in ((3.6, 38.4), (13.4, 33.6), (9.5, 33.6)):
        b.add(M.tree(x, y, 0.0, 3.8, (240, 170, 70)))
    for x, y, yaw in ((24.0, 14.0, 0.0), (24.0, 17.0, 0.0), (26.5, 12.0, math.pi / 2), (10.0, 26.0, -math.pi / 2)):
        b.add(M.awning(x, y, 2.5, yaw, (240, 96, 80) if int(x + y) % 2 else (70, 160, 200)))
    for x, y in ((10.5, 18.5), (10.5, 25.5), (23.5, 18.5), (23.5, 25.5), (24.5, 12.5), (24.5, 31.5), (20.5, 39.5)):
        b.add(M.lamp(x, y, float(md.fh[int(y), int(x)]), (255, 214, 130)))
    b.add(M.bunting(10.0, 19.0, 20.5, 25.0, 5.2, 9))
    b.add(M.bunting(24.5, 12.5, 31.5, 15.0, 5.6, 8))
    b.add(M.bunting(24.5, 31.0, 31.5, 28.5, 5.6, 8))
    b.add(M.stall(25.5, 21.5, 0.0, (250, 190, 60)))
    b.mirror_x()
    # the middle (not mirrored): the monument and a windmill over the arcade
    b.add(M.monument(32.0, 22.0, 1.6))
    b.add(M.windmill(32.0, 2.5, float(md.fh[2, 31]), (246, 236, 214), (70, 150, 150)))
    b.add(M.bunting(20.5, 25.5, 43.5, 25.5, 6.4, 14))
    md.point = (32.0, 22.0, 5.2)
    md.control_rooms = (1, 2)
    md.sky = {"zenith": (0.20, 0.47, 0.86), "horizon": (0.74, 0.86, 0.94), "sun": (0.46, 0.74, -0.49),
              "sun_col": (1.0, 0.93, 0.78), "amb_sky": (0.50, 0.60, 0.74), "amb_gnd": (0.36, 0.31, 0.27),
              "fog": (0.76, 0.85, 0.92), "fog_d": 120.0, "kind": 0.0, "cloud": 0.55}
    md.overview = (32.0, 22.0, 30.0)
    return md


# =====================================================================================================================
# SUNDOWN SPUR - escort
# =====================================================================================================================
def sundown_spur() -> MapData:
    md = MapData("sundown", "SUNDOWN SPUR", "escort", 124, 54)
    md.blurb = ("A mining spur line at the end of the day. Walk the tank car out of the depot, down the street of "
                "the canyon town and into the foundry yard.")
    b = Builder(md, 3)
    rock_cols = ((196, 104, 70), (206, 118, 76), (184, 94, 66), (214, 132, 88))
    for y in range(md.h):
        for x in range(md.w):
            n = (math.sin(x * 0.21 + 1.3) + math.sin(y * 0.27 + 0.4) + math.sin((x + y) * 0.13)) / 3
            h = 9.0 + round(3.0 * n + 2.0 * _h01(x // 5, y // 5, 2))
            b.fill(x, y, x + 1, y + 1, h, (ROCK, rock_cols[int(_h01(x // 4, y // 4, 9) * 3.99)]),
                   (ROCK, rock_cols[int(_h01(x // 7, y // 7, 1) * 3.99)]))
    sand = (SAND, (222, 178, 124))
    dirt = (SAND, (200, 150, 104))
    wood = (WOOD, (170, 118, 78))
    board = (WOOD, (190, 140, 92))
    conc = (CONCRETE, (176, 170, 164))
    metal = (METAL, (120, 134, 150))
    ledge = (ROCK, (220, 150, 104))
    track = [(13.0, 27.0), (40.0, 27.0), (52.0, 17.0), (78.0, 17.0), (90.0, 33.0), (113.0, 33.0)]
    # --- the depot yard ---
    b.fill(10, 20, 46, 35, 0.0, sand)
    b.fill(3, 14, 12, 21, 0.0, dirt)
    b.stairs(12, 15, 18, 18, 0.0, 3.0, "x", ledge)
    b.fill(18, 13, 44, 20, 3.0, ledge)
    b.fill(3, 33, 10, 41, 0.0, dirt)
    b.fill(10, 37, 46, 41, 0.0, dirt)
    b.fill(14, 35, 17, 37, 0.0, dirt)
    b.fill(38, 35, 41, 37, 0.0, dirt)
    b.slab(10, 37, 46, 41, 3.2, 3.7, (ROCK, (184, 94, 66)))
    b.room(1, 2, 22, 9, 32, 0.0, wood, [(4.5, 24.0, 0.0), (4.5, 27.0, 0.0), (4.5, 30.0, 0.0), (6.5, 25.5, 0.0),
                                         (6.5, 28.5, 0.0)])
    b.door(1, 9, 25, 10, 29, wood)
    b.door(1, 4, 21, 7, 22, wood)
    b.door(1, 4, 32, 7, 33, wood)
    # --- the cut to the town, the street ---
    b.path([track[1], track[2]], 9.0, 0.0, sand)
    b.fill(50, 11, 84, 23, 0.0, sand)
    b.fill(52, 22, 80, 23, 0.5, board)
    b.fill(52, 23, 80, 27, 1.0, board)
    for x in range(50, 84):  # shop fronts along the north side
        lot = x // 7
        b.fill(x, 5, x + 1, 11, 7.0 + round(_h01(lot, 4) * 3), (ROOF, (150, 84, 60)),
               (PLASTER, ((226, 190, 130), (204, 150, 110), (170, 190, 180), (232, 206, 160))[lot % 4]))
    for x in range(52, 80):  # and behind the boardwalk
        lot = (x + 3) // 8
        b.fill(x, 27, x + 1, 36, 6.0 + round(_h01(lot, 8) * 3), (ROOF, (120, 96, 110)),
               (PLASTER, ((214, 160, 120), (176, 196, 190), (236, 200, 140))[lot % 3]))
    b.fill(56, 6, 76, 10, 0.0, wood)  # through the shops
    b.fill(57, 10, 60, 11, 0.0, wood)
    b.fill(72, 10, 75, 11, 0.0, wood)
    b.slab(56, 6, 76, 10, 3.2, 3.6, (WOOD, (150, 100, 70)))
    b.room(2, 58, 28, 66, 35, 1.0, wood, [(60.0, 33.0, -math.pi / 2), (62.0, 33.0, -math.pi / 2),
                                          (64.0, 33.0, -math.pi / 2), (61.0, 31.0, -math.pi / 2),
                                          (63.0, 31.0, -math.pi / 2)])
    b.door(2, 60, 27, 64, 28, wood)
    b.door(2, 66, 30, 67, 33, wood)
    b.fill(67, 29, 80, 34, 1.0, board)
    # --- the cut to the foundry, the yard ---
    b.path([track[3], track[4]], 9.0, 0.0, sand)
    b.stairs(80, 29, 82, 34, 1.0, 0.0, "x", board)
    b.fill(82, 28, 86, 35, 0.0, sand)
    b.fill(86, 25, 120, 45, 0.0, conc)
    b.room(3, 92, 9, 100, 17, 0.0, metal, [(94.0, 11.0, math.pi / 2), (96.0, 11.0, math.pi / 2),
                                           (98.0, 11.0, math.pi / 2), (95.0, 13.0, math.pi / 2),
                                           (97.0, 13.0, math.pi / 2)])
    b.door(3, 94, 17, 98, 18, metal)
    b.door(3, 91, 11, 92, 15, metal)
    b.fill(93, 18, 99, 25, 0.0, conc)
    b.fill(84, 10, 91, 16, 0.0, sand)
    b.room(4, 104, 47, 114, 52, 0.0, metal, [(106.0, 50.0, -math.pi / 2), (108.0, 50.0, -math.pi / 2),
                                             (110.0, 50.0, -math.pi / 2), (112.0, 50.0, -math.pi / 2),
                                             (109.0, 48.5, -math.pi / 2)])
    b.door(4, 105, 46, 108, 47, metal)
    b.door(4, 110, 46, 113, 47, metal)
    b.fill(104, 45, 114, 46, 0.0, conc)
    b.fill(88, 38, 96, 45, 3.0, (METAL, (150, 156, 166)), (CONCRETE, (150, 146, 142)))  # a loading deck
    b.stairs(96, 40, 102, 43, 3.0, 0.0, "x", (METAL, (150, 156, 166)))
    b.fill(100, 26, 103, 29, 5.0, (METAL, (190, 120, 70)))  # a silo
    # cover (clear of the rails)
    for x, y, h in ((20, 23, 1.2), (21, 23, 1.2), (20, 30, 1.2), (28, 31, 1.7), (29, 31, 1.2), (34, 22, 1.2),
                    (35, 22, 1.7), (30, 16, 1.1), (24, 14, 1.1), (60, 13, 1.2), (61, 13, 1.2), (68, 20, 1.2),
                    (69, 20, 1.7), (74, 13, 1.2), (55, 20, 1.2), (98, 36, 1.2), (99, 36, 1.2), (106, 38, 1.2),
                    (107, 38, 1.7), (107, 39, 1.2), (110, 28, 1.2), (94, 30, 1.2), (116, 37, 1.2), (20, 39, 1.1),
                    (66, 7, 1.1), (91, 41, 1.1)):
        b.crate(x, y, h, (188, 140, 86) if (x * 7 + y) % 3 else (120, 150, 170))
    b.pack(26.5, 39.0)
    b.pack(31.0, 16.5, True)
    b.pack(66.5, 8.5)
    b.pack(72.0, 25.0, True)
    b.pack(111.0, 27.0)
    b.pack(92.0, 42.0, True)
    md.track = track
    seg = [math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(track, track[1:], strict=False)]
    md.checkpoints = [seg[0] + seg[1], seg[0] + seg[1] + seg[2], sum(seg)]
    md.phases = [(1, 2), (2, 3), (3, 4)]
    b.add(M.rails(track))
    for d in md.checkpoints[:-1]:
        x, y, yaw = track_at(track, d)
        b.add(M.gate(x, y, yaw, 5.6))
    x, y, yaw = track_at(track, md.checkpoints[-1])
    b.add(M.end_pad(x, y))
    for x, y in ((12.0, 21.5), (12.0, 34.0), (44.0, 21.0), (51.0, 12.0), (83.0, 12.0), (51.0, 22.0), (87.0, 26.0),
                 (119.0, 26.0), (119.0, 44.0), (87.0, 44.0)):
        b.add(M.lamp(x, y, float(md.fh[int(y), int(x)]), (255, 190, 110)))
    for x, y, s in ((16.0, 21.5, 1.0), (43.0, 33.5, 1.3), (53.0, 12.0, 0.9), (82.0, 21.5, 1.2), (118.5, 30.0, 1.1),
                    (30.0, 34.0, 0.8)):
        b.add(M.cactus(x, y, float(md.fh[int(y), int(x)]), s))
    for x in (53.5, 60.5, 67.5, 74.5):
        b.add(M.awning(x + 1.5, 11.0, 2.6, math.pi / 2, (226, 110, 70) if int(x) % 2 else (240, 200, 110)))
    b.add(M.water_tower(47.0, 33.0, float(md.fh[33, 47])))
    b.add(M.chimney(106.0, 22.0, float(md.fh[22, 106])))
    b.add(M.chimney(111.0, 21.0, float(md.fh[21, 111]), 0.8))
    b.add(M.bunting(52.0, 11.5, 80.0, 22.0, 5.4, 14))
    md.sky = {"zenith": (0.24, 0.30, 0.62), "horizon": (0.98, 0.66, 0.40), "sun": (-0.62, 0.42, -0.66),
              "sun_col": (1.0, 0.72, 0.46), "amb_sky": (0.46, 0.44, 0.62), "amb_gnd": (0.40, 0.26, 0.20),
              "fog": (0.94, 0.66, 0.46), "fog_d": 150.0, "kind": 1.0, "cloud": 0.4}
    md.overview = (62.0, 24.0, 44.0)
    return md


def showroom() -> MapData:
    """The little stage the menus show heroes on."""
    md = MapData("showroom", "SHOWROOM", "none", 24, 24)
    b = Builder(md, 1)
    b.fill(0, 0, 24, 24, 0.0, (PAVING, (70, 80, 110)))
    for y in range(24):
        for x in range(24):
            d = math.hypot(x + 0.5 - 12, y + 0.5 - 12)
            if d < 3.6:
                b.fill(x, y, x + 1, y + 1, 0.3, (WOOD, (214, 160, 100)), (TRIM, (240, 232, 214)))
            elif d < 4.6:
                b.fill(x, y, x + 1, y + 1, 0.15, (TRIM, (240, 232, 214)))
    md.sky = {"zenith": (0.10, 0.16, 0.36), "horizon": (0.42, 0.52, 0.78), "sun": (0.5, 0.72, -0.48),
              "sun_col": (1.0, 0.92, 0.80), "amb_sky": (0.46, 0.52, 0.70), "amb_gnd": (0.30, 0.28, 0.34),
              "fog": (0.36, 0.44, 0.68), "fog_d": 60.0, "kind": 2.0, "cloud": 0.0}
    md.rooms[1] = {"spawns": [(12.0, 12.0, 0.0)], "doors": [], "rect": (0, 0, 0, 0), "h": 0.0}
    return md


def track_length(track: list) -> float:
    return sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(track, track[1:], strict=False))


def track_at(track: list, d: float) -> tuple[float, float, float]:
    """(x, y, heading) at distance ``d`` along the track."""
    for (ax, ay), (bx, by) in zip(track, track[1:], strict=False):
        ln = math.hypot(bx - ax, by - ay)
        if d <= ln:
            t = d / max(ln, 1e-9)
            return ax + (bx - ax) * t, ay + (by - ay) * t, math.atan2(by - ay, bx - ax)
        d -= ln
    (ax, ay), (bx, by) = track[-2], track[-1]
    return bx, by, math.atan2(by - ay, bx - ax)


MAPS = {"marigold": marigold_market, "sundown": sundown_spur}
MAP_INFO = {
    "marigold": ("MARIGOLD MARKET", "control", "A sunny market square: bandstand, arcade and terrace."),
    "sundown": ("SUNDOWN SPUR", "escort", "A canyon rail spur at dusk: depot, town street and foundry."),
}
MAPS_BY_MODE = {"control": ["marigold"], "escort": ["sundown"]}


def build(key: str) -> MapData:
    return MAPS[key]() if key in MAPS else showroom()
