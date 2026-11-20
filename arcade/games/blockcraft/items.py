"""Inventories, crafting recipes (2x2 and 3x3, shaped and shapeless, mirrored) and furnace smelting."""

from __future__ import annotations

from app.arcade.games.blockcraft import blocks as B

Stack = list  # [item, count, damage]


def stack(item: int, count: int = 1, damage: int = 0) -> Stack:
    return [int(item), int(count), int(damage)]


def same_kind(a: Stack | None, b: Stack | None) -> bool:
    return a is not None and b is not None and a[0] == b[0] and a[2] == b[2]


class Inventory:
    """36 slots: 0-8 the hotbar, 9-35 the main inventory; each None or [item, count, damage]."""

    SIZE = 36

    def __init__(self) -> None:
        self.slots: list[Stack | None] = [None] * self.SIZE
        self.selected = 0
        self.armor: list[Stack | None] = [None] * 4  # helmet, chestplate, leggings, boots

    def armor_points(self) -> int:
        """How well the worn armour protects (0..20: each point takes 4 % off a blow)."""
        total = 0
        for s in self.armor:
            info = B.armor_info(s[0]) if s else None
            if info is not None:
                total += B.ARMOR_POINTS[info[0]][info[1]]
        return total

    def wear_armor(self) -> None:
        """A blow wears every piece a little; a worn-out piece is gone."""
        for k, s in enumerate(self.armor):
            if s is not None:
                s[2] += 1
                if s[2] >= B.max_damage(s[0]):
                    self.armor[k] = None

    def armor_json(self) -> list:
        return [list(s) if s else None for s in self.armor]

    def load_armor(self, data) -> None:  # noqa: ANN001
        self.armor = [None] * 4
        for k, entry in enumerate((data or [])[:4]):
            try:
                item, count, dmg = (int(v) for v in entry[:3])
            except (TypeError, ValueError):
                continue
            info = B.armor_info(item)
            if info is not None and info[1] == k and count > 0:
                self.armor[k] = stack(item, 1, max(dmg, 0))

    @property
    def held(self) -> Stack | None:
        return self.slots[self.selected]

    def add(self, item: int, count: int = 1, damage: int = 0) -> int:
        """Put items in (topping up stacks first, then the first empty slot, hotbar first); returns what did not
        fit."""
        limit = B.max_stack(item)
        for s in self.slots:
            if count <= 0:
                break
            if s is not None and s[0] == item and s[2] == damage and s[1] < limit:
                n = min(limit - s[1], count)
                s[1] += n
                count -= n
        for i in range(self.SIZE):
            if count <= 0:
                break
            if self.slots[i] is None:
                n = min(limit, count)
                self.slots[i] = stack(item, n, damage)
                count -= n
        return count

    def take_held(self, n: int = 1) -> Stack | None:
        s = self.slots[self.selected]
        if s is None:
            return None
        n = min(n, s[1])
        s[1] -= n
        if s[1] <= 0:
            self.slots[self.selected] = None
        return stack(s[0], n, s[2])

    def count(self, item: int) -> int:
        return sum(s[1] for s in self.slots if s is not None and s[0] == item)

    def damage_held(self, amount: int = 1) -> bool:
        """Wear the held tool; True when it broke."""
        s = self.held
        md = B.max_damage(s[0]) if s else 0
        if not md:
            return False
        s[2] += amount
        if s[2] >= md:
            self.slots[self.selected] = None
            return True
        return False

    def to_json(self) -> list:
        return [[i, s[0], s[1], s[2]] for i, s in enumerate(self.slots) if s is not None]

    def load(self, data: list) -> None:
        self.slots = [None] * self.SIZE
        for entry in data or []:
            try:
                i, item, count, dmg = (int(v) for v in entry[:4])
            except (TypeError, ValueError):
                continue
            if 0 <= i < self.SIZE and item in B.NAMES and count > 0:
                self.slots[i] = stack(item, min(count, B.max_stack(item)), max(dmg, 0))


# --- crafting ----------------------------------------------------------------------------------------------------
PLANKS = frozenset({B.OAK_PLANKS, B.BIRCH_PLANKS, B.SPRUCE_PLANKS, B.ACACIA_PLANKS, B.KAPOK_PLANKS,
                    B.BLACKWOOD_PLANKS, B.MANGROVE_PLANKS})
NEW_WOODS = ((B.ACACIA_LOG, B.ACACIA_PLANKS), (B.KAPOK_LOG, B.KAPOK_PLANKS), (B.BLACKWOOD_LOG, B.BLACKWOOD_PLANKS),
             (B.MANGROVE_LOG, B.MANGROVE_PLANKS))
LOGS = frozenset({B.OAK_LOG, B.BIRCH_LOG, B.SPRUCE_LOG, B.PALE_LOG, B.GLOWSTEM, *(log for log, _ in NEW_WOODS)})
_MATERIALS = (PLANKS, frozenset({B.COBBLE}), frozenset({B.IRON_INGOT}), frozenset({B.GOLD_INGOT}),
              frozenset({B.DIAMOND}))
_TOOL_SHAPES = (("MMM", " S ", " S "), ("MM", "MS", " S"), ("M", "S", "S"), ("M", "M", "S"))


def _key(item: int | frozenset) -> frozenset:
    return item if isinstance(item, frozenset) else frozenset({item})


SHAPED: list[tuple[tuple[str, ...], dict, tuple[int, int]]] = [
    (("P", "P"), {"P": PLANKS}, (B.STICK, 4)),
    (("PP", "PP"), {"P": PLANKS}, (B.CRAFTING_TABLE, 1)),
    (("C", "S"), {"C": B.COAL, "S": B.STICK}, (B.TORCH, 4)),
    (("CCC", "C C", "CCC"), {"C": B.COBBLE}, (B.FURNACE, 1)),
    (("SS", "SS"), {"S": B.SAND}, (B.SANDSTONE, 1)),
    (("SS", "SS"), {"S": B.STONE}, (B.STONE_BRICKS, 4)),
    (("III", "III", "III"), {"I": B.IRON_INGOT}, (B.IRON_BLOCK, 1)),
    (("III", "III", "III"), {"I": B.GOLD_INGOT}, (B.GOLD_BLOCK, 1)),
    (("III", "III", "III"), {"I": B.DIAMOND}, (B.DIAMOND_BLOCK, 1)),
    (("PPP", "SSS", "PPP"), {"P": PLANKS, "S": B.STICK}, (B.BOOKSHELF, 1)),
    (("WWW",), {"W": B.WHEAT_ITEM}, (B.BREAD, 1)),
    (("I I", " I "), {"I": B.IRON_INGOT}, (B.BUCKET, 1)),
    (("WWW", "PPP"), {"W": frozenset(range(B.WOOL, B.WOOL + 16)), "P": PLANKS}, (B.BED, 1)),
    (("WWW", "WWW", "WWW"), {"W": B.WHEAT_ITEM}, (B.HAY, 1)),
    (("SSS", "SDS", "SSS"), {"S": B.STONE_BRICKS, "D": B.DIAMOND}, (B.DEEP_GATE, 1)),  # the way down
    (("GSG", "SDS", "GSG"), {"G": B.GLOWSTONE, "S": B.STARSTONE, "D": B.DIAMOND}, (B.SKY_GATE, 1)),  # the way up
    (("SS", "SS"), {"S": B.RED_SAND}, (B.RED_SANDSTONE, 1)),
    (("MM", "MM"), {"M": B.MUD}, (B.MUD_BRICKS, 4)),
    (("TT", "TT"), {"T": frozenset(B.TERRACOTTAS)}, (B.CLAY_TILES, 4)),
    (("II", "II"), {"I": B.ICE}, (B.PACKED_ICE, 1)),
    (("DD", "DD"), {"D": frozenset({B.STALACTITE, B.STALAGMITE})}, (B.DRIPSTONE, 1)),
    (("HH", "HH"), {"H": B.RABBIT_HIDE}, (B.LEATHER, 1)),
    (("SAS", "ACA", "SAS"), {"S": B.SULFUR, "A": frozenset({B.SAND, B.RED_SAND}), "C": B.COAL}, (B.TNT, 1)),
]
_ARMOR_SHAPES = (("MMM", "M M"), ("M M", "MMM", "MMM"), ("MMM", "M M", "M M"), ("M M", "M M"))
for _m, _mat in enumerate((B.IRON_INGOT, B.GOLD_INGOT, B.DIAMOND)):
    for _k, _shape in enumerate(_ARMOR_SHAPES):
        SHAPED.append((_shape, {"M": _mat}, (B.ARMOR0 + _m * 4 + _k, 1)))
for _m, _mat in enumerate(_MATERIALS):
    for _k, _shape in enumerate(_TOOL_SHAPES):
        SHAPED.append((_shape, {"M": _mat, "S": B.STICK}, (B.tool_id(_m, _k), 1)))
    SHAPED.append((("MM", " S", " S"), {"M": _mat, "S": B.STICK}, (B.HOE0 + _m, 1)))
SHAPELESS: list[tuple[list, tuple[int, int]]] = [
    ([B.OAK_LOG], (B.OAK_PLANKS, 4)), ([B.BIRCH_LOG], (B.BIRCH_PLANKS, 4)),
    ([B.SPRUCE_LOG], (B.SPRUCE_PLANKS, 4)), ([B.PALE_LOG], (B.BIRCH_PLANKS, 4)), ([B.GLOWSTEM], (B.OAK_PLANKS, 4)),
    ([B.IRON_BLOCK], (B.IRON_INGOT, 9)), ([B.GOLD_BLOCK], (B.GOLD_INGOT, 9)), ([B.DIAMOND_BLOCK], (B.DIAMOND, 9)),
    ([B.IRON_INGOT, B.FLINT], (B.FLINT_STEEL, 1)),
    ([B.SUGAR_CANE], (B.SUGAR, 1)), ([B.HAY], (B.WHEAT_ITEM, 9)),
    *(([log], (planks, 4)) for log, planks in NEW_WOODS),
    ([B.MOSS, B.COBBLE], (B.MOSSY_COBBLE, 1)),
]


def _trim(grid: list[int], width: int) -> list[list[int]]:
    rows = [grid[i:i + width] for i in range(0, len(grid), width)]
    rows = [r for r in rows if any(r)] if any(grid) else []
    if not rows:
        return []
    cols = [c for c in range(width) if any(r[c] for r in rows)]
    return [r[cols[0]:cols[-1] + 1] for r in rows]


def match(grid: list[int], width: int) -> tuple[int, int] | None:
    """The (item, count) a crafting grid (item ids, 0 = empty, row by row) makes, or None."""
    shape = _trim(grid, width)
    if not shape:
        return None
    items = sorted(i for i in grid if i)
    for need, result in SHAPELESS:
        if sorted(need) == items:
            return result
    for pattern, keys, result in SHAPED:
        if len(pattern) != len(shape) or len(pattern[0]) != len(shape[0]):
            continue
        for mirror in (False, True):
            ok = True
            for r, row in enumerate(pattern):
                row = row[::-1] if mirror else row
                for c, ch in enumerate(row):
                    have = shape[r][c]
                    if ch == " ":
                        ok &= have == 0
                    else:
                        ok &= have in _key(keys[ch])
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return result
    return None


# --- furnace -----------------------------------------------------------------------------------------------------
SMELT = {B.COBBLE: B.STONE, B.SAND: B.GLASS, B.IRON_ORE: B.IRON_INGOT, B.GOLD_ORE: B.GOLD_INGOT,
         B.OAK_LOG: B.COAL, B.BIRCH_LOG: B.COAL, B.SPRUCE_LOG: B.COAL, B.PORK: B.COOKED_PORK,
         B.BEEF: B.STEAK, B.CLAY: B.BRICKS, B.FISH: B.COOKED_FISH, B.CHICKEN: B.COOKED_CHICKEN,
         B.DIAMOND_ORE: B.DIAMOND, B.COAL_ORE: B.COAL, B.REDSTONE_ORE: B.REDSTONE, B.RED_SAND: B.GLASS,
         B.RAW_RABBIT: B.COOKED_RABBIT, B.MUD: B.TERRACOTTA, **{log: B.COAL for log, _ in NEW_WOODS}}
SMELT_XP = {B.IRON_INGOT: 0.7, B.GOLD_INGOT: 1.0, B.DIAMOND: 1.0, B.STONE: 0.1, B.GLASS: 0.1, B.COOKED_PORK: 0.35,
            B.STEAK: 0.35, B.COAL: 0.15, B.COOKED_FISH: 0.35, B.COOKED_CHICKEN: 0.35, B.COOKED_RABBIT: 0.35,
            B.TERRACOTTA: 0.1}
SMELT_TIME = 10.0  # seconds per item


def fuel_time(item: int) -> float:
    """Seconds an item burns in a furnace (0: not fuel)."""
    info = B.tool_info(item)
    if info is not None and info[0] == 0:
        return 10.0
    return {B.COAL: 80.0, B.OAK_PLANKS: 15.0, B.BIRCH_PLANKS: 15.0, B.SPRUCE_PLANKS: 15.0, B.OAK_LOG: 15.0,
            B.BIRCH_LOG: 15.0, B.SPRUCE_LOG: 15.0, B.STICK: 5.0, B.CRAFTING_TABLE: 15.0,
            B.BOOKSHELF: 15.0, **{b: 15.0 for pair in NEW_WOODS for b in pair}}.get(item, 0.0)


class Furnace:
    """A furnace's contents: slots 0 input, 1 fuel, 2 output; ``burn`` seconds of fire left."""

    def __init__(self) -> None:
        self.slots: list[Stack | None] = [None, None, None]
        self.burn = 0.0
        self.burn_total = 0.0
        self.progress = 0.0
        self.xp = 0.0

    def can_smelt(self) -> bool:
        src = self.slots[0]
        if src is None or src[0] not in SMELT:
            return False
        out = self.slots[2]
        result = SMELT[src[0]]
        return out is None or (out[0] == result and out[1] < B.max_stack(result))

    def tick(self, dt: float) -> bool:
        """Advance; True when it is lit (to show the lit furnace)."""
        if self.burn > 0:
            self.burn -= dt
        if self.burn <= 0 and self.can_smelt():
            fuel = self.slots[1]
            t = fuel_time(fuel[0]) if fuel else 0.0
            if t > 0:
                self.burn = self.burn_total = t
                fuel[1] -= 1
                if fuel[1] <= 0:
                    self.slots[1] = None
        if self.burn > 0 and self.can_smelt():
            self.progress += dt
            if self.progress >= SMELT_TIME:
                self.progress = 0.0
                src = self.slots[0]
                result = SMELT[src[0]]
                src[1] -= 1
                if src[1] <= 0:
                    self.slots[0] = None
                if self.slots[2] is None:
                    self.slots[2] = stack(result, 1)
                else:
                    self.slots[2][1] += 1
                self.xp += SMELT_XP.get(result, 0.1)
        else:
            self.progress = max(0.0, self.progress - dt * 2)
        if self.burn < 0:
            self.burn = 0.0
        return self.burn > 0

    def to_json(self) -> list:
        flat = []
        for s in self.slots:
            flat += s if s else [0, 0, 0]
        return flat + [round(self.burn, 2), round(self.burn_total, 2), round(self.progress, 2)]

    def load(self, data: list) -> None:
        try:
            for k in range(3):
                item, count, dmg = (int(v) for v in data[k * 3:k * 3 + 3])
                self.slots[k] = stack(item, count, dmg) if item in B.NAMES and count > 0 else None
            self.burn, self.burn_total, self.progress = (float(v) for v in data[9:12])
        except (TypeError, ValueError, IndexError):
            pass
