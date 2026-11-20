"""Blocks and items of BlockCraft: ids, properties (hardness, tools, drops, light), and their 16x16 pixel-art
textures (a numpy atlas the renderer samples) and item sprites.

The pictures are drawn here by code. When the "Pixel Perfection" texture pack folder lies beside this file
(XSSheep's and Nova_Wostra's art, their own style, CC BY-SA 4.0: see its pack.txt), its block and item pictures
are used instead, each one that is there and 16 pixels wide; without the folder the game looks as drawn here."""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
from pathlib import Path

import numpy as np

# --- block ids ---------------------------------------------------------------------------------------------------
AIR, STONE, GRASS, DIRT, COBBLE, OAK_PLANKS, BEDROCK, WATER, LAVA, SAND, GRAVEL = range(11)
GOLD_ORE, IRON_ORE, COAL_ORE, OAK_LOG, OAK_LEAVES, GLASS, DIAMOND_ORE, REDSTONE_ORE, BRICKS, TNT = range(11, 21)
BOOKSHELF, MOSSY_COBBLE, OBSIDIAN, CRAFTING_TABLE = 21, 22, 23, 24
FURNACE = 25  # 25..28 unlit facing N/E/S/W, 29..32 lit
FURNACE_LIT = 29
BIRCH_LOG, BIRCH_LEAVES, BIRCH_PLANKS, SNOW, SANDSTONE, STONE_BRICKS, GLOWSTONE = range(33, 40)
IRON_BLOCK, GOLD_BLOCK, DIAMOND_BLOCK, TORCH, DANDELION, POPPY, TALL_GRASS, DEAD_BUSH = range(40, 48)
WOOL = 48  # 48..63: the 16 colours
SPRUCE_LOG, SPRUCE_LEAVES, SPRUCE_PLANKS, CACTUS, ICE, SNOWY_GRASS, CLAY = range(64, 71)
FERN, SUGAR_CANE, SEAGRASS, KELP, FARMLAND = range(71, 76)
WHEAT = 76  # 76..79: four stages of the crop
OAK_SAPLING, BED, PUMPKIN = 80, 81, 82
FLOW = 83  # FLOW + level (1..7): running water, lower the further it has run
CORNFLOWER, ALLIUM, DAISY, RED_MUSHROOM, BROWN_MUSHROOM, HAY = range(91, 97)
# the Deep (a world of fire under the world) and the Sky Isles (islands over nothing): BlockCraft's own realms
(DEEPSTONE, ASH, EMBER_MOSS, GLOWCAP, GLOWSTEM, DEEP_CRYSTAL, BASALT, CHARRED_LOG, STARSTONE, DEEP_GATE, SKY_GATE,
 VOIDSTONE, MOONMOSS, PALE_LOG, PALE_LEAVES, CLOUD, STORM_ALTAR) = range(97, 114)
BORDER = 255  # not loaded yet / outside the world (never placed)

# --- item ids (not blocks) ---------------------------------------------------------------------------------------
STICK, COAL, IRON_INGOT, GOLD_INGOT, DIAMOND, REDSTONE, APPLE = range(256, 263)
PORK, COOKED_PORK, BEEF, STEAK, FLINT, FLINT_STEEL, ROTTEN_FLESH = range(263, 270)
WHEAT_SEEDS, WHEAT_ITEM, BREAD, SUGAR = range(292, 296)
HOE0 = 296  # 296 + material
BUCKET, WATER_BUCKET, FISH, COOKED_FISH, FEATHER, CHICKEN, COOKED_CHICKEN, STRING, SLIME_BALL, GEM = range(301, 311)
ARMOR0 = 311  # 311 + material * 4 + piece
ARMOR_MATERIALS = ("iron", "golden", "diamond")
ARMOR_PIECES = ("helmet", "chestplate", "leggings", "boots")
ARMOR_POINTS = ((2, 6, 5, 2), (2, 5, 3, 1), (3, 8, 6, 3))  # of 20: each takes 4 % off a blow
ARMOR_USES = ((165, 240, 225, 195), (77, 112, 105, 91), (363, 528, 495, 429))
TOOL0 = 272  # 272 + material * 4 + kind
TOOL_KINDS = ("pickaxe", "axe", "shovel", "sword")
MATERIALS = ("wooden", "stone", "iron", "golden", "diamond")
TOOL_SPEED = (2.0, 4.0, 6.0, 12.0, 8.0)
TOOL_TIER = (1, 2, 3, 1, 4)  # harvest level (0 = hand)
TOOL_USES = (59, 131, 250, 32, 1561)
SWORD_DAMAGE = (4, 5, 6, 4, 7)


def tool_id(material: int, kind: int) -> int:
    return TOOL0 + material * 4 + kind


def tool_info(item: int) -> tuple[int, int] | None:
    """(material, kind) of a tool, None for anything else (a hoe is no digging tool: see ``hoe_material``)."""
    if TOOL0 <= item < TOOL0 + 20:
        return divmod(item - TOOL0, 4)
    return None


def hoe_material(item: int) -> int | None:
    return item - HOE0 if HOE0 <= item < HOE0 + 5 else None


def armor_info(item: int) -> tuple[int, int] | None:
    """(material, piece: 0 helmet .. 3 boots) of a piece of armour, None for anything else."""
    if ARMOR0 <= item < ARMOR0 + 12:
        return divmod(item - ARMOR0, 4)
    return None


WOOL_COLOURS = ((233, 236, 236), (240, 118, 19), (189, 68, 179), (58, 175, 217), (248, 198, 39), (112, 185, 25),
                (237, 141, 172), (62, 68, 71), (142, 142, 134), (21, 137, 145), (121, 42, 172), (53, 57, 157),
                (114, 71, 40), (84, 109, 27), (161, 39, 34), (20, 21, 25))
WOOL_NAMES = ("White", "Orange", "Magenta", "Light Blue", "Yellow", "Lime", "Pink", "Gray", "Light Gray", "Cyan",
              "Purple", "Blue", "Brown", "Green", "Red", "Black")

# kinds used by the renderer: 0 air, 1 opaque cube, 2 cutout cube (leaves, glass), 3 water, 4 cross sprite
K_AIR, K_OPAQUE, K_CUTOUT, K_WATER, K_CROSS = range(5)

NAMES: dict[int, str] = {
    STONE: "Stone", GRASS: "Grass Block", DIRT: "Dirt", COBBLE: "Cobblestone", OAK_PLANKS: "Oak Planks",
    BEDROCK: "Bedrock", WATER: "Water", LAVA: "Lava", SAND: "Sand", GRAVEL: "Gravel", GOLD_ORE: "Gold Ore",
    IRON_ORE: "Iron Ore", COAL_ORE: "Coal Ore", OAK_LOG: "Oak Log", OAK_LEAVES: "Oak Leaves", GLASS: "Glass",
    DIAMOND_ORE: "Diamond Ore", REDSTONE_ORE: "Redstone Ore", BRICKS: "Bricks", TNT: "TNT", BOOKSHELF: "Bookshelf",
    MOSSY_COBBLE: "Mossy Cobblestone", OBSIDIAN: "Obsidian", CRAFTING_TABLE: "Crafting Table", FURNACE: "Furnace",
    BIRCH_LOG: "Birch Log", BIRCH_LEAVES: "Birch Leaves", BIRCH_PLANKS: "Birch Planks", SNOW: "Snow Block",
    SANDSTONE: "Sandstone", STONE_BRICKS: "Stone Bricks", GLOWSTONE: "Glowstone", IRON_BLOCK: "Block of Iron",
    GOLD_BLOCK: "Block of Gold", DIAMOND_BLOCK: "Block of Diamond", TORCH: "Torch", DANDELION: "Dandelion",
    POPPY: "Poppy", TALL_GRASS: "Grass", DEAD_BUSH: "Dead Bush", SPRUCE_LOG: "Spruce Log",
    SPRUCE_LEAVES: "Spruce Leaves", SPRUCE_PLANKS: "Spruce Planks", CACTUS: "Cactus", ICE: "Ice",
    SNOWY_GRASS: "Snowy Grass Block", CLAY: "Clay", FERN: "Fern", SUGAR_CANE: "Sugar Cane", SEAGRASS: "Seagrass",
    KELP: "Kelp", FARMLAND: "Farmland", OAK_SAPLING: "Oak Sapling", BED: "Bed", PUMPKIN: "Pumpkin",
    CORNFLOWER: "Cornflower", ALLIUM: "Allium", DAISY: "Oxeye Daisy", RED_MUSHROOM: "Red Mushroom",
    BROWN_MUSHROOM: "Brown Mushroom", HAY: "Hay Bale", DEEPSTONE: "Deepstone", ASH: "Ash",
    EMBER_MOSS: "Ember Moss", GLOWCAP: "Glowcap", GLOWSTEM: "Glowcap Stem", DEEP_CRYSTAL: "Deep Crystal",
    BASALT: "Basalt", CHARRED_LOG: "Charred Log", STARSTONE: "Starstone", DEEP_GATE: "Deep Gate",
    SKY_GATE: "Sky Gate", VOIDSTONE: "Voidstone", MOONMOSS: "Moonmoss", PALE_LOG: "Pale Log",
    PALE_LEAVES: "Pale Leaves", CLOUD: "Cloud", STORM_ALTAR: "Storm Altar",
    WHEAT_SEEDS: "Wheat Seeds", WHEAT_ITEM: "Wheat",
    BREAD: "Bread", SUGAR: "Sugar", BUCKET: "Bucket", WATER_BUCKET: "Water Bucket", FISH: "Raw Fish",
    COOKED_FISH: "Cooked Fish", FEATHER: "Feather", CHICKEN: "Raw Chicken", COOKED_CHICKEN: "Cooked Chicken",
    STRING: "String", SLIME_BALL: "Slime Ball", GEM: "Trade Gem",
    STICK: "Stick", COAL: "Coal", IRON_INGOT: "Iron Ingot", GOLD_INGOT: "Gold Ingot", DIAMOND: "Diamond",
    REDSTONE: "Redstone Dust", APPLE: "Apple", PORK: "Raw Porkchop", COOKED_PORK: "Cooked Porkchop",
    BEEF: "Raw Beef", STEAK: "Steak", FLINT: "Flint", FLINT_STEEL: "Flint and Steel", ROTTEN_FLESH: "Rotten Flesh",
}
for _i in range(1, 8):
    NAMES[FURNACE + _i] = "Furnace"
for _i, _n in enumerate(WOOL_NAMES):
    NAMES[WOOL + _i] = f"{_n} Wool"
for _m, _mn in enumerate(MATERIALS):
    for _k, _kn in enumerate(TOOL_KINDS):
        NAMES[tool_id(_m, _k)] = f"{_mn.capitalize()} {_kn.capitalize()}"
    NAMES[HOE0 + _m] = f"{_mn.capitalize()} Hoe"
for _m, _mn in enumerate(ARMOR_MATERIALS):
    for _k, _kn in enumerate(ARMOR_PIECES):
        NAMES[ARMOR0 + _m * 4 + _k] = f"{_mn.capitalize()} {_kn.capitalize()}"
for _i in range(4):
    NAMES[WHEAT + _i] = "Wheat Crop"
for _i in range(1, 8):
    NAMES[FLOW + _i] = "Water"

FOOD = {APPLE: (4, 2.4), PORK: (3, 1.8), COOKED_PORK: (8, 12.8), BEEF: (3, 1.8), STEAK: (8, 12.8),
        ROTTEN_FLESH: (4, 0.8), BREAD: (5, 6.0), FISH: (2, 0.4), COOKED_FISH: (5, 6.0), CHICKEN: (2, 1.2),
        COOKED_CHICKEN: (6, 7.2)}

# --- block properties --------------------------------------------------------------------------------------------
KIND = np.zeros(256, np.uint8)
HARDNESS = np.zeros(256, np.float32)
TOOL = np.full(256, -1, np.int8)  # preferred tool kind (0 pickaxe, 1 axe, 2 shovel)
TIER = np.zeros(256, np.int8)  # harvest level needed for a drop (0: anything, 1: wooden pickaxe ...)
EMIT = np.zeros(256, np.uint8)  # block light
MATERIAL: dict[int, str] = {}  # sound material

_SOLID = {STONE: (1.5, 0, 1, "stone"), GRASS: (0.6, 2, 0, "grass"), DIRT: (0.5, 2, 0, "gravel"),
          COBBLE: (2.0, 0, 1, "stone"), OAK_PLANKS: (2.0, 1, 0, "wood"), BEDROCK: (-1, 0, 9, "stone"),
          SAND: (0.5, 2, 0, "sand"), GRAVEL: (0.6, 2, 0, "gravel"), GOLD_ORE: (3.0, 0, 3, "stone"),
          IRON_ORE: (3.0, 0, 2, "stone"), COAL_ORE: (3.0, 0, 1, "stone"), OAK_LOG: (2.0, 1, 0, "wood"),
          DIAMOND_ORE: (3.0, 0, 3, "stone"), REDSTONE_ORE: (3.0, 0, 3, "stone"), BRICKS: (2.0, 0, 1, "stone"),
          TNT: (0.0, -1, 0, "grass"), BOOKSHELF: (1.5, 1, 0, "wood"), MOSSY_COBBLE: (2.0, 0, 1, "stone"),
          OBSIDIAN: (50.0, 0, 4, "stone"), CRAFTING_TABLE: (2.5, 1, 0, "wood"), BIRCH_LOG: (2.0, 1, 0, "wood"),
          BIRCH_PLANKS: (2.0, 1, 0, "wood"), SNOW: (0.2, 2, 0, "snow"), SANDSTONE: (0.8, 0, 1, "stone"),
          STONE_BRICKS: (1.5, 0, 1, "stone"), GLOWSTONE: (0.3, -1, 0, "glass"), IRON_BLOCK: (5.0, 0, 2, "metal"),
          GOLD_BLOCK: (3.0, 0, 3, "metal"), DIAMOND_BLOCK: (5.0, 0, 3, "metal"), LAVA: (-1, -1, 9, "stone"),
          SPRUCE_LOG: (2.0, 1, 0, "wood"), SPRUCE_PLANKS: (2.0, 1, 0, "wood"), CACTUS: (0.4, -1, 0, "cloth"),
          ICE: (0.5, 0, 0, "glass"), SNOWY_GRASS: (0.6, 2, 0, "snow"), CLAY: (0.6, 2, 0, "gravel"),
          FARMLAND: (0.6, 2, 0, "gravel"), BED: (0.2, -1, 0, "cloth"), PUMPKIN: (1.0, 1, 0, "wood"),
          HAY: (0.5, -1, 0, "grass"), DEEPSTONE: (2.0, 0, 1, "stone"), ASH: (0.5, 2, 0, "sand"),
          EMBER_MOSS: (0.6, 2, 0, "grass"), GLOWCAP: (0.4, -1, 0, "cloth"), GLOWSTEM: (2.0, 1, 0, "wood"),
          DEEP_CRYSTAL: (1.5, 0, 1, "glass"), BASALT: (1.8, 0, 1, "stone"), CHARRED_LOG: (1.5, 1, 0, "wood"),
          STARSTONE: (4.0, 0, 2, "stone"), DEEP_GATE: (8.0, 0, 2, "stone"), SKY_GATE: (8.0, 0, 2, "stone"),
          VOIDSTONE: (2.2, 0, 1, "stone"), MOONMOSS: (0.6, 2, 0, "grass"), PALE_LOG: (2.0, 1, 0, "wood"),
          CLOUD: (0.2, -1, 0, "cloth"), STORM_ALTAR: (30.0, 0, 3, "stone")}
for _b, (_h, _t, _tier, _m) in _SOLID.items():
    KIND[_b], HARDNESS[_b], TOOL[_b], TIER[_b], MATERIAL[_b] = K_OPAQUE, _h, _t, _tier, _m
for _b in range(FURNACE, FURNACE + 8):
    KIND[_b], HARDNESS[_b], TOOL[_b], TIER[_b], MATERIAL[_b] = K_OPAQUE, 3.5, 0, 1, "stone"
for _b in range(WOOL, WOOL + 16):
    KIND[_b], HARDNESS[_b], MATERIAL[_b] = K_OPAQUE, 0.8, "cloth"
LEAVES = (OAK_LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES)
for _b in LEAVES:
    KIND[_b], HARDNESS[_b], MATERIAL[_b] = K_CUTOUT, 0.2, "grass"
KIND[GLASS], HARDNESS[GLASS], MATERIAL[GLASS] = K_CUTOUT, 0.3, "glass"
KIND[PALE_LEAVES], HARDNESS[PALE_LEAVES], MATERIAL[PALE_LEAVES] = K_CUTOUT, 0.2, "grass"
KIND[WATER], HARDNESS[WATER] = K_WATER, -1
FLOWERS = (DANDELION, POPPY, CORNFLOWER, ALLIUM, DAISY)
CROPS = (WHEAT, WHEAT + 1, WHEAT + 2, WHEAT + 3)
for _b in (TORCH, TALL_GRASS, DEAD_BUSH, FERN, SUGAR_CANE, SEAGRASS, KELP, OAK_SAPLING, RED_MUSHROOM, BROWN_MUSHROOM,
           *FLOWERS, *CROPS):
    KIND[_b], HARDNESS[_b], MATERIAL[_b] = K_CROSS, 0.0, "grass"
for _i in range(1, 8):
    KIND[FLOW + _i], HARDNESS[FLOW + _i] = K_WATER, -1
# water in all its forms: the still block, the running levels, and the plants that grow inside it
FLUID = np.zeros(256, bool)  # still or running water
FLUID[WATER] = True
FLUID[FLOW + 1:FLOW + 8] = True
WATERY = FLUID.copy()  # ... and what is under water with it (swimming, drowning, the view from inside)
WATERY[[SEAGRASS, KELP]] = True
WATERY_SET = frozenset(int(v) for v in np.flatnonzero(WATERY))
LEVEL = np.zeros(256, np.int8)  # how much water: 8 a still block, 7..1 running
LEVEL[WATER] = 8
LEVEL[FLOW + 1:FLOW + 8] = np.arange(1, 8)
MATERIAL[TORCH] = "wood"
EMIT[TORCH], EMIT[GLOWSTONE], EMIT[LAVA] = 14, 15, 15
EMIT[[GLOWCAP, DEEP_CRYSTAL, STARSTONE, DEEP_GATE, SKY_GATE, STORM_ALTAR, PALE_LEAVES]] = 12, 10, 8, 11, 13, 9, 4
GLOWING = (GLOWCAP, DEEP_CRYSTAL, DEEP_GATE, SKY_GATE)  # drawn at full brightness
EMIT[FURNACE_LIT:FURNACE_LIT + 4] = 13
KIND[BORDER] = K_OPAQUE

OPAQUE = KIND == K_OPAQUE  # blocks light and the player's view
SOLID = (KIND == K_OPAQUE) | (KIND == K_CUTOUT)  # collides
SOLID[LAVA] = False
LIGHT_ATT = np.zeros(256, np.uint8)  # extra light loss passing through
LIGHT_ATT[[OAK_LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES, PALE_LEAVES]] = 1
LIGHT_ATT[WATERY] = 2
GRAVITY = {SAND, GRAVEL}
PLACEABLE = set(NAMES) - {WATER, LAVA, *CROPS} - set(range(FLOW, FLOW + 8)) - set(range(256, 512))
CREATIVE_ITEMS = ([GRASS, DIRT, STONE, COBBLE, MOSSY_COBBLE, STONE_BRICKS, BRICKS, OAK_LOG, OAK_PLANKS, BIRCH_LOG,
                   BIRCH_PLANKS, SPRUCE_LOG, SPRUCE_PLANKS, OAK_LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES, SAND, SANDSTONE,
                   GRAVEL, CLAY, GLASS, SNOW, SNOWY_GRASS, ICE, CACTUS, BEDROCK, OBSIDIAN,
                   COAL_ORE, IRON_ORE, GOLD_ORE, REDSTONE_ORE, DIAMOND_ORE, IRON_BLOCK, GOLD_BLOCK, DIAMOND_BLOCK,
                   GLOWSTONE, BOOKSHELF, CRAFTING_TABLE, FURNACE, TNT, TORCH, BED, HAY, PUMPKIN, FARMLAND, DANDELION,
                   POPPY, CORNFLOWER, ALLIUM, DAISY, TALL_GRASS, FERN, DEAD_BUSH, OAK_SAPLING, SUGAR_CANE,
                   RED_MUSHROOM, BROWN_MUSHROOM, SEAGRASS, KELP, DEEP_GATE, SKY_GATE, DEEPSTONE, BASALT, ASH,
                   EMBER_MOSS, GLOWCAP, GLOWSTEM, CHARRED_LOG, DEEP_CRYSTAL, STARSTONE, VOIDSTONE, MOONMOSS,
                   PALE_LOG, PALE_LEAVES, CLOUD]
                  + [WOOL + i for i in range(16)]
                  + [tool_id(m, k) for m in range(5) for k in range(4)]
                  + [HOE0 + m for m in range(5)] + [ARMOR0 + i for i in range(12)]
                  + [BUCKET, WATER_BUCKET, STICK, COAL, IRON_INGOT, GOLD_INGOT, DIAMOND, REDSTONE, GEM, FLINT,
                     FLINT_STEEL, STRING, FEATHER, SLIME_BALL, WHEAT_SEEDS, WHEAT_ITEM, SUGAR, APPLE, BREAD, PORK,
                     COOKED_PORK, BEEF, STEAK, CHICKEN, COOKED_CHICKEN, FISH, COOKED_FISH, ROTTEN_FLESH])


def is_block(item: int) -> bool:
    return 0 < item < 256


def max_stack(item: int) -> int:
    if tool_info(item) is not None or hoe_material(item) is not None or armor_info(item) is not None or \
            item in (FLINT_STEEL, WATER_BUCKET, BED):
        return 1
    return 16 if item == BUCKET else 64


def max_damage(item: int) -> int:
    info = tool_info(item)
    if info is not None:
        return TOOL_USES[info[0]]
    hoe = hoe_material(item)
    if hoe is not None:
        return TOOL_USES[hoe]
    arm = armor_info(item)
    if arm is not None:
        return ARMOR_USES[arm[0]][arm[1]]
    return 64 if item == FLINT_STEEL else 0


def base_block(b: int) -> int:
    """The item a block is (furnace facings -> furnace)."""
    return FURNACE if FURNACE <= b < FURNACE + 8 else b


def drops(b: int, tool: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """(item, count) a block drops when broken with ``tool`` (an item id or 0) in survival."""
    if HARDNESS[b] < 0:
        return []
    need = TIER[b]
    info = tool_info(tool)
    if need:
        if info is None or info[1] != 0 or TOOL_TIER[info[0]] < need:
            return []
    b = base_block(b)
    table = {STONE: COBBLE, GRASS: DIRT, COAL_ORE: COAL, DIAMOND_ORE: DIAMOND, GLOWSTONE: GLOWSTONE,
             BOOKSHELF: OAK_PLANKS, SNOWY_GRASS: DIRT, FARMLAND: DIRT}
    if FLUID[b] or b == SEAGRASS:
        return []
    if b in (TALL_GRASS, FERN):
        return [(WHEAT_SEEDS, 1)] if rng.random() < 0.14 else []
    if b in CROPS:
        if b == WHEAT + 3:
            return [(WHEAT_ITEM, 1), (WHEAT_SEEDS, int(rng.integers(1, 4)))]
        return [(WHEAT_SEEDS, 1)]
    if b == PALE_LEAVES:
        return []
    if b in (GLASS, DEAD_BUSH, ICE):
        return [(STICK, 1)] if b == DEAD_BUSH and rng.random() < 0.5 else []
    if b in LEAVES:
        if b == OAK_LEAVES and rng.random() < 0.07:
            return [(OAK_SAPLING, 1)]
        if rng.random() < 0.05:
            return [(STICK, int(rng.integers(1, 3)))]
        return [(APPLE, 1)] if b == OAK_LEAVES and rng.random() < 0.04 else []
    if b == REDSTONE_ORE:
        return [(REDSTONE, int(rng.integers(4, 6)))]
    if b == GRAVEL and rng.random() < 0.1:
        return [(FLINT, 1)]
    if b == BOOKSHELF:
        return [(OAK_PLANKS, 3)]
    return [(table.get(b, b), 1)]


def dig_time(b: int, tool: int, on_ground: bool = True, in_water: bool = False) -> float:
    """Seconds to break a block (the Java Edition formula); inf for unbreakable."""
    h = float(HARDNESS[b])
    if h < 0:
        return float("inf")
    if h == 0:
        return 0.0
    info = tool_info(tool)
    speed = 1.0
    if info is not None and info[1] == TOOL[b]:
        speed = TOOL_SPEED[info[0]]
    elif info is not None and info[1] == 3 and b in LEAVES:
        speed = 1.5
    can = TIER[b] == 0 or (info is not None and info[1] == 0 and TOOL_TIER[info[0]] >= TIER[b])
    t = h * (1.5 if can else 5.0) / speed
    if in_water:
        t *= 5
    if not on_ground:
        t *= 5
    return math.ceil(t * 20 - 1e-6) / 20  # whole ticks


# --- textures ----------------------------------------------------------------------------------------------------
TEXTURES: list[str] = []
_TEX: dict[str, np.ndarray] = {}


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _pick(rng: np.random.Generator, colours: list, weights: list | None = None, shape=(16, 16)) -> np.ndarray:
    cols = np.array([_hex(c) if isinstance(c, str) else c for c in colours], np.float32)
    w = np.array(weights if weights else [1] * len(cols), np.float64)
    idx = rng.choice(len(cols), size=shape, p=w / w.sum())
    return cols[idx]


def _smooth_noise(rng: np.random.Generator, n: int = 16, blur: int = 1) -> np.ndarray:
    a = rng.random((n, n))
    for _ in range(blur):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5
    a -= a.min()
    return a / max(a.max(), 1e-6)


def _rgba(rgb: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    out = np.empty((16, 16, 4), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255)
    out[..., 3] = 255 if alpha is None else alpha
    return out


def _add(name: str, img: np.ndarray) -> None:
    _TEX[name] = img if img.shape[-1] == 4 else _rgba(img)
    TEXTURES.append(name)


def _dirt(rng) -> np.ndarray:
    return _pick(rng, ["866043", "966c4a", "79553a", "593d29", "b9855c"], [34, 26, 26, 7, 7])


def _grass_top(rng) -> np.ndarray:
    return _pick(rng, ["5f9b35", "6aa83c", "77b545", "528a2e", "84c24f", "4a7d29"], [24, 24, 18, 14, 10, 10])


def _stone(rng) -> np.ndarray:
    n = _smooth_noise(rng, blur=1) * 0.6 + rng.random((16, 16)) * 0.4
    levels = np.array([0x6c, 0x76, 0x7f, 0x88, 0x91], np.float32)
    g = levels[np.clip((n * 5).astype(int), 0, 4)]
    return np.repeat(g[..., None], 3, -1)


def _cobble(rng, mossy: bool = False) -> np.ndarray:
    pts = rng.random((11, 2)) * 16
    yy, xx = np.mgrid[0:16, 0:16] + 0.5
    best = np.full((16, 16), 99.0)
    second = np.full((16, 16), 99.0)
    cell = np.zeros((16, 16), int)
    for i, (py, px) in enumerate(pts):
        for oy in (-16, 0, 16):
            for ox in (-16, 0, 16):
                d = np.hypot(yy - py - oy, xx - px - ox)
                closer = d < best
                second = np.where(closer, best, np.minimum(second, d))
                cell = np.where(closer, i, cell)
                best = np.where(closer, d, best)
    edge = (second - best) < 1.3
    shades = rng.uniform(0.85, 1.12, 11)[cell]
    base = 0x7f * shades - best * 3.5 + rng.random((16, 16)) * 12
    g = np.where(edge, 0x4a + rng.random((16, 16)) * 12, base + 18)
    rgb = np.repeat(g[..., None], 3, -1)
    if mossy:
        moss = (_smooth_noise(rng, blur=2) > 0.55) & ~edge
        rgb[moss] = _pick(rng, ["4d7a32", "5a8a3a", "3f6a2a"])[moss]
    return rgb


def _planks(rng, light: str, mid: str, dark: str, line: str) -> np.ndarray:
    rgb = _pick(rng, [light, mid, mid, dark], [3, 4, 4, 2])
    for y in range(16):  # grain: horizontal streaks
        if rng.random() < 0.5:
            x0 = rng.integers(0, 12)
            rgb[y, x0:x0 + rng.integers(3, 8)] = _hex(dark)
    for k, y in enumerate((3, 7, 11, 15)):
        rgb[y, :] = _hex(line)
        seam = (4 + k * 7) % 16
        rgb[y - 3:y, seam] = _hex(line)
    return rgb


def _log_side(rng, birch: bool, cols: list | None = None, streak: str = "3f2e1a") -> np.ndarray:
    if birch:
        rgb = _pick(rng, ["d7d3c8", "e3dfd5", "c9c4b9", "f0ece2"], [4, 4, 2, 2])
        for _ in range(9):
            y, x = rng.integers(0, 16), rng.integers(0, 14)
            rgb[y, x:x + rng.integers(2, 4)] = _hex("2f2b26" if rng.random() < 0.6 else "5c574f")
        return rgb
    rgb = np.zeros((16, 16, 3), np.float32)
    cols = _pick(rng, cols or ["6b5132", "5b4429", "7a5d3a", "4b3820"], [4, 3, 2, 2], shape=(16,))
    for x in range(16):
        rgb[:, x] = cols[x]
        dark = rng.random(16) < 0.15
        rgb[dark, x] = _hex(streak)
    return rgb


def _log_top(rng, birch: bool) -> np.ndarray:
    yy, xx = np.mgrid[0:16, 0:16]
    ring = np.maximum(np.abs(xx - 7.5), np.abs(yy - 7.5))
    inner = ["c8aa72", "b8945f", "a7844f"] if not birch else ["d9c99a", "cdbb88", "bfae7c"]
    cols = np.array([_hex(c) for c in inner], np.float32)
    rgb = cols[(ring.astype(int) // 2) % 3] + rng.normal(0, 4, (16, 16, 1))
    bark = _log_side(rng, birch)
    border = ring >= 7
    rgb[border] = bark[border]
    return rgb


def _leaves(rng, colours: list) -> np.ndarray:
    rgb = _pick(rng, colours, [3, 4, 3, 2])
    holes = rng.random((16, 16)) < 0.22
    alpha = np.where(holes, 0, 255)
    return _rgba(rgb, alpha)


def _ore(rng, base: np.ndarray, colours: list, blobs: int = 5) -> np.ndarray:
    rgb = base.copy()
    for _ in range(blobs):
        cy, cx = rng.integers(2, 14, 2)
        shape = [(0, 0), (0, 1), (1, 0), (1, 1), (-1, 0), (0, -1)]
        for dy, dx in shape:
            if rng.random() < 0.8:
                c = colours[rng.integers(0, len(colours))]
                rgb[(cy + dy) % 16, (cx + dx) % 16] = _hex(c)
        rgb[(cy + 1) % 16, (cx + 1) % 16] = _hex(colours[-1])  # a dark edge
    return rgb


def _bricks() -> np.ndarray:
    rng = np.random.default_rng(19)
    rgb = _pick(rng, ["965a48", "a3654f", "8a4f3e", "b07560"], [4, 3, 3, 1])
    mortar = np.array(_hex("a59f96"), np.float32)
    for y in range(16):
        if y % 4 == 3:
            rgb[y] = mortar
        else:
            off = 0 if (y // 4) % 2 == 0 else 4
            rgb[y, (off + 7) % 16] = mortar
            rgb[y, (off + 15) % 16] = mortar
    return rgb


def _sand(rng) -> np.ndarray:
    return _pick(rng, ["dbd3a0", "e3dbb0", "d1c78f", "c9be87", "e8e1bb"], [4, 3, 3, 1, 1])


def _gravel(rng) -> np.ndarray:
    rgb = _pick(rng, ["7f7c7b", "6b6766", "918c8b", "5a5554", "a19c9b", "86766d"], [3, 3, 3, 2, 1, 1])
    for _ in range(8):  # pebbles
        y, x = rng.integers(0, 15, 2)
        c = np.array(_hex(["9a9796", "625e5c", "8e8279"][rng.integers(0, 3)]), np.float32)
        rgb[y:y + 2, x:x + 2] = c
        rgb[y + 1, x + 1] = c * 0.7
    return rgb


def _bedrock(rng) -> np.ndarray:
    n = _smooth_noise(rng, blur=1)
    levels = np.array([0x1e, 0x3a, 0x55, 0x6e, 0x8a], np.float32)
    g = levels[np.clip((n * 5).astype(int), 0, 4)]
    return np.repeat(g[..., None], 3, -1)


def _glass() -> np.ndarray:
    rgb = np.full((16, 16, 3), 220, np.float32)
    rgb[..., 2] = 235
    alpha = np.zeros((16, 16), np.uint8)
    alpha[0, :] = alpha[15, :] = alpha[:, 0] = alpha[:, 15] = 255
    rgb[15, :] = rgb[:, 15] = (150, 170, 180)
    for i in range(3):  # glints
        alpha[3 + i, 4 - i] = 200
        alpha[6 + i, 4 - i] = 200
        alpha[8 + i, 12 - i] = 200
    return _rgba(rgb, alpha)


def _water(frame: int) -> np.ndarray:
    """One of 8 frames of rippling water: whole waves across the tile, so neighbouring blocks join without a seam
    and the eighth frame runs into the first."""
    yy, xx = np.mgrid[0:16, 0:16]
    k = 2 * np.pi / 16
    ph = frame / 8 * 2 * np.pi
    w = np.sin((xx + yy) * k + ph) + np.sin((2 * xx - yy) * k - ph) * 0.6 + np.sin((xx - 3 * yy) * k + 2 * ph) * 0.3
    w = (w + 1.9) / 3.8
    base = np.array(_hex("2f5fd8"), np.float32)
    hi = np.array(_hex("4f82ea"), np.float32)
    rgb = base + (hi - base) * (np.round(np.clip(w, 0, 1) * 4) / 4)[..., None]
    return rgb


def _lava(frame: int) -> np.ndarray:
    yy, xx = np.mgrid[0:16, 0:16]
    k = 2 * np.pi / 16
    ph = frame / 8 * 2 * np.pi
    w = np.sin((2 * xx + yy) * k + ph) + np.sin((2 * yy - xx) * k - ph)
    w = (w + 2) / 4
    lo = np.array(_hex("c23c08"), np.float32)
    mid = np.array(_hex("f07d14"), np.float32)
    hi = np.array(_hex("ffd34a"), np.float32)
    rgb = np.where(w[..., None] < 0.55, lo + (mid - lo) * (w[..., None] / 0.55),
                   mid + (hi - mid) * ((w[..., None] - 0.55) / 0.45))
    return rgb


def _tnt_side() -> np.ndarray:
    rng = np.random.default_rng(5)
    rgb = _pick(rng, ["db2f1f", "c42a1b", "e33a28"], [3, 2, 2])
    rgb[:, ::4] *= 0.8
    rgb[4:12, :] = _pick(rng, ["e7e3dd", "d9d5cd"], [3, 1], shape=(8, 16))
    letters = ["###.#..#.###", ".#..##.#..#.", ".#..#.##..#.", ".#..#..#..#."]
    for r, row in enumerate(letters):
        for c, ch in enumerate(row):
            if ch == "#":
                rgb[6 + r, 2 + c] = (30, 30, 30)
    return rgb


def _crafting_side(rng, front: bool) -> np.ndarray:
    rgb = _planks(rng, "b8945f", "a2834f", "8a6e40", "6e5634")
    dark = np.array(_hex("4e3a22"), np.float32)
    rgb[0:2, :] = _hex("6b5132")
    if front:  # a saw
        rgb[5:7, 3:12] = (175, 175, 175)
        rgb[7, 4:12:2] = (120, 120, 120)
        rgb[5:7, 12:14] = dark
    else:  # a hammer and a pair of shears
        rgb[4:7, 3:7] = (140, 140, 140)
        rgb[7:13, 4:6] = dark
        rgb[5:12, 10] = (150, 150, 150)
        rgb[5:12, 12] = (150, 150, 150)
    return rgb


def _crafting_top(rng) -> np.ndarray:
    rgb = _planks(rng, "b8945f", "a2834f", "8a6e40", "6e5634")
    rgb[0, :] = rgb[15, :] = rgb[:, 0] = rgb[:, 15] = _hex("6b5132")
    for k in (5, 10):
        rgb[2:14, k] = _hex("5b4429")
        rgb[k, 2:14] = _hex("5b4429")
    return rgb


def _furnace_side(rng) -> np.ndarray:
    rgb = _stone(rng) * 0.95
    rgb[0, :] = rgb[:, 0] = 0x8a
    rgb[15, :] = rgb[:, 15] = 0x55
    return rgb


def _furnace_front(rng, lit: bool) -> np.ndarray:
    rgb = _furnace_side(rng)
    rgb[3:5, 3:13] = 0x45
    rgb[8:13, 4:12] = 0x20
    if lit:
        fire = _pick(rng, ["ffb02e", "ff7a12", "ffdd55", "e85a0c"], shape=(3, 8))
        rgb[10:13, 4:12] = fire
    for x in range(4, 12, 2):
        rgb[8:13, x] = 0x50
    return rgb


def _bookshelf(rng) -> np.ndarray:
    rgb = _planks(rng, "b8945f", "a2834f", "8a6e40", "6e5634")
    cols = ["a52a2a", "2a52a5", "2a8a3a", "c8a032", "6a2a8a", "d0d0d0"]
    for shelf in (1, 9):
        x = 1
        while x < 15:
            w = 1 if rng.random() < 0.7 else 2
            h = int(rng.integers(4, 7))
            c = np.array(_hex(cols[rng.integers(0, len(cols))]), np.float32)
            rgb[shelf + 6 - h:shelf + 6, x:x + w] = c
            rgb[shelf + 6 - h, x:x + w] = c * 1.2
            x += w
        rgb[shelf + 6, :] = _hex("5b4429")
    rgb[0, :] = rgb[15, :] = _hex("5b4429")
    return rgb


def _flower(head: list, centre: str | None = None, petal_shape: str = "dandelion") -> np.ndarray:
    rgb = np.zeros((16, 16, 3), np.float32)
    alpha = np.zeros((16, 16), np.uint8)
    stem = np.array(_hex("3f7a20"), np.float32)
    rgb[8:16, 7] = stem
    alpha[8:16, 7] = 255
    rgb[11, 8] = rgb[12, 6] = stem
    alpha[11, 8] = alpha[12, 6] = 255
    if petal_shape == "dandelion":
        pts = [(5, 6), (5, 7), (5, 8), (6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (4, 7), (6, 5), (6, 9)]
    else:
        pts = [(3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (5, 5), (5, 6), (5, 7), (5, 8),
               (5, 9), (6, 6), (6, 7), (6, 8), (7, 7)]
    for i, (y, x) in enumerate(pts):
        rgb[y, x] = _hex(head[i % len(head)])
        alpha[y, x] = 255
    if centre:
        rgb[5, 7] = _hex(centre)
    return _rgba(rgb, alpha)


def _tall_grass(rng, dead: bool = False) -> np.ndarray:
    rgb = np.zeros((16, 16, 3), np.float32)
    alpha = np.zeros((16, 16), np.uint8)
    cols = ["6aa83c", "5f9b35", "77b545", "4a7d29"] if not dead else ["8a6a3a", "6b5030", "9a7a48"]
    for _ in range(9 if not dead else 6):
        x = int(rng.integers(1, 15))
        top = int(rng.integers(3 if not dead else 5, 11))
        lean = rng.choice([-1, 0, 1])
        for y in range(top, 16):
            xx = int(np.clip(x + lean * ((16 - y) // 5), 0, 15))
            rgb[y, xx] = _hex(cols[rng.integers(0, len(cols))])
            alpha[y, xx] = 255
    return _rgba(rgb, alpha)


def _torch() -> np.ndarray:
    rgb = np.zeros((16, 16, 3), np.float32)
    alpha = np.zeros((16, 16), np.uint8)
    rgb[6:16, 7:9] = _hex("6b5132")
    rgb[6:16, 8] = _hex("4b3820")
    rgb[6:8, 7:9] = _hex("ffffaa")
    rgb[5, 7:9] = _hex("ffd34a")
    rgb[4, 7] = _hex("ff9a1a")
    alpha[4:16, 7:9] = 255
    alpha[4, 8] = 0
    return _rgba(rgb, alpha)


def _wool(rng, colour: tuple) -> np.ndarray:
    n = _smooth_noise(rng, blur=1) * 0.12 + rng.random((16, 16)) * 0.08 + 0.85
    yy, xx = np.mgrid[0:16, 0:16]
    weave = 1 + 0.05 * (((xx + yy) % 4 == 0) & (yy % 2 == 0))
    return np.array(colour, np.float32) * (n * weave)[..., None]


def _metal_block(rng, light: str, mid: str, dark: str) -> np.ndarray:
    rgb = np.empty((16, 16, 3), np.float32)
    rgb[:] = _hex(mid)
    rgb[1:3, 1:15] = _hex(light)
    rgb[1:15, 1:3] = _hex(light)
    rgb[13:15, 1:15] = _hex(dark)
    rgb[1:15, 13:15] = _hex(dark)
    rgb[0, :] = rgb[:, 0] = _hex(light)
    rgb[15, :] = rgb[:, 15] = _hex(dark)
    return rgb + rng.normal(0, 3, (16, 16, 1))


def _build_textures() -> None:
    def r(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    stone = _stone(r(1))
    dirt = _dirt(r(2))
    _add("stone", stone)
    _add("dirt", dirt)
    top = _grass_top(r(3))
    _add("grass_top", top)
    side = dirt.copy()
    rng = r(4)
    fringe = _grass_top(rng)
    for x in range(16):
        depth = 3 + (rng.random() < 0.55) + (rng.random() < 0.2)
        side[:depth, x] = fringe[:depth, x]
    _add("grass_side", side)
    _add("cobblestone", _cobble(r(5)))
    _add("mossy_cobblestone", _cobble(r(5), True))
    _add("oak_planks", _planks(r(6), "b8945f", "a2834f", "8a6e40", "6e5634"))
    _add("birch_planks", _planks(r(7), "d7c98f", "c8b77a", "b5a46a", "9a8a56"))
    _add("bedrock", _bedrock(r(8)))
    for f in range(8):
        _add(f"water{f}", _water(f))
        _add(f"lava{f}", _lava(f))
    _add("sand", _sand(r(9)))
    _add("gravel", _gravel(r(10)))
    _add("gold_ore", _ore(r(11), stone, ["fcee4b", "ffffb5", "fcee4b", "b88a15"]))
    _add("iron_ore", _ore(r(12), stone, ["d8af93", "e3c0a7", "c79a7c", "8f6a52"]))
    _add("coal_ore", _ore(r(13), stone, ["2b2b2b", "3a3a3a", "4a4a4a", "141414"], 6))
    _add("diamond_ore", _ore(r(14), stone, ["5decf5", "c2fcf5", "4ad5de", "1f8f8f"]))
    _add("redstone_ore", _ore(r(15), stone, ["ff1a1a", "c40000", "ff5a5a", "780000"], 6))
    _add("oak_log", _log_side(r(16), False))
    _add("oak_log_top", _log_top(r(16), False))
    _add("birch_log", _log_side(r(17), True))
    _add("birch_log_top", _log_top(r(17), True))
    _add("oak_leaves", _leaves(r(18), ["3f7a1f", "4a8a26", "356a19", "5a9a32"]))
    _add("birch_leaves", _leaves(r(19), ["6d9a44", "80a755", "5e8a3a", "93b866"]))
    spruce = ["3d2a18", "33230f", "47311c", "281a0b"]
    _add("spruce_log", _log_side(r(50), False, spruce, "1d1206"))
    st = _log_top(r(50), False)
    yy, xx = np.mgrid[0:16, 0:16]
    ring = np.maximum(np.abs(xx - 7.5), np.abs(yy - 7.5)) >= 7
    st[ring] = _log_side(r(50), False, spruce, "1d1206")[ring]
    st[~ring] *= 0.72
    _add("spruce_log_top", st)
    _add("spruce_leaves", _leaves(r(51), ["2f5a35", "376640", "264d2c", "3f7048"]))
    _add("spruce_planks", _planks(r(52), "85633a", "74552f", "644826", "4e371c"))
    cs = _pick(r(53), ["4f8f2f", "5a9c36", "47822a", "62a83e"], [3, 3, 2, 2])
    cs[:, ::4] = _pick(r(54), ["3a6e22", "356620"], shape=(16, 4))
    cs[r(55).random((16, 16)) < 0.07] = (222, 226, 190)  # spines
    _add("cactus_side", cs)
    ct = _pick(r(56), ["5a9c36", "62a83e", "4f8f2f"], [3, 2, 2])
    ct[0, :] = ct[15, :] = ct[:, 0] = ct[:, 15] = _hex("3a6e22")
    ct[6:10, 6:10] = _pick(r(57), ["7cc04c", "8ccc5a"], shape=(4, 4))
    _add("cactus_top", ct)
    ice = _pick(r(58), ["8fb8f0", "9cc2f5", "86aee8", "a9cdf8"], [4, 3, 2, 2])
    for k in range(5):
        y, x = r(59 + k).integers(1, 12, 2)
        for i in range(4):
            ice[y + i, x + i] = (208, 230, 255)
    _add("ice", ice)
    _add("clay", _pick(r(66), ["9da3ae", "a5abb6", "959ba6", "aeb4be"], [4, 3, 3, 1]))
    snow_side = dirt.copy()
    rs = r(67)
    white = _pick(rs, ["f0fbfb", "e6f2f2", "ffffff"], [4, 3, 2])
    for x in range(16):
        depth = 3 + (rs.random() < 0.5) + (rs.random() < 0.2)
        snow_side[:depth, x] = white[:depth, x]
    _add("snowy_grass_side", snow_side)
    _add("glass", _glass())
    _add("bricks", _bricks())
    _add("tnt_side", _tnt_side())
    t = _pick(r(20), ["db2f1f", "c42a1b"], [3, 2])
    t[6:10, 6:10] = (40, 40, 40)
    _add("tnt_top", t)
    _add("tnt_bottom", _pick(r(21), ["db2f1f", "c42a1b"], [3, 2]))
    _add("bookshelf", _bookshelf(r(22)))
    _add("obsidian", _pick(r(23), ["14121d", "1d1730", "2a2140", "0c0b10", "3b2d5a"], [4, 3, 2, 2, 1]))
    _add("crafting_table_top", _crafting_top(r(24)))
    _add("crafting_table_front", _crafting_side(r(25), True))
    _add("crafting_table_side", _crafting_side(r(26), False))
    _add("furnace_side", _furnace_side(r(27)))
    _add("furnace_top", _stone(r(28)) * 0.95)
    _add("furnace_front", _furnace_front(r(27), False))
    _add("furnace_front_lit", _furnace_front(r(27), True))
    _add("snow", _pick(r(29), ["f0fbfb", "e6f2f2", "ffffff", "d9e8ec"], [4, 3, 2, 1]))
    ss = _pick(r(30), ["e0d6a6", "d8cc98", "e8dfb2"], [3, 3, 2])
    ss[12:, :] *= 0.92
    ss[0:2, :] *= 1.03
    _add("sandstone", ss)
    _add("sandstone_top", _pick(r(31), ["e0d6a6", "d8cc98", "e8dfb2"], [3, 3, 2]))
    sb = _stone(r(32))
    for y in (7, 15):
        sb[y, :] = 0x5a
    sb[0:7, 7] = sb[8:15, 15] = 0x5a
    sb[0, :] = sb[8, :] = sb[:, 0] = 0x9a
    _add("stone_bricks", sb)
    _add("glowstone", _pick(r(33), ["ffd98a", "e0a040", "fff0b5", "b8803a", "ffc860"], [3, 3, 2, 2, 2]))
    _add("iron_block", _metal_block(r(34), "ffffff", "dcdcdc", "a8a8a8"))
    _add("gold_block", _metal_block(r(35), "fffbb5", "f8d73a", "c59a14"))
    _add("diamond_block", _metal_block(r(36), "d4fffb", "62e8de", "2aa69a"))
    _add("torch", _torch())
    _add("dandelion", _flower(["ffe419", "f5c515", "ffee66"], "f5a000", "dandelion"))
    _add("poppy", _flower(["e4141b", "c40d12", "ff3a3a"], "2b1a0a", "poppy"))
    _add("tall_grass", _tall_grass(r(37)))
    _add("dead_bush", _tall_grass(r(38), dead=True))
    for i, c in enumerate(WOOL_COLOURS):
        _add(f"wool{i}", _wool(r(40), c))
    # the "inside a block" / missing texture
    _add("missing", np.where((np.add.outer(np.arange(16) // 8, np.arange(16) // 8) % 2)[..., None] == 0,
                             np.array([248, 0, 248], np.float32), np.array([0, 0, 0], np.float32)))


def _blades(rng, cols: list, count: int, lo: int, hi: int, heads: str | None = None) -> np.ndarray:
    """A plant of ``count`` leaning blades reaching between ``lo`` and ``hi`` pixels from the top."""
    rgb = np.zeros((16, 16, 3), np.float32)
    alpha = np.zeros((16, 16), np.uint8)
    for _ in range(count):
        x = int(rng.integers(1, 15))
        top = int(rng.integers(lo, hi + 1))
        lean = int(rng.choice([-1, 0, 1]))
        for y in range(top, 16):
            xx = int(np.clip(x + lean * ((16 - y) // 5), 0, 15))
            rgb[y, xx] = _hex(cols[int(rng.integers(0, len(cols)))])
            alpha[y, xx] = 255
        if heads:
            rgb[top:top + 3, max(x - 1, 0):x + 2] = _hex(heads)
            alpha[top:top + 3, max(x - 1, 0):x + 2] = 255
    return _rgba(rgb, alpha)


def _build_more() -> None:
    def r(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    green = ["6aa83c", "5f9b35", "77b545", "4a7d29"]
    _add("fern", _blades(r(70), ["4f8f3a", "5fa246", "437b30"], 12, 4, 9))
    cane = np.zeros((16, 16, 4), np.uint8)
    for x in (3, 7, 12):
        cane[:, x:x + 2, :3] = _pick(r(71 + x), ["7cbf52", "6aa83c", "8ccc5a"], shape=(16, 2))
        cane[:, x:x + 2, 3] = 255
        cane[4::5, x:x + 2, :3] = _hex("4a7d29")
    _add("sugar_cane", cane)
    _add("seagrass", _blades(r(72), ["2f7d4a", "3a9058", "276a3d"], 9, 2, 8))
    _add("kelp", _blades(r(73), ["2a6b3c", "357d47", "235a32"], 5, 0, 2))
    farm = _dirt(r(74)) * 0.62
    farm[:, 3::4] *= 0.8
    _add("farmland", farm)
    for stage, (top, col) in enumerate(((11, green), (8, green), (5, ["9fb545", "8aa83c", "b5c24f"]),
                                        (2, ["d9c65a", "c9b44a", "e6d470"]))):
        _add(f"wheat{stage}", _blades(r(75 + stage), col, 10, top, top + 2, "e6d470" if stage == 3 else None))
    _add("oak_sapling", _blades(r(80), ["3f7a1f", "4a8a26", "6b5132"], 4, 5, 9, "4a8a26"))
    bed = _wool(r(81), WOOL_COLOURS[14])
    bed[:5] = _wool(r(82), WOOL_COLOURS[0])[:5]
    _add("bed_top", bed)
    pump = _pick(r(83), ["e08a1e", "d47c16", "ec9a2c"], [4, 3, 2])
    pump[:, ::4] = _hex("b8650c")
    _add("pumpkin_side", pump)
    ptop = _pick(r(84), ["e08a1e", "d47c16", "ec9a2c"], [4, 3, 2])
    ptop[6:10, 6:10] = _hex("5a7a2a")
    _add("pumpkin_top", ptop)
    _add("cornflower", _flower(["4a6ee0", "3a5ad0", "6a8af0"], "ffe419", "dandelion"))
    _add("allium", _flower(["b06ad8", "c88aee", "9a52c4"], None, "poppy"))
    _add("daisy", _flower(["f4f4f4", "e2e2e2", "ffffff"], "ffe419", "dandelion"))
    _add("red_mushroom", _flower(["d8281e", "f0f0f0", "c01810"], None, "poppy"))
    _add("brown_mushroom", _flower(["9a7252", "86603f", "ad8562"], None, "dandelion"))
    hay = _pick(r(85), ["d9b03a", "c99f2c", "e6c04c", "b8902a"], [4, 3, 2, 1])
    hay[:, ::3] *= 0.86
    hay[4:6, :] = hay[11:13, :] = _hex("a5342a")
    _add("hay_side", hay)
    _add("hay_top", _pick(r(86), ["d9b03a", "c99f2c", "e6c04c", "b8902a"], [4, 3, 2, 1]))


def _build_realms() -> None:
    """The Deep and the Sky Isles (drawn here: they are BlockCraft's own places)."""
    def r(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    deep = _stone(r(200)) * np.array([0.42, 0.34, 0.36], np.float32)
    _add("deepstone", deep)
    _add("ash", _pick(r(201), ["4a4644", "56524f", "3e3b39", "625d59"], [4, 3, 3, 1]))
    moss = _pick(r(202), ["a8321c", "bf4222", "8f2616", "d4592a"], [4, 3, 3, 1])
    moss[r(203).random((16, 16)) < 0.06] = (255, 190, 70)
    _add("ember_moss", moss)
    side = deep.copy()
    rng = r(204)
    for x in range(16):
        d = 2 + (rng.random() < 0.5) + (rng.random() < 0.2)
        side[:d, x] = moss[:d, x]
    _add("ember_moss_side", side)
    cap = _pick(r(205), ["ffb347", "ff9a2e", "ffcf6b", "f07d1c"], [4, 3, 2, 2])
    cap[r(206).random((16, 16)) < 0.08] = (255, 244, 200)
    _add("glowcap", cap)
    _add("glowstem", _log_side(r(207), False, ["c9b79a", "b8a586", "d6c6ab", "a8946f"], "8f7a58"))
    cry = _pick(r(208), ["4fd6e8", "7ee6f2", "36b8d0", "b4f4fa"], [4, 3, 2, 1])
    for k in range(4):
        y, x = r(209 + k).integers(0, 11, 2)
        for i in range(5):
            cry[y + i, x + i] = (236, 252, 255)
    _add("deep_crystal", cry)
    bas = _pick(r(214), ["3a3a40", "44444b", "303036", "4e4e56"], [4, 3, 3, 1])
    bas[:, ::4] *= 0.78
    _add("basalt", bas)
    top = _pick(r(215), ["3a3a40", "44444b", "303036"], [4, 3, 3])
    top[::4, :] *= 0.8
    top[:, ::4] *= 0.8
    _add("basalt_top", top)
    _add("charred_log", _log_side(r(216), False, ["2a2420", "211c19", "332b26", "191513"], "0e0b0a"))
    star = deep.copy()
    for k in range(6):
        y, x = r(217 + k).integers(2, 14, 2)
        star[y, x] = (255, 250, 210)
        star[y - 1, x] = star[y + 1, x] = star[y, x - 1] = star[y, x + 1] = (244, 214, 120)
    _add("starstone", star)
    for name, ring, core in (("deep_gate", "d4592a", "ffcf6b"), ("sky_gate", "7ee6f2", "f4fdff")):
        g = _stone(r(223)) * 0.5
        yy, xx = np.mgrid[0:16, 0:16]
        dist = np.maximum(np.abs(xx - 7.5), np.abs(yy - 7.5))
        g[(dist > 3) & (dist < 6)] = _hex(ring)
        g[dist < 2] = _hex(core)
        _add(name, g)
    _add("voidstone", _stone(r(224)) * np.array([0.92, 1.0, 1.12], np.float32) + 34)
    mm = _pick(r(225), ["3fbfa8", "52d0b8", "34a892", "6ee0c8"], [4, 3, 3, 1])
    _add("moonmoss", mm)
    ms = np.clip(_stone(r(224)) * np.array([0.92, 1.0, 1.12], np.float32) + 34, 0, 255)
    rng = r(226)
    for x in range(16):
        d = 3 + (rng.random() < 0.5) + (rng.random() < 0.2)
        ms[:d, x] = mm[:d, x]
    _add("moonmoss_side", ms)
    _add("pale_log", _log_side(r(227), False, ["dfe3ea", "ccd2dc", "eef1f5", "b9c0cc"], "9aa3b2"))
    _add("pale_leaves", _leaves(r(228), ["9fe8f0", "b8f2f6", "86d8e4", "d2fafc"]))
    _add("cloud", _pick(r(229), ["f4f8fc", "ffffff", "e8eef6"], [4, 3, 2]))
    alt = _stone(r(230)) * 0.45
    alt[2:14, 2:14] = _pick(r(231), ["2a3550", "34405e", "232c44"], shape=(12, 12))
    alt[6:10, 6:10] = _pick(r(232), ["7ee6f2", "f4fdff"], shape=(4, 4))
    _add("storm_altar", alt)


_build_textures()
_build_more()
_build_realms()

# --- the texture pack ------------------------------------------------------------------------------------------------
log = logging.getLogger(__name__)
PACK_NAME = "Pixel Perfection"  # a folder beside this file whose name starts with this
PACK_CREDIT = ""  # shown on the title screen when the pack is in use
CRACK_RGBA: np.ndarray | None = None  # the pack's 10 crack stages (16, 16, 4)
_PACK_ITEMS: dict[int, np.ndarray] = {}
_CUTOUT_TEX = {"oak_leaves", "birch_leaves", "spruce_leaves", "glass", "torch", "dandelion", "poppy", "tall_grass",
               "pale_leaves",
               "dead_bush", "fern", "sugar_cane", "seagrass", "kelp", "wheat0", "wheat1", "wheat2", "wheat3",
               "oak_sapling", "cornflower", "allium", "daisy", "red_mushroom", "brown_mushroom"}
# our texture -> the pack's block picture (most have the same name)
_PACK_BLOCKS = {n: n for n in (
    "stone", "dirt", "cobblestone", "mossy_cobblestone", "oak_planks", "birch_planks", "spruce_planks", "bedrock",
    "sand", "gravel", "gold_ore", "iron_ore", "coal_ore", "diamond_ore", "redstone_ore", "oak_log", "oak_log_top",
    "birch_log", "birch_log_top", "spruce_log", "spruce_log_top", "oak_leaves", "birch_leaves", "spruce_leaves",
    "glass", "bricks", "tnt_side", "tnt_top", "tnt_bottom", "bookshelf", "obsidian", "crafting_table_top",
    "crafting_table_front", "crafting_table_side", "furnace_side", "furnace_top", "furnace_front", "snow",
    "sandstone", "sandstone_top", "stone_bricks", "glowstone", "iron_block", "gold_block", "diamond_block", "torch",
    "dandelion", "poppy", "dead_bush", "cactus_side", "cactus_top", "ice", "clay", "fern", "sugar_cane", "seagrass",
    "kelp", "oak_sapling", "pumpkin_side", "pumpkin_top", "cornflower", "allium", "red_mushroom", "brown_mushroom")}
_PACK_BLOCKS.update({
    "grass_top": "grass_block_top", "furnace_front_lit": "furnace_front_on", "tall_grass": "short_grass",
    "snowy_grass_side": "grass_block_snow", "farmland": "farmland_moist", "daisy": "oxeye_daisy",
    "hay_side": "hay_block_side", "hay_top": "hay_block_top", "bed_top": "red_wool",
    "wheat0": "wheat_stage0", "wheat1": "wheat_stage2", "wheat2": "wheat_stage4", "wheat3": "wheat_stage6"})
_PACK_BLOCKS.update({f"wool{i}": f"{c}_wool" for i, c in enumerate(
    ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray", "cyan", "purple",
     "blue", "brown", "green", "red", "black"))})
_PACK_ITEM_FILES = {STICK: "stick", COAL: "coal", IRON_INGOT: "iron_ingot", GOLD_INGOT: "gold_ingot",
                    DIAMOND: "diamond", REDSTONE: "redstone", APPLE: "apple", PORK: "porkchop",
                    COOKED_PORK: "cooked_porkchop", BEEF: "beef", STEAK: "cooked_beef", FLINT: "flint",
                    FLINT_STEEL: "flint_and_steel", ROTTEN_FLESH: "rotten_flesh", WHEAT_SEEDS: "wheat_seeds",
                    WHEAT_ITEM: "wheat", BREAD: "bread", SUGAR: "sugar", BUCKET: "bucket",
                    WATER_BUCKET: "water_bucket", FISH: "cod", COOKED_FISH: "cooked_cod", FEATHER: "feather",
                    CHICKEN: "chicken", COOKED_CHICKEN: "cooked_chicken", STRING: "string",
                    SLIME_BALL: "slime_ball", GEM: "emerald"}
for _m, _mn in enumerate(MATERIALS):
    for _k, _kn in enumerate(TOOL_KINDS):
        _PACK_ITEM_FILES[tool_id(_m, _k)] = f"{_mn}_{_kn}"
    _PACK_ITEM_FILES[HOE0 + _m] = f"{_mn}_hoe"
for _m, _mn in enumerate(ARMOR_MATERIALS):
    for _k, _kn in enumerate(ARMOR_PIECES):
        _PACK_ITEM_FILES[ARMOR0 + _m * 4 + _k] = f"{_mn}_{_kn}"
# what the game engine these packs are drawn for colours in itself: the pack's own colour maps for grass and oak
# leaves (read at a mild, green spot), and fixed colours for the other leaves and for water
_TINT_FIXED = {"birch_leaves": (128, 167, 85), "spruce_leaves": (97, 153, 97), "water": (63, 118, 228)}
_TINT_GRASS = ("grass_top", "tall_grass", "fern")
_COLOURMAP_SPOT = (173, 51)  # (row, column)


def _read_png(path: Path) -> np.ndarray | None:
    """A picture file as (h, w, 4) RGBA bytes (None when it cannot be read)."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("RGBA")).copy()
    except Exception:  # noqa: BLE001 - no Pillow here, or not a picture: Qt reads it
        pass
    try:
        from PySide6.QtGui import QImage

        img = QImage(str(path))
        if img.isNull():
            return None
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        raw = np.frombuffer(img.constBits(), np.uint8, img.sizeInBytes()).reshape(h, img.bytesPerLine())
        return raw[:, :w * 4].reshape(h, w, 4).copy()
    except Exception:  # noqa: BLE001
        return None


def _apply_pack() -> None:
    """Use the texture pack's block and item pictures where it has them (see the module docstring)."""
    global PACK_CREDIT, CRACK_RGBA  # noqa: PLW0603 - set once, as the module loads
    if os.environ.get("TDS_BLOCKCRAFT_PACK", "1") == "0":
        return
    root = next((d / "assets" / "minecraft" / "textures" for d in sorted(Path(__file__).parent.glob(PACK_NAME + "*"))
                 if (d / "assets" / "minecraft" / "textures" / "block" / "stone.png").is_file()), None)
    if root is None:
        return

    def tile(folder: str, name: str, frame: int = 0) -> np.ndarray | None:
        img = _read_png(root / folder / (name + ".png"))
        if img is None or img.shape[1] != 16 or img.shape[0] < 16 * (frame + 1):
            return None
        return img[frame * 16:frame * 16 + 16].copy()

    def tinted(img: np.ndarray, colour: tuple) -> np.ndarray:
        out = img.copy()
        out[..., :3] = (img[..., :3].astype(np.float32) * (np.array(colour[:3], np.float32) / 255)).astype(np.uint8)
        return out

    grass_tint = foliage_tint = None
    for name in ("grass", "foliage"):
        cm = _read_png(root / "colormap" / (name + ".png"))
        if cm is not None and cm.shape[0] > _COLOURMAP_SPOT[0] and cm.shape[1] > _COLOURMAP_SPOT[1]:
            colour = tuple(int(v) for v in cm[_COLOURMAP_SPOT][:3])
            if name == "grass":
                grass_tint = colour
            else:
                foliage_tint = colour
    used = 0
    for ours, theirs in _PACK_BLOCKS.items():
        img = tile("block", theirs)
        if img is None or ours not in _TEX:
            continue
        if ours in _TINT_GRASS and grass_tint:
            img = tinted(img, grass_tint)
        elif ours == "oak_leaves" and foliage_tint:
            img = tinted(img, foliage_tint)
        elif ours in _TINT_FIXED:
            img = tinted(img, _TINT_FIXED[ours])
        if ours in _CUTOUT_TEX:
            img[..., 3] = np.where(img[..., 3] > 127, 255, 0)
        else:
            img[..., 3] = 255
        _TEX[ours] = img
        used += 1
    side, over = tile("block", "grass_block_side"), tile("block", "grass_block_side_overlay")
    if side is not None:
        if over is not None and grass_tint:
            a = (over[..., 3:4] > 127)
            side[..., :3] = np.where(a, tinted(over, grass_tint)[..., :3], side[..., :3])
        side[..., 3] = 255
        _TEX["grass_side"] = side
    for liquid, tint in (("water", _TINT_FIXED["water"]), ("lava", None)):
        strip = _read_png(root / "block" / f"{liquid}_still.png")
        if strip is None or strip.shape[1] != 16 or strip.shape[0] < 16:
            continue
        frames = strip.shape[0] // 16
        for f in range(8):
            img = strip[(f * frames // 8) * 16:(f * frames // 8) * 16 + 16].copy()
            if tint and float(np.ptp(img[..., :3].astype(np.int16), axis=-1).mean()) < 12:  # drawn grey: tint it
                img = tinted(img, tint)
            img[..., 3] = 255
            _TEX[f"{liquid}{f}"] = img
    cracks = [tile("block", f"destroy_stage_{i}") for i in range(10)]
    if all(c is not None for c in cracks):
        CRACK_RGBA = np.stack(cracks)
    for item, name in _PACK_ITEM_FILES.items():
        img = tile("item", name)
        if img is not None:
            _PACK_ITEMS[item] = img
    PACK_CREDIT = "Textures: Pixel Perfection by XSSheep and Nova_Wostra (CC BY-SA 4.0)"
    log.info("BlockCraft uses the %s texture pack (%d block pictures, %d item pictures)", PACK_NAME, used,
             len(_PACK_ITEMS))


if multiprocessing.parent_process() is None:  # (the helper processes never draw: they keep the plain pictures)
    try:
        _apply_pack()
    except Exception:  # noqa: BLE001 - then the game looks as drawn by code
        log.warning("BlockCraft texture pack not used", exc_info=True)

TEX_INDEX = {name: i for i, name in enumerate(TEXTURES)}
ATLAS = np.stack([_TEX[n] for n in TEXTURES])  # (T, 16, 16, 4)
ATLAS_RGB = ATLAS[..., :3].astype(np.float32)
ATLAS_ALPHA = ATLAS[..., 3] > 127
WATER_TEX = [TEX_INDEX[f"water{f}"] for f in range(8)]
LAVA_TEX = [TEX_INDEX[f"lava{f}"] for f in range(8)]

# per block and face (0 +x east, 1 -x west, 2 top, 3 bottom, 4 +z south, 5 -z north) -> texture index
FACE_TEX = np.full((256, 6), TEX_INDEX["missing"], np.int16)


def _faces(b: int, side: str, top: str | None = None, bottom: str | None = None) -> None:
    FACE_TEX[b, :] = TEX_INDEX[side]
    FACE_TEX[b, 2] = TEX_INDEX[top or side]
    FACE_TEX[b, 3] = TEX_INDEX[bottom or top or side]


for _b, _n in ((STONE, "stone"), (DIRT, "dirt"), (COBBLE, "cobblestone"), (OAK_PLANKS, "oak_planks"),
               (BEDROCK, "bedrock"), (SAND, "sand"), (GRAVEL, "gravel"), (GOLD_ORE, "gold_ore"),
               (IRON_ORE, "iron_ore"), (COAL_ORE, "coal_ore"), (OAK_LEAVES, "oak_leaves"), (GLASS, "glass"),
               (DIAMOND_ORE, "diamond_ore"), (REDSTONE_ORE, "redstone_ore"), (BRICKS, "bricks"),
               (MOSSY_COBBLE, "mossy_cobblestone"), (OBSIDIAN, "obsidian"), (BIRCH_LEAVES, "birch_leaves"),
               (BIRCH_PLANKS, "birch_planks"), (SNOW, "snow"), (STONE_BRICKS, "stone_bricks"),
               (GLOWSTONE, "glowstone"), (IRON_BLOCK, "iron_block"), (GOLD_BLOCK, "gold_block"),
               (DIAMOND_BLOCK, "diamond_block"), (TORCH, "torch"), (DANDELION, "dandelion"), (POPPY, "poppy"),
               (TALL_GRASS, "tall_grass"), (DEAD_BUSH, "dead_bush"), (WATER, "water0"), (LAVA, "lava0"),
               (SPRUCE_LEAVES, "spruce_leaves"), (SPRUCE_PLANKS, "spruce_planks"), (ICE, "ice"), (CLAY, "clay"),
               (FERN, "fern"), (SUGAR_CANE, "sugar_cane"), (SEAGRASS, "seagrass"), (KELP, "kelp"),
               (OAK_SAPLING, "oak_sapling"), (CORNFLOWER, "cornflower"), (ALLIUM, "allium"), (DAISY, "daisy"),
               (RED_MUSHROOM, "red_mushroom"), (BROWN_MUSHROOM, "brown_mushroom")):
    _faces(_b, _n)
for _i in range(4):
    _faces(WHEAT + _i, f"wheat{_i}")
for _i in range(1, 8):
    _faces(FLOW + _i, "water0")
for _b, _n in ((DEEPSTONE, "deepstone"), (ASH, "ash"), (GLOWCAP, "glowcap"), (GLOWSTEM, "glowstem"),
               (DEEP_CRYSTAL, "deep_crystal"), (CHARRED_LOG, "charred_log"), (STARSTONE, "starstone"),
               (DEEP_GATE, "deep_gate"), (SKY_GATE, "sky_gate"), (VOIDSTONE, "voidstone"), (PALE_LOG, "pale_log"),
               (PALE_LEAVES, "pale_leaves"), (CLOUD, "cloud"), (STORM_ALTAR, "storm_altar")):
    _faces(_b, _n)
_faces(EMBER_MOSS, "ember_moss_side", "ember_moss", "deepstone")
_faces(MOONMOSS, "moonmoss_side", "moonmoss", "voidstone")
_faces(BASALT, "basalt", "basalt_top")
_faces(FARMLAND, "dirt", "farmland", "dirt")
_faces(BED, "oak_planks", "bed_top", "oak_planks")
_faces(PUMPKIN, "pumpkin_side", "pumpkin_top")
_faces(HAY, "hay_side", "hay_top")
_faces(SPRUCE_LOG, "spruce_log", "spruce_log_top")
_faces(CACTUS, "cactus_side", "cactus_top")
_faces(SNOWY_GRASS, "snowy_grass_side", "snow", "dirt")
_faces(GRASS, "grass_side", "grass_top", "dirt")
_faces(OAK_LOG, "oak_log", "oak_log_top")
_faces(BIRCH_LOG, "birch_log", "birch_log_top")
_faces(TNT, "tnt_side", "tnt_top", "tnt_bottom")
_faces(BOOKSHELF, "bookshelf", "oak_planks")
_faces(SANDSTONE, "sandstone", "sandstone_top")
_faces(CRAFTING_TABLE, "crafting_table_side", "crafting_table_top", "oak_planks")
FACE_TEX[CRAFTING_TABLE, 4] = FACE_TEX[CRAFTING_TABLE, 5] = TEX_INDEX["crafting_table_front"]
for _i in range(16):
    _faces(WOOL + _i, f"wool{_i}")
for _i, _face in enumerate((5, 0, 4, 1)):  # the furnace front faces north / east / south / west
    for _lit in (0, 1):
        _b = FURNACE + _i + 4 * _lit
        _faces(_b, "furnace_side", "furnace_top")
        FACE_TEX[_b, _face] = TEX_INDEX["furnace_front_lit" if _lit else "furnace_front"]

# the colour a block shows on a map / in particles (average texture colour)
AVG_COLOUR = np.zeros((256, 3), np.float32)
for _b in range(256):
    _t = FACE_TEX[_b, 4]
    _a = ATLAS_ALPHA[_t]
    AVG_COLOUR[_b] = ATLAS_RGB[_t][_a].mean(0) if _a.any() else (128, 128, 128)

# --- block breaking cracks ---------------------------------------------------------------------------------------


def _cracks() -> np.ndarray:
    """10 stages of crack masks (True = darken)."""
    rng = np.random.default_rng(77)
    stages = np.zeros((10, 16, 16), bool)
    walkers = [(7.5, 7.5, a) for a in rng.uniform(0, 2 * np.pi, 5)]
    mask = np.zeros((16, 16), bool)
    for s in range(10):
        new = []
        for y, x, a in walkers:
            for _ in range(2 + s // 3):
                a += rng.normal(0, 0.6)
                y, x = y + np.sin(a), x + np.cos(a)
                if 0 <= y < 16 and 0 <= x < 16:
                    mask[int(y), int(x)] = True
            new.append((y % 16, x % 16, a))
            if rng.random() < 0.25:
                new.append((y % 16, x % 16, a + rng.normal(0, 1.2)))
        walkers = new[:14]
        stages[s] = mask
    return stages


CRACKS = _cracks()

# --- item sprites (16x16 RGBA) -----------------------------------------------------------------------------------
_TOOL_COLOURS = (("a2834f", "6e5634"), ("8a8a8a", "4e4e4e"), ("e6e6e6", "8c8c8c"), ("fcee4b", "b88a15"),
                 ("5decf5", "1f8f8f"))
_HANDLE = ("8a6a3a", "4b3820")

_SHAPES = {
    "pickaxe": ["....mmmmmm......",
                "...mmmmmmmm.....",
                "..mm.....mmm....",
                "........hmmm....",
                ".......h..mmm...",
                "......h....mm...",
                ".....h.....mm...",
                "....h.......m...",
                "...h............",
                "..h.............",
                ".h..............",
                "................"],
    "axe": ["......mm........",
            ".....mmmm.......",
            "....mmmmmm......",
            "....mmmmhmm.....",
            ".....mmh.mm.....",
            "......h.........",
            ".....h..........",
            "....h...........",
            "...h............",
            "..h.............",
            ".h..............",
            "................"],
    "shovel": [".........mmm....",
               "........mmmmm...",
               "........mmmmm...",
               ".........mmm....",
               "........h.......",
               ".......h........",
               "......h.........",
               ".....h..........",
               "....h...........",
               "...h............",
               "..h.............",
               "................"],
    "sword": ["...........mm...",
              "..........mmm...",
              ".........mmm....",
              "........mmm.....",
              ".......mmm......",
              "......mmm.......",
              ".m...mmm........",
              "..m.mmm.........",
              "...hmm..........",
              "..h.m...........",
              ".h...m..........",
              "................"],
}


def _sprite_from(rows: list[str], colours: dict[str, tuple], offset: tuple[int, int] = (2, 2),
                 thicken: str = "h") -> np.ndarray:
    """An item sprite from ASCII art: edge pixels darker, a light pixel per run; 'h' strokes are 2 px wide."""
    img = np.zeros((16, 16, 4), np.uint8)
    oy, ox = offset
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in colours:
                for dy, dx in ((0, 0), (1, 0)) if ch in thicken else ((0, 0),):
                    yy, xx = y + oy + dy, x + ox + dx
                    if 0 <= yy < 16 and 0 <= xx < 16:
                        img[yy, xx, :3] = colours[ch][0]
                        img[yy, xx, 3] = 255
    solid = img[..., 3] > 0
    edge = solid & ~(np.roll(solid, -1, 0) & np.roll(solid, -1, 1))  # the shadow side: below / right open
    for ch, (_light, dark) in colours.items():
        sel = edge & np.all(img[..., :3] == np.array(colours[ch][0], np.uint8), -1)
        img[sel, :3] = dark
    return img


_PLAIN = {WHEAT_SEEDS: ("7cbf52", "4a7d29"), WHEAT_ITEM: ("e6d470", "b8902a"), BREAD: ("c98f4a", "8a5a22"),
          SUGAR: ("f4f4f4", "c8c8c8"), BUCKET: ("c8c8c8", "6a6a6a"), WATER_BUCKET: ("5a8cf0", "2f5fd8"),
          FISH: ("8aa8c8", "4a6a8a"), COOKED_FISH: ("d8b890", "8a6a42"), FEATHER: ("f0f0f0", "b0b0b0"),
          CHICKEN: ("f0c0b0", "b08070"), COOKED_CHICKEN: ("c88a5a", "7a4a2a"), STRING: ("e8e8e8", "a8a8a8"),
          SLIME_BALL: ("7cd060", "3a8a2a"), GEM: ("3ad878", "187a3a")}


_ARMOR_SHAPES = (["..mmmmmm..", ".mmmmmmmm.", ".mmmmmmmm.", ".mm....mm.", ".mm....mm."],
                 ["mm......mm", "mmm....mmm", "mmmmmmmmmm", ".mmmmmmmm.", ".mmmmmmmm.", ".mmmmmmmm.", ".mmmmmmmm."],
                 [".mmmmmmmm.", ".mmmmmmmm.", ".mmm..mmm.", ".mm....mm.", ".mm....mm.", ".mm....mm.", ".mm....mm."],
                 [".mm....mm.", ".mm....mm.", ".mm....mm.", "mmm....mmm", "mmm....mmm"])


def _item_sprite(item: int) -> np.ndarray:
    arm = armor_info(item)
    if arm is not None:
        light, dark = (_TOOL_COLOURS[2], _TOOL_COLOURS[3], _TOOL_COLOURS[4])[arm[0]]
        rows = _ARMOR_SHAPES[arm[1]]
        return _sprite_from(rows, {"m": (_hex(light), _hex(dark))}, ((16 - len(rows)) // 2, 3))
    if item in _PLAIN:  # a plain rounded shape in the item's colours
        lt, dk = _PLAIN[item]
        rows = ["...mmmm...", "..mmmmmm..", ".mmmmmmmm.", ".mmmmmmmm.", "..mmmmmm..", "...mmmm..."]
        return _sprite_from(rows, {"m": (_hex(lt), _hex(dk))}, (5, 3))
    hoe = hoe_material(item)
    if hoe is not None:
        light, dark = _TOOL_COLOURS[hoe]
        rows = ["......mmmm......", ".......hmm......", "......h.........", ".....h..........", "....h...........",
                "...h............", "..h.............", ".h.............."]
        return _sprite_from(rows, {"m": (_hex(light), _hex(dark)), "h": (_hex(_HANDLE[0]), _hex(_HANDLE[1]))},
                            thicken="hm")
    info = tool_info(item)
    if info is not None:
        mat, kind = info
        light, dark = _TOOL_COLOURS[mat]
        cols = {"m": (_hex(light), _hex(dark)), "h": (_hex(_HANDLE[0]), _hex(_HANDLE[1]))}
        return _sprite_from(_SHAPES[TOOL_KINDS[kind]], cols, thicken="hm")
    shapes = {
        STICK: ([".........h", "........h.", ".......h..", "......h...", ".....h....", "....h.....", "...h......",
                 "..h.......", ".h........"], {"h": ("8a6a3a", "4b3820")}),
        COAL: (["....kkk...", "..kkkkkk..", ".kkkgkkkk.", ".kkkkkkgk.", ".kgkkkkkk.", "..kkkkkk..", "...kkk...."],
               {"k": ("2b2b2b", "111111"), "g": ("5a5a5a", "3a3a3a")}),
        IRON_INGOT: (["..........", "...iiiiii.", "..iiiiiii.", ".iiiiiii..", "iiiiiii...", "iiiiii...."],
                     {"i": ("e6e6e6", "8c8c8c")}),
        GOLD_INGOT: (["..........", "...iiiiii.", "..iiiiiii.", ".iiiiiii..", "iiiiiii...", "iiiiii...."],
                     {"i": ("fcee4b", "b88a15")}),
        DIAMOND: (["...dddd...", "..dddddd..", ".dddwdddd.", "dddwddddd.", ".dddddddd.", "..dddddd..",
                   "...dddd...", "....dd...."], {"d": ("5decf5", "1f8f8f"), "w": ("ffffff", "ffffff")}),
        REDSTONE: (["...r..r...", ".r..rr..r.", "..rrrrrr..", ".rrrrrrr..", "..rrrrr.r.", ".r..rr....",
                    "....r..r.."], {"r": ("ff1a1a", "8a0000")}),
        APPLE: (["....s.....", "...ls.....", "..rrrrrr..", ".rrwrrrrr.", ".rrrrrrrr.", ".rrrrrrrr.", "..rrrrrr..",
                 "...rr.rr.."], {"r": ("e4141b", "8a0000"), "s": ("6b5132", "4b3820"), "l": ("4a8a26", "2a5a10"),
                                 "w": ("ffffff", "ffffff")}),
        FLINT: (["...ff.....", "..ffff....", ".ffffff...", ".fffffff..", "..ffffff..", "...ffff..."],
                {"f": ("4a4a4a", "1e1e1e")}),
        FLINT_STEEL: (["......ss..", ".....s..s.", "....s....s", "...s.....s", ".ff.......", "ffff......",
                       ".ff......."], {"s": ("c8c8c8", "6a6a6a"), "f": ("4a4a4a", "1e1e1e")}),
        ROTTEN_FLESH: (["..rrr.....", ".rgrrr....", ".rrrgrr...", "..rrrrrr..", "...rrgrr..", "....rrr..."],
                       {"r": ("a86a3a", "6a3a1a"), "g": ("6a8a3a", "3a5a1a")}),
    }
    meat = {PORK: ("f0a0a0", "b06060"), COOKED_PORK: ("c88a5a", "7a4a2a"), BEEF: ("d8403a", "8a1a1a"),
            STEAK: ("8a5030", "4a2a12")}
    if item in meat:
        rows = ["...mmmm...", "..mmmmmm..", ".mmmmmmmmw", ".mmmmmmmww", "..mmmmmww.", "...mmmm..."]
        lt, dk = meat[item]
        return _sprite_from(rows, {"m": (_hex(lt), _hex(dk)), "w": (_hex("f0e8d8"), _hex("b0a898"))}, (5, 3))
    rows, cols = shapes.get(item, (["#"], {"#": ("ff00ff", "000000")}))
    cols = {k: (_hex(a), _hex(b)) for k, (a, b) in cols.items()}
    h, w = len(rows), max(len(r) for r in rows)
    return _sprite_from(rows, cols, ((16 - h) // 2, (16 - w) // 2), thicken="h" if item == STICK else "")


_SPRITES: dict[int, np.ndarray] = {}


def item_sprite(item: int) -> np.ndarray:
    """16x16 RGBA for a non-block item (or a cross-sprite block)."""
    if item not in _SPRITES:
        if is_block(item):
            _SPRITES[item] = ATLAS[FACE_TEX[item, 4]].copy()
        elif item in _PACK_ITEMS:
            _SPRITES[item] = _PACK_ITEMS[item]
        else:
            _SPRITES[item] = _item_sprite(item)
    return _SPRITES[item]


def flat_item(item: int) -> bool:
    """True when the item is drawn as a flat sprite (items, plants, torches) rather than a little cube."""
    return not is_block(item) or KIND[item] == K_CROSS
