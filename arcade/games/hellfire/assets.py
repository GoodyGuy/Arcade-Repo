"""The art bank the renderer and the HUD draw from: wall and floor tiles (128 x 128 RGBA layers), the sky panoramas
and one atlas of every sprite (monsters from 8 sides, pickups, decorations, projectiles, the first-person weapons).

The pictures come from the ``freedoom`` folder beside this file when they are there (the Freedoom project's art,
BSD licence: see ``freedoom/COPYING.adoc``); anything missing falls back to Hellfire's own procedural art
(``textures.py``, ``sprites.py``, ``models.py``), so the game runs with the folder deleted."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage

from .palette import PALETTE

FD = Path(__file__).resolve().parent / "freedoom"
T = 128  # tile layer size (1 texel = 1 map unit)
PAD = 2  # transparent border around each sprite in the atlas
ATLAS_W = 2048

_png: dict[tuple[str, str], np.ndarray | None] = {}


def qimage_to_rgba(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8, count=img.bytesPerLine() * h).reshape(h, img.bytesPerLine())
    return buf[:, :w * 4].reshape(h, w, 4).copy()


def rgba_to_qimage(rgba: np.ndarray) -> QImage:
    rgba = np.ascontiguousarray(rgba, np.uint8)
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()


def png(sub: str, name: str) -> np.ndarray | None:
    """A Freedoom picture as (h, w, 4) RGBA, or None when the file is not there."""
    key = (sub, name)
    if key not in _png:
        arr = None
        path = FD / sub / (name + ".png")
        try:
            if path.is_file():
                img = QImage(str(path))
                if not img.isNull() and 0 < img.width() <= 1024 and 0 < img.height() <= 1024:
                    arr = qimage_to_rgba(img)
        except OSError:
            arr = None
        _png[key] = arr
    return _png[key]


def have_freedoom() -> bool:
    return png("sprites", "possa1") is not None and png("patches", "wall54_1") is not None


# --- tiles ------------------------------------------------------------------------------------------------------------

def tile(img: np.ndarray, w: int = T, h: int = T) -> np.ndarray:
    """Repeat / crop a picture to ``w`` x ``h``; see-through pixels become dark metal."""
    ih, iw = img.shape[:2]
    out = np.tile(img, (math.ceil(h / ih), math.ceil(w / iw), 1))[:h, :w].copy()
    clear = out[..., 3] < 128
    out[clear] = (18, 18, 20, 255)
    out[..., 3] = 255
    return out


def over(base: np.ndarray, patch: np.ndarray, x: int, y: int, scale: int = 1) -> np.ndarray:
    """``patch`` (scaled by a whole number) drawn on ``base`` at (x, y)."""
    if scale > 1:
        patch = np.repeat(np.repeat(patch, scale, axis=0), scale, axis=1)
    out = base.copy()
    ph, pw = patch.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + pw, out.shape[1]), min(y + ph, out.shape[0])
    if x0 >= x1 or y0 >= y1:
        return out
    sub = patch[y0 - y:y1 - y, x0 - x:x1 - x]
    m = sub[..., 3] >= 128
    out[y0:y1, x0:x1][m] = sub[m]
    out[..., 3] = 255
    return out


def _whole(*names: str, sub: str = "patches"):  # noqa: ANN202 - a recipe
    def make() -> list[np.ndarray] | None:
        out = []
        for n in names:
            a = png(sub, n)
            if a is None:
                return None
            out.append(tile(a))
        return out
    return make


def _flat(*names: str):  # noqa: ANN202
    return _whole(*names, sub="flats")


def _crop(*names: str, width: int = T):  # noqa: ANN202 - the middle ``width`` columns of wider patches
    def make() -> list[np.ndarray] | None:
        out = []
        for n in names:
            a = png("patches", n)
            if a is None:
                return None
            x = max((a.shape[1] - width) // 2, 0)
            out.append(tile(a[:, x:x + width]))
        return out
    return make


def _keydoor(colour: tuple[int, int, int]):  # noqa: ANN202 - a door with glowing bars of the card's colour
    def make() -> list[np.ndarray] | None:
        a = png("patches", "door15_3")
        if a is None:
            return None
        d = tile(a[:, :64], 64, T)
        c = np.array(colour, np.float32)
        for x0 in (2, 56):
            for i in range(6):
                f = 1.0 - abs(i - 2.5) / 4.0
                d[6:122, x0 + i, :3] = np.clip(c * (0.55 + 0.6 * f), 0, 255).astype(np.uint8)
            d[4:6, x0:x0 + 6, :3] = d[122:124, x0:x0 + 6, :3] = 20
        d[58:70, 14:50, :3] = np.clip(c * 0.9, 0, 255).astype(np.uint8)
        d[58, 14:50, :3] = d[69, 14:50, :3] = 20
        d[58:70, 14, :3] = d[58:70, 49, :3] = 20
        return [tile(d)]
    return make


def _panel(base: str, *parts: tuple[str, int, int, int]):  # noqa: ANN202 - small patches on a 64-wide wall panel
    """``parts``: (patch, x, y, scale) placed on each 64-unit half of the tile (y from the tile's top)."""
    def make() -> list[np.ndarray] | None:
        b = png("patches", base)
        if b is None:
            return None
        t = tile(b)
        for name, x, y, scale in parts:
            p = png("patches", name)
            if p is None:
                return None
            for ox in (0, 64):
                t = over(t, p, ox + x, y, scale)
        return [t]
    return make


# name -> (the Freedoom recipe, the procedural texture used without the pictures)
WALLS: dict[str, tuple] = {
    "TECHTAN": (_whole("wall54_1"), "TECHTAN"), "TECHGRY": (_whole("aqpanl01"), "TECHGRY"),
    "STONE": (_whole("brick1"), "STONE"), "BROWN": (_whole("stonew1"), "BROWN"),
    "BRNPANEL": (_whole("w98_1"), "BRNPANEL"),
    "COMPUTE": (_crop("dosha1", "doshb1", "doshc1", "doshd1"), "COMPUTE"),
    "COMPSTA": (_crop("doblip1a", "doblip2a", "doblip3a", "doblip4a", width=64), "COMPSTA"),
    "METAL": (_whole("aqmetl13"), "METAL"), "SUPPORT": (_whole("aqsupp02"), "SUPPORT"),
    "DOOR": (_whole("door15_1"), "DOOR"), "DOORBLU": (_keydoor((70, 110, 255)), "DOORBLU"),
    "DOORYEL": (_keydoor((255, 214, 40)), "DOORYEL"), "DOORRED": (_keydoor((255, 50, 40)), "DOORRED"),
    "DOORTRAK": (_whole("doortrak"), "DOORTRAK"), "HAZARD": (_whole("aqpanl09"), "HAZARD"),
    "SWOFF": (_panel("w33_5", ("sw1s0", 16, 68, 1)), "SWOFF"), "SWON": (_panel("w33_5", ("sw1s1", 16, 68, 1)), "SWON"),
    "EXITOFF": (_panel("w33_5", ("sw2s0", 16, 68, 1), ("exit_red", 0, 26, 2)), "EXITOFF"),
    "EXITON": (_panel("w33_5", ("sw2s1", 16, 68, 1), ("exit_red", 0, 26, 2)), "EXITON"),
    "EXITSIGN": (_panel("w33_5", ("exit_red", 0, 44, 2)), "EXITSIGN"),
    "LITE": (_whole("yellite3"), "LITE"), "MARBLE": (_whole("marble1"), "MARBLE"),
    "SKIN": (_whole("rw47_3"), "SKIN"), "GSTONE": (_whole("mc1"), "GSTONE"), "SLIME": (_whole("tile"), "SLIME"),
    "PIPES": (_whole("w94_1"), "PIPES"), "STEP": (_whole("aqmetl03"), "STEP"),
    "TEKWALL": (_whole("aqpanl02"), "TEKWALL"), "LIFT": (_whole("aqmetl04"), "LIFT"),
    "REDBRICK": (_whole("w64b_1"), "REDBRICK"),
    "LAVAFALL": (_whole("lfall1", "lfall2", "lfall3", "lfall4"), "LAVAFALL"),
    "SKULLWAL": (_whole("mwall4_1"), "SKULLWAL"),
    # walls added with the pictures
    "CRATE": (_whole("rw13_1"), "BRNPANEL"), "SLIMEFAL": (_whole("sfall1", "sfall2", "sfall3", "sfall4"), "SLIME"),
    "SILVER": (_whole("shawn01"), "TECHGRY"), "ROCKWALL": (_whole("rock"), "STONE"),
    "CAVE": (_whole("cavern1"), "BROWN"), "FLESH": (_whole("hell5_1"), "SKIN"),
    "SKULLS": (_whole("wall76_1"), "SKULLWAL"), "PANEL2": (_whole("wall52_1"), "TECHTAN"),
    "PANEL3": (_whole("wall53_1"), "TECHTAN"), "PANEL4": (_whole("wall55_1"), "TECHTAN"),
    "WOOD": (_whole("wall40_1"), "BRNPANEL"), "IDOL": (_whole("mwall5_1"), "GSTONE"),
    "HOTROCK": (_whole("rw44_2"), "REDBRICK"), "FIREWALL": (_whole("w73a_1"), "REDBRICK"),
    "RUST": (_whole("aqrust05"), "METAL"), "CONC": (_whole("concrete"), "STONE"),
    "GMARBLE": (_whole("mc10"), "GSTONE"), "BRICK": (_whole("pbrick8"), "BROWN"),
    "PLATE": (_whole("w33_5"), "TECHGRY"), "DUCT": (_whole("aqpipe08"), "PIPES"),
    "CLIFF": (_whole("wall63_1"), "STONE"), "RIVET": (_whole("wall62_1"), "BRNPANEL"),
}

FLATS: dict[str, tuple] = {
    "FLOOR4": (_flat("floor4_8"), "FLOOR4"), "FLOOR5": (_flat("floor5_1"), "FLOOR5"),
    "COMPFLR": (_flat("ceil4_2"), "COMPFLR"), "CEIL3": (_flat("ceil3_5"), "CEIL3"),
    "TLITE": (_flat("tlite6_4"), "TLITE"), "LIGHTSQ": (_flat("flat17"), "LIGHTSQ"),
    "NUKAGE": (_flat("nukage1", "nukage2", "nukage3"), "NUKAGE"),
    "BLOOD": (_flat("blood1", "blood2", "blood3"), "BLOOD"),
    "LAVA": (_flat("lava1", "lava2", "lava3", "lava4"), "LAVA"), "ROCK": (_flat("flat5_7"), "ROCK"),
    "DIRT": (_flat("flat10"), "DIRT"), "STEPTOP": (_flat("flat23"), "STEPTOP"),
    "EXITPAD": (_flat("gate3"), "EXITPAD"), "MARBFLR": (_flat("dem1_5"), "MARBFLR"),
    "METAL": (_flat("ceil5_1"), "METAL"), "HELLROCK": (_flat("rrock05"), "HELLROCK"),
    # flats added with the pictures
    "WATER": (_flat("fwater1", "fwater2", "fwater3", "fwater4"), "NUKAGE"), "CRATOP": (_flat("cratop1"), "DIRT"),
    "GRASS": (_flat("grass1"), "DIRT"), "SLAB": (_flat("slime14"), "FLOOR5"), "HEXFLR": (_flat("aqf076"), "FLOOR4"),
    "GRATE": (_flat("aqf054"), "METAL"), "HAZFLR": (_flat("aqf068"), "STEPTOP"),
    "SKULLFLR": (_flat("flat5_6"), "HELLROCK"), "FLESHFLR": (_flat("sflr6_1"), "HELLROCK"),
    "TILE": (_flat("floor0_6"), "FLOOR5"), "PLATE": (_flat("flat20"), "FLOOR4"), "CAVEFLR": (_flat("rrock09"), "ROCK"),
    "PAD2": (_flat("gate4"), "EXITPAD"), "BLUEFLR": (_flat("flat14"), "COMPFLR"), "ASH": (_flat("rrock03"), "ROCK"),
    "MAGMA": (_flat("rrock01"), "LAVA"), "DARKTILE": (_flat("mflr8_2"), "FLOOR5"),
    "WHITELIT": (_flat("flat2"), "LIGHTSQ"), "REDLITE": (_flat("tlite6_5"), "TLITE"),
    "CIRCUIT": (_flat("cons1_1"), "COMPFLR"), "WOODFLR": (_flat("flat5_5"), "DIRT"),
}


def _from_palette(idx: np.ndarray) -> np.ndarray:
    rgb = PALETTE[idx]
    return np.dstack([rgb, np.full(idx.shape, 255, np.uint8)])


class Tiles:
    """Every wall texture and flat as 128 x 128 layers (animated ones have several); ``wall_ids`` / ``flat_ids``
    name a texture, ``layer_of(tick)`` gives each id's current layer."""

    def __init__(self) -> None:
        self.layers: list[np.ndarray] = []
        self.wall_ids: dict[str, int] = {}
        self.flat_ids: dict[str, int] = {}
        self._first: list[int] = []
        self._count: list[int] = []
        self._proc = None
        for table, ids, walls in ((WALLS, self.wall_ids, True), (FLATS, self.flat_ids, False)):
            for name, (recipe, fallback) in table.items():
                frames = recipe()
                if not frames:
                    frames = self._procedural(fallback, walls)
                ids[name] = len(self._first)
                self._first.append(len(self.layers))
                self._count.append(len(frames))
                self.layers.extend(frames)
        self.first = np.array(self._first, np.int32)
        self.count = np.array(self._count, np.int32)
        self.n_ids = len(self._first)
        self.average = np.array([ly[..., :3].reshape(-1, 3).mean(axis=0) for ly in self.layers], np.float32)

    def _procedural(self, name: str, walls: bool) -> list[np.ndarray]:
        if self._proc is None:
            from .textures import textures
            self._proc = textures()
        ts = self._proc
        if walls:
            first, count = ts.wall_anim[ts.wall_ids[name]]
            return [tile(_from_palette(ts.walls[first + i])) for i in range(count)]
        first, count = ts.flat_anim[ts.flat_ids[name]]
        return [tile(_from_palette(ts.flats[first + i])) for i in range(count)]

    def layer_of(self, tick: int) -> np.ndarray:
        return self.first + (tick // 8) % self.count

    def colour(self, tex_id: int) -> tuple[int, int, int]:
        """The average colour of a texture (the automap)."""
        c = self.average[self._first[tex_id]]
        return int(c[0]), int(c[1]), int(c[2])

    def mips(self) -> list[np.ndarray]:
        """The layers as (n, s, s, 4) arrays for s = 128, 64, ... 1 (each a box filter of the last)."""
        level = np.stack(self.layers).astype(np.float32)
        out = [level.astype(np.uint8)]
        while level.shape[1] > 1:
            level = (level[:, 0::2, 0::2] + level[:, 1::2, 0::2] + level[:, 0::2, 1::2] + level[:, 1::2, 1::2]) / 4
            out.append(np.clip(level + 0.5, 0, 255).astype(np.uint8))
        return out


# --- the sky ----------------------------------------------------------------------------------------------------------

SKY_W, SKY_H = 4096, 768
SKIES = {
    # zenith, horizon, cloud, cloud rim, ridge, stars
    "dusk": ((14, 16, 38), (214, 104, 52), (38, 26, 52), (250, 150, 80), (16, 12, 20), 0.9),
    "toxic": ((8, 20, 20), (132, 160, 70), (34, 48, 40), (190, 220, 110), (10, 16, 12), 0.5),
    "hell": ((26, 0, 2), (240, 70, 14), (34, 4, 4), (255, 150, 40), (8, 0, 0), 0.0),
    "night": ((4, 6, 16), (40, 52, 96), (16, 20, 36), (90, 110, 170), (4, 5, 10), 1.0),
}
_sky_cache: dict[str, np.ndarray] = {}


def _fbm(h: int, w: int, cells: int, seed: int, octaves: int = 5) -> np.ndarray:
    """Value noise in [0, 1] that wraps around horizontally."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    for o in range(octaves):
        cx = cells * 2 ** o
        cy = max(int(cx * h / w * 2), 2)
        g = rng.random((cy + 1, cx), dtype=np.float32)
        fy = ys / h * cy
        fx = xs / w * cx
        y0 = np.minimum(fy.astype(np.int32), cy - 1)
        x0 = fx.astype(np.int32) % cx
        ty = (fy - y0)[:, None]
        tx = (fx - np.floor(fx))[None, :]
        ty = ty * ty * (3 - 2 * ty)
        tx = tx * tx * (3 - 2 * tx)
        x1 = (x0 + 1) % cx
        a = g[y0][:, x0] * (1 - tx) + g[y0][:, x1] * tx
        b = g[y0 + 1][:, x0] * (1 - tx) + g[y0 + 1][:, x1] * tx
        out += (a * (1 - ty) + b * ty) * amp
        total += amp
        amp *= 0.5
    return out / total


def sky(theme: str) -> np.ndarray:
    """A 360 degree panorama (SKY_H x SKY_W RGBA): row 0 high in the sky, the horizon at 70 % of the height."""
    if theme not in SKIES:
        theme = "dusk"
    if theme in _sky_cache:
        return _sky_cache[theme]
    zen, hor, cloud, rim, ridge, stars = (np.array(c, np.float32) if isinstance(c, tuple) else c
                                          for c in SKIES[theme])
    h, w = SKY_H, SKY_W
    horizon = int(h * 0.7)
    seed = sum(ord(c) for c in theme)
    t = np.clip(np.arange(h, dtype=np.float32) / horizon, 0, 1)[:, None, None] ** 1.7
    img = zen[None, None, :] * (1 - t) + hor[None, None, :] * t
    img = np.broadcast_to(img, (h, w, 3)).copy()
    rng = np.random.default_rng(seed)
    if stars > 0:
        n = int(2600 * stars)
        sy = (rng.random(n) ** 1.6 * horizon * 0.8).astype(np.int32)
        sx = rng.integers(0, w, n)
        br = (rng.random(n) ** 3 * 200 + 40)[:, None] * (1 - sy[:, None] / horizon)
        img[sy, sx] = np.maximum(img[sy, sx], br * np.array([1.0, 0.95, 0.85], np.float32))
        big = rng.random(n) > 0.9
        img[sy[big], (sx[big] + 1) % w] = np.maximum(img[sy[big], (sx[big] + 1) % w], br[big] * 0.6)
    n1 = _fbm(h, w, 6, seed + 1)
    n2 = _fbm(h, w, 14, seed + 2, 4)
    dens = np.clip((n1 * 0.7 + n2 * 0.3 - 0.44) * 3.4, 0, 1)
    dens *= np.clip(np.arange(h, dtype=np.float32) / (horizon * 0.25), 0, 1)[:, None] ** 0.5
    lit = np.clip((np.roll(n1, -6, axis=0) - n1) * 9 + 0.25, 0, 1)[..., None]
    ccol = cloud[None, None, :] * (1 - lit) + rim[None, None, :] * lit
    img = img * (1 - dens[..., None]) + ccol * dens[..., None]
    # a far ridge of rock along the horizon, a nearer darker one in front of it
    xs = np.arange(w, dtype=np.float32) / w * 2 * np.pi
    for k, (height, col_f) in enumerate(((0.16, 0.55), (0.10, 1.0))):
        prof = np.zeros(w, np.float32)
        for i in range(1, 9):
            f = rng.integers(2, 6) * i
            prof += np.sin(xs * f + rng.random() * 6.28) / i ** 1.1
        prof = (prof - prof.min()) / (prof.max() - prof.min())
        jag = _fbm(2, w, 64, seed + 7 + k, 4)[0]
        top = horizon - (prof * 0.75 + jag * 0.25) * h * height - h * 0.012
        rows = np.arange(h, dtype=np.float32)[:, None]
        cover = np.clip(rows - top[None, :], 0, 1)[..., None]
        base = ridge[None, None, :] * col_f + hor[None, None, :] * (1 - col_f) * 0.45
        img = img * (1 - cover) + base * cover
    img[horizon:] = img[horizon:] * 0.35 + ridge[None, None, :] * 0.65
    out = np.dstack([np.clip(img, 0, 255).astype(np.uint8), np.full((h, w), 255, np.uint8)])
    _sky_cache[theme] = out
    return out


# --- sprites ----------------------------------------------------------------------------------------------------------

# monster kind -> (Freedoom sprite name, {frame group: frame letters})
MONSTER_FD = {
    "zombie": ("poss", {"walk": "abcd", "attack": "efe", "pain": "g", "death": "hijkl", "gib": "mnopqrstu"}),
    "sergeant": ("spos", {"walk": "abcd", "attack": "efe", "pain": "g", "death": "hijkl", "gib": "mnopqrstu"}),
    "heavy": ("cpos", {"walk": "abcd", "attack": "efe", "pain": "g", "death": "hijklmn", "gib": "opqrst"}),
    "imp": ("troo", {"walk": "abcd", "attack": "efg", "pain": "h", "death": "ijklm", "gib": "nopqrstu"}),
    "demon": ("sarg", {"walk": "abcd", "attack": "efg", "pain": "h", "death": "ijklmn"}),
    "shade": ("sarg", {"walk": "abcd", "attack": "efg", "pain": "h", "death": "ijklmn"}),
    "soul": ("skul", {"walk": "ab", "attack": "cd", "pain": "e", "death": "fghijk"}),
    "caco": ("head", {"walk": "a", "attack": "bcd", "pain": "ef", "death": "ghijkl"}),
    "baron": ("boss", {"walk": "abcd", "attack": "efg", "pain": "h", "death": "ijklmno"}),
    "titan": ("cybr", {"walk": "abcd", "attack": "efe", "pain": "g", "death": "hijklmnop"}),
}
# what the procedural models stand in with for monsters they do not have
MONSTER_PROC_ALIAS = {"heavy": "sergeant", "titan": "baron", "shade": "demon"}

# thing -> Freedoom frames ("name" + frame letter + rotation); a frame of 8 sides is given as its name + letter
THING_FD = {
    "hbonus": ["bon1a0", "bon1b0", "bon1c0", "bon1d0", "bon1c0", "bon1b0"],
    "abonus": ["bon2a0", "bon2b0", "bon2c0", "bon2d0", "bon2c0", "bon2b0"],
    "stim": ["stima0"], "medikit": ["media0"], "garmor": ["arm1a0", "arm1b0"], "barmor": ["arm2a0", "arm2b0"],
    "soulsphere": ["soula0", "soulb0", "soulc0", "sould0", "soulc0", "soulb0"],
    "clip": ["clipa0"], "ammobox": ["ammoa0"], "shells": ["shela0"], "shellbox": ["sboxa0"], "rocket": ["rocka0"],
    "rocketbox": ["broka0"], "cell": ["cella0"], "cellpack": ["celpa0"], "backpack": ["bpaka0"],
    "bluecard": ["bkeya0", "bkeyb0"], "yellowcard": ["ykeya0", "ykeyb0"], "redcard": ["rkeya0", "rkeyb0"],
    "w_shotgun": ["shota0"], "w_chaingun": ["mguna0"], "w_launcher": ["launa0"], "w_plasma": ["plasa0"],
    "w_bfg": ["bfuga0"], "w_saw": ["csawa0"],
    "berserk": ["pstra0"], "invuln": ["pinva0", "pinvb0", "pinvc0", "pinvd0"], "suit": ["suita0"],
    "scanner": ["pmapa0", "pmapb0", "pmapc0", "pmapd0", "pmapc0", "pmapb0"],
    "techlamp": ["tlmpa0", "tlmpb0", "tlmpc0", "tlmpd0"], "lamp": ["tlp2a0", "tlp2b0", "tlp2c0", "tlp2d0"],
    "torch": ["treda0", "tredb0", "tredc0", "tredd0"], "torch_g": ["tgrna0", "tgrnb0", "tgrnc0", "tgrnd0"],
    "torch_b": ["tblua0", "tblub0", "tbluc0", "tblud0"], "column": ["colua0"], "candle": ["canda0"],
    "candelabra": ["cbraa0"], "pylon": ["eleca0"], "pillar": ["col1a0"], "pillar_r": ["col3a0"],
    "firecan": ["fcana0", "fcanb0", "fcanc0"], "tree": ["tre1a0"], "deadguy": ["possl0"], "bloodpool": ["pob1a0"],
    "gibs": ["pol5a0"],
    "barrel": ["bar1a0", "bar1b0"], "barrel_x": ["bexpa0", "bexpb0", "bexpc0", "bexpd0", "bexpe0"],
    "imp_ball": ["bal1a0", "bal1b0"], "imp_x": ["bal1c0", "bal1d0", "bal1e0"],
    "caco_ball": ["bal2a0", "bal2b0"], "caco_x": ["bal2c0", "bal2d0", "bal2e0"],
    "baron_ball": ["bal7a", "bal7b"], "baron_x": ["bal7c0", "bal7d0", "bal7e0"],
    "rocket_fly": ["misla"], "rocket_x": ["mislb0", "mislc0", "misld0"],
    "plasma_ball": ["plssa0", "plssb0"], "plasma_x": ["plsea0", "plseb0", "plsec0", "plsed0", "plsee0"],
    "bfg_ball": ["bfs1a0", "bfs1b0"], "bfg_x": ["bfe1a0", "bfe1b0", "bfe1c0", "bfe1d0", "bfe1e0", "bfe1f0"],
    "tracer_x": ["bfe2a0", "bfe2b0", "bfe2c0", "bfe2d0"],
    "puff": ["puffa0", "puffb0", "puffc0", "puffd0"], "blood": ["bludc0", "bludb0", "bluda0"],
    "tfog": ["tfoga0", "tfogb0", "tfogc0", "tfogd0", "tfoge0", "tfogf0", "tfogg0", "tfogh0", "tfogi0", "tfogj0"],
}
# things the procedural art draws under another name
THING_PROC_ALIAS = {
    "torch_g": "torch", "torch_b": "torch", "column": "techlamp", "candle": "lamp", "candelabra": "lamp",
    "pylon": "techlamp", "pillar": "techlamp", "pillar_r": "techlamp", "firecan": "torch", "tree": "techlamp",
    "gibs": "bloodpool", "tfog": "plasma_x", "w_saw": "w_chaingun", "berserk": "medikit", "invuln": "soulsphere",
    "suit": "garmor", "scanner": "cellpack",
}
# anchored at their middle (they fly); the rest stand on their bottom edge
CENTRED = {"imp_ball", "imp_x", "caco_ball", "caco_x", "baron_ball", "baron_x", "rocket_fly", "rocket_x",
           "plasma_ball", "plasma_x", "bfg_ball", "bfg_x", "tracer_x", "puff", "blood", "tfog", "barrel_x"}
BRIGHT = {"techlamp", "lamp", "torch", "torch_g", "torch_b", "candle", "candelabra", "pylon", "firecan",
          "soulsphere", "invuln", "scanner", "barrel_x", "imp_ball", "imp_x", "caco_ball", "caco_x", "baron_ball",
          "baron_x", "rocket_fly", "rocket_x", "plasma_ball", "plasma_x", "bfg_ball", "bfg_x", "tracer_x", "tfog"}

# first-person weapons on a 320 x 200 screen: frames (picture, centre x, bottom y) and muzzle flashes
# (picture, centre x, bottom y, drawn instead of the gun?)
WEAPON_FD = {
    "fist": {"frames": [("punga0", 176, 206), ("pungb0", 160, 204), ("pungc0", 214, 206), ("pungd0", 204, 206)],
             "flash": []},
    "saw": {"frames": [("sawgc0", 172, 212), ("sawgd0", 172, 214), ("sawga0", 170, 212), ("sawgb0", 170, 210)],
            "flash": []},
    "pistol": {"frames": [("pisga0", 166, 204), ("pisgb0", 166, 206), ("pisgc0", 166, 204), ("pisgd0", 166, 206),
                          ("pisge0", 166, 208)],
               "flash": [("pisfa0", 166, 150, False)]},
    "shotgun": {"frames": [("shtga0", 160, 204), ("shtgb0", 176, 228), ("shtgc0", 170, 236), ("shtgd0", 180, 230)],
                "flash": [("shtfa0", 160, 156, False), ("shtfb0", 160, 158, False)]},
    "chaingun": {"frames": [("chgga0", 160, 204), ("chggb0", 160, 204)],
                 "flash": [("chgfa0", 160, 166, False), ("chgfb0", 160, 162, False)]},
    "launcher": {"frames": [("misga0", 160, 204), ("misgb0", 160, 206)],
                 "flash": [("misfa0", 160, 168, False), ("misfb0", 160, 170, False), ("misfc0", 160, 166, False),
                           ("misfd0", 160, 164, False)]},
    "plasma": {"frames": [("plsga0", 176, 204), ("plsgb0", 174, 206)],
               "flash": [("plsfa0", 176, 204, True), ("plsfb0", 176, 204, True)]},
    "bfg": {"frames": [("bfgga0", 160, 204), ("bfggb0", 160, 204), ("bfggc0", 160, 204)],
            "flash": [("bfgfa0", 160, 204, True), ("bfgfb0", 160, 204, True)]},
}
# the procedural weapon pictures (sprites.Art.weapons) that stand for each frame above
WEAPON_PROC = {
    "fist": ("fist", [0, 1, 2, 2]), "saw": ("fist", [0, 0, 1, 2]), "pistol": ("pistol", [0, 1, 2, 2, 2]),
    "shotgun": ("shotgun", [0, 2, 3, 2]), "chaingun": ("chaingun", [0, 2]), "launcher": ("launcher", [0, 2]),
    "plasma": ("plasma", [0, 2]), "bfg": ("bfg", [0, 2, 2]),
}
WEAPON_PROC_FLASH = {"pistol": [1], "shotgun": [1, 1], "chaingun": [1, 1], "launcher": [1, 1, 1, 1],
                     "plasma": [1, 1], "bfg": [1, 1]}

_FRAME_RE = re.compile(r"^([a-z0-9]{4})([a-z\[\]\^])([0-9])(?:([a-z\[\]\^])([0-9]))?$")
# sprite side shown for each of the eight view directions (index: see ``side_index``), for a procedural frame of
# five views (front, 45, 90, 135 degrees, back): (view, mirrored)
_PROC_SIDES = ((0, False), (1, True), (2, True), (3, True), (4, False), (3, False), (2, False), (1, False))


def side_index(rel: float) -> int:
    """Which of a sprite's eight sides the viewer sees: ``rel`` = angle from the thing's facing to the viewer
    (radians, map angles: clockwise on the map). 0 = its front, 2 = its left side, 4 = its back, 6 = its right."""
    return int(math.floor(((-math.degrees(rel) + 22.5) % 360.0) / 45.0)) % 8


def _bleed(rgba: np.ndarray, passes: int = 2) -> np.ndarray:
    """Spread the colour of the outline into the see-through pixels beside it (no dark fringe when filtered)."""
    out = rgba.copy()
    solid = out[..., 3] >= 128
    out[~solid] = 0
    known = solid.copy()
    for _ in range(passes):
        acc = np.zeros(out.shape[:2] + (3,), np.float32)
        cnt = np.zeros(out.shape[:2], np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src = np.roll(out[..., :3].astype(np.float32), (dy, dx), axis=(0, 1))
            k = np.roll(known, (dy, dx), axis=(0, 1))
            if dy == 1:
                k[0] = False
            elif dy == -1:
                k[-1] = False
            elif dx == 1:
                k[:, 0] = False
            else:
                k[:, -1] = False
            acc += src * k[..., None]
            cnt += k
        new = (~known) & (cnt > 0)
        out[new, :3] = (acc[new] / cnt[new][:, None]).astype(np.uint8)
        known |= new
    out[..., 3] = np.where(solid, 255, 0)
    return out


class Sprites:
    """Every sprite in one atlas. ``rect[id]`` = (u0, v0, u1, v1, width, height, x offset, y offset): the picture's
    corner coordinates in the atlas (0..1), its size in map units, the column of the thing's centre and the height
    of the picture's bottom above the thing's feet.

    ``things[key]`` and ``monsters[kind][group]`` are lists of frames; a frame is (ids, mirrored): two arrays of
    eight, one entry per side (``side_index``). ``weapons[name]`` = (frames, flashes) of (id, left, top, instead)
    on a 320 x 200 screen."""

    def __init__(self) -> None:
        self._images: list[np.ndarray] = []
        self._offs: list[tuple[float, float]] = []
        self._by_key: dict[object, int] = {}
        self.things: dict[str, list] = {}
        self.monsters: dict[str, dict[str, list]] = {}
        self.weapons: dict[str, tuple[list, list]] = {}
        self.bright: set[str] = set(BRIGHT)
        self.freedoom = have_freedoom()
        self._proc_art = None
        self._fd_index: dict[str, dict] = {}
        for key in THING_FD:
            self.things[key] = self._thing(key)
        for kind in MONSTER_FD:
            self.monsters[kind] = self._monster(kind)
        for name in WEAPON_FD:
            self.weapons[name] = self._weapon(name)
        self._pack()

    # --- adding pictures ----------------------------------------------------------------------------------------

    def _add(self, key: object, rgba: np.ndarray, xoff: float | None = None, yoff: float = 0.0) -> int:
        if key in self._by_key:
            return self._by_key[key]
        i = len(self._images)
        self._images.append(rgba)
        self._offs.append((rgba.shape[1] / 2 if xoff is None else xoff, yoff))
        self._by_key[key] = i
        return i

    def _fd(self, name: str, centred: bool = False) -> int | None:
        a = png("sprites", name)
        if a is None:
            return None
        return self._add(("fd", name), a, None, -a.shape[0] / 2 if centred else 0.0)

    def _fd_sides(self, sprite: str, letter: str, centred: bool = False):  # noqa: ANN202
        """A frame of a Freedoom sprite: (ids, mirrored) for the eight sides, or None when a picture is missing."""
        idx = self._fd_index.get(sprite)
        if idx is None:
            idx = self._fd_index[sprite] = {}
            folder = FD / "sprites"
            if folder.is_dir():
                for path in folder.glob(sprite + "*.png"):
                    m = _FRAME_RE.match(path.stem)
                    if not m:
                        continue
                    idx[(m.group(2), int(m.group(3)))] = (path.stem, False)
                    if m.group(4):
                        idx[(m.group(4), int(m.group(5)))] = (path.stem, True)
        if (letter, 0) in idx:
            i = self._fd(idx[(letter, 0)][0], centred)
            if i is None:
                return None
            return np.full(8, i, np.int32), np.zeros(8, bool)
        ids, flips = np.zeros(8, np.int32), np.zeros(8, bool)
        for r in range(8):
            entry = idx.get((letter, r + 1))
            if entry is None:
                return None
            i = self._fd(entry[0], centred)
            if i is None:
                return None
            ids[r], flips[r] = i, entry[1]
        return ids, flips

    def _proc(self):  # noqa: ANN202 - the procedural 2D art, made the first time it is needed
        if self._proc_art is None:
            from .sprites import art
            self._proc_art = art()
        return self._proc_art

    def _add_proc(self, key: object, s) -> int:  # noqa: ANN001 - a models.Sprite (palette indices + mask)
        rgba = _from_palette(s.img)
        rgba[..., 3] = np.where(s.mask, 255, 0)
        return self._add(key, rgba, float(s.xoff), float(s.yoff))

    @staticmethod
    def _all(i: int):  # noqa: ANN205
        return np.full(8, i, np.int32), np.zeros(8, bool)

    def _thing(self, key: str) -> list:
        frames = []
        for name in THING_FD[key]:
            if len(name) == 5:  # a frame of eight sides
                f = self._fd_sides(name[:4], name[4], key in CENTRED)
            else:
                i = self._fd(name, key in CENTRED)
                f = None if i is None else self._all(i)
            if f is None:
                frames = []
                break
            frames.append(f)
        if frames:
            return frames
        pkey = THING_PROC_ALIAS.get(key, key)
        lst = self._proc().sprites.get(pkey) or self._proc().sprites["puff"]
        return [self._all(self._add_proc(("proc", pkey, n), s)) for n, s in enumerate(lst)]

    def _monster(self, kind: str) -> dict[str, list]:
        sprite, groups = MONSTER_FD[kind]
        out: dict[str, list] = {}
        ok = True
        for group, letters in groups.items():
            frames = []
            for letter in letters:
                f = self._fd_sides(sprite, letter)
                if f is None:
                    ok = False
                    break
                frames.append(f)
            if not ok:
                break
            out[group] = frames
        if ok:
            return out
        from .models import monster_frames
        pkind = MONSTER_PROC_ALIAS.get(kind, kind)
        out = {}
        for group, lst in monster_frames(pkind).items():
            frames = []
            for n, item in enumerate(lst):
                if isinstance(item, list):
                    ids = [self._add_proc(("mon", pkind, group, n, v), s) for v, s in enumerate(item)]
                    frames.append((np.array([ids[v] for v, _ in _PROC_SIDES], np.int32),
                                   np.array([m for _, m in _PROC_SIDES], bool)))
                else:
                    frames.append(self._all(self._add_proc(("mon", pkind, group, n), item)))
            out[group] = frames
        return out

    def _weapon(self, name: str) -> tuple[list, list]:
        spec = WEAPON_FD[name]
        frames, flashes = [], []
        ok = True
        for pic, cx, by in spec["frames"]:
            a = png("sprites", pic)
            if a is None:
                ok = False
                break
            i = self._add(("fd", pic), a)
            frames.append((i, cx - a.shape[1] / 2, by - a.shape[0], False))
        if ok:
            for pic, cx, by, instead in spec["flash"]:
                a = png("sprites", pic)
                if a is None:
                    continue
                i = self._add(("fd", pic), a)
                flashes.append((i, cx - a.shape[1] / 2, by - a.shape[0], instead))
            return frames, flashes
        from .weapons import WEAPON_TOP
        pname, order = WEAPON_PROC[name]
        lst = self._proc().weapons[pname]
        made = []
        for n, s in enumerate(lst):
            i = self._add_proc(("wpn", pname, n), s)
            made.append((i, 160 - float(s.xoff), float(WEAPON_TOP.get(pname, 168 - s.img.shape[0])) + 32, True))
        frames = [made[min(k, len(made) - 1)] for k in order]
        flashes = [made[min(k, len(made) - 1)] for k in WEAPON_PROC_FLASH.get(name, [])]
        return frames, flashes

    # --- the atlas ----------------------------------------------------------------------------------------------

    def _pack(self) -> None:
        order = sorted(range(len(self._images)), key=lambda i: -self._images[i].shape[0])
        x = y = PAD
        row_h = 0
        place = [(0, 0)] * len(order)
        for i in order:
            h, w = self._images[i].shape[:2]
            if x + w + PAD > ATLAS_W:
                x = PAD
                y += row_h + PAD
                row_h = 0
            place[i] = (x, y)
            x += w + PAD
            row_h = max(row_h, h)
        height = y + row_h + PAD
        height = (height + 63) // 64 * 64
        atlas = np.zeros((height, ATLAS_W, 4), np.uint8)
        rect = np.zeros((len(order), 8), np.float32)
        for i, (px, py) in enumerate(place):
            img = self._images[i]
            h, w = img.shape[:2]
            atlas[py:py + h, px:px + w] = img
            rect[i] = (px / ATLAS_W, py / height, (px + w) / ATLAS_W, (py + h) / height, w, h, *self._offs[i])
        self.atlas = _bleed(atlas)
        self.rect = rect
        self.size = (ATLAS_W, height)
        self._qimages: dict[int, QImage] = {}

    def image(self, sprite_id: int) -> QImage:
        """A sprite as a QImage (the HUD's icons)."""
        img = self._qimages.get(sprite_id)
        if img is None:
            img = self._qimages[sprite_id] = rgba_to_qimage(self._images[sprite_id])
        return img

    def icon(self, key: str, frame: int = 0) -> QImage:
        lst = self.things.get(key)
        if not lst:
            return QImage()
        return self.image(int(lst[frame % len(lst)][0][0]))


_TILES: Tiles | None = None
_SPRITES: Sprites | None = None


def tiles() -> Tiles:
    global _TILES
    if _TILES is None:
        _TILES = Tiles()
    return _TILES


def sprites() -> Sprites:
    global _SPRITES
    if _SPRITES is None:
        _SPRITES = Sprites()
    return _SPRITES
