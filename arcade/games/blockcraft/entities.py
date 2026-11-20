"""Things that move: the player (Minecraft's 20 Hz movement physics: friction, sprint, sneak edges, swimming,
creative flight, fall damage), animals and monsters with box models and simple AI (each animal has its biomes: see
HABITAT), dropped items, primed TNT, falling sand and gravel, and particles."""

from __future__ import annotations

import math

import numpy as np

from app.arcade.games.blockcraft import blocks as B

TICK = 0.05  # s: the game ticks 20 times a second, like the real one


def _clip(boxes: list, box: list, d: float, axis: int) -> float:
    """Minecraft's collision clip: how far a box may move along an axis past the block boxes."""
    a, b = (axis + 1) % 3, (axis + 2) % 3
    for bb in boxes:
        if bb[a + 3] <= box[a] or bb[a] >= box[a + 3] or bb[b + 3] <= box[b] or bb[b] >= box[b + 3]:
            continue
        if d > 0 and box[axis + 3] <= bb[axis]:
            d = min(d, bb[axis] - box[axis + 3])
        elif d < 0 and box[axis] >= bb[axis + 3]:
            d = max(d, bb[axis + 3] - box[axis])
    return d


class Body:
    """An axis-aligned box that moves and collides with blocks (feet position x, y, z)."""

    width = 0.6
    height = 1.8

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z
        self.px, self.py, self.pz = x, y, z  # the previous tick (drawing interpolates)
        self.vx = self.vy = self.vz = 0.0
        self.on_ground = False
        self.hit_wall = False
        self.fall = 0.0
        self.in_water = False
        self.in_lava = False
        self.yaw = 0.0

    def box(self) -> list:
        hw = self.width / 2
        return [self.x - hw, self.y, self.z - hw, self.x + hw, self.y + self.height, self.z + hw]

    def lerp(self, a: float) -> tuple[float, float, float]:
        return (self.px + (self.x - self.px) * a, self.py + (self.y - self.py) * a, self.pz + (self.z - self.pz) * a)

    def remember(self) -> None:
        self.px, self.py, self.pz = self.x, self.y, self.z

    def _boxes(self, world, box: list, dx: float, dy: float, dz: float) -> list:  # noqa: ANN001
        return world.solid_boxes(min(box[0], box[0] + dx) - 0.01, min(box[1], box[1] + dy) - 0.01,
                                 min(box[2], box[2] + dz) - 0.01, max(box[3], box[3] + dx) + 0.01,
                                 max(box[4], box[4] + dy) + 0.01, max(box[5], box[5] + dz) + 0.01)

    def move(self, world, dx: float, dy: float, dz: float, sneak: bool = False) -> None:  # noqa: ANN001
        box = self.box()
        if sneak and self.on_ground:  # never walk off an edge while sneaking
            step = 0.05

            def floorless(ox: float, oz: float) -> bool:
                b = [box[0] + ox, box[1] - 1.0, box[2] + oz, box[3] + ox, box[4] - 1.0, box[5] + oz]
                return not any(_overlaps(b, bb) for bb in world.solid_boxes(b[0], b[1], b[2], b[3], b[4], b[5]))

            while dx and floorless(dx, 0):
                dx = 0.0 if abs(dx) < step else dx - math.copysign(step, dx)
            while dz and floorless(0, dz):
                dz = 0.0 if abs(dz) < step else dz - math.copysign(step, dz)
            while dx and dz and floorless(dx, dz):
                dx = 0.0 if abs(dx) < step else dx - math.copysign(step, dx)
                dz = 0.0 if abs(dz) < step else dz - math.copysign(step, dz)
        odx, ody, odz = dx, dy, dz
        boxes = self._boxes(world, box, dx, dy, dz)
        dy = _clip(boxes, box, dy, 1)
        box[1] += dy
        box[4] += dy
        dx = _clip(boxes, box, dx, 0)
        box[0] += dx
        box[3] += dx
        dz = _clip(boxes, box, dz, 2)
        box[2] += dz
        box[5] += dz
        self.on_ground = ody < 0 and dy != ody
        self.hit_wall = dx != odx or dz != odz
        if dx != odx:
            self.vx = 0.0
        if dz != odz:
            self.vz = 0.0
        if dy != ody:
            self.vy = 0.0
        self.x = (box[0] + box[3]) / 2
        self.y = box[1]
        self.z = (box[2] + box[5]) / 2
        if dy < 0 and not self.on_ground:
            self.fall -= dy

    def check_liquids(self, world) -> None:  # noqa: ANN001
        b = self.box()
        found = world.blocks_in(b[0] + 0.001, b[1] + 0.001, b[2] + 0.001, b[3] - 0.001, b[1] + 0.6, b[5] - 0.001)
        self.in_water = not found.isdisjoint(B.WATERY_SET)
        self.in_lava = B.LAVA in found

    def move_relative(self, strafe: float, forward: float, friction: float) -> None:
        f = strafe * strafe + forward * forward
        if f < 1e-4:
            return
        f = max(math.sqrt(f), 1.0)
        f = friction / f
        strafe *= f
        forward *= f
        s, c = math.sin(math.radians(self.yaw)), math.cos(math.radians(self.yaw))
        self.vx += strafe * c - forward * s
        self.vz += forward * c + strafe * s


def _overlaps(a: list, b: tuple) -> bool:
    return a[0] < b[3] and a[3] > b[0] and a[1] < b[4] and a[4] > b[1] and a[2] < b[5] and a[5] > b[2]


class Player(Body):
    """The player: Minecraft's movement constants per 20 Hz tick."""

    EYE = 1.62

    def __init__(self, x: float, y: float, z: float) -> None:
        super().__init__(x, y, z)
        self.pitch = 0.0
        self.flying = False
        self.sprinting = False
        self.sneaking = False
        self.walked = 0.0  # distance walked (view bobbing, footsteps)
        self.prev_walked = 0.0
        self.bob = 0.0
        self.prev_bob = 0.0
        self.landed_fall = 0.0  # set when landing: the fall distance (damage, sounds)
        self.jumped = False

    def eye_height(self) -> float:
        return self.EYE - (0.08 if self.sneaking and not self.flying else 0.0)

    def tick(self, world, forward: float, strafe: float, jump: bool, sneak: bool, sprint: bool,  # noqa: ANN001
             creative: bool) -> None:
        self.remember()
        self.prev_walked = self.walked
        self.prev_bob = self.bob
        self.check_liquids(world)
        self.sneaking = sneak and not self.flying
        forward *= 0.98
        strafe *= 0.98
        if self.sneaking:
            forward *= 0.3
            strafe *= 0.3
        if forward <= 0 or self.sneaking or self.hit_wall and not self.flying:
            sprint = False
        self.sprinting = sprint and forward > 0
        self.jumped = False
        self.landed_fall = 0.0
        x0, z0 = self.x, self.z
        if self.flying and creative:
            if jump:
                self.vy += 0.15
            if sneak:
                self.vy -= 0.15
            self.move_relative(strafe, forward, 0.1 if self.sprinting else 0.05)
            self.move(world, self.vx, self.vy, self.vz)
            self.vy *= 0.6
            self.vx *= 0.91
            self.vz *= 0.91
            self.fall = 0.0
            if self.on_ground:
                self.flying = False
        elif self.in_water or self.in_lava:
            if jump:
                self.vy += 0.04
            self.move_relative(strafe, forward, 0.02)
            self.move(world, self.vx, self.vy, self.vz)
            k = 0.8 if self.in_water else 0.5
            self.vx *= k
            self.vy *= k
            self.vz *= k
            self.vy -= 0.02
            if self.hit_wall and jump:  # climb out onto the bank
                self.vy = 0.3
            self.fall = 0.0
        else:
            if jump and self.on_ground:
                self.vy = 0.42
                self.jumped = True
                if self.sprinting:
                    y = math.radians(self.yaw)
                    self.vx -= math.sin(y) * 0.2
                    self.vz += math.cos(y) * 0.2
            slip = 0.6 * 0.91 if self.on_ground else 0.91
            if self.on_ground:
                accel = 0.1 * (1.3 if self.sprinting else 1.0) * (0.16277136 / slip ** 3)
            else:
                accel = 0.026 if self.sprinting else 0.02
            self.move_relative(strafe, forward, accel)
            was_falling = self.fall
            self.move(world, self.vx, self.vy, self.vz, sneak=self.sneaking)
            if self.on_ground:
                self.landed_fall = was_falling if was_falling > 0 else 0.0
                self.fall = 0.0
            self.vy = (self.vy - 0.08) * 0.98
            self.vx *= slip
            self.vz *= slip
        dist = math.hypot(self.x - x0, self.z - z0)
        if self.on_ground and not self.flying:
            self.walked += dist
            self.bob += (min(dist * 20 / 4.3, 1.0) - self.bob) * 0.4
        else:
            self.bob *= 0.6


# --- models ------------------------------------------------------------------------------------------------------

def _face_tex(colour: tuple, seed: int, n: int = 8, noise: float = 10.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.array(colour, np.float32) + rng.normal(0, noise, (n, n, 1))
    return np.clip(t, 0, 255).astype(np.float32)


def _box_tex(colour: tuple, seed: int, front: np.ndarray | None = None, patches: tuple | None = None) -> np.ndarray:
    faces = np.stack([_face_tex(colour, seed + k) for k in range(6)])
    if patches:
        rng = np.random.default_rng(seed + 50)
        for k in range(6):
            for _ in range(2):
                y, x = rng.integers(0, 6, 2)
                faces[k, y:y + 3, x:x + 3] = patches
    if front is not None:
        faces[4] = front
    return faces


def _head_front(skin: tuple, eye: tuple = (20, 20, 20), white: bool = True, seed: int = 0,
                mouth: tuple | None = None) -> np.ndarray:
    t = _face_tex(skin, seed)
    if white:
        t[3, 1] = t[3, 6] = (240, 240, 240)
    t[3, 2] = t[3, 5] = eye
    if mouth:
        t[6, 2:6] = mouth
    return t


# part: (size px (w, h, d), pivot px (x, y, z), box offset from pivot (x, y, z) of its centre, swing, texture)
def _pig() -> list:
    pink, dark = (238, 162, 160), (222, 140, 138)
    head = _box_tex(pink, 1, _head_front(pink, seed=2))
    snout = _box_tex(dark, 3)
    snout[4, 3:5, 2] = snout[4, 3:5, 5] = (150, 80, 80)
    body = _box_tex(pink, 4, patches=dark)
    leg = _box_tex(dark, 5)
    parts = [((10, 8, 16), (0, 6, 0), (0, 4, 0), None, body),
             ((8, 8, 8), (0, 11, 7), (0, 0, 3), "head", head),
             ((4, 3, 1), (0, 11, 7), (0, -1.5, 7.5), "head", snout)]
    for i, (x, z) in enumerate(((-3, 5), (3, 5), (-3, -5), (3, -5))):
        parts.append(((4, 6, 4), (x, 6, z), (0, -3, 0), "a" if i in (0, 3) else "b", leg))
    return parts


def _cow() -> list:
    brown, white = (82, 58, 38), (225, 225, 225)
    head = _box_tex(brown, 11, _head_front(brown, seed=12, mouth=(180, 150, 130)))
    body = _box_tex(brown, 13, patches=white)
    leg = _box_tex(brown, 14, patches=white)
    horn = _box_tex((215, 210, 190), 15)
    parts = [((12, 10, 18), (0, 12, 0), (0, 5, 0), None, body),
             ((8, 8, 6), (0, 18, 9), (0, 1, 3), "head", head),
             ((1, 3, 1), (0, 18, 9), (-4.5, 6, 3), "head", horn), ((1, 3, 1), (0, 18, 9), (4.5, 6, 3), "head", horn)]
    for i, (x, z) in enumerate(((-4, 6), (4, 6), (-4, -6), (4, -6))):
        parts.append(((4, 12, 4), (x, 12, z), (0, -6, 0), "a" if i in (0, 3) else "b", leg))
    return parts


def _sheep() -> list:
    wool, face = (232, 232, 232), (214, 186, 165)
    head = _box_tex(face, 21, _head_front(face, seed=22))
    body = _box_tex(wool, 23)
    leg = _box_tex(face, 24)
    parts = [((12, 10, 18), (0, 11, 0), (0, 5, 0), None, body),
             ((6, 6, 8), (0, 17, 8), (0, 1, 4), "head", head),
             ((8, 4, 4), (0, 17, 8), (0, 4.5, 1.5), "head", _box_tex(wool, 25))]
    for i, (x, z) in enumerate(((-3, 6), (3, 6), (-3, -6), (3, -6))):
        parts.append(((4, 12, 4), (x, 12, z), (0, -6, 0), "a" if i in (0, 3) else "b", leg))
    return parts


def _zombie() -> list:
    skin, shirt, pants = (96, 140, 76), (38, 152, 160), (64, 60, 150)
    face = _head_front(skin, eye=(20, 40, 20), white=False, seed=31, mouth=(60, 90, 50))
    head = _box_tex(skin, 32, face)
    head[2, 0:2, :] = (60, 90, 50)  # darker top
    body = _box_tex(shirt, 33)
    arm = _box_tex(skin, 34)
    arm[:, 0:3] = _face_tex(shirt, 35)[:3]
    leg = _box_tex(pants, 36)
    return [((8, 12, 4), (0, 12, 0), (0, 6, 0), None, body),
            ((8, 8, 8), (0, 24, 0), (0, 4, 0), "head", head),
            ((4, 12, 4), (-6, 22, 0), (0, -5, 0), "arm", arm), ((4, 12, 4), (6, 22, 0), (0, -5, 0), "arm", arm),
            ((4, 12, 4), (-2, 12, 0), (0, -6, 0), "a", leg), ((4, 12, 4), (2, 12, 0), (0, -6, 0), "b", leg)]


def _chicken() -> list:
    white, red, yellow = (240, 240, 235), (200, 40, 40), (235, 180, 50)
    head = _box_tex(white, 41, _head_front(white, seed=42, white=False))
    body, leg = _box_tex(white, 45, patches=(222, 222, 214)), _box_tex(yellow, 46)
    return [((6, 6, 8), (0, 5, 0), (0, 3, 0), None, body),
            ((4, 6, 3), (0, 9, 4), (0, 2, 1), "head", head),
            ((2, 2, 2), (0, 9, 4), (0, 1, 3.5), "head", _box_tex(yellow, 43)),
            ((2, 2, 1), (0, 9, 4), (0, 5.5, 1), "head", _box_tex(red, 44)),
            ((1, 5, 1), (-1.5, 5, 1), (0, -2.5, 0), "a", leg), ((1, 5, 1), (1.5, 5, 1), (0, -2.5, 0), "b", leg)]


def _fish() -> list:
    scale, fin = (104, 150, 196), (70, 104, 150)
    body = _box_tex(scale, 47, patches=(150, 190, 220))
    body[4, 2, 1] = body[4, 2, 6] = (20, 20, 20)
    return [((3, 4, 8), (0, 2, 0), (0, 2, 0), None, body),
            ((1, 4, 3), (0, 2, -4), (0, 2, -1.5), "a", _box_tex(fin, 48)),
            ((1, 2, 4), (0, 6, 0), (0, 1, 0), None, _box_tex(fin, 49))]


def _spider() -> list:
    dark, darker = (58, 48, 44), (38, 32, 30)
    face = _face_tex(dark, 51)
    face[3, 1:3] = face[3, 5:7] = face[5, 2] = face[5, 5] = (210, 30, 30)
    leg = _box_tex(dark, 54)
    parts = [((10, 6, 12), (0, 6, -4), (0, 3, 0), None, _box_tex(darker, 53, patches=(74, 62, 54))),
             ((8, 6, 6), (0, 6, 5), (0, 3, 0), "head", _box_tex(dark, 52, face))]
    for i, z in enumerate((4, 1, -2, -5)):
        for side in (-1, 1):
            parts.append(((9, 2, 2), (side * 4, 7, z), (side * 4.5, -1.5, 0), "a" if (i + side) % 2 else "b", leg))
    return parts


def _slime() -> list:
    green = (112, 200, 92)
    face = _head_front(green, eye=(24, 70, 24), white=False, seed=61, mouth=(44, 120, 44))
    return [((12, 12, 12), (0, 0, 0), (0, 6, 0), None, _box_tex(green, 62, face, patches=(140, 220, 120)))]


def _settler() -> list:
    """The people of the hamlets: a tunic, a straw hat."""
    skin, tunic, straw, pants = (214, 170, 128), (122, 88, 54), (222, 192, 96), (70, 72, 94)
    head = _box_tex(skin, 72, _head_front(skin, seed=71, mouth=(150, 92, 72)))
    hat, body, arm, leg = _box_tex(straw, 73), _box_tex(tunic, 74, patches=(104, 74, 44)), _box_tex(tunic, 75), \
        _box_tex(pants, 76)
    arm[:, 6:8] = _face_tex(skin, 77)[:2]
    return [((8, 12, 4), (0, 12, 0), (0, 6, 0), None, body),
            ((8, 8, 8), (0, 24, 0), (0, 4, 0), "head", head),
            ((12, 1, 12), (0, 24, 0), (0, 8.5, 0), "head", hat), ((6, 3, 6), (0, 24, 0), (0, 10.5, 0), "head", hat),
            ((4, 12, 4), (-6, 22, 0), (0, -5, 0), "b", arm), ((4, 12, 4), (6, 22, 0), (0, -5, 0), "a", arm),
            ((4, 12, 4), (-2, 12, 0), (0, -6, 0), "a", leg), ((4, 12, 4), (2, 12, 0), (0, -6, 0), "b", leg)]


def _guardian() -> list:
    """The hamlets' keeper: a tall figure of mossy stone with amber eyes."""
    stone, moss = (134, 132, 126), (88, 122, 72)
    face = _face_tex(stone, 81)
    face[3, 1:3] = face[3, 5:7] = (255, 184, 64)
    face[6, 2:6] = (92, 90, 86)
    limb = _box_tex(stone, 84, patches=moss)
    return [((14, 14, 8), (0, 18, 0), (0, 7, 0), None, _box_tex(stone, 83, patches=moss)),
            ((8, 8, 8), (0, 32, 0), (0, 4, 0), "head", _box_tex(stone, 82, face, patches=moss)),
            ((5, 20, 5), (-9.5, 31, 0), (0, -9, 0), "b", limb), ((5, 20, 5), (9.5, 31, 0), (0, -9, 0), "a", limb),
            ((6, 18, 6), (-3.5, 18, 0), (0, -9, 0), "a", limb), ((6, 18, 6), (3.5, 18, 0), (0, -9, 0), "b", limb)]


def _cinder() -> list:
    """A little thing of the Deep made of embers: quick, and it sets alight what it touches."""
    hot, core = (236, 110, 30), (255, 214, 96)
    face = _face_tex(hot, 91, noise=18)
    face[3, 1:3] = face[3, 5:7] = (255, 250, 200)
    face[6, 2:6] = (120, 40, 10)
    limb = _box_tex((150, 52, 18), 94, patches=hot)
    return [((6, 8, 4), (0, 8, 0), (0, 4, 0), None, _box_tex(hot, 93, patches=core)),
            ((6, 6, 6), (0, 16, 0), (0, 3, 0), "head", _box_tex(hot, 92, face, patches=core)),
            ((2, 8, 2), (-4, 15, 0), (0, -4, 0), "b", limb), ((2, 8, 2), (4, 15, 0), (0, -4, 0), "a", limb),
            ((2, 8, 2), (-1.5, 8, 0), (0, -4, 0), "a", limb), ((2, 8, 2), (1.5, 8, 0), (0, -4, 0), "b", limb)]


def _wyrm() -> list:
    """The Sky Wyrm: a head (the first box) and the segment its long body is made of (the second)."""
    scale, belly = (58, 96, 150), (150, 214, 232)
    face = _face_tex(scale, 96, noise=14)
    face[2:4, 1:3] = face[2:4, 5:7] = (255, 236, 120)
    face[5:7, 1:7] = (20, 28, 48)
    face[5, 1::2] = (240, 244, 250)
    head = _box_tex(scale, 97, face, patches=belly)
    body = _box_tex(scale, 98, patches=belly)
    body[2, 3:5, :] = (200, 236, 244)  # a pale ridge down the back
    return [((22, 14, 26), (0, 0, 0), (0, 7, 0), None, head), ((16, 11, 20), (0, 0, 0), (0, 6, 0), None, body)]


def _rabbit() -> list:
    fur, pale = (150, 112, 78), (232, 226, 214)
    head = _box_tex(fur, 101, _head_front(fur, seed=102, white=False))
    body, leg, ear = _box_tex(fur, 103, patches=(132, 96, 64)), _box_tex(fur, 104), _box_tex(fur, 105)
    ear[4, 1:7, 3:5] = (214, 150, 150)
    return [((4, 4, 7), (0, 2, 0), (0, 2, 0), None, body),
            ((4, 4, 4), (0, 6, 3), (0, 1, 2), "head", head),
            ((1, 4, 1), (0, 6, 3), (-1, 5, 1.5), "head", ear), ((1, 4, 1), (0, 6, 3), (1, 5, 1.5), "head", ear),
            ((2, 2, 2), (0, 4, -3.5), (0, 0, -1), None, _box_tex(pale, 106)),
            ((1, 2, 1), (-1.5, 2, 2.5), (0, -1, 0), "a", leg), ((1, 2, 1), (1.5, 2, 2.5), (0, -1, 0), "b", leg),
            ((2, 2, 3), (-1.5, 2, -2.5), (0, -1, 0), "b", leg), ((2, 2, 3), (1.5, 2, -2.5), (0, -1, 0), "a", leg)]


def _horse() -> list:
    coat, mane, sock = (124, 74, 40), (48, 30, 18), (226, 220, 208)
    face = _head_front(coat, seed=111, mouth=(88, 52, 30))
    face[1:6, 3:5] = sock  # a blaze down the nose
    head = _box_tex(coat, 112, face)
    body, neck, dark = _box_tex(coat, 113), _box_tex(coat, 114), _box_tex(mane, 115)
    leg = _box_tex(coat, 116)
    leg[:, 6:8] = _face_tex(sock, 117)[:2]
    parts = [((10, 10, 22), (0, 12, 0), (0, 5, 0), None, body),
             ((4, 9, 5), (0, 19, 9), (0, 4.5, 1.5), "head", neck),
             ((5, 5, 9), (0, 19, 9), (0, 10, 5), "head", head),
             ((1, 2, 1), (0, 19, 9), (-1.5, 13.5, 2), "head", dark), ((1, 2, 1), (0, 19, 9), (1.5, 13.5, 2), "head", dark),
             ((2, 9, 2), (0, 19, 9), (0, 5.5, -1.5), "head", dark),
             ((2, 9, 2), (0, 21, -11), (0, -4.5, -1), None, dark)]
    for i, (x, z) in enumerate(((-3.5, 9), (3.5, 9), (-3.5, -9), (3.5, -9))):
        parts.append(((3, 12, 3), (x, 12, z), (0, -6, 0), "a" if i in (0, 3) else "b", leg))
    return parts


def _polar_bear() -> list:
    white, shade = (238, 238, 230), (214, 214, 204)
    face = _head_front(white, seed=121, white=False)
    head = _box_tex(white, 122, face)
    snout = _box_tex(shade, 123)
    snout[4, 2:4, 3:5] = (30, 30, 34)
    body, leg = _box_tex(white, 124, patches=shade), _box_tex(shade, 125)
    parts = [((14, 11, 22), (0, 10, 0), (0, 5.5, 0), None, body),
             ((8, 7, 8), (0, 16, 11), (0, 0, 4), "head", head),
             ((4, 3, 3), (0, 16, 11), (0, -1.5, 9.5), "head", snout),
             ((2, 2, 1), (0, 16, 11), (-3, 4, 2), "head", snout), ((2, 2, 1), (0, 16, 11), (3, 4, 2), "head", snout)]
    for i, (x, z) in enumerate(((-4.5, 8), (4.5, 8), (-4.5, -8), (4.5, -8))):
        parts.append(((5, 10, 5), (x, 10, z), (0, -5, 0), "a" if i in (0, 3) else "b", leg))
    return parts


def _camel() -> list:
    tan, dark = (196, 160, 104), (164, 128, 78)
    head = _box_tex(tan, 131, _head_front(tan, seed=132, mouth=(120, 90, 60)))
    body, hump, leg = _box_tex(tan, 133, patches=dark), _box_tex(dark, 134), _box_tex(tan, 135)
    parts = [((10, 9, 22), (0, 14, 0), (0, 4.5, 0), None, body),
             ((6, 5, 6), (0, 14, 0), (0, 11, 4.5), None, hump), ((6, 5, 6), (0, 14, 0), (0, 11, -4.5), None, hump),
             ((4, 12, 4), (0, 21, 10), (0, 5, 1), "head", _box_tex(tan, 136)),
             ((5, 5, 9), (0, 21, 10), (0, 12, 4), "head", head)]
    for i, (x, z) in enumerate(((-3.5, 8), (3.5, 8), (-3.5, -8), (3.5, -8))):
        parts.append(((3, 14, 3), (x, 14, z), (0, -7, 0), "a" if i in (0, 3) else "b", leg))
    return parts


def _frog() -> list:
    green, belly, eye = (88, 150, 60), (198, 206, 140), (226, 200, 60)
    body = _box_tex(green, 141, patches=(64, 120, 44))
    body[3] = _face_tex(belly, 142)
    ball = _box_tex(eye, 143)
    ball[:, 3:5, 3:5] = (20, 20, 20)
    leg = _box_tex(green, 144)
    return [((6, 3, 6), (0, 1, 0), (0, 1.5, 0), None, body),
            ((5, 3, 3), (0, 3, 3), (0, 0.5, 1), "head", _box_tex(green, 145)),
            ((2, 2, 2), (0, 3, 3), (-1.5, 2.5, 0.5), "head", ball), ((2, 2, 2), (0, 3, 3), (1.5, 2.5, 0.5), "head", ball),
            ((2, 2, 4), (-3.5, 2, -2), (0, -1, 0), "a", leg), ((2, 2, 4), (3.5, 2, -2), (0, -1, 0), "b", leg),
            ((1, 2, 1), (-3, 2, 2), (0, -1, 0), "b", leg), ((1, 2, 1), (3, 2, 2), (0, -1, 0), "a", leg)]


def _parrot() -> list:
    red, blue, yellow, beak = (206, 44, 40), (40, 84, 200), (240, 200, 50), (60, 56, 52)
    face = _face_tex(red, 151)
    face[2:4, 1:3] = face[2:4, 5:7] = (244, 244, 240)
    face[3, 2] = face[3, 5] = (20, 20, 20)
    wing = _box_tex(blue, 153)
    wing[:, 0:3] = _face_tex(yellow, 154)[:3]
    leg = _box_tex(beak, 156)
    return [((4, 6, 4), (0, 3, 0), (0, 3, 0), None, _box_tex(red, 152)),
            ((3, 3, 4), (0, 9, 1), (0, 1.5, 0.5), "head", _box_tex(red, 155, face)),
            ((1, 2, 2), (0, 9, 1), (0, 1, 3), "head", leg),
            ((1, 5, 3), (-2.5, 8, 0), (0, -2.5, 0), "a", wing), ((1, 5, 3), (2.5, 8, 0), (0, -2.5, 0), "b", wing),
            ((3, 1, 4), (0, 3, -2), (0, 0.5, -2), None, _box_tex(blue, 157)),
            ((1, 3, 1), (-1, 3, 0), (0, -1.5, 0), "a", leg), ((1, 3, 1), (1, 3, 0), (0, -1.5, 0), "b", leg)]


def _wrapped(colour: tuple, seed: int, band: tuple) -> np.ndarray:
    """Linen wound round and round: light and dark turns."""
    faces = _box_tex(colour, seed)
    faces[:, 1::3] = np.array(band, np.float32)
    faces[:, 5, 2:5] = np.array(band, np.float32) * 0.8
    return faces


def _mummy() -> list:
    """What walks the dry lands at night: a dead thing wound in old linen, two green lights in the gap."""
    linen, band = (206, 196, 160), (160, 148, 112)
    head = _wrapped(linen, 161, band)
    head[4, 2:4, 1:7] = (26, 22, 18)
    head[4, 3, 2] = head[4, 3, 5] = (150, 255, 120)
    body, arm, leg = _wrapped(linen, 162, band), _wrapped(linen, 163, band), _wrapped(linen, 164, band)
    strip = _box_tex(band, 165)
    return [((8, 12, 4), (0, 12, 0), (0, 6, 0), None, body),
            ((8, 8, 8), (0, 24, 0), (0, 4, 0), "head", head),
            ((4, 12, 4), (-6, 22, 0), (0, -5, 0), "arm", arm), ((4, 12, 4), (6, 22, 0), (0, -5, 0), "arm", arm),
            ((1, 7, 1), (-6, 22, 0), (-1, -9, -2.5), "arm", strip), ((1, 5, 1), (0, 24, 0), (3, 0.5, -4.5), "head", strip),
            ((4, 12, 4), (-2, 12, 0), (0, -6, 0), "a", leg), ((4, 12, 4), (2, 12, 0), (0, -6, 0), "b", leg)]


def _wight() -> list:
    """What walks the frozen lands at night: tall and thin, of blue ice, with a crown of icicles."""
    ice, deep, glow = (150, 196, 232), (96, 140, 190), (236, 252, 255)
    face = _face_tex(ice, 171)
    face[2:4, 1:3] = face[2:4, 5:7] = (30, 48, 80)
    face[3, 1] = face[3, 6] = glow
    face[6, 2:6] = (30, 48, 80)
    limb, spike = _box_tex(deep, 174, patches=ice), _box_tex(glow, 175)
    return [((6, 14, 4), (0, 14, 0), (0, 7, 0), None, _box_tex(ice, 173, patches=deep)),
            ((6, 7, 6), (0, 28, 0), (0, 3.5, 0), "head", _box_tex(ice, 172, face, patches=glow)),
            ((1, 4, 1), (0, 28, 0), (-2, 9, 0), "head", spike), ((1, 6, 1), (0, 28, 0), (0, 10, 0), "head", spike),
            ((1, 4, 1), (0, 28, 0), (2, 9, 0), "head", spike),
            ((2, 3, 2), (-4, 28, 0), (0, 1.5, 0), None, spike), ((2, 3, 2), (4, 28, 0), (0, 1.5, 0), None, spike),
            ((2, 16, 2), (-4, 27, 0), (0, -8, 0), "b", limb), ((2, 16, 2), (4, 27, 0), (0, -8, 0), "a", limb),
            ((2, 14, 2), (-2, 14, 0), (0, -7, 0), "a", limb), ((2, 14, 2), (2, 14, 0), (0, -7, 0), "b", limb)]


MODELS = {"pig": _pig(), "cow": _cow(), "sheep": _sheep(), "zombie": _zombie(), "chicken": _chicken(),
          "fish": _fish(), "spider": _spider(), "slime": _slime(), "settler": _settler(), "guardian": _guardian(),
          "cinder": _cinder(), "wyrm": _wyrm(), "rabbit": _rabbit(), "horse": _horse(),
          "polar_bear": _polar_bear(), "camel": _camel(), "frog": _frog(), "parrot": _parrot(), "mummy": _mummy(),
          "wight": _wight()}
MOB_SIZE = {"pig": (0.9, 0.9), "cow": (0.9, 1.4), "sheep": (0.9, 1.3), "zombie": (0.6, 1.95), "chicken": (0.5, 0.7),
            "fish": (0.4, 0.35), "spider": (1.2, 0.8), "slime": (0.75, 0.75), "settler": (0.6, 1.9),
            "guardian": (1.2, 2.6), "cinder": (0.5, 1.3), "wyrm": (2.6, 1.8), "rabbit": (0.4, 0.5),
            "horse": (1.1, 1.6), "polar_bear": (1.2, 1.4), "camel": (1.1, 2.0), "frog": (0.45, 0.35),
            "parrot": (0.4, 0.8), "mummy": (0.6, 1.95), "wight": (0.6, 2.3)}
MOB_HEALTH = {"pig": 10, "cow": 10, "sheep": 8, "zombie": 20, "chicken": 4, "fish": 3, "spider": 16, "slime": 8,
              "settler": 20, "guardian": 80, "cinder": 12, "wyrm": 160, "rabbit": 3, "horse": 22,
              "polar_bear": 30, "camel": 24, "frog": 4, "parrot": 5, "mummy": 24, "wight": 18}
MOB_DROPS = {"pig": [(B.PORK, 1, 3)], "cow": [(B.BEEF, 1, 3)], "sheep": [(B.WOOL, 1, 1)],
             "zombie": [(B.ROTTEN_FLESH, 0, 2)], "chicken": [(B.CHICKEN, 1, 1), (B.FEATHER, 0, 2)],
             "fish": [(B.FISH, 1, 1)], "spider": [(B.STRING, 0, 2)], "slime": [(B.SLIME_BALL, 0, 2)],
             "cinder": [(B.COAL, 0, 2)], "wyrm": [(B.GEM, 12, 16), (B.DIAMOND, 5, 8), (B.GOLD_INGOT, 6, 10)],
             "rabbit": [(B.RAW_RABBIT, 1, 1), (B.RABBIT_HIDE, 0, 1)], "horse": [(B.LEATHER, 0, 2)],
             "polar_bear": [(B.FISH, 0, 2), (B.LEATHER, 0, 1)], "camel": [(B.LEATHER, 0, 2)],
             "frog": [(B.SLIME_BALL, 0, 1)], "parrot": [(B.FEATHER, 1, 2)],
             "mummy": [(B.STRING, 0, 2), (B.ROTTEN_FLESH, 0, 2), (B.GOLD_INGOT, 0, 1)],
             "wight": [(B.PACKED_ICE, 0, 2)]}
HOSTILE = frozenset({"zombie", "spider", "slime", "cinder", "mummy", "wight"})
# wander the land, kept in a saved world
ANIMALS = ("pig", "cow", "sheep", "chicken", "rabbit", "horse", "polar_bear", "camel", "frog", "parrot")
MOB_SPEED = {"zombie": 0.25, "spider": 0.32, "slime": 0.2, "guardian": 0.2, "settler": 0.13, "chicken": 0.12,
             "cinder": 0.3, "rabbit": 0.2, "horse": 0.2, "polar_bear": 0.16, "camel": 0.13, "frog": 0.12,
             "parrot": 0.14, "mummy": 0.2, "wight": 0.3}
MOB_DAMAGE = {"zombie": 3, "spider": 2, "slime": 2, "cinder": 3, "wyrm": 7, "mummy": 4, "wight": 3, "polar_bear": 5}
HOP = {"rabbit": 0.3, "frog": 0.4}  # these go by hops: the jump of each
LIGHT_FOOTED = frozenset({"rabbit", "frog", "parrot", "chicken"})  # a fall does not hurt them
# where each animal lives (the biomes) and what it stands on; an animal of no biome here: any of the first four
HABITAT = {
    "plains": ("pig", "cow", "sheep", "chicken", "horse", "rabbit"), "forest": ("pig", "cow", "chicken", "rabbit"),
    "birch_forest": ("pig", "cow", "sheep", "chicken"), "sunflower_meadow": ("horse", "rabbit", "sheep", "cow", "chicken"),
    "desert": ("camel", "camel", "rabbit"), "painted_mesa": ("camel",), "mesa_woodland": ("camel", "rabbit"),
    "mesa_spires": ("camel",), "savanna": ("horse", "camel", "cow", "sheep"),
    "savanna_plateau": ("horse", "camel", "cow"), "savanna_crags": ("sheep", "camel"),
    "taiga": ("rabbit", "pig", "cow", "sheep", "chicken"), "snowy_plains": ("polar_bear", "rabbit", "rabbit"),
    "ice_spires": ("polar_bear", "rabbit"), "frozen_ocean": ("polar_bear",), "marsh": ("frog", "frog", "chicken", "pig"),
    "rainforest": ("parrot", "parrot", "pig", "chicken"), "old_growth_forest": ("pig", "rabbit", "cow"),
    "mountains": ("sheep", "rabbit"),
}
_GREEN = (B.GRASS, B.SNOWY_GRASS, B.DRY_GRASS, B.MARSH_GRASS)
GROUND = {"rabbit": (*_GREEN, B.SAND, B.RED_SAND, B.SNOW), "horse": (B.GRASS, B.DRY_GRASS),
          "polar_bear": (B.SNOW, B.SNOWY_GRASS, B.ICE, B.PACKED_ICE),
          "camel": (B.SAND, B.RED_SAND, B.DRY_GRASS, *B.TERRACOTTAS),
          "frog": (B.MARSH_GRASS, B.MUD, B.GRASS, B.WATER),
          "parrot": (B.GRASS, B.KAPOK_LEAVES, B.OAK_LEAVES, B.KAPOK_LOG)}
HERD = {"polar_bear": (1, 2), "camel": (1, 3), "frog": (1, 3), "parrot": (1, 3), "horse": (2, 4), "rabbit": (1, 3)}
DRY_BIOMES = frozenset({"desert", "painted_mesa", "mesa_woodland", "mesa_spires", "savanna", "savanna_plateau",
                        "savanna_crags"})  # where mummies walk at night
COLD_BIOMES = frozenset({"snowy_plains", "ice_spires", "taiga", "frozen_ocean"})  # ... and wights


def ground_of(kind: str) -> tuple:
    """The blocks an animal may appear on."""
    return GROUND.get(kind, _GREEN)


def _rot_y(yaw: float) -> np.ndarray:
    s, c = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _rot_x(a: float) -> np.ndarray:
    s, c = math.sin(a), math.cos(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_z(a: float) -> np.ndarray:
    s, c = math.sin(a), math.cos(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def model_boxes(kind: str, pos: tuple, yaw: float, head_pitch: float, swing: float, swing_amount: float,
                light: np.ndarray, death: float = 0.0, attack: float = 0.0) -> list:
    """The renderer's boxes for a mob: (centre, half extents, rotation, face textures, light)."""
    out = []
    ry = _rot_y(yaw)
    if death > 0:  # falls over sideways
        ry = ry @ _rot_z(min(death, 1.0) * math.pi / 2)
    base = np.array(pos, np.float64)
    for size, pivot, offset, part, tex in MODELS[kind]:
        half = np.array(size, np.float64) / 32.0
        a = 0.0
        if part == "a":
            a = math.cos(swing * 0.6662) * 1.4 * swing_amount
        elif part == "b":
            a = -math.cos(swing * 0.6662) * 1.4 * swing_amount
        elif part == "head":
            a = head_pitch
        elif part == "arm":
            a = -math.pi / 2 + math.sin(swing * 0.3) * 0.05 - attack * 0.6
        r = ry @ _rot_x(-a) if a else ry
        centre = base + ry @ (np.array(pivot, np.float64) / 16) + r @ (np.array(offset, np.float64) / 16)
        out.append((centre, half, r, tex, light))
    return out


class Mob(Body):
    """An animal or a zombie."""

    def __init__(self, kind: str, x: float, y: float, z: float, rng: np.random.Generator) -> None:
        super().__init__(x, y, z)
        self.kind = kind
        self.width, self.height = MOB_SIZE[kind]
        self.health = float(MOB_HEALTH[kind])
        self.rng = rng
        self.yaw = float(rng.uniform(0, 360))
        self.target_yaw = self.yaw
        self.wander = 0.0
        self.moving = False
        self.panic = 0.0
        self.hurt = 0.0
        self.death = 0.0
        self.swing = 0.0
        self.swing_amount = 0.0
        self.attack_cool = 0.0
        self.attack_anim = 0.0
        self.burning = 0.0
        self.head_pitch = 0.0
        self.age = 0.0
        self.sound_cool = float(rng.uniform(4, 12))
        self.hostile = kind in HOSTILE
        self.angry = 0.0  # seconds a struck bear goes for the player
        self.home: tuple | None = None  # (x, y, z, hamlet id) it stays near
        self.leash = 9.0
        self.foe: Mob | None = None  # the monster a guardian goes for
        self.offers: list = []  # a settler's trades: (item wanted, how many, item given, how many)
        self.job = ""
        self.offer_shown = -1
        self.dive = 0.0
        self.trail: list = []  # where a wyrm has been (its body follows)
        self.mode, self.angle, self.phase_t, self.aim = "circle", float(rng.uniform(0, 6.28)), 6.0, (x, y, z)

    @property
    def dead(self) -> bool:
        return self.health <= 0

    def tick(self, world, player, day: float) -> list[str]:  # noqa: ANN001
        """One AI + physics tick; returns events ("attack", "say", "burn")."""
        self.remember()
        self.age += TICK
        events: list[str] = []
        if self.hurt > 0:
            self.hurt -= TICK
        if self.dead:
            self.death += TICK
            self.vy -= 0.08
            self.move(world, 0, self.vy, 0)
            return events
        if self.kind == "wyrm":
            self._fly(player, events)
            return events
        self.check_liquids(world)
        if self.kind == "fish":
            self._swim(world)
            return events
        forward = 0.0
        speed = MOB_SPEED.get(self.kind, 0.15)
        dx, dz = player.x - self.x, player.z - self.z
        dist = math.hypot(dx, dz)
        foe = self.foe if self.foe is not None and not self.foe.dead else None
        if foe is not None:  # a guardian goes for a monster
            fx, fz = foe.x - self.x, foe.z - self.z
            fd = math.hypot(fx, fz)
            self.target_yaw = math.degrees(math.atan2(-fx, fz))
            forward = 1.0 if fd > 1.3 else 0.0
            speed = 0.3
            if fd < 2.5 and abs(foe.y - self.y) < 3 and self.attack_cool <= 0:
                events.append("smash")
                self.attack_cool = 1.2
                self.attack_anim = 1.0
        elif (self.hostile or self.angry > 0) and dist < 18 and abs(player.y - self.y) < 8:
            if self.angry > 0:
                self.angry -= TICK
                speed = 0.3
            self.target_yaw = math.degrees(math.atan2(-dx, dz))
            forward = 1.0 if dist > 0.8 else 0.0
            self.head_pitch = 0.0
            if dist < 0.9 + self.width / 2 and abs(player.y - self.y) < 1.5 and self.attack_cool <= 0:
                events.append("attack")
                self.attack_cool = 1.0
                self.attack_anim = 1.0
            elif self.kind == "spider" and self.on_ground and 2.5 < dist < 6 and self.rng.random() < 0.06:
                self.vy = 0.4  # a spider leaps
                self.vx += -math.sin(math.radians(self.target_yaw)) * 0.5
                self.vz += math.cos(math.radians(self.target_yaw)) * 0.5
        else:
            if self.panic > 0:
                self.panic -= TICK
                speed = 0.35
                forward = 1.0
                if self.rng.random() < 0.1:
                    self.target_yaw = self.yaw + float(self.rng.uniform(-90, 90))
            else:
                self.wander -= TICK
                if self.wander <= 0:
                    self.wander = float(self.rng.uniform(2, 7))
                    self.moving = self.rng.random() < 0.45
                    self.target_yaw = float(self.rng.uniform(0, 360))
                forward = 1.0 if self.moving else 0.0
                # look at a nearby player
                self.head_pitch = 0.0
                if dist < 6 and not self.moving:
                    self.target_yaw = math.degrees(math.atan2(-dx, dz))
                if self.home is not None:  # never far from home
                    hx, hz = self.home[0] - self.x, self.home[2] - self.z
                    if math.hypot(hx, hz) > self.leash:
                        self.target_yaw = math.degrees(math.atan2(-hx, hz))
                        forward, self.moving = 1.0, True
        if self.attack_cool > 0:
            self.attack_cool -= TICK
        self.attack_anim = max(0.0, self.attack_anim - TICK * 3)
        diff = (self.target_yaw - self.yaw + 180) % 360 - 180
        self.yaw += max(-20.0, min(20.0, diff))
        # don't walk off cliffs or into lava (animals)
        if forward and not self.hostile:
            fx = self.x - math.sin(math.radians(self.yaw)) * 0.9
            fz = self.z + math.cos(math.radians(self.yaw)) * 0.9
            below = [world.get(math.floor(fx), math.floor(self.y) - k, math.floor(fz)) for k in range(1, 4)]
            if not any(B.SOLID[b] for b in below) or B.LAVA in below or \
                    world.get(math.floor(fx), math.floor(self.y), math.floor(fz)) == B.LAVA:
                forward = 0.0
                self.target_yaw = self.yaw + 150
                self.moving = False
        if self.in_water or self.in_lava:
            self.vy += 0.04 if self.vy < 0.1 else 0.0
            self.move_relative(0, forward, 0.02)
            self.move(world, self.vx, self.vy, self.vz)
            self.vx *= 0.8
            self.vy = self.vy * 0.8 - 0.02
            self.vz *= 0.8
        else:
            if self.on_ground and forward and (self.hit_wall or (self.kind == "slime" and self.rng.random() < 0.15)):
                self.vy = 0.42
            elif self.on_ground and forward and self.kind in HOP and self.rng.random() < 0.5:
                self.vy = HOP[self.kind]
            elif self.kind == "parrot" and self.on_ground and forward and self.rng.random() < 0.08:
                self.vy = 0.5  # a flutter up
            slip = 0.546 if self.on_ground else 0.91
            accel = speed * 0.3 if self.on_ground else 0.02
            self.move_relative(0, forward, accel)
            self.move(world, self.vx, self.vy, self.vz)
            if self.on_ground:
                if self.fall > 3 and self.kind not in LIGHT_FOOTED:
                    self.health -= math.ceil(self.fall - 3)
                    self.hurt = 0.5
                self.fall = 0.0
            self.vy = (self.vy - 0.08) * 0.98
            if self.kind == "parrot" and self.vy < -0.07:
                self.vy = -0.07  # it glides down on its wings
            self.vx *= slip
            self.vz *= slip
        moved = math.hypot(self.x - self.px, self.z - self.pz)
        self.swing_amount += (min(moved * 4, 1.0) - self.swing_amount) * 0.4
        self.swing += moved * 4 * 2.5
        # zombies burn in the sun
        if self.kind == "zombie" and day > 0.5 and not self.in_water:
            sky, _ = world.light_at(math.floor(self.x), math.floor(self.y + 1.6), math.floor(self.z))
            if sky >= 15:
                self.burning = 3.0
        if self.burning > 0:
            self.burning -= TICK
            if int((self.burning + TICK) * 2) != int(self.burning * 2):
                self.health -= 1
                self.hurt = 0.3
                events.append("burn")
        if self.in_lava:
            self.health -= 0.2
            self.burning = 5.0
        self.sound_cool -= TICK
        if self.sound_cool <= 0:
            self.sound_cool = float(self.rng.uniform(6, 16))
            events.append("say")
        if self.y < -20:
            self.health = 0
        return events

    def _fly(self, player, events: list) -> None:  # noqa: ANN001 - the Sky Wyrm
        """Circles high over its altar; now and then dives at where the player stands, then hangs there a few
        seconds, low and open to blows, before it climbs away again."""
        self.trail.insert(0, (self.x, self.y, self.z))
        del self.trail[30:]
        hx, hy, hz = self.home[:3] if self.home else (self.x, self.y - 20, self.z)
        self.phase_t -= TICK
        if self.attack_cool > 0:
            self.attack_cool -= TICK
        reach = math.dist((self.x, self.y, self.z), (player.x, player.y + 0.9, player.z))
        if self.mode == "circle":
            self.angle += 0.02
            tx, tz = hx + math.cos(self.angle) * 30, hz + math.sin(self.angle) * 30
            ty, speed = hy + 20 + math.sin(self.angle * 2) * 4, 0.5
            if self.phase_t <= 0 and math.hypot(player.x - hx, player.z - hz) < 70:
                self.mode, self.phase_t, self.aim = "dive", 7.0, (player.x, player.y + 1.0, player.z)
        elif self.mode == "dive":
            (tx, ty, tz), speed = self.aim, 0.95
            if math.dist((self.x, self.y, self.z), self.aim) < 2.5 or self.phase_t <= 0:
                self.mode, self.phase_t = "rest", 3.5
            if reach < 3.4 and self.attack_cool <= 0:
                events.append("attack")
                self.attack_cool = 1.5
        else:
            tx, ty, tz = self.aim[0], self.aim[1] + 1.2 + math.sin(self.age * 3) * 0.3, self.aim[2]
            speed = 0.12
            if self.phase_t <= 0:
                self.mode, self.phase_t = "circle", 6.0 + float(self.rng.uniform(0, 4))
        dx, dy, dz = tx - self.x, ty - self.y, tz - self.z
        d = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)
        k = min(speed, d) / d
        self.vx += (dx * k - self.vx) * 0.12
        self.vy += (dy * k - self.vy) * 0.12
        self.vz += (dz * k - self.vz) * 0.12
        self.x, self.y, self.z = self.x + self.vx, self.y + self.vy, self.z + self.vz
        if abs(self.vx) + abs(self.vz) > 0.02:
            self.yaw = math.degrees(math.atan2(-self.vx, self.vz))

    def _swim(self, world) -> None:  # noqa: ANN001 - a fish: wanders under water, flops and dies out of it
        self.swing += 0.7
        self.swing_amount = 0.35
        if self.in_water:
            self.wander -= TICK
            if self.wander <= 0 or self.hit_wall:
                self.wander = float(self.rng.uniform(1, 4))
                self.target_yaw = self.yaw + float(self.rng.uniform(-120, 120)) + (180 if self.hit_wall else 0)
                self.dive = float(self.rng.uniform(-0.04, 0.04))
            diff = (self.target_yaw - self.yaw + 180) % 360 - 180
            self.yaw += max(-12.0, min(12.0, diff))
            self.move_relative(0, 1.0, 0.014)
            self.vy += (self.dive - self.vy) * 0.2
            if not B.WATERY[world.get(math.floor(self.x), math.floor(self.y + 0.9), math.floor(self.z))]:
                self.vy = min(self.vy, -0.03)  # stay under the surface
            self.move(world, self.vx, self.vy, self.vz)
            self.vx *= 0.86
            self.vz *= 0.86
        else:
            if self.on_ground and self.rng.random() < 0.2:
                self.vy = 0.25
                self.vx, self.vz = float(self.rng.uniform(-0.1, 0.1)), float(self.rng.uniform(-0.1, 0.1))
            self.move(world, self.vx, self.vy, self.vz)
            self.vy = (self.vy - 0.08) * 0.98
            self.vx *= 0.7
            self.vz *= 0.7
            self.health -= 0.04
        if self.y < -20:
            self.health = 0

    def damage(self, amount: float, from_x: float, from_z: float) -> None:
        if self.dead:
            return
        self.health -= amount
        self.hurt = 0.5
        d = math.hypot(self.x - from_x, self.z - from_z) or 1.0
        self.vx += (self.x - from_x) / d * 0.4
        self.vz += (self.z - from_z) / d * 0.4
        self.vy = 0.36
        if self.kind == "polar_bear":
            self.angry = 20.0  # a struck bear strikes back
        elif not self.hostile:
            self.panic = 5.0

    def boxes(self, a: float, light: np.ndarray) -> list:
        pos = self.lerp(a)
        tint = light.copy()
        if self.hurt > 0 or self.death > 0:
            tint = tint * np.array([1.0, 0.45, 0.45])
        if self.burning > 0:
            tint = np.maximum(tint, np.array([1.0, 0.7, 0.3]))
        swing = self.swing
        return model_boxes(self.kind, pos, self.yaw, self.head_pitch, swing, self.swing_amount, tint, self.death,
                           self.attack_anim)

    def to_json(self) -> list:
        return [self.kind, round(self.x, 2), round(self.y, 2), round(self.z, 2), round(self.yaw), round(self.health)]


class ItemEntity(Body):
    """A dropped item: pops out, falls, bobs and spins; picked up when the player walks over it."""

    width = 0.25
    height = 0.25

    def __init__(self, x: float, y: float, z: float, item: list, vx: float = 0.0, vy: float = 0.2,
                 vz: float = 0.0, delay: float = 0.5) -> None:
        super().__init__(x, y, z)
        self.item = item
        self.vx, self.vy, self.vz = vx, vy, vz
        self.age = 0.0
        self.delay = delay
        self.spin = float(np.random.default_rng().uniform(0, 360))

    def tick(self, world) -> None:  # noqa: ANN001
        self.remember()
        self.age += TICK
        self.delay -= TICK
        self.check_liquids(world)
        if self.in_water:
            self.vy += 0.005
        self.move(world, self.vx, self.vy, self.vz)
        self.vy = (self.vy - 0.04) * 0.98
        f = 0.6 * 0.98 if self.on_ground else 0.98
        self.vx *= f
        self.vz *= f
        if self.on_ground:
            self.vy *= -0.5 if self.vy < -0.1 else 0
        self.spin += 4

    def boxes(self, a: float, light: np.ndarray, t: float) -> list:
        x, y, z = self.lerp(a)
        bob = math.sin((self.age + t) * 2.5) * 0.06 + 0.12
        item = self.item[0]
        r = _rot_y(self.spin + a * 4)
        if B.flat_item(item):
            spr = B.item_sprite(item).astype(np.float32)
            faces = np.zeros((6, 16, 16, 4), np.float32)
            faces[4] = spr
            faces[5] = spr[:, ::-1]
            return [(np.array([x, y + bob + 0.2, z]), np.array([0.2, 0.2, 0.015]), r, faces, light)]
        tex = np.stack([B.ATLAS[B.FACE_TEX[item, k]].astype(np.float32) for k in range(6)])
        n = min(self.item[1], 3)
        out = []
        for k in range(1 if n == 1 else 2):
            off = np.array([k * 0.05, k * 0.05, -k * 0.05])
            out.append((np.array([x, y + bob + 0.125, z]) + off, np.array([0.125, 0.125, 0.125]), r, tex, light))
        return out


class BlockEntity(Body):
    """Primed TNT (``fuse`` > 0) or a falling block of sand / gravel."""

    width = 0.98
    height = 0.98

    def __init__(self, x: float, y: float, z: float, block: int, fuse: float = 0.0) -> None:
        super().__init__(x, y, z)
        self.block = block
        self.fuse = fuse
        self.vy = 0.2 if fuse else 0.0
        self.age = 0.0

    def tick(self, world) -> None:  # noqa: ANN001
        self.remember()
        self.age += TICK
        self.move(world, self.vx, self.vy, self.vz)
        self.vy = (self.vy - 0.04) * 0.98
        self.vx *= 0.98
        self.vz *= 0.98
        if self.on_ground:
            self.vx *= 0.7
            self.vz *= 0.7
        if self.fuse:
            self.fuse = max(self.fuse - TICK, 0.0001)

    def boxes(self, a: float, light: np.ndarray) -> list:
        x, y, z = self.lerp(a)
        tex = np.stack([B.ATLAS[B.FACE_TEX[self.block, k]][..., :3].astype(np.float32) for k in range(6)])
        if self.fuse and int(self.fuse * 5) % 2 == 0:
            tex = tex * 0.4 + 153
        scale = 1.0
        if self.fuse and self.fuse < 0.5:
            scale += (0.5 - self.fuse) * 0.4
        return [(np.array([x, y + 0.49, z]), np.array([0.49, 0.49, 0.49]) * scale, np.eye(3), tex, light)]


class Particles:
    """Block-breaking crumbs and smoke: (x, y, z, vx, vy, vz, life, colour, size)."""

    def __init__(self) -> None:
        self.items: list[list] = []
        self.rng = np.random.default_rng(9)

    def burst(self, x: int, y: int, z: int, block: int, n: int = 16) -> None:
        tex = B.ATLAS[B.FACE_TEX[block, 4]]
        ok = tex[..., 3] > 127
        cols = tex[ok][:, :3] if ok.any() else np.array([[128, 128, 128]], np.uint8)
        for _ in range(n):
            c = cols[self.rng.integers(0, len(cols))]
            self.items.append([x + self.rng.uniform(0.15, 0.85), y + self.rng.uniform(0.15, 0.85),
                               z + self.rng.uniform(0.15, 0.85), self.rng.uniform(-0.08, 0.08),
                               self.rng.uniform(0.0, 0.15), self.rng.uniform(-0.08, 0.08),
                               self.rng.uniform(0.5, 1.2), tuple(int(v) for v in c), 0.09])
        if len(self.items) > 300:
            del self.items[:len(self.items) - 300]

    def puff(self, x: float, y: float, z: float, colour: tuple = (220, 220, 220), n: int = 10,
             size: float = 0.2) -> None:
        for _ in range(n):
            self.items.append([x + self.rng.uniform(-0.4, 0.4), y + self.rng.uniform(0, 0.8),
                               z + self.rng.uniform(-0.4, 0.4), self.rng.uniform(-0.03, 0.03),
                               self.rng.uniform(0.02, 0.06), self.rng.uniform(-0.03, 0.03),
                               self.rng.uniform(0.6, 1.4), colour, size])

    def tick(self, world, dt: float) -> None:  # noqa: ANN001
        keep = []
        for p in self.items:
            p[6] -= dt
            if p[6] <= 0:
                continue
            k = dt * 20
            nx, ny, nz = p[0] + p[3] * k, p[1] + p[4] * k, p[2] + p[5] * k
            if B.SOLID[world.get(math.floor(nx), math.floor(ny), math.floor(nz))]:
                p[3] = p[5] = 0.0
                p[4] = 0.0
            else:
                p[0], p[1], p[2] = nx, ny, nz
                if p[8] < 0.15:
                    p[4] -= 0.04 * k
            keep.append(p)
        self.items = keep

    def view(self) -> list:
        return [(p[0], p[1], p[2], p[7], p[8]) for p in self.items]
