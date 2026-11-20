"""Running water. Still water (the seas, a bucket poured out) stays where it is; where a block beside or below it
opens, water runs in: straight down, and sideways over solid ground up to seven blocks, a little lower each
block. Two still blocks beside an empty one over solid ground make a third, so a pond refills. Water with nothing
feeding it drains away.

Nothing is simulated until something changes: the game ``poke``s the cells around every changed block and calls
``step`` four times a second; each step looks at the poked cells only, and pokes the neighbours of what it
changed."""

from __future__ import annotations

import numpy as np

from app.arcade.games.blockcraft import blocks as B

SOFT = B.KIND == B.K_CROSS  # what water runs into and washes away: air and every plant on land
SOFT[B.AIR] = True
SOFT[[B.SEAGRASS, B.KELP]] = False
SOFT[list(B.DOORS)] = False  # (a door, open or shut, holds water back)
_AROUND = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, 0, 0))
_SIDES = ((1, 0), (-1, 0), (0, 1), (0, -1))
_LEVEL = [int(v) for v in B.LEVEL]
_SOFT = [bool(v) for v in SOFT]
_FLUID = [bool(v) for v in B.FLUID]
_OPEN = [bool(v) for v in (SOFT | B.FLUID)]  # water can fall into it


class Water:
    """The cells whose water may have to change."""

    def __init__(self) -> None:
        self.queue: set[tuple[int, int, int]] = set()

    def poke(self, x: int, y: int, z: int) -> None:
        add = self.queue.add
        for dx, dy, dz in _AROUND:
            add((x + dx, y + dy, z + dz))

    @staticmethod
    def want(get, x: int, y: int, z: int) -> int:  # noqa: ANN001 - World.get
        """How much water belongs in an open cell: 0 none, 1..7 running, 8 still."""
        if _FLUID[get(x, y + 1, z)]:
            return 7  # falling
        best = still = 0
        for dx, dz in _SIDES:
            level = _LEVEL[get(x + dx, y, z + dz)]
            if level == 8:
                still += 1
                best = 7
            elif level > best + 1 and not _OPEN[get(x + dx, y - 1, z + dz)]:
                best = level - 1  # running water spreads only where it cannot fall
        if still >= 2:
            below = get(x, y - 1, z)
            if below == B.WATER or not _OPEN[below]:
                return 8
        return best

    def step(self, world, limit: int = 500) -> tuple[list, list]:  # noqa: ANN001 - a World
        """One step: (block changes for ``World.set_many``, plants washed away as (x, y, z, block))."""
        if not self.queue:
            return [], []
        cells = list(self.queue) if len(self.queue) <= limit else [self.queue.pop() for _ in range(limit)]
        if len(self.queue) <= limit:
            self.queue = set()
        get = world.get
        changes, washed = [], []
        for x, y, z in cells:
            b = get(x, y, z)
            now = _LEVEL[b]
            if now:  # water beside lava cools it to obsidian
                for dx, dy, dz in _AROUND[:6]:
                    if get(x + dx, y + dy, z + dz) == B.LAVA:
                        changes.append((x + dx, y + dy, z + dz, B.OBSIDIAN))
            if now == 8 or not (now or _SOFT[b]):
                continue  # still water stays; a solid block holds none
            want = self.want(get, x, y, z)
            if want == now:
                continue
            if want == 8:
                new = B.WATER
            elif want:
                new = B.FLOW + want
            elif now:
                new = B.AIR
            else:
                continue
            if b != B.AIR and not now:
                washed.append((x, y, z, b))
            changes.append((x, y, z, new))
        for x, y, z, _b in changes:
            self.poke(x, y, z)
        return changes, washed


__all__ = ["SOFT", "Water", "np"]
