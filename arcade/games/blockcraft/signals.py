"""Signals: a lever or a button gives one; signal dust laid on the ground carries it fifteen blocks, a little weaker
each block (and up or down a step); a signal lamp beside it lights, a door beside it opens.

How strong the signal is in a piece of dust is the block itself (``blocks.DUST_LEVELS``: sixteen blocks), so the
state of every circuit is saved with the world and nothing else has to be kept.

Like the water (fluids.py), nothing is worked out until something changes: the game ``poke``s the cells around
every changed block; ``step`` (ten times a second) looks at the poked cells only, and at no more than ``CAP`` of
them: the rest wait for the next step, so a huge circuit settles over a few steps instead of stalling one."""

from __future__ import annotations

from app.arcade.games.blockcraft import blocks as B

CAP = 256  # cells looked at in one step, at most
REACH = 15  # blocks of dust a signal runs along
BUTTON_STEPS = 10  # steps a pressed button stays in (a second)
_LEVEL = [-1] * 256  # the signal in a block of dust (-1: not dust)
for _i, _b in enumerate(B.DUST_LEVELS):
    _LEVEL[_b] = _i
_SOURCE = [b in (B.LEVER_ON, B.BUTTON_ON) for b in range(256)]
_SIGNAL = [b in B.SIGNAL_BLOCKS for b in range(256)]
_SOLID = [bool(v) for v in B.SOLID]
_STANDS = [b in (B.LEVER, B.LEVER_ON, B.BUTTON, B.BUTTON_ON) for b in range(256)]
_LAMP = [b in (B.LAMP, B.LAMP_ON) for b in range(256)]
_DOOR = [b in B.DOORS for b in range(256)]
_AROUND = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
# where dust joins dust: beside it, and one step up or down
_WIRE = _AROUND[:2] + _AROUND[4:] + tuple((dx, dy, dz) for dy in (1, -1) for dx, _, dz in _AROUND[:2] + _AROUND[4:])
_NEAR = tuple(dict.fromkeys(_AROUND + _WIRE + ((0, 0, 0),)))


class Signals:
    """The cells whose signal may have to change."""

    def __init__(self) -> None:
        self.queue: set[tuple[int, int, int]] = set()
        self.edges: set[tuple[int, int, int]] = set()  # ... because a signal beside them went on or off
        self.timers: dict[tuple[int, int, int], int] = {}  # pressed buttons: steps until each springs back
        self.work = 0  # cells looked at in the last step

    def poke(self, get, x: int, y: int, z: int) -> None:  # noqa: ANN001 - World.get
        """A block changed here: whatever signal block is near has to look again."""
        add = self.queue.add
        for dx, dy, dz in _NEAR:
            if _SIGNAL[get(x + dx, y + dy, z + dz)]:
                add((x + dx, y + dy, z + dz))

    def edge(self, get, x: int, y: int, z: int) -> None:  # noqa: ANN001
        """A signal went on or off here (a lever thrown, a button pressed): doors beside it follow it."""
        for dx, dy, dz in _AROUND:
            p = (x + dx, y + dy, z + dz)
            if _SIGNAL[get(*p)]:
                self.queue.add(p)
                self.edges.add(p)

    def press(self, pos: tuple) -> None:
        self.timers[pos] = BUTTON_STEPS

    @staticmethod
    def powered(get, x: int, y: int, z: int) -> bool:  # noqa: ANN001
        """Is there a signal right beside this cell (live dust, a lever that is on, a pressed button)?"""
        for dx, dy, dz in _AROUND:
            b = get(x + dx, y + dy, z + dz)
            if _SOURCE[b] or _LEVEL[b] > 0:
                return True
        return False

    def step(self, world, limit: int = CAP) -> tuple[list, list]:  # noqa: ANN001 - a World
        """One step: (block changes for ``World.set_many``, what happened: ("drop", pos, item),
        ("door", pos, open), ("lamp", pos, lit), ("click", pos))."""
        self.work = 0
        if not (self.queue or self.timers):
            return [], []
        pending: dict[tuple[int, int, int], int] = {}
        events: list = []
        wget = world.get
        queue, edges = self.queue, self.edges

        def get(x: int, y: int, z: int) -> int:
            b = pending.get((x, y, z))
            return wget(x, y, z) if b is None else b

        def put(pos: tuple, block: int) -> None:
            pending[pos] = block
            self.poke(get, *pos)

        for pos in list(self.timers):  # buttons spring back
            self.timers[pos] -= 1
            if self.timers[pos] <= 0:
                del self.timers[pos]
                if get(*pos) == B.BUTTON_ON:
                    put(pos, B.BUTTON)
                    self.edge(get, *pos)
                    events.append(("click", pos))
        budget = limit
        while queue and budget > 0:
            pos = queue.pop()
            budget -= 1
            is_edge = pos in edges
            edges.discard(pos)
            x, y, z = pos
            b = get(x, y, z)
            lv = _LEVEL[b]
            if lv >= 0:  # dust: as strong as the strongest dust it touches, less one; full beside a source
                if not _SOLID[get(x, y - 1, z)]:
                    put(pos, B.AIR)  # nothing under it any more
                    events.append(("drop", pos, B.REDSTONE))
                    if lv:
                        self.edge(get, x, y, z)
                    continue
                want = 0
                for dx, dy, dz in _AROUND:
                    if _SOURCE[get(x + dx, y + dy, z + dz)]:
                        want = REACH
                        break
                else:
                    for dx, dy, dz in _WIRE:
                        n = _LEVEL[get(x + dx, y + dy, z + dz)] - 1
                        if n > want:
                            want = n
                if want != lv:
                    put(pos, B.DUST_LEVELS[want])
                    if (want > 0) != (lv > 0):
                        self.edge(get, x, y, z)
            elif _STANDS[b]:  # a lever or a button: gone with the block it stood on
                if not _SOLID[get(x, y - 1, z)]:
                    put(pos, B.AIR)
                    events.append(("drop", pos, B.base_block(b)))
                    self.timers.pop(pos, None)
                    self.edge(get, x, y, z)
                elif b == B.BUTTON_ON and pos not in self.timers:
                    self.timers[pos] = BUTTON_STEPS
            elif _LAMP[b]:
                lit = self.powered(get, x, y, z)
                if lit != (b == B.LAMP_ON):
                    put(pos, B.LAMP_ON if lit else B.LAMP)
                    events.append(("lamp", pos, lit))
            elif _DOOR[b]:  # both halves as one: open while either has a signal beside it
                k = b - B.DOOR
                oy = y - 1 if k & 1 else y + 1
                other = get(x, oy, z)
                if not _DOOR[other] or (other - B.DOOR) & 1 == k & 1:
                    put(pos, B.AIR)  # half a door is no door
                    if not k & 1:
                        events.append(("drop", pos, B.DOOR))
                    continue
                if not k & 1 and not _SOLID[get(x, y - 1, z)]:
                    put(pos, B.AIR)
                    put((x, oy, z), B.AIR)
                    events.append(("drop", pos, B.DOOR))
                    continue
                live = self.powered(get, x, y, z) or self.powered(get, x, oy, z)
                is_open = bool(k & 2)
                if (is_edge and live != is_open) or (live and not is_open):
                    nk = (k & ~2) | (2 if live else 0)
                    put(pos, B.DOOR + nk)
                    put((x, oy, z), B.DOOR + (nk ^ 1))
                    events.append(("door", pos, live))
        self.work = limit - budget
        return [(x, y, z, b) for (x, y, z), b in pending.items()], events


__all__ = ["CAP", "REACH", "Signals"]
