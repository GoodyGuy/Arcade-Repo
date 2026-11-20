"""Rail Rush sounds, all synthesized (no sound files): pickups, moves, crashes, the yard drone's alarm and an
easy-going looping beat."""

from __future__ import annotations

import numpy as np

from app.arcade.audio import RATE, concat, envelope, mix, noise, silence, tone


def _t(dur: float) -> np.ndarray:
    return np.arange(int(dur * RATE)) / RATE


def _alarm(dur: float = 0.5) -> np.ndarray:
    """The drone's two-tone alarm beep."""
    t = _t(dur)
    f = np.where((t * 8).astype(int) % 2 == 0, 1320.0, 990.0)
    x = np.sign(np.sin(2 * np.pi * np.cumsum(f) / RATE)) * 0.18 + np.sin(2 * np.pi * np.cumsum(f) / RATE) * 0.12
    return envelope(x, 0.01, 0.06)


def _siren(dur: float = 0.6) -> np.ndarray:
    """A short rising-falling siren chirp."""
    t = _t(dur)
    f = 700 + 500 * np.sin(np.pi * t / dur) ** 2
    x = np.sin(2 * np.pi * np.cumsum(f) / RATE) * 0.3
    return envelope(x, 0.02, 0.12)


def _kick() -> np.ndarray:
    return tone(110, 0.18, "sine", 0.9, slide_to=42, decay=14, release=0.04)


def _snare() -> np.ndarray:
    return mix(noise(0.16, 0.45, decay=22, seed=3), tone(190, 0.08, "triangle", 0.25, decay=30))


def _hat(seed: int) -> np.ndarray:
    return noise(0.04, 0.16, decay=90, seed=seed)


def _beat() -> np.ndarray:
    """Two bars of a relaxed hip-hop groove (94 bpm) with a bouncy bass line."""
    bpm = 94
    step = 60.0 / bpm / 4  # sixteenth notes
    n = 32
    out = np.zeros(int(step * n * RATE) + RATE)
    kicks = {0, 6, 10, 16, 22, 26, 27}
    snares = {8, 24}
    bass = {0: 55.0, 6: 55.0, 10: 65.4, 14: 73.4, 16: 49.0, 22: 49.0, 26: 58.3, 30: 61.7}
    chords = {0: (220.0, 261.6, 329.6), 16: (196.0, 246.9, 293.7)}
    for i in range(n):
        at = int(i * step * RATE)
        parts = []
        if i in kicks:
            parts.append(_kick())
        if i in snares:
            parts.append(_snare())
        if i % 2 == 0 or i % 4 == 3:
            parts.append(_hat(i) * (1.0 if i % 4 == 2 else 0.6))
        if i in bass:
            parts.append(tone(bass[i], step * 3.2, "triangle", 0.42, release=0.05))
        if i in chords:
            stab = mix(*(tone(f, step * 1.5, "saw", 0.05, release=0.06) for f in chords[i]))
            parts.append(stab)
        for part in parts:
            out[at:at + len(part)] += part
    out = out[:int(step * n * RATE)]
    return np.tanh(out * 0.9) * 0.8


def build(bank) -> None:  # noqa: ANN001 - SoundBank
    bank.add("coin", concat(tone(1975, 0.045, "triangle", 0.32, release=0.01),
                            tone(2637, 0.17, "triangle", 0.32, decay=16)), 0.45)
    bank.add("jump", mix(noise(0.2, 0.28, decay=12, smooth=6, seed=11),
                         tone(260, 0.18, "sine", 0.22, slide_to=620)), 0.55)
    bank.add("sjump", mix(noise(0.3, 0.3, decay=8, smooth=5, seed=12),
                          tone(300, 0.3, "triangle", 0.2, slide_to=1100)), 0.6)
    bank.add("roll", mix(noise(0.32, 0.4, decay=7, smooth=14, seed=13),
                         tone(180, 0.25, "sine", 0.18, slide_to=80)), 0.55)
    bank.add("lane", noise(0.12, 0.3, decay=22, smooth=4, seed=14), 0.4)
    bank.add("land", noise(0.1, 0.4, decay=30, smooth=24, seed=15), 0.4)
    bank.add("crash", mix(noise(0.7, 0.8, decay=5, smooth=3, seed=16), tone(95, 0.45, "sine", 0.7, slide_to=38),
                          tone(410, 0.3, "square", 0.12, decay=14)), 0.8)
    bank.add("stumble", mix(tone(230, 0.16, "sine", 0.4, slide_to=130), noise(0.14, 0.4, decay=25, smooth=18,
                                                                            seed=17)), 0.6)
    bank.add("alarm", concat(_alarm(0.3), silence(0.05), _alarm(0.4)), 0.38)
    bank.add("siren", _siren(), 0.35)
    bank.add("powerup", concat(*(tone(f, 0.065, "square", 0.22, release=0.02) for f in (523, 659, 784, 1046)),
                               tone(1318, 0.22, "triangle", 0.3, decay=8)), 0.5)
    bank.add("jetpack", mix(noise(1.2, 0.3, decay=0.0, smooth=3, seed=20), tone(70, 1.2, "saw", 0.08)), 0.3,
             loop=True)
    bank.add("board", mix(tone(220, 0.45, "saw", 0.16, slide_to=880), tone(440, 0.45, "sine", 0.18, slide_to=1320)),
             0.5)
    bank.add("board_break", mix(noise(0.5, 0.6, decay=7, seed=21), tone(900, 0.4, "square", 0.15, slide_to=120)),
             0.6)
    bank.add("key", concat(*(tone(f, 0.07, "triangle", 0.3, release=0.02) for f in (1318, 1760, 2349)),
                           tone(2637, 0.25, "sine", 0.25, decay=9)), 0.5)
    bank.add("mission", concat(*(tone(f, 0.09, "square", 0.2, release=0.02) for f in (784, 988, 1175)),
                               tone(1568, 0.35, "triangle", 0.3, decay=5)), 0.5)
    bank.add("highscore", concat(*(tone(f, 0.1, "square", 0.22, release=0.02) for f in (523, 659, 784, 659, 784)),
                                 tone(1046, 0.5, "triangle", 0.32, decay=4)), 0.55)
    bank.add("click", tone(1400, 0.035, "sine", 0.3, release=0.02), 0.4)
    bank.add("buy", concat(tone(1046, 0.06, "square", 0.2), tone(1568, 0.16, "triangle", 0.3, decay=10)), 0.5)
    bank.add("deny", tone(180, 0.18, "square", 0.2, slide_to=140), 0.4)
    bank.add("rattle", concat(*(noise(0.05, 0.4, decay=40, smooth=3, seed=30 + i) for i in range(6))), 0.5)
    bank.add("pop", mix(noise(0.25, 0.6, decay=14, seed=40), tone(520, 0.3, "triangle", 0.3, slide_to=1300)), 0.6)
    bank.add("train_horn", mix(tone(311, 0.7, "saw", 0.14, attack=0.04, release=0.15),
                               tone(370, 0.7, "saw", 0.12, attack=0.04, release=0.15)), 0.35)
    bank.add("music", _beat(), 0.22, loop=True)
