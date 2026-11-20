"""Case Clash's synthesized sounds (no sound files)."""

from __future__ import annotations

import numpy as np

from app.arcade.audio import RATE, SoundBank, concat, mix, noise, silence, tone


def _lowpass(x: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(x, np.ones(n) / n, mode="same")


def _shot(dur: float, body: float, crack: float, low: float, seed: int) -> np.ndarray:
    return mix(noise(dur, crack, decay=28, seed=seed), _lowpass(noise(dur, body, decay=14, seed=seed + 1), 6),
               tone(low, dur * 0.6, "sine", 0.6, slide_to=low * 0.45, decay=12))


def build(bank: SoundBank) -> None:
    """Add every sound of the game to the bank."""
    bank.add("click", tone(1900, 0.035, "sine", 0.35, decay=60), 0.6)
    bank.add("hover", tone(2600, 0.02, "sine", 0.15, decay=90), 0.3)
    bank.add("tab", mix(tone(1200, 0.05, "sine", 0.3, decay=50), tone(2400, 0.04, "sine", 0.12, decay=70)), 0.6)
    bank.add("tick", mix(tone(3200, 0.018, "square", 0.18, decay=200), noise(0.015, 0.25, decay=250, seed=3)), 0.55)
    bank.add("unlock", concat(
        mix(_lowpass(noise(0.12, 0.9, decay=30, seed=11), 4), tone(110, 0.12, "sine", 0.8, slide_to=70, decay=20)),
        silence(0.05),
        mix(tone(900, 0.25, "sine", 0.35, decay=12), tone(1350, 0.2, "sine", 0.18, decay=14),
            noise(0.05, 0.3, decay=60, seed=12))), 0.9)
    for i, notes in enumerate(((523, 659), (523, 659, 784), (587, 740, 880, 1175), (659, 831, 988, 1319, 1661))):
        parts = [concat(silence(0.07 * k), tone(f, 0.5 - 0.05 * k, "triangle", 0.3, decay=4)) for k, f in
                 enumerate(notes)]
        bank.add(f"reveal{i + 2}", mix(*parts, noise(0.2, 0.1 * (i + 1), decay=10, seed=20 + i)), 0.8)
    sparkle = [concat(silence(0.06 * k), tone(1047 * 2 ** (k / 6), 0.35, "sine", 0.22, decay=6)) for k in range(12)]
    bank.add("reveal6", mix(*sparkle, tone(262, 1.4, "triangle", 0.35, decay=1.5),
                            tone(392, 1.4, "triangle", 0.25, decay=1.5)), 1.0)
    bank.add("cash", concat(mix(tone(1320, 0.07, "square", 0.18, decay=30), noise(0.05, 0.25, decay=80, seed=31)),
                            mix(tone(2640, 0.3, "sine", 0.35, decay=10), tone(3960, 0.2, "sine", 0.12, decay=14))),
             0.7)
    bank.add("buy", mix(tone(880, 0.12, "sine", 0.3, decay=20), tone(1320, 0.15, "sine", 0.2, decay=18)), 0.6)
    bank.add("deny", concat(tone(220, 0.08, "square", 0.2), silence(0.03), tone(180, 0.12, "square", 0.2)), 0.5)
    bank.add("stamp", mix(_lowpass(noise(0.25, 1.0, decay=22, seed=41), 8), tone(80, 0.25, "sine", 0.9,
                                                                                  slide_to=50, decay=14)), 1.0)
    bank.add("slot", mix(tone(700, 0.06, "sine", 0.3, decay=40), noise(0.03, 0.2, decay=120, seed=42)), 0.6)
    bank.add("shot_pistol", _shot(0.22, 0.7, 0.6, 160, 51), 0.55)
    bank.add("shot_rifle", _shot(0.28, 0.9, 0.7, 120, 53), 0.5)
    bank.add("shot_sniper", _shot(0.6, 1.0, 0.9, 80, 55), 0.7)
    bank.add("shot_heavy", _shot(0.4, 1.0, 0.6, 90, 57), 0.6)
    bank.add("ding", mix(tone(1568, 0.5, "sine", 0.45, decay=7), tone(3136, 0.4, "sine", 0.18, decay=9),
                         tone(4704, 0.2, "sine", 0.06, decay=14)), 0.8)
    bank.add("hit", mix(tone(1100, 0.04, "square", 0.15, decay=80), noise(0.03, 0.3, decay=150, seed=61)), 0.6)
    bank.add("hurt", mix(_lowpass(noise(0.2, 0.8, decay=15, seed=62), 10), tone(90, 0.2, "saw", 0.3, decay=10)), 0.7)
    bank.add("empty", tone(2200, 0.02, "square", 0.2, decay=150), 0.5)
    bank.add("reload", concat(noise(0.04, 0.4, decay=60, seed=63), silence(0.18), noise(0.05, 0.5, decay=50, seed=64),
                              tone(1500, 0.03, "square", 0.1, decay=100)), 0.6)
    arp = [concat(silence(0.09 * k), tone(f, 0.3, "square", 0.14, decay=8)) for k, f in
           enumerate((523, 659, 784, 1047, 1319))]
    bank.add("levelup", mix(*arp), 0.7)
    bank.add("achieve", mix(tone(988, 0.6, "sine", 0.3, decay=4), concat(silence(0.12), tone(1319, 0.6, "sine", 0.3,
                                                                                           decay=4))), 0.7)
    bank.add("go", concat(tone(660, 0.12, "square", 0.15), tone(990, 0.25, "square", 0.15, decay=6)), 0.5)
    bank.add("beep", tone(880, 0.08, "square", 0.12), 0.5)
    bank.add("whoosh", _lowpass(noise(0.35, 0.6, decay=6, seed=70), 20) * np.hanning(int(0.35 * RATE)), 0.5)
    # surf
    bank.add("surf_go", mix(tone(784, 0.18, "triangle", 0.3, decay=10), tone(1175, 0.22, "sine", 0.18, decay=9)), 0.6)
    bank.add("surf_cp", mix(concat(tone(1047, 0.09, "sine", 0.3, decay=12), tone(1568, 0.3, "sine", 0.3, decay=7)),
                            noise(0.05, 0.08, decay=60, seed=80)), 0.7)
    bank.add("surf_fall", _lowpass(noise(0.5, 0.7, decay=5, seed=81), 24) * np.hanning(int(0.5 * RATE))
             + tone(300, 0.5, "sine", 0.2, slide_to=90, decay=4)[:int(0.5 * RATE)], 0.6)
    bank.add("surf_jump", mix(_lowpass(noise(0.08, 0.5, decay=40, seed=82), 6), tone(180, 0.08, "sine", 0.3,
                                                                                    slide_to=260, decay=30)), 0.4)
    bank.add("surf_land", mix(_lowpass(noise(0.14, 0.9, decay=26, seed=83), 10), tone(90, 0.14, "sine", 0.6,
                                                                                     slide_to=55, decay=22)), 0.6)
    fan = [concat(silence(0.1 * k), tone(f, 0.45, "triangle", 0.26, decay=5)) for k, f in
           enumerate((523, 659, 784, 1047))]
    bank.add("surf_finish", mix(*fan), 0.8)
    bank.add("surf_record", mix(*fan, concat(silence(0.42), tone(1319, 0.7, "sine", 0.3, decay=3),
                                             tone(1568, 0.8, "sine", 0.25, decay=3))), 0.9)
