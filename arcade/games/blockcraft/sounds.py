"""BlockCraft's sounds, synthesised with numpy (no sound files): digging and footsteps per material, hurt, splash,
item pickup, eating, animals, zombies, explosions, UI clicks, and a few soft piano notes for the calm music."""

from __future__ import annotations

import numpy as np

from app.arcade.audio import RATE, concat, envelope, mix, noise, silence, tone

MATERIALS = ("stone", "wood", "grass", "gravel", "sand", "glass", "cloth", "snow", "metal")
MUSIC_NOTES = (261.6, 293.7, 329.6, 392.0, 440.0, 523.3, 587.3, 659.3, 784.0, 880.0)


def _band(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """A crude band-pass: low-pass by ``lo`` taps minus low-pass by ``hi`` taps."""
    a = np.convolve(x, np.ones(lo) / lo, mode="same")
    b = np.convolve(x, np.ones(hi) / hi, mode="same")
    return (a - b) * 2.0


def _material(name: str, dur: float, vol: float, seed: int) -> np.ndarray:
    if name == "stone":
        x = mix(_band(noise(dur, 1.0, 26, seed=seed), 2, 9), tone(140, dur * 0.6, "sine", 0.25, decay=40))
    elif name == "wood":
        x = mix(tone(210, dur, "triangle", 0.45, slide_to=170, decay=30),
                _band(noise(dur, 0.6, 30, seed=seed), 3, 14) * 0.7)
    elif name == "grass":
        x = _band(noise(dur, 1.0, 16, seed=seed), 3, 10)
        x = x * (0.6 + 0.4 * np.sign(np.sin(np.arange(len(x)) / RATE * 2 * np.pi * 37)))
    elif name == "gravel":
        x = _band(noise(dur, 1.0, 14, seed=seed), 2, 6)
        x = x * (0.5 + 0.5 * (np.random.default_rng(seed).random(len(x) // 300 + 1).repeat(300)[:len(x)] > 0.4))
    elif name == "sand":
        x = _band(noise(dur, 0.9, 12, seed=seed), 4, 20)
    elif name == "glass":
        x = mix(_band(noise(dur, 0.8, 18, seed=seed), 1, 3), tone(2400, dur, "sine", 0.25, decay=25),
                tone(3100, dur, "sine", 0.2, decay=30), tone(1800, dur, "sine", 0.18, decay=20))
    elif name == "cloth":
        x = _band(noise(dur, 1.0, 20, seed=seed), 8, 30)
    elif name == "snow":
        x = _band(noise(dur, 0.8, 14, seed=seed), 5, 18)
    else:  # metal
        x = mix(tone(620, dur, "triangle", 0.35, decay=18), tone(930, dur, "sine", 0.2, decay=22),
                _band(noise(dur, 0.4, 40, seed=seed), 1, 4))
    return envelope(np.clip(x, -1, 1) * vol, 0.002, 0.03)


def build(sfx) -> None:  # noqa: ANN001 - the game's SoundBank
    """Add every sound to the bank."""
    for i, m in enumerate(MATERIALS):
        sfx.add(f"dig_{m}", _material(m, 0.2, 0.85, 11 + i), 0.8)
        sfx.add(f"place_{m}", _material(m, 0.14, 0.75, 31 + i), 0.7)
        sfx.add(f"step_{m}", _material(m, 0.09, 0.45, 51 + i), 0.35)
        sfx.add(f"hit_{m}", _material(m, 0.08, 0.4, 71 + i), 0.3)
    hurt = mix(tone(240, 0.22, "saw", 0.45, slide_to=150, decay=6),
               np.convolve(noise(0.22, 0.4, 8, seed=3), np.ones(6) / 6, mode="same"))
    sfx.add("hurt", envelope(np.convolve(hurt, np.ones(4) / 4, mode="same"), 0.005, 0.06), 0.8)
    splash = mix(np.convolve(noise(0.7, 1.0, 5, seed=4), np.ones(5) / 5, mode="same") * 1.5,
                 tone(500, 0.35, "sine", 0.15, slide_to=900, decay=8))
    sfx.add("splash", splash, 0.6)
    sfx.add("swim", np.convolve(noise(0.3, 0.6, 9, seed=5), np.ones(8) / 8, mode="same") * 1.6, 0.3)
    sfx.add("pop", tone(520, 0.07, "sine", 0.5, slide_to=1250, release=0.03), 0.5)
    sfx.add("click", concat(tone(1500, 0.012, "square", 0.3), tone(900, 0.02, "square", 0.25)), 0.5)
    eat = concat(*[_band(noise(0.06, 0.9, 30, seed=60 + k), 2, 8) for k in range(3)])
    sfx.add("eat", eat, 0.6)
    sfx.add("burp", tone(110, 0.3, "saw", 0.4, slide_to=80, decay=4), 0.6)
    sfx.add("orb", mix(tone(1320, 0.12, "sine", 0.35, decay=18), tone(1760, 0.12, "sine", 0.2, decay=20)), 0.4)
    sfx.add("levelup", concat(tone(784, 0.1, "sine", 0.35), tone(988, 0.1, "sine", 0.35),
                              tone(1175, 0.3, "sine", 0.35, decay=4)), 0.5)
    sfx.add("break_tool", mix(_material("metal", 0.25, 0.6, 99), _band(noise(0.25, 0.7, 12, seed=98), 1, 4)), 0.7)
    sfx.add("attack", _band(noise(0.08, 0.8, 30, seed=90), 4, 16), 0.5)
    t = np.arange(int(0.7 * RATE)) / RATE
    groan = np.sign(np.sin(2 * np.pi * (95 + 8 * np.sin(2 * np.pi * 5 * t)) * t)) * np.exp(-t * 2.5) * 0.4
    sfx.add("zombie", envelope(np.convolve(groan, np.ones(12) / 12, mode="same"), 0.05, 0.15), 0.6)
    oink = concat(tone(380, 0.08, "square", 0.3, slide_to=300), silence(0.05), tone(360, 0.09, "square", 0.3,
                                                                                    slide_to=280))
    sfx.add("pig", np.convolve(oink, np.ones(5) / 5, mode="same"), 0.5)
    t = np.arange(int(1.0 * RATE)) / RATE
    moo = np.sin(2 * np.pi * (150 - 30 * t) * t) + 0.5 * np.sign(np.sin(2 * np.pi * (150 - 30 * t) * t * 2))
    sfx.add("cow", envelope(np.convolve(moo * 0.3, np.ones(20) / 20, mode="same"), 0.08, 0.3), 0.5)
    t = np.arange(int(0.5 * RATE)) / RATE
    baa = np.sign(np.sin(2 * np.pi * 320 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 28 * t)) * 0.3
    sfx.add("sheep", envelope(np.convolve(baa, np.ones(10) / 10, mode="same"), 0.03, 0.15), 0.5)
    boom = mix(np.convolve(noise(1.4, 1.0, 3, seed=7), np.ones(14) / 14, mode="same") * 2.4,
               tone(55, 1.0, "sine", 0.6, slide_to=30, decay=3))
    sfx.add("explode", np.clip(boom, -1, 1), 1.0)
    sfx.add("fuse", _band(noise(1.0, 0.5, 1, seed=8), 1, 3) * 0.6, 0.5)
    sfx.add("fire", _band(noise(0.25, 0.5, 6, seed=12), 1, 5), 0.3)
    sfx.add("door", _material("wood", 0.2, 0.7, 77), 0.6)
    for i, f in enumerate(MUSIC_NOTES):
        note = mix(tone(f, 3.0, "sine", 0.32, decay=1.3, attack=0.01, release=0.6),
                   tone(f * 2, 3.0, "sine", 0.08, decay=2.2, release=0.6),
                   tone(f * 3, 3.0, "sine", 0.03, decay=3.0, release=0.6))
        sfx.add(f"note{i}", note, 0.32)
    _voices(sfx)
    _ambience(sfx)


def _sweep(f0: float, f1: float, dur: float, wave_: str = "sine", vibrato: float = 0.0, rate_hz: float = 0.0,
           ) -> np.ndarray:
    """A voice gliding from ``f0`` to ``f1`` Hz, with an optional wobble."""
    t = np.arange(int(dur * RATE)) / RATE
    f = np.linspace(f0, f1, len(t)) * (1 + vibrato * np.sin(2 * np.pi * rate_hz * t))
    ph = 2 * np.pi * np.cumsum(f) / RATE
    if wave_ == "saw":
        return 2.0 * ((ph / (2 * np.pi)) % 1.0) - 1.0
    if wave_ == "square":
        return np.sign(np.sin(ph))
    return np.sin(ph)


def _soft(x: np.ndarray, taps: int) -> np.ndarray:
    return np.convolve(x, np.ones(taps) / taps, mode="same")


def _voices(sfx) -> None:  # noqa: ANN001
    """Animals, monsters and the people of the hamlets (rounder than the first ones, which they replace)."""
    moo = _soft(_sweep(118, 96, 1.1, "saw", 0.012, 5.5), 14) + 0.5 * _soft(_sweep(236, 190, 1.1, "saw", 0.012, 5.5), 22)
    sfx.add("cow", envelope(moo * 0.34, 0.12, 0.4), 0.55)
    grunt = concat(envelope(_soft(_sweep(210, 150, 0.11, "saw", 0.05, 38), 7), 0.01, 0.04), silence(0.05),
                   envelope(_soft(_sweep(240, 140, 0.16, "saw", 0.06, 42), 7), 0.01, 0.06))
    sfx.add("pig", grunt * 0.42, 0.5)
    baa = _soft(_sweep(330, 290, 0.62, "saw", 0.055, 24), 9)
    sfx.add("sheep", envelope(baa * 0.36, 0.04, 0.2), 0.5)
    cluck = concat(*[envelope(_sweep(680 + 60 * k, 420, 0.07, "sine") * 0.5 +
                              _band(noise(0.07, 0.25, 30, seed=120 + k), 2, 6), 0.004, 0.03)
                     for k in range(3)][:3], silence(0.04),
                   envelope(_sweep(820, 520, 0.16, "sine", 0.03, 30) * 0.5, 0.004, 0.08))
    sfx.add("chicken", cluck * 0.6, 0.45)
    groan = _soft(_sweep(92, 70, 0.9, "saw", 0.03, 4.2), 16) + 0.4 * _soft(noise(0.9, 0.5, 2, seed=121), 30)
    sfx.add("zombie", envelope(groan * 0.4, 0.1, 0.3), 0.6)
    hiss = _band(noise(0.55, 1.0, 3, seed=122), 1, 3) * (0.6 + 0.4 * np.sin(np.arange(int(0.55 * RATE)) / RATE * 190))
    sfx.add("spider", envelope(hiss * 0.5, 0.03, 0.2), 0.5)
    squelch = _soft(_sweep(160, 60, 0.22, "sine"), 3) * 0.6 + _band(noise(0.22, 0.5, 14, seed=123), 6, 22)
    sfx.add("slime", envelope(squelch, 0.005, 0.08), 0.5)
    hum = concat(envelope(_soft(_sweep(170, 190, 0.16, "saw", 0.01, 6), 18), 0.03, 0.06), silence(0.04),
                 envelope(_soft(_sweep(200, 150, 0.24, "saw", 0.01, 6), 18), 0.03, 0.1))
    sfx.add("settler", hum * 0.5, 0.45)
    rumble = _soft(_sweep(58, 44, 0.9, "saw", 0.02, 3), 26) + _soft(noise(0.9, 0.6, 2, seed=124), 60) * 1.5
    sfx.add("guardian", envelope(rumble * 0.5, 0.1, 0.35), 0.6)
    # water: a deeper plunge with bubbles, softer strokes, and footsteps through grass with two parts
    t = np.arange(int(0.9 * RATE)) / RATE
    plunge = _soft(noise(0.9, 1.0, 4, seed=125), 7) * 1.6 + _soft(noise(0.9, 1.0, 9, seed=126), 28) * 1.4
    bubbles = sum(np.sin(2 * np.pi * (500 + 260 * k) * t * (1 + 0.6 * t)) * np.exp(-((t - 0.12 - 0.09 * k) / 0.03) ** 2)
                  for k in range(6)) * 0.12
    sfx.add("splash", np.clip(plunge + bubbles, -1, 1), 0.6)
    sfx.add("swim", envelope(_soft(noise(0.42, 0.8, 5, seed=127), 12) * 1.7, 0.06, 0.2), 0.3)
    for i in range(3):
        rustle = concat(_band(noise(0.05, 1.0, 22, seed=130 + i), 3, 9), _band(noise(0.07, 0.7, 16, seed=140 + i), 4, 14))
        sfx.add(f"step_grass{i}" if i else "step_grass", envelope(rustle * 0.5, 0.004, 0.03), 0.35)


def _ambience(sfx) -> None:  # noqa: ANN001
    """What the world sounds like around the player, and the music."""
    rng = np.random.default_rng(77)
    for k in range(4):  # birds: a few rising and falling chirps
        notes = []
        for _ in range(int(rng.integers(2, 6))):
            f0 = float(rng.uniform(2300, 4200))
            notes += [envelope(_sweep(f0, f0 * float(rng.uniform(0.75, 1.3)), float(rng.uniform(0.05, 0.12)), "sine",
                                      0.02, 40) * 0.3, 0.006, 0.03), silence(float(rng.uniform(0.03, 0.12)))]
        sfx.add(f"bird{k}", concat(*notes), 0.22)
    t = np.arange(int(1.6 * RATE)) / RATE
    chirp = np.sin(2 * np.pi * 4300 * t) * (np.sin(2 * np.pi * 34 * t) > 0.2) * (np.sin(2 * np.pi * 2.6 * t) > 0)
    sfx.add("crickets", envelope(chirp * 0.16, 0.1, 0.3), 0.2)
    lap = _soft(noise(2.4, 1.0, 0.0, seed=150), 40) * 2.2 * (0.35 + 0.65 * np.sin(np.pi * np.arange(int(2.4 * RATE)) /
                                                                               int(2.4 * RATE)) ** 2)
    sfx.add("shore", envelope(np.clip(lap, -1, 1) * 0.5, 0.5, 0.8), 0.3)
    for k in range(2):  # a drip and its echo, deep underground
        f0 = 900.0 + 400 * k
        drip = envelope(_sweep(f0, f0 * 1.9, 0.07, "sine") * 0.5, 0.002, 0.05)
        sfx.add(f"drip{k}", concat(drip, silence(0.22), drip * 0.35, silence(0.25), drip * 0.12), 0.35)
    wind = _soft(noise(3.0, 1.0, 0.0, seed=151), 90) * 3.0 * (0.4 + 0.6 * np.sin(np.pi * np.arange(int(3.0 * RATE)) /
                                                                                 int(3.0 * RATE)) ** 2)
    sfx.add("wind", envelope(np.clip(wind, -1, 1) * 0.5, 0.8, 1.0), 0.22)
    for k in range(MUSIC_PIECES):
        sfx.add(f"music{k}", _piece(k), 0.5)


MUSIC_PIECES = 4
_SCALE = (261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25, 783.99)  # C major pentatonic, two octaves
_CHORDS = (((130.81, 196.0, 329.63), (110.0, 164.81, 261.63), (87.31, 174.61, 261.63), (98.0, 146.83, 246.94)),
           ((87.31, 130.81, 220.0), (130.81, 196.0, 329.63), (98.0, 146.83, 246.94), (110.0, 164.81, 261.63)),
           ((110.0, 164.81, 261.63), (87.31, 174.61, 261.63), (130.81, 196.0, 329.63), (98.0, 196.0, 293.66)),
           ((130.81, 164.81, 246.94), (87.31, 130.81, 220.0), (110.0, 164.81, 261.63), (98.0, 146.83, 246.94)))


def _piece(k: int) -> np.ndarray:
    """A calm piece of about half a minute: four slow warm chords, a few soft notes over them, a long echo."""
    rng = np.random.default_rng(900 + k)
    bar = 6.5
    n = int(bar * 4 * RATE) + 4 * RATE
    out = np.zeros(n)
    for i, chord in enumerate(_CHORDS[k % len(_CHORDS)]):
        t = np.arange(int((bar + 2.5) * RATE)) / RATE
        env = np.minimum(t / 2.2, 1.0) * np.minimum((t[-1] - t) / 2.6, 1.0)
        pad = sum(np.sin(2 * np.pi * f * t * (1 + d)) for f in chord for d in (-0.0018, 0.0, 0.0021)) / 9
        pad = pad + 0.25 * sum(np.sin(2 * np.pi * 2 * f * t) for f in chord) / 3
        a = int(i * bar * RATE)
        out[a:a + len(t)] += pad * env * 0.16
        for _ in range(int(rng.integers(2, 5))):  # soft notes, like a felted piano
            f = _SCALE[int(rng.integers(0, len(_SCALE)))]
            tn = np.arange(int(3.2 * RATE)) / RATE
            note = (np.sin(2 * np.pi * f * tn) + 0.3 * np.sin(2 * np.pi * 2 * f * tn) * np.exp(-tn * 3) +
                    0.1 * np.sin(2 * np.pi * 3 * f * tn) * np.exp(-tn * 5)) * np.exp(-tn * 1.5) * np.minimum(tn / 0.012, 1)
            b = a + int(rng.uniform(0.4, bar - 0.6) * RATE)
            out[b:b + len(note)] += note[:max(0, n - b)] * float(rng.uniform(0.07, 0.13))
    for delay, gain in ((0.37, 0.32), (0.74, 0.18), (1.11, 0.09)):  # the echo of a big quiet room
        d = int(delay * RATE)
        out[d:] += out[:-d] * gain
    fade = int(1.5 * RATE)
    out[-fade:] *= np.linspace(1, 0, fade)
    return np.clip(out, -1, 1)


def material_of(block: int) -> str:
    from app.arcade.games.blockcraft import blocks as B

    return B.MATERIAL.get(B.base_block(block), "stone")
