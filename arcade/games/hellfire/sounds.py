"""Synthesised, crunchy lo-fi sound effects (8-bit, 11 kHz feel) and an original chugging synth-metal loop."""

from __future__ import annotations

import numpy as np

from app.arcade.audio import RATE, SoundBank, concat, envelope, mix, noise, silence, tone


def crush(x: np.ndarray, hold: int = 2, bits: int = 7, drive: float = 1.0) -> np.ndarray:
    """Sample-and-hold + bit reduction + a little drive: the crunch of the original's 8-bit samples."""
    x = np.tanh(x * drive) if drive != 1.0 else x
    if hold > 1:
        x = np.repeat(x[::hold], hold)[:len(x)]
    q = 2 ** (bits - 1)
    return np.round(x * q) / q


def lowpass(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _t(dur: float) -> np.ndarray:
    return np.arange(int(dur * RATE)) / RATE


def bang(dur: float, vol: float, decay: float, smooth: int, seed: int, body: float = 0.0,
         f0: float = 90.0) -> np.ndarray:
    n = noise(dur, vol, decay, smooth, seed=seed)
    if body:
        t = _t(dur)
        f = np.linspace(f0, f0 * 0.4, len(t))
        n = n[:len(t)] + body * np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-t * decay * 0.6)
    return n


def growl(f0: float, f1: float, dur: float, rough: float = 0.6, vol: float = 0.6, seed: int = 1,
          vowel: float = 1.0) -> np.ndarray:
    """A monster voice: a buzzy glide with noise and wobble."""
    t = _t(dur)
    rng = np.random.default_rng(seed)
    f = np.linspace(f0, f1, len(t)) * (1 + 0.08 * np.sin(2 * np.pi * 7 * t + rng.random() * 6))
    ph = 2 * np.pi * np.cumsum(f) / RATE
    buzz = np.sign(np.sin(ph)) * 0.5 + 0.5 * (2 * ((ph / (2 * np.pi)) % 1) - 1)
    buzz = buzz * (1 + 0.6 * np.sin(ph * vowel * 0.5))
    n = rng.uniform(-1, 1, len(t))
    x = buzz * (1 - rough) + n * rough * (0.6 + 0.4 * np.sin(ph * 0.25))
    x = lowpass(x, 3)
    env = np.minimum(1.0, t / 0.03) * np.minimum(1.0, (dur - t) / 0.12)
    return x * env * vol


def scream(f0: float, f1: float, dur: float, vol: float = 0.6, seed: int = 2) -> np.ndarray:
    return growl(f0, f1, dur, rough=0.35, vol=vol, seed=seed, vowel=1.7)


def build_sounds(bank: SoundBank) -> None:
    s = {}
    s["pistol"] = crush(mix(bang(0.28, 0.9, 16, 2, 1, body=0.6, f0=180), tone(900, 0.03, "square", 0.3)), 2, 6, 1.6)
    s["shotgn"] = crush(mix(bang(0.6, 1.0, 7, 3, 2, body=0.9, f0=110), bang(0.15, 0.6, 30, 1, 3)), 2, 6, 2.0)
    s["sgcock"] = crush(concat(bang(0.06, 0.6, 40, 1, 4), silence(0.12), bang(0.08, 0.7, 35, 1, 5)), 2, 6)
    s["plasma"] = crush(mix(tone(1400, 0.18, "square", 0.35, slide_to=300, decay=12),
                            tone(700, 0.18, "saw", 0.25, slide_to=200)), 2, 6)
    s["rlaunc"] = crush(mix(bang(0.5, 0.7, 5, 6, 6, body=0.6, f0=70), tone(200, 0.4, "saw", 0.2, slide_to=60)), 2, 6,
                        1.5)
    s["barexp"] = crush(bang(1.1, 1.0, 3.2, 8, 7, body=1.0, f0=60), 2, 6, 2.2)
    s["rxplod"] = crush(bang(0.9, 1.0, 3.5, 6, 8, body=0.8, f0=80), 2, 6, 2.0)
    s["firxpl"] = crush(bang(0.5, 0.8, 7, 5, 9, body=0.4, f0=120), 2, 6, 1.6)
    s["firsht"] = crush(mix(bang(0.45, 0.6, 5, 10, 10), tone(300, 0.4, "saw", 0.2, slide_to=120)), 2, 6, 1.4)
    s["bfg"] = crush(mix(tone(200, 1.0, "saw", 0.4, slide_to=1600), tone(203, 1.0, "square", 0.25, slide_to=1640),
                         bang(1.0, 0.3, 1.5, 2, 11)), 2, 6, 1.4)
    s["punch"] = crush(bang(0.15, 0.9, 25, 6, 12, body=0.7, f0=90), 2, 6, 1.8)
    s["doropn"] = crush(mix(growl(60, 75, 0.9, 0.85, 0.5, 13), bang(0.9, 0.25, 1.0, 12, 14)), 2, 6, 1.5)
    s["dorcls"] = crush(concat(mix(growl(75, 60, 0.75, 0.85, 0.5, 15), bang(0.75, 0.2, 1.0, 12, 16)),
                               bang(0.18, 0.9, 20, 4, 17, body=0.7, f0=70)), 2, 6, 1.5)
    s["swtchn"] = crush(concat(bang(0.05, 0.8, 50, 1, 18), silence(0.03), bang(0.07, 0.9, 40, 2, 19, body=0.4,
                                                                                  f0=300)), 2, 6)
    s["pstart"] = crush(mix(growl(50, 55, 0.5, 0.9, 0.4, 20), bang(0.5, 0.3, 2, 10, 21)), 2, 6)
    s["pstop"] = crush(bang(0.25, 0.8, 14, 5, 22, body=0.6, f0=60), 2, 6)
    s["itemup"] = crush(concat(tone(660, 0.05, "square", 0.3), tone(990, 0.07, "square", 0.3)), 2, 6)
    s["wpnup"] = crush(concat(tone(440, 0.06, "square", 0.3), tone(660, 0.06, "square", 0.3),
                              tone(880, 0.12, "square", 0.3)), 2, 6)
    s["getpow"] = crush(mix(tone(300, 0.8, "triangle", 0.4, slide_to=900), tone(450, 0.8, "sine", 0.3,
                                                                                  slide_to=1350)), 2, 6)
    s["secret"] = crush(concat(tone(523, 0.1, "square", 0.3), tone(659, 0.1, "square", 0.3),
                               tone(784, 0.1, "square", 0.3), tone(1046, 0.25, "square", 0.3)), 2, 6)
    s["oof"] = crush(growl(140, 90, 0.2, 0.4, 0.6, 23), 2, 6)
    s["noway"] = s["oof"]
    s["plpain"] = crush(growl(220, 150, 0.35, 0.45, 0.7, 24), 2, 6, 1.4)
    s["pldeth"] = crush(scream(330, 120, 1.0, 0.7, 25), 2, 6, 1.4)
    s["pdiehi"] = crush(mix(scream(420, 100, 1.2, 0.7, 26), bang(0.6, 0.5, 4, 3, 27)), 2, 6, 1.4)
    s["slop"] = crush(mix(bang(0.6, 0.9, 5, 14, 28, body=0.5, f0=60), growl(80, 40, 0.5, 0.95, 0.4, 29)), 2, 6, 1.6)
    s["posit"] = crush(growl(120, 70, 0.6, 0.55, 0.6, 30), 2, 6, 1.5)
    s["posact"] = crush(growl(90, 80, 0.7, 0.7, 0.4, 31), 2, 6, 1.3)
    s["popain"] = crush(growl(180, 110, 0.35, 0.5, 0.6, 32), 2, 6, 1.5)
    s["podth"] = crush(scream(200, 70, 0.9, 0.6, 33), 2, 6, 1.4)
    s["bgsit"] = crush(mix(growl(90, 160, 0.9, 0.6, 0.6, 34), growl(140, 60, 0.9, 0.7, 0.3, 35)), 2, 6, 1.6)
    s["bgact"] = crush(growl(70, 90, 0.8, 0.75, 0.45, 36), 2, 6, 1.4)
    s["bgdth"] = crush(scream(260, 60, 1.1, 0.6, 37), 2, 6, 1.6)
    s["sgtsit"] = crush(growl(60, 110, 0.8, 0.7, 0.7, 38), 2, 6, 1.8)
    s["sgtatk"] = crush(mix(growl(100, 60, 0.4, 0.6, 0.7, 39), bang(0.3, 0.5, 9, 3, 40)), 2, 6, 1.7)
    s["sgtdth"] = crush(growl(110, 40, 1.0, 0.65, 0.7, 41), 2, 6, 1.7)
    s["dmact"] = crush(growl(55, 70, 0.9, 0.8, 0.45, 42), 2, 6, 1.4)
    s["dmpain"] = crush(growl(150, 80, 0.4, 0.6, 0.6, 43), 2, 6, 1.6)
    s["cacsit"] = crush(growl(70, 140, 1.0, 0.5, 0.7, 44), 2, 6, 1.8)
    s["cacdth"] = crush(growl(140, 40, 1.2, 0.55, 0.7, 45), 2, 6, 1.8)
    s["brssit"] = crush(mix(growl(45, 90, 1.3, 0.6, 0.8, 46), growl(90, 45, 1.3, 0.5, 0.4, 47)), 2, 6, 2.0)
    s["brsdth"] = crush(growl(100, 30, 1.6, 0.6, 0.8, 48), 2, 6, 2.0)
    s["sklatk"] = crush(mix(bang(0.6, 0.5, 3, 12, 49), growl(120, 260, 0.6, 0.6, 0.4, 50)), 2, 6, 1.5)
    s["claw"] = crush(mix(bang(0.2, 0.8, 18, 1, 51), tone(1800, 0.1, "saw", 0.15, slide_to=600)), 2, 6, 1.5)
    s["tink"] = crush(tone(1200, 0.04, "square", 0.25), 2, 6)
    s["menu"] = crush(bang(0.12, 0.6, 30, 2, 52, body=0.4, f0=200), 2, 6)
    # the ripsaw: a two-stroke snarl; biting meat adds a wet chop
    s["sawful"] = crush(mix(tone(95, 0.14, "saw", 0.45, slide_to=120), tone(190, 0.14, "square", 0.2, slide_to=240),
                            bang(0.14, 0.25, 6, 2, 53)), 2, 6, 1.7)
    s["sawhit"] = crush(mix(s["sawful"], bang(0.14, 0.7, 14, 4, 54, body=0.6, f0=70)), 2, 6, 1.8)
    # the siege titan: a grinding roar and a long metallic death
    s["cybsit"] = crush(mix(growl(38, 70, 1.6, 0.7, 0.9, 55), tone(76, 1.6, "saw", 0.25, slide_to=55)), 2, 6, 2.0)
    s["cybdth"] = crush(mix(growl(80, 25, 2.2, 0.7, 0.9, 56), bang(2.2, 0.6, 1.2, 10, 57, body=0.8, f0=50)), 2, 6,
                        2.0)
    for name, x in s.items():
        bank.add(name, np.clip(x, -1, 1), volume=0.8)
    bank.add("music", np.clip(music_loop(), -1, 1), volume=0.45, loop=True)


def music_loop(bpm: float = 138.0) -> np.ndarray:
    """An original E-minor chug: palm-muted power chords, a lead hook, kick and snare. 8 bars."""
    beat = 60.0 / bpm
    eighth = beat / 2
    bars = 8
    n = int(bars * 4 * beat * RATE)
    out = np.zeros(n)
    rng = np.random.default_rng(5)

    def put(x: np.ndarray, t: float) -> None:
        i = int(t * RATE)
        if i < n:
            k = min(len(x), n - i)
            out[i:i + k] += x[:k]

    def chord(root: float, dur: float, mute: bool) -> np.ndarray:
        x = sum(tone(root * m, dur, "saw", 0.22, release=0.02) for m in (1.0, 1.5, 2.0))
        t = _t(len(x) / RATE)
        if mute:
            x = x * np.exp(-t * 14)
        return np.tanh(x * 4.0) * 0.32

    riff = [82.4, 82.4, 98.0, 82.4, 82.4, 110.0, 103.8, 98.0]  # E E G E E A G# G (eighths)
    roots = [82.4, 82.4, 73.4, 98.0, 82.4, 82.4, 110.0, 123.5]
    for bar in range(bars):
        t0 = bar * 4 * beat
        for i in range(8):
            r = riff[i] if bar % 4 != 3 else roots[bar] * (1 if i < 4 else 1.122)
            mute = i not in (2, 5)
            put(chord(r, eighth * (0.95 if mute else 1.9), mute), t0 + i * eighth)
        for b in range(4):
            put(bang(0.25, 0.9, 18, 6, 60 + b, body=1.0, f0=60) * 0.6, t0 + b * beat)
            if b in (1, 3):
                put(noise(0.22, 0.5, 14, 2, seed=70 + b) * 0.55, t0 + b * beat)
            for k in range(2):
                put(noise(0.04, 0.12, 60, 1, seed=80 + k) * 0.5, t0 + b * beat + k * eighth)
        if bar % 2 == 1:
            lead = [329.6, 392.0, 440.0, 392.0, 493.9, 440.0]
            for i, f in enumerate(lead):
                x = tone(f, eighth * 0.9, "square", 0.12, release=0.03) * (1 + 0.1 * rng.random())
                put(np.tanh(x * 2) * 0.5, t0 + 2 * beat + i * eighth * 0.66)
    return crush(out * 0.8, 2, 7)


__all__ = ["build_sounds", "music_loop", "envelope"]
