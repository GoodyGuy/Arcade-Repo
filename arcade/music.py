"""The home screen's chiptune loop and its sound effects, synthesised with numpy (an original tune: A minor, 112 bpm,
eight bars that loop seamlessly), and ``HomeAudio`` that plays them through a ``SoundBank`` (silent offscreen, muted
with the arcade)."""

from __future__ import annotations

import numpy as np

from app.arcade.audio import RATE, SoundBank, concat, mix, noise, silence, tone

BPM = 112
EIGHTH = 60.0 / BPM / 2
BAR = EIGHTH * 8

# 8th-note slots per bar: a MIDI note, "-" holds the previous note, "." is a rest
MELODY = (
    "76 - 81 - 79 76 - 74", "76 - - 72 74 - 72 69", "72 - 76 - 79 - 76 72", "74 - - - 71 74 79 -",
    "81 - 79 76 - 79 81 -", "84 - 81 - 79 77 76 -", "74 - 76 79 - 76 74 71", "68 - 71 - 76 - 74 -",
)
CHORDS = ((57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62), (57, 60, 64), (53, 57, 60), (55, 59, 62),
          (52, 56, 59))  # Am F C G Am F G E
BASS = (45, 41, 48, 43, 45, 41, 43, 40)


def midi(note: float) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def pulse(freq: float, dur: float, duty: float = 0.25, volume: float = 0.3, decay: float = 0.0,
          release: float = 0.03, vibrato: float = 0.0) -> np.ndarray:
    """A pulse wave (softer than a square at duty 0.25 / 0.125), with an optional vibrato."""
    t = np.arange(max(int(dur * RATE), 1)) / RATE
    f = freq * (1 + vibrato * np.sin(2 * np.pi * 5.5 * t) * np.clip(t * 4 - 0.4, 0, 1))
    phase = np.cumsum(f) / RATE % 1.0
    x = np.where(phase < duty, 1.0, -1.0) * volume
    if decay:
        x = x * np.exp(-decay * t)
    a, r = min(int(0.004 * RATE), len(x)), min(int(release * RATE), len(x))
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if r:
        x[-r:] *= np.linspace(1, 0, r)
    return x


def _place(track: np.ndarray, part: np.ndarray, start: float) -> None:
    """Add ``part`` at ``start`` s; what runs past the end wraps to the beginning (a seamless loop)."""
    i = int(round(start * RATE)) % len(track)
    n = len(part)
    end = min(i + n, len(track))
    track[i:end] += part[:end - i]
    if i + n > len(track):
        rest = part[end - i:]
        track[:len(rest)] += rest


def theme_loop(tempo_scale: float = 1.0) -> np.ndarray:
    """The home screen music: lead, arpeggio, bass and drums, eight bars."""
    eighth = EIGHTH / tempo_scale
    bar = eighth * 8
    track = np.zeros(int(round(bar * 8 * RATE)))
    for b, line in enumerate(MELODY):  # lead
        slots = line.split()
        k = 0
        while k < len(slots):
            if slots[k] in ("-", "."):
                k += 1
                continue
            n = 1
            while k + n < len(slots) and slots[k + n] == "-":
                n += 1
            note = float(slots[k])
            dur = n * eighth
            _place(track, pulse(midi(note), dur * 0.95, 0.25, 0.16, decay=0.6, vibrato=0.006 if n > 1 else 0.0),
                   b * bar + k * eighth)
            _place(track, pulse(midi(note), dur * 0.9, 0.125, 0.05, decay=1.2), b * bar + k * eighth + eighth * 0.75)
            k += n
    for b, chord in enumerate(CHORDS):  # 16th-note arpeggio
        for s in range(16):
            note = chord[(0, 1, 2, 1)[s % 4]] + 12
            _place(track, pulse(midi(note), eighth * 0.42, 0.125, 0.045, decay=9.0, release=0.01),
                   b * bar + s * eighth / 2)
    for b, root in enumerate(BASS):  # octave bass
        for s in range(8):
            note = root + (12 if s % 2 else 0)
            _place(track, tone(midi(note), eighth * 0.85, "triangle", 0.30, release=0.02), b * bar + s * eighth)
    kick = tone(140, 0.16, "sine", 0.55, slide_to=42, decay=14, release=0.03)
    snare = mix(noise(0.14, 0.22, decay=22, seed=3), tone(190, 0.08, "triangle", 0.12, decay=30))
    hat = noise(0.035, 0.07, decay=70, seed=11)
    for b in range(8):
        for beat in range(4):
            t0 = b * bar + beat * eighth * 2
            _place(track, kick if beat % 2 == 0 else snare, t0)
            _place(track, hat, t0 + eighth)
        if b % 4 == 3:  # a little fill
            _place(track, snare * 0.6, b * bar + eighth * 7)
            _place(track, snare * 0.4, b * bar + eighth * 7.5)
    peak = np.max(np.abs(track)) or 1.0
    return track * (0.85 / peak)


def _arp(notes: tuple[float, ...], step: float, wave: str = "square", volume: float = 0.25) -> np.ndarray:
    return concat(*(tone(midi(n), step, wave, volume, release=0.02) for n in notes))


def _coin() -> np.ndarray:
    fall = concat(tone(1318, 0.05, "sine", 0.3, decay=30), silence(0.07), tone(1568, 0.05, "sine", 0.25, decay=30),
                  silence(0.05))
    clink = mix(tone(1976, 0.35, "sine", 0.35, decay=9), tone(2637, 0.3, "sine", 0.22, decay=12),
                tone(3951, 0.15, "sine", 0.1, decay=20), noise(0.03, 0.15, decay=90, seed=5))
    return concat(fall, clink)


# name: (make the samples, volume, loop) - made the first time the sound plays
SOUNDS = {
    "music": (theme_loop, 0.32, True),
    "move": (lambda: mix(tone(midi(76), 0.05, "square", 0.18, slide_to=midi(83)), noise(0.03, 0.04, decay=60)), 0.6,
             False),
    "bump": (lambda: tone(midi(48), 0.08, "square", 0.2, slide_to=midi(43)), 0.6, False),
    "coin": (_coin, 0.8, False),
    "ready": (lambda: concat(_arp((72, 76, 79, 84), 0.06, "square", 0.2),
                             tone(midi(88), 0.25, "square", 0.2, decay=4)), 0.6, False),
    "ticket": (lambda: tone(2200, 0.018, "square", 0.12, release=0.005), 0.5, False),
    "bonus": (lambda: _arp((72, 76, 79, 84, 88, 91), 0.055, "square", 0.18), 0.6, False),
    "pop": (lambda: mix(tone(400, 0.12, "square", 0.25, slide_to=1400), noise(0.1, 0.2, decay=30, seed=9)), 0.6,
            False),
    "hit": (lambda: tone(300, 0.07, "square", 0.22, slide_to=180), 0.5, False),
    "prize": (lambda: concat(_arp((84, 88, 91), 0.05, "square", 0.2),
                             mix(tone(midi(96), 0.3, "square", 0.18, decay=6), noise(0.25, 0.12, decay=10, seed=2))),
              0.65, False),
    "deny": (lambda: concat(tone(110, 0.12, "square", 0.25), silence(0.03), tone(98, 0.18, "square", 0.25)), 0.5,
             False),
    "open": (lambda: concat(tone(midi(67), 0.05, "square", 0.15), tone(midi(74), 0.08, "square", 0.15)), 0.5, False),
    "close": (lambda: concat(tone(midi(74), 0.05, "square", 0.15), tone(midi(67), 0.08, "square", 0.15)), 0.5, False),
    "party": (lambda: concat(_arp((60, 64, 67, 72, 76, 79, 84, 88), 0.05, "square", 0.2),
                             mix(tone(midi(84), 0.5, "square", 0.15, decay=3),
                                 tone(midi(88), 0.5, "square", 0.12, decay=3),
                                 tone(midi(91), 0.5, "square", 0.1, decay=3))), 0.6, False),
    "confetti": (lambda: mix(noise(0.25, 0.3, decay=12, smooth=2, seed=4),
                             tone(900, 0.1, "square", 0.1, slide_to=1800)), 0.55, False),
    "power": (lambda: mix(tone(60, 0.35, "saw", 0.25, slide_to=220, decay=4), noise(0.3, 0.06, decay=8, smooth=4)),
              0.45, False),
    "meow": (lambda: concat(tone(700, 0.08, "triangle", 0.2, slide_to=1100),
                            tone(1100, 0.16, "triangle", 0.2, slide_to=600)), 0.5, False),
}


def build_sounds(bank: SoundBank, names: tuple[str, ...] | None = None) -> None:
    """The home screen's effects (and its music) into ``bank`` (``names``: only these)."""
    for name in names or tuple(SOUNDS):
        make, volume, loop = SOUNDS[name]
        bank.add(name, make(), volume=volume, loop=loop)


class HomeAudio:
    """The home screen's sounds; the music loop runs while ``set_music(True)`` (never offscreen or muted). Each sound
    is made the first time it plays."""

    def __init__(self) -> None:
        self.bank = SoundBank()
        self.music_on = False

    def play(self, name: str, volume: float = 1.0) -> None:
        if not self.bank._enabled or name not in SOUNDS:  # noqa: SLF001 - nothing to make when there is no sound
            return
        if name not in self.bank:
            build_sounds(self.bank, (name,))
        self.bank.play(name, volume)

    def set_music(self, on: bool) -> None:
        from app.arcade import audio

        on = bool(on) and not audio.is_muted()
        if on == self.music_on:
            return
        self.music_on = on
        if on:
            self.play("music")
        else:
            self.bank.stop("music")

    def stop_all(self) -> None:
        self.music_on = False
        self.bank.stop_all()
