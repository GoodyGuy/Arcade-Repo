"""Vanguard's sounds, all made with the arcade's little synthesiser (no sound files)."""

from __future__ import annotations

import numpy as np

from app.arcade.audio import RATE, concat, envelope, mix, noise, silence, tone


def _chord(freqs: tuple, dur: float, wave: str = "triangle", vol: float = 0.3, decay: float = 3.0) -> np.ndarray:
    return mix(*[tone(f, dur, wave, vol, decay=decay, release=0.08) for f in freqs])


def _arp(freqs: tuple, step: float, wave: str = "square", vol: float = 0.3, hold: float = 0.25) -> np.ndarray:
    out = np.zeros(int((step * len(freqs) + hold) * RATE) + 1)
    for i, f in enumerate(freqs):
        n = tone(f, step + hold, wave, vol, decay=5.0, release=0.1)
        a = int(i * step * RATE)
        out[a:a + len(n)] += n[:len(out) - a]
    return out


def build(sfx) -> None:  # noqa: ANN001 - audio.SoundBank
    add = sfx.add
    # weapons
    add("rivet", mix(noise(0.09, 0.5, 30, 2, seed=3), tone(170, 0.09, "square", 0.3, slide_to=80, decay=24)), 0.5)
    add("rifle", mix(noise(0.2, 0.7, 16, 1, seed=5), tone(420, 0.16, "saw", 0.35, slide_to=90, decay=14),
                     tone(1800, 0.05, "sine", 0.2, decay=40)), 0.6)
    add("smg", mix(noise(0.05, 0.5, 45, 1, seed=9), tone(520, 0.05, "square", 0.22, slide_to=220, decay=40)), 0.4)
    add("launcher", mix(noise(0.2, 0.5, 14, 6, seed=11), tone(120, 0.22, "sine", 0.6, slide_to=50, decay=10)), 0.6)
    add("carbine", mix(tone(1250, 0.07, "triangle", 0.4, slide_to=620, decay=28),
                       tone(2500, 0.05, "sine", 0.18, decay=40), noise(0.04, 0.2, 50, 1, seed=13)), 0.45)
    add("flame", mix(noise(0.24, 0.42, 3, 5, seed=17), tone(70, 0.24, "saw", 0.12, decay=2)), 0.4)
    add("mist", mix(noise(0.24, 0.3, 3, 2, seed=19), tone(900, 0.24, "sine", 0.05, slide_to=1100)), 0.35)
    add("reload", concat(noise(0.04, 0.3, 40, 2, seed=2), silence(0.12), tone(700, 0.04, "square", 0.2, decay=40),
                         silence(0.05), noise(0.05, 0.35, 40, 1, seed=4)), 0.5)
    add("melee", mix(noise(0.14, 0.4, 14, 8, seed=6), tone(240, 0.12, "sine", 0.3, slide_to=90)), 0.5)
    # hits
    add("hit", tone(1100, 0.045, "square", 0.3, decay=50), 0.4)
    add("hit_head", concat(tone(1500, 0.04, "square", 0.3, decay=50), tone(2100, 0.07, "square", 0.3, decay=30)), 0.45)
    add("hurt", mix(noise(0.1, 0.4, 22, 6, seed=8), tone(150, 0.12, "sine", 0.4, slide_to=80)), 0.5)
    add("elim", _arp((880, 1320, 1760), 0.07, "triangle", 0.35, 0.3), 0.7)
    add("death", mix(tone(330, 0.7, "saw", 0.3, slide_to=55, decay=3), noise(0.5, 0.3, 5, 8, seed=21)), 0.6)
    add("down", mix(tone(260, 0.25, "triangle", 0.3, slide_to=110, decay=8), noise(0.12, 0.2, 20, 6, seed=23)), 0.4)
    add("respawn", _arp((392, 523, 659, 784), 0.06, "triangle", 0.3, 0.3), 0.5)
    add("swap", _arp((523, 784), 0.06, "square", 0.22, 0.12), 0.4)
    add("shield_break", mix(noise(0.25, 0.4, 12, 1, seed=25), tone(2200, 0.25, "sine", 0.25, slide_to=500, decay=10)),
        0.5)
    add("boom", mix(noise(0.6, 0.9, 6, 10, seed=27), tone(75, 0.5, "sine", 0.8, slide_to=32, decay=6)), 0.75)
    add("break", mix(noise(0.3, 0.5, 12, 2, seed=29), tone(900, 0.25, "triangle", 0.3, slide_to=200, decay=10)), 0.5)
    add("pack", _arp((660, 990), 0.06, "sine", 0.4, 0.2), 0.5)
    # abilities
    add("ready", tone(1320, 0.07, "sine", 0.3, decay=20), 0.3)
    add("cast", mix(tone(500, 0.1, "triangle", 0.3, slide_to=900), noise(0.06, 0.15, 30, 2, seed=31)), 0.4)
    add("ult_ready", _arp((523, 659, 784, 1047), 0.08, "triangle", 0.35, 0.5), 0.7)
    add("ult_ally", mix(_arp((392, 523, 659, 784, 1047), 0.06, "saw", 0.22, 0.5),
                        tone(98, 0.8, "sine", 0.4, decay=3)), 0.75)
    add("ult_enemy", mix(_arp((466, 440, 370, 311, 233), 0.07, "saw", 0.24, 0.5),
                         tone(58, 0.9, "saw", 0.3, decay=3)), 0.8)
    add("engine", mix(tone(60, 0.7, "saw", 0.4, slide_to=140), noise(0.7, 0.2, 2, 4, seed=33)), 0.6)
    add("barrier", mix(tone(300, 0.3, "sine", 0.4, slide_to=900), tone(1800, 0.3, "sine", 0.12, decay=8)), 0.5)
    add("windup", tone(120, 0.45, "saw", 0.4, slide_to=420), 0.5)
    add("quake", mix(noise(0.9, 1.0, 4, 14, seed=35), tone(50, 0.9, "sine", 0.9, slide_to=24, decay=4)), 0.9)
    add("absorb", mix(tone(900, 0.4, "sine", 0.3, slide_to=200), noise(0.3, 0.15, 6, 3, seed=37)), 0.5)
    add("lob", mix(noise(0.1, 0.3, 20, 8, seed=39), tone(300, 0.14, "sine", 0.3, slide_to=520)), 0.45)
    add("spring", tone(200, 0.25, "triangle", 0.45, slide_to=1100, decay=4), 0.5)
    add("flare", mix(noise(0.3, 0.35, 8, 2, seed=41), tone(700, 0.3, "saw", 0.2, slide_to=1500)), 0.5)
    add("mark", _arp((1568, 1175, 1568), 0.06, "sine", 0.3, 0.2), 0.5)
    add("dash", mix(noise(0.16, 0.4, 14, 3, seed=43), tone(900, 0.16, "sine", 0.2, slide_to=300)), 0.5)
    add("wind", _arp((440, 554, 659), 0.07, "sine", 0.35, 0.3), 0.5)
    add("whistle", tone(2400, 0.9, "sine", 0.3, slide_to=500, attack=0.2), 0.6)
    add("buff", _arp((587, 740, 880, 1175), 0.05, "square", 0.2, 0.2), 0.45)
    add("deploy", concat(tone(220, 0.06, "square", 0.3, decay=20), tone(440, 0.06, "square", 0.3, decay=20),
                         tone(660, 0.12, "triangle", 0.3, decay=10)), 0.45)
    add("tag", mix(tone(1047, 0.25, "sine", 0.35, decay=8), tone(1568, 0.25, "sine", 0.2, decay=8)), 0.5)
    add("splash", mix(noise(0.3, 0.5, 9, 4, seed=45), tone(500, 0.2, "sine", 0.2, slide_to=160)), 0.5)
    add("clang", mix(tone(420, 0.3, "square", 0.3, decay=12), tone(633, 0.3, "triangle", 0.25, decay=10),
                     noise(0.06, 0.4, 40, 1, seed=47)), 0.55)
    # the match
    add("go", _arp((523, 523, 784), 0.12, "square", 0.3, 0.4), 0.6)
    add("tick", tone(880, 0.06, "square", 0.25, decay=30), 0.4)
    add("unlock", _arp((659, 880, 1319), 0.09, "triangle", 0.35, 0.4), 0.6)
    add("capture_us", mix(_chord((523, 659, 784), 0.7, "triangle", 0.25), _arp((784, 1047), 0.12, "square", 0.2)),
        0.7)
    add("capture_them", mix(_chord((311, 370, 466), 0.7, "saw", 0.2), tone(110, 0.6, "saw", 0.25, decay=3)), 0.7)
    add("checkpoint", _arp((523, 659, 784, 1047, 1319), 0.08, "triangle", 0.3, 0.5), 0.7)
    add("overtime", concat(*[envelope(tone(740 if i % 2 else 988, 0.16, "square", 0.3), 0.01, 0.03)
                             for i in range(6)]), 0.6)
    add("round_win", _arp((523, 659, 784, 1047), 0.11, "triangle", 0.35, 0.6), 0.7)
    add("round_lose", _arp((392, 349, 311, 262), 0.13, "triangle", 0.35, 0.6), 0.7)
    add("victory", concat(_arp((523, 659, 784), 0.12, "square", 0.25, 0.1),
                          _chord((523, 659, 784, 1047), 1.2, "triangle", 0.25, 1.5)), 0.8)
    add("defeat", concat(_arp((392, 370, 311), 0.16, "saw", 0.22, 0.1),
                         _chord((196, 233, 294), 1.2, "triangle", 0.3, 1.5)), 0.8)
    # menus
    add("ui_move", tone(660, 0.04, "square", 0.2, decay=40), 0.35)
    add("ui_ok", _arp((660, 990), 0.05, "square", 0.22, 0.1), 0.4)
    add("ui_back", _arp((660, 440), 0.05, "square", 0.22, 0.1), 0.4)
