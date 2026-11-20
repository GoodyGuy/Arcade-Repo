"""BREACH POINT's sounds, all synthesised with the arcade's numpy helpers: a report per gun class (and a soft one for
suppressed guns), handling, footsteps, hits, the charge's plant, beep, defuse and blast, a cue per ability family, the
round and match stings, and menu ticks."""

from __future__ import annotations

import numpy as np

from app.arcade.audio import RATE, SoundBank, concat, envelope, mix, noise, silence, tone


def _f(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def _lp(x: np.ndarray, k: int) -> np.ndarray:
    return x if k <= 1 else np.convolve(x, np.ones(k) / k, mode="same")


def _shot(body: float, decay: float, smooth: int, boom: float, boom_to: float, crack: float = 0.0, vol: float = 0.9,
          seed: int = 3) -> np.ndarray:
    parts = [noise(body, vol, decay, smooth, seed=seed),
             tone(boom, min(body, 0.22), "sine", vol * 0.85, slide_to=boom_to, release=0.07, decay=decay * 0.6)]
    if crack:
        parts.append(noise(0.028, vol * 0.8, 70, 1, seed=seed + 1))
        parts.append(tone(crack, 0.018, "square", 0.22, release=0.01))
    tail = noise(body * 2.4, vol * 0.16, decay * 0.3, smooth * 3 + 5, seed=seed + 2)
    return mix(*parts, concat(silence(0.03), tail))


def _soft(dur: float, vol: float, seed: int) -> np.ndarray:
    return mix(_lp(noise(dur, vol, 34, 4, seed=seed), 3), tone(210, 0.05, "sine", vol * 0.5, slide_to=90, release=0.03),
               noise(0.012, vol * 0.5, 120, 1, seed=seed + 1))


def _click(vol: float = 0.5, f: float = 2400, seed: int = 9) -> np.ndarray:
    return mix(noise(0.03, vol, 90, 1, seed=seed), tone(f, 0.015, "square", vol * 0.3, release=0.008))


def _notes(notes: list, wave: str, vol: float, beat: float, decay: float = 3.0) -> np.ndarray:
    out = []
    for n, beats in notes:
        d = beats * beat
        out.append(silence(d) if n <= 0 else mix(tone(_f(n), d, wave, vol, release=min(0.06, d * 0.3), decay=decay),
                                                 tone(_f(n - 12), d, "sine", vol * 0.6, release=0.05, decay=decay)))
    return concat(*out)


def _whoosh(dur: float, vol: float, seed: int = 5, smooth: int = 6) -> np.ndarray:
    x = noise(dur, vol, 0.0, smooth, seed=seed)
    t = np.arange(len(x)) / RATE
    return x * np.sin(np.pi * t / dur) ** 2


def build(bank: SoundBank) -> None:
    add = bank.add
    # guns
    add("shot_sidearm", _shot(0.16, 22, 2, 260, 80, 1900, 0.8, 3), 0.75)
    add("shot_sidearm_q", _soft(0.12, 0.7, 4), 0.6)
    add("shot_smg", _shot(0.13, 26, 2, 300, 110, 2300, 0.75, 5), 0.7)
    add("shot_smg_q", _soft(0.1, 0.65, 6), 0.55)
    add("shot_rifle", _shot(0.2, 18, 2, 210, 60, 1500, 0.95, 7), 0.85)
    add("shot_rifle_q", _soft(0.14, 0.8, 8), 0.62)
    add("shot_marksman", _shot(0.24, 15, 2, 180, 55, 1300, 1.0, 9), 0.9)
    add("shot_shotgun", _shot(0.3, 12, 3, 150, 45, 900, 1.0, 11), 0.95)
    add("shot_sniper", _shot(0.42, 9, 3, 130, 38, 1100, 1.0, 13), 1.0)
    add("shot_lmg", _shot(0.19, 17, 3, 170, 55, 1200, 0.95, 15), 0.85)
    add("dry", _click(0.35, 3000, 17), 0.5)
    add("reload", concat(_click(0.4, 1600, 19), silence(0.16), _click(0.5, 1100, 20), silence(0.2),
                         mix(_click(0.55, 2000, 21), tone(700, 0.04, "square", 0.12))), 0.55)
    add("equip", concat(_whoosh(0.08, 0.2, 23, 3), _click(0.3, 1800, 24)), 0.4)
    add("knife", _whoosh(0.16, 0.5, 25, 2), 0.6)
    add("knife_hit", mix(noise(0.1, 0.7, 30, 3, seed=26), tone(140, 0.1, "sine", 0.6, slide_to=70)), 0.8)
    add("pickup", concat(_click(0.4, 1200, 27), tone(660, 0.06, "triangle", 0.3)), 0.5)
    # bodies
    add("step", mix(_lp(noise(0.07, 0.5, 40, 5, seed=29), 3), tone(95, 0.05, "sine", 0.4, slide_to=60)), 0.42)
    add("land", mix(_lp(noise(0.12, 0.7, 25, 6, seed=31), 4), tone(80, 0.1, "sine", 0.6, slide_to=45)), 0.55)
    add("hurt", mix(noise(0.07, 0.5, 40, 3, seed=33), tone(190, 0.09, "saw", 0.3, slide_to=110)), 0.5)
    add("death", mix(_lp(noise(0.3, 0.6, 9, 6, seed=35), 4), tone(160, 0.3, "saw", 0.3, slide_to=55, decay=6)), 0.6)
    add("hit", mix(tone(1500, 0.04, "square", 0.3, release=0.02), noise(0.03, 0.3, 80, 1, seed=37)), 0.5)
    add("headshot", mix(tone(2200, 0.07, "triangle", 0.5, release=0.04), tone(3300, 0.05, "sine", 0.3),
                        noise(0.03, 0.3, 80, 1, seed=39)), 0.6)
    add("kill", concat(tone(_f(76), 0.07, "triangle", 0.5), tone(_f(83), 0.14, "triangle", 0.5, decay=6)), 0.6)
    # the charge
    add("plant_start", concat(*[mix(_click(0.4, 1500 + i * 150, 41 + i), tone(500 + i * 60, 0.03, "square", 0.15))
                                for i in range(4)]), 0.6)
    add("planted", concat(tone(_f(64), 0.1, "square", 0.35), tone(_f(71), 0.1, "square", 0.35),
                          tone(_f(76), 0.28, "square", 0.35, decay=4)), 0.6)
    add("beep", tone(1240, 0.07, "square", 0.45, release=0.03), 0.5)
    add("defuse_start", concat(tone(900, 0.05, "sine", 0.4), silence(0.04), tone(1200, 0.08, "sine", 0.4)), 0.6)
    add("defused", concat(tone(_f(79), 0.09, "sine", 0.5), tone(_f(74), 0.09, "sine", 0.5),
                          tone(_f(67), 0.4, "sine", 0.5, decay=4)), 0.7)
    add("boom", mix(_lp(noise(1.6, 1.0, 2.6, 8, seed=47), 6), tone(70, 1.2, "sine", 1.0, slide_to=24, decay=2.5),
                    noise(0.08, 1.0, 30, 1, seed=48)), 1.0)
    # abilities
    add("ability", concat(_whoosh(0.1, 0.3, 51, 3), tone(880, 0.05, "triangle", 0.25)), 0.5)
    add("throw", _whoosh(0.18, 0.45, 53, 4), 0.5)
    add("bounce", _click(0.3, 900, 55), 0.35)
    add("flash", mix(noise(0.25, 0.9, 14, 1, seed=57), tone(3200, 0.5, "sine", 0.35, slide_to=2600, decay=5)), 0.8)
    add("smoke", _lp(_whoosh(0.7, 0.6, 59, 10), 6), 0.6)
    add("zone", mix(_whoosh(0.45, 0.4, 61, 8), tone(160, 0.4, "saw", 0.18, slide_to=90, decay=4)), 0.55)
    add("fire", mix(_lp(noise(0.7, 0.7, 4, 5, seed=63), 4), tone(120, 0.3, "sine", 0.4, slide_to=60)), 0.7)
    add("concuss", mix(tone(90, 0.5, "sine", 0.9, slide_to=40, decay=5), _lp(noise(0.3, 0.7, 12, 5, seed=65), 4)), 0.8)
    add("blast", mix(_lp(noise(0.7, 1.0, 6, 5, seed=67), 4), tone(85, 0.5, "sine", 0.9, slide_to=32, decay=4),
                     noise(0.05, 0.9, 40, 1, seed=68)), 0.95)
    add("ping", concat(tone(1320, 0.05, "sine", 0.4), tone(1760, 0.18, "sine", 0.4, decay=8)), 0.5)
    add("sensor", concat(tone(1500, 0.05, "square", 0.25), silence(0.04), tone(1500, 0.05, "square", 0.25)), 0.5)
    add("barricade_down", mix(_lp(noise(0.35, 0.6, 10, 6, seed=71), 4), tone(110, 0.25, "sine", 0.4, slide_to=60)), 0.6)
    add("ult", mix(tone(_f(52), 0.7, "saw", 0.3, slide_to=_f(64), decay=2), _whoosh(0.7, 0.4, 73, 8),
                   tone(_f(76), 0.6, "sine", 0.25, attack=0.3)), 0.7)
    add("ult_ready", concat(tone(_f(72), 0.08, "triangle", 0.4), tone(_f(76), 0.08, "triangle", 0.4),
                            tone(_f(79), 0.2, "triangle", 0.4, decay=5)), 0.5)
    # rounds
    add("round_start", _notes([(57, 1), (64, 1), (0, 0.5), (69, 2)], "triangle", 0.3, 0.11), 0.55)
    add("go", concat(tone(_f(69), 0.08, "square", 0.3), tone(_f(81), 0.22, "square", 0.3, decay=5)), 0.5)
    add("round_win", _notes([(64, 1), (68, 1), (71, 1), (76, 3)], "triangle", 0.34, 0.1), 0.65)
    add("round_loss", _notes([(64, 1), (63, 1), (59, 1), (52, 3)], "triangle", 0.34, 0.11), 0.6)
    add("match_win", _notes([(60, 1), (64, 1), (67, 1), (72, 2), (0, 0.5), (67, 1), (72, 4)], "triangle", 0.36, 0.12,
                            2.0), 0.7)
    add("match_loss", _notes([(60, 2), (58, 2), (55, 2), (48, 5)], "triangle", 0.36, 0.14, 2.0), 0.65)
    add("tick10", tone(980, 0.05, "square", 0.25, release=0.02), 0.4)
    # menus
    add("menu_move", _click(0.25, 2000, 81), 0.35)
    add("menu_select", concat(tone(_f(72), 0.04, "triangle", 0.35), tone(_f(79), 0.09, "triangle", 0.35)), 0.45)
    add("menu_back", concat(tone(_f(72), 0.04, "triangle", 0.3), tone(_f(65), 0.08, "triangle", 0.3)), 0.4)
    add("buy", concat(_click(0.4, 1400, 83), tone(_f(84), 0.07, "triangle", 0.3)), 0.5)
    add("deny", tone(180, 0.12, "square", 0.3, release=0.04), 0.4)
    _ = envelope
