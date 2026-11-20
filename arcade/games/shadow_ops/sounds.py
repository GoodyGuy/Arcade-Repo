"""SHADOW OPS II sounds, synthesised with numpy: a gunshot per weapon class, reloads, hit markers, the knife and
grenades, zombie groans and swipes, planks, the Lucky Crate and tonic jingles (original melodies), power-up chimes, round
chords, multiplayer kill and scorestreak cues, menu ticks and two ambient loops."""

from __future__ import annotations

import numpy as np

from app.arcade.audio import RATE, SoundBank, concat, envelope, mix, noise, silence, tone


def _f(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def _lowpass(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _shot(body: float, decay: float, smooth: int, boom: float, boom_to: float, crack: float = 0.0,
          vol: float = 0.9, seed: int = 3) -> np.ndarray:
    parts = [noise(body, vol, decay, smooth, seed=seed),
             tone(boom, min(body, 0.25), "sine", vol * 0.9, slide_to=boom_to, release=0.08, decay=decay * 0.6)]
    if crack:
        parts.append(noise(0.03, vol * 0.8, 60, 1, seed=seed + 1))
        parts.append(tone(crack, 0.02, "square", 0.25, release=0.01))
    tail = noise(body * 2.2, vol * 0.18, decay * 0.35, smooth * 3 + 4, seed=seed + 2)  # the echo
    return mix(*parts, concat(silence(0.03), tail))


def _click(vol: float = 0.5, f: float = 2400, seed: int = 9) -> np.ndarray:
    return mix(noise(0.03, vol, 90, 1, seed=seed), tone(f, 0.015, "square", vol * 0.3, release=0.008))


def _melody(notes: list[tuple[int, float]], wave: str, vol: float, bpm: float, decay: float = 4.0,
            bass: bool = False) -> np.ndarray:
    beat = 60.0 / bpm
    out = []
    for n, beats in notes:
        d = beats * beat
        if n <= 0:
            out.append(silence(d))
            continue
        x = tone(_f(n), d, wave, vol, release=min(0.05, d * 0.3), decay=decay)
        if bass:
            x = mix(x, tone(_f(n - 24), d, "triangle", vol * 0.7, release=0.04, decay=decay * 0.6))
        out.append(x)
    return concat(*out)


def _chord(notes: list[int], dur: float, wave: str = "saw", vol: float = 0.25, attack: float = 0.2,
           release: float = 0.6, smooth: int = 6) -> np.ndarray:
    x = mix(*[tone(_f(n), dur, wave, vol / len(notes) * 2, attack=attack, release=release) for n in notes])
    return _lowpass(x, smooth)


def _groan(base: float, dur: float, seed: int, harsh: float = 0.0) -> np.ndarray:
    t = np.arange(int(dur * RATE)) / RATE
    rng = np.random.default_rng(seed)
    wob = 1 + 0.08 * np.sin(2 * np.pi * (3 + rng.random() * 3) * t) + 0.15 * np.sin(np.pi * t / dur)
    f = base * wob * np.linspace(1.0, 0.75 + rng.random() * 0.3, len(t))
    phase = 2 * np.pi * np.cumsum(f) / RATE
    x = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
    x = _lowpass(x, 5) * 0.5 + _lowpass(rng.uniform(-1, 1, len(t)), 3) * (0.12 + harsh)
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 7 * t + rng.random() * 6) ** 2
    return envelope(x * am * np.sin(np.pi * t / dur) ** 0.5 * 0.7, 0.05, 0.15)


def _whoosh(dur: float, vol: float, seed: int = 5) -> np.ndarray:
    x = noise(dur, vol, 0.0, 6, seed=seed)
    t = np.arange(len(x)) / RATE
    return x * np.sin(np.pi * t / dur) ** 2


def build(bank: SoundBank) -> None:
    """Add every sound to ``bank``."""
    add = bank.add
    # guns
    add("pistol", _shot(0.16, 28, 1, 170, 70, 2600, 0.85, 3), 0.8)
    add("smg", _shot(0.11, 40, 2, 210, 90, 3000, 0.75, 4), 0.7)
    add("ar", _shot(0.2, 20, 2, 120, 50, 2200, 0.9, 5), 0.8)
    add("rifle", _shot(0.26, 15, 2, 110, 42, 1900, 0.95, 6), 0.85)
    add("lmg", _shot(0.22, 18, 3, 95, 40, 1800, 0.95, 7), 0.85)
    add("shotgun", mix(_shot(0.42, 9, 3, 85, 34, 1500, 1.0, 8), concat(silence(0.4), _click(0.3, 900),
                                                                        silence(0.1), _click(0.35, 700))), 0.95)
    add("sniper", _shot(0.75, 5, 4, 80, 30, 2400, 1.0, 10), 1.0)
    add("launcher", mix(tone(70, 0.3, "sine", 0.9, slide_to=40, decay=6), _whoosh(0.5, 0.4)), 0.9)
    add("raygun", mix(tone(1400, 0.22, "square", 0.25, slide_to=260, decay=8),
                      tone(700, 0.22, "sine", 0.35, slide_to=1800, decay=5), noise(0.08, 0.2, 30, 1)), 0.7)
    add("dry", _click(0.4, 1800), 0.6)
    add("reload", concat(_click(0.5, 1200, 11), silence(0.35), noise(0.05, 0.4, 40, 2, seed=12), silence(0.25),
                         _click(0.6, 900, 13), silence(0.05), _click(0.5, 1500, 14)), 0.6)
    add("weapon_take", concat(_click(0.5, 1100), silence(0.12), _click(0.6, 800), silence(0.05), _click(0.4, 1600)),
        0.7)
    add("hitmarker", mix(noise(0.035, 0.5, 80, 1, seed=21), tone(3200, 0.03, "sine", 0.3, release=0.02)), 0.6)
    add("headshot", mix(tone(1800, 0.12, "sine", 0.35, decay=20), noise(0.05, 0.4, 60, 1, seed=22)), 0.7)
    add("kill_ding", mix(tone(1320, 0.18, "sine", 0.35, decay=14), tone(1980, 0.18, "sine", 0.2, decay=18)), 0.7)
    add("knife_swing", _whoosh(0.22, 0.6, 31), 0.6)
    add("knife_hit", mix(_whoosh(0.15, 0.5, 32), noise(0.12, 0.7, 30, 4, seed=33),
                         tone(90, 0.12, "sine", 0.7, decay=20)), 0.8)
    add("throw", _whoosh(0.3, 0.5, 34), 0.5)
    add("explosion", mix(noise(1.3, 1.0, 3.2, 8, seed=35), tone(55, 0.8, "sine", 1.0, slide_to=30, decay=4),
                         noise(0.2, 0.8, 20, 1, seed=36)), 1.0)
    add("flashbang", mix(noise(0.3, 0.9, 12, 1, seed=37), tone(3500, 2.0, "sine", 0.18, release=1.5)), 0.8)
    # zombies
    for i, (b, d) in enumerate(((95, 1.4), (120, 1.1), (80, 1.7))):
        add(f"groan{i + 1}", _groan(b, d, 40 + i), 0.55)
    add("sprint_groan", _groan(230, 0.8, 44, harsh=0.25), 0.55)
    add("swipe", mix(_whoosh(0.25, 0.6, 45), _groan(150, 0.35, 46, 0.3)), 0.7)
    add("zombie_die", _groan(70, 0.6, 47, 0.2), 0.6)
    add("rise", noise(0.9, 0.6, 2.5, 12, seed=48), 0.6)
    add("player_hit", mix(noise(0.18, 0.8, 18, 3, seed=49), tone(120, 0.15, "sine", 0.6, slide_to=70, decay=12)),
        0.9)
    add("plank_tear", mix(noise(0.12, 0.9, 25, 1, seed=50), tone(300, 0.25, "saw", 0.2, slide_to=140, decay=8)),
        0.7)
    add("plank_fix", concat(*[mix(noise(0.05, 0.7, 50, 2, seed=51 + k), tone(220, 0.05, "sine", 0.5, decay=40),
                                  silence(0.09)) for k in range(3)]), 0.7)
    add("purchase", concat(tone(1568, 0.08, "sine", 0.4, decay=10), tone(2093, 0.25, "sine", 0.4, decay=8)), 0.7)
    add("deny", tone(110, 0.25, "square", 0.25, release=0.05), 0.6)
    add("door_open", mix(tone(180, 0.6, "saw", 0.2, slide_to=90, decay=3), noise(0.4, 0.6, 6, 8, seed=55)), 0.8)
    add("power_on", mix(tone(50, 2.0, "saw", 0.3, slide_to=120, attack=0.3, release=0.5),
                        concat(silence(0.1), noise(0.2, 0.9, 15, 2, seed=56))), 0.9)
    box_tune = [(76, 0.5), (79, 0.5), (83, 0.5), (88, 1), (86, 0.5), (83, 0.5), (79, 0.5), (81, 1), (84, 0.5),
                (88, 0.5), (91, 0.5), (88, 1.5)]
    add("box_jingle", _melody(box_tune, "sine", 0.3, 300, decay=6), 0.6)
    add("box_open", mix(tone(140, 0.5, "saw", 0.2, slide_to=260, decay=4), noise(0.3, 0.3, 8, 6, seed=57)), 0.7)
    add("box_close", mix(noise(0.2, 0.6, 15, 6, seed=58), tone(90, 0.2, "sine", 0.6, decay=15)), 0.7)
    add("bear", _melody([(84, 0.5), (80, 0.5), (77, 0.5), (73, 1.5)], "triangle", 0.35, 200, decay=3), 0.8)
    add("box_move", _whoosh(1.2, 0.6, 59), 0.7)
    add("box_land", mix(noise(0.5, 0.9, 6, 10, seed=60), tone(60, 0.4, "sine", 0.9, decay=8)), 0.8)
    add("pap_work", concat(*[mix(noise(0.25, 0.4, 6, 3, seed=61 + k), tone(80 + k * 30, 0.25, "square", 0.12))
                             for k in range(8)]), 0.7)
    add("pap_ding", mix(tone(1047, 0.8, "sine", 0.35, decay=3), tone(1568, 0.8, "sine", 0.25, decay=3)), 0.8)
    add("drink", concat(*[mix(tone(300 + k * 40, 0.08, "sine", 0.35, decay=25), noise(0.08, 0.2, 30, 6, seed=70 + k))
                          for k in range(5)], silence(0.1), tone(2600, 0.06, "sine", 0.3, decay=30)), 0.7)
    jingles = {
        "jugg": ([(45, 1), (45, 0.5), (48, 0.5), (50, 1), (45, 1), (53, 1), (52, 0.5), (50, 0.5), (48, 2)],
                 "square", 0.22, 150, True),
        "swift": ([(72, 0.5), (76, 0.5), (79, 0.5), (84, 0.5), (83, 0.5), (79, 0.5), (76, 0.5), (79, 0.5), (81, 1),
                   (77, 0.5), (81, 0.5), (84, 1.5)], "triangle", 0.35, 260, False),
        "double": ([(60, 0.75), (63, 0.25), (65, 0.5), (66, 0.5), (67, 1), (70, 0.5), (67, 0.5), (65, 0.5), (63, 0.5),
                    (60, 2)], "saw", 0.16, 170, True),
        "rally": ([(67, 0.5), (72, 0.5), (76, 0.5), (72, 0.5), (74, 0.5), (77, 0.5), (81, 1), (79, 0.5), (76, 0.5),
                   (72, 1.5)], "square", 0.18, 220, False),
        "stamina": ([(62, 0.5), (66, 0.5), (69, 0.5), (74, 1), (73, 0.5), (69, 0.5), (66, 1), (64, 0.5), (66, 0.5),
                     (69, 2)], "triangle", 0.32, 200, True),
        "deadeye": ([(57, 1), (64, 0.5), (69, 0.5), (68, 1), (64, 1), (62, 0.5), (60, 0.5), (59, 1), (57, 2)],
                    "sine", 0.34, 132, True),
    }
    for k, (notes, wave, vol, bpm, bass) in jingles.items():
        add("jingle_" + k, _lowpass(_melody(notes, wave, vol, bpm, decay=2.5, bass=bass), 3), 0.55)
    add("powerup_spawn", concat(*[tone(_f(n), 0.09, "sine", 0.25, decay=10) for n in (84, 88, 91, 96)]), 0.6)
    add("powerup_grab", mix(_chord([72, 76, 79, 84], 0.9, "triangle", 0.5, 0.01, 0.5, 1),
                            noise(0.3, 0.2, 10, 1, seed=80)), 0.8)
    add("powerup_end", concat(*[tone(_f(n), 0.12, "sine", 0.3, decay=8) for n in (84, 79, 76, 72)]), 0.6)
    stingers = {"max_ammo": [48, 55, 60, 64], "insta_kill": [45, 52, 57, 60], "double_points": [50, 57, 62, 66],
                "nuke": [36, 43, 48, 51], "carpenter": [53, 60, 65, 69], "fire_sale": [55, 59, 62, 67],
                "bonus_points": [60, 64, 67, 72]}
    for k, notes in stingers.items():
        add("announce_" + k, _chord(notes, 1.0, "saw", 0.5, 0.02, 0.6, 8), 0.7)
    round_start = mix(_chord([45, 52, 57, 60], 4.0, "saw", 0.45, 1.0, 2.0, 10),
                      concat(silence(0.8), _melody([(69, 1), (68, 1), (64, 1), (57, 3)], "sine", 0.3, 90, 1.5)))
    add("round_start", round_start, 0.8)
    round_end = mix(_chord([41, 48, 53, 57], 4.5, "saw", 0.45, 0.6, 2.5, 10),
                    concat(silence(0.5), _melody([(65, 1), (64, 1), (60, 1), (62, 1), (57, 3)], "sine", 0.3, 100,
                                                 1.5)))
    add("round_end", round_end, 0.8)
    add("game_over", mix(_chord([33, 40, 45, 48], 6.0, "saw", 0.5, 0.5, 3.0, 12),
                         _melody([(64, 1.5), (62, 1.5), (60, 1.5), (59, 1.5), (57, 4)], "sine", 0.3, 70, 1.0)), 0.9)
    add("down", concat(*[mix(tone(55, 0.12, "sine", 0.9, decay=20), silence(0.45)) for _ in range(4)]), 0.9)
    add("revive", _chord([60, 64, 67, 72], 1.2, "triangle", 0.5, 0.3, 0.6, 2), 0.8)
    # multiplayer
    add("streak_ready", concat(tone(1200, 0.08, "square", 0.25), silence(0.06), tone(1600, 0.12, "square", 0.25)),
        0.7)
    add("call_in", mix(noise(0.6, 0.25, 2, 1, seed=90), concat(silence(0.3), tone(900, 0.1, "square", 0.2),
                                                             silence(0.05), tone(1200, 0.2, "square", 0.2))), 0.7)
    add("uav", tone(400, 1.2, "sine", 0.3, slide_to=1400, attack=0.3, release=0.4), 0.6)
    add("missile", mix(_whoosh(1.5, 0.7, 91), tone(900, 1.5, "saw", 0.08, slide_to=300)), 0.8)
    add("rank_up", _melody([(72, 0.5), (76, 0.5), (79, 0.5), (84, 1.5)], "square", 0.25, 220, 2.5, True), 0.8)
    add("match_win", _melody([(60, 1), (64, 1), (67, 1), (72, 2), (67, 1), (72, 3)], "square", 0.22, 160, 2, True),
        0.8)
    add("match_lose", _melody([(60, 1), (58, 1), (55, 1), (53, 3)], "saw", 0.18, 120, 2, True), 0.8)
    add("enemy_down", tone(660, 0.1, "triangle", 0.25, decay=20), 0.5)
    add("spawn", _chord([52, 59, 64], 1.0, "saw", 0.4, 0.05, 0.6, 6), 0.6)
    add("footstep", noise(0.06, 0.3, 50, 6, seed=95), 0.35)
    # menus
    add("menu_move", tone(1800, 0.025, "square", 0.15, release=0.01), 0.5)
    add("menu_select", mix(tone(200, 0.12, "square", 0.3, decay=20), noise(0.08, 0.3, 40, 2, seed=96)), 0.6)
    add("menu_back", tone(600, 0.06, "triangle", 0.25, slide_to=300), 0.5)
    add("glitch", noise(0.15, 0.3, 5, 1, seed=97), 0.4)
    # loops
    t = np.arange(int(6.0 * RATE)) / RATE
    wind = _lowpass(np.random.default_rng(98).uniform(-1, 1, len(t)), 40) * 2.2
    wind *= 0.5 + 0.5 * np.sin(2 * np.pi * t / 6.0) ** 2
    drone = np.sin(2 * np.pi * 55 * t) * 0.12 + np.sin(2 * np.pi * 82.4 * t) * 0.06
    add("ambient_z", np.clip(wind * 0.35 + drone, -1, 1), 0.4, loop=True)
    pad = sum(np.sin(2 * np.pi * _f(n) * t) * 0.07 for n in (45, 52, 57, 60))
    arp = np.zeros_like(t)
    for k, n in enumerate((69, 72, 76, 72, 69, 72, 76, 79) * 3):
        s = int(k * 0.25 * RATE)
        seg = tone(_f(n), 0.25, "triangle", 0.12, decay=6)
        arp[s:s + len(seg)] += seg[:max(0, len(arp) - s)]
    add("menu_music", np.clip(pad * (0.7 + 0.3 * np.sin(np.pi * t / 6.0)) + arp, -1, 1), 0.45, loop=True)
