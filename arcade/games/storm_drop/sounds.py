"""Storm Drop's sounds, synthesised with numpy (no sound files): a shot per weapon class, the sledge on wood, brick
and metal, pieces going up and coming down, chests, pick-ups, healing, the storm's cues, the canopy, and the jingles."""

from __future__ import annotations

import numpy as np

from app.arcade.audio import SoundBank, concat, mix, noise, silence, tone


def _f(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def _low(x: np.ndarray, k: int) -> np.ndarray:
    return x if k <= 1 else np.convolve(x, np.ones(k) / k, mode="same")


def _shot(body: float, decay: float, smooth: int, boom: float, boom_to: float, crack: float = 0.0, vol: float = 0.9,
          seed: int = 3) -> np.ndarray:
    parts = [noise(body, vol, decay, smooth, seed=seed),
             tone(boom, min(body, 0.25), "sine", vol * 0.9, slide_to=boom_to, release=0.08, decay=decay * 0.6)]
    if crack:
        parts.append(noise(0.03, vol * 0.8, 60, 1, seed=seed + 1))
        parts.append(tone(crack, 0.02, "square", 0.22, release=0.01))
    tail = noise(body * 2.0, vol * 0.16, decay * 0.35, smooth * 3 + 4, seed=seed + 2)
    return mix(*parts, concat(silence(0.03), tail))


def _notes(seq: list[tuple[int, float]], wave: str = "triangle", vol: float = 0.4, decay: float = 5.0) -> np.ndarray:
    out = []
    for n, d in seq:
        out.append(silence(d) if n <= 0 else mix(tone(_f(n), d, wave, vol, release=min(0.06, d * 0.4), decay=decay),
                                                 tone(_f(n + 12), d, "sine", vol * 0.3, release=0.04,
                                                      decay=decay * 1.5)))
    return concat(*out)


def _knock(freq: float, dur: float, ring: float, grit: float, seed: int) -> np.ndarray:
    return mix(tone(freq, dur, "sine", 0.7, slide_to=freq * 0.6, release=dur * 0.5, decay=ring),
               tone(freq * 2.7, dur * 0.6, "triangle", 0.25, release=dur * 0.3, decay=ring * 1.6),
               noise(dur * 0.5, grit, 40, 2, seed=seed))


def build(bank: SoundBank) -> None:
    add = bank.add
    # weapons
    add("shot_light", _shot(0.16, 26, 2, 300, 110, crack=2100, vol=0.75, seed=11), 0.55)
    add("shot_smg", _shot(0.11, 34, 1, 340, 140, crack=2500, vol=0.65, seed=12), 0.45)
    add("shot_shell", _shot(0.34, 12, 3, 150, 48, vol=1.0, seed=13), 0.75)
    add("shot_mid", _shot(0.2, 20, 2, 220, 80, crack=1700, vol=0.85, seed=14), 0.6)
    add("shot_heavy", _shot(0.28, 14, 2, 190, 60, crack=1300, vol=0.95, seed=15), 0.7)
    add("shot_sniper", mix(_shot(0.5, 8, 3, 150, 40, crack=900, vol=1.0, seed=16),
                           concat(silence(0.12), noise(0.9, 0.2, 4, 12, seed=17))), 0.85)
    add("shot_rocket", mix(noise(0.5, 0.7, 6, 5, seed=18), tone(160, 0.5, "saw", 0.35, slide_to=520, release=0.2)),
        0.7)
    add("shot_lob", mix(tone(220, 0.12, "sine", 0.8, slide_to=90, decay=14), noise(0.1, 0.4, 30, 3, seed=19)), 0.7)
    add("boom", mix(noise(0.9, 1.0, 4.5, 6, seed=20), tone(90, 0.7, "sine", 1.0, slide_to=30, decay=5),
                    noise(0.1, 0.9, 30, 1, seed=21)), 0.95)
    add("empty", mix(noise(0.03, 0.4, 90, 1, seed=22), tone(1800, 0.02, "square", 0.15, release=0.01)), 0.5)
    add("reload", concat(_knock(900, 0.05, 40, 0.3, 23), silence(0.14), _knock(600, 0.06, 30, 0.35, 24),
                         silence(0.1), _knock(1300, 0.04, 50, 0.25, 25)), 0.5)
    add("switch", _knock(1100, 0.05, 45, 0.25, 26), 0.4)
    add("swing", _low(noise(0.2, 0.6, 9, 6, seed=27) * np.linspace(0.2, 1.0, len(noise(0.2))) , 3), 0.5)
    add("hit", mix(tone(1250, 0.06, "square", 0.3, release=0.03), noise(0.03, 0.3, 80, 1, seed=28)), 0.5)
    add("headshot", concat(tone(1560, 0.05, "square", 0.3, release=0.02), tone(2340, 0.11, "triangle", 0.5,
                                                                              release=0.07, decay=10)), 0.6)
    add("hurt", mix(tone(190, 0.16, "saw", 0.5, slide_to=110, release=0.08), noise(0.08, 0.35, 30, 3, seed=29)), 0.6)
    add("elim", _notes([(76, 0.07), (81, 0.07), (88, 0.2)], "square", 0.3, 6), 0.6)
    add("elim_far", mix(tone(520, 0.1, "triangle", 0.3, slide_to=260, release=0.06), noise(0.06, 0.15, 40, 4)), 0.4)
    # harvesting, building, breaking: wood, brick, metal
    for name, f0, ring, grit, seed in (("wood", 330, 30, 0.45, 31), ("brick", 210, 38, 0.6, 32),
                                       ("metal", 620, 9, 0.3, 33)):
        add(f"chop_{name}", _knock(f0, 0.14, ring, grit, seed), 0.65)
        add(f"thud_{name}", _knock(f0 * 1.3, 0.07, ring * 1.5, grit * 0.7, seed + 10), 0.4)
        add(f"build_{name}", concat(_knock(f0 * 1.2, 0.06, ring, grit * 0.6, seed + 20),
                                    _knock(f0 * 1.6, 0.09, ring, grit * 0.5, seed + 21)), 0.6)
        add(f"break_{name}", mix(noise(0.45, 0.8, 7, 3 if name != "metal" else 1, seed=seed + 30),
                                 tone(f0 * 0.6, 0.3, "saw" if name == "metal" else "sine", 0.4,
                                      slide_to=f0 * 0.25, decay=8)), 0.7)
    # loot
    add("chest", _notes([(72, 0.08), (76, 0.08), (79, 0.08), (84, 0.3)], "triangle", 0.45, 3.5), 0.6)
    add("ammo_box", concat(_knock(700, 0.05, 30, 0.3, 41), _knock(1050, 0.08, 20, 0.2, 42)), 0.5)
    add("pickup", concat(tone(880, 0.045, "triangle", 0.4, release=0.02), tone(1320, 0.07, "triangle", 0.4,
                                                                             release=0.04)), 0.45)
    add("pickup_weapon", concat(_knock(800, 0.05, 30, 0.3, 43), tone(1100, 0.05, "triangle", 0.4, release=0.02),
                                tone(1650, 0.1, "triangle", 0.4, release=0.06)), 0.5)
    add("use_start", tone(300, 0.2, "sine", 0.3, slide_to=520, attack=0.05, release=0.08), 0.4)
    add("heal", _notes([(69, 0.07), (73, 0.07), (76, 0.16)], "sine", 0.5, 4), 0.55)
    add("shield", mix(tone(520, 0.3, "sine", 0.4, slide_to=1240, attack=0.02, release=0.12),
                      tone(1040, 0.3, "triangle", 0.15, slide_to=2480, release=0.12)), 0.5)
    # moving
    add("jump", tone(260, 0.09, "sine", 0.3, slide_to=420, release=0.05), 0.3)
    add("land", mix(noise(0.14, 0.6, 22, 6, seed=44), tone(110, 0.12, "sine", 0.6, slide_to=60, decay=14)), 0.5)
    add("jump_out", mix(noise(0.5, 0.5, 5, 8, seed=45), tone(420, 0.3, "sine", 0.3, slide_to=180, release=0.15)),
        0.5)
    add("canopy", mix(noise(0.3, 0.7, 10, 5, seed=46), tone(180, 0.2, "sine", 0.5, slide_to=90, decay=10)), 0.55)
    wind = noise(2.0, 0.5, 0.0, 14, seed=47)
    wind = wind * (0.7 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, len(wind))))
    add("wind", wind, 0.35, loop=True)
    # the storm
    add("storm_mark", _notes([(64, 0.14), (71, 0.14), (76, 0.3)], "sine", 0.45, 3), 0.55)
    add("storm_move", mix(tone(98, 1.1, "saw", 0.35, slide_to=74, attack=0.1, release=0.5),
                          _low(noise(1.1, 0.5, 2.0, 10, seed=48), 4)), 0.6)
    add("storm_tick", mix(tone(140, 0.09, "square", 0.3, release=0.05), noise(0.05, 0.3, 50, 3, seed=49)), 0.4)
    # jingles (own melodies)
    add("win", mix(_notes([(72, 0.13), (76, 0.13), (79, 0.13), (84, 0.2), (0, 0.06), (79, 0.1), (84, 0.1),
                           (88, 0.5)], "square", 0.3, 2.5),
                   _notes([(48, 0.39), (55, 0.26), (60, 0.7)], "triangle", 0.4, 2.0)), 0.7)
    add("lose", _notes([(67, 0.18), (63, 0.18), (60, 0.2), (55, 0.45)], "triangle", 0.45, 3.0), 0.6)
    add("unlock", _notes([(79, 0.07), (83, 0.07), (86, 0.07), (91, 0.3)], "triangle", 0.4, 4), 0.55)
    add("menu_move", tone(660, 0.035, "triangle", 0.35, release=0.02), 0.4)
    add("menu_select", concat(tone(660, 0.05, "triangle", 0.4, release=0.02), tone(990, 0.09, "triangle", 0.4,
                                                                                  release=0.05)), 0.45)
    add("menu_back", tone(440, 0.07, "triangle", 0.35, slide_to=330, release=0.04), 0.4)
    add("deny", tone(160, 0.1, "square", 0.25, release=0.05), 0.4)
