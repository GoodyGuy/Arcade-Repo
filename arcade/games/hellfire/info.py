"""Data tables: thing types (size, health, speed, pain chance, sounds, state sequences), weapons and skills, with the
1993 game's numbers (35 tics per second, map units)."""

from __future__ import annotations

TICRATE = 35
MELEERANGE = 64.0
MISSILERANGE = 2048.0
USERANGE = 64.0
VIEWHEIGHT = 41.0
PLAYER_RADIUS = 16.0
PLAYER_HEIGHT = 56.0
MAXSTEP = 24.0

# state: (frame, tics, action, bright); frame = (set, index) into the sprite frames, or an int for 2D art
# a sequence ends by jumping to ``next`` (None: stay on the last state)


def seq(*states, next_=None) -> dict:  # noqa: ANN002
    return {"states": list(states), "next": next_}


def _monster_states(walk_tics: int, nwalk: int, attack: list, pain_tics: tuple, death: list, gib: list,
                    melee: list | None = None) -> dict:
    walk = [(("walk", (i // 2) % nwalk), walk_tics, "chase", False) for i in range(8)]
    st = {
        "spawn": seq((("walk", 0), 10, "look", False), (("walk", 1 % nwalk), 10, "look", False), next_="spawn"),
        "see": seq(*walk, next_="see"),
        "pain": seq((("pain", 0), pain_tics[0], None, False), (("pain", 0), pain_tics[1], "pain", False),
                    next_="see"),
        "death": seq(*death),
        "xdeath": seq(*gib),
    }
    if attack:
        st["missile"] = seq(*attack, next_="see")
    if melee is not None:
        st["melee"] = seq(*melee, next_="see")
    return st


def _deaths(n: int, tics: int, scream_at: int = 1, fall_at: int = 2) -> list:
    out = []
    for i in range(n):
        act = "scream" if i == scream_at else "fall" if i == fall_at else None
        out.append((("death", i), tics if i < n - 1 else -1, act, False))
    return out


def _gibs(n: int = 7) -> list:
    out = []
    for i in range(n):
        act = "xscream" if i == 1 else "fall" if i == 2 else None
        out.append((("gib", i), 5 if i < n - 1 else -1, act, False))
    return out


MONSTER_INFO = {
    "zombie": dict(
        radius=20, height=56, hp=20, speed=8, painchance=200, mass=100, reaction=8, drop="clip_drop",
        sounds=dict(sight="posit", active="posact", pain="popain", death="podth", attack=None),
        states=_monster_states(4, 4, [(("attack", 0), 10, "face", False), (("attack", 1), 8, "posattack", True),
                                      (("attack", 2), 8, None, False)], (3, 3), _deaths(5, 5), _gibs())),
    "sergeant": dict(
        radius=20, height=56, hp=30, speed=8, painchance=170, mass=100, reaction=8, drop="w_shotgun_drop",
        sounds=dict(sight="posit", active="posact", pain="popain", death="podth", attack=None),
        states=_monster_states(3, 4, [(("attack", 0), 10, "face", False), (("attack", 1), 10, "sposattack", True),
                                      (("attack", 2), 10, None, False)], (3, 3), _deaths(5, 5), _gibs())),
    "imp": dict(
        radius=20, height=56, hp=60, speed=8, painchance=200, mass=100, reaction=8,
        sounds=dict(sight="bgsit", active="bgact", pain="popain", death="bgdth", attack="claw"),
        states=_monster_states(3, 4, [(("attack", 0), 8, "face", True), (("attack", 1), 8, "face", True),
                                      (("attack", 2), 6, "troopattack", False)], (2, 2), _deaths(5, 8, 1, 3),
                               _gibs(),
                               melee=[(("attack", 0), 8, "face", False), (("attack", 1), 8, "face", False),
                                      (("attack", 2), 6, "troopattack", False)])),
    "demon": dict(
        radius=30, height=56, hp=150, speed=10, painchance=180, mass=400, reaction=8,
        sounds=dict(sight="sgtsit", active="dmact", pain="dmpain", death="sgtdth", attack="sgtatk"),
        states=_monster_states(2, 4, [], (2, 2), _deaths(6, 8, 1, 3), _gibs(),
                               melee=[(("attack", 0), 8, "face", False), (("attack", 1), 8, "face", False),
                                      (("attack", 2), 8, "sargattack", False)])),
    "soul": dict(
        radius=16, height=56, hp=100, speed=8, painchance=256, mass=50, reaction=8, float=True, nogravity=True,
        bright=True, countkill=True,
        sounds=dict(sight=None, active="dmact", pain="dmpain", death="firxpl", attack="sklatk"),
        states={
            "spawn": seq((("walk", 0), 10, "look", True), (("walk", 1), 10, "look", True), next_="spawn"),
            "see": seq((("walk", 0), 6, "chase", True), (("walk", 1), 6, "chase", True), next_="see"),
            "missile": seq((("attack", 0), 10, "face", True), (("attack", 1), 4, "skullattack", True),
                           (("attack", 0), 4, None, True), (("attack", 1), 4, None, True), next_="charge"),
            "charge": seq((("attack", 0), 4, None, True), (("attack", 1), 4, None, True), next_="charge"),
            "pain": seq((("pain", 0), 3, None, True), (("pain", 0), 3, "pain", True), next_="see"),
            "death": seq(*[(("death", i), 6, "scream" if i == 1 else "fall" if i == 3 else None, True)
                           for i in range(6)], next_="remove"),
        }),
    "caco": dict(
        radius=31, height=56, hp=400, speed=8, painchance=128, mass=400, reaction=8, float=True, nogravity=True,
        sounds=dict(sight="cacsit", active="dmact", pain="dmpain", death="cacdth", attack=None),
        states={
            "spawn": seq((("walk", 0), 10, "look", False), next_="spawn"),
            "see": seq((("walk", 0), 3, "chase", False), (("walk", 1), 3, "chase", False), next_="see"),
            "missile": seq((("attack", 0), 5, "face", False), (("attack", 1), 5, "face", False),
                           (("attack", 2), 5, "headattack", True), next_="see"),
            "pain": seq((("pain", 0), 3, None, False), (("pain", 0), 3, "pain", False), (("pain", 0), 6, None, False),
                        next_="see"),
            "death": seq(*[(("death", i), 8 if i < 5 else -1, "scream" if i == 1 else "fall" if i == 4 else None,
                            False) for i in range(6)]),
        }),
    "baron": dict(
        radius=24, height=64, hp=1000, speed=8, painchance=50, mass=1000, reaction=8, boss=True,
        sounds=dict(sight="brssit", active="dmact", pain="dmpain", death="brsdth", attack="claw"),
        states=_monster_states(3, 4, [(("attack", 0), 8, "face", True), (("attack", 1), 8, "face", True),
                                      (("attack", 2), 8, "bruisattack", False)], (2, 2), _deaths(7, 8, 1, 3),
                               _gibs(),
                               melee=[(("attack", 0), 8, "face", False), (("attack", 1), 8, "face", False),
                                      (("attack", 2), 8, "bruisattack", False)])),
    # the gatling ghoul: a hitscan rotary gun it keeps firing while it can see you
    "heavy": dict(
        radius=20, height=56, hp=70, speed=8, painchance=170, mass=100, reaction=8, drop="w_chaingun_drop",
        sounds=dict(sight="posit", active="posact", pain="popain", death="podth", attack=None),
        states=_monster_states(3, 4, [(("attack", 0), 10, "face", False), (("attack", 1), 4, "cposattack", True),
                                      (("attack", 0), 4, "cposattack", False),
                                      (("attack", 1), 1, "cposrefire", True)], (3, 3), _deaths(7, 5),
                               _gibs(6))),
    # the shade: a maw hound you can barely see
    "shade": dict(
        radius=30, height=56, hp=150, speed=10, painchance=180, mass=400, reaction=8, shadow=True,
        sounds=dict(sight="sgtsit", active="dmact", pain="dmpain", death="sgtdth", attack="sgtatk"),
        states=_monster_states(2, 4, [], (2, 2), _deaths(6, 8, 1, 3), _gibs(),
                               melee=[(("attack", 0), 8, "face", False), (("attack", 1), 8, "face", False),
                                      (("attack", 2), 8, "sargattack", False)])),
    # the siege titan: the gate's last guardian, rocket volleys, shrugs off blast damage
    "titan": dict(
        radius=40, height=110, hp=4000, speed=16, painchance=20, mass=1000, reaction=8, boss=True, noradius=True,
        sounds=dict(sight="cybsit", active="dmact", pain="dmpain", death="cybdth", attack=None),
        states=_monster_states(3, 4, [(("attack", 0), 6, "face", False), (("attack", 1), 12, "cyberattack", True),
                                      (("attack", 0), 12, "face", False), (("attack", 1), 12, "cyberattack", True),
                                      (("attack", 0), 12, "face", False), (("attack", 1), 12, "cyberattack", True)],
                               (5, 5), _deaths(9, 10, 1, 7), _gibs(1))),
}
MONSTER_INFO["titan"]["states"].pop("xdeath", None)
MONSTER_INFO["demon"]["states"].pop("xdeath", None)
MONSTER_INFO["shade"]["states"].pop("xdeath", None)
for _k, _n in (("zombie", 9), ("sergeant", 9), ("imp", 8)):
    MONSTER_INFO[_k]["states"]["xdeath"] = seq(*_gibs(_n))

# the names the game shows (the tally, the help page)
MONSTER_NAMES = {
    "zombie": "Ghoul trooper", "sergeant": "Shotgun ghoul", "heavy": "Gatling ghoul", "imp": "Cinder fiend",
    "demon": "Maw hound", "shade": "Shade", "soul": "Wisp skull", "caco": "Bloat eye", "baron": "Gate warden",
    "titan": "Siege titan",
}
for _k, _v in MONSTER_INFO.items():
    _v.setdefault("countkill", True)
    _v["kind"] = _k

# missiles: speed (units / tic), damage multiplier (1d8 x), radius, height, sprite frames, explosion frames
MISSILE_INFO = {
    "imp_ball": dict(speed=10, damage=3, radius=6, height=8, fly="imp_ball", boom="imp_x", sound="firsht",
                     hit="firxpl"),
    "caco_ball": dict(speed=10, damage=5, radius=6, height=8, fly="caco_ball", boom="caco_x", sound="firsht",
                      hit="firxpl"),
    "baron_ball": dict(speed=15, damage=8, radius=6, height=16, fly="baron_ball", boom="baron_x", sound="firsht",
                       hit="firxpl"),
    "rocket": dict(speed=20, damage=20, radius=11, height=8, fly="rocket_fly", boom="rocket_x", sound="rlaunc",
                   hit="barexp", splash=128),
    "plasma": dict(speed=25, damage=5, radius=13, height=8, fly="plasma_ball", boom="plasma_x", sound="plasma",
                   hit="firxpl"),
    "bfg": dict(speed=25, damage=100, radius=13, height=8, fly="bfg_ball", boom="bfg_x", sound="bfg",
                hit="rxplod", bfg=True),
}

# pickups: kind -> what it gives; count: counts toward the items tally
ITEM_INFO = {
    "hbonus": dict(count=True), "abonus": dict(count=True), "stim": {}, "medikit": {}, "garmor": {},
    "barmor": {}, "soulsphere": dict(count=True, bright=True), "clip": {}, "clip_drop": {}, "ammobox": {},
    "shells": {}, "shellbox": {}, "rocket": {}, "rocketbox": {}, "cell": {}, "cellpack": {}, "backpack": {},
    "bluecard": dict(key="blue"), "yellowcard": dict(key="yellow"), "redcard": dict(key="red"),
    "w_shotgun": {}, "w_shotgun_drop": {}, "w_chaingun": {}, "w_chaingun_drop": {}, "w_launcher": {},
    "w_plasma": {}, "w_bfg": {}, "w_saw": {},
    "berserk": dict(count=True), "invuln": dict(count=True, bright=True), "suit": {},
    "scanner": dict(count=True, bright=True),
}

DECOR_INFO = {
    "techlamp": dict(solid=True, radius=16, bright=True), "lamp": dict(solid=True, radius=16, bright=True),
    "torch": dict(solid=True, radius=16, bright=True), "torch_g": dict(solid=True, radius=16, bright=True),
    "torch_b": dict(solid=True, radius=16, bright=True), "column": dict(solid=True, radius=16, bright=True),
    "candle": dict(bright=True), "candelabra": dict(solid=True, radius=16, bright=True),
    "pylon": dict(solid=True, radius=16, bright=True), "pillar": dict(solid=True, radius=16),
    "pillar_r": dict(solid=True, radius=16), "firecan": dict(solid=True, radius=16, bright=True),
    "tree": dict(solid=True, radius=16), "deadguy": {}, "bloodpool": {}, "gibs": {},
}
POWER_TICS = {"invuln": 30 * 35, "suit": 60 * 35}

AMMO_MAX = {"bullets": 200, "shells": 50, "rockets": 50, "cells": 300}
AMMO_NAMES = ("bullets", "shells", "rockets", "cells")

# weapon: slot, ammo, per shot, states: (frame, tics, action, muzzle flash) - the action runs on entering the
# state; frame and flash index the weapon's pictures (assets.WEAPON_FD), flash -1 for none
WEAPONS = {
    "fist": dict(slot=1, ammo=None, use=0, fire=[(1, 4, None, -1), (2, 4, "punch", -1), (3, 5, None, -1),
                                                    (2, 4, None, -1), (1, 5, "refire", -1)]),
    "saw": dict(slot=1, ammo=None, use=0, fire=[(2, 4, "saw", -1), (3, 4, "saw", -1), (3, 0, "refire", -1)]),
    "pistol": dict(slot=2, ammo="bullets", use=1, fire=[(0, 4, None, -1), (1, 6, "pistol", 0), (2, 4, None, -1),
                                                         (1, 5, "refire", -1)]),
    "shotgun": dict(slot=3, ammo="shells", use=1, fire=[(0, 3, None, -1), (0, 4, "shotgun", 0), (0, 3, None, 1),
                                                         (1, 5, None, -1), (2, 5, "pump", -1), (3, 4, None, -1),
                                                         (2, 5, None, -1), (1, 5, None, -1), (0, 3, None, -1),
                                                         (0, 7, "refire", -1)]),
    "chaingun": dict(slot=4, ammo="bullets", use=1, fire=[(0, 4, "cgun", 0), (1, 4, "cgun", 1),
                                                           (1, 0, "refire", -1)]),
    "launcher": dict(slot=5, ammo="rockets", use=1, fire=[(1, 4, "flash", 0), (1, 4, None, 1), (1, 6, "rocket", 2),
                                                           (1, 6, None, 3), (1, 0, "refire", -1)]),
    "plasma": dict(slot=6, ammo="cells", use=1, fire=[(0, 3, "plasma", 0), (1, 20, "refire", -1)]),
    "bfg": dict(slot=7, ammo="cells", use=40, fire=[(0, 20, "bfgsound", -1), (1, 10, None, 0), (1, 10, "bfg", 1),
                                                     (2, 20, "refire", -1)]),
}
WEAPON_ORDER = ("fist", "saw", "pistol", "shotgun", "chaingun", "launcher", "plasma", "bfg")
# when the ammo runs out: the best weapon still loaded
AUTO_ORDER = ("plasma", "chaingun", "shotgun", "pistol", "launcher", "bfg", "saw", "fist")
WEAPON_NAMES = {"fist": "Fists", "saw": "Ripsaw", "pistol": "Sidearm", "shotgun": "Pump shotgun",
                "chaingun": "Rotary cannon", "launcher": "Rocket launcher", "plasma": "Plasma rifle",
                "bfg": "Hellfire cannon"}

SKILLS = (
    "Can I Keep My Teddy?",
    "Go Easy, Sarge",
    "Bring The Pain",
    "Maximum Mayhem",
    "Absolute Hellfire!",
)
SKILL_SHORT = ("Teddy", "Go easy", "Bring the pain", "Max mayhem", "Hellfire")
