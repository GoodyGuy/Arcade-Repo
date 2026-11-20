"""Every number of BREACH POINT in one place: the match and the economy, movement, the guns, armour, the five agents and
their abilities, and the bots' skill levels. Nothing here draws or simulates."""

from __future__ import annotations

from dataclasses import dataclass, field

# --- the match -------------------------------------------------------------------------------------------------------

BUY_TIME = 20.0  # s before each round (Enter: ready, cuts it to 3 s)
ROUND_TIME = 90.0  # s to plant
PLANT_TIME = 4.0
FUSE_TIME = 40.0  # s from plant to detonation
DEFUSE_TIME = 7.0
DEFUSE_HALF = 3.5  # progress kept once reached
END_TIME = 5.0  # s between the end of a round and the next buy phase
BLAST_RADIUS = 11.0  # m: the charge kills inside this
WIN_ROUNDS = 7  # default: first to 7, sides swap after 6 rounds
MATCH_LENGTHS = (4, 7, 10)  # rounds to win offered in the menu

CREDITS_START = 800
CREDITS_CAP = 9000
CREDITS_DECIDER = 5000  # everyone, in a last round at match point each
CREDIT_KILL = 200
CREDIT_PLANT = 300
CREDIT_DEFUSE = 300
CREDIT_WIN = 3000
CREDIT_LOSS = (1900, 2400, 2900)  # by losses in a row
CREDIT_SURVIVED_LOSS = 1000  # alive at the end of a lost round: no loss bonus

ULT_KILL = 1
ULT_OBJECTIVE = 1
ULT_ROUND = 1  # every agent, each round

# --- movement (metres, seconds) --------------------------------------------------------------------------------------

RUN = 6.4  # m/s with the knife out; a gun multiplies it
WALK = 0.55  # Shift: silent
CROUCH = 0.45
ACCEL = 14.0  # 1/s towards the wanted speed
BRAKE = 11.0
AIR_CONTROL = 1.6
GRAVITY = 19.0
JUMP = 5.7  # m/s up: about 0.85 m
STEP = 0.36  # m a walker climbs without jumping
RADIUS = 0.34
HEIGHT = 1.8
EYE = 1.62
CROUCH_DROP = 0.5  # the eye and the head come down by this
STILL = 0.275  # of the gun's top speed: below it the shot is as steady as standing
TAG_SLOW = 0.45  # speed factor right after being hit
TAG_TIME = 0.35
HP = 100

# --- the world -------------------------------------------------------------------------------------------------------

WALL = 5.0  # m: a full wall
ROOF = 4.2  # m: the ceiling of a roofed cell
PEN_POWER = {"low": 0.75, "med": 1.45, "high": 2.5}
PEN_LOSS = 0.55  # of the damage, at the gun's whole penetration used up


@dataclass(frozen=True)
class Weapon:
    id: str
    name: str
    cls: str  # sidearm / smg / shotgun / rifle / marksman / sniper / lmg / melee
    group: str  # buy menu column: side / light / heavy
    price: int
    dmg: float  # body
    head: float  # multiplier
    leg: float
    falloff: tuple  # ((from metres, multiplier), ...)
    rps: float
    mag: int
    reload: float
    equip: float
    speed: float  # of RUN
    pen: str
    auto: bool
    spread: float  # degrees, standing still, first shot
    bloom: float  # degrees added per shot
    bloom_max: float
    recover: float  # s for bloom and recoil to settle
    rv: float  # degrees the muzzle climbs per shot
    rh: float  # degrees it wanders sideways once the climb tops out
    rmax: float  # degrees: the top of the climb
    move: float = 4.0  # degrees added at a full run
    pellets: int = 1
    cone: float = 0.0  # degrees: pellet cone
    zoom: float = 1.0  # right mouse button
    scoped: bool = False  # a real scope: accurate only through it
    quiet: bool = False
    blurb: str = ""


def _w(*a, **k) -> Weapon:  # noqa: ANN002, ANN003
    return Weapon(*a, **k)


WEAPONS: dict[str, Weapon] = {w.id: w for w in (
    _w("knife", "TOOL KNIFE", "melee", "", 0, 50, 1.0, 1.0, (), 1.6, 0, 0, 0.4, 1.0, "low", True, 0, 0, 0, 0.2, 0, 0, 0,
       blurb="Fastest on your feet. 50 to the front, 150 to the back."),
    _w("tack", "TACK-9", "sidearm", "side", 0, 26, 3.0, 0.85, ((30, 0.85),), 6.5, 12, 1.7, 0.55, 0.9, "low", False,
       0.32, 0.55, 3.0, 0.34, 1.0, 0.3, 4.0, move=2.6, blurb="The issue pistol. Free every round."),
    _w("vesper", "VESPER", "sidearm", "side", 500, 30, 3.5, 0.85, ((30, 0.84),), 6.2, 13, 1.5, 0.55, 0.9, "med",
       False, 0.22, 0.5, 2.6, 0.32, 0.9, 0.25, 3.6, move=2.4, quiet=True,
       blurb="Suppressed and straight. One clean headshot ends an unarmoured target."),
    _w("maul", "MAUL .50", "sidearm", "side", 800, 55, 2.9, 0.85, ((30, 0.9),), 3.6, 6, 2.2, 0.75, 0.88, "high",
       False, 0.28, 1.7, 4.2, 0.5, 3.2, 0.5, 7.0, move=3.4, blurb="Six heavy rounds. Punches through cover."),
    _w("wasp", "WASP", "smg", "light", 1100, 27, 2.5, 0.85, ((15, 0.85),), 15.0, 20, 2.1, 0.6, 0.9, "low", True,
       0.55, 0.2, 3.2, 0.3, 0.52, 0.55, 5.0, move=1.8, zoom=1.15, blurb="Empties fast. Best inside ten metres."),
    _w("lynx", "LYNX", "smg", "light", 1600, 26, 3.0, 0.85, ((15, 0.85), (30, 0.77)), 13.0, 30, 2.2, 0.6, 0.9, "low",
       True, 0.42, 0.17, 2.6, 0.3, 0.46, 0.42, 4.4, move=1.6, zoom=1.15, quiet=True,
       blurb="Suppressed, steady on the move. The eco round's rifle."),
    _w("brace", "BRACE-12", "shotgun", "light", 900, 17, 2.0, 0.85, ((8, 0.75), (12, 0.5)), 1.15, 5, 2.5, 0.8, 0.86,
       "low", False, 0.0, 0.0, 0.0, 0.5, 4.5, 0.0, 6.0, move=0.6, pellets=12, cone=2.6,
       blurb="Twelve pellets a pull. Owns a doorway, nothing else."),
    _w("lancer", "LANCER", "sniper", "light", 950, 101, 2.0, 0.85, (), 1.4, 5, 2.5, 1.0, 0.82, "med", False, 3.2, 0.0,
       0.0, 0.6, 2.6, 0.0, 4.0, move=5.0, zoom=3.0, scoped=True,
       blurb="A light bolt rifle. Body, body. Or one to the head."),
    _w("sentry", "SENTRY", "marksman", "heavy", 2250, 65, 3.0, 0.75, (), 5.0, 12, 2.5, 0.85, 0.82, "high", False, 0.07,
       0.95, 2.6, 0.45, 1.9, 0.3, 5.0, move=4.6, zoom=1.5, blurb="Semi-automatic and exact. Two to the chest."),
    _w("kestrel", "KESTREL", "rifle", "heavy", 2900, 40, 4.0, 0.85, (), 9.6, 25, 2.5, 0.85, 0.82, "med", True, 0.2,
       0.28, 3.4, 0.38, 0.95, 0.75, 6.4, move=5.0, zoom=1.25,
       blurb="No falloff: one to the head at any range. Kicks hard past four rounds."),
    _w("mistral", "MISTRAL", "rifle", "heavy", 2900, 39, 4.0, 0.85, ((20, 0.9),), 11.0, 30, 2.5, 0.85, 0.82, "med",
       True, 0.17, 0.21, 2.8, 0.36, 0.7, 0.52, 5.2, move=4.8, zoom=1.25, quiet=True,
       blurb="Suppressed, faster, easier to hold down. Loses bite past twenty metres."),
    _w("bastion", "BASTION", "lmg", "heavy", 3200, 38, 2.5, 0.85, ((30, 0.82),), 12.0, 80, 4.6, 1.2, 0.78, "high", True,
       0.5, 0.11, 2.6, 0.5, 0.42, 0.7, 4.6, move=5.4, zoom=1.15,
       blurb="Eighty rounds through anything thin. Slow to carry, slow to load."),
    _w("longbow", "LONGBOW", "sniper", "heavy", 4700, 150, 1.7, 0.8, (), 0.65, 5, 3.6, 1.3, 0.78, "high", False, 4.5,
       0.0, 0.0, 0.8, 3.4, 0.0, 5.0, move=6.0, zoom=2.8, scoped=True,
       blurb="Ends any target above the knee. One shot, then a long wait."),
)}
BUY_COLUMNS = (("SIDEARMS", ("tack", "vesper", "maul")), ("LIGHT", ("wasp", "lynx", "brace", "lancer")),
               ("HEAVY", ("sentry", "kestrel", "mistral", "bastion", "longbow")))
ARMOUR = {"light": (400, 25, "LIGHT PLATE"), "heavy": (1000, 50, "HEAVY PLATE")}
ARMOUR_ABSORB = 0.66
MAG_RESERVE = 3  # magazines in reserve
SCOPED_SPEED = 0.8


# --- agents ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Ability:
    id: str
    name: str
    slot: str  # C / Q bought, E free on a recharge, X the ultimate
    kind: str
    desc: str
    cost: int = 0  # credits a charge (C / Q)
    charges: int = 1
    recharge: float = 0.0  # s (E)
    points: int = 0  # ultimate points (X)
    p: dict = field(default_factory=dict)  # the effect's own numbers


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    role: str
    color: tuple
    line: str
    about: str
    abilities: tuple  # C, Q, E, X

    def ability(self, slot: str) -> Ability:
        return self.abilities["CQEX".index(slot)]


def _ab(*a, **k) -> Ability:  # noqa: ANN002, ANN003
    return Ability(*a, **k)


ROLES = {"Entry": "First through the door. Makes the space the team plants in.",
         "Control": "Cuts the map with smoke and bad ground so a fight happens on your terms.",
         "Recon": "Finds them first. Information the whole team sees.",
         "Anchor": "Holds a site alone long enough for help to arrive."}

AGENTS: dict[str, Agent] = {a.id: a for a in (
    Agent("kite", "KITE", "Entry", (255, 150, 56), "Gone before the echo.",
          "A courier who never learned to stop. Lunges into a site behind her own flare and wins the first fight "
          "on speed.",
          (_ab("flare", "FLARE", "C", "flash", "Throw a flare that bursts after a beat and blinds anyone looking at "
               "it.", cost=200, charges=2, p={"fuse": 0.75, "blind": 2.2}),
           _ab("overclock", "OVERCLOCK", "Q", "buff", "Four seconds of faster feet and a steadier gun on the move.",
               cost=200, p={"dur": 4.0, "speed": 1.22, "steady": 0.35}),
           _ab("lunge", "LUNGE", "E", "dash", "A short burst the way you are moving. Returns on its own.",
               recharge=24.0, p={"dist": 6.5, "time": 0.2}),
           _ab("blitz", "BLITZ", "X", "buff", "Eight seconds faster in every way: feet, trigger and reload. Each kill "
               "adds two more.", points=6, p={"dur": 8.0, "speed": 1.2, "steady": 0.4, "rate": 1.2, "reload": 0.6,
                                               "extend": 2.0}))),
    Agent("fuse", "FUSE", "Entry", (255, 214, 64), "Knock first. Loudly.",
          "A demolition tech who opens rooms the direct way: a stun through the door, a frag behind it, and a line "
          "of charges for whoever is left.",
          (_ab("frag", "FRAG", "C", "frag", "A grenade on a short fuse. Bounces, then clears a corner.", cost=300,
               p={"fuse": 1.5, "radius": 5.5, "dmg": 110}),
           _ab("adrenal", "ADRENAL", "Q", "buff", "Patch yourself up: 45 health over four seconds and a little pace.",
               cost=200, p={"dur": 4.0, "speed": 1.08, "heal": 45}),
           _ab("stinger", "STINGER", "E", "concuss", "A puck that bursts where it lands and leaves everyone near it "
               "dazed.", recharge=32.0, p={"radius": 4.6, "daze": 3.0}),
           _ab("chain", "CHAIN REACTION", "X", "chain", "Five blasts march forward from you, one after another.",
               points=7, p={"count": 5, "step": 3.6, "radius": 4.0, "dmg": 95, "delay": 0.8, "gap": 0.32}))),
    Agent("murk", "MURK", "Control", (156, 110, 255), "You can't hit what the fog keeps.",
          "A harbour pilot who reads weather nobody else sees. Closes a sightline with a shroud, drags a push "
          "through mire and turns the lights down on a whole team.",
          (_ab("mire", "MIRE", "C", "slow", "Throw a patch of clinging ground: everyone in it moves at half speed.",
               cost=200, charges=2, p={"radius": 5.0, "dur": 7.0, "slow": 0.5}),
           _ab("veil", "VEIL", "Q", "curtain", "Raise a long curtain of fog straight ahead. It hides, it does not "
               "stop.", cost=300, p={"length": 16.0, "dur": 10.0, "r": 1.7}),
           _ab("shroud", "SHROUD", "E", "smoke", "Throw an orb that blooms into a ball of smoke where it lands.",
               charges=2, recharge=28.0, p={"radius": 4.2, "dur": 14.0}),
           _ab("nightfall", "NIGHTFALL", "X", "nearsight", "Every enemy sees no further than eight metres for seven "
               "seconds.", points=7, p={"dur": 7.0, "range": 8.0}))),
    Agent("echo", "ECHO", "Recon", (60, 232, 176), "Heard you three rooms ago.",
          "A survey engineer with a bag of listening gear. Her pings draw enemies on everyone's map; her static "
          "takes theirs away.",
          (_ab("static", "STATIC", "C", "static", "A puck that jams a patch of ground: enemies in it see a few "
               "metres and lose their map.", cost=200, charges=2, p={"radius": 5.0, "dur": 6.0, "range": 6.0}),
           _ab("post", "LISTENING POST", "Q", "sensor", "Set a sensor at your feet. It marks any enemy that comes "
               "within nine metres.", cost=200, p={"radius": 9.0, "dur": 60.0}),
           _ab("ping", "PING", "E", "sonar", "Throw a sounder. Three pulses show enemies near it through walls.",
               recharge=34.0, p={"radius": 12.0, "pulses": 3, "gap": 1.0, "reveal": 2.2}),
           _ab("sweep", "FULL SWEEP", "X", "sweep", "Mark every enemy on the map for six seconds.", points=7,
               p={"dur": 6.0}))),
    Agent("bulwark", "BULWARK", "Anchor", (90, 160, 255), "This one's taken.",
          "A site foreman who builds in a hurry. Throws up a barricade across a doorway, burns the next one and "
          "keeps the crew standing.",
          (_ab("scorch", "SCORCH", "C", "fire", "A canister that floods a circle with fire for seven seconds.",
               cost=300, p={"radius": 4.0, "dur": 7.0, "dps": 30}),
           _ab("patch", "PATCH KIT", "Q", "heal", "Mend the most hurt teammate within eight metres, or yourself: 60 "
               "health over five seconds.", cost=200, p={"heal": 60, "dur": 5.0, "range": 8.0}),
           _ab("barricade", "BARRICADE", "E", "barricade", "Raise a solid wall five metres wide in front of you for "
               "twenty seconds.", recharge=38.0, p={"width": 5, "dur": 20.0, "h": 2.7}),
           _ab("stronghold", "STRONGHOLD", "X", "stronghold", "Claim the ground around you for twelve seconds: enemies "
               "inside are slowed and marked, teammates inside mend armour.", points=8,
               p={"radius": 11.0, "dur": 12.0, "slow": 0.6, "armour": 10.0}))),
)}
AGENT_ORDER = ("kite", "fuse", "murk", "echo", "bulwark")
THROW_SPEED = 17.0
THROW_GRAVITY = 15.0

# --- bots ------------------------------------------------------------------------------------------------------------

DIFFICULTY = {
    #          reaction s, aim error deg, settle 1/s, turn deg/s, head aim chance, recoil control, ability use
    "easy": dict(react=0.72, err=6.5, settle=1.1, turn=240, head=0.1, control=0.2, util=0.5, lag=0.2,
                 label="RECRUIT"),
    "normal": dict(react=0.46, err=4.4, settle=1.7, turn=340, head=0.28, control=0.45, util=0.8, lag=0.13,
                   label="REGULAR"),
    "hard": dict(react=0.3, err=2.9, settle=2.5, turn=460, head=0.48, control=0.7, util=1.0, lag=0.07,
                 label="VETERAN"),
}
TEAM_NAMES = {"atk": "ATTACK", "def": "DEFENCE"}
BOT_NAMES = ("Harlow", "Pike", "Juno", "Sable", "Marsh", "Quill", "Rook", "Tamsin", "Brandt", "Odile", "Vance", "Iris",
             "Corbin", "Lark", "Dane", "Wren", "Holt", "Nessa")

# --- colours ---------------------------------------------------------------------------------------------------------

ACCENT = (255, 198, 46)  # the game's own amber
INK = (20, 18, 31)
PAPER = (240, 236, 226)
ALLY = (63, 184, 255)
ENEMY = (255, 61, 139)
