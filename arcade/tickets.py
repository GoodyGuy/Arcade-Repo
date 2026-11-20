"""Arcade tickets and the prize counter: tickets are earned by visiting (a daily bonus that grows with the streak),
inserting a coin in a game (the first time each game each day), playing (play time) and catching the critters that
wander over the home screen; they buy cosmetic unlocks for the home screen. Everything is kept in the arcade's own save
(``stats`` / ``settings`` of ``SaveStore.data``), never anywhere else:

* ``stats["tickets"]`` the balance, ``stats["tickets_earned"]`` all tickets ever earned;
* ``stats["last_visit"]`` / ``stats["streak"]`` the daily bonus, ``stats["launch_day"]`` the launch bonuses of today;
* ``stats["ticket_paid"]`` play seconds already paid in tickets (per game), ``stats["coins"]`` coins inserted,
  ``stats["critters"]`` critters caught, ``stats["parties"]`` secret parties thrown;
* ``stats["unlocks"]`` the prizes bought; ``settings["home_theme"]`` / ``["cabinet_trim"]`` / ``["mascot"]`` the
  prizes in use.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from dataclasses import dataclass

from app.arcade.saves import SaveStore

WELCOME_BONUS = 50
DAILY_BONUS = 25
STREAK_STEP = 5  # more per day of the streak
STREAK_MAX_BONUS = 60
LAUNCH_BONUS = 5
SECONDS_PER_TICKET = 20.0
PLAY_TICKETS_MAX = 150  # one session pays at most this much
PARTY_BONUS = 100  # the secret code, once a day (later the same day: PARTY_REPEAT)
PARTY_REPEAT = 10


@dataclass(frozen=True)
class Prize:
    prize_id: str
    kind: str  # theme / trim / mascot (one of each in use) or toy (owned is enough)
    name: str
    price: int
    blurb: str

    @property
    def value(self) -> str:
        return self.prize_id.split(".", 1)[1]


PRIZES = (
    Prize("theme.synthwave", "theme", "SYNTHWAVE", 0, "Neon sun, endless grid"),
    Prize("theme.carpet", "theme", "80S CARPET", 150, "Cosmic carpet, spotlights"),
    Prize("theme.cyber", "theme", "CYBER CITY", 300, "Rainy neon skyline"),
    Prize("theme.spooky", "theme", "SPOOKY NIGHT", 400, "Full moon, bats, fog"),
    Prize("trim.neon", "trim", "GAME NEON", 0, "Each game's own colour"),
    Prize("trim.chrome", "trim", "CHROME", 100, "Polished steel trim"),
    Prize("trim.gold", "trim", "GOLD", 250, "For high rollers"),
    Prize("trim.rainbow", "trim", "RAINBOW", 500, "Every colour, always"),
    Prize("mascot.none", "mascot", "NO MASCOT", 0, "Just the games"),
    Prize("mascot.invader", "mascot", "PIXEL PAL", 120, "A friendly invader"),
    Prize("mascot.ghost", "mascot", "BOO", 200, "A shy little ghost"),
    Prize("mascot.cat", "mascot", "ARCADE CAT", 350, "Naps on the marquee"),
    Prize("toy.popper", "toy", "CONFETTI POPPER", 60, "Press C to celebrate"),
    Prize("toy.disco", "toy", "DISCO BALL", 450, "Press B for a party"),
)
KIND_SETTING = {"theme": "home_theme", "trim": "cabinet_trim", "mascot": "mascot"}
DEFAULTS = {"theme": "synthwave", "trim": "neon", "mascot": "none"}


def prize(prize_id: str) -> Prize | None:
    return next((p for p in PRIZES if p.prize_id == prize_id), None)


def _today(today: datetime.date | None) -> datetime.date:
    return today or datetime.date.today()


class TicketBook:
    """The ticket balance and the prizes, kept in ``store`` (see the module docstring). ``on_award(n, reason)`` is
    called for every award (the home screen shows the tickets flying in)."""

    def __init__(self, store: SaveStore) -> None:
        self.store = store
        self.on_award: Callable[[int, str], None] | None = None

    @property
    def stats(self) -> dict:
        return self.store.data["stats"]

    @property
    def balance(self) -> int:
        return int(self.stats.get("tickets", 0))

    @property
    def earned(self) -> int:
        return int(self.stats.get("tickets_earned", 0))

    def count(self, key: str, n: int = 1) -> int:
        self.stats[key] = int(self.stats.get(key, 0)) + n
        return self.stats[key]

    def award(self, n: int, reason: str) -> int:
        n = int(max(n, 0))
        if n:
            self.stats["tickets"] = self.balance + n
            self.stats["tickets_earned"] = self.earned + n
            if self.on_award is not None:
                self.on_award(n, reason)
        return n

    # --- ways to earn ---------------------------------------------------------------------------------------------

    def daily_visit(self, today: datetime.date | None = None) -> int:
        """The first visit of a day: the daily bonus (and the welcome bonus the very first time)."""
        day = _today(today)
        last = self.stats.get("last_visit")
        if last == day.isoformat():
            return 0
        total = 0
        if last is None:
            total += self.award(WELCOME_BONUS, "WELCOME BONUS")
            streak = 1
        else:
            try:
                gap = (day - datetime.date.fromisoformat(last)).days
            except (TypeError, ValueError):
                gap = 99
            streak = int(self.stats.get("streak", 0)) + 1 if gap == 1 else 1
        self.stats["last_visit"] = day.isoformat()
        self.stats["streak"] = streak
        self.stats["best_streak"] = max(int(self.stats.get("best_streak", 0)), streak)
        bonus = min(DAILY_BONUS + STREAK_STEP * (streak - 1), STREAK_MAX_BONUS)
        total += self.award(bonus, f"DAY {streak} BONUS" if streak > 1 else "DAILY BONUS")
        return total

    def launch_bonus(self, game_id: str, today: datetime.date | None = None) -> int:
        """A coin inserted: tickets for the first launch of each game each day."""
        self.count("coins")
        day = _today(today).isoformat()
        launched = self.stats.get("launch_day")
        if not isinstance(launched, dict) or launched.get("date") != day:
            launched = {"date": day, "games": []}
        if game_id in launched["games"]:
            self.stats["launch_day"] = launched
            return 0
        launched["games"].append(game_id)
        self.stats["launch_day"] = launched
        return self.award(LAUNCH_BONUS, "COIN BONUS")

    def pay_play_time(self) -> int:
        """Tickets for the play time not paid yet (one per ``SECONDS_PER_TICKET``)."""
        played = self.store.data["stats"].get("play_seconds", {})
        paid = self.stats.setdefault("ticket_paid", {})
        total = 0
        for game_id, seconds in played.items():
            done = min(float(paid.get(game_id, 0.0)), float(seconds))  # a reset game starts again from 0
            n = int((seconds - done) // SECONDS_PER_TICKET)
            paid[game_id] = round(done + n * SECONDS_PER_TICKET, 1)
            total += n
        for game_id in list(paid):
            if game_id not in played:
                paid.pop(game_id)
        return self.award(min(total, PLAY_TICKETS_MAX), "PLAY TIME")

    def game_reset(self, game_id: str) -> None:
        """A game's progress (and play time) was erased: its play time is counted from 0 again."""
        self.stats.setdefault("ticket_paid", {}).pop(game_id, None)

    def critter(self, value: int) -> int:
        self.count("critters")
        return self.award(value, "CRITTER!")

    def party(self, today: datetime.date | None = None) -> int:
        """The secret code: a big bonus once a day."""
        self.count("parties")
        day = _today(today).isoformat()
        if self.stats.get("party_day") == day:
            return self.award(PARTY_REPEAT, "PARTY!")
        self.stats["party_day"] = day
        return self.award(PARTY_BONUS, "SECRET BONUS")

    # --- prizes ---------------------------------------------------------------------------------------------------

    @property
    def unlocks(self) -> list[str]:
        unlocks = self.stats.get("unlocks")
        if not isinstance(unlocks, list):
            unlocks = self.stats["unlocks"] = []
        return unlocks

    def owns(self, prize_id: str) -> bool:
        item = prize(prize_id)
        return item is not None and (item.price == 0 or prize_id in self.unlocks)

    def equipped(self, kind: str) -> str:
        """The value in use for a kind (theme / trim / mascot), a default one when it is not owned."""
        value = self.store.settings.get(KIND_SETTING[kind], DEFAULTS[kind])
        return value if self.owns(f"{kind}.{value}") else DEFAULTS[kind]

    def in_use(self, prize_id: str) -> bool:
        item = prize(prize_id)
        if item is None:
            return False
        if item.kind == "toy":
            return self.owns(prize_id)
        return self.equipped(item.kind) == item.value

    def buy(self, prize_id: str) -> str:
        """'equipped' (owned: now in use), 'bought' (and in use), 'short' (not enough tickets) or 'unknown'."""
        item = prize(prize_id)
        if item is None:
            return "unknown"
        if not self.owns(prize_id):
            if self.balance < item.price:
                return "short"
            self.stats["tickets"] = self.balance - item.price
            self.unlocks.append(prize_id)
            self.equip(prize_id)
            return "bought"
        self.equip(prize_id)
        return "equipped"

    def equip(self, prize_id: str) -> None:
        item = prize(prize_id)
        if item is not None and item.kind in KIND_SETTING and self.owns(prize_id):
            self.store.settings[KIND_SETTING[item.kind]] = item.value
