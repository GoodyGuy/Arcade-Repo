"""Case Clash: the game widget - Panorama-style menus (play, inventory, market, trade up, cases, surf, stats), the
case opening reel and reveal, the inspect view, trade up contracts, the Aim Arena for money, and the first-person
surf maps (surf.py) that pay for runs under par and drop a case on a map's first clear."""

from __future__ import annotations

import datetime as _dt
import json
import math
import random
import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient

from app.arcade.base import ArcadeGame
from app.arcade.games.case_clash import art as A
from app.arcade.games.case_clash import data as D
from app.arcade.games.case_clash import sounds
from app.arcade.games.case_clash import ui as U
from app.arcade.games.case_clash import surf_maps as SM
from app.arcade.games.case_clash.arena import Arena, scene_preview
from app.arcade.games.case_clash.surf import SurfRun
from app.arcade.games.case_clash.surf import fmt as surf_fmt
from app.arcade.games.case_clash.surf import placeholder as surf_placeholder
from app.arcade.games.case_clash.surf import preview as surf_preview
from app.arcade.games.case_clash.surf_render import SurfRenderer

AL = Qt.AlignmentFlag
TABS = (("play", "Play"), ("inventory", "Inventory"), ("market", "Market"), ("tradeup", "Trade Up"),
        ("cases", "Cases"), ("surf", "Surf"), ("stats", "Stats"))
SURF_TIERS = {1: "EASY", 2: "EASY", 3: "MEDIUM", 4: "HARD", 5: "HARD"}
FILTERS = (("all", "All"), ("special", "★ Knives & Gloves"), ("rifle", "Rifles"), ("sniper", "Snipers"),
           ("pistol", "Pistols"), ("smg", "SMGs"), ("heavy", "Heavy"))
SORTS = (("newest", "Newest"), ("rarity", "Rarity"), ("value", "Value"), ("name", "Name"))
START_CASES = {"sandstorm": 2, "glacier": 1, "neon": 1}
DAILY_BASE = 300  # cents
REEL_SECONDS = 6.2


class Reel:
    """The CS:GO case reel: ~50 tiles, the won item near the end; the marker lands at a random point inside it."""

    N = 52
    WIN = 44

    def __init__(self, case_id: str, item: D.Item, rng: random.Random, duration: float = REEL_SECONDS) -> None:
        self.case_id = case_id
        self.item = item
        case = D.CASES[case_id]
        self.tiles: list[D.Item] = []
        for i in range(self.N):
            if i == self.WIN:
                self.tiles.append(item)
                continue
            sk = rng.choice(case.tier(D.roll_rarity(rng)) or case.tier(D.MILSPEC))
            self.tiles.append(D.Item(-1, sk.id, (sk.fmin + sk.fmax) / 2, 17))
        self.offset = rng.uniform(0.08, 0.92)
        self.start = 2.5
        self.end = self.WIN + self.offset
        self.duration = max(duration, 0.0)
        self.t = 0.0
        self.last = int(self.start)

    def pos(self) -> float:
        """The marker's position in tiles from the first tile's left edge."""
        if self.duration <= 0:
            return self.end
        u = min(self.t / self.duration, 1.0)
        return self.start + (self.end - self.start) * (1 - (1 - u) ** 3.6)

    @property
    def done(self) -> bool:
        return self.t >= self.duration

    def landed(self) -> D.Item:
        return self.tiles[int(math.floor(self.pos()))]


class CaseClashGame(ArcadeGame):
    game_id = "case_clash"
    title = "Case Clash"
    subtitle = "Weapon cases · skins · trade ups · surf"
    description = ("Open weapon cases on the reel, chase ★ knives, gloves and blue gems, trade up ten skins for a "
                   "better one, sell on the market, earn your keys in the desert Aim Arena, and race three "
                   "first-person surf maps for cash and case drops.")
    controls = (("Mouse", "Menus and buttons; aim (Aim Arena); look and steer (Surf)"), ("Wheel", "Scroll lists"),
                ("1-7", "Switch tabs"), ("Space", "Open another case on the reveal; jump (Surf, hold to hop)"),
                ("R", "Reload (Aim Arena); back to the checkpoint (Surf)"),
                ("W A S D", "Surf: walk; strafe into the ramp"), ("Esc", "Close a window / pause"))
    accent = "#e4ae39"
    # Space and R also jump and go back to the checkpoint in Surf: one key cannot be the default of two actions
    bindings = (("Play tab", Qt.Key.Key_1), ("Inventory tab", Qt.Key.Key_2), ("Market tab", Qt.Key.Key_3),
                ("Trade Up tab", Qt.Key.Key_4), ("Cases tab", Qt.Key.Key_5), ("Stats tab", Qt.Key.Key_6),
                ("Open another case", Qt.Key.Key_Space), ("Keep the drop", Qt.Key.Key_Return),
                ("Reload (Aim Arena)", Qt.Key.Key_R), ("Surf tab", Qt.Key.Key_7),
                ("Surf: forward", Qt.Key.Key_W), ("Surf: back", Qt.Key.Key_S),
                ("Surf: strafe left", Qt.Key.Key_A), ("Surf: strafe right", Qt.Key.Key_D))
    TAB_KEYS = {Qt.Key.Key_1: "play", Qt.Key.Key_2: "inventory", Qt.Key.Key_3: "market", Qt.Key.Key_4: "tradeup",
                Qt.Key.Key_5: "cases", Qt.Key.Key_6: "stats", Qt.Key.Key_7: "surf"}

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        sounds.build(self.sfx)
        self.rng = random.Random()
        self.hits = U.Hits()
        self._bg_cache: dict = {}
        self._pm_cache: dict = {}
        self._values: dict[int, tuple[int, int]] = {}
        self._tiles: dict[tuple, QPixmap] = {}
        self._under: tuple | None = None
        self._screen: tuple | None = None
        self._day = 0
        self._day_at = -1e9
        self.reset_progress()
        # ui state
        self.screen = "play"
        self.modal: str | None = None
        self.modal_case = D.CASE_ORDER[0]
        self.reel: Reel | None = None
        self.reveal_item: D.Item | None = None
        self.reveal_t = 0.0
        self.reveal_from = "case"
        self.inspect_item: D.Item | None = None
        self.confirm: tuple[str, object] | None = None
        self.confirm_back: str | None = None
        self.contract: list[int] = []
        self.contract_t = 0.0
        self.contract_result: D.Item | None = None
        self.inv_filter = "all"
        self.inv_sort = "newest"
        self.select_mode = False
        self.selected: set[int] = set()
        self.scroll: dict[str, float] = {}
        self.scroll_max: dict[str, float] = {}
        self.toasts: list[list] = []
        self.particles: list[list] = []
        self.feed: list[list] = []
        self.feed_in = 0.0
        self.arena: Arena | None = None
        self.surf: SurfRun | None = None
        self._surf_renderer: SurfRenderer | None = None
        for _ in range(8):
            self._market_sale(backdate=True)

    # --- progress -------------------------------------------------------------------------------------------------

    def reset_progress(self) -> None:
        self.wallet = D.START_CENTS
        self.keys_owned = 1
        self.cases = dict(START_CASES)
        self.items: list[D.Item] = []
        self.next_uid = 1
        self.loadout: dict[str, int | None] = {"gun": None, "knife": None, "gloves": None}
        self.xp = 0
        self.stats = self._blank_stats()
        self.achievements: set[str] = set()
        self.last_daily = ""
        self.streak = 0
        self.settings = {"quick": False}
        self.sales: list[list] = []
        self.surf_data = self._blank_surf()

    @staticmethod
    def _blank_stats() -> dict:
        return {"opened": 0, "keys": 0, "spent": 0, "earned": 0, "sold": 0, "trade_ups": 0, "drops": [0] * 7,
                "best": None, "counter": 0, "rounds": 0, "kills": 0, "hs": 0, "shots": 0, "hits": 0,
                "best_round": 0, "arena_earned": 0, "surf_runs": 0, "surf_finishes": 0, "surf_earned": 0}

    @staticmethod
    def _blank_surf() -> dict:
        """Surf progress: best time and its splits per map, the maps cleared and those done under gold."""
        return {"best": {}, "splits": {}, "clears": [], "golds": []}

    @staticmethod
    def _load_surf(raw) -> dict:  # noqa: ANN001 - whatever an earlier version saved (or nothing)
        out = CaseClashGame._blank_surf()
        if not isinstance(raw, dict):
            return out
        best = raw.get("best")
        if isinstance(best, dict):
            for k, v in best.items():
                if k in SM.INFO and isinstance(v, (int, float)) and not isinstance(v, bool) and 1.0 < v < 36000:
                    out["best"][k] = round(float(v), 2)
        splits = raw.get("splits")
        if isinstance(splits, dict):
            for k, v in splits.items():
                if k in out["best"] and isinstance(v, list) and all(
                        isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                    out["splits"][k] = [round(float(x), 2) for x in v[:12]]
        for key in ("clears", "golds"):
            if isinstance(raw.get(key), list):
                out[key] = [k for k in SM.MAP_ORDER if k in raw[key]]
        return out

    def save_state(self) -> dict:
        return {"v": 1, "wallet": self.wallet, "keys": self.keys_owned, "cases": dict(self.cases),
                "items": [i.to_json() for i in self.items], "uid": self.next_uid, "loadout": dict(self.loadout),
                "xp": self.xp, "stats": json.loads(json.dumps(self.stats)), "ach": sorted(self.achievements),
                "daily": self.last_daily, "streak": self.streak, "settings": dict(self.settings),
                "sales": [list(s) for s in self.sales[:20]], "surf": json.loads(json.dumps(self.surf_data))}

    def load_state(self, data: dict) -> None:
        self.reset_progress()
        if not isinstance(data, dict) or not data:
            return

        def num(key: str, default: int) -> int:
            try:
                return int(data.get(key, default))
            except (TypeError, ValueError):
                return default

        self.wallet = max(0, num("wallet", D.START_CENTS))
        self.keys_owned = max(0, num("keys", 0))
        cases = data.get("cases")
        if isinstance(cases, dict):
            self.cases = {k: max(0, int(v)) for k, v in cases.items() if k in D.CASES and isinstance(v, int)}
        items = data.get("items")
        if isinstance(items, list):
            seen: set[int] = set()
            for it in (D.Item.from_json(d) for d in items if isinstance(d, dict)):
                if it is not None and it.uid not in seen:
                    seen.add(it.uid)
                    self.items.append(it)
        self.next_uid = max([num("uid", 1)] + [i.uid + 1 for i in self.items])
        uids = {i.uid: i for i in self.items}
        lo = data.get("loadout")
        if isinstance(lo, dict):
            for slot in self.loadout:
                v = lo.get(slot)
                it = uids.get(v) if isinstance(v, int) else None
                kind = it.skin.kind if it else ""
                wanted = {"knife": ("knife",), "gloves": ("gloves",)}.get(
                    slot, ("pistol", "smg", "rifle", "sniper", "heavy"))
                self.loadout[slot] = v if kind in wanted else None
        self.xp = max(0, num("xp", 0))
        st = data.get("stats")
        if isinstance(st, dict):
            for k, v in st.items():
                if k in self.stats and k not in ("drops", "best") and isinstance(v, int):
                    self.stats[k] = max(0, v)
            drops = st.get("drops")
            if isinstance(drops, list) and len(drops) == 7 and all(isinstance(x, int) for x in drops):
                self.stats["drops"] = [max(0, x) for x in drops]
            best = st.get("best")
            if isinstance(best, dict) and isinstance(best.get("name"), str):
                try:
                    self.stats["best"] = {"name": best["name"], "value": int(best.get("value", 0) or 0),
                                          "rarity": int(best.get("rarity", 2) or 2) % 7}
                except (TypeError, ValueError):
                    self.stats["best"] = None
        ach = data.get("ach")
        if isinstance(ach, list):
            self.achievements = {a for a in ach if a in D.ACHIEVEMENT_IDS}
        self.last_daily = str(data.get("daily", ""))
        self.streak = max(0, num("streak", 0))
        if isinstance(data.get("settings"), dict):
            self.settings["quick"] = bool(data["settings"].get("quick", False))
        if isinstance(data.get("sales"), list):
            self.sales = [s for s in data["sales"] if isinstance(s, list) and len(s) == 3 and isinstance(s[0], str)
                          and isinstance(s[1], int) and isinstance(s[2], (int, float))][:20]
        self.surf_data = self._load_surf(data.get("surf"))

    @classmethod
    def paint_thumbnail(cls, p: QPainter, rect: QRectF, t: float) -> None:
        _paint_thumbnail(p, QRectF(rect), t)

    @classmethod
    def summary(cls, data: dict) -> str:
        if not isinstance(data, dict) or not data:
            return f"New game · {D.money(D.START_CENTS)} and {sum(START_CASES.values())} cases to start"
        raw = data.get("items")
        items = [it for it in (D.Item.from_json(d) for d in (raw if isinstance(raw, list) else [])
                               if isinstance(d, dict)) if it]
        try:
            wallet = int(data.get("wallet", 0))
        except (TypeError, ValueError):
            wallet = 0
        line = f"{D.money(wallet)} · {len(items)} item{'s' if len(items) != 1 else ''}"
        if items:
            best = max(items, key=D.item_value)
            line += f" · best {best.skin.title}"
        return line

    # --- money, xp, achievements ----------------------------------------------------------------------------------

    def today(self) -> _dt.date:
        return _dt.date.today()

    def day(self) -> int:
        """Today's market day (looked up at most every few seconds)."""
        now = time.monotonic()
        if now - self._day_at > 5.0:
            self._day = D.today_index(self.today())
            self._day_at = now
        return self._day

    def value(self, item: D.Item) -> int:
        day = self.day()
        got = self._values.get(item.uid)
        if item.uid < 0 or got is None or got[0] != day:
            v = D.item_value(item, day)
            if item.uid >= 0:
                self._values[item.uid] = (day, v)
            return v
        return got[1]

    def earn(self, cents: int, kind: str = "") -> None:
        self.wallet += cents
        self.stats["earned"] += cents
        if kind == "arena":
            self.stats["arena_earned"] += cents
        if self.wallet >= 10000:
            self.unlock("rich")

    def spend(self, cents: int) -> bool:
        if cents > self.wallet:
            self.toast("Not enough funds in your wallet", "#d64545")
            self.sfx.play("deny")
            return False
        self.wallet -= cents
        self.stats["spent"] += cents
        return True

    @property
    def level(self) -> int:
        return D.level_of(self.xp)[0]

    def add_xp(self, n: int) -> None:
        before = self.level
        self.xp += n
        after = self.level
        for lv in range(before + 1, after + 1):
            cid = self.rng.choice(D.CASE_ORDER)
            self.cases[cid] = self.cases.get(cid, 0) + 1
            self.toast(f"LEVEL {lv} · {D.rank_name(lv)} · level-up drop: {D.CASES[cid].name}", "#e4ae39")
            self.sfx.play("levelup")
            if lv >= 10:
                self.unlock("level_10")

    def unlock(self, aid: str) -> None:
        if aid in self.achievements or aid not in D.ACHIEVEMENT_IDS:
            return
        self.achievements.add(aid)
        name = next(a[1] for a in D.ACHIEVEMENTS if a[0] == aid)
        self.wallet += 100
        self.stats["earned"] += 100
        self.toast(f"ACHIEVEMENT UNLOCKED · {name} · +$1.00", "#8bd45b")
        self.sfx.play("achieve")

    def check_item_achievements(self, item: D.Item) -> None:
        s = item.skin
        if item.rarity >= D.COVERT:
            self.unlock("first_covert")
        if s.special:
            self.unlock("first_knife")
        if s.finish == "hardened" and D.blue_share(item.pattern) >= 0.8:
            self.unlock("blue_gem")
        if s.finish == "doppler" and D.doppler_phase(item.pattern) in D.GEMS:
            self.unlock("gem")
        if item.counter:
            self.unlock("counter")
        if item.float < 0.01:
            self.unlock("fn_low")

    def toast(self, msg: str, color: str = "#e9edf1") -> None:
        self.toasts.append([msg, color, 0.0])
        del self.toasts[:-4]

    # --- store ----------------------------------------------------------------------------------------------------

    def buy_key(self, n: int = 1) -> bool:
        if not self.spend(D.KEY_PRICE * n):
            return False
        self.keys_owned += n
        self.stats["keys"] += n
        self.sfx.play("buy")
        self.progress_changed.emit()
        return True

    def buy_case(self, case_id: str, n: int = 1) -> bool:
        if not self.spend(D.case_price(case_id, self.day()) * n):
            return False
        self.cases[case_id] = self.cases.get(case_id, 0) + n
        self.sfx.play("buy")
        self.progress_changed.emit()
        return True

    def open_cost(self, case_id: str) -> int:
        cost = 0 if self.cases.get(case_id, 0) > 0 else D.case_price(case_id, self.day())
        return cost + (0 if self.keys_owned > 0 else D.KEY_PRICE)

    def open_case(self, case_id: str) -> D.Item | None:
        """Unlock a case (buying the case and / or the key first when needed); the item goes straight into the
        inventory and is saved, then the reel shows it."""
        cost = self.open_cost(case_id)
        if cost > self.wallet:
            self.spend(cost)  # refused with a message
            return None
        if self.cases.get(case_id, 0) <= 0:
            self.wallet -= D.case_price(case_id, self.day())
            self.stats["spent"] += D.case_price(case_id, self.day())
            self.cases[case_id] = 1
        if self.keys_owned <= 0:
            self.wallet -= D.KEY_PRICE
            self.stats["spent"] += D.KEY_PRICE
            self.stats["keys"] += 1
            self.keys_owned = 1
        self.keys_owned -= 1
        self.cases[case_id] -= 1
        item = D.roll_item(case_id, self.rng, self.next_uid, int(time.time()))
        self.next_uid += 1
        self.items.append(item)
        st = self.stats
        st["opened"] += 1
        st["drops"][item.rarity] += 1
        if item.counter:
            st["counter"] += 1
        v = self.value(item)
        if not st["best"] or v > st["best"].get("value", 0):
            st["best"] = {"name": item.full_name, "value": v, "rarity": item.rarity}
        self.add_xp(5)
        self.unlock("first_case")
        if st["opened"] >= 25:
            self.unlock("cases_25")
        if st["opened"] >= 100:
            self.unlock("cases_100")
        self.check_item_achievements(item)
        self.sfx.play("unlock")
        self.modal_case = case_id
        self.reel = Reel(case_id, item, self.rng, 0.0 if self.settings["quick"] else REEL_SECONDS)
        self.modal = "reel"
        self.reveal_item = item
        self.progress_changed.emit()
        if self.settings["quick"]:
            self.show_reveal(item, "case")
        return item

    def sell(self, items: list[D.Item]) -> int:
        """Sell on the market: the price minus the 15 % fee goes to the wallet."""
        mine = [i for i in items if i in self.items]
        if not mine:
            return 0
        total = 0
        for it in mine:
            gross = self.value(it)
            net = D.sale_proceeds(gross)
            total += net
            self.items.remove(it)
            self.sales.insert(0, [it.full_name, gross, int(time.time())])
            for slot, uid in self.loadout.items():
                if uid == it.uid:
                    self.loadout[slot] = None
            self.selected.discard(it.uid)
            if it.uid in self.contract:
                self.contract.remove(it.uid)
        del self.sales[20:]
        self.wallet += total
        self.stats["earned"] += total
        self.stats["sold"] += len(mine)
        if self.wallet >= 10000:
            self.unlock("rich")
        self.sfx.play("cash")
        self.toast(f"Sold {len(mine)} item{'s' if len(mine) > 1 else ''} · +{D.money(total)} after the 15% fee",
                   "#8bd45b")
        self.progress_changed.emit()
        return total

    def equip(self, item: D.Item) -> None:
        slot = {"knife": "knife", "gloves": "gloves"}.get(item.skin.kind, "gun")
        self.loadout[slot] = item.uid
        self.toast(f"Equipped {item.skin.title}", "#5b9bd5")
        self.sfx.play("slot")
        self.progress_changed.emit()

    def item_by_uid(self, uid: int | None) -> D.Item | None:
        if uid is None:
            return None
        return next((i for i in self.items if i.uid == uid), None)

    def equipped_gun(self) -> tuple[D.Skin, int, float, D.Item | None]:
        it = self.item_by_uid(self.loadout.get("gun"))
        if it is None:
            return D.DEFAULT_GUN, 0, 0.0, None
        return it.skin, it.pattern, it.float, it

    def equipped_gloves(self) -> D.Skin | None:
        it = self.item_by_uid(self.loadout.get("gloves"))
        return it.skin if it else None

    def can_claim_daily(self) -> bool:
        return self.last_daily != self.today().isoformat()

    def claim_daily(self) -> bool:
        if not self.can_claim_daily():
            return False
        today = self.today()
        yesterday = (today - _dt.timedelta(days=1)).isoformat()
        self.streak = self.streak + 1 if self.last_daily == yesterday else 1
        self.last_daily = today.isoformat()
        cents = DAILY_BASE + 50 * min(self.streak - 1, 6)
        cid = self.rng.choice(D.CASE_ORDER)
        self.cases[cid] = self.cases.get(cid, 0) + 1
        self.earn(cents)
        self.sfx.play("cash")
        self.toast(f"Daily bonus · +{D.money(cents)} and a {D.CASES[cid].name} (day {self.streak} streak)", "#e4ae39")
        self.progress_changed.emit()
        return True

    # --- trade up -------------------------------------------------------------------------------------------------

    def contract_items(self) -> list[D.Item]:
        return [it for it in (self.item_by_uid(u) for u in self.contract) if it]

    def contract_fits(self, item: D.Item) -> bool:
        cur = self.contract_items()
        return len(cur) < 10 and item.uid not in self.contract and not D.trade_problem(cur + [item])

    def contract_toggle(self, item: D.Item) -> None:
        if item.uid in self.contract:
            self.contract.remove(item.uid)
            self.sfx.play("click")
        elif self.contract_fits(item):
            self.contract.append(item.uid)
            self.sfx.play("slot")
        else:
            cur = self.contract_items()
            why = D.trade_problem(cur + [item]) or "The contract is full"
            self.toast(why, "#d64545")
            self.sfx.play("deny")

    def contract_autofill(self) -> None:
        cur = self.contract_items()
        pool = sorted((i for i in self.items if i.uid not in self.contract and i.uid not in self.loadout.values()),
                      key=self.value)
        if not cur:
            for r in D.TRADE_RARITIES:  # the lowest rarity with ten of a kind
                for counter in (False, True):
                    group = [i for i in pool if i.rarity == r and i.counter == counter and not i.skin.special]
                    if len(group) >= 10:
                        self.contract = [i.uid for i in group[:10]]
                        self.sfx.play("slot")
                        return
            pool_ok = [i for i in pool if not D.trade_problem([i])]
            if pool_ok:
                self.contract = [pool_ok[0].uid]
                cur = self.contract_items()
        for it in pool:
            if len(self.contract) >= 10:
                break
            if it.uid not in self.contract and not D.trade_problem(cur + [it]):
                self.contract.append(it.uid)
                cur.append(it)
        self.sfx.play("slot")

    def submit_contract(self) -> D.Item | None:
        items = self.contract_items()
        if len(items) != 10 or D.trade_problem(items):
            self.toast("A trade up contract takes ten items of the same rarity", "#d64545")
            self.sfx.play("deny")
            return None
        out = D.trade_up(items, self.rng, self.next_uid, int(time.time()))
        self.next_uid += 1
        for it in items:
            self.items.remove(it)
            for slot, uid in self.loadout.items():
                if uid == it.uid:
                    self.loadout[slot] = None
        self.items.append(out)
        self.contract = []
        self.stats["trade_ups"] += 1
        v = self.value(out)
        if not self.stats["best"] or v > self.stats["best"].get("value", 0):
            self.stats["best"] = {"name": out.full_name, "value": v, "rarity": out.rarity}
        self.unlock("trade_up")
        self.check_item_achievements(out)
        self.add_xp(20)
        self.contract_result = out
        self.contract_t = 0.0
        self.modal = "contract"
        self.sfx.play("whoosh")
        self.progress_changed.emit()
        return out

    # --- aim arena ------------------------------------------------------------------------------------------------

    def start_arena(self) -> None:
        self.modal = None
        self.arena = Arena(self)
        self.set_arena_cursor(True)
        self.sfx.play("tab")

    def close_arena(self) -> None:
        self.arena = None
        self.set_arena_cursor(False)
        self.screen = "play"
        self.progress_changed.emit()

    def set_arena_cursor(self, on: bool) -> None:
        """Capture the mouse for aiming while a round runs (sensitivity / invert from the Controls panel); a free
        cursor between rounds and in the menus."""
        if on:
            self.set_mouse_captured(True)
        else:
            self.set_mouse_captured(False)
            self.unsetCursor()

    def key_label(self, default: int) -> str:
        """The name of the key the player uses for an action whose default key is ``default``."""
        from app.arcade.controls import key_name

        return key_name(next((k for k, d in self._remap.items() if d == default), default))

    def open_controls(self) -> None:
        """The arcade's Controls panel (mouse sensitivity, invert, swapped buttons, keys)."""
        self.controls_requested.emit()

    def arena_done(self, st, alive: bool) -> dict:  # noqa: ANN001 - ArenaStats
        """The round is over: a bonus, XP, maybe a case drop, the stats."""
        acc = st.hits / st.shots if st.shots else 0.0
        bonus = int(st.kills * 3 * acc) + (50 if alive else 0) + st.best_combo * 2
        self.earn(bonus, "arena")
        xp = 40 + st.kills * 2
        self.add_xp(xp)
        drop = None
        if self.rng.random() < min(0.6, 0.15 + st.kills * 0.012):
            drop = self.rng.choice(D.CASE_ORDER)
            self.cases[drop] = self.cases.get(drop, 0) + 1
            self.sfx.play("unlock")
        else:
            self.sfx.play("levelup" if alive else "hurt")
        s = self.stats
        s["rounds"] += 1
        s["kills"] += st.kills
        s["hs"] += st.headshots
        s["shots"] += st.shots
        s["hits"] += st.hits
        s["best_round"] = max(s["best_round"], st.kills)
        if st.headshots >= 10:
            self.unlock("one_tap")
        if st.kills >= 50:
            self.unlock("arena_50")
        self.progress_changed.emit()
        return {"bonus": bonus, "xp": xp + st.kills * 10 + st.headshots * 6, "drop": drop}

    # --- surf -----------------------------------------------------------------------------------------------------

    def surf_renderer(self) -> SurfRenderer:
        if self._surf_renderer is None:
            self._surf_renderer = SurfRenderer()
        return self._surf_renderer

    def surf_best(self, map_id: str) -> float | None:
        return self.surf_data["best"].get(map_id)

    def surf_splits(self, map_id: str) -> list[float]:
        return list(self.surf_data["splits"].get(map_id, []))

    def surf_next_map(self, map_id: str) -> str | None:
        i = SM.MAP_ORDER.index(map_id)
        return SM.MAP_ORDER[i + 1] if i + 1 < len(SM.MAP_ORDER) else None

    def start_surf(self, map_id: str) -> None:
        if not self.surf_renderer().available():
            self.toast("Surf needs OpenGL 3.3, which this computer does not offer", "#d64545")
            self.sfx.play("deny")
            return
        self.modal = None
        self.surf = SurfRun(self, map_id)
        self.stats["surf_runs"] += 1
        self.set_surf_cursor(True)
        self.sfx.play("tab")

    def close_surf(self) -> None:
        self.surf = None
        self.set_surf_cursor(False)
        self.screen = "surf"
        self._screen = None
        self.progress_changed.emit()

    def set_surf_cursor(self, on: bool) -> None:
        """Mouse look while running (the arcade's sensitivity and invert-Y); a free cursor on the results."""
        self.set_mouse_captured(on)
        if not on:
            self.unsetCursor()

    def surf_finished(self, map_id: str, seconds: float, splits: list[float], falls: int, top: float) -> dict:
        """A run reached the end zone: the record, the pay (under par, double under gold), XP, the first clear's
        case drop, achievements."""
        m = SM.INFO[map_id]
        sd = self.surf_data
        seconds = round(seconds, 2)
        old = sd["best"].get(map_id)
        record = old is None or seconds < old
        if record:
            sd["best"][map_id] = seconds
            sd["splits"][map_id] = [round(x, 2) for x in splits]
        under_par, gold = seconds <= m["par"], seconds <= m["gold"]
        pay = (m["reward"] if under_par else 0) + (m["reward"] if gold else 0)
        if record and old is not None:
            pay += 50  # a new personal best
        drop = None
        if map_id not in sd["clears"]:
            sd["clears"] = [k for k in SM.MAP_ORDER if k in sd["clears"] or k == map_id]
            drop = m["case_id"]
            self.cases[drop] = self.cases.get(drop, 0) + 1
        if gold and map_id not in sd["golds"]:
            sd["golds"] = [k for k in SM.MAP_ORDER if k in sd["golds"] or k == map_id]
        if pay:
            self.earn(pay)
            self.stats["surf_earned"] += pay
        self.stats["surf_finishes"] += 1
        xp = 30 + 25 * m["tier"]
        self.add_xp(xp)
        self.unlock("surf_first")
        if gold:
            self.unlock("surf_gold")
        if len(sd["clears"]) == len(SM.MAP_ORDER):
            self.unlock("surf_all")
        self.sfx.play("surf_finish" if not record else "surf_record")
        self.progress_changed.emit()
        return {"record": record, "par": under_par, "gold": gold, "pay": pay, "xp": xp, "drop": drop,
                "falls": falls, "top": top}

    def stop(self) -> None:
        super().stop()
        if self._surf_renderer is not None:
            self._surf_renderer.close()
            self._surf_renderer = None

    # --- market feed ----------------------------------------------------------------------------------------------

    def _market_sale(self, backdate: bool = False) -> None:
        cid = self.rng.choice(D.CASE_ORDER)
        odds = {D.MILSPEC: 0.55, D.RESTRICTED: 0.25, D.CLASSIFIED: 0.12, D.COVERT: 0.06, D.SPECIAL: 0.02}
        sk = self.rng.choice(D.CASES[cid].tier(D.roll_rarity(self.rng, odds)))
        it = D.make_item(sk, self.rng, -1)
        ago = self.rng.uniform(5, 600) if backdate else 0.0
        self.feed.insert(0, [it.full_name, D.item_value(it), time.time() - ago, it.rarity])
        self.feed.sort(key=lambda e: -e[2])
        del self.feed[14:]

    # --- loop -----------------------------------------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        for t in self.toasts:
            t[2] += dt
        self.toasts = [t for t in self.toasts if t[2] < 4.0]
        if self.particles:
            for pt in self.particles:
                pt[0] += pt[2] * dt
                pt[1] += pt[3] * dt
                pt[3] += 420 * dt
                pt[4] -= dt
            self.particles = [pt for pt in self.particles if pt[4] > 0]
        self.feed_in -= dt
        if self.feed_in <= 0:
            self._market_sale()
            self.feed_in = self.rng.uniform(2.5, 6.0)
        if self.arena is not None:
            self.arena.tick(dt)
            return
        if self.surf is not None:
            self.surf.tick(dt)
            return
        if self.modal == "reel" and self.reel is not None:
            r = self.reel
            r.t += dt
            idx = int(math.floor(r.pos()))
            if idx != r.last:
                r.last = idx
                self.sfx.play("tick")
            if r.t >= r.duration + 0.45:
                self.show_reveal(r.item, "case")
        elif self.modal in ("reveal", "inspect"):
            self.reveal_t += dt
        elif self.modal == "contract":
            before = self.contract_t
            self.contract_t += dt
            if before < 1.35 <= self.contract_t:
                self.sfx.play("stamp")
            if self.contract_t >= 2.3 and self.contract_result is not None:
                self.show_reveal(self.contract_result, "contract")

    def show_reveal(self, item: D.Item, source: str) -> None:
        self.modal = "reveal"
        self.reveal_item = item
        self.reveal_from = source
        self.reveal_t = 0.0
        self.sfx.play(f"reveal{min(max(item.rarity, 2), 6)}")
        self.burst(item.rarity)

    def burst(self, rarity: int) -> None:
        w, h = self.width(), self.height()
        n = (20, 20, 30, 50, 80, 110, 220)[rarity]
        col = D.RARITY_COLORS[rarity]
        for _ in range(n):
            a = self.rng.uniform(0, math.tau)
            sp = self.rng.uniform(120, 520) * (1.4 if rarity == D.SPECIAL else 1.0)
            c = col if rarity < D.SPECIAL or self.rng.random() < 0.6 else self.rng.choice(("#fff2c4", "#ffd479"))
            self.particles.append([w / 2, h * 0.4, math.cos(a) * sp, math.sin(a) * sp - 260,
                                   self.rng.uniform(0.8, 1.8), c, self.rng.uniform(2, 5)])
        del self.particles[:-400]

    # --- input ----------------------------------------------------------------------------------------------------

    def handle_escape(self) -> bool:
        if self.arena is not None:
            if self.arena.phase == "results":
                self.close_arena()
                return True
            return False
        if self.surf is not None:
            if self.surf.phase == "done":
                self.close_surf()
                return True
            return False
        if self.modal == "reel" or self.modal == "contract":
            return False  # let the arcade pause; the reel waits
        if self.modal is not None:
            self.close_modal()
            return True
        if self.select_mode:
            self.select_mode = False
            self.selected.clear()
            return True
        return False

    def close_modal(self) -> None:
        back = self.confirm_back if self.modal == "confirm" else None
        self.confirm = None
        self.confirm_back = None
        self.particles.clear()
        if back == "inspect" and self.inspect_item in self.items:
            self.modal = "inspect"
            return
        self.modal = None
        self.reel = None

    def key_pressed(self, key: int, event) -> None:  # noqa: ANN001
        if self.arena is not None:
            self.arena.key(key)
            return
        if self.surf is not None:
            self.surf.key(key)
            return
        if self.modal == "reveal" and key in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if key == Qt.Key.Key_Space and self.reveal_from == "case":
                self.open_another()
            else:
                self.close_modal()
            return
        if self.modal is None and key in self.TAB_KEYS:
            self.set_screen(self.TAB_KEYS[key])

    def mouse_pressed(self, button: Qt.MouseButton, pos: QPointF) -> None:
        self.hits.mouse = QPointF(pos)
        if self.arena is not None:
            if self.arena.phase == "results":
                found = self.hits.find(pos)
                if found:
                    self.sfx.play(found[1])
                    found[0]()
            elif button == Qt.MouseButton.LeftButton:
                self.arena.press(pos)
            return
        if self.surf is not None:
            if self.surf.phase == "done":
                found = self.hits.find(pos)
                if found:
                    self.sfx.play(found[1])
                    found[0]()
            elif not self.mouse_captured:
                self.set_surf_cursor(True)
            return
        if button != Qt.MouseButton.LeftButton:
            if button == Qt.MouseButton.RightButton and self.modal in ("inspect", "case"):
                self.close_modal()
            return
        found = self.hits.find(pos)
        if found:
            action, sound = found
            if sound:
                self.sfx.play(sound)
            action()
        self.update()

    def mouse_released(self, button: Qt.MouseButton, pos: QPointF) -> None:
        if self.arena is not None and button == Qt.MouseButton.LeftButton:
            self.arena.release()

    def mouse_moved(self, pos: QPointF) -> None:
        self.hits.mouse = QPointF(pos)
        if self.arena is not None:
            self.arena.move(pos)

    def wheel(self, steps: int) -> None:
        if self.arena is not None or self.surf is not None:
            return
        key = self.modal_scroll_key() if self.modal else self.screen
        self.scroll[key] = min(max(self.scroll.get(key, 0.0) - steps * 90 * self.ui_scale(), 0.0),
                               self.scroll_max.get(key, 0.0))

    def modal_scroll_key(self) -> str:
        return "modal_" + str(self.modal)

    def on_pause(self) -> None:
        if self.arena is not None:
            self.arena.release()
        self.unsetCursor()
        self.sfx.stop_all()

    def on_resume(self) -> None:
        if self.arena is not None and self.arena.phase != "results":
            self.set_arena_cursor(True)
        if self.surf is not None and self.surf.phase == "run":
            self.set_surf_cursor(True)

    def set_screen(self, screen: str) -> None:
        if screen != self.screen:
            self.screen = screen
            self.select_mode = False
            self.selected.clear()

    # --- actions (bound to buttons) -------------------------------------------------------------------------------

    def open_case_modal(self, case_id: str) -> None:
        self.modal_case = case_id
        self.modal = "case"
        self.scroll["modal_case"] = 0.0

    def open_another(self) -> None:
        cid = self.reel.case_id if self.reel else self.modal_case
        self.particles.clear()
        if self.open_case(cid) is None:
            self.modal = "case"

    def inspect(self, item: D.Item) -> None:
        self.inspect_item = item
        self.modal = "inspect"
        self.reveal_t = 0.0
        self.particles.clear()

    def ask(self, question: str, action) -> None:  # noqa: ANN001
        self.confirm = (question, action)
        self.confirm_back = self.modal
        self.modal = "confirm"

    def toggle_select(self, item: D.Item) -> None:
        if item.uid in self.selected:
            self.selected.discard(item.uid)
        else:
            self.selected.add(item.uid)

    def sell_selected(self) -> None:
        items = [i for i in self.items if i.uid in self.selected]
        if not items:
            return
        net = sum(D.sale_proceeds(self.value(i)) for i in items)

        def go() -> None:
            self.sell(items)
            self.modal = None
            self.select_mode = False

        self.ask(f"Sell {len(items)} item{'s' if len(items) > 1 else ''} on the market for {D.money(net)} "
                 "(after the 15% fee)?", go)

    def sell_one(self, item: D.Item, confirm: bool = True) -> None:
        def go() -> None:
            self.sell([item])
            self.inspect_item = None
            self.modal = None

        if confirm and (item.skin.special or self.value(item) >= 500):
            self.ask(f"Sell {item.full_name} for {D.money(D.sale_proceeds(self.value(item)))} (after the fee)?", go)
        else:
            go()

    # --- painting -------------------------------------------------------------------------------------------------

    def ui_scale(self) -> float:
        return max(0.62, min(self.width() / 1280, self.height() / 780, 1.5))

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        p = QPainter(self)
        try:
            self.paint(p)
        finally:
            p.end()

    def paint(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            return
        s = self.ui_scale()
        A.set_dpr(self.devicePixelRatioF())  # every cached picture is made at the screen's own pixels
        prev = self.hits.regions
        self.hits.begin()
        if self.arena is not None:
            self._under = self._screen = None
            self.arena.paint(p, w, h, self.hits)
            self.paint_toasts(p, w, s, 70 * s)
            return
        if self.surf is not None:
            self._under = self._screen = None
            self.surf.paint(p, w, h, self.hits)
            self.paint_toasts(p, w, s, 96 * s)
            return
        nav = 56 * s
        if self.modal is None:
            # a menu screen only changes with the progress, the view settings or the hovered button: reuse it
            self._under = None
            sig = self.screen_signature(w, h, prev)
            if self._screen is None or self._screen[0] != sig:
                pm = A.new_pixmap(w, h)
                q = QPainter(pm)
                self.paint_screen(q, w, h, s)
                self.paint_nav(q, w, nav, s)
                q.end()
                self._screen = (sig, pm, list(self.hits.regions))
            else:
                self.hits.regions = list(self._screen[2])
            p.drawPixmap(0, 0, self._screen[1])
        else:
            # the screen under a modal is a still picture (taken when the modal opened)
            self._screen = None
            key = (w, h, self.screen, A.dpr())
            if self._under is None or self._under[0] != key:
                pm = A.new_pixmap(w, h)
                q = QPainter(pm)
                self.hits.live = False
                self.paint_screen(q, w, h, s)
                q.fillRect(QRectF(0, 0, w, h), QColor(4, 6, 9, 175))
                q.end()
                self._under = (key, pm)
            p.drawPixmap(0, 0, self._under[1])
            self.hits.live = False
            self.paint_nav(p, w, nav, s)
            p.fillRect(QRectF(0, 0, w, nav), QColor(4, 6, 9, 150))
            self.hits.regions = []
            self.hits.live = True
            getattr(self, "draw_modal_" + self.modal)(p, w, h, s)
        self.paint_particles(p)
        self.paint_toasts(p, w, s, nav + 8 * s)

    def screen_signature(self, w: int, h: int, regions: list) -> tuple:
        """Everything a menu screen shows (a change repaints it)."""
        m = self.hits.mouse
        hover = next((tuple(r.getRect()) for r, _a, _s, _l in reversed(regions) if r.contains(m)), None)
        st = self.stats
        live = (len(self.feed), self.feed[0][2] if self.feed else 0, int(time.time() // 5)) \
            if self.screen == "market" else None
        return (w, h, self.screen, hover, self.wallet, self.xp, self.keys_owned, self.next_uid, len(self.items),
                tuple(sorted(self.cases.items())), tuple(self.loadout.values()), self.last_daily,
                len(self.achievements), st["opened"], st["sold"], st["rounds"], st["kills"], st["trade_ups"],
                tuple(self.contract), len(self.selected), hash(frozenset(self.selected)), self.inv_filter,
                self.inv_sort, self.select_mode, self.settings["quick"], tuple(sorted(self.scroll.items())),
                self.day(), live, self.devicePixelRatioF(),
                json.dumps(self.surf_data, sort_keys=True) if self.screen == "surf" else None)

    def paint_screen(self, p: QPainter, w: int, h: int, s: float) -> None:
        p.drawPixmap(0, 0, U.background(w, h, self._bg_cache))
        nav = 56 * s
        pad = 18 * s
        content = QRectF(pad, nav + pad, w - 2 * pad, h - nav - 2 * pad)
        getattr(self, "paint_" + self.screen)(p, content, s)

    def paint_nav(self, p: QPainter, w: int, nav: float, s: float) -> None:
        g = QLinearGradient(0, 0, 0, nav)
        g.setColorAt(0.0, QColor("#1c232c"))
        g.setColorAt(1.0, QColor("#11161c"))
        p.fillRect(QRectF(0, 0, w, nav), g)
        p.fillRect(QRectF(0, nav - 1, w, 1), QColor(255, 255, 255, 30))
        # logo
        x = 18 * s
        U.text(p, QRectF(x, 0, 40 * s, nav), "★", 26 * s, U.GOLD, weight="bold")
        U.text(p, QRectF(x + 28 * s, 0, 160 * s, nav), "CASE CLASH", 24 * s, U.TEXT, weight="bold", cond=True)
        x += 175 * s
        # tabs
        right_w = 300 * s
        avail = w - x - right_w
        tw = min(118 * s, avail / len(TABS))
        for i, (key, label) in enumerate(TABS):
            r = QRectF(x + i * tw, 0, tw, nav)
            active = key == self.screen
            hov = self.hits.hover(r)
            if active:
                p.fillRect(r, QColor(255, 255, 255, 16))
                p.fillRect(QRectF(r.left(), nav - 3 * s, r.width(), 3 * s), U.TEXT)
            elif hov:
                p.fillRect(r, QColor(255, 255, 255, 9))
            U.text(p, r, label.upper(), 16 * s, U.TEXT if active or hov else U.DIM, AL.AlignHCenter, cond=True)
            if key == "play" and self.can_claim_daily():  # a bonus is waiting
                p.save()
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(U.GOLD)
                p.drawEllipse(QPointF(r.center().x() + U.text_width(label.upper(), 16 * s, cond=True) / 2 + 8 * s,
                                      r.center().y() - 7 * s), 3.5 * s, 3.5 * s)
                p.restore()
            self.hits.add(r, lambda k=key: self.set_screen(k), "tab", label)
        # wallet and level
        lvl, into, need = D.level_of(self.xp)
        rr = QRectF(w - right_w, 0, right_w - 16 * s, nav)
        U.text(p, QRectF(rr.left(), 6 * s, rr.width() - 64 * s, nav * 0.5), D.money(self.wallet), 22 * s, U.MONEY,
               AL.AlignRight, "bold", True)
        U.text(p, QRectF(rr.left(), nav * 0.5, rr.width() - 64 * s, nav * 0.42),
               f"{self.keys_owned} KEY{'S' if self.keys_owned != 1 else ''} · {sum(self.cases.values())} CASES",
               12 * s, U.DIM, AL.AlignRight, cond=True)
        self.paint_rank(p, QPointF(rr.right() - 24 * s, nav / 2), 19 * s, lvl, into / need)

    def paint_rank(self, p: QPainter, c: QPointF, r: float, lvl: int, frac: float) -> None:
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        g = QRadialGradient(c, r)
        g.setColorAt(0.0, QColor("#3b4f6b"))
        g.setColorAt(1.0, QColor("#1a2433"))
        p.setBrush(g)
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor(255, 255, 255, 40), r * 0.16))
        p.drawEllipse(c, r * 0.9, r * 0.9)
        p.setPen(QPen(QColor("#5b9bd5"), r * 0.16))
        p.drawArc(QRectF(c.x() - r * 0.9, c.y() - r * 0.9, r * 1.8, r * 1.8), 90 * 16, int(-360 * 16 * frac))
        p.restore()
        U.text(p, QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), str(lvl), r * 0.9, U.TEXT, AL.AlignHCenter, "bold")

    def paint_toasts(self, p: QPainter, w: int, s: float, y: float) -> None:
        for msg, color, t in self.toasts:
            a = min(1.0, t / 0.15, (4.0 - t) / 0.5)
            tw = min(U.text_width(msg, 15 * s) + 40 * s, w - 20 * s)
            r = QRectF((w - tw) / 2, y, tw, 34 * s)
            p.setOpacity(max(0.0, a))
            p.fillRect(r, QColor(12, 16, 21, 235))
            p.fillRect(QRectF(r.left(), r.top(), 3 * s, r.height()), QColor(color))
            p.setPen(QPen(QColor(255, 255, 255, 30), 1))
            p.drawRect(r)
            U.text(p, r.adjusted(14 * s, 0, -8 * s, 0), msg, 15 * s, QColor(color), AL.AlignHCenter)
            p.setOpacity(1.0)
            y += 40 * s

    def paint_particles(self, p: QPainter) -> None:
        if not self.particles:
            return
        for x, y, _vx, _vy, life, col, size in self.particles:
            c = QColor(col)
            c.setAlpha(int(255 * min(1.0, life / 0.5)))
            p.fillRect(QRectF(x, y, size, size), c)

    # --- item tiles -----------------------------------------------------------------------------------------------

    def paint_tile(self, p: QPainter, r: QRectF, item: D.Item, s: float, hover: bool = False,
                   selected: bool = False, dim: bool = False, hidden: bool = False) -> None:
        w, h = int(r.width()), int(r.height())
        if hidden and item.skin.special:
            p.drawPixmap(r.topLeft(), A.star_tile(w, h))
            return
        equipped = item.uid >= 0 and item.uid in self.loadout.values()
        key = ("tile", item.skin_id, item.pattern, round(item.float, 3), item.counter, item.uid >= 0, equipped, w, h,
               A.dpr())
        pm = self._tiles.get(key)
        if pm is None:
            if len(self._tiles) > 360:
                self._tiles.clear()
            pm = self._tiles[key] = self.render_tile(item, w, h, s, equipped)
        p.drawPixmap(r.topLeft(), pm)
        if hover:
            p.fillRect(r, QColor(255, 255, 255, 18))
            p.setPen(QPen(QColor(255, 255, 255, 120), 1))
            p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))
        if selected:
            p.setPen(QPen(QColor("#5b9bd5"), 2.5 * s))
            p.drawRect(r.adjusted(1, 1, -1, -1))
            cb = QRectF(r.right() - 22 * s, r.top() + 6 * s, 16 * s, 16 * s)
            p.fillRect(cb, QColor("#5b9bd5"))
            U.text(p, cb, "✓", 13 * s, U.TEXT, AL.AlignHCenter, "bold")
        if dim:
            p.fillRect(r, QColor(10, 12, 16, 160))

    def render_tile(self, item: D.Item, w: int, h: int, s: float, equipped: bool) -> QPixmap:
        """A Panorama item tile: background with the rarity glow and bar, the skin, its names and tags."""
        sk = item.skin
        pm = A.tile_bg(sk.rarity, w, h).copy()
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        aw, ah = int(w * 0.88), int(h * 0.6)
        p.drawPixmap(QPointF(w * 0.06, h * 0.035), A.art_pixmap(sk.id, item.pattern, item.float, aw, ah))
        fs = max(h * 0.095, 9)
        tx = QRectF(w * 0.06, 0, w * 0.89, h)
        wl = QRectF(tx.left(), h * 0.64, tx.width(), fs * 1.3)
        name = ("★ " if sk.special else "") + sk.weapon_name
        if item.counter and item.uid >= 0:
            U.text(p, wl, f"{D.COUNTER_NAME} {name}", fs, U.ORANGE, cond=True)
        else:
            U.text(p, wl, name, fs, U.DIM, cond=True)
        U.text(p, QRectF(tx.left(), wl.bottom(), tx.width(), fs * 1.35), sk.name or "Vanilla", fs * 1.1, U.TEXT,
               cond=True)
        if item.uid >= 0 and sk.finish != "vanilla":
            U.text(p, QRectF(5 * s, 3 * s, 40 * s, fs * 1.3), item.wear_short, fs * 0.85, U.FAINT, cond=True)
        if equipped:
            p.fillRect(QRectF(w - 9 * s, 5 * s, 4 * s, 4 * s), U.BLUE)
            U.text(p, QRectF(0, 2 * s, w - 16 * s, fs * 1.3), "EQUIPPED", fs * 0.75, U.BLUE, AL.AlignRight, cond=True)
        p.end()
        return pm

    def grid(self, area: QRectF, n: int, tw: float, th: float, gap: float, key: str) -> tuple[int, list[QRectF]]:
        """Tile rects of a scrolling grid (only the visible ones are returned with their index)."""
        cols = max(1, int((area.width() + gap) // (tw + gap)))
        tw = (area.width() - gap * (cols - 1)) / cols
        th = tw * th
        rows = (n + cols - 1) // cols
        total = rows * (th + gap) - gap
        self.scroll_max[key] = max(0.0, total - area.height())
        off = min(self.scroll.get(key, 0.0), self.scroll_max[key])
        self.scroll[key] = off
        out = []
        for i in range(n):
            r = QRectF(area.left() + (i % cols) * (tw + gap), area.top() + (i // cols) * (th + gap) - off, tw, th)
            if r.bottom() >= area.top() and r.top() <= area.bottom():
                out.append((i, r))
        return cols, out

    def scrollbar(self, p: QPainter, area: QRectF, key: str, s: float) -> None:
        mx = self.scroll_max.get(key, 0.0)
        if mx <= 0:
            return
        frac = area.height() / (area.height() + mx)
        bh = max(area.height() * frac, 30 * s)
        y = area.top() + (area.height() - bh) * (self.scroll.get(key, 0.0) / mx)
        p.fillRect(QRectF(area.right() + 4 * s, area.top(), 4 * s, area.height()), QColor(255, 255, 255, 12))
        p.fillRect(QRectF(area.right() + 4 * s, y, 4 * s, bh), QColor(255, 255, 255, 70))

    # --- screens --------------------------------------------------------------------------------------------------

    def cached(self, key: tuple, make) -> QPixmap:  # noqa: ANN001
        pm = self._pm_cache.get(key)
        if pm is None:
            if len(self._pm_cache) > 24:
                self._pm_cache.clear()
            pm = self._pm_cache[key] = make()
        return pm

    def paint_play(self, p: QPainter, c: QRectF, s: float) -> None:
        gap = 16 * s
        lw = c.width() * 0.62
        hero = QRectF(c.left(), c.top(), lw, c.height() * 0.6)
        pm = self.cached(("arena", int(hero.width()), int(hero.height()), A.dpr()),
                         lambda: scene_preview(int(hero.width()), int(hero.height())))
        p.drawPixmap(hero.topLeft(), pm)
        g = QLinearGradient(hero.topLeft(), hero.bottomLeft())
        g.setColorAt(0.35, QColor(8, 10, 14, 0))
        g.setColorAt(1.0, QColor(8, 10, 14, 235))
        p.fillRect(hero, g)
        p.setPen(QPen(U.BORDER, 1))
        p.drawRect(hero)
        U.text(p, QRectF(hero.left() + 22 * s, hero.bottom() - 120 * s, lw, 46 * s), "AIM ARENA", 40 * s, U.TEXT,
               weight="bold", cond=True, shadow=True)
        U.text(p, QRectF(hero.left() + 22 * s, hero.bottom() - 76 * s, lw * 0.62, 22 * s),
               "Bots pop up from cover. Headshots pay more; combos multiply the bounty.", 14 * s, U.DIM)
        sk, _seed, _f, it = self.equipped_gun()
        U.text(p, QRectF(hero.left() + 22 * s, hero.bottom() - 48 * s, lw * 0.6, 22 * s),
               f"WEAPON · {(it.full_name if it else 'USP-S (stock)').upper()}", 13 * s, U.ORANGE if it and it.counter
               else U.FAINT, cond=True)
        U.button(p, self.hits, QRectF(hero.right() - 200 * s, hero.bottom() - 74 * s, 180 * s, 54 * s), "GO",
                 self.start_arena, s, "primary", size=26)
        U.button(p, self.hits, QRectF(hero.right() - 200 * s, hero.top() + 16 * s, 180 * s, 34 * s), "Controls…",
                 self.open_controls, s, "default", size=14)
        U.button(p, self.hits, QRectF(hero.right() - 200 * s, hero.top() + 58 * s, 180 * s, 34 * s), "Surf maps ›",
                 lambda: self.set_screen("surf"), s, "blue", size=14, sound="tab")
        # daily bonus and featured case
        y2 = hero.bottom() + gap
        bw = (lw - gap) / 2
        daily = QRectF(c.left(), y2, bw, c.bottom() - y2)
        body = U.panel(p, daily, "Daily bonus", s)
        claim = self.can_claim_daily()
        streak = self.streak if not claim else (self.streak + 1 if self.last_daily == (
            self.today() - _dt.timedelta(days=1)).isoformat() else 1)
        amount = DAILY_BASE + 50 * min(streak - 1, 6)
        U.text(p, QRectF(body.left() + 16 * s, body.top() + 8 * s, bw, 40 * s), D.money(amount) + " + A CASE",
               30 * s, U.GOLD if claim else U.FAINT, weight="bold", cond=True)
        U.text(p, QRectF(body.left() + 16 * s, body.top() + 48 * s, bw - 32 * s, 20 * s),
               f"Day {streak} streak · log in every day for up to +$3.00 more", 13 * s, U.DIM)
        bh = min(44 * s, max(body.height() - 86 * s, 24 * s))
        U.button(p, self.hits, QRectF(body.left() + 16 * s, body.bottom() - bh - 12 * s, bw - 32 * s, bh),
                 "Claim" if claim else "Come back tomorrow", self.claim_daily, s, "gold", claim, 17)
        feat = QRectF(daily.right() + gap, y2, bw, c.bottom() - y2)
        body = U.panel(p, feat, "Featured case", s)
        cid = D.CASE_ORDER[self.day() % len(D.CASE_ORDER)]
        ch = body.height() - 20 * s
        cr = QRectF(body.left() + 8 * s, body.top() + 8 * s, min(ch * 1.3, bw * 0.45), ch)
        p.drawPixmap(cr.topLeft(), A.case_pixmap(cid, int(cr.width()), int(cr.height())))
        tx = cr.right() + 10 * s
        U.text(p, QRectF(tx, body.top() + 10 * s, feat.right() - tx - 10 * s, 26 * s), D.CASES[cid].name.upper(),
               19 * s, U.TEXT, weight="bold", cond=True)
        U.text(p, QRectF(tx, body.top() + 38 * s, feat.right() - tx, 20 * s),
               f"{D.money(D.case_price(cid, self.day()))} · own {self.cases.get(cid, 0)}", 14 * s,
               U.DIM)
        U.button(p, self.hits, QRectF(tx, body.bottom() - bh - 12 * s, feat.right() - tx - 14 * s, bh), "View",
                 lambda: self.open_case_modal(cid), s, "blue", size=16)
        # right column: profile, loadout, recent drops
        rx = c.left() + lw + gap
        rw = c.right() - rx
        prof = QRectF(rx, c.top(), rw, c.height() * 0.3)
        body = U.panel(p, prof, "Profile", s)
        lvl, into, need = D.level_of(self.xp)
        rad = min(body.height() * 0.34, 40 * s)
        self.paint_rank(p, QPointF(body.left() + 20 * s + rad, body.top() + body.height() * 0.42), rad, lvl,
                        into / need)
        tx = body.left() + 36 * s + rad * 2
        U.text(p, QRectF(tx, body.top() + 8 * s, rw, 28 * s), D.rank_name(lvl).upper(), 21 * s, U.TEXT,
               weight="bold", cond=True)
        U.text(p, QRectF(tx, body.top() + 36 * s, rw, 20 * s), f"LEVEL {lvl} · {into} / {need} XP", 13 * s, U.DIM,
               cond=True)
        U.bar(p, QRectF(tx, body.top() + 60 * s, prof.right() - tx - 16 * s, 5 * s), into / need, U.BLUE)
        U.text(p, QRectF(body.left() + 16 * s, body.bottom() - 30 * s, rw - 32 * s, 24 * s),
               f"{len(self.items)} ITEMS · {sum(self.cases.values())} CASES · {self.keys_owned} "
               f"KEY{'S' if self.keys_owned != 1 else ''} · "
               f"{len(self.achievements)}/{len(D.ACHIEVEMENTS)} ACHIEVEMENTS", 12 * s, U.FAINT, cond=True)
        slot_w = (rw - 4 * 12 * s) / 3
        load = QRectF(rx, prof.bottom() + gap, rw, min(30 * s + 20 * s + slot_w * 0.8, c.height() * 0.34))
        body = U.panel(p, load, "Loadout", s)
        sw = (body.width() - 4 * 12 * s) / 3
        for i, (slot, label) in enumerate((("gun", "Weapon"), ("knife", "Knife"), ("gloves", "Gloves"))):
            r = QRectF(body.left() + 12 * s + i * (sw + 12 * s), body.top() + 10 * s, sw,
                       min(sw * 0.8, body.height() - 20 * s))
            it = self.item_by_uid(self.loadout[slot])
            if it:
                self.paint_tile(p, r, it, s, self.hits.hover(r))
                self.hits.add(r, lambda it=it: self.inspect(it))
            else:
                p.fillRect(r, QColor(255, 255, 255, 8))
                p.setPen(QPen(QColor(255, 255, 255, 30), 1, Qt.PenStyle.DashLine))
                p.drawRect(r)
                U.text(p, r, label.upper(), 13 * s, U.FAINT, AL.AlignHCenter, cond=True)
                self.hits.add(r, lambda: self.set_screen("inventory"))
        rec = QRectF(rx, load.bottom() + gap, rw, c.bottom() - load.bottom() - gap)
        body = U.panel(p, rec, "Recent drops", s)
        rh = 30 * s
        recent = sorted(self.items, key=lambda i: -i.uid)[:max(int((body.height() - 8 * s) // rh), 0)]
        if not recent:
            U.text(p, body, "Open a case to see your drops here", 14 * s, U.FAINT, AL.AlignHCenter)
        for k, it in enumerate(recent):
            r = QRectF(body.left() + 12 * s, body.top() + 6 * s + k * rh, body.width() - 24 * s, rh - 4 * s)
            p.fillRect(QRectF(r.left(), r.top() + 3 * s, 3 * s, r.height() - 6 * s), U.rarity_color(it.rarity))
            U.text(p, r.adjusted(10 * s, 0, -80 * s, 0), it.full_name, 14 * s,
                   U.ORANGE if it.counter else U.TEXT)
            U.text(p, r, D.money(self.value(it)), 14 * s, U.MONEY, AL.AlignRight)
            if self.hits.hover(r):
                p.fillRect(r, QColor(255, 255, 255, 12))
            self.hits.add(r, lambda it=it: self.inspect(it))

    def inventory_view(self) -> list[D.Item]:
        f = self.inv_filter
        items = [i for i in self.items if f == "all" or (f == "special" and i.skin.special)
                 or (f != "special" and i.skin.kind == f)]
        if self.inv_sort == "rarity":
            items.sort(key=lambda i: (-i.rarity, -self.value(i)))
        elif self.inv_sort == "value":
            items.sort(key=lambda i: -self.value(i))
        elif self.inv_sort == "name":
            items.sort(key=lambda i: (i.skin.weapon_name, i.skin.name, i.float))
        else:
            items.sort(key=lambda i: -i.uid)
        return items

    def cycle_sort(self) -> None:
        keys = [k for k, _ in SORTS]
        self.inv_sort = keys[(keys.index(self.inv_sort) + 1) % len(keys)]

    def set_filter(self, f: str) -> None:
        self.inv_filter = f
        self.scroll["inventory"] = 0.0

    def toggle_select_mode(self) -> None:
        self.select_mode = not self.select_mode
        self.selected.clear()

    def paint_inventory(self, p: QPainter, c: QRectF, s: float) -> None:
        bar_h = 38 * s
        x = c.left()
        for key, label in FILTERS:
            cw = U.text_width(label.upper(), 13 * s, cond=True) + 26 * s
            U.chip(p, self.hits, QRectF(x, c.top(), cw, bar_h), label, self.inv_filter == key,
                   lambda k=key: self.set_filter(k), s)
            x += cw + 2 * s
        sort_label = dict(SORTS)[self.inv_sort]
        bw = 150 * s
        U.button(p, self.hits, QRectF(c.right() - bw, c.top() + 3 * s, bw, bar_h - 6 * s), f"Sort: {sort_label} ▾",
                 self.cycle_sort, s, "ghost", size=13)
        U.button(p, self.hits, QRectF(c.right() - bw * 2 - 8 * s, c.top() + 3 * s, bw, bar_h - 6 * s),
                 "Done" if self.select_mode else "Select items", self.toggle_select_mode, s,
                 "blue" if self.select_mode else "ghost", size=13)
        items = self.inventory_view()
        total = sum(self.value(i) for i in items)
        info_y = c.top() + bar_h + 6 * s
        U.text(p, QRectF(c.left(), info_y, c.width(), 22 * s),
               f"{len(items)} ITEMS · INVENTORY VALUE {D.money(total)}", 13 * s, U.FAINT, cond=True)
        foot = 54 * s if self.select_mode else 0.0
        area = QRectF(c.left(), info_y + 28 * s, c.width() - 12 * s, c.bottom() - info_y - 28 * s - foot)
        if not items:
            U.text(p, area, "No items here yet - open a case in the CASES tab", 18 * s, U.FAINT, AL.AlignHCenter)
        p.save()
        p.setClipRect(area.adjusted(-2, -2, 2, 2))
        _cols, vis = self.grid(area, len(items), 176 * s, 0.78, 10 * s, "inventory")
        for i, r in vis:
            it = items[i]
            hov = self.hits.hover(r) and area.contains(self.hits.mouse)
            self.paint_tile(p, r, it, s, hov, it.uid in self.selected)
            if area.intersects(r):
                vr = r.intersected(area)
                if self.select_mode:
                    self.hits.add(vr, lambda it=it: self.toggle_select(it), "slot")
                else:
                    self.hits.add(vr, lambda it=it: self.inspect(it), "click", "tile")
        p.restore()
        self.scrollbar(p, area, "inventory", s)
        if self.select_mode:
            fr = QRectF(c.left(), c.bottom() - foot + 8 * s, c.width(), foot - 8 * s)
            p.fillRect(fr, QColor(12, 16, 21, 220))
            sel = [i for i in self.items if i.uid in self.selected]
            net = sum(D.sale_proceeds(self.value(i)) for i in sel)
            U.text(p, fr.adjusted(16 * s, 0, 0, 0), f"{len(sel)} SELECTED · YOU RECEIVE {D.money(net)} AFTER THE 15% "
                   "MARKET FEE", 15 * s, U.TEXT, cond=True)
            U.button(p, self.hits, QRectF(fr.right() - 220 * s, fr.top() + 7 * s, 210 * s, fr.height() - 14 * s),
                     f"Sell {len(sel)} items", self.sell_selected, s, "primary", bool(sel), 15)
            U.button(p, self.hits, QRectF(fr.right() - 440 * s, fr.top() + 7 * s, 100 * s, fr.height() - 14 * s),
                     "Clear", self.selected.clear, s, "ghost", bool(sel), 14)
            U.button(p, self.hits, QRectF(fr.right() - 330 * s, fr.top() + 7 * s, 100 * s, fr.height() - 14 * s),
                     "All", lambda: self.selected.update(i.uid for i in items), s, "ghost", bool(items), 14)

    def paint_market(self, p: QPainter, c: QRectF, s: float) -> None:
        gap = 16 * s
        lw = c.width() * 0.58
        left = QRectF(c.left(), c.top(), lw, c.height())
        body = U.panel(p, left, "Sell on the community market · 15% fee", s)
        rows = sorted(self.items, key=lambda i: -self.value(i))
        rh = 56 * s
        area = body.adjusted(8 * s, 8 * s, -16 * s, -8 * s)
        total = len(rows) * rh
        self.scroll_max["market"] = max(0.0, total - area.height())
        off = min(self.scroll.get("market", 0.0), self.scroll_max["market"])
        self.scroll["market"] = off
        if not rows:
            U.text(p, area, "Nothing to sell yet", 16 * s, U.FAINT, AL.AlignHCenter)
        p.save()
        p.setClipRect(area)
        day = self.day()
        first = int(off // rh)
        for k in range(first, min(len(rows), first + int(area.height() // rh) + 2)):
            it = rows[k]
            r = QRectF(area.left(), area.top() + k * rh - off, area.width(), rh - 4 * s)
            key = ("row", it.uid, it.kills, day, int(r.width()), int(r.height()), A.dpr())
            pm = self._tiles.get(key)
            if pm is None:
                if len(self._tiles) > 360:
                    self._tiles.clear()
                pm = self._tiles[key] = self.render_market_row(it, int(r.width()), int(r.height()), s, day)
            p.drawPixmap(r.topLeft(), pm)
            if self.hits.hover(r):
                p.fillRect(r, QColor(255, 255, 255, 10))
            if area.intersects(r):
                U.button(p, self.hits, QRectF(r.right() - 96 * s, r.top() + 10 * s, 86 * s, r.height() - 20 * s),
                         "Sell", lambda it=it: self.sell_one(it), s, "primary", size=14, sound="")
                self.hits.add(QRectF(r.left(), r.top(), r.width() - 100 * s, r.height()).intersected(area),
                              lambda it=it: self.inspect(it))
        p.restore()
        self.scrollbar(p, area, "market", s)
        self.paint_store(p, c, left, s, gap, day)

    def render_market_row(self, it: D.Item, w: int, h: int, s: float, day: int) -> QPixmap:
        pm = A.new_pixmap(max(w, 1), max(h, 1))
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        r = QRectF(0, 0, w, h)
        p.fillRect(r, QColor(255, 255, 255, 7))
        p.fillRect(QRectF(0, 0, 3 * s, h), U.rarity_color(it.rarity))
        ar = QRectF(10 * s, 3 * s, 78 * s, h - 6 * s)
        p.drawPixmap(ar.topLeft(), A.art_pixmap(it.skin_id, it.pattern, it.float, int(ar.width()), int(ar.height())))
        nx = ar.right() + 10 * s
        price_x = w - 220 * s
        namew = max(price_x - nx - w * 0.15, 40 * s)
        U.text(p, QRectF(nx, 4 * s, namew, 22 * s), it.full_name, 15 * s,
               U.ORANGE if it.counter else U.rarity_color(it.rarity) if it.skin.special else U.TEXT)
        note = D.pattern_note(it.skin, it.pattern)
        U.text(p, QRectF(nx, 26 * s, namew, 20 * s),
               f"{D.RARITY_SHORT[it.rarity]} · float {it.float:.4f}" + (f" · {note}" if note else ""), 12 * s, U.DIM)
        hist = D.price_history(it, 21, day)
        spark = QRectF(nx + namew + 8 * s, 10 * s, price_x - nx - namew - 16 * s, h - 20 * s)
        U.sparkline(p, spark, hist, U.MONEY if hist[-1] >= hist[0] else U.RED)
        price = D.item_value(it, day)
        U.text(p, QRectF(price_x, 4 * s, 110 * s, 22 * s), D.money(price), 16 * s, U.TEXT, AL.AlignRight, "bold")
        U.text(p, QRectF(price_x, 26 * s, 110 * s, 20 * s), f"you get {D.money(D.sale_proceeds(price))}", 11 * s,
               U.MONEY, AL.AlignRight)
        p.end()
        return pm

    def paint_store(self, p: QPainter, c: QRectF, left: QRectF, s: float, gap: float, day: int) -> None:
        rx = left.right() + gap
        rw = c.right() - rx
        store = QRectF(rx, c.top(), rw, c.height() * 0.56)
        body = U.panel(p, store, "Store", s)
        n = len(D.CASE_ORDER) + 1
        rh2 = min((body.height() - 12 * s) / n, 70 * s)
        for k in range(n):
            r = QRectF(body.left() + 10 * s, body.top() + 6 * s + k * rh2, body.width() - 20 * s, rh2 - 6 * s)
            p.fillRect(r, QColor(255, 255, 255, 7))
            ir = QRectF(r.left() + 4 * s, r.top() + 2 * s, r.height() * 1.25, r.height() - 4 * s)
            if k < len(D.CASE_ORDER):
                cid = D.CASE_ORDER[k]
                p.drawPixmap(ir.topLeft(), A.case_pixmap(cid, int(ir.width()), int(ir.height())))
                name, price, own = D.CASES[cid].name, D.case_price(cid, day), self.cases.get(cid, 0)
                buy1, buy5 = (lambda c=cid: self.buy_case(c)), (lambda c=cid: self.buy_case(c, 5))
            else:
                p.drawPixmap(ir.topLeft(), A.key_pixmap(int(ir.width()), int(ir.height())))
                name, price, own = "Case Key", D.KEY_PRICE, self.keys_owned
                buy1, buy5 = (lambda: self.buy_key()), (lambda: self.buy_key(5))
            tx = ir.right() + 10 * s
            U.text(p, QRectF(tx, r.top(), r.width() * 0.5, r.height() * 0.55), name.upper(), 15 * s, U.TEXT,
                   weight="bold", cond=True)
            U.text(p, QRectF(tx, r.top() + r.height() * 0.5, r.width() * 0.5, r.height() * 0.45),
                   f"{D.money(price)} · own {own}", 12 * s, U.DIM)
            bwid = 62 * s
            bh = min(30 * s, r.height() - 10 * s)
            by = r.center().y() - bh / 2
            U.button(p, self.hits, QRectF(r.right() - bwid * 2 - 12 * s, by, bwid, bh), "Buy", buy1, s, "blue",
                     self.wallet >= price, 13, "")
            U.button(p, self.hits, QRectF(r.right() - bwid - 6 * s, by, bwid, bh), "×5", buy5, s, "blue",
                     self.wallet >= price * 5, 13, "")
        act = QRectF(rx, store.bottom() + gap, rw, c.bottom() - store.bottom() - gap)
        body = U.panel(p, act, "Recent activity", s)
        now = time.time()
        rh3 = 24 * s
        entries = [(n_, cents, t, None, True) for n_, cents, t in self.sales[:4]]
        entries += [(n_, cents, t, r_, False) for n_, cents, t, r_ in self.feed]
        for k, (nm, cents, t, rar, mine) in enumerate(entries[:max(int((body.height() - 8 * s) // rh3), 0)]):
            r = QRectF(body.left() + 12 * s, body.top() + 4 * s + k * rh3, body.width() - 24 * s, rh3)
            ago = max(0, int(now - t))
            when = f"{ago}s" if ago < 60 else f"{ago // 60}m" if ago < 3600 else f"{ago // 3600}h"
            col = U.MONEY if mine else (U.rarity_color(rar) if rar is not None and rar >= D.CLASSIFIED else U.DIM)
            U.text(p, r.adjusted(0, 0, -120 * s, 0), ("You sold " if mine else "") + nm, 12 * s, col)
            U.text(p, r.adjusted(0, 0, -44 * s, 0), D.money(cents), 12 * s, U.TEXT, AL.AlignRight)
            U.text(p, r, when, 11 * s, U.FAINT, AL.AlignRight)

    def paint_tradeup(self, p: QPainter, c: QRectF, s: float) -> None:
        gap = 14 * s
        top = QRectF(c.left(), c.top(), c.width(), min(c.height() * 0.5, 330 * s))
        body = U.panel(p, top, "Trade up contract · ten items of one rarity → one of the next", s)
        items = self.contract_items()
        sw = (body.width() - 24 * s - 9 * 8 * s) / 10
        sh = min(sw * 0.78, body.height() * 0.44)
        sw = min(sw, sh / 0.78)
        x0 = body.left() + (body.width() - (sw * 10 + 72 * s)) / 2
        for k in range(10):
            r = QRectF(x0 + k * (sw + 8 * s), body.top() + 12 * s, sw, sh)
            if k < len(items):
                it = items[k]
                self.paint_tile(p, r, it, s, self.hits.hover(r))
                self.hits.add(r, lambda it=it: self.contract_toggle(it), "")
            else:
                p.fillRect(r, QColor(255, 255, 255, 6))
                p.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.PenStyle.DashLine))
                p.drawRect(r)
                U.text(p, r, str(k + 1), 22 * s, U.FAINT, AL.AlignHCenter, "bold")
        y = body.top() + 22 * s + sh
        outs = D.trade_outcomes(items)
        if items:
            r0 = items[0].rarity
            avg = sum(i.float for i in items) / len(items)
            info = (f"{D.RARITY_SHORT[r0].upper()} → {D.RARITY_SHORT[r0 + 1].upper()}"
                    + (f" · {D.COUNTER_NAME.upper()}" if items[0].counter else "")
                    + f" · AVERAGE FLOAT {avg:.4f} · {len(items)}/10")
        else:
            info = "PICK ITEMS BELOW: MIL-SPEC, RESTRICTED OR CLASSIFIED, ALL KILLTALLY OR NONE, NO KNIVES OR GLOVES"
        U.text(p, QRectF(body.left() + 16 * s, y, body.width() * 0.6, 22 * s), info, 14 * s, U.DIM, cond=True)
        oy = y + 26 * s
        oh = body.bottom() - oy - 10 * s
        if outs and oh > 20 * s:
            ow = min(oh / 0.78, 120 * s)
            maxn = max(int((body.width() * 0.64) // (ow + 6 * s)), 1)
            for k, (sk, pr) in enumerate(sorted(outs, key=lambda o: -o[1])[:maxn]):
                r = QRectF(body.left() + 16 * s + k * (ow + 6 * s), oy, ow, min(oh, ow * 0.78))
                fake = D.Item(-1, sk.id, D.trade_float(items, sk), 17)
                self.paint_tile(p, r, fake, s * 0.8)
                U.text(p, QRectF(r.left(), r.top() + 2 * s, r.width() - 4 * s, 16 * s), f"{pr * 100:.1f}%",
                       11 * s, U.GOLD, AL.AlignRight, "bold")
                U.text(p, QRectF(r.left() + 4 * s, r.top() + 2 * s, r.width(), 16 * s),
                       D.WEARS[D.wear_index(fake.float)][1], 11 * s, U.DIM)
        bx = body.right() - 250 * s
        bh = min(46 * s, max(oh * 0.45, 26 * s))
        U.button(p, self.hits, QRectF(bx, oy, 234 * s, bh), "Submit contract", self.submit_contract, s, "gold",
                 len(items) == 10, 17, "")
        U.button(p, self.hits, QRectF(bx, oy + bh + 8 * s, 113 * s, bh * 0.8), "Auto-fill", self.contract_autofill, s,
                 "ghost", len(items) < 10, 14, "")
        U.button(p, self.hits, QRectF(bx + 121 * s, oy + bh + 8 * s, 113 * s, bh * 0.8), "Clear", self.contract.clear,
                 s, "ghost", bool(items), 14)
        # candidates
        low = QRectF(c.left(), top.bottom() + gap, c.width(), c.bottom() - top.bottom() - gap)
        body = U.panel(p, low, "Your eligible items", s)
        cands = sorted((i for i in self.items if i.rarity in D.TRADE_RARITIES and not i.skin.special
                        and i.uid not in self.contract), key=lambda i: (i.rarity, i.counter, self.value(i)))
        area = body.adjusted(12 * s, 10 * s, -20 * s, -10 * s)
        if not cands:
            U.text(p, area, "Open cases to collect Mil-Spec, Restricted and Classified items", 16 * s, U.FAINT,
                   AL.AlignHCenter)
        p.save()
        p.setClipRect(area.adjusted(-2, -2, 2, 2))
        _cols, vis = self.grid(area, len(cands), 150 * s, 0.78, 8 * s, "tradeup")
        for i, r in vis:
            it = cands[i]
            fits = self.contract_fits(it)
            self.paint_tile(p, r, it, s, fits and self.hits.hover(r) and area.contains(self.hits.mouse), dim=not fits)
            if area.intersects(r):
                self.hits.add(r.intersected(area), lambda it=it: self.contract_toggle(it), "")
        p.restore()
        self.scrollbar(p, area, "tradeup", s)

    def paint_cases(self, p: QPainter, c: QRectF, s: float) -> None:
        head = QRectF(c.left(), c.top(), c.width(), 40 * s)
        U.text(p, head, "WEAPON CASES", 24 * s, U.TEXT, weight="bold", cond=True)
        kx = c.right() - 470 * s
        p.drawPixmap(QRectF(kx, head.top() + 4 * s, 44 * s, 32 * s).topLeft(), A.key_pixmap(int(44 * s), int(32 * s)))
        U.text(p, QRectF(kx + 50 * s, head.top(), 110 * s, head.height()), f"{self.keys_owned} KEYS", 16 * s, U.TEXT,
               cond=True)
        U.button(p, self.hits, QRectF(kx + 140 * s, head.top() + 4 * s, 150 * s, 32 * s),
                 f"Buy key {D.money(D.KEY_PRICE)}", self.buy_key, s, "blue", self.wallet >= D.KEY_PRICE, 14, "")
        quick = self.settings["quick"]
        U.button(p, self.hits, QRectF(c.right() - 170 * s, head.top() + 4 * s, 170 * s, 32 * s),
                 f"Quick open: {'on' if quick else 'off'}", self.toggle_quick, s, "ghost", size=14)
        U.button(p, self.hits, QRectF(kx - 150 * s, head.top() + 4 * s, 130 * s, 32 * s), "Controls…",
                 self.open_controls, s, "ghost", size=14)
        area = QRectF(c.left(), head.bottom() + 14 * s, c.width(), c.bottom() - head.bottom() - 14 * s)
        n = len(D.CASE_ORDER)
        cols = 4 if area.width() / area.height() > 1.6 else 2
        rows = (n + cols - 1) // cols
        gap = 16 * s
        cw = (area.width() - gap * (cols - 1)) / cols
        ch = (area.height() - gap * (rows - 1)) / rows
        day = self.day()
        for k, cid in enumerate(D.CASE_ORDER):
            r = QRectF(area.left() + (k % cols) * (cw + gap), area.top() + (k // cols) * (ch + gap), cw, ch)
            hov = self.hits.hover(r)
            case = D.CASES[cid]
            g = QLinearGradient(r.topLeft(), r.bottomLeft())
            g.setColorAt(0.0, QColor(46, 55, 66, 235 if hov else 215))
            g.setColorAt(1.0, QColor(20, 25, 31, 235))
            p.fillRect(r, g)
            glow = QRadialGradient(QPointF(r.center().x(), r.top() + ch * 0.4), cw * 0.6)
            gc = QColor(case.colors[0])
            gc.setAlpha(70 if hov else 40)
            glow.setColorAt(0.0, gc)
            glow.setColorAt(1.0, QColor(gc.red(), gc.green(), gc.blue(), 0))
            p.fillRect(r, glow)
            p.setPen(QPen(QColor(255, 255, 255, 80 if hov else 30), 1))
            p.drawRect(r)
            btn_h = min(40 * s, ch * 0.1)
            art_h = ch * 0.42
            aw = min(cw * 0.86, art_h * 1.35)
            ah = aw / 1.35
            ar = QRectF(r.center().x() - aw / 2, r.top() + 10 * s + (art_h - ah) / 2 - (4 * s if hov else 0), aw, ah)
            if aw > 10:
                p.drawPixmap(ar.topLeft(), A.case_pixmap(cid, int(aw), int(ah)))
            ty = r.top() + 10 * s + art_h
            U.text(p, QRectF(r.left() + 12 * s, ty, r.width() - 24 * s, 28 * s), case.name.upper(), 19 * s, U.TEXT,
                   AL.AlignHCenter, "bold", True)
            U.text(p, QRectF(r.left() + 12 * s, ty + 28 * s, r.width() - 24 * s, 20 * s),
                   f"{D.money(D.case_price(cid, day))} · you own {self.cases.get(cid, 0)}", 13 * s, U.DIM,
                   AL.AlignHCenter)
            # the top of the collection: its coverts and the gold tile
            by = r.bottom() - btn_h - 14 * s
            top_y = ty + 56 * s
            strip_h = by - top_y - 10 * s
            tops = case.tier(D.COVERT)[:2]
            n_t = len(tops) + 1
            tw = min((r.width() - 28 * s - 6 * s * (n_t - 1)) / n_t, strip_h / 0.78)
            th = tw * 0.78
            if th > 24 * s:
                x0 = r.center().x() - (tw * n_t + 6 * s * (n_t - 1)) / 2
                yy = top_y + (strip_h - th) / 2
                for j, sk in enumerate(tops):
                    self.paint_tile(p, QRectF(x0 + j * (tw + 6 * s), yy, tw, th),
                                    D.Item(-1, sk.id, (sk.fmin + sk.fmax) / 2, 17), s)
                p.drawPixmap(QPointF(x0 + len(tops) * (tw + 6 * s), yy), A.star_tile(int(tw), int(th)))
            cost = self.open_cost(cid)
            label = "Open" if cost == 0 else f"Buy & open {D.money(cost)}"
            U.button(p, self.hits, QRectF(r.left() + 14 * s, by, r.width() - 28 * s, btn_h),
                     label, lambda c_=cid: self.open_case_modal(c_), s, "primary" if cost == 0 else "blue", size=15)
            self.hits.regions.insert(0, (r, lambda c_=cid: self.open_case_modal(c_), "click", "card " + cid))

    def toggle_quick(self) -> None:
        self.settings["quick"] = not self.settings["quick"]

    def surf_card_picture(self, map_id: str, w: int, h: int) -> QPixmap:
        dpr = self.devicePixelRatioF()

        def make() -> QPixmap:
            pm = surf_preview(self.surf_renderer(), map_id, w, h, dpr)
            return pm if pm is not None else surf_placeholder(w, h, SM.INFO[map_id]["theme_id"])

        return self.cached(("surf", map_id, w, h, dpr), make)

    def paint_surf(self, p: QPainter, c: QRectF, s: float) -> None:
        head = QRectF(c.left(), c.top(), c.width(), 40 * s)
        U.text(p, head, "SURF", 24 * s, U.TEXT, weight="bold", cond=True)
        U.text(p, QRectF(c.left() + 80 * s, c.top(), c.width() - 260 * s, 40 * s),
               "First-person ramp runs. Under par pays cash, under gold pays double, and the first clear of a map "
               "drops its case.", 13 * s, U.DIM)
        U.button(p, self.hits, QRectF(c.right() - 150 * s, head.top() + 4 * s, 150 * s, 32 * s), "Controls…",
                 self.open_controls, s, "ghost", size=14)
        tips_h = 104 * s
        gap = 16 * s
        area = QRectF(c.left(), head.bottom() + 12 * s, c.width(), c.bottom() - head.bottom() - 12 * s - tips_h - gap)
        n = len(SM.MAP_ORDER)
        cw = (area.width() - gap * (n - 1)) / n
        sd = self.surf_data
        gl_ok = self.surf_renderer().available()
        for k, mid in enumerate(SM.MAP_ORDER):
            info = SM.INFO[mid]
            r = QRectF(area.left() + k * (cw + gap), area.top(), cw, area.height())
            hov = self.hits.hover(r)
            p.fillRect(r, QColor(28, 34, 42, 235 if hov else 215))
            ph = min(r.height() * 0.5, cw * 0.62)
            pic = QRectF(r.left(), r.top(), cw, ph)
            p.drawPixmap(pic.topLeft(), self.surf_card_picture(mid, int(cw), int(ph)))
            g = QLinearGradient(pic.topLeft(), pic.bottomLeft())
            g.setColorAt(0.55, QColor(8, 10, 14, 0))
            g.setColorAt(1.0, QColor(8, 10, 14, 220))
            p.fillRect(pic, g)
            tier = f"TIER {info['tier']} · {SURF_TIERS.get(info['tier'], '')}"
            tw = U.text_width(tier, 12 * s, cond=True) + 18 * s
            p.fillRect(QRectF(pic.left() + 10 * s, pic.top() + 10 * s, tw, 22 * s), QColor(0, 0, 0, 160))
            U.text(p, QRectF(pic.left() + 10 * s, pic.top() + 10 * s, tw, 22 * s), tier, 12 * s,
                   (U.MONEY, U.MONEY, U.GOLD, U.RED, U.RED)[min(info["tier"], 5) - 1], AL.AlignHCenter, cond=True)
            U.text(p, QRectF(pic.left() + 14 * s, pic.bottom() - 40 * s, cw - 28 * s, 34 * s), info["name"].upper(),
                   26 * s, U.TEXT, weight="bold", cond=True, shadow=True)
            y = pic.bottom() + 8 * s
            p.save()
            p.setFont(U.F(13 * s, "regular"))
            p.setPen(U.DIM)
            p.drawText(QRectF(r.left() + 14 * s, y, cw - 28 * s, 40 * s),
                       int(AL.AlignLeft | AL.AlignTop | Qt.TextFlag.TextWordWrap), info["blurb"])
            p.restore()
            y += 44 * s
            best = sd["best"].get(mid)
            rows = (("Personal best", surf_fmt(best), U.GOLD if mid in sd["golds"] else U.TEXT),
                    ("Par · pays", f"{surf_fmt(info['par'])} · {D.money(info['reward'])}", U.TEXT),
                    ("Gold · pays", f"{surf_fmt(info['gold'])} · {D.money(info['reward'] * 2)}", U.GOLD),
                    ("First clear", ("✓ " if mid in sd["clears"] else "") + D.CASES[info["case_id"]].name,
                     U.MONEY if mid in sd["clears"] else U.TEXT))
            rh = min(24 * s, max((r.bottom() - y - 56 * s) / len(rows), 12 * s))
            for label, value, col in rows:
                rr = QRectF(r.left() + 14 * s, y, cw - 28 * s, rh)
                U.text(p, rr, label.upper(), 12 * s, U.FAINT, cond=True)
                U.text(p, rr, value, 13 * s, col, AL.AlignRight, "bold")
                y += rh
            U.button(p, self.hits, QRectF(r.left() + 14 * s, r.bottom() - 52 * s, cw - 28 * s, 40 * s),
                     "Ride" if gl_ok else "Needs OpenGL 3.3", lambda m_=mid: self.start_surf(m_), s, "primary",
                     gl_ok, 17)
            p.setPen(QPen(QColor(255, 255, 255, 70 if hov else 28), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
        tips = QRectF(c.left(), c.bottom() - tips_h, c.width(), tips_h)
        body = U.panel(p, tips, "How to surf", s)
        K = Qt.Key  # noqa: N806
        lines = (f"Walk off the platform with {self.key_label(K.Key_W)}. On a ramp, let go of it.",
                 f"Hold the strafe key towards the ramp ({self.key_label(K.Key_A)} / {self.key_label(K.Key_D)}): "
                 "it presses you into the face and you slide without friction.",
                 "Look along the ramp and steer with smooth mouse moves. Riding low turns the fall into speed; "
                 "turning into the ramp brakes.",
                 f"In the air, strafe and turn the same way to curve. A fall takes you back to the stage's platform "
                 f"({self.key_label(K.Key_R)} too); hold {self.key_label(K.Key_Space)} to hop across platforms.")
        tw = (body.width() - 24 * s) / len(lines)
        for k, line in enumerate(lines):
            p.save()
            p.setFont(U.F(12 * s, "regular"))
            p.setPen(U.DIM)
            p.drawText(QRectF(body.left() + 12 * s + k * tw, body.top() + 8 * s, tw - 14 * s, body.height() - 14 * s),
                       int(AL.AlignLeft | AL.AlignTop | Qt.TextFlag.TextWordWrap), f"{k + 1}. {line}")
            p.restore()

    def paint_stats(self, p: QPainter, c: QRectF, s: float) -> None:
        gap = 16 * s
        cw = (c.width() - 2 * gap) / 3
        st = self.stats
        left = QRectF(c.left(), c.top(), cw, c.height())
        body = U.panel(p, left, "Career", s)
        net = st["earned"] - st["spent"]
        acc = st["hits"] / st["shots"] * 100 if st["shots"] else 0
        rows = (("Cases opened", str(st["opened"])), ("Keys bought", str(st["keys"])),
                ("Money spent", D.money(st["spent"])), ("Money earned", D.money(st["earned"])),
                ("Net", D.money(net)), ("Items sold", str(st["sold"])), ("Trade ups", str(st["trade_ups"])),
                (f"{D.COUNTER_NAME} drops", str(st["counter"])), ("Arena rounds", str(st["rounds"])),
                ("Arena kills", f"{st['kills']} ({st['hs']} HS)"), ("Arena accuracy", f"{acc:.0f}%"),
                ("Best round", f"{st['best_round']} kills"), ("Arena earnings", D.money(st["arena_earned"])),
                ("Surf runs finished", f"{st['surf_finishes']} of {st['surf_runs']}"),
                ("Surf earnings", D.money(st["surf_earned"])))
        rh = min((body.height() - 110 * s) / len(rows), 30 * s)
        for k, (a, b) in enumerate(rows):
            r = QRectF(body.left() + 16 * s, body.top() + 8 * s + k * rh, body.width() - 32 * s, rh)
            U.text(p, r, a.upper(), 13 * s, U.DIM, cond=True)
            col = U.money_color(net) if a == "Net" else U.TEXT
            U.text(p, r, b, 15 * s, col, AL.AlignRight, "bold")
        best = st.get("best")
        br = QRectF(body.left() + 16 * s, body.bottom() - 96 * s, body.width() - 32 * s, 84 * s)
        p.fillRect(br, QColor(255, 255, 255, 8))
        U.text(p, QRectF(br.left() + 10 * s, br.top() + 6 * s, br.width(), 20 * s), "BEST DROP", 12 * s, U.FAINT,
               cond=True)
        if best:
            U.text(p, QRectF(br.left() + 10 * s, br.top() + 28 * s, br.width() - 20 * s, 24 * s), best["name"],
                   16 * s, U.rarity_color(int(best.get("rarity", 2)) % 7), weight="bold")
            U.text(p, QRectF(br.left() + 10 * s, br.top() + 54 * s, br.width() - 20 * s, 20 * s),
                   D.money(int(best.get("value", 0))), 14 * s, U.MONEY)
        else:
            U.text(p, QRectF(br.left() + 10 * s, br.top() + 28 * s, br.width(), 24 * s), "—", 16 * s, U.FAINT)
        # rarity histogram
        mid = QRectF(left.right() + gap, c.top(), cw, c.height())
        body = U.panel(p, mid, "Unboxed by rarity", s)
        drops = st["drops"]
        tot = max(sum(drops), 1)
        tiers = (D.MILSPEC, D.RESTRICTED, D.CLASSIFIED, D.COVERT, D.SPECIAL)
        bh = min((body.height() - 40 * s) / len(tiers), 70 * s)
        mx = max(max(drops[r] for r in tiers), 1)
        for k, r in enumerate(tiers):
            y = body.top() + 16 * s + k * bh
            U.text(p, QRectF(body.left() + 16 * s, y, body.width() - 32 * s, 20 * s), D.RARITY_NAMES[r].upper(),
                   13 * s, U.rarity_color(r), cond=True)
            U.text(p, QRectF(body.left() + 16 * s, y, body.width() - 32 * s, 20 * s),
                   f"{drops[r]} · {drops[r] / tot * 100:.1f}% (odds {D.CASE_ODDS[r] * 100:.2f}%)", 12 * s, U.DIM,
                   AL.AlignRight)
            U.bar(p, QRectF(body.left() + 16 * s, y + 24 * s, body.width() - 32 * s, max(bh - 40 * s, 6 * s)),
                  drops[r] / mx, U.rarity_color(r), QColor(255, 255, 255, 14))
        # achievements
        right = QRectF(mid.right() + gap, c.top(), cw, c.height())
        body = U.panel(p, right, f"Achievements {len(self.achievements)}/{len(D.ACHIEVEMENTS)}", s)
        ah = (body.height() - 12 * s) / len(D.ACHIEVEMENTS)
        for k, (aid, name, desc) in enumerate(D.ACHIEVEMENTS):
            r = QRectF(body.left() + 12 * s, body.top() + 6 * s + k * ah, body.width() - 24 * s, ah - 3 * s)
            done = aid in self.achievements
            p.fillRect(r, QColor(228, 174, 57, 26) if done else QColor(255, 255, 255, 6))
            U.text(p, QRectF(r.left() + 8 * s, r.top(), 24 * s, r.height()), "★" if done else "☆", 15 * s,
                   U.GOLD if done else U.FAINT)
            U.text(p, QRectF(r.left() + 32 * s, r.top(), r.width() * 0.4, r.height()), name.upper(), 13 * s,
                   U.TEXT if done else U.DIM, cond=True)
            U.text(p, QRectF(r.left() + r.width() * 0.42, r.top(), r.width() * 0.58 - 6 * s, r.height()), desc,
                   11.5 * s, U.DIM if done else U.FAINT, AL.AlignRight)

    # --- modals ---------------------------------------------------------------------------------------------------

    def modal_rect(self, w: int, h: int, s: float, fw: float = 0.86, fh: float = 0.86) -> QRectF:
        mw, mh = min(w * fw, 1300 * s), min(h * fh, 820 * s)
        return QRectF((w - mw) / 2, (h - mh) / 2, mw, mh)

    def paint_contents(self, p: QPainter, area: QRectF, case: D.Case, s: float, dim: bool) -> None:
        """'Contains one of the following' - the case's skins by rarity and the gold ★ tile."""
        guns = sorted((sk for sk in case.skins if not sk.special), key=lambda sk: (-sk.rarity, sk.weapon))
        n = len(guns) + 1
        cols = max(4, min(8, int(area.width() // (150 * s))))
        rows = (n + cols - 1) // cols
        gap = 8 * s
        tw = (area.width() - gap * (cols - 1)) / cols
        th = min(tw * 0.78, (area.height() - gap * (rows - 1)) / rows)
        tw = min(tw, th / 0.66)
        x0 = area.left() + (area.width() - (tw * cols + gap * (cols - 1))) / 2
        for k in range(n):
            r = QRectF(x0 + (k % cols) * (tw + gap), area.top() + (k // cols) * (th + gap), tw, th)
            if k < len(guns):
                sk = guns[k]
                self.paint_tile(p, r, D.Item(-1, sk.id, (sk.fmin + sk.fmax) / 2, 17), s,
                                not dim and self.hits.hover(r))
            else:
                p.drawPixmap(r.topLeft(), A.star_tile(int(tw), int(th)))
                lab = "★ Exceedingly Rare Glove" if case.has_gloves else "★ Rare Special Item"
                U.text(p, QRectF(r.left(), r.bottom() - th * 0.3, r.width(), th * 0.22), lab, max(th * 0.1, 9),
                       U.GOLD, AL.AlignHCenter, cond=True)
        if dim:
            p.fillRect(area.adjusted(-4, -4, 4, 4), QColor(10, 12, 16, 150))

    def draw_modal_case(self, p: QPainter, w: int, h: int, s: float) -> None:
        r = self.modal_rect(w, h, s)
        case = D.CASES[self.modal_case]
        body = U.panel(p, r, "Unlock container", s, 245)
        U.close_x(p, self.hits, QRectF(r.right() - 30 * s, r.top(), 30 * s, 30 * s), self.close_modal, s)
        U.text(p, QRectF(body.left() + 24 * s, body.top() + 10 * s, body.width(), 34 * s), case.name.upper(), 28 * s,
               U.TEXT, weight="bold", cond=True)
        U.text(p, QRectF(body.left() + 24 * s, body.top() + 44 * s, body.width(), 20 * s),
               "CONTAINS ONE OF THE FOLLOWING:", 13 * s, U.DIM, cond=True)
        foot = 96 * s
        area = QRectF(body.left() + 24 * s, body.top() + 72 * s, body.width() - 48 * s,
                      body.height() - 72 * s - foot - 10 * s)
        self.paint_contents(p, area, case, s, False)
        fr = QRectF(r.left(), r.bottom() - foot, r.width(), foot)
        p.fillRect(fr, QColor(0, 0, 0, 90))
        p.fillRect(QRectF(fr.left(), fr.top(), fr.width(), 1), QColor(255, 255, 255, 30))
        ch = foot - 24 * s
        p.drawPixmap(QPointF(fr.left() + 24 * s, fr.top() + 12 * s), A.case_pixmap(case.id, int(ch * 1.35), int(ch)))
        kx = fr.left() + 36 * s + ch * 1.35
        p.drawPixmap(QPointF(kx, fr.top() + 12 * s + ch * 0.2), A.key_pixmap(int(ch * 0.9), int(ch * 0.6)))
        tx = kx + ch + 10 * s
        own = self.cases.get(case.id, 0)
        U.text(p, QRectF(tx, fr.top() + 12 * s, fr.width() * 0.45, 24 * s),
               f"YOU OWN {own} · {self.keys_owned} KEY{'S' if self.keys_owned != 1 else ''} · "
               f"UNLOCK WITH A CASE KEY ({D.money(D.KEY_PRICE)})", 16 * s, U.TEXT, cond=True)
        U.text(p, QRectF(tx, fr.top() + 38 * s, fr.width() * 0.45, 20 * s),
               "Mil-Spec 79.92% · Restricted 15.98% · Classified 3.20% · Covert 0.64% · ★ 0.26%", 12 * s, U.DIM)
        U.text(p, QRectF(tx, fr.top() + 58 * s, fr.width() * 0.45, 20 * s),
               f"{D.COUNTER_NAME} kill counter on 1 in 10 drops (not gloves)", 12 * s, U.ORANGE)
        cost = self.open_cost(case.id)
        label = "Unlock container" if cost == 0 else f"Buy & unlock · {D.money(cost)}"
        U.button(p, self.hits, QRectF(fr.right() - 300 * s, fr.top() + 22 * s, 276 * s, foot - 44 * s), label,
                 lambda: self.open_case(case.id), s, "primary", self.wallet >= cost, 18, "")

    def draw_modal_reel(self, p: QPainter, w: int, h: int, s: float) -> None:
        reel = self.reel
        if reel is None:
            self.modal = None
            return
        r = self.modal_rect(w, h, s)
        case = D.CASES[reel.case_id]
        body = U.panel(p, r, f"Unlocking · {case.name}", s, 245)
        strip_h = min(body.height() * 0.36, 210 * s)
        strip = QRectF(r.left(), body.top() + 30 * s, r.width(), strip_h)
        p.fillRect(strip, QColor(8, 10, 13, 230))
        tw = min(w * 0.14, 200 * s, strip_h * 1.25)
        th = tw * 0.78
        step = tw + 5 * s
        cx = strip.center().x()
        pos = reel.pos()
        p.save()
        p.setClipRect(strip)
        first = max(0, int(pos - cx / step) - 2)
        last = min(len(reel.tiles), int(pos + (strip.width() - cx + strip.left()) / step) + 3)
        ty = strip.center().y() - th / 2
        for i in range(first, last):
            x = cx + (i - pos) * step
            self.paint_tile(p, QRectF(x, ty, tw, th), reel.tiles[i], s, hidden=True)
        # edge fades
        for left_side in (True, False):
            g = QLinearGradient(strip.left() if left_side else strip.right(), 0,
                                strip.left() + strip.width() * 0.22 if left_side else strip.right() -
                                strip.width() * 0.22, 0)
            g.setColorAt(0.0, QColor(8, 10, 13, 245))
            g.setColorAt(1.0, QColor(8, 10, 13, 0))
            p.fillRect(strip, g)
        p.restore()
        # the yellow centre marker
        p.fillRect(QRectF(cx - 1.5 * s, strip.top() - 6 * s, 3 * s, strip.height() + 12 * s), QColor("#f5d142"))
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor("#f5d142"))
        p.setPen(Qt.PenStyle.NoPen)
        for y, d in ((strip.top() - 6 * s, 1), (strip.bottom() + 6 * s, -1)):
            tri = QPainterPath(QPointF(cx - 9 * s, y))
            tri.lineTo(cx + 9 * s, y)
            tri.lineTo(cx, y + d * 11 * s)
            tri.closeSubpath()
            p.drawPath(tri)
        p.restore()
        area = QRectF(body.left() + 24 * s, strip.bottom() + 30 * s, body.width() - 48 * s,
                      body.bottom() - strip.bottom() - 46 * s)
        self.paint_contents(p, area, case, s, True)

    def render_rays(self, w: int, h: int, cx: float, cy: float, col: QColor, rarity: int) -> QPixmap:
        """The light rays behind a rare reveal (drawn once per size)."""
        pm = A.new_pixmap(max(w, 1), max(h, 1))
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if rarity >= D.CLASSIFIED:
            rl = max(w, h) * 0.75
            rg = QRadialGradient(QPointF(cx, cy), rl)
            rc = QColor(col)
            rc.setAlpha(70 if rarity < D.SPECIAL else 110)
            rg.setColorAt(0.0, rc)
            rg.setColorAt(1.0, QColor(rc.red(), rc.green(), rc.blue(), 0))
            p.setBrush(rg)
            p.setPen(Qt.PenStyle.NoPen)
            for k in range(14):
                a = k * math.tau / 14 + 0.1
                ray = QPainterPath(QPointF(cx, cy))
                ray.lineTo(cx + math.cos(a - 0.07) * rl, cy + math.sin(a - 0.07) * rl)
                ray.lineTo(cx + math.cos(a + 0.07) * rl, cy + math.sin(a + 0.07) * rl)
                ray.closeSubpath()
                p.drawPath(ray)
        gr = min(w, h) * 0.45
        glow = QRadialGradient(QPointF(cx, cy), gr)
        gc = QColor(col)
        gc.setAlpha(110)
        glow.setColorAt(0.0, gc)
        glow.setColorAt(1.0, QColor(gc.red(), gc.green(), gc.blue(), 0))
        p.fillRect(QRectF(cx - gr, cy - gr, 2 * gr, 2 * gr), glow)
        p.end()
        return pm

    def draw_modal_reveal(self, p: QPainter, w: int, h: int, s: float) -> None:
        it = self.reveal_item
        if it is None:
            self.modal = None
            return
        t = self.reveal_t
        sk = it.skin
        col = U.rarity_color(it.rarity)
        cx, cy = w / 2, h * 0.38
        # the rarity glow (and rays for the rare ones), drawn once per size
        light = self.cached(("rays", w, h, it.rarity, A.dpr()), lambda: self.render_rays(w, h, cx, cy, col, it.rarity))
        p.setOpacity(0.75 + 0.25 * math.sin(t * 2.4))
        p.drawPixmap(0, 0, light)
        p.setOpacity(1.0)
        k = min(t / 0.35, 1.0)
        k = 1 - (1 - k) ** 3
        aw, ah = min(w * 0.5, 640 * s), min(h * 0.36, 270 * s)
        pm = A.art_pixmap(it.skin_id, it.pattern, it.float, int(aw), int(ah))
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.translate(cx, cy + math.sin(t * 1.6) * 4 * s)
        p.scale(0.6 + 0.4 * k, 0.6 + 0.4 * k)
        p.rotate(math.sin(t * 0.9) * 1.5)
        p.setOpacity(k)
        pw, ph = A.lsize(pm)
        p.drawPixmap(QPointF(-pw / 2, -ph / 2), pm)
        p.restore()
        y = cy + ah / 2 + 10 * s
        if sk.special and t < 3.0:
            U.text(p, QRectF(0, h * 0.04 + 56 * s, w, 50 * s), "★ RARE SPECIAL ITEM ★", 34 * s, U.GOLD,
                   AL.AlignHCenter, "bold", True, shadow=True)
        U.text(p, QRectF(0, y, w, 40 * s), it.full_name, 30 * s, U.ORANGE if it.counter else U.TEXT, AL.AlignHCenter,
               "bold", True, shadow=True)
        U.text(p, QRectF(0, y + 40 * s, w, 24 * s), sk.grade.upper(), 16 * s, col, AL.AlignHCenter, cond=True)
        note = D.pattern_note(sk, it.pattern)
        details = f"FLOAT {it.float:.6f} · PATTERN {it.pattern}" + (f" · {note.upper()}" if note else "")
        if it.counter:
            details += f" · {D.COUNTER_NAME.upper()} CONFIRMED KILLS: {it.kills}"
        U.text(p, QRectF(0, y + 66 * s, w, 22 * s), details, 14 * s, U.DIM, AL.AlignHCenter, cond=True)
        v = self.value(it)
        U.text(p, QRectF(0, y + 92 * s, w, 30 * s), f"MARKET VALUE {D.money(v)}", 22 * s, U.MONEY, AL.AlignHCenter,
               "bold", True)
        by = min(y + 136 * s, h - 60 * s)
        bw = 190 * s
        in_inv = it in self.items
        n = 3 if self.reveal_from == "case" else 2
        x0 = w / 2 - (bw * n + 12 * s * (n - 1)) / 2
        U.button(p, self.hits, QRectF(x0, by, bw, 46 * s), "Keep", self.close_modal, s, "default", size=17)
        U.button(p, self.hits, QRectF(x0 + bw + 12 * s, by, bw, 46 * s),
                 f"Sell · {D.money(D.sale_proceeds(v))}", lambda: self.sell_one(it, False), s, "blue", in_inv, 16, "")
        if n == 3:
            cost = self.open_cost(self.reel.case_id if self.reel else self.modal_case)
            label = "Open another" if cost == 0 else f"Open another · {D.money(cost)}"
            U.button(p, self.hits, QRectF(x0 + 2 * (bw + 12 * s), by, bw, 46 * s), label, self.open_another, s,
                     "primary", self.wallet >= cost, 16, "")

    def draw_modal_inspect(self, p: QPainter, w: int, h: int, s: float) -> None:
        it = self.inspect_item
        if it is None or it not in self.items:
            self.modal = None
            return
        r = self.modal_rect(w, h, s, 0.9, 0.84)
        sk = it.skin
        body = U.panel(p, r, "Inspect", s, 245)
        U.close_x(p, self.hits, QRectF(r.right() - 30 * s, r.top(), 30 * s, 30 * s), self.close_modal, s)
        view = QRectF(body.left(), body.top(), body.width() * 0.62, body.height())
        glow = QRadialGradient(view.center(), view.width() * 0.5)
        gc = U.rarity_color(it.rarity)
        gc.setAlpha(60)
        glow.setColorAt(0.0, gc)
        glow.setColorAt(1.0, QColor(gc.red(), gc.green(), gc.blue(), 0))
        p.fillRect(view, glow)
        aw, ah = view.width() * 0.9, view.height() * 0.62
        pm = A.art_pixmap(it.skin_id, it.pattern, it.float, int(aw), int(ah))
        t = self.reveal_t
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.translate(view.center().x(), view.center().y() + math.sin(t * 1.3) * 5 * s)
        p.rotate(math.sin(t * 0.7) * 2.2)
        p.scale(1.0 + math.sin(t * 0.5) * 0.015, 1.0 + math.sin(t * 0.5) * 0.015)
        pw, ph = A.lsize(pm)
        p.drawPixmap(QPointF(-pw / 2, -ph / 2), pm)
        # a light sweep over the finish
        sweep = (t * 0.35) % 1.6 - 0.3
        g = QLinearGradient(QPointF(-pw / 2, 0), QPointF(pw / 2, 0))
        for f, a in ((sweep - 0.08, 0), (sweep, 40), (sweep + 0.08, 0)):
            g.setColorAt(min(max(f, 0.0), 1.0), QColor(255, 255, 255, a))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setOpacity(0.6)
        p.fillRect(QRectF(-pw / 2, -ph / 2, pw, ph), g)
        p.restore()
        # details
        x = view.right() + 10 * s
        dw = body.right() - x - 24 * s
        y = body.top() + 18 * s
        U.text(p, QRectF(x, y, dw, 22 * s), ("★ " if sk.special else "")
               + (f"{D.COUNTER_NAME} " if it.counter else "") + sk.weapon_name.upper(), 15 * s,
               U.ORANGE if it.counter else U.DIM, cond=True)
        U.text(p, QRectF(x, y + 22 * s, dw, 36 * s), sk.name or "Vanilla", 30 * s, U.TEXT, weight="bold", cond=True)
        U.text(p, QRectF(x, y + 60 * s, dw, 20 * s), sk.grade.upper(), 14 * s,
               U.rarity_color(it.rarity), cond=True)
        y += 92 * s
        if sk.finish != "vanilla":
            U.text(p, QRectF(x, y, dw, 20 * s), f"EXTERIOR: {it.wear.upper()}", 14 * s, U.TEXT, cond=True)
            y += 26 * s
            self.paint_float_bar(p, QRectF(x, y, dw, 10 * s), it.float, s)
            y += 18 * s
            U.text(p, QRectF(x, y, dw, 20 * s), f"WEAR RATING {it.float:.9f}", 13 * s, U.DIM, cond=True)
            U.text(p, QRectF(x, y, dw, 20 * s), f"RANGE {sk.fmin:.2f}-{sk.fmax:.2f}", 12 * s, U.FAINT, AL.AlignRight,
                   cond=True)
            y += 28 * s
        U.text(p, QRectF(x, y, dw, 20 * s), f"PATTERN TEMPLATE {it.pattern}", 13 * s, U.DIM, cond=True)
        y += 22 * s
        note = D.pattern_note(sk, it.pattern)
        if note:
            U.text(p, QRectF(x, y, dw, 22 * s), note.upper(), 16 * s, U.GOLD, weight="bold", cond=True)
            y += 26 * s
        if it.counter:
            U.text(p, QRectF(x, y, dw, 22 * s), f"{D.COUNTER_NAME.upper()} CONFIRMED KILLS: {it.kills}", 14 * s,
                   U.ORANGE, cond=True)
            y += 24 * s
        U.text(p, QRectF(x, y, dw, 20 * s), f"FROM THE {D.CASES[sk.case].name.upper()}", 12 * s, U.FAINT, cond=True)
        y += 34 * s
        v = self.value(it)
        U.text(p, QRectF(x, y, dw, 30 * s), D.money(v), 28 * s, U.MONEY, weight="bold", cond=True)
        U.text(p, QRectF(x, y + 30 * s, dw, 18 * s), f"MARKET PRICE TODAY · YOU RECEIVE {D.money(D.sale_proceeds(v))}",
               11 * s, U.FAINT, cond=True)
        hist = self.cached(("hist", it.uid, self.day()),
                           lambda: D.price_history(it, 21, self.day()))
        U.sparkline(p, QRectF(x, y + 54 * s, dw, 34 * s), hist, U.MONEY if hist[-1] >= hist[0] else U.RED)
        bh = 42 * s
        by = body.bottom() - bh - 18 * s
        equipped = it.uid in self.loadout.values()
        U.button(p, self.hits, QRectF(x, by - bh - 10 * s, dw, bh), "Equipped" if equipped else "Equip",
                 lambda: self.equip(it), s, "blue", not equipped, 16, "")
        U.button(p, self.hits, QRectF(x, by, dw * 0.6 - 5 * s, bh), f"Sell · {D.money(D.sale_proceeds(v))}",
                 lambda: self.sell_one(it), s, "primary", size=16, sound="")
        U.button(p, self.hits, QRectF(x + dw * 0.6 + 5 * s, by, dw * 0.4 - 5 * s, bh), "Close", self.close_modal, s,
                 size=16)

    def paint_float_bar(self, p: QPainter, r: QRectF, f: float, s: float) -> None:
        cols = ("#3fb950", "#7fcf5a", "#d6c53a", "#e08a2e", "#d64545")
        lo = 0.0
        for (_n, _sh, top), c in zip(D.WEARS, cols, strict=True):
            p.fillRect(QRectF(r.left() + r.width() * lo, r.top(), r.width() * (top - lo), r.height()), QColor(c))
            lo = top
        x = r.left() + r.width() * f
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(U.TEXT)
        p.setPen(QPen(QColor(0, 0, 0, 200), 1))
        tri = QPainterPath(QPointF(x, r.top() - 1))
        tri.lineTo(x - 6 * s, r.top() - 10 * s)
        tri.lineTo(x + 6 * s, r.top() - 10 * s)
        tri.closeSubpath()
        p.drawPath(tri)
        p.restore()

    def draw_modal_confirm(self, p: QPainter, w: int, h: int, s: float) -> None:
        if self.confirm is None:
            self.modal = None
            return
        q, action = self.confirm
        r = QRectF((w - min(560 * s, w * 0.9)) / 2, h / 2 - 90 * s, min(560 * s, w * 0.9), 180 * s)
        body = U.panel(p, r, "Confirm", s, 250)
        p.save()
        p.setFont(U.F(16 * s, "regular"))
        p.setPen(U.TEXT)
        p.drawText(body.adjusted(24 * s, 12 * s, -24 * s, -60 * s), int(AL.AlignCenter | Qt.TextFlag.TextWordWrap), q)
        p.restore()
        bw = 150 * s
        U.button(p, self.hits, QRectF(r.center().x() - bw - 6 * s, r.bottom() - 56 * s, bw, 40 * s), "Sell", action,
                 s, "primary", size=16, sound="")
        U.button(p, self.hits, QRectF(r.center().x() + 6 * s, r.bottom() - 56 * s, bw, 40 * s), "Cancel",
                 self.close_modal, s, size=16)

    def draw_modal_contract(self, p: QPainter, w: int, h: int, s: float) -> None:
        t = self.contract_t
        r = QRectF(w / 2 - min(420 * s, w * 0.4), h * 0.12, min(840 * s, w * 0.8), h * 0.76)
        # the paper contract
        p.fillRect(r, QColor("#e9e2cf"))
        p.setPen(QPen(QColor("#8c7f5f"), 2))
        p.drawRect(r.adjusted(8 * s, 8 * s, -8 * s, -8 * s))
        U.text(p, QRectF(r.left(), r.top() + 24 * s, r.width(), 40 * s), "TRADE UP CONTRACT", 34 * s,
               QColor("#3a3326"), AL.AlignHCenter, "bold", True)
        U.text(p, QRectF(r.left(), r.top() + 66 * s, r.width(), 20 * s),
               "I hereby exchange ten items for one item of the next rarity.", 14 * s, QColor("#5a5040"),
               AL.AlignHCenter)
        p.setPen(QPen(QColor("#8c7f5f"), 1))
        for k in range(8):
            y = r.top() + 110 * s + k * 34 * s
            if y < r.bottom() - 120 * s:
                p.drawLine(QPointF(r.left() + 60 * s, y), QPointF(r.right() - 60 * s, y))
        # items fly in and shrink to the centre
        res = self.contract_result
        k = min(t / 1.1, 1.0)
        if res is not None and k < 1.0:
            for i in range(10):
                a = i / 10 * math.tau
                rad = (1 - k) * r.width() * 0.45
                x = r.center().x() + math.cos(a) * rad
                y = r.center().y() + math.sin(a) * rad * 0.6
                sz = 60 * s * (1 - k * 0.7)
                p.fillRect(QRectF(x - sz / 2, y - sz * 0.39, sz, sz * 0.78), U.rarity_color(max(res.rarity - 1, 0)))
        if t > 1.35:
            a = min((t - 1.35) / 0.12, 1.0)
            p.save()
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.translate(r.center().x() + 140 * s, r.bottom() - 120 * s)
            p.rotate(-14)
            sc = 1.6 - 0.6 * a
            p.scale(sc, sc)
            p.setOpacity(a)
            p.setPen(QPen(QColor("#b3261e"), 5 * s))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(-130 * s, -40 * s, 260 * s, 80 * s), 10 * s, 10 * s)
            U.text(p, QRectF(-130 * s, -40 * s, 260 * s, 80 * s), "APPROVED", 40 * s, QColor("#b3261e"),
                   AL.AlignHCenter, "bold", True)
            p.restore()


def _paint_thumbnail(p: QPainter, rect: QRectF, t: float) -> None:
    p.save()
    p.setClipRect(rect)
    g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    g.setColorAt(0.0, QColor("#232b36"))
    g.setColorAt(1.0, QColor("#0d1015"))
    p.fillRect(rect, g)
    pulse = 0.75 + 0.25 * math.sin(t * 2.2)
    glow = QRadialGradient(QPointF(rect.center().x(), rect.top() + rect.height() * 0.55), rect.width() * 0.55)
    glow.setColorAt(0.0, QColor(228, 174, 57, int(170 * pulse)))
    glow.setColorAt(0.5, QColor(228, 174, 57, int(50 * pulse)))
    glow.setColorAt(1.0, QColor(228, 174, 57, 0))
    p.fillRect(rect, glow)
    cw = rect.width() * 0.5
    ch = cw / 1.35
    p.drawPixmap(QPointF(rect.center().x() - cw / 2, rect.bottom() - ch - rect.height() * 0.04),
                 A.case_pixmap("neon", int(cw), int(ch)))
    kw, kh = rect.width() * 0.62, rect.height() * 0.3
    knife = A.art_pixmap("sandstorm.karambit.afterglow", 95, 0.01, int(kw), int(kh))
    p.save()
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.translate(rect.center().x(), rect.top() + rect.height() * 0.32 + math.sin(t * 1.8) * rect.height() * 0.025)
    p.rotate(-8 + math.sin(t * 1.1) * 6)
    kw2, kh2 = A.lsize(knife)
    p.drawPixmap(QPointF(-kw2 / 2, -kh2 / 2), knife)
    p.restore()
    p.setPen(Qt.PenStyle.NoPen)
    for i in range(7):
        a = t * 1.3 + i * 0.9
        x = rect.center().x() + math.cos(a * 1.7 + i) * rect.width() * 0.36
        y = rect.top() + rect.height() * (0.5 + 0.35 * math.sin(a + i * 2))
        sz = rect.width() * 0.012 * (1 + math.sin(a * 3) * 0.5)
        p.setBrush(QColor(255, 220, 140, int(160 + 90 * math.sin(a * 2))))
        p.drawEllipse(QPointF(x, y), sz, sz)
    p.restore()

