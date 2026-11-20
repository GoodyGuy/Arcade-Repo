"""The arcade's games, in the order of the home screen. Each module is imported only when the arcade opens."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameEntry:
    module: str
    cls: str

    def load(self):  # noqa: ANN201 - the ArcadeGame subclass, None when it does not import
        try:
            return getattr(importlib.import_module(f"app.arcade.games.{self.module}"), self.cls)
        except Exception:  # noqa: BLE001 - one broken game must not close the arcade
            log.warning("Arcade game %s not available", self.module, exc_info=True)
            return None


GAMES = (
    GameEntry("rail_rush", "RailRushGame"),  # endless runner (Subway Surfers style)
    GameEntry("blockcraft", "BlockCraftGame"),  # voxel sandbox (Minecraft style)
    GameEntry("hellfire", "HellfireGame"),  # 1993 shooter (Doom style)
    GameEntry("case_clash", "CaseClashGame"),  # weapon case opening (CS:GO style)
    GameEntry("shadow_ops", "ShadowOpsGame"),  # zombies and multiplayer shooter (Black Ops II style)
    GameEntry("bo1_web", "Bo1WebGame"),  # Black Ops 1 Zombies website, opened in the browser
    GameEntry("storm_drop", "StormDropGame"),  # battle royale with building, against bots
    GameEntry("vanguard", "VanguardGame"),  # hero team objective shooter, against bots
    GameEntry("breach_point", "BreachPointGame"),  # round-based plant-and-defuse tactical shooter, against bots
    GameEntry("ashworth_hall", "AshworthHallGame"),  # lawn restoration: one web page, played in Qt's own web view
)


def load_games() -> list:
    return [g for g in (entry.load() for entry in GAMES) if g is not None]
