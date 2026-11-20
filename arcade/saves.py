"""The arcade's saved progress: one JSON file in the user's application data folder (``TDS_ARCADE_DIR`` overrides it),
never in a project or the program's settings. A damaged file is kept aside and the arcade starts fresh."""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

SAVE_VERSION = 1
FILE_NAME = "arcade_save.json"


def arcade_dir() -> Path:
    override = os.environ.get("TDS_ARCADE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "TankDesignStudio" / "Arcade"


class SaveStore:
    """Every game's progress (``games[game_id]``: whatever the game's save_state() gave) and the arcade settings."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or arcade_dir() / FILE_NAME
        self.data: dict = self._fresh()
        self.load()

    @staticmethod
    def _fresh() -> dict:
        return {"version": SAVE_VERSION, "games": {}, "settings": {"muted": False, "volume": 0.7},
                "stats": {"opened": 0, "play_seconds": {}}}

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or not isinstance(loaded.get("games"), dict):
                raise ValueError("not an arcade save")
        except (OSError, ValueError):
            log.warning("Arcade save %s unreadable: kept aside, starting fresh", self.path, exc_info=True)
            try:
                self.path.replace(self.path.with_name(f"{self.path.stem}.damaged-{int(time.time())}.json"))
            except OSError:
                pass
            return
        fresh = self._fresh()
        for key, value in fresh.items():
            if isinstance(value, dict):
                value.update(loaded.get(key) or {})
            loaded[key] = value if isinstance(value, dict) else loaded.get(key, value)
        self.data = loaded

    def game(self, game_id: str) -> dict:
        """A copy of the game's saved progress ({} when it was never played)."""
        return copy.deepcopy(self.data["games"].get(game_id) or {})

    def put(self, game_id: str, state: dict) -> None:
        self.data["games"][game_id] = copy.deepcopy(state)

    def reset(self, game_id: str) -> None:
        self.data["games"].pop(game_id, None)
        self.data["stats"]["play_seconds"].pop(game_id, None)

    def add_play_time(self, game_id: str, seconds: float) -> None:
        times = self.data["stats"]["play_seconds"]
        times[game_id] = round(times.get(game_id, 0.0) + max(seconds, 0.0), 1)

    @property
    def settings(self) -> dict:
        return self.data["settings"]

    def flush(self) -> bool:
        """Write the file (atomically); False when it could not be written."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self.path)
            return True
        except (OSError, TypeError, ValueError):
            log.warning("Arcade progress not saved", exc_info=True)
            return False
