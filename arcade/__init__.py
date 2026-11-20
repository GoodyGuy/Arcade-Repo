"""The break room arcade: a hidden game menu (Ctrl + 6 + 7 + 9 held together anywhere in the program).

Nothing here touches a project, the settings or the program's windows: the hotkey only watches the key presses (it never
consumes one), the games load the first time the arcade opens, and their progress is kept in its own file
(app/arcade/saves.py)."""

from __future__ import annotations

__all__ = ["ArcadeHotkey", "install_hotkey"]


def __getattr__(name: str):  # noqa: ANN202 - the hotkey (and Qt) load when first asked for: a game's helper
    if name in __all__:  # processes import this package without them
        from app.arcade import hotkey

        return getattr(hotkey, name)
    raise AttributeError(name)
