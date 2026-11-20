"""VANGUARD - the arcade's five-against-five hero team shooter (original heroes and maps, played with bots)."""

__all__ = ["VanguardGame"]


def __getattr__(name: str):  # noqa: ANN202 - the game (and Qt) load only when asked for: helpers stay light
    if name == "VanguardGame":
        from app.arcade.games.vanguard.game import VanguardGame

        return VanguardGame
    raise AttributeError(name)
