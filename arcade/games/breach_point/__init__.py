"""BREACH POINT - a five-against-five, round-based tactical shooter against and alongside bots for the Break Room
Arcade: carry the charge to a site, plant it and hold, or stop the plant and defuse in time."""

__all__ = ["BreachPointGame"]


def __getattr__(name: str):  # noqa: ANN202 - the game (and Qt) load only when asked for: helpers stay light
    if name == "BreachPointGame":
        from app.arcade.games.breach_point.game import BreachPointGame

        return BreachPointGame
    raise AttributeError(name)
