"""Storm Drop - the arcade's third-person battle royale: drop onto Brightwater Isle, loot, build and outlast the bots."""

__all__ = ["StormDropGame"]


def __getattr__(name: str):  # noqa: ANN202 - the game (and Qt) load only when asked for: helpers stay light
    if name == "StormDropGame":
        from app.arcade.games.storm_drop.game import StormDropGame

        return StormDropGame
    raise AttributeError(name)
