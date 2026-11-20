"""ASHWORTH HALL - a lawn restoration game written as one web page (ashworth_hall.html), played inside the arcade."""

__all__ = ["AshworthHallGame"]


def __getattr__(name: str):  # noqa: ANN202 - the game (and Qt's web view) load only when asked for
    if name == "AshworthHallGame":
        from app.arcade.games.ashworth_hall.game import AshworthHallGame

        return AshworthHallGame
    raise AttributeError(name)
