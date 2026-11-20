"""BlockCraft - a Minecraft-style voxel sandbox for the Break Room Arcade."""

__all__ = ["BlockCraftGame"]


def __getattr__(name: str):  # noqa: ANN202 - the game (and Qt) load only when asked for: helpers stay light
    if name == "BlockCraftGame":
        from app.arcade.games.blockcraft.game import BlockCraftGame

        return BlockCraftGame
    raise AttributeError(name)
