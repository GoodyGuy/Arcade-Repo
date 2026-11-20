These pictures are unmodified copies of art from the Freedoom project (https://freedoom.github.io/,
https://github.com/freedoom/freedoom), used under the BSD licence in COPYING.adoc beside this file.
CREDITS lists the Freedoom contributors.

sprites/  monster, item, decoration, projectile and first-person weapon frames
patches/  wall pictures (Hellfire combines a few of them, e.g. switches drawn on a wall panel)
flats/    floor and ceiling pictures

Hellfire reads them through assets.py. If this folder is missing, the game falls back to its own
procedural art (textures.py, sprites.py, models.py) and still plays.
