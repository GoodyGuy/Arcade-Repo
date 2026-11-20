# Break Room Arcade

A hidden arcade for a PySide6 desktop program. Hold **Ctrl + 6 + 7 + 9** together and the arcade opens.

## Install

Copy the `arcade` folder into the program's `app` folder, so that it is `app/arcade`. Replacing an older
`app/arcade` with this one is enough.

The program installs the hotkey once at start-up, after the `QApplication` exists:

```python
try:
    from app.arcade import install_hotkey

    install_hotkey(app)
except Exception:
    pass  # the program is the same without the arcade
```

The arcade needs only what the program already has: Python 3, PySide6 and NumPy. It adds no other packages.
A game that fails to import is left off the home screen; the rest still open.

## Games

| Folder | Title | What it is |
| --- | --- | --- |
| `games/rail_rush` | Rail Rush | endless runner |
| `games/blockcraft` | BlockCraft | voxel sandbox: infinite world, farming, running water, creatures, settlers, two further realms, a boss, recipe book and goals |
| `games/hellfire` | Hellfire | 1990s-style corridor shooter, seven maps |
| `games/case_clash` | Case Clash | case opening, with a surf mode on three maps |
| `games/shadow_ops` | Shadow Ops II | wave survival and team matches against bots |
| `games/bo1_web.py` | (web) | opens a website in the browser |
| `games/storm_drop` | Storm Drop | battle royale with building, against bots |
| `games/vanguard` | Vanguard | hero team objective shooter, against bots |
| `games/breach_point` | Breach Point | round-based plant-and-defuse tactical shooter, against bots |
| `games/ashworth_hall` | Ashworth Hall | lawn restoration on a ride-on mower: one self-contained web page, played in Qt's built-in web view (or the browser where PySide6 has none) |

All games are single player. Sound and music are synthesised in code; no audio files are shipped.

## Credits and licences

- **BlockCraft textures:** Pixel Perfection by XSSheep, continued by Nova_Wostra, with the contributors named in
  `arcade/games/blockcraft/Pixel Perfection 26.3-97.2/pack.txt`. Published by its authors under
  CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Only the block, item and colormap textures
  that the game reads are included here, unchanged. Without the folder BlockCraft draws its own textures.
- **Hellfire art:** unmodified pictures from the Freedoom project (https://freedoom.github.io/), under the BSD
  licence in `arcade/games/hellfire/freedoom/COPYING.adoc`; contributors in `CREDITS` beside it. Without the
  folder Hellfire draws its own art.
- Everything else is drawn and synthesised by the code in this repository.

The games are original works in familiar genres. They are not affiliated with, or endorsed by, the makers of
any commercial game.
