"""Case Clash's weapon silhouettes: side views with the muzzle / blade tip to the RIGHT, like the inventory icons.

Each weapon is a list of parts drawn in order. Kinds: ``p`` painted with the skin's finish, ``d`` dark gunmetal or
polymer, ``m`` bright steel, ``h`` a handle, ``s`` a soft groove / seam line. Paths are SVG-like (M, L, Q, C, Z,
absolute); ``O cx cy r`` adds a circle; sub-paths inside one part cut holes (odd-even fill) unless the path starts
with ``W`` (winding fill)."""

from __future__ import annotations

# --- pistols (about 64 x 44) -----------------------------------------------------------------------------------------


def _guard(x0: float, x1: float, y0: float, y1: float, round_front: bool = True) -> str:
    """A trigger guard ring with its trigger: front bottom corner rounded unless ``round_front`` is False."""
    t = 1.4
    if round_front:
        outer = f"M{x0} {y0} L{x1} {y0} L{x1} {y1 - 3} Q{x1} {y1} {x1 - 3} {y1} L{x0} {y1} Z"
        inner = (f"M{x0 + t} {y0 + t} L{x1 - t} {y0 + t} L{x1 - t} {y1 - 3} Q{x1 - t} {y1 - t} {x1 - 3} {y1 - t} "
                 f"L{x0 + t} {y1 - t} Z")
    else:
        outer = f"M{x0} {y0} L{x1} {y0} L{x1} {y1} L{x0} {y1} Z"
        inner = f"M{x0 + t} {y0 + t} L{x1 - t} {y0 + t} L{x1 - t} {y1 - t} L{x0 + t} {y1 - t} Z"
    tx = x0 + (x1 - x0) * 0.35
    trigger = f"M{tx} {y0} L{tx + 1.6} {y0} Q{tx + 2.2} {y0 + 3} {tx + 0.6} {y1 - t - 0.6} L{tx - 0.4} {y1 - t - 0.6} Z"
    return outer + " " + inner + " " + trigger


def _grip(x_back: float, x_front: float, top: float, bottom: float, slant: float, beaver: float = 2.0) -> str:
    """A pistol grip raked backwards by ``slant``, with a rounded beavertail at the top of the back strap."""
    return (f"M{x_back - beaver} {top} L{x_front} {top} L{x_front - slant} {bottom - 1.5} "
            f"Q{x_front - slant} {bottom} {x_front - slant - 2} {bottom} L{x_back - slant + 1.5} {bottom} "
            f"Q{x_back - slant - 0.5} {bottom} {x_back - slant - 0.3} {bottom - 2} L{x_back} {top + 6} "
            f"Q{x_back - beaver - 0.8} {top + 4.5} {x_back - beaver} {top} Z")


def _serrations(x0: float, y0: float, y1: float, n: int = 5, step: float = 1.4) -> str:
    return " ".join(f"M{x0 + k * step} {y0} L{x0 + k * step + 0.5} {y0} L{x0 + k * step + 0.5} {y1} "
                    f"L{x0 + k * step} {y1} Z" for k in range(n))


PISTOLS = {
    "glock": (
        ("p", _grip(6, 21, 15.5, 43, 4.5, 3.0)),
        ("d", "M1.5 41 L17 41 L17 43.6 L1.2 43.6 Z"),  # magazine base plate
        ("p", "M4 13 L58 13 L58 17.5 L38 17.5 L36 16.8 L21 16.8 L6 16.5 Z"),  # frame / dust cover
        ("d", "M44 17.5 L57 17.5 L57 18.6 L44 18.6 Z"),  # accessory rail
        ("d", _guard(21, 35, 16.8, 25.5, False)),
        ("p", "M3 4.5 L59 4.5 Q61.5 4.5 61.5 7 L61.5 13 L3 13 L3 6 Z"),  # slide
        ("s", _serrations(6, 6, 12, 6)),
        ("d", "M58.5 7.5 L61.5 7.5 L61.5 10.5 L58.5 10.5 Z"),  # muzzle
        ("d", "M55 3 L57.5 3 L57.5 4.5 L55 4.5 Z M6 2.8 L11 2.8 L11 4.5 L6 4.5 Z"),  # sights
    ),
    "usps": (
        ("d", _grip(7, 21, 16, 42, 4.0, 3.5)),
        ("d", "M3 40 L16.5 40 L16.5 42.6 L2.7 42.6 Z"),
        ("d", "M2.5 6 L7 5 L8 10 L4 10.5 Z"),  # hammer
        ("p", "M5 13 L45 13 L45 17.5 L39 18 L22 17.5 L6 17 Z"),
        ("d", _guard(21, 33, 17.3, 25.5)),
        ("p", "M5 4.5 L43 4.5 Q46 4.5 46 7.5 L46 13 L5 13 Z"),
        ("s", _serrations(8, 6, 12, 5)),
        ("d", "M45 6.8 L49 6.8 L49 14.2 L45 14.2 Z"),  # thread / collar
        ("p", "M49 6 L96 6 Q99 6 99 9 L99 12 Q99 15 96 15 L49 15 Z"),  # the long suppressor
        ("s", "M91 6 L92.2 6 L92.2 15 L91 15 Z M53 6 L54 6 L54 15 L53 15 Z"),
        ("d", "M40 3 L42.5 3 L42.5 4.5 L40 4.5 Z M7 3 L11 3 L11 4.5 L7 4.5 Z"),
    ),
    "p250": (
        ("p", _grip(7, 21, 16, 42.5, 5.0, 2.5)),
        ("d", "M1.5 40.5 L16 40.5 L16 43 L1.2 43 Z"),
        ("d", "M-1 6.5 L5 5.5 L5.5 10.5 L1 10.5 Z"),  # hammer spur
        ("p", "M5 13 L53 13 L53 16.5 L40 17.8 L22 17.5 L6 17 Z"),
        ("d", _guard(21, 36, 17.3, 26.5)),
        ("p", "M5 5 L51 5 L55 7.5 L55 13 L5 13 L4 9 Z"),  # bevelled slide nose
        ("s", _serrations(8, 6.5, 12, 5)),
        ("d", "M53 8.5 L55 8.5 L55 11.5 L53 11.5 Z"),
        ("d", "M48 3.5 L50.5 3.5 L50.5 5 L48 5 Z M8 3.5 L12 3.5 L12 5 L8 5 Z"),
    ),
    "fiveseven": (
        ("p", _grip(6, 21, 16, 43, 2.0, 2.0)),
        ("d", "M3.5 41 L18.8 41 L18.8 43.6 L3.3 43.6 Z"),
        ("p", "M3 13 L60 13 L60 17.5 L37 17.5 L21 17 L4 16.8 Z"),
        ("d", "M42 17.5 L59 17.5 L59 19 L42 19 Z"),
        ("d", _guard(21, 37, 17, 25, False)),
        ("p", "M2.5 6 L59.5 6 L61 7.5 L61 13 L2.5 13 Z"),  # long flat slide
        ("s", "M8 9 L52 9 L52 9.8 L8 9.8 Z"),  # slide groove
        ("s", _serrations(4.5, 7, 12, 4, 1.3)),
        ("d", "M58.5 8.5 L61 8.5 L61 11.5 L58.5 11.5 Z"),
        ("d", "M56 4.5 L58.5 4.5 L58.5 6 L56 6 Z M5 4.5 L9 4.5 L9 6 L5 6 Z"),
    ),
    "deagle": (
        ("d", _grip(7, 22, 17.5, 45, 4.5, 2.5)),
        ("d", "M2 43 L17 43 L17 45.6 L1.8 45.6 Z"),
        ("d", "M0 4.5 L6 3.5 L7 9 L2.5 9.5 Z"),  # hammer
        ("p", "M6 11 L56 11 L40 18.8 L6 18.8 Z"),  # frame with the triangular nose under the barrel
        ("d", _guard(21, 38.5, 18.5, 28)),
        ("p", "M5 2.5 L66 2.5 L67.5 4 L67.5 11 L5 11 Z"),  # the big slide
        ("s", "M9 7.4 L64 7.4 L64 8.2 L9 8.2 Z"),
        ("s", _serrations(9, 3.8, 6.8, 4, 1.8)),
        ("d", "M66 5.3 L67.5 5.3 L67.5 8.7 L66 8.7 Z"),
        ("d", "M8 1.2 L13 1.2 L13 2.5 L8 2.5 Z M62 1 L64.5 1 L64.5 2.5 L62 2.5 Z"),
    ),
}

# --- rifles, SMGs, shotguns and the LMG ----------------------------------------------------------------------------

_M4_STOCK = (("d", "M11 13.8 L28 13.8 L28 17.6 L11 17.6 Z"),  # buffer tube
             ("p", "M0 10.5 L13 10.5 L15.5 12.8 L22 12.8 L22 18.8 L15.5 18.8 L13 25 L2 25 Q0 25 0 23 Z"))
_M4_BODY = (("d", "M30 20 L36 20 L34 30.5 L28 29.5 Z"),  # pistol grip
            ("d", _guard(36, 45, 19.5, 23.5, False)),
            ("p", "M26 14.5 L57 14.5 L57 20 L49 21.5 L41 21.5 L36 20.5 L26 20.5 Z"),  # lower receiver
            ("p", "M26 9.5 L60 9.5 L60 14.5 L26 14.5 Z"),  # upper receiver
            ("d", "M27 8 L59 8 L59 9.5 L27 9.5 Z"),  # rail
            ("d", "M28 5.5 L32.5 5.5 L33 8 L28 8 Z"))  # rear sight
RIFLES = {
    "ak47": (
        ("p", "M0 14 L27 12.5 L27 19 L21 19.5 Q12 22 1.5 28 L0 28 Z"),  # wooden stock, dropping to the butt
        ("d", "M-1 13.8 L0.6 13.8 L0.6 28.2 L-1 28.2 Z"),  # butt plate
        ("p", "M32.5 19 L39 19 L37 29.5 L31 29 Z"),  # wooden pistol grip
        ("d", _guard(39, 46.5, 19, 23.5)),
        ("p", "M46 19 L54.5 19 Q56 27 61.5 34.5 L54 37.5 Q48.5 30 46 19 Z"),  # curved magazine
        ("d", "M54.5 36 L61.5 33.3 L62.3 34.8 L55 37.8 Z"),
        ("p", "M26 11.5 L60 11.5 L60 19 L26 19.5 Z"),  # receiver
        ("s", "M27 13 L58 13 L58 13.8 L27 13.8 Z"),  # dust cover seam
        ("d", "M57 9.5 L63 9.5 L63 12 L57 12 Z"),  # rear sight block
        ("p", "M60 13 L79 13.5 L79 19 L60 19 Z"),  # lower handguard
        ("p", "M62 9.8 L77 9.8 L78 12.8 L62 12.8 Z"),  # gas tube cover
        ("d", "M77 9.5 L82 9.5 L82 14.5 L77 14.5 Z"),  # gas block
        ("d", "M79 14 L97 14 L97 16.5 L79 16.5 Z M82 17 L94 17 L94 17.8 L82 17.8 Z"),  # barrel, cleaning rod
        ("d", "M90 9 L92.5 9 L93 14 L90 14 Z"),  # front sight post
        ("d", "M96 13 L100.5 13 L100.5 17.3 L96 17.3 Z"),  # muzzle brake
    ),
    "m4a4": _M4_STOCK + (("p", "M44.5 20.5 L52.5 20.5 L54 33 L46 34 Z"),) + _M4_BODY + (
        ("p", "M60 9.5 L82 9.5 L82 17.5 L60 17.5 Z"),  # quad-rail handguard
        ("d", "M61 8.2 L81 8.2 L81 9.5 L61 9.5 Z"),
        ("s", "M61 12.6 L81 12.6 L81 13.4 L61 13.4 Z"),
        ("d", "M77 4 L79.5 4 L82 9.5 L76 9.5 Z"),  # front sight
        ("d", "M82 12 L95 12 L95 14.8 L82 14.8 Z"),
        ("d", "M94 10.8 L100.5 10.8 L100.5 16 L94 16 Z"),  # flash hider
    ),
    "m4a1s": _M4_STOCK + (("p", "M44.5 20.5 L52.5 20.5 Q53 26 55 31 L47.5 33 Q45.5 27 44.5 20.5 Z"),) + _M4_BODY + (
        ("p", "M60 10 L80 10 L80 17 L60 17 Z"),  # slim handguard
        ("s", "M61 13 L79 13 L79 13.8 L61 13.8 Z"),
        ("d", "M80 11.8 L84 11.8 L84 15.2 L80 15.2 Z"),
        ("p", "M84 9.8 L108 9.8 Q110 9.8 110 11.8 L110 15.2 Q110 17.2 108 17.2 L84 17.2 Z"),  # suppressor
        ("s", "M86.5 9.8 L87.5 9.8 L87.5 17.2 L86.5 17.2 Z"),
    ),
    "awp": (
        ("p", "M0 13 L5 12 L33 14 L36.5 14 L36.5 21.5 L30 22 Q26.5 24 24.5 28.5 L4 28.5 Q0 28.5 0 25.5 Z "
              "M12.5 18 L22.5 18 Q24.5 18 23.8 20 L22.6 23 Q22 24.2 20.5 24.2 L13.5 24.2 Q12 24.2 12 22.7 "
              "L12 19 Q12 18 12.5 18 Z"),  # thumbhole stock
        ("d", "M-1.5 12.6 L0.6 12.6 L0.6 28.5 L-1.5 28.5 Z"),
        ("d", _guard(37, 46, 21, 25.5)),
        ("d", "M48 21 L57 21 L57 26.5 L48 26.5 Z"),  # magazine
        ("p", "M35 13 L64 13 L64 21 L35 21.5 Z"),  # receiver
        ("d", "M43 7.5 L46 7.5 L46 13 L43 13 Z M56 7.5 L59 7.5 L59 13 L56 13 Z"),  # scope rings
        ("p", "M35 4.5 L40 2.8 L43 5 L58 5 L62 2.2 L69 2.2 L69 9.8 L62 9.8 L58 8 L43 8 L40 10.2 L35 8.5 Z"),
        ("d", "M49 2.8 L52 2.8 L52 5 L49 5 Z M68.2 2.2 L69.5 2.2 L69.5 9.8 L68.2 9.8 Z"),  # turret, lens
        ("d", "M50 13 L52 13 L53.5 17 L51.5 17.5 Z"),  # bolt handle
        ("p", "M64 14 L87 15 L87 20 L64 21 Z"),  # fore-end
        ("d", "M87 15.8 L119 16.1 L119 18.3 L87 18.6 Z"),  # long barrel
        ("d", "M116 14.8 L123 14.8 L123 19.6 L116 19.6 Z"),  # muzzle brake
    ),
    "ssg08": (
        ("p", "M0 14 L5 13 L30 15.5 L33.5 15.5 L33.5 20.5 L28 21 Q25 22.5 22 27 L3 27 Q0 27 0 24 Z "
              "M10.5 18 L21 18 L19.5 22.5 L11 22.5 Z"),
        ("d", _guard(34, 42, 20, 24)),
        ("d", "M44 20 L50 20 L50 24.5 L44 24.5 Z"),
        ("p", "M32.5 14 L58 14 L58 20 L32.5 20.5 Z"),
        ("d", "M39 8.2 L41.5 8.2 L41.5 14 L39 14 Z M51 8.2 L53.5 8.2 L53.5 14 L51 14 Z"),
        ("p", "M34 6.5 L38 5 L41 7 L53 7 L56.5 5 L62 5 L62 10.5 L56.5 10.5 L53 9 L41 9 L38 11 L34 10 Z"),
        ("d", "M45.5 5.3 L47.5 5.3 L47.5 7 L45.5 7 Z"),
        ("p", "M58 15 L78 16 L78 19 L58 20 Z"),
        ("d", "M78 16.3 L113 16.7 L113 18.2 L78 18.6 Z"),  # long thin barrel
        ("d", "M111 15.8 L115 15.8 L115 18.9 L111 18.9 Z"),
    ),
    "galil": (
        ("p", "M0 10 L26 12.5 L26 19.5 L0 26 Q-1 18 0 10 Z M4 13.5 L22 14.8 L22 17.8 L4 21.5 Z"),  # skeleton stock
        ("p", "M33 19 L39 19 L37 29.5 L31 29 Z"),
        ("d", _guard(39, 46.5, 19, 23.5)),
        ("p", "M47 19 L55 19 Q56.5 26 60.5 32.5 L53.5 35.5 Q49 28 47 19 Z"),
        ("p", "M26 11.5 L61 11.5 L61 19 L26 19.5 Z"),
        ("d", "M38 8.5 L50 8.5 L50 11.5 L38 11.5 Z"),  # carry handle base
        ("p", "M61 12.2 L83 12.2 L83 19 L61 19 Z"),
        ("d", "M61 10 L84 10 L84 12.2 L61 12.2 Z"),  # gas tube
        ("d", "M83 14 L99 14 L99 16.5 L83 16.5 Z"),
        ("d", "M91 8 L93.5 8 L93.5 14 L91 14 Z"),  # tall front sight
        ("d", "M97.5 13.2 L102 13.2 L102 17.2 L97.5 17.2 Z"),
        ("d", "M84 16.5 L86 16.5 L85 23 L83.5 23 Z"),  # folded bipod
    ),
    "famas": (
        ("p", "M0 13 Q0 11 3 11 L72 11 L76 13 L76 19.5 L58 19.5 L55 21.5 L22 21.5 L20 25 L2 25 Q0 25 0 23 Z"),
        ("p", "M14 11 L18 3 L68 3 L70 11 L66 11 L64 6.5 L21 6.5 L18 11 Z"),  # the long carry handle
        ("d", "M49 19.5 L58 19.5 L56.5 30 L50 30 Z"),
        ("d", _guard(40, 52, 19.5, 26, False)),
        ("d", "M28 21.5 L37 21.5 L37 32 L28 32 Z"),  # magazine behind the grip
        ("d", "M76 14.3 L94 14.3 L94 16.9 L76 16.9 Z"),
        ("d", "M92 13.6 L97.5 13.6 L97.5 17.6 L92 17.6 Z"),
        ("d", "M70 19.5 L88 19.8 L88 21 L70 20.8 Z"),  # folded bipod
    ),
    "mp9": (
        ("d", "M0 7.5 L15 7.5 L15 10 L2.5 10 L2.5 22 L8 22 L8 24.5 L0 24.5 Z"),  # wire stock
        ("p", "M22 17 L32 17 L31.5 39 L23 39 Z"),  # grip with the magazine inside
        ("d", "M22.5 38 L32 38 L32 40.5 L22.5 40.5 Z"),
        ("d", _guard(32, 41, 17, 22.5)),
        ("d", "M47 17 L53 17 L52 29 Q52 30 51 30 L48 30 Q47 30 47 29 Z"),  # vertical fore grip
        ("p", "M14 6 L56 6 L58.5 8.5 L58.5 17 L14 17 Z"),
        ("d", "M18 4.5 L52 4.5 L52 6 L18 6 Z"),
        ("d", "M58.5 10 L66 10 L66 13.5 L58.5 13.5 Z"),
    ),
    "mac10": (
        ("d", "M0 6 L8 6 L8 7.5 L1.5 7.5 L1.5 15 L8 15 L8 16.5 L0 16.5 Z"),  # folded stock rods
        ("p", "M20 17 L30 17 L30 41 L20 41 Z"),  # long magazine through the grip
        ("d", "M19.5 40 L30.5 40 L30.5 43 L19.5 43 Z"),
        ("d", _guard(30, 38.5, 17, 22.5)),
        ("p", "M8 5 L52 5 L52 17 L8 17 Z"),
        ("d", "M27 2.5 L32 2.5 L32 5 L27 5 Z"),  # cocking knob
        ("d", "M52 8.5 L61 8.5 L61 12.5 L52 12.5 Z"),
        ("d", "M44 17 L48 17 L48 20 L44 20 Z"),  # strap loop
    ),
    "p90": (
        ("p", "M0 15 Q0 6 10 6 L66 6 L76 9 L80 12 L80 18 L66 19 Q62 20 60 24 L58 29 Q57 31 54 31 L48 31 "
              "Q46 31 46 29 L46 23 L35 23 Q33 32 25 33 L10 33 Q0 33 0 22 Z "
              "M10 18 L23 18 Q24 18 23.5 20 Q22 27 17 27.5 L12 27.5 Q9.5 27 9.5 24 L9.5 19 Q9.5 18 10 18 Z "
              "M48.5 23 L57.5 22.5 Q56 27 54 28.5 L50 28.5 Q48.5 28.5 48.5 27 Z"),  # bullpup shell, thumbhole
        ("d", "M12 3 L66 3 L66 6 L12 6 Z"),  # the magazine on top
        ("d", "M80 13 L88 13 L88 16.5 L80 16.5 Z"),
        ("d", "M38 2 L46 2 L46 3 L38 3 Z"),
    ),
    "ump45": (
        ("d", "M0 8 L17 8 L17 10.5 L3 10.5 L3 24 L17 24 L17 26.5 L0 26.5 Z"),  # skeleton stock
        ("p", "M24 17.5 L32 17.5 L29 31 L21.5 30 Z"),
        ("d", _guard(32, 41, 17.5, 22.5)),
        ("p", "M45 17.5 L53 17.5 L55.5 37 L47.5 37.5 Z"),  # long straight magazine
        ("p", "M16 7 L66 7 L70 9.5 L70 17.5 L16 17.5 Z"),
        ("d", "M22 5.5 L62 5.5 L62 7 L22 7 Z"),
        ("d", "M70 11 L80 11 L80 14.5 L70 14.5 Z"),
    ),
    "nova": (
        ("p", "M0 13 L6 11 L29 13.5 L29 21 L23 22 Q16 24 6 27.5 L1 27.5 Q0 27.5 0 26 Z"),
        ("d", _guard(31, 40, 20.5, 25)),
        ("p", "M28 11.5 L52 11.5 L52 20.5 L28 21 Z"),
        ("d", "M52 12 L101 12 L101 15 L52 15 Z"),  # barrel
        ("d", "M52 15.5 L95 15.5 L95 18.5 L52 18.5 Z"),  # magazine tube
        ("p", "M60 14.5 L82 14.5 L82 21 L60 21 Z"),  # the pump
        ("s", "M63 17.5 L79 17.5 L79 18.3 L63 18.3 Z"),
        ("d", "M99 10.8 L100.5 10.8 L100.5 12 L99 12 Z"),
    ),
    "xm1014": _M4_STOCK + (
        ("d", "M30 19 L36 19 L34 29.5 L28.5 28.5 Z"),
        ("d", _guard(36, 44, 19, 23)),
        ("p", "M26 10 L56 10 L56 19 L26 19 Z"),
        ("d", "M48 7 L52 7 L52 10 L48 10 Z"),  # ghost ring
        ("d", "M56 11 L101 11 L101 14 L56 14 Z"),
        ("d", "M56 14.5 L94 14.5 L94 18 L56 18 Z"),
        ("p", "M58 13.5 L78 13.5 L78 19.5 L58 19.5 Z"),
    ),
    "negev": (
        ("p", "M0 10.5 L3 10 L22 12.5 L22 18.5 L3 23 L0 23 Z M5 13.8 L18 14.8 L18 17 L5 19.6 Z"),  # skeleton stock
        ("d", "M25 18 L31 18 L29 28 L23.5 27 Z"),
        ("d", _guard(31, 39, 18, 22.5)),
        ("p", "M40 18 L54 18 L54 28.5 L40 28.5 Z"),  # ammo box
        ("p", "M22 9 L60 9 L60 18 L22 18 Z"),  # receiver
        ("s", "M26 11.5 L58 11.5 L58 12.2 L26 12.2 Z"),
        ("d", "M45 3.5 L57 3.5 L57 5 L55 5 L55 9 L47 9 L47 5 L45 5 Z"),  # carry handle
        ("p", "M60 10 L86 10 L86 16.5 L60 16.5 Z O64 13.25 1.1 O68 13.25 1.1 O72 13.25 1.1 O76 13.25 1.1 "
              "O80 13.25 1.1"),  # vented barrel shroud
        ("d", "M64 16.5 L68.5 16.5 L67.5 25 L64.5 25 Z"),  # fore grip
        ("d", "M86 11.8 L104 11.8 L104 14.6 L86 14.6 Z"),
        ("d", "M86 15 L100 15.5 L100 16.8 L86 16.5 Z"),  # folded bipod
        ("d", "M98 8.5 L100 8.5 L100 11.8 L98 11.8 Z"),
        ("d", "M102 11 L107 11 L107 15.4 L102 15.4 Z"),
    ),
}

# --- knives (blade to the right, edge down) ------------------------------------------------------------------------

KNIVES = {
    "karambit": (("h", "M2 20 Q10 13 26 14 L36 16 L35 23 L22 23 Q12 23 8 28 Z"), ("h", "O4 27 7 O4 27 4"),
                 ("m", "O30 18.5 1.2 O22 18 1.2"),
                 ("p", "M35 15 Q52 10 64 21 Q71 31 67 43 Q62 33 53 27 Q44 22.5 35 23 Z")),
    "talon": (("h", "M2 18 Q12 13 30 15 L40 16 L39 23 L26 23 Q14 22 8 26 Z"), ("h", "O5 25 6 O5 25 3.5"),
              ("p", "M39 15.5 Q60 11 76 20 Q86 28 84 40 Q78 31 66 26 Q52 22.5 39 23 Z")),
    "butterfly": (("h", "M0 15.8 L40 15.3 L40 24.7 L0 24.2 Z M4 18.6 L36 18.4 L36 21.6 L4 21.4 Z"),  # split handles
                  ("m", "M-3.5 18.3 L1 18.3 L1 21.7 L-3.5 21.7 Z"),  # latch
                  ("p", "M40 16 L78 15.4 Q90 15.8 97 19.8 L84 23.4 L40 24.2 Z"),
                  ("m", "O40 17.5 1.4 O40 22.5 1.4")),
    "m9": (("h", "M0 16 Q-1.5 20 0 24 L34 25 L34 15 Z M3 19.3 L30 19.6 L30 20.4 L3 20.7 Z"),
           ("m", "M34 9 L38 9 L38 31 L34 31 Z O36 28.5 1.2"),
           ("p", "M38 13 L42 11 L44 13 L46 11 L48 13 L50 11 L52 13 L54 11 L56 13 L80 13 L95 18.5 L88 22 L74 25.5 "
                 "L38 26 Z M60 17 L70 17 Q72 19 70 21 L60 21 Q58 19 60 17 Z")),
    "bayonet": (("h", "M0 15.5 L34 15 L34 25 L0 24.5 Z M3 17.5 L31 17.3 L31 18.2 L3 18.4 Z M3 21.6 L31 21.8 "
                      "L31 22.7 L3 22.5 Z"),
                ("m", "M34 9.5 L37 9.5 L37 30.5 L34 30.5 Z O35.5 12 1.3"),
                ("p", "M37 14.5 L84 14 L97 19.5 L84 23.5 L37 24.5 Z M42 17.6 L80 17.3 L80 18.4 L42 18.7 Z")),
    "flip": (("h", "M0 15 Q-1.5 20 0 25 L40 24 L40 14 Z"),
             ("m", "O38 19 1.5"),
             ("p", "M40 14 L80 12 Q92 13 97 19.5 L82 24 L40 23.5 Z O47 17.5 2.2")),
    "gut": (("h", "M0 15 L34 14 L34 25 L0 24 Q-1.5 20 0 15 Z"), ("m", "M34 13 L37 13 L37 26 L34 26 Z"),
            ("p", "M37 14 L58 14 Q60 9.5 65 9.5 Q69 10 70 13.5 L66 13.8 Q64 12.5 62 15 L84 15 Q95 18 91 22 "
                  "L37 25 Z")),
    "huntsman": (("h", "M0 16 Q4 14 34 14 L34 26 Q26 25 22 28 Q16 25 10 28 Q4 26 0 25 Z"),
                 ("m", "M34 11 L38 11 L38 29 L34 29 Z"),
                 ("p", "M38 12 L46 10 L48 12 L50 10 L52 12 L54 10 L56 12 L66 11 L73 13 L96 18.5 L82 24 L38 26 Z")),
    "falchion": (("h", "M0 19 Q2 15 10 15 L36 15 L36 24 L10 24 Q2 25 0 19 Z"),
                 ("m", "M35 14 L38 14 L38 25 L35 25 Z"),
                 ("p", "M38 15 L72 14 Q90 12 99 6 Q97 18 82 24 L38 24 Z")),
}

_GLOVE = (("p", "W M10 40 L8 24 Q8 18 14 16 L20 4 Q22 1 25 3 L24 15 L28 2 Q31 0 33 3 L31 15 L35 4 Q38 2 40 5 "
                "L37 17 L41 9 Q44 7 46 10 L41 25 Q38 34 32 40 Z M11 27 Q4 21 2 15 Q2 10 6 11 L14 21 Z"),
          ("h", "M9 38 L33 38 L34 47 L8 47 Z"))

SHAPES: dict[str, tuple[tuple[str, str], ...]] = {**PISTOLS, **RIFLES, **KNIVES}
for _g in ("sport", "driver", "wraps", "moto", "specialist"):
    SHAPES[_g] = _GLOVE

# where the shooting hand holds each gun (x0, y0, x1, y1 in the shape's units), for the Aim Arena's view model
GRIPS = {"glock": (3, 17, 20, 30), "usps": (4, 17, 20, 30), "p250": (3, 17, 20, 30), "fiveseven": (3, 17, 20, 30),
         "deagle": (3, 19, 21, 32), "ak47": (31, 19, 39, 28), "m4a4": (28, 20, 36, 29), "m4a1s": (28, 20, 36, 29),
         "galil": (31, 19, 39, 28), "awp": (24, 21, 31, 28), "ssg08": (22, 20, 28, 26), "famas": (50, 20, 57, 29),
         "mp9": (23, 17, 32, 28), "mac10": (20, 17, 30, 28), "p90": (46, 23, 58, 30), "ump45": (22, 18, 32, 29),
         "nova": (24, 19, 30, 25), "xm1014": (28, 19, 35, 28), "negev": (24, 18, 31, 27)}
