"""Rail Rush screens drawn over the scene: the in-run HUD, the title screen, the shop (power-up upgrades, items, the
crew), missions, stats, the revive offer, game over and the lost parcel. Buttons are registered while drawing (mouse and
keyboard both use them)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QFontMetricsF, QImage, QLinearGradient, QPainter, QRadialGradient

from app.arcade.games.rail_rush import art
from app.arcade.games.rail_rush.art import NOBRUSH, NOPEN, Pose, pen, qc
from app.arcade.games.rail_rush.data import (BOX_PRICE, CHARACTERS, HOVERBOARD_PRICE, KEY_PRICE, MAX_MULTIPLIER,
                                             POWERUPS, UPGRADE_COSTS, UPGRADE_LEVELS, mission_done, mission_text,
                                             powerup_duration)

AL = Qt.AlignmentFlag
CENTER = AL.AlignCenter
STYLES = {  # top, bottom, lip
    "green": ("#5fe05a", "#22a336", "#13661f"),
    "blue": ("#4fb0ff", "#1f6fd6", "#123f80"),
    "orange": ("#ffcf3f", "#f0801a", "#9a4a08"),
    "red": ("#ff6b5b", "#d92b2b", "#7a1212"),
    "purple": ("#c18bff", "#7b3fd9", "#43207a"),
    "grey": ("#a7afb8", "#6b737d", "#3b4048"),
    "dark": ("#3b4250", "#252a33", "#12151a"),
}


@dataclass
class Button:
    rect: QRectF
    action: str
    enabled: bool = True


def f_(size: float, weight: QFont.Weight = QFont.Weight.Black) -> QFont:
    f = QFont("Segoe UI")
    f.setPixelSize(max(1, int(size)))
    f.setWeight(weight)
    return f


def outlined(p: QPainter, rect: QRectF, align, text: str, size: float, color: str = "#ffffff",  # noqa: ANN001
             outline: str = "#14161b", ow: float = 2.0, weight: QFont.Weight = QFont.Weight.Black) -> None:
    p.setFont(f_(size, weight))
    p.setPen(qc(outline))
    for dx, dy in ((-ow, 0), (ow, 0), (0, -ow), (0, ow), (ow, ow * 1.6)):
        p.drawText(rect.translated(dx, dy), align, text)
    p.setPen(qc(color))
    p.drawText(rect, align, text)


_GLYPHS: dict = {}


def number(p: QPainter, right: float, cy: float, text: str, size: float, color: str = "#ffffff") -> float:
    """Outlined digits right-aligned at ``right`` (cached glyph images: cheap every frame); returns the width."""
    key = (int(size), color)
    glyphs = _GLYPHS.get(key)
    if glyphs is None:
        if len(_GLYPHS) > 16:
            _GLYPHS.clear()
        glyphs = _GLYPHS[key] = {}
        fm = QFontMetricsF(f_(size))
        ow = max(1.5, size * 0.055)
        for ch in "0123456789,x":
            adv = fm.horizontalAdvance(ch)
            img = QImage(int(adv + ow * 4 + 2), int(fm.height() + ow * 4 + 2),
                         QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(Qt.GlobalColor.transparent)
            q = QPainter(img)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            q.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            r = QRectF(ow * 2, ow * 2, adv, fm.height())
            outlined(q, r, CENTER, ch, size, color, ow=ow)
            q.end()
            glyphs[ch] = (img, adv, ow * 2)
    total = sum(glyphs[ch][1] for ch in text if ch in glyphs)
    x = right - total
    for ch in text:
        g = glyphs.get(ch)
        if g is None:
            continue
        img, adv, pad = g
        p.drawImage(QPointF(x - pad, cy - img.height() / 2), img)
        x += adv
    return total


def panel(p: QPainter, rect: QRectF, color: str = "#1d2a44", alpha: int = 235, radius: float = 18) -> None:
    p.setPen(NOPEN)
    p.setBrush(qc("#000000", 90))
    p.drawRoundedRect(rect.translated(0, 6), radius, radius)
    g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    g.setColorAt(0, qc(color, alpha).lighter(125))
    g.setColorAt(1, qc(color, alpha))
    p.setBrush(g)
    p.setPen(pen(qc("#ffffff", 70), 2))
    p.drawRoundedRect(rect, radius, radius)


def pill(p: QPainter, rect: QRectF, icon: str, text: str, u: float) -> None:
    p.setPen(NOPEN)
    p.setBrush(qc("#000000", 120))
    p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
    r = rect.height() * 0.42
    cx, cy = rect.left() + rect.height() / 2, rect.center().y()
    if icon == "coin":
        art.draw_coin(p, cx, cy, r, 0.0)
    elif icon == "key":
        art.draw_heart(p, cx, cy, r * 1.05)
    elif icon == "board":
        art.draw_shield_icon(p, cx, cy, r * 1.05)
    outlined(p, rect.adjusted(rect.height() * 0.95, 0, -8, 0), AL.AlignVCenter | AL.AlignLeft, text,
             rect.height() * 0.52, ow=1.5)


def mult_badge(p: QPainter, rect: QRectF, mult: int, hot: bool = False) -> None:
    g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    if hot:
        g.setColorAt(0, qc("#9ff3ff"))
        g.setColorAt(1, qc("#2a8fff"))
    else:
        g.setColorAt(0, qc("#fff36b"))
        g.setColorAt(1, qc("#ff9a1a"))
    p.setPen(pen("#14161b", max(2.0, rect.height() * 0.07)))
    p.setBrush(g)
    p.drawRoundedRect(rect, rect.height() * 0.3, rect.height() * 0.3)
    outlined(p, rect, CENTER, f"x{mult}", rect.height() * 0.6, "#ffffff", ow=1.6)


def add_button(g, p: QPainter, rect: QRectF, label: str, action: str, style: str = "blue",  # noqa: ANN001
               enabled: bool = True, sub: str = "", size: float | None = None) -> None:
    idx = len(g.buttons)
    g.buttons.append(Button(rect, action, enabled))
    focused = idx == g.focus
    top, bot, lip = STYLES[style if enabled else "grey"]
    r = min(rect.height() * 0.32, 22)
    rr = QRectF(rect)
    if focused:
        grow = 3 + 2 * math.sin(g.anim_t * 6)
        rr = rect.adjusted(-grow, -grow, grow, grow)
        p.setPen(NOPEN)
        p.setBrush(qc("#ffffff", 200))
        p.drawRoundedRect(rr.adjusted(-4, -4, 4, 8), r + 4, r + 4)
    p.setPen(NOPEN)
    p.setBrush(qc(lip))
    p.drawRoundedRect(rr.translated(0, 5), r, r)
    gr = QLinearGradient(rr.topLeft(), rr.bottomLeft())
    gr.setColorAt(0, qc(top))
    gr.setColorAt(1, qc(bot))
    p.setBrush(gr)
    p.drawRoundedRect(rr, r, r)
    p.setBrush(qc("#ffffff", 60))
    p.drawRoundedRect(QRectF(rr.left() + 6, rr.top() + 4, rr.width() - 12, rr.height() * 0.32), r * 0.6, r * 0.6)
    size = size or rr.height() * (0.36 if sub else 0.44)
    if sub:
        outlined(p, QRectF(rr.left(), rr.top() + rr.height() * 0.06, rr.width(), rr.height() * 0.55), CENTER,
                 label, size)
        outlined(p, QRectF(rr.left(), rr.top() + rr.height() * 0.52, rr.width(), rr.height() * 0.4), CENTER, sub,
                 size * 0.62, "#fff7c0", ow=1.2, weight=QFont.Weight.Bold)
    else:
        outlined(p, rr, CENTER, label, size)


def dim(p: QPainter, w: float, h: float, alpha: int = 150) -> None:
    p.fillRect(QRectF(0, 0, w, h), qc("#0b1020", alpha))


# --- in the run ----------------------------------------------------------------------------------------------------

def hud(g, p: QPainter) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = min(w / 1280, h / 780)
    wd = g.world
    m = 16 * u
    # score and multiplier
    score = int(wd.score)
    sz = 46 * u
    tw = number(p, w - m, m + sz * 0.6, f"{score:,}", sz)
    hot = wd.power.get("double", 0.0) > 0
    mult = wd.multiplier * (2 if hot else 1)
    bw = 70 * u
    mult_badge(p, QRectF(w - m - tw - bw - 14 * u, m + sz * 0.15, bw, sz * 0.9), mult, hot)
    # coins
    cr = QRectF(w - m - 200 * u, m + sz * 1.35, 200 * u, 40 * u)
    cw = number(p, w - m, cr.center().y(), f"{wd.run_coins:,}", 32 * u, "#ffe35a")
    art.draw_coin(p, w - m - cw - 24 * u, cr.center().y(), 15 * u, 0.5 * math.sin(g.anim_t * 3))
    # power-up timers
    y = m + 4 * u
    timers = []
    for kind in ("jetpack", "sneakers", "magnet", "double"):
        left = wd.power.get(kind, 0.0)
        if left > 0:
            timers.append((kind, left / max(wd.duration(kind), 0.1)))
    if wd.board > 0:
        timers.append(("hoverboard", wd.board / 30.0))
    for kind, frac in timers:
        r = 26 * u
        cx, cy = m + r, y + r
        p.setPen(NOPEN)
        p.setBrush(qc("#000000", 130))
        p.drawRoundedRect(QRectF(m - 4 * u, y - 4 * u, 210 * u, 2 * r + 8 * u), r, r)
        p.setBrush(qc("#ffffff", 230))
        p.drawEllipse(QPointF(cx, cy), r, r)
        art.draw_icon(p, kind, cx, cy, r * 0.72)
        bar = QRectF(cx + r + 8 * u, cy - 8 * u, 140 * u, 16 * u)
        p.setBrush(qc("#000000", 160))
        p.drawRoundedRect(bar, 8 * u, 8 * u)
        warn = frac < 0.25 and int(g.anim_t * 6) % 2 == 0
        col = "#ff5b4a" if warn else {"jetpack": "#4fd2ff", "sneakers": "#ff5f8a", "magnet": "#ff7a3a",
                                      "double": "#ffd23f", "hoverboard": "#7ffcff"}[kind]
        p.setBrush(qc(col))
        p.drawRoundedRect(QRectF(bar.left(), bar.top(), max(bar.height(), bar.width() * frac), bar.height()),
                          8 * u, 8 * u)
        y += 2 * r + 16 * u
    # shield reminder
    if wd.board <= 0 and g.profile.hoverboards > 0 and wd.running and not wd.dead:
        rr = QRectF(m, h - m - 46 * u, 200 * u, 46 * u)
        pill(p, rr, "board", f"x{g.profile.hoverboards}   [H]", u)
    # start hint
    if wd.running and wd.run_time < 4.0 and not wd.dead:
        a = int(255 * min(1.0, (4.0 - wd.run_time)))
        p.setOpacity(a / 255)
        outlined(p, QRectF(0, h * 0.9 - 30 * u, w, 40 * u), CENTER,
                 "Left / Right: lanes    Up: jump    Down: roll    H: shield", 22 * u, ow=1.5)
        p.setOpacity(1.0)
    # new high score banner
    if g.hs_banner > 0:
        k = min(1.0, (2.8 - g.hs_banner) * 4, g.hs_banner * 3)
        y0 = h * 0.2
        rr = QRectF(w / 2 - 260 * u, y0 - 40 * u, 520 * u, 80 * u)
        p.setOpacity(max(0.0, min(1.0, k)))
        p.save()
        p.translate(rr.center())
        sc = 0.8 + 0.2 * min(1.0, k)
        p.scale(sc, sc)
        p.translate(-rr.center())
        gr = QLinearGradient(rr.topLeft(), rr.bottomLeft())
        gr.setColorAt(0, qc("#ff5fae"))
        gr.setColorAt(1, qc("#b42a7a"))
        p.setPen(pen("#14161b", 4))
        p.setBrush(gr)
        p.drawRoundedRect(rr, 20 * u, 20 * u)
        outlined(p, rr, CENTER, "NEW HIGH SCORE!", 40 * u, "#fff36b", ow=2.5)
        p.restore()
        p.setOpacity(1.0)


def toasts(g, p: QPainter) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = min(w / 1280, h / 780)
    y = h * 0.3
    for toast in g.toasts:
        left, dur = toast["t"], toast["dur"]
        k = min(1.0, (dur - left) * 5, left * 3)
        p.setOpacity(max(0.0, min(1.0, k)))
        p.setFont(f_(26 * u))
        tw = p.fontMetrics().horizontalAdvance(toast["text"]) + 60 * u
        rr = QRectF(w / 2 - tw / 2, y, tw, 50 * u)
        p.setPen(pen("#14161b", 3))
        p.setBrush(qc(toast.get("color", "#2c6fd8"), 235))
        p.drawRoundedRect(rr, 25 * u, 25 * u)
        outlined(p, rr, CENTER, toast["text"], 26 * u, ow=1.6)
        y += 60 * u
    p.setOpacity(1.0)


# --- title --------------------------------------------------------------------------------------------------------

def top_bar(g, p: QPainter, u: float) -> None:  # noqa: ANN001
    m = 16 * u
    pr = g.profile
    pill(p, QRectF(m, m, 190 * u, 46 * u), "coin", f"{pr.coins:,}", u)
    pill(p, QRectF(m + 200 * u, m, 110 * u, 46 * u), "key", f"{pr.keys}", u)
    pill(p, QRectF(m + 320 * u, m, 110 * u, 46 * u), "board", f"{pr.hoverboards}", u)


def title(g, p: QPainter) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = min(w / 1280, h / 780)
    g2 = QLinearGradient(0, 0, 0, h * 0.3)
    g2.setColorAt(0, qc("#000000", 70))
    g2.setColorAt(1, qc("#000000", 0))
    p.fillRect(QRectF(0, 0, w, h * 0.3), g2)
    art.draw_logo(p, w / 2, h * 0.17, h * 0.13, g.anim_t)
    top_bar(g, p, u)
    pr = g.profile
    m = 16 * u
    outlined(p, QRectF(w - m - 400 * u, m, 400 * u, 30 * u), AL.AlignRight | AL.AlignVCenter, "BEST", 20 * u,
             "#fff36b", ow=1.5)
    outlined(p, QRectF(w - m - 400 * u, m + 26 * u, 400 * u, 44 * u), AL.AlignRight | AL.AlignVCenter,
             f"{pr.best:,}", 38 * u, ow=2)
    mult_badge(p, QRectF(w - m - 84 * u, m + 78 * u, 84 * u, 44 * u), pr.multiplier)
    # the character's name plate
    ch = g.char
    outlined(p, QRectF(0, h * 0.73, w, 36 * u), CENTER, ch["name"].upper(), 30 * u, "#ffffff", ow=2)
    # buttons
    pulse = 1.0 + 0.04 * math.sin(g.anim_t * 5)
    bw, bh = 330 * u * pulse, 74 * u * pulse
    add_button(g, p, QRectF(w / 2 - bw / 2, h * 0.79, bw, bh), "PLAY", "play", "green", sub="press Space")
    names = (("SHOP", "shop", "orange"), ("MISSIONS", "missions", "purple"), ("STATS", "stats", "blue"),
             ("CONTROLS", "controls", "green"), ("ARCADE", "arcade", "dark"))
    sw, sh, gap = min(170 * u, (w - 40 * u) / 5 - 12 * u), 52 * u, 12 * u
    x = w / 2 - (len(names) * sw + (len(names) - 1) * gap) / 2
    for label, action, style in names:
        add_button(g, p, QRectF(x, h * 0.79 + 92 * u, sw, sh), label, action, style)
        x += sw + gap
    # missions summary card
    card = QRectF(m, h * 0.36, 300 * u, 190 * u)
    if w > 1000 * u:
        panel(p, card, "#1d2a44", 200, 14 * u)
        outlined(p, QRectF(card.left() + 14 * u, card.top() + 8 * u, card.width(), 30 * u),
                 AL.AlignLeft | AL.AlignVCenter, "MISSIONS", 22 * u, "#fff36b", ow=1.5)
        y = card.top() + 44 * u
        for mi in pr.missions:
            done = mission_done(mi)
            p.setPen(NOPEN)
            p.setBrush(qc("#3bd16f" if done else "#ffffff", 230 if done else 60))
            p.drawEllipse(QPointF(card.left() + 24 * u, y + 13 * u), 9 * u, 9 * u)
            p.setFont(f_(15 * u, QFont.Weight.DemiBold))
            p.setPen(qc("#ffffff" if not done else "#9ff0b8"))
            p.drawText(QRectF(card.left() + 42 * u, y, card.width() - 52 * u, 40 * u),
                       AL.AlignLeft | AL.AlignVCenter | Qt.TextFlag.TextWordWrap, mission_text(mi))
            y += 46 * u


# --- shop ----------------------------------------------------------------------------------------------------------

def shop(g, p: QPainter) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = min(w / 1280, h / 780)
    dim(p, w, h, 170)
    top_bar(g, p, u)
    outlined(p, QRectF(0, 14 * u, w, 54 * u), CENTER, "SHOP", 46 * u, "#fff36b", ow=2.5)
    tabs = (("POWER-UPS", "upgrades"), ("ITEMS", "items"), ("COURIERS", "crew"))
    tw, th = 200 * u, 50 * u
    x = w / 2 - (3 * tw + 2 * 12 * u) / 2
    for label, key in tabs:
        add_button(g, p, QRectF(x, 80 * u, tw, th), label, f"tab:{key}", "orange" if g.shop_tab == key else "dark")
        x += tw + 12 * u
    body = QRectF(w / 2 - 560 * u, 150 * u, 1120 * u, h - 250 * u)
    panel(p, body, "#1d2a44", 225, 18 * u)
    pr = g.profile
    if g.shop_tab == "upgrades":
        rh = min(118 * u, (body.height() - 24 * u) / 4)
        y = body.top() + 12 * u
        for kind, (name, _base, _step, blurb) in POWERUPS.items():
            lvl = pr.upgrades[kind]
            row = QRectF(body.left() + 14 * u, y, body.width() - 28 * u, rh - 10 * u)
            p.setPen(NOPEN)
            p.setBrush(qc("#ffffff", 22))
            p.drawRoundedRect(row, 12 * u, 12 * u)
            r = row.height() * 0.36
            p.setBrush(qc("#ffffff", 235))
            p.drawEllipse(QPointF(row.left() + r + 14 * u, row.center().y()), r, r)
            art.draw_icon(p, kind, row.left() + r + 14 * u, row.center().y(), r * 0.72)
            tx = row.left() + 2 * r + 34 * u
            outlined(p, QRectF(tx, row.top() + 6 * u, 500 * u, row.height() * 0.4), AL.AlignLeft | AL.AlignVCenter,
                     name, 26 * u, ow=1.6)
            p.setFont(f_(15 * u, QFont.Weight.DemiBold))
            p.setPen(qc("#cfe3ff"))
            cur, nxt = powerup_duration(kind, lvl), powerup_duration(kind, lvl + 1)
            txt = blurb + (f"  {cur:.0f}s -> {nxt:.0f}s" if lvl < UPGRADE_LEVELS else f"  {cur:.0f}s (max)")
            p.drawText(QRectF(tx, row.top() + row.height() * 0.42, 560 * u, row.height() * 0.3),
                       AL.AlignLeft | AL.AlignVCenter, txt)
            for i in range(UPGRADE_LEVELS):
                pip = QRectF(tx + i * 34 * u, row.top() + row.height() * 0.74, 28 * u, 10 * u)
                p.setBrush(qc("#3bd16f") if i < lvl else qc("#ffffff", 50))
                p.drawRoundedRect(pip, 4 * u, 4 * u)
            bw2 = 190 * u
            br = QRectF(row.right() - bw2 - 12 * u, row.center().y() - 30 * u, bw2, 60 * u)
            if lvl >= UPGRADE_LEVELS:
                add_button(g, p, br, "MAX", f"up:{kind}", "grey", enabled=False)
            else:
                cost = UPGRADE_COSTS[lvl]
                add_button(g, p, br, "UPGRADE", f"up:{kind}", "green", enabled=pr.coins >= cost, sub=f"{cost:,} coins")
            y += rh
    elif g.shop_tab == "items":
        items = (("hoverboard", "Bubble Shield", "Press H in a run: a bubble for 30 s that takes one crash for you.",
                  HOVERBOARD_PRICE, f"Owned: {pr.hoverboards}"),
                 ("key", "Heart", "Spend hearts to revive and carry on after a crash (1, 2, 4, 8... per run).",
                  KEY_PRICE, f"Owned: {pr.keys}"),
                 ("mystery", "Lost Parcel", "Coins, shields, hearts - or the jackpot.", BOX_PRICE, ""))
        rh = min(140 * u, (body.height() - 24 * u) / 3)
        y = body.top() + 12 * u
        for kind, name, blurb, price, owned in items:
            row = QRectF(body.left() + 14 * u, y, body.width() - 28 * u, rh - 12 * u)
            p.setPen(NOPEN)
            p.setBrush(qc("#ffffff", 22))
            p.drawRoundedRect(row, 12 * u, 12 * u)
            r = row.height() * 0.36
            p.setBrush(qc("#ffffff", 235))
            p.drawEllipse(QPointF(row.left() + r + 14 * u, row.center().y()), r, r)
            art.draw_icon(p, kind, row.left() + r + 14 * u, row.center().y(), r * 0.72)
            tx = row.left() + 2 * r + 34 * u
            outlined(p, QRectF(tx, row.top() + 8 * u, 600 * u, row.height() * 0.38), AL.AlignLeft | AL.AlignVCenter,
                     name, 26 * u, ow=1.6)
            p.setFont(f_(15 * u, QFont.Weight.DemiBold))
            p.setPen(qc("#cfe3ff"))
            p.drawText(QRectF(tx, row.top() + row.height() * 0.45, 640 * u, row.height() * 0.25),
                       AL.AlignLeft | AL.AlignVCenter, blurb)
            p.setPen(qc("#fff36b"))
            p.drawText(QRectF(tx, row.top() + row.height() * 0.7, 640 * u, row.height() * 0.25),
                       AL.AlignLeft | AL.AlignVCenter, owned)
            bw2 = 190 * u
            br = QRectF(row.right() - bw2 - 12 * u, row.center().y() - 30 * u, bw2, 60 * u)
            add_button(g, p, br, "OPEN" if kind == "mystery" else "BUY", f"buy:{kind}", "green",
                       enabled=pr.coins >= price, sub=f"{price:,} coins")
            y += rh
    else:
        n = len(CHARACTERS)
        cw = (body.width() - 28 * u - (n - 1) * 12 * u) / n
        for i, ch in enumerate(CHARACTERS):
            card = QRectF(body.left() + 14 * u + i * (cw + 12 * u), body.top() + 14 * u, cw, body.height() - 28 * u)
            sel = pr.selected == ch["id"]
            owned = ch["id"] in pr.unlocked
            p.setPen(pen(qc("#fff36b") if sel else qc("#ffffff", 40), 4 if sel else 2))
            gr = QLinearGradient(card.topLeft(), card.bottomLeft())
            gr.setColorAt(0, qc(ch["top"]).lighter(120))
            gr.setColorAt(1, qc(ch["top"]).darker(220))
            p.setBrush(gr)
            p.drawRoundedRect(card, 14 * u, 14 * u)
            p.setPen(NOPEN)
            p.setBrush(qc("#000000", 60))
            p.drawEllipse(QPointF(card.center().x(), card.top() + card.height() * 0.6), cw * 0.32, cw * 0.07)
            s = min(card.height() * 0.42 / 2.0, cw * 0.55)
            pose = Pose(mode="idle", t=g.anim_t + i, front=True)
            if not owned:
                p.setOpacity(0.55)
            art.draw_runner(p, card.center().x(), card.top() + card.height() * 0.6, s, ch, pose)
            p.setOpacity(1.0)
            outlined(p, QRectF(card.left(), card.top() + 8 * u, cw, 40 * u), CENTER, ch["name"].upper(), 26 * u,
                     ow=1.6)
            p.setFont(f_(13 * u, QFont.Weight.DemiBold))
            p.setPen(qc("#e6efff"))
            p.drawText(QRectF(card.left() + 8 * u, card.top() + card.height() * 0.63, cw - 16 * u,
                              card.height() * 0.16), CENTER | Qt.TextFlag.TextWordWrap, ch["blurb"])
            br = QRectF(card.left() + 10 * u, card.bottom() - 70 * u, cw - 20 * u, 56 * u)
            if sel:
                add_button(g, p, br, "SELECTED", f"char:{ch['id']}", "orange")
            elif owned:
                add_button(g, p, br, "SELECT", f"char:{ch['id']}", "blue")
            else:
                add_button(g, p, br, "UNLOCK", f"char:{ch['id']}", "green", enabled=pr.coins >= ch["price"],
                           sub=f"{ch['price']:,}")
    add_button(g, p, QRectF(w / 2 - 110 * u, h - 86 * u, 220 * u, 60 * u), "BACK", "back", "red")


def box_overlay(g, p: QPainter) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = min(w / 1280, h / 780)
    dim(p, w, h, 190)
    bx = g.box
    t = bx["t"]
    cx, cy = w / 2, h * 0.48
    r = 110 * u
    if t < 1.3:
        shake = math.sin(t * 50) * 10 * u * min(1.0, t * 2)
        p.save()
        p.translate(cx + shake, cy)
        p.rotate(math.sin(t * 40) * 6 * min(1.0, t))
        art.draw_icon(p, "mystery", 0, 0, r)
        p.restore()
        outlined(p, QRectF(0, h * 0.18, w, 60 * u), CENTER, "LOST PARCEL", 48 * u, "#ffcf7a", ow=2.5)
    else:
        k = min(1.0, (t - 1.3) * 4)
        p.save()
        p.translate(cx, cy)
        p.rotate(t * 25)
        for _ in range(12):
            p.rotate(30)
            p.setPen(NOPEN)
            p.setBrush(qc("#fff36b", 70))
            p.drawPolygon([QPointF(0, 0), QPointF(-40 * u, -400 * u), QPointF(40 * u, -400 * u)])
        p.restore()
        glow = QRadialGradient(cx, cy, r * 1.8)
        glow.setColorAt(0, qc("#ffffff", 200))
        glow.setColorAt(1, qc("#ffffff", 0))
        p.setPen(NOPEN)
        p.setBrush(glow)
        p.drawEllipse(QPointF(cx, cy), r * 1.8, r * 1.8)
        kind = bx["kind"]
        p.save()
        p.translate(cx, cy)
        p.scale(k, k)
        art.draw_icon(p, {"coins": "coin", "hoverboard": "hoverboard", "key": "key"}[kind], 0, 0, r * 0.9)
        p.restore()
        outlined(p, QRectF(0, h * 0.18, w, 60 * u), CENTER, "YOU GOT", 40 * u, "#ffffff", ow=2.5)
        outlined(p, QRectF(0, cy + r * 1.1, w, 70 * u), CENTER, bx["label"], 52 * u, "#fff36b", ow=3)
        add_button(g, p, QRectF(w / 2 - 120 * u, h * 0.82, 240 * u, 64 * u), "COLLECT", "box_close", "green")


# --- missions and stats ----------------------------------------------------------------------------------------------

def missions(g, p: QPainter) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = min(w / 1280, h / 780)
    dim(p, w, h, 170)
    pr = g.profile
    outlined(p, QRectF(0, 20 * u, w, 60 * u), CENTER, "MISSIONS", 48 * u, "#d9a8ff", ow=2.5)
    body = QRectF(w / 2 - 460 * u, 110 * u, 920 * u, h - 220 * u)
    panel(p, body, "#2a1d44", 225, 18 * u)
    mult_badge(p, QRectF(body.left() + 24 * u, body.top() + 20 * u, 110 * u, 64 * u), pr.multiplier)
    p.setFont(f_(18 * u, QFont.Weight.DemiBold))
    p.setPen(qc("#ffffff"))
    nxt = min(MAX_MULTIPLIER, pr.multiplier + 1)
    msg = (f"Mission set {pr.mission_set + 1}: complete all three to raise your score multiplier to x{nxt}."
           if pr.multiplier < MAX_MULTIPLIER else "Top multiplier reached - x30!")
    p.drawText(QRectF(body.left() + 150 * u, body.top() + 20 * u, body.width() - 170 * u, 64 * u),
               AL.AlignLeft | AL.AlignVCenter | Qt.TextFlag.TextWordWrap, msg)
    ch = (body.height() - 120 * u) / 3
    y = body.top() + 104 * u
    for mi in pr.missions:
        done = mission_done(mi)
        card = QRectF(body.left() + 20 * u, y, body.width() - 40 * u, ch - 14 * u)
        p.setPen(NOPEN)
        p.setBrush(qc("#3bd16f", 70) if done else qc("#ffffff", 24))
        p.drawRoundedRect(card, 14 * u, 14 * u)
        cr = min(card.height() * 0.32, 30 * u)
        cxy = QPointF(card.left() + cr + 18 * u, card.center().y())
        p.setBrush(qc("#3bd16f") if done else qc("#ffffff", 50))
        p.drawEllipse(cxy, cr, cr)
        if done:
            p.setPen(pen("#ffffff", 5 * u))
            p.drawPolyline([QPointF(cxy.x() - cr * 0.45, cxy.y()), QPointF(cxy.x() - cr * 0.1, cxy.y() + cr * 0.4),
                            QPointF(cxy.x() + cr * 0.5, cxy.y() - cr * 0.35)])
        tx = card.left() + 2 * cr + 36 * u
        outlined(p, QRectF(tx, card.top() + 6 * u, card.width() - tx + card.left() - 20 * u, card.height() * 0.5),
                 AL.AlignLeft | AL.AlignVCenter, mission_text(mi), 24 * u, ow=1.5)
        bar = QRectF(tx, card.top() + card.height() * 0.62, card.width() - (tx - card.left()) - 30 * u, 16 * u)
        frac = min(1.0, mi["progress"] / max(1, mi["target"]))
        p.setPen(NOPEN)
        p.setBrush(qc("#000000", 120))
        p.drawRoundedRect(bar, 8 * u, 8 * u)
        p.setBrush(qc("#3bd16f" if done else "#ffd23f"))
        if frac > 0:
            p.drawRoundedRect(QRectF(bar.left(), bar.top(), max(bar.height(), bar.width() * frac), bar.height()),
                              8 * u, 8 * u)
        p.setFont(f_(14 * u, QFont.Weight.Bold))
        p.setPen(qc("#ffffff"))
        p.drawText(bar.adjusted(0, -22 * u, 0, -18 * u), AL.AlignRight | AL.AlignVCenter,
                   f"{min(mi['progress'], mi['target']):,} / {mi['target']:,}")
        y += ch
    add_button(g, p, QRectF(w / 2 - 110 * u, h - 90 * u, 220 * u, 60 * u), "BACK", "back", "red")


def stats(g, p: QPainter) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = min(w / 1280, h / 780)
    dim(p, w, h, 170)
    pr = g.profile
    st = pr.stats
    outlined(p, QRectF(0, 20 * u, w, 60 * u), CENTER, "STATS", 48 * u, "#9ff3ff", ow=2.5)
    body = QRectF(w / 2 - 460 * u, 110 * u, 920 * u, h - 220 * u)
    panel(p, body, "#16324a", 225, 18 * u)
    mins = int(st["play_seconds"] // 60)
    rows = (("High score", f"{pr.best:,}"), ("Runs", f"{st['runs']:,}"),
            ("Total distance", f"{st['distance']:,} m"), ("Longest run", f"{st['best_distance']:,} m"),
            ("Coins collected", f"{st['coins']:,}"), ("Most coins in a run", f"{st['best_coins']:,}"),
            ("Jumps", f"{st['jumps']:,}"), ("Rolls", f"{st['rolls']:,}"),
            ("Crashes", f"{st['crashes']:,}"), ("Stumbles", f"{st['stumbles']:,}"),
            ("Shields used", f"{st['hoverboards']:,}"), ("Power-ups", f"{st['powerups']:,}"),
            ("Balloon rides", f"{st['jetpacks']:,}"), ("Parcels opened", f"{st['boxes']:,}"),
            ("Missions done", f"{st['missions']:,}"), ("Time running", f"{mins // 60}h {mins % 60:02d}m"),
            ("Score multiplier", f"x{pr.multiplier}"), ("Crew unlocked", f"{len(pr.unlocked)} / {len(CHARACTERS)}"))
    cols = 2
    per = math.ceil(len(rows) / cols)
    rh = (body.height() - 40 * u) / per
    cw = (body.width() - 60 * u) / cols
    for i, (k, v) in enumerate(rows):
        c, r = divmod(i, per)
        rr = QRectF(body.left() + 20 * u + c * (cw + 20 * u), body.top() + 20 * u + r * rh, cw, rh - 4 * u)
        p.setPen(NOPEN)
        p.setBrush(qc("#ffffff", 18 if r % 2 == 0 else 8))
        p.drawRoundedRect(rr, 8 * u, 8 * u)
        p.setFont(f_(min(18 * u, rh * 0.45), QFont.Weight.DemiBold))
        p.setPen(qc("#cfe3ff"))
        p.drawText(rr.adjusted(14 * u, 0, 0, 0), AL.AlignLeft | AL.AlignVCenter, k)
        p.setFont(f_(min(20 * u, rh * 0.5)))
        p.setPen(qc("#ffffff"))
        p.drawText(rr.adjusted(0, 0, -14 * u, 0), AL.AlignRight | AL.AlignVCenter, v)
    add_button(g, p, QRectF(w / 2 - 110 * u, h - 90 * u, 220 * u, 60 * u), "BACK", "back", "red")


# --- the end of a run ------------------------------------------------------------------------------------------------

def save_me(g, p: QPainter) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = min(w / 1280, h / 780)
    dim(p, w, h, 120)
    cx, cy = w / 2, h * 0.46
    r = 120 * u
    frac = max(0.0, g.saveme_t / g.SAVE_TIME)
    cost = g.save_cost
    can = g.profile.keys >= cost
    p.setPen(NOPEN)
    p.setBrush(qc("#000000", 120))
    p.drawEllipse(QPointF(cx, cy), r + 18 * u, r + 18 * u)
    p.setBrush(NOBRUSH)
    p.setPen(pen("#ffffff", 12 * u, Qt.PenCapStyle.RoundCap))
    p.drawArc(QRectF(cx - r - 9 * u, cy - r - 9 * u, 2 * r + 18 * u, 2 * r + 18 * u), 90 * 16,
              int(frac * 360 * 16))
    bt = QRectF(cx - r, cy - r, 2 * r, 2 * r)
    idx = len(g.buttons)
    g.buttons.append(Button(bt, "saveme", can))
    gr = QRadialGradient(cx - r * 0.3, cy - r * 0.4, r * 1.6)
    gr.setColorAt(0, qc("#9ff3ff" if can else "#c9cdd6"))
    gr.setColorAt(1, qc("#1f6fd6" if can else "#6b737d"))
    p.setPen(pen("#14161b" if g.focus != idx else "#ffffff", 5 * u))
    p.setBrush(gr)
    p.drawEllipse(QPointF(cx, cy), r, r)
    outlined(p, QRectF(cx - r, cy - r * 0.62, 2 * r, r * 0.5), CENTER, "REVIVE!", 34 * u, ow=2)
    art.draw_heart(p, cx - 26 * u, cy + r * 0.2, 30 * u)
    outlined(p, QRectF(cx, cy + r * 0.2 - 30 * u, r, 60 * u), AL.AlignLeft | AL.AlignVCenter, f"x{cost}", 40 * u,
             ow=2)
    sub = f"You have {g.profile.keys} heart{'s' if g.profile.keys != 1 else ''}" if can else \
        f"Not enough hearts ({g.profile.keys}/{cost})"
    outlined(p, QRectF(0, cy + r + 30 * u, w, 36 * u), CENTER, sub, 24 * u, "#fff36b" if can else "#ffb0a0", ow=1.6)
    outlined(p, QRectF(0, cy - r - 90 * u, w, 50 * u), CENTER, "Space: revive      N: skip", 22 * u, ow=1.5)
    add_button(g, p, QRectF(w / 2 - 100 * u, cy + r + 80 * u, 200 * u, 56 * u), "SKIP", "skip", "red")


def game_over(g, p: QPainter) -> None:  # noqa: ANN001
    w, h = g.width(), g.height()
    u = min(w / 1280, h / 780)
    dim(p, w, h, 150)
    last = g.last_run
    body = QRectF(w / 2 - 330 * u, h * 0.05, 660 * u, min(h * 0.74, 610 * u))
    panel(p, body, "#1d2a44", 235, 22 * u)
    outlined(p, QRectF(body.left(), body.top() + 10 * u, body.width(), 50 * u), CENTER, "GAME OVER", 40 * u,
             "#ff6b5b", ow=2.5)
    outlined(p, QRectF(body.left(), body.top() + 66 * u, body.width(), 30 * u), CENTER, "SCORE", 22 * u, "#cfe3ff",
             ow=1.4)
    outlined(p, QRectF(body.left(), body.top() + 92 * u, body.width(), 80 * u), CENTER, f"{last['score']:,}", 70 * u,
             ow=3)
    if last["new_best"]:
        k = 1.0 + 0.06 * math.sin(g.anim_t * 6)
        rr = QRectF(body.center().x() - 160 * u * k, body.top() + 172 * u, 320 * u * k, 46 * u * k)
        p.setPen(pen("#14161b", 3))
        p.setBrush(qc("#ff5fae"))
        p.drawRoundedRect(rr, 20 * u, 20 * u)
        outlined(p, rr, CENTER, "NEW HIGH SCORE!", 24 * u, "#fff36b", ow=1.6)
    y = body.top() + 232 * u
    rows = (("coin", "Coins", f"{last['coins']:,}"), ("", "Distance", f"{last['distance']:,} m"),
            ("", "Best", f"{g.profile.best:,}"), ("", "Multiplier", f"x{last['mult']}"))
    for icon, k, v in rows:
        rr = QRectF(body.left() + 40 * u, y, body.width() - 80 * u, 40 * u)
        p.setPen(NOPEN)
        p.setBrush(qc("#ffffff", 20))
        p.drawRoundedRect(rr, 10 * u, 10 * u)
        if icon == "coin":
            art.draw_coin(p, rr.left() + 22 * u, rr.center().y(), 13 * u, 0.0)
        p.setFont(f_(19 * u, QFont.Weight.DemiBold))
        p.setPen(qc("#cfe3ff"))
        p.drawText(rr.adjusted(44 * u, 0, 0, 0), AL.AlignLeft | AL.AlignVCenter, k)
        p.setFont(f_(22 * u))
        p.setPen(qc("#ffffff"))
        p.drawText(rr.adjusted(0, 0, -16 * u, 0), AL.AlignRight | AL.AlignVCenter, v)
        y += 46 * u
    y += 4 * u
    outlined(p, QRectF(body.left() + 40 * u, y, body.width(), 28 * u), AL.AlignLeft | AL.AlignVCenter,
             f"MISSIONS  -  multiplier x{g.profile.multiplier}", 18 * u, "#d9a8ff", ow=1.3)
    y += 32 * u
    for mi in g.profile.missions:
        done = mission_done(mi)
        rr = QRectF(body.left() + 40 * u, y, body.width() - 80 * u, 28 * u)
        p.setPen(NOPEN)
        p.setBrush(qc("#3bd16f") if done else qc("#ffffff", 60))
        p.drawEllipse(QPointF(rr.left() + 10 * u, rr.center().y()), 7 * u, 7 * u)
        p.setFont(f_(15 * u, QFont.Weight.DemiBold))
        p.setPen(qc("#9ff0b8" if done else "#ffffff"))
        p.drawText(rr.adjusted(26 * u, 0, 0, 0), AL.AlignLeft | AL.AlignVCenter, mission_text(mi))
        p.drawText(rr, AL.AlignRight | AL.AlignVCenter, f"{min(mi['progress'], mi['target']):,} / {mi['target']:,}")
        y += 30 * u
    bw, bh = 220 * u, 64 * u
    yb = body.bottom() + 20 * u
    add_button(g, p, QRectF(w / 2 - bw * 1.5 - 16 * u, yb, bw, bh), "PLAY AGAIN", "again", "green", sub="Space")
    add_button(g, p, QRectF(w / 2 - bw / 2, yb, bw, bh), "SHOP", "shop", "orange")
    add_button(g, p, QRectF(w / 2 + bw / 2 + 16 * u, yb, bw, bh), "MAIN MENU", "menu", "blue", sub="M")
    add_button(g, p, QRectF(w / 2 - 90 * u, yb + bh + 18 * u, 180 * u, 42 * u), "CONTROLS (C)", "controls", "dark")
