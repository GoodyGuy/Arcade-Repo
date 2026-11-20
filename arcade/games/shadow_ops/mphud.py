"""The multiplayer HUD: a rotating minimap (allies, enemies that fire or are spotted by the Recon Drone), the team
score and clock, the killfeed, score pop-ups and medals, scorestreak icons, the death screen and the scoreboard."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF

from . import hud
from .data import STREAKS, level_progress
from .hud import ORANGE, text

ALLY = QColor(90, 190, 255)
ENEMY = QColor(255, 80, 60)
_MINIMAPS: dict = {}


def minimap_image(m) -> QImage:  # noqa: ANN001 - MapData
    img = _MINIMAPS.get(m.key)
    if img is None:
        k = 8
        img = QImage(m.w * k, m.h * k, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(30, 34, 30, 220))
        p = QPainter(img)
        for y, row in enumerate(m.rows):
            for x, ch in enumerate(row):
                if m.solid[y, x]:
                    col = QColor(200, 200, 190, 235)
                elif m.glass[y, x] >= 0:
                    col = QColor(150, 150, 140, 160)
                elif ch in "_":
                    col = QColor(60, 62, 64, 230)
                elif m.ceil[y, x] >= 0:
                    col = QColor(92, 80, 64, 230)
                else:
                    col = QColor(52, 72, 44, 230)
                p.fillRect(x * k, y * k, k, k, col)
        p.end()
        img = _MINIMAPS[m.key] = img
    return img


def draw_minimap(p: QPainter, r: QRectF, m, t: float) -> None:  # noqa: ANN001 - MPMode
    pl = m.player
    k = 8
    zoom = r.width() / (14 * k)
    p.save()
    clip = QPainterPath()
    clip.addRoundedRect(r, 10, 10)
    p.setClipPath(clip)
    p.fillRect(r, QColor(10, 12, 10, 200))
    p.translate(r.center())
    p.rotate(-math.degrees(pl.angle) - 90)
    p.scale(zoom, zoom)
    p.translate(-pl.x * k, -pl.y * k)
    p.drawImage(0, 0, minimap_image(m.m))
    p.setPen(Qt.PenStyle.NoPen)
    for b in m.bots:
        if b.dead or b.team != "ally":
            continue
        p.save()
        p.translate(b.x * k, b.y * k)
        p.rotate(math.degrees(b.angle) + 90)
        p.setBrush(ALLY)
        p.drawPolygon(QPolygonF([QPointF(0, -5), QPointF(4, 4), QPointF(-4, 4)]))
        p.restore()
    for x, y, recon in m.enemy_dots():
        p.setBrush(ENEMY if not recon else QColor(255, 60, 40, int(160 + 90 * math.sin(t * 6))))
        p.drawEllipse(QPointF(x * k, y * k), 4.5, 4.5)
    if m.sentries:
        p.setBrush(QColor(255, 220, 90))
        for s in m.sentries:
            p.drawRect(QRectF(s[0] * k - 3, s[1] * k - 3, 6, 6))
    p.restore()
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 230, 120))
    c = r.center()
    p.drawPolygon(QPolygonF([QPointF(c.x(), c.y() - 8), QPointF(c.x() + 6, c.y() + 6), QPointF(c.x(), c.y() + 3),
                             QPointF(c.x() - 6, c.y() + 6)]))
    p.setPen(QPen(QColor(242, 107, 29, 160), 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(r, 10, 10)
    if "recon" in m.active:
        text(p, QRectF(r.x(), r.bottom() + 2, r.width(), 18), "RECON DRONE ONLINE", 12, ALLY,
             Qt.AlignmentFlag.AlignHCenter)
    if "enemy_recon" in m.active:
        text(p, QRectF(r.x(), r.bottom() + 18, r.width(), 18), "ENEMY RECON DRONE", 12, ENEMY,
             Qt.AlignmentFlag.AlignHCenter)
    p.restore()


def draw_mp_hud(p: QPainter, w: int, h: int, m, game, t: float) -> None:  # noqa: ANN001
    s = h / 780
    pl = m.player
    draw_minimap(p, QRectF(24 * s, 24 * s, 190 * s, 190 * s), m, t)
    # score and clock
    r = QRectF(24 * s, h - 120 * s, 300 * s, 80 * s)
    g = QLinearGradient(r.topLeft(), r.topRight())
    g.setColorAt(0, QColor(0, 0, 0, 170))
    g.setColorAt(1, QColor(0, 0, 0, 0))
    p.fillRect(r, g)
    mins, secs = divmod(max(int(m.time_left), 0), 60)
    text(p, QRectF(r.x() + 10 * s, r.y() + 4 * s, 120 * s, 26 * s), f"{mins}:{secs:02d}", 22 * s,
         QColor(255, 90, 60) if m.time_left < 30 else QColor(255, 255, 255))
    for i, (team, col, _name) in enumerate((("ally", ALLY, "VANGUARD"), ("enemy", ENEMY, "ONYX"))):
        y = r.y() + 34 * s + i * 22 * s
        frac = m.scores[team] / 40
        p.fillRect(QRectF(r.x() + 10 * s, y + 6 * s, 200 * s, 10 * s), QColor(255, 255, 255, 35))
        p.fillRect(QRectF(r.x() + 10 * s, y + 6 * s, 200 * s * min(frac, 1), 10 * s), col)
        text(p, QRectF(r.x() + 218 * s, y, 80 * s, 22 * s), f"{m.scores[team]}", 18 * s, col)
    # killfeed
    y = 250 * s
    for f in m.feed[-5:]:
        a = int(255 * min(1.0, (7.0 - f.t) / 0.6))
        kc = QColor(ALLY if f.kteam == "ally" else ENEMY)
        vc = QColor(ALLY if f.vteam == "ally" else ENEMY)
        kc.setAlpha(a)
        vc.setAlpha(a)
        p.fillRect(QRectF(20 * s, y, 330 * s, 22 * s), QColor(0, 0, 0, int(a * 0.4)))
        text(p, QRectF(26 * s, y, 110 * s, 22 * s), f.killer, 15 * s, kc, Qt.AlignmentFlag.AlignLeft, True, 1)
        text(p, QRectF(136 * s, y, 110 * s, 22 * s), f"[{f.weapon}]" + (" ⦿" if f.head else ""), 12 * s,
             QColor(230, 230, 230, a), Qt.AlignmentFlag.AlignHCenter, False, 1)
        text(p, QRectF(246 * s, y, 100 * s, 22 * s), f.victim, 15 * s, vc, Qt.AlignmentFlag.AlignRight, True, 1)
        y += 25 * s
    # score pop-ups
    if m.pops:
        total = sum(sp.points for sp in m.pops if sp.t < 1.6)
        last = m.pops[-1]
        a = int(255 * max(0.0, min(1.0, (2.2 - last.t) / 0.5)))
        if total:
            text(p, QRectF(w / 2 + 40 * s, h / 2 - 60 * s, 200 * s, 40 * s), f"+{total}", 32 * s,
                 QColor(255, 230, 120, a))
        for i, sp in enumerate(reversed(m.pops[-3:])):
            aa = int(255 * max(0.0, min(1.0, (2.2 - sp.t) / 0.5)))
            text(p, QRectF(w / 2 + 40 * s, h / 2 - 20 * s + i * 20 * s, 260 * s, 20 * s), sp.label.upper(), 15 * s,
                 QColor(255, 255, 255, aa))
    if m.medal[1] > 0:
        k = min(m.medal[1] / 0.4, 1.0)
        text(p, QRectF(0, h * 0.2, w, 50 * s), m.medal[0], 36 * s, QColor(242, 140, 50, int(255 * k)),
             Qt.AlignmentFlag.AlignHCenter, True, 3, "Impact")
    if m.message[1] > 0:
        text(p, QRectF(0, h * 0.3, w, 30 * s), m.message[0], 20 * s, QColor(255, 255, 255,
                                                                         int(255 * min(m.message[1], 1.0))),
             Qt.AlignmentFlag.AlignHCenter)
    # scorestreaks on the right
    mult = 0.85 if "hardbargain" in m.perks else 1.0
    for i, sid in enumerate(m.streak_ids):
        st = STREAKS[sid]
        cost = int(st.cost * mult)
        y = h * 0.36 + i * 54 * s
        box = QRectF(w - 70 * s, y, 46 * s, 46 * s)
        ready = sid in m.earned
        p.setPen(QPen(ORANGE if ready else QColor(255, 255, 255, 80), 2))
        p.setBrush(QColor(242, 107, 29, 110) if ready else QColor(0, 0, 0, 120))
        p.drawRect(box)
        frac = min(m.life_score / max(cost, 1), 1.0)
        p.fillRect(QRectF(box.x(), box.bottom() - 4 * s, box.width() * frac, 4 * s), ORANGE)
        text(p, box, str(i + 3), 20 * s, QColor(255, 255, 255) if ready else QColor(255, 255, 255, 110),
             Qt.AlignmentFlag.AlignHCenter)
        text(p, QRectF(w - 270 * s, y + 4 * s, 190 * s, 20 * s), st.name, 14 * s,
             QColor(255, 255, 255, 230 if ready else 140), Qt.AlignmentFlag.AlignRight)
        text(p, QRectF(w - 270 * s, y + 22 * s, 190 * s, 18 * s), f"{cost}", 12 * s, QColor(255, 255, 255, 120),
             Qt.AlignmentFlag.AlignRight, False)
    gun = pl.gun
    if gun is not None and not pl.dead:
        hud.draw_ammo(p, w, h, gun.name, gun.mag, gun.reserve, gun.mag_size(), pl.grenades, pl.tacticals)
        hud.draw_low_ammo(p, w, h, gun.mag, gun.reserve, gun.mag_size(), t)
    if any(c[2] == 0 and math.hypot(c[0] - pl.x, c[1] - pl.y) < 1.2 for c in m.crates):
        hud.draw_prompt(p, w, h, "Hold F to open the Supply Drop")
    # XP bar
    lv, into, need = level_progress(game.profile["xp"])
    p.fillRect(QRectF(0, h - 4 * s, w, 4 * s), QColor(0, 0, 0, 120))
    p.fillRect(QRectF(0, h - 4 * s, w * into / max(need, 1), 4 * s), ORANGE)
    if pl.dead and not m.over:
        p.fillRect(QRectF(0, 0, w, h), QColor(60, 0, 0, 150))
        text(p, QRectF(0, h * 0.38, w, 50 * s), m.killed_by, 28 * s, QColor(255, 255, 255),
             Qt.AlignmentFlag.AlignHCenter)
        text(p, QRectF(0, h * 0.46, w, 30 * s), f"Respawning in {max(3.2 - m.dead_t, 0):.1f}", 18 * s,
             QColor(220, 220, 220), Qt.AlignmentFlag.AlignHCenter, False)


def draw_scoreboard(p: QPainter, w: int, h: int, m, game) -> None:  # noqa: ANN001
    s = h / 780
    p.fillRect(QRectF(0, 0, w, h), QColor(0, 0, 0, 150 if not m.over else 200))
    top = h * 0.14
    if m.over:
        label = "VICTORY" if m.won else ("DRAW" if m.won is None else "DEFEAT")
        col = QColor(90, 200, 255) if m.won else (QColor(255, 255, 255) if m.won is None else QColor(255, 80, 60))
        text(p, QRectF(0, h * 0.03, w, 80 * s), label, 72 * s, col, Qt.AlignmentFlag.AlignHCenter, True, 3, "Impact")
        top = h * 0.17
    rows = m.team_rows()
    cw = min(w * 0.42, 520 * s)
    for i, (team, col, name) in enumerate((("ally", ALLY, "VANGUARD"), ("enemy", ENEMY, "ONYX"))):
        x = w / 2 - cw - 10 * s if i == 0 else w / 2 + 10 * s
        hdr = QRectF(x, top, cw, 40 * s)
        p.fillRect(hdr, QColor(col.red(), col.green(), col.blue(), 150))
        text(p, hdr.adjusted(12 * s, 0, 0, 0), f"{name}   {m.scores[team]}", 22 * s, QColor(255, 255, 255))
        text(p, QRectF(x + cw * 0.5, top + 44 * s, cw * 0.5 - 12 * s, 20 * s), "SCORE     K     D", 13 * s,
             QColor(200, 200, 200), Qt.AlignmentFlag.AlignRight, False)
        for k, (nm, score, kills, deaths, me) in enumerate(rows[team]):
            y = top + 68 * s + k * 32 * s
            rr = QRectF(x, y, cw, 28 * s)
            p.fillRect(rr, QColor(242, 107, 29, 120) if me else QColor(20, 20, 24, 200))
            text(p, rr.adjusted(12 * s, 0, 0, 0), nm, 17 * s, QColor(255, 255, 255))
            text(p, QRectF(x + cw * 0.5, y, cw * 0.5 - 12 * s, 28 * s), f"{score:>5}    {kills:>2}    {deaths:>2}",
                 16 * s, QColor(235, 235, 235), Qt.AlignmentFlag.AlignRight, False, 1, "Consolas")
    if m.over:
        y = top + 68 * s + 7 * 32 * s
        bonus = getattr(m, "match_bonus", 0)
        text(p, QRectF(0, y, w, 30 * s), f"Match bonus +{bonus} XP   ·   Score +{m.p_score}", 20 * s,
             QColor(255, 230, 120), Qt.AlignmentFlag.AlignHCenter)
        if m.over_t > 2.0:
            text(p, QRectF(0, h * 0.9, w, 30 * s), "Press SPACE or click to continue", 18 * s, QColor(255, 255, 255),
                 Qt.AlignmentFlag.AlignHCenter, False)
