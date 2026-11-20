"""Surf mode: a first-person run down one of Case Clash's surf maps inside the game widget - the movement on a fixed
100 Hz tick, mouse look, stages with checkpoints, the clock and splits, falls back to the stage's platform, the view
drawn by the graphics card (surf_render.py) and the heads-up display over it."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen, QPixmap

from app.arcade.games.case_clash import data as D
from app.arcade.games.case_clash import surf_maps as SM
from app.arcade.games.case_clash import ui as U
from app.arcade.games.case_clash.surf_physics import EYE, TICK, Player
from app.arcade.games.case_clash.surf_render import SurfRenderer

AL = Qt.AlignmentFlag
K = Qt.Key
LOOK = 0.11  # degrees of view turn per pixel of mouse motion (times the player's sensitivity)
FOV = 76.0  # vertical, degrees; widens a little with speed
MAX_TICKS = 8  # physics ticks per frame at most (a long stall does not fast-forward the run)
ZONE_COLORS = {"start": (0.25, 1.0, 0.45), "cp": (0.3, 0.65, 1.0), "end": (1.0, 0.3, 0.3)}


def fmt(seconds: float | None) -> str:
    """0:15.84 / 1:02.30 / -"""
    if seconds is None:
        return "—"
    m, s = divmod(max(seconds, 0.0), 60.0)
    return f"{int(m)}:{s:05.2f}"


def fmt_delta(d: float) -> str:
    return ("−" if d < 0 else "+") + f"{abs(d):.2f}"


class SurfRun:
    """One surf session on one map (the game widget delegates its loop, input and painting here)."""

    def __init__(self, game, map_id: str) -> None:  # noqa: ANN001 - CaseClashGame
        self.g = game
        self.m = SM.get_map(map_id)
        self.pl = Player(self.m.world)
        self.zones = self.m.zone_list()
        self.best = game.surf_best(map_id)
        self.best_splits = game.surf_splits(map_id)
        self.acc = 0.0
        self.clock = 0.0  # the run's time
        self.started = False
        self.stage = 0
        self.splits: list[float] = []
        self.falls = 0
        self.top = 0.0
        self.phase = "run"  # run / done
        self.result: dict = {}
        self.popup: tuple[str, str, float] | None = None  # text, colour, age
        self.fade = 0.0
        self.t = 0.0  # seconds in the session (sky animation)
        self.hint_t = 0.0
        self._img: QImage | None = None
        self._fov = FOV
        self.respawn(0)

    # --- flow ---------------------------------------------------------------------------------------------------

    def respawn(self, stage: int, fell: bool = False) -> None:
        st = self.m.stages[stage]
        self.stage = stage
        self.pl.place(st.spawn)
        self.yaw = st.yaw
        self.pitch = math.radians(14.0)
        self.acc = 0.0
        if stage == 0:
            self.clock = 0.0
            self.started = False
            self.splits = []
        if fell:
            self.falls += 1
            self.fade = 0.35
            self.g.sfx.play("surf_fall")

    def restart(self) -> None:
        self.phase = "run"
        self.result = {}
        self.falls = 0
        self.top = 0.0
        self.popup = None
        self.respawn(0)
        self.g.set_surf_cursor(True)

    def finish(self) -> None:
        self.phase = "done"
        self.result = self.g.surf_finished(self.m.id, self.clock, list(self.splits), self.falls, self.top)
        self.best = self.g.surf_best(self.m.id)
        self.best_splits = self.g.surf_splits(self.m.id)
        self.g.set_surf_cursor(False)

    # --- input ----------------------------------------------------------------------------------------------------

    def key(self, key: int) -> None:
        if self.phase == "done":
            if key in (K.Key_Return, K.Key_Enter, K.Key_Space):
                self.restart()
            return
        if key == K.Key_R:
            self.respawn(self.stage)
            self.g.sfx.play("click")

    def inputs(self) -> tuple[float, float, bool]:
        keys = self.g.keys
        fwd = (K.Key_W in keys) - (K.Key_S in keys)
        side = (K.Key_D in keys) - (K.Key_A in keys)
        return float(fwd), float(side), K.Key_Space in keys

    # --- loop -----------------------------------------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        self.t += dt
        self.fade = max(0.0, self.fade - dt)
        if self.popup is not None:
            text, col, age = self.popup
            self.popup = (text, col, age + dt) if age + dt < 2.6 else None
        if self.g.mouse_captured:
            dx, dy = self.g.take_mouse_delta()
            self.yaw = (self.yaw - math.radians(dx * LOOK)) % math.tau
            self.pitch = min(max(self.pitch + math.radians(dy * LOOK), -1.55), 1.55)
        if self.phase != "run":
            return
        self.hint_t += dt
        self.acc += dt
        n = 0
        while self.acc >= TICK and n < MAX_TICKS:
            self.acc -= TICK
            n += 1
            self.step()
            if self.phase != "run":
                break
        if n == MAX_TICKS:
            self.acc = min(self.acc, TICK)

    def step(self) -> None:
        """One 100 Hz tick: move, then the zones, the clock and falls."""
        fwd, side, jump = self.inputs()
        pl = self.pl
        was_ground = pl.on_ground
        pl.tick(fwd, side, jump, self.yaw)
        if pl.jumped:
            self.g.sfx.play("surf_jump", 0.5)
        elif pl.landed > 420 and (pl.on_ground or not was_ground):
            self.g.sfx.play("surf_land", min(1.0, pl.landed / 1200))
        self.top = max(self.top, pl.speed)
        start = self.zones[0]
        if self.stage == 0 and not self.started:
            if not start.contains(pl.pos):
                self.started = True
                self.g.sfx.play("surf_go")
        elif self.started:
            self.clock += TICK
        nxt = self.stage + 1
        if nxt < len(self.zones) and self.zones[nxt].contains(pl.pos):
            if nxt == len(self.zones) - 1:
                self.finish()
                return
            self.stage = nxt
            self.splits.append(round(self.clock, 2))
            k = len(self.splits) - 1
            msg = f"CHECKPOINT {nxt + 1}/{len(self.m.stages)}  ·  {fmt(self.clock)}"
            col = "#7fb2ff"
            if k < len(self.best_splits):
                d = self.clock - self.best_splits[k]
                msg += f"  ({fmt_delta(d)})"
                col = "#8bd45b" if d < 0 else "#e58a5b"
            self.popup = (msg, col, 0.0)
            self.g.sfx.play("surf_cp")
        if pl.pos[2] < self.m.stages[self.stage].kill_z:
            self.respawn(self.stage, fell=True)

    # --- drawing --------------------------------------------------------------------------------------------------

    def camera(self) -> tuple[tuple, float, float, float]:
        a = min(max(self.acc / TICK, 0.0), 1.0)
        x, y, z = self.pl.lerp(a)
        spd = self.pl.speed
        target = FOV + min(max(spd - 600.0, 0.0) / 1600.0, 1.0) * 10.0
        self._fov += (target - self._fov) * 0.08
        return (x, y, z + EYE), self.yaw, self.pitch, self._fov

    def zone_draws(self, eye: tuple) -> list[tuple]:
        out = []
        pulse = 0.75 + 0.25 * math.sin(self.t * 3.0)
        for k, z in enumerate(self.zones):
            kind = "start" if k == 0 else "end" if k == len(self.zones) - 1 else "cp"
            r, g, b = ZONE_COLORS[kind]
            a = 0.42 * pulse
            if k <= self.stage and kind == "cp":
                a = 0.12  # reached
            if z.contains((eye[0], eye[1], eye[2] - EYE)):
                a *= 0.25  # standing in it: a faint tint, not a wall of colour
            out.append((k, r, g, b, a))
        return out

    def paint(self, p: QPainter, w: int, h: int, hits: U.Hits) -> None:
        r = self.g.surf_renderer()
        dpr = self.g.devicePixelRatioF()
        rw, rh = max(int(w * dpr), 16), max(int(h * dpr), 16)
        if r.available():
            if not self.g.paused or self._img is None or self._img.width() != rw:
                eye, yaw, pitch, fov = self.camera()
                self._img = r.render(self.m, eye, yaw, pitch, fov, rw, rh, self.t, self.zone_draws(eye))
            p.drawImage(QRectF(0, 0, w, h), self._img)
        else:
            p.fillRect(QRectF(0, 0, w, h), QColor("#10141a"))
            U.text(p, QRectF(0, 0, w, h), "The surf view needs OpenGL 3.3: " + r.error, 18, U.RED, AL.AlignHCenter)
        s = max(min(w / 1280, h / 780), 0.6)
        if self.fade > 0:
            p.fillRect(QRectF(0, 0, w, h), QColor(0, 0, 0, int(255 * min(1.0, self.fade / 0.2))))
        self.paint_hud(p, w, h, s)
        if self.phase == "done":
            self.paint_results(p, w, h, s, hits)

    def paint_hud(self, p: QPainter, w: int, h: int, s: float) -> None:
        g = self.g
        if self.phase == "run":
            c = QPointF(w / 2, h / 2)  # crosshair
            p.save()
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            for col, wd in ((QColor(0, 0, 0, 150), 3.2 * s), (QColor(255, 255, 255, 230), 1.6 * s)):
                p.setPen(QPen(col, wd))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p.drawLine(QPointF(c.x() + dx * 4 * s, c.y() + dy * 4 * s),
                               QPointF(c.x() + dx * 10 * s, c.y() + dy * 10 * s))
            p.restore()
        # top: the clock, the stage, the record
        tb = QRectF(w / 2 - 150 * s, 10 * s, 300 * s, 70 * s)
        p.fillRect(tb, QColor(0, 0, 0, 110))
        U.text(p, QRectF(tb.left(), tb.top() + 2 * s, tb.width(), 40 * s), fmt(self.clock), 34 * s,
               U.TEXT if self.started or self.phase == "done" else U.DIM, AL.AlignHCenter, "bold", True)
        U.text(p, QRectF(tb.left(), tb.top() + 42 * s, tb.width(), 22 * s),
               f"STAGE {self.stage + 1}/{len(self.m.stages)}   ·   BEST {fmt(self.best)}   ·   PAR {fmt(self.m.par)}",
               12 * s, U.DIM, AL.AlignHCenter, cond=True)
        # top left: the map
        lb = QRectF(14 * s, 12 * s, 290 * s, 48 * s)
        p.fillRect(lb, QColor(0, 0, 0, 100))
        U.text(p, QRectF(lb.left() + 10 * s, lb.top() + 2 * s, lb.width(), 26 * s), self.m.name.upper(), 19 * s,
               U.TEXT, weight="bold", cond=True)
        U.text(p, QRectF(lb.left() + 10 * s, lb.top() + 26 * s, lb.width(), 18 * s),
               f"TIER {self.m.tier} · FALLS {self.falls} · TOP {self.top:.0f} U/S", 11 * s, U.DIM, cond=True)
        # bottom: the speedometer
        spd = self.pl.speed
        col = QColor("#ffffff") if spd < 900 else QColor("#ffd479") if spd < 1600 else QColor("#ff9a4a")
        U.text(p, QRectF(0, h - 92 * s, w, 54 * s), f"{spd:.0f}", 48 * s, col, AL.AlignHCenter, "bold", True,
               shadow=True)
        U.text(p, QRectF(0, h - 42 * s, w, 20 * s), "UNITS / SECOND", 11 * s, U.DIM, AL.AlignHCenter, cond=True,
               shadow=True)
        bar = QRectF(w / 2 - 120 * s, h - 22 * s, 240 * s, 4 * s)
        p.fillRect(bar, QColor(255, 255, 255, 40))
        p.fillRect(QRectF(bar.left(), bar.top(), bar.width() * min(spd / 2500.0, 1.0), bar.height()), col)
        # a checkpoint split
        if self.popup is not None:
            text, color, age = self.popup
            a = min(1.0, age / 0.12, (2.6 - age) / 0.5)
            p.setOpacity(max(a, 0.0))
            U.text(p, QRectF(0, h * 0.24, w, 40 * s), text, 24 * s, QColor(color), AL.AlignHCenter, "bold", True,
                   shadow=True)
            p.setOpacity(1.0)
        if self.phase == "run":
            hint = self.hint()
            if hint:
                U.text(p, QRectF(0, h * 0.68, w, 26 * s), hint, 16 * s, QColor(255, 255, 255, 220), AL.AlignHCenter,
                       cond=True, shadow=True)
            if not g.mouse_captured and not g.paused:
                U.text(p, QRectF(0, h * 0.62, w, 26 * s), "CLICK TO LOOK AROUND WITH THE MOUSE", 16 * s, U.GOLD,
                       AL.AlignHCenter, "bold", True, shadow=True)

    def hint(self) -> str:
        """Help for a first run: what to press right now."""
        g, pl = self.g, self.pl
        if not self.started and self.stage == 0:
            return (f"{g.key_label(K.Key_W)} TO WALK OFF THE PLATFORM  ·  STRAFE INTO THE RAMP, LOOK ALONG IT  ·  "
                    f"{g.key_label(K.Key_Space)} JUMP (HOLD TO HOP)  ·  {g.key_label(K.Key_R)} BACK TO THE CHECKPOINT")
        if self.m.tier == 1 and self.best is None and pl.surfing and pl.surf_normal is not None:
            st = self.m.stages[self.stage]
            ramp = None
            best = 1e18
            for r in st.ramps:
                d = min((r.ridge[:, 0] - pl.pos[0]) ** 2 + (r.ridge[:, 1] - pl.pos[1]) ** 2)
                if d < best:
                    best, ramp = d, r
            if ramp is not None:
                # the face's normal points away from the ridge: strafe the other way
                n = pl.surf_normal
                right = (math.sin(self.yaw), -math.cos(self.yaw))
                into_right = -(n[0] * right[0] + n[1] * right[1]) > 0
                key = g.key_label(K.Key_D if into_right else K.Key_A)
                held = (K.Key_D if into_right else K.Key_A) in g.keys
                if K.Key_W in g.keys:
                    return "LET GO OF " + g.key_label(K.Key_W) + ": IT PULLS YOU OFF THE RAMP"
                if not held:
                    return f"HOLD {key} TO PRESS INTO THE RAMP"
                if self.hint_t < 40:
                    return "MOVE THE MOUSE SMOOTHLY TO STEER  ·  RIDE LOW FOR SPEED"
        return ""

    def paint_results(self, p: QPainter, w: int, h: int, s: float, hits: U.Hits) -> None:
        p.fillRect(QRectF(0, 0, w, h), QColor(0, 0, 0, 140))
        pw, ph = min(600 * s, w * 0.92), min(470 * s, h * 0.92)
        r = QRectF((w - pw) / 2, (h - ph) / 2, pw, ph)
        res = self.result
        body = U.panel(p, r, f"{self.m.name} · finished", s, 245)
        U.text(p, QRectF(body.left(), body.top() + 10 * s, pw, 60 * s), fmt(self.clock), 54 * s,
               U.GOLD if res.get("gold") else U.TEXT, AL.AlignHCenter, "bold", True)
        tag = "NEW PERSONAL BEST" if res.get("record") else f"PERSONAL BEST {fmt(self.best)}"
        U.text(p, QRectF(body.left(), body.top() + 70 * s, pw, 24 * s), tag, 16 * s,
               U.MONEY if res.get("record") else U.DIM, AL.AlignHCenter, "bold", True)
        rows = [("Par", f"{fmt(self.m.par)}  {'✓' if res.get('par') else '✗'}"),
                ("Gold", f"{fmt(self.m.gold)}  {'✓' if res.get('gold') else '✗'}"),
                ("Splits", "  ".join(fmt(x) for x in self.splits) or "—"),
                ("Falls · top speed", f"{self.falls} · {self.top:.0f} u/s"),
                ("Paid", D.money(res.get("pay", 0))), ("XP", f"+{res.get('xp', 0)}")]
        y = body.top() + 104 * s
        rh = 28 * s
        for k, v in rows:
            U.text(p, QRectF(body.left() + 30 * s, y, pw * 0.4, rh), k.upper(), 14 * s, U.DIM, cond=True)
            U.text(p, QRectF(body.left(), y, pw - 30 * s, rh), v, 16 * s, U.MONEY if "$" in v else U.TEXT,
                   AL.AlignRight, "bold", True)
            y += rh
        drop = res.get("drop")
        if drop:
            U.text(p, QRectF(body.left() + 30 * s, y + 4 * s, pw - 60 * s, 26 * s),
                   f"FIRST CLEAR · CASE DROP: {D.CASES[drop].name.upper()}", 16 * s, U.GOLD, weight="bold", cond=True)
        bw = (pw - 80 * s) / 3
        by = r.bottom() - 60 * s
        U.button(p, hits, QRectF(r.left() + 30 * s, by, bw, 40 * s), "Run again", self.restart, s, "primary", size=17)
        nxt = self.g.surf_next_map(self.m.id)
        U.button(p, hits, QRectF(r.left() + 40 * s + bw, by, bw, 40 * s), "Next map",
                 (lambda: self.g.start_surf(nxt)) if nxt else None, s, "blue", bool(nxt), 16)
        U.button(p, hits, QRectF(r.right() - 30 * s - bw, by, bw, 40 * s), "Surf maps", self.g.close_surf, s, size=17)


# --- the menu cards' pictures ----------------------------------------------------------------------------------------

def preview(renderer: SurfRenderer, map_id: str, w: int, h: int, dpr: float) -> QPixmap | None:
    """The map from above its start, for the menu (None without OpenGL)."""
    if not renderer.available():
        return None
    m = SM.get_map(map_id)
    st = m.stages[0]
    back = 900.0
    eye = (st.spawn[0] - math.cos(st.yaw) * back, st.spawn[1] - math.sin(st.yaw) * back, st.spawn[2] + 700.0)
    zones = [(0, *ZONE_COLORS["start"], 0.35)]
    img = renderer.render(m, eye, st.yaw + 0.25, math.radians(20.0), 64.0, int(w * dpr), int(h * dpr), 5.0, zones)
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(dpr)
    return pm


def placeholder(w: int, h: int, theme_id: str) -> QPixmap:
    """A gradient in the map's colours when there is no OpenGL."""
    from app.arcade.games.case_clash import art as A

    th = SM.THEMES[theme_id]
    pm = A.new_pixmap(max(w, 1), max(h, 1))
    p = QPainter(pm)
    g = QLinearGradient(0, 0, 0, h)
    g.setColorAt(0.0, QColor.fromRgbF(*th.sky_top))
    g.setColorAt(0.6, QColor.fromRgbF(*th.sky_horizon))
    g.setColorAt(1.0, QColor.fromRgbF(*th.sea))
    p.fillRect(0, 0, w, h, g)
    p.end()
    return pm
