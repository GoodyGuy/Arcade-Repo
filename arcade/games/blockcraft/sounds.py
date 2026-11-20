"""BlockCraft's sounds. The effects are short recordings from free sound packs (the ``sfx`` folder beside this file,
credited in ``sfx/CREDITS.txt``): several takes of each, played at random with a little spread in pitch and loudness.
Every sound also has a version synthesised with numpy, which plays when the folder or a file is missing (as for the
rabbit, horse, polar bear, camel, frog and wight, which have no recording yet). The calm music is synthesised."""

from __future__ import annotations

import logging
import os
import random
import threading
import time
import wave
from collections import OrderedDict
from pathlib import Path

import numpy as np

from app.arcade import audio
from app.arcade.audio import RATE, SoundBank, concat, envelope, mix, noise, silence, tone

log = logging.getLogger(__name__)

MATERIALS = ("stone", "wood", "grass", "gravel", "sand", "glass", "cloth", "snow", "metal")
MUSIC_NOTES = (261.6, 293.7, 329.6, 392.0, 440.0, 523.3, 587.3, 659.3, 784.0, 880.0)
SFX_ENV = "TDS_BLOCKCRAFT_SFX"  # another folder of recordings (a missing one: the synthesised sounds only)

# --- recorded sounds -------------------------------------------------------------------------------------------------

# how loud each group of recordings plays (the files are levelled per group; footsteps under digging, the world's
# ambience under everything)
VOLUME = {"step": 0.5, "place": 0.75, "dig": 0.9, "splash": 0.8, "splash_out": 0.6, "swim": 0.6, "bubbles": 0.5,
          "shore": 0.5, "cow": 0.7, "sheep": 0.7, "pig": 0.7, "pig_hurt": 0.7, "chicken": 0.65, "chicken_hurt": 0.65,
          "settler": 0.6, "zombie": 0.75, "zombie_hurt": 0.75, "spider": 0.7, "slime": 0.7, "cinder": 0.7,
          "guardian": 0.8, "wyrm": 0.9, "hurt": 0.9, "attack": 0.7, "eat": 0.7, "burp": 0.6, "pop": 0.7, "orb": 0.6,
          "levelup": 0.7, "click": 0.6, "door": 0.7, "break_tool": 0.8, "explode": 1.0, "fuse": 0.7, "fire": 0.6,
          "bird": 0.5, "crickets": 0.5, "wind": 0.55, "cave": 0.5, "drip": 0.6, "rabbit": 0.5, "horse": 0.7,
          "polar_bear": 0.75, "camel": 0.7, "frog": 0.55, "parrot": 0.6, "mummy": 0.75, "wight": 0.7}
# how far the pitch of a take may sit from the recording (a ratio; 0: always as recorded)
PITCH = {"step": 0.05, "place": 0.05, "dig": 0.05, "levelup": 0.0, "fuse": 0.0, "cave": 0.0, "wind": 0.03,
         "crickets": 0.0, "shore": 0.03, "bird": 0.03, "click": 0.03, "explode": 0.06}
PITCH_DEFAULT = 0.04
LOUDNESS_SPREAD = 0.12  # each play is this much louder or quieter, at random
HIT_GAIN = 0.65  # the taps of mining are the placing sounds, quieter
# names the game plays -> the group of recordings (the names keep their own synthesised sound as the fallback)
ALIASES = {"step_grass1": "step_grass", "step_grass2": "step_grass", "bird0": "bird", "bird1": "bird",
           "bird2": "bird", "bird3": "bird", "drip0": "drip", "drip1": "drip",
           # creatures with no recording of their own borrow the nearest one there is (else: their synthesised voice)
           "mummy": "zombie", "mummy_hurt": "zombie_hurt", "parrot": "bird"}
_EMPTY = np.zeros(1)


def sfx_folder() -> Path:
    """Where the recordings are: beside this file, or the folder named by TDS_BLOCKCRAFT_SFX."""
    other = os.environ.get(SFX_ENV)
    return Path(other) if other else Path(__file__).resolve().parent / "sfx"


def read_wav(path: Path) -> np.ndarray:
    """A PCM WAV file (8 or 16 bit, any channels, any rate) as mono 16-bit samples at the bank's rate."""
    with wave.open(str(path), "rb") as w:
        channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(frames)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0
    else:
        raise ValueError(f"{path.name}: {width * 8}-bit WAV")
    if channels > 1:
        x = x[:len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    if rate != RATE and len(x) > 1:
        x = np.interp(np.arange(0.0, len(x) - 1, rate / RATE), np.arange(len(x)), x)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError(f"{path.name}: no sound")
    return np.clip(x, -32767, 32767).astype(np.int16)


def _group(pool: str) -> str:
    """The key of a group of recordings in VOLUME / PITCH: "step_grass" -> "step"."""
    return pool if pool in VOLUME else pool.split("_")[0]


class Bank(SoundBank):
    """The game's sounds: ``play(name)`` picks one of the recorded takes of that name (never the same twice in a row),
    a little higher or lower and louder or quieter each time; a name with no recording plays its synthesised sound.
    The files are read once, in the background or the first time they are needed; a take's player is made when it
    first plays, and the least recently heard ones are let go."""

    MAX_LIVE = 120  # takes that keep a player (each is one voice)

    def __init__(self, folder: Path | str | None = None) -> None:
        super().__init__()
        self.folder = Path(folder) if folder is not None else sfx_folder()
        self._files: dict[str, list[Path]] = {}  # group -> its files, found by scan()
        self._clips: dict[str, list[np.ndarray]] = {}  # group -> the recordings (16-bit)
        self._takes: dict[str, list[str]] = {}  # group -> the names of its takes in the bank
        self._source: dict[str, tuple[str, int, float]] = {}  # take -> (group, clip, pitch)
        self._last: dict[str, str] = {}
        self._live: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._random = random.Random()
        self.load_ms = 0.0  # time spent reading the recordings
        self.skipped: list[str] = []  # files that could not be read

    # --- finding and reading the files ---

    def scan(self) -> int:
        """Look for ``<name>_<n>.wav`` in the folder (nothing is read yet); the number of groups found."""
        files: dict[str, list[Path]] = {}
        try:
            names = sorted(os.listdir(self.folder))
        except OSError:
            names = []
        for name in names:
            pool, _, number = name[:-4].rpartition("_")
            if name.lower().endswith(".wav") and pool and number.isdigit():
                files.setdefault(pool, []).append(self.folder / name)
        with self._lock:
            self._files = files
            self._clips.clear()
            self._takes.clear()
        return len(files)

    def _load(self, pool: str) -> list[str]:
        with self._lock:
            takes = self._takes.get(pool)
            if takes is not None:
                return takes
            start = time.perf_counter()
            clips = []
            for path in self._files.get(pool, ()):
                try:
                    clips.append(read_wav(path))
                except Exception:  # noqa: BLE001 - a bad file is one take fewer, never an error
                    log.debug("blockcraft sound %s", path, exc_info=True)
                    self.skipped.append(path.name)
            group = _group(pool)
            spread = PITCH.get(group, PITCH_DEFAULT)
            steps = max(1, min(3, 6 // max(len(clips), 1))) if spread else 1
            if steps == 1 and spread and 0 < len(clips) < 6:
                steps = 2
            volume = VOLUME.get(group, 0.6)
            takes = []
            for i in range(len(clips)):
                for k in range(steps):
                    pitch = 1.0 + spread * (2.0 * k / (steps - 1) - 1.0) if steps > 1 else 1.0
                    key = f"{pool}#{i + 1}@{pitch:.3f}"
                    self._source[key] = (pool, i, pitch)
                    self._samples[key] = (_EMPTY, volume, False)  # the samples are made when the take first plays
                    takes.append(key)
            self._clips[pool] = clips
            self._takes[pool] = takes
            self.load_ms += (time.perf_counter() - start) * 1000.0
            return takes

    def load_all(self) -> None:
        for pool in list(self._files):
            self._load(pool)

    def start(self, job=None) -> None:  # noqa: ANN001 - a callable run after the files are read
        """Read every file in the background, then run ``job`` there (the music)."""
        def work() -> None:
            try:
                self.load_all()
                if job is not None:
                    job()
            except Exception:  # noqa: BLE001 - sounds are never worth an error
                log.debug("blockcraft sounds", exc_info=True)

        self._worker = threading.Thread(target=work, name="blockcraft-sounds", daemon=True)
        self._worker.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Until the background reading is done (tests); False when it is still going after ``timeout`` s."""
        if self._worker is not None:
            self._worker.join(timeout)
            return not self._worker.is_alive()
        return True

    # --- choosing what plays ---

    def render(self, take: str) -> np.ndarray:
        """The samples of a take: its recording, played a little faster or slower."""
        pool, i, pitch = self._source[take]
        x = self._clips[pool][i].astype(np.float64) / 32768.0
        if abs(pitch - 1.0) > 1e-6 and len(x) > 2:
            x = np.interp(np.arange(0.0, len(x) - 1, pitch), np.arange(len(x)), x)
        return x

    def resolve(self, name: str) -> tuple[str, float, str]:
        """(the group of recordings, its gain, the synthesised sound to fall back on) for a name the game plays."""
        if name in ALIASES:
            return ALIASES[name], 1.0, name
        if name.startswith("hit_"):
            return "place_" + name[4:], HIT_GAIN, name
        if name.endswith("_hurt"):  # a struck creature: its cry of pain, or its usual voice when it has none
            voice = name[:-5]
            return (name if name in self._files else voice), 1.0, voice
        return name, 1.0, name

    def pick(self, name: str, volume: float = 1.0) -> tuple[str | None, float]:
        """What ``play(name, volume)`` plays: (a take or a synthesised sound, or None for an unknown name; volume)."""
        pool, gain, fallback = self.resolve(name)
        takes = self._takes.get(pool)
        if takes is None:
            takes = self._load(pool) if pool in self._files else []
        if not takes:
            return (fallback if fallback in self._samples else None), volume
        take = takes[self._random.randrange(len(takes))]
        if take == self._last.get(pool) and len(takes) > 1:
            take = takes[(takes.index(take) + 1 + self._random.randrange(len(takes) - 1)) % len(takes)]
        self._last[pool] = take
        return take, volume * gain * self._random.uniform(1.0 - LOUDNESS_SPREAD, 1.0 + LOUDNESS_SPREAD)

    def play(self, name: str, volume: float = 1.0) -> None:
        if not self._enabled or audio.is_muted():
            return
        try:
            take, volume = self.pick(name, volume)
        except Exception:  # noqa: BLE001
            log.debug("blockcraft sound %s", name, exc_info=True)
            take = name if name in self._samples else None
        if take is None:
            return
        super().play(take, volume)
        if take in self._source and take in self._players:
            self._live[take] = None
            self._live.move_to_end(take)
            while len(self._live) > self.MAX_LIVE:
                old, _ = self._live.popitem(last=False)
                self._next.pop(old, None)
                for fx in self._players.pop(old, ()):
                    fx.stop()
                    fx.deleteLater()

    def _make(self, name: str) -> list:
        if name not in self._source:
            return super()._make(name)
        _samples, volume, loop = self._samples[name]
        self._samples[name] = (self.render(name), volume, loop)
        self.VOICES = 1  # the takes of a sound overlap each other; one voice each is enough
        try:
            return super()._make(name)
        finally:
            del self.VOICES
            self._samples[name] = (_EMPTY, volume, loop)

    def stop(self, name: str | None = None) -> None:
        pool = self.resolve(name)[0] if name is not None else None
        for key, players in self._players.items():
            if name is None or key == name or self._source.get(key, ("",))[0] == pool:
                for fx in players:
                    fx.stop()

    def names(self) -> list[str]:
        """Every group of recordings found in the folder."""
        return sorted(self._files)


# --- synthesised sounds (the fallback for every name, and the music) --------------------------------------------------

def _band(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """A crude band-pass: low-pass by ``lo`` taps minus low-pass by ``hi`` taps."""
    a = np.convolve(x, np.ones(lo) / lo, mode="same")
    b = np.convolve(x, np.ones(hi) / hi, mode="same")
    return (a - b) * 2.0


def _material(name: str, dur: float, vol: float, seed: int) -> np.ndarray:
    if name == "stone":
        x = mix(_band(noise(dur, 1.0, 26, seed=seed), 2, 9), tone(140, dur * 0.6, "sine", 0.25, decay=40))
    elif name == "wood":
        x = mix(tone(210, dur, "triangle", 0.45, slide_to=170, decay=30),
                _band(noise(dur, 0.6, 30, seed=seed), 3, 14) * 0.7)
    elif name == "grass":
        x = _band(noise(dur, 1.0, 16, seed=seed), 3, 10)
        x = x * (0.6 + 0.4 * np.sign(np.sin(np.arange(len(x)) / RATE * 2 * np.pi * 37)))
    elif name == "gravel":
        x = _band(noise(dur, 1.0, 14, seed=seed), 2, 6)
        x = x * (0.5 + 0.5 * (np.random.default_rng(seed).random(len(x) // 300 + 1).repeat(300)[:len(x)] > 0.4))
    elif name == "sand":
        x = _band(noise(dur, 0.9, 12, seed=seed), 4, 20)
    elif name == "glass":
        x = mix(_band(noise(dur, 0.8, 18, seed=seed), 1, 3), tone(2400, dur, "sine", 0.25, decay=25),
                tone(3100, dur, "sine", 0.2, decay=30), tone(1800, dur, "sine", 0.18, decay=20))
    elif name == "cloth":
        x = _band(noise(dur, 1.0, 20, seed=seed), 8, 30)
    elif name == "snow":
        x = _band(noise(dur, 0.8, 14, seed=seed), 5, 18)
    else:  # metal
        x = mix(tone(620, dur, "triangle", 0.35, decay=18), tone(930, dur, "sine", 0.2, decay=22),
                _band(noise(dur, 0.4, 40, seed=seed), 1, 4))
    return envelope(np.clip(x, -1, 1) * vol, 0.002, 0.03)


def build(sfx) -> None:  # noqa: ANN001 - the game's SoundBank
    """Add every sound to the bank: the synthesised ones now, and (in a ``Bank``) the recordings and the music in the
    background, so that starting the game does not wait for them."""
    for i, m in enumerate(MATERIALS):
        sfx.add(f"dig_{m}", _material(m, 0.2, 0.85, 11 + i), 0.8)
        sfx.add(f"place_{m}", _material(m, 0.14, 0.75, 31 + i), 0.7)
        sfx.add(f"step_{m}", _material(m, 0.09, 0.45, 51 + i), 0.35)
        sfx.add(f"hit_{m}", _material(m, 0.08, 0.4, 71 + i), 0.3)
    hurt = mix(tone(240, 0.22, "saw", 0.45, slide_to=150, decay=6),
               np.convolve(noise(0.22, 0.4, 8, seed=3), np.ones(6) / 6, mode="same"))
    sfx.add("hurt", envelope(np.convolve(hurt, np.ones(4) / 4, mode="same"), 0.005, 0.06), 0.8)
    sfx.add("pop", tone(520, 0.07, "sine", 0.5, slide_to=1250, release=0.03), 0.5)
    sfx.add("click", concat(tone(1500, 0.012, "square", 0.3), tone(900, 0.02, "square", 0.25)), 0.5)
    eat = concat(*[_band(noise(0.06, 0.9, 30, seed=60 + k), 2, 8) for k in range(3)])
    sfx.add("eat", eat, 0.6)
    sfx.add("burp", tone(110, 0.3, "saw", 0.4, slide_to=80, decay=4), 0.6)
    sfx.add("orb", mix(tone(1320, 0.12, "sine", 0.35, decay=18), tone(1760, 0.12, "sine", 0.2, decay=20)), 0.4)
    sfx.add("levelup", concat(tone(784, 0.1, "sine", 0.35), tone(988, 0.1, "sine", 0.35),
                              tone(1175, 0.3, "sine", 0.35, decay=4)), 0.5)
    sfx.add("break_tool", mix(_material("metal", 0.25, 0.6, 99), _band(noise(0.25, 0.7, 12, seed=98), 1, 4)), 0.7)
    sfx.add("attack", _band(noise(0.08, 0.8, 30, seed=90), 4, 16), 0.5)
    boom = mix(np.convolve(noise(1.4, 1.0, 3, seed=7), np.ones(14) / 14, mode="same") * 2.4,
               tone(55, 1.0, "sine", 0.6, slide_to=30, decay=3))
    sfx.add("explode", np.clip(boom, -1, 1), 1.0)
    sfx.add("fuse", _band(noise(1.0, 0.5, 1, seed=8), 1, 3) * 0.6, 0.5)
    sfx.add("fire", _band(noise(0.25, 0.5, 6, seed=12), 1, 5), 0.3)
    sfx.add("cinder", envelope(_band(noise(0.6, 0.6, 3, seed=13), 1, 6), 0.05, 0.3), 0.35)
    sfx.add("door", _material("wood", 0.2, 0.7, 77), 0.6)
    for i, f in enumerate(MUSIC_NOTES):
        note = mix(tone(f, 3.0, "sine", 0.32, decay=1.3, attack=0.01, release=0.6),
                   tone(f * 2, 3.0, "sine", 0.08, decay=2.2, release=0.6),
                   tone(f * 3, 3.0, "sine", 0.03, decay=3.0, release=0.6))
        sfx.add(f"note{i}", note, 0.32)
    _voices(sfx)
    _ambience(sfx)
    for name, (samples, volume, loop) in list(sfx._samples.items()):  # noqa: SLF001 - none may clip when written
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if peak > 0.97:
            sfx.add(name, samples * (0.97 / peak), volume, loop)
    if isinstance(sfx, Bank):
        sfx.scan()
        sfx.start(lambda: _music(sfx))
    else:
        _music(sfx)


def _sweep(f0: float, f1: float, dur: float, wave_: str = "sine", vibrato: float = 0.0, rate_hz: float = 0.0,
           ) -> np.ndarray:
    """A voice gliding from ``f0`` to ``f1`` Hz, with an optional wobble."""
    t = np.arange(int(dur * RATE)) / RATE
    f = np.linspace(f0, f1, len(t)) * (1 + vibrato * np.sin(2 * np.pi * rate_hz * t))
    ph = 2 * np.pi * np.cumsum(f) / RATE
    if wave_ == "saw":
        return 2.0 * ((ph / (2 * np.pi)) % 1.0) - 1.0
    if wave_ == "square":
        return np.sign(np.sin(ph))
    return np.sin(ph)


def _soft(x: np.ndarray, taps: int) -> np.ndarray:
    return np.convolve(x, np.ones(taps) / taps, mode="same")


def _voices(sfx) -> None:  # noqa: ANN001
    """Animals, monsters and the people of the hamlets, and the water."""
    moo = _soft(_sweep(118, 96, 1.1, "saw", 0.012, 5.5), 14) + 0.5 * _soft(_sweep(236, 190, 1.1, "saw", 0.012, 5.5), 22)
    sfx.add("cow", envelope(moo * 0.34, 0.12, 0.4), 0.55)
    grunt = concat(envelope(_soft(_sweep(210, 150, 0.11, "saw", 0.05, 38), 7), 0.01, 0.04), silence(0.05),
                   envelope(_soft(_sweep(240, 140, 0.16, "saw", 0.06, 42), 7), 0.01, 0.06))
    sfx.add("pig", grunt * 0.42, 0.5)
    baa = _soft(_sweep(330, 290, 0.62, "saw", 0.055, 24), 9)
    sfx.add("sheep", envelope(baa * 0.36, 0.04, 0.2), 0.5)
    cluck = concat(*[envelope(_sweep(680 + 60 * k, 420, 0.07, "sine") * 0.5 +
                              _band(noise(0.07, 0.25, 30, seed=120 + k), 2, 6), 0.004, 0.03)
                     for k in range(3)][:3], silence(0.04),
                   envelope(_sweep(820, 520, 0.16, "sine", 0.03, 30) * 0.5, 0.004, 0.08))
    sfx.add("chicken", cluck * 0.6, 0.45)
    groan = _soft(_sweep(92, 70, 0.9, "saw", 0.03, 4.2), 16) + 0.4 * _soft(noise(0.9, 0.5, 2, seed=121), 30)
    sfx.add("zombie", envelope(groan * 0.4, 0.1, 0.3), 0.6)
    hiss = _band(noise(0.55, 1.0, 3, seed=122), 1, 3) * (0.6 + 0.4 * np.sin(np.arange(int(0.55 * RATE)) / RATE * 190))
    sfx.add("spider", envelope(hiss * 0.5, 0.03, 0.2), 0.5)
    squelch = _soft(_sweep(160, 60, 0.22, "sine"), 3) * 0.6 + _band(noise(0.22, 0.5, 14, seed=123), 6, 22)
    sfx.add("slime", envelope(squelch, 0.005, 0.08), 0.5)
    hum = concat(envelope(_soft(_sweep(170, 190, 0.16, "saw", 0.01, 6), 18), 0.03, 0.06), silence(0.04),
                 envelope(_soft(_sweep(200, 150, 0.24, "saw", 0.01, 6), 18), 0.03, 0.1))
    sfx.add("settler", hum * 0.5, 0.45)
    rumble = _soft(_sweep(58, 44, 0.9, "saw", 0.02, 3), 26) + _soft(noise(0.9, 0.6, 2, seed=124), 60) * 1.5
    sfx.add("guardian", envelope(rumble * 0.5, 0.1, 0.35), 0.6)
    roar = _soft(_sweep(74, 40, 1.3, "saw", 0.04, 7), 22) + _soft(noise(1.3, 0.7, 1.5, seed=128), 40) * 1.6
    sfx.add("wyrm", envelope(np.clip(roar * 0.5, -1, 1), 0.08, 0.5), 0.7)
    # the animals and the dead of the newer biomes
    sfx.add("rabbit", envelope(_sweep(1500, 2100, 0.07, "sine") * 0.3, 0.004, 0.04), 0.35)
    neigh = _soft(_sweep(980, 380, 0.85, "saw", 0.09, 13), 5) + 0.4 * _soft(_sweep(490, 190, 0.85, "saw", 0.09, 13), 9)
    sfx.add("horse", envelope(neigh * 0.3, 0.03, 0.3), 0.5)
    growl = _soft(_sweep(112, 70, 0.95, "saw", 0.05, 9), 10) + _soft(noise(0.95, 0.5, 2, seed=160), 24) * 1.2
    sfx.add("polar_bear", envelope(np.clip(growl * 0.45, -1, 1), 0.08, 0.35), 0.6)
    sfx.add("camel", envelope(_soft(_sweep(156, 104, 1.0, "saw", 0.035, 7), 12) * 0.4, 0.1, 0.35), 0.55)
    t = np.arange(int(0.14 * RATE)) / RATE
    croak = _soft(_sweep(230, 170, 0.14, "square"), 4) * (0.55 + 0.45 * np.sin(2 * np.pi * 55 * t))
    sfx.add("frog", concat(envelope(croak * 0.4, 0.01, 0.04), silence(0.09), envelope(croak * 0.32, 0.01, 0.05)), 0.4)
    squawk = concat(*[envelope(_soft(_sweep(1900 - 150 * k, 1250, 0.16, "saw", 0.08, 45), 3) * 0.3, 0.008, 0.06)
                      for k in range(2)])
    sfx.add("parrot", squawk, 0.4)
    rasp = _band(noise(1.0, 0.8, 2, seed=161), 2, 12) * 0.7 + _soft(_sweep(84, 62, 1.0, "saw", 0.02, 3.5), 18) * 0.35
    sfx.add("mummy", envelope(rasp, 0.15, 0.4), 0.55)
    wail = _sweep(1350, 880, 0.9, "sine", 0.03, 6) * 0.22 + _band(noise(0.9, 0.5, 1, seed=162), 1, 3) * 0.5
    sfx.add("wight", envelope(wail, 0.12, 0.45), 0.5)
    # water: a deep plunge with bubbles, a lighter splash coming out, soft strokes, bubbles under the surface
    t = np.arange(int(0.9 * RATE)) / RATE
    plunge = _soft(noise(0.9, 1.0, 4, seed=125), 7) * 1.6 + _soft(noise(0.9, 1.0, 9, seed=126), 28) * 1.4
    bubbles = sum(np.sin(2 * np.pi * (500 + 260 * k) * t * (1 + 0.6 * t)) * np.exp(-((t - 0.12 - 0.09 * k) / 0.03) ** 2)
                  for k in range(6)) * 0.12
    sfx.add("splash", np.clip(plunge + bubbles, -1, 1), 0.6)
    sfx.add("splash_out", envelope(_soft(noise(0.4, 0.9, 7, seed=129), 6) * 1.5, 0.01, 0.15), 0.35)
    sfx.add("swim", envelope(_soft(noise(0.42, 0.8, 5, seed=127), 12) * 1.7, 0.06, 0.2), 0.3)
    sfx.add("bubbles", envelope(bubbles[:int(0.75 * RATE)] * 2.0, 0.01, 0.1), 0.3)
    for i in range(3):  # footsteps through grass, in two parts
        rustle = concat(_band(noise(0.05, 1.0, 22, seed=130 + i), 3, 9), _band(noise(0.07, 0.7, 16, seed=140 + i), 4, 14))
        sfx.add(f"step_grass{i}" if i else "step_grass", envelope(rustle * 0.5, 0.004, 0.03), 0.35)


def _ambience(sfx) -> None:  # noqa: ANN001
    """What the world sounds like around the player."""
    rng = np.random.default_rng(77)
    for k in range(4):  # birds: a few rising and falling chirps
        notes = []
        for _ in range(int(rng.integers(2, 6))):
            f0 = float(rng.uniform(2300, 4200))
            notes += [envelope(_sweep(f0, f0 * float(rng.uniform(0.75, 1.3)), float(rng.uniform(0.05, 0.12)), "sine",
                                      0.02, 40) * 0.3, 0.006, 0.03), silence(float(rng.uniform(0.03, 0.12)))]
        sfx.add(f"bird{k}", concat(*notes), 0.22)
    t = np.arange(int(1.6 * RATE)) / RATE
    chirp = np.sin(2 * np.pi * 4300 * t) * (np.sin(2 * np.pi * 34 * t) > 0.2) * (np.sin(2 * np.pi * 2.6 * t) > 0)
    sfx.add("crickets", envelope(chirp * 0.16, 0.1, 0.3), 0.2)
    lap = _soft(noise(2.4, 1.0, 0.0, seed=150), 40) * 2.2 * (0.35 + 0.65 * np.sin(np.pi * np.arange(int(2.4 * RATE)) /
                                                                               int(2.4 * RATE)) ** 2)
    sfx.add("shore", envelope(np.clip(lap, -1, 1) * 0.5, 0.5, 0.8), 0.3)
    for k in range(2):  # a drip and its echo, deep underground
        f0 = 900.0 + 400 * k
        drip = envelope(_sweep(f0, f0 * 1.9, 0.07, "sine") * 0.5, 0.002, 0.05)
        sfx.add(f"drip{k}", concat(drip, silence(0.22), drip * 0.35, silence(0.25), drip * 0.12), 0.35)
    wind = _soft(noise(3.0, 1.0, 0.0, seed=151), 90) * 3.0 * (0.4 + 0.6 * np.sin(np.pi * np.arange(int(3.0 * RATE)) /
                                                                                 int(3.0 * RATE)) ** 2)
    sfx.add("wind", envelope(np.clip(wind, -1, 1) * 0.5, 0.8, 1.0), 0.22)
    n = int(4.0 * RATE)  # the hollow air of a cave: slow, dark noise with a low hum under it
    drone = _soft(_soft(noise(4.0, 1.0, 0.0, seed=152), 160), 160) * 9.0 * (0.5 + 0.5 * np.sin(np.pi * np.arange(n) / n))
    hum = np.sin(2 * np.pi * 58 * np.arange(n) / RATE) * 0.12
    sfx.add("cave", envelope(np.clip(drone + hum, -1, 1) * 0.5, 1.2, 1.4), 0.25)


def _music(sfx) -> None:  # noqa: ANN001
    for k in range(MUSIC_PIECES):
        sfx.add(f"music{k}", _piece(k), 0.42)


MUSIC_PIECES = 4
_SCALE = (261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25, 783.99)  # C major pentatonic, two octaves
_CHORDS = (((130.81, 196.0, 329.63), (110.0, 164.81, 261.63), (87.31, 174.61, 261.63), (98.0, 146.83, 246.94)),
           ((87.31, 130.81, 220.0), (130.81, 196.0, 329.63), (98.0, 146.83, 246.94), (110.0, 164.81, 261.63)),
           ((110.0, 164.81, 261.63), (87.31, 174.61, 261.63), (130.81, 196.0, 329.63), (98.0, 196.0, 293.66)),
           ((130.81, 164.81, 246.94), (87.31, 130.81, 220.0), (110.0, 164.81, 261.63), (98.0, 146.83, 246.94)))


def _swell(n: int, attack: float, release: float) -> np.ndarray:
    """A level that rises and falls along a smooth (raised cosine) curve; ``attack`` and ``release`` in s."""
    env = np.ones(n)
    a, r = min(int(attack * RATE), n // 2), min(int(release * RATE), n // 2)
    if a:
        env[:a] = 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    if r:
        env[n - r:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r)
    return env


def _hall(x: np.ndarray, seconds: float = 2.8, wet: float = 0.45, seed: int = 1) -> np.ndarray:
    """The sound in a large soft room: it is convolved with dark noise that dies away over ``seconds``."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    tail = _soft(rng.standard_normal(n) * np.exp(-t * 6.9 / seconds), 11)
    tail[:int(0.02 * RATE)] *= np.linspace(0.0, 1.0, int(0.02 * RATE))  # the walls are a little way off
    tail /= np.sqrt(np.sum(tail ** 2))
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    echo = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(tail, size), size)[:len(x)]
    return x + wet * echo


def _piece(k: int) -> np.ndarray:
    """A calm piece of about half a minute: four slow, warm chords that melt into each other, a few soft notes over
    them like a felted piano heard from the next room, and the long tail of a hall."""
    rng = np.random.default_rng(900 + k)
    bar = 7.5
    n = int((bar * 4 + 7.0) * RATE)
    out = np.zeros(n)
    for i, chord in enumerate(_CHORDS[k % len(_CHORDS)]):
        m = int((bar + 4.0) * RATE)
        t = np.arange(m) / RATE
        pad = sum(np.sin(2 * np.pi * f * t * (1 + d) + p) for f, p in zip(chord, (0.0, 1.3, 2.1))
                  for d in (-0.0016, 0.0, 0.0019)) / 9
        pad = pad + 0.10 * sum(np.sin(2 * np.pi * 2 * f * t) for f in chord) / 3
        pad = pad * (1.0 + 0.07 * np.sin(2 * np.pi * 0.11 * t + i))  # it breathes
        a = int(i * bar * RATE)
        out[a:a + m] += (pad * _swell(m, 3.6, 4.2) * 0.13)[:n - a]
        beats = sorted(rng.choice(8, size=int(rng.integers(1, 4)), replace=False))
        for beat in beats:  # sparse soft notes on a slow pulse
            f = _SCALE[int(rng.integers(0, 7))]
            tn = np.arange(int(4.5 * RATE)) / RATE
            note = (np.sin(2 * np.pi * f * tn) + 0.5 * np.sin(2 * np.pi * f * 1.003 * tn) +
                    0.16 * np.sin(2 * np.pi * 2 * f * tn) * np.exp(-tn * 4)) * np.exp(-tn * 1.15)
            note *= np.minimum(tn / 0.035, 1.0) ** 2 * _swell(len(tn), 0.0, 1.0)
            b = a + int((1.0 + beat * (bar - 1.5) / 8 + float(rng.uniform(-0.04, 0.04))) * RATE)
            part = note[:max(0, n - b)] * float(rng.uniform(0.035, 0.065))
            out[b:b + len(part)] += part
    out = _hall(out, seed=40 + k)
    out *= _swell(n, 1.5, 5.0)
    out *= 0.07 / (float(np.sqrt(np.mean(out * out))) or 1.0)  # every piece equally (and gently) loud
    return out * min(1.0, 0.7 / (float(np.max(np.abs(out))) or 1.0))


def material_of(block: int) -> str:
    from app.arcade.games.blockcraft import blocks as B

    return B.MATERIAL.get(B.base_block(block), "stone")
