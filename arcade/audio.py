"""Sound effects made with numpy (no sound files ship with the program): a game builds its sounds once with the
synthesis helpers, then ``SoundBank.play(name)``. Silent on the offscreen platform (tests), when muted, or when Qt
Multimedia is unavailable."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RATE = 22050
_state = {"muted": False, "volume": 0.7}


def set_muted(muted: bool) -> None:
    _state["muted"] = bool(muted)


def is_muted() -> bool:
    return _state["muted"]


def set_volume(volume: float) -> None:
    _state["volume"] = float(min(max(volume, 0.0), 1.0))


# --- synthesis ---------------------------------------------------------------------------------------------------

def _t(dur: float, rate: int = RATE) -> np.ndarray:
    return np.arange(max(int(dur * rate), 1)) / rate


def envelope(x: np.ndarray, attack: float = 0.005, release: float = 0.05, rate: int = RATE) -> np.ndarray:
    """Linear fade in / fade out."""
    n = len(x)
    env = np.ones(n)
    a, r = min(int(attack * rate), n), min(int(release * rate), n)
    if a:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r:
        env[n - r:] *= np.linspace(1.0, 0.0, r)
    return x * env


def tone(freq: float, dur: float, wave_: str = "square", volume: float = 0.5, slide_to: float | None = None,
         attack: float = 0.005, release: float = 0.05, decay: float = 0.0, rate: int = RATE) -> np.ndarray:
    """A note: ``wave_`` square / sine / saw / triangle; ``slide_to`` glides the pitch; ``decay`` (1/s) an exponential
    fall."""
    t = _t(dur, rate)
    f = np.full_like(t, freq) if slide_to is None else np.linspace(freq, slide_to, len(t))
    phase = 2 * np.pi * np.cumsum(f) / rate
    if wave_ == "sine":
        x = np.sin(phase)
    elif wave_ == "saw":
        x = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
    elif wave_ == "triangle":
        x = 2.0 * np.abs(2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
    else:
        x = np.sign(np.sin(phase))
    if decay:
        x = x * np.exp(-decay * t)
    return envelope(x * volume, attack, release, rate)


def noise(dur: float, volume: float = 0.5, decay: float = 8.0, smooth: int = 1, rate: int = RATE,
          seed: int = 7) -> np.ndarray:
    """White noise with an exponential decay (shots, explosions, footsteps); ``smooth`` > 1 low-passes it."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, len(_t(dur, rate)))
    if smooth > 1:
        x = np.convolve(x, np.ones(smooth) / smooth, mode="same") * np.sqrt(smooth)
    return envelope(x * volume * np.exp(-decay * _t(dur, rate)), 0.001, 0.02, rate)


def mix(*parts: np.ndarray) -> np.ndarray:
    n = max(len(p) for p in parts)
    out = np.zeros(n)
    for p in parts:
        out[:len(p)] += p
    return out


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts)


def silence(dur: float, rate: int = RATE) -> np.ndarray:
    return np.zeros(len(_t(dur, rate)))


# --- playback ----------------------------------------------------------------------------------------------------

def _audio_available() -> bool:
    from PySide6.QtGui import QGuiApplication

    return QGuiApplication.instance() is not None and not QGuiApplication.platformName().startswith("offscreen")


class SoundBank:
    """A game's named sounds. ``add`` keeps the samples; the WAV and its players are made the first time the sound
    plays (a few players per sound, so a sound can overlap itself)."""

    VOICES = 3

    def __init__(self) -> None:
        self._samples: dict[str, tuple[np.ndarray, float, bool]] = {}
        self._players: dict[str, list] = {}
        self._next: dict[str, int] = {}
        self._enabled = _audio_available()

    def add(self, name: str, samples: np.ndarray, volume: float = 1.0, loop: bool = False) -> None:
        self._samples[name] = (np.asarray(samples, dtype=float), volume, loop)

    def __contains__(self, name: str) -> bool:
        return name in self._samples

    def _make(self, name: str) -> list:
        from PySide6.QtCore import QUrl
        from PySide6.QtMultimedia import QSoundEffect

        samples, _volume, loop = self._samples[name]
        pcm = (np.clip(samples, -1.0, 1.0) * 32000).astype("<i2").tobytes()
        folder = Path(tempfile.gettempdir()) / "tds_arcade_sfx"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{hashlib.sha1(pcm).hexdigest()[:16]}.wav"
        if not path.is_file():
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(pcm)
        players = []
        for _ in range(1 if loop else self.VOICES):
            fx = QSoundEffect()
            fx.setSource(QUrl.fromLocalFile(str(path)))
            if loop:
                fx.setLoopCount(QSoundEffect.Loop.Infinite.value)
            players.append(fx)
        return players

    def play(self, name: str, volume: float = 1.0) -> None:
        if not self._enabled or _state["muted"] or name not in self._samples:
            return
        try:
            players = self._players.get(name)
            if players is None:
                players = self._players[name] = self._make(name)
            i = self._next.get(name, 0)
            self._next[name] = (i + 1) % len(players)
            fx = players[i]
            fx.setVolume(min(1.0, self._samples[name][1] * volume * _state["volume"]))
            fx.play()
        except Exception:  # noqa: BLE001 - a sound is never worth an error
            log.debug("arcade sound %s", name, exc_info=True)
            self._enabled = False

    def stop(self, name: str | None = None) -> None:
        for key, players in self._players.items():
            if name is None or key == name:
                for fx in players:
                    fx.stop()

    def stop_all(self) -> None:
        self.stop(None)
