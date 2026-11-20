"""The helpers that build the BlockCraft world off the game's own thread: they generate terrain regions and light and
mesh chunks, so walking into new land never stalls a frame.

Helper *processes* are used when the program runs from source (numpy holds the interpreter lock between its calls,
so threads share one core). Each has a feeder thread here that hands it one job at a time and waits for the
answer, so the game itself never touches a pipe: it only ``submit``s jobs and ``poll``s finished ones. A frozen
program, or a helper process that cannot start or dies, does the same work in the feeder thread instead."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading

log = logging.getLogger(__name__)

_generators: dict = {}
_lock = threading.Lock()
_start_lock = threading.Lock()


def _start_quietly(proc) -> None:  # noqa: ANN001 - a Process
    """Start a helper process that does not load the program's main script again (it needs none of it: only
    this package). Whatever the program was started from, a helper can then never open a second copy of it."""
    main = sys.modules.get("__main__")
    with _start_lock:
        saved = {k: getattr(main, k) for k in ("__spec__", "__file__") if hasattr(main, k)} if main else {}
        try:
            for k in saved:
                setattr(main, k, None)
            proc.start()
        finally:
            for k, v in saved.items():
                setattr(main, k, v)


def _generator(seed: int, dim: str):  # noqa: ANN202
    from app.arcade.games.blockcraft.worldgen import make_generator

    with _lock:
        gen = _generators.get((seed, dim))
        if gen is None:
            if len(_generators) > 4:
                _generators.clear()
            gen = _generators[seed, dim] = make_generator(seed, dim)
    return gen


def run_job(job: tuple) -> tuple:
    """Do one job: ("gen", world id, seed, rx, rz, realm) or ("mesh", world id, cx, cz, version, blocks, smooth)."""
    try:
        if job[0] == "gen":
            _, wid, seed, rx, rz, dim = job
            blk, biome = _generator(seed, dim).region(rx, rz)
            return "gen", wid, rx, rz, blk, biome
        from app.arcade.games.blockcraft.mesher import light_and_mesh

        _, wid, cx, cz, version, vol, smooth = job
        sky, bl, solid, wet = light_and_mesh(vol, smooth)
        return "mesh", wid, cx, cz, version, sky, bl, solid, wet
    except Exception as exc:  # noqa: BLE001 - the game drops the job and goes on
        where = (job[3], job[4]) if job[0] == "gen" else (job[2], job[3])
        return "error", job[1], job[0], where, repr(exc)


def _process_main(conn) -> None:  # noqa: ANN001 - a Connection
    try:
        while True:
            job = conn.recv()
            if job is None:
                break
            conn.send(run_job(job))
    except (EOFError, OSError, KeyboardInterrupt):
        pass


class ChunkPool:
    """A few helpers and the jobs they hold. ``workers`` 0 picks a number from the computer's cores."""

    def __init__(self, workers: int = 0, processes: bool | None = None) -> None:
        cores = os.cpu_count() or 2
        self.count = workers or max(1, min(4, cores // 2 - 1))
        if processes is None:
            processes = not getattr(sys, "frozen", False) and mp.parent_process() is None
        self.processes = bool(processes)
        if not self.processes:
            self.count = min(self.count, 3)
        self.jobs: queue.Queue = queue.Queue()
        self.done: queue.Queue = queue.Queue()
        self.pending = 0
        self.closed = False
        self.lost = False  # (kept for the world: jobs are never lost, a feeder finishes them itself)
        self.procs: list = []
        self.threads: list[threading.Thread] = []
        for i in range(self.count):
            t = threading.Thread(target=self._feed, daemon=True, name=f"blockcraft-chunks-{i}")
            t.start()
            self.threads.append(t)

    def _feed(self) -> None:
        """One helper: a process fed through a pipe, or (without one) the work done right here."""
        conn = proc = None
        if self.processes:
            try:
                ctx = mp.get_context("spawn")
                conn, child = ctx.Pipe()
                proc = ctx.Process(target=_process_main, args=(child,), daemon=True, name="blockcraft-chunks")
                _start_quietly(proc)
                child.close()
                self.procs.append(proc)
            except Exception:  # noqa: BLE001 - then this thread does the work
                log.info("BlockCraft helper process unavailable; using a thread", exc_info=True)
                conn = None
        while True:
            job = self.jobs.get()
            if job is None:
                break
            if conn is not None:
                try:
                    conn.send(job)
                    self.done.put(conn.recv())
                    continue
                except (EOFError, OSError, ValueError):
                    if self.closed:
                        break
                    log.warning("BlockCraft helper process lost; using a thread", exc_info=True)
                    conn = None
            self.done.put(run_job(job))
        if conn is not None:
            try:
                conn.send(None)
                conn.close()
            except (OSError, ValueError):
                pass

    @property
    def free(self) -> int:
        """How many more jobs the helpers should be given now."""
        return self.count * 2 - self.pending

    def submit(self, job: tuple) -> None:
        self.pending += 1
        self.jobs.put(job)

    def poll(self) -> list[tuple]:
        """The jobs finished since the last call."""
        out = []
        while True:
            try:
                out.append(self.done.get_nowait())
            except queue.Empty:
                break
        self.pending = max(0, self.pending - len(out))
        return out

    def close(self) -> None:
        """Stop the helpers (nothing is left running)."""
        if self.closed:
            return
        self.closed = True
        try:
            while True:
                self.jobs.get_nowait()
        except queue.Empty:
            pass
        for _ in self.threads:
            self.jobs.put(None)
        for t in self.threads:
            t.join(timeout=1.0)
        for proc in self.procs:
            proc.join(timeout=0.3)
            if proc.is_alive():
                proc.terminate()
        self.threads, self.procs = [], []
