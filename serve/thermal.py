"""Serving stops when an AMD GPU gets too hot: MAYA_THERMAL_ABORT_C=<°C> (unset: off).  For cards whose cooling cannot
hold them below their own limits for hours - passively cooled server cards (Radeon PRO V620, Instinct) in a
workstation or an open frame reach their junction limit long before anything slows them down.

Every amdgpu card's temperature sensors (its hwmon folder: edge, junction and memory) are read every
MAYA_THERMAL_EVERY_S seconds (2).  At the first reading at or above the limit the program says `THERMAL ABORT: ...`
(which sensor, how hot, which card), ends everything it started (SIGTERM, SIGKILL after 10 s) and exits with status
90, which a supervisor can tell from a crash (systemd: RestartPreventExitStatus=90 keeps the cards down instead of
starting them again) - the status opengfx1030/vllm-rdna's serve launchers exit with on their own THERMAL_ABORT_C.  A
start is refused while a card already reads the limit, or when no sensor can be read, and sensors that stop answering
for three readings in a row end the run too: with the setting on, nothing runs unwatched.

Every amdgpu card is watched, not only the ones the engine runs on: a wrong guess at which card is which would leave
the busy ones unwatched.  MAYA_THERMAL_GPUS (PCI addresses: 0000:6f:00.0,0000:72:00.0) narrows it on a PC whose other
AMD cards belong to something else.  The lowest limit set counts (the environment's or the config's "env"), so a
supervisor that sets one cannot be overridden upwards.  Linux (sysfs) only; NVIDIA cards are not read."""
from __future__ import annotations

import os
import re
import signal
import sys
import threading
import time

THERMAL_EXIT = 90                    # the exit status of a thermal abort
HWMON = "/sys/class/hwmon"
PROC = "/proc"
OFF = ("", "0", "off", "no", "false", "none")
MISSES = 3                           # readings in a row without a sensor that end the run


def pci(s) -> str | None:
    """"0000:6f:00.0", "6f:00.0" or "0000:6f:00" -> "0000:6f:00" (the card, its PCI function left out); None for
    anything else."""
    m = re.fullmatch(r"(?:([0-9a-f]{4,8}):)?([0-9a-f]{2}):([0-9a-f]{2})(?:\.[0-7])?", str(s or "").strip().lower())
    return f"{int(m.group(1) or '0', 16):04x}:{m.group(2)}:{m.group(3)}" if m else None


def _text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def readings(root: str | None = None, only: set | None = None) -> list[tuple[str, str, float]]:
    """(the card's PCI address, the sensor, °C) for every temperature sensor of every amdgpu hwmon folder; the sensor
    is its label (edge, junction, mem) or temp<N>.  `only`: the cards to read (pci() keys), None = every amdgpu card;
    `root`: the hwmon folder (HWMON)."""
    root = root or HWMON
    out = []
    try:
        folders = sorted(os.listdir(root))
    except OSError:
        return out
    for h in folders:
        d = os.path.join(root, h)
        if _text(os.path.join(d, "name")) != "amdgpu":
            continue
        card = os.path.basename(os.path.realpath(os.path.join(d, "device")))
        if only is not None and pci(card) not in only:
            continue
        try:
            files = sorted(os.listdir(d))
        except OSError:
            continue
        for f in files:
            m = re.fullmatch(r"temp(\d+)_input", f)
            v = _text(os.path.join(d, f)) if m else None
            if v is None or not re.fullmatch(r"-?\d+", v):
                continue
            label = _text(os.path.join(d, f"temp{m.group(1)}_label")) or f"temp{m.group(1)}"
            out.append((card, label, int(v) / 1000.0))
    return out


def hottest(rows, limit: float) -> str | None:
    """The hottest of `rows` (readings()) at or above `limit`, as "junction 91°C on 0000:6f:00.0"; None below it."""
    hot = [r for r in rows if r[2] >= limit]
    if not hot:
        return None
    card, label, t = max(hot, key=lambda r: r[2])
    return f"{label} {t:g}°C on {card}"


def settings(*envs) -> tuple:
    """(limit in °C or None when off, the cards to read or None for all, seconds between readings) from these
    environments - os.environ first, then a config's "env" (under the names the code reads, STRATA_*): the lowest
    limit set counts, the other two are the first one set.  ValueError for a value that is none of these."""
    limits, only, every = [], None, None
    for env in envs:
        v = str(env.get("STRATA_THERMAL_ABORT_C") or "").strip()
        if v.lower() not in OFF:
            if not re.fullmatch(r"\d+", v) or not 1 <= int(v) <= 150:
                raise ValueError(f"MAYA_THERMAL_ABORT_C={v!r}: a whole number of °C from 1 to 150 (0 or unset: off)")
            limits.append(int(v))
        g = str(env.get("STRATA_THERMAL_GPUS") or "").strip()
        if g and only is None:
            only = {pci(x) for x in g.split(",") if x.strip()}
            if None in only:
                raise ValueError(f"MAYA_THERMAL_GPUS={g!r}: PCI addresses, e.g. 0000:6f:00.0,0000:72:00.0")
        e = str(env.get("STRATA_THERMAL_EVERY_S") or "").strip()
        if e and every is None:
            try:
                every = float(e)
            except ValueError:
                every = -1.0
            if not 0.5 <= every <= 60:
                raise ValueError(f"MAYA_THERMAL_EVERY_S={e!r}: seconds between readings, 0.5 to 60")
    return (min(limits) if limits else None), only, (every or 2.0)


class Guard:
    """The sensors read against the limit: refusal() before a start, start() while it runs."""

    def __init__(self, limit: int, only: set | None = None, every: float = 2.0, root: str | None = None):
        self.limit, self.only, self.every, self.root = limit, only, every, root or HWMON
        self._stop = threading.Event()

    def describe(self) -> str:
        rows = readings(self.root, self.only)
        cards = len({r[0] for r in rows})
        return (f"stops at {self.limit}°C on any of {len(rows)} sensors of {cards} AMD GPU{'s' * (cards != 1)}, read "
                f"every {self.every:g} s (hottest now: {hottest(rows, -273) or '?'})")

    def refusal(self) -> str | None:
        """Why a start must not go ahead - no sensor to read, or a card at the limit already - or None."""
        rows = readings(self.root, self.only)
        if not rows:
            return ("no AMD GPU temperature sensor can be read (amdgpu hwmon under " + self.root +
                    (", MAYA_THERMAL_GPUS=" + ",".join(sorted(self.only)) if self.only is not None else "") +
                    "): MAYA_THERMAL_ABORT_C does not run unwatched")
        hot = hottest(rows, self.limit)
        return f"a GPU already reads {hot} (MAYA_THERMAL_ABORT_C={self.limit}): let it cool first" if hot else None

    def start(self, trip) -> None:
        """Reads the sensors every `every` seconds on a thread of its own until stop(); calls trip(why) once - at the
        first reading at or above the limit, or after MISSES readings in a row found no sensor."""
        def run():
            misses = 0
            while not self._stop.wait(self.every):
                rows = readings(self.root, self.only)
                misses = 0 if rows else misses + 1
                if misses >= MISSES:
                    trip(f"no GPU temperature sensor answered for {MISSES} readings in a row")
                    return
                hot = hottest(rows, self.limit)
                if hot:
                    trip(f"{hot} (MAYA_THERMAL_ABORT_C={self.limit})")
                    return
        threading.Thread(target=run, name="thermal", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


def from_env(*envs, root: str | None = None) -> Guard | None:
    """The guard these environments ask for (settings()), or None when MAYA_THERMAL_ABORT_C is off.  ValueError as
    settings()."""
    limit, only, every = settings(*envs)
    return None if limit is None else Guard(limit, only, every, root)


def _state(pid: int, proc: str = PROC) -> tuple | None:
    """(state, parent pid) of a process from /proc/<pid>/stat, None when it is gone."""
    s = _text(os.path.join(proc, str(pid), "stat"))
    if not s or ")" not in s:
        return None
    f = s.rsplit(")", 1)[1].split()                 # after the command's name, which may hold spaces or ')'
    return (f[0], int(f[1])) if len(f) > 1 and f[1].isdigit() else None


def descendants(pid: int | None = None, proc: str = PROC) -> list[int]:
    """The children of `pid` (this process) and theirs, the deepest first."""
    pid = os.getpid() if pid is None else pid
    kids: dict[int, list[int]] = {}
    try:
        names = os.listdir(proc)
    except OSError:
        return []
    for n in names:
        st = _state(int(n), proc) if n.isdigit() else None
        if st is not None:
            kids.setdefault(st[1], []).append(int(n))
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(0), []):
            out.append(c)
            todo.append(c)
    return out[::-1]


def _alive(pid: int, proc: str = PROC) -> bool:
    st = _state(pid, proc)
    return st is not None and st[0] != "Z"


def end_children(grace: float = 10.0, proc: str = PROC) -> list[int]:
    """SIGTERM to every descendant of this process (the engine, an image encoder, MCP servers), SIGKILL to the ones
    still there after `grace` seconds: the processes ended."""
    pids = descendants(proc=proc)
    for sig in (signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM)):
        for p in pids:
            if _alive(p, proc):
                try:
                    os.kill(p, sig)
                except OSError:
                    pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(_alive(p, proc) for p in pids):
            time.sleep(0.2)
    return pids


def abort(why: str, say=None, grace: float = 10.0) -> None:
    """The thermal abort, from any thread: says why, ends every process this one started and exits THERMAL_EXIT."""
    say = say or (lambda s: print(s, flush=True))
    say(f"[maya] THERMAL ABORT: {why}: ending the engine")
    ended = end_children(grace)
    say(f"[maya] THERMAL ABORT: {len(ended)} process{'es' * (len(ended) != 1)} ended; exiting with status "
        f"{THERMAL_EXIT}")
    for f in (sys.stdout, sys.stderr):
        try:
            f.flush()
        except (OSError, ValueError):
            pass
    os._exit(THERMAL_EXIT)
