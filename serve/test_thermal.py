"""MAYA_THERMAL_ABORT_C (serve/thermal.py): a fake /sys/class/hwmon of amdgpu cards (and one sensor that is not a
GPU's), the limit rules, the guard's trip, the start refusals, the processes ended and the exit status - no GPU
needed.  Linux only: the guard reads sysfs and /proc."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import thermal as TH  # noqa: E402


def card(hwmon: Path, n: int, bdf: str, temps: dict, name: str = "amdgpu") -> Path:
    """hwmon<n> -> a card's device folder named `bdf`, with temp<i>_input / _label for {label: millidegrees}."""
    dev = hwmon.parent / "devices" / bdf
    dev.mkdir(parents=True, exist_ok=True)
    h = hwmon / f"hwmon{n}"
    h.mkdir(parents=True)
    (h / "name").write_text(name + "\n")
    (h / "device").symlink_to(dev)
    for i, (label, milli) in enumerate(temps.items(), 1):
        (h / f"temp{i}_label").write_text(label + "\n")
        (h / f"temp{i}_input").write_text(f"{milli}\n")
    return h


def set_temp(h: Path, label: str, milli: int) -> None:
    for f in h.glob("temp*_label"):
        if f.read_text().strip() == label:
            (h / f.name.replace("_label", "_input")).write_text(f"{milli}\n")
            return
    raise KeyError(label)


@unittest.skipUnless(sys.platform.startswith("linux"), "sysfs and /proc")
class Thermal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.hw = Path(self.tmp.name) / "hwmon"
        self.hw.mkdir()
        # two V620s as amdgpu shows them (edge, junction, mem) and the CPU's own sensor, which is not watched
        self.a = card(self.hw, 0, "0000:64:00.0", {"edge": 41000, "junction": 45000, "mem": 44000})
        self.b = card(self.hw, 1, "0000:6f:00.0", {"edge": 40000, "junction": 47000, "mem": 43000})
        card(self.hw, 2, "coretemp.0", {"Package id 0": 99000}, name="coretemp")

    def test_readings_are_the_amdgpu_sensors_only(self):
        rows = TH.readings(str(self.hw))
        self.assertEqual(len(rows), 6)
        self.assertIn(("0000:6f:00.0", "junction", 47.0), rows)
        self.assertFalse(any(r[0].startswith("coretemp") for r in rows))
        only = {TH.pci("6f:00.0")}
        self.assertEqual({r[0] for r in TH.readings(str(self.hw), only)}, {"0000:6f:00.0"})

    def test_pci_addresses(self):
        for s in ("0000:6f:00.0", "6f:00.0", "0000:6F:00", "00000000:6f:00.0"):
            self.assertEqual(TH.pci(s), "0000:6f:00", s)
        for s in ("", "6f", "gpu0", "0000:6f:00.9"):
            self.assertIsNone(TH.pci(s), s)

    def test_the_limit_is_at_or_above(self):
        set_temp(self.b, "mem", 89999)
        rows = TH.readings(str(self.hw))
        self.assertIsNone(TH.hottest(rows, 90))
        set_temp(self.b, "mem", 90000)
        self.assertEqual(TH.hottest(TH.readings(str(self.hw)), 90), "mem 90°C on 0000:6f:00.0")
        set_temp(self.a, "junction", 95500)
        self.assertEqual(TH.hottest(TH.readings(str(self.hw)), 90), "junction 95.5°C on 0000:64:00.0")

    def test_settings(self):
        self.assertEqual(TH.settings({}, {}), (None, None, 2.0))
        for off in ("", "0", "off", "none"):
            self.assertIsNone(TH.settings({"STRATA_THERMAL_ABORT_C": off})[0])
        # the lowest limit set counts: a config cannot raise a supervisor's, nor switch it off
        self.assertEqual(TH.settings({"STRATA_THERMAL_ABORT_C": "90"}, {"STRATA_THERMAL_ABORT_C": "95"})[0], 90)
        self.assertEqual(TH.settings({"STRATA_THERMAL_ABORT_C": "90"}, {"STRATA_THERMAL_ABORT_C": "85"})[0], 85)
        self.assertEqual(TH.settings({"STRATA_THERMAL_ABORT_C": "90"}, {"STRATA_THERMAL_ABORT_C": "0"})[0], 90)
        lim, only, every = TH.settings({"STRATA_THERMAL_GPUS": "0000:6f:00.0, 72:00.0", "STRATA_THERMAL_EVERY_S": "1"},
                                       {"STRATA_THERMAL_GPUS": "0000:64:00.0"})
        self.assertEqual((only, every), ({"0000:6f:00", "0000:72:00"}, 1.0))
        for bad in ({"STRATA_THERMAL_ABORT_C": "ninety"}, {"STRATA_THERMAL_ABORT_C": "200"},
                    {"STRATA_THERMAL_GPUS": "card0"}, {"STRATA_THERMAL_EVERY_S": "0.1"}):
            with self.assertRaises(ValueError):
                TH.settings(bad)

    def test_refusals(self):
        g = TH.Guard(90, root=str(self.hw))
        self.assertIsNone(g.refusal())
        self.assertIn("stops at 90°C on any of 6 sensors of 2 AMD GPUs", g.describe())
        set_temp(self.a, "junction", 92000)
        self.assertIn("already reads junction 92°C on 0000:64:00.0", g.refusal())
        self.assertIn("no AMD GPU temperature sensor", TH.Guard(90, root=str(self.hw / "nothing")).refusal())
        self.assertIn("MAYA_THERMAL_GPUS=0000:99:00", TH.Guard(90, {"0000:99:00"}, root=str(self.hw)).refusal())

    def test_trips_once_at_the_limit(self):
        g = TH.Guard(90, every=0.05, root=str(self.hw))
        why, done = [], threading.Event()
        g.start(lambda w: (why.append(w), done.set()))
        time.sleep(0.2)
        self.assertEqual(why, [])
        set_temp(self.b, "junction", 90000)
        self.assertTrue(done.wait(5))
        time.sleep(0.2)
        self.assertEqual(why, ["junction 90°C on 0000:6f:00.0 (MAYA_THERMAL_ABORT_C=90)"])

    def test_trips_when_the_sensors_stop_answering(self):
        g = TH.Guard(90, every=0.05, root=str(self.hw))
        why, done = [], threading.Event()
        g.start(lambda w: (why.append(w), done.set()))
        for h in (self.a, self.b):
            for f in h.glob("temp*_input"):
                f.unlink()
        self.assertTrue(done.wait(5))
        self.assertIn(f"for {TH.MISSES} readings in a row", why[0])

    def test_stop_ends_the_watch(self):
        g = TH.Guard(90, every=0.05, root=str(self.hw))
        why = []
        g.start(why.append)
        g.stop()
        set_temp(self.a, "mem", 99000)
        time.sleep(0.3)
        self.assertEqual(why, [])

    def test_end_children_ends_every_descendant(self):
        # in a process of its own (end_children ends every descendant - here it would end other tests' helpers):
        # a child that ends on SIGTERM, one that ignores it (SIGKILL after the grace), and a grandchild
        script = r"""
import json, signal, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from serve import thermal as TH
stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
kids = [subprocess.Popen(["sleep", "60"]), subprocess.Popen([sys.executable, "-c", stubborn]),
        subprocess.Popen(["sh", "-c", "sleep 60 & wait"])]
time.sleep(0.5)
grand = [p for p in TH.descendants() if p not in {k.pid for k in kids}]
ended = TH.end_children(grace=1.0)
codes = [k.wait(5) for k in kids]
print(json.dumps({"kids": [k.pid for k in kids], "grand": grand, "ended": ended, "codes": codes,
                  "alive": [p for p in grand if TH._alive(p)]}))
"""
        r = subprocess.run([sys.executable, "-c", script, str(ROOT)], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(out["grand"], "the shell's sleep")
        self.assertTrue(set(out["kids"]) <= set(out["ended"]))
        self.assertEqual(out["codes"][1], -signal.SIGKILL)
        self.assertEqual(out["alive"], [])

    def test_abort_exits_with_the_thermal_status(self):
        script = ("import subprocess, sys; sys.path.insert(0, sys.argv[1]); from serve import thermal as TH; "
                  "k = subprocess.Popen(['sleep', '60']); print(k.pid, flush=True); TH.abort('test reason', grace=2)")
        r = subprocess.run([sys.executable, "-c", script, str(ROOT)], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, TH.THERMAL_EXIT, r.stderr)
        pid = int(r.stdout.split()[0])
        self.assertIn("THERMAL ABORT: test reason: ending the engine", r.stdout)
        self.assertIn("exiting with status 90", r.stdout)
        self.assertFalse(TH._alive(pid))

    def test_server_refuses_a_start_it_cannot_watch(self):
        from serve import server as SV
        with mock.patch.object(TH, "HWMON", str(self.hw)):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("STRATA_THERMAL_ABORT_C", None)
                self.assertIsNone(SV.thermal_guard({}))                         # off: nothing read
                g = SV.thermal_guard({"env": {"STRATA_THERMAL_ABORT_C": "90"}})  # the config's
                self.assertEqual(g.limit, 90)
                os.environ["STRATA_THERMAL_ABORT_C"] = "80"                      # the environment's, lower
                self.assertEqual(SV.thermal_guard({"env": {"STRATA_THERMAL_ABORT_C": "90"}}).limit, 80)
                set_temp(self.b, "edge", 85000)
                with self.assertRaises(SystemExit) as e:
                    SV.thermal_guard({})
                self.assertIn("not serving: a GPU already reads edge 85°C", str(e.exception))
                os.environ["STRATA_THERMAL_ABORT_C"] = "hot"
                with self.assertRaises(SystemExit):
                    SV.thermal_guard({})


if __name__ == "__main__":
    unittest.main()
