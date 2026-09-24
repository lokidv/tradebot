# -*- coding: utf-8 -*-
"""مسیرِ زنده: دفترها، زمان‌بند و پشتیبان‌ها (یافته‌های LP-* ممیزی).

همه‌چیز stub است: هیچ درخواستِ شبکه و هیچ فایلِ bot/data واقعی.
"""
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import bracket  # noqa: E402
import candidates  # noqa: E402
import engine  # noqa: E402
import meta_gate  # noqa: E402
import report  # noqa: E402

H = 3_600_000


def _bars(n, t0, step=H, start=100.0, drift=0.5, wick=0.3):
    c = start + drift * np.arange(n)
    return {"t": (t0 + np.arange(n, dtype=np.int64) * step).tolist(), "o": c.tolist(),
            "h": (c + wick).tolist(), "l": (c - wick).tolist(), "c": c.tolist(),
            "v": np.full(n, 1.0).tolist()}


def _row(**over):
    row = {"symbol": "BTCUSDT", "side": "long", "setup": None, "setup_observed": False,
           "zt": int(time.time() * 1000) - 60_000, "entry": 100.0, "sl": 97.4, "tp": 104.68,
           "atr14": 2.0, "cost": 0.11, "tradeable": False}
    row.update(over)
    return row


class _Ledger:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = (candidates.CAND_PATH, candidates.RESULT_PATH)
        candidates.CAND_PATH = os.path.join(self.tmp.name, "candidates.jsonl")
        candidates.RESULT_PATH = os.path.join(self.tmp.name, "results.jsonl")
        candidates._reset_cache()

    def tearDown(self):
        candidates.CAND_PATH, candidates.RESULT_PATH = self._old
        candidates._reset_cache()
        self.tmp.cleanup()


# ───────────────────────── LP-2: ستاپِ معلق اسکن را تمام نمی‌کند ─────────────────────────
class SuspendedSetupScanTests(unittest.TestCase):
    N = 420

    def _analyze(self, signals, suspended):
        c = 100 + np.linspace(0, 4, self.N) + 0.3 * np.sin(np.arange(self.N) / 7)
        kl = {"t": (np.arange(self.N, dtype=np.int64) * H).tolist(), "o": c.tolist(),
              "h": (c + 0.4).tolist(), "l": (c - 0.4).tolist(), "c": c.tolist(),
              "v": np.full(self.N, 1_000.0).tolist()}
        sug = {"side": "long", "entry": 104.0, "sl": 103.0, "tp": 105.8, "risk_pct": 0.96,
               "rr": 1.8, "grade": "A", "viable": True, "tradeable": True, "status": "آماده",
               "reasons": [], "time_stop_min": 2_400}
        with mock.patch.object(engine, "setup_signal",
                               side_effect=lambda _cs, _o, _h, _l, _c, i: signals.get(i, (0, None))), \
                mock.patch.object(engine, "trade_suggestion", side_effect=lambda *a, **k: dict(sug)), \
                mock.patch.object(engine, "quick_backtest",
                                  return_value={"n": 0, "win_rate": None, "avg_r": None,
                                                "timeouts": 0, "profit_factor": None}), \
                mock.patch.object(meta_gate, "evaluate", return_value={"approve": False}):
            return engine.analyze(kl, "1h", extras={"symbol": "BTCUSDT", "suspended": suspended})["trade"]

    def test_suspended_setup_on_last_bar_lets_the_scan_reach_the_previous_bar(self):
        sig = {self.N - 1: (1, "fd"), self.N - 2: (1, "sq")}
        self.assertEqual(self._analyze(sig, {})["setup"], "fd")          # بی‌تعلیق: کندلِ آخر
        tr = self._analyze(sig, {"fd": -0.2})
        self.assertEqual(tr["setup"], "sq")                               # break اینجا None می‌داد
        self.assertTrue(tr["setup_observed"])

    def test_suspended_setup_alone_is_as_if_there_was_no_signal(self):
        tr = self._analyze({self.N - 1: (1, "fd")}, {"fd": -0.2})
        self.assertIsNone(tr.get("setup"))
        self.assertFalse(tr.get("setup_observed"))


if __name__ == "__main__":
    unittest.main()
