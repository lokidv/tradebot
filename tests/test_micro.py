# -*- coding: utf-8 -*-
"""سیگنال‌های ۱۵دقیقه‌ای: هیچ‌کدام از آینده نمی‌خوانند و معامله دقیقاً طبقِ قاعدهٔ ثبت‌شده اجرا می‌شود."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import micro  # noqa: E402

T0 = 1_640_995_200_000       # 2022-01-01 (شنبه)


def _k(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    w = np.abs(rng.normal(0, 0.001, n)) * c
    v = rng.lognormal(3, 0.6, n)
    return {"t": T0 + np.arange(n, dtype=np.int64) * micro.BAR_MS, "o": o, "h": np.maximum(o, c) + w,
            "l": np.minimum(o, c) - w, "c": c, "v": v, "tbv": v * rng.uniform(0.3, 0.7, n)}


class NoLookAheadTests(unittest.TestCase):
    def test_future_bars_never_change_a_signal(self):
        k = _k()
        base = {h: micro.signals(h, k) for h in micro.HYPOTHESES}
        d = 3500
        k2 = {key: v.copy() for key, v in k.items()}
        for col in ("o", "h", "l", "c"):
            k2[col][d + 1:] *= 1.3
        k2["v"][d + 1:] *= 9
        k2["tbv"][d + 1:] = k2["v"][d + 1:] * 0.95
        for h in micro.HYPOTHESES:
            self.assertTrue(np.array_equal(base[h][:d + 1], micro.signals(h, k2)[:d + 1]), h)

    def test_z_score_excludes_the_current_bar(self):
        x = np.arange(3000, dtype=float)
        z = micro._prior_z(x, look=10)
        prior = x[2989:2999]
        self.assertAlmostEqual(z[2999], (x[2999] - prior.mean()) / prior.std(ddof=1))


class TradeTests(unittest.TestCase):
    def test_entry_next_open_exit_after_eight_bars_and_cost(self):
        k = _k(200)
        k["h"] = np.maximum(k["h"], k["c"]) * 1.0001
        sig = np.zeros(200)
        sig[50] = 1
        tr = micro.trades(k, sig, cost=0.14)
        self.assertEqual(len(tr), 1)
        t, side, gross, net = tr[0]
        entry, exit_ = k["o"][51], k["c"][58]
        if not (k["l"][51:59] <= entry - 3 * micro.engine.atr(k["h"], k["l"], k["c"], 14)[50]).any():
            self.assertAlmostEqual(gross, (exit_ / entry - 1) * 100)
        self.assertAlmostEqual(net, gross - 0.14)

    def test_stop_is_checked_intrabar_and_signals_during_a_trade_are_ignored(self):
        k = _k(200)
        sig = np.zeros(200)
        sig[50] = -1
        sig[52] = 1                                     # در حینِ معامله ⇒ نادیده
        k["h"][53] = k["o"][51] * 1.5                   # حدضررِ شورت درون‌کندل
        tr = micro.trades(k, sig, cost=0.0)
        self.assertEqual(len(tr), 1)
        self.assertLess(tr[0][2], 0)

    def test_us_open_signal_fires_only_on_the_1400_utc_bar_on_weekdays(self):
        k = _k(96 * 9)
        s = micro.signals("M5_US_OPEN", k)
        idx = np.flatnonzero(s != 0)
        tod = (k["t"][idx] % micro.DAY_MS) // micro.BAR_MS
        wd = ((k["t"][idx] // micro.DAY_MS) + 3) % 7
        self.assertTrue((tod == 55).all() and (wd < 5).all())
        self.assertGreater(len(idx), 3)


if __name__ == "__main__":
    unittest.main()
