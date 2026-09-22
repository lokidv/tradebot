# -*- coding: utf-8 -*-
"""زمان‌بندیِ BTC/ETH: هیچ تصمیمی از آینده نمی‌خواند، و آمارِ «لبه» همان است که ادعا می‌کند."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import timing  # noqa: E402

DAY = timing.DAY_MS
T0 = 1_577_836_800_000        # 2020-01-01


def _series(n=800, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n)))
    t = T0 + np.arange(n, dtype=np.int64) * DAY
    fund = np.array([[T0 + k * 8 * 3_600_000, rng.normal(1e-4, 1e-4)] for k in range(n * 3)])
    return t, c, fund


class NoLookAheadTests(unittest.TestCase):
    def test_changing_today_or_later_never_changes_todays_position(self):
        t, c, fund = _series()
        f7 = timing.funding_7d(t, fund)
        for hyp in timing.HYPOTHESES:
            base = timing.positions(hyp, t, c, f7)
            for d in (400, 555, 700):
                c2 = c.copy()
                c2[d:] *= 3.0                                    # آینده از d به بعد عوض شد
                fund2 = fund.copy()
                fund2[fund2[:, 0] > t[d], 1] = 0.05             # فاندینگِ پس از لحظهٔ تصمیمِ d
                p2 = timing.positions(hyp, t, c2, timing.funding_7d(t, fund2))
                self.assertTrue(np.array_equal(base[:d + 1], p2[:d + 1]), (hyp, d))

    def test_funding_window_uses_only_settlements_up_to_the_decision(self):
        t, _c, fund = _series(n=60)
        f7 = timing.funding_7d(t, fund)
        d = 30
        m = (fund[:, 0] <= t[d]) & (fund[:, 0] > t[d] - 7 * DAY)
        self.assertAlmostEqual(f7[d], fund[m, 1].mean())


class StatisticTests(unittest.TestCase):
    def test_edge_series_mean_equals_in_minus_out(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.02, 500)
        pos = (rng.random(500) < 0.4).astype(float)
        e = timing.edge_series(pos, r)
        self.assertAlmostEqual(e.mean(), r[pos == 1].mean() - r[pos == 0].mean(), places=12)

    def test_switch_costs_are_charged_on_every_change(self):
        pos = np.array([0, 1, 1, 0, 1.0])
        r = np.zeros(5)
        self.assertAlmostEqual(timing.strategy_returns(pos, r, cost=0.002).sum(), -0.002 * 3)

    def test_decision_rule(self):
        good = {"edge_daily_pct": 0.05, "strategy": {"sharpe": 1.1}, "buy_and_hold": {"sharpe": 0.9}}
        bad = dict(good, edge_daily_pct=-0.01)
        self.assertEqual(timing.decide(good, good, 0.05), "ADOPT")
        self.assertEqual(timing.decide(good, good, 0.30), "CONSISTENT")
        self.assertEqual(timing.decide(good, bad, 0.01), "REJECT")
        worse = dict(good, strategy={"sharpe": 0.5})
        self.assertEqual(timing.decide(good, worse, 0.05), "CONSISTENT")

    def test_windows_match_the_preregistration(self):
        import json
        with open(os.path.join(ROOT, "bot", "data", "research", "prereg_btc_eth_timing.json"), encoding="utf-8") as f:
            pre = json.load(f)
        self.assertEqual(pre["windows"]["discovery"], ["2021-01-01", "2023-06-30"])
        self.assertEqual(timing.DISCOVERY, (1_609_459_200_000, 1_688_169_600_000))
        self.assertEqual(timing.CONFIRMATION[1], 1_788_220_800_000)          # 2026-09-01 انحصاری


if __name__ == "__main__":
    unittest.main()
