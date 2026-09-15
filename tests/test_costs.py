# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ فاز ۲b — هزینهٔ واقعی داخلِ برچسب."""
import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import calib  # noqa: E402
import costs  # noqa: E402

from test_bracket_contract import synthetic_klines  # noqa: E402

H8 = costs.FUNDING_INTERVAL_MS


class TierTests(unittest.TestCase):
    def test_tiers_match_the_live_cost_table(self):
        self.assertEqual(costs.tier_cost(6e8), 0.08)
        self.assertEqual(costs.tier_cost(2e8), 0.11)
        self.assertEqual(costs.tier_cost(5e7), 0.18)
        self.assertEqual(costs.tier_cost(1e6), 0.30)
        self.assertEqual(costs.tier_cost(None), 0.30)

    def test_point_in_time_volume_uses_only_the_past(self):
        c = np.full(100, 10.0)
        v = np.concatenate([np.full(50, 1.0), np.full(50, 1e9)])   # حجم بعدِ کندلِ ۵۰ منفجر می‌شود
        early = costs.quote_volume_24h(c, v, 40, "1h")
        late = costs.quote_volume_24h(c, v, 90, "1h")
        self.assertAlmostEqual(early, 24 * 10.0)        # هیچ اثری از آینده
        self.assertGreater(late, early)
        self.assertEqual(costs.point_in_time_tier(c, v, 40, "1h"), 0.30)
        self.assertEqual(costs.point_in_time_tier(c, v, 90, "1h"), 0.08)


class FundingTests(unittest.TestCase):
    def test_settlement_count(self):
        self.assertEqual(costs.settlements_between(0, H8 - 1), 0)
        self.assertEqual(costs.settlements_between(1, H8), 1)
        self.assertEqual(costs.settlements_between(1, 3 * H8), 3)
        self.assertEqual(costs.settlements_between(5, 5), 0)

    def test_known_history_is_summed_exactly_and_is_side_aware(self):
        rows = [[H8, 0.0001], [2 * H8, 0.0003], [3 * H8, -0.0002]]
        self.assertAlmostEqual(costs.funding_cost_pct("long", 0, 3 * H8, rows), 0.02)
        self.assertAlmostEqual(costs.funding_cost_pct("short", 0, 3 * H8, rows), -0.02)
        self.assertAlmostEqual(costs.funding_cost_pct(1, 0, H8, rows), 0.01)

    def test_missing_history_is_always_charged_never_credited(self):
        """ندانستنِ فاندینگ نباید به سودِ فرضی تبدیل شود."""
        long_cost = costs.funding_cost_pct("long", 0, 3 * H8, None)
        short_cost = costs.funding_cost_pct("short", 0, 3 * H8, None)
        self.assertAlmostEqual(long_cost, 3 * costs.DEFAULT_FUNDING_PER_SETTLEMENT_PCT)
        self.assertAlmostEqual(short_cost, long_cost)
        self.assertGreater(short_cost, 0)

    def test_history_that_starts_after_the_trade_falls_back_to_the_default(self):
        rows = [[10 * H8, 0.0005]]                     # تاریخچه بعد از معامله شروع می‌شود
        self.assertAlmostEqual(costs.funding_cost_pct("long", 0, 3 * H8, rows),
                               3 * costs.DEFAULT_FUNDING_PER_SETTLEMENT_PCT)

    def test_forty_day_hold_costs_about_the_whole_reward(self):
        forty_days = 40 * 86_400_000
        cost = costs.funding_cost_pct("long", 0, forty_days, None)
        self.assertAlmostEqual(cost, 1.2, places=2)


class LabelsCarryRealCostTests(unittest.TestCase):
    def test_every_setup_event_carries_its_own_cost(self):
        kl = synthetic_klines(n=2000, seed=5)
        events, *_ = calib.extract_events("TESTUSDT", kl, "1h", [], {}, None, None)
        self.assertGreater(len(events), 5)
        for e in events:
            self.assertIn("cost_pct", e)
            # حجمِ مصنوعی کوچک است ⇒ ردهٔ ۰٫۳۰ + دست‌کم یک تسویهٔ فاندینگ
            self.assertGreaterEqual(e["cost_pct"], 0.30)

    def test_longer_holds_pay_more_funding(self):
        kl = synthetic_klines(n=2000, seed=5)
        events, *_ = calib.extract_events("TESTUSDT", kl, "1h", [], {}, None, None)
        by_len = sorted(events, key=lambda e: e["bars_held"])
        self.assertLessEqual(by_len[0]["cost_pct"], by_len[-1]["cost_pct"])

    def test_dense_direction_samples_carry_a_cost_column(self):
        kl = synthetic_klines(n=2000, seed=5)
        _events, _z, _dx, _dy, dr = calib.extract_events("TESTUSDT", kl, "1h", [], {}, None, None)
        self.assertGreater(len(dr), 0)
        self.assertEqual(len(dr[0]), 4)
        self.assertGreater(dr[0][3], 0)

    def test_edge_model_uses_per_event_cost_not_the_flat_default(self):
        rng = np.random.RandomState(17)
        cheap, dear = [], []
        for i in range(1500):
            x = rng.normal(size=6)
            r = 0.62 * x[0] + 0.12 * rng.normal() + 0.15
            base = {"ts": i * calib.TF_MS["1h"], "feats": x.tolist(),
                    "risk_pct": 1.0, "r": r, "regime": "trend"}
            cheap.append(dict(base, cost_pct=0.08))
            dear.append(dict(base, cost_pct=0.60))
        m_cheap = calib._fit_edge_model(cheap, "1h", calib.COST_PCT)
        m_dear = calib._fit_edge_model(dear, "1h", calib.COST_PCT)
        self.assertIsNotNone(m_cheap)
        self.assertIsNotNone(m_dear)
        # بازدهٔ خالصِ واقعیِ پنجرهٔ آزمون مستقیم اندازه گرفته می‌شود: با ریسکِ ۱٪،
        # اختلافِ هزینهٔ ۰٫۵۲٪ باید تقریباً ۰٫۵۲R بازده را کم کند (clip ممکن است کمی بخورد).
        cheap_r = m_cheap["by_regime"]["trend"]["avg_net_r"]
        dear_r = m_dear["by_regime"]["trend"]["avg_net_r"]
        self.assertAlmostEqual(cheap_r - dear_r, 0.52, delta=0.05)

    def test_model_version_bumped_for_the_cost_change(self):
        self.assertGreaterEqual(calib.CALIB_VERSION, 20)


if __name__ == "__main__":
    unittest.main()
