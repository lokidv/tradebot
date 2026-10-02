# -*- coding: utf-8 -*-
"""ردیابِ سایهٔ فاندینگ‌کَری: ثبتِ بدونِ تکرار، حسابِ بازده، حکمِ ازپیش‌ثبت‌شده."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import carry  # noqa: E402

START = carry.TRACKING_START_MS
H8 = 8 * 3600_000


def _rows(n, rate=0.0001, first=START):
    return [[first + i * H8, rate] for i in range(n)]


class LedgerTests(unittest.TestCase):
    def setUp(self):
        if os.path.exists(carry.LEDGER_PATH):
            os.remove(carry.LEDGER_PATH)

    def test_records_each_settlement_once_and_ignores_the_past(self):
        rows = [[START - H8, 0.01]] + _rows(5)
        self.assertEqual(carry.record("BTCUSDT", rows), 5)
        self.assertEqual(carry.record("BTCUSDT", rows), 0)
        self.assertEqual(carry.record("ETHUSDT", rows), 5)
        self.assertEqual(len(carry._read()), 10)

    def test_a_later_fetch_only_adds_the_new_settlements(self):
        carry.record("BTCUSDT", _rows(3))
        self.assertEqual(carry.record("BTCUSDT", _rows(6)), 3)


class StatsTests(unittest.TestCase):
    def _led(self, rows):
        return [{"sym": "BTCUSDT", "ts": t, "rate": r} for t, r in rows]

    def test_empty_ledger_has_only_the_entry_cost(self):
        s = carry.stats([])
        self.assertEqual(s["verdict"], "IN_PROGRESS")
        self.assertAlmostEqual(s["net_return_pct"], -0.075, places=3)

    def test_return_is_half_the_funding_minus_one_entry_cost(self):
        s = carry.stats(self._led(_rows(30, 0.0001)))
        self.assertAlmostEqual(s["funding_pct"], 30 * 0.0001 / 2 * 100, places=3)
        self.assertAlmostEqual(s["net_return_pct"], (30 * 0.0001 / 2 - carry.ENTRY_COST_CAPITAL) * 100, delta=0.001)
        self.assertEqual(s["mean_rate_annual_pct"], round(0.0001 * 3 * 365 * 100, 2))

    def test_negative_funding_loses_and_is_counted(self):
        s = carry.stats(self._led(_rows(6, -0.0002)))
        self.assertLess(s["net_return_pct"], 0)
        self.assertEqual(s["negative_settlements_pct"], 100.0)

    def test_in_progress_before_84_days(self):
        s = carry.stats(self._led(_rows(30)))
        self.assertEqual(s["verdict"], "IN_PROGRESS")

    def test_go_small_after_84_days_when_positive_and_fast_enough(self):
        s = carry.stats(self._led(_rows(3 * 90, 0.0001)))     # ~5.5٪ سالانه روی سرمایه
        self.assertEqual(s["verdict"], "GO_SMALL")

    def test_no_go_after_84_days_when_the_pace_is_too_low(self):
        s = carry.stats(self._led(_rows(3 * 90, 0.00001)))
        self.assertEqual(s["verdict"], "NO_GO")

    def test_brake_on_a_deep_drawdown(self):
        s = carry.stats(self._led(_rows(3, -0.02)))
        self.assertEqual(s["verdict"], "BRAKE")

    def test_duplicate_timestamps_are_not_double_counted(self):
        s = carry.stats(self._led(_rows(4) + _rows(4)))
        self.assertEqual(s["settlements"], 4)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        if os.path.exists(carry.LEDGER_PATH):
            os.remove(carry.LEDGER_PATH)

    def test_snapshot_records_and_reports_missing_data(self):
        def fetch(sym):
            return _rows(9) if sym == "BTCUSDT" else []
        snap = carry.snapshot(force=True, fetch=fetch)
        self.assertEqual(snap["assets"]["BTCUSDT"]["settlements"], 9)
        self.assertEqual(snap["missing_data"], ["ETHUSDT"])

    def test_a_failing_source_does_not_raise(self):
        def fetch(sym):
            raise RuntimeError("down")
        snap = carry.snapshot(force=True, fetch=fetch)
        self.assertEqual(sorted(snap["missing_data"]), ["BTCUSDT", "ETHUSDT"])


if __name__ == "__main__":
    unittest.main()
