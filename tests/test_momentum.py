# -*- coding: utf-8 -*-
"""مومنتومِ ۲۸روزهٔ زنده باید دقیقاً همان قاعدهٔ پژوهش (timing.H3_TSMOM28) باشد."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import momentum  # noqa: E402
import paper  # noqa: E402
import timing  # noqa: E402

DAY = momentum.DAY_MS
MON = 1_790_553_600_000                  # دوشنبه 2026-09-28


def _series(n=200, seed=2, end=MON + 3 * DAY):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    t = end - (n - np.arange(n, dtype=np.int64)) * DAY          # آخرین کندلِ بسته = دیروزِ end
    return t, c


class RuleParityTests(unittest.TestCase):
    def test_live_decisions_equal_the_research_rule_every_monday(self):
        t, c = _series(400)
        pos = timing.positions("H3_TSMOM28", t, c, None)
        checked = 0
        for d in range(40, len(t)):
            if momentum.time.gmtime(t[d] / 1000).tm_wday == 0:
                dec = momentum.decision_at(t, c, int(t[d]))
                self.assertEqual(dec["in_market"], bool(pos[d]))
                checked += 1
        self.assertGreater(checked, 40)

    def test_monday_helper(self):
        self.assertEqual(momentum.monday_on_or_before(MON + 3 * DAY + 5000), MON)
        self.assertEqual(momentum.monday_on_or_before(MON), MON)

    def test_state_reports_this_week_and_the_next_decision(self):
        t, c = _series()
        st = momentum.state(t, c, MON + 3 * DAY + 3600_000)
        self.assertEqual(st["current"]["monday_ms"], MON)
        self.assertEqual(st["next_decision_ms"], MON + 7 * DAY)
        self.assertIsNotNone(st["provisional_next"])


class GoNoGoTests(unittest.TestCase):
    """حکمِ فاز ۵ دقیقاً همان معیارِ ازپیش‌ثبت‌شده است."""

    def _rows(self, weekly, in_market):
        rows, prev = [], None
        for i, (g, inm) in enumerate(zip(weekly, in_market)):
            rows.append({"sym": "BTCUSDT", "monday_ms": MON + i * 7 * DAY, "in_market": inm,
                         "prev_in_market": prev, "prev_week_market_ret": g if prev is not None else None})
            prev = inm
        return rows

    def test_in_progress_before_twelve_weeks(self):
        f = momentum.forward_stats(self._rows([0.01] * 5, [True] * 5))["BTCUSDT"]
        self.assertEqual(f["verdict"], "IN_PROGRESS")

    def test_go_when_close_to_buy_and_hold_with_smaller_drawdown(self):
        weekly = [0.03, -0.10, -0.10, 0.05, 0.04, 0.03, 0.02, 0.03, 0.01, 0.02, 0.02, 0.01, 0.02, 0.01]
        inm = [True, False, False, True, True, True, True, True, True, True, True, True, True, True]
        f = momentum.forward_stats(self._rows(weekly, inm))["BTCUSDT"]
        self.assertGreaterEqual(f["weeks"], 12)
        self.assertEqual(f["verdict"], "GO_SMALL")

    def test_no_go_when_it_trails_buy_and_hold(self):
        weekly = [0.05] * 14
        inm = [True, False] * 7
        f = momentum.forward_stats(self._rows(weekly, inm))["BTCUSDT"]
        self.assertEqual(f["verdict"], "NO_GO")

    def test_brake_when_live_drawdown_exceeds_one_and_a_half_times_the_test(self):
        weekly = [-0.2, -0.2, -0.2]
        f = momentum.forward_stats(self._rows(weekly, [True] * 3), conf_dd={"BTCUSDT": 20.0})["BTCUSDT"]
        self.assertEqual(f["verdict"], "BRAKE")


class DemoTests(unittest.TestCase):
    def setUp(self):
        for p in (paper.PATH, paper._ledger_path(), momentum.LEDGER_PATH):
            if os.path.exists(p):
                os.remove(p)
        paper.reset_wallet(10_000)

    def _snap(self, in_market):
        return {"assets": {"BTCUSDT": {"current": {"in_market": in_market, "monday_ms": MON}}}}

    def test_demo_buys_only_when_in_market_and_sizes_by_allocation(self):
        with self.assertRaises(ValueError):
            momentum.open_demo("BTCUSDT", 20, snap=self._snap(False), price=100.0)
        pos = momentum.open_demo("BTCUSDT", 20, snap=self._snap(True), price=100.0)
        self.assertEqual((pos["strategy"], pos["market"], pos["tp"]), ("tsmom28", "spot", None))
        self.assertAlmostEqual(pos["size_usdt"], 2000.0)
        with self.assertRaises(ValueError):
            momentum.open_demo("BTCUSDT", 20, snap=self._snap(True), price=100.0)

    def test_a_cash_decision_closes_the_demo_at_the_monday_open(self):
        pos = momentum.open_demo("BTCUSDT", 20, snap=self._snap(True), price=100.0)
        res = momentum.sync_demo(snap=self._snap(False), bar_fn=lambda s: {"t": MON, "o": 104.0, "c": 99.0})
        self.assertEqual(res["closed"], 1)
        closed = [p for p in paper.list_positions()["closed"] if p["id"] == pos["id"]][0]
        self.assertEqual(closed["exit_price"], 104.0)

    def test_ledger_records_each_monday_once(self):
        t, c = _series(260, end=MON + 15 * DAY)
        self.assertEqual(momentum.record("BTCUSDT", t, c), 3)          # 09-28، 10-05، 10-12
        self.assertEqual(momentum.record("BTCUSDT", t, c), 0)
        self.assertEqual(momentum.forward_stats()["BTCUSDT"]["weeks"], 3)


if __name__ == "__main__":
    unittest.main()
