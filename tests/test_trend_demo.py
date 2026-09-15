# -*- coding: utf-8 -*-
"""«خرید در دمو» برای سیگنالِ روند: همان حدضررِ قاعده، بی‌هدف، اسپات، و حدضررِ دنباله‌دار.

پوزیشنِ بی‌هدف قبلاً اصلاً ساخته نمی‌شد (paper هدف را اجباری می‌کرد) و مشاور با هدفِ
None می‌شکست؛ این تست‌ها هر دو را می‌پوشانند.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import advisor  # noqa: E402
import explore  # noqa: E402
import paper  # noqa: E402
import trend  # noqa: E402


def _row(sym="FILUSDT", category="join", stop=0.773, last=0.936, days=1, in_position=True):
    return {"sym": sym, "category": category, "stop": stop, "last": last, "days_in": days,
            "in_position": in_position, "stop_distance_pct": (last - stop) / last * 100}


class DemoTests(unittest.TestCase):
    def setUp(self):
        for p in (paper.PATH, paper._ledger_path()):
            if os.path.exists(p):
                os.remove(p)
        paper.reset_wallet(10_000)

    def test_opens_a_spot_trend_position_sized_by_risk_without_target(self):
        pos = trend.open_demo("FILUSDT", risk_pct=0.5, price=0.936, now_rows=[_row()])
        self.assertIsNone(pos["tp"])
        self.assertEqual((pos["strategy"], pos["market"], pos["side"], pos["tf"]),
                         ("trend20", "spot", "long", "1d"))
        self.assertEqual(pos["sl"], 0.773)
        self.assertEqual(pos["cost_pct"], explore.SPOT_ROUND_TRIP_PCT)
        # ضررِ حداکثر ≈ ۰٫۵٪ موجودی: ارزش × فاصلهٔ حدضرر
        self.assertAlmostEqual(pos["size_usdt"] * (0.936 - 0.773) / 0.936, 50.0, delta=0.1)

    def test_only_todays_actionable_signals_can_be_opened(self):
        with self.assertRaises(ValueError):
            trend.open_demo("BTCUSDT", price=100, now_rows=[_row("BTCUSDT", category="late", stop=90, last=100, days=40)])
        with self.assertRaises(ValueError):
            trend.open_demo("XRPUSDT", price=1, now_rows=[_row("XRPUSDT", category="watch", stop=0.9, last=1.0)])

    def test_the_same_coin_is_not_opened_twice(self):
        trend.open_demo("FILUSDT", price=0.936, now_rows=[_row()])
        with self.assertRaises(ValueError):
            trend.open_demo("FILUSDT", price=0.936, now_rows=[_row()])

    def test_price_already_below_the_stop_is_refused(self):
        with self.assertRaises(ValueError):
            trend.open_demo("FILUSDT", price=0.70, now_rows=[_row()])

    def test_risk_is_clamped_and_never_leveraged(self):
        pos = trend.open_demo("FILUSDT", risk_pct=50, price=0.936, now_rows=[_row(stop=0.935)])
        self.assertLessEqual(pos["size_usdt"], 10_000.0)                    # بی‌اهرم

    def test_trailing_stop_only_rises_and_the_position_closes_when_the_rule_exits(self):
        pos = trend.open_demo("FILUSDT", price=0.936, now_rows=[_row()])
        res = trend.sync_demo([_row(stop=0.80)], price_fn=lambda s: 0.95)
        self.assertEqual(res["moved"], 1)
        trend.sync_demo([_row(stop=0.78)], price_fn=lambda s: 0.95)            # پایین‌تر ⇒ نه
        live = [p for p in paper.list_positions()["open"] if p["id"] == pos["id"]][0]
        self.assertEqual(live["sl"], 0.80)
        res = trend.sync_demo([_row(category="watch", in_position=False)], price_fn=lambda s: 0.85)
        self.assertEqual(res["closed"], 1)
        self.assertFalse(any(p["id"] == pos["id"] for p in paper.list_positions()["open"]))

    def test_a_position_without_target_closes_on_its_stop_and_pays_no_funding(self):
        pos = trend.open_demo("FILUSDT", price=0.936, now_rows=[_row()])
        paper.refresh({"FILUSDT": 1.20}, funding_fn=lambda s: [[4_102_444_800_000, 0.001]])
        live = [p for p in paper.list_positions()["open"] if p["id"] == pos["id"]][0]
        self.assertEqual(live.get("funding_pct", 0.0), 0.0)                  # اسپات
        paper.refresh({"FILUSDT": 0.70})
        self.assertFalse(any(p["id"] == pos["id"] for p in paper.list_positions()["open"]))

    def test_advisor_says_hold_the_trend_instead_of_crashing(self):
        pos = trend.open_demo("FILUSDT", price=0.936, now_rows=[_row()])
        adv = advisor.advise(dict(pos, last_price=1.0), {"p_up": 40})
        self.assertEqual(adv["action"], "hold_trend")


if __name__ == "__main__":
    unittest.main()
