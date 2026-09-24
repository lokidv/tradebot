# -*- coding: utf-8 -*-
"""میزِ معاملهٔ کوکوین: حجم دقیقاً ریسکِ تعیین‌شده را می‌سازد، لیکوئید پیش از حدضرر هشدار می‌دهد، و دفتر درست حساب می‌کند."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import kucoin_desk as kd  # noqa: E402

SOL = {"multiplier": 0.1, "lot": 1.0, "mmr": 0.007, "maker": 0.0002, "taker": 0.0006}


class SizingTests(unittest.TestCase):
    def test_loss_at_stop_including_fees_matches_the_risk(self):
        r = kd.size_position(1000, 1.0, 115.0, 113.5, SOL)
        self.assertLessEqual(r["loss_at_stop"], 10.0 + 1e-9)            # گردکردنِ قرارداد فقط رو به پایین
        self.assertGreater(r["loss_at_stop"], 9.7)
        self.assertEqual(r["side"], "long")
        self.assertFalse(r["liq_before_stop"])

    def test_short_side_and_high_leverage_liquidation_warning(self):
        r = kd.size_position(1000, 1.0, 115.0, 118.0, SOL, leverage=50)
        self.assertEqual(r["side"], "short")
        self.assertTrue(r["liq_before_stop"])                            # با اهرمِ ۵۰ پیش از حدضرر لیکوئید

    def test_stop_equal_to_entry_is_rejected(self):
        with self.assertRaises(ValueError):
            kd.size_position(1000, 1.0, 115.0, 115.0, SOL)


class JournalTests(unittest.TestCase):
    def setUp(self):
        if os.path.exists(kd.JOURNAL):
            os.remove(kd.JOURNAL)

    def test_pnl_r_and_stats(self):
        kd.add_trade({"sym": "SOLUSDT", "side": "long", "entry": 100, "exit": 102, "notional": 1000, "fees": 1.2,
                      "stop": 99})
        kd.add_trade({"sym": "SOLUSDT", "side": "short", "entry": 100, "exit": 101, "notional": 1000, "fees": 1.2})
        s = kd.journal_stats()
        self.assertEqual(s["n"], 2)
        win = [t for t in s["trades"] if t["side"] == "long"][0]
        self.assertAlmostEqual(win["pnl"], 20 - 1.2)
        self.assertAlmostEqual(win["r"], (20 - 1.2) / 10, places=3)
        self.assertAlmostEqual(s["total_pnl"], 18.8 - 11.2)
        self.assertFalse(s["enough_data"])

    def test_delete_is_append_only(self):
        row = kd.add_trade({"sym": "BTCUSDT", "side": "long", "entry": 1, "exit": 1.1, "notional": 10})
        kd.delete_trade(row["id"])
        self.assertEqual(kd.journal_stats()["n"], 0)
        with open(kd.JOURNAL, encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), 2)


if __name__ == "__main__":
    unittest.main()
