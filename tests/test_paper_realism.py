# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ شبیه‌سازِ واقع‌گرا.

سه خوش‌بینیِ یک‌طرفهٔ شبیه‌سازِ قبلی: ویکِ بینِ دو نمونهٔ قیمت دیده نمی‌شد،
خروجِ استاپ بدونِ لغزش فرض می‌شد، و فاندینگِ پرپچوال اصلاً حساب نمی‌شد.
"""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import paper  # noqa: E402


class _PaperMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = paper.PATH
        paper.PATH = os.path.join(self.tmp.name, "positions.json")

    def tearDown(self):
        paper.PATH = self._old
        self.tmp.cleanup()


class WickAwareExitTests(_PaperMixin, unittest.TestCase):
    def test_wick_between_samples_stops_the_position_out(self):
        pos = paper.open_position("BTCUSDT", "1h", "long", 100, 98, 103.6, 1_000, 2_400,
                                  cost_pct=0.11)
        opened = datetime.fromisoformat(pos["opened_at"]).timestamp() * 1000
        kl = {"t": [opened + 3_600_000],
              "o": [100.5], "h": [101.0], "l": [97.5], "c": [100.2], "v": [1.0]}
        # قیمتِ نمونه‌برداری‌شده ۱۰۰٫۲ است و هیچ حدی را نزده — ولی ویک ۹۷٫۵ حدضرر را زده
        db = paper.refresh({"BTCUSDT": 100.2}, klines_fn=lambda *_a: kl)
        self.assertEqual(len(db["open"]), 0)
        closed = db["closed"][0]
        self.assertEqual(closed["close_reason"], "حدضرر")
        self.assertLess(closed["exit_price"], 98.0)      # لغزش، بدتر از سطحِ برنامه

    def test_gap_open_below_the_stop_exits_at_the_observed_open(self):
        pos = paper.open_position("BTCUSDT", "1h", "long", 100, 98, 103.6, 1_000, 2_400,
                                  cost_pct=0.30)
        opened = datetime.fromisoformat(pos["opened_at"]).timestamp() * 1000
        kl = {"t": [opened + 3_600_000],
              "o": [94.0], "h": [95.0], "l": [93.0], "c": [94.5], "v": [1.0]}
        db = paper.refresh({"BTCUSDT": 94.5}, klines_fn=lambda *_a: kl)
        closed = db["closed"][0]
        self.assertEqual(closed["close_reason"], "حدضرر (گپ)")
        self.assertAlmostEqual(closed["exit_price"], 94.0 * (1 - 20 / 10_000))

    def test_stop_wins_when_both_barriers_are_touched_in_one_bar(self):
        pos = paper.open_position("BTCUSDT", "1h", "long", 100, 98, 103.6, 1_000, 2_400,
                                  cost_pct=0.08)
        opened = datetime.fromisoformat(pos["opened_at"]).timestamp() * 1000
        kl = {"t": [opened + 3_600_000],
              "o": [100.0], "h": [104.0], "l": [97.0], "c": [103.0], "v": [1.0]}
        db = paper.refresh({"BTCUSDT": 103.0}, klines_fn=lambda *_a: kl)
        self.assertEqual(db["closed"][0]["close_reason"], "حدضرر")

    def test_target_fills_exactly_at_the_planned_level(self):
        pos = paper.open_position("BTCUSDT", "1h", "long", 100, 98, 103.6, 1_000, 2_400,
                                  cost_pct=0.08)
        opened = datetime.fromisoformat(pos["opened_at"]).timestamp() * 1000
        kl = {"t": [opened + 3_600_000],
              "o": [100.0], "h": [104.0], "l": [99.5], "c": [103.9], "v": [1.0]}
        db = paper.refresh({"BTCUSDT": 103.9}, klines_fn=lambda *_a: kl)
        closed = db["closed"][0]
        self.assertEqual(closed["close_reason"], "هدف ✅")
        self.assertAlmostEqual(closed["exit_price"], 103.6)

    def test_slippage_scales_with_the_liquidity_tier(self):
        self.assertLess(paper.slippage_bps(0.08), paper.slippage_bps(0.30))
        self.assertAlmostEqual(paper.slippage_bps(0.08), 3.0)
        self.assertAlmostEqual(paper.slippage_bps(0.30), 20.0)
        self.assertAlmostEqual(paper.slippage_bps(None), paper.DEFAULT_SLIPPAGE_BPS)


class FundingAccrualTests(_PaperMixin, unittest.TestCase):
    """فاندینگ فقط برای تسویه‌هایی که **واقعاً گذشته‌اند** کسر می‌شود، پس پوزیشن عقب‌تاریخ می‌شود."""

    @staticmethod
    def _backdate(pos_id, hours):
        db = paper.list_positions()
        for p in db["open"]:
            if p["id"] == pos_id:
                p["opened_at"] = (datetime.now(timezone.utc)
                                  - timedelta(hours=hours)).isoformat()
        paper._save(db)
        return datetime.fromisoformat(
            next(p["opened_at"] for p in db["open"] if p["id"] == pos_id)).timestamp() * 1000

    def test_long_pays_positive_funding_at_every_eight_hour_settlement(self):
        pos = paper.open_position("BTCUSDT", "4h", "long", 100, 98, 103.6, 1_000, 9_600,
                                  cost_pct=0.11)
        opened = self._backdate(pos["id"], 30)
        rows = [[opened + k * 8 * 3600_000, 0.0001] for k in range(1, 4)]  # ۳ تسویهٔ +۰٫۰۱٪
        db = paper.refresh({"BTCUSDT": 100.0}, funding_fn=lambda _s: rows)
        live = db["open"][0]
        self.assertAlmostEqual(live["funding_pct"], 0.03, places=6)
        # سود ناخالص صفر است؛ کلِ زیان = کارمزد + فاندینگ
        self.assertAlmostEqual(live["pnl_pct"], -(0.11 + 0.03), places=6)

    def test_future_settlements_are_not_charged_early(self):
        pos = paper.open_position("BTCUSDT", "4h", "long", 100, 98, 103.6, 1_000, 9_600,
                                  cost_pct=0.11)
        opened = datetime.fromisoformat(pos["opened_at"]).timestamp() * 1000
        rows = [[opened + 8 * 3600_000, 0.0005]]        # هنوز نرسیده
        db = paper.refresh({"BTCUSDT": 100.0}, funding_fn=lambda _s: rows)
        self.assertAlmostEqual(db["open"][0].get("funding_pct", 0.0), 0.0)

    def test_short_receives_positive_funding(self):
        pos = paper.open_position("BTCUSDT", "4h", "short", 100, 102, 96.4, 1_000, 9_600,
                                  cost_pct=0.11)
        opened = self._backdate(pos["id"], 12)
        rows = [[opened + 8 * 3600_000, 0.0001]]
        db = paper.refresh({"BTCUSDT": 100.0}, funding_fn=lambda _s: rows)
        self.assertAlmostEqual(db["open"][0]["funding_pct"], -0.01, places=6)

    def test_funding_is_not_double_counted_across_refreshes(self):
        pos = paper.open_position("BTCUSDT", "4h", "long", 100, 98, 103.6, 1_000, 9_600,
                                  cost_pct=0.11)
        opened = self._backdate(pos["id"], 12)
        rows = [[opened + 8 * 3600_000, 0.0002]]
        for _ in range(4):
            db = paper.refresh({"BTCUSDT": 100.0}, funding_fn=lambda _s: rows)
        self.assertAlmostEqual(db["open"][0]["funding_pct"], 0.02, places=6)

    def test_long_hold_funding_is_material_against_the_reward(self):
        """۴۰ کندلِ روزانه ≈ ۴۰ روز ⇒ ~۱۱۷ تسویه؛ با ۰٫۰۱٪ هر تسویه یعنی ~۱٫۲٪."""
        pos = paper.open_position("BTCUSDT", "1d", "long", 100, 99.3, 101.26, 1_000,
                                  57_600, cost_pct=0.11)
        opened = self._backdate(pos["id"], 39 * 24)
        rows = [[opened + k * 8 * 3600_000, 0.0001] for k in range(1, 118)]
        db = paper.refresh({"BTCUSDT": 100.0}, funding_fn=lambda _s: rows)
        funding = db["open"][0]["funding_pct"]
        self.assertAlmostEqual(funding, 1.17, places=6)
        reward_pct = (101.26 - 100) / 100 * 100          # کلِ پاداشِ 1.8R
        self.assertGreater(funding, reward_pct * 0.9)    # فاندینگ تقریباً کلِ پاداش را می‌خورد


class BackwardCompatibilityTests(_PaperMixin, unittest.TestCase):
    def test_price_only_refresh_still_works(self):
        paper.open_position("BTCUSDT", "1h", "long", 100, 98, 103.6, 1_000, 2_400,
                            cost_pct=0.11)
        db = paper.refresh({"BTCUSDT": 101.0})
        self.assertEqual(len(db["open"]), 1)
        self.assertAlmostEqual(db["open"][0]["pnl_pct"], 1.0 - 0.11, places=6)


if __name__ == "__main__":
    unittest.main()
