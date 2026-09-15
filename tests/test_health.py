# -*- coding: utf-8 -*-
"""تستِ نقطهٔ پایشِ سلامت — چون خطاها در سراسرِ کد بلعیده می‌شوند، این تنها
راهِ فهمیدنِ «دادهٔ من مرده است» یا «مدلِ من با کد نمی‌خواند» است."""
import os
import sys
import unittest
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import main  # noqa: E402


class HealthTests(unittest.TestCase):
    OK_CALIB = {"version": 18, "required_version": 18, "age_hours": 3.0,
                "stale": False, "building": False, "error": None}

    def _health(self, calib_status=None, counts=None):
        with mock.patch.object(main.calib, "status",
                               return_value=dict(calib_status or self.OK_CALIB)), \
                mock.patch.object(main.candidates, "counts",
                                  return_value=dict(counts or {
                                      "candidates": 10, "resolved": 4, "open": 6,
                                      "last_logged_at": main.time.time(),
                                      "last_resolved_at": main.time.time()})):
            return main.health()

    def test_green_when_model_matches_and_candidates_are_fresh(self):
        h = self._health()
        self.assertEqual(h["status"], "green", h["problems"] + h["warnings"])
        self.assertEqual(h["problems"], [])

    def test_model_version_mismatch_is_red(self):
        h = self._health(calib_status=dict(self.OK_CALIB, version=17))
        self.assertEqual(h["status"], "red")
        self.assertTrue(any("نسخهٔ مدل" in p for p in h["problems"]))

    def test_stalled_candidate_logging_is_red(self):
        stale = main.time.time() - 10 * 86400
        h = self._health(counts={"candidates": 3, "resolved": 3, "open": 0,
                                 "last_logged_at": stale, "last_resolved_at": stale})
        self.assertEqual(h["status"], "red")
        self.assertTrue(any("کاندید" in p for p in h["problems"]))

    def test_build_error_is_surfaced_instead_of_being_swallowed(self):
        h = self._health(calib_status=dict(self.OK_CALIB, error="boom"))
        self.assertEqual(h["status"], "red")
        self.assertTrue(any("boom" in p for p in h["problems"]))

    def test_empty_ledger_is_a_warning_not_a_failure(self):
        h = self._health(counts={"candidates": 0, "resolved": 0, "open": 0,
                                 "last_logged_at": None, "last_resolved_at": None})
        self.assertEqual(h["status"], "amber")
        self.assertEqual(h["problems"], [])

    def _cache_health(self, klines_by_tf, now):
        stats = {"klines_by_tf": klines_by_tf, "universe_age_sec": 5.0, "universe_size": 200}
        with mock.patch.object(main.market, "cache_stats", return_value=stats), \
                mock.patch.object(main.time, "time", return_value=now):
            return self._health()

    def test_a_daily_candle_fetched_hours_ago_is_not_stale(self):
        """کندلِ روزانهٔ دیروز که ۶ ساعت پیش گرفته شده تا نیمه‌شب کاملاً به‌روز است."""
        now = 1_789_516_800 + 18 * 3600                       # ساعتِ ۱۸ UTC
        h = self._cache_health({"1d": {"symbols": 20, "newest_age_sec": 6 * 3600,
                                       "oldest_age_sec": 6 * 3600, "behind_candles_min": 0}}, now)
        self.assertFalse([w for w in h["warnings"] if "کندل" in w], h["warnings"])

    def test_data_left_behind_after_the_grace_period_is_flagged(self):
        now = 1_789_516_800 + 3600 + 20 * 60                  # ۲۰ دقیقه پس از بسته‌شدنِ کندلِ ساعتی
        h = self._cache_health({"1h": {"symbols": 30, "newest_age_sec": 70 * 60,
                                       "oldest_age_sec": 80 * 60, "behind_candles_min": 1}}, now)
        self.assertTrue(any("1h" in w and "عقب" in w for w in h["warnings"]), h["warnings"])

    def test_an_unused_timeframe_is_not_a_health_problem(self):
        now = 1_789_516_800 + 3 * 3600 + 20 * 60
        h = self._cache_health({"15m": {"symbols": 5, "newest_age_sec": 3 * 3600,
                                        "oldest_age_sec": 3 * 3600, "behind_candles_min": 12}}, now)
        self.assertFalse([w for w in h["warnings"] if "کندل" in w], h["warnings"])

    def test_cache_stats_counts_candles_behind_the_last_close(self):
        tf_ms = 3_600_000
        now = (1_789_516_800_000 + 5 * tf_ms + 10 * 60_000) / 1000      # ۱۰ دقیقه پس از بسته‌شدنِ کندلِ ۵
        fresh = {"t": [1_789_516_800_000 + 4 * tf_ms]}                  # آخرین کندلِ بسته‌شده ⇒ ۰
        old = {"t": [1_789_516_800_000 + 2 * tf_ms]}                    # دو کندل عقب
        with mock.patch.dict(main.market._kline_cache, {("A", "1h"): (now - 60, fresh),
                                                         ("B", "1h"): (now - 9000, old)}, clear=True):
            row = main.market.cache_stats(now=now)["klines_by_tf"]["1h"]
        self.assertEqual(row["behind_candles_min"], 0)

    def test_report_always_carries_the_capital_lock_state(self):
        h = self._health()
        self.assertIn("gates", h)
        self.assertIn("live_effective", h["gates"])
        self.assertFalse(h["gates"]["live_effective"])


if __name__ == "__main__":
    unittest.main()
