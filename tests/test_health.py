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

    def test_report_always_carries_the_capital_lock_state(self):
        h = self._health()
        self.assertIn("gates", h)
        self.assertIn("live_effective", h["gates"])
        self.assertFalse(h["gates"]["live_effective"])


if __name__ == "__main__":
    unittest.main()
