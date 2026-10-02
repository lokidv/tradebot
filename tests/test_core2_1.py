# -*- coding: utf-8 -*-
"""هستهٔ v2.1: ثابت‌ها همان پیش‌ثبت، مدلِ دورِ ثابت، قاعدهٔ تصمیم و نگهبانِ holdout."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401

import core2  # noqa: E402
import core2_1 as c21  # noqa: E402

PREREG = os.path.join(ROOT, "bot", "data", "research", "prereg_core2_1.json")


class PreregTests(unittest.TestCase):
    def test_constants_match_the_preregistration(self):
        pr = c21.check_prereg(PREREG)
        self.assertEqual(c21.window_ms("validation"), (core2._ms("2025-07-01"), core2._ms("2026-01-01")))
        self.assertEqual(c21.window_ms("holdout")[0], core2._ms("2026-01-01"))
        self.assertIn("E_L > 0", pr["decision_rule_fixed"])
        self.assertEqual(core2.GAP_BARS, 81)

    def test_params_are_the_v21_spec_and_never_early_stop(self):
        p = c21.lgb_params("1h", 10)
        self.assertEqual((p["objective"], p["learning_rate"], p["num_leaves"], p["max_depth"], p["feature_fraction"],
                          p["lambda_l2"], p["min_data_in_leaf"]), ("regression", 0.02, 15, 5, 0.5, 50.0, 200))
        self.assertEqual(c21.lgb_params("5m", 1_000_000)["min_data_in_leaf"], 2500)
        self.assertEqual(c21.lgb_params("15m", 1)["min_data_in_leaf"], 500)
        self.assertEqual(c21.lgb_params("4h", 1)["min_data_in_leaf"], 100)
        d = c21.lgb_params("1d", 1)
        self.assertEqual((d["num_leaves"], d["min_data_in_leaf"]), (7, 60))
        self.assertEqual(c21.ROUNDS, {"5m": 300, "15m": 300, "1h": 300, "4h": 300, "1d": 150})

    def test_n_min(self):
        self.assertEqual((c21.n_min("1h"), c21.n_min("1d")), (60, 20))
        self.assertEqual((c21.HOLD_N_MIN.get("1d"), c21.HOLD_N_MIN_DEFAULT), (10, 30))


class DecideTests(unittest.TestCase):
    def test_primary_rule_is_positive_and_better_side(self):
        EL = np.array([0.01, 0.2, -0.1, 0.0, 0.3, np.nan, 0.05])
        ES = np.array([-0.2, 0.3, -0.2, 0.0, 0.3, 0.1, 0.04])
        self.assertEqual(c21.decide(EL, ES).tolist(), [1, -1, 0, 0, 0, 0, 1])

    def test_margin_variants_are_stricter(self):
        EL, ES = np.array([0.03, 0.07, 0.2]), np.array([-0.1, -0.1, 0.0])
        self.assertEqual(c21.decide(EL, ES, 0.05).tolist(), [0, 1, 1])
        self.assertEqual(c21.decide(EL, ES, 0.10).tolist(), [0, 0, 1])


class FitTests(unittest.TestCase):
    def test_fit_uses_exactly_the_fixed_rounds_and_predicts_the_mean(self):
        rng = np.random.RandomState(0)
        n = 3000
        X = rng.normal(size=(n, 4)).astype(np.float32)
        yL = (0.5 * X[:, 0] - 0.1 + rng.normal(0, 0.3, n)).astype(np.float32)
        yS = (-0.5 * X[:, 0] - 0.1 + rng.normal(0, 0.3, n)).astype(np.float32)
        sym = (np.arange(n) % 5).astype(np.int8)
        old = dict(c21.ROUNDS)
        c21.ROUNDS["1d"] = 20
        try:
            predict, meta = c21.fit_pair(X, yL, yS, np.arange(n), sym, "1d", list("abcd"), 2)
        finally:
            c21.ROUNDS.clear()
            c21.ROUNDS.update(old)
        EL, ES = predict(X)
        self.assertEqual(meta["rounds"], 20)
        self.assertGreater(float(np.corrcoef(EL, X[:, 0])[0, 1]), 0.5)
        self.assertLess(float(np.corrcoef(ES, X[:, 0])[0, 1]), -0.5)
        self.assertGreater(float(EL.max()), 0.0)           # بر خلافِ v2، می‌تواند مثبت پیش‌بینی کند


class GuardTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(c21, "PREREG_PATH", PREREG)     # دادهٔ تست‌ها در پوشهٔ موقت است
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.mkdtemp()
        self.rep_path = os.path.join(self.tmp, "core2_1_validation_x.json")
        self.rep = {"window": "validation", "complete": True, "timeframes": {"1h": {"adopted": True}, "4h": {"adopted": False}}}
        with open(self.rep_path, "w", encoding="utf-8") as f:
            json.dump(self.rep, f)
        self.pin = os.path.join(self.tmp, "pin.json")
        self.log = os.path.join(self.tmp, "log.jsonl")
        self.head = self._read_head

    @staticmethod
    def _read_head(p):
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return f.read()

    def _guard(self, tf="1h", head=None):
        return c21.holdout_guard(tf, self.rep, self.rep_path, out_dir=self.tmp, pin_path=self.pin,
                                 log_path=self.log, head_reader=head or self.head)

    def test_refuses_without_pin_record_and_commit_then_allows(self):
        ok, why = self._guard()
        self.assertFalse(ok)
        self.assertIn("سنجاق", why)
        c21.make_pin(self.rep_path, self.pin)
        ok, why = self._guard()
        self.assertFalse(ok)
        self.assertIn("رکورد", why)
        c21.write_start_record("1h", self.rep_path, self.log)
        ok, why = self._guard(head=lambda p: "")
        self.assertFalse(ok)
        self.assertIn("commit", why)
        self.assertEqual(self._guard(), (True, ""))

    def test_refuses_unadopted_changed_report_and_second_run(self):
        c21.make_pin(self.rep_path, self.pin)
        c21.write_start_record("1h", self.rep_path, self.log)
        self.assertFalse(self._guard("4h")[0])
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "holdout_done", "tf": "1h"}) + "\n")
        self.assertIn("یک‌بار", self._guard()[1])
        os.remove(self.log)
        c21.write_start_record("1h", self.rep_path, self.log)
        with open(self.rep_path, "a", encoding="utf-8") as f:
            f.write(" ")
        self.assertIn("هش", self._guard()[1])

    def test_run_window_refuses_holdout_without_guard(self):
        with self.assertRaises(PermissionError):
            c21.run_window("1h", "holdout")
        with self.assertRaises(PermissionError):
            c21.run_window("1h", "holdout", guard=(False, "x"))


class AdoptTests(unittest.TestCase):
    def _res(self, mean, n, diff_lo, coins, share, rule_mean=0.0):
        return {"v21": {"mean": mean, "n": n, "coins_positive": coins, "max_coin_share": share},
                "rule": {"mean": rule_mean}, "bootstrap": {"diff_ci95": [diff_lo, 1.0]}}

    def test_all_four_conditions_needed(self):
        self.assertTrue(c21.adopt_checks("1h", self._res(0.1, 60, 0.01, 3, 0.4), True)["adopted"])
        self.assertFalse(c21.adopt_checks("1h", self._res(0.1, 59, 0.01, 3, 0.4), True)["adopted"])
        self.assertFalse(c21.adopt_checks("1h", self._res(0.1, 60, 0.01, 3, 0.4), False)["adopted"])
        self.assertFalse(c21.adopt_checks("1h", self._res(0.1, 60, -0.01, 3, 0.4), True)["adopted"])
        self.assertFalse(c21.adopt_checks("1h", self._res(0.1, 60, 0.01, 2, 0.4), True)["adopted"])
        self.assertFalse(c21.adopt_checks("1h", self._res(0.1, 60, 0.01, 3, 0.6), True)["adopted"])
        self.assertTrue(c21.adopt_checks("1d", self._res(0.1, 20, 0.01, 3, 0.4), True)["adopted"])


if __name__ == "__main__":
    unittest.main()
