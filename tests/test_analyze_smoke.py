# -*- coding: utf-8 -*-
"""دودِ سفید: مسیرِ کاملِ ``engine.analyze`` روی دادهٔ مصنوعی، بدونِ شبکه.

هدف این نیست که سیگنال درست باشد؛ هدف این است که مسیرِ تصمیم از ابتدا تا انتها
بدونِ استثنا اجرا شود و پیش‌فرضِ خروجی «قابلِ معامله نیست» باقی بماند.
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import bracket  # noqa: E402
import engine  # noqa: E402
import gates  # noqa: E402

from test_bracket_contract import synthetic_klines  # noqa: E402


class AnalyzeSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = gates.GATES_PATH
        gates.GATES_PATH = os.path.join(self.tmp.name, "gates.json")
        with open(gates.GATES_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(gates.DEFAULTS), f)
        gates.load_gates(force=True)

    def tearDown(self):
        gates.GATES_PATH = self._old
        gates._cache.update(mtime=None, data=None)
        self.tmp.cleanup()

    def test_analyze_runs_end_to_end_and_defaults_to_not_tradeable(self):
        for seed in (1, 5, 11, 23):
            with self.subTest(seed=seed):
                res = engine.analyze(synthetic_klines(seed=seed), "1h")
                self.assertIn("trade", res)
                self.assertIn("forecast", res)
                self.assertFalse(res["trade"].get("tradeable", False))
                self.assertIsNone(res["trade"].get("recommendation"))

    def test_any_proposed_bracket_obeys_the_canonical_contract(self):
        seen = 0
        for seed in range(1, 40):
            tr = engine.analyze(synthetic_klines(seed=seed), "1h")["trade"]
            if not tr.get("side"):
                continue
            seen += 1
            r_dist = abs(tr["entry"] - tr["sl"])
            self.assertGreater(r_dist, 0)
            self.assertAlmostEqual(tr["rr"], 1.8, places=9)
            # خروجی برای نمایش به ۸ رقمِ معنادار گرد می‌شود، پس تلورانس کوچکی لازم است
            self.assertAlmostEqual(abs(tr["tp"] - tr["entry"]) / r_dist, 1.8, places=4)
            self.assertEqual(tr["time_stop_bars"], bracket.MAX_BARS)
        self.assertGreater(seen, 0, "هیچ پیشنهادی ساخته نشد — تست بی‌اثر است")

    def test_probability_is_not_marked_calibrated_without_a_model(self):
        res = engine.analyze(synthetic_klines(seed=7), "4h")
        self.assertFalse(res["p_calibrated"])
        self.assertIsNotNone(res["p_up"])


if __name__ == "__main__":
    unittest.main()
