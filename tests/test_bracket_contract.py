# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ فاز ۱ — قراردادِ واحدِ براکت.

برچسبِ آموزش، بک‌تستِ سریع، پیشنهادِ زنده و داوریِ سایه باید برای یک کندل
دقیقاً یک معامله را توصیف کنند؛ وگرنه ``p_win``/``EV`` مربوط به معامله‌ای است
که هرگز اجرا نمی‌شود.
"""
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import bracket  # noqa: E402
import calib  # noqa: E402
import engine  # noqa: E402
import shadow  # noqa: E402


def synthetic_klines(n=420, seed=5):
    rng = np.random.RandomState(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    wig = np.abs(rng.normal(0, 0.003, n)) * c
    return {
        "t": (np.arange(n, dtype=np.int64) * 3_600_000).tolist(),
        "o": np.concatenate([[c[0]], c[:-1]]).tolist(),
        "h": (np.maximum(c, np.concatenate([[c[0]], c[:-1]])) + wig).tolist(),
        "l": (np.minimum(c, np.concatenate([[c[0]], c[:-1]])) - wig).tolist(),
        "c": c.tolist(),
        "v": np.full(n, 1_000.0).tolist(),
    }


class BracketRulesTests(unittest.TestCase):
    def test_levels_match_the_training_label_geometry(self):
        lv = bracket.levels(entry=100.0, atr14=2.0, sig=1)
        self.assertAlmostEqual(lv["r"], 2.6)                 # 1.3 × ATR غالب است
        self.assertAlmostEqual(lv["sl"], 97.4)
        self.assertAlmostEqual(lv["tp"], 100.0 + 1.8 * 2.6)
        self.assertAlmostEqual(lv["rr"], 1.8)

    def test_minimum_risk_floor_binds_on_quiet_symbols(self):
        lv = bracket.levels(entry=100.0, atr14=0.01, sig=-1)
        self.assertAlmostEqual(lv["r"], 0.25)                # کفِ ۰٫۲۵٪
        self.assertAlmostEqual(lv["sl"], 100.25)
        self.assertAlmostEqual(lv["risk_pct"], 0.25)

    def test_stop_and_target_in_one_bar_counts_as_a_loss(self):
        o = [100.0, 100.0]
        h = [100.0, 103.0]        # هدف لمس شد
        l = [100.0, 97.0]         # حدضرر هم لمس شد
        c = [100.0, 102.0]
        res = bracket.resolve_path(o, h, l, c, 1, 1, 100.0, 98.0, 103.6)
        self.assertEqual(res["outcome"], bracket.OUTCOME_STOP)
        self.assertAlmostEqual(res["gross_r"], -1.0)

    def test_gap_through_the_stop_is_worse_than_minus_one_r(self):
        o = [100.0, 95.0]         # با گپ زیرِ حدضرر باز شد
        h = [100.0, 96.0]
        l = [100.0, 94.0]
        c = [100.0, 95.5]
        res = bracket.resolve_path(o, h, l, c, 1, 1, 100.0, 98.0, 103.6)
        self.assertEqual(res["outcome"], bracket.OUTCOME_GAP_STOP)
        self.assertAlmostEqual(res["gross_r"], -2.5)
        self.assertLess(res["gross_r"], -1.0)

    def test_timeout_marks_to_the_last_allowed_close(self):
        o = [100.0] * 5
        h = [100.5] * 5
        l = [99.5] * 5
        c = [100.0, 100.2, 100.4, 100.6, 100.8]
        # از کندلِ ۱ شروع، سقفِ ۳ کندل ⇒ آخرین کندلِ مجاز ۳ است: (100.6 − 100) / 2R = +0.3R
        res = bracket.resolve_path(o, h, l, c, 1, 1, 100.0, 98.0, 103.6, max_bars=3)
        self.assertEqual(res["outcome"], bracket.OUTCOME_TIMEOUT)
        self.assertTrue(res["timed_out"])
        self.assertEqual(res["bars_held"], 3)
        self.assertEqual(res["exit_idx"], 3)
        self.assertAlmostEqual(res["gross_r"], 0.3)

    def test_zero_risk_trade_is_refused_rather_than_producing_absurd_r(self):
        with self.assertRaises(ValueError):
            bracket.resolve_path([1e-6], [1e-6], [1e-6], [1e-6], 0, 1, 4e-6, 4e-6, 5e-6)

    def test_breakeven_win_rate_is_the_documented_threshold(self):
        self.assertAlmostEqual(bracket.breakeven_win_rate(), 0.3571, places=4)


class AllCallersAgreeTests(unittest.TestCase):
    """برچسبِ آموزش و بک‌تستِ سریع باید عددِ یکسان بدهند."""

    def test_training_labels_and_quick_backtest_agree_on_the_same_signal(self):
        kl = synthetic_klines()
        o = np.array(kl["o"], float)
        h = np.array(kl["h"], float)
        l = np.array(kl["l"], float)
        c = np.array(kl["c"], float)
        cs = engine.component_series(o, h, l, c, np.array(kl["v"], float))

        events, _z, _dx, _dy, _dr = calib.extract_events(
            "TESTUSDT", kl, "1h", [], {}, None, None)
        self.assertGreater(len(events), 0, "دادهٔ مصنوعی هیچ ستاپی نساخت")

        ts_to_idx = {t: i for i, t in enumerate(kl["t"])}
        for ev in events:
            i = ts_to_idx[ev["ts"]]
            sig, _setup = engine.setup_signal(cs, o, h, l, c, i)
            ref = bracket.signal_trade(o, h, l, c, cs["a14"][i], i, sig)
            self.assertAlmostEqual(ev["r"], round(ref["gross_r"], 3), places=3)
            self.assertAlmostEqual(ev["risk_pct"], round(ref["risk_pct"], 4), places=4)

    def test_quick_backtest_uses_the_same_bracket(self):
        kl = synthetic_klines()
        o = np.array(kl["o"], float)
        h = np.array(kl["h"], float)
        l = np.array(kl["l"], float)
        c = np.array(kl["c"], float)
        cs = engine.component_series(o, h, l, c, np.array(kl["v"], float))
        fire_at = 300

        def one_signal(_cs, _o, _h, _l, _c, i):
            return (1, "zx") if i == fire_at else (0, None)

        with mock.patch.object(engine, "setup_signal", side_effect=one_signal):
            bt = engine.quick_backtest(o, h, l, c, cs, "1h", cost_pct=0.15)
        ref = bracket.signal_trade(o, h, l, c, cs["a14"][fire_at], fire_at, 1)
        expected = bracket.net_r(ref["gross_r"], ref["risk_pct"], 0.15)
        self.assertEqual(bt["n"], 1)
        self.assertAlmostEqual(bt["avg_r"], expected, places=9)


class LiveSuggestionUsesTheLabelBracketTests(unittest.TestCase):
    def test_suggestion_always_has_rr_one_point_eight_and_the_label_risk(self):
        kl = synthetic_klines(seed=11)
        o = np.array(kl["o"], float)
        h = np.array(kl["h"], float)
        l = np.array(kl["l"], float)
        c = np.array(kl["c"], float)
        cs = engine.component_series(o, h, l, c, np.array(kl["v"], float))
        fc = engine.forecast(c, cs, "1h", 3, 0)
        for side in ("long", "short"):
            tr = engine.trade_suggestion(c, cs, fc, "1h", 3, 0, 0.0, force_side=side)
            r_dist = abs(tr["entry"] - tr["sl"])
            self.assertAlmostEqual(tr["rr"], 1.8, places=9)
            self.assertAlmostEqual(abs(tr["tp"] - tr["entry"]) / r_dist, 1.8, places=9)
            self.assertAlmostEqual(
                r_dist, max(1.3 * float(cs["a14"][-1]), 0.0025 * float(c[-1])), places=9)
            self.assertEqual(tr["time_stop_bars"], bracket.MAX_BARS)

    def test_forecast_target_no_longer_moves_the_take_profit(self):
        kl = synthetic_klines(seed=11)
        c = np.array(kl["c"], float)
        cs = engine.component_series(
            np.array(kl["o"], float), np.array(kl["h"], float),
            np.array(kl["l"], float), c, np.array(kl["v"], float))
        base = engine.forecast(c, cs, "1h", 3, 0)
        moon = dict(base, target=base["target"] * 3.0)
        tp_base = engine.trade_suggestion(c, cs, base, "1h", 3, 0, 0.0, force_side="long")["tp"]
        tp_moon = engine.trade_suggestion(c, cs, moon, "1h", 3, 0, 0.0, force_side="long")["tp"]
        self.assertAlmostEqual(tp_base, tp_moon, places=9)


class PoisonedShadowRowsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = shadow.SHADOW_PATH
        shadow.SHADOW_PATH = os.path.join(self.tmp.name, "signals.json")

    def tearDown(self):
        shadow.SHADOW_PATH = self._old
        self.tmp.cleanup()

    def test_zero_risk_signal_is_never_logged(self):
        ok = shadow.log_signal("SHIBUSDT", "1d", "long", 4e-06, 4e-06, 5e-06, "dm",
                               38.7, 1.0, int(time.time() * 1000), 57_600)
        self.assertFalse(ok)
        self.assertEqual(shadow._load()["pending"], [])

    def test_stale_candle_signal_is_never_logged(self):
        old_ts = int((time.time() - 40 * 86400) * 1000)
        ok = shadow.log_signal("COCOSUSDT", "1d", "long", 1.75, 1.56, 1.9, "dm",
                               38.7, 0.7, old_ts, 57_600)
        self.assertFalse(ok)

    def test_fresh_signal_is_logged(self):
        ok = shadow.log_signal("BTCUSDT", "1h", "long", 100.0, 99.0, 101.8, "zx",
                               50.0, 0.3, int(time.time() * 1000) - 10_000, 2_400)
        self.assertTrue(ok)

    def test_stats_ignore_poisoned_rows_already_in_the_log(self):
        poisoned = {"ts": int(time.time() * 1000) - 10_000, "tf": "1d", "side": "long",
                    "setup": "dm", "entry": 4e-06, "sl": 4e-06, "tp": 5e-06, "r_mult": 997.0}
        clean = {"ts": int(time.time() * 1000) - 10_000, "tf": "1d", "side": "long",
                 "setup": "dm", "entry": 100.0, "sl": 90.0, "tp": 118.0, "r_mult": -0.5}
        shadow._save({"pending": [], "resolved": [poisoned, clean]})
        st = shadow.stats("1d")
        self.assertEqual(st["n"], 1)
        self.assertAlmostEqual(st["avg_r"], -0.5)
        self.assertFalse(shadow.is_clean_row(poisoned))
        self.assertTrue(shadow.is_clean_row(clean))


if __name__ == "__main__":
    unittest.main()
