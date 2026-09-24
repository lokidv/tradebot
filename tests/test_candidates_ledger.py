# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ دفترِ کاندیدها — شکستنِ بن‌بستِ یادگیریِ زنده.

معیارِ اصلی: وقتی **همهٔ** گیت‌ها بسته‌اند، باز هم ردیف ثبت شود؛ وگرنه سیستم
هرگز نمی‌فهمد لبه‌ای هست یا نه.
"""
import os
import sys
import tempfile
import time
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import bracket  # noqa: E402
import candidates  # noqa: E402


def make_row(**over):
    row = {
        "symbol": "BTCUSDT", "side": "long", "setup": "zx",
        "zt": int(time.time() * 1000) - 60_000,
        "entry": 100.0, "sl": 97.4, "tp": 104.68, "atr14": 2.0,
        "cost": 0.11, "p_win": 41.0, "ev_pct": -0.2, "regime": "رونددار",
        "viable": False, "tradeable": False, "gate_allowed": False,
        "authority": None, "policy_trusted": False, "policy_pass": False,
        "regime_veto": False, "signal_score": None,
    }
    row.update(over)
    return row


def bars(n, start_price=100.0, drift=0.0, wick=0.3, t0=None, step=3_600_000):
    """کندل‌های مصنوعی که به «حالا» ختم می‌شوند تا گاردِ فیدِ مرده فعال نشود."""
    if t0 is None:
        t0 = int(time.time() * 1000) - n * step
    c = start_price + drift * np.arange(n)
    return {
        "t": (t0 + np.arange(n, dtype=np.int64) * step).tolist(),
        "o": c.tolist(),
        "h": (c + wick).tolist(),
        "l": (c - wick).tolist(),
        "c": c.tolist(),
        "v": np.full(n, 1.0).tolist(),
    }


class _LedgerMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = (candidates.CAND_PATH, candidates.RESULT_PATH)
        candidates.CAND_PATH = os.path.join(self.tmp.name, "candidates.jsonl")
        candidates.RESULT_PATH = os.path.join(self.tmp.name, "results.jsonl")
        candidates._reset_cache()

    def tearDown(self):
        candidates.CAND_PATH, candidates.RESULT_PATH = self._old
        candidates._reset_cache()
        self.tmp.cleanup()


class LoggingEveryCandidateTests(_LedgerMixin, unittest.TestCase):
    def test_blocked_candidate_is_still_logged(self):
        self.assertTrue(candidates.log_candidate(make_row(), "1h"))
        rows = candidates._read(candidates.CAND_PATH)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["gate_state"]["tradeable"])
        self.assertFalse(rows[0]["gate_state"]["gate_allowed"])

    def test_blocking_gates_are_recorded_for_later_analysis(self):
        candidates.log_candidate(make_row(regime_veto=True, tf_suspended=-0.2), "1h")
        gs = candidates._read(candidates.CAND_PATH)[0]["gate_state"]
        blocking = candidates._blocking_gates(gs)
        self.assertIn("gates_allowlist", blocking)
        self.assertIn("policy_untrusted", blocking)
        self.assertIn("regime_veto", blocking)
        self.assertIn("tf_suspended", blocking)

    def test_same_candle_is_never_logged_twice(self):
        row = make_row()
        self.assertTrue(candidates.log_candidate(row, "1h"))
        self.assertFalse(candidates.log_candidate(row, "1h"))
        self.assertEqual(len(candidates._read(candidates.CAND_PATH)), 1)

    def test_both_sides_of_the_same_candle_are_distinct_candidates(self):
        self.assertTrue(candidates.log_candidate(make_row(side="long"), "1h"))
        self.assertTrue(candidates.log_candidate(
            make_row(side="short", sl=102.6, tp=95.32), "1h"))
        self.assertEqual(len(candidates._read(candidates.CAND_PATH)), 2)

    def test_zero_risk_and_dead_feed_rows_are_refused(self):
        self.assertFalse(candidates.log_candidate(
            make_row(symbol="SHIBUSDT", entry=4e-06, sl=4e-06, atr14=1e-12), "1d"))
        stale = int((time.time() - 40 * 86400) * 1000)
        self.assertFalse(candidates.log_candidate(make_row(zt=stale), "1h"))
        self.assertEqual(candidates._read(candidates.CAND_PATH), [])


class ResolutionMatchesTheTrainingLabelTests(_LedgerMixin, unittest.TestCase):
    def test_entry_is_the_next_bar_open_not_the_displayed_price(self):
        kl = bars(60, start_price=100.0, drift=0.5)
        cand = {"id": "x", "symbol": "BTCUSDT", "tf": "1h", "side": "long",
                "setup": "zx", "candle_ts": kl["t"][10], "atr14": 2.0,
                "proposed_entry": 999.0,          # عددِ نمایشیِ بی‌ربط
                "cost_pct": 0.0, "gate_state": {"tradeable": False}}
        res = candidates.resolve_one(cand, kl)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["entry"], kl["o"][11])
        lv = bracket.levels(kl["o"][11], 2.0, 1)
        self.assertAlmostEqual(res["sl"], lv["sl"])
        self.assertAlmostEqual(res["tp"], lv["tp"])

    def test_target_hit_gives_one_point_eight_r_before_cost(self):
        kl = bars(60, start_price=100.0, drift=0.5)     # روندِ صعودیِ پیوسته
        cand = {"id": "x", "symbol": "B", "tf": "1h", "side": "long", "setup": "zx",
                "candle_ts": kl["t"][10], "atr14": 2.0, "cost_pct": 0.0,
                "gate_state": {}}
        res = candidates.resolve_one(cand, kl)
        self.assertEqual(res["outcome"], bracket.OUTCOME_TARGET)
        self.assertAlmostEqual(res["gross_r"], 1.8, places=6)
        self.assertAlmostEqual(res["net_r"], 1.8, places=6)

    def test_cost_is_subtracted_in_r_units(self):
        kl = bars(60, start_price=100.0, drift=0.5)
        cand = {"id": "x", "symbol": "B", "tf": "1h", "side": "long", "setup": "zx",
                "candle_ts": kl["t"][10], "atr14": 2.0, "cost_pct": 0.26,
                "gate_state": {}}
        res = candidates.resolve_one(cand, kl)
        # ۱R ≈ ۲٫۶ روی قیمتِ ~۱۰۵ ⇒ risk_pct ≈ ۲٫۴۷٪ ⇒ هزینه ≈ ۰٫۱۱R
        self.assertLess(res["net_r"], res["gross_r"])
        self.assertAlmostEqual(
            res["net_r"], round(res["gross_r"] - 0.26 / res["risk_pct"], 4), places=4)

    def test_open_candidate_is_not_resolved_early(self):
        kl = bars(12, start_price=100.0, drift=0.01)    # نه باری خورد، نه ۴۰ کندل گذشت
        cand = {"id": "x", "symbol": "B", "tf": "1h", "side": "long", "setup": "zx",
                "candle_ts": kl["t"][5], "atr14": 2.0, "cost_pct": 0.1, "gate_state": {}}
        self.assertIsNone(candidates.resolve_one(cand, kl))

    def test_resolve_walks_the_ledger_and_is_idempotent(self):
        # کندل‌های ساعتیِ واقعی‌شکل: کندلِ سیگنال یک ساعت پیش (از گاردِ «فیدِ مرده» رد می‌شود) و
        # کندلِ ورود دقیقاً یک کندل بعد — نه گامِ یک‌دقیقه‌ای که حالا «no_data» می‌شود (LP-7)
        kl = bars(80, start_price=100.0, drift=0.5, t0=int(time.time() * 1000) - 11 * 3_600_000)
        self.assertTrue(candidates.log_candidate(make_row(zt=kl["t"][10]), "1h"))
        self.assertEqual(candidates.resolve(lambda *_a: kl), 1)
        self.assertEqual(candidates.resolve(lambda *_a: kl), 0)
        rows = candidates._read(candidates.RESULT_PATH)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("skipped", rows[0])
        self.assertAlmostEqual(rows[0]["entry"], kl["o"][11])


class HonestStatsTests(_LedgerMixin, unittest.TestCase):
    def test_stats_cover_all_candidates_and_can_isolate_the_tradeable_ones(self):
        now = int(time.time() * 1000)
        v = candidates.SETUP_KEY_V          # ردیفِ نوع‌جدید: «zx» یعنی واقعاً کراسِ z
        rows = [
            {"id": "a", "tf": "1h", "setup": "zx", "setup_v": v, "side": "long", "candle_ts": now,
             "net_r": 1.6, "was_tradeable": True, "blocking_gates": []},
            {"id": "b", "tf": "1h", "setup": "zx", "setup_v": v, "side": "long", "candle_ts": now,
             "net_r": -1.1, "was_tradeable": False, "blocking_gates": ["gates_allowlist"]},
            {"id": "c", "tf": "1h", "setup": "zx", "setup_v": v, "side": "short", "candle_ts": now,
             "net_r": -1.1, "was_tradeable": False, "blocking_gates": ["gates_allowlist"]},
        ]
        for r in rows:
            candidates._append(candidates.RESULT_PATH, r)
        every = candidates.stats("1h")
        self.assertEqual(every["n"], 3)
        self.assertAlmostEqual(every["avg_net_r"], round((1.6 - 1.1 - 1.1) / 3, 4))
        only = candidates.stats("1h", only_tradeable=True)
        self.assertEqual(only["n"], 1)
        self.assertAlmostEqual(only["avg_net_r"], 1.6)
        self.assertEqual(every["by_blocking_gate"]["gates_allowlist"]["n"], 2)
        self.assertEqual(every["by_combo"]["1h|zx|long"]["n"], 2)

    def test_breakeven_reference_is_reported_next_to_the_win_rate(self):
        self.assertAlmostEqual(candidates.stats()["breakeven_win_rate"], 35.7)


if __name__ == "__main__":
    unittest.main()
