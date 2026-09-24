# -*- coding: utf-8 -*-
"""مسیرِ زنده: دفترها، زمان‌بند و پشتیبان‌ها (یافته‌های LP-* ممیزی).

همه‌چیز stub است: هیچ درخواستِ شبکه و هیچ فایلِ bot/data واقعی.
"""
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import bracket  # noqa: E402
import candidates  # noqa: E402
import engine  # noqa: E402
import meta_gate  # noqa: E402
import report  # noqa: E402

H = 3_600_000


def _bars(n, t0, step=H, start=100.0, drift=0.5, wick=0.3):
    c = start + drift * np.arange(n)
    return {"t": (t0 + np.arange(n, dtype=np.int64) * step).tolist(), "o": c.tolist(),
            "h": (c + wick).tolist(), "l": (c - wick).tolist(), "c": c.tolist(),
            "v": np.full(n, 1.0).tolist()}


def _row(**over):
    row = {"symbol": "BTCUSDT", "side": "long", "setup": None, "setup_observed": False,
           "zt": int(time.time() * 1000) - 60_000, "entry": 100.0, "sl": 97.4, "tp": 104.68,
           "atr14": 2.0, "cost": 0.11, "tradeable": False}
    row.update(over)
    return row


class _Ledger:
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


# ───────────────────────── LP-2: ستاپِ معلق اسکن را تمام نمی‌کند ─────────────────────────
class SuspendedSetupScanTests(unittest.TestCase):
    N = 420

    def _analyze(self, signals, suspended):
        c = 100 + np.linspace(0, 4, self.N) + 0.3 * np.sin(np.arange(self.N) / 7)
        kl = {"t": (np.arange(self.N, dtype=np.int64) * H).tolist(), "o": c.tolist(),
              "h": (c + 0.4).tolist(), "l": (c - 0.4).tolist(), "c": c.tolist(),
              "v": np.full(self.N, 1_000.0).tolist()}
        sug = {"side": "long", "entry": 104.0, "sl": 103.0, "tp": 105.8, "risk_pct": 0.96,
               "rr": 1.8, "grade": "A", "viable": True, "tradeable": True, "status": "آماده",
               "reasons": [], "time_stop_min": 2_400}
        with mock.patch.object(engine, "setup_signal",
                               side_effect=lambda _cs, _o, _h, _l, _c, i: signals.get(i, (0, None))), \
                mock.patch.object(engine, "trade_suggestion", side_effect=lambda *a, **k: dict(sug)), \
                mock.patch.object(engine, "quick_backtest",
                                  return_value={"n": 0, "win_rate": None, "avg_r": None,
                                                "timeouts": 0, "profit_factor": None}), \
                mock.patch.object(meta_gate, "evaluate", return_value={"approve": False}):
            return engine.analyze(kl, "1h", extras={"symbol": "BTCUSDT", "suspended": suspended})["trade"]

    def test_suspended_setup_on_last_bar_lets_the_scan_reach_the_previous_bar(self):
        sig = {self.N - 1: (1, "fd"), self.N - 2: (1, "sq")}
        self.assertEqual(self._analyze(sig, {})["setup"], "fd")          # بی‌تعلیق: کندلِ آخر
        tr = self._analyze(sig, {"fd": -0.2})
        self.assertEqual(tr["setup"], "sq")                               # break اینجا None می‌داد
        self.assertTrue(tr["setup_observed"])

    def test_suspended_setup_alone_is_as_if_there_was_no_signal(self):
        tr = self._analyze({self.N - 1: (1, "fd")}, {"fd": -0.2})
        self.assertIsNone(tr.get("setup"))
        self.assertFalse(tr.get("setup_observed"))


# ───────────────────────── LP-6: قاعده ≠ zx ─────────────────────────
class SetupKeyTests(_Ledger, unittest.TestCase):
    def test_rule_trade_is_stored_as_rule_and_a_real_z_cross_as_zx(self):
        self.assertTrue(candidates.log_candidate(_row(side="long"), "4h"))
        self.assertTrue(candidates.log_candidate(_row(side="short", setup="zx", setup_observed=True), "4h"))
        rows = {r["side"]: r for r in candidates._read(candidates.CAND_PATH)}
        self.assertEqual(rows["long"]["setup"], candidates.RULE_SETUP)
        self.assertFalse(rows["long"]["setup_observed"])
        self.assertEqual(rows["short"]["setup"], "zx")
        self.assertTrue(rows["short"]["setup_observed"])
        self.assertEqual({r["setup_v"] for r in rows.values()}, {candidates.SETUP_KEY_V})

    def test_legacy_zx_rows_are_ambiguous_never_zx(self):
        self.assertEqual(candidates.setup_key({"setup": "zx"}), candidates.LEGACY_ZX)
        self.assertEqual(candidates.setup_key({"setup": None}), candidates.LEGACY_ZX)
        self.assertEqual(candidates.setup_key({"setup": "sq"}), "sq")          # همیشه واقعی بود
        self.assertEqual(candidates.setup_key({"setup": "zx", "setup_v": 2}), "zx")
        self.assertEqual(candidates.setup_key({"setup": None, "setup_v": 2}), "rule")

    def test_preregistered_zx_combo_only_sees_real_z_cross_rows(self):
        now = int(time.time() * 1000)
        rows = [  # نتیجهٔ نوع‌جدید: یک کراسِ z، یک قاعده؛ و یک «zx»ِ قدیمیِ مبهم
            {"id": "z", "tf": "4h", "setup": "zx", "setup_v": 2, "side": "long", "symbol": "BTCUSDT",
             "candle_ts": now, "net_r": 1.0},
            {"id": "r", "tf": "4h", "setup": "rule", "setup_v": 2, "side": "long", "symbol": "BTCUSDT",
             "candle_ts": now, "net_r": -1.0},
            {"id": "old", "tf": "4h", "setup": "zx", "side": "long", "symbol": "BTCUSDT",
             "candle_ts": now, "net_r": -1.0},
        ]
        for r in rows:
            candidates._append(candidates.RESULT_PATH, r)
        by = candidates.stats("4h")["by_combo"]
        self.assertEqual(by["4h|zx|long"]["n"], 1)
        self.assertEqual(by["4h|rule|long"]["n"], 1)
        self.assertEqual(by[f"4h|{candidates.LEGACY_ZX}|long"]["n"], 1)
        self.assertEqual(candidates.stats("4h", setup="zx")["n"], 1)

    def test_rule_rows_logged_now_never_reach_a_preregistered_zx_combo_in_the_report(self):
        t0 = int(time.time() * 1000) - 11 * 4 * H
        kl = _bars(80, t0, step=4 * H)
        self.assertTrue(candidates.log_candidate(_row(zt=kl["t"][10]), "4h"))                  # قاعده
        self.assertTrue(candidates.log_candidate(
            _row(side="long", setup="zx", setup_observed=True, zt=kl["t"][9]), "4h"))          # کراسِ z
        self.assertEqual(candidates.resolve(lambda *_a: kl), 2)
        sh = report.shadow_section(0, ["4h|zx|long"], [], now=time.time())
        self.assertEqual(sh["n"], 1)
        self.assertEqual(set(sh["by_combo"]), {"4h|zx|long"})

    def test_resolved_row_carries_the_setup_key(self):
        t0 = int(time.time() * 1000) - 11 * H
        kl = _bars(80, t0)
        self.assertTrue(candidates.log_candidate(_row(zt=kl["t"][10]), "1h"))
        self.assertEqual(candidates.resolve(lambda *_a: kl), 1)
        res = candidates._read(candidates.RESULT_PATH)[0]
        self.assertEqual(res["setup"], "rule")
        legacy = {"id": "L", "symbol": "B", "tf": "1h", "side": "long", "setup": "zx",
                  "candle_ts": kl["t"][10], "atr14": 2.0, "cost_pct": 0.1, "gate_state": {}}
        self.assertEqual(candidates.resolve_one(legacy, kl)["setup"], candidates.LEGACY_ZX)


# ───────────────────────── LP-7: فقط کندلِ دقیقاً بعد ─────────────────────────
class NextBarEntryTests(_Ledger, unittest.TestCase):
    def _cand(self, ts):
        return {"id": "x", "symbol": "B", "tf": "1h", "side": "long", "setup": "rule",
                "setup_v": 2, "candle_ts": ts, "atr14": 2.0, "cost_pct": 0.0, "gate_state": {}}

    def test_signal_older_than_the_window_is_skipped_not_judged_on_a_later_bar(self):
        kl = _bars(420, 500 * H)                                # پنجره از کندلِ ۵۰۰ شروع می‌شود
        res = candidates.resolve_one(self._cand(0), kl)          # سیگنال روی کندلِ ۰
        self.assertEqual(res["skipped"], "no_data")
        self.assertNotIn("net_r", res)
        self.assertEqual(res["candle_ts"], 0)

    def test_gap_right_after_the_signal_is_skipped(self):
        kl = _bars(100, 0)
        del kl["t"][11]                                          # کندلِ ورود غایب
        for k in ("o", "h", "l", "c", "v"):
            del kl[k][11]
        self.assertEqual(candidates.resolve_one(self._cand(kl["t"][10]), kl)["skipped"], "no_data")

    def test_skipped_rows_stay_out_of_the_stats(self):
        kl = _bars(420, 500 * H)
        candidates._append(candidates.CAND_PATH, self._cand(0))
        self.assertEqual(candidates.resolve(lambda *_a: kl), 1)       # بسته شد (بی‌تکرار در دفعهٔ بعد)
        self.assertEqual(candidates.resolve(lambda *_a: kl), 0)
        self.assertEqual(candidates.stats("1h", days=100000)["n"], 0)

    def test_next_bar_still_resolves_normally(self):
        kl = _bars(100, 0)
        res = candidates.resolve_one(self._cand(kl["t"][10]), kl)
        self.assertAlmostEqual(res["entry"], kl["o"][11])
        self.assertEqual(res["outcome"], bracket.OUTCOME_TARGET)


# ───────────────────────── LP-8: مسابقهٔ داوری و هزینهٔ خواندن ─────────────────────────
class ResolveRaceTests(_Ledger, unittest.TestCase):
    def _log(self, n=5):
        t0 = int(time.time() * 1000) - 11 * H
        kl = _bars(80, t0)
        for i in range(n):
            self.assertTrue(candidates.log_candidate(_row(symbol=f"S{i}USDT", zt=kl["t"][10]), "1h"))
        return kl

    def test_two_concurrent_calls_never_duplicate_results(self):
        kl = self._log()
        gate = threading.Barrier(2)

        def slow(*_a):
            try:
                gate.wait(timeout=2)            # هر دو نخ پیش از افزودن به اینجا می‌رسند
            except threading.BrokenBarrierError:
                pass
            return kl

        out = []
        th = [threading.Thread(target=lambda: out.append(candidates.resolve(slow))) for _ in range(2)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        rows = candidates._read(candidates.RESULT_PATH)
        self.assertEqual(sorted(out), [0, 5])
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({r["id"] for r in rows}), 5)

    def test_stats_and_counts_dedup_old_duplicates(self):
        now = int(time.time() * 1000)
        r = {"id": "a", "tf": "1h", "setup": "rule", "setup_v": 2, "side": "long",
             "candle_ts": now, "net_r": 1.0}
        candidates._append(candidates.CAND_PATH, {"id": "a", "logged_at": 1.0})
        candidates._append(candidates.RESULT_PATH, r)
        candidates._append(candidates.RESULT_PATH, dict(r))      # تکرارِ مسابقهٔ قدیمی
        self.assertEqual(candidates.stats("1h")["n"], 1)
        c = candidates.counts()
        self.assertEqual((c["resolved"], c["open"]), (1, 0))

    def test_results_file_is_read_once_per_call_not_once_per_candidate(self):
        kl = self._log(20)
        reads = []
        real = candidates._read

        def counting(path):
            reads.append(path)
            return real(path)

        with mock.patch.object(candidates, "_read", side_effect=counting):
            candidates.resolve(lambda *_a: kl)
        self.assertLessEqual(reads.count(candidates.RESULT_PATH), 2)

    def test_cost_basis_is_labelled_on_stats_and_results(self):
        kl = self._log(1)
        candidates.resolve(lambda *_a: kl)
        res = candidates._read(candidates.RESULT_PATH)[0]
        self.assertEqual((res["cost_pct"], res["cost_model"]), (0.11, "tier"))
        cb = candidates.stats("1h")["cost_basis"]
        self.assertEqual(cb["model"], "tier")
        self.assertFalse(cb["funding_included"])
        self.assertIn("event_cost_pct", cb["pre_registered"])


if __name__ == "__main__":
    unittest.main()
