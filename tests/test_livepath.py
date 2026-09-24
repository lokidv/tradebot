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

import httpx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import bracket  # noqa: E402
import candidates  # noqa: E402
import edge_book  # noqa: E402
import engine  # noqa: E402
import market  # noqa: E402
import meta_gate  # noqa: E402
import report  # noqa: E402
import shadow  # noqa: E402

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


# ───────────────────────── LP-9 (+LP-6): ترتیبِ زمانیِ جیب‌ها و بلوکِ هر تایم‌فریم ─────────────────────────
class _Shadow:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = shadow.SHADOW_PATH
        shadow.SHADOW_PATH = os.path.join(self.tmp.name, "signals.json")

    def tearDown(self):
        shadow.SHADOW_PATH = self._old
        edge_book._cache.update(ts=0.0, pockets={}, suspended={}, health={})
        self.tmp.cleanup()

    @staticmethod
    def _resolved(ts, r_mult, setup="zx", side="long", tf="1h", sym="BTCUSDT"):
        return {"id": f"{sym}{ts}", "ts": ts, "symbol": sym, "tf": tf, "side": side, "setup": setup,
                "entry": 100.0, "sl": 99.0, "tp": 101.8, "r_mult": r_mult, "resolved_at": ts / 1000}


class EdgeBookOrderTests(_Shadow, unittest.TestCase):
    def _save_newest_first(self, old_r, new_r, n=30):
        now = int(time.time() * 1000)
        rows = [self._resolved(now - (n - i) * 20 * H // 2, old_r if i < n // 2 else new_r)
                for i in range(n)]                                # قدیمی→جدید
        shadow._save({"pending": [], "resolved": rows[::-1]})     # مثلِ shadow.resolve: جدیدترین اول

    def test_recent_half_is_the_newest_trades_even_when_stored_newest_first(self):
        self._save_newest_first(1.8, -1.0)                        # ۱۵ برد قدیمی، ۱۵ باختِ تازه
        vals, stamps = edge_book._combo_rows(30)["1h|zx|long"]
        self.assertEqual(stamps, sorted(stamps))
        q = edge_book._series_quality(vals, stamps, edge_book.block_ms("1h"))
        self.assertAlmostEqual(q["recent"], -1.0)
        self.assertAlmostEqual(q["half1"], -1.0)
        pockets, _susp, _h = edge_book.refresh(force=True)
        self.assertNotIn("1h|zx|long", pockets)                   # قبلاً درجهٔ B می‌گرفت

    def test_block_scales_with_the_timeframe(self):
        self.assertEqual(edge_book.block_ms("1h"), edge_book.BLOCK_MS)
        self.assertEqual(edge_book.block_ms("4h"), bracket.MAX_BARS * 4 * H)
        self.assertEqual(edge_book.block_ms("15m"), bracket.MAX_BARS * H // 4)
        self.assertEqual(edge_book.block_ms("?"), edge_book.BLOCK_MS)

    def test_rule_rows_are_keyed_rule_not_zx(self):
        now = int(time.time() * 1000)
        shadow._save({"pending": [], "resolved": [self._resolved(now - H, 0.5, setup=None)]})
        self.assertIn("1h|rule|long", edge_book._combo_rows(30))
        self.assertIn("rule", shadow.stats("1h")["by_setup"])
        self.assertNotIn("zx", shadow.stats("1h")["by_setup"])


class ShadowDedupTests(_Shadow, unittest.TestCase):
    def test_dedup_looks_at_the_newest_resolved_rows(self):
        candle = int(time.time() * 1000) - 60_000
        newest = self._resolved(candle, 1.0)
        older = [self._resolved(candle - (i + 1) * H, 1.0, sym="ETHUSDT") for i in range(300)]
        shadow._save({"pending": [], "resolved": [newest] + older})
        ok = shadow.log_signal("BTCUSDT", "1h", "long", 100.0, 99.0, 101.8, "zx", 55, 0.4, candle, 60)
        self.assertFalse(ok)                                       # قبلاً ۲۰۰تای قدیمی را می‌دید ⇒ تکرار


# ───────────────────────── LP-4: کشِ کندل پس از مرز، دادهٔ پیش از مرز را تازه نمی‌شمارد ─────────────────────────
class _Clock:
    def __init__(self, t):
        self.t = float(t)

    def time(self):
        return self.t


def _binance_rows(now_s, step, n=421):
    """بایننسِ ساختگی: n کندلِ step تا کندلِ باز در لحظهٔ now_s."""
    last = int(now_s * 1000) // step * step
    return [[t, "1", "2", "0.5", "1.5", "10", t + step - 1, "15", "7", "4", "6", "0"]
            for t in range(last - (n - 1) * step, last + 1, step)]


class _MarketBase:
    SYM = "ZZLPUSDT"

    def setUp(self):
        market._host_dead.clear()
        market._neg_until.clear()
        with market._lock:
            market._kfund_cache.clear()

    def tearDown(self):
        self.setUp()
        with market._lock:
            for k in [k for k in market._kline_cache if k[0].startswith("ZZ")]:
                market._kline_cache.pop(k, None)


class KlineCacheBoundaryTests(_MarketBase, unittest.TestCase):
    B = 1_790_280_000.0                     # یک مرزِ ساعتیِ UTC

    def test_pre_boundary_fetch_is_not_served_after_the_boundary(self):
        clock, calls = _Clock(self.B - 10), []

        def binance(_path, params=None):
            calls.append(clock.t)
            return _binance_rows(clock.t, H)

        with mock.patch.object(market, "time", clock), \
                mock.patch.object(market, "_binance_json", side_effect=binance):
            first = market.get_klines(self.SYM, "1h")
            self.assertEqual(first["t"][-1], int(self.B * 1000) - 2 * H)
            clock.t = self.B + 20                        # دورِ زمان‌بند: کندلِ B−1h تازه بسته شده
            kl = market.get_klines(self.SYM, "1h")
            clock.t = self.B + 1800
            again = market.get_klines(self.SYM, "1h")    # همان کندل: از کش
        self.assertEqual(kl["t"][-1], int(self.B * 1000) - H)   # مهلتِ ۴۵ثانیه‌ای این را کهنه می‌داد
        self.assertIs(again, kl)
        self.assertEqual(len(calls), 2)

    def test_lagging_exchange_is_retried_within_the_bar_not_cached_for_it(self):
        step = 4 * H
        B = self.B - (self.B % (step / 1000))
        hit = (B + 20, {"t": [int(B * 1000) - 2 * step]})       # پس از مرز گرفته شد ولی کندلِ تازه نیامده
        self.assertTrue(market._kl_fresh(hit, "4h", B + 60))
        self.assertFalse(market._kl_fresh(hit, "4h", B + 20 + market.KLINE_TTL["4h"]))
        ok = (B + 20, {"t": [int(B * 1000) - step]})
        self.assertTrue(market._kl_fresh(ok, "4h", B + 3 * 3600))
        self.assertFalse(market._kl_fresh(ok, "4h", B + 4 * 3600 + 1))


# ───────────────────────── LP-11: پشتیبانِ OKX ─────────────────────────
class _Okx:
    """OKXِ ساختگی: /market/candles جدیدترین-اول، حداکثر ۳۰۰، با after؛ کندلِ باز confirm=0."""

    def __init__(self, now_s, step, n_total=2000):
        self.step, self.calls = step, []
        self.last = int(now_s * 1000) // step * step
        self.first = self.last - (n_total - 1) * step

    def __call__(self, url, params=None):
        assert "okx.com/api/v5/market/candles" in url, url
        self.calls.append(dict(params))
        top = self.last if "after" not in params else int(params["after"]) - self.step
        lim = min(int(params["limit"]), 300)
        ts = [t for t in range(top, self.first - 1, -self.step)][:lim]
        return {"data": [[str(t), "1", "2", "0.5", "1.5", "10", "10", "15", "0" if t == self.last else "1"]
                         for t in ts]}


class OkxFallbackTests(_MarketBase, unittest.TestCase):
    def _fallback(self, tf, step, limit=420):
        now = time.time()
        okx = _Okx(now, step)
        with mock.patch.object(market, "_binance_json", side_effect=httpx.ConnectError("down")), \
                mock.patch.object(market, "_get_json", side_effect=okx):
            kl = market.get_klines(self.SYM + tf, tf, limit)
        return kl, okx, now

    def test_fallback_pages_to_the_420_bars_the_live_engine_assumes(self):
        kl, okx, now = self._fallback("1h", H)
        self.assertEqual(kl["src"], "okx")
        self.assertEqual(len(kl["t"]), 420)
        self.assertTrue(all(b - a == H for a, b in zip(kl["t"], kl["t"][1:])))
        self.assertEqual(kl["t"][-1], int(now * 1000) // H * H - H)           # آخرین کندلِ بسته
        self.assertEqual(len(okx.calls), 2)
        self.assertEqual(okx.calls[1]["after"], str(okx.last - 299 * H))

    def test_daily_fallback_asks_for_utc_candles(self):
        _kl, okx, _now = self._fallback("1d", 24 * H)
        self.assertEqual({c["bar"] for c in okx.calls}, {"1Dutc"})
        self.assertEqual(market.TF_OKX["1d"], "1Dutc")


# ───────────────────────── feat_flow-1: کشِ فاندینگِ کوکوین و موعدِ تسویه ─────────────────────────
class KucoinFundingSettlementTests(_MarketBase, unittest.TestCase):
    T = 1_790_265_600_000                               # یک تسویهٔ ۸ساعته (۱۶:۰۰ UTC)

    def _hit(self, fetched_s, last_ms, rows=181):
        t = np.array([last_ms - 8 * H * i for i in range(rows)][::-1], dtype=np.int64)
        return (fetched_s, {"t": t, "rate": np.zeros(rows), "interval_h": np.full(rows, 8.0)}, rows)

    def test_cache_filled_before_a_settlement_is_stale_right_after_it(self):
        hit = self._hit(self.T / 1000 - 300, self.T - 8 * H)            # پر شده ۵ دقیقه پیش از T
        self.assertTrue(market._kfund_fresh(hit, 181, now=self.T / 1000 - 60))
        self.assertFalse(market._kfund_fresh(hit, 181, now=self.T / 1000 + 30))   # قبلاً تا T+۵ دقیقه تازه

    def test_venue_lag_is_retried_briefly_then_the_normal_ttl_applies(self):
        lag = self._hit(self.T / 1000 + 10, self.T - 8 * H)            # پس از T گرفته شد، ردیفِ T هنوز نیست
        self.assertTrue(market._kfund_fresh(lag, 181, now=self.T / 1000 + 20))
        self.assertFalse(market._kfund_fresh(lag, 181, now=self.T / 1000 + 10 + market.KFUND_RETRY_SEC))
        late = self._hit(self.T / 1000 + 900, self.T - 8 * H)          # ۱۵ دقیقه پس از موعد: برنامه عوض شده؟
        self.assertTrue(market._kfund_fresh(late, 181, now=self.T / 1000 + 900 + 120))
        ok = self._hit(self.T / 1000 + 60, self.T)
        self.assertTrue(market._kfund_fresh(ok, 181, now=self.T / 1000 + 500))

    def test_live_fetch_after_the_settlement_picks_up_the_record_at_the_close(self):
        listed = [self.T - 8 * H * i for i in range(400)][::-1]
        clock = _Clock(self.T / 1000 - 300)

        def venue(url, params=None):
            vis = [t for t in listed if t <= clock.t * 1000 and params["from"] <= t <= params["to"]]
            vis = sorted(vis, reverse=True)[:100]
            return {"code": "200000", "data": [{"timepoint": t, "fundingRate": 1e-4} for t in vis]}

        with mock.patch.object(market, "time", clock), mock.patch.object(market, "_get_json", side_effect=venue):
            before = market.get_kucoin_funding("ZZBTCUSDT")
            self.assertEqual(int(before["t"][-1]), self.T - 8 * H)
            clock.t = self.T / 1000 + 30                                   # کندلِ بسته‌شده روی T
            after = market.get_kucoin_funding("ZZBTCUSDT")
        self.assertEqual(int(after["t"][-1]), self.T)


if __name__ == "__main__":
    unittest.main()
