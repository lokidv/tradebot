# -*- coding: utf-8 -*-
"""میزِ تصمیم در main: endpointِ ‎/api/decisions، کشِ کوتاهِ خطا، و سقفِ انتظارِ منتظرِ تک‌پروازی.

همه چیز stub است: هیچ درخواستِ شبکه، هیچ سرور و هیچ فایلِ bot/data واقعی.
"""
import inspect
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

from fastapi.testclient import TestClient  # noqa: E402

import decision  # noqa: E402
import main  # noqa: E402
import market  # noqa: E402

PAYLOAD_KEYS = ("generated_at", "coins", "tfs", "cells", "note", "scorecard_meta", "live_overall")


def _stub_scorecards(*_a, **_k):
    return {"version": getattr(decision, "SCORECARD_VERSION", 1), "built_at": None,
            "cells": {}, "building": True}


def _no_klines(*_a, **_k):
    raise RuntimeError("stub: بدونِ شبکه")


class _Counter:
    """get_analysisِ ساختگی: هر کلید را می‌شمارد و خطای کنترل‌شده برمی‌گرداند."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = {}

    def __call__(self, symbol, tf, max_age=None):
        with self.lock:
            self.calls[(symbol, tf)] = self.calls.get((symbol, tf), 0) + 1
        if (symbol, tf) == ("BTCUSDT", "4h"):            # یک خانهٔ واقعی‌نما: لانگِ قاعدهٔ z/رأی
            zt = int(time.time() // 14400 * 14400 * 1000) - 14400000
            return {"symbol": symbol, "tf": tf, "zt": zt, "price": 100.0, "z": 1.5,
                    "votes_bull": 4, "votes_bear": 0, "components": {}, "regime": "trend",
                    "trade": {"side": "long", "setup": None, "setup_observed": False, "grade": "B",
                              "entry": 100.0, "sl": 98.0, "tp": 103.6, "rr": 1.8,
                              "risk_pct": 2.0, "atr14": 1.5}}
        return {"symbol": symbol, "tf": tf, "error": "stub"}


class DecisionsEndpointTests(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(main.app)               # بی‌context manager ⇒ رویدادِ startup اجرا نمی‌شود
        self.tmp = tempfile.mkdtemp(prefix="decisions-api-")
        self.patches = [
            mock.patch.object(decision, "LEDGER_PATH", os.path.join(self.tmp, "ledger.jsonl")),
            mock.patch.object(decision, "scorecards", _stub_scorecards),
            mock.patch.object(market, "get_klines", _no_klines),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def test_endpoint_is_sync_so_it_runs_in_threadpool(self):
        self.assertFalse(inspect.iscoroutinefunction(main.decisions_view))

    def test_returns_payload_and_computes_each_cell_once(self):
        stub = _Counter()
        with mock.patch.object(main, "get_analysis", stub):
            r = self.c.get("/api/decisions")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        for k in PAYLOAD_KEYS:
            self.assertIn(k, body)
        self.assertEqual(list(body["coins"]), list(decision.SYMBOLS))
        self.assertEqual(list(body["tfs"]), list(decision.TFS))
        self.assertEqual(body["tfs"][0], "5m")                    # ماتریس 5m را دارد
        self.assertIn("5m", body["cells"]["BTCUSDT"])
        self.assertIn("maker_cost_pct", body["scorecard_meta"])
        for sym in decision.SYMBOLS:
            for tf in decision.TFS:
                self.assertIn(tf, body["cells"][sym])
        self.assertEqual(body["cells"]["BTCUSDT"]["4h"]["action"], "long")
        self.assertEqual(body["cells"]["ETHUSDT"]["1h"]["action"], "wait")
        self.assertTrue(os.path.exists(decision.LEDGER_PATH))   # لانگ در دفترِ موقت ثبت شد
        # هر خانه دقیقاً یک‌بار (موازی روی _pool) — نه دوباره در collect
        want = {(s, tf) for s in decision.SYMBOLS for tf in decision.TFS}
        self.assertEqual(set(stub.calls), want)
        self.assertTrue(all(v == 1 for v in stub.calls.values()), stub.calls)

    def test_failure_is_persian_500(self):
        with mock.patch.object(decision, "refresh", side_effect=RuntimeError("boom")), \
                mock.patch.object(main, "get_analysis", _Counter()):
            r = self.c.get("/api/decisions")
        self.assertEqual(r.status_code, 500)
        self.assertIn("تصمیم", r.json().get("detail", ""))

    def test_lookup_falls_back_for_unknown_key(self):
        stub = _Counter()
        with mock.patch.object(main, "get_analysis", stub):
            look = main._decision_lookup(symbols=("BTCUSDT",), tfs=("1h",))
            self.assertEqual(look("BTCUSDT", "1h")["tf"], "1h")
            self.assertEqual(look("ETHUSDT", "4h")["symbol"], "ETHUSDT")
        self.assertEqual(stub.calls, {("BTCUSDT", "1h"): 1, ("ETHUSDT", "4h"): 1})

    def test_background_cycle_records_and_survives_errors(self):
        with mock.patch.object(main, "get_analysis", _Counter()),                 mock.patch.dict(main._decision_state, {"last_boundary": None}),                 mock.patch.object(decision, "resolve", side_effect=RuntimeError("down")):
            out = main._decision_cycle()           # دورِ اول ⇒ همهٔ تایم‌فریم‌ها
        self.assertEqual(out["decisions"], len(decision.SYMBOLS) * len(decision.TFS))
        self.assertEqual(out["tfs"], list(decision.TFS))
        self.assertEqual(out["recorded"], 1)       # فقط خانهٔ لانگِ BTC 4h ثبت می‌شود
        self.assertIsNone(out["resolved"])         # خطای resolve بلعیده و لاگ شد

    def test_next_run_is_20s_after_each_five_minutes(self):
        self.assertEqual(main.DECISION_PERIOD, 300)
        b = 1_700_000_100 - (1_700_000_100 % 300)
        self.assertEqual(main._next_decision_run(b + 5), b + main.DECISION_LAG)
        self.assertEqual(main._next_decision_run(b + main.DECISION_LAG), b + 300 + main.DECISION_LAG)
        self.assertEqual(main._next_decision_run(b + 200), b + 300 + main.DECISION_LAG)

    def test_only_timeframes_whose_bar_just_closed_are_refreshed(self):
        day = 1_700_000_000 - (1_700_000_000 % 86400)           # نیمه‌شبِ UTC
        due = main._due_tfs
        self.assertEqual(due(day, None), tuple(decision.TFS))     # دورِ اول ⇒ همه
        self.assertEqual(due(day, day - 300), tuple(decision.TFS))          # نیمه‌شب ⇒ همه بسته شدند
        self.assertEqual(due(day + 300, day), ("5m",))
        self.assertEqual(due(day + 900, day + 600), ("5m", "15m"))
        self.assertEqual(due(day + 3600, day + 3300), ("5m", "15m", "1h"))
        self.assertEqual(due(day + 14400, day + 14100), ("5m", "15m", "1h", "4h"))
        self.assertEqual(due(day + 3900, day + 3300), ("5m", "15m", "1h"))  # دورِ جاافتاده: مرزِ ساعت گم نمی‌شود
        self.assertEqual(due(day + 300, day + 300), ())                    # همان مرز ⇒ هیچ
        self.assertEqual(main._decision_boundary(day + 300 + main.DECISION_LAG + 3), day + 300)

    def test_cycle_collects_only_the_due_timeframes(self):
        day = 1_700_000_000 - (1_700_000_000 % 86400)
        stub = _Counter()
        # خانه‌های سالم (کندلِ همین مرز)؛ خانهٔ خطادار حالا دورِ بعد دوباره تحلیل می‌شود (test_livepath)
        real = stub.__call__

        def fresh(symbol, tf, max_age=None):
            real(symbol, tf, max_age)
            return {"symbol": symbol, "tf": tf, "zt": main._expected_bar(tf, day + 900),
                    "price": 1.0, "z": 0.0, "votes_bull": 0, "votes_bear": 0, "trade": {}}
        with mock.patch.object(main, "get_analysis", fresh),                 mock.patch.dict(main._decision_state, {"last_boundary": day + 600, "retry": {}}),                 mock.patch.object(decision, "resolve", return_value=0) as res:
            out = main._decision_cycle(now=day + 900 + main.DECISION_LAG)
            self.assertEqual(main._decision_state["last_boundary"], day + 900)
            again = main._decision_cycle(now=day + 900 + main.DECISION_LAG + 30)   # همان مرز: فقط داوری
        self.assertEqual(out["tfs"], ["5m", "15m"])
        self.assertEqual(out["decisions"], len(decision.SYMBOLS) * 2)
        self.assertEqual({tf for _, tf in stub.calls}, {"5m", "15m"})
        self.assertEqual((again["tfs"], again["decisions"]), ([], 0))
        self.assertEqual(res.call_count, 2)                                   # داوری هر دور برای همه


class ErrorTtlTests(unittest.TestCase):
    T = 1_700_000_000 - (1_700_000_000 % 86400) + 43200     # ظهرِ UTC: دور از مرزِ روزانه

    def _fresh(self, age, res, tf="1d"):
        with mock.patch.object(main, "time", types.SimpleNamespace(time=lambda: self.T)):
            return main._an_fresh((self.T - age, res), tf, None)

    def test_error_expires_after_60s_even_within_candle(self):
        err = {"symbol": "X", "tf": "1d", "error": "گذرا"}
        self.assertTrue(self._fresh(30, err))
        self.assertFalse(self._fresh(main.ERROR_TTL + 1, err))

    def test_good_result_still_lasts_until_candle_boundary(self):
        self.assertTrue(self._fresh(3600, {"symbol": "X", "tf": "1d", "z": 0.1}))

    def test_expired_error_is_recomputed(self):
        key = ("TTLUSDT", "1d")
        calls = []

        def compute(symbol, tf):
            calls.append((symbol, tf))
            return {"symbol": symbol, "tf": tf, "z": 0.0}
        try:
            with main._an_lock:
                main._an_cache[key] = (time.time() - main.ERROR_TTL - 5, {"symbol": key[0], "tf": key[1],
                                                                          "error": "گذرا"})
            with mock.patch.object(main, "_compute_analysis", compute):
                res = main.get_analysis(*key)
            self.assertNotIn("error", res)
            self.assertEqual(calls, [key])
        finally:
            with main._an_lock:
                main._an_cache.pop(key, None)


class WaiterCapTests(unittest.TestCase):
    def test_waiter_gives_up_without_duplicate_compute_or_touching_marker(self):
        key = ("CAPUSDT", "1h")
        release = threading.Event()
        started = threading.Event()
        calls = []

        def slow(symbol, tf):
            calls.append((symbol, tf))
            started.set()
            release.wait(10)
            return {"symbol": symbol, "tf": tf, "z": 0.5}

        owner_out = {}
        with mock.patch.object(main, "_compute_analysis", slow), \
                mock.patch.object(main, "INFLIGHT_MAX_WAIT", 0.3), \
                mock.patch.object(main, "INFLIGHT_POLL", 0.05):
            try:
                th = threading.Thread(target=lambda: owner_out.update(r=main.get_analysis(*key)), daemon=True)
                th.start()
                self.assertTrue(started.wait(5))
                with main._an_lock:
                    marker = main._inflight.get(key)
                self.assertIsNotNone(marker)

                t0 = time.time()
                res = main.get_analysis(*key)                  # منتظر: باید پس از ~۰٫۳ ثانیه رها کند
                waited = time.time() - t0
                self.assertLess(waited, 3.0)
                self.assertGreaterEqual(waited, 0.25)
                self.assertEqual(res.get("symbol"), key[0])
                self.assertEqual(res.get("tf"), key[1])
                self.assertIn("مهلت", res.get("error", ""))
                with main._an_lock:
                    self.assertIs(main._inflight.get(key), marker)   # نشانگرِ محاسبه‌گر دست‌نخورده
                    self.assertNotIn(key, main._an_cache)             # خطای انتظار کش نمی‌شود
                self.assertEqual(len(calls), 1)                        # محاسبهٔ تکراری ساخته نشد

                release.set()
                th.join(5)
                self.assertEqual(owner_out["r"].get("z"), 0.5)
                self.assertEqual(main.get_analysis(*key).get("z"), 0.5)   # حالا از کش
                self.assertEqual(len(calls), 1)
            finally:
                release.set()
                with main._an_lock:
                    main._an_cache.pop(key, None)
                    main._inflight.pop(key, None)


if __name__ == "__main__":
    unittest.main()
