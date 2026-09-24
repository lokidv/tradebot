# -*- coding: utf-8 -*-
"""سرعت و درستیِ موتور: تک‌پروازیِ تحلیل، کول‌داونِ میزبان، تک‌دریافتِ کندل/تیکر،
و سه اصلاحِ engine (p_up بی‌win_rate، ستاپِ خالی به‌جای «zx» ساختگی، متنِ ۲ کندل).

همه بدونِ شبکه: شبکه در سطحِ ``market._client`` یا ``market._binance_json`` جایگزین می‌شود.
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import httpx  # noqa: E402

import engine  # noqa: E402
import gates  # noqa: E402
import main  # noqa: E402
import market  # noqa: E402

from test_bracket_contract import synthetic_klines  # noqa: E402


def _run_threads(fn, n):
    out = [None] * n
    errs = []

    def work(i):
        try:
            out[i] = fn()
        except BaseException as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    return out, errs


# ───────────────────────── get_analysis: تک‌پروازی ─────────────────────────
class InflightTests(unittest.TestCase):
    def setUp(self):
        main._an_cache.clear()
        main._inflight.clear()
        self._poll = main.INFLIGHT_POLL
        main.INFLIGHT_POLL = 0.02            # منتظر ۳۰ بار بیشتر از «مهلت» صبر می‌کند

    def tearDown(self):
        main.INFLIGHT_POLL = self._poll
        main._an_cache.clear()
        main._inflight.clear()

    def test_concurrent_callers_share_one_computation(self):
        calls = []

        def slow(symbol, tf):
            calls.append((symbol, tf))
            time.sleep(0.6)                  # بسیار بیشتر از INFLIGHT_POLL (باگِ قبلی: wait(timeout) ⇒ کارِ تکراری)
            return {"symbol": symbol, "tf": tf, "n": len(calls)}

        with mock.patch.object(main, "_compute_analysis", side_effect=slow):
            res, errs = _run_threads(lambda: main.get_analysis("BTCUSDT", "1h"), 8)
        self.assertEqual(errs, [])
        self.assertEqual(len(calls), 1, "محاسبهٔ تکراری برای یک کلید")
        self.assertTrue(all(r is res[0] for r in res), "منتظرها باید نتیجهٔ همان محاسبه‌گر را بگیرند")
        self.assertEqual(main._inflight, {})

    def test_warm_call_does_not_compute(self):
        with mock.patch.object(main, "_compute_analysis", return_value={"x": 1}) as m:
            a = main.get_analysis("ETHUSDT", "4h")
            b = main.get_analysis("ETHUSDT", "4h")
        self.assertEqual(m.call_count, 1)
        self.assertIs(a, b)

    def test_owner_that_dies_hands_over_to_exactly_one_waiter(self):
        """محاسبه‌گری که با استثنا بیرون می‌رود نشانگرش را برمی‌دارد و منتظرها گیر نمی‌کنند."""
        calls = []
        started = threading.Event()

        def flaky(symbol, tf):
            calls.append(1)
            if len(calls) == 1:
                started.set()
                time.sleep(0.3)
                raise RuntimeError("boom")
            return {"ok": True}

        with mock.patch.object(main, "_compute_analysis", side_effect=flaky):
            owner = threading.Thread(target=lambda: self._swallow(main.get_analysis, "SOLUSDT", "15m"))
            owner.start()
            started.wait(5)
            res, errs = _run_threads(lambda: main.get_analysis("SOLUSDT", "15m"), 4)
            owner.join(5)
        self.assertEqual(errs, [])
        self.assertEqual(len(calls), 2, "پس از مرگِ محاسبه‌گر فقط یک منتظر باید دوباره حساب کند")
        self.assertTrue(all(r == {"ok": True} for r in res))
        self.assertEqual(main._inflight, {})

    @staticmethod
    def _swallow(fn, *a):
        try:
            fn(*a)
        except RuntimeError:
            pass

    def test_htf_chain_computes_each_key_once(self):
        """۵ ارز × زنجیرهٔ 15m→4h→1d و BTC: هر کلید دقیقاً یک‌بار — مثلِ overview واقعی."""
        calls = []
        lock = threading.Lock()

        def chain(symbol, tf):
            with lock:
                calls.append((symbol, tf))
            if symbol != "BTCUSDT":
                main.get_analysis("BTCUSDT", tf)
            htf = main.HTF_OF.get(tf)
            if htf:
                main.get_analysis(symbol, htf)
            time.sleep(0.05)
            return {"symbol": symbol, "tf": tf}

        syms = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "TRXUSDT"]
        with mock.patch.object(main, "_compute_analysis", side_effect=chain):
            res = list(main._pool.map(lambda s: main.get_analysis(s, "15m"), syms))
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(len(calls), 15)       # ۵ ارز × {15m, 4h, 1d}
        self.assertEqual([r["symbol"] for r in res], syms)


# ───────────────────────── market: یک دریافت برای هم‌زمان‌ها، کول‌داونِ میزبان ─────────────────────────
def _fake_kline_rows(n=421, tf_ms=3_600_000):
    now = int(time.time() * 1000) // tf_ms * tf_ms
    t0 = now - (n - 1) * tf_ms            # آخرین ردیف کندلِ باز است و حذف می‌شود
    return [[t0 + i * tf_ms, "1", "1.1", "0.9", "1.05", "10"] for i in range(n)]


class _Resp:
    def __init__(self, data, code, url):
        self._d, self.status_code, self._url = data, code, url

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("GET", self._url)
            raise httpx.HTTPStatusError(str(self.status_code), request=req,
                                        response=httpx.Response(self.status_code, request=req))

    def json(self):
        return self._d


class MarketTests(unittest.TestCase):
    def setUp(self):
        market._host_dead.clear()
        market._neg_until.clear()
        market._top_cache.update(ts=0.0, symbols=[], tickers={}, n_req=0)

    def tearDown(self):
        self.setUp()
        with market._lock:
            for k in [k for k in market._kline_cache if k[0].startswith("ZZ")]:
                market._kline_cache.pop(k, None)

    def test_top_symbols_50_100_200_is_one_ticker_fetch(self):
        now = time.time() * 1000
        rows = [{"symbol": f"C{i}USDT", "lastPrice": "10", "highPrice": "11", "lowPrice": "9",
                 "quoteVolume": str(1e9 - i), "priceChangePercent": "1", "closeTime": now}
                for i in range(300)]
        with mock.patch.object(market, "_binance_json", return_value=rows) as m:
            s50, _ = market.get_top_symbols(50)
            s100, _ = market.get_top_symbols(100)
            s200, tick = market.get_top_symbols(200)
        self.assertEqual(m.call_count, 1)
        self.assertEqual((len(s50), len(s100), len(s200)), (50, 100, 200))
        self.assertEqual(s100[:50], s50)
        self.assertIn("C199USDT", tick)

    def test_concurrent_get_klines_is_one_network_call(self):
        calls = []

        def slow(path, params=None):
            calls.append(params["symbol"])
            time.sleep(0.3)
            return _fake_kline_rows()

        with mock.patch.object(market, "_binance_json", side_effect=slow):
            res, errs = _run_threads(lambda: market.get_klines("ZZAUSDT", "1h"), 6)
        self.assertEqual(errs, [])
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(r is res[0] for r in res))
        self.assertEqual(len(res[0]["c"]), 420)

    def test_dead_host_is_skipped_until_cooldown_ends(self):
        seen = []

        class Client:
            def get(self, url, params=None):
                host = market._host_of(url)
                seen.append(host)
                if host == "data-api.binance.vision":
                    raise httpx.ConnectTimeout("hang")
                return _Resp({"price": "1"}, 200, url)

        with mock.patch.object(market, "_client", Client()):
            market._binance_json("/api/v3/ticker/price", {"symbol": "X"})
            market._binance_json("/api/v3/ticker/price", {"symbol": "X"})
            self.assertEqual(seen.count("data-api.binance.vision"), 1, "میزبانِ مرده دوباره امتحان شد")
            self.assertIn("data-api.binance.vision", market.host_status())
            market._host_dead["data-api.binance.vision"] = time.time() - 1      # پایانِ کول‌داون
            market._binance_json("/api/v3/ticker/price", {"symbol": "X"})
        self.assertEqual(seen.count("data-api.binance.vision"), 2)

    def test_bad_request_does_not_cascade_or_cool_the_host(self):
        seen = []

        class Client:
            def get(self, url, params=None):
                seen.append(url)
                return _Resp({"code": -1121}, 400, url)

        with mock.patch.object(market, "_client", Client()):
            with self.assertRaises(httpx.HTTPStatusError):
                market._binance_json("/api/v3/klines", {"symbol": "NOPEUSDT"})
        self.assertEqual(len(seen), 1)
        self.assertEqual(market.host_status(), {})

    def test_unreachable_funding_is_not_locked_on_disk_for_12h(self):
        sym = "ZZFUNDUSDT"
        path = os.path.join(market.HIST_DIR, f"funding_{sym}.json")
        n = []

        def down(*a, **k):
            n.append(1)
            raise httpx.ConnectError("down")

        with mock.patch.object(market, "_fapi_json", side_effect=down), \
                mock.patch.object(market, "_get_json", side_effect=down):
            self.assertEqual(market.get_funding_history(sym), [])
            self.assertEqual(market.get_funding_history(sym), [])          # از کشِ منفیِ حافظه
        self.assertFalse(os.path.exists(path), "«خالی» ناشی از قطعی نباید ۱۲ ساعت روی دیسک بماند")
        self.assertEqual(len(n), 2)                                        # یک fapi + یک OKX، فقط یک‌بار

    def test_answered_empty_funding_is_still_cached_on_disk(self):
        sym = "ZZNOPERPUSDT"
        path = os.path.join(market.HIST_DIR, f"funding_{sym}.json")
        with mock.patch.object(market, "_fapi_json", return_value=[]):
            self.assertEqual(market.get_funding_history(sym), [])
        self.assertTrue(os.path.exists(path))
        os.remove(path)


# ───────────────────────── engine: p_up، ستاپِ خالی، متنِ دلیل ─────────────────────────
class EngineFixTests(unittest.TestCase):
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

    def test_p_up_is_the_forecast_not_blended_with_backtest_win_rate(self):
        seen = {}
        real_fc = engine.forecast

        def spy(*a, **k):
            fc = real_fc(*a, **k)
            seen["p_up"] = fc["p_up"]
            return fc

        n_bt = 0
        # ستاپ در هر کندل ⇒ بک‌تست n≥8 دارد و شاخهٔ قدیمیِ «۰٫۶۵×p_up + ۰٫۳۵×win_rate» فعال می‌شد
        every_bar = lambda cs, o, h, l, c, i: (1 if i % 2 else -1, "pb")  # noqa: E731
        for seed in (3, 5, 9, 17):
            with mock.patch.object(engine, "forecast", side_effect=spy), \
                    mock.patch.object(engine, "setup_signal", side_effect=every_bar):
                res = engine.analyze(synthetic_klines(seed=seed), "1h")
            self.assertFalse(res["p_calibrated"])
            self.assertAlmostEqual(res["p_up"], round(seen["p_up"], 1), places=6)
            n_bt += res["backtest"]["n"] >= 8
        self.assertGreater(n_bt, 0, "هیچ بک‌تستی با n≥8 نبود — شاخهٔ قدیمیِ ترکیب آزموده نشد")

    def test_rule_side_without_an_event_setup_has_no_setup_label(self):
        real_ts = engine.trade_suggestion

        def force_long(*a, **k):
            k["force_side"] = k.get("force_side") or "long"
            return real_ts(*a, **k)

        with mock.patch.object(engine, "setup_signal", return_value=(0, None)), \
                mock.patch.object(engine, "trade_suggestion", side_effect=force_long):
            tr = engine.analyze(synthetic_klines(seed=5), "1h")["trade"]
        self.assertEqual(tr["side"], "long")
        self.assertIsNone(tr["setup"])
        self.assertIsNone(tr["setup_fa"])
        self.assertFalse(tr["setup_observed"])
        self.assertFalse(tr["gate_allowed"])
        self.assertFalse(tr["tradeable"])

    def test_event_setup_keeps_its_label(self):
        kl = synthetic_klines(seed=5)
        last = len(kl["c"]) - 1
        with mock.patch.object(engine, "setup_signal",
                               side_effect=lambda cs, o, h, l, c, i: (1, "pb") if i == last else (0, None)):
            tr = engine.analyze(kl, "1h")["trade"]
        self.assertEqual(tr["side"], "long")
        self.assertEqual(tr["setup"], "pb")
        self.assertEqual(tr["setup_fa"], engine.SETUP_FA["pb"])
        self.assertTrue(tr["setup_observed"])

    def test_no_side_trade_row_says_two_bars_and_has_no_setup(self):
        with mock.patch.object(engine, "setup_signal", return_value=(0, None)):
            for seed in range(1, 30):
                tr = engine.analyze(synthetic_klines(seed=seed), "1h")["trade"]
                if tr.get("side"):
                    continue
                text = " ".join(tr["reasons"])
                self.assertIn("۲ کندل اخیر", text)
                self.assertNotIn("۴ کندل", text)
                self.assertIsNone(tr["setup"])
                self.assertFalse(tr["setup_observed"])
                return
        self.fail("هیچ ردیفِ بی‌سیگنالی ساخته نشد")

    def test_lookback_constant_matches_the_text(self):
        self.assertEqual(engine.SETUP_LOOKBACK, 2)
        self.assertEqual(engine.SETUP_LOOKBACK_FA, "۲")


if __name__ == "__main__":
    unittest.main()
