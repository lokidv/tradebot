# -*- coding: utf-8 -*-
"""وضعیتِ بازار (posdata): فقط نمایشی، بی‌شبکه — HTTP با تابعِ ساختگی جایگزین می‌شود.

پارس کردنِ هر منبع، کنار گذاشتنِ ردیفِ ناقصِ OKX، ریاضیِ عدمِ توازنِ دفترِ سفارش، کشِ ۶۰ ثانیه،
جدا ماندنِ خرابیِ هر منبع، فاصلهٔ ۵ دقیقه‌ایِ ثبت و مسیرِ هرمتیک."""
import json
import os
import shutil
import sys
import unittest
from unittest import mock

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import paths  # noqa: E402
import posdata  # noqa: E402

BAR = 300_000
H = 3_600_000
T_END = 1_790_236_800_000          # مرزِ ۵ دقیقه (و ۸ ساعتِ UTC)
NOW_MS = T_END + 60_000            # یک دقیقه داخلِ کندلِ باز
NOW = NOW_MS / 1000.0


def spot_rows():
    """۲۸۹ کندلِ بسته + یک کندلِ باز. هر کندل ۱۰۰ USDT؛ خرید ۶۰، ولی ۱۲ کندلِ آخر ۸۰."""
    out = []
    for i in range(290):
        t = T_END - (289 - i) * BAR
        if t == T_END:
            tbq = 1e9                               # کندلِ باز — نباید شمرده شود
        elif t >= T_END - H:
            tbq = 80.0
        else:
            tbq = 60.0
        c = 100.0 + i * 0.01
        out.append([t, str(c), str(c), str(c), str(c), "1.0", t + BAR - 1, "100.0", 10, "0.6", str(tbq), "0"])
    return out


def okx(rows):
    return {"code": "0", "msg": "", "data": rows}


def okx_taker_rows():
    # جدیدترین اول: [ts, sellVol, buyVol]؛ ردیفِ اول ناقص است (و بزرگ تا اگر شمرده شد معلوم شود)
    rows = [[str(T_END - BAR), "0", "1000000000"]]
    for i in range(2, 74):
        rows.append([str(T_END - i * BAR), "10", "30"])
    return okx(rows)


def okx_lsr_rows():
    rows = []
    for i in range(300):
        r = {0: "2.0", 12: "1.8", 288: "1.0"}.get(i, "1.5")
        rows.append([str(T_END - i * BAR), r])
    return okx(rows)


def okx_oi_rows():
    rows = []
    for i in range(300):
        oi = {0: "1100", 12: "1000", 288: "880"}.get(i, "950")
        vol = "1000000000" if i == 0 else "1"      # ردیفِ ۰ هنوز باز است (T_END + BAR > NOW)
        rows.append([str(T_END - i * BAR), oi, vol])
    return okx(rows)


def kc_contract():
    return {"code": "200000", "data": {
        "symbol": "XBTUSDTM", "openInterest": "2000", "multiplier": 0.001, "markPrice": 100.5, "indexPrice": 100.0,
        "fundingFeeRate": 0.0001, "predictedFundingFeeRate": None, "fundingRateGranularity": 28_800_000,
        "nextFundingRateDateTime": T_END + 8 * H}}


BIDS = [[99.99, 100], [99.6, 100], [99.2, 200], [98.0, 1000]]
ASKS = [[100.01, 50], [100.4, 50], [100.9, 100], [102.0, 1000]]


def kc_book():
    return {"code": "200000", "data": {"symbol": "XBTUSDTM", "sequence": 1, "bids": BIDS, "asks": ASKS,
                                       "ts": (NOW_MS - 500) * 1_000_000}}


FUND_BP = [2, 1, -1, 1, 1, 0, 1, 1, 1]


def kc_funding():
    rows = [{"symbol": "XBTUSDTM", "fundingRate": bp / 1e4, "timepoint": T_END - i * 8 * H} for i, bp in enumerate(FUND_BP)]
    rows += [{"symbol": "XBTUSDTM", "fundingRate": 0.009, "timepoint": T_END - i * 8 * H} for i in range(9, 12)]
    return {"code": "200000", "data": list(reversed(rows))}   # ترتیبِ برعکس: مرتب‌سازی باید درست کار کند


ROUTES = [
    ("/api/v3/klines", spot_rows),
    ("/rubik/stat/taker-volume", okx_taker_rows),
    ("/long-short-account-ratio", okx_lsr_rows),
    ("/open-interest-volume", okx_oi_rows),
    ("/api/v1/contracts/", kc_contract),
    ("/api/v1/level2/snapshot", kc_book),
    ("/api/v1/contract/funding-rates", kc_funding),
]


class Stub:
    """HTTPِ ساختگی: مسیر -> JSON. fail: پیشوندِ میزبانی که خطا بدهد."""

    def __init__(self, fail=None, override=None):
        self.calls = []
        self.fail = fail
        self.override = override or {}

    def __call__(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        if self.fail and url.startswith(self.fail):
            raise httpx.ConnectTimeout("stub timeout")
        for key, fn in ROUTES:
            if key in url:
                return self.override.get(key, fn)() if key in self.override else fn()
        raise AssertionError(f"مسیرِ پیش‌بینی‌نشده: {url}")


def reset():
    posdata._cache.update(ts=0.0, data=None)
    posdata._last_rec.clear()
    shutil.rmtree(posdata.LIVE_POS_DIR, ignore_errors=True)


class ParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snap = posdata.snapshot("BTCUSDT", get=Stub(), now=NOW)
        cls.f = cls.snap["fields"]

    def test_every_field_has_the_contract_shape(self):
        self.assertTrue(self.snap["display_only"])
        self.assertEqual(self.snap["note"], "فقط برای دیدنِ وضعیتِ بازار؛ ورودیِ مدل نیست و لبه‌اش آزموده نشده")
        self.assertEqual(self.snap["errors"], {})
        self.assertEqual(set(self.f), set(posdata.FIELDS))
        for name, fld in self.f.items():
            self.assertTrue({"value", "source", "as_of", "note"} <= set(fld), name)
            self.assertIsNotNone(fld["value"], name)
            self.assertNotIn("error", fld)
        self.assertNotIn("tradeable", json.dumps(self.snap))

    def test_spot_taker_windows_drop_the_open_bar(self):
        v1, v4, v24 = (self.f[k]["value"] for k in ("spot_taker_1h", "spot_taker_4h", "spot_taker_24h"))
        self.assertEqual((v1["buy"], v1["sell"], v1["bars"]), (960, 240, 12))
        self.assertEqual(v1["buy_pct"], 80.0)
        self.assertEqual(v1["imb"], 0.6)
        self.assertEqual((v4["buy"], v4["sell"], v4["bars"]), (3120, 1680, 48))
        self.assertEqual(v4["buy_pct"], 65.0)
        self.assertEqual((v24["buy"], v24["sell"], v24["bars"]), (17520, 11280, 288))
        self.assertEqual(self.f["spot_taker_1h"]["as_of"], T_END)

    def test_spot_cvd_series(self):
        v = self.f["spot_cvd_24h"]["value"]
        self.assertEqual(len(v["t"]), 288)
        self.assertEqual(len(v["cvd"]), 288)
        self.assertEqual(len(v["close"]), 288)
        self.assertEqual(v["t"][-1], T_END)
        self.assertEqual(v["cvd"][0], 20.0)
        self.assertEqual(v["last"], 17520 - 11280)
        self.assertEqual(v["cvd"][-1], v["last"])

    def test_okx_newest_taker_row_is_dropped(self):
        v1, v4 = self.f["okx_taker_1h"]["value"], self.f["okx_taker_4h"]["value"]
        self.assertEqual((v1["buy"], v1["sell"], v1["bars"]), (360, 120, 12))
        self.assertEqual(v1["buy_pct"], 75.0)
        self.assertEqual((v4["buy"], v4["sell"], v4["bars"]), (1440, 480, 48))
        self.assertEqual(self.f["okx_taker_1h"]["as_of"], T_END - BAR)

    def test_okx_open_bucket_is_dropped_too(self):
        # اگر ردیفِ یکی‌مانده‌به‌آخر هم هنوز باز باشد (ساعتِ عقب)، کنار می‌رود
        vals = posdata._src_okx_taker("BTCUSDT", Stub(), T_END - BAR - 1000)
        self.assertEqual(vals["okx_taker_1h"][1], T_END - 2 * BAR)
        self.assertEqual(vals["okx_taker_1h"][0]["bars"], 12)

    def test_okx_lsr_and_oi(self):
        lsr = self.f["okx_lsr"]["value"]
        self.assertEqual(lsr["ratio"], 2.0)
        self.assertAlmostEqual(lsr["chg_1h"], 0.2)
        self.assertAlmostEqual(lsr["chg_24h"], 1.0)
        self.assertAlmostEqual(lsr["long_pct"], 66.67)
        oi = self.f["okx_oi"]["value"]
        self.assertEqual(oi["oi_usd"], 1100)
        self.assertAlmostEqual(oi["chg_1h_pct"], 10.0)
        self.assertAlmostEqual(oi["chg_24h_pct"], 25.0)
        self.assertEqual(oi["vol_24h_usd"], 288)           # حجمِ ردیفِ باز شمرده نشد
        self.assertEqual(self.f["okx_oi"]["as_of"], T_END)

    def test_kucoin_contract_fields(self):
        oi = self.f["kc_open_interest"]["value"]
        self.assertEqual((oi["lots"], oi["coin"], oi["usdt"]), (2000.0, 2.0, 201.0))
        fu = self.f["kc_funding"]["value"]
        self.assertEqual(fu["current_bp"], 1.0)
        self.assertIsNone(fu["predicted_bp"])
        self.assertEqual(fu["interval_h"], 8.0)
        self.assertIn("پیش‌بینی", self.f["kc_funding"]["note"])
        self.assertEqual(self.f["kc_premium_bp"]["value"]["bp"], 50.0)

    def test_kucoin_funding_history_last_nine(self):
        v = self.f["kc_funding_hist"]["value"]
        self.assertEqual(v["n"], 9)
        self.assertEqual([r["bp"] for r in v["rows"]], [float(b) for b in FUND_BP])
        self.assertEqual(v["last_bp"], 2.0)
        self.assertAlmostEqual(v["mean_bp"], 7 / 9, places=3)
        self.assertAlmostEqual(v["persist"], 6 / 9, places=3)
        self.assertEqual(self.f["kc_funding_hist"]["as_of"], T_END)

    def test_book_as_of_from_nanoseconds(self):
        self.assertEqual(self.f["kc_book"]["as_of"], NOW_MS - 500)
        self.assertAlmostEqual(self.f["kc_book"]["value"]["imb_05"], 0.3315, places=3)


class BookMathTests(unittest.TestCase):
    def test_imbalance_within_bands(self):
        v = posdata.book_depth(BIDS, ASKS, 1.0)
        self.assertEqual(v["mid"], 100.0)
        bid05, ask05 = 99.99 * 100 + 99.6 * 100, 100.01 * 50 + 100.4 * 50
        bid1, ask1 = bid05 + 99.2 * 200, ask05 + 100.9 * 100
        self.assertAlmostEqual(v["bid_05"], bid05, delta=1)
        self.assertAlmostEqual(v["ask_05"], ask05, delta=1)
        self.assertAlmostEqual(v["imb_05"], (bid05 - ask05) / (bid05 + ask05), places=4)
        self.assertAlmostEqual(v["bid_1"], bid1, delta=1)
        self.assertAlmostEqual(v["ask_1"], ask1, delta=1)
        self.assertAlmostEqual(v["imb_1"], (bid1 - ask1) / (bid1 + ask1), places=4)
        self.assertTrue(v["covered_05"] and v["covered_1"])
        self.assertAlmostEqual(v["spread_bp"], 2.0, places=3)

    def test_multiplier_scales_notional_not_imbalance(self):
        a, b = posdata.book_depth(BIDS, ASKS, 1.0), posdata.book_depth(BIDS, ASKS, 0.001)
        self.assertAlmostEqual(b["bid_1"], round(a["bid_1"] * 0.001), delta=1)
        self.assertEqual(a["imb_1"], b["imb_1"])

    def test_shallow_book_is_flagged(self):
        v = posdata.book_depth([[99.9, 1], [99.7, 1]], [[100.1, 1], [100.3, 1]], 1.0)
        self.assertFalse(v["covered_05"])
        self.assertFalse(v["covered_1"])

    def test_empty_side_is_an_error(self):
        with self.assertRaises(RuntimeError):
            posdata.book_depth([], ASKS, 1.0)


class FailureIsolationTests(unittest.TestCase):
    def setUp(self):
        reset()

    def test_okx_down_leaves_other_sources_intact(self):
        snap = posdata.snapshot("ETHUSDT", get=Stub(fail=posdata.OKX), now=NOW)
        self.assertEqual(set(snap["errors"]), {"okx_taker", "okx_lsr", "okx_oi"})
        for name, fld in snap["fields"].items():
            if name.startswith("okx_"):
                self.assertIsNone(fld["value"], name)
                self.assertIn("ConnectTimeout", fld["error"])
            else:
                self.assertIsNotNone(fld["value"], name)
                self.assertNotIn("error", fld)

    def test_okx_error_code_fails_only_that_source(self):
        stub = Stub(override={"/long-short-account-ratio": lambda: {"code": "50011", "msg": "Too Many Requests", "data": []}})
        snap = posdata.snapshot("SOLUSDT", get=stub, now=NOW)
        self.assertEqual(list(snap["errors"]), ["okx_lsr"])
        self.assertIn("50011", snap["fields"]["okx_lsr"]["error"])
        self.assertIsNotNone(snap["fields"]["okx_oi"]["value"])

    def test_everything_down_never_raises(self):
        def boom(url, params=None):
            raise RuntimeError("network gone")
        out = posdata.snapshot_all(get=boom, now=NOW)
        self.assertEqual(list(out), list(posdata.watchlist.SYMBOLS))
        for sym, snap in out.items():
            self.assertTrue(snap["display_only"])
            self.assertEqual(set(snap["errors"]), set(posdata.SOURCES))
            self.assertTrue(all(f["value"] is None for f in snap["fields"].values()))
        self.assertEqual(posdata.record(out, now=NOW), [])   # چیزی برای ثبت نیست

    def test_unknown_symbol_does_not_raise(self):
        snap = posdata.snapshot("DOGEUSDT", get=Stub(), now=NOW)
        self.assertTrue(snap["display_only"])
        self.assertTrue(all(f["value"] is None for f in snap["fields"].values()))


class CacheTests(unittest.TestCase):
    def setUp(self):
        reset()

    def test_sixty_second_cache(self):
        stub = Stub()
        first = posdata.snapshot_all(get=stub, now=NOW)
        n = len(stub.calls)
        self.assertEqual(n, 7 * len(posdata.watchlist.SYMBOLS))
        self.assertEqual(set(first), set(posdata.watchlist.SYMBOLS))
        posdata.snapshot_all(get=stub, now=NOW + 59)
        self.assertEqual(len(stub.calls), n)
        posdata.snapshot_all(get=stub, now=NOW + 61)
        self.assertEqual(len(stub.calls), 2 * n)
        posdata.snapshot_all(get=stub, now=NOW + 62, force=True)
        self.assertEqual(len(stub.calls), 3 * n)
        age, data = posdata.peek(now=NOW + 70)
        self.assertEqual(age, 8.0)
        self.assertEqual(set(data), set(posdata.watchlist.SYMBOLS))
        self.assertEqual(len(stub.calls), 3 * n)                # peek به شبکه نمی‌رود

    def test_peek_before_any_fetch(self):
        self.assertEqual(posdata.peek(now=NOW), (None, None))


class HostTests(unittest.TestCase):
    def test_only_allowed_hosts_are_requested(self):
        stub = Stub()
        posdata.snapshot("TRXUSDT", get=stub, now=NOW)
        for url, _ in stub.calls:
            self.assertTrue(url.startswith(tuple(h + "/" for h in posdata.ALLOWED_HOSTS)), url)
            for bad in ("fapi", "/futures/data", "bybit", "api.binance.com", "www.binance.com"):
                self.assertNotIn(bad, url)
        self.assertEqual(posdata.ALLOWED_HOSTS, ("https://data-api.binance.vision", "https://www.okx.com",
                                                 "https://api-futures.kucoin.com"))

    def test_http_get_refuses_other_hosts_before_any_request(self):
        client = mock.Mock()
        client.get.side_effect = AssertionError("نباید درخواستی برود")
        with mock.patch.object(posdata, "_client", client):
            for url in ("https://fapi.binance.com/fapi/v1/klines", "https://www.binance.com/fapi/v1/klines",
                        "https://api.bybit.com/v5/market/tickers", "https://data-api.binance.vision.evil.com/x"):
                with self.assertRaises(ValueError):
                    posdata._http_get(url)
        client.get.assert_not_called()

    def test_okx_rate_limiter(self):
        clock = [0.0]
        sleeps = []

        def sleep(s):
            sleeps.append(s)
            clock[0] += s
        lim = posdata._Limiter(5, 2.2, clock=lambda: clock[0], sleep=sleep)
        for _ in range(5):
            lim.acquire()
        self.assertEqual(sleeps, [])
        lim.acquire()
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 2.21, places=2)
        lim2 = posdata._Limiter(1, 2.2, clock=lambda: clock[0], sleep=sleep)
        lim2.acquire()
        with self.assertRaises(RuntimeError):
            lim2.acquire(max_wait=0.5)


class RecordTests(unittest.TestCase):
    def setUp(self):
        reset()
        self.snaps = posdata.snapshot_all(get=Stub(), now=NOW)

    def tearDown(self):
        reset()

    def _lines(self, sym):
        with open(os.path.join(posdata.LIVE_POS_DIR, f"{sym}.jsonl"), encoding="utf-8") as f:
            return [json.loads(x) for x in f]

    def test_hermetic_path(self):
        self.assertTrue(os.path.normcase(posdata.LIVE_POS_DIR).startswith(os.path.normcase(_hermetic.DATA_DIR)))
        self.assertEqual(posdata.LIVE_POS_DIR, paths.data("hist_research", "live_pos"))
        real = os.path.normcase(os.path.join(ROOT, "bot", "data"))
        self.assertFalse(os.path.normcase(posdata.LIVE_POS_DIR).startswith(real))

    def test_one_line_per_coin_at_most_every_five_minutes(self):
        syms = list(posdata.watchlist.SYMBOLS)
        self.assertEqual(posdata.record(self.snaps, now=NOW), syms)
        self.assertEqual(posdata.record(self.snaps, now=NOW + 100), [])
        self.assertEqual(posdata.record(self.snaps, now=NOW + 301), syms)
        for s in syms:
            self.assertEqual(len(self._lines(s)), 2)
        posdata._last_rec.clear()                              # مثلِ راه‌اندازیِ دوباره: از دمِ فایل خوانده می‌شود
        self.assertEqual(posdata.record(self.snaps, now=NOW + 400), [])
        self.assertEqual(posdata.record(self.snaps["BTCUSDT"], now=NOW + 602), ["BTCUSDT"])
        self.assertEqual(len(self._lines("BTCUSDT")), 3)
        self.assertEqual(len(self._lines("ETHUSDT")), 2)

    def test_line_is_compact_and_marked_display_only(self):
        posdata.record(self.snaps, now=NOW)
        row = self._lines("BTCUSDT")[0]
        self.assertEqual((row["sym"], row["ts"], row["v"], row["display_only"]), ("BTCUSDT", NOW_MS, 1, True))
        self.assertEqual(set(row["fields"]), set(posdata.FIELDS))
        cvd = row["fields"]["spot_cvd_24h"]["value"]
        self.assertEqual(set(cvd), {"last", "chg_1h"})              # سریِ ۲۸۸تایی ذخیره نمی‌شود
        self.assertEqual(cvd["chg_1h"], 960 - 240)
        self.assertNotIn("rows", row["fields"]["kc_funding_hist"]["value"])
        self.assertEqual(row["fields"]["kc_book"]["value"]["imb_05"], self.snaps["BTCUSDT"]["fields"]["kc_book"]["value"]["imb_05"])

    def test_bad_symbol_is_never_a_path(self):
        bad = dict(self.snaps["BTCUSDT"], symbol="../../evil")
        self.assertEqual(posdata.record([bad], now=NOW), [])
        self.assertFalse(os.path.exists(posdata.LIVE_POS_DIR) and os.listdir(posdata.LIVE_POS_DIR))


if __name__ == "__main__":
    unittest.main()
