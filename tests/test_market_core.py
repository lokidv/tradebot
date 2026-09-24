# -*- coding: utf-8 -*-
"""لایهٔ بازار برای هستهٔ v2: فیلدهای جریانِ خرید/فروش (qv, n, tbv)، NaN در پشتیبانِ OKX، کندلِ بلندِ
صفحه‌بندی‌شدهٔ اسپات، فاندینگِ کوکوین، OIِ OKX با instId و بی‌کشِ «خالی» پس از ۴xx، و تایم‌فریمِ ۵m.

همه بدونِ شبکه: شبکه در سطحِ ``market._get_json`` / ``market._binance_json`` جایگزین می‌شود.
"""
import math
import os
import sys
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import httpx  # noqa: E402
import numpy as np  # noqa: E402

import market  # noqa: E402

H = 3_600_000


class _Clock:
    def __init__(self, t):
        self.t = float(t)

    def time(self):
        return self.t


def _kl_row(t, step):
    """ردیفِ ۱۲فیلدیِ بایننس با مقادیرِ متمایز: v=10، qv=15، n=7، tbv=4 + (t/step)%3."""
    k = (t // step) % 3
    return [t, "1", "2", "0.5", "1.5", "10", t + step - 1, "15", "7", str(4 + k), "6", "0"]


class _Exchange:
    """بایننسِ ساختگی: کندل‌های step تا کندلِ باز در لحظهٔ now()؛ limit و endTime مثلِ /api/v3/klines."""

    def __init__(self, step, n_total, now=None):
        self.step, self.n_total, self.now = step, n_total, now or (lambda: time.time())
        self.calls = []

    def rows(self, limit, end=None):
        step = self.step
        last = int(self.now() * 1000) // step * step               # کندلِ در حالِ شکل‌گیری
        first = last - (self.n_total - 1) * step
        if end is not None:
            last = min(last, end // step * step)
        start = max(first, last - (int(limit) - 1) * step)
        return [_kl_row(t, step) for t in range(start, last + 1, step)]

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        return self.rows(params["limit"], params.get("endTime"))


def _status_error(code, url="https://x/y"):
    req = httpx.Request("GET", url)
    return httpx.HTTPStatusError(str(code), request=req, response=httpx.Response(code, request=req))


class _Base(unittest.TestCase):
    def setUp(self):
        market._host_dead.clear()
        market._neg_until.clear()
        with market._lock:
            market._long_cache.clear()
            market._kfund_cache.clear()

    def tearDown(self):
        self.setUp()
        with market._lock:
            for k in [k for k in market._kline_cache if k[0].startswith("ZZ")]:
                market._kline_cache.pop(k, None)


class TimeframeMapsTests(unittest.TestCase):
    def test_5m_is_in_every_timeframe_map(self):
        maps = {"TF_BINANCE": market.TF_BINANCE, "TF_OKX": market.TF_OKX, "TF_MINUTES": market.TF_MINUTES,
                "KLINE_TTL": market.KLINE_TTL, "TF_OKX_OI": market.TF_OKX_OI}
        for name, m in maps.items():
            self.assertEqual(set(m), {"5m", "15m", "1h", "4h", "1d"}, name)
        self.assertEqual(market.TF_BINANCE["5m"], "5m")
        self.assertEqual(market.TF_OKX["5m"], "5m")
        self.assertEqual(market.TF_MINUTES["5m"], 5)
        self.assertEqual(market.KLINE_TTL["5m"], 30)

    def test_no_route_around_the_binance_location_block(self):
        with open(market.__file__, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("www.binance.com", src)
        self.assertEqual(market._host_of(market.SPOT_DATA), "data-api.binance.vision")


class KlineFieldsTests(_Base):
    def test_binance_klines_keep_qv_n_tbv_and_drop_the_open_bar(self):
        step = H
        ex = _Exchange(step, 2000)
        with mock.patch.object(market, "_binance_json", side_effect=lambda p, params=None: ex.rows(params["limit"])):
            kl = market.get_klines("ZZFLOWUSDT", "1h", 300)
        self.assertEqual(kl["src"], "binance")
        self.assertEqual(len(kl["t"]), 300)
        for k in market.KLINE_KEYS:
            self.assertEqual(len(kl[k]), 300, k)
        self.assertLessEqual(kl["t"][-1] + step, time.time() * 1000, "کندلِ باز حذف نشد")
        self.assertEqual(set(kl["qv"]), {15.0})
        self.assertEqual(set(kl["n"]), {7.0})
        self.assertEqual(kl["tbv"][-1], 4.0 + (kl["t"][-1] // step) % 3)       # فیلدِ ۹، نه ۱۰
        self.assertEqual(kl["v"][0], 10.0)

    def test_okx_fallback_has_nan_flow_never_a_fake_half(self):
        now = int(time.time() * 1000) // H * H
        # OKX جدیدترین اول؛ ردیفِ اول کندلِ تأییدنشده (confirm=0)
        data = [[str(now - i * H), "1", "2", "0.5", "1.5", "10", "10", "15", "0" if i == 0 else "1"]
                for i in range(300)]

        def okx(url, params=None):
            self.assertIn("okx.com", url)
            return {"data": data}

        with mock.patch.object(market, "_binance_json", side_effect=httpx.ConnectError("down")), \
                mock.patch.object(market, "_get_json", side_effect=okx):
            kl = market.get_klines("ZZOKXUSDT", "1h", 299)
        self.assertEqual(kl["src"], "okx")
        self.assertEqual(len(kl["c"]), 299)
        self.assertEqual(kl["t"][-1], now - H)
        for k in ("qv", "n", "tbv"):
            self.assertEqual(len(kl[k]), 299)
            self.assertTrue(all(math.isnan(x) for x in kl[k]), k)

    def test_short_binance_rows_get_nan_not_zero(self):
        r = market._binance_row([1, "1", "2", "0.5", "1.5", "10"])
        self.assertEqual(r[:6], (1, 1.0, 2.0, 0.5, 1.5, 10.0))
        self.assertTrue(all(math.isnan(x) for x in r[6:]))


class KlinesLongTests(_Base):
    def test_pages_back_until_enough_closed_bars(self):
        step = 300_000
        clock = _Clock(1_789_999_800 + 10)                  # ۱۰ ثانیه پس از یک مرزِ ۵ دقیقه
        ex = _Exchange(step, 50_000, now=clock.time)
        with mock.patch.object(market, "time", clock), mock.patch.object(market, "_get_json", side_effect=ex.get_json):
            kl = market.get_klines_long("ZZLONGUSDT", "5m", 2500)
        self.assertEqual(len(ex.calls), 3)
        for url, params in ex.calls:
            self.assertEqual(url, "https://data-api.binance.vision/api/v3/klines")
            self.assertEqual(params["limit"], 1000)
            self.assertEqual(params["interval"], "5m")
        self.assertNotIn("endTime", ex.calls[0][1])
        self.assertEqual(ex.calls[1][1]["endTime"], int(kl["t"][-999]) - 1)
        t = kl["t"]
        self.assertGreaterEqual(len(t), 2500)
        self.assertEqual(t.dtype, np.int64)
        self.assertTrue((np.diff(t) == step).all(), "صفحه‌ها پیوسته و بی‌تکرار نیستند")
        self.assertEqual(int(t[-1]) + step, int(clock.t * 1000) // step * step, "آخرین کندلِ بسته‌شده")
        self.assertEqual(kl["src"], "binance-spot")
        self.assertTrue(np.isfinite(kl["tbv"]).all())
        self.assertEqual(float(kl["qv"][0]), 15.0)

    def test_cached_within_the_bar_then_only_the_tail_is_fetched(self):
        step = 300_000
        clock = _Clock(1_789_999_800 + 10)
        ex = _Exchange(step, 50_000, now=clock.time)
        with mock.patch.object(market, "time", clock), mock.patch.object(market, "_get_json", side_effect=ex.get_json):
            first = market.get_klines_long("ZZTAILUSDT", "5m", 1500)
            n_full = len(ex.calls)
            clock.t += 15                                     # همان کندل، زیرِ TTL
            self.assertIs(market.get_klines_long("ZZTAILUSDT", "5m", 1500), first)
            self.assertEqual(len(ex.calls), n_full)
            clock.t += 600                                    # دو کندلِ تازه بسته شد
            second = market.get_klines_long("ZZTAILUSDT", "5m", 1500)
        self.assertEqual(len(ex.calls), n_full + 1)
        self.assertLessEqual(ex.calls[-1][1]["limit"], 10)
        self.assertNotIn("endTime", ex.calls[-1][1])
        self.assertEqual(len(second["t"]), len(first["t"]))    # پنجرهٔ غلتان
        self.assertEqual(int(second["t"][-1]), int(first["t"][-1]) + 2 * step)
        self.assertTrue((np.diff(second["t"]) == step).all())

    def test_concurrent_long_fetches_are_one_network_fetch(self):
        ex = _Exchange(H, 5000)

        def slow(url, params=None):
            time.sleep(0.2)
            return ex.get_json(url, params)

        res, errs = [], []

        def run():
            try:
                res.append(market.get_klines_long("ZZFLIGHTUSDT", "1h", 1500))
            except Exception as e:  # noqa: BLE001
                errs.append(e)

        with mock.patch.object(market, "_get_json", side_effect=slow):
            th = [threading.Thread(target=run) for _ in range(5)]
            for x in th:
                x.start()
            for x in th:
                x.join()
        self.assertEqual(errs, [])
        self.assertEqual(len(ex.calls), 2)                    # ۱۵۰۰ کندل = دو صفحه، فقط یک‌بار
        self.assertTrue(all(r is res[0] for r in res))

    def test_short_listing_returns_what_exists(self):
        ex = _Exchange(H, 700)
        with mock.patch.object(market, "_get_json", side_effect=ex.get_json):
            kl = market.get_klines_long("ZZNEWUSDT", "1h", 2000)
        self.assertEqual(len(ex.calls), 1)
        self.assertEqual(len(kl["t"]), 699)


class KucoinFundingTests(_Base):
    @staticmethod
    def _venue(ts, rates):
        calls = []

        def get(url, params=None):
            calls.append(dict(params))
            assert url == "https://api-futures.kucoin.com/api/v1/contract/funding-rates", url
            sel = sorted(((t, r) for t, r in zip(ts, rates) if params["from"] <= t <= params["to"]), reverse=True)
            return {"code": "200000", "data": [{"symbol": params["symbol"], "fundingRate": r, "timepoint": t}
                                               for t, r in sel[:100]]}
        return get, calls

    def test_contract_names(self):
        self.assertEqual(market.kucoin_contract("BTCUSDT"), "XBTUSDTM")
        self.assertEqual(market.kucoin_contract("TRXUSDT"), "TRXUSDTM")

    def test_pages_cover_the_whole_range_without_duplicates(self):
        ts = [1_640_995_200_000 + 4 * H + i * 8 * H for i in range(537)]
        get, calls = self._venue(ts, [1e-4 * (i % 5) for i in range(537)])
        rows = market.kucoin_funding_pages("XBTUSDTM", ts[0], ts[-1] + H, get=get)
        self.assertEqual([r[0] for r in rows], ts)
        self.assertEqual(len(calls), 6)                      # ۵ صفحهٔ پر + یک نیمه؛ to از from گذشت → پایان
        self.assertTrue(all(c["from"] == ts[0] for c in calls))

    def test_interval_is_inferred_from_spacing_and_survives_a_missing_settlement(self):
        t = np.array([0, 8, 16, 24, 40, 48, 52, 56]) * H      # ۴۰ = تسویهٔ غایب؛ ۵۲ = تغییرِ برنامه به ۴ ساعت
        self.assertEqual(market.infer_interval_h(t).tolist(), [8, 8, 8, 8, 8, 8, 4, 4])
        self.assertEqual(market.infer_interval_h(np.array([5 * H])).tolist(), [8.0])
        self.assertEqual(len(market.infer_interval_h(np.array([], dtype=np.int64))), 0)

    def test_live_funding_is_recent_settled_and_cached(self):
        last = int(time.time() * 1000) // (8 * H) * (8 * H)
        ts = [last - 8 * H * i for i in range(400)][::-1]
        get, calls = self._venue(ts, [1e-4] * 400)
        with mock.patch.object(market, "_get_json", side_effect=get):
            d = market.get_kucoin_funding("BTCUSDT")
            self.assertIs(market.get_kucoin_funding("BTCUSDT"), d)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["symbol"], "XBTUSDTM")
        self.assertGreaterEqual(len(d["t"]), 181)
        self.assertEqual(int(d["t"][-1]), last)
        self.assertTrue((np.diff(d["t"]) > 0).all())
        self.assertEqual(set(d["interval_h"].tolist()), {8.0})
        self.assertEqual(set(d) - {"src"}, {"t", "rate", "interval_h"})

    def test_logical_error_raises_and_is_not_cached(self):
        bad = {"code": "404000", "msg": "This contract does not exist."}
        with mock.patch.object(market, "_get_json", return_value=bad):
            with self.assertRaises(RuntimeError):
                market.get_kucoin_funding("ZZNOPEUSDT")
        self.assertNotIn("ZZNOPEUSDT", market._kfund_cache)


class OkxOpenInterestTests(_Base):
    def _path(self, sym, period="1h"):
        return os.path.join(market.HIST_DIR, f"oi_{sym}_{period}.json")

    def test_request_carries_inst_id_and_parses_coin_units(self):
        sym = "ZZOIUSDT"
        seen = []
        now = int(time.time() * 1000) // H * H
        data = [[str(now - i * H), str(1000 + i), str(10 + i), "1e6"] for i in range(60)]

        def okx(url, params=None):
            seen.append((url, dict(params)))
            return {"code": "0", "data": data}

        with mock.patch.object(market, "_fapi_json", side_effect=RuntimeError("451")), \
                mock.patch.object(market, "_get_json", side_effect=okx):
            rows = market.get_oi_history(sym, period="1h", limit=48)
        url, params = seen[0]
        self.assertTrue(url.endswith("/api/v5/rubik/stat/contracts/open-interest-history"))
        self.assertEqual(params["instId"], "ZZOI-USDT-SWAP")
        self.assertEqual(params["period"], "1H")
        self.assertNotIn("uly", params)
        self.assertEqual(len(rows), 48)
        self.assertEqual(rows[-1], [now, 10.0])                  # oiCcy (ارزِ پایه)، صعودی
        self.assertTrue(os.path.exists(self._path(sym)))
        os.remove(self._path(sym))

    def test_a_4xx_is_never_cached_on_disk(self):
        sym = "ZZOI4XXUSDT"
        n = []

        def bad(url, params=None):
            n.append(1)
            raise _status_error(400, url)

        with mock.patch.object(market, "_fapi_json", side_effect=_status_error(400)), \
                mock.patch.object(market, "_get_json", side_effect=bad):
            self.assertEqual(market.get_oi_history(sym), [])
            self.assertEqual(market.get_oi_history(sym), [])       # کشِ منفیِ حافظه، نه دیسک
        self.assertFalse(os.path.exists(self._path(sym)), "«[]» پس از ۴xx روی دیسک ماند")
        self.assertEqual(len(n), 1)
        self.assertLessEqual(market._neg_until[self._path(sym)] - time.time(), market.NEG_TTL)

    def test_a_stale_empty_file_from_the_old_bug_is_not_served(self):
        sym = "ZZOIOLDUSDT"
        os.makedirs(market.HIST_DIR, exist_ok=True)
        with open(self._path(sym), "w", encoding="utf-8") as f:
            f.write("[]")
        now = int(time.time() * 1000) // H * H
        data = [[str(now - i * H), "1", "5", "1"] for i in range(20)]
        with mock.patch.object(market, "_fapi_json", side_effect=RuntimeError("451")), \
                mock.patch.object(market, "_get_json", return_value={"code": "0", "data": data}):
            rows = market.get_oi_history(sym)
        self.assertEqual(len(rows), 20)
        os.remove(self._path(sym))

    def test_5m_period_maps_to_okx_5m(self):
        seen = []
        with mock.patch.object(market, "_fapi_json", side_effect=RuntimeError("451")), \
                mock.patch.object(market, "_get_json",
                                  side_effect=lambda u, p=None: seen.append(p) or {"code": "0", "data": []}):
            self.assertEqual(market.get_oi_history("ZZOI5USDT", period="5m"), [])
        self.assertEqual(seen[0]["period"], "5m")
        self.assertFalse(os.path.exists(self._path("ZZOI5USDT", "5m")))


if __name__ == "__main__":
    unittest.main()
