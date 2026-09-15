# -*- coding: utf-8 -*-
"""جمعیتِ تحلیل: دارایی‌های میخ‌شده (استیبل‌کوین) بازار نیستند.

فهرستِ نام‌ها عقب ماند: استیبل‌کوینِ تازهٔ «U» (1.0002) در جدول «ستاپ A» نشان می‌داد
و USDG واردِ جمعیتِ آموزش و breadth/RS شده بود.
"""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import market  # noqa: E402


def _ticker(sym, last, high, low, qv):
    return {"symbol": sym, "lastPrice": str(last), "highPrice": str(high), "lowPrice": str(low),
            "quoteVolume": str(qv), "priceChangePercent": "0.1"}


class PeggedAssetsTests(unittest.TestCase):
    def setUp(self):
        market._top_cache.update(ts=0.0, symbols=[], tickers={})

    def tearDown(self):
        market._top_cache.update(ts=0.0, symbols=[], tickers={})

    def test_an_unlisted_stablecoin_is_dropped_by_its_behaviour(self):
        rows = [
            _ticker("UUSDT", 1.0002, 1.0006, 0.9998, 9e9),          # نامش در STABLE_BASES نیست
            _ticker("USDGUSDT", 1.0008, 1.0010, 1.0005, 8e9),
            # آرام‌ترین روزِ واقعیِ BTC در ۲۹۹۹ روز: بازهٔ ۰٫۳۴٪ — نباید حذف شود
            _ticker("BTCUSDT", 60000, 60100, 59896, 7e9),
            _ticker("ETHUSDT", 3000, 3060, 2950, 6e9),
        ]
        with mock.patch.object(market, "_binance_json", return_value=rows):
            symbols, _t = market.get_top_symbols(10)
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT"])

    def test_the_threshold_sits_far_below_the_quietest_major(self):
        self.assertFalse(market._is_pegged(60000, 60100, 59896))   # ۰٫۳۴٪
        self.assertTrue(market._is_pegged(1.0002, 1.0006, 0.9998))  # ۰٫۰۸٪
        self.assertFalse(market._is_pegged(0, 1, 0))                # دادهٔ خراب = تصمیم نگیر


if __name__ == "__main__":
    unittest.main()
