# -*- coding: utf-8 -*-
"""سیگنالِ v3 در main: ‎/api/signals، حجم با کفِ یک لات، سیاستِ نمایش در ‎/api/decisions و دورِ پس‌زمینه.

همه چیز stub است: هیچ درخواستِ شبکه و هیچ فایلِ bot/data واقعی.
"""
import inspect
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

from fastapi.testclient import TestClient  # noqa: E402

import decision  # noqa: E402
import kucoin_desk  # noqa: E402
import main  # noqa: E402
import market  # noqa: E402
import signal_v3  # noqa: E402

SPEC = {"BTCUSDT": {"contract": "XBTUSDTM", "multiplier": 0.001, "tick": 0.1, "lot": 1.0, "mark": 60000.0,
                    "maker": 0.0002, "taker": 0.0006, "mmr": 0.004, "max_leverage": 125.0, "funding_rate": 0.0001}}


class SignalsEndpointTests(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(main.app)               # بی‌context manager ⇒ رویدادِ startup اجرا نمی‌شود

    def test_endpoint_is_sync_and_returns_the_view(self):
        self.assertFalse(inspect.iscoroutinefunction(main.signals_view))
        fake = {"version": 1, "signals": [], "actionable": True, "tfs": {}, "cells": {}}
        with mock.patch.object(signal_v3, "view", return_value=fake) as v:
            r = self.c.get("/api/signals")
            self.c.get("/api/signals?force=true")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), fake)
        self.assertEqual([c.kwargs for c in v.call_args_list], [{"force": False}, {"force": True}])

    def test_no_data_is_a_persian_503_never_a_guess(self):
        with mock.patch.object(signal_v3, "view", side_effect=RuntimeError("down")):
            r = self.c.get("/api/signals")
        self.assertEqual(r.status_code, 503)
        self.assertIn("سیگنال", r.json()["detail"])

    def test_decisions_payload_states_the_display_policy(self):
        stub = {"generated_at": 1, "cells": {}}
        with mock.patch.object(decision, "refresh", return_value=dict(stub)), \
                mock.patch.object(main, "_decision_lookup", return_value=lambda s, tf: {}):
            body = self.c.get("/api/decisions").json()
        self.assertEqual(body["signal_policy"]["signal_tfs"], ["4h", "1d"])
        self.assertEqual(body["signal_policy"]["status_only_tfs"], ["5m", "15m", "1h"])
        self.assertEqual(body["signal_policy"]["source"], "/api/signals")


class SizeTests(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(main.app)
        p = mock.patch.object(kucoin_desk, "contracts", return_value=SPEC)
        p.start()
        self.addCleanup(p.stop)

    def _size(self, **kw):
        body = {"sym": "BTCUSDT", "balance": 1000.0, "risk_pct": 1.0, "entry": 60000.0, "stop": 58800.0}
        body.update(kw)
        return self.c.post("/api/signals/size", json=body)

    def test_size_from_risk(self):
        r = self._size()
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        # ریسک ۱۰ دلار ÷ (۲٪ فاصله + ۰٫۱۲٪ کارمزد) = ۴۷۱٫۷ دلار ⇒ ۷ قراردادِ ۶۰ دلاری
        self.assertEqual((d["contracts"], d["notional"], d["too_small"]), (7, 420.0, False))
        self.assertAlmostEqual(d["loss_at_stop"], 420 * 0.02 + 420 * 0.0012, places=2)
        self.assertLessEqual(d["loss_at_stop_pct"], 1.0)
        self.assertEqual(d["leverage"], 1.0)                       # بی‌نیاز به اهرم
        self.assertEqual(d["contract"], "XBTUSDTM")
        self.assertEqual(d["min"]["contracts"], 1)

    def test_small_balance_gets_the_one_lot_floor_and_its_real_risk(self):
        d = self._size(balance=100.0).json()
        # ریسک ۱ دلار ⇒ ۴۷ دلار ⇒ کمتر از یک قراردادِ ۶۰ دلاری
        self.assertEqual((d["contracts"], d["too_small"]), (0, True))
        self.assertEqual((d["min"]["contracts"], d["min"]["notional"]), (1, 60.0))
        self.assertAlmostEqual(d["min"]["loss_at_stop"], 60 * (0.02 + 0.0012), places=2)
        self.assertAlmostEqual(d["min"]["loss_at_stop_pct"], 1.272, places=3)   # بیش از ۱٪ خواسته‌شده
        self.assertEqual(d["min"]["min_leverage"], 1.0)

    def test_short_side_and_bad_input(self):
        d = self._size(stop=61200.0).json()
        self.assertEqual(d["side"], "short")
        self.assertEqual(self._size(balance=0).status_code, 400)
        self.assertEqual(self._size(stop=60000.0).status_code, 400)
        self.assertEqual(self._size(sym="DOGEUSDT").status_code, 404)

    def test_min_position_helper(self):
        m = kucoin_desk.min_position(50.0, 60000.0, 58800.0, SPEC["BTCUSDT"])
        self.assertEqual((m["contracts"], m["notional"], m["min_leverage"]), (1, 60.0, 1.2))
        with self.assertRaises(ValueError):
            kucoin_desk.min_position(50.0, 100.0, 100.0, SPEC["BTCUSDT"])


class BackgroundTests(unittest.TestCase):
    def test_tick_counts_signals_and_swallows_errors(self):
        with mock.patch.object(signal_v3, "snapshot", return_value={"signals": [1, 2, 3]}):
            self.assertEqual(main._signals_tick(), 3)
        with mock.patch.object(signal_v3, "snapshot", side_effect=RuntimeError("no network")):
            self.assertIsNone(main._signals_tick())

    def test_decision_cycle_does_not_touch_the_signal_snapshot(self):
        # دفترِ قاعدهٔ اصلی و آزمونِ گونه‌ها نباید معطلِ دادهٔ بلندِ v3 شوند: دو حلقهٔ جدا
        with mock.patch.object(signal_v3, "snapshot") as snap, \
                mock.patch.object(main, "get_analysis", lambda s, tf, max_age=None: {"symbol": s, "tf": tf, "error": "x"}), \
                mock.patch.dict(main._decision_state, {"last_boundary": None, "retry": {}}), \
                mock.patch.object(decision, "resolve", return_value=0), \
                mock.patch.object(decision, "scorecards", return_value={}), \
                mock.patch.object(main.candidates, "resolve", return_value=0):
            main._decision_cycle()
        snap.assert_not_called()

    def test_loader_asks_only_binance_spot_for_enough_closed_bars(self):
        calls = []
        with mock.patch.object(market, "get_klines_long", lambda s, tf, bars: calls.append((s, tf, bars)) or {"t": []}):
            signal_v3.market_loader("BTCUSDT", "4h", now_ms=signal_v3.T0_MS + 100 * 4 * 3_600_000)
        self.assertEqual(calls, [("BTCUSDT", "4h", 2500)])


if __name__ == "__main__":
    unittest.main()
