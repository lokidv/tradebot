# -*- coding: utf-8 -*-
"""متا-گیت با کدِ دلیل، و قفلِ سفارش‌های صبور بینِ دو نخ."""
import os
import sys
import tempfile
import threading
import time
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import autotrader  # noqa: E402
import fill_quality  # noqa: E402
import journal  # noqa: E402
import meta_gate  # noqa: E402


class HardBlockCodeTests(unittest.TestCase):
    def test_crowded_funding_is_a_hard_block_by_code(self):
        res = meta_gate.evaluate("long", {"funding_z": 2.5, "funding_persist": 5})
        self.assertIn("funding_crowded", res["block_codes"])
        self.assertTrue(res["hard_block"])
        self.assertFalse(res["approve"])

    def test_soft_block_does_not_hard_reject_on_its_own(self):
        res = meta_gate.evaluate("long", {"btc_z": -0.9})
        self.assertIn("btc_momentum_against", res["block_codes"])
        self.assertFalse(res["hard_block"])

    def test_rewording_a_message_cannot_change_risk_behaviour(self):
        """قبلاً «شلوغ» در متن ⇒ بلاکِ سخت. حالا فقط کد مهم است."""
        self.assertNotIn("btc_momentum_against", meta_gate.HARD_BLOCKS)
        self.assertIn("breadth_against", meta_gate.HARD_BLOCKS)
        res = meta_gate.evaluate("long", {"breadth": -0.6})
        self.assertIn("breadth_against", res["block_codes"])
        self.assertTrue(res["hard_block"])

    def test_every_block_message_has_a_code(self):
        res = meta_gate.evaluate("long", {"funding_z": 3, "funding_persist": 4, "breadth": -0.8,
                                          "btc_z": -1.2, "oi_ok": True, "oi_z": 2.0,
                                          "adx_hysteresis": {"stable": False, "recent": ["trend", "range"]}})
        self.assertEqual(len(res["blocks"]), min(len(res["block_codes"]), 4))


class PendingLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = (journal.PATH, fill_quality.PATH)
        journal.PATH = os.path.join(self.tmp.name, "j.jsonl")
        fill_quality.PATH = os.path.join(self.tmp.name, "fq.json")
        journal._reset_cache()
        autotrader._pending.clear()

    def tearDown(self):
        journal.PATH, fill_quality.PATH = self._old
        journal._reset_cache()
        autotrader._pending.clear()
        self.tmp.cleanup()

    def test_clear_waits_for_an_in_flight_fill(self):
        """لغو نباید وسطِ پرشدنِ یک سفارش در نخِ دیگر اجرا شود."""
        autotrader._pending["BTCUSDT"] = {"side": "long", "tf": "1h", "expires": time.time() + 60}
        order_of_events = []
        entered = threading.Event()

        def filler():
            with autotrader._pending_lock:
                entered.set()
                order_of_events.append("fill_start")
                time.sleep(0.2)
                autotrader._pending.pop("BTCUSDT", None)
                order_of_events.append("fill_end")

        t = threading.Thread(target=filler)
        t.start()
        entered.wait(1)
        autotrader._clear_pending("test")
        order_of_events.append("cleared")
        t.join()
        self.assertEqual(order_of_events, ["fill_start", "fill_end", "cleared"])

    def test_lock_is_reentrant_for_clear_inside_processing(self):
        with autotrader._pending_lock:
            autotrader._clear_pending("nested")          # نباید بن‌بست شود
        self.assertEqual(autotrader._pending, {})


if __name__ == "__main__":
    unittest.main()
