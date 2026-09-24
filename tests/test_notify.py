# -*- coding: utf-8 -*-
"""اعلانِ تلگرام: بی‌تنظیمات هیچ چیزی فرستاده نمی‌شود؛ هر دوشنبه فقط یک‌بار."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import notify  # noqa: E402

MON = 1_790_553_600_000
SNAP = {"assets": {"BTCUSDT": {"current": {"in_market": True, "monday_ms": MON, "sunday_close": 81000.0,
                                           "change_pct": 4.4}}}}


class _Resp:
    status_code = 200


class NotifyTests(unittest.TestCase):
    def setUp(self):
        if os.path.exists(notify.CFG_PATH):
            os.remove(notify.CFG_PATH)
        self.sent = []

    def post(self, url, data):
        self.sent.append((url, data))
        return _Resp()

    def test_nothing_is_sent_without_user_configuration(self):
        self.assertFalse(notify.maybe_notify_monday(SNAP, post=self.post))
        self.assertEqual(self.sent, [])

    def test_each_monday_is_sent_once_and_the_token_is_never_exposed(self):
        notify.save_cfg({"enabled": True, "token": "123:SECRET", "chat_id": "42"})
        self.assertTrue(notify.maybe_notify_monday(SNAP, post=self.post))
        self.assertFalse(notify.maybe_notify_monday(SNAP, post=self.post))
        self.assertEqual(len(self.sent), 1)
        self.assertIn("BTC", self.sent[0][1]["text"])
        self.assertNotIn("SECRET", str(notify.public_cfg()))


if __name__ == "__main__":
    unittest.main()
