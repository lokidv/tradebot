# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ فاز ۱f — ترمزهای ریسک نباید با ری‌استارت پاک شوند.

بدترین سناریو: ربات حدِ ضررِ روزانه را می‌زند، برنامه ری‌استارت می‌شود، و چون
``_risk_off_day`` فقط در حافظه بود، همان روز دوباره شروع به معامله می‌کند.
"""
import os
import sys
import tempfile
import time
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import autotrader  # noqa: E402
import fill_quality  # noqa: E402
import journal  # noqa: E402


class _JournalMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old_journal = journal.PATH
        self._old_fq = fill_quality.PATH
        journal.PATH = os.path.join(self.tmp.name, "journal.jsonl")
        fill_quality.PATH = os.path.join(self.tmp.name, "fill_quality.json")
        journal._reset_cache()
        self._reset_bot()

    def tearDown(self):
        journal.PATH = self._old_journal
        fill_quality.PATH = self._old_fq
        journal._reset_cache()
        self._reset_bot()
        self.tmp.cleanup()

    @staticmethod
    def _reset_bot():
        autotrader._pending.clear()
        autotrader._cooldown.clear()
        autotrader._storm_until = 0.0
        autotrader._risk_off_day = None


class AppendAndReplayTests(_JournalMixin, unittest.TestCase):
    def test_sequence_numbers_are_strictly_increasing(self):
        seqs = [journal.append(journal.STORM, until=i)["seq"] for i in range(5)]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), 5)

    def test_sequence_continues_across_a_simulated_restart(self):
        journal.append(journal.STORM, until=1)
        journal.append(journal.STORM, until=2)
        journal._reset_cache()                       # مثل بالا آمدنِ دوبارهٔ پروسه
        self.assertEqual(journal.append(journal.STORM, until=3)["seq"], 3)

    def test_half_written_line_does_not_break_the_ledger(self):
        journal.append(journal.STORM, until=1)
        with open(journal.PATH, "a", encoding="utf-8") as f:
            f.write('{"seq": 2, "kind": "storm", "unti')      # قطعِ برق
        journal.append(journal.STORM, until=3)
        self.assertEqual(len(journal.events()), 2)

    def test_sequence_is_read_from_the_tail_of_a_large_journal(self):
        with open(journal.PATH, "w", encoding="utf-8") as f:
            for i in range(1, 3001):                      # ≈۱۵۰KB، بزرگ‌تر از TAIL_BYTES
                f.write(f'{{"seq": {i}, "ts": 0, "kind": "storm", "until": 0, "pad": "{"x" * 20}"}}\n')
        self.assertGreater(os.path.getsize(journal.PATH), journal.TAIL_BYTES)
        self.assertEqual(journal.append(journal.STORM, until=1)["seq"], 3001)


class QuarantineTests(_JournalMixin, unittest.TestCase):
    """ژورنال بازنویسی نمی‌شود؛ رویدادِ اشتباه با اصلاحیه باطل می‌شود و خطِ اصلی می‌ماند."""

    def test_quarantined_events_vanish_from_queries_but_stay_on_disk(self):
        journal.append(journal.GATE_CHANGE, reason="real")
        bad = journal.append(journal.GATE_CHANGE, reason="from a test")
        journal.quarantine([bad], "written by the test suite")
        self.assertEqual([e["reason"] for e in journal.events(kinds=[journal.GATE_CHANGE])], ["real"])
        with open(journal.PATH, encoding="utf-8") as f:
            self.assertIn("from a test", f.read())
        self.assertEqual(journal.summary()["quarantined"], 1)

    def test_a_quarantined_brake_is_not_restored_on_restart(self):
        ev = journal.append(journal.STORM, until=time.time() + 3600)
        journal.quarantine([ev], "test")
        self.assertEqual(journal.replay()["storm_until"], 0.0)

    def test_a_shared_seq_is_told_apart_by_its_timestamp(self):
        """پیش از قفلِ بین‌پروسه‌ای دو رویداد seq ۱۱۱ گرفتند؛ باطل‌کردنِ یکی نباید دیگری را ببرد."""
        with open(journal.PATH, "w", encoding="utf-8") as f:
            f.write('{"seq": 7, "ts": 100.5, "kind": "gate_change", "reason": "real"}\n')
            f.write('{"seq": 7, "ts": 200.25, "kind": "gate_change", "reason": "test"}\n')
        journal.quarantine([{"seq": 7, "ts": 200.25}], "test")
        self.assertEqual([e["reason"] for e in journal.events(kinds=[journal.GATE_CHANGE])], ["real"])


class CrossProcessSequenceTests(_JournalMixin, unittest.TestCase):
    """برنامه و پروسهٔ بازسازی هم‌زمان در ژورنال می‌نویسند؛ هر کدام شمارندهٔ خودش
    را داشت و seq ۱۱۱ دو بار نوشته شد. seq باید بین پروسه‌ها یکتا بماند."""

    def test_two_processes_appending_at_once_never_share_a_seq(self):
        import subprocess
        child = ("import sys; sys.path.insert(0, sys.argv[1]); import journal; journal.PATH = sys.argv[2]\n"
                 "for i in range(40): journal.append('storm', until=0, who=sys.argv[3], i=i)\n")
        bot = os.path.join(ROOT, "bot")
        procs = [subprocess.Popen([sys.executable, "-c", child, bot, journal.PATH, who])
                 for who in ("app", "rebuild")]
        for p in procs:
            self.assertEqual(p.wait(timeout=60), 0)
        seqs = [int(r["seq"]) for r in journal._read_raw()]
        self.assertEqual(len(seqs), 80)
        self.assertEqual(sorted(seqs), list(range(1, 81)), "seq تکراری یا جاافتاده")


class RiskBrakesSurviveRestartTests(_JournalMixin, unittest.TestCase):
    def test_daily_loss_halt_is_still_in_force_after_restart(self):
        today = time.strftime("%Y-%m-%d", time.gmtime())
        autotrader._set_risk_off(today)
        self._reset_bot()                            # ری‌استارت: حافظه پاک
        self.assertIsNone(autotrader._risk_off_day)
        autotrader.restore_state()
        self.assertEqual(autotrader._risk_off_day, today)

    def test_storm_brake_is_restored_with_its_remaining_time(self):
        until = time.time() + 300
        autotrader._set_storm(until)
        self._reset_bot()
        autotrader.restore_state()
        self.assertAlmostEqual(autotrader._storm_until, until, places=3)

    def test_expired_storm_does_not_come_back(self):
        autotrader._set_storm(time.time() - 60)
        self._reset_bot()
        autotrader.restore_state()
        self.assertLess(autotrader._storm_until, time.time())

    def test_cooldown_survives_and_expired_cooldown_does_not(self):
        autotrader._set_cooldown("BTCUSDT", time.time() + 600)
        autotrader._set_cooldown("ETHUSDT", time.time() - 600)
        self._reset_bot()
        autotrader.restore_state()
        self.assertIn("BTCUSDT", autotrader._cooldown)
        self.assertNotIn("ETHUSDT", autotrader._cooldown)

    def test_cooldown_only_ever_extends(self):
        autotrader._set_cooldown("BTCUSDT", time.time() + 900)
        autotrader._set_cooldown("BTCUSDT", time.time() + 10)
        self.assertGreater(autotrader._cooldown["BTCUSDT"], time.time() + 800)


class PendingOrdersSurviveRestartTests(_JournalMixin, unittest.TestCase):
    @staticmethod
    def _order(expires_in=600):
        return {"side": "long", "target": 100.0, "tf": "1h", "r_abs": 2.0,
                "tp_abs": 3.6, "size": 500.0, "tstop": 2400, "grade": "A",
                "score": 70, "cost": 0.11, "expires": time.time() + expires_in,
                "authority": "policy"}

    def test_live_order_is_restored(self):
        journal.append(journal.ORDER_PLACED, symbol="BTCUSDT", order=self._order())
        autotrader.restore_state()
        self.assertIn("BTCUSDT", autotrader._pending)
        self.assertEqual(autotrader._pending["BTCUSDT"]["side"], "long")

    def test_filled_order_is_not_resurrected(self):
        journal.append(journal.ORDER_PLACED, symbol="BTCUSDT", order=self._order())
        fill_quality.log_event("filled", symbol="BTCUSDT", tf="1h", side="long")
        autotrader.restore_state()
        self.assertNotIn("BTCUSDT", autotrader._pending)

    def test_cancelled_and_expired_orders_are_not_resurrected(self):
        journal.append(journal.ORDER_PLACED, symbol="ETHUSDT", order=self._order())
        journal.append(journal.ORDER_PLACED, symbol="SOLUSDT", order=self._order())
        fill_quality.log_event("cancelled", symbol="ETHUSDT", why="heat")
        fill_quality.log_event("expired", symbol="SOLUSDT")
        autotrader.restore_state()
        self.assertEqual(autotrader._pending, {})

    def test_order_whose_ttl_lapsed_while_offline_is_dropped(self):
        journal.append(journal.ORDER_PLACED, symbol="BTCUSDT",
                       order=self._order(expires_in=-10))
        autotrader.restore_state()
        self.assertNotIn("BTCUSDT", autotrader._pending)

    def test_clear_pending_records_every_cancellation(self):
        autotrader._pending["BTCUSDT"] = self._order()
        autotrader._pending["ETHUSDT"] = self._order()
        journal.append(journal.ORDER_PLACED, symbol="BTCUSDT",
                       order=autotrader._pending["BTCUSDT"])
        journal.append(journal.ORDER_PLACED, symbol="ETHUSDT",
                       order=autotrader._pending["ETHUSDT"])
        autotrader._clear_pending("bot_disabled")
        self.assertEqual(autotrader._pending, {})
        self._reset_bot()
        autotrader.restore_state()
        self.assertEqual(autotrader._pending, {})       # لغو شده‌اند، برنمی‌گردند

    def test_stale_events_beyond_the_replay_window_are_ignored(self):
        ev = journal.append(journal.ORDER_PLACED, symbol="BTCUSDT", order=self._order())
        rows = journal._read_raw()
        rows[0]["ts"] = time.time() - 30 * 86400
        with open(journal.PATH, "w", encoding="utf-8") as f:
            import json as _json
            f.write(_json.dumps(rows[0]) + "\n")
        self.assertEqual(journal.replay()["pending"], {})
        self.assertEqual(ev["kind"], journal.ORDER_PLACED)


if __name__ == "__main__":
    unittest.main()
