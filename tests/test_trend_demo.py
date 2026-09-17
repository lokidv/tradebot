# -*- coding: utf-8 -*-
"""«خرید در دمو» برای سیگنالِ روند: قیمتِ زنده، همان حدضررِ قاعده، بی‌هدف، اسپات — و خروجِ صادقانه.

دو معاملهٔ دموی ۲۰۲۶-۰۹-۱۵ سه ایراد را نشان دادند که این تست‌ها قفلشان می‌کنند:
کارت قیمتِ دیروز را نشان می‌داد (نه قیمتِ لحظهٔ خرید)؛ سیگنالی که حدضررش همان روز خورده بود زنده
می‌ماند؛ و دمو به‌جای قیمتِ حدضرر، با قیمتِ لحظه‌ای که برنامه دوباره باز شد بسته می‌شد.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import advisor  # noqa: E402
import explore  # noqa: E402
import paper  # noqa: E402
import trend  # noqa: E402

E_TS = 1_789_344_000_000          # ورودِ قاعده: 2026-09-14 00:00 UTC
ATR = 0.0597


def _row(sym="FILUSDT", category="join", stop=0.773, last=0.9356, days=1, in_position=True, entry_ts=E_TS,
         exits=()):
    return {"sym": sym, "category": category, "stop": stop, "last": last, "days_in": days, "atr": ATR,
            "in_position": in_position, "stop_distance_pct": (last - stop) / last * 100,
            "entry_ts": entry_ts if in_position else None,
            "rule_entry_ts": entry_ts if (in_position or category == "new") else None,
            "recent_exits": list(exits)}


def _bar(c, o=0.9355, l=None, t=E_TS + 86_400_000):
    return {"t": t, "o": o, "h": max(o, c), "l": min(o, c) if l is None else l, "c": c}


class _Base(unittest.TestCase):
    def setUp(self):
        for p in (paper.PATH, paper._ledger_path()):
            if os.path.exists(p):
                os.remove(p)
        paper.reset_wallet(10_000)
        trend._live_cache.clear()


class LiveViewTests(_Base):
    def test_the_card_carries_the_live_price_not_yesterdays_close(self):
        lv = trend.with_live(_row(), _bar(0.852, l=0.845))
        self.assertEqual(lv["live"], 0.852)
        self.assertAlmostEqual(lv["moved_from_open_pct"], (0.852 / 0.9355 - 1) * 100)
        self.assertAlmostEqual(lv["live_stop_distance_pct"], (0.852 - 0.773) / 0.852 * 100)
        self.assertTrue(lv["valid"])

    def test_a_cushion_under_one_and_a_half_atr_is_flagged_tight(self):
        self.assertTrue(trend.with_live(_row(), _bar(0.852, l=0.845))["tight"])          # ۱٫۳ ATR
        self.assertFalse(trend.with_live(_row(), _bar(0.93, l=0.925))["tight"])          # ۲٫۶ ATR

    def test_a_stop_touched_earlier_today_kills_the_signal_even_if_price_recovered(self):
        lv = trend.with_live(_row(), _bar(0.80, l=0.7557))
        self.assertFalse(lv["valid"])
        self.assertEqual(lv["invalid_reason"], "stop_breached_today")

    def test_no_live_price_means_not_actionable(self):
        def boom(_s):
            raise RuntimeError("net")
        row, = trend.actionable_live([_row()], bar_fn=boom)
        self.assertFalse(row["valid"])
        self.assertEqual(row["invalid_reason"], "no_live_price")

    def test_rows_that_are_not_actionable_are_left_untouched(self):
        late = _row(category="late", days=40)
        self.assertEqual(trend.actionable_live([late], bar_fn=lambda s: _bar(0.9)), [late])


class OpenDemoTests(_Base):
    def test_opens_at_the_live_price_sized_by_the_live_stop_distance(self):
        pos = trend.open_demo("FILUSDT", risk_pct=0.5, now_rows=[_row()], bar=_bar(0.852, l=0.845))
        self.assertIsNone(pos["tp"])
        self.assertEqual((pos["strategy"], pos["market"], pos["side"], pos["tf"]),
                         ("trend20", "spot", "long", "1d"))
        self.assertEqual((pos["entry"], pos["sl"]), (0.852, 0.773))
        self.assertEqual(pos["cost_pct"], explore.SPOT_ROUND_TRIP_PCT)
        self.assertEqual(pos["ref"], {"rule": trend.PRIMARY, "rule_entry_ts": E_TS})
        self.assertAlmostEqual(pos["size_usdt"] * (0.852 - 0.773) / 0.852, 50.0, delta=0.1)   # ۰٫۵٪ موجودی

    def test_only_todays_actionable_signals_can_be_opened(self):
        with self.assertRaises(ValueError):
            trend.open_demo("BTCUSDT", now_rows=[_row("BTCUSDT", category="late", stop=90, last=100, days=40)],
                            bar=_bar(100, o=100))
        with self.assertRaises(ValueError):
            trend.open_demo("XRPUSDT", now_rows=[_row("XRPUSDT", category="watch", in_position=False)],
                            bar=_bar(1.0, o=1.0))

    def test_a_signal_whose_stop_was_hit_today_cannot_be_bought(self):
        with self.assertRaises(ValueError) as cm:
            trend.open_demo("FILUSDT", now_rows=[_row()], bar=_bar(0.80, l=0.7557))
        self.assertIn("باطل", str(cm.exception))

    def test_the_same_coin_is_not_opened_twice(self):
        trend.open_demo("FILUSDT", now_rows=[_row()], bar=_bar(0.9))
        with self.assertRaises(ValueError):
            trend.open_demo("FILUSDT", now_rows=[_row()], bar=_bar(0.9))

    def test_risk_is_clamped_and_never_leveraged(self):
        pos = trend.open_demo("FILUSDT", risk_pct=50, now_rows=[_row(stop=0.899)], bar=_bar(0.9))
        self.assertLessEqual(pos["size_usdt"], 10_000.0)                    # بی‌اهرم

    def test_total_open_trend_risk_is_capped(self):
        """ارزها با هم می‌ریزند: ۴٪ ریسکِ باز سقف است، هرچند هر معامله جداگانه کوچک باشد."""
        for i in range(2):
            trend.open_demo(f"C{i}USDT", risk_pct=2.0, now_rows=[_row(f"C{i}USDT")], bar=_bar(0.9))
        self.assertAlmostEqual(trend._demo_heat_pct(), 4.0, delta=0.05)
        with self.assertRaises(ValueError) as cm:
            trend.open_demo("C9USDT", risk_pct=0.5, now_rows=[_row("C9USDT")], bar=_bar(0.9))
        self.assertIn("سقف", str(cm.exception))


class SyncDemoTests(_Base):
    def _open(self, price=0.852):
        return trend.open_demo("FILUSDT", now_rows=[_row()], bar=_bar(price, l=price - 0.005))

    def test_the_fil_case_closes_at_the_rules_stop_not_at_whatever_the_price_is_later(self):
        pos = self._open()
        gone = _row(category="watch", in_position=False,
                    exits=[{"entry_ts": E_TS, "exit_ts": E_TS + 2 * 86_400_000, "exit_px": 0.772996,
                            "outcome": "stop"}])
        res = trend.sync_demo([gone], price_fn=lambda s: 0.7967)
        self.assertEqual(res["closed"], 1)
        closed = [p for p in paper.list_positions()["closed"] if p["id"] == pos["id"]][0]
        self.assertEqual(closed["close_reason"], "حدضررِ قاعدهٔ روند")
        self.assertLess(closed["exit_price"], 0.773)                          # حدضرر + لغزش، نه ۰٫۷۹۶۷
        self.assertAlmostEqual(closed["exit_price"], 0.772996 * (1 - paper.slippage_bps(0.4) / 1e4), places=6)
        # ≈ −۱R: ۰٫۵٪ موجودی + کارمزد
        self.assertLess(closed["pnl_usdt"], -49.0)

    def test_trailing_stop_only_rises_while_the_same_rule_trade_is_open(self):
        pos = self._open()
        trend.sync_demo([_row(stop=0.80)])
        trend.sync_demo([_row(stop=0.78)])                                     # پایین‌تر ⇒ نه
        live = [p for p in paper.list_positions()["open"] if p["id"] == pos["id"]][0]
        self.assertEqual(live["sl"], 0.80)

    def test_a_newer_rule_trade_does_not_keep_the_old_demo_position_alive(self):
        pos = self._open()
        newer = _row(entry_ts=E_TS + 9 * 86_400_000, stop=0.70,
                     exits=[{"entry_ts": E_TS, "exit_ts": E_TS + 2 * 86_400_000, "exit_px": 0.773,
                             "outcome": "stop"}])
        self.assertEqual(trend.sync_demo([newer])["closed"], 1)
        self.assertFalse(any(p["id"] == pos["id"] for p in paper.list_positions()["open"]))

    def test_a_rule_timeout_exit_is_mirrored_at_its_own_price_without_stop_slippage(self):
        pos = self._open()
        gone = _row(category="watch", in_position=False,
                    exits=[{"entry_ts": E_TS, "exit_ts": E_TS + 120 * 86_400_000, "exit_px": 1.50,
                            "outcome": "timeout"}])
        trend.sync_demo([gone])
        closed = [p for p in paper.list_positions()["closed"] if p["id"] == pos["id"]][0]
        self.assertEqual((closed["exit_price"], closed["close_reason"]), (1.50, "خروجِ قاعدهٔ روند"))

    def test_an_unknown_reference_trade_falls_back_to_the_market_price(self):
        pos = self._open()
        gone = _row(category="watch", in_position=False, exits=[])
        trend.sync_demo([gone], price_fn=lambda s: 0.81)
        closed = [p for p in paper.list_positions()["closed"] if p["id"] == pos["id"]][0]
        self.assertEqual(closed["exit_price"], 0.81)

    def test_a_position_without_target_closes_on_its_stop_and_pays_no_funding(self):
        pos = self._open(0.936)
        paper.refresh({"FILUSDT": 1.20}, funding_fn=lambda s: [[4_102_444_800_000, 0.001]])
        live = [p for p in paper.list_positions()["open"] if p["id"] == pos["id"]][0]
        self.assertEqual(live.get("funding_pct", 0.0), 0.0)                  # اسپات
        paper.refresh({"FILUSDT": 0.70})
        self.assertFalse(any(p["id"] == pos["id"] for p in paper.list_positions()["open"]))

    def test_advisor_says_hold_the_trend_instead_of_crashing(self):
        pos = self._open(0.936)
        adv = advisor.advise(dict(pos, last_price=1.0), {"p_up": 40})
        self.assertEqual(adv["action"], "hold_trend")


if __name__ == "__main__":
    unittest.main()
