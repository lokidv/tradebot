# -*- coding: utf-8 -*-
"""اجرای روند روی تست‌نت: فقط تست‌نت، فقط با مجوزِ کاربر، همان حدضررِ کاغذ، و هرگز بی‌محافظ.

صرافیِ جعلی زیرکلاسِ ``broker.BinanceTestnet`` است (بی‌شبکه)، تا همان نگهبانِ نوعِ کلاینت
که در اجرای واقعی هست اینجا هم سنجیده شود.
"""
import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import broker  # noqa: E402
import gates  # noqa: E402
import journal  # noqa: E402
import report  # noqa: E402
import trend  # noqa: E402
import trend_exec  # noqa: E402

DAY = 86_400_000
SIG_TS = 1_789_516_800_000               # 2026-09-16 00:00 UTC — کندلِ سیگنال
BAR_OPEN = SIG_TS + DAY                  # ورود در openِ کندلِ بعد
RULE = trend_exec.RULE


class FakeTestnet(broker.BinanceTestnet):
    def __init__(self, marks, equity=10_000.0, listed=("BTCUSDT", "ETHUSDT")):
        self.leverage = 2
        self.marks = dict(marks)
        self.equity = equity
        self.listed = set(listed)
        self.pos, self.orders, self.fills, self.realized = {}, {}, [], {}
        self.calls, self._next, self.glitch = [], 100, 0

    def symbol_rules(self, symbol):
        if symbol not in self.listed:
            raise broker.TestnetError("not listed")
        return 0.001, 0.1, 0.001, 5.0

    def mark_price(self, symbol):
        return self.marks[symbol]

    def balance_usdt(self):
        return self.equity, self.equity

    def positions(self):
        if self.glitch:
            self.glitch -= 1
            return []
        return [{"symbol": s, "side": "long", "qty": p["qty"], "entry": p["entry"]} for s, p in self.pos.items()]

    def _order(self, symbol, stop, qty):
        self._next += 1
        self.orders[self._next] = {"symbol": symbol, "stop": stop, "qty": qty}
        return self._next

    def open_with_stop(self, symbol, side, qty, stop_price):
        fill = self.marks[symbol]
        self.pos[symbol] = {"qty": qty, "entry": fill}
        oid = self._order(symbol, stop_price, qty)
        self.calls.append(("open", symbol, qty, stop_price))
        return {"order_id": 1, "qty": qty, "fill": fill, "stop_order_id": oid}

    def move_stop(self, symbol, side, qty, new_stop):
        new_id = self._order(symbol, new_stop, qty)
        self.orders = {o: v for o, v in self.orders.items() if v["symbol"] != symbol or o == new_id}
        self.calls.append(("move", symbol, new_stop))
        return new_id

    def _flatten(self, symbol, px):
        p = self.pos.pop(symbol)
        self.orders = {o: v for o, v in self.orders.items() if v["symbol"] != symbol}
        self.fills.append({"symbol": symbol, "side": "SELL", "price": px, "qty": p["qty"]})
        self.realized[symbol] = (px - p["entry"]) * p["qty"]

    def close_symbol(self, symbol):
        self.calls.append(("close", symbol))
        if symbol in self.pos:
            self._flatten(symbol, self.marks[symbol])
            return True
        return False

    def realized_pnl_since(self, symbol, start_ms):
        return self.realized.get(symbol, 0.0)

    def fills_since(self, symbol, start_ms):
        return [f for f in self.fills if f["symbol"] == symbol]


def _snap(signals=(), open_rows=(), closed=()):
    return {"last_closed_bar_ms": SIG_TS,
            "rules": {RULE: {"signals": list(signals), "open": list(open_rows), "closed_recent": list(closed)}}}


def _signal(sym="BTCUSDT", dist=6_000.0):
    return {"sym": sym, "signal_ts": SIG_TS, "close": 80_000.0, "stop_distance": dist, "risk_pct": 7.5}


class _Base(unittest.TestCase):
    def setUp(self):
        for p in (trend_exec.EVENTS_PATH, trend.LEDGER_PATH, journal.PATH):
            if os.path.exists(p):
                os.remove(p)
        with open(gates.GATES_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(gates.DEFAULTS), f)
        gates.load_gates(force=True)
        self.market = mock.patch.multiple(trend_exec.market,
                                          current_bar=mock.Mock(return_value={
                                              "t": BAR_OPEN, "o": 80_000.0, "h": 80_500.0,
                                              "l": 79_500.0, "c": 80_100.0}),
                                          spot_price=mock.Mock(return_value=80_100.0))
        self.market.start()

    def tearDown(self):
        self.market.stop()
        gates._cache.update(mtime=None, data=None)

    def permit(self, on=True):
        ver = gates.load_gates(force=True)["version"]
        gates.set_testnet_research([trend_exec.STRATEGY] if on else [], "test", user_token=f"user:{ver}")

    def step(self, bk, snap, hours=1.0):
        return trend_exec.step(bk=bk, snap=snap, now=(BAR_OPEN + hours * 3_600_000) / 1000)

    def enter(self):
        bk = FakeTestnet({"BTCUSDT": 80_100.0})
        self.permit()
        self.step(bk, _snap(signals=[_signal()]))
        return bk


class PermissionTests(_Base):
    def test_nothing_happens_without_the_users_permission(self):
        bk = FakeTestnet({"BTCUSDT": 80_100.0})
        self.assertEqual(self.step(bk, _snap(signals=[_signal()]))["status"], "not_permitted")
        self.assertEqual(bk.calls, [])

    def test_only_the_testnet_client_is_ever_used(self):
        self.permit()
        self.assertEqual(trend_exec.step(bk=object(), snap=_snap(signals=[_signal()]))["status"], "no_testnet")
        with mock.patch.object(broker, "make_broker", return_value=None):          # شبیه‌سازِ محلی
            self.assertEqual(trend_exec.step(snap=_snap(signals=[_signal()]))["status"], "no_testnet")

    def test_permission_needs_the_explicit_user_token(self):
        with self.assertRaises(gates.GateError):
            gates.set_testnet_research([trend_exec.STRATEGY], "x", user_token="judge:1")

    def test_kill_clears_the_permission_and_flattens(self):
        bk = self.enter()
        gates.kill("test")
        self.assertFalse(gates.testnet_research_allowed(trend_exec.STRATEGY))
        self.assertEqual(trend_exec.close_all("test", bk=bk), 1)
        self.assertEqual(bk.pos, {})

    def test_the_permission_is_not_a_change_to_what_money_may_do(self):
        self.permit()
        evs = [e for e in report.gate_changes_since(0) if e.get("action") == "testnet_research"]
        self.assertTrue(evs)
        self.assertEqual(report.gate_changes_during_proving("h", 0), [])     # ساعتِ اثبات صفر نمی‌شود


class EntryTests(_Base):
    def test_signal_enters_at_market_with_the_paper_stop_and_risk_sizing(self):
        bk = self.enter()
        (_kind, sym, qty, stop), = bk.calls
        self.assertEqual(sym, "BTCUSDT")
        self.assertEqual(stop, 80_000.0 - 6_000.0)                          # همان حدضررِ کاغذ
        risk = 10_000.0 * gates.risk_caps()["max_risk_pct_per_trade"] / 100
        actual = qty * (80_100.0 - stop)
        # گردکردن به گامِ صرافی فقط رو به پایین: هرگز بیش از سقف ریسک نمی‌کند
        self.assertLessEqual(actual, risk)
        self.assertGreater(actual, risk - 0.001 * (80_100.0 - stop))
        tr, = [t for t in trend_exec.replay()[0].values()]
        self.assertAlmostEqual(tr["entry_slippage_bps"], (80_100.0 / 80_000.0 - 1) * 1e4)

    def test_the_same_signal_is_never_entered_twice(self):
        bk = self.enter()
        self.step(bk, _snap(signals=[_signal()]), hours=2)
        self.assertEqual([c[0] for c in bk.calls], ["open"])

    def test_a_late_signal_is_missed_not_chased(self):
        bk = FakeTestnet({"BTCUSDT": 80_100.0})
        self.permit()
        self.step(bk, _snap(signals=[_signal()]), hours=7)
        self.assertEqual(bk.calls, [])
        self.assertEqual(trend_exec.summary()["execution"]["missed_by_reason"], {"late": 1})

    def test_testnet_price_far_from_the_real_market_is_skipped(self):
        bk = FakeTestnet({"BTCUSDT": 83_000.0})                             # ۳٪+ دور از اسپات
        self.permit()
        self.step(bk, _snap(signals=[_signal()]))
        self.assertEqual(bk.calls, [])
        self.assertIn("testnet_price_gap", trend_exec.summary()["execution"]["missed_by_reason"])

    def test_a_stop_already_touched_today_kills_the_entry(self):
        """کفِ امروز حدضرر را زده و قیمت برگشته: قاعدهٔ کاغذی بیرون است؛ ورود یعنی معاملهٔ نیازموده."""
        bk = FakeTestnet({"BTCUSDT": 80_100.0})
        self.permit()
        bar = {"t": BAR_OPEN, "o": 80_000.0, "h": 80_500.0, "l": 73_900.0, "c": 80_100.0}
        with mock.patch.object(trend_exec.market, "current_bar", return_value=bar):
            self.step(bk, _snap(signals=[_signal()]))
        self.assertEqual(bk.calls, [])
        self.assertEqual(trend_exec.summary()["execution"]["missed_by_reason"], {"stop_breached_today": 1})

    def test_symbol_missing_on_testnet_is_recorded_as_missed(self):
        bk = FakeTestnet({"SOLUSDT": 100.0}, listed=())
        self.permit()
        self.step(bk, _snap(signals=[_signal("SOLUSDT", dist=8.0)]))
        self.assertEqual(trend_exec.summary()["execution"]["missed_by_reason"], {"not_on_testnet": 1})

    def test_a_position_it_did_not_open_is_left_alone(self):
        bk = FakeTestnet({"BTCUSDT": 80_100.0})
        bk.pos["BTCUSDT"] = {"qty": 1.0, "entry": 70_000.0}
        self.permit()
        self.step(bk, _snap(signals=[_signal()]))
        self.assertEqual(bk.calls, [])
        self.assertEqual(bk.pos["BTCUSDT"]["qty"], 1.0)

    def test_transient_errors_are_retried_on_the_next_round(self):
        bk = FakeTestnet({"BTCUSDT": 80_100.0})
        self.permit()
        with mock.patch.object(trend_exec.market, "current_bar", side_effect=RuntimeError("net")):
            self.step(bk, _snap(signals=[_signal()]))
        self.assertEqual(bk.calls, [])
        self.step(bk, _snap(signals=[_signal()]), hours=1.2)
        self.assertEqual([c[0] for c in bk.calls], ["open"])


class ManagementTests(_Base):
    def _open_row(self, stop):
        tid = f"{RULE}:BTCUSDT:{SIG_TS}"
        return {"id": tid, "sym": "BTCUSDT", "signal_ts": SIG_TS, "entry_ts": BAR_OPEN,
                "entry_px": 80_000.0, "stop": stop, "last": 82_000.0, "open_r": 0.3, "risk_pct": 7.5}

    def test_trailing_stop_only_moves_up(self):
        bk = self.enter()
        self.step(bk, _snap(open_rows=[self._open_row(76_500.0)]), hours=25)
        self.step(bk, _snap(open_rows=[self._open_row(75_000.0)]), hours=49)   # پایین‌تر ⇒ نه
        moves = [c for c in bk.calls if c[0] == "move"]
        self.assertEqual(moves, [("move", "BTCUSDT", 76_500.0)])
        self.assertEqual([v["stop"] for v in bk.orders.values()], [76_500.0])   # حدضررِ قدیمی لغو شد

    def test_exchange_stop_is_recorded_with_real_r_and_stop_slippage(self):
        bk = self.enter()
        bk._flatten("BTCUSDT", 73_900.0)                                      # زیرِ حدضرر پر شد
        self.step(bk, _snap(open_rows=[self._open_row(74_000.0)]), hours=5)
        tr, = trend_exec.replay()[0].values()
        self.assertEqual((tr["status"], tr["reason"]), ("closed", "exchange_closed"))
        self.assertAlmostEqual(tr["realized_r"], tr["realized_usdt"] / tr["risk_usdt"])
        self.assertGreater(tr["stop_slippage_bps"], 0)

    def test_rule_exit_closes_the_exchange_position(self):
        bk = self.enter()
        with open(trend.LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "exit", "id": f"{RULE}:BTCUSDT:{SIG_TS}", "rule": RULE,
                                "net_r": -1.0, "entry_ts": BAR_OPEN}) + "\n")
        self.step(bk, _snap(), hours=30)
        self.assertIn(("close", "BTCUSDT"), bk.calls)
        tr, = trend_exec.replay()[0].values()
        self.assertEqual(tr["reason"], "rule_exit_sync")
        self.assertEqual(trend_exec.summary()["closed"][0]["paper_r"], -1.0)

    def test_a_momentary_empty_positions_reply_does_not_abandon_a_live_trade(self):
        bk = self.enter()
        bk.glitch = 1                                                         # فقط پاسخِ اول خالی
        self.step(bk, _snap(open_rows=[self._open_row(74_000.0)]), hours=5)
        tr, = trend_exec.replay()[0].values()
        self.assertEqual(tr["status"], "open")

    def test_without_permission_open_trades_are_still_managed_but_nothing_new_opens(self):
        bk = self.enter()
        self.permit(False)
        bk.marks["ETHUSDT"] = 3_000.0
        res = self.step(bk, _snap(signals=[_signal("ETHUSDT", dist=200.0)],
                                  open_rows=[self._open_row(77_000.0)]), hours=25)
        self.assertEqual(res["status"], "winding_down")
        self.assertIn(("move", "BTCUSDT", 77_000.0), bk.calls)
        self.assertNotIn("ETHUSDT", bk.pos)

    def test_state_survives_a_restart(self):
        bk = self.enter()
        trades, decided = trend_exec.replay()
        self.assertEqual(len(trades), 1)
        self.assertIn(f"{RULE}:BTCUSDT:{SIG_TS}", decided)
        self.step(bk, _snap(signals=[_signal()]), hours=3)                   # «ری‌استارت»: فقط از فایل
        self.assertEqual([c[0] for c in bk.calls], ["open"])


if __name__ == "__main__":
    unittest.main()
