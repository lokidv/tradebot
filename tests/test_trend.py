# -*- coding: utf-8 -*-
"""ردیابیِ رو-به-جلوِ روندِ روزانه: همان قاعدهٔ پژوهش، بی‌انتخاب، و ثبت‌شده برای همیشه.

اگر ردیابِ زنده حتی کمی با آنچه آزموده شد فرق کند، شواهدِ تازه دربارهٔ چیزِ دیگری است.
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import explore  # noqa: E402
import trend  # noqa: E402

DAY = 86_400_000
T0 = 1_735_689_600_000          # 2025-01-01


def _series(n=420, seed=3, drift=0.002, vol=0.03, start=T0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    wig = np.abs(rng.normal(0, vol / 2, n)) * c
    return {"t": (start + np.arange(n, dtype=np.int64) * DAY), "o": o,
            "h": np.maximum(o, c) + wig, "l": np.minimum(o, c) - wig, "c": c,
            "v": np.full(n, 1e8)}


def _as_lists(k):
    return {key: np.asarray(v).tolist() for key, v in k.items()}


class ForwardTrackingTests(unittest.TestCase):
    def setUp(self):
        if os.path.exists(trend.LEDGER_PATH):
            os.remove(trend.LEDGER_PATH)
        trend._cache.update(ts=0.0, data=None)

    def test_closed_trades_match_the_research_loop_exactly(self):
        k = _series()
        daily = {"BTCUSDT": k}
        live = trend.evaluate(daily, start_ms=0)["T_don20_long"]["closed"]
        ref = explore.trend_trades(daily, trend._FIXED_UNIVERSE, trend.RULES["T_don20_long"])
        ref = [r for r in ref if r["outcome"] != "end_of_data"]
        self.assertGreater(len(ref), 3)
        pick = lambda r: (r["ts"], r["entry_ts"], r["exit_ts"], round(r["net_r"], 9))
        self.assertEqual([pick(r) for r in live], [pick(r) for r in ref])

    def test_nothing_before_the_start_counts_and_the_chain_starts_flat(self):
        k = _series()
        start = int(k["t"][250])
        state = trend.evaluate({"BTCUSDT": k}, start_ms=start)
        for rows in state.values():
            for r in rows["closed"]:
                self.assertGreaterEqual(r["ts"], start)
            for r in rows["open"]:
                self.assertGreaterEqual(r["signal_ts"], start)

    def test_result_does_not_depend_on_where_the_data_window_begins(self):
        """پنجرهٔ ۴۲۰ کندلی هر روز جلو می‌رود؛ معامله‌های رو-به-جلو نباید عوض شوند."""
        k = _series(n=520)
        start = int(k["t"][400])
        cut = {key: v[60:] for key, v in k.items()}
        a = trend.evaluate({"BTCUSDT": k}, start_ms=start)
        b = trend.evaluate({"BTCUSDT": cut}, start_ms=start)
        for key in trend.RULES:
            ids = lambda s: sorted(r["id"] for r in s[key]["closed"] + s[key]["open"])
            self.assertEqual(ids(a), ids(b), key)

    def test_open_position_reports_the_live_trailing_stop(self):
        k = _series(seed=8, drift=0.004)
        state = trend.evaluate({"BTCUSDT": k}, start_ms=0)
        opens = [p for rows in state.values() for p in rows["open"]]
        for p in opens:
            self.assertLess(p["stop"], p["last"] * 1.5)
            self.assertGreater(p["risk_pct"], 0)

    def test_a_breakout_on_the_last_closed_bar_is_a_signal_for_tomorrow(self):
        k = _series(n=300, seed=5, drift=0.0, vol=0.01)
        k["c"][-1] = k["h"][-21:-1].max() * 1.08
        k["h"][-1] = k["c"][-1] * 1.001
        state = trend.evaluate({"BTCUSDT": k}, start_ms=int(k["t"][-1]))
        sig = state["T_don20_long"]["signals"]
        self.assertEqual([s["signal_ts"] for s in sig], [int(k["t"][-1])])

    def test_ledger_records_each_entry_and_exit_exactly_once(self):
        k = _series()
        state = trend.evaluate({"BTCUSDT": k}, start_ms=0)
        first = trend.record(state)
        self.assertGreater(first, 0)
        self.assertEqual(trend.record(state), 0)
        rows = trend._read()
        exits = [r for r in rows if r["kind"] == "exit"]
        entries = {r["id"] for r in rows if r["kind"] == "entry"}
        self.assertTrue(all(e["id"] in entries for e in exits))
        fwd = trend.forward_stats(rows)
        self.assertEqual(sum(s["n"] for s in fwd.values()), len(exits))

    def test_state_now_shows_every_coin_and_marks_pre_start_positions_as_uncounted(self):
        up = _series(seed=8, drift=0.004)
        flat = _series(seed=12, drift=0.0, vol=0.01)
        rows = trend.state_now({"BTCUSDT": up, "ETHUSDT": flat}, trend.PRIMARY)
        self.assertEqual({r["sym"] for r in rows}, {"BTCUSDT", "ETHUSDT"})
        for r in rows:
            if r["in_position"]:
                self.assertIn("counted", r)
                self.assertLess(r["stop"], r["last"] * 1.5)
            elif r["category"] == "watch":
                # سطحِ ماشه همان سقفِ ۲۰ کندلِ بسته‌شده است
                k = {"BTCUSDT": up, "ETHUSDT": flat}[r["sym"]]
                self.assertAlmostEqual(r["trigger"], float(np.max(k["h"][-20:])))
        order = [trend.CATEGORY_ORDER[r["category"]] for r in rows]
        self.assertEqual(order, sorted(order))              # تازه، پیوستن، دیر، در کمین

    def test_a_coin_already_in_position_never_also_gets_a_new_entry_signal(self):
        """روندِ پیوسته هر روز سقفِ تازه می‌زند؛ این برای قاعده‌ای که از قبل خریده «ورودِ تازه» نیست."""
        n = 200
        c = 100 * (1 + 0.004 * np.arange(n))                  # هر بسته سقفِ ۲۰روزهٔ تازه است
        o = np.concatenate([[c[0]], c[:-1]])
        k = {"t": T0 + np.arange(n, dtype=np.int64) * DAY, "o": o, "h": np.maximum(o, c) * 1.002,
             "l": np.minimum(o, c) * 0.998, "c": c, "v": np.full(n, 1e8)}
        state = trend.evaluate({"BTCUSDT": k}, start_ms=0)
        for key, rows in state.items():
            held = {p["sym"] for p in rows["open"]}
            self.assertFalse(held & {s["sym"] for s in rows["signals"]}, key)
        row, = trend.state_now({"BTCUSDT": k}, trend.PRIMARY)
        self.assertTrue(row["in_position"])
        self.assertNotEqual(row["category"], "new")

    def test_joining_is_offered_only_inside_the_tested_window(self):
        """پیوستن تا ۱۰ روز پس از شکست آزموده و مثبت بود؛ دیرتر نه."""
        for days_after in (3, 10, 25):
            n = 200
            b = n - days_after                                # کندلِ ورودِ قاعده (سیگنال روی b−1)
            c = np.empty(n)
            c[:b - 1] = 100 * (1 - 0.001 * np.arange(b - 1))  # پیش از شکست: نزولِ آرام، بی‌هیچ شکستی
            c[b - 1] = c[b - 2] * 1.06                        # شکستِ واضحِ سقفِ ۲۰روزه
            c[b:] = c[b - 1] * (1 + 0.004 * np.arange(1, n - b + 1))   # بعد روندِ پیوسته
            o = np.concatenate([[c[0]], c[:-1]])
            k = {"t": T0 + np.arange(n, dtype=np.int64) * DAY, "o": o,
                 "h": np.maximum(o, c) * 1.002, "l": np.minimum(o, c) * 0.998, "c": c,
                 "v": np.full(n, 1e8)}
            row, = trend.state_now({"BTCUSDT": k}, trend.PRIMARY)
            self.assertTrue(row["in_position"], days_after)
            self.assertEqual(row["days_in"], days_after)
            self.assertEqual(row["category"], "join" if days_after <= trend.JOIN_MAX_DAYS else "late")
            self.assertAlmostEqual(row["stop_distance_pct"], (row["last"] - row["stop"]) / row["last"] * 100)

    def test_the_extension_is_thirty_distinct_non_major_coins(self):
        self.assertEqual(len(trend.EXT_SYMBOLS), 30)
        self.assertEqual(len(set(trend.ALL_SYMBOLS)), 50)
        self.assertFalse(set(trend.SYMBOLS) & set(trend.EXT_SYMBOLS))
        self.assertFalse([s for s in trend.EXT_SYMBOLS if explore._excluded(s)])
        self.assertNotIn("PAXGUSDT", trend.EXT_SYMBOLS)                 # طلا دارایی دیگری است
        self.assertEqual((trend.universe_of("BTCUSDT"), trend.universe_of("SUIUSDT")), ("majors", "top50"))

    def test_forward_record_keeps_the_majors_and_the_extension_apart(self):
        rows = [{"kind": "exit", "rule": trend.PRIMARY, "sym": "BTCUSDT", "net_r": 1.0, "entry_ts": 1},
                {"kind": "exit", "rule": trend.PRIMARY, "sym": "SUIUSDT", "net_r": -1.0, "entry_ts": 2},
                {"kind": "exit", "rule": trend.PRIMARY, "sym": "PEPEUSDT", "net_r": 2.0, "entry_ts": 3,
                 "universe": "top50"}]
        fwd = trend.forward_stats(rows)
        self.assertEqual(fwd[trend.PRIMARY]["n"], 1)
        self.assertEqual(fwd[f"{trend.PRIMARY}@top50"]["n"], 2)
        self.assertEqual(fwd[f"{trend.PRIMARY}@top50"]["sum_r"], 1.0)

    def test_the_prestated_expansion_rule_needs_all_four_conditions(self):
        ok = {"n": 800, "mean": 0.38, "lcb": 0.14, "first_half_mean": 0.63, "second_half_mean": 0.14,
              "mean_stressed": 0.32}
        self.assertTrue(explore.expansion_passes(ok))
        for k, bad in (("lcb", -0.01), ("second_half_mean", -0.02), ("mean_stressed", -0.1), ("mean", -0.1)):
            self.assertFalse(explore.expansion_passes(dict(ok, **{k: bad})), k)

    def test_snapshot_reports_missing_and_off_grid_data_and_never_authorizes(self):
        good = _as_lists(_series())
        okx = _as_lists(_series(seed=9))
        okx["t"] = [t + 16 * 3_600_000 for t in okx["t"]]

        def fetch(sym):
            if sym == "BTCUSDT":
                return good
            if sym == "ETHUSDT":
                return okx
            raise RuntimeError("no data")
        snap = trend.snapshot(force=True, fetch=fetch)
        self.assertFalse(snap["authorized"])
        self.assertIn("ETHUSDT", snap["missing_data"])
        self.assertNotIn("BTCUSDT", snap["missing_data"])
        self.assertEqual(snap["primary"], "T_don20_long")


if __name__ == "__main__":
    unittest.main()
