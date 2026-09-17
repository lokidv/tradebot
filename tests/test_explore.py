# -*- coding: utf-8 -*-
"""ابزارِ جست‌وجوی لبه باید خودش سالم باشد — وگرنه «لبهٔ پیداشده» خطای ابزار است.

نگاه به آینده، دادهٔ پنجرهٔ منجمد، و حدضررِ خوش‌بینانه سه راهِ ساختنِ لبهٔ جعلی‌اند.
"""
import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import calib  # noqa: E402
import explore  # noqa: E402
import universe  # noqa: E402

from test_bracket_contract import synthetic_klines  # noqa: E402

DAY = 86_400_000
T0 = 1_577_836_800_000          # 2020-01-01 (چهارشنبه)


def _daily(n, drift=0.0, vol=0.02, seed=1, start=T0, price=100.0, volume=1e7):
    rng = np.random.default_rng(seed)
    c = price * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    wig = np.abs(rng.normal(0, vol / 2, n)) * c
    return {"t": (start + np.arange(n, dtype=np.int64) * DAY).tolist(), "o": o.tolist(),
            "h": (np.maximum(o, c) + wig).tolist(), "l": (np.minimum(o, c) - wig).tolist(),
            "c": c.tolist(), "v": np.full(n, volume).tolist()}


def _arrays(k):
    return {key: np.asarray(v, np.int64 if key == "t" else float) for key, v in k.items()}


class CutoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _put(self, sym, k, tf="1d"):
        with open(os.path.join(self.tmp.name, f"{sym}_{tf}.json"), "w", encoding="utf-8") as f:
            json.dump(k, f)

    def test_no_bar_at_or_after_the_cutoff_is_loaded(self):
        self._put("AAAUSDT", _daily(400))
        cutoff = T0 + 300 * DAY
        panel = explore.load_panel("1d", cutoff, hist_dir=self.tmp.name)
        self.assertEqual(len(panel["AAAUSDT"]["t"]), 300)
        self.assertLess(int(panel["AAAUSDT"]["t"].max()), cutoff)

    def test_pegged_and_duplicate_assets_are_excluded(self):
        peg = _daily(200, vol=0.0001, price=1.0)
        self._put("USDGUSDT", peg)
        self._put("BETHUSDT", _daily(200, seed=3))
        self._put("AAAUSDT", _daily(200, seed=4))
        panel = explore.load_panel("1d", T0 + 999 * DAY, hist_dir=self.tmp.name)
        self.assertEqual(sorted(panel), ["AAAUSDT"])

    def test_candles_off_the_exchange_grid_are_excluded(self):
        """۱۶:۰۰ یعنی OKX: در بایننس معامله‌پذیر نیست و تقویمِ هفتگی را می‌شکست."""
        okx = _daily(200, seed=6)
        okx["t"] = [t + 16 * 3_600_000 for t in okx["t"]]
        self._put("OKBUSDT", okx)
        self._put("AAAUSDT", _daily(200, seed=7))
        panel = explore.load_panel("1d", T0 + 999 * DAY, hist_dir=self.tmp.name)
        self.assertEqual(sorted(panel), ["AAAUSDT"])

    def test_cutoff_is_the_earliest_frozen_window(self):
        doc = {"final_windows": {"4h": {"start_ms": 500, "end_ms": 900},
                                 "1d": {"start_ms": 300, "end_ms": 800}}}
        self.assertEqual(explore.dev_cutoff_ms(doc), 300)

    def test_no_registration_means_no_exploration(self):
        with self.assertRaises(ValueError):
            explore.dev_cutoff_ms({"final_windows": {}})


class WalkTests(unittest.TestCase):
    """حدضرر و خروج باید دقیقاً مثلِ اجرای واقعی باشند، نه خوش‌بینانه‌تر."""

    def _series(self, o, h, l, c):
        a = np.full(len(c), 1.0)
        return [np.asarray(x, float) for x in (o, h, l, c)] + [a]

    def test_intrabar_stop_exits_at_the_stop_price(self):
        o, h, l, c, a = self._series([100, 100, 99], [101, 101, 100], [99.5, 96, 98], [100, 97, 99])
        res = explore.walk(o, h, l, c, a, 1, 1, 100.0, R=3.0)
        self.assertEqual((res["outcome"], res["exit_px"], res["exit_idx"]), ("stop", 97.0, 1))

    def test_a_gap_through_the_stop_exits_at_the_worse_open(self):
        o, h, l, c, a = self._series([100, 100, 95], [101, 101, 96], [99, 99.5, 94], [100, 100, 95])
        res = explore.walk(o, h, l, c, a, 1, 1, 100.0, R=3.0)
        self.assertEqual((res["outcome"], res["exit_px"]), ("gap_stop", 95.0))

    def test_trailing_stop_only_tightens_and_uses_closed_bars(self):
        # بستهٔ ۱۱۰ در کندلِ ۲ ⇒ حدِ جدید 110−3 = 107 از کندلِ ۳ اعمال می‌شود
        o = [100, 100, 104, 110, 108]
        h = [100, 105, 110, 111, 108.5]
        l = [100, 99, 103, 108, 106]
        c = [100, 104, 110, 108.5, 107]
        o, h, l, c, a = self._series(o, h, l, c)
        res = explore.walk(o, h, l, c, a, 1, 1, 100.0, R=3.0, trail_atr=3.0)
        self.assertEqual(res["exit_idx"], 4)
        self.assertEqual(res["exit_px"], 107.0)

    def test_exit_signal_leaves_at_the_next_open(self):
        o, h, l, c, a = self._series([100, 100, 101, 102], [101, 102, 103, 103],
                                     [99, 99.5, 100, 101], [100, 101, 102, 102.5])
        res = explore.walk(o, h, l, c, a, 1, 1, 100.0, R=5.0, exit_fn=lambda j: j == 2)
        self.assertEqual((res["outcome"], res["exit_idx"], res["exit_px"]), ("signal", 3, 102.0))

    def test_open_trade_at_the_end_of_data_is_marked_censored(self):
        o, h, l, c, a = self._series([100, 100, 101], [101, 102, 103], [99, 99.5, 100], [100, 101, 102])
        res = explore.walk(o, h, l, c, a, 1, 1, 100.0, R=5.0, max_hold=10)
        self.assertEqual(res["outcome"], "end_of_data")


class EntryTimingTests(unittest.TestCase):
    def test_breakout_is_judged_on_past_highs_and_entered_at_the_next_open(self):
        k = _arrays(_daily(120, vol=0.005, seed=2))
        i = 80
        k["c"][i] = k["h"][i - 20:i].max() * 1.05          # شکستِ واضح روی بستهٔ کندلِ i
        k["h"][i] = max(k["h"][i], k["c"][i])
        snaps = [(T0 - DAY, frozenset({"AAAUSDT"}))]
        trades = explore.trend_trades({"AAAUSDT": k}, snaps,
                                      {"rule": "donchian", "n": 20, "side": 1})
        hit = [tr for tr in trades if tr["ts"] == int(k["t"][i])]
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0]["entry_ts"], int(k["t"][i + 1]))

    def test_a_symbol_outside_the_point_in_time_universe_is_never_traded(self):
        k = _arrays(_daily(200, drift=0.01, vol=0.01, seed=9))
        trades = explore.trend_trades({"AAAUSDT": k}, [(T0 - DAY, frozenset({"BBBUSDT"}))],
                                      {"rule": "donchian", "n": 20, "side": 1})
        self.assertEqual(trades, [])

    def test_daily_trend_filter_reads_only_a_closed_daily_bar(self):
        k = _arrays(_daily(150, drift=0.01, vol=0.002, seed=4))
        fn = explore.daily_trend_lookup({"AAAUSDT": k}, sma_n=100)
        t_last = int(k["t"][-1])
        self.assertEqual(fn("AAAUSDT", t_last + DAY), 1)       # کندلِ آخر بسته شده
        # پیش از بسته‌شدنِ اولین کندلی که SMA دارد، روند نامعلوم است
        self.assertEqual(fn("AAAUSDT", int(k["t"][99]) + DAY - 1), 0)


class JoinTests(unittest.TestCase):
    """پیوستن دیرتر به همان معامله: همان خروج، ولی ورود و R از روزِ پیوستن."""

    def test_a_join_leaves_with_the_rule_and_is_measured_from_its_own_stop(self):
        k = _arrays(_daily(400, drift=0.004, vol=0.02, seed=31))
        snaps = [(T0 - DAY, frozenset({"AAAUSDT"}))]
        spec = {"rule": "donchian", "n": 20, "side": 1}
        rule = {t["entry_ts"]: t for t in explore.trend_trades({"AAAUSDT": k}, snaps, spec)}
        joins = explore.join_trades({"AAAUSDT": k}, snaps, spec, 5)
        self.assertGreater(len(joins), 0)
        for jt in joins:
            parent, = [t for ets, t in rule.items() if ets == jt["entry_ts"] - 5 * DAY]
            self.assertEqual(jt["exit_ts"], parent["exit_ts"])             # همان روزِ خروجِ قاعده
            d = int(np.searchsorted(k["t"], jt["entry_ts"]))
            join_px = k["o"][d]
            stop_px = join_px * (1 - jt["risk_pct"] / 100)
            self.assertLess(stop_px, join_px)
            # و همان قیمتِ خروج: فقط ورود و R فرق دارند
            e = int(np.searchsorted(k["t"], parent["entry_ts"]))
            parent_r = parent["risk_pct"] / 100 * k["o"][e]
            parent_exit = k["o"][e] + parent["gross_r"] * parent_r
            join_exit = join_px + jt["gross_r"] * (join_px - stop_px)
            self.assertAlmostEqual(join_exit, parent_exit, places=6)

    def test_spot_costs_have_no_funding(self):
        k = _arrays(_daily(400, drift=0.004, vol=0.02, seed=31))
        snaps = [(T0 - DAY, frozenset({"AAAUSDT"}))]
        for jt in explore.join_trades({"AAAUSDT": k}, snaps, {"rule": "donchian", "n": 20, "side": 1}, 5):
            self.assertAlmostEqual(jt["net_r_spot"],
                                   jt["gross_r"] - explore.SPOT_ROUND_TRIP_PCT / max(jt["risk_pct"], 0.05))


class FilterStudyTests(unittest.TestCase):
    """ابزارِ آزمونِ فیلترها: معیارِ پذیرش ازپیش‌گفته است و باید همان‌طور که گفته شد اجرا شود."""

    SPEC = {"rule": "donchian", "n": 20, "side": 1}

    def _panel(self):
        return {"AAAUSDT": _arrays(_daily(500, drift=0.003, vol=0.03, seed=51))}, [(T0 - DAY, frozenset({"AAAUSDT"}))]

    def test_offset_zero_is_exactly_the_rules_own_trade(self):
        panel, snaps = self._panel()
        rule = explore.trend_trades(panel, snaps, self.SPEC)
        base = [r for r in explore.opportunities(panel, snaps, self.SPEC) if r["offset"] == 0]
        self.assertEqual(len(base), len(rule))
        for a, b in zip(base, rule):
            self.assertEqual((a["entry_ts"], a["exit_ts"]), (b["entry_ts"], b["exit_ts"]))
            self.assertAlmostEqual(a["net_r"], b["net_r"], places=9)
            self.assertAlmostEqual(a["cushion_atr"], explore.TREND_STOP_ATR, places=6)    # حدضررِ اولیه = 3×ATR

    def test_join_features_use_the_entry_open_and_yesterdays_btc(self):
        """هیچ ویژگی‌ای از آینده نمی‌آید: BTC = بستهٔ **دیروز**؛ سقف = ۲۰ کندلِ پیش از سیگنال؛ قیمت = openِ ورود."""
        panel, snaps = self._panel()
        k = panel["AAAUSDT"]
        n = len(k["t"])
        btc_c = np.concatenate([np.linspace(300, 100, n // 2), np.linspace(100, 400, n - n // 2)])   # نزولی، بعد صعودی
        btc = {"t": k["t"], "c": btc_c}
        m = explore.sma(btc_c, explore.BTC_REGIME_SMA)
        rows = [r for r in explore.opportunities(panel, snaps, self.SPEC, btc=btc) if r["offset"] > 0]
        self.assertGreater(len(rows), 10)
        seen = set()
        for r in rows:
            d = int(np.searchsorted(k["t"], r["entry_ts"]))
            expected = bool(np.isfinite(m[d - 1]) and btc_c[d - 1] > m[d - 1])
            self.assertEqual(r["btc_up"], expected)
            seen.add(expected)
            i = d - r["offset"] - 1                                           # کندلِ سیگنالِ همان معاملهٔ قاعده
            level = float(np.max(k["h"][i - 20:i]))
            self.assertEqual(r["above_level"], bool(k["o"][d] > level))
        self.assertEqual(seen, {True, False})                                 # هر دو رژیم واقعاً آزموده شدند

    def test_profit_target_exits_at_the_target_and_loses_a_same_bar_tie(self):
        o = np.array([100, 100, 101.0]); h = np.array([100, 104, 103.0])
        l = np.array([100, 99.5, 100.0]); c = np.array([100, 103, 102.0]); a = np.ones(3)
        res = explore.walk(o, h, l, c, a, 1, 1, 100.0, R=3.0, target_r=1.0)
        self.assertEqual((res["outcome"], res["exit_px"]), ("target", 103.0))
        l[1] = 96.0                                                         # همان کندل حدضرر را هم زده
        self.assertEqual(explore.walk(o, h, l, c, a, 1, 1, 100.0, R=3.0, target_r=1.0)["outcome"], "stop")

    def test_diff_lcb_sees_a_real_separation_and_refuses_tiny_groups(self):
        rng = np.random.default_rng(3)
        ts = T0 + np.arange(600, dtype=np.int64) * DAY
        mask = np.arange(600) % 2 == 0
        vals = np.where(mask, 1.0, -1.0) + rng.normal(0, 0.5, 600)
        self.assertGreater(explore.diff_lcb(vals, mask, ts, 30 * DAY, 0.0167, n_boot=300), 1.5)
        noise = rng.normal(0, 1.0, 600)
        self.assertLess(explore.diff_lcb(noise, mask, ts, 30 * DAY, 0.0167, n_boot=300), 0.0)
        self.assertIsNone(explore.diff_lcb(vals[:3], mask[:3], ts[:3], 30 * DAY, 0.0167))

    def test_adoption_needs_every_prestated_condition(self):
        ok = {"all_mean": 0.39, "keep_fraction": 0.7, "diff_lcb": 0.05,
              "kept": {"n": 900, "mean": 0.47, "lcb": 0.16, "first_half_mean": 0.8, "second_half_mean": 0.09}}
        self.assertTrue(explore.filter_passes(ok))
        self.assertFalse(explore.filter_passes(dict(ok, diff_lcb=-0.21)))            # نتیجهٔ واقعیِ فیلترِ BTC
        self.assertFalse(explore.filter_passes(dict(ok, keep_fraction=0.3)))
        self.assertFalse(explore.filter_passes(dict(ok, kept=dict(ok["kept"], second_half_mean=-0.01))))
        self.assertFalse(explore.filter_passes(dict(ok, kept=dict(ok["kept"], mean=0.30))))

    def test_trade_profile_shows_where_the_profit_comes_from(self):
        rows = [{"entry_ts": i, "net_r": r} for i, r in enumerate([-1, -1, -1, 5, -1, -1, 0.5, -1, -1, 9])]
        p = explore.trade_profile(rows)
        self.assertEqual((p["win_rate"], p["max_consecutive_losses"]), (30.0, 3))
        self.assertAlmostEqual(p["top_decile_share_of_gains"], 9 / 14.5, places=3)
        self.assertLess(p["mean_without_top_decile"], 0)


class SetupParityTests(unittest.TestCase):
    """ستاپ‌های اکتشاف باید دقیقاً همان رویدادهای آموزش باشند؛ وگرنه دو آزمون دو چیزِ متفاوت را می‌سنجند."""

    def test_setup_events_match_calib_extract_events(self):
        kl = synthetic_klines(n=1200, seed=11)
        mine = explore.setup_events("AAAUSDT", _arrays(kl), "4h")
        ref, *_ = calib.extract_events("AAAUSDT", kl, "4h", [], {}, {}, {})
        key = lambda e: (int(e["ts"]), e["dir"], e["setup"], round(e["r"], 3),
                         round(e["risk_pct"], 4), round(float(e["cost_pct"]), 6))
        self.assertGreater(len(ref), 0)
        self.assertEqual([key(e) for e in mine], [key(e) for e in ref])


class CrossSectionTests(unittest.TestCase):
    def _panel(self, drifts, n=200):
        return {f"C{i:02d}USDT": _arrays(_daily(n, drift=d, vol=0.01, seed=20 + i))
                for i, d in enumerate(drifts)}

    def test_momentum_earns_when_winners_keep_winning_and_pays_costs(self):
        drifts = np.linspace(-0.01, 0.01, 12)
        panel = self._panel(drifts)
        snaps = [(T0 - DAY, frozenset(panel))]
        weeks = explore.xsec_weeks(panel, snaps, {"lookback": 14, "sign": 1})
        self.assertGreater(len(weeks), 10)
        self.assertGreater(np.mean([w["net_pct"] for w in weeks]), 0)
        self.assertTrue(all(w["net_pct"] < w["gross_pct"] for w in weeks))

    def test_entry_is_monday_open_and_ranking_stops_at_sunday_close(self):
        panel = self._panel(np.linspace(-0.01, 0.01, 12))
        weeks = explore.xsec_weeks(panel, [(T0 - DAY, frozenset(panel))], {"lookback": 7, "sign": 1})
        self.assertTrue(all(time.gmtime(w["ts"] / 1000).tm_wday == 0 for w in weeks))

    def test_the_weekly_calendar_covers_the_whole_history(self):
        panel = self._panel(np.linspace(-0.01, 0.01, 12), n=400)
        weeks = explore.xsec_weeks(panel, [(T0 - DAY, frozenset(panel))], {"lookback": 7, "sign": 1})
        self.assertGreaterEqual(len(weeks), 400 // 7 - 3)


class RunSmokeTests(unittest.TestCase):
    def test_full_run_on_a_tiny_panel_produces_a_verdict_for_every_variant(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            for i, d in enumerate(np.linspace(-0.004, 0.004, 24)):
                with open(os.path.join(tmp.name, f"C{i:02d}USDT_1d.json"), "w", encoding="utf-8") as f:
                    json.dump(_daily(500, drift=d, vol=0.03, seed=40 + i, volume=1e7 * (i + 1)), f)
            specs = tuple(v for v in explore.VARIANTS if v["family"] != "setup_trend")
            rep = explore.run(cutoff_ms=T0 + 480 * DAY, variants=specs, hist_dir=tmp.name,
                              progress=lambda *_: None)
            self.assertEqual(set(rep["variants"]), {v["key"] for v in specs})
            for blob in rep["variants"].values():
                self.assertIn(blob["verdict"], ("promising", "weak", "inconclusive", "no_edge"))
            self.assertLess(max(max(blob["stats"].get("by_year") or {"0": 0}) for blob in rep["variants"].values()),
                            "2022")
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
