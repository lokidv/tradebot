# -*- coding: utf-8 -*-
"""سیگنالِ v3: همان قاعده با دو تغییرِ ثبت‌شده، بازپخشِ بی‌نگاه‌به‌آینده، کارنامهٔ matured، آزمون، ترمز و حکمِ یک‌باره."""
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import bracket  # noqa: E402
import decision  # noqa: E402
import engine  # noqa: E402
import signal_v3 as s3  # noqa: E402

PREREG = os.path.join(ROOT, "bot", "data", "research", "prereg_signal_v3.json")
H4 = 4 * 3_600_000
D1 = 86_400_000
STEP = {"4h": H4, "1d": D1}
T0 = s3.T0_MS
MON = (int(s3.week_of([T0])[0]) + 1) * s3.WEEK_MS + s3.MONDAY0_MS      # نخستین دوشنبهٔ پس از T0 (T0 سه‌شنبه است)


def synth(n, seed=3, step_ms=H4, t_first=None, before=None):
    """گامِ تصادفی با رژیم‌های روند/رنج. ``before``: تعدادِ کندل‌های پیش از T0 (کندلِ before دقیقاً در T0 باز می‌شود)."""
    rng = np.random.RandomState(seed)
    drift = np.repeat(rng.choice([-1.0, 0.0, 1.0], n // 150 + 1), 150)[:n] * 0.0015
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.0005, n))
    wig = np.abs(rng.normal(0, 0.003, n)) * c
    if before is not None:
        t_first = T0 - int(before) * step_ms
    elif t_first is None:
        t_first = 400_000 * H4
    return {"t": t_first + np.arange(n, dtype=np.int64) * step_ms,
            "o": o, "h": np.maximum(o, c) + wig, "l": np.minimum(o, c) - wig, "c": c,
            "v": rng.lognormal(10, 0.5, n)}


def trade(t, r, side="long", matured=True):
    return {"t": int(t), "net_r": float(r), "gross_r": float(r) + 0.1, "side": side, "matured": matured,
            "risk_pct": 1.4}


class PreregTests(unittest.TestCase):
    def test_constants_match_the_preregistration(self):
        pr = s3.check_prereg(PREREG)
        self.assertEqual(s3.SIGNAL_TFS, ("4h", "1d"))
        self.assertEqual(s3.SKIP_SETUPS, frozenset({"fd", "sq"}))
        self.assertIsNone(s3.MIN_VOTES)
        self.assertEqual(s3.T0_MS, 1_788_220_800_000)
        self.assertIn("Forward-only", pr["title"])

    def test_a_drifted_constant_is_refused(self):
        for name, val in (("MOM_DAYS", 30), ("BRAKE_MIN_N", 10), ("BOOT_SEED", 1), ("MIN_VOTES", 4),
                          ("SKIP_SETUPS", frozenset({"fd"})), ("WARMUP_BARS", 500)):
            with mock.patch.object(s3, name, val):
                with self.assertRaises(ValueError, msg=name):
                    s3.check_prereg(PREREG)

    def test_replay_refuses_a_start_inside_the_sealed_holdout(self):
        kl = synth(700)
        for bad in (s3.SEALED_MS[0], T0 - 1):
            with self.assertRaises(ValueError):
                s3.replay(kl, start_ts=bad)
        s3.replay(kl, start_ts=s3.SEALED_MS[0] - 1)       # پیش از ۲۰۲۶ مجاز است
        s3.replay(kl, start_ts=T0)


class MomentumTests(unittest.TestCase):
    def test_close_against_the_close_28_days_earlier(self):
        for step, k in ((H4, 168), (D1, 28)):
            kl = synth(400, step_ms=step)
            m = s3.momentum(kl["t"], kl["c"])
            self.assertTrue(np.isnan(m[:k]).all())
            np.testing.assert_allclose(m[k:], kl["c"][k:] / kl["c"][:-k] - 1)

    def test_a_missing_reference_bar_is_unknown_not_a_guess(self):
        kl = synth(400, step_ms=H4)
        keep = np.ones(400, bool)
        keep[100] = False                                    # کندلِ مرجعِ کندلِ ۲۶۸ حذف شد
        t, c = kl["t"][keep], kl["c"][keep]
        m = s3.momentum(t, c)
        j = int(np.flatnonzero(t == kl["t"][268])[0])
        self.assertTrue(np.isnan(m[j]))
        self.assertAlmostEqual(m[j + 1], kl["c"][269] / kl["c"][101] - 1)


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kl = synth(1500, seed=11)
        cls.first = decision.LIVE_BARS - 1
        cls.ev = s3.evaluate(cls.kl, cls.first)

    def test_without_skips_and_conditions_it_is_exactly_the_rule(self):
        for seed in (5, 11):
            kl = synth(1300, seed=seed)
            side, setup, _ = decision.side_series(kl, self.first)
            ev = s3.evaluate(kl, self.first, skip=(), trend=False)
            np.testing.assert_array_equal(ev["base"], side)
            np.testing.assert_array_equal(ev["side"], side)
            self.assertEqual(ev["setup"], setup)
            self.assertGreater(int(np.count_nonzero(side)), 20)

    def test_registered_spec_never_uses_fd_or_sq(self):
        rule = decision.side_series(self.kl, self.first)[1]
        self.assertTrue(any(s in ("fd", "sq") for s in rule))          # آزمون بی‌محتوا نیست
        self.assertFalse(any(s in s3.SKIP_SETUPS for s in self.ev["setup"]))

    def test_a_skipped_setup_falls_through_to_the_previous_bar_then_the_z_rule(self):
        kl = self.kl
        real = engine.setup_signal
        n = len(kl["c"])
        target = n - 300

        def fake(cs, o, h, l, c, j):
            if j == target:
                return 1, "fd"
            if j == target - 1:
                return -1, "pb"
            if j in (target + 1, target - 2):
                return 0, None
            return real(cs, o, h, l, c, j)

        with mock.patch.object(engine, "setup_signal", fake):
            with_fd = s3.evaluate(kl, self.first, skip=(), trend=False)
            without = s3.evaluate(kl, self.first, trend=False)
        self.assertEqual((int(with_fd["base"][target]), with_fd["setup"][target]), (1, "fd"))
        # fd کنار رفت ⇒ اسکن به کندلِ قبل می‌رود و ستاپِ pb آن (اگر هنوز معتبر است) تصمیم می‌گیرد
        valid = -1 * (kl["c"][target] - kl["c"][target - 1]) >= -1.2 * float(without["a14"][target])
        if valid:
            self.assertEqual((int(without["base"][target]), without["setup"][target]), (-1, "pb"))
        else:
            self.assertIsNone(without["setup"][target])

    def test_trend_condition_keeps_only_sides_with_the_28_day_move(self):
        ev = self.ev
        base, mom, side = ev["base"], ev["mom"], ev["side"]
        with np.errstate(invalid="ignore"):
            agree = ((base == 1) & (mom > 0)) | ((base == -1) & (mom < 0))
        np.testing.assert_array_equal(side, np.where(agree, base, 0))
        self.assertGreater(int(np.count_nonzero(side)), 10)
        self.assertGreater(int(np.count_nonzero((base != 0) & (side == 0))), 3)     # چیزی هم فیلتر شده
        self.assertTrue((side[np.isnan(mom)] == 0).all())                           # نامعلوم ⇒ صبر

    def test_zero_momentum_is_wait(self):
        ev = dict(self.ev)
        with mock.patch.object(s3, "momentum", lambda t, c, days=28: np.zeros(len(c))):
            ev0 = s3.evaluate(self.kl, self.first)
        self.assertEqual(int(np.count_nonzero(ev0["side"])), 0)
        self.assertGreater(int(np.count_nonzero(ev["base"])), 0)

    def test_votes_are_the_same_side_count_of_five(self):
        ev = self.ev
        long_, short_ = ev["base"] == 1, ev["base"] == -1
        np.testing.assert_array_equal(ev["votes"][long_], ev["bull5"][long_])
        np.testing.assert_array_equal(ev["votes"][short_], ev["bear5"][short_])
        self.assertTrue((ev["votes"][ev["base"] == 0] == 0).all())
        only4 = s3.evaluate(self.kl, self.first, min_votes=4)
        self.assertTrue((only4["votes"][only4["side"] != 0] >= 4).all())
        self.assertLess(int(np.count_nonzero(only4["side"])), int(np.count_nonzero(ev["side"])))

    def test_no_lookahead(self):
        kl = synth(1500, seed=5)
        d = 1100
        base = s3.replay(kl, start_ts=int(kl["t"][500]))
        rng = np.random.RandomState(99)
        pert = {k: np.array(v, copy=True) for k, v in kl.items()}
        shock = np.exp(np.cumsum(rng.normal(0, 0.02, len(pert["c"]) - d - 1)))
        for k in ("o", "h", "l", "c"):
            pert[k][d + 1:] *= shock
        pert["v"][d + 1:] = rng.lognormal(12, 1.0, len(shock))
        other = s3.replay(pert, start_ts=int(kl["t"][500]))
        for key in ("base", "side", "votes"):
            np.testing.assert_array_equal(base["ev"][key][:d + 1], other["ev"][key][:d + 1], err_msg=key)
        np.testing.assert_array_equal(base["ev"]["mom"][:d + 1], other["ev"]["mom"][:d + 1])
        a = [(x["i"], x["side"], x["entry"]) for x in base["trades"] if x["i"] < d]
        b = [(x["i"], x["side"], x["entry"]) for x in other["trades"] if x["i"] < d]
        self.assertEqual(a, b)
        self.assertGreater(len(a), 3)

    def test_a_prefix_gives_the_same_last_bar(self):
        full = self.ev
        for m in (700, 955, 1310):
            pre = s3.evaluate({k: v[:m] for k, v in self.kl.items()}, self.first)
            for key in ("base", "side", "votes"):
                self.assertEqual(int(pre[key][-1]), int(full[key][m - 1]), f"{key} m={m}")
            np.testing.assert_equal(pre["mom"][-1], full["mom"][m - 1])


class SimulateTests(unittest.TestCase):
    def test_rule_sides_give_exactly_the_rule_trades(self):
        kl = synth(1400, seed=8)
        first = decision.LIVE_BARS - 1
        ev = s3.evaluate(kl, first, skip=(), trend=False)
        mine = s3.simulate(ev)["trades"]
        ref = decision.replay(kl, first=first)["trades"]
        keys = ("i", "t", "side", "setup", "entry", "sl", "tp", "outcome", "exit_idx", "bars_held")
        self.assertEqual([tuple(x[k] for k in keys) for x in mine], [tuple(x[k] for k in keys) for x in ref])
        for a, b in zip(mine, ref):
            self.assertAlmostEqual(a["net_r"], b["net_r"])
            self.assertAlmostEqual(a["net_r"], bracket.net_r(a["gross_r"], a["risk_pct"], 0.14))
        self.assertGreater(len(mine), 5)

    def test_one_trade_at_a_time_and_maturity(self):
        kl = synth(1500, seed=11)
        out = s3.replay(kl, start_ts=int(kl["t"][500]))
        tr, n = out["trades"], len(out["ev"]["t"])
        self.assertGreater(len(tr), 5)
        for prev, nxt in zip(tr, tr[1:]):
            self.assertGreaterEqual(nxt["i"], prev["exit_idx"])
        for x in tr:
            self.assertEqual(x["matured"], x["i"] + 40 <= n - 1)
            self.assertEqual(x["entry_t"], x["t"] + H4)
            self.assertLessEqual(x["bars_held"], 40)
        self.assertEqual([x["t"] for x in s3.matured(tr)], sorted(x["t"] for x in tr if x["matured"]))

    def _one_signal(self, n, at, sg=1, path="flat"):
        """سریِ ساختگی با دقیقاً یک سیگنال در کندلِ ``at`` و مسیرِ قیمتِ دلخواه پس از آن."""
        t = T0 + np.arange(n, dtype=np.int64) * H4
        c = np.full(n, 100.0)
        if path == "up":
            c[at + 1:] = 100.0 + np.arange(1, n - at) * 0.5
        ev = {"t": t, "o": c.copy(), "h": c + 0.01, "l": c - 0.01, "c": c, "first": 0,
              "side": np.zeros(n, np.int8), "setup": [None] * n, "votes": np.zeros(n, int),
              "mom": np.full(n, 0.1), "z": np.zeros(n), "a14": np.full(n, 1.0)}
        ev["side"][at] = sg
        return ev

    def test_signal_on_the_last_closed_bar_is_fresh_not_a_trade(self):
        ev = self._one_signal(50, 49)
        out = s3.simulate(ev)
        self.assertEqual(out["trades"], [])
        self.assertIsNone(out["open"])
        lv = bracket.levels(100.0, 1.0, 1)
        self.assertEqual((out["fresh"]["entry"], out["fresh"]["sl"], out["fresh"]["tp"]), (lv["entry"], lv["sl"], lv["tp"]))
        self.assertEqual(out["fresh"]["t"], int(ev["t"][49]))

    def test_unresolved_trade_is_open_and_blocks_later_signals(self):
        ev = self._one_signal(60, 30)
        ev["side"][45] = -1                                   # سیگنالِ دوم وقتی اولی هنوز باز است
        ev["side"][59] = 1                                    # و سیگنالِ کندلِ آخر
        out = s3.simulate(ev)
        self.assertEqual(out["trades"], [])
        self.assertIsNone(out["fresh"])                       # در معامله‌ایم ⇒ سیگنالِ تازه‌ای نیست
        self.assertEqual((out["open"]["t"], out["open"]["bars_held"]), (int(ev["t"][30]), 29))
        self.assertAlmostEqual(out["open"]["unrealized_r"], 0.0)

    def test_timeout_inside_the_data_is_a_matured_trade(self):
        ev = self._one_signal(100, 10)
        out = s3.simulate(ev)
        self.assertEqual(len(out["trades"]), 1)
        x = out["trades"][0]
        self.assertEqual((x["outcome"], x["bars_held"], x["matured"]), ("timeout", 40, True))
        self.assertAlmostEqual(x["net_r"], -0.14 / x["risk_pct"])

    def test_a_quick_win_near_the_end_is_resolved_but_not_matured(self):
        ev = self._one_signal(60, 40, path="up")
        ev["h"] = ev["c"] + 0.01
        out = s3.simulate(ev)
        self.assertEqual(len(out["trades"]), 1)
        self.assertEqual(out["trades"][0]["outcome"], "target")
        self.assertFalse(out["trades"][0]["matured"])
        self.assertEqual(s3.matured(out["trades"]), [])


class WindowTests(unittest.TestCase):
    def test_result_does_not_depend_on_how_much_history_is_loaded(self):
        kl = synth(3300, seed=4)
        start = int(kl["t"][2700])
        a = s3.replay(kl, start_ts=start)
        b = s3.replay({k: v[400:] for k, v in kl.items()}, start_ts=start)     # ۳۰۰ کندلِ اضافه پیش از گرم‌شدن
        keys = ("t", "side", "entry", "sl", "tp", "net_r", "exit_t")
        self.assertEqual([tuple(x[k] for k in keys) for x in a["trades"]],
                         [tuple(x[k] for k in keys) for x in b["trades"]])
        self.assertGreater(len(a["trades"]), 3)
        sub, i0 = s3.window(kl, start)
        self.assertEqual((i0, len(sub["t"])), (2000, 2600))


class StatsTests(unittest.TestCase):
    def test_mean_and_weekly_cluster_error(self):
        w = s3.WEEK_MS
        tr = [trade(T0, 1.8), trade(T0 + 1, 1.8), trade(T0 + w, -1.0), trade(T0 + w + 5, -1.0, "short"),
              trade(T0 + 2 * w, 1.8, "short")]
        st = s3.stats(tr)
        rs = np.array([1.8, 1.8, -1.0, -1.0, 1.8])
        m = rs.mean()
        self.assertEqual((st["n"], st["weeks"], st["long_n"], st["short_n"]), (5, 3, 3, 2))
        self.assertAlmostEqual(st["mean"], round(m, 4))
        self.assertAlmostEqual(st["se"], round(rs.std(ddof=1) / np.sqrt(5), 4))
        S, N = np.array([3.6, -2.0, 1.8]), np.array([2.0, 2.0, 1.0])
        se_w = np.sqrt(((S - m * N) ** 2).sum()) / 5
        self.assertAlmostEqual(st["se_week"], round(se_w, 4))
        self.assertAlmostEqual(st["ci_lo"], round(m - 1.96 * se_w, 4))
        self.assertGreater(st["se_week"], st["se"])            # معامله‌های هم‌هفته هم‌جهت‌اند ⇒ خطای واقعی بزرگ‌تر
        self.assertEqual(s3.stats([])["n"], 0)
        self.assertIsNone(s3.stats([])["mean"])

    def test_bootstrap_p_value(self):
        w = s3.WEEK_MS
        win = [trade(MON + k * w, 1.0) for k in range(30)]
        lose = [trade(MON + k * w, -1.0) for k in range(30)]
        a = s3.bootstrap(win, MON, MON + 30 * w)
        self.assertEqual((a["weeks"], a["n"], a["mean"]), (30, 30, 1.0))
        self.assertAlmostEqual(a["p"], round(1 / 10001, 5))
        self.assertEqual(s3.bootstrap(lose, MON, MON + 30 * w)["p"], 1.0)
        none = s3.bootstrap([], MON, MON + 30 * w)
        self.assertEqual((none["p"], none["mean"]), (None, None))
        again = s3.bootstrap(win, MON, MON + 30 * w)
        self.assertEqual(a, again)                             # بذرِ ثابت ⇒ همان عدد
        # هفتهٔ تقویمی، نه «۷ روز از شروع»: بازه‌ای که وسطِ هفته شروع شود یک بلوکِ ناقص هم دارد
        self.assertEqual(s3.bootstrap(win[:1], MON - 86_400_000, MON + 30 * w)["weeks"], 31)

    def test_bootstrap_counts_empty_weeks_as_blocks_and_no_trade_draws_as_failures(self):
        w = s3.WEEK_MS
        one = [trade(MON + 3 * w, 1.0)]
        bt = s3.bootstrap(one, MON, MON + 10 * w, n_boot=4000)
        self.assertEqual(bt["weeks"], 10)
        # تکراری که هفتهٔ معامله‌دار را نکشد میانگین ندارد ⇒ ≤ ۰ حساب می‌شود: (۹/۱۰)^۱۰ ≈ ۰٫۳۵
        self.assertAlmostEqual(bt["p"], 0.9 ** 10, delta=0.03)
        with self.assertRaises(ValueError):
            s3.bootstrap([trade(MON + 11 * w, 1.0)], MON, MON + 10 * w)

    def test_bootstrap_reports_the_difference_to_the_rule(self):
        w = s3.WEEK_MS
        v3 = [trade(T0 + k * w, 0.5) for k in range(20)]
        rule = [trade(T0 + k * w, -0.5) for k in range(20)]
        bt = s3.bootstrap(v3, T0, T0 + 20 * w, rule=rule)
        self.assertEqual((bt["rule"]["n"], bt["rule"]["mean"], bt["rule"]["diff"]), (20, -0.5, 1.0))
        self.assertLess(bt["rule"]["p_diff"], 0.001)

    def test_brake_total_negative_needs_twenty_matured_trades(self):
        mk = lambda rs, **kw: [trade(T0 + i * H4, r, **kw) for i, r in enumerate(rs)]
        self.assertFalse(s3.brake(mk([-1.0] * 19))["muted"])
        self.assertFalse(s3.brake(mk([-1.0] * 19))["armed"])
        b = s3.brake(mk([-1.0] * 20))
        self.assertEqual((b["muted"], b["code"], b["n"], b["mean"]), (True, "total_negative", 20, -1.0))
        self.assertFalse(s3.brake(mk([1.8] * 8 + [-1.0] * 12))["muted"])           # میانگین +۰٫۱۲
        self.assertTrue(s3.brake(mk([1.8] * 7 + [-1.0] * 13))["muted"])            # میانگین −۰٫۰۲
        self.assertFalse(s3.brake(mk([0.0] * 20))["muted"])                        # صفر منفی نیست
        self.assertFalse(s3.brake(mk([-1.0] * 30, matured=False))["muted"])        # فقط matured شمرده می‌شود
        self.assertEqual(s3.brake([])["n"], 0)

    def test_brake_recent_drop_uses_the_latest_hundred(self):
        mk = lambda rs: [trade(T0 + i * H4, r) for i, r in enumerate(rs)]
        good = [1.8] * 60                                           # شروعِ خوب
        drop = ([1.8] * 30 + [-1.0] * 70)                           # ۱۰۰ تای آخر: میانگین −۰٫۱۶
        b = s3.brake(mk(good + drop))
        self.assertEqual((b["muted"], b["code"]), (True, "recent_drop"))
        self.assertGreater(b["mean"], 0)
        self.assertAlmostEqual(b["recent_mean"], -0.16)
        mild = ([1.8] * 33 + [-1.0] * 67)                           # ۱۰۰ تای آخر: −۰٫۰۷۶ ⇒ بالای کف
        self.assertFalse(s3.brake(mk(good + mild))["muted"])
        self.assertIsNone(s3.brake(mk([1.8] * 99))["recent_mean"])  # زیرِ ۱۰۰ معامله بندِ دوم خاموش است

    def test_judge_two_looks(self):
        w = s3.WEEK_MS
        look1, look2 = s3.LOOKS
        strong = [trade(T0 + k * w, 0.6) for k in range(50)]
        weak = [trade(T0 + k * w, 1.8 if k % 3 == 0 else -1.0) for k in range(51)]    # میانگین −۰٫۰۷
        mixed = [trade(T0 + k * w, 1.8 if k % 5 < 2 else -1.0) for k in range(50)]     # میانگین +۰٫۱۲، p بزرگ
        self.assertEqual(s3.judge(look1, strong)["result"], "pass")
        self.assertEqual(s3.judge(look1, weak)["result"], "fail")                      # میانگین ≤ ۰ ⇒ بیهودگی
        j = s3.judge(look1, mixed)
        self.assertEqual(j["result"], "continue")
        self.assertGreater(j["p"], 0.01)
        self.assertEqual(s3.judge(look2, mixed)["result"], "pass" if s3.judge(look2, mixed)["p"] <= 0.04 else "fail")
        self.assertEqual(s3.judge(look2, weak)["result"], "fail")
        late = strong + [trade(look1["cutoff_ms"] + 5, -50.0)]                         # پس از سقفِ نگاهِ ۱: نادیده
        self.assertEqual(s3.judge(look1, late)["n"], 50)
        self.assertEqual(s3.judge(look1, [])["result"], "fail")
        self.assertEqual(s3.judge(look1, [dict(x, matured=False) for x in strong])["n"], 0)


class PlanTests(unittest.TestCase):
    SIG = {"t": T0, "side": "long", "entry": 100.0, "sl": 98.0, "tp": 103.6, "setup": "pb", "votes": 4,
           "z": 1.44, "mom": 0.0712, "atr": 1.5}

    def test_fresh_signal_plan(self):
        p = s3.plan("4h", self.SIG)
        self.assertEqual((p["entry"], p["sl"], p["tp"], p["rr"]), (100.0, 98.0, 103.6, 1.8))
        self.assertEqual((p["risk_pct"], p["gain_pct"]), (2.0, 3.6))
        self.assertEqual(p["entry_t"], T0 + H4)                       # بازِ کندلِ بعد
        self.assertEqual(p["valid_until"], T0 + 2 * H4)               # تا بسته‌شدنِ کندلِ ورود
        self.assertEqual(p["time_stop_t"], T0 + H4 + 40 * H4)
        self.assertEqual((p["entry_lo"], p["entry_hi"], p["chase_limit"]), (99.5, 100.5, 100.5))
        self.assertEqual(p["cost_r"], 0.07)                           # ۰٫۱۴٪ از حدضررِ ۲٪
        self.assertEqual((p["setup_fa"], p["mom_pct"]), (engine.SETUP_FA["pb"], 7.12))
        short = s3.plan("1d", dict(self.SIG, side="short", sl=102.0, tp=96.4))
        self.assertEqual((short["chase_limit"], short["valid_until"]), (99.5, T0 + 2 * D1))

    def test_open_trade_plan(self):
        p = s3.plan("4h", dict(self.SIG, entry_t=T0 + H4, bars_held=7, last=101.0, unrealized_r=0.5))
        self.assertEqual((p["bars_held"], p["bars_left"], p["unrealized_r"], p["last"]), (7, 33, 0.5, 101.0))
        self.assertNotIn("valid_until", p)


class BuildTests(unittest.TestCase):
    SYMS = ("AAAUSDT", "BBBUSDT")

    @classmethod
    def setUpClass(cls):
        cls.data = {}
        for k, sym in enumerate(cls.SYMS):
            cls.data[(sym, "4h")] = synth(2700, seed=20 + k, step_ms=H4, before=2100)
            cls.data[(sym, "1d")] = synth(2350, seed=30 + k, step_ms=D1, before=2100)
        cls.now_ms = int(max(d["t"][-1] for d in cls.data.values())) + D1
        cls.snap = s3.build(lambda s, tf: cls.data[(s, tf)], cls.now_ms, symbols=cls.SYMS, keep_cells=True)

    def test_cells_and_counts(self):
        snap = self.snap
        self.assertTrue(snap["complete"])
        self.assertEqual(snap["errors"], {})
        self.assertEqual((snap["signal_tfs"], snap["status_only_tfs"]), (["4h", "1d"], ["5m", "15m", "1h"]))
        n_tf = {tf: 0 for tf in s3.SIGNAL_TFS}
        for c in snap["_cells"]:
            m = s3.matured(c["trades"])
            cell = snap["cells"][c["sym"]][c["tf"]]
            self.assertEqual(cell["stats"]["n"], len(m))
            self.assertEqual(cell["pending"], len(c["trades"]) - len(m))
            self.assertTrue(all(x["t"] >= T0 for x in c["trades"]))            # هیچ سیگنالی پیش از T0
            self.assertEqual(cell["bars"]["warmup"], 2000)
            self.assertEqual(cell["bars"]["gaps"], 0)
            n_tf[c["tf"]] += len(m)
        for tf in s3.SIGNAL_TFS:
            self.assertEqual(snap["tfs"][tf]["stats"]["n"], n_tf[tf])
            self.assertEqual(snap["tfs"][tf]["brake"]["n"], n_tf[tf])
        self.assertEqual(snap["pooled"]["stats"]["n"], sum(n_tf.values()))
        self.assertEqual(snap["pooled"]["test"]["n"], sum(n_tf.values()))
        self.assertGreater(n_tf["4h"], 10)
        self.assertEqual(snap["status"], "testing")

    def test_rule_stream_is_the_decision_replay_on_the_same_bars(self):
        c = next(x for x in self.snap["_cells"] if (x["sym"], x["tf"]) == (self.SYMS[0], "4h"))
        sub, _ = s3.window(self.data[(self.SYMS[0], "4h")], T0)
        ref = decision.replay(sub, start_ts=T0)["trades"]
        self.assertEqual([(x["t"], x["side"], x["net_r"]) for x in c["rule_trades"]],
                         [(x["t"], x["side"], x["net_r"]) for x in ref])
        self.assertEqual(self.snap["tfs"]["4h"]["rule"]["n"],
                         sum(len(s3.matured(x["rule_trades"])) for x in self.snap["_cells"] if x["tf"] == "4h"))

    def test_states_and_signal_list(self):
        snap = self.snap
        listed = {(s["sym"], s["tf"]): s for s in snap["signals"]}
        for c in snap["_cells"]:
            cell = snap["cells"][c["sym"]][c["tf"]]
            if c["open"]:
                self.assertEqual(cell["state"], "in_trade")
                self.assertEqual(listed[(c["sym"], c["tf"])]["entry"], cell["open"]["entry"])
            elif c["fresh"]:
                self.assertEqual(cell["state"], "signal")
                self.assertIn("valid_until", listed[(c["sym"], c["tf"])])
            else:
                self.assertIn(cell["state"], ("wait", "filtered"))
                self.assertNotIn((c["sym"], c["tf"]), listed)
            self.assertEqual(cell["muted"], snap["tfs"][c["tf"]]["brake"]["muted"])

    def test_filtered_state_when_the_rule_points_against_the_28_day_move(self):
        cells = [dict(c) for c in self.snap["_cells"]]
        c0 = cells[0]
        c0.update(open=None, fresh=None, last=dict(c0["last"], base="long", side=None, trend_ok=False))
        out = s3.assemble(cells, self.now_ms, symbols=self.SYMS)
        cell = out["cells"][c0["sym"]][c0["tf"]]
        self.assertEqual((cell["state"], cell["state_fa"]), ("filtered", s3.STATE_FA["filtered"]))

    def test_brake_and_failed_test_mute_the_timeframe(self):
        cells = [dict(c) for c in self.snap["_cells"]]
        for c in cells:
            if c["tf"] == "4h":
                c["trades"] = [dict(x, net_r=-1.0) for x in c["trades"]]
        out = s3.assemble(cells, self.now_ms, symbols=self.SYMS)
        self.assertEqual(out["tfs"]["4h"]["brake"]["code"], "total_negative")
        self.assertTrue(all(out["cells"][s]["4h"]["muted"] for s in self.SYMS))
        self.assertTrue(all(s["muted"] for s in out["signals"] if s["tf"] == "4h"))
        failed = s3.assemble(self.snap["_cells"], self.now_ms, symbols=self.SYMS,
                             judgements=[{"look": "look_1", "result": "fail"}])
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(all(failed["tfs"][tf]["brake"]["muted"] for tf in s3.SIGNAL_TFS))
        self.assertEqual(failed["tfs"]["1d"]["brake"]["code"], "failed_test")
        passed = s3.assemble(self.snap["_cells"], self.now_ms, symbols=self.SYMS,
                             judgements=[{"look": "look_1", "result": "pass"}])
        self.assertEqual(passed["status"], "passed")

    def test_short_warmup_or_missing_data_is_an_error_cell_and_incomplete(self):
        def loader(sym, tf):
            if (sym, tf) == (self.SYMS[1], "1d"):
                return {k: v[1900:] for k, v in self.data[(sym, tf)].items()}     # فقط ۲۰۰ کندل پیش از T0
            if (sym, tf) == (self.SYMS[1], "4h"):
                raise RuntimeError("شبکه")
            return self.data[(sym, tf)]
        out = s3.build(loader, self.now_ms, symbols=self.SYMS)
        self.assertFalse(out["complete"])
        self.assertEqual(sorted(out["errors"]), [f"{self.SYMS[1]}|1d", f"{self.SYMS[1]}|4h"])
        self.assertIn("گرم‌شدن", out["errors"][f"{self.SYMS[1]}|1d"])

    def test_bars_needed_covers_warmup_and_grows_in_steps(self):
        self.assertEqual(s3.bars_needed("4h", T0), 2500)
        self.assertEqual(s3.bars_needed("4h", T0 + 500 * H4), 3000)
        self.assertEqual(s3.bars_needed("1d", T0 + 85 * D1), 2500)
        self.assertGreaterEqual(s3.bars_needed("4h", T0 + 3000 * H4), 2000 + 3000 + 60)


class LiveTests(unittest.TestCase):
    SYMS = ("AAAUSDT", "BBBUSDT")
    DAYS = 100                    # روزهای پس از T0؛ هر دو سری دقیقاً تا همین لحظه کندلِ بسته دارند

    @classmethod
    def setUpClass(cls):
        cls.data = {}
        for k, sym in enumerate(cls.SYMS):
            cls.data[(sym, "4h")] = synth(2100 + 6 * cls.DAYS, seed=40 + k, step_ms=H4, before=2100)
            cls.data[(sym, "1d")] = synth(2100 + cls.DAYS, seed=50 + k, step_ms=D1, before=2100)

    def setUp(self):
        for p in (s3.SNAPSHOT_PATH, s3.JUDGEMENT_PATH):
            if os.path.exists(p):
                os.remove(p)
        s3._cache.update(snap=None, tried=0.0)
        self.calls = []
        patcher = mock.patch.object(decision, "SYMBOLS", self.SYMS)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.now = (T0 + self.DAYS * D1 + 60_000) / 1000.0      # یک دقیقه پس از بسته‌شدنِ آخرین کندلِ هر دو

    def loader(self, sym, tf):
        self.calls.append((sym, tf))
        return self.data[(sym, tf)]

    def test_expected_last_closed_bar(self):
        self.assertEqual(s3.expected_last("4h", T0 + 5 * H4 + 1), T0 + 4 * H4)
        self.assertEqual(s3.expected_last("4h", T0 + 5 * H4), T0 + 4 * H4)
        self.assertEqual(s3.expected_last("1d", T0 + D1 - 1), T0 - D1)
        for tf in s3.SIGNAL_TFS:
            self.assertEqual(int(self.data[(self.SYMS[0], tf)]["t"][-1]), s3.expected_last(tf, self.now * 1000))

    def test_snapshot_is_built_once_per_closed_bar(self):
        now, calls = self.now, self.calls
        a = s3.snapshot(loader=self.loader, now=now)
        self.assertTrue(a["complete"])
        self.assertTrue(s3.is_current(a, int(now * 1000)))
        self.assertEqual(len(calls), 4)
        b = s3.snapshot(loader=self.loader, now=now + 600)             # همان کندل ⇒ بی‌ساختِ دوباره
        self.assertEqual(len(calls), 4)
        self.assertEqual(a["built_at"], b["built_at"])
        s3._cache.update(snap=None, tried=0.0)                         # راه‌اندازیِ دوباره: از دیسک
        c = s3.snapshot(loader=self.loader, now=now + 900)
        self.assertEqual(len(calls), 4)
        self.assertEqual(c["built_at"], a["built_at"])
        v = s3.view(loader=self.loader, now=now + 900)
        self.assertTrue(v["current"] and v["actionable"])
        # یک کندلِ 4h بعد: کارنامه دیگر جاری نیست ⇒ دوباره ساخته می‌شود؛ داده هنوز کندلِ تازه را ندارد ⇒ بی‌اعتبار
        later = now + 4 * 3600 + 5
        d = s3.view(loader=self.loader, now=later)
        self.assertEqual(len(calls), 8)
        self.assertFalse(d["current"])
        self.assertFalse(d["actionable"])
        s3.view(loader=self.loader, now=later + 5)                     # تا RETRY_SEC دوباره نمی‌سازد
        self.assertEqual(len(calls), 8)
        s3.view(loader=self.loader, now=later + s3.RETRY_SEC + 1)
        self.assertEqual(len(calls), 12)

    def test_failed_build_keeps_the_last_snapshot_as_stale(self):
        first = s3.snapshot(loader=self.loader, now=self.now, force=True)
        self.assertTrue(first["complete"])

        def boom(sym, tf):
            raise RuntimeError("no network")

        v = s3.view(loader=boom, now=self.now + 86400, force=True)
        self.assertTrue(v["stale"])
        self.assertFalse(v["actionable"])
        self.assertEqual(v["built_at"], first["built_at"])
        self.assertIn(f"{self.SYMS[0]}|4h", v["build_error"])

    def test_first_build_without_data_is_incomplete_and_never_actionable(self):
        def boom(sym, tf):
            raise RuntimeError("no network")

        v = s3.view(loader=boom, now=self.now)
        self.assertFalse(v["complete"])
        self.assertFalse(v["actionable"])
        self.assertEqual(len(v["errors"]), 4)
        self.assertEqual(v["signals"], [])
        self.assertFalse(s3.is_current(v, int(self.now * 1000)))

    def test_old_snapshot_is_not_actionable(self):
        snap = s3.snapshot(loader=self.loader, now=self.now, force=True)
        with mock.patch.object(s3, "is_current", lambda s, n: True):
            v = s3.view(loader=self.loader, now=self.now + s3.STALE_AFTER_SEC + 10)
        self.assertFalse(v["actionable"])
        self.assertEqual(v["built_at"], snap["built_at"])

    def test_judgement_is_written_once_and_only_when_due(self):
        cells = [{"trades": [trade(T0 + k * s3.WEEK_MS, 0.6) for k in range(50)]}]
        look1, look2 = s3.LOOKS
        self.assertIsNone(s3.record_judgement(cells, look1["due_ms"] - 1))
        self.assertEqual(s3.read_judgements(), [])
        row = s3.record_judgement(cells, look1["due_ms"])
        self.assertEqual((row["look"], row["result"]), ("look_1", "pass"))
        self.assertIsNone(s3.record_judgement(cells, look1["due_ms"] + 10))        # حکمِ دوم ممکن نیست
        self.assertIsNone(s3.record_judgement(cells, look2["due_ms"] + 10))        # پس از قبولی نگاهی نیست
        self.assertEqual(len(s3.read_judgements()), 1)
        with open(s3.JUDGEMENT_PATH, "a", encoding="utf-8") as f:                  # ردیفِ تکراری/خراب نادیده
            f.write(json.dumps({"look": "look_1", "result": "fail"}) + "\nnot json\n")
        self.assertEqual([j["result"] for j in s3.read_judgements()], ["pass"])

    def test_continue_leads_to_the_second_look(self):
        mixed = [{"trades": [trade(T0 + k * s3.WEEK_MS, 1.8 if k % 5 < 2 else -1.0) for k in range(50)]}]
        look1, look2 = s3.LOOKS
        self.assertEqual(s3.record_judgement(mixed, look1["due_ms"])["result"], "continue")
        self.assertIsNone(s3.record_judgement(mixed, look2["due_ms"] - 1))
        second = s3.record_judgement(mixed, look2["due_ms"])
        self.assertEqual(second["look"], "look_2")
        self.assertIn(second["result"], ("pass", "fail"))
        self.assertIsNone(s3.due_look(s3.read_judgements(), look2["due_ms"] + 10 ** 9))

    def test_a_failed_first_look_ends_the_test(self):
        bad = [{"trades": [trade(T0 + k * s3.WEEK_MS, -0.2) for k in range(50)]}]
        self.assertEqual(s3.record_judgement(bad, s3.LOOKS[0]["due_ms"])["result"], "fail")
        self.assertIsNone(s3.due_look(s3.read_judgements(), s3.LOOKS[1]["due_ms"] + 1))


if __name__ == "__main__":
    unittest.main()
