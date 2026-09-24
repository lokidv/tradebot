# -*- coding: utf-8 -*-
"""تصمیمِ لانگ/شورت/صبر: همان قاعدهٔ موتور، kNN علّیِ برابر با engine.knn_ml، بازپخشِ بی‌نگاه‌به‌آینده،
دفترِ فقط‌افزودنی و برچسبِ کارنامه."""
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import bracket  # noqa: E402
import decision  # noqa: E402
import engine  # noqa: E402
import gates  # noqa: E402

H_MS = 3_600_000
T0 = 472_222 * H_MS            # مرزِ ساعتِ UTC


def synth(n=1200, seed=3, step_ms=H_MS):
    """گامِ تصادفی با رژیم‌های روند/رنج تا ستاپ‌ها و قاعدهٔ z هر دو رخ دهند."""
    rng = np.random.RandomState(seed)
    drift = np.repeat(rng.choice([-1.0, 0.0, 1.0], n // 150 + 1), 150)[:n] * 0.0015
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.0005, n))
    wig = np.abs(rng.normal(0, 0.003, n)) * c
    return {"t": T0 + np.arange(n, dtype=np.int64) * step_ms,
            "o": o, "h": np.maximum(o, c) + wig, "l": np.minimum(o, c) - wig, "c": c,
            "v": rng.lognormal(10, 0.5, n)}


def _as_lists(kl):
    return {k: np.asarray(v).tolist() for k, v in kl.items()}


def _row(side=None, **kw):
    a = {"symbol": "BTCUSDT", "tf": "1h", "zt": T0, "price": 100.0, "z": 1.8,
         "votes_bull": 4, "votes_bear": 0, "regime": "رونددار",
         "components": {"روند": 0.5, "مومنتوم": 0.3, "حجم": 0.1, "ساختار": 0.6, "هوش مصنوعی": 0.4},
         "trade": {"side": side}}
    if side:
        a["trade"].update(grade="A", entry=100.0, sl=98.7, tp=102.34, atr14=1.0, rr=1.8, risk_pct=1.3,
                          setup="zx", setup_observed=False)
    a.update(kw)
    return a


class FromAnalysisTests(unittest.TestCase):
    def test_long_from_the_z_rule_has_no_setup(self):
        d = decision.from_analysis(_row("long"))
        self.assertEqual(d["action"], "long")
        self.assertEqual(d["lean"], "long")
        self.assertIsNone(d["setup"])                    # «zx»ِ پیش‌فرضِ analyze ستاپ نیست
        self.assertEqual(d["basis"], "rule")
        self.assertEqual(d["strength"], round(50 + 25 * 1.8 / 2.5 + 25 * 4 / 5))
        self.assertEqual([c["vote"] for c in d["components"]], [1, 1, 0, 1, 1])
        self.assertEqual([c["key"] for c in d["components"]], ["trend", "mom", "vol", "struct", "ai"])
        self.assertEqual((d["entry"], d["sl"], d["tp"], d["risk_pct"], d["atr"]), (100.0, 98.7, 102.34, 1.3, 1.0))
        self.assertEqual(d["candle_ts"], T0)
        self.assertTrue(d["reasons"])

    def test_short_with_an_observed_setup(self):
        a = _row("short", z=-1.4, votes_bull=0, votes_bear=3,
                 components={"روند": -0.5, "مومنتوم": -0.3, "حجم": 0.0, "ساختار": -0.6, "هوش مصنوعی": 0.0})
        a["trade"].update(setup="pb", setup_observed=True, sl=101.3, tp=97.66)
        d = decision.from_analysis(a)
        self.assertEqual(d["action"], "short")
        self.assertEqual(d["setup"], "pb")
        self.assertEqual(d["setup_fa"], engine.SETUP_FA["pb"])
        self.assertEqual(d["basis"], "setup")
        self.assertEqual(d["strength"], round(50 + 25 * 1.4 / 2.5 + 25 * 3 / 5))
        self.assertIn("پولبک", d["reasons"][0])

    def test_opposite_z_adds_no_strength(self):
        a = _row("long", z=-0.8)
        a["trade"].update(setup="fd", setup_observed=True)
        d = decision.from_analysis(a)
        self.assertEqual(d["strength"], round(50 + 25 * 4 / 5))
        self.assertTrue(any("خلافِ این جهت" in r for r in d["reasons"]))

    def test_wait_reports_readiness_and_what_is_missing(self):
        a = _row(None, z=0.7, votes_bull=2, votes_bear=1)
        a["trade"].update(readiness=62, ready_side="long")
        d = decision.from_analysis(a)
        self.assertEqual(d["action"], "wait")
        self.assertEqual(d["lean"], "long")
        self.assertEqual((d["readiness"], d["strength"]), (62, 62))
        self.assertIsNone(d["entry"])
        self.assertIsNone(d["risk_pct"])
        self.assertIsNone(d["grade"])
        self.assertTrue(any("1.2" in r for r in d["reasons"]))
        self.assertTrue(any("۳ لازم" in r for r in d["reasons"]))

    def test_low_readiness_has_no_lean(self):
        a = _row(None, z=0.1, votes_bull=1, votes_bear=1)
        a["trade"].update(readiness=20, ready_side="short")
        self.assertIsNone(decision.from_analysis(a)["lean"])

    def test_error_row_is_wait_with_the_error_as_reason(self):
        d = decision.from_analysis({"symbol": "SOLUSDT", "tf": "4h", "error": "فیدِ داده مرده است"})
        self.assertEqual(d["action"], "wait")
        self.assertEqual(d["reasons"], ["فیدِ داده مرده است"])
        self.assertEqual((d["sym"], d["tf"]), ("SOLUSDT", "4h"))

    def test_rounded_threshold_values_follow_the_engine_totals(self):
        comps = {"روند": 0.30, "مومنتوم": 0.4, "حجم": 0.0, "ساختار": 0.6, "هوش مصنوعی": 0.1}
        d2 = decision.from_analysis(_row(None, votes_bull=2, votes_bear=0, components=comps))
        d3 = decision.from_analysis(_row(None, votes_bull=3, votes_bear=0, components=comps))
        self.assertEqual(d2["components"][0]["vote"], 0)     # ۰٫۲۹۶ گردشده به ۰٫۳۰ — موتور رأی نداده
        self.assertEqual(d3["components"][0]["vote"], 1)

    def test_ai_vote_is_exact_and_only_continuous_values_are_reconciled(self):
        # روند ۰٫۲۹۶ → ۰٫۳۰ (مبهم)، kNN دقیقاً ۰٫۲ (روی شبکهٔ ۰٫۱ — رأی قطعی)؛ موتور ۳ رأی شمرده:
        # مومنتوم + ساختار + kNN. رأیِ سوم مالِ kNN است، نه روند.
        comps = {"روند": 0.30, "مومنتوم": 0.4, "حجم": 0.0, "ساختار": 0.6, "هوش مصنوعی": 0.2}
        d = decision.from_analysis(_row(None, votes_bull=3, votes_bear=0, components=comps))
        votes = {c["key"]: c["vote"] for c in d["components"]}
        self.assertEqual((votes["ai"], votes["trend"]), (1, 0))
        self.assertEqual(sum(v == 1 for v in votes.values()), 3)
        neg = {"روند": -0.30, "مومنتوم": -0.4, "حجم": 0.0, "ساختار": -0.6, "هوش مصنوعی": -0.2}
        d = decision.from_analysis(_row(None, votes_bull=0, votes_bear=3, components=neg))
        votes = {c["key"]: c["vote"] for c in d["components"]}
        self.assertEqual((votes["ai"], votes["trend"]), (-1, 0))

    def test_engine_blocks_and_cautions_become_warnings(self):
        status = "مسدود — مخالفتِ شدید با روند بیت‌کوین (z=-1.1)"
        dw = "جهت‌یاب کمکی مخالف است (احتمال رشد 40٪)"
        a = _row("long")
        a["trade"].update(status=status, btc_align=False, direction_warning=dw, risk_pct=0.4)
        d = decision.from_analysis(a)
        w = d["warnings"]
        self.assertEqual(w[0], status)
        self.assertIn(dw, w)
        self.assertTrue(any("۰٫۳۵R" in x for x in w))                   # هزینه ۰٫۱۴٪ روی حدضرر ۰٫۴٪
        self.assertLessEqual(len(w), decision.MAX_WARNINGS)
        self.assertEqual(len(w), len(set(w)))
        self.assertTrue(d["scorecard_applies"])

    def test_generic_status_is_not_a_warning_and_duplicates_collapse(self):
        a = _row("long")
        a["trade"]["status"] = "منتظر یکی از ستاپ‌های آموزش‌دیده؛ جهت یا امتیاز تکنیکال به‌تنهایی مجوز ورود نیست"
        self.assertEqual(decision.from_analysis(a)["warnings"], [])
        status = "مسدود — مخالفتِ شدید با روند بیت‌کوین (z=-1.1)"
        a = _row("long", btc_z=-1.1)
        a["trade"].update(status=status, btc_align=False)
        self.assertEqual(decision.from_analysis(a)["warnings"], [status])
        knife = "مشاهده — بازگشتِ خلافِ روند در بازار نزولی بیت‌کوین (چاقوی سقوط)"
        w = _row(None)
        w["trade"].update(status=knife, readiness=40, ready_side="long")
        self.assertEqual(decision.from_analysis(w)["warnings"], [knife])

    def test_at_most_four_warnings(self):
        a = _row("long", btc_z=-1.5, tf_suspended=-0.12)
        a["trade"].update(status="ردِ سیاست نهایی — امتیاز زیر آستانه است", regime_veto=True,
                          direction_warning="جهت‌یاب کمکی خنثی است (50٪)", meta_blocks=["فاندینگ شلوغ"],
                          risk_pct=0.3)
        w = decision.from_analysis(a)["warnings"]
        self.assertEqual(len(w), 4)
        self.assertTrue(w[0].startswith("ردِ سیاست"))

    def test_policy_path_is_not_covered_by_the_scorecard(self):
        a = _row("long")
        a["trade"].update(setup="ap", setup_observed=False)
        d = decision.from_analysis(a)
        self.assertEqual(d["basis"], "policy")
        self.assertFalse(d["scorecard_applies"])
        self.assertTrue(any("سیاست" in x for x in d["warnings"]))
        err = decision.from_analysis({"symbol": "SOLUSDT", "tf": "4h", "error": "x"})
        self.assertEqual((err["warnings"], err["scorecard_applies"]), ([], True))

    def test_collect_turns_a_missing_analysis_into_an_error(self):
        def ga(sym, tf):
            if sym == "SOLUSDT":
                raise RuntimeError("boom")
            return None if sym == "BTCUSDT" else {}
        out = decision.collect(ga, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"), tfs=("1h",))
        self.assertEqual([d["error"] for d in out], ["تحلیل در دسترس نیست", "تحلیل در دسترس نیست", "boom"])
        self.assertTrue(all(d["action"] == "wait" and d["sym"] and d["tf"] == "1h" for d in out))


class LiveRuleParityTests(unittest.TestCase):
    """بازپخش روی همان پنجرهٔ ۴۲۰ کندلی که موتور می‌بیند باید دقیقاً همان جهت را بدهد."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = gates.GATES_PATH
        gates.GATES_PATH = os.path.join(self.tmp.name, "gates.json")
        with open(gates.GATES_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(gates.DEFAULTS), f)
        gates.load_gates(force=True)

    def tearDown(self):
        gates.GATES_PATH = self._old
        gates._cache.update(mtime=None, data=None)
        self.tmp.cleanup()

    def test_side_series_matches_engine_analyze(self):
        kl = synth(1300, seed=11)
        sides = []
        for end in range(430, 1300, 29):
            w = {k: v[end - decision.LIVE_BARS:end] for k, v in kl.items()}
            a = engine.analyze(_as_lists(w), "1h")
            side, setup, _ = decision.side_series(w, decision.LIVE_BARS - 1)
            live = {"long": 1, "short": -1}.get(a["trade"].get("side"), 0)
            self.assertEqual(int(side[-1]), live, f"end={end}")
            d = decision.from_analysis(dict(a, symbol="BTCUSDT"))
            self.assertEqual(sum(c["vote"] == 1 for c in d["components"]), a["votes_bull"])
            self.assertEqual(sum(c["vote"] == -1 for c in d["components"]), a["votes_bear"])
            if live:
                self.assertEqual(d["setup"], setup[-1])
            sides.append(live)
        self.assertTrue(any(sides) and not all(sides))   # آزمون بی‌محتوا نیست


class KnnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kl = synth(900, seed=7)
        cls.kl = kl
        cls.cs = engine.component_series(kl["o"], kl["h"], kl["l"], kl["c"], kl["v"])

    def _prefix_ref(self, i, **kw):
        kl = {k: v[:i + 1] for k, v in self.kl.items()}
        csp = engine.component_series(kl["o"], kl["h"], kl["l"], kl["c"], kl["v"])
        return engine.knn_ml(kl["h"], kl["l"], kl["c"], csp, **kw)

    def test_equals_engine_knn_on_prefixes(self):
        idx = [50, 70, 125, 222, 419, 600, 899]
        got = decision.knn_series(self.kl["h"], self.kl["l"], self.kl["c"], self.cs, idx=idx)
        got_small = decision.knn_series(self.kl["h"], self.kl["l"], self.kl["c"], self.cs, idx=idx,
                                        max_samples=40)
        vals = []
        for p, i in enumerate(idx):
            self.assertEqual(got[p], self._prefix_ref(i), f"i={i}")
            self.assertEqual(got_small[p], self._prefix_ref(i, max_samples=40), f"i={i}")
            vals.append(got[p])
        self.assertTrue(any(v != 0 for v in vals))

    def test_window_mode_uses_only_the_live_window(self):
        h, l, c = self.kl["h"], self.kl["l"], self.kl["c"]
        w = 300
        full = decision.knn_series(h, l, c, self.cs, window=None)
        win = decision.knn_series(h, l, c, self.cs, window=w)
        np.testing.assert_array_equal(win[:w], full[:w])      # پنجره هنوز کلِ پیشوند است
        f = decision.knn_features(h, l, c, self.cs)
        for i in (450, 777, 899):                            # مرجعِ حلقه‌ای، عیناً مثلِ knn_ml روی پنجره
            s = i - w + 1
            dists, labels = [], []
            for j in range(s + 60, i - 4, 2):
                lbl = np.sign(c[j + 4] - c[j])
                if lbl == 0:
                    continue
                dists.append(float(np.log1p(np.abs(f[j] - f[i])).sum()))
                labels.append(lbl)
            ref = float(np.mean(np.array(labels)[np.argsort(dists)[:10]]))
            self.assertEqual(win[i], ref, f"i={i}")


class ReplayTests(unittest.TestCase):
    def test_no_lookahead(self):
        kl = synth(1400, seed=5)
        d = 1000
        base = decision.replay(kl, first=420)
        rng = np.random.RandomState(99)
        pert = {k: np.array(v, copy=True) for k, v in kl.items()}
        shock = np.exp(np.cumsum(rng.normal(0, 0.02, len(pert["c"]) - d - 1)))
        for k in ("o", "h", "l", "c"):
            pert[k][d + 1:] *= shock
        pert["v"][d + 1:] = rng.lognormal(12, 1.0, len(shock))
        other = decision.replay(pert, first=420)
        np.testing.assert_array_equal(base["side"][:d + 1], other["side"][:d + 1])
        self.assertEqual(base["setup"][:d + 1], other["setup"][:d + 1])
        a = [(x["i"], x["side"], x["setup"]) for x in base["trades"] if x["i"] <= d]
        b = [(x["i"], x["side"], x["setup"]) for x in other["trades"] if x["i"] <= d]
        self.assertEqual(a, b)
        self.assertGreater(len(a), 3)
        ea = [x["entry"] for x in base["trades"] if x["i"] < d]
        eb = [x["entry"] for x in other["trades"] if x["i"] < d]
        self.assertEqual(ea, eb)

    def test_one_trade_at_a_time_and_net_cost(self):
        rep = decision.replay(synth(1400, seed=8), first=420)
        tr = rep["trades"]
        self.assertGreater(len(tr), 3)
        for prev, nxt in zip(tr, tr[1:]):
            self.assertGreaterEqual(nxt["i"], prev["exit_idx"])
        for x in tr:
            self.assertAlmostEqual(x["net_r"], bracket.net_r(x["gross_r"], x["risk_pct"], 0.14))
            self.assertEqual(x["side"], "long" if rep["side"][x["i"]] == 1 else "short")

    def test_resample_uses_utc_buckets(self):
        n = 4 * 30 + 4                                       # ۳۰ ساعتِ کامل + دو سطلِ ناتمام در دو سر
        kl = synth(n, seed=1, step_ms=900_000)
        kl["t"] = kl["t"] + 2 * 900_000                       # شروع از وسطِ ساعت: سطلِ اول ناقص است
        r = decision.resample(kl, 60)
        self.assertTrue(np.all(r["t"] % H_MS == 0))
        self.assertEqual(len(r["t"]), 30)                     # سطلِ ناقصِ اول و آخر حذف شد
        self.assertEqual(r["o"][0], kl["o"][2])
        self.assertEqual(r["c"][0], kl["c"][5])
        self.assertEqual(r["h"][0], kl["h"][2:6].max())
        self.assertEqual(r["l"][0], kl["l"][2:6].min())
        self.assertAlmostEqual(r["v"][0], kl["v"][2:6].sum())


class ScorecardTests(unittest.TestCase):
    def test_verdict_thresholds(self):
        V = decision.VERDICT_FA
        self.assertEqual(decision.verdict(29, 0.5, 0.01), V["small"])
        self.assertEqual(decision.verdict(0, None), V["small"])
        self.assertEqual(decision.verdict(30, 0.1, None), V["small"])
        self.assertEqual(decision.verdict(30, -0.1, 0.04), V["loss"])        # −0.1 + 1.96×0.04 < 0
        self.assertEqual(decision.verdict(30, -0.1, 0.06), V["unclear"])     # بازه صفر را می‌گیرد
        self.assertEqual(decision.verdict(30, 0.1, 0.05), V["positive"])     # 0.1 − 0.098 > 0
        self.assertEqual(V, {"small": "نمونهٔ کم", "loss": "زیان‌ده", "positive": "مثبت (تأییدنشده)",
                             "unclear": "نامشخص — بازه شامل صفر"})

    def test_verdict_uses_the_confidence_interval_of_net_r(self):
        loss = [-1.1] * 140 + [1.7] * 60                     # میانگین −0.26R روی ۲۰۰ معامله
        s = decision._stats(loss)
        se = float(np.std(loss, ddof=1) / np.sqrt(len(loss)))
        self.assertAlmostEqual(s["se"], round(se, 4))
        self.assertAlmostEqual(s["ci_lo"], round(np.mean(loss) - 1.96 * se, 4))
        self.assertAlmostEqual(s["ci_hi"], round(np.mean(loss) + 1.96 * se, 4))
        self.assertLess(s["ci_hi"], 0)
        self.assertEqual((s["verdict_code"], s["verdict"]), ("loss", "زیان‌ده"))
        noisy = [1.7] * 14 + [-1.1] * 18                      # +0.125R ولی فقط ۳۲ معامله
        s = decision._stats(noisy)
        self.assertGreater(s["avg_r"], 0)
        self.assertLess(s["ci_lo"], 0)
        self.assertEqual(s["verdict_code"], "unclear")
        self.assertEqual(decision._stats([1.7] * 29)["verdict_code"], "small")
        self.assertEqual(decision._stats([1.7] * 60 + [-1.1] * 40)["verdict_code"], "positive")
        e = decision._stats([])
        self.assertEqual((e["n"], e["se"], e["verdict_code"]), (0, None, "small"))

    def test_replay_uses_the_engine_setup_lookback(self):
        self.assertEqual(decision.SETUP_LOOKBACK, engine.SETUP_LOOKBACK)

    def test_short_fallback_history_is_an_error_cell(self):
        fake = types.SimpleNamespace(get_history=lambda sym, tf, bars=3000: synth(300))
        self.assertFalse(os.path.exists(decision.MICRO_DIR))           # آرشیوِ محلی در تست نیست
        with mock.patch.dict(sys.modules, {"market": fake}):
            with self.assertRaises(ValueError) as cm:
                decision.load_history("BTCUSDT", "1h")
            self.assertIn(str(decision.MIN_BARS), str(cm.exception))
            sc = decision.build_scorecards(symbols=("BTCUSDT",), tfs=("1h",))
        cell = sc["cells"]["BTCUSDT"]["1h"]
        self.assertIn("300", cell["error"])
        self.assertEqual((cell["n"], cell["verdict_code"]), (0, "small"))
        self.assertIsNone(sc["window_from"])
        fake.get_history = lambda sym, tf, bars=3000: synth(900)
        with mock.patch.dict(sys.modules, {"market": fake}):
            kl, src = decision.load_history("BTCUSDT", "1h")
        self.assertEqual((len(kl["t"]), src), (900, decision.SOURCE_FALLBACK))

    def test_summarize(self):
        tr = [{"net_r": 1.5, "gross_r": 1.6, "risk_pct": 1.4, "side": "long", "setup": "pb"},
              {"net_r": -1.1, "gross_r": -1.0, "risk_pct": 1.4, "side": "short", "setup": None},
              {"net_r": -0.1, "gross_r": 0.0, "risk_pct": 1.4, "side": "long", "setup": None}]
        s = decision.summarize(tr, 0.14, T0, T0 + 86_400_000)
        self.assertEqual((s["n"], s["long_n"], s["short_n"], s["setup_n"], s["rule_n"]), (3, 2, 1, 1, 2))
        self.assertAlmostEqual(s["win_rate"], 33.3)
        self.assertAlmostEqual(s["profit_factor"], round(1.5 / 1.2, 2))
        self.assertAlmostEqual(s["avg_r"], round(0.3 / 3, 4))
        self.assertEqual(s["verdict"], "نمونهٔ کم")
        self.assertEqual(s["cost_pct"], 0.14)

    def test_cache_is_rebuilt_at_most_once_a_day(self):
        self.assertTrue(decision.SCORECARD_PATH.startswith(_hermetic.DATA_DIR))
        calls = []

        def loader(sym, tf):
            calls.append((sym, tf))
            return synth(900, seed=len(calls)), "test"

        sc = decision.scorecards(force=True, loader=loader)
        per_build = len(decision.SYMBOLS) * len(decision.TFS)                # ۵ ارز × ۵ تایم‌فریم (با 5m)
        self.assertEqual(per_build, 25)
        self.assertEqual(len(calls), per_build)
        self.assertEqual(sc["version"], decision.SCORECARD_VERSION)
        self.assertGreaterEqual(decision.SCORECARD_VERSION, 3)
        self.assertEqual((sc["cost_pct"], sc["maker_cost_pct"]), (0.14, 0.10))
        self.assertTrue(os.path.exists(decision.SCORECARD_PATH))
        cell = sc["cells"]["BTCUSDT"]["1h"]
        for key in ("n", "win_rate", "avg_r", "profit_factor", "total_r", "long_n", "short_n",
                    "from", "to", "cost_pct", "verdict", "verdict_code", "se", "ci_lo", "ci_hi",
                    "avg_gross_r", "avg_cost_r"):
            self.assertIn(key, cell)
        self.assertEqual(cell["cost_pct"], 0.14)
        self.assertIn(cell["verdict_code"], ("loss", "positive", "unclear", "small"))
        self.assertEqual(sc["source"], "test")
        self.assertEqual(sc["window_to"], decision._date(int(synth(900)["t"][-1])))
        self.assertLessEqual(sc["window_from"], cell["from"])
        self.assertGreater(sc["window_days_actual"], 0)
        self.assertLess(sc["window_days_actual"], decision.WINDOW_DAYS)  # پنجرهٔ واقعی، نه اسمی
        # گونهٔ ثانویِ میکر: همان معامله‌ها، هزینهٔ کمتر؛ قراردادِ اصلی دست‌نخورده
        mk = sc["cells"]["BTCUSDT"]["5m"]["maker"]
        self.assertEqual(set(mk), set(decision.MAKER_KEYS))
        self.assertEqual(mk["cost_pct"], 0.10)
        self.assertEqual(sc["cells"]["BTCUSDT"]["5m"]["cost_pct"], 0.14)
        again = decision.scorecards(loader=loader, now=sc["built_at"] + 3600)
        self.assertEqual(len(calls), per_build)
        self.assertEqual(again["built_at"], sc["built_at"])
        decision.scorecards(loader=loader, now=sc["built_at"] + 86_401)
        self.assertEqual(len(calls), 2 * per_build)

    def test_maker_variant_is_the_same_trades_at_lower_cost(self):
        rep = decision.replay(synth(1500, seed=11), first=decision.LIVE_BARS)
        self.assertGreater(len(rep["trades"]), 3)
        tk = decision.summarize(rep["trades"], decision.COST_PCT)
        mk = decision.maker_variant(rep["trades"])
        self.assertEqual(mk["n"], tk["n"])
        self.assertGreater(mk["avg_r"], tk["avg_r"])
        want = np.mean([bracket.net_r(x["gross_r"], x["risk_pct"], 0.10) for x in rep["trades"]])
        self.assertAlmostEqual(mk["avg_r"], round(float(want), 4), places=4)
        self.assertEqual(decision.maker_variant([])["verdict_code"], "small")

    def test_load_history_reads_the_native_archive_and_never_mislabels(self):
        tmp = tempfile.mkdtemp(prefix="micro-")

        def write(tf, n):
            kl = synth(n, step_ms=decision.TF_MINUTES[tf] * 60_000)
            np.savez(os.path.join(tmp, f"um_BTCUSDT_{tf}.npz"), **{k: np.asarray(v) for k, v in kl.items()})

        write("5m", 1500)
        write("15m", 1500)
        fake = types.SimpleNamespace(get_history=lambda sym, tf, bars=3000: synth(900))
        with mock.patch.object(decision, "MICRO_DIR", tmp), mock.patch.dict(sys.modules, {"market": fake}):
            kl, src = decision.load_history("BTCUSDT", "5m")
            self.assertEqual(src, "binance-futures-5m-archive")
            self.assertEqual(int(np.median(np.diff(kl["t"]))), 300_000)          # کندلِ واقعیِ ۵ دقیقه
            kl, src = decision.load_history("BTCUSDT", "15m")
            self.assertEqual((src, int(np.median(np.diff(kl["t"])))), ("binance-futures-15m-archive", 900_000))
            kl, src = decision.load_history("BTCUSDT", "1h")                     # بومی نیست ⇒ تجمیعِ ریزتر
            self.assertEqual(src, "binance-futures-15m-archive→1h")
            self.assertEqual(int(np.median(np.diff(kl["t"]))), 3_600_000)
            os.remove(os.path.join(tmp, "um_BTCUSDT_5m.npz"))
            kl, src = decision.load_history("BTCUSDT", "5m")                     # 15m هرگز «5m» نمی‌شود
            self.assertEqual(src, decision.SOURCE_FALLBACK)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        if os.path.exists(decision.LEDGER_PATH):
            os.remove(decision.LEDGER_PATH)
        self.now = (T0 + 1.5 * H_MS) / 1000

    def _lines(self):
        with open(decision.LEDGER_PATH, encoding="utf-8") as f:
            return f.readlines()

    @staticmethod
    def _klines(n_after=6):
        t = [T0 - 2 * H_MS, T0 - H_MS, T0] + [T0 + (k + 1) * H_MS for k in range(n_after)]
        o = [100.0] * len(t)
        h = [100.5] * len(t)
        l = [99.5] * len(t)
        c = [100.0] * len(t)
        h[4], c[4] = 103.0, 102.8                              # کندلِ دومِ پس از ورود به هدف می‌رسد
        return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": [1.0] * len(t)}

    def test_append_only_dedup_and_resolve(self):
        self.assertTrue(decision.LEDGER_PATH.startswith(_hermetic.DATA_DIR))
        long_d = decision.from_analysis(_row("long"))
        wait_d = decision.from_analysis(_row(None))
        self.assertEqual(decision.record([long_d, wait_d], now=self.now), 1)
        self.assertEqual(decision.record([long_d], now=self.now), 0)          # بی‌تکرار
        self.assertEqual(len(self._lines()), 1)
        first = self._lines()[0]
        self.assertEqual(json.loads(first)["id"], f"BTCUSDT|1h|{T0}|long")

        self.assertEqual(decision.resolve(lambda s, tf: self._klines(), now=self.now + 7200), 1)
        self.assertEqual(decision.resolve(lambda s, tf: self._klines(), now=self.now + 7200), 0)
        lines = self._lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], first)                                    # هیچ خطی بازنویسی نشد
        res = json.loads(lines[1])
        self.assertEqual((res["kind"], res["outcome"]), ("result", "target"))
        self.assertEqual(res["exit_ts"], T0 + 2 * H_MS)

        st = decision.live_stats()
        cell = st["cells"]["BTCUSDT"]["1h"]
        self.assertEqual((cell["n"], cell["win_rate"], cell["open"]), (1, 100.0, 0))
        self.assertAlmostEqual(cell["avg_r"], round(1.8 - 0.14 / 1.3, 4), places=3)
        self.assertEqual(st["overall"]["n"], 1)

    def test_overlapping_and_open_signals(self):
        d1 = decision.from_analysis(_row("long"))
        d2 = decision.from_analysis(_row("long", zt=T0 + H_MS))             # پیش از خروجِ اولی
        d3 = decision.from_analysis(_row("short", tf="4h"))                   # هنوز داوری‌نشدنی
        d3["sl"] = 101.3
        old = decision.from_analysis(_row("long", zt=T0 - 50 * H_MS))         # فیدِ مرده
        self.assertEqual(decision.record([d1, d3, old], now=self.now), 2)
        self.assertEqual(decision.record([d2], now=self.now + 3600), 1)

        def kl(sym, tf):
            return self._klines() if tf == "1h" else {k: v[:3] for k, v in self._klines().items()}

        decision.resolve(kl, now=self.now + 7200)
        st = decision.live_stats()
        c1 = st["cells"]["BTCUSDT"]["1h"]
        self.assertEqual((c1["n"], c1["signals"], c1["overlapping"]), (1, 2, 1))
        c4 = st["cells"]["BTCUSDT"]["4h"]
        self.assertEqual((c4["n"], c4["open"]), (0, 1))
        self.assertEqual(st["overall"]["open"], 1)

    def test_entry_bar_must_be_exactly_the_next_bar(self):
        decision.record([decision.from_analysis(_row("long"))], now=self.now)
        kl = self._klines()
        gap = {k: v[:3] + v[4:] for k, v in kl.items()}                      # کندلِ T0+1h نیست
        self.assertEqual(decision.resolve(lambda s, tf: gap, now=self.now + 7200), 1)
        res = json.loads(self._lines()[1])
        self.assertEqual(res["skipped"], "no_data")
        self.assertNotIn("net_r", res)
        self.assertEqual(decision.live_stats()["cells"]["BTCUSDT"]["1h"]["n"], 0)

    def test_concurrent_resolve_cannot_duplicate_a_result(self):
        decision.record([decision.from_analysis(_row("long"))], now=self.now)
        inner = []

        def gk(sym, tf):
            if not inner:                       # فراخوانیِ دیگری وسطِ کار همین ردیف را داوری می‌کند
                inner.append(decision.resolve(lambda s, t: self._klines(), now=self.now + 7200))
            return self._klines()

        self.assertEqual(decision.resolve(gk, now=self.now + 7200), 0)
        self.assertEqual(inner, [1])
        self.assertEqual(len(self._lines()), 2)

    def _write_malformed(self):
        ok = {"kind": "open", "id": f"BTCUSDT|1h|{T0}|long", "sym": "BTCUSDT", "tf": "1h", "side": "long",
              "candle_ts": T0, "entry": 100.0, "sl": 98.7, "tp": 102.34, "atr": 1.0, "risk_pct": 1.3,
              "cost_pct": 0.14}
        rows = [
            '{"kind": "open", "id": "x"}',                                                   # بی sym/tf/ts/side
            json.dumps({"kind": "open", "id": "BTCUSDT|1h|abc|long", "sym": "BTCUSDT", "tf": "1h",
                        "side": "long", "candle_ts": "abc"}),
            json.dumps({"kind": "open", "id": "BTCUSDT|7m|1|long", "sym": "BTCUSDT", "tf": "7m",
                        "side": "long", "candle_ts": 1}),
            json.dumps({"kind": "open", "id": "BTCUSDT|4h|1|up", "sym": "BTCUSDT", "tf": "4h",
                        "side": "up", "candle_ts": 1}),
            json.dumps(dict(ok, id=f"BTCUSDT|1h|{T0 - H_MS}|short", side="short", candle_ts=T0 - H_MS,
                            atr="x")),                                                    # ATR خراب
            json.dumps({"kind": "open", "id": "BTCUSDT|4h|1|long", "sym": "BTCUSDT", "tf": "4h",
                        "side": "long", "candle_ts": 1, "atr": 1.0}),
            json.dumps({"kind": "result", "id": "BTCUSDT|4h|1|long", "net_r": "bad", "exit_ts": "zz"}),
            json.dumps(dict(ok, id=f"ETHUSDT|1h|{T0}|long", sym="ETHUSDT")),               # get_klines خطا
            "[1, 2]", '"just a string"', "not json", "{\"kind\": \"open\", \"id\": 5, \"sym\": null}",
            json.dumps(ok),
        ]
        with open(decision.LEDGER_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")

    def test_malformed_ledger_rows_never_raise(self):
        self._write_malformed()

        def gk(sym, tf):
            if sym == "ETHUSDT":
                raise ConnectionError("down")
            return self._klines()

        self.assertEqual(decision.resolve(gk, now=self.now + 7200), 2)          # T0 هدف + ردیفِ ATR خراب
        results = [json.loads(x) for x in self._lines() if '"result"' in x]
        by = {r["id"]: r for r in results}
        self.assertEqual(by[f"BTCUSDT|1h|{T0}|long"]["outcome"], "target")
        self.assertEqual(by[f"BTCUSDT|1h|{T0 - H_MS}|short"]["skipped"], "bad_row")
        st = decision.live_stats()
        self.assertEqual(st["cells"]["BTCUSDT"]["1h"]["n"], 1)
        self.assertEqual(st["cells"]["BTCUSDT"]["4h"]["n"], 0)                    # net_r خراب شمرده نشد
        self.assertEqual(st["cells"]["ETHUSDT"]["1h"]["open"], 1)
        self.assertEqual(decision.resolve(gk, now=self.now + 7200), 0)

    def test_refresh_with_stubs_and_a_malformed_ledger(self):
        self._write_malformed()
        recent = (int(time.time() * 1000) // 300_000) * 300_000 - 300_000   # تازه حتی برای 5m (STALE_BARS)

        def ga(sym, tf):
            if sym == "ETHUSDT":
                return None
            if sym == "SOLUSDT":
                raise RuntimeError("boom")
            return _row("long", symbol=sym, tf=tf, zt=recent)

        def gk(sym, tf):
            if sym != "BTCUSDT":
                raise ConnectionError("down")
            return self._klines()

        p = decision.refresh(ga, gk, scorecard={"cells": {}, "building": True})
        json.dumps(p, ensure_ascii=False)
        self.assertEqual(p["cells"]["ETHUSDT"]["1h"]["error"], "تحلیل در دسترس نیست")
        self.assertEqual(p["cells"]["SOLUSDT"]["4h"]["error"], "boom")
        btc = p["cells"]["BTCUSDT"]["1h"]
        self.assertEqual(btc["action"], "long")
        self.assertIsInstance(btc["warnings"], list)
        self.assertTrue(btc["scorecard_applies"])
        self.assertEqual((btc["live"]["n"], btc["live"]["open"]), (1, 1))      # T0 داوری شد، تازه باز است
        self.assertTrue(p["scorecard_meta"]["building"])
        self.assertEqual(p["scorecard_meta"]["cost_pct"], 0.14)
        opens = [json.loads(x) for x in self._lines() if '"open"' in x and "BNBUSDT" in x]
        self.assertEqual(len(opens), len(decision.TFS))                           # یک پیشنهاد برای هر تایم‌فریم (با 5m)


class PayloadTests(unittest.TestCase):
    def test_cells_carry_decision_scorecard_and_live(self):
        ds = [decision.from_analysis(_row("long")), decision.from_analysis(_row(None, tf="4h"))]
        sc = {"built_at": 1.0, "cells": {"BTCUSDT": {"1h": {"n": 40, "verdict": "زیان‌ده"}}}}
        lv = {"cells": {}, "overall": {"n": 0}}
        p = decision.payload(ds, scorecard=sc, live=lv)
        self.assertEqual(p["coins"], list(decision.SYMBOLS))
        self.assertEqual(p["tfs"], ["5m", "15m", "1h", "4h", "1d"])          # کوتاه→بلند، 5m اول
        self.assertEqual(p["tfs"], list(decision.TFS))
        cell = p["cells"]["BTCUSDT"]["1h"]
        self.assertEqual(cell["action"], "long")
        self.assertEqual(cell["scorecard"]["verdict"], "زیان‌ده")
        self.assertEqual(cell["live"]["n"], 0)
        self.assertIsNone(p["cells"]["BTCUSDT"]["4h"]["scorecard"])
        self.assertEqual((cell["warnings"], cell["scorecard_applies"]), ([], True))
        self.assertEqual(cell["live"]["verdict_code"], "small")
        meta = p["scorecard_meta"]
        for key in ("building", "built_at", "window_from", "window_to", "window_days_actual", "source",
                    "cost_pct"):
            self.assertIn(key, meta)
        self.assertIs(meta["building"], False)
        self.assertEqual(meta["cost_pct"], 0.14)
        self.assertIn("gates.json", p["note"])
        self.assertNotIn("ژوئیه", p["note"])                                  # عددِ سایهٔ ژوئیه مالِ این قاعده نیست
        self.assertIn("15m", p["note"])
        json.dumps(p, ensure_ascii=False)                                     # قابلِ سریال‌سازی


if __name__ == "__main__":
    unittest.main()
