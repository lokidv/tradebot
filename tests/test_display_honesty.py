# -*- coding: utf-8 -*-
"""آنچه کاربر می‌بیند باید همان باشد که عدد واقعاً هست (ممیزیِ نمایش).

* p_win (احتمالِ بردِ براکتِ 1R/1.8R، پایه ~۳۶٪) هرگز «احتمالِ رشد» p_up نمی‌شود؛ p_calibrated فقط با
  مدلِ جهت‌یابِ معتبر؛ p_up_kind می‌گوید عدد احتمالِ مدل است یا امتیازِ اکتشافی.
* مشاورِ پوزیشن فقط با مدلِ کالیبره تیلت می‌گیرد؛ بی‌مدل «فقط هندسه» است و قاعدهٔ EV فعال نمی‌شود.
* بک‌تستِ کوچکِ همین نماد بازهٔ ویلسون و پرچمِ «نمونهٔ کم» دارد.
* آمادگیِ «صبر» گرد به پایین است، شرطِ رأیِ مخالف را هم می‌شمارد و هرگز اقدام را عوض نمی‌کند.
* رأیِ kNN «الگوی مشابه» نام دارد، نه «هوش مصنوعی» — رأی و شمارش دست‌نخورده.
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import advisor  # noqa: E402
import bracket  # noqa: E402
import decision  # noqa: E402
import engine  # noqa: E402
import gates  # noqa: E402

from test_bracket_contract import synthetic_klines  # noqa: E402

BASE_PWIN = round(bracket.breakeven_win_rate() * 100, 1)      # ۳۵٫۷


class _GatesTmp(unittest.TestCase):
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


def _analyze(sig, seed=5, **kw):
    """analyze با ستاپِ اجباری در هر کندل (sig=+1 لانگ، −1 شورت، ۰ بی‌ستاپ) و p_upِ خامِ forecast."""
    seen = {}
    real_fc = engine.forecast

    def spy(*a, **k):
        fc = real_fc(*a, **k)
        seen["p_up"] = fc["p_up"]
        return fc

    every_bar = lambda cs, o, h, l, c, i: (sig, "pb" if sig else None)  # noqa: E731
    with mock.patch.object(engine, "forecast", side_effect=spy), \
            mock.patch.object(engine, "setup_signal", side_effect=every_bar):
        res = engine.analyze(synthetic_klines(seed=seed), "1h", **kw)
    return res, seen["p_up"]


def _predict(source, p_win, trusted=False):
    def fn(_tf, _feats, _risk, **_kw):
        return {"source": source, "p_win": p_win, "policy_trusted": trusted, "policy_pass": False,
                "n": 300, "avg_r": 0.05, "reliability": "خوب", "oos_lift": 6.0}
    return fn


# ───────────────────────── display-1 / calib-F5: p_win ≠ p_up ─────────────────────────
class PWinIsNotPUpTests(_GatesTmp):
    def test_setup_model_p_win_is_never_shown_as_probability_of_rise(self):
        for sig, side in ((1, "long"), (-1, "short")):
            for p_win in (36.0, 40.0, 55.0):
                with self.subTest(side=side, p_win=p_win):
                    res, fc_p = _analyze(sig, predict_fn=_predict("model", p_win))
                    self.assertEqual(res["trade"]["side"], side)
                    self.assertEqual(res["trade"]["p_win"], p_win)          # p_win جدا می‌ماند
                    # قبلاً p_up = p_win (لانگ) یا ۱۰۰−p_win (شورت) با «کالیبره ✓» بود
                    self.assertAlmostEqual(res["p_up"], round(fc_p, 1), places=6)
                    self.assertFalse(res["p_calibrated"])
                    self.assertEqual(res["p_up_kind"], "heuristic")
                    self.assertEqual(res["trade"]["p_win_base"], BASE_PWIN)
                    self.assertTrue(res["trade"]["p_win_calibrated"])

    def test_trusted_policy_p_win_is_not_mapped_either(self):
        res, fc_p = _analyze(1, predict_fn=_predict("policy", 40.0, trusted=True))
        self.assertAlmostEqual(res["p_up"], round(fc_p, 1), places=6)
        self.assertFalse(res["p_calibrated"])
        self.assertTrue(res["trade"]["p_win_calibrated"])

    def test_untrusted_p_win_is_flagged_as_such(self):
        res, _ = _analyze(1, predict_fn=_predict("policy", 40.0, trusted=False))
        self.assertFalse(res["trade"]["p_win_calibrated"])
        self.assertEqual(res["trade"]["p_win_base"], BASE_PWIN)

    def test_action_policy_heuristic_p_up_is_not_marked_calibrated(self):
        def action_fn(_tf, _lf, _sf, _risk, **_kw):
            return {"side": "short", "policy_pass": True, "policy_trusted": True,
                    "source": "action_policy", "p_win": None}
        res, fc_p = _analyze(0, action_fn=action_fn)
        self.assertEqual(res["trade"]["side"], "short")
        self.assertEqual(res["trade"]["setup"], "ap")
        self.assertAlmostEqual(res["p_up"], round(fc_p, 1), places=6)
        self.assertFalse(res["p_calibrated"])                     # قبلاً True با عددِ اکتشافی
        self.assertEqual(res["p_up_kind"], "heuristic")
        self.assertNotIn("p_win_base", res["trade"])               # p_winی در کار نیست

    def test_only_a_trusted_direction_model_makes_p_up_calibrated(self):
        dir_fn = lambda _tf, _f: {"p_up": 63.0, "oos_lift": 5.0}  # noqa: E731
        res, _ = _analyze(1, dir_fn=dir_fn, predict_fn=_predict("model", 40.0))
        self.assertEqual(res["p_up"], 63.0)
        self.assertTrue(res["p_calibrated"])
        self.assertEqual(res["p_up_kind"], "model")
        self.assertEqual(res["trade"]["p_win"], 40.0)

    def test_no_model_at_all_is_heuristic(self):
        res = engine.analyze(synthetic_klines(seed=7), "4h")
        self.assertEqual(res["p_up_kind"], "heuristic")
        self.assertFalse(res["p_calibrated"])


# ───────────────────────── display-2 / display-7: مشاور ─────────────────────────
def _pos(price, side="long", entry=100.0, sl=98.0, tp=103.6, **kw):
    p = {"id": "p1", "symbol": "ETHUSDT", "tf": "1h", "side": side, "entry": entry, "sl": sl, "tp": tp,
         "last_price": price, "opened_at": datetime.now(timezone.utc).isoformat(),
         "time_stop_min": 2400, "pnl_usdt": 0.0, "size_usdt": 100.0, "mode": "paper"}
    p.update(kw)
    return p


class AdvisorTiltTests(unittest.TestCase):
    def test_heuristic_or_unlabelled_p_up_gives_geometry_only(self):
        base = round(2.0 / 5.6 * 100, 1)                      # d_sl/(d_sl+d_tp) در قیمتِ ورود
        for a in (None, {"p_up": 30}, {"p_up": 30, "p_calibrated": False},
                  {"p_up": 30, "p_up_kind": "heuristic"}):
            with self.subTest(a=a):
                adv = advisor.advise(_pos(100.0), a)
                self.assertEqual(adv["basis"], "geometry")
                self.assertEqual(adv["p_hit_tp"], base)
                self.assertEqual(adv["ev_hold_r"], 0.0)
                self.assertNotEqual(adv["action"], "close_now")     # p_up=۳۰ دیگر «ببند» نمی‌سازد

    def test_calibrated_model_keeps_the_previous_tilt_exactly(self):
        # همان اعدادِ ممیزی: لانگ ۱۰۰/۹۸/۱۰۳٫۶ در قیمتِ ورود
        m = lambda p: {"p_up": p, "p_calibrated": True}  # noqa: E731
        adv = advisor.advise(_pos(100.0), m(60))
        self.assertEqual((adv["basis"], adv["p_hit_tp"], adv["ev_hold_r"]), ("model", 41.4, 0.16))
        adv = advisor.advise(_pos(100.0), m(30))
        self.assertEqual((adv["action"], adv["urgency"], adv["p_hit_tp"]), ("close_now", 3, 25.6))
        adv = advisor.advise(_pos(100.0), {"p_up": 50, "p_up_kind": "model"})
        self.assertEqual((adv["basis"], adv["ev_hold_r"]), ("model", 0.0))

    def test_geometry_mode_does_not_fire_ev_artifact_rules(self):
        # در ۰٫۸R سود: با مدلِ کالیبرهٔ خنثی قفلِ سود (EV<۰٫۱۵)؛ بی‌مدل EV همیشه صفر است و قفل مصنوع می‌شد
        self.assertEqual(advisor.advise(_pos(101.6), {"p_up": 50, "p_calibrated": True})["action"],
                         "lock_profit")
        adv = advisor.advise(_pos(101.6), None)
        self.assertEqual((adv["action"], adv["urgency"], adv["basis"]), ("hold", 0, "geometry"))
        self.assertIn("فقط هندسه", adv["reasons"][0])
        # قاعده‌های ساختاری بی‌مدل هم می‌مانند
        self.assertEqual(advisor.advise(_pos(102.2), None)["action"], "move_be")
        self.assertEqual(advisor.advise(_pos(104.0), None)["basis"], "rule")

    def test_positions_endpoint_passes_p_up_only_when_calibrated(self):
        import main
        pos = _pos(100.0)
        seen = []
        real = advisor.advise

        def spy(p, a, btc_z=None):
            seen.append(a)
            return real(p, a, btc_z=btc_z)

        for cal, want in ((False, "geometry"), (True, "model")):
            seen.clear()
            ana = {"ETHUSDT": {"p_up": 30.0, "p_calibrated": cal, "z": 0.1},
                   "BTCUSDT": {"p_up": 50.0, "p_calibrated": cal, "z": 0.2}}
            with self.subTest(p_calibrated=cal), \
                    mock.patch.object(main.broker, "load_cfg", return_value={}), \
                    mock.patch.object(main.broker, "make_broker", return_value=None), \
                    mock.patch.object(main.paper, "list_positions", return_value={"open": [], "closed": []}), \
                    mock.patch.object(main.paper, "refresh", return_value={"open": [dict(pos)], "closed": []}), \
                    mock.patch.object(main.paper, "wallet_summary", return_value={}), \
                    mock.patch.object(main, "_market_state", return_value={}), \
                    mock.patch.object(main, "get_analysis", side_effect=lambda s, tf: ana[s]), \
                    mock.patch.object(main.advisor, "advise", side_effect=spy):
                out = main.positions()
                adv = out["open"][0]["advice"]
                self.assertEqual(adv["basis"], want)
                if cal:
                    self.assertEqual(seen[0], {"p_up": 30.0, "p_calibrated": True})
                    self.assertEqual(adv["action"], "close_now")
                else:
                    self.assertIsNone(seen[0])
                    self.assertNotEqual(adv["action"], "close_now")


# ───────────────────────── display-4: بک‌تستِ کوچک ─────────────────────────
class QuickBacktestSampleTests(unittest.TestCase):
    def test_wilson_interval(self):
        self.assertEqual(engine.wilson_ci(0, 0), (None, None))
        self.assertEqual(engine.wilson_ci(3, 6), (18.8, 81.2))
        lo, hi = engine.wilson_ci(0, 6)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 30)                               # ۰ از ۶ یعنی «نمی‌دانیم»، نه «هرگز»
        lo, hi = engine.wilson_ci(6, 6)
        self.assertEqual(hi, 100.0)
        self.assertLess(lo, 70)

    def test_quick_backtest_reports_ci_and_small_sample_flag(self):
        n = 300
        o = np.full(n, 100.0)
        h = np.full(n, 100.1)
        l = np.full(n, 99.9)
        c = np.full(n, 100.0)
        cs = {"a14": np.full(n, 1.0)}
        one = lambda _cs, _o, _h, _l, _c, i: (1, "pb") if i in (220, 240) else (0, None)  # noqa: E731
        with mock.patch.object(engine, "setup_signal", side_effect=one):
            bt = engine.quick_backtest(o, h, l, c, cs, "1h", cost_pct=0.15)
        self.assertEqual(bt["n"], 2)
        self.assertTrue(bt["small_n"])
        self.assertEqual((bt["win_rate_lo"], bt["win_rate_hi"]), engine.wilson_ci(0, 2))
        self.assertEqual(bt["scope"], "setups")
        with mock.patch.object(engine, "setup_signal", return_value=(0, None)):
            empty = engine.quick_backtest(o, h, l, c, cs, "1h")
        self.assertEqual((empty["n"], empty["win_rate_lo"], empty["small_n"]), (0, None, True))

    def test_live_analysis_carries_the_flag(self):
        bt = engine.analyze(synthetic_klines(seed=5), "1h")["backtest"]
        self.assertIn("small_n", bt)
        self.assertEqual(bt["small_n"], bt["n"] < engine.BT_MIN_N)


# ───────────────────────── display-6: آمادگی ─────────────────────────
def _wait(z, vb, vs, s_ml=0.0):
    c = np.array([100.0])
    cs = {"a14": np.array([1.0]), "z": np.array([float(z)])}
    return engine.trade_suggestion(c, cs, {"trend_w": 0.5, "target": 100.0}, "1h", vb, vs, s_ml)


class ReadinessTests(unittest.TestCase):
    def test_just_below_threshold_is_not_rounded_up_to_100(self):
        tr = _wait(1.195, 3, 0)
        self.assertIsNone(tr["side"])
        self.assertEqual(tr["readiness"], 99)                    # قبلاً round(99.8)=100

    def test_opposing_votes_count_against_readiness(self):
        tr = _wait(1.5, 3, 2)                                    # z و ۳ رأی پر، ولی ۲ رأیِ مخالف
        self.assertIsNone(tr["side"])
        self.assertEqual(tr["readiness"], 85)                    # یک رأی باید برگردد: 0.55 + 0.45×⅔
        self.assertEqual(tr["ready_side"], "long")

    def test_unchanged_when_the_opposing_condition_holds(self):
        for z, vb, vs in ((0.6, 2, 0), (0.9, 1, 1), (-0.8, 0, 2), (0.0, 0, 0)):
            with self.subTest(z=z, vb=vb, vs=vs):
                old_l = 0.55 * min(max(z, 0) / 1.2, 1) + 0.45 * min(vb / 3, 1)
                old_s = 0.55 * min(max(-z, 0) / 1.2, 1) + 0.45 * min(vs / 3, 1)
                tr = _wait(z, vb, vs)
                self.assertEqual(tr["readiness"], int(max(old_l, old_s) * 100 + 1e-9))

    def test_readiness_is_display_only_and_never_changes_the_side(self):
        rng = np.random.RandomState(0)
        for _ in range(3000):
            z = float(rng.uniform(-2.5, 2.5))
            vb = int(rng.randint(0, 5))
            vs = int(rng.randint(0, 5 - vb))
            s_ml = float(rng.choice([-0.4, 0.0, 0.4]))
            bull5 = vb + (1 if s_ml >= 0.2 else 0)
            bear5 = vs + (1 if s_ml <= -0.2 else 0)
            rule = ("long" if (z >= 1.2 and bull5 >= 3 and bear5 <= 1)
                    else "short" if (z <= -1.2 and bear5 >= 3 and bull5 <= 1) else None)
            tr = _wait(z, vb, vs, s_ml)
            self.assertEqual(tr["side"], rule)
            if rule is None:
                self.assertTrue(0 <= tr["readiness"] <= 99, tr)
                a = {"symbol": "BTCUSDT", "tf": "1h", "z": z, "votes_bull": bull5, "votes_bear": bear5,
                     "components": {}, "trade": tr}
                d = decision.from_analysis(a)
                self.assertEqual(d["action"], "wait")
                self.assertLessEqual(d["strength"], 99)


# ───────────────────────── RULE-2: برچسبِ kNN ─────────────────────────
class KnnLabelTests(unittest.TestCase):
    def test_analysis_labels_the_knn_vote_honestly(self):
        comp = engine.analyze(synthetic_klines(seed=11), "1h")["components"]
        self.assertIn(engine.KNN_LABEL, comp)
        self.assertNotIn("هوش مصنوعی", comp)
        self.assertEqual(len(comp), 5)

    def test_votes_are_identical_under_the_new_and_the_legacy_label(self):
        base = {"روند": 0.5, "مومنتوم": 0.3, "حجم": 0.1, "ساختار": -0.6}
        for v in (-0.4, -0.2, 0.0, 0.1, 0.2, 0.6):
            new = decision._component_votes(dict(base, **{engine.KNN_LABEL: v}))
            old = decision._component_votes(dict(base, **{"هوش مصنوعی": v}))
            self.assertEqual(new, old)
            ai = [r for r in new[0] if r["key"] == "ai"][0]
            self.assertEqual(ai["fa"], engine.KNN_LABEL)
            self.assertEqual(ai["vote"], 1 if v >= 0.2 else -1 if v <= -0.2 else 0)

    def test_user_texts_do_not_call_it_ai(self):
        a = {"symbol": "BTCUSDT", "tf": "1h", "zt": 0, "price": 100.0, "z": 1.8,
             "votes_bull": 4, "votes_bear": 1, "regime": "رونددار",
             "components": {"روند": 0.5, "مومنتوم": 0.3, "حجم": 0.3, "ساختار": 0.6, engine.KNN_LABEL: -0.4},
             "trade": {"side": "long", "grade": "A", "entry": 100.0, "sl": 98.7, "tp": 102.34, "atr14": 1.0,
                       "rr": 1.8, "risk_pct": 1.3, "setup": None, "setup_observed": False}}
        d = decision.from_analysis(a)
        self.assertEqual((d["votes_bull"], d["votes_bear"]), (4, 1))
        self.assertTrue(any("الگوی مشابه (kNN)" in r for r in d["reasons"]), d["reasons"])
        texts = " ".join(d["reasons"]) + decision.NOTE_FA + " ".join(c[1] for c in decision.COMPONENTS)
        self.assertNotIn("هوش مصنوعی", texts)

if __name__ == "__main__":
    unittest.main()
